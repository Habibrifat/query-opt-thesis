"""Single-query, plain-language demo of the whole evaluation pipeline.

WHY THIS FILE EXISTS
---------------------
The supervisor's feedback was: stop showing 5-table stress tests and raw
f0..f80 feature dumps. Walk through ONE real, recognizable query, in plain
language, from SQL -> features -> model guess -> PostgreSQL's own guess ->
the real answer -> which guess was closer and why that matters.

This does NOT retrain anything and does NOT shrink the model's input size.
Your trained models (rf/xgb/lgbm/mlp) were all trained on the same 81-number
feature vector (8 table flags + 9 join flags + 4 numbers x 16 predicate
columns) - that shape is baked into the saved model files, so a demo script
cannot feed them anything smaller without breaking them.

What CAN be simplified is the PRESENTATION: of those 81 numbers, a 3-table
query like this one only ever touches a handful of them (the rest are
always 0 - "this table/join/column is not used"). So instead of printing
all 81 numbers, this script prints only the active ones, each with a plain
English label - that's the "reduced, meaningful" view the supervisor asked
for, without changing what the model actually receives.

THE DEMO QUERY: TPC-H Query 3 (the real, official benchmark query) -
customer + orders + lineitem, filtered by market segment and two dates.
Chosen because it's a real, citable query (not an invented stress test),
and because it's your smallest realistic 3-table join.

Usage:
    python src/demo_single_query.py
"""
from datetime import date
from pathlib import Path

from config import DB_CONFIG
import psycopg2

from gen_training_data import (
    JOIN_EDGES, TABLES, PRED_COLS, FEAT_DIM,
    build_body, featurize, load_col_stats, norm,
)
from evaluate_tpch import eq_cat, rng_pred, qerror
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# THE ONE DEMO QUERY — TPC-H Query 3 (customer -> orders -> lineitem)
# ---------------------------------------------------------------------------
DEMO_TABLES = ["customer", "orders", "lineitem"]


def build_demo_query(stats):
    """The real TPC-H Query 3 predicates, in plain English below each one."""
    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    preds = [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        #   -> "only customers in the BUILDING market segment"
        rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
        #   -> "only orders placed before 1995-03-15"
        rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
        #   -> "only line items shipped after 1995-03-15"
    ]
    return preds


PLAIN_LABELS = {
    ("customer", "c_mktsegment"): "customer's market segment",
    ("orders", "o_orderdate"):    "date the order was placed",
    ("lineitem", "l_shipdate"):   "date the item was shipped",
}


# ---------------------------------------------------------------------------
# Step-by-step walkthrough printer
# ---------------------------------------------------------------------------
def describe_features(tables, joins, preds):
    """Print ONLY the active (non-zero) feature slots, in plain language —
    not all 81 numbers. This is the 'reduced, meaningful' view."""
    print("  Tables involved:")
    for t in tables:
        print(f"    - {t}")
    print("  Joins used (how the tables connect):")
    for a, ac, b, bc in joins:
        print(f"    - {a}.{ac} = {b}.{bc}")
    print("  Filters applied (this is what the model is told about the WHERE clause):")
    for (t, c), kind, v1, v2, sql_text in preds:
        label = PLAIN_LABELS.get((t, c), f"{t}.{c}")
        if kind == "eq":
            print(f"    - {label}: must equal a specific value  ({sql_text})")
        else:
            print(f"    - {label}: must fall in a range          ({sql_text})")
    total_slots = FEAT_DIM
    active_slots = len(tables) + len(joins) + 2 * len(preds)  # rough count of non-zero numbers
    print(f"\n  (Behind the scenes this becomes a vector of {total_slots} numbers for the "
          f"model to read — only {active_slots} of them are non-zero for this query; "
          f"the rest are 0, meaning 'not used'.)")


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    preds = build_demo_query(stats)
    joins = [e for e in JOIN_EDGES if e[0] in DEMO_TABLES and e[2] in DEMO_TABLES]
    body = build_body(DEMO_TABLES, joins, [p[4] for p in preds])
    sql = "SELECT * " + body

    print("=" * 78)
    print("STEP 1 — The query (this is real TPC-H Query 3's core join)")
    print("=" * 78)
    print(f"  {sql}\n")

    print("=" * 78)
    print("STEP 2 — What the model actually sees (plain-English feature view)")
    print("=" * 78)
    describe_features(DEMO_TABLES, joins, preds)

    print("\n" + "=" * 78)
    print("STEP 3 — PostgreSQL's own guess, and the real answer")
    print("=" * 78)
    cur.execute("EXPLAIN (FORMAT JSON) " + sql)
    pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
    cur.execute("SELECT COUNT(*) " + body)
    actual = cur.fetchone()[0]
    print(f"  PostgreSQL's guess (before running the query) : {pg_est:>12,} rows")
    print(f"  The real answer (after actually running it)   : {actual:>12,} rows")
    q_pg = qerror(pg_est, actual)
    print(f"  -> PostgreSQL's q-error = {q_pg:.2f}  "
          f"({'a perfect guess' if q_pg < 1.05 else 'off by a factor of ' + f'{q_pg:.1f}x'})")

    print("\n" + "=" * 78)
    print("STEP 4 — Each trained model's guess, for the SAME query")
    print("=" * 78)
    names = available_models()
    if not names:
        print("  (no trained models found in results/ — run train_*.py first)")
        conn.close()
        return

    x = [featurize(DEMO_TABLES, joins, preds)]
    rows = []
    for n in names:
        model = CardModel(n)
        pred = float(model.predict_rows(x, [pg_est])[0])
        q = qerror(pred, actual)
        rows.append((n, pred, q))

    print(f"  {'Model':<10}{'Guessed rows':>16}{'Q-error':>12}{'vs PostgreSQL':>18}")
    print("  " + "-" * 56)
    print(f"  {'PostgreSQL':<10}{pg_est:>16,}{q_pg:>12.2f}{'(baseline)':>18}")
    for n, pred, q in rows:
        verdict = "BETTER" if q < q_pg else ("SAME" if abs(q - q_pg) < 0.05 else "WORSE")
        print(f"  {n.upper():<10}{pred:>16,.0f}{q:>12.2f}{verdict:>18}")

    print("\n" + "=" * 78)
    print("STEP 5 — What this demonstrates (the evaluation metric in plain words)")
    print("=" * 78)
    print(
        "  Q-error answers one question: 'how many times off was the guess?'\n"
        "  1.0 = perfect. 2.0 = guessed double or half the real amount.\n\n"
        "  If a model's q-error is LOWER than PostgreSQL's, that model's cardinality\n"
        "  estimate is more accurate than PostgreSQL's built-in one — which is the\n"
        "  first, most basic requirement for 'query optimization is working': you\n"
        "  cannot pick a faster execution plan from a wrong guess about how much\n"
        "  data is involved.\n"
    )

    conn.close()


if __name__ == "__main__":
    main()
