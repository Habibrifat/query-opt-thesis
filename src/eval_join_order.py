"""Join-order experiment: does a model's predicted cardinality, used to
pick a join order, produce a plan that's actually faster than PostgreSQL's
default planner? Also reports cost-model accuracy: does PostgreSQL's own
Total Cost estimate track real execution time on these candidates?

Replaces phase5_latency.py + phase5_latency_q3.py (previously duplicated
join_clause/ml_cost/exec_time_ms with slightly different candidate sets).

Usage:
  python src/eval_join_order.py --scenario 5table --model rf
  python src/eval_join_order.py --scenario q3     --model xgb
"""
import argparse
import statistics
from datetime import date

import numpy as np
import psycopg2
from scipy.stats import spearmanr

from config import DB_CONFIG
from evaluate_tpch import eq_cat, rng_pred
from gen_training_data import JOIN_EDGES, build_body, featurize, load_col_stats
from model_loader import CardModel

# ---------------------------------------------------------------------------
# Scenario definitions. Each scenario = (default_tables, preds_fn, candidates)
# candidates: list of dicts with "name", "steps" (sub-joins for ml_cost),
# and a "sql" callable that builds the full ordered query.
# ---------------------------------------------------------------------------

TABLES_5 = ["customer", "orders", "lineitem", "supplier", "nation"]
TABLES_3 = ["customer", "orders", "lineitem"]


def join_clause(t, prefix):
    for a, ac, b, bc in JOIN_EDGES:
        if a == t and b in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
        if b == t and a in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
    raise ValueError(f"no join edge available for {t}")


def build_ordered_sql(seq, pred_sqls):
    sql = f"FROM {seq[0]} " + " ".join(
        join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
    if pred_sqls:
        sql += " WHERE " + " AND ".join(pred_sqls)
    return "SELECT * " + sql


def preds_5table(stats):
    return [rng_pred(stats, "orders", "o_orderdate",
                      date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal())]


def candidates_5table():
    seqs = {
        "dim-first  (n-c-o-l-s)": ["nation", "customer", "orders", "lineitem", "supplier"],
        "cust-first (c-o-l-s-n)": ["customer", "orders", "lineitem", "supplier", "nation"],
        "fact-early (o-l-s-c-n)": ["orders", "lineitem", "supplier", "customer", "nation"],
        "supp-first (s-l-o-c-n)": ["supplier", "lineitem", "orders", "customer", "nation"],
    }
    out = []
    for name, seq in seqs.items():
        steps = [seq[:i + 1] for i in range(1, len(seq))]
        out.append({"name": name, "steps": steps,
                    "sql": lambda pred_sqls, seq=seq: build_ordered_sql(seq, pred_sqls)})
    return out


def preds_q3(stats):
    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
    ]


def candidates_q3():
    def right_deep_sql(pred_sqls):
        sql = ("SELECT * FROM customer JOIN "
               "(orders JOIN lineitem ON l_orderkey = o_orderkey) "
               "ON c_custkey = o_custkey")
        if pred_sqls:
            sql += " WHERE " + " AND ".join(pred_sqls)
        return sql

    return [
        {"name": "left-deep  (c-o-l)",
         "steps": [["customer", "orders"], TABLES_3],
         "sql": lambda ps: build_ordered_sql(["customer", "orders", "lineitem"], ps)},
        {"name": "left-deep  (o-l-c)",
         "steps": [["orders", "lineitem"], TABLES_3],
         "sql": lambda ps: build_ordered_sql(["orders", "lineitem", "customer"], ps)},
        {"name": "right-deep (c-(o-l))",
         "steps": [["orders", "lineitem"], TABLES_3],
         "sql": right_deep_sql},
    ]


SCENARIOS = {
    "5table": dict(default_tables=TABLES_5, preds_fn=preds_5table, candidates_fn=candidates_5table),
    "q3": dict(default_tables=TABLES_3, preds_fn=preds_q3, candidates_fn=candidates_q3),
}

# ---------------------------------------------------------------------------


def pg_est_of(cur, body):
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def ml_cost(model, cur, steps, preds):
    total = 0.0
    for sub in steps:
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        body = build_body(sub, joins, [p[4] for p in sub_preds])
        x = [featurize(sub, joins, sub_preds)]
        total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
    return total


def exec_stats(cur, sql, reps=3):
    """Run EXPLAIN ANALYZE `reps` times -> (median exec ms, PG Total Cost)."""
    times, cost = [], None
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]
        times.append(plan["Execution Time"])
        cost = plan["Plan"]["Total Cost"]
    return statistics.median(times), cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "lgbm", "rf", "xgb"])
    ap.add_argument("--scenario", default="5table", choices=list(SCENARIOS))
    args = ap.parse_args()
    scen = SCENARIOS[args.scenario]

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    preds = scen["preds_fn"](stats)
    pred_sqls = [p[4] for p in preds]
    candidates = scen["candidates_fn"]()
    model = CardModel(args.model)

    print(f"scenario: {args.scenario}   model: {model}\n")
    cur.execute("SET join_collapse_limit = 1")  # preserve written join order
    print(f"{'join order':<26}{'ML cost (rows)':>16}{'PG cost':>12}{'exec time':>12}")
    results = []
    for c in candidates:
        cost_ml = ml_cost(model, cur, c["steps"], preds)
        t, cost_pg = exec_stats(cur, c["sql"](pred_sqls))
        results.append((c["name"], cost_ml, cost_pg, t))
        print(f"{c['name']:<26}{cost_ml:>16.0f}{cost_pg:>12.1f}{t:>10.1f} ms")

    cur.execute("SET join_collapse_limit = 8")  # PostgreSQL default
    joins = [e for e in JOIN_EDGES if e[0] in scen["default_tables"] and e[2] in scen["default_tables"]]
    t_default, cost_default = exec_stats(cur, "SELECT * " + build_body(scen["default_tables"], joins, pred_sqls))
    print(f"{'PG default planner':<26}{'-':>16}{cost_default:>12.1f}{t_default:>10.1f} ms")

    best_ml = min(results, key=lambda r: r[1])
    best_pg = min(results, key=lambda r: r[2])
    fastest = min(results, key=lambda r: r[3])
    print(f"\nML chose (predicted rows) : {best_ml[0].strip():<24} ({best_ml[3]:.1f} ms)")
    print(f"PG cost chose (Total Cost): {best_pg[0].strip():<24} ({best_pg[3]:.1f} ms)")
    print(f"Actual fastest             : {fastest[0].strip():<24} ({fastest[3]:.1f} ms)")
    print(f"PG default planner         : {t_default:.1f} ms")
    print(f"ML ranking correct         : {'YES' if best_ml[0] == fastest[0] else 'NO'}")

    if len(results) >= 3:
        times_ = [r[3] for r in results]
        rho_pg, _ = spearmanr([r[2] for r in results], times_)
        rho_ml, _ = spearmanr([r[1] for r in results], times_)
        print(f"\nCost-model accuracy (rank correlation with real exec time, {len(results)} candidates):")
        print(f"  PostgreSQL Total Cost vs actual time: rho={rho_pg:+.2f}")
        print(f"  ML predicted rows    vs actual time: rho={rho_ml:+.2f}")
        print("  (rho near +1.0 = ranks candidates correctly; near 0/negative = unreliable)")

    conn.close()


if __name__ == "__main__":
    main()
