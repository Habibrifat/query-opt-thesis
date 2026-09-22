"""Query-optimization evaluation.

For a scenario (5-table Q5-like, or 3-table Q3-like):
  * builds a fixed set of candidate join orders,
  * for each model, computes its predicted cost per candidate,
  * runs EXPLAIN ANALYZE to get real execution time,
  * reports:
       - plan-choice accuracy       (did the model pick the fastest order?)
       - cost-model accuracy        (Spearman rho between predicted cost and real time)
       - end-to-end latency         (real ms per candidate, per model)

Usage:
  python src/eval_query_opt.py --scenario 5table                       # all models
  python src/eval_query_opt.py --scenario q3 --models mlp lgbm xgb     # subset
"""
import argparse
import csv
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2
from scipy.stats import spearmanr

from config import DB_CONFIG
from evaluate_tpch import eq_cat, rng_pred
from gen_training_data import JOIN_EDGES, build_body, featurize, load_col_stats
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "results"

TABLES_5 = ["customer", "orders", "lineitem", "supplier", "nation"]
TABLES_3 = ["customer", "orders", "lineitem"]


# --------------------------------------------------------------------------
# SQL builders
# --------------------------------------------------------------------------
def join_clause(t, prefix):
    for a, ac, b, bc in JOIN_EDGES:
        if a == t and b in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
        if b == t and a in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
    raise ValueError(f"no join edge for {t}")


def build_ordered_sql(seq, pred_sqls):
    sql = f"FROM {seq[0]} " + " ".join(
        join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
    if pred_sqls:
        sql += " WHERE " + " AND ".join(pred_sqls)
    return "SELECT * " + sql


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
def preds_5table(stats):
    return [rng_pred(stats, "orders", "o_orderdate",
                     date(1994, 1, 1).toordinal(),
                     date(1995, 1, 1).toordinal())]


def candidates_5table():
    seqs = {
        "dim-first  (n-c-o-l-s)": ["nation", "customer", "orders", "lineitem", "supplier"],
        "cust-first (c-o-l-s-n)": ["customer", "orders", "lineitem", "supplier", "nation"],
        "fact-early (o-l-s-c-n)": ["orders", "lineitem", "supplier", "customer", "nation"],
        "supp-first (s-l-o-c-n)": ["supplier", "lineitem", "orders", "customer", "nation"],
    }
    return [
        {"name": name,
         "steps": [seq[:i + 1] for i in range(1, len(seq))],
         "sql": (lambda ps, s=seq: build_ordered_sql(s, ps))}
        for name, seq in seqs.items()
    ]


def preds_q3(stats):
    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders", "o_orderdate", omin,
                 date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_shipdate",
                 date(1995, 3, 15).toordinal(), smax),
    ]


def candidates_q3():
    def right_deep(ps):
        sql = ("SELECT * FROM customer JOIN "
               "(orders JOIN lineitem ON l_orderkey = o_orderkey) "
               "ON c_custkey = o_custkey")
        if ps:
            sql += " WHERE " + " AND ".join(ps)
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
         "sql": right_deep},
    ]


SCENARIOS = {
    "5table": dict(default_tables=TABLES_5,
                   preds_fn=preds_5table,
                   candidates_fn=candidates_5table),
    "q3":     dict(default_tables=TABLES_3,
                   preds_fn=preds_q3,
                   candidates_fn=candidates_q3),
}


# --------------------------------------------------------------------------
# Core measurements
# --------------------------------------------------------------------------
def pg_est_of(cur, body):
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def ml_cost(model, cur, steps, preds):
    total = 0.0
    for sub in steps:
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        body = build_body(sub, joins, [p[4] for p in sub_preds])
        pg = pg_est_of(cur, body)
        x = [featurize(sub, joins, sub_preds)]
        total += float(model.predict_rows(x, [pg])[0])
    return total


def exec_stats(cur, sql, reps=3):
    times, cost = [], None
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]
        times.append(plan["Execution Time"])
        cost = plan["Plan"]["Total Cost"]
    return statistics.median(times), cost


# --------------------------------------------------------------------------
# Per-model evaluation
# --------------------------------------------------------------------------
def evaluate_one_model(cur, model, scen, preds, pred_sqls, candidates):
    """Return a list of rows: (model, candidate, ml_cost, pg_cost, exec_ms)."""
    rows = []
    for c in candidates:
        cost_ml = ml_cost(model, cur, c["steps"], preds)
        t, cost_pg = exec_stats(cur, c["sql"](pred_sqls))
        rows.append((model.name, c["name"], cost_ml, cost_pg, t))
    return rows


def summarize(rows, model_name):
    """Rank correlation + plan-choice accuracy for one model."""
    times = np.array([r[4] for r in rows], dtype=float)
    ml    = np.array([r[2] for r in rows], dtype=float)
    pg    = np.array([r[3] for r in rows], dtype=float)

    best_ml = rows[int(np.argmin(ml))]
    best_pg = rows[int(np.argmin(pg))]
    fastest = rows[int(np.argmin(times))]

    rho_ml, _ = spearmanr(ml, times)
    rho_pg, _ = spearmanr(pg, times)

    return {
        "model":            model_name,
        "ml_choice":        best_ml[1],
        "pg_choice":        best_pg[1],
        "fastest":          fastest[1],
        "ml_correct":       best_ml[1] == fastest[1],
        "pg_correct":       best_pg[1] == fastest[1],
        "rho_ml":           float(rho_ml),
        "rho_pg":           float(rho_pg),
        "ml_time_ms":       best_ml[4],
        "pg_time_ms":       best_pg[4],
        "fastest_time_ms":  fastest[4],
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="5table", choices=list(SCENARIOS))
    ap.add_argument("--models", nargs="*", default=None,
                    help="mlp lgbm rf xgb (default: all available)")
    args = ap.parse_args()
    scen = SCENARIOS[args.scenario]

    names = args.models or available_models()
    if not names:
        raise SystemExit("No trained models found. Run train_*.py first.")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    preds = scen["preds_fn"](stats)
    pred_sqls = [p[4] for p in preds]
    candidates = scen["candidates_fn"]()

    print(f"Scenario: {args.scenario}")
    print(f"Models  : {', '.join(names)}\n")

    all_rows = []
    summaries = []
    for name in names:
        model = CardModel(name)
        print(f"===== {name.upper()} =====")
        print(f"{'join order':<26}{'ML cost':>14}{'PG cost':>14}{'exec ms':>14}")
        rows = evaluate_one_model(cur, model, scen, preds, pred_sqls, candidates)
        for m, cand, cml, cpg, t in rows:
            print(f"{cand:<26}{cml:>14.0f}{cpg:>14.1f}{t:>14.1f}")
        s = summarize(rows, name)
        summaries.append(s)
        print(f"  ML chose: {s['ml_choice'].strip():<24} "
              f"({s['ml_time_ms']:.1f} ms)")
        print(f"  PG chose: {s['pg_choice'].strip():<24} "
              f"({s['pg_time_ms']:.1f} ms)")
        print(f"  Fastest : {s['fastest'].strip():<24} "
              f"({s['fastest_time_ms']:.1f} ms)")
        print(f"  ML correct: {'YES' if s['ml_correct'] else 'NO'}    "
              f"PG correct: {'YES' if s['pg_correct'] else 'NO'}")
        print(f"  rho (ML vs time) = {s['rho_ml']:+.2f}   "
              f"rho (PG vs time) = {s['rho_pg']:+.2f}")
        print()
        all_rows.extend(rows)

    # PostgreSQL default planner on the same scenario
    cur.execute("SET join_collapse_limit = 8")
    joins = [e for e in JOIN_EDGES
             if e[0] in scen["default_tables"] and e[2] in scen["default_tables"]]
    t_default, cost_default = exec_stats(
        cur, "SELECT * " + build_body(scen["default_tables"], joins, pred_sqls))
    print(f"PostgreSQL default planner: {t_default:.1f} ms  "
          f"(Total Cost {cost_default:.1f})")

    conn.close()

    # ---- save CSVs ----
    OUT_DIR.mkdir(exist_ok=True)
    rows_csv = OUT_DIR / f"eval_query_opt_{args.scenario}_rows.csv"
    with open(rows_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "candidate", "ml_cost", "pg_cost", "exec_ms"])
        w.writerows(all_rows)

    sum_csv = OUT_DIR / f"eval_query_opt_{args.scenario}_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    print(f"\nSaved per-candidate rows -> {rows_csv}")
    print(f"Saved per-model summary  -> {sum_csv}")


if __name__ == "__main__":
    main()