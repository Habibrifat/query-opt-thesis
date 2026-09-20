"""Phase 5 (TPC-H Q3): can the model pick a better JOIN ORDER than PG?

Same fixes as phase5_latency.py: --model {mlp,lgbm,rf,xgb} and
ratio-corrected scoring (pg_est * exp(raw)) with EXPLAIN per sub-plan.
Usage:
  python src/phase5_latency_q3.py --model xgb
"""
import argparse
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2

from config import DB_CONFIG
from evaluate_tpch import eq_cat, rng_pred
from gen_training_data import JOIN_EDGES, FEAT_DIM, build_body, featurize, load_col_stats
from model_loader import CardModel

BASE = Path(__file__).resolve().parent.parent

TABLES_3 = ["customer", "orders", "lineitem"]

CANDIDATES = [
    {"name": "left-deep  (c-o-l)", "seq": ["customer", "orders", "lineitem"],
     "steps": [["customer", "orders"], TABLES_3]},
    {"name": "left-deep  (o-l-c)", "seq": ["orders", "lineitem", "customer"],
     "steps": [["orders", "lineitem"], TABLES_3]},
    {"name": "right-deep (c-(o-l))", "seq": None,
     "sql": ("FROM customer JOIN (orders JOIN lineitem ON l_orderkey = o_orderkey) "
             "ON c_custkey = o_custkey"),
     "steps": [["orders", "lineitem"], TABLES_3]},
]


def join_clause(t, prefix):
    for a, ac, b, bc in JOIN_EDGES:
        if a == t and b in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
        if b == t and a in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
    raise ValueError(f"no join edge available for {t}")


def pg_est_of(cur, body):
    """PostgreSQL's row estimate for an arbitrary FROM/WHERE body."""
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def ml_cost(model, cur, steps, preds):
    """Score a join order: sum of predicted intermediate cardinalities."""
    total = 0.0
    for sub in steps:
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        body = build_body(sub, joins, [p[4] for p in sub_preds])
        x = [featurize(sub, joins, sub_preds)]
        total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
    return total


def build_sql(cand, pred_sqls):
    if cand["seq"] is not None:
        seq = cand["seq"]
        sql = f"SELECT * FROM {seq[0]} " + " ".join(
            join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
    else:
        sql = "SELECT * " + cand["sql"]
    if pred_sqls:
        sql += " WHERE " + " AND ".join(pred_sqls)
    return sql


def exec_time_ms(cur, sql, reps=3):
    times = []
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        times.append(cur.fetchone()[0][0]["Execution Time"])
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "lgbm", "rf", "xgb"])
    args = ap.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    preds = [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
    ]
    pred_sqls = [p[4] for p in preds]

    model = CardModel(args.model)

    cur.execute("SET join_collapse_limit = 1")
    print(f"model: {model}\n")
    print(f"{'join order':<24}{'ML cost (rows)':>18}{'exec time':>12}")
    results = []
    for cand in CANDIDATES:
        cost = ml_cost(model, cur, cand["steps"], preds)
        t = exec_time_ms(cur, build_sql(cand, pred_sqls))
        results.append((cand["name"], cost, t))
        print(f"{cand['name']:<24}{cost:>18.0f}{t:>10.1f} ms")

    cur.execute("SET join_collapse_limit = 8")
    joins = [e for e in JOIN_EDGES if e[0] in TABLES_3 and e[2] in TABLES_3]
    t_default = exec_time_ms(cur, "SELECT * " + build_body(TABLES_3, joins, pred_sqls))
    print(f"{'PG default planner':<24}{'-':>18}{t_default:>10.1f} ms")

    best = min(results, key=lambda r: r[1])
    fastest = min(results, key=lambda r: r[2])
    print(f"\nML chose                : {best[0].strip()} ({best[2]:.1f} ms)")
    print(f"Actual fastest candidate: {fastest[0].strip()} ({fastest[2]:.1f} ms)")
    print(f"PG default              : {t_default:.1f} ms")
    print(f"ML ranking correct      : {'YES' if best[0] == fastest[0] else 'NO'}")
    print(f"Difference vs PG default: {t_default - best[2]:+.1f} ms "
          f"({'ML order faster' if best[2] < t_default else 'PG plan faster'})")
    conn.close()


if __name__ == "__main__":
    main()
