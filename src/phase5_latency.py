"""Phase 5: can the model pick a better JOIN ORDER than PostgreSQL?

FIXED vs old version:
  * old: loaded ONLY results/model.pt (FileNotFoundError - the MLP is now
         saved as mlp_model.pt) and scored with expm1(raw) (wrong for
         ratio models)
  * new: --model {mlp,lgbm,rf,xgb}; scoring = pg_est * exp(raw), where
         pg_est for every sub-plan is obtained with EXPLAIN; adds the
         'ML ranking correct' check (like the q3 script).
Usage:
  python src/phase5_latency.py --model xgb
"""
import argparse
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2

from config import DB_CONFIG
from evaluate_tpch import rng_pred
from gen_training_data import JOIN_EDGES, FEAT_DIM, build_body, featurize, load_col_stats
from model_loader import CardModel

BASE = Path(__file__).resolve().parent.parent

TABLES_5 = ["customer", "orders", "lineitem", "supplier", "nation"]

CANDIDATES = [
    ("dim-first  (n-c-o-l-s)", ["nation", "customer", "orders", "lineitem", "supplier"]),
    ("cust-first (c-o-l-s-n)", ["customer", "orders", "lineitem", "supplier", "nation"]),
    ("fact-early (o-l-s-c-n)", ["orders", "lineitem", "supplier", "customer", "nation"]),
    ("supp-first (s-l-o-c-n)", ["supplier", "lineitem", "orders", "customer", "nation"]),
]


def join_clause(t, prefix):
    for a, ac, b, bc in JOIN_EDGES:
        if a == t and b in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
        if b == t and a in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
    raise ValueError(f"no join edge available for {t}")


def build_ordered_sql(seq, pred_sqls):
    sql = f"FROM {seq[0]} " + " ".join(
        join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1)
    )
    if pred_sqls:
        sql += " WHERE " + " AND ".join(pred_sqls)
    return "SELECT * " + sql


def pg_est_of(cur, body):
    """PostgreSQL's row estimate for an arbitrary FROM/WHERE body."""
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def ml_cost(model, cur, seq, preds):
    """Score a join order: sum of predicted intermediate cardinalities.

    Ratio model: pred(sub-plan) = pg_est(sub-plan) * exp(model output).
    """
    total = 0.0
    for i in range(1, len(seq)):
        sub = seq[:i + 1]
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        body = build_body(sub, joins, [p[4] for p in sub_preds])
        x = [featurize(sub, joins, sub_preds)]
        total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
    return total


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

    preds = [rng_pred(stats, "orders", "o_orderdate",
                      date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal())]
    pred_sqls = [p[4] for p in preds]

    model = CardModel(args.model)

    cur.execute("SET join_collapse_limit = 1")  # preserve our written join order
    print(f"model: {model}\n")
    print(f"{'join order':<28}{'ML cost (rows)':>18}{'exec time':>14}")
    results = []
    for name, seq in CANDIDATES:
        cost = ml_cost(model, cur, seq, preds)
        t = exec_time_ms(cur, build_ordered_sql(seq, pred_sqls))
        results.append((name, cost, t))
        print(f"{name:<28}{cost:>18.0f}{t:>12.1f} ms")

    cur.execute("SET join_collapse_limit = 8")  # PostgreSQL default
    joins = [e for e in JOIN_EDGES if e[0] in TABLES_5 and e[2] in TABLES_5]
    default_sql = "SELECT * " + build_body(TABLES_5, joins, pred_sqls)
    t_default = exec_time_ms(cur, default_sql)
    print(f"{'PG default planner':<28}{'-':>18}{t_default:>12.1f} ms")

    best = min(results, key=lambda r: r[1])
    fastest = min(results, key=lambda r: r[2])
    print(f"\nML chose               : {best[0].strip()}  ({best[2]:.1f} ms)")
    print(f"Actual fastest candidate: {fastest[0].strip()}  ({fastest[2]:.1f} ms)")
    print(f"PG default              : {t_default:.1f} ms")
    print(f"ML ranking correct      : {'YES' if best[0] == fastest[0] else 'NO'}")
    print(f"Difference vs PG default: {t_default - best[2]:+.1f} ms "
          f"({'ML order faster' if best[2] < t_default else 'PG plan faster'})")
    conn.close()


if __name__ == "__main__":
    main()
