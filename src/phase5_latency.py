import statistics
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.nn as nn

from config import DB_CONFIG
from evaluate_tpch import rng_pred
from gen_training_data import JOIN_EDGES, FEAT_DIM, build_body, featurize, load_col_stats

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "results" / "model.pt"

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


def ml_cost(model, seq, preds):
    """Score a join order: sum of predicted intermediate cardinalities."""
    total = 0.0
    for i in range(1, len(seq)):
        sub = seq[:i + 1]
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        x = torch.tensor([featurize(sub, joins, sub_preds)], dtype=torch.float32)
        with torch.no_grad():
            total += float(np.expm1(model(x).item()))
    return total


def exec_time_ms(cur, sql, reps=3):
    times = []
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        times.append(cur.fetchone()[0][0]["Execution Time"])
    return statistics.median(times)


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    preds = [rng_pred(stats, "orders", "o_orderdate",
                      date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal())]
    pred_sqls = [p[4] for p in preds]

    model = nn.Sequential(
        nn.Linear(FEAT_DIM, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    model.load_state_dict(torch.load(MODEL, weights_only=True))
    model.eval()

    cur.execute("SET join_collapse_limit = 1")  # preserve our written join order
    print(f"{'join order':<28}{'ML cost (rows)':>18}{'exec time':>14}")
    results = []
    for name, seq in CANDIDATES:
        cost = ml_cost(model, seq, preds)
        t = exec_time_ms(cur, build_ordered_sql(seq, pred_sqls))
        results.append((name, cost, t))
        print(f"{name:<28}{cost:>18.0f}{t:>12.1f} ms")

    cur.execute("SET join_collapse_limit = 8")  # PostgreSQL default
    joins = [e for e in JOIN_EDGES if e[0] in TABLES_5 and e[2] in TABLES_5]
    default_sql = "SELECT * " + build_body(TABLES_5, joins, pred_sqls)
    t_default = exec_time_ms(cur, default_sql)
    print(f"{'PG default planner':<28}{'-':>18}{t_default:>12.1f} ms")

    best = min(results, key=lambda r: r[1])
    print(f"\nML chose      : {best[0].strip()}  ({best[2]:.1f} ms)")
    print(f"PG default    : {t_default:.1f} ms")
    print(f"Difference    : {t_default - best[2]:+.1f} ms "
          f"({'ML order faster' if best[2] < t_default else 'PG plan faster'})")
    conn.close()


if __name__ == "__main__":
    main()