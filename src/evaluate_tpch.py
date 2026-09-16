import csv
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.nn as nn

from config import DB_CONFIG
from gen_training_data import (
    JOIN_EDGES, FEAT_DIM, build_body, featurize, load_col_stats, norm,
)

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "results" / "model.pt"
OUT = BASE / "results" / "tpch_eval.csv"


# ---------- predicate helpers ----------
def rng_pred(s, t, c, lo, hi):
    st = s[(t, c)]
    n1, n2 = norm(lo, st["min"], st["max"]), norm(hi, st["min"], st["max"])
    if st["kind"] == "date":
        sql = (f"{c} BETWEEN DATE '{date.fromordinal(int(lo)).isoformat()}'"
               f" AND DATE '{date.fromordinal(int(hi)).isoformat()}'")
    else:
        sql = f"{c} BETWEEN {lo:.2f} AND {hi:.2f}"
    return ((t, c), "range", n1, n2, sql)


def eq_num(s, t, c, v):
    st = s[(t, c)]
    return ((t, c), "eq", norm(v, st["min"], st["max"]), 0.0, f"{c} = {v:.2f}")


def eq_cat(s, t, c, v):
    # FIXED: strip() handles CHAR(n) padding, e.g. 'BUILDING  ' -> 'BUILDING'
    dom = [str(d).strip() for d in s[(t, c)]["domain"]]
    v = str(v).strip()
    return ((t, c), "eq", (dom.index(v) + 1) / (len(dom) + 1), 0.0, f"{c} = '{v}'")


def qerror(p, a):
    p, a = max(p, 1), max(a, 1)
    return max(p / a, a / p)


# ---------- the TPC-H queries, in our feature language ----------
def build_specs(st):
    smin = st[("lineitem", "l_shipdate")]["min"]
    smax = st[("lineitem", "l_shipdate")]["max"]
    omin = st[("orders", "o_orderdate")]["min"]

    return {
        "q1": dict(
            tables=["lineitem"], note="full query",
            preds=[rng_pred(st, "lineitem", "l_shipdate", smin, date(1998, 9, 2).toordinal())],
        ),
        "q3": dict(
            tables=["customer", "orders", "lineitem"], note="full query",
            preds=[
                eq_cat(st, "customer", "c_mktsegment", "BUILDING"),
                rng_pred(st, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
                rng_pred(st, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
            ],
        ),
        "q6": dict(
            tables=["lineitem"], note="full query",
            preds=[
                rng_pred(st, "lineitem", "l_shipdate", date(1994, 1, 1).toordinal(),
                         date(1995, 1, 1).toordinal()),
                rng_pred(st, "lineitem", "l_discount", 0.02, 0.06),
                rng_pred(st, "lineitem", "l_quantity", st[("lineitem", "l_quantity")]["min"], 24.0),
            ],
        ),
        "q5-core": dict(
            tables=["customer", "orders", "lineitem", "supplier", "nation"],
            note="core w/o r_name='ASIA' (not encodable yet)",
            preds=[rng_pred(st, "orders", "o_orderdate", date(1994, 1, 1).toordinal(),
                            date(1995, 1, 1).toordinal())],
        ),
        "q10-core": dict(
            tables=["customer", "orders", "lineitem", "nation"],
            note="core w/o n_name='GERMANY' (not encodable yet)",
            preds=[
                eq_cat(st, "lineitem", "l_returnflag", "R"),
                rng_pred(st, "orders", "o_orderdate", date(1994, 8, 1).toordinal(),
                         date(1994, 11, 1).toordinal()),
            ],
        ),
        "q14-core": dict(
            tables=["lineitem", "part"],
            note="core w/o p_brand LIKE 'PROMO%' (not encodable yet)",
            preds=[rng_pred(st, "lineitem", "l_shipdate", date(1995, 1, 1).toordinal(),
                            date(1996, 1, 1).toordinal())],
        ),
    }


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)
    specs = build_specs(stats)

    model = nn.Sequential(
        nn.Linear(FEAT_DIM, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    model.load_state_dict(torch.load(MODEL, weights_only=True))
    model.eval()

    print(f"{'query':<10}{'ML model':>14}{'PostgreSQL':>14}{'actual':>14}"
          f"{'q-err ML':>10}{'q-err PG':>10}  note")
    rows_out = []
    for name, spec in specs.items():
        tables = spec["tables"]
        joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
        preds = spec["preds"]
        body = build_body(tables, joins, [p[4] for p in preds])

        cur.execute("SELECT COUNT(*) " + body)
        actual = cur.fetchone()[0]
        cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]

        with torch.no_grad():
            x = torch.tensor([featurize(tables, joins, preds)], dtype=torch.float32)
            ml = float(np.expm1(model(x).item()))

        qml, qpg = qerror(ml, actual), qerror(pg_est, actual)
        print(f"{name:<10}{ml:>14.0f}{pg_est:>14}{actual:>14}"
              f"{qml:>10.2f}{qpg:>10.2f}  {spec['note']}")
        rows_out.append([name, ml, pg_est, actual, round(qml, 2), round(qpg, 2), spec["note"]])

    conn.close()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "ml_pred_rows", "pg_est_rows", "actual_rows", "qerr_ml", "qerr_pg", "note"])
        w.writerows(rows_out)
    print(f"\nSaved -> {OUT}")
    print("Coverage: 6 of 22 TPC-H queries encodable with the current predicate language.")


if __name__ == "__main__":
    main()