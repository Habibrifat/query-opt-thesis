"""Evaluate ALL trained cardinality models on TPC-H query cores.

FIXED vs the old version:
  * old: loaded ONLY results/model.pt (wrong name -> FileNotFoundError)
         and predicted expm1(raw)  (wrong for ratio models)
  * new: --models mlp lgbm rf xgb (default: every model that has a saved
         artifact), prediction = pg_est * exp(raw) via model_loader,
         one combined CSV (results/tpch_eval_all.csv) for the thesis table.
Usage:
  python src/evaluate_tpch.py                 # all available models
  python src/evaluate_tpch.py --models mlp xgb
"""
import argparse
import csv
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2

from config import DB_CONFIG
from gen_training_data import (
    JOIN_EDGES, FEAT_DIM, build_body, featurize, load_col_stats, norm,
)
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "tpch_eval_all.csv"


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
    # strip() handles CHAR(n) padding, e.g. 'BUILDING  ' -> 'BUILDING'
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
                rng_pred(st, "lineitem", "l_quantity",
                         st[("lineitem", "l_quantity")]["min"], 24.0),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of: mlp lgbm rf xgb (default: all available)")
    args = ap.parse_args()
    names = args.models or available_models()
    if not names:
        raise SystemExit("No trained models found in results/ - run the train_*.py scripts first.")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)
    specs = build_specs(stats)

    # ---- per query: build SQL once, get actual + PG estimate once ----
    cases = []
    for name, spec in specs.items():
        tables = spec["tables"]
        joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
        preds = spec["preds"]
        body = build_body(tables, joins, [p[4] for p in preds])
        cur.execute("SELECT COUNT(*) " + body)
        actual = cur.fetchone()[0]
        cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
        cases.append(dict(name=name, spec=spec, body=body, joins=joins,
                          preds=preds, actual=actual, pg_est=pg_est))

    models = {n: CardModel(n) for n in names}   # load each model once

    print(f"{'query':<10}{'actual':>12}{'PG est':>14}{'q-err PG':>10}", end="")
    for n in names:
        print(f"{n + ' est':>14}{('q-err ' + n):>10}", end="")
    print("  note")
    print("-" * (56 + 24 * len(names)))

    rows_out, per_model_q = [], {n: [] for n in names}
    for case in cases:
        x = [featurize(case["spec"]["tables"], case["joins"], case["preds"])]
        print(f"{case['name']:<10}{case['actual']:>12}{case['pg_est']:>14}"
              f"{qerror(case['pg_est'], case['actual']):>10.2f}", end="")
        rows_out.append([case["name"], case["actual"], case["pg_est"],
                         round(qerror(case["pg_est"], case["actual"]), 2), case["spec"]["note"]])
        for n in names:
            ml = float(models[n].predict_rows(x, [case["pg_est"]])[0])
            qml = qerror(ml, case["actual"])
            per_model_q[n].append(qml)
            print(f"{ml:>14.0f}{qml:>10.2f}", end="")
            rows_out[-1] += [round(ml, 1), round(qml, 2)]
        print(f"  {case['spec']['note']}")

    print("\n=== Summary across the 6 TPC-H cores (lower = better) ===")
    print(f"{'model':<10}{'median q-err':>14}{'PG median':>12}")
    q_pg_all = [qerror(c["pg_est"], c["actual"]) for c in cases]
    for n in names:
        med = float(np.median(per_model_q[n]))
        print(f"{n:<10}{med:>14.2f}{float(np.median(q_pg_all)):>12.2f}")
        rows_out.append([f"median_{n}", "", "", round(float(np.median(q_pg_all)), 2), "",
                         "", round(med, 2)])

    conn.close()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "actual_rows", "pg_est_rows", "qerr_pg", "note"]
                   + [c for n in names for c in (f"{n}_pred_rows", f"qerr_{n}")])
        w.writerows(rows_out)
    print(f"\nSaved -> {OUT}")
    print("Coverage: 6 of 22 TPC-H queries encodable with the current predicate language.")


if __name__ == "__main__":
    main()
