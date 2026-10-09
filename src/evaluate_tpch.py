"""Evaluate trained cardinality models on query cores over customer/orders/lineitem.

FIXED vs the old version:
  * ImportError: build_body renamed to build_where in gen_training_data; import updated.
  * Predicate tuple is now (key, lo, hi, sql) — 4 slots, not 5 (kind was dropped).
  * build_where(joins, pred_sqls) replaces build_body(tables, joins, pred_sqls).
  * Out-of-scope queries (q5-core, q10-core, q14-core) removed — they reference
    supplier/nation/part which are outside the 3-table training scope.
  * CUSTOM_SPECS dict at the top: add or comment out any query you like — no
    other code needs to change.

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
    JOIN_EDGES, FEAT_DIM, build_where, featurize, load_col_stats, norm,
)
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "tpch_eval_all.csv"


# -----------------------------------------------------------------------
# CUSTOM QUERY SPECS — add, remove, or comment out entries freely.
# All queries must use only: customer, orders, lineitem.
#
# Each entry is:
#   "label": dict(
#       tables = [...],   # subset of ["customer", "orders", "lineitem"]
#       note   = "...",   # printed in the summary table
#       preds  = [...],   # list of predicate tuples built by rng_pred / eq_num / eq_cat
#   )
# -----------------------------------------------------------------------
def build_specs(st):
    smin = st[("lineitem", "l_shipdate")]["min"]   # keep for Q1 / Q6
    smax = st[("lineitem", "l_shipdate")]["max"]
    omin = st[("orders",   "o_orderdate")]["min"]

    return {
        # ---- standard TPC-H cores (in-scope only) ----
        # "q1": dict(
        #     tables=["lineitem"], note="lineitem only, shipdate filter",
        #     preds=[rng_pred(st, "lineitem", "l_shipdate", smin,
        #                     date(1998, 9, 2).toordinal())],
        # ),
        # "q3": dict(
        #     tables=["customer", "orders", "lineitem"], note="full 3-table join",
        #     preds=[
        #         eq_cat(st, "customer", "c_mktsegment", "BUILDING"),
        #         rng_pred(st, "orders",   "o_orderdate",
        #                  omin, date(1995, 3, 15).toordinal()),
        #         rng_pred(st, "lineitem", "l_shipdate",
        #                  date(1995, 3, 15).toordinal(), smax),
        #     ],
        # ),
        # "q6": dict(
        #     tables=["lineitem"], note="lineitem only, shipdate+discount+qty",
        #     preds=[
        #         rng_pred(st, "lineitem", "l_shipdate",
        #                  date(1994, 1, 1).toordinal(),
        #                  date(1995, 1, 1).toordinal()),
        #         rng_pred(st, "lineitem", "l_discount", 0.02, 0.06),
        #         rng_pred(st, "lineitem", "l_quantity",
        #                  st[("lineitem", "l_quantity")]["min"], 24.0),
        #     ],
        # ),
        # "my-custom": dict(
        #     tables=["customer", "orders"],
        #     note="customer + orders, AUTOMOBILE segment, 1996",
        #     preds=[
        #         eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
        #         rng_pred(st, "orders", "o_orderdate",
        #                 date(1996, 1, 1).toordinal(), date(1996, 12, 31).toordinal()),
        #     ],
        # ),

        # ---- YOUR CUSTOM QUERY ----
        # "my-custom": dict(
        #     tables=["lineitem"],
        #     note="SELECT * FROM lineitem WHERE l_discount = 0.10;",
        #     preds=[
        #         eq_num(st, "lineitem", "l_discount", 0.10)
        #     ],
        # ),

        # ---- your new query ----
        "my-3table": dict(
            tables=["customer", "orders", "lineitem"],
            note="AUTOMOBILE + l_discount=0.10 (3-table join)",
            preds=[
                eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
                eq_num(st, "lineitem", "l_discount",   0.10),
            ],
        ),

        # Update Query check
        # "q-three-pred": dict(
        #     tables=["customer", "orders", "lineitem"],
        #     note="BUILDING + orderdate 1995-Q1 + l_quantity<10 (all 3 tables filtered)",
        #     preds=[
        #         eq_cat(st,  "customer", "c_mktsegment", "BUILDING"),
        #         rng_pred(st, "orders",  "o_orderdate",
        #                 date(1995, 1, 1).toordinal(),
        #                 date(1995, 3, 15).toordinal()),
        #         rng_pred(st, "lineitem", "l_quantity",
        #                 st[("lineitem", "l_quantity")]["min"], 10.0),
        #     ],
        # ),

        

        # ---- your custom queries — comment/uncomment as needed ----
        # "my-q-orders": dict(
        #     tables=["orders"],
        #     note="orders only, totalprice range",
        #     preds=[rng_pred(st, "orders", "o_totalprice", 1000.0, 50000.0)],
        # ),
        # "my-q-cust-orders": dict(
        #     tables=["customer", "orders"],
        #     note="customer+orders, AUTOMOBILE segment",
        #     preds=[
        #         eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
        #         rng_pred(st, "orders", "o_orderdate",
        #                  date(1993, 1, 1).toordinal(), date(1997, 12, 31).toordinal()),
        #     ],
        # ),

        # "machinery-totalprice": dict(
        #     tables=["customer", "orders", "lineitem"],
        #     note="MACHINERY + narrow totalprice + shipdate (one pred per table)",
        #     preds=[
        #         eq_cat(st,   "customer", "c_mktsegment",  "MACHINERY"),
        #         rng_pred(st, "orders",   "o_totalprice",   222545.92, 229639.44),
        #         rng_pred(st, "lineitem", "l_shipdate",
        #                 date(1996, 10, 27).toordinal(),
        #                 date(1998, 5,  28).toordinal()),
        #     ],
        # ),
    }


# ---------- predicate helpers ----------
# Predicate tuple format: ((table, col), lo, hi, sql_fragment)
# For exact match: lo == hi.  For range: lo < hi.  All values normalized.

def rng_pred(s, t, c, lo, hi):
    st = s[(t, c)]
    n1 = norm(lo, st["min"], st["max"])
    n2 = norm(hi, st["min"], st["max"])
    if st["kind"] == "date":
        sql = (f"{c} BETWEEN DATE '{date.fromordinal(int(lo)).isoformat()}'"
               f" AND DATE '{date.fromordinal(int(hi)).isoformat()}'")
    else:
        sql = f"{c} BETWEEN {lo:.2f} AND {hi:.2f}"
    return ((t, c), n1, n2, sql)


def eq_num(s, t, c, v):
    st = s[(t, c)]
    nv = norm(v, st["min"], st["max"])
    return ((t, c), nv, nv, f"{c} = {v:.2f}")


def eq_cat(s, t, c, v):
    dom = [str(d).strip() for d in s[(t, c)]["domain"]]
    v = str(v).strip()
    pos = (dom.index(v) + 1) / (len(dom) + 1)
    return ((t, c), pos, pos, f"{c} = '{v}'")


def qerror(p, a):
    p, a = max(p, 1), max(a, 1)
    return max(p / a, a / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of: mlp lgbm rf xgb (default: all available)")
    args = ap.parse_args()
    names = args.models or available_models()
    if not names:
        raise SystemExit("No trained models found in results/ — run the train_*.py scripts first.")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)
    specs = build_specs(stats)

    # ---- per query: build SQL once, get actual count + PG estimate once ----
    cases = []
    for label, spec in specs.items():
        tables = spec["tables"]
        joins  = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
        preds  = spec["preds"]
        # FIX: use build_where(joins, sqls) — build_body no longer exists
        where  = build_where(joins, [p[3] for p in preds])
        from_  = f"FROM {', '.join(tables)}"
        cur.execute(f"SELECT COUNT(*) {from_}{where}")
        actual = cur.fetchone()[0]
        cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_}{where}")
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
        cases.append(dict(label=label, spec=spec, from_=from_, where=where,
                          joins=joins, preds=preds, actual=actual, pg_est=pg_est))

    models = {n: CardModel(n) for n in names}

    print(f"{'query':<14}{'actual':>12}{'PG est':>14}{'q-err PG':>10}", end="")
    for n in names:
        print(f"{n + ' est':>14}{('q-err ' + n):>10}", end="")
    print("  note")
    print("-" * (62 + 24 * len(names)))

    rows_out, per_model_q = [], {n: [] for n in names}
    for case in cases:
        # FIX: featurize expects (tables, joins, preds) where each pred is
        # ((t,c), lo, hi, sql) — 4-tuple, matching the new gen_training_data API
        x = [featurize(case["spec"]["tables"], case["joins"], case["preds"])]
        print(f"{case['label']:<14}{case['actual']:>12}{case['pg_est']:>14}"
              f"{qerror(case['pg_est'], case['actual']):>10.2f}", end="")
        rows_out.append([case["label"], case["actual"], case["pg_est"],
                         round(qerror(case["pg_est"], case["actual"]), 2),
                         case["spec"]["note"]])
        for n in names:
            ml    = float(models[n].predict_rows(x, [case["pg_est"]])[0])
            qml   = qerror(ml, case["actual"])
            per_model_q[n].append(qml)
            print(f"{ml:>14.0f}{qml:>10.2f}", end="")
            rows_out[-1] += [round(ml, 1), round(qml, 2)]
        print(f"  {case['spec']['note']}")

    print("\n=== Summary (lower q-error = better) ===")
    print(f"{'model':<10}{'median q-err':>14}{'PG median':>12}")
    q_pg_all = [qerror(c["pg_est"], c["actual"]) for c in cases]
    for n in names:
        med = float(np.median(per_model_q[n]))
        print(f"{n:<10}{med:>14.2f}{float(np.median(q_pg_all)):>12.2f}")
        rows_out.append([f"median_{n}", "", "",
                         round(float(np.median(q_pg_all)), 2), "",
                         "", round(med, 2)])

    conn.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "actual_rows", "pg_est_rows", "qerr_pg", "note"]
                   + [c for n in names for c in (f"{n}_pred_rows", f"qerr_{n}")])
        w.writerows(rows_out)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()