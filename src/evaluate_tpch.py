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
    JOIN_EDGES, build_where, featurize, load_col_stats, norm,
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
        "q1": dict(
            tables=["lineitem"], note="lineitem only, shipdate filter",
            preds=[rng_pred(st, "lineitem", "l_shipdate", smin,
                            date(1998, 9, 2).toordinal())],
        ),
        "q3": dict(
            tables=["customer", "orders", "lineitem"], note="full 3-table join",
            preds=[
                eq_cat(st, "customer", "c_mktsegment", "BUILDING"),
                # Official TPC-H Q3 uses strict date boundaries.
                bounded_pred(st, "orders", "o_orderdate",
                             omin, date(1995, 3, 15).toordinal(),
                             upper_op="<"),
                bounded_pred(st, "lineitem", "l_shipdate",
                             date(1995, 3, 15).toordinal(), smax,
                             lower_op=">"),
            ],
        ),
        "q6": dict(
            tables=["lineitem"], note="lineitem only, shipdate+discount+qty",
            preds=[
                # Official TPC-H Q6: [1994-01-01, 1995-01-01), discount
                # [0.04, 0.06], and quantity < 24.
                bounded_pred(st, "lineitem", "l_shipdate",
                             date(1994, 1, 1).toordinal(),
                             date(1995, 1, 1).toordinal(),
                             upper_op="<"),
                bounded_pred(st, "lineitem", "l_discount", 0.04, 0.06),
                bounded_pred(st, "lineitem", "l_quantity",
                             st[("lineitem", "l_quantity")]["min"], 24.0,
                             upper_op="<"),
            ],
        ),

        "my-custom": dict(
            tables=["customer", "orders"],
            note="customer + orders, AUTOMOBILE segment, 1996",
            preds=[
                eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
                rng_pred(st, "orders", "o_orderdate",
                        date(1996, 1, 1).toordinal(), date(1996, 12, 31).toordinal()),
            ],
        ),

        # Keep custom queries commented out of the accuracy summary.  They
        # are useful for experiments, but are not official TPC-H Q1/Q3/Q6.
        # "my-custom": dict(
        #     tables=["lineitem"],
        #     note="lineitem only, l_discount = 0.10",
        #     preds=[eq_num(st, "lineitem", "l_discount", 0.10)],
        # ),
        # "my-3table": dict(
        #     tables=["customer", "orders", "lineitem"],
        #     note="AUTOMOBILE + l_discount = 0.10 (3-table join)",
        #     preds=[
        #         eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
        #         eq_num(st, "lineitem", "l_discount", 0.10),
        #     ],
        # ),

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

#         SELECT * FROM customer
# JOIN orders   ON customer.c_custkey  = orders.o_custkey
# JOIN lineitem ON orders.o_orderkey   = lineitem.l_orderkey
# WHERE c_mktsegment = 'AUTOMOBILE'     -- string eq on customer
#   AND l_discount   = 0.10             -- numeric eq on lineitem

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

def bounded_pred(s, t, c, lo, hi, lower_op=">=", upper_op="<="):
    """Build a range feature and SQL with independently controlled bounds."""
    st = s[(t, c)]
    n1 = norm(lo, st["min"], st["max"])
    n2 = norm(hi, st["min"], st["max"])
    if st["kind"] == "date":
        sql = (f"{c} {lower_op} DATE '{date.fromordinal(int(lo)).isoformat()}'"
               f" AND {c} {upper_op} DATE '{date.fromordinal(int(hi)).isoformat()}'")
    else:
        sql = f"{c} {lower_op} {lo:.2f} AND {c} {upper_op} {hi:.2f}"
    return ((t, c), n1, n2, sql)


def rng_pred(s, t, c, lo, hi):
    return bounded_pred(s, t, c, lo, hi)


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


def metric_summary(predictions, actuals):
    """Calculate metrics over a list of query predictions.

    Every query contributes one value to each percentage metric. Therefore:
      * median_qerror is the median across queries;
      * mape_pct and mae_rows are means across queries;
      * accuracy_pct is mean symmetric accuracy across queries;
      * within_2x_pct is the percentage of queries within a factor of two.
    """
    # Values below one are floored so zero-row cases do not cause division
    # by zero. Report zero-row cases separately if they occur.
    pred = np.maximum(np.asarray(predictions, dtype=float), 1.0)
    actual = np.maximum(np.asarray(actuals, dtype=float), 1.0)
    # Absolute percentage error: magnitude of the error relative to reality.
    ape = np.abs(pred - actual) / actual
    q = np.maximum(pred / actual, actual / pred)
    # Symmetric accuracy avoids MAPE's unbounded 100 - MAPE problem.
    # 100% is perfect; 0% is the worst possible value.
    symmetric_accuracy = 100.0 * (
        1.0 - (np.abs(pred - actual) / (pred + actual))
    )
    mape = float(np.mean(ape) * 100.0)
    return {
        "median_qerror": float(np.median(q)),
        "mean_qerror": float(np.mean(q)),
        "p95_qerror": float(np.percentile(q, 95)),
        "mape_pct": mape,
        "median_ape_pct": float(np.median(ape) * 100.0),
        "accuracy_pct": float(np.mean(symmetric_accuracy)),
        "median_accuracy_pct": float(np.median(symmetric_accuracy)),
        "within_2x_pct": float(np.mean(q <= 2.0) * 100.0),
        "mae_rows": float(np.mean(np.abs(pred - actual))),
    }


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

    # =====================================================================
    # ACCURACY MEASUREMENT - START
    # This block obtains ground-truth rows and PostgreSQL's baseline estimate.
    # Keep this marker when reporting which code calculates accuracy.
    # =====================================================================
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

    # Load the models used for ML cardinality predictions.
    models = {n: CardModel(n) for n in names}

    # ACCURACY MEASUREMENT - per-query q-error:
    # q-error = max(predicted / actual, actual / predicted)
    print(f"{'query':<14}{'actual':>12}{'PG est':>14}{'q-err PG':>10}", end="")
    for n in names:
        print(f"{n + ' est':>14}{('q-err ' + n):>10}", end="")
    print("  note")
    print("-" * (62 + 24 * len(names)))

    rows_out = []
    model_predictions = {n: [] for n in names}
    actual_all = [case["actual"] for case in cases]
    pg_all = [case["pg_est"] for case in cases]
    for case in cases:
        # The model receives query features plus PostgreSQL's estimate.
        x = [featurize(case["spec"]["tables"], case["joins"], case["preds"])]
        pg_metrics = metric_summary([case["pg_est"]], [case["actual"]])
        print(f"{case['label']:<14}{case['actual']:>12}{case['pg_est']:>14}"
              f"{qerror(case['pg_est'], case['actual']):>10.2f}", end="")
        rows_out.append([
            case["label"], case["actual"], case["pg_est"],
            round(pg_metrics["median_qerror"], 2),
            round(pg_metrics["mean_qerror"], 2),
            round(pg_metrics["p95_qerror"], 2),
            round(pg_metrics["mape_pct"], 2),
            round(pg_metrics["median_ape_pct"], 2),
            round(pg_metrics["accuracy_pct"], 2),
            round(pg_metrics["median_accuracy_pct"], 2),
            round(pg_metrics["within_2x_pct"], 2),
            round(pg_metrics["mae_rows"], 2),
            case["spec"]["note"],
        ])
        model_metrics = {}
        for n in names:
            ml    = float(models[n].predict_rows(x, [case["pg_est"]])[0])
            qml   = qerror(ml, case["actual"])
            model_predictions[n].append(ml)
            model_metrics[n] = metric_summary([ml], [case["actual"]])
            print(f"{ml:>14.0f}{qml:>10.2f}", end="")
            rows_out[-1] += [round(ml, 1), round(qml, 2)]
        for metric_name in ("mape_pct", "accuracy_pct", "within_2x_pct", "mae_rows"):
            rows_out[-1] += [
                round(model_metrics[n][metric_name], 2) for n in names
            ]
        print(f"  {case['spec']['note']}")

    # ACCURACY MEASUREMENT - aggregate result:
    # MAPE = mean absolute percentage error across queries.
    # Accuracy = mean [100 * (1 - abs(pred-actual)/(pred+actual))].
    # This symmetric accuracy is bounded from 0% to 100%.
    # Within 2x = percentage of queries whose q-error is at most 2.
    summaries = {"PostgreSQL": metric_summary(pg_all, actual_all)}
    for n in names:
        summaries[n] = metric_summary(model_predictions[n], actual_all)

    print("\n=== Summary across evaluated queries ===")
    print("Median q-error is the primary metric; lower is better.")
    print("MAPE/MAE are means; accuracy and within-2x are percentages.")
    print(f"{'method':<14}{'median q-err':>15}{'mean q-err':>12}"
          f"{'p95 q-err':>12}{'MAPE %':>10}{'accuracy %':>13}"
          f"{'within 2x %':>14}{'MAE rows':>14}")
    for method, summary in summaries.items():
        print(f"{method:<14}{summary['median_qerror']:>15.2f}"
              f"{summary['mean_qerror']:>12.2f}{summary['p95_qerror']:>12.2f}"
              f"{summary['mape_pct']:>10.2f}{summary['accuracy_pct']:>13.2f}"
              f"{summary['within_2x_pct']:>14.2f}{summary['mae_rows']:>14.1f}")

        rows_out.append([
            f"summary_{method.lower().replace(' ', '_')}", "", "",
            round(summary["median_qerror"], 4),
            round(summary["mean_qerror"], 4),
            round(summary["p95_qerror"], 4),
            round(summary["mape_pct"], 4),
            round(summary["median_ape_pct"], 4),
            round(summary["accuracy_pct"], 4),
            round(summary["median_accuracy_pct"], 4),
            round(summary["within_2x_pct"], 4),
            round(summary["mae_rows"], 4),
            "",
        ] + [""] * (6 * len(names)))

    # ACCURACY MEASUREMENT - CSV output:
    # Stores actual rows, PostgreSQL rows, ML rows, and all q-errors.
    conn.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "query", "actual_rows", "pg_est_rows", "qerr_pg",
            "mean_qerr_pg", "p95_qerr_pg", "mape_pg_pct",
            "median_ape_pg_pct", "accuracy_pg_pct",
            "median_accuracy_pg_pct", "within_2x_pg_pct", "mae_pg_rows",
            "note",
        ] + [
            c for n in names for c in (
                f"{n}_pred_rows", f"qerr_{n}",
            )
        ] + [
            f"{n}_mape_pct" for n in names
        ] + [
            f"{n}_accuracy_pct" for n in names
        ] + [
            f"{n}_within_2x_pct" for n in names
        ] + [
            f"{n}_mae_rows" for n in names
        ])
        w.writerows(rows_out)
    print(f"\nSaved -> {OUT}")
    # =====================================================================
    # ACCURACY MEASUREMENT - END
    # =====================================================================


if __name__ == "__main__":
    main()
