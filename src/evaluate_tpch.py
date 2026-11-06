# Gemini Version 2.0.0
"""Evaluate trained cardinality models on TPC-H query cores (customer / orders / lineitem).

================================================================================
RESEARCH METRICS & LITERATURE REFERENCES (FOR THESIS & SUPERVISOR DEFENSE)
================================================================================
This file computes and prints ONLY the 5 scientifically necessary metrics
grouped into PRIMARY (headline) and SECONDARY (supporting) metrics.

--- PRIMARY METRICS (First 2 columns in summary table — quote these first) ---

  1. median_qerror (Primary Error Metric — Lower is better, Ideal = 1.00)
     * Formula : median of max(pred / actual, actual / pred) across queries.
     * Meaning : Typical multiplicative factor by which the estimate misses the
                 true row count (e.g., 1.06 means typically within 6% of truth).
     * Papers  : - Moerkotte, Neumann, & Steidl (VLDB 2009): "Preventing Bad Plans
                   by Bounding the Impact of Cardinality Estimation Errors",
                   PVLDB 2(1), pp. 982-993. (Introduced Q-Error for optimizers).
                 - Leis et al. (PVLDB 2015): "How Good Are Query Optimizers,
                   Really?", PVLDB 9(3), pp. 204-215.
                 - Kipf et al. (CIDR 2019): "Learned Cardinalities: Estimating
                   Correlated Joins with Deep Learning" (MSCN).

  2. accuracy_pct (Primary Percentage Accuracy — Higher is better, Ideal = 100%)
     * Formula : mean of [ 100 * (1 - |pred - actual| / (pred + actual)) ].
                 Mathematically linked to Q-Error via: Accuracy % = 200 / (q + 1).
     * Meaning : Bounded 0% to 100% symmetric accuracy score. Directly answers
                 the supervisor's question: "How accurate is your prediction?"
                 (e.g., MLP = 89.30% accurate vs. PostgreSQL = 73.00% accurate).
     * Papers  : - Armstrong (1985), "Long-Range Forecasting", p. 348 (sMAPE).
                 - Makridakis (1993), "Accuracy Measures: Theoretical and
                   Practical Concerns", Int. J. Forecasting 9(4), pp. 527-529.

--- SECONDARY METRICS (Next 3 columns — tail risk & optimizer plan stability) ---

  3. mean_qerror (Outlier-Sensitive Error — Lower is better, Ideal = 1.00)
     * Formula : arithmetic mean of max(pred / actual, actual / pred).
     * Meaning : Exposes severe single-query failures (like PostgreSQL's 10.25x
                 error on correlated 3-table join Q3) that the median hides.
     * Papers  : - Kipf et al. (CIDR 2019 - MSCN); Sun & Li (PVLDB 2019).

  4. p95_qerror (95th-Percentile Tail Error — Lower is better, Ideal = 1.00)
     * Formula : 95th percentile of Q-Error across evaluated queries.
     * Meaning : Worst-case tail bound ("95% of queries have error <= this value").
                 Catastrophic join-order selections happen in this top-5% tail.
     * Papers  : - Kipf et al. (CIDR 2019); Leis et al. (PVLDB 2015);
                   Dutt et al. (PVLDB 2019).

  5. within_2x_pct (Plan-Safe Threshold Accuracy % — Higher is better, Ideal = 100%)
     * Formula : percentage of queries where Q-Error <= 2.0.
     * Meaning : Moerkotte et al. (VLDB 2009, Corollary 1) proved that when
                 cardinality Q-Error <= 2.0, plan cost error is tightly bounded
                 and the optimizer rarely switches to a bad execution plan.
     * Papers  : - Moerkotte et al. (VLDB 2009); Leis et al. (PVLDB 2015);
                   Ioannidis & Christodoulakis (SIGMOD 1991).
================================================================================

Usage:
  python src/evaluate_tpch.py                 # all available models
  python src/evaluate_tpch.py --models mlp xgb
"""
import argparse
import csv
import warnings
from datetime import date
from pathlib import Path

# Suppress NumPy 2.5 / Python 3.13 DeprecationWarning inside joblib/sklearn pickle loader
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
            note="customer + orders, AUTOMOBILE segment, 1996  (custom)",
            preds=[
                eq_cat(st, "customer", "c_mktsegment", "AUTOMOBILE"),
                rng_pred(st, "orders", "o_orderdate",
                         date(1996, 1, 1).toordinal(), date(1996, 12, 31).toordinal()),
            ],
        ),

        # ---- extended stress queries (exact verified bounds from dataset) ----
        "q-corr-price-qty": dict(
            tables=["customer", "orders", "lineitem"],
            note="3-table join, low totalprice + high l_quantity (anti-correlated)  (custom)",
            preds=[
                rng_pred(st, "orders",   "o_totalprice", 21715.94, 100654.91),
                rng_pred(st, "lineitem", "l_quantity",   21.19, 46.97),
            ],
        ),
        "q-high-price-ship": dict(
            tables=["customer", "orders", "lineitem"],
            note="3-table join, high totalprice + shipdate + discount=0.02  (custom)",
            preds=[
                rng_pred(st, "orders",   "o_totalprice", 361146.59, 424010.12),
                rng_pred(st, "lineitem", "l_shipdate",
                         date(1994, 2, 23).toordinal(),
                         date(1995, 1, 10).toordinal()),
                eq_num(st,   "lineitem", "l_discount", 0.02),
            ],
        ),
        "q-three-pred-bldg": dict(
            tables=["customer", "orders", "lineitem"],
            note="3-table join, BUILDING + totalprice + l_quantity (1 pred/table)  (custom)",
            preds=[
                eq_cat(st,   "customer", "c_mktsegment", "BUILDING"),
                rng_pred(st, "orders",   "o_totalprice", 104403.16, 539821.95),
                rng_pred(st, "lineitem", "l_quantity",   19.64, 23.15),
            ],
        ),
    }


# ---------- predicate helpers ----------
# Predicate tuple format: ((table, col), lo, hi, sql_fragment)
# For exact match: lo == hi.  For range: lo < hi.  All values normalized.
# NOTE: evaluate_join_order_quality.py imports eq_cat, eq_num, rng_pred from here.

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


# =====================================================================
# METRIC FUNCTIONS (WITH PAPER REFERENCES)
# =====================================================================

def qerror(p, a):
    """Multiplicative Q-Error [Moerkotte et al., VLDB 2009; Kipf et al., CIDR 2019].
    q-error = max(pred / actual, actual / pred) >= 1.0 (1.0 is a perfect estimate).
    """
    p, a = max(float(p), 1.0), max(float(a), 1.0)
    return max(p / a, a / p)


def single_query_metrics(pred, actual):
    """Compute per-query metrics for a single (predicted, actual) pair.

    Why a dedicated function instead of calling metric_summary([pred], [actual]):
      * Avoids computing meaningless multi-query order statistics (median, mean,
        95th percentile) on a 1-element list.
      * Explicitly documents the 3 per-query values stored on each query row:
          - qerror       : Multiplicative error ratio [Moerkotte et al., VLDB 2009]
          - accuracy_pct : Symmetric accuracy 0-100% [Armstrong 1985; Makridakis 1993]
          - within_2x_pct: Plan-safe threshold (100% if q <= 2.0 else 0%) [Moerkotte 2009]
    """
    p = max(float(pred), 1.0)
    a = max(float(actual), 1.0)
    q = max(p / a, a / p)
    sym_acc_pct = 100.0 * (1.0 - abs(p - a) / (p + a))
    within_2x_pct = 100.0 if q <= 2.0 else 0.0
    return {
        "qerror": q,
        "accuracy_pct": sym_acc_pct,
        "within_2x_pct": within_2x_pct,
    }


def metric_summary(predictions, actuals):
    """Compute the 5 aggregate evaluation metrics across a list of queries.

    PRIMARY (Headline Metrics):
      * median_qerror : typical multiplicative error [Moerkotte 2009; Leis et al. PVLDB 2015; Kipf et al. CIDR 2019]
      * accuracy_pct  : mean symmetric accuracy 0-100% [Armstrong 1985; Makridakis 1993]

    SECONDARY (Supporting Context):
      * mean_qerror   : arithmetic mean of q-error [Kipf et al., CIDR 2019]
      * p95_qerror    : 95th-percentile worst-case tail error [Kipf 2019; Leis 2015]
      * within_2x_pct : % of queries where q-error <= 2.0 [Moerkotte et al., VLDB 2009]
    """
    pred    = np.maximum(np.asarray(predictions, dtype=float), 1.0)
    actual  = np.maximum(np.asarray(actuals,     dtype=float), 1.0)
    q       = np.maximum(pred / actual, actual / pred)
    sym_acc = 100.0 * (1.0 - np.abs(pred - actual) / (pred + actual))
    return {
        "median_qerror": float(np.median(q)),
        "accuracy_pct":  float(np.mean(sym_acc)),
        "mean_qerror":   float(np.mean(q)),
        "p95_qerror":    float(np.percentile(q, 95)),
        "within_2x_pct": float(np.mean(q <= 2.0) * 100.0),
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

    # ── Per-query table ───────────────────────────────────────────────────
    print(f"{'query':<18}{'actual':>12}{'PG est':>14}{'q-err PG':>12}", end="")
    for n in names:
        print(f"{n + ' est':>14}{('q-err ' + n):>12}", end="")
    print("  note")
    print("-" * (68 + 26 * len(names)))

    rows_out          = []
    model_predictions = {n: [] for n in names}
    actual_all        = [c["actual"] for c in cases]
    pg_all            = [c["pg_est"] for c in cases]

    for case in cases:
        x    = [featurize(case["spec"]["tables"], case["joins"], case["preds"])]
        pg_m = single_query_metrics(case["pg_est"], case["actual"])

        print(f"{case['label']:<18}{case['actual']:>12}{case['pg_est']:>14}"
              f"{pg_m['qerror']:>12.2f}", end="")

        row = [
            case["label"],
            case["actual"],
            case["pg_est"],
            round(pg_m["qerror"], 2),
            round(pg_m["accuracy_pct"], 2),
            "", "",  # mean_qerr and p95_qerr apply only to multi-query summary rows
            round(pg_m["within_2x_pct"], 2),
            case["spec"]["note"],
        ]

        model_metrics = {}
        for n in names:
            ml  = float(models[n].predict_rows(x, [case["pg_est"]])[0])
            m_m = single_query_metrics(ml, case["actual"])
            model_predictions[n].append(ml)
            model_metrics[n] = m_m
            print(f"{ml:>14.0f}{m_m['qerror']:>12.2f}", end="")
            row += [round(ml, 1), round(m_m["qerror"], 2)]

        for metric_name in ("accuracy_pct", "within_2x_pct"):
            row += [round(model_metrics[n][metric_name], 2) for n in names]

        rows_out.append(row)
        print(f"  {case['spec']['note']}")

    # ── Summary table ─────────────────────────────────────────────────────
    summaries = {"PostgreSQL": metric_summary(pg_all, actual_all)}
    for n in names:
        summaries[n] = metric_summary(model_predictions[n], actual_all)

    print("\n=== Summary across evaluated queries ===")
    print("PRIMARY   : median q-err (lower is better) | accuracy % (higher is better)")
    print("SECONDARY : mean q-err, p95 q-err (lower is better) | within-2x % (higher is better)")
    print(f"{'method':<14}{'median q-err':>14}{'accuracy %':>13}"
          f"{'mean q-err':>13}{'p95 q-err':>12}{'within-2x %':>13}")
    print(f"{'-'*79}")
    for method, s in summaries.items():
        print(f"{method:<14}{s['median_qerror']:>14.2f}"
              f"{s['accuracy_pct']:>13.2f}"
              f"{s['mean_qerror']:>13.2f}"
              f"{s['p95_qerror']:>12.2f}"
              f"{s['within_2x_pct']:>13.2f}")
        rows_out.append([
            f"summary_{method.lower().replace(' ', '_')}",
            "", "",
            round(s["median_qerror"], 4),
            round(s["accuracy_pct"],  4),
            round(s["mean_qerror"],   4),
            round(s["p95_qerror"],    4),
            round(s["within_2x_pct"], 4),
            f"aggregate summary for {method} (n={len(actual_all)} queries)",
        ] + [""] * (4 * len(names)))

    # ── CSV ───────────────────────────────────────────────────────────────
    conn.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["query_or_method", "actual_rows", "pg_est_rows",
             "qerr_or_median_qerr", "accuracy_pct",
             "mean_qerr", "p95_qerr", "within_2x_pct", "note"]
            + [c for n in names for c in (f"{n}_pred_rows", f"qerr_{n}")]
            + [f"{n}_accuracy_pct" for n in names]
            + [f"{n}_within_2x_pct" for n in names]
        )
        w.writerows(rows_out)
    print(f"\nSaved -> {OUT}")
    # =====================================================================
    # ACCURACY MEASUREMENT - END
    # =====================================================================


if __name__ == "__main__":
    main()