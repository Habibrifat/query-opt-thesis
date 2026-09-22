"""Join-order / plan-choice evaluation (replaces phase5_latency*.py).

WHAT IT MEASURES - the four layers of query-optimization evaluation:
  1. ESTIMATION QUALITY : median q-error of the model's predicted sub-plan
                          cardinalities vs real COUNT(*)   (are the numbers right?)
  2. COST-MODEL QUALITY : Spearman rho between a cost score and real exec time
                          across candidates                (does lower score mean faster?)
  3. PLAN-CHOICE QUALITY: is the cheapest candidate actually the fastest?
                          regret = t(chosen) / t(fastest)  (how costly is a wrong pick?)
  4. END-TO-END IMPACT  : chosen plan's exec time vs PostgreSQL's default plan
                          speedup = t_default / t_chosen   (the bottom line)

NEW vs the old version:
  * --models mlp rf xgb lgbm (default: every model you have) -> one combined
    results/join_order_quality_eval.csv
  * PostgreSQL's DEFAULT join order is added as a candidate, so
    'ML ranking correct' is a fair fight (your old run: PG default won with
    a plan that was not even among the candidates)
  * regret + speedup reported explicitly
  * sub-plan cardinality q-error column (estimation vs ranking separated)
  * --reps N to trade accuracy for time (EXPLAIN ANALYZE really executes
    the query - a 250 s candidate with reps=3 costs ~13 min)

Usage:
    python src/evaluate_join_order_quality.py                          # all models, 5table
    python src/evaluate_join_order_quality.py --scenario q3 --models mlp xgb --reps 1
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
OUT = BASE / "results" / "join_order_quality_eval.csv"

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


def extract_join_order(node, acc):
    """Pull table order out of a PostgreSQL plan tree (heuristic DFS)."""
    if node.get("Relation Name"):
        acc.append(node["Relation Name"])
    for ch in node.get("Plans", []):
        extract_join_order(ch, acc)


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
    return [{"name": n, "steps": [s[:i + 1] for i in range(1, len(s))],
             "sql": lambda ps, s=s: build_ordered_sql(s, ps)} for n, s in seqs.items()]


def preds_q3(stats):
    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
    ]


def candidates_q3():
    def right_deep_sql(ps):
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
         "sql": right_deep_sql},
    ]


SCENARIOS = {
    "5table": dict(default_tables=TABLES_5, preds_fn=preds_5table,
                   candidates_fn=candidates_5table),
    "q3": dict(default_tables=TABLES_3, preds_fn=preds_q3,
               candidates_fn=candidates_q3),
}


def pg_est_of(cur, body):
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def qerror(p, a):
    p, a = max(p, 1), max(a, 1)
    return max(p / a, a / p)


def plan_parts(steps, preds):
    """Yield (sub_tables, body, feature_vector) for every sub-plan."""
    for sub in steps:
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        yield sub, build_body(sub, joins, [p[4] for p in sub_preds]), \
            [featurize(sub, joins, sub_preds)]


def ml_cost(model, cur, steps, preds):
    total = 0.0
    for _, body, x in plan_parts(steps, preds):
        total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
    return total


def subplan_median_qerror(model, cur, steps, preds):
    qs = []
    for _, body, x in plan_parts(steps, preds):
        pred = float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
        cur.execute("SELECT COUNT(*) " + body)
        qs.append(qerror(pred, cur.fetchone()[0]))
    return float(np.median(qs))


def exec_stats(cur, sql, reps):
    times, cost = [], None
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]
        times.append(plan["Execution Time"])
        cost = plan["Plan"]["Total Cost"]
    return statistics.median(times), cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of mlp lgbm rf xgb (default: all available)")
    ap.add_argument("--scenario", default="5table", choices=list(SCENARIOS))
    ap.add_argument("--reps", type=int, default=3,
                    help="EXPLAIN ANALYZE repetitions per candidate (time vs noise)")
    args = ap.parse_args()
    scen = SCENARIOS[args.scenario]
    names = args.models or available_models()

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)

    preds = scen["preds_fn"](stats)
    pred_sqls = [p[4] for p in preds]
    candidates = scen["candidates_fn"]()

    # --- add PostgreSQL's own default join order as a candidate ------------
    joins_all = [e for e in JOIN_EDGES
                 if e[0] in scen["default_tables"] and e[2] in scen["default_tables"]]
    default_body = build_body(scen["default_tables"], joins_all, pred_sqls)
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + default_body)
    order = []
    extract_join_order(cur.fetchone()[0][0]["Plan"], order)
    order = [t for t in order if t in scen["default_tables"]]
    if len(order) == len(set(order)) and set(order) == set(scen["default_tables"]):
        candidates.append({"name": "pg-default-order",
                           "steps": [order[:i + 1] for i in range(1, len(order))],
                           "sql": lambda ps, s=order: build_ordered_sql(s, ps)})
    else:
        print(f"note: could not extract a clean default join order "
              f"(got {order}); keeping handwritten candidates only")

    cur.execute("SET join_collapse_limit = 1")  # preserve written join order

    # --- execute each candidate ONCE to get ground-truth times -------------
    print(f"scenario: {args.scenario} | candidates: {len(candidates)} | reps: {args.reps}")
    measured = []
    for c in candidates:
        t, c_pg = exec_stats(cur, c["sql"](pred_sqls), args.reps)
        measured.append((c, t, c_pg))
        print(f"  measured {c['name']:<24} {t:>10.1f} ms  (PG cost {c_pg:,.0f})")

    t_default, _ = exec_stats(cur, "SELECT * " + default_body, args.reps)
    cur.execute("SET join_collapse_limit = 8")
    print(f"  PG default planner      {t_default:>10.1f} ms\n")

    # --- score candidates with each model ----------------------------------
    times = np.array([m[1] for m in measured])
    pg_costs = np.array([m[2] for m in measured])
    fastest_idx = int(np.argmin(times))
    csv_rows, summary = [], []

    for n in names:
        model = CardModel(n)
        print(f"=== model: {n} ({model.target}) ===")
        print(f"{'join order':<26}{'ML rows':>14}{'med q-err':>10}"
              f"{'PG cost':>12}{'exec time':>12}")
        rows = []
        for c, t, c_pg in measured:
            cost_ml = ml_cost(model, cur, c["steps"], preds)
            med_q = subplan_median_qerror(model, cur, c["steps"], preds)
            rows.append((c["name"], cost_ml, med_q, c_pg, t))
            print(f"{c['name']:<26}{cost_ml:>14.0f}{med_q:>10.2f}"
                  f"{c_pg:>12.1f}{t:>10.1f} ms")
            csv_rows.append([args.scenario, n, c["name"], round(cost_ml, 1),
                             round(med_q, 2), round(c_pg, 1), round(t, 1),
                             "", "", "", "", ""])

        ml_idx = int(np.argmin([r[1] for r in rows]))
        pg_idx = int(np.argmin(pg_costs))
        regret = times[ml_idx] / times[fastest_idx]
        speedup = t_default / times[ml_idx]
        correct = (ml_idx == fastest_idx)
        rho_ml = spearmanr([r[1] for r in rows], times).statistic \
            if len(rows) >= 3 else float("nan")
        rho_pg = spearmanr(pg_costs, times).statistic

        print(f"  ML chose   : {rows[ml_idx][0].strip():<22} -> {times[ml_idx]:.1f} ms")
        print(f"  Actual best: {rows[fastest_idx][0].strip():<22} -> {times[fastest_idx]:.1f} ms")
        print(f"  PG default : {t_default:.1f} ms")
        print(f"  PLAN-CHOICE: ranking correct = {'YES' if correct else 'NO'} | "
              f"regret = {regret:.2f}x | speedup vs PG default = {speedup:.2f}x")
        print(f"  COST MODEL : rho(ML rows, time) = {rho_ml:+.2f} | "
              f"rho(PG cost, time) = {rho_pg:+.2f}")
        print(f"  ESTIMATION : median sub-plan q-error = "
              f"{np.median([r[2] for r in rows]):.2f}\n")
        csv_rows.append([args.scenario, n, "_summary", "", "", "",
                         round(times[ml_idx], 1), correct, round(regret, 2),
                         round(speedup, 2), round(rho_ml, 2), round(rho_pg, 2)])
        summary.append((n, correct, regret, speedup, rho_ml, rho_pg))

    print("=== model comparison (this scenario) ===")
    print(f"{'model':<8}{'rank OK':>9}{'regret':>9}{'speedup':>9}"
          f"{'rho ML':>9}{'rho PG':>9}")
    for n, correct, regret, speedup, rho_ml, rho_pg in summary:
        print(f"{n:<8}{str(correct):>9}{regret:>8.2f}x{speedup:>8.2f}x"
              f"{rho_ml:>+9.2f}{rho_pg:>+9.2f}")
    print("\nHow to read: regret 1.00x + speedup > 1 + rho near +1 = the model "
          "genuinely optimizes queries. regret >> 1 or rho <= 0 = it does not (yet).")

    conn.close()
    write_header = (not OUT.exists()) or OUT.stat().st_size == 0
    with open(OUT, "a", newline="") as f:   # append: safe to re-run per scenario
        w = csv.writer(f)
        if write_header:
            w.writerow(["scenario", "model", "candidate", "ml_rows", "med_qerr",
                        "pg_cost", "exec_ms", "ranking_correct", "regret",
                        "speedup_vs_pg", "rho_ml", "rho_pg"])
        w.writerows(csv_rows)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
