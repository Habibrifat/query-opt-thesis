# """Join-order / plan-choice evaluation (replaces phase5_latency*.py).

# WHAT IT MEASURES - the four layers of query-optimization evaluation:
#   1. ESTIMATION QUALITY : median q-error of the model's predicted sub-plan
#                           cardinalities vs real COUNT(*)   (are the numbers right?)
#   2. COST-MODEL QUALITY : Spearman rho between a cost score and real exec time
#                           across candidates                (does lower score mean faster?)
#   3. PLAN-CHOICE QUALITY: is the cheapest candidate actually the fastest?
#                           regret = t(chosen) / t(fastest)  (how costly is a wrong pick?)
#   4. END-TO-END IMPACT  : chosen plan's exec time vs PostgreSQL's default plan
#                           speedup = t_default / t_chosen   (the bottom line)

# NEW vs the old version:
#   * --models mlp rf xgb lgbm (default: every model you have) -> one combined
#     results/join_order_quality_eval.csv
#   * PostgreSQL's DEFAULT join order is added as a candidate, so
#     'ML ranking correct' is a fair fight (your old run: PG default won with
#     a plan that was not even among the candidates)
#   * regret + speedup reported explicitly
#   * sub-plan cardinality q-error column (estimation vs ranking separated)
#   * --reps N to trade accuracy for time (EXPLAIN ANALYZE really executes
#     the query - a 250 s candidate with reps=3 costs ~13 min)

# Usage:
#     python src/evaluate_join_order_quality.py                          # all models, 5table
#     python src/evaluate_join_order_quality.py --scenario q3 --models mlp xgb --reps 1
# """
# import argparse
# import csv
# import statistics
# from datetime import date
# from pathlib import Path

# import numpy as np
# import psycopg2
# from scipy.stats import spearmanr

# from config import DB_CONFIG
# from evaluate_tpch import eq_cat, rng_pred
# from gen_training_data import JOIN_EDGES, build_body, featurize, load_col_stats
# from model_loader import CardModel, available_models

# BASE = Path(__file__).resolve().parent.parent
# OUT = BASE / "results" / "join_order_quality_eval.csv"

# TABLES_5 = ["customer", "orders", "lineitem", "supplier", "nation"]
# TABLES_3 = ["customer", "orders", "lineitem"]


# def join_clause(t, prefix):
#     for a, ac, b, bc in JOIN_EDGES:
#         if a == t and b in prefix:
#             return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
#         if b == t and a in prefix:
#             return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
#     raise ValueError(f"no join edge available for {t}")


# def build_ordered_sql(seq, pred_sqls):
#     sql = f"FROM {seq[0]} " + " ".join(
#         join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
#     if pred_sqls:
#         sql += " WHERE " + " AND ".join(pred_sqls)
#     return "SELECT * " + sql


# def extract_join_order(node, acc):
#     """Pull table order out of a PostgreSQL plan tree (heuristic DFS)."""
#     if node.get("Relation Name"):
#         acc.append(node["Relation Name"])
#     for ch in node.get("Plans", []):
#         extract_join_order(ch, acc)


# def preds_5table(stats):
#     return [rng_pred(stats, "orders", "o_orderdate",
#                      date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal())]


# def candidates_5table():
#     seqs = {
#         "dim-first  (n-c-o-l-s)": ["nation", "customer", "orders", "lineitem", "supplier"],
#         "cust-first (c-o-l-s-n)": ["customer", "orders", "lineitem", "supplier", "nation"],
#         "fact-early (o-l-s-c-n)": ["orders", "lineitem", "supplier", "customer", "nation"],
#         "supp-first (s-l-o-c-n)": ["supplier", "lineitem", "orders", "customer", "nation"],
#     }
#     return [{"name": n, "steps": [s[:i + 1] for i in range(1, len(s))],
#              "sql": lambda ps, s=s: build_ordered_sql(s, ps)} for n, s in seqs.items()]


# def preds_q3(stats):
#     smax = stats[("lineitem", "l_shipdate")]["max"]
#     omin = stats[("orders", "o_orderdate")]["min"]
#     return [
#         eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
#         rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
#         rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
#     ]


# def candidates_q3():
#     def right_deep_sql(ps):
#         sql = ("SELECT * FROM customer JOIN "
#                "(orders JOIN lineitem ON l_orderkey = o_orderkey) "
#                "ON c_custkey = o_custkey")
#         if ps:
#             sql += " WHERE " + " AND ".join(ps)
#         return sql
#     return [
#         {"name": "left-deep  (c-o-l)",
#          "steps": [["customer", "orders"], TABLES_3],
#          "sql": lambda ps: build_ordered_sql(["customer", "orders", "lineitem"], ps)},
#         {"name": "left-deep  (o-l-c)",
#          "steps": [["orders", "lineitem"], TABLES_3],
#          "sql": lambda ps: build_ordered_sql(["orders", "lineitem", "customer"], ps)},
#         {"name": "right-deep (c-(o-l))",
#          "steps": [["orders", "lineitem"], TABLES_3],
#          "sql": right_deep_sql},
#     ]


# SCENARIOS = {
#     "5table": dict(default_tables=TABLES_5, preds_fn=preds_5table,
#                    candidates_fn=candidates_5table),
#     "q3": dict(default_tables=TABLES_3, preds_fn=preds_q3,
#                candidates_fn=candidates_q3),
# }


# def pg_est_of(cur, body):
#     cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
#     return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


# def qerror(p, a):
#     p, a = max(p, 1), max(a, 1)
#     return max(p / a, a / p)


# def plan_parts(steps, preds):
#     """Yield (sub_tables, body, feature_vector) for every sub-plan."""
#     for sub in steps:
#         joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
#         sub_preds = [p for p in preds if p[0][0] in sub]
#         yield sub, build_body(sub, joins, [p[4] for p in sub_preds]), \
#             [featurize(sub, joins, sub_preds)]


# def ml_cost(model, cur, steps, preds):
#     total = 0.0
#     for _, body, x in plan_parts(steps, preds):
#         total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
#     return total


# def subplan_median_qerror(model, cur, steps, preds):
#     qs = []
#     for _, body, x in plan_parts(steps, preds):
#         pred = float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
#         cur.execute("SELECT COUNT(*) " + body)
#         qs.append(qerror(pred, cur.fetchone()[0]))
#     return float(np.median(qs))


# def exec_stats(cur, sql, reps):
#     times, cost = [], None
#     for _ in range(reps):
#         cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
#         plan = cur.fetchone()[0][0]
#         times.append(plan["Execution Time"])
#         cost = plan["Plan"]["Total Cost"]
#     return statistics.median(times), cost


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--models", nargs="*", default=None,
#                     help="subset of mlp lgbm rf xgb (default: all available)")
#     ap.add_argument("--scenario", default="5table", choices=list(SCENARIOS))
#     ap.add_argument("--reps", type=int, default=3,
#                     help="EXPLAIN ANALYZE repetitions per candidate (time vs noise)")
#     args = ap.parse_args()
#     scen = SCENARIOS[args.scenario]
#     names = args.models or available_models()

#     conn = psycopg2.connect(**DB_CONFIG)
#     conn.autocommit = True
#     cur = conn.cursor()
#     stats = load_col_stats(cur)

#     preds = scen["preds_fn"](stats)
#     pred_sqls = [p[4] for p in preds]
#     candidates = scen["candidates_fn"]()

#     # --- add PostgreSQL's own default join order as a candidate ------------
#     joins_all = [e for e in JOIN_EDGES
#                  if e[0] in scen["default_tables"] and e[2] in scen["default_tables"]]
#     default_body = build_body(scen["default_tables"], joins_all, pred_sqls)
#     cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + default_body)
#     order = []
#     extract_join_order(cur.fetchone()[0][0]["Plan"], order)
#     order = [t for t in order if t in scen["default_tables"]]
#     if len(order) == len(set(order)) and set(order) == set(scen["default_tables"]):
#         candidates.append({"name": "pg-default-order",
#                            "steps": [order[:i + 1] for i in range(1, len(order))],
#                            "sql": lambda ps, s=order: build_ordered_sql(s, ps)})
#     else:
#         print(f"note: could not extract a clean default join order "
#               f"(got {order}); keeping handwritten candidates only")

#     cur.execute("SET join_collapse_limit = 1")  # preserve written join order

#     # --- execute each candidate ONCE to get ground-truth times -------------
#     print(f"scenario: {args.scenario} | candidates: {len(candidates)} | reps: {args.reps}")
#     measured = []
#     for c in candidates:
#         t, c_pg = exec_stats(cur, c["sql"](pred_sqls), args.reps)
#         measured.append((c, t, c_pg))
#         print(f"  measured {c['name']:<24} {t:>10.1f} ms  (PG cost {c_pg:,.0f})")

#     t_default, _ = exec_stats(cur, "SELECT * " + default_body, args.reps)
#     cur.execute("SET join_collapse_limit = 8")
#     print(f"  PG default planner      {t_default:>10.1f} ms\n")

#     # --- score candidates with each model ----------------------------------
#     times = np.array([m[1] for m in measured])
#     pg_costs = np.array([m[2] for m in measured])
#     fastest_idx = int(np.argmin(times))
#     csv_rows, summary = [], []

#     for n in names:
#         model = CardModel(n)
#         print(f"=== model: {n} ({model.target}) ===")
#         print(f"{'join order':<26}{'ML rows':>14}{'med q-err':>10}"
#               f"{'PG cost':>12}{'exec time':>12}")
#         rows = []
#         for c, t, c_pg in measured:
#             cost_ml = ml_cost(model, cur, c["steps"], preds)
#             med_q = subplan_median_qerror(model, cur, c["steps"], preds)
#             rows.append((c["name"], cost_ml, med_q, c_pg, t))
#             print(f"{c['name']:<26}{cost_ml:>14.0f}{med_q:>10.2f}"
#                   f"{c_pg:>12.1f}{t:>10.1f} ms")
#             csv_rows.append([args.scenario, n, c["name"], round(cost_ml, 1),
#                              round(med_q, 2), round(c_pg, 1), round(t, 1),
#                              "", "", "", "", ""])

#         ml_idx = int(np.argmin([r[1] for r in rows]))
#         pg_idx = int(np.argmin(pg_costs))
#         regret = times[ml_idx] / times[fastest_idx]
#         speedup = t_default / times[ml_idx]
#         correct = (ml_idx == fastest_idx)
#         rho_ml = spearmanr([r[1] for r in rows], times).statistic \
#             if len(rows) >= 3 else float("nan")
#         rho_pg = spearmanr(pg_costs, times).statistic

#         print(f"  ML chose   : {rows[ml_idx][0].strip():<22} -> {times[ml_idx]:.1f} ms")
#         print(f"  Actual best: {rows[fastest_idx][0].strip():<22} -> {times[fastest_idx]:.1f} ms")
#         print(f"  PG default : {t_default:.1f} ms")
#         print(f"  PLAN-CHOICE: ranking correct = {'YES' if correct else 'NO'} | "
#               f"regret = {regret:.2f}x | speedup vs PG default = {speedup:.2f}x")
#         print(f"  COST MODEL : rho(ML rows, time) = {rho_ml:+.2f} | "
#               f"rho(PG cost, time) = {rho_pg:+.2f}")
#         print(f"  ESTIMATION : median sub-plan q-error = "
#               f"{np.median([r[2] for r in rows]):.2f}\n")
#         csv_rows.append([args.scenario, n, "_summary", "", "", "",
#                          round(times[ml_idx], 1), correct, round(regret, 2),
#                          round(speedup, 2), round(rho_ml, 2), round(rho_pg, 2)])
#         summary.append((n, correct, regret, speedup, rho_ml, rho_pg))

#     print("=== model comparison (this scenario) ===")
#     print(f"{'model':<8}{'rank OK':>9}{'regret':>9}{'speedup':>9}"
#           f"{'rho ML':>9}{'rho PG':>9}")
#     for n, correct, regret, speedup, rho_ml, rho_pg in summary:
#         print(f"{n:<8}{str(correct):>9}{regret:>8.2f}x{speedup:>8.2f}x"
#               f"{rho_ml:>+9.2f}{rho_pg:>+9.2f}")
#     print("\nHow to read: regret 1.00x + speedup > 1 + rho near +1 = the model "
#           "genuinely optimizes queries. regret >> 1 or rho <= 0 = it does not (yet).")

#     conn.close()
#     write_header = (not OUT.exists()) or OUT.stat().st_size == 0
#     with open(OUT, "a", newline="") as f:   # append: safe to re-run per scenario
#         w = csv.writer(f)
#         if write_header:
#             w.writerow(["scenario", "model", "candidate", "ml_rows", "med_qerr",
#                         "pg_cost", "exec_ms", "ranking_correct", "regret",
#                         "speedup_vs_pg", "rho_ml", "rho_pg"])
#         w.writerows(csv_rows)
#     print(f"Saved -> {OUT}")


# if __name__ == "__main__":
#     main()







# """Join-order / plan-choice evaluation.

# WHAT IT MEASURES - the four layers of query-optimization evaluation:
#   1. ESTIMATION QUALITY : median q-error of the model's predicted sub-plan
#                           cardinalities vs real COUNT(*)   (are the numbers right?)
#   2. COST-MODEL QUALITY : Spearman rho between a cost score and real exec time
#                           across candidates                (does lower score mean faster?)
#   3. PLAN-CHOICE QUALITY: is the cheapest candidate actually the fastest?
#                           regret = t(chosen) / t(fastest)  (how costly is a wrong pick?)
#   4. END-TO-END IMPACT  : chosen plan's exec time vs PostgreSQL's default plan
#                           speedup = t_default / t_chosen   (the bottom line)

# FIXED vs old version:
#   * build_body renamed to build_where in gen_training_data; import + all
#     call sites updated.
#   * Predicate tuple is now 4-slot (key, lo, hi, sql); index [3] for sql,
#     not [4].
#   * 5table scenario replaced with q3-variants (3-table only) to stay within
#     the customer/orders/lineitem training scope. supplier/nation are out of
#     scope and would break featurize().
#   * plan_parts() now builds "FROM ..." + WHERE separately (build_where API).

# Usage:
#     python src/evaluate_join_order_quality.py                   # all models, q3
#     python src/evaluate_join_order_quality.py --scenario q3-ordersonly --models mlp xgb --reps 1
# """
# import argparse
# import csv
# import statistics
# from datetime import date
# from pathlib import Path

# import numpy as np
# import psycopg2
# from scipy.stats import spearmanr

# from config import DB_CONFIG
# from evaluate_tpch import eq_cat, rng_pred
# from gen_training_data import JOIN_EDGES, build_where, featurize, load_col_stats
# from model_loader import CardModel, available_models

# BASE = Path(__file__).resolve().parent.parent
# OUT = BASE / "results" / "join_order_quality_eval.csv"

# TABLES_3 = ["customer", "orders", "lineitem"]


# # ---------- SQL helpers ----------

# def join_clause(t, prefix):
#     for a, ac, b, bc in JOIN_EDGES:
#         if a == t and b in prefix:
#             return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
#         if b == t and a in prefix:
#             return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
#     raise ValueError(f"no join edge available for {t}")


# def build_ordered_sql(seq, pred_sqls):
#     """Build a runnable SELECT * with an explicit join order."""
#     from_part = f"FROM {seq[0]} " + " ".join(
#         join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
#     where_part = (" WHERE " + " AND ".join(pred_sqls)) if pred_sqls else ""
#     return "SELECT * " + from_part + where_part


# def extract_join_order(node, acc):
#     """Pull table order out of a PostgreSQL plan tree (heuristic DFS)."""
#     if node.get("Relation Name"):
#         acc.append(node["Relation Name"])
#     for ch in node.get("Plans", []):
#         extract_join_order(ch, acc)


# # ---------- scenario definitions (3-table only) ----------

# def preds_q3(stats):
#     # FIX: pred tuple is now (key, lo, hi, sql) — 4 slots
#     omin = stats[("orders", "o_orderdate")]["min"]
#     return [
#         eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
#         rng_pred(stats, "orders", "o_orderdate",
#                  omin, date(1995, 3, 15).toordinal()),
#     ]


# def candidates_q3():
#     return [
#         {"name": "left-deep  (c-o-l)",
#          "steps": [["customer", "orders"], TABLES_3],
#          "sql": lambda ps: build_ordered_sql(["customer", "orders", "lineitem"], ps)},
#         {"name": "left-deep  (o-l-c)",
#          "steps": [["orders", "lineitem"], TABLES_3],
#          "sql": lambda ps: build_ordered_sql(["orders", "lineitem", "customer"], ps)},
#         {"name": "left-deep  (l-o-c)",
#          "steps": [["lineitem", "orders"], TABLES_3],
#          "sql": lambda ps: build_ordered_sql(["lineitem", "orders", "customer"], ps)},
#     ]


# def preds_orders_only(stats):
#     omin = stats[("orders", "o_orderdate")]["min"]
#     return [
#         rng_pred(stats, "orders", "o_orderdate",
#                  date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal()),
#     ]


# def candidates_orders_only():
#     """2-table join candidates: orders joined with customer or lineitem."""
#     return [
#         {"name": "orders-customer",
#          "steps": [["orders", "customer"]],
#          "sql": lambda ps: build_ordered_sql(["orders", "customer"], ps)},
#         {"name": "customer-orders",
#          "steps": [["customer", "orders"]],
#          "sql": lambda ps: build_ordered_sql(["customer", "orders"], ps)},
#         {"name": "orders-lineitem",
#          "steps": [["orders", "lineitem"]],
#          "sql": lambda ps: build_ordered_sql(["orders", "lineitem"], ps)},
#         {"name": "lineitem-orders",
#          "steps": [["lineitem", "orders"]],
#          "sql": lambda ps: build_ordered_sql(["lineitem", "orders"], ps)},
#     ]


# SCENARIOS = {
#     "q3": dict(
#         default_tables=TABLES_3,
#         preds_fn=preds_q3,
#         candidates_fn=candidates_q3,
#     ),
#     "q3-ordersonly": dict(
#         default_tables=["orders", "customer"],
#         preds_fn=preds_orders_only,
#         candidates_fn=candidates_orders_only,
#     ),
# }


# # ---------- evaluation helpers ----------

# def pg_est_of(cur, from_clause, where_clause):
#     cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_clause}{where_clause}")
#     return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


# def qerror(p, a):
#     p, a = max(p, 1), max(a, 1)
#     return max(p / a, a / p)


# def plan_parts(steps, preds):
#     """Yield (sub_tables, from_clause, where_clause, feature_vector) per sub-plan.

#     FIX: uses build_where(joins, sqls) + separate from_clause string.
#          Predicate sql is now at index [3] (was [4] in the old 5-slot tuple).
#     """
#     for sub in steps:
#         joins     = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
#         sub_preds = [p for p in preds if p[0][0] in sub]
#         from_cl   = f"FROM {', '.join(sub)}"
#         where_cl  = build_where(joins, [p[3] for p in sub_preds])   # FIX: [3] not [4]
#         yield sub, from_cl, where_cl, [featurize(sub, joins, sub_preds)]


# def ml_cost(model, cur, steps, preds):
#     total = 0.0
#     for _, from_cl, where_cl, x in plan_parts(steps, preds):
#         pg_est = pg_est_of(cur, from_cl, where_cl)
#         total += float(model.predict_rows(x, [pg_est])[0])
#     return total


# def subplan_median_qerror(model, cur, steps, preds):
#     qs = []
#     for _, from_cl, where_cl, x in plan_parts(steps, preds):
#         pg_est = pg_est_of(cur, from_cl, where_cl)
#         pred   = float(model.predict_rows(x, [pg_est])[0])
#         cur.execute(f"SELECT COUNT(*) {from_cl}{where_cl}")
#         qs.append(qerror(pred, cur.fetchone()[0]))
#     return float(np.median(qs))


# def exec_stats(cur, sql, reps):
#     times, cost = [], None
#     for _ in range(reps):
#         cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
#         plan = cur.fetchone()[0][0]
#         times.append(plan["Execution Time"])
#         cost = plan["Plan"]["Total Cost"]
#     return statistics.median(times), cost


# # ---------- main ----------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--models", nargs="*", default=None,
#                     help="subset of mlp lgbm rf xgb (default: all available)")
#     ap.add_argument("--scenario", default="q3", choices=list(SCENARIOS),
#                     help="which join-order scenario to run (default: q3)")
#     ap.add_argument("--reps", type=int, default=3,
#                     help="EXPLAIN ANALYZE repetitions per candidate")
#     args = ap.parse_args()
#     scen  = SCENARIOS[args.scenario]
#     names = args.models or available_models()

#     conn = psycopg2.connect(**DB_CONFIG)
#     conn.autocommit = True
#     cur = conn.cursor()
#     stats = load_col_stats(cur)

#     preds     = scen["preds_fn"](stats)
#     pred_sqls = [p[3] for p in preds]          # FIX: index [3] not [4]
#     candidates = scen["candidates_fn"]()

#     # --- add PostgreSQL's own default join order as a candidate ------------
#     default_tables = scen["default_tables"]
#     joins_all  = [e for e in JOIN_EDGES
#                   if e[0] in default_tables and e[2] in default_tables]
#     from_def   = f"FROM {', '.join(default_tables)}"
#     where_def  = build_where(joins_all, pred_sqls)             # FIX: build_where
#     cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_def}{where_def}")
#     order = []
#     extract_join_order(cur.fetchone()[0][0]["Plan"], order)
#     order = [t for t in order if t in default_tables]
#     if len(order) == len(set(order)) and set(order) == set(default_tables):
#         candidates.append({
#             "name": "pg-default-order",
#             "steps": [order[:i + 1] for i in range(1, len(order))],
#             "sql": lambda ps, s=order: build_ordered_sql(s, ps),
#         })
#     else:
#         print(f"note: could not extract clean default join order "
#               f"(got {order}); keeping handwritten candidates only")

#     cur.execute("SET join_collapse_limit = 1")   # preserve written join order

#     # --- execute each candidate to get ground-truth times ------------------
#     print(f"scenario: {args.scenario} | candidates: {len(candidates)} | reps: {args.reps}")
#     measured = []
#     for c in candidates:
#         t, c_pg = exec_stats(cur, c["sql"](pred_sqls), args.reps)
#         measured.append((c, t, c_pg))
#         print(f"  measured {c['name']:<26} {t:>10.1f} ms  (PG cost {c_pg:,.0f})")

#     t_default, _ = exec_stats(cur, f"SELECT * {from_def}{where_def}", args.reps)
#     cur.execute("SET join_collapse_limit = 8")
#     print(f"  PG default planner        {t_default:>10.1f} ms\n")

#     # --- score candidates with each model ----------------------------------
#     times       = np.array([m[1] for m in measured])
#     pg_costs    = np.array([m[2] for m in measured])
#     fastest_idx = int(np.argmin(times))
#     csv_rows, summary = [], []

#     for n in names:
#         model = CardModel(n)
#         print(f"=== model: {n} ({model.target}) ===")
#         print(f"{'join order':<28}{'ML rows':>14}{'med q-err':>10}"
#               f"{'PG cost':>12}{'exec time':>12}")
#         rows = []
#         for c, t, c_pg in measured:
#             cost_ml = ml_cost(model, cur, c["steps"], preds)
#             med_q   = subplan_median_qerror(model, cur, c["steps"], preds)
#             rows.append((c["name"], cost_ml, med_q, c_pg, t))
#             print(f"{c['name']:<28}{cost_ml:>14.0f}{med_q:>10.2f}"
#                   f"{c_pg:>12.1f}{t:>10.1f} ms")
#             csv_rows.append([args.scenario, n, c["name"], round(cost_ml, 1),
#                              round(med_q, 2), round(c_pg, 1), round(t, 1),
#                              "", "", "", "", ""])

#         ml_idx  = int(np.argmin([r[1] for r in rows]))
#         regret  = times[ml_idx] / times[fastest_idx]
#         speedup = t_default / times[ml_idx]
#         correct = (ml_idx == fastest_idx)
#         rho_ml  = spearmanr([r[1] for r in rows], times).statistic \
#             if len(rows) >= 3 else float("nan")
#         rho_pg  = spearmanr(pg_costs, times).statistic \
#             if len(rows) >= 3 else float("nan")

#         print(f"  ML chose   : {rows[ml_idx][0].strip():<24} -> {times[ml_idx]:.1f} ms")
#         print(f"  Actual best: {rows[fastest_idx][0].strip():<24} -> {times[fastest_idx]:.1f} ms")
#         print(f"  PG default : {t_default:.1f} ms")
#         print(f"  PLAN-CHOICE: ranking correct = {'YES' if correct else 'NO'} | "
#               f"regret = {regret:.2f}x | speedup vs PG default = {speedup:.2f}x")
#         print(f"  COST MODEL : rho(ML rows, time) = {rho_ml:+.2f} | "
#               f"rho(PG cost, time) = {rho_pg:+.2f}")
#         print(f"  ESTIMATION : median sub-plan q-error = "
#               f"{np.median([r[2] for r in rows]):.2f}\n")

#         csv_rows.append([args.scenario, n, "_summary", "", "", "",
#                          round(times[ml_idx], 1), correct, round(regret, 2),
#                          round(speedup, 2), round(rho_ml, 2), round(rho_pg, 2)])
#         summary.append((n, correct, regret, speedup, rho_ml, rho_pg))

#     print("=== model comparison (this scenario) ===")
#     print(f"{'model':<8}{'rank OK':>9}{'regret':>9}{'speedup':>9}"
#           f"{'rho ML':>9}{'rho PG':>9}")
#     for n, correct, regret, speedup, rho_ml, rho_pg in summary:
#         print(f"{n:<8}{str(correct):>9}{regret:>8.2f}x{speedup:>8.2f}x"
#               f"{rho_ml:>+9.2f}{rho_pg:>+9.2f}")
#     print("\nHow to read: regret 1.00x + speedup > 1 + rho near +1 = the model "
#           "genuinely optimizes queries. regret >> 1 or rho <= 0 = it does not (yet).")

#     conn.close()
#     OUT.parent.mkdir(parents=True, exist_ok=True)
#     write_header = (not OUT.exists()) or OUT.stat().st_size == 0
#     with open(OUT, "a", newline="") as f:
#         w = csv.writer(f)
#         if write_header:
#             w.writerow(["scenario", "model", "candidate", "ml_rows", "med_qerr",
#                         "pg_cost", "exec_ms", "ranking_correct", "regret",
#                         "speedup_vs_pg", "rho_ml", "rho_pg"])
#         w.writerows(csv_rows)
#     print(f"Saved -> {OUT}")


# if __name__ == "__main__":
#     main()









"""Join-order / plan-choice evaluation.

WHAT IT MEASURES - the four layers of query-optimization evaluation:
  1. ESTIMATION QUALITY : median q-error of the model's predicted sub-plan
                          cardinalities vs real COUNT(*)   (are the numbers right?)
  2. COST-MODEL QUALITY : Spearman rho between a cost score and real exec time
                          across candidates                (does lower score mean faster?)
  3. PLAN-CHOICE QUALITY: is the cheapest candidate actually the fastest?
                          regret = t(chosen) / t(fastest)  (how costly is a wrong pick?)
  4. END-TO-END IMPACT  : chosen plan's exec time vs PostgreSQL's default plan
                          speedup = t_default / t_chosen   (the bottom line)

USAGE EXAMPLES:
  # predefined scenarios
  python src/evaluate_join_order_quality.py --scenario q3
  python src/evaluate_join_order_quality.py --scenario q3 --reps 10
  python src/evaluate_join_order_quality.py --scenario q3-ordersonly --reps 1

  # ad-hoc query: any tables/predicates without changing the code
  python src/evaluate_join_order_quality.py --scenario custom \\
      --tables customer orders lineitem \\
      --pred "c_mktsegment = 'AUTOMOBILE'" \\
      --pred "o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1997-12-31'"

  --tables  : 2 or 3 of: customer orders lineitem
  --pred    : raw SQL WHERE fragment (repeat for multiple predicates)
              no need to normalize; featurize() handles that
"""
import argparse
import csv
import statistics
from datetime import date
from itertools import permutations
from pathlib import Path

import numpy as np
import psycopg2
from scipy.stats import spearmanr

from config import DB_CONFIG
# from evaluate_tpch import eq_cat, rng_pred
from evaluate_tpch import eq_cat, eq_num, rng_pred
from gen_training_data import JOIN_EDGES, build_where, featurize, load_col_stats
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "results" / "join_order_quality_eval.csv"

TABLES_3 = ["customer", "orders", "lineitem"]


# ───────────────────────── SQL helpers ──────────────────────────────────────

def join_clause(t, prefix):
    for a, ac, b, bc in JOIN_EDGES:
        if a == t and b in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
        if b == t and a in prefix:
            return f"JOIN {t} ON {a}.{ac} = {b}.{bc}"
    raise ValueError(f"no join edge for {t} given prefix {prefix}")


def build_ordered_sql(seq, pred_sqls):
    """SELECT * with explicit join order; returns full runnable SQL string."""
    from_part  = f"FROM {seq[0]} " + " ".join(
        join_clause(t, seq[:i]) for i, t in enumerate(seq[1:], start=1))
    where_part = (" WHERE " + " AND ".join(pred_sqls)) if pred_sqls else ""
    return "SELECT * " + from_part + where_part


def extract_join_order(node, acc):
    if node.get("Relation Name"):
        acc.append(node["Relation Name"])
    for ch in node.get("Plans", []):
        extract_join_order(ch, acc)


# ───────────────────────── scenario definitions ──────────────────────────────

def preds_q3(stats):
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        eq_cat(stats,  "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders",  "o_orderdate",
                 omin, date(1995, 3, 15).toordinal()),
    ]

# def candidates_q3():
#     return [
#         {"name": "left-deep (c-o-l)",
#          "seq":  ["customer", "orders", "lineitem"],
#          "steps":[["customer","orders"], TABLES_3]},
#         {"name": "left-deep (o-l-c)",
#          "seq":  ["orders", "lineitem", "customer"],
#          "steps":[["orders","lineitem"], TABLES_3]},
#         {"name": "left-deep (l-o-c)",
#          "seq":  ["lineitem", "orders", "customer"],
#          "steps":[["lineitem","orders"], TABLES_3]},
#     ]

# def candidates_q3():
#     return [
#         {"name": "customer-orders-lineitem",
#          "seq":  ["customer", "orders", "lineitem"],
#          "steps":[["customer", "orders"], TABLES_3]},
#         {"name": "orders-lineitem-customer",
#          "seq":  ["orders", "lineitem", "customer"],
#          "steps":[["orders", "lineitem"], TABLES_3]},
#         {"name": "lineitem-orders-customer",
#          "seq":  ["lineitem", "orders", "customer"],
#          "steps":[["lineitem", "orders"], TABLES_3]},
#     ]

# def candidates_q3():
#     """All 6 permutations of customer / orders / lineitem."""
#     return [
#         {"name": "customer-orders-lineitem",
#          "seq":  ["customer", "orders", "lineitem"],
#          "steps": [["customer", "orders"], TABLES_3]},
#         {"name": "customer-lineitem-orders",
#          "seq":  ["customer", "lineitem", "orders"],
#          "steps": [["customer", "lineitem"], TABLES_3]},
#         {"name": "orders-customer-lineitem",
#          "seq":  ["orders", "customer", "lineitem"],
#          "steps": [["orders", "customer"], TABLES_3]},
#         {"name": "orders-lineitem-customer",
#          "seq":  ["orders", "lineitem", "customer"],
#          "steps": [["orders", "lineitem"], TABLES_3]},
#         {"name": "lineitem-customer-orders",
#          "seq":  ["lineitem", "customer", "orders"],
#          "steps": [["lineitem", "customer"], TABLES_3]},
#         {"name": "lineitem-orders-customer",
#          "seq":  ["lineitem", "orders", "customer"],
#          "steps": [["lineitem", "orders"], TABLES_3]},
#     ]

def candidates_q3():
    """4 valid join permutations — customer-lineitem direct join is impossible.
    
    Schema: customer -[c_custkey=o_custkey]- orders -[o_orderkey=l_orderkey]- lineitem
    customer and lineitem share no direct join key, so:
      customer-lineitem-orders  ✗  (lineitem has no edge to customer)
      lineitem-customer-orders  ✗  (customer has no edge to lineitem)
    are both invalid and excluded.
    """
    return [
        {"name": "customer-orders-lineitem",
         "seq":  ["customer", "orders", "lineitem"],
         "steps": [["customer", "orders"], TABLES_3]},
        {"name": "orders-customer-lineitem",
         "seq":  ["orders", "customer", "lineitem"],
         "steps": [["orders", "customer"], TABLES_3]},
        {"name": "orders-lineitem-customer",
         "seq":  ["orders", "lineitem", "customer"],
         "steps": [["orders", "lineitem"], TABLES_3]},
        {"name": "lineitem-orders-customer",
         "seq":  ["lineitem", "orders", "customer"],
         "steps": [["lineitem", "orders"], TABLES_3]},
    ]

# My Custom
def preds_automobile_discount(stats):
    """AUTOMOBILE customers joined with orders and lineitem discount=0.10."""
    return [
        eq_cat(stats, "customer", "c_mktsegment", "AUTOMOBILE"),
        eq_num(stats, "lineitem", "l_discount",   0.10),
    ]

def candidates_automobile_discount():
    """4 valid join permutations for customer-orders-lineitem chain."""
    return [
        {"name": "customer-orders-lineitem",
         "seq":  ["customer", "orders", "lineitem"],
         "steps": [["customer", "orders"], TABLES_3]},
        {"name": "orders-customer-lineitem",
         "seq":  ["orders", "customer", "lineitem"],
         "steps": [["orders", "customer"], TABLES_3]},
        {"name": "orders-lineitem-customer",
         "seq":  ["orders", "lineitem", "customer"],
         "steps": [["orders", "lineitem"], TABLES_3]},
        {"name": "lineitem-orders-customer",
         "seq":  ["lineitem", "orders", "customer"],
         "steps": [["lineitem", "orders"], TABLES_3]},
    ]

def preds_three_pred(stats):
    """One predicate per table — gives model maximum signal to distinguish
    join orders. Each sub-plan has different active predicates so
    featurize() produces distinct vectors for all 4 candidates."""
    return [
        eq_cat(stats,  "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders",  "o_orderdate",
                 date(1995, 1, 1).toordinal(),
                 date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_quantity",
                 stats[("lineitem", "l_quantity")]["min"], 10.0),
    ]

def candidates_3table():
    """4 valid join orderings for customer/orders/lineitem chain.
    Shared by ALL 3-table scenarios — never duplicated.
    """
    return [
        {"name": "customer-orders-lineitem",
         "seq":  ["customer", "orders", "lineitem"],
         "steps": [["customer", "orders"], TABLES_3]},
        {"name": "orders-customer-lineitem",
         "seq":  ["orders", "customer", "lineitem"],
         "steps": [["orders", "customer"], TABLES_3]},
        {"name": "orders-lineitem-customer",
         "seq":  ["orders", "lineitem", "customer"],
         "steps": [["orders", "lineitem"], TABLES_3]},
        {"name": "lineitem-orders-customer",
         "seq":  ["lineitem", "orders", "customer"],
         "steps": [["lineitem", "orders"], TABLES_3]},
    ]


def preds_orders_only(stats):
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        rng_pred(stats, "orders", "o_orderdate",
                 date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal()),
    ]

def candidates_orders_only():
    return [
        {"name": "orders-customer",
         "seq":  ["orders", "customer"],
         "steps":[["orders","customer"]]},
        {"name": "customer-orders",
         "seq":  ["customer", "orders"],
         "steps":[["customer","orders"]]},
        {"name": "orders-lineitem",
         "seq":  ["orders", "lineitem"],
         "steps":[["orders","lineitem"]]},
        {"name": "lineitem-orders",
         "seq":  ["lineitem", "orders"],
         "steps":[["lineitem","orders"]]},
    ]


SCENARIOS = {
    "q3": dict(
        default_tables=TABLES_3,
        preds_fn=preds_q3,
        candidates_fn=candidates_q3,
    ),
    "q3-ordersonly": dict(
        default_tables=["orders", "customer"],
        preds_fn=preds_orders_only,
        candidates_fn=candidates_orders_only,
    ),
    # "custom" is built dynamically from --tables / --pred CLI args
    "automobile-discount": dict(         # ← new
        default_tables=TABLES_3,
        preds_fn=preds_automobile_discount,
        candidates_fn=candidates_automobile_discount,
    ),

    # add to SCENARIOS:
    "three-pred": dict(
        default_tables=TABLES_3,
        preds_fn=preds_three_pred,
        candidates_fn=candidates_3table,
    ),
}


# ───────────────────────── evaluation helpers ────────────────────────────────

def pg_est_of(cur, from_cl, where_cl):
    cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_cl}{where_cl}")
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def qerror(p, a):
    p, a = max(p, 1), max(a, 1)
    return max(p / a, a / p)


def plan_parts(steps, preds):
    """Yield (sub_tables, from_cl, where_cl, x) for each sub-plan step."""
    for sub in steps:
        joins     = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        from_cl   = f"FROM {', '.join(sub)}"
        where_cl  = build_where(joins, [p[3] for p in sub_preds])
        yield sub, from_cl, where_cl, [featurize(sub, joins, sub_preds)]


def ml_cost(model, cur, steps, preds):
    total = 0.0
    for _, from_cl, where_cl, x in plan_parts(steps, preds):
        total += float(model.predict_rows(x, [pg_est_of(cur, from_cl, where_cl)])[0])
    return total


def subplan_median_qerror(model, cur, steps, preds):
    qs = []
    for _, from_cl, where_cl, x in plan_parts(steps, preds):
        pred = float(model.predict_rows(x, [pg_est_of(cur, from_cl, where_cl)])[0])
        cur.execute(f"SELECT COUNT(*) {from_cl}{where_cl}")
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

# join sequence Validator

def validate_candidates(candidates, join_edges):
    """Check every candidate's join sequence is valid given JOIN_EDGES.
    
    A permutation [A, B, C, ...] is valid only if each table at position i
    can join to at least one table already in the prefix [0..i-1] via a
    known join edge. Raises ValueError with a clear explanation if not.
    """
    edge_set = set()
    for a, _, b, _ in join_edges:
        edge_set.add((a, b))
        edge_set.add((b, a))   # edges are undirected

    invalid = []
    for c in candidates:
        seq = c["seq"]
        for i, t in enumerate(seq[1:], start=1):
            prefix = seq[:i]
            can_join = any((t, p) in edge_set for p in prefix)
            if not can_join:
                invalid.append(
                    f"  candidate '{c['name']}': table '{t}' at position {i} "
                    f"has no join edge to any of {prefix}.\n"
                    f"    known edges involving '{t}': "
                    f"{[e for e in edge_set if e[0] == t]}"
                )
                break   # report first bad step per candidate, not all

    if invalid:
        msg = (
            f"\n{'='*60}\n"
            f"INVALID JOIN PERMUTATION(S) DETECTED\n"
            f"{'='*60}\n"
            + "\n".join(invalid) +
            f"\n\nOnly tables connected by a direct join edge can be "
            f"adjacent in a join sequence.\n"
            f"Your schema join graph:\n"
            f"  customer -[c_custkey=o_custkey]- orders "
            f"-[o_orderkey=l_orderkey]- lineitem\n"
            f"Fix: remove the invalid candidate(s) listed above."
        )
        raise ValueError(msg)

    print(f"  [ok] all {len(candidates)} candidates passed join-edge validation.")


# ───────────────────────── custom scenario builder ───────────────────────────

def build_custom_scenario(tables, raw_pred_sqls):
    """
    Build a scenario dynamically from CLI --tables and --pred args.
    Generates ALL permutations of the given tables as candidates.
    Predicates are passed as raw SQL strings (no normalization needed for
    execution; featurize() gets empty preds since we can't parse raw SQL
    back into the internal tuple format — estimation quality won't be shown).
    """
    allowed = set(TABLES_3)
    for t in tables:
        if t not in allowed:
            raise ValueError(f"--tables: '{t}' not in {allowed}. "
                             f"Only customer/orders/lineitem are supported.")
    if len(tables) < 2:
        raise ValueError("--tables needs at least 2 tables.")

    def preds_fn(_stats):
        # raw SQL preds: return dummy internal tuples with empty featurize input
        # sql fragment is at index [3]; lo/hi set to 0 as placeholders
        return [((None, None), 0.0, 0.0, sql) for sql in raw_pred_sqls]

    def candidates_fn():
        cands = []
        for perm in permutations(tables):
            seq   = list(perm)
            # name  = "-".join(t[0] for t in seq)   # e.g. "c-o-l"
            # new — full table name
            name = "-".join(seq)                  # "customer-orders-lineitem"
            steps = [list(seq[:i+1]) for i in range(1, len(seq))]
            cands.append({"name": name, "seq": seq, "steps": steps})
        return cands

    return dict(
        default_tables=list(tables),
        preds_fn=preds_fn,
        candidates_fn=candidates_fn,
        raw_pred_sqls=raw_pred_sqls,   # for printing
        is_custom=True,
    )


# ───────────────────────── main ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--models",   nargs="*", default=None)
    ap.add_argument("--scenario", default="q3",
                    choices=list(SCENARIOS) + ["custom"])
    ap.add_argument("--reps",     type=int, default=3)
    # custom scenario args
    ap.add_argument("--tables", nargs="+", default=None,
                    metavar="TABLE",
                    help="tables for custom scenario: e.g. --tables customer orders lineitem")
    ap.add_argument("--pred", action="append", dest="preds", default=[],
                    metavar="SQL",
                    help="raw SQL predicate fragment (repeat for multiple): "
                         "--pred \"c_mktsegment = 'AUTOMOBILE'\"")
    args  = ap.parse_args()
    names = args.models or available_models()

    # ---- resolve scenario ----
    if args.scenario == "custom":
        if not args.tables:
            ap.error("--scenario custom requires --tables")
        scen = build_custom_scenario(args.tables, args.preds)
        print(f"Custom scenario: tables={args.tables}")
        if args.preds:
            for p in args.preds:
                print(f"  predicate: {p}")
    else:
        scen = SCENARIOS[args.scenario]

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur  = conn.cursor()
    stats = load_col_stats(cur)

    preds      = scen["preds_fn"](stats)
    pred_sqls  = [p[3] for p in preds]
    candidates = scen["candidates_fn"]()

    # ── validate all candidates before doing any DB work ──────────────
    validate_candidates(candidates, JOIN_EDGES)

    # ---- add PG's default join order as a candidate ----
    default_tables = scen["default_tables"]
    joins_all  = [e for e in JOIN_EDGES
                  if e[0] in default_tables and e[2] in default_tables]
    from_def   = f"FROM {', '.join(default_tables)}"
    where_def  = build_where(joins_all, pred_sqls)
    cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_def}{where_def}")
    order = []
    extract_join_order(cur.fetchone()[0][0]["Plan"], order)
    order = [t for t in order if t in default_tables]
    if len(order) == len(set(order)) and set(order) == set(default_tables):
        pg_seq = order
        candidates.append({
            "name":  "pg-default-order",
            "seq":   pg_seq,
            "steps": [pg_seq[:i+1] for i in range(1, len(pg_seq))],
        })
    else:
        pg_seq = None
        print(f"note: could not extract clean PG join order (got {order})")

    cur.execute("SET join_collapse_limit = 1")

    # ---- measure exec time + show SQL per candidate ----
    print(f"\nscenario: {args.scenario} | candidates: {len(candidates)} | reps: {args.reps}")
    print("─" * 80)
    measured = []
    for c in candidates:
        sql    = build_ordered_sql(c["seq"], pred_sqls)
        t, cpg = exec_stats(cur, sql, args.reps)
        measured.append((c, t, cpg, sql))
        # ── NEW: print the SQL for this candidate ──
        print(f"\n  candidate : {c['name']}")
        print(f"  SQL       : {sql}")
        print(f"  exec time : {t:,.1f} ms  |  PG cost: {cpg:,.0f}")

    t_default, _ = exec_stats(cur, f"SELECT * {from_def}{where_def}", args.reps)
    cur.execute("SET join_collapse_limit = 8")
    print(f"\n  PG default planner (free plan): {t_default:,.1f} ms")
    print(f"  SQL: SELECT * {from_def}{where_def}")
    print("─" * 80)

    # ---- score with each model ----
    times       = np.array([m[1] for m in measured])
    pg_costs    = np.array([m[2] for m in measured])
    fastest_idx = int(np.argmin(times))
    csv_rows, summary = [], []

    for n in names:
        model = CardModel(n)
        is_custom = scen.get("is_custom", False)
        print(f"\n=== model: {n} ({model.target}) ===")
        print(f"{'join order':<20} {'SQL':<50} {'ML rows':>10} {'med q-err':>10} "
              f"{'PG cost':>12} {'exec ms':>10}")
        print("─" * 120)
        rows = []
        for c, t, cpg, sql in measured:
            if is_custom:
                # can't featurize raw SQL preds — skip estimation, use pg_est as cost
                cost_ml = pg_est_of(cur, f"FROM {', '.join(c['seq'])}", where_def)
                med_q   = float("nan")
            else:
                cost_ml = ml_cost(model, cur, c["steps"], preds)
                med_q   = subplan_median_qerror(model, cur, c["steps"], preds)
            rows.append((c["name"], cost_ml, med_q, cpg, t, sql))
            # ── NEW: show sql inline in the results table ──
            short_sql = sql if len(sql) <= 50 else sql[:47] + "..."
            print(f"{c['name']:<20} {short_sql:<50} {cost_ml:>10.0f} "
                  f"{med_q:>10.2f} {cpg:>12.1f} {t:>10.1f} ms")

        ml_idx  = int(np.argmin([r[1] for r in rows]))
        regret  = times[ml_idx] / times[fastest_idx]
        speedup = t_default / times[ml_idx]
        correct = (ml_idx == fastest_idx)
        rho_ml  = spearmanr([r[1] for r in rows], times).statistic \
            if len(rows) >= 3 else float("nan")
        rho_pg  = spearmanr(pg_costs, times).statistic \
            if len(rows) >= 3 else float("nan")

        print()
        print(f"  ML chose   : {rows[ml_idx][0]:<24} SQL: {rows[ml_idx][5]}")
        print(f"               -> {times[ml_idx]:,.1f} ms")
        print(f"  Actual best: {rows[fastest_idx][0]:<24} SQL: {rows[fastest_idx][5]}")
        print(f"               -> {times[fastest_idx]:,.1f} ms")
        print(f"  PG default : {t_default:,.1f} ms")
        print(f"  PLAN-CHOICE: ranking correct = {'YES ✓' if correct else 'NO ✗'} | "
              f"regret = {regret:.2f}x | speedup vs PG default = {speedup:.2f}x")
        print(f"  COST MODEL : rho(ML rows, time) = {rho_ml:+.2f} | "
              f"rho(PG cost, time) = {rho_pg:+.2f}")
        if not is_custom:
            print(f"  ESTIMATION : median sub-plan q-error = "
                  f"{np.median([r[2] for r in rows]):.2f}")

        csv_rows.append([args.scenario, n, "_summary", "", "", "",
                         round(times[ml_idx], 1), correct, round(regret, 2),
                         round(speedup, 2), round(rho_ml, 2), round(rho_pg, 2)])
        for c, t, cpg, sql in measured:
            r = next(r for r in rows if r[0] == c["name"])
            csv_rows.append([args.scenario, n, c["name"], round(r[1], 1),
                             round(r[2], 2) if not is_custom else "",
                             round(cpg, 1), round(t, 1), "", "", "", "", ""])
        summary.append((n, correct, regret, speedup, rho_ml, rho_pg))

    print("\n" + "─" * 80)
    print("=== model comparison summary ===")
    print(f"{'model':<8} {'rank OK':>9} {'regret':>9} {'speedup':>9} "
          f"{'rho ML':>9} {'rho PG':>9}")
    for n, correct, regret, speedup, rho_ml, rho_pg in summary:
        print(f"{n:<8} {str(correct):>9} {regret:>8.2f}x {speedup:>8.2f}x "
              f"{rho_ml:>+9.2f} {rho_pg:>+9.2f}")
    print("\nHow to read: regret=1.00x + speedup>1 + rho≈+1 → model genuinely helps.")

    conn.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not OUT.exists()) or OUT.stat().st_size == 0
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scenario", "model", "candidate", "ml_rows", "med_qerr",
                        "pg_cost", "exec_ms", "ranking_correct", "regret",
                        "speedup_vs_pg", "rho_ml", "rho_pg"])
        w.writerows(csv_rows)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

