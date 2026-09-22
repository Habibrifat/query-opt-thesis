"""Join-order experiment: does a model's predicted cardinality, used to
pick a join order, produce a plan that's actually faster than PostgreSQL's
default planner? Also reports cost-model accuracy: does PostgreSQL's own
Total Cost estimate track real execution time on these candidates?

Replaces phase5_latency.py + phase5_latency_q3.py.

FIX vs previous version: --model used to run exactly ONE model per
invocation with no way to compare them, unlike evaluate_tpch.py which
loops every model automatically. Now --models defaults to every model
with a saved artifact (same pattern as evaluate_tpch.py), and writes one
combined CSV per scenario.

Usage:
  python src/eval_join_order.py --scenario 5table            # all available models
  python src/eval_join_order.py --scenario q3 --models rf xgb
"""
import argparse
import csv
import statistics
from datetime import date

import psycopg2
from scipy.stats import spearmanr

from config import DB_CONFIG
from evaluate_tpch import eq_cat, rng_pred
from gen_training_data import JOIN_EDGES, build_body, featurize, load_col_stats
from model_loader import CardModel, available_models, BASE

OUT_DIR = BASE / "results"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
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
    out = []
    for name, seq in seqs.items():
        steps = [seq[:i + 1] for i in range(1, len(seq))]
        out.append({"name": name, "steps": steps,
                    "sql": lambda pred_sqls, seq=seq: build_ordered_sql(seq, pred_sqls)})
    return out


def preds_q3(stats):
    smax = stats[("lineitem", "l_shipdate")]["max"]
    omin = stats[("orders", "o_orderdate")]["min"]
    return [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders", "o_orderdate", omin, date(1995, 3, 15).toordinal()),
        rng_pred(stats, "lineitem", "l_shipdate", date(1995, 3, 15).toordinal(), smax),
    ]


def candidates_q3():
    def right_deep_sql(pred_sqls):
        sql = ("SELECT * FROM customer JOIN "
               "(orders JOIN lineitem ON l_orderkey = o_orderkey) "
               "ON c_custkey = o_custkey")
        if pred_sqls:
            sql += " WHERE " + " AND ".join(pred_sqls)
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


TABLES_2 = ["customer", "orders"]
TABLES_4 = ["customer", "orders", "lineitem", "nation"]


def preds_2table(stats):
    return []  # no predicates - simplest possible case, 2 candidate orders only


def candidates_2table():
    seqs = {
        "cust-first (c-o)": ["customer", "orders"],
        "ord-first  (o-c)": ["orders", "customer"],
    }
    out = []
    for name, seq in seqs.items():
        steps = [seq[:i + 1] for i in range(1, len(seq))]
        out.append({"name": name, "steps": steps,
                    "sql": lambda pred_sqls, seq=seq: build_ordered_sql(seq, pred_sqls)})
    return out


def preds_q10(stats):
    # Same predicates as evaluate_tpch.py's "q10-core" - this is the exact
    # 4-table shape (customer+orders+lineitem+nation) that showed the worst
    # generalization gap (24x-180x q-error on all 4 models).
    return [
        eq_cat(stats, "lineitem", "l_returnflag", "R"),
        rng_pred(stats, "orders", "o_orderdate", date(1994, 8, 1).toordinal(),
                 date(1994, 11, 1).toordinal()),
    ]


def candidates_q10():
    # Valid connected orderings on the path nation-customer-orders-lineitem
    # (no direct nation<->lineitem edge, so every order must pass through
    # customer or orders to stay connected at each step)
    seqs = {
        "dim-first  (n-c-o-l)": ["nation", "customer", "orders", "lineitem"],
        "cust-first (c-n-o-l)": ["customer", "nation", "orders", "lineitem"],
        "cust-first2(c-o-l-n)": ["customer", "orders", "lineitem", "nation"],
        "fact-first (o-l-c-n)": ["orders", "lineitem", "customer", "nation"],
    }
    out = []
    for name, seq in seqs.items():
        steps = [seq[:i + 1] for i in range(1, len(seq))]
        out.append({"name": name, "steps": steps,
                    "sql": lambda pred_sqls, seq=seq: build_ordered_sql(seq, pred_sqls)})
    return out


SCENARIOS = {
    "5table": dict(default_tables=TABLES_5, preds_fn=preds_5table, candidates_fn=candidates_5table),
    "q3": dict(default_tables=TABLES_3, preds_fn=preds_q3, candidates_fn=candidates_q3),
    # NEW: extra complexity levels, added to answer "why only these two?"
    "2table": dict(default_tables=TABLES_2, preds_fn=preds_2table, candidates_fn=candidates_2table),
    "q10": dict(default_tables=TABLES_4, preds_fn=preds_q10, candidates_fn=candidates_q10),
}

# ---------------------------------------------------------------------------


def pg_est_of(cur, body):
    cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def ml_cost(model, cur, steps, preds):
    total = 0.0
    for sub in steps:
        joins = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        body = build_body(sub, joins, [p[4] for p in sub_preds])
        x = [featurize(sub, joins, sub_preds)]
        total += float(model.predict_rows(x, [pg_est_of(cur, body)])[0])
    return total


def exec_stats(cur, sql, reps=3):
    times, cost = [], None
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]
        times.append(plan["Execution Time"])
        cost = plan["Plan"]["Total Cost"]
    return statistics.median(times), cost


def run_one_model(cur, model, scen, pred_sqls, preds, candidates):
    cur.execute("SET join_collapse_limit = 1")  # preserve written join order
    results = []
    for c in candidates:
        cost_ml = ml_cost(model, cur, c["steps"], preds)
        t, cost_pg = exec_stats(cur, c["sql"](pred_sqls))
        results.append((c["name"], cost_ml, cost_pg, t))

    cur.execute("SET join_collapse_limit = 8")  # PostgreSQL default
    joins = [e for e in JOIN_EDGES if e[0] in scen["default_tables"] and e[2] in scen["default_tables"]]
    t_default, cost_default = exec_stats(cur, "SELECT * " + build_body(scen["default_tables"], joins, pred_sqls))

    best_ml = min(results, key=lambda r: r[1])
    best_pg = min(results, key=lambda r: r[2])
    fastest = min(results, key=lambda r: r[3])

    print(f"{'join order':<26}{'ML cost (rows)':>16}{'PG cost':>12}{'exec time':>12}")
    for name, cost_ml, cost_pg, t in results:
        print(f"{name:<26}{cost_ml:>16.0f}{cost_pg:>12.1f}{t:>10.1f} ms")
    print(f"{'PG default planner':<26}{'-':>16}{cost_default:>12.1f}{t_default:>10.1f} ms")
    print(f"ML chose (predicted rows) : {best_ml[0].strip():<24} ({best_ml[3]:.1f} ms)")
    print(f"PG cost chose (Total Cost): {best_pg[0].strip():<24} ({best_pg[3]:.1f} ms)")
    print(f"Actual fastest             : {fastest[0].strip():<24} ({fastest[3]:.1f} ms)")
    print(f"PG default planner         : {t_default:.1f} ms")
    ranking_correct = best_ml[0] == fastest[0]
    print(f"ML ranking correct         : {'YES' if ranking_correct else 'NO'}")

    rho_pg = rho_ml = None
    if len(results) >= 3:
        times_ = [r[3] for r in results]
        rho_pg, _ = spearmanr([r[2] for r in results], times_)
        rho_ml, _ = spearmanr([r[1] for r in results], times_)
        print(f"Cost-model accuracy: PG Total Cost rho={rho_pg:+.2f}  ML rows rho={rho_ml:+.2f}")
        if rho_ml is not None and rho_ml < 0:
            print("  WARNING: ML cost proxy is NEGATIVELY correlated with real time -"
                  " trusting it here would pick a WORSE plan than random.")

    return dict(
        ml_chosen=best_ml[0].strip(), ml_chosen_ms=best_ml[3],
        pg_cost_chosen=best_pg[0].strip(), pg_cost_chosen_ms=best_pg[3],
        actual_fastest=fastest[0].strip(), actual_fastest_ms=fastest[3],
        pg_default_ms=t_default, ranking_correct=ranking_correct,
        rho_pg_cost=rho_pg, rho_ml_cost=rho_ml,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                     help="which models to test (default: every model with a saved artifact)")
    ap.add_argument("--scenario", default="5table", choices=list(SCENARIOS))
    args = ap.parse_args()
    names = args.models or available_models()
    if not names:
        print("No trained models found in results/ — run the train_*.py scripts first.")
        return

    scen = SCENARIOS[args.scenario]
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    stats = load_col_stats(cur)
    preds = scen["preds_fn"](stats)
    pred_sqls = [p[4] for p in preds]

    summary = {}
    for name in names:
        model = CardModel(name)
        print(f"\n=== scenario={args.scenario}  model={model} ===")
        candidates = scen["candidates_fn"]()  # rebuild per model (fresh lambdas)
        summary[name] = run_one_model(cur, model, scen, pred_sqls, preds, candidates)
    conn.close()

    print(f"\n=== Summary: scenario={args.scenario} ===")
    print(f"{'model':<8}{'ML chose':<22}{'ranking ok':>12}{'ML ms':>10}{'fastest ms':>12}{'PG default ms':>15}")
    for name, r in summary.items():
        print(f"{name:<8}{r['ml_chosen']:<22}{('YES' if r['ranking_correct'] else 'NO'):>12}"
              f"{r['ml_chosen_ms']:>10.1f}{r['actual_fastest_ms']:>12.1f}{r['pg_default_ms']:>15.1f}")

    out_path = OUT_DIR / f"join_order_eval_{args.scenario}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ml_chosen", "ml_chosen_ms", "pg_cost_chosen", "pg_cost_chosen_ms",
                    "actual_fastest", "actual_fastest_ms", "pg_default_ms", "ranking_correct",
                    "rho_pg_cost", "rho_ml_cost"])
        for name, r in summary.items():
            w.writerow([name, r["ml_chosen"], r["ml_chosen_ms"], r["pg_cost_chosen"], r["pg_cost_chosen_ms"],
                        r["actual_fastest"], r["actual_fastest_ms"], r["pg_default_ms"], r["ranking_correct"],
                        r["rho_pg_cost"], r["rho_ml_cost"]])
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()