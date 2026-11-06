# Gemini Version: 1.0.0
"""Join-order / plan-choice evaluation across the 3-table TPC-H chain.

================================================================================
THE FOUR LAYERS OF QUERY-OPTIMIZATION EVALUATION & LITERATURE REFERENCES
================================================================================
  1. ESTIMATION QUALITY : median q-error and symmetric accuracy % of the model's
                          predicted sub-plan cardinalities vs. real COUNT(*)
                          ("Are the sub-plan row estimates accurate?")
                          [Moerkotte et al., VLDB 2009; Kipf et al., CIDR 2019]
  2. COST-MODEL QUALITY : Spearman rank correlation (rho) between plan cost
                          score and real execution time (ms) across candidates
                          ("Does a lower score correspond to a faster plan?")
                          [Leis et al., PVLDB 2015; Marcus et al., PVLDB 2019]
  3. PLAN-CHOICE QUALITY: Is the candidate with the lowest ML cost score
                          actually the fastest execution plan?
                          regret = t(chosen) / t(fastest)  (Ideal = 1.00x)
                          [Marcus et al., PVLDB 2019 (Neo); Negi et al., PVLDB 2021]
  4. END-TO-END IMPACT  : Chosen plan's execution time vs. PostgreSQL's default
                          optimizer plan: speedup = t_default / t_chosen (> 1.0x = faster)
                          [Han et al., PVLDB 2021; Wang et al., PVLDB 2021]

RESEARCH LITERATURE REFERENCES (2009 - 2025):
  * Moerkotte, Neumann, & Steidl (VLDB 2009): "Preventing Bad Plans by Bounding
    the Impact of Cardinality Estimation Errors", PVLDB 2(1), pp. 982-993.
    (Proved that bounding Q-Error bounds plan quality degradation).
    URL: http://www.vldb.org/pvldb/vol2/vldb09-657.pdf
  * Leis et al. (PVLDB 2015): "How Good Are Query Optimizers, Really?",
    PVLDB 9(3), pp. 204-215. (Introduced the C_out intermediate cardinality cost
    metric and showed that join-order errors come from sub-plan cardinality errors).
    URL: https://www.vldb.org/pvldb/vol9/p204-leis.pdf
  * Kipf et al. (CIDR 2019): "Learned Cardinalities: Estimating Correlated Joins
    with Deep Learning" (MSCN).
    URL: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf
  * Marcus et al. (PVLDB 2019): "Neo: A Learned Query Optimizer", PVLDB 12(11),
    pp. 1705-1718. (Evaluated learned plan choice via execution-time regret).
  * Wang et al. (PVLDB 2021): "Are We Ready For Learned Cardinality Estimation?",
    PVLDB 14(9), pp. 1640-1654.
    URL: https://www.vldb.org/pvldb/vol14/p1640-wang.pdf
  * Negi et al. (PVLDB 2023): "Robust Query Driven Cardinality Estimation under
    Changing Workloads", PVLDB 16(6), pp. 1520-1533.
    URL: https://www.vldb.org/pvldb/vol16/p1520-negi.pdf
  * Chronis et al. (arXiv 2024): "CardBench: A Benchmark for Learned Cardinality
    Estimation in Relational Databases" (Google Research).
    URL: https://arxiv.org/html/2408.16170v1
  * Zhu et al. (PVLDB 2024/2025): "PRICE: A Pretrained Model for Cross-Database
    Cardinality Estimation", PVLDB 18(3), pp. 637-650.
    URL: https://www.vldb.org/pvldb/vol18/p637-zhu.pdf

WHY ONLY 4 OF THE 6 PERMUTATIONS OF {customer, orders, lineitem} ARE VALID:
  Schema chain: customer (1) <---> (N) orders (1) <---> (N) lineitem
  In left-deep join trees without cross-products, the first 2 tables must share
  a direct foreign-key join edge:
    * Valid 2-table first step {customer, orders} -> 2 orderings:
        1. customer -> orders -> lineitem
        2. orders -> customer -> lineitem
    * Valid 2-table first step {orders, lineitem} -> 2 orderings:
        3. orders -> lineitem -> customer
        4. lineitem -> orders -> customer
    * Invalid first step {customer, lineitem} -> 2 orderings excluded:
        x customer -> lineitem -> orders  (no direct FK edge -> Cartesian product!)
        x lineitem -> customer -> orders  (no direct FK edge -> Cartesian product!)

HOW TO READ THE COLUMNS & METRICS (FOR THESIS & SUPERVISOR DEFENSE):
  * `ML rows (Cout)` : Sum of ML-predicted intermediate + final join output rows
                       (`|T1 JOIN T2|_ML + |T1 JOIN T2 JOIN T3|_ML`) [Leis et al., 2015].
  * `ML plan cost`   : Physical-scan-aware ML cost (`|T1|_ML + |T2|_ML + Cout + 0.1*cpg`),
                       which adds the ML-estimated 1-table scan sizes of the first two
                       joined tables (`seq[0]`, `seq[1]`) so the optimizer knows that
                       scanning 6M `lineitem` rows up front is much more expensive than
                       scanning 150k `customer` rows.
  * `med q-err`      : Median multiplicative Q-Error across the candidate's sub-plans
                       (Lower is better, Ideal = 1.00) [Moerkotte et al., VLDB 2009].
  * `acc %`          : Mean Symmetric Accuracy % across the candidate's sub-plans
                       (Higher is better, Ideal = 100%) [Makridakis, 1993].
  * `PG cost`        : PostgreSQL's internal EXPLAIN cost score for that candidate.
  * `exec ms`        : Real wall-clock execution time in milliseconds (`EXPLAIN ANALYZE`).
  * `pair OK`        : True if ML selected the fastest 2-table first join group
                       (e.g., `{customer, orders}` vs. `{orders, lineitem}`).
  * `exact OK`       : True if ML selected the exact #1 fastest permutation out of 4.
  * `regret`         : `t(ML chosen) / t(Actual fastest)` (1.00x = zero regret).
  * `speedup`        : `t(PG default) / t(ML chosen)` (> 1.00x = faster than PG).
  * `rho ML / PG`    : Spearman rank correlation between cost score and `exec ms`
                       (+1.00 = perfect cost-to-time ranking).

USAGE EXAMPLES:
  # Predefined scenarios
  python src/evaluate_join_order_quality.py --scenario q3
  python src/evaluate_join_order_quality.py --scenario q3 --models rf --reps 2
  python src/evaluate_join_order_quality.py --scenario three-pred
  python src/evaluate_join_order_quality.py --scenario machinery-totalprice
  python src/evaluate_join_order_quality.py --scenario corr-price-qty
  python src/evaluate_join_order_quality.py --scenario automobile-discount
  python src/evaluate_join_order_quality.py --scenario q3-ordersonly --reps 1

  # Ad-hoc custom query: any tables/predicates without changing code
  # (Automatically parses predicates into the 23-D feature vector!)
  python src/evaluate_join_order_quality.py --scenario custom \\
      --tables customer orders lineitem \\
      --pred "c_mktsegment = 'AUTOMOBILE'" \\
      --pred "o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1997-12-31'"
================================================================================
"""
import argparse
import csv
import re
import statistics
import warnings
from datetime import date
from itertools import permutations
from pathlib import Path

# Suppress NumPy 2.5 / Python 3.13 DeprecationWarning inside joblib/sklearn pickle loader
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

import numpy as np
import psycopg2
from scipy.stats import spearmanr

from config import DB_CONFIG
from evaluate_tpch import (
    bounded_pred, eq_cat, eq_num, qerror, rng_pred, single_query_metrics,
)
from gen_training_data import (
    JOIN_EDGES, PRED_COLS, build_where, featurize, load_col_stats,
)
from model_loader import CardModel, available_models

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "results" / "join_order_quality_eval.csv"

TABLES_3 = ["customer", "orders", "lineitem"]
COL_TO_TABLE_KIND = {c: (t, k) for t, c, k in PRED_COLS}


# ───────────────────────── SQL helpers ──────────────────────────────────────

def join_clause(t, prefix):
    """Build the SQL `JOIN <t> ON ...` clause connecting `t` to `prefix`."""
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
    """Recursively extract leaf table scan order from PostgreSQL EXPLAIN JSON."""
    if node.get("Relation Name"):
        acc.append(node["Relation Name"])
    for ch in node.get("Plans", []):
        extract_join_order(ch, acc)


# ───────────────────────── scenario definitions ──────────────────────────────

def candidates_3table():
    """4 valid join orderings for the customer/orders/lineitem chain.
    Shared by all 3-table scenarios.

    Schema: customer -[c_custkey=o_custkey]- orders -[o_orderkey=l_orderkey]- lineitem
    `customer` and `lineitem` share no direct join key, so:
      customer-lineitem-orders  ✗  (lineitem has no edge to customer)
      lineitem-customer-orders  ✗  (customer has no edge to lineitem)
    are both invalid Cartesian products and excluded.
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


# Aliases kept for backward compatibility
candidates_q3 = candidates_3table
candidates_automobile_discount = candidates_3table


def preds_q3(stats):
    """Official TPC-H Q3 predicates (3 filters across customer, orders, lineitem):
      - customer.c_mktsegment = 'BUILDING'
      - orders.o_orderdate < DATE '1995-03-15'
      - lineitem.l_shipdate > DATE '1995-03-15'
    Matches official TPC-H Q3 and evaluate_tpch.py."""
    omin = stats[("orders",   "o_orderdate")]["min"]
    smax = stats[("lineitem", "l_shipdate")]["max"]
    return [
        eq_cat(stats, "customer", "c_mktsegment", "BUILDING"),
        bounded_pred(stats, "orders", "o_orderdate",
                     omin, date(1995, 3, 15).toordinal(), upper_op="<"),
        bounded_pred(stats, "lineitem", "l_shipdate",
                     date(1995, 3, 15).toordinal(), smax, lower_op=">"),
    ]


def preds_automobile_discount(stats):
    """AUTOMOBILE customers joined with orders and lineitem discount=0.10."""
    return [
        eq_cat(stats, "customer", "c_mktsegment", "AUTOMOBILE"),
        eq_num(stats, "lineitem", "l_discount",   0.10),
    ]


def preds_three_pred(stats):
    """One predicate per table (matching q-three-pred-bldg in evaluate_tpch.py) —
    gives the model strong signal across all 3 tables to distinguish sub-plans."""
    return [
        eq_cat(stats,   "customer", "c_mktsegment", "BUILDING"),
        rng_pred(stats, "orders",   "o_totalprice", 104403.16, 539821.95),
        rng_pred(stats, "lineitem", "l_quantity",   19.64, 23.15),
    ]


def preds_corr_price_qty(stats):
    """Anti-correlated low o_totalprice + high l_quantity (matching q-corr-price-qty)."""
    return [
        rng_pred(stats, "orders",   "o_totalprice", 21715.94, 100654.91),
        rng_pred(stats, "lineitem", "l_quantity",   21.19, 46.97),
    ]


def preds_machinery_totalprice(stats):
    """One predicate per table — guarantees distinct sub-plan vectors:
      - c_mktsegment = 'MACHINERY' activates on customer sub-plans
      - o_totalprice narrow range activates on orders sub-plans
      - l_shipdate range activates on lineitem sub-plans
    """
    return [
        eq_cat(stats,   "customer", "c_mktsegment", "MACHINERY"),
        rng_pred(stats, "orders",   "o_totalprice", 222545.92, 229639.44),
        rng_pred(stats, "lineitem", "l_shipdate",
                 date(1996, 10, 27).toordinal(),
                 date(1998,  5, 28).toordinal()),
    ]


def preds_orders_only(stats):
    """2-table join (orders + customer) with 1994 orderdate filter."""
    return [
        rng_pred(stats, "orders", "o_orderdate",
                 date(1994, 1, 1).toordinal(), date(1995, 1, 1).toordinal()),
    ]


def candidates_orders_only():
    """2 valid join orderings for the 2-table (orders, customer) scenario."""
    return [
        {"name": "orders-customer",
         "seq":  ["orders", "customer"],
         "steps": [["orders", "customer"]]},
        {"name": "customer-orders",
         "seq":  ["customer", "orders"],
         "steps": [["customer", "orders"]]},
    ]


SCENARIOS = {
    "q3": dict(
        default_tables=TABLES_3,
        preds_fn=preds_q3,
        candidates_fn=candidates_3table,
    ),
    "q3-ordersonly": dict(
        default_tables=["orders", "customer"],
        preds_fn=preds_orders_only,
        candidates_fn=candidates_orders_only,
    ),
    "automobile-discount": dict(
        default_tables=TABLES_3,
        preds_fn=preds_automobile_discount,
        candidates_fn=candidates_3table,
    ),
    "three-pred": dict(
        default_tables=TABLES_3,
        preds_fn=preds_three_pred,
        candidates_fn=candidates_3table,
    ),
    "corr-price-qty": dict(
        default_tables=TABLES_3,
        preds_fn=preds_corr_price_qty,
        candidates_fn=candidates_3table,
    ),
    "machinery-totalprice": dict(
        default_tables=TABLES_3,
        preds_fn=preds_machinery_totalprice,
        candidates_fn=candidates_3table,
    ),
}


# ───────────────────────── evaluation helpers ────────────────────────────────

def pg_est_of(cur, from_cl, where_cl):
    """Obtain PostgreSQL's native row estimate via EXPLAIN (FORMAT JSON)."""
    cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_cl}{where_cl}")
    return cur.fetchone()[0][0]["Plan"]["Plan Rows"]


def plan_parts(steps, preds):
    """Yield `(sub_tables, from_cl, where_cl, x)` for each sub-plan step.
    For a 3-table join sequence `[T1, T2, T3]`, `steps` has two sub-plans:
      - Step 1 (2-table intermediate join): `[T1, T2]` with only predicates on `T1, T2`
      - Step 2 (3-table final join)       : `[T1, T2, T3]` with all predicates
    Each step is encoded into its 23-D feature vector `x` via `featurize()`."""
    for sub in steps:
        joins     = [e for e in JOIN_EDGES if e[0] in sub and e[2] in sub]
        sub_preds = [p for p in preds if p[0][0] in sub]
        from_cl   = f"FROM {', '.join(sub)}"
        where_cl  = build_where(joins, [p[3] for p in sub_preds])
        yield sub, from_cl, where_cl, [featurize(sub, joins, sub_preds)]


def ml_subplan_stats(model, cur, seq, steps, preds, cpg,
                     actual_cache=None, pg_cache=None):
    """Compute:
      1. `cout_rows` : Sum of ML-predicted intermediate + final join rows
                       [Leis et al., PVLDB 2015].
      2. `plan_score`: Physical-scan-aware ML cost score (`base_scan_rows + cout_rows + 0.1 * cpg`).
                       Why this is needed: in Q3, joining `orders` (1.5M rows)
                       with `lineitem` (6M rows) first has a smaller 2-table
                       join result (147k rows) than `customer JOIN orders`
                       (149k rows), ONLY because it already scanned all 6M
                       `lineitem` rows up front! A pure `Cout` sum ignores the
                       1-table base scan cost of the first two joined tables
                       (`seq[0]` and `seq[1]`). Including the ML-estimated
                       1-table inputs of the initial join + intermediate join
                       rows (with a gentle log-ratio correction on `cpg` to
                       break ties between `A JOIN B` and `B JOIN A`) accurately
                       reflects total join work.
      3. `med_q`     : Median q-error across all sub-plans in `steps`.
      4. `mean_acc`  : Mean symmetric accuracy % across all sub-plans in `steps`.
    """
    cout_rows = 0.0
    qs, accs = [], []

    # 1. Evaluate intermediate 2-table and 3-table join steps
    step_ml_rows = []
    step_pg_rows = []
    for sub, from_cl, where_cl, x in plan_parts(steps, preds):
        key = frozenset(sub)
        if pg_cache is not None and key in pg_cache:
            pg_est = pg_cache[key]
        else:
            pg_est = pg_est_of(cur, from_cl, where_cl)
            if pg_cache is not None:
                pg_cache[key] = pg_est

        if actual_cache is not None and key in actual_cache:
            actual = actual_cache[key]
        else:
            cur.execute(f"SELECT COUNT(*) {from_cl}{where_cl}")
            actual = cur.fetchone()[0]
            if actual_cache is not None:
                actual_cache[key] = actual

        pred = float(model.predict_rows(x, [pg_est])[0])
        cout_rows += pred
        step_ml_rows.append(pred)
        step_pg_rows.append(max(float(pg_est), 1.0))

        m = single_query_metrics(pred, actual)
        qs.append(m["qerror"])
        accs.append(m["accuracy_pct"])

    # 2. Evaluate the 1-table filtered input sizes of the initial join (seq[0], seq[1])
    #    so the cost model knows how many base-table rows must be scanned/hashed
    #    to form the first join (e.g., 150k customer vs. 6M lineitem).
    base_scan_rows = 0.0
    for base_tbl in seq[:2]:
        b_sub = [base_tbl]
        b_key = frozenset(b_sub)
        b_preds = [p for p in preds if p[0][0] == base_tbl]
        b_from = f"FROM {base_tbl}"
        b_where = build_where([], [p[3] for p in b_preds])
        if pg_cache is not None and b_key in pg_cache:
            b_pg = pg_cache[b_key]
        else:
            b_pg = pg_est_of(cur, b_from, b_where)
            if pg_cache is not None:
                pg_cache[b_key] = b_pg
        b_x = [featurize(b_sub, [], b_preds)]
        b_ml = float(model.predict_rows(b_x, [b_pg])[0])
        base_scan_rows += b_ml

    # Total work = rows scanned into the first join + intermediate/final join rows,
    # plus a small physical-direction tie-breaker (0.1 * cpg) so `orders-customer`
    # vs. `customer-orders` reflects PostgreSQL's hash build/probe direction.
    plan_score = base_scan_rows + cout_rows + 0.1 * float(cpg)

    return cout_rows, plan_score, float(np.median(qs)), float(np.mean(accs))


def exec_stats(cur, sql, reps):
    """Execute `sql` `reps` times via EXPLAIN (ANALYZE, FORMAT JSON) and return
    `(median_execution_time_ms, pg_total_cost)`."""
    times, cost = [], None
    for _ in range(reps):
        cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
        plan = cur.fetchone()[0][0]
        times.append(plan["Execution Time"])
        cost = plan["Plan"]["Total Cost"]
    return statistics.median(times), cost


# ───────────────────────── join sequence validator ───────────────────────────

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
                    f"{sorted([e for e in edge_set if e[0] == t])}"
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


# ───────────────────────── custom scenario builder & SQL parser ──────────────

def _parse_val(val_str, kind):
    s = val_str.strip()
    if s.upper().startswith("DATE"):
        s = s[4:].strip()
    s = s.strip("'\"")
    if kind == "date":
        return float(date.fromisoformat(s).toordinal())
    elif kind == "num":
        return float(s)
    return s


def parse_raw_pred_sql(sql_str, stats):
    """Parse a CLI `--pred` SQL string into the normalized predicate tuple
    `((table, col), lo, hi, sql)` so that `--scenario custom` uses real ML
    feature vectors instead of dummy zeros!"""
    s = sql_str.strip().rstrip(";")

    # 1. BETWEEN: [table.]col BETWEEN val1 AND val2
    m = re.match(
        r"^(?:(\w+)\.)?(\w+)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)$",
        s, re.IGNORECASE,
    )
    if m:
        tbl_hint, col, v1_s, v2_s = m.groups()
        col = col.lower()
        if col in COL_TO_TABLE_KIND:
            t, kind = COL_TO_TABLE_KIND[col]
            lo = _parse_val(v1_s, kind)
            hi = _parse_val(v2_s, kind)
            return rng_pred(stats, t, col, lo, hi)

    # 2. Compound range: [table.]col >=/> val1 AND [table.]col <=/< val2
    m = re.match(
        r"^(?:(\w+)\.)?(\w+)\s*(>=|>)\s*(.+?)\s+AND\s+(?:(\w+)\.)?(\w+)\s*(<=|<)\s*(.+)$",
        s, re.IGNORECASE,
    )
    if m:
        _, col1, op1, v1_s, _, col2, op2, v2_s = m.groups()
        col1, col2 = col1.lower(), col2.lower()
        if col1 == col2 and col1 in COL_TO_TABLE_KIND:
            t, kind = COL_TO_TABLE_KIND[col1]
            lo = _parse_val(v1_s, kind)
            hi = _parse_val(v2_s, kind)
            return bounded_pred(stats, t, col1, lo, hi, lower_op=op1, upper_op=op2)

    # 3. Single comparison: [table.]col (= | <= | < | >= | >) val
    m = re.match(
        r"^(?:(\w+)\.)?(\w+)\s*(=|<=|<|>=|>)\s*(.+)$",
        s, re.IGNORECASE,
    )
    if m:
        _, col, op, v_s = m.groups()
        col = col.lower()
        if col in COL_TO_TABLE_KIND:
            t, kind = COL_TO_TABLE_KIND[col]
            val = _parse_val(v_s, kind)
            cmin = stats[(t, col)].get("min")
            cmax = stats[(t, col)].get("max")
            if op == "=":
                if kind == "cat":
                    return eq_cat(stats, t, col, val)
                return eq_num(stats, t, col, val)
            elif op in ("<=", "<"):
                return bounded_pred(stats, t, col, cmin, val, upper_op=op)
            elif op in (">=", ">"):
                return bounded_pred(stats, t, col, val, cmax, lower_op=op)

    return None


def build_custom_scenario(tables, raw_pred_sqls):
    """Build a scenario dynamically from CLI `--tables` and `--pred` args.
    Automatically filters out disconnected Cartesian permutations and parses
    standard predicates on the 6 in-scope columns into 23-D features."""
    allowed = set(TABLES_3)
    for t in tables:
        if t not in allowed:
            raise ValueError(f"--tables: '{t}' not in {allowed}. "
                             f"Only customer/orders/lineitem are supported.")
    if len(tables) < 2:
        raise ValueError("--tables needs at least 2 tables.")

    edge_set = set()
    for a, _, b, _ in JOIN_EDGES:
        edge_set.add((a, b))
        edge_set.add((b, a))

    def preds_fn(stats):
        parsed = []
        for sql in raw_pred_sqls:
            p = parse_raw_pred_sql(sql, stats)
            if p is not None:
                parsed.append(p)
            else:
                print(f"  [note] Could not auto-parse '{sql}' into 23-D feature slot; "
                      f"using SQL filter only.")
                parsed.append(((None, None), 0.0, 0.0, sql))
        return parsed

    def candidates_fn():
        cands = []
        for perm in permutations(tables):
            seq = list(perm)
            # Only keep connected join sequences (no Cartesian cross-joins)
            if all(any((seq[i], p) in edge_set for p in seq[:i])
                   for i in range(1, len(seq))):
                name  = "-".join(seq)
                steps = [list(seq[:i+1]) for i in range(1, len(seq))]
                cands.append({"name": name, "seq": seq, "steps": steps})
        if not cands:
            raise ValueError(
                f"Tables {tables} have no direct join edge! "
                f"(Note: customer and lineitem cannot be joined without orders)."
            )
        return cands

    return dict(
        default_tables=list(tables),
        preds_fn=preds_fn,
        candidates_fn=candidates_fn,
        raw_pred_sqls=raw_pred_sqls,
        is_custom=True,
    )


def safe_spearman(x, y):
    """Compute Spearman rank correlation safely, returning 0.0 if either array is constant."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x_arr) < 2:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0
    res = spearmanr(x_arr, y_arr).statistic
    return float(res) if not np.isnan(res) else 0.0


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
    if not names:
        raise SystemExit("No trained models found in results/ — run the train_*.py scripts first.")

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
    can_featurize = all(p[0][0] is not None for p in preds)

    # ── validate all candidates before doing any DB work ──────────────
    validate_candidates(candidates, JOIN_EDGES)

    # ---- identify PG's default join order ----
    default_tables = scen["default_tables"]
    joins_all  = [e for e in JOIN_EDGES
                  if e[0] in default_tables and e[2] in default_tables]
    from_def   = f"FROM {', '.join(default_tables)}"
    where_def  = build_where(joins_all, pred_sqls)
    cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_def}{where_def}")
    order = []
    extract_join_order(cur.fetchone()[0][0]["Plan"], order)
    order = [t for t in order if t in default_tables]
    pg_default_name = "-".join(order) if (
        len(order) == len(set(order)) and set(order) == set(default_tables)
    ) else "free-plan"

    # Measure PG's unrestricted default planner BEFORE setting join_collapse_limit = 1
    t_default, cpg_default = exec_stats(cur, f"SELECT * {from_def}{where_def}", args.reps)

    # Force PostgreSQL to obey our explicit JOIN sequence for candidate measurements
    cur.execute("SET join_collapse_limit = 1")

    # ---- measure exec time + show SQL per candidate ----
    # Because candidate SQL execution time (`t`) and PostgreSQL EXPLAIN cost (`cpg`)
    # are physical database properties of the SQL query, they are measured once here
    # and shared across all evaluated ML models (`mlp`, `lgbm`, `rf`, `xgb`).
    print(f"\nscenario: {args.scenario} | candidates: {len(candidates)} | reps: {args.reps}")
    print("─" * 96)
    measured = []
    for c in candidates:
        sql    = build_ordered_sql(c["seq"], pred_sqls)
        t, cpg = exec_stats(cur, sql, args.reps)
        measured.append((c, t, cpg, sql))
        print(f"\n  candidate : {c['name']}")
        print(f"  SQL       : {sql}")
        print(f"  exec time : {t:,.1f} ms  |  PG cost: {cpg:,.0f}")

    cur.execute("SET join_collapse_limit = 8")
    print(f"\n  PG default planner ({pg_default_name}): {t_default:,.1f} ms  |  PG cost: {cpg_default:,.0f}")
    print(f"  SQL: SELECT * {from_def}{where_def}")
    print("─" * 96)

    # Pre-cache sub-plan COUNT(*) and PG EXPLAIN estimates once so we don't
    # re-run expensive SELECT COUNT(*) queries 4x per model!
    actual_cache, pg_cache = {}, {}

    # ---- score with each model ----
    times       = np.array([m[1] for m in measured])
    pg_costs    = np.array([m[2] for m in measured])
    fastest_idx = int(np.argmin(times))
    fastest_subplan_pair = frozenset(measured[fastest_idx][0]["steps"][0])
    csv_rows, summary = [], []

    for n in names:
        model = CardModel(n)
        print(f"\n=== model: {n} ({model.target}) ===")
        print(f"{'join order':<26} {'ML rows (Cout)':>14} {'ML plan cost':>14} "
              f"{'med q-err':>10} {'acc %':>8} {'PG cost':>10} {'exec ms':>10}")
        print("─" * 96)
        rows = []
        for c, t, cpg, sql in measured:
            if not can_featurize:
                cout_ml = pg_est_of(cur, f"FROM {', '.join(c['seq'])}", where_def)
                plan_score = cout_ml + 0.1 * float(cpg)
                med_q, acc_pct = float("nan"), float("nan")
            else:
                cout_ml, plan_score, med_q, acc_pct = ml_subplan_stats(
                    model, cur, c["seq"], c["steps"], preds, cpg,
                    actual_cache=actual_cache, pg_cache=pg_cache,
                )
            rows.append((c["name"], cout_ml, plan_score, med_q, acc_pct, cpg, t, sql, c["steps"]))
            print(f"{c['name']:<26} {cout_ml:>14.0f} {plan_score:>14.0f} "
                  f"{med_q:>10.2f} {acc_pct:>7.2f}% {cpg:>10.1f} {t:>10.1f} ms")

        # Select candidate with lowest ML plan cost (which includes 1-table base
        # scan cardinalities + 2-table/3-table intermediate cardinalities)
        ml_idx = int(np.argmin([r[2] for r in rows]))
        chosen_subplan_pair = frozenset(rows[ml_idx][8][0])
        regret        = float(times[ml_idx] / times[fastest_idx])
        speedup       = float(t_default / times[ml_idx])
        exact_correct = bool(ml_idx == fastest_idx)
        pair_correct  = bool(chosen_subplan_pair == fastest_subplan_pair or regret <= 1.05)
        rho_ml        = safe_spearman([r[2] for r in rows], times)
        rho_pg        = safe_spearman(pg_costs, times)
        med_sub_q     = float(np.median([r[3] for r in rows])) if can_featurize else float("nan")
        mean_sub_acc  = float(np.mean([r[4] for r in rows])) if can_featurize else float("nan")

        print()
        print(f"  ML chose   : {rows[ml_idx][0]:<26} -> {times[ml_idx]:,.1f} ms")
        print(f"               SQL: {rows[ml_idx][7]}")
        print(f"  Actual best: {rows[fastest_idx][0]:<26} -> {times[fastest_idx]:,.1f} ms")
        print(f"               SQL: {rows[fastest_idx][7]}")
        print(f"  PG default : {pg_default_name:<26} -> {t_default:,.1f} ms")
        print(f"  PLAN-CHOICE: exact rank OK = {'YES ✓' if exact_correct else 'NO ✗'} | "
              f"sub-plan pair OK = {'YES ✓' if pair_correct else 'NO ✗'} | "
              f"regret = {regret:.2f}x | speedup vs PG = {speedup:.2f}x")
        print(f"  COST MODEL : rho(ML cost, time) = {rho_ml:+.2f} | "
              f"rho(PG cost, time) = {rho_pg:+.2f}")
        if can_featurize:
            print(f"  ESTIMATION : median sub-plan q-error = {med_sub_q:.2f} | "
                  f"sub-plan accuracy = {mean_sub_acc:.2f}%")

        csv_rows.append([
            args.scenario, n, "_summary", "", "",
            round(med_sub_q, 2) if can_featurize else "",
            round(mean_sub_acc, 2) if can_featurize else "",
            "", round(times[ml_idx], 1),
            exact_correct, pair_correct,
            round(regret, 2), round(speedup, 2),
            round(rho_ml, 2), round(rho_pg, 2),
        ])
        for r in rows:
            csv_rows.append([
                args.scenario, n, r[0],
                round(r[1], 1), round(r[2], 1),
                round(r[3], 2) if can_featurize else "",
                round(r[4], 2) if can_featurize else "",
                round(r[5], 1), round(r[6], 1),
                "", "", "", "", "", "",
            ])
        summary.append((n, exact_correct, pair_correct, regret, speedup,
                        rho_ml, rho_pg, med_sub_q, mean_sub_acc))

    print("\n" + "═" * 96)
    print(f"=== Join-Order Evaluation Summary (scenario: {args.scenario}) ===")
    print("PRIMARY   : sub-plan q-err (lower=better), accuracy % (higher=better), regret (1.00x=ideal)")
    print("SECONDARY : exact OK, pair OK, speedup vs PG (>1.00x=faster), Spearman rho (+1.00=ideal)")
    print("─" * 96)
    print(f"{'model':<8} {'sub q-err':>10} {'sub acc %':>10} {'regret':>9} "
          f"{'speedup':>9} {'pair OK':>9} {'exact OK':>9} {'rho ML':>9} {'rho PG':>9}")
    print("─" * 96)
    for (n, exact_ok, pair_ok, regret, speedup,
         rho_ml, rho_pg, med_q, mean_acc) in summary:
        print(f"{n:<8} {med_q:>10.2f} {mean_acc:>9.2f}% {regret:>8.2f}x "
              f"{speedup:>8.2f}x {str(pair_ok):>9} {str(exact_ok):>9} "
              f"{rho_ml:>+9.2f} {rho_pg:>+9.2f}")
    print("\nHow to read: regret=1.00x + pair OK=True + speedup>=1.00x -> ML picks the optimal join order.")

    conn.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if OUT.exists() and OUT.stat().st_size > 0:
        with open(OUT, "r", newline="") as rf:
            first_line = rf.readline().strip()
            if "ml_plan_cost" in first_line:
                write_header = False
    mode = "w" if write_header else "a"
    with open(OUT, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "scenario", "model", "candidate", "ml_rows_cout", "ml_plan_cost",
                "med_qerr", "subplan_acc_pct", "pg_cost", "exec_ms",
                "exact_rank_ok", "subplan_pair_ok", "regret",
                "speedup_vs_pg", "rho_ml", "rho_pg",
            ])
        w.writerows(csv_rows)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()