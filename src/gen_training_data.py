# Gemini Version: 2.0.0
"""Generates results/training_data.csv for the cardinality models.

================================================================================
DATASET DIMENSIONS, ENCODING/DECODING & RESEARCH LITERATURE REFERENCES
================================================================================

1. DATASET DIMENSIONS:
   * Feature Dimension (FEAT_DIM = 23):
       - 3 table indicator flags : has_customer, has_orders, has_lineitem
       - 2 join indicator flags  : join_orders_customer, join_lineitem_orders
       - 6 x 3 = 18 filter slots : <table_col>_filter_used, <table_col>_value_lo,
                                   <table_col>_value_hi for each of the 6 filter
                                   columns (c_mktsegment, o_orderdate, o_totalprice,
                                   l_shipdate, l_quantity, l_discount).
   * Metadata & Target Columns (4 columns):
       - query_sql, pg_est_rows, actual_rows, source
   * Total CSV Columns = 27 (23 feature columns + 4 metadata/target columns).
   * Total Rows (Default) = 1,710 (1,500 random + 210 official-shape stratified),
     or pass a custom random count on the CLI, e.g.:
       python src/gen_training_data.py 3000   # -> 3,000 random + 210 = 3,210 rows

2. HOW QUERIES ARE ENCODED (SQL -> 23-D Vector) & DECODED (23-D Vector -> Meaning):
   * Step A — Table & Join Indicators (First 5 features, indices 0..4):
       - Indices 0..2 (`has_customer`, `has_orders`, `has_lineitem`):
           * Encode: 1.0 if the table appears in `FROM ...`, else 0.0.
           * Decode: Reading which flags are 1.0 reconstructs the `FROM` clause.
       - Indices 3..4 (`join_orders_customer`, `join_lineitem_orders`):
           * Encode: 1.0 if both endpoint tables of that foreign-key join are
             present in the query, else 0.0.
           * Decode: Reading which join flags are 1.0 reconstructs the foreign-key
             join conditions (`orders.o_custkey = customer.c_custkey` and/or
             `lineitem.l_orderkey = orders.o_orderkey`).

   * Step B — Filter Slot Encoding Contract (Next 18 features, indices 5..22):
     Each of the 6 filterable columns owns a 3-slot block `[used, lo, hi]`:
       - Unused column (`no predicate`):
           * Encode: `used = 0.0, lo = 0.0, hi = 0.0`
           * Decode: When `used == 0.0`, the column is not filtered in `WHERE`.
             (Having a dedicated `used` flag is critical because a normalized lower
             bound of `lo = 0.0` at the column's minimum is distinguishable from
             an unused filter slot).
       - Equality predicate (`col = v`):
           * Encode: `used = 1.0, lo = norm(v), hi = norm(v)` (i.e., `lo == hi`)
           * Decode: When `used == 1.0` and `lo == hi`, the predicate is an exact
             match (`=`).
       - Range predicate (`col BETWEEN a AND b`):
           * Encode: `used = 1.0, lo = norm(a), hi = norm(b)` (with `lo < hi`)
           * Decode: When `used == 1.0` and `lo < hi`, the predicate is a range
             filter spanning `[a, b]`.

   * Step C — Value Normalization (Encode) & Inverse Formula (Decode):
       - Numeric & Date Columns (`o_orderdate`, `o_totalprice`, `l_shipdate`,
         `l_quantity`, `l_discount`):
           * Encode (Min-Max Normalization to [0.0, 1.0]):
               norm_val = (raw_val - col_min) / (col_max - col_min + 1e-9)
             (Dates are first converted to integer proleptic Gregorian ordinals
             via `date.toordinal()` before min-max normalization).
           * Decode (Normalized [0.0, 1.0] -> Raw SQL Literal):
               raw_val = col_min + norm_val * (col_max - col_min)
             (For dates, `date.fromordinal(round(raw_val))` restores `YYYY-MM-DD`).
       - Categorical Column (`customer.c_mktsegment`):
           * Alphabetical domain (`M = 5` categories after `.strip()`):
               ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'HOUSEHOLD', 'MACHINERY']
           * Encode (Stable evenly-spaced ordinal position in (0, 1)):
               pos = (index + 1) / (M + 1)
             which maps the 5 categories to:
               AUTOMOBILE -> 1/6 (0.1667)
               BUILDING   -> 2/6 (0.3333)
               FURNITURE  -> 3/6 (0.5000)
               HOUSEHOLD  -> 4/6 (0.6667)
               MACHINERY  -> 5/6 (0.8333)
             Note: This compact encoding keeps the feature vector fixed at 23D;
             it is an identifier code for a nominal category, not a magnitude.
           * Decode (Position -> Category String):
               index = round(pos * (M + 1)) - 1  ->  domain[index]

3. RESEARCH LITERATURE REFERENCES (2009 - 2025):
   * Kipf et al. (CIDR 2019): "Learned Cardinalities: Estimating Correlated Joins
     with Deep Learning" (MSCN). Introduced the (table, join, predicate) triple-set
     encoding and interval [lo, hi] representation adapted here into a flat 23-D vector.
     URL: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf
   * Wang et al. (PVLDB 2021): "Are We Ready For Learned Cardinality Estimation?",
     PVLDB 14(9), pp. 1640-1654. Evaluated query-driven vs. data-driven estimators
     under workload drift and correlated predicates.
     URL: https://www.vldb.org/pvldb/vol14/p1640-wang.pdf
   * Negi et al. (PVLDB 2023): "Robust Query Driven Cardinality Estimation under
     Changing Workloads", PVLDB 16(6), pp. 1520-1533. Demonstrated why training on
     broad random workloads prevents catastrophic errors on unseen custom queries.
     URL: https://www.vldb.org/pvldb/vol16/p1520-negi.pdf
   * Chronis et al. (arXiv 2024): "CardBench: A Benchmark for Learned Cardinality
     Estimation in Relational Databases" (Google Research). Standardizes diverse
     single-table and multi-table workload generation for learned cardinality models.
     URL: https://arxiv.org/html/2408.16170v1
   * Zhu et al. (PVLDB 2024/2025): "PRICE: A Pretrained Model for Cross-Database
     Cardinality Estimation", PVLDB 18(3), pp. 637-650. Shows the importance of
     combining table/join topology with continuous range normalization.
     URL: https://www.vldb.org/pvldb/vol18/p637-zhu.pdf
   * Moerkotte, Neumann, & Steidl (VLDB 2009) & Leis et al. (PVLDB 2015):
     Established Q-Error and showed how attribute-value independence (AVI) fails on
     multi-table joins with correlated filters.
     URLs: http://www.vldb.org/pvldb/vol2/vldb09-657.pdf
           https://www.vldb.org/pvldb/vol9/p204-leis.pdf
================================================================================
"""
import csv
import random
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import psycopg2
from config import DB_CONFIG

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "training_data.csv"
SEED = 42

if __name__ == "__main__":
    try:
        N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    except ValueError as ex:
        raise SystemExit("Usage: python src/gen_training_data.py [non-negative random row count]") from ex
    if N_RANDOM < 0:
        raise SystemExit("The random row count must be non-negative.")
else:
    N_RANDOM = 1500

STATEMENT_TIMEOUT_MS = 15_000
SANITY_CAP_ROWS = 20_000_000

# --------------------------------------------------------------------------
# SCHEMA & JOIN GRAPH TOPOLOGY (3 Tables, 2 Foreign-Key Joins, 6 Filter Columns)
# --------------------------------------------------------------------------
# The 3 in-scope TPC-H tables form a linear 1-to-many chain:
#
#   [customer] (1) <---> (N) [orders] (1) <---> (N) [lineitem]
#        c_custkey = o_custkey      o_orderkey = l_orderkey
#
# ALL 6 VALID CONNECTED TABLE / JOIN COMBINATIONS:
#   1. ["customer"]                         -> 1-table scan (no joins)
#   2. ["orders"]                           -> 1-table scan (no joins)
#   3. ["lineitem"]                         -> 1-table scan (no joins; e.g., Q1, Q6)
#   4. ["customer", "orders"]               -> 2-table join:
#                                              orders.o_custkey = customer.c_custkey
#   5. ["orders", "lineitem"]               -> 2-table join:
#                                              lineitem.l_orderkey = orders.o_orderkey
#   6. ["customer", "orders", "lineitem"]   -> Full 3-table chain join (both join edges; e.g., Q3):
#                                              orders.o_custkey = customer.c_custkey
#                                              AND lineitem.l_orderkey = orders.o_orderkey
#
# INVALID COMBINATION (Why 1 subset out of 2^3 - 1 = 7 is excluded):
#   x  ["customer", "lineitem"] (WITHOUT "orders") -> INVALID!
#      `customer` and `lineitem` share no direct foreign key; joining them
#      without `orders` produces an unconstrained Cartesian product (cross join)
#      of 150,000 x 6,001,215 = ~900 billion rows.
# --------------------------------------------------------------------------
TABLES = ["customer", "orders", "lineitem"]

JOIN_EDGES = [
    ("orders",   "o_custkey",  "customer", "c_custkey"),
    ("lineitem", "l_orderkey", "orders",   "o_orderkey"),
]

# (table, column, kind) - all 6 columns a generated query may filter on.
# IMPORTANT: Keep all 6 columns (especially orders.o_totalprice) so FEAT_DIM = 23
# matches the trained models in results/ and custom queries in evaluate_tpch.py.
PRED_COLS = [
    ("customer", "c_mktsegment", "cat"),
    ("orders",   "o_orderdate",  "date"),
    ("orders",   "o_totalprice", "num"),
    ("lineitem", "l_shipdate",   "date"),
    ("lineitem", "l_quantity",   "num"),
    ("lineitem", "l_discount",   "num"),
]

NEIGHBORS = {t: set() for t in TABLES}
for a, _, b, _ in JOIN_EDGES:
    NEIGHBORS[a].add(b)
    NEIGHBORS[b].add(a)

# --------------------------------------------------------------------------
# STRATIFIED QUERY SHAPES (Official TPC-H shapes active; Custom shapes commented out)
# --------------------------------------------------------------------------
# Why only Q1, Q3, Q6 are active by default:
#   * Following Negi et al. (PVLDB 2023) and Wang et al. (PVLDB 2021), custom
#     evaluation queries in evaluate_tpch.py should NOT be hardcoded into the
#     training generator so they remain a true test of out-of-distribution
#     generalization, and so you can change custom queries in evaluate_tpch.py
#     at any time without regenerating training data or retraining models.
#
# HOW TO ADD ANY CUSTOM QUERY SHAPE HERE (IF EVER NEEDED):
#   Each entry is a 4-tuple:
#     (label_str, tables_list, forced_predicates_list, num_queries_int)
#   where:
#     * label_str              : tag saved in the CSV `source` column
#     * tables_list            : one of the 6 valid connected table combinations above
#     * forced_predicates_list : list of (table, column, kind, mode) tuples
#                                - kind is "cat", "date", or "num" (matching PRED_COLS)
#                                - mode is "eq" (=) or "range" (BETWEEN a AND b)
#     * num_queries_int        : how many random-literal queries of this shape to generate
# --------------------------------------------------------------------------
STRATIFIED_SHAPES = [
    # ---- Active: Official in-scope TPC-H query cores (210 queries total) ----
    ("q1-like", ["lineitem"],
     [("lineitem", "l_shipdate", "date", "range")], 60),
    ("q3-like", ["customer", "orders", "lineitem"],
     [("customer", "c_mktsegment", "cat",  "eq"),
      ("orders",   "o_orderdate",  "date", "range"),
      ("lineitem", "l_shipdate",   "date", "range")], 90),
    ("q6-like", ["lineitem"],
     [("lineitem", "l_shipdate", "date", "range"),
      ("lineitem", "l_discount", "num",  "range"),
      ("lineitem", "l_quantity", "num",  "range")], 60),

    # ---- Optional / Commented-Out Custom Query Shape Templates ----
    # Leave these commented out for clean train/test separation (recommended),
    # or uncomment/copy the pattern below if your supervisor ever asks to inject
    # a specific custom query template into the training set:
    #
    # # Example 1: 2-table join (customer + orders) with segment + orderdate range
    # ("custom-cust-ord", ["customer", "orders"],
    #  [("customer", "c_mktsegment", "cat",  "eq"),
    #   ("orders",   "o_orderdate",  "date", "range")], 40),
    #
    # # Example 2: 2-table join (orders + lineitem) with totalprice + shipdate range
    # ("custom-ord-line", ["orders", "lineitem"],
    #  [("orders",   "o_totalprice", "num",  "range"),
    #   ("lineitem", "l_shipdate",   "date", "range")], 40),
    #
    # # Example 3: 3-table join with 1 filter per table (segment + totalprice + quantity)
    # ("custom-3tbl-bldg", ["customer", "orders", "lineitem"],
    #  [("customer", "c_mktsegment", "cat", "eq"),
    #   ("orders",   "o_totalprice", "num", "range"),
    #   ("lineitem", "l_quantity",   "num", "range")], 40),
    #
    # # Example 4: 3-table join with anti-correlated price + quantity ranges
    # ("custom-price-qty", ["customer", "orders", "lineitem"],
    #  [("orders",   "o_totalprice", "num", "range"),
    #   ("lineitem", "l_quantity",   "num", "range")], 40),
]

TABLE_COLS = [f"has_{t}" for t in TABLES]                       # 3
JOIN_COLS = [f"join_{a}_{b}" for a, _, b, _ in JOIN_EDGES]      # 2
PRED_FEATURE_COLS = []
for t, c, _ in PRED_COLS:
    PRED_FEATURE_COLS += [f"{t}_{c}_filter_used",
                          f"{t}_{c}_value_lo",
                          f"{t}_{c}_value_hi"]                  # 6 x 3 = 18
FEATURE_COLS = TABLE_COLS + JOIN_COLS + PRED_FEATURE_COLS       # 23
FEAT_DIM = len(FEATURE_COLS)

TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
PRED_IDX = {(t, c): i for i, (t, c, _) in enumerate(PRED_COLS)}


def norm(v, lo, hi):
    """Min-Max normalize value `v` in `[lo, hi]` into `[0.0, 1.0]`.
    To decode a normalized value `nv` back to its raw domain:
      raw_v = lo + nv * (hi - lo)
    """
    return (v - lo) / (hi - lo + 1e-9)


def load_col_stats(cur):
    stats = {}
    for t, c, kind in PRED_COLS:
        if kind == "cat":
            cur.execute(f'SELECT DISTINCT "{c}" FROM {t} ORDER BY 1')
            # c_mktsegment is CHAR(n) in TPC-H, so PostgreSQL returns values
            # padded with trailing spaces ('BUILDING  '). strip() keeps the
            # category strings clean and domain order stable.
            stats[(t, c)] = {"kind": kind,
                             "domain": [str(r[0]).strip() for r in cur.fetchall()]}
        else:
            cur.execute(f'SELECT MIN("{c}"), MAX("{c}") FROM {t}')
            lo, hi = cur.fetchone()
            if kind == "date":
                lo, hi = lo.toordinal(), hi.toordinal()
            stats[(t, c)] = {"kind": kind, "min": float(lo), "max": float(hi)}
    return stats


def sample_tables(rng):
    """Sample any connected subgraph of {customer, orders, lineitem}.
    By walking NEIGHBORS along JOIN_EDGES, this function guarantees only the
    6 valid connected table combinations are ever sampled (never the invalid
    disconnected ['customer', 'lineitem'] cross join).
    Weights [2, 3, 4] favor 2-table and 3-table joins while keeping 1-table scans."""
    n = rng.choices([1, 2, 3], weights=[2, 3, 4])[0]
    tables = [rng.choice(TABLES)]
    while len(tables) < n:
        # Sorting is required for reproducibility: set iteration order can
        # differ between Python processes even when the random seed is fixed.
        cands = sorted({x for t in tables for x in NEIGHBORS[t]
                        if x not in tables})
        if not cands:
            break
        tables.append(rng.choice(cands))
    return tables


def _one_predicate(rng, col_stats, t, c, kind, mode=None):
    """Return ((table, col), value_lo, value_hi, sql_fragment).
    The feature encoding uses value_lo == value_hi for equality and
    value_lo < value_hi for a range [Kipf et al., CIDR 2019; Chronis et al.,
    CardBench 2024].
    `filter_used` is added later by `featurize`; it distinguishes an equality
    at 0.0 from an unused filter slot.
    mode forces "eq" or "range" (used by the stratified shapes);
    mode=None picks randomly (40% exact match for discrete/date columns,
    while continuous o_totalprice always uses a range to avoid 0-row '=' queries)."""
    s = col_stats[(t, c)]
    if kind == "cat":
        dom = s["domain"]
        v = rng.choice(dom)
        # This is an ordinal code for a nominal category, not a measurement.
        # Tree models can use it compactly; one-hot encoding is an alternative
        # if a future experiment needs to remove any artificial ordering.
        pos = (dom.index(v) + 1) / (len(dom) + 1)
        return ((t, c), pos, pos, f"{c} = '{v}'")

    # Continuous currency column o_totalprice should always use range filters
    if c == "o_totalprice" and mode is None:
        use_eq = False
    else:
        use_eq = (mode == "eq") or (mode is None and rng.random() < 0.4)

    if use_eq:
        v = rng.uniform(s["min"], s["max"])
        if kind == "date":
            v = int(round(v))
            sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
        elif c == "l_quantity":
            # In TPC-H, l_quantity takes integer values 1..50
            v = float(round(v))
            sql = f"{c} = {v:.2f}"
        else:
            v = round(v, 2)
            sql = f"{c} = {v:.2f}"
        nv = norm(v, s["min"], s["max"])
        return ((t, c), nv, nv, sql)

    # Range predicate: support two-sided [a, b], left-clamped [min, b], and
    # right-clamped [a, max].
    r_type = rng.random()
    if r_type < 0.20:
        a, b = s["min"], rng.uniform(s["min"], s["max"])
    elif r_type < 0.40:
        a, b = rng.uniform(s["min"], s["max"]), s["max"]
    else:
        a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
        if a > b:
            a, b = b, a

    if kind == "date":
        a, b = int(a), int(b)
        if a >= b:
            b = min(int(s["max"]), a + 1)
            if a >= b:
                a = max(int(s["min"]), b - 1)
        sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
    else:
        a, b = round(a, 2), round(b, 2)
        if a >= b:
            b = round(min(s["max"], a + 0.01), 2)
            if a >= b:
                a = round(max(s["min"], b - 0.01), 2)
        sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
    return ((t, c), norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql)


def sample_predicates(rng, col_stats, tables):
    """Sample 0 to len(cols) predicates across the active tables.
    When multiple tables are joined and k_preds >= len(tables), half the time
    we pick one predicate from each joined table first so multi-table filter
    interactions are naturally well-represented without hardcoding custom shapes."""
    cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
    max_k = len(cols)
    possible_k = list(range(0, max_k + 1))
    weights = [1 if k == 0 else (4 if k <= 3 else 2) for k in possible_k]
    k_preds = rng.choices(possible_k, weights=weights)[0]

    if len(tables) > 1 and k_preds >= len(tables) and rng.random() < 0.5:
        chosen = []
        for tbl in tables:
            tbl_cols = [(t, c, k) for t, c, k in cols if t == tbl]
            if tbl_cols:
                chosen.append(rng.choice(tbl_cols))
        remaining = [x for x in cols if x not in chosen]
        rng.shuffle(remaining)
        chosen += remaining[: max(0, k_preds - len(chosen))]
    else:
        rng.shuffle(cols)
        chosen = cols[:k_preds]

    return [_one_predicate(rng, col_stats, t, c, kind)
            for t, c, kind in chosen]


def stratified_predicates(rng, col_stats, forced, label=""):
    """Build predicates for the stratified TPC-H query shapes.
    For 'q3-like', 50% of generated examples use a shared cutoff date D where
    o_orderdate is bounded above by D and l_shipdate is bounded below by D
    (implemented as BETWEEN predicates, mirroring official TPC-H Q3)."""
    if label == "q3-like" and rng.random() < 0.5:
        c_pred = _one_predicate(rng, col_stats, "customer", "c_mktsegment", "cat", "eq")
        o_st = col_stats[("orders", "o_orderdate")]
        l_st = col_stats[("lineitem", "l_shipdate")]
        lo_pivot = max(o_st["min"] + 180, l_st["min"] + 180)
        hi_pivot = min(o_st["max"] - 180, l_st["max"] - 180)
        pivot = int(rng.uniform(lo_pivot, hi_pivot))
        omin, smax = int(o_st["min"]), int(l_st["max"])
        o_sql = (f"o_orderdate BETWEEN DATE '{date.fromordinal(omin).isoformat()}'"
                 f" AND DATE '{date.fromordinal(pivot).isoformat()}'")
        l_sql = (f"l_shipdate BETWEEN DATE '{date.fromordinal(pivot).isoformat()}'"
                 f" AND DATE '{date.fromordinal(smax).isoformat()}'")
        return [
            c_pred,
            (("orders", "o_orderdate"), norm(omin, o_st["min"], o_st["max"]),
             norm(pivot, o_st["min"], o_st["max"]), o_sql),
            (("lineitem", "l_shipdate"), norm(pivot, l_st["min"], l_st["max"]),
             norm(smax, l_st["min"], l_st["max"]), l_sql),
        ]
    return [_one_predicate(rng, col_stats, t, c, kind, mode) for t, c, kind, mode in forced]


def build_where(joins, pred_sqls):
    wheres = [f"{a}.{ac} = {b}.{bc}" for a, ac, b, bc in joins] + pred_sqls
    return " WHERE " + " AND ".join(wheres) if wheres else ""


def featurize(tables, joins, preds):
    """Encode `(tables, joins, preds)` into the flat 23-element feature vector `f`.

    Vector layout (length 23):
      * f[0..2]  : Table flags `[has_customer, has_orders, has_lineitem]` (0.0 or 1.0)
      * f[3..4]  : Join flags `[join_orders_customer, join_lineitem_orders]` (0.0 or 1.0)
      * f[5..22] : Six 3-element blocks `[filter_used, value_lo, value_hi]`, one per
                   column in PRED_COLS:
                     - f[5..7]   : customer.c_mktsegment
                     - f[8..10]  : orders.o_orderdate
                     - f[11..13] : orders.o_totalprice
                     - f[14..16] : lineitem.l_shipdate
                     - f[17..19] : lineitem.l_quantity
                     - f[20..22] : lineitem.l_discount
    """
    f = [0.0] * FEAT_DIM
    for t in tables:
        f[TABLE_IDX[t]] = 1.0
    off = len(TABLES)
    for e in joins:
        f[off + JOIN_IDX[e]] = 1.0
    off += len(JOIN_EDGES)
    for (t, c), lo, hi, _ in preds:
        b = off + 3 * PRED_IDX[(t, c)]
        f[b], f[b + 1], f[b + 2] = 1.0, lo, hi
    return f


def run_one(cur, tables, preds, source):
    """Execute one query. Returns (row, pg_est, actual) or None (SQL error/
    timeout) or "REJECTED" (sanity cap tripped). `source` tags the row so
    you can later tell random rows apart from q1/q3/q6-shaped rows."""
    joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
    where = build_where(joins, [p[3] for p in preds])
    from_clause = f"FROM {', '.join(tables)}"

    # The FULL runnable statement, not just the FROM/WHERE body, so any
    # row's query_sql can be pasted directly into pgAdmin and run.
    count_sql = f"SELECT COUNT(*) {from_clause}{where}"
    display_sql = f"SELECT * {from_clause}{where};"

    try:
        cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_clause}{where}")
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
        cur.execute(count_sql)
        actual = cur.fetchone()[0]
    except psycopg2.Error as ex:
        print(f"  skipped ({type(ex).__name__}): {str(ex).strip()[:80]}")
        cur.connection.rollback()
        return None
    if actual > SANITY_CAP_ROWS:
        return "REJECTED"
    row = featurize(tables, joins, preds) + [display_sql, pg_est, actual, source]
    return row, pg_est, actual


def main():
    rng = random.Random(SEED)
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    col_stats = load_col_stats(cur)

    n_stratified = sum(c for *_, c in STRATIFIED_SHAPES)
    print(f"Tables      : {', '.join(TABLES)}")
    print(f"Joins       : {len(JOIN_EDGES)}  |  Filter columns: {len(PRED_COLS)}")
    print(f"Feature dim : {FEAT_DIM} (named columns, 3 per filter: used/lo/hi)")
    print(f"Plan        : {N_RANDOM} random + {n_stratified} stratified (q1/q3/q6-like) "
          f"= {N_RANDOM + n_stratified} total\n")

    out_rows, pg_qerrs, rejected = [], [], 0

    # --- stratified batch first (small, shaped like the real eval queries) ---
    for label, tables, forced, count in STRATIFIED_SHAPES:
        made, attempts = 0, 0
        while made < count and attempts < count * 5:
            attempts += 1
            preds = stratified_predicates(rng, col_stats, forced, label=label)
            result = run_one(cur, tables, preds, source=label)
            if result is None:
                continue
            if result == "REJECTED":
                rejected += 1
                continue
            row, pg_est, actual = result
            out_rows.append(row)
            pg_qerrs.append(max(max(actual, 1) / max(pg_est, 1),
                                max(pg_est, 1) / max(actual, 1)))
            made += 1
        print(f"  [{label:<8}] {made}/{count} generated")

    # --- random batch (the bulk of the dataset) ---
    done = 0
    t_start = time.time()
    while done < N_RANDOM:
        tables = sample_tables(rng)
        preds = sample_predicates(rng, col_stats, tables)
        result = run_one(cur, tables, preds, source="random")
        if result is None:
            continue
        if result == "REJECTED":
            rejected += 1
            continue
        row, pg_est, actual = result
        out_rows.append(row)
        pg_qerrs.append(max(max(actual, 1) / max(pg_est, 1),
                            max(pg_est, 1) / max(actual, 1)))
        done += 1
        if done % 50 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta_min = (N_RANDOM - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  random: {done}/{N_RANDOM} | rejected: {rejected} | "
                  f"PG median q-error so far: {statistics.median(pg_qerrs):.2f} | "
                  f"~{rate:.1f}/s | ETA {eta_min:.1f} min")

    fields = FEATURE_COLS + ["query_sql", "pg_est_rows", "actual_rows", "source"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(fields)
        w.writerows(out_rows)
    conn.close()
    print(f"\nSaved {len(out_rows)} examples -> {OUT}")
    print(f"Rejected {rejected} oversized queries during generation")
    print(f"PostgreSQL baseline on generated set: median q-error = {statistics.median(pg_qerrs):.2f}")


if __name__ == "__main__":
    main()