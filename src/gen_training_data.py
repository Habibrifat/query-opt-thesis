# """Generates results/training_data.csv for the cardinality models.

# SCOPE (per supervisor's advice): the "RetailMart" 3-table dataset -
# customers place orders, orders contain line items:
#     TABLES    = customer, orders, lineitem
#     JOINS     = orders.o_custkey = customer.c_custkey
#                 lineitem.l_orderkey = orders.o_orderkey
#     PREDICATES on 7 meaningful columns only.

# FEATURES: 33 human-readable columns (no more f0-f80):
#     3  x table_is_<table>            (0/1 - which tables appear)
#     2  x join_<a>_to_<b>             (0/1 - which links appear)
#     7  x 4 slots per predicate col:  <col>_is_eq, <col>_eq_val, <col>_lo, <col>_hi

# KEPT from the previous version: sanity cap (never trips on this 3-table
# chain schema - joins are many-to-one, max ~6M rows - kept as a safety
# net), statement timeout, rejection counting, progress with ETA, source
# column, fixed seed 42, stratified real-query-shaped batches (now only
# the in-scope shapes: q1/q3/q6).
# """
# import csv
# import random
# import statistics
# import sys
# import time
# from datetime import date
# from pathlib import Path

# import psycopg2
# from config import DB_CONFIG
# from schema import TABLES, JOIN_EDGES, PRED_COLS

# BASE = Path(__file__).resolve().parent.parent
# OUT = BASE / "results" / "training_data.csv"
# SEED = 42

# if __name__ == '__main__':
#     N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
# else:
#     N_RANDOM = 4000

# # Safety net: reject anything absurdly large. On the 3-table chain schema
# # (customer -> orders -> lineitem) joins are many-to-one, so real results
# # never exceed lineitem's ~6M rows - this should never trip.
# SANITY_CAP_ROWS = 20_000_000

# # A single slow/stuck query is skipped and counted, not fatal.
# STATEMENT_TIMEOUT_MS = 15_000

# # Stratified batches shaped like the real TPC-H queries we evaluate on
# # (evaluate_tpch.py: q1, q3, q6). Same tables + predicate COLUMNS, random
# # VALUES - so the model learns the shapes, not the test literals.
# # ~9% of the dataset: enough to teach the shapes, too little to memorize.
# STRATIFIED_SHAPES = [
#     ("q1-like", ["lineitem"],
#      [("lineitem", "l_shipdate", "date", "range")], 100),
#     ("q3-like", ["customer", "orders", "lineitem"],
#      [("customer", "c_mktsegment", "cat", "eq"),
#       ("orders", "o_orderdate", "date", "range"),
#       ("lineitem", "l_shipdate", "date", "range")], 200),
#     ("q6-like", ["lineitem"],
#      [("lineitem", "l_shipdate", "date", "range"),
#       ("lineitem", "l_discount", "num", "range"),
#       ("lineitem", "l_quantity", "num", "range")], 100),
# ]

# TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
# JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
# PRED_IDX = {c: i for i, c in enumerate([(t, c) for t, c, _ in PRED_COLS])}
# FEAT_DIM = len(TABLES) + len(JOIN_EDGES) + 4 * len(PRED_COLS)

# NEIGHBORS = {t: set() for t in TABLES}
# for a, _, b, _ in JOIN_EDGES:
#     NEIGHBORS[a].add(b)
#     NEIGHBORS[b].add(a)


# def feature_names():
#     """Human-readable column names for the CSV header (replaces f0..fN)."""
#     names = [f"table_is_{t}" for t in TABLES]
#     names += [f"join_{a}_to_{b}" for a, _, b, _ in JOIN_EDGES]
#     for t, c, kind in PRED_COLS:
#         names += [f"{c}_is_eq", f"{c}_eq_val", f"{c}_lo", f"{c}_hi"]
#     return names


# def norm(v, lo, hi):
#     return (v - lo) / (hi - lo + 1e-9)


# def load_col_stats(cur):
#     stats = {}
#     for t, c, kind in PRED_COLS:
#         if kind == "cat":
#             cur.execute(f'SELECT DISTINCT "{c}" FROM {t} ORDER BY 1')
#             # strip(): CHAR(n) values arrive padded ('BUILDING  ') - order
#             # is preserved by stripping, so encoded positions stay stable.
#             stats[(t, c)] = {"kind": kind,
#                              "domain": [str(r[0]).strip() for r in cur.fetchall()]}
#         else:
#             cur.execute(f'SELECT MIN("{c}"), MAX("{c}") FROM {t}')
#             lo, hi = cur.fetchone()
#             if kind == "date":
#                 lo, hi = lo.toordinal(), hi.toordinal()
#             stats[(t, c)] = {"kind": kind, "min": float(lo), "max": float(hi)}
#     return stats


# def sample_tables(rng):
#     # Favor 2-3 table queries (the mid-level complexity our evaluation uses);
#     # 1-table queries kept as the easy tier.
#     n = rng.choices([1, 2, 3], weights=[1, 2, 3])[0]
#     tables = [rng.choice(TABLES)]
#     while len(tables) < n:
#         cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
#         if not cands:
#             break
#         tables.append(rng.choice(cands))
#     return tables


# def _one_predicate(rng, col_stats, t, c, kind, mode=None):
#     """mode forces "eq" or "range"; None = random (40% eq for num/date)."""
#     s = col_stats[(t, c)]
#     if kind == "cat":
#         dom = s["domain"]
#         v = rng.choice(dom)
#         pos = (dom.index(v) + 1) / (len(dom) + 1)
#         return ((t, c), "eq", pos, 0.0, f"{c} = '{v}'")
#     use_eq = (mode == "eq") or (mode is None and rng.random() < 0.4)
#     if use_eq:
#         v = rng.uniform(s["min"], s["max"])
#         if kind == "date":
#             v = int(round(v))
#             sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
#         else:
#             sql = f"{c} = {v:.2f}"
#         return ((t, c), "eq", norm(v, s["min"], s["max"]), 0.0, sql)
#     a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
#     if a > b:
#         a, b = b, a
#     if kind == "date":
#         a, b = int(a), int(b)
#         sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
#     else:
#         sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
#     return ((t, c), "range", norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql)


# def sample_predicates(rng, col_stats, tables, max_preds=3):
#     cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
#     rng.shuffle(cols)
#     return [_one_predicate(rng, col_stats, t, c, kind)
#             for t, c, kind in cols[: rng.randint(0, max_preds)]]


# def stratified_predicates(rng, col_stats, forced):
#     return [_one_predicate(rng, col_stats, t, c, kind, mode) for t, c, kind, mode in forced]


# def build_body(tables, joins, pred_sqls):
#     wheres = [f"{a}.{ac} = {b}.{bc}" for a, ac, b, bc in joins] + pred_sqls
#     where = " WHERE " + " AND ".join(wheres) if wheres else ""
#     return f"FROM {', '.join(tables)}{where}"


# def featurize(tables, joins, preds):
#     f = [0.0] * FEAT_DIM
#     for t in tables:
#         f[TABLE_IDX[t]] = 1.0
#     off = len(TABLES)
#     for e in joins:
#         f[off + JOIN_IDX[e]] = 1.0
#     off += len(JOIN_EDGES)
#     for (t, c), kind, v1, v2, _ in preds:
#         b = off + 4 * PRED_IDX[(t, c)]
#         if kind == "eq":
#             f[b], f[b + 1] = 1.0, v1
#         else:
#             f[b + 2], f[b + 3] = v1, v2
#     return f


# def run_one(cur, tables, preds, source):
#     """Execute one query. Returns (row, pg_est, actual), None (error/timeout),
#     or "REJECTED" (sanity cap). `source` tags the row for per-source analysis."""
#     joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
#     body = build_body(tables, joins, [p[4] for p in preds])
#     try:
#         cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
#         pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
#         cur.execute("SELECT COUNT(*) " + body)
#         actual = cur.fetchone()[0]
#     except psycopg2.Error as ex:
#         print(f"  skipped ({type(ex).__name__}): {str(ex).strip()[:80]}")
#         cur.connection.rollback()
#         return None
#     if actual > SANITY_CAP_ROWS:
#         return "REJECTED"
#     row = featurize(tables, joins, preds) + [body, pg_est, actual, source]
#     return row, pg_est, actual


# def main():
#     rng = random.Random(SEED)
#     conn = psycopg2.connect(**DB_CONFIG)
#     conn.autocommit = True
#     cur = conn.cursor()
#     cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
#     col_stats = load_col_stats(cur)

#     n_stratified = sum(c for *_, c in STRATIFIED_SHAPES)
#     print(f"Feature dim: {FEAT_DIM}  = {len(TABLES)} tables + {len(JOIN_EDGES)} joins "
#           f"+ {len(PRED_COLS)} predicate cols x 4")
#     print(f"Features: {', '.join(feature_names())}")
#     print(f"Plan: {N_RANDOM} random + {n_stratified} stratified "
#           f"= {N_RANDOM + n_stratified} total")
#     print(f"Sanity cap: {SANITY_CAP_ROWS:,} rows (safety net)\n")

#     out_rows, pg_qerrs, rejected = [], [], 0

#     # --- stratified batch first ---
#     for label, tables, forced, count in STRATIFIED_SHAPES:
#         made = 0
#         attempts = 0
#         while made < count and attempts < count * 5:
#             attempts += 1
#             preds = stratified_predicates(rng, col_stats, forced)
#             result = run_one(cur, tables, preds, source=label)
#             if result is None:
#                 continue
#             if result == "REJECTED":
#                 rejected += 1
#                 continue
#             row, pg_est, actual = result
#             out_rows.append(row)
#             pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
#             made += 1
#         print(f"  [{label:<9}] {made}/{count} generated")

#     # --- random batch ---
#     done = 0
#     t_start = time.time()
#     while done < N_RANDOM:
#         tables = sample_tables(rng)
#         preds = sample_predicates(rng, col_stats, tables)
#         result = run_one(cur, tables, preds, source="random")
#         if result is None:
#             continue
#         if result == "REJECTED":
#             rejected += 1
#             continue
#         row, pg_est, actual = result
#         out_rows.append(row)
#         pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
#         done += 1
#         if done % 50 == 0:
#             elapsed = time.time() - t_start
#             rate = done / elapsed
#             eta_min = (N_RANDOM - done) / rate / 60 if rate > 0 else float("nan")
#             print(f"  random: {done}/{N_RANDOM} | rejected: {rejected} | "
#                   f"PG median q-err so far: {statistics.median(pg_qerrs):.2f} | "
#                   f"~{rate:.1f}/s | ETA {eta_min:.1f} min")

#     fields = feature_names() + ["query", "pg_est_rows", "actual_rows", "source"]
#     with open(OUT, "w", newline="") as fp:
#         w = csv.writer(fp)
#         w.writerow(fields)
#         w.writerows(out_rows)
#     conn.close()
#     print(f"\nSaved {len(out_rows)} examples -> {OUT}")
#     print(f"Rejected {rejected} pathological queries during generation")
#     print(f"PostgreSQL baseline on generated set: median q-error = {statistics.median(pg_qerrs):.2f}")


# if __name__ == "__main__":
#     main()


# """Generates results/training_data.csv for the cardinality models.

# SIMPLIFIED per supervisor's guidance (second pass):
#   * Only 3 tables: customer -> orders -> lineitem (a simple chain).
#   * Only 5 filter columns (was 9) - the most meaningful ones.
#   * Each filter is encoded with just 3 numbers instead of 5:
#         <col>_filter_used   -> 1 if this query filters on this column, else 0
#         <col>_value_lo      -> normalized lower bound (0 if unused)
#         <col>_value_hi      -> normalized upper bound (0 if unused)
#     An exact match (col = value) is just value_lo == value_hi.
#     A range (col BETWEEN a AND b) is value_lo < value_hi.
#     This cuts total columns from 50 down to 20 and removes the separate
#     is_eq/is_range flags, which were a second source of the "too many
#     zero columns" look.
#   * query_sql now stores the FULL runnable statement (SELECT * FROM ...),
#     not just the FROM/WHERE fragment - the earlier version could not be
#     pasted directly into pgAdmin and run, which is the syntax error seen
#     before ("FROM lineitem WHERE ..." has no SELECT).
# """
# import csv
# import random
# import statistics
# import sys
# import time
# from datetime import date
# from pathlib import Path

# import psycopg2
# from config import DB_CONFIG

# BASE = Path(__file__).resolve().parent.parent
# OUT = BASE / "results" / "training_data.csv"
# SEED = 42

# if __name__ == '__main__':
#     N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
# else:
#     N_QUERIES = 1500

# STATEMENT_TIMEOUT_MS = 15_000
# SANITY_CAP_ROWS = 20_000_000

# # --------------------------------------------------------------------------
# # Schema: 3 tables, 2 joins, 5 filter columns. Everything the generator
# # needs is written right here - nothing imported from schema.py.
# # --------------------------------------------------------------------------
# TABLES = ["customer", "orders", "lineitem"]

# JOIN_EDGES = [
#     ("orders",   "o_custkey",  "customer", "c_custkey"),
#     ("lineitem", "l_orderkey", "orders",   "o_orderkey"),
# ]

# # (table, column, kind) - the 5 columns a generated query may filter on.
# PRED_COLS = [
#     ("customer", "c_mktsegment", "cat"),
#     ("orders",   "o_orderdate",  "date"),
#     ("orders",   "o_totalprice", "num"),
#     ("lineitem", "l_quantity",   "num"),
#     ("lineitem", "l_discount",   "num"),
# ]

# NEIGHBORS = {t: set() for t in TABLES}
# for a, _, b, _ in JOIN_EDGES:
#     NEIGHBORS[a].add(b)
#     NEIGHBORS[b].add(a)

# # --------------------------------------------------------------------------
# # Feature columns - 20 total, every one self-explanatory by name.
# # --------------------------------------------------------------------------
# TABLE_COLS = [f"has_{t}" for t in TABLES]                       # 3
# JOIN_COLS = [f"join_{a}_{b}" for a, _, b, _ in JOIN_EDGES]        # 2
# PRED_FEATURE_COLS = []
# for t, c, _ in PRED_COLS:
#     PRED_FEATURE_COLS += [f"{t}_{c}_filter_used",
#                           f"{t}_{c}_value_lo",
#                           f"{t}_{c}_value_hi"]                   # 5 x 3 = 15
# FEATURE_COLS = TABLE_COLS + JOIN_COLS + PRED_FEATURE_COLS         # 20
# FEAT_DIM = len(FEATURE_COLS)

# TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
# JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
# PRED_IDX = {(t, c): i for i, (t, c, _) in enumerate(PRED_COLS)}


# def norm(v, lo, hi):
#     return (v - lo) / (hi - lo + 1e-9)


# def load_col_stats(cur):
#     stats = {}
#     for t, c, kind in PRED_COLS:
#         if kind == "cat":
#             cur.execute(f'SELECT DISTINCT "{c}" FROM {t} ORDER BY 1')
#             stats[(t, c)] = {"kind": kind, "domain": [r[0] for r in cur.fetchall()]}
#         else:
#             cur.execute(f'SELECT MIN("{c}"), MAX("{c}") FROM {t}')
#             lo, hi = cur.fetchone()
#             if kind == "date":
#                 lo, hi = lo.toordinal(), hi.toordinal()
#             stats[(t, c)] = {"kind": kind, "min": float(lo), "max": float(hi)}
#     return stats


# def sample_tables(rng):
#     n = rng.choices([1, 2, 3], weights=[2, 3, 4])[0]  # favor the full 3-table chain
#     tables = [rng.choice(TABLES)]
#     while len(tables) < n:
#         cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
#         if not cands:
#             break
#         tables.append(rng.choice(cands))
#     return tables


# def sample_predicates(rng, col_stats, tables, max_preds=3):
#     """Returns a list of ((table, col), value_lo, value_hi, sql_fragment).
#     value_lo == value_hi means an exact match; value_lo < value_hi means a
#     range. (This replaces the old separate eq/range encoding.)"""
#     cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
#     rng.shuffle(cols)
#     preds = []
#     for t, c, kind in cols[: rng.randint(0, max_preds)]:
#         s = col_stats[(t, c)]
#         if kind == "cat":
#             dom = s["domain"]
#             v = rng.choice(dom)
#             pos = (dom.index(v) + 1) / (len(dom) + 1)
#             preds.append(((t, c), pos, pos, f"{c} = '{v}'"))
#         elif rng.random() < 0.4:  # exact match on numeric/date
#             v = rng.uniform(s["min"], s["max"])
#             if kind == "date":
#                 v = int(round(v))
#                 sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
#             else:
#                 sql = f"{c} = {v:.2f}"
#             nv = norm(v, s["min"], s["max"])
#             preds.append(((t, c), nv, nv, sql))
#         else:  # range
#             a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
#             if a > b:
#                 a, b = b, a
#             if kind == "date":
#                 a, b = int(a), int(b)
#                 sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
#             else:
#                 sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
#             preds.append(((t, c), norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql))
#     return preds


# def build_where(joins, pred_sqls):
#     wheres = [f"{a}.{ac} = {b}.{bc}" for a, ac, b, bc in joins] + pred_sqls
#     return " WHERE " + " AND ".join(wheres) if wheres else ""


# def featurize(tables, joins, preds):
#     f = [0.0] * FEAT_DIM
#     for t in tables:
#         f[TABLE_IDX[t]] = 1.0
#     off = len(TABLES)
#     for e in joins:
#         f[off + JOIN_IDX[e]] = 1.0
#     off += len(JOIN_EDGES)
#     for (t, c), lo, hi, _ in preds:
#         b = off + 3 * PRED_IDX[(t, c)]
#         f[b], f[b + 1], f[b + 2] = 1.0, lo, hi
#     return f


# def run_one(cur, tables, preds):
#     """Execute one query. Returns (row, pg_est, actual, full_sql) or None
#     (SQL error/timeout) or "REJECTED" (sanity cap tripped)."""
#     joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
#     where = build_where(joins, [p[3] for p in preds])
#     from_clause = f"FROM {', '.join(tables)}"

#     # FIX: store the FULL runnable statement, not just the FROM/WHERE body,
#     # so any row's query_sql can be pasted directly into pgAdmin and run.
#     count_sql = f"SELECT COUNT(*) {from_clause}{where}"
#     display_sql = f"SELECT * {from_clause}{where};"

#     try:
#         cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * {from_clause}{where}")
#         pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
#         cur.execute(count_sql)
#         actual = cur.fetchone()[0]
#     except psycopg2.Error as ex:
#         print(f"  skipped ({type(ex).__name__}): {str(ex).strip()[:80]}")
#         cur.connection.rollback()
#         return None
#     if actual > SANITY_CAP_ROWS:
#         return "REJECTED"
#     row = featurize(tables, joins, preds) + [display_sql, pg_est, actual, "random"]
#     return row, pg_est, actual


# def main():
#     rng = random.Random(SEED)
#     conn = psycopg2.connect(**DB_CONFIG)
#     conn.autocommit = True
#     cur = conn.cursor()
#     cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
#     col_stats = load_col_stats(cur)

#     print(f"Tables      : {', '.join(TABLES)}")
#     print(f"Joins       : {len(JOIN_EDGES)}  |  Filter columns: {len(PRED_COLS)}")
#     print(f"Feature dim : {FEAT_DIM} (named columns, 3 per filter: used/lo/hi)")
#     print(f"Generating  : {N_QUERIES} queries\n")

#     out_rows, pg_qerrs, done, rejected = [], [], 0, 0
#     t_start = time.time()
#     while done < N_QUERIES:
#         tables = sample_tables(rng)
#         preds = sample_predicates(rng, col_stats, tables)
#         result = run_one(cur, tables, preds)
#         if result is None:
#             continue
#         if result == "REJECTED":
#             rejected += 1
#             continue
#         row, pg_est, actual = result
#         out_rows.append(row)
#         pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
#         done += 1
#         if done % 50 == 0:
#             elapsed = time.time() - t_start
#             rate = done / elapsed
#             eta_min = (N_QUERIES - done) / rate / 60 if rate > 0 else float("nan")
#             print(f"  {done}/{N_QUERIES} | rejected: {rejected} | "
#                   f"PG median q-error so far: {statistics.median(pg_qerrs):.2f} | "
#                   f"~{rate:.1f}/s | ETA {eta_min:.1f} min")

#     fields = FEATURE_COLS + ["query_sql", "pg_est_rows", "actual_rows", "source"]
#     with open(OUT, "w", newline="") as fp:
#         w = csv.writer(fp)
#         w.writerow(fields)
#         w.writerows(out_rows)
#     conn.close()
#     print(f"\nSaved {done} examples -> {OUT}")
#     print(f"Rejected {rejected} oversized queries during generation")
#     print(f"PostgreSQL baseline on generated set: median q-error = {statistics.median(pg_qerrs):.2f}")


# if __name__ == "__main__":
#     main()


"""Generates results/training_data.csv for the cardinality models.

SIMPLIFIED per supervisor's guidance:
  * Only 3 tables: customer -> orders -> lineitem (a simple chain).
  * Only 5 filter columns, each encoded with just 3 numbers:
        <col>_filter_used   -> 1 if this query filters on this column, else 0
        <col>_value_lo      -> normalized lower bound (0 if unused)
        <col>_value_hi      -> normalized upper bound (0 if unused)
    An exact match (col = value) is just value_lo == value_hi.
    A range (col BETWEEN a AND b) is value_lo < value_hi.
    Total feature columns: 20 (3 table flags + 2 join flags + 5x3 predicate
    features) - all self-explanatory by name, nothing like f0..f80.
  * query_sql stores the FULL runnable statement (SELECT * FROM ...;), so
    any row can be pasted directly into pgAdmin and run - not just the
    FROM/WHERE fragment (that used to cause a "syntax error at FROM").
  * Categorical values (customer.c_mktsegment) are stripped of trailing
    padding - this column is CHAR(n) in TPC-H, so PostgreSQL returns values
    like 'BUILDING  ' with trailing spaces unless stripped, which would
    otherwise silently split one category into two.
  * Mostly RANDOM queries over the 3 tables, plus a small "stratified"
    slice shaped like the real queries this project evaluates on
    (Q1, Q3, Q6 - the only official TPC-H queries that touch only
    customer/orders/lineitem; every other TPC-H query needs supplier,
    part, partsupp, nation or region, which are out of scope here).
    This stratified slice is kept small (~10% of the data) on purpose:
    enough for the model to have seen that shape, too little to just
    memorize it instead of learning the general pattern.
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

if __name__ == '__main__':
    N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
else:
    N_RANDOM = 1500

STATEMENT_TIMEOUT_MS = 15_000
SANITY_CAP_ROWS = 20_000_000

# --------------------------------------------------------------------------
# Schema: 3 tables, 2 joins, 5 filter columns. Everything the generator
# needs is written right here - nothing imported from schema.py.
# --------------------------------------------------------------------------
TABLES = ["customer", "orders", "lineitem"]

JOIN_EDGES = [
    ("orders",   "o_custkey",  "customer", "c_custkey"),
    ("lineitem", "l_orderkey", "orders",   "o_orderkey"),
]

# (table, column, kind) - the 5 columns a generated query may filter on.
PRED_COLS = [
    ("customer", "c_mktsegment", "cat"),
    ("orders",   "o_orderdate",  "date"),
    ("orders",   "o_totalprice", "num"),
    ("lineitem", "l_shipdate",   "date"),   # <-- add this
    ("lineitem", "l_quantity",   "num"),
    ("lineitem", "l_discount",   "num"),
]

NEIGHBORS = {t: set() for t in TABLES}
for a, _, b, _ in JOIN_EDGES:
    NEIGHBORS[a].add(b)
    NEIGHBORS[b].add(a)

# --------------------------------------------------------------------------
# A small batch of queries shaped like the real TPC-H queries this project
# evaluates on (see evaluate_tpch.py). Same tables/columns, random values -
# so the model learns the SHAPE, not the exact test literals. Only Q1, Q3,
# Q6 are listed because those are the only official TPC-H queries whose
# tables are a subset of {customer, orders, lineitem}; Q2/Q4/Q5/Q7-Q22 all
# need supplier/part/partsupp/nation/region, which this project doesn't use.
STRATIFIED_SHAPES = [
    ("q1-like", ["lineitem"],
     [("lineitem", "l_quantity", "num", "range"),
      ("lineitem", "l_discount", "num", "range")], 60),
    ("q3-like", ["customer", "orders", "lineitem"],
     [("customer", "c_mktsegment", "cat", "eq"),
      ("orders",   "o_orderdate",  "date", "range")], 90),
    ("q6-like", ["lineitem"],
     [("lineitem", "l_discount", "num", "range"),
      ("lineitem", "l_quantity", "num", "range")], 60),
]

TABLE_COLS = [f"has_{t}" for t in TABLES]                       # 3
JOIN_COLS = [f"join_{a}_{b}" for a, _, b, _ in JOIN_EDGES]        # 2
PRED_FEATURE_COLS = []
for t, c, _ in PRED_COLS:
    PRED_FEATURE_COLS += [f"{t}_{c}_filter_used",
                          f"{t}_{c}_value_lo",
                          f"{t}_{c}_value_hi"]                   # 5 x 3 = 15
FEATURE_COLS = TABLE_COLS + JOIN_COLS + PRED_FEATURE_COLS         # 20
FEAT_DIM = len(FEATURE_COLS)

TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
PRED_IDX = {(t, c): i for i, (t, c, _) in enumerate(PRED_COLS)}


def norm(v, lo, hi):
    return (v - lo) / (hi - lo + 1e-9)


def load_col_stats(cur):
    stats = {}
    for t, c, kind in PRED_COLS:
        if kind == "cat":
            cur.execute(f'SELECT DISTINCT "{c}" FROM {t} ORDER BY 1')
            # FIX: c_mktsegment is CHAR(n) in TPC-H, so PostgreSQL returns
            # values padded with trailing spaces ('BUILDING  '). Without
            # stripping, that padded string wouldn't match what you'd type
            # in a WHERE clause, and could even look like a separate
            # category. strip() keeps domain order stable either way.
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
    n = rng.choices([1, 2, 3], weights=[2, 3, 4])[0]  # favor the full 3-table chain
    tables = [rng.choice(TABLES)]
    while len(tables) < n:
        cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
        if not cands:
            break
        tables.append(rng.choice(cands))
    return tables


def _one_predicate(rng, col_stats, t, c, kind, mode=None):
    """Returns ((table, col), value_lo, value_hi, sql_fragment).
    value_lo == value_hi means an exact match; value_lo < value_hi means a
    range. mode forces "eq" or "range" (used by the stratified shapes);
    mode=None picks randomly (40% exact match for numeric/date)."""
    s = col_stats[(t, c)]
    if kind == "cat":
        dom = s["domain"]
        v = rng.choice(dom)
        pos = (dom.index(v) + 1) / (len(dom) + 1)
        return ((t, c), pos, pos, f"{c} = '{v}'")
    use_eq = (mode == "eq") or (mode is None and rng.random() < 0.4)
    if use_eq:
        v = rng.uniform(s["min"], s["max"])
        if kind == "date":
            v = int(round(v))
            sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
        else:
            sql = f"{c} = {v:.2f}"
        nv = norm(v, s["min"], s["max"])
        return ((t, c), nv, nv, sql)
    a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
    if a > b:
        a, b = b, a
    if kind == "date":
        a, b = int(a), int(b)
        sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
    else:
        sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
    return ((t, c), norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql)


def sample_predicates(rng, col_stats, tables, max_preds=3):
    cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
    rng.shuffle(cols)
    return [_one_predicate(rng, col_stats, t, c, kind)
            for t, c, kind in cols[: rng.randint(0, max_preds)]]


def stratified_predicates(rng, col_stats, forced):
    return [_one_predicate(rng, col_stats, t, c, kind, mode) for t, c, kind, mode in forced]


def build_where(joins, pred_sqls):
    wheres = [f"{a}.{ac} = {b}.{bc}" for a, ac, b, bc in joins] + pred_sqls
    return " WHERE " + " AND ".join(wheres) if wheres else ""


def featurize(tables, joins, preds):
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
            preds = stratified_predicates(rng, col_stats, forced)
            result = run_one(cur, tables, preds, source=label)
            if result is None:
                continue
            if result == "REJECTED":
                rejected += 1
                continue
            row, pg_est, actual = result
            out_rows.append(row)
            pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
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
        pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
        done += 1
        if done % 50 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta_min = (N_RANDOM - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  random: {done}/{N_RANDOM} | rejected: {rejected} | "
                  f"PG median q-error so far: {statistics.median(pg_qerrs):.2f} | "
                  f"~{rate:.1f}/s | ETA {eta_min:.1f} min")

    fields = FEATURE_COLS + ["query_sql", "pg_est_rows", "actual_rows", "source"]
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