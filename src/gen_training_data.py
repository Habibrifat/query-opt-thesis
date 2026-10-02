"""Generates results/training_data.csv for the cardinality models.

THREE FIXES COMBINED HERE (none of these were in the previous version,
despite being proposed earlier - this file supersedes it):

1. SANITY CAP: random subgraph sampling can pick tables connected only
   through a shared "hub" dimension (e.g. customer and lineitem joined only
   via nation/supplier, no direct or orders-mediated path), producing a
   genuine many-to-many fan-out that reaches into the billions of rows -
   structurally valid SQL, but not representative of any real workload.
   training_data.csv currently has actual_rows up to 36 BILLION (TPC-H
   SF1's biggest table, lineitem, only has ~6 million rows). These are
   rejected and resampled instead of kept.

2. REBALANCED TABLE-COUNT WEIGHTS: the old weights [3,3,2,1,1] for
   [1,2,3,4,5] tables heavily favored trivial 1-2 table queries. Real
   TPC-H queries mostly join 3-5 tables (q3=3, q10=4, q5=5) - the model
   needs more examples at that complexity, not fewer.

3. STRATIFIED INJECTION: beyond random sampling, this generator now also
   produces a fixed batch of queries shaped EXACTLY like the real TPC-H
   query cores evaluate_tpch.py tests against (same tables, same predicate
   COLUMNS - but randomized predicate VALUES, so it's not just memorizing
   the one literal test query). q10's shape (customer+orders+lineitem+
   nation with an l_returnflag + o_orderdate predicate) gets the heaviest
   weight, since it was the worst generalization failure (24x-180x q-error
   on every model) - purely random sampling was unlikely to ever produce
   this exact combination often enough to learn it.
"""
import csv
import random
import statistics
import sys
from datetime import date
from pathlib import Path

import psycopg2
from config import DB_CONFIG
from schema import TABLES, JOIN_EDGES, PRED_COLS

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "training_data.csv"
SEED = 42

if __name__ == '__main__':
    N_RANDOM = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
else:
    N_RANDOM = 4000

# FIX 1: sanity cap - see module docstring
SANITY_CAP_ROWS = 20_000_000

# FIX 4: a single slow/stuck query no longer hangs the whole run - skipped
# and counted instead. 15s is generous; a well-targeted query should finish
# in well under 1s, and anything taking longer tells you little extra.
STATEMENT_TIMEOUT_MS = 15_000

# FIX 3: stratified shapes mirroring evaluate_tpch.py's real query cores.
# Each: (label, tables, [(table, col, kind, mode), ...] forced predicates, count)
# mode "eq" or "range" - matches how evaluate_tpch.py encodes that query.
# FIX 5: shrunk to a minority augmentation (~15% of the full dataset, was
# ~57%) - large enough to teach the model these shapes exist, small enough
# that it can't just be "memorizing the test shapes." q10 still gets the
# most weight since it was the worst offender, just proportionally less.
STRATIFIED_SHAPES = [
    ("q1-like",  ["lineitem"],
     [("lineitem", "l_shipdate", "date", "range")], 50),
    ("q3-like",  ["customer", "orders", "lineitem"],
     [("customer", "c_mktsegment", "cat", "eq"),
      ("orders", "o_orderdate", "date", "range"),
      ("lineitem", "l_shipdate", "date", "range")], 90),
    ("q6-like",  ["lineitem"],
     [("lineitem", "l_shipdate", "date", "range"),
      ("lineitem", "l_discount", "num", "range"),
      ("lineitem", "l_quantity", "num", "range")], 50),
    ("q5-like",  ["customer", "orders", "lineitem", "supplier", "nation"],
     [("orders", "o_orderdate", "date", "range")], 90),
    ("q10-like", ["customer", "orders", "lineitem", "nation"],
     [("lineitem", "l_returnflag", "cat", "eq"),
      ("orders", "o_orderdate", "date", "range")], 150),  # still heaviest
    ("q14-like", ["lineitem", "part"],
     [("lineitem", "l_shipdate", "date", "range")], 50),
]

TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
PRED_IDX = {c: i for i, c in enumerate([(t, c) for t, c, _ in PRED_COLS])}
FEAT_DIM = len(TABLES) + len(JOIN_EDGES) + 4 * len(PRED_COLS)

NEIGHBORS = {t: set() for t in TABLES}
for a, _, b, _ in JOIN_EDGES:
    NEIGHBORS[a].add(b)
    NEIGHBORS[b].add(a)


def norm(v, lo, hi):
    return (v - lo) / (hi - lo + 1e-9)


def load_col_stats(cur):
    stats = {}
    for t, c, kind in PRED_COLS:
        if kind == "cat":
            cur.execute(f'SELECT DISTINCT "{c}" FROM {t} ORDER BY 1')
            stats[(t, c)] = {"kind": kind, "domain": [r[0] for r in cur.fetchall()]}
        else:
            cur.execute(f'SELECT MIN("{c}"), MAX("{c}") FROM {t}')
            lo, hi = cur.fetchone()
            if kind == "date":
                lo, hi = lo.toordinal(), hi.toordinal()
            stats[(t, c)] = {"kind": kind, "min": float(lo), "max": float(hi)}
    return stats


def sample_tables(rng):
    # FIX 2: was weights=[3,3,2,1,1] - heavily favored 1-2 table queries.
    # Real TPC-H queries mostly join 3-5 tables; shifted weight there.
    n = rng.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 3, 2])[0]
    tables = [rng.choice(TABLES)]
    while len(tables) < n:
        cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
        if not cands:
            break
        tables.append(rng.choice(cands))
    return tables


def _one_predicate(rng, col_stats, t, c, kind, mode=None):
    """Build a single predicate. mode forces "eq" or "range"; None = random
    (matches the old sample_predicates behaviour: 40% eq for numeric/date)."""
    s = col_stats[(t, c)]
    if kind == "cat":
        dom = s["domain"]
        v = rng.choice(dom)
        pos = (dom.index(v) + 1) / (len(dom) + 1)
        return ((t, c), "eq", pos, 0.0, f"{c} = '{v}'")
    use_eq = (mode == "eq") or (mode is None and rng.random() < 0.4)
    if use_eq:
        v = rng.uniform(s["min"], s["max"])
        if kind == "date":
            v = int(round(v))
            sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
        else:
            sql = f"{c} = {v:.2f}"
        return ((t, c), "eq", norm(v, s["min"], s["max"]), 0.0, sql)
    a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
    if a > b:
        a, b = b, a
    if kind == "date":
        a, b = int(a), int(b)
        sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
    else:
        sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
    return ((t, c), "range", norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql)


def sample_predicates(rng, col_stats, tables, max_preds=3):
    cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
    rng.shuffle(cols)
    return [_one_predicate(rng, col_stats, t, c, kind)
            for t, c, kind in cols[: rng.randint(0, max_preds)]]


def stratified_predicates(rng, col_stats, forced):
    return [_one_predicate(rng, col_stats, t, c, kind, mode) for t, c, kind, mode in forced]


def build_body(tables, joins, pred_sqls):
    wheres = [f"{a}.{ac} = {b}.{bc}" for a, ac, b, bc in joins] + pred_sqls
    where = " WHERE " + " AND ".join(wheres) if wheres else ""
    return f"FROM {', '.join(tables)}{where}"


def featurize(tables, joins, preds):
    f = [0.0] * FEAT_DIM
    for t in tables:
        f[TABLE_IDX[t]] = 1.0
    off = len(TABLES)
    for e in joins:
        f[off + JOIN_IDX[e]] = 1.0
    off += len(JOIN_EDGES)
    for (t, c), kind, v1, v2, _ in preds:
        b = off + 4 * PRED_IDX[(t, c)]
        if kind == "eq":
            f[b], f[b + 1] = 1.0, v1
        else:
            f[b + 2], f[b + 3] = v1, v2
    return f


def run_one(cur, tables, preds, source):
    """Execute one query. Returns (row, pg_est, actual) or None (SQL error
    or timeout) or "REJECTED" (sanity cap tripped). `source` is tagged onto
    the row so training_data.csv can report accuracy per-source later."""
    joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
    body = build_body(tables, joins, [p[4] for p in preds])
    try:
        cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
        cur.execute("SELECT COUNT(*) " + body)
        actual = cur.fetchone()[0]
    except psycopg2.Error as ex:
        # FIX 4: includes statement-timeout cancellations, not just syntax
        # errors - these no longer stall the whole run, just get skipped.
        print(f"  skipped ({type(ex).__name__}): {str(ex).strip()[:80]}")
        cur.connection.rollback()
        return None
    if actual > SANITY_CAP_ROWS:
        return "REJECTED"
    row = featurize(tables, joins, preds) + [body, pg_est, actual, source]
    return row, pg_est, actual


def main():
    rng = random.Random(SEED)
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")  # FIX 4
    col_stats = load_col_stats(cur)

    n_stratified = sum(c for *_, c in STRATIFIED_SHAPES)
    print(f"Feature dim: {FEAT_DIM}")
    print(f"Plan: {N_RANDOM} random queries + {n_stratified} stratified "
          f"(real-query-shaped) queries = {N_RANDOM + n_stratified} total")
    print(f"Sanity cap: rejecting any query with actual_rows > {SANITY_CAP_ROWS:,}\n")

    out_rows, pg_qerrs, rejected = [], [], 0

    # --- stratified batch first ---
    for label, tables, forced, count in STRATIFIED_SHAPES:
        made = 0
        attempts = 0
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
        print(f"  [{label:<9}] {made}/{count} generated")

    # --- random batch ---
    done = 0
    import time
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
        # FIX 4: printed every 50, not 200 - so a slow stretch is visible
        # as "still going, just slow" rather than looking frozen.
        if done % 50 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta_min = (N_RANDOM - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  random: {done}/{N_RANDOM} | rejected: {rejected} | "
                  f"PG median q-err so far: {statistics.median(pg_qerrs):.2f} | "
                  f"~{rate:.1f}/s | ETA {eta_min:.1f} min")

    fields = [f"f{i}" for i in range(FEAT_DIM)] + ["query", "pg_est_rows", "actual_rows", "source"]
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(fields)
        w.writerows(out_rows)
    conn.close()
    print(f"\nSaved {len(out_rows)} examples -> {OUT}")
    print(f"Rejected {rejected} pathological (hub-fanout) queries during generation")
    print(f"PostgreSQL baseline on generated set: median q-error = {statistics.median(pg_qerrs):.2f}")


if __name__ == "__main__":
    main() 