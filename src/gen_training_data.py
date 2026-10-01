# import csv
# import random
# import statistics
# import sys
# from datetime import date
# from pathlib import Path

# import psycopg2
# from config import DB_CONFIG
# from schema import TABLES, JOIN_EDGES, PRED_COLS

# BASE = Path(__file__).resolve().parent.parent
# OUT = BASE / "results" / "training_data.csv"
# SEED = 42
# # N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

# if __name__ == '__main__':
#     N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
# else:
#     N_QUERIES = 1000 # Default value when imported

# TABLE_IDX = {t: i for i, t in enumerate(TABLES)}
# JOIN_IDX = {e: i for i, e in enumerate(JOIN_EDGES)}
# PRED_IDX = {c: i for i, c in enumerate([(t, c) for t, c, _ in PRED_COLS])}
# FEAT_DIM = len(TABLES) + len(JOIN_EDGES) + 4 * len(PRED_COLS)

# NEIGHBORS = {t: set() for t in TABLES}
# for a, _, b, _ in JOIN_EDGES:
#     NEIGHBORS[a].add(b)
#     NEIGHBORS[b].add(a)


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
#     # n = rng.choices([1, 2, 3], weights=[3, 4, 3])[0]
#     n = rng.choices([1, 2, 3, 4, 5], weights=[3, 3, 2, 1, 1])[0]
#     # n = rng.choices([1, 2, 3, 4, 5], weights=[5, 4, 2, 1, 1])[0]
#     tables = [rng.choice(TABLES)]
#     while len(tables) < n:
#         cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
#         if not cands:
#             break
#         tables.append(rng.choice(cands))
#     return tables


# def sample_predicates(rng, col_stats, tables, max_preds=3):
#     cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
#     rng.shuffle(cols)
#     preds = []
#     for t, c, kind in cols[: rng.randint(0, max_preds)]:
#         s = col_stats[(t, c)]
#         if kind == "cat":
#             dom = s["domain"]
#             v = rng.choice(dom)
#             pos = (dom.index(v) + 1) / (len(dom) + 1)
#             preds.append(((t, c), "eq", pos, 0.0, f"{c} = '{v}'"))
#         elif rng.random() < 0.4:  # equality on numeric/date
#             v = rng.uniform(s["min"], s["max"])
#             if kind == "date":
#                 v = int(round(v))
#                 sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
#             else:
#                 sql = f"{c} = {v:.2f}"
#             preds.append(((t, c), "eq", norm(v, s["min"], s["max"]), 0.0, sql))
#         else:  # range predicate
#             a, b = rng.uniform(s["min"], s["max"]), rng.uniform(s["min"], s["max"])
#             if a > b:
#                 a, b = b, a
#             if kind == "date":
#                 a, b = int(a), int(b)
#                 sql = f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}' AND DATE '{date.fromordinal(b).isoformat()}'"
#             else:
#                 sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
#             preds.append(((t, c), "range", norm(a, s["min"], s["max"]), norm(b, s["min"], s["max"]), sql))
#     return preds


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


# def main():
#     rng = random.Random(SEED)  # fixed seed = reproducible dataset
#     conn = psycopg2.connect(**DB_CONFIG)
#     conn.autocommit = True
#     cur = conn.cursor()
#     col_stats = load_col_stats(cur)
#     print(f"Feature dim: {FEAT_DIM}  |  Generating {N_QUERIES} queries...")

#     out_rows, pg_qerrs, done = [], [], 0
#     for i in range(1, N_QUERIES + 1):
#         tables = sample_tables(rng)
#         joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
#         preds = sample_predicates(rng, col_stats, tables)
#         body = build_body(tables, joins, [p[4] for p in preds])
#         try:
#             cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
#             pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
#             cur.execute("SELECT COUNT(*) " + body)
#             actual = cur.fetchone()[0]
#         except psycopg2.Error as ex:
#             print("skipped a query:", ex)
#             continue
#         out_rows.append(featurize(tables, joins, preds) + [body, pg_est, actual])
#         pg_qerrs.append(max(max(actual, 1) / pg_est, pg_est / max(actual, 1)))
#         done += 1
#         # if i % 100 == 0:
#         # To this:
#         if i % 10 == 0:
#             print(f"  {i}/{N_QUERIES} done | PostgreSQL median q-error so far: {statistics.median(pg_qerrs):.2f}")

#     fields = [f"f{i}" for i in range(FEAT_DIM)] + ["query", "pg_est_rows", "actual_rows"]
#     with open(OUT, "w", newline="") as fp:
#         w = csv.writer(fp)
#         w.writerow(fields)
#         w.writerows(out_rows)
#     conn.close()
#     print(f"\nSaved {done} examples -> {OUT}")
#     print(f"PostgreSQL baseline on generated set: median q-error = {statistics.median(pg_qerrs):.2f}")


# if __name__ == "__main__":
#     main()



"""Training-data generator (fast, deduplicated, balanced version).

WHY THIS VERSION
  1. FAST   : generate ALL query specs first (pure Python, instant), dedupe
              them, then label (EXPLAIN + COUNT(*)) in parallel with a pool
              of DB connections. Labelling is the slow part - parallelize it.
  2. CLEAN  : identical query bodies are generated only once (dedupe).
  3. FAIR   : balanced table-count weights [2,2,2,2,2] so 4-5 table joins
              (your join-order evaluation uses them) are well represented.
  4. HARD   : range selectivities are sampled in LOG space (10^-4 .. 10^0)
              instead of uniform endpoints, so the model sees easy AND hard
              predicates. Q-error is multiplicative, so this matters.
  5. SAFE   : statement_timeout per worker; queries that error are skipped;
              rows with actual_rows = 0 are dropped (they teach a trivial
              "predict 0" pattern and inflate q-error math).

Public interface (build_body / featurize / load_col_stats / norm / JOIN_EDGES /
FEAT_DIM / N_QUERIES) is UNCHANGED - train_*.py and evaluate_*.py keep working.

Usage:
    python src/gen_training_data.py 20000
"""
import csv
import math
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import psycopg2
from config import DB_CONFIG
from schema import TABLES, JOIN_EDGES, PRED_COLS

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "training_data.csv"
SEED = 42
WORKERS = 8               # DB connections used in parallel (<= your max_connections)
TIMEOUT = "60s"           # skip queries whose COUNT(*) is slower than this
DROP_ZERO = True          # drop rows whose actual row count is 0

if __name__ == "__main__":
    N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
else:
    N_QUERIES = 1000

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


# ---------------- spec generation (fast, pure Python) ----------------

def sample_tables(rng):
    # balanced: your join-order eval uses 5-table queries, so train on them too
    n = rng.choices([1, 2, 3, 4, 5], weights=[2, 2, 2, 2, 2])[0]
    tables = [rng.choice(TABLES)]
    while len(tables) < n:
        cands = [x for t in tables for x in NEIGHBORS[t] if x not in tables]
        if not cands:
            break
        tables.append(rng.choice(cands))
    return tables


def log_uniform_range(rng, s):
    """Pick [a, b] with selectivity uniform in LOG space: 10^-4 .. 10^0."""
    lo, hi = s["min"], s["max"]
    span = hi - lo
    sel = 10 ** rng.uniform(-4.0, 0.0)          # 0.01% .. 100% of the column
    a = rng.uniform(lo, hi - sel * span)
    b = a + sel * span
    return a, b


def sample_predicates(rng, col_stats, tables, max_preds=3):
    cols = [(t, c, k) for t, c, k in PRED_COLS if t in tables]
    rng.shuffle(cols)
    preds = []
    for t, c, kind in cols[: rng.randint(0, max_preds)]:
        s = col_stats[(t, c)]
        if kind == "cat":
            dom = s["domain"]
            v = rng.choice(dom)
            pos = (dom.index(v) + 1) / (len(dom) + 1)
            preds.append(((t, c), "eq", pos, 0.0, f"{c} = '{v}'"))
        elif rng.random() < 0.4:  # equality on numeric/date
            v = rng.uniform(s["min"], s["max"])
            if kind == "date":
                v = int(round(v))
                sql = f"{c} = DATE '{date.fromordinal(v).isoformat()}'"
            else:
                sql = f"{c} = {v:.2f}"
            preds.append(((t, c), "eq", norm(v, s["min"], s["max"]), 0.0, sql))
        else:  # range predicate, log-space selectivity
            a, b = log_uniform_range(rng, s)
            if kind == "date":
                a, b = int(a), int(b)
                if a == b:
                    b = a + 1
                sql = (f"{c} BETWEEN DATE '{date.fromordinal(a).isoformat()}'"
                       f" AND DATE '{date.fromordinal(b).isoformat()}'")
            else:
                sql = f"{c} BETWEEN {a:.2f} AND {b:.2f}"
            preds.append(((t, c), "range", norm(a, s["min"], s["max"]),
                          norm(b, s["min"], s["max"]), sql))
    return preds


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


def make_specs(rng, col_stats, n_queries):
    """Generate candidate specs and dedupe identical SQL bodies."""
    specs, seen = [], set()
    attempts = 0
    while len(specs) < n_queries and attempts < n_queries * 3:
        attempts += 1
        tables = sample_tables(rng)
        joins = [e for e in JOIN_EDGES if e[0] in tables and e[2] in tables]
        preds = sample_predicates(rng, col_stats, tables)
        body = build_body(tables, joins, [p[4] for p in preds])
        if body in seen:
            continue
        seen.add(body)
        specs.append((tables, joins, preds, body))
    return specs


# ---------------- parallel labelling ----------------

_tls = threading.local()


def _conn():
    if not hasattr(_tls, "c") or _tls.c.closed:
        _tls.c = psycopg2.connect(**DB_CONFIG)
        _tls.c.autocommit = True
        _tls.c.cursor().execute(f"SET statement_timeout = '{TIMEOUT}'")
    return _tls.c


def label_spec(spec):
    tables, joins, preds, body = spec
    try:
        cur = _conn().cursor()
        cur.execute("EXPLAIN (FORMAT JSON) SELECT * " + body)
        pg_est = cur.fetchone()[0][0]["Plan"]["Plan Rows"]
        cur.execute("SELECT COUNT(*) " + body)
        actual = cur.fetchone()[0]
    except psycopg2.Error:
        return None
    return featurize(tables, joins, preds) + [body, pg_est, actual]


def main():
    rng = random.Random(SEED)
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    col_stats = load_col_stats(conn.cursor())
    conn.close()
    print(f"Feature dim: {FEAT_DIM} | target: {N_QUERIES} unique queries")

    specs = make_specs(rng, col_stats, N_QUERIES)
    print(f"Generated {len(specs)} unique specs "
          f"(dedupe saved {N_QUERIES - len(specs)} duplicate COUNT(*) runs)")

    out_rows, skipped = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, res in enumerate(pool.map(label_spec, specs), start=1):
            if res is None:
                skipped += 1
            elif DROP_ZERO and res[-1] == 0:
                skipped += 1
            else:
                out_rows.append(res)
            if i % 200 == 0 or i == len(specs):
                print(f"  labelled {i}/{len(specs)} | kept {len(out_rows)} "
                      f"| skipped {skipped}")

    fields = [f"f{i}" for i in range(FEAT_DIM)] + ["query", "pg_est_rows", "actual_rows"]
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(fields)
        w.writerows(out_rows)

    qerrs = [max(max(r[-1], 1) / r[-2], r[-2] / max(r[-1], 1)) for r in out_rows]
    print(f"\nSaved {len(out_rows)} examples -> {OUT}")
    print(f"Skipped: {skipped} (errors/timeouts/zero-actual)")
    print(f"PostgreSQL baseline on generated set: median q-error = "
          f"{statistics.median(qerrs):.2f}")


if __name__ == "__main__":
    main()
