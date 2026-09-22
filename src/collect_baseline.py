import csv
import json
from pathlib import Path

import psycopg2
from config import DB_CONFIG

BASE = Path(__file__).resolve().parent.parent
QUERIES = BASE / "queries"
PLANS = BASE / "results" / "plans"
PLANS.mkdir(parents=True, exist_ok=True)
OUT = BASE / "results" / "baseline_nodes.csv"


def walk(node, query_name, rows):
    est, act = node.get("Plan Rows"), node.get("Actual Rows")
    if est is not None and act is not None:
        rows.append({
            "query": query_name,
            "node_type": node.get("Node Type"),
            "relation": node.get("Relation Name"),
            "estimated_rows": est,
            "actual_rows": act,
        })
    for child in node.get("Plans", []):
        walk(child, query_name, rows)


def main():
    rows, failed = [], []
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True                      # FIX 1: isolate each query
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")  # FIX 2: avoid 10-min hangs
    for qf in sorted(QUERIES.glob("*.sql")):
        sql = qf.read_text(encoding="utf-8")
        try:                                    # FIX 3: one bad query != dead run
            cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql)
            plan = cur.fetchone()[0]
            (PLANS / f"{qf.stem}.json").write_text(json.dumps(plan, indent=2))
            walk(plan[0]["Plan"], qf.stem, rows)
            print(f"{qf.stem}: plan captured")
        except psycopg2.Error as ex:
            failed.append(qf.stem)
            print(f"{qf.stem}: FAILED - {ex}")
    conn.close()
    if not rows:                                # FIX 4: don't crash on empty
        print("No plans captured at all - check queries and DB.")
        return
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} plan nodes -> {OUT}")
    if failed:
        print(f"Failed queries: {failed}")


if __name__ == "__main__":
    main()