from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
csv_path = BASE / "results" / "baseline_nodes.csv"
if not csv_path.exists():
    raise SystemExit("baseline_nodes.csv not found - run collect_baseline.py first")

df = pd.read_csv(csv_path)

est = df["estimated_rows"].clip(lower=1)
act = df["actual_rows"].clip(lower=1)
q = pd.concat([est / act, act / est], axis=1).max(axis=1)

print(f"Plan nodes analyzed : {len(df)}")
print(f"Median Q-error      : {q.median():.2f}")
print(f"P90  Q-error        : {q.quantile(0.90):.2f}")
print(f"P95  Q-error        : {q.quantile(0.95):.2f}")
print(f"Max  Q-error        : {q.max():.2f}")

per_query = df.assign(qerror=q).groupby("query")["qerror"].median().round(2)
print("\nPer query (median):")
print(per_query)
per_query.to_csv(BASE / "results" / "baseline_qerror_per_query.csv")
print("\nSaved -> results/baseline_qerror_per_query.csv")