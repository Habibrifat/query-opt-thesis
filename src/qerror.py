from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "results" / "baseline_nodes.csv")

est = df["estimated_rows"].clip(lower=1)
act = df["actual_rows"].clip(lower=1)
q = pd.concat([est / act, act / est], axis=1).max(axis=1)

print(f"Plan nodes analyzed : {len(df)}")
print(f"Median Q-error      : {q.median():.2f}")
print(f"P95  Q-error        : {q.quantile(0.95):.2f}")
print(f"Max  Q-error        : {q.max():.2f}")
print("\nPer query (median):")
print(df.assign(qerror=q).groupby("query")["qerror"].median())