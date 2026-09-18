from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "results" / "training_data.csv"
SEEDS = [1, 2, 3]


def qerror(p, a):
    p, a = np.maximum(p, 1), np.maximum(a, 1)
    return np.maximum(p / a, a / p)


def main():
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c.startswith("f")]
    X = df[feats].values.astype(np.float32)
    y = np.log1p(df["actual_rows"].values.astype(np.float32))
    pg = df["pg_est_rows"].values.astype(np.float64)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        X_tr, y_tr = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
        model = nn.Sequential(
            nn.Linear(X.shape[1], 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for ep in range(150):
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = loss_fn(model(X_tr[b]).squeeze(-1), y_tr[b])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = np.expm1(model(torch.tensor(X[val_idx])).squeeze(-1).numpy()).clip(min=1)
        q = qerror(pred, df["actual_rows"].values[val_idx])
        rows.append([seed, np.median(q), np.percentile(q, 95), q.max()])
        print(f"seed {seed}: median={np.median(q):.2f}  p95={np.percentile(q, 95):.2f}  max={q.max():.0f}")

    r = np.array([row[1:] for row in rows])
    print(f"\nmean +- std: median={r[:,0].mean():.2f}+-{r[:,0].std():.2f}  "
          f"p95={r[:,1].mean():.2f}+-{r[:,1].std():.2f}  max={r[:,2].mean():.0f}+-{r[:,2].std():.0f}")
    qpg = qerror(pg[val_idx], df["actual_rows"].values[val_idx])
    print(f"PostgreSQL:  median={np.median(qpg):.2f}  p95={np.percentile(qpg, 95):.2f}  max={qpg.max():.0f}")


if __name__ == "__main__":
    main()