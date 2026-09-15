from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "results" / "training_data.csv"
MODEL_OUT = BASE / "results" / "model.pt"
SEED = 42


def qerror(pred, actual):
    pred = np.maximum(pred, 1)
    actual = np.maximum(actual, 1)
    return np.maximum(pred / actual, actual / pred)


def main():
    df = pd.read_csv(CSV)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = np.log1p(df["actual_rows"].values.astype(np.float32))  # log-space: multiplicative errors -> additive
    pg = df["pg_est_rows"].values.astype(np.float64)

    # 80/20 train/validation split (validation = unseen queries)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X[tr_idx])
    y_tr = torch.tensor(y[tr_idx])
    X_val = torch.tensor(X[val_idx])

    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_epochs, batch = 150, 64
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        tot, nb = 0.0, 0
        for i in range(0, len(X_tr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[b]).squeeze(-1), y_tr[b])
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 25 == 0:
            print(f"epoch {ep+1:3d} | train MSE (log-space): {tot/nb:.4f}")

    model.eval()
    with torch.no_grad():
        pred_rows = np.expm1(model(X_val).squeeze(-1).numpy()).clip(min=1)

    act = df["actual_rows"].values[val_idx]
    q_ml = qerror(pred_rows, act)
    q_pg = qerror(pg[val_idx], act)

    print("\n=== Validation set (20% unseen queries) ===")
    print(f"{'':<16}{'ML model':>12}{'PostgreSQL':>12}")
    print(f"{'median q-error':<16}{np.median(q_ml):>12.2f}{np.median(q_pg):>12.2f}")
    print(f"{'p95 q-error':<16}{np.percentile(q_ml, 95):>12.2f}{np.percentile(q_pg, 95):>12.2f}")
    print(f"{'max q-error':<16}{q_ml.max():>12.2f}{q_pg.max():>12.2f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nModel saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()