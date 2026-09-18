from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "results" / "training_data.csv"
MODEL_OUT = BASE / "results" / "rf_model.pkl"
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

    rng = np.random.RandomState(42)          # SAME split as all other models
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]
    act_val = df["actual_rows"].values[val_idx]

    meds, p95s, maxs = [], [], []
    model = None
    for seed in SEEDS:
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        model.fit(X[tr_idx], y[tr_idx])
        pred = np.expm1(model.predict(X[val_idx])).clip(min=1)
        q = qerror(pred, act_val)
        meds.append(np.median(q))
        p95s.append(np.percentile(q, 95))
        maxs.append(q.max())
        print(f"seed {seed}: median={np.median(q):.2f}  p95={np.percentile(q, 95):.2f}  max={q.max():.0f}")

    print(f"\nRandom Forest mean ± std over {len(SEEDS)} seeds:")
    print(f"  median={np.mean(meds):.2f}±{np.std(meds):.2f}  "
          f"p95={np.mean(p95s):.2f}±{np.std(p95s):.2f}  "
          f"max={np.mean(maxs):.0f}±{np.std(maxs):.0f}")

    qpg = qerror(pg[val_idx], act_val)
    print(f"  PostgreSQL (same val set): median={np.median(qpg):.2f}  "
          f"p95={np.percentile(qpg, 95):.2f}  max={qpg.max():.0f}")

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()