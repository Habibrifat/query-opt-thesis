from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "results" / "training_data.csv"
MODEL_OUT = BASE / "results" / "lgbm_model.txt"
SEED = 42


def qerror(pred, actual):
    pred = np.maximum(pred, 1)
    actual = np.maximum(actual, 1)
    return np.maximum(pred / actual, actual / pred)


def main():
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c.startswith("f")]
    X = df[feats].values.astype(np.float32)
    y = np.log1p(df["actual_rows"].values.astype(np.float32))
    pg = df["pg_est_rows"].values.astype(np.float64)

    # SAME 80/20 split as the MLP script -> fair model comparison
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X[tr_idx], y[tr_idx])

    pred_rows = np.expm1(model.predict(X[val_idx])).clip(min=1)
    act = df["actual_rows"].values[val_idx]
    q_ml = qerror(pred_rows, act)
    q_pg = qerror(pg[val_idx], act)

    print("=== LightGBM vs PostgreSQL (validation: unseen 20%) ===")
    print(f"{'':<16}{'LightGBM':>12}{'PostgreSQL':>12}")
    print(f"{'median q-error':<16}{np.median(q_ml):>12.2f}{np.median(q_pg):>12.2f}")
    print(f"{'p95 q-error':<16}{np.percentile(q_ml, 95):>12.2f}{np.percentile(q_pg, 95):>12.2f}")
    print(f"{'max q-error':<16}{q_ml.max():>12.2f}{q_pg.max():>12.2f}")

    model.booster_.save_model(str(MODEL_OUT))
    print(f"\nModel saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()