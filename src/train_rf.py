from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

from train_common import (BASE, SEEDS, SeedRun, load_split, predict_rows,
                          qerror, pg_baseline, print_block, save_meta)

MODEL_OUT = BASE / "results" / "rf_model.pkl"


def main():
    X, y, pg, act, tr_idx, val_idx = load_split()
    runs = SeedRun()

    for seed in SEEDS:
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                      random_state=seed)
        model.fit(X[tr_idx], y[tr_idx])
        pred = predict_rows(model.predict(X[val_idx]), pg[val_idx])
        runs.add(seed, qerror(pred, act[val_idx]), model)

    runs.report()
    _, best_seed, best_model = runs.best
    q_ml = [q for s, q in runs.records if s == best_seed][0]
    print_block("Random Forest", q_ml, pg_baseline(pg, act, val_idx))

    joblib.dump(best_model, MODEL_OUT)
    save_meta(MODEL_OUT, "RandomForest")
    print(f"Model saved   -> {MODEL_OUT} (best seed {best_seed})")


if __name__ == "__main__":
    main()
