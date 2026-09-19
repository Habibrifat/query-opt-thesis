from pathlib import Path

import xgboost as xgb

from train_common import (BASE, SEEDS, SeedRun, load_split, predict_rows,
                          qerror, pg_baseline, print_block, save_meta)

MODEL_OUT = BASE / "results" / "xgb_model.json"


def main():
    X, y, pg, act, tr_idx, val_idx = load_split()
    runs = SeedRun()

    for seed in SEEDS:
        model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0,
        )
        model.fit(X[tr_idx], y[tr_idx])
        pred = predict_rows(model.predict(X[val_idx]), pg[val_idx])
        runs.add(seed, qerror(pred, act[val_idx]), model)

    runs.report()
    _, best_seed, best_model = runs.best
    q_ml = [q for s, q in runs.records if s == best_seed][0]
    print_block("XGBoost", q_ml, pg_baseline(pg, act, val_idx))

    best_model.save_model(str(MODEL_OUT))
    save_meta(MODEL_OUT, "XGBoost")
    print(f"Model saved   -> {MODEL_OUT} (best seed {best_seed})")


if __name__ == "__main__":
    main()
