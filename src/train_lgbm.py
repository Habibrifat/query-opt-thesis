from pathlib import Path

import lightgbm as lgb

from train_common import (BASE, SEEDS, SeedRun, load_split, predict_rows,
                          qerror, pg_baseline, print_block, save_meta)

MODEL_OUT = BASE / "results" / "lgbm_model.txt"


def main():
    X, y, pg, act, tr_idx, val_idx = load_split()
    runs = SeedRun()

    for seed in SEEDS:
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        model.fit(X[tr_idx], y[tr_idx])
        pred = predict_rows(model.predict(X[val_idx]), pg[val_idx])
        runs.add(seed, qerror(pred, act[val_idx]), model)

    runs.report()
    _, best_seed, best_model = runs.best
    q_ml = [q for s, q in runs.records if s == best_seed][0]
    print_block("LightGBM", q_ml, pg_baseline(pg, act, val_idx))

    best_model.booster_.save_model(str(MODEL_OUT))
    save_meta(MODEL_OUT, "LightGBM")
    print(f"Model saved   -> {MODEL_OUT} (best seed {best_seed})")


if __name__ == "__main__":
    main()
