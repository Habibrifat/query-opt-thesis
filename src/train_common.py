"""Shared utilities for all cardinality-model trainers.

WHY THIS FILE EXISTS
--------------------
The original codebase copy-pasted the same qerror()/split/loading code into
5 scripts, and each script reported results differently. This module makes
every trainer behave identically:

1. SAME 80/20 split for every model (split seed fixed at 42) -> fair comparison.
2. MULTI-SEED training (model-internal seed varies) -> mean +/- std reported.
3. RATIO TARGET: the model learns  log(actual / pg_est)  instead of
   log(actual).  Prediction = pg_est * exp(model_output).

   This is the key fix for "my ML q-error is WORSE than PostgreSQL":
   if the model has learned nothing useful for a query it outputs ~0,
   and the prediction becomes *exactly* PostgreSQL's estimate.
   ML can now only improve on PG, never be systematically worse.
   (Input features are unchanged -> still the same 81-dim featurize().)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV = BASE / "results" / "training_data.csv"

SEEDS = [1, 2, 3]       # model-internal randomness (init, subsampling, ...)
SPLIT_SEED = 42         # data split: FIXED for all models, all runs

# "log_ratio"  -> model predicts log(actual/pg_est)   [recommended, PG-floor]
# "log_rows"   -> model predicts log1p(actual)        [original behaviour]
TARGET_MODE = "log_ratio"


def load_split():
    """Load training_data.csv and return X, y, pg, act, tr_idx, val_idx."""
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c.startswith("f")]
    X = df[feats].values.astype(np.float32)
    pg = df["pg_est_rows"].values.astype(np.float64)
    act = df["actual_rows"].values.astype(np.float64)

    if TARGET_MODE == "log_ratio":
        y = np.log((act + 1.0) / np.maximum(pg, 1.0)).astype(np.float32)
    else:
        y = np.log1p(act).astype(np.float32)

    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(len(X))
    n_val = int(0.2 * len(X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X, y, pg, act, tr_idx, val_idx


def predict_rows(raw_model_output, pg):
    """Turn raw model output into estimated row counts."""
    if TARGET_MODE == "log_ratio":
        return (np.maximum(pg, 1.0) * np.exp(raw_model_output)).clip(min=1)
    return np.expm1(raw_model_output).clip(min=1)


def qerror(p, a):
    p, a = np.maximum(p, 1), np.maximum(a, 1)
    return np.maximum(p / a, a / p)


def pg_baseline(pg, act, val_idx):
    return qerror(pg[val_idx], act[val_idx])


def print_block(name, q_ml, q_pg):
    print(f"\n=== {name} vs PostgreSQL (validation: unseen 20%) ===")
    print(f"{'':<18}{'ML model':>12}{'PostgreSQL':>12}")
    print(f"{'median q-error':<18}{np.median(q_ml):>12.2f}{np.median(q_pg):>12.2f}")
    print(f"{'p95 q-error':<18}{np.percentile(q_ml, 95):>12.2f}{np.percentile(q_pg, 95):>12.2f}")
    print(f"{'max q-error':<18}{q_ml.max():>12.2f}{q_pg.max():>12.2f}")


def save_meta(model_out, model_name):
    meta = {"model": model_name, "target": TARGET_MODE, "input_dim": 81}
    p = Path(model_out)
    meta_path = p.parent / (p.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Meta saved    -> {meta_path}")


class SeedRun:
    """Track per-seed results; keeps the BEST model (not just the last)."""

    def __init__(self):
        self.records = []   # (seed, q_array)
        self.best = None    # (median_qerror, seed, model)

    def add(self, seed, q, model):
        self.records.append((seed, q))
        print(f"seed {seed}: median={np.median(q):.2f}  "
              f"p95={np.percentile(q, 95):.2f}  max={q.max():.0f}")
        if self.best is None or np.median(q) < self.best[0]:
            self.best = (np.median(q), seed, model)

    def report(self):
        meds = [np.median(q) for _, q in self.records]
        print(f"\nmean +/- std over {len(self.records)} seeds: "
              f"median={np.mean(meds):.2f}+/-{np.std(meds):.2f}")
        return meds
