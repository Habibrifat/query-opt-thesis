"""Fixed MLP trainer.

Merge of the old train_mlp.py + train_mlp_multiseed.py:
  * multi-seed (SEEDS from train_common), best model saved
  * ratio target -> prediction = pg_est * exp(model output)
  * epoch printing kept but only every 25 epochs
The old train_mlp_multiseed.py can be DELETED.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_common import (BASE, SEEDS, SeedRun, load_split, predict_rows,
                          qerror, pg_baseline, print_block, save_meta)

MODEL_OUT = BASE / "results" / "mlp_model.pt"
EPOCHS, BATCH = 150, 64
VERBOSE_EPOCHS = True   # set False for a clean, seed-only log


def build_model(feat_dim):
    return nn.Sequential(
        nn.Linear(feat_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


def main():
    X, y, pg, act, tr_idx, val_idx = load_split()
    runs = SeedRun()

    X_tr = torch.tensor(X[tr_idx])
    y_tr = torch.tensor(y[tr_idx])
    X_val = torch.tensor(X[val_idx])
    act_val = act[val_idx]
    pg_val = pg[val_idx]

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_model(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(X_tr))
            tot, nb = 0.0, 0
            for i in range(0, len(X_tr), BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad()
                loss = loss_fn(model(X_tr[b]).squeeze(-1), y_tr[b])
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if VERBOSE_EPOCHS and (ep + 1) % 25 == 0:
                print(f"  seed {seed} epoch {ep+1:3d} | "
                      f"train MSE: {tot/nb:.4f}")

        model.eval()
        with torch.no_grad():
            raw = model(X_val).squeeze(-1).numpy()
        pred = predict_rows(raw, pg_val)
        runs.add(seed, qerror(pred, act_val), model)

    runs.report()
    _, best_seed, best_model = runs.best
    q_ml = [q for s, q in runs.records if s == best_seed][0]
    print_block("MLP", q_ml, pg_baseline(pg, act, val_idx))

    torch.save(best_model.state_dict(), MODEL_OUT)
    save_meta(MODEL_OUT, "MLP")
    print(f"Model saved   -> {MODEL_OUT} (best seed {best_seed})")


if __name__ == "__main__":
    main()
