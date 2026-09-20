"""Unified loader for ALL trained cardinality models.

One interface for MLP / LightGBM / RandomForest / XGBoost:
    model = CardModel("mlp")          # or "lgbm" | "rf" | "xgb"
    est   = model.predict_rows(X, pg) # pg = PostgreSQL estimate per query

Ratio handling (the fix from train_common.py): trainers now learn
    y = log(actual / pg_est)
so inference MUST apply   pred = pg_est * exp(raw).
The target is read from results/<model>.meta.json; old models without
meta default to the original expm1 behaviour (a warning is printed).
The MLP architecture is reconstructed from the state_dict itself, so
this loader never hardcodes layer sizes.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

REGISTRY = {
    "mlp":  ("mlp_model.pt",   "torch"),
    "lgbm": ("lgbm_model.txt", "lightgbm"),
    "rf":   ("rf_model.pkl",   "sklearn"),
    "xgb":  ("xgb_model.json", "xgboost"),
}


def _load_meta(artifact):
    meta_path = RESULTS / (Path(artifact).stem + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    print(f"WARNING: {meta_path.name} not found - assuming old 'log_rows' model")
    return {"target": "log_rows", "model": Path(artifact).stem}


def _mlp_from_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    weights = [v for k, v in sd.items() if "weight" in k]
    biases = [v for k, v in sd.items() if "bias" in k]
    layers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(weights) - 1:          # ReLU between hidden layers only
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


class CardModel:
    def __init__(self, name):
        if name not in REGISTRY:
            raise KeyError(f"unknown model '{name}' - choose from {list(REGISTRY)}")
        artifact, kind = REGISTRY[name]
        self.name = name
        self.kind = kind
        self.path = RESULTS / artifact
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found - run the matching src/train_{name}.py first")
        self.meta = _load_meta(artifact)
        self.target = self.meta.get("target", "log_rows")
        if kind == "torch":
            self._m = _mlp_from_state_dict(self.path)
        elif kind == "lightgbm":
            self._m = lgb.Booster(model_file=str(self.path))
        elif kind == "sklearn":
            self._m = joblib.load(self.path)
        elif kind == "xgboost":
            self._m = xgb.XGBRegressor()
            self._m.load_model(str(self.path))

    def raw(self, X):
        """Raw model output (log-ratio or log1p rows), numpy array."""
        X = np.asarray(X, dtype=np.float32)
        if self.kind == "torch":
            with torch.no_grad():
                return self._m(torch.tensor(X)).squeeze(-1).numpy()
        return np.asarray(self._m.predict(X), dtype=np.float64)

    def predict_rows(self, X, pg):
        """Estimated row counts, ratio-corrected. pg = PG estimate (per row of X)."""
        raw = self.raw(X)
        pg = np.maximum(np.asarray(pg, dtype=np.float64), 1.0)
        if self.target == "log_ratio":
            return (pg * np.exp(raw)).clip(min=1)
        return np.expm1(raw).clip(min=1)

    def __repr__(self):
        return f"CardModel({self.name}, target={self.target})"


def available_models():
    """Names of models whose artifact file currently exists in results/."""
    return [n for n, (a, _) in REGISTRY.items() if (RESULTS / a).exists()]
