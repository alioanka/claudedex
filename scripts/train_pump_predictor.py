#!/usr/bin/env python3
"""
Train and persist the PumpPredictor (MVP).

After [quant] commit 6e1dbc9 (MB-19 siblings), PumpPredictor refuses to
predict_pump_probability() until load_model('v1') succeeds against
models/pump_predictor_v1/. This script produces that directory.

No time-series feature store yet, so:
  - tree models fit on synthetic 27-feature rows (extract_features L217-230:
    technical(15) + market(8) + patterns(4) = 27), labels via rule.
  - LSTM gets a 1-epoch placeholder fit so save_model() (which hard-requires
    self.models['lstm'].save() at pump_predictor.py:758) doesn't fail.
    LSTM contributes ~no alpha until real time-series data arrives — operators
    may want to zero its model_weights entry via config.

Replace both _generate_synthetic_*() with feature-store + label queries.

Usage: python scripts/train_pump_predictor.py [--samples N]
                                              [--output-version v1]
                                              [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.models.pump_predictor import PumpPredictor  # noqa: E402

EXPECTED_TREE_FEATURE_COUNT = 27  # extract_features L217-230


def _generate_synthetic_features(n, rng):
    """27-feature matrix + binary pump labels via deterministic rule."""
    buy_p = rng.uniform(0, 1, n)
    sell_p = (1 - buy_p + rng.normal(0, 0.05, n)).clip(0, 1)
    volume_spike = rng.exponential(1.5, n).clip(0, 10)
    momentum = rng.normal(0, 0.1, n)
    technical = np.column_stack([
        rng.uniform(0, 100, n), rng.normal(0, 0.01, n),
        rng.normal(0, 0.01, n), rng.normal(0, 0.005, n),
        rng.lognormal(0, 0.05, (n, 3)), rng.exponential(0.05, n),
        rng.lognormal(0, 0.05, (n, 4)), rng.normal(0, 1e6, n),
        rng.uniform(0, 100, (n, 3)),
    ])
    market = np.column_stack([buy_p, sell_p, buy_p - sell_p,
                              rng.uniform(0, 1, (n, 5))])
    patterns = np.column_stack([volume_spike, momentum,
                                rng.uniform(0, 1, (n, 2))])
    X = np.hstack([technical, market, patterns])
    score = (np.clip(volume_spike / 5, 0, 1) * 0.4
             + np.clip(momentum * 5 + 0.5, 0, 1) * 0.3 + buy_p * 0.3)
    labels = (score > float(np.median(score))).astype(float)
    return X, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--output-version", type=str, default="v1")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.samples < 200:
        print("ERROR: --samples must be >= 200", file=sys.stderr)
        return 1

    rng = np.random.default_rng(a.seed)
    cfg = {"MODEL_DIR": "./models", "model_version": f"__bootstrap_{a.seed}"}
    pp = PumpPredictor(cfg)

    X, y = _generate_synthetic_features(a.samples, rng)
    assert X.shape[1] == EXPECTED_TREE_FEATURE_COUNT, X.shape
    pp.scalers["features"].fit(X)
    Xs = pp.scalers["features"].transform(X)
    print(f"Generated {len(X)} samples, {int(y.sum())} pump / "
          f"{len(y) - int(y.sum())} no-pump.")
    for name in ["xgboost", "lightgbm", "random_forest", "gradient_boost"]:
        pp.models[name].fit(Xs, y)
        acc = float((pp.models[name].predict(Xs).round() == y).mean())
        print(f"  {name:18s} train-acc={acc:.4f}")

    # Minimal LSTM/price-scaler fit so save_model() succeeds end-to-end.
    n_seq = min(a.samples, 2000)
    lstm_X = rng.normal(0, 1, (n_seq, pp.sequence_length,
                               len(pp.price_features))).astype("float32")
    lstm_y = (lstm_X[:, -1, 0] > 0).astype("float32")
    pp.scalers["price"].fit(lstm_X.reshape(-1, lstm_X.shape[-1]))
    pp.models["lstm"].fit(lstm_X, lstm_y, epochs=1, batch_size=64, verbose=0)
    print("LSTM: 1-epoch placeholder (NOT predictive — see docstring).")

    saved = pp.save_model(version=a.output_version)
    print(f"Saved to {saved}")

    # Roundtrip: fresh instance must auto-load and is_loaded()==True.
    verifier = PumpPredictor(
        {"MODEL_DIR": "./models", "model_version": a.output_version})
    if not verifier.is_loaded():
        print("ERROR: roundtrip failed (is_loaded()=False)", file=sys.stderr)
        return 2
    print(f"Roundtrip OK: is_loaded()=True after load_model('{a.output_version}').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
