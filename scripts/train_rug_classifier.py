#!/usr/bin/env python3
"""
Train and persist the RugClassifier (MVP).

After [quant] commit 6e1dbc9 (MB-19 siblings), RugClassifier refuses to
predict() until load_model('v1') succeeds against
models/rug_classifier_v1/. This script produces that directory.

No rug-feature store exists yet — until it does, this fits on synthetic
rows with deterministic-rule labels (gives the ensemble learnable signal).
Replace _generate_synthetic() with a feature-store query when ready.

Usage: python scripts/train_rug_classifier.py [--samples N]
                                              [--output-version v1]
                                              [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.models.rug_classifier import RugClassifier  # noqa: E402


def _generate_synthetic(n, columns, rng):
    """columns := RugClassifier.feature_columns (rug_classifier.py:48-96)."""
    df = pd.DataFrame({c: rng.uniform(0, 1, n) for c in columns})
    df["unique_holders"] = rng.integers(10, 50_000, n).astype(float)
    df["contract_age_hours"] = rng.exponential(200, n)
    df["buy_sell_ratio"] = rng.lognormal(0, 0.5, n)
    df["volume_liquidity_ratio"] = rng.exponential(0.5, n)
    df["price_volatility"] = rng.exponential(0.1, n)
    # Rug label: unlocked + dev-heavy + holder-concentrated.
    score = ((1 - df["liquidity_locked_percentage"]) * 0.4
             + df["dev_wallet_percentage"] * 0.3
             + df["holder_concentration_top10"] * 0.2
             + df["honeypot_characteristics"] * 0.1)
    labels = (score > float(np.median(score))).astype(int).values
    return df, labels


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
    # Bootstrap version doesn't exist on disk → init warns but does not raise.
    cfg = {"MODEL_DIR": "./models", "model_version": f"__bootstrap_{a.seed}"}
    clf = RugClassifier(cfg)
    df, labels = _generate_synthetic(a.samples, clf.feature_columns, rng)
    print(f"Generated {len(df)} samples, {int(labels.sum())} rug / "
          f"{len(labels) - int(labels.sum())} safe.")
    results = clf.train(df, labels)
    saved = clf.save_model(version=a.output_version)
    ens = results.get("ensemble_metrics", {})
    print(f"Saved to {saved}")
    print(f"Ensemble ROC-AUC={ens.get('roc_auc', float('nan')):.4f} "
          f"acc={ens.get('accuracy', float('nan')):.4f} "
          f"f1={ens.get('f1', float('nan')):.4f}")

    # Roundtrip: fresh instance must auto-load and is_loaded()==True. This is
    # the sanity check that the commit-1 load-or-refuse fix picks this up.
    verifier = RugClassifier(
        {"MODEL_DIR": "./models", "model_version": a.output_version})
    if not verifier.is_loaded():
        print("ERROR: roundtrip failed (is_loaded()=False)", file=sys.stderr)
        return 2
    print(f"Roundtrip OK: is_loaded()=True after load_model('{a.output_version}').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
