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
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.feature_store import load_labeled_features  # noqa: E402
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
    p.add_argument("--from-feature-store", action="store_true")
    p.add_argument("--feature-store-limit", type=int, default=5000)
    p.add_argument("--feature-store-threshold", type=int, default=200)
    p.add_argument("--database-url", type=str,
                   default=os.getenv("DATABASE_URL",
                                     "postgresql://bot_user:bot_password@localhost:5432/tradingbot"))
    a = p.parse_args()
    if a.samples < 200:
        print("ERROR: --samples must be >= 200", file=sys.stderr)
        return 1

    rng = np.random.default_rng(a.seed)
    # Bootstrap version doesn't exist on disk → init warns but does not raise.
    cfg = {"MODEL_DIR": "./models", "model_version": f"__bootstrap_{a.seed}"}
    clf = RugClassifier(cfg)
    df, labels = None, None
    if a.from_feature_store:
        result = asyncio.run(load_labeled_features(
            a.database_url, feature_key="rug_v1", limit=a.feature_store_limit))
        if result is not None and result[0].shape[0] >= a.feature_store_threshold:
            X, pnl, _won = result
            if X.shape[1] == len(clf.feature_columns):
                df = pd.DataFrame(X, columns=clf.feature_columns)
                # Heuristic label: catastrophic loss flags a rug. A real label
                # would come from on-chain rug detection.
                labels = (pnl < -0.5).astype(int)
                print(f"✓ Fitting on {len(df)} REAL rows from ai_feature_store (key='rug_v1')")
            else:
                print(f"⚠ rug_v1 width {X.shape[1]} != expected {len(clf.feature_columns)}; "
                      "falling back to synthetic")
        else:
            n = 0 if result is None else result[0].shape[0]
            print(f"⚠ Only {n} real rug_v1 rows (< {a.feature_store_threshold}); "
                  "falling back to synthetic")
    if df is None:
        if not a.from_feature_store:
            print(f"ℹ Fitting on {a.samples} SYNTHETIC rows "
                  "(pass --from-feature-store to use real data when available)")
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
