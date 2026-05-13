#!/usr/bin/env python3
"""
Train and persist the StandardScaler used by trading/strategies/ai_strategy.py.

Background
----------
ai_strategy._extract_features (commit 9d49e77 / MB-19) refuses to predict
without a fitted scaler at models/ai_strategy_scaler.pkl. This script
produces that file.

Approach
--------
Historical engineered features are not persisted in ai_trades / solana_trades —
those tables store trade metadata, not feature vectors. Until a feature-store
layer exists, this script fits the scaler on synthetic feature vectors drawn
from plausible per-feature distributions. Once a feature store lands, replace
this script's data-generation block with a DB query.

Usage
-----
    python scripts/train_ai_strategy_scaler.py [--samples 10000]
                                               [--output models/ai_strategy_scaler.pkl]
                                               [--seed 42]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.feature_store import load_labeled_features  # noqa: E402


# Feature shape MUST stay in sync with
# trading/strategies/ai_strategy.py:_extract_features (lines 282-337).
#
# In current code (audited 2026-05-12), only `returns` returns a list; the other
# indicators (`volatility`, `volume_ratio`, `rsi`, `macd_signal`,
# `pattern_strength`) return scalars. So _extract_features emits:
#   [0]      market_cap
#   [1]      liquidity
#   [2]      holders_count
#   [3]      24h_transactions
#   [4..8]   returns_last5            (list → last-5 slice)
#   [9]      volatility               (scalar)
#   [10]     volume_ratio             (scalar)
#   [11]     rsi                      (scalar)
#   [12]     macd_signal              (scalar)
#   [13]     pattern_strength         (scalar)
#   [14]     social_score
#   [15]     sentiment_score
# = 16 features. If a future change converts any scalar indicator to a list,
# update EXPECTED_FEATURE_COUNT, FEATURE_NAMES, and _generate_synthetic.
EXPECTED_FEATURE_COUNT = 16

FEATURE_NAMES = (
    ["market_cap", "liquidity", "holders_count", "24h_transactions"]
    + [f"returns_t-{i}" for i in range(5)]
    + ["volatility", "volume_ratio", "rsi", "macd_signal", "pattern_strength"]
    + ["social_score", "sentiment_score"]
)


def _generate_synthetic(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample plausible feature vectors. Distributions are operator-tunable."""
    parts = []
    # Market scalars — log-uniform across realistic ranges.
    parts.append(np.exp(rng.uniform(np.log(1e6), np.log(1e10), n)))   # market_cap
    parts.append(np.exp(rng.uniform(np.log(1e3), np.log(1e8), n)))    # liquidity
    parts.append(np.exp(rng.uniform(np.log(1e2), np.log(1e6), n)))    # holders
    parts.append(np.exp(rng.uniform(np.log(1.0), np.log(1e5), n)))    # 24h txs
    # 5-step rolling returns (the only list-valued indicator).
    parts.append(rng.normal(0, 0.05, (n, 5)))     # returns_last5
    # Scalar indicators.
    parts.append(rng.exponential(0.05, n))        # volatility (annualized stdev)
    parts.append(rng.lognormal(0, 0.5, n))        # volume_ratio
    parts.append(rng.uniform(0, 100, n))          # rsi
    parts.append(rng.normal(0, 0.002, n))         # macd_signal
    parts.append(rng.uniform(0, 1, n))            # pattern_strength
    # Social scalars.
    parts.append(rng.uniform(0, 1, n))            # social_score
    parts.append(rng.uniform(-1, 1, n))           # sentiment_score
    # Concatenate along feature axis.
    cols = []
    for part in parts:
        if part.ndim == 1:
            cols.append(part.reshape(-1, 1))
        else:
            cols.append(part)
    return np.hstack(cols)


def _print_diagnostics(scaler: StandardScaler, matrix: np.ndarray) -> None:
    print(f"Samples fitted:    {matrix.shape[0]}")
    print(f"Feature count:     {matrix.shape[1]} (expected {EXPECTED_FEATURE_COUNT})")
    print(f"Mean range:        {scaler.mean_.min():.4g} .. {scaler.mean_.max():.4g}")
    print(f"Std range:         {scaler.scale_.min():.4g} .. {scaler.scale_.max():.4g}")
    print()
    print("Per-feature mean / std:")
    for i in range(matrix.shape[1]):
        print(f"  [{i:2d}] {FEATURE_NAMES[i]:24s}  "
              f"mean={scaler.mean_[i]:.4g}  std={scaler.scale_[i]:.4g}")
    print()
    sample_row = matrix[:1]
    transformed = scaler.transform(sample_row)
    print("Roundtrip sanity (transform of first sample):")
    print(f"  pre-scale  : {sample_row[0, :4]} ...")
    print(f"  post-scale : {transformed[0, :4]} ...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of synthetic samples to fit on (default: 10000)")
    parser.add_argument("--output", type=Path,
                        default=Path("models/ai_strategy_scaler.pkl"),
                        help="Output path for the fitted scaler")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--from-feature-store", action="store_true",
                        help="Pull labeled rows from ai_feature_store instead of synthetic")
    parser.add_argument("--feature-store-limit", type=int, default=5000)
    parser.add_argument("--feature-store-threshold", type=int, default=200,
                        help="Min real rows required; below this, fall back to synthetic")
    parser.add_argument("--database-url", type=str,
                        default=os.getenv("DATABASE_URL",
                                          "postgresql://bot_user:bot_password@localhost:5432/tradingbot"))
    args = parser.parse_args()

    if args.samples < 100:
        print("ERROR: --samples must be >= 100 for a meaningful fit", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    matrix = None
    if args.from_feature_store:
        result = asyncio.run(load_labeled_features(
            args.database_url, feature_key="scaler_v1",
            limit=args.feature_store_limit))
        if result is not None and result[0].shape[0] >= args.feature_store_threshold:
            matrix = result[0]
            print(f"✓ Fitting on {matrix.shape[0]} REAL rows from ai_feature_store (key='scaler_v1')")
        else:
            n = 0 if result is None else result[0].shape[0]
            print(f"⚠ Only {n} real rows in feature store (< {args.feature_store_threshold}); "
                  "falling back to synthetic")
    if matrix is None:
        if not args.from_feature_store:
            print(f"ℹ Fitting on {args.samples} SYNTHETIC rows "
                  "(pass --from-feature-store to use real data when available)")
        matrix = _generate_synthetic(args.samples, rng)
    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        print(f"ERROR: matrix has {matrix.shape[1]} features, "
              f"expected {EXPECTED_FEATURE_COUNT}.", file=sys.stderr)
        return 2

    scaler = StandardScaler()
    scaler.fit(matrix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, args.output)
    print(f"Scaler written to {args.output}")
    print()
    _print_diagnostics(scaler, matrix)
    print()
    print("Next:")
    print("  - Restart the AI module subprocess so AIStrategy._load_scaler picks it up.")
    print("  - Watch logs/ai_analysis/ai_analysis.log for "
          "'AI strategy: loaded scaler from ... (n_features=16)'.")
    print("  - Replace this synthetic-fit script with a real-feature-store fit "
          "once the feature store ships.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
