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
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.feature_store import load_labeled_features  # noqa: E402
from ml.models.pump_predictor import PumpPredictor  # noqa: E402
from trading.strategies.ai_strategy import (  # noqa: E402
    PUMP_FEATURE_NAMES,
    EXPECTED_PUMP_FEATURE_COUNT,
)

EXPECTED_TREE_FEATURE_COUNT = EXPECTED_PUMP_FEATURE_COUNT  # canonical layout


def _generate_synthetic_features(n, rng):
    """Generate 27 columns matching PUMP_FEATURE_NAMES canonical layout.

    Each column is drawn from a plausible distribution for that feature's
    semantic meaning. The synthetic-trained model thus has semantic
    continuity with live-inference rows once feature-store backfill kicks
    in via --from-feature-store.
    """
    assert EXPECTED_PUMP_FEATURE_COUNT == 27, "feature count drift"

    # [0] vol_acceleration   log-normal centred on 1.0
    vol_acc      = rng.lognormal(mean=0.0, sigma=0.3, size=n)
    # [1] price_momentum     log-normal centred on 1.0
    price_mom    = rng.lognormal(mean=0.0, sigma=0.05, size=n)
    # [2] social_volume_24h  exponential (heavy right tail)
    social_vol   = rng.exponential(500.0, n)
    # [3] social_engagement  uniform [0, 1]
    social_eng   = rng.uniform(0, 1, n)
    # [4] unique_buyers_1h   log-uniform [1, 10_000]
    unique_buy   = np.exp(rng.uniform(np.log(1), np.log(10_000), n))
    # [5] buy_sell_ratio     log-normal around 1
    buy_sell_r   = rng.lognormal(0.0, 0.5, n)
    # [6] txn_count_1h       log-uniform [1, 10_000]
    txn_count    = np.exp(rng.uniform(np.log(1), np.log(10_000), n))
    # [7] last_txn_age_s     exponential (heavy right tail of stale tokens)
    last_txn_age = rng.exponential(60.0, n)
    # [8-11] price_change windows — std scales with window size
    pc_5m        = rng.normal(0.0, 0.02, n)
    pc_1h        = rng.normal(0.0, 0.05, n)
    pc_4h        = rng.normal(0.0, 0.10, n)
    pc_24h       = rng.normal(0.0, 0.20, n)
    # [12] liquidity_usd     log-uniform [1e3, 1e8]
    liquidity    = np.exp(rng.uniform(np.log(1e3),  np.log(1e8),  n))
    # [13] market_cap        log-uniform [1e5, 1e10]
    mcap         = np.exp(rng.uniform(np.log(1e5),  np.log(1e10), n))
    # [14] holders_count     log-uniform [10, 1e6]
    holders      = np.exp(rng.uniform(np.log(10),   np.log(1e6),  n))
    # [15] top10_holder_pct  uniform [0, 1]
    top10_pct    = rng.uniform(0, 1, n)
    # [16] dev_holder_pct    uniform [0, 0.5]  (dev rarely > 50%)
    dev_pct      = rng.uniform(0, 0.5, n)
    # [17-26] scaled tail    normal(0, 1) — post-StandardScaler output
    scaled_tail  = rng.normal(0.0, 1.0, (n, 10))

    X = np.column_stack([
        vol_acc, price_mom, social_vol, social_eng,
        unique_buy, buy_sell_r, txn_count, last_txn_age,
        pc_5m, pc_1h, pc_4h, pc_24h,
        liquidity, mcap, holders, top10_pct, dev_pct,
        scaled_tail,
    ])
    assert X.shape == (n, EXPECTED_PUMP_FEATURE_COUNT), \
        f"shape drift: got {X.shape}, expected (n, {EXPECTED_PUMP_FEATURE_COUNT})"

    # Deterministic-rule label: combine "trend up" + "active" + "not concentrated".
    # log(x) for the multiplicative ratios so they centre around 0.
    score = (
        np.log(price_mom).clip(-1, 1) * 0.30      # strong up-momentum positive
        + np.log(vol_acc).clip(-1, 1) * 0.20      # volume acceleration positive
        + pc_24h.clip(-1, 1) * 0.20               # 24h price change positive
        + (1.0 - top10_pct) * 0.15                # less whale-concentration positive
        + np.tanh(social_vol / 1000.0) * 0.15     # social signal positive (bounded)
    )
    # ~30% positive rate via the 70th percentile threshold.
    threshold = float(np.quantile(score, 0.70))
    labels = (score > threshold).astype(float)
    return X, labels


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
    cfg = {"MODEL_DIR": "./models", "model_version": f"__bootstrap_{a.seed}"}
    pp = PumpPredictor(cfg)

    X, y = None, None
    if a.from_feature_store:
        result = asyncio.run(load_labeled_features(
            a.database_url, feature_key="pump_v1", limit=a.feature_store_limit))
        if result is not None and result[0].shape[0] >= a.feature_store_threshold:
            Xr, pnl, _won = result
            if Xr.shape[1] == EXPECTED_TREE_FEATURE_COUNT:
                X = Xr
                # Heuristic label: significant gain flags a pump.
                y = (pnl > 0.20).astype(float)
                print(f"✓ Fitting on {X.shape[0]} REAL rows from ai_feature_store (key='pump_v1')")
            else:
                print(f"⚠ pump_v1 width {Xr.shape[1]} != expected {EXPECTED_TREE_FEATURE_COUNT}; "
                      "falling back to synthetic")
        else:
            n = 0 if result is None else result[0].shape[0]
            print(f"⚠ Only {n} real pump_v1 rows (< {a.feature_store_threshold}); "
                  "falling back to synthetic")
    if X is None:
        if not a.from_feature_store:
            print(f"ℹ Fitting on {a.samples} SYNTHETIC rows "
                  "(pass --from-feature-store to use real data when available)")
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
    print(f"  Trained on {EXPECTED_PUMP_FEATURE_COUNT} features in canonical order:")
    print(f"    {', '.join(PUMP_FEATURE_NAMES[:8])}...")
    print(f"    ...{', '.join(PUMP_FEATURE_NAMES[-4:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
