# tests/unit/test_pump_predictor_scaler.py
"""P1-08: PumpPredictor scaler must NOT see test-set statistics.

Regression test for the standardization-leakage bug in
`ml/models/pump_predictor.py:201` where `MinMaxScaler.fit_transform(data)`
ran over the entire dataset before any train/test split. Fitting the scaler
on the full window means test-set min/max bleed into train scaling, which
biases LSTM training and inflates validation metrics.
"""
from __future__ import annotations

import pytest

# CI / lightweight runners may not carry the full ML stack. Skip the file
# cleanly rather than ERROR the suite.
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("tensorflow")

from ml.models.pump_predictor import PumpPredictor  # noqa: E402


def _build_price_history(n: int = 200) -> pd.DataFrame:
    """Build a synthetic frame whose tail values blow past the head range.

    If the scaler is fit on the FULL frame, `data_max_` will reflect the
    tail's huge values. If fit on a temporal prefix only, `data_max_` will
    reflect only the (small) head values. This makes the leakage detectable
    by direct comparison.
    """
    rng = np.random.default_rng(seed=42)
    head = rng.uniform(1.0, 2.0, size=int(n * 0.8))
    tail = rng.uniform(100.0, 200.0, size=n - int(n * 0.8))  # huge outliers
    price = np.concatenate([head, tail])
    return pd.DataFrame({
        'price': price,
        'volume': rng.uniform(1.0, 10.0, size=n),
        'liquidity': rng.uniform(1.0, 10.0, size=n),
        'market_cap': rng.uniform(1.0, 10.0, size=n),
        'price_change_5m': rng.uniform(-0.1, 0.1, size=n),
        'price_change_15m': rng.uniform(-0.1, 0.1, size=n),
        'price_change_1h': rng.uniform(-0.1, 0.1, size=n),
        'volume_change_5m': rng.uniform(-0.1, 0.1, size=n),
        'volume_change_1h': rng.uniform(-0.1, 0.1, size=n),
    })


@pytest.mark.unit
def test_prepare_sequences_refuses_when_scaler_unfit():
    """Default path must NOT silently fit the scaler on inference data."""
    pp = PumpPredictor(config={'MODEL_DIR': '/tmp/_pump_test_unfit'})
    df = _build_price_history(n=100)
    with pytest.raises(RuntimeError, match="not fitted"):
        pp.prepare_sequences(df)


@pytest.mark.unit
def test_fit_price_scaler_train_only_excludes_test_window():
    """Train-prefix fit must not see tail-window statistics."""
    pp = PumpPredictor(config={'MODEL_DIR': '/tmp/_pump_test_split'})
    df = _build_price_history(n=200)

    split_idx = pp.fit_price_scaler_train_only(df, train_frac=0.8)
    assert split_idx == 160

    # Train-only fit sees head only (values 1.0–2.0). Tail max (100–200)
    # MUST NOT leak into `data_max_`.
    price_idx = pp.price_features.index('price')
    train_only_max = pp.scalers['price'].data_max_[price_idx]
    assert train_only_max < 10.0, (
        f"Scaler leaked test-set stats: data_max_[price]={train_only_max} "
        f"(expected <10.0 — train head is 1–2, tail is 100–200)"
    )

    # And the legacy fit_scaler=True path SHOULD see the tail outliers —
    # this proves the regression detector itself is sensitive.
    pp_full = PumpPredictor(config={'MODEL_DIR': '/tmp/_pump_test_split_full'})
    pp_full.prepare_sequences(df, fit_scaler=True)
    full_max = pp_full.scalers['price'].data_max_[price_idx]
    assert full_max > 50.0, (
        f"Sanity check: full-fit should see tail outliers, got {full_max}"
    )


@pytest.mark.unit
def test_prepare_sequences_transform_only_uses_train_scaler():
    """After train-only fit, prepare_sequences must transform with that scaler."""
    pp = PumpPredictor(config={'MODEL_DIR': '/tmp/_pump_test_transform'})
    df = _build_price_history(n=200)

    pp.fit_price_scaler_train_only(df, train_frac=0.8)
    scaler_max_before = pp.scalers['price'].data_max_.copy()

    X, y = pp.prepare_sequences(df)

    # prepare_sequences must NOT have re-fit the scaler.
    np.testing.assert_array_equal(
        pp.scalers['price'].data_max_, scaler_max_before,
        err_msg="prepare_sequences re-fit the scaler — P1-08 regression"
    )
    # Sanity: output shape is as expected.
    assert X.shape[0] == len(df) - pp.sequence_length
    assert X.shape[1] == pp.sequence_length
    assert X.shape[2] == len(pp.price_features)
    assert y.shape[0] == X.shape[0]


@pytest.mark.unit
def test_fit_price_scaler_train_only_rejects_invalid_frac():
    pp = PumpPredictor(config={'MODEL_DIR': '/tmp/_pump_test_frac'})
    df = _build_price_history(n=50)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            pp.fit_price_scaler_train_only(df, train_frac=bad)
