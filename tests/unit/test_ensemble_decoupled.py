"""AI-Q-08 — EnsembleModel feature-decoupled predict path.

Verifies that `EnsemblePredictor.predict_decoupled(token, chain, features)`:
  * accepts a numpy feature vector and returns a complete result dict
    WITHOUT touching `data.collectors.dexscreener` — i.e. no DexScreener
    fetch happens in the hot path.
  * accepts a dict in the same schema as `extract_features` consumes.
  * surfaces a safe-default result + error key on exception.
  * tags the result with `source='decoupled'` so dashboards can split
    legacy-vs-decoupled traffic.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ml.models.ensemble_model import EnsemblePredictor  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_predict_decoupled_accepts_ndarray():
    """Pre-extracted feature vector is consumed directly."""
    p = EnsemblePredictor(config={})
    features = np.zeros(95, dtype=np.float32)  # extract_features() width
    result = _run(p.predict_decoupled('0xABC', 'ethereum', features))
    assert result['token'] == '0xABC'
    assert result['chain'] == 'ethereum'
    assert result['source'] == 'decoupled'
    assert 0.0 <= result['pump_probability'] <= 1.0
    assert 0.0 <= result['rug_probability'] <= 1.0


def test_predict_decoupled_accepts_dict():
    """Dict input is auto-converted via extract_features()."""
    p = EnsemblePredictor(config={})
    payload = {
        'price_data': {'current_price': 1.0, 'price_change_24h': 5.0},
        'volume_data': {'volume_24h': 50_000.0},
        'liquidity_data': {'total_liquidity': 100_000.0},
        'holder_data': {'total_holders': 500},
    }
    result = _run(p.predict_decoupled('0xDEF', 'bsc', payload))
    assert result['token'] == '0xDEF'
    assert result['chain'] == 'bsc'
    assert result['source'] == 'decoupled'


def test_predict_decoupled_no_dexscreener_import():
    """The decoupled path must not import DexScreenerCollector.

    We snapshot sys.modules before the call and assert no new entries
    referencing `dexscreener` were added.
    """
    p = EnsemblePredictor(config={})
    before = set(sys.modules)
    features = np.zeros(95, dtype=np.float32)
    _run(p.predict_decoupled('0xABC', 'ethereum', features))
    after = set(sys.modules)
    new = after - before
    assert not any('dexscreener' in m.lower() for m in new), \
        f"decoupled path leaked a DexScreener import: {new}"


def test_predict_decoupled_safe_default_on_error():
    """If features can't be coerced, return a neutral dict + error key."""
    p = EnsemblePredictor(config={})
    # Strings can't become a float ndarray; conversion raises.
    result = _run(p.predict_decoupled('0xABC', 'ethereum', 'not-features'))
    assert result['pump_probability'] == 0.5
    assert result['rug_probability'] == 0.5
    assert result['source'] == 'decoupled'
    assert 'error' in result
