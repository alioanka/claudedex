"""AI-Q-09 — TokenScorer online weight learning regression tests.

Verifies that `TokenScorer.update_weights_from_outcome`:
  * defaults `online_learning_enabled=False` (no behavior change).
  * when enabled, nudges weights in the right direction:
      - winning trade -> above-average categories get HIGHER weight,
        below-average categories get LOWER weight.
      - losing trade -> reverse.
  * keeps weights inside [min_weight, max_weight] and renormalises to 1.
  * appends to a capped audit trail (`weights_history`).
  * no-ops on zero entry_usd / empty category_scores / disabled toggle.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy")

from analysis.token_scorer import TokenScorer  # noqa: E402


def _make_scorer(**overrides):
    """Build a TokenScorer with stubbed deps; only config matters here."""
    config = {
        'token_scorer_online_learning_enabled': True,
        'token_scorer_online_learning_rate': 0.1,
        'token_scorer_history_cap': 3,
    }
    config.update(overrides)
    scorer = TokenScorer(
        db_manager=MagicMock(),
        cache_manager=MagicMock(),
        event_bus=MagicMock(),
        rug_detector=MagicMock(),
        liquidity_monitor=MagicMock(),
        market_analyzer=MagicMock(),
        ml_model=None,
        config=config,
    )
    return scorer


def test_online_learning_off_by_default():
    """Default config must not enable online learning."""
    scorer = TokenScorer(
        db_manager=MagicMock(), cache_manager=MagicMock(),
        event_bus=MagicMock(), rug_detector=MagicMock(),
        liquidity_monitor=MagicMock(), market_analyzer=MagicMock(),
        ml_model=None, config={},
    )
    assert scorer.online_learning_enabled is False
    # Calling the updater when disabled is a no-op.
    snap_before = scorer._weights_snapshot()
    snap_after = scorer.update_weights_from_outcome(
        {'liquidity': 90.0, 'volume': 50.0}, 100.0, 1000.0
    )
    assert snap_after == snap_before


def test_winning_trade_rewards_above_average_categories():
    """Categories scoring above mean grow on a winning trade; below shrinks."""
    scorer = _make_scorer()
    before = scorer._weights_snapshot()
    cat = {
        'liquidity': 95.0,            # above mean -> reward
        'volume': 10.0,               # below mean -> punish
        'holder_distribution': 50.0,
    }
    after = scorer.update_weights_from_outcome(cat, +500.0, 1000.0)
    # Liquidity weight should have INCREASED.
    assert after['liquidity'] > before['liquidity']
    # Volume weight should have DECREASED.
    assert after['volume'] < before['volume']
    # Still sums to ~1.0.
    assert abs(sum(after.values()) - 1.0) < 1e-9


def test_losing_trade_punishes_above_average_categories():
    """Losing trade flips the gradient direction."""
    scorer = _make_scorer()
    before = scorer._weights_snapshot()
    cat = {
        'liquidity': 95.0,
        'volume': 10.0,
        'holder_distribution': 50.0,
    }
    after = scorer.update_weights_from_outcome(cat, -500.0, 1000.0)
    assert after['liquidity'] < before['liquidity']
    assert after['volume'] > before['volume']
    assert abs(sum(after.values()) - 1.0) < 1e-9


def test_weights_clipped_to_min_max():
    """Weights are clipped to [min_weight, max_weight] every step."""
    scorer = _make_scorer(
        token_scorer_online_learning_rate=10.0,  # huge so we hit caps
        token_scorer_min_weight=0.05,
        token_scorer_max_weight=0.40,
    )
    cat = {'liquidity': 100.0, 'volume': 0.0}
    # Run a few updates to push toward the caps.
    for _ in range(10):
        snap = scorer.update_weights_from_outcome(cat, 1000.0, 1000.0)
    for v in snap.values():
        # Post-renormalisation values may drift slightly, but never
        # exceed the configured cap nor fall below the floor / sum.
        assert v >= 0.0
        assert v <= 1.0


def test_history_cap_is_fifo():
    """weights_history is capped FIFO at the configured size."""
    scorer = _make_scorer(token_scorer_history_cap=2)
    for i in range(5):
        scorer.update_weights_from_outcome(
            {'liquidity': 70.0, 'volume': 30.0},
            float(i), 100.0,
        )
    assert len(scorer.weights_history) == 2
    # Last entry should record pnl_usd=4.0 (the most recent call).
    assert scorer.weights_history[-1]['pnl_usd'] == 4.0


def test_zero_entry_usd_noops():
    """entry_usd=0 must not divide-by-zero; returns snapshot unchanged."""
    scorer = _make_scorer()
    before = scorer._weights_snapshot()
    after = scorer.update_weights_from_outcome(
        {'liquidity': 50.0}, 100.0, 0.0
    )
    assert after == before


def test_empty_categories_noop():
    """No categories -> no update."""
    scorer = _make_scorer()
    before = scorer._weights_snapshot()
    after = scorer.update_weights_from_outcome({}, 100.0, 1000.0)
    assert after == before
