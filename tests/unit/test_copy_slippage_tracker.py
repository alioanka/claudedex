"""Unit tests for COPY_TRADING slippage-decay tracker (CT-W3-01).

Pure-Python coverage of the SlippageObservation math + rolling_stats
aggregator. DB writes are tested separately at the integration layer
(persist_observation is a thin asyncpg wrapper).

Run: pytest tests/unit/test_copy_slippage_tracker.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from modules.copy_trading.slippage_tracker import (
    SlippageObservation,
    median_signed,
    rolling_stats,
)


# ---------------------------------------------------------------------
# SlippageObservation.slippage_bps -- signed BUY / SELL convention
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_slippage_bps_buy_positive_when_we_paid_more():
    """BUY: we paid 1.01 vs leader 1.00 -> +100 bps (worse fill)."""
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="buy",
        leader_fill_price_usd=1.00, our_fill_price_usd=1.01,
    )
    assert obs.slippage_bps == 100.0


@pytest.mark.unit
def test_slippage_bps_buy_negative_when_we_got_better():
    """BUY: we paid 0.99 vs leader 1.00 -> -100 bps (better fill, rare)."""
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="buy",
        leader_fill_price_usd=1.00, our_fill_price_usd=0.99,
    )
    assert obs.slippage_bps == -100.0


@pytest.mark.unit
def test_slippage_bps_sell_positive_when_we_sold_lower():
    """SELL: we got 0.95 vs leader 1.00 -> +500 bps (worse fill)."""
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="sell",
        leader_fill_price_usd=1.00, our_fill_price_usd=0.95,
    )
    assert obs.slippage_bps == 500.0


@pytest.mark.unit
def test_slippage_bps_none_when_missing_price():
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="buy",
        leader_fill_price_usd=None, our_fill_price_usd=1.00,
    )
    assert obs.slippage_bps is None


@pytest.mark.unit
def test_slippage_bps_none_on_unknown_side():
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="something_else",
        leader_fill_price_usd=1.0, our_fill_price_usd=1.1,
    )
    assert obs.slippage_bps is None


@pytest.mark.unit
def test_slippage_bps_none_when_leader_price_zero():
    """Guard division-by-zero on garbage data."""
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="buy",
        leader_fill_price_usd=0.0, our_fill_price_usd=1.0,
    )
    assert obs.slippage_bps is None


# ---------------------------------------------------------------------
# delta_ms -- wall-clock latency
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_delta_ms_positive_when_we_fill_after_leader():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T",
        side="buy",
        leader_fill_ts=t0,
        our_fill_ts=t0 + timedelta(milliseconds=350),
    )
    assert obs.delta_ms == 350


@pytest.mark.unit
def test_delta_ms_none_when_missing_timestamps():
    obs = SlippageObservation(
        chain="solana", leader_wallet="W", token_address="T", side="buy",
    )
    assert obs.delta_ms is None


# ---------------------------------------------------------------------
# median_signed -- empty + mixed input
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_median_signed_empty_returns_none():
    assert median_signed([]) is None
    assert median_signed([None, None]) is None


@pytest.mark.unit
def test_median_signed_handles_negatives():
    # -100, -50, 0, 50, 100 -> median 0
    assert median_signed([-100.0, -50.0, 0.0, 50.0, 100.0]) == 0.0


# ---------------------------------------------------------------------
# rolling_stats -- window filtering + bad-fill ratio
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_rolling_stats_filters_outside_window():
    now = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    # Two recent, two too-old (cutoff at 7d ago).
    rows = [
        {"recorded_at": now - timedelta(days=1),  "slippage_bps": 50,  "delta_ms": 200},
        {"recorded_at": now - timedelta(days=3),  "slippage_bps": 150, "delta_ms": 600},
        {"recorded_at": now - timedelta(days=30), "slippage_bps": 999, "delta_ms": 9999},
        {"recorded_at": now - timedelta(days=10), "slippage_bps": 999, "delta_ms": 9999},
    ]
    stats = rolling_stats(rows, window_days=7, as_of=now)
    assert stats["sample_size"] == 2
    # median([50, 150]) = 100
    assert stats["median_slippage_bps"] == 100.0
    assert stats["median_delta_ms"] == 400.0
    assert stats["bad_fill_ratio"] == 1.0  # both positive


@pytest.mark.unit
def test_rolling_stats_bad_fill_ratio_mixed_sign():
    now = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"recorded_at": now, "slippage_bps":  50, "delta_ms": 100},
        {"recorded_at": now, "slippage_bps": -25, "delta_ms": 100},
        {"recorded_at": now, "slippage_bps":   0, "delta_ms": 100},
        {"recorded_at": now, "slippage_bps":  80, "delta_ms": 100},
    ]
    stats = rolling_stats(rows, window_days=7, as_of=now)
    # 2 of 4 are strictly positive -> bad_fill_ratio = 0.5
    assert stats["bad_fill_ratio"] == 0.5


@pytest.mark.unit
def test_rolling_stats_empty_returns_none_medians():
    stats = rolling_stats([], window_days=7)
    assert stats["sample_size"] == 0
    assert stats["median_slippage_bps"] is None
    assert stats["median_delta_ms"] is None
    assert stats["bad_fill_ratio"] is None
