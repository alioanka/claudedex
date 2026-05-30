# tests/unit/test_price_buffer.py
"""Unit tests for modules/solana_trading/core/price_buffer.TokenPriceBuffer.

Asserts FIFO behaviour, recency eviction, and bad-input rejection. The
buffer feeds the pump-predictor gate at engine `_open_position` time, so
correctness here is load-bearing: a leaky window or accepting zero/None
prices would either crash the gate or feed it junk.
"""
from __future__ import annotations

import pytest

from modules.solana_trading.core.price_buffer import (
    PricePoint,
    TokenPriceBuffer,
)


@pytest.mark.unit
def test_append_and_snapshot_round_trip():
    buf = TokenPriceBuffer(maxlen=5)
    for i, p in enumerate([10.0, 11.0, 12.0]):
        buf.append("mintA", p, ts=float(i))
    snap = buf.snapshot("mintA")
    assert [pp.price for pp in snap] == [10.0, 11.0, 12.0]
    assert [pp.ts for pp in snap] == [0.0, 1.0, 2.0]
    assert buf.size("mintA") == 3


@pytest.mark.unit
def test_fifo_drops_oldest_when_full():
    """FIFO bounded behaviour: maxlen=3, append 5 -> last 3 retained."""
    buf = TokenPriceBuffer(maxlen=3)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        buf.append("mintA", p, ts=float(i))
    prices = buf.prices("mintA")
    assert prices == [3.0, 4.0, 5.0]
    assert buf.size("mintA") == 3
    # Cap is hard: no leak above maxlen.
    assert buf.size("mintA") <= buf.maxlen


@pytest.mark.unit
def test_per_token_isolation():
    buf = TokenPriceBuffer(maxlen=10)
    buf.append("A", 1.0)
    buf.append("B", 2.0)
    buf.append("A", 3.0)
    assert buf.prices("A") == [1.0, 3.0]
    assert buf.prices("B") == [2.0]
    assert len(buf) == 2  # two tracked tokens


@pytest.mark.unit
def test_has_enough_threshold():
    buf = TokenPriceBuffer(maxlen=100)
    for i in range(60):
        buf.append("M", float(i + 1))
    assert buf.has_enough("M", 60) is True
    assert buf.has_enough("M", 61) is False
    assert buf.has_enough("UNSEEN", 1) is False


@pytest.mark.unit
def test_drop_removes_token_entirely():
    buf = TokenPriceBuffer(maxlen=5)
    buf.append("A", 1.0)
    buf.drop("A")
    assert buf.size("A") == 0
    assert buf.prices("A") == []
    # Dropping again is a no-op.
    buf.drop("A")


@pytest.mark.unit
def test_bad_inputs_silently_dropped():
    """Engine feeds prices from many sources -- never crash on bad input."""
    buf = TokenPriceBuffer(maxlen=5)
    buf.append("", 1.0)        # empty mint
    buf.append("A", None)       # None price
    buf.append("A", 0.0)        # non-positive
    buf.append("A", -3.5)       # negative
    buf.append("A", "not-a-number")  # type-error material
    assert buf.size("A") == 0
    assert len(buf) == 0


@pytest.mark.unit
def test_evict_stale_drops_old_tokens():
    buf = TokenPriceBuffer(maxlen=5, max_age_s=10.0)
    buf.append("FRESH", 1.0, ts=1000.0)
    buf.append("STALE", 1.0, ts=10.0)
    evicted = buf.evict_stale(now=1005.0)
    assert evicted == 1
    assert buf.size("FRESH") == 1
    assert buf.size("STALE") == 0


@pytest.mark.unit
def test_evict_stale_noop_when_disabled():
    buf = TokenPriceBuffer(maxlen=5)  # max_age_s=None
    buf.append("OLD", 1.0, ts=0.0)
    assert buf.evict_stale(now=1e9) == 0
    assert buf.size("OLD") == 1


@pytest.mark.unit
def test_constructor_rejects_invalid_maxlen():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            TokenPriceBuffer(maxlen=bad)


@pytest.mark.unit
def test_constructor_rejects_invalid_max_age():
    with pytest.raises(ValueError):
        TokenPriceBuffer(maxlen=5, max_age_s=0)
    with pytest.raises(ValueError):
        TokenPriceBuffer(maxlen=5, max_age_s=-1.5)


@pytest.mark.unit
def test_price_point_frozen():
    p = PricePoint(ts=1.0, price=2.0)
    with pytest.raises(Exception):
        p.price = 3.0  # type: ignore[misc]
