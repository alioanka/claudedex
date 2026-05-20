"""Unit tests for the per-(chain, dex_pair, pair_symbol) realized-slippage
estimator added in Wave-3 (commit f2e95d3).

Covers:
- _slip_key normalisation
- get_realized_slippage min-sample gating + None fallback
- _refresh_realized_slippage aggregation: median + p90 + negative-clamp +
  per-trade pair_symbol bucketing
- Cold-start path (no DB rows) leaves the cache empty so callers fall back
  to the static CHAIN_CONFIGS default

The full EVMArbitrageEngine wires up a Web3 client, secrets manager, and
RPC pool at __init__ time. To keep this test pure-unit we construct a
bare object via object.__new__() and bolt on only the fields the methods
under test actually read.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from modules.arbitrage.arbitrage_engine import EVMArbitrageEngine


def _bare_engine(chain_name: str = "ethereum") -> EVMArbitrageEngine:
    """Construct an EVMArbitrageEngine without running __init__ (which needs
    Web3 + secrets + RPC pool). Bolts on only the attributes the slippage
    helpers read."""
    engine = object.__new__(EVMArbitrageEngine)
    engine.chain_name = chain_name
    engine.chain_id = 1 if chain_name == "ethereum" else 42161
    engine.chain_config = {
        "default_slippage_pct": 0.005,
        "flash_loan_fee_pct": 0.0005,
    }
    engine._realized_slip_cache = {}
    engine._realized_slip_refreshed_at = None
    engine._realized_slip_ttl_s = 3600
    engine._realized_slip_min_samples = 5
    engine.db_pool = None
    engine.logger = MagicMock()
    return engine


@pytest.mark.unit
def test_slip_key_format():
    """Key format must be stable across reads / writes / refresh."""
    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    assert key == "uniswap_v2->sushiswap|USDC/WETH"


@pytest.mark.unit
def test_get_realized_slippage_empty_cache_returns_none():
    """Cold start: no samples => None so caller falls back to chain default."""
    eng = _bare_engine()
    assert eng.get_realized_slippage("uniswap_v2", "sushiswap", "USDC/WETH") is None


@pytest.mark.unit
def test_get_realized_slippage_below_min_samples_returns_none():
    """Below min_samples (default 5) we don't trust the estimate."""
    eng = _bare_engine()
    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    eng._realized_slip_cache[key] = (0.003, 0.007, 4)  # only 4 samples
    assert eng.get_realized_slippage("uniswap_v2", "sushiswap", "USDC/WETH") is None


@pytest.mark.unit
def test_get_realized_slippage_returns_median_by_default():
    """Default lookup returns median (not p90)."""
    eng = _bare_engine()
    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    eng._realized_slip_cache[key] = (0.003, 0.007, 10)
    val = eng.get_realized_slippage("uniswap_v2", "sushiswap", "USDC/WETH")
    assert val == pytest.approx(0.003)


@pytest.mark.unit
def test_get_realized_slippage_p90_flag():
    """use_p90=True returns p90 for the pre-execute gate."""
    eng = _bare_engine()
    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    eng._realized_slip_cache[key] = (0.003, 0.007, 10)
    val = eng.get_realized_slippage(
        "uniswap_v2", "sushiswap", "USDC/WETH", use_p90=True
    )
    assert val == pytest.approx(0.007)


@pytest.mark.unit
def test_get_realized_slippage_unknown_key_returns_none():
    """Different pair => None (falls back to static default)."""
    eng = _bare_engine()
    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    eng._realized_slip_cache[key] = (0.003, 0.007, 10)
    assert eng.get_realized_slippage("uniswap_v2", "sushiswap", "DAI/WETH") is None
    assert eng.get_realized_slippage("uniswap_v2", "curve", "USDC/WETH") is None


def _mock_trade_row(
    *,
    buy_dex: str,
    sell_dex: str,
    pair_symbol: str,
    spread_pct: float,
    profit_loss_pct: float,
    entry_usd: float = 10_000.0,
    gas_cost: float = 30.0,
    ts: datetime | None = None,
) -> dict:
    """Build the asyncpg-style mapping the engine reads in _refresh_realized_slippage."""
    return {
        "buy_dex": buy_dex,
        "sell_dex": sell_dex,
        "spread_pct": spread_pct,
        "profit_loss_pct": profit_loss_pct,
        "entry_usd": entry_usd,
        "metadata": json.dumps({"pair_symbol": pair_symbol, "gas_cost": gas_cost}),
        "entry_timestamp": ts or datetime.utcnow(),
    }


def _make_mock_pool(rows: List[dict]):
    """Return an asyncpg-style pool whose acquire().fetch returns `rows`
    and acquire().execute is a no-op AsyncMock."""
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    conn.execute = AsyncMock(return_value=None)

    class _AcquireCtx:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, exc_type, exc, tb):
            return None

    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=lambda: _AcquireCtx())
    return pool, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_realized_slippage_aggregates_median_and_p90():
    """7d aggregation: median + p90 + sample count, per (dex_pair, pair_symbol).

    Construct 10 rows for one bucket with gross 1.0% and slippage costs
    spanning 0.10%..1.00% in 0.10% increments (after subtracting the live
    gas + flash-loan deduction). Expected median=0.50% (index 5 in sorted
    even-length list), p90=0.90%, count=10.
    """
    eng = _bare_engine()
    rows = []
    base = datetime.utcnow() - timedelta(days=1)
    # gross 1% per trade; deductions: flash_fee=0.05%, gas_pct=0.30% (gas $30
    # / entry $10000). slippage_target = 0.1%..1.0% step 0.1%.
    # net = gross - flash_fee - gas_pct - slippage_target
    for i in range(10):
        slip = 0.001 * (i + 1)  # 0.001..0.010
        gross = 0.01
        net = gross - 0.0005 - 0.003 - slip
        rows.append(_mock_trade_row(
            buy_dex="uniswap_v2",
            sell_dex="sushiswap",
            pair_symbol="USDC/WETH",
            spread_pct=gross * 100.0,
            profit_loss_pct=net * 100.0,
            entry_usd=10_000.0,
            gas_cost=30.0,
            ts=base + timedelta(hours=i),
        ))
    pool, conn = _make_mock_pool(rows)
    eng.db_pool = pool

    await eng._refresh_realized_slippage()

    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    assert key in eng._realized_slip_cache
    median, p90, n = eng._realized_slip_cache[key]
    assert n == 10
    # Sorted slip values: 0.001, 0.002, ..., 0.010. Median (n//2 = 5) = 0.006.
    assert median == pytest.approx(0.006, abs=1e-9)
    # p90 (idx = int(0.9 * 10) = 9, clamped to n-1=9) = 0.010.
    assert p90 == pytest.approx(0.010, abs=1e-9)
    # Upsert called once per bucket key (only one here).
    assert conn.execute.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_realized_slippage_drops_negative_samples():
    """A row where costs > gross (failed-tx logging artifact) yields a
    negative realized slippage; the refresh must drop it before computing
    median so the estimate isn't biased downward."""
    eng = _bare_engine()
    # 1 healthy row (slip=0.5%) + 1 artifact row (would compute slip=-0.3%).
    rows = [
        _mock_trade_row(
            buy_dex="uniswap_v2", sell_dex="sushiswap", pair_symbol="USDC/WETH",
            spread_pct=1.0, profit_loss_pct=0.15,  # gross 1% - fees - 0.5% slip = 0.15%
        ),
        _mock_trade_row(
            buy_dex="uniswap_v2", sell_dex="sushiswap", pair_symbol="USDC/WETH",
            spread_pct=0.1, profit_loss_pct=0.5,   # gross 0.1% < flash_fee+gas_pct => negative slip
        ),
    ]
    pool, _ = _make_mock_pool(rows)
    eng.db_pool = pool

    await eng._refresh_realized_slippage()

    key = EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH")
    median, _, n = eng._realized_slip_cache[key]
    # Only the healthy row should have been bucketed.
    assert n == 1
    assert median == pytest.approx(0.005, abs=1e-9)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_realized_slippage_buckets_by_dex_pair_and_pair_symbol():
    """Different (buy_dex, sell_dex) or pair_symbol must land in distinct buckets."""
    eng = _bare_engine()
    rows = [
        _mock_trade_row(
            buy_dex="uniswap_v2", sell_dex="sushiswap", pair_symbol="USDC/WETH",
            spread_pct=1.0, profit_loss_pct=0.15,  # slip 0.5%
        ),
        _mock_trade_row(
            buy_dex="curve", sell_dex="balancer", pair_symbol="USDC/WETH",
            spread_pct=1.0, profit_loss_pct=0.55,  # slip 0.1%
        ),
        _mock_trade_row(
            buy_dex="uniswap_v2", sell_dex="sushiswap", pair_symbol="DAI/WETH",
            spread_pct=1.0, profit_loss_pct=0.25,  # slip 0.4%
        ),
    ]
    pool, _ = _make_mock_pool(rows)
    eng.db_pool = pool

    await eng._refresh_realized_slippage()

    assert len(eng._realized_slip_cache) == 3
    assert EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "USDC/WETH") in eng._realized_slip_cache
    assert EVMArbitrageEngine._slip_key("curve", "balancer", "USDC/WETH") in eng._realized_slip_cache
    assert EVMArbitrageEngine._slip_key("uniswap_v2", "sushiswap", "DAI/WETH") in eng._realized_slip_cache


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_realized_slippage_ttl_skips_when_fresh():
    """Within TTL the method is a no-op (no DB hit)."""
    eng = _bare_engine()
    eng._realized_slip_refreshed_at = datetime.now()  # just refreshed
    pool, conn = _make_mock_pool([])
    eng.db_pool = pool

    await eng._refresh_realized_slippage()
    # fetch should NOT have been called
    conn.fetch.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_realized_slippage_no_db_is_noop():
    """No db_pool => method returns without raising."""
    eng = _bare_engine()
    eng.db_pool = None
    await eng._refresh_realized_slippage()
    assert eng._realized_slip_cache == {}


@pytest.mark.unit
def test_realized_slippage_caller_fallback_path():
    """When get_realized_slippage returns None, the caller (the pre-execute
    gate / _log_arb_trade) must fall back to chain_config['default_slippage_pct'].
    This test pins the contract: the helper returns None, NOT 0.0, so the
    fallback `if x is None: x = static` works correctly. A buggy version that
    returned 0.0 would have us treat new pairs as zero-slippage and over-
    execute."""
    eng = _bare_engine()
    out = eng.get_realized_slippage("uniswap_v2", "sushiswap", "USDC/WETH")
    assert out is None  # NOT 0.0
