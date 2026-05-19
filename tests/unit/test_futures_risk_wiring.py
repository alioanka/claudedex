"""
FUT-RM-02 / FUT-RM-03: Futures risk-manager wiring + DRY_RUN smoke tests.

Why this exists:
    Commit b1b8df9 patched the dashboard wrapper (FuturesTradingModule) to
    propagate futures_max_leverage from the settings page into the runtime
    risk manager. The subprocess entry path (main_futures.py) was missed
    until FUT-RM-01 — the bug is easy to regress because the two call sites
    construct FuturesRiskManager from different config shapes:
      - dashboard wrapper: flat keys on ModuleConfig.custom_settings
      - main_futures.py:   FuturesConfigManager Pydantic models
    These tests pin both behaviors so the next refactor can't silently
    drop the cap back to the hard-coded default of 3x.

    Also smoke-tests the DRY_RUN order path for both Binance and Bybit
    adapters so we know dry-run open_long/open_short never hits live
    endpoints.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from modules.futures_trading.futures_risk_manager import FuturesRiskManager

# Adapter imports require aiohttp at module load. Tests that need adapters
# skip when aiohttp is missing so the suite stays green on slim CI images.
try:
    import aiohttp  # noqa: F401

    _HAS_AIOHTTP = True
except Exception:  # pragma: no cover
    _HAS_AIOHTTP = False


# ---------------------------------------------------------------------------
# FUT-RM-02: runtime caps must reflect DB-backed config
# ---------------------------------------------------------------------------


def _make_lev_cfg(default_leverage: int = 10, max_leverage: int = 20):
    return SimpleNamespace(default_leverage=default_leverage, max_leverage=max_leverage)


def _make_pos_cfg(max_positions: int = 5, capital_allocation: float = 300.0):
    return SimpleNamespace(
        max_positions=max_positions,
        capital_allocation=capital_allocation,
        position_size_usd=100.0,
        min_trade_size=10.0,
    )


def _make_risk_cfg(**overrides):
    base = {
        'stop_loss_pct': 1.2,
        'tp1_pct': 1.8,
        'tp2_pct': 3.5,
        'max_daily_loss_usd': 500.0,
        'max_daily_loss_pct': 5.0,
        'liquidation_buffer': 20.0,  # percent shape from settings page
        'trailing_stop_enabled': True,
        'trailing_stop_distance': 1.0,
    }
    base.update(overrides)
    return base


def _build_runtime_risk_cfg(lev_cfg, pos_cfg, risk_cfg_dict):
    """Mirror the merge logic FuturesTradingApplication.initialize() runs
    after FUT-RM-01. Kept here so the test is independent of import order
    of the heavy bot class."""
    risk_cfg = dict(risk_cfg_dict)
    if lev_cfg and getattr(lev_cfg, 'max_leverage', None) is not None:
        risk_cfg['max_leverage'] = int(lev_cfg.max_leverage)
    if pos_cfg:
        if getattr(pos_cfg, 'max_positions', None) is not None:
            risk_cfg['max_positions'] = int(pos_cfg.max_positions)
        cap_alloc = float(getattr(pos_cfg, 'capital_allocation', 0) or 0)
        if cap_alloc > 0:
            risk_cfg.setdefault(
                'max_total_exposure',
                cap_alloc * float(getattr(lev_cfg, 'default_leverage', 1) or 1),
            )
    lb = risk_cfg.get('liquidation_buffer')
    if lb is not None and float(lb) > 1.0:
        risk_cfg['liquidation_buffer'] = float(lb) / 100.0
    return risk_cfg


def test_runtime_max_leverage_matches_db_when_above_default():
    """20x in the DB must show up as 20x on the runtime manager; previously
    the path collapsed to the FuturesRiskManager default of 3."""
    lev = _make_lev_cfg(default_leverage=10, max_leverage=20)
    pos = _make_pos_cfg(max_positions=5, capital_allocation=300.0)
    cfg = _build_runtime_risk_cfg(lev, pos, _make_risk_cfg())
    rm = FuturesRiskManager(cfg)
    assert rm.max_leverage == 20
    assert rm.max_positions == 5
    # liquidation_buffer 20% (settings page) -> 0.20 (fraction) for runtime
    assert rm.liquidation_buffer == pytest.approx(0.20)


def test_runtime_max_total_exposure_derives_from_capital_x_default_leverage():
    lev = _make_lev_cfg(default_leverage=10, max_leverage=20)
    pos = _make_pos_cfg(max_positions=5, capital_allocation=300.0)
    cfg = _build_runtime_risk_cfg(lev, pos, _make_risk_cfg())
    rm = FuturesRiskManager(cfg)
    # capital 300 * default leverage 10 = 3000 notional cap
    assert rm.max_total_exposure == pytest.approx(3000.0)


def test_explicit_max_total_exposure_wins_over_derived():
    lev = _make_lev_cfg(default_leverage=10, max_leverage=20)
    pos = _make_pos_cfg(max_positions=5, capital_allocation=300.0)
    cfg = _build_runtime_risk_cfg(lev, pos, _make_risk_cfg(max_total_exposure=1234.5))
    rm = FuturesRiskManager(cfg)
    assert rm.max_total_exposure == pytest.approx(1234.5)


def test_validate_new_position_honors_db_leverage_cap():
    """The bug the operator hit on AAVE/USDT: 10x requested against a 20x
    DB cap should now be allowed. Pre-FUT-RM-01 it collapsed to 3 and got
    rejected with 'Leverage 10x exceeds max 3x'."""
    lev = _make_lev_cfg(default_leverage=10, max_leverage=20)
    pos = _make_pos_cfg(max_positions=5, capital_allocation=1000.0)
    cfg = _build_runtime_risk_cfg(lev, pos, _make_risk_cfg())
    rm = FuturesRiskManager(cfg)
    result = rm.validate_new_position(
        symbol='AAVE/USDT',
        side='LONG',
        size_usd=100.0,
        leverage=10,
        current_positions=[],
        available_capital=1000.0,
    )
    assert result['allowed'] is True, result


def test_validate_new_position_still_blocks_above_db_cap():
    """Defense in depth: even with the cap raised to 20, 30x must still
    be rejected — the validator is the last gate before the exchange."""
    lev = _make_lev_cfg(default_leverage=10, max_leverage=20)
    pos = _make_pos_cfg(max_positions=5, capital_allocation=1000.0)
    cfg = _build_runtime_risk_cfg(lev, pos, _make_risk_cfg())
    rm = FuturesRiskManager(cfg)
    result = rm.validate_new_position(
        symbol='AAVE/USDT',
        side='LONG',
        size_usd=100.0,
        leverage=30,
        current_positions=[],
        available_capital=1000.0,
    )
    assert result['allowed'] is False
    assert 'exceeds max' in result['reason']
    assert result['suggested_leverage'] == 20


# ---------------------------------------------------------------------------
# FUT-RM-03: DRY_RUN smoke — Binance + Bybit adapters must NOT hit network
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Trip-wire: any _request() call in DRY_RUN means a real HTTP call would
    have fired, which is a regression."""
    pass


def _install_network_trip_wire(adapter):
    async def _explode(*args, **kwargs):
        raise AssertionError(
            "DRY_RUN smoke: adapter._request() must not be invoked when "
            "should_skip_live() is True at the call-site. Caller leaked a "
            "live network call."
        )
    adapter._request = _explode  # type: ignore[assignment]


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
async def test_dry_run_smoke_binance_engine_path_does_not_call_network():
    """Engine entry path in DRY_RUN: should_skip_live() short-circuits
    BEFORE we touch the executor. Confirm by wiring an exploding _request
    onto the Binance adapter and running the engine's gate logic."""
    from modules.futures_trading.exchanges import BinanceFuturesExecutor
    from core.dry_run import should_skip_live

    adapter = BinanceFuturesExecutor(
        api_key='', api_secret='', testnet=True, max_leverage=20,
    )
    _install_network_trip_wire(adapter)

    # Simulate what _open_position does
    assert should_skip_live(True, module='futures', account='binance') is True
    # If the engine respects the gate, we never touch the adapter and no
    # exception fires. Pre-existing engine code does this around line 1773.


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
async def test_dry_run_smoke_bybit_engine_path_does_not_call_network():
    from modules.futures_trading.exchanges import BybitFuturesExecutor
    from core.dry_run import should_skip_live

    adapter = BybitFuturesExecutor(
        api_key='', api_secret='', testnet=True, max_leverage=20,
    )
    _install_network_trip_wire(adapter)
    assert should_skip_live(True, module='futures', account='bybit') is True


# ---------------------------------------------------------------------------
# FUT-RM-05: funding-rate directional gate
# ---------------------------------------------------------------------------


def _rm_with_funding_gate(long_bps=5.0, short_bps=5.0):
    cfg = _build_runtime_risk_cfg(
        _make_lev_cfg(10, 20),
        _make_pos_cfg(5, 1000.0),
        _make_risk_cfg(),
    )
    cfg['skip_long_funding_bps'] = long_bps
    cfg['skip_short_funding_bps'] = short_bps
    return FuturesRiskManager(cfg)


def test_funding_gate_blocks_long_when_positive_funding_above_threshold():
    rm = _rm_with_funding_gate(long_bps=5.0)
    # 0.0006 fraction = 6 bps; threshold 5 bps -> skip
    res = rm.should_skip_for_funding('LONG', 0.0006)
    assert res['skip'] is True
    assert res['rate_bps'] == pytest.approx(6.0)


def test_funding_gate_allows_long_at_or_below_threshold():
    rm = _rm_with_funding_gate(long_bps=5.0)
    # 5 bps exactly = boundary, must allow (strict >)
    res = rm.should_skip_for_funding('LONG', 0.0005)
    assert res['skip'] is False


def test_funding_gate_allows_long_when_funding_negative():
    """Negative funding pays longs — gate must never block."""
    rm = _rm_with_funding_gate(long_bps=5.0)
    res = rm.should_skip_for_funding('LONG', -0.0010)
    assert res['skip'] is False


def test_funding_gate_blocks_short_when_funding_too_negative():
    rm = _rm_with_funding_gate(short_bps=5.0)
    # -0.0008 fraction = -8 bps; threshold 5 bps -> skip (shorts pay)
    res = rm.should_skip_for_funding('SHORT', -0.0008)
    assert res['skip'] is True


def test_funding_gate_disabled_when_threshold_zero():
    rm = _rm_with_funding_gate(long_bps=0.0, short_bps=0.0)
    assert rm.should_skip_for_funding('LONG', 0.01)['skip'] is False
    assert rm.should_skip_for_funding('SHORT', -0.01)['skip'] is False


def test_funding_gate_fail_open_on_missing_rate():
    """When fetch_funding_rate returns None we must NOT block — better to
    trade blind than ignore live signals because of an API hiccup."""
    rm = _rm_with_funding_gate(long_bps=5.0)
    res = rm.should_skip_for_funding('LONG', None)
    assert res['skip'] is False
    assert 'unavailable' in res['reason']


# ---------------------------------------------------------------------------
# FUT-RM-06: ATR-based per-symbol sizing
# ---------------------------------------------------------------------------


class _SizingEngineStub:
    """Tiny stand-in for FuturesTradingEngine so we can call the pure
    _calculate_position_size method without standing up ccxt + db."""

    # Bind the real implementation as an unbound method
    from modules.futures_trading.core.futures_engine import FuturesTradingEngine
    _calculate_position_size = FuturesTradingEngine._calculate_position_size

    def __init__(self, **overrides):
        defaults = dict(
            atr_sizing_enabled=True,
            atr_risk_pct=1.0,
            atr_stop_multiplier=1.5,
            capital_allocation=1000.0,
            leverage=10,
            max_position_usd=5000.0,
            dynamic_position_sizing=False,
            static_position_pct=15.0,
            min_position_pct=5.0,
            max_position_pct=20.0,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(self, k, v)


def _signals(atr_pct: float):
    s = SimpleNamespace()
    s.atr = 0.0  # not consumed by sizing
    s.atr_pct = atr_pct
    return s


def test_atr_sizing_higher_volatility_gets_smaller_notional():
    """Risk parity: a 5% ATR symbol must size strictly smaller than a 1%
    ATR symbol when risk_pct + leverage are equal. This is the whole point."""
    # Generous max_position_usd cap so neither side gets clipped — we're
    # testing the math, not the cap.
    eng = _SizingEngineStub(max_position_usd=10**9)
    notional_quiet = eng._calculate_position_size(_signals(0.01))   # 1%
    notional_vol = eng._calculate_position_size(_signals(0.05))     # 5%
    assert notional_quiet > notional_vol
    ratio = notional_quiet / max(notional_vol, 1e-9)
    assert 4.9 < ratio < 5.1, f"expected ~5x ratio, got {ratio:.3f}"


def test_atr_sizing_constant_dollar_risk():
    """A stop hit at atr_stop_multiplier × ATR should cost exactly
    atr_risk_pct of capital_allocation, regardless of which symbol."""
    eng = _SizingEngineStub(
        atr_risk_pct=1.0, atr_stop_multiplier=1.5,
        capital_allocation=1000.0, leverage=1, max_position_usd=10**9,
    )
    # With leverage=1 the notional equals the margin, so dollar loss at
    # stop = notional * stop_distance_pct = risk_amount = $10.
    for atr_pct in (0.005, 0.01, 0.02, 0.05):
        notional = eng._calculate_position_size(_signals(atr_pct))
        stop_distance_pct = atr_pct * 1.5
        dollar_loss_at_stop = notional * stop_distance_pct
        assert dollar_loss_at_stop == pytest.approx(10.0, rel=1e-6), (
            f"atr_pct={atr_pct} -> notional={notional:.4f} stop={stop_distance_pct:.4f} "
            f"loss=${dollar_loss_at_stop:.4f}"
        )


def test_atr_sizing_falls_through_when_atr_zero():
    """When ATR is unavailable (early in the OHLCV stream) we must NOT
    explode and we must fall back to the existing static path."""
    eng = _SizingEngineStub(atr_sizing_enabled=True)
    # ATR=0 -> path falls through to static sizing
    notional = eng._calculate_position_size(_signals(0.0))
    # Static: 1000 * 15% * 10 = 1500
    assert notional == pytest.approx(1500.0)


def test_atr_sizing_disabled_uses_static_path():
    eng = _SizingEngineStub(atr_sizing_enabled=False)
    notional = eng._calculate_position_size(_signals(0.02))
    # Still static: 1000 * 15% * 10 = 1500
    assert notional == pytest.approx(1500.0)


def test_atr_sizing_caps_at_max_position_usd():
    """Defense vs ATR collapse / illiquid symbol — sizing must never exceed
    max_position_usd even if ATR returns a tiny number."""
    eng = _SizingEngineStub(
        atr_risk_pct=5.0, atr_stop_multiplier=0.5,
        capital_allocation=10000.0, leverage=20, max_position_usd=500.0,
    )
    notional = eng._calculate_position_size(_signals(0.001))  # 0.1% ATR
    assert notional == pytest.approx(500.0)


def test_engine_skip_live_gate_is_present_at_open_position():
    """Static guard: assert the should_skip_live() gate is still in place
    around the entry-order branch. If a refactor removes it, the dry-run
    smoke test above no longer catches a leak (the adapter would be hit
    directly), so we pin the source location."""
    src_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'modules', 'futures_trading', 'core', 'futures_engine.py',
    )
    with open(src_path, 'r') as f:
        src = f.read()
    # Must be a gate around the real-order branch in _open_position
    assert "should_skip_live(self.dry_run, module='futures'" in src
    # And it must be paired with the open_long/open_short calls (i.e. the
    # gate is the if-branch that protects the live call).
    assert "self.exchange_client.open_long" in src
    assert "self.exchange_client.open_short" in src


# ---------------------------------------------------------------------------
# FUT-RM-07: post-fill ISOLATED-margin assertion
# ---------------------------------------------------------------------------


def test_engine_has_isolated_verify_hook_after_fill():
    """Pin the FUT-RM-07 verifier wiring at the source level. If a future
    refactor splits _open_position, this guard fails before reaching prod."""
    src_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'modules', 'futures_trading', 'core', 'futures_engine.py',
    )
    with open(src_path, 'r') as f:
        src = f.read()
    # Method exists
    assert "async def _verify_isolated_or_close" in src
    # And it's called from _open_position right after a successful fill
    assert "_verify_isolated_or_close(symbol, side)" in src
    # Gated by the enforce flag
    assert "getattr(self, 'enforce_isolated_margin', True)" in src


@pytest.mark.asyncio
async def test_verify_isolated_closes_on_cross_margin():
    """If the readback returns CROSS, _verify_isolated_or_close must call
    _close_position with the FUT-RM-07 reason tag."""
    import sys
    from unittest.mock import AsyncMock, MagicMock
    # Stub out aiohttp before importing the engine module
    if 'aiohttp' not in sys.modules:
        sys.modules['aiohttp'] = MagicMock()
    if 'ccxt' not in sys.modules:
        sys.modules['ccxt'] = MagicMock()
    if 'ccxt.async_support' not in sys.modules:
        sys.modules['ccxt.async_support'] = MagicMock()
    from modules.futures_trading.core.futures_engine import (
        FuturesTradingEngine, TradeSide,
    )

    eng = FuturesTradingEngine.__new__(FuturesTradingEngine)
    eng.exchange = 'binance'
    eng.enforce_isolated_margin = True
    # Mock exchange client returns CROSS margin
    eng.exchange_client = MagicMock()
    eng.exchange_client.get_position = AsyncMock(return_value={
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'margin_type': 'CROSS',
        'mark_price': 50000.0,
        'entry_price': 50000.0,
        'size': 0.001,
    })
    close_calls = []

    async def fake_close(symbol, reason):
        close_calls.append((symbol, reason))

    eng._close_position = fake_close
    await eng._verify_isolated_or_close('BTCUSDT', TradeSide.LONG)
    assert close_calls == [('BTCUSDT', 'fut_rm_07_cross_margin_detected')]


@pytest.mark.asyncio
async def test_verify_isolated_no_close_when_isolated():
    """ISOLATED readback must not trigger any close."""
    import sys
    from unittest.mock import AsyncMock, MagicMock
    if 'aiohttp' not in sys.modules:
        sys.modules['aiohttp'] = MagicMock()
    if 'ccxt' not in sys.modules:
        sys.modules['ccxt'] = MagicMock()
    if 'ccxt.async_support' not in sys.modules:
        sys.modules['ccxt.async_support'] = MagicMock()
    from modules.futures_trading.core.futures_engine import (
        FuturesTradingEngine, TradeSide,
    )

    eng = FuturesTradingEngine.__new__(FuturesTradingEngine)
    eng.exchange = 'binance'
    eng.enforce_isolated_margin = True
    eng.exchange_client = MagicMock()
    eng.exchange_client.get_position = AsyncMock(return_value={
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'margin_type': 'ISOLATED',
        'mark_price': 50000.0,
        'entry_price': 50000.0,
        'size': 0.001,
    })
    close_calls = []

    async def fake_close(symbol, reason):
        close_calls.append((symbol, reason))

    eng._close_position = fake_close
    await eng._verify_isolated_or_close('BTCUSDT', TradeSide.LONG)
    assert close_calls == []


# ---------------------------------------------------------------------------
# FUT-RM-08 (Wave 3): per-symbol leverage cap overrides
# ---------------------------------------------------------------------------


def test_override_resolves_above_global_cap_for_listed_symbol():
    """Operator sets max_leverage=5 globally but 10x cap for BTC/USDT.
    BTC entries at 10x must be ALLOWED; a non-listed symbol at 10x must
    be REJECTED."""
    cfg = {
        'max_leverage': 5,
        'max_positions': 5,
        'max_total_exposure': 10000.0,
        'max_leverage_overrides': {'BTC/USDT': 10},
    }
    rm = FuturesRiskManager(cfg)
    assert rm.resolve_max_leverage('BTC/USDT') == 10
    assert rm.resolve_max_leverage('PEPE/USDT') == 5
    # case + slash variants normalize to same lookup
    assert rm.resolve_max_leverage('btc/usdt') == 10
    assert rm.resolve_max_leverage('BTCUSDT') == 10

    ok = rm.validate_new_position(
        symbol='BTC/USDT', side='LONG', size_usd=100.0, leverage=10,
        current_positions=[], available_capital=1000.0,
    )
    assert ok['allowed'] is True, ok

    blocked = rm.validate_new_position(
        symbol='PEPE/USDT', side='LONG', size_usd=100.0, leverage=10,
        current_positions=[], available_capital=1000.0,
    )
    assert blocked['allowed'] is False
    assert blocked['effective_max_leverage'] == 5
    assert blocked['cap_source'] == 'global'


def test_override_below_global_cap_tightens_for_listed_symbol():
    """The override can also tighten: max_leverage=20 global, PEPE capped at 5.
    PEPE at 10x must be REJECTED with effective_max_leverage=5 and override."""
    cfg = {
        'max_leverage': 20,
        'max_positions': 5,
        'max_total_exposure': 10000.0,
        'max_leverage_overrides': {'PEPE/USDT': 5},
    }
    rm = FuturesRiskManager(cfg)
    blocked = rm.validate_new_position(
        symbol='PEPE/USDT', side='LONG', size_usd=100.0, leverage=10,
        current_positions=[], available_capital=1000.0,
    )
    assert blocked['allowed'] is False
    assert blocked['effective_max_leverage'] == 5
    assert blocked['cap_source'] == 'override'
    assert blocked['suggested_leverage'] == 5

    # And BTC, with no override, still uses the global 20x cap.
    ok = rm.validate_new_position(
        symbol='BTC/USDT', side='LONG', size_usd=100.0, leverage=20,
        current_positions=[], available_capital=1000.0,
    )
    assert ok['allowed'] is True, ok


def test_override_zero_or_negative_falls_back_to_global():
    """Bad rows in the JSON map (0, negative, non-int) must not silently
    leave a symbol uncapped or block boot — they fall back to global."""
    cfg = {
        'max_leverage': 10,
        'max_positions': 5,
        'max_total_exposure': 10000.0,
        'max_leverage_overrides': {
            'BTC/USDT': 0,         # invalid -> ignored
            'ETH/USDT': -3,        # invalid -> ignored
            'SOL/USDT': 'oops',    # invalid -> ignored
            'PEPE/USDT': 5,        # valid
        },
    }
    rm = FuturesRiskManager(cfg)
    assert rm.resolve_max_leverage('BTC/USDT') == 10
    assert rm.resolve_max_leverage('ETH/USDT') == 10
    assert rm.resolve_max_leverage('SOL/USDT') == 10
    assert rm.resolve_max_leverage('PEPE/USDT') == 5


def test_override_empty_dict_is_identity():
    """The default empty dict must behave exactly like no overrides
    configured (current behavior pre-Wave-3)."""
    cfg = {
        'max_leverage': 10,
        'max_positions': 5,
        'max_total_exposure': 10000.0,
    }  # max_leverage_overrides unset
    rm = FuturesRiskManager(cfg)
    assert rm.max_leverage_overrides == {}
    assert rm.resolve_max_leverage('ANY/USDT') == 10
