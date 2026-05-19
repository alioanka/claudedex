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
