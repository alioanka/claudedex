"""Unit tests for modules.backtest_replay.

Covers replay_engine + strategies with synthetic data. No DB
required — pure functions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from modules.backtest_replay.core.replay_engine import (
    ModuleReplayResult, ReplayReport, run_replay,
    _max_drawdown_pct, _sharpe,
)
from modules.backtest_replay.core.strategies import (
    STRATEGY_FUNCS, get_strategy, approve_all, approve_on_confidence,
    never_approve, operator_replay,
)
from modules.backtest_replay.core.trade_loader import RecRow, TradeRow


def _trades(module: str, n: int, pnl_each: float, base: datetime):
    return [
        TradeRow(module=module, ts=base + timedelta(minutes=i),
                 pnl_usd=pnl_each, is_simulated=True)
        for i in range(n)
    ]


# ----- strategies -----

def test_strategy_approve_all_rejects_hold():
    rec = RecRow(ts=datetime.utcnow(), module='m', recommended='hold',
                 confidence=0.9, approved=None)
    assert approve_all(rec, {}) is False


def test_strategy_approve_all_accepts_to_live():
    rec = RecRow(ts=datetime.utcnow(), module='m', recommended='to_live',
                 confidence=0.1, approved=None)
    assert approve_all(rec, {}) is True


def test_strategy_approve_on_confidence_threshold_inclusive():
    rec = RecRow(ts=datetime.utcnow(), module='m', recommended='to_live',
                 confidence=0.8, approved=None)
    assert approve_on_confidence(rec, {'confidence_threshold': 0.8}) is True
    assert approve_on_confidence(rec, {'confidence_threshold': 0.81}) is False


def test_strategy_never_approve():
    rec = RecRow(ts=datetime.utcnow(), module='m', recommended='to_live',
                 confidence=1.0, approved=True)
    assert never_approve(rec, {}) is False


def test_strategy_operator_replay_matches_approved_field():
    base = datetime.utcnow()
    r1 = RecRow(ts=base, module='m', recommended='to_live', confidence=0.9, approved=True)
    r2 = RecRow(ts=base, module='m', recommended='to_live', confidence=0.9, approved=False)
    r3 = RecRow(ts=base, module='m', recommended='to_live', confidence=0.9, approved=None)
    assert operator_replay(r1, {}) is True
    assert operator_replay(r2, {}) is False
    assert operator_replay(r3, {}) is False


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError):
        get_strategy('does-not-exist')


# ----- helper functions -----

def test_sharpe_constant_series_returns_none():
    # All-same pnl → stdev=0 → None
    assert _sharpe([1.0] * 10) is None


def test_sharpe_basic_math():
    pnls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    s = _sharpe(pnls)
    assert s is not None
    assert 0 < s < 5


def test_max_drawdown_no_decline():
    eq = [0, 1, 2, 3, 4, 5]
    assert _max_drawdown_pct(eq) == 0


def test_max_drawdown_simple_decline():
    eq = [0, 10, 5, 7]  # peak at 10, trough at 5 → 50% DD
    dd = _max_drawdown_pct(eq)
    assert 40 < dd < 60


# ----- end-to-end replay -----

def test_replay_empty_data_succeeds():
    rep = run_replay({}, [], 'approve_all')
    assert rep.total_actual_pnl_usd == 0
    assert rep.total_counterfactual_pnl_usd == 0
    assert rep.per_module == []


def test_replay_constant_pnl_no_recs():
    base = datetime.utcnow()
    trades = {'sniper': _trades('sniper', 20, 5.0, base)}
    rep = run_replay(trades, [], 'never_approve')
    pm = rep.per_module[0]
    assert pm.module == 'sniper'
    assert pm.n_trades == 20
    assert pm.actual_pnl_usd == 100.0
    assert pm.counterfactual_pnl_usd == 100.0
    assert pm.pnl_delta_usd == 0.0
    assert pm.n_recs_approved == 0


def test_replay_counts_approved_recs():
    base = datetime.utcnow()
    trades = {'sniper': _trades('sniper', 5, 1.0, base)}
    recs = [
        RecRow(ts=base + timedelta(minutes=1), module='sniper',
               recommended='to_live', confidence=0.9, approved=None),
        RecRow(ts=base + timedelta(minutes=2), module='sniper',
               recommended='hold', confidence=0.5, approved=None),
        RecRow(ts=base + timedelta(minutes=3), module='sniper',
               recommended='to_dry', confidence=0.6, approved=None),
    ]
    rep = run_replay(trades, recs, 'approve_all')
    pm = rep.per_module[0]
    assert pm.n_recs_seen == 3
    # approve_all rejects 'hold' → only 2 approved
    assert pm.n_recs_approved == 2


def test_replay_threshold_strategy_filters_by_confidence():
    base = datetime.utcnow()
    trades = {'sniper': _trades('sniper', 5, 1.0, base)}
    recs = [
        RecRow(ts=base + timedelta(minutes=1), module='sniper',
               recommended='to_live', confidence=0.6, approved=None),
        RecRow(ts=base + timedelta(minutes=2), module='sniper',
               recommended='to_live', confidence=0.95, approved=None),
    ]
    rep = run_replay(trades, recs, 'approve_on_confidence',
                     {'confidence_threshold': 0.9})
    pm = rep.per_module[0]
    assert pm.n_recs_seen == 2
    assert pm.n_recs_approved == 1  # only the 0.95-confidence one


def test_replay_equity_curve_sampled_below_cap():
    base = datetime.utcnow()
    # 300 trades — equity_curve should be sampled down to ~200
    trades = {'sniper': _trades('sniper', 300, 1.0, base)}
    rep = run_replay(trades, [], 'never_approve')
    pm = rep.per_module[0]
    assert pm.n_trades == 300
    assert len(pm.equity_curve) <= 201


def test_replay_unknown_strategy_raises():
    with pytest.raises(ValueError):
        run_replay({}, [], 'nope')
