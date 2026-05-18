"""Unit tests for modules.orchestrator_ai.core.performance_scorer.

Pure-function tests — no DB, no async, no fixtures other than the
inputs dataclass. Each test pins one decision path.
"""
import pytest

from modules.orchestrator_ai.core.performance_scorer import (
    ModuleScoreInputs,
    score_module,
)


def _inputs(**overrides):
    """Default input that does NOT cross any threshold; tweak per case."""
    base = dict(
        module="sniper",
        closed_trades=50,
        winning_trades=25,
        total_pnl_usd=5.0,
        total_volume_usd=1000.0,
        live_trades=0,
        detection_p95_ms=None,
        jupiter_fallback_hits=None,
        btc_24h_change_pct=None,
        eth_24h_change_pct=None,
    )
    base.update(overrides)
    return ModuleScoreInputs(**base)


def test_sparse_data_returns_hold_with_low_confidence():
    r = score_module(_inputs(closed_trades=10, winning_trades=6))
    assert r.recommended == "hold"
    assert r.confidence < 0.2


def test_meets_all_live_thresholds_recommends_to_live():
    r = score_module(_inputs(
        closed_trades=200,
        winning_trades=120,
        total_pnl_usd=50.0,
    ))
    assert r.recommended == "to_live"
    assert r.confidence == 1.0


def test_just_under_min_trades_does_not_recommend_live():
    # 99 trades < 100 threshold
    r = score_module(_inputs(
        closed_trades=99,
        winning_trades=80,
        total_pnl_usd=50.0,
    ))
    assert r.recommended != "to_live"


def test_just_under_win_rate_does_not_recommend_live():
    # 54% win rate < 55% threshold
    r = score_module(_inputs(
        closed_trades=200,
        winning_trades=108,
        total_pnl_usd=50.0,
    ))
    assert r.recommended != "to_live"


def test_negative_pnl_within_window_holds():
    r = score_module(_inputs(
        closed_trades=80,
        winning_trades=40,
        total_pnl_usd=-30.0,
    ))
    assert r.recommended == "hold"


def test_loss_over_50_dollars_recommends_disable():
    r = score_module(_inputs(
        closed_trades=80,
        winning_trades=20,
        total_pnl_usd=-75.0,
    ))
    assert r.recommended == "disable"
    assert "$-75.00" in r.reason


def test_live_with_negative_pnl_recommends_to_dry():
    r = score_module(_inputs(
        closed_trades=50,
        winning_trades=20,
        total_pnl_usd=-10.0,
        live_trades=5,
    ))
    assert r.recommended == "to_dry"


def test_live_with_positive_pnl_holds():
    # Currently live + making money → orchestrator stays out of the way
    r = score_module(_inputs(
        closed_trades=120,
        winning_trades=70,
        total_pnl_usd=25.0,
        live_trades=10,
    ))
    assert r.recommended == "hold"


def test_score_components_present():
    r = score_module(_inputs())
    assert set(r.components.keys()) == {
        "win_rate", "volume_factor", "pnl_signal", "regime_signal"
    }
    for k, v in r.components.items():
        assert 0.0 <= v <= 1.0, f"{k} out of [0,1]"


def test_zero_trades_does_not_crash():
    r = score_module(_inputs(closed_trades=0, winning_trades=0, total_pnl_usd=0.0))
    assert r.recommended == "hold"
    assert r.confidence == 0.0


def test_btc_volatility_inflates_regime_signal():
    quiet = score_module(_inputs(btc_24h_change_pct=0.5))
    loud = score_module(_inputs(btc_24h_change_pct=15.0))
    assert loud.components["regime_signal"] > quiet.components["regime_signal"]


def test_disable_takes_priority_over_to_live():
    # Even with strong win rate, large loss wins (defensive default).
    r = score_module(_inputs(
        closed_trades=200,
        winning_trades=140,
        total_pnl_usd=-120.0,  # below -$50 threshold
    ))
    assert r.recommended == "disable"


def test_confidence_scales_with_trade_count():
    r1 = score_module(_inputs(closed_trades=10))
    r2 = score_module(_inputs(closed_trades=50))
    r3 = score_module(_inputs(closed_trades=200))
    assert r1.confidence < r2.confidence < r3.confidence + 1e-9
    assert r3.confidence == 1.0
