"""Pure decision math for the meta controller.

No DB calls, no async, no LLM — every number in the output is derivable
by the operator from the inputs with a calculator. The engine
(meta_engine.py) collects per-module DRY_RUN / LIVE aggregates from the
trade tables and hands them to ``decide()``; this file turns them into
a transparent ACTIVATE / KEEP / PAUSE decision plus a health score.

Self-test: ``python -m modules.meta_controller.core.health_scorer``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

DECISION_ACTIVATE = "activate"
DECISION_KEEP = "keep"
DECISION_PAUSE = "pause"


@dataclass
class TrackStats:
    """Aggregates for ONE track (dry or live) over the lookback window."""

    closed_trades: int = 0
    winning_trades: int = 0
    total_pnl_usd: float = 0.0
    # Most-recent-first per-trade P&L series (capped upstream, ~500).
    trade_pnls: List[float] = field(default_factory=list)


@dataclass
class ModulePerf:
    module: str
    dry: TrackStats = field(default_factory=TrackStats)
    live: TrackStats = field(default_factory=TrackStats)


@dataclass
class MetaThresholds:
    """All knobs DB-configurable via config_settings('meta_config', ...)."""

    pause_score: float = 0.35      # health below this (w/ confidence) -> PAUSE
    activate_score: float = 0.55   # health above this (w/ confidence) -> ACTIVATE
    min_confidence: float = 0.5    # below this, never actuate (KEEP + advisory note)
    pause_loss_usd: float = 25.0   # hard rule: LIVE window loss beyond this -> PAUSE
    min_trades: int = 10           # window trade floor for any non-KEEP decision
    live_weight: float = 0.7       # blend weight of LIVE track when both exist
    n_target: int = 50             # trades at which confidence saturates to 1.0


@dataclass
class MetaDecision:
    module: str
    decision: str        # 'activate' | 'keep' | 'pause'
    health_score: float  # 0..1 blended health/edge score (0 when unscorable)
    confidence: float    # 0..1, tracks data sufficiency
    reason: str          # operator-readable, contains the actual numbers
    components: dict     # full per-signal breakdown for the dashboard


# ---------------------------------------------------------------- helpers

def sharpe(pnls: List[float]) -> Optional[float]:
    """Per-trade Sharpe = mean / stdev. None when < 5 trades or stdev 0.
    Matches the orchestrator's convention so the two scores are comparable."""
    if len(pnls) < 5:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
    stdev = math.sqrt(var)
    if stdev == 0:
        return None
    return mean / stdev


def max_drawdown(pnls: List[float]) -> tuple:
    """(dd_usd, dd_frac) of the cumulative P&L curve.

    `pnls` is most-recent-first (how the engine fetches them); we reverse
    to chronological order before accumulating. dd_frac is the worst
    peak-to-trough drop normalized by the running peak AT THE TROUGH
    (standard max-drawdown %), not by the final peak. A window whose
    cumulative curve never goes positive maps to dd_frac 1.0.
    """
    if not pnls:
        return 0.0, 0.0
    cum = 0.0
    peak = 0.0
    ever_positive = False
    dd_usd = 0.0
    dd_frac = 0.0
    for p in reversed(pnls):
        cum += p
        if cum > peak:
            peak = cum
        if peak > 0:
            ever_positive = True
        drawdown = peak - cum
        if drawdown > dd_usd:
            dd_usd = drawdown
        if peak > 0:
            frac = drawdown / peak
            if frac > dd_frac:
                dd_frac = frac
    if dd_usd <= 0:
        return 0.0, 0.0
    if not ever_positive:
        # curve never went positive: an all-loss window maps to 1.0.
        dd_frac = 1.0
    return dd_usd, min(1.0, dd_frac)


def profit_factor(pnls: List[float]) -> Optional[float]:
    """gross wins / gross losses. None when there are no losses
    (undefined — treated as best-case upstream)."""
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses <= 0:
        return None
    return wins / losses


def score_track(stats: TrackStats, n_target: int = 50) -> tuple:
    """(score | None, confidence, components) for one track.

    Five readable signals, each mapped to [0,1]:
      win_rate_sig    raw win rate
      sharpe_sig      per-trade Sharpe -0.5..+1.5 -> 0..1 (None -> 0.5 neutral)
      expectancy_sig  expectancy / mean(|pnl|), -0.5..+0.5 -> 0..1
      pf_sig          profit factor 0..2 -> 0..1 (no losses -> 1.0 if any win)
      dd_sig          1 - drawdown fraction of the cumulative curve

    score = 0.25*win_rate + 0.25*sharpe + 0.20*expectancy + 0.15*pf + 0.15*dd
    confidence = min(1, closed_trades / n_target)
    """
    n = stats.closed_trades
    if n <= 0:
        return None, 0.0, {"closed_trades": 0}

    win_rate = stats.winning_trades / n
    pnls = stats.trade_pnls

    s = sharpe(pnls)
    sharpe_sig = 0.5 if s is None else max(0.0, min(1.0, (s + 0.5) / 2.0))

    expectancy = stats.total_pnl_usd / n
    mean_abs = (sum(abs(p) for p in pnls) / len(pnls)) if pnls else 0.0
    if mean_abs > 0:
        e_norm = expectancy / mean_abs            # roughly -1..+1
        expectancy_sig = max(0.0, min(1.0, 0.5 + e_norm))
    else:
        expectancy_sig = 0.5

    pf = profit_factor(pnls)
    if pf is None:
        pf_sig = 1.0 if stats.winning_trades > 0 else 0.5
    else:
        pf_sig = max(0.0, min(1.0, pf / 2.0))

    dd_usd, dd_frac = max_drawdown(pnls)
    dd_sig = 1.0 - dd_frac

    score = (
        0.25 * win_rate
        + 0.25 * sharpe_sig
        + 0.20 * expectancy_sig
        + 0.15 * pf_sig
        + 0.15 * dd_sig
    )
    confidence = min(1.0, n / max(1, n_target))
    components = {
        "closed_trades": n,
        "win_rate": round(win_rate, 4),
        "total_pnl_usd": round(stats.total_pnl_usd, 4),
        "expectancy_usd": round(expectancy, 4),
        "sharpe": round(s, 4) if s is not None else None,
        "profit_factor": round(pf, 4) if pf is not None else None,
        "max_drawdown_usd": round(dd_usd, 4),
        "max_drawdown_frac": round(dd_frac, 4),
        "signals": {
            "win_rate_sig": round(win_rate, 4),
            "sharpe_sig": round(sharpe_sig, 4),
            "expectancy_sig": round(expectancy_sig, 4),
            "pf_sig": round(pf_sig, 4),
            "dd_sig": round(dd_sig, 4),
        },
        "score": round(score, 4),
        "confidence": round(confidence, 4),
    }
    return score, confidence, components


def score_module_health(perf: ModulePerf, th: MetaThresholds) -> tuple:
    """(health_score | None, confidence, components) — DRY and LIVE
    scored separately, then blended. LIVE evidence dominates when both
    exist (live_weight); a dry-only module is scored on dry alone."""
    dry_score, dry_conf, dry_comp = score_track(perf.dry, th.n_target)
    live_score, live_conf, live_comp = score_track(perf.live, th.n_target)

    components = {"dry": dry_comp, "live": live_comp,
                  "live_weight": th.live_weight}

    if live_score is not None and dry_score is not None:
        w = th.live_weight
        health = w * live_score + (1.0 - w) * dry_score
        conf = w * live_conf + (1.0 - w) * dry_conf
        components["blend"] = "live+dry"
    elif live_score is not None:
        health, conf = live_score, live_conf
        components["blend"] = "live_only"
    elif dry_score is not None:
        health, conf = dry_score, dry_conf
        components["blend"] = "dry_only"
    else:
        return None, 0.0, components
    components["health_score"] = round(health, 4)
    components["health_confidence"] = round(conf, 4)
    return health, conf, components


def decide(
    perf: ModulePerf,
    *,
    currently_paused: bool,
    th: MetaThresholds,
) -> MetaDecision:
    """Transparent decision rules, evaluated in order:

    1. INSUFFICIENT DATA: total window trades < min_trades -> KEEP
       (confidence 0, never actuated).
    2. HARD LOSS RULE: LIVE window P&L < -pause_loss_usd -> PAUSE
       (this fires regardless of score; losing real money outranks any
       blended signal). Dry-only modules trip the same rule at 2x the
       threshold (paper losses matter less, but a badly bleeding dry
       track wastes attention and signals a broken edge).
    3. SCORE BAND with hysteresis:
         health <= pause_score                       -> PAUSE
         health >= activate_score AND paused         -> ACTIVATE
         otherwise                                   -> KEEP
       The activate/pause gap is the anti-flap hysteresis; the engine
       additionally enforces a minimum dwell time between actuations.
    Low confidence (< min_confidence) downgrades any PAUSE/ACTIVATE to
    an advisory KEEP-with-note so thin data never actuates.
    """
    n_total = perf.dry.closed_trades + perf.live.closed_trades
    health, conf, components = score_module_health(perf, th)

    if n_total < th.min_trades or health is None:
        return MetaDecision(
            module=perf.module, decision=DECISION_KEEP,
            health_score=health if health is not None else 0.0,
            confidence=0.0,
            reason=(
                f"Insufficient data: {n_total} closed trades in window "
                f"(need >= {th.min_trades}). KEEP current state."
            ),
            components={**components, "rule": "insufficient_data"},
        )

    # Hard loss rule.
    live_pnl = perf.live.total_pnl_usd
    dry_pnl = perf.dry.total_pnl_usd
    if perf.live.closed_trades > 0 and live_pnl < -th.pause_loss_usd:
        return MetaDecision(
            module=perf.module, decision=DECISION_PAUSE,
            health_score=health, confidence=max(conf, 0.9),
            reason=(
                f"HARD LOSS RULE: LIVE P&L ${live_pnl:.2f} over window "
                f"breaches -${th.pause_loss_usd:.2f}. PAUSE."
            ),
            components={**components, "rule": "hard_loss_live"},
        )
    if perf.live.closed_trades == 0 and dry_pnl < -2.0 * th.pause_loss_usd:
        return MetaDecision(
            module=perf.module, decision=DECISION_PAUSE,
            health_score=health, confidence=conf,
            reason=(
                f"HARD LOSS RULE (dry track): DRY_RUN P&L ${dry_pnl:.2f} "
                f"breaches -${2.0 * th.pause_loss_usd:.2f} (2x live "
                f"threshold). PAUSE — edge looks broken even on paper."
            ),
            components={**components, "rule": "hard_loss_dry"},
        )

    # Score band with hysteresis.
    if health <= th.pause_score:
        decision, rule = DECISION_PAUSE, "score_below_pause"
        reason = (
            f"Health {health:.3f} <= pause threshold {th.pause_score:.2f} "
            f"(conf {conf:.2f}). PAUSE."
        )
    elif currently_paused and health >= th.activate_score:
        decision, rule = DECISION_ACTIVATE, "score_above_activate"
        reason = (
            f"Health {health:.3f} >= activate threshold "
            f"{th.activate_score:.2f} (conf {conf:.2f}) and module is "
            f"paused. ACTIVATE."
        )
    else:
        decision, rule = DECISION_KEEP, "score_in_band"
        reason = (
            f"Health {health:.3f} in [{th.pause_score:.2f}, "
            f"{th.activate_score:.2f}) hysteresis band or no state change "
            f"needed (paused={currently_paused}, conf {conf:.2f}). KEEP."
        )

    # Thin data never actuates: downgrade to advisory KEEP.
    if decision != DECISION_KEEP and conf < th.min_confidence:
        reason = (
            f"Signal says {decision.upper()} (health {health:.3f}) but "
            f"confidence {conf:.2f} < {th.min_confidence:.2f}. Downgraded "
            f"to KEEP (advisory only)."
        )
        decision, rule = DECISION_KEEP, f"low_confidence_downgrade"

    return MetaDecision(
        module=perf.module, decision=decision,
        health_score=health, confidence=conf,
        reason=reason,
        components={**components, "rule": rule},
    )


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    th = MetaThresholds()

    # 1. Insufficient data -> KEEP, confidence 0.
    d = decide(ModulePerf("dex"), currently_paused=False, th=th)
    assert d.decision == DECISION_KEEP and d.confidence == 0.0, d

    # 2. Healthy live module -> KEEP (not paused, nothing to activate).
    winners = [5.0, 4.0, 6.0, 5.5, -2.0] * 12   # 60 trades, strongly positive
    perf = ModulePerf(
        "futures",
        live=TrackStats(60, 48, sum(winners), winners),
    )
    d = decide(perf, currently_paused=False, th=th)
    assert d.decision == DECISION_KEEP, d
    assert d.health_score > th.activate_score, d

    # 3. Same healthy module but paused -> ACTIVATE.
    d = decide(perf, currently_paused=True, th=th)
    assert d.decision == DECISION_ACTIVATE, d

    # 4. Hard loss rule on live track -> PAUSE regardless of score.
    losers = [-3.0, 2.0, -4.0, -3.5, 1.0] * 12   # net -$90
    perf = ModulePerf("sniper", live=TrackStats(60, 18, sum(losers), losers))
    d = decide(perf, currently_paused=False, th=th)
    assert d.decision == DECISION_PAUSE and "HARD LOSS" in d.reason, d

    # 5. Dry-only bleed at 2x threshold -> PAUSE.
    perf = ModulePerf("arbitrage", dry=TrackStats(60, 18, -60.0, losers))
    d = decide(perf, currently_paused=False, th=th)
    assert d.decision == DECISION_PAUSE and "dry track" in d.reason, d

    # 6. Low-score but thin-confidence module -> downgraded KEEP.
    thin = [-1.0, -1.2, 0.5, -0.8, -1.1, -0.9, 0.3, -1.0, -0.7, -1.3,
            -0.5, -0.6]  # 12 trades, net ~ -$9.3 (above hard-loss line)
    perf = ModulePerf("ai", dry=TrackStats(12, 2, sum(thin), thin))
    d = decide(perf, currently_paused=False, th=th)
    assert d.decision == DECISION_KEEP and "Downgraded" in d.reason, d

    # 7. Blend: live dominates dry at live_weight.
    perf = ModulePerf(
        "copy_trading",
        dry=TrackStats(60, 50, 100.0, winners),
        live=TrackStats(60, 10, -10.0, [-0.2] * 30 + [0.1] * 30),
    )
    h, c, comp = score_module_health(perf, th)
    assert comp["blend"] == "live+dry" and h is not None
    live_only, _, _ = score_track(perf.live, th.n_target)
    dry_only, _, _ = score_track(perf.dry, th.n_target)
    assert live_only < h < dry_only, (live_only, h, dry_only)

    # 8. max_drawdown sanity: chronological loss-then-recovery.
    dd_usd, dd_frac = max_drawdown([5.0, -3.0, 10.0])  # most-recent-first
    # chronological: +10 -> 10 peak; -3 -> dd 3; +5 -> recovered.
    assert abs(dd_usd - 3.0) < 1e-9 and abs(dd_frac - 0.3) < 1e-9

    print("health_scorer self-test OK")


if __name__ == "__main__":
    _selftest()
