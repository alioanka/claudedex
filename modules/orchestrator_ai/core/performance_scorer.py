"""Per-module performance scoring.

Pure functions — no DB calls, no async, easy to unit-test. The
orchestrator engine collects raw inputs from DB + market data, hands
them to score_module(), and writes the resulting recommendation row.

The score is deliberately simple at this stage: a weighted sum of
four readable signals. We can swap in something fancier (gradient-
boosted on features the operator labels via approval/rejection) once
we have enough labeled history.

Signal weights are tuned for the "DRY_RUN tells us what to ship live"
question. They are NOT tuned for live performance attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModuleScoreInputs:
    """Raw inputs the orchestrator feeds into score_module()."""

    module: str
    # Trade aggregates over the lookback window (default 24h).
    closed_trades: int
    winning_trades: int
    total_pnl_usd: float
    total_volume_usd: float       # used for fee/gas ratio if available
    # Live (non-simulated) row counts. If non-zero we DOWNGRADE any
    # to_live recommendation (operator already on it).
    live_trades: int
    # Per-trade pnl series for the window — used to compute Sharpe
    # (mean / stdev). When empty, sharpe_signal contributes neutral.
    trade_pnls: List[float] = field(default_factory=list)
    # Sniper-specific extras (None for other modules).
    detection_p95_ms: Optional[float] = None
    jupiter_fallback_hits: Optional[int] = None
    # Market regime — fed from /api/dashboard/summary's external feed.
    btc_24h_change_pct: Optional[float] = None
    eth_24h_change_pct: Optional[float] = None


def _sharpe(pnls: List[float]) -> Optional[float]:
    """Per-trade Sharpe = mean / stdev. Returns None when the series
    has fewer than 5 trades (too noisy) or stdev is zero (all same
    P&L, can happen with paper trades at a constant entry size)."""
    if len(pnls) < 5:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
    stdev = math.sqrt(var)
    if stdev == 0:
        return None
    return mean / stdev


@dataclass
class ModuleScore:
    """Output of score_module()."""

    score: float            # 0.0..1.0
    confidence: float       # 0.0..1.0
    recommended: str        # 'enable' | 'disable' | 'to_dry' | 'to_live' | 'hold'
    reason: str
    components: dict        # breakdown of the four signals for audit


# Thresholds that determine the recommendation. Adjustable via config
# in a future commit; hard-coded here so unit tests are deterministic.
_MIN_TRADES_FOR_LIVE_RECOMMENDATION = 100
_MIN_WIN_RATE_FOR_LIVE = 0.55
_MIN_PNL_USD_FOR_LIVE = 10.0
_MAX_LOSS_FOR_DISABLE_RECOMMENDATION = -50.0


def score_module(inputs: ModuleScoreInputs) -> ModuleScore:
    """Score a single module's recent DRY_RUN performance.

    Returns a ModuleScore with one of:
      - 'to_live'  : DRY_RUN performance looks robust enough to flip live
      - 'to_dry'   : was running live but recent performance suggests pulling back
      - 'disable'  : losing > $50 over the lookback window
      - 'enable'   : module is DISABLED but conditions favor turning it on
      - 'hold'     : keep current state (the most common output)
    """
    # ----- Win rate (signal 1) -----
    win_rate = (
        inputs.winning_trades / inputs.closed_trades
        if inputs.closed_trades > 0 else 0.0
    )

    # ----- Trade volume sufficiency (signal 2) -----
    # If we have too few trades the score is unreliable. confidence
    # is multiplied by `volume_factor` so under-sampled modules get
    # low confidence on whatever they recommend.
    if inputs.closed_trades <= 0:
        volume_factor = 0.0
    elif inputs.closed_trades >= _MIN_TRADES_FOR_LIVE_RECOMMENDATION:
        volume_factor = 1.0
    else:
        volume_factor = inputs.closed_trades / _MIN_TRADES_FOR_LIVE_RECOMMENDATION

    # ----- Profitability ratio (signal 3) -----
    pnl_signal = 0.5
    if inputs.total_pnl_usd > _MIN_PNL_USD_FOR_LIVE:
        # Saturating: $10..$100 maps to 0.5..1.0
        pnl_signal = 0.5 + min(
            (inputs.total_pnl_usd - _MIN_PNL_USD_FOR_LIVE) / 180.0, 0.5
        )
    elif inputs.total_pnl_usd < 0:
        # Saturating: 0..-$50 maps to 0.5..0.0
        pnl_signal = max(0.5 + inputs.total_pnl_usd / 100.0, 0.0)

    # ----- Market regime (signal 4) -----
    # Trending markets favor sniper + arb + futures-momentum strategies;
    # chop favors mean-reversion + copy-trading. Without a per-module
    # regime preference we just neutral-bias here. Future work: per-
    # module regime preferences in DB.
    if inputs.btc_24h_change_pct is not None:
        regime_signal = 0.5 + min(abs(inputs.btc_24h_change_pct) / 20.0, 0.5)
    else:
        regime_signal = 0.5

    # ----- Sharpe (signal 5) -----
    # Per-trade Sharpe over the lookback window. Strong positive
    # Sharpe (> 1.0 per trade is exceptional) bumps the score; zero
    # or negative drags it down. None (insufficient data) is neutral.
    sharpe = _sharpe(inputs.trade_pnls)
    if sharpe is None:
        sharpe_signal = 0.5
    else:
        # Saturating: Sharpe of -0.5 .. +1.5 maps to 0 .. 1.
        sharpe_signal = max(0.0, min((sharpe + 0.5) / 2.0, 1.0))

    # ----- Weighted combination -----
    components = {
        'win_rate':      round(win_rate, 4),
        'volume_factor': round(volume_factor, 4),
        'pnl_signal':    round(pnl_signal, 4),
        'regime_signal': round(regime_signal, 4),
        'sharpe_signal': round(sharpe_signal, 4),
        'sharpe':        round(sharpe, 4) if sharpe is not None else None,
    }
    # Reweighted: Sharpe earns 0.20 from pnl_signal + volume_factor.
    score = (
        0.35 * win_rate
        + 0.20 * pnl_signal
        + 0.20 * sharpe_signal
        + 0.10 * regime_signal
        + 0.15 * volume_factor
    )
    confidence = volume_factor  # confidence tracks data sufficiency

    # ----- Decide recommendation -----
    if inputs.total_pnl_usd < _MAX_LOSS_FOR_DISABLE_RECOMMENDATION:
        return ModuleScore(
            score=score,
            confidence=confidence,
            recommended='disable',
            reason=(
                f"P&L over window = ${inputs.total_pnl_usd:.2f} "
                f"(threshold ${_MAX_LOSS_FOR_DISABLE_RECOMMENDATION:.2f})"
            ),
            components=components,
        )

    if (
        inputs.closed_trades >= _MIN_TRADES_FOR_LIVE_RECOMMENDATION
        and win_rate >= _MIN_WIN_RATE_FOR_LIVE
        and inputs.total_pnl_usd >= _MIN_PNL_USD_FOR_LIVE
        and inputs.live_trades == 0
    ):
        return ModuleScore(
            score=score,
            confidence=confidence,
            recommended='to_live',
            reason=(
                f"DRY_RUN stats: {inputs.closed_trades} trades, "
                f"win_rate={win_rate*100:.1f}%, P&L=${inputs.total_pnl_usd:.2f} "
                f"— PAPER evidence only (no real-fill validation); confirm "
                f"execution quality (TCA) and net-of-cost edge before approving"
            ),
            components=components,
        )

    # Pulled back from live: live_trades > 0 AND total_pnl negative
    if inputs.live_trades > 0 and inputs.total_pnl_usd < 0:
        return ModuleScore(
            score=score,
            confidence=confidence,
            recommended='to_dry',
            reason=(
                f"Live P&L over window negative (${inputs.total_pnl_usd:.2f}); "
                f"recommend reverting to DRY_RUN to avoid further loss."
            ),
            components=components,
        )

    return ModuleScore(
        score=score,
        confidence=confidence,
        recommended='hold',
        reason='Within thresholds; no action recommended.',
        components=components,
    )
