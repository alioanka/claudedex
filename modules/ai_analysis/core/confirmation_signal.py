"""
AI cross-source confirmation signal (AI-QC-01) — pure logic, no I/O.

Edge source (one sentence): the AI module's market-wide LLM sentiment score
fires on headlines alone, so it routinely buys into falling tape and sells
into rising tape; requiring FREE per-symbol price/volume tape to agree with
the headline direction trades signal frequency for precision (fewer entries,
higher hit-rate), which is the right trade for a strategy whose observed
confidence clusters barely above its execution threshold.

Inputs (all already available to the module for free — no new paid calls):
  1. sentiment_score — the LLM headline score the engine has ALREADY computed
     this cycle (we reuse it; this module never calls an LLM, so there is
     nothing for core/llm_budget to cap here. If anyone later adds paid-LLM
     narration around this signal it MUST go through
     core.llm_budget.try_consume).
  2. closes / volumes — CLOSED candles from the free Binance public klines
     endpoint the module already uses for prices. The caller must drop the
     in-progress candle before passing data in (no look-ahead).

Outputs: a combined directional score in [-1, 1], a confidence in [0, 1],
and a boolean `agree` gate. The engine only uses `agree` to SKIP entries the
sentiment path would otherwise have taken — it never creates entries, never
broadcasts anything, and downstream RiskManager.validate_trade still gates
execution.

Cost/benefit model: skipping a trade costs at most the forgone edge of one
sentiment entry; taking a counter-tape entry historically costs SL distance
(default -3%) plus round-trip fees. With sentiment hit-rates near 50%, a
filter only needs to be mildly informative for the asymmetry to pay.

Self-test: `python -m modules.ai_analysis.core.confirmation_signal`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class ConfirmationParams:
    """Tunables (DB-backed via ai_config; defaults are the migration seeds)."""
    # Minimum combined confidence for the gate to pass.
    min_confidence: float = 0.45
    # Weight of tape momentum in the combined score; (1 - w) goes to the
    # LLM sentiment. 0 makes the gate sentiment-only (pass-through-ish),
    # 1 makes it tape-only.
    momentum_weight: float = 0.5
    # Hard veto: if tape momentum opposes the sentiment direction by more
    # than this (in normalized momentum units), refuse regardless of the
    # combined score. Catches "bullish headlines, falling knife".
    max_opposing_momentum: float = 0.3
    # Momentum horizon definition, in bars of whatever timeframe the caller
    # fetched (engine uses 15m): short = 4 bars (~1h), long = 16 bars (~4h).
    short_lookback: int = 4
    long_lookback: int = 16
    # Returns that saturate the normalized momentum to +-1 (decimal returns:
    # 0.005 = 0.5% over the short horizon, 1.5% over the long horizon).
    short_full_scale: float = 0.005
    long_full_scale: float = 0.015
    # Volume confirmation: mean(last `vol_recent` bars) vs mean(prior
    # `vol_baseline` bars). Ratio scales confidence in [0.75, 1.25].
    vol_recent: int = 4
    vol_baseline: int = 20

    def min_bars(self) -> int:
        """Closed bars required for a fully-formed evaluation."""
        return max(self.long_lookback + 1,
                   self.vol_recent + self.vol_baseline)


@dataclass
class ConfirmationResult:
    agree: bool
    score: float          # combined directional score [-1, 1]
    confidence: float     # [0, 1]
    reason: str
    components: Dict[str, float] = field(default_factory=dict)


def _momentum(closes: Sequence[float], lookback: int,
              full_scale: float) -> Optional[float]:
    """Normalized return over the last `lookback` closed bars, in [-1, 1]."""
    if len(closes) < lookback + 1:
        return None
    past = float(closes[-1 - lookback])
    if past <= 0:
        return None
    ret = float(closes[-1]) / past - 1.0
    if full_scale <= 0:
        return None
    return _clamp(ret / full_scale)


def _volume_factor(volumes: Sequence[float], recent: int,
                   baseline: int) -> float:
    """Confidence multiplier in [0.75, 1.25]; 1.0 at average volume."""
    if len(volumes) < recent + baseline:
        return 1.0
    recent_v = list(volumes)[-recent:]
    base_v = list(volumes)[-(recent + baseline):-recent]
    base_mean = sum(base_v) / len(base_v)
    if base_mean <= 0:
        return 1.0
    ratio = (sum(recent_v) / len(recent_v)) / base_mean
    return _clamp(0.75 + 0.25 * ratio, 0.75, 1.25)


def evaluate_confirmation(
    sentiment_score: float,
    closes: Sequence[float],
    volumes: Sequence[float],
    params: Optional[ConfirmationParams] = None,
) -> ConfirmationResult:
    """Cross-source confirmation of an LLM sentiment direction.

    All requirements must hold for `agree`:
      1. non-zero sentiment with enough CLOSED bars of tape history
         (no evidence -> no confirmation; the gate fails CLOSED),
      2. tape momentum does not oppose the sentiment direction by more than
         max_opposing_momentum (hard veto),
      3. the combined score points the same way as the sentiment,
      4. combined confidence (|combined| scaled by the volume factor)
         >= min_confidence.

    closes/volumes are oldest -> newest and must contain CLOSED candles only.
    """
    p = params or ConfirmationParams()
    if sentiment_score == 0.0:
        return ConfirmationResult(False, 0.0, 0.0, 'zero_sentiment')
    if len(closes) < p.min_bars() or len(volumes) < p.min_bars():
        return ConfirmationResult(
            False, 0.0, 0.0,
            f'insufficient_bars {min(len(closes), len(volumes))}<{p.min_bars()}')

    mom_s = _momentum(closes, p.short_lookback, p.short_full_scale)
    mom_l = _momentum(closes, p.long_lookback, p.long_full_scale)
    if mom_s is None or mom_l is None:
        return ConfirmationResult(False, 0.0, 0.0, 'momentum_unavailable')

    # Short horizon dominates (entries are intraday) but the long horizon
    # stops us confirming a one-bar bounce inside a steady decline.
    momentum = _clamp(0.6 * mom_s + 0.4 * mom_l)
    vol_factor = _volume_factor(volumes, p.vol_recent, p.vol_baseline)

    sent = _clamp(float(sentiment_score))
    w = _clamp(p.momentum_weight, 0.0, 1.0)
    combined = _clamp(w * momentum + (1.0 - w) * sent)
    confidence = _clamp(abs(combined) * vol_factor, 0.0, 1.0)

    components = {
        'sentiment': round(sent, 4),
        'momentum_short': round(mom_s, 4),
        'momentum_long': round(mom_l, 4),
        'momentum': round(momentum, 4),
        'volume_factor': round(vol_factor, 4),
        'combined': round(combined, 4),
    }

    direction = 1.0 if sent > 0 else -1.0
    if momentum * direction < -p.max_opposing_momentum:
        return ConfirmationResult(
            False, combined, confidence,
            f'tape_opposes sentiment (momentum {momentum:+.2f} vs '
            f'direction {direction:+.0f})', components)
    if combined * direction <= 0:
        return ConfirmationResult(
            False, combined, confidence, 'combined_sign_flip', components)
    if confidence < p.min_confidence:
        return ConfirmationResult(
            False, combined, confidence,
            f'confidence {confidence:.2f} < {p.min_confidence:.2f}',
            components)
    return ConfirmationResult(
        True, combined, confidence, 'confirmed', components)


def _self_test() -> None:
    p = ConfirmationParams()
    n = p.min_bars() + 5

    # 1. Bullish sentiment + rising tape + above-average volume -> agree.
    closes = [100.0 * (1.0 + 0.0008 * i) for i in range(n)]
    vols = [100.0] * (n - p.vol_recent) + [180.0] * p.vol_recent
    r = evaluate_confirmation(0.6, closes, vols, p)
    assert r.agree and r.score > 0, r

    # 2. Bullish sentiment vs falling tape -> hard veto.
    closes_dn = [100.0 * (1.0 - 0.0012 * i) for i in range(n)]
    r = evaluate_confirmation(0.6, closes_dn, vols, p)
    assert not r.agree and r.reason.startswith('tape_opposes'), r

    # 3. Bearish sentiment + falling tape -> agree (symmetry).
    r = evaluate_confirmation(-0.6, closes_dn, vols, p)
    assert r.agree and r.score < 0, r

    # 4. Weak sentiment + flat tape -> confidence too low.
    flat = [100.0] * n
    r = evaluate_confirmation(0.36, flat, [100.0] * n, p)
    assert not r.agree and 'confidence' in r.reason, r

    # 5. Not enough history -> fail CLOSED (no evidence, no confirmation).
    r = evaluate_confirmation(0.9, closes[:10], vols[:10], p)
    assert not r.agree and r.reason.startswith('insufficient_bars'), r

    # 6. Zero sentiment never confirms.
    r = evaluate_confirmation(0.0, closes, vols, p)
    assert not r.agree and r.reason == 'zero_sentiment', r

    # 7. Low volume dampens confidence vs the same tape on high volume.
    lo_vol = [100.0] * (n - p.vol_recent) + [20.0] * p.vol_recent
    hi = evaluate_confirmation(0.6, closes, vols, p)
    lo = evaluate_confirmation(0.6, closes, lo_vol, p)
    assert lo.confidence < hi.confidence, (lo.confidence, hi.confidence)

    # 8. One-bar bounce inside a steady decline must not confirm a long:
    # long-horizon leg drags momentum down (and the veto catches it).
    bounce = [100.0 * (1.0 - 0.002 * i) for i in range(n - 1)]
    bounce.append(bounce[-1] * 1.004)  # last closed bar pops +0.4%
    r = evaluate_confirmation(0.6, bounce, vols, p)
    assert not r.agree, r

    # 9. momentum_weight=0 -> sentiment-dominated combined score.
    p0 = ConfirmationParams(momentum_weight=0.0)
    r = evaluate_confirmation(0.8, flat, [100.0] * n, p0)
    assert abs(r.score - 0.8) < 1e-9, r

    print('confirmation_signal self-test OK')


if __name__ == '__main__':
    _self_test()
