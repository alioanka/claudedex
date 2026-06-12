"""Pure volatility-regime classification + regime-conditioned weight math.

No DB calls, no async, no network, no LLM — every number in the output is
derivable by the operator from the inputs with a calculator. The engine
(regime_engine.py) fetches BTC/ETH closes from a free public API and hands the
chronological close series to ``classify()``; this file turns them into a
transparent regime label + confidence, and ``propose_weights()`` turns a regime
into per-module capital-weight proposals (ADVISORY — nothing here trades).

Regime taxonomy (vol dimension x trend dimension):
    trend_expansion    vol expanding, market trending   -> momentum modules up
    chop_expansion     vol expanding, no trend (chop)   -> de-risk, arb up, extra reserve
    trend_compression  vol compressing, grind trend     -> carry/copy/futures grind
    range_compression  vol compressing, rangebound      -> arb / mean-capture up
    neutral            vol signal unclear               -> no tilt (all 1.0)

Self-test: ``python -m modules.regime_allocator.core.regime_classifier``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

REGIME_TREND_EXPANSION = "trend_expansion"
REGIME_CHOP_EXPANSION = "chop_expansion"
REGIME_TREND_COMPRESSION = "trend_compression"
REGIME_RANGE_COMPRESSION = "range_compression"
REGIME_NEUTRAL = "neutral"

ALL_REGIMES = (
    REGIME_TREND_EXPANSION, REGIME_CHOP_EXPANSION,
    REGIME_TREND_COMPRESSION, REGIME_RANGE_COMPRESSION, REGIME_NEUTRAL,
)

# Modules this allocator reasons about (matches portfolio_allocator/_MODULES
# and the orchestrator's module set — short engine keys).
ALLOC_MODULES = ["sniper", "arbitrage", "copy_trading", "futures", "solana", "dex", "ai"]

# Regime -> per-module tilt multipliers (1.0 = no change). DB-overridable via
# config_settings('regime_allocator','regime_tilts') as a JSON string. The
# direction follows the desk logic: momentum/trend modules earn more when vol
# expands WITH direction; arb / mean-capture earns its keep when vol compresses
# (spreads grind, gas wars calm down); expanding vol WITHOUT direction (chop)
# is where momentum books bleed, so everything de-risks and reserve grows.
DEFAULT_TILTS: Dict[str, Dict[str, float]] = {
    REGIME_TREND_EXPANSION: {
        "futures": 1.40, "dex": 1.30, "solana": 1.20, "sniper": 1.20,
        "copy_trading": 1.10, "ai": 1.00, "arbitrage": 0.70,
    },
    REGIME_CHOP_EXPANSION: {
        "futures": 0.80, "dex": 0.80, "solana": 0.80, "sniper": 0.70,
        "copy_trading": 0.80, "ai": 0.90, "arbitrage": 1.20,
    },
    REGIME_TREND_COMPRESSION: {
        "futures": 1.20, "dex": 1.00, "solana": 1.00, "sniper": 0.90,
        "copy_trading": 1.20, "ai": 1.00, "arbitrage": 1.10,
    },
    REGIME_RANGE_COMPRESSION: {
        "futures": 0.80, "dex": 0.90, "solana": 0.90, "sniper": 0.80,
        "copy_trading": 1.00, "ai": 1.00, "arbitrage": 1.40,
    },
    REGIME_NEUTRAL: {m: 1.00 for m in ALLOC_MODULES},
}


@dataclass
class RegimeParams:
    """All knobs DB-configurable via config_settings('regime_allocator', ...)."""

    short_vol_bars: int = 24      # recent realized-vol window (bars)
    long_vol_bars: int = 96       # baseline realized-vol window (bars)
    trend_bars: int = 42          # efficiency-ratio window (bars)
    vol_expand_ratio: float = 1.15   # short_rv/long_rv at/above -> expansion
    vol_compress_ratio: float = 0.85  # at/below -> compression
    er_trend: float = 0.35        # efficiency ratio at/above -> trending
    er_range: float = 0.20        # at/below -> rangebound
    btc_weight: float = 0.6       # BTC vote weight (ETH gets 1 - btc_weight)


@dataclass
class AssetState:
    """Per-asset regime inputs — every field operator-checkable."""

    symbol: str
    bars: int
    vol_ratio: Optional[float] = None   # short RV / long RV
    efficiency_ratio: Optional[float] = None  # Kaufman ER over trend_bars
    direction: int = 0                  # sign of net move over trend_bars
    vol_state: str = "neutral"          # 'expansion' | 'compression' | 'neutral'
    trend_state: str = "neutral"        # 'trend' | 'range' | 'neutral'


@dataclass
class RegimeResult:
    regime: str
    confidence: float          # 0..1 — agreement between assets + signal strength
    reason: str                # operator-readable, contains the actual numbers
    assets: List[AssetState] = field(default_factory=list)
    components: dict = field(default_factory=dict)


@dataclass
class WeightProposal:
    module: str
    weight_pct: float
    reason: str
    components: dict = field(default_factory=dict)


# ------------------------------------------------------------------ math

def log_returns(closes: Sequence[float]) -> List[float]:
    """Chronological log returns; non-positive prices are skipped (fail-soft)."""
    out: List[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def realized_vol(returns: Sequence[float], bars: int) -> Optional[float]:
    """Population stdev of the LAST `bars` returns. None when data is thin or
    flat. Per-bar (not annualized) — the regime test is a ratio, so the
    annualization constant cancels."""
    window = list(returns[-bars:])
    if len(window) < max(5, bars // 2):
        return None
    mean = sum(window) / len(window)
    var = sum((r - mean) ** 2 for r in window) / len(window)
    sd = math.sqrt(var)
    return sd if sd > 0 else None


def efficiency_ratio(closes: Sequence[float], bars: int) -> Optional[float]:
    """Kaufman efficiency ratio over the last `bars` closes:
    |net move| / sum(|bar moves|). 1.0 = perfect trend, ~0 = pure chop."""
    window = list(closes[-(bars + 1):])
    if len(window) < max(5, bars // 2):
        return None
    path = sum(abs(b - a) for a, b in zip(window, window[1:]))
    if path <= 0:
        return None
    net = abs(window[-1] - window[0])
    return net / path


def classify_asset(symbol: str, closes: Sequence[float], p: RegimeParams) -> AssetState:
    """One asset -> (vol_state, trend_state) with the raw numbers attached."""
    st = AssetState(symbol=symbol, bars=len(closes))
    if len(closes) < p.long_vol_bars + 2:
        return st  # insufficient data -> neutral/neutral
    rets = log_returns(closes)
    short_rv = realized_vol(rets, p.short_vol_bars)
    long_rv = realized_vol(rets, p.long_vol_bars)
    if short_rv is not None and long_rv is not None and long_rv > 0:
        st.vol_ratio = short_rv / long_rv
        if st.vol_ratio >= p.vol_expand_ratio:
            st.vol_state = "expansion"
        elif st.vol_ratio <= p.vol_compress_ratio:
            st.vol_state = "compression"
    er = efficiency_ratio(closes, p.trend_bars)
    if er is not None:
        st.efficiency_ratio = er
        if er >= p.er_trend:
            st.trend_state = "trend"
        elif er <= p.er_range:
            st.trend_state = "range"
    window = list(closes[-(p.trend_bars + 1):])
    if len(window) >= 2:
        delta = window[-1] - window[0]
        st.direction = 1 if delta > 0 else (-1 if delta < 0 else 0)
    return st


def _vote(states: Sequence[str], weights: Sequence[float],
          pos: str, neg: str) -> float:
    """Weighted vote in [-1, +1]: +1 = all-pos, -1 = all-neg, 0 = neutral."""
    total = 0.0
    for s, w in zip(states, weights):
        if s == pos:
            total += w
        elif s == neg:
            total -= w
    return total


def _strength(asset: AssetState, p: RegimeParams) -> float:
    """How far past its thresholds this asset's signals sit, 0..1.
    Mean of the vol-ratio exceedance and the ER exceedance, each normalized
    by a fixed scale so the number is comparable across configs."""
    parts: List[float] = []
    if asset.vol_ratio is not None:
        if asset.vol_state == "expansion":
            parts.append(min(1.0, (asset.vol_ratio - p.vol_expand_ratio) / 0.5))
        elif asset.vol_state == "compression":
            parts.append(min(1.0, (p.vol_compress_ratio - asset.vol_ratio) / 0.4))
        else:
            parts.append(0.0)
    if asset.efficiency_ratio is not None:
        if asset.trend_state == "trend":
            parts.append(min(1.0, (asset.efficiency_ratio - p.er_trend) / 0.3))
        elif asset.trend_state == "range":
            parts.append(min(1.0, (p.er_range - asset.efficiency_ratio) / 0.15))
        else:
            parts.append(0.0)
    if not parts:
        return 0.0
    return max(0.0, min(1.0, sum(parts) / len(parts)))


def classify(btc_closes: Sequence[float], eth_closes: Sequence[float],
             p: Optional[RegimeParams] = None) -> RegimeResult:
    """Chronological BTC + ETH closes -> regime label + confidence.

    Decision rules, in order:
      1. INSUFFICIENT DATA on both assets -> neutral, confidence 0.
      2. Weighted vol vote (btc_weight vs 1-btc_weight): >= +0.5 -> expansion,
         <= -0.5 -> compression, else neutral vol -> regime 'neutral'.
      3. Weighted trend vote: >= +0.5 -> trending, else not.
      4. (vol, trend) -> regime per the taxonomy in the module docstring.
    Confidence = 0.5*cross-asset agreement + 0.5*mean threshold exceedance.
    """
    p = p or RegimeParams()
    btc = classify_asset("BTC", btc_closes, p)
    eth = classify_asset("ETH", eth_closes, p)
    assets = [btc, eth]
    weights = [p.btc_weight, 1.0 - p.btc_weight]

    usable = [a for a in assets if a.vol_ratio is not None]
    if not usable:
        return RegimeResult(
            regime=REGIME_NEUTRAL, confidence=0.0,
            reason=(f"Insufficient data: BTC bars={btc.bars}, ETH bars={eth.bars} "
                    f"(need >= {p.long_vol_bars + 2}). NEUTRAL, confidence 0."),
            assets=assets, components={"rule": "insufficient_data"},
        )

    vol_vote = _vote([a.vol_state for a in assets], weights, "expansion", "compression")
    trend_vote = _vote([a.trend_state for a in assets], weights, "trend", "range")

    if vol_vote >= 0.5:
        vol_dim = "expansion"
    elif vol_vote <= -0.5:
        vol_dim = "compression"
    else:
        vol_dim = "neutral"
    trending = trend_vote >= 0.5

    if vol_dim == "expansion":
        regime = REGIME_TREND_EXPANSION if trending else REGIME_CHOP_EXPANSION
    elif vol_dim == "compression":
        regime = REGIME_TREND_COMPRESSION if trending else REGIME_RANGE_COMPRESSION
    else:
        regime = REGIME_NEUTRAL

    agreement = 0.0
    if btc.vol_state == eth.vol_state:
        agreement += 0.5
    if btc.trend_state == eth.trend_state:
        agreement += 0.5
    strength = sum(_strength(a, p) * w for a, w in zip(assets, weights))
    confidence = max(0.0, min(1.0, 0.5 * agreement + 0.5 * strength))
    if regime == REGIME_NEUTRAL:
        confidence = min(confidence, 0.5)  # a non-signal is never high-conviction

    def _fmt(a: AssetState) -> str:
        vr = f"{a.vol_ratio:.2f}" if a.vol_ratio is not None else "n/a"
        er = f"{a.efficiency_ratio:.2f}" if a.efficiency_ratio is not None else "n/a"
        return f"{a.symbol}: vol_ratio={vr} ({a.vol_state}), ER={er} ({a.trend_state})"

    reason = (f"{regime.upper()} (conf {confidence:.2f}). "
              f"{_fmt(btc)}; {_fmt(eth)}. "
              f"vol_vote={vol_vote:+.2f}, trend_vote={trend_vote:+.2f}.")
    return RegimeResult(
        regime=regime, confidence=confidence, reason=reason, assets=assets,
        components={
            "vol_vote": round(vol_vote, 3),
            "trend_vote": round(trend_vote, 3),
            "agreement": agreement,
            "strength": round(strength, 3),
            "btc": {"vol_ratio": btc.vol_ratio, "efficiency_ratio": btc.efficiency_ratio,
                    "vol_state": btc.vol_state, "trend_state": btc.trend_state,
                    "direction": btc.direction, "bars": btc.bars},
            "eth": {"vol_ratio": eth.vol_ratio, "efficiency_ratio": eth.efficiency_ratio,
                    "vol_state": eth.vol_state, "trend_state": eth.trend_state,
                    "direction": eth.direction, "bars": eth.bars},
            "params": {"short_vol_bars": p.short_vol_bars, "long_vol_bars": p.long_vol_bars,
                       "trend_bars": p.trend_bars, "vol_expand_ratio": p.vol_expand_ratio,
                       "vol_compress_ratio": p.vol_compress_ratio,
                       "er_trend": p.er_trend, "er_range": p.er_range},
            "rule": "vote",
        },
    )


# ------------------------------------------------------------- weights

def propose_weights(
    regime: str,
    confidence: float,
    enabled_modules: Dict[str, bool],
    *,
    base_weights: Optional[Dict[str, float]] = None,
    tilts: Optional[Dict[str, Dict[str, float]]] = None,
    reserve_pct: float = 10.0,
    chop_extra_reserve_pct: float = 10.0,
    min_confidence: float = 0.25,
) -> List[WeightProposal]:
    """Regime + confidence -> per-module weight proposals (% of book).

    Math (all operator-checkable):
      eff_tilt_i = 1 + confidence * (tilt_i(regime) - 1)   # confidence-scaled
      raw_i      = base_i * eff_tilt_i                      # enabled modules only
      weight_i   = raw_i / sum(raw) * (100 - reserve)       # normalized
    Below min_confidence the tilt collapses to 1.0 (a low-conviction regime
    read must not move capital). chop_expansion adds extra reserve — expanding
    vol without direction is where momentum books bleed. Disabled modules get
    an explicit 0% row so the dashboard table stays complete.
    """
    tilts = tilts or DEFAULT_TILTS
    regime_tilts = tilts.get(regime, DEFAULT_TILTS[REGIME_NEUTRAL])
    base = base_weights or {m: 1.0 for m in ALLOC_MODULES}
    conf = max(0.0, min(1.0, confidence))
    applied_conf = conf if conf >= min_confidence else 0.0

    reserve = max(0.0, min(95.0, reserve_pct))
    if regime == REGIME_CHOP_EXPANSION and applied_conf > 0:
        reserve = max(0.0, min(95.0, reserve + chop_extra_reserve_pct * applied_conf))
    risk_budget = 100.0 - reserve

    raw: Dict[str, float] = {}
    eff_tilts: Dict[str, float] = {}
    for m in ALLOC_MODULES:
        if not enabled_modules.get(m, False):
            continue
        tilt = float(regime_tilts.get(m, 1.0))
        eff = 1.0 + applied_conf * (tilt - 1.0)
        eff_tilts[m] = eff
        raw[m] = max(0.0, float(base.get(m, 1.0))) * eff

    total = sum(raw.values())
    proposals: List[WeightProposal] = []
    for m in ALLOC_MODULES:
        if m in raw and total > 0:
            pct = round(raw[m] / total * risk_budget, 2)
            note = ("" if applied_conf > 0 else
                    f" (confidence {conf:.2f} < {min_confidence:.2f} floor -> tilt suppressed)")
            reason = (f"{regime}: base={base.get(m, 1.0):.2f} x eff_tilt="
                      f"{eff_tilts[m]:.3f} -> {pct:.2f}% of book{note}")
            proposals.append(WeightProposal(
                module=m, weight_pct=pct, reason=reason,
                components={
                    "regime": regime, "confidence": round(conf, 4),
                    "applied_confidence": round(applied_conf, 4),
                    "base_weight": base.get(m, 1.0),
                    "raw_tilt": regime_tilts.get(m, 1.0),
                    "effective_tilt": round(eff_tilts[m], 4),
                    "reserve_pct": round(reserve, 2),
                },
            ))
        else:
            proposals.append(WeightProposal(
                module=m, weight_pct=0.0,
                reason="module disabled" if not enabled_modules.get(m, False)
                else "no allocatable weight",
                components={"regime": regime, "reserve_pct": round(reserve, 2)},
            ))
    return proposals


# ---------------------------------------------------------------- self-test

def _series_trend_expansion(n: int = 160) -> List[float]:
    """Quiet early, then a 24-bar directional burst with much larger two-sided
    bars (deterministic). The burst length matches short_vol_bars so the short
    realized vol re-rates while the long baseline stays anchored on the quiet
    majority of the window."""
    closes = [100.0]
    for i in range(1, n):
        if i < n - 24:
            step = 0.08 if i % 2 == 0 else -0.06   # quiet two-sided drift
        else:
            j = i - (n - 24)
            step = -1.5 if j % 3 == 2 else 2.0     # burst: big bars, net up
        closes.append(closes[-1] + step)
    return closes


def _series_range_compression(n: int = 160) -> List[float]:
    """Big early oscillation decaying to a tight flat range (deterministic)."""
    closes = []
    for i in range(n):
        amp = 6.0 * (1.0 - i / n) ** 2 + 0.15
        closes.append(100.0 + amp * math.sin(i * 0.9))
    return closes


def _selftest() -> None:
    p = RegimeParams()

    # 1. Insufficient data -> neutral, confidence 0.
    r = classify([100.0] * 10, [50.0] * 10, p)
    assert r.regime == REGIME_NEUTRAL and r.confidence == 0.0, r

    # 2. Trending expansion on both assets -> trend_expansion.
    up = _series_trend_expansion()
    r = classify(up, [c * 0.05 for c in up], p)
    assert r.regime == REGIME_TREND_EXPANSION, (r.regime, r.reason)
    assert r.confidence > 0.5, r.confidence

    # 3. Decaying range on both -> range_compression.
    rng = _series_range_compression()
    r2 = classify(rng, [c * 0.05 for c in rng], p)
    assert r2.regime == REGIME_RANGE_COMPRESSION, (r2.regime, r2.reason)

    # 4. Determinism: same inputs -> identical output.
    r3 = classify(rng, [c * 0.05 for c in rng], p)
    assert r3.regime == r2.regime and r3.confidence == r2.confidence

    # 5. Weights: sum + reserve == 100 (within rounding), tilt direction holds.
    enabled = {m: True for m in ALLOC_MODULES}
    w_trend = {x.module: x.weight_pct
               for x in propose_weights(REGIME_TREND_EXPANSION, 0.9, enabled)}
    w_range = {x.module: x.weight_pct
               for x in propose_weights(REGIME_RANGE_COMPRESSION, 0.9, enabled)}
    assert abs(sum(w_trend.values()) - 90.0) < 0.5, sum(w_trend.values())
    assert w_trend["futures"] > w_range["futures"], (w_trend, w_range)
    assert w_trend["arbitrage"] < w_range["arbitrage"], (w_trend, w_range)

    # 6. Low confidence suppresses the tilt -> near-equal weights.
    w_low = {x.module: x.weight_pct
             for x in propose_weights(REGIME_TREND_EXPANSION, 0.1, enabled)}
    vals = list(w_low.values())
    assert max(vals) - min(vals) < 0.05, w_low

    # 7. chop_expansion grows the reserve (weights sum below the normal budget).
    w_chop = propose_weights(REGIME_CHOP_EXPANSION, 1.0, enabled)
    assert sum(x.weight_pct for x in w_chop) < 85.0, sum(x.weight_pct for x in w_chop)
    assert all(x.components["reserve_pct"] >= 19.9 for x in w_chop if x.weight_pct > 0)

    # 8. Disabled modules get an explicit 0% row; enabled sum still = budget.
    enabled2 = dict(enabled)
    enabled2["sniper"] = False
    rows = propose_weights(REGIME_NEUTRAL, 0.5, enabled2)
    sniper_row = next(x for x in rows if x.module == "sniper")
    assert sniper_row.weight_pct == 0.0 and "disabled" in sniper_row.reason
    assert abs(sum(x.weight_pct for x in rows) - 90.0) < 0.5

    print("regime_classifier self-test OK")


if __name__ == "__main__":
    _selftest()
