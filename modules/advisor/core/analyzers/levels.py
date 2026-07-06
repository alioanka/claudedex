"""
levels — shared, horizon-aware trade-level + confidence model for advisor analyzers.

TRANSPARENCY NOTICE
-------------------
These are HEURISTIC levels, not guarantees. The advisor is ADVICE-ONLY; no orders
are placed. Targets/stops are derived from a transparent volatility-scaled band model
and the confidence number is a deterministic function of real signal inputs. None of
this predicts the future — the operator owns every decision.

WHY THIS MODULE EXISTS (root cause of the defects it fixes)
-----------------------------------------------------------
Previously every analyzer computed entry/target/stop with FIXED percentage bands
(e.g. entry +/-0.5%, target +/-5%, stop +/-3%) that ignored the horizon entirely.
Result: BTC short/mid/long all showed the SAME entry/target/stop. And per-analyzer
`_compute_confidence` returned only abs(vote)/3 (+0.1 vol boost) -> a tiny discrete
ladder (0, 0.33, 0.43, 0.67, ...) that looked frozen at "43%" across symbols/horizons.

This module fixes both:

1. HORIZON-AWARE LEVELS
   The distance of target/stop from entry scales with BOTH:
     - a per-horizon multiplier (short < mid < long), and
     - the asset's own recent volatility (ATR% if available, else Bollinger-band
       half-width as a daily-vol proxy).
   So a longer horizon -> wider target and wider stop, and a more volatile asset ->
   wider bands than a quiet one. Levels now differ per horizon for the same symbol.

   target_distance_frac = vol_unit * horizon_target_mult[horizon] * target_rr
   stop_distance_frac   = vol_unit * horizon_stop_mult[horizon]
   entry_band_frac      = vol_unit * horizon_entry_mult[horizon]

   where vol_unit = clamp(atr_pct_or_bb_proxy, vol_floor, vol_ceiling).

2. REAL, VARYING CONFIDENCE
   confidence in [conf_floor, conf_ceiling] built from:
     - signed vote agreement (how aligned the indicators are, and how strongly),
     - signal magnitude (RSI/BB extremity),
     - volume confirmation,
     - a horizon factor (longer horizon = slightly more uncertainty),
     - a dual-advice DISAGREE penalty (anthropic vs openai disagreement LOWERS it).
   Because each term is continuous, the output varies per symbol AND per horizon.

All multipliers/floors are operator-tunable via advisor_config (migration 067).
"""

from __future__ import annotations

import math

from typing import Optional, Tuple

from modules.advisor.core.models import Direction, Horizon

# ---------------------------------------------------------------------------
# Defaults (overridable via advisor_config — see migration 067)
# ---------------------------------------------------------------------------
# Per-horizon multipliers applied to the volatility unit (vol_unit, a per-bar
# fractional move). These widen the target/stop/entry bands as the horizon grows.
# Calibration intuition (daily-bar vol_unit ~= 1 ATR-day):
#   short ~ a few days of vol, mid ~ a few weeks, long ~ a quarter+.
_DEFAULTS = {
    # entry band half-width as a fraction of vol_unit
    "levels_entry_mult_short": 0.25,
    "levels_entry_mult_mid":   0.50,
    "levels_entry_mult_long":  1.00,
    # target distance multiplier (× vol_unit, before reward:risk)
    "levels_target_mult_short": 3.0,
    "levels_target_mult_mid":   8.0,
    "levels_target_mult_long":  20.0,
    # stop distance multiplier (× vol_unit)
    "levels_stop_mult_short": 1.5,
    "levels_stop_mult_mid":   4.0,
    "levels_stop_mult_long":  10.0,
    # global reward:risk shaping applied to target only (keeps target >= stop dist)
    "levels_target_rr": 1.0,
    # vol_unit clamps (fractional per-bar vol). Floors avoid degenerate 0-width
    # bands on ultra-quiet series; ceilings avoid absurd bands on a vol spike.
    "levels_vol_floor": 0.005,   # 0.5% min per-bar vol
    "levels_vol_ceiling": 0.15,  # 15% max per-bar vol
    # confidence shaping
    "levels_conf_floor": 0.05,
    "levels_conf_ceiling": 0.95,
    "levels_conf_disagree_penalty": 0.20,  # subtracted when providers disagree
}


def _cfg_float(config: Optional[dict], key: str) -> float:
    default = _DEFAULTS[key]
    if not config:
        return default
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Volatility unit
# ---------------------------------------------------------------------------

def vol_unit_from_signals(signals: dict, config: Optional[dict] = None) -> float:
    """
    Derive a per-bar fractional volatility unit from a signals dict.

    Preference order:
      1. signals['atr_pct']  (ATR-14 as a fraction of close) if the analyzer
         supplied it — most accurate.
      2. Bollinger half-width: (bb_upper - bb_lower) / 2 / close. The advisor's
         BB is a 20-day, 2-sigma band, so half-width ~= 2 * daily-stdev; we divide
         by 2 again to get ~1 sigma (a clean daily-vol proxy).
      3. Fallback: vol_floor.

    Clamped to [vol_floor, vol_ceiling].
    """
    floor = _cfg_float(config, "levels_vol_floor")
    ceiling = _cfg_float(config, "levels_vol_ceiling")

    close = float(signals.get("close") or 0.0)
    atr_pct = signals.get("atr_pct")
    if atr_pct is not None:
        try:
            vu = float(atr_pct)
            if vu > 0:
                return max(floor, min(vu, ceiling))
        except (TypeError, ValueError):
            pass

    bb_upper = signals.get("bb_upper")
    bb_lower = signals.get("bb_lower")
    if close > 0 and bb_upper is not None and bb_lower is not None:
        try:
            half_width = (float(bb_upper) - float(bb_lower)) / 2.0
            # half_width ~= 2 sigma -> divide by 2 for ~1 sigma per-bar vol
            vu = (half_width / 2.0) / close
            if vu > 0:
                return max(floor, min(vu, ceiling))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return floor


# ---------------------------------------------------------------------------
# Horizon-aware levels
# ---------------------------------------------------------------------------

def _horizon_key(horizon: Horizon) -> str:
    if horizon == Horizon.SHORT:
        return "short"
    if horizon == Horizon.LONG:
        return "long"
    return "mid"


def horizon_levels(
    signals: dict,
    direction: Direction,
    horizon: Horizon,
    config: Optional[dict] = None,
    price_decimals: int = 8,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute horizon-aware (entry_low, entry_high, target_price, stop_price).

    Bands scale with the asset's volatility unit AND the horizon multiplier, so
    short < mid < long target/stop distances for the same symbol/inputs.

    Returns (entry_low, entry_high, target_price, stop_price). target/stop are
    None for a NEUTRAL direction (no actionable trade).
    """
    # Non-finite guard (Wave-F5 fix 3): a NaN close passes `close <= 0`
    # (all NaN comparisons are False) and poisons every derived level, which
    # then persists as 'NaN'::float8 and breaks the dashboard JSON. Reject it.
    try:
        close = float(signals.get("close") or 0.0)
    except (TypeError, ValueError):
        return None, None, None, None
    if not math.isfinite(close) or close <= 0:
        return None, None, None, None

    hk = _horizon_key(horizon)
    vu = vol_unit_from_signals(signals, config)
    if not math.isfinite(vu) or vu <= 0:
        # Defensive: vol_unit_from_signals clamps to [floor, ceiling], but a
        # non-finite ATR/BB input must never reach the level math.
        return None, None, None, None

    entry_mult = _cfg_float(config, f"levels_entry_mult_{hk}")
    target_mult = _cfg_float(config, f"levels_target_mult_{hk}")
    stop_mult = _cfg_float(config, f"levels_stop_mult_{hk}")
    target_rr = _cfg_float(config, "levels_target_rr")

    entry_frac = vu * entry_mult
    target_frac = vu * target_mult * target_rr
    stop_frac = vu * stop_mult

    entry_low = round(close * (1.0 - entry_frac), price_decimals)
    entry_high = round(close * (1.0 + entry_frac), price_decimals)

    if direction == Direction.LONG:
        target = round(close * (1.0 + target_frac), price_decimals)
        stop = round(close * (1.0 - stop_frac), price_decimals)
    elif direction == Direction.SHORT:
        target = round(close * (1.0 - target_frac), price_decimals)
        stop = round(close * (1.0 + stop_frac), price_decimals)
    else:
        target = None
        stop = None

    return entry_low, entry_high, target, stop


# ---------------------------------------------------------------------------
# Real, varying confidence
# ---------------------------------------------------------------------------

def signal_confidence(
    signals: dict,
    horizon: Horizon,
    config: Optional[dict] = None,
    providers_disagree: bool = False,
    n_votes: int = 3,
) -> float:
    """
    Deterministic, varying confidence in [conf_floor, conf_ceiling].

    Inputs (all real signal-derived quantities):
      - vote agreement: signed sum of available signal votes, normalised by the
        number of votes -> [0,1] strength of alignment.
      - magnitude: RSI/BB extremity boost (how far from neutral the readings are).
      - volume: confirmation boost when relative volume is elevated.
      - horizon factor: longer horizons carry a small uncertainty discount.
      - disagree penalty: dual-advice (anthropic vs openai) DISAGREE subtracts.

    Formula:
      base       = 0.55 * agreement + 0.30 * magnitude + 0.15 * volume_conf
      adjusted   = base * horizon_factor - disagree_penalty(if disagree)
      confidence = clamp(adjusted, conf_floor, conf_ceiling)

    Because agreement/magnitude/volume are continuous, the output varies per
    symbol and per horizon — it is NOT a frozen constant.
    """
    floor = _cfg_float(config, "levels_conf_floor")
    ceiling = _cfg_float(config, "levels_conf_ceiling")

    votes = (
        float(signals.get("sma_signal", 0) or 0)
        + float(signals.get("rsi_signal", 0) or 0)
        + float(signals.get("bb_signal", 0) or 0)
    )
    n = max(1, n_votes)
    agreement = abs(votes) / float(n)  # [0,1]

    # Magnitude: how extreme RSI is (distance from 50, scaled) + BB extremity.
    rsi = float(signals.get("rsi", 50.0) or 50.0)
    rsi_extremity = min(abs(rsi - 50.0) / 50.0, 1.0)  # 0 at neutral, 1 at 0/100
    bb_extremity = 1.0 if abs(float(signals.get("bb_signal", 0) or 0)) >= 1 else 0.0
    magnitude = 0.7 * rsi_extremity + 0.3 * bb_extremity  # [0,1]

    # Volume confirmation: rel-volume > 1 boosts, capped at 2x.
    vol_ratio = float(signals.get("vol_ratio", 1.0) or 1.0)
    volume_conf = min(max(vol_ratio - 1.0, 0.0), 1.0)  # [0,1]

    base = 0.55 * agreement + 0.30 * magnitude + 0.15 * volume_conf

    # Horizon factor: longer horizon -> slightly lower confidence (more unknowns).
    horizon_factor = {
        Horizon.SHORT: 1.00,
        Horizon.MID: 0.92,
        Horizon.LONG: 0.85,
    }.get(horizon, 0.92)

    confidence = base * horizon_factor

    if providers_disagree:
        confidence -= _cfg_float(config, "levels_conf_disagree_penalty")

    return round(max(floor, min(confidence, ceiling)), 4)


# ---------------------------------------------------------------------------
# Self-check — run `python -m modules.advisor.core.analyzers.levels`
# Verifies short/mid/long produce DIFFERENT target/stop for the same inputs and
# that confidence varies with signal strength and dual-advice disagreement.
# ---------------------------------------------------------------------------

def _self_check() -> None:
    sig = {
        "close": 67000.0, "bb_upper": 70000.0, "bb_lower": 64000.0,
        "rsi": 62.0, "sma_signal": 1, "rsi_signal": 0, "bb_signal": 1,
        "vol_ratio": 1.3, "atr_pct": 0.03,
    }
    tgts, stops = [], []
    for h in (Horizon.SHORT, Horizon.MID, Horizon.LONG):
        _, _, t, s = horizon_levels(sig, Direction.LONG, h)
        tgts.append(t)
        stops.append(s)
    assert len(set(tgts)) == 3, f"targets not horizon-differentiated: {tgts}"
    assert len(set(stops)) == 3, f"stops not horizon-differentiated: {stops}"
    # target distance must widen short < mid < long (LONG direction)
    assert tgts[0] < tgts[1] < tgts[2], f"target ordering wrong: {tgts}"
    assert stops[0] > stops[1] > stops[2], f"stop ordering wrong: {stops}"

    weak = dict(sig, sma_signal=1, rsi_signal=-1, bb_signal=0, rsi=50, vol_ratio=1.0)
    strong = dict(sig, sma_signal=1, rsi_signal=1, bb_signal=1, rsi=78, vol_ratio=1.8)
    c_weak = signal_confidence(weak, Horizon.SHORT)
    c_strong = signal_confidence(strong, Horizon.SHORT)
    c_disagree = signal_confidence(strong, Horizon.SHORT, providers_disagree=True)
    assert c_strong > c_weak, f"strong({c_strong}) !> weak({c_weak})"
    assert c_disagree < c_strong, f"disagree({c_disagree}) !< strong({c_strong})"
    # horizon affects confidence (longer = lower, all else equal)
    assert (
        signal_confidence(strong, Horizon.SHORT)
        > signal_confidence(strong, Horizon.LONG)
    ), "horizon factor not applied"
    print("levels self-check PASSED")
    print(f"  targets short/mid/long = {tgts}")
    print(f"  stops   short/mid/long = {stops}")
    print(f"  conf weak/strong/disagree = {c_weak}/{c_strong}/{c_disagree}")


if __name__ == "__main__":
    _self_check()
