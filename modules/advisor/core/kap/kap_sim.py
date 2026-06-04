"""
kap_sim.py — KAP-driven BIST sim opener (ADVICE-ONLY, fail-soft).

When a KAP disclosure is classified with a STRONG polarity prior
(STRONG_POSITIVE -> LONG, VERY_NEGATIVE -> SHORT) AND confidence is high
(advisor_kap_alert_min_confidence, default 0.5), open a dry-run sim position for
the disclosure's BIST ticker in the dedicated 'kap' sim channel (migration 077).

Design / guarantees
-------------------
- ENTRY/TARGET/STOP come from the EXISTING BIST analyzer pipeline: we run
  bist_analyzer.analyze(ticker, horizon) to get a real, current price + levels,
  then OVERRIDE the direction to the KAP-implied one and recompute target/stop
  for that direction via the shared levels.horizon_levels() helper (no new model,
  no leakage — same volatility-scaled bands every other sim uses).
- If BIST data is UNAVAILABLE (analyzer returns no usable entry price), we fall
  back to a sensible synthetic level band derived from the analyzer's last-known
  price if present, else we SKIP (no sim opened) — never fabricate a price.
- CHANNEL = 'kap' (independent 15-slot cap). Cap + dedupe enforced before open:
  no duplicate open kap sim for the same ticker.
- FAIL-SOFT: every error is swallowed and logged at DEBUG/WARNING. This function
  NEVER raises into the KAP classifier worker.

This module imports only advisor-internal code; it touches NO trading module.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.advisor.core.models import (
    AdviceResult,
    DataSourceStatus,
    Direction,
    Horizon,
    Market,
)

logger = logging.getLogger("advisor.kap.kap_sim")

# KAP strong polarities that justify a directional sim.
_STRONG_LONG = "STRONG_POSITIVE"
_STRONG_SHORT = "VERY_NEGATIVE"
# Moderate polarities — only acted on when advisor_kap_sim_include_moderate=true.
_MOD_LONG = "POSITIVE"
_MOD_SHORT = "NEGATIVE"

_DEFAULT_MIN_CONF = 0.5
# Horizon used for KAP-driven sims: disclosures are near-term catalysts -> short.
_KAP_HORIZON = Horizon.SHORT


def kap_polarity_to_direction(
    base_polarity: str, include_moderate: bool = False
) -> Optional[Direction]:
    """
    Map a KAP base_polarity string to a sim direction, or None if the polarity
    is not strong enough to act on.

      STRONG_POSITIVE -> LONG
      VERY_NEGATIVE   -> SHORT
      POSITIVE        -> LONG   (only if include_moderate)
      NEGATIVE        -> SHORT  (only if include_moderate)
      (NEUTRAL / UNCLASSIFIED / anything else) -> None

    `include_moderate` widens the firing set to POSITIVE/NEGATIVE — useful when
    strong-polarity disclosures are too rare to ever open a sim. Pure function —
    safe to unit-test without a DB.
    """
    p = str(base_polarity).upper()
    if p == _STRONG_LONG:
        return Direction.LONG
    if p == _STRONG_SHORT:
        return Direction.SHORT
    if include_moderate:
        if p == _MOD_LONG:
            return Direction.LONG
        if p == _MOD_SHORT:
            return Direction.SHORT
    return None


def _min_confidence(config: dict) -> float:
    try:
        return float(config.get("advisor_kap_alert_min_confidence", _DEFAULT_MIN_CONF))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_CONF


async def _has_open_kap_sim(portfolio, ticker: str) -> bool:
    """True if there is already an OPEN kap-channel sim for this ticker (dedupe)."""
    try:
        for s in await portfolio.list_open_sims():
            ch = s.channel or ""
            if ch == "kap" and str(s.symbol).upper() == str(ticker).upper():
                return True
    except Exception as exc:
        logger.debug("[kap.sim] dedupe check failed for %s: %s", ticker, exc)
    return False


def _rebuild_levels_for_direction(
    base_result: AdviceResult, direction: Direction, config: dict
) -> Optional[AdviceResult]:
    """
    Take the BIST analyzer's AdviceResult and re-derive entry/target/stop for the
    KAP-implied `direction` using the shared levels helper. Returns a NEW
    AdviceResult tagged for a kap sim, or None if no usable entry price exists.
    """
    # Determine a usable reference price: midpoint of the analyzer's entry band,
    # else either band edge.
    entry_mid = None
    if base_result.entry_low is not None and base_result.entry_high is not None:
        entry_mid = (base_result.entry_low + base_result.entry_high) / 2.0
    elif base_result.entry_low is not None:
        entry_mid = base_result.entry_low
    elif base_result.entry_high is not None:
        entry_mid = base_result.entry_high

    if entry_mid is None or entry_mid <= 0:
        return None

    try:
        from modules.advisor.core.analyzers.levels import horizon_levels
        entry_low, entry_high, target, stop = horizon_levels(
            {"close": entry_mid}, direction, _KAP_HORIZON,
            config=config, price_decimals=2,
        )
    except Exception as exc:
        logger.debug("[kap.sim] level rebuild failed: %s", exc)
        # Synthetic fallback: fixed band around the reference price.
        entry_low = round(entry_mid * 0.995, 2)
        entry_high = round(entry_mid * 1.005, 2)
        if direction == Direction.LONG:
            target, stop = round(entry_mid * 1.06, 2), round(entry_mid * 0.97, 2)
        else:
            target, stop = round(entry_mid * 0.94, 2), round(entry_mid * 1.03, 2)

    try:
        amount = float(config.get("sim_default_amount_usd", 1000))
    except (TypeError, ValueError):
        amount = 1000.0

    return AdviceResult(
        market=Market.BIST,
        symbol=base_result.symbol,
        horizon=_KAP_HORIZON,
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
        target_price=target,
        stop_price=stop,
        confidence=base_result.confidence,
        rationale="KAP strong-polarity disclosure (advice-only prior).",
        model_id=base_result.model_id,
        data_source_status=base_result.data_source_status,
        data_source_note=base_result.data_source_note,
        sim_enabled=True,                # KAP-driven sims always open if gated OK
        sim_amount_usd=amount,
        extra={"origin": "kap", "kap_driven": True},
    )


def _synthetic_from_last_price(
    bist_analyzer, ticker: str, base_result
) -> Optional[AdviceResult]:
    """
    Build a minimal AdviceResult from the BIST analyzer's last-known price for
    `ticker` (analyzer.last_price[ticker]) so a KAP sim can still open when live
    BIST data is momentarily unavailable. Returns None if no last_price exists
    (we never fabricate a price). Pure-ish: reads analyzer.last_price only.
    """
    last = None
    try:
        lp = getattr(bist_analyzer, "last_price", None)
        if isinstance(lp, dict):
            last = lp.get(ticker) or lp.get(str(ticker).upper())
    except Exception:
        last = None
    if last is None:
        return None
    try:
        last = float(last)
    except (TypeError, ValueError):
        return None
    if last <= 0:
        return None

    band = round(last * 0.002, 2)
    return AdviceResult(
        market=Market.BIST,
        symbol=ticker,
        horizon=_KAP_HORIZON,
        direction=Direction.NEUTRAL,
        entry_low=round(last - band, 2),
        entry_high=round(last + band, 2),
        confidence=getattr(base_result, "confidence", 0.5) if base_result else 0.5,
        data_source_status=DataSourceStatus.DEGRADED,
        data_source_note="KAP sim using analyzer last-known price (BIST live data unavailable).",
    )


async def maybe_open_kap_sim(
    *,
    ticker: str,
    base_polarity: str,
    confidence: float,
    config: dict,
    portfolio,
    bist_analyzer,
    risk=None,
) -> Optional[int]:
    """
    Open a KAP-driven BIST sim in the 'kap' channel if all gates pass. Returns the
    new sim_id, or None if skipped (not strong enough / capped / duplicate /
    no data / error). NEVER raises.

    Gates (in order):
      1. base_polarity is STRONG_POSITIVE or VERY_NEGATIVE.
      2. confidence >= advisor_kap_alert_min_confidence (default 0.5).
      3. ticker present + a BIST analyzer + portfolio engine available.
      4. no existing OPEN kap sim for this ticker (dedupe).
      5. kap channel below its per-channel cap (advisor_sim_cap_per_channel).
      6. BIST analyzer yields a usable entry price.
    """
    try:
        if not ticker:
            return None

        include_moderate = str(
            config.get("advisor_kap_sim_include_moderate", "false")
        ).lower() == "true"
        direction = kap_polarity_to_direction(
            base_polarity, include_moderate=include_moderate
        )
        if direction is None:
            return None

        if confidence < _min_confidence(config):
            return None

        if portfolio is None or bist_analyzer is None:
            logger.debug("[kap.sim] portfolio/analyzer unavailable; skip %s", ticker)
            return None

        # Dedupe: one open kap sim per ticker at a time.
        if await _has_open_kap_sim(portfolio, ticker):
            logger.debug("[kap.sim] open kap sim already exists for %s; skip.", ticker)
            return None

        # Per-channel cap (migration 077): kap channel has its own 15 slots.
        try:
            cap = int(float(config.get("advisor_sim_cap_per_channel",
                                       config.get("max_sim_positions", 15))))
        except (TypeError, ValueError):
            cap = 15
        try:
            counts = await portfolio.count_open_sims_by_channel()
            kap_open = int(counts.get("kap", 0))
        except Exception:
            kap_open = 0
        if kap_open >= cap:
            logger.info(
                "[kap.sim] kap channel at cap (%d/%d); skip %s.", kap_open, cap, ticker
            )
            return None

        # Run the EXISTING BIST analyzer to get a real, current price + levels.
        try:
            base_result = await bist_analyzer.analyze(ticker, _KAP_HORIZON)
        except Exception as exc:
            logger.debug("[kap.sim] BIST analyze failed for %s: %s", ticker, exc)
            return None

        if base_result is None or base_result.data_source_status in (
            DataSourceStatus.NOT_CONFIGURED,
            DataSourceStatus.ERROR,
        ):
            # BIST live data is unavailable. Rather than always skip (which left
            # the kap channel permanently empty on hosts without BIST data), try
            # the analyzer's last-known price for this ticker as a synthetic
            # reference. We NEVER fabricate a price — if no last_price exists, we
            # still skip.
            synth = _synthetic_from_last_price(bist_analyzer, ticker, base_result)
            if synth is None:
                logger.debug(
                    "[kap.sim] BIST data unavailable for %s (status=%s) and no "
                    "last_price; skip.",
                    ticker,
                    getattr(base_result, "data_source_status", "n/a"),
                )
                return None
            base_result = synth

        sim_result = _rebuild_levels_for_direction(base_result, direction, config)
        if sim_result is None:
            logger.debug("[kap.sim] no usable entry price for %s; skip.", ticker)
            return None

        sim_id = await portfolio.open_sim_position(sim_result)
        if sim_id is not None:
            logger.info(
                "[kap.sim] Opened KAP-driven %s sim for %s (channel=kap, sim#%s).",
                direction.value, ticker, sim_id,
            )
        return sim_id
    except Exception as exc:
        # Absolute fail-soft backstop: never break the KAP classifier worker.
        logger.warning("[kap.sim] maybe_open_kap_sim error for %s (fail-soft): %s",
                       ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Self-test (no DB / no network): polarity->direction mapping + level rebuild.
#   python -m modules.advisor.core.kap.kap_sim
# ---------------------------------------------------------------------------
def _selftest() -> None:
    assert kap_polarity_to_direction("STRONG_POSITIVE") == Direction.LONG
    assert kap_polarity_to_direction("VERY_NEGATIVE") == Direction.SHORT
    assert kap_polarity_to_direction("POSITIVE") is None
    assert kap_polarity_to_direction("NEGATIVE") is None
    assert kap_polarity_to_direction("NEUTRAL") is None
    # include_moderate widens the firing set to POSITIVE/NEGATIVE.
    assert kap_polarity_to_direction("POSITIVE", include_moderate=True) == Direction.LONG
    assert kap_polarity_to_direction("NEGATIVE", include_moderate=True) == Direction.SHORT
    assert kap_polarity_to_direction("NEUTRAL", include_moderate=True) is None

    # Synthetic fallback: uses analyzer.last_price when no live BIST result.
    class _FakeAnalyzer:
        last_price = {"THYAO": 50.0}
    synth = _synthetic_from_last_price(_FakeAnalyzer(), "THYAO", None)
    assert synth is not None and synth.entry_low and synth.entry_high
    assert _synthetic_from_last_price(_FakeAnalyzer(), "MISSING", None) is None
    assert _synthetic_from_last_price(object(), "THYAO", None) is None

    base = AdviceResult(
        market=Market.BIST, symbol="THYAO", horizon=Horizon.SHORT,
        direction=Direction.NEUTRAL, entry_low=100.0, entry_high=102.0,
        target_price=None, stop_price=None, confidence=0.8,
        data_source_status=DataSourceStatus.DEGRADED,
    )
    long_sim = _rebuild_levels_for_direction(base, Direction.LONG, config={})
    assert long_sim is not None and long_sim.direction == Direction.LONG
    assert long_sim.extra["kap_driven"] is True and long_sim.extra["origin"] == "kap"
    # LONG: target above entry mid (~101), stop below.
    assert long_sim.target_price > 101.0 > long_sim.stop_price, (
        long_sim.target_price, long_sim.stop_price)

    short_sim = _rebuild_levels_for_direction(base, Direction.SHORT, config={})
    assert short_sim.target_price < 101.0 < short_sim.stop_price, (
        short_sim.target_price, short_sim.stop_price)

    # No entry price -> None.
    empty = AdviceResult(
        market=Market.BIST, symbol="X", horizon=Horizon.SHORT,
        direction=Direction.NEUTRAL, entry_low=None, entry_high=None,
        data_source_status=DataSourceStatus.DEGRADED,
    )
    assert _rebuild_levels_for_direction(empty, Direction.LONG, config={}) is None

    print("kap_sim self-test OK: polarity mapping + directional level rebuild")


if __name__ == "__main__":
    _selftest()
