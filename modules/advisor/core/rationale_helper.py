"""
rationale_helper — shared LLM rationale generator for all advisor market analyzers.

All five analyzers (crypto, us_equities, bist, fx, midas_funds) call
`build_rationale()` with the same technical-signal dict and get back a
consistent natural-language advice string.

Key resolution order (mirrors AI module pattern from modules/ai_analysis/CLAUDE.md):
  1. config dict key "advisor_anthropic_api_key"
  2. os.environ["ADVISOR_ANTHROPIC_API_KEY"]
  If neither exists → rule-based fallback (no exception raised).

Model ID resolution:
  config["advisor_anthropic_model"] → default "claude-opus-4-5"
  Loud WARNING (not silent) on 404 / not_found — operator must update DB.

ADVICE-ONLY. This helper generates plain-text rationale.
It never places orders, reads portfolio state, or calls trading engines.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from modules.advisor.core.models import Direction, Horizon, Market

logger = logging.getLogger("advisor.rationale_helper")

# Horizon → natural-language label (keep in sync with each analyzer)
_HORIZON_LABEL: dict = {
    Horizon.SHORT: "1 day – 1 week",
    Horizon.MID: "1 week – 3 months",
    Horizon.LONG: "3 months – 2 years",
}

# Market → human-readable name used in LLM prompt
_MARKET_LABEL: dict = {
    Market.CRYPTO: "cryptocurrency",
    Market.US_EQUITIES: "US equity",
    Market.BIST: "Borsa Istanbul equity",
    Market.FX: "foreign exchange / commodity",
    Market.MIDAS_FUNDS: "Turkish mutual fund",
}

# Max characters we allow from any user-supplied string going into the LLM prompt
_MAX_SYMBOL_LEN = 64


def _sanitize(text: str, max_len: int = _MAX_SYMBOL_LEN) -> str:
    """Strip control/non-printable characters; truncate to max_len."""
    cleaned = "".join(c for c in text if c.isprintable())
    return cleaned[:max_len]


def _resolve_api_key(config: dict) -> Optional[str]:
    """
    Resolve the Anthropic API key using the two-step pattern from the AI module:
      1. config dict (DB-sourced by AdvisorConfigManager)
      2. environment variable
    Returns None (without raising) if neither is set.
    """
    key = config.get("advisor_anthropic_api_key") or ""
    if key:
        return key
    key = os.getenv("ADVISOR_ANTHROPIC_API_KEY", "")
    return key or None


def build_rule_based_rationale(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
) -> str:
    """
    Deterministic rule-based rationale.  Used as fallback when the Anthropic
    API is unavailable, key is missing, or the call fails.

    Works for full OHLCV signals (SMA/RSI/BB/vol_ratio) AND for
    price-only fund signals (sma_signal / rsi only, no BB / vol_ratio).
    """
    d = direction.value.upper()
    h = _HORIZON_LABEL.get(horizon, horizon.value)
    sym = _sanitize(symbol)
    mkt = _MARKET_LABEL.get(market, market.value)

    rsi = signals.get("rsi", 50)
    sma_cross = "above" if signals.get("sma_signal", 0) > 0 else "below"

    parts = [
        f"{sym} ({mkt}) shows a {d} bias for the {h} horizon.",
        f"SMA20 is {sma_cross} SMA50.",
        f"RSI-14 = {rsi:.1f}.",
    ]

    if "bb_upper" in signals and "bb_lower" in signals:
        parts.append(
            f"Bollinger Band range: {signals['bb_lower']:.6g} – {signals['bb_upper']:.6g}."
        )
    if "vol_ratio" in signals:
        parts.append(
            f"Volume is {signals['vol_ratio']:.2f}x the 20-day average."
        )

    parts.append(
        "This is an automated technical signal; conduct your own due diligence before acting."
    )
    return " ".join(parts)


def _build_signal_summary(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
) -> str:
    """Build the sanitized signal block sent to the LLM."""
    sym = _sanitize(symbol)
    h = _HORIZON_LABEL.get(horizon, horizon.value)
    mkt = _MARKET_LABEL.get(market, market.value)

    lines = [
        f"Asset: {sym} ({mkt})",
        f"Horizon: {h}",
        f"Close: {signals.get('close', 0):.6g}",
        f"SMA20: {signals.get('sma20', 0):.6g} | SMA50: {signals.get('sma50', 0):.6g}",
        f"RSI-14: {signals.get('rsi', 0):.1f}",
    ]
    if "bb_upper" in signals:
        lines.append(
            f"BB upper: {signals['bb_upper']:.6g} | lower: {signals.get('bb_lower', 0):.6g}"
        )
    if "vol_ratio" in signals:
        lines.append(f"Volume ratio vs 20d avg: {signals['vol_ratio']:.2f}x")
    lines.append(f"Computed direction: {direction.value.upper()}")
    return "\n".join(lines)


async def build_rationale(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
    config: dict,
    caller_logger: Optional[logging.Logger] = None,
) -> str:
    """
    Primary entry point for all analyzer rationale generation.

    Tries Anthropic LLM first; falls back to rule-based text without raising.

    Parameters
    ----------
    symbol          : Asset ticker / fund code (sanitized before LLM).
    market          : Market enum value.
    horizon         : Horizon enum value.
    signals         : Dict produced by _fetch_and_compute (keys: close,
                      sma20, sma50, rsi, bb_upper, bb_lower, bb_mid,
                      sma_signal, rsi_signal, bb_signal, vol_ratio).
                      Partial dicts (fund NAV-only) are tolerated.
    direction       : Direction the technicals voted.
    config          : Advisor config dict (from AdvisorConfigManager).
    caller_logger   : Optional logger from the calling analyzer class.

    Returns
    -------
    str — always non-empty, always advice-only text.
    """
    log = caller_logger or logger
    fallback = build_rule_based_rationale(symbol, market, horizon, signals, direction)

    api_key = _resolve_api_key(config)
    if not api_key:
        return fallback

    model_id = config.get("advisor_anthropic_model", "claude-opus-4-5")
    signal_summary = _build_signal_summary(symbol, market, horizon, signals, direction)
    mkt_label = _MARKET_LABEL.get(market, market.value)

    prompt = (
        f"You are a professional financial analyst advising on {mkt_label} instruments. "
        "Given the following technical signals, write a concise 2-4 sentence rationale "
        "explaining the directional advice for the stated horizon. "
        "Focus on the key technical factors. "
        "Do NOT recommend specific trade sizes or execution venues. "
        "Do NOT make promises about future returns. "
        "End with a brief reminder that this is an automated signal and the reader "
        "should do their own due diligence.\n\n"
        f"{signal_summary}"
    )

    try:
        import anthropic  # imported lazily — not required at module load

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model_id,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else fallback
        return text[:2000]  # cap to DB column limit

    except Exception as exc:
        exc_str = str(exc)
        log.warning(
            f"[rationale_helper] Anthropic call failed for {symbol} "
            f"(model={model_id}): {exc_str[:200]}. Using rule-based fallback."
        )
        if "404" in exc_str or "not_found" in exc_str.lower():
            log.warning(
                f"[rationale_helper] MODEL NOT FOUND: {model_id}. "
                "Update advisor_anthropic_model in advisor_config DB table "
                "(Settings > Advisor Settings)."
            )
        return fallback
