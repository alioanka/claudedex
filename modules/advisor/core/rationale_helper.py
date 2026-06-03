"""
rationale_helper — shared LLM rationale generator for all advisor market analyzers.

All five analyzers (crypto, us_equities, bist, fx, midas_funds) call
`build_rationale()` with the same technical-signal dict and get back a
consistent natural-language advice string.

Key resolution order (Anthropic — mirrors AI module pattern):
  1. config dict key "advisor_anthropic_api_key"
  2. os.environ["ADVISOR_ANTHROPIC_API_KEY"]
  If neither exists -> rule-based fallback (no exception raised).

Key resolution order (OpenAI dual-advice):
  1. config dict key "advisor_openai_api_key"
  2. os.environ["ADVISOR_OPENAI_API_KEY"]
  If absent or disabled -> OpenAI call is silently skipped; Anthropic-only result used.

Model ID resolution:
  Anthropic: config["advisor_anthropic_model"] -> default "claude-opus-4-8"
  OpenAI:    config["advisor_openai_model"]    -> default "gpt-4o"
  Loud WARNING (not silent) on 404 / not_found for either provider.

Dual-advice mode (config["advisor_dual_advice_mode"]):
  "off"       : Anthropic-only (default; original behaviour preserved).
  "both"      : Call both providers; store results in AdviceResult.extra
                  extra["rationale_anthropic"] and extra["rationale_openai"].
                  The primary rationale field gets the Anthropic text (or
                  rule-based if Anthropic fails).
  "consensus" : Call both; if they AGREE on direction, boost confidence;
                  if DISAGREE, set extra["providers_disagree"]=True and lower
                  confidence. Agreement is direction-keyword comparison (LONG /
                  SHORT / NEUTRAL extracted from rationale text).

build_rationale() always returns a plain-text str (primary rationale).
Dual-advice artefacts are written into the `extra` dict passed in — callers
should pass extra={} and inspect it afterwards.

ADVICE-ONLY. No orders are placed from this module.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

from modules.advisor.core.models import Direction, Horizon, Market

logger = logging.getLogger("advisor.rationale_helper")

# ---------------------------------------------------------------------------
# LLM rationale cache (issue #12 — don't spam the LLM API every cycle)
# ---------------------------------------------------------------------------
# The numeric levels and confidence are computed locally (core/analyzers/levels.py);
# the LLM only writes the prose rationale. There is no reason to re-call the LLM
# every advice cycle when the underlying signal/direction has NOT materially
# changed. We cache the generated rationale keyed on
#   market | symbol | horizon | direction | rounded(close, sma20, sma50, rsi, bb)
# and reuse it until either (a) that key changes (signal moved enough to round
# differently / direction flipped) or (b) the cached entry exceeds a max age.
#
# Cache is in-process (the advisor runs as a single long-lived subprocess).
# Toggle/age are DB-configurable (advisor_config); see _cache_settings().
_RATIONALE_CACHE: dict = {}            # key -> (rationale_text, created_ts)
_RATIONALE_CACHE_MAX_ENTRIES = 2000    # hard cap to bound memory

_CFG_CACHE_ENABLED = "advisor_llm_rationale_cache_enabled"   # bool, default true
_CFG_CACHE_MAX_AGE = "advisor_llm_rationale_max_age_minutes"  # int,  default 360 (6h)
_DEFAULT_CACHE_MAX_AGE_MIN = 360


def _cache_settings(config: dict) -> Tuple[bool, float]:
    """Resolve (enabled, max_age_seconds) from advisor_config (fail-soft)."""
    enabled = str(config.get(_CFG_CACHE_ENABLED, "true")).lower() != "false"
    try:
        age_min = float(config.get(_CFG_CACHE_MAX_AGE, _DEFAULT_CACHE_MAX_AGE_MIN))
    except (ValueError, TypeError):
        age_min = _DEFAULT_CACHE_MAX_AGE_MIN
    if age_min < 0:
        age_min = _DEFAULT_CACHE_MAX_AGE_MIN
    return enabled, age_min * 60.0


def _signal_fingerprint(signals: dict) -> str:
    """
    A coarse, rounded fingerprint of the signal so that small price wiggles do
    NOT invalidate the cached rationale. Prices rounded to 4 significant figures,
    RSI to the nearest integer, volume ratio to 1 decimal.
    """
    def sig4(x) -> str:
        try:
            v = float(x)
        except (ValueError, TypeError):
            return "_"
        if v == 0:
            return "0"
        return f"{v:.4g}"

    parts = [
        sig4(signals.get("close")),
        sig4(signals.get("sma20")),
        sig4(signals.get("sma50")),
        f"{float(signals.get('rsi', 0)):.0f}" if signals.get("rsi") is not None else "_",
        sig4(signals.get("bb_upper")),
        sig4(signals.get("bb_lower")),
        f"{float(signals.get('vol_ratio', 0)):.1f}" if signals.get("vol_ratio") is not None else "_",
    ]
    return ".".join(parts)


def _cache_key(symbol: str, market: Market, horizon: Horizon,
               direction: Direction, signals: dict, dual_mode: str) -> str:
    return "|".join([
        market.value, _sanitize(symbol), horizon.value, direction.value,
        dual_mode, _signal_fingerprint(signals),
    ])


def _cache_get(key: str, max_age_s: float) -> Optional[str]:
    entry = _RATIONALE_CACHE.get(key)
    if not entry:
        return None
    text, created = entry
    if (time.monotonic() - created) > max_age_s:
        _RATIONALE_CACHE.pop(key, None)
        return None
    return text


def _cache_put(key: str, text: str) -> None:
    if len(_RATIONALE_CACHE) >= _RATIONALE_CACHE_MAX_ENTRIES:
        # Evict ~10% oldest entries (cheap; cache churn is low).
        for k in sorted(_RATIONALE_CACHE, key=lambda k: _RATIONALE_CACHE[k][1])[:200]:
            _RATIONALE_CACHE.pop(k, None)
    _RATIONALE_CACHE[key] = (text, time.monotonic())

# Horizon -> natural-language label (keep in sync with each analyzer)
_HORIZON_LABEL: dict = {
    Horizon.SHORT: "1 day - 1 week",
    Horizon.MID: "1 week - 3 months",
    Horizon.LONG: "3 months - 2 years",
}

# Market -> human-readable name used in LLM prompt
_MARKET_LABEL: dict = {
    Market.CRYPTO: "cryptocurrency",
    Market.US_EQUITIES: "US equity",
    Market.BIST: "Borsa Istanbul equity",
    Market.FX: "foreign exchange / commodity",
    Market.MIDAS_FUNDS: "Turkish mutual fund",
}

# Max characters from any user-supplied string before LLM call
_MAX_SYMBOL_LEN = 64

# Dual-advice mode values
_DUAL_MODE_OFF       = "off"
_DUAL_MODE_BOTH      = "both"
_DUAL_MODE_CONSENSUS = "consensus"


# ---------------------------------------------------------------------------
# Key resolution helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str, max_len: int = _MAX_SYMBOL_LEN) -> str:
    """Strip control/non-printable characters; truncate to max_len."""
    cleaned = "".join(c for c in text if c.isprintable())
    return cleaned[:max_len]


def _resolve_api_key(config: dict) -> Optional[str]:
    """
    Resolve the Anthropic API key (two-step: config dict, then env var).
    Returns None without raising if neither is set.
    """
    key = config.get("advisor_anthropic_api_key") or ""
    if key:
        return key
    return os.getenv("ADVISOR_ANTHROPIC_API_KEY") or None


def _resolve_openai_key(config: dict) -> Optional[str]:
    """
    Resolve the OpenAI API key for advisor dual-advice.
    Resolution order:
      1. config["advisor_openai_api_key"]  (DB-stored, e.g. via Secure Credentials panel)
      2. os.environ["ADVISOR_OPENAI_API_KEY"]
    Returns None if absent — dual-advice is silently skipped.
    """
    key = config.get("advisor_openai_api_key") or ""
    if key:
        return key
    return os.getenv("ADVISOR_OPENAI_API_KEY") or None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def build_rule_based_rationale(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
) -> str:
    """
    Deterministic rule-based rationale. Used as fallback when LLM APIs are
    unavailable, keys are missing, or calls fail.

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
            f"Bollinger Band range: {signals['bb_lower']:.6g} - {signals['bb_upper']:.6g}."
        )
    if "vol_ratio" in signals:
        parts.append(
            f"Volume is {signals['vol_ratio']:.2f}x the 20-day average."
        )

    parts.append(
        "This is an automated technical signal; conduct your own due diligence before acting."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Signal summary builder (shared prompt block)
# ---------------------------------------------------------------------------

def _build_signal_summary(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
) -> str:
    """Build the sanitized signal block sent to LLMs."""
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


def _build_prompt(signal_summary: str, mkt_label: str) -> str:
    """Shared prompt template used by both Anthropic and OpenAI calls."""
    return (
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


# ---------------------------------------------------------------------------
# Provider call helpers
# ---------------------------------------------------------------------------

async def _call_anthropic(
    prompt: str,
    model_id: str,
    api_key: str,
    fallback: str,
    log: logging.Logger,
    symbol: str,
) -> str:
    """
    Call Anthropic API. Returns rationale text or fallback on any failure.
    Loud WARNING on 404 / not_found (operator must update DB).
    """
    try:
        import anthropic  # lazy import

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model_id,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else fallback
        return text[:2000]

    except Exception as exc:
        exc_str = str(exc)
        log.warning(
            "[rationale_helper] Anthropic call failed for %s (model=%s): %s. "
            "Using rule-based fallback.",
            symbol, model_id, exc_str[:200],
        )
        if "404" in exc_str or "not_found" in exc_str.lower():
            log.warning(
                "[rationale_helper] ANTHROPIC MODEL NOT FOUND: %s. "
                "Update advisor_anthropic_model in advisor_config DB table "
                "(Settings > Advisor Settings).",
                model_id,
            )
        return fallback


async def _call_openai(
    prompt: str,
    model_id: str,
    api_key: str,
    fallback: str,
    log: logging.Logger,
    symbol: str,
) -> str:
    """
    Call OpenAI chat completions API (aiohttp). Returns rationale text or fallback.
    Loud WARNING on 404 / model_not_found so operator can update advisor_openai_model.
    """
    try:
        import aiohttp  # lazy import — optional dependency
        import json as _json

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.3,
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    log.warning(
                        "[rationale_helper] OpenAI call failed for %s "
                        "(model=%s, status=%d): %s. Skipping OpenAI advice.",
                        symbol, model_id, resp.status, err_text[:200],
                    )
                    if resp.status == 404 or (
                        "model_not_found" in err_text or "invalid_request_error" in err_text
                    ):
                        log.warning(
                            "[rationale_helper] OPENAI MODEL NOT FOUND: %s. "
                            "Update advisor_openai_model in advisor_config DB table.",
                            model_id,
                        )
                    return fallback

                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return fallback
                text = choices[0].get("message", {}).get("content", fallback)
                return str(text)[:2000]

    except ImportError:
        log.debug("[rationale_helper] aiohttp not installed; OpenAI dual-advice skipped.")
        return fallback
    except Exception as exc:
        log.warning(
            "[rationale_helper] OpenAI call error for %s (model=%s): %s. "
            "Skipping OpenAI advice.",
            symbol, model_id, str(exc)[:200],
        )
        return fallback


# ---------------------------------------------------------------------------
# Consensus helper
# ---------------------------------------------------------------------------

def _extract_direction_from_text(text: str) -> Optional[str]:
    """
    Attempt to extract directional bias from rationale text.
    Returns 'long', 'short', or 'neutral' (or None if undetectable).
    Heuristic: keyword counting. Not perfect — used only for consensus check.
    """
    t = text.lower()
    bullish_count = t.count("bullish") + t.count(" long ") + t.count("upside") + t.count("buy")
    bearish_count = t.count("bearish") + t.count(" short ") + t.count("downside") + t.count("sell")

    if bullish_count > bearish_count and bullish_count > 0:
        return "long"
    if bearish_count > bullish_count and bearish_count > 0:
        return "short"
    if bullish_count == 0 and bearish_count == 0:
        return "neutral"
    return "neutral"  # tied


# ---------------------------------------------------------------------------
# Primary public entry point
# ---------------------------------------------------------------------------

async def build_rationale(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
    config: dict,
    caller_logger: Optional[logging.Logger] = None,
    extra: Optional[dict] = None,
) -> str:
    """
    Primary entry point for all analyzer rationale generation.

    Anthropic is tried first (existing behaviour preserved). If dual-advice mode
    is enabled and an OpenAI key is present, OpenAI is also called. Results from
    both providers are stored in the `extra` dict (if supplied) under:
      extra["rationale_anthropic"] : str
      extra["rationale_openai"]    : str (only if OpenAI called)
      extra["providers_disagree"]  : bool (consensus mode only)
      extra["dual_advice_mode"]    : str  (which mode was active)

    Parameters
    ----------
    symbol          : Asset ticker / fund code (sanitized before LLM call).
    market          : Market enum value.
    horizon         : Horizon enum value.
    signals         : Technical-signal dict (keys: close, sma20, sma50, rsi,
                      bb_upper, bb_lower, bb_mid, sma_signal, rsi_signal,
                      bb_signal, vol_ratio). Partial dicts are tolerated.
    direction       : Direction the technicals voted (LONG/SHORT/NEUTRAL).
    config          : Advisor config dict (from AdvisorConfigManager).
    caller_logger   : Optional logger from the calling analyzer class.
    extra           : Optional dict to receive dual-advice artefacts.

    Returns
    -------
    str — always non-empty, always advice-only text (primary rationale).
    The Anthropic text is the primary; OpenAI text is supplementary.
    """
    log = caller_logger or logger
    fallback = build_rule_based_rationale(symbol, market, horizon, signals, direction)

    # COST SHORT-CIRCUIT (discovery layer): when the caller injects
    # _advisor_force_rule_based=true into config, return the FREE rule-based
    # rationale immediately and make NO paid LLM call. The discovery pass uses
    # this so trending candidates default to zero-cost narration; only the
    # explicitly budgeted top-N are LLM-narrated by the engine afterwards.
    if str(config.get("_advisor_force_rule_based", "false")).lower() == "true":
        if extra is not None:
            extra["rationale_rule_based_forced"] = True
        return fallback

    dual_mode = str(config.get("advisor_dual_advice_mode", _DUAL_MODE_OFF)).lower().strip()
    if dual_mode not in (_DUAL_MODE_OFF, _DUAL_MODE_BOTH, _DUAL_MODE_CONSENSUS):
        log.warning(
            "[rationale_helper] Unknown advisor_dual_advice_mode='%s'; defaulting to 'off'.",
            dual_mode,
        )
        dual_mode = _DUAL_MODE_OFF

    anthropic_key = _resolve_api_key(config)
    openai_key    = _resolve_openai_key(config) if dual_mode != _DUAL_MODE_OFF else None

    # --- LLM rationale cache (issue #12) ---
    # Reuse the previously generated rationale when the signal/direction has not
    # materially changed, so we do NOT re-hit the paid LLM API every advice cycle.
    # Applied only in single-provider ('off') mode — the dual-advice modes need
    # fresh per-cycle artefacts (providers_disagree) in `extra`.
    cache_enabled, cache_max_age_s = _cache_settings(config)
    use_cache = cache_enabled and dual_mode == _DUAL_MODE_OFF and bool(anthropic_key)
    cache_key = (
        _cache_key(symbol, market, horizon, direction, signals, dual_mode)
        if use_cache else None
    )
    if cache_key is not None:
        cached = _cache_get(cache_key, cache_max_age_s)
        if cached is not None:
            log.debug(
                "[rationale_helper] Rationale cache HIT for %s/%s/%s (%s) — "
                "skipping LLM call.",
                _sanitize(symbol), market.value, horizon.value, direction.value,
            )
            if extra is not None:
                extra["dual_advice_mode"] = dual_mode
                extra["rationale_cached"] = True
            return cached

    # --- Anthropic call ---
    # Hard global daily cap on paid LLM calls (shared with KAP classifier). When
    # exhausted, fall back to rule-based prose and make NO API call.
    from modules.advisor.core.llm_budget import try_consume
    if anthropic_key and try_consume(config, kind="advice_rationale", log=log):
        model_id      = config.get("advisor_anthropic_model", "claude-opus-4-8")
        signal_summary = _build_signal_summary(symbol, market, horizon, signals, direction)
        mkt_label      = _MARKET_LABEL.get(market, market.value)
        prompt         = _build_prompt(signal_summary, mkt_label)
        anthropic_text = await _call_anthropic(
            prompt, model_id, anthropic_key, fallback, log, symbol
        )
    else:
        anthropic_text = fallback

    primary_rationale = anthropic_text

    # --- Dual-advice: OpenAI call ---
    openai_text: Optional[str] = None
    if (dual_mode != _DUAL_MODE_OFF and openai_key
            and try_consume(config, kind="advice_openai", log=log)):
        openai_model  = config.get("advisor_openai_model", "gpt-4o")
        signal_summary = _build_signal_summary(symbol, market, horizon, signals, direction)
        mkt_label      = _MARKET_LABEL.get(market, market.value)
        prompt         = _build_prompt(signal_summary, mkt_label)
        openai_text    = await _call_openai(
            prompt, openai_model, openai_key, fallback, log, symbol
        )

    # --- Populate extra artefacts ---
    if extra is not None:
        extra["dual_advice_mode"] = dual_mode
        if dual_mode != _DUAL_MODE_OFF:
            extra["rationale_anthropic"] = anthropic_text
            if openai_text is not None:
                extra["rationale_openai"] = openai_text

        # Consensus mode: compare directional keywords extracted from both rationales.
        if dual_mode == _DUAL_MODE_CONSENSUS and openai_text is not None:
            dir_a = _extract_direction_from_text(anthropic_text)
            dir_o = _extract_direction_from_text(openai_text)
            disagree = (dir_a is not None and dir_o is not None and dir_a != dir_o)
            extra["providers_disagree"] = disagree
            if disagree:
                log.info(
                    "[rationale_helper] Consensus DISAGREE for %s: "
                    "anthropic=%s openai=%s",
                    _sanitize(symbol), dir_a, dir_o,
                )
            else:
                log.debug(
                    "[rationale_helper] Consensus AGREE for %s: %s",
                    _sanitize(symbol), dir_a,
                )

    # Cache the freshly generated rationale (only when the LLM actually produced
    # it — never cache the rule-based fallback, so a transient API failure can
    # retry next cycle instead of being pinned for hours).
    if cache_key is not None and primary_rationale and primary_rationale != fallback:
        _cache_put(cache_key, primary_rationale)
        if extra is not None:
            extra["rationale_cached"] = False

    return primary_rationale
