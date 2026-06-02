"""
AdvisorTelegramBot — SEPARATE Telegram bot for the advisor module.

This is NOT the shared TelegramBotController used by the trading modules.
The advisor uses its own bot token and chat ID so that:
  - Advice messages are routed to a different chat/group than trade alerts.
  - The advisor can be silenced/enabled independently.
  - Keys are isolated: ADVISOR_TELEGRAM_BOT_TOKEN, ADVISOR_TELEGRAM_CHAT_ID.

Bot token / chat ID resolution order (mirrors Wave-6 AI module pattern):
  1. secrets_manager.get_async('ADVISOR_TELEGRAM_BOT_TOKEN')
     (Docker secret -> encrypted advisor_config DB row -> env fallback)
  2. os.getenv('ADVISOR_TELEGRAM_BOT_TOKEN')

Config keys (advisor_config DB):
  advisor_telegram_bot_token  : bot token (stored in Secure Credentials)
  advisor_telegram_chat_id    : target chat/group/channel ID
  advisor_telegram_enabled    : "true" | "false" (default "true")
  advisor_telegram_parse_mode : "HTML" | "Markdown" (default "HTML")

Message format (rich HTML — TG specialist agent fills formatting in Wave-21):
  Header: market emoji + symbol + direction arrow
  Body: entry range, target, stop, confidence bar, horizon
  Footer: rationale (truncated), model ID, timestamp
  Kronos line: appended when kronos_signal is not None.

STUB STATUS (Wave-20): send_advice() is a stub that logs the intent and
returns without actually sending. The TG specialist agent implements the
HTTP call to Telegram Bot API in Wave-21.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from modules.advisor.core.models import AdviceResult, Direction, Market

logger = logging.getLogger("advisor.telegram_notifier")

# Market → emoji prefix for messages
_MARKET_EMOJI = {
    Market.CRYPTO: "CRYPTO",
    Market.US_EQUITIES: "US EQ",
    Market.BIST: "BIST",
    Market.FX: "FX",
    Market.MIDAS_FUNDS: "FONU",
}

_DIRECTION_ARROW = {
    Direction.LONG: "BUY",
    Direction.SHORT: "SELL",
    Direction.NEUTRAL: "WATCH",
}


class AdvisorTelegramBot:
    """
    Telegram notifier for the advisor module.

    Fully isolated from monitoring.telegram_bot (the trading-module bot).
    Uses ADVISOR_TELEGRAM_BOT_TOKEN + ADVISOR_TELEGRAM_CHAT_ID exclusively.
    """

    def __init__(self, config: dict):
        self.config = config
        self._token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._enabled: bool = True
        self._session = None   # aiohttp.ClientSession, lazy-init

    async def initialize(self, db_pool=None) -> bool:
        """
        Resolve bot token and chat ID.
        Returns True if fully configured, False if not (silent on failure).
        """
        self._enabled = (
            self.config.get("advisor_telegram_enabled", "true").lower() == "true"
        )
        if not self._enabled:
            logger.info("[advisor_tg] Telegram notifications disabled by config.")
            return False

        # Resolve token: secrets_manager first, then env
        self._token = await _resolve_secret(
            "ADVISOR_TELEGRAM_BOT_TOKEN",
            self.config.get("advisor_telegram_bot_token"),
            db_pool,
        )
        self._chat_id = await _resolve_secret(
            "ADVISOR_TELEGRAM_CHAT_ID",
            self.config.get("advisor_telegram_chat_id"),
            db_pool,
        )

        if not self._token or not self._chat_id:
            logger.warning(
                "[advisor_tg] Bot token or chat ID not configured. "
                "Set ADVISOR_TELEGRAM_BOT_TOKEN + ADVISOR_TELEGRAM_CHAT_ID "
                "in Secure Credentials or advisor_config."
            )
            return False

        logger.info(
            f"[advisor_tg] Configured: token=****{self._token[-4:]} "
            f"chat={self._chat_id}"
        )
        return True

    async def send_advice(self, result: AdviceResult) -> bool:
        """
        Send an advice notification to the advisor Telegram chat.

        STUB (Wave-20): logs the intent; actual HTTP send deferred to
        Wave-21 TG specialist agent. Return True when send succeeded.

        Wave-21 implementation steps:
          1. Build HTML message via _format_advice_message(result).
          2. POST to https://api.telegram.org/bot{token}/sendMessage
             with chat_id, text, parse_mode=HTML, disable_web_page_preview=True.
          3. Handle 429 rate-limit with exponential back-off.
          4. Handle 403/404 token errors with loud WARNING (never silent fail).
        """
        if not self._enabled or not self._token:
            return False

        msg = _format_advice_message(result)

        # STUB: log the message that would be sent
        logger.info(
            f"[advisor_tg] STUB send_advice: {result.market.value} "
            f"{result.symbol} {result.direction.value} "
            f"horizon={result.horizon.value} "
            f"conf={result.confidence:.2f}"
        )
        logger.debug(f"[advisor_tg] Message preview:\n{msg}")

        # Wave-21: replace the stub above with real HTTP send
        # return await self._send_text(msg)
        return True

    async def send_text(self, text: str) -> bool:
        """
        Send arbitrary text to the advisor chat.
        STUB — same as send_advice; Wave-21 wires the HTTP call.
        """
        if not self._enabled or not self._token:
            return False
        logger.info(f"[advisor_tg] STUB send_text: {text[:80]}")
        return True

    async def close(self) -> None:
        """Close the aiohttp session if open."""
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Formatting helpers (stubs — TG specialist fills rich HTML in Wave-21)
# ---------------------------------------------------------------------------

def _format_advice_message(result: AdviceResult) -> str:
    """
    Build a Telegram HTML message for an advice result.
    STUB: plain-text format. TG specialist upgrades to rich HTML in Wave-21.
    """
    market_label = _MARKET_EMOJI.get(result.market, result.market.value.upper())
    direction_label = _DIRECTION_ARROW.get(result.direction, result.direction.value)

    lines = [
        f"[{market_label}] {result.symbol} — {direction_label} ({result.horizon.value.upper()})",
        f"Confidence: {result.confidence:.0%}",
    ]
    if result.entry_low and result.entry_high:
        lines.append(
            f"Entry: {result.entry_low:.4f} – {result.entry_high:.4f}"
        )
    if result.target_price:
        lines.append(f"Target: {result.target_price:.4f}")
    if result.stop_price:
        lines.append(f"Stop: {result.stop_price:.4f}")
    if result.kronos_signal is not None:
        bias = "bullish" if result.kronos_signal > 0 else "bearish"
        lines.append(f"Kronos: {bias} ({result.kronos_signal:+.3f})")
    if result.rationale:
        # Truncate rationale for Telegram (4096 char total limit)
        lines.append(f"\n{result.rationale[:600]}")
    lines.append(f"\nModel: {result.model_id}")
    lines.append(f"Generated: {result.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("\nADVICE ONLY — no trade is executed automatically.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Secret resolution helper (mirrors Wave-6 AI module pattern)
# ---------------------------------------------------------------------------

async def _resolve_secret(
    env_key: str,
    config_value: Optional[str],
    db_pool,
) -> Optional[str]:
    """
    Resolve a secret in order:
    1. secrets_manager.get_async (DB-backed encrypted store)
    2. config_value (from advisor_config DB row, if pre-loaded)
    3. os.getenv(env_key)
    """
    if db_pool is not None:
        try:
            from security.secrets_manager import secrets as _secrets
            _secrets.initialize(db_pool)
            val = await _secrets.get_async(env_key, log_access=False)
            if val:
                return val
        except Exception as exc:
            logger.debug(f"[advisor_tg] secrets_manager lookup failed for {env_key}: {exc}")

    if config_value:
        return config_value

    return os.getenv(env_key)
