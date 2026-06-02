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
  advisor_telegram_bot_token    : bot token (Secure Credentials)
  advisor_telegram_chat_id      : target chat/group/channel ID
  advisor_telegram_enabled      : "true" | "false" (default "true")
  advisor_telegram_digest_hour  : int 0-23, local hour for daily digest
  advisor_telegram_digest_tz    : IANA tz name (default "Europe/Istanbul")

Message format (rich HTML):
  Header: market emoji + symbol + direction arrow
  Body: entry range, target, stop, confidence bar (filled blocks), horizon
  Footer: rationale (truncated to 500 chars), model ID, timestamp
  Kronos line: appended when kronos_signal is not None.

Daily digest design:
  - One message per market with that day's short/mid/long advice.
  - Sent at advisor_telegram_digest_hour in digest timezone.
  - De-dup: a set of sent advice IDs (in-memory) prevents re-sending
    the same DB row. Reset at midnight each day.
  - A daily portfolio + sim summary message is sent after market digests.
  - If ADVISOR_TELEGRAM_BOT_TOKEN not set: log-only, no crash.

Rate limits:
  - 429 response triggers exponential back-off (1s→2s→4s→8s→16s, max 5
    retries, then warns and returns False).
  - 403/404 token errors log loud WARNING; no retry.
  - All errors are non-fatal: the advice cycle continues.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Set

from modules.advisor.core.models import AdviceResult, Direction, Market, Horizon

logger = logging.getLogger("advisor.telegram_notifier")

# ---------------------------------------------------------------------------
# Market and direction display helpers
# ---------------------------------------------------------------------------

_MARKET_EMOJI: Dict[Market, str] = {
    Market.CRYPTO:       "₿ CRYPTO",
    Market.US_EQUITIES:  "📈 US EQUITIES",
    Market.BIST:         "🇹🇷 BIST",
    Market.FX:           "💱 FX & METALS",
    Market.MIDAS_FUNDS:  "🏦 MIDAS FUNDS",
}

_DIRECTION_EMOJI: Dict[Direction, str] = {
    Direction.LONG:    "🟢 LONG",
    Direction.SHORT:   "🔴 SHORT",
    Direction.NEUTRAL: "🟡 HOLD",
}

_HORIZON_LABEL: Dict[Horizon, str] = {
    Horizon.SHORT: "Short (1d–1w)",
    Horizon.MID:   "Mid (1w–3mo)",
    Horizon.LONG:  "Long (3mo–2y)",
}

_CONF_BLOCKS = "▓▓▓▓▓▓▓▓▓▓"
_EMPTY_BLOCKS = "░░░░░░░░░░"

_TG_API_BASE = "https://api.telegram.org"
_MAX_MESSAGE_LEN = 4096
_MAX_RETRIES = 5


def _conf_bar(conf: float, width: int = 10) -> str:
    """ASCII confidence bar using filled/empty Unicode blocks."""
    filled = round(conf * width)
    return _CONF_BLOCKS[:filled] + _EMPTY_BLOCKS[:width - filled]


def _fmt_price(n: Optional[float]) -> str:
    if n is None:
        return "—"
    if n >= 1000:
        return f"{n:,.2f}"
    if n >= 1:
        return f"{n:.4f}"
    return f"{n:.6f}"


def _html(s: str) -> str:
    """Escape HTML special chars for Telegram HTML parse mode."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class AdvisorTelegramBot:
    """
    Telegram notifier for the advisor module.

    Fully isolated from monitoring.telegram_bot (the trading-module bot).
    Uses ADVISOR_TELEGRAM_BOT_TOKEN + ADVISOR_TELEGRAM_CHAT_ID exclusively.

    send_advice(result):
        Sends a single rich HTML message immediately. Called by AdviceEngine
        per advice event. This is intentionally immediate (not batched) so the
        operator gets per-event alerts when advice is produced.

    send_daily_digest(advice_rows_by_market):
        Sends one message per market with the day's short/mid/long summary.
        De-dup: tracks sent advice IDs so the same row is never re-sent even
        if digest is called multiple times in the same day.
        Called by AdviceEngine once per day at advisor_telegram_digest_hour.

    send_portfolio_summary(holdings, sim_summary):
        Sends a daily portfolio value + sim P&L summary message.
    """

    def __init__(self, config: dict):
        self.config = config
        self._token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._enabled: bool = True
        self._session = None       # aiohttp.ClientSession, lazy-init
        self._sent_advice_ids: Set[int] = set()   # de-dup per-session
        self._digest_day: Optional[int] = None    # day-of-year for nightly reset

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, db_pool=None) -> bool:
        """
        Resolve bot token and chat ID.
        Returns True if fully configured, False if not (no crash on failure).
        """
        self._enabled = str(
            self.config.get("advisor_telegram_enabled", "true")
        ).lower() == "true"
        if not self._enabled:
            logger.info("[advisor_tg] Telegram notifications disabled by config.")
            return False

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

    async def close(self) -> None:
        """Close the aiohttp session if open."""
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    async def send_advice(self, result: AdviceResult) -> bool:
        """
        Send a single rich HTML advice notification immediately.
        De-dup by advice DB id (result.extra.get('db_id')) if present.
        Returns True if message was sent (or already sent); False on error.
        """
        if not self._enabled or not self._token:
            return False

        db_id: Optional[int] = result.extra.get("db_id")
        if db_id is not None and db_id in self._sent_advice_ids:
            logger.debug(f"[advisor_tg] Skipping duplicate advice id={db_id}")
            return True

        msg = _format_advice_message(result)
        ok = await self._send_text(msg)
        if ok and db_id is not None:
            self._sent_advice_ids.add(db_id)
        return ok

    async def send_daily_digest(
        self,
        advice_by_market: Dict[str, list],
    ) -> bool:
        """
        Send one Telegram message per market with the day's advice summary.

        advice_by_market: { market_value_str: [AdviceResult, ...] }
        Each market produces one consolidated HTML message showing the
        short/mid/long signals for every symbol in that market.

        De-dup: advice IDs already in _sent_advice_ids are skipped.
        _sent_advice_ids is reset at midnight (new digest_day).
        """
        if not self._enabled or not self._token:
            return False

        today = datetime.now(timezone.utc).timetuple().tm_yday
        if today != self._digest_day:
            self._digest_day = today
            self._sent_advice_ids.clear()
            logger.debug("[advisor_tg] De-dup set reset for new digest day")

        sent_any = False
        for market_str, results in advice_by_market.items():
            if not results:
                continue
            msg = _format_market_digest(market_str, results, self._sent_advice_ids)
            if not msg:
                continue
            ok = await self._send_text(msg)
            if ok:
                # Mark all included db_ids as sent
                for r in results:
                    db_id = r.extra.get("db_id")
                    if db_id is not None:
                        self._sent_advice_ids.add(db_id)
                sent_any = True
            # Small pause between market messages to avoid flooding
            await asyncio.sleep(0.5)

        return sent_any

    async def send_portfolio_summary(
        self,
        total_value: float,
        total_pnl: float,
        open_sims: int,
        sim_pnl_usd: float,
        win_rate: float,
    ) -> bool:
        """
        Send a daily portfolio + simulation summary message.
        """
        if not self._enabled or not self._token:
            return False

        pnl_sign = "+" if total_pnl >= 0 else ""
        sim_sign = "+" if sim_pnl_usd >= 0 else ""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "📊 <b>Daily Portfolio &amp; Sim Summary</b>",
            "",
            f"💼 Portfolio Value: <b>${total_value:,.2f}</b>",
            f"  Unrealized P&amp;L: <b>{pnl_sign}${total_pnl:,.2f}</b>",
            "",
            "🧪 <b>Simulation Tracker</b>",
            f"  Open positions: <b>{open_sims}</b>",
            f"  Closed P&amp;L: <b>{sim_sign}${abs(sim_pnl_usd):,.2f}</b>",
            f"  Win rate: <b>{win_rate:.1f}%</b>",
            "",
            f"<i>{ts}</i>",
            "<i>ADVICE ONLY — no trades executed automatically.</i>",
        ]
        return await self._send_text("\n".join(lines))

    async def send_text(self, text: str) -> bool:
        """
        Send arbitrary text to the advisor chat (HTML parse mode).
        No-op if token not configured.
        """
        if not self._enabled or not self._token:
            return False
        return await self._send_text(text)

    # ------------------------------------------------------------------
    # Internal HTTP transport
    # ------------------------------------------------------------------

    async def _get_session(self):
        """Lazy-init aiohttp.ClientSession."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15)
                )
            except ImportError:
                logger.error(
                    "[advisor_tg] aiohttp not installed — cannot send Telegram messages. "
                    "Run: pip install aiohttp"
                )
                return None
        return self._session

    async def _send_text(self, text: str) -> bool:
        """
        POST text to Telegram sendMessage API with exponential back-off on 429.
        Handles 403/404 with loud WARNING and no retry.
        Returns True on success, False on failure.
        """
        if not self._token or not self._chat_id:
            return False

        # Telegram hard limit: 4096 chars per message
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[:_MAX_MESSAGE_LEN - 10] + "\n<i>…</i>"

        session = await self._get_session()
        if session is None:
            return False

        url = f"{_TG_API_BASE}/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        delay = 1.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    if resp.status == 429:
                        body = await resp.json()
                        retry_after = body.get("parameters", {}).get("retry_after", delay)
                        wait = max(float(retry_after), delay)
                        logger.warning(
                            f"[advisor_tg] 429 rate-limit on attempt {attempt}/{_MAX_RETRIES}. "
                            f"Waiting {wait:.1f}s…"
                        )
                        await asyncio.sleep(wait)
                        delay = min(delay * 2, 60.0)
                        continue
                    if resp.status in (401, 403, 404):
                        body = await resp.text()
                        logger.warning(
                            f"[advisor_tg] HTTP {resp.status} — likely invalid token or "
                            f"bot not in chat. Response: {body[:200]}. "
                            f"Check ADVISOR_TELEGRAM_BOT_TOKEN in Secure Credentials."
                        )
                        return False
                    # Other HTTP error
                    body = await resp.text()
                    logger.warning(f"[advisor_tg] HTTP {resp.status}: {body[:200]}")
                    return False

            except Exception as exc:
                logger.warning(f"[advisor_tg] send attempt {attempt} failed: {exc}")
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)

        logger.error(
            f"[advisor_tg] Failed to send message after {_MAX_RETRIES} attempts. "
            "Check network connectivity and token validity."
        )
        return False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_advice_message(result: AdviceResult) -> str:
    """
    Build a rich Telegram HTML message for a single advice result.
    Mobile-friendly: short lines, clear emoji hierarchy, price fields
    on separate lines.
    """
    market_label = _MARKET_EMOJI.get(result.market, result.market.value.upper())
    dir_label = _DIRECTION_EMOJI.get(result.direction, result.direction.value.upper())
    horizon_label = _HORIZON_LABEL.get(result.horizon, result.horizon.value)
    conf_pct = round(result.confidence * 100)
    bar = _conf_bar(result.confidence)

    lines = [
        f"<b>{market_label}</b>",
        f"<b>{_html(result.symbol)}</b>  {dir_label}",
        f"Horizon: <i>{horizon_label}</i>",
        "",
        f"Confidence: <b>{conf_pct}%</b>  <code>{bar}</code>",
    ]

    if result.entry_low is not None and result.entry_high is not None:
        lines.append(
            f"Entry:  <code>{_fmt_price(result.entry_low)} – {_fmt_price(result.entry_high)}</code>"
        )
    if result.target_price is not None:
        lines.append(f"Target: <code>{_fmt_price(result.target_price)}</code>  🎯")
    if result.stop_price is not None:
        lines.append(f"Stop:   <code>{_fmt_price(result.stop_price)}</code>  🛑")

    if result.kronos_signal is not None:
        bias = "Bullish" if result.kronos_signal > 0 else "Bearish"
        k_icon = "📈" if result.kronos_signal > 0 else "📉"
        lines.append(
            f"Kronos: {k_icon} {bias} (<code>{result.kronos_signal:+.3f}</code>)"
        )

    if result.data_source_note:
        lines.append(f"<i>Data: {_html(result.data_source_note[:80])}</i>")

    if result.rationale:
        rat = result.rationale[:500]
        if len(result.rationale) > 500:
            rat += "…"
        lines.append("")
        lines.append(_html(rat))

    ts = result.created_at.strftime("%Y-%m-%d %H:%M UTC")
    lines.append("")
    lines.append(
        f"<i>{_html(result.model_id or 'unknown')} | {ts}</i>"
    )
    lines.append("<i>ADVICE ONLY — no trade is executed automatically.</i>")

    return "\n".join(lines)


def _format_market_digest(
    market_str: str,
    results: list,
    already_sent: Set[int],
) -> str:
    """
    Build a single Telegram message summarising all advice in a market for the
    current day. Skips advice IDs already in already_sent.

    Format:
      Header: MARKET NAME
      Per symbol (grouped): SYMBOL + dir per horizon
      Footer: count + timestamp
    """
    # Group by symbol
    by_symbol: Dict[str, Dict[str, AdviceResult]] = {}
    for r in results:
        db_id = r.extra.get("db_id")
        if db_id is not None and db_id in already_sent:
            continue
        if r.symbol not in by_symbol:
            by_symbol[r.symbol] = {}
        # Keep highest-confidence per horizon
        existing = by_symbol[r.symbol].get(r.horizon.value if hasattr(r.horizon, 'value') else r.horizon)
        if existing is None or r.confidence > existing.confidence:
            h = r.horizon.value if hasattr(r.horizon, "value") else r.horizon
            by_symbol[r.symbol][h] = r

    if not by_symbol:
        return ""  # All already sent

    try:
        market_enum = Market(market_str)
        market_label = _MARKET_EMOJI.get(market_enum, market_str.upper())
    except ValueError:
        market_label = market_str.upper()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"<b>{market_label}</b>",
        f"<i>Daily digest — {today}</i>",
        "",
    ]

    for symbol, horizons in sorted(by_symbol.items()):
        sym_lines = [f"<b>{_html(symbol)}</b>"]
        for h_name in ("short", "mid", "long"):
            r = horizons.get(h_name)
            if not r:
                continue
            dir_icon = {"long": "🟢", "short": "🔴", "neutral": "🟡"}.get(
                r.direction.value if hasattr(r.direction, "value") else r.direction, "⚪"
            )
            dir_str = (r.direction.value if hasattr(r.direction, "value") else r.direction).upper()
            conf_pct = round(r.confidence * 100)
            entry_str = ""
            if r.entry_low and r.entry_high:
                entry_str = f"  entry {_fmt_price(r.entry_low)}–{_fmt_price(r.entry_high)}"
            sym_lines.append(
                f"  {h_name.capitalize()}: {dir_icon} <b>{dir_str}</b> "
                f"({conf_pct}%){entry_str}"
            )
            if r.target_price:
                sym_lines.append(f"    Target {_fmt_price(r.target_price)}")
        lines.extend(sym_lines)
        lines.append("")

    count = sum(len(v) for v in by_symbol.values())
    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines.append(f"<i>{count} signal(s) | {ts}</i>")
    lines.append("<i>ADVICE ONLY — no trade is executed automatically.</i>")

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
            logger.debug(
                f"[advisor_tg] secrets_manager lookup failed for {env_key}: {exc}"
            )

    if config_value and str(config_value).strip():
        return config_value

    return os.getenv(env_key)
