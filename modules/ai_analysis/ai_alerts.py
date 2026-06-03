"""
Telegram Alerts for the AI Analysis Module
===========================================
Engine-routed alert helper that mirrors ``futures_alerts.py`` /
``solana_alerts.py`` so AI trade events land in the AI forum topic with the
``[AI]`` module header via the shared :class:`TelegramNotificationEngine`.

Why this exists
---------------
The AI engine (``core/sentiment_engine.py`` / ``core/ai_trading_engine.py``)
historically emitted NOTHING to Telegram — only the module startup/shutdown
banner went out via ``telegram_bot``. This helper gives the engine a ready,
topic-routed alert object so per-trade entries/exits and errors reach the AI
topic. It is fail-soft and DRY_RUN-safe: every send is best-effort and never
raises into the trading loop.

Wire-in (one line in the engine, owning agent):
    from modules.ai_analysis.ai_alerts import AITelegramAlerts
    self.telegram_alerts = AITelegramAlerts()
    ...
    await self.telegram_alerts.send_entry_alert(AITradeAlert(...))
    await self.telegram_alerts.send_exit_alert(AITradeAlert(...))

Routing: every message goes through ``TelegramNotificationEngine.notify`` with
module='ai' so the engine prepends ``format_header('ai', ...)`` and routes to
``topic_thread_id_ai`` (thread 15). When the group is not configured the engine
falls back to the single-DM chat automatically.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger("ai.alerts")


@dataclass
class AITradeAlert:
    """AI trade alert data structure (entry or exit)."""
    symbol: str
    action: str          # 'entry' | 'exit' | 'stop_loss' | 'take_profit' | 'ai_recommendation'
    direction: str       # 'long' | 'short'
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    pnl_pct: Optional[float] = None
    reason: Optional[str] = None
    is_simulated: bool = True
    provider: Optional[str] = None    # 'openai' | 'anthropic' | ...
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class AITelegramAlerts:
    """Topic-routed Telegram alerts manager for the AI module."""

    MODULE = "ai"

    def __init__(self, enabled: bool = True):
        # Enablement is ultimately decided by the engine config
        # (telegram_group_id / notifications_enabled). We keep a local toggle
        # so a caller can hard-disable without touching DB config.
        self.enabled = enabled

    def _esc(self, text: Any) -> str:
        """MarkdownV2 escape via the engine's canonical escaper."""
        try:
            from monitoring.notification_engine import escape_mdv2
            return escape_mdv2(str(text))
        except Exception:
            special = '_*[]()~`>#+-=|{}.!'
            out = []
            for ch in str(text):
                if ch in special:
                    out.append('\\')
                out.append(ch)
            return ''.join(out)

    def _format_entry(self, alert: AITradeAlert) -> str:
        sim = "\\[SIM\\] " if alert.is_simulated else ""
        side_emoji = "🟢" if alert.direction.lower() == "long" else "🔴"
        lines = [
            f"{sim}🧠 *AI ENTRY*",
            "",
            f"{side_emoji} *{self._esc(alert.symbol)}* \\- {self._esc(alert.direction.upper())}",
            f"💰 Entry: ${self._esc(f'{alert.entry_price:.6f}')}",
        ]
        if alert.confidence is not None:
            lines.append(f"📊 Confidence: {self._esc(f'{alert.confidence:.2f}')}")
        if alert.strategy:
            lines.append(f"🎯 Strategy: {self._esc(alert.strategy)}")
        if alert.provider:
            lines.append(f"🤖 Provider: {self._esc(alert.provider)}")
        lines.append("")
        lines.append(f"_⏰ {self._esc(alert.timestamp.strftime('%Y-%m-%d %H:%M UTC'))}_")
        return "\n".join(lines)

    def _format_exit(self, alert: AITradeAlert) -> str:
        sim = "\\[SIM\\] " if alert.is_simulated else ""
        side_emoji = "🟢" if alert.direction.lower() == "long" else "🔴"
        pnl = alert.pnl_pct or 0.0
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        reason = self._esc((alert.reason or alert.action or "exit").replace("_", " ").upper())
        lines = [
            f"{sim}🧠 *AI EXIT* \\- {reason}",
            "",
            f"{side_emoji} *{self._esc(alert.symbol)}* \\- {self._esc(alert.direction.upper())}",
            f"💵 Entry: ${self._esc(f'{alert.entry_price:.6f}')}",
        ]
        if alert.exit_price is not None:
            lines.append(f"💰 Exit: ${self._esc(f'{alert.exit_price:.6f}')}")
        lines.append(f"{pnl_emoji} *P&L:* {self._esc(f'{pnl:+.2f}')}%")
        lines.append("")
        lines.append(f"_⏰ {self._esc(alert.timestamp.strftime('%Y-%m-%d %H:%M UTC'))}_")
        return "\n".join(lines)

    async def _send(self, message: str, category: str = "trade", level: str = "info") -> bool:
        """Route through the shared notification engine. Fail-soft."""
        if not self.enabled:
            return False
        try:
            from monitoring.notification_engine import get_engine
            engine = get_engine()
            cfg = await engine._load_config()
            if not (cfg.notifications_enabled and (cfg.telegram_group_id or engine._chat_id)):
                return False
            return await engine.notify(self.MODULE, category, message, level=level)
        except Exception as e:
            logger.debug(f"AI telegram alert send skipped: {e}")
            return False

    async def send_entry_alert(self, alert: AITradeAlert) -> bool:
        return await self._send(self._format_entry(alert), category="trade")

    async def send_exit_alert(self, alert: AITradeAlert) -> bool:
        return await self._send(self._format_exit(alert), category="trade")

    async def send_error(self, error: str) -> bool:
        """Route an error to the AI/error topic with engine de-dup."""
        try:
            from monitoring.notification_engine import format_header
            header = format_header(self.MODULE, "Error")
            text = f"{header}\n{self._esc(str(error)[:800])}"
        except Exception:
            text = self._esc(str(error)[:800])
        return await self._send(text, category="error", level="error")

    async def send_custom(self, title: str, content: str, category: str = "trade") -> bool:
        try:
            from monitoring.notification_engine import format_header
            header = format_header(self.MODULE, title)
            text = f"{header}\n{self._esc(content)}"
        except Exception:
            text = f"*{self._esc(title)}*\n{self._esc(content)}"
        return await self._send(text, category=category)
