"""
Telegram Alerts for the Copy Trading Module
============================================
Engine-routed alert helper that mirrors ``futures_alerts.py`` /
``solana_alerts.py`` so copy-trade events land in the COPY forum topic with the
``[COPY]`` module header via the shared :class:`TelegramNotificationEngine`.

Why this exists
---------------
The copy engine (``copy_engine.py``) historically emitted NOTHING to Telegram —
only the module startup/shutdown banner went out via ``telegram_bot``. This
helper gives the engine a ready, topic-routed alert object so each mirrored
BUY/SELL (or a skip/error) reaches the COPY topic. Fail-soft and DRY_RUN-safe.

Wire-in (one line in the engine, owning web3 agent):
    from modules.copy_trading.copy_alerts import CopyTelegramAlerts, CopyTradeAlert
    self.telegram_alerts = CopyTelegramAlerts()
    ...
    await self.telegram_alerts.send_copy_alert(CopyTradeAlert(
        action='buy', token='BONK', chain='solana',
        leader='Coyadnds...', amount_usd=42.0, is_simulated=self.dry_run))

Routing: every message goes through ``TelegramNotificationEngine.notify`` with
module='copy' so the engine prepends ``format_header('copy', ...)`` and routes
to ``topic_thread_id_copy`` (thread 18). When the group is not configured the
engine falls back to the single-DM chat automatically.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

logger = logging.getLogger("copy.alerts")


@dataclass
class CopyTradeAlert:
    """Copy-trade alert data structure (mirrored BUY or SELL)."""
    action: str          # 'buy' | 'sell'
    token: str           # token symbol or address
    chain: str           # 'solana' | 'ethereum' | 'base' | ...
    leader: str = ""     # leader wallet (label or short address)
    amount_usd: Optional[float] = None
    tx_hash: Optional[str] = None
    pnl_pct: Optional[float] = None   # populated on SELL/close
    reason: Optional[str] = None
    is_simulated: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CopyTelegramAlerts:
    """Topic-routed Telegram alerts manager for the Copy Trading module."""

    MODULE = "copy"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _esc(self, text: Any) -> str:
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

    def _short(self, addr: str) -> str:
        addr = str(addr or "")
        if len(addr) > 14:
            return f"{addr[:6]}…{addr[-4:]}"
        return addr

    def _format_copy(self, alert: CopyTradeAlert) -> str:
        sim = "\\[SIM\\] " if alert.is_simulated else ""
        is_buy = alert.action.lower() == "buy"
        action_emoji = "🟢 BUY" if is_buy else "🔴 SELL"
        lines = [
            f"{sim}👯 *COPY {action_emoji}*",
            "",
            f"🪙 Token: *{self._esc(alert.token)}*",
            f"⛓ Chain: {self._esc(alert.chain.upper())}",
        ]
        if alert.leader:
            lines.append(f"👤 Leader: `{self._esc(self._short(alert.leader))}`")
        if alert.amount_usd is not None:
            lines.append(f"💵 Size: ${self._esc(f'{alert.amount_usd:.2f}')}")
        if alert.pnl_pct is not None:
            pnl_emoji = "✅" if alert.pnl_pct >= 0 else "❌"
            lines.append(f"{pnl_emoji} P&L: {self._esc(f'{alert.pnl_pct:+.2f}')}%")
        if alert.tx_hash:
            lines.append(f"🔗 Tx: `{self._esc(self._short(alert.tx_hash))}`")
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
            logger.debug(f"Copy telegram alert send skipped: {e}")
            return False

    async def send_copy_alert(self, alert: CopyTradeAlert) -> bool:
        return await self._send(self._format_copy(alert), category="trade")

    async def send_error(self, error: str) -> bool:
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
