"""
Telegram Alerts for Futures Trading Module
Sends real-time notifications for trade entries, exits, and performance updates
"""

import os
import logging
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FuturesTradeAlert:
    """Trade alert data structure"""
    symbol: str
    side: str  # 'long' or 'short'
    action: str  # 'entry', 'exit', 'sl_hit', 'tp_hit', 'trailing_stop', 'manual_close'
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0
    leverage: int = 1
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    reason: Optional[str] = None
    is_simulated: bool = True
    exchange: str = 'binance'
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FuturesTelegramAlerts:
    """
    Telegram alerts manager for Futures trading
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram alerts

        Args:
            bot_token: Telegram bot token (from .env if not provided)
            chat_id: Telegram chat ID (from .env if not provided)
            enabled: Whether alerts are enabled
        """
        # Try secrets manager first (reads from database), then fall back to env
        try:
            from security.secrets_manager import secrets
            self.bot_token = bot_token or secrets.get('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
            self.chat_id = chat_id or secrets.get('TELEGRAM_CHAT_ID') or os.getenv('TELEGRAM_CHAT_ID')
        except Exception:
            self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
            self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)

        if not self.enabled:
            logger.warning("Futures Telegram alerts disabled - missing bot token or chat ID")
        else:
            logger.info("✅ Futures Telegram alerts initialized")

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = str(text).replace(char, f'\\{char}')
        return text

    def _format_entry_alert(self, alert: FuturesTradeAlert) -> str:
        """Format entry trade alert message"""
        sim_tag = "[SIM] " if alert.is_simulated else ""
        side_emoji = "🟢" if alert.side == 'long' else "🔴"

        message = f"""
{sim_tag}📈 *FUTURES ENTRY*

{side_emoji} *{alert.symbol}* \\- {alert.side.upper()}

💰 Entry: ${self._escape_markdown(f"{alert.entry_price:.4f}")}
📊 Size: {self._escape_markdown(f"{alert.size:.4f}")}
⚡ Leverage: {alert.leverage}x
🏦 Exchange: {alert.exchange.upper()}

🛑 Stop Loss: ${self._escape_markdown(f"{alert.stop_loss:.4f}" if alert.stop_loss else "N/A")}
🎯 Take Profit: ${self._escape_markdown(f"{alert.take_profit:.4f}" if alert.take_profit else "N/A")}
📉 Trailing SL: {self._escape_markdown(f"{alert.trailing_stop}%" if alert.trailing_stop else "N/A")}

⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_exit_alert(self, alert: FuturesTradeAlert) -> str:
        """Format exit trade alert message"""
        sim_tag = "[SIM] " if alert.is_simulated else ""
        side_emoji = "🟢" if alert.side == 'long' else "🔴"

        # Determine exit reason emoji
        reason_map = {
            'take_profit': '🎯 TP HIT',
            'take_profit_1': '🎯 TP1 HIT',
            'take_profit_2': '🎯 TP2 HIT',
            'take_profit_3': '🎯 TP3 HIT',
            'take_profit_4': '🎯 TP4 HIT',
            'stop_loss': '🛑 SL HIT',
            'trailing_stop': '📉 TRAILING SL',
            'manual_close': '👤 MANUAL CLOSE',
            'signal_reversal': '📊 SIGNAL REVERSAL',
            'signal': '📊 SIGNAL EXIT',
            'liquidation': '💥 LIQUIDATION',
            'liquidation_protection': '💥 LIQ PROTECTION'
        }
        # Get exit reason and escape it for MarkdownV2
        exit_reason = reason_map.get(alert.action, self._escape_markdown(alert.reason or 'EXIT'))

        # PnL formatting
        pnl = alert.pnl or 0
        pnl_pct = alert.pnl_pct or 0
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        pnl_color = "🟩" if pnl >= 0 else "🟥"

        message = f"""
{sim_tag}📉 *FUTURES EXIT* \\- {exit_reason}

{side_emoji} *{alert.symbol}* \\- {alert.side.upper()}

💵 Entry: ${self._escape_markdown(f"{alert.entry_price:.4f}")}
💰 Exit: ${self._escape_markdown(f"{alert.exit_price:.4f}" if alert.exit_price else "N/A")}
📊 Size: {self._escape_markdown(f"{alert.size:.4f}")}
⚡ Leverage: {alert.leverage}x

{pnl_color} *P&L:* {pnl_emoji} ${self._escape_markdown(f"{pnl:+.2f}")} \\({self._escape_markdown(f"{pnl_pct:+.2f}")}%\\)

🏦 Exchange: {alert.exchange.upper()}
⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_stats_alert(self, stats: Dict[str, Any]) -> str:
        """Format daily stats summary"""
        message = f"""
📊 *FUTURES DAILY STATS*

📈 Total Trades: {self._escape_markdown(str(stats.get('total_trades', 0)))}
✅ Winning: {self._escape_markdown(str(stats.get('winning_trades', 0)))}
❌ Losing: {self._escape_markdown(str(stats.get('losing_trades', 0)))}
🎯 Win Rate: {self._escape_markdown(str(stats.get('win_rate', '0%')))}

💰 Total P&L: {self._escape_markdown(str(stats.get('net_pnl', '$0.00')))}
📅 Daily P&L: {self._escape_markdown(str(stats.get('daily_pnl', '$0.00')))}
📊 Unrealized: {self._escape_markdown(str(stats.get('unrealized_pnl', '$0.00')))}

📉 Max Drawdown: {self._escape_markdown(str(stats.get('max_drawdown_pct', '0%')))}
📍 Active Positions: {self._escape_markdown(str(stats.get('active_positions', 0)))}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    async def send_message(
        self,
        message: str,
        parse_mode: str = 'MarkdownV2',
        category: str = 'trade',
    ) -> bool:
        """
        Send a message to Telegram.

        Wave-19: tries topic routing via TelegramNotificationEngine first
        (respects group/thread_id config, rate-limits, verbosity mode).
        Falls back to direct sendMessage when the engine is not configured
        or unavailable so existing single-DM behaviour is preserved.

        Args:
            message: MarkdownV2 formatted message text
            parse_mode: Telegram parse mode (MarkdownV2 or Markdown)
            category: notification category ('trade', 'error', 'summary')
        """
        if not self.enabled:
            return False

        # Wave-19: try notification engine (topic routing + rate-limit)
        try:
            from monitoring.notification_engine import get_engine
            engine = get_engine()
            cfg = await engine._load_config()
            if cfg.notifications_enabled and cfg.telegram_group_id:
                sent = await engine.notify('futures', category, message, level='info')
                if sent:
                    return True
                # Fall through to direct send if engine suppressed (rate-limit/verbosity)
                # but still needs to send (engine returns False on suppress, not error)
        except Exception:
            pass

        # Backward-compatible direct send
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_entry_alert(self, alert: FuturesTradeAlert) -> bool:
        """Send trade entry alert"""
        message = self._format_entry_alert(alert)
        return await self.send_message(message, category='trade')

    async def send_exit_alert(self, alert: FuturesTradeAlert) -> bool:
        """Send trade exit alert"""
        message = self._format_exit_alert(alert)
        return await self.send_message(message, category='trade')

    async def send_stats_summary(self, stats: Dict[str, Any]) -> bool:
        """Send daily stats summary"""
        message = self._format_stats_alert(stats)
        return await self.send_message(message, category='summary')

    async def send_custom_alert(self, title: str, content: str) -> bool:
        """Send a custom alert message"""
        message = f"""
🔔 *{self._escape_markdown(title)}*

{self._escape_markdown(content)}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return await self.send_message(message.strip())
