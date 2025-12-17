"""
Telegram Alerts for Solana Trading Module
Sends real-time notifications for trade entries, exits, and performance updates
Distinguishable from other modules with Solana-themed emojis
"""

import os
import logging
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SolanaTradeAlert:
    """Trade alert data structure for Solana trades"""
    token_symbol: str
    token_mint: str
    strategy: str  # 'jupiter', 'pumpfun', 'drift'
    action: str  # 'entry', 'exit', 'sl_hit', 'tp_hit', 'time_exit', 'manual_close'
    entry_price: float
    exit_price: Optional[float] = None
    amount_sol: float = 0
    token_amount: float = 0
    pnl_sol: Optional[float] = None
    pnl_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    reason: Optional[str] = None
    is_simulated: bool = True
    sol_price_usd: float = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SolanaTelegramAlerts:
    """
    Telegram alerts manager for Solana trading
    Uses distinct Solana-themed styling to differentiate from other modules
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
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)

        if not self.enabled:
            logger.warning("Solana Telegram alerts disabled - missing bot token or chat ID")
        else:
            logger.info("✅ Solana Telegram alerts initialized")

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = str(text).replace(char, f'\\{char}')
        return text

    def _get_strategy_emoji(self, strategy: str) -> str:
        """Get emoji for strategy"""
        strategy_map = {
            'jupiter': '🪐',  # Jupiter planet
            'pumpfun': '🚀',  # Rocket for meme tokens
            'drift': '📊',    # Chart for perps
        }
        return strategy_map.get(strategy.lower(), '💎')

    def _get_strategy_label(self, strategy: str) -> str:
        """Get display label for strategy"""
        strategy_map = {
            'jupiter': 'JUPITER',
            'pumpfun': 'PUMP.FUN',
            'drift': 'DRIFT',
        }
        return strategy_map.get(strategy.lower(), strategy.upper())

    def _format_entry_alert(self, alert: SolanaTradeAlert) -> str:
        """Format entry trade alert message with Solana branding"""
        sim_tag = "\\[SIM\\] " if alert.is_simulated else ""
        strategy_emoji = self._get_strategy_emoji(alert.strategy)
        strategy_label = self._get_strategy_label(alert.strategy)

        # Calculate USD value
        usd_value = alert.amount_sol * alert.sol_price_usd if alert.sol_price_usd else 0

        message = f"""
{sim_tag}☀️ *SOLANA ENTRY* \\| {strategy_emoji} {strategy_label}

🪙 *{self._escape_markdown(alert.token_symbol)}*

💰 Entry: ${self._escape_markdown(f"{alert.entry_price:.8f}")}
📦 Amount: {self._escape_markdown(f"{alert.amount_sol:.4f}")} SOL \\(${self._escape_markdown(f"{usd_value:.2f}")}\\)
🎯 Tokens: {self._escape_markdown(f"{alert.token_amount:,.2f}")}

🛑 Stop Loss: {self._escape_markdown(f"{alert.stop_loss_pct:.1f}%" if alert.stop_loss_pct else "N/A")}
✅ Take Profit: {self._escape_markdown(f"{alert.take_profit_pct:.1f}%" if alert.take_profit_pct else "N/A")}

🔗 `{self._escape_markdown(alert.token_mint[:8])}...{self._escape_markdown(alert.token_mint[-6:])}`
⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_exit_alert(self, alert: SolanaTradeAlert) -> str:
        """Format exit trade alert message with Solana branding"""
        sim_tag = "\\[SIM\\] " if alert.is_simulated else ""
        strategy_emoji = self._get_strategy_emoji(alert.strategy)
        strategy_label = self._get_strategy_label(alert.strategy)

        # Determine exit reason emoji
        reason_map = {
            'take_profit': '🎯 TP HIT',
            'stop_loss': '🛑 SL HIT',
            'time_exit': '⏱️ TIME EXIT',
            'manual_close': '👤 MANUAL',
            'signal': '📊 SIGNAL',
            'max_age': '⏰ MAX AGE',
        }
        exit_reason = reason_map.get(alert.action, self._escape_markdown(alert.reason or 'EXIT'))

        # PnL formatting
        pnl_sol = alert.pnl_sol or 0
        pnl_pct = alert.pnl_pct or 0
        pnl_usd = pnl_sol * alert.sol_price_usd if alert.sol_price_usd else 0
        pnl_emoji = "✅" if pnl_sol >= 0 else "❌"
        pnl_color = "🟩" if pnl_sol >= 0 else "🟥"

        message = f"""
{sim_tag}🌅 *SOLANA EXIT* \\| {exit_reason}

{strategy_emoji} {strategy_label} \\| 🪙 *{self._escape_markdown(alert.token_symbol)}*

💵 Entry: ${self._escape_markdown(f"{alert.entry_price:.8f}")}
💰 Exit: ${self._escape_markdown(f"{alert.exit_price:.8f}" if alert.exit_price else "N/A")}
📦 Amount: {self._escape_markdown(f"{alert.amount_sol:.4f}")} SOL

{pnl_color} *P&L:* {pnl_emoji} {self._escape_markdown(f"{pnl_sol:+.4f}")} SOL \\(${self._escape_markdown(f"{pnl_usd:+.2f}")}\\)
📈 *Return:* {self._escape_markdown(f"{pnl_pct:+.2f}")}%

🔗 `{self._escape_markdown(alert.token_mint[:8])}...{self._escape_markdown(alert.token_mint[-6:])}`
⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_stats_alert(self, stats: Dict[str, Any]) -> str:
        """Format daily stats summary with Solana branding"""
        message = f"""
☀️ *SOLANA DAILY STATS*

📈 Total Trades: {self._escape_markdown(str(stats.get('total_trades', 0)))}
✅ Winning: {self._escape_markdown(str(stats.get('winning_trades', 0)))}
❌ Losing: {self._escape_markdown(str(stats.get('losing_trades', 0)))}
🎯 Win Rate: {self._escape_markdown(str(stats.get('win_rate', '0%')))}

💰 Total P&L: {self._escape_markdown(str(stats.get('total_pnl', '0.0000 SOL')))}
📅 Daily P&L: {self._escape_markdown(str(stats.get('daily_pnl', '0.0000 SOL')))}

📊 Strategies:
  🪐 Jupiter: {self._escape_markdown(str(stats.get('jupiter_trades', 0)))} trades
  🚀 Pump\\.fun: {self._escape_markdown(str(stats.get('pumpfun_trades', 0)))} trades
  📊 Drift: {self._escape_markdown(str(stats.get('drift_trades', 0)))} trades

📍 Active Positions: {self._escape_markdown(str(stats.get('active_positions', 0)))}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    async def send_message(self, message: str, parse_mode: str = 'MarkdownV2') -> bool:
        """
        Send a message to Telegram

        Args:
            message: Message text
            parse_mode: Telegram parse mode

        Returns:
            Success status
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.debug("Solana Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Solana Telegram message: {e}")
            return False

    async def send_entry_alert(self, alert: SolanaTradeAlert) -> bool:
        """Send trade entry alert"""
        message = self._format_entry_alert(alert)
        return await self.send_message(message)

    async def send_exit_alert(self, alert: SolanaTradeAlert) -> bool:
        """Send trade exit alert"""
        message = self._format_exit_alert(alert)
        return await self.send_message(message)

    async def send_stats_summary(self, stats: Dict[str, Any]) -> bool:
        """Send daily stats summary"""
        message = self._format_stats_alert(stats)
        return await self.send_message(message)

    async def send_custom_alert(self, title: str, content: str) -> bool:
        """Send a custom alert message with Solana branding"""
        message = f"""
☀️ *{self._escape_markdown(title)}*

{self._escape_markdown(content)}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return await self.send_message(message.strip())

    async def send_error_alert(self, error_type: str, details: str) -> bool:
        """Send error alert"""
        message = f"""
⚠️ *SOLANA ERROR* \\| {self._escape_markdown(error_type)}

{self._escape_markdown(details)}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return await self.send_message(message.strip())

    async def send_risk_alert(self, risk_type: str, details: str) -> bool:
        """Send risk management alert"""
        message = f"""
🚨 *SOLANA RISK ALERT* \\| {self._escape_markdown(risk_type)}

{self._escape_markdown(details)}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return await self.send_message(message.strip())
