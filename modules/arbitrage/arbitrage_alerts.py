"""
Telegram Alerts for Arbitrage Trading Module
Sends real-time notifications for arbitrage trades, opportunities, and errors
Distinct styling per chain: ETH (blue), ARB (orange), Base (blue), Solana (purple), Triangular (gold)
"""

import os
import logging
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ArbitrageChain(Enum):
    """Supported arbitrage chains"""
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    BASE = "base"
    SOLANA = "solana"
    TRIANGULAR = "triangular"


@dataclass
class ArbitrageTradeAlert:
    """Trade alert data structure for arbitrage trades"""
    chain: ArbitrageChain
    token_symbol: str
    buy_dex: str
    sell_dex: str
    amount: float  # Amount in native token (ETH/SOL)
    amount_usd: float
    profit_pct: float
    profit_amount: float  # Profit in native token
    profit_usd: float
    tx_hash: Optional[str] = None
    is_simulated: bool = True
    gas_cost_usd: Optional[float] = None
    flash_loan_fee: Optional[float] = None
    intermediate_token: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ArbitrageOpportunityAlert:
    """Alert for detected arbitrage opportunity"""
    chain: ArbitrageChain
    token_symbol: str
    buy_dex: str
    sell_dex: str
    raw_spread: float
    net_spread: float
    amount: float
    amount_usd: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ArbitrageErrorAlert:
    """Alert for arbitrage errors"""
    chain: ArbitrageChain
    error_type: str
    details: str
    token_symbol: Optional[str] = None
    tx_hash: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ArbitrageTelegramAlerts:
    """
    Telegram alerts manager for Arbitrage trading
    Uses distinct chain-themed styling to differentiate from other modules

    Chain Themes:
    - Ethereum: 💎 Blue diamond theme
    - Arbitrum: 🔶 Orange theme
    - Base: 🔵 Coinbase blue theme
    - Solana: ☀️ Purple/teal gradient theme
    - Triangular: 🔺 Gold triangle theme
    """

    # Chain-specific emojis and colors
    CHAIN_CONFIG = {
        ArbitrageChain.ETHEREUM: {
            'icon': '💎',
            'name': 'ETHEREUM',
            'color': '🔷',
            'entry_icon': '💎',
            'success_icon': '✨',
            'native_token': 'ETH',
        },
        ArbitrageChain.ARBITRUM: {
            'icon': '🔶',
            'name': 'ARBITRUM',
            'color': '🟠',
            'entry_icon': '⚡',
            'success_icon': '🔥',
            'native_token': 'ETH',
        },
        ArbitrageChain.BASE: {
            'icon': '🔵',
            'name': 'BASE',
            'color': '🔹',
            'entry_icon': '🅱️',
            'success_icon': '💫',
            'native_token': 'ETH',
        },
        ArbitrageChain.SOLANA: {
            'icon': '☀️',
            'name': 'SOLANA',
            'color': '🟣',
            'entry_icon': '🌊',
            'success_icon': '🌟',
            'native_token': 'SOL',
        },
        ArbitrageChain.TRIANGULAR: {
            'icon': '🔺',
            'name': 'TRIANGULAR',
            'color': '🟡',
            'entry_icon': '📐',
            'success_icon': '🏆',
            'native_token': 'ETH',
        },
    }

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Telegram alerts for Arbitrage module

        Args:
            bot_token: Telegram bot token (from secrets/env if not provided)
            chat_id: Telegram chat ID (from secrets/env if not provided)
            enabled: Whether alerts are enabled
        """
        # Try secrets manager first (reads from database), then fall back to env
        if bot_token:
            self.bot_token = bot_token
        else:
            try:
                from security.secrets_manager import secrets
                self.bot_token = secrets.get('TELEGRAM_BOT_TOKEN', log_access=False) or os.getenv('TELEGRAM_BOT_TOKEN')
            except Exception:
                self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

        if chat_id:
            self.chat_id = chat_id
        else:
            try:
                from security.secrets_manager import secrets
                self.chat_id = secrets.get('TELEGRAM_CHAT_ID', log_access=False) or os.getenv('TELEGRAM_CHAT_ID')
            except Exception:
                self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)

        if not self.enabled:
            logger.warning("Arbitrage Telegram alerts disabled - missing bot token or chat ID")
        else:
            logger.info("✅ Arbitrage Telegram alerts initialized")

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = str(text).replace(char, f'\\{char}')
        return text

    def _get_chain_config(self, chain: ArbitrageChain) -> Dict[str, str]:
        """Get chain-specific configuration"""
        return self.CHAIN_CONFIG.get(chain, self.CHAIN_CONFIG[ArbitrageChain.ETHEREUM])

    def _format_trade_alert(self, alert: ArbitrageTradeAlert) -> str:
        """Format successful arbitrage trade alert"""
        sim_tag = "\\[SIM\\] " if alert.is_simulated else ""
        config = self._get_chain_config(alert.chain)

        # Profit formatting
        pnl_emoji = "✅" if alert.profit_usd >= 0 else "❌"
        pnl_color = "🟩" if alert.profit_usd >= 0 else "🟥"

        # Gas cost info
        gas_info = ""
        if alert.gas_cost_usd is not None:
            gas_info = f"\n⛽ Gas: ${self._escape_markdown(f'{alert.gas_cost_usd:.2f}')}"

        # Flash loan fee info
        flash_info = ""
        if alert.flash_loan_fee is not None:
            flash_info = f"\n💳 Flash Fee: ${self._escape_markdown(f'{alert.flash_loan_fee:.2f}')}"

        # TX hash
        tx_info = ""
        if alert.tx_hash and alert.tx_hash != "DRY_RUN":
            short_hash = f"{alert.tx_hash[:10]}...{alert.tx_hash[-6:]}"
            tx_info = f"\n🔗 `{self._escape_markdown(short_hash)}`"

        message = f"""
{sim_tag}{config['success_icon']} *{config['name']} ARB TRADE* {config['icon']}

🪙 *{self._escape_markdown(alert.token_symbol)}*
📊 {self._escape_markdown(alert.buy_dex)} → {self._escape_markdown(alert.sell_dex)}

💰 Amount: {self._escape_markdown(f"{alert.amount:.4f}")} {config['native_token']} \\(${self._escape_markdown(f"{alert.amount_usd:.2f}")}\\)

{pnl_color} *Profit:* {pnl_emoji} {self._escape_markdown(f"{alert.profit_amount:+.6f}")} {config['native_token']}
💵 *USD P&L:* ${self._escape_markdown(f"{alert.profit_usd:+.2f}")} \\({self._escape_markdown(f"{alert.profit_pct:+.2f}")}%\\){gas_info}{flash_info}{tx_info}

⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_opportunity_alert(self, alert: ArbitrageOpportunityAlert) -> str:
        """Format arbitrage opportunity detection alert"""
        config = self._get_chain_config(alert.chain)

        message = f"""
🚨 *{config['name']} OPPORTUNITY* {config['icon']}

🪙 *{self._escape_markdown(alert.token_symbol)}*
📊 Buy: {self._escape_markdown(alert.buy_dex)} → Sell: {self._escape_markdown(alert.sell_dex)}

📈 Raw Spread: {self._escape_markdown(f"{alert.raw_spread:.2%}")}
📉 Net Spread: {self._escape_markdown(f"{alert.net_spread:.2%}")}
💰 Amount: {self._escape_markdown(f"{alert.amount:.4f}")} {config['native_token']} \\(${self._escape_markdown(f"{alert.amount_usd:.2f}")}\\)

⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_error_alert(self, alert: ArbitrageErrorAlert) -> str:
        """Format arbitrage error alert"""
        config = self._get_chain_config(alert.chain)

        token_info = ""
        if alert.token_symbol:
            token_info = f"\n🪙 Token: {self._escape_markdown(alert.token_symbol)}"

        tx_info = ""
        if alert.tx_hash:
            short_hash = f"{alert.tx_hash[:10]}...{alert.tx_hash[-6:]}"
            tx_info = f"\n🔗 `{self._escape_markdown(short_hash)}`"

        message = f"""
❌ *{config['name']} ARB ERROR* {config['icon']}

⚠️ *{self._escape_markdown(alert.error_type)}*
{token_info}

📋 Details:
{self._escape_markdown(alert.details)}{tx_info}

⏰ {self._escape_markdown(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return message.strip()

    def _format_stats_alert(self, stats: Dict[str, Any], chain: ArbitrageChain) -> str:
        """Format daily stats summary for a chain"""
        config = self._get_chain_config(chain)

        message = f"""
{config['icon']} *{config['name']} ARB STATS*

📈 Opportunities: {self._escape_markdown(str(stats.get('opportunities_found', 0)))}
✅ Executed: {self._escape_markdown(str(stats.get('trades_executed', 0)))}
❌ Failed: {self._escape_markdown(str(stats.get('trades_failed', 0)))}

💰 Total Profit: {self._escape_markdown(str(stats.get('total_profit', '0.0000')))} {config['native_token']}
💵 USD Profit: ${self._escape_markdown(str(stats.get('total_profit_usd', '0.00')))}
📊 Win Rate: {self._escape_markdown(str(stats.get('win_rate', '0%')))}

⛽ Gas Spent: ${self._escape_markdown(str(stats.get('total_gas_usd', '0.00')))}
📉 Avg Spread: {self._escape_markdown(str(stats.get('avg_spread', '0.00%')))}

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
                        logger.debug("Arbitrage Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Arbitrage Telegram message: {e}")
            return False

    async def send_trade_alert(self, alert: ArbitrageTradeAlert) -> bool:
        """Send successful trade alert"""
        message = self._format_trade_alert(alert)
        return await self.send_message(message)

    async def send_opportunity_alert(self, alert: ArbitrageOpportunityAlert) -> bool:
        """Send opportunity detection alert"""
        message = self._format_opportunity_alert(alert)
        return await self.send_message(message)

    async def send_error_alert(self, alert: ArbitrageErrorAlert) -> bool:
        """Send error alert"""
        message = self._format_error_alert(alert)
        return await self.send_message(message)

    async def send_stats_summary(self, stats: Dict[str, Any], chain: ArbitrageChain) -> bool:
        """Send daily stats summary for a specific chain"""
        message = self._format_stats_alert(stats, chain)
        return await self.send_message(message)

    async def send_custom_alert(self, chain: ArbitrageChain, title: str, content: str) -> bool:
        """Send a custom alert message with chain branding"""
        config = self._get_chain_config(chain)
        message = f"""
{config['icon']} *{self._escape_markdown(title)}*

{self._escape_markdown(content)}

⏰ {self._escape_markdown(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}
"""
        return await self.send_message(message.strip())

    # Convenience methods for each chain
    async def send_eth_trade(self, **kwargs) -> bool:
        """Send Ethereum arbitrage trade alert"""
        alert = ArbitrageTradeAlert(chain=ArbitrageChain.ETHEREUM, **kwargs)
        return await self.send_trade_alert(alert)

    async def send_arb_trade(self, **kwargs) -> bool:
        """Send Arbitrum arbitrage trade alert"""
        alert = ArbitrageTradeAlert(chain=ArbitrageChain.ARBITRUM, **kwargs)
        return await self.send_trade_alert(alert)

    async def send_base_trade(self, **kwargs) -> bool:
        """Send Base arbitrage trade alert"""
        alert = ArbitrageTradeAlert(chain=ArbitrageChain.BASE, **kwargs)
        return await self.send_trade_alert(alert)

    async def send_sol_arb_trade(self, **kwargs) -> bool:
        """Send Solana arbitrage trade alert"""
        alert = ArbitrageTradeAlert(chain=ArbitrageChain.SOLANA, **kwargs)
        return await self.send_trade_alert(alert)

    async def send_triangular_trade(self, **kwargs) -> bool:
        """Send Triangular arbitrage trade alert"""
        alert = ArbitrageTradeAlert(chain=ArbitrageChain.TRIANGULAR, **kwargs)
        return await self.send_trade_alert(alert)

    async def send_eth_error(self, error_type: str, details: str, **kwargs) -> bool:
        """Send Ethereum arbitrage error alert"""
        alert = ArbitrageErrorAlert(chain=ArbitrageChain.ETHEREUM, error_type=error_type, details=details, **kwargs)
        return await self.send_error_alert(alert)

    async def send_arb_error(self, error_type: str, details: str, **kwargs) -> bool:
        """Send Arbitrum arbitrage error alert"""
        alert = ArbitrageErrorAlert(chain=ArbitrageChain.ARBITRUM, error_type=error_type, details=details, **kwargs)
        return await self.send_error_alert(alert)

    async def send_base_error(self, error_type: str, details: str, **kwargs) -> bool:
        """Send Base arbitrage error alert"""
        alert = ArbitrageErrorAlert(chain=ArbitrageChain.BASE, error_type=error_type, details=details, **kwargs)
        return await self.send_error_alert(alert)
