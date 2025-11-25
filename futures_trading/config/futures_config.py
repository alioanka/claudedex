"""
Futures Trading Configuration
Configuration settings for futures trading module
"""

import os
from typing import Dict, List


class FuturesConfig:
    """Configuration for futures trading"""

    def __init__(self):
        # Exchange settings
        self.exchange = os.getenv('FUTURES_EXCHANGE', 'binance')
        self.testnet = os.getenv('FUTURES_TESTNET', 'true').lower() == 'true'

        # Trading parameters
        self.leverage = int(os.getenv('FUTURES_LEVERAGE', '10'))
        self.max_positions = int(os.getenv('FUTURES_MAX_POSITIONS', '5'))
        self.position_size_usd = float(os.getenv('FUTURES_POSITION_SIZE_USD', '100'))

        # Risk management
        self.stop_loss_pct = float(os.getenv('FUTURES_STOP_LOSS_PCT', '-5.0'))
        self.take_profit_pct = float(os.getenv('FUTURES_TAKE_PROFIT_PCT', '10.0'))
        self.max_daily_loss_usd = float(os.getenv('FUTURES_MAX_DAILY_LOSS_USD', '500'))

        # Symbols to trade
        default_symbols = 'BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT'
        symbols_str = os.getenv('FUTURES_SYMBOLS', default_symbols)
        self.symbols: List[str] = [s.strip() for s in symbols_str.split(',')]

        # API credentials
        if self.exchange == 'binance':
            self.api_key = os.getenv('BINANCE_API_KEY', '')
            self.api_secret = os.getenv('BINANCE_API_SECRET', '')
        elif self.exchange == 'bybit':
            self.api_key = os.getenv('BYBIT_API_KEY', '')
            self.api_secret = os.getenv('BYBIT_API_SECRET', '')
        else:
            self.api_key = ''
            self.api_secret = ''

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'exchange': self.exchange,
            'testnet': self.testnet,
            'leverage': self.leverage,
            'max_positions': self.max_positions,
            'position_size_usd': self.position_size_usd,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_loss_usd': self.max_daily_loss_usd,
            'symbols': self.symbols,
            'has_credentials': bool(self.api_key and self.api_secret)
        }

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []

        if not self.api_key or not self.api_secret:
            errors.append(f"{self.exchange.upper()}_API_KEY and {self.exchange.upper()}_API_SECRET required")

        if self.leverage < 1 or self.leverage > 125:
            errors.append("Leverage must be between 1 and 125")

        if self.max_positions < 1:
            errors.append("max_positions must be at least 1")

        if self.position_size_usd <= 0:
            errors.append("position_size_usd must be positive")

        if not self.symbols:
            errors.append("At least one symbol required")

        return errors
