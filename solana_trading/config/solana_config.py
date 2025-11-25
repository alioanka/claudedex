"""
Solana Trading Configuration
Configuration settings for Solana trading module
"""

import os
from typing import Dict, List


class SolanaConfig:
    """Configuration for Solana trading"""

    def __init__(self):
        # Solana connection
        self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
        self.ws_url = os.getenv('SOLANA_WS_URL', 'wss://api.mainnet-beta.solana.com')
        self.commitment = os.getenv('SOLANA_COMMITMENT', 'confirmed')

        # Wallet - SOLANA MODULE uses separate wallet from DEX module
        self.private_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY', '')
        self.wallet_address = os.getenv('SOLANA_MODULE_WALLET', '')

        # Trading strategies
        default_strategies = 'jupiter,drift'
        strategies_str = os.getenv('SOLANA_STRATEGIES', default_strategies)
        self.strategies: List[str] = [s.strip() for s in strategies_str.split(',')]

        # Trading parameters
        self.max_positions = int(os.getenv('SOLANA_MAX_POSITIONS', '3'))
        self.position_size_sol = float(os.getenv('SOLANA_POSITION_SIZE_SOL', '1.0'))

        # Risk management
        self.stop_loss_pct = float(os.getenv('SOLANA_STOP_LOSS_PCT', '-10.0'))
        self.take_profit_pct = float(os.getenv('SOLANA_TAKE_PROFIT_PCT', '50.0'))
        self.max_daily_loss_sol = float(os.getenv('SOLANA_MAX_DAILY_LOSS_SOL', '5.0'))

        # Jupiter configuration
        self.jupiter_api_key = os.getenv('JUPITER_API_KEY', '')
        self.jupiter_tier = os.getenv('JUPITER_TIER', 'public')  # lite, public, ultra
        self.jupiter_slippage_bps = int(os.getenv('JUPITER_SLIPPAGE_BPS', '50'))  # 0.5%

        # Drift configuration
        self.drift_leverage = int(os.getenv('DRIFT_LEVERAGE', '5'))
        self.drift_markets = os.getenv('DRIFT_MARKETS', 'SOL-PERP,BTC-PERP,ETH-PERP').split(',')

        # Pump.fun configuration
        self.pumpfun_program_id = os.getenv('PUMPFUN_PROGRAM_ID', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P')
        self.pumpfun_min_liquidity = float(os.getenv('PUMPFUN_MIN_LIQUIDITY', '10'))  # SOL
        self.pumpfun_max_age_seconds = int(os.getenv('PUMPFUN_MAX_AGE_SECONDS', '300'))
        self.pumpfun_buy_amount_sol = float(os.getenv('PUMPFUN_BUY_AMOUNT_SOL', '0.1'))

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'rpc_url': self.rpc_url,
            'ws_url': self.ws_url,
            'commitment': self.commitment,
            'strategies': self.strategies,
            'max_positions': self.max_positions,
            'position_size_sol': self.position_size_sol,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_daily_loss_sol': self.max_daily_loss_sol,
            'jupiter': {
                'tier': self.jupiter_tier,
                'slippage_bps': self.jupiter_slippage_bps,
                'has_api_key': bool(self.jupiter_api_key)
            },
            'drift': {
                'leverage': self.drift_leverage,
                'markets': self.drift_markets
            },
            'pumpfun': {
                'min_liquidity': self.pumpfun_min_liquidity,
                'max_age_seconds': self.pumpfun_max_age_seconds,
                'buy_amount_sol': self.pumpfun_buy_amount_sol
            },
            'has_wallet': bool(self.private_key),
            'wallet_address': self.wallet_address
        }

    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []

        if not self.private_key:
            errors.append("SOLANA_MODULE_PRIVATE_KEY required")

        if not self.rpc_url:
            errors.append("SOLANA_RPC_URL required")

        if self.max_positions < 1:
            errors.append("max_positions must be at least 1")

        if self.position_size_sol <= 0:
            errors.append("position_size_sol must be positive")

        if not self.strategies:
            errors.append("At least one strategy required")

        # Validate strategy-specific configs
        if 'drift' in self.strategies:
            if self.drift_leverage < 1 or self.drift_leverage > 20:
                errors.append("drift_leverage must be between 1 and 20")

        if 'jupiter' in self.strategies:
            if self.jupiter_slippage_bps < 1 or self.jupiter_slippage_bps > 1000:
                errors.append("jupiter_slippage_bps must be between 1 and 1000")

        return errors
