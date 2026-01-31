"""
Solana Strategies Module Configuration Manager
Database-backed configuration with .env integration for sensitive data

Uses the config_settings table (same as dashboard) for consistency.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Key Mapping
# Maps dashboard keys (solana_*) to internal config structure
# ============================================================================
CONFIG_KEY_MAPPING = {
    # General settings
    'enabled': ('solana_general', 'bool'),
    'capital': ('solana_general', 'float'),
    'position_size': ('solana_general', 'float'),
    'max_positions': ('solana_general', 'int'),
    'min_position': ('solana_general', 'float'),
    'rpc_timeout': ('solana_rpc', 'int'),
    'max_retries': ('solana_rpc', 'int'),
    'commitment': ('solana_rpc', 'string'),

    # Risk settings
    'stop_loss': ('solana_risk', 'float'),
    'take_profit': ('solana_risk', 'float'),
    'daily_loss_limit': ('solana_risk', 'float'),
    'priority_fee': ('solana_priority', 'int'),

    # Jupiter settings
    'jupiter_enabled': ('solana_jupiter', 'bool'),
    'jupiter_tier': ('solana_jupiter', 'string'),
    'jupiter_slippage': ('solana_jupiter', 'int'),
    'jupiter_auto_route': ('solana_jupiter', 'bool'),
    'jupiter_direct_only': ('solana_jupiter', 'bool'),
    'jupiter_tokens': ('solana_jupiter', 'string'),  # Comma-separated token list
    'jupiter_position_size': ('solana_jupiter', 'float'),  # Position size for Jupiter trades (0 = use default)

    # Drift settings
    'drift_enabled': ('solana_drift', 'bool'),
    'drift_leverage': ('solana_drift', 'int'),
    'drift_markets': ('solana_drift', 'string'),
    'drift_margin': ('solana_drift', 'string'),

    # Pump.fun settings
    'pumpfun_enabled': ('solana_pumpfun', 'bool'),
    'pumpfun_buy_amount': ('solana_pumpfun', 'float'),
    'pumpfun_min_liquidity': ('solana_pumpfun', 'float'),
    'pumpfun_min_liquidity_usd': ('solana_pumpfun', 'float'),
    'pumpfun_min_volume_24h': ('solana_pumpfun', 'float'),
    'pumpfun_max_age': ('solana_pumpfun', 'int'),
    'pumpfun_auto_sell': ('solana_pumpfun', 'int'),
    'pumpfun_jito': ('solana_pumpfun', 'bool'),
    'pumpfun_jito_tip': ('solana_pumpfun', 'float'),
    'pumpfun_stop_loss': ('solana_pumpfun', 'float'),
    'pumpfun_take_profit': ('solana_pumpfun', 'float'),
    'pumpfun_max_positions': ('solana_pumpfun', 'int'),  # Separate position limit for pump.fun
    'pumpfun_trailing_enabled': ('solana_pumpfun', 'bool'),  # Enable/disable trailing stops
    'pumpfun_tier0_sl': ('solana_pumpfun', 'float'),  # Tier 0 stop loss % (default -12)
    'pumpfun_partial_exit_pct': ('solana_pumpfun', 'float'),  # % to exit per tier (default 20)

    # Jupiter strategy-specific settings
    'jupiter_stop_loss': ('solana_jupiter', 'float'),
    'jupiter_take_profit': ('solana_jupiter', 'float'),
    'jupiter_auto_exit': ('solana_jupiter', 'int'),  # Time-based exit in seconds
}


class SolanaConfigManager:
    """
    Configuration manager for Solana Trading Module

    Reads from config_settings table (same as dashboard uses).
    Provides centralized access to all Solana configuration.
    """

    # Default values matching the dashboard defaults
    DEFAULTS = {
        # General
        'enabled': False,
        'capital': 10.0,
        'position_size': 1.0,
        'max_positions': 3,
        'min_position': 0.05,
        'rpc_timeout': 30,
        'max_retries': 3,
        'commitment': 'confirmed',

        # Risk
        'stop_loss': 10.0,
        'take_profit': 50.0,
        'daily_loss_limit': 5.0,
        'priority_fee': 10000,

        # Jupiter
        'jupiter_enabled': True,
        'jupiter_tier': 'lite',
        'jupiter_slippage': 50,
        'jupiter_auto_route': True,
        'jupiter_direct_only': False,
        'jupiter_tokens': 'BONK:DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263,JTO:jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL,WIF:EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm,PYTH:HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3,RAY:4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R,ORCA:orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
        'jupiter_position_size': 0.0,  # 0 = use default position_size

        # Drift
        'drift_enabled': False,
        'drift_leverage': 5,
        'drift_markets': 'SOL-PERP,BTC-PERP,ETH-PERP',
        'drift_margin': 'cross',

        # Pump.fun - IMPROVED trailing stop system for capturing 10x-100x gains
        'pumpfun_enabled': False,
        'pumpfun_buy_amount': 0.1,
        'pumpfun_min_liquidity': 10.0,  # Min liquidity in SOL
        'pumpfun_min_liquidity_usd': 1000.0,  # Min liquidity in USD
        'pumpfun_min_volume_24h': 5000.0,  # Min 24h volume in USD
        'pumpfun_max_age': 300,
        'pumpfun_auto_sell': 0,
        'pumpfun_jito': True,
        'pumpfun_jito_tip': 0.001,
        'pumpfun_stop_loss': 12.0,  # Tighter initial SL for Pump.fun (was 20%)
        'pumpfun_take_profit': 100.0,  # Pump.fun: higher TP for meme tokens
        'pumpfun_max_positions': 3,  # Separate position limit for pump.fun snipes
        'pumpfun_trailing_enabled': True,  # Enable trailing stops (recommended)
        'pumpfun_tier0_sl': 12.0,  # Tier 0 stop loss % (tighter protection)
        'pumpfun_partial_exit_pct': 20.0,  # Exit 20% per tier to let winners run

        # Jupiter strategy-specific (established tokens need tighter TP/SL)
        'jupiter_stop_loss': 5.0,  # Jupiter: tighter SL for established tokens
        'jupiter_take_profit': 10.0,  # Jupiter: realistic TP for established tokens
        'jupiter_auto_exit': 0,  # Time-based exit in seconds (0 = disabled)
    }

    def __init__(self, db_pool=None):
        """
        Initialize Solana configuration manager

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool
        self._config_cache: Dict[str, Any] = {}
        self._cache_loaded = False
        self._db_loaded_keys: set = set()  # Track keys explicitly loaded from database

        # Environment config (sensitive data from .env)
        self._env_config = self._load_environment_config()

        logger.info("SolanaConfigManager initialized")

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load sensitive configuration from .env"""
        env_config = {}

        sensitive_keys = [
            'SOLANA_MODULE_PRIVATE_KEY',
            'SOLANA_MODULE_WALLET',
            'SOLANA_PRIVATE_KEY',
            'SOLANA_WALLET',
            'SOLANA_RPC_URL',
            'SOLANA_RPC_URLS',
            'SOLANA_BACKUP_RPCS',
            'SOLANA_WS_URL',
            'HELIUS_API_KEY',
            'JUPITER_API_KEY',
            'JUPITER_API_URL',
            'DRIFT_API_KEY',
            'JITO_TIP_ACCOUNT',
            'JITO_BLOCK_ENGINE_URL',
            'DRY_RUN',
        ]

        for var in sensitive_keys:
            value = os.getenv(var)
            if value and value not in ('null', 'None', ''):
                env_config[var] = value

        return env_config

    async def initialize(self) -> None:
        """Initialize and load all configurations from database"""
        try:
            await self._load_all_configs()
            logger.info("✅ Solana configuration loaded from database")
        except Exception as e:
            logger.error(f"Failed to load Solana config from DB, using defaults: {e}")
            self._config_cache = self.DEFAULTS.copy()
            self._cache_loaded = True

    async def set_db_pool(self, db_pool) -> None:
        """Set database pool and reload configs"""
        self.db_pool = db_pool
        if db_pool:
            logger.info("Database pool set, reloading Solana configs...")
            await self._load_all_configs()

    async def _load_all_configs(self) -> None:
        """Load all Solana configuration from config_settings table"""
        if not self.db_pool:
            self._config_cache = self.DEFAULTS.copy()
            self._db_loaded_keys = set()  # No keys loaded from DB
            self._cache_loaded = True
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT config_type, key, value, value_type
                    FROM config_settings
                    WHERE config_type LIKE 'solana_%'
                    ORDER BY config_type, key
                """)

                # Start with defaults
                self._config_cache = self.DEFAULTS.copy()
                self._db_loaded_keys = set()  # Reset loaded keys tracking

                # Override with database values
                for row in rows:
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    # Convert value based on type
                    try:
                        if value_type == 'int':
                            self._config_cache[key] = int(value)
                        elif value_type == 'float':
                            self._config_cache[key] = float(value)
                        elif value_type == 'bool':
                            self._config_cache[key] = value.lower() in ('true', '1', 'yes')
                        else:
                            self._config_cache[key] = value
                        # Track that this key was explicitly loaded from DB
                        self._db_loaded_keys.add(key)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing config {key}: {e}")

                self._cache_loaded = True
                logger.debug(f"Loaded {len(rows)} Solana config values from database: {self._db_loaded_keys}")

        except Exception as e:
            logger.error(f"Database error loading Solana config: {e}")
            self._config_cache = self.DEFAULTS.copy()
            self._db_loaded_keys = set()
            self._cache_loaded = True

    async def reload(self) -> None:
        """Reload configuration from database"""
        await self._load_all_configs()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (e.g., 'position_size', 'jupiter_slippage')
            default: Default value if not found

        Returns:
            Configuration value
        """
        # Check environment config first for sensitive data
        env_key = key.upper()
        if env_key in self._env_config:
            return self._env_config[env_key]

        # Check cache
        if key in self._config_cache:
            return self._config_cache[key]

        # Return default from DEFAULTS or provided default
        return self.DEFAULTS.get(key, default)

    def get_env(self, key: str, default: str = '') -> str:
        """Get environment variable value"""
        return self._env_config.get(key, os.getenv(key, default))

    @property
    def is_enabled(self) -> bool:
        """Check if Solana module is enabled"""
        return self.get('enabled', False)

    @property
    def is_dry_run(self) -> bool:
        """Check if running in dry run mode"""
        dry_run = self.get_env('DRY_RUN', 'true')
        return dry_run.lower() in ('true', '1', 'yes')

    @property
    def capital_sol(self) -> float:
        """Get capital allocation in SOL"""
        return self.get('capital', 10.0)

    @property
    def position_size_sol(self) -> float:
        """Get default position size in SOL"""
        return self.get('position_size', 1.0)

    @property
    def max_positions(self) -> int:
        """Get maximum concurrent positions"""
        return self.get('max_positions', 3)

    @property
    def stop_loss_pct(self) -> float:
        """Get stop loss percentage"""
        return self.get('stop_loss', 10.0)

    @property
    def take_profit_pct(self) -> float:
        """Get take profit percentage"""
        return self.get('take_profit', 50.0)

    @property
    def daily_loss_limit_sol(self) -> float:
        """Get daily loss limit in SOL"""
        return self.get('daily_loss_limit', 5.0)

    @property
    def jupiter_enabled(self) -> bool:
        """Check if Jupiter strategy is enabled"""
        return self.get('jupiter_enabled', True)

    @property
    def jupiter_slippage_bps(self) -> int:
        """Get Jupiter slippage in basis points"""
        return self.get('jupiter_slippage', 50)

    @property
    def jupiter_tokens(self) -> List[tuple]:
        """
        Get Jupiter tokens to scan as list of (symbol, mint_address) tuples.

        Format in DB: "BONK:DezXAZ...,JTO:jtojto..."
        Returns: [('BONK', 'DezXAZ...'), ('JTO', 'jtojto...')]
        """
        tokens_str = self.get('jupiter_tokens', '')
        if not tokens_str:
            return []

        tokens = []
        for token_pair in tokens_str.split(','):
            token_pair = token_pair.strip()
            if ':' in token_pair:
                parts = token_pair.split(':', 1)
                symbol = parts[0].strip()
                mint = parts[1].strip()
                if symbol and mint:
                    tokens.append((symbol, mint))
        return tokens

    @property
    def jupiter_position_size_sol(self) -> float:
        """Get Jupiter-specific position size in SOL (0 = use default position_size)"""
        size = self.get('jupiter_position_size', 0.0)
        # If 0, fall back to general position size
        if size <= 0:
            return self.position_size_sol
        return size

    @property
    def drift_enabled(self) -> bool:
        """Check if Drift strategy is enabled"""
        return self.get('drift_enabled', False)

    @property
    def drift_leverage(self) -> int:
        """Get Drift default leverage"""
        return self.get('drift_leverage', 5)

    @property
    def drift_markets(self) -> List[str]:
        """Get Drift markets list"""
        markets_str = self.get('drift_markets', 'SOL-PERP,BTC-PERP,ETH-PERP')
        return [m.strip() for m in markets_str.split(',') if m.strip()]

    @property
    def pumpfun_enabled(self) -> bool:
        """Check if Pump.fun strategy is enabled"""
        return self.get('pumpfun_enabled', False)

    @property
    def pumpfun_buy_amount_sol(self) -> float:
        """Get Pump.fun buy amount in SOL"""
        return self.get('pumpfun_buy_amount', 0.1)

    @property
    def pumpfun_stop_loss_pct(self) -> float:
        """Get Pump.fun strategy-specific stop loss percentage"""
        return self.get('pumpfun_stop_loss', 20.0)

    @property
    def pumpfun_take_profit_pct(self) -> float:
        """Get Pump.fun strategy-specific take profit percentage"""
        return self.get('pumpfun_take_profit', 100.0)

    @property
    def pumpfun_max_positions(self) -> int:
        """Get Pump.fun strategy-specific max positions limit"""
        return self.get('pumpfun_max_positions', 3)

    @property
    def pumpfun_trailing_enabled(self) -> bool:
        """Check if Pump.fun trailing stops are enabled"""
        return self.get('pumpfun_trailing_enabled', True)

    @property
    def pumpfun_tier0_sl(self) -> float:
        """Get Pump.fun Tier 0 stop loss percentage (initial protection)"""
        return self.get('pumpfun_tier0_sl', 12.0)

    @property
    def pumpfun_partial_exit_pct(self) -> float:
        """Get Pump.fun partial exit percentage per tier"""
        return self.get('pumpfun_partial_exit_pct', 20.0)

    @property
    def jupiter_stop_loss_pct(self) -> float:
        """Get Jupiter strategy-specific stop loss percentage.
        Falls back to global stop_loss if Jupiter-specific not set."""
        # Check if Jupiter-specific value was explicitly set in database
        if 'jupiter_stop_loss' in self._db_loaded_keys:
            return self._config_cache['jupiter_stop_loss']
        # Fall back to global stop_loss setting (which may also be from DB or defaults)
        return self.get('stop_loss', 5.0)

    @property
    def jupiter_take_profit_pct(self) -> float:
        """Get Jupiter strategy-specific take profit percentage.
        Falls back to global take_profit if Jupiter-specific not set."""
        # Check if Jupiter-specific value was explicitly set in database
        if 'jupiter_take_profit' in self._db_loaded_keys:
            return self._config_cache['jupiter_take_profit']
        # Fall back to global take_profit setting (which may also be from DB or defaults)
        return self.get('take_profit', 10.0)

    @property
    def jupiter_auto_exit_seconds(self) -> int:
        """Get Jupiter time-based auto exit in seconds (0 = disabled)"""
        return self.get('jupiter_auto_exit', 0)

    @property
    def priority_fee_lamports(self) -> int:
        """Get priority fee in lamports"""
        return self.get('priority_fee', 10000)

    @property
    def rpc_url(self) -> str:
        """Get primary RPC URL"""
        return self.get_env('SOLANA_RPC_URL', '')

    @property
    def rpc_urls(self) -> List[str]:
        """Get all RPC URLs"""
        urls_str = self.get_env('SOLANA_RPC_URLS', '')
        if urls_str:
            return [url.strip() for url in urls_str.split(',') if url.strip()]
        primary = self.rpc_url
        return [primary] if primary else []

    @property
    def wallet_address(self) -> str:
        """Get wallet address (prefer SOLANA_MODULE_WALLET)"""
        return self.get_env('SOLANA_MODULE_WALLET') or self.get_env('SOLANA_WALLET', '')

    @property
    def private_key(self) -> str:
        """Get private key (prefer SOLANA_MODULE_PRIVATE_KEY)"""
        return self.get_env('SOLANA_MODULE_PRIVATE_KEY') or self.get_env('SOLANA_PRIVATE_KEY', '')

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategies"""
        strategies = []
        if self.jupiter_enabled:
            strategies.append('jupiter')
        if self.drift_enabled:
            strategies.append('drift')
        if self.pumpfun_enabled:
            strategies.append('pumpfun')
        return strategies

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        config = self._config_cache.copy()

        # Add computed properties
        config['enabled_strategies'] = self.get_enabled_strategies()
        config['is_dry_run'] = self.is_dry_run

        # Add environment config indicators (without exposing secrets)
        config['_has_solana_wallet'] = bool(self.wallet_address)
        config['_has_solana_private_key'] = bool(self.private_key)
        config['_has_solana_rpc'] = bool(self.rpc_url)
        config['_has_jupiter_api'] = bool(self.get_env('JUPITER_API_KEY'))
        config['_has_helius_api'] = bool(self.get_env('HELIUS_API_KEY'))

        return config

    async def save_config(self, key: str, value: Any, user_id: str = None) -> bool:
        """
        Save a single configuration value to database

        Args:
            key: Configuration key
            value: Value to save
            user_id: User making the change

        Returns:
            bool: True if saved successfully
        """
        if not self.db_pool:
            logger.warning("No database pool, cannot save config")
            return False

        try:
            # Determine config type from key
            config_type = CONFIG_KEY_MAPPING.get(key, ('solana_general', 'string'))[0]
            value_type = CONFIG_KEY_MAPPING.get(key, ('solana_general', 'string'))[1]

            # Convert value to string
            if isinstance(value, bool):
                value_str = str(value).lower()
                value_type = 'bool'
            elif isinstance(value, int):
                value_str = str(value)
                value_type = 'int'
            elif isinstance(value, float):
                value_str = str(value)
                value_type = 'float'
            else:
                value_str = str(value)
                value_type = 'string'

            async with self.db_pool.acquire() as conn:
                # Get old value for history
                old_row = await conn.fetchrow("""
                    SELECT value FROM config_settings
                    WHERE config_type = $1 AND key = $2
                """, config_type, key)
                old_value = old_row['value'] if old_row else None

                # Upsert the value
                await conn.execute("""
                    INSERT INTO config_settings (config_type, key, value, value_type, updated_by)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (config_type, key) DO UPDATE
                    SET value = $3, value_type = $4, updated_by = $5, updated_at = NOW()
                """, config_type, key, value_str, value_type, user_id)

                # Log to history if changed
                if old_value != value_str:
                    await conn.execute("""
                        INSERT INTO config_history (config_type, key, old_value, new_value, change_source, changed_by)
                        VALUES ($1, $2, $3, $4, 'config_manager', $5)
                    """, config_type, key, old_value, value_str, user_id)

            # Update cache
            self._config_cache[key] = value

            logger.info(f"✅ Saved Solana config: {key} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error saving Solana config {key}: {e}", exc_info=True)
            return False


# Global instance for convenience
_config_manager: Optional[SolanaConfigManager] = None


def get_solana_config() -> SolanaConfigManager:
    """Get the global Solana config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SolanaConfigManager()
    return _config_manager


async def init_solana_config(db_pool) -> SolanaConfigManager:
    """Initialize and return the global Solana config manager"""
    global _config_manager
    _config_manager = SolanaConfigManager(db_pool)
    await _config_manager.initialize()
    return _config_manager
