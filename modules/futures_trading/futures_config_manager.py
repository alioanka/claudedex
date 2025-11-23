"""
Futures Trading Module Configuration Manager
Database-backed configuration with .env integration for sensitive data
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class FuturesConfigType(Enum):
    """Futures configuration types"""
    GENERAL = "futures_general"
    TRADING = "futures_trading"
    RISK = "futures_risk"
    EXCHANGES = "futures_exchanges"
    STRATEGIES = "futures_strategies"
    POSITIONS = "futures_positions"
    LEVERAGE = "futures_leverage"


class FuturesGeneralConfig(BaseModel):
    """General futures trading configuration"""
    enabled: bool = False
    mode: str = "testnet"  # testnet or mainnet
    capital_allocation: float = 300.0
    max_positions: int = 5
    default_exchange: str = "binance"  # binance or bybit
    auto_compound: bool = False


class FuturesRiskConfig(BaseModel):
    """Futures risk management configuration"""
    max_leverage: int = 3
    max_position_size_usd: float = 60.0
    risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_daily_loss_usd: float = 50.0
    max_drawdown_pct: float = 0.15
    trailing_stop_enabled: bool = True
    trailing_stop_distance_pct: float = 0.02
    max_correlation: float = 0.7  # Max correlation between positions


class FuturesTradingConfig(BaseModel):
    """Futures trading configuration"""
    min_opportunity_score: float = 0.70
    max_slippage_bps: int = 50
    order_type: str = "market"  # market or limit
    use_stop_limit: bool = True
    partial_tp_enabled: bool = True
    tp_levels: List[float] = [0.02, 0.05, 0.08, 0.12]  # TP at +2%, +5%, +8%, +12%
    tp_quantities: List[float] = [0.25, 0.25, 0.25, 0.25]  # Close 25% each
    hedge_mode: bool = False  # Allow both LONG and SHORT simultaneously


class FuturesExchangesConfig(BaseModel):
    """Exchange-specific configuration"""
    # Binance
    binance_enabled: bool = True
    binance_testnet: bool = True
    binance_api_key: str = ""  # Will be loaded from .env
    binance_api_secret: str = ""  # Will be loaded from .env
    binance_max_leverage: int = 3

    # Bybit
    bybit_enabled: bool = False
    bybit_testnet: bool = True
    bybit_api_key: str = ""  # Will be loaded from .env
    bybit_api_secret: str = ""  # Will be loaded from .env
    bybit_max_leverage: int = 3


class FuturesStrategiesConfig(BaseModel):
    """Futures trading strategies configuration"""
    # Trend Following
    trend_following_enabled: bool = True
    trend_min_strength: float = 0.6
    trend_lookback_period: int = 20

    # Mean Reversion
    mean_reversion_enabled: bool = False
    mean_reversion_threshold: float = 2.0  # Standard deviations

    # Funding Rate Arbitrage
    funding_arbitrage_enabled: bool = False
    funding_rate_threshold: float = 0.01  # 1% annualized

    # Breakout Strategy
    breakout_enabled: bool = True
    breakout_volume_multiplier: float = 2.0


class FuturesConfigManager:
    """
    Configuration manager for Futures Trading Module

    Follows the same pattern as DEX config_manager.py:
    - Database-backed storage
    - .env integration for sensitive data
    - Pydantic validation
    - Hot-reload support
    """

    def __init__(self, db_pool=None):
        """
        Initialize Futures configuration manager

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool

        # Configuration models
        self.config_models = {
            FuturesConfigType.GENERAL: FuturesGeneralConfig,
            FuturesConfigType.RISK: FuturesRiskConfig,
            FuturesConfigType.TRADING: FuturesTradingConfig,
            FuturesConfigType.EXCHANGES: FuturesExchangesConfig,
            FuturesConfigType.STRATEGIES: FuturesStrategiesConfig,
        }

        # Loaded configs
        self.configs: Dict[FuturesConfigType, BaseModel] = {}

        # Environment config (sensitive data from .env)
        self._env_config = self._load_environment_config()

        logger.info("FuturesConfigManager initialized")

    def _load_environment_config(self) -> Dict[str, Any]:
        """
        Load sensitive configuration from .env

        Returns:
            Dict: Environment configuration
        """
        env_config = {}

        # Futures-specific environment variables
        sensitive_keys = [
            'BINANCE_FUTURES_API_KEY',
            'BINANCE_FUTURES_API_SECRET',
            'BINANCE_FUTURES_TESTNET_API_KEY',
            'BINANCE_FUTURES_TESTNET_API_SECRET',
            'BYBIT_FUTURES_API_KEY',
            'BYBIT_FUTURES_API_SECRET',
            'BYBIT_FUTURES_TESTNET_API_KEY',
            'BYBIT_FUTURES_TESTNET_API_SECRET',
        ]

        for var in sensitive_keys:
            value = os.getenv(var)
            if value and value not in ('null', 'None', ''):
                env_config[var] = value

        return env_config

    async def initialize(self) -> None:
        """Initialize and load all configurations"""
        try:
            await self._load_all_configs()
            logger.info("✅ Futures configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Futures config: {e}", exc_info=True)
            # Load defaults if database fails
            for config_type, model_class in self.config_models.items():
                self.configs[config_type] = model_class()

    async def set_db_pool(self, db_pool) -> None:
        """
        Set database pool and reload configs

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool
        if db_pool:
            logger.info("Database pool set, reloading Futures configs...")
            await self._load_all_configs()

    async def _load_all_configs(self) -> None:
        """Load all configuration types from database"""
        for config_type, model_class in self.config_models.items():
            try:
                config = await self._load_config_from_db(config_type)
                if config:
                    self.configs[config_type] = model_class(**config)
                else:
                    # Use defaults if not in database
                    self.configs[config_type] = model_class()
            except Exception as e:
                logger.error(f"Error loading {config_type.value}: {e}")
                self.configs[config_type] = model_class()

    async def _load_config_from_db(self, config_type: FuturesConfigType) -> Optional[Dict]:
        """
        Load configuration from database

        Args:
            config_type: Configuration type to load

        Returns:
            Optional[Dict]: Configuration data or None
        """
        if not self.db_pool:
            return None

        try:
            query = """
                SELECT config_data
                FROM module_configs
                WHERE module_name = 'futures_trading'
                AND config_type = $1
                ORDER BY updated_at DESC
                LIMIT 1
            """

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, config_type.value)
                if row:
                    return row['config_data']
                return None

        except Exception as e:
            logger.error(f"Database error loading {config_type.value}: {e}")
            return None

    async def save_config(self, config_type: FuturesConfigType, config_data: Dict[str, Any]) -> bool:
        """
        Save configuration to database

        Args:
            config_type: Configuration type
            config_data: Configuration data to save

        Returns:
            bool: True if saved successfully
        """
        if not self.db_pool:
            logger.warning("No database pool, cannot save config")
            return False

        try:
            # Validate with Pydantic model
            model_class = self.config_models[config_type]
            validated_config = model_class(**config_data)

            # Save to database
            query = """
                INSERT INTO module_configs (module_name, config_type, config_data, updated_at)
                VALUES ('futures_trading', $1, $2, $3)
                ON CONFLICT (module_name, config_type)
                DO UPDATE SET
                    config_data = EXCLUDED.config_data,
                    updated_at = EXCLUDED.updated_at
            """

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    config_type.value,
                    validated_config.dict(),
                    datetime.now()
                )

            # Update in-memory config
            self.configs[config_type] = validated_config

            logger.info(f"✅ Saved {config_type.value} configuration")
            return True

        except Exception as e:
            logger.error(f"Error saving {config_type.value}: {e}", exc_info=True)
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (can include category prefix like "risk.max_leverage")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # Check environment config first for sensitive data
        if key.upper() in self._env_config:
            return self._env_config[key.upper()]

        # Parse key (e.g., "risk.max_leverage" or just "max_leverage")
        parts = key.split('.', 1)

        if len(parts) == 2:
            category, field = parts
            # Find config type matching category
            for config_type, config_obj in self.configs.items():
                if category in config_type.value:
                    return getattr(config_obj, field, default)

        # Search all configs for the key
        for config_obj in self.configs.values():
            if hasattr(config_obj, key):
                return getattr(config_obj, key)

        return default

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration as a dictionary

        Returns:
            Dict: All configuration data
        """
        all_config = {}

        for config_type, config_obj in self.configs.items():
            all_config[config_type.value] = config_obj.dict()

        # Add environment config (without exposing secrets)
        all_config['_has_binance_api'] = bool(os.getenv('BINANCE_FUTURES_API_KEY'))
        all_config['_has_bybit_api'] = bool(os.getenv('BYBIT_FUTURES_API_KEY'))

        return all_config
