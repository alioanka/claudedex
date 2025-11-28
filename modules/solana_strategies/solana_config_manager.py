"""
Solana Strategies Module Configuration Manager
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


class SolanaConfigType(Enum):
    """Solana configuration types"""
    GENERAL = "solana_general"
    TRADING = "solana_trading"
    RISK = "solana_risk"
    STRATEGIES = "solana_strategies"
    PUMPFUN = "solana_pumpfun"
    JUPITER = "solana_jupiter"
    DRIFT = "solana_drift"
    RAYDIUM = "solana_raydium"


class SolanaGeneralConfig(BaseModel):
    """General Solana strategies configuration"""
    enabled: bool = False
    capital_allocation: float = 400.0
    max_positions: int = 8
    rpc_url: str = ""  # Will be loaded from .env
    use_jito: bool = True
    priority_fee_lamports: int = 10000
    max_compute_units: int = 200000


class SolanaRiskConfig(BaseModel):
    """Solana risk management configuration"""
    max_position_size_usd: float = 80.0
    risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.20
    max_daily_loss_usd: float = 100.0
    max_slippage_bps: int = 100
    min_liquidity_usd: float = 10000
    max_price_impact_pct: float = 0.05


class SolanaTradingConfig(BaseModel):
    """Solana trading configuration"""
    min_opportunity_score: float = 0.60
    use_jupiter_routing: bool = True
    jupiter_slippage_bps: int = 50
    max_transaction_retries: int = 3
    transaction_timeout_seconds: int = 60
    use_versioned_transactions: bool = True
    compute_unit_price_micro_lamports: int = 1


class SolanaPumpFunConfig(BaseModel):
    """Pump.fun strategy configuration"""
    enabled: bool = True
    monitor_new_tokens: bool = True
    min_initial_liquidity: float = 5000
    max_buy_amount_sol: float = 0.5
    min_holder_count: int = 50
    blacklist_suspicious: bool = True
    auto_sell_on_profit_pct: float = 0.30
    auto_sell_on_loss_pct: float = 0.15
    max_hold_time_minutes: int = 30


class SolanaJupiterConfig(BaseModel):
    """Jupiter aggregator configuration"""
    enabled: bool = True
    use_v6_api: bool = True
    max_accounts: int = 64
    slippage_bps: int = 50
    only_direct_routes: bool = False
    as_legacy_transaction: bool = False


class SolanaDriftConfig(BaseModel):
    """Drift Protocol configuration"""
    enabled: bool = False
    use_perpetuals: bool = True
    max_leverage: int = 5
    markets: List[str] = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]
    funding_rate_threshold: float = 0.01
    auto_compound_rewards: bool = True


class SolanaRaydiumConfig(BaseModel):
    """Raydium DEX configuration"""
    enabled: bool = True
    use_cpmm: bool = True  # Constant product market maker
    use_clmm: bool = True  # Concentrated liquidity
    min_pool_tvl: float = 50000
    max_pool_age_days: int = 30


class SolanaConfigManager:
    """
    Configuration manager for Solana Strategies Module

    Follows the same pattern as DEX config_manager.py:
    - Database-backed storage
    - .env integration for sensitive data
    - Pydantic validation
    - Hot-reload support
    """

    def __init__(self, db_pool=None):
        """
        Initialize Solana configuration manager

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool

        # Configuration models
        self.config_models = {
            SolanaConfigType.GENERAL: SolanaGeneralConfig,
            SolanaConfigType.RISK: SolanaRiskConfig,
            SolanaConfigType.TRADING: SolanaTradingConfig,
            SolanaConfigType.PUMPFUN: SolanaPumpFunConfig,
            SolanaConfigType.JUPITER: SolanaJupiterConfig,
            SolanaConfigType.DRIFT: SolanaDriftConfig,
            SolanaConfigType.RAYDIUM: SolanaRaydiumConfig,
        }

        # Loaded configs
        self.configs: Dict[SolanaConfigType, BaseModel] = {}

        # Environment config (sensitive data from .env)
        self._env_config = self._load_environment_config()

        logger.info("SolanaConfigManager initialized")

    def _load_environment_config(self) -> Dict[str, Any]:
        """
        Load sensitive configuration from .env

        Returns:
            Dict: Environment configuration
        """
        env_config = {}

        # Solana Module uses dedicated wallet (separate from DEX Module's Solana wallet)
        # SOLANA_PRIVATE_KEY/SOLANA_WALLET = DEX Module (Solana chain trading)
        # SOLANA_MODULE_PRIVATE_KEY/SOLANA_MODULE_WALLET = Solana Module (Jupiter/Drift/Pump.fun)
        sensitive_keys = [
            'SOLANA_MODULE_PRIVATE_KEY',
            'SOLANA_MODULE_WALLET',
            'SOLANA_RPC_URL',
            'SOLANA_RPC_URLS',
            'SOLANA_BACKUP_RPCS',
            'SOLANA_WS_URL',
            'HELIUS_API_KEY',
            'JUPITER_API_KEY',
            'DRIFT_API_KEY',
            'JITO_TIP_ACCOUNT',
            'JITO_BLOCK_ENGINE_URL',
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
            logger.info("✅ Solana configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Solana config: {e}", exc_info=True)
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
            logger.info("Database pool set, reloading Solana configs...")
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

    async def _load_config_from_db(self, config_type: SolanaConfigType) -> Optional[Dict]:
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
                WHERE module_name = 'solana_strategies'
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

    async def save_config(self, config_type: SolanaConfigType, config_data: Dict[str, Any]) -> bool:
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
                VALUES ('solana_strategies', $1, $2, $3)
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
            key: Configuration key (can include category prefix like "risk.max_position_size_usd")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # Check environment config first for sensitive data
        if key.upper() in self._env_config:
            return self._env_config[key.upper()]

        # Parse key (e.g., "risk.max_position_size_usd" or just "max_position_size_usd")
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

        # Add environment config indicators (without exposing secrets)
        # Solana Module uses dedicated wallet (SOLANA_MODULE_*)
        all_config['_has_solana_wallet'] = bool(os.getenv('SOLANA_MODULE_WALLET'))
        all_config['_has_solana_private_key'] = bool(os.getenv('SOLANA_MODULE_PRIVATE_KEY'))
        all_config['_has_solana_rpc'] = bool(os.getenv('SOLANA_RPC_URL'))

        return all_config
