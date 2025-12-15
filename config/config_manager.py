"""
Configuration Manager for DexScreener Trading Bot
Centralized configuration management with validation, hot-reloading, and environment handling
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Callable, Union

import os
import json
import yaml
import asyncio
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

import aiofiles
from pydantic import BaseModel, ValidationError, validator, Field
from pydantic.types import SecretStr
from jsonschema import validate, ValidationError as JsonValidationError

from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class ConfigType(Enum):
    """Configuration types"""
    GENERAL = "general"
    TRADING = "trading"
    STRATEGIES = "strategies"
    SECURITY = "security"
    DATABASE = "database"
    API = "api"
    MONITORING = "monitoring"
    ML_MODELS = "ml_models"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO = "portfolio"
    CHAIN = "chain"
    POSITION_MANAGEMENT = "position_management"
    VOLATILITY = "volatility"
    EXIT_STRATEGY = "exit_strategy"
    SOLANA = "solana"
    JUPITER = "jupiter"
    PERFORMANCE = "performance"
    LOGGING = "logging"
    FEATURE_FLAGS = "feature_flags"
    GAS_PRICE = "gas_price"
    TRADING_LIMITS = "trading_limits"
    BACKTESTING = "backtesting"
    NETWORK = "network"
    DEBUG = "debug"
    DASHBOARD = "dashboard"


class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    DEFAULT = "default"

@dataclass
class ConfigChange:
    """Configuration change record"""
    timestamp: datetime
    config_type: ConfigType
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    user: Optional[str]
    reason: Optional[str]

class GeneralConfig(BaseModel):
    ml_enabled: bool = False
    mode: str = "production"
    dry_run: bool = True

class PortfolioConfig(BaseModel):
    initial_balance: float = 400.0
    initial_balance_per_chain: float = 100.0
    max_position_size_usd: float = 10.0
    min_position_size_usd: float = 5.0
    max_position_size_pct: float = 0.10
    max_positions: int = 40
    max_positions_per_chain: int = 10
    max_concurrent_positions: int = 4

class RiskManagementConfig(BaseModel):
    max_risk_per_trade: float = 0.10
    max_portfolio_risk: float = 0.25
    daily_loss_limit_usd: float = 40.0
    daily_loss_limit_pct: float = 0.10
    risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.12
    take_profit_pct: float = 0.24
    position_cooldown_minutes: int = 30
    breaker_error_rate_max: int = 20
    breaker_slippage_realized_bps_max: int = 120
    breaker_max_consecutive_losses: int = 5
    breaker_max_drawdown_pct: int = 15
    breaker_max_daily_loss_pct: int = 10
    max_position_size_percent: int = 10
    max_daily_loss_percent: int = 10
    max_drawdown_percent: int = 25

class TradingConfig(BaseModel):
    max_slippage_bps: int = 50
    expected_slippage_bps: int = 10
    max_price_impact_bps: int = 100
    dex_fee_bps: int = 30
    min_opportunity_score: float = 0.25
    solana_min_opportunity_score: float = 0.20

    # Dynamic weights for opportunity scoring
    score_weight_volume: float = 0.30
    score_weight_liquidity: float = 0.35
    score_weight_price_change: float = 0.10
    score_weight_risk: float = 0.20
    score_weight_age: float = 0.05

class StrategiesConfig(BaseModel):
    """Trading strategies configuration"""
    # Momentum Strategy
    momentum_enabled: bool = True
    momentum_lookback_period: int = 20
    momentum_min_momentum_score: float = 0.5
    momentum_volume_threshold: float = 10000

    # Scalping Strategy
    scalping_enabled: bool = True
    scalping_profit_target: float = 0.02
    scalping_max_hold_time: int = 5  # minutes
    scalping_min_spread: float = 0.001

    # AI Strategy
    ai_enabled: bool = True  # Changed from False to True
    ai_ml_confidence_threshold: float = 0.65
    ai_min_pump_probability: float = 0.50
    ai_ensemble_min_models: int = 3
    ai_use_lstm: bool = True
    ai_use_xgboost: bool = True
    ai_use_lightgbm: bool = True

    # Strategy Selection
    strategy_selection_mode: str = "auto"  # auto, single, multi
    default_strategy: str = "momentum"  # Used when mode is "single"
    multi_strategy_enabled: bool = True  # Allow multiple strategies per opportunity

    def to_strategy_manager_dict(self) -> Dict[str, Any]:
        """
        Convert flat config to nested structure expected by StrategyManager.

        StrategyManager expects both flat enabled flags and nested strategy configs:
        {
            'momentum_enabled': True,
            'momentum': { 'lookback_period': 20, ... },
            'scalping_enabled': True,
            'scalping': { 'profit_target': 0.02, ... },
            ...
        }
        """
        return {
            # Momentum
            'momentum_enabled': self.momentum_enabled,
            'momentum': {
                'lookback_period': self.momentum_lookback_period,
                'min_momentum_score': self.momentum_min_momentum_score,
                'volume_threshold': self.momentum_volume_threshold,
            },
            # Scalping
            'scalping_enabled': self.scalping_enabled,
            'scalping': {
                'profit_target': self.scalping_profit_target,
                'max_hold_time': self.scalping_max_hold_time,
                'min_spread': self.scalping_min_spread,
            },
            # AI Strategy
            'ai_enabled': self.ai_enabled,
            'ai': {
                'ml_confidence_threshold': self.ai_ml_confidence_threshold,
                'min_pump_probability': self.ai_min_pump_probability,
                'ensemble_min_models': self.ai_ensemble_min_models,
                'use_lstm': self.ai_use_lstm,
                'use_xgboost': self.ai_use_xgboost,
                'use_lightgbm': self.ai_use_lightgbm,
            },
            # Strategy Selection
            'strategy_selection_mode': self.strategy_selection_mode,
            'default_strategy': self.default_strategy,
            'multi_strategy_enabled': self.multi_strategy_enabled,
        }

class ChainConfig(BaseModel):
    enabled_chains: str = "ethereum,bsc,base,solana,monad,pulsechain"
    default_chain: str = "ethereum"
    max_pairs_per_chain: int = 50
    discovery_interval_seconds: int = 300
    chain_id: int = 1
    # Chain enabled flags
    ethereum_enabled: bool = True
    bsc_enabled: bool = True
    base_enabled: bool = True
    arbitrum_enabled: bool = False  # Low activity
    polygon_enabled: bool = False
    solana_enabled: bool = True
    monad_enabled: bool = True       # Monad - supported by DexScreener Nov 2025
    pulsechain_enabled: bool = True
    fantom_enabled: bool = False
    cronos_enabled: bool = False
    avalanche_enabled: bool = False
    # Chain minimum liquidity thresholds (USD)
    ethereum_min_liquidity: int = 3000
    bsc_min_liquidity: int = 500
    base_min_liquidity: int = 2000
    arbitrum_min_liquidity: int = 3000
    polygon_min_liquidity: int = 500
    solana_min_liquidity: int = 2000
    monad_min_liquidity: int = 2000
    pulsechain_min_liquidity: int = 1000
    fantom_min_liquidity: int = 500
    cronos_min_liquidity: int = 500
    avalanche_min_liquidity: int = 1000

    def get_enabled_chains_list(self) -> list:
        """Get list of enabled chains from the enabled_chains string"""
        if not self.enabled_chains:
            return []
        return [c.strip().lower() for c in self.enabled_chains.split(',') if c.strip()]

    def is_chain_enabled(self, chain: str) -> bool:
        """Check if a specific chain is enabled"""
        chain_lower = chain.lower()
        # First check enabled_chains string
        if chain_lower in self.get_enabled_chains_list():
            # Then check individual flag if exists
            flag_name = f"{chain_lower}_enabled"
            if hasattr(self, flag_name):
                return getattr(self, flag_name)
            return True
        return False

    def get_chain_min_liquidity(self, chain: str) -> int:
        """Get minimum liquidity threshold for a chain"""
        chain_lower = chain.lower()
        attr_name = f"{chain_lower}_min_liquidity"
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        return 1000  # Default minimum liquidity

class PositionManagementConfig(BaseModel):
    default_stop_loss_percent: int = 12
    default_take_profit_percent: int = 24
    max_hold_time_minutes: int = 60
    trailing_stop_enabled: bool = True
    trailing_stop_percent: int = 6
    trailing_stop_activation: int = 10
    trailing_stop_distance: int = 5
    position_update_interval_seconds: int = 10
    enable_position_monitoring: bool = True
    use_ml_exits: bool = False

class VolatilityConfig(BaseModel):
    max_volatility_percent: int = 200
    min_volatility_percent: int = 10
    volatility_window_minutes: int = 60

class ExitStrategyConfig(BaseModel):
    exit_check_interval_seconds: int = 10

class SolanaConfig(BaseModel):
    solana_priority_fee: int = 5000
    solana_compute_unit_price: int = 1000
    solana_compute_unit_limit: int = 200000
    solana_min_liquidity: int = 2000
    solana_max_position_size_sol: int = 5
    solana_min_trade_size_sol: float = 0.1

class JupiterConfig(BaseModel):
    jupiter_url: str = "https://quote-api.jup.ag/v6"
    jupiter_max_slippage_bps: int = 50  # FIXED: Was 500 (5%), now 50 (0.5%)

class PerformanceConfig(BaseModel):
    max_workers: int = 4
    batch_size: int = 100
    cache_ttl: int = 300

class LoggingConfig(BaseModel):
    log_level: str = "INFO"
    log_file: str = "/app/logs/trading_bot.log"

class FeatureFlagsConfig(BaseModel):
    enable_experimental_features: bool = False
    use_ai_sentiment: bool = False
    use_whale_tracking: bool = True
    use_mempool_monitoring: bool = True

class GasPriceConfig(BaseModel):
    max_gas_price: int = 50  # FIXED: Was 500 (=$150-300/tx), now 50 Gwei
    priority_gas_multiplier: float = 1.2

class TradingLimitsConfig(BaseModel):
    min_balance_eth: float = 0.01
    min_trade_size_usd: int = 5
    max_trade_size_usd: int = 10

class MLModelsConfig(BaseModel):
    ml_retrain_interval_hours: int = 24
    ml_min_confidence: float = 0.7

class BacktestingConfig(BaseModel):
    backtest_start_date: str = "2024-01-01"
    backtest_end_date: str = "2024-12-31"
    backtest_initial_balance: float = 1.0

class NetworkConfig(BaseModel):
    http_timeout: int = 30
    ws_ping_interval: int = 30
    max_retries: int = 3

class DebugConfig(BaseModel):
    debug: bool = False
    testing: bool = False
    verbose: bool = False

class DashboardConfig(BaseModel):
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080

class SecurityConfig(BaseModel):
    """Security configuration schema"""
    private_key: Optional[str] = None
    solana_private_key: Optional[str] = None
    encryption_key: Optional[str] = None
    flashbots_signing_key: Optional[str] = None
    api_rate_limit: int = 1000  # requests per hour
    max_login_attempts: int = 5
    session_timeout: int = 3600  # seconds
    require_2fa: bool = True
    encryption_key_rotation_days: int = 30
    backup_encryption_enabled: bool = True
    audit_log_retention_days: int = 365
    real_time_alerts: bool = True
    hardware_wallet_required: bool = False
    multisig_threshold: int = 2
    
    @validator('api_rate_limit')
    def validate_rate_limit(cls, v):
        if v <= 0:
            raise ValueError('api_rate_limit must be positive')
        return v

class DatabaseConfig(BaseModel):
    """Database configuration schema"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = "bot_user"
    password: SecretStr = SecretStr("bot_password")
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo_sql: bool = False
    enable_query_cache: bool = True
    query_cache_size: int = 1000

class APIConfig(BaseModel):
    """API configuration schema"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    jwt_secret: SecretStr = SecretStr("ti4o1XMf9NWB35RzTFvE0nqrCHblvviJC-N878PWoPs")
    jwt_expiry: int = 3600
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

class MonitoringConfig(BaseModel):
    """Monitoring configuration schema"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"
    log_rotation_size: str = "100MB"
    log_retention_days: int = 30
    telegram_bot_token: Optional[SecretStr] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[SecretStr] = None
    email_smtp_server: Optional[str] = None
    email_username: Optional[str] = None
    email_password: Optional[SecretStr] = None

class ConfigManager:
    """
    Centralized configuration management system with:
    - Schema validation using Pydantic
    - Hot-reloading capabilities
    - Environment variable override
    - Secure secret management
    - Configuration change tracking
    - Multi-source configuration merging
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self.configs: Dict[ConfigType, BaseModel] = {}
        self.config_schemas: Dict[ConfigType, type] = {
            ConfigType.GENERAL: GeneralConfig,
            ConfigType.PORTFOLIO: PortfolioConfig,
            ConfigType.RISK_MANAGEMENT: RiskManagementConfig,
            ConfigType.TRADING: TradingConfig,
            ConfigType.STRATEGIES: StrategiesConfig,
            ConfigType.CHAIN: ChainConfig,
            ConfigType.POSITION_MANAGEMENT: PositionManagementConfig,
            ConfigType.VOLATILITY: VolatilityConfig,
            ConfigType.EXIT_STRATEGY: ExitStrategyConfig,
            ConfigType.SOLANA: SolanaConfig,
            ConfigType.JUPITER: JupiterConfig,
            ConfigType.PERFORMANCE: PerformanceConfig,
            ConfigType.LOGGING: LoggingConfig,
            ConfigType.FEATURE_FLAGS: FeatureFlagsConfig,
            ConfigType.GAS_PRICE: GasPriceConfig,
            ConfigType.TRADING_LIMITS: TradingLimitsConfig,
            ConfigType.ML_MODELS: MLModelsConfig,
            ConfigType.BACKTESTING: BacktestingConfig,
            ConfigType.NETWORK: NetworkConfig,
            ConfigType.DEBUG: DebugConfig,
            ConfigType.DASHBOARD: DashboardConfig,
            ConfigType.SECURITY: SecurityConfig,
            ConfigType.DATABASE: DatabaseConfig,
            ConfigType.API: APIConfig,
            ConfigType.MONITORING: MonitoringConfig,
        }
        self.encryption_manager = None
        self._raw_config = {}
        self._env_config = self._load_environment_config()
        self._raw_config.update(self._env_config)
    
        logger.info("ConfigManager initialized")

    async def initialize(self, encryption_key: Optional[str] = None) -> None:
        """Initialize configuration manager"""
        try:
            if encryption_key:
                encryption_config = {'encryption_key': encryption_key}
                self.encryption_manager = EncryptionManager(encryption_config)

            await self._load_all_configs()

            for config_type, config_obj in self.configs.items():
                config_key = config_type.value
                self._raw_config[config_key] = config_obj.dict()

            self._raw_config.update(self._env_config)

            logger.info("Configuration manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            raise

    async def set_db_pool(self, db_pool) -> None:
        """
        Set database pool and reload configs from database
        This should be called after database connection is established
        """
        self.db_pool = db_pool
        if db_pool:
            logger.info("Database pool set, reloading configs from database...")
            await self._load_all_configs()
            for config_type, config_obj in self.configs.items():
                config_key = config_type.value
                self._raw_config[config_key] = config_obj.dict()
            logger.info("✅ Configs reloaded from database")

    def _load_environment_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables
        This provides fallback values for .get() method
        """
        env_config = {}
        
        sensitive_keys = [
            'ENCRYPTION_KEY', 'DEX_MODULE_EVM_PRIVATE_KEY', 'DEX_MODULE_SOLANA_PRIVATE_KEY', 'SOLANA_MODULE_PRIVATE_KEY',
            'DATABASE_URL', 'REDIS_URL', 'TELEGRAM_BOT_TOKEN', 'DISCORD_WEBHOOK_URL'
        ]
        
        for var in sensitive_keys:
            value = os.getenv(var)
            if value and value not in ('null', 'None', ''):
                env_config[var] = value
        
        return env_config


    def get(self, key: str, default: Any = None) -> Any:
        """
        Dictionary-style get method for backward compatibility
        Supports both flat keys (e.g., 'PRIVATE_KEY') and nested keys (e.g., 'security.private_key')
        
        Args:
            key: Configuration key (flat or dotted notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            env_key = key.upper().replace('.', '_')
            env_value = os.getenv(env_key)
            if env_value is not None and env_value not in ('', 'null', 'None'):
                return env_value
            
            if hasattr(self, '_raw_config') and self._raw_config:
                if '.' in key:
                    parts = key.split('.')
                    value = self._raw_config
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    if value is not None:
                        return value
                elif key in self._raw_config:
                    return self._raw_config[key]
            
            key_lower = key.lower()
            
            for config_type in ConfigType:
                if config_type in self.configs:
                    config_obj = self.configs[config_type]
                    if hasattr(config_obj, key_lower):
                        return getattr(config_obj, key_lower)
            
            return default
            
        except Exception as e:
            logger.debug(f"Error getting config key '{key}': {e}")
            return default


    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Configuration key '{key}' not found")
        return result


    def __setitem__(self, key: str, value: Any):
        if not hasattr(self, '_raw_config'):
            self._raw_config = {}
        self._raw_config[key] = value


    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


    def keys(self) -> List[str]:
        keys_set = set()
        if hasattr(self, '_raw_config') and self._raw_config:
            keys_set.update(self._raw_config.keys())
        if hasattr(self, '_env_config') and self._env_config:
            keys_set.update(self._env_config.keys())
        return list(keys_set)

    async def _load_all_configs(self) -> None:
        """Load all configuration types"""
        for config_type in ConfigType:
            try:
                await self._load_config(config_type)
            except Exception as e:
                logger.error(f"Failed to load {config_type.value} config: {e}")
                await self._load_default_config(config_type)

    async def _load_config(self, config_type: ConfigType) -> None:
        """Load configuration from multiple sources"""
        config_data = {}
        
        schema_class = self.config_schemas.get(config_type)
        if schema_class:
            default_config = schema_class()
            config_data.update(default_config.dict())
        
        env_data = self._load_config_from_env(config_type)
        if env_data:
            config_data.update(env_data)
        
        db_data = await self._load_config_from_database(config_type)
        if db_data:
            config_data.update(db_data)
        
        if self.encryption_manager:
            config_data = self._decrypt_sensitive_values(config_data)
        
        if schema_class:
            try:
                validated_config = schema_class(**config_data)
                self.configs[config_type] = validated_config
                logger.info(f"Loaded {config_type.value} configuration")
            except ValidationError as e:
                logger.error(f"Validation error in {config_type.value} config: {e}")
                raise
        else:
            logger.warning(f"No schema defined for {config_type.value}")

    def _load_config_from_env(self, config_type: ConfigType) -> Dict:
        """Load configuration from environment variables"""
        env_data = {}
        
        schema_class = self.config_schemas.get(config_type)
        if not schema_class:
            return env_data

        for field_name in schema_class.__fields__:
            env_var = field_name.upper()
            value = os.getenv(env_var)
            if value is not None:
                env_data[field_name] = value

        return env_data

    async def _load_config_from_database(self, config_type: ConfigType) -> Optional[Dict]:
        """Load configuration from database"""
        if not self.db_pool:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT key, value, value_type
                    FROM config_settings
                    WHERE config_type = $1
                    AND is_editable = TRUE
                """, config_type.value)

                if not rows:
                    return None

                config_data = {}
                for row in rows:
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    if value_type == 'int':
                        config_data[key] = int(value)
                    elif value_type == 'float':
                        config_data[key] = float(value)
                    elif value_type == 'bool':
                        config_data[key] = value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == 'json':
                        import json
                        config_data[key] = json.loads(value)
                    else:
                        config_data[key] = value

                return config_data

        except Exception as e:
            logger.error(f"Failed to load {config_type.value} config from database: {e}")
            return None

    def _decrypt_sensitive_values(self, config_data: Dict) -> Dict:
        """Decrypt sensitive configuration values"""
        if not self.encryption_manager:
            return config_data
        
        sensitive_keys = ['password', 'secret', 'token', 'key']
        
        def decrypt_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and value.startswith('encrypted:'):
                            try:
                                data[key] = self.encryption_manager.decrypt_sensitive_data(value[10:])
                            except Exception as e:
                                logger.warning(f"Failed to decrypt {key}: {e}")
                    else:
                        data[key] = decrypt_recursive(value)
            elif isinstance(data, list):
                return [decrypt_recursive(item) for item in data]
            return data
        
        return decrypt_recursive(config_data.copy())

    def validate_environment(self) -> List[str]:
        """Validate required environment variables are set"""
        required_vars = [
            'ENCRYPTION_KEY',
            'DATABASE_URL',
            'REDIS_URL'
        ]
        
        # Module-specific secrets
        if os.getenv('DEX_MODULE_ENABLED', 'false').lower() == 'true':
            required_vars.extend(['DEX_MODULE_EVM_PRIVATE_KEY', 'DEX_MODULE_SOLANA_PRIVATE_KEY'])
        if os.getenv('FUTURES_MODULE_ENABLED', 'false').lower() == 'true':
            # Futures needs at least one set of keys
            if not (os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET')) and \
               not (os.getenv('BYBIT_API_KEY') and os.getenv('BYBIT_API_SECRET')):
                required_vars.append("BINANCE_OR_BYBIT_API_KEYS")
        if os.getenv('SOLANA_MODULE_ENABLED', 'false').lower() == 'true':
            required_vars.append('SOLANA_MODULE_PRIVATE_KEY')

        missing = []
        for var in required_vars:
            if var == "BINANCE_OR_BYBIT_API_KEYS":
                if not (os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET')) and \
                   not (os.getenv('BYBIT_API_KEY') and os.getenv('BYBIT_API_SECRET')):
                    missing.append(var)
            elif not os.getenv(var):
                missing.append(var)
        
        if missing:
            logger.warning(f"Missing environment variables: {', '.join(missing)}")
        
        return missing

    async def _load_default_config(self, config_type: ConfigType) -> None:
        """Load default configuration when file loading fails"""
        schema_class = self.config_schemas.get(config_type)
        if schema_class:
            self.configs[config_type] = schema_class()
            logger.info(f"Loaded default {config_type.value} configuration")

    def get_config(self, config_type: ConfigType) -> Optional[BaseModel]:
        """Get configuration for specified type"""
        return self.configs.get(config_type)

    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.configs.get(ConfigType.TRADING, TradingConfig())

    def get_strategies_config(self) -> StrategiesConfig:
        """Get strategies configuration"""
        return self.configs.get(ConfigType.STRATEGIES, StrategiesConfig())

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.configs.get(ConfigType.SECURITY, SecurityConfig())

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.configs.get(ConfigType.DATABASE, DatabaseConfig())

    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.configs.get(ConfigType.API, APIConfig())

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.configs.get(ConfigType.MONITORING, MonitoringConfig())

    def get_ml_models_config(self) -> MLModelsConfig:
        """Get ML models configuration"""
        return self.configs.get(ConfigType.ML_MODELS, MLModelsConfig())

    def get_risk_management_config(self) -> RiskManagementConfig:
        """Get risk management configuration"""
        return self.configs.get(ConfigType.RISK_MANAGEMENT, RiskManagementConfig())

    def get_portfolio_config(self) -> PortfolioConfig:
        """Get portfolio configuration"""
        return self.configs.get(ConfigType.PORTFOLIO, PortfolioConfig())

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.configs.get(ConfigType.LOGGING, LoggingConfig())

    def get_general_config(self) -> GeneralConfig:
        """Get general configuration"""
        return self.configs.get(ConfigType.GENERAL, GeneralConfig())

    def get_chain_config(self) -> ChainConfig:
        return self.configs.get(ConfigType.CHAIN, ChainConfig())

    def get_position_management_config(self) -> PositionManagementConfig:
        return self.configs.get(ConfigType.POSITION_MANAGEMENT, PositionManagementConfig())

    def get_volatility_config(self) -> VolatilityConfig:
        return self.configs.get(ConfigType.VOLATILITY, VolatilityConfig())

    def get_exit_strategy_config(self) -> ExitStrategyConfig:
        return self.configs.get(ConfigType.EXIT_STRATEGY, ExitStrategyConfig())

    def get_solana_config(self) -> SolanaConfig:
        return self.configs.get(ConfigType.SOLANA, SolanaConfig())

    def get_jupiter_config(self) -> JupiterConfig:
        return self.configs.get(ConfigType.JUPITER, JupiterConfig())

    def get_performance_config(self) -> PerformanceConfig:
        return self.configs.get(ConfigType.PERFORMANCE, PerformanceConfig())

    def get_feature_flags_config(self) -> FeatureFlagsConfig:
        return self.configs.get(ConfigType.FEATURE_FLAGS, FeatureFlagsConfig())

    def get_gas_price_config(self) -> GasPriceConfig:
        return self.configs.get(ConfigType.GAS_PRICE, GasPriceConfig())

    def get_trading_limits_config(self) -> TradingLimitsConfig:
        return self.configs.get(ConfigType.TRADING_LIMITS, TradingLimitsConfig())

    def get_backtesting_config(self) -> BacktestingConfig:
        return self.configs.get(ConfigType.BACKTESTING, BacktestingConfig())

    def get_network_config(self) -> NetworkConfig:
        return self.configs.get(ConfigType.NETWORK, NetworkConfig())

    def get_debug_config(self) -> DebugConfig:
        return self.configs.get(ConfigType.DEBUG, DebugConfig())

    def get_dashboard_config(self) -> DashboardConfig:
        return self.configs.get(ConfigType.DASHBOARD, DashboardConfig())
