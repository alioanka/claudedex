"""
Configuration Manager for DexScreener Trading Bot
Centralized configuration management with validation, hot-reloading, and environment handling
"""
from __future__ import annotations
from typing import Tuple

import os
import json
import yaml
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

import aiofiles
from pydantic import BaseModel, ValidationError, validator
from pydantic.types import SecretStr
from jsonschema import validate, ValidationError as JsonValidationError

from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class ConfigType(Enum):
    """Configuration types"""
    #SYSTEM = "system"
    TRADING = "trading"
    SECURITY = "security"
    DATABASE = "database"
    API = "api"
    MONITORING = "monitoring"
    ML_MODELS = "ml_models"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO = "portfolio"  # Add this

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

class TradingConfig(BaseModel):
    """Trading configuration schema"""
    max_position_size: float = 0.05  # 5% of portfolio
    max_daily_trades: int = 100
    max_slippage: float = 0.02  # 2%
    stop_loss_percentage: float = 0.05  # 5%
    take_profit_percentage: float = 0.15  # 15%
    min_liquidity_threshold: float = 50000  # $50k
    max_gas_price: int = 100  # Gwei
    max_slippage_bps: int = 50  # Max slippage tolerance (50 bps = 0.5%)
    expected_slippage_bps: int = 10  # Expected slippage (10 bps = 0.1%)
    max_price_impact_bps: int = 100  # Max price impact (100 bps = 1%)
    dex_fee_bps: int = 30  # DEX trading fee (30 bps = 0.3%)

    min_opportunity_score: float = 0.05  # Minimum score to consider an opportunity

    # Circuit Breaker Thresholds
    BREAKER_ERROR_RATE_MAX= int = 20             # Max error rate percentage
    BREAKER_SLIPPAGE_REALIZED_BPS_MAX= int = 120  # Max realized slippage in basis points
    BREAKER_MAX_CONSECUTIVE_LOSSES= int = 5       # Max consecutive losing trades
    BREAKER_MAX_DRAWDOWN_PCT= int = 15            # Max portfolio drawdown percentage
    BREAKER_MAX_DAILY_LOSS_PCT= int = 10          # Max daily loss percentage

    # ADD THIS:
    strategies: Dict[str, Dict[str, Any]] = {
        'momentum': {'enabled': True, 'lookback_period': 20},
        'scalping': {'enabled': True, 'profit_target': 0.02},
        'ai_strategy': {'enabled': False}
    }
    
    # Strategy settings
    enable_momentum_strategy: bool = True
    enable_scalping_strategy: bool = True
    enable_ai_strategy: bool = False
    
    # DEX settings
    enabled_dexes: List[str] = ["uniswap_v2", "uniswap_v3", "pancakeswap"]
    preferred_dex: str = "uniswap_v3"
    
    @validator('max_position_size')
    def validate_position_size(cls, v):
        if not 0 < v <= 1:
            raise ValueError('max_position_size must be between 0 and 1')
        return v

class SecurityConfig(BaseModel):
    """Security configuration schema"""
    api_rate_limit: int = 1000  # requests per hour
    max_login_attempts: int = 5
    session_timeout: int = 3600  # seconds
    require_2fa: bool = True
    
    # Encryption settings
    encryption_key_rotation_days: int = 30
    backup_encryption_enabled: bool = True
    
    # Audit settings
    audit_log_retention_days: int = 365
    real_time_alerts: bool = True
    
    # Wallet security
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
    password: SecretStr = SecretStr("bot_password")  # ✅ Add default value
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # Performance settings
    echo_sql: bool = False
    enable_query_cache: bool = True
    query_cache_size: int = 1000

class APIConfig(BaseModel):
    """API configuration schema"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    
    # Authentication
    jwt_secret: SecretStr = SecretStr("ti4o1XMf9NWB35RzTFvE0nqrCHblvviJC-N878PWoPs")  # ✅ Add default
    jwt_expiry: int = 3600  # seconds
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

class MonitoringConfig(BaseModel):
    """Monitoring configuration schema"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_rotation_size: str = "100MB"
    log_retention_days: int = 30
    
    # Alerts
    telegram_bot_token: Optional[SecretStr] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[SecretStr] = None
    email_smtp_server: Optional[str] = None
    email_username: Optional[str] = None
    email_password: Optional[SecretStr] = None

class MLModelsConfig(BaseModel):
    """ML Models configuration schema"""

    model_config = {'protected_namespaces': ()}  # Add this line
    # Model settings
    enable_ensemble: bool = True
    model_update_frequency: int = 24  # hours
    min_training_samples: int = 1000
    
    # Feature engineering
    lookback_period: int = 100  # candles
    feature_selection_threshold: float = 0.05
    
    # Training settings
    train_test_split: float = 0.8
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Model types
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_lstm: bool = True
    enable_transformer: bool = False  # Computationally intensive
    
    # Thresholds
    rug_detection_threshold: float = 0.7
    pump_prediction_threshold: float = 0.8
    volume_validation_threshold: float = 0.6

class RiskManagementConfig(BaseModel):
    """Risk management configuration schema"""
    # Portfolio limits
    max_portfolio_risk: float = 0.02  # 2% VaR
    max_position_correlation: float = 0.7
    max_sector_exposure: float = 0.3  # 30% per sector
    
    # Position sizing
    kelly_criterion_enabled: bool = True
    position_sizing_method: str = "kelly"  # "fixed", "kelly", "optimal_f"
    
    # Stop losses
    dynamic_stop_loss: bool = True
    atr_multiplier: float = 2.0
    trailing_stop_enabled: bool = True
    
    # Emergency controls
    emergency_stop_drawdown: float = 0.2  # 20%
    circuit_breaker_enabled: bool = True
    max_consecutive_losses: int = 5
    
    @validator('max_portfolio_risk')
    def validate_portfolio_risk(cls, v):
        if not 0 < v <= 0.1:
            raise ValueError('max_portfolio_risk must be between 0 and 0.1 (10%)')
        return v

class PortfolioConfig(BaseModel):
    """Portfolio configuration schema"""
    # ✅ CHANGE THESE VALUES:
    initial_balance: float = 400.0              # Was 10000, now 400
    max_positions: int = 40                      # Was 10, now 8
    max_position_size_pct: float = 0.10         # Keep 10%
    max_risk_per_trade: float = 0.10            # Was 0.05, now 0.10
    max_portfolio_risk: float = 0.25            # Keep 25%
    min_position_size: float = 5.0              # Was 100.0, now 5.0
    max_position_size: float = 15.0
    allocation_strategy: str = "DYNAMIC"        # Keep
    daily_loss_limit: float = 0.10              # Was 0.15, now 0.10
    consecutive_losses_limit: int = 5           # Keep
    correlation_threshold: float = 0.7          # Keep
    rebalance_frequency: str = "daily"          # Keep

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

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self.configs: Dict[ConfigType, BaseModel] = {}
        self.config_schemas: Dict[ConfigType, type] = {
            ConfigType.TRADING: TradingConfig,
            ConfigType.SECURITY: SecurityConfig,
            ConfigType.DATABASE: DatabaseConfig,
            ConfigType.API: APIConfig,
            ConfigType.MONITORING: MonitoringConfig,
            ConfigType.ML_MODELS: MLModelsConfig,
            ConfigType.RISK_MANAGEMENT: RiskManagementConfig,
            ConfigType.PORTFOLIO: PortfolioConfig  # Add this
        }
        
        # Configuration change tracking
        self.change_history: List[ConfigChange] = []
        self.config_watchers: Dict[ConfigType, List[Callable]] = {}
        self.file_watchers: Dict[str, float] = {}  # filename -> last_modified
        
        # Encryption for sensitive data
        self.encryption_manager = None
        
        # Hot-reload settings
        self.auto_reload_enabled = True
        self.reload_check_interval = 5  # seconds
        self._reload_task = None
        
        logger.info("ConfigManager initialized")

    async def initialize(self, encryption_key: Optional[str] = None) -> None:
        """Initialize configuration manager"""
        try:
            # Initialize encryption for sensitive data
            if encryption_key:
                encryption_config = {'encryption_key': encryption_key}
                self.encryption_manager = EncryptionManager(encryption_config)
            
            # Load all configurations
            await self._load_all_configs()
            
            # Start auto-reload if enabled
            if self.auto_reload_enabled:
                await self._start_auto_reload()
            
            logger.info("Configuration manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration manager: {e}")
            raise

    async def _load_all_configs(self) -> None:
        """Load all configuration types"""
        for config_type in ConfigType:
            try:
                await self._load_config(config_type)
            except Exception as e:
                logger.error(f"Failed to load {config_type.value} config: {e}")
                # Load default configuration
                await self._load_default_config(config_type)

    async def _load_config(self, config_type: ConfigType) -> None:
        """Load configuration from multiple sources"""
        config_data = {}
        
        # 1. Load default values from schema
        schema_class = self.config_schemas.get(config_type)
        if schema_class:
            default_config = schema_class()
            config_data.update(default_config.dict())
        
        # 2. Load from file (YAML/JSON)
        file_data = await self._load_config_from_file(config_type)
        if file_data:
            config_data.update(file_data)
        
        # 3. Override with environment variables
        env_data = self._load_config_from_env(config_type)
        if env_data:
            config_data.update(env_data)
        
        # 4. Load from database (if available)
        db_data = await self._load_config_from_database(config_type)
        if db_data:
            config_data.update(db_data)
        
        # 5. Decrypt sensitive values
        if self.encryption_manager:
            config_data = self._decrypt_sensitive_values(config_data)
        
        # 6. Validate and create config object
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

    async def _load_config_from_file(self, config_type: ConfigType) -> Optional[Dict]:
        """Load configuration from file"""
        config_file = self.config_dir / f"{config_type.value}.yaml"
        
        if not config_file.exists():
            # Try JSON format
            config_file = self.config_dir / f"{config_type.value}.json"
            if not config_file.exists():
                return None
        
        try:
            async with aiofiles.open(config_file, 'r') as f:
                content = await f.read()
            
            # Update file watcher
            stat = config_file.stat()
            self.file_watchers[str(config_file)] = stat.st_mtime
            
            if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                return yaml.safe_load(content)
            else:
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return None

    def _load_config_from_env(self, config_type: ConfigType) -> Dict:
        """Load configuration from environment variables"""
        env_data = {}
        
        # ✅ SPECIAL HANDLING FOR TRADING CONFIG
        if config_type == ConfigType.TRADING:
            if os.getenv('MIN_OPPORTUNITY_SCORE'):
                env_data['min_opportunity_score'] = float(os.getenv('MIN_OPPORTUNITY_SCORE'))
            if os.getenv('MAX_POSITION_SIZE_PERCENT'):
                env_data['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE_PERCENT')) / 100
            if os.getenv('MAX_SLIPPAGE'):
                env_data['max_slippage'] = float(os.getenv('MAX_SLIPPAGE'))
        
        # ✅ ADD THIS: Parse ENABLED_CHAINS from .env
        if config_type == ConfigType.API:  # Store chains config here temporarily
            enabled_chains_str = os.getenv('ENABLED_CHAINS')
            if enabled_chains_str:
                env_data['enabled_chains'] = enabled_chains_str
        
        # Generic prefix-based loading (existing code continues)
        prefix = f"BOT_{config_type.value.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Parse value type
                if value.lower() in ('true', 'false'):
                    env_data[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    env_data[config_key] = int(value)
                elif self._is_float(value):
                    env_data[config_key] = float(value)
                else:
                    env_data[config_key] = value
        
        return env_data

    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    async def _load_config_from_database(self, config_type: ConfigType) -> Optional[Dict]:
        """Load configuration from database"""
        # This would integrate with your database
        # For now, return None (no database config)
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


    # ✅ ADD THIS METHOD around line 450:
    def validate_environment(self) -> List[str]:
        """Validate required environment variables are set"""
        required_vars = [
            'ALCHEMY_API_KEY',
            'PRIVATE_KEY',
            'DATABASE_URL'
        ]
        
        missing = []
        for var in required_vars:
            if not os.getenv(var):
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

    async def _start_auto_reload(self) -> None:
        """Start automatic configuration reloading"""
        self._reload_task = asyncio.create_task(self._auto_reload_loop())

    async def _auto_reload_loop(self) -> None:
        """Auto-reload loop to check for configuration changes"""
        while True:
            try:
                await asyncio.sleep(self.reload_check_interval)
                await self._check_for_changes()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-reload loop: {e}")

    async def _check_for_changes(self) -> None:
        """Check for configuration file changes"""
        for file_path, last_modified in self.file_watchers.items():
            try:
                current_modified = Path(file_path).stat().st_mtime
                
                if current_modified > last_modified:
                    logger.info(f"Configuration file changed: {file_path}")
                    
                    # Determine config type from filename
                    filename = Path(file_path).stem
                    config_type = None
                    
                    for ct in ConfigType:
                        if ct.value == filename:
                            config_type = ct
                            break
                    
                    if config_type:
                        await self._reload_config(config_type)
                        self.file_watchers[file_path] = current_modified
                    
            except FileNotFoundError:
                # File was deleted
                logger.warning(f"Configuration file deleted: {file_path}")
                del self.file_watchers[file_path]
            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")

    async def _reload_config(self, config_type: ConfigType) -> None:
        """Reload specific configuration type"""
        try:
            old_config = self.configs.get(config_type)
            await self._load_config(config_type)
            new_config = self.configs.get(config_type)
            
            # Track changes
            if old_config and new_config:
                await self._track_config_changes(config_type, old_config, new_config)
            
            # Notify watchers
            await self._notify_config_watchers(config_type, new_config)
            
            logger.info(f"Reloaded {config_type.value} configuration")
            
        except Exception as e:
            logger.error(f"Failed to reload {config_type.value} configuration: {e}")

    async def _track_config_changes(self, config_type: ConfigType, old_config: BaseModel, new_config: BaseModel) -> None:
        """Track configuration changes"""
        old_dict = old_config.dict()
        new_dict = new_config.dict()
        
        for key in set(old_dict.keys()) | set(new_dict.keys()):
            old_value = old_dict.get(key)
            new_value = new_dict.get(key)
            
            if old_value != new_value:
                change = ConfigChange(
                    timestamp=datetime.utcnow(),
                    config_type=config_type,
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    source=ConfigSource.FILE,
                    user=None,
                    reason="auto_reload"
                )
                
                self.change_history.append(change)
                logger.info(f"Config change detected: {config_type.value}.{key} = {new_value}")

    async def _notify_config_watchers(self, config_type: ConfigType, new_config: BaseModel) -> None:
        """Notify registered watchers about configuration changes"""
        watchers = self.config_watchers.get(config_type, [])
        
        for watcher in watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    await watcher(config_type, new_config)
                else:
                    watcher(config_type, new_config)
            except Exception as e:
                logger.error(f"Error notifying config watcher: {e}")

    # Add these methods to ConfigManager class:

    def load_config(self, env: str) -> Dict:
        """
        Load configuration for specific environment
        Matches API specification signature
        
        Args:
            env: Environment name (development, staging, production)
            
        Returns:
            Configuration dictionary
        """
        # If configs are already loaded, return them
        if self.configs:
            result = {}
            for config_type, config in self.configs.items():
                result[config_type.value] = config.dict() if hasattr(config, 'dict') else config
            return result
        
        # If configs aren't loaded yet, they should have been loaded by initialize()
        # This is a fallback - return defaults synchronously
        logger.warning("load_config called but configs not initialized - returning defaults")
        
        for config_type in ConfigType:
            if config_type not in self.configs:
                schema_class = self.config_schemas.get(config_type)
                if schema_class:
                    self.configs[config_type] = schema_class()
        
        # Return combined config dict
        result = {}
        for config_type, config in self.configs.items():
            result[config_type.value] = config.dict() if hasattr(config, 'dict') else config
        
        return result

    async def reload_config(self) -> None:
        """
        Reload all configurations
        Matches API specification signature
        """
        # Clear existing configs
        self.configs.clear()
        
        # Reload all
        await self._load_all_configs()
        
        # Notify watchers
        for config_type, config in self.configs.items():
            await self._notify_config_watchers(config_type, config)

    def validate_config(self, config: Dict) -> bool:
        """
        Validate configuration dictionary
        Matches API specification signature
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate each config type
            for config_type_str, config_data in config.items():
                # Find matching ConfigType enum
                config_type = None
                for ct in ConfigType:
                    if ct.value == config_type_str:
                        config_type = ct
                        break
                
                if not config_type:
                    return False
                
                # Get schema and validate
                schema_class = self.config_schemas.get(config_type)
                if schema_class:
                    schema_class(**config_data)  # Will raise ValidationError if invalid
                
            return True
            
        except ValidationError:
            return False
        except Exception:
            return False

    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value by key
        Matches API specification signature
        
        Args:
            key: Configuration key in format "type.field"
            value: New value
        """
        # Parse key
        parts = key.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid key format: {key}")
        
        config_type_str = parts[0]
        field_name = '.'.join(parts[1:])
        
        # Find config type
        config_type = None
        for ct in ConfigType:
            if ct.value == config_type_str:
                config_type = ct
                break
        
        if not config_type:
            raise ValueError(f"Unknown config type: {config_type_str}")
        
        # Update using existing method
        asyncio.run(self.update_config_internal(
            config_type=config_type,
            updates={field_name: value}
        ))

    def get_config(self, key: str) -> Any:
        """
        Get configuration value by key
        Matches API specification signature
        
        Args:
            key: Configuration key in format "type.field"
            
        Returns:
            Configuration value
        """
        # Parse key
        parts = key.split('.')
        if len(parts) < 1:
            raise ValueError(f"Invalid key format: {key}")
        
        # If just config type requested
        if len(parts) == 1:
            config_type_str = parts[0]
            for ct in ConfigType:
                if ct.value == config_type_str:
                    config = self.configs.get(ct)
                    return config.dict() if config else None
            return None
        
        # Get specific field
        config_type_str = parts[0]
        field_path = parts[1:]
        
        for ct in ConfigType:
            if ct.value == config_type_str:
                config = self.configs.get(ct)
                if config:
                    result = config.dict() if hasattr(config, 'dict') else config
                    # Navigate nested path
                    for field in field_path:
                        if isinstance(result, dict):
                            result = result.get(field)
                        else:
                            return None
                    return result
        
        return None

    # Rename the existing update_config to update_config_internal to avoid conflict


    def get_portfolio_config(self) -> PortfolioConfig:
        """Get portfolio configuration"""
        return self.configs.get(ConfigType.PORTFOLIO, PortfolioConfig())

    def get_config_internal(self, config_type: ConfigType) -> Optional[BaseModel]:
        """Get configuration for specified type"""
        return self.configs.get(config_type)

    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.configs.get(ConfigType.TRADING, TradingConfig())

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

    async def update_config_internal(self, 
                          config_type: ConfigType, 
                          updates: Dict[str, Any],
                          user: Optional[str] = None,
                          reason: Optional[str] = None,
                          persist: bool = True) -> bool:
        """Update configuration with validation"""
        try:
            current_config = self.configs.get(config_type)
            if not current_config:
                logger.error(f"No config found for type {config_type.value}")
                return False
            
            # Get current config as dict
            config_dict = current_config.dict()
            
            # Track changes
            changes = []
            for key, new_value in updates.items():
                old_value = config_dict.get(key)
                if old_value != new_value:
                    changes.append(ConfigChange(
                        timestamp=datetime.utcnow(),
                        config_type=config_type,
                        key=key,
                        old_value=old_value,
                        new_value=new_value,
                        source=ConfigSource.DATABASE,
                        user=user,
                        reason=reason
                    ))
            
            # Apply updates
            config_dict.update(updates)
            
            # Validate updated configuration
            schema_class = self.config_schemas.get(config_type)
            if schema_class:
                try:
                    new_config = schema_class(**config_dict)
                except ValidationError as e:
                    logger.error(f"Validation error for {config_type.value}: {e}")
                    return False
            else:
                logger.error(f"No schema for config type {config_type.value}")
                return False
            
            # Update in memory
            self.configs[config_type] = new_config
            
            # Record changes
            self.change_history.extend(changes)
            
            # Persist if requested
            if persist:
                await self._persist_config(config_type, new_config)
            
            # Notify watchers
            await self._notify_config_watchers(config_type, new_config)
            
            logger.info(f"Updated {config_type.value} configuration with {len(updates)} changes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update {config_type.value} configuration: {e}")
            return False

    async def _persist_config(self, config_type: ConfigType, config: BaseModel) -> None:
        """Persist configuration to file"""
        try:
            config_file = self.config_dir / f"{config_type.value}.yaml"
            config_dict = config.dict()
            
            # Encrypt sensitive values before saving
            if self.encryption_manager:
                config_dict = self._encrypt_sensitive_values(config_dict)
            
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(yaml.dump(config_dict, default_flow_style=False))
            
            # Update file watcher
            stat = config_file.stat()
            self.file_watchers[str(config_file)] = stat.st_mtime
            
            logger.info(f"Persisted {config_type.value} configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist {config_type.value} configuration: {e}")

    def _encrypt_sensitive_values(self, config_data: Dict) -> Dict:
        """Encrypt sensitive configuration values before saving"""
        if not self.encryption_manager:
            return config_data
        
        sensitive_keys = ['password', 'secret', 'token', 'key']
        
        def encrypt_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and not value.startswith('encrypted:'):
                            try:
                                data[key] = 'encrypted:' + self.encryption_manager.encrypt_sensitive_data(value)
                            except Exception as e:
                                logger.warning(f"Failed to encrypt {key}: {e}")
                    else:
                        data[key] = encrypt_recursive(value)
            elif isinstance(data, list):
                return [encrypt_recursive(item) for item in data]
            return data
        
        return encrypt_recursive(config_data.copy())

    def register_watcher(self, config_type: ConfigType, callback: Callable) -> None:
        """Register a callback for configuration changes"""
        if config_type not in self.config_watchers:
            self.config_watchers[config_type] = []
        
        self.config_watchers[config_type].append(callback)
        logger.info(f"Registered watcher for {config_type.value} configuration")

    def unregister_watcher(self, config_type: ConfigType, callback: Callable) -> None:
        """Unregister a configuration watcher"""
        watchers = self.config_watchers.get(config_type, [])
        if callback in watchers:
            watchers.remove(callback)
            logger.info(f"Unregistered watcher for {config_type.value} configuration")

    def get_change_history(self, 
                          config_type: Optional[ConfigType] = None,
                          limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history"""
        changes = self.change_history
        
        if config_type:
            changes = [c for c in changes if c.config_type == config_type]
        
        # Sort by timestamp (newest first) and limit
        changes = sorted(changes, key=lambda x: x.timestamp, reverse=True)
        return changes[:limit]

    async def backup_configs(self, backup_path: str) -> bool:
        """Backup all configurations"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            
            for config_type, config in self.configs.items():
                backup_file = backup_dir / f"{config_type.value}_{timestamp}.yaml"
                
                async with aiofiles.open(backup_file, 'w') as f:
                    await f.write(yaml.dump(config.dict(), default_flow_style=False))
            
            logger.info(f"Backed up configurations to {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup configurations: {e}")
            return False

    async def restore_configs(self, backup_path: str) -> bool:
        """Restore configurations from backup"""
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                logger.error(f"Backup directory does not exist: {backup_dir}")
                return False
            
            for config_type in ConfigType:
                # Find latest backup file for this config type
                pattern = f"{config_type.value}_*.yaml"
                backup_files = list(backup_dir.glob(pattern))
                
                if backup_files:
                    latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
                    
                    async with aiofiles.open(latest_backup, 'r') as f:
                        content = await f.read()
                    
                    config_data = yaml.safe_load(content)
                    
                    # Validate and restore
                    schema_class = self.config_schemas.get(config_type)
                    if schema_class:
                        try:
                            restored_config = schema_class(**config_data)
                            self.configs[config_type] = restored_config
                            logger.info(f"Restored {config_type.value} from {latest_backup}")
                        except ValidationError as e:
                            logger.error(f"Validation error restoring {config_type.value}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore configurations: {e}")
            return False

    def validate_config_internal(self, config_type: ConfigType, config_data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate configuration data against schema"""
        try:
            schema_class = self.config_schemas.get(config_type)
            if not schema_class:
                return False, f"No schema defined for {config_type.value}"
            
            schema_class(**config_data)
            return True, None
            
        except ValidationError as e:
            return False, str(e)

    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration manager status"""
        return {
            'loaded_configs': list(self.configs.keys()),
            'auto_reload_enabled': self.auto_reload_enabled,
            'watched_files': len(self.file_watchers),
            'total_changes': len(self.change_history),
            'registered_watchers': {
                config_type.value: len(watchers) 
                for config_type, watchers in self.config_watchers.items()
            }
        }

    async def cleanup(self) -> None:
        """Cleanup configuration manager"""
        try:
            # Cancel auto-reload task
            if self._reload_task:
                self._reload_task.cancel()
                try:
                    await self._reload_task
                except asyncio.CancelledError:
                    pass
            
            # Clear configurations
            self.configs.clear()
            self.config_watchers.clear()
            self.file_watchers.clear()
            
            logger.info("Configuration manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during configuration manager cleanup: {e}")