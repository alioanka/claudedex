"""
Futures Trading Module Configuration Manager
Database-backed configuration with .env integration for sensitive data

Architecture:
- All trading configuration is stored in the database (config_settings table)
- Only sensitive data (API keys, private keys) is read from .env
- Settings page writes/reads from database via API endpoints
- Hot-reload support for live configuration changes
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
    """Futures configuration types - maps to config_type in database"""
    GENERAL = "futures_general"
    POSITION = "futures_position"
    LEVERAGE = "futures_leverage"
    RISK = "futures_risk"
    PAIRS = "futures_pairs"
    STRATEGY = "futures_strategy"
    FUNDING = "futures_funding"


# Pydantic models for validation

class FuturesGeneralConfig(BaseModel):
    """General futures trading configuration"""
    enabled: bool = False
    exchange: str = "binance"  # binance or bybit
    # CRITICAL: Default to mainnet (False) for live trading safety
    # Set testnet=True in database settings if you want to use testnet
    # Or set FUTURES_TESTNET=true in environment
    testnet: bool = False
    contract_type: str = "perpetual"


class FuturesPositionConfig(BaseModel):
    """Position sizing configuration"""
    capital_allocation: float = 300.0
    # Dynamic position sizing
    dynamic_position_sizing: bool = True  # Enable/disable dynamic sizing
    min_position_pct: float = 5.0  # Minimum position size as % of capital
    max_position_pct: float = 20.0  # Maximum position size as % of capital
    max_position_usd: float = 500.0  # Maximum position size in USD (cap)
    # Static position sizing (when dynamic is disabled)
    static_position_pct: float = 15.0  # Fixed position size as % of capital
    position_size_usd: float = 100.0  # Legacy: fixed USD position size (deprecated)
    max_positions: int = 5
    min_trade_size: float = 10.0


class FuturesLeverageConfig(BaseModel):
    """Leverage configuration"""
    default_leverage: int = 10
    max_leverage: int = 20
    margin_mode: str = "isolated"  # isolated or cross


class FuturesRiskConfig(BaseModel):
    """Risk management configuration

    IMPORTANT: For profitability, TP1 must be > SL and position sizing must be front-loaded.

    Risk/Reward Math Example (with defaults below):
    - SL at 1.2%: Max loss = 1.2% of position
    - TP1 at 1.8% (40% closed): Lock in 0.72% profit
    - TP2 at 3.5% (30% closed): Lock in 1.05% more
    - After TP2: Stop moves to breakeven, risk-free ride to TP3/TP4

    With 54% win rate and these settings:
    - Expected TP1 profit: 0.54 × 1.8% × 40% = 0.39% per trade (partial)
    - Expected SL loss: 0.46 × 1.2% × 100% = 0.55% per trade
    - But TP2+ adds extra profit making overall EV positive
    """
    # CRITICAL: SL must be LESS than TP1 for positive expectancy
    stop_loss_pct: float = 1.2  # Tighter stop - cut losses quickly (was 2.0)

    # Multiple Take Profits - Front-loaded for early profit capture
    # TP1 should be wider than SL to ensure R:R > 1
    tp1_pct: float = 1.8  # First TP at 1.8% (wider than SL) - was 2.0
    tp2_pct: float = 3.5  # Second TP at 3.5% - was 4.0
    tp3_pct: float = 6.0  # Third TP at 6% (unchanged)
    tp4_pct: float = 10.0  # Fourth TP at 10% (unchanged)

    # Position size distribution - FRONT-LOADED for early profit locking
    # Take more profit early to ensure wins outweigh losses
    tp1_size_pct: float = 40.0  # Close 40% at TP1 (was 25%) - lock in profits early
    tp2_size_pct: float = 30.0  # Close 30% at TP2 (was 25%) - capture momentum
    tp3_size_pct: float = 20.0  # Close 20% at TP3 (was 25%)
    tp4_size_pct: float = 10.0  # Close remaining 10% at TP4 (was 25%) - let runners run

    # Legacy single TP (deprecated, use tp1_pct instead)
    take_profit_pct: float = 4.0

    # Daily loss limits
    max_daily_loss_usd: float = 500.0
    max_daily_loss_pct: float = 5.0
    liquidation_buffer: float = 20.0

    # Trailing stop - tighter for better profit protection
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 1.0  # Tighter trailing (was 1.5)

    # Risk controls
    max_consecutive_losses: int = 4  # Reduced from 5 to pause earlier

    # Market condition filters (new)
    require_trend_confirmation: bool = True  # Only trade with confirmed trends
    min_volume_multiplier: float = 1.2  # Require 20% above average volume


class FuturesPairsConfig(BaseModel):
    """Trading pairs configuration"""
    allowed_pairs: str = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT"
    both_directions: bool = True
    preferred_direction: str = "both"  # long, short, both

    @property
    def pairs_list(self) -> List[str]:
        """Get allowed pairs as a list"""
        return [p.strip() for p in self.allowed_pairs.split(',')]


class FuturesStrategyConfig(BaseModel):
    """Strategy parameters configuration

    Signal Quality:
    - 5 indicators (RSI, MACD, Volume, Bollinger, EMA) each score -2 to +2
    - Total score range: -10 to +10
    - Higher min_signal_score = fewer but higher quality trades
    - Recommended: 4-5 for balanced, 6+ for conservative
    """
    signal_timeframe: str = "15m"  # Timeframe for signal analysis: 1m, 5m, 15m, 30m, 1h, 4h
    scan_interval_seconds: int = 30  # How often to scan for opportunities

    # RSI thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_weak_oversold: float = 40.0
    rsi_weak_overbought: float = 60.0

    # Signal quality threshold - CRITICAL for profitability
    # Higher = fewer trades but better win rate
    min_signal_score: int = 4  # Increased from 3 for better entries

    # Additional filters for trade quality
    require_trend_alignment: bool = True  # Trade only in direction of trend
    require_volume_confirmation: bool = True  # Require above-average volume

    verbose_signals: bool = True
    cooldown_minutes: int = 10  # Increased from 5 to avoid overtrading after losses


class FuturesFundingConfig(BaseModel):
    """Funding rate settings"""
    funding_arbitrage_enabled: bool = False
    max_funding_rate: float = 0.1


class FuturesConfigManager:
    """
    Configuration manager for Futures Trading Module

    Follows the same pattern as DEX config_manager.py:
    - Database-backed storage for all trading configuration
    - .env integration for sensitive data ONLY (API keys, secrets)
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
            FuturesConfigType.POSITION: FuturesPositionConfig,
            FuturesConfigType.LEVERAGE: FuturesLeverageConfig,
            FuturesConfigType.RISK: FuturesRiskConfig,
            FuturesConfigType.PAIRS: FuturesPairsConfig,
            FuturesConfigType.STRATEGY: FuturesStrategyConfig,
            FuturesConfigType.FUNDING: FuturesFundingConfig,
        }

        # Loaded configs
        self.configs: Dict[FuturesConfigType, BaseModel] = {}

        # Environment config (sensitive data ONLY from .env)
        self._env_config = self._load_environment_config()

        # Load default configs
        for config_type, model_class in self.config_models.items():
            self.configs[config_type] = model_class()

        logger.info("FuturesConfigManager initialized")

    def _load_environment_config(self) -> Dict[str, Any]:
        """
        Load ONLY sensitive configuration from secrets manager or .env

        Priority:
        1. Secrets manager (Docker secrets, encrypted database)
        2. Environment variables (.env fallback)

        Returns:
            Dict: Environment configuration (sensitive data only)
        """
        env_config = {}

        # ONLY sensitive keys - API credentials
        sensitive_keys = [
            # Binance
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'BINANCE_TESTNET_API_KEY',
            'BINANCE_TESTNET_API_SECRET',
            # Bybit
            'BYBIT_API_KEY',
            'BYBIT_API_SECRET',
            'BYBIT_TESTNET_API_KEY',
            'BYBIT_TESTNET_API_SECRET',
        ]

        # Try secrets manager first
        try:
            from security.secrets_manager import secrets
            for var in sensitive_keys:
                value = secrets.get(var, log_access=False)
                if value and value not in ('null', 'None', '', 'your_testnet_api_key', 'your_mainnet_api_key', 'PLACEHOLDER'):
                    env_config[var] = value
                    logger.debug(f"Loaded {var} from secrets manager")
        except Exception as e:
            logger.debug(f"Secrets manager not available, using environment: {e}")
            # Fallback to environment
            for var in sensitive_keys:
                value = os.getenv(var)
                if value and value not in ('null', 'None', '', 'your_testnet_api_key', 'your_mainnet_api_key'):
                    env_config[var] = value

        return env_config

    async def initialize(self) -> None:
        """Initialize and load all configurations from database"""
        try:
            # Reload sensitive credentials from secrets manager (async)
            # This is needed because __init__ uses sync secrets.get() which skips DB in async context
            await self._reload_sensitive_credentials()
            await self._load_all_configs()
            logger.info("✅ Futures configuration loaded from database")
        except Exception as e:
            logger.error(f"Failed to load Futures config from database: {e}", exc_info=True)
            logger.info("Using default configuration")

    async def set_db_pool(self, db_pool) -> None:
        """
        Set database pool and reload configs

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool
        if db_pool:
            logger.info("Database pool set, reloading Futures configs...")
            # Reload environment config with secrets manager now that DB is available
            await self._reload_sensitive_credentials()
            await self._load_all_configs()

    async def _reload_sensitive_credentials(self) -> None:
        """Reload API credentials from secrets manager after DB pool is available"""
        try:
            from security.secrets_manager import secrets

            # Initialize secrets manager with db_pool (re-init if in bootstrap mode)
            if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                secrets.initialize(self.db_pool)

            sensitive_keys = [
                'BINANCE_API_KEY', 'BINANCE_API_SECRET',
                'BINANCE_TESTNET_API_KEY', 'BINANCE_TESTNET_API_SECRET',
                'BYBIT_API_KEY', 'BYBIT_API_SECRET',
                'BYBIT_TESTNET_API_KEY', 'BYBIT_TESTNET_API_SECRET',
            ]

            for var in sensitive_keys:
                value = await secrets.get_async(var)
                if value and value not in ('null', 'None', '', 'PLACEHOLDER', 'your_testnet_api_key', 'your_mainnet_api_key'):
                    # Check if value is encrypted and needs decryption
                    if value.startswith('gAAAAAB'):
                        decrypted = await self._decrypt_value(value)
                        if decrypted:
                            self._env_config[var] = decrypted
                            logger.debug(f"Loaded and decrypted {var} from secrets manager")
                    else:
                        self._env_config[var] = value
                        logger.debug(f"Loaded {var} from secrets manager")

            logger.info(f"✅ Reloaded {len(self._env_config)} API credentials from secrets manager")

        except Exception as e:
            logger.warning(f"Failed to reload credentials from secrets manager: {e}")

    async def _decrypt_value(self, encrypted_value: str) -> Optional[str]:
        """Decrypt a Fernet-encrypted value"""
        try:
            from pathlib import Path
            from cryptography.fernet import Fernet

            encryption_key = None
            key_file = Path('.encryption_key')
            if key_file.exists():
                encryption_key = key_file.read_text().strip()
            if not encryption_key:
                encryption_key = os.getenv('ENCRYPTION_KEY')

            if encryption_key:
                f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                return f.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
        return None

    async def _load_all_configs(self) -> None:
        """Load all configuration types from database"""
        if not self.db_pool:
            logger.warning("No database pool, using default configs")
            return

        for config_type, model_class in self.config_models.items():
            try:
                config = await self._load_config_from_db(config_type)
                if config:
                    self.configs[config_type] = model_class(**config)
                    logger.debug(f"Loaded {config_type.value} from database")
            except Exception as e:
                logger.error(f"Error loading {config_type.value}: {e}")

    async def _load_config_from_db(self, config_type: FuturesConfigType) -> Optional[Dict]:
        """
        Load configuration from database config_settings table

        Args:
            config_type: Configuration type to load

        Returns:
            Optional[Dict]: Configuration data or None
        """
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

                    # Convert value based on type
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
            logger.error(f"Database error loading {config_type.value}: {e}")
            return None

    async def save_config(self, config_type: FuturesConfigType, config_data: Dict[str, Any],
                         user_id: int = None, reason: str = None) -> bool:
        """
        Save configuration to database

        Args:
            config_type: Configuration type
            config_data: Configuration data to save
            user_id: User ID making the change
            reason: Reason for the change

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

            async with self.db_pool.acquire() as conn:
                for key, value in validated_config.dict().items():
                    # Determine value type
                    if isinstance(value, bool):
                        value_type = 'bool'
                        value_str = str(value).lower()
                    elif isinstance(value, int):
                        value_type = 'int'
                        value_str = str(value)
                    elif isinstance(value, float):
                        value_type = 'float'
                        value_str = str(value)
                    else:
                        value_type = 'string'
                        value_str = str(value)

                    # Get old value for history
                    old_row = await conn.fetchrow("""
                        SELECT value FROM config_settings
                        WHERE config_type = $1 AND key = $2
                    """, config_type.value, key)
                    old_value = old_row['value'] if old_row else None

                    # Update or insert config
                    await conn.execute("""
                        INSERT INTO config_settings (config_type, key, value, value_type, updated_by)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (config_type, key) DO UPDATE
                        SET value = $3,
                            value_type = $4,
                            updated_by = $5,
                            updated_at = NOW()
                    """, config_type.value, key, value_str, value_type, user_id)

                    # Log to history
                    if old_value != value_str:
                        await conn.execute("""
                            INSERT INTO config_history (
                                config_type, key, old_value, new_value,
                                change_source, changed_by, reason
                            )
                            VALUES ($1, $2, $3, $4, 'api', $5, $6)
                        """, config_type.value, key, old_value, value_str, user_id, reason)

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
            key: Configuration key (can include category prefix like "risk.stop_loss_pct")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # Check environment config first for sensitive data
        env_key = key.upper().replace('.', '_')
        if env_key in self._env_config:
            return self._env_config[env_key]

        # Parse key (e.g., "risk.stop_loss_pct" or just "stop_loss_pct")
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

    def get_general(self) -> FuturesGeneralConfig:
        """Get general configuration"""
        return self.configs.get(FuturesConfigType.GENERAL, FuturesGeneralConfig())

    def get_position(self) -> FuturesPositionConfig:
        """Get position configuration"""
        return self.configs.get(FuturesConfigType.POSITION, FuturesPositionConfig())

    def get_leverage(self) -> FuturesLeverageConfig:
        """Get leverage configuration"""
        return self.configs.get(FuturesConfigType.LEVERAGE, FuturesLeverageConfig())

    def get_risk(self) -> FuturesRiskConfig:
        """Get risk configuration"""
        return self.configs.get(FuturesConfigType.RISK, FuturesRiskConfig())

    def get_pairs(self) -> FuturesPairsConfig:
        """Get pairs configuration"""
        return self.configs.get(FuturesConfigType.PAIRS, FuturesPairsConfig())

    def get_strategy(self) -> FuturesStrategyConfig:
        """Get strategy configuration"""
        return self.configs.get(FuturesConfigType.STRATEGY, FuturesStrategyConfig())

    def get_funding(self) -> FuturesFundingConfig:
        """Get funding configuration"""
        return self.configs.get(FuturesConfigType.FUNDING, FuturesFundingConfig())

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a flat dictionary for the settings page

        Returns:
            Dict: All settings with prefixed keys
        """
        all_settings = {}

        for config_type, config_obj in self.configs.items():
            for key, value in config_obj.dict().items():
                # Use the key directly (settings page uses flat keys)
                all_settings[f"futures_{key}"] = value

        # Add flags for API key availability (don't expose actual values)
        all_settings['_has_binance_api'] = bool(self._env_config.get('BINANCE_API_KEY') or
                                                 self._env_config.get('BINANCE_TESTNET_API_KEY'))
        all_settings['_has_bybit_api'] = bool(self._env_config.get('BYBIT_API_KEY') or
                                               self._env_config.get('BYBIT_TESTNET_API_KEY'))

        return all_settings

    async def update_from_settings_page(self, settings: Dict[str, Any], user_id: int = None) -> bool:
        """
        Update configuration from settings page submission

        Args:
            settings: Dictionary of settings from the page (with futures_ prefix)
            user_id: User making the change

        Returns:
            bool: True if all updates succeeded
        """
        # Map settings to their config types
        config_map = {
            'enabled': FuturesConfigType.GENERAL,
            'exchange': FuturesConfigType.GENERAL,
            'testnet': FuturesConfigType.GENERAL,
            'trading_mode': FuturesConfigType.GENERAL,  # alias for testnet
            'contract_type': FuturesConfigType.GENERAL,
            # Position settings
            'capital': FuturesConfigType.POSITION,  # alias for capital_allocation
            'capital_allocation': FuturesConfigType.POSITION,
            'position_size_usd': FuturesConfigType.POSITION,
            'max_position_pct': FuturesConfigType.POSITION,
            'max_positions': FuturesConfigType.POSITION,
            'min_trade_size': FuturesConfigType.POSITION,
            # Dynamic position sizing
            'dynamic_position_sizing': FuturesConfigType.POSITION,
            'min_position_pct': FuturesConfigType.POSITION,
            'max_position_usd': FuturesConfigType.POSITION,
            'static_position_pct': FuturesConfigType.POSITION,
            # Leverage settings
            'leverage': FuturesConfigType.LEVERAGE,  # alias for default_leverage
            'default_leverage': FuturesConfigType.LEVERAGE,
            'max_leverage': FuturesConfigType.LEVERAGE,
            'margin_mode': FuturesConfigType.LEVERAGE,
            # Risk settings - SL
            'stop_loss': FuturesConfigType.RISK,  # alias
            'stop_loss_pct': FuturesConfigType.RISK,
            # Risk settings - Legacy single TP
            'take_profit': FuturesConfigType.RISK,  # alias
            'take_profit_pct': FuturesConfigType.RISK,
            # Risk settings - Multiple TPs
            'tp1_pct': FuturesConfigType.RISK,
            'tp2_pct': FuturesConfigType.RISK,
            'tp3_pct': FuturesConfigType.RISK,
            'tp4_pct': FuturesConfigType.RISK,
            'tp1_size_pct': FuturesConfigType.RISK,
            'tp2_size_pct': FuturesConfigType.RISK,
            'tp3_size_pct': FuturesConfigType.RISK,
            'tp4_size_pct': FuturesConfigType.RISK,
            # Risk settings - Daily loss
            'daily_loss_limit': FuturesConfigType.RISK,  # alias
            'max_daily_loss_usd': FuturesConfigType.RISK,
            'max_daily_loss_pct': FuturesConfigType.RISK,
            'max_consecutive_losses': FuturesConfigType.RISK,
            'liquidation_buffer': FuturesConfigType.RISK,
            # Risk settings - Trailing stop
            'trailing_stop': FuturesConfigType.RISK,  # alias
            'trailing_stop_enabled': FuturesConfigType.RISK,
            'trailing_distance': FuturesConfigType.RISK,  # alias
            'trailing_stop_distance': FuturesConfigType.RISK,
            # Pairs settings
            'allowed_pairs': FuturesConfigType.PAIRS,
            'both_directions': FuturesConfigType.PAIRS,
            'preferred_direction': FuturesConfigType.PAIRS,
            # Strategy settings
            'signal_timeframe': FuturesConfigType.STRATEGY,
            'timeframe': FuturesConfigType.STRATEGY,  # alias
            'scan_interval_seconds': FuturesConfigType.STRATEGY,
            'scan_interval': FuturesConfigType.STRATEGY,  # alias
            'rsi_oversold': FuturesConfigType.STRATEGY,
            'rsi_overbought': FuturesConfigType.STRATEGY,
            'rsi_weak_oversold': FuturesConfigType.STRATEGY,
            'rsi_weak_overbought': FuturesConfigType.STRATEGY,
            'min_signal_score': FuturesConfigType.STRATEGY,
            'verbose_signals': FuturesConfigType.STRATEGY,
            'cooldown_minutes': FuturesConfigType.STRATEGY,
            'require_trend_alignment': FuturesConfigType.STRATEGY,
            'require_volume_confirmation': FuturesConfigType.STRATEGY,
            # Risk - new market filters
            'require_trend_confirmation': FuturesConfigType.RISK,
            'min_volume_multiplier': FuturesConfigType.RISK,
            # Funding settings
            'funding_arb': FuturesConfigType.FUNDING,  # alias
            'funding_arbitrage_enabled': FuturesConfigType.FUNDING,
            'max_funding_rate': FuturesConfigType.FUNDING,
        }

        # Group settings by config type
        grouped: Dict[FuturesConfigType, Dict[str, Any]] = {}

        for key, value in settings.items():
            # Remove futures_ prefix if present
            clean_key = key.replace('futures_', '') if key.startswith('futures_') else key

            config_type = config_map.get(clean_key)
            if config_type:
                if config_type not in grouped:
                    # Start with current config values
                    grouped[config_type] = self.configs[config_type].dict()

                # Handle aliases
                actual_key = self._resolve_alias(clean_key)
                grouped[config_type][actual_key] = value

        # Save each config type
        success = True
        for config_type, config_data in grouped.items():
            if not await self.save_config(config_type, config_data, user_id, "Settings page update"):
                success = False

        return success

    def _resolve_alias(self, key: str) -> str:
        """Resolve setting aliases to actual field names"""
        aliases = {
            'trading_mode': 'testnet',  # paper = testnet=true, live = testnet=false
            'capital': 'capital_allocation',
            'leverage': 'default_leverage',
            'stop_loss': 'stop_loss_pct',
            'take_profit': 'take_profit_pct',
            'daily_loss_limit': 'max_daily_loss_usd',
            'trailing_stop': 'trailing_stop_enabled',
            'trailing_distance': 'trailing_stop_distance',
            'funding_arb': 'funding_arbitrage_enabled',
            'timeframe': 'signal_timeframe',
            'scan_interval': 'scan_interval_seconds',
        }
        return aliases.get(key, key)

    def get_api_credentials(self, exchange: str = None, testnet: bool = None) -> Dict[str, str]:
        """
        Get API credentials for the specified exchange

        Args:
            exchange: Exchange name (binance or bybit), defaults to config
            testnet: Whether to use testnet, defaults to config

        Returns:
            Dict with 'api_key' and 'api_secret'
        """
        if exchange is None:
            exchange = self.get_general().exchange
        if testnet is None:
            testnet = self.get_general().testnet

        exchange = exchange.upper()

        if testnet:
            api_key = self._env_config.get(f'{exchange}_TESTNET_API_KEY', '')
            api_secret = self._env_config.get(f'{exchange}_TESTNET_API_SECRET', '')
        else:
            api_key = self._env_config.get(f'{exchange}_API_KEY', '')
            api_secret = self._env_config.get(f'{exchange}_API_SECRET', '')

        return {
            'api_key': api_key,
            'api_secret': api_secret
        }
