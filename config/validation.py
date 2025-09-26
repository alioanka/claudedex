"""
Configuration Validation Module for DexScreener Trading Bot
Comprehensive validation rules for all configuration types
"""

import re
import ipaddress
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime
import logging
from urllib.parse import urlparse
from pathlib import Path

from pydantic import ValidationError, validator
from web3 import Web3

logger = logging.getLogger(__name__)

class ValidationResult:
    """Validation result container"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
    
    def add_error(self, message: str) -> None:
        """Add validation error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning"""
        self.warnings.append(message)
    
    def add_recommendation(self, message: str) -> None:
        """Add recommendation"""
        self.recommendations.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result"""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.recommendations.extend(other.recommendations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }

class ConfigValidator:
    """
    Comprehensive configuration validator with:
    - Type validation
    - Range validation
    - Format validation
    - Cross-field validation
    - Security validation
    - Performance validation
    """
    
    def __init__(self):
        self.validators = {
            'trading': self._validate_trading_config,
            'security': self._validate_security_config,
            'database': self._validate_database_config,
            'api': self._validate_api_config,
            'monitoring': self._validate_monitoring_config,
            'ml_models': self._validate_ml_models_config,
            'risk_management': self._validate_risk_management_config
        }
    
    def validate_config(self, config_type: str, config_data: Dict[str, Any]) -> ValidationResult:
        """Validate configuration based on type"""
        result = ValidationResult()
        
        try:
            # Get validator function
            validator_func = self.validators.get(config_type)
            if not validator_func:
                result.add_error(f"No validator found for config type: {config_type}")
                return result
            
            # Run validation
            validator_func(config_data, result)
            
            # Run cross-validation checks
            self._validate_cross_config(config_type, config_data, result)
            
            return result
            
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
            return result
    
    def _validate_trading_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate trading configuration"""
        
        # Position size validation
        max_position_size = config.get('max_position_size', 0)
        if not isinstance(max_position_size, (int, float)) or max_position_size <= 0:
            result.add_error("max_position_size must be a positive number")
        elif max_position_size > 0.2:  # 20%
            result.add_warning("max_position_size above 20% is very risky")
        elif max_position_size > 0.1:  # 10%
            result.add_recommendation("Consider reducing max_position_size below 10%")
        
        # Daily trades validation
        max_daily_trades = config.get('max_daily_trades', 0)
        if not isinstance(max_daily_trades, int) or max_daily_trades <= 0:
            result.add_error("max_daily_trades must be a positive integer")
        elif max_daily_trades > 1000:
            result.add_warning("max_daily_trades above 1000 may hit rate limits")
        
        # Slippage validation
        max_slippage = config.get('max_slippage', 0)
        if not isinstance(max_slippage, (int, float)) or max_slippage <= 0:
            result.add_error("max_slippage must be a positive number")
        elif max_slippage > 0.1:  # 10%
            result.add_warning("max_slippage above 10% is very high")
        
        # Stop loss validation
        stop_loss = config.get('stop_loss_percentage', 0)
        if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
            result.add_error("stop_loss_percentage must be a positive number")
        elif stop_loss > 0.5:  # 50%
            result.add_warning("stop_loss_percentage above 50% defeats the purpose")
        
        # Take profit validation
        take_profit = config.get('take_profit_percentage', 0)
        if not isinstance(take_profit, (int, float)) or take_profit <= 0:
            result.add_error("take_profit_percentage must be a positive number")
        elif take_profit < stop_loss:
            result.add_error("take_profit_percentage should be higher than stop_loss_percentage")
        
        # Liquidity threshold validation
        min_liquidity = config.get('min_liquidity_threshold', 0)
        if not isinstance(min_liquidity, (int, float)) or min_liquidity <= 0:
            result.add_error("min_liquidity_threshold must be a positive number")
        elif min_liquidity < 10000:  # $10k
            result.add_warning("min_liquidity_threshold below $10k increases rug pull risk")
        
        # Gas price validation
        max_gas_price = config.get('max_gas_price', 0)
        if not isinstance(max_gas_price, int) or max_gas_price <= 0:
            result.add_error("max_gas_price must be a positive integer")
        elif max_gas_price > 500:  # 500 Gwei
            result.add_warning("max_gas_price above 500 Gwei is extremely high")
        
        # DEX validation
        enabled_dexes = config.get('enabled_dexes', [])
        if not isinstance(enabled_dexes, list) or not enabled_dexes:
            result.add_error("enabled_dexes must be a non-empty list")
        
        supported_dexes = ['uniswap_v2', 'uniswap_v3', 'pancakeswap', 'sushiswap']
        for dex in enabled_dexes:
            if dex not in supported_dexes:
                result.add_error(f"Unsupported DEX: {dex}")
        
        preferred_dex = config.get('preferred_dex', '')
        if preferred_dex and preferred_dex not in enabled_dexes:
            result.add_error("preferred_dex must be in enabled_dexes list")
    
    def _validate_security_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate security configuration"""
        
        # Rate limit validation
        api_rate_limit = config.get('api_rate_limit', 0)
        if not isinstance(api_rate_limit, int) or api_rate_limit <= 0:
            result.add_error("api_rate_limit must be a positive integer")
        elif api_rate_limit < 100:
            result.add_warning("api_rate_limit below 100 may be too restrictive")
        
        # Login attempts validation
        max_login_attempts = config.get('max_login_attempts', 0)
        if not isinstance(max_login_attempts, int) or max_login_attempts <= 0:
            result.add_error("max_login_attempts must be a positive integer")
        elif max_login_attempts > 10:
            result.add_warning("max_login_attempts above 10 reduces security")
        
        # Session timeout validation
        session_timeout = config.get('session_timeout', 0)
        if not isinstance(session_timeout, int) or session_timeout <= 0:
            result.add_error("session_timeout must be a positive integer")
        elif session_timeout > 86400:  # 24 hours
            result.add_warning("session_timeout above 24 hours is a security risk")
        elif session_timeout < 300:  # 5 minutes
            result.add_warning("session_timeout below 5 minutes may be too short")
        
        # 2FA validation
        require_2fa = config.get('require_2fa', False)
        if not isinstance(require_2fa, bool):
            result.add_error("require_2fa must be a boolean")
        elif not require_2fa:
            result.add_recommendation("Enable 2FA for enhanced security")
        
        # Key rotation validation
        key_rotation_days = config.get('encryption_key_rotation_days', 0)
        if not isinstance(key_rotation_days, int) or key_rotation_days <= 0:
            result.add_error("encryption_key_rotation_days must be a positive integer")
        elif key_rotation_days > 90:
            result.add_warning("Key rotation interval above 90 days reduces security")
        
        # Audit retention validation
        audit_retention = config.get('audit_log_retention_days', 0)
        if not isinstance(audit_retention, int) or audit_retention <= 0:
            result.add_error("audit_log_retention_days must be a positive integer")
        elif audit_retention < 90:
            result.add_warning("Audit log retention below 90 days may not meet compliance requirements")
        
        # Hardware wallet validation
        hardware_wallet = config.get('hardware_wallet_required', False)
        if not isinstance(hardware_wallet, bool):
            result.add_error("hardware_wallet_required must be a boolean")
        
        # Multisig validation
        multisig_threshold = config.get('multisig_threshold', 0)
        if not isinstance(multisig_threshold, int) or multisig_threshold < 0:
            result.add_error("multisig_threshold must be a non-negative integer")
        elif multisig_threshold > 5:
            result.add_warning("multisig_threshold above 5 may be impractical")
    
    def _validate_database_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate database configuration"""
        
        # Host validation
        host = config.get('host', '')
        if not host:
            result.add_error("Database host is required")
        elif not self._is_valid_hostname(host):
            result.add_error("Invalid database host format")
        
        # Port validation
        port = config.get('port', 0)
        if not isinstance(port, int) or not (1 <= port <= 65535):
            result.add_error("Database port must be between 1 and 65535")
        
        # Database name validation
        database = config.get('database', '')
        if not database:
            result.add_error("Database name is required")
        elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', database):
            result.add_error("Invalid database name format")
        
        # Username validation
        username = config.get('username', '')
        if not username:
            result.add_error("Database username is required")
        
        # Password validation
        password = config.get('password', '')
        if not password:
            result.add_error("Database password is required")
        elif len(password) < 8:
            result.add_warning("Database password should be at least 8 characters")
        
        # Pool size validation
        pool_size = config.get('pool_size', 0)
        if not isinstance(pool_size, int) or pool_size <= 0:
            result.add_error("pool_size must be a positive integer")
        elif pool_size > 100:
            result.add_warning("pool_size above 100 may consume too many resources")
        
        # Max overflow validation
        max_overflow = config.get('max_overflow', 0)
        if not isinstance(max_overflow, int) or max_overflow < 0:
            result.add_error("max_overflow must be a non-negative integer")
        
        # Pool timeout validation
        pool_timeout = config.get('pool_timeout', 0)
        if not isinstance(pool_timeout, int) or pool_timeout <= 0:
            result.add_error("pool_timeout must be a positive integer")
        elif pool_timeout > 300:  # 5 minutes
            result.add_warning("pool_timeout above 5 minutes may cause connection issues")
    
    def _validate_api_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate API configuration"""
        
        # Host validation
        host = config.get('host', '')
        if not host:
            result.add_error("API host is required")
        elif host not in ['0.0.0.0', 'localhost', '127.0.0.1'] and not self._is_valid_ip(host):
            result.add_warning("API host should typically be 0.0.0.0 for production")
        
        # Port validation
        port = config.get('port', 0)
        if not isinstance(port, int) or not (1024 <= port <= 65535):
            result.add_error("API port must be between 1024 and 65535")
        elif port < 8000:
            result.add_warning("API port below 8000 may require root privileges")
        
        # Debug validation
        debug = config.get('debug', False)
        if not isinstance(debug, bool):
            result.add_error("debug must be a boolean")
        elif debug:
            result.add_warning("Debug mode should not be enabled in production")
        
        # CORS origins validation
        cors_origins = config.get('cors_origins', [])
        if not isinstance(cors_origins, list):
            result.add_error("cors_origins must be a list")
        elif '*' in cors_origins and len(cors_origins) > 1:
            result.add_warning("CORS wildcard '*' should not be mixed with specific origins")
        elif '*' in cors_origins:
            result.add_warning("CORS wildcard '*' is a security risk in production")
        
        # JWT secret validation
        jwt_secret = config.get('jwt_secret', '')
        if not jwt_secret:
            result.add_error("JWT secret is required")
        elif len(jwt_secret) < 32:
            result.add_error("JWT secret should be at least 32 characters")
        
        # JWT expiry validation
        jwt_expiry = config.get('jwt_expiry', 0)
        if not isinstance(jwt_expiry, int) or jwt_expiry <= 0:
            result.add_error("jwt_expiry must be a positive integer")
        elif jwt_expiry > 86400:  # 24 hours
            result.add_warning("JWT expiry above 24 hours is a security risk")
        elif jwt_expiry < 300:  # 5 minutes
            result.add_warning("JWT expiry below 5 minutes may be too short")
        
        # Rate limiting validation
        rate_limit_requests = config.get('rate_limit_requests', 0)
        if not isinstance(rate_limit_requests, int) or rate_limit_requests <= 0:
            result.add_error("rate_limit_requests must be a positive integer")
        
        rate_limit_window = config.get('rate_limit_window', 0)
        if not isinstance(rate_limit_window, int) or rate_limit_window <= 0:
            result.add_error("rate_limit_window must be a positive integer")
    
    def _validate_monitoring_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate monitoring configuration"""
        
        # Metrics validation
        enable_metrics = config.get('enable_metrics', False)
        if not isinstance(enable_metrics, bool):
            result.add_error("enable_metrics must be a boolean")
        
        metrics_port = config.get('metrics_port', 0)
        if not isinstance(metrics_port, int) or not (1024 <= metrics_port <= 65535):
            result.add_error("metrics_port must be between 1024 and 65535")
        
        # Log level validation
        log_level = config.get('log_level', '')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            result.add_error(f"log_level must be one of: {valid_levels}")
        
        # Log format validation
        log_format = config.get('log_format', '')
        valid_formats = ['json', 'text', 'structured']
        if log_format not in valid_formats:
            result.add_error(f"log_format must be one of: {valid_formats}")
        
        # Log rotation validation
        log_rotation_size = config.get('log_rotation_size', '')
        if log_rotation_size and not re.match(r'^\d+[KMGT]B$', log_rotation_size):
            result.add_error("log_rotation_size must be in format like '100MB', '1GB'")
        
        # Log retention validation
        log_retention_days = config.get('log_retention_days', 0)
        if not isinstance(log_retention_days, int) or log_retention_days <= 0:
            result.add_error("log_retention_days must be a positive integer")
        elif log_retention_days < 7:
            result.add_warning("log_retention_days below 7 may not be sufficient")
        
        # Telegram validation
        telegram_token = config.get('telegram_bot_token')
        if telegram_token:
            if not re.match(r'^\d+:[A-Za-z0-9_-]{35}$', telegram_token):
                result.add_error("Invalid Telegram bot token format")
        
        telegram_chat_id = config.get('telegram_chat_id')
        if telegram_chat_id:
            if not re.match(r'^-?\d+$', str(telegram_chat_id)):
                result.add_error("Invalid Telegram chat ID format")
        
        # Discord validation
        discord_webhook = config.get('discord_webhook_url')
        if discord_webhook:
            if not self._is_valid_url(discord_webhook):
                result.add_error("Invalid Discord webhook URL")
            elif 'discord.com' not in discord_webhook:
                result.add_warning("Discord webhook URL should contain 'discord.com'")
        
        # Email validation
        email_server = config.get('email_smtp_server')
        if email_server:
            if not self._is_valid_hostname(email_server):
                result.add_error("Invalid email SMTP server")
        
        email_username = config.get('email_username')
        if email_username:
            if not self._is_valid_email(email_username):
                result.add_error("Invalid email username format")
    
    def _validate_ml_models_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate ML models configuration"""
        
        # Model update frequency validation
        update_frequency = config.get('model_update_frequency', 0)
        if not isinstance(update_frequency, int) or update_frequency <= 0:
            result.add_error("model_update_frequency must be a positive integer")
        elif update_frequency < 6:  # 6 hours
            result.add_warning("model_update_frequency below 6 hours may be too frequent")
        
        # Training samples validation
        min_samples = config.get('min_training_samples', 0)
        if not isinstance(min_samples, int) or min_samples <= 0:
            result.add_error("min_training_samples must be a positive integer")
        elif min_samples < 100:
            result.add_warning("min_training_samples below 100 may not be sufficient")
        
        # Lookback period validation
        lookback_period = config.get('lookback_period', 0)
        if not isinstance(lookback_period, int) or lookback_period <= 0:
            result.add_error("lookback_period must be a positive integer")
        elif lookback_period > 1000:
            result.add_warning("lookback_period above 1000 may slow down processing")
        
        # Feature selection threshold validation
        feature_threshold = config.get('feature_selection_threshold', 0)
        if not isinstance(feature_threshold, (int, float)) or not (0 <= feature_threshold <= 1):
            result.add_error("feature_selection_threshold must be between 0 and 1")
        
        # Train/test split validation
        train_test_split = config.get('train_test_split', 0)
        if not isinstance(train_test_split, (int, float)) or not (0.5 <= train_test_split <= 0.9):
            result.add_error("train_test_split must be between 0.5 and 0.9")
        
        # Validation split validation
        validation_split = config.get('validation_split', 0)
        if not isinstance(validation_split, (int, float)) or not (0.1 <= validation_split <= 0.5):
            result.add_error("validation_split must be between 0.1 and 0.5")
        
        # Early stopping patience validation
        early_stopping = config.get('early_stopping_patience', 0)
        if not isinstance(early_stopping, int) or early_stopping <= 0:
            result.add_error("early_stopping_patience must be a positive integer")
        
        # Threshold validations
        thresholds = ['rug_detection_threshold', 'pump_prediction_threshold', 'volume_validation_threshold']
        for threshold_name in thresholds:
            threshold = config.get(threshold_name, 0)
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                result.add_error(f"{threshold_name} must be between 0 and 1")
    
    def _validate_risk_management_config(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate risk management configuration"""
        
        # Portfolio risk validation
        max_portfolio_risk = config.get('max_portfolio_risk', 0)
        if not isinstance(max_portfolio_risk, (int, float)) or max_portfolio_risk <= 0:
            result.add_error("max_portfolio_risk must be a positive number")
        elif max_portfolio_risk > 0.1:  # 10%
            result.add_warning("max_portfolio_risk above 10% is very high")
        
        # Position correlation validation
        max_correlation = config.get('max_position_correlation', 0)
        if not isinstance(max_correlation, (int, float)) or not (0 <= max_correlation <= 1):
            result.add_error("max_position_correlation must be between 0 and 1")
        elif max_correlation > 0.8:
            result.add_warning("max_position_correlation above 0.8 reduces diversification")
        
        # Sector exposure validation
        max_sector_exposure = config.get('max_sector_exposure', 0)
        if not isinstance(max_sector_exposure, (int, float)) or not (0 <= max_sector_exposure <= 1):
            result.add_error("max_sector_exposure must be between 0 and 1")
        elif max_sector_exposure > 0.5:  # 50%
            result.add_warning("max_sector_exposure above 50% concentrates risk")
        
        # Position sizing method validation
        sizing_method = config.get('position_sizing_method', '')
        valid_methods = ['fixed', 'kelly', 'optimal_f']
        if sizing_method not in valid_methods:
            result.add_error(f"position_sizing_method must be one of: {valid_methods}")
        
        # Kelly criterion validation
        kelly_enabled = config.get('kelly_criterion_enabled', False)
        if not isinstance(kelly_enabled, bool):
            result.add_error("kelly_criterion_enabled must be a boolean")
        
        # ATR multiplier validation
        atr_multiplier = config.get('atr_multiplier', 0)
        if not isinstance(atr_multiplier, (int, float)) or atr_multiplier <= 0:
            result.add_error("atr_multiplier must be a positive number")
        elif atr_multiplier > 5:
            result.add_warning("atr_multiplier above 5 may be too loose")
        
        # Emergency stop drawdown validation
        emergency_drawdown = config.get('emergency_stop_drawdown', 0)
        if not isinstance(emergency_drawdown, (int, float)) or emergency_drawdown <= 0:
            result.add_error("emergency_stop_drawdown must be a positive number")
        elif emergency_drawdown > 0.5:  # 50%
            result.add_warning("emergency_stop_drawdown above 50% may be too high")
        elif emergency_drawdown < 0.1:  # 10%
            result.add_warning("emergency_stop_drawdown below 10% may trigger too often")
        
        # Consecutive losses validation
        max_losses = config.get('max_consecutive_losses', 0)
        if not isinstance(max_losses, int) or max_losses <= 0:
            result.add_error("max_consecutive_losses must be a positive integer")
        elif max_losses > 10:
            result.add_warning("max_consecutive_losses above 10 may be too high")
        elif max_losses < 3:
            result.add_warning("max_consecutive_losses below 3 may trigger too often")
    
    def _validate_cross_config(self, config_type: str, config: Dict[str, Any], result: ValidationResult) -> None:
        """Validate cross-configuration dependencies"""
        
        # Trading vs Risk Management consistency
        if config_type == 'trading':
            max_position = config.get('max_position_size', 0)
            stop_loss = config.get('stop_loss_percentage', 0)
            
            # Check if position size is consistent with risk tolerance
            if max_position > 0.1 and stop_loss < 0.02:  # 10% position, 2% stop loss
                result.add_warning("Large position size with tight stop loss may cause frequent exits")
        
        # Security vs API consistency
        if config_type == 'api':
            debug = config.get('debug', False)
            cors_origins = config.get('cors_origins', [])
            
            if debug and '*' in cors_origins:
                result.add_warning("Debug mode with CORS wildcard is a security risk")
        
        # ML Models vs Performance consistency
        if config_type == 'ml_models':
            update_freq = config.get('model_update_frequency', 0)
            enable_ensemble = config.get('enable_ensemble', False)
            enable_transformer = config.get('enable_transformer', False)
            
            if update_freq < 12 and enable_transformer:
                result.add_warning("Frequent updates with Transformer models may impact performance")
            
            if enable_ensemble and sum([
                config.get('enable_xgboost', False),
                config.get('enable_lightgbm', False),
                config.get('enable_lstm', False),
                config.get('enable_transformer', False)
            ]) < 2:
                result.add_warning("Ensemble requires at least 2 models to be effective")
    
    def _is_valid_hostname(self, hostname: str) -> bool:
        """Validate hostname format"""
        if not hostname or len(hostname) > 253:
            return False
        
        # Remove trailing dot
        if hostname.endswith('.'):
            hostname = hostname[:-1]
        
        # Check each label
        labels = hostname.split('.')
        for label in labels:
            if not label or len(label) > 63:
                return False
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$', label):
                return False
        
        return True
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
        return re.match(pattern, email) is not None
    
    def validate_ethereum_address(self, address: str) -> bool:
        """Validate Ethereum address format"""
        if not address or not isinstance(address, str):
            return False
        
        # Check if it's a valid hex address
        if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
            return False
        
        # Check if it's a valid checksum address (optional)
        try:
            return Web3.is_address(address)
        except Exception:
            return True  # Basic format is correct even if Web3 check fails
    
    def validate_private_key(self, private_key: str) -> bool:
        """Validate private key format"""
        if not private_key or not isinstance(private_key, str):
            return False
        
        # Remove 0x prefix if present
        if private_key.startswith('0x'):
            private_key = private_key[2:]
        
        # Check if it's a valid 64-character hex string
        return re.match(r'^[a-fA-F0-9]{64}$', private_key) is not None
    
    def validate_percentage(self, value: Any, min_val: float = 0, max_val: float = 1) -> bool:
        """Validate percentage value"""
        try:
            if isinstance(value, str):
                if value.endswith('%'):
                    value = float(value[:-1]) / 100
                else:
                    value = float(value)
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                return False
            
            return min_val <= value <= max_val
        except (ValueError, TypeError):
            return False
    
    def validate_decimal_string(self, value: str) -> bool:
        """Validate decimal string format"""
        try:
            Decimal(value)
            return True
        except (InvalidOperation, TypeError):
            return False
    
    def validate_json_config(self, config_str: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate JSON configuration string"""
        try:
            import json
            config = json.loads(config_str)
            if not isinstance(config, dict):
                return False, None, "Configuration must be a JSON object"
            return True, config, None
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Error parsing configuration: {str(e)}"
    
    def validate_yaml_config(self, config_str: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate YAML configuration string"""
        try:
            import yaml
            config = yaml.safe_load(config_str)
            if not isinstance(config, dict):
                return False, None, "Configuration must be a YAML object"
            return True, config, None
        except yaml.YAMLError as e:
            return False, None, f"Invalid YAML: {str(e)}"
        except Exception as e:
            return False, None, f"Error parsing configuration: {str(e)}"
    
    def generate_validation_report(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation report for all configurations"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_configs': len(configs),
            'valid_configs': 0,
            'configs_with_warnings': 0,
            'configs_with_errors': 0,
            'details': {},
            'summary': {
                'errors': [],
                'warnings': [],
                'recommendations': []
            }
        }
        
        for config_type, config_data in configs.items():
            result = self.validate_config(config_type, config_data)
            report['details'][config_type] = result.to_dict()
            
            if result.is_valid:
                report['valid_configs'] += 1
            else:
                report['configs_with_errors'] += 1
            
            if result.warnings:
                report['configs_with_warnings'] += 1
            
            # Aggregate summary
            report['summary']['errors'].extend([f"{config_type}: {error}" for error in result.errors])
            report['summary']['warnings'].extend([f"{config_type}: {warning}" for warning in result.warnings])
            report['summary']['recommendations'].extend([f"{config_type}: {rec}" for rec in result.recommendations])
        
        return report

    def validate_api_keys(self, keys: Dict) -> Dict:
        """Validate API keys"""
        result = {}
        
        for key_name, key_value in keys.items():
            if not key_value:
                result[key_name] = {'valid': False, 'error': 'Key is empty'}
            elif len(key_value) < 10:
                result[key_name] = {'valid': False, 'error': 'Key too short'}
            else:
                result[key_name] = {'valid': True}
        
        return result

    def check_required_fields(self, config: Dict) -> List[str]:
        """Check for required configuration fields"""
        missing_fields = []
        
        # Define required fields per config type
        required = {
            'database': ['host', 'port', 'database', 'username', 'password'],
            'api': ['host', 'port', 'jwt_secret'],
            'trading': ['max_position_size', 'stop_loss_percentage'],
            'security': ['api_rate_limit', 'session_timeout']
        }
        
        # Check all required fields
        for config_type, fields in required.items():
            type_config = config.get(config_type, {})
            for field in fields:
                if field not in type_config:
                    missing_fields.append(f"{config_type}.{field}")
        
        return missing_fields