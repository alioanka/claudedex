"""
Global Settings and Constants for DexScreener Trading Bot
Centralized settings management with environment-aware configurations
"""

import os
from typing import Dict, List, Optional, Any
from decimal import Decimal
from pathlib import Path
from enum import Enum

class Environment(Enum):
    """Runtime environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ChainConfig:
    """Blockchain configuration"""
    def __init__(self, name: str, chain_id: int, rpc_url: str, 
                 explorer_url: str, native_token: str, is_testnet: bool = False):
        self.name = name
        self.chain_id = chain_id
        self.rpc_url = rpc_url
        self.explorer_url = explorer_url
        self.native_token = native_token
        self.is_testnet = is_testnet

class Settings:
    """Global application settings"""
    
    # Environment
    ENVIRONMENT = Environment(os.getenv('ENVIRONMENT', 'development'))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Application
    APP_NAME = "DexScreener Trading Bot"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Advanced automated cryptocurrency trading bot"
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    BACKUPS_DIR = BASE_DIR / "backups"
    
    # Create directories if they don't exist
    for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR, MODELS_DIR, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', None)
    JWT_SECRET = os.getenv('JWT_SECRET', 'jwt-secret-key')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/trading_bot')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # API Keys
    DEXSCREENER_API_KEY = os.getenv('DEXSCREENER_API_KEY', '')
    MORALIS_API_KEY = os.getenv('MORALIS_API_KEY', '')
    ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', '')
    INFURA_API_KEY = os.getenv('INFURA_API_KEY', '')
    QUICKNODE_API_KEY = os.getenv('QUICKNODE_API_KEY', '')
    
    # Social APIs
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    
    # Wallet Configuration
    DEFAULT_WALLET_PASSWORD = os.getenv('WALLET_PASSWORD', '')
    HARDWARE_WALLET_ENABLED = os.getenv('HARDWARE_WALLET_ENABLED', 'false').lower() == 'true'
    
    # Trading Limits (Safety Defaults)
    MAX_POSITION_SIZE = Decimal(os.getenv('MAX_POSITION_SIZE', '0.05'))  # 5%
    MAX_DAILY_VOLUME = Decimal(os.getenv('MAX_DAILY_VOLUME', '100'))     # $100
    MAX_SLIPPAGE = Decimal(os.getenv('MAX_SLIPPAGE', '0.02'))            # 2%
    MIN_LIQUIDITY = Decimal(os.getenv('MIN_LIQUIDITY', '50000'))         # $50k
    
    # Risk Management
    EMERGENCY_STOP_DRAWDOWN = Decimal(os.getenv('EMERGENCY_STOP_DRAWDOWN', '0.20'))  # 20%
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '5'))
    PORTFOLIO_VAR_LIMIT = Decimal(os.getenv('PORTFOLIO_VAR_LIMIT', '0.02'))  # 2%
    
    # ML Models
    MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', '24'))  # hours
    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))
    FEATURE_LOOKBACK_PERIOD = int(os.getenv('FEATURE_LOOKBACK_PERIOD', '100'))
    
    # Rate Limiting
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '1000'))  # per hour
    WEB3_RATE_LIMIT = int(os.getenv('WEB3_RATE_LIMIT', '100'))  # per minute
    
    # Monitoring
    METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))
    HEALTH_CHECK_PORT = int(os.getenv('HEALTH_CHECK_PORT', '8080'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')
    LOG_ROTATION_SIZE = os.getenv('LOG_ROTATION_SIZE', '100MB')
    LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '30'))
    
    # Performance
    WORKER_THREADS = int(os.getenv('WORKER_THREADS', '4'))
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '100'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # seconds
    
    # Cache
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes default
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '10000'))
    
    # Blockchain Networks
    SUPPORTED_CHAINS: Dict[str, ChainConfig] = {
        'ethereum': ChainConfig(
            name='Ethereum',
            chain_id=1,
            rpc_url=os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.alchemyapi.io/v2/your-api-key'),
            explorer_url='https://etherscan.io',
            native_token='ETH'
        ),
        'bsc': ChainConfig(
            name='Binance Smart Chain',
            chain_id=56,
            rpc_url=os.getenv('BSC_RPC_URL', 'https://bsc-dataseed1.binance.org'),
            explorer_url='https://bscscan.com',
            native_token='BNB'
        ),
        'polygon': ChainConfig(
            name='Polygon',
            chain_id=137,
            rpc_url=os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),
            explorer_url='https://polygonscan.com',
            native_token='MATIC'
        ),
        'arbitrum': ChainConfig(
            name='Arbitrum One',
            chain_id=42161,
            rpc_url=os.getenv('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),
            explorer_url='https://arbiscan.io',
            native_token='ETH'
        ),
        'avalanche': ChainConfig(
            name='Avalanche',
            chain_id=43114,
            rpc_url=os.getenv('AVALANCHE_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc'),
            explorer_url='https://snowtrace.io',
            native_token='AVAX'
        ),
        'fantom': ChainConfig(
            name='Fantom',
            chain_id=250,
            rpc_url=os.getenv('FANTOM_RPC_URL', 'https://rpc.fantom.network'),
            explorer_url='https://ftmscan.com',
            native_token='FTM'
        ),
        'base': ChainConfig(
            name='Base',
            chain_id=8453,
            rpc_url=os.getenv('BASE_RPC_URL', 'https://mainnet.base.org'),
            explorer_url='https://basescan.org',
            native_token='ETH'
        )
    }
    
    # Testnet Chains (for development)
    if ENVIRONMENT in [Environment.DEVELOPMENT, Environment.TESTING]:
        SUPPORTED_CHAINS.update({
            'goerli': ChainConfig(
                name='Ethereum Goerli',
                chain_id=5,
                rpc_url=os.getenv('GOERLI_RPC_URL', 'https://eth-goerli.alchemyapi.io/v2/your-api-key'),
                explorer_url='https://goerli.etherscan.io',
                native_token='ETH',
                is_testnet=True
            ),
            'bsc_testnet': ChainConfig(
                name='BSC Testnet',
                chain_id=97,
                rpc_url=os.getenv('BSC_TESTNET_RPC_URL', 'https://data-seed-prebsc-1-s1.binance.org:8545'),
                explorer_url='https://testnet.bscscan.com',
                native_token='BNB',
                is_testnet=True
            )
        })
    
    # DEX Configurations
    SUPPORTED_DEXES = {
        'uniswap_v2': {
            'name': 'Uniswap V2',
            'router_address': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'factory_address': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
            'chains': ['ethereum', 'goerli'],
            'fee': 0.003  # 0.3%
        },
        'uniswap_v3': {
            'name': 'Uniswap V3',
            'router_address': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'factory_address': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'chains': ['ethereum', 'polygon', 'arbitrum', 'goerli'],
            'fee_tiers': [0.0005, 0.003, 0.01]  # 0.05%, 0.3%, 1%
        },
        'pancakeswap': {
            'name': 'PancakeSwap',
            'router_address': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
            'factory_address': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73',
            'chains': ['bsc', 'bsc_testnet'],
            'fee': 0.0025  # 0.25%
        },
        'sushiswap': {
            'name': 'SushiSwap',
            'router_address': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
            'factory_address': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
            'chains': ['ethereum', 'polygon', 'arbitrum', 'fantom'],
            'fee': 0.003  # 0.3%
        }
    }
    
    # Token Lists
    STABLECOIN_ADDRESSES = {
        'ethereum': {
            'USDC': '0xA0b86a33E6441c8fb7e61b9e5E4F8b23C3b7b54c',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'BUSD': '0x4Fabb145d64652a948d72533023f6E7A623C7C53'
        },
        'bsc': {
            'USDC': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
            'USDT': '0x55d398326f99059fF775485246999027B3197955',
            'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',
            'DAI': '0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3'
        },
        'polygon': {
            'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
            'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
            'DAI': '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063'
        }
    }
    
    # Blacklisted Tokens (Known scams/rugs)
    BLACKLISTED_TOKENS = set([
        '0x0000000000000000000000000000000000000000',  # Null address
        # Add known scam tokens here
    ])
    
    # Gas Configuration
    GAS_CONFIG = {
        'ethereum': {
            'slow': {'max_fee': 20, 'max_priority_fee': 1},
            'standard': {'max_fee': 25, 'max_priority_fee': 2},
            'fast': {'max_fee': 30, 'max_priority_fee': 3},
            'urgent': {'max_fee': 50, 'max_priority_fee': 5}
        },
        'bsc': {
            'slow': {'gas_price': 5},
            'standard': {'gas_price': 10},
            'fast': {'gas_price': 15},
            'urgent': {'gas_price': 20}
        },
        'polygon': {
            'slow': {'max_fee': 40, 'max_priority_fee': 30},
            'standard': {'max_fee': 50, 'max_priority_fee': 35},
            'fast': {'max_fee': 60, 'max_priority_fee': 40},
            'urgent': {'max_fee': 100, 'max_priority_fee': 50}
        }
    }
    
    # Feature Flags (Environment-dependent)
    FEATURE_FLAGS = {
        'development': {
            'enable_paper_trading': True,
            'enable_testnet': True,
            'enable_debug_logging': True,
            'enable_hot_reload': True,
            'enable_profiling': True,
            'skip_wallet_verification': True,
            'mock_external_apis': True
        },
        'testing': {
            'enable_paper_trading': True,
            'enable_testnet': True,
            'enable_debug_logging': True,
            'enable_hot_reload': False,
            'enable_profiling': False,
            'skip_wallet_verification': True,
            'mock_external_apis': True
        },
        'staging': {
            'enable_paper_trading': True,
            'enable_testnet': False,
            'enable_debug_logging': False,
            'enable_hot_reload': False,
            'enable_profiling': False,
            'skip_wallet_verification': False,
            'mock_external_apis': False
        },
        'production': {
            'enable_paper_trading': False,
            'enable_testnet': False,
            'enable_debug_logging': False,
            'enable_hot_reload': False,
            'enable_profiling': False,
            'skip_wallet_verification': False,
            'mock_external_apis': False
        }
    }
    
    # Default Trading Strategy Parameters
    STRATEGY_DEFAULTS = {
        'momentum': {
            'lookback_period': 20,
            'momentum_threshold': 0.05,
            'volume_threshold': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        },
        'scalping': {
            'profit_target': 0.02,  # 2%
            'stop_loss': 0.01,      # 1%
            'holding_time': 300,    # 5 minutes
            'min_spread': 0.001     # 0.1%
        },
        'ai_strategy': {
            'confidence_threshold': 0.7,
            'prediction_horizon': 24,  # hours
            'feature_importance_threshold': 0.1,
            'ensemble_weight_decay': 0.95
        }
    }
    
    # ML Model Configurations
    ML_MODEL_CONFIG = {
        'rug_detector': {
            'threshold': 0.7,
            'features': [
                'liquidity_ratio', 'holder_count', 'transaction_count',
                'price_volatility', 'volume_consistency', 'dev_wallet_percentage',
                'contract_verification', 'social_sentiment', 'whale_concentration'
            ],
            'model_types': ['xgboost', 'lightgbm', 'random_forest']
        },
        'pump_predictor': {
            'threshold': 0.8,
            'lookback_hours': 24,
            'prediction_horizon': 4,  # hours
            'features': [
                'price_momentum', 'volume_surge', 'social_buzz',
                'whale_activity', 'technical_indicators', 'market_regime'
            ],
            'model_types': ['lstm', 'transformer', 'xgboost']
        },
        'volume_validator': {
            'threshold': 0.6,
            'min_transactions': 10,
            'time_window': 300,  # 5 minutes
            'wash_trading_indicators': [
                'round_number_trading', 'ping_pong_pattern',
                'same_wallet_frequency', 'unrealistic_spread'
            ]
        }
    }
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        'high_profit': 0.20,      # 20% profit
        'high_loss': -0.10,       # 10% loss
        'unusual_volume': 5.0,    # 5x average volume
        'price_spike': 0.50,      # 50% price increase
        'rug_pull_risk': 0.8,     # High rug pull probability
        'liquidity_removal': 0.3, # 30% liquidity removed
        'whale_transaction': 100000,  # $100k transaction
        'gas_price_spike': 100    # 100 Gwei
    }
    
    # Compliance Settings
    COMPLIANCE_CONFIG = {
        'max_transaction_amount': 10000,  # $10k (KYC threshold)
        'daily_volume_limit': 50000,      # $50k daily
        'monthly_volume_limit': 1000000,  # $1M monthly
        'suspicious_pattern_threshold': 5,
        'report_large_transactions': True,
        'geo_restrictions': [],  # List of restricted countries
        'sanctioned_addresses': set()  # OFAC sanctioned addresses
    }
    
    # Performance Monitoring
    PERFORMANCE_METRICS = {
        'latency_thresholds': {
            'api_response': 1000,      # 1 second
            'trade_execution': 5000,   # 5 seconds
            'data_processing': 10000,  # 10 seconds
            'ml_prediction': 30000     # 30 seconds
        },
        'success_rate_thresholds': {
            'trade_execution': 0.95,   # 95%
            'data_collection': 0.99,   # 99%
            'api_calls': 0.98,         # 98%
            'ml_predictions': 0.80     # 80%
        }
    }
    
    @classmethod
    def get_current_features(cls) -> Dict[str, bool]:
        """Get feature flags for current environment"""
        return cls.FEATURE_FLAGS.get(cls.ENVIRONMENT.value, cls.FEATURE_FLAGS['production'])
    
    @classmethod
    def is_feature_enabled(cls, feature: str) -> bool:
        """Check if a feature is enabled in current environment"""
        features = cls.get_current_features()
        return features.get(feature, False)
    
    @classmethod
    def get_chain_config(cls, chain_name: str) -> Optional[ChainConfig]:
        """Get configuration for specific blockchain"""
        return cls.SUPPORTED_CHAINS.get(chain_name)
    
    @classmethod
    def get_dex_config(cls, dex_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific DEX"""
        return cls.SUPPORTED_DEXES.get(dex_name)
    
    @classmethod
    def get_gas_config(cls, chain_name: str, speed: str = 'standard') -> Dict[str, Any]:
        """Get gas configuration for chain and speed"""
        chain_gas = cls.GAS_CONFIG.get(chain_name, {})
        return chain_gas.get(speed, chain_gas.get('standard', {}))
    
    @classmethod
    def is_token_blacklisted(cls, token_address: str) -> bool:
        """Check if token is blacklisted"""
        return token_address.lower() in {addr.lower() for addr in cls.BLACKLISTED_TOKENS}
    
    @classmethod
    def get_stablecoin_address(cls, chain: str, symbol: str) -> Optional[str]:
        """Get stablecoin address for specific chain"""
        return cls.STABLECOIN_ADDRESSES.get(chain, {}).get(symbol)
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate current environment configuration"""
        required_vars = []
        
        if cls.ENVIRONMENT == Environment.PRODUCTION:
            required_vars = [
                'SECRET_KEY', 'ENCRYPTION_KEY', 'DATABASE_URL',
                'DEXSCREENER_API_KEY', 'WALLET_PASSWORD'
            ]
        elif cls.ENVIRONMENT == Environment.STAGING:
            required_vars = [
                'SECRET_KEY', 'DATABASE_URL', 'DEXSCREENER_API_KEY'
            ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            'environment': cls.ENVIRONMENT.value,
            'debug': cls.DEBUG,
            'app_name': cls.APP_NAME,
            'app_version': cls.APP_VERSION,
            'supported_chains': list(cls.SUPPORTED_CHAINS.keys()),
            'supported_dexes': list(cls.SUPPORTED_DEXES.keys()),
            'feature_flags': cls.get_current_features(),
            'directories': {
                'base': str(cls.BASE_DIR),
                'config': str(cls.CONFIG_DIR),
                'data': str(cls.DATA_DIR),
                'logs': str(cls.LOGS_DIR),
                'models': str(cls.MODELS_DIR),
                'backups': str(cls.BACKUPS_DIR)
            }
        }

    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL - wrapper for documented signature"""
        return cls.DATABASE_URL

    @classmethod  
    def get_redis_url(cls) -> str:
        """Get Redis URL - wrapper for documented signature"""
        return cls.REDIS_URL

# Validate environment on import
try:
    Settings.validate_environment()
except ValueError as e:
    if Settings.ENVIRONMENT == Environment.PRODUCTION:
        raise e
    else:
        print(f"Warning: {e}")