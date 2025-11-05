#!/usr/bin/env python3
"""
DexScreener Trading Bot - Main Entry Point
Advanced automated trading system with ML-powered decision making
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import argparse
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import TradingBotEngine
from monitoring.logger import StructuredLogger  # Changed from setup_logger
from config.config_manager import ConfigManager  # Changed import
from config.validation import validate_config_at_startup
from data.storage.database import DatabaseManager
from security.encryption import EncryptionManager  # Changed from SecurityManager
from monitoring.enhanced_dashboard import DashboardEndpoints
from data.storage.cache import CacheManager
from core.portfolio_manager import PortfolioManager
from trading.orders.order_manager import OrderManager
from core.risk_manager import RiskManager
from monitoring.alerts import AlertsSystem
# Load environment variables
load_dotenv()

# Global logger
logger = None

# ============================================================================
# 🔧 PATCH 1: Multi-Chain Configuration Helper
# ADD THIS ENTIRE FUNCTION (around line 55)
# ============================================================================

def setup_multichain_config() -> Dict:
    """
    Setup multi-chain configuration from environment variables
    This enables the bot to discover opportunities across multiple blockchains
    """
    import os
    
    # Parse enabled chains from environment
    enabled_chains_str = os.getenv('ENABLED_CHAINS', 'ethereum,bsc,base')
    enabled_chains = [c.strip() for c in enabled_chains_str.split(',')]
    
    print(f"🌐 Enabled chains: {', '.join(enabled_chains)}")
    
    # Multi-chain configuration
    chains_config = {
        'enabled': enabled_chains,
        'default': os.getenv('DEFAULT_CHAIN', 'ethereum'),
        'max_pairs_per_chain': int(os.getenv('MAX_PAIRS_PER_CHAIN', '50')),
        'discovery_interval': int(os.getenv('DISCOVERY_INTERVAL_SECONDS', '300')),
    }
    
    # Chain-specific settings
    for chain in enabled_chains:
        chain_upper = chain.upper()
        chains_config[chain] = {
            'enabled': True,
            'min_liquidity': float(os.getenv(f'{chain_upper}_MIN_LIQUIDITY', '10000')),
            'min_volume': float(os.getenv(f'{chain_upper}_MIN_VOLUME', '5000')),
            'max_age_hours': int(os.getenv(f'{chain_upper}_MAX_AGE_HOURS', '24'))
        }
        print(f"  └─ {chain}: min_liq=${chains_config[chain]['min_liquidity']:,.0f}")
    
    return chains_config

# Add this function to main.py after imports:
def setup_logger(name: str, mode: str) -> logging.Logger:
    """Setup logger using StructuredLogger"""
    from monitoring.logger import StructuredLogger
    structured_logger = StructuredLogger(name, {'mode': mode})
    return logging.getLogger(name)

async def test_connection():
    """Test database connection"""
    db_manager = DatabaseManager({})
    await db_manager.connect()
    await db_manager.disconnect()
    
async def test_redis_connection():
    """Test Redis connection"""
    from data.storage.cache import CacheManager
    import os
    # Pass Redis config from environment
    redis_config = {
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379/0'),
        'REDIS_HOST': 'redis',
        'REDIS_PORT': 6379,
        'REDIS_DB': 0
    }
    cache = CacheManager(redis_config)
    await cache.connect()
    await cache.disconnect()

async def test_web3_connection():
    """Test Web3 connection"""
    import os
    import logging
    from web3 import Web3
    
    # Use module logger instead of global logger
    _logger = logging.getLogger(__name__)
    
    try:
        provider_url = os.getenv('WEB3_PROVIDER_URL')
        if not provider_url:
            _logger.warning("No WEB3_PROVIDER_URL configured")
            return True  # Skip check if not configured
        
        w3 = Web3(Web3.HTTPProvider(provider_url))
        is_connected = w3.is_connected()
        
        if is_connected:
            _logger.info(f"✅ Web3 connected to {provider_url}")
        else:
            _logger.warning(f"⚠️  Web3 connection failed to {provider_url}")
            
        return is_connected
        
    except Exception as e:
        _logger.error(f"Web3 connection error: {e}")
        return False  # Don't fail startup on Web3 error in development
    
async def test_api_connection():
    """Test DexScreener API connection"""
async def verify_models_loaded():
    """Verify ML models are loaded"""
    from ml.models.ensemble_model import EnsemblePredictor
    model = EnsemblePredictor("./models")
    # Basic check - will raise if models can't load
    
async def check_wallet_balance():
    """Check wallet balance"""
    from security.wallet_security import WalletSecurityManager
    wallet = WalletSecurityManager({})
    # Return mock balance for now
    return 1.0  # ETH
    
async def close_all_connections():
    """Close all database connections"""
    # Implementation needed based on your database setup
    pass

class HealthChecker:
    """Simple health checker for the system"""
    def __init__(self, engine):
        self.engine = engine
        
    async def monitor(self):
        """Monitor system health"""
        while True:
            try:
                # Basic health checks
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Health check error: {e}")

  
class TradingBotApplication:
    """Main application class for the trading bot"""
    
    def __init__(self, config_path: str = "config/settings.yaml", mode: str = "production"):
        """
        Initialize the trading bot application
        
        Args:
            config_path: Path to configuration file
            mode: Operating mode (development, testing, production)
        """
        self.mode = mode
        self.config_path = config_path
        self.engine = None
        self.health_checker = None
        self.shutdown_event = asyncio.Event()
        self.logger = setup_logger("TradingBot", mode)

        # Initialize ConfigManager but don't load configs yet
        self.config_manager = ConfigManager(config_dir='config')
        
        # These will be initialized in the async initialize() method
        self.config = None
        self.db_manager = None
        self.cache_manager = None
        self.portfolio_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.alerts_system = None
        self.dashboard = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 DexScreener Trading Bot Starting...")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Time: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)
            
            # Load configuration asynchronously
            self.logger.info("Loading configuration...")
            await self.config_manager.initialize()
            # NEW - convert to nested dict:
            raw_config = self.config_manager.load_config(self.mode)
            self.config = {}
            for config_type, config_obj in raw_config.items():
                if hasattr(config_obj, 'dict'):
                    self.config[config_type] = config_obj.dict()
                else:
                    self.config[config_type] = config_obj
            
            # Validate environment
            self._validate_environment()
            
            # Initialize components that need config
            from data.storage.cache import CacheManager
            from core.portfolio_manager import PortfolioManager
            from trading.orders.order_manager import OrderManager
            from core.risk_manager import RiskManager
            from monitoring.alerts import AlertsSystem
            
            self.db_manager = DatabaseManager(self.config.get('database', {}))
            self.cache_manager = CacheManager({
                'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379/0')
            })
            self.portfolio_manager = PortfolioManager(self.config.get('portfolio', {}))
            self.order_manager = OrderManager(self.config_manager)
            self.risk_manager = RiskManager(self.config_manager)
            self.alerts_system = AlertsSystem(self.config.get('notifications', {}))
            
            # Initialize security using EncryptionManager
            self.logger.info("Initializing security manager...")
            security_manager = EncryptionManager({})
            
            # Initialize database
            self.logger.info("Connecting to database...")
            await self.db_manager.connect()

            # Ensure all required config sections exist
            if 'portfolio' not in self.config:
                self.config['portfolio'] = {
                    'initial_balance': float(os.getenv('BACKTEST_INITIAL_BALANCE', '1.0')),
                    'max_positions': 40,
                    'max_position_size_pct': 0.1,
                    'max_risk_per_trade': 0.05,
                    'max_portfolio_risk': 0.25,
                    'min_position_size': float(os.getenv('MIN_TRADE_SIZE_USD', '100')),
                    'allocation_strategy': 'DYNAMIC',
                    'daily_loss_limit': float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10')) / 100,
                    'consecutive_losses_limit': 5,
                    'correlation_threshold': 0.7,
                    'rebalance_frequency': 'daily'
                }

            if 'risk_management' not in self.config:
                self.config['risk_management'] = {
                    'max_position_size': float(os.getenv('MAX_POSITION_SIZE_PERCENT', '5')) / 100,
                    'max_portfolio_risk': 0.02,
                    'stop_loss_percentage': 0.05,
                    'emergency_stop_drawdown': float(os.getenv('MAX_DRAWDOWN_PERCENT', '25')) / 100
                }

            # ============================================================================
            # 🔧 PATCH 2: Multi-Chain Data Sources Configuration
            # REPLACE the existing 'data_sources' block with this (around line 340)
            # ============================================================================
            
            if 'data_sources' not in self.config:
                # Setup multi-chain configuration
                chains_config = setup_multichain_config()
                
                self.config['data_sources'] = {
                    'dexscreener': {
                        'api_key': os.getenv('DEXSCREENER_API_KEY', ''),
                        'base_url': 'https://api.dexscreener.com',
                        'rate_limit': 300,
                        'chains': chains_config['enabled'],  # ✅ ADD THIS
                        'min_liquidity': 10000,
                        'min_volume': 5000,
                        'max_age_hours': 24,
                        'cache_duration': 60
                    },
                    'social': {
                        'twitter_api_key': os.getenv('TWITTER_API_KEY', ''),
                        'twitter_api_secret': os.getenv('TWITTER_API_SECRET', ''),
                        'enabled': bool(os.getenv('TWITTER_API_KEY'))
                    }
                }
                
                # ✅ ADD THIS: Store chains config at top level too
                self.config['chains'] = chains_config

                # Propagate the default config back to the config_manager
                self.config_manager['data_sources'] = self.config['data_sources']
                self.config_manager['chains'] = self.config['chains']

            if 'web3' not in self.config:
                self.config['web3'] = {
                    'provider_url': os.getenv('WEB3_PROVIDER_URL'),
                    'backup_providers': [
                        os.getenv('WEB3_BACKUP_PROVIDER_1'),
                        os.getenv('WEB3_BACKUP_PROVIDER_2')
                    ],
                    'chain_id': int(os.getenv('CHAIN_ID', '1')),
                    'gas_multiplier': float(os.getenv('PRIORITY_GAS_MULTIPLIER', '1.2')),
                    'max_gas_price': int(os.getenv('MAX_GAS_PRICE', '500'))
                }

            if 'trading' not in self.config:
                self.config['trading'] = {
                    'strategies': {
                        'momentum': {'enabled': True, 'lookback_period': 20},
                        'scalping': {'enabled': True, 'profit_target': 0.02},
                        'ai_strategy': {'enabled': False}
                    },
                    'min_opportunity_score': float(os.getenv('MIN_OPPORTUNITY_SCORE', '0.25')),  # ✅ From .env
                    'max_slippage': float(os.getenv('MAX_SLIPPAGE', '2')) / 100,
                    'max_gas_price': int(os.getenv('MAX_GAS_PRICE', '500'))
                }

            if 'notifications' not in self.config:
                from monitoring.alerts import AlertPriority, NotificationChannel
                
                self.config['notifications'] = {
                    'telegram': {
                        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                        'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                        'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN'))
                    },
                    'discord': {
                        'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
                        'enabled': bool(os.getenv('DISCORD_WEBHOOK_URL'))
                    },
                    # ✅ CORRECT - Dict mapping AlertPriority to list of NotificationChannel
                    'channel_priorities': {
                        AlertPriority.LOW: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.MEDIUM: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.HIGH: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.CRITICAL: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else []
                    },
                    # ✅ Also add priority_cooldowns to avoid the second error
                    'priority_cooldowns': {
                        AlertPriority.LOW: 300,      # 5 minutes
                        AlertPriority.MEDIUM: 60,    # 1 minute  
                        AlertPriority.HIGH: 10,      # 10 seconds
                        AlertPriority.CRITICAL: 0    # No cooldown
                    },
                    # Other alert settings
                    'enabled': True,
                    'max_retries': 3,
                    'retry_delay': 30,
                    'aggregation_enabled': True,
                    'aggregation_window': 60
                }

            if 'ml' not in self.config:
                self.config['ml'] = {
                    'retrain_interval_hours': int(os.getenv('ML_RETRAIN_INTERVAL_HOURS', '24')),
                    'min_confidence': float(os.getenv('ML_MIN_CONFIDENCE', '0.7'))
                }

            # After loading config
            trading_config = self.config_manager.get_trading_config()
            self.logger.info(f"📊 Min Opportunity Score: {trading_config.min_opportunity_score}")

            # Replace the entire "if 'security' not in self.config:" block with this:
            # ALWAYS set/override security config from environment
            encrypted_key = os.getenv('PRIVATE_KEY')
            encryption_key = os.getenv('ENCRYPTION_KEY')

            # Decrypt private key if encrypted
            decrypted_key = encrypted_key
            if encrypted_key and encrypted_key.startswith('gAAAAAB') and encryption_key:
                try:
                    from cryptography.fernet import Fernet
                    f = Fernet(encryption_key.encode())
                    decrypted_key = f.decrypt(encrypted_key.encode()).decode()
                    self.logger.info("✅ Successfully decrypted private key")
                except Exception as e:
                    self.logger.error(f"Failed to decrypt private key: {e}")
                    raise ValueError("Cannot decrypt PRIVATE_KEY - check ENCRYPTION_KEY")

            # Override or create security config
            if decrypted_key and not decrypted_key.startswith('0x'):
                decrypted_key = '0x' + decrypted_key

            self.config['security'] = {
                'encryption_key': encryption_key,
                'private_key': decrypted_key
            }
            self.logger.info(f"DEBUG: Set security config with private_key: {decrypted_key[:10] if decrypted_key else 'None'}...")

            # ADD THIS BLOCK to ensure the decrypted key is used everywhere
            if decrypted_key:
                self.config_manager['private_key'] = decrypted_key
                self.config_manager['security']['private_key'] = decrypted_key
                # Also update the flat_config that will be created next
                self.config['private_key'] = decrypted_key

            # Before creating engine, flatten security and web3 config:
            flat_config = self.config.copy()
            if 'security' in self.config:
                flat_config.update(self.config['security'])
            if 'web3' in self.config:
                flat_config.update(self.config['web3'])

            # Correctly parse DRY_RUN from environment, defaulting to True for safety
            dry_run_str = os.getenv('DRY_RUN', 'true')
            flat_config['dry_run'] = dry_run_str.strip().lower() in ('true', '1', 't')
            self.logger.info(f"DRY_RUN mode is {'ENABLED' if flat_config['dry_run'] else 'DISABLED'}")

            self.config_manager['dry_run'] = flat_config['dry_run']
            # ADD THIS - explicitly set it again to be safe:
            flat_config['private_key'] = self.config.get('security', {}).get('private_key')
            flat_config['encryption_key'] = self.config.get('security', {}).get('encryption_key')

            # Right after line 295 (after creating flat_config), add:
            self.logger.info(f"DEBUG: flat_config keys: {flat_config.keys()}")
            self.logger.info(f"DEBUG: flat_config has private_key: {'private_key' in flat_config}")
            if 'private_key' in flat_config:
                # Only show first 10 chars for security
                self.logger.info(f"DEBUG: private_key value: {str(flat_config.get('private_key'))[:10]}...")

            # ✅ ADD THIS: Parse enabled_chains from .env and add to config
            enabled_chains_str = os.getenv('ENABLED_CHAINS', 'ethereum,bsc,base,arbitrum,polygon')
            self.config['enabled_chains'] = enabled_chains_str
            self.config['chains'] = self.config.get('chains', {})
            self.config['chains']['enabled'] = [c.strip() for c in enabled_chains_str.split(',')]

            # Parse chain-specific liquidity settings
            self.config['chains']['ethereum'] = {'min_liquidity': int(os.getenv('ETHEREUM_MIN_LIQUIDITY', 10000))}
            self.config['chains']['bsc'] = {'min_liquidity': int(os.getenv('BSC_MIN_LIQUIDITY', 1000))}
            self.config['chains']['base'] = {'min_liquidity': int(os.getenv('BASE_MIN_LIQUIDITY', 5000))}
            self.config['chains']['arbitrum'] = {'min_liquidity': int(os.getenv('ARBITRUM_MIN_LIQUIDITY', 10000))}
            self.config['chains']['polygon'] = {'min_liquidity': int(os.getenv('POLYGON_MIN_LIQUIDITY', 1000))}

            self.config['chains']['max_pairs_per_chain'] = int(os.getenv('MAX_PAIRS_PER_CHAIN', 50))
            self.config['chains']['discovery_interval'] = int(os.getenv('DISCOVERY_INTERVAL_SECONDS', 300))

            self.logger.info("✅ Chain configuration loaded:")
            self.logger.info(f"  Enabled chains: {self.config['chains']['enabled']}")
            self.logger.info(f"  Chain settings: {self.config['chains']}")

            # ✅ ADD THIS ENTIRE SECTION HERE (BEFORE "Initialize trading engine"):
            # ============================================================================
            # 🔧 SOLANA CONFIGURATION - Add to config dict
            # ============================================================================

            # Check if Solana is enabled
            solana_enabled = os.getenv('SOLANA_ENABLED', 'false').lower() == 'true'

            if solana_enabled:
                self.logger.info("🟣 Configuring Solana integration...")
                
                # Decrypt Solana private key if needed
                solana_pk_encrypted = os.getenv('SOLANA_PRIVATE_KEY')
                solana_pk_decrypted = solana_pk_encrypted
                if solana_pk_encrypted and solana_pk_encrypted.startswith('gAAAAAB') and encryption_key:
                    try:
                        from cryptography.fernet import Fernet
                        f = Fernet(encryption_key.encode())
                        solana_pk_decrypted = f.decrypt(solana_pk_encrypted.encode()).decode()
                        self.logger.info("✅ Successfully decrypted Solana private key")
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt Solana private key: {e}")
                        # Continue with encrypted key, might fail later but won't crash here

                # Add Solana to flat config for TradingEngine
                self.config['solana_enabled'] = True
                self.config['solana_rpc_url'] = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
                self.config['solana_private_key'] = solana_pk_decrypted
                self.config['jupiter_max_slippage_bps'] = int(os.getenv('JUPITER_MAX_SLIPPAGE_BPS', '500'))
                self.config['solana_min_liquidity'] = float(os.getenv('SOLANA_MIN_LIQUIDITY', '5000'))
                
                # Also add as nested config (for consistency)
                self.config['solana'] = {
                    'enabled': True,
                    'rpc_url': os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
                    'private_key': os.getenv('SOLANA_PRIVATE_KEY'),
                    'max_slippage_bps': int(os.getenv('JUPITER_MAX_SLIPPAGE_BPS', '500')),
                    'min_liquidity': float(os.getenv('SOLANA_MIN_LIQUIDITY', '5000')),
                }
                
                # Add Solana to chains config
                if 'solana' not in self.config['chains']['enabled']:
                    self.config['chains']['enabled'].append('solana')
                
                self.config['chains']['solana'] = {
                    'enabled': True,
                    'min_liquidity': float(os.getenv('SOLANA_MIN_LIQUIDITY', '5000')),
                    'min_volume': float(os.getenv('SOLANA_MIN_VOLUME', '2500')),
                    'max_age_hours': int(os.getenv('SOLANA_MAX_AGE_HOURS', '24'))
                }
                
                self.logger.info("  ✅ Solana enabled")
                self.logger.info(f"     RPC: {self.config['solana_rpc_url'][:50]}...")
                self.logger.info(f"     Min Liquidity: ${self.config['solana_min_liquidity']:,.0f}")
                self.logger.info(f"     Max Slippage: {self.config['jupiter_max_slippage_bps']/100:.2f}%")
            else:
                self.logger.info("ℹ️  Solana integration disabled")
                self.config['solana_enabled'] = False


            # Initialize trading engine
            # Initialize trading engine
            self.logger.info("Initializing trading engine...")

            # ✅ CRITICAL: Merge Solana config into flat_config BEFORE creating engine
            if self.config.get('solana_enabled'):
                self.logger.info("🔧 Merging Solana config into flat_config...")
                solana_config = {
                    'solana_enabled': self.config['solana_enabled'],
                    'solana_rpc_url': self.config['solana_rpc_url'],
                    'solana_private_key': self.config['solana_private_key'],
                    'jupiter_max_slippage_bps': self.config.get('jupiter_max_slippage_bps', 500),
                    'solana': self.config.get('solana', {})
                }
                flat_config.update(solana_config)
                self.config_manager.configs.update(solana_config)
                self.logger.info(f"   ✅ Solana config merged: {list(flat_config.get('solana', {}).keys())}")

            self.engine = TradingBotEngine(self.config_manager, mode=self.mode)
            await self.engine.initialize()
            
            # Initialize dashboard
            self.dashboard = DashboardEndpoints(
                host="0.0.0.0",
                port=8080,
                config=self.config,
                trading_engine=self.engine,
                portfolio_manager=self.portfolio_manager,
                order_manager=self.order_manager,
                risk_manager=self.risk_manager,
                alerts_system=self.alerts_system,
                config_manager=self.config_manager,
                db_manager=self.db_manager
            )
            self.logger.info("Enhanced dashboard initialized")
            
            # Initialize health checker
            self.logger.info("Starting health monitoring...")
            self.health_checker = HealthChecker(self.engine)
            
            # Perform system checks
            await self._perform_system_checks()
            
            self.logger.info("✅ Initialization complete!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise
            
    def _validate_environment(self):
        """Validate required environment variables"""
        required_vars = [
            'DB_URL',
            'REDIS_URL',
            'WEB3_PROVIDER_URL',
            'PRIVATE_KEY',
            'WALLET_ADDRESS',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        # Optional API keys - log warnings but don't fail
        optional_vars = {
            'DEXSCREENER_API_KEY': 'DexScreener API (rate limits will be lower)',
            'GOPLUS_API_KEY': 'GoPlus security checks'
            #'TOKENSNIFFER_API_KEY': 'TokenSniffer analysis'
        }
        
        for var, description in optional_vars.items():
            if not os.getenv(var):
                self.logger.warning(f"Optional API key missing: {var} - {description}")


    # And add this method to TradingBotApplication class:
    async def _position_monitor(self):
        """Monitor positions separately to ensure it's running"""
        while not self.shutdown_event.is_set():
            try:
                if self.engine and self.engine.active_positions:
                    positions_info = []
                    for addr, pos in self.engine.active_positions.items():
                        positions_info.append(
                            f"{pos.get('token_symbol', 'TOKEN')}: "
                            f"P&L {pos.get('pnl_percentage', 0):.2f}%"
                        )
                    
                    if positions_info:
                        self.logger.info(f"📊 Active Positions: {', '.join(positions_info)}")
                
                await asyncio.sleep(30)  # Report every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)


    async def _perform_system_checks(self):
        """Perform pre-flight system checks"""
        checks = [
            ("Database connectivity", self._check_database),
            ("Redis connectivity", self._check_redis),
            ("Web3 connectivity", self._check_web3),
            ("ML models", self._check_models),
            ("Wallet balance", self._check_wallet)
        ]
        
        for check_name, check_func in checks:
            try:
                self.logger.info(f"Checking {check_name}...")
                await check_func()
                self.logger.info(f"✔ {check_name} OK")
            except Exception as e:
                self.logger.error(f"✗ {check_name} failed: {e}")
                if self.mode == "production":
                    raise
                    
    async def _check_database(self):
        """Check database connectivity"""
#        from data.storage.database import test_connection
        await test_connection()
        
    async def _check_redis(self):
        """Check Redis connectivity"""
#        from data.storage.cache import test_redis_connection
        await test_redis_connection()
        
    async def _check_web3(self):
        """Check Web3 connectivity"""
#        from trading.executors.direct_dex import test_web3_connection
        await test_web3_connection()
        
    async def _check_models(self):
        """Check ML models are loaded"""
#        from ml.models import verify_models_loaded
        await verify_models_loaded()
        
    async def _check_wallet(self):
        """Check wallet balance and permissions"""
#        from security.wallet_manager import check_wallet_balance
        balance = await check_wallet_balance()
        
        if self.mode == "production" and balance < 0.1:  # Minimum ETH required
            raise ValueError(f"Insufficient wallet balance: {balance} ETH")
            
    async def run(self):
        """Main application loop"""
        try:
#            logger.info("Starting DexScreener Trading Bot...")

            # To this:
            self.logger.info("Starting DexScreener Trading Bot...")
            # Initialize components
            await self.initialize()

            # Test DexScreener before starting
            await test_dex_collector(self.logger)  # Pass the logger          
            
            # Start the trading engine
            self.logger.info("🎯 Starting trading engine...")

            # Log multi-chain status
            if 'chains' in self.config:
                chains = self.config['chains']['enabled']
                self.logger.info(f"🌐 Multi-chain mode: {len(chains)} chains enabled")
                self.logger.info(f"   Chains: {', '.join(chains)}")
            else:
                self.logger.warning("⚠️ Single-chain mode (multi-chain not configured)")

            
            # Create main tasks
            tasks = [
                asyncio.create_task(self.engine.run()),
                asyncio.create_task(self.health_checker.monitor()),
                asyncio.create_task(self._status_reporter()),
                asyncio.create_task(self.dashboard.start()),
                asyncio.create_task(self._shutdown_monitor()),
                asyncio.create_task(self._position_monitor())  # Ensure positions are monitored
                
            ]
            
            # Wait for shutdown or error
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            # Check for exceptions
            for task in done:
                if task.exception():
                    self.logger.error(f"Task failed: {task.exception()}")
                    
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}", exc_info=True)
            
        finally:
            await self.shutdown()
            
    async def _status_reporter(self):
        """Periodically report system status"""
        while not self.shutdown_event.is_set():
            try:
                if self.engine:
                    stats = await self.engine.get_stats()
                    self.logger.info(f"📊 Status: {stats}")
                    
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                self.logger.error(f"Error in status reporter: {e}")
                await asyncio.sleep(60)
                
    async def _shutdown_monitor(self):
        """Monitor for shutdown signal"""
        await self.shutdown_event.wait()
        self.logger.info("Shutdown signal received")
        
    async def shutdown(self):
        """Graceful shutdown procedure"""
        try:
            self.logger.info("Initiating graceful shutdown...")
            
            if self.engine:
                # Close all positions if configured
                if self.mode == "production":
                    self.logger.info("Closing all open positions...")
                    await self.engine.emergency_close_all_positions()
                    
                # Save state
                self.logger.info("Saving system state...")
                await self.engine.save_state()
                
                # Shutdown engine
                await self.engine.shutdown()
                
            # Close database connections
            await close_all_connections()
            
            self.logger.info("✅ Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Rest of the file remains the same...
            
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DexScreener Trading Bot - Advanced Automated Trading System"
    )
    
    parser.add_argument(
        '--mode',
        choices=['development', 'testing', 'production'],
        default='production',
        help='Operating mode'
    )
    
    parser.add_argument(
        '--config',
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in simulation mode without real trades'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode'
    )
    
    parser.add_argument(
        '--train-models',
        action='store_true',
        help='Train ML models before starting'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

async def test_dex_collector(logger_instance):
    """Test DexScreener connectivity"""
    from data.collectors.dexscreener import DexScreenerCollector
    import os

    logger_instance.info("Testing DexScreener API connection...")

    # Read enabled chains from environment, default to 'ethereum'
    enabled_chains_str = os.getenv('ENABLED_CHAINS', 'ethereum')
    enabled_chains = [c.strip() for c in enabled_chains_str.split(',')]
    test_chain = enabled_chains[0] if enabled_chains else 'ethereum'

    logger_instance.info(f"Using chain '{test_chain}' for DexScreener test...")

    collector = DexScreenerCollector({'api_key': '', 'chains': [test_chain]})
    await collector.initialize()

    success = False
    try:
        # Try to fetch some data from the specified chain
        pairs = await collector.get_new_pairs(chain=test_chain, limit=1)
        if pairs:
            logger_instance.info("DexScreener API connection successful, data received.")
            success = True
        else:
            # This is not necessarily a failure, could be no new pairs.
            # We consider it a success if the call completes without error.
            logger_instance.info("DexScreener API call successful, but no new pairs returned for this chain.")
            success = True

    except Exception as e:
        logger_instance.error(f"Error during DexScreener API test: {e}", exc_info=True)
        success = False

    finally:
        await collector.close()

    if not success:
        logger_instance.warning("⚠️ DexScreener API connection failed - bot will not find new pairs!")
    else:
        logger_instance.info("✅ DexScreener API connection successful")
    
    return success

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set environment variables based on arguments
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
        
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
    # Handle special modes
    if args.backtest:
        from scripts.backtest import run_backtest
        await run_backtest(args.config)
        return
        
    if args.train_models:
        from scripts.train_models import train_all_models
        await train_all_models()
        return

    # ========================================================================
    # NEW: VALIDATE CONFIGURATION BEFORE STARTING BOT
    # ========================================================================
    from config.validation import validate_config_at_startup
    from config.config_manager import ConfigManager
    
    try:
        # Load configuration
        config = ConfigManager(config_path=args.config)
        
        # Validate configuration (will raise ValueError if invalid)
        validate_config_at_startup(config)
        
    except ValueError as e:
        print(f"❌ Configuration validation failed:")
        print(f"   {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        sys.exit(1)
    # ========================================================================
    

    # Create and run application
    app = TradingBotApplication(
        config_path=args.config,
        mode=args.mode
    )
    
    try:
        await app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)
        
    # Run the application
    asyncio.run(main())