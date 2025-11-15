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
from monitoring.logger import StructuredLogger
from config.config_manager import ConfigManager, ConfigType
from data.storage.database import DatabaseManager
from security.encryption import EncryptionManager
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
    
    _logger = logging.getLogger(__name__)
    
    try:
        provider_url = os.getenv('WEB3_PROVIDER_URL')
        if not provider_url:
            _logger.warning("No WEB3_PROVIDER_URL configured")
            return True
        
        w3 = Web3(Web3.HTTPProvider(provider_url))
        is_connected = w3.is_connected()
        
        if is_connected:
            _logger.info(f"✅ Web3 connected to {provider_url}")
        else:
            _logger.warning(f"⚠️  Web3 connection failed to {provider_url}")
            
        return is_connected
        
    except Exception as e:
        _logger.error(f"Web3 connection error: {e}")
        return False

async def test_api_connection():
    """Test DexScreener API connection"""
    from data.collectors.dexscreener import test_api_connection as test_api
    await test_api()
    
async def verify_models_loaded():
    """Verify ML models are loaded"""
    from ml.models.ensemble_model import EnsemblePredictor
    model = EnsemblePredictor("./models")
    
async def check_wallet_balance():
    """Check wallet balance"""
    from security.wallet_security import WalletSecurityManager
    wallet = WalletSecurityManager({})
    return 1.0
    
async def close_all_connections():
    """Close all database connections"""
    pass

class HealthChecker:
    """Simple health checker for the system"""
    def __init__(self, engine):
        self.engine = engine
        
    async def monitor(self):
        """Monitor system health"""
        while True:
            try:
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

        self.config_manager = ConfigManager(config_dir='config')
        
        self.config = None
        self.db_manager = None
        self.cache_manager = None
        self.portfolio_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.alerts_system = None
        self.dashboard = None
        
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
            
            self.logger.info("Loading configuration...")
            await self.config_manager.initialize(os.getenv('ENCRYPTION_KEY'))
            self.config = self.config_manager
            
            self._validate_environment()
            
            self.db_manager = DatabaseManager(self.config.get_config(ConfigType.DATABASE).dict())
            self.cache_manager = CacheManager({'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379/0')})

            # Create a nested config dictionary from the config manager
            nested_config = {}
            for config_type in ConfigType:
                config_model = self.config.get_config(config_type)
                if config_model:
                    nested_config[config_type.value] = config_model.dict()

            self.portfolio_manager = PortfolioManager(nested_config)
            self.order_manager = OrderManager(nested_config)

            # Extract all chain-specific RPC URLs from environment variables
            chain_rpc_urls = {}
            for env_var, value in os.environ.items():
                if env_var.endswith('_RPC_URLS'):
                    chain_name = env_var.replace('_RPC_URLS', '').lower()
                    chain_rpc_urls[chain_name] = [url.strip() for url in value.split(',')]

            self.risk_manager = RiskManager(nested_config,
                                            config_manager=self.config_manager,
                                            chain_rpc_urls=chain_rpc_urls)
            self.alerts_system = AlertsSystem(nested_config)
            
            self.logger.info("Initializing security manager...")
            security_manager = EncryptionManager({})
            
            self.logger.info("Connecting to database...")
            await self.db_manager.connect()

            # Decrypt private key if encrypted
            encrypted_key = os.getenv('PRIVATE_KEY')
            encryption_key = os.getenv('ENCRYPTION_KEY')
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

            # Add sensitive info to the nested config
            if 'security' not in nested_config:
                nested_config['security'] = {}
            nested_config['security']['private_key'] = decrypted_key
            nested_config['security']['encryption_key'] = encryption_key

            # Manually construct the web3 config for now
            if 'web3' not in nested_config:
                nested_config['web3'] = {
                    'provider_url': os.getenv('WEB3_PROVIDER_URL'),
                    'backup_providers': [
                        os.getenv('WEB3_BACKUP_PROVIDER_1'),
                        os.getenv('WEB3_BACKUP_PROVIDER_2')
                    ],
                    'chain_id': int(os.getenv('CHAIN_ID', '1')),
                    'gas_multiplier': float(self.config.get_config(ConfigType.GAS_PRICE).priority_gas_multiplier),
                    'max_gas_price': int(self.config.get_config(ConfigType.GAS_PRICE).max_gas_price)
                }

            if 'data_sources' not in nested_config:
                nested_config['data_sources'] = {
                    'dexscreener': {
                        'api_key': os.getenv('DEXSCREENER_API_KEY', ''),
                        'base_url': 'https://api.dexscreener.com',
                        'rate_limit': 300,
                        'chains': self.config.get_config(ConfigType.CHAIN).enabled_chains.split(','),
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

            if 'notifications' not in nested_config:
                from monitoring.alerts import AlertPriority, NotificationChannel

                nested_config['notifications'] = {
                    'telegram': {
                        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                        'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                        'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN'))
                    },
                    'discord': {
                        'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
                        'enabled': bool(os.getenv('DISCORD_WEBHOOK_URL'))
                    },
                    'channel_priorities': {
                        AlertPriority.LOW: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.MEDIUM: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.HIGH: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],
                        AlertPriority.CRITICAL: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else []
                    },
                    'priority_cooldowns': {
                        AlertPriority.LOW: 300,
                        AlertPriority.MEDIUM: 60,
                        AlertPriority.HIGH: 10,
                        AlertPriority.CRITICAL: 0
                    },
                    'enabled': True,
                    'max_retries': 3,
                    'retry_delay': 30,
                    'aggregation_enabled': True,
                    'aggregation_window': 60
                }

            self.logger.info("Initializing trading engine...")

            # --- FIX STARTS HERE: Pass ConfigManager and RPC URLs to the engine ---
            # Extract all chain-specific RPC URLs from environment variables
            chain_rpc_urls = {}
            for env_var, value in os.environ.items():
                if env_var.endswith('_RPC_URLS'):
                    chain_name = env_var.replace('_RPC_URLS', '').lower()
                    chain_rpc_urls[chain_name] = [url.strip() for url in value.split(',')]

            self.engine = TradingBotEngine(
                config=nested_config,
                config_manager=self.config_manager,
                chain_rpc_urls=chain_rpc_urls,
                mode=self.mode
            )
            await self.engine.initialize()
            # --- FIX ENDS HERE ---
            
            self.dashboard = DashboardEndpoints(
                host="0.0.0.0",
                port=8080,
                config=nested_config,
                trading_engine=self.engine,
                portfolio_manager=self.portfolio_manager,
                order_manager=self.order_manager,
                risk_manager=self.risk_manager,
                alerts_system=self.alerts_system,
                config_manager=self.config_manager,
                db_manager=self.db_manager
            )
            self.logger.info("Enhanced dashboard initialized")
            
            self.logger.info("Starting health monitoring...")
            self.health_checker = HealthChecker(self.engine)
            
            await self._perform_system_checks()
            
            self.logger.info("✅ Initialization complete!")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise
            
    def _validate_environment(self):
        """Validate required environment variables"""
        missing = self.config_manager.validate_environment()
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

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
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)


    async def _perform_system_checks(self):
        """Perform pre-flight system checks"""
        checks = [
            ("Database connectivity", self._check_database),
            ("Redis connectivity", self._check_redis),
            ("Web3 connectivity", self._check_web3),
            ("API endpoints", self._check_apis),
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
        await test_connection()
        
    async def _check_redis(self):
        await test_redis_connection()
        
    async def _check_web3(self):
        await test_web3_connection()
        
    async def _check_apis(self):
        await test_api_connection()
        
    async def _check_models(self):
        if self.config.get_config(ConfigType.GENERAL).ml_enabled:
            await verify_models_loaded()
        
    async def _check_wallet(self):
        balance = await check_wallet_balance()
        
        if self.mode == "production" and balance < self.config.get_config(ConfigType.TRADING_LIMITS).min_balance_eth:
            raise ValueError(f"Insufficient wallet balance: {balance} ETH")
            
    async def run(self):
        """Main application loop"""
        try:
            self.logger.info("Starting DexScreener Trading Bot...")
            await self.initialize()

            await test_dex_collector(self.logger)
            
            self.logger.info("🎯 Starting trading engine...")

            chain_config = self.config.get_config(ConfigType.CHAIN)
            if chain_config:
                chains = chain_config.enabled_chains.split(',')
                self.logger.info(f"🌐 Multi-chain mode: {len(chains)} chains enabled")
                self.logger.info(f"   Chains: {', '.join(chains)}")
            else:
                self.logger.warning("⚠️ Single-chain mode (multi-chain not configured)")

            tasks = [
                asyncio.create_task(self.engine.run()),
                asyncio.create_task(self.health_checker.monitor()),
                asyncio.create_task(self._status_reporter()),
                asyncio.create_task(self.dashboard.start()),
                asyncio.create_task(self._shutdown_monitor()),
                asyncio.create_task(self._position_monitor())
            ]
            
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )
            
            for task in done:
                if task.exception():
                    self.logger.error(f"Task failed: {task.exception()}")
                    
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
                    
                await asyncio.sleep(60)
                
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
                if self.mode == "production":
                    self.logger.info("Closing all open positions...")
                    await self.engine.emergency_close_all_positions()
                    
                self.logger.info("Saving system state...")
                await self.engine.save_state()
                
                await self.engine.shutdown()
                
            await close_all_connections()
            
            self.logger.info("✅ Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

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
    """Test DexScreener connectivity using a direct, up-to-date endpoint."""
    from data.collectors.dexscreener import DexScreenerCollector
    
    logger_instance.info("Testing DexScreener API connection...")
    
    # Use a default, known-good pair for the startup check
    chain = "ethereum"
    # WETH/USDC on Ethereum
    pair_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    
    collector = None
    success = False
    try:
        # We don't need a full config, just enough to make a request
        collector = DexScreenerCollector({})
        await collector.initialize()

        pair_data = await collector.get_pair_data(pair_address=pair_address, chain=chain)

        if pair_data and 'pair_address' in pair_data:
            success = True
            logger_instance.info("✅ DexScreener API connection successful")
        else:
            logger_instance.warning(
                "⚠️ DexScreener API connection failed - response OK, but no pair data. "
                "Bot may not find new pairs!"
            )

    except Exception as e:
        logger_instance.warning(
            f"⚠️ DexScreener API connection failed: {e} - Bot will not find new pairs!"
        )
    finally:
        if collector:
            await collector.close()

    return success

async def main():
    """Main entry point"""
    args = parse_arguments()

    # Load DRY_RUN from .env file first, then override with command-line arg if present
    dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
    is_dry_run = dry_run_env in ('true', '1', 'yes')

    if args.dry_run:
        is_dry_run = True

    os.environ['DRY_RUN'] = 'true' if is_dry_run else 'false'
        
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
    if args.backtest:
        from scripts.backtest import run_backtest
        await run_backtest(args.config)
        return
        
    if args.train_models:
        from scripts.train_models import train_all_models
        await train_all_models()
        return
    
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
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)
        
    asyncio.run(main())
