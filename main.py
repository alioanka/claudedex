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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import TradingBotEngine
from monitoring.logger import StructuredLogger  # Changed from setup_logger
from config.config_manager import ConfigManager  # Changed import
from data.storage.database import DatabaseManager
from security.encryption import EncryptionManager  # Changed from SecurityManager

# Load environment variables
load_dotenv()

# Global logger
logger = None

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
    cache = CacheManager({})
    await cache.connect()
    await cache.disconnect()

async def test_web3_connection():
    """Test Web3 connection - imported from base_executor"""
    from trading.executors.base_executor import test_web3_connection as test_web3
    await test_web3()
    
async def test_api_connection():
    """Test DexScreener API connection"""
    from data.collectors.dexscreener import test_api_connection as test_api
    await test_api()
    
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
            
            # Load configuration using ConfigManager
            self.logger.info("Loading configuration...")
            config_manager = ConfigManager(self.config_path)
            config = config_manager.get_config('trading')  # Get trading config
            
            # Validate environment
            self._validate_environment()
            
            # Initialize security using EncryptionManager
            self.logger.info("Initializing security manager...")
            security_manager = EncryptionManager({})
            
            # Initialize database
            self.logger.info("Connecting to database...")
            db_config = config_manager.get_database_config() if hasattr(config_manager, 'get_database_config') else {}
            db_manager = DatabaseManager(db_config)
            await db_manager.connect()
            
            # Initialize trading engine
            self.logger.info("Initializing trading engine...")
            self.engine = TradingBotEngine(config, mode=self.mode)
            await self.engine.initialize()
            
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
            'DATABASE_URL',
            'REDIS_URL',
            'DEXSCREENER_API_KEY',
            'WEB3_PROVIDER_URL',
            'PRIVATE_KEY',  # Should be encrypted
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
            
        # Validate API keys format
        if len(os.getenv('DEXSCREENER_API_KEY', '')) < 20:
            raise ValueError("Invalid DEXSCREENER_API_KEY format")
            
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
        
    async def _check_apis(self):
        """Check external API connectivity"""
#        from data.collectors.dexscreener import test_api_connection
        await test_api_connection()
        
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
            # Initialize components
            await self.initialize()
            
            # Start the trading engine
            self.logger.info("🎯 Starting trading engine...")
            
            # Create main tasks
            tasks = [
                asyncio.create_task(self.engine.run()),
                asyncio.create_task(self.health_checker.monitor()),
                asyncio.create_task(self._status_reporter()),
                asyncio.create_task(self._shutdown_monitor())
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