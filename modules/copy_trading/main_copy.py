#!/usr/bin/env python3
"""
Copy Trading Module - Entry Point
"""
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import RPCProvider for centralized RPC management
try:
    from config.rpc_provider import RPCProvider
except ImportError:
    RPCProvider = None

# Load env
load_dotenv()

# Setup Logging
log_dir = Path("logs/copy_trading")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Root logger
logger = logging.getLogger("CopyTradingModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'copy_trading.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'copy_trading_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# 3. Trades Log
trade_logger = logging.getLogger("CopyTradingModule.Trades")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
trade_handler = RotatingFileHandler(log_dir / 'copy_trading_trades.log', maxBytes=10*1024*1024, backupCount=5)
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.copy_trading.copy_engine import CopyTradingEngine, CopyTradeExecutor
from config.config_manager import ConfigManager

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller
except ImportError:
    get_telegram_controller = None

# Also configure logging for engine classes
for engine_name in ["CopyTradingEngine", "CopyTradeExecutor"]:
    engine_logger = logging.getLogger(engine_name)
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(main_handler)
    engine_logger.addHandler(error_handler)
    engine_logger.addHandler(console)

async def main():
    logger.info("👯 Copy Trading Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

    # Check for API keys and RPC URLs - use Pool Engine for RPC management
    try:
        from security.secrets_manager import secrets
        etherscan_key = secrets.get('ETHERSCAN_API_KEY', log_access=False)
        helius_key = secrets.get('HELIUS_API_KEY', log_access=False)
    except Exception:
        # Fallback to env if secrets manager unavailable
        etherscan_key = os.getenv('ETHERSCAN_API_KEY')
        helius_key = os.getenv('HELIUS_API_KEY')

    # Use Pool Engine for Solana RPC
    solana_rpc = None
    if RPCProvider:
        solana_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
    if not solana_rpc:
        solana_rpc = os.getenv('SOLANA_RPC_URL')

    logger.info(f"   ETHERSCAN_API_KEY: {'Configured' if etherscan_key else 'NOT SET - EVM monitoring disabled'}")
    logger.info(f"   SOLANA_RPC_URL: {'Configured' if solana_rpc else 'NOT SET - Solana monitoring disabled'}")
    logger.info(f"   HELIUS_API_KEY: {'Configured' if helius_key else 'Not set (optional)'}")

    if not etherscan_key and not solana_rpc:
        logger.warning("⚠️ No API keys configured - Copy Trading will not monitor any wallets")

    # Init DB - Use Docker secrets or environment
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')

    if not db_url:
        logger.error("No database credentials found (Docker secrets or DATABASE_URL)")
        return

    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Engine
    config = {'copy_trading_enabled': True}
    engine = CopyTradingEngine(config, db_pool)

    # Initialize Telegram controller for remote control (credentials from secrets manager)
    telegram_controller = None
    if get_telegram_controller:
        try:
            telegram_controller = get_telegram_controller(db_pool)
            if await telegram_controller.initialize():
                telegram_controller.register_module(
                    name='copy_trading',
                    engine=engine,
                    start_method='run',
                    stop_method='stop',
                    positions_attr='active_copies'
                )
                await telegram_controller.start_polling()
                logger.info("📱 Telegram remote control enabled")
                await telegram_controller.notify("Copy Trading Module started. Send /help for commands.", priority="normal")
        except Exception as e:
            logger.warning(f"Telegram controller failed to initialize: {e}")

    try:
        logger.info("✅ Copy Trading Engine initialized successfully")
        await engine.run()
    except KeyboardInterrupt:
        if telegram_controller:
            await telegram_controller.notify("Copy Trading module shutting down...", priority="high")
            await telegram_controller.stop_polling()
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Engine error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if telegram_controller:
            await telegram_controller.stop_polling()
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
