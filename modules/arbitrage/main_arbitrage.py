#!/usr/bin/env python3
"""
Arbitrage Module - Entry Point
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

# Load env
load_dotenv()

# Setup Logging
log_dir = Path("logs/arbitrage")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Root logger
logger = logging.getLogger("ArbitrageModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'arbitrage.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'arbitrage_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# 3. Trades Log
trade_logger = logging.getLogger("ArbitrageModule.Trades")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
trade_handler = RotatingFileHandler(log_dir / 'arbitrage_trades.log', maxBytes=10*1024*1024, backupCount=5)
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.arbitrage.arbitrage_engine import ArbitrageEngine
from config.config_manager import ConfigManager

# Also configure logging for the engine
engine_logger = logging.getLogger("ArbitrageEngine")
engine_logger.setLevel(logging.INFO)
engine_logger.addHandler(main_handler)
engine_logger.addHandler(error_handler)
engine_logger.addHandler(console)

async def main():
    logger.info("⚖️ Arbitrage Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

    # Check RPC URL
    rpc_url = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))
    if not rpc_url:
        logger.error("❌ No ETHEREUM_RPC_URL or WEB3_PROVIDER_URL found in .env")
        logger.error("   Arbitrage module requires an EVM RPC URL to function")
        # Don't exit - keep module alive but log status
        while True:
            logger.info("⚖️ Arbitrage Module IDLE - Waiting for RPC configuration...")
            await asyncio.sleep(300)
        return

    logger.info(f"   RPC URL: {rpc_url[:50]}...")

    # Init DB
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("No DATABASE_URL found")
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
    config = {'arbitrage_enabled': True}
    engine = ArbitrageEngine(config, db_pool)

    try:
        await engine.initialize()
        logger.info("✅ Arbitrage Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Engine initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    try:
        await engine.run()
    except KeyboardInterrupt:
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Engine error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
