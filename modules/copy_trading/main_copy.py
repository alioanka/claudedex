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

    # Check for API keys and RPC URLs
    etherscan_key = os.getenv('ETHERSCAN_API_KEY')
    solana_rpc = os.getenv('SOLANA_RPC_URL')
    helius_key = os.getenv('HELIUS_API_KEY')

    logger.info(f"   ETHERSCAN_API_KEY: {'Configured' if etherscan_key else 'NOT SET - EVM monitoring disabled'}")
    logger.info(f"   SOLANA_RPC_URL: {'Configured' if solana_rpc else 'NOT SET - Solana monitoring disabled'}")
    logger.info(f"   HELIUS_API_KEY: {'Configured' if helius_key else 'Not set (optional)'}")

    if not etherscan_key and not solana_rpc:
        logger.warning("⚠️ No API keys configured - Copy Trading will not monitor any wallets")

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
    config = {'copy_trading_enabled': True}
    engine = CopyTradingEngine(config, db_pool)

    try:
        logger.info("✅ Copy Trading Engine initialized successfully")
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
