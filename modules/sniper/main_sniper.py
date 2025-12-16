#!/usr/bin/env python3
"""
Sniper Module - Entry Point
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
log_dir = Path("logs/sniper")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Root logger
logger = logging.getLogger("SniperModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'sniper.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'sniper_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# 3. Trades Log
trade_logger = logging.getLogger("SniperModule.Trades")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
trade_handler = RotatingFileHandler(log_dir / 'sniper_trades.log', maxBytes=10*1024*1024, backupCount=5)
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.sniper.core.sniper_engine import SniperEngine
from config.config_manager import ConfigManager
from data.storage.database import DatabaseManager

async def main():
    logger.info("🔫 Sniper Module Starting...")

    # Init DB
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("No DATABASE_URL found")
        return

    import asyncpg
    db_pool = await asyncpg.create_pool(db_url)

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()
    config = {
        'sniper': {
            'evm_enabled': True,
            'solana_enabled': True
        }
    }

    engine = SniperEngine(config, config_manager, db_pool)
    await engine.initialize()

    try:
        await engine.run()
    except KeyboardInterrupt:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
