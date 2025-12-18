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

# Redirect stderr to a rotating file to prevent 100MB+ stderr.log
class StderrToRotatingFile:
    """Redirect stderr to a rotating file handler"""
    def __init__(self, filepath, max_bytes=5*1024*1024, backup_count=3):
        self.handler = RotatingFileHandler(filepath, maxBytes=max_bytes, backupCount=backup_count)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - STDERR - %(message)s'))
        self.original_stderr = sys.stderr

    def write(self, message):
        if message.strip():  # Only write non-empty messages
            self.handler.stream.write(f"{message}")
            self.handler.doRollover() if self.handler.shouldRollover(None) else None
        # Also write to original stderr for console visibility
        if self.original_stderr:
            self.original_stderr.write(message)

    def flush(self):
        self.handler.stream.flush()
        if self.original_stderr:
            self.original_stderr.flush()

# Install stderr redirect with rotation (5MB max, 3 backups)
stderr_redirect = StderrToRotatingFile(log_dir / 'stderr.log')
sys.stderr = stderr_redirect

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

# Also configure logging for all engine components
for engine_name in ["SniperEngine", "EVMListener", "SolanaListener", "TokenSafetyChecker", "TradeExecutor"]:
    engine_logger = logging.getLogger(engine_name)
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(main_handler)
    engine_logger.addHandler(error_handler)
    engine_logger.addHandler(console)

async def main():
    logger.info("🔫 Sniper Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

    # Check for RPC URLs
    solana_rpc = os.getenv('SOLANA_RPC_URL')
    evm_rpc = os.getenv('WEB3_PROVIDER_URL') or os.getenv('ETHEREUM_RPC_URL')

    logger.info(f"   Solana RPC: {'Configured' if solana_rpc else 'Not configured'}")
    logger.info(f"   EVM RPC: {'Configured' if evm_rpc else 'Not configured'}")

    if not solana_rpc and not evm_rpc:
        logger.warning("⚠️ No RPC URLs configured - sniper will have limited functionality")

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
    config = {
        'sniper': {
            'evm_enabled': bool(evm_rpc),
            'solana_enabled': bool(solana_rpc)
        }
    }

    engine = SniperEngine(config, config_manager, db_pool)

    try:
        await engine.initialize()
        logger.info("✅ Sniper Engine initialized successfully")
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
