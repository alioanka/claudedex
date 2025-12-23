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
    """Redirect stderr to a rotating file handler with size-based rotation"""
    def __init__(self, filepath, max_bytes=5*1024*1024, backup_count=3):
        self.filepath = Path(filepath)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.original_stderr = sys.stderr
        self._open_file()

    def _open_file(self):
        """Open or reopen the file"""
        self.file = open(self.filepath, 'a', encoding='utf-8')

    def _should_rollover(self):
        """Check if we should rotate based on file size"""
        try:
            return self.filepath.stat().st_size >= self.max_bytes
        except:
            return False

    def _do_rollover(self):
        """Perform the rollover"""
        try:
            self.file.close()
            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                src = f"{self.filepath}.{i}"
                dst = f"{self.filepath}.{i + 1}"
                if Path(src).exists():
                    Path(src).rename(dst)
            # Move current to .1
            if self.filepath.exists():
                self.filepath.rename(f"{self.filepath}.1")
            self._open_file()
        except Exception:
            pass  # Silently handle rotation errors

    def write(self, message):
        if message and message.strip():
            try:
                self.file.write(message)
                if self._should_rollover():
                    self._do_rollover()
            except:
                pass
        # Also write to original stderr for console visibility
        if self.original_stderr:
            try:
                self.original_stderr.write(message)
            except:
                pass

    def flush(self):
        try:
            self.file.flush()
        except:
            pass
        if self.original_stderr:
            try:
                self.original_stderr.flush()
            except:
                pass

# Install stderr redirect with rotation (5MB max, 3 backups)
try:
    stderr_redirect = StderrToRotatingFile(log_dir / 'stderr.log')
    sys.stderr = stderr_redirect
except Exception:
    pass  # Keep original stderr if redirect fails

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

    # Check for RPC URLs - use Pool Engine with fallback
    solana_rpc = None
    evm_rpc = None
    try:
        from config.rpc_provider import RPCProvider
        solana_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
        evm_rpc = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
    except Exception:
        pass

    if not solana_rpc:
        solana_rpc = os.getenv('SOLANA_RPC_URL')
    if not evm_rpc:
        evm_rpc = os.getenv('WEB3_PROVIDER_URL') or os.getenv('ETHEREUM_RPC_URL')

    logger.info(f"   Solana RPC: {'Configured' if solana_rpc else 'Not configured'}")
    logger.info(f"   EVM RPC: {'Configured' if evm_rpc else 'Not configured'}")

    if not solana_rpc and not evm_rpc:
        logger.warning("⚠️ No RPC URLs configured - sniper will have limited functionality")

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

    # Initialize Pool Engine BEFORE using RPCProvider
    try:
        from config.pool_engine import PoolEngine
        from config.rpc_provider import RPCProvider

        pool_engine = await PoolEngine.get_instance()
        await pool_engine.initialize(db_pool)
        RPCProvider.set_pool_engine(pool_engine)
        logger.info("✅ Pool Engine initialized for RPC management")

        # Now get RPCs from Pool Engine (with proper initialization)
        solana_rpc = await RPCProvider.get_rpc('SOLANA_RPC')
        evm_rpc = await RPCProvider.get_rpc('ETHEREUM_RPC')
        logger.info(f"   Solana RPC from Pool Engine: {'OK' if solana_rpc else 'Not available'}")
        logger.info(f"   EVM RPC from Pool Engine: {'OK' if evm_rpc else 'Not available'}")
    except Exception as e:
        logger.warning(f"⚠️ Pool Engine init failed, using .env fallback: {e}")
        # Keep the RPCs we already fetched from env above

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()
    config = {
        'sniper': {
            'evm_enabled': bool(evm_rpc),
            'solana_enabled': bool(solana_rpc)
        },
        'solana': {'rpc_url': solana_rpc},
        'web3': {'provider_url': evm_rpc}
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
