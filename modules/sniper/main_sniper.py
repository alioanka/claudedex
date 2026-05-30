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

# Setup Logging. We write structured logger output to sniper.log /
# sniper_errors.log under logs/sniper/. Subprocess stdout/stderr are
# captured and rotated by the parent (main.py RotatingLogFile) into
# logs/sniper/{stdout,stderr}.log — do NOT install a subprocess-side
# stderr redirect here or it will double-write to the same path the
# parent owns.
log_dir = Path("logs/sniper")
log_dir.mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger
logger = logging.getLogger("SniperModule")
logger.setLevel(logging.INFO)

# 1. Main Log — INFO level (DEBUG hot-path lines are filtered out here so
# logs/sniper/ stays small). Cap reduced 10MB x5 -> 10MB x3 (30MB total).
main_handler = RotatingFileHandler(log_dir / 'sniper.log', maxBytes=10*1024*1024, backupCount=3)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'sniper_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.sniper.core.sniper_engine import SniperEngine
from config.config_manager import ConfigManager
from data.storage.database import DatabaseManager

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller
except ImportError:
    get_telegram_controller = None

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
    # Per-module DRY_RUN override (Phase 3 A2). SNIPER_DRY_RUN env beats
    # DRY_RUN env. Mirror into DRY_RUN so the engine's internal env
    # reads pick up the resolved value.
    try:
        from core.dry_run import resolve_module_dry_run
        sniper_dry = resolve_module_dry_run('sniper', default=True)
        os.environ['DRY_RUN'] = 'true' if sniper_dry else 'false'
        logger.info(f"   DRY_RUN (resolved per-module): {sniper_dry}")
    except Exception as e:
        logger.warning(f"   Could not resolve per-module DRY_RUN: {e}")

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

    # Initialize secrets manager with database pool (before any config managers)
    try:
        from security.secrets_manager import secrets
        secrets.initialize(db_pool)
        logger.info("✅ Secrets manager initialized with database")
    except Exception as e:
        logger.warning(f"Could not initialize secrets manager: {e}")

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

    # Wave-13: inject cross-module RiskManager (entry-only gate, consistent with
    # SOLANA/ARB/COPY pattern). Fail-soft: if construction fails the engine runs
    # without the gate, which is safe because TokenSafetyChecker still runs locally.
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager(config={}, config_manager=config_manager)
        engine.set_risk_manager(risk_manager)
        logger.info("✅ RiskManager wired into Sniper engine")
    except Exception as e:
        logger.warning(
            f"RiskManager init failed (engine will run without cross-module gate): {e}"
        )

    try:
        await engine.initialize()
        logger.info("✅ Sniper Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Engine initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Initialize Telegram controller for remote control (credentials from secrets manager)
    telegram_controller = None
    if get_telegram_controller:
        try:
            # Wave-11 FIX 2: tag with module_name so start_polling honors TELEGRAM_POLL_OWNER.
            telegram_controller = get_telegram_controller(db_pool, module_name='sniper')
            if await telegram_controller.initialize():
                telegram_controller.register_module(
                    name='sniper',
                    engine=engine,
                    start_method='run',
                    stop_method='stop',
                    positions_attr='active_snipes'
                )
                await telegram_controller.start_polling()
                logger.info("📱 Telegram remote control enabled")
                await telegram_controller.notify("Sniper Module started. Send /help for commands.", priority="normal")
        except Exception as e:
            logger.warning(f"Telegram controller failed to initialize: {e}")

    try:
        await engine.run()
    except KeyboardInterrupt:
        if telegram_controller:
            await telegram_controller.notify("Sniper module shutting down...", priority="high")
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
