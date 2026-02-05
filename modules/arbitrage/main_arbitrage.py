#!/usr/bin/env python3
"""
Arbitrage Module - Multi-Chain Entry Point

Supports:
- Ethereum (Uniswap, Sushiswap, Curve, Balancer)
- Arbitrum (SushiSwap, Camelot, Uniswap V3)
- Base (Aerodrome, BaseSwap, Uniswap V3)
- Solana (Jupiter, Raydium, Orca)
- Triangular Arbitrage
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

# Import engines
from modules.arbitrage.arbitrage_engine import (
    ETHArbitrageEngine,
    ARBArbitrageEngine,
    BaseArbitrageEngine,
)
from config.config_manager import ConfigManager

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller, TelegramBotController
except ImportError:
    get_telegram_controller = None
    TelegramBotController = None

# Configure logging for all engines - each engine has its own dedicated logger
ENGINE_LOGGERS = [
    "ETHArbitrageEngine",    # Ethereum mainnet
    "ARBArbitrageEngine",    # Arbitrum One
    "BaseArbitrageEngine",   # Base L2
    "SolanaArbitrageEngine", # Solana
    "TriangularArbitrageEngine",  # Triangular (ETH)
]

for engine_name in ENGINE_LOGGERS:
    eng_logger = logging.getLogger(engine_name)
    eng_logger.setLevel(logging.INFO)
    eng_logger.addHandler(main_handler)
    eng_logger.addHandler(error_handler)
    eng_logger.addHandler(console)


class MultiChainArbitrageManager:
    """Manages multiple arbitrage engines across chains"""

    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.engines = []
        self.tasks = []
        self.settings = {}

    async def _load_settings_from_db(self):
        """Load arbitrage settings from database config_settings table"""
        if not self.db_pool:
            logger.warning("No DB pool - using default settings")
            return {}

        settings = {}
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value FROM config_settings WHERE config_type = 'arbitrage_config'"
                )
                for row in rows:
                    key = row['key']
                    val = row['value']
                    # Parse value types
                    if val.lower() in ('true', 'false'):
                        val = val.lower() == 'true'
                    elif val.replace('.', '', 1).replace('-', '', 1).isdigit():
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    settings[key] = val

            logger.info(f"📋 Loaded {len(settings)} arbitrage settings from database")
        except Exception as e:
            logger.error(f"Failed to load arbitrage settings from DB: {e}")

        return settings

    async def initialize(self):
        logger.info("🌐 Multi-Chain Arbitrage Manager Starting...")

        # Load settings from database
        self.settings = await self._load_settings_from_db()
        logger.info(f"   Settings: {self.settings}")

        # Check for Ethereum RPC - use Pool Engine with fallback
        eth_rpc = None
        try:
            from config.rpc_provider import RPCProvider
            eth_rpc = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
        except Exception:
            pass
        if not eth_rpc:
            eth_rpc = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))

        # ═══════════════════════════════════════════════════════════════════════════
        # ETHEREUM ARBITRAGE ENGINE
        # ═══════════════════════════════════════════════════════════════════════════
        if eth_rpc and self.settings.get('ethereum_enabled', True):
            try:
                config = {'arbitrage_enabled': True, 'rpc_url': eth_rpc, **self.settings}
                eth_engine = ETHArbitrageEngine(config, self.db_pool)
                await eth_engine.initialize()
                self.engines.append(('ethereum', eth_engine))
                logger.info("✅ ETHArbitrageEngine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init Ethereum engine: {e}")
        else:
            if not eth_rpc:
                logger.warning("⚠️ No Ethereum RPC - Ethereum arbitrage disabled")
            else:
                logger.info("ℹ️ Ethereum arbitrage disabled in settings")

        # ═══════════════════════════════════════════════════════════════════════════
        # ARBITRUM ARBITRAGE ENGINE
        # ═══════════════════════════════════════════════════════════════════════════
        arb_rpc = None
        try:
            arb_rpc = RPCProvider.get_rpc_sync('ARBITRUM_RPC')
        except Exception:
            pass
        if not arb_rpc:
            arb_rpc = os.getenv('ARBITRUM_RPC_URL')

        if arb_rpc and self.settings.get('arbitrum_enabled', False):
            try:
                config = {'arbitrage_enabled': True, 'rpc_url': arb_rpc, **self.settings}
                arb_engine = ARBArbitrageEngine(config, self.db_pool)
                await arb_engine.initialize()
                self.engines.append(('arbitrum', arb_engine))
                logger.info("✅ ARBArbitrageEngine initialized")
                logger.info("   💡 Arbitrum has ~95% lower gas costs than Ethereum mainnet")
            except Exception as e:
                logger.error(f"❌ Failed to init Arbitrum engine: {e}")
        else:
            if not arb_rpc:
                logger.info("ℹ️ No Arbitrum RPC - set ARBITRUM_RPC_URL or add via Pool Engine")
            else:
                logger.info("ℹ️ Arbitrum arbitrage disabled in settings (set arbitrum_enabled=true)")

        # ═══════════════════════════════════════════════════════════════════════════
        # BASE ARBITRAGE ENGINE
        # ═══════════════════════════════════════════════════════════════════════════
        base_rpc = None
        try:
            base_rpc = RPCProvider.get_rpc_sync('BASE_RPC')
        except Exception:
            pass
        if not base_rpc:
            base_rpc = os.getenv('BASE_RPC_URL')

        if base_rpc and self.settings.get('base_enabled', False):
            try:
                config = {'arbitrage_enabled': True, 'rpc_url': base_rpc, **self.settings}
                base_engine = BaseArbitrageEngine(config, self.db_pool)
                await base_engine.initialize()
                self.engines.append(('base', base_engine))
                logger.info("✅ BaseArbitrageEngine initialized")
                logger.info("   💡 Base has ~97% lower gas costs than Ethereum mainnet")
            except Exception as e:
                logger.error(f"❌ Failed to init Base engine: {e}")
        else:
            if not base_rpc:
                logger.info("ℹ️ No Base RPC - set BASE_RPC_URL or add via Pool Engine")
            else:
                logger.info("ℹ️ Base arbitrage disabled in settings (set base_enabled=true)")

        # ═══════════════════════════════════════════════════════════════════════════
        # SOLANA ARBITRAGE ENGINE
        # ═══════════════════════════════════════════════════════════════════════════
        sol_rpc = None
        try:
            sol_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
        except Exception:
            pass
        if not sol_rpc:
            sol_rpc = os.getenv('SOLANA_RPC_URL')

        if sol_rpc and self.settings.get('solana_enabled', False):
            try:
                from modules.arbitrage.solana_engine import SolanaArbitrageEngine
                config = {'arbitrage_enabled': True, **self.settings}
                sol_engine = SolanaArbitrageEngine(config, self.db_pool)
                await sol_engine.initialize()
                self.engines.append(('solana', sol_engine))
                logger.info("✅ SolanaArbitrageEngine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init Solana engine: {e}")
        else:
            if not sol_rpc:
                logger.info("ℹ️ No Solana RPC - set SOLANA_RPC_URL or add via Pool Engine")
            else:
                logger.info("ℹ️ Solana arbitrage disabled in settings (set solana_enabled=true)")

        # ═══════════════════════════════════════════════════════════════════════════
        # TRIANGULAR ARBITRAGE ENGINE (uses Ethereum RPC)
        # ═══════════════════════════════════════════════════════════════════════════
        if eth_rpc and self.settings.get('triangular_enabled', False):
            try:
                from modules.arbitrage.triangular_engine import TriangularArbitrageEngine
                config = {'arbitrage_enabled': True, **self.settings}
                tri_engine = TriangularArbitrageEngine(config, self.db_pool)
                await tri_engine.initialize()
                self.engines.append(('triangular', tri_engine))
                logger.info("✅ TriangularArbitrageEngine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Triangular engine not available: {e}")
        else:
            if eth_rpc:
                logger.info("ℹ️ Triangular arbitrage disabled in settings (set triangular_enabled=true)")

        if not self.engines:
            logger.error("❌ No arbitrage engines could be initialized!")
            return False

        logger.info(f"🌐 Initialized {len(self.engines)} arbitrage engines")
        return True

    async def run(self):
        """Run all engines concurrently"""
        if not self.engines:
            logger.error("No engines to run")
            return

        logger.info(f"🚀 Starting {len(self.engines)} arbitrage engines...")

        # Create tasks for each engine
        for chain, engine in self.engines:
            task = asyncio.create_task(engine.run())
            self.tasks.append((chain, task))
            logger.info(f"   Started {chain} engine")

        # Wait for all tasks (they run indefinitely)
        try:
            await asyncio.gather(*[t for _, t in self.tasks])
        except asyncio.CancelledError:
            logger.info("Engines cancelled")

    async def stop(self):
        """Stop all engines"""
        logger.info("Stopping all arbitrage engines...")

        # Cancel tasks
        for chain, task in self.tasks:
            task.cancel()
            logger.info(f"   Cancelled {chain} engine")

        # Stop engines
        for chain, engine in self.engines:
            try:
                await engine.stop()
                logger.info(f"   Stopped {chain} engine")
            except Exception as e:
                logger.error(f"   Error stopping {chain}: {e}")

        logger.info("🛑 All arbitrage engines stopped")


async def main():
    logger.info("⚖️ Multi-Chain Arbitrage Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

    # Check for at least one RPC
    eth_rpc = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))
    sol_rpc = os.getenv('SOLANA_RPC_URL')

    if not eth_rpc and not sol_rpc:
        logger.error("❌ No RPC URLs configured (ETHEREUM_RPC_URL or SOLANA_RPC_URL)")
        logger.info("   Set at least one RPC URL in your .env file")
        while True:
            logger.info("⚖️ Arbitrage Module IDLE - Waiting for RPC configuration...")
            await asyncio.sleep(300)
        return

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
    except Exception as e:
        logger.warning(f"⚠️ Pool Engine init failed, using .env fallback: {e}")

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Multi-Chain Manager
    manager = MultiChainArbitrageManager(db_pool)

    if not await manager.initialize():
        logger.error("❌ Failed to initialize arbitrage manager")
        return

    # Initialize Telegram controller for remote control (credentials from secrets manager)
    telegram_controller = None
    if get_telegram_controller:
        try:
            telegram_controller = get_telegram_controller(db_pool)
            if await telegram_controller.initialize():
                # Register all arbitrage engines for remote control
                for chain, engine in manager.engines:
                    telegram_controller.register_module(
                        name=f'arbitrage_{chain}',
                        engine=engine,
                        start_method='run',
                        stop_method='stop',
                        positions_attr='active_positions'  # Arbitrage engines don't have positions
                    )
                await telegram_controller.start_polling()
                logger.info("📱 Telegram remote control enabled")
                await telegram_controller.notify(
                    f"Arbitrage Module started with {len(manager.engines)} engines. Send /help for commands.",
                    priority="normal"
                )
        except Exception as e:
            logger.warning(f"Telegram controller failed to initialize: {e}")

    try:
        await manager.run()
    except KeyboardInterrupt:
        if telegram_controller:
            await telegram_controller.notify("Arbitrage module shutting down...", priority="high")
            await telegram_controller.stop_polling()
        await manager.stop()
    except Exception as e:
        logger.error(f"❌ Manager error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
