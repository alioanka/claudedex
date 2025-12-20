#!/usr/bin/env python3
"""
Arbitrage Module - Multi-Chain Entry Point

Supports:
- Ethereum (Uniswap, Sushiswap, Curve, Balancer)
- Solana (Jupiter, Raydium, Orca)
- Triangular Arbitrage
- Cross-chain price monitoring
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

# Configure logging for all engines
for engine_name in ["ArbitrageEngine", "SolanaArbitrageEngine", "TriangularArbitrageEngine"]:
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

        if eth_rpc and self.settings.get('ethereum_enabled', True):
            try:
                from modules.arbitrage.arbitrage_engine import ArbitrageEngine
                config = {'arbitrage_enabled': True, 'rpc_url': eth_rpc, **self.settings}
                eth_engine = ArbitrageEngine(config, self.db_pool)
                await eth_engine.initialize()
                self.engines.append(('ethereum', eth_engine))
                logger.info("✅ Ethereum Arbitrage Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init Ethereum engine: {e}")
        else:
            if not eth_rpc:
                logger.warning("⚠️ No Ethereum RPC - Ethereum arbitrage disabled")
            else:
                logger.info("ℹ️ Ethereum arbitrage disabled in settings")

        # Check for Solana RPC - use Pool Engine with fallback
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
                logger.info("✅ Solana Arbitrage Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to init Solana engine: {e}")
        else:
            if not sol_rpc:
                logger.warning("⚠️ No Solana RPC - Solana arbitrage disabled")
            else:
                logger.info("ℹ️ Solana arbitrage disabled in settings")

        # Initialize Triangular Arbitrage (uses existing Ethereum connection)
        if eth_rpc and self.settings.get('triangular_enabled', False):
            try:
                from modules.arbitrage.triangular_engine import TriangularArbitrageEngine
                config = {'arbitrage_enabled': True, **self.settings}
                tri_engine = TriangularArbitrageEngine(config, self.db_pool)
                await tri_engine.initialize()
                self.engines.append(('triangular', tri_engine))
                logger.info("✅ Triangular Arbitrage Engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ Triangular engine not available: {e}")
        else:
            if eth_rpc:
                logger.info("ℹ️ Triangular arbitrage disabled in settings")

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

    # Init Multi-Chain Manager
    manager = MultiChainArbitrageManager(db_pool)

    if not await manager.initialize():
        logger.error("❌ Failed to initialize arbitrage manager")
        return

    try:
        await manager.run()
    except KeyboardInterrupt:
        await manager.stop()
    except Exception as e:
        logger.error(f"❌ Manager error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
