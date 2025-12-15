#!/usr/bin/env python3
"""
Futures Trading Bot - Main Entry Point
Handles futures trading on Binance and Bybit exchanges
"""
import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging
from aiohttp import web

# Add project root for consistent imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.module_loader import ModuleWrapper
from futures_trading.core.futures_engine import FuturesTradingEngine
from modules.futures_trading.futures_config_manager import FuturesConfigManager

# Initialize logger for this module
logger = logging.getLogger("FuturesTrading")

class FuturesModule(ModuleWrapper):
    """
    The Futures trading module, inheriting the common lifecycle from ModuleWrapper.
    """
    def __init__(self, mode: str = "production"):
        super().__init__(module_name="FuturesTrading", mode=mode)
        self.engine: FuturesTradingEngine | None = None
        self.health_server: web.AppRunner | None = None
        self.futures_config_manager: FuturesConfigManager | None = None

    async def initialize_module(self):
        """
        Initialize all Futures-specific components using the services
        (db_manager, config_manager) provided by the base class.
        """
        self.logger.info("Initializing Futures-specific components...")

        # 1. Initialize Futures-specific config manager
        self.futures_config_manager = FuturesConfigManager(db_pool=self.db_pool)
        await self.futures_config_manager.initialize()
        self.logger.info("✅ Futures config manager loaded from database.")

        # 2. Initialize Futures Trading Engine
        self.logger.info("Initializing Futures trading engine...")
        self.engine = FuturesTradingEngine(
            config_manager=self.futures_config_manager,
            mode=self.mode,
            db_pool=self.db_pool,
            global_event_bus=self.global_event_bus
        )
        await self.engine.initialize()

        # 3. Initialize Health Server
        await self._start_health_server()
        self.logger.info("✅ Futures Module initialization complete.")

    async def run_module_tasks(self):
        """Run the main long-running tasks for the Futures module."""
        self.logger.info("🎯 Starting main Futures module tasks (engine)...")
        engine_task = asyncio.create_task(self.engine.run(), name="futures_engine_run")

        done, pending = await asyncio.wait(
            [engine_task, asyncio.create_task(self.shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            if task.exception():
                self.logger.error(f"Task '{task.get_name()}' exited with an exception.", exc_info=task.exception())

        self.shutdown_event.set()
        for task in pending:
            task.cancel()

    async def shutdown_module(self):
        """Gracefully shut down all Futures-specific components."""
        self.logger.info("Shutting down Futures-specific components...")
        if self.health_server:
            await self.health_server.cleanup()
        if self.engine:
            await self.engine.shutdown()
        self.logger.info("✅ Futures components shut down.")

    async def _start_health_server(self):
        """Initializes and starts the AIOHTTP health server."""
        health_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
        app = web.Application()
        app.router.add_get('/health', self._health_handler)
        app.router.add_get('/ready', self._ready_handler)

        self.health_server = web.AppRunner(app)
        await self.health_server.setup()
        site = web.TCPSite(self.health_server, '0.0.0.0', health_port)
        await site.start()
        self.logger.info(f"📡 Health server started on http://0.0.0.0:{health_port}")

    async def _health_handler(self, request):
        """Basic health check endpoint."""
        if self.engine:
            health = await self.engine.get_health()
            status = 200 if health.get('status') == 'healthy' else 503
            return web.json_response(health, status=status)
        return web.json_response({'status': 'initializing'}, status=503)

    async def _ready_handler(self, request):
        """Readiness probe for orchestration."""
        is_ready = self.engine and self.engine.is_running
        return web.json_response({'ready': is_ready}, status=200 if is_ready else 503)

def parse_arguments():
    """Parse command line arguments for the Futures module."""
    parser = argparse.ArgumentParser(description="Futures Trading Bot Module")
    parser.add_argument('--mode', choices=['development', 'production'], default='production', help='Operating mode')
    parser.add_argument('--dry-run', action='store_true', help='Force simulation mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

async def main():
    """Main entry point for the Futures module."""
    args = parse_arguments()

    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    module = FuturesModule(mode=args.mode)
    await module.run()

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ is required.")
        sys.exit(1)

    asyncio.run(main())
