#!/usr/bin/env python3
"""
Solana Trading Bot - Main Entry Point
Handles trading on Solana blockchain (Jupiter, Drift, Pump.fun)
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
from solana_trading.core.solana_engine import SolanaTradingEngine
from modules.solana_strategies.solana_config_manager import SolanaConfigManager
from security.encryption import EncryptionManager

# Initialize logger for this module
logger = logging.getLogger("SolanaTrading")

class SolanaModule(ModuleWrapper):
    """
    The Solana trading module, inheriting the common lifecycle from ModuleWrapper.
    """
    def __init__(self, mode: str = "production"):
        super().__init__(module_name="SolanaTrading", mode=mode)
        self.engine: SolanaTradingEngine | None = None
        self.health_server: web.AppRunner | None = None
        self.solana_config_manager: SolanaConfigManager | None = None

    async def initialize_module(self):
        """
        Initialize all Solana-specific components using the services
        (db_manager, config_manager) provided by the base class.
        """
        self.logger.info("Initializing Solana-specific components...")

        # 1. Initialize Solana-specific config manager
        self.solana_config_manager = SolanaConfigManager(self.db_pool)
        await self.solana_config_manager.initialize()
        self.logger.info("✅ Solana config manager loaded from database.")

        # 2. Decrypt and prepare secrets
        encryption_key = os.getenv('ENCRYPTION_KEY')
        fernet = EncryptionManager({'encryption_key': encryption_key}).fernet

        encrypted_sol_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY')
        if not encrypted_sol_key:
            raise ValueError("SOLANA_MODULE_PRIVATE_KEY is not set.")

        try:
            decrypted_sol_key = fernet.decrypt(encrypted_sol_key.encode()).decode()
            self.logger.info("✅ Successfully decrypted Solana Module private key.")
        except Exception as e:
            raise ValueError(f"Cannot decrypt SOLANA_MODULE_PRIVATE_KEY: {e}")

        # 3. Get configuration from managers
        rpc_urls = os.getenv('SOLANA_RPC_URLS', 'https://api.mainnet-beta.solana.com')
        strategies = self.solana_config_manager.get_enabled_strategies()
        max_positions = self.solana_config_manager.max_positions

        # 4. Initialize Solana Trading Engine
        self.logger.info("Initializing Solana trading engine...")
        self.engine = SolanaTradingEngine(
            rpc_url=rpc_urls.split(',')[0],  # Use the first RPC URL as primary
            private_key=decrypted_sol_key,
            strategies=strategies,
            max_positions=max_positions,
            mode=self.mode,
            config_manager=self.solana_config_manager,
            db_pool=self.db_pool
        )
        await self.engine.initialize()

        # 5. Initialize Health Server
        await self._start_health_server()
        self.logger.info("✅ Solana Module initialization complete.")

    async def run_module_tasks(self):
        """Run the main long-running tasks for the Solana module."""
        self.logger.info("🎯 Starting main Solana module tasks (engine)...")
        engine_task = asyncio.create_task(self.engine.run(), name="solana_engine_run")

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
        """Gracefully shut down all Solana-specific components."""
        self.logger.info("Shutting down Solana-specific components...")
        if self.health_server:
            await self.health_server.cleanup()
        if self.engine:
            await self.engine.shutdown()
        self.logger.info("✅ Solana components shut down.")

    async def _start_health_server(self):
        """Initializes and starts the AIOHTTP health server."""
        health_port = int(os.getenv('SOLANA_HEALTH_PORT', '8082'))
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
    """Parse command line arguments for the Solana module."""
    parser = argparse.ArgumentParser(description="Solana Trading Bot Module")
    parser.add_argument('--mode', choices=['development', 'production'], default='production', help='Operating mode')
    parser.add_argument('--dry-run', action='store_true', help='Force simulation mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

async def main():
    """Main entry point for the Solana module."""
    args = parse_arguments()

    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    module = SolanaModule(mode=args.mode)
    await module.run()

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ is required.")
        sys.exit(1)

    asyncio.run(main())
