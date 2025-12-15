#!/usr/bin/env python3
"""
DEX Trading Bot - Main Entry Point
Advanced automated DEX trading system with ML-powered decision making
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Dict

# Add project root to path for consistent imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.module_loader import ModuleWrapper
from core.engine import TradingBotEngine
from config.config_manager import ConfigType
from security.encryption import EncryptionManager
from monitoring.enhanced_dashboard import DashboardEndpoints
from core.portfolio_manager import PortfolioManager
from trading.orders.order_manager import OrderManager
from core.risk_manager import RiskManager
from monitoring.alerts import AlertsSystem
from core.analytics_engine import AnalyticsEngine

# Initialize logger for this module
logger = logging.getLogger("DEXTrading")

class DEXModule(ModuleWrapper):
    """
    The DEX trading module, inheriting the common lifecycle from ModuleWrapper.
    """
    def __init__(self, mode: str = "production"):
        super().__init__(module_name="DEXTrading", mode=mode)
        self.engine: TradingBotEngine | None = None
        self.dashboard: DashboardEndpoints | None = None

    async def initialize_module(self):
        """
        Initialize all DEX-specific components using the services
        (db_manager, config_manager) provided by the base class.
        """
        self.logger.info("Initializing DEX-specific components...")

        # 1. Build the final nested_config from the fully loaded ConfigManager
        nested_config = {}
        for config_type in ConfigType:
            config_model = self.config_manager.get_config(config_type)
            if config_model:
                nested_config[config_type.value] = config_model.dict()

        # 2. Decrypt and Inject Secrets into the config
        security_config = nested_config.get('security', {})
        encryption_key = os.getenv('ENCRYPTION_KEY')
        fernet = EncryptionManager({'encryption_key': encryption_key}).fernet

        # Decrypt and inject EVM private key
        encrypted_evm_key = os.getenv('DEX_MODULE_EVM_PRIVATE_KEY')
        if encrypted_evm_key:
            try:
                decrypted_evm_key = fernet.decrypt(encrypted_evm_key.encode()).decode()
                security_config['private_key'] = decrypted_evm_key
                self.logger.info("✅ Successfully decrypted DEX Module EVM private key.")
            except Exception as e:
                raise ValueError(f"Cannot decrypt DEX_MODULE_EVM_PRIVATE_KEY: {e}")
        
        # Decrypt and inject Solana private key for DEX module's Jupiter use
        encrypted_sol_key = os.getenv('DEX_MODULE_SOLANA_PRIVATE_KEY')
        if encrypted_sol_key:
            try:
                decrypted_sol_key = fernet.decrypt(encrypted_sol_key.encode()).decode()
                security_config['solana_private_key'] = decrypted_sol_key
                self.logger.info("✅ Successfully decrypted DEX Module Solana private key.")
            except Exception as e:
                raise ValueError(f"Cannot decrypt DEX_MODULE_SOLANA_PRIVATE_KEY: {e}")

        nested_config['security'] = security_config

        # 3. Extract RPC URLs from environment
        chain_rpc_urls = {
            name.replace('_RPC_URLS', '').lower(): [url.strip() for url in value.split(',')]
            for name, value in os.environ.items() if name.endswith('_RPC_URLS') and value
        }

        # 4. Initialize Core Components
        portfolio_manager = PortfolioManager(nested_config)
        order_manager = OrderManager(nested_config)
        risk_manager = RiskManager(nested_config, config_manager=self.config_manager, chain_rpc_urls=chain_rpc_urls)
        alerts_system = AlertsSystem(nested_config)
        analytics_engine = AnalyticsEngine(db_manager=self.db_manager)
        await analytics_engine.initialize()

        # 5. Initialize Trading Engine
        self.logger.info("Initializing trading engine...")
        self.engine = TradingBotEngine(
            config=nested_config,
            config_manager=self.config_manager,
            chain_rpc_urls=chain_rpc_urls,
                mode=self.mode,
                global_event_bus=self.global_event_bus
        )
        await self.engine.initialize()

        # 6. Initialize Dashboard
        self.logger.info("Initializing dashboard...")
        dashboard_port = self.config_manager.get_dashboard_config().dashboard_port
        self.dashboard = DashboardEndpoints(
            host="0.0.0.0",
            port=dashboard_port,
            config=nested_config,
            trading_engine=self.engine,
            portfolio_manager=portfolio_manager,
            order_manager=order_manager,
            risk_manager=risk_manager,
            alerts_system=alerts_system,
            config_manager=self.config_manager,
            db_manager=self.db_manager,
            analytics_engine=analytics_engine
        )
        self.logger.info(f"✅ DEX Module initialization complete. Dashboard will be on port {dashboard_port}.")

    async def run_module_tasks(self):
        """
        Run the main long-running tasks for the DEX module.
        This will run the engine and the dashboard concurrently and wait for either to complete.
        """
        self.logger.info("🎯 Starting main DEX module tasks (engine and dashboard)...")
        engine_task = asyncio.create_task(self.engine.run(), name="engine_run")
        dashboard_task = asyncio.create_task(self.dashboard.start(), name="dashboard_start")
        
        # The main run loop waits for either the engine or dashboard to stop,
        # or for the shutdown event to be set.
        done, pending = await asyncio.wait(
            [engine_task, dashboard_task, asyncio.create_task(self.shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )

        # If a task failed, log the exception
        for task in done:
            if task.exception():
                self.logger.error(f"Task '{task.get_name()}' exited with an exception.", exc_info=task.exception())

        # Trigger shutdown for the rest of the module
        self.shutdown_event.set()
        for task in pending:
            task.cancel()

    async def shutdown_module(self):
        """Gracefully shut down all DEX-specific components."""
        self.logger.info("Shutting down DEX-specific components...")
        if self.engine:
            await self.engine.shutdown()
        if self.dashboard:
            # The dashboard's AIOHTTP server should shut down automatically when its task is cancelled.
            pass
        self.logger.info("✅ DEX components shut down.")

def parse_arguments():
    """Parse command line arguments for the DEX module."""
    parser = argparse.ArgumentParser(description="DEX Trading Bot Module")
    parser.add_argument('--mode', choices=['development', 'production'], default='production', help='Operating mode')
    parser.add_argument('--dry-run', action='store_true', help='Force simulation mode, overriding .env settings')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging level')
    return parser.parse_args()

async def main():
    """Main entry point for the DEX module."""
    args = parse_arguments()

    # DRY_RUN is a global override. If the CLI flag is set, it forces dry run.
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    module = DEXModule(mode=args.mode)
    await module.run()

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ is required.")
        sys.exit(1)
        
    asyncio.run(main())
