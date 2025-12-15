#!/usr/bin/env python3
"""
Module Wrapper - Abstract Base Class for Trading Modules
Provides a unified structure for initializing, running, and shutting down trading modules.
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from abc import ABC, abstractmethod

# Add project root to path for consistent imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_manager import ConfigManager
from data.storage.database import DatabaseManager
from core.global_event_bus import GlobalEventBus

# Load environment variables from the root .env file
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

class ModuleWrapper(ABC):
    """
    An abstract base class that provides a standardized lifecycle for all trading modules.
    It handles common tasks such as:
    - Signal handling for graceful shutdowns.
    - Database and configuration manager initialization.
    - A structured run/shutdown loop.
    - Environment validation.
    """
    def __init__(self, module_name: str, mode: str = "production"):
        self.module_name = module_name
        self.mode = mode
        self.logger = logging.getLogger(self.module_name)
        self.shutdown_event = asyncio.Event()

        self.db_manager: DatabaseManager | None = None
        self.config_manager: ConfigManager | None = None
        self.global_event_bus: GlobalEventBus | None = None

        # Set up signal handlers for SIGINT (Ctrl+C) and SIGTERM
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            self.logger.warning("Cannot set signal handlers in a non-main thread.")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully by setting the shutdown event."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    def _validate_environment(self):
        """
        Validates that all required environment variables (secrets) for this module are set.
        Relies on the ConfigManager's validation logic.
        """
        self.logger.info("Validating required environment secrets...")
        missing = self.config_manager.validate_environment()
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.logger.info("✅ Environment secrets validation passed.")

    async def _initialize_core_services(self):
        """Initializes the database connection and the configuration manager."""
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 {self.module_name} Module Starting...")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Time: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

        # 1. Connect to Database
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL must be set for all modules.")

        self.db_manager = DatabaseManager({'dsn': db_url})
        self.logger.info("Connecting to database...")
        await self.db_manager.connect()
        self.logger.info("✅ Database connected.")

        # 2. Initialize ConfigManager with DB pool
        self.logger.info("Initializing configuration manager...")
        self.config_manager = ConfigManager(db_pool=self.db_manager.pool)
        await self.config_manager.initialize(os.getenv('ENCRYPTION_KEY'))
        self.logger.info("✅ Configuration loaded from database.")

        # 3. Validate secrets from .env
        self._validate_environment()

        # 4. Connect Global Event Bus
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            raise ValueError("REDIS_URL must be set for the Global Event Bus.")
        self.global_event_bus = GlobalEventBus(redis_url)
        await self.global_event_bus.connect()

    @abstractmethod
    async def initialize_module(self):
        """
        Abstract method for module-specific initialization.
        Subclasses must implement this to set up their specific engines,
        services, and tasks.
        """
        pass

    @abstractmethod
    async def run_module_tasks(self):
        """
        Abstract method to run the main, long-running tasks for the module.
        Subclasses must implement this to start their core logic (e.g., trading engine).
        """
        pass

    @abstractmethod
    async def shutdown_module(self):
        """
        Abstract method for module-specific cleanup.
        Subclasses must implement this to gracefully stop their components.
        """
        pass

    async def run(self):
        """
        The main public method to start and manage the module's lifecycle.
        """
        try:
            await self._initialize_core_services()
            await self.initialize_module()

            # This task waits for the shutdown signal.
            shutdown_task = asyncio.create_task(self.shutdown_event.wait(), name=f"{self.module_name}_shutdown_waiter")

            # Run the module's main logic. This should block until a critical error or shutdown.
            await self.run_module_tasks()

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info(f"Shutdown signal detected for {self.module_name}.")
        except Exception as e:
            self.logger.error(f"❌ A fatal error occurred in {self.module_name}: {e}", exc_info=True)
            # In case of a fatal startup or runtime error, ensure shutdown is triggered.
            self.shutdown_event.set()
        finally:
            self.logger.info(f"Initiating shutdown sequence for {self.module_name}...")
            await self.shutdown_module()
            if self.global_event_bus:
                await self.global_event_bus.close()
            if self.db_manager:
                await self.db_manager.disconnect()
            self.logger.info(f"✅ {self.module_name} has been shut down gracefully.")
