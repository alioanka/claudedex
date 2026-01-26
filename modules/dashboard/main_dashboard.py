#!/usr/bin/env python3
"""
Standalone Dashboard Module

This module runs the dashboard independently of trading modules.
The dashboard will:
- Start even if trading modules fail or are disabled
- Connect directly to the database
- Show module status (online/offline) based on database state
- Provide access to all monitoring and configuration features

This decouples the dashboard from the DEX module so that:
- Dashboard is always accessible regardless of module health
- Users can configure modules via dashboard even when they're offline
- Pool Engine RPCs can be managed via dashboard
"""

import asyncio
import os
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Setup logging
log_dir = Path("logs/dashboard")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
    handlers=[
        logging.FileHandler(log_dir / 'dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Dashboard")


class StandaloneDashboard:
    """
    Standalone dashboard that runs independently of trading modules.

    This ensures the dashboard is always accessible for:
    - Monitoring module status
    - Viewing trading history and analytics
    - Configuring modules and Pool Engine RPCs
    - Managing system settings
    """

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.db_manager = None
        self.config_manager = None
        self.dashboard = None
        self.pool_engine = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def initialize(self):
        """Initialize dashboard with minimal dependencies"""
        logger.info("=" * 80)
        logger.info("🖥️  Standalone Dashboard Starting...")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        try:
            # Build database config from environment
            db_config = {
                'DB_HOST': os.getenv('DB_HOST', 'postgres'),
                'DB_PORT': int(os.getenv('DB_PORT', 5432)),
                'DB_NAME': os.getenv('DB_NAME', 'tradingbot'),
                'DB_USER': os.getenv('DB_USER', 'bot_user'),
                'DB_PASSWORD': os.getenv('DB_PASSWORD', ''),
                'DB_POOL_MIN': 5,
                'DB_POOL_MAX': 10,
            }

            # Initialize database connection
            logger.info("Connecting to database...")
            from data.storage.database import DatabaseManager
            self.db_manager = DatabaseManager(db_config)
            await self.db_manager.connect()
            logger.info("✅ Database connected")

            # Initialize secrets manager with database
            logger.info("Initializing secrets manager...")
            from security.secrets_manager import secrets
            secrets.initialize(self.db_manager.pool)  # Not async - returns bool
            logger.info("✅ Secrets manager initialized")

            # Initialize config manager
            logger.info("Initializing config manager...")
            from config.config_manager import ConfigManager
            self.config_manager = ConfigManager()
            await self.config_manager.initialize()
            self.config_manager.set_db_pool(self.db_manager.pool)
            logger.info("✅ Config manager initialized")

            # Try to initialize Pool Engine (for RPC management)
            try:
                logger.info("Initializing Pool Engine...")
                from config.pool_engine import PoolEngine
                self.pool_engine = await PoolEngine.get_instance()
                await self.pool_engine.initialize(self.db_manager.pool)
                logger.info("✅ Pool Engine initialized")
            except Exception as e:
                logger.warning(f"Pool Engine initialization failed (optional): {e}")
                self.pool_engine = None

            # Get configuration
            config = {}
            try:
                config = self.config_manager.get_all_config()
            except Exception as e:
                logger.warning(f"Could not load full config: {e}")

            # Initialize dashboard with minimal dependencies
            logger.info("Initializing dashboard endpoints...")
            from monitoring.enhanced_dashboard import DashboardEndpoints

            self.dashboard = DashboardEndpoints(
                host="0.0.0.0",
                port=8080,
                config=config,
                trading_engine=None,  # Will show as "Module Offline"
                portfolio_manager=None,
                order_manager=None,
                risk_manager=None,
                alerts_system=None,
                config_manager=self.config_manager,
                db_manager=self.db_manager,
                analytics_engine=None,
                pool_engine=self.pool_engine
            )

            logger.info("✅ Dashboard initialized")
            logger.info("=" * 80)
            logger.info("🖥️  Dashboard ready at http://0.0.0.0:8080")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}", exc_info=True)
            return False

    async def run(self):
        """Run the dashboard"""
        try:
            if not await self.initialize():
                logger.error("Failed to initialize, exiting...")
                return

            # Start dashboard
            dashboard_task = asyncio.create_task(self.dashboard.start())

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Cancel dashboard task
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Critical error in dashboard: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating dashboard shutdown...")

        try:
            if self.dashboard:
                await self.dashboard.stop()
                logger.info("✅ Dashboard stopped")
        except Exception as e:
            logger.warning(f"Error stopping dashboard: {e}")

        try:
            if self.db_manager:
                await self.db_manager.disconnect()
                logger.info("✅ Database disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting database: {e}")

        logger.info("✅ Dashboard shutdown complete")


async def main():
    """Main entry point"""
    dashboard = StandaloneDashboard()

    try:
        await dashboard.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)

    asyncio.run(main())
