#!/usr/bin/env python3
"""
Solana Trading Bot - Main Entry Point
Handles trading on Solana blockchain (Jupiter, Drift, Pump.fun)

Features:
- Jupiter V6 aggregator integration
- Drift Protocol perpetuals
- Pump.fun token sniping
- Real-time price monitoring
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure solana-specific logging
log_dir = Path("logs/solana")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'solana_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SolanaTrading")


class SolanaTradingApplication:
    """Main application class for Solana trading"""

    def __init__(self, mode: str = "production"):
        """
        Initialize the Solana trading application

        Args:
            mode: Operating mode (development, testing, production)
        """
        self.mode = mode
        self.engine = None
        self.shutdown_event = asyncio.Event()
        self.logger = logger

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Module config
        self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
        self.strategies = os.getenv('SOLANA_STRATEGIES', 'jupiter,drift').split(',')
        self.max_positions = int(os.getenv('SOLANA_MAX_POSITIONS', '3'))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 Solana Trading Bot Starting...")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"RPC: {self.rpc_url}")
            self.logger.info(f"Strategies: {', '.join(self.strategies)}")
            self.logger.info(f"Time: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)

            # Import Solana engine
            from solana_trading.core.solana_engine import SolanaTradingEngine

            # Initialize engine
            self.engine = SolanaTradingEngine(
                rpc_url=self.rpc_url,
                strategies=self.strategies,
                max_positions=self.max_positions,
                mode=self.mode
            )

            await self.engine.initialize()

            self.logger.info("✅ Solana trading engine initialized")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise

    async def run(self):
        """Main application loop"""
        try:
            self.logger.info("Starting Solana Trading Bot...")
            await self.initialize()

            self.logger.info("🎯 Starting Solana trading engine...")

            tasks = [
                asyncio.create_task(self.engine.run()),
                asyncio.create_task(self._status_reporter()),
                asyncio.create_task(self._shutdown_monitor())
            ]

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_EXCEPTION
            )

            for task in done:
                if task.exception():
                    self.logger.error(f"Task failed: {task.exception()}")

            for task in pending:
                task.cancel()

        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}", exc_info=True)

        finally:
            await self.shutdown()

    async def _status_reporter(self):
        """Periodically report system status"""
        while not self.shutdown_event.is_set():
            try:
                if self.engine:
                    stats = await self.engine.get_stats()
                    self.logger.info(f"📊 Solana Stats: {stats}")

                await asyncio.sleep(120)  # Report every 2 minutes

            except Exception as e:
                self.logger.error(f"Error in status reporter: {e}")
                await asyncio.sleep(120)

    async def _shutdown_monitor(self):
        """Monitor for shutdown signal"""
        await self.shutdown_event.wait()
        self.logger.info("Shutdown signal received")

    async def shutdown(self):
        """Graceful shutdown procedure"""
        try:
            self.logger.info("Initiating graceful shutdown...")

            if self.engine:
                if self.mode == "production":
                    self.logger.info("Closing all open positions...")
                    await self.engine.close_all_positions()

                self.logger.info("Stopping engine...")
                await self.engine.shutdown()

            self.logger.info("✅ Shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Solana Trading Bot - Jupiter/Drift/Pump.fun Trading"
    )

    parser.add_argument(
        '--mode',
        choices=['development', 'testing', 'production'],
        default='production',
        help='Operating mode'
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['jupiter', 'drift', 'pumpfun'],
        default=None,
        help='Trading strategies to enable'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in simulation mode without real trades'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()

    # Set strategies from args
    if args.strategies:
        os.environ['SOLANA_STRATEGIES'] = ','.join(args.strategies)

    # Handle dry-run
    dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
    is_dry_run = dry_run_env in ('true', '1', 'yes')

    if args.dry_run:
        is_dry_run = True

    os.environ['DRY_RUN'] = 'true' if is_dry_run else 'false'

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = SolanaTradingApplication(mode=args.mode)

    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)

    asyncio.run(main())
