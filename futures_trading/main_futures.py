#!/usr/bin/env python3
"""
Futures Trading Bot - Main Entry Point
Handles futures trading on Binance and Bybit exchanges

Features:
- Multi-exchange support (Binance, Bybit)
- Leverage trading with risk management
- Position monitoring and auto-liquidation protection
- Technical indicator-based strategies
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

# Configure futures-specific logging
log_dir = Path("logs/futures")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'futures_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FuturesTrading")


class FuturesTradingApplication:
    """Main application class for futures trading"""

    def __init__(self, mode: str = "production"):
        """
        Initialize the futures trading application

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
        self.exchange = os.getenv('FUTURES_EXCHANGE', 'binance')  # binance or bybit
        self.leverage = int(os.getenv('FUTURES_LEVERAGE', '10'))
        self.max_positions = int(os.getenv('FUTURES_MAX_POSITIONS', '5'))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 Futures Trading Bot Starting...")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Exchange: {self.exchange.upper()}")
            self.logger.info(f"Leverage: {self.leverage}x")
            self.logger.info(f"Time: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)

            # Import futures engine
            from futures_trading.core.futures_engine import FuturesTradingEngine

            # Initialize engine
            self.engine = FuturesTradingEngine(
                exchange=self.exchange,
                leverage=self.leverage,
                max_positions=self.max_positions,
                mode=self.mode
            )

            await self.engine.initialize()

            self.logger.info("✅ Futures trading engine initialized")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise

    async def run(self):
        """Main application loop"""
        try:
            self.logger.info("Starting Futures Trading Bot...")
            await self.initialize()

            self.logger.info("🎯 Starting futures trading engine...")

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
                    self.logger.info(f"📊 Futures Stats: {stats}")

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
        description="Futures Trading Bot - Binance/Bybit Futures Trading"
    )

    parser.add_argument(
        '--mode',
        choices=['development', 'testing', 'production'],
        default='production',
        help='Operating mode'
    )

    parser.add_argument(
        '--exchange',
        choices=['binance', 'bybit'],
        default=os.getenv('FUTURES_EXCHANGE', 'binance'),
        help='Exchange to trade on'
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

    # Set exchange from args
    if args.exchange:
        os.environ['FUTURES_EXCHANGE'] = args.exchange

    # Handle dry-run
    dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
    is_dry_run = dry_run_env in ('true', '1', 'yes')

    if args.dry_run:
        is_dry_run = True

    os.environ['DRY_RUN'] = 'true' if is_dry_run else 'false'

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    app = FuturesTradingApplication(mode=args.mode)

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
