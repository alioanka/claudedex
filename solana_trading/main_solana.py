#!/usr/bin/env python3
"""
Solana Trading Bot - Main Entry Point
Handles trading on Solana blockchain (Jupiter, Drift, Pump.fun)

Features:
- Jupiter V6 aggregator integration
- Drift Protocol perpetuals
- Pump.fun token sniping
- Real-time price monitoring
- HTTP health/metrics endpoints for monitoring
- Database-backed configuration
- Separate log files for errors and trades
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
import json
from aiohttp import web

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING SETUP - Separate files for main, errors, and trades
# ============================================================================
log_dir = Path("logs/solana")
log_dir.mkdir(parents=True, exist_ok=True)

# Create formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Root logger for Solana module
logger = logging.getLogger("SolanaTrading")
logger.setLevel(logging.INFO)

# Main log file - all messages
main_handler = logging.FileHandler(log_dir / 'solana_trading.log')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Error log file - only errors and above
error_handler = logging.FileHandler(log_dir / 'solana_errors.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)
logger.addHandler(error_handler)

# Trade logger - separate logger for trades only
trade_logger = logging.getLogger("SolanaTrading.Trades")
trade_logger.setLevel(logging.INFO)
trade_handler = logging.FileHandler(log_dir / 'solana_trades.log')
trade_handler.setLevel(logging.INFO)
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)
trade_logger.propagate = False  # Don't send to parent logger

# Configure engine logger with all handlers
engine_logger = logging.getLogger("SolanaTradingEngine")
engine_logger.setLevel(logging.INFO)
engine_logger.addHandler(main_handler)
engine_logger.addHandler(console_handler)
engine_logger.addHandler(error_handler)


class HealthServer:
    """HTTP server for health and metrics endpoints"""

    def __init__(self, app: 'SolanaTradingApplication', host: str = "0.0.0.0", port: int = 8082):
        self.app = app
        self.host = host
        self.port = port
        self.web_app = web.Application()
        self._setup_routes()
        self.runner = None

    def _setup_routes(self):
        """Setup HTTP routes"""
        self.web_app.router.add_get('/health', self.health_handler)
        self.web_app.router.add_get('/healthz', self.health_handler)  # Kubernetes standard
        self.web_app.router.add_get('/ready', self.ready_handler)
        self.web_app.router.add_get('/metrics', self.metrics_handler)
        self.web_app.router.add_get('/stats', self.stats_handler)
        self.web_app.router.add_post('/close-position', self.close_position_handler)
        self.web_app.router.add_post('/close-all-positions', self.close_all_positions_handler)

    async def health_handler(self, request):
        """Liveness probe endpoint"""
        if self.app.engine:
            health = await self.app.engine.get_health()
            status = 200 if health.get('status') == 'healthy' else 503
            return web.json_response(health, status=status)
        return web.json_response({'status': 'initializing'}, status=503)

    async def ready_handler(self, request):
        """Readiness probe endpoint"""
        if self.app.engine and self.app.engine.is_running:
            health = await self.app.engine.get_health()
            if health.get('rpc_connected') and health.get('risk_can_trade'):
                return web.json_response({'ready': True, **health}, status=200)
        return web.json_response({'ready': False}, status=503)

    async def metrics_handler(self, request):
        """Prometheus-style metrics endpoint"""
        metrics = []
        if self.app.engine:
            stats = await self.app.engine.get_stats()
            health = await self.app.engine.get_health()

            # Trading metrics
            metrics.append(f'solana_trades_total {stats.get("total_trades", 0)}')
            metrics.append(f'solana_winning_trades {stats.get("winning_trades", 0)}')
            metrics.append(f'solana_losing_trades {stats.get("losing_trades", 0)}')
            metrics.append(f'solana_active_positions {stats.get("active_positions", 0)}')
            metrics.append(f'solana_daily_trades {stats.get("daily_trades", 0)}')

            # PnL metrics (extract numeric value)
            total_pnl = stats.get("total_pnl", "0.0000 SOL").split()[0]
            daily_pnl = stats.get("daily_pnl", "0.0000 SOL").split()[0]
            metrics.append(f'solana_total_pnl_sol {float(total_pnl)}')
            metrics.append(f'solana_daily_pnl_sol {float(daily_pnl)}')

            # Health metrics
            metrics.append(f'solana_engine_running {1 if health.get("engine_running") else 0}')
            metrics.append(f'solana_rpc_connected {1 if health.get("rpc_connected") else 0}')
            metrics.append(f'solana_dry_run {1 if health.get("dry_run") else 0}')
            metrics.append(f'solana_risk_can_trade {1 if health.get("risk_can_trade") else 0}')
            metrics.append(f'solana_wallet_balance_sol {health.get("wallet_balance_sol", 0)}')

            # RPC latency metrics
            latencies = health.get('rpc_latencies', {})
            for rpc_url, latency in latencies.items():
                # Sanitize RPC URL for Prometheus label
                safe_url = rpc_url[:30].replace('/', '_').replace(':', '_').replace('.', '_')
                metrics.append(f'solana_rpc_latency_ms{{rpc="{safe_url}"}} {latency}')

        return web.Response(text='\n'.join(metrics), content_type='text/plain')

    async def stats_handler(self, request):
        """Full statistics endpoint"""
        if self.app.engine:
            stats = await self.app.engine.get_stats()
            health = await self.app.engine.get_health()
            return web.json_response({
                'module': 'solana',
                'stats': stats,
                'health': health,
                'timestamp': datetime.now().isoformat()
            })
        return web.json_response({'error': 'Engine not initialized'}, status=503)

    async def close_position_handler(self, request):
        """Close a specific position by token mint address"""
        try:
            if not self.app.engine:
                return web.json_response({'success': False, 'error': 'Engine not initialized'}, status=503)

            data = await request.json()
            mint = data.get('mint')

            if not mint:
                return web.json_response({'success': False, 'error': 'Missing mint address'}, status=400)

            # Check if position exists
            if mint not in self.app.engine.positions:
                return web.json_response({'success': False, 'error': f'No position found for {mint}'}, status=404)

            # Close the position
            position = self.app.engine.positions[mint]
            await self.app.engine._close_position(mint, "manual_close")

            logger.info(f"📤 Manually closed position: {position.token_symbol}")

            return web.json_response({
                'success': True,
                'message': f'Position {position.token_symbol} closed successfully',
                'mint': mint
            })

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def close_all_positions_handler(self, request):
        """Close all open positions"""
        try:
            if not self.app.engine:
                return web.json_response({'success': False, 'error': 'Engine not initialized'}, status=503)

            positions_count = len(self.app.engine.positions)

            if positions_count == 0:
                return web.json_response({'success': True, 'message': 'No positions to close', 'closed': 0})

            # Close all positions
            await self.app.engine.close_all_positions()

            logger.info(f"📤 Manually closed all {positions_count} positions")

            return web.json_response({
                'success': True,
                'message': f'Closed {positions_count} positions successfully',
                'closed': positions_count
            })

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def start(self):
        """Start the HTTP server"""
        self.runner = web.AppRunner(self.web_app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"📡 Health server started on http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP server"""
        if self.runner:
            await self.runner.cleanup()


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
        self.health_server = None
        self.shutdown_event = asyncio.Event()
        self.logger = logger
        self.config_manager = None
        self.db_pool = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Default config (will be overridden from DB if available)
        self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
        self.strategies = os.getenv('SOLANA_STRATEGIES', 'jupiter,drift').split(',')
        self.max_positions = int(os.getenv('SOLANA_MAX_POSITIONS', '3'))
        self.health_port = int(os.getenv('SOLANA_HEALTH_PORT', '8082'))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def _init_database(self):
        """Initialize database connection pool"""
        try:
            import asyncpg
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                self.db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
                self.logger.info("✅ Database pool initialized")
                return True
        except Exception as e:
            self.logger.warning(f"Database not available, using defaults: {e}")
        return False

    async def _init_config_manager(self):
        """Initialize and load configuration from database"""
        try:
            from modules.solana_strategies.solana_config_manager import SolanaConfigManager

            self.config_manager = SolanaConfigManager(self.db_pool)
            await self.config_manager.initialize()

            # Override settings from database config
            if self.config_manager._cache_loaded:
                self.strategies = self.config_manager.get_enabled_strategies()
                if not self.strategies:
                    # Default to jupiter if nothing enabled
                    self.strategies = ['jupiter']
                self.max_positions = self.config_manager.max_positions
                self.logger.info(f"✅ Loaded config from database: strategies={self.strategies}, max_positions={self.max_positions}")

        except Exception as e:
            self.logger.warning(f"Config manager not available: {e}")

    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 Solana Trading Bot Starting...")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Time: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)

            # Initialize database connection
            await self._init_database()

            # Initialize config manager (loads settings from DB)
            await self._init_config_manager()

            self.logger.info(f"RPC: {self.rpc_url}")
            self.logger.info(f"Strategies: {', '.join(self.strategies)}")

            # Import Solana engine
            from solana_trading.core.solana_engine import SolanaTradingEngine

            # Initialize engine with config manager
            self.engine = SolanaTradingEngine(
                rpc_url=self.rpc_url,
                strategies=self.strategies,
                max_positions=self.max_positions,
                mode=self.mode,
                config_manager=self.config_manager
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

            # Start health server
            self.health_server = HealthServer(self, port=self.health_port)
            await self.health_server.start()

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

            # Stop health server first
            if self.health_server:
                self.logger.info("Stopping health server...")
                await self.health_server.stop()

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
