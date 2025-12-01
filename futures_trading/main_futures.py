#!/usr/bin/env python3
"""
Futures Trading Bot - Main Entry Point
Handles futures trading on Binance and Bybit exchanges

Features:
- Multi-exchange support (Binance, Bybit)
- Leverage trading with risk management
- Position monitoring and auto-liquidation protection
- Technical indicator-based strategies
- HTTP health/metrics endpoints for monitoring
- Database-backed configuration via FuturesConfigManager

Configuration Architecture:
- All trading parameters are stored in the database (config_settings table)
- Only sensitive data (API keys, secrets) is read from .env
- Settings page writes/reads from database via API endpoints
- Config changes take effect without restart (hot-reload)
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
import asyncpg

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Configure futures-specific logging with multiple log files
log_dir = Path("logs/futures")
log_dir.mkdir(parents=True, exist_ok=True)

# Custom filter for trade-related messages (positions and stats only, not signal analysis)
class TradeLogFilter(logging.Filter):
    """Filter to capture only actual trade events (opened, closed, stats)"""
    # Keywords that indicate actual trade events
    INCLUDE_KEYWORDS = [
        'opened position', 'closed position', 'position opened', 'position closed',
        'entry price', 'exit price', 'stop loss hit', 'take profit hit',
        'liquidation', 'daily pnl', 'daily stats', 'trading stats',
        'total trades', 'win rate', 'total pnl', 'net pnl',
        '✅ opened', '✅ closed', '🎯', '💰', '📉 closed',
        'unrealized p&l', 'max drawdown', 'active positions',
        'reason:', 'entry:', 'pnl:', 'fees:',
        '📊 daily stats', '=====', 'futures trading module'
    ]
    # Keywords that should be excluded even if they contain trade-related words
    EXCLUDE_KEYWORDS = [
        'signal analysis', 'scanning', 'rejected', 'combined score',
        'rsi:', 'macd:', 'volume:', 'bollinger:', 'ema:', 'trend:',
        'analyzing', 'fetching candles', 'checking', 'evaluating'
    ]

    def filter(self, record):
        msg = record.getMessage().lower()
        # First check exclusions
        if any(excl in msg for excl in self.EXCLUDE_KEYWORDS):
            return False
        # Then check inclusions
        return any(incl in msg for incl in self.INCLUDE_KEYWORDS)

# Log format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 1. Main log file - all logs (UTF-8 encoding for emoji support)
main_handler = logging.FileHandler(log_dir / 'futures_trading.log', encoding='utf-8')
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(formatter)

# 2. Errors log file - only ERROR and WARNING
error_handler = logging.FileHandler(log_dir / 'futures_errors.log', encoding='utf-8')
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)

# 3. Trades log file - only trade-related messages
trades_handler = logging.FileHandler(log_dir / 'futures_trades.log', encoding='utf-8')
trades_handler.setLevel(logging.INFO)
trades_handler.setFormatter(formatter)
trades_handler.addFilter(TradeLogFilter())

# 4. Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add all handlers to root logger
root_logger.addHandler(main_handler)
root_logger.addHandler(error_handler)
root_logger.addHandler(trades_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger("FuturesTrading")


class HealthServer:
    """HTTP server for health and metrics endpoints"""

    def __init__(self, app: 'FuturesTradingApplication', host: str = "0.0.0.0", port: int = 8081):
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
        self.web_app.router.add_get('/positions', self.positions_handler)
        self.web_app.router.add_get('/trades', self.trades_handler)
        self.web_app.router.add_post('/position/close', self.close_position_handler)
        self.web_app.router.add_post('/positions/close-all', self.close_all_positions_handler)
        self.web_app.router.add_post('/trading/unblock', self.unblock_trading_handler)
        self.web_app.router.add_get('/trading/status', self.trading_status_handler)

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
            if health.get('exchange_connected') and health.get('risk_can_trade'):
                return web.json_response({'ready': True, **health}, status=200)
        return web.json_response({'ready': False}, status=503)

    async def metrics_handler(self, request):
        """Prometheus-style metrics endpoint"""
        metrics = []
        if self.app.engine:
            stats = await self.app.engine.get_stats()
            health = await self.app.engine.get_health()

            # Trading metrics
            metrics.append(f'futures_trades_total {stats.get("total_trades", 0)}')
            metrics.append(f'futures_winning_trades {stats.get("winning_trades", 0)}')
            metrics.append(f'futures_losing_trades {stats.get("losing_trades", 0)}')
            metrics.append(f'futures_active_positions {stats.get("active_positions", 0)}')
            metrics.append(f'futures_daily_trades {stats.get("daily_trades", 0)}')

            # PnL metrics (extract numeric value)
            total_pnl = stats.get("total_pnl", "$0.00").replace("$", "").replace(",", "")
            daily_pnl = stats.get("daily_pnl", "$0.00").replace("$", "").replace(",", "")
            metrics.append(f'futures_total_pnl_usd {float(total_pnl)}')
            metrics.append(f'futures_daily_pnl_usd {float(daily_pnl)}')

            # Health metrics
            metrics.append(f'futures_engine_running {1 if health.get("engine_running") else 0}')
            metrics.append(f'futures_exchange_connected {1 if health.get("exchange_connected") else 0}')
            metrics.append(f'futures_dry_run {1 if health.get("dry_run") else 0}')
            metrics.append(f'futures_risk_can_trade {1 if health.get("risk_can_trade") else 0}')
            metrics.append(f'futures_consecutive_losses {health.get("consecutive_losses", 0)}')

        return web.Response(text='\n'.join(metrics), content_type='text/plain')

    async def stats_handler(self, request):
        """Full statistics endpoint"""
        if self.app.engine:
            stats = await self.app.engine.get_stats()
            health = await self.app.engine.get_health()
            return web.json_response({
                'module': 'futures',
                'stats': stats,
                'health': health,
                'timestamp': datetime.now().isoformat()
            })
        return web.json_response({'error': 'Engine not initialized'}, status=503)

    async def positions_handler(self, request):
        """Get all active positions with full details"""
        if self.app.engine:
            positions = []
            for symbol, pos in self.app.engine.active_positions.items():
                positions.append({
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'size': pos.size,
                    'notional_value': pos.notional_value,
                    'leverage': pos.leverage,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'trailing_stop': pos.metadata.get('trailing_stop'),
                    'liquidation_price': pos.liquidation_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'opened_at': pos.opened_at.isoformat(),
                    'is_simulated': pos.is_simulated
                })
            return web.json_response({
                'success': True,
                'positions': positions,
                'count': len(positions)
            })
        return web.json_response({'error': 'Engine not initialized'}, status=503)

    async def trades_handler(self, request):
        """Get recent closed trades from database"""
        if not self.app.engine:
            return web.json_response({'error': 'Engine not initialized'}, status=503)

        # Get limit from query params, default to 50
        limit = int(request.query.get('limit', 50))
        trades = []

        # First try to get from database for persistence
        if self.app.db_pool:
            try:
                async with self.app.db_pool.acquire() as conn:
                    records = await conn.fetch("""
                        SELECT
                            id, symbol, side, entry_price, exit_price, size,
                            notional_value, leverage, pnl, pnl_pct, fees, net_pnl,
                            exit_reason, entry_time, exit_time, duration_seconds,
                            is_simulated, exchange, network
                        FROM futures_trades
                        WHERE is_simulated = $1
                          AND exchange = $2
                          AND network = $3
                        ORDER BY exit_time DESC
                        LIMIT $4
                    """, self.app.engine.dry_run, self.app.engine.exchange,
                    'testnet' if self.app.engine.testnet else 'mainnet', limit)

                    for record in records:
                        trades.append({
                            'trade_id': str(record['id']),
                            'symbol': record['symbol'],
                            'side': record['side'],
                            'entry_price': float(record['entry_price']),
                            'exit_price': float(record['exit_price']),
                            'size': float(record['size']),
                            'notional_value': float(record['notional_value']),
                            'leverage': record['leverage'],
                            'pnl': float(record['pnl']),
                            'pnl_pct': float(record['pnl_pct']),
                            'fees': float(record['fees']),
                            'net_pnl': float(record['net_pnl']),
                            'opened_at': record['entry_time'].isoformat() if record['entry_time'] else None,
                            'closed_at': record['exit_time'].isoformat() if record['exit_time'] else None,
                            'close_reason': record['exit_reason'],
                            'is_simulated': record['is_simulated'],
                            'duration_seconds': record['duration_seconds']
                        })
            except Exception as e:
                logger.warning(f"Could not fetch trades from DB: {e}")

        # Fallback to in-memory trade history if DB failed
        if not trades and self.app.engine.trade_history:
            for trade in self.app.engine.trade_history[-limit:]:
                trades.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'notional_value': trade.notional_value,
                    'leverage': trade.leverage,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'fees': trade.fees,
                    'opened_at': trade.opened_at.isoformat(),
                    'closed_at': trade.closed_at.isoformat(),
                    'close_reason': trade.close_reason,
                    'is_simulated': trade.is_simulated
                })

        return web.json_response({
            'success': True,
            'trades': trades,
            'count': len(trades)
        })

    async def close_position_handler(self, request):
        """Close a specific position"""
        if not self.app.engine:
            return web.json_response({'error': 'Engine not initialized'}, status=503)

        try:
            data = await request.json()
            symbol = data.get('symbol')

            if not symbol:
                return web.json_response({
                    'success': False,
                    'error': 'Symbol is required'
                }, status=400)

            if symbol not in self.app.engine.active_positions:
                return web.json_response({
                    'success': False,
                    'error': f'No active position for {symbol}'
                }, status=404)

            # Close the position
            await self.app.engine._close_position(symbol, "manual_close")
            logger.info(f"✅ Position {symbol} closed via API")

            return web.json_response({
                'success': True,
                'message': f'Position {symbol} closed successfully'
            })

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def close_all_positions_handler(self, request):
        """Close all active positions"""
        if not self.app.engine:
            return web.json_response({'error': 'Engine not initialized'}, status=503)

        try:
            positions_count = len(self.app.engine.active_positions)

            if positions_count == 0:
                return web.json_response({
                    'success': True,
                    'message': 'No positions to close'
                })

            # Close all positions
            await self.app.engine.close_all_positions()
            logger.info(f"✅ All {positions_count} positions closed via API")

            return web.json_response({
                'success': True,
                'message': f'Closed {positions_count} positions successfully'
            })

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def trading_status_handler(self, request):
        """Get current trading status including block status"""
        if not self.app.engine:
            return web.json_response({'error': 'Engine not initialized'}, status=503)

        risk = self.app.engine.risk_metrics

        return web.json_response({
            'success': True,
            'trading_blocked': not risk.can_trade,
            'block_reasons': [],
            'daily_pnl': risk.daily_pnl,
            'daily_loss_limit': risk.daily_loss_limit,
            'consecutive_losses': risk.consecutive_losses,
            'max_consecutive_losses': 5,
            'risk_level': risk.risk_level,
            'daily_trades': risk.daily_trades,
            'can_trade': risk.can_trade
        })

    async def unblock_trading_handler(self, request):
        """Reset daily loss and unblock trading"""
        if not self.app.engine:
            return web.json_response({'error': 'Engine not initialized'}, status=503)

        try:
            data = await request.json() if request.content_length else {}
            reset_type = data.get('reset_type', 'all')  # 'daily', 'consecutive', 'all'

            risk = self.app.engine.risk_metrics
            old_daily_pnl = risk.daily_pnl
            old_consecutive = risk.consecutive_losses

            if reset_type in ['daily', 'all']:
                risk.daily_pnl = 0.0
                risk.daily_trades = 0
                logger.info(f"🔓 Daily PnL reset from ${old_daily_pnl:.2f} to $0.00")

            if reset_type in ['consecutive', 'all']:
                risk.consecutive_losses = 0
                logger.info(f"🔓 Consecutive losses reset from {old_consecutive} to 0")

            risk.last_reset = datetime.now()

            return web.json_response({
                'success': True,
                'message': f'Trading unblocked ({reset_type} reset)',
                'previous_daily_pnl': old_daily_pnl,
                'previous_consecutive_losses': old_consecutive,
                'can_trade': risk.can_trade
            })

        except Exception as e:
            logger.error(f"Error unblocking trading: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

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
        self.health_server = None
        self.shutdown_event = asyncio.Event()
        self.logger = logger
        self.db_pool = None
        self.config_manager = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Health server port (from .env since it's infrastructure config)
        self.health_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def _init_database(self):
        """Initialize database connection pool"""
        try:
            db_url = os.getenv('DATABASE_URL', os.getenv('DB_URL'))
            if db_url:
                self.db_pool = await asyncpg.create_pool(
                    db_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=60
                )
                self.logger.info("✅ Database connection pool created")
            else:
                self.logger.warning("⚠️ No DATABASE_URL, using default configuration")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.logger.warning("Using default configuration")

    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 Futures Trading Bot Starting...")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Time: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)

            # Initialize database connection
            await self._init_database()

            # Initialize config manager (loads settings from database)
            from modules.futures_trading.futures_config_manager import FuturesConfigManager
            self.config_manager = FuturesConfigManager(db_pool=self.db_pool)
            await self.config_manager.initialize()

            # Get configuration from database
            general_config = self.config_manager.get_general()
            leverage_config = self.config_manager.get_leverage()
            position_config = self.config_manager.get_position()

            self.logger.info(f"Exchange: {general_config.exchange.upper()}")
            self.logger.info(f"Testnet: {general_config.testnet}")
            self.logger.info(f"Leverage: {leverage_config.default_leverage}x")
            self.logger.info(f"Max Positions: {position_config.max_positions}")

            # Import futures engine
            from futures_trading.core.futures_engine import FuturesTradingEngine

            # Initialize engine with config manager and database pool
            self.engine = FuturesTradingEngine(
                config_manager=self.config_manager,
                mode=self.mode,
                db_pool=self.db_pool
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

            # Start health server
            self.health_server = HealthServer(self, port=self.health_port)
            await self.health_server.start()

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
                    # Log in a format that TradeLogFilter will capture for futures_trades.log
                    self.logger.info("=" * 60)
                    self.logger.info("📊 DAILY STATS - Futures Trading Module")
                    self.logger.info(f"   Total Trades: {stats.get('total_trades', 0)}")
                    self.logger.info(f"   Win Rate: {stats.get('win_rate', '0%')}")
                    self.logger.info(f"   Total PnL: {stats.get('net_pnl', '$0.00')}")
                    self.logger.info(f"   Daily PnL: {stats.get('daily_pnl', '$0.00')}")
                    self.logger.info(f"   Active Positions: {stats.get('active_positions', 0)}")
                    self.logger.info(f"   Unrealized P&L: {stats.get('unrealized_pnl', '$0.00')}")
                    self.logger.info(f"   Max Drawdown: {stats.get('max_drawdown_pct', '0%')}")
                    self.logger.info("=" * 60)

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
                # Check if we should close positions on shutdown
                close_on_shutdown = os.getenv('CLOSE_POSITIONS_ON_SHUTDOWN', 'false').lower() == 'true'

                if self.mode == "production" and close_on_shutdown:
                    self.logger.info("Closing all open positions (CLOSE_POSITIONS_ON_SHUTDOWN=true)...")
                    await self.engine.close_all_positions()
                elif self.mode == "production":
                    active_count = len(self.engine.active_positions) if self.engine.active_positions else 0
                    if active_count > 0:
                        self.logger.warning(f"⚠️ Keeping {active_count} positions open on shutdown")
                        self.logger.warning("Set CLOSE_POSITIONS_ON_SHUTDOWN=true to close positions on restart")

                self.logger.info("Stopping engine...")
                await self.engine.shutdown()

            # Close database pool
            if self.db_pool:
                self.logger.info("Closing database pool...")
                await self.db_pool.close()

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
