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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Configure futures-specific logging with multiple log files
# Aligned with main.py's per-subprocess stdout/stderr log dir
# (self.name "Futures Trading" → logs/futures_trading/). Earlier this
# was logs/futures/ which left the operator with two parallel dirs
# for the same module and confused log-tail commands.
log_dir = Path("logs/futures_trading")
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
        '✅ opened', '✅ closed', '✅ position', '🎯', '💰', '📉 closed',
        'unrealized p&l', 'max drawdown', 'active positions',
        'reason:', 'entry:', 'pnl:', 'fees:',
        '📊 daily stats', '=====', 'futures trading module',
        # Add SL/TP related keywords for trade entry logging
        '🛑 stop loss:', '📈 take profit:', 'tp1:', 'tp2:', 'tp3:', 'tp4:',
        'notional:', 'leverage:', 'partial close', 'fully closed',
        '🔵 [dry_run]', '🔴 [dry_run]', '🟢 partial'
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

from logging.handlers import RotatingFileHandler

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller
except ImportError:
    get_telegram_controller = None

# Log format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 1. Main log file - all logs (UTF-8 encoding for emoji support) - Rotated 10MB
main_handler = RotatingFileHandler(
    log_dir / 'futures_trading.log',
    encoding='utf-8',
    maxBytes=10*1024*1024,
    backupCount=5
)
main_handler.setLevel(logging.INFO)
main_handler.setFormatter(formatter)

# 2. Errors log file - only ERROR and WARNING - Rotated 10MB
error_handler = RotatingFileHandler(
    log_dir / 'futures_errors.log',
    encoding='utf-8',
    maxBytes=10*1024*1024,
    backupCount=5
)
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(formatter)

# 3. Trades log file - only trade-related messages - Rotated 10MB
trades_handler = RotatingFileHandler(
    log_dir / 'futures_trades.log',
    encoding='utf-8',
    maxBytes=10*1024*1024,
    backupCount=5
)
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
                # IMPORTANT: Use dictionary key 'symbol' for consistency with close_position_handler lookup
                positions.append({
                    'position_id': pos.position_id,
                    'symbol': symbol,  # Use dict key, not pos.symbol, to ensure close works
                    'side': pos.side.value,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'size': pos.size,
                    'original_size': pos.original_size,
                    'notional_value': pos.notional_value,
                    'leverage': pos.leverage,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'tp_levels': pos.tp_levels,  # Multiple TP levels with hit status
                    'trailing_stop': pos.metadata.get('trailing_stop'),
                    'trailing_stop_price': pos.trailing_stop_price,
                    'highest_price': pos.highest_price,
                    'lowest_price': pos.lowest_price,
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

            # Clean the symbol - strip whitespace
            if symbol:
                symbol = symbol.strip()

            # Debug logging to trace position lookup
            active_keys = list(self.app.engine.active_positions.keys())
            logger.info(f"🔍 Close position request: symbol='{symbol}' (len={len(symbol) if symbol else 0})")
            logger.info(f"🔍 Active positions: {active_keys}")
            for key in active_keys:
                logger.info(f"🔍   Key: '{key}' (len={len(key)}), match={key == symbol}")

            if not symbol:
                return web.json_response({
                    'success': False,
                    'error': 'Symbol is required'
                }, status=400)

            # Try exact match first
            matched_symbol = None
            if symbol in self.app.engine.active_positions:
                matched_symbol = symbol
            else:
                # Try case-insensitive match as fallback
                for key in self.app.engine.active_positions.keys():
                    if key.upper() == symbol.upper():
                        matched_symbol = key
                        logger.info(f"🔍 Found case-insensitive match: '{key}' for '{symbol}'")
                        break

            if not matched_symbol:
                logger.warning(f"❌ Position {symbol} not found. Active: {active_keys}")
                return web.json_response({
                    'success': False,
                    'error': f'Position {symbol} not found in active positions. Active positions: {active_keys}',
                    'already_closed': True,
                    'active_positions': active_keys
                }, status=404)

            # Close the position using the matched symbol
            await self.app.engine._close_position(matched_symbol, "manual_close")
            logger.info(f"✅ Position {matched_symbol} closed via API")

            return web.json_response({
                'success': True,
                'message': f'Position {matched_symbol} closed successfully'
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
        self.telegram_controller = None
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
            # Use Docker secrets or environment
            try:
                from security.docker_secrets import get_database_url
                db_url = get_database_url()
            except ImportError:
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
                self.logger.warning("⚠️ No database credentials found, using default configuration")
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

            # Initialize secrets manager with database pool FIRST (before any config managers)
            try:
                from security.secrets_manager import secrets
                secrets.initialize(self.db_pool)
                self.logger.info("✅ Secrets manager initialized with database")
            except Exception as e:
                self.logger.warning(f"Could not initialize secrets manager with database: {e}")

            # Initialize config manager (loads settings from database)
            from modules.futures_trading.config.futures_config_manager import FuturesConfigManager
            self.config_manager = FuturesConfigManager(db_pool=self.db_pool)
            await self.config_manager.initialize()

            # Get configuration from database
            general_config = self.config_manager.get_general()
            leverage_config = self.config_manager.get_leverage()
            position_config = self.config_manager.get_position()

            self.logger.info(f"Exchange: {general_config.exchange.upper()}")
            # Wave-13: clarify this is the raw DB value — engine applies
            # env/DRY_RUN override precedence before live-trading decisions.
            self.logger.info(
                f"Testnet (DB config): {general_config.testnet} "
                f"— engine resolves final value via env/DRY_RUN precedence"
            )
            self.logger.info(f"Leverage: {leverage_config.default_leverage}x")
            self.logger.info(f"Max Positions: {position_config.max_positions}")

            # Import futures engine
            from modules.futures_trading.core.futures_engine import FuturesTradingEngine

            # Initialize engine with config manager and database pool
            self.engine = FuturesTradingEngine(
                config_manager=self.config_manager,
                mode=self.mode,
                db_pool=self.db_pool
            )

            await self.engine.initialize()

            # Phase 2 #5: inject FuturesRiskManager so MB-17's entry validator is live.
            # FuturesTradingEngine.set_risk_manager (commit c53b73b) had no caller until now.
            #
            # FUT-RM-01 (b1b8df9 follow-up): commit b1b8df9 patched the dashboard
            # wrapper (FuturesTradingModule.initialize) to merge futures_max_leverage
            # into the risk dict, but this main_futures.py subprocess path bypassed
            # the wrapper entirely and still constructed FuturesRiskManager from
            # FuturesRiskConfig only — which has NO max_leverage / max_positions /
            # max_total_exposure fields. Result: the risk manager silently
            # defaulted to max_leverage=3, max_positions=3, max_total_exposure=500
            # regardless of operator settings, and every entry at >3x was rejected.
            # Now we merge leverage_config + position_config into risk_cfg so all
            # three sources of truth land on the runtime manager.
            try:
                from modules.futures_trading.futures_risk_manager import FuturesRiskManager
                risk_cfg: dict = {}
                if self.config_manager is not None:
                    if hasattr(self.config_manager, 'get_risk'):
                        risk_obj = self.config_manager.get_risk()
                        if hasattr(risk_obj, 'model_dump'):
                            risk_cfg = risk_obj.model_dump()
                        elif hasattr(risk_obj, 'dict'):
                            risk_cfg = risk_obj.dict()
                        elif isinstance(risk_obj, dict):
                            risk_cfg = risk_obj
                    # Merge leverage cap from FuturesLeverageConfig.
                    lev_cfg = self.config_manager.get_leverage()
                    if lev_cfg and getattr(lev_cfg, 'max_leverage', None) is not None:
                        risk_cfg['max_leverage'] = int(lev_cfg.max_leverage)
                    # FUT-RM-08: per-symbol leverage cap overrides.
                    if lev_cfg:
                        overrides = getattr(lev_cfg, 'max_leverage_overrides', None)
                        if overrides:
                            risk_cfg['max_leverage_overrides'] = dict(overrides)
                    # Merge position cap + exposure cap from FuturesPositionConfig.
                    pos_cfg = self.config_manager.get_position()
                    if pos_cfg:
                        if getattr(pos_cfg, 'max_positions', None) is not None:
                            risk_cfg['max_positions'] = int(pos_cfg.max_positions)
                        # max_total_exposure: prefer explicit cap, fall back to
                        # capital_allocation * default_leverage as a soft cap.
                        cap_alloc = float(getattr(pos_cfg, 'capital_allocation', 0) or 0)
                        if cap_alloc > 0:
                            risk_cfg.setdefault(
                                'max_total_exposure',
                                cap_alloc * float(getattr(lev_cfg, 'default_leverage', 1) or 1)
                            )
                    # liquidation_buffer on settings page is a percentage
                    # (e.g. 20 = 20%); FuturesRiskManager expects a fraction.
                    lb = risk_cfg.get('liquidation_buffer')
                    if lb is not None and float(lb) > 1.0:
                        risk_cfg['liquidation_buffer'] = float(lb) / 100.0
                    # FUT-RM-05: pull directional funding thresholds from
                    # FuturesFundingConfig and forward to the risk manager.
                    if hasattr(self.config_manager, 'get_funding'):
                        try:
                            fund_cfg = self.config_manager.get_funding()
                            if fund_cfg is not None:
                                for key in (
                                    'skip_long_funding_bps',
                                    'skip_short_funding_bps',
                                ):
                                    val = getattr(fund_cfg, key, None)
                                    if val is not None:
                                        risk_cfg[key] = float(val)
                        except Exception as e:
                            self.logger.debug(f"funding config not available: {e}")
                self.risk_manager = FuturesRiskManager(risk_cfg)
                self.engine.set_risk_manager(self.risk_manager)

                # FUT-RM-02: startup assertion — runtime risk manager must
                # reflect the DB-configured caps. Surfaces silent regressions.
                self._assert_runtime_risk_matches_config()

                self.logger.info(
                    "✅ FuturesRiskManager injected — MB-17 entry validator is active "
                    f"(max_leverage={self.risk_manager.max_leverage}, "
                    f"max_positions={self.risk_manager.max_positions}, "
                    f"max_total_exposure=${self.risk_manager.max_total_exposure:.2f})"
                )
            except Exception as e:
                self.logger.warning(f"FuturesRiskManager wiring failed: {e}; engine will run without validator (legacy behaviour)")

            self.logger.info("✅ Futures trading engine initialized")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}", exc_info=True)
            raise

    def _assert_runtime_risk_matches_config(self) -> None:
        """FUT-RM-02: Startup assertion that runtime FuturesRiskManager caps
        match the DB-backed config. A mismatch means the wiring above silently
        regressed (e.g. someone re-introduced the hard-coded default=3).

        Soft-fail by default: logs a loud error + an alert if available. The
        operator can flip FUTURES_RISK_ASSERT_HARD=1 in .env to raise instead
        (preferred for CI / canary runs)."""
        try:
            if not self.config_manager or not self.risk_manager:
                return
            lev_cfg = self.config_manager.get_leverage()
            pos_cfg = self.config_manager.get_position()
            expected_max_lev = int(getattr(lev_cfg, 'max_leverage', 0) or 0)
            expected_max_pos = int(getattr(pos_cfg, 'max_positions', 0) or 0)
            mismatches = []
            if expected_max_lev and int(self.risk_manager.max_leverage) != expected_max_lev:
                mismatches.append(
                    f"max_leverage runtime={self.risk_manager.max_leverage} "
                    f"db={expected_max_lev}"
                )
            if expected_max_pos and int(self.risk_manager.max_positions) != expected_max_pos:
                mismatches.append(
                    f"max_positions runtime={self.risk_manager.max_positions} "
                    f"db={expected_max_pos}"
                )
            if mismatches:
                msg = (
                    "FUTURES RISK CONFIG MISMATCH at startup: "
                    + "; ".join(mismatches)
                )
                self.logger.error(msg)
                hard = os.getenv('FUTURES_RISK_ASSERT_HARD', '').strip().lower() in (
                    '1', 'true', 'yes'
                )
                if hard:
                    raise RuntimeError(msg)
            else:
                self.logger.info(
                    f"Runtime risk caps match DB config: "
                    f"max_leverage={expected_max_lev}, "
                    f"max_positions={expected_max_pos}"
                )
        except RuntimeError:
            raise
        except Exception as e:
            self.logger.warning(f"Risk-config assertion errored (non-fatal): {e}")

    # Wave-15: retry interval (seconds) when initialize() fails after all
    # per-attempt retries in _init_binance are exhausted.
    _INIT_RETRY_INTERVAL_SECONDS: int = 300  # 5 minutes

    async def _initialize_with_degraded_loop(self) -> None:
        """Wave-15: call initialize(); if it fails, enter a degraded retry loop.

        Rather than crashing the entire module process when Binance's API is
        transiently unreachable, we log a clear ERROR and retry
        _INIT_RETRY_INTERVAL_SECONDS later so the module self-heals once the
        exchange comes back. The shutdown_event is checked between retries so
        a SIGTERM / kill-switch still exits cleanly.
        """
        attempt = 0
        while not self.shutdown_event.is_set():
            attempt += 1
            try:
                await self.initialize()
                if attempt > 1:
                    self.logger.info(
                        f"[wave-15] Futures engine initialized successfully on retry attempt {attempt}."
                    )
                return  # success — caller proceeds normally
            except Exception as exc:
                retry_in = self._INIT_RETRY_INTERVAL_SECONDS
                self.logger.error(
                    f"[wave-15] Futures engine init failed (attempt {attempt}): "
                    f"{type(exc).__name__}: {exc}. "
                    f"Module entering degraded mode — will retry in {retry_in}s. "
                    f"(Set FUTURES_MODULE_ENABLED=false to suppress.)",
                    exc_info=True,
                )
                # Reset engine so state is clean on the next attempt.
                self.engine = None
                # Wait, but wake immediately on shutdown.
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=float(retry_in),
                    )
                except asyncio.TimeoutError:
                    pass  # timeout expired — loop and retry
        # shutdown_event was set during the wait; raise so run() exits cleanly.
        raise asyncio.CancelledError("Shutdown requested during degraded init loop")

    async def run(self):
        """Main application loop"""
        try:
            self.logger.info("Starting Futures Trading Bot...")

            # Wave-15: use degraded-mode loop so a transient Binance outage at
            # startup doesn't hard-crash the module and show "not initialized"
            # forever on the dashboard.
            await self._initialize_with_degraded_loop()

            # Start health server
            self.health_server = HealthServer(self, port=self.health_port)
            await self.health_server.start()

            # Initialize Telegram controller for remote control (credentials from secrets manager)
            if get_telegram_controller:
                try:
                    # ISSUE-19 fix (mirrors AI cca8d94): the shared
                    # TelegramBotController.__init__ resolves the token via the
                    # SYNCHRONOUS secrets.get(), which deliberately short-circuits
                    # the DB lookup when called from inside a running event loop
                    # (see secrets_manager._get_from_database_sync "use get_async()
                    # in async context" guard at L341). Operators who store the
                    # token in the Secure Engine (encrypted secure_credentials DB
                    # row) but NOT in .env therefore got a None token and the
                    # controller logged "TELEGRAM_BOT_TOKEN not set". DEX/Solana
                    # never hit this because they resolve the token via get_async.
                    # Pre-warm the secrets cache here with the async resolver so the
                    # controller's sync get() finds the value in secrets._cache.
                    try:
                        from security.secrets_manager import secrets as _secrets
                        for _k in (
                            'TELEGRAM_BOT_TOKEN',
                            'TELEGRAM_CHAT_ID',
                            'TELEGRAM_ADMIN_IDS',
                        ):
                            await _secrets.get_async(_k, log_access=False)
                    except Exception as _warm_err:
                        self.logger.debug(
                            f"Telegram secret pre-warm skipped (non-fatal): {_warm_err}"
                        )
                    # Wave-11 FIX 2: tag with module_name so the singleton's
                    # start_polling() honors TELEGRAM_POLL_OWNER (default dashboard).
                    self.telegram_controller = get_telegram_controller(self.db_pool, module_name='futures')
                    if await self.telegram_controller.initialize():
                        self.telegram_controller.register_module(
                            name='futures',
                            engine=self.engine,
                            start_method='run',
                            stop_method='shutdown',
                            positions_attr='active_positions'
                        )
                        await self.telegram_controller.start_polling()
                        self.logger.info("Telegram remote control enabled")
                        await self.telegram_controller.notify(
                            "Futures Trading Bot started. Send /help for commands.",
                            priority="normal"
                        )
                except Exception as e:
                    self.logger.warning(f"Telegram controller failed to initialize: {e}")

            self.logger.info("Starting futures trading engine...")

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

        except asyncio.CancelledError:
            self.logger.info("Futures bot cancelled (shutdown during init retry).")

        except Exception as e:
            self.logger.error(f"Critical error in main loop: {e}", exc_info=True)

        finally:
            await self.shutdown()

    async def _status_reporter(self):
        """Periodically report system status"""
        last_hourly_log = datetime.now()

        while not self.shutdown_event.is_set():
            try:
                if self.engine:
                    stats = await self.engine.get_stats()

                    # Check if an hour has passed for DAILY STATS logging to futures_trades.log
                    now = datetime.now()
                    if (now - last_hourly_log).total_seconds() >= 3600:  # 1 hour
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
                        last_hourly_log = now

                await asyncio.sleep(120)  # Check every 2 minutes

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

            # Stop Telegram controller first
            if self.telegram_controller:
                self.logger.info("Stopping Telegram controller...")
                await self.telegram_controller.notify("Futures bot shutting down...", priority="high")
                await self.telegram_controller.stop_polling()

            # Stop health server
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

    # Handle dry-run — honors per-module override (Phase 3 A5).
    # Precedence: --dry-run CLI > FUTURES_DRY_RUN env > DRY_RUN env >
    # default True. DB row check happens later in the engine once the
    # config manager has connected.
    from core.dry_run import resolve_module_dry_run
    is_dry_run = resolve_module_dry_run('futures', default=True)
    if args.dry_run:
        is_dry_run = True
    # Mirror the resolved value into DRY_RUN so downstream `os.getenv
    # ('DRY_RUN')` checks in the engine pick up the per-module flip.
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
