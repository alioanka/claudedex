"""
Enhanced Dashboard for DexScreener Trading Bot
Professional web-based monitoring, control, and analytics interface
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
import io
import aiohttp
from aiohttp import web
import aiohttp_cors
from aiohttp_sse import sse_response
import socketio
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
import numpy as np
import csv
from config.config_manager import PortfolioConfig
from pydantic.types import SecretStr

# Authentication imports
try:
    from auth.auth_service import AuthService
    from auth.middleware import auth_middleware_factory, require_auth, require_admin
    from monitoring.auth_routes import AuthRoutes
    AUTH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Authentication system not available: {e}")
    AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardEndpoints:
    """Enhanced dashboard with comprehensive features"""
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 config: Optional[Dict] = None,
                 trading_engine = None,
                 portfolio_manager = None,
                 order_manager = None,
                 risk_manager = None,
                 alerts_system = None,
                 config_manager = None,
                 db_manager = None,
                 module_manager = None,
                 analytics_engine = None,
                 advanced_alerts = None):

        self.host = host
        self.port = port
        self.config = config or {}

        # Core components
        self.engine = trading_engine
        self.portfolio = portfolio_manager
        self.orders = order_manager
        self.risk = risk_manager
        self.alerts = alerts_system
        self.config_mgr = config_manager
        self.db = db_manager
        self.db_pool = db_manager.pool if db_manager else None  # Add db_pool for easy access
        self.module_manager = module_manager  # Module manager for Phase 1 & 2

        # Phase 4: Advanced Analytics & Alerts
        self.analytics_engine = analytics_engine
        self.advanced_alerts = advanced_alerts

        # Authentication
        self.auth_service = None
        self.auth_enabled = False

        # Web application
        self.app = web.Application()
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        self.sio.attach(self.app)

        # Template engine
        self.jinja_env = Environment(
            loader=FileSystemLoader('dashboard/templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # ========== WALLET BALANCE CACHING ==========
        # Cache wallet balances to prevent instability from intermittent RPC failures
        self._wallet_cache = {}
        self._wallet_cache_time = None
        self._wallet_cache_ttl = 60  # Cache for 60 seconds
        self._price_cache = {'ETH': 3500, 'BNB': 600, 'MATIC': 0.80, 'SOL': 200}
        self._price_cache_time = None

        # ========== SIMULATOR DATA CACHING ==========
        # Cache simulator data to prevent rapid polling of module /stats endpoints
        self._simulator_cache = None
        self._simulator_cache_time = None
        self._simulator_cache_ttl = 3  # Cache for 3 seconds

        # Setup routes
        self._setup_routes()
        self._setup_socketio()

        # Setup module routes if module manager available
        if self.module_manager:
            self._setup_module_routes()
        else:
            # Register fallback routes for module pages when module_manager is not available
            self._setup_fallback_module_routes()

        # Setup analytics routes if analytics engine available
        if self.analytics_engine:
            self._setup_analytics_routes()

        # Register startup handler for auth initialization
        self.app.on_startup.append(self._on_startup)

        # Start update tasks
        asyncio.create_task(self._broadcast_loop())

        # In-memory storage for backtests
        self.backtests = {}

    @staticmethod
    def _serialize_decimals(obj):
        """Convert Decimal objects to float for JSON serialization"""
        if isinstance(obj, dict):
            return {k: DashboardEndpoints._serialize_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DashboardEndpoints._serialize_decimals(item) for item in obj]
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    async def _on_startup(self, app):
        """Called when app starts - initialize auth system before serving requests"""
        try:
            logger.info("=" * 80)
            logger.info("🚀 APP STARTUP HANDLER CALLED")
            logger.info("=" * 80)
            logger.info("Initializing authentication system...")
            await self._initialize_auth_async()
        except Exception as e:
            logger.error(f"❌ CRITICAL: Startup handler failed: {e}", exc_info=True)

    async def _initialize_auth_async(self):
        """Initialize authentication system asynchronously"""
        logger.info("📍 Starting auth initialization...")

        if not AUTH_AVAILABLE:
            logger.error("❌ Authentication system not available - bcrypt/pyotp not installed")
            logger.error("   Install required packages: pip install bcrypt pyotp")
            logger.error("   SECURITY WARNING: Dashboard will be UNSECURED!")
            return

        logger.info(f"✅ Auth modules available (bcrypt, pyotp)")

        # Wait for database to be ready (with retries)
        max_retries = 10
        retry_delay = 1

        logger.info(f"🔍 Checking database connection...")
        logger.info(f"   self.db = {self.db}")
        logger.info(f"   self.db type = {type(self.db)}")

        if self.db:
            logger.info(f"   hasattr(pool) = {hasattr(self.db, 'pool')}")
            if hasattr(self.db, 'pool'):
                logger.info(f"   self.db.pool = {self.db.pool}")

        for attempt in range(max_retries):
            if self.db and hasattr(self.db, 'pool') and self.db.pool:
                logger.info(f"✅ Database connection ready (attempt {attempt + 1}/{max_retries})")
                break

            logger.warning(f"⏳ Waiting for database connection... (attempt {attempt + 1}/{max_retries})")
            logger.warning(f"   DB status: self.db={bool(self.db)}, has pool={hasattr(self.db, 'pool') if self.db else False}, pool={getattr(self.db, 'pool', None) if self.db else None}")
            await asyncio.sleep(retry_delay)
        else:
            logger.error("❌ Database not available after retries - CANNOT INITIALIZE AUTH")
            logger.error("   SECURITY WARNING: Dashboard will be UNSECURED!")
            logger.error(f"   Final DB state: {self.db}")
            return

        try:
            logger.info("🔐 Initializing authentication system...")
            logger.info(f"   Database pool: {self.db.pool}")

            # Create auth service
            logger.info("   Creating AuthService...")
            self.auth_service = AuthService(
                db_pool=self.db.pool,
                session_timeout=3600,  # 1 hour
                max_failed_attempts=5
            )
            logger.info("   ✅ AuthService created")

            # Store in app for middleware access
            self.app['auth_service'] = self.auth_service
            logger.info("   ✅ Auth service stored in app")

            # Setup auth routes
            logger.info("   Setting up auth routes...")
            AuthRoutes(self.app, self.auth_service)
            logger.info("   ✅ Auth routes registered")

            # Add auth middleware at beginning of middleware stack
            # Check if already added to avoid duplicates
            logger.info("   Checking middleware stack...")
            middleware_names = [m.__name__ if hasattr(m, '__name__') else str(m) for m in self.app.middlewares]
            logger.info(f"   Current middlewares: {middleware_names}")

            if 'middleware' not in middleware_names:
                self.app.middlewares.insert(0, auth_middleware_factory)
                logger.info("   ✅ Auth middleware registered")
            else:
                logger.info("   ⚠️  Auth middleware already registered")

            self.auth_enabled = True

            logger.info("=" * 80)
            logger.info("✅ AUTHENTICATION SYSTEM ACTIVE")
            logger.info(f"   Login URL: http://{self.host}:{self.port}/login")
            logger.info("   Default credentials: admin / admin123")
            logger.info("   ⚠️  CHANGE PASSWORD IMMEDIATELY AFTER FIRST LOGIN!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Failed to initialize authentication system: {e}", exc_info=True)
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            logger.error("   SECURITY WARNING: Dashboard will be UNSECURED!")
            self.auth_enabled = False

    async def login_placeholder(self, request):
        """Temporary login page shown while auth system initializes"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Loading...</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .message {
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    text-align: center;
                }
                .spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <meta http-equiv="refresh" content="3">
        </head>
        <body>
            <div class="message">
                <div class="spinner"></div>
                <h2>Authentication System Loading...</h2>
                <p>Please wait while we initialize the security system.</p>
                <p><small>This page will refresh automatically.</small></p>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    def _setup_routes(self):
        """Setup all routes"""

        # ✅ ADD: Add error handling middleware FIRST
        self.app.middlewares.append(self.error_handler_middleware)

        # Static files
        self.app.router.add_static('/static', 'dashboard/static', name='static')

        # ⚠️ Auth routes (including /login) will be added during startup (see _on_startup handler)
        # This ensures the login route is available when the app serves requests

        # Pages - all will be protected by auth middleware if enabled
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/dashboard', self.dashboard_page)
        self.app.router.add_get('/trades', self.trades_page)
        self.app.router.add_get('/positions', self.positions_page)
        self.app.router.add_get('/performance', self.performance_page)
        self.app.router.add_get('/settings', self.settings_page)
        self.app.router.add_get('/reports', self.reports_page)
        self.app.router.add_get('/backtest', self.backtest_page)
        self.app.router.add_get('/logs', self.logs_page)
        self.app.router.add_get('/analysis', self.analysis_page)
        self.app.router.add_get('/analytics', self.analytics_page)
        self.app.router.add_get('/simulator', self.simulator_page)

        # API - Data endpoints
        self.app.router.add_get('/api/dashboard/summary', self.api_dashboard_summary)
        self.app.router.add_get('/api/logs', self.api_get_logs)
        self.app.router.add_get('/api/analysis', self.api_get_analysis)
        self.app.router.add_get('/api/insights', self.api_get_insights)
        self.app.router.add_get('/api/trades/recent', self.api_recent_trades)
        self.app.router.add_get('/api/trades/history', self.api_trade_history)
        self.app.router.add_get('/api/trades/export/{format}', self.api_export_trades)
        self.app.router.add_get('/api/positions/open', self.api_open_positions)
        self.app.router.add_get('/api/positions/history', self.api_positions_history)
        self.app.router.add_get('/api/performance/metrics', self.api_performance_metrics)
        self.app.router.add_get('/api/performance/charts', self.api_performance_charts)
        self.app.router.add_get('/api/alerts/recent', self.api_recent_alerts)
        self.app.router.add_get('/api/risk/metrics', self.api_risk_metrics)
        self.app.router.add_get('/api/wallets/balances', self.api_wallet_balances)

        # API - Simulator
        self.app.router.add_get('/api/simulator/data', self.api_simulator_data)
        self.app.router.add_get('/api/simulator/export', self.api_simulator_export)

        # API - Bot control
        self.app.router.add_post('/api/bot/start', self.api_bot_start)
        self.app.router.add_post('/api/bot/stop', self.api_bot_stop)
        self.app.router.add_post('/api/bot/restart', self.api_bot_restart)
        self.app.router.add_post('/api/bot/emergency_exit', self.api_emergency_exit)
        self.app.router.add_get('/api/bot/status', self.api_bot_status)
        
        # API - Settings
        self.app.router.add_get('/api/settings/all', self.api_get_settings)
        self.app.router.add_post('/api/settings/update', self.api_update_settings)
        self.app.router.add_post('/api/settings/revert', self.api_revert_settings)
        self.app.router.add_get('/api/settings/history', self.api_settings_history)

        # API - Module-specific Settings (database-backed)
        self.app.router.add_get('/api/settings/futures', self.api_get_futures_settings)
        self.app.router.add_post('/api/settings/futures', self.api_save_futures_settings)
        self.app.router.add_get('/api/settings/solana', self.api_get_solana_settings)
        self.app.router.add_post('/api/settings/solana', self.api_save_solana_settings)

        # API - Sensitive Configuration (Admin only)
        self.app.router.add_get('/api/settings/sensitive/list', require_auth(require_admin(self.api_list_sensitive_configs)))
        self.app.router.add_get('/api/settings/sensitive/{key}', require_auth(require_admin(self.api_get_sensitive_config)))
        self.app.router.add_post('/api/settings/sensitive', require_auth(require_admin(self.api_set_sensitive_config)))
        self.app.router.add_delete('/api/settings/sensitive/{key}', require_auth(require_admin(self.api_delete_sensitive_config)))

        # API - Futures Position Management (proxy to Futures module)
        self.app.router.add_get('/api/futures/positions', self.api_futures_positions)
        self.app.router.add_get('/api/futures/trades', self.api_futures_trades)
        self.app.router.add_post('/api/futures/position/close', self.api_futures_close_position)
        self.app.router.add_post('/api/futures/positions/close-all', self.api_futures_close_all_positions)
        self.app.router.add_get('/api/futures/trading/status', self.api_futures_trading_status)
        self.app.router.add_post('/api/futures/trading/unblock', self.api_futures_trading_unblock)

        # API - Trading controls
        self.app.router.add_post('/api/trade/execute', self.api_execute_trade)
        self.app.router.add_post('/api/position/close', self.api_close_position)
        self.app.router.add_post('/api/position/modify', self.api_modify_position)
        self.app.router.add_post('/api/order/cancel', self.api_cancel_order)
        
        # API - Reports
        self.app.router.add_post('/api/reports/generate', self.api_generate_report)
        self.app.router.add_get('/api/reports/export/{format}', self.api_export_report)
        self.app.router.add_get('/api/reports/custom', self.api_custom_report)
        
        # API - Backtesting
        self.app.router.add_post('/api/backtest/run', self.api_run_backtest)
        self.app.router.add_get('/api/backtest/results/{test_id}', self.api_backtest_results)
        
        # API - Strategy
        self.app.router.add_get('/api/strategy/parameters', self.api_get_strategy_params)
        self.app.router.add_post('/api/strategy/parameters', self.api_update_strategy_params)
        
        # API - Portfolio Trading Block Management
        self.app.router.add_get('/api/portfolio/block-status', self.api_get_block_status)
        self.app.router.add_post('/api/portfolio/reset-block', self.api_reset_block)

        # SSE for real-time updates
        self.app.router.add_get('/api/stream', self.sse_handler)
        
        # Setup CORS - EXCLUDE socket.io routes
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to routes, but skip socket.io routes
        for route in list(self.app.router.routes()):
            # Skip socket.io routes (they handle CORS internally)
            if not route.resource or '/socket.io/' not in str(route.resource):
                try:
                    cors.add(route)
                except ValueError as e:
                    # Skip routes that already have OPTIONS handler
                    logger.debug(f"Skipping CORS for route: {route.resource}")

    def _setup_module_routes(self):
        """Setup module management routes"""
        try:
            from monitoring.module_routes import ModuleRoutes

            logger.info("Setting up module management routes...")

            # Create module routes handler
            module_routes = ModuleRoutes(
                module_manager=self.module_manager,
                jinja_env=self.jinja_env
            )

            # Setup all module routes
            module_routes.setup_routes(self.app)

            logger.info("✅ Module management routes initialized")

        except Exception as e:
            logger.error(f"Failed to setup module routes: {e}", exc_info=True)

    def _setup_analytics_routes(self):
        """Setup analytics routes"""
        try:
            from monitoring.analytics_routes import AnalyticsRoutes

            logger.info("Setting up analytics routes...")

            # Create analytics routes handler
            analytics_routes = AnalyticsRoutes(
                analytics_engine=self.analytics_engine,
                jinja_env=self.jinja_env
            )

            # Setup all analytics routes
            analytics_routes.setup_routes(self.app)

            logger.info("✅ Analytics routes initialized")

        except Exception as e:
            logger.error(f"Failed to setup analytics routes: {e}", exc_info=True)
            logger.warning("Module management will not be available")

    def _setup_fallback_module_routes(self):
        """Setup fallback routes for module pages when module_manager is not available"""
        logger.info("Setting up fallback module routes (module_manager not available)")

        # DEX Module Pages
        self.app.router.add_get('/dex/dashboard', self._fallback_dex_dashboard)
        self.app.router.add_get('/dex/settings', self._fallback_dex_settings)

        # Futures Module Pages
        self.app.router.add_get('/futures/dashboard', self._fallback_futures_dashboard)
        self.app.router.add_get('/futures/positions', self._fallback_futures_positions)
        self.app.router.add_get('/futures/trades', self._fallback_futures_trades)
        self.app.router.add_get('/futures/performance', self._fallback_futures_performance)
        self.app.router.add_get('/futures/settings', self._fallback_futures_settings)

        # Solana Module Pages
        self.app.router.add_get('/solana/dashboard', self._fallback_solana_dashboard)
        self.app.router.add_get('/solana/positions', self._fallback_solana_positions)
        self.app.router.add_get('/solana/trades', self._fallback_solana_trades)
        self.app.router.add_get('/solana/performance', self._fallback_solana_performance)
        self.app.router.add_get('/solana/settings', self._fallback_solana_settings)

        # Module Control and Modules Pages
        self.app.router.add_get('/module-control', self._fallback_module_control)
        self.app.router.add_get('/modules', self._fallback_modules_page)

        # API endpoints that return empty data when module_manager is unavailable
        self.app.router.add_get('/api/modules', self._fallback_api_modules)

        # Module control API endpoints (enable/disable/pause/start)
        self.app.router.add_post('/api/modules/{module}/enable', self._api_module_enable)
        self.app.router.add_post('/api/modules/{module}/disable', self._api_module_disable)
        self.app.router.add_post('/api/modules/{module}/pause', self._api_module_pause)
        self.app.router.add_post('/api/modules/{module}/start', self._api_module_start)

        logger.info("✅ Fallback module routes registered")

    # Fallback page handlers
    async def _fallback_dex_dashboard(self, request):
        template = self.jinja_env.get_template('dashboard.html')
        return web.Response(text=template.render(page='dex_dashboard'), content_type='text/html')

    async def _fallback_dex_settings(self, request):
        template = self.jinja_env.get_template('settings_dex.html')
        return web.Response(text=template.render(page='dex_settings'), content_type='text/html')

    async def _fallback_futures_dashboard(self, request):
        template = self.jinja_env.get_template('dashboard_futures.html')
        return web.Response(text=template.render(page='futures_dashboard'), content_type='text/html')

    async def _fallback_futures_positions(self, request):
        template = self.jinja_env.get_template('positions_futures.html')
        return web.Response(text=template.render(page='futures_positions'), content_type='text/html')

    async def _fallback_futures_trades(self, request):
        template = self.jinja_env.get_template('trades_futures.html')
        return web.Response(text=template.render(page='futures_trades'), content_type='text/html')

    async def _fallback_futures_performance(self, request):
        template = self.jinja_env.get_template('performance_futures.html')
        return web.Response(text=template.render(page='futures_performance'), content_type='text/html')

    async def _fallback_futures_settings(self, request):
        template = self.jinja_env.get_template('settings_futures.html')
        return web.Response(text=template.render(page='futures_settings'), content_type='text/html')

    async def _fallback_solana_dashboard(self, request):
        template = self.jinja_env.get_template('dashboard_solana.html')
        return web.Response(text=template.render(page='solana_dashboard'), content_type='text/html')

    async def _fallback_solana_positions(self, request):
        template = self.jinja_env.get_template('positions_solana.html')
        return web.Response(text=template.render(page='solana_positions'), content_type='text/html')

    async def _fallback_solana_trades(self, request):
        template = self.jinja_env.get_template('trades_solana.html')
        return web.Response(text=template.render(page='solana_trades'), content_type='text/html')

    async def _fallback_solana_performance(self, request):
        template = self.jinja_env.get_template('performance_solana.html')
        return web.Response(text=template.render(page='solana_performance'), content_type='text/html')

    async def _fallback_solana_settings(self, request):
        template = self.jinja_env.get_template('settings_solana.html')
        return web.Response(text=template.render(page='solana_settings'), content_type='text/html')

    async def _fallback_module_control(self, request):
        template = self.jinja_env.get_template('module_control.html')
        return web.Response(text=template.render(page='module_control'), content_type='text/html')

    async def _fallback_modules_page(self, request):
        template = self.jinja_env.get_template('modules.html')
        return web.Response(text=template.render(page='modules', modules=[], metrics={}), content_type='text/html')

    async def _fallback_api_modules(self, request):
        """Return module data from .env settings and database"""
        # Reload .env from the correct path to get latest values
        env_path = self._get_env_file_path()
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded .env from: {env_path}")
        else:
            logger.warning(f".env file not found at: {env_path}")

        # Read module enabled status from .env (default to DEX enabled, others disabled)
        dex_enabled = os.getenv('DEX_MODULE_ENABLED', 'true').lower() == 'true'
        futures_enabled = os.getenv('FUTURES_MODULE_ENABLED', 'false').lower() == 'true'
        solana_enabled = os.getenv('SOLANA_MODULE_ENABLED', 'false').lower() == 'true'

        logger.info(f"Module status from env: DEX={dex_enabled}, Futures={futures_enabled}, Solana={solana_enabled}")

        # Check if Futures and Solana modules are actually running by contacting their health endpoints
        import aiohttp
        futures_running = False
        solana_running = False
        futures_health_data = {}
        solana_health_data = {}

        try:
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{futures_port}/health', timeout=2) as resp:
                    if resp.status == 200:
                        futures_running = True
                        futures_health_data = await resp.json()
                        logger.info(f"Futures module is running: {futures_health_data}")

                # Also fetch stats to get metrics
                if futures_running:
                    async with session.get(f'http://localhost:{futures_port}/stats', timeout=5) as resp:
                        if resp.status == 200:
                            stats_data = await resp.json()
                            stats = stats_data.get('stats', stats_data)
                            futures_metrics['total_trades'] = stats.get('total_trades', 0)
                            futures_metrics['positions'] = stats.get('active_positions', 0)
                            # Parse PnL which may be a string like "$-0.76"
                            net_pnl = stats.get('net_pnl', '$0.00')
                            if isinstance(net_pnl, str):
                                net_pnl = float(net_pnl.replace('$', '').replace(',', ''))
                            futures_metrics['pnl'] = net_pnl
                            # Parse win rate which may be a string like "33.3%"
                            win_rate = stats.get('win_rate', '0%')
                            if isinstance(win_rate, str):
                                win_rate = float(win_rate.replace('%', ''))
                            futures_metrics['win_rate'] = win_rate
                            logger.info(f"Futures metrics from stats: {futures_metrics}")
        except Exception as e:
            logger.debug(f"Futures module not reachable: {e}")

        try:
            solana_port = int(os.getenv('SOLANA_HEALTH_PORT', '8082'))
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{solana_port}/health', timeout=2) as resp:
                    if resp.status == 200:
                        solana_running = True
                        solana_health_data = await resp.json()
                        logger.info(f"Solana module is running: {solana_health_data}")
        except Exception as e:
            logger.debug(f"Solana module not reachable: {e}")

        # Get metrics from database
        dex_metrics = {'total_trades': 0, 'pnl': 0.0, 'positions': 0, 'win_rate': 0.0}
        futures_metrics = {'total_trades': 0, 'pnl': 0.0, 'positions': 0, 'win_rate': 0.0}
        solana_metrics = {'total_trades': 0, 'pnl': 0.0, 'positions': 0, 'win_rate': 0.0}

        # Count positions by chain FROM ENGINE (same source as Open Positions API)
        if self.engine and hasattr(self.engine, 'active_positions') and self.engine.active_positions:
            try:
                for token_addr, pos in self.engine.active_positions.items():
                    chain = (pos.get('chain') or pos.get('network') or 'SOLANA').upper()
                    # Calculate unrealized P&L from position
                    entry_price = float(pos.get('entry_price', 0))
                    current_price = float(pos.get('current_price', entry_price))
                    amount = float(pos.get('amount', 0))
                    unrealized_pnl = (current_price - entry_price) * amount

                    if chain == 'SOLANA':
                        solana_metrics['positions'] += 1
                        solana_metrics['pnl'] += unrealized_pnl
                    elif chain in ['ETHEREUM', 'BSC', 'BASE', 'POLYGON', 'ARBITRUM']:
                        dex_metrics['positions'] += 1
                        dex_metrics['pnl'] += unrealized_pnl

                logger.info(f"Module metrics from engine positions: DEX={dex_metrics}, Solana={solana_metrics}")
            except Exception as e:
                logger.warning(f"Error getting positions from engine: {e}")

        if self.db:
            try:
                # Get trades by chain/module
                trades = await self.db.get_recent_trades(limit=1000)
                logger.info(f"Got {len(trades) if trades else 0} trades from database")

                # Define chain mappings - handle various formats
                SOLANA_CHAINS = {'SOLANA', 'SOL'}
                DEX_CHAINS = {'ETHEREUM', 'ETH', 'BSC', 'BNB', 'BASE', 'POLYGON', 'MATIC', 'ARBITRUM', 'ARB', 'AVALANCHE', 'AVAX', 'FANTOM', 'FTM'}

                if trades:
                    # Log unique chains for debugging
                    unique_chains = set((t.get('chain') or t.get('network') or 'UNKNOWN') for t in trades)
                    logger.info(f"Unique chains in trades: {unique_chains}")

                    closed_trades = [t for t in trades if t.get('status') == 'closed']
                    logger.info(f"Closed trades: {len(closed_trades)} out of {len(trades)}")

                    for trade in trades:
                        chain = (trade.get('chain') or trade.get('network') or '').upper()
                        is_closed = trade.get('status') == 'closed'
                        pnl = float(trade.get('profit_loss') or 0) if is_closed else 0

                        if chain in SOLANA_CHAINS:
                            solana_metrics['total_trades'] += 1
                            if is_closed:
                                solana_metrics['pnl'] += pnl
                        elif chain in DEX_CHAINS or chain:  # Count any non-empty chain as DEX if not Solana
                            dex_metrics['total_trades'] += 1
                            if is_closed:
                                dex_metrics['pnl'] += pnl

                    # Calculate win rates
                    dex_wins = 0
                    dex_closed = 0
                    solana_wins = 0
                    solana_closed = 0

                    for t in trades:
                        chain = (t.get('chain') or t.get('network') or '').upper()
                        is_closed = t.get('status') == 'closed'
                        is_win = is_closed and float(t.get('profit_loss') or 0) > 0

                        if chain in SOLANA_CHAINS:
                            if is_closed:
                                solana_closed += 1
                                if is_win:
                                    solana_wins += 1
                        elif chain in DEX_CHAINS or chain:
                            if is_closed:
                                dex_closed += 1
                                if is_win:
                                    dex_wins += 1

                    dex_metrics['win_rate'] = (dex_wins / dex_closed * 100) if dex_closed > 0 else 0
                    solana_metrics['win_rate'] = (solana_wins / solana_closed * 100) if solana_closed > 0 else 0

                    logger.info(f"Module metrics: DEX trades={dex_metrics['total_trades']}, closed={dex_closed}, wins={dex_wins}, pnl={dex_metrics['pnl']:.2f}")
                    logger.info(f"Module metrics: Solana trades={solana_metrics['total_trades']}, closed={solana_closed}, wins={solana_wins}, pnl={solana_metrics['pnl']:.2f}")

            except Exception as e:
                logger.error(f"Error getting module metrics from DB: {e}", exc_info=True)

        # Get capital allocations from module config files
        dex_capital = 500.0  # Default
        futures_capital = 300.0  # Default
        solana_capital = 400.0  # Default

        try:
            import yaml
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'modules')

            # Read DEX config
            dex_config_path = os.path.join(config_dir, 'dex_trading.yaml')
            if os.path.exists(dex_config_path):
                with open(dex_config_path, 'r') as f:
                    dex_config = yaml.safe_load(f)
                    if dex_config and 'capital' in dex_config:
                        dex_capital = float(dex_config['capital'].get('allocation', 500.0))

            # Read Futures config
            futures_config_path = os.path.join(config_dir, 'futures_trading.yaml')
            if os.path.exists(futures_config_path):
                with open(futures_config_path, 'r') as f:
                    futures_config = yaml.safe_load(f)
                    if futures_config and 'capital' in futures_config:
                        futures_capital = float(futures_config['capital'].get('allocation', 300.0))

            # Read Solana config
            solana_config_path = os.path.join(config_dir, 'solana_strategies.yaml')
            if os.path.exists(solana_config_path):
                with open(solana_config_path, 'r') as f:
                    solana_config = yaml.safe_load(f)
                    if solana_config and 'capital' in solana_config:
                        solana_capital = float(solana_config['capital'].get('allocation', 400.0))

            logger.info(f"Capital from config: DEX=${dex_capital}, Futures=${futures_capital}, Solana=${solana_capital}")
        except Exception as e:
            logger.warning(f"Error reading module config files: {e}")

        # Determine actual status for each module
        # DEX: RUNNING if enabled (DEX runs in same process as dashboard)
        dex_status = 'RUNNING' if dex_enabled else 'DISABLED'

        # Futures: Check if actually running via health endpoint
        if futures_running:
            futures_status = 'RUNNING'
        elif futures_enabled:
            futures_status = 'ENABLED'  # Enabled but not running
        else:
            futures_status = 'DISABLED'

        # Solana: Check if actually running via health endpoint
        if solana_running:
            solana_status = 'RUNNING'
        elif solana_enabled:
            solana_status = 'ENABLED'  # Enabled but not running
        else:
            solana_status = 'DISABLED'

        return web.json_response({
            'success': True,
            'data': {
                'modules': {
                    'dex_trading': {
                        'name': 'DEX Trading',
                        'enabled': dex_enabled,
                        'status': dex_status,
                        'capital': dex_capital,
                        'metrics': dex_metrics
                    },
                    'futures_trading': {
                        'name': 'Futures Trading',
                        'enabled': futures_enabled,
                        'status': futures_status,
                        'capital': futures_capital,
                        'metrics': futures_metrics,
                        'health': futures_health_data
                    },
                    'solana_strategies': {
                        'name': 'Solana Strategies',
                        'enabled': solana_enabled,
                        'status': solana_status,
                        'capital': solana_capital,
                        'metrics': solana_metrics,
                        'health': solana_health_data
                    }
                }
            }
        })

    def _get_env_file_path(self) -> str:
        """Get the absolute path to the .env file in the project root"""
        # Get project root from monitoring directory (go up one level)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, '.env')
        logger.debug(f"Using .env file at: {env_path}")
        return env_path

    def _update_env_file(self, key: str, value: str) -> bool:
        """Update a key in the .env file"""
        try:
            env_path = self._get_env_file_path()
            logger.info(f"Updating .env file at: {env_path}, setting {key}={value}")

            # Read existing content or create default if file doesn't exist
            lines = []
            key_found = False

            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith(f'{key}='):
                            lines.append(f'{key}={value}\n')
                            key_found = True
                        else:
                            lines.append(line)
            else:
                # Create default .env with all module settings
                logger.info(f"Creating new .env file at: {env_path}")
                lines = [
                    "# Trading Module Configuration\n",
                    "DEX_MODULE_ENABLED=true\n",
                    "FUTURES_MODULE_ENABLED=false\n",
                    "SOLANA_MODULE_ENABLED=false\n",
                    "\n",
                    "# Environment\n",
                    "ENVIRONMENT=development\n",
                    "DEBUG=false\n",
                ]
                # Check if the key we're setting is in the defaults
                for i, line in enumerate(lines):
                    if line.strip().startswith(f'{key}='):
                        lines[i] = f'{key}={value}\n'
                        key_found = True
                        break

            # If key wasn't found, add it
            if not key_found:
                lines.append(f'{key}={value}\n')

            # Write back
            with open(env_path, 'w') as f:
                f.writelines(lines)

            logger.info(f"Successfully wrote .env file with {key}={value}")

            # Reload environment variables
            load_dotenv(env_path, override=True)

            # Also update os.environ directly for immediate effect
            os.environ[key] = value
            logger.info(f"Environment variable {key} set to: {os.environ.get(key)}")

            return True
        except Exception as e:
            logger.error(f"Error updating .env file: {e}", exc_info=True)
            return False

    async def _api_module_enable(self, request):
        """Enable a module by updating .env"""
        module = request.match_info.get('module', '')

        module_env_map = {
            'dex_trading': 'DEX_MODULE_ENABLED',
            'futures_trading': 'FUTURES_MODULE_ENABLED',
            'solana_strategies': 'SOLANA_MODULE_ENABLED'
        }

        if module not in module_env_map:
            return web.json_response({'error': f'Unknown module: {module}'}, status=400)

        env_key = module_env_map[module]
        if self._update_env_file(env_key, 'true'):
            logger.info(f"Module {module} enabled via API")
            return web.json_response({'success': True, 'message': f'{module} enabled'})
        else:
            return web.json_response({'error': 'Failed to update .env file'}, status=500)

    async def _api_module_disable(self, request):
        """Disable a module by updating .env"""
        module = request.match_info.get('module', '')

        module_env_map = {
            'dex_trading': 'DEX_MODULE_ENABLED',
            'futures_trading': 'FUTURES_MODULE_ENABLED',
            'solana_strategies': 'SOLANA_MODULE_ENABLED'
        }

        if module not in module_env_map:
            return web.json_response({'error': f'Unknown module: {module}'}, status=400)

        env_key = module_env_map[module]
        if self._update_env_file(env_key, 'false'):
            logger.info(f"Module {module} disabled via API")
            return web.json_response({'success': True, 'message': f'{module} disabled'})
        else:
            return web.json_response({'error': 'Failed to update .env file'}, status=500)

    async def _api_module_pause(self, request):
        """Pause a module (sets to paused state)"""
        module = request.match_info.get('module', '')
        # For now, pause acts like disable - in a full implementation this would set a PAUSED state
        logger.info(f"Module {module} paused via API")
        return web.json_response({'success': True, 'message': f'{module} paused'})

    async def _api_module_start(self, request):
        """Start/resume a module"""
        module = request.match_info.get('module', '')

        module_env_map = {
            'dex_trading': 'DEX_MODULE_ENABLED',
            'futures_trading': 'FUTURES_MODULE_ENABLED',
            'solana_strategies': 'SOLANA_MODULE_ENABLED'
        }

        if module not in module_env_map:
            return web.json_response({'error': f'Unknown module: {module}'}, status=400)

        env_key = module_env_map[module]
        if self._update_env_file(env_key, 'true'):
            logger.info(f"Module {module} started via API")
            return web.json_response({'success': True, 'message': f'{module} started'})
        else:
            return web.json_response({'error': 'Failed to update .env file'}, status=500)

    def _setup_socketio(self):
        """Setup Socket.IO handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")
            # Send initial data
            await self._send_initial_data(sid)
        
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"Client disconnected: {sid}")
    
    # ==================== PAGE HANDLERS ====================
    
    async def index(self, request):
        """Index page - render main dashboard with modules overview"""
        template = self.jinja_env.get_template('index.html')
        return web.Response(
            text=template.render(page='main_dashboard'),
            content_type='text/html'
        )
    
    async def dashboard_page(self, request):
        """Main dashboard page"""
        template = self.jinja_env.get_template('dashboard.html')
        return web.Response(
            text=template.render(page='dashboard'),
            content_type='text/html'
        )
    
    async def trades_page(self, request):
        """Recent trades page"""
        template = self.jinja_env.get_template('trades.html')
        return web.Response(
            text=template.render(page='trades'),
            content_type='text/html'
        )
    
    async def positions_page(self, request):
        """Positions page"""
        template = self.jinja_env.get_template('positions.html')
        return web.Response(
            text=template.render(page='positions'),
            content_type='text/html'
        )
    
    async def performance_page(self, request):
        """Performance analytics page"""
        template = self.jinja_env.get_template('performance.html')
        return web.Response(
            text=template.render(page='performance'),
            content_type='text/html'
        )

    @web.middleware
    async def error_handler_middleware(self, request, handler):
        """
        Middleware to handle errors gracefully and suppress scanner spam
        """
        try:
            return await handler(request)
        except web.HTTPException as e:
            # Let HTTP exceptions through normally
            raise
        except asyncio.CancelledError:
            # Client disconnected - this is normal, don't log
            raise
        except ConnectionResetError:
            # Client closed connection - normal, don't log
            return web.Response(status=499, text="Client Closed Request")
        except Exception as e:
            # Log actual errors but don't spam
            if not any(x in str(e).lower() for x in ['bad request', 'invalid method', 'connection reset']):
                logger.error(f"Request error: {e}")
            return web.Response(status=500, text="Internal Server Error")
    
    async def settings_page(self, request):
        """Settings management page"""
        template = self.jinja_env.get_template('settings.html')
        return web.Response(
            text=template.render(page='settings'),
            content_type='text/html'
        )
    
    async def reports_page(self, request):
        """Reports generation page"""
        template = self.jinja_env.get_template('reports.html')
        return web.Response(
            text=template.render(page='reports'),
            content_type='text/html'
        )
    
    async def backtest_page(self, request):
        """Backtesting interface page"""
        template = self.jinja_env.get_template('backtest.html')
        return web.Response(
            text=template.render(page='backtest'),
            content_type='text/html'
        )

    async def logs_page(self, request):
        """Logs viewer page"""
        template = self.jinja_env.get_template('logs.html')
        return web.Response(
            text=template.render(page='logs'),
            content_type='text/html'
        )
    
    # ==================== API - DATA ENDPOINTS ====================

    async def analysis_page(self, request):
        """Trade analysis page"""
        template = self.jinja_env.get_template('analysis.html')
        return web.Response(
            text=template.render(page='analysis'),
            content_type='text/html'
        )

    async def analytics_page(self, request):
        """Advanced analytics page"""
        template = self.jinja_env.get_template('analytics.html')
        return web.Response(
            text=template.render(page='analytics'),
            content_type='text/html'
        )

    async def simulator_page(self, request):
        """Trade simulator page for dry-run validation"""
        template = self.jinja_env.get_template('simulator.html')
        return web.Response(
            text=template.render(page='simulator'),
            content_type='text/html'
        )

    async def api_simulator_data(self, request):
        """Get simulator data from all modules including historical trades from DB"""
        try:
            import aiohttp
            from time import time

            # Check cache first to prevent rapid polling of module /stats endpoints
            if (self._simulator_cache is not None and
                self._simulator_cache_time is not None and
                (time() - self._simulator_cache_time) < self._simulator_cache_ttl):
                return web.json_response(self._simulator_cache)

            simulator_data = {
                'futures': None,
                'solana': None,
                'dex': None
            }

            # Try to fetch from Futures module health endpoint
            dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')
            futures_enabled = os.getenv('FUTURES_MODULE_ENABLED', 'false').lower() == 'true'
            try:
                futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{futures_port}/stats', timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Extract stats from nested structure
                            if 'stats' in data:
                                simulator_data['futures'] = data['stats']
                            else:
                                simulator_data['futures'] = data
                            simulator_data['futures']['status'] = 'Active'
                            logger.debug(f"Futures stats fetched: {simulator_data['futures']}")
            except Exception as e:
                logger.warning(f"Could not fetch futures stats: {e}")
                # Fallback: use env variables to determine mode (like DEX does)
                simulator_data['futures'] = {
                    'status': 'Offline' if not futures_enabled else 'Starting',
                    'mode': 'DRY_RUN' if dry_run else 'LIVE',
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': '$0.00',
                    'win_rate': '0%'
                }

            # Fetch futures trades from database for trade log
            if self.db and self.db.pool:
                try:
                    async with self.db.pool.acquire() as conn:
                        futures_trades = await conn.fetch("""
                            SELECT
                                id, symbol, side, entry_price, exit_price, size,
                                notional_value, leverage, pnl, pnl_pct, fees, net_pnl,
                                exit_reason, entry_time, exit_time, duration_seconds,
                                is_simulated, exchange, network
                            FROM futures_trades
                            ORDER BY exit_time DESC
                            LIMIT 100
                        """)

                        trades_list = []
                        for record in futures_trades:
                            trades_list.append({
                                'trade_id': str(record['id']),
                                'symbol': record['symbol'],
                                'side': record['side'],
                                'entry_price': float(record['entry_price']),
                                'exit_price': float(record['exit_price']),
                                'size': float(record['size']),
                                'pnl': float(record['net_pnl'] or record['pnl']),
                                'pnl_pct': float(record['pnl_pct']),
                                'fees': float(record['fees']),
                                'duration_seconds': record['duration_seconds'] or 0,
                                'time': record['exit_time'].isoformat() if record['exit_time'] else None,
                                'exit_reason': record['exit_reason'],
                                'is_simulated': record['is_simulated'],
                                'module': 'futures'
                            })

                        # Initialize futures data if not already set
                        if simulator_data['futures'] is None:
                            simulator_data['futures'] = {}
                        simulator_data['futures']['trades'] = trades_list
                except Exception as e:
                    logger.warning(f"Could not fetch futures trades from DB: {e}")
                    if simulator_data['futures'] is None:
                        simulator_data['futures'] = {}
                    simulator_data['futures']['trades'] = []

            # Try to fetch from Solana module health endpoint
            solana_enabled = os.getenv('SOLANA_MODULE_ENABLED', 'false').lower() == 'true'
            try:
                solana_port = int(os.getenv('SOLANA_HEALTH_PORT', '8082'))
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{solana_port}/stats', timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Extract stats from nested structure
                            if 'stats' in data:
                                simulator_data['solana'] = data['stats']
                            else:
                                simulator_data['solana'] = data
                            simulator_data['solana']['status'] = 'Active'
                            logger.debug(f"Solana stats fetched: {simulator_data['solana']}")
            except Exception as e:
                logger.debug(f"Could not fetch solana stats: {e}")
                # Fallback: use env variables to determine mode (like DEX does)
                simulator_data['solana'] = {
                    'status': 'Offline' if not solana_enabled else 'Starting',
                    'mode': 'DRY_RUN' if dry_run else 'LIVE',
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': '$0.00',
                    'win_rate': '0%'
                }

            # Get DEX data from database (historical trades)
            # dry_run already defined above
            dex_data = {
                'status': 'Offline',
                'mode': 'DRY_RUN' if dry_run else 'LIVE',
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0,
                'win_rate': '0%',
                'trades': []
            }

            # Check if engine is running
            if hasattr(self, 'engine') and self.engine:
                try:
                    is_active = hasattr(self.engine, 'state') and self.engine.state.value == 'running'
                    dex_data['status'] = 'Active' if is_active else 'Stopped'
                except Exception:
                    pass

            # Fetch historical trades from database
            if self.db and self.db.pool:
                try:
                    async with self.db.pool.acquire() as conn:
                        # Get trade statistics
                        stats = await conn.fetchrow("""
                            SELECT
                                COUNT(*) as total_trades,
                                COUNT(*) FILTER (WHERE profit_loss > 0) as winning_trades,
                                COUNT(*) FILTER (WHERE profit_loss <= 0) as losing_trades,
                                COALESCE(SUM(profit_loss), 0) as total_pnl,
                                COALESCE(SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END), 0) as gross_profit,
                                COALESCE(SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END), 0) as gross_loss
                            FROM trades
                            WHERE status = 'closed'
                        """)

                        if stats:
                            total = stats['total_trades'] or 0
                            wins = stats['winning_trades'] or 0
                            losses = stats['losing_trades'] or 0
                            pnl = float(stats['total_pnl'] or 0)
                            gross_profit = float(stats['gross_profit'] or 0)
                            gross_loss = float(stats['gross_loss'] or 0)

                            win_rate = (wins / total * 100) if total > 0 else 0
                            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

                            dex_data['total_trades'] = total
                            dex_data['winning_trades'] = wins
                            dex_data['losing_trades'] = losses
                            dex_data['total_pnl'] = f"${pnl:.2f}"
                            dex_data['win_rate'] = f"{win_rate:.1f}%"
                            dex_data['profit_factor'] = f"{profit_factor:.2f}"

                        # Fetch recent trades for charts/log (last 100)
                        trades_records = await conn.fetch("""
                            SELECT
                                trade_id,
                                token_address,
                                chain,
                                side,
                                entry_price,
                                exit_price,
                                amount,
                                usd_value,
                                profit_loss,
                                profit_loss_percentage,
                                entry_timestamp,
                                exit_timestamp,
                                EXTRACT(EPOCH FROM (exit_timestamp - entry_timestamp))::integer as duration_seconds,
                                gas_fee,
                                status,
                                metadata
                            FROM trades
                            WHERE status = 'closed'
                            ORDER BY exit_timestamp DESC
                            LIMIT 100
                        """)

                        trades_list = []
                        for record in trades_records:
                            metadata = record['metadata'] or {}
                            if isinstance(metadata, str):
                                try:
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}

                            trades_list.append({
                                'trade_id': record['trade_id'],
                                'symbol': metadata.get('token_symbol', record['token_address'][:8] + '...'),
                                'side': record['side'],
                                'entry_price': float(record['entry_price']) if record['entry_price'] else 0,
                                'exit_price': float(record['exit_price']) if record['exit_price'] else 0,
                                'size': float(record['amount']) if record['amount'] else 0,
                                'pnl': float(record['profit_loss']) if record['profit_loss'] else 0,
                                'pnl_pct': float(record['profit_loss_percentage']) if record['profit_loss_percentage'] else 0,
                                'fees': float(record['gas_fee']) if record['gas_fee'] else 0,
                                'duration_seconds': record['duration_seconds'] or 0,
                                'time': record['exit_timestamp'].isoformat() if record['exit_timestamp'] else None,
                                'exit_reason': metadata.get('exit_reason', 'signal'),
                                'is_simulated': dry_run,
                                'module': 'dex'
                            })

                        dex_data['trades'] = trades_list

                except Exception as e:
                    logger.error(f"Error fetching DEX trades from DB: {e}")

            simulator_data['dex'] = dex_data

            # Cache the result to prevent rapid polling
            from time import time
            self._simulator_cache = simulator_data
            self._simulator_cache_time = time()

            return web.json_response(simulator_data)

        except Exception as e:
            logger.error(f"Error fetching simulator data: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_simulator_export(self, request):
        """Export simulator data as JSON file"""
        try:
            import aiohttp
            from datetime import datetime

            export_data = {
                'exported_at': datetime.now().isoformat(),
                'futures': {},
                'solana': {},
                'dex': {}
            }

            # Fetch data from modules
            try:
                futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{futures_port}/stats', timeout=2) as resp:
                        if resp.status == 200:
                            export_data['futures'] = await resp.json()
            except Exception:
                pass

            try:
                solana_port = int(os.getenv('SOLANA_HEALTH_PORT', '8082'))
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{solana_port}/stats', timeout=2) as resp:
                        if resp.status == 200:
                            export_data['solana'] = await resp.json()
            except Exception:
                pass

            if hasattr(self, 'engine') and self.engine:
                try:
                    export_data['dex'] = await self.engine.get_stats()
                except Exception:
                    pass

            # Return as downloadable JSON
            return web.Response(
                body=json.dumps(export_data, indent=2, default=str),
                content_type='application/json',
                headers={
                    'Content-Disposition': f'attachment; filename="simulator_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
                }
            )

        except Exception as e:
            logger.error(f"Error exporting simulator data: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_get_logs(self, request):
        """Get recent log entries from all available log files."""
        try:
            log_dir = "/app/logs"
            log_files = [
                "TradingBot.log",
                "TradingBot_errors.log",
                "TradingBot_trades.log"
            ]
            # Add rotated logs
            for i in range(1, 11):
                log_files.append(f"TradingBot.log.{i}")

            all_lines = []
            for lf in log_files:
                try:
                    full_path = f"{log_dir}/{lf}"
                    with open(full_path, 'r') as f:
                        for line in f:
                            try:
                                log_entry = json.loads(line)
                                # Ensure timestamp exists for sorting
                                if 'timestamp' in log_entry:
                                    all_lines.append(log_entry)
                            except (json.JSONDecodeError, TypeError):
                                pass
                except FileNotFoundError:
                    # It's normal for some rotated files not to exist
                    continue

            # Sort all log entries by timestamp
            all_lines.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # Return last 500 log lines
            return web.json_response({'success': True, 'data': all_lines[:500]})
        except Exception as e:
            logger.error(f"Error reading log files: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def api_get_analysis(self, request):
        """Get trade analysis data"""
        try:
            if not self.db:
                return web.json_response({'error': 'Database connection not available.'}, status=503)

            query = "SELECT * FROM trades WHERE status = 'closed';"
            trades = await self.db.pool.fetch(query)

            if not trades:
                return web.json_response({'success': True, 'data': {
                    'strategy_performance': [],
                    'hourly_profitability': [],
                }})

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], utc=True)

            # --- FIX: Use strategy column from DB, fallback to metadata if empty ---
            def get_strategy(row):
                # First try the direct strategy column from the database
                if 'strategy' in row and row['strategy'] and row['strategy'] != 'unknown':
                    return row['strategy']
                # Fallback to extracting from metadata
                return self._get_strategy_from_metadata(row.get('metadata'))

            df['strategy'] = df.apply(get_strategy, axis=1)
            strategy_performance = df.groupby('strategy')['profit_loss'].sum().reset_index()
            strategy_performance.columns = ['strategy', 'total_pnl']

            # Profitability by hour
            df['hour'] = df['exit_timestamp'].dt.hour
            hourly_profitability = df.groupby('hour')['profit_loss'].mean().reset_index()
            hourly_profitability.columns = ['hour', 'avg_pnl']

            return web.json_response({
                'success': True,
                'data': {
                    'strategy_performance': strategy_performance.to_dict('records'),
                    'hourly_profitability': hourly_profitability.to_dict('records'),
                }
            })
        except Exception as e:
            logger.error(f"Error in api_get_analysis: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def api_get_insights(self, request):
        """Generate comprehensive performance insights and actionable recommendations"""
        try:
            if not self.db:
                return web.json_response({'error': 'Database connection not available.'}, status=503)

            query = "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_timestamp DESC;"
            trades = await self.db.pool.fetch(query)

            insights = []
            if not trades:
                insights.append({
                    'type': 'info',
                    'title': 'Getting Started',
                    'message': 'No completed trades yet. Start trading to receive personalized performance insights and recommendations.'
                })
                return web.json_response({'success': True, 'data': insights})

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # Calculate key metrics
            total_trades = len(df)
            winning_trades = df[df['profit_loss'] > 0]
            losing_trades = df[df['profit_loss'] <= 0]
            win_rate = (df['profit_loss'] > 0).mean() * 100
            total_pnl = df['profit_loss'].sum()

            avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['profit_loss'].mean()) if len(losing_trades) > 0 else 0

            # Insight 1: Overall Performance Summary
            if total_pnl > 0:
                insights.append({
                    'type': 'success',
                    'title': '🎉 Profitable Trading',
                    'message': f'Great job! Your total P&L is ${total_pnl:.2f} across {total_trades} trades. Keep maintaining your winning edge.'
                })
            elif total_pnl < 0:
                insights.append({
                    'type': 'warning',
                    'title': '⚠️ Negative Performance',
                    'message': f'Your account is down ${abs(total_pnl):.2f}. Review your strategy and consider reducing position sizes until performance improves.'
                })

            # Insight 2: Win Rate Analysis with Specific Actions
            if win_rate < 40:
                insights.append({
                    'type': 'critical',
                    'title': '🔴 Low Win Rate Alert',
                    'message': f'Win rate at {win_rate:.1f}% is critically low. Action: Pause trading and backtest your strategy. Consider stricter entry filters and better technical indicators.'
                })
            elif win_rate < 50:
                insights.append({
                    'type': 'suggestion',
                    'title': '💡 Improve Win Rate',
                    'message': f'Win rate: {win_rate:.1f}%. To improve: (1) Wait for stronger confirmation signals, (2) Avoid trading in choppy markets, (3) Use tighter stop losses.'
                })
            elif win_rate > 60:
                insights.append({
                    'type': 'success',
                    'title': '✅ Excellent Win Rate',
                    'message': f'Win rate: {win_rate:.1f}%. Outstanding! Consider gradually increasing position sizes to maximize profits while maintaining discipline.'
                })

            # Insight 3: Risk/Reward Ratio with Actionable Steps
            if avg_loss > 0:
                risk_reward = avg_win / avg_loss
                if risk_reward < 1.0:
                    insights.append({
                        'type': 'critical',
                        'title': '🔴 Poor Risk/Reward',
                        'message': f'R:R ratio {risk_reward:.2f}:1 is unsustainable. Action: Set take-profit at 2x your stop-loss distance. Let winners run longer.'
                    })
                elif risk_reward < 1.5:
                    insights.append({
                        'type': 'suggestion',
                        'title': '📊 Optimize Risk/Reward',
                        'message': f'R:R ratio {risk_reward:.2f}:1. Target minimum 2:1. Tip: Move stop-loss to breakeven after 1:1 gain, and let profits run to 2-3x targets.'
                    })
                elif risk_reward > 2.0:
                    insights.append({
                        'type': 'success',
                        'title': '⭐ Strong Risk/Reward',
                        'message': f'R:R ratio {risk_reward:.2f}:1. Excellent risk management! Maintain this discipline.'
                    })

            # Insight 4: Profit Factor
            total_wins = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            if profit_factor > 0:
                if profit_factor < 1.0:
                    insights.append({
                        'type': 'warning',
                        'title': '⚠️ Negative Profit Factor',
                        'message': f'Profit factor {profit_factor:.2f} means you lose more than you win. Reduce trade frequency and be more selective with entries.'
                    })
                elif profit_factor > 2.0:
                    insights.append({
                        'type': 'success',
                        'title': '🏆 Excellent Profit Factor',
                        'message': f'Profit factor {profit_factor:.2f}. You\'re making ${profit_factor:.1f} for every $1 lost. Keep it up!'
                    })

            # Insight 5: Consecutive Loss Streak Detection
            df['is_loss'] = df['profit_loss'] <= 0
            current_streak = 0
            max_streak = 0
            temp_streak = 0

            for is_loss in df['is_loss'].values:
                if is_loss:
                    temp_streak += 1
                    max_streak = max(max_streak, temp_streak)
                else:
                    temp_streak = 0

            # Check current streak
            for is_loss in df.head(10)['is_loss'].values:
                if is_loss:
                    current_streak += 1
                else:
                    break

            if current_streak >= 3:
                insights.append({
                    'type': 'critical',
                    'title': '🚨 Loss Streak Alert',
                    'message': f'You have {current_streak} consecutive losses. STOP TRADING NOW. Take a break, review your strategy, and reduce position size by 50% when you return.'
                })
            elif max_streak >= 5:
                insights.append({
                    'type': 'warning',
                    'title': '⚠️ Streak Risk',
                    'message': f'Your longest loss streak was {max_streak} trades. Implement a rule: After 3 consecutive losses, reduce position size by 50% until you get 2 wins.'
                })

            # Insight 6: Strategy Performance Analysis
            df['strategy'] = df['metadata'].apply(self._get_strategy_from_metadata)
            strategy_stats = df.groupby('strategy').agg({
                'profit_loss': ['sum', 'count', lambda x: (x > 0).mean() * 100]
            }).round(2)
            strategy_stats.columns = ['pnl', 'trades', 'win_rate']

            best_strategy = strategy_stats.nlargest(1, 'pnl')
            worst_strategy = strategy_stats.nsmallest(1, 'pnl')

            if not best_strategy.empty and best_strategy['pnl'].values[0] > 0:
                strat_name = best_strategy.index[0]
                strat_pnl = best_strategy['pnl'].values[0]
                strat_wr = best_strategy['win_rate'].values[0]
                insights.append({
                    'type': 'success',
                    'title': f'⭐ Best Strategy: {strat_name}',
                    'message': f'${strat_pnl:.2f} profit with {strat_wr:.1f}% win rate. Focus more capital on this strategy and analyze what makes it successful.'
                })

            if not worst_strategy.empty and worst_strategy['pnl'].values[0] < 0:
                strat_name = worst_strategy.index[0]
                strat_pnl = worst_strategy['pnl'].values[0]
                strat_wr = worst_strategy['win_rate'].values[0]
                insights.append({
                    'type': 'warning',
                    'title': f'❌ Underperforming: {strat_name}',
                    'message': f'${strat_pnl:.2f} loss with {strat_wr:.1f}% win rate. Disable this strategy or reduce allocation to 10% of normal size for testing.'
                })

            # Insight 7: Chain/Network Performance
            if 'chain' in df.columns:
                chain_pnl = df.groupby('chain')['profit_loss'].agg(['sum', 'count']).round(2)
                chain_pnl.columns = ['pnl', 'trades']

                best_chain = chain_pnl.nlargest(1, 'pnl')
                if not best_chain.empty and best_chain['pnl'].values[0] > 0:
                    chain_name = best_chain.index[0].upper()
                    chain_profit = best_chain['pnl'].values[0]
                    insights.append({
                        'type': 'info',
                        'title': f'🔗 Best Network: {chain_name}',
                        'message': f'${chain_profit:.2f} profit on {chain_name}. Consider allocating more trading capital to this network.'
                    })

            # Insight 8: Trade Frequency & Overtrading
            recent_24h = df[df['exit_timestamp'] > (pd.Timestamp.utcnow() - pd.Timedelta(days=1))]
            if len(recent_24h) > 20:
                insights.append({
                    'type': 'warning',
                    'title': '⚠️ Overtrading Detected',
                    'message': f'{len(recent_24h)} trades in 24h. Quality > Quantity. Reduce trade frequency and wait for higher-probability setups.'
                })

            # Insight 9: Average Hold Time
            df['hold_time'] = (df['exit_timestamp'] - df['entry_timestamp']).dt.total_seconds() / 60  # minutes
            avg_hold_time = df['hold_time'].mean()

            if avg_hold_time < 5:
                insights.append({
                    'type': 'suggestion',
                    'title': '⏱️ Very Short Holds',
                    'message': f'Average hold time: {avg_hold_time:.1f} minutes. You might be exiting too quickly. Give trades more time to develop (aim for 15-30 min).'
                })

            # Insight 10: Recent Performance Trend
            recent_10 = df.head(10)['profit_loss'].sum()
            if recent_10 < 0 and total_pnl > 0:
                insights.append({
                    'type': 'warning',
                    'title': '📉 Recent Downturn',
                    'message': f'Last 10 trades: ${recent_10:.2f}. Your edge may be deteriorating. Review recent trades and consider taking a break.'
                })
            elif recent_10 > 0 and len(df) > 10:
                older_pnl = df.iloc[10:]['profit_loss'].sum()
                if recent_10 > older_pnl * 0.5:  # Recent performance much better
                    insights.append({
                        'type': 'success',
                        'title': '📈 Improving Performance',
                        'message': 'Your recent trades are performing better! Whatever changes you made are working. Keep it up!'
                    })

            # Insight 11: Position Sizing Recommendation
            if avg_loss > 0:
                max_recommended_loss = 2.0  # $2 max loss per trade
                if avg_loss > max_recommended_loss:
                    reduction = ((avg_loss - max_recommended_loss) / avg_loss) * 100
                    insights.append({
                        'type': 'suggestion',
                        'title': '💰 Reduce Position Size',
                        'message': f'Average loss: ${avg_loss:.2f}. Reduce position size by {reduction:.0f}% to limit losses to $2 per trade maximum.'
                    })

            # Insight 12: Best Time to Trade (if enough data)
            if len(df) > 50:
                df['hour'] = df['entry_timestamp'].dt.hour
                hourly_pnl = df.groupby('hour')['profit_loss'].sum().sort_values(ascending=False)
                best_hours = hourly_pnl.head(3).index.tolist()
                worst_hours = hourly_pnl.tail(3).index.tolist()

                insights.append({
                    'type': 'info',
                    'title': '🕐 Optimal Trading Hours',
                    'message': f'Most profitable hours: {", ".join(map(str, best_hours))}:00 UTC. Avoid hours: {", ".join(map(str, worst_hours))}:00 UTC.'
                })

            # Always add at least one positive insight if performance is decent
            if not any(i['type'] == 'success' for i in insights) and win_rate >= 45:
                insights.append({
                    'type': 'success',
                    'title': '👍 Solid Foundation',
                    'message': 'You have a solid trading foundation. Focus on consistency, proper risk management, and continuous improvement.'
                })

            return web.json_response({'success': True, 'data': self._serialize_decimals(insights)})

        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def api_dashboard_summary(self, request):
        """Get dashboard summary data from database including all modules"""
        try:
            # Get initial balance from config
            try:
                from config.config_manager import PortfolioConfig
                config = PortfolioConfig()
                starting_balance = float(config.initial_balance or 400)
            except Exception:
                starting_balance = 400.0

            # Initialize with defaults
            total_pnl = 0.0
            win_rate = 0.0
            open_positions_count = 0
            total_trades = 0
            winning_trades_count = 0

            # Get DEX data from database
            if self.db:
                try:
                    # Get all trades for P&L calculation
                    trades = await self.db.get_recent_trades(limit=1000)
                    closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('profit_loss') is not None]

                    total_pnl = sum(float(t.get('profit_loss', 0)) for t in closed_trades)
                    total_trades = len(closed_trades)

                    # Calculate win rate
                    winning_trades_count = sum(1 for t in closed_trades if float(t.get('profit_loss', 0)) > 0)

                except Exception as e:
                    logger.warning(f"Error getting data from database: {e}")

            # Get DEX open positions from ENGINE
            if self.engine and hasattr(self.engine, 'active_positions') and self.engine.active_positions:
                open_positions_count = len(self.engine.active_positions)

            # Get Futures module data
            futures_pnl = 0.0
            futures_trades = 0
            futures_positions = 0
            futures_winning = 0
            try:
                futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:{futures_port}/stats', timeout=5) as resp:
                        if resp.status == 200:
                            stats_data = await resp.json()
                            stats = stats_data.get('stats', stats_data)
                            futures_trades = stats.get('total_trades', 0)
                            futures_positions = stats.get('active_positions', 0)
                            futures_winning = stats.get('winning_trades', 0)
                            # Parse PnL which may be a string like "$-0.76"
                            net_pnl = stats.get('net_pnl', '$0.00')
                            if isinstance(net_pnl, str):
                                net_pnl = float(net_pnl.replace('$', '').replace(',', ''))
                            futures_pnl = net_pnl
                            logger.debug(f"Futures summary: trades={futures_trades}, pnl={futures_pnl}, positions={futures_positions}")
            except Exception as e:
                logger.debug(f"Could not fetch futures stats for summary: {e}")

            # Combine totals
            total_pnl += futures_pnl
            total_trades += futures_trades
            open_positions_count += futures_positions
            winning_trades_count += futures_winning

            # Calculate combined win rate
            win_rate = (winning_trades_count / total_trades * 100) if total_trades > 0 else 0

            # Calculate portfolio value
            portfolio_value = starting_balance + total_pnl

            summary = {
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'total_value': portfolio_value,
                'net_profit': total_pnl,
                'open_positions': open_positions_count,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'starting_balance': starting_balance,
                'pending_orders': 0,
                'active_alerts': 0
            }

            return web.json_response({
                'success': True,
                'data': summary
            })
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_recent_trades(self, request):
        """Get recent trades with token symbol and network"""
        try:
            limit = int(request.query.get('limit', 50))
            
            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)
            
            trades = await self.db.get_recent_trades(limit=limit)
            
            # Enrich trades with token symbols from metadata
            enriched_trades = []
            for trade in trades:
                enriched_trade = dict(trade)
                
                # Extract token_symbol from metadata JSON if present
                token_symbol = trade.get('token_symbol', 'UNKNOWN')
                
                if token_symbol == 'UNKNOWN' and trade.get('metadata'):
                    try:
                        import json
                        metadata = trade.get('metadata')
                        
                        # If metadata is string, parse it
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        
                        # Extract token_symbol from metadata
                        token_symbol = metadata.get('token_symbol', 'UNKNOWN')
                    except:
                        pass
                
                enriched_trade['token_symbol'] = token_symbol
                enriched_trade['network'] = trade.get('chain', 'unknown')
                enriched_trades.append(enriched_trade)
            
            return web.json_response({
                'success': True,
                'data': self._serialize_decimals(enriched_trades),
                'count': len(enriched_trades)
            })
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_trade_history(self, request):
        """Get trade history with filters"""
        try:
            # Parse query parameters
            start_date = request.query.get('start_date')
            end_date = request.query.get('end_date')
            status = request.query.get('status')

            # ✅ FIX: Add await
            trades = await self.db.get_recent_trades(limit=1000, status=status)

            # Apply date filters if provided
            if start_date:
                start = datetime.fromisoformat(start_date)
                trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= start]

            if end_date:
                end = datetime.fromisoformat(end_date)
                trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) <= end]

            return web.json_response({
                'success': True,
                'data': trades,
                'count': len(trades)
            })
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_export_trades(self, request):
        """Export all trades in CSV or Excel format with comprehensive data"""
        try:
            format_type = request.match_info['format']

            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)

            # Get all trades from database with full details
            async with self.db.pool.acquire() as conn:
                trades_records = await conn.fetch("""
                    SELECT
                        id,
                        trade_id,
                        token_address,
                        chain,
                        strategy,
                        side,
                        amount,
                        entry_price,
                        exit_price,
                        entry_timestamp,
                        exit_timestamp,
                        status,
                        profit_loss,
                        profit_loss_percentage,
                        gas_fee,
                        slippage,
                        usd_value,
                        risk_score,
                        ml_confidence,
                        metadata
                    FROM trades
                    ORDER BY entry_timestamp DESC
                """)

            # Convert to dict and enrich with metadata
            trades = []
            for record in trades_records:
                trade_dict = dict(record)

                # Extract metadata
                metadata = trade_dict.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}

                # Extract all fields with multiple fallbacks
                token_symbol = (
                    metadata.get('token_symbol') or
                    metadata.get('token') or
                    trade_dict.get('token_symbol') or
                    'Unknown'
                )

                # Extract exit/close reason with multiple fallbacks
                exit_reason_raw = (
                    metadata.get('close_reason') or  # Primary: close_reason from engine
                    metadata.get('exit_reason') or   # Fallback: exit_reason
                    'N/A'
                )

                # Map exit reasons to human-readable format
                exit_reason_map = {
                    'take_profit': 'Take Profit Hit',
                    'stop_loss': 'Stop Loss Hit',
                    'trailing_stop': 'Trailing Stop Loss',
                    'time_limit': 'Max Hold Time Reached',
                    'high_volatility': 'High Volatility',
                    'Manual close via dashboard': 'Manual Close (Dashboard)',
                    'manual_close': 'Manual Close',
                    'manual': 'Manual Close',
                }
                exit_reason = exit_reason_map.get(exit_reason_raw, exit_reason_raw)

                # Use ROI from database or calculate if not available
                roi = float(trade_dict.get('profit_loss_percentage', 0) or 0)
                if roi == 0 and trade_dict.get('profit_loss'):
                    entry_value = float(trade_dict.get('entry_price', 0) or 0) * float(trade_dict.get('amount', 0) or 0)
                    profit_loss = float(trade_dict.get('profit_loss', 0) or 0)
                    roi = (profit_loss / entry_value * 100) if entry_value > 0 else 0

                # Calculate hold time
                hold_time = 'N/A'
                if trade_dict.get('exit_timestamp') and trade_dict.get('entry_timestamp'):
                    try:
                        entry_ts = trade_dict['entry_timestamp']
                        exit_ts = trade_dict['exit_timestamp']
                        if isinstance(entry_ts, str):
                            entry_ts = datetime.fromisoformat(entry_ts)
                        if isinstance(exit_ts, str):
                            exit_ts = datetime.fromisoformat(exit_ts)
                        delta = exit_ts - entry_ts
                        hours = delta.total_seconds() / 3600
                        hold_time = f"{hours:.2f}h"
                    except:
                        pass

                # Enrich trade dict with additional fields
                trade_dict.update({
                    'token_symbol': token_symbol,
                    'exit_reason': exit_reason,
                    'roi': round(roi, 4),
                    'hold_time': hold_time,
                    'stop_loss': metadata.get('stop_loss') or metadata.get('stop_loss_price'),
                    'take_profit': metadata.get('take_profit') or metadata.get('take_profit_price'),
                    'gas_cost': trade_dict.get('gas_fee'),  # Map gas_fee to gas_cost for backwards compatibility
                    'tx_hash': metadata.get('tx_hash', metadata.get('transaction_hash')),
                })

                trades.append(trade_dict)

            # Export based on format
            if format_type == 'csv':
                return await self._export_trades_csv(trades)
            elif format_type == 'excel':
                return await self._export_trades_excel(trades)
            else:
                return web.json_response({'error': f'Unsupported format: {format_type}'}, status=400)

        except Exception as e:
            logger.error(f"Error exporting trades: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def _export_trades_csv(self, trades):
        """Export trades to CSV format"""
        import csv
        from decimal import Decimal

        output = io.StringIO()

        # Define comprehensive columns
        columns = [
            'ID', 'Token Symbol', 'Token Address', 'Chain', 'Strategy',
            'Side', 'Entry Timestamp', 'Exit Timestamp', 'Hold Time',
            'Entry Price', 'Exit Price', 'Amount', 'Entry Value', 'Exit Value',
            'Profit/Loss', 'ROI (%)', 'Status', 'Exit Reason',
            'Stop Loss', 'Take Profit', 'Gas Cost', 'Slippage', 'TX Hash'
        ]

        writer = csv.writer(output)
        writer.writerow(columns)

        # Helper function to safely convert values
        def safe_float(val, default=0):
            """Safely convert value to float, handling Decimal, None, etc."""
            if val is None or val == '':
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, Decimal):
                return float(val)
            try:
                return float(val)
            except:
                return default

        def safe_str(val, default=''):
            """Safely convert value to string"""
            if val is None:
                return default
            return str(val)

        for trade in trades:
            try:
                entry_price = safe_float(trade.get('entry_price'))
                exit_price = safe_float(trade.get('exit_price'))
                amount = safe_float(trade.get('amount'))

                entry_value = entry_price * amount
                exit_value = exit_price * amount if exit_price > 0 else 0

                writer.writerow([
                    trade.get('id', ''),
                    safe_str(trade.get('token_symbol', 'Unknown')),
                    safe_str(trade.get('token_address')),
                    safe_str(trade.get('chain')),
                    safe_str(trade.get('strategy')),
                    safe_str(trade.get('side')),
                    safe_str(trade.get('entry_timestamp')),
                    safe_str(trade.get('exit_timestamp')),
                    safe_str(trade.get('hold_time', 'N/A')),
                    round(entry_price, 8) if entry_price else '',
                    round(exit_price, 8) if exit_price else '',
                    round(amount, 8) if amount else '',
                    round(entry_value, 8) if entry_value else '',
                    round(exit_value, 8) if exit_value else '',
                    round(safe_float(trade.get('profit_loss')), 8),
                    round(safe_float(trade.get('roi')), 4),
                    safe_str(trade.get('status')),
                    safe_str(trade.get('exit_reason', 'N/A')),
                    round(safe_float(trade.get('stop_loss')), 8) if trade.get('stop_loss') else '',
                    round(safe_float(trade.get('take_profit')), 8) if trade.get('take_profit') else '',
                    round(safe_float(trade.get('gas_cost')), 8) if trade.get('gas_cost') else '',
                    round(safe_float(trade.get('slippage')), 4) if trade.get('slippage') else '',
                    safe_str(trade.get('tx_hash')),
                ])
            except Exception as e:
                logger.error(f"Error writing CSV row for trade {trade.get('id')}: {e}", exc_info=True)
                continue

        csv_content = output.getvalue()
        output.close()

        response = web.Response(
            body=csv_content.encode('utf-8'),
            content_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="trades_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'}
        )
        return response

    async def _export_trades_excel(self, trades):
        """Export trades to Excel format with professional formatting"""
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.cell import MergedCell
        from decimal import Decimal

        output = io.BytesIO()
        wb = Workbook()
        ws = wb.active
        ws.title = 'Trades Export'

        # Header styling
        header_fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF', size=11)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Define columns
        columns = [
            'ID', 'Token Symbol', 'Token Address', 'Chain', 'Strategy',
            'Side', 'Entry Timestamp', 'Exit Timestamp', 'Hold Time',
            'Entry Price', 'Exit Price', 'Amount', 'Entry Value', 'Exit Value',
            'Profit/Loss', 'ROI (%)', 'Status', 'Exit Reason',
            'Stop Loss', 'Take Profit', 'Gas Cost', 'Slippage', 'TX Hash'
        ]

        # Write header
        for col_num, column_name in enumerate(columns, 1):
            cell = ws.cell(row=1, column=col_num, value=column_name)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border

        # Helper function to safely convert values
        def safe_float(val, default=0):
            """Safely convert value to float, handling Decimal, None, etc."""
            if val is None or val == '':
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, Decimal):
                return float(val)
            try:
                return float(val)
            except:
                return default

        def safe_str(val, default=''):
            """Safely convert value to string"""
            if val is None:
                return default
            return str(val)

        # Write data
        for row_num, trade in enumerate(trades, 2):
            try:
                entry_price = safe_float(trade.get('entry_price'))
                exit_price = safe_float(trade.get('exit_price'))
                amount = safe_float(trade.get('amount'))

                entry_value = entry_price * amount
                exit_value = exit_price * amount if exit_price > 0 else 0

                row_data = [
                    int(trade.get('id', 0)) if trade.get('id') else '',
                    safe_str(trade.get('token_symbol', 'Unknown')),
                    safe_str(trade.get('token_address')),
                    safe_str(trade.get('chain')),
                    safe_str(trade.get('strategy')),
                    safe_str(trade.get('side')),
                    safe_str(trade.get('entry_timestamp')),
                    safe_str(trade.get('exit_timestamp')),
                    safe_str(trade.get('hold_time', 'N/A')),
                    round(entry_price, 8) if entry_price else '',
                    round(exit_price, 8) if exit_price else '',
                    round(amount, 8) if amount else '',
                    round(entry_value, 8) if entry_value else '',
                    round(exit_value, 8) if exit_value else '',
                    round(safe_float(trade.get('profit_loss')), 8),
                    round(safe_float(trade.get('roi')), 4),
                    safe_str(trade.get('status')),
                    safe_str(trade.get('exit_reason', 'N/A')),
                    round(safe_float(trade.get('stop_loss')), 8) if trade.get('stop_loss') else '',
                    round(safe_float(trade.get('take_profit')), 8) if trade.get('take_profit') else '',
                    round(safe_float(trade.get('gas_cost')), 8) if trade.get('gas_cost') else '',
                    round(safe_float(trade.get('slippage')), 4) if trade.get('slippage') else '',
                    safe_str(trade.get('tx_hash')),
                ]

                for col_num, value in enumerate(row_data, 1):
                    cell = ws.cell(row=row_num, column=col_num, value=value)
                    cell.border = border

                    # Color-code P&L and ROI
                    if columns[col_num-1] in ['Profit/Loss', 'ROI (%)']:
                        try:
                            val = float(value) if value and value != '' else 0
                            if val > 0:
                                cell.font = Font(color='00B050', bold=True)
                                cell.fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
                            elif val < 0:
                                cell.font = Font(color='FF0000', bold=True)
                                cell.fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
                        except:
                            pass
            except Exception as e:
                logger.error(f"Error writing row {row_num}: {e}", exc_info=True)
                continue

        # Auto-adjust column widths
        for col in ws.iter_cols():
            if col and not isinstance(col[0], MergedCell):
                try:
                    max_length = max(len(str(cell.value or '')) for cell in col)
                    ws.column_dimensions[col[0].column_letter].width = min(max_length + 2, 50)
                except:
                    pass

        # Save workbook
        try:
            wb.save(output)
            output.seek(0)
        except Exception as e:
            logger.error(f"Error saving Excel workbook: {e}", exc_info=True)
            raise

        response = web.Response(
            body=output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename="trades_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx"'}
        )
        return response

    async def api_open_positions(self, request):
        """Get open positions with ALL required fields"""
        try:
            positions = []
            
            if self.engine and hasattr(self.engine, 'active_positions'):
                # ✅ Create a snapshot copy to avoid "dictionary changed size" error
                active_positions_snapshot = dict(self.engine.active_positions.items())
                
                for token_address, position in active_positions_snapshot.items():
                    # Extract and calculate all required fields
                    entry_price = float(position.get('entry_price', 0))
                    current_price = float(position.get('current_price', entry_price))
                    amount = float(position.get('amount', 0))
                    
                    # Calculate value
                    value = amount * current_price
                    
                    # Calculate P&L
                    entry_value = amount * entry_price
                    unrealized_pnl = value - entry_value
                    
                    # Calculate ROI
                    roi = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    # ✅ Always enrich from database for definitive SL/TP values
                    entry_timestamp = position.get('entry_timestamp') or position.get('opened_at') or position.get('timestamp')
                    stop_loss = None
                    take_profit = None

                    if self.db:
                        try:
                            async with self.db.pool.acquire() as conn:
                                trade = await conn.fetchrow("""
                                    SELECT entry_timestamp, metadata
                                    FROM trades
                                    WHERE token_address = $1 
                                    AND status = 'open'
                                    ORDER BY entry_timestamp DESC
                                    LIMIT 1
                                """, token_address)
                                
                                if trade:
                                    if not entry_timestamp and trade['entry_timestamp']:
                                        entry_timestamp = trade['entry_timestamp']
                                    
                                    if trade['metadata']:
                                        metadata = trade['metadata']
                                        if isinstance(metadata, str):
                                            metadata = json.loads(metadata)

                                        # Prefer database metadata values
                                        stop_loss = metadata.get('stop_loss')
                                        take_profit = metadata.get('take_profit')
                        except (json.JSONDecodeError, AttributeError, Exception) as e:
                            logger.error(f"Error enriching position from DB: {e}")

                    # --- FIX STARTS HERE ---
                    # Fallback logic to calculate SL/TP if they are not in the database metadata
                    if stop_loss is None:
                        sl_pct = position.get('stop_loss_percentage')
                        if sl_pct and entry_price > 0:
                            stop_loss = entry_price * (1 - sl_pct)
                        # Final fallback if even percentage is missing
                        else:
                            stop_loss = entry_price * 0.88 # Default to 12% SL

                    if take_profit is None:
                        tp_pct = position.get('take_profit_percentage')
                        if tp_pct and entry_price > 0:
                            take_profit = entry_price * (1 + tp_pct)
                        # Final fallback
                        else:
                            take_profit = entry_price * 1.24 # Default to 24% TP
                    # --- FIX ENDS HERE ---
                    
                    # Calculate duration
                    duration_str = 'unknown'
                    
                    if entry_timestamp:
                        try:
                            if isinstance(entry_timestamp, str):
                                entry_time = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
                            elif isinstance(entry_timestamp, datetime):
                                entry_time = entry_timestamp
                            else:
                                entry_time = datetime.fromtimestamp(float(entry_timestamp))
                            
                            # Make sure entry_time is timezone-naive for comparison
                            if entry_time.tzinfo:
                                entry_time = entry_time.replace(tzinfo=None)
                            
                            duration_delta = datetime.utcnow() - entry_time
                            total_seconds = duration_delta.total_seconds()
                            
                            if total_seconds < 60:
                                duration_str = 'just now'
                            elif total_seconds < 3600:
                                minutes = int(total_seconds // 60)
                                duration_str = f"{minutes}m ago"
                            elif total_seconds < 86400:
                                hours = int(total_seconds // 3600)
                                minutes = int((total_seconds % 3600) // 60)
                                duration_str = f"{hours}h {minutes}m ago"
                            else:
                                days = int(total_seconds // 86400)
                                hours = int((total_seconds % 86400) // 3600)
                                duration_str = f"{days}d {hours}h ago"
                        except Exception as e:
                            logger.error(f"Error calculating duration: {e}")
                            duration_str = 'unknown'
                    
                    positions.append({
                        'id': position.get('id', str(token_address)),
                        'token_address': token_address,
                        'token_symbol': position.get('token_symbol', position.get('symbol', 'UNKNOWN')),
                        'entry_price': round(entry_price, 8),
                        'current_price': round(current_price, 8),
                        'amount': round(amount, 4),
                        'value': round(value, 2),
                        'unrealized_pnl': round(unrealized_pnl, 2),
                        'roi': round(roi, 2),
                        'stop_loss': stop_loss,  # ✅ Now enriched from DB
                        'take_profit': take_profit,  # ✅ Now enriched from DB
                        'entry_timestamp': entry_timestamp.isoformat() if isinstance(entry_timestamp, datetime) else entry_timestamp,  # ✅ Now enriched from DB
                        'duration': duration_str,
                        'status': position.get('status', 'open'),
                        'chain': position.get('chain', 'unknown'),
                        'network': position.get('chain', 'unknown')
                    })
            
            logger.info(f"Returning {len(positions)} open positions")
            
            return web.json_response({
                'success': True,
                'data': positions,
                'count': len(positions)
            })
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'data': [],
                'count': 0
            }, status=200)
    
    async def api_positions_history(self, request):
        """Get closed positions history"""
        try:
            limit = int(request.query.get('limit', 100))
            closed_positions = []
            
            if self.db:
                trades = await self.db.get_closed_trades(limit=limit)
                
                for trade in trades:
                    # Calculate duration
                    duration = 'unknown'
                    if trade.get('entry_timestamp') and trade.get('exit_timestamp'):
                        try:
                            entry = datetime.fromisoformat(str(trade['entry_timestamp']).replace('Z', '+00:00'))
                            exit_time = datetime.fromisoformat(str(trade['exit_timestamp']).replace('Z', '+00:00'))
                            delta = exit_time - entry
                            
                            hours = int(delta.total_seconds() // 3600)
                            minutes = int((delta.total_seconds() % 3600) // 60)
                            
                            if hours > 24:
                                days = hours // 24
                                remaining_hours = hours % 24
                                duration = f"{days}d {remaining_hours}h"
                            elif hours > 0:
                                duration = f"{hours}h {minutes}m"
                            else:
                                duration = f"{minutes}m"
                        except:
                            pass
                    
                    # ✅ Extract token_symbol from metadata
                    token_symbol = trade.get('token_symbol', 'UNKNOWN')
                    
                    if token_symbol == 'UNKNOWN' and trade.get('metadata'):
                        try:
                            import json
                            metadata = trade.get('metadata')
                            
                            if isinstance(metadata, str):
                                metadata = json.loads(metadata)
                            
                            token_symbol = metadata.get('token_symbol', 'UNKNOWN')
                        except:
                            pass
                    
                    # Convert timestamps to ISO string
                    entry_ts = trade.get('entry_timestamp')
                    exit_ts = trade.get('exit_timestamp')
                    
                    closed_positions.append({
                        'id': trade.get('id'),
                        'token_symbol': token_symbol,  # ✅ Use extracted symbol
                        'token_address': trade.get('token_address'),
                        'entry_price': float(trade.get('entry_price', 0)),
                        'exit_price': float(trade.get('exit_price', 0)),
                        'amount': float(trade.get('amount', 0)),
                        'profit_loss': float(trade.get('profit_loss', 0)),
                        'roi': float(trade.get('roi', 0)) if trade.get('roi') else (float(trade.get('profit_loss_percentage', 0)) if trade.get('profit_loss_percentage') else 0),
                        'entry_timestamp': entry_ts.isoformat() if entry_ts else None,
                        'exit_timestamp': exit_ts.isoformat() if exit_ts else None,
                        'duration': duration,
                        'exit_reason': trade.get('exit_reason', 'manual')
                    })
            
            return web.json_response({
                'success': True,
                'data': self._serialize_decimals(closed_positions),
                'count': len(closed_positions)
            })
        except Exception as e:
            logger.error(f"Error getting closed positions: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'data': [],
                'count': 0
            }, status=200)

    async def api_risk_metrics(self, request):
        """Calculate risk metrics from trade history"""
        try:
            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)
            
            trades = await self.db.get_recent_trades(limit=1000)
            closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('profit_loss') is not None]
            
            if len(closed_trades) < 2:
                return web.json_response({
                    'success': True,
                    'data': {
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'var_95': 0.0,
                        'portfolio_beta': 0.0
                    }
                })
            
            returns = [float(t.get('profit_loss', 0)) for t in closed_trades]
            
            import statistics
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            sharpe_ratio = (mean_return / std_return * (252 ** 0.5)) if std_return != 0 else 0
            
            cumulative = []
            cum_sum = 0
            for ret in returns:
                cum_sum += ret
                cumulative.append(cum_sum)
            
            max_drawdown = 0
            peak = cumulative[0]
            for value in cumulative:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / abs(peak) if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            losses = [r for r in returns if r < 0]
            var_95 = abs(statistics.quantiles(losses, n=20)[0]) if len(losses) > 10 else 0
            
            return web.json_response({
                'success': True,
                'data': {
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'max_drawdown': round(max_drawdown * 100, 2),
                    'var_95': round(var_95, 2),
                    'portfolio_beta': 0.0
                }
            })
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return web.json_response({
                'success': True,
                'data': {
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'var_95': 0.0,
                    'portfolio_beta': 0.0
                }
            })

    # ============================================================================
    # WALLET BALANCES API - Shows portfolio values by chain
    # ============================================================================

    async def api_wallet_balances(self, request):
        """Get wallet balances by chain - shows portfolio values and real balances

        Uses caching to prevent instability from intermittent RPC failures.
        Blockchain balances are cached for 60 seconds.
        """
        try:
            balances = {}
            total_value = 0
            total_pnl = 0
            now = datetime.now()

            # Initialize chains with zero values
            chains = ['ETHEREUM', 'BSC', 'POLYGON', 'ARBITRUM', 'BASE', 'SOLANA']
            for chain in chains:
                balances[chain] = {
                    'balance': 0.0,
                    'native_balance': 0.0,
                    'native_symbol': 'ETH' if chain in ['ETHEREUM', 'ARBITRUM', 'BASE'] else ('BNB' if chain == 'BSC' else ('MATIC' if chain == 'POLYGON' else 'SOL')),
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'positions': 0
                }

            # ========== GET PORTFOLIO VALUES FROM ENGINE (same source as Open Positions API) ==========
            if self.engine and hasattr(self.engine, 'active_positions') and self.engine.active_positions:
                try:
                    active_positions_snapshot = dict(self.engine.active_positions.items())

                    for token_address, pos in active_positions_snapshot.items():
                        chain = (pos.get('chain') or pos.get('network') or 'SOLANA').upper()

                        if chain in balances:
                            entry_price = float(pos.get('entry_price', 0))
                            current_price = float(pos.get('current_price', entry_price))
                            amount = float(pos.get('amount', 0))

                            entry_value = amount * entry_price
                            current_value = amount * current_price
                            unrealized_pnl = current_value - entry_value

                            balances[chain]['balance'] += current_value
                            balances[chain]['pnl'] += unrealized_pnl
                            balances[chain]['positions'] += 1
                            total_value += current_value
                            total_pnl += unrealized_pnl

                except Exception as e:
                    logger.warning(f"Error getting positions from engine: {e}")

            # Also get realized P&L from closed trades
            if self.db:
                try:
                    trades = await self.db.get_recent_trades(limit=500)
                    if trades:
                        for trade in trades:
                            if trade.get('status') == 'closed':
                                chain = (trade.get('chain') or trade.get('network') or 'unknown').upper()
                                realized_pnl = float(trade.get('profit_loss') or 0)
                                if chain in balances:
                                    balances[chain]['pnl'] += realized_pnl
                except Exception as e:
                    logger.warning(f"Error getting closed trades: {e}")

            # ========== CHECK CACHE - Only fetch blockchain balances every 60 seconds ==========
            cache_valid = (
                self._wallet_cache_time and
                (now - self._wallet_cache_time).total_seconds() < self._wallet_cache_ttl and
                self._wallet_cache
            )

            if cache_valid:
                # Use cached blockchain balances
                cached = self._wallet_cache
                prices = self._price_cache

                # Apply cached native balances to chains without positions
                for chain in chains:
                    if balances[chain]['positions'] == 0:
                        cached_chain = cached.get(chain, {})
                        if cached_chain.get('native_balance', 0) > 0:
                            balances[chain]['native_balance'] = cached_chain['native_balance']
                            balances[chain]['balance'] = cached_chain.get('balance', 0)
                            total_value += balances[chain]['balance']

                # Apply cached Solana module
                if 'SOLANA_MODULE' in cached:
                    balances['SOLANA_MODULE'] = cached['SOLANA_MODULE']
                    total_value += cached['SOLANA_MODULE'].get('balance', 0)
                else:
                    balances['SOLANA_MODULE'] = {'balance': 0.0, 'native_balance': 0.0, 'native_symbol': 'SOL', 'pnl': 0.0, 'positions': 0}

                # Apply cached exchange balances
                if 'FUTURES' in cached:
                    balances['FUTURES'] = cached['FUTURES']
                    total_value += cached['FUTURES'].get('balance', 0)
                if 'SPOT' in cached:
                    balances['SPOT'] = cached['SPOT']
                    total_value += cached['SPOT'].get('balance', 0)
                if 'EXCHANGE_TOTAL' in cached:
                    balances['EXCHANGE_TOTAL'] = cached['EXCHANGE_TOTAL']

            else:
                # ========== FETCH FRESH DATA FROM BLOCKCHAIN ==========
                # Use a single session for all requests to reduce overhead
                prices = self._price_cache.copy()  # Start with cached prices

                # Pre-load cached native balances as defaults (so failures don't reset to 0)
                if self._wallet_cache:
                    for chain in chains:
                        if chain in self._wallet_cache and self._wallet_cache[chain].get('native_balance', 0) > 0:
                            balances[chain]['native_balance'] = self._wallet_cache[chain]['native_balance']

                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                        # ===== FETCH PRICES (once) =====
                        try:
                            url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum,binancecoin,matic-network,solana&vs_currencies=usd"
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    prices = {
                                        'ETH': data.get('ethereum', {}).get('usd', self._price_cache.get('ETH', 3500)),
                                        'BNB': data.get('binancecoin', {}).get('usd', self._price_cache.get('BNB', 600)),
                                        'MATIC': data.get('matic-network', {}).get('usd', self._price_cache.get('MATIC', 0.80)),
                                        'SOL': data.get('solana', {}).get('usd', self._price_cache.get('SOL', 200))
                                    }
                                    self._price_cache = prices  # Update cache
                        except Exception as e:
                            logger.debug(f"Price fetch failed, using cache: {e}")

                        chain_prices = {
                            'ETHEREUM': prices['ETH'], 'BSC': prices['BNB'], 'POLYGON': prices['MATIC'],
                            'ARBITRUM': prices['ETH'], 'BASE': prices['ETH']
                        }

                        # ===== SOLANA WALLETS =====
                        solana_rpc = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')

                        # Main Solana wallet
                        solana_wallet = os.getenv('SOLANA_WALLET')
                        if solana_wallet:
                            sol_fetched = False
                            try:
                                payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [solana_wallet]}
                                async with session.post(solana_rpc, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if 'error' not in data:
                                            lamports = data.get('result', {}).get('value', 0)
                                            sol_native = lamports / 1e9
                                            sol_usd = sol_native * prices['SOL']
                                            balances['SOLANA']['native_balance'] = sol_native
                                            if balances['SOLANA']['positions'] == 0:
                                                balances['SOLANA']['balance'] = sol_usd
                                                total_value += sol_usd
                                            sol_fetched = True
                                        else:
                                            logger.warning(f"Solana RPC error: {data.get('error')}")
                            except Exception as e:
                                logger.warning(f"Solana wallet fetch failed: {e}")

                            # Fallback to cached value if fetch failed
                            if not sol_fetched and self._wallet_cache and 'SOLANA' in self._wallet_cache:
                                cached_sol = self._wallet_cache['SOLANA']
                                if cached_sol.get('native_balance', 0) > 0:
                                    balances['SOLANA']['native_balance'] = cached_sol['native_balance']
                                    if balances['SOLANA']['positions'] == 0:
                                        balances['SOLANA']['balance'] = cached_sol['native_balance'] * prices['SOL']
                                        total_value += balances['SOLANA']['balance']

                        # Solana module wallet
                        solana_module_wallet = os.getenv('SOLANA_MODULE_WALLET')
                        if solana_module_wallet:
                            sol_mod_fetched = False
                            try:
                                payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [solana_module_wallet]}
                                async with session.post(solana_rpc, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if 'error' not in data:
                                            lamports = data.get('result', {}).get('value', 0)
                                            sol_native = lamports / 1e9
                                            sol_usd = sol_native * prices['SOL']
                                            balances['SOLANA_MODULE'] = {
                                                'balance': sol_usd, 'native_balance': sol_native,
                                                'native_symbol': 'SOL', 'pnl': 0.0, 'positions': 0
                                            }
                                            total_value += sol_usd
                                            sol_mod_fetched = True
                            except Exception as e:
                                logger.warning(f"Solana module fetch failed: {e}")

                            if not sol_mod_fetched:
                                # Use cached or default
                                if self._wallet_cache and 'SOLANA_MODULE' in self._wallet_cache:
                                    balances['SOLANA_MODULE'] = self._wallet_cache['SOLANA_MODULE'].copy()
                                    total_value += balances['SOLANA_MODULE'].get('balance', 0)
                                else:
                                    balances['SOLANA_MODULE'] = {'balance': 0.0, 'native_balance': 0.0, 'native_symbol': 'SOL', 'pnl': 0.0, 'positions': 0}
                        else:
                            balances['SOLANA_MODULE'] = {'balance': 0.0, 'native_balance': 0.0, 'native_symbol': 'SOL', 'pnl': 0.0, 'positions': 0}

                        # ===== EVM WALLETS =====
                        wallet_address = os.getenv('WALLET_ADDRESS')
                        if wallet_address:
                            evm_rpcs = {
                                'ETHEREUM': os.getenv('ETH_RPC_URL', 'https://eth.llamarpc.com'),
                                'BSC': os.getenv('BSC_RPC_URL', 'https://bsc-dataseed1.binance.org'),
                                'POLYGON': os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),
                                'ARBITRUM': os.getenv('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),
                                'BASE': os.getenv('BASE_RPC_URL', 'https://mainnet.base.org')
                            }

                            for chain, rpc_url in evm_rpcs.items():
                                chain_fetched = False
                                try:
                                    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance", "params": [wallet_address, "latest"]}
                                    async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                                        if resp.status == 200:
                                            data = await resp.json()
                                            if 'error' not in data:
                                                hex_balance = data.get('result', '0x0')
                                                if hex_balance and hex_balance != '0x0':
                                                    wei_balance = int(hex_balance, 16)
                                                    native_balance = wei_balance / 1e18
                                                    if native_balance > 0:
                                                        usd_value = native_balance * chain_prices.get(chain, 0)
                                                        balances[chain]['native_balance'] = native_balance
                                                        if balances[chain]['positions'] == 0:
                                                            balances[chain]['balance'] = usd_value
                                                            total_value += usd_value
                                                        chain_fetched = True
                                except Exception as e:
                                    logger.debug(f"{chain} balance fetch failed: {e}")

                                # Fallback to cached value if fetch failed
                                if not chain_fetched and self._wallet_cache and chain in self._wallet_cache:
                                    cached_chain = self._wallet_cache[chain]
                                    if cached_chain.get('native_balance', 0) > 0:
                                        balances[chain]['native_balance'] = cached_chain['native_balance']
                                        if balances[chain]['positions'] == 0:
                                            balances[chain]['balance'] = cached_chain['native_balance'] * chain_prices.get(chain, 0)
                                            total_value += balances[chain]['balance']

                        # ===== EXCHANGE BALANCES =====
                        import hmac
                        import hashlib
                        import time as time_module

                        futures_balance = spot_balance = futures_margin = futures_pnl = futures_available = 0.0
                        spot_assets = []
                        exchange_fetched = False

                        binance_key = os.getenv('BINANCE_API_KEY')
                        binance_secret = os.getenv('BINANCE_API_SECRET')

                        if binance_key and binance_secret:
                            headers = {'X-MBX-APIKEY': binance_key}

                            # Futures
                            try:
                                timestamp = int(time_module.time() * 1000)
                                query_string = f"timestamp={timestamp}"
                                signature = hmac.new(binance_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
                                url = f"https://fapi.binance.com/fapi/v2/balance?{query_string}&signature={signature}"
                                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                    if resp.status == 200:
                                        resp_data = await resp.json()
                                        for asset in resp_data:
                                            if asset.get('asset') == 'USDT':
                                                futures_balance = float(asset.get('balance', 0))
                                                futures_available = float(asset.get('availableBalance', 0))
                                                futures_pnl = float(asset.get('crossUnPnl', 0))
                                                futures_margin = futures_balance - futures_available
                                                exchange_fetched = True
                                                break
                                    else:
                                        logger.warning(f"Binance futures API returned status {resp.status}")
                            except Exception as e:
                                logger.warning(f"Binance futures fetch failed: {e}")

                            # Spot
                            try:
                                timestamp = int(time_module.time() * 1000)
                                query_string = f"timestamp={timestamp}"
                                signature = hmac.new(binance_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
                                url = f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}"
                                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                                    if resp.status == 200:
                                        resp_data = await resp.json()
                                        for bal in resp_data.get('balances', []):
                                            total_amt = float(bal.get('free', 0)) + float(bal.get('locked', 0))
                                            asset_name = bal.get('asset', '')
                                            if total_amt > 0:
                                                usd_value = 0
                                                if asset_name in ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD']:
                                                    usd_value = total_amt
                                                elif asset_name == 'BTC':
                                                    usd_value = total_amt * 95000
                                                elif asset_name == 'ETH':
                                                    usd_value = total_amt * prices['ETH']
                                                elif asset_name == 'BNB':
                                                    usd_value = total_amt * prices['BNB']
                                                elif asset_name == 'SOL':
                                                    usd_value = total_amt * prices['SOL']
                                                if usd_value >= 1:
                                                    spot_balance += usd_value
                                                    spot_assets.append({'asset': asset_name, 'total': total_amt, 'usd_value': usd_value})
                                                    exchange_fetched = True
                                    else:
                                        logger.warning(f"Binance spot API returned status {resp.status}")
                            except Exception as e:
                                logger.warning(f"Binance spot fetch failed: {e}")

                        # Set exchange balances (or use cached if fetch failed)
                        if exchange_fetched or not self._wallet_cache:
                            balances['FUTURES'] = {'total_balance': futures_balance, 'margin_used': futures_margin, 'unrealized_pnl': futures_pnl, 'available': futures_available, 'balance': futures_balance, 'pnl': futures_pnl, 'positions': 0}
                            balances['SPOT'] = {'total_balance': spot_balance, 'assets': spot_assets, 'balance': spot_balance}
                            balances['EXCHANGE_TOTAL'] = {'futures': futures_balance, 'spot': spot_balance, 'total': futures_balance + spot_balance}
                            total_value += futures_balance + spot_balance
                        else:
                            # Use cached exchange values
                            if 'FUTURES' in self._wallet_cache:
                                balances['FUTURES'] = self._wallet_cache['FUTURES'].copy()
                                total_value += balances['FUTURES'].get('balance', 0)
                            else:
                                balances['FUTURES'] = {'total_balance': 0, 'margin_used': 0, 'unrealized_pnl': 0, 'available': 0, 'balance': 0, 'pnl': 0, 'positions': 0}
                            if 'SPOT' in self._wallet_cache:
                                balances['SPOT'] = self._wallet_cache['SPOT'].copy()
                                total_value += balances['SPOT'].get('balance', 0)
                            else:
                                balances['SPOT'] = {'total_balance': 0, 'assets': [], 'balance': 0}
                            if 'EXCHANGE_TOTAL' in self._wallet_cache:
                                balances['EXCHANGE_TOTAL'] = self._wallet_cache['EXCHANGE_TOTAL'].copy()
                            else:
                                balances['EXCHANGE_TOTAL'] = {'futures': 0, 'spot': 0, 'total': 0}

                except Exception as e:
                    logger.error(f"Error fetching blockchain balances: {e}")
                    # Use cached values if fetch fails
                    if self._wallet_cache:
                        for key in ['SOLANA_MODULE', 'FUTURES', 'SPOT', 'EXCHANGE_TOTAL']:
                            if key in self._wallet_cache:
                                balances[key] = self._wallet_cache[key]

                # ===== UPDATE CACHE =====
                self._wallet_cache = {k: v.copy() if isinstance(v, dict) else v for k, v in balances.items()}
                self._wallet_cache_time = now

            # ========== CALCULATE P&L PERCENTAGE ==========
            for chain in chains:
                if balances[chain]['balance'] > 0 and balances[chain]['pnl'] != 0:
                    cost_basis = balances[chain]['balance'] - balances[chain]['pnl']
                    if cost_basis > 0:
                        balances[chain]['pnl_pct'] = (balances[chain]['pnl'] / cost_basis) * 100

            # ========== TOTAL ==========
            balances['TOTAL'] = {
                'balance': total_value,
                'pnl': total_pnl,
                'pnl_pct': 0.0,
                'positions': sum(balances[c].get('positions', 0) for c in chains)
            }

            return web.json_response({
                'status': 'success',
                'balances': balances,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Error getting wallet balances: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'error': str(e)
            }, status=500)


    def _calculate_duration(self, start, end):
        """Calculate duration between two timestamps"""
        if not start or not end:
            return "Unknown"
        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
            delta = end_dt - start_dt
            hours = delta.total_seconds() / 3600
            if hours < 1:
                return f"{int(delta.total_seconds() / 60)}m"
            elif hours < 24:
                return f"{int(hours)}h"
            else:
                return f"{int(hours / 24)}d"
        except:
            return "Unknown"
    
    async def api_performance_metrics(self, request):
        """Get detailed performance metrics from all closed trades."""
        try:
            if not self.db:
                return web.json_response({'error': 'Database connection not available.'}, status=503)

            # Fetch all closed trades from the database
            query = "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_timestamp ASC;"
            trades = await self.db.pool.fetch(query)

            if not trades:
                initial_balance = self.config_mgr.get_portfolio_config().initial_balance
                default_metrics = {
                    'initial_balance': initial_balance,
                    'total_pnl': 0.0, 'roi': 0.0, 'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0, 'daily_volatility': 0.0, 'annual_volatility': 0.0,
                    'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                    'best_trade': 0.0, 'worst_trade': 0.0, 'profit_factor': 0.0,
                    'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                }
                return web.json_response({'success': True, 'data': {'historical': default_metrics}})

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # --- FIX STARTS HERE ---
            # Basic metrics
            total_pnl = df['profit_loss'].sum()
            total_trades = len(df)
            winning_trades = df[df['profit_loss'] > 0]
            losing_trades = df[df['profit_loss'] <= 0]
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

            # Advanced metrics with safe defaults
            avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0
            best_trade = df['profit_loss'].max() if not df.empty else 0
            worst_trade = df['profit_loss'].min() if not df.empty else 0

            sum_of_wins = winning_trades['profit_loss'].sum()
            sum_of_losses = abs(losing_trades['profit_loss'].sum())
            profit_factor = sum_of_wins / sum_of_losses if sum_of_losses > 0 else (9999.99 if sum_of_wins > 0 else 0.0)

            # Sharpe Ratio (annualized, assuming risk-free rate is 0)
            daily_returns = df.set_index('exit_timestamp')['profit_loss'].resample('D').sum()
            # Ensure there's more than one period to calculate std dev
            if len(daily_returns) > 1 and daily_returns.std() != 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
            else:
                sharpe_ratio = 0.0

            # Correct Max Drawdown calculation based on equity
            initial_balance = self.config_mgr.get_portfolio_config().initial_balance
            df['cumulative_pnl'] = df['profit_loss'].cumsum()
            df['equity'] = initial_balance + df['cumulative_pnl']

            peak = df['equity'].expanding(min_periods=1).max()
            # Ensure division by zero is handled if peak is 0
            drawdown = ((df['equity'] - peak) / peak).replace([np.inf, -np.inf], 0)
            max_drawdown = abs(drawdown.min() * 100) if not drawdown.empty else 0

            # --- FIX STARTS HERE ---
            # Calculate ROI
            roi = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0

            # Calculate Sortino Ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else 0
            sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(365) if downside_std != 0 else 0

            # Calculate Calmar Ratio
            calmar_ratio = (daily_returns.mean() * 365) / (max_drawdown / 100) if max_drawdown != 0 else 0

            # Calculate Volatility
            daily_volatility = daily_returns.std() * 100
            annual_volatility = daily_returns.std() * np.sqrt(365) * 100
            # --- FIX ENDS HERE ---


            metrics = {
                'initial_balance': initial_balance,
                'total_pnl': total_pnl,
                'roi': roi,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'daily_volatility': daily_volatility,
                'annual_volatility': annual_volatility,
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
            }

            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = 0.0

            return web.json_response({
                'success': True,
                'data': {'historical': self._serialize_decimals(metrics)}
            })

        except Exception as e:
            logger.error(f"Error in api_performance_metrics: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    def _filter_by_period(self, trades, period):
        """Filter trades by time period"""
        if period == 'all':
            return trades
        
        now = datetime.utcnow()
        period_map = {
            '1h': timedelta(hours=1),
            '24h': timedelta(days=1),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
            '90d': timedelta(days=90)
        }
        
        delta = period_map.get(period, timedelta(days=7))
        cutoff = now - delta
        
        return [t for t in trades if t.get('exit_timestamp') and 
                datetime.fromisoformat(str(t['exit_timestamp']).replace('Z', '+00:00')) > cutoff]
    
    async def api_performance_charts(self, request):
        """Get performance chart data"""
        try:
            timeframe = request.query.get('timeframe', '7d')
            
            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)

            query = "SELECT exit_timestamp, profit_loss, strategy, metadata FROM trades WHERE status = 'closed' ORDER BY exit_timestamp ASC;"
            trades = await self.db.pool.fetch(query)

            if not trades:
                return web.json_response({'success': True, 'data': {
                    'equity_curve': [],
                    'cumulative_pnl': [],
                    'portfolio_history': [],  # For dashboard.html compatibility
                    'pnl_history': [],  # For dashboard.html compatibility
                    'strategy_performance': [],
                    'win_loss': {'wins': 0, 'losses': 0},
                    'monthly': [],
                }})

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # Use strategy column from DB, fallback to metadata if empty
            def get_strategy(row):
                if 'strategy' in row and row['strategy'] and row['strategy'] != 'unknown':
                    return row['strategy']
                return self._get_strategy_from_metadata(row.get('metadata'))

            df['strategy'] = df.apply(get_strategy, axis=1)

            # --- FIX STARTS HERE ---
            
            # 1. Calculate Equity Curve on the ENTIRE dataset first
            initial_balance = self.config_mgr.get_portfolio_config().initial_balance
            df_full = df.copy() # Use a copy for full history calculations
            df_full['cumulative_pnl'] = df_full['profit_loss'].cumsum()
            df_full['equity'] = initial_balance + df_full['cumulative_pnl']

            # 2. Now, filter the DataFrame by the requested timeframe
            if timeframe != 'all':
                now = pd.Timestamp.utcnow()
                # Use a mapping for timedelta
                time_delta_map = {
                    '1h': pd.Timedelta(hours=1),
                    '24h': pd.Timedelta(days=1),
                    '7d': pd.Timedelta(days=7),
                    '30d': pd.Timedelta(days=30),
                    '90d': pd.Timedelta(days=90)
                }
                delta = time_delta_map.get(timeframe, pd.Timedelta(days=7)) # Default to 7d

                # Filter both the main df and the full history df for display
                df = df[df['exit_timestamp'] >= (now - delta)]
                df_full_filtered = df_full[df_full['exit_timestamp'] >= (now - delta)]
            else:
                # If 'all' time, the filtered version is the same as the full
                df_full_filtered = df_full

            # 3. Generate chart data from the correctly filtered data
            # The equity curve uses the filtered full history, preserving the correct starting equity
            equity_curve_data = [{'timestamp': ts.isoformat(), 'value': val if not np.isnan(val) else 0.0} for ts, val in df_full_filtered[['exit_timestamp', 'equity']].values] if not df_full_filtered.empty else []
            cumulative_pnl_data = [{'timestamp': ts.isoformat(), 'cumulative_pnl': val if not np.isnan(val) else 0.0} for ts, val in df_full_filtered[['exit_timestamp', 'cumulative_pnl']].values] if not df_full_filtered.empty else []

            # Generate individual P&L history for bar chart (not cumulative)
            pnl_history_data = [{'timestamp': ts.isoformat(), 'value': float(val) if not np.isnan(val) else 0.0} for ts, val in df_full_filtered[['exit_timestamp', 'profit_loss']].values] if not df_full_filtered.empty else []

            # --- FIX ENDS HERE ---

            # Strategy Performance
            strategy_performance = df.groupby('strategy')['profit_loss'].sum().reset_index()
            strategy_performance.columns = ['strategy', 'pnl']

            # Win/Loss Distribution
            win_loss_distribution = {
                'wins': len(df[df['profit_loss'] > 0]),
                'losses': len(df[df['profit_loss'] <= 0])
            }

            # Monthly Performance
            df['month'] = df['exit_timestamp'].dt.to_period('M').astype(str)
            monthly_performance = df.groupby('month')['profit_loss'].sum().reset_index()
            monthly_performance.columns = ['month', 'pnl']

            return web.json_response({
                'success': True,
                'data': self._serialize_decimals({
                    'equity_curve': equity_curve_data,
                    'cumulative_pnl': cumulative_pnl_data,
                    'portfolio_history': equity_curve_data,  # For dashboard.html compatibility
                    'pnl_history': pnl_history_data,  # Individual P&L values for bar chart
                    'strategy_performance': strategy_performance.to_dict('records'),
                    'win_loss': win_loss_distribution,
                    'monthly': monthly_performance.to_dict('records'),
                })
            })

        except Exception as e:
            logger.error(f"Error getting performance charts: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_recent_alerts(self, request):
        """Get recent alerts"""
        try:
            limit = int(request.query.get('limit', 50))
            
            if not self.alerts:
                return web.json_response({'error': 'Alerts system not available'}, status=503)
            
            stats = self.alerts.get_alert_stats()
            recent = stats.get('recent_alerts', [])[-limit:]
            
            return web.json_response({
                'success': True,
                'data': recent,
                'count': len(recent)
            })
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # ==================== API - BOT CONTROL ====================
    
    async def api_bot_start(self, request):
        """Start the trading bot"""
        try:
            if not self.engine:
                return web.json_response({'error': 'Engine not available'}, status=503)
            
            await self.engine.start()
            
            return web.json_response({
                'success': True,
                'message': 'Bot started successfully'
            })
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_bot_stop(self, request):
        """Stop the trading bot"""
        try:
            if not self.engine:
                return web.json_response({'error': 'Engine not available'}, status=503)
            
            await self.engine.stop()
            
            return web.json_response({
                'success': True,
                'message': 'Bot stopped successfully'
            })
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_bot_restart(self, request):
        """Restart the trading bot"""
        try:
            if not self.engine:
                return web.json_response({'error': 'Engine not available'}, status=503)
            
            await self.engine.stop()
            await asyncio.sleep(2)
            await self.engine.start()
            
            return web.json_response({
                'success': True,
                'message': 'Bot restarted successfully'
            })
        except Exception as e:
            logger.error(f"Error restarting bot: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_emergency_exit(self, request):
        """Emergency exit - close all positions"""
        try:
            closed = []
            failed = []
            
            # ✅ Use self.engine.active_positions instead of self.portfolio
            if self.engine and hasattr(self.engine, 'active_positions'):
                positions = list(self.engine.active_positions.items())
                
                for token_address, pos in positions:
                    try:
                        # ✅ Call engine's close_position method
                        if hasattr(self.engine, 'close_position'):
                            result = await self.engine.close_position(
                                token_address=token_address,
                                reason='emergency_exit'
                            )
                            if result:
                                closed.append(token_address)
                            else:
                                failed.append(token_address)
                        else:
                            # Fallback: manually update status
                            pos['status'] = 'closed'
                            closed.append(token_address)
                    except Exception as e:
                        logger.error(f"Error closing position {token_address}: {e}")
                        failed.append(token_address)
            
            return web.json_response({
                'success': True,
                'message': f'Emergency exit completed. Closed: {len(closed)}, Failed: {len(failed)}',
                'closed': closed,
                'failed': failed
            })
        except Exception as e:
            logger.error(f"Error in emergency exit: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_bot_status(self, request):
        """Get the bot's running status and uptime."""
        try:
            from core.engine import BotState
            is_running = self.engine and self.engine.state == BotState.RUNNING
            uptime_str = "N/A"

            if is_running and self.engine.stats.get('start_time'):
                uptime_delta = datetime.utcnow() - self.engine.stats['start_time']
                hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                uptime_str = f"{hours}h {minutes}m"

            status = {
                'running': is_running,
                'uptime': uptime_str,
                'mode': self.config_mgr.get_general_config().mode,
                'dry_run': self.config_mgr.get_general_config().dry_run,
                'version': '1.0.0', # Replace with actual version if available
            }
            
            return web.json_response({
                'success': True,
                'data': status
            })
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            # Return success with offline status instead of error
            return web.json_response({
                'success': True,
                'data': {
                    'running': False,
                    'uptime': 'N/A',
                    'mode': 'unknown',
                    'version': '1.0.0',
                    'last_health_check': datetime.utcnow().isoformat()
                }
            }, status=200)
    
    # ==================== API - PORTFOLIO BLOCK MANAGEMENT ====================

    async def api_get_block_status(self, request):
        """Get detailed information about why trading is blocked"""
        try:
            # CRITICAL FIX: Use engine's portfolio manager (the one actually updated by trades)
            # The self.portfolio passed from main_dex is a DIFFERENT instance than engine.portfolio_manager
            portfolio_mgr = None
            if self.engine and hasattr(self.engine, 'portfolio_manager'):
                portfolio_mgr = self.engine.portfolio_manager
            elif self.portfolio:
                portfolio_mgr = self.portfolio  # Fallback to passed-in portfolio

            if not portfolio_mgr:
                return web.json_response({
                    'success': False,
                    'error': 'Portfolio manager not available'
                }, status=503)

            # Get block reason details from portfolio manager
            block_info = portfolio_mgr.get_block_reason()

            # ========== OVERRIDE WITH REAL DATA ==========
            # Get actual open positions count from database
            actual_positions_count = 0
            if self.db:
                try:
                    positions = await self.db.get_open_positions()
                    actual_positions_count = len(positions) if positions else 0
                except Exception as e:
                    logger.debug(f"Could not get positions from DB: {e}")

            # Also check engine's active_positions
            engine_positions_count = 0
            if self.engine and hasattr(self.engine, 'active_positions'):
                engine_positions_count = len(self.engine.active_positions) if self.engine.active_positions else 0

            # Use the higher of the two counts (most accurate)
            real_positions_count = max(actual_positions_count, engine_positions_count)

            # Override portfolio manager's positions count with real count
            block_info['positions_count'] = real_positions_count
            block_info['positions_count_db'] = actual_positions_count
            block_info['positions_count_engine'] = engine_positions_count

            # ========== OVERRIDE BALANCE WITH REAL DATA ==========
            # Get real balance from historical P&L
            if self.db:
                try:
                    # Get initial balance from config
                    initial_balance = 400.0  # Default
                    if self.config_mgr:
                        try:
                            portfolio_config = self.config_mgr.get_portfolio_config()
                            initial_balance = float(portfolio_config.initial_balance or 400.0)
                        except:
                            pass

                    # Calculate actual portfolio value from historical trades
                    trades = await self.db.get_recent_trades(limit=1000)
                    closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('profit_loss') is not None]
                    total_pnl = sum(float(t.get('profit_loss', 0)) for t in closed_trades)

                    # Starting balance + P&L = current portfolio value
                    real_portfolio_value = initial_balance + total_pnl

                    # Get value locked in open positions from ENGINE (same source as Open Positions API)
                    value_in_positions = 0.0
                    unrealized_pnl = 0.0
                    if self.engine and hasattr(self.engine, 'active_positions') and self.engine.active_positions:
                        for token_addr, pos in self.engine.active_positions.items():
                            entry_price = float(pos.get('entry_price', 0))
                            current_price = float(pos.get('current_price', entry_price))
                            amount = float(pos.get('amount', 0))
                            entry_val = amount * entry_price
                            current_val = amount * current_price
                            value_in_positions += entry_val
                            unrealized_pnl += (current_val - entry_val)

                    # Calculate real available balance (what can be used to open new positions)
                    real_available = real_portfolio_value - value_in_positions

                    # Override with real values
                    block_info['balance'] = real_portfolio_value
                    block_info['available_balance'] = max(0, real_available)
                    block_info['value_in_positions'] = value_in_positions
                    block_info['unrealized_pnl'] = unrealized_pnl
                    block_info['total_pnl'] = total_pnl
                    block_info['initial_balance'] = initial_balance

                except Exception as e:
                    logger.warning(f"Could not calculate real balance: {e}")

            # Recalculate can_trade based on real data
            # CRITICAL: Preserve original reasons from portfolio manager (consecutive losses, daily loss limit, etc.)
            original_reasons = block_info.get('reasons', [])
            original_can_trade = block_info.get('can_trade', True)
            max_positions = block_info.get('max_positions', 10)
            min_position_size = block_info.get('min_position_size', 5)
            available_balance = block_info.get('available_balance', 0)

            # Keep non-position/balance reasons intact (consecutive losses, daily loss, risk exposure)
            preserved_reasons = [r for r in original_reasons if 'Max positions' not in r and 'Insufficient balance' not in r]

            # Re-check position limit with real count
            if real_positions_count >= max_positions:
                preserved_reasons.append(f"Max positions reached: {real_positions_count}/{max_positions}")

            # Re-check balance with real available balance
            if available_balance < min_position_size:
                preserved_reasons.append(f"Insufficient balance: ${available_balance:.2f} < ${min_position_size:.2f} required")

            block_info['reasons'] = preserved_reasons
            # CRITICAL: can_trade is False if ANY reason exists
            block_info['can_trade'] = len(preserved_reasons) == 0

            # Log block status for debugging
            if not block_info['can_trade']:
                logger.info(f"Trading BLOCKED - Reasons: {preserved_reasons}")

            return web.json_response({
                'success': True,
                'data': block_info
            })

        except Exception as e:
            logger.error(f"Error getting block status: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_reset_block(self, request):
        """Manually reset trading block (use with caution!)"""
        try:
            # CRITICAL FIX: Use engine's portfolio manager (same as api_get_block_status)
            portfolio_mgr = None
            if self.engine and hasattr(self.engine, 'portfolio_manager'):
                portfolio_mgr = self.engine.portfolio_manager
            elif self.portfolio:
                portfolio_mgr = self.portfolio

            if not portfolio_mgr:
                return web.json_response({
                    'success': False,
                    'error': 'Portfolio manager not available'
                }, status=503)

            # Get reason from request body if provided
            try:
                data = await request.json()
                reason = data.get('reason', 'Manual reset via dashboard')
            except:
                reason = 'Manual reset via dashboard'

            # Call the manual reset method
            result = await portfolio_mgr.manual_reset_block(reason=reason)

            if result.get('success'):
                return web.json_response({
                    'success': True,
                    'message': result.get('message'),
                    'data': result
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }, status=400)

        except Exception as e:
            logger.error(f"Error resetting block: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    # ==================== API - SETTINGS ====================

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, SecretStr):
            return obj.get_secret_value() if obj else None
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Type {type(obj)} not serializable")

    async def api_get_settings(self, request):
        """Get all settings from database config_settings table."""
        try:
            # Load settings from database instead of Pydantic models
            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            async with self.db_pool.acquire() as conn:
                # Get all editable config settings from database
                rows = await conn.fetch("""
                    SELECT config_type, key, value, value_type, description, is_editable, requires_restart
                    FROM config_settings
                    WHERE is_editable = TRUE
                    ORDER BY config_type, key
                """)

                # Group by config_type
                all_configs = {}
                for row in rows:
                    config_type = row['config_type']
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    # Convert value based on type
                    if value_type == 'bool':
                        converted_value = value.lower() in ('true', '1', 'yes')
                    elif value_type == 'int':
                        converted_value = int(value)
                    elif value_type == 'float':
                        converted_value = float(value)
                    elif value_type == 'json':
                        converted_value = json.loads(value)
                    else:  # string
                        converted_value = value

                    # Add to config type group
                    if config_type not in all_configs:
                        all_configs[config_type] = {}

                    all_configs[config_type][key] = {
                        'value': converted_value,
                        'description': row['description'],
                        'requires_restart': row['requires_restart'],
                        'value_type': value_type
                    }

            return web.json_response({
                'success': True,
                'data': all_configs
            })

        except Exception as e:
            logger.error(f"Error getting settings: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_update_settings(self, request):
        """Update settings in database"""
        try:
            data = await request.json()
            config_type = data.get('config_type')
            updates = data.get('updates', {})

            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            if not config_type or not updates:
                return web.json_response({'error': 'config_type and updates required'}, status=400)

            # Get user info for audit
            user = request.get('user')
            user_id = user.id if user else None
            username = user.username if user else 'unknown'

            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    for key, value in updates.items():
                        # Get the current value for audit log
                        old_row = await conn.fetchrow("""
                            SELECT value, value_type FROM config_settings
                            WHERE config_type = $1 AND key = $2
                        """, config_type, key)

                        if not old_row:
                            logger.warning(f"Config {config_type}.{key} not found, skipping")
                            continue

                        # Convert value to string based on type
                        value_type = old_row['value_type']
                        if value_type == 'bool':
                            new_value_str = 'true' if value else 'false'
                        else:
                            new_value_str = str(value)

                        # Update the config setting
                        await conn.execute("""
                            UPDATE config_settings
                            SET value = $1, updated_at = NOW(), updated_by = $2
                            WHERE config_type = $3 AND key = $4
                        """, new_value_str, user_id, config_type, key)

                        # Log the change in config_history
                        await conn.execute("""
                            INSERT INTO config_history
                            (config_type, key, old_value, new_value, change_source, changed_by, changed_by_username, ip_address)
                            VALUES ($1, $2, $3, $4, 'api', $5, $6, $7)
                        """, config_type, key, old_row['value'], new_value_str, user_id, username,
                             request.remote)

            return web.json_response({
                'success': True,
                'message': f'Settings updated: {config_type}'
            })
        except Exception as e:
            logger.error(f"Error updating settings: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_revert_settings(self, request):
        """Revert settings to previous version"""
        return web.json_response({
            'success': False,
            'error': 'This feature is temporarily disabled.'
        }, status=503)

    async def api_settings_history(self, request):
        """Get settings change history"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            # Get history from database
            history = await self.config_mgr.get_config_history(limit=100)

            return web.json_response({
                'success': True,
                'data': history,
                'count': len(history)
            })
        except Exception as e:
            logger.error(f"Error getting settings history: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_get_futures_settings(self, request):
        """Get all futures module settings from database"""
        try:
            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            settings = {}
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT config_type, key, value, value_type
                    FROM config_settings
                    WHERE config_type LIKE 'futures_%'
                    ORDER BY config_type, key
                """)

                for row in rows:
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    # Convert value based on type
                    if value_type == 'int':
                        settings[f"futures_{key}"] = int(value)
                    elif value_type == 'float':
                        settings[f"futures_{key}"] = float(value)
                    elif value_type == 'bool':
                        settings[f"futures_{key}"] = value.lower() in ('true', '1', 'yes')
                    else:
                        settings[f"futures_{key}"] = value

            # Add API key availability flags (don't expose actual keys)
            import os
            settings['_has_binance_api'] = bool(os.getenv('BINANCE_TESTNET_API_KEY') or os.getenv('BINANCE_API_KEY'))
            settings['_has_bybit_api'] = bool(os.getenv('BYBIT_TESTNET_API_KEY') or os.getenv('BYBIT_API_KEY'))

            return web.json_response({
                'success': True,
                'settings': settings
            })
        except Exception as e:
            logger.error(f"Error getting futures settings: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_save_futures_settings(self, request):
        """Save futures module settings to database"""
        try:
            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            data = await request.json()
            user_id = request.get('user_id', None)  # From auth middleware if available

            # Key mapping from UI field names to config field names
            key_mapping = {
                'daily_loss_limit': 'max_daily_loss_pct',  # UI uses % field
                'stop_loss': 'stop_loss_pct',
                'take_profit': 'take_profit_pct',
                'trailing_stop': 'trailing_stop_enabled',
                'trailing_distance': 'trailing_stop_distance',
                'leverage': 'default_leverage',
                'capital': 'capital_allocation',
                'funding_arb': 'funding_arbitrage_enabled',
                'signal_timeframe': 'signal_timeframe',
                'scan_interval': 'scan_interval_seconds',
                'signal_score': 'min_signal_score',
                'cooldown': 'cooldown_minutes',
            }

            async with self.db_pool.acquire() as conn:
                for key, value in data.items():
                    # Remove futures_ prefix if present
                    clean_key = key.replace('futures_', '') if key.startswith('futures_') else key

                    # Apply key mapping
                    clean_key = key_mapping.get(clean_key, clean_key)

                    # Determine config_type from key
                    config_type = self._get_futures_config_type(clean_key)
                    if not config_type:
                        continue

                    # Determine value type
                    if isinstance(value, bool):
                        value_type = 'bool'
                        value_str = str(value).lower()
                    elif isinstance(value, int):
                        value_type = 'int'
                        value_str = str(value)
                    elif isinstance(value, float):
                        value_type = 'float'
                        value_str = str(value)
                    else:
                        value_type = 'string'
                        value_str = str(value)

                    # Get old value for history
                    old_row = await conn.fetchrow("""
                        SELECT value FROM config_settings
                        WHERE config_type = $1 AND key = $2
                    """, config_type, clean_key)
                    old_value = old_row['value'] if old_row else None

                    # Update or insert
                    await conn.execute("""
                        INSERT INTO config_settings (config_type, key, value, value_type, updated_by)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (config_type, key) DO UPDATE
                        SET value = $3, value_type = $4, updated_by = $5, updated_at = NOW()
                    """, config_type, clean_key, value_str, value_type, user_id)

                    # Log to history if changed
                    if old_value != value_str:
                        await conn.execute("""
                            INSERT INTO config_history (config_type, key, old_value, new_value, change_source, changed_by)
                            VALUES ($1, $2, $3, $4, 'api', $5)
                        """, config_type, clean_key, old_value, value_str, user_id)

            return web.json_response({'success': True, 'message': 'Settings saved'})
        except Exception as e:
            logger.error(f"Error saving futures settings: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    def _get_futures_config_type(self, key: str) -> str:
        """Map a setting key to its config_type"""
        type_map = {
            'enabled': 'futures_general', 'exchange': 'futures_general', 'testnet': 'futures_general',
            'trading_mode': 'futures_general', 'contract_type': 'futures_general',
            'capital_allocation': 'futures_position', 'capital': 'futures_position',
            'position_size_usd': 'futures_position', 'max_position_pct': 'futures_position',
            'max_positions': 'futures_position', 'min_trade_size': 'futures_position',
            'default_leverage': 'futures_leverage', 'leverage': 'futures_leverage',
            'max_leverage': 'futures_leverage', 'margin_mode': 'futures_leverage',
            'stop_loss_pct': 'futures_risk', 'stop_loss': 'futures_risk',
            'take_profit_pct': 'futures_risk', 'take_profit': 'futures_risk',
            'max_daily_loss_usd': 'futures_risk', 'daily_loss_limit': 'futures_risk',
            'max_daily_loss_pct': 'futures_risk', 'liquidation_buffer': 'futures_risk',
            'trailing_stop_enabled': 'futures_risk', 'trailing_stop': 'futures_risk',
            'trailing_stop_distance': 'futures_risk', 'trailing_distance': 'futures_risk',
            'max_consecutive_losses': 'futures_risk',
            'allowed_pairs': 'futures_pairs', 'both_directions': 'futures_pairs',
            'preferred_direction': 'futures_pairs',
            'rsi_oversold': 'futures_strategy', 'rsi_overbought': 'futures_strategy',
            'rsi_weak_oversold': 'futures_strategy', 'rsi_weak_overbought': 'futures_strategy',
            'min_signal_score': 'futures_strategy', 'verbose_signals': 'futures_strategy',
            'cooldown_minutes': 'futures_strategy',
            'funding_arbitrage_enabled': 'futures_funding', 'funding_arb': 'futures_funding',
            'max_funding_rate': 'futures_funding',
        }
        return type_map.get(key)

    async def api_get_solana_settings(self, request):
        """Get all solana module settings from database"""
        try:
            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            settings = {}
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT config_type, key, value, value_type
                    FROM config_settings
                    WHERE config_type LIKE 'solana_%'
                    ORDER BY config_type, key
                """)

                for row in rows:
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    # Convert value based on type
                    if value_type == 'int':
                        settings[f"solana_{key}"] = int(value)
                    elif value_type == 'float':
                        settings[f"solana_{key}"] = float(value)
                    elif value_type == 'bool':
                        settings[f"solana_{key}"] = value.lower() in ('true', '1', 'yes')
                    else:
                        settings[f"solana_{key}"] = value

            # Add API key availability flags
            import os
            settings['_has_solana_wallet'] = bool(os.getenv('SOLANA_WALLET') or os.getenv('SOLANA_MODULE_WALLET'))
            settings['_has_jupiter_api'] = bool(os.getenv('JUPITER_API_KEY'))
            settings['_has_helius_api'] = bool(os.getenv('HELIUS_API_KEY'))

            return web.json_response({
                'success': True,
                'settings': settings
            })
        except Exception as e:
            logger.error(f"Error getting solana settings: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_save_solana_settings(self, request):
        """Save solana module settings to database"""
        try:
            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            data = await request.json()
            user_id = request.get('user_id', None)

            async with self.db_pool.acquire() as conn:
                for key, value in data.items():
                    clean_key = key.replace('solana_', '') if key.startswith('solana_') else key
                    config_type = self._get_solana_config_type(clean_key)
                    if not config_type:
                        continue

                    if isinstance(value, bool):
                        value_type = 'bool'
                        value_str = str(value).lower()
                    elif isinstance(value, int):
                        value_type = 'int'
                        value_str = str(value)
                    elif isinstance(value, float):
                        value_type = 'float'
                        value_str = str(value)
                    else:
                        value_type = 'string'
                        value_str = str(value)

                    old_row = await conn.fetchrow("""
                        SELECT value FROM config_settings
                        WHERE config_type = $1 AND key = $2
                    """, config_type, clean_key)
                    old_value = old_row['value'] if old_row else None

                    await conn.execute("""
                        INSERT INTO config_settings (config_type, key, value, value_type, updated_by)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (config_type, key) DO UPDATE
                        SET value = $3, value_type = $4, updated_by = $5, updated_at = NOW()
                    """, config_type, clean_key, value_str, value_type, user_id)

                    if old_value != value_str:
                        await conn.execute("""
                            INSERT INTO config_history (config_type, key, old_value, new_value, change_source, changed_by)
                            VALUES ($1, $2, $3, $4, 'api', $5)
                        """, config_type, clean_key, old_value, value_str, user_id)

            return web.json_response({'success': True, 'message': 'Settings saved'})
        except Exception as e:
            logger.error(f"Error saving solana settings: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    def _get_solana_config_type(self, key: str) -> str:
        """Map a solana setting key to its config_type"""
        type_map = {
            'enabled': 'solana_general', 'strategies': 'solana_general',
            'capital_allocation_sol': 'solana_position', 'position_size_sol': 'solana_position',
            'max_positions': 'solana_position',
            'stop_loss_pct': 'solana_risk', 'take_profit_pct': 'solana_risk',
            'max_daily_loss_sol': 'solana_risk',
            'priority_fee': 'solana_tx', 'compute_unit_price': 'solana_tx',
            'compute_unit_limit': 'solana_tx',
            'slippage_bps': 'solana_jupiter', 'api_tier': 'solana_jupiter',
            'leverage': 'solana_drift', 'markets': 'solana_drift',
            'min_liquidity': 'solana_pumpfun', 'max_age_seconds': 'solana_pumpfun',
            'buy_amount_sol': 'solana_pumpfun',
        }
        # Handle enabled flags for sub-strategies
        if key.startswith('jupiter_'):
            return 'solana_jupiter'
        if key.startswith('drift_'):
            return 'solana_drift'
        if key.startswith('pumpfun_'):
            return 'solana_pumpfun'
        return type_map.get(key)

    async def api_list_sensitive_configs(self, request):
        """List all sensitive configuration keys (admin only)"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            # Get list of sensitive config keys (without values)
            configs = await self.config_mgr.list_sensitive_configs()

            return web.json_response({
                'success': True,
                'data': configs
            })
        except Exception as e:
            logger.error(f"Error listing sensitive configs: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_get_sensitive_config(self, request):
        """Get a specific sensitive configuration with decrypted value (admin only)"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            key = request.match_info.get('key')

            if not key:
                return web.json_response({
                    'success': False,
                    'error': 'Key parameter is required'
                }, status=400)

            # Get sensitive config with decrypted value and metadata
            config = await self.config_mgr.get_sensitive_config_with_metadata(key)

            if config:
                user = request.get('user')
                logger.info(f"Admin {user.username if user else 'unknown'} accessed sensitive config: {key}")

                return web.json_response({
                    'success': True,
                    'data': config
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Sensitive config "{key}" not found'
                }, status=404)

        except Exception as e:
            logger.error(f"Error getting sensitive config: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_set_sensitive_config(self, request):
        """Set or update a sensitive configuration value (admin only)"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            data = await request.json()
            key = data.get('key')
            value = data.get('value')
            description = data.get('description', '')
            rotation_days = data.get('rotation_days', 30)

            if not key or not value:
                return web.json_response({
                    'success': False,
                    'error': 'Key and value are required'
                }, status=400)

            # Get user ID from request
            user = request.get('user')
            user_id = user.id if user else None

            # Set the sensitive config (will be encrypted)
            success = await self.config_mgr.set_sensitive_config(
                key=key,
                value=value,
                description=description,
                user_id=user_id,
                rotation_days=rotation_days
            )

            if success:
                logger.info(f"Admin {user.username if user else 'unknown'} set sensitive config: {key}")
                return web.json_response({
                    'success': True,
                    'message': f'Sensitive config {key} saved successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to save sensitive config'
                }, status=500)

        except Exception as e:
            logger.error(f"Error setting sensitive config: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_delete_sensitive_config(self, request):
        """Delete a sensitive configuration (admin only)"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            key = request.match_info.get('key')

            if not key:
                return web.json_response({
                    'success': False,
                    'error': 'Key is required'
                }, status=400)

            # Get user for logging
            user = request.get('user')

            # Delete the sensitive config
            success = await self.config_mgr.delete_sensitive_config(key)

            if success:
                logger.info(f"Admin {user.username if user else 'unknown'} deleted sensitive config: {key}")
                return web.json_response({
                    'success': True,
                    'message': f'Sensitive config {key} deleted successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to delete sensitive config'
                }, status=500)

        except Exception as e:
            logger.error(f"Error deleting sensitive config: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    # ==================== API - TRADING CONTROLS ====================
    
    async def api_execute_trade(self, request):
        """Execute manual trade"""
        try:
            data = await request.json()
            
            token = data.get('token')
            side = data.get('side')  # buy/sell
            amount = data.get('amount')
            order_type = data.get('order_type', 'market')
            
            if not all([token, side, amount]):
                return web.json_response({
                    'success': False,
                    'error': 'Missing required fields'
                }, status=400)
            
            # Create and execute order
            order = {
                'token': token,
                'side': side,
                'amount': amount,
                'type': order_type,
                'source': 'manual_dashboard'
            }
            
            result = await self.orders.create_order_from_params(
                token_address=token,
                side=side,
                amount=Decimal(str(amount)),
                order_type=order_type,
                price=None
            )
            
            return web.json_response({
                'success': True,
                'message': 'Trade executed successfully',
                'order_id': result
            })
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_close_position(self, request):
        """Close a position"""
        try:
            data = await request.json()
            position_id = data.get('position_id')
            
            if not position_id:
                return web.json_response({
                    'success': False,
                    'error': 'Position ID required'
                }, status=400)
            
            # ✅ FIX: Look for position in ENGINE's active_positions
            if not self.engine or not hasattr(self.engine, 'active_positions'):
                return web.json_response({
                    'success': False,
                    'error': 'Trading engine not available'
                }, status=503)
            
            # Find the position by ID
            position = None
            token_address = None
            
            for addr, pos in self.engine.active_positions.items():
                if pos.get('id') == position_id:
                    position = pos
                    token_address = addr
                    break
            
            if not position:
                return web.json_response({
                    'success': False,
                    'error': f'Position {position_id} not found in active positions'
                }, status=404)
            
            # ✅ Close the position via engine
            try:
                # Call the engine's close position method
                await self.engine._close_position(position, reason="Manual close via dashboard")
                
                logger.info(f"Position {position_id} closed successfully via dashboard")
                
                return web.json_response({
                    'success': True,
                    'message': f"Position closed: {position.get('token_symbol', 'Unknown')}",
                    'data': {
                        'position_id': position_id,
                        'token_symbol': position.get('token_symbol'),
                        'closed': True
                    }
                })
            except Exception as e:
                logger.error(f"Error closing position via engine: {e}")
                return web.json_response({
                    'success': False,
                    'error': f'Failed to close position: {str(e)}'
                }, status=500)
                
        except Exception as e:
            logger.error(f"Error in api_close_position: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    # ========== FUTURES POSITION MANAGEMENT ==========

    async def api_futures_positions(self, request):
        """Get all futures positions from the Futures module"""
        try:
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{futures_port}/positions', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return web.json_response(data)
                    else:
                        return web.json_response({
                            'success': False,
                            'error': f'Futures module returned status {resp.status}'
                        }, status=resp.status)
        except Exception as e:
            logger.error(f"Error fetching futures positions: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_futures_trades(self, request):
        """Get recent futures trades from the Futures module"""
        try:
            limit = request.query.get('limit', '50')
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{futures_port}/trades?limit={limit}', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return web.json_response(data)
                    else:
                        return web.json_response({
                            'success': False,
                            'error': f'Futures module returned status {resp.status}'
                        }, status=resp.status)
        except Exception as e:
            logger.error(f"Error fetching futures trades: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_futures_close_position(self, request):
        """Close a specific futures position"""
        try:
            data = await request.json()
            symbol = data.get('symbol')

            # Debug logging to trace symbol through proxy
            logger.info(f"🔍 Proxying close request for symbol: '{symbol}'")

            if not symbol:
                return web.json_response({
                    'success': False,
                    'error': 'Symbol is required'
                }, status=400)

            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://localhost:{futures_port}/position/close',
                    json={'symbol': symbol},
                    timeout=10
                ) as resp:
                    response_data = await resp.json()
                    logger.info(f"🔍 Futures module response: status={resp.status}, data={response_data}")
                    return web.json_response(response_data, status=resp.status)

        except Exception as e:
            logger.error(f"Error closing futures position: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_futures_close_all_positions(self, request):
        """Close all futures positions"""
        try:
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://localhost:{futures_port}/positions/close-all',
                    timeout=30
                ) as resp:
                    data = await resp.json()
                    return web.json_response(data, status=resp.status)

        except Exception as e:
            logger.error(f"Error closing all futures positions: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_futures_trading_status(self, request):
        """Get futures trading status including block status"""
        try:
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://localhost:{futures_port}/trading/status', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return web.json_response(data)
                    else:
                        return web.json_response({
                            'success': False,
                            'error': f'Futures module returned status {resp.status}'
                        }, status=resp.status)
        except Exception as e:
            logger.error(f"Error fetching futures trading status: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_futures_trading_unblock(self, request):
        """Unblock futures trading by resetting daily loss/consecutive losses"""
        try:
            futures_port = int(os.getenv('FUTURES_HEALTH_PORT', '8081'))
            data = await request.json() if request.content_length else {}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'http://localhost:{futures_port}/trading/unblock',
                    json=data,
                    timeout=5
                ) as resp:
                    result = await resp.json()
                    return web.json_response(result, status=resp.status)

        except Exception as e:
            logger.error(f"Error unblocking futures trading: {e}")
            return web.json_response({
                'success': False,
                'error': f'Futures module not available: {str(e)}'
            }, status=503)

    async def api_modify_position(self, request):
        """Modify position (stop loss, take profit)"""
        try:
            data = await request.json()
            position_id = data.get('position_id')
            modifications = data.get('modifications', {})
            
            if not position_id:
                return web.json_response({
                    'success': False,
                    'error': 'Position ID required'
                }, status=400)
            
            result = self.portfolio.update_position(position_id, modifications)
            
            return web.json_response({
                'success': True,
                'message': 'Position modified successfully',
                'data': result
            })
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_cancel_order(self, request):
        """Cancel an order"""
        try:
            data = await request.json()
            order_id = data.get('order_id')
            
            if not order_id:
                return web.json_response({
                    'success': False,
                    'error': 'Order ID required'
                }, status=400)
            
            result = self.orders.cancel_order(order_id)
            
            return web.json_response({
                'success': True,
                'message': 'Order cancelled successfully',
                'data': result
            })
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # ==================== API - REPORTS ====================
    
    async def api_generate_report(self, request):
        """Generate performance report"""
        try:
            data = await request.json()
            
            period = data.get('period', 'daily')  # daily, weekly, monthly, custom
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            metrics = data.get('metrics', ['all'])
            
            # Generate report based on parameters
            report = await self._generate_report(period, start_date, end_date, metrics)
            
            return web.json_response({
                'success': True,
                'data': report
            })
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_export_report(self, request):
        """Export report in various formats"""
        try:
            format_type = request.match_info['format']  # csv, excel, pdf, json

            # Get query parameters for custom date range
            period = request.query.get('period', 'custom')
            start_date = request.query.get('start_date')
            end_date = request.query.get('end_date')
            metrics = request.query.get('metrics', 'all')

            # If no custom dates provided, use period-based report
            if not start_date or not end_date:
                period = request.query.get('period', 'daily')
                report = await self._generate_report(period, None, None, ['all'])
            else:
                # Use custom date range
                report = await self._generate_report('custom', start_date, end_date, ['all'])

            if format_type == 'csv':
                return await self._export_csv(report)
            elif format_type == 'excel':
                return await self._export_excel(report)
            elif format_type == 'pdf':
                return await self._export_pdf(report)
            elif format_type == 'json':
                return web.json_response(report)
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Invalid format'
                }, status=400)
        except Exception as e:
            logger.error(f"Error exporting report: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_custom_report(self, request):
        """Generate custom report with specific filters"""
        try:
            # Parse query parameters for custom filters
            filters = {
                'tokens': request.query.getall('token', []),
                'strategies': request.query.getall('strategy', []),
                'min_pnl': request.query.get('min_pnl'),
                'max_pnl': request.query.get('max_pnl'),
                'start_date': request.query.get('start_date'),
                'end_date': request.query.get('end_date')
            }
            
            report = await self._generate_custom_report(filters)
            
            return web.json_response({
                'success': True,
                'data': report
            })
        except Exception as e:
            logger.error(f"Error generating custom report: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # ==================== API - BACKTESTING ====================
    
    async def api_run_backtest(self, request):
        """Run backtest with parameters"""
        try:
            data = await request.json()
            
            strategy = data.get('strategy')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_balance = data.get('initial_balance', 1.0)
            parameters = data.get('parameters', {})
            
            # Run backtest
            test_id = f"backtest_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Store backtest task
            self.backtests[test_id] = {'status': 'running', 'progress': 'Initializing...'}
            asyncio.create_task(self._run_backtest_task(
                test_id, strategy, start_date, end_date, initial_balance, parameters
            ))
            
            return web.json_response({
                'success': True,
                'test_id': test_id,
                'message': 'Backtest started'
            })
        except Exception as e:
            logger.error(f"Error starting backtest: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_backtest_results(self, request):
        """Get backtest results"""
        try:
            test_id = request.match_info['test_id']
            results = self.backtests.get(test_id, {'status': 'not_found'})
            
            return web.json_response({
                'success': True,
                'data': results
            })
        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # ==================== API - STRATEGY ====================
    
    async def api_get_strategy_params(self, request):
        """Get strategy parameters"""
        try:
            strategy_name = request.query.get('strategy', 'all')
            
            # Get parameters from config or strategy manager
            params = {}
            
            return web.json_response({
                'success': True,
                'data': params
            })
        except Exception as e:
            logger.error(f"Error getting strategy parameters: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_update_strategy_params(self, request):
        """Update strategy parameters"""
        try:
            data = await request.json()
            strategy_name = data.get('strategy')
            parameters = data.get('parameters', {})
            
            # Update strategy parameters
            # This should update the strategy configuration
            
            return web.json_response({
                'success': True,
                'message': f'Strategy parameters updated: {strategy_name}'
            })
        except Exception as e:
            logger.error(f"Error updating strategy parameters: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    # ==================== SSE HANDLER ====================
    
    async def sse_handler(self, request):
        """Server-Sent Events for real-time updates"""
        async with sse_response(request) as resp:
            try:
                # ✅ ADD: Send initial connection message
                await resp.send(json.dumps({
                    'type': 'connected',
                    'timestamp': datetime.utcnow().isoformat()
                }))
                
                while True:
                    try:
                        # ✅ FIX: Increase interval from 2 to 10 seconds
                        await asyncio.sleep(10)  # Changed from 2 to 10
                        
                        # Send updates
                        # Send updates
                        if self.db:
                            try:
                                async with self.db.pool.acquire() as conn:
                                    result = await conn.fetchrow("""
                                        SELECT 
                                            COALESCE(SUM(CASE 
                                                WHEN status = 'closed' 
                                                THEN (exit_price - entry_price) * amount 
                                                ELSE 0 
                                            END), 0) as total_pnl
                                        FROM trades
                                    """)
                                    if result:
                                        total_pnl = float(result['total_pnl'])
                                        portfolio_value = 400 + total_pnl
                                        
                                        await resp.send(json.dumps({
                                            'type': 'portfolio_update',
                                            'data': {
                                                'value': portfolio_value,
                                                'pnl': total_pnl,
                                                'timestamp': datetime.utcnow().isoformat()
                                            }
                                        }))
                            except Exception as e:
                                logger.debug(f"Error getting portfolio update: {e}")
                        
                    except asyncio.CancelledError:
                        logger.debug("SSE connection cancelled")
                        break
                        
            except ConnectionResetError:
                logger.debug("SSE connection reset by client")
            except Exception as e:
                logger.error(f"SSE error: {e}")
            finally:
                logger.debug("SSE connection closed")
        
        return resp
    
    # ==================== HELPER METHODS ====================

    def _get_strategy_from_metadata(self, metadata: Any) -> str:
        """
        Robustly search for a 'strategy' name in the metadata, which can be
        a JSON string or a dictionary. Handles multiple formats.
        """
        default_name = "unknown"

        if not metadata:
            return default_name

        # If metadata is a string, parse it to a dict
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                # If it's just a plain string (not JSON), it might be the strategy name
                return metadata if metadata else default_name

        if not isinstance(metadata, dict):
            return default_name

        # --- FIX STARTS HERE: Handle multiple common metadata structures ---
        # 1. Direct 'strategy_name' key
        if 'strategy_name' in metadata and isinstance(metadata['strategy_name'], str):
            return metadata['strategy_name']

        # 2. Nested 'strategy' dictionary with a 'name' key
        if 'strategy' in metadata and isinstance(metadata['strategy'], dict):
            return metadata['strategy'].get('name', default_name)

        # 3. Direct 'strategy' key that is a string
        if 'strategy' in metadata and isinstance(metadata['strategy'], str):
            return metadata['strategy']

        # 4. Fallback for other potential nested structures (recursive)
        for key, value in metadata.items():
            if isinstance(value, dict):
                strategy = self._get_strategy_from_metadata(value)
                if strategy != default_name:
                    return strategy
        # --- FIX ENDS HERE ---

        return default_name
    
    async def _send_initial_data(self, sid):
        """Send initial data to newly connected client"""
        try:
            # Get portfolio summary
            # Calculate real P&L from database
            initial_balance = 400.0
            cumulative_pnl = 0.0

            if self.db:
                async with self.db.pool.acquire() as conn:
                    result = await conn.fetchrow("""
                        SELECT COALESCE(SUM(CASE 
                            WHEN status = 'closed' 
                            THEN (exit_price - entry_price) * amount 
                            ELSE 0 
                        END), 0) as total_pnl
                        FROM trades
                    """)
                    if result:
                        cumulative_pnl = float(result['total_pnl'])

            portfolio_value = initial_balance + cumulative_pnl

            portfolio_data = {
                'total_value': portfolio_value,
                'cash_balance': initial_balance,
                'realized_pnl': cumulative_pnl,
                'daily_pnl': cumulative_pnl,
                'open_positions': len(self.engine.active_positions) if self.engine else 0
            }
            
            # ✅ FIX: Get ACTUAL positions from engine
            positions_data = []
            if self.engine and hasattr(self.engine, 'active_positions'):
                for token_address, position in self.engine.active_positions.items():
                    positions_data.append({
                        'id': position.get('id'),
                        'token_address': token_address,
                        'token_symbol': position.get('token_symbol', 'Unknown'),
                        'entry_price': float(position.get('entry_price', 0)),
                        'current_price': float(position.get('current_price', position.get('entry_price', 0))),
                        'amount': float(position.get('amount', 0)),
                        'unrealized_pnl': float(position.get('unrealized_pnl', 0)),
                        'status': position.get('status', 'open')
                    })
            
            # ✅ FIX: Get ACTUAL recent orders from database
            orders_data = []
            if self.db:
                try:
                    recent_trades = await self.db.get_recent_trades(limit=10)
                    orders_data = self._serialize_decimals(recent_trades)
                except Exception as e:
                    logger.error(f"Error getting recent trades: {e}")
            
            # Send initial data
            await self.sio.emit('initial_data', {
                'portfolio': portfolio_data,
                'positions': positions_data,
                'orders': orders_data
            }, room=sid)
            
            logger.debug(f"Sent initial data to {sid}: {len(positions_data)} positions, {len(orders_data)} orders")
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def _broadcast_loop(self):
        """Broadcast updates to all connected clients"""
        while True:
            try:
                await asyncio.sleep(5)
                
                if not self.sio.manager.rooms:
                    continue
                
                # Broadcast dashboard updates
                # Get open positions count from engine
                open_positions = 0
                if self.engine and hasattr(self.engine, 'active_positions'):
                    open_positions = len(self.engine.active_positions)

                # Get P&L and portfolio value from database
                # Get P&L and portfolio value from database
                total_pnl = 0
                portfolio_value = 400  # Default initial balance
                starting_balance = 400

                if self.db:
                    try:
                        # Query database directly for accurate P&L
                        async with self.db.pool.acquire() as conn:
                            result = await conn.fetchrow("""
                                SELECT 
                                    COALESCE(SUM(CASE 
                                        WHEN status = 'closed' 
                                        THEN (exit_price - entry_price) * amount 
                                        ELSE 0 
                                    END), 0) as total_pnl
                                FROM trades
                            """)
                            if result:
                                total_pnl = float(result['total_pnl'])
                                portfolio_value = starting_balance + total_pnl
                    except Exception as e:
                        logger.debug(f"Error getting performance data for broadcast: {e}")

                # Broadcast the update
                try:
                    await self.sio.emit('dashboard_update', {
                        'portfolio_value': float(portfolio_value),
                        'daily_pnl': float(total_pnl),
                        'open_positions': open_positions,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"Error broadcasting dashboard update: {e}")
                
                # Broadcast wallet balance updates
                # Broadcast wallet balance updates check differences
                try:
                    # Get cumulative P&L from database (same as dashboard_update)
                    cumulative_pnl = 0
                    if self.db:
                        try:
                            async with self.db.pool.acquire() as conn:
                                result = await conn.fetchrow("""
                                    SELECT 
                                        COALESCE(SUM(CASE 
                                            WHEN status = 'closed' 
                                            THEN (exit_price - entry_price) * amount 
                                            ELSE 0 
                                        END), 0) as total_pnl
                                    FROM trades
                                """)
                                if result:
                                    cumulative_pnl = float(result['total_pnl'])
                        except Exception as e:
                            logger.debug(f"Error getting cumulative PnL: {e}")
                    
                    # Calculate total portfolio value
                    starting_balance = 400.0
                    total_portfolio_value = starting_balance + cumulative_pnl
                    
                    # Calculate position values by chain
                    positions_by_chain = {}
                    if self.engine and self.engine.active_positions:
                        for pos_id, position in self.engine.active_positions.items():
                            chain = position.get('chain', 'unknown').upper()
                            if chain not in positions_by_chain:
                                positions_by_chain[chain] = []
                            positions_by_chain[chain].append(position)
                    
                    balances = {}
                    for chain in ['ETHEREUM', 'BSC', 'BASE', 'SOLANA']:
                        chain_positions = positions_by_chain.get(chain, [])
                        
                        # Calculate position value for this chain
                        chain_position_value = 0
                        chain_position_cost = 0
                        
                        for pos in chain_positions:
                            entry_price = float(pos.get('entry_price', 0))
                            current_price = float(pos.get('current_price', entry_price))
                            amount = float(pos.get('amount', 0))
                            
                            pos_cost = entry_price * amount
                            pos_value = current_price * amount
                            
                            chain_position_value += pos_value
                            chain_position_cost += pos_cost
                        
                        # Calculate unrealized P&L for this chain
                        chain_unrealized_pnl = chain_position_value - chain_position_cost
                        chain_pnl_pct = (chain_unrealized_pnl / chain_position_cost * 100) if chain_position_cost > 0 else 0
                        
                        # Each chain gets equal allocation of total portfolio
                        chain_allocated = total_portfolio_value / 4.0
                        
                        balances[chain] = {
                            'balance': float(chain_allocated),  # Total allocated to this chain
                            'in_positions': float(chain_position_value),
                            'available': float(chain_allocated - chain_position_value) if chain_position_value < chain_allocated else 0,
                            'pnl': float(chain_unrealized_pnl),
                            'pnl_pct': float(chain_pnl_pct),
                            'positions': len(chain_positions)
                        }
                    
                    # Broadcast wallet update
                    await self.sio.emit('wallet_update', {
                        'balances': balances,
                        'total_portfolio': float(total_portfolio_value),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"Error broadcasting wallet update: {e}")
                
                # ✅ FIX: Add await for async method
                if self.db:
                    try:
                        perf_data = await self.db.get_performance_summary()
                        if 'error' not in perf_data:  # Only broadcast if no error
                            await self.sio.emit('performance_update', {
                                **perf_data,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                    except Exception as e:
                        logger.debug(f"Error broadcasting performance update: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def _generate_report(self, period, start_date, end_date, metrics):
        """Generate detailed performance report."""
        report = {
            'period': period,
            'start_date': start_date,
            'end_date': end_date,
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': {},
            'trades': []
        }

        if not self.db:
            return report

        # --- FIX STARTS HERE ---
        # 1. Determine date range based on period
        now = datetime.utcnow()
        if period == 'daily':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif period == 'weekly':
            start_date = now - timedelta(days=7)
            end_date = now
        elif period == 'monthly':
            start_date = now - timedelta(days=30)
            end_date = now
        elif start_date and end_date:
            # Use provided dates for custom reports
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59)
        else:
            # Default to last 7 days if no period is matched
            start_date = now - timedelta(days=7)
            end_date = now

        # 2. Fetch trades from the database within the calculated date range
        # Filtering on exit_timestamp for closed trades makes more sense for reports
        query = """
            SELECT * FROM trades
            WHERE status = 'closed' AND exit_timestamp >= $1 AND exit_timestamp <= $2
            ORDER BY exit_timestamp ASC;
        """
        closed_trades = await self.db.pool.fetch(query, start_date, end_date)

        # Calculate metrics if there are any closed trades
        if not closed_trades:
            return report

        # Process trades to extract metadata fields and calculate ROI
        trades_list = []
        for idx, trade in enumerate(closed_trades):
            trade_dict = dict(trade)

            # Extract and parse metadata (JSONB column)
            metadata = trade_dict.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse metadata for trade {trade_dict.get('id')}: {e}")
                    metadata = {}

            # Debug: Log first trade to see structure
            if idx == 0:
                logger.info(f"Sample trade structure - Available fields: {list(trade_dict.keys())}")
                logger.info(f"Sample metadata structure: {metadata}")

            # Extract token_symbol with multiple fallbacks
            token_symbol = (
                metadata.get('token_symbol') or
                metadata.get('token') or
                trade_dict.get('token_symbol') or
                trade_dict.get('token') or
                trade_dict.get('token_address', 'Unknown')[:10]  # Use first 10 chars of address if nothing else
            )
            trade_dict['token_symbol'] = token_symbol

            # Extract exit_reason with multiple fallbacks
            exit_reason = (
                metadata.get('exit_reason') or
                metadata.get('reason') or
                trade_dict.get('exit_reason') or
                'Manual'  # Default if not specified
            )
            trade_dict['exit_reason'] = exit_reason

            # Extract strategy with fallbacks
            strategy = metadata.get('strategy', 'N/A')
            if isinstance(strategy, dict):
                strategy = strategy.get('name', 'N/A')
            trade_dict['strategy'] = strategy

            # Extract chain
            trade_dict['chain'] = metadata.get('chain', trade_dict.get('chain', 'N/A'))

            # Calculate ROI percentage
            profit_loss = float(trade_dict.get('profit_loss', 0) or 0)
            entry_value = float(trade_dict.get('entry_value', 0) or 0)

            # If entry_value is 0, try to calculate from entry_price * amount
            if entry_value == 0:
                entry_price = float(trade_dict.get('entry_price', 0) or 0)
                amount = float(trade_dict.get('amount', 0) or 0)
                entry_value = entry_price * amount

            if entry_value > 0:
                roi = (profit_loss / entry_value) * 100
                trade_dict['roi'] = round(roi, 4)
            else:
                trade_dict['roi'] = 0

            trades_list.append(trade_dict)

        # The report should only contain closed trades with enriched data
        report['trades'] = self._serialize_decimals(trades_list)
        # --- FIX ENDS HERE ---

        # Convert records to a list of dicts for DataFrame creation
        trade_list = [dict(row) for row in closed_trades]
        df = pd.DataFrame(trade_list)

        # Ensure 'profit_loss' column exists and handle potential missing values
        if 'profit_loss' not in df.columns:
            df['profit_loss'] = 0
        else:
            df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0)

        # --- FIX STARTS HERE: Calculate all missing report metrics ---
        winning_trades_df = df[df['profit_loss'] > 0]
        losing_trades_df = df[df['profit_loss'] <= 0]

        total_pnl = df['profit_loss'].sum()
        total_trades = len(df)
        win_rate = (len(winning_trades_df) / total_trades) * 100 if total_trades > 0 else 0

        avg_win = winning_trades_df['profit_loss'].mean() if not winning_trades_df.empty else 0
        avg_loss = abs(losing_trades_df['profit_loss'].mean()) if not losing_trades_df.empty else 0

        gross_profit = winning_trades_df['profit_loss'].sum()
        gross_loss = abs(losing_trades_df['profit_loss'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (9999.99 if gross_profit > 0 else 0.0)

        initial_balance = self.config_mgr.get_portfolio_config().initial_balance
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['equity'] = initial_balance + df['cumulative_pnl']
        peak = df['equity'].expanding(min_periods=1).max()
        drawdown = ((df['equity'] - peak) / peak).replace([np.inf, -np.inf], 0).fillna(0)
        max_drawdown = abs(drawdown.min() * 100) if not drawdown.empty else 0

        # Populate metrics dictionary
        report['metrics'] = {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'winning_trades': len(winning_trades_df),
            'losing_trades': len(losing_trades_df),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'recovery_factor': 0 # Placeholder for now
        }
        # --- FIX ENDS HERE ---

        return report
    
    async def _generate_custom_report(self, filters):
        """Generate custom report with filters"""
        # Implement custom filtering logic
        report = {
            'filters': filters,
            'generated_at': datetime.utcnow().isoformat(),
            'data': []
        }
        
        return report
    
    async def _export_csv(self, report):
        """Export report as CSV"""
        output = io.StringIO()
        writer = csv.writer(output)

        # --- FIX STARTS HERE: Write both metrics and trade data ---
        # Write metrics
        writer.writerow(['Metric', 'Value'])
        if 'metrics' in report:
            for key, value in report['metrics'].items():
                # Format numbers for better readability in CSV
                if isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([key.replace('_', ' ').title(), value])

        writer.writerow([]) # Add a blank line for separation

        # Write comprehensive trade data with all available fields
        writer.writerow([
            'ID', 'Token Symbol', 'Token Address', 'Chain', 'Strategy',
            'Entry Timestamp', 'Exit Timestamp', 'Entry Price', 'Exit Price',
            'Amount', 'Entry Value', 'Exit Value', 'Profit/Loss', 'ROI (%)',
            'Exit Reason', 'Status', 'Gas Cost'
        ])

        if 'trades' in report and report['trades']:
            for trade in report['trades']:
                writer.writerow([
                    trade.get('id'),
                    trade.get('token_symbol', 'N/A'),
                    trade.get('token_address', trade.get('token', 'N/A')),
                    trade.get('chain', 'N/A'),
                    trade.get('strategy', 'N/A'),
                    trade.get('entry_timestamp'),
                    trade.get('exit_timestamp'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('amount'),
                    trade.get('entry_value'),
                    trade.get('exit_value'),
                    trade.get('profit_loss'),
                    trade.get('roi', 0),
                    trade.get('exit_reason', 'N/A'),
                    trade.get('status', 'closed'),
                    trade.get('gas_cost', 0)
                ])
        else:
            writer.writerow(['No trade data available for this period.'])
        # --- FIX ENDS HERE ---
        
        response = web.Response(
            body=output.getvalue().encode('utf-8'),
            content_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv"'
            }
        )
        return response
    
    async def _export_excel(self, report):
        """Export comprehensive Excel report with multiple sheets and formatting"""
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.chart import BarChart, LineChart, Reference
        from openpyxl.cell import MergedCell

        output = io.BytesIO()
        wb = Workbook()

        # Remove default sheet
        if 'Sheet' in wb.sheetnames:
            del wb['Sheet']

        # ==================== SHEET 1: SUMMARY ====================
        ws_summary = wb.create_sheet('Summary', 0)

        # Title
        ws_summary['A1'] = 'Trading Performance Report'
        ws_summary['A1'].font = Font(size=16, bold=True, color='FFFFFF')
        ws_summary['A1'].fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
        ws_summary.merge_cells('A1:D1')

        # Report info
        ws_summary['A2'] = 'Generated:'
        ws_summary['B2'] = report.get('generated_at', datetime.utcnow().isoformat())
        ws_summary['A3'] = 'Period:'
        ws_summary['B3'] = report.get('period', 'custom')

        # Performance Metrics Header
        ws_summary['A5'] = 'Performance Metrics'
        ws_summary['A5'].font = Font(size=14, bold=True, color='FFFFFF')
        ws_summary['A5'].fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        ws_summary.merge_cells('A5:D5')

        # Metrics table
        row = 6
        if 'metrics' in report:
            metrics = report['metrics']

            # Header
            ws_summary['A6'] = 'Metric'
            ws_summary['B6'] = 'Value'
            ws_summary['A6'].font = Font(bold=True)
            ws_summary['B6'].font = Font(bold=True)
            ws_summary['A6'].fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
            ws_summary['B6'].fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')

            row = 7
            for key, value in metrics.items():
                ws_summary[f'A{row}'] = key.replace('_', ' ').title()

                # Format value based on metric type
                if isinstance(value, float):
                    if 'rate' in key.lower() or 'drawdown' in key.lower():
                        ws_summary[f'B{row}'] = f"{value:.2f}%"
                    else:
                        ws_summary[f'B{row}'] = f"{value:.4f}"
                else:
                    ws_summary[f'B{row}'] = value

                # Color code based on performance
                if 'pnl' in key.lower() or 'profit' in key.lower():
                    if isinstance(value, (int, float)) and value > 0:
                        ws_summary[f'B{row}'].font = Font(color='00B050', bold=True)
                    elif isinstance(value, (int, float)) and value < 0:
                        ws_summary[f'B{row}'].font = Font(color='FF0000', bold=True)

                row += 1

        # Adjust column widths
        ws_summary.column_dimensions['A'].width = 25
        ws_summary.column_dimensions['B'].width = 20

        # ==================== SHEET 2: DETAILED TRADES ====================
        ws_trades = wb.create_sheet('Detailed Trades', 1)

        if 'trades' in report and report['trades']:
            # Convert trades to DataFrame
            df_trades = pd.DataFrame(report['trades'])

            # Select and order columns
            columns = [
                'id', 'token_symbol', 'token_address', 'chain', 'strategy',
                'entry_timestamp', 'exit_timestamp', 'entry_price', 'exit_price',
                'amount', 'entry_value', 'exit_value', 'profit_loss', 'roi',
                'exit_reason', 'status', 'gas_cost'
            ]

            # Only include columns that exist
            columns = [col for col in columns if col in df_trades.columns]
            df_trades = df_trades[columns]

            # Rename columns for better readability
            column_names = {
                'id': 'ID',
                'token_symbol': 'Token',
                'token_address': 'Address',
                'chain': 'Chain',
                'strategy': 'Strategy',
                'entry_timestamp': 'Entry Time',
                'exit_timestamp': 'Exit Time',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'amount': 'Amount',
                'entry_value': 'Entry Value',
                'exit_value': 'Exit Value',
                'profit_loss': 'P&L',
                'roi': 'ROI %',
                'exit_reason': 'Exit Reason',
                'status': 'Status',
                'gas_cost': 'Gas Cost'
            }
            df_trades = df_trades.rename(columns=column_names)

            # Write header
            for col_num, column_name in enumerate(df_trades.columns, 1):
                cell = ws_trades.cell(row=1, column=col_num, value=column_name)
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                cell.alignment = Alignment(horizontal='center')

            # Write data
            for row_num, row_data in enumerate(df_trades.values, 2):
                for col_num, value in enumerate(row_data, 1):
                    cell = ws_trades.cell(row=row_num, column=col_num, value=value)

                    # Color code P&L and ROI
                    if df_trades.columns[col_num-1] in ['P&L', 'ROI %']:
                        try:
                            val = float(value) if value else 0
                            if val > 0:
                                cell.font = Font(color='00B050', bold=True)
                                cell.fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')
                            elif val < 0:
                                cell.font = Font(color='FF0000', bold=True)
                                cell.fill = PatternFill(start_color='FCE4D6', end_color='FCE4D6', fill_type='solid')
                        except:
                            pass

            # Auto-adjust column widths (skip merged cells)
            for col in ws_trades.iter_cols():
                if col and not isinstance(col[0], MergedCell):
                    try:
                        length = max(len(str(cell.value or '')) for cell in col)
                        ws_trades.column_dimensions[col[0].column_letter].width = min(length + 2, 40)
                    except:
                        pass

        # ==================== SHEET 3: ANALYSIS BY STRATEGY ====================
        ws_strategy = wb.create_sheet('Strategy Analysis', 2)

        if 'trades' in report and report['trades']:
            df_all = pd.DataFrame(report['trades'])

            if 'strategy' in df_all.columns and 'profit_loss' in df_all.columns:
                # Group by strategy
                strategy_analysis = df_all.groupby('strategy').agg({
                    'profit_loss': ['count', 'sum', 'mean', lambda x: (x > 0).sum(), lambda x: (x <= 0).sum()]
                }).round(4)

                strategy_analysis.columns = ['Total Trades', 'Total P&L', 'Avg P&L', 'Wins', 'Losses']
                strategy_analysis['Win Rate %'] = (strategy_analysis['Wins'] / strategy_analysis['Total Trades'] * 100).round(2)
                strategy_analysis = strategy_analysis.reset_index()

                # Write header
                ws_strategy['A1'] = 'Strategy Performance Analysis'
                ws_strategy['A1'].font = Font(size=14, bold=True, color='FFFFFF')
                ws_strategy['A1'].fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                ws_strategy.merge_cells('A1:G1')

                # Write data
                for col_num, column_name in enumerate(strategy_analysis.columns, 1):
                    cell = ws_strategy.cell(row=2, column=col_num, value=column_name)
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill(start_color='70AD47', end_color='70AD47', fill_type='solid')

                for row_num, row_data in enumerate(strategy_analysis.values, 3):
                    for col_num, value in enumerate(row_data, 1):
                        ws_strategy.cell(row=row_num, column=col_num, value=value)

                # Auto-adjust widths (skip merged cells)
                for col in ws_strategy.iter_cols():
                    if col and not isinstance(col[0], MergedCell):
                        try:
                            max_length = max(len(str(cell.value or '')) for cell in col)
                            ws_strategy.column_dimensions[col[0].column_letter].width = max_length + 2
                        except:
                            pass

        # ==================== SHEET 4: ANALYSIS BY TOKEN ====================
        ws_token = wb.create_sheet('Token Analysis', 3)

        if 'trades' in report and report['trades']:
            df_all = pd.DataFrame(report['trades'])

            if 'token_symbol' in df_all.columns and 'profit_loss' in df_all.columns:
                # Group by token
                token_analysis = df_all.groupby('token_symbol').agg({
                    'profit_loss': ['count', 'sum', 'mean', lambda x: (x > 0).sum()]
                }).round(4)

                token_analysis.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Wins']
                token_analysis['Win Rate %'] = (token_analysis['Wins'] / token_analysis['Trades'] * 100).round(2)
                token_analysis = token_analysis.sort_values('Total P&L', ascending=False).reset_index()

                # Write header
                ws_token['A1'] = 'Token Performance Analysis'
                ws_token['A1'].font = Font(size=14, bold=True, color='FFFFFF')
                ws_token['A1'].fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                ws_token.merge_cells('A1:F1')

                # Write data
                for col_num, column_name in enumerate(token_analysis.columns, 1):
                    cell = ws_token.cell(row=2, column=col_num, value=column_name)
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill(start_color='FFC000', end_color='FFC000', fill_type='solid')

                for row_num, row_data in enumerate(token_analysis.values, 3):
                    for col_num, value in enumerate(row_data, 1):
                        cell = ws_token.cell(row=row_num, column=col_num, value=value)

                        # Highlight top/bottom performers
                        if token_analysis.columns[col_num-1] == 'Total P&L':
                            try:
                                val = float(value) if value else 0
                                if val > 0:
                                    cell.font = Font(color='00B050', bold=True)
                                elif val < 0:
                                    cell.font = Font(color='FF0000', bold=True)
                            except:
                                pass

                # Auto-adjust widths (skip merged cells)
                for col in ws_token.iter_cols():
                    if col and not isinstance(col[0], MergedCell):
                        try:
                            max_length = max(len(str(cell.value or '')) for cell in col)
                            ws_token.column_dimensions[col[0].column_letter].width = max_length + 2
                        except:
                            pass

        # Save workbook to BytesIO
        wb.save(output)
        output.seek(0)

        response = web.Response(
            body=output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename="trading_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.xlsx"'
            }
        )
        return response
    
    async def _export_pdf(self, report):
        """Export report as PDF"""
        # Implement PDF generation (using reportlab or similar)
        # For now, return JSON
        return web.json_response(report)
    
    async def _run_backtest_task(self, test_id, strategy, start_date, end_date, initial_balance, parameters):
        """Run backtest in the background using historical data."""
        try:
            logger.info(f"Starting backtest {test_id} from {start_date} to {end_date}")
            self.backtests[test_id]['progress'] = 'Fetching historical data...'

            if not self.db:
                raise Exception("Database connection is not available.")

            # Fetch historical trades from the database within the specified date range
            # Filter trades by the selected strategy
            query = """
                SELECT * FROM trades
                WHERE status = 'closed'
                AND exit_timestamp >= $1
                AND exit_timestamp <= $2
                AND (
                    (metadata->'strategy'->>'name' = $3) OR
                    (jsonb_typeof(metadata->'strategy') = 'string' AND metadata->>'strategy' = $3)
                )
                ORDER BY exit_timestamp ASC;
            """
            trades = await self.db.pool.fetch(
                query,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date),
                strategy,
            )

            if not trades:
                self.backtests[test_id] = {
                    'status': 'completed',
                    'message': 'No trades found for the selected period.',
                    'equity_curve': []
                }
                return

            self.backtests[test_id]['progress'] = f'Simulating {len(trades)} trades.'
            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # Simulate trades instead of just replaying old P&L
            balance = float(initial_balance)
            equity_curve = [{'timestamp': start_date, 'value': balance}]
            position_size_per_trade = balance * 0.1  # 10% of initial balance per trade

            for index, trade in df.iterrows():
                entry_price = float(trade.get('entry_price', 0))
                exit_price = float(trade.get('exit_price', 0))
                roi = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                simulated_pnl = position_size_per_trade * roi
                balance += simulated_pnl
                df.at[index, 'simulated_pnl'] = simulated_pnl
                equity_curve.append({'timestamp': trade['exit_timestamp'].isoformat(), 'value': balance})

            # Use simulated P&L for all metrics
            df['profit_loss'] = df['simulated_pnl']

            # Final metrics
            final_balance = balance
            total_pnl = final_balance - float(initial_balance)
            total_return_pct = (total_pnl / float(initial_balance)) * 100 if initial_balance > 0 else 0

            total_trades = len(df)
            winning_trades_df = df[df['profit_loss'] > 0]
            losing_trades_df = df[df['profit_loss'] <= 0]
            win_rate = (len(winning_trades_df) / total_trades) * 100 if total_trades > 0 else 0.0

            # Max Drawdown from equity curve
            equity_df = pd.DataFrame(equity_curve)
            equity_df['value'] = pd.to_numeric(equity_df['value'])
            peak = equity_df['value'].expanding(min_periods=1).max()
            drawdown = ((equity_df['value'] - peak) / peak).replace([np.inf, -np.inf], 0).fillna(0)
            max_drawdown = abs(float(drawdown.min()) * 100) if not drawdown.empty else 0.0

            # Backtesting statistics
            gross_profit = float(winning_trades_df['profit_loss'].sum())
            gross_loss = abs(float(losing_trades_df['profit_loss'].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (9999.99 if gross_profit > 0 else 0.0)

            avg_win = float(winning_trades_df['profit_loss'].mean()) if not winning_trades_df.empty else 0.0
            avg_loss = abs(float(losing_trades_df['profit_loss'].mean())) if not losing_trades_df.empty else 0.0

            largest_win = float(winning_trades_df['profit_loss'].max()) if not winning_trades_df.empty else 0.0
            largest_loss = abs(float(losing_trades_df['profit_loss'].min())) if not losing_trades_df.empty else 0.0

            self.backtests[test_id] = {
                'status': 'completed',
                'final_balance': float(final_balance),
                'total_pnl': float(total_pnl),
                'profit_factor': float(profit_factor),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'largest_win': float(largest_win),
                'largest_loss': float(largest_loss),
                'total_return': float(total_return_pct),
                'total_trades': total_trades,
                'win_rate': float(win_rate),
                'winning_trades': len(winning_trades_df),
                'losing_trades': len(losing_trades_df),
                'sharpe_ratio': 0,  # placeholder
                'sortino_ratio': 0,  # placeholder
                'max_drawdown': max_drawdown,
                'equity_curve': equity_curve
            }
            logger.info(f"Backtest {test_id} completed successfully.")
        except Exception as e:
            logger.error(f"Error in backtest task {test_id}: {e}", exc_info=True)
            self.backtests[test_id] = {'status': 'failed', 'error': str(e)}
    
    async def start(self):
        """Start the dashboard server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Enhanced dashboard running on http://{self.host}:{self.port}")
        
        # Keep running
        await asyncio.Event().wait()