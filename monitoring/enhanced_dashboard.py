"""
Enhanced Dashboard for DexScreener Trading Bot
Professional web-based monitoring, control, and analytics interface
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
import io
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
                 db_manager = None):
        
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
        
        # Web application
        self.app = web.Application()
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        self.sio.attach(self.app)
        
        # Template engine
        self.jinja_env = Environment(
            loader=FileSystemLoader('dashboard/templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
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
    
    def _setup_routes(self):
        """Setup all routes"""

        # ✅ ADD: Add error handling middleware FIRST
        self.app.middlewares.append(self.error_handler_middleware)
        
        # Static files
        self.app.router.add_static('/static', 'dashboard/static', name='static')
        
        # Pages
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
        
        # API - Data endpoints
        self.app.router.add_get('/api/dashboard/summary', self.api_dashboard_summary)
        self.app.router.add_get('/api/logs', self.api_get_logs)
        self.app.router.add_get('/api/analysis', self.api_get_analysis)
        self.app.router.add_get('/api/insights', self.api_get_insights)
        self.app.router.add_get('/api/trades/recent', self.api_recent_trades)
        self.app.router.add_get('/api/trades/history', self.api_trade_history)
        self.app.router.add_get('/api/positions/open', self.api_open_positions)
        self.app.router.add_get('/api/positions/history', self.api_positions_history)
        self.app.router.add_get('/api/performance/metrics', self.api_performance_metrics)
        self.app.router.add_get('/api/performance/charts', self.api_performance_charts)
        self.app.router.add_get('/api/alerts/recent', self.api_recent_alerts)
        self.app.router.add_get('/api/risk/metrics', self.api_risk_metrics)
        self.app.router.add_get('/api/wallets/balances', self.api_wallet_balances)
        
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
        """Index page - redirect to dashboard"""
        raise web.HTTPFound('/dashboard')
    
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
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # --- FIX STARTS HERE: Use centralized strategy parsing ---
            df['strategy'] = df['metadata'].apply(self._get_strategy_from_metadata)
            strategy_performance = df.groupby('strategy')['profit_loss'].sum().reset_index()
            strategy_performance.columns = ['strategy', 'total_pnl']
            # --- FIX ENDS HERE ---

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
        """Generate performance tuning suggestions"""
        try:
            if not self.db:
                return web.json_response({'error': 'Database connection not available.'}, status=503)

            query = "SELECT * FROM trades WHERE status = 'closed';"
            trades = await self.db.pool.fetch(query)

            insights = []
            if not trades:
                # Return a friendly info message instead of an empty list
                insights.append({
                    'type': 'info',
                    'title': 'No closed trades yet',
                    'message': 'Once you have some closed trades, performance insights will appear here.'
                })
                return web.json_response({'success': True, 'data': insights})

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])

            # Insight 1: Win rate analysis
            win_rate = (df['profit_loss'] > 0).mean()
            if win_rate < 0.5:
                insights.append({
                    'type': 'suggestion',
                    'title': 'Improve Win Rate',
                    'message': 'Your win rate is below 50%. Consider tightening entry criteria or adjusting your stop-loss strategy to be more conservative.'
                })

            # Insight 2: Risk/Reward Ratio
            avg_win = df[df['profit_loss'] > 0]['profit_loss'].mean()
            avg_loss = abs(df[df['profit_loss'] <= 0]['profit_loss'].mean())
            if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss > 0 and (avg_win / avg_loss < 1.5):
                insights.append({
                    'type': 'suggestion',
                    'title': 'Increase Risk/Reward Ratio',
                    'message': 'Your average win is less than 1.5x your average loss. Aim for a higher risk/reward ratio by adjusting take-profit levels or seeking trades with better potential.'
                })

            # Insight 3: Identify problematic strategies
            df['strategy'] = df['metadata'].apply(self._get_strategy_from_metadata)
            strategy_pnl = df.groupby('strategy')['profit_loss'].sum()
            losing_strategies = strategy_pnl[strategy_pnl < 0]
            if not losing_strategies.empty:
                for strategy, pnl in losing_strategies.items():
                    insights.append({
                        'type': 'warning',
                        'title': f'Review Strategy: {strategy}',
                        'message': f'The "{strategy}" strategy has a total P&L of {pnl:.2f}. It may require tuning or deactivation.'
                    })

            return web.json_response({'success': True, 'data': self._serialize_decimals(insights)})

        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def api_dashboard_summary(self, request):
        """Get dashboard summary data"""
        try:
            summary = {}

            # Get portfolio data if available
            if self.portfolio:
                # Get portfolio summary
                portfolio_summary = self.engine.portfolio_manager.get_portfolio_summary()

                # Get config with fallback
                try:
                    from config.config_manager import PortfolioConfig
                    config = PortfolioConfig()
                    default_balance = config.initial_balance
                except Exception as e:
                    logger.warning(f"Could not load PortfolioConfig: {e}, using default 400")
                    default_balance = 400

                summary = {
                    'cash_balance': float(portfolio_summary.get('cash_balance', default_balance)),
                    'positions_value': float(portfolio_summary.get('positions_value', 0)),
                    'daily_pnl': float(portfolio_summary.get('daily_pnl', 0)),
                    'sharpe_ratio': float(portfolio_summary.get('sharpe_ratio', 0)),
                    'max_drawdown': float(portfolio_summary.get('max_drawdown', 0)),
                }

            # ✅ Get ACTUAL cumulative P&L and metrics from database
            total_pnl = 0
            win_rate = 0
            open_positions_count = 0

            if self.db:
                try:
                    perf_data = await self.db.get_performance_summary()
                    if perf_data and 'total_pnl' in perf_data:
                        total_pnl = float(perf_data.get('total_pnl', 0))
                        win_rate = float(perf_data.get('win_rate', 0))
                except Exception as e:
                    logger.error(f"Error getting performance data: {e}")

            # Get actual open positions count from engine
            if self.engine and hasattr(self.engine, 'active_positions'):
                open_positions_count = len(self.engine.active_positions)

            # Calculate portfolio value: starting balance + cumulative P&L
            try:
                config = PortfolioConfig()
                starting_balance = config.initial_balance
            except Exception:
                starting_balance = 400
            portfolio_value = starting_balance + total_pnl

            # Update summary with calculated values
            summary.update({
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'total_value': portfolio_value,
                'net_profit': total_pnl,
                'open_positions': open_positions_count,
                'win_rate': win_rate,  # ✅ Now includes actual win rate
                'pending_orders': len(self.orders.active_orders) if self.orders else 0,
                'active_alerts': len(self.alerts.alerts_queue._queue) if self.alerts else 0
            })

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
                
                enriched_trade['token_symbol'] = self._get_token_symbol_from_metadata(trade)
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
                    
                    token_symbol = self._get_token_symbol_from_metadata(trade)
                    
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
    # FIND the api_wallet_balances method you added (around line 650-750)
    # REPLACE THE ENTIRE METHOD with this corrected version:
    # ============================================================================

    async def api_wallet_balances(self, request):
        """Get wallet balances for all chains"""
        try:
            balances = {}
            total_value = 0
            total_pnl = 0
            
            # Get positions from engine.active_positions (portfolio.positions is empty)
            if self.engine and self.engine.active_positions:
                # Aggregate positions by chain
                positions_by_chain = {}
                for pos_id, position in self.engine.active_positions.items():
                    chain = position.get('chain', 'unknown').upper()
                    if chain not in positions_by_chain:
                        positions_by_chain[chain] = []
                    positions_by_chain[chain].append(position)
                
                # Calculate metrics for each chain
                for chain, positions in positions_by_chain.items():
                    chain_value = 0
                    chain_cost = 0
                    
                    for pos in positions:
                        # Use correct field names from engine.py
                        entry_price = float(pos.get('entry_price', 0))
                        current_price = float(pos.get('current_price', entry_price))
                        # Field is 'amount', not 'position_size'
                        amount = float(pos.get('amount', 0))
                        
                        pos_cost = entry_price * amount
                        pos_value = current_price * amount
                        
                        chain_value += pos_value
                        chain_cost += pos_cost
                    
                    chain_pnl = chain_value - chain_cost
                    chain_pnl_pct = (chain_pnl / chain_cost * 100) if chain_cost > 0 else 0
                    
                    balances[chain] = {
                        'balance': float(chain_value),
                        'pnl': float(chain_pnl),
                        'pnl_pct': float(chain_pnl_pct),
                        'positions': len(positions),
                        'available': float(chain_value),
                        'locked': 0.0
                    }
                    
                    total_value += chain_value
                    total_pnl += chain_pnl
                
                # Calculate overall stats
                overall_pnl_pct = (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) > 0 else 0
                
                # Add overall total
                balances['TOTAL'] = {
                    'balance': float(total_value),
                    'pnl': float(total_pnl),
                    'pnl_pct': float(overall_pnl_pct),
                    'positions': len(self.engine.active_positions)
                }
            else:
                # No positions - show initial balances
                initial_per_chain = 100.0
                balances = {
                    'ETHEREUM': {'balance': initial_per_chain, 'pnl': 0.0, 'pnl_pct': 0.0, 'positions': 0},
                    'BSC': {'balance': initial_per_chain, 'pnl': 0.0, 'pnl_pct': 0.0, 'positions': 0},
                    'BASE': {'balance': initial_per_chain, 'pnl': 0.0, 'pnl_pct': 0.0, 'positions': 0},
                    'SOLANA': {'balance': initial_per_chain, 'pnl': 0.0, 'pnl_pct': 0.0, 'positions': 0},
                    'TOTAL': {'balance': 400.0, 'pnl': 0.0, 'pnl_pct': 0.0, 'positions': 0}
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
                return web.json_response({'success': True, 'data': {'historical': {}}})

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
            profit_factor = sum_of_wins / sum_of_losses if sum_of_losses > 0 else float('inf')

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
        """Get performance chart data for dashboard & performance pages."""
        try:
            timeframe = request.query.get('timeframe', '7d')

            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)

            # 1) Load ALL closed trades
            query = """
                SELECT exit_timestamp, profit_loss, metadata
                FROM trades
                WHERE status = 'closed'
                ORDER BY exit_timestamp ASC;
            """
            trades = await self.db.pool.fetch(query)

            if not trades:
                return web.json_response({
                    'success': True,
                    'data': {
                        'equity_curve': [], 'cumulative_pnl': [],
                        'strategy_performance': [], 'win_loss': {'wins': 0, 'losses': 0},
                        'monthly': []
                    }
                })

            df = pd.DataFrame([dict(trade) for trade in trades])
            df['profit_loss'] = pd.to_numeric(df['profit_loss'])
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], utc=True).dt.tz_convert(None)
            df['strategy'] = df['metadata'].apply(self._get_strategy_from_metadata)

            # 2) Calculate cumulative data on the FULL history first
            df = df.sort_values('exit_timestamp').reset_index(drop=True)
            initial_balance = self.config_mgr.get_portfolio_config().initial_balance
            df['cumulative_pnl'] = df['profit_loss'].cumsum()
            df['equity'] = initial_balance + df['cumulative_pnl']

            # 3) Now, filter by the requested timeframe
            now = datetime.utcnow()
            start_time = None
            if timeframe == '1h':
                start_time = now - timedelta(hours=1)
            elif timeframe == '24h':
                start_time = now - timedelta(days=1)
            elif timeframe == '7d':
                start_time = now - timedelta(days=7)
            elif timeframe == '30d':
                start_time = now - timedelta(days=30)

            filtered_df = df.copy()
            if start_time:
                # Find the last data point *before* the window starts to anchor the chart
                anchor_row = df[df['exit_timestamp'] < start_time].tail(1)
                # Filter the main data to be within the window
                main_df = df[df['exit_timestamp'] >= start_time]

                # If there's data before the window, combine it with the window's data
                if not anchor_row.empty:
                    filtered_df = pd.concat([anchor_row, main_df]).reset_index(drop=True)
                else:
                    filtered_df = main_df # No data before the window, just use what's inside
            else: # 'all' timeframe
                filtered_df = df


            # 4) Generate chart data from the correctly filtered dataframe
            equity_curve_data = []
            cumulative_pnl_data = []

            if not filtered_df.empty:
                equity_curve_data = [
                    {'timestamp': ts.isoformat(), 'value': float(val)}
                    for ts, val in zip(filtered_df['exit_timestamp'], filtered_df['equity'])
                ]
                cumulative_pnl_data = [
                    {'timestamp': ts.isoformat(), 'cumulative_pnl': float(val)}
                    for ts, val in zip(filtered_df['exit_timestamp'], filtered_df['cumulative_pnl'])
                ]

            # 5) Other stats (can be calculated from the full or filtered dataframe as needed)
            strategy_performance_df = df.groupby('strategy')['profit_loss'].sum().reset_index()
            strategy_performance_df.columns = ['strategy', 'pnl']

            # Use the time-filtered df for win/loss stats for the current period
            period_df = df[df['exit_timestamp'] >= start_time] if start_time else df
            wins = int((period_df['profit_loss'] > 0).sum())
            losses = int((period_df['profit_loss'] <= 0).sum())
            win_loss_distribution = {'wins': wins, 'losses': losses}

            monthly_performance = []
            if not df.empty:
                monthly_df = df.copy()
                monthly_df['month'] = monthly_df['exit_timestamp'].dt.to_period('M').astype(str)
                monthly_performance = (
                    monthly_df.groupby('month')['profit_loss'].sum()
                    .reset_index().rename(columns={'profit_loss': 'pnl'})
                    .to_dict('records')
                )

            payload = {
                'equity_curve': equity_curve_data,
                'cumulative_pnl': cumulative_pnl_data,
                'strategy_performance': strategy_performance_df.to_dict('records'),
                'win_loss': win_loss_distribution,
                'monthly': monthly_performance,
            }

            return web.json_response({
                'success': True,
                'data': self._serialize_decimals(payload),
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
        """Get all settings, correctly handling SecretStr."""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)

            # --- FIX STARTS HERE ---
            # Fetch all configuration models from the config manager
            all_configs = {
                'general': self.config_mgr.get_general_config(),
                'portfolio': self.config_mgr.get_portfolio_config(),
                'risk_management': self.config_mgr.get_risk_management_config(),
                'trading': self.config_mgr.get_trading_config(),
                'chain': self.config_mgr.get_chain_config(),
                'position_management': self.config_mgr.get_position_management_config(),
                'volatility': self.config_mgr.get_volatility_config(),
                'exit_strategy': self.config_mgr.get_exit_strategy_config(),
                'solana': self.config_mgr.get_solana_config(),
                'jupiter': self.config_mgr.get_jupiter_config(),
                'performance': self.config_mgr.get_performance_config(),
                'logging': self.config_mgr.get_logging_config(),
                'feature_flags': self.config_mgr.get_feature_flags_config(),
                'gas_price': self.config_mgr.get_gas_price_config(),
                'trading_limits': self.config_mgr.get_trading_limits_config(),
                'ml_models': self.config_mgr.get_ml_models_config(),
                'backtesting': self.config_mgr.get_backtesting_config(),
                'network': self.config_mgr.get_network_config(),
                'debug': self.config_mgr.get_debug_config(),
                'dashboard': self.config_mgr.get_dashboard_config(),
                'security': self.config_mgr.get_security_config(),
                'database': self.config_mgr.get_database_config(),
                'api': self.config_mgr.get_api_config(),
                'monitoring': self.config_mgr.get_monitoring_config(),
            }

            # Use aiohttp's json_response with a custom dumps function
            # This is the correct way to handle custom serialization in aiohttp
            return web.json_response(
                {'success': True, 'data': all_configs},
                dumps=lambda data: json.dumps(data, default=self._json_serializer)
            )
            # --- FIX ENDS HERE ---

        except Exception as e:
            logger.error(f"Error getting settings: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_update_settings(self, request):
        """Update settings"""
        try:
            data = await request.json()
            config_type = data.get('config_type')
            updates = data.get('updates', {})
            
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)
            
            # Apply updates dynamically
            self.config_mgr.update_config_internal(
                config_type=config_type,
                updates=updates,
                user='dashboard',
                reason='Dashboard update',
                persist=True
            )
            
            return web.json_response({
                'success': True,
                'message': f'Settings updated: {config_type}'
            })
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_revert_settings(self, request):
        """Revert settings to previous version"""
        return web.json_response({
            'success': False,
            'error': 'This feature is temporarily disabled.'
        }, status=503)

    async def api_settings_history(self, request):
        """Get settings change history"""
        return web.json_response({
            'success': True,
            'data': [],
            'count': 0,
            'message': 'This feature is temporarily disabled.'
        })
    
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
            format_type = request.match_info['format'].lower()
            # Normalize period so 'Daily', 'WEEKLY' etc. work correctly
            raw_period = request.query.get('period', 'custom') or 'custom'
            period = raw_period.lower()

            # These may be empty for daily/weekly/monthly and will be handled
            # inside _generate_report
            start_date_str = request.query.get('start_date')
            end_date_str = request.query.get('end_date')

            # Generate the report with the correct timeframe
            report = await self._generate_report(period, start_date_str, end_date_str, ['all'])

            # --- NEW FIX STARTS HERE: Unified and corrected export logic ---
            if format_type == 'csv':
                return await self._export_csv(report)
            elif format_type in ['xls', 'xlsx', 'excel']:
                return await self._export_excel(report)
            elif format_type == 'pdf':
                # We generate a .txt file as a placeholder for PDF
                return await self._export_pdf_as_txt(report)
            elif format_type == 'json':
                return web.json_response(
                    self._serialize_decimals(report),
                    headers={'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json"'}
                )
            else:
                return web.json_response({'error': f"Invalid format: {format_type}"}, status=400)
            # --- NEW FIX ENDS HERE ---
        except Exception as e:
            logger.error(f"Error exporting report: {e}", exc_info=True)
            return web.json_response({'error': 'An internal error occurred during report export.'}, status=500)
    
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
        default_name = "All Trades"

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

        # --- NEW FIX STARTS HERE: More robust search logic ---
        # Common keys for strategy name, in order of priority
        keys_to_check = ['strategy_name', 'strategy_id', 'strategy']

        for key in keys_to_check:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, str) and value:
                    return value
                if isinstance(value, dict):
                    # Check for a 'name' key inside a nested strategy object
                    name = value.get('name')
                    if isinstance(name, str) and name:
                        return name

        # Final fallback: recursive search for any key containing 'strategy'
        for key, value in metadata.items():
            if 'strategy' in key.lower() and isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                result = self._get_strategy_from_metadata(value)
                if result != default_name:
                    return result
        # --- NEW FIX ENDS HERE ---

        return default_name

    def _get_token_symbol_from_metadata(self, trade: Dict[str, Any]) -> str:
        """
        Robustly extract token symbol from a trade row.

        Works whether token_symbol is in:
        - trade['token_symbol']
        - trade['symbol']
        - JSON metadata (string or dict) under 'token_symbol' / 'symbol'
        """
        try:
            # 1) Direct column first (most reliable)
            direct = trade.get('token_symbol') or trade.get('symbol')
            if direct:
                return str(direct)

            # 2) Metadata field (can be str or dict)
            metadata = trade.get('metadata')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = None

            if isinstance(metadata, dict):
                for key in ['token_symbol', 'symbol', 'base_token', 'pair_symbol']:
                    if metadata.get(key):
                        return str(metadata[key])

            # 3) Fallback: if engine moved symbol to top-level under another name
            for key in ['base_symbol', 'pair', 'token']:
                if trade.get(key):
                    return str(trade[key])

        except Exception as e:
            logger.error(f"Error extracting token symbol from trade: {e}", exc_info=True)

        return 'N/A'


    def _get_exit_reason_from_trade(self, trade: Dict[str, Any]) -> str:
        """
        Robustly determine exit reason for a trade.

        Checks:
        - direct 'exit_reason' column
        - metadata['exit_reason'] / 'closing_reason' / 'reason'
        - heuristic based on P&L, tp/sl flags, etc.
        """
        try:
            # 1) Direct DB column if present
            direct = trade.get('exit_reason')
            if direct:
                return str(direct)

            # 2) Metadata (string or dict)
            metadata = trade.get('metadata')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = None

            if isinstance(metadata, dict):
                for key in ['exit_reason', 'closing_reason', 'reason']:
                    if metadata.get(key):
                        return str(metadata[key])

            # 3) Heuristic fallback based on P&L
            pnl = trade.get('profit_loss')
            try:
                pnl_val = float(pnl) if pnl is not None else 0.0
            except (TypeError, ValueError):
                pnl_val = 0.0

            if pnl_val > 0:
                return 'take_profit'
            elif pnl_val < 0:
                return 'stop_loss'

            return 'manual'

        except Exception as e:
            logger.error(f"Error determining exit reason: {e}", exc_info=True)
            return 'manual'



    def _get_exit_reason_from_metadata(self, trade_record: Dict[str, Any]) -> str:
        """Robustly extract the exit reason from a trade record."""
        # 1. Check for a direct key on the record (for older records)
        if 'exit_reason' in trade_record and trade_record['exit_reason'] not in [None, 'N/A']:
            return trade_record['exit_reason']

        # 2. Check the metadata field
        metadata = trade_record.get('metadata')
        if not metadata:
            return "N/A"

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return "N/A"

        if isinstance(metadata, dict):
            # Check for the new location of exit_reason
            return metadata.get('exit_reason', 'N/A')

        return "N/A"
    
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
    
    async def _generate_report(self, period, start_date_str, end_date_str, metrics):
        """Generate detailed performance report with correct timeframe filtering."""
        report = {
            'period': period,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': {},
            'trades': []
        }

        if not self.db:
            return report

        # --- NEW FIX STARTS HERE: Correct date range handling ---
        now = datetime.utcnow()
        period_normalized = (period or 'custom').lower()
        start_date, end_date = None, None

        def _parse_date_safe(s: str) -> Optional[datetime]:
            if not s:
                return None
            try:
                # Accept plain date ("2025-11-17") or full ISO with time
                return datetime.fromisoformat(s.replace('Z', '+00:00')) \
                    if 'T' in s else datetime.fromisoformat(s)
            except Exception:
                return None

        # Try to use UI-provided dates if they exist
        ui_start = _parse_date_safe(start_date_str) if start_date_str else None
        ui_end = _parse_date_safe(end_date_str) if end_date_str else None

        if period_normalized == 'daily':
            # If UI sends a specific day, use that; otherwise use "today"
            if ui_start:
                start_date = ui_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (ui_end or ui_start).replace(hour=23, minute=59, second=59)
            else:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = now

        elif period_normalized == 'weekly':
            # If UI gives a week start, use that; else "last 7 days"
            if ui_start and ui_end:
                start_date = ui_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = ui_end.replace(hour=23, minute=59, second=59)
            elif ui_start and not ui_end:
                start_date = ui_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = (ui_start + timedelta(days=7)).replace(
                    hour=23, minute=59, second=59
                )
            else:
                start_date = now - timedelta(days=7)
                end_date = now

        elif period_normalized == 'monthly':
            if ui_start and ui_end:
                start_date = ui_start.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = ui_end.replace(hour=23, minute=59, second=59)
            else:
                start_date = now - timedelta(days=30)
                end_date = now

        elif period_normalized == 'custom' and (ui_start and ui_end):
            start_date = ui_start.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = ui_end.replace(hour=23, minute=59, second=59)

        else:
            # Fallback: last 7 days
            start_date = now - timedelta(days=7)
            end_date = now

        report['start_date'] = start_date.isoformat()
        report['end_date'] = end_date.isoformat()

        query = """
            SELECT * FROM trades
            WHERE status = 'closed' AND exit_timestamp >= $1 AND exit_timestamp <= $2
            ORDER BY exit_timestamp ASC;
        """
        closed_trades = await self.db.pool.fetch(query, start_date, end_date)
        report['trades'] = self._serialize_decimals([dict(row) for row in closed_trades])
        # --- FIX ENDS HERE ---

        if not closed_trades:
            return report

        df = pd.DataFrame([dict(row) for row in closed_trades])
        if 'profit_loss' in df.columns:
            df['profit_loss'] = pd.to_numeric(df['profit_loss'], errors='coerce').fillna(0)
        else:
            df['profit_loss'] = 0

        winning_trades_df = df[df['profit_loss'] > 0]
        losing_trades_df = df[df['profit_loss'] <= 0]
        total_pnl = df['profit_loss'].sum()
        total_trades = len(df)
        win_rate = (len(winning_trades_df) / total_trades) * 100 if total_trades > 0 else 0
        avg_win = winning_trades_df['profit_loss'].mean() if not winning_trades_df.empty else 0
        avg_loss = abs(losing_trades_df['profit_loss'].mean()) if not losing_trades_df.empty else 0
        gross_profit = winning_trades_df['profit_loss'].sum()
        gross_loss = abs(losing_trades_df['profit_loss'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        initial_balance = self.config_mgr.get_portfolio_config().initial_balance
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['equity'] = initial_balance + df['cumulative_pnl']
        peak = df['equity'].expanding(min_periods=1).max()
        drawdown = ((df['equity'] - peak) / peak).fillna(0)
        max_drawdown = abs(drawdown.min() * 100) if not drawdown.empty else 0

        report['metrics'] = {
            'total_pnl': total_pnl, 'total_trades': total_trades, 'win_rate': win_rate,
            'winning_trades': len(winning_trades_df), 'losing_trades': len(losing_trades_df),
            'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor, 'max_drawdown': max_drawdown
        }

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
        if 'metrics' in report and report['metrics']:
            for key, value in report['metrics'].items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                writer.writerow([key.replace('_', ' ').title(), value])
        else:
            writer.writerow(['No metrics available.'])


        writer.writerow([])

        # Write trade data headers
        headers = [
            'ID', 'Token Symbol', 'Entry Timestamp', 'Exit Timestamp',
            'Entry Price', 'Exit Price', 'Amount', 'Profit/Loss', 'ROI (%)', 'Exit Reason'
        ]
        writer.writerow(headers)

        if 'trades' in report and report['trades']:
            for trade in report['trades']:
                writer.writerow([
                    trade.get('id'),
                    self._get_token_symbol_from_metadata(trade),
                    trade.get('entry_timestamp'),
                    trade.get('exit_timestamp'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('amount'),
                    trade.get('profit_loss'),
                    trade.get('roi', trade.get('profit_loss_percentage', 0)),
                    self._get_exit_reason_from_trade(trade) # Use the new helper
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
        """Export report as a multi-sheet Excel file."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Metrics Summary
            if 'metrics' in report and report['metrics']:
                metrics_df = pd.DataFrame(list(report['metrics'].items()), columns=['Metric', 'Value'])
                metrics_df.to_excel(writer, sheet_name='Summary', index=False)
            else:
                pd.DataFrame([{'Status': 'No metrics available'}]).to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Trade Log
            if 'trades' in report and report['trades']:
                trades_df = pd.DataFrame(report['trades'])
                # --- NEW FIX: Add token symbol and clean up data ---
                trades_df['token_symbol'] = trades_df.apply(self._get_token_symbol_from_metadata, axis=1)
                trades_df['exit_reason'] = trades_df.apply(self._get_exit_reason_from_trade, axis=1) # Use the new helper
                desired_columns = [
                    'id', 'token_symbol', 'entry_timestamp', 'exit_timestamp', 'entry_price',
                    'exit_price', 'amount', 'profit_loss', 'roi', 'exit_reason'
                ]
                existing_columns = [col for col in desired_columns if col in trades_df.columns]
                trades_df[existing_columns].to_excel(writer, sheet_name='Trade Log', index=False)
            else:
                pd.DataFrame([{'Status': 'No trade data available'}]).to_excel(writer, sheet_name='Trade Log', index=False)

        output.seek(0)
        return web.Response(
            body=output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.xlsx"'}
        )

    async def _export_pdf_as_txt(self, report):
        """Export report as a well-formatted text file."""
        output = io.StringIO()
        output.write("========================================\n")
        output.write("         Performance Report\n")
        output.write("========================================\n\n")
        output.write(f"Generated At: {report.get('generated_at', 'N/A')}\n")
        output.write(f"Period: {report.get('period', 'N/A').title()}\n")
        output.write(f"Date Range: {report.get('start_date', 'N/A')} to {report.get('end_date', 'N/A')}\n\n")

        output.write("---------- Metrics Summary ----------\n")
        if 'metrics' in report and report['metrics']:
            for key, value in report['metrics'].items():
                if isinstance(value, float): value = f"{value:.4f}"
                output.write(f"{key.replace('_', ' ').title():<20}: {value}\n")
        else:
            output.write("No metrics available.\n")

        output.write("\n\n---------- Trade Log ----------\n")
        if 'trades' in report and report['trades']:
            headers = f"{'ID':<8} {'Symbol':<12} {'P&L':<12} {'ROI (%)':<10} {'Exit Reason':<20}\n"
            output.write(headers)
            output.write("-" * len(headers) + "\n")
            for trade in report['trades']:
                symbol = self._get_token_symbol_from_metadata(trade)
                pnl = trade.get('profit_loss', 0)
                roi = trade.get('roi', trade.get('profit_loss_percentage', 0))
                exit_reason = self._get_exit_reason_from_trade(trade) # Use the new helper
                output.write(
                    f"{trade.get('id', ''):<8} {symbol:<12} {pnl:<12.4f} {roi:<10.2f} {str(exit_reason):<20}\n"
                )
        else:
            output.write("No trade data available.\n")

        return web.Response(
            body=output.getvalue().encode('utf-8'),
            content_type='text/plain',
            headers={'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.txt"'}
        )
    
    async def _run_backtest_task(self, test_id, strategy, start_date, end_date, initial_balance, parameters):
        """Run backtest in the background using historical data."""
        try:
            logger.info(f"Starting backtest {test_id} for strategy '{strategy}' from {start_date} to {end_date}")
            self.backtests[test_id]['progress'] = 'Fetching historical data...'

            if not self.db:
                raise Exception("Database connection is not available.")

            # --- NEW FIX STARTS HERE: A completely rewritten backtesting logic ---
            # Query all trades and filter in memory for maximum compatibility
            query = """
                SELECT * FROM trades
                WHERE status = 'closed' AND exit_timestamp >= $1 AND exit_timestamp <= $2
                ORDER BY exit_timestamp ASC;
            """
            all_trades = await self.db.pool.fetch(
                query,
                datetime.fromisoformat(start_date),
                datetime.fromisoformat(end_date)
            )

            # Filter trades by the selected strategy using the robust helper function
            # Filter trades by the selected strategy, but never leave the backtest empty.
            # If strategy is empty / "all", we just use all trades.
            if strategy and strategy.lower() not in ("all", "all_strategies"):
                strategy_trades = [
                    trade for trade in all_trades
                    if self._get_strategy_from_metadata(trade.get('metadata')) == strategy
                ]
            else:
                strategy_trades = list(all_trades)

            # If filtering yielded nothing, fall back to all trades
            if not strategy_trades:
                strategy_trades = list(all_trades)

            self.backtests[test_id]['progress'] = f'Simulating {len(strategy_trades)} trades...'
            df = pd.DataFrame([dict(trade) for trade in strategy_trades])

            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

            # Simulation logic
            balance = float(initial_balance)
            equity_curve = [{'timestamp': start_date, 'value': balance}]
            # Use a consistent position size for each trade to simulate a fixed-risk strategy
            position_size_per_trade = balance * 0.1  # Assume 10% of initial capital per trade

            for index, trade in df.iterrows():
                entry_price = float(trade.get('entry_price', 0))
                exit_price = float(trade.get('exit_price', 0))

                # Calculate the percentage return of the historical trade
                trade_return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

                # Apply this return to the simulated position size to get the simulated P&L
                simulated_pnl = position_size_per_trade * trade_return_pct

                # Update the balance and record the equity curve
                balance += simulated_pnl
                df.at[index, 'simulated_pnl'] = simulated_pnl
                equity_curve.append({'timestamp': trade['exit_timestamp'].isoformat(), 'value': balance})

            # Use the simulated P&L for all subsequent metric calculations
            df['profit_loss'] = df['simulated_pnl']

            # Calculate final metrics
            final_balance = balance
            total_pnl = final_balance - float(initial_balance)
            total_return_pct = (total_pnl / float(initial_balance)) * 100 if initial_balance > 0 else 0

            total_trades = len(df)
            winning_trades_df = df[df['profit_loss'] > 0]
            losing_trades_df = df[df['profit_loss'] <= 0]
            win_rate = (len(winning_trades_df) / total_trades) * 100 if total_trades > 0 else 0.0

            # Correct max drawdown calculation from the simulated equity curve
            equity_df = pd.DataFrame(equity_curve)
            equity_df['value'] = pd.to_numeric(equity_df['value'])
            peak = equity_df['value'].expanding(min_periods=1).max()
            drawdown = ((equity_df['value'] - peak) / peak).fillna(0)
            max_drawdown = abs(float(drawdown.min()) * 100) if not drawdown.empty else 0.0

            # Calculate other statistics
            gross_profit = float(winning_trades_df['profit_loss'].sum())
            gross_loss = abs(float(losing_trades_df['profit_loss'].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_win = float(winning_trades_df['profit_loss'].mean()) if not winning_trades_df.empty else 0.0
            avg_loss = abs(float(losing_trades_df['profit_loss'].mean())) if not losing_trades_df.empty else 0.0
            largest_win = float(winning_trades_df['profit_loss'].max()) if not winning_trades_df.empty else 0.0
            largest_loss = abs(float(losing_trades_df['profit_loss'].min())) if not losing_trades_df.empty else 0.0

            self.backtests[test_id] = {
                'status': 'completed', 'final_balance': float(final_balance), 'total_pnl': float(total_pnl),
                'profit_factor': float(profit_factor), 'avg_win': float(avg_win), 'avg_loss': float(avg_loss),
                'largest_win': float(largest_win), 'largest_loss': float(largest_loss),
                'total_return': float(total_return_pct), 'total_trades': total_trades, 'win_rate': float(win_rate),
                'winning_trades': len(winning_trades_df), 'losing_trades': len(losing_trades_df),
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': max_drawdown, 'equity_curve': equity_curve
            }
            logger.info(f"Backtest {test_id} completed successfully with new logic.")
            # --- NEW FIX ENDS HERE ---
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