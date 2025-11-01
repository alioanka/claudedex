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
import csv
from config.config_manager import PortfolioConfig

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
        
        # API - Data endpoints
        self.app.router.add_get('/api/dashboard/summary', self.api_dashboard_summary)
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
    
    # ==================== API - DATA ENDPOINTS ====================
    
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
                    
                    # ✅ Get entry_timestamp and stop_loss/take_profit from database
                    entry_timestamp = position.get('entry_timestamp') or position.get('opened_at') or position.get('timestamp')
                    stop_loss = position.get('stop_loss')
                    take_profit = position.get('take_profit')
                    
                    # ✅ Try to enrich from database if fields are missing
                    if (not entry_timestamp or not stop_loss or not take_profit) and self.db:
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
                                    
                                    # Extract stop_loss/take_profit from metadata
                                    if trade['metadata']:
                                        try:
                                            import json
                                            metadata = trade['metadata']
                                            if isinstance(metadata, str):
                                                metadata = json.loads(metadata)
                                            
                                            if not stop_loss:
                                                stop_loss = metadata.get('stop_loss')
                                            if not take_profit:
                                                take_profit = metadata.get('take_profit')
                                        except:
                                            pass
                        except Exception as e:
                            logger.error(f"Error enriching position from DB: {e}")
                    
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
        """Get performance metrics - SAFE VERSION"""
        try:
            # Simple safe metrics structure
            metrics = {
                'historical': {
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0
                }
            }
            
            # Try to get data from database if available
            if self.db:
                try:
                    # Try to get performance summary
                    if hasattr(self.db, 'get_performance_summary'):
                        perf = await self.db.get_performance_summary()
                        if perf:
                            metrics['historical']['total_pnl'] = float(perf.get('total_pnl', 0))
                            metrics['historical']['win_rate'] = float(perf.get('win_rate', 0))
                            metrics['historical']['total_trades'] = int(perf.get('total_trades', 0))
                            metrics['historical']['winning_trades'] = int(perf.get('winning_trades', 0))  # ✅ ADD
                            metrics['historical']['losing_trades'] = int(perf.get('losing_trades', 0))    # ✅ ADD
                            metrics['historical']['avg_win'] = float(perf.get('avg_win', 0))
                            metrics['historical']['avg_loss'] = float(perf.get('avg_loss', 0))
                except Exception as db_error:
                    logger.error(f"Error getting performance data: {db_error}")
            
            return web.json_response({
                'success': True,
                'data': self._serialize_decimals(metrics)
            })
            
        except Exception as e:
            logger.error(f"Error in api_performance_metrics: {e}")
            return web.json_response({
                'success': True,
                'data': {
                    'historical': {
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0
                    }
                }
            }, status=200)

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
            timeframe = request.query.get('timeframe', '24h')
            
            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)
            
            # Get closed trades
            all_trades = await self.db.get_recent_trades(limit=1000)
            all_trades = self._serialize_decimals(all_trades)
            
            # Filter only closed trades with exit timestamps
            closed_trades = [t for t in all_trades if t.get('status') == 'closed' and t.get('exit_timestamp')]
            
            if not closed_trades:
                return web.json_response({
                    'success': True,
                    'data': {
                        'portfolio_history': [],
                        'pnl_history': [],
                        'strategy_performance': {}
                    }
                })
            
            # Sort by exit timestamp
            closed_trades.sort(key=lambda x: x['exit_timestamp'])
            
            # Filter by timeframe
            from datetime import datetime, timedelta
            now = datetime.utcnow()
            
            timeframe_hours = {'1h': 1, '24h': 24, '7d': 168, '30d': 720}
            hours = timeframe_hours.get(timeframe, 24)
            cutoff = now - timedelta(hours=hours)
            
            # ✅ Calculate cumulative P&L before the timeframe
            cumulative_pnl_before = 0
            filtered_trades = []
            
            for trade in closed_trades:
                try:
                    trade_time = datetime.fromisoformat(trade['exit_timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
                    trade_time = trade_time.replace(tzinfo=None)  # Make naive for comparison
                    cutoff_naive = cutoff.replace(tzinfo=None)
                    
                    if trade_time < cutoff_naive:
                        cumulative_pnl_before += float(trade.get('profit_loss', 0))
                    else:
                        filtered_trades.append(trade)
                except:
                    pass
            
            # Generate portfolio history
            portfolio_history = []
            pnl_history = []
            from collections import defaultdict
            daily_pnl = defaultdict(float)
            
            # Calculate performance metrics
            try:
                from config.config_manager import PortfolioConfig
                config = PortfolioConfig()
                starting_value = config.initial_balance
            except Exception:
                starting_value = 400  # Fallback
            cumulative_pnl = cumulative_pnl_before
            
            for trade in filtered_trades:
                pnl = float(trade.get('profit_loss', 0))
                cumulative_pnl += pnl
                portfolio_value = starting_value + cumulative_pnl
                
                portfolio_history.append({
                    'timestamp': trade['exit_timestamp'],
                    'value': portfolio_value
                })
                
                date_key = trade['exit_timestamp'][:10]
                daily_pnl[date_key] += pnl
            
            pnl_history = [
                {'timestamp': date, 'value': pnl}
                for date, pnl in sorted(daily_pnl.items())
            ]
            
            logger.info(f"Chart data: {len(portfolio_history)} portfolio points, {len(pnl_history)} P&L points")
            
            return web.json_response({
                'success': True,
                'data': {
                    'portfolio_history': portfolio_history,
                    'pnl_history': pnl_history,
                    'strategy_performance': {}
                }
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
        """Get bot status with robust detection"""
        try:
            # ✅ FIX: Check multiple possible state indicators
            is_running = False
            uptime_str = 'N/A'
            
            if self.engine:
                # Try different state attribute names
                if hasattr(self.engine, 'state'):
                    # Check for various "running" values
                    state_val = str(self.engine.state).lower()
                    is_running = state_val in ['running', 'active', 'started', 'true', '1']
                elif hasattr(self.engine, 'running'):
                    is_running = bool(self.engine.running)
                elif hasattr(self.engine, 'is_running'):
                    is_running = bool(self.engine.is_running)
                elif hasattr(self.engine, '_running'):
                    is_running = bool(self.engine._running)
                elif hasattr(self.engine, 'active'):
                    is_running = bool(self.engine.active)
                
                # Calculate uptime if running
                if is_running:
                    start_time = None
                    if hasattr(self.engine, 'start_time') and self.engine.start_time:
                        start_time = self.engine.start_time
                    elif hasattr(self.engine, 'started_at') and self.engine.started_at:
                        start_time = self.engine.started_at
                    elif hasattr(self.engine, 'startup_time') and self.engine.startup_time:
                        start_time = self.engine.startup_time
                    
                    if start_time:
                        try:
                            uptime_delta = datetime.utcnow() - start_time
                            hours = int(uptime_delta.total_seconds() // 3600)
                            minutes = int((uptime_delta.total_seconds() % 3600) // 60)
                            uptime_str = f"{hours}h {minutes}m"
                        except Exception as e:
                            logger.warning(f"Error calculating uptime: {e}")
                            uptime_str = "Running"
            
            status = {
                'running': is_running,
                'uptime': uptime_str,
                'mode': self.config.get('mode', 'unknown'),
                'version': '1.0.0',
                'last_health_check': datetime.utcnow().isoformat()
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
    
    async def api_get_settings(self, request):
        """Get all settings"""
        try:
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)
            
            settings = {
                'trading': self.config_mgr.get_trading_config(),
                'risk': self.config_mgr.get_risk_management_config(),
                'api': self.config_mgr.get_api_config(),
                'monitoring': self.config_mgr.get_monitoring_config(),
                'ml_models': self.config_mgr.get_ml_models_config()
            }
            
            return web.json_response({
                'success': True,
                'data': settings
            })
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
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
        try:
            data = await request.json()
            config_type = data.get('config_type')
            version = data.get('version', -1)  # -1 for previous, -2 for two versions back, etc.
            
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)
            
            # Get change history
            history = self.config_mgr.get_change_history(config_type, limit=abs(version) + 1)
            
            if len(history) >= abs(version):
                old_config = history[abs(version) - 1]['new_config']
                
                # Apply old configuration
                self.config_mgr.update_config_internal(
                    config_type=config_type,
                    updates=old_config,
                    user='dashboard',
                    reason=f'Reverted to version {version}',
                    persist=True
                )
                
                return web.json_response({
                    'success': True,
                    'message': f'Settings reverted to version {version}'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Version not found in history'
                }, status=404)
        except Exception as e:
            logger.error(f"Error reverting settings: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_settings_history(self, request):
        """Get settings change history"""
        try:
            config_type = request.query.get('config_type')
            limit = int(request.query.get('limit', 50))
            
            if not self.config_mgr:
                return web.json_response({'error': 'Config manager not available'}, status=503)
            
            history = self.config_mgr.get_change_history(config_type, limit=limit)
            
            return web.json_response({
                'success': True,
                'data': history,
                'count': len(history)
            })
        except Exception as e:
            logger.error(f"Error getting settings history: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
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
            format_type = request.match_info['format']  # csv, excel, pdf, json
            
            # Get report data
            period = request.query.get('period', 'daily')
            report = await self._generate_report(period, None, None, ['all'])
            
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
            logger.error(f"Error exporting report: {e}")
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
            
            # Retrieve results (implement storage mechanism)
            results = {}  # Fetch from storage
            
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
#                if self.db:
#                    try:
#                        perf_data = await self.db.get_performance_summary()
#                        if 'error' not in perf_data:  # Only broadcast if no error
#                            await self.sio.emit('performance_update', {
#                                **perf_data,
#                                'timestamp': datetime.utcnow().isoformat()
#                            })
#                    except Exception as e:
#                        logger.debug(f"Error broadcasting performance update: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def _generate_report(self, period, start_date, end_date, metrics):
        """Generate performance report"""
        report = {
            'period': period,
            'generated_at': datetime.utcnow().isoformat(),
            'metrics': {}
        }
        
        # Get trades for period
        trades = self.db.get_recent_trades(limit=1000)
        
        # Apply date filters
        if start_date:
            start = datetime.fromisoformat(start_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) >= start]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            trades = [t for t in trades if datetime.fromisoformat(t['timestamp']) <= end]
        
        # Calculate metrics
        if 'all' in metrics or 'pnl' in metrics:
            total_pnl = sum(float(t.get('pnl', 0)) for t in trades if t.get('status') == 'closed')
            report['metrics']['total_pnl'] = total_pnl
        
        if 'all' in metrics or 'win_rate' in metrics:
            winning_trades = len([t for t in trades if float(t.get('pnl', 0)) > 0])
            total_trades = len([t for t in trades if t.get('status') == 'closed'])
            report['metrics']['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        if 'all' in metrics or 'trades_count' in metrics:
            report['metrics']['total_trades'] = len(trades)
            report['metrics']['open_trades'] = len([t for t in trades if t.get('status') == 'open'])
            report['metrics']['closed_trades'] = len([t for t in trades if t.get('status') == 'closed'])
        
        # Add more metrics as needed
        
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
        
        # Write CSV data
        if 'metrics' in report:
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Value'])
            for key, value in report['metrics'].items():
                writer.writerow([key, value])
        
        # Return as file
        response = web.Response(
            body=output.getvalue().encode('utf-8'),
            content_type='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv"'
            }
        )
        return response
    
    async def _export_excel(self, report):
        """Export report as Excel"""
        # Create Excel file using pandas or openpyxl
        output = io.BytesIO()
        
        if 'metrics' in report:
            df = pd.DataFrame(list(report['metrics'].items()), columns=['Metric', 'Value'])
            df.to_excel(output, index=False, engine='openpyxl')
        
        output.seek(0)
        
        response = web.Response(
            body=output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename="report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.xlsx"'
            }
        )
        return response
    
    async def _export_pdf(self, report):
        """Export report as PDF"""
        # Implement PDF generation (using reportlab or similar)
        # For now, return JSON
        return web.json_response(report)
    
    async def _run_backtest_task(self, test_id, strategy, start_date, end_date, initial_balance, parameters):
        """Run backtest in background"""
        try:
            logger.info(f"Starting backtest {test_id}")
            
            # Run backtest logic here
            # Store results
            
            logger.info(f"Backtest {test_id} completed")
        except Exception as e:
            logger.error(f"Error in backtest {test_id}: {e}")
    
    async def start(self):
        """Start the dashboard server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Enhanced dashboard running on http://{self.host}:{self.port}")
        
        # Keep running
        await asyncio.Event().wait()