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
            if not self.portfolio:
                return web.json_response({'error': 'Portfolio manager not available'}, status=503)
            
            summary = self.portfolio.get_portfolio_summary()
            
            return web.json_response({
                'success': True,
                'data': {
                    'portfolio_value': float(summary.get('total_value', 0)),
                    'cash_balance': float(summary.get('cash_balance', 0)),
                    'positions_value': float(summary.get('positions_value', 0)),
                    'daily_pnl': float(summary.get('daily_pnl', 0)),
                    'total_pnl': float(summary.get('net_profit', 0)),
                    'open_positions': summary.get('open_positions', 0),
                    'pending_orders': len(self.orders.active_orders) if self.orders else 0,
                    'win_rate': float(summary.get('win_rate', 0)),
                    'sharpe_ratio': float(summary.get('sharpe_ratio', 0)),
                    'max_drawdown': float(summary.get('max_drawdown', 0)),
                    'active_alerts': len(self.alerts.alerts_queue._queue) if self.alerts else 0
                }
            })
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_recent_trades(self, request):
        """Get recent trades"""
        try:
            limit = int(request.query.get('limit', 50))
            
            if not self.db:
                return web.json_response({'error': 'Database not available'}, status=503)
            
            # ✅ FIX: Add await
            trades = await self.db.get_recent_trades(limit=limit)
            
            return web.json_response({
                'success': True,
                'data': trades,
                'count': len(trades)
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
        """Get open positions"""
        try:
            # ✅ FIX: Get positions from ENGINE, not portfolio
            if not self.engine:
                return web.json_response({'error': 'Trading engine not available'}, status=503)
            
            # Get active positions from engine
            positions = []
            if hasattr(self.engine, 'active_positions'):
                for token_address, position in self.engine.active_positions.items():
                    positions.append({
                        'id': position.get('id'),
                        'token_address': token_address,
                        'token_symbol': position.get('token_symbol', 'Unknown'),
                        'entry_price': float(position.get('entry_price', 0)),
                        'current_price': float(position.get('current_price', position.get('entry_price', 0))),
                        'amount': float(position.get('amount', 0)),
                        'entry_value': float(position.get('entry_value', 0)),
                        'unrealized_pnl': float(position.get('unrealized_pnl', 0)),
                        'status': position.get('status', 'open'),
                        'entry_time': position.get('entry_time').isoformat() if position.get('entry_time') else None,
                        'chain': position.get('chain', 'unknown')
                    })
            
            return web.json_response({
                'success': True,
                'data': positions,
                'count': len(positions)
            })
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_positions_history(self, request):
        """Get positions history"""
        try:
            limit = int(request.query.get('limit', 100))
            
            # Get closed positions from database
            positions = self.db.get_active_positions()  # You'll need to add closed positions query
            
            return web.json_response({
                'success': True,
                'data': positions,
                'count': len(positions)
            })
        except Exception as e:
            logger.error(f"Error getting positions history: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_performance_metrics(self, request):
        """Get performance metrics"""
        try:
            period = request.query.get('period', 'all')
            
            if not self.portfolio:
                return web.json_response({'error': 'Portfolio manager not available'}, status=503)
            
            portfolio_metrics = self.portfolio.get_performance_report()
            
            # ✅ FIX: Add await since get_performance_summary is async
            db_metrics = await self.db.get_performance_summary() if self.db else {}
            
            combined_metrics = {
                **portfolio_metrics,
                'historical': db_metrics,
                'period': period
            }
            
            return web.json_response({
                'success': True,
                'data': combined_metrics
            })
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def api_performance_charts(self, request):
        """Get data for performance charts"""
        try:
            # Get historical data for charts
            timeframe = request.query.get('timeframe', '1d')
            
            # Portfolio value over time
            portfolio_history = []  # Fetch from database
            
            # P&L over time
            pnl_history = []  # Fetch from database
            
            # Win rate by strategy
            strategy_performance = {}  # Calculate from trades
            
            return web.json_response({
                'success': True,
                'data': {
                    'portfolio_history': portfolio_history,
                    'pnl_history': pnl_history,
                    'strategy_performance': strategy_performance
                }
            })
        except Exception as e:
            logger.error(f"Error getting performance charts: {e}")
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
            if not self.portfolio:
                return web.json_response({'error': 'Portfolio manager not available'}, status=503)
            
            # Get all open positions
            positions = self.portfolio.get_open_positions()
            
            closed = []
            failed = []
            
            for pos in positions:
                try:
                    result = self.portfolio.close_position(pos['id'])
                    if result.get('success'):
                        closed.append(pos['id'])
                    else:
                        failed.append(pos['id'])
                except Exception as e:
                    logger.error(f"Error closing position {pos['id']}: {e}")
                    failed.append(pos['id'])
            
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
        """Get bot status"""
        try:
            status = {
                'running': self.engine.state == 'running' if self.engine else False,
                'uptime': str(datetime.utcnow() - self.engine.start_time) if self.engine and hasattr(self.engine, 'start_time') else 'N/A',
                'mode': self.config.get('mode', 'unknown'),
                'version': '1.0.0',  # Add version tracking
                'last_health_check': datetime.utcnow().isoformat()
            }
            
            return web.json_response({
                'success': True,
                'data': status
            })
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
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
            
            result = self.portfolio.close_position(position_id)
            
            return web.json_response({
                'success': True,
                'message': 'Position closed successfully',
                'data': result
            })
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
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
                        if self.portfolio:
                            summary = self.portfolio.get_portfolio_summary()
                            await resp.send(json.dumps({
                                'type': 'portfolio_update',
                                'data': {
                                    'value': float(summary.get('total_value', 0)),
                                    'pnl': float(summary.get('daily_pnl', 0)),
                                    'timestamp': datetime.utcnow().isoformat()
                                }
                            }))
                        
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
        """Send initial data to connected client"""
        try:
            data = {}
            
            if self.portfolio:
                data['portfolio'] = self.portfolio.get_portfolio_summary()
                data['positions'] = self.portfolio.get_open_positions()
            
            if self.orders:
                data['orders'] = self.orders.get_active_orders()
            
            await self.sio.emit('initial_data', data, to=sid)
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
                if self.portfolio:
                    try:
                        summary = self.portfolio.get_portfolio_summary()
                        await self.sio.emit('dashboard_update', {
                            'portfolio_value': float(summary.get('total_value', 0)),
                            'daily_pnl': float(summary.get('daily_pnl', 0)),
                            'open_positions': summary.get('open_positions', 0),
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Error broadcasting portfolio update: {e}")
                
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