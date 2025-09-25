"""
Dashboard for DexScreener Trading Bot
Real-time web-based monitoring and control interface
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
from aiohttp import web
import aiohttp_cors
from aiohttp_sse import sse_response
import socketio
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

class DashboardSection(Enum):
    """Dashboard sections"""
    OVERVIEW = "overview"
    POSITIONS = "positions"
    PERFORMANCE = "performance"
    ORDERS = "orders"
    ALERTS = "alerts"
    RISK = "risk"
    SETTINGS = "settings"
    CHARTS = "charts"

@dataclass
class DashboardData:
    """Dashboard data structure"""
    timestamp: datetime
    portfolio_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    daily_pnl: Decimal
    total_pnl: Decimal
    open_positions: int
    pending_orders: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    active_alerts: int

@dataclass
class ChartData:
    """Chart data structure"""
    chart_id: str
    chart_type: str  # line, candlestick, bar, pie
    title: str
    data: List[Dict]
    options: Dict[str, Any] = field(default_factory=dict)
    update_interval: int = 5  # seconds

class Dashboard:
    """
    Web-based dashboard for real-time monitoring
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        config: Optional[Dict] = None
    ):
        """Initialize dashboard"""
        self.host = host
        self.port = port
        self.config = config or self._default_config()
        
        # Web application
        self.app = web.Application()
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        self.sio.attach(self.app)
        
        # Data storage
        self.dashboard_data = None
        self.positions_data = []
        self.orders_data = []
        self.performance_data = {}
        self.risk_data = {}
        self.alerts_data = []
        self.charts_data: Dict[str, ChartData] = {}
        
        # Connected clients
        self.connected_clients: Set[str] = set()
        
        # Template engine
        self.jinja_env = Environment(
            loader=FileSystemLoader('templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Initialize routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Start update tasks
        asyncio.create_task(self._update_dashboard_data())
        asyncio.create_task(self._broadcast_updates())
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # Update intervals (seconds)
            "data_update_interval": 1,
            "chart_update_interval": 5,
            "broadcast_interval": 1,
            
            # Display settings
            "max_positions_display": 20,
            "max_orders_display": 50,
            "max_alerts_display": 100,
            "chart_history_points": 100,
            
            # Features
            "enable_trading_controls": True,
            "enable_position_management": True,
            "enable_alert_management": True,
            "enable_strategy_controls": True,
            
            # Security
            "require_authentication": False,
            "api_key": None,
            "allowed_ips": [],
            
            # Appearance
            "theme": "dark",
            "show_animations": True,
            "compact_mode": False
        }
    
    def _setup_routes(self):
        """Setup web routes"""
        # Static files
        self.app.router.add_static('/static', 'static', name='static')
        
        # Main pages
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/dashboard', self.dashboard_handler)
        
        # API endpoints
        self.app.router.add_get('/api/status', self.status_handler)
        self.app.router.add_get('/api/positions', self.positions_handler)
        self.app.router.add_get('/api/orders', self.orders_handler)
        self.app.router.add_get('/api/performance', self.performance_handler)
        self.app.router.add_get('/api/risk', self.risk_handler)
        self.app.router.add_get('/api/alerts', self.alerts_handler)
        self.app.router.add_get('/api/charts/{chart_id}', self.chart_handler)
        
        # Control endpoints
        self.app.router.add_post('/api/position/close', self.close_position_handler)
        self.app.router.add_post('/api/position/modify', self.modify_position_handler)
        self.app.router.add_post('/api/order/cancel', self.cancel_order_handler)
        self.app.router.add_post('/api/order/modify', self.modify_order_handler)
        self.app.router.add_post('/api/strategy/toggle', self.toggle_strategy_handler)
        self.app.router.add_post('/api/alert/dismiss', self.dismiss_alert_handler)
        
        # SSE endpoint for real-time updates
        self.app.router.add_get('/api/stream', self.sse_handler)
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _setup_socketio_handlers(self):
        """Setup Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            self.connected_clients.add(sid)
            logger.info(f"Client connected: {sid}")
            
            # Send initial data
            await self.sio.emit('initial_data', {
                'dashboard': self._serialize_dashboard_data(),
                'positions': self.positions_data,
                'orders': self.orders_data,
                'performance': self.performance_data,
                'alerts': self.alerts_data[-10:]  # Last 10 alerts
            }, to=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            self.connected_clients.discard(sid)
            logger.info(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def subscribe_chart(sid, data):
            """Subscribe to chart updates"""
            chart_id = data.get('chart_id')
            if chart_id in self.charts_data:
                await self.sio.emit('chart_update', {
                    'chart_id': chart_id,
                    'data': asdict(self.charts_data[chart_id])
                }, to=sid)
        
        @self.sio.event
        async def execute_action(sid, data):
            """Execute trading action"""
            action = data.get('action')
            params = data.get('params', {})
            
            result = await self._execute_action(action, params)
            
            await self.sio.emit('action_result', {
                'action': action,
                'success': result.get('success'),
                'message': result.get('message')
            }, to=sid)
    
    async def index_handler(self, request):
        """Handle index page"""
        try:
            template = self.jinja_env.get_template('index.html')
            html = template.render(
                title="DexScreener Trading Bot",
                config=self.config
            )
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            logger.error(f"Error rendering index: {e}")
            return web.Response(text="Template not found", status=500)
    
    async def dashboard_handler(self, request):
        """Handle dashboard page"""
        try:
            template = self.jinja_env.get_template('dashboard.html')
            html = template.render(
                title="Trading Dashboard",
                config=self.config,
                initial_data=self._serialize_dashboard_data()
            )
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            logger.error(f"Error rendering dashboard: {e}")
            return web.Response(text="Template not found", status=500)
    
    async def status_handler(self, request):
        """Handle status API request"""
        return web.json_response({
            'status': 'online',
            'timestamp': datetime.utcnow().isoformat(),
            'dashboard_data': self._serialize_dashboard_data(),
            'connected_clients': len(self.connected_clients)
        })
    
    async def positions_handler(self, request):
        """Handle positions API request"""
        return web.json_response({
            'positions': self.positions_data,
            'count': len(self.positions_data),
            'total_value': sum(p.get('current_value', 0) for p in self.positions_data)
        })
    
    async def orders_handler(self, request):
        """Handle orders API request"""
        return web.json_response({
            'orders': self.orders_data,
            'count': len(self.orders_data),
            'pending': sum(1 for o in self.orders_data if o.get('status') == 'pending')
        })
    
    async def performance_handler(self, request):
        """Handle performance API request"""
        return web.json_response(self.performance_data)
    
    async def risk_handler(self, request):
        """Handle risk API request"""
        return web.json_response(self.risk_data)
    
    async def alerts_handler(self, request):
        """Handle alerts API request"""
        limit = int(request.query.get('limit', 100))
        return web.json_response({
            'alerts': self.alerts_data[-limit:],
            'count': len(self.alerts_data),
            'active': sum(1 for a in self.alerts_data if not a.get('dismissed'))
        })
    
    async def chart_handler(self, request):
        """Handle chart data request"""
        chart_id = request.match_info['chart_id']
        if chart_id in self.charts_data:
            return web.json_response(asdict(self.charts_data[chart_id]))
        return web.json_response({'error': 'Chart not found'}, status=404)
    
    async def close_position_handler(self, request):
        """Handle close position request"""
        try:
            data = await request.json()
            position_id = data.get('position_id')
            
            # This would call the actual position manager
            result = await self._close_position(position_id)
            
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def modify_position_handler(self, request):
        """Handle modify position request"""
        try:
            data = await request.json()
            position_id = data.get('position_id')
            modifications = data.get('modifications', {})
            
            # This would call the actual position manager
            result = await self._modify_position(position_id, modifications)
            
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def cancel_order_handler(self, request):
        """Handle cancel order request"""
        try:
            data = await request.json()
            order_id = data.get('order_id')
            
            # This would call the actual order manager
            result = await self._cancel_order(order_id)
            
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def modify_order_handler(self, request):
        """Handle modify order request"""
        try:
            data = await request.json()
            order_id = data.get('order_id')
            modifications = data.get('modifications', {})
            
            # This would call the actual order manager
            result = await self._modify_order(order_id, modifications)
            
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def toggle_strategy_handler(self, request):
        """Handle toggle strategy request"""
        try:
            data = await request.json()
            strategy_name = data.get('strategy')
            enabled = data.get('enabled')
            
            # This would call the actual strategy manager
            result = await self._toggle_strategy(strategy_name, enabled)
            
            return web.json_response(result)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def dismiss_alert_handler(self, request):
        """Handle dismiss alert request"""
        try:
            data = await request.json()
            alert_id = data.get('alert_id')
            
            # Dismiss alert
            for alert in self.alerts_data:
                if alert.get('id') == alert_id:
                    alert['dismissed'] = True
                    break
            
            return web.json_response({'success': True})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def sse_handler(self, request):
        """Handle Server-Sent Events for real-time updates"""
        async with sse_response(request) as resp:
            try:
                while True:
                    # Send dashboard update
                    await resp.send(json.dumps({
                        'type': 'dashboard_update',
                        'data': self._serialize_dashboard_data()
                    }))
                    
                    # Send position updates
                    await resp.send(json.dumps({
                        'type': 'positions_update',
                        'data': self.positions_data
                    }))
                    
                    await asyncio.sleep(self.config["broadcast_interval"])
                    
            except ConnectionResetError:
                pass
        return resp
    
    def update_dashboard_data(
        self,
        portfolio_value: Decimal,
        cash_balance: Decimal,
        positions_value: Decimal,
        daily_pnl: Decimal,
        total_pnl: Decimal,
        open_positions: int,
        pending_orders: int,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
        active_alerts: int
    ):
        """Update dashboard data"""
        self.dashboard_data = DashboardData(
            timestamp=datetime.utcnow(),
            portfolio_value=portfolio_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            open_positions=open_positions,
            pending_orders=pending_orders,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            active_alerts=active_alerts
        )
    
    def update_positions(self, positions: List[Dict]):
        """Update positions data"""
        self.positions_data = positions[:self.config["max_positions_display"]]
    
    def update_orders(self, orders: List[Dict]):
        """Update orders data"""
        self.orders_data = orders[:self.config["max_orders_display"]]
    
    def update_performance(self, performance: Dict):
        """Update performance data"""
        self.performance_data = performance
    
    def update_risk(self, risk: Dict):
        """Update risk data"""
        self.risk_data = risk
    
    def add_alert(self, alert: Dict):
        """Add new alert"""
        self.alerts_data.append({
            **alert,
            'timestamp': datetime.utcnow().isoformat(),
            'dismissed': False
        })
        
        # Limit alerts
        if len(self.alerts_data) > self.config["max_alerts_display"]:
            self.alerts_data = self.alerts_data[-self.config["max_alerts_display"]:]
    
    def add_chart(
        self,
        chart_id: str,
        chart_type: str,
        title: str,
        data: List[Dict],
        options: Optional[Dict] = None,
        update_interval: int = 5
    ):
        """Add or update chart"""
        self.charts_data[chart_id] = ChartData(
            chart_id=chart_id,
            chart_type=chart_type,
            title=title,
            data=data,
            options=options or {},
            update_interval=update_interval
        )
    
    def update_chart_data(self, chart_id: str, data: List[Dict]):
        """Update chart data"""
        if chart_id in self.charts_data:
            chart = self.charts_data[chart_id]
            chart.data = data[-self.config["chart_history_points"]:]
    
    async def _update_dashboard_data(self):
        """Background task to update dashboard data"""
        while True:
            try:
                # This would fetch real data from the trading system
                # For now, using placeholder update
                if self.dashboard_data:
                    self.dashboard_data.timestamp = datetime.utcnow()
                
                await asyncio.sleep(self.config["data_update_interval"])
                
            except Exception as e:
                logger.error(f"Error updating dashboard data: {e}")
                await asyncio.sleep(5)
    
    async def _broadcast_updates(self):
        """Background task to broadcast updates to clients"""
        while True:
            try:
                if self.connected_clients and self.dashboard_data:
                    # Broadcast dashboard update
                    await self.sio.emit('dashboard_update', {
                        'data': self._serialize_dashboard_data()
                    })
                    
                    # Broadcast positions update
                    if self.positions_data:
                        await self.sio.emit('positions_update', {
                            'data': self.positions_data
                        })
                    
                    # Broadcast new alerts
                    recent_alerts = [
                        a for a in self.alerts_data[-5:]
                        if not a.get('dismissed')
                    ]
                    if recent_alerts:
                        await self.sio.emit('alerts_update', {
                            'data': recent_alerts
                        })
                
                await asyncio.sleep(self.config["broadcast_interval"])
                
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
                await asyncio.sleep(5)
    
    def _serialize_dashboard_data(self) -> Dict:
        """Serialize dashboard data for transmission"""
        if not self.dashboard_data:
            return {}
        
        return {
            'timestamp': self.dashboard_data.timestamp.isoformat(),
            'portfolio_value': float(self.dashboard_data.portfolio_value),
            'cash_balance': float(self.dashboard_data.cash_balance),
            'positions_value': float(self.dashboard_data.positions_value),
            'daily_pnl': float(self.dashboard_data.daily_pnl),
            'total_pnl': float(self.dashboard_data.total_pnl),
            'open_positions': self.dashboard_data.open_positions,
            'pending_orders': self.dashboard_data.pending_orders,
            'win_rate': self.dashboard_data.win_rate,
            'sharpe_ratio': self.dashboard_data.sharpe_ratio,
            'max_drawdown': self.dashboard_data.max_drawdown,
            'active_alerts': self.dashboard_data.active_alerts
        }
    
    async def _execute_action(self, action: str, params: Dict) -> Dict:
        """Execute trading action"""
        # This would interface with the actual trading system
        # Placeholder implementation
        return {
            'success': True,
            'message': f"Action {action} executed successfully"
        }
    
    async def _close_position(self, position_id: str) -> Dict:
        """Close position (placeholder)"""
        return {'success': True, 'message': f"Position {position_id} closed"}
    
    async def _modify_position(self, position_id: str, mods: Dict) -> Dict:
        """Modify position (placeholder)"""
        return {'success': True, 'message': f"Position {position_id} modified"}
    
    async def _cancel_order(self, order_id: str) -> Dict:
        """Cancel order (placeholder)"""
        return {'success': True, 'message': f"Order {order_id} cancelled"}
    
    async def _modify_order(self, order_id: str, mods: Dict) -> Dict:
        """Modify order (placeholder)"""
        return {'success': True, 'message': f"Order {order_id} modified"}
    
    async def _toggle_strategy(self, strategy: str, enabled: bool) -> Dict:
        """Toggle strategy (placeholder)"""
        state = "enabled" if enabled else "disabled"
        return {'success': True, 'message': f"Strategy {strategy} {state}"}
    
    async def start(self):
        """Start dashboard server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"Dashboard running on http://{self.host}:{self.port}")
    
    def create_portfolio_chart(self) -> str:
        """Create portfolio value chart"""
        chart_id = "portfolio_value"
        
        # Generate sample data (would use real data)
        data = []
        base_value = 10000
        for i in range(100):
            timestamp = datetime.utcnow() - timedelta(minutes=100-i)
            value = base_value + (i * 10) + (i % 10 - 5) * 50
            data.append({
                'x': timestamp.isoformat(),
                'y': value
            })
        
        self.add_chart(
            chart_id=chart_id,
            chart_type="line",
            title="Portfolio Value",
            data=data,
            options={
                'color': '#00ff88',
                'fillArea': True,
                'showGrid': True,
                'animation': True
            }
        )
        
        return chart_id
    
    def create_positions_chart(self) -> str:
        """Create positions P&L chart"""
        chart_id = "positions_pnl"
        
        # Generate sample data
        data = []
        if self.positions_data:
            for pos in self.positions_data:
                data.append({
                    'label': pos.get('symbol', 'Unknown'),
                    'value': pos.get('unrealized_pnl', 0),
                    'color': '#00ff88' if pos.get('unrealized_pnl', 0) > 0 else '#ff4444'
                })
        
        self.add_chart(
            chart_id=chart_id,
            chart_type="bar",
            title="Positions P&L",
            data=data,
            options={
                'horizontal': True,
                'showValues': True,
                'animation': True
            }
        )
        
        return chart_id
    
    def create_win_rate_chart(self) -> str:
        """Create win rate pie chart"""
        chart_id = "win_rate"
        
        win_rate = self.performance_data.get('win_rate', 0)
        data = [
            {'label': 'Wins', 'value': win_rate, 'color': '#00ff88'},
            {'label': 'Losses', 'value': 1 - win_rate, 'color': '#ff4444'}
        ]
        
        self.add_chart(
            chart_id=chart_id,
            chart_type="pie",
            title="Win Rate",
            data=data,
            options={
                'showPercentage': True,
                'donut': True,
                'animation': True
            }
        )
        
        return chart_id
    
    def create_volume_chart(self) -> str:
        """Create trading volume chart"""
        chart_id = "trading_volume"
        
        # Generate sample data
        data = []
        for i in range(24):
            hour = datetime.utcnow() - timedelta(hours=24-i)
            volume = 5000 + (i * 100) + (i % 5) * 200
            data.append({
                'x': hour.isoformat(),
                'y': volume
            })
        
        self.add_chart(
            chart_id=chart_id,
            chart_type="bar",
            title="24h Trading Volume",
            data=data,
            options={
                'color': '#0099ff',
                'showGrid': True
            }
        )
        
        return chart_id
    
    def create_risk_gauge(self) -> str:
        """Create risk level gauge"""
        chart_id = "risk_gauge"
        
        risk_score = self.risk_data.get('portfolio_risk_score', 50)
        data = [{
            'value': risk_score,
            'min': 0,
            'max': 100,
            'zones': [
                {'from': 0, 'to': 30, 'color': '#00ff88', 'label': 'Low'},
                {'from': 30, 'to': 70, 'color': '#ffaa00', 'label': 'Medium'},
                {'from': 70, 'to': 100, 'color': '#ff4444', 'label': 'High'}
            ]
        }]
        
        self.add_chart(
            chart_id=chart_id,
            chart_type="gauge",
            title="Risk Level",
            data=data,
            options={
                'showNeedle': True,
                'animation': True
            }
        )
        
        return chart_id


class DashboardManager:
    """
    Manager for dashboard components and data flow
    """
    
    def __init__(self, dashboard: Dashboard):
        """Initialize dashboard manager"""
        self.dashboard = dashboard
        self.update_tasks = []
        
    async def connect_to_trading_system(
        self,
        engine,
        portfolio_manager,
        order_manager,
        alerts_system
    ):
        """Connect dashboard to trading system components"""
        self.engine = engine
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.alerts_system = alerts_system
        
        # Start update tasks
        self.update_tasks = [
            asyncio.create_task(self._update_portfolio_data()),
            asyncio.create_task(self._update_positions_data()),
            asyncio.create_task(self._update_orders_data()),
            asyncio.create_task(self._update_alerts_data()),
            asyncio.create_task(self._update_performance_data()),
            asyncio.create_task(self._update_risk_data())
        ]
    
    async def _update_portfolio_data(self):
        """Update portfolio data from trading system"""
        while True:
            try:
                if self.portfolio_manager:
                    summary = self.portfolio_manager.get_portfolio_summary()
                    
                    self.dashboard.update_dashboard_data(
                        portfolio_value=Decimal(summary.get('total_value', 0)),
                        cash_balance=Decimal(summary.get('cash_balance', 0)),
                        positions_value=Decimal(summary.get('positions_value', 0)),
                        daily_pnl=Decimal(summary.get('daily_pnl', 0)),
                        total_pnl=Decimal(summary.get('net_profit', 0)),
                        open_positions=summary.get('open_positions', 0),
                        pending_orders=len(self.order_manager.active_orders) if self.order_manager else 0,
                        win_rate=float(summary.get('win_rate', 0)),
                        sharpe_ratio=float(summary.get('sharpe_ratio', 0)),
                        max_drawdown=float(summary.get('max_drawdown', 0)),
                        active_alerts=len(self.alerts_system.alerts_queue._queue) if self.alerts_system else 0
                    )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error updating portfolio data: {e}")
                await asyncio.sleep(5)
    
    async def _update_positions_data(self):
        """Update positions data"""
        while True:
            try:
                if self.portfolio_manager:
                    positions = self.portfolio_manager.get_open_positions()
                    self.dashboard.update_positions(positions)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(5)
    
    async def _update_orders_data(self):
        """Update orders data"""
        while True:
            try:
                if self.order_manager:
                    orders = self.order_manager.get_active_orders()
                    self.dashboard.update_orders(orders)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error updating orders: {e}")
                await asyncio.sleep(5)
    
    async def _update_alerts_data(self):
        """Update alerts data"""
        while True:
            try:
                if self.alerts_system:
                    # Subscribe to new alerts
                    async def alert_callback(alert):
                        self.dashboard.add_alert({
                            'id': alert.alert_id,
                            'type': alert.alert_type.value,
                            'priority': alert.priority.value,
                            'title': alert.title,
                            'message': alert.message
                        })
                    
                    # Get recent alerts
                    stats = self.alerts_system.get_alert_stats()
                    for alert in stats.get('recent_alerts', []):
                        self.dashboard.add_alert(alert)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error updating alerts: {e}")
                await asyncio.sleep(5)
    
    async def _update_performance_data(self):
        """Update performance data"""
        while True:
            try:
                if self.portfolio_manager:
                    performance = self.portfolio_manager.get_performance_report()
                    self.dashboard.update_performance(performance)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(10)
    
    async def _update_risk_data(self):
        """Update risk data"""
        while True:
            try:
                if self.portfolio_manager:
                    risk_metrics = {
                        'portfolio_risk_score': 50,  # Placeholder
                        'var_95': str(self.portfolio_manager.risk_metrics.get('var_95', 0)),
                        'cvar_95': str(self.portfolio_manager.risk_metrics.get('cvar_95', 0)),
                        'portfolio_beta': self.portfolio_manager.risk_metrics.get('portfolio_beta', 1.0),
                        'portfolio_volatility': self.portfolio_manager.risk_metrics.get('portfolio_volatility', 0)
                    }
                    self.dashboard.update_risk(risk_metrics)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error updating risk data: {e}")
                await asyncio.sleep(10)