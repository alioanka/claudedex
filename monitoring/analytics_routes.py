"""
Analytics Dashboard Routes

API endpoints for advanced analytics:
- Performance metrics
- Risk analysis
- Module comparison
- Historical data
- Real-time updates
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from aiohttp import web
import json

from core.analytics_engine import AnalyticsEngine, TimeFrame

logger = logging.getLogger("AnalyticsRoutes")


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class AnalyticsRoutes:
    """
    Analytics dashboard routes

    Provides RESTful API for analytics data
    """

    def __init__(
        self,
        analytics_engine: AnalyticsEngine,
        jinja_env=None
    ):
        """
        Initialize analytics routes

        Args:
            analytics_engine: Analytics engine instance
            jinja_env: Jinja2 environment for templates
        """
        self.analytics = analytics_engine
        self.jinja_env = jinja_env
        self.logger = logger

    def setup_routes(self, app: web.Application):
        """Setup analytics routes"""
        app.router.add_get('/analytics', self.analytics_page)
        app.router.add_get('/api/analytics/performance/{module}', self.get_performance)
        app.router.add_get('/api/analytics/risk/{module}', self.get_risk)
        app.router.add_get('/api/analytics/comparison', self.get_comparison)
        app.router.add_get('/api/analytics/portfolio', self.get_portfolio)
        app.router.add_get('/api/analytics/equity/{module}', self.get_equity_curve)
        app.router.add_get('/api/analytics/trades/{module}', self.get_trade_history)
        app.router.add_get('/api/analytics/daily-pnl/{module}', self.get_daily_pnl)

        self.logger.info("Analytics routes configured")

    async def analytics_page(self, request: web.Request) -> web.Response:
        """Render analytics dashboard page"""
        try:
            if not self.jinja_env:
                return web.Response(text="Analytics dashboard not configured", status=500)

            # Get portfolio summary
            summary = await self.analytics.get_portfolio_summary()

            # Render template
            template = self.jinja_env.get_template('analytics.html')
            html = template.render(
                summary=summary,
                timestamp=datetime.now()
            )

            return web.Response(text=html, content_type='text/html')

        except Exception as e:
            self.logger.error(f"Error rendering analytics page: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def get_performance(self, request: web.Request) -> web.Response:
        """Get performance metrics for a module"""
        try:
            module_name = request.match_info['module']
            timeframe_str = request.query.get('timeframe', '24h')

            # Parse timeframe
            timeframe_map = {
                '1h': TimeFrame.HOUR_1,
                '4h': TimeFrame.HOUR_4,
                '24h': TimeFrame.HOUR_24,
                '7d': TimeFrame.DAY_7,
                '30d': TimeFrame.DAY_30,
                'all': TimeFrame.ALL
            }
            timeframe = timeframe_map.get(timeframe_str, TimeFrame.HOUR_24)

            # Get performance metrics
            metrics = await self.analytics.get_module_performance(module_name, timeframe)

            # Convert to dict
            data = {
                'module_name': metrics.module_name,
                'timeframe': metrics.timeframe.value,
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': round(metrics.win_rate * 100, 2),
                'total_pnl': float(metrics.total_pnl),
                'realized_pnl': float(metrics.realized_pnl),
                'unrealized_pnl': float(metrics.unrealized_pnl),
                'net_pnl': float(metrics.net_pnl),
                'total_fees': float(metrics.total_fees),
                'profit_factor': round(metrics.profit_factor, 2),
                'sharpe_ratio': round(metrics.sharpe_ratio, 2),
                'sortino_ratio': round(metrics.sortino_ratio, 2),
                'calmar_ratio': round(metrics.calmar_ratio, 2),
                'max_drawdown': round(metrics.max_drawdown * 100, 2),
                'current_drawdown': round(metrics.current_drawdown * 100, 2),
                'max_drawdown_duration_hours': metrics.max_drawdown_duration,
                'avg_win': float(metrics.avg_win),
                'avg_loss': float(metrics.avg_loss),
                'avg_trade_duration_seconds': metrics.avg_trade_duration,
                'avg_daily_pnl': float(metrics.avg_daily_pnl),
                'current_streak': metrics.current_streak,
                'max_win_streak': metrics.max_win_streak,
                'max_loss_streak': metrics.max_loss_streak,
                'best_trade': float(metrics.best_trade),
                'worst_trade': float(metrics.worst_trade),
                'total_volume': float(metrics.total_volume),
                'avg_position_size': float(metrics.avg_position_size),
                'start_time': metrics.start_time.isoformat() if metrics.start_time else None,
                'end_time': metrics.end_time.isoformat() if metrics.end_time else None
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting performance: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_risk(self, request: web.Request) -> web.Response:
        """Get risk metrics for a module"""
        try:
            module_name = request.match_info['module']

            # Get risk metrics
            metrics = await self.analytics.get_risk_metrics(module_name)

            # Convert to dict
            data = {
                'module_name': metrics.module_name,
                'total_exposure': float(metrics.total_exposure),
                'long_exposure': float(metrics.long_exposure),
                'short_exposure': float(metrics.short_exposure),
                'net_exposure': float(metrics.net_exposure),
                'largest_position_pct': round(metrics.largest_position_pct, 2),
                'top_5_positions_pct': round(metrics.top_5_positions_pct, 2),
                'var_95': float(metrics.var_95),
                'var_99': float(metrics.var_99),
                'cvar_95': float(metrics.cvar_95),
                'daily_volatility': round(metrics.daily_volatility * 100, 2),
                'annual_volatility': round(metrics.annual_volatility * 100, 2),
                'avg_leverage': round(metrics.avg_leverage, 2),
                'max_leverage': round(metrics.max_leverage, 2),
                'avg_liquidity_score': round(metrics.avg_liquidity_score, 2),
                'low_liquidity_positions': metrics.low_liquidity_positions
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_comparison(self, request: web.Request) -> web.Response:
        """Get module comparison"""
        try:
            comparison = await self.analytics.compare_modules()

            data = {
                'best_performer': comparison.best_performer,
                'worst_performer': comparison.worst_performer,
                'most_active': comparison.most_active,
                'least_active': comparison.least_active,
                'highest_sharpe': comparison.highest_sharpe,
                'lowest_drawdown': comparison.lowest_drawdown,
                'rankings': comparison.module_rankings
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting comparison: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_portfolio(self, request: web.Request) -> web.Response:
        """Get portfolio summary"""
        try:
            summary = await self.analytics.get_portfolio_summary()
            return web.json_response({'success': True, 'data': summary})

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_equity_curve(self, request: web.Request) -> web.Response:
        """Get equity curve for a module"""
        try:
            module_name = request.match_info['module']
            timeframe_str = request.query.get('timeframe', '7d')

            # Parse timeframe
            timeframe_map = {
                '1h': TimeFrame.HOUR_1,
                '4h': TimeFrame.HOUR_4,
                '24h': TimeFrame.HOUR_24,
                '7d': TimeFrame.DAY_7,
                '30d': TimeFrame.DAY_30,
                'all': TimeFrame.ALL
            }
            timeframe = timeframe_map.get(timeframe_str, TimeFrame.DAY_7)

            # Get performance metrics (includes equity curve)
            metrics = await self.analytics.get_module_performance(module_name, timeframe)

            data = {
                'equity_curve': metrics.equity_curve,
                'start_time': metrics.start_time.isoformat() if metrics.start_time else None,
                'end_time': metrics.end_time.isoformat() if metrics.end_time else None
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting equity curve: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_trade_history(self, request: web.Request) -> web.Response:
        """Get trade history for a module"""
        try:
            module_name = request.match_info['module']
            limit = int(request.query.get('limit', 100))
            offset = int(request.query.get('offset', 0))

            # Get performance metrics (includes trade history)
            metrics = await self.analytics.get_module_performance(
                module_name,
                TimeFrame.DAY_30
            )

            # Get trades with pagination
            trades = metrics.trade_history[offset:offset + limit]

            # Convert to serializable format
            trades_data = []
            for trade in trades:
                trade_dict = dict(trade)
                # Convert Decimal and datetime objects
                for key, value in trade_dict.items():
                    if isinstance(value, Decimal):
                        trade_dict[key] = float(value)
                    elif isinstance(value, datetime):
                        trade_dict[key] = value.isoformat()
                trades_data.append(trade_dict)

            data = {
                'trades': trades_data,
                'total': len(metrics.trade_history),
                'limit': limit,
                'offset': offset
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)

    async def get_daily_pnl(self, request: web.Request) -> web.Response:
        """Get daily PnL for a module"""
        try:
            module_name = request.match_info['module']
            timeframe_str = request.query.get('timeframe', '30d')

            # Parse timeframe
            timeframe_map = {
                '7d': TimeFrame.DAY_7,
                '30d': TimeFrame.DAY_30,
                'all': TimeFrame.ALL
            }
            timeframe = timeframe_map.get(timeframe_str, TimeFrame.DAY_30)

            # Get performance metrics (includes daily PnL)
            metrics = await self.analytics.get_module_performance(module_name, timeframe)

            # Convert daily PnL to list of floats
            daily_pnl = [float(pnl) for pnl in metrics.daily_pnl]

            # Create date labels
            if metrics.start_time:
                dates = [
                    (metrics.start_time + timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range(len(daily_pnl))
                ]
            else:
                dates = [f"Day {i+1}" for i in range(len(daily_pnl))]

            data = {
                'dates': dates,
                'pnl': daily_pnl,
                'cumulative': [sum(daily_pnl[:i+1]) for i in range(len(daily_pnl))]
            }

            return web.json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting daily PnL: {e}", exc_info=True)
            return web.json_response({'success': False, 'error': str(e)}, status=500)
