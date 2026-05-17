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


def json_response(data, status=200):
    """Helper function to return JSON response with proper Decimal handling"""
    return web.Response(
        text=json.dumps(data, cls=DecimalEncoder),
        content_type='application/json',
        status=status
    )


class AnalyticsRoutes:
    """
    Analytics dashboard routes

    Provides RESTful API for analytics data
    """

    def __init__(
        self,
        analytics_engine: AnalyticsEngine,
        jinja_env=None,
        db_manager=None,
    ):
        """
        Initialize analytics routes

        Args:
            analytics_engine: Analytics engine instance
            jinja_env: Jinja2 environment for templates
            db_manager: optional db_manager (with .pool) used as a fallback
                read-only source when analytics_engine is None. FAILURE B:
                the standalone dashboard subprocess passes
                analytics_engine=None and previously every endpoint here
                returned 503 — operators saw a fully zero'd page with no
                indication WHY. With db_manager the endpoints can serve a
                minimal subset of the same shape from the DB.
        """
        self.analytics = analytics_engine
        self.jinja_env = jinja_env
        self.db = db_manager
        self.logger = logger
        # Per-module table mapping for the DB fallback path. Each entry:
        #   (table_name, pnl_column, optional_unit_multiplier)
        # The Solana table denominates pnl in SOL; we deliberately do NOT
        # convert here — the fallback returns raw rows for the page to
        # render counts/percentages without bringing the sol-USD rate
        # dependency into this module.
        self._module_tables = {
            'dex_trading':      ('trades',             'profit_loss'),
            'sniper':           ('sniper_trades',      'profit_loss'),
            'arbitrage':        ('arbitrage_trades',   'profit_loss'),
            'futures_trading':  ('futures_trades',     'net_pnl'),
            'solana_trading':   ('solana_trades',      'pnl_sol'),
            'solana_strategies':('solana_trades',      'pnl_sol'),
            'copy_trading':     ('copytrading_trades', 'profit_loss'),
            'ai_analysis':      ('ai_trades',          'profit_loss'),
        }

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

        self.logger.info(
            "Analytics routes configured%s",
            "" if self.analytics is not None else " (engine=None — endpoints will return 503)"
        )

    def _require_engine(self):
        """Return None if engine is wired, else a 503 response.

        Lets every API endpoint fail-soft when self.analytics is None
        (which is normal in the standalone dashboard subprocess that
        doesn't construct an analytics engine). Without this guard
        every endpoint would AttributeError into a 500.

        FAILURE B: callers should prefer to check _has_engine() and
        fall back to _db_*() methods when the engine is missing but
        db_manager is wired, so the analytics page can render real
        data instead of zeros.
        """
        if self.analytics is None:
            return json_response(
                {'success': False, 'error': 'analytics engine not initialized',
                 'data': {}},
                status=503,
            )
        return None

    def _has_engine(self) -> bool:
        return self.analytics is not None

    def _has_db(self) -> bool:
        return self.db is not None and getattr(self.db, 'pool', None) is not None

    async def _db_perf(self, module_name: str) -> dict:
        """DB-fallback performance dict for the given module."""
        tbl = self._module_tables.get(module_name)
        if not tbl:
            return self._empty_perf(module_name)
        table, pnl_col = tbl
        async with self.db.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT {pnl_col} AS pnl FROM {table} WHERE status='closed'"
                )
            except Exception as e:
                self.logger.debug(f"_db_perf {table} error: {e}")
                return self._empty_perf(module_name)
        pnls = [float(r['pnl'] or 0) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total = len(pnls)
        total_pnl = sum(pnls)
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        win_rate_pct = (len(wins) / total * 100) if total > 0 else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses)) or 1
        profit_factor = gross_profit / gross_loss
        return {
            'module_name': module_name, 'timeframe': 'all',
            'total_trades': total,
            'winning_trades': len(wins), 'losing_trades': len(losses),
            'win_rate': round(win_rate_pct, 2),
            'total_pnl': total_pnl, 'realized_pnl': total_pnl,
            'unrealized_pnl': 0.0, 'net_pnl': total_pnl,
            'total_fees': 0.0,
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'calmar_ratio': 0.0,
            'max_drawdown': 0.0, 'current_drawdown': 0.0,
            'max_drawdown_duration_hours': 0,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'avg_trade_duration_seconds': 0,
            'avg_daily_pnl': 0.0,
            'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0,
            'best_trade': max(pnls) if pnls else 0.0,
            'worst_trade': min(pnls) if pnls else 0.0,
            'total_volume': 0.0, 'avg_position_size': 0.0,
            'start_time': None, 'end_time': None,
        }

    def _empty_perf(self, module_name: str) -> dict:
        return {
            'module_name': module_name, 'timeframe': 'all',
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'realized_pnl': 0.0,
            'unrealized_pnl': 0.0, 'net_pnl': 0.0, 'total_fees': 0.0,
            'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
            'calmar_ratio': 0.0, 'max_drawdown': 0.0, 'current_drawdown': 0.0,
            'max_drawdown_duration_hours': 0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'avg_trade_duration_seconds': 0, 'avg_daily_pnl': 0.0,
            'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0,
            'best_trade': 0.0, 'worst_trade': 0.0, 'total_volume': 0.0,
            'avg_position_size': 0.0, 'start_time': None, 'end_time': None,
        }

    async def _db_equity(self, module_name: str) -> dict:
        """DB-fallback equity curve list for the given module."""
        tbl = self._module_tables.get(module_name)
        if not tbl:
            return {'equity_curve': [], 'start_time': None, 'end_time': None}
        table, pnl_col = tbl
        async with self.db.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT {pnl_col} AS pnl, exit_timestamp FROM {table} "
                    f"WHERE status='closed' AND exit_timestamp IS NOT NULL "
                    f"ORDER BY exit_timestamp"
                )
            except Exception as e:
                self.logger.debug(f"_db_equity {table} error: {e}")
                return {'equity_curve': [], 'start_time': None, 'end_time': None}
        cum = 0.0
        curve = []
        for r in rows:
            cum += float(r['pnl'] or 0)
            curve.append(cum)
        start_ts = rows[0]['exit_timestamp'] if rows else None
        end_ts = rows[-1]['exit_timestamp'] if rows else None
        return {
            'equity_curve': curve,
            'start_time': start_ts.isoformat() if start_ts else None,
            'end_time': end_ts.isoformat() if end_ts else None,
        }

    async def _db_daily_pnl(self, module_name: str) -> dict:
        """DB-fallback daily PnL series."""
        tbl = self._module_tables.get(module_name)
        if not tbl:
            return {'dates': [], 'pnl': [], 'cumulative': []}
        table, pnl_col = tbl
        async with self.db.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT DATE(exit_timestamp) AS d, SUM({pnl_col}) AS p "
                    f"FROM {table} WHERE status='closed' AND exit_timestamp IS NOT NULL "
                    f"GROUP BY d ORDER BY d"
                )
            except Exception as e:
                self.logger.debug(f"_db_daily_pnl {table} error: {e}")
                return {'dates': [], 'pnl': [], 'cumulative': []}
        dates = [r['d'].strftime('%Y-%m-%d') for r in rows]
        pnls = [float(r['p'] or 0) for r in rows]
        cum = []
        running = 0.0
        for p in pnls:
            running += p
            cum.append(running)
        return {'dates': dates, 'pnl': pnls, 'cumulative': cum}

    async def _db_trades(self, module_name: str, limit: int) -> dict:
        """DB-fallback trade history."""
        tbl = self._module_tables.get(module_name)
        if not tbl:
            return {'trades': [], 'total': 0, 'limit': limit, 'offset': 0}
        table, pnl_col = tbl
        async with self.db.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"SELECT trade_id, token_address, entry_price, "
                    f"exit_price, amount, {pnl_col} AS pnl, "
                    f"entry_timestamp, exit_timestamp "
                    f"FROM {table} WHERE status='closed' "
                    f"ORDER BY exit_timestamp DESC NULLS LAST LIMIT $1",
                    int(limit)
                )
            except Exception as e:
                self.logger.debug(f"_db_trades {table} error: {e}")
                return {'trades': [], 'total': 0, 'limit': limit, 'offset': 0}
        out = []
        for r in rows:
            dur = None
            if r['entry_timestamp'] and r['exit_timestamp']:
                dur = int((r['exit_timestamp'] - r['entry_timestamp']).total_seconds())
            out.append({
                'trade_id': r['trade_id'],
                'token': r['token_address'],
                'side': 'BUY',
                'entry_price': float(r['entry_price'] or 0),
                'exit_price': float(r['exit_price'] or 0),
                'size': float(r['amount'] or 0),
                'pnl': float(r['pnl'] or 0),
                'timestamp': r['exit_timestamp'].isoformat() if r['exit_timestamp'] else None,
                'duration_seconds': dur,
            })
        return {'trades': out, 'total': len(out), 'limit': limit, 'offset': 0}

    async def _db_portfolio(self) -> dict:
        """DB-fallback portfolio summary across all module tables."""
        if not self._has_db():
            return {
                'total_pnl': 0.0, 'total_trades': 0,
                'active_modules': 0, 'best_performer': None,
            }
        total_pnl = 0.0
        total_trades = 0
        per_module = {}
        async with self.db.pool.acquire() as conn:
            for module_name, (table, pnl_col) in self._module_tables.items():
                try:
                    row = await conn.fetchrow(
                        f"SELECT COUNT(*) AS n, COALESCE(SUM({pnl_col}), 0) AS p "
                        f"FROM {table} WHERE status='closed'"
                    )
                    if row:
                        n = int(row['n'] or 0)
                        p = float(row['p'] or 0)
                        per_module[module_name] = p
                        total_trades += n
                        total_pnl += p
                except Exception as e:
                    self.logger.debug(f"_db_portfolio {table} skipped: {e}")
        best = None
        if per_module:
            best = max(per_module.items(), key=lambda kv: kv[1])[0]
        active = sum(1 for v in per_module.values() if v != 0)
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'active_modules': active,
            'best_performer': best,
        }

    async def analytics_page(self, request: web.Request) -> web.Response:
        """Render analytics dashboard page.

        Fail-soft when self.analytics is None: still render the template
        with summary=None and a flag so the JS can decide what to show.
        Previously the page wouldn't even register without an engine; now
        it always serves so operators can see the static layout and the
        JS-side empty states.
        """
        try:
            if not self.jinja_env:
                return web.Response(text="Analytics dashboard not configured", status=500)

            summary = None
            if self.analytics is not None:
                try:
                    summary = await self.analytics.get_portfolio_summary()
                except Exception as e:
                    self.logger.warning(f"analytics.get_portfolio_summary failed: {e}")

            template = self.jinja_env.get_template('analytics.html')
            html = template.render(
                summary=summary,
                analytics_available=(self.analytics is not None),
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
            # FAILURE B: DB fallback when no analytics_engine. Lets the
            # standalone dashboard subprocess serve real numbers.
            if not self._has_engine():
                if self._has_db():
                    return json_response({'success': True, 'data': await self._db_perf(module_name)})
                guard = self._require_engine()
                if guard is not None:
                    return guard
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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting performance: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_risk(self, request: web.Request) -> web.Response:
        """Get risk metrics for a module"""
        try:
            module_name = request.match_info['module']
            # FAILURE B: risk needs live state we don't have in the
            # standalone dashboard — return a zero'd shape from DB
            # totals so the UI labels render correctly instead of 503ing.
            if not self._has_engine():
                if self._has_db():
                    perf = await self._db_perf(module_name)
                    return json_response({'success': True, 'data': {
                        'module_name': module_name,
                        'total_exposure': 0.0, 'long_exposure': 0.0,
                        'short_exposure': 0.0, 'net_exposure': 0.0,
                        'largest_position_pct': 0.0, 'top_5_positions_pct': 0.0,
                        'var_95': 0.0, 'var_99': 0.0, 'cvar_95': 0.0,
                        'daily_volatility': 0.0, 'annual_volatility': 0.0,
                        'avg_leverage': 1.0, 'max_leverage': 1.0,
                        'avg_liquidity_score': 0.0, 'low_liquidity_positions': 0,
                    }})
                guard = self._require_engine()
                if guard is not None:
                    return guard

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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_comparison(self, request: web.Request) -> web.Response:
        """Get module comparison"""
        try:
            # FAILURE B: serve from DB when no engine.
            if not self._has_engine():
                if self._has_db():
                    pf = await self._db_portfolio()
                    return json_response({'success': True, 'data': {
                        'best_performer': pf['best_performer'],
                        'worst_performer': None, 'most_active': None,
                        'least_active': None, 'highest_sharpe': None,
                        'lowest_drawdown': None, 'rankings': [],
                    }})
                guard = self._require_engine()
                if guard is not None:
                    return guard
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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting comparison: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_portfolio(self, request: web.Request) -> web.Response:
        """Get portfolio summary"""
        try:
            # FAILURE B: serve from DB when no engine.
            if not self._has_engine():
                if self._has_db():
                    return json_response({'success': True, 'data': await self._db_portfolio()})
                guard = self._require_engine()
                if guard is not None:
                    return guard
            summary = await self.analytics.get_portfolio_summary()
            return json_response({'success': True, 'data': summary})

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_equity_curve(self, request: web.Request) -> web.Response:
        """Get equity curve for a module"""
        try:
            module_name = request.match_info['module']
            # FAILURE B: DB fallback.
            if not self._has_engine():
                if self._has_db():
                    return json_response({'success': True, 'data': await self._db_equity(module_name)})
                guard = self._require_engine()
                if guard is not None:
                    return guard
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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting equity curve: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_trade_history(self, request: web.Request) -> web.Response:
        """Get trade history for a module"""
        try:
            module_name = request.match_info['module']
            # Default to 10000 to show all trades (was 100)
            limit = int(request.query.get('limit', 10000))
            # FAILURE B: DB fallback.
            if not self._has_engine():
                if self._has_db():
                    return json_response({'success': True, 'data': await self._db_trades(module_name, limit)})
                guard = self._require_engine()
                if guard is not None:
                    return guard
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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)

    async def get_daily_pnl(self, request: web.Request) -> web.Response:
        """Get daily PnL for a module"""
        try:
            module_name = request.match_info['module']
            # FAILURE B: DB fallback.
            if not self._has_engine():
                if self._has_db():
                    return json_response({'success': True, 'data': await self._db_daily_pnl(module_name)})
                guard = self._require_engine()
                if guard is not None:
                    return guard
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

            return json_response({'success': True, 'data': data})

        except Exception as e:
            self.logger.error(f"Error getting daily PnL: {e}", exc_info=True)
            return json_response({'success': False, 'error': str(e)}, status=500)
