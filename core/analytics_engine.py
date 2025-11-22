"""
Advanced Analytics Engine

Provides comprehensive analytics across all trading modules:
- Performance metrics (win rate, PnL, Sharpe ratio)
- Risk analytics (drawdown, exposure, correlation)
- Cross-module comparison
- Real-time aggregation
- Historical analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger("AnalyticsEngine")


class TimeFrame(Enum):
    """Analytics time frames"""
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_24 = "24h"
    DAY_7 = "7d"
    DAY_30 = "30d"
    ALL = "all"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a module or strategy"""
    module_name: str
    timeframe: TimeFrame

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # PnL metrics
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    # Performance ratios
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in hours

    # Average metrics
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    avg_trade_duration: int = 0  # in seconds
    avg_daily_pnl: Decimal = Decimal("0")

    # Streak metrics
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Best/Worst
    best_trade: Decimal = Decimal("0")
    worst_trade: Decimal = Decimal("0")

    # Volume
    total_volume: Decimal = Decimal("0")
    avg_position_size: Decimal = Decimal("0")

    # Time metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Additional data
    daily_pnl: List[Decimal] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio or module"""
    module_name: str

    # Exposure
    total_exposure: Decimal = Decimal("0")
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")

    # Concentration
    largest_position_pct: float = 0.0
    top_5_positions_pct: float = 0.0

    # Value at Risk
    var_95: Decimal = Decimal("0")  # 95% VaR
    var_99: Decimal = Decimal("0")  # 99% VaR
    cvar_95: Decimal = Decimal("0")  # Conditional VaR

    # Volatility
    daily_volatility: float = 0.0
    annual_volatility: float = 0.0

    # Correlation
    correlation_with_btc: float = 0.0
    correlation_with_eth: float = 0.0

    # Leverage
    avg_leverage: float = 0.0
    max_leverage: float = 0.0

    # Liquidity
    avg_liquidity_score: float = 0.0
    low_liquidity_positions: int = 0


@dataclass
class ModuleComparison:
    """Comparison metrics across modules"""
    best_performer: str = ""
    worst_performer: str = ""
    most_active: str = ""
    least_active: str = ""
    highest_sharpe: str = ""
    lowest_drawdown: str = ""

    module_rankings: Dict[str, int] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AnalyticsEngine:
    """
    Advanced Analytics Engine

    Provides comprehensive analytics across all trading modules:
    - Performance tracking
    - Risk analysis
    - Cross-module comparison
    - Real-time metrics
    - Historical analysis
    """

    def __init__(
        self,
        db_manager=None,
        cache_manager=None,
        module_manager=None
    ):
        """
        Initialize Analytics Engine

        Args:
            db_manager: Database manager
            cache_manager: Cache manager
            module_manager: Module manager
        """
        self.db = db_manager
        self.cache = cache_manager
        self.module_manager = module_manager

        self.logger = logger

        # Cache for analytics
        self.performance_cache: Dict[str, Dict[TimeFrame, PerformanceMetrics]] = defaultdict(dict)
        self.risk_cache: Dict[str, RiskMetrics] = {}
        self.comparison_cache: Optional[ModuleComparison] = None

        # Cache TTL
        self.cache_ttl = {
            TimeFrame.HOUR_1: 60,  # 1 minute
            TimeFrame.HOUR_4: 300,  # 5 minutes
            TimeFrame.HOUR_24: 600,  # 10 minutes
            TimeFrame.DAY_7: 1800,  # 30 minutes
            TimeFrame.DAY_30: 3600,  # 1 hour
            TimeFrame.ALL: 3600
        }

        # Last update times
        self.last_update: Dict[str, datetime] = {}

        self.logger.info("Analytics Engine initialized")

    async def get_module_performance(
        self,
        module_name: str,
        timeframe: TimeFrame = TimeFrame.HOUR_24
    ) -> PerformanceMetrics:
        """
        Get performance metrics for a module

        Args:
            module_name: Name of the module
            timeframe: Time frame for analysis

        Returns:
            PerformanceMetrics: Performance metrics
        """
        try:
            # Check cache
            cache_key = f"{module_name}:{timeframe.value}"
            if cache_key in self.last_update:
                age = (datetime.now() - self.last_update[cache_key]).total_seconds()
                if age < self.cache_ttl[timeframe]:
                    cached = self.performance_cache.get(module_name, {}).get(timeframe)
                    if cached:
                        return cached

            # Calculate metrics
            metrics = await self._calculate_performance_metrics(module_name, timeframe)

            # Cache results
            self.performance_cache[module_name][timeframe] = metrics
            self.last_update[cache_key] = datetime.now()

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting module performance: {e}", exc_info=True)
            return PerformanceMetrics(module_name=module_name, timeframe=timeframe)

    async def _calculate_performance_metrics(
        self,
        module_name: str,
        timeframe: TimeFrame
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a module"""
        try:
            # Get time range
            end_time = datetime.now()
            start_time = self._get_start_time(timeframe, end_time)

            # Get trades from database
            trades = await self._get_trades(module_name, start_time, end_time)

            if not trades:
                return PerformanceMetrics(
                    module_name=module_name,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )

            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = win_count / total_trades if total_trades > 0 else 0.0

            # Calculate PnL
            total_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in trades)
            total_fees = sum(Decimal(str(t.get('fees', 0))) for t in trades)
            net_pnl = total_pnl - total_fees

            # Calculate win/loss averages
            avg_win = (
                sum(Decimal(str(t['pnl'])) for t in winning_trades) / win_count
                if win_count > 0 else Decimal("0")
            )
            avg_loss = (
                sum(Decimal(str(abs(t['pnl']))) for t in losing_trades) / loss_count
                if loss_count > 0 else Decimal("0")
            )

            # Calculate profit factor
            total_wins = sum(t.get('pnl', 0) for t in winning_trades)
            total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            # Calculate equity curve and daily PnL
            equity_curve, daily_pnl = self._calculate_equity_curve(trades, start_time)

            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(daily_pnl)

            # Calculate Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(daily_pnl)

            # Calculate drawdown
            max_dd, current_dd, dd_duration = self._calculate_drawdown(equity_curve)

            # Calculate Calmar ratio
            calmar_ratio = (
                float(total_pnl) / max_dd if max_dd > 0 else 0.0
            )

            # Calculate streaks
            current_streak, max_win_streak, max_loss_streak = self._calculate_streaks(trades)

            # Calculate best/worst trades
            best_trade = max((Decimal(str(t.get('pnl', 0))) for t in trades), default=Decimal("0"))
            worst_trade = min((Decimal(str(t.get('pnl', 0))) for t in trades), default=Decimal("0"))

            # Calculate volume
            total_volume = sum(Decimal(str(t.get('volume', 0))) for t in trades)
            avg_position_size = total_volume / total_trades if total_trades > 0 else Decimal("0")

            # Calculate average trade duration
            durations = [
                (t.get('exit_time', datetime.now()) - t.get('entry_time', datetime.now())).total_seconds()
                for t in trades if t.get('entry_time') and t.get('exit_time')
            ]
            avg_duration = int(sum(durations) / len(durations)) if durations else 0

            # Calculate average daily PnL
            days = (end_time - start_time).days or 1
            avg_daily_pnl = net_pnl / days

            metrics = PerformanceMetrics(
                module_name=module_name,
                timeframe=timeframe,
                total_trades=total_trades,
                winning_trades=win_count,
                losing_trades=loss_count,
                win_rate=win_rate,
                total_pnl=total_pnl,
                realized_pnl=total_pnl,
                total_fees=total_fees,
                net_pnl=net_pnl,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                max_drawdown_duration=dd_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                avg_trade_duration=avg_duration,
                avg_daily_pnl=avg_daily_pnl,
                current_streak=current_streak,
                max_win_streak=max_win_streak,
                max_loss_streak=max_loss_streak,
                best_trade=best_trade,
                worst_trade=worst_trade,
                total_volume=total_volume,
                avg_position_size=avg_position_size,
                start_time=start_time,
                end_time=end_time,
                daily_pnl=daily_pnl,
                equity_curve=equity_curve,
                trade_history=trades
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return PerformanceMetrics(module_name=module_name, timeframe=timeframe)

    async def get_risk_metrics(self, module_name: str) -> RiskMetrics:
        """Get risk metrics for a module"""
        try:
            # Check cache (5 minute TTL for risk metrics)
            cache_key = f"risk:{module_name}"
            if cache_key in self.last_update:
                age = (datetime.now() - self.last_update[cache_key]).total_seconds()
                if age < 300:  # 5 minutes
                    cached = self.risk_cache.get(module_name)
                    if cached:
                        return cached

            # Get module
            if not self.module_manager:
                return RiskMetrics(module_name=module_name)

            module = self.module_manager.modules.get(module_name)
            if not module:
                return RiskMetrics(module_name=module_name)

            # Get current positions
            positions = await module.get_positions()

            # Calculate exposure
            long_exposure = sum(
                Decimal(str(p.get('size', 0)))
                for p in positions
                if p.get('side', 'LONG') == 'LONG'
            )
            short_exposure = sum(
                Decimal(str(p.get('size', 0)))
                for p in positions
                if p.get('side') == 'SHORT'
            )
            total_exposure = long_exposure + short_exposure
            net_exposure = long_exposure - short_exposure

            # Calculate concentration
            position_sizes = [Decimal(str(p.get('size', 0))) for p in positions]
            largest_position = max(position_sizes) if position_sizes else Decimal("0")
            largest_position_pct = (
                float(largest_position / total_exposure * 100)
                if total_exposure > 0 else 0.0
            )

            # Calculate top 5 positions
            top_5 = sorted(position_sizes, reverse=True)[:5]
            top_5_total = sum(top_5)
            top_5_pct = (
                float(top_5_total / total_exposure * 100)
                if total_exposure > 0 else 0.0
            )

            # Get historical PnL for VaR calculation
            trades = await self._get_trades(module_name, datetime.now() - timedelta(days=30), datetime.now())
            daily_returns = self._calculate_daily_returns(trades)

            # Calculate VaR
            var_95 = self._calculate_var(daily_returns, 0.95)
            var_99 = self._calculate_var(daily_returns, 0.99)
            cvar_95 = self._calculate_cvar(daily_returns, 0.95)

            # Calculate volatility
            daily_vol = float(np.std(daily_returns)) if daily_returns else 0.0
            annual_vol = daily_vol * np.sqrt(252)

            # Calculate average leverage
            leverages = [p.get('leverage', 1) for p in positions]
            avg_leverage = sum(leverages) / len(leverages) if leverages else 0.0
            max_leverage = max(leverages) if leverages else 0.0

            # Calculate liquidity score
            liquidity_scores = [p.get('liquidity_score', 0) for p in positions]
            avg_liquidity = sum(liquidity_scores) / len(liquidity_scores) if liquidity_scores else 0.0
            low_liq_count = sum(1 for score in liquidity_scores if score < 50)

            metrics = RiskMetrics(
                module_name=module_name,
                total_exposure=total_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                net_exposure=net_exposure,
                largest_position_pct=largest_position_pct,
                top_5_positions_pct=top_5_pct,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                daily_volatility=daily_vol,
                annual_volatility=annual_vol,
                avg_leverage=avg_leverage,
                max_leverage=max_leverage,
                avg_liquidity_score=avg_liquidity,
                low_liquidity_positions=low_liq_count
            )

            # Cache results
            self.risk_cache[module_name] = metrics
            self.last_update[cache_key] = datetime.now()

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}", exc_info=True)
            return RiskMetrics(module_name=module_name)

    async def compare_modules(self) -> ModuleComparison:
        """Compare performance across all modules"""
        try:
            if not self.module_manager:
                return ModuleComparison()

            module_names = list(self.module_manager.modules.keys())
            if not module_names:
                return ModuleComparison()

            # Get performance for all modules
            performances = {}
            for name in module_names:
                perf = await self.get_module_performance(name, TimeFrame.DAY_7)
                performances[name] = perf

            # Find best/worst performers
            sorted_by_pnl = sorted(
                performances.items(),
                key=lambda x: x[1].net_pnl,
                reverse=True
            )
            best_performer = sorted_by_pnl[0][0] if sorted_by_pnl else ""
            worst_performer = sorted_by_pnl[-1][0] if sorted_by_pnl else ""

            # Most/least active
            sorted_by_trades = sorted(
                performances.items(),
                key=lambda x: x[1].total_trades,
                reverse=True
            )
            most_active = sorted_by_trades[0][0] if sorted_by_trades else ""
            least_active = sorted_by_trades[-1][0] if sorted_by_trades else ""

            # Highest Sharpe
            sorted_by_sharpe = sorted(
                performances.items(),
                key=lambda x: x[1].sharpe_ratio,
                reverse=True
            )
            highest_sharpe = sorted_by_sharpe[0][0] if sorted_by_sharpe else ""

            # Lowest drawdown
            sorted_by_dd = sorted(
                performances.items(),
                key=lambda x: x[1].max_drawdown
            )
            lowest_drawdown = sorted_by_dd[0][0] if sorted_by_dd else ""

            # Create rankings
            rankings = {name: i+1 for i, (name, _) in enumerate(sorted_by_pnl)}

            comparison = ModuleComparison(
                best_performer=best_performer,
                worst_performer=worst_performer,
                most_active=most_active,
                least_active=least_active,
                highest_sharpe=highest_sharpe,
                lowest_drawdown=lowest_drawdown,
                module_rankings=rankings
            )

            self.comparison_cache = comparison
            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing modules: {e}", exc_info=True)
            return ModuleComparison()

    def _get_start_time(self, timeframe: TimeFrame, end_time: datetime) -> datetime:
        """Get start time for a timeframe"""
        if timeframe == TimeFrame.HOUR_1:
            return end_time - timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            return end_time - timedelta(hours=4)
        elif timeframe == TimeFrame.HOUR_24:
            return end_time - timedelta(hours=24)
        elif timeframe == TimeFrame.DAY_7:
            return end_time - timedelta(days=7)
        elif timeframe == TimeFrame.DAY_30:
            return end_time - timedelta(days=30)
        else:  # ALL
            return datetime(2020, 1, 1)  # Far past

    async def _get_trades(
        self,
        module_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """Get trades from database"""
        try:
            if not self.db:
                return []

            query = """
                SELECT * FROM trades
                WHERE module = %s
                AND timestamp >= %s
                AND timestamp <= %s
                ORDER BY timestamp ASC
            """

            result = await self.db.fetch_all(
                query,
                (module_name, start_time, end_time)
            )

            return [dict(row) for row in result]

        except Exception as e:
            self.logger.error(f"Error fetching trades: {e}")
            return []

    def _calculate_equity_curve(
        self,
        trades: List[Dict],
        start_time: datetime
    ) -> Tuple[List[float], List[Decimal]]:
        """Calculate equity curve and daily PnL"""
        if not trades:
            return [], []

        # Sort trades by time
        sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', start_time))

        # Calculate running equity
        equity = []
        cumulative = 0.0

        for trade in sorted_trades:
            pnl = trade.get('pnl', 0)
            cumulative += pnl
            equity.append(cumulative)

        # Calculate daily PnL
        daily_pnl = []
        current_day = start_time.date()
        day_pnl = Decimal("0")

        for trade in sorted_trades:
            trade_date = trade.get('timestamp', start_time).date()

            if trade_date != current_day:
                daily_pnl.append(day_pnl)
                current_day = trade_date
                day_pnl = Decimal("0")

            day_pnl += Decimal(str(trade.get('pnl', 0)))

        if day_pnl != Decimal("0"):
            daily_pnl.append(day_pnl)

        return equity, daily_pnl

    def _calculate_sharpe_ratio(
        self,
        daily_returns: List[Decimal],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0

        returns_array = np.array([float(r) for r in daily_returns])
        excess_returns = returns_array - (risk_free_rate / 252)

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return float(sharpe * np.sqrt(252))

    def _calculate_sortino_ratio(
        self,
        daily_returns: List[Decimal],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0

        returns_array = np.array([float(r) for r in daily_returns])
        excess_returns = returns_array - (risk_free_rate / 252)

        # Calculate downside deviation
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return 0.0

        downside_dev = np.std(downside_returns)
        if downside_dev == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_dev
        return float(sortino * np.sqrt(252))

    def _calculate_drawdown(
        self,
        equity_curve: List[float]
    ) -> Tuple[float, float, int]:
        """
        Calculate max drawdown, current drawdown, and duration

        Returns:
            (max_drawdown, current_drawdown, duration_hours)
        """
        if not equity_curve:
            return 0.0, 0.0, 0

        peak = equity_curve[0]
        max_dd = 0.0
        current_dd = 0.0
        dd_duration = 0
        current_duration = 0

        for value in equity_curve:
            if value > peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1

            drawdown = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, drawdown)
            current_dd = (peak - equity_curve[-1]) / peak if peak != 0 else 0
            dd_duration = max(dd_duration, current_duration)

        return max_dd, current_dd, dd_duration

    def _calculate_streaks(self, trades: List[Dict]) -> Tuple[int, int, int]:
        """Calculate current streak, max win streak, max loss streak"""
        if not trades:
            return 0, 0, 0

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0
        last_result = None

        for trade in sorted(trades, key=lambda t: t.get('timestamp', datetime.now())):
            pnl = trade.get('pnl', 0)
            is_win = pnl > 0

            if last_result is None:
                temp_streak = 1
            elif last_result == is_win:
                temp_streak += 1
            else:
                temp_streak = 1

            if is_win:
                max_win_streak = max(max_win_streak, temp_streak)
            else:
                max_loss_streak = max(max_loss_streak, temp_streak)

            last_result = is_win
            current_streak = temp_streak if is_win else -temp_streak

        return current_streak, max_win_streak, max_loss_streak

    def _calculate_daily_returns(self, trades: List[Dict]) -> List[float]:
        """Calculate daily returns from trades"""
        if not trades:
            return []

        daily_pnl = defaultdict(float)

        for trade in trades:
            trade_date = trade.get('timestamp', datetime.now()).date()
            daily_pnl[trade_date] += trade.get('pnl', 0)

        return list(daily_pnl.values())

    def _calculate_var(self, returns: List[float], confidence: float) -> Decimal:
        """Calculate Value at Risk"""
        if not returns:
            return Decimal("0")

        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = abs(sorted_returns[index]) if index < len(sorted_returns) else 0

        return Decimal(str(var))

    def _calculate_cvar(self, returns: List[float], confidence: float) -> Decimal:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if not returns:
            return Decimal("0")

        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))

        tail_returns = sorted_returns[:index] if index > 0 else sorted_returns[:1]
        cvar = abs(sum(tail_returns) / len(tail_returns)) if tail_returns else 0

        return Decimal(str(cvar))

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get overall portfolio summary"""
        try:
            if not self.module_manager:
                return {}

            # Get all module performances
            module_perfs = {}
            total_pnl = Decimal("0")
            total_trades = 0

            for module_name in self.module_manager.modules.keys():
                perf = await self.get_module_performance(module_name, TimeFrame.DAY_7)
                module_perfs[module_name] = perf
                total_pnl += perf.net_pnl
                total_trades += perf.total_trades

            # Get module comparison
            comparison = await self.compare_modules()

            summary = {
                "total_pnl": float(total_pnl),
                "total_trades": total_trades,
                "active_modules": len([
                    m for m, p in module_perfs.items()
                    if p.total_trades > 0
                ]),
                "best_performer": comparison.best_performer,
                "worst_performer": comparison.worst_performer,
                "module_performances": {
                    name: {
                        "pnl": float(perf.net_pnl),
                        "trades": perf.total_trades,
                        "win_rate": perf.win_rate,
                        "sharpe": perf.sharpe_ratio
                    }
                    for name, perf in module_perfs.items()
                },
                "rankings": comparison.module_rankings
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}", exc_info=True)
            return {}
