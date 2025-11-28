"""
Real-time PnL Tracker
Lightweight performance tracking with Sharpe/Sortino calculations
Designed to be embedded in trading engines for real-time metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from collections import deque
import math

logger = logging.getLogger("PnLTracker")


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    symbol: str
    side: str  # 'long' or 'short' or 'buy' or 'sell'
    entry_price: float
    exit_price: float
    size: float
    pnl: float  # Gross PnL
    fees: float
    net_pnl: float  # PnL after fees
    pnl_pct: float  # Percentage return
    entry_time: datetime
    exit_time: datetime
    duration_seconds: int
    is_simulated: bool = False


@dataclass
class DailyStats:
    """Daily performance statistics"""
    date: str  # YYYY-MM-DD
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    volume: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Current performance snapshot"""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # PnL
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0

    # Averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    avg_trade_duration: float = 0.0  # seconds

    # Streaks
    current_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Best/Worst
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Time metrics
    trading_days: int = 0
    avg_daily_pnl: float = 0.0

    # Daily returns for charts
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class PnLTracker:
    """
    Lightweight PnL tracker for real-time performance monitoring

    Features:
    - Real-time Sharpe/Sortino ratio calculation
    - Rolling window metrics (7-day, 30-day)
    - Fee tracking
    - Drawdown calculation
    - Trade history
    """

    # Risk-free rate (annualized, e.g., 2%)
    RISK_FREE_RATE = 0.02
    TRADING_DAYS_PER_YEAR = 365  # Crypto trades 365 days

    def __init__(
        self,
        initial_capital: float = 0.0,
        max_history: int = 10000,
        currency: str = "USD"
    ):
        """
        Initialize PnL tracker

        Args:
            initial_capital: Starting capital (for drawdown calculations)
            max_history: Maximum trade records to keep
            currency: Base currency (USD, SOL, etc.)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.currency = currency

        # Trade history (limited to max_history)
        self.trades: deque = deque(maxlen=max_history)

        # Daily stats
        self.daily_stats: Dict[str, DailyStats] = {}

        # Daily returns for Sharpe/Sortino
        self._daily_pnl: Dict[str, float] = {}  # date -> net_pnl

        # Running totals
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_pnl = 0.0
        self.total_fees = 0.0
        self.net_pnl = 0.0

        # Win/loss totals for profit factor
        self.total_wins = 0.0
        self.total_losses = 0.0

        # Drawdown tracking
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

        # Streak tracking
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

        # Best/Worst trades
        self.best_trade = 0.0
        self.worst_trade = 0.0

        # Trade duration tracking
        self._total_duration = 0.0

        # Unrealized PnL (updated externally)
        self.unrealized_pnl = 0.0

        logger.info(f"PnL Tracker initialized (currency: {currency}, capital: {initial_capital})")

    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade

        Args:
            trade: TradeRecord with trade details
        """
        # Add to history
        self.trades.append(trade)

        # Update totals
        self.total_trades += 1
        self.gross_pnl += trade.pnl
        self.total_fees += trade.fees
        self.net_pnl += trade.net_pnl

        # Update capital
        self.current_capital += trade.net_pnl

        # Track peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        current_dd = self.peak_capital - self.current_capital
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
            if self.peak_capital > 0:
                self.max_drawdown_pct = (current_dd / self.peak_capital) * 100

        # Win/Loss tracking
        if trade.net_pnl > 0:
            self.winning_trades += 1
            self.total_wins += trade.net_pnl
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            if self.current_streak > self.max_win_streak:
                self.max_win_streak = self.current_streak
        elif trade.net_pnl < 0:
            self.losing_trades += 1
            self.total_losses += abs(trade.net_pnl)
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            if abs(self.current_streak) > self.max_loss_streak:
                self.max_loss_streak = abs(self.current_streak)

        # Best/Worst trade
        if trade.net_pnl > self.best_trade:
            self.best_trade = trade.net_pnl
        if trade.net_pnl < self.worst_trade:
            self.worst_trade = trade.net_pnl

        # Trade duration
        self._total_duration += trade.duration_seconds

        # Daily stats
        date_str = trade.exit_time.strftime('%Y-%m-%d')
        if date_str not in self.daily_stats:
            self.daily_stats[date_str] = DailyStats(date=date_str)

        daily = self.daily_stats[date_str]
        daily.trades += 1
        daily.gross_pnl += trade.pnl
        daily.fees += trade.fees
        daily.net_pnl += trade.net_pnl
        daily.volume += abs(trade.size * trade.entry_price)
        if trade.net_pnl > 0:
            daily.wins += 1
        elif trade.net_pnl < 0:
            daily.losses += 1

        # Daily PnL for Sharpe calculation
        if date_str not in self._daily_pnl:
            self._daily_pnl[date_str] = 0.0
        self._daily_pnl[date_str] += trade.net_pnl

        logger.debug(f"Trade recorded: {trade.symbol} PnL: {trade.net_pnl:.2f} {self.currency}")

    def update_unrealized_pnl(self, unrealized: float):
        """Update unrealized PnL from open positions"""
        self.unrealized_pnl = unrealized

    def get_sharpe_ratio(self, days: int = 30) -> float:
        """
        Calculate Sharpe ratio over a period

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Args:
            days: Number of days to calculate over

        Returns:
            Annualized Sharpe ratio
        """
        daily_returns = self._get_daily_returns(days)
        if len(daily_returns) < 2:
            return 0.0

        mean_return = sum(daily_returns) / len(daily_returns)
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        # Annualize Sharpe ratio
        sharpe = (mean_return - daily_rf) / std_dev
        annualized_sharpe = sharpe * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return round(annualized_sharpe, 2)

    def get_sortino_ratio(self, days: int = 30) -> float:
        """
        Calculate Sortino ratio over a period

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

        Uses only negative returns in denominator (penalizes downside risk only)

        Args:
            days: Number of days to calculate over

        Returns:
            Annualized Sortino ratio
        """
        daily_returns = self._get_daily_returns(days)
        if len(daily_returns) < 2:
            return 0.0

        mean_return = sum(daily_returns) / len(daily_returns)
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in daily_returns if r < 0]
        if not negative_returns:
            # No negative returns = infinite Sortino (cap at reasonable value)
            return 10.0

        downside_variance = sum(r ** 2 for r in negative_returns) / len(daily_returns)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0

        if downside_std == 0:
            return 0.0

        # Annualize Sortino ratio
        sortino = (mean_return - daily_rf) / downside_std
        annualized_sortino = sortino * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return round(annualized_sortino, 2)

    def get_calmar_ratio(self, days: int = 365) -> float:
        """
        Calculate Calmar ratio

        Calmar = Annualized Return / Max Drawdown

        Args:
            days: Number of days to calculate over

        Returns:
            Calmar ratio
        """
        if self.max_drawdown == 0:
            return 0.0

        daily_returns = self._get_daily_returns(days)
        if not daily_returns:
            return 0.0

        total_return = sum(daily_returns)
        annualized_return = (total_return / len(daily_returns)) * self.TRADING_DAYS_PER_YEAR

        calmar = annualized_return / self.max_drawdown

        return round(calmar, 2)

    def get_profit_factor(self) -> float:
        """
        Calculate profit factor

        Profit Factor = Total Wins / Total Losses

        Returns:
            Profit factor (> 1 is profitable)
        """
        if self.total_losses == 0:
            return float('inf') if self.total_wins > 0 else 0.0

        return round(self.total_wins / self.total_losses, 2)

    def get_win_rate(self) -> float:
        """Get win rate as percentage"""
        if self.total_trades == 0:
            return 0.0
        return round((self.winning_trades / self.total_trades) * 100, 1)

    def get_current_drawdown(self) -> Tuple[float, float]:
        """
        Get current drawdown (absolute and percentage)

        Returns:
            (absolute_drawdown, percentage_drawdown)
        """
        if self.peak_capital == 0:
            return 0.0, 0.0

        current_dd = self.peak_capital - self.current_capital
        current_dd_pct = (current_dd / self.peak_capital) * 100 if self.peak_capital > 0 else 0.0

        return round(current_dd, 2), round(current_dd_pct, 2)

    def get_avg_trade_duration(self) -> float:
        """Get average trade duration in seconds"""
        if self.total_trades == 0:
            return 0.0
        return self._total_duration / self.total_trades

    def get_snapshot(self) -> PerformanceSnapshot:
        """
        Get complete performance snapshot

        Returns:
            PerformanceSnapshot with all metrics
        """
        current_dd, current_dd_pct = self.get_current_drawdown()

        # Calculate averages
        avg_win = self.total_wins / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.total_losses / self.losing_trades if self.losing_trades > 0 else 0.0
        avg_trade = self.net_pnl / self.total_trades if self.total_trades > 0 else 0.0

        # Get daily returns for chart
        daily_returns = list(self._daily_pnl.values())

        # Build equity curve
        equity_curve = [self.initial_capital]
        running_total = self.initial_capital
        for pnl in daily_returns:
            running_total += pnl
            equity_curve.append(running_total)

        # Trading days
        trading_days = len(self.daily_stats)
        avg_daily = self.net_pnl / trading_days if trading_days > 0 else 0.0

        return PerformanceSnapshot(
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            win_rate=self.get_win_rate(),
            gross_pnl=round(self.gross_pnl, 2),
            total_fees=round(self.total_fees, 2),
            net_pnl=round(self.net_pnl, 2),
            realized_pnl=round(self.net_pnl, 2),
            unrealized_pnl=round(self.unrealized_pnl, 2),
            sharpe_ratio=self.get_sharpe_ratio(),
            sortino_ratio=self.get_sortino_ratio(),
            calmar_ratio=self.get_calmar_ratio(),
            profit_factor=self.get_profit_factor(),
            max_drawdown=round(self.max_drawdown, 2),
            max_drawdown_pct=round(self.max_drawdown_pct, 2),
            current_drawdown=current_dd,
            current_drawdown_pct=current_dd_pct,
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_trade=round(avg_trade, 2),
            avg_trade_duration=self.get_avg_trade_duration(),
            current_streak=self.current_streak,
            max_win_streak=self.max_win_streak,
            max_loss_streak=self.max_loss_streak,
            best_trade=round(self.best_trade, 2),
            worst_trade=round(self.worst_trade, 2),
            trading_days=trading_days,
            avg_daily_pnl=round(avg_daily, 2),
            daily_returns=daily_returns[-30:],  # Last 30 days
            equity_curve=equity_curve[-100:]  # Last 100 points
        )

    def get_daily_stats(self, date: Optional[str] = None) -> Optional[DailyStats]:
        """
        Get stats for a specific day

        Args:
            date: Date string (YYYY-MM-DD) or None for today

        Returns:
            DailyStats or None if no data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        return self.daily_stats.get(date)

    def get_recent_trades(self, limit: int = 20) -> List[TradeRecord]:
        """Get most recent trades"""
        trades_list = list(self.trades)
        return trades_list[-limit:] if trades_list else []

    def _get_daily_returns(self, days: int) -> List[float]:
        """Get daily returns for last N days as percentages"""
        if not self._daily_pnl:
            return []

        # Sort by date and get last N days
        sorted_dates = sorted(self._daily_pnl.keys())[-days:]

        returns = []
        for date in sorted_dates:
            pnl = self._daily_pnl[date]
            # Convert to percentage return (relative to initial capital)
            if self.initial_capital > 0:
                returns.append(pnl / self.initial_capital)
            else:
                returns.append(pnl)

        return returns

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        snapshot = self.get_snapshot()
        return {
            'total_trades': snapshot.total_trades,
            'winning_trades': snapshot.winning_trades,
            'losing_trades': snapshot.losing_trades,
            'win_rate': f"{snapshot.win_rate}%",
            'gross_pnl': snapshot.gross_pnl,
            'total_fees': snapshot.total_fees,
            'net_pnl': snapshot.net_pnl,
            'unrealized_pnl': snapshot.unrealized_pnl,
            'sharpe_ratio': snapshot.sharpe_ratio,
            'sortino_ratio': snapshot.sortino_ratio,
            'calmar_ratio': snapshot.calmar_ratio,
            'profit_factor': snapshot.profit_factor,
            'max_drawdown': snapshot.max_drawdown,
            'max_drawdown_pct': f"{snapshot.max_drawdown_pct}%",
            'current_drawdown': snapshot.current_drawdown,
            'avg_win': snapshot.avg_win,
            'avg_loss': snapshot.avg_loss,
            'avg_trade': snapshot.avg_trade,
            'best_trade': snapshot.best_trade,
            'worst_trade': snapshot.worst_trade,
            'current_streak': snapshot.current_streak,
            'max_win_streak': snapshot.max_win_streak,
            'max_loss_streak': snapshot.max_loss_streak,
            'trading_days': snapshot.trading_days,
            'avg_daily_pnl': snapshot.avg_daily_pnl,
            'currency': self.currency
        }

    def reset(self):
        """Reset all tracking data"""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trades.clear()
        self.daily_stats.clear()
        self._daily_pnl.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_pnl = 0.0
        self.total_fees = 0.0
        self.net_pnl = 0.0
        self.total_wins = 0.0
        self.total_losses = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.current_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.best_trade = 0.0
        self.worst_trade = 0.0
        self._total_duration = 0.0
        self.unrealized_pnl = 0.0
        logger.info("PnL Tracker reset")
