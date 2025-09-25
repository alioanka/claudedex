"""
Performance Tracker and Logger for DexScreener Trading Bot
Comprehensive performance analytics and structured logging system
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from decimal import Decimal
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict, deque
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import traceback

# Custom log levels
TRADE_LOG = 25  # Between INFO and WARNING
PROFIT_LOG = 26
LOSS_LOG = 27

logging.addLevelName(TRADE_LOG, "TRADE")
logging.addLevelName(PROFIT_LOG, "PROFIT")
logging.addLevelName(LOSS_LOG, "LOSS")

class MetricType(Enum):
    """Types of performance metrics"""
    RETURN = "return"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    VOLUME = "volume"
    WIN_RATE = "win_rate"
    DRAWDOWN = "drawdown"

@dataclass
class TradeRecord:
    """Complete trade record"""
    trade_id: str
    timestamp: datetime
    token_address: str
    token_symbol: str
    strategy: str
    signal_id: Optional[str]
    
    # Entry
    entry_time: datetime
    entry_price: Decimal
    entry_amount: Decimal
    entry_value: Decimal
    entry_fees: Decimal
    
    # Exit
    exit_time: datetime
    exit_price: Decimal
    exit_amount: Decimal
    exit_value: Decimal
    exit_fees: Decimal
    
    # Performance
    pnl: Decimal
    pnl_percent: float
    roi: float
    holding_period: timedelta
    max_profit: Decimal
    max_loss: Decimal
    
    # Risk metrics
    risk_score: float
    sharpe_contribution: float
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    period: str  # hourly, daily, weekly, monthly
    
    # Returns
    period_return: float
    cumulative_return: float
    
    # P&L
    period_pnl: Decimal
    cumulative_pnl: Decimal
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: Decimal
    
    # Volume metrics
    volume_traded: Decimal
    fees_paid: Decimal
    
    # Efficiency
    avg_holding_time: Optional[timedelta]
    trades_per_day: float
    
    # By strategy breakdown
    strategy_performance: Dict[str, Dict] = field(default_factory=dict)

class PerformanceTracker:
    """
    Comprehensive performance tracking and analytics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance tracker"""
        self.config = config or self._default_config()
        
        # Trade records
        self.trades: List[TradeRecord] = []
        self.trades_by_date: Dict[date, List[TradeRecord]] = defaultdict(list)
        self.trades_by_strategy: Dict[str, List[TradeRecord]] = defaultdict(list)
        
        # Performance snapshots
        self.hourly_snapshots: deque = deque(maxlen=24*7)  # 1 week
        self.daily_snapshots: deque = deque(maxlen=365)    # 1 year
        self.weekly_snapshots: deque = deque(maxlen=52)    # 1 year
        self.monthly_snapshots: deque = deque(maxlen=24)   # 2 years
        
        # Real-time metrics
        self.current_metrics = self._initialize_metrics()
        
        # Database
        self.db_path = Path(self.config["db_path"])
        self._init_database()
        
        # Start background tasks
        asyncio.create_task(self._calculate_snapshots())
        asyncio.create_task(self._persist_data())
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "db_path": "data/performance.db",
            "snapshot_intervals": {
                "hourly": 3600,
                "daily": 86400,
                "weekly": 604800,
                "monthly": 2592000
            },
            "persist_interval": 300,  # 5 minutes
            "benchmark_returns": 0.0,  # Risk-free rate for Sharpe
            "min_trades_for_stats": 10,
            "outlier_threshold": 3.0,  # Standard deviations
        }
    
    def _initialize_metrics(self) -> Dict:
        """Initialize current metrics"""
        return {
            "total_trades": 0,
            "open_trades": 0,
            "total_pnl": Decimal("0"),
            "total_fees": Decimal("0"),
            "total_volume": Decimal("0"),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "peak_value": Decimal("10000"),  # Starting capital
            "current_value": Decimal("10000"),
            "best_trade": None,
            "worst_trade": None,
            "longest_winning_streak": 0,
            "longest_losing_streak": 0,
            "current_streak": 0,
            "avg_win": Decimal("0"),
            "avg_loss": Decimal("0"),
            "largest_win": Decimal("0"),
            "largest_loss": Decimal("0"),
            "avg_holding_time": timedelta(0),
            "strategy_metrics": {}
        }
    
    def _init_database(self):
        """Initialize SQLite database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT,
                token_address TEXT,
                token_symbol TEXT,
                strategy TEXT,
                signal_id TEXT,
                entry_time TEXT,
                entry_price REAL,
                entry_amount REAL,
                entry_value REAL,
                entry_fees REAL,
                exit_time TEXT,
                exit_price REAL,
                exit_amount REAL,
                exit_value REAL,
                exit_fees REAL,
                pnl REAL,
                pnl_percent REAL,
                roi REAL,
                holding_period_seconds INTEGER,
                max_profit REAL,
                max_loss REAL,
                risk_score REAL,
                sharpe_contribution REAL,
                metadata TEXT
            )
        """)
        
        # Create snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                timestamp TEXT,
                period TEXT,
                period_return REAL,
                cumulative_return REAL,
                period_pnl REAL,
                cumulative_pnl REAL,
                volatility REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                calmar_ratio REAL,
                max_drawdown REAL,
                var_95 REAL,
                cvar_95 REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                expectancy REAL,
                volume_traded REAL,
                fees_paid REAL,
                avg_holding_seconds INTEGER,
                trades_per_day REAL,
                strategy_performance TEXT,
                PRIMARY KEY (timestamp, period)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_period ON snapshots(period)")
        
        conn.commit()
        conn.close()
    
    def record_trade(self, trade: TradeRecord):
        """Record completed trade"""
        try:
            # Add to memory
            self.trades.append(trade)
            self.trades_by_date[trade.exit_time.date()].append(trade)
            self.trades_by_strategy[trade.strategy].append(trade)
            
            # Update metrics
            self._update_metrics(trade)
            
            # Log trade
            self._log_trade(trade)
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _update_metrics(self, trade: TradeRecord):
        """Update real-time metrics with new trade"""
        metrics = self.current_metrics
        
        # Update counts
        metrics["total_trades"] += 1
        metrics["total_pnl"] += trade.pnl
        metrics["total_fees"] += trade.entry_fees + trade.exit_fees
        metrics["total_volume"] += trade.entry_value + trade.exit_value
        
        # Update win/loss stats
        if trade.pnl > 0:
            metrics["win_rate"] = (
                metrics.get("winning_trades", 0) + 1
            ) / metrics["total_trades"]
            
            if metrics["current_streak"] > 0:
                metrics["current_streak"] += 1
            else:
                metrics["current_streak"] = 1
            
            metrics["longest_winning_streak"] = max(
                metrics["longest_winning_streak"],
                metrics["current_streak"]
            )
            
            # Update average win
            wins = [t.pnl for t in self.trades if t.pnl > 0]
            metrics["avg_win"] = sum(wins) / len(wins) if wins else Decimal("0")
            metrics["largest_win"] = max(metrics["largest_win"], trade.pnl)
            
        else:
            if metrics["current_streak"] < 0:
                metrics["current_streak"] -= 1
            else:
                metrics["current_streak"] = -1
            
            metrics["longest_losing_streak"] = max(
                metrics["longest_losing_streak"],
                abs(metrics["current_streak"])
            )
            
            # Update average loss
            losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]
            metrics["avg_loss"] = sum(losses) / len(losses) if losses else Decimal("0")
            metrics["largest_loss"] = min(metrics["largest_loss"], trade.pnl)
        
        # Update best/worst trade
        if not metrics["best_trade"] or trade.roi > metrics["best_trade"].roi:
            metrics["best_trade"] = trade
        
        if not metrics["worst_trade"] or trade.roi < metrics["worst_trade"].roi:
            metrics["worst_trade"] = trade
        
        # Update profit factor
        total_wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        metrics["profit_factor"] = float(total_wins / total_losses) if total_losses > 0 else 0
        
        # Update holding time
        holding_times = [t.holding_period for t in self.trades]
        if holding_times:
            metrics["avg_holding_time"] = sum(holding_times, timedelta()) / len(holding_times)
        
        # Update portfolio value
        metrics["current_value"] += trade.pnl
        
        # Update drawdown
        if metrics["current_value"] > metrics["peak_value"]:
            metrics["peak_value"] = metrics["current_value"]
            metrics["current_drawdown"] = 0
        else:
            metrics["current_drawdown"] = float(
                (metrics["peak_value"] - metrics["current_value"]) / metrics["peak_value"]
            )
            metrics["max_drawdown"] = max(metrics["max_drawdown"], metrics["current_drawdown"])
        
        # Update Sharpe ratio
        if len(self.trades) >= self.config["min_trades_for_stats"]:
            returns = [float(t.pnl / t.entry_value) for t in self.trades]
            if len(returns) > 1:
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                risk_free_rate = self.config["benchmark_returns"]
                
                if std_return > 0:
                    metrics["sharpe_ratio"] = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
        
        # Update strategy metrics
        if trade.strategy not in metrics["strategy_metrics"]:
            metrics["strategy_metrics"][trade.strategy] = {
                "trades": 0,
                "pnl": Decimal("0"),
                "win_rate": 0.0,
                "avg_return": 0.0
            }
        
        strategy_metrics = metrics["strategy_metrics"][trade.strategy]
        strategy_metrics["trades"] += 1
        strategy_metrics["pnl"] += trade.pnl
        
        strategy_trades = self.trades_by_strategy[trade.strategy]
        strategy_wins = sum(1 for t in strategy_trades if t.pnl > 0)
        strategy_metrics["win_rate"] = strategy_wins / len(strategy_trades) if strategy_trades else 0
        
        strategy_returns = [float(t.roi) for t in strategy_trades]
        strategy_metrics["avg_return"] = statistics.mean(strategy_returns) if strategy_returns else 0
    
    def _log_trade(self, trade: TradeRecord):
        """Log trade with custom level"""
        if trade.pnl > 0:
            level = PROFIT_LOG
            emoji = "💰"
        else:
            level = LOSS_LOG
            emoji = "💸"
        
        logger.log(
            level,
            f"{emoji} Trade {trade.trade_id}: {trade.token_symbol} "
            f"P&L: ${trade.pnl:.2f} ({trade.roi:.2%}) "
            f"Strategy: {trade.strategy}"
        )
    
    async def _calculate_snapshots(self):
        """Background task to calculate periodic snapshots"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Hourly snapshot
                if not self.hourly_snapshots or \
                   (current_time - self.hourly_snapshots[-1].timestamp).total_seconds() >= 3600:
                    snapshot = self._create_snapshot("hourly")
                    self.hourly_snapshots.append(snapshot)
                
                # Daily snapshot
                if not self.daily_snapshots or \
                   (current_time - self.daily_snapshots[-1].timestamp).total_seconds() >= 86400:
                    snapshot = self._create_snapshot("daily")
                    self.daily_snapshots.append(snapshot)
                
                # Weekly snapshot
                if not self.weekly_snapshots or \
                   (current_time - self.weekly_snapshots[-1].timestamp).total_seconds() >= 604800:
                    snapshot = self._create_snapshot("weekly")
                    self.weekly_snapshots.append(snapshot)
                
                # Monthly snapshot
                if not self.monthly_snapshots or \
                   (current_time - self.monthly_snapshots[-1].timestamp).total_seconds() >= 2592000:
                    snapshot = self._create_snapshot("monthly")
                    self.monthly_snapshots.append(snapshot)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error calculating snapshots: {e}")
                await asyncio.sleep(3600)
    
    def _create_snapshot(self, period: str) -> PerformanceSnapshot:
        """Create performance snapshot"""
        metrics = self.current_metrics
        
        # Get period trades
        if period == "hourly":
            cutoff = datetime.utcnow() - timedelta(hours=1)
        elif period == "daily":
            cutoff = datetime.utcnow() - timedelta(days=1)
        elif period == "weekly":
            cutoff = datetime.utcnow() - timedelta(weeks=1)
        else:  # monthly
            cutoff = datetime.utcnow() - timedelta(days=30)
        
        period_trades = [t for t in self.trades if t.exit_time >= cutoff]
        
        # Calculate period metrics
        period_pnl = sum(t.pnl for t in period_trades)
        period_return = float(period_pnl / metrics["peak_value"]) if metrics["peak_value"] > 0 else 0
        
        # Risk metrics
        returns = [float(t.roi) for t in period_trades]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_dev = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
        sortino_ratio = (period_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = period_return / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0
        
        # VaR and CVaR
        if returns:
            sorted_returns = sorted(returns)
            var_index = int(len(sorted_returns) * 0.05)
            var_95 = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
            cvar_95 = abs(statistics.mean(sorted_returns[:var_index])) if var_index > 0 else 0
        else:
            var_95 = cvar_95 = 0
        
        # Trading metrics
        period_wins = sum(1 for t in period_trades if t.pnl > 0)
        period_losses = len(period_trades) - period_wins
        period_win_rate = period_wins / len(period_trades) if period_trades else 0
        
        # Profit factor
        period_gross_profit = sum(t.pnl for t in period_trades if t.pnl > 0)
        period_gross_loss = abs(sum(t.pnl for t in period_trades if t.pnl <= 0))
        period_profit_factor = float(period_gross_profit / period_gross_loss) if period_gross_loss > 0 else 0
        
        # Expectancy
        period_expectancy = period_pnl / len(period_trades) if period_trades else Decimal("0")
        
        # Volume metrics
        period_volume = sum(t.entry_value + t.exit_value for t in period_trades)
        period_fees = sum(t.entry_fees + t.exit_fees for t in period_trades)
        
        # Efficiency
        if period_trades:
            avg_holding = sum(
                [t.holding_period for t in period_trades],
                timedelta()
            ) / len(period_trades)
            trades_per_day = len(period_trades) / max(
                (datetime.utcnow() - period_trades[0].exit_time).total_seconds() / 86400,
                1
            )
        else:
            avg_holding = None
            trades_per_day = 0
        
        # Strategy breakdown
        strategy_performance = {}
        for strategy, trades in self.trades_by_strategy.items():
            strategy_period_trades = [t for t in trades if t.exit_time >= cutoff]
            if strategy_period_trades:
                strategy_performance[strategy] = {
                    "trades": len(strategy_period_trades),
                    "pnl": float(sum(t.pnl for t in strategy_period_trades)),
                    "win_rate": sum(1 for t in strategy_period_trades if t.pnl > 0) / len(strategy_period_trades),
                    "avg_return": statistics.mean([t.roi for t in strategy_period_trades])
                }
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            period=period,
            period_return=period_return,
            cumulative_return=float(metrics["total_pnl"] / Decimal("10000")),
            period_pnl=period_pnl,
            cumulative_pnl=metrics["total_pnl"],
            volatility=volatility,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=metrics["max_drawdown"],
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(period_trades),
            winning_trades=period_wins,
            losing_trades=period_losses,
            win_rate=period_win_rate,
            profit_factor=period_profit_factor,
            expectancy=period_expectancy,
            volume_traded=period_volume,
            fees_paid=period_fees,
            avg_holding_time=avg_holding,
            trades_per_day=trades_per_day,
            strategy_performance=strategy_performance
        )
    
    async def _persist_data(self):
        """Persist data to database"""
        while True:
            try:
                await asyncio.sleep(self.config["persist_interval"])
                
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Save recent trades
                for trade in self.trades[-100:]:  # Last 100 trades
                    cursor.execute("""
                        INSERT OR REPLACE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.trade_id,
                        trade.timestamp.isoformat(),
                        trade.token_address,
                        trade.token_symbol,
                        trade.strategy,
                        trade.signal_id,
                        trade.entry_time.isoformat(),
                        float(trade.entry_price),
                        float(trade.entry_amount),
                        float(trade.entry_value),
                        float(trade.entry_fees),
                        trade.exit_time.isoformat(),
                        float(trade.exit_price),
                        float(trade.exit_amount),
                        float(trade.exit_value),
                        float(trade.exit_fees),
                        float(trade.pnl),
                        trade.pnl_percent,
                        trade.roi,
                        int(trade.holding_period.total_seconds()),
                        float(trade.max_profit),
                        float(trade.max_loss),
                        trade.risk_score,
                        trade.sharpe_contribution,
                        json.dumps(trade.metadata)
                    ))
                
                # Save recent snapshots
                all_snapshots = list(self.hourly_snapshots) + list(self.daily_snapshots) + \
                               list(self.weekly_snapshots) + list(self.monthly_snapshots)
                
                for snapshot in all_snapshots:
                    cursor.execute("""
                        INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot.timestamp.isoformat(),
                        snapshot.period,
                        snapshot.period_return,
                        snapshot.cumulative_return,
                        float(snapshot.period_pnl),
                        float(snapshot.cumulative_pnl),
                        snapshot.volatility,
                        snapshot.sharpe_ratio,
                        snapshot.sortino_ratio,
                        snapshot.calmar_ratio,
                        snapshot.max_drawdown,
                        snapshot.var_95,
                        snapshot.cvar_95,
                        snapshot.total_trades,
                        snapshot.winning_trades,
                        snapshot.losing_trades,
                        snapshot.win_rate,
                        snapshot.profit_factor,
                        float(snapshot.expectancy),
                        float(snapshot.volume_traded),
                        float(snapshot.fees_paid),
                        int(snapshot.avg_holding_time.total_seconds()) if snapshot.avg_holding_time else None,
                        snapshot.trades_per_day,
                        json.dumps(snapshot.strategy_performance)
                    ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Error persisting data: {e}")
    
    def get_performance_report(self, period: str = "all") -> Dict:
        """Get comprehensive performance report"""
        if period == "all":
            trades = self.trades
        elif period == "daily":
            trades = [t for t in self.trades if t.exit_time.date() == date.today()]
        elif period == "weekly":
            week_start = date.today() - timedelta(days=date.today().weekday())
            trades = [t for t in self.trades if t.exit_time.date() >= week_start]
        elif period == "monthly":
            month_start = date.today().replace(day=1)
            trades = [t for t in self.trades if t.exit_time.date() >= month_start]
        else:
            trades = self.trades
        
        if not trades:
            return {"error": "No trades in period"}
        
        return {
            "period": period,
            "summary": {
                "total_trades": len(trades),
                "total_pnl": float(sum(t.pnl for t in trades)),
                "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
                "avg_return": statistics.mean([t.roi for t in trades]),
                "sharpe_ratio": self.current_metrics["sharpe_ratio"],
                "max_drawdown": self.current_metrics["max_drawdown"]
            },
            "trades": [
                {
                    "id": t.trade_id,
                    "symbol": t.token_symbol,
                    "pnl": float(t.pnl),
                    "roi": t.roi,
                    "strategy": t.strategy
                }
                for t in trades[-10:]  # Last 10 trades
            ],
            "by_strategy": self.current_metrics["strategy_metrics"]
        }

    async def track_trade(self, trade: Dict) -> None:
        """
        Track a trade (wrapper for record_trade for consistency with documentation)
        
        Args:
            trade: Dictionary containing trade information
        """
        # Convert dict to TradeRecord if needed
        if isinstance(trade, dict):
            trade_record = TradeRecord(
                trade_id=trade.get('trade_id', str(uuid.uuid4())),
                timestamp=trade.get('timestamp', datetime.utcnow()),
                token_address=trade['token_address'],
                token_symbol=trade['token_symbol'],
                strategy=trade.get('strategy', 'unknown'),
                signal_id=trade.get('signal_id'),
                entry_time=trade['entry_time'],
                entry_price=Decimal(str(trade['entry_price'])),
                entry_amount=Decimal(str(trade['entry_amount'])),
                entry_value=Decimal(str(trade['entry_value'])),
                entry_fees=Decimal(str(trade.get('entry_fees', 0))),
                exit_time=trade['exit_time'],
                exit_price=Decimal(str(trade['exit_price'])),
                exit_amount=Decimal(str(trade['exit_amount'])),
                exit_value=Decimal(str(trade['exit_value'])),
                exit_fees=Decimal(str(trade.get('exit_fees', 0))),
                pnl=Decimal(str(trade['pnl'])),
                pnl_percent=trade.get('pnl_percent', 0),
                roi=trade.get('roi', 0),
                holding_period=trade.get('holding_period', timedelta()),
                max_profit=Decimal(str(trade.get('max_profit', 0))),
                max_loss=Decimal(str(trade.get('max_loss', 0))),
                risk_score=trade.get('risk_score', 0),
                sharpe_contribution=trade.get('sharpe_contribution', 0),
                metadata=trade.get('metadata', {})
            )
            self.record_trade(trade_record)
        else:
            self.record_trade(trade)
    
    async def calculate_metrics(self) -> Dict:
        """
        Calculate and return current performance metrics
        
        Returns:
            Dictionary containing all performance metrics
        """
        # Trigger a metrics update if needed
        if self.trades:
            # Recalculate key metrics
            returns = [float(t.pnl / t.entry_value) for t in self.trades if t.entry_value > 0]
            
            if returns:
                # Update Sharpe ratio
                self.current_metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
                
                # Update max drawdown
                equity_curve = []
                cumulative = Decimal("10000")  # Starting capital
                for trade in sorted(self.trades, key=lambda x: x.exit_time):
                    cumulative += trade.pnl
                    equity_curve.append(float(cumulative))
                
                if equity_curve:
                    self.current_metrics["max_drawdown"] = self.calculate_max_drawdown(equity_curve)
        
        return self.current_metrics.copy()
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calculate Sharpe ratio from returns
        
        Args:
            returns: List of period returns
            
        Returns:
            Sharpe ratio (annualized)
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        risk_free_rate = self.config.get("benchmark_returns", 0.0)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: List of portfolio values over time
            
        Returns:
            Maximum drawdown as a percentage
        """
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def generate_report(self) -> Dict:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary containing detailed performance report
        """
        # Calculate current metrics
        metrics = await self.calculate_metrics()
        
        # Get snapshots for different periods
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "current_metrics": metrics,
            "performance_by_period": {}
        }
        
        # Add hourly performance
        if self.hourly_snapshots:
            latest_hourly = self.hourly_snapshots[-1]
            report["performance_by_period"]["hourly"] = {
                "return": latest_hourly.period_return,
                "pnl": float(latest_hourly.period_pnl),
                "trades": latest_hourly.total_trades,
                "win_rate": latest_hourly.win_rate
            }
        
        # Add daily performance
        if self.daily_snapshots:
            latest_daily = self.daily_snapshots[-1]
            report["performance_by_period"]["daily"] = {
                "return": latest_daily.period_return,
                "pnl": float(latest_daily.period_pnl),
                "trades": latest_daily.total_trades,
                "win_rate": latest_daily.win_rate,
                "sharpe_ratio": latest_daily.sharpe_ratio
            }
        
        # Add weekly performance
        if self.weekly_snapshots:
            latest_weekly = self.weekly_snapshots[-1]
            report["performance_by_period"]["weekly"] = {
                "return": latest_weekly.period_return,
                "pnl": float(latest_weekly.period_pnl),
                "trades": latest_weekly.total_trades,
                "win_rate": latest_weekly.win_rate,
                "max_drawdown": latest_weekly.max_drawdown
            }
        
        # Add monthly performance  
        if self.monthly_snapshots:
            latest_monthly = self.monthly_snapshots[-1]
            report["performance_by_period"]["monthly"] = {
                "return": latest_monthly.period_return,
                "pnl": float(latest_monthly.period_pnl),
                "trades": latest_monthly.total_trades,
                "win_rate": latest_monthly.win_rate,
                "profit_factor": latest_monthly.profit_factor
            }
        
        # Add strategy breakdown
        report["strategy_performance"] = metrics.get("strategy_metrics", {})
        
        # Add risk metrics
        report["risk_metrics"] = {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "current_drawdown": metrics.get("current_drawdown", 0),
            "var_95": self._calculate_var(0.95) if self.trades else 0,
            "profit_factor": metrics.get("profit_factor", 0)
        }
        
        # Add best and worst trades
        if metrics.get("best_trade"):
            report["best_trade"] = {
                "symbol": metrics["best_trade"].token_symbol,
                "roi": metrics["best_trade"].roi,
                "pnl": float(metrics["best_trade"].pnl)
            }
        
        if metrics.get("worst_trade"):
            report["worst_trade"] = {
                "symbol": metrics["worst_trade"].token_symbol,
                "roi": metrics["worst_trade"].roi,
                "pnl": float(metrics["worst_trade"].pnl)
            }
        
        return report
    
    def _calculate_var(self, confidence_level: float) -> float:
        """Helper to calculate Value at Risk"""
        if not self.trades:
            return 0.0
        
        returns = sorted([float(t.pnl / t.entry_value) for t in self.trades if t.entry_value > 0])
        if not returns:
            return 0.0
        
        index = int(len(returns) * (1 - confidence_level))
        return abs(returns[index]) if index < len(returns) else 0.0



class StructuredLogger:
    """
    Structured logging system with multiple outputs
    """
    
    def __init__(self, name: str = "TradingBot", config: Optional[Dict] = None):
        """Initialize structured logger"""
        self.name = name
        self.config = config or self._default_config()
        self.setup_logging()
        
    def _default_config(self) -> Dict:
        """Default logging configuration"""
        return {
            "log_level": "INFO",
            "log_dir": "logs",
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 10,
            "format": "json",  # json or text
            "outputs": ["console", "file", "database"],
            "performance_tracking": True,
            "error_tracking": True
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config["log_level"]))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        if "console" in self.config["outputs"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_formatter("console"))
            logger.addHandler(console_handler)
        
        # File handler
        if "file" in self.config["outputs"]:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                log_dir / f"{self.name}.log",
                maxBytes=self.config["max_file_size"],
                backupCount=self.config["backup_count"]
            )
            file_handler.setFormatter(self._get_formatter("file"))
            logger.addHandler(file_handler)
            
            # Separate error log
            if self.config["error_tracking"]:
                error_handler = RotatingFileHandler(
                    log_dir / f"{self.name}_errors.log",
                    maxBytes=self.config["max_file_size"],
                    backupCount=self.config["backup_count"]
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(self._get_formatter("file"))
                logger.addHandler(error_handler)
            
            # Separate trade log
            if self.config["performance_tracking"]:
                trade_handler = RotatingFileHandler(
                    log_dir / f"{self.name}_trades.log",
                    maxBytes=self.config["max_file_size"],
                    backupCount=self.config["backup_count"]
                )
                trade_handler.setLevel(TRADE_LOG)
                trade_handler.setFormatter(self._get_formatter("trade"))
                logger.addHandler(trade_handler)
    
    def _get_formatter(self, output_type: str):
        """Get appropriate formatter for output type"""
        if self.config["format"] == "json" and output_type != "console":
            return JsonFormatter()
        elif output_type == "trade":
            return TradeFormatter()
        else:
            return ColoredFormatter() if output_type == "console" else StandardFormatter()


class JsonFormatter(logging.Formatter):
    """JSON log formatter"""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, 'trade_data'):
            log_obj["trade"] = record.trade_data
        
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'TRADE': '\033[35m',     # Magenta
        'PROFIT': '\033[92m',    # Bright Green
        'LOSS': '\033[91m',      # Bright Red
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m'   # Red Background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class StandardFormatter(logging.Formatter):
    """Standard text formatter"""
    
    def __init__(self):
        super().__init__(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class TradeFormatter(logging.Formatter):
    """Trade-specific formatter"""
    
    def format(self, record):
        if hasattr(record, 'trade_data'):
            trade = record.trade_data
            return (
                f"{record.created} | {trade['symbol']} | "
                f"{trade['action']} | P&L: {trade['pnl']} | "
                f"ROI: {trade['roi']} | {trade['strategy']}"
            )
        return super().format(record)

#there is only one file provided as Performance Tracker & Logger System