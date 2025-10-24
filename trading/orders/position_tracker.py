from __future__ import annotations

"""
Position Tracker for DexScreener Trading Bot
Tracks all open positions and portfolio performance
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status states"""
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"

class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"

class RiskLevel(Enum):
    """Position risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    """Position data structure"""
    position_id: str
    token_address: str
    token_symbol: str
    position_type: PositionType
    status: PositionStatus
    
    # Entry details
    entry_price: Decimal
    entry_amount: Decimal
    entry_value: Decimal
    entry_time: datetime
    entry_order_ids: List[str]
    
    # Current state
    current_price: Decimal
    current_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: float
    
    # Exit details
    exit_price: Optional[Decimal] = None
    exit_amount: Optional[Decimal] = None
    exit_value: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    exit_order_ids: List[str] = field(default_factory=list)
    
    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: List[Decimal] = field(default_factory=list)
    trailing_stop_active: bool = False
    trailing_stop_distance: Optional[float] = None
    max_drawdown: float = 0.0
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Performance metrics
    realized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    roi: float = 0.0
    holding_period: Optional[timedelta] = None
    max_profit: Decimal = Decimal("0")
    
    # Strategy info
    strategy_name: Optional[str] = None
    signal_id: Optional[str] = None
    confidence_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    
    # Risk metrics
    position_size_percent: float = 0.0  # % of portfolio
    correlation_score: float = 0.0  # With other positions
    volatility: float = 0.0
    beta: float = 0.0
    sharpe_ratio: float = 0.0



    # Similarly for Position:
    def build_position(
        token_address: str,
        token_symbol: str,
        position_type: PositionType,
        entry_price: Decimal,
        entry_amount: Decimal,
        **kwargs
    ) -> Position:
        """
        Helper to build Position object from parameters
        
        Args:
            token_address: Token address
            token_symbol: Token symbol
            position_type: Long/Short
            entry_price: Entry price
            entry_amount: Position size
            **kwargs: Additional parameters
            
        Returns:
            Position object
        """
        entry_value = entry_price * entry_amount
        
        return Position(
            position_id=f"POS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{token_symbol}",
            token_address=token_address,
            token_symbol=token_symbol,
            position_type=position_type,
            status=PositionStatus.OPENING,
            entry_price=entry_price,
            entry_amount=entry_amount,
            entry_value=entry_value,
            entry_time=datetime.utcnow(),
            entry_order_ids=kwargs.get('order_ids', []),
            current_price=entry_price,
            current_value=entry_value,
            unrealized_pnl=Decimal("0"),
            unrealized_pnl_percent=0.0,
            stop_loss=kwargs.get('stop_loss'),
            take_profit=kwargs.get('take_profit', []),
            trailing_stop_distance=kwargs.get('trailing_stop_distance'),
            strategy_name=kwargs.get('strategy_name'),
            signal_id=kwargs.get('signal_id'),
            confidence_score=kwargs.get('confidence_score', 0.5),
            metadata=kwargs.get('metadata', {})
        )

@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    open_positions: int
    total_positions: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    win_rate: float
    average_roi: float
    sharpe_ratio: float
    max_drawdown: float
    risk_score: float

@dataclass
class PerformanceMetrics:
    """Detailed performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float
    expectancy: Decimal
    max_consecutive_wins: int
    max_consecutive_losses: int
    average_holding_time: timedelta
    total_fees: Decimal
    net_profit: Decimal
    roi: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    recovery_time: Optional[timedelta]
    risk_adjusted_return: float

class PositionTracker:
    """
    Tracks and manages all trading positions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize position tracker"""
        self.config = config or self._default_config()
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.portfolio_history: deque = deque(maxlen=10000)
        
        # Portfolio state
        self.total_portfolio_value = Decimal("0")
        initial_balance = self.config.get('initial_balance', 400)
        self.cash_balance = Decimal(str(initial_balance))
        logger.info(f"Position tracker initialized with balance: ${initial_balance}")
        self.positions_value = Decimal("0")
        
        # Risk tracking
        self.risk_metrics = {
            "var_95": Decimal("0"),  # Value at Risk
            "cvar_95": Decimal("0"),  # Conditional VaR
            "portfolio_beta": 0.0,
            "portfolio_volatility": 0.0,
            "correlation_matrix": {}
        }
        
        # Performance tracking
        self.performance_metrics = self._initialize_performance_metrics()
        
        # Price cache
        self.price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        
        # Start monitoring
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._calculate_portfolio_metrics())
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # Position limits
            "max_positions": 10,
            "max_position_size": 0.1,  # 10% of portfolio
            "max_correlated_positions": 3,
            "correlation_threshold": 0.7,
            
            # Risk parameters
            "max_portfolio_risk": 0.2,  # 20% VaR
            "max_position_risk": 0.02,  # 2% per position
            "stop_loss_default": 0.05,  # 5%
            "take_profit_default": [0.05, 0.10, 0.15],  # 5%, 10%, 15%
            
            # Monitoring
            "update_interval": 5,  # seconds
            "metrics_interval": 60,  # seconds
            "snapshot_interval": 300,  # 5 minutes
            
            # Performance thresholds
            "min_sharpe_ratio": 1.0,
            "max_drawdown_limit": 0.2,  # 20%
            "position_review_threshold": -0.05,  # Review at -5%
            
            # Alerts
            "alert_on_drawdown": 0.1,  # 10%
            "alert_on_correlation": 0.8,
            "alert_on_risk_score": 80,  # 0-100 scale
        }
    
    def _initialize_performance_metrics(self) -> PerformanceMetrics:
        """Initialize performance metrics"""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            profit_factor=0.0,
            expectancy=Decimal("0"),
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            average_holding_time=timedelta(0),
            total_fees=Decimal("0"),
            net_profit=Decimal("0"),
            roi=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            recovery_time=None,
            risk_adjusted_return=0.0
        )

    # ============================================
    # FIX 3: PositionTracker.open_position signature
    # ============================================
    # Expected: async def open_position(position: Position) -> str
    # Current: async def open_position(token_address, token_symbol, ...) -> Position

    # Add this wrapper method to PositionTracker:

    async def open_position(self, position: Position) -> str:
        """
        Open position from Position object (API-compliant signature)
        
        Args:
            position: Pre-built Position object
            
        Returns:
            Position ID string
        """
        # Validate position
        if not await self._check_position_limits(position.token_address, position.entry_amount):
            raise ValueError("Position exceeds risk limits")
        
        # Calculate risk metrics
        position.risk_score = await self._calculate_position_risk(position)
        position.risk_level = self._determine_risk_level(position.risk_score)
        
        # Store position
        self.positions[position.position_id] = position
        
        # Update portfolio
        self.cash_balance -= position.entry_value
        await self._update_portfolio_value()
        
        # Log
        logger.info(f"Opened position {position.position_id}")
        
        # Update status
        position.status = PositionStatus.OPEN
        
        return position.position_id

    async def open_position_from_params(
        self,
        token_address: str,
        token_symbol: str,
        position_type: PositionType,
        entry_price: Decimal,
        entry_amount: Decimal,
        order_ids: List[str],
        **kwargs
    ) -> Position:
        """
        Open a new position
        
        Args:
            token_address: Token contract address
            token_symbol: Token symbol
            position_type: Long or short
            entry_price: Entry price
            entry_amount: Position size
            order_ids: Associated order IDs
            **kwargs: Additional position parameters
            
        Returns:
            Created Position object
        """
        try:
            # Check position limits
            if not await self._check_position_limits(token_address, entry_amount):
                raise ValueError("Position exceeds risk limits")
            
            # Calculate entry value
            entry_value = entry_price * entry_amount
            
            # Check portfolio allocation
            position_size_percent = float(entry_value / self.total_portfolio_value)
            if position_size_percent > self.config["max_position_size"]:
                raise ValueError(f"Position size {position_size_percent:.2%} exceeds limit")
            
            # Create position
            position = Position(
                position_id=f"POS-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{token_symbol}",
                token_address=token_address,
                token_symbol=token_symbol,
                position_type=position_type,
                status=PositionStatus.OPENING,
                entry_price=entry_price,
                entry_amount=entry_amount,
                entry_value=entry_value,
                entry_time=datetime.utcnow(),
                entry_order_ids=order_ids,
                current_price=entry_price,
                current_value=entry_value,
                unrealized_pnl=Decimal("0"),
                unrealized_pnl_percent=0.0,
                stop_loss=kwargs.get("stop_loss"),
                take_profit=kwargs.get("take_profit", []),
                trailing_stop_distance=kwargs.get("trailing_stop_distance"),
                strategy_name=kwargs.get("strategy_name"),
                signal_id=kwargs.get("signal_id"),
                confidence_score=kwargs.get("confidence_score", 0.5),
                position_size_percent=position_size_percent,
                metadata=kwargs.get("metadata", {})
            )
            
            # Set default risk management if not provided
            if not position.stop_loss:
                position.stop_loss = entry_price * (
                    Decimal("1") - Decimal(str(self.config["stop_loss_default"]))
                )
            
            if not position.take_profit:
                position.take_profit = [
                    entry_price * (Decimal("1") + Decimal(str(tp)))
                    for tp in self.config["take_profit_default"]
                ]
            
            # Calculate initial risk metrics
            position.risk_score = await self._calculate_position_risk(position)
            position.risk_level = self._determine_risk_level(position.risk_score)
            
            # Update portfolio
            self.positions[position.position_id] = position
            self.cash_balance -= entry_value
            await self._update_portfolio_value()
            
            # Log position opening
            logger.info(
                f"Opened position {position.position_id}: "
                f"{position_type.value} {entry_amount} {token_symbol} @ {entry_price}"
            )
            
            # Update status to open
            position.status = PositionStatus.OPEN
            
            return position
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            raise
 
    # ============================================
    # FIX 5: PositionTracker.update_position signature
    # ============================================
    # Expected: async def update_position(position_id: str, updates: Dict) -> bool
    # Current signature has current_price as separate param

    # Add this wrapper method:
    async def update_position(self, position_id: str, updates: Dict) -> bool:
        """
        Update position with dictionary (API-compliant signature)
        
        Args:
            position_id: Position ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if updated successfully
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                return False
            
            # Extract current_price if in updates
            current_price = updates.pop('current_price', None)
            
            # If current_price provided, update with it
            if current_price:
                result = await self.update_position_with_price(
                    position_id,
                    Decimal(str(current_price)),
                    **updates
                )
            else:
                # Just apply the updates directly
                for key, value in updates.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                result = {"success": True}
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
            return False
 
    # Rename existing update_position to update_position_with_price:
    async def update_position_with_price(
        self,
        position_id: str,
        current_price: Decimal,
        **updates
    ) -> Optional[Dict]:
        """
        Update position with current market data
        
        Args:
            position_id: Position ID to update
            current_price: Current market price
            **updates: Additional updates
            
        Returns:
            Update result with any actions to take
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                return None
            
            # Update price
            position.current_price = current_price
            
            # Calculate current value and P&L
            if position.position_type == PositionType.LONG:
                position.current_value = current_price * position.entry_amount
                position.unrealized_pnl = position.current_value - position.entry_value
            else:  # SHORT
                price_change = position.entry_price - current_price
                position.unrealized_pnl = price_change * position.entry_amount
                position.current_value = position.entry_value + position.unrealized_pnl
            
            # Calculate P&L percentage
            position.unrealized_pnl_percent = float(
                position.unrealized_pnl / position.entry_value
            )
            
            # Track max profit
            if position.unrealized_pnl > position.max_profit:
                position.max_profit = position.unrealized_pnl
            
            # Calculate drawdown from peak
            if position.max_profit > 0:
                drawdown = float(
                    (position.max_profit - position.unrealized_pnl) / position.max_profit
                )
                position.max_drawdown = max(position.max_drawdown, drawdown)
            
            # Apply any additional updates
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, value)
            
            # Check risk management rules
            actions = await self._check_position_rules(position)
            
            # Update risk metrics
            position.risk_score = await self._calculate_position_risk(position)
            position.risk_level = self._determine_risk_level(position.risk_score)
            
            # Cache price
            self.price_cache[position.token_address] = (current_price, datetime.utcnow())
            
            return actions
            
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
            return None

    # ============================================
    # FIX 4: PositionTracker.close_position signature
    # ============================================
    # Expected: async def close_position(position_id: str) -> Dict
    # Current has extra parameters, so add a simpler version:

    async def close_position(self, position_id: str) -> Dict:
        """
        Close position completely (API-compliant signature)
        
        Args:
            position_id: Position ID to close
            
        Returns:
            Dictionary with closure details
        """
        position = self.positions.get(position_id)
        if not position:
            return {"success": False, "error": "Position not found"}
        
        # Get current price from cache or market
        current_price = position.current_price
        
        # Close at current market price
        closed_position = await self.close_position_with_details(
            position_id=position_id,
            exit_price=current_price,
            exit_amount=position.entry_amount,
            reason="manual_close"
        )
        
        if closed_position:
            return {
                "success": True,
                "position_id": position_id,
                "exit_price": str(closed_position.exit_price),
                "realized_pnl": str(closed_position.realized_pnl),
                "roi": closed_position.roi
            }
        else:
            return {"success": False, "error": "Failed to close position"}


    # Rename existing close_position to close_position_with_details:
    async def close_position_with_details(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_amount: Optional[Decimal] = None,
        order_ids: Optional[List[str]] = None,
        reason: Optional[str] = None
    ) -> Optional[Position]:
        """
        Close a position
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_amount: Amount to close (partial close if less than position)
            order_ids: Associated order IDs
            reason: Reason for closing
            
        Returns:
            Updated Position object
        """
        try:
            position = self.positions.get(position_id)
            if not position:
                return None
            
            # Default to full position
            if exit_amount is None:
                exit_amount = position.entry_amount
            
            # Calculate exit value
            exit_value = exit_price * exit_amount
            
            # Calculate realized P&L
            if position.position_type == PositionType.LONG:
                realized_pnl = (exit_price - position.entry_price) * exit_amount
            else:  # SHORT
                realized_pnl = (position.entry_price - exit_price) * exit_amount
            
            # Check if partial or full close
            if exit_amount >= position.entry_amount:
                # Full close
                position.status = PositionStatus.CLOSING
                position.exit_price = exit_price
                position.exit_amount = exit_amount
                position.exit_value = exit_value
                position.exit_time = datetime.utcnow()
                position.realized_pnl = realized_pnl
                
                if order_ids:
                    position.exit_order_ids.extend(order_ids)
                
                # Calculate final metrics
                position.holding_period = position.exit_time - position.entry_time
                position.roi = float(position.realized_pnl / position.entry_value)
                
                # Add to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                
                # Update portfolio
                self.cash_balance += exit_value
                
                # Update performance metrics
                await self._update_performance_metrics(position)
                
                # Log closure
                logger.info(
                    f"Closed position {position_id}: "
                    f"P&L: {realized_pnl:.2f} ({position.roi:.2%})"
                    f"{f' Reason: {reason}' if reason else ''}"
                )
                
                position.status = PositionStatus.CLOSED
                
            else:
                # Partial close
                remaining_amount = position.entry_amount - exit_amount
                position.entry_amount = remaining_amount
                position.entry_value = position.entry_price * remaining_amount
                position.realized_pnl += realized_pnl
                
                # Update portfolio
                self.cash_balance += exit_value
                
                logger.info(
                    f"Partial close {position_id}: "
                    f"Closed {exit_amount} @ {exit_price}, "
                    f"Remaining: {remaining_amount}"
                )
            
            # Add note
            if reason:
                position.notes.append(f"{datetime.utcnow()}: {reason}")
            
            await self._update_portfolio_value()
            
            return position
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return None
    
    async def _check_position_limits(
        self,
        token_address: str,
        amount: Decimal
    ) -> bool:
        """Check if new position meets risk limits"""
        try:
            # Check max positions
            if len(self.positions) >= self.config["max_positions"]:
                logger.warning("Maximum positions limit reached")
                return False
            
            # Check if already have position in this token
            existing = [
                p for p in self.positions.values()
                if p.token_address == token_address
            ]
            if existing:
                logger.warning(f"Already have position in {token_address}")
                return False
            
            # Check correlation with existing positions
            correlated_count = 0
            for position in self.positions.values():
                correlation = await self._calculate_correlation(
                    token_address,
                    position.token_address
                )
                if correlation > self.config["correlation_threshold"]:
                    correlated_count += 1
            
            if correlated_count >= self.config["max_correlated_positions"]:
                logger.warning("Too many correlated positions")
                return False
            
            # Check portfolio risk
            portfolio_risk = await self._calculate_portfolio_risk()
            if portfolio_risk > self.config["max_portfolio_risk"]:
                logger.warning(f"Portfolio risk {portfolio_risk:.2%} exceeds limit")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False
    
    async def _check_position_rules(self, position: Position) -> Dict[str, Any]:
        """Check position against risk management rules"""
        actions = {}
        
        try:
            # Check stop loss
            if position.stop_loss:
                if position.position_type == PositionType.LONG:
                    if position.current_price <= position.stop_loss:
                        actions["close"] = {
                            "reason": "stop_loss_hit",
                            "price": position.current_price
                        }
                else:  # SHORT
                    if position.current_price >= position.stop_loss:
                        actions["close"] = {
                            "reason": "stop_loss_hit",
                            "price": position.current_price
                        }
            
            # Check take profit levels
            for i, tp_price in enumerate(position.take_profit):
                if position.position_type == PositionType.LONG:
                    if position.current_price >= tp_price:
                        actions["partial_close"] = {
                            "reason": f"take_profit_{i+1}_hit",
                            "price": position.current_price,
                            "amount": position.entry_amount / len(position.take_profit)
                        }
                        break
                else:  # SHORT
                    if position.current_price <= tp_price:
                        actions["partial_close"] = {
                            "reason": f"take_profit_{i+1}_hit",
                            "price": position.current_price,
                            "amount": position.entry_amount / len(position.take_profit)
                        }
                        break
            
            # Check trailing stop
            if position.trailing_stop_active and position.trailing_stop_distance:
                if position.position_type == PositionType.LONG:
                    trailing_stop = position.current_price * (
                        Decimal("1") - Decimal(str(position.trailing_stop_distance))
                    )
                    if position.stop_loss is None or trailing_stop > position.stop_loss:
                        actions["adjust_stop"] = {
                            "new_stop": trailing_stop,
                            "reason": "trailing_stop_update"
                        }
                        position.stop_loss = trailing_stop
            
            # Check drawdown limit
            if position.max_drawdown > 0.1:  # 10% drawdown from peak
                actions["review"] = {
                    "reason": "high_drawdown",
                    "drawdown": position.max_drawdown
                }
            
            # Check time-based exit
            if position.metadata.get("max_holding_time"):
                max_time = timedelta(seconds=position.metadata["max_holding_time"])
                if datetime.utcnow() - position.entry_time > max_time:
                    actions["close"] = {
                        "reason": "max_time_reached",
                        "price": position.current_price
                    }
            
            # Check risk score
            if position.risk_score > self.config["alert_on_risk_score"]:
                actions["alert"] = {
                    "reason": "high_risk_score",
                    "risk_score": position.risk_score
                }
            
            return actions
            
        except Exception as e:
            logger.error(f"Error checking position rules: {e}")
            return {}
    
    async def _calculate_position_risk(self, position: Position) -> float:
        """Calculate position risk score (0-100)"""
        try:
            risk_score = 0.0
            
            # Size risk (0-30 points)
            size_risk = min(position.position_size_percent / 0.1, 1.0) * 30
            risk_score += size_risk
            
            # Volatility risk (0-25 points)
            if position.volatility > 0:
                vol_risk = min(position.volatility / 0.5, 1.0) * 25
                risk_score += vol_risk
            
            # Drawdown risk (0-20 points)
            dd_risk = min(position.max_drawdown / 0.2, 1.0) * 20
            risk_score += dd_risk
            
            # P&L risk (0-15 points)
            if position.unrealized_pnl_percent < -0.05:
                pnl_risk = min(abs(position.unrealized_pnl_percent) / 0.1, 1.0) * 15
                risk_score += pnl_risk
            
            # Correlation risk (0-10 points)
            corr_risk = min(position.correlation_score, 1.0) * 10
            risk_score += corr_risk
            
            return min(risk_score, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return 50.0  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score < 25:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    async def _calculate_correlation(
        self,
        token1: str,
        token2: str
    ) -> float:
        """Calculate correlation between two tokens"""
        try:
            # This would use historical price data
            # For now, return a placeholder
            return 0.0
        except:
            return 0.0
    
    async def _calculate_portfolio_risk(self) -> float:
        """Calculate overall portfolio risk"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate Value at Risk (VaR)
            position_values = [p.current_value for p in self.positions.values()]
            total_value = sum(position_values)
            
            if total_value == 0:
                return 0.0
            
            # Simple risk calculation based on position concentration
            max_position = max(position_values)
            concentration_risk = float(max_position / total_value)
            
            # Add volatility component
            avg_volatility = statistics.mean(
                [p.volatility for p in self.positions.values()]
                if all(p.volatility > 0 for p in self.positions.values())
                else [0.1]
            )
            
            portfolio_risk = concentration_risk * 0.6 + avg_volatility * 0.4
            
            return min(portfolio_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return 0.0
    
    async def _update_portfolio_value(self):
        """Update total portfolio value"""
        try:
            # Sum position values
            self.positions_value = sum(
                p.current_value for p in self.positions.values()
            )
            
            # Total portfolio value
            self.total_portfolio_value = self.cash_balance + self.positions_value
            
            # Calculate unrealized P&L
            total_unrealized = sum(
                p.unrealized_pnl for p in self.positions.values()
            )
            
            # Store snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                total_value=self.total_portfolio_value,
                cash_balance=self.cash_balance,
                positions_value=self.positions_value,
                open_positions=len(self.positions),
                total_positions=len(self.positions) + len(self.closed_positions),
                realized_pnl=sum(p.realized_pnl for p in self.closed_positions),
                unrealized_pnl=total_unrealized,
                win_rate=self.performance_metrics.win_rate,
                average_roi=self.performance_metrics.roi,
                sharpe_ratio=self.performance_metrics.sharpe_ratio,
                max_drawdown=self.performance_metrics.max_drawdown,
                risk_score=await self._calculate_portfolio_risk() * 100
            )
            
            self.portfolio_history.append(snapshot)
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    async def _update_performance_metrics(self, closed_position: Position):
        """Update performance metrics after position closure"""
        try:
            metrics = self.performance_metrics
            
            # Update trade counts
            metrics.total_trades += 1
            
            if closed_position.realized_pnl > 0:
                metrics.winning_trades += 1
                
                # Update winning streak
                current_streak = metrics.max_consecutive_wins
                if self.closed_positions and self.closed_positions[-1].realized_pnl > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                metrics.max_consecutive_wins = max(
                    metrics.max_consecutive_wins,
                    current_streak
                )
            else:
                metrics.losing_trades += 1
                
                # Update losing streak
                current_streak = metrics.max_consecutive_losses
                if self.closed_positions and self.closed_positions[-1].realized_pnl <= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                metrics.max_consecutive_losses = max(
                    metrics.max_consecutive_losses,
                    current_streak
                )
            
            # Calculate win rate
            metrics.win_rate = (
                metrics.winning_trades / metrics.total_trades
                if metrics.total_trades > 0 else 0
            )
            
            # Calculate average win/loss
            wins = [p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]
            losses = [abs(p.realized_pnl) for p in self.closed_positions if p.realized_pnl <= 0]
            
            metrics.average_win = sum(wins) / len(wins) if wins else Decimal("0")
            metrics.average_loss = sum(losses) / len(losses) if losses else Decimal("0")
            
            # Calculate profit factor
            total_wins = sum(wins) if wins else Decimal("0")
            total_losses = sum(losses) if losses else Decimal("1")
            metrics.profit_factor = float(total_wins / total_losses) if total_losses > 0 else 0
            
            # Calculate expectancy
            metrics.expectancy = (
                metrics.win_rate * metrics.average_win -
                (1 - metrics.win_rate) * metrics.average_loss
            )
            
            # Update holding time
            holding_times = [
                p.holding_period for p in self.closed_positions
                if p.holding_period
            ]
            if holding_times:
                metrics.average_holding_time = sum(
                    holding_times, timedelta()
                ) / len(holding_times)
            
            # Update fees
            metrics.total_fees = sum(
                p.total_fees for p in self.closed_positions
            )
            
            # Calculate net profit
            metrics.net_profit = sum(
                p.realized_pnl for p in self.closed_positions
            ) - metrics.total_fees
            
            # Calculate ROI
            initial_balance = Decimal("10000")  # From config
            metrics.roi = float(metrics.net_profit / initial_balance)
            
            # Calculate Sharpe ratio
            if len(self.closed_positions) > 1:
                returns = [
                    float(p.realized_pnl / p.entry_value)
                    for p in self.closed_positions
                ]
                
                if returns:
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0.01
                    
                    # Annualized Sharpe ratio (assuming daily trading)
                    metrics.sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            
            # Calculate maximum drawdown
            if self.portfolio_history:
                peak_value = self.portfolio_history[0].total_value
                max_dd = 0.0
                
                for snapshot in self.portfolio_history:
                    if snapshot.total_value > peak_value:
                        peak_value = snapshot.total_value
                    
                    drawdown = float(
                        (peak_value - snapshot.total_value) / peak_value
                    ) if peak_value > 0 else 0
                    
                    max_dd = max(max_dd, drawdown)
                
                metrics.max_drawdown = max_dd
            
            # Calculate risk-adjusted return
            if metrics.max_drawdown > 0:
                metrics.risk_adjusted_return = metrics.roi / metrics.max_drawdown
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _monitor_positions(self):
        """Background task to monitor positions"""
        while True:
            try:
                for position_id, position in list(self.positions.items()):
                    # Get latest price
                    current_price = await self._get_current_price(position.token_address)
                    
                    if current_price:
                        # Update position
                        actions = await self.update_position(
                            position_id,
                            current_price
                        )
                        
                        # Execute any required actions
                        if actions:
                            await self._execute_position_actions(
                                position,
                                actions
                            )
                
                # Update portfolio metrics
                await self._update_portfolio_value()
                
                await asyncio.sleep(self.config.get("update_interval", 5))
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)
    
    async def _calculate_portfolio_metrics(self):
        """Background task to calculate detailed portfolio metrics"""
        while True:
            try:
                await asyncio.sleep(self.config.get("metrics_interval", 60))
                
                if not self.positions:
                    continue
                
                # Calculate VaR and CVaR
                await self._calculate_value_at_risk()
                
                # Calculate portfolio beta
                self.risk_metrics["portfolio_beta"] = await self._calculate_portfolio_beta()
                
                # Calculate portfolio volatility
                self.risk_metrics["portfolio_volatility"] = await self._calculate_portfolio_volatility()
                
                # Update correlation matrix
                await self._update_correlation_matrix()
                
                # Check for alerts
                await self._check_portfolio_alerts()
                
            except Exception as e:
                logger.error(f"Error calculating portfolio metrics: {e}")
    
    async def _get_current_price(self, token_address: str) -> Optional[Decimal]:
        """Get current price for token"""
        try:
            # Check cache first
            if token_address in self.price_cache:
                price, timestamp = self.price_cache[token_address]
                if (datetime.utcnow() - timestamp).total_seconds() < 30:
                    return price
            
            # Fetch from market data source (placeholder)
            # In production, this would call the data collector
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {token_address}: {e}")
            return None
    
    async def _execute_position_actions(
        self,
        position: Position,
        actions: Dict[str, Any]
    ):
        """Execute position management actions"""
        try:
            if "close" in actions:
                await self.close_position(
                    position.position_id,
                    actions["close"]["price"],
                    reason=actions["close"].get("reason")
                )
            
            elif "partial_close" in actions:
                await self.close_position(
                    position.position_id,
                    actions["partial_close"]["price"],
                    exit_amount=actions["partial_close"].get("amount"),
                    reason=actions["partial_close"].get("reason")
                )
            
            if "adjust_stop" in actions:
                position.stop_loss = actions["adjust_stop"]["new_stop"]
                logger.info(
                    f"Adjusted stop loss for {position.position_id}: "
                    f"{actions['adjust_stop']['new_stop']}"
                )
            
            if "alert" in actions:
                logger.warning(
                    f"Alert for {position.position_id}: "
                    f"{actions['alert']['reason']}"
                )
            
            if "review" in actions:
                logger.info(
                    f"Position {position.position_id} flagged for review: "
                    f"{actions['review']['reason']}"
                )
                
        except Exception as e:
            logger.error(f"Error executing position actions: {e}")
    
    async def _calculate_value_at_risk(self):
        """Calculate portfolio VaR and CVaR"""
        try:
            if not self.portfolio_history:
                return
            
            # Get portfolio returns
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev = self.portfolio_history[i-1].total_value
                curr = self.portfolio_history[i].total_value
                if prev > 0:
                    ret = float((curr - prev) / prev)
                    returns.append(ret)
            
            if not returns:
                return
            
            # Calculate VaR at 95% confidence
            sorted_returns = sorted(returns)
            var_index = int(len(sorted_returns) * 0.05)
            self.risk_metrics["var_95"] = Decimal(str(abs(sorted_returns[var_index])))
            
            # Calculate CVaR (expected loss beyond VaR)
            cvar_returns = sorted_returns[:var_index]
            if cvar_returns:
                self.risk_metrics["cvar_95"] = Decimal(str(abs(statistics.mean(cvar_returns))))
                
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta relative to market"""
        try:
            # This would compare portfolio returns to market index
            # Placeholder for now
            return 1.0
        except:
            return 1.0
    
    async def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, min(len(self.portfolio_history), 100)):
                prev = float(self.portfolio_history[i-1].total_value)
                curr = float(self.portfolio_history[i].total_value)
                if prev > 0:
                    returns.append((curr - prev) / prev)
            
            if len(returns) > 1:
                return statistics.stdev(returns)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    async def _update_correlation_matrix(self):
        """Update correlation matrix between positions"""
        try:
            # Calculate pairwise correlations
            correlations = {}
            
            positions = list(self.positions.values())
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    correlation = await self._calculate_correlation(
                        pos1.token_address,
                        pos2.token_address
                    )
                    correlations[f"{pos1.token_symbol}-{pos2.token_symbol}"] = correlation
                    
                    # Update position correlation scores
                    pos1.correlation_score = max(pos1.correlation_score, correlation)
                    pos2.correlation_score = max(pos2.correlation_score, correlation)
            
            self.risk_metrics["correlation_matrix"] = correlations
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    async def _check_portfolio_alerts(self):
        """Check for portfolio-level alerts"""
        try:
            # Check drawdown
            if self.performance_metrics.max_drawdown > self.config["alert_on_drawdown"]:
                logger.warning(
                    f"Portfolio drawdown alert: "
                    f"{self.performance_metrics.max_drawdown:.2%}"
                )
            
            # Check risk score
            risk_score = await self._calculate_portfolio_risk() * 100
            if risk_score > self.config["alert_on_risk_score"]:
                logger.warning(f"Portfolio risk alert: {risk_score:.1f}")
            
            # Check correlation
            if self.risk_metrics["correlation_matrix"]:
                max_corr = max(self.risk_metrics["correlation_matrix"].values())
                if max_corr > self.config["alert_on_correlation"]:
                    logger.warning(f"High correlation alert: {max_corr:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking portfolio alerts: {e}")
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return [
            {
                "position_id": p.position_id,
                "symbol": p.token_symbol,
                "type": p.position_type.value,
                "entry_price": str(p.entry_price),
                "current_price": str(p.current_price),
                "pnl": str(p.unrealized_pnl),
                "pnl_percent": f"{p.unrealized_pnl_percent:.2%}",
                "risk_level": p.risk_level.value,
                "holding_time": str(datetime.utcnow() - p.entry_time)
            }
            for p in self.positions.values()
        ]
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            "total_value": str(self.total_portfolio_value),
            "cash_balance": str(self.cash_balance),
            "positions_value": str(self.positions_value),
            "open_positions": len(self.positions),
            "total_trades": self.performance_metrics.total_trades,
            "win_rate": f"{self.performance_metrics.win_rate:.2%}",
            "net_profit": str(self.performance_metrics.net_profit),
            "roi": f"{self.performance_metrics.roi:.2%}",
            "sharpe_ratio": f"{self.performance_metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.performance_metrics.max_drawdown:.2%}",
            "var_95": str(self.risk_metrics["var_95"]),
            "portfolio_risk": f"{self.risk_metrics['portfolio_volatility']:.2%}"
        }
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        metrics = self.performance_metrics
        return {
            "summary": {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "win_rate": f"{metrics.win_rate:.2%}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "expectancy": str(metrics.expectancy),
                "net_profit": str(metrics.net_profit),
                "roi": f"{metrics.roi:.2%}"
            },
            "risk_metrics": {
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "var_95": str(self.risk_metrics["var_95"]),
                "cvar_95": str(self.risk_metrics["cvar_95"]),
                "portfolio_beta": f"{self.risk_metrics['portfolio_beta']:.2f}",
                "portfolio_volatility": f"{self.risk_metrics['portfolio_volatility']:.2%}"
            },
            "trading_stats": {
                "average_win": str(metrics.average_win),
                "average_loss": str(metrics.average_loss),
                "max_consecutive_wins": metrics.max_consecutive_wins,
                "max_consecutive_losses": metrics.max_consecutive_losses,
                "average_holding_time": str(metrics.average_holding_time),
                "total_fees": str(metrics.total_fees)
            }
        }

    # Additional patch for position_tracker.py to match exact documentation:
    # Add this method to PositionTracker class:

    async def get_active_positions(self) -> List[Dict]:
        """
        Get all active positions
        Wrapper for get_open_positions() to match documentation
        
        Returns:
            List of active position dictionaries
        """
        return self.get_open_positions()

    async def calculate_pnl(self, position_id: str) -> Dict:
        """
        Calculate P&L for a specific position
        
        Args:
            position_id: Position ID to calculate P&L for
            
        Returns:
            P&L details dictionary
        """
        position = self.positions.get(position_id)
        if not position:
            return {"error": "Position not found"}
        
        return {
            "position_id": position_id,
            "unrealized_pnl": str(position.unrealized_pnl),
            "unrealized_pnl_percent": position.unrealized_pnl_percent,
            "realized_pnl": str(position.realized_pnl),
            "total_pnl": str(position.unrealized_pnl + position.realized_pnl),
            "roi": position.roi,
            "max_profit": str(position.max_profit),
            "max_drawdown": position.max_drawdown
        }

    async def check_stop_loss(self, position_id: str) -> bool:
        """
        Check if stop loss has been hit for a position
        
        Args:
            position_id: Position ID to check
            
        Returns:
            True if stop loss hit, False otherwise
        """
        position = self.positions.get(position_id)
        if not position or not position.stop_loss:
            return False
        
        if position.position_type == PositionType.LONG:
            return position.current_price <= position.stop_loss
        else:  # SHORT
            return position.current_price >= position.stop_loss