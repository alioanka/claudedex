"""
Portfolio Manager - Portfolio optimization and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import asyncio
from collections import defaultdict
from monitoring.logger import log_portfolio_update

@dataclass
class Position:
    """Represents a trading position"""
    id: str
    token_address: str
    pair_address: str
    chain: str
    entry_price: float
    current_price: float
    size: float
    cost: float
    entry_time: datetime
    last_updated: datetime
    stop_loss: float
    take_profits: List[float]
    tp_hit: List[bool]
    strategy: str
    pnl: float = 0
    pnl_percentage: float = 0
    status: str = 'open'  # open, closing, closed
    initial_liquidity: float = 0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.size
        
    @property
    def value(self) -> float:
        """Current position value"""
        return self.current_price * self.size
        
    @property
    def age(self) -> timedelta:
        """Position age"""
        return datetime.now() - self.entry_time

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    available_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    positions_count: int
    risk_exposure: float
    diversification_score: float
    
class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY = "kelly"
    MARKOWITZ = "markowitz"
    DYNAMIC = "dynamic"

class PortfolioManager:
    """Advanced portfolio management and optimization"""
    
    def __init__(self, config: Dict):
        """
        Initialize portfolio manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Portfolio parameters
        self.max_positions = config.get('max_positions', 10)
        self.max_position_size_pct = config.get('max_position_size_pct', 0.1)  # 10% max
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.05)  # 5% max
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.25)  # 25% max
        self.min_position_size = config.get('min_position_size', 5.0)  # $100 min
        
        # Allocation strategy
        self.allocation_strategy = AllocationStrategy[
            config.get('allocation_strategy', 'DYNAMIC').upper()
        ]
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.balance = config.get('initial_balance', 400)
        self.initial_balance = self.balance
        
        # Performance tracking
        self.trade_history = []
        self.daily_returns = []
        self.peak_balance = self.balance
        self.valley_balance = self.balance
        
        # Risk limits
        self.daily_loss_limit = config.get('daily_loss_limit', 0.1)  # 10% daily loss limit
        self.consecutive_losses_limit = config.get('consecutive_losses_limit', 5)
        self.consecutive_losses = 0
        
        # Correlation tracking
        self.correlation_matrix = pd.DataFrame()
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
        # Rebalancing
        self.rebalance_frequency = config.get('rebalance_frequency', 'daily')
        self.last_rebalance = datetime.now()
        self.max_position_size_usd = config.get('max_position_size_usd', 10.0)  # $10 max per trade
        
    def can_open_position(self) -> bool:
        """Check if new position can be opened"""
        try:
            # Check position count
            if len(self.positions) >= self.max_positions:
                return False
                
            # Check available balance
            if self.get_available_balance() < self.min_position_size:
                return False
                
            # Check daily loss limit
            if self._check_daily_loss_limit():
                return False
                
            # Check consecutive losses
            if self.consecutive_losses >= self.consecutive_losses_limit:
                return False
                
            # Check total risk exposure
            if self._calculate_total_risk() >= self.max_portfolio_risk:
                return False
                
            return True
            
        except Exception as e:
            print(f"Position check error: {e}")
            return False
            
    def get_available_balance(self) -> float:
        """Get available balance for trading"""
        try:
            # Calculate locked balance in positions
            locked_balance = sum(pos.value for pos in self.positions.values())
            
            # Available balance
            available = self.balance - locked_balance
            
            # Reserve for fees and slippage
            reserve = self.balance * 0.05  # 5% reserve
            
            available = max(0, available - reserve)
            
            # ✅ ENFORCE: Never return more than max_position_size_usd
            # This caps each trade at $10 max
            return min(available, self.max_position_size_usd)

            
        except Exception as e:
            print(f"Balance calculation error: {e}")
            return 0

    def get_max_position_size(self, chain: str = None) -> float:
        """
        Get maximum position size for a trade
        
        Args:
            chain: Blockchain (optional, for chain-specific limits)
            
        Returns:
            Maximum position size in USD
        """
        try:
            # Get base max position size
            max_size = self.max_position_size_usd if hasattr(self, 'max_position_size_usd') else 10.0
            
            # Could implement chain-specific limits here
            if chain:
                # Example: Different limits per chain
                chain_limits = {
                    'ethereum': max_size,
                    'base': max_size,
                    'bsc': max_size,
                    'arbitrum': max_size,
                    'polygon': max_size,
                    'solana': max_size
                }
                max_size = chain_limits.get(chain.lower(), max_size)
            
            # Never exceed available balance
            available = self.get_available_balance()
            
            return min(max_size, available)
            
        except Exception as e:
            print(f"Error getting max position size: {e}")
            return 10.0  # Safe default

            
    async def allocate_capital(self, opportunities: List[Dict]) -> Dict[str, float]:
        """
        Allocate capital across opportunities
        
        Args:
            opportunities: List of trading opportunities
            
        Returns:
            Dictionary of allocations {opportunity_id: amount}
        """
        try:
            if not opportunities:
                return {}
                
            available = self.get_available_balance()
            
            if self.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
                return self._equal_weight_allocation(opportunities, available)
                
            elif self.allocation_strategy == AllocationStrategy.RISK_PARITY:
                return await self._risk_parity_allocation(opportunities, available)
                
            elif self.allocation_strategy == AllocationStrategy.KELLY:
                return await self._kelly_allocation(opportunities, available)
                
            elif self.allocation_strategy == AllocationStrategy.MARKOWITZ:
                return await self._markowitz_allocation(opportunities, available)
                
            else:  # DYNAMIC
                return await self._dynamic_allocation(opportunities, available)
                
        except Exception as e:
            print(f"Allocation error: {e}")
            return {}
            
    def _equal_weight_allocation(self, opportunities: List[Dict], available: float) -> Dict[str, float]:
        """Equal weight allocation across opportunities"""
        allocations = {}
        
        try:
            # Number of positions to open
            n_positions = min(len(opportunities), self.max_positions - len(self.positions))
            
            if n_positions == 0:
                return {}
                
            # Equal allocation
            allocation_per_position = available / n_positions
            
            # Apply position size limits
            max_size = self.balance * self.max_position_size_pct
            allocation_per_position = min(allocation_per_position, max_size)
            
            for opp in opportunities[:n_positions]:
                if allocation_per_position >= self.min_position_size:
                    allocations[opp['id']] = allocation_per_position
                    
            return allocations
            
        except Exception as e:
            print(f"Equal weight allocation error: {e}")
            return {}
            
    async def _risk_parity_allocation(self, opportunities: List[Dict], 
                                      available: float) -> Dict[str, float]:
        """Risk parity allocation - equal risk contribution"""
        allocations = {}
        
        try:
            # Calculate risk for each opportunity
            risks = []
            for opp in opportunities:
                risk_score = opp.get('risk_score', 0.5)
                volatility = opp.get('volatility', 0.1)
                combined_risk = (risk_score + volatility) / 2
                risks.append(combined_risk)
                
            if not risks:
                return {}
                
            # Inverse risk weighting
            risk_array = np.array(risks)
            inverse_risks = 1 / (risk_array + 0.01)  # Add small value to avoid division by zero
            weights = inverse_risks / inverse_risks.sum()
            
            # Allocate based on weights
            for i, opp in enumerate(opportunities):
                allocation = available * weights[i]
                
                # Apply limits
                max_size = self.balance * self.max_position_size_pct
                allocation = min(allocation, max_size)
                
                if allocation >= self.min_position_size:
                    allocations[opp['id']] = allocation
                    
            return allocations

        except Exception as e:
            print(f"Risk parity allocation error: {e}")
            return {}

    # Add these methods to existing PortfolioManager class
    
    async def update_portfolio(self, trade: Dict) -> None:
        """Update portfolio with new trade"""
        try:
            token = trade['token_address']
            
            if trade['side'] == 'buy':
                # Add or update position
                if token in self.positions:
                    position = self.positions[token]
                    # Average in
                    total_cost = position.cost + trade['cost']
                    total_size = position.size + trade['amount']
                    position.entry_price = total_cost / total_size
                    position.size = total_size
                    position.cost = total_cost
                else:
                    # New position
                    self.positions[token] = Position(
                        id=trade.get('id', str(uuid.uuid4())),
                        token_address=token,
                        pair_address=trade.get('pair_address', ''),
                        chain=trade.get('chain', 'eth'),
                        entry_price=trade['price'],
                        current_price=trade['price'],
                        size=trade['amount'],
                        cost=trade['cost'],
                        entry_time=datetime.now(),
                        last_updated=datetime.now(),
                        stop_loss=trade.get('stop_loss', 0),
                        take_profits=trade.get('take_profits', []),
                        tp_hit=[False] * len(trade.get('take_profits', [])),
                        strategy=trade.get('strategy', 'unknown')
                    )
                
                # Update balance
                self.balance -= trade['cost']
                
            elif trade['side'] == 'sell':
                if token in self.positions:
                    position = self.positions[token]
                    
                    if trade['amount'] >= position.size:
                        # Close position
                        self.balance += trade['proceeds']
                        del self.positions[token]
                    else:
                        # Partial close
                        position.size -= trade['amount']
                        self.balance += trade['proceeds']

                        # 🆕 PATCH: Log portfolio performance
                        self.log_current_performance()
                        
            # Update trade history
            self.trade_history.append(trade)
            
            # Update performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            print(f"Portfolio update error: {e}")
    
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value"""
        try:
            positions_value = sum(p.value for p in self.positions.values())
            total_value = Decimal(str(self.balance + positions_value))
            return total_value
            
        except Exception as e:
            print(f"Portfolio value calculation error: {e}")
            return Decimal("0")
    
    async def rebalance_portfolio(self) -> Dict:
        """Rebalance portfolio based on strategy"""
        try:
            rebalance_actions = []
            current_allocations = {}
            target_allocations = {}
            
            # Calculate current allocations
            total_value = float(await self.get_portfolio_value())
            for token, position in self.positions.items():
                current_allocations[token] = position.value / total_value
            
            # Determine target allocations based on strategy
            if self.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
                n_positions = len(self.positions)
                if n_positions > 0:
                    target_weight = 1.0 / n_positions
                    for token in self.positions:
                        target_allocations[token] = target_weight
                        
            # Calculate rebalancing trades
            for token in self.positions:
                current = current_allocations.get(token, 0)
                target = target_allocations.get(token, 0)
                diff = target - current
                
                if abs(diff) > 0.05:  # 5% threshold
                    value_diff = diff * total_value
                    rebalance_actions.append({
                        'token': token,
                        'action': 'buy' if diff > 0 else 'sell',
                        'amount': abs(value_diff)
                    })
            
            return {
                'current_allocations': current_allocations,
                'target_allocations': target_allocations,
                'actions': rebalance_actions,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Rebalancing error: {e}")
            return {}
    
    async def calculate_allocation(self, token: str) -> Decimal:
        """Calculate allocation for a specific token"""
        try:
            total_value = await self.get_portfolio_value()
            
            if total_value == 0:
                return Decimal("0")
            
            if token in self.positions:
                position_value = Decimal(str(self.positions[token].value))
                allocation = position_value / total_value
                return allocation
            
            return Decimal("0")
            
        except Exception as e:
            print(f"Allocation calculation error: {e}")
            return Decimal("0")
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        try:
            total_value = sum(p.value for p in self.positions.values()) + self.balance
            positions_value = sum(p.value for p in self.positions.values())
            unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
            
            # Calculate realized PnL from trade history
            realized_pnl = sum(
                t.get('pnl', 0) for t in self.trade_history 
                if t.get('side') == 'sell'
            )
            
            # Calculate win rate
            winning_trades = sum(
                1 for t in self.trade_history 
                if t.get('pnl', 0) > 0 and t.get('side') == 'sell'
            )
            total_closed = sum(
                1 for t in self.trade_history 
                if t.get('side') == 'sell'
            )
            win_rate = winning_trades / total_closed if total_closed > 0 else 0
            
            # Calculate drawdown
            drawdown = (self.peak_balance - total_value) / self.peak_balance if self.peak_balance > 0 else 0
            
            return PortfolioMetrics(
                total_value=float(total_value),
                available_balance=float(self.balance),
                positions_value=float(positions_value),
                unrealized_pnl=float(unrealized_pnl),
                realized_pnl=float(realized_pnl),
                total_pnl=float(unrealized_pnl + realized_pnl),
                win_rate=win_rate,
                average_win=self._calculate_avg_win(),
                average_loss=self._calculate_avg_loss(),
                profit_factor=self._calculate_profit_factor(),
                sharpe_ratio=0.0,  # Would need returns history
                max_drawdown=float(drawdown),
                positions_count=len(self.positions),
                risk_exposure=self._calculate_risk_exposure(),
                diversification_score=self._calculate_diversification_score()
            ).__dict__
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return {}

    def get_chain_metrics(self) -> Dict[str, Dict]:
        """
        Get portfolio metrics broken down by chain
        
        Returns:
            Dictionary with per-chain metrics:
            {
                'ethereum': {
                    'positions': 2,
                    'value': 150.0,
                    'cost': 140.0,
                    'unrealized_pnl': 10.0,
                    'allocation': 0.375,
                    'roi': 7.14
                },
                'base': {...},
                'bsc': {...},
                'solana': {...}
            }
        """
        try:
            chain_data = defaultdict(lambda: {
                'positions': 0,
                'value': 0.0,
                'cost': 0.0,
                'unrealized_pnl': 0.0
            })
            
            # Aggregate positions by chain
            for position in self.positions.values():
                chain = position.chain.lower()
                chain_data[chain]['positions'] += 1
                chain_data[chain]['value'] += position.value
                chain_data[chain]['cost'] += position.cost
                chain_data[chain]['unrealized_pnl'] += position.unrealized_pnl
            
            # Calculate total portfolio value
            total_value = self.balance + sum(p.value for p in self.positions.values())
            
            # Calculate derived metrics for each chain
            result = {}
            for chain, data in chain_data.items():
                # Allocation percentage
                data['allocation'] = data['value'] / total_value if total_value > 0 else 0.0
                
                # ROI percentage
                data['roi'] = ((data['value'] - data['cost']) / data['cost'] * 100 
                              if data['cost'] > 0 else 0.0)
                
                result[chain] = data
            
            return result
            
        except Exception as e:
            print(f"Error calculating chain metrics: {e}")
            return {}
    
    def get_chain_balance(self, chain: str) -> float:
        """
        Get allocated balance for a specific chain
        
        Args:
            chain: Chain name (ethereum, bsc, base, solana, etc.)
            
        Returns:
            Total value of positions on that chain
        """
        try:
            chain = chain.lower()
            chain_value = sum(
                p.value for p in self.positions.values() 
                if p.chain.lower() == chain
            )
            return float(chain_value)
            
        except Exception as e:
            print(f"Error calculating chain balance: {e}")
            return 0.0
    
    async def check_diversification(self) -> bool:
        """Check if portfolio is properly diversified"""
        try:
            if len(self.positions) < 3:
                return False  # Need at least 3 positions for diversification
            
            # Check allocation concentration
            total_value = await self.get_portfolio_value()
            max_allocation = 0
            
            for position in self.positions.values():
                allocation = position.value / float(total_value)
                max_allocation = max(max_allocation, allocation)
            
            # No single position should be more than 30%
            if max_allocation > 0.3:
                return False
            
            # Check chain diversification
            chains = set(p.chain for p in self.positions.values())
            if len(chains) < 2:
                return False  # Should have positions on multiple chains
            
            return True
            
        except Exception as e:
            print(f"Diversification check error: {e}")
            return False

    # Add this method to the PortfolioManager class in portfolio_manager.py

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary
        
        Returns:
            Dictionary with portfolio summary data
        """
        try:
            metrics = self.get_portfolio_metrics()
            
            return {
                'total_value': metrics.get('total_value', 0),
                'cash_balance': self.balance,
                'positions_value': metrics.get('positions_value', 0),
                'unrealized_pnl': metrics.get('unrealized_pnl', 0),
                'realized_pnl': metrics.get('realized_pnl', 0),
                'daily_pnl': metrics.get('total_pnl', 0),  # For simplicity, using total_pnl
                'net_profit': metrics.get('total_pnl', 0),
                'open_positions': metrics.get('positions_count', 0),
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'average_win': metrics.get('average_win', 0),
                'average_loss': metrics.get('average_loss', 0),
            }
            
        except Exception as e:
            print(f"Error getting portfolio summary: {e}")
            return {
                'total_value': float(self.balance),
                'cash_balance': float(self.balance),
                'positions_value': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'daily_pnl': 0,
                'net_profit': 0,
                'open_positions': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
            }

    def get_open_positions(self) -> List[Dict]:
        """Get list of open positions"""
        try:
            positions_list = []
            for token, position in self.positions.items():
                positions_list.append({
                    'id': position.id,
                    'token_address': position.token_address,
                    'pair_address': position.pair_address,
                    'chain': position.chain,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'size': position.size,
                    'value': position.value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'pnl_percentage': position.pnl_percentage,
                    'stop_loss': position.stop_loss,
                    'take_profits': position.take_profits,
                    'strategy': position.strategy,
                    'age': str(position.age),
                    'status': position.status
                })
            return positions_list
        except Exception as e:
            print(f"Error getting open positions: {e}")
            return []

    def get_position(self, position_id: str = None, 
                    token_address: str = None) -> Optional[Position]:
        """
        Get a position by ID or token address
        
        Args:
            position_id: Position ID to search for
            token_address: Token address to search for
            
        Returns:
            Position object or None if not found
        """
        if position_id:
            for position in self.positions.values():
                if position.id == position_id:
                    return position
        
        if token_address:
            return self.positions.get(token_address)
        
        return None

    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return self.get_portfolio_metrics()

    def close_position(self, position_id: str) -> Dict:
        """Close a specific position with proper P&L tracking"""
        try:
            # Find position by ID
            for token, position in list(self.positions.items()):
                if position.id == position_id:
                    # ✅ FIXED: Proper P&L tracking
                    
                    # Calculate realized P&L
                    realized_pnl = position.value - position.cost
                    pnl_percentage = ((position.value - position.cost) / position.cost * 100 
                                     if position.cost > 0 else 0)
                    
                    # Create trade record
                    trade = {
                        'id': position.id,
                        'token_address': token,
                        'pair_address': position.pair_address,
                        'chain': position.chain,
                        'side': 'sell',
                        'price': position.current_price,
                        'amount': position.size,
                        'cost': position.cost,
                        'proceeds': position.value,
                        'pnl': realized_pnl,
                        'pnl_percentage': pnl_percentage,
                        'strategy': position.strategy,
                        'timestamp': datetime.now(),
                        'entry_time': position.entry_time,
                        'exit_reason': 'manual_close'
                    }
                    
                    # Add to trade history
                    self.trade_history.append(trade)
                    
                    # Update consecutive losses tracking
                    if realized_pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Update balance
                    self.balance += position.value
                    
                    # Update peak/valley
                    current_value = self.balance + sum(p.value for p in self.positions.values() if p.id != position_id)
                    if current_value > self.peak_balance:
                        self.peak_balance = current_value
                    if current_value < self.valley_balance:
                        self.valley_balance = current_value
                    
                    # Remove position
                    del self.positions[token]
                    
                    # Log the trade exit
                    try:
                        from monitoring.logger import log_trade_exit
                        log_trade_exit(
                            token_address=token,
                            chain=position.chain,
                            exit_price=position.current_price,
                            amount=position.size,
                            pnl=realized_pnl,
                            reason='manual_close'
                        )
                    except Exception as log_error:
                        print(f"Warning: Could not log trade exit: {log_error}")
                    
                    return {
                        'success': True,
                        'position_id': position_id,
                        'message': 'Position closed successfully',
                        'pnl': realized_pnl,
                        'pnl_percentage': pnl_percentage,
                        'proceeds': position.value
                    }
            
            return {
                'success': False,
                'error': 'Position not found'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)

    def update_position(self, position_id: str, modifications: Dict) -> Dict:
        """Update position parameters"""
        try:
            # Find and update position
            for token, position in self.positions.items():
                if position.id == position_id:
                    # Update stop loss
                    if 'stop_loss' in modifications:
                        position.stop_loss = modifications['stop_loss']
                    
                    # Update take profits
                    if 'take_profits' in modifications:
                        position.take_profits = modifications['take_profits']
                    
                    # ✅ UPDATE: Current price and P&L
                    if 'current_price' in modifications:
                        position.current_price = float(modifications['current_price'])
                        # Recalculate P&L
                        position.pnl = ((position.current_price - position.entry_price) 
                                       * position.size)
                        if position.entry_price > 0:
                            position.pnl_percentage = ((position.current_price - 
                                                       position.entry_price) / 
                                                      position.entry_price * 100)
                    
                    # Update timestamp
                    position.last_updated = datetime.now()
                    
                    return {
                        'success': True,
                        'position_id': position_id,
                        'message': 'Position updated successfully'
                    }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def update_all_positions(self, price_data: Dict[str, float]) -> int:
        """
        Update all open positions with current prices
        
        Args:
            price_data: Dictionary mapping token_address to current_price
                       {token_address: current_price, ...}
            
        Returns:
            Number of positions updated
        """
        updated_count = 0
        
        try:
            for token_address, position in self.positions.items():
                if token_address in price_data:
                    new_price = float(price_data[token_address])
                    
                    # Skip if price hasn't changed (within 0.1%)
                    if abs(new_price - position.current_price) / position.current_price < 0.001:
                        continue
                    
                    # Update position using update_position method
                    result = self.update_position(
                        position.id,
                        {'current_price': new_price}
                    )
                    
                    if result.get('success'):
                        updated_count += 1
            
            return updated_count
            
        except Exception as e:
            print(f"Error updating positions: {e}")
            return updated_count

    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        winning_trades = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0]
        return sum(winning_trades) / len(winning_trades) if winning_trades else 0

    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        losing_trades = [t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0]
        return sum(losing_trades) / len(losing_trades) if losing_trades else 0

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0

    def _calculate_risk_exposure(self) -> float:
        """Calculate total risk exposure"""
        total_value = self.balance + sum(p.value for p in self.positions.values())
        positions_value = sum(p.value for p in self.positions.values())
        return positions_value / total_value if total_value > 0 else 0

    def _calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score"""
        if len(self.positions) == 0:
            return 0
        # Simple diversification: more positions = better diversification
        return min(len(self.positions) / self.max_positions, 1.0)

    def _calculate_total_risk(self) -> float:
        """Calculate total portfolio risk"""
        return self._calculate_risk_exposure()

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit"""
        # Simplified check
        daily_pnl = sum(t.get('pnl', 0) for t in self.trade_history if t.get('side') == 'sell')
        daily_loss = abs(daily_pnl) if daily_pnl < 0 else 0
        return daily_loss > (self.balance * self.daily_loss_limit)

    async def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            current_value = self.balance + sum(p.value for p in self.positions.values())
            
            # Update peak
            if current_value > self.peak_balance:
                self.peak_balance = current_value
            
            # Update valley
            if current_value < self.valley_balance:
                self.valley_balance = current_value
                
        except Exception as e:
            print(f"Error updating performance metrics: {e}")

    def log_current_performance(self):
        """Log current portfolio performance"""
        try:
            stats = self.get_statistics()
            
            log_portfolio_update(
                balance_before=float(self.initial_balance),
                balance_after=float(self.balance),
                trades_executed=stats.get('total_trades', 0),
                wins=stats.get('winning_trades', 0),
                losses=stats.get('losing_trades', 0),
                total_pnl=float(self.balance - self.initial_balance)
            )
        except Exception as e:
            logger.warning(f"Failed to log portfolio update: {e}")
