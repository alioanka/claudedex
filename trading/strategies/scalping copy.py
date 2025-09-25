"""
Scalping Trading Strategy for DexScreener Bot
High-frequency, low-timeframe trading for quick profits
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class ScalpType(Enum):
    """Types of scalping opportunities"""
    SPREAD = "spread"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    ARBITRAGE = "arbitrage"
    LIQUIDITY = "liquidity"
    MICROSTRUCTURE = "microstructure"

class OrderBookImbalance(Enum):
    """Order book imbalance types"""
    STRONG_BID = "strong_bid"
    STRONG_ASK = "strong_ask"
    NEUTRAL = "neutral"

@dataclass
class ScalpSignal:
    """Scalping signal data"""
    timestamp: datetime
    token_address: str
    signal_type: ScalpType
    entry_price: Decimal
    target_price: Decimal
    stop_loss: Decimal
    expected_duration: int  # seconds
    confidence: float
    spread: float
    liquidity_depth: Decimal
    order_book_imbalance: OrderBookImbalance
    execution_priority: int  # 1-10, higher is more urgent
    gas_price: Decimal
    slippage_tolerance: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MicrostructureData:
    """Market microstructure analysis"""
    bid_ask_spread: float
    effective_spread: float
    quoted_spread: float
    depth_imbalance: float
    order_flow_toxicity: float
    price_impact: float
    tick_volatility: float

@dataclass
class ScalpMetrics:
    """Scalping performance metrics"""
    win_rate: float
    average_profit: Decimal
    average_duration: float
    sharpe_ratio: float
    max_drawdown: float
    trades_per_hour: float
    success_by_type: Dict[ScalpType, float]

class ScalpingStrategy:
    """
    High-frequency scalping strategy for quick profits on DEX
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize scalping strategy"""
        self.config = config or self._default_config()
        self.active_scalps: Dict[str, ScalpSignal] = {}
        self.completed_scalps: Deque[Dict] = deque(maxlen=1000)
        self.order_book_cache: Dict[str, Dict] = {}
        self.tick_data: Dict[str, Deque] = {}
        self.performance_tracker = ScalpMetrics(0, Decimal("0"), 0, 0, 0, 0, {})
        self._initialize_components()
        
    def _default_config(self) -> Dict:
        """Default configuration for scalping"""
        return {
            # Scalping parameters
            "min_spread": 0.002,  # 0.2% minimum spread
            "max_spread": 0.05,   # 5% maximum spread
            "target_profit": 0.005,  # 0.5% target per trade
            "stop_loss": 0.003,   # 0.3% stop loss
            "max_position_time": 300,  # 5 minutes max
            
            # Execution parameters
            "min_liquidity": 10000,  # $10k minimum liquidity
            "max_slippage": 0.002,   # 0.2% max slippage
            "gas_multiplier": 1.2,    # Gas price multiplier for priority
            "execution_timeout": 30,   # 30 seconds timeout
            
            # Risk parameters
            "max_concurrent_scalps": 5,
            "max_exposure": 0.1,  # 10% of capital
            "position_size": 0.02,  # 2% per scalp
            "correlation_limit": 0.7,  # Max correlation between positions
            
            # Market microstructure
            "min_tick_data": 50,  # Minimum ticks for analysis
            "orderbook_depth": 10,  # Levels to analyze
            "imbalance_threshold": 0.6,  # Order book imbalance threshold
            "toxicity_threshold": 0.3,  # Max order flow toxicity
            
            # Strategy specific
            "momentum_lookback": 60,  # Seconds
            "reversal_sensitivity": 0.7,
            "arbitrage_threshold": 0.003,  # 0.3% price difference
            "microstructure_weight": 0.4
        }
    
    def _initialize_components(self):
        """Initialize strategy components"""
        self.spread_analyzer = SpreadAnalyzer(self.config)
        self.microstructure_analyzer = MicrostructureAnalyzer(self.config)
        self.execution_optimizer = ExecutionOptimizer(self.config)
        
    async def analyze(self, market_data: Dict) -> Optional[ScalpSignal]:
        """
        Analyze market for scalping opportunities
        
        Args:
            market_data: Real-time market data including order book
            
        Returns:
            ScalpSignal if opportunity found
        """
        try:
            token_address = market_data.get("token_address")
            if not token_address:
                return None
            
            # Update tick data
            await self._update_tick_data(token_address, market_data)
            
            # Check if we have enough data
            if not self._has_sufficient_data(token_address):
                return None
            
            # Analyze market microstructure
            microstructure = await self._analyze_microstructure(market_data)
            
            # Check for different scalp types
            signals = await asyncio.gather(
                self._check_spread_scalp(market_data, microstructure),
                self._check_momentum_scalp(market_data, microstructure),
                self._check_reversal_scalp(market_data, microstructure),
                self._check_arbitrage_scalp(market_data, microstructure),
                self._check_liquidity_scalp(market_data, microstructure),
                return_exceptions=True
            )
            
            # Filter valid signals
            valid_signals = [s for s in signals if isinstance(s, ScalpSignal)]
            
            if not valid_signals:
                return None
            
            # Select best opportunity
            best_signal = await self._select_best_scalp(valid_signals, market_data)
            
            # Validate execution feasibility
            if await self._validate_execution(best_signal, market_data):
                # Optimize execution parameters
                optimized_signal = await self._optimize_execution(best_signal, market_data)
                
                # Store active scalp
                self.active_scalps[token_address] = optimized_signal
                
                return optimized_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing scalp opportunity: {e}")
            return None
    
    async def _update_tick_data(self, token_address: str, market_data: Dict):
        """Update tick data for token"""
        if token_address not in self.tick_data:
            self.tick_data[token_address] = deque(maxlen=1000)
        
        tick = {
            "timestamp": datetime.utcnow(),
            "price": market_data.get("price"),
            "volume": market_data.get("volume"),
            "bid": market_data.get("bid"),
            "ask": market_data.get("ask"),
            "spread": market_data.get("spread")
        }
        
        self.tick_data[token_address].append(tick)
    
    def _has_sufficient_data(self, token_address: str) -> bool:
        """Check if we have enough tick data"""
        if token_address not in self.tick_data:
            return False
        return len(self.tick_data[token_address]) >= self.config["min_tick_data"]
    
    async def _analyze_microstructure(self, market_data: Dict) -> MicrostructureData:
        """Analyze market microstructure"""
        try:
            order_book = market_data.get("order_book", {})
            
            # Calculate spreads
            bid_ask_spread = self._calculate_bid_ask_spread(order_book)
            effective_spread = self._calculate_effective_spread(market_data)
            quoted_spread = self._calculate_quoted_spread(order_book)
            
            # Calculate depth imbalance
            depth_imbalance = self._calculate_depth_imbalance(order_book)
            
            # Calculate order flow toxicity
            toxicity = await self._calculate_order_flow_toxicity(market_data)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(order_book)
            
            # Calculate tick volatility
            tick_volatility = self._calculate_tick_volatility(
                self.tick_data.get(market_data["token_address"], [])
            )
            
            return MicrostructureData(
                bid_ask_spread=bid_ask_spread,
                effective_spread=effective_spread,
                quoted_spread=quoted_spread,
                depth_imbalance=depth_imbalance,
                order_flow_toxicity=toxicity,
                price_impact=price_impact,
                tick_volatility=tick_volatility
            )
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return MicrostructureData(0, 0, 0, 0, 0, 0, 0)
    
    async def _check_spread_scalp(
        self,
        market_data: Dict,
        microstructure: MicrostructureData
    ) -> Optional[ScalpSignal]:
        """Check for spread capture opportunity"""
        try:
            spread = microstructure.bid_ask_spread
            
            # Check if spread is within profitable range
            if spread < self.config["min_spread"] or spread > self.config["max_spread"]:
                return None
            
            # Check liquidity
            liquidity = Decimal(str(market_data.get("liquidity", 0)))
            if liquidity < self.config["min_liquidity"]:
                return None
            
            # Check for stable spread
            if microstructure.tick_volatility > 0.02:  # 2% volatility threshold
                return None
            
            # Calculate entry and exit prices
            bid = Decimal(str(market_data.get("bid", 0)))
            ask = Decimal(str(market_data.get("ask", 0)))
            mid_price = (bid + ask) / 2
            
            # Buy at bid, sell at ask strategy
            entry_price = bid * Decimal("1.001")  # Slightly above bid
            target_price = ask * Decimal("0.999")  # Slightly below ask
            stop_loss = bid * Decimal("0.997")    # Below bid
            
            # Calculate expected profit
            expected_profit = float((target_price - entry_price) / entry_price)
            if expected_profit < self.config["target_profit"]:
                return None
            
            return ScalpSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=ScalpType.SPREAD,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_duration=30,  # Quick in and out
                confidence=0.7 + (0.3 * (1 - microstructure.tick_volatility)),
                spread=spread,
                liquidity_depth=liquidity,
                order_book_imbalance=self._determine_imbalance(microstructure.depth_imbalance),
                execution_priority=8,
                gas_price=Decimal(str(market_data.get("gas_price", 0))),
                slippage_tolerance=self.config["max_slippage"],
                metadata={
                    "reversal_type": reversal_type,
                    "z_score": z_score,
                    "rsi": rsi,
                    "mean_price": price_mean
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking reversal scalp: {e}")
            return None
    
    async def _check_arbitrage_scalp(
        self,
        market_data: Dict,
        microstructure: MicrostructureData
    ) -> Optional[ScalpSignal]:
        """Check for arbitrage opportunities across DEXs"""
        try:
            # Get prices from multiple DEXs
            dex_prices = market_data.get("dex_prices", {})
            if len(dex_prices) < 2:
                return None
            
            # Find price discrepancies
            prices = list(dex_prices.values())
            min_price = min(prices)
            max_price = max(prices)
            
            price_diff = (max_price - min_price) / min_price
            
            # Check if profitable after fees
            if price_diff < self.config["arbitrage_threshold"]:
                return None
            
            # Identify DEXs
            buy_dex = min(dex_prices, key=dex_prices.get)
            sell_dex = max(dex_prices, key=dex_prices.get)
            
            # Calculate execution prices with slippage
            entry_price = Decimal(str(min_price)) * Decimal("1.002")  # Buy with slippage
            target_price = Decimal(str(max_price)) * Decimal("0.998")  # Sell with slippage
            stop_loss = entry_price * Decimal("0.995")
            
            # Check profitability after gas
            gas_cost = float(market_data.get("gas_price", 0)) * 2  # Two transactions
            expected_profit = float(target_price - entry_price) - gas_cost
            
            if expected_profit < float(entry_price) * self.config["target_profit"]:
                return None
            
            return ScalpSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=ScalpType.ARBITRAGE,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_duration=15,  # Very quick execution needed
                confidence=0.9 if price_diff > 0.01 else 0.7,
                spread=microstructure.bid_ask_spread,
                liquidity_depth=Decimal(str(market_data.get("liquidity", 0))),
                order_book_imbalance=OrderBookImbalance.NEUTRAL,
                execution_priority=10,  # Highest priority
                gas_price=Decimal(str(market_data.get("gas_price", 0))) * Decimal("1.5"),  # Higher gas for speed
                slippage_tolerance=self.config["max_slippage"] * 2,
                metadata={
                    "buy_dex": buy_dex,
                    "sell_dex": sell_dex,
                    "price_difference": price_diff,
                    "expected_profit": expected_profit
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking arbitrage scalp: {e}")
            return None
    
    async def _check_liquidity_scalp(
        self,
        market_data: Dict,
        microstructure: MicrostructureData
    ) -> Optional[ScalpSignal]:
        """Check for liquidity provision scalping"""
        try:
            # Check if there's a liquidity imbalance to exploit
            order_book = market_data.get("order_book", {})
            
            bid_liquidity = sum(order["size"] for order in order_book.get("bids", [])[:5])
            ask_liquidity = sum(order["size"] for order in order_book.get("asks", [])[:5])
            
            if bid_liquidity == 0 or ask_liquidity == 0:
                return None
            
            liquidity_ratio = bid_liquidity / ask_liquidity
            
            # Look for significant imbalances
            if 0.5 < liquidity_ratio < 2.0:
                return None  # Not enough imbalance
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            if liquidity_ratio > 2.0:
                # More bids than asks - price likely to go up
                entry_price = current_price * Decimal("1.001")
                target_price = current_price * Decimal("1.004")
                stop_loss = current_price * Decimal("0.998")
                direction = "long"
            else:
                # More asks than bids - price likely to go down
                entry_price = current_price * Decimal("0.999")
                target_price = current_price * Decimal("0.996")
                stop_loss = current_price * Decimal("1.002")
                direction = "short"
            
            return ScalpSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=ScalpType.LIQUIDITY,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_duration=45,
                confidence=min(0.6 + abs(np.log(liquidity_ratio)) * 0.2, 0.85),
                spread=microstructure.bid_ask_spread,
                liquidity_depth=Decimal(str(bid_liquidity + ask_liquidity)),
                order_book_imbalance=self._determine_imbalance(microstructure.depth_imbalance),
                execution_priority=5,
                gas_price=Decimal(str(market_data.get("gas_price", 0))),
                slippage_tolerance=self.config["max_slippage"],
                metadata={
                    "liquidity_ratio": liquidity_ratio,
                    "bid_liquidity": bid_liquidity,
                    "ask_liquidity": ask_liquidity,
                    "direction": direction
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking liquidity scalp: {e}")
            return None
    
    def _calculate_bid_ask_spread(self, order_book: Dict) -> float:
        """Calculate bid-ask spread"""
        try:
            best_bid = order_book.get("bids", [{}])[0].get("price", 0)
            best_ask = order_book.get("asks", [{}])[0].get("price", 0)
            
            if best_bid == 0:
                return 0
            
            return (best_ask - best_bid) / best_bid
        except:
            return 0
    
    def _calculate_effective_spread(self, market_data: Dict) -> float:
        """Calculate effective spread from recent trades"""
        try:
            trades = market_data.get("recent_trades", [])
            if len(trades) < 2:
                return self._calculate_bid_ask_spread(market_data.get("order_book", {}))
            
            # Calculate spread from actual execution prices
            buy_trades = [t["price"] for t in trades if t.get("side") == "buy"]
            sell_trades = [t["price"] for t in trades if t.get("side") == "sell"]
            
            if buy_trades and sell_trades:
                avg_buy = np.mean(buy_trades)
                avg_sell = np.mean(sell_trades)
                return abs(avg_sell - avg_buy) / avg_buy
            
            return self._calculate_bid_ask_spread(market_data.get("order_book", {}))
        except:
            return 0
    
    def _calculate_quoted_spread(self, order_book: Dict) -> float:
        """Calculate quoted spread at different depths"""
        try:
            spreads = []
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            for i in range(min(5, len(bids), len(asks))):
                if bids[i]["price"] > 0:
                    spread = (asks[i]["price"] - bids[i]["price"]) / bids[i]["price"]
                    spreads.append(spread)
            
            return np.mean(spreads) if spreads else 0
        except:
            return 0
    
    def _calculate_depth_imbalance(self, order_book: Dict) -> float:
        """Calculate order book depth imbalance"""
        try:
            bid_depth = sum(
                order["price"] * order["size"] 
                for order in order_book.get("bids", [])[:10]
            )
            ask_depth = sum(
                order["price"] * order["size"] 
                for order in order_book.get("asks", [])[:10]
            )
            
            total_depth = bid_depth + ask_depth
            if total_depth == 0:
                return 0
            
            return (bid_depth - ask_depth) / total_depth  # -1 to 1
        except:
            return 0
    
    async def _calculate_order_flow_toxicity(self, market_data: Dict) -> float:
        """Calculate VPIN or similar toxicity measure"""
        try:
            trades = market_data.get("recent_trades", [])
            if len(trades) < 20:
                return 0
            
            # Classify trades as buyer or seller initiated
            buy_volume = sum(t["size"] for t in trades if t.get("side") == "buy")
            sell_volume = sum(t["size"] for t in trades if t.get("side") == "sell")
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0
            
            # Simple toxicity measure based on volume imbalance
            toxicity = abs(buy_volume - sell_volume) / total_volume
            
            return min(toxicity, 1.0)
        except:
            return 0
    
    def _calculate_price_impact(self, order_book: Dict) -> float:
        """Calculate expected price impact for typical trade size"""
        try:
            # Assume typical trade size
            trade_size = 1000  # $1000
            
            # Calculate impact for buy
            buy_impact = self._walk_order_book(
                order_book.get("asks", []), 
                trade_size
            )
            
            # Calculate impact for sell
            sell_impact = self._walk_order_book(
                order_book.get("bids", []), 
                trade_size, 
                is_sell=True
            )
            
            return (buy_impact + sell_impact) / 2
        except:
            return 0
    
    def _walk_order_book(
        self, 
        orders: List[Dict], 
        size: float, 
        is_sell: bool = False
    ) -> float:
        """Walk through order book to calculate price impact"""
        if not orders:
            return 0
        
        initial_price = orders[0]["price"]
        remaining_size = size
        total_cost = 0
        
        for order in orders:
            order_value = order["price"] * order["size"]
            
            if order_value >= remaining_size:
                # This order can fill the remaining size
                fill_size = remaining_size / order["price"]
                total_cost += remaining_size
                break
            else:
                # Take the entire order
                total_cost += order_value
                remaining_size -= order_value
        
        if remaining_size > 0:
            # Not enough liquidity
            return 1.0  # Max impact
        
        avg_price = total_cost / size
        impact = abs(avg_price - initial_price) / initial_price
        
        return impact
    
    def _calculate_tick_volatility(self, ticks: List[Dict]) -> float:
        """Calculate tick-by-tick volatility"""
        if len(ticks) < 2:
            return 0
        
        prices = [t["price"] for t in ticks if t["price"] > 0]
        if len(prices) < 2:
            return 0
        
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        
        return np.std(returns) if returns else 0
    
    def _determine_imbalance(self, depth_imbalance: float) -> OrderBookImbalance:
        """Determine order book imbalance type"""
        if depth_imbalance > 0.3:
            return OrderBookImbalance.STRONG_BID
        elif depth_imbalance < -0.3:
            return OrderBookImbalance.STRONG_ASK
        else:
            return OrderBookImbalance.NEUTRAL
    
    def _check_orderbook_reversal_support(
        self, 
        order_book: Dict, 
        reversal_type: str
    ) -> bool:
        """Check if order book supports expected reversal"""
        try:
            if reversal_type == "overbought":
                # For overbought reversal, we want to see selling pressure building
                asks = order_book.get("asks", [])
                bids = order_book.get("bids", [])
                
                if not asks or not bids:
                    return False
                
                # Check if asks are building up
                ask_volume = sum(order["size"] for order in asks[:5])
                bid_volume = sum(order["size"] for order in bids[:5])
                
                return ask_volume > bid_volume * 1.2
                
            elif reversal_type == "oversold":
                # For oversold reversal, we want to see buying pressure building
                asks = order_book.get("asks", [])
                bids = order_book.get("bids", [])
                
                if not asks or not bids:
                    return False
                
                # Check if bids are building up
                ask_volume = sum(order["size"] for order in asks[:5])
                bid_volume = sum(order["size"] for order in bids[:5])
                
                return bid_volume > ask_volume * 1.2
                
            return False
            
        except:
            return False
    
    async def _select_best_scalp(
        self, 
        signals: List[ScalpSignal], 
        market_data: Dict
    ) -> ScalpSignal:
        """Select the best scalping opportunity"""
        scored_signals = []
        
        for signal in signals:
            # Calculate score based on multiple factors
            score = 0
            
            # Confidence weight (30%)
            score += signal.confidence * 30
            
            # Execution priority weight (25%)
            score += (signal.execution_priority / 10) * 25
            
            # Expected profit weight (25%)
            expected_profit = float(
                (signal.target_price - signal.entry_price) / signal.entry_price
            )
            score += min(expected_profit / 0.01, 1.0) * 25  # Normalize to 1% max
            
            # Liquidity weight (10%)
            liquidity_score = min(float(signal.liquidity_depth) / 100000, 1.0)
            score += liquidity_score * 10
            
            # Spread weight (10%)
            spread_score = 1 - min(signal.spread / 0.05, 1.0)  # Lower spread is better
            score += spread_score * 10
            
            scored_signals.append((score, signal))
        
        # Return highest scoring signal
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return scored_signals[0][1]
    
    async def _validate_execution(
        self, 
        signal: ScalpSignal, 
        market_data: Dict
    ) -> bool:
        """Validate if scalp can be executed"""
        try:
            # Check position limits
            if len(self.active_scalps) >= self.config["max_concurrent_scalps"]:
                return False
            
            # Check correlation with existing positions
            for active_signal in self.active_scalps.values():
                if active_signal.token_address == signal.token_address:
                    return False  # Already have position in this token
            
            # Check liquidity is sufficient
            if signal.liquidity_depth < self.config["min_liquidity"]:
                return False
            
            # Check spread isn't too wide
            if signal.spread > self.config["max_spread"]:
                return False
            
            # Check toxicity
            if signal.metadata.get("microstructure", {}).order_flow_toxicity > self.config["toxicity_threshold"]:
                return False
            
            # Check gas economics
            gas_cost = float(signal.gas_price) * 2  # Entry and exit
            min_profit = float(signal.entry_price) * self.config["target_profit"]
            if gas_cost > min_profit * 0.2:  # Gas shouldn't be more than 20% of profit
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating execution: {e}")
            return False
    
    async def _optimize_execution(
        self, 
        signal: ScalpSignal, 
        market_data: Dict
    ) -> ScalpSignal:
        """Optimize execution parameters"""
        try:
            # Optimize gas price based on urgency
            if signal.execution_priority >= 8:
                signal.gas_price *= Decimal("1.5")  # 50% higher gas for urgent trades
            elif signal.execution_priority >= 6:
                signal.gas_price *= Decimal("1.2")
            
            # Adjust slippage based on volatility
            volatility = signal.metadata.get("microstructure", {}).tick_volatility
            if volatility > 0.02:
                signal.slippage_tolerance *= 1.5
            
            # Add execution instructions
            signal.metadata["execution"] = {
                "route": self._determine_best_route(market_data),
                "batch_size": self._calculate_batch_size(signal),
                "timeout": self.config["execution_timeout"],
                "retry_count": 2 if signal.execution_priority >= 7 else 1
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error optimizing execution: {e}")
            return signal
    
    def _determine_best_route(self, market_data: Dict) -> str:
        """Determine best execution route"""
        # This would analyze multiple DEXs and aggregators
        # For now, return the DEX with best liquidity
        dex_liquidity = market_data.get("dex_liquidity", {})
        if dex_liquidity:
            return max(dex_liquidity, key=dex_liquidity.get)
        return "uniswap"  # Default
    
    def _calculate_batch_size(self, signal: ScalpSignal) -> int:
        """Calculate optimal batch size for execution"""
        # Break large orders into smaller batches
        position_value = float(signal.entry_price) * 1000  # Assuming $1000 position
        
        if position_value > 10000:
            return 5
        elif position_value > 5000:
            return 3
        else:
            return 1
    
    async def update_scalp(
        self, 
        token_address: str, 
        market_data: Dict
    ) -> Optional[Dict]:
        """Update active scalp position"""
        try:
            if token_address not in self.active_scalps:
                return None
            
            signal = self.active_scalps[token_address]
            current_price = Decimal(str(market_data.get("price", 0)))
            current_time = datetime.utcnow()
            
            # Check time limit
            time_elapsed = (current_time - signal.timestamp).total_seconds()
            if time_elapsed > self.config["max_position_time"]:
                return {
                    "action": "close",
                    "reason": "time_limit_reached",
                    "price": current_price
                }
            
            # Check stop loss
            if current_price <= signal.stop_loss:
                return {
                    "action": "close",
                    "reason": "stop_loss_hit",
                    "price": current_price
                }
            
            # Check target reached
            if current_price >= signal.target_price:
                return {
                    "action": "close",
                    "reason": "target_reached",
                    "price": current_price
                }
            
            # Check for deteriorating conditions
            microstructure = await self._analyze_microstructure(market_data)
            
            # Exit if spread widens too much
            if microstructure.bid_ask_spread > self.config["max_spread"]:
                return {
                    "action": "close",
                    "reason": "spread_too_wide",
                    "price": current_price
                }
            
            # Exit if toxicity increases
            if microstructure.order_flow_toxicity > self.config["toxicity_threshold"] * 1.5:
                return {
                    "action": "close",
                    "reason": "high_toxicity",
                    "price": current_price
                }
            
            # Dynamic stop loss adjustment for profitable positions
            profit = (current_price - signal.entry_price) / signal.entry_price
            if profit > Decimal("0.003"):  # 0.3% profit
                new_stop = signal.entry_price * Decimal("1.001")  # Move to breakeven
                if new_stop > signal.stop_loss:
                    return {
                        "action": "adjust_stop",
                        "new_stop": new_stop,
                        "reason": "trailing_stop"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating scalp: {e}")
            return None
    
    def record_completed_scalp(self, scalp_result: Dict):
        """Record completed scalp for performance tracking"""
        self.completed_scalps.append(scalp_result)
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if not self.completed_scalps:
            return
        
        # Calculate win rate
        wins = sum(1 for s in self.completed_scalps if s["pnl"] > 0)
        self.performance_tracker.win_rate = wins / len(self.completed_scalps)
        
        # Calculate average profit
        profits = [Decimal(str(s["pnl"])) for s in self.completed_scalps]
        self.performance_tracker.average_profit = sum(profits) / len(profits)
        
        # Calculate average duration
        durations = [s["duration"] for s in self.completed_scalps]
        self.performance_tracker.average_duration = np.mean(durations)
        
        # Calculate Sharpe ratio (simplified)
        if len(profits) > 1:
            returns = [float(p) for p in profits]
            self.performance_tracker.sharpe_ratio = (
                np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            )
        
        # Calculate max drawdown
        cumulative = []
        cum_sum = Decimal("0")
        for p in profits:
            cum_sum += p
            cumulative.append(cum_sum)
        
        if cumulative:
            peak = cumulative[0]
            max_dd = 0
            for val in cumulative:
                if val > peak:
                    peak = val
                dd = float((peak - val) / peak) if peak != 0 else 0
                max_dd = max(max_dd, dd)
            self.performance_tracker.max_drawdown = max_dd
        
        # Calculate trades per hour
        if self.completed_scalps:
            time_span = (
                self.completed_scalps[-1]["timestamp"] - 
                self.completed_scalps[0]["timestamp"]
            ).total_seconds() / 3600
            if time_span > 0:
                self.performance_tracker.trades_per_hour = len(self.completed_scalps) / time_span
        
        # Success by type
        for scalp in self.completed_scalps:
            scalp_type = scalp["signal_type"]
            if scalp_type not in self.performance_tracker.success_by_type:
                self.performance_tracker.success_by_type[scalp_type] = []
            self.performance_tracker.success_by_type[scalp_type].append(scalp["pnl"] > 0)
        
        for scalp_type, results in self.performance_tracker.success_by_type.items():
            if results:
                self.performance_tracker.success_by_type[scalp_type] = sum(results) / len(results)

    # PATCHES FOR scalping.py
    # The scalping.py file is mostly complete, but needs a wrapper for the analyze method
    # to match the documented signature. Add this adapter method:

    async def analyze_market_data(self, market_data: Dict) -> Optional[ScalpingOpportunity]:
        """
        Wrapper method matching documented signature
        Redirects to the actual analyze method
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            ScalpingOpportunity if found
        """
        return await self.analyze(market_data)


class SpreadAnalyzer:
    """Analyze spread trading opportunities"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def analyze(self, order_book: Dict) -> Dict:
        """Analyze spread characteristics"""
        # Implementation details for spread analysis
        pass


class MicrostructureAnalyzer:
    """Analyze market microstructure"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def analyze(self, market_data: Dict) -> Dict:
        """Analyze microstructure patterns"""
        # Implementation details for microstructure analysis
        pass


class ExecutionOptimizer:
    """Optimize trade execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def optimize(self, signal: ScalpSignal, market_data: Dict) -> Dict:
        """Optimize execution parameters"""
        # Implementation details for execution optimization
        pass_price", 0))),
                slippage_tolerance=self.config["max_slippage"],
                metadata={
                    "microstructure": microstructure,
                    "expected_profit": expected_profit
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking spread scalp: {e}")
            return None
    
    async def _check_momentum_scalp(
        self,
        market_data: Dict,
        microstructure: MicrostructureData
    ) -> Optional[ScalpSignal]:
        """Check for momentum scalping opportunity"""
        try:
            token_address = market_data["token_address"]
            ticks = self.tick_data.get(token_address, [])
            
            if len(ticks) < 10:
                return None
            
            # Calculate recent momentum
            recent_prices = [t["price"] for t in list(ticks)[-10:]]
            price_changes = [
                (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                for i in range(1, len(recent_prices))
            ]
            
            momentum = np.mean(price_changes)
            
            # Check for strong momentum
            if abs(momentum) < 0.001:  # 0.1% minimum momentum
                return None
            
            # Check order book supports momentum
            imbalance = microstructure.depth_imbalance
            if momentum > 0 and imbalance < 0.6:  # Bullish momentum needs bid support
                return None
            elif momentum < 0 and imbalance > -0.6:  # Bearish momentum needs ask pressure
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            if momentum > 0:
                # Bullish momentum scalp
                entry_price = current_price * Decimal("1.001")
                target_price = current_price * Decimal("1.006")  # 0.6% target
                stop_loss = current_price * Decimal("0.997")
            else:
                # Bearish momentum scalp (short)
                entry_price = current_price * Decimal("0.999")
                target_price = current_price * Decimal("0.994")
                stop_loss = current_price * Decimal("1.003")
            
            return ScalpSignal(
                timestamp=datetime.utcnow(),
                token_address=token_address,
                signal_type=ScalpType.MOMENTUM,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_duration=60,
                confidence=min(0.5 + abs(momentum) * 50, 0.9),
                spread=microstructure.bid_ask_spread,
                liquidity_depth=Decimal(str(market_data.get("liquidity", 0))),
                order_book_imbalance=self._determine_imbalance(imbalance),
                execution_priority=7,
                gas_price=Decimal(str(market_data.get("gas_price", 0))),
                slippage_tolerance=self.config["max_slippage"] * 1.5,  # Allow more slippage for momentum
                metadata={
                    "momentum": momentum,
                    "momentum_strength": abs(momentum),
                    "direction": "long" if momentum > 0 else "short"
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking momentum scalp: {e}")
            return None
    
    async def _check_reversal_scalp(
        self,
        market_data: Dict,
        microstructure: MicrostructureData
    ) -> Optional[ScalpSignal]:
        """Check for reversal scalping opportunity"""
        try:
            token_address = market_data["token_address"]
            ticks = list(self.tick_data.get(token_address, []))
            
            if len(ticks) < 20:
                return None
            
            # Detect potential reversal
            recent_prices = [t["price"] for t in ticks[-20:]]
            
            # Check for overextension
            price_mean = np.mean(recent_prices)
            price_std = np.std(recent_prices)
            current_price = recent_prices[-1]
            
            z_score = (current_price - price_mean) / price_std if price_std > 0 else 0
            
            # Look for extreme z-scores
            if abs(z_score) < 2:  # Not overextended enough
                return None
            
            # Check RSI for reversal confirmation
            rsi = market_data.get("technical_indicators", {}).get("rsi", 50)
            
            if z_score > 2 and rsi > 70:
                # Overbought reversal (short)
                reversal_type = "overbought"
                entry_price = Decimal(str(current_price)) * Decimal("0.999")
                target_price = Decimal(str(price_mean))
                stop_loss = Decimal(str(current_price)) * Decimal("1.005")
                
            elif z_score < -2 and rsi < 30:
                # Oversold reversal (long)
                reversal_type = "oversold"
                entry_price = Decimal(str(current_price)) * Decimal("1.001")
                target_price = Decimal(str(price_mean))
                stop_loss = Decimal(str(current_price)) * Decimal("0.995")
            else:
                return None
            
            # Check order book for reversal support
            if not self._check_orderbook_reversal_support(
                market_data.get("order_book", {}),
                reversal_type
            ):
                return None
            
            return ScalpSignal(
                timestamp=datetime.utcnow(),
                token_address=token_address,
                signal_type=ScalpType.REVERSAL,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                expected_duration=120,  # 2 minutes for reversal
                confidence=min(0.6 + abs(z_score) * 0.1, 0.85),
                spread=microstructure.bid_ask_spread,
                liquidity_depth=Decimal(str(market_data.get("liquidity", 0))),
                order_book_imbalance=self._determine_imbalance(microstructure.depth_imbalance),
                execution_priority=6,
                gas_price=Decimal(str(market_data.get("gas

#missing some things