"""
Scalping Strategy - High-frequency quick profit trading
Fast entry and exit trades targeting small price movements
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum

from utils.helpers import (
    calculate_percentage_change, calculate_ema,
    calculate_moving_average, round_to_significant_digits
)
from utils.constants import (
    DEFAULT_SLIPPAGE, MAX_SLIPPAGE, SignalStrength,
    SIGNAL_THRESHOLDS, OrderType, TradingMode
)

logger = logging.getLogger(__name__)

class ScalpingSignal(Enum):
    """Scalping signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class ScalpingOpportunity:
    """Scalping trade opportunity"""
    token_address: str
    chain: str
    signal: ScalpingSignal
    entry_price: Decimal
    target_price: Decimal
    stop_loss: Decimal
    confidence: float
    expected_profit: Decimal
    time_window: int  # seconds
    volume_profile: Dict
    indicators: Dict
    risk_score: float
    timestamp: datetime

class ScalpingStrategy:
    """High-frequency scalping trading strategy"""
    
    def __init__(self, config: Dict = None):
        """Initialize scalping strategy with configuration"""
        self.config = config or {}
        
        # Scalping parameters
        self.min_profit_target = Decimal(str(self.config.get("min_profit_target", "0.005")))  # 0.5%
        self.max_profit_target = Decimal(str(self.config.get("max_profit_target", "0.02")))   # 2%
        self.stop_loss_percent = Decimal(str(self.config.get("stop_loss_percent", "0.003")))  # 0.3%
        self.max_hold_time = self.config.get("max_hold_time", 300)  # 5 minutes max
        self.min_volume_usd = self.config.get("min_volume_usd", 50000)  # $50k minimum volume
        self.min_liquidity_usd = self.config.get("min_liquidity_usd", 100000)  # $100k minimum liquidity
        
        # Technical indicators settings
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.ema_fast = self.config.get("ema_fast", 5)
        self.ema_slow = self.config.get("ema_slow", 15)
        self.volume_ma_period = self.config.get("volume_ma_period", 20)
        
        # Risk management
        self.max_concurrent_trades = self.config.get("max_concurrent_trades", 5)
        self.max_exposure_percent = Decimal(str(self.config.get("max_exposure_percent", "0.3")))  # 30% max
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Performance tracking
        self.active_trades = {}
        self.completed_trades = []
        self.win_rate = 0.0
        self.avg_profit = Decimal("0")
        self.total_volume = Decimal("0")
        
    async def analyze(self, market_data: Dict) -> Optional[ScalpingOpportunity]:
        """
        Analyze market data for scalping opportunities
        Returns opportunity if found, None otherwise
        """
        try:
            # Validate basic requirements
            if not self._validate_market_conditions(market_data):
                return None
                
            # Extract price data
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            
            if len(prices) < 30:  # Need at least 30 data points
                return None
                
            # Calculate technical indicators
            indicators = await self._calculate_indicators(prices, volumes)
            
            # Identify entry signal
            signal = self._identify_signal(indicators, market_data)
            
            if signal == ScalpingSignal.NEUTRAL:
                return None
                
            # Calculate entry/exit points
            current_price = Decimal(str(prices[-1]))
            entry_price = self._calculate_entry_price(current_price, signal, market_data)
            target_price = self._calculate_target_price(entry_price, signal, indicators)
            stop_loss = self._calculate_stop_loss(entry_price, signal)
            
            # Analyze volume profile
            volume_profile = await self._analyze_volume_profile(volumes, prices)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(indicators, volume_profile, market_data)
            
            if confidence < self.confidence_threshold:
                return None
                
            # Calculate expected profit
            expected_profit = self._calculate_expected_profit(
                entry_price, target_price, stop_loss, confidence
            )
            
            # Assess risk
            risk_score = self._assess_risk(market_data, indicators, volume_profile)
            
            # Determine time window for trade
            time_window = self._determine_time_window(indicators, volume_profile)
            
            return ScalpingOpportunity(
                token_address=market_data.get("token_address"),
                chain=market_data.get("chain"),
                signal=signal,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                expected_profit=expected_profit,
                time_window=time_window,
                volume_profile=volume_profile,
                indicators=indicators,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Scalping analysis failed: {e}")
            return None
            
    def _validate_market_conditions(self, market_data: Dict) -> bool:
        """Validate if market conditions are suitable for scalping"""
        # Check volume
        volume_24h = market_data.get("volume_24h", 0)
        if volume_24h < self.min_volume_usd:
            return False
            
        # Check liquidity
        liquidity = market_data.get("liquidity", 0)
        if liquidity < self.min_liquidity_usd:
            return False
            
        # Check spread
        spread = market_data.get("spread", 0)
        if spread > 0.02:  # 2% max spread
            return False
            
        # Check volatility (good for scalping)
        volatility = market_data.get("volatility", 0)
        if volatility < 0.005:  # Need at least 0.5% volatility
            return False
            
        return True
        
    async def _calculate_indicators(self, prices: List[float], volumes: List[float]) -> Dict:
        """Calculate technical indicators for scalping"""
        indicators = {}
        
        # Price-based indicators
        indicators["ema_fast"] = calculate_ema(prices, self.ema_fast)
        indicators["ema_slow"] = calculate_ema(prices, self.ema_slow)
        
        # RSI
        indicators["rsi"] = self._calculate_rsi(prices, self.rsi_period)
        
        # MACD
        indicators["macd"], indicators["signal"], indicators["histogram"] = self._calculate_macd(prices)
        
        # Bollinger Bands
        indicators["bb_upper"], indicators["bb_middle"], indicators["bb_lower"] = self._calculate_bollinger_bands(prices)
        
        # Volume indicators
        indicators["volume_ma"] = calculate_moving_average(tuple(volumes), self.volume_ma_period)
        indicators["volume_ratio"] = volumes[-1] / indicators["volume_ma"] if indicators["volume_ma"] > 0 else 1
        
        # Price action
        indicators["price_change_1m"] = calculate_percentage_change(
            Decimal(str(prices[-2])), Decimal(str(prices[-1]))
        )
        indicators["price_change_5m"] = calculate_percentage_change(
            Decimal(str(prices[-6])), Decimal(str(prices[-1]))
        ) if len(prices) >= 6 else Decimal("0")
        
        # Support/Resistance levels
        indicators["support"], indicators["resistance"] = self._identify_support_resistance(prices)
        
        # Momentum
        indicators["momentum"] = self._calculate_momentum(prices)
        
        return indicators
        
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def _calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD indicator"""
        ema_12 = calculate_ema(prices, 12)
        ema_26 = calculate_ema(prices, 26)
        
        macd = [ema_12[i] - ema_26[i] for i in range(len(ema_12))]
        signal = calculate_ema(macd, 9)
        histogram = [macd[i] - signal[i] for i in range(len(signal))]
        
        return macd, signal, histogram
        
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return [], [], []
            
        sma = []
        upper = []
        lower = []
        
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window)
            
            sma.append(mean)
            upper.append(mean + (std * std_dev))
            lower.append(mean - (std * std_dev))
            
        return upper, sma, lower
        
    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate price momentum"""
        if len(prices) < period:
            return 0.0
            
        return ((prices[-1] / prices[-period]) - 1) * 100
        
    def _identify_support_resistance(self, prices: List[float]) -> Tuple[float, float]:
        """Identify support and resistance levels"""
        if len(prices) < 20:
            return min(prices), max(prices)
            
        # Use recent lows for support, highs for resistance
        recent_prices = prices[-20:]
        support = np.percentile(recent_prices, 20)
        resistance = np.percentile(recent_prices, 80)
        
        return support, resistance
        
    async def _analyze_volume_profile(self, volumes: List[float], prices: List[float]) -> Dict:
        """Analyze volume profile for better entry/exit"""
        profile = {
            "avg_volume": np.mean(volumes) if volumes else 0,
            "current_volume": volumes[-1] if volumes else 0,
            "volume_trend": "neutral",
            "volume_spike": False,
            "buy_pressure": 0.5,
            "sell_pressure": 0.5
        }
        
        if len(volumes) < 10:
            return profile
            
        # Determine volume trend
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-10:-5])
        
        if recent_avg > older_avg * 1.2:
            profile["volume_trend"] = "increasing"
        elif recent_avg < older_avg * 0.8:
            profile["volume_trend"] = "decreasing"
            
        # Check for volume spike
        if volumes[-1] > profile["avg_volume"] * 2:
            profile["volume_spike"] = True
            
        # Estimate buy/sell pressure from price-volume correlation
        price_changes = [prices[i] - prices[i-1] for i in range(1, min(len(prices), len(volumes)))]
        
        buy_volume = sum(volumes[i] for i in range(len(price_changes)) if price_changes[i] > 0)
        sell_volume = sum(volumes[i] for i in range(len(price_changes)) if price_changes[i] < 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            profile["buy_pressure"] = buy_volume / total_volume
            profile["sell_pressure"] = sell_volume / total_volume
            
        return profile
        
    def _identify_signal(self, indicators: Dict, market_data: Dict) -> ScalpingSignal:
        """Identify trading signal based on indicators"""
        signals = []
        
        # RSI signals
        rsi = indicators.get("rsi", 50)
        if rsi < self.rsi_oversold:
            signals.append(ScalpingSignal.BUY)
        elif rsi > self.rsi_overbought:
            signals.append(ScalpingSignal.SELL)
            
        # EMA crossover signals
        if indicators.get("ema_fast") and indicators.get("ema_slow"):
            if indicators["ema_fast"][-1] > indicators["ema_slow"][-1]:
                if len(indicators["ema_fast"]) > 1 and indicators["ema_fast"][-2] <= indicators["ema_slow"][-2]:
                    signals.append(ScalpingSignal.STRONG_BUY)
                else:
                    signals.append(ScalpingSignal.BUY)
            elif indicators["ema_fast"][-1] < indicators["ema_slow"][-1]:
                if len(indicators["ema_fast"]) > 1 and indicators["ema_fast"][-2] >= indicators["ema_slow"][-2]:
                    signals.append(ScalpingSignal.STRONG_SELL)
                else:
                    signals.append(ScalpingSignal.SELL)
                    
        # MACD signals
        if indicators.get("histogram"):
            if indicators["histogram"][-1] > 0 and (len(indicators["histogram"]) < 2 or indicators["histogram"][-2] <= 0):
                signals.append(ScalpingSignal.BUY)
            elif indicators["histogram"][-1] < 0 and (len(indicators["histogram"]) < 2 or indicators["histogram"][-2] >= 0):
                signals.append(ScalpingSignal.SELL)
                
        # Bollinger Band signals
        current_price = market_data.get("current_price", 0)
        if indicators.get("bb_lower") and current_price < indicators["bb_lower"][-1]:
            signals.append(ScalpingSignal.BUY)
        elif indicators.get("bb_upper") and current_price > indicators["bb_upper"][-1]:
            signals.append(ScalpingSignal.SELL)
            
        # Volume confirmation
        volume_ratio = indicators.get("volume_ratio", 1)
        if volume_ratio < 0.5:
            # Low volume, reduce signal strength
            signals = [ScalpingSignal.NEUTRAL]
            
        # Aggregate signals
        if not signals:
            return ScalpingSignal.NEUTRAL
            
        buy_signals = sum(1 for s in signals if s in [ScalpingSignal.BUY, ScalpingSignal.STRONG_BUY])
        sell_signals = sum(1 for s in signals if s in [ScalpingSignal.SELL, ScalpingSignal.STRONG_SELL])
        
        if buy_signals > sell_signals + 1:
            return ScalpingSignal.STRONG_BUY if buy_signals >= 3 else ScalpingSignal.BUY
        elif sell_signals > buy_signals + 1:
            return ScalpingSignal.STRONG_SELL if sell_signals >= 3 else ScalpingSignal.SELL
        else:
            return ScalpingSignal.NEUTRAL
            
    def _calculate_entry_price(self, current_price: Decimal, signal: ScalpingSignal, 
                              market_data: Dict) -> Decimal:
        """Calculate optimal entry price"""
        spread = Decimal(str(market_data.get("spread", 0.001)))
        
        if signal in [ScalpingSignal.BUY, ScalpingSignal.STRONG_BUY]:
            # For buys, add a small premium to ensure fill
            entry_price = current_price * (Decimal("1") + spread / Decimal("2"))
        else:
            # For sells, subtract to ensure fill
            entry_price = current_price * (Decimal("1") - spread / Decimal("2"))
            
        return round_to_significant_digits(entry_price)
        
    def _calculate_target_price(self, entry_price: Decimal, signal: ScalpingSignal, 
                               indicators: Dict) -> Decimal:
        """Calculate target price based on signal strength"""
        # Base target based on signal strength
        if signal == ScalpingSignal.STRONG_BUY:
            target_percent = self.max_profit_target
        elif signal == ScalpingSignal.BUY:
            target_percent = (self.min_profit_target + self.max_profit_target) / Decimal("2")
        elif signal == ScalpingSignal.SELL:
            target_percent = (self.min_profit_target + self.max_profit_target) / Decimal("2")
        else:  # STRONG_SELL
            target_percent = self.max_profit_target
            
        # Adjust based on momentum
        momentum = indicators.get("momentum", 0)
        if abs(momentum) > 5:
            target_percent *= Decimal("1.2")
        elif abs(momentum) < 2:
            target_percent *= Decimal("0.8")
            
        # Calculate target
        if signal in [ScalpingSignal.BUY, ScalpingSignal.STRONG_BUY]:
            target_price = entry_price * (Decimal("1") + target_percent)
        else:
            target_price = entry_price * (Decimal("1") - target_percent)
            
        return round_to_significant_digits(target_price)
        
    def _calculate_stop_loss(self, entry_price: Decimal, signal: ScalpingSignal) -> Decimal:
        """Calculate stop loss price"""
        if signal in [ScalpingSignal.BUY, ScalpingSignal.STRONG_BUY]:
            stop_loss = entry_price * (Decimal("1") - self.stop_loss_percent)
        else:
            stop_loss = entry_price * (Decimal("1") + self.stop_loss_percent)
            
        return round_to_significant_digits(stop_loss)
        
    def _calculate_confidence(self, indicators: Dict, volume_profile: Dict, 
                            market_data: Dict) -> float:
        """Calculate confidence score for the trade"""
        confidence = 0.5  # Base confidence
        
        # RSI confidence
        rsi = indicators.get("rsi", 50)
        if rsi < 30 or rsi > 70:
            confidence += 0.1
            
        # Volume confirmation
        if volume_profile.get("volume_spike"):
            confidence += 0.15
        if volume_profile.get("volume_trend") == "increasing":
            confidence += 0.1
            
        # Trend alignment
        if indicators.get("ema_fast") and indicators.get("ema_slow"):
            if abs(indicators["ema_fast"][-1] - indicators["ema_slow"][-1]) > 0.01:
                confidence += 0.1
                
        # Momentum confirmation
        momentum = indicators.get("momentum", 0)
        if abs(momentum) > 3:
            confidence += 0.05
            
        # Liquidity bonus
        liquidity = market_data.get("liquidity", 0)
        if liquidity > self.min_liquidity_usd * 2:
            confidence += 0.1
            
        return min(confidence, 1.0)
        
    def _calculate_expected_profit(self, entry_price: Decimal, target_price: Decimal,
                                  stop_loss: Decimal, confidence: float) -> Decimal:
        """Calculate expected profit considering risk"""
        potential_profit = abs(target_price - entry_price)
        potential_loss = abs(entry_price - stop_loss)
        
        # Risk-reward ratio
        if potential_loss > 0:
            risk_reward = potential_profit / potential_loss
        else:
            risk_reward = Decimal("1")
            
        # Expected value
        win_probability = Decimal(str(confidence))
        expected_profit = (potential_profit * win_probability) - (potential_loss * (Decimal("1") - win_probability))
        
        return expected_profit
        
    def _assess_risk(self, market_data: Dict, indicators: Dict, volume_profile: Dict) -> float:
        """Assess risk score for the trade"""
        risk = 0.0
        
        # Spread risk
        spread = market_data.get("spread", 0)
        if spread > 0.01:
            risk += 20
            
        # Liquidity risk
        liquidity = market_data.get("liquidity", 0)
        if liquidity < self.min_liquidity_usd * 1.5:
            risk += 15
            
        # Volume risk
        if volume_profile.get("volume_trend") == "decreasing":
            risk += 10
            
        # Volatility risk
        volatility = market_data.get("volatility", 0)
        if volatility > 0.05:
            risk += 15
        elif volatility < 0.01:
            risk += 10  # Too stable for scalping
            
        # Technical divergence
        if indicators.get("rsi", 50) > 70 and indicators.get("momentum", 0) < 0:
            risk += 20  # Bearish divergence
            
        return min(risk, 100)
        
    def _determine_time_window(self, indicators: Dict, volume_profile: Dict) -> int:
        """Determine optimal time window for the trade"""
        base_time = 180  # 3 minutes base
        
        # Adjust based on momentum
        momentum = abs(indicators.get("momentum", 0))
        if momentum > 5:
            base_time = 120  # Faster movement, shorter window
        elif momentum < 2:
            base_time = 240  # Slower movement, longer window
            
        # Adjust based on volume
        if volume_profile.get("volume_spike"):
            base_time = int(base_time * 0.8)  # Reduce time on spikes
            
        return min(base_time, self.max_hold_time)
        
    async def execute(self, opportunity: ScalpingOpportunity, order_manager) -> Dict:
        """Execute scalping trade"""
        try:
            # Create order
            order = {
                "type": OrderType.LIMIT,
                "token": opportunity.token_address,
                "chain": opportunity.chain,
                "side": "buy" if opportunity.signal in [ScalpingSignal.BUY, ScalpingSignal.STRONG_BUY] else "sell",
                "price": str(opportunity.entry_price),
                "amount": self._calculate_position_size(opportunity),
                "stop_loss": str(opportunity.stop_loss),
                "take_profit": str(opportunity.target_price),
                "time_in_force": "IOC",  # Immediate or cancel for scalping
                "metadata": {
                    "strategy": "scalping",
                    "confidence": opportunity.confidence,
                    "expected_profit": str(opportunity.expected_profit),
                    "time_window": opportunity.time_window
                }
            }
            
            # Execute through order manager
            result = await order_manager.execute_order(order)
            
            if result.get("status") == "filled":
                # Track active trade
                self.active_trades[result["order_id"]] = {
                    "opportunity": opportunity,
                    "entry_time": datetime.now(),
                    "entry_price": result["fill_price"],
                    "amount": result["fill_amount"]
                }
                
                # Start monitoring
                asyncio.create_task(self._monitor_position(result["order_id"], opportunity))
                
            return result
            
        except Exception as e:
            logger.error(f"Scalping execution failed: {e}")
            return {"status": "failed", "error": str(e)}
            
    def _calculate_position_size(self, opportunity: ScalpingOpportunity) -> str:
        """Calculate appropriate position size"""
        # This would integrate with portfolio management
        # For now, using fixed size
        return "100"
        
    async def _monitor_position(self, order_id: str, opportunity: ScalpingOpportunity):
        """Monitor active scalping position"""
        start_time = datetime.now()
        
        while order_id in self.active_trades:
            try:
                # Check time limit
                elapsed = (datetime.now() - start_time).seconds
                if elapsed > opportunity.time_window:
                    await self._close_position(order_id, "time_limit")
                    break
                    
                # Check other exit conditions
                # This would check current price against targets
                
                await asyncio.sleep(1)  # Check every second for scalping
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                break
                
    async def _close_position(self, order_id: str, reason: str):
        """Close scalping position"""
        if order_id in self.active_trades:
            trade = self.active_trades.pop(order_id)
            
            # Record completed trade
            self.completed_trades.append({
                "order_id": order_id,
                "close_reason": reason,
                "duration": (datetime.now() - trade["entry_time"]).seconds,
                "timestamp": datetime.now()
            })
            
            # Update statistics
            self._update_statistics()
            
    def _update_statistics(self):
        """Update strategy performance statistics"""
        if not self.completed_trades:
            return
            
        # Calculate win rate
        profitable = sum(1 for t in self.completed_trades if t.get("profit", 0) > 0)
        self.win_rate = profitable / len(self.completed_trades)
        
        # Calculate average profit
        profits = [t.get("profit", 0) for t in self.completed_trades]
        self.avg_profit = Decimal(str(sum(profits) / len(profits))) if profits else Decimal("0")
        
    def get_statistics(self) -> Dict:
        """Get strategy statistics"""
        return {
            "active_trades": len(self.active_trades),
            "completed_trades": len(self.completed_trades),
            "win_rate": self.win_rate,
            "avg_profit": float(self.avg_profit),
            "total_volume": float(self.total_volume)
        }