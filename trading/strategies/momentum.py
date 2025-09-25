"""
Momentum Trading Strategy for DexScreener Bot
Identifies and trades tokens with strong momentum indicators
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

class MomentumType(Enum):
    """Types of momentum patterns"""
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    VOLUME_SURGE = "volume_surge"
    SMART_MONEY = "smart_money"
    SOCIAL_MOMENTUM = "social_momentum"
    CROSS_DEX = "cross_dex"

class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = 60
    M5 = 300
    M15 = 900
    H1 = 3600
    H4 = 14400
    D1 = 86400

@dataclass
class MomentumSignal:
    """Momentum signal data"""
    timestamp: datetime
    token_address: str
    signal_type: MomentumType
    strength: float  # 0-100
    timeframe: TimeFrame
    entry_price: Decimal
    target_prices: List[Decimal]
    stop_loss: Decimal
    volume_confirmation: bool
    smart_money_flow: Optional[Dict] = None
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    risk_score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MomentumMetrics:
    """Momentum performance metrics"""
    rsi: float
    macd_signal: float
    volume_ratio: float
    price_velocity: float
    trend_strength: float
    breakout_probability: float
    smart_money_score: float

class MomentumStrategy:
    """
    Advanced momentum trading strategy with multiple confirmation layers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize momentum strategy"""
        self.config = config or self._default_config()
        self.active_signals: Dict[str, MomentumSignal] = {}
        self.signal_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, float] = {}
        self._initialize_parameters()
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            # Momentum thresholds
            "min_rsi": 55,
            "max_rsi": 85,
            "min_volume_ratio": 2.0,
            "min_price_velocity": 0.02,  # 2% minimum
            "min_trend_strength": 0.6,
            
            # Breakout parameters
            "breakout_lookback": 20,
            "breakout_volume_multiplier": 3.0,
            "breakout_confirmation_candles": 2,
            
            # Risk management
            "max_risk_per_trade": 0.02,  # 2%
            "default_stop_loss": 0.05,    # 5%
            "trailing_stop_activation": 0.03,  # 3% profit
            "trailing_stop_distance": 0.02,    # 2%
            
            # Position sizing
            "base_position_size": 0.01,  # 1% of portfolio
            "max_position_size": 0.05,   # 5% of portfolio
            "momentum_size_multiplier": 2.0,
            
            # Smart money tracking
            "smart_money_min_score": 70,
            "smart_money_weight": 0.3,
            
            # Timeframe weights
            "timeframe_weights": {
                TimeFrame.M1: 0.1,
                TimeFrame.M5: 0.2,
                TimeFrame.M15: 0.25,
                TimeFrame.H1: 0.25,
                TimeFrame.H4: 0.15,
                TimeFrame.D1: 0.05
            }
        }
    
    def _initialize_parameters(self):
        """Initialize strategy parameters"""
        self.momentum_scores: Dict[str, float] = {}
        self.volume_trackers: Dict[str, deque] = {}
        self.price_trackers: Dict[str, deque] = {}
        
    async def analyze(self, market_data: Dict) -> Optional[MomentumSignal]:
        """
        Analyze market data for momentum opportunities
        
        Args:
            market_data: Current market data including price, volume, indicators
            
        Returns:
            MomentumSignal if opportunity found, None otherwise
        """
        try:
            token_address = market_data.get("token_address")
            if not token_address:
                return None
            
            # Calculate momentum metrics
            metrics = await self._calculate_momentum_metrics(market_data)
            
            # Check for different momentum patterns
            signals = await asyncio.gather(
                self._check_breakout_momentum(market_data, metrics),
                self._check_trend_momentum(market_data, metrics),
                self._check_volume_momentum(market_data, metrics),
                self._check_smart_money_momentum(market_data, metrics),
                return_exceptions=True
            )
            
            # Filter valid signals
            valid_signals = [s for s in signals if isinstance(s, MomentumSignal)]
            
            if not valid_signals:
                return None
            
            # Select best signal
            best_signal = self._select_best_signal(valid_signals)
            
            # Validate and enhance signal
            if await self._validate_signal(best_signal, market_data):
                enhanced_signal = await self._enhance_signal(best_signal, market_data)
                self.active_signals[token_address] = enhanced_signal
                self.signal_history.append(enhanced_signal)
                return enhanced_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return None
    
    async def _calculate_momentum_metrics(self, market_data: Dict) -> MomentumMetrics:
        """Calculate comprehensive momentum metrics"""
        try:
            # Extract indicators
            indicators = market_data.get("technical_indicators", {})
            
            # Calculate RSI momentum
            rsi = indicators.get("rsi", 50)
            rsi_momentum = self._calculate_rsi_momentum(rsi)
            
            # Calculate MACD momentum
            macd = indicators.get("macd", {})
            macd_signal = self._calculate_macd_momentum(macd)
            
            # Calculate volume momentum
            volume_data = market_data.get("volume", {})
            volume_ratio = await self._calculate_volume_ratio(volume_data)
            
            # Calculate price velocity
            price_history = market_data.get("price_history", [])
            price_velocity = self._calculate_price_velocity(price_history)
            
            # Calculate trend strength
            trend_strength = await self._calculate_trend_strength(market_data)
            
            # Calculate breakout probability
            breakout_prob = await self._calculate_breakout_probability(market_data)
            
            # Calculate smart money score
            smart_money = market_data.get("smart_money", {})
            smart_money_score = self._calculate_smart_money_score(smart_money)
            
            return MomentumMetrics(
                rsi=rsi_momentum,
                macd_signal=macd_signal,
                volume_ratio=volume_ratio,
                price_velocity=price_velocity,
                trend_strength=trend_strength,
                breakout_probability=breakout_prob,
                smart_money_score=smart_money_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {e}")
            return MomentumMetrics(0, 0, 0, 0, 0, 0, 0)
    
    async def _check_breakout_momentum(
        self, 
        market_data: Dict, 
        metrics: MomentumMetrics
    ) -> Optional[MomentumSignal]:
        """Check for breakout momentum pattern"""
        try:
            # Check breakout conditions
            if metrics.breakout_probability < 0.7:
                return None
            
            if metrics.volume_ratio < self.config["breakout_volume_multiplier"]:
                return None
            
            # Calculate resistance levels
            resistance = await self._find_resistance_levels(market_data)
            if not resistance:
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            # Check if breaking resistance
            breaking_resistance = any(
                current_price > Decimal(str(r)) * Decimal("0.99")
                for r in resistance
            )
            
            if not breaking_resistance:
                return None
            
            # Calculate targets based on resistance levels
            targets = self._calculate_breakout_targets(
                current_price, 
                resistance,
                metrics
            )
            
            # Calculate stop loss
            stop_loss = self._calculate_breakout_stop_loss(
                current_price,
                market_data
            )
            
            return MomentumSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=MomentumType.BREAKOUT,
                strength=metrics.breakout_probability * 100,
                timeframe=TimeFrame.M15,
                entry_price=current_price,
                target_prices=targets,
                stop_loss=stop_loss,
                volume_confirmation=True,
                technical_indicators={
                    "rsi": metrics.rsi,
                    "volume_ratio": metrics.volume_ratio,
                    "breakout_probability": metrics.breakout_probability
                },
                confidence=metrics.breakout_probability,
                metadata={"resistance_levels": resistance}
            )
            
        except Exception as e:
            logger.error(f"Error checking breakout momentum: {e}")
            return None
    
    async def _check_trend_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[MomentumSignal]:
        """Check for trend-following momentum"""
        try:
            # Check trend conditions
            if metrics.trend_strength < self.config["min_trend_strength"]:
                return None
            
            # Check RSI for trend continuation
            rsi = market_data.get("technical_indicators", {}).get("rsi", 50)
            if rsi < self.config["min_rsi"] or rsi > self.config["max_rsi"]:
                return None
            
            # Check moving averages alignment
            ma_aligned = await self._check_ma_alignment(market_data)
            if not ma_aligned:
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            # Calculate trend targets
            targets = self._calculate_trend_targets(
                current_price,
                metrics.trend_strength,
                market_data
            )
            
            # Calculate trend stop loss
            stop_loss = self._calculate_trend_stop_loss(
                current_price,
                market_data
            )
            
            return MomentumSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=MomentumType.TREND_FOLLOWING,
                strength=metrics.trend_strength * 100,
                timeframe=TimeFrame.H1,
                entry_price=current_price,
                target_prices=targets,
                stop_loss=stop_loss,
                volume_confirmation=metrics.volume_ratio > 1.5,
                technical_indicators={
                    "rsi": rsi,
                    "trend_strength": metrics.trend_strength,
                    "macd_signal": metrics.macd_signal
                },
                confidence=metrics.trend_strength,
                metadata={"ma_alignment": ma_aligned}
            )
            
        except Exception as e:
            logger.error(f"Error checking trend momentum: {e}")
            return None
    
    async def _check_volume_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[MomentumSignal]:
        """Check for volume-based momentum"""
        try:
            # Check volume surge conditions
            if metrics.volume_ratio < self.config["min_volume_ratio"]:
                return None
            
            # Check price action confirmation
            if metrics.price_velocity < self.config["min_price_velocity"]:
                return None
            
            # Check for unusual volume patterns
            volume_pattern = await self._analyze_volume_pattern(market_data)
            if volume_pattern["type"] != "surge":
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            # Calculate volume-based targets
            targets = self._calculate_volume_targets(
                current_price,
                metrics.volume_ratio,
                volume_pattern
            )
            
            # Calculate stop loss
            stop_loss = current_price * Decimal("0.95")  # 5% stop
            
            return MomentumSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=MomentumType.VOLUME_SURGE,
                strength=min(metrics.volume_ratio * 20, 100),
                timeframe=TimeFrame.M5,
                entry_price=current_price,
                target_prices=targets,
                stop_loss=stop_loss,
                volume_confirmation=True,
                technical_indicators={
                    "volume_ratio": metrics.volume_ratio,
                    "price_velocity": metrics.price_velocity
                },
                confidence=min(metrics.volume_ratio / 5, 1.0),
                metadata={"volume_pattern": volume_pattern}
            )
            
        except Exception as e:
            logger.error(f"Error checking volume momentum: {e}")
            return None
    
    async def _check_smart_money_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[MomentumSignal]:
        """Check for smart money driven momentum"""
        try:
            # Check smart money score
            if metrics.smart_money_score < self.config["smart_money_min_score"]:
                return None
            
            # Get smart money flow data
            smart_money = market_data.get("smart_money", {})
            
            # Check for whale accumulation
            whale_activity = smart_money.get("whale_activity", {})
            if whale_activity.get("net_flow", 0) <= 0:
                return None
            
            # Check smart money velocity
            smart_velocity = await self._calculate_smart_money_velocity(smart_money)
            if smart_velocity < 0.5:
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            
            # Calculate smart money targets
            targets = self._calculate_smart_money_targets(
                current_price,
                smart_money,
                metrics.smart_money_score
            )
            
            # Calculate stop loss based on whale levels
            stop_loss = self._calculate_smart_money_stop_loss(
                current_price,
                whale_activity
            )
            
            return MomentumSignal(
                timestamp=datetime.utcnow(),
                token_address=market_data["token_address"],
                signal_type=MomentumType.SMART_MONEY,
                strength=metrics.smart_money_score,
                timeframe=TimeFrame.H1,
                entry_price=current_price,
                target_prices=targets,
                stop_loss=stop_loss,
                volume_confirmation=True,
                smart_money_flow=smart_money,
                technical_indicators={
                    "smart_money_score": metrics.smart_money_score,
                    "smart_velocity": smart_velocity
                },
                confidence=metrics.smart_money_score / 100,
                metadata={"whale_activity": whale_activity}
            )
            
        except Exception as e:
            logger.error(f"Error checking smart money momentum: {e}")
            return None
    
    def _calculate_rsi_momentum(self, rsi: float) -> float:
        """Calculate momentum based on RSI"""
        if rsi < 30:
            return 0  # Oversold, no momentum
        elif rsi > 70:
            return max(0, 100 - rsi) / 30  # Overbought, decreasing momentum
        else:
            # Optimal momentum zone
            return (rsi - 30) / 40
    
    def _calculate_macd_momentum(self, macd: Dict) -> float:
        """Calculate momentum from MACD"""
        try:
            signal = macd.get("signal", 0)
            histogram = macd.get("histogram", 0)
            
            if histogram > 0 and signal > 0:
                return min(histogram / abs(signal) if signal != 0 else 0, 1.0)
            return 0
        except:
            return 0
    
    async def _calculate_volume_ratio(self, volume_data: Dict) -> float:
        """Calculate volume ratio vs average"""
        try:
            current_volume = volume_data.get("current", 0)
            avg_volume = volume_data.get("average", 1)
            
            if avg_volume == 0:
                return 0
            
            return current_volume / avg_volume
        except:
            return 0
    
    def _calculate_price_velocity(self, price_history: List) -> float:
        """Calculate rate of price change"""
        try:
            if len(price_history) < 2:
                return 0
            
            recent_prices = price_history[-10:]
            if len(recent_prices) < 2:
                return 0
            
            price_changes = [
                (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                for i in range(1, len(recent_prices))
                if recent_prices[i-1] != 0
            ]
            
            if not price_changes:
                return 0
            
            return np.mean(price_changes)
        except:
            return 0
    
    async def _calculate_trend_strength(self, market_data: Dict) -> float:
        """Calculate overall trend strength"""
        try:
            indicators = market_data.get("technical_indicators", {})
            
            # ADX for trend strength
            adx = indicators.get("adx", 0)
            
            # Moving average convergence
            ma_data = indicators.get("moving_averages", {})
            ma_trend = self._calculate_ma_trend(ma_data)
            
            # Combine metrics
            trend_strength = (adx / 100 * 0.6) + (ma_trend * 0.4)
            
            return min(trend_strength, 1.0)
        except:
            return 0
    
    async def _calculate_breakout_probability(self, market_data: Dict) -> float:
        """Calculate probability of successful breakout"""
        try:
            # Get price action data
            candles = market_data.get("candles", [])
            if len(candles) < 20:
                return 0
            
            # Check for consolidation pattern
            consolidation = self._detect_consolidation(candles)
            
            # Check volume buildup
            volume_buildup = self._detect_volume_buildup(candles)
            
            # Check for tightening range
            range_tightening = self._detect_range_tightening(candles)
            
            # Calculate probability
            prob = (consolidation * 0.3 + volume_buildup * 0.4 + range_tightening * 0.3)
            
            return min(prob, 1.0)
        except:
            return 0
    
    def _calculate_smart_money_score(self, smart_money: Dict) -> float:
        """Calculate smart money influence score"""
        try:
            whale_score = smart_money.get("whale_score", 0)
            smart_wallets = smart_money.get("smart_wallet_count", 0)
            accumulation = smart_money.get("accumulation_score", 0)
            
            # Weight components
            score = (
                whale_score * 0.4 +
                min(smart_wallets / 10, 1.0) * 0.3 +
                accumulation * 0.3
            ) * 100
            
            return min(score, 100)
        except:
            return 0
    
    async def _find_resistance_levels(self, market_data: Dict) -> List[float]:
        """Find key resistance levels"""
        try:
            candles = market_data.get("candles", [])
            if len(candles) < 20:
                return []
            
            highs = [c["high"] for c in candles]
            
            # Find local maxima
            resistance_levels = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_levels.append(highs[i])
            
            # Cluster nearby levels
            clustered = self._cluster_price_levels(resistance_levels)
            
            return sorted(clustered)[-3:]  # Top 3 resistance levels
        except:
            return []
    
    def _cluster_price_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _select_best_signal(self, signals: List[MomentumSignal]) -> MomentumSignal:
        """Select the best signal from multiple options"""
        # Score each signal
        scored_signals = []
        for signal in signals:
            score = (
                signal.strength * 0.3 +
                signal.confidence * 100 * 0.4 +
                (100 if signal.volume_confirmation else 0) * 0.2 +
                (100 if signal.smart_money_flow else 50) * 0.1
            )
            scored_signals.append((score, signal))
        
        # Return highest scoring signal
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return scored_signals[0][1]
    
    async def _validate_signal(self, signal: MomentumSignal, market_data: Dict) -> bool:
        """Validate momentum signal"""
        try:
            # Check liquidity
            liquidity = market_data.get("liquidity", 0)
            if liquidity < 50000:  # $50k minimum
                return False
            
            # Check spread
            spread = market_data.get("spread", 1.0)
            if spread > 0.05:  # 5% max spread
                return False
            
            # Check contract safety
            contract_verified = market_data.get("contract_verified", False)
            if not contract_verified:
                return False
            
            # Check for red flags
            red_flags = market_data.get("red_flags", [])
            if red_flags:
                logger.warning(f"Red flags detected: {red_flags}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    async def _enhance_signal(
        self, 
        signal: MomentumSignal, 
        market_data: Dict
    ) -> MomentumSignal:
        """Enhance signal with additional data"""
        try:
            # Add risk scoring
            risk_score = market_data.get("risk_score", {})
            signal.risk_score = risk_score.get("total", 50)
            
            # Add market conditions
            signal.metadata["market_conditions"] = {
                "volatility": market_data.get("volatility", 0),
                "correlation": market_data.get("correlation", {}),
                "market_cap": market_data.get("market_cap", 0),
                "holder_count": market_data.get("holders", 0)
            }
            
            # Adjust position sizing based on confidence
            signal.metadata["suggested_position_size"] = self._calculate_position_size(
                signal.confidence,
                signal.risk_score
            )
            
            # Add trailing stop parameters
            signal.metadata["trailing_stop"] = {
                "activation": float(signal.entry_price) * (1 + self.config["trailing_stop_activation"]),
                "distance": self.config["trailing_stop_distance"]
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error enhancing signal: {e}")
            return signal
    
    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate position size based on confidence and risk"""
        base_size = self.config["base_position_size"]
        
        # Adjust for confidence
        confidence_multiplier = 1 + (confidence - 0.5) * 2  # 0-2x
        
        # Adjust for risk (inverse relationship)
        risk_multiplier = 1.5 - (risk_score / 100) * 0.5  # 0.5-1.5x
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * risk_multiplier
        
        # Cap at maximum
        return min(position_size, self.config["max_position_size"])
    
    def _calculate_breakout_targets(
        self,
        entry_price: Decimal,
        resistance_levels: List[float],
        metrics: MomentumMetrics
    ) -> List[Decimal]:
        """Calculate breakout target prices"""
        targets = []
        
        # First target: Next resistance level
        next_resistance = None
        for level in resistance_levels:
            if level > float(entry_price):
                next_resistance = level
                break
        
        if next_resistance:
            targets.append(Decimal(str(next_resistance)))
        else:
            # Default 5% target
            targets.append(entry_price * Decimal("1.05"))
        
        # Second target: Fibonacci extension
        targets.append(entry_price * Decimal("1.08"))
        
        # Third target: Based on momentum strength
        momentum_target = entry_price * (
            Decimal("1") + Decimal(str(metrics.breakout_probability * 0.15))
        )
        targets.append(momentum_target)
        
        return sorted(targets)
    
    def _calculate_breakout_stop_loss(
        self,
        entry_price: Decimal,
        market_data: Dict
    ) -> Decimal:
        """Calculate stop loss for breakout trade"""
        # Find recent support
        candles = market_data.get("candles", [])
        if candles:
            recent_low = min(c["low"] for c in candles[-5:])
            return Decimal(str(recent_low)) * Decimal("0.98")  # 2% below support
        
        # Default stop loss
        return entry_price * Decimal("0.95")
    
    async def update_signal(
        self, 
        token_address: str, 
        market_data: Dict
    ) -> Optional[Dict]:
        """
        Update existing momentum signal with new data
        
        Returns:
            Update instructions (adjust stop, take profit, close, etc.)
        """
        try:
            if token_address not in self.active_signals:
                return None
            
            signal = self.active_signals[token_address]
            current_price = Decimal(str(market_data.get("price", 0)))
            
            # Calculate profit/loss
            pnl_percent = float((current_price - signal.entry_price) / signal.entry_price)
            
            # Check for stop loss
            if current_price <= signal.stop_loss:
                return {
                    "action": "close",
                    "reason": "stop_loss_hit",
                    "price": current_price
                }
            
            # Check for take profit levels
            for i, target in enumerate(signal.target_prices):
                if current_price >= target:
                    if i == len(signal.target_prices) - 1:
                        # Final target reached
                        return {
                            "action": "close",
                            "reason": "final_target_reached",
                            "price": current_price
                        }
                    else:
                        # Partial profit and adjust stop
                        return {
                            "action": "partial_close",
                            "percent": 0.33,  # Take 1/3 profit
                            "new_stop": signal.entry_price * Decimal("1.01"),  # Move stop to breakeven
                            "reason": f"target_{i+1}_reached"
                        }
            
            # Check for trailing stop activation
            trailing_config = signal.metadata.get("trailing_stop", {})
            if trailing_config and pnl_percent >= trailing_config["activation"]:
                new_stop = current_price * (Decimal("1") - Decimal(str(trailing_config["distance"])))
                if new_stop > signal.stop_loss:
                    return {
                        "action": "adjust_stop",
                        "new_stop": new_stop,
                        "reason": "trailing_stop_update"
                    }
            
            # Check for momentum loss
            new_metrics = await self._calculate_momentum_metrics(market_data)
            if signal.signal_type == MomentumType.TREND_FOLLOWING:
                if new_metrics.trend_strength < 0.3:
                    return {
                        "action": "close",
                        "reason": "momentum_lost",
                        "price": current_price
                    }
            
            # Check for reversal signals
            if await self._check_reversal_signals(market_data):
                return {
                    "action": "close",
                    "reason": "reversal_detected",
                    "price": current_price
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating signal: {e}")
            return None
    
    async def _check_reversal_signals(self, market_data: Dict) -> bool:
        """Check for trend reversal signals"""
        try:
            indicators = market_data.get("technical_indicators", {})
            
            # Check RSI divergence
            rsi = indicators.get("rsi", 50)
            if rsi > 80 or rsi < 20:
                return True
            
            # Check MACD crossover
            macd = indicators.get("macd", {})
            if macd.get("histogram", 0) < 0 and macd.get("previous_histogram", 0) > 0:
                return True
            
            # Check volume decline
            volume = market_data.get("volume", {})
            if volume.get("current", 0) < volume.get("average", 1) * 0.5:
                return True
            
            return False
            
        except:
            return False
    
    def _calculate_ma_trend(self, ma_data: Dict) -> float:
        """Calculate trend from moving averages"""
        try:
            ma_20 = ma_data.get("ma_20", 0)
            ma_50 = ma_data.get("ma_50", 0)
            ma_200 = ma_data.get("ma_200", 0)
            
            if not all([ma_20, ma_50, ma_200]):
                return 0
            
            # Check alignment
            if ma_20 > ma_50 > ma_200:
                return 1.0  # Strong uptrend
            elif ma_20 < ma_50 < ma_200:
                return -1.0  # Strong downtrend
            else:
                return 0.5  # Mixed trend
        except:
            return 0
    
    def _detect_consolidation(self, candles: List[Dict]) -> float:
        """Detect price consolidation pattern"""
        try:
            if len(candles) < 10:
                return 0
            
            # Calculate price range
            highs = [c["high"] for c in candles[-10:]]
            lows = [c["low"] for c in candles[-10:]]
            
            price_range = (max(highs) - min(lows)) / np.mean(lows)
            
            # Tighter range = higher consolidation score
            if price_range < 0.05:  # Less than 5% range
                return 1.0
            elif price_range < 0.10:
                return 0.7
            elif price_range < 0.15:
                return 0.4
            else:
                return 0
        except:
            return 0
    
    def _detect_volume_buildup(self, candles: List[Dict]) -> float:
        """Detect volume buildup pattern"""
        try:
            if len(candles) < 10:
                return 0
            
            recent_volumes = [c["volume"] for c in candles[-5:]]
            older_volumes = [c["volume"] for c in candles[-10:-5]]
            
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            
            if older_avg == 0:
                return 0
            
            buildup_ratio = recent_avg / older_avg
            
            return min(buildup_ratio / 2, 1.0)  # Normalize to 0-1
        except:
            return 0
    
    def _detect_range_tightening(self, candles: List[Dict]) -> float:
        """Detect tightening price range"""
        try:
            if len(candles) < 15:
                return 0
            
            # Compare recent range vs older range
            recent_range = self._calculate_range(candles[-5:])
            older_range = self._calculate_range(candles[-15:-10])
            
            if older_range == 0:
                return 0
            
            tightening = 1 - (recent_range / older_range)
            
            return max(0, min(tightening, 1.0))
        except:
            return 0
    
    def _calculate_range(self, candles: List[Dict]) -> float:
        """Calculate price range for candles"""
        if not candles:
            return 0
        
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        
        return max(highs) - min(lows)
    
    async def _check_ma_alignment(self, market_data: Dict) -> bool:
        """Check if moving averages are properly aligned"""
        try:
            ma_data = market_data.get("technical_indicators", {}).get("moving_averages", {})
            
            ma_20 = ma_data.get("ma_20", 0)
            ma_50 = ma_data.get("ma_50", 0)
            ma_200 = ma_data.get("ma_200", 0)
            current_price = market_data.get("price", 0)
            
            # Bullish alignment: Price > MA20 > MA50 > MA200
            return current_price > ma_20 > ma_50 > ma_200
        except:
            return False
    
    def _calculate_trend_targets(
        self,
        entry_price: Decimal,
        trend_strength: float,
        market_data: Dict
    ) -> List[Decimal]:
        """Calculate trend-following targets"""
        targets = []
        
        # Base target on trend strength
        base_multiplier = Decimal(str(1.03 + trend_strength * 0.07))  # 3-10%
        
        # Three progressive targets
        targets.append(entry_price * base_multiplier)
        targets.append(entry_price * (base_multiplier + Decimal("0.03")))
        targets.append(entry_price * (base_multiplier + Decimal("0.06")))
        
        return targets
    
    def _calculate_trend_stop_loss(
        self,
        entry_price: Decimal,
        market_data: Dict
    ) -> Decimal:
        """Calculate trend-following stop loss"""
        # Use recent swing low
        candles = market_data.get("candles", [])
        if len(candles) > 10:
            swing_low = min(c["low"] for c in candles[-10:])
            return Decimal(str(swing_low)) * Decimal("0.98")
        
        # Default to 3% stop
        return entry_price * Decimal("0.97")
    
    async def _analyze_volume_pattern(self, market_data: Dict) -> Dict:
        """Analyze volume patterns"""
        try:
            candles = market_data.get("candles", [])
            if len(candles) < 20:
                return {"type": "unknown"}
            
            volumes = [c["volume"] for c in candles]
            avg_volume = np.mean(volumes)
            recent_volume = volumes[-1]
            
            # Detect surge
            if recent_volume > avg_volume * 3:
                return {
                    "type": "surge",
                    "magnitude": recent_volume / avg_volume,
                    "sustained": sum(1 for v in volumes[-3:] if v > avg_volume * 2) >= 2
                }
            
            # Detect accumulation
            elif all(v > avg_volume * 1.5 for v in volumes[-5:]):
                return {
                    "type": "accumulation",
                    "duration": 5,
                    "strength": np.mean(volumes[-5:]) / avg_volume
                }
            
            # Detect distribution
            elif recent_volume < avg_volume * 0.5:
                return {
                    "type": "distribution",
                    "weakness": avg_volume / recent_volume
                }
            
            return {"type": "normal"}
            
        except:
            return {"type": "unknown"}
    
    def _calculate_volume_targets(
        self,
        entry_price: Decimal,
        volume_ratio: float,
        volume_pattern: Dict
    ) -> List[Decimal]:
        """Calculate volume-based targets"""
        targets = []
        
        # Aggressive targets for volume surges
        if volume_pattern.get("type") == "surge":
            magnitude = volume_pattern.get("magnitude", 3)
            base_target = min(magnitude * 0.02, 0.15)  # 2% per volume multiple, max 15%
            
            targets.append(entry_price * (Decimal("1") + Decimal(str(base_target * 0.5))))
            targets.append(entry_price * (Decimal("1") + Decimal(str(base_target * 0.75))))
            targets.append(entry_price * (Decimal("1") + Decimal(str(base_target))))
        else:
            # Conservative targets
            targets.append(entry_price * Decimal("1.03"))
            targets.append(entry_price * Decimal("1.05"))
            targets.append(entry_price * Decimal("1.08"))
        
        return targets
    
    async def _calculate_smart_money_velocity(self, smart_money: Dict) -> float:
        """Calculate smart money flow velocity"""
        try:
            inflows = smart_money.get("recent_inflows", [])
            if not inflows:
                return 0
            
            # Calculate acceleration of inflows
            if len(inflows) < 2:
                return 0
            
            recent_flow = sum(inflows[-3:]) if len(inflows) >= 3 else sum(inflows)
            older_flow = sum(inflows[-6:-3]) if len(inflows) >= 6 else sum(inflows[:-1])
            
            if older_flow == 0:
                return 1.0 if recent_flow > 0 else 0
            
            velocity = recent_flow / older_flow
            return min(velocity, 2.0) / 2  # Normalize to 0-1
            
        except:
            return 0
    
    def _calculate_smart_money_targets(
        self,
        entry_price: Decimal,
        smart_money: Dict,
        smart_score: float
    ) -> List[Decimal]:
        """Calculate targets based on smart money analysis"""
        targets = []
        
        # Base targets on smart money confidence
        base_multiplier = 1.05 + (smart_score / 100) * 0.10  # 5-15%
        
        # Consider whale targets if available
        whale_targets = smart_money.get("whale_targets", [])
        if whale_targets:
            targets.extend([Decimal(str(t)) for t in whale_targets[:2]])
        
        # Add calculated targets
        targets.append(entry_price * Decimal(str(base_multiplier)))
        targets.append(entry_price * Decimal(str(base_multiplier * 1.5)))
        
        return sorted(set(targets))[:3]  # Top 3 unique targets
    
    def _calculate_smart_money_stop_loss(
        self,
        entry_price: Decimal,
        whale_activity: Dict
    ) -> Decimal:
        """Calculate stop loss based on whale levels"""
        # Check whale support levels
        support_levels = whale_activity.get("support_levels", [])
        if support_levels:
            # Use highest support below entry
            valid_supports = [s for s in support_levels if s < float(entry_price)]
            if valid_supports:
                return Decimal(str(max(valid_supports))) * Decimal("0.98")
        
        # Default stop
        return entry_price * Decimal("0.96")
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        return {
            "total_signals": len(self.signal_history),
            "active_signals": len(self.active_signals),
            "signal_distribution": self._get_signal_distribution(),
            "average_strength": self._calculate_average_strength(),
            "success_rate": self._calculate_success_rate(),
            "momentum_scores": self.momentum_scores
        }
    
    def _get_signal_distribution(self) -> Dict:
        """Get distribution of signal types"""
        distribution = {}
        for signal in self.signal_history:
            signal_type = signal.signal_type.value
            distribution[signal_type] = distribution.get(signal_type, 0) + 1
        return distribution
    
    def _calculate_average_strength(self) -> float:
        """Calculate average signal strength"""
        if not self.signal_history:
            return 0
        return np.mean([s.strength for s in self.signal_history])
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate of signals"""
        # This would need tracking of closed positions
        # Placeholder for now
        return 0.0

    # PATCHES FOR momentum.py
    # Add these methods to the MomentumStrategy class:

    def calculate_momentum(self, prices: List[float]) -> float:
        """
        Calculate momentum indicator
        Public interface for momentum calculation
        
        Args:
            prices: List of price values
            
        Returns:
            Momentum value as float
        """
        return self._calculate_momentum(prices)

    def identify_entry_points(self, momentum_data: Dict) -> List[Dict]:
        """
        Identify optimal entry points based on momentum
        
        Args:
            momentum_data: Dictionary containing momentum indicators
            
        Returns:
            List of entry point dictionaries with price and confidence
        """
        entry_points = []
        
        # Check for momentum breakout
        if momentum_data.get("breakout_probability", 0) > 0.7:
            entry_points.append({
                "type": "breakout",
                "price": momentum_data.get("current_price", 0),
                "confidence": momentum_data.get("breakout_probability", 0),
                "timeframe": "5m"
            })
        
        # Check for trend continuation
        if momentum_data.get("trend_strength", 0) > 0.6:
            entry_points.append({
                "type": "trend_continuation", 
                "price": momentum_data.get("current_price", 0) * 1.001,  # Slight premium
                "confidence": momentum_data.get("trend_strength", 0),
                "timeframe": "15m"
            })
        
        # Check for volume surge entry
        if momentum_data.get("volume_ratio", 1) > 3:
            entry_points.append({
                "type": "volume_surge",
                "price": momentum_data.get("current_price", 0),
                "confidence": min(momentum_data.get("volume_ratio", 0) / 5, 1.0),
                "timeframe": "1m"
            })
        
        # Sort by confidence
        entry_points.sort(key=lambda x: x["confidence"], reverse=True)
        
        return entry_points

    def calculate_position_size(self, signal: MomentumSignal) -> Decimal:
        """
        Calculate position size for momentum trade
        
        Args:
            signal: MomentumSignal object
            
        Returns:
            Position size as Decimal
        """
        # Use internal method if available
        if hasattr(self, '_calculate_position_size'):
            return Decimal(str(self._calculate_position_size(
                signal.confidence,
                signal.risk_score
            )))
        
        # Default calculation
        base_size = Decimal(str(self.config.get("base_position_size", 0.01)))
        
        # Adjust for signal strength
        strength_multiplier = Decimal(str(signal.strength / 100))
        
        # Adjust for confidence
        confidence_multiplier = Decimal(str(signal.confidence))
        
        # Calculate final size
        position_size = base_size * strength_multiplier * confidence_multiplier
        
        # Apply limits
        max_size = Decimal(str(self.config.get("max_position_size", 0.05)))
        return min(position_size, max_size)

    async def execute(self, signal: MomentumSignal) -> Dict:
        """
        Execute momentum trading signal
        
        Args:
            signal: MomentumSignal to execute
            
        Returns:
            Execution result dictionary
        """
        try:
            # Validate signal
            if signal.confidence < self.config.get("min_confidence", 0.6):
                return {
                    "status": "skipped",
                    "reason": "low_confidence",
                    "confidence": signal.confidence
                }
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            # Create execution order
            order = {
                "token_address": signal.token_address,
                "signal_type": signal.signal_type.value,
                "entry_price": float(signal.entry_price),
                "target_prices": [float(tp) for tp in signal.target_prices],
                "stop_loss": float(signal.stop_loss),
                "position_size": float(position_size),
                "timeframe": signal.timeframe.value,
                "metadata": signal.metadata
            }
            
            # Log execution
            logger.info(f"Executing momentum signal: {signal.signal_type.value} for {signal.token_address[:10]}...")
            
            # Update active signals
            self.active_signals[signal.token_address] = signal
            
            return {
                "status": "success",
                "signal_id": id(signal),
                "order": order,
                "timestamp": signal.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Momentum execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }