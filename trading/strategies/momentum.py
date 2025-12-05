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

from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength

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
class MomentumMetrics:
    """Momentum performance metrics"""
    rsi: float
    macd_signal: float
    volume_ratio: float
    price_velocity: float
    trend_strength: float
    breakout_probability: float
    smart_money_score: float

class MomentumStrategy(BaseStrategy):
    """
    Advanced momentum trading strategy with multiple confirmation layers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize momentum strategy"""
        super().__init__(config)
        self.momentum_scores: Dict[str, float] = {}
        self.volume_trackers: Dict[str, deque] = {}
        self.price_trackers: Dict[str, deque] = {}
        
    async def analyze(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Analyze market data for momentum opportunities
        
        Args:
            market_data: Current market data including price, volume, indicators
            
        Returns:
            TradingSignal if opportunity found, None otherwise
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
            valid_signals = [s for s in signals if isinstance(s, TradingSignal)]
            
            if not valid_signals:
                return None
            
            # Select best signal
            best_signal = self._select_best_signal(valid_signals)
            
            # Validate and enhance signal
            if await self.validate_signal(best_signal, market_data):
                # Enhancement logic should be part of the signal creation
                self.active_positions[token_address] = best_signal
                self.signal_history.append(best_signal)
                return best_signal
            
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
    ) -> Optional[TradingSignal]:
        """Check for breakout momentum pattern"""
        try:
            # Check breakout conditions
            if metrics.breakout_probability < 0.7:
                return None
            
            if metrics.volume_ratio < self.config.get("breakout_volume_multiplier", 3.0):
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

            signal = TradingSignal(
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                token_address=market_data["token_address"],
                chain=market_data.get("chain", "ethereum"),
                entry_price=current_price,
                target_price=targets[0] if targets else None,
                stop_loss=stop_loss,
                confidence=metrics.breakout_probability,
                metadata={
                    "resistance_levels": resistance,
                    "momentum_type": "breakout"
                }
            )
            return signal
            
        except Exception as e:
            logger.error(f"Error checking breakout momentum: {e}")
            return None
    
    async def _check_trend_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[TradingSignal]:
        """Check for trend-following momentum"""
        try:
            if metrics.trend_strength < self.config.get("min_trend_strength", 0.6):
                return None
            
            rsi = market_data.get("technical_indicators", {}).get("rsi", 50)
            if rsi < self.config.get("min_rsi", 55) or rsi > self.config.get("max_rsi", 85):
                return None
            
            if not await self._check_ma_alignment(market_data):
                return None
            
            current_price = Decimal(str(market_data.get("price", 0)))
            targets = self._calculate_trend_targets(current_price, metrics.trend_strength, market_data)
            stop_loss = self._calculate_trend_stop_loss(current_price, market_data)
            
            return TradingSignal(
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                token_address=market_data["token_address"],
                chain=market_data.get("chain", "ethereum"),
                entry_price=current_price,
                target_price=targets[0] if targets else None,
                stop_loss=stop_loss,
                confidence=metrics.trend_strength,
                metadata={"momentum_type": "trend_following"}
            )
        except Exception as e:
            logger.error(f"Error checking trend momentum: {e}")
            return None

    async def _check_volume_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[TradingSignal]:
        """Check for volume-based momentum"""
        try:
            if metrics.volume_ratio < self.config.get("min_volume_ratio", 2.0):
                return None
            if metrics.price_velocity < self.config.get("min_price_velocity", 0.02):
                return None
            
            volume_pattern = await self._analyze_volume_pattern(market_data)
            if volume_pattern.get("type") != "surge":
                return None

            current_price = Decimal(str(market_data.get("price", 0)))
            targets = self._calculate_volume_targets(current_price, metrics.volume_ratio, volume_pattern)
            stop_loss = current_price * Decimal("0.95")

            return TradingSignal(
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                token_address=market_data["token_address"],
                chain=market_data.get("chain", "ethereum"),
                entry_price=current_price,
                target_price=targets[0] if targets else None,
                stop_loss=stop_loss,
                confidence=min(metrics.volume_ratio / 5, 1.0),
                metadata={"momentum_type": "volume_surge"}
            )
        except Exception as e:
            logger.error(f"Error checking volume momentum: {e}")
            return None

    async def _check_smart_money_momentum(
        self,
        market_data: Dict,
        metrics: MomentumMetrics
    ) -> Optional[TradingSignal]:
        """Check for smart money driven momentum"""
        try:
            if metrics.smart_money_score < self.config.get("smart_money_min_score", 70):
                return None
            
            smart_money = market_data.get("smart_money", {})
            whale_activity = smart_money.get("whale_activity", {})
            if whale_activity.get("net_flow", 0) <= 0:
                return None

            current_price = Decimal(str(market_data.get("price", 0)))
            targets = self._calculate_smart_money_targets(current_price, smart_money, metrics.smart_money_score)
            stop_loss = self._calculate_smart_money_stop_loss(current_price, whale_activity)

            return TradingSignal(
                strategy_name=self.name,
                signal_type=SignalType.BUY,
                strength=SignalStrength.VERY_STRONG,
                token_address=market_data["token_address"],
                chain=market_data.get("chain", "ethereum"),
                entry_price=current_price,
                target_price=targets[0] if targets else None,
                stop_loss=stop_loss,
                confidence=metrics.smart_money_score / 100,
                metadata={"momentum_type": "smart_money"}
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
    
    def _select_best_signal(self, signals: List[TradingSignal]) -> TradingSignal:
        """Select the best signal from multiple options based on confidence."""
        return max(signals, key=lambda s: s.confidence)

    async def calculate_indicators(self, price_data: List[float], volume_data: List[float]) -> Dict[str, Any]:
        """
        Calculate strategy-specific technical indicators.
        For momentum, this is not strictly needed as metrics are calculated in analyze.
        """
        return {}

    def validate_signal(self, signal: TradingSignal, market_data: Dict) -> bool:
        """Validate momentum signal"""
        try:
            if market_data.get("liquidity", 0) < 50000:
                return False
            if market_data.get("spread", 1.0) > 0.05:
                return False
            if not market_data.get("contract_verified", False):
                return False
            if market_data.get("red_flags"):
                logger.warning(f"Red flags detected: {market_data.get('red_flags')}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
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

    # ========================================================================
    # MISSING HELPER METHODS (Added to fix Momentum strategy)
    # ========================================================================

    async def _check_ma_alignment(self, market_data: Dict) -> bool:
        """
        Check if moving averages are properly aligned for trend following.
        Returns True if short MA > medium MA > long MA (uptrend) or vice versa (downtrend).
        """
        try:
            indicators = market_data.get("technical_indicators", {})

            # Get MAs from market data
            ma_short = indicators.get("ema_9") or indicators.get("sma_9")
            ma_medium = indicators.get("ema_21") or indicators.get("sma_21")
            ma_long = indicators.get("ema_50") or indicators.get("sma_50")

            # If no MAs available, try to calculate from price history
            if not all([ma_short, ma_medium, ma_long]):
                candles = market_data.get("candles", [])
                if len(candles) >= 50:
                    prices = [c.get("close", 0) for c in candles]
                    ma_short = sum(prices[-9:]) / 9
                    ma_medium = sum(prices[-21:]) / 21
                    ma_long = sum(prices[-50:]) / 50
                else:
                    # Not enough data, be lenient
                    return True

            # Check for uptrend alignment: short > medium > long
            if ma_short > ma_medium > ma_long:
                return True

            # Check for downtrend alignment: short < medium < long (for shorts)
            # For simplicity, we're only supporting uptrend momentum signals

            return False
        except Exception as e:
            logger.error(f"Error checking MA alignment: {e}")
            return False

    def _calculate_trend_targets(
        self,
        current_price: Decimal,
        trend_strength: float,
        market_data: Dict
    ) -> List[Decimal]:
        """Calculate target prices for trend-following trades."""
        targets = []

        # Target based on trend strength (stronger trend = larger targets)
        base_target_pct = Decimal("0.03") + Decimal(str(trend_strength * 0.05))  # 3-8%

        # First target: Conservative
        targets.append(current_price * (Decimal("1") + base_target_pct))

        # Second target: Moderate
        targets.append(current_price * (Decimal("1") + base_target_pct * Decimal("1.5")))

        # Third target: Aggressive
        targets.append(current_price * (Decimal("1") + base_target_pct * Decimal("2")))

        return targets

    def _calculate_trend_stop_loss(
        self,
        current_price: Decimal,
        market_data: Dict
    ) -> Decimal:
        """Calculate stop loss for trend-following trades."""
        candles = market_data.get("candles", [])

        if candles and len(candles) >= 5:
            # Use recent swing low as support
            recent_lows = [c.get("low", float(current_price)) for c in candles[-10:]]
            swing_low = min(recent_lows)
            stop_loss = Decimal(str(swing_low)) * Decimal("0.98")  # 2% below swing low

            # Ensure stop loss is reasonable (not more than 8% from entry)
            max_stop_loss = current_price * Decimal("0.92")
            return max(stop_loss, max_stop_loss)

        # Default: 5% stop loss
        return current_price * Decimal("0.95")

    async def _analyze_volume_pattern(self, market_data: Dict) -> Dict:
        """
        Analyze volume patterns to identify surges, accumulation, or distribution.
        """
        try:
            candles = market_data.get("candles", [])
            if not candles or len(candles) < 10:
                return {"type": "unknown", "score": 0}

            volumes = [c.get("volume", 0) for c in candles]
            recent_volume = sum(volumes[-3:]) / 3
            average_volume = sum(volumes[:-3]) / max(len(volumes) - 3, 1)

            # Calculate volume ratio
            volume_ratio = recent_volume / average_volume if average_volume > 0 else 1.0

            # Determine pattern type
            if volume_ratio >= 3.0:
                return {"type": "surge", "score": min(volume_ratio / 5, 1.0), "ratio": volume_ratio}
            elif volume_ratio >= 2.0:
                return {"type": "elevated", "score": volume_ratio / 4, "ratio": volume_ratio}
            elif volume_ratio <= 0.5:
                return {"type": "declining", "score": 0.2, "ratio": volume_ratio}
            else:
                return {"type": "normal", "score": 0.5, "ratio": volume_ratio}
        except Exception as e:
            logger.error(f"Error analyzing volume pattern: {e}")
            return {"type": "unknown", "score": 0}

    def _calculate_volume_targets(
        self,
        current_price: Decimal,
        volume_ratio: float,
        volume_pattern: Dict
    ) -> List[Decimal]:
        """Calculate target prices for volume-surge trades."""
        targets = []

        # Higher volume = potentially larger move
        base_target = Decimal("0.02") + Decimal(str(min(volume_ratio, 5) * 0.01))  # 2-7%

        # First target: Quick scalp
        targets.append(current_price * (Decimal("1") + base_target))

        # Second target: Extended move
        targets.append(current_price * (Decimal("1") + base_target * Decimal("1.5")))

        # Third target: Full extension
        targets.append(current_price * (Decimal("1") + base_target * Decimal("2")))

        return targets

    def _calculate_smart_money_targets(
        self,
        current_price: Decimal,
        smart_money: Dict,
        score: float
    ) -> List[Decimal]:
        """Calculate target prices for smart money-driven trades."""
        targets = []

        # Smart money accumulation suggests larger moves
        whale_activity = smart_money.get("whale_activity", {})
        net_flow = whale_activity.get("net_flow", 0)

        # Base target on score and net flow
        base_pct = Decimal("0.05") + Decimal(str(score / 2000))  # 5-10%

        if net_flow > 100000:  # Large whale inflow
            base_pct *= Decimal("1.3")  # 30% bonus

        targets.append(current_price * (Decimal("1") + base_pct))
        targets.append(current_price * (Decimal("1") + base_pct * Decimal("1.5")))
        targets.append(current_price * (Decimal("1") + base_pct * Decimal("2")))

        return targets

    def _calculate_smart_money_stop_loss(
        self,
        current_price: Decimal,
        whale_activity: Dict
    ) -> Decimal:
        """Calculate stop loss for smart money trades."""
        # Smart money trades typically have wider stops due to higher conviction
        # But we want to exit if the thesis is invalidated

        avg_whale_entry = whale_activity.get("avg_entry_price")
        if avg_whale_entry and avg_whale_entry > 0:
            # Set stop just below whale average entry
            whale_entry = Decimal(str(avg_whale_entry))
            stop_loss = whale_entry * Decimal("0.95")

            # Don't set stop too far from current price
            max_distance = current_price * Decimal("0.92")
            return max(stop_loss, max_distance)

        # Default: 7% stop loss (wider for conviction trades)
        return current_price * Decimal("0.93")
