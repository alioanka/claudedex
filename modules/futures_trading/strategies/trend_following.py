"""
Trend Following Strategy for Futures

Opens short positions during confirmed downtrends
Opens long positions during confirmed uptrends

Indicators used:
- Moving averages (SMA/EMA)
- RSI
- MACD
- Volume trends
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class TrendFollowingStrategy:
    """
    Trend following strategy for futures trading

    Entry Signals:
    - SHORT: Price below MA, RSI < 50, declining volume
    - LONG: Price above MA, RSI > 50, increasing volume

    Exit Signals:
    - Trend reversal
    - Stop loss hit
    - Take profit targets
    """

    def __init__(self, config: Dict):
        """
        Initialize trend following strategy

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger("TrendFollowing")

        # Strategy parameters
        self.trend_strength_threshold = config.get('trend_strength_threshold', 0.7)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.min_volume_change = config.get('min_volume_change', 0.2)  # 20% volume increase

    def analyze_trend(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market data for trend signals

        Args:
            market_data: Market data with price, volume, indicators

        Returns:
            Optional[Dict]: Signal dict or None
        """
        try:
            # Extract data
            price = market_data.get('price', 0)
            sma_50 = market_data.get('sma_50', 0)
            sma_200 = market_data.get('sma_200', 0)
            rsi = market_data.get('rsi', 50)
            volume = market_data.get('volume', 0)
            volume_sma = market_data.get('volume_sma', 0)
            macd = market_data.get('macd', {})

            # Determine trend direction
            trend_direction = self._determine_trend(
                price, sma_50, sma_200, rsi, macd
            )

            if not trend_direction:
                return None

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                price, sma_50, sma_200, rsi, volume, volume_sma
            )

            if trend_strength < self.trend_strength_threshold:
                return None

            # Generate signal
            if trend_direction == 'DOWN':
                return {
                    'action': 'SHORT',
                    'symbol': market_data.get('symbol'),
                    'price': price,
                    'trend_strength': trend_strength,
                    'rsi': rsi,
                    'reason': 'Confirmed downtrend',
                    'stop_loss_pct': 0.03,  # 3% stop loss
                    'take_profit_pct': 0.08,  # 8% take profit
                    'recommended_leverage': min(2, self._get_leverage_for_strength(trend_strength))
                }
            elif trend_direction == 'UP':
                return {
                    'action': 'LONG',
                    'symbol': market_data.get('symbol'),
                    'price': price,
                    'trend_strength': trend_strength,
                    'rsi': rsi,
                    'reason': 'Confirmed uptrend',
                    'stop_loss_pct': 0.03,
                    'take_profit_pct': 0.08,
                    'recommended_leverage': min(2, self._get_leverage_for_strength(trend_strength))
                }

            return None

        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return None

    def _determine_trend(
        self,
        price: float,
        sma_50: float,
        sma_200: float,
        rsi: float,
        macd: Dict
    ) -> Optional[str]:
        """
        Determine trend direction

        Returns:
            'UP', 'DOWN', or None
        """
        try:
            # Downtrend signals
            if (
                price < sma_50 < sma_200 and  # Price below both MAs
                rsi < 50 and  # RSI bearish
                macd.get('histogram', 0) < 0  # MACD bearish
            ):
                return 'DOWN'

            # Uptrend signals
            if (
                price > sma_50 > sma_200 and  # Price above both MAs
                rsi > 50 and  # RSI bullish
                macd.get('histogram', 0) > 0  # MACD bullish
            ):
                return 'UP'

            return None

        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return None

    def _calculate_trend_strength(
        self,
        price: float,
        sma_50: float,
        sma_200: float,
        rsi: float,
        volume: float,
        volume_sma: float
    ) -> float:
        """
        Calculate trend strength (0-1)

        Returns:
            float: Trend strength score
        """
        try:
            strength = 0.0

            # MA alignment (30%)
            if sma_50 > 0 and sma_200 > 0:
                ma_spread = abs(sma_50 - sma_200) / sma_200
                strength += min(ma_spread * 10, 0.3)  # Cap at 30%

            # RSI extremes (30%)
            if rsi < self.rsi_oversold:
                strength += 0.3 * (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                strength += 0.3 * (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

            # Volume confirmation (40%)
            if volume_sma > 0:
                volume_ratio = volume / volume_sma
                if volume_ratio > 1:
                    strength += min((volume_ratio - 1) * 0.4, 0.4)

            return min(strength, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0

    def _get_leverage_for_strength(self, strength: float) -> int:
        """
        Recommend leverage based on trend strength

        Args:
            strength: Trend strength (0-1)

        Returns:
            int: Recommended leverage
        """
        if strength >= 0.9:
            return 3
        elif strength >= 0.7:
            return 2
        else:
            return 1

    def should_exit(self, position: Dict, current_data: Dict) -> Optional[Dict]:
        """
        Check if position should be exited

        Args:
            position: Current position info
            current_data: Current market data

        Returns:
            Optional[Dict]: Exit signal or None
        """
        try:
            price = current_data.get('price', 0)
            entry_price = position.get('entry_price', price)
            side = position.get('side', 'LONG')

            # Calculate PnL percentage
            if side == 'LONG':
                pnl_pct = (price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - price) / entry_price

            # Stop loss check
            if pnl_pct <= -0.03:  # -3% stop loss
                return {
                    'action': 'CLOSE',
                    'reason': 'Stop loss hit',
                    'pnl_pct': pnl_pct
                }

            # Take profit check
            if pnl_pct >= 0.08:  # 8% take profit
                return {
                    'action': 'CLOSE',
                    'reason': 'Take profit target reached',
                    'pnl_pct': pnl_pct
                }

            # Trend reversal check
            trend = self._determine_trend(
                price,
                current_data.get('sma_50', 0),
                current_data.get('sma_200', 0),
                current_data.get('rsi', 50),
                current_data.get('macd', {})
            )

            if side == 'SHORT' and trend == 'UP':
                return {
                    'action': 'CLOSE',
                    'reason': 'Trend reversal detected',
                    'pnl_pct': pnl_pct
                }
            elif side == 'LONG' and trend == 'DOWN':
                return {
                    'action': 'CLOSE',
                    'reason': 'Trend reversal detected',
                    'pnl_pct': pnl_pct
                }

            return None

        except Exception as e:
            self.logger.error(f"Error checking exit condition: {e}")
            return None
