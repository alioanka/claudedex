"""
Pattern Analyzer - Technical analysis and pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import talib
from scipy import signal
from sklearn.preprocessing import StandardScaler
import asyncio

class PatternType(Enum):
    """Types of patterns detected"""
    # Bullish patterns
    ASCENDING_TRIANGLE = "ascending_triangle"
    BULL_FLAG = "bull_flag"
    CUP_AND_HANDLE = "cup_and_handle"
    DOUBLE_BOTTOM = "double_bottom"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    BREAKOUT = "breakout"
    MOMENTUM_SURGE = "momentum_surge"
    
    # Bearish patterns
    DESCENDING_TRIANGLE = "descending_triangle"
    BEAR_FLAG = "bear_flag"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_shoulders"
    BREAKDOWN = "breakdown"
    MOMENTUM_CRASH = "momentum_crash"
    
    # Neutral patterns
    CONSOLIDATION = "consolidation"
    RANGE_BOUND = "range_bound"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"

@dataclass
class Pattern:
    """Represents a detected pattern"""
    pattern_type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    strength: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
@dataclass
class TrendInfo:
    """Information about current trend"""
    direction: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0-1
    duration: int  # candles
    slope: float
    r_squared: float  # Trend quality
    
@dataclass
class SupportResistance:
    """Support and resistance levels"""
    supports: List[float]
    resistances: List[float]
    key_level: float
    strength_map: Dict[float, float]  # level -> strength

class PatternAnalyzer:
    """Advanced pattern recognition and technical analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pattern analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_pattern_confidence = self.config.get('min_pattern_confidence', 0.7)
        self.lookback_periods = self.config.get('lookback_periods', {
            'short': 20,
            'medium': 50,
            'long': 200
        })
        
        # Pattern detection parameters
        self.pattern_params = {
            'min_touches': 2,  # Min touches for support/resistance
            'tolerance': 0.02,  # 2% tolerance for pattern matching
            'volume_confirmation': True,
            'breakout_volume_multiplier': 1.5
        }
        
        # Cache for calculations
        self._cache = {}
        self._cache_ttl = 60  # seconds
        
    # Replace the existing analyze_patterns method with this corrected version
    # The API expects: async def analyze_patterns(price_data: List[Dict]) -> Dict

    async def analyze_patterns(self, price_data: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis
        
        Args:
            price_data: List of price dictionaries with OHLCV data
                    Each dict should have: {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
            
        Returns:
            Dictionary with all detected patterns and analysis
        """
        try:
            # Convert price_data list to the internal data format
            data = {
                'prices': price_data,
                'ohlcv': price_data  # For compatibility
            }
            
            # Convert data to DataFrame for analysis
            df = self._prepare_dataframe(data)
            
            if df is None or len(df) < self.lookback_periods['medium']:
                return {
                    'patterns': [],
                    'trend': None,
                    'support_resistance': None,
                    'indicators': {},
                    'score': 0
                }
            
            # Parallel analysis tasks
            tasks = [
                self._detect_chart_patterns(df),
                self._analyze_trend(df),
                self._find_support_resistance(df),
                self._calculate_indicators(df),
                self._analyze_volume_profile(df),
                self._detect_candlestick_patterns(df),
                self._analyze_momentum(df),
                self._detect_breakouts(df)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any errors in parallel tasks
            results = [r if not isinstance(r, Exception) else None for r in results]
            
            patterns, trend, sr_levels, indicators, volume_profile, candlesticks, momentum, breakouts = results
            
            # Combine all detected patterns
            all_patterns = []
            if patterns:
                all_patterns.extend(patterns)
            if candlesticks:
                all_patterns.extend(candlesticks)
            if breakouts:
                all_patterns.extend(breakouts)
                
            # Calculate overall pattern score
            pattern_score = self._calculate_pattern_score(
                all_patterns, trend, indicators, momentum
            )
            
            return {
                'patterns': all_patterns,
                'trend': trend,
                'support_resistance': sr_levels,
                'indicators': indicators,
                'volume_profile': volume_profile,
                'momentum': momentum,
                'score': pattern_score,
                'signal': self._generate_signal(all_patterns, trend, indicators),
                'risk_reward': self._calculate_risk_reward(df, sr_levels),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Pattern analysis error: {e}")
            return {
                'patterns': [],
                'trend': None,
                'support_resistance': None,
                'indicators': {},
                'score': 0,
                'error': str(e)
            }
            
    async def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect classical chart patterns"""
        patterns = []
        
        try:
            # Get price data
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            # Detect various patterns
            patterns.extend(await self._detect_triangles(closes, highs, lows))
            patterns.extend(await self._detect_double_patterns(closes, highs, lows))
            patterns.extend(await self._detect_head_shoulders(closes, highs, lows))
            patterns.extend(await self._detect_flags(closes, volumes))
            patterns.extend(await self._detect_cup_handle(closes, volumes))
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= self.min_pattern_confidence]
            
            return patterns
            
        except Exception as e:
            print(f"Chart pattern detection error: {e}")
            return []
            
    async def _detect_triangles(self, closes: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray) -> List[Pattern]:
        """Detect triangle patterns"""
        patterns = []
        
        try:
            # Find peaks and troughs
            peaks = signal.find_peaks(highs, distance=5)[0]
            troughs = signal.find_peaks(-lows, distance=5)[0]
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Check for ascending triangle
                peak_trend = np.polyfit(peaks, highs[peaks], 1)[0]
                trough_trend = np.polyfit(troughs, lows[troughs], 1)[0]
                
                if abs(peak_trend) < 0.001 and trough_trend > 0.001:  # Flat top, rising bottom
                    patterns.append(Pattern(
                        pattern_type=PatternType.ASCENDING_TRIANGLE,
                        confidence=0.8,
                        start_time=datetime.now() - timedelta(days=len(closes)),
                        end_time=datetime.now(),
                        price_target=highs[peaks[-1]] * 1.05,
                        stop_loss=lows[troughs[-1]] * 0.98,
                        strength=0.75
                    ))
                    
                # Check for descending triangle
                elif peak_trend < -0.001 and abs(trough_trend) < 0.001:  # Falling top, flat bottom
                    patterns.append(Pattern(
                        pattern_type=PatternType.DESCENDING_TRIANGLE,
                        confidence=0.8,
                        start_time=datetime.now() - timedelta(days=len(closes)),
                        end_time=datetime.now(),
                        price_target=lows[troughs[-1]] * 0.95,
                        stop_loss=highs[peaks[-1]] * 1.02,
                        strength=0.75
                    ))
                    
        except Exception as e:
            print(f"Triangle detection error: {e}")
            
        return patterns
        
    async def _detect_double_patterns(self, closes: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray) -> List[Pattern]:
        """Detect double top/bottom patterns"""
        patterns = []
        
        try:
            # Find peaks for double top
            peaks = signal.find_peaks(highs, distance=10, prominence=closes.std()*0.5)[0]
            
            if len(peaks) >= 2:
                # Check if two peaks are at similar levels
                for i in range(len(peaks)-1):
                    if abs(highs[peaks[i]] - highs[peaks[i+1]]) / highs[peaks[i]] < 0.02:
                        # Double top found
                        patterns.append(Pattern(
                            pattern_type=PatternType.DOUBLE_TOP,
                            confidence=0.75,
                            start_time=datetime.now() - timedelta(days=peaks[i+1]-peaks[i]),
                            end_time=datetime.now(),
                            price_target=closes[-1] * 0.95,
                            stop_loss=highs[peaks[i+1]] * 1.02,
                            strength=0.7
                        ))
                        
            # Find troughs for double bottom
            troughs = signal.find_peaks(-lows, distance=10, prominence=closes.std()*0.5)[0]
            
            if len(troughs) >= 2:
                # Check if two troughs are at similar levels
                for i in range(len(troughs)-1):
                    if abs(lows[troughs[i]] - lows[troughs[i+1]]) / lows[troughs[i]] < 0.02:
                        # Double bottom found
                        patterns.append(Pattern(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            confidence=0.75,
                            start_time=datetime.now() - timedelta(days=troughs[i+1]-troughs[i]),
                            end_time=datetime.now(),
                            price_target=closes[-1] * 1.05,
                            stop_loss=lows[troughs[i+1]] * 0.98,
                            strength=0.7
                        ))
                        
        except Exception as e:
            print(f"Double pattern detection error: {e}")
            
        return patterns
        
    async def _analyze_trend(self, df: pd.DataFrame) -> Optional[TrendInfo]:
        """Analyze current trend"""
        try:
            closes = df['close'].values
            
            # Linear regression for trend
            x = np.arange(len(closes))
            z = np.polyfit(x, closes, 1)
            slope = z[0]
            
            # Calculate R-squared
            p = np.poly1d(z)
            yhat = p(x)
            ybar = np.mean(closes)
            ssreg = np.sum((yhat - ybar)**2)
            sstot = np.sum((closes - ybar)**2)
            r_squared = ssreg / sstot if sstot > 0 else 0
            
            # Determine trend direction
            if slope > closes[-1] * 0.001:  # 0.1% per period
                direction = 'uptrend'
            elif slope < -closes[-1] * 0.001:
                direction = 'downtrend'
            else:
                direction = 'sideways'
                
            # Calculate trend strength
            strength = min(abs(slope) / (closes[-1] * 0.01), 1.0)
            
            # Count trend duration
            duration = self._calculate_trend_duration(closes, direction)
            
            return TrendInfo(
                direction=direction,
                strength=strength,
                duration=duration,
                slope=slope,
                r_squared=r_squared
            )
            
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return None
            
    async def _find_support_resistance(self, df: pd.DataFrame) -> Optional[SupportResistance]:
        """Find support and resistance levels"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Find local maxima and minima
            peaks = signal.find_peaks(highs, distance=5)[0]
            troughs = signal.find_peaks(-lows, distance=5)[0]
            
            # Cluster levels
            levels = []
            strength_map = {}
            
            # Add peak levels
            for peak in peaks:
                level = highs[peak]
                levels.append(level)
                strength_map[level] = strength_map.get(level, 0) + 1
                
            # Add trough levels
            for trough in troughs:
                level = lows[trough]
                levels.append(level)
                strength_map[level] = strength_map.get(level, 0) + 1
                
            # Cluster nearby levels
            clustered_levels = self._cluster_levels(levels)
            
            # Separate into support and resistance
            current_price = closes[-1]
            supports = [l for l in clustered_levels if l < current_price]
            resistances = [l for l in clustered_levels if l > current_price]
            
            # Find key level (most tested)
            key_level = max(strength_map.keys(), key=lambda k: strength_map[k]) if strength_map else current_price
            
            return SupportResistance(
                supports=sorted(supports, reverse=True)[:3],  # Top 3 supports
                resistances=sorted(resistances)[:3],  # Top 3 resistances
                key_level=key_level,
                strength_map=strength_map
            )
            
        except Exception as e:
            print(f"Support/Resistance error: {e}")
            return None
            
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = talib.SMA(closes, timeperiod=20)[-1]
            indicators['sma_50'] = talib.SMA(closes, timeperiod=50)[-1]
            indicators['ema_12'] = talib.EMA(closes, timeperiod=12)[-1]
            indicators['ema_26'] = talib.EMA(closes, timeperiod=26)[-1]
            
            # Momentum indicators
            indicators['rsi'] = talib.RSI(closes, timeperiod=14)[-1]
            macd, signal_line, _ = talib.MACD(closes)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = signal_line[-1]
            indicators['macd_histogram'] = macd[-1] - signal_line[-1]
            
            # Volatility indicators
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
            indicators['bb_upper'] = upper[-1]
            indicators['bb_middle'] = middle[-1]
            indicators['bb_lower'] = lower[-1]
            indicators['bb_width'] = (upper[-1] - lower[-1]) / middle[-1]
            
            indicators['atr'] = talib.ATR(highs, lows, closes, timeperiod=14)[-1]
            
            # Volume indicators
            indicators['obv'] = talib.OBV(closes, volumes)[-1]
            indicators['volume_sma'] = talib.SMA(volumes, timeperiod=20)[-1]
            indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
            
            # Trend indicators
            indicators['adx'] = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
            indicators['cci'] = talib.CCI(highs, lows, closes, timeperiod=20)[-1]
            
            # Stochastic
            slowk, slowd = talib.STOCH(highs, lows, closes)
            indicators['stoch_k'] = slowk[-1]
            indicators['stoch_d'] = slowd[-1]
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(highs, lows, closes, timeperiod=14)[-1]
            
            # Money Flow Index
            indicators['mfi'] = talib.MFI(highs, lows, closes, volumes, timeperiod=14)[-1]
            
            return indicators

        except Exception as e:
            print(f"Indicator calculation error: {e}")
            return []

    # Add these sync method wrappers to existing PatternAnalyzer class
    
    def detect_chart_patterns(self, prices: List[float]) -> List[str]:
        """Sync wrapper for chart pattern detection"""
        try:
            # Create DataFrame from prices
            df = pd.DataFrame({'close': prices, 'high': prices, 'low': prices, 'volume': [1000]*len(prices)})
            
            # Run async method synchronously
            loop = asyncio.get_event_loop()
            patterns = loop.run_until_complete(self._detect_chart_patterns(df))
            
            # Convert to string list
            return [p.pattern_type.value for p in patterns]
            
        except Exception as e:
            print(f"Chart pattern detection error: {e}")
            return []
    
    def detect_candlestick_patterns(self, ohlc_data: List[Dict]) -> List[str]:
        """Sync wrapper for candlestick pattern detection"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data)
            
            # Run async method synchronously
            loop = asyncio.get_event_loop()
            patterns = loop.run_until_complete(self._detect_candlestick_patterns(df))
            
            # Convert to string list
            return [p.pattern_type.value for p in patterns]
            
        except Exception as e:
            print(f"Candlestick pattern detection error: {e}")
            return []
    
    def calculate_support_resistance(self, prices: List[float]) -> Dict:
        """Sync wrapper for support/resistance calculation"""
        try:
            # Create DataFrame
            df = pd.DataFrame({'close': prices, 'high': prices, 'low': prices})
            
            # Run async method synchronously
            loop = asyncio.get_event_loop()
            sr_levels = loop.run_until_complete(self._find_support_resistance(df))
            
            if sr_levels:
                return {
                    'supports': sr_levels.supports,
                    'resistances': sr_levels.resistances,
                    'key_level': sr_levels.key_level
                }
            
            return {'supports': [], 'resistances': [], 'key_level': prices[-1] if prices else 0}
            
        except Exception as e:
            print(f"Support/resistance calculation error: {e}")
            return {'supports': [], 'resistances': [], 'key_level': 0}
    
    def identify_trend(self, prices: List[float]) -> str:
        """Sync wrapper for trend identification"""
        try:
            # Create DataFrame
            df = pd.DataFrame({'close': prices})
            
            # Run async method synchronously
            loop = asyncio.get_event_loop()
            trend_info = loop.run_until_complete(self._analyze_trend(df))
            
            return trend_info.direction if trend_info else 'sideways'
            
        except Exception as e:
            print(f"Trend identification error: {e}")
            return 'sideways'

    # Add these missing helper methods to the PatternAnalyzer class

    def _prepare_dataframe(self, data: Dict) -> Optional[pd.DataFrame]:
        """Convert input data to DataFrame for analysis"""
        try:
            if 'prices' in data and isinstance(data['prices'], list):
                # Handle list of price dictionaries
                df = pd.DataFrame(data['prices'])
                
                # Ensure required columns exist
                required_cols = ['close']
                if not all(col in df.columns for col in required_cols):
                    # If only prices as numbers, create DataFrame
                    if all(isinstance(p, (int, float)) for p in data['prices']):
                        df = pd.DataFrame({
                            'close': data['prices'],
                            'high': data['prices'],
                            'low': data['prices'],
                            'open': data['prices'],
                            'volume': [1000] * len(data['prices'])
                        })
                    else:
                        return None
            else:
                # Handle other data formats
                df = pd.DataFrame(data)
                
            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Remove NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"DataFrame preparation error: {e}")
            return None

    def _calculate_trend_duration(self, closes: np.ndarray, direction: str) -> int:
        """Calculate how long the current trend has been active"""
        try:
            duration = 0
            
            if direction == 'uptrend':
                for i in range(len(closes) - 1, 0, -1):
                    if closes[i] > closes[i-1]:
                        duration += 1
                    else:
                        break
            elif direction == 'downtrend':
                for i in range(len(closes) - 1, 0, -1):
                    if closes[i] < closes[i-1]:
                        duration += 1
                    else:
                        break
            else:  # sideways
                # Count periods within a range
                recent_mean = np.mean(closes[-20:])
                recent_std = np.std(closes[-20:])
                for i in range(len(closes) - 1, 0, -1):
                    if abs(closes[i] - recent_mean) < recent_std:
                        duration += 1
                    else:
                        break
                        
            return duration
            
        except Exception as e:
            print(f"Trend duration calculation error: {e}")
            return 0

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        """Cluster nearby price levels"""
        try:
            if not levels:
                return []
                
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    # Save cluster average
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
                    
            # Add last cluster
            if current_cluster:
                clusters.append(np.mean(current_cluster))
                
            return clusters
            
        except Exception as e:
            print(f"Level clustering error: {e}")
            return []

    async def _detect_head_shoulders(self, closes: np.ndarray, highs: np.ndarray,
                                    lows: np.ndarray) -> List[Pattern]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            # Find peaks
            peaks = signal.find_peaks(highs, distance=5, prominence=closes.std()*0.3)[0]
            
            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                for i in range(len(peaks) - 2):
                    left_shoulder = highs[peaks[i]]
                    head = highs[peaks[i + 1]]
                    right_shoulder = highs[peaks[i + 2]]
                    
                    # Head should be higher than shoulders
                    if head > left_shoulder and head > right_shoulder:
                        # Shoulders should be roughly equal
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                            patterns.append(Pattern(
                                pattern_type=PatternType.HEAD_SHOULDERS,
                                confidence=0.8,
                                start_time=datetime.now() - timedelta(days=peaks[i + 2] - peaks[i]),
                                end_time=datetime.now(),
                                price_target=closes[-1] * 0.93,
                                stop_loss=head * 1.02,
                                strength=0.75
                            ))
                            
            # Check for inverse head and shoulders
            troughs = signal.find_peaks(-lows, distance=5, prominence=closes.std()*0.3)[0]
            
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left_shoulder = lows[troughs[i]]
                    head = lows[troughs[i + 1]]
                    right_shoulder = lows[troughs[i + 2]]
                    
                    # Head should be lower than shoulders
                    if head < left_shoulder and head < right_shoulder:
                        # Shoulders should be roughly equal
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                            patterns.append(Pattern(
                                pattern_type=PatternType.INVERSE_HEAD_SHOULDERS,
                                confidence=0.8,
                                start_time=datetime.now() - timedelta(days=troughs[i + 2] - troughs[i]),
                                end_time=datetime.now(),
                                price_target=closes[-1] * 1.07,
                                stop_loss=head * 0.98,
                                strength=0.75
                            ))
                            
        except Exception as e:
            print(f"Head and shoulders detection error: {e}")
            
        return patterns

    async def _detect_flags(self, closes: np.ndarray, volumes: np.ndarray) -> List[Pattern]:
        """Detect flag patterns"""
        patterns = []
        
        try:
            # Look for strong move followed by consolidation
            for i in range(20, len(closes) - 10):
                # Check for strong move (pole)
                pole_start = i - 20
                pole_end = i
                pole_change = (closes[pole_end] - closes[pole_start]) / closes[pole_start]
                
                # Check for consolidation (flag)
                flag_prices = closes[i:i + 10]
                flag_std = np.std(flag_prices) / np.mean(flag_prices)
                
                # Bull flag: strong up move + tight consolidation
                if pole_change > 0.15 and flag_std < 0.03:
                    patterns.append(Pattern(
                        pattern_type=PatternType.BULL_FLAG,
                        confidence=0.75,
                        start_time=datetime.now() - timedelta(days=30),
                        end_time=datetime.now(),
                        price_target=closes[-1] * (1 + pole_change * 0.7),
                        stop_loss=min(flag_prices) * 0.98,
                        strength=0.7
                    ))
                    
                # Bear flag: strong down move + tight consolidation
                elif pole_change < -0.15 and flag_std < 0.03:
                    patterns.append(Pattern(
                        pattern_type=PatternType.BEAR_FLAG,
                        confidence=0.75,
                        start_time=datetime.now() - timedelta(days=30),
                        end_time=datetime.now(),
                        price_target=closes[-1] * (1 + pole_change * 0.7),
                        stop_loss=max(flag_prices) * 1.02,
                        strength=0.7
                    ))
                    
        except Exception as e:
            print(f"Flag pattern detection error: {e}")
            
        return patterns

    async def _detect_cup_handle(self, closes: np.ndarray, volumes: np.ndarray) -> List[Pattern]:
        """Detect cup and handle patterns"""
        patterns = []
        
        try:
            # Simplified cup and handle detection
            if len(closes) >= 50:
                # Find potential cup bottom
                min_idx = np.argmin(closes[:40])
                
                # Check if we have a U-shape
                left_rim = closes[0]
                bottom = closes[min_idx]
                right_rim = closes[40]
                
                # Cup should be U-shaped
                if left_rim > bottom and right_rim > bottom:
                    # Rims should be similar height
                    if abs(left_rim - right_rim) / left_rim < 0.1:
                        # Check for handle (small pullback after right rim)
                        handle_prices = closes[40:50]
                        handle_low = min(handle_prices)
                        
                        if handle_low > bottom and handle_low < right_rim:
                            patterns.append(Pattern(
                                pattern_type=PatternType.CUP_AND_HANDLE,
                                confidence=0.7,
                                start_time=datetime.now() - timedelta(days=50),
                                end_time=datetime.now(),
                                price_target=right_rim * 1.15,
                                stop_loss=handle_low * 0.98,
                                strength=0.65
                            ))
                            
        except Exception as e:
            print(f"Cup and handle detection error: {e}")
            
        return patterns

    async def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile"""
        try:
            closes = df['close'].values
            volumes = df['volume'].values
            
            # Calculate volume-weighted average price (VWAP)
            vwap = np.sum(closes * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else closes[-1]
            
            # Find high volume nodes
            price_bins = np.linspace(min(closes), max(closes), 20)
            volume_profile = {}
            
            for i in range(len(price_bins) - 1):
                mask = (closes >= price_bins[i]) & (closes < price_bins[i + 1])
                volume_profile[price_bins[i]] = np.sum(volumes[mask])
                
            # Find point of control (highest volume price)
            poc = max(volume_profile, key=volume_profile.get) if volume_profile else closes[-1]
            
            return {
                'vwap': vwap,
                'poc': poc,
                'volume_profile': volume_profile,
                'current_volume': volumes[-1] if len(volumes) > 0 else 0,
                'avg_volume': np.mean(volumes) if len(volumes) > 0 else 0
            }
            
        except Exception as e:
            print(f"Volume profile analysis error: {e}")
            return {}

    async def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect candlestick patterns using TA-Lib"""
        patterns = []
        
        try:
            opens = df['open'].values if 'open' in df.columns else df['close'].values
            highs = df['high'].values if 'high' in df.columns else df['close'].values
            lows = df['low'].values if 'low' in df.columns else df['close'].values
            closes = df['close'].values
            
            # Detect various candlestick patterns
            # Note: These would need proper mapping to PatternType enum
            
            # Bullish patterns
            hammer = talib.CDLHAMMER(opens, highs, lows, closes)
            if hammer[-1] > 0:
                patterns.append(Pattern(
                    pattern_type=PatternType.MOMENTUM_SURGE,  # Using available enum
                    confidence=0.7,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    strength=0.65
                ))
                
            # Bearish patterns
            shooting_star = talib.CDLSHOOTINGSTAR(opens, highs, lows, closes)
            if shooting_star[-1] < 0:
                patterns.append(Pattern(
                    pattern_type=PatternType.MOMENTUM_CRASH,  # Using available enum
                    confidence=0.7,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    strength=0.65
                ))
                
        except Exception as e:
            print(f"Candlestick pattern detection error: {e}")
            
        return patterns

    async def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            closes = df['close'].values
            volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(closes))
            
            # Calculate momentum
            momentum_periods = 10
            if len(closes) > momentum_periods:
                momentum = (closes[-1] - closes[-momentum_periods]) / closes[-momentum_periods]
            else:
                momentum = 0
                
            # Calculate rate of change
            roc_periods = 20
            if len(closes) > roc_periods:
                roc = (closes[-1] - closes[-roc_periods]) / closes[-roc_periods]
            else:
                roc = 0
                
            # Volume momentum
            if len(volumes) > 20:
                recent_vol = np.mean(volumes[-5:])
                older_vol = np.mean(volumes[-20:-15])
                volume_momentum = recent_vol / older_vol if older_vol > 0 else 1
            else:
                volume_momentum = 1
                
            # Determine momentum direction
            if momentum > 0.05:
                direction = 'bullish'
            elif momentum < -0.05:
                direction = 'bearish'
            else:
                direction = 'neutral'
                
            return {
                'momentum': momentum,
                'momentum_score': abs(momentum),
                'roc': roc,
                'volume_momentum': volume_momentum,
                'direction': direction
            }
            
        except Exception as e:
            print(f"Momentum analysis error: {e}")
            return {'momentum': 0, 'momentum_score': 0, 'direction': 'neutral'}

    async def _detect_breakouts(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect breakout patterns"""
        patterns = []
        
        try:
            closes = df['close'].values
            highs = df['high'].values if 'high' in df.columns else closes
            lows = df['low'].values if 'low' in df.columns else closes
            volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(closes))
            
            # Find recent resistance
            if len(highs) >= 20:
                recent_resistance = max(highs[-20:-1])
                
                # Check if current price broke resistance
                if closes[-1] > recent_resistance * 1.02:  # 2% above resistance
                    # Check volume confirmation
                    avg_volume = np.mean(volumes[-20:])
                    if volumes[-1] > avg_volume * 1.5:
                        patterns.append(Pattern(
                            pattern_type=PatternType.BREAKOUT,
                            confidence=0.8,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            price_target=closes[-1] * 1.1,
                            stop_loss=recent_resistance * 0.98,
                            strength=0.75
                        ))
                        
            # Find recent support
            if len(lows) >= 20:
                recent_support = min(lows[-20:-1])
                
                # Check if current price broke support
                if closes[-1] < recent_support * 0.98:  # 2% below support
                    # Check volume confirmation
                    avg_volume = np.mean(volumes[-20:])
                    if volumes[-1] > avg_volume * 1.5:
                        patterns.append(Pattern(
                            pattern_type=PatternType.BREAKDOWN,
                            confidence=0.8,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            price_target=closes[-1] * 0.9,
                            stop_loss=recent_support * 1.02,
                            strength=0.75
                        ))
                        
        except Exception as e:
            print(f"Breakout detection error: {e}")
            
        return patterns

    def _calculate_pattern_score(self, patterns: List[Pattern], trend: Optional[TrendInfo],
                                indicators: Dict, momentum: Dict) -> float:
        """Calculate overall pattern score"""
        try:
            score = 0.0
            
            # Pattern contribution
            if patterns:
                pattern_scores = [p.confidence * p.strength for p in patterns]
                score += np.mean(pattern_scores) * 0.3
                
            # Trend contribution
            if trend:
                score += trend.strength * trend.r_squared * 0.2
                
            # Indicator contribution
            if indicators:
                # RSI score
                rsi = indicators.get('rsi', 50)
                if 30 < rsi < 70:
                    score += 0.1
                    
                # MACD score
                if indicators.get('macd_histogram', 0) > 0:
                    score += 0.1
                    
                # ADX score (trend strength)
                adx = indicators.get('adx', 0)
                if adx > 25:
                    score += 0.1
                    
            # Momentum contribution
            if momentum:
                score += momentum.get('momentum_score', 0) * 0.2
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Pattern score calculation error: {e}")
            return 0.0

    def _generate_signal(self, patterns: List[Pattern], trend: Optional[TrendInfo],
                        indicators: Dict) -> str:
        """Generate trading signal based on analysis"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Check patterns
            for pattern in patterns:
                if 'bull' in pattern.pattern_type.value.lower() or 'ascending' in pattern.pattern_type.value.lower():
                    bullish_signals += 1
                elif 'bear' in pattern.pattern_type.value.lower() or 'descending' in pattern.pattern_type.value.lower():
                    bearish_signals += 1
                    
            # Check trend
            if trend:
                if trend.direction == 'uptrend':
                    bullish_signals += 1
                elif trend.direction == 'downtrend':
                    bearish_signals += 1
                    
            # Check indicators
            if indicators:
                if indicators.get('rsi', 50) < 30:
                    bullish_signals += 1
                elif indicators.get('rsi', 50) > 70:
                    bearish_signals += 1
                    
                if indicators.get('macd_histogram', 0) > 0:
                    bullish_signals += 1
                elif indicators.get('macd_histogram', 0) < 0:
                    bearish_signals += 1
                    
            # Generate signal
            if bullish_signals > bearish_signals + 1:
                return 'strong_buy'
            elif bullish_signals > bearish_signals:
                return 'buy'
            elif bearish_signals > bullish_signals + 1:
                return 'strong_sell'
            elif bearish_signals > bullish_signals:
                return 'sell'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Signal generation error: {e}")
            return 'neutral'

    def _calculate_risk_reward(self, df: pd.DataFrame, sr_levels: Optional[SupportResistance]) -> Dict:
        """Calculate risk/reward ratio"""
        try:
            current_price = df['close'].values[-1]
            risk_reward = {}
            
            if sr_levels:
                # Find nearest support (risk)
                supports = [s for s in sr_levels.supports if s < current_price]
                if supports:
                    stop_loss = supports[0]
                    risk = (current_price - stop_loss) / current_price
                else:
                    risk = 0.05  # Default 5% risk
                    
                # Find nearest resistance (reward)
                resistances = [r for r in sr_levels.resistances if r > current_price]
                if resistances:
                    take_profit = resistances[0]
                    reward = (take_profit - current_price) / current_price
                else:
                    reward = 0.10  # Default 10% reward
                    
                risk_reward = {
                    'ratio': reward / risk if risk > 0 else 0,
                    'risk_percent': risk * 100,
                    'reward_percent': reward * 100
                }
            else:
                risk_reward = {
                    'ratio': 2.0,  # Default 2:1
                    'risk_percent': 5.0,
                    'reward_percent': 10.0
                }
                
            return risk_reward
            
        except Exception as e:
            print(f"Risk/reward calculation error: {e}")
            return {'ratio': 0, 'risk_percent': 0, 'reward_percent': 0}