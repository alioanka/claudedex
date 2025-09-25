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
        
    async def analyze_patterns(self, data: Dict) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis
        
        Args:
            data: Market data including prices, volumes, etc.
            
        Returns:
            Dictionary with all detected patterns and analysis
        """
        try:
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
