# analysis/market_analyzer.py

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import aiohttp

from ..core.event_bus import EventBus
from ..data.storage.database import DatabaseManager
from ..data.storage.cache import CacheManager
from ..ml.models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)

@dataclass
class MarketCondition:
    """Represents current market conditions"""
    trend: str  # 'bullish', 'bearish', 'neutral'
    volatility: str  # 'low', 'medium', 'high', 'extreme'
    liquidity: str  # 'poor', 'fair', 'good', 'excellent'
    sentiment: str  # 'fearful', 'neutral', 'greedy'
    momentum: float  # -1 to 1
    strength: float  # 0 to 1
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    market_phase: str  # 'accumulation', 'markup', 'distribution', 'markdown'
    confidence: float  # 0 to 1

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    primary_trend: str
    trend_strength: float
    trend_duration: int  # in hours
    support_levels: List[Decimal]
    resistance_levels: List[Decimal]
    breakout_probability: float
    reversal_probability: float
    next_target: Optional[Decimal]

@dataclass
class CorrelationMatrix:
    """Token correlation analysis"""
    correlations: pd.DataFrame
    clusters: List[List[str]]  # Grouped correlated tokens
    independent_tokens: List[str]
    max_correlation_pairs: List[Tuple[str, str, float]]
    portfolio_risk: float

class MarketAnalyzer:
    """Advanced market analysis and pattern recognition"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_manager: CacheManager,
        event_bus: EventBus,
        ml_model: Optional[EnsembleModel],
        config: Dict[str, Any]
    ):
        self.db = db_manager
        self.cache = cache_manager
        self.event_bus = event_bus
        self.ml_model = ml_model
        self.config = config
        
        # Analysis parameters
        self.timeframes = config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.min_data_points = config.get('min_data_points', 100)
        
        # Market indicators
        self.indicators = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb': {'period': 20, 'std': 2},
            'ema': {'periods': [9, 21, 50, 200]},
            'volume': {'ma_period': 20},
            'atr': {'period': 14}
        }
        
        # Market regimes
        self.market_regimes = {
            'trending': {'volatility': (0.1, 0.3), 'trend_strength': (0.6, 1.0)},
            'ranging': {'volatility': (0.05, 0.15), 'trend_strength': (0, 0.3)},
            'volatile': {'volatility': (0.3, 1.0), 'trend_strength': (0, 1.0)},
            'quiet': {'volatility': (0, 0.05), 'trend_strength': (0, 0.2)}
        }
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, Any] = {}
        self._correlation_cache: Optional[pd.DataFrame] = None
        self._last_correlation_update: Optional[datetime] = None
        
    async def analyze_market_conditions(
        self,
        tokens: Optional[List[str]] = None,
        chain: str = 'ethereum'
    ) -> MarketCondition:
        """Analyze overall market conditions"""
        try:
            # Get market data
            if tokens:
                market_data = await self._get_market_data(tokens, chain)
            else:
                # Get top tokens if not specified
                market_data = await self._get_top_tokens_data(chain, limit=100)
                
            # Calculate market metrics
            metrics = await self._calculate_market_metrics(market_data)
            
            # Determine trend
            trend = await self._determine_market_trend(metrics)
            
            # Calculate volatility
            volatility = await self._calculate_market_volatility(metrics)
            
            # Assess liquidity
            liquidity = await self._assess_market_liquidity(market_data)
            
            # Get sentiment indicators
            sentiment = await self._analyze_market_sentiment(metrics)
            
            # Calculate momentum
            momentum = await self._calculate_market_momentum(metrics)
            
            # Determine market phase
            market_phase = await self._identify_market_phase(metrics, trend)
            
            # Calculate risk level
            risk_level = self._calculate_market_risk(volatility, liquidity, sentiment)
            
            # Calculate confidence
            confidence = await self._calculate_analysis_confidence(metrics)
            
            # Cache results
            condition = MarketCondition(
                trend=trend['direction'],
                volatility=volatility['level'],
                liquidity=liquidity['level'],
                sentiment=sentiment['level'],
                momentum=momentum['value'],
                strength=trend['strength'],
                risk_level=risk_level,
                market_phase=market_phase,
                confidence=confidence
            )
            
            # Emit market condition event
            await self.event_bus.emit('market.condition', condition.__dict__)
            
            return condition
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return MarketCondition(
                trend='unknown',
                volatility='unknown',
                liquidity='unknown',
                sentiment='neutral',
                momentum=0.0,
                strength=0.0,
                risk_level='high',
                market_phase='unknown',
                confidence=0.0
            )
            
    async def identify_trends(
        self,
        token: str,
        chain: str = 'ethereum',
        timeframe: str = '1h'
    ) -> TrendAnalysis:
        """Identify and analyze price trends"""
        try:
            # Get historical price data
            price_data = await self._get_price_history(token, chain, timeframe)
            
            if len(price_data) < self.min_data_points:
                logger.warning(f"Insufficient data for trend analysis: {token}")
                return self._get_default_trend_analysis()
                
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['price'] = pd.to_numeric(df['price'])
            
            # Calculate trend indicators
            trend_indicators = await self._calculate_trend_indicators(df)
            
            # Identify support and resistance levels
            support_resistance = await self._identify_support_resistance(df)
            
            # Calculate trend strength and direction
            trend_metrics = await self._calculate_trend_metrics(df, trend_indicators)
            
            # Predict breakout/reversal probabilities
            predictions = await self._predict_trend_changes(df, trend_indicators)
            
            # Calculate next target
            next_target = await self._calculate_price_target(
                df,
                trend_metrics,
                support_resistance
            )
            
            return TrendAnalysis(
                primary_trend=trend_metrics['direction'],
                trend_strength=trend_metrics['strength'],
                trend_duration=trend_metrics['duration'],
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance'],
                breakout_probability=predictions['breakout_prob'],
                reversal_probability=predictions['reversal_prob'],
                next_target=next_target
            )
            
        except Exception as e:
            logger.error(f"Error identifying trends for {token}: {e}")
            return self._get_default_trend_analysis()
            
    async def calculate_correlations(
        self,
        tokens: List[str],
        chain: str = 'ethereum',
        period: int = 30  # days
    ) -> CorrelationMatrix:
        """Calculate correlation matrix between tokens"""
        try:
            # Check cache
            cache_key = f"correlations:{chain}:{'-'.join(sorted(tokens))}"
            cached = await self.cache.get(cache_key)
            
            if cached and self._is_correlation_cache_valid():
                return CorrelationMatrix(**cached)
                
            # Get price data for all tokens
            price_data = {}
            for token in tokens:
                data = await self.db.get_historical_data(
                    token=token,
                    chain=chain,
                    days=period
                )
                if data:
                    price_data[token] = pd.Series(
                        [d['price'] for d in data],
                        index=[d['timestamp'] for d in data]
                    )
                    
            if len(price_data) < 2:
                logger.warning("Insufficient tokens for correlation analysis")
                return self._get_default_correlation_matrix()
                
            # Create price matrix
            price_matrix = pd.DataFrame(price_data)
            
            # Calculate returns
            returns = price_matrix.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Identify clusters of correlated tokens
            clusters = await self._identify_correlation_clusters(correlation_matrix)
            
            # Find independent tokens
            independent = await self._find_independent_tokens(correlation_matrix)
            
            # Find maximum correlation pairs
            max_pairs = await self._find_max_correlation_pairs(correlation_matrix)
            
            # Calculate portfolio risk
            portfolio_risk = await self._calculate_portfolio_risk(returns, correlation_matrix)
            
            # Create result
            result = CorrelationMatrix(
                correlations=correlation_matrix,
                clusters=clusters,
                independent_tokens=independent,
                max_correlation_pairs=max_pairs,
                portfolio_risk=portfolio_risk
            )
            
            # Cache results
            await self.cache.set(cache_key, result.__dict__, ttl=3600)
            self._correlation_cache = correlation_matrix
            self._last_correlation_update = datetime.utcnow()
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return self._get_default_correlation_matrix()
            
    async def detect_market_regime(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Detect current market regime for a token"""
        try:
            # Get recent price and volume data
            data = await self._get_recent_market_data(token, chain, hours=24)
            
            if not data:
                return {'regime': 'unknown', 'confidence': 0.0}
                
            # Calculate regime indicators
            volatility = await self._calculate_volatility(data)
            trend_strength = await self._calculate_trend_strength(data)
            volume_profile = await self._analyze_volume_profile(data)
            
            # Determine regime
            regime = 'unknown'
            confidence = 0.0
            
            for regime_name, thresholds in self.market_regimes.items():
                vol_match = thresholds['volatility'][0] <= volatility <= thresholds['volatility'][1]
                trend_match = thresholds['trend_strength'][0] <= trend_strength <= thresholds['trend_strength'][1]
                
                if vol_match and trend_match:
                    regime = regime_name
                    # Calculate confidence based on how well it matches
                    vol_center = (thresholds['volatility'][0] + thresholds['volatility'][1]) / 2
                    trend_center = (thresholds['trend_strength'][0] + thresholds['trend_strength'][1]) / 2
                    
                    vol_distance = abs(volatility - vol_center) / (thresholds['volatility'][1] - thresholds['volatility'][0])
                    trend_distance = abs(trend_strength - trend_center) / (thresholds['trend_strength'][1] - thresholds['trend_strength'][0])
                    
                    confidence = 1.0 - (vol_distance + trend_distance) / 2
                    break
                    
            # Get regime characteristics
            characteristics = await self._get_regime_characteristics(
                regime, volatility, trend_strength, volume_profile
            )
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_profile': volume_profile,
                'characteristics': characteristics,
                'trading_recommendation': self._get_regime_recommendation(regime),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime for {token}: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
            
    async def find_market_inefficiencies(
        self,
        tokens: List[str],
        chain: str = 'ethereum'
    ) -> List[Dict[str, Any]]:
        """Find market inefficiencies and arbitrage opportunities"""
        try:
            inefficiencies = []
            
            # Get current prices across multiple DEXs
            price_data = await self._get_multi_dex_prices(tokens, chain)
            
            for token in tokens:
                if token not in price_data:
                    continue
                    
                token_prices = price_data[token]
                
                # Check for price discrepancies
                if len(token_prices) >= 2:
                    prices = [p['price'] for p in token_prices.values()]
                    min_price = min(prices)
                    max_price = max(prices)
                    
                    spread_percentage = ((max_price - min_price) / min_price) * 100
                    
                    if spread_percentage > 1.0:  # More than 1% spread
                        min_dex = min(token_prices.items(), key=lambda x: x[1]['price'])[0]
                        max_dex = max(token_prices.items(), key=lambda x: x[1]['price'])[0]
                        
                        inefficiencies.append({
                            'type': 'price_discrepancy',
                            'token': token,
                            'chain': chain,
                            'spread_percentage': spread_percentage,
                            'buy_dex': min_dex,
                            'buy_price': min_price,
                            'sell_dex': max_dex,
                            'sell_price': max_price,
                            'potential_profit': spread_percentage - 0.6,  # Minus fees
                            'liquidity_buy': token_prices[min_dex].get('liquidity', 0),
                            'liquidity_sell': token_prices[max_dex].get('liquidity', 0),
                            'risk_score': await self._calculate_arbitrage_risk(
                                spread_percentage,
                                token_prices[min_dex].get('liquidity', 0),
                                token_prices[max_dex].get('liquidity', 0)
                            )
                        })
                        
            # Sort by potential profit
            inefficiencies.sort(key=lambda x: x.get('potential_profit', 0), reverse=True)
            
            return inefficiencies
            
        except Exception as e:
            logger.error(f"Error finding market inefficiencies: {e}")
            return []
            
    async def analyze_volume_patterns(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Analyze volume patterns and anomalies"""
        try:
            # Get volume data
            volume_data = await self._get_volume_history(token, chain, days=7)
            
            if not volume_data:
                return {}
                
            df = pd.DataFrame(volume_data)
            df['volume'] = pd.to_numeric(df['volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate volume metrics
            metrics = {
                'average_volume': float(df['volume'].mean()),
                'median_volume': float(df['volume'].median()),
                'volume_std': float(df['volume'].std()),
                'current_volume': float(df['volume'].iloc[-1]) if len(df) > 0 else 0
            }
            
            # Detect volume spikes
            volume_spikes = self._detect_volume_spikes(df)
            
            # Analyze volume trend
            volume_trend = self._analyze_volume_trend(df)
            
            # Check for wash trading patterns
            wash_trading_score = await self._detect_wash_trading(df)
            
            # Volume profile analysis
            volume_profile = self._create_volume_profile(df)
            
            # Calculate relative volume
            relative_volume = metrics['current_volume'] / metrics['average_volume'] if metrics['average_volume'] > 0 else 0
            
            return {
                'token': token,
                'chain': chain,
                'metrics': metrics,
                'relative_volume': relative_volume,
                'volume_spikes': volume_spikes,
                'volume_trend': volume_trend,
                'wash_trading_score': wash_trading_score,
                'volume_profile': volume_profile,
                'analysis_timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns for {token}: {e}")
            return {}
            
    # Private helper methods
    
    async def _calculate_market_metrics(self, market_data: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate market metrics"""
        try:
            prices = [d['price'] for d in market_data]
            volumes = [d['volume'] for d in market_data]
            market_caps = [d.get('market_cap', 0) for d in market_data]
            
            # Price metrics
            price_changes = [d.get('price_change_24h', 0) for d in market_data]
            avg_price_change = np.mean(price_changes)
            
            # Volume metrics
            total_volume = sum(volumes)
            avg_volume = np.mean(volumes)
            
            # Market cap metrics
            total_market_cap = sum(market_caps)
            
            # Breadth indicators
            advancing = sum(1 for p in price_changes if p > 0)
            declining = sum(1 for p in price_changes if p < 0)
            advance_decline_ratio = advancing / max(declining, 1)
            
            # Volatility
            price_volatility = np.std(price_changes) if price_changes else 0
            
            return {
                'avg_price_change': avg_price_change,
                'total_volume': total_volume,
                'avg_volume': avg_volume,
                'total_market_cap': total_market_cap,
                'advance_decline_ratio': advance_decline_ratio,
                'price_volatility': price_volatility,
                'advancing': advancing,
                'declining': declining,
                'neutral': len(market_data) - advancing - declining
            }
            
        except Exception as e:
            logger.error(f"Error calculating market metrics: {e}")
            return {}
            
    async def _determine_market_trend(self, metrics: Dict) -> Dict[str, Any]:
        """Determine overall market trend"""
        trend_score = 0
        
        # Price change factor
        if metrics.get('avg_price_change', 0) > 2:
            trend_score += 2
        elif metrics.get('avg_price_change', 0) > 0:
            trend_score += 1
        elif metrics.get('avg_price_change', 0) < -2:
            trend_score -= 2
        elif metrics.get('avg_price_change', 0) < 0:
            trend_score -= 1
            
        # Breadth factor
        if metrics.get('advance_decline_ratio', 1) > 2:
            trend_score += 2
        elif metrics.get('advance_decline_ratio', 1) > 1.2:
            trend_score += 1
        elif metrics.get('advance_decline_ratio', 1) < 0.5:
            trend_score -= 2
        elif metrics.get('advance_decline_ratio', 1) < 0.8:
            trend_score -= 1
            
        # Determine direction
        if trend_score >= 2:
            direction = 'bullish'
        elif trend_score <= -2:
            direction = 'bearish'
        else:
            direction = 'neutral'
            
        # Calculate strength
        strength = min(abs(trend_score) / 4, 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'score': trend_score
        }
        
    async def _calculate_market_volatility(self, metrics: Dict) -> Dict[str, str]:
        """Calculate market volatility level"""
        volatility = metrics.get('price_volatility', 0)
        
        if volatility < 2:
            level = 'low'
        elif volatility < 5:
            level = 'medium'
        elif volatility < 10:
            level = 'high'
        else:
            level = 'extreme'
            
        return {
            'level': level,
            'value': volatility
        }
        
    async def _identify_correlation_clusters(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[List[str]]:
        """Identify clusters of highly correlated tokens"""
        clusters = []
        processed = set()
        
        for token1 in correlation_matrix.columns:
            if token1 in processed:
                continue
                
            cluster = [token1]
            processed.add(token1)
            
            for token2 in correlation_matrix.columns:
                if token2 != token1 and token2 not in processed:
                    if abs(correlation_matrix.loc[token1, token2]) >= threshold:
                        cluster.append(token2)
                        processed.add(token2)
                        
            if len(cluster) > 1:
                clusters.append(cluster)
                
        return clusters
        
    def _calculate_market_risk(
        self,
        volatility: Dict,
        liquidity: Dict,
        sentiment: Dict
    ) -> str:
        """Calculate overall market risk level"""
        risk_score = 0
        
        # Volatility component
        if volatility['level'] == 'extreme':
            risk_score += 3
        elif volatility['level'] == 'high':
            risk_score += 2
        elif volatility['level'] == 'medium':
            risk_score += 1
            
        # Liquidity component
        if liquidity['level'] == 'poor':
            risk_score += 3
        elif liquidity['level'] == 'fair':
            risk_score += 1
            
        # Sentiment component
        if sentiment['level'] == 'fearful':
            risk_score += 2
        elif sentiment['level'] == 'greedy':
            risk_score += 1
            
        # Determine risk level
        if risk_score >= 6:
            return 'extreme'
        elif risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
            
    def _detect_volume_spikes(self, df: pd.DataFrame) -> List[Dict]:
        """Detect unusual volume spikes"""
        spikes = []
        
        # Calculate rolling statistics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Detect spikes (volume > mean + 2*std)
        df['is_spike'] = df['volume'] > (df['volume_ma'] + 2 * df['volume_std'])
        
        spike_indices = df[df['is_spike']].index
        
        for idx in spike_indices:
            spikes.append({
                'timestamp': df.loc[idx, 'timestamp'],
                'volume': float(df.loc[idx, 'volume']),
                'magnitude': float((df.loc[idx, 'volume'] - df.loc[idx, 'volume_ma']) / df.loc[idx, 'volume_std'])
                if df.loc[idx, 'volume_std'] > 0 else 0
            })
            
        return spikes
        
    def _get_regime_recommendation(self, regime: str) -> str:
        """Get trading recommendation based on market regime"""
        recommendations = {
            'trending': "Use momentum strategies. Follow the trend with proper stops.",
            'ranging': "Use mean reversion strategies. Buy support, sell resistance.",
            'volatile': "Reduce position sizes. Use wider stops. Consider options.",
            'quiet': "Wait for clearer signals. Accumulation phase possible.",
            'unknown': "Exercise caution. Market conditions unclear."
        }
        
        return recommendations.get(regime, "Monitor closely before trading.")

    # ============================================================================
    # PATCH FOR: market_analyzer.py
    # Add these methods to the MarketAnalyzer class
    # ============================================================================

    async def analyze_market(self, chain: str = 'ethereum') -> Dict:
        """
        Analyze market conditions (wrapper for analyze_market_conditions)
        
        Args:
            chain: Blockchain network
            
        Returns:
            Market analysis data
        """
        # Use existing analyze_market_conditions
        condition = await self.analyze_market_conditions(tokens=None, chain=chain)
        
        # Convert to expected format
        return {
            'chain': chain,
            'trend': condition.trend,
            'volatility': condition.volatility,
            'liquidity': condition.liquidity,
            'sentiment': condition.sentiment,
            'momentum': condition.momentum,
            'strength': condition.strength,
            'risk_level': condition.risk_level,
            'market_phase': condition.market_phase,
            'confidence': condition.confidence,
            'timestamp': datetime.utcnow()
        }

    def calculate_market_sentiment(self) -> Dict:
        """
        Calculate market sentiment
        
        Returns:
            Market sentiment data
        """
        # Use cached analysis data if available
        if self._analysis_cache:
            # Aggregate sentiment from recent analyses
            sentiments = []
            for analysis in self._analysis_cache.values():
                if isinstance(analysis, dict) and 'sentiment' in analysis:
                    sentiments.append(analysis['sentiment'])
            
            if sentiments:
                # Calculate average sentiment
                avg_sentiment = sum(s for s in sentiments if isinstance(s, (int, float))) / len(sentiments)
                
                # Determine sentiment level
                if avg_sentiment > 0.6:
                    level = 'greedy'
                elif avg_sentiment > 0.4:
                    level = 'optimistic'
                elif avg_sentiment < -0.4:
                    level = 'fearful'
                elif avg_sentiment < -0.2:
                    level = 'pessimistic'
                else:
                    level = 'neutral'
                
                return {
                    'level': level,
                    'score': avg_sentiment,
                    'fear_greed_index': int((avg_sentiment + 1) * 50),  # Convert to 0-100 scale
                    'timestamp': datetime.utcnow()
                }
        
        # Default neutral sentiment
        return {
            'level': 'neutral',
            'score': 0.0,
            'fear_greed_index': 50,
            'timestamp': datetime.utcnow()
        }

    def identify_market_trends(self) -> List[str]:
        """
        Identify current market trends
        
        Returns:
            List of identified trends
        """
        trends = []
        
        # Analyze cached data for trends
        if self._analysis_cache:
            # Count trend occurrences
            trend_counts = defaultdict(int)
            
            for analysis in self._analysis_cache.values():
                if isinstance(analysis, dict):
                    if 'trend' in analysis:
                        trend_counts[analysis['trend']] += 1
                    if 'market_phase' in analysis:
                        trend_counts[analysis['market_phase']] += 1
            
            # Identify dominant trends
            for trend, count in trend_counts.items():
                if count >= 2:  # Threshold for trend identification
                    trends.append(trend)
        
        # Add time-based trends
        current_hour = datetime.utcnow().hour
        if 14 <= current_hour <= 20:  # US market hours (UTC)
            trends.append('us_trading_hours')
        elif 1 <= current_hour <= 7:  # Asian market hours (UTC)
            trends.append('asian_trading_hours')
        
        # Default trends if none identified
        if not trends:
            trends = ['neutral', 'consolidating']
        
        return trends

    async def get_market_indicators(self) -> Dict:
        """
        Get key market indicators
        
        Returns:
            Market indicators dictionary
        """
        indicators = {
            'timestamp': datetime.utcnow(),
            'indicators': {}
        }
        
        try:
            # Get market conditions
            conditions = await self.analyze_market_conditions()
            
            # Extract indicators
            indicators['indicators'] = {
                'trend_strength': conditions.strength,
                'momentum': conditions.momentum,
                'volatility': conditions.volatility,
                'risk_level': conditions.risk_level,
                'market_phase': conditions.market_phase
            }
            
            # Add calculated indicators
            sentiment = self.calculate_market_sentiment()
            indicators['indicators']['sentiment'] = sentiment['level']
            indicators['indicators']['fear_greed_index'] = sentiment['fear_greed_index']
            
            # Add identified trends
            trends = self.identify_market_trends()
            indicators['indicators']['active_trends'] = trends
            
        except Exception as e:
            logger.error(f"Error getting market indicators: {e}")
            
        return indicators
