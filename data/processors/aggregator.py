"""
Data Aggregator - Aggregates data from multiple sources for ClaudeDex Trading Bot
Combines and reconciles data from various APIs and sources
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from loguru import logger


class DataAggregator:
    """
    Aggregates and reconciles data from multiple sources
    Handles conflicts, validates consistency, and provides unified view
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Source priorities (higher number = higher priority)
        self.source_priorities = {
            'dexscreener': 10,
            'coingecko': 9,
            'coinmarketcap': 8,
            '1inch': 7,
            'uniswap': 6,
            'pancakeswap': 5,
            'sushiswap': 4,
            'default': 1
        }
        
        # Source reliability scores (track historical accuracy)
        self.source_reliability = defaultdict(lambda: 1.0)
        
        # Aggregation methods
        self.aggregation_methods = {
            'price': 'weighted_median',
            'volume': 'sum',
            'liquidity': 'max',
            'holders': 'max',
            'market_cap': 'weighted_average',
            'social_metrics': 'average'
        }
        
        # Data cache with TTL
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 60)  # seconds
        
        # Conflict resolution history
        self.conflict_history = deque(maxlen=1000)
        
        # Quality metrics
        self.data_quality_scores = {}
        
        logger.info("DataAggregator initialized")
    
    async def aggregate_token_data(
        self,
        token_address: str,
        chain: str,
        data_sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate token data from multiple sources
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            data_sources: Dictionary mapping source names to their data
            
        Returns:
            Aggregated and reconciled data
        """
        try:
            # Check cache
            cache_key = f"{chain}:{token_address}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_ttl:
                    return cached_data
            
            # Validate and clean data from each source
            cleaned_sources = {}
            for source_name, source_data in data_sources.items():
                cleaned_data = self._clean_source_data(source_data)
                if self._validate_source_data(cleaned_data):
                    cleaned_sources[source_name] = cleaned_data
                else:
                    logger.warning(f"Invalid data from source {source_name}")
            
            if not cleaned_sources:
                logger.error("No valid data sources available")
                return {}
            
            # Aggregate different data types
            aggregated = {
                'token_address': token_address,
                'chain': chain,
                'timestamp': datetime.now(),
                'sources_count': len(cleaned_sources),
                'sources': list(cleaned_sources.keys())
            }
            
            # Price aggregation
            price_data = self._aggregate_prices(cleaned_sources)
            aggregated.update(price_data)
            
            # Volume aggregation
            volume_data = self._aggregate_volumes(cleaned_sources)
            aggregated.update(volume_data)
            
            # Liquidity aggregation
            liquidity_data = self._aggregate_liquidity(cleaned_sources)
            aggregated.update(liquidity_data)
            
            # Holder metrics aggregation
            holder_data = self._aggregate_holder_metrics(cleaned_sources)
            aggregated.update(holder_data)
            
            # Social metrics aggregation
            social_data = self._aggregate_social_metrics(cleaned_sources)
            aggregated.update(social_data)
            
            # Technical indicators aggregation
            technical_data = self._aggregate_technical_indicators(cleaned_sources)
            aggregated.update(technical_data)
            
            # Calculate data quality score
            aggregated['data_quality_score'] = self._calculate_quality_score(cleaned_sources)
            
            # Detect and flag anomalies
            anomalies = self._detect_anomalies(aggregated, cleaned_sources)
            aggregated['anomalies'] = anomalies
            
            # Update cache
            self.cache[cache_key] = (aggregated, datetime.now())
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data for {token_address}: {e}")
            return {}
    
    def _clean_source_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize data from a source"""
        cleaned = {}
        
        for key, value in data.items():
            # Remove None values
            if value is None:
                continue
            
            # Standardize numeric values
            if key in ['price', 'volume', 'liquidity', 'market_cap']:
                try:
                    cleaned[key] = float(value) if value else 0
                except (ValueError, TypeError):
                    continue
            
            # Standardize percentage values
            elif key.endswith('_percent') or key.endswith('_percentage'):
                try:
                    val = float(value)
                    # Convert to 0-100 range if needed
                    if -1 <= val <= 1:
                        val = val * 100
                    cleaned[key] = val
                except (ValueError, TypeError):
                    continue
            
            # Standardize counts
            elif key in ['holders', 'transactions', 'transfers']:
                try:
                    cleaned[key] = int(value) if value else 0
                except (ValueError, TypeError):
                    continue
            
            # Keep other values as-is
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _validate_source_data(self, data: Dict[str, Any]) -> bool:
        """Validate data from a source"""
        # Must have at least price
        if 'price' not in data or data['price'] <= 0:
            return False
        
        # Check for suspicious values
        if 'volume' in data and data['volume'] < 0:
            return False
        
        if 'liquidity' in data and data['liquidity'] < 0:
            return False
        
        if 'holders' in data and data['holders'] < 0:
            return False
        
        return True
    
    def _aggregate_prices(self, sources: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate price data from multiple sources"""
        prices = []
        weights = []
        
        for source_name, data in sources.items():
            if 'price' in data:
                price = data['price']
                weight = self.source_priorities.get(source_name, 1) * self.source_reliability[source_name]
                prices.append(price)
                weights.append(weight)
        
        if not prices:
            return {}
        
        # Weighted median
        sorted_pairs = sorted(zip(prices, weights))
        cumsum = np.cumsum([w for _, w in sorted_pairs])
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        
        result = {
            'price': sorted_pairs[median_idx][0],
            'price_min': min(prices),
            'price_max': max(prices),
            'price_std': statistics.stdev(prices) if len(prices) > 1 else 0,
            'price_sources': len(prices)
        }
        
        # Detect price discrepancies
        if result['price_std'] / result['price'] > 0.05:  # >5% deviation
            result['price_discrepancy'] = True
            self._record_conflict('price', sources)
        
        return result
    
    def _aggregate_volumes(self, sources: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate volume data from multiple sources"""
        volumes = []
        
        for source_name, data in sources.items():
            if 'volume' in data or 'volume_24h' in data:
                volume = data.get('volume', data.get('volume_24h', 0))
                volumes.append(volume)
        
        if not volumes:
            return {}
        
        # Sum volumes (assuming different DEXes)
        total_volume = sum(volumes)
        
        return {
            'volume_24h': total_volume,
            'volume_avg_per_source': total_volume / len(volumes),
            'volume_sources': len(volumes)
        }
    
    def _aggregate_liquidity(self, sources: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate liquidity data from multiple sources"""
        liquidities = []
        locked_percentages = []
        
        for source_name, data in sources.items():
            if 'liquidity' in data:
                liquidities.append(data['liquidity'])
            
            if 'liquidity_locked' in data or 'locked_percent' in data:
                locked = data.get('liquidity_locked', data.get('locked_percent', 0))
                locked_percentages.append(locked)
        
        if not liquidities:
            return {}
        
        result = {
            'liquidity': max(liquidities),  # Use maximum as most accurate
            'liquidity_min': min(liquidities),
            'liquidity_sources': len(liquidities)
        }
        
        if locked_percentages:
            result['liquidity_locked_percent'] = statistics.mean(locked_percentages)
        
        return result
    
    def _aggregate_holder_metrics(self, sources: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate holder-related metrics"""
        holders_counts = []
        whale_percentages = []
        
        for source_name, data in sources.items():
            if 'holders' in data:
                holders_counts.append(data['holders'])
            
            if 'whale_percent' in data or 'top10_percent' in data:
                whale = data.get('whale_percent', data.get('top10_percent', 0))
                whale_percentages.append(whale)
        
        if not holders_counts:
            return {}
        
        result = {
            'holders': max(holders_counts),  # Use maximum
            'holders_min': min(holders_counts),
            'holders_sources': len(holders_counts)
        }
        
        if whale_percentages:
            result['whale_concentration'] = statistics.mean(whale_percentages)
        
        return result
    
    def _aggregate_social_metrics(self, sources: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate social metrics from multiple sources"""
        social_scores = []
        sentiments = []
        
        for source_name, data in sources.items():
            if 'social_score' in data:
                social_scores.append(data['social_score'])
            
            if 'sentiment' in data or 'social_sentiment' in data:
                sentiment = data.get('sentiment', data.get('social_sentiment', 0.5))
                sentiments.append(sentiment)
        
        result = {}
        
        if social_scores:
            result['social_score'] = statistics.mean(social_scores)
        
        if sentiments:
            result['social_sentiment'] = statistics.mean(sentiments)
        
        return result
    
    def _aggregate_technical_indicators(self, sources: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate technical indicators"""
        indicators = defaultdict(list)
        
        # Common indicators to aggregate
        indicator_names = ['rsi', 'macd', 'volume_ratio', 'volatility']
        
        for source_name, data in sources.items():
            for indicator in indicator_names:
                if indicator in data:
                    indicators[indicator].append(data[indicator])
        
        result = {}
        for indicator, values in indicators.items():
            if values:
                result[indicator] = statistics.mean(values)
        
        return result
    
    def _calculate_quality_score(self, sources: Dict[str, Dict]) -> float:
        """Calculate data quality score"""
        scores = []
        
        # Score based on number of sources
        source_score = min(len(sources) / 5, 1.0)  # Max score at 5+ sources
        scores.append(source_score * 0.3)
        
        # Score based on data completeness
        required_fields = ['price', 'volume', 'liquidity', 'holders']
        completeness_scores = []
        
        for source_name, data in sources.items():
            fields_present = sum(1 for field in required_fields if field in data)
            completeness = fields_present / len(required_fields)
            completeness_scores.append(completeness)
        
        avg_completeness = statistics.mean(completeness_scores) if completeness_scores else 0
        scores.append(avg_completeness * 0.4)
        
        # Score based on data consistency
        price_values = [d.get('price', 0) for d in sources.values() if 'price' in d]
        if len(price_values) > 1:
            cv = statistics.stdev(price_values) / statistics.mean(price_values)
            consistency_score = max(0, 1 - cv)  # Lower CV = higher score
        else:
            consistency_score = 0.5
        scores.append(consistency_score * 0.3)
        
        return sum(scores)
    
    def _detect_anomalies(self, aggregated: Dict, sources: Dict[str, Dict]) -> List[str]:
        """Detect anomalies in aggregated data"""
        anomalies = []
        
        # Price anomalies
        if 'price_std' in aggregated and 'price' in aggregated:
            if aggregated['price_std'] / aggregated['price'] > 0.1:
                anomalies.append("High price variance across sources")
        
        # Volume anomalies
        if 'volume_24h' in aggregated:
            if aggregated['volume_24h'] < 1000:
                anomalies.append("Suspiciously low volume")
        
        # Liquidity anomalies
        if 'liquidity' in aggregated:
            if aggregated['liquidity'] < 10000:
                anomalies.append("Low liquidity warning")
        
        # Holder anomalies
        if 'holders' in aggregated:
            if aggregated['holders'] < 50:
                anomalies.append("Very few holders")
        
        # Data source anomalies
        if len(sources) == 1:
            anomalies.append("Single source only - limited validation")
        
        return anomalies
    
    def _record_conflict(self, field: str, sources: Dict[str, Dict]) -> None:
        """Record a data conflict for analysis"""
        conflict = {
            'timestamp': datetime.now(),
            'field': field,
            'values': {source: data.get(field) for source, data in sources.items()}
        }
        self.conflict_history.append(conflict)
    
    async def aggregate_market_data(
        self,
        chain: str,
        data_sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate market-wide data"""
        aggregated = {
            'chain': chain,
            'timestamp': datetime.now(),
            'sources': list(data_sources.keys())
        }
        
        # Aggregate market metrics
        total_volumes = []
        total_liquidity = []
        active_pairs = []
        
        for source_name, data in data_sources.items():
            if 'total_volume' in data:
                total_volumes.append(data['total_volume'])
            if 'total_liquidity' in data:
                total_liquidity.append(data['total_liquidity'])
            if 'active_pairs' in data:
                active_pairs.append(data['active_pairs'])
        
        if total_volumes:
            aggregated['total_volume_24h'] = sum(total_volumes)
        
        if total_liquidity:
            aggregated['total_liquidity'] = sum(total_liquidity)
        
        if active_pairs:
            aggregated['active_pairs'] = max(active_pairs)
        
        return aggregated
    
    def merge_time_series(
        self,
        series_list: List[pd.DataFrame],
        method: str = 'average'
    ) -> pd.DataFrame:
        """
        Merge multiple time series data
        
        Args:
            series_list: List of DataFrames with time series
            method: Aggregation method ('average', 'median', 'max', 'min')
            
        Returns:
            Merged DataFrame
        """
        if not series_list:
            return pd.DataFrame()
        
        # Align all series to same time index
        merged = pd.concat(series_list, axis=1, join='outer')
        
        # Apply aggregation method
        if method == 'average':
            result = merged.mean(axis=1)
        elif method == 'median':
            result = merged.median(axis=1)
        elif method == 'max':
            result = merged.max(axis=1)
        elif method == 'min':
            result = merged.min(axis=1)
        else:
            result = merged.mean(axis=1)
        
        return pd.DataFrame(result, columns=['value'])
    
    def update_source_reliability(
        self,
        source: str,
        accuracy_score: float
    ) -> None:
        """
        Update reliability score for a data source
        
        Args:
            source: Source name
            accuracy_score: Accuracy score (0-1)
        """
        # Exponential moving average
        alpha = 0.1
        old_score = self.source_reliability[source]
        new_score = alpha * accuracy_score + (1 - alpha) * old_score
        self.source_reliability[source] = new_score
    
    def get_reliability_report(self) -> Dict[str, float]:
        """Get reliability scores for all sources"""
        return dict(self.source_reliability)
    
    def get_conflict_report(self) -> Dict[str, Any]:
        """Get report of recent data conflicts"""
        if not self.conflict_history:
            return {'conflicts': [], 'total': 0}
        
        # Group conflicts by field
        conflicts_by_field = defaultdict(list)
        for conflict in self.conflict_history:
            conflicts_by_field[conflict['field']].append(conflict)
        
        return {
            'conflicts': dict(conflicts_by_field),
            'total': len(self.conflict_history),
            'fields_affected': list(conflicts_by_field.keys())
        }