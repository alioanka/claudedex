"""
Feature Extractor - ML feature engineering for ClaudeDex Trading Bot
Extracts and engineers features for machine learning models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from scipy import stats
import talib
from loguru import logger

# Conditionally import scikit-learn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False
    # Mock scalers if scikit-learn is not installed
    class StandardScaler:
        def fit_transform(self, data):
            return data
        def transform(self, data):
            return data
    class MinMaxScaler:
        def fit_transform(self, data):
            return data
        def transform(self, data):
            return data


class FeatureExtractor:
    """
    Extracts and engineers features for ML models
    Generates 150+ features from raw market data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Feature configuration
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20, 50, 100])
        self.volume_periods = self.config.get('volume_periods', [24, 48, 168])  # hours
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_periods = self.config.get('macd_periods', (12, 26, 9))
        
        # Scalers for normalization
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.indicator_scaler = MinMaxScaler()
        
        # Feature groups
        self.feature_groups = {
            'price': [],
            'volume': [],
            'technical': [],
            'pattern': [],
            'market': [],
            'social': [],
            'chain': [],
            'risk': []
        }
        
        logger.info("FeatureExtractor initialized")
    
    def extract_all_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features from market data
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Extract different feature groups
        features.update(self.extract_price_features(data))
        features.update(self.extract_volume_features(data))
        features.update(self.extract_technical_indicators(data))
        features.update(self.extract_pattern_features(data))
        features.update(self.extract_market_features(data))
        features.update(self.extract_social_features(data))
        features.update(self.extract_chain_features(data))
        features.update(self.extract_risk_features(data))
        
        # Add interaction features
        features.update(self.extract_interaction_features(features))
        
        # Add derived features
        features.update(self.extract_derived_features(features))
        
        return features
    
    def extract_price_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract price-related features"""
        features = {}
        
        prices = data.get('prices', [])
        if not prices:
            return features
        
        prices = np.array([float(p) for p in prices])
        current_price = prices[-1] if len(prices) > 0 else 0
        
        # Basic price features
        features['price_current'] = current_price
        features['price_mean'] = np.mean(prices)
        features['price_std'] = np.std(prices)
        features['price_min'] = np.min(prices)
        features['price_max'] = np.max(prices)
        features['price_range'] = features['price_max'] - features['price_min']
        
        # Price changes over different periods
        for period in self.lookback_periods:
            if len(prices) > period:
                features[f'price_change_{period}'] = (prices[-1] - prices[-period]) / prices[-period]
                features[f'price_volatility_{period}'] = np.std(prices[-period:])
                features[f'price_mean_{period}'] = np.mean(prices[-period:])
        
        # Price position metrics
        if features['price_range'] > 0:
            features['price_position'] = (current_price - features['price_min']) / features['price_range']
        else:
            features['price_position'] = 0.5
        
        # Price momentum
        if len(prices) > 1:
            features['price_momentum'] = prices[-1] - prices[-2]
            features['price_acceleration'] = 0
            if len(prices) > 2:
                momentum_1 = prices[-1] - prices[-2]
                momentum_2 = prices[-2] - prices[-3]
                features['price_acceleration'] = momentum_1 - momentum_2
        
        # Log returns
        if len(prices) > 1:
            log_returns = np.log(prices[1:] / prices[:-1])
            features['log_return_mean'] = np.mean(log_returns)
            features['log_return_std'] = np.std(log_returns)
            features['log_return_skew'] = stats.skew(log_returns)
            features['log_return_kurt'] = stats.kurtosis(log_returns)
        
        # Price efficiency ratio
        if len(prices) > 10:
            direction = abs(prices[-1] - prices[-10])
            volatility = sum(abs(prices[i] - prices[i-1]) for i in range(-9, 0))
            features['price_efficiency'] = direction / volatility if volatility > 0 else 0
        
        return features
    
    def extract_volume_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract volume-related features"""
        features = {}
        
        volumes = data.get('volumes', [])
        if not volumes:
            return features
        
        volumes = np.array([float(v) for v in volumes])
        
        # Basic volume features
        features['volume_current'] = volumes[-1] if len(volumes) > 0 else 0
        features['volume_mean'] = np.mean(volumes)
        features['volume_std'] = np.std(volumes)
        features['volume_total'] = np.sum(volumes)
        
        # Volume changes over periods
        for period in self.lookback_periods:
            if len(volumes) > period:
                features[f'volume_change_{period}'] = (volumes[-1] - volumes[-period]) / (volumes[-period] + 1)
                features[f'volume_mean_{period}'] = np.mean(volumes[-period:])
                features[f'volume_std_{period}'] = np.std(volumes[-period:])
        
        # Volume patterns
        if len(volumes) > 1:
            features['volume_trend'] = np.polyfit(range(len(volumes)), volumes, 1)[0]
            features['volume_acceleration'] = 0
            if len(volumes) > 2:
                vol_change_1 = volumes[-1] - volumes[-2]
                vol_change_2 = volumes[-2] - volumes[-3]
                features['volume_acceleration'] = vol_change_1 - vol_change_2
        
        # Volume distribution
        features['volume_skew'] = stats.skew(volumes)
        features['volume_kurt'] = stats.kurtosis(volumes)
        
        # Volume concentration
        if len(volumes) > 24:  # Last 24 periods
            last_24 = volumes[-24:]
            features['volume_concentration_1h'] = last_24[-1] / (np.sum(last_24) + 1)
            features['volume_concentration_6h'] = np.sum(last_24[-6:]) / (np.sum(last_24) + 1)
        
        # Volume spikes
        if len(volumes) > 10:
            mean_vol = np.mean(volumes[-10:])
            std_vol = np.std(volumes[-10:])
            features['volume_spike'] = (volumes[-1] - mean_vol) / (std_vol + 1)
        
        # On-Balance Volume (OBV) trend
        if 'prices' in data and len(data['prices']) == len(volumes):
            prices = np.array([float(p) for p in data['prices']])
            obv = np.zeros(len(prices))
            obv[0] = volumes[0]
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif prices[i] < prices[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            
            features['obv_current'] = obv[-1]
            if len(obv) > 1:
                features['obv_trend'] = np.polyfit(range(len(obv)), obv, 1)[0]
        
        return features
    
    def extract_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract technical analysis indicators"""
        features = {}
        
        prices = data.get('prices', [])
        if len(prices) < 2:
            return features
        
        prices = np.array([float(p) for p in prices], dtype=np.float64)
        high = np.array(data.get('high', prices), dtype=np.float64)
        low = np.array(data.get('low', prices), dtype=np.float64)
        close = prices
        volume = np.array([float(v) for v in data.get('volumes', [0]*len(prices))], dtype=np.float64)
        
        # RSI
        if len(close) >= self.rsi_period:
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            features['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
            features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0
            features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
        
        # MACD
        if len(close) >= self.macd_periods[1]:
            macd, signal, hist = talib.MACD(close, 
                                           fastperiod=self.macd_periods[0],
                                           slowperiod=self.macd_periods[1], 
                                           signalperiod=self.macd_periods[2])
            if not np.isnan(macd[-1]):
                features['macd'] = macd[-1]
                features['macd_signal'] = signal[-1]
                features['macd_histogram'] = hist[-1]
                features['macd_cross'] = 1 if macd[-1] > signal[-1] and macd[-2] <= signal[-2] else 0
        
        # Bollinger Bands
        if len(close) >= 20:
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            if not np.isnan(upper[-1]):
                features['bb_upper'] = upper[-1]
                features['bb_middle'] = middle[-1]
                features['bb_lower'] = lower[-1]
                features['bb_width'] = upper[-1] - lower[-1]
                features['bb_position'] = (close[-1] - lower[-1]) / (upper[-1] - lower[-1] + 0.0001)
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                sma = talib.SMA(close, timeperiod=period)
                ema = talib.EMA(close, timeperiod=period)
                if not np.isnan(sma[-1]):
                    features[f'sma_{period}'] = sma[-1]
                    features[f'price_to_sma_{period}'] = close[-1] / sma[-1]
                if not np.isnan(ema[-1]):
                    features[f'ema_{period}'] = ema[-1]
                    features[f'price_to_ema_{period}'] = close[-1] / ema[-1]
        
        # Stochastic Oscillator
        if len(high) >= 14 and len(low) >= 14:
            slowk, slowd = talib.STOCH(high, low, close)
            if not np.isnan(slowk[-1]):
                features['stoch_k'] = slowk[-1]
                features['stoch_d'] = slowd[-1]
                features['stoch_oversold'] = 1 if slowk[-1] < 20 else 0
                features['stoch_overbought'] = 1 if slowk[-1] > 80 else 0
        
        # ADX (Average Directional Index)
        if len(high) >= 14 and len(low) >= 14:
            adx = talib.ADX(high, low, close, timeperiod=14)
            if not np.isnan(adx[-1]):
                features['adx'] = adx[-1]
                features['trend_strength'] = 'strong' if adx[-1] > 25 else 'weak'
        
        # ATR (Average True Range)
        if len(high) >= 14 and len(low) >= 14:
            atr = talib.ATR(high, low, close, timeperiod=14)
            if not np.isnan(atr[-1]):
                features['atr'] = atr[-1]
                features['volatility_atr'] = atr[-1] / close[-1] if close[-1] > 0 else 0
        
        # CCI (Commodity Channel Index)
        if len(high) >= 20 and len(low) >= 20:
            cci = talib.CCI(high, low, close, timeperiod=20)
            if not np.isnan(cci[-1]):
                features['cci'] = cci[-1]
                features['cci_overbought'] = 1 if cci[-1] > 100 else 0
                features['cci_oversold'] = 1 if cci[-1] < -100 else 0
        
        # MFI (Money Flow Index)
        if len(high) >= 14 and len(low) >= 14 and len(volume) >= 14:
            mfi = talib.MFI(high, low, close, volume, timeperiod=14)
            if not np.isnan(mfi[-1]):
                features['mfi'] = mfi[-1]
                features['mfi_overbought'] = 1 if mfi[-1] > 80 else 0
                features['mfi_oversold'] = 1 if mfi[-1] < 20 else 0
        
        # Williams %R
        if len(high) >= 14 and len(low) >= 14:
            willr = talib.WILLR(high, low, close, timeperiod=14)
            if not np.isnan(willr[-1]):
                features['williams_r'] = willr[-1]
        
        # ROC (Rate of Change)
        if len(close) >= 10:
            roc = talib.ROC(close, timeperiod=10)
            if not np.isnan(roc[-1]):
                features['roc'] = roc[-1]
        
        return features
    
    def extract_pattern_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract chart pattern features"""
        features = {}
        
        prices = data.get('prices', [])
        if len(prices) < 10:
            return features
        
        prices = np.array([float(p) for p in prices])
        
        # Trend detection
        if len(prices) >= 20:
            trend = np.polyfit(range(len(prices)), prices, 1)[0]
            features['trend_slope'] = trend
            features['trend_direction'] = 1 if trend > 0 else -1
            
            # R-squared for trend strength
            y_pred = np.polyval([trend, np.mean(prices)], range(len(prices)))
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            features['trend_r2'] = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Support and resistance levels
        if len(prices) >= 50:
            # Find local maxima and minima
            from scipy.signal import argrelextrema
            local_maxima = argrelextrema(prices, np.greater, order=5)[0]
            local_minima = argrelextrema(prices, np.less, order=5)[0]
            
            if len(local_maxima) > 0:
                resistance_levels = prices[local_maxima]
                features['resistance_nearest'] = min(resistance_levels[resistance_levels > prices[-1]], 
                                                    default=prices[-1] * 1.1)
                features['resistance_count'] = len(resistance_levels)
            
            if len(local_minima) > 0:
                support_levels = prices[local_minima]
                features['support_nearest'] = max(support_levels[support_levels < prices[-1]], 
                                                 default=prices[-1] * 0.9)
                features['support_count'] = len(support_levels)
        
        # Pattern recognition flags
        features['pattern_double_top'] = self._detect_double_top(prices)
        features['pattern_double_bottom'] = self._detect_double_bottom(prices)
        features['pattern_head_shoulders'] = self._detect_head_shoulders(prices)
        features['pattern_triangle'] = self._detect_triangle(prices)
        features['pattern_flag'] = self._detect_flag(prices)
        
        # Candlestick patterns (if OHLC data available)
        if all(k in data for k in ['open', 'high', 'low', 'close']):
            features['candle_doji'] = self._detect_doji(data)
            features['candle_hammer'] = self._detect_hammer(data)
            features['candle_engulfing'] = self._detect_engulfing(data)
        
        return features
    
    def extract_market_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market-wide features"""
        features = {}
        
        # Market sentiment
        features['market_fear_greed'] = data.get('fear_greed_index', 50)
        features['market_dominance_btc'] = data.get('btc_dominance', 40)
        features['market_dominance_eth'] = data.get('eth_dominance', 20)
        
        # Market metrics
        features['market_cap_total'] = float(data.get('total_market_cap', 0))
        features['market_volume_24h'] = float(data.get('total_volume_24h', 0))
        features['market_cap_change_24h'] = data.get('market_cap_change_24h', 0)
        
        # Correlation with major assets
        features['correlation_btc'] = data.get('btc_correlation', 0)
        features['correlation_eth'] = data.get('eth_correlation', 0)
        features['correlation_sp500'] = data.get('sp500_correlation', 0)
        
        # Market phase
        market_phase = data.get('market_phase', 'neutral')
        features['market_bull'] = 1 if market_phase == 'bull' else 0
        features['market_bear'] = 1 if market_phase == 'bear' else 0
        features['market_neutral'] = 1 if market_phase == 'neutral' else 0
        
        # DEX metrics
        features['dex_volume_total'] = float(data.get('dex_volume_total', 0))
        features['dex_trades_count'] = data.get('dex_trades_count', 0)
        features['dex_unique_traders'] = data.get('dex_unique_traders', 0)
        
        # Gas prices (for timing)
        features['gas_price_gwei'] = data.get('gas_price', 20)
        features['gas_price_high'] = 1 if features['gas_price_gwei'] > 100 else 0
        
        return features
    
    def extract_social_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract social sentiment features"""
        features = {}
        
        # Social metrics
        features['social_volume'] = data.get('social_volume', 0)
        features['social_sentiment'] = data.get('social_sentiment', 0.5)
        features['social_sentiment_positive'] = data.get('positive_mentions', 0)
        features['social_sentiment_negative'] = data.get('negative_mentions', 0)
        
        # Platform-specific metrics
        features['twitter_followers'] = data.get('twitter_followers', 0)
        features['twitter_mentions_24h'] = data.get('twitter_mentions', 0)
        features['reddit_subscribers'] = data.get('reddit_subscribers', 0)
        features['reddit_active_users'] = data.get('reddit_active', 0)
        features['telegram_members'] = data.get('telegram_members', 0)
        
        # Influencer metrics
        features['influencer_mentions'] = data.get('influencer_mentions', 0)
        features['influencer_sentiment'] = data.get('influencer_sentiment', 0.5)
        
        # Social velocity
        features['social_growth_rate'] = data.get('social_growth_rate', 0)
        features['social_engagement_rate'] = data.get('engagement_rate', 0)
        
        # FOMO indicators
        features['fomo_index'] = data.get('fomo_index', 50)
        features['trending_score'] = data.get('trending_score', 0)
        
        return features
    
    def extract_chain_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract blockchain-specific features"""
        features = {}
        
        # Token metrics
        features['holders_count'] = data.get('holders', 0)
        features['holders_change_24h'] = data.get('holders_change_24h', 0)
        features['whale_concentration'] = data.get('whale_concentration', 0)
        features['top10_holding_percent'] = data.get('top10_percent', 0)
        
        # Transaction metrics
        features['tx_count_24h'] = data.get('transactions_24h', 0)
        features['tx_volume_24h'] = float(data.get('volume_24h', 0))
        features['avg_tx_size'] = float(data.get('avg_transaction', 0))
        features['active_addresses_24h'] = data.get('active_addresses', 0)
        
        # Liquidity metrics
        features['liquidity_total'] = float(data.get('liquidity', 0))
        features['liquidity_locked_percent'] = data.get('locked_liquidity_percent', 0)
        features['liquidity_change_24h'] = data.get('liquidity_change_24h', 0)
        
        # Contract features
        features['contract_age_days'] = data.get('contract_age', 0)
        features['contract_verified'] = 1 if data.get('verified', False) else 0
        features['contract_renounced'] = 1 if data.get('renounced', False) else 0
        features['has_mint_function'] = 1 if data.get('mintable', False) else 0
        features['has_pause_function'] = 1 if data.get('pausable', False) else 0
        
        # DEX distribution
        features['dex_count'] = data.get('dex_count', 1)
        features['primary_dex_dominance'] = data.get('primary_dex_percent', 100)
        
        return features
    
    def extract_risk_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk-related features"""
        features = {}
        
        # Risk scores
        features['risk_score_overall'] = data.get('risk_score', 50)
        features['risk_rug_pull'] = data.get('rug_risk', 0)
        features['risk_honeypot'] = data.get('honeypot_risk', 0)
        features['risk_liquidity'] = data.get('liquidity_risk', 0)
        
        # Security flags
        features['is_honeypot'] = 1 if data.get('is_honeypot', False) else 0
        features['has_trading_cooldown'] = 1 if data.get('trading_cooldown', False) else 0
        features['has_max_tx_amount'] = 1 if data.get('max_tx_amount', False) else 0
        features['has_blacklist'] = 1 if data.get('blacklist', False) else 0
        
        # Tax features
        features['buy_tax'] = data.get('buy_tax', 0)
        features['sell_tax'] = data.get('sell_tax', 0)
        features['total_tax'] = features['buy_tax'] + features['sell_tax']
        features['high_tax'] = 1 if features['total_tax'] > 10 else 0
        
        # Developer features
        features['dev_wallet_percent'] = data.get('dev_holdings', 0)
        features['dev_activity_score'] = data.get('dev_activity', 0)
        features['dev_reputation'] = data.get('dev_reputation', 0)
        
        # Audit features
        features['is_audited'] = 1 if data.get('audited', False) else 0
        features['audit_score'] = data.get('audit_score', 0)
        
        return features
    
    def extract_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract interaction features between existing features"""
        interaction_features = {}
        
        # Price-Volume interactions
        if 'price_current' in features and 'volume_current' in features:
            interaction_features['price_volume_product'] = features['price_current'] * features['volume_current']
            
        if 'price_change_5' in features and 'volume_change_5' in features:
            interaction_features['price_volume_correlation'] = features['price_change_5'] * features['volume_change_5']
        
        # Technical indicator combinations
        if 'rsi' in features and 'macd_histogram' in features:
            interaction_features['rsi_macd_signal'] = features['rsi'] * features['macd_histogram']
        
        if 'adx' in features and 'trend_slope' in features:
            interaction_features['trend_strength_combined'] = features['adx'] * abs(features.get('trend_slope', 0))
        
        # Risk-Liquidity interaction
        if 'risk_score_overall' in features and 'liquidity_total' in features:
            interaction_features['risk_adjusted_liquidity'] = features['liquidity_total'] / (features['risk_score_overall'] + 1)
        
        # Social-Price interaction
        if 'social_sentiment' in features and 'price_change_24' in features:
            interaction_features['sentiment_price_alignment'] = features['social_sentiment'] * features.get('price_change_24', 0)
        
        return interaction_features
    
    def extract_derived_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extract derived features from existing features"""
        derived = {}
        
        # Composite risk score
        risk_components = ['risk_rug_pull', 'risk_honeypot', 'risk_liquidity', 'high_tax', 'dev_wallet_percent']
        risk_sum = sum(features.get(component, 0) for component in risk_components)
        derived['composite_risk'] = risk_sum / len(risk_components)
        
        # Momentum score
        momentum_components = ['price_change_5', 'price_change_10', 'volume_change_5', 'volume_change_10']
        momentum_sum = sum(features.get(component, 0) for component in momentum_components if component in features)
        derived['momentum_score'] = momentum_sum / 4
        
        # Technical score
        bullish_signals = sum([
            features.get('rsi_oversold', 0),
            features.get('macd_cross', 0),
            features.get('stoch_oversold', 0),
            1 if features.get('price_current', 0) > features.get('sma_20', 1) else 0
        ])
        derived['technical_score'] = bullish_signals / 4
        
        # Liquidity quality
        if features.get('liquidity_total', 0) > 0:
            derived['liquidity_quality'] = (
                features.get('liquidity_locked_percent', 0) / 100 * 0.5 +
                (1 - features.get('whale_concentration', 0) / 100) * 0.3 +
                min(features.get('holders_count', 0) / 1000, 1) * 0.2
            )
        else:
            derived['liquidity_quality'] = 0
        
        # Trading opportunity score
        derived['opportunity_score'] = (
            derived['technical_score'] * 0.3 +
            derived['momentum_score'] * 0.3 +
            (1 - derived['composite_risk']) * 0.2 +
            derived['liquidity_quality'] * 0.2
        )
        
        return derived
    
    def _detect_double_top(self, prices: np.ndarray) -> int:
        """Detect double top pattern"""
        if len(prices) < 20:
            return 0
        
        # Simplified detection: look for two similar peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prices, distance=5)
        
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            peak_values = prices[last_two_peaks]
            if abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.03:  # Within 3%
                return 1
        return 0
    
    def _detect_double_bottom(self, prices: np.ndarray) -> int:
        """Detect double bottom pattern"""
        if len(prices) < 20:
            return 0
        
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-prices, distance=5)
        
        if len(valleys) >= 2:
            last_two_valleys = valleys[-2:]
            valley_values = prices[last_two_valleys]
            if abs(valley_values[0] - valley_values[1]) / valley_values[0] < 0.03:
                return 1
        return 0
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> int:
        """Detect head and shoulders pattern"""
        if len(prices) < 30:
            return 0
        
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prices, distance=5)
        
        if len(peaks) >= 3:
            last_three_peaks = peaks[-3:]
            peak_values = prices[last_three_peaks]
            
            # Middle peak should be highest (head)
            if peak_values[1] > peak_values[0] and peak_values[1] > peak_values[2]:
                # Shoulders should be roughly equal
                if abs(peak_values[0] - peak_values[2]) / peak_values[0] < 0.05:
                    return 1
        return 0
    
    def _detect_triangle(self, prices: np.ndarray) -> int:
        """Detect triangle pattern"""
        if len(prices) < 20:
            return 0
        
        # Check if price range is decreasing (converging)
        first_half_range = np.max(prices[:len(prices)//2]) - np.min(prices[:len(prices)//2])
        second_half_range = np.max(prices[len(prices)//2:]) - np.min(prices[len(prices)//2:])
        
        if second_half_range < first_half_range * 0.7:
            return 1
        return 0
    
    def _detect_flag(self, prices: np.ndarray) -> int:
        """Detect flag pattern"""
        if len(prices) < 15:
            return 0
        
        # Look for strong move followed by consolidation
        first_third = prices[:len(prices)//3]
        last_two_thirds = prices[len(prices)//3:]
        
        first_move = abs(first_third[-1] - first_third[0]) / first_third[0]
        consolidation_range = (np.max(last_two_thirds) - np.min(last_two_thirds)) / np.mean(last_two_thirds)
        
        if first_move > 0.1 and consolidation_range < 0.05:
            return 1
        return 0
    
    def _detect_doji(self, data: Dict) -> int:
        """Detect doji candlestick"""
        open_price = data.get('open', [])[-1] if data.get('open') else 0
        close_price = data.get('close', [])[-1] if data.get('close') else 0
        high = data.get('high', [])[-1] if data.get('high') else 0
        low = data.get('low', [])[-1] if data.get('low') else 0
        
        if high - low > 0:
            body = abs(close_price - open_price)
            range_hl = high - low
            if body / range_hl < 0.1:  # Small body relative to range
                return 1
        return 0
    
    def _detect_hammer(self, data: Dict) -> int:
        """Detect hammer candlestick"""
        open_price = data.get('open', [])[-1] if data.get('open') else 0
        close_price = data.get('close', [])[-1] if data.get('close') else 0
        high = data.get('high', [])[-1] if data.get('high') else 0
        low = data.get('low', [])[-1] if data.get('low') else 0
        
        body = abs(close_price - open_price)
        lower_shadow = min(open_price, close_price) - low
        upper_shadow = high - max(open_price, close_price)
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            return 1
        return 0
    
    def _detect_engulfing(self, data: Dict) -> int:
        """Detect engulfing pattern"""
        if not all(k in data for k in ['open', 'close']) or len(data['open']) < 2:
            return 0
        
        prev_open = data['open'][-2]
        prev_close = data['close'][-2]
        curr_open = data['open'][-1]
        curr_close = data['close'][-1]
        
        # Bullish engulfing
        if prev_close < prev_open:  # Previous was bearish
            if curr_close > curr_open:  # Current is bullish
                if curr_open <= prev_close and curr_close >= prev_open:
                    return 1
        
        # Bearish engulfing
        if prev_close > prev_open:  # Previous was bullish
            if curr_close < curr_open:  # Current is bearish
                if curr_open >= prev_close and curr_close <= prev_open:
                    return -1
        
        return 0
    
    def create_feature_vector(self, features: Dict[str, float], feature_list: List[str]) -> np.ndarray:
        """
        Create a feature vector in the correct order for ML models
        
        Args:
            features: Dictionary of features
            feature_list: Ordered list of feature names
            
        Returns:
            Feature vector as numpy array
        """
        vector = []
        for feature_name in feature_list:
            value = features.get(feature_name, 0)
            # Handle non-numeric values
            if isinstance(value, bool):
                value = float(value)
            elif not isinstance(value, (int, float)):
                value = 0
            vector.append(value)
        
        return np.array(vector)
    
    def get_feature_importance(self, model, feature_list: List[str]) -> Dict[str, float]:
        """
        Get feature importance from a trained model
        
        Args:
            model: Trained ML model with feature_importances_ attribute
            feature_list: List of feature names
            
        Returns:
            Dictionary of feature importances
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {feature: importance for feature, importance in zip(feature_list, importances)}
        return {}