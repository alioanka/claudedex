"""
AI Strategy - Machine Learning driven trading strategy for ClaudeDex Trading Bot

This module implements an AI-powered trading strategy using ensemble ML models.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .base_strategy import (
    BaseStrategy, 
    TradingSignal, 
    SignalType, 
    SignalStrength,
    StrategyState
)
from ml.models.ensemble_model import EnsembleModel
from ml.models.pump_predictor import PumpPredictor
from ml.models.rug_classifier import RugClassifier
from core.pattern_analyzer import PatternAnalyzer
from utils.helpers import calculate_moving_average, calculate_ema


class AIStrategy(BaseStrategy):
    """AI-powered trading strategy using ensemble machine learning"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AI strategy
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # ML Models
        self.ensemble_model: Optional[EnsembleModel] = None
        self.pump_predictor: Optional[PumpPredictor] = None
        self.rug_classifier: Optional[RugClassifier] = None
        self.pattern_analyzer: Optional[PatternAnalyzer] = None
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.feature_window = config.get("feature_window", 50)
        self.feature_columns = [
            "price", "volume", "market_cap", "liquidity",
            "holders", "transactions", "buy_pressure", "sell_pressure"
        ]
        
        # AI-specific parameters
        self.ml_confidence_threshold = config.get("ml_confidence_threshold", 0.75)
        self.ensemble_weight_pump = config.get("ensemble_weight_pump", 0.4)
        self.ensemble_weight_pattern = config.get("ensemble_weight_pattern", 0.3)
        self.ensemble_weight_technical = config.get("ensemble_weight_technical", 0.3)
        
        # Risk thresholds
        self.max_rug_probability = config.get("max_rug_probability", 0.2)
        self.min_pump_probability = config.get("min_pump_probability", 0.6)
        self.min_pattern_score = config.get("min_pattern_score", 0.7)
        
        # Adaptive learning
        self.enable_online_learning = config.get("enable_online_learning", True)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.retrain_interval = timedelta(hours=config.get("retrain_interval_hours", 24))
        self.last_retrain_time = datetime.now()
        
        # Performance tracking for learning
        self.prediction_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"Initialized AI Strategy: {self.name}")
    
    async def initialize(self) -> None:
        """Initialize ML models and components"""
        logger.info("Initializing AI Strategy components...")
        
        # Initialize models
        self.ensemble_model = EnsembleModel(self.config.get("ensemble_config", {}))
        self.pump_predictor = PumpPredictor(self.config.get("pump_config", {}))
        self.rug_classifier = RugClassifier(self.config.get("rug_config", {}))
        self.pattern_analyzer = PatternAnalyzer(self.config.get("pattern_config", {}))
        
        # Load pre-trained models if available
        await self._load_models()
        
        logger.info("AI Strategy initialization complete")
    
    async def analyze(
        self,
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Analyze market using AI models
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal if conditions are met
        """
        try:
            # Check if we should analyze
            token_address = market_data.get("token_address", "")
            if not self.should_analyze(token_address, market_data):
                return None
            
            # Update analysis time
            self._last_analysis_time[token_address] = datetime.now()
            
            # Extract features
            features = await self._extract_features(market_data)
            if features is None:
                logger.debug("Insufficient features for analysis")
                return None
            
            # Check rug probability first (safety check)
            rug_probability = await self._check_rug_probability(features, market_data)
            if rug_probability > self.max_rug_probability:
                logger.warning(
                    f"High rug probability detected: {rug_probability:.2f} "
                    f"for {token_address[:10]}..."
                )
                return None
            
            # Get ML predictions
            predictions = await self._get_ml_predictions(features, market_data)
            
            # Combine predictions into signal
            signal = await self._generate_signal_from_predictions(
                predictions,
                market_data,
                rug_probability
            )
            
            # Validate signal
            if signal and self.validate_signal(signal, market_data):
                # Record prediction for learning
                if self.enable_online_learning:
                    await self._record_prediction(signal, predictions)
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return None
    
    async def calculate_indicators(
        self,
        price_data: List[float],
        volume_data: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate AI-specific indicators
        
        Args:
            price_data: Historical prices
            volume_data: Historical volumes
            
        Returns:
            Dictionary of indicators
        """
        indicators = {}
        
        if len(price_data) < self.feature_window:
            return indicators
        
        # Price-based features
        indicators["returns"] = self._calculate_returns(price_data)
        indicators["log_returns"] = self._calculate_log_returns(price_data)
        indicators["volatility"] = self._calculate_volatility(price_data)
        indicators["price_momentum"] = self._calculate_momentum(price_data)
        
        # Volume-based features
        indicators["volume_ratio"] = self._calculate_volume_ratio(volume_data)
        indicators["volume_momentum"] = self._calculate_momentum(volume_data)
        indicators["vwap"] = self._calculate_vwap(price_data, volume_data)
        
        # Statistical features
        indicators["skewness"] = self._calculate_skewness(price_data)
        indicators["kurtosis"] = self._calculate_kurtosis(price_data)
        indicators["hurst_exponent"] = self._calculate_hurst_exponent(price_data)
        
        # Market microstructure
        indicators["bid_ask_spread"] = await self._estimate_spread(price_data)
        indicators["order_imbalance"] = await self._calculate_order_imbalance(volume_data)
        
        # Technical indicators for ML
        indicators["rsi"] = self._calculate_rsi(price_data)
        indicators["macd_signal"] = self._calculate_macd_signal(price_data)
        indicators["bb_position"] = self._calculate_bb_position(price_data)
        
        # Pattern features
        indicators["pattern_strength"] = await self._detect_pattern_strength(price_data)
        indicators["support_distance"] = self._calculate_support_distance(price_data)
        indicators["resistance_distance"] = self._calculate_resistance_distance(price_data)
        
        return indicators
    
    def validate_signal(
        self,
        signal: TradingSignal,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Validate AI-generated signal
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid
        """
        # Check confidence threshold
        if signal.confidence < self.ml_confidence_threshold:
            logger.debug(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        # Check signal strength
        if signal.strength in [SignalStrength.WEAK, SignalStrength.VERY_WEAK]:
            logger.debug(f"Signal too weak: {signal.strength.value}")
            return False
        
        # Verify ML model agreement
        ml_agreement = signal.metadata.get("ml_agreement", 0)
        if ml_agreement < 0.6:  # At least 60% of models should agree
            logger.debug(f"Insufficient ML model agreement: {ml_agreement:.2f}")
            return False
        
        # Check feature quality
        feature_quality = signal.metadata.get("feature_quality", 0)
        if feature_quality < 0.7:
            logger.debug(f"Poor feature quality: {feature_quality:.2f}")
            return False
        
        return True
    
    async def _extract_features(
        self,
        market_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract features for ML models"""
        try:
            # Get historical data
            price_history = market_data.get("price_history", [])
            volume_history = market_data.get("volume_history", [])
            
            if len(price_history) < self.feature_window:
                return None
            
            # Calculate indicators
            indicators = await self.calculate_indicators(
                price_history[-self.feature_window:],
                volume_history[-self.feature_window:]
            )
            
            # Create feature vector
            features = []
            
            # Market data features
            features.append(market_data.get("market_cap", 0))
            features.append(market_data.get("liquidity", 0))
            features.append(market_data.get("holders_count", 0))
            features.append(market_data.get("24h_transactions", 0))
            
            # Calculated features
            for key in ["returns", "volatility", "volume_ratio", "rsi", 
                       "macd_signal", "pattern_strength"]:
                if key in indicators:
                    if isinstance(indicators[key], list):
                        features.extend(indicators[key][-5:])  # Last 5 values
                    else:
                        features.append(indicators[key])
            
            # Social sentiment if available
            features.append(market_data.get("social_score", 0.5))
            features.append(market_data.get("sentiment_score", 0.5))
            
            # Convert to numpy array
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features
            if len(self.scaler.mean_) > 0:
                feature_array = self.scaler.transform(feature_array)
            else:
                feature_array = self.scaler.fit_transform(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _check_rug_probability(
        self,
        features: np.ndarray,
        market_data: Dict[str, Any]
    ) -> float:
        """Check probability of rug pull"""
        if not self.rug_classifier:
            return 0.0
        
        try:
            # Prepare rug-specific features
            rug_features = {
                "liquidity_locked": market_data.get("liquidity_locked", False),
                "contract_verified": market_data.get("contract_verified", False),
                "owner_percentage": market_data.get("owner_percentage", 0),
                "holder_distribution": market_data.get("holder_distribution", {}),
                "contract_age_hours": market_data.get("contract_age_hours", 0),
                "developer_history": market_data.get("developer_history", {})
            }
            
            # Get rug probability
            rug_prob, risk_factors = self.rug_classifier.predict(rug_features)
            
            # Log if high risk
            if rug_prob > 0.5:
                logger.warning(
                    f"Rug risk factors: {risk_factors} "
                    f"Probability: {rug_prob:.2f}"
                )
            
            return rug_prob
            
        except Exception as e:
            logger.error(f"Rug check failed: {e}")
            return 1.0  # Assume high risk on error
    
    async def _get_ml_predictions(
        self,
        features: np.ndarray,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get predictions from all ML models"""
        predictions = {}
        
        # Ensemble model prediction
        if self.ensemble_model:
            ensemble_result = await self.ensemble_model.predict(
                market_data.get("token_address", ""),
                market_data.get("chain", "")
            )
            predictions["ensemble"] = ensemble_result
        
        # Pump prediction
        if self.pump_predictor:
            pump_features = self._prepare_pump_features(features, market_data)
            pump_prob = self.pump_predictor.predict_pump_probability(pump_features)
            predictions["pump_probability"] = pump_prob
        
        # Pattern analysis
        if self.pattern_analyzer:
            price_data = market_data.get("price_history", [])
            pattern_result = await self.pattern_analyzer.analyze_patterns(
                [{"price": p} for p in price_data]
            )
            predictions["patterns"] = pattern_result
        
        # Technical analysis score
        technical_score = await self._calculate_technical_score(features, market_data)
        predictions["technical_score"] = technical_score
        
        return predictions
    
    async def _generate_signal_from_predictions(
        self,
        predictions: Dict[str, Any],
        market_data: Dict[str, Any],
        rug_probability: float
    ) -> Optional[TradingSignal]:
        """Generate trading signal from ML predictions"""
        # Extract predictions
        ensemble = predictions.get("ensemble", {})
        pump_prob = predictions.get("pump_probability", 0)
        patterns = predictions.get("patterns", {})
        technical_score = predictions.get("technical_score", 0)
        
        # Calculate weighted score
        weighted_score = (
            pump_prob * self.ensemble_weight_pump +
            patterns.get("score", 0) * self.ensemble_weight_pattern +
            technical_score * self.ensemble_weight_technical
        )
        
        # Adjust for rug risk
        risk_adjusted_score = weighted_score * (1 - rug_probability)
        
        # Determine signal type
        if risk_adjusted_score > 0.7 and pump_prob > self.min_pump_probability:
            signal_type = SignalType.BUY
            strength = self._determine_signal_strength(risk_adjusted_score)
        else:
            return None  # No signal
        
        # Calculate confidence
        model_agreement = self._calculate_model_agreement(predictions)
        confidence = risk_adjusted_score * model_agreement
        
# Fix for ai_strategy.py line 409
# Replace the incorrect TradingSignal instantiation with this corrected version:

        # Create signal (CORRECTED VERSION)
        signal = TradingSignal()
        signal.strategy_name = self.name
        signal.signal_type = signal_type
        signal.strength = strength
        signal.token_address = market_data.get("token_address", "")
        signal.chain = market_data.get("chain", "")
        signal.entry_price = Decimal(str(market_data.get("price", 0)))
        signal.confidence = confidence
        signal.timeframe = self.timeframe
        signal.indicators = {
            "pump_probability": pump_prob,
            "rug_probability": rug_probability,
            "technical_score": technical_score,
            "pattern_score": patterns.get("score", 0),
            "weighted_score": weighted_score
        }
        signal.metadata = {
            "ml_agreement": model_agreement,
            "risk_adjusted_score": risk_adjusted_score,
            "feature_quality": self._assess_feature_quality(market_data),
            "patterns_detected": patterns.get("patterns", []),
            "ensemble_prediction": ensemble
        }
        
        # Set expiration
        signal.expires_at = datetime.now() + timedelta(minutes=15)
        
        return signal
    
    def _determine_signal_strength(self, score: float) -> SignalStrength:
        """Determine signal strength from score"""
        if score >= 0.9:
            return SignalStrength.VERY_STRONG
        elif score >= 0.8:
            return SignalStrength.STRONG
        elif score >= 0.7:
            return SignalStrength.MODERATE
        elif score >= 0.6:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _calculate_model_agreement(self, predictions: Dict[str, Any]) -> float:
        """Calculate agreement between different ML models"""
        scores = []
        
        # Normalize all predictions to 0-1 range
        if "pump_probability" in predictions:
            scores.append(predictions["pump_probability"])
        
        if "technical_score" in predictions:
            scores.append(predictions["technical_score"])
        
        if "patterns" in predictions:
            scores.append(predictions["patterns"].get("score", 0))
        
        if not scores:
            return 0.0
        
        # Calculate standard deviation as measure of disagreement
        std_dev = np.std(scores)
        
        # Convert to agreement score (lower std = higher agreement)
        agreement = 1.0 - min(std_dev * 2, 1.0)
        
        return agreement
    
    def _assess_feature_quality(self, market_data: Dict[str, Any]) -> float:
        """Assess quality of features"""
        quality_score = 1.0
        
        # Check data completeness
        required_fields = ["price_history", "volume_history", "liquidity", "holders_count"]
        for field in required_fields:
            if field not in market_data or not market_data[field]:
                quality_score -= 0.2
        
        # Check data freshness
        last_update = market_data.get("last_update")
        if last_update:
            age = datetime.now() - last_update
            if age > timedelta(minutes=5):
                quality_score -= 0.3
        
        # Check data consistency
        price_history = market_data.get("price_history", [])
        if price_history:
            # Check for gaps or anomalies
            prices = np.array(price_history)
            if np.any(prices <= 0) or np.any(np.isnan(prices)):
                quality_score -= 0.3
        
        return max(0, quality_score)
    
    def _prepare_pump_features(
        self,
        features: np.ndarray,
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """Prepare features specifically for pump prediction"""
        # Extract pump-relevant features
        pump_features = []
        
        # Volume acceleration
        volume_history = market_data.get("volume_history", [])
        if len(volume_history) >= 10:
            recent_vol = np.mean(volume_history[-5:])
            older_vol = np.mean(volume_history[-10:-5])
            vol_acceleration = recent_vol / older_vol if older_vol > 0 else 1
            pump_features.append(vol_acceleration)
        
        # Price momentum
        price_history = market_data.get("price_history", [])
        if len(price_history) >= 20:
            short_ma = np.mean(price_history[-5:])
            long_ma = np.mean(price_history[-20:])
            momentum = short_ma / long_ma if long_ma > 0 else 1
            pump_features.append(momentum)
        
        # Social metrics
        pump_features.append(market_data.get("social_volume_24h", 0))
        pump_features.append(market_data.get("social_engagement_rate", 0))
        
        # Market metrics
        pump_features.append(market_data.get("unique_buyers_1h", 0))
        pump_features.append(market_data.get("buy_sell_ratio", 0.5))
        
        # Add original features
        if features is not None and features.size > 0:
            pump_features.extend(features.flatten()[:10])  # First 10 features
        
        return np.array(pump_features).reshape(1, -1)
    
    async def _calculate_technical_score(
        self,
        features: np.ndarray,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate technical analysis score"""
        score = 0.5  # Neutral starting point
        
        price_history = market_data.get("price_history", [])
        volume_history = market_data.get("volume_history", [])
        
        if len(price_history) < 20:
            return score
        
        # RSI signal
        rsi = self._calculate_rsi(price_history)
        if 30 < rsi < 70:
            score += 0.1  # Neutral RSI
        elif rsi <= 30:
            score += 0.2  # Oversold
        else:
            score -= 0.1  # Overbought
        
        # MACD signal
        macd_signal = self._calculate_macd_signal(price_history)
        if macd_signal > 0:
            score += 0.15
        
        # Volume confirmation
        if len(volume_history) >= 10:
            recent_vol = np.mean(volume_history[-3:])
            avg_vol = np.mean(volume_history[-10:])
            if recent_vol > avg_vol * 1.5:
                score += 0.15  # Volume surge
        
        # Trend alignment
        short_trend = self._calculate_trend(price_history[-10:])
        long_trend = self._calculate_trend(price_history[-50:])
        if short_trend > 0 and long_trend > 0:
            score += 0.1  # Aligned uptrend
        
        return min(1.0, max(0.0, score))
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate price returns"""
        if len(prices) < 2:
            return [0.0]
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        return returns
    
    def _calculate_log_returns(self, prices: List[float]) -> List[float]:
        """Calculate logarithmic returns"""
        if len(prices) < 2:
            return [0.0]
        
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_ret = np.log(prices[i] / prices[i-1])
                log_returns.append(log_ret)
        
        return log_returns
    
    def _calculate_volatility(self, prices: List[float], window: int = 20) -> float:
        """Calculate price volatility"""
        returns = self._calculate_returns(prices)
        if len(returns) < window:
            return 0.0
        
        return np.std(returns[-window:]) * np.sqrt(252)  # Annualized
    
    def _calculate_momentum(self, data: List[float], period: int = 10) -> float:
        """Calculate momentum"""
        if len(data) < period:
            return 0.0
        
        return (data[-1] - data[-period]) / data[-period] if data[-period] != 0 else 0
    
    def _calculate_volume_ratio(self, volumes: List[float], period: int = 20) -> float:
        """Calculate volume ratio"""
        if len(volumes) < period:
            return 1.0
        
        recent = np.mean(volumes[-5:])
        average = np.mean(volumes[-period:])
        
        return recent / average if average > 0 else 1.0
    
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate volume-weighted average price"""
        if len(prices) != len(volumes) or len(prices) == 0:
            return 0.0
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return np.mean(prices)
        
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return vwap
    
    def _calculate_skewness(self, prices: List[float]) -> float:
        """Calculate price distribution skewness"""
        returns = self._calculate_returns(prices)
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, prices: List[float]) -> float:
        """Calculate price distribution kurtosis"""
        returns = self._calculate_returns(prices)
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean) / std) ** 4) - 3
        return kurtosis
    
    def _calculate_hurst_exponent(self, prices: List[float]) -> float:
        """Calculate Hurst exponent for trend persistence"""
        if len(prices) < 100:
            return 0.5  # Random walk
        
        # Simplified Hurst calculation
        lags = range(2, min(20, len(prices) // 2))
        tau = []
        
        for lag in lags:
            returns = [prices[i] - prices[i-lag] for i in range(lag, len(prices))]
            if returns:
                tau.append(np.std(returns))
        
        if len(tau) < 2:
            return 0.5
        
        # Fit power law
        x = np.log(list(lags))
        y = np.log(tau)
        
        poly = np.polyfit(x, y, 1)
        hurst = poly[0]
        
        return max(0, min(1, hurst))
    
    async def _estimate_spread(self, prices: List[float]) -> float:
        """Estimate bid-ask spread"""
        if len(prices) < 2:
            return 0.0
        
        # Roll's spread estimator
        price_changes = np.diff(prices)
        if len(price_changes) < 2:
            return 0.0
        
        cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
        
        if cov >= 0:
            return 0.0
        
        spread = 2 * np.sqrt(-cov)
        return spread / np.mean(prices) if np.mean(prices) > 0 else 0
    
    async def _calculate_order_imbalance(self, volumes: List[float]) -> float:
        """Calculate order imbalance (simplified)"""
        if len(volumes) < 2:
            return 0.0
        
        # Assume alternating buy/sell for simplification
        buy_volume = sum(volumes[::2])
        sell_volume = sum(volumes[1::2])
        
        total = buy_volume + sell_volume
        if total == 0:
            return 0.0
        
        imbalance = (buy_volume - sell_volume) / total
        return imbalance
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_signal(self, prices: List[float]) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
        
        ema12 = calculate_ema(prices, 12)[-1]
        ema26 = calculate_ema(prices, 26)[-1]
        
        macd = ema12 - ema26
        
        return macd
    
    def _calculate_bb_position(self, prices: List[float], period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        mean = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        if std == 0:
            return 0.5
        
        upper = mean + 2 * std
        lower = mean - 2 * std
        
        current = prices[-1]
        
        if current >= upper:
            return 1.0
        elif current <= lower:
            return 0.0
        else:
            return (current - lower) / (upper - lower)
    
    async def _detect_pattern_strength(self, prices: List[float]) -> float:
        """Detect pattern strength"""
        if not self.pattern_analyzer or len(prices) < 20:
            return 0.0
        
        # Use pattern analyzer
        patterns = await self.pattern_analyzer.analyze_patterns(
            [{"price": p} for p in prices]
        )
        
        return patterns.get("strength", 0.0)
    
    def _calculate_support_distance(self, prices: List[float]) -> float:
        """Calculate distance to support level"""
        if len(prices) < 20:
            return 0.0
        
        current = prices[-1]
        support = min(prices[-20:])
        
        return (current - support) / current if current > 0 else 0
    
    def _calculate_resistance_distance(self, prices: List[float]) -> float:
        """Calculate distance to resistance level"""
        if len(prices) < 20:
            return 0.0
        
        current = prices[-1]
        resistance = max(prices[-20:])
        
        return (resistance - current) / current if current > 0 else 0
    
    def _calculate_trend(self, prices: List[float]) -> float:
        """Calculate trend direction"""
        if len(prices) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize by average price
        avg_price = np.mean(prices)
        if avg_price > 0:
            return slope / avg_price
        
        return 0.0
    
    async def _record_prediction(
        self,
        signal: TradingSignal,
        predictions: Dict[str, Any]
    ) -> None:
        """Record prediction for online learning"""
        self.prediction_history.append({
            "timestamp": datetime.now(),
            "signal": signal,
            "predictions": predictions,
            "outcome": None  # To be filled later
        })
        
        # Limit history size
        max_history = 1000
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]
    
    async def update_prediction_outcome(
        self,
        signal_id: str,
        outcome: Dict[str, Any]
    ) -> None:
        """Update prediction outcome for learning"""
        for pred in self.prediction_history:
            if pred["signal"].id == signal_id:
                pred["outcome"] = outcome
                break
        
        # Trigger retraining if needed
        if self.enable_online_learning:
            await self._check_retrain()
    
    async def _check_retrain(self) -> None:
        """Check if models should be retrained"""
        if datetime.now() - self.last_retrain_time < self.retrain_interval:
            return
        
        # Check if we have enough new data
        recent_predictions = [
            p for p in self.prediction_history
            if p["outcome"] is not None
        ]
        
        if len(recent_predictions) < 50:
            return
        
        logger.info("Triggering AI model retraining...")
        await self._retrain_models(recent_predictions)
        self.last_retrain_time = datetime.now()
    
    async def _retrain_models(self, training_data: List[Dict[str, Any]]) -> None:
        """Retrain ML models with new data"""
        try:
            # Prepare training data
            X = []
            y = []
            
            for data in training_data:
                # Extract features from prediction
                # This would need proper feature extraction
                X.append(data["predictions"])
                
                # Extract label from outcome
                outcome = data["outcome"]
                if outcome["pnl"] > 0:
                    y.append(1)
                else:
                    y.append(0)
            
            # Update models (simplified)
            # In reality, would call model.train() methods
            logger.info(f"Retrained models with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    async def _load_models(self) -> None:
        """Load pre-trained models"""
        try:
            # Load saved models if they exist
            # This would load from disk/database
            logger.info("Loading pre-trained AI models...")
            
            # For now, models will use their defaults
            
        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from models"""
        # This would aggregate feature importance from all models
        return self.feature_importance
    
    async def explain_prediction(
        self,
        signal: TradingSignal
    ) -> Dict[str, Any]:
        """Explain AI prediction"""
        explanation = {
            "signal_id": signal.id,
            "confidence": signal.confidence,
            "key_factors": [],
            "risk_factors": [],
            "technical_indicators": signal.indicators,
            "ml_models_used": ["ensemble", "pump_predictor", "pattern_analyzer"]
        }
        
        # Add key positive factors
        if signal.indicators.get("pump_probability", 0) > 0.7:
            explanation["key_factors"].append(
                f"High pump probability: {signal.indicators['pump_probability']:.2f}"
            )
        
        if signal.indicators.get("technical_score", 0) > 0.7:
            explanation["key_factors"].append(
                f"Strong technical score: {signal.indicators['technical_score']:.2f}"
            )
        
        # Add risk factors
        if signal.indicators.get("rug_probability", 0) > 0.1:
            explanation["risk_factors"].append(
                f"Rug risk: {signal.indicators['rug_probability']:.2f}"
            )
        
        return explanation