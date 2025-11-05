"""
Volume Validator ML Model - Machine learning-based volume validation for ClaudeDex Trading Bot
Uses ensemble methods to detect fake volume and predict genuine trading patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import joblib
# Conditionally import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    import xgboost as xgb
    import lightgbm as lgb
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VolumeValidationResult:
    """Result of volume validation"""
    token_address: str
    is_genuine: bool
    confidence: float
    volume_score: float  # 0-100, higher is better
    anomaly_score: float  # -1 to 1, negative means anomaly
    predicted_real_volume: Decimal
    features_importance: Dict[str, float]
    risk_factors: List[str]
    recommendation: str


class VolumeValidatorML:
    """
    Machine learning model for detecting fake volume and wash trading
    Uses ensemble of models for robust predictions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ml_enabled = self.config.get('ml_enabled', False) and ML_LIBS_AVAILABLE
        
        # Model parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.anomaly_threshold = self.config.get("anomaly_threshold", -0.1)
        
        # Initialize models
        self.rf_classifier: Optional[RandomForestClassifier] = None
        self.xgb_classifier: Optional[xgb.XGBClassifier] = None
        self.lgb_classifier: Optional[lgb.LGBMClassifier] = None
        self.isolation_forest: Optional[IsolationForest] = None
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Feature names
        self.feature_names = [
            # Volume features
            'volume_24h', 'volume_velocity', 'volume_acceleration',
            'volume_variance', 'volume_skewness', 'volume_kurtosis',
            
            # Trade pattern features
            'trade_count', 'avg_trade_size', 'trade_size_variance',
            'trade_frequency', 'trade_regularity_score',
            
            # Time distribution features
            'hour_concentration', 'day_concentration', 'peak_hour_ratio',
            'off_peak_ratio', 'weekend_ratio',
            
            # Trader features
            'unique_traders', 'trader_concentration', 'new_trader_ratio',
            'repeat_trader_ratio', 'trader_network_density',
            
            # Price impact features
            'price_volume_correlation', 'slippage_ratio', 'price_impact',
            'bid_ask_spread', 'price_volatility',
            
            # DEX distribution features
            'dex_count', 'primary_dex_dominance', 'cross_dex_arbitrage',
            
            # Liquidity features
            'liquidity_depth', 'liquidity_ratio', 'liquidity_stability',
            
            # Whale activity features
            'whale_trade_ratio', 'whale_accumulation', 'smart_money_flow',
            
            # Technical indicators
            'volume_rsi', 'volume_macd', 'obv_trend', 'vwap_deviation',
            
            # Anomaly features
            'circular_trading_score', 'bot_pattern_score', 'wash_trading_score'
        ]
        
        # Model weights for ensemble
        self.model_weights = {
            'random_forest': 0.3,
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'isolation_forest': 0.1
        }
        
        # Performance metrics
        self.performance_history = []
        self.last_training_date: Optional[datetime] = None
        
        logger.info("Volume Validator ML initialized")
    
    def extract_features(self, volume_data: Dict) -> np.ndarray:
        """
        Extract features from volume data for ML prediction
        
        Args:
            volume_data: Dictionary containing volume metrics
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Volume features
        features.append(float(volume_data.get('volume_24h', 0)))
        features.append(volume_data.get('volume_velocity', 0))
        features.append(volume_data.get('volume_acceleration', 0))
        features.append(volume_data.get('volume_variance', 0))
        features.append(volume_data.get('volume_skewness', 0))
        features.append(volume_data.get('volume_kurtosis', 0))
        
        # Trade pattern features
        features.append(volume_data.get('trade_count', 0))
        features.append(float(volume_data.get('avg_trade_size', 0)))
        features.append(volume_data.get('trade_size_variance', 0))
        features.append(volume_data.get('trade_frequency', 0))
        features.append(volume_data.get('trade_regularity_score', 0))
        
        # Time distribution features
        features.append(volume_data.get('hour_concentration', 0))
        features.append(volume_data.get('day_concentration', 0))
        features.append(volume_data.get('peak_hour_ratio', 0))
        features.append(volume_data.get('off_peak_ratio', 0))
        features.append(volume_data.get('weekend_ratio', 0))
        
        # Trader features
        features.append(volume_data.get('unique_traders', 0))
        features.append(volume_data.get('trader_concentration', 0))
        features.append(volume_data.get('new_trader_ratio', 0))
        features.append(volume_data.get('repeat_trader_ratio', 0))
        features.append(volume_data.get('trader_network_density', 0))
        
        # Price impact features
        features.append(volume_data.get('price_volume_correlation', 0))
        features.append(volume_data.get('slippage_ratio', 0))
        features.append(volume_data.get('price_impact', 0))
        features.append(volume_data.get('bid_ask_spread', 0))
        features.append(volume_data.get('price_volatility', 0))
        
        # DEX distribution features
        features.append(volume_data.get('dex_count', 1))
        features.append(volume_data.get('primary_dex_dominance', 1))
        features.append(volume_data.get('cross_dex_arbitrage', 0))
        
        # Liquidity features
        features.append(float(volume_data.get('liquidity_depth', 0)))
        features.append(volume_data.get('liquidity_ratio', 0))
        features.append(volume_data.get('liquidity_stability', 0))
        
        # Whale activity features
        features.append(volume_data.get('whale_trade_ratio', 0))
        features.append(volume_data.get('whale_accumulation', 0))
        features.append(volume_data.get('smart_money_flow', 0))
        
        # Technical indicators
        features.append(volume_data.get('volume_rsi', 50))
        features.append(volume_data.get('volume_macd', 0))
        features.append(volume_data.get('obv_trend', 0))
        features.append(volume_data.get('vwap_deviation', 0))
        
        # Anomaly features
        features.append(volume_data.get('circular_trading_score', 0))
        features.append(volume_data.get('bot_pattern_score', 0))
        features.append(volume_data.get('wash_trading_score', 0))
        
        return np.array(features).reshape(1, -1)
    
    def train(
        self,
        training_data: pd.DataFrame,
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the ensemble of volume validation models
        
        Args:
            training_data: DataFrame with features
            labels: Binary labels (1 = genuine, 0 = fake)
            validation_split: Fraction for validation
            
        Returns:
            Dictionary with training metrics
        """
        if not self.ml_enabled:
            logger.warning("ML is disabled. Skipping volume validator training.")
            return {}
        try:
            logger.info(f"Training volume validator on {len(training_data)} samples")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                training_data,
                labels,
                test_size=validation_split,
                random_state=42,
                stratify=labels
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train Random Forest
            logger.info("Training Random Forest...")
            self.rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.rf_classifier.fit(X_train_scaled, y_train)
            rf_score = self.rf_classifier.score(X_val_scaled, y_val)
            
            # Train XGBoost
            logger.info("Training XGBoost...")
            self.xgb_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.xgb_classifier.fit(X_train_scaled, y_train)
            xgb_score = self.xgb_classifier.score(X_val_scaled, y_val)
            
            # Train LightGBM
            logger.info("Training LightGBM...")
            self.lgb_classifier = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self.lgb_classifier.fit(X_train_scaled, y_train)
            lgb_score = self.lgb_classifier.score(X_val_scaled, y_val)
            
            # Train Isolation Forest for anomaly detection
            logger.info("Training Isolation Forest...")
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_train_scaled)
            
            # Calculate ensemble score
            ensemble_predictions = self._ensemble_predict(X_val_scaled)
            ensemble_accuracy = np.mean(ensemble_predictions == y_val)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                self.rf_classifier,
                X_train_scaled,
                y_train,
                cv=5,
                scoring='accuracy'
            )
            
            metrics = {
                'rf_accuracy': rf_score,
                'xgb_accuracy': xgb_score,
                'lgb_accuracy': lgb_score,
                'ensemble_accuracy': ensemble_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            self.last_training_date = datetime.now()
            self.performance_history.append(metrics)
            
            logger.info(f"Training completed. Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, volume_data: Dict) -> VolumeValidationResult:
        """
        Predict if volume is genuine or fake
        
        Args:
            volume_data: Dictionary with volume metrics
            
        Returns:
            VolumeValidationResult with predictions
        """
        if not self.ml_enabled:
            return VolumeValidationResult(
                token_address=volume_data.get('token_address', ''),
                is_genuine=False,
                confidence=0.0,
                volume_score=0.0,
                anomaly_score=-1.0,
                predicted_real_volume=Decimal('0'),
                features_importance={},
                risk_factors=['ML is disabled'],
                recommendation='Unable to validate - treat as suspicious'
            )
        try:
            # Extract features
            features = self.extract_features(volume_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from each model
            predictions = {}
            probabilities = {}
            
            if self.rf_classifier:
                predictions['rf'] = self.rf_classifier.predict(features_scaled)[0]
                probabilities['rf'] = self.rf_classifier.predict_proba(features_scaled)[0, 1]
            
            if self.xgb_classifier:
                predictions['xgb'] = self.xgb_classifier.predict(features_scaled)[0]
                probabilities['xgb'] = self.xgb_classifier.predict_proba(features_scaled)[0, 1]
            
            if self.lgb_classifier:
                predictions['lgb'] = self.lgb_classifier.predict(features_scaled)[0]
                probabilities['lgb'] = self.lgb_classifier.predict_proba(features_scaled)[0, 1]
            
            # Anomaly detection
            anomaly_score = 0
            if self.isolation_forest:
                anomaly_score = self.isolation_forest.score_samples(features_scaled)[0]
            
            # Calculate ensemble prediction
            if probabilities:
                weighted_prob = sum(
                    prob * self.model_weights.get(model, 0.25)
                    for model, prob in probabilities.items()
                )
                is_genuine = weighted_prob > 0.5
                confidence = abs(weighted_prob - 0.5) * 2  # Scale to 0-1
            else:
                # Fallback if models not trained
                is_genuine = self._heuristic_validation(volume_data)
                confidence = 0.5
                weighted_prob = 0.5
            
            # Calculate volume score (0-100)
            volume_score = weighted_prob * 100
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(volume_data, anomaly_score)
            
            # Estimate real volume
            predicted_real_volume = self._estimate_real_volume(
                volume_data,
                is_genuine,
                confidence
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                is_genuine,
                confidence,
                anomaly_score,
                risk_factors
            )
            
            return VolumeValidationResult(
                token_address=volume_data.get('token_address', ''),
                is_genuine=is_genuine,
                confidence=confidence,
                volume_score=volume_score,
                anomaly_score=anomaly_score,
                predicted_real_volume=predicted_real_volume,
                features_importance=feature_importance,
                risk_factors=risk_factors,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return conservative result on error
            return VolumeValidationResult(
                token_address=volume_data.get('token_address', ''),
                is_genuine=False,
                confidence=0.0,
                volume_score=0.0,
                anomaly_score=-1.0,
                predicted_real_volume=Decimal('0'),
                features_importance={},
                risk_factors=['Prediction error'],
                recommendation='Unable to validate - treat as suspicious'
            )
    
    def _ensemble_predict(self, features: np.ndarray) -> np.ndarray:
        """Get ensemble predictions"""
        predictions = []
        
        if self.rf_classifier:
            predictions.append(self.rf_classifier.predict_proba(features)[:, 1])
        if self.xgb_classifier:
            predictions.append(self.xgb_classifier.predict_proba(features)[:, 1])
        if self.lgb_classifier:
            predictions.append(self.lgb_classifier.predict_proba(features)[:, 1])
        
        if predictions:
            # Weighted average
            weights = [self.model_weights.get(m, 0.33) for m in ['random_forest', 'xgboost', 'lightgbm']]
            weights = weights[:len(predictions)]
            weights = np.array(weights) / sum(weights)
            
            ensemble_proba = np.average(predictions, axis=0, weights=weights)
            return (ensemble_proba > 0.5).astype(int)
        
        return np.array([0])
    
    def _heuristic_validation(self, volume_data: Dict) -> bool:
        """Fallback heuristic validation when models not available"""
        suspicious_indicators = 0
        
        # Check wash trading score
        if volume_data.get('wash_trading_score', 0) > 0.5:
            suspicious_indicators += 1
        
        # Check bot pattern score
        if volume_data.get('bot_pattern_score', 0) > 0.5:
            suspicious_indicators += 1
        
        # Check unique traders
        if volume_data.get('unique_traders', 0) < 50:
            suspicious_indicators += 1
        
        # Check trade regularity
        if volume_data.get('trade_regularity_score', 0) > 0.8:
            suspicious_indicators += 1
        
        # Check circular trading
        if volume_data.get('circular_trading_score', 0) > 0.3:
            suspicious_indicators += 1
        
        return suspicious_indicators < 2
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        if not self.rf_classifier:
            return {}
        
        importance = self.rf_classifier.feature_importances_
        
        # Create importance dictionary
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            if i < len(importance):
                feature_importance[name] = float(importance[i])
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return sorted_importance
    
    def _identify_risk_factors(self, volume_data: Dict, anomaly_score: float) -> List[str]:
        """Identify specific risk factors in the volume data"""
        risk_factors = []
        
        # Anomaly detection
        if anomaly_score < self.anomaly_threshold:
            risk_factors.append("Anomalous trading pattern detected")
        
        # Wash trading
        if volume_data.get('wash_trading_score', 0) > 0.7:
            risk_factors.append("High wash trading probability")
        elif volume_data.get('wash_trading_score', 0) > 0.5:
            risk_factors.append("Moderate wash trading indicators")
        
        # Bot trading
        if volume_data.get('bot_pattern_score', 0) > 0.7:
            risk_factors.append("Bot trading patterns detected")
        
        # Low unique traders
        if volume_data.get('unique_traders', 0) < 30:
            risk_factors.append("Very low number of unique traders")
        elif volume_data.get('unique_traders', 0) < 50:
            risk_factors.append("Low number of unique traders")
        
        # Trade regularity
        if volume_data.get('trade_regularity_score', 0) > 0.9:
            risk_factors.append("Unnaturally regular trading intervals")
        
        # Circular trading
        if volume_data.get('circular_trading_score', 0) > 0.5:
            risk_factors.append("Circular trading patterns present")
        
        # Concentration
        if volume_data.get('trader_concentration', 0) > 0.7:
            risk_factors.append("High trader concentration")
        
        # Price impact
        if volume_data.get('price_impact', 0) < 0.001:
            risk_factors.append("Suspiciously low price impact for volume")
        
        # DEX concentration
        if volume_data.get('primary_dex_dominance', 0) > 0.95:
            risk_factors.append("Volume concentrated on single DEX")
        
        return risk_factors
    
    def _estimate_real_volume(
        self,
        volume_data: Dict,
        is_genuine: bool,
        confidence: float
    ) -> Decimal:
        """Estimate the real organic volume"""
        reported_volume = Decimal(str(volume_data.get('volume_24h', 0)))
        
        if is_genuine and confidence > 0.8:
            # High confidence genuine volume
            return reported_volume * Decimal('0.95')
        elif is_genuine:
            # Lower confidence genuine
            return reported_volume * Decimal('0.85')
        else:
            # Fake volume detected
            # Estimate based on unique traders and average organic trade size
            unique_traders = volume_data.get('unique_traders', 10)
            avg_trade_size = Decimal(str(volume_data.get('avg_trade_size', 100)))
            
            # Rough estimate: unique_traders * average_trades_per_trader * avg_size
            estimated_real = Decimal(unique_traders) * Decimal('5') * avg_trade_size
            
            # Don't exceed reported volume
            return min(estimated_real, reported_volume * Decimal('0.3'))
    
    def _generate_recommendation(
        self,
        is_genuine: bool,
        confidence: float,
        anomaly_score: float,
        risk_factors: List[str]
    ) -> str:
        """Generate trading recommendation based on validation"""
        if is_genuine and confidence > 0.8:
            return "Volume appears genuine - Safe to trade with normal precautions"
        elif is_genuine and confidence > 0.6:
            return "Volume likely genuine - Monitor for changes"
        elif is_genuine:
            return "Volume possibly genuine - Exercise caution"
        elif len(risk_factors) > 3:
            return "Multiple red flags detected - Avoid trading"
        elif anomaly_score < -0.3:
            return "Highly anomalous volume - Strong risk of manipulation"
        else:
            return "Suspicious volume patterns - Not recommended for trading"
    
    def save_model(self, filepath: str = "volume_validator_model.pkl") -> None:
        """Save trained models to disk"""
        model_data = {
            'rf_classifier': self.rf_classifier,
            'xgb_classifier': self.xgb_classifier,
            'lgb_classifier': self.lgb_classifier,
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'last_training_date': self.last_training_date,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "volume_validator_model.pkl") -> None:
        """Load trained models from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.rf_classifier = model_data.get('rf_classifier')
            self.xgb_classifier = model_data.get('xgb_classifier')
            self.lgb_classifier = model_data.get('lgb_classifier')
            self.isolation_forest = model_data.get('isolation_forest')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.model_weights = model_data.get('model_weights', self.model_weights)
            self.last_training_date = model_data.get('last_training_date')
            self.performance_history = model_data.get('performance_history', [])
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        if not self.performance_history:
            return {"status": "No training history available"}
        
        latest = self.performance_history[-1]
        
        return {
            "last_training": self.last_training_date.isoformat() if self.last_training_date else None,
            "ensemble_accuracy": latest.get('ensemble_accuracy', 0),
            "rf_accuracy": latest.get('rf_accuracy', 0),
            "xgb_accuracy": latest.get('xgb_accuracy', 0),
            "lgb_accuracy": latest.get('lgb_accuracy', 0),
            "cv_mean": latest.get('cv_mean', 0),
            "cv_std": latest.get('cv_std', 0),
            "training_samples": latest.get('training_samples', 0),
            "total_trainings": len(self.performance_history)
        }
    
    def needs_retraining(self, days_threshold: int = 7) -> bool:
        """Check if model needs retraining"""
        if not self.last_training_date:
            return True
        
        days_since_training = (datetime.now() - self.last_training_date).days
        return days_since_training > days_threshold