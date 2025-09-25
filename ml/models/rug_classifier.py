# ml/models/rug_classifier.py

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class RugClassifier:
    """
    Machine learning model for detecting potential rug pulls in cryptocurrency tokens.
    Uses ensemble methods combining multiple classifiers for robust predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_weights = {
            'xgboost': 0.35,
            'lightgbm': 0.30,
            'random_forest': 0.20,
            'gradient_boost': 0.15
        }
        
        # Feature configuration
        self.feature_columns = [
            # Liquidity features
            'liquidity_ratio',
            'liquidity_locked_percentage',
            'liquidity_removal_risk',
            'lp_burn_percentage',
            
            # Holder features
            'holder_concentration_top10',
            'holder_concentration_top50',
            'unique_holders',
            'whale_percentage',
            'dev_wallet_percentage',
            
            # Contract features
            'has_mint_function',
            'has_pause_function',
            'has_blacklist',
            'has_trading_cooldown',
            'has_max_tx_amount',
            'contract_verified',
            'hidden_owner',
            'proxy_contract',
            
            # Trading features
            'buy_sell_ratio',
            'unique_buyers_sellers_ratio',
            'volume_liquidity_ratio',
            'price_volatility',
            'abnormal_volume_spikes',
            
            # Developer features
            'dev_wallet_activity',
            'dev_selling_pressure',
            'team_token_locks',
            'contract_age_hours',
            
            # Social features
            'social_score',
            'twitter_followers_growth',
            'telegram_members_growth',
            'website_quality_score',
            
            # Technical patterns
            'pump_dump_pattern',
            'honeypot_characteristics',
            'wash_trading_score',
            'bot_activity_score'
        ]
        
        # Initialize models
        self._initialize_models()
        
        # Model paths
        self.model_dir = Path(config.get('MODEL_DIR', './models/rug_classifier'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_models(self):
        """Initialize ensemble of classifiers."""
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            verbose=-1
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    
    def extract_features(self, token_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from token data for prediction."""
        features = []
        
        # Liquidity features
        features.append(token_data.get('liquidity_usd', 0) / max(token_data.get('market_cap', 1), 1))
        features.append(token_data.get('liquidity_locked_percentage', 0))
        features.append(self._calculate_liquidity_removal_risk(token_data))
        features.append(token_data.get('lp_burn_percentage', 0))
        
        # Holder features
        features.append(token_data.get('holder_concentration_top10', 0))
        features.append(token_data.get('holder_concentration_top50', 0))
        features.append(token_data.get('unique_holders', 0))
        features.append(token_data.get('whale_percentage', 0))
        features.append(token_data.get('dev_wallet_percentage', 0))
        
        # Contract features
        features.append(int(token_data.get('has_mint_function', False)))
        features.append(int(token_data.get('has_pause_function', False)))
        features.append(int(token_data.get('has_blacklist', False)))
        features.append(int(token_data.get('has_trading_cooldown', False)))
        features.append(int(token_data.get('has_max_tx_amount', False)))
        features.append(int(token_data.get('contract_verified', False)))
        features.append(int(token_data.get('hidden_owner', False)))
        features.append(int(token_data.get('proxy_contract', False)))
        
        # Trading features
        features.append(token_data.get('buy_sell_ratio', 1.0))
        features.append(token_data.get('unique_buyers_sellers_ratio', 1.0))
        features.append(token_data.get('volume_liquidity_ratio', 0))
        features.append(token_data.get('price_volatility', 0))
        features.append(token_data.get('abnormal_volume_spikes', 0))
        
        # Developer features
        features.append(token_data.get('dev_wallet_activity', 0))
        features.append(token_data.get('dev_selling_pressure', 0))
        features.append(token_data.get('team_token_locks', 0))
        features.append(token_data.get('contract_age_hours', 0))
        
        # Social features
        features.append(token_data.get('social_score', 0))
        features.append(token_data.get('twitter_followers_growth', 0))
        features.append(token_data.get('telegram_members_growth', 0))
        features.append(token_data.get('website_quality_score', 0))
        
        # Technical patterns
        features.append(token_data.get('pump_dump_pattern', 0))
        features.append(token_data.get('honeypot_characteristics', 0))
        features.append(token_data.get('wash_trading_score', 0))
        features.append(token_data.get('bot_activity_score', 0))
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_liquidity_removal_risk(self, token_data: Dict) -> float:
        """Calculate risk of liquidity removal."""
        risk_score = 0.0
        
        # Check if liquidity is locked
        if not token_data.get('liquidity_locked', False):
            risk_score += 0.4
        
        # Check liquidity lock duration
        lock_duration = token_data.get('liquidity_lock_duration_days', 0)
        if lock_duration < 30:
            risk_score += 0.3
        elif lock_duration < 90:
            risk_score += 0.2
        elif lock_duration < 180:
            risk_score += 0.1
        
        # Check LP token distribution
        if token_data.get('lp_concentration_top1', 0) > 0.5:
            risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    def train(
        self,
        historical_data: pd.DataFrame,
        labels: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the rug classifier ensemble."""
        logger.info("Starting rug classifier training...")
        
        # Prepare data
        X = historical_data[self.feature_columns].values
        y = labels
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Training results
        results = {
            'models': {},
            'ensemble_metrics': {},
            'feature_importance': {}
        }
        
        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Scale features
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_val_scaled = self.scalers[model_name].transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Calibrate probabilities
            calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
            calibrated.fit(X_train_scaled, y_train)
            self.models[model_name] = calibrated
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Store metrics
            results['models'][model_name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(
                    zip(self.feature_columns, model.feature_importances_)
                )
        
        # Ensemble predictions
        ensemble_proba = self._ensemble_predict_proba(X_val)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Ensemble metrics
        results['ensemble_metrics'] = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred),
            'recall': recall_score(y_val, ensemble_pred),
            'f1': f1_score(y_val, ensemble_pred),
            'roc_auc': roc_auc_score(y_val, ensemble_proba),
            'confusion_matrix': confusion_matrix(y_val, ensemble_pred).tolist()
        }
        
        # Calculate overall feature importance
        results['feature_importance'] = self._calculate_ensemble_importance()
        
        logger.info(f"Training complete. Ensemble ROC-AUC: {results['ensemble_metrics']['roc_auc']:.4f}")
        
        return results
    
    def predict(self, token_features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Predict rug probability for a token.
        Returns probability and individual model predictions.
        """
        # Extract features
        features = self.extract_features(token_features)
        
        # Get predictions from each model
        predictions = {}
        weighted_sum = 0.0
        
        for model_name, model in self.models.items():
            # Scale features
            features_scaled = self.scalers[model_name].transform(features)
            
            # Predict probability
            proba = model.predict_proba(features_scaled)[0, 1]
            predictions[model_name] = proba
            
            # Add to weighted sum
            weighted_sum += proba * self.model_weights[model_name]
        
        # Final ensemble prediction
        ensemble_proba = weighted_sum
        
        return ensemble_proba, predictions
    
    def _ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions."""
        ensemble_proba = np.zeros(len(X))
        
        for model_name, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[model_name].transform(X)
            
            # Get predictions
            proba = model.predict_proba(X_scaled)[:, 1]
            
            # Add weighted contribution
            ensemble_proba += proba * self.model_weights[model_name]
        
        return ensemble_proba
    
    def _calculate_ensemble_importance(self) -> Dict[str, float]:
        """Calculate weighted ensemble feature importance."""
        ensemble_importance = {feature: 0.0 for feature in self.feature_columns}
        
        for model_name, importance in self.feature_importance.items():
            weight = self.model_weights[model_name]
            for feature, value in importance.items():
                ensemble_importance[feature] += value * weight
        
        # Normalize
        total = sum(ensemble_importance.values())
        if total > 0:
            ensemble_importance = {k: v/total for k, v in ensemble_importance.items()}
        
        # Sort by importance
        return dict(sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive rug pull analysis for a token.
        Returns detailed risk assessment.
        """
        # Get predictions
        rug_probability, model_predictions = self.predict(token_data)
        
        # Risk level classification
        if rug_probability < 0.3:
            risk_level = "LOW"
        elif rug_probability < 0.6:
            risk_level = "MEDIUM"
        elif rug_probability < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Identify red flags
        red_flags = self._identify_red_flags(token_data)
        
        # Generate warnings
        warnings = self._generate_warnings(token_data, rug_probability)
        
        return {
            'rug_probability': rug_probability,
            'risk_level': risk_level,
            'model_predictions': model_predictions,
            'red_flags': red_flags,
            'warnings': warnings,
            'recommendation': self._get_recommendation(rug_probability),
            'confidence': self._calculate_confidence(model_predictions),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _identify_red_flags(self, token_data: Dict[str, Any]) -> List[str]:
        """Identify specific red flags in token data."""
        red_flags = []
        
        # Contract red flags
        if token_data.get('has_mint_function', False):
            red_flags.append("Contract has mint function - unlimited supply risk")
        
        if token_data.get('hidden_owner', False):
            red_flags.append("Contract has hidden owner - transparency issue")
        
        if not token_data.get('contract_verified', True):
            red_flags.append("Contract not verified - code transparency issue")
        
        if token_data.get('has_blacklist', False):
            red_flags.append("Contract has blacklist function - trading restriction risk")
        
        # Liquidity red flags
        if token_data.get('liquidity_locked_percentage', 100) < 50:
            red_flags.append("Less than 50% liquidity locked - removal risk")
        
        if token_data.get('liquidity_usd', float('inf')) < 10000:
            red_flags.append("Low liquidity - high slippage and manipulation risk")
        
        # Holder red flags
        if token_data.get('holder_concentration_top10', 0) > 50:
            red_flags.append("Top 10 holders own >50% - high concentration risk")
        
        if token_data.get('dev_wallet_percentage', 0) > 10:
            red_flags.append("Dev wallet holds >10% - dump risk")
        
        if token_data.get('unique_holders', float('inf')) < 100:
            red_flags.append("Less than 100 holders - low distribution")
        
        # Trading red flags
        if token_data.get('buy_sell_ratio', 1) < 0.5:
            red_flags.append("More sells than buys - selling pressure")
        
        if token_data.get('wash_trading_score', 0) > 0.5:
            red_flags.append("High wash trading detected - fake volume")
        
        # Age red flags
        if token_data.get('contract_age_hours', float('inf')) < 24:
            red_flags.append("Contract less than 24 hours old - new token risk")
        
        return red_flags
    
    def _generate_warnings(self, token_data: Dict[str, Any], rug_probability: float) -> List[str]:
        """Generate specific warnings based on analysis."""
        warnings = []
        
        if rug_probability > 0.7:
            warnings.append("⚠️ CRITICAL: Very high rug pull probability detected")
        
        if token_data.get('honeypot_characteristics', 0) > 0.5:
            warnings.append("⚠️ Honeypot characteristics detected - cannot sell")
        
        if token_data.get('pump_dump_pattern', 0) > 0.7:
            warnings.append("⚠️ Pump and dump pattern detected")
        
        if token_data.get('dev_selling_pressure', 0) > 0.5:
            warnings.append("⚠️ Developer is actively selling tokens")
        
        if token_data.get('liquidity_removal_risk', 0) > 0.7:
            warnings.append("⚠️ High risk of liquidity removal")
        
        return warnings
    
    def _get_recommendation(self, rug_probability: float) -> str:
        """Get trading recommendation based on rug probability."""
        if rug_probability < 0.3:
            return "Low risk - Can consider trading with proper risk management"
        elif rug_probability < 0.5:
            return "Moderate risk - Trade with caution and small position size"
        elif rug_probability < 0.7:
            return "High risk - Not recommended for trading"
        else:
            return "AVOID - Extremely high rug pull risk"
    
    def _calculate_confidence(self, model_predictions: Dict[str, float]) -> float:
        """Calculate prediction confidence based on model agreement."""
        predictions = list(model_predictions.values())
        
        # Calculate standard deviation
        std_dev = np.std(predictions)
        
        # Lower std means higher agreement/confidence
        confidence = max(0, 1 - (std_dev * 2))
        
        return confidence
    
    def save_model(self, version: Optional[str] = None) -> str:
        """Save trained models to disk."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"rug_classifier_{version}"
        model_path.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            joblib.dump(model, model_path / f"{model_name}.pkl")
            joblib.dump(self.scalers[model_name], model_path / f"{model_name}_scaler.pkl")
        
        # Save model weights and feature importance
        metadata = {
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'feature_columns': self.feature_columns,
            'version': version,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(model_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, version: str) -> None:
        """Load trained models from disk."""
        model_path = self.model_dir / f"rug_classifier_{version}"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Load each model
        for model_name in self.models.keys():
            model_file = model_path / f"{model_name}.pkl"
            scaler_file = model_path / f"{model_name}_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                self.models[model_name] = joblib.load(model_file)
                self.scalers[model_name] = joblib.load(scaler_file)
        
        # Load metadata
        with open(model_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.model_weights = metadata['model_weights']
            self.feature_importance = metadata['feature_importance']
            self.feature_columns = metadata['feature_columns']
        
        logger.info(f"Model loaded from {model_path}")
    
    def update_model(self, new_data: pd.DataFrame, new_labels: np.ndarray) -> Dict[str, Any]:
        """Update model with new labeled data (online learning)."""
        # Implement incremental learning for models that support it
        results = {}
        
        X = new_data[self.feature_columns].values
        y = new_labels
        
        for model_name in ['xgboost', 'lightgbm']:  # These support incremental learning
            if model_name in self.models:
                X_scaled = self.scalers[model_name].transform(X)
                
                # Partial fit for incremental learning
                if hasattr(self.models[model_name], 'fit'):
                    # Re-train with combined data (simplified approach)
                    # In production, use proper incremental learning
                    self.models[model_name].fit(X_scaled, y)
                    
                    # Evaluate on new data
                    y_pred = self.models[model_name].predict(X_scaled)
                    results[model_name] = {
                        'accuracy': accuracy_score(y, y_pred),
                        'samples_updated': len(y)
                    }
        
        logger.info(f"Model updated with {len(y)} new samples")
        return results