"""
Advanced Machine Learning Ensemble System
Combines multiple models for robust predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass
# Conditionally import ML libraries to allow running without them
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    import xgboost as xgb
    import lightgbm as lgb
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PredictionResult:
    """Result from ensemble prediction"""
    pump_probability: float
    rug_probability: float
    expected_return: float
    confidence: float
    time_to_pump: Optional[float]  # Hours
    risk_adjusted_score: float
    model_agreements: Dict[str, float]
    feature_importance: Dict[str, float]
    prediction_timestamp: datetime = None
    
    def __post_init__(self):
        if self.prediction_timestamp is None:
            self.prediction_timestamp = datetime.now()

class LSTMPricePredictor(nn.Module):
    """LSTM model for price movement prediction"""
    
    def __init__(self, input_dim: int = 150, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 outputs: pump_prob, rug_prob, expected_return
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        last_output = attn_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        # Apply sigmoid to probabilities, leave expected return as is
        output[:, :2] = torch.sigmoid(output[:, :2])
        
        return output

class TransformerPredictor(nn.Module):
    """Transformer model for pattern recognition"""
    
    def __init__(self, input_dim: int = 150, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.2,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = x.transpose(0, 1)  # Transformer expects seq_len first
        encoded = self.transformer(x)
        
        # Global average pooling
        pooled = encoded.mean(dim=0)
        
        # Classification
        output = self.classifier(pooled)
        output[:, :2] = torch.sigmoid(output[:, :2])
        
        return output

class EnsemblePredictor:
    """Main ensemble prediction system"""
    
    def __init__(self, config: Dict[str, Any], model_dir: str = "models/"):
        """
        Initialize ensemble predictor
        
        Args:
            config: Configuration dictionary
            model_dir: Directory containing trained models
        """
        self.config = config
        self.ml_enabled = self.config.get('ml_enabled', False) and ML_LIBS_AVAILABLE
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'xgboost_rug': None,
            'xgboost_pump': None,
            'lightgbm_rug': None,
            'lightgbm_pump': None,
            'random_forest': None,
            'gradient_boosting': None,
            'lstm': None,
            'transformer': None,
            'isolation_forest': None
        }
        
        # Feature engineering
        self.scaler = RobustScaler()
        self.feature_names = []
        
        # Model weights for ensemble
        self.model_weights = {
            'xgboost_rug': 0.20,
            'xgboost_pump': 0.20,
            'lightgbm_rug': 0.15,
            'lightgbm_pump': 0.15,
            'random_forest': 0.10,
            'gradient_boosting': 0.10,
            'lstm': 0.15,
            'transformer': 0.15,
            'isolation_forest': 0.10
        }
        
        # Performance tracking
        self.model_performance = {}
        
        # Thread pool for parallel predictions
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_models(self):
        """Load all trained models"""
        if not self.ml_enabled:
            print("ML is disabled. Skipping model loading.")
            return
        try:
            # XGBoost models
            xgb_rug_path = self.model_dir / "xgboost_rug.pkl"
            if xgb_rug_path.exists():
                self.models['xgboost_rug'] = joblib.load(xgb_rug_path)
            else:
                self.models['xgboost_rug'] = self._create_xgboost_model()
                
            xgb_pump_path = self.model_dir / "xgboost_pump.pkl"
            if xgb_pump_path.exists():
                self.models['xgboost_pump'] = joblib.load(xgb_pump_path)
            else:
                self.models['xgboost_pump'] = self._create_xgboost_model()
                
            # LightGBM models
            lgb_rug_path = self.model_dir / "lightgbm_rug.pkl"
            if lgb_rug_path.exists():
                self.models['lightgbm_rug'] = joblib.load(lgb_rug_path)
            else:
                self.models['lightgbm_rug'] = self._create_lightgbm_model()
                
            lgb_pump_path = self.model_dir / "lightgbm_pump.pkl"
            if lgb_pump_path.exists():
                self.models['lightgbm_pump'] = joblib.load(lgb_pump_path)
            else:
                self.models['lightgbm_pump'] = self._create_lightgbm_model()
                
            # Random Forest
            rf_path = self.model_dir / "random_forest.pkl"
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
            else:
                self.models['random_forest'] = self._create_random_forest_model()
                
            # Gradient Boosting
            gb_path = self.model_dir / "gradient_boosting.pkl"
            if gb_path.exists():
                self.models['gradient_boosting'] = joblib.load(gb_path)
            else:
                self.models['gradient_boosting'] = self._create_gradient_boosting_model()
                
            # LSTM
            lstm_path = self.model_dir / "lstm_model.pt"
            if lstm_path.exists():
                self.models['lstm'] = LSTMPricePredictor()
                self.models['lstm'].load_state_dict(torch.load(lstm_path))
                self.models['lstm'].eval()
            else:
                self.models['lstm'] = LSTMPricePredictor()
                
            # Transformer
            transformer_path = self.model_dir / "transformer_model.pt"
            if transformer_path.exists():
                self.models['transformer'] = TransformerPredictor()
                self.models['transformer'].load_state_dict(torch.load(transformer_path))
                self.models['transformer'].eval()
            else:
                self.models['transformer'] = TransformerPredictor()
                
            # Isolation Forest for anomaly detection
            iso_path = self.model_dir / "isolation_forest.pkl"
            if iso_path.exists():
                self.models['isolation_forest'] = joblib.load(iso_path)
            else:
                self.models['isolation_forest'] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                
            # Load feature names
            features_path = self.model_dir / "features.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                    
        except Exception as e:
            print(f"Error loading models: {e}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
            
    def _create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with optimized parameters"""
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.01,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
        
    def _create_lightgbm_model(self) -> lgb.LGBMClassifier:
        """Create LightGBM model with optimized parameters"""
        return lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            verbosity=-1
        )
        
    def _create_random_forest_model(self) -> RandomForestClassifier:
        """Create Random Forest model"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
    def _create_gradient_boosting_model(self) -> GradientBoostingClassifier:
        """Create Gradient Boosting model"""
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        )
        
    def extract_features(self, data: Dict) -> np.ndarray:
        """
        Extract features from raw data
        
        Args:
            data: Dictionary containing token data
            
        Returns:
            Feature array
        """
        features = []
        
        # Price features
        price_data = data.get('price_data', {})
        features.extend([
            price_data.get('current_price', 0),
            price_data.get('price_change_1h', 0),
            price_data.get('price_change_24h', 0),
            price_data.get('price_change_7d', 0),
            price_data.get('volatility_24h', 0),
            price_data.get('price_std', 0),
            price_data.get('price_skew', 0),
            price_data.get('price_kurtosis', 0)
        ])
        
        # Volume features
        volume_data = data.get('volume_data', {})
        features.extend([
            volume_data.get('volume_24h', 0),
            volume_data.get('volume_change_24h', 0),
            volume_data.get('buy_volume_ratio', 0),
            volume_data.get('large_trades_ratio', 0),
            volume_data.get('unique_traders_24h', 0),
            volume_data.get('avg_trade_size', 0),
            volume_data.get('volume_to_mcap_ratio', 0)
        ])
        
        # Liquidity features
        liquidity_data = data.get('liquidity_data', {})
        features.extend([
            liquidity_data.get('total_liquidity', 0),
            liquidity_data.get('liquidity_change_24h', 0),
            liquidity_data.get('liquidity_locked_percent', 0),
            liquidity_data.get('liquidity_to_mcap_ratio', 0),
            liquidity_data.get('impermanent_loss_risk', 0)
        ])
        
        # Holder features
        holder_data = data.get('holder_data', {})
        features.extend([
            holder_data.get('total_holders', 0),
            holder_data.get('holder_growth_24h', 0),
            holder_data.get('top_10_holders_percent', 0),
            holder_data.get('whale_count', 0),
            holder_data.get('avg_holding_time', 0),
            holder_data.get('holder_concentration_index', 0)
        ])
        
        # Contract features
        contract_data = data.get('contract_data', {})
        features.extend([
            float(contract_data.get('is_verified', False)),
            float(contract_data.get('has_mint_function', False)),
            float(contract_data.get('has_pause_function', False)),
            float(contract_data.get('ownership_renounced', False)),
            float(contract_data.get('is_proxy', False)),
            contract_data.get('contract_age_days', 0),
            contract_data.get('transaction_count', 0)
        ])
        
        # Social features
        social_data = data.get('social_data', {})
        features.extend([
            social_data.get('twitter_followers', 0),
            social_data.get('twitter_engagement_rate', 0),
            social_data.get('telegram_members', 0),
            social_data.get('telegram_growth_rate', 0),
            social_data.get('reddit_mentions', 0),
            social_data.get('sentiment_score', 0),
            social_data.get('fomo_index', 0)
        ])
        
        # Technical indicators
        technical_data = data.get('technical_data', {})
        features.extend([
            technical_data.get('rsi', 50),
            technical_data.get('macd', 0),
            technical_data.get('macd_signal', 0),
            technical_data.get('bollinger_upper', 0),
            technical_data.get('bollinger_lower', 0),
            technical_data.get('ema_9', 0),
            technical_data.get('ema_21', 0),
            technical_data.get('ema_50', 0),
            technical_data.get('obv', 0),
            technical_data.get('adx', 0)
        ])
        
        # Risk scores
        risk_data = data.get('risk_data', {})
        features.extend([
            risk_data.get('liquidity_risk', 0),
            risk_data.get('developer_risk', 0),
            risk_data.get('contract_risk', 0),
            risk_data.get('volume_risk', 0),
            risk_data.get('holder_risk', 0),
            risk_data.get('honeypot_probability', 0)
        ])
        
        # Market features
        market_data = data.get('market_data', {})
        features.extend([
            market_data.get('btc_correlation', 0),
            market_data.get('eth_correlation', 0),
            market_data.get('market_cap', 0),
            market_data.get('fully_diluted_valuation', 0),
            market_data.get('circulating_supply_percent', 0)
        ])
        
        # Time-based features
        time_data = data.get('time_data', {})
        features.extend([
            time_data.get('hour_of_day', 0),
            time_data.get('day_of_week', 0),
            time_data.get('days_since_launch', 0),
            time_data.get('hours_since_ath', 0),
            time_data.get('hours_since_atl', 0)
        ])
        
        # Pattern features
        pattern_data = data.get('pattern_data', {})
        features.extend([
            float(pattern_data.get('has_cup_handle', False)),
            float(pattern_data.get('has_ascending_triangle', False)),
            float(pattern_data.get('has_double_bottom', False)),
            float(pattern_data.get('has_golden_cross', False)),
            float(pattern_data.get('has_death_cross', False)),
            pattern_data.get('trend_strength', 0),
            pattern_data.get('support_level_distance', 0),
            pattern_data.get('resistance_level_distance', 0)
        ])
        
        # Mempool features
        mempool_data = data.get('mempool_data', {})
        features.extend([
            mempool_data.get('pending_buy_volume', 0),
            mempool_data.get('pending_sell_volume', 0),
            mempool_data.get('large_pending_trades', 0),
            mempool_data.get('sandwich_attack_risk', 0)
        ])
        
        # Whale activity
        whale_data = data.get('whale_data', {})
        features.extend([
            whale_data.get('whale_accumulation_score', 0),
            whale_data.get('whale_distribution_score', 0),
            whale_data.get('smart_money_flow', 0),
            whale_data.get('institutional_interest', 0)
        ])
        
        return np.array(features, dtype=np.float32)

    # ============================================
    # FIX 1: ensemble_model.py - EnsemblePredictor.predict
    # ============================================
    # Current signature: async def predict(self, features: np.ndarray) -> PredictionResult
    # Expected signature: async def predict(self, token: str, chain: str) -> Dict

    # Add this wrapper method to EnsemblePredictor class in ensemble_model.py:

    # Replace the current predict() method (around line 565) with this fixed version:

    async def predict(self, token: str, chain: str) -> Dict:
        """
        Predict pump/rug probability for a token (API-compliant signature)
        
        Args:
            token: Token address
            chain: Blockchain network
            
        Returns:
            Dictionary with prediction results
        """
        if not self.ml_enabled:
            return {
                'token': token,
                'chain': chain,
                'pump_probability': 0.5,
                'rug_probability': 0.5,
                'expected_return': 0.0,
                'confidence': 0.1,
                'time_to_pump': None,
                'risk_adjusted_score': 0.0,
                'model_agreements': {},
                'feature_importance': {},
                'timestamp': datetime.now().isoformat(),
                'error': 'ML is disabled in the configuration.'
            }
        try:
            # Import at runtime to avoid circular dependency
            import sys
            from pathlib import Path
            
            # Add parent directory to path if needed
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from data.collectors.dexscreener import DexScreenerCollector
            from config.config_manager import ConfigManager
            
            # ============================================
            # FIX: Create config dict for DexScreenerCollector
            # ============================================
            config = {
                'api_key': '',
                'chains': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'base'],
                'rate_limit': 100,
                'cache_duration': 60,
                'min_liquidity': 10000,
                'min_volume': 5000,
                'max_age_hours': 24
            }
            
            # Fetch token data - NOW WITH CONFIG PARAMETER
            collector = DexScreenerCollector(config)
            await collector.initialize()
            token_data = await collector.get_token_info(token, chain)
            
            # Extract features and get prediction
            features = self.extract_features(token_data)
            result = await self._predict_from_features(features)
            
            # Return API-compliant format
            return {
                'token': token,
                'chain': chain,
                'pump_probability': result.pump_probability,
                'rug_probability': result.rug_probability,
                'expected_return': result.expected_return,
                'confidence': result.confidence,
                'time_to_pump': result.time_to_pump,
                'risk_adjusted_score': result.risk_adjusted_score,
                'model_agreements': result.model_agreements,
                'feature_importance': result.feature_importance,
                'timestamp': result.prediction_timestamp.isoformat()
            }
        except Exception as e:
            # Return safe default on error
            return {
                'token': token,
                'chain': chain,
                'pump_probability': 0.5,
                'rug_probability': 0.5,
                'expected_return': 0.0,
                'confidence': 0.1,
                'time_to_pump': None,
                'risk_adjusted_score': 0.0,
                'model_agreements': {},
                'feature_importance': {},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    # Rename existing predict method to _predict_from_features:
    async def _predict_from_features(self, features: np.ndarray) -> PredictionResult:
        """
        Internal method for making predictions from feature array
        (This is the current predict method, just renamed)
        """
        # ... existing implementation ...
        try:
            # Convert to numpy array if needed
            if isinstance(features, dict):
                features = self.extract_features(features)
                
            # Ensure 2D array
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from all models
            predictions = {}
            
            # Tree-based models
            if self.models['xgboost_rug'] is not None:
                predictions['xgboost_rug'] = self.models['xgboost_rug'].predict_proba(features_scaled)[0][1]
            else:
                predictions['xgboost_rug'] = 0.5
                
            if self.models['xgboost_pump'] is not None:
                predictions['xgboost_pump'] = self.models['xgboost_pump'].predict_proba(features_scaled)[0][1]
            else:
                predictions['xgboost_pump'] = 0.5
                
            if self.models['lightgbm_rug'] is not None:
                predictions['lightgbm_rug'] = self.models['lightgbm_rug'].predict_proba(features_scaled)[0][1]
            else:
                predictions['lightgbm_rug'] = 0.5
                
            if self.models['lightgbm_pump'] is not None:
                predictions['lightgbm_pump'] = self.models['lightgbm_pump'].predict_proba(features_scaled)[0][1]
            else:
                predictions['lightgbm_pump'] = 0.5
                
            if self.models['random_forest'] is not None:
                rf_pred = self.models['random_forest'].predict_proba(features_scaled)[0]
                predictions['random_forest_pump'] = rf_pred[1] if len(rf_pred) > 1 else 0.5
                predictions['random_forest_rug'] = 1 - predictions['random_forest_pump']
            else:
                predictions['random_forest_pump'] = 0.5
                predictions['random_forest_rug'] = 0.5
                
            if self.models['gradient_boosting'] is not None:
                gb_pred = self.models['gradient_boosting'].predict_proba(features_scaled)[0]
                predictions['gradient_boosting_pump'] = gb_pred[1] if len(gb_pred) > 1 else 0.5
                predictions['gradient_boosting_rug'] = 1 - predictions['gradient_boosting_pump']
            else:
                predictions['gradient_boosting_pump'] = 0.5
                predictions['gradient_boosting_rug'] = 0.5
                
            # Neural network predictions
            if self.models['lstm'] is not None:
                with torch.no_grad():
                    lstm_input = torch.FloatTensor(features_scaled).unsqueeze(0)
                    lstm_output = self.models['lstm'](lstm_input)
                    predictions['lstm_pump'] = lstm_output[0][0].item()
                    predictions['lstm_rug'] = lstm_output[0][1].item()
                    predictions['lstm_return'] = lstm_output[0][2].item()
            else:
                predictions['lstm_pump'] = 0.5
                predictions['lstm_rug'] = 0.5
                predictions['lstm_return'] = 0.0
                
            if self.models['transformer'] is not None:
                with torch.no_grad():
                    trans_input = torch.FloatTensor(features_scaled).unsqueeze(0)
                    trans_output = self.models['transformer'](trans_input)
                    predictions['transformer_pump'] = trans_output[0][0].item()
                    predictions['transformer_rug'] = trans_output[0][1].item()
                    predictions['transformer_return'] = trans_output[0][2].item()
            else:
                predictions['transformer_pump'] = 0.5
                predictions['transformer_rug'] = 0.5
                predictions['transformer_return'] = 0.0
                
            # Anomaly detection
            if self.models['isolation_forest'] is not None:
                anomaly_score = self.models['isolation_forest'].decision_function(features_scaled)[0]
                predictions['anomaly'] = 1 / (1 + np.exp(-anomaly_score))  # Convert to probability
            else:
                predictions['anomaly'] = 0.5
                
            # Calculate ensemble predictions
            pump_probability = self._calculate_weighted_average(
                [predictions.get(k, 0.5) for k in ['xgboost_pump', 'lightgbm_pump', 
                 'random_forest_pump', 'gradient_boosting_pump', 'lstm_pump', 'transformer_pump']],
                [0.2, 0.15, 0.15, 0.15, 0.2, 0.15]
            )
            
            rug_probability = self._calculate_weighted_average(
                [predictions.get(k, 0.5) for k in ['xgboost_rug', 'lightgbm_rug',
                 'random_forest_rug', 'gradient_boosting_rug', 'lstm_rug', 'transformer_rug']],
                [0.2, 0.15, 0.15, 0.15, 0.2, 0.15]
            )
            
            # Adjust for anomaly
            if predictions['anomaly'] > 0.7:
                rug_probability = min(rug_probability * 1.3, 1.0)
                pump_probability = pump_probability * 0.8
                
            # Calculate expected return
            expected_return = np.mean([
                predictions.get('lstm_return', 0),
                predictions.get('transformer_return', 0),
                (pump_probability - rug_probability) * 100  # Simple estimate
            ])
            
            # Calculate confidence
            model_agreement = 1 - np.std([v for k, v in predictions.items() if 'pump' in k])
            confidence = min(model_agreement * 1.2, 1.0)  # Scale up slightly
            
            # Calculate risk-adjusted score
            risk_adjusted_score = (pump_probability * expected_return) / (rug_probability + 0.1)
            
            # Estimate time to pump (hours)
            time_to_pump = None
            if pump_probability > 0.6:
                # Simple estimation based on patterns
                time_to_pump = max(1, 24 * (1 - pump_probability) * 2)
                
            # Get feature importance
            feature_importance = self._get_feature_importance(features_scaled)
            
            return PredictionResult(
                pump_probability=pump_probability,
                rug_probability=rug_probability,
                expected_return=expected_return,
                confidence=confidence,
                time_to_pump=time_to_pump,
                risk_adjusted_score=risk_adjusted_score,
                model_agreements=predictions,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            # Return neutral prediction on error
            return PredictionResult(
                pump_probability=0.5,
                rug_probability=0.5,
                expected_return=0.0,
                confidence=0.1,
                time_to_pump=None,
                risk_adjusted_score=0.0,
                model_agreements={},
                feature_importance={}
            )
            
    def _calculate_weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average of predictions"""
        if not values:
            return 0.5
            
        # Normalize weights
        total_weight = sum(weights[:len(values)])
        if total_weight == 0:
            return np.mean(values)
            
        normalized_weights = [w/total_weight for w in weights[:len(values)]]
        
        return sum(v * w for v, w in zip(values, normalized_weights))
        
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance from models"""
        importance = {}
        
        try:
            # Get importance from tree models
            if self.models['xgboost_pump'] is not None and hasattr(self.models['xgboost_pump'], 'feature_importances_'):
                xgb_importance = self.models['xgboost_pump'].feature_importances_
                for i, imp in enumerate(xgb_importance[:10]):  # Top 10 features
                    if i < len(self.feature_names):
                        importance[self.feature_names[i]] = float(imp)
                        
        except Exception:
            pass
            
        return importance
        
    async def retrain(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain models with new data
        
        Args:
            training_data: DataFrame with features and labels
            
        Returns:
            Dictionary of new trained models
        """
        if not self.ml_enabled:
            print("ML is disabled. Skipping model retraining.")
            return {}
        try:
            # Prepare data
            X = training_data.drop(['label', 'pump_label', 'rug_label'], axis=1, errors='ignore')
            y_pump = training_data.get('pump_label', pd.Series([0] * len(training_data)))
            y_rug = training_data.get('rug_label', pd.Series([0] * len(training_data)))
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Train models in parallel
            new_models = {}
            
            # XGBoost models
            new_models['xgboost_pump'] = self._create_xgboost_model()
            new_models['xgboost_pump'].fit(X_scaled, y_pump)
            
            new_models['xgboost_rug'] = self._create_xgboost_model()
            new_models['xgboost_rug'].fit(X_scaled, y_rug)
            
            # LightGBM models
            new_models['lightgbm_pump'] = self._create_lightgbm_model()
            new_models['lightgbm_pump'].fit(X_scaled, y_pump)
            
            new_models['lightgbm_rug'] = self._create_lightgbm_model()
            new_models['lightgbm_rug'].fit(X_scaled, y_rug)
            
            # Random Forest
            new_models['random_forest'] = self._create_random_forest_model()
            new_models['random_forest'].fit(X_scaled, y_pump)  # Use pump as primary target
            
            # Gradient Boosting
            new_models['gradient_boosting'] = self._create_gradient_boosting_model()
            new_models['gradient_boosting'].fit(X_scaled, y_pump)
            
            # Train neural networks if enough data
            if len(training_data) > 1000:
                # Prepare data for neural networks
                X_tensor = torch.FloatTensor(X_scaled)
                y_tensor = torch.FloatTensor(np.column_stack([y_pump, y_rug, training_data.get('returns', np.zeros(len(y_pump)))]))
                
                # Train LSTM
                new_models['lstm'] = await self._train_lstm(X_tensor, y_tensor)
                
                # Train Transformer
                new_models['transformer'] = await self._train_transformer(X_tensor, y_tensor)
                
            # Train Isolation Forest
            new_models['isolation_forest'] = IsolationForest(contamination=0.1, random_state=42)
            new_models['isolation_forest'].fit(X_scaled)
            
            return new_models
            
        except Exception as e:
            print(f"Error retraining models: {e}")
            return {}
            
    async def _train_lstm(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50) -> LSTMPricePredictor:
        """Train LSTM model"""
        model = LSTMPricePredictor(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = TensorDataset(X.unsqueeze(1), y)  # Add sequence dimension
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        model.eval()
        return model
        
    async def _train_transformer(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50) -> TransformerPredictor:
        """Train Transformer model"""
        model = TransformerPredictor(input_dim=X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = TensorDataset(X.unsqueeze(1), y)  # Add sequence dimension
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        model.eval()
        return model
        
    async def update_models(self, new_models: Dict[str, Any]):
        """Update ensemble with new models"""
        for name, model in new_models.items():
            if model is not None:
                self.models[name] = model
                
        # Save models
        await self.save_models()
        
    async def save_models(self):
        """Save all models to disk"""
        try:
            # Save sklearn models
            for name in ['xgboost_rug', 'xgboost_pump', 'lightgbm_rug', 'lightgbm_pump',
                        'random_forest', 'gradient_boosting', 'isolation_forest']:
                if self.models[name] is not None:
                    joblib.dump(self.models[name], self.model_dir / f"{name}.pkl")
                    
            # Save PyTorch models
            if self.models['lstm'] is not None:
                torch.save(self.models['lstm'].state_dict(), self.model_dir / "lstm_model.pt")
                
            if self.models['transformer'] is not None:
                torch.save(self.models['transformer'].state_dict(), self.model_dir / "transformer_model.pt")
                
            # Save scaler
            joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
            
            # Save feature names
            with open(self.model_dir / "features.json", 'w') as f:
                json.dump(self.feature_names, f)
                
        except Exception as e:
            print(f"Error saving models: {e}")

    # Add these methods to the EnsemblePredictor class in ensemble_model.py

    async def predict_from_token(self, token: str, chain: str) -> Dict:
        """
        Wrapper method to match documented signature
        Fetches token data and makes prediction
        
        Args:
            token: Token address
            chain: Blockchain network
            
        Returns:
            Prediction results dictionary
        """
        # This would fetch token data from collectors
        # For now, return a placeholder
        from data.collectors.dexscreener import DexScreenerCollector
        from config.config_manager import ConfigManager

        # Initialize config if needed
        config = ConfigManager(config_dir="config")
        await config.initialize()

        # Fetch token data
        collector = DexScreenerCollector(config)  # ← Line 938 - This is CORRECT
        token_data = await collector.get_token_info(token, chain)
        
        # Extract features and predict
        features = self.extract_features(token_data)
        result = await self.predict(features)
        
        return {
            'token': token,
            'chain': chain,
            'pump_probability': result.pump_probability,
            'rug_probability': result.rug_probability,
            'expected_return': result.expected_return,
            'confidence': result.confidence,
            'risk_adjusted_score': result.risk_adjusted_score
        }

    async def combine_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Combined prediction result
        """
        if not predictions:
            return {}
        
        # Aggregate predictions
        pump_probs = [p.get('pump_probability', 0.5) for p in predictions]
        rug_probs = [p.get('rug_probability', 0.5) for p in predictions]
        expected_returns = [p.get('expected_return', 0) for p in predictions]
        confidences = [p.get('confidence', 0.5) for p in predictions]
        
        # Calculate weighted averages using model weights
        weights = list(self.model_weights.values())[:len(predictions)]
        total_weight = sum(weights)
        
        combined = {
            'pump_probability': sum(p * w for p, w in zip(pump_probs, weights)) / total_weight,
            'rug_probability': sum(p * w for p, w in zip(rug_probs, weights)) / total_weight,
            'expected_return': sum(r * w for r, w in zip(expected_returns, weights)) / total_weight,
            'confidence': sum(c * w for c, w in zip(confidences, weights)) / total_weight,
            'model_count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        return combined

    def calculate_weighted_score(self, scores: Dict, weights: Dict) -> float:
        """
        Calculate weighted score from multiple model scores
        
        Args:
            scores: Dictionary of model scores
            weights: Dictionary of model weights
            
        Returns:
            Weighted average score
        """
        total_score = 0
        total_weight = 0
        
        for model_name, score in scores.items():
            weight = weights.get(model_name, 1.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Default neutral score
        
        return total_score / total_weight

    async def get_confidence_level(self, predictions: Dict) -> float:
        """
        Calculate confidence level from model predictions
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Confidence level (0-1)
        """
        if not predictions:
            return 0.0
        
        # Calculate standard deviation of predictions
        values = list(predictions.values())
        if len(values) < 2:
            return 0.5
        
        std_dev = np.std(values)
        
        # Lower standard deviation means higher agreement = higher confidence
        # Map std dev to confidence: std=0 -> conf=1, std=0.5 -> conf=0
        confidence = max(0, min(1, 1 - (std_dev * 2)))
        
        # Adjust based on number of models agreeing
        agreement_bonus = len(values) / len(self.models) * 0.2
        confidence = min(1.0, confidence + agreement_bonus)
        
        return confidence

    def update_weights(self, performance_data: Dict) -> None:
        """
        Update model weights based on performance
        
        Args:
            performance_data: Dictionary with model performance metrics
        """
        # Calculate new weights based on performance
        for model_name, metrics in performance_data.items():
            if model_name in self.model_weights:
                # Use accuracy or ROC-AUC as performance metric
                performance_score = metrics.get('accuracy', metrics.get('roc_auc', 0.5))
                
                # Update weight with exponential moving average
                alpha = 0.1  # Learning rate
                old_weight = self.model_weights[model_name]
                new_weight = alpha * performance_score + (1 - alpha) * old_weight
                
                self.model_weights[model_name] = new_weight
        
        # Normalize weights to sum to 1
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v/total for k, v in self.model_weights.items()}
        
        # Save updated weights
        self.model_performance.update(performance_data)
        
        # Log weight updates
        print(f"Updated model weights: {self.model_weights}")

# Add this at the END of ml/models/ensemble_model.py file:

# Create alias for backward compatibility
# Some tests/modules expect EnsembleModel instead of EnsemblePredictor
EnsembleModel = EnsemblePredictor