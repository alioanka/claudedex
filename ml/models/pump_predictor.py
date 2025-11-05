# ml/models/pump_predictor.py

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import joblib

# Conditionally import ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    import lightgbm as lgb
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PumpPredictor:
    """
    Machine learning model for predicting cryptocurrency pump events.
    Uses ensemble of time-series models and technical indicators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ml_enabled = self.config.get('ml_enabled', False) and ML_LIBS_AVAILABLE
        self.models = {}
        self.scalers = {}
        self.sequence_length = 20  # Look back period for LSTM
        
        # Model weights for ensemble
        self.model_weights = {
            'lstm': 0.35,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'random_forest': 0.10,
            'gradient_boost': 0.10
        }
        
        # Feature configuration
        self.price_features = [
            'price', 'volume', 'liquidity', 'market_cap',
            'price_change_5m', 'price_change_15m', 'price_change_1h',
            'volume_change_5m', 'volume_change_1h'
        ]
        
        self.technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_lower', 'bollinger_width',
            'ema_9', 'ema_21', 'ema_50',
            'volume_weighted_price', 'obv', 'adx',
            'stochastic_k', 'stochastic_d'
        ]
        
        self.market_features = [
            'buy_pressure', 'sell_pressure', 'order_book_imbalance',
            'large_trades_ratio', 'whale_activity',
            'social_sentiment', 'search_trend', 'mention_velocity'
        ]
        
        # Initialize models
        self._initialize_models()
        
        # Model paths
        self.model_dir = Path(config.get('MODEL_DIR', './models/pump_predictor'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_models(self):
        """Initialize ensemble of prediction models."""
        if not self.ml_enabled:
            return
        
        # LSTM model for time-series prediction
        self.models['lstm'] = self._build_lstm_model()
        
        # XGBoost regressor
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
        
        # LightGBM regressor
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression',
            metric='rmse',
            random_state=42,
            verbose=-1
        )
        
        # Random Forest regressor
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting regressor
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Initialize scalers
        self.scalers['price'] = MinMaxScaler(feature_range=(0, 1))
        self.scalers['features'] = StandardScaler()
    
    def _build_lstm_model(self) -> Sequential:
        """Build LSTM neural network for pump prediction."""
        model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.price_features))),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')  # Output: pump probability
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def prepare_sequences(
        self,
        price_data: pd.DataFrame,
        lookback: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training/prediction."""
        if lookback is None:
            lookback = self.sequence_length
        
        # Select features
        data = price_data[self.price_features].values
        
        # Scale data
        scaled_data = self.scalers['price'].fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            
            # Target: significant price increase in next period
            future_price = price_data.iloc[i]['price']
            current_price = price_data.iloc[i-1]['price']
            pump_threshold = 1.10  # 10% increase
            
            y.append(1 if future_price > current_price * pump_threshold else 0)
        
        return np.array(X), np.array(y)
    
    def extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract all features for pump prediction."""
        features = []
        
        # Technical indicators
        features.extend(self._calculate_technical_indicators(market_data))
        
        # Market microstructure
        features.extend(self._calculate_market_features(market_data))
        
        # Pattern recognition
        features.extend(self._identify_pump_patterns(market_data))
        
        return np.array(features)
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> List[float]:
        """Calculate technical indicators."""
        indicators = []
        
        # RSI
        rsi = self._calculate_rsi(data['price'].values)
        indicators.append(rsi[-1] if len(rsi) > 0 else 50)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(data['price'].values)
        indicators.extend([
            macd[-1] if len(macd) > 0 else 0,
            signal[-1] if len(signal) > 0 else 0,
            histogram[-1] if len(histogram) > 0 else 0
        ])
        
        # Bollinger Bands
        upper, lower, width = self._calculate_bollinger_bands(data['price'].values)
        indicators.extend([upper, lower, width])
        
        # EMAs
        for period in [9, 21, 50]:
            ema = self._calculate_ema(data['price'].values, period)
            indicators.append(ema[-1] if len(ema) > 0 else data['price'].iloc[-1])
        
        # Volume indicators
        vwap = self._calculate_vwap(data)
        obv = self._calculate_obv(data)
        indicators.extend([vwap, obv])
        
        # ADX
        adx = self._calculate_adx(data)
        indicators.append(adx)
        
        # Stochastic
        k, d = self._calculate_stochastic(data)
        indicators.extend([k, d])
        
        return indicators
    
    def _calculate_market_features(self, data: pd.DataFrame) -> List[float]:
        """Calculate market microstructure features."""
        features = []
        
        # Buy/Sell pressure
        buy_pressure = data['buy_volume_5m'].iloc[-1] / max(data['volume_5m'].iloc[-1], 1)
        sell_pressure = data['sell_volume_5m'].iloc[-1] / max(data['volume_5m'].iloc[-1], 1)
        features.extend([buy_pressure, sell_pressure])
        
        # Order book imbalance (simulated)
        features.append(buy_pressure - sell_pressure)
        
        # Large trades ratio
        large_trades = data['whale_trades'].iloc[-1] if 'whale_trades' in data else 0
        total_trades = data['total_trades'].iloc[-1] if 'total_trades' in data else 1
        features.append(large_trades / max(total_trades, 1))
        
        # Whale activity
        whale_activity = data['whale_activity'].iloc[-1] if 'whale_activity' in data else 0
        features.append(whale_activity)
        
        # Social metrics
        features.append(data['social_sentiment'].iloc[-1] if 'social_sentiment' in data else 0.5)
        features.append(data['search_trend'].iloc[-1] if 'search_trend' in data else 0)
        features.append(data['mention_velocity'].iloc[-1] if 'mention_velocity' in data else 0)
        
        return features
    
    def _identify_pump_patterns(self, data: pd.DataFrame) -> List[float]:
        """Identify pump-related patterns in price/volume data."""
        patterns = []
        
        # Volume spike pattern
        recent_volume = data['volume_5m'].iloc[-5:].mean()
        historical_volume = data['volume_5m'].iloc[-50:-5].mean()
        volume_spike = recent_volume / max(historical_volume, 1)
        patterns.append(min(volume_spike, 10))  # Cap at 10x
        
        # Price momentum
        price_momentum = (data['price'].iloc[-1] - data['price'].iloc[-10]) / data['price'].iloc[-10]
        patterns.append(price_momentum)
        
        # Accumulation pattern
        accumulation_score = self._detect_accumulation(data)
        patterns.append(accumulation_score)
        
        # Breakout pattern
        breakout_score = self._detect_breakout(data)
        patterns.append(breakout_score)
        
        return patterns
    
    def _detect_accumulation(self, data: pd.DataFrame) -> float:
        """Detect accumulation pattern."""
        # Look for steady buying with minimal price movement
        if len(data) < 20:
            return 0.0
        
        recent_data = data.iloc[-20:]
        
        # Calculate metrics
        price_stability = 1 - recent_data['price'].std() / recent_data['price'].mean()
        volume_increase = recent_data['volume_5m'].iloc[-5:].mean() / recent_data['volume_5m'].iloc[:5].mean()
        buy_dominance = recent_data['buy_volume_5m'].sum() / recent_data['volume_5m'].sum()
        
        # Combine into score
        score = (price_stability * 0.3 + min(volume_increase, 2) * 0.3 + buy_dominance * 0.4)
        
        return min(score, 1.0)
    
    def _detect_breakout(self, data: pd.DataFrame) -> float:
        """Detect breakout pattern."""
        if len(data) < 50:
            return 0.0
        
        # Calculate resistance level
        resistance = data['price'].iloc[-50:-10].max()
        current_price = data['price'].iloc[-1]
        
        # Check for breakout
        if current_price > resistance * 1.02:  # 2% above resistance
            # Calculate breakout strength
            volume_confirmation = data['volume_5m'].iloc[-1] / data['volume_5m'].iloc[-50:].mean()
            price_strength = (current_price - resistance) / resistance
            
            score = min(volume_confirmation * 0.6 + price_strength * 10 * 0.4, 1.0)
            return score
        
        return 0.0
    
    def train(
        self,
        price_history: pd.DataFrame,
        pump_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train pump prediction models."""
        if not self.ml_enabled:
            logger.warning("ML is disabled. Skipping pump predictor training.")
            return {}
        logger.info("Starting pump predictor training...")
        
        # Prepare LSTM sequences
        X_lstm, y_lstm = self.prepare_sequences(price_history)
        
        # Prepare features for tree-based models
        X_features = []
        for i in range(self.sequence_length, len(price_history)):
            window_data = price_history.iloc[i-self.sequence_length:i]
            features = self.extract_features(window_data)
            X_features.append(features)
        
        X_features = np.array(X_features)
        y = y_lstm if pump_labels is None else pump_labels[self.sequence_length:]
        
        # Split data
        split_idx = int(len(X_lstm) * 0.8)
        
        # LSTM data
        X_lstm_train, X_lstm_val = X_lstm[:split_idx], X_lstm[split_idx:]
        
        # Features data
        X_train, X_val = X_features[:split_idx], X_features[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Results
        results = {'models': {}}
        
        # Train LSTM
        logger.info("Training LSTM model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        history = self.models['lstm'].fit(
            X_lstm_train, y_train,
            validation_data=(X_lstm_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        lstm_pred = self.models['lstm'].predict(X_lstm_val)
        results['models']['lstm'] = {
            'mse': mean_squared_error(y_val, lstm_pred),
            'mae': mean_absolute_error(y_val, lstm_pred)
        }
        
        # Train other models
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
            logger.info(f"Training {model_name}...")
            
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_val_scaled)
            
            results['models'][model_name] = {
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
        
        logger.info("Training complete")
        return results
    
    # ============================================
    # FIX 3: pump_predictor.py - PumpPredictor.predict_pump_probability
    # ============================================
    # Current signature: def predict_pump_probability(self, current_data: pd.DataFrame)
    # Expected signature: def predict_pump_probability(self, features: np.ndarray) -> float

    # Add both methods to support both signatures:

    def predict_pump_probability(
        self,
        features: np.ndarray
    ) -> float:
        """
        Predict pump probability from features (API-compliant signature)
        
        Args:
            features: Feature array
            
        Returns:
            Pump probability (0-1)
        """
        if not self.ml_enabled:
            return 0.5
        predictions = {}
        
        # LSTM prediction if we have sequence data
        if len(features.shape) == 3:  # Sequence data for LSTM
            lstm_prob = self.models['lstm'].predict(features[-1:])[0, 0]
            predictions['lstm'] = lstm_prob
        
        # Ensure 2D for tree models
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) == 3:
            # Use last timestep features for tree models
            features = features[-1].reshape(1, -1)
        
        # Feature-based predictions
        features_scaled = self.scalers['features'].transform(features)
        
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
            prob = self.models[model_name].predict(features_scaled)[0]
            predictions[model_name] = prob
        
        # Calculate ensemble probability
        ensemble_prob = sum(
            prob * self.model_weights.get(name, 0)
            for name, prob in predictions.items()
        )
        
        return ensemble_prob

    def predict_pump_probability_detailed(
        self,
        current_data: pd.DataFrame
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Detailed pump probability prediction with signals (original method)
        
        Args:
            current_data: DataFrame with current market data
            
        Returns:
            Tuple of (probability, individual_predictions, signals)
        """
        # This is the existing implementation, renamed for clarity
        predictions = {}
        
        # LSTM prediction
        if len(current_data) >= self.sequence_length:
            X_lstm, _ = self.prepare_sequences(current_data)
            if len(X_lstm) > 0:
                lstm_prob = self.models['lstm'].predict(X_lstm[-1:])[0, 0]
                predictions['lstm'] = lstm_prob
        
        # Feature-based predictions
        features = self.extract_features(current_data)
        features_scaled = self.scalers['features'].transform(features.reshape(1, -1))
        
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
            prob = self.models[model_name].predict(features_scaled)[0]
            predictions[model_name] = prob
        
        # Calculate ensemble probability
        ensemble_prob = sum(
            prob * self.model_weights.get(name, 0)
            for name, prob in predictions.items()
        )
        
        # Generate signals
        signals = self._generate_pump_signals(current_data, ensemble_prob)
        
        return ensemble_prob, predictions, signals

    
    def _generate_pump_signals(
        self,
        data: pd.DataFrame,
        pump_probability: float
    ) -> Dict[str, Any]:
        """Generate trading signals based on pump probability."""
        signals = {
            'pump_probability': pump_probability,
            'confidence': 'LOW',
            'action': 'HOLD',
            'reasons': []
        }
        
        # Determine confidence level
        if pump_probability > 0.8:
            signals['confidence'] = 'HIGH'
            signals['action'] = 'BUY'
            signals['reasons'].append("Very high pump probability")
        elif pump_probability > 0.6:
            signals['confidence'] = 'MEDIUM'
            signals['action'] = 'WATCH'
            signals['reasons'].append("Moderate pump probability")
        
        # Check supporting indicators
        if 'volume_spike' in data.columns and data['volume_spike'].iloc[-1] > 2:
            signals['reasons'].append("Significant volume spike detected")
            
        if 'whale_activity' in data.columns and data['whale_activity'].iloc[-1] > 0.7:
            signals['reasons'].append("High whale activity")
        
        if 'social_sentiment' in data.columns and data['social_sentiment'].iloc[-1] > 0.8:
            signals['reasons'].append("Very positive social sentiment")
        
        return signals
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 50
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100 - 100 / (1 + rs)
        
        return rsi
    
    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return prices[-1], prices[-1], 0
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = upper - lower
        
        return upper, lower, width
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate VWAP."""
        if 'volume' not in data.columns:
            return data['price'].iloc[-1]
        
        typical_price = (data['high'] + data['low'] + data['price']) / 3 if 'high' in data.columns else data['price']
        vwap = (typical_price * data['volume']).sum() / data['volume'].sum()
        return vwap
    
    def _calculate_obv(self, data: pd.DataFrame) -> float:
        """Calculate On-Balance Volume."""
        obv = 0
        for i in range(1, len(data)):
            if data['price'].iloc[i] > data['price'].iloc[i-1]:
                obv += data['volume'].iloc[i]
            elif data['price'].iloc[i] < data['price'].iloc[i-1]:
                obv -= data['volume'].iloc[i]
        
        return obv
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX."""
        if len(data) < period + 1:
            return 0
        
        # Simplified ADX calculation
        high = data['high'].values if 'high' in data.columns else data['price'].values
        low = data['low'].values if 'low' in data.columns else data['price'].values
        close = data['price'].values
        
        # Calculate directional movement
        plus_dm = np.zeros(len(data))
        minus_dm = np.zeros(len(data))
        
        for i in range(1, len(data)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        # Calculate ATR
        tr = np.zeros(len(data))
        for i in range(1, len(data)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = self._calculate_ema(tr, period)
        
        # Calculate DI
        plus_di = 100 * self._calculate_ema(plus_dm, period) / atr
        minus_di = 100 * self._calculate_ema(minus_dm, period) / atr
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = self._calculate_ema(dx, period)
        
        return adx[-1] if len(adx) > 0 else 0
    
    def _calculate_stochastic(
        self,
        data: pd.DataFrame,
        period: int = 14,
        smooth: int = 3
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(data) < period:
            return 50, 50
        
        high = data['high'].values if 'high' in data.columns else data['price'].values
        low = data['low'].values if 'low' in data.columns else data['price'].values
        close = data['price'].values
        
        # Calculate %K
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (moving average of %K)
        # Simplified: using current K as both K and D
        d = k
        
        return k, d
    
    def save_model(self, version: Optional[str] = None) -> str:
        """Save trained models to disk."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"pump_predictor_{version}"
        model_path.mkdir(exist_ok=True)
        
        # Save LSTM model
        self.models['lstm'].save(model_path / "lstm_model.h5")
        
        # Save tree-based models
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
            joblib.dump(self.models[model_name], model_path / f"{model_name}.pkl")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, model_path / f"scaler_{scaler_name}.pkl")
        
        # Save metadata
        metadata = {
            'model_weights': self.model_weights,
            'sequence_length': self.sequence_length,
            'price_features': self.price_features,
            'technical_features': self.technical_features,
            'market_features': self.market_features,
            'version': version,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(model_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Pump predictor saved to {model_path}")
        return str(model_path)
    
    def load_model(self, version: str) -> None:
        """Load trained models from disk."""
        model_path = self.model_dir / f"pump_predictor_{version}"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        # Load LSTM model
        self.models['lstm'] = load_model(model_path / "lstm_model.h5")
        
        # Load tree-based models
        for model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost']:
            model_file = model_path / f"{model_name}.pkl"
            if model_file.exists():
                self.models[model_name] = joblib.load(model_file)
        
        # Load scalers
        for scaler_name in self.scalers.keys():
            scaler_file = model_path / f"scaler_{scaler_name}.pkl"
            if scaler_file.exists():
                self.scalers[scaler_name] = joblib.load(scaler_file)
        
        # Load metadata
        with open(model_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.model_weights = metadata['model_weights']
            self.sequence_length = metadata['sequence_length']
            self.price_features = metadata['price_features']
            self.technical_features = metadata['technical_features']
            self.market_features = metadata['market_features']
        
        logger.info(f"Pump predictor loaded from {model_path}")

    # Add these methods to the PumpPredictor class:

    def prepare_features(self, market_data: Dict) -> np.ndarray:
        """
        Wrapper method matching documented signature
        Prepares features from market data dictionary
        """
        # Convert dict to DataFrame if needed
        if isinstance(market_data, dict):
            df = pd.DataFrame([market_data])
        else:
            df = market_data
        
        return self.extract_features(df)

    def train_lstm(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        """
        Train LSTM model directly with sequences
        Matches documented signature
        """
        # Configure callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Split for validation
        split_idx = int(len(sequences) * 0.8)
        X_train = sequences[:split_idx]
        X_val = sequences[split_idx:]
        y_train = targets[:split_idx]
        y_val = targets[split_idx:]
        
        # Train the LSTM model
        self.models['lstm'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM training completed")

    def backtest(self, historical_data: pd.DataFrame) -> Dict:
        """
        Backtest the pump prediction strategy
        
        Args:
            historical_data: Historical price/volume data
            
        Returns:
            Dictionary with backtest results
        """
        results = {
            'total_signals': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'roi': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades': []
        }
        
        # Prepare sequences for testing
        sequences, actual_pumps = self.prepare_sequences(historical_data)
        
        if len(sequences) < 100:
            logger.warning("Insufficient data for backtesting")
            return results
        
        # Track portfolio value
        initial_capital = 10000
        portfolio_value = initial_capital
        portfolio_history = []
        
        # Iterate through sequences
        for i in range(len(sequences) - 1):
            # Get prediction
            window_data = historical_data.iloc[i:i+self.sequence_length]
            pump_prob, _, signals = self.predict_pump_probability(window_data)
            
            # Check if we predicted a pump
            predicted_pump = pump_prob > 0.6
            actual_pump = actual_pumps[i] == 1
            
            results['total_signals'] += 1 if predicted_pump else 0
            
            # Track prediction accuracy
            if predicted_pump and actual_pump:
                results['correct_predictions'] += 1
                # Simulate profit (10% gain)
                portfolio_value *= 1.10
            elif predicted_pump and not actual_pump:
                results['false_positives'] += 1
                # Simulate loss (5% loss)
                portfolio_value *= 0.95
            elif not predicted_pump and actual_pump:
                results['false_negatives'] += 1
            
            portfolio_history.append(portfolio_value)
            
            # Record trade
            if predicted_pump:
                trade = {
                    'timestamp': historical_data.index[i] if historical_data.index.name else i,
                    'pump_probability': pump_prob,
                    'actual_pump': actual_pump,
                    'portfolio_value': portfolio_value
                }
                results['trades'].append(trade)
        
        # Calculate final metrics
        if results['total_signals'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_signals']
            results['precision'] = results['correct_predictions'] / (
                results['correct_predictions'] + results['false_positives'] + 0.0001
            )
            results['recall'] = results['correct_predictions'] / (
                results['correct_predictions'] + results['false_negatives'] + 0.0001
            )
        
        # Calculate ROI
        results['roi'] = ((portfolio_value - initial_capital) / initial_capital) * 100
        
        # Calculate Sharpe ratio
        if len(portfolio_history) > 1:
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
            if returns.std() > 0:
                results['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Calculate max drawdown
        peak = portfolio_history[0]
        max_dd = 0
        for value in portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        results['max_drawdown'] = max_dd * 100
        
        return results