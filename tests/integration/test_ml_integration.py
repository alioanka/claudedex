# tests/integration/test_ml_integration.py
"""
Integration tests for ML models with data pipeline
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

from ml.models.ensemble_model import EnsembleModel
from ml.models.rug_classifier import RugClassifier
from ml.models.pump_predictor import PumpPredictor
from data.storage.database import DatabaseManager

@pytest.mark.integration
@pytest.mark.requires_db
class TestMLIntegration:
    """Test ML model integration"""

    # Add mock_config fixture at the top of TestMLIntegration class (after line 15):
    @pytest.fixture
    def mock_config(self):
        """Mock ML configuration"""
        return {
            "ml": {
                "model_dir": "models/",
                "feature_dim": 100,
                "ensemble_weights": {
                    "xgboost": 0.3,
                    "lightgbm": 0.25,
                    "random_forest": 0.2
                }
            },
            "data": {
                "cache_ttl": 300
            }
        }
    
    @pytest.fixture
    async def training_data(self, db_manager):
        """Generate training data"""
        # Create synthetic training data
        tokens = []
        for i in range(100):
            is_rug = i % 10 == 0  # 10% rugs
            token = {
                "address": f"0x{'0' * 38}{i:02x}",
                "liquidity": Decimal("10000") if not is_rug else Decimal("1000"),
                "holders": 500 if not is_rug else 50,
                "dev_holdings_pct": Decimal("5") if not is_rug else Decimal("50"),
                "contract_verified": not is_rug,
                "liquidity_locked": not is_rug,
                "is_rug": is_rug
            }
            tokens.append(token)
            
            # Save to database
            await db_manager.save_token_analysis({
                "token": token["address"],
                "chain": "ethereum",
                "analysis": token,
                "timestamp": datetime.now()
            })
        
        return pd.DataFrame(tokens)
    
    @pytest.mark.asyncio
    async def test_rug_classifier_training(self, training_data, mock_config):
        """Test rug classifier training with real data"""
        classifier = RugClassifier(mock_config)
        
        # Prepare features and labels
        features = training_data.drop(["address", "is_rug"], axis=1)
        labels = training_data["is_rug"].astype(int).values
        
        # Train model
        metrics = classifier.train(features, labels)
        
        assert metrics["accuracy"] > 0.7
        assert metrics["precision"] > 0.6
        assert metrics["recall"] > 0.6
        
        # Test prediction
        sample_token = {
            "liquidity": 5000,
            "holders": 200,
            "dev_holdings_pct": 30,
            "contract_verified": False,
            "liquidity_locked": False
        }
        
        probability, analysis = classifier.predict(sample_token)
        assert 0 <= probability <= 1
        assert "risk_factors" in analysis
    
    @pytest.mark.asyncio
    async def test_pump_predictor_training(self, db_manager, mock_config):
        """Test pump predictor with historical data"""
        predictor = PumpPredictor(mock_config)
        
        # Generate price history
        price_data = []
        base_price = Decimal("0.001")
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(720):  # 30 days of hourly data
            # Simulate pump pattern
            if 200 <= i <= 220:  # Pump period
                price = base_price * Decimal(1 + (i - 200) * 0.1)
            else:
                price = base_price * Decimal(1 + np.random.normal(0, 0.02))
            
            price_data.append({
                "timestamp": base_time + timedelta(hours=i),
                "price": float(price),
                "volume": float(Decimal("10000") * (1 + abs(np.random.normal(0, 0.3))))
            })
        
        df = pd.DataFrame(price_data)
        
        # Train model
        metrics = predictor.train(df)
        
        assert "mse" in metrics
        assert metrics["mse"] < 0.1
        
        # Test prediction
        current_data = df.tail(24)  # Last 24 hours
        probability, prediction, confidence = predictor.predict_pump_probability(current_data)
        
        assert 0 <= probability <= 1
        assert "next_price" in prediction
        assert "time_to_pump" in confidence
    
    @pytest.mark.asyncio
    # Replace lines 120-125 with:
    async def test_ensemble_model(self, training_data, db_manager, mock_config):
        """Test ensemble model integration"""
        ensemble = EnsembleModel(model_dir="models/test/")
        await ensemble.load_models()
        
        # Initialize component models
        ensemble.models['rug_classifier'] = RugClassifier(mock_config)
        ensemble.models['pump_predictor'] = PumpPredictor(mock_config)
        
        # Mock some predictions
        token = "0x1234567890123456789012345678901234567890"
        
        with pytest.mock.patch.object(ensemble.rug_classifier, 'predict', 
                                     return_value=(0.2, {"risk": "low"})):
            with pytest.mock.patch.object(ensemble.pump_predictor, 'predict_pump_probability',
                                         return_value=(0.7, {"next_price": 0.0012}, {"confidence": 0.8})):
                
                # Get ensemble prediction
                prediction = await ensemble.predict(token, "ethereum")
                
                assert "action" in prediction
                assert "confidence" in prediction
                assert "signals" in prediction
                assert prediction["signals"]["rug_risk"] == 0.2
                assert prediction["signals"]["pump_probability"] == 0.7
                
                # Should recommend buy with low rug risk and high pump probability
                assert prediction["action"] in ["buy", "strong_buy"]
