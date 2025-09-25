# tests/smoke/test_smoke.py
"""
Smoke tests for quick validation of core functionality
"""
import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

@pytest.mark.smoke
class TestSmoke:
    """Quick validation tests"""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, db_manager):
        """Test database connectivity"""
        # Simple connection test
        result = await db_manager.execute("SELECT 1")
        assert result is not None
        
        # Test basic CRUD
        trade = {
            "token": "0xtest",
            "type": "buy",
            "amount": Decimal("100"),
            "price": Decimal("0.001"),
            "timestamp": datetime.now()
        }
        trade_id = await db_manager.save_trade(trade)
        assert trade_id is not None
    
    @pytest.mark.asyncio
    async def test_cache_connection(self, cache_manager):
        """Test Redis connectivity"""
        # Simple connection test
        await cache_manager.set("test_key", "test_value", ttl=60)
        value = await cache_manager.get("test_key")
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_config_loading(self, config_manager):
        """Test configuration loading"""
        trading_config = config_manager.get_trading_config()
        assert trading_config is not None
        assert trading_config.max_position_size > 0
        
        security_config = config_manager.get_security_config()
        assert security_config is not None
        assert security_config.require_2fa == True
    
    @pytest.mark.asyncio
    async def test_ml_model_loading(self):
        """Test ML model initialization"""
        from ml.models.rug_classifier import RugClassifier
        
        classifier = RugClassifier()
        assert classifier is not None
        
        # Test simple prediction
        token_data = {
            "liquidity": 10000,
            "holders": 100,
            "dev_holdings_pct": 10,
            "contract_verified": True,
            "liquidity_locked": True
        }
        
        probability, analysis = classifier.predict(token_data)
        assert 0 <= probability <= 1
    
    @pytest.mark.asyncio
    async def test_api_endpoints_available(self):
        """Test external API availability"""
        import aiohttp
        
        endpoints = [
            "https://api.dexscreener.com/latest/dex/search",
            "https://api.coingecko.com/api/v3/ping"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(endpoint, timeout=5) as response:
                        assert response.status in [200, 403, 429]  # OK or rate limited
                except:
                    pytest.skip(f"API {endpoint} not available")
    
    @pytest.mark.asyncio
    async def test_engine_startup(self, risk_manager):
        """Test engine basic startup"""
        from core.engine import TradingBotEngine
        
        engine = TradingBotEngine()
        engine.risk_manager = risk_manager
        
        # Mock dependencies
        engine.db_manager = AsyncMock()
        engine.cache_manager = AsyncMock()
        
        assert engine is not None
        assert engine.is_running == False
        
        # Test can start and stop
        engine._monitor_new_pairs = AsyncMock()
        engine._monitor_existing_positions = AsyncMock()
        
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)
        assert engine.is_running == True
        
        await engine.stop()
        assert engine.is_running == False
