# tests/fixtures/test_helpers.py
"""
Test helper functions
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

class TestHelpers:
    """Helper functions for testing"""
    
    @staticmethod
    async def wait_for_condition(
        condition_func, 
        timeout: float = 5.0, 
        interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    def create_mock_web3_contract():
        """Create mock Web3 contract"""
        contract = MagicMock()
        
        # Mock common contract methods
        contract.functions.balanceOf = MagicMock()
        contract.functions.balanceOf().call = AsyncMock(return_value=1000000000000000000)
        
        contract.functions.totalSupply = MagicMock()
        contract.functions.totalSupply().call = AsyncMock(return_value=1000000000000000000000)
        
        contract.functions.decimals = MagicMock()
        contract.functions.decimals().call = AsyncMock(return_value=18)
        
        contract.functions.symbol = MagicMock()
        contract.functions.symbol().call = AsyncMock(return_value="TEST")
        
        contract.functions.name = MagicMock()
        contract.functions.name().call = AsyncMock(return_value="Test Token")
        
        return contract
    
    @staticmethod
    def assert_trade_valid(trade: Dict):
        """Assert trade data is valid"""
        assert "token" in trade
        assert "type" in trade
        assert trade["type"] in ["buy", "sell"]
        assert "amount" in trade
        assert isinstance(trade["amount"], (Decimal, int, float))
        assert trade["amount"] > 0
        assert "price" in trade
        assert isinstance(trade["price"], (Decimal, int, float))
        assert trade["price"] > 0
        assert "timestamp" in trade
        assert isinstance(trade["timestamp"], datetime)
    
    @staticmethod
    def assert_position_valid(position: Dict):
        """Assert position data is valid"""
        assert "token" in position
        assert "entry_price" in position
        assert isinstance(position["entry_price"], (Decimal, int, float))
        assert position["entry_price"] > 0
        assert "quantity" in position
        assert isinstance(position["quantity"], (Decimal, int, float))
        assert position["quantity"] > 0
        assert "status" in position
        assert position["status"] in ["open", "closed", "pending"]
    
    @staticmethod
    async def cleanup_test_data(db_manager, cache_manager):
        """Clean up test data after tests"""
        # Clean database
        if db_manager:
            async with db_manager.pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE trades, positions, market_data CASCADE")
        
        # Clean cache
        if cache_manager:
            await cache_manager.redis.flushdb()
    
    @staticmethod
    def compare_decimals(a: Decimal, b: Decimal, tolerance: Decimal = Decimal("0.0001")) -> bool:
        """Compare two decimal values with tolerance"""
        return abs(a - b) < tolerance
    
    @staticmethod
    def generate_mock_config() -> Dict:
        """Generate mock configuration"""
        return {
            "trading": {
                "max_position_size": 1000,
                "max_slippage": 0.05,
                "min_liquidity": 50000,
                "default_gas_price": 30,
                "max_gas_price": 200
            },
            "risk": {
                "max_daily_loss": 0.20,
                "max_position_pct": 0.10,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.20
            },
            "ml": {
                "min_confidence": 0.70,
                "retrain_interval": 86400,
                "feature_count": 50
            },
            "monitoring": {
                "alert_channels": ["telegram", "discord"],
                "metrics_interval": 60,
                "health_check_interval": 30
            }
        }
