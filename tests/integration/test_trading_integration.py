# tests/integration/test_trading_integration.py
"""
Integration tests for trading execution
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from trading.orders.order_manager import OrderManager
from trading.orders.position_tracker import PositionTracker
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.scalping import ScalpingStrategy

@pytest.mark.integration
class TestTradingIntegration:
    """Test trading system integration"""
    
    @pytest.mark.asyncio
    async def test_order_execution_flow(self, db_manager):
        """Test complete order execution flow"""
        order_manager = OrderManager()
        position_tracker = PositionTracker()
        
        order_manager.db_manager = db_manager
        position_tracker.db_manager = db_manager
        
        # Mock DEX executor
        order_manager.executor = AsyncMock()
        order_manager.executor.execute_swap = AsyncMock(return_value={
            "status": "success",
            "tx_hash": "0xabc123",
            "executed_price": Decimal("0.001"),
            "executed_amount": Decimal("1000"),
            "gas_used": 150000
        })
        
        # Create order
        order = {
            "token": "0x1234567890123456789012345678901234567890",
            "chain": "ethereum",
            "type": "buy",
            "amount": Decimal("1000"),
            "price": Decimal("0.001"),
            "slippage": Decimal("0.01")
        }
        
        order_id = await order_manager.create_order(order)
        assert order_id is not None
        
        # Execute order
        success = await order_manager.execute_order(order_id)
        assert success == True
        
        # Open position
        position = {
            "token": order["token"],
            "chain": order["chain"],
            "entry_price": Decimal("0.001"),
            "quantity": Decimal("1000"),
            "stop_loss": Decimal("0.00095"),
            "take_profit": Decimal("0.0012")
        }
        
        position_id = await position_tracker.open_position(position)
        assert position_id is not None
        
        # Check position
        active_position = await position_tracker.get_position(position_id)
        assert active_position is not None
        assert active_position["status"] == "open"
    
    @pytest.mark.asyncio
    async def test_strategy_signal_execution(self, db_manager):
        """Test strategy signal to execution"""
        momentum = MomentumStrategy()
        order_manager = OrderManager()
        
        momentum.db_manager = db_manager
        order_manager.db_manager = db_manager
        order_manager.executor = AsyncMock()
        
        # Mock market data
        market_data = {
            "token": "0x1234567890123456789012345678901234567890",
            "price": Decimal("0.001"),
            "volume_24h": Decimal("100000"),
            "price_change_24h": Decimal("25"),
            "rsi": 65,
            "macd_signal": "bullish"
        }
        
        # Generate signal
        signal = await momentum.analyze(market_data)
        
        if signal["action"] == "buy":
            # Create order from signal
            order = {
                "token": market_data["token"],
                "type": "buy",
                "amount": signal["suggested_amount"],
                "price": market_data["price"]
            }
            
            order_id = await order_manager.create_order(order)
            assert order_id is not None
    
    @pytest.mark.asyncio
    async def test_position_monitoring_and_exit(self, db_manager):
        """Test position monitoring and exit conditions"""
        position_tracker = PositionTracker()
        order_manager = OrderManager()
        
        position_tracker.db_manager = db_manager
        order_manager.db_manager = db_manager
        
        # Create position
        position = {
            "token": "0x1234567890123456789012345678901234567890",
            "chain": "ethereum",
            "entry_price": Decimal("0.001"),
            "quantity": Decimal("1000"),
            "stop_loss": Decimal("0.00095"),
            "take_profit": Decimal("0.0012"),
            "entry_time": datetime.now()
        }
        
        position_id = await position_tracker.open_position(position)
        
        # Simulate price movement - hit take profit
        await position_tracker.update_position(position_id, {
            "current_price": Decimal("0.0012")
        })
        
        # Check take profit
        should_exit = await position_tracker.check_take_profit(
            await position_tracker.get_position(position_id)
        )
        assert should_exit == True
        
        # Close position
        result = await position_tracker.close_position(position_id, "take_profit")
        assert result["final_pnl"] > 0
        assert result["status"] == "closed"