# tests/unit/test_engine.py
"""
Unit tests for TradingBotEngine
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List

from core.engine import TradingBotEngine
from core.risk_manager import RiskManager

@pytest.mark.unit
class TestTradingBotEngine:
    """Test cases for TradingBotEngine"""
    
    @pytest.fixture
    def engine(self, risk_manager):
        """Create engine instance for testing"""
        engine = TradingBotEngine()
        engine.risk_manager = risk_manager
        engine.db_manager = AsyncMock()
        engine.cache_manager = AsyncMock()
        engine.order_manager = AsyncMock()
        engine.position_tracker = AsyncMock()
        engine.alert_system = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.is_running == False
        assert engine.active_tasks == []
        assert engine.shutdown_event is not None
        assert engine.risk_manager is not None
    
    @pytest.mark.asyncio
    async def test_start_stop(self, engine):
        """Test engine start and stop"""
        # Mock dependencies
        engine._monitor_new_pairs = AsyncMock()
        engine._monitor_existing_positions = AsyncMock()
        
        # Start engine
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)  # Let it start
        
        assert engine.is_running == True
        assert len(engine.active_tasks) > 0
        
        # Stop engine
        await engine.stop()
        assert engine.is_running == False
    
    @pytest.mark.asyncio
    async def test_analyze_opportunity(self, engine, mock_dex_api):
        """Test opportunity analysis"""
        engine.honeypot_checker = AsyncMock()
        engine.honeypot_checker.check_token.return_value = {
            "is_honeypot": False,
            "buy_tax": Decimal("1"),
            "sell_tax": Decimal("1")
        }
        
        engine.rug_detector = AsyncMock()
        engine.rug_detector.detect_rug.return_value = {
            "is_rug": False,
            "confidence": 0.1,
            "red_flags": []
        }
        
        engine.pump_predictor = AsyncMock()
        engine.pump_predictor.predict_pump.return_value = {
            "pump_probability": 0.75,
            "expected_return": Decimal("0.25"),
            "confidence": 0.80
        }
        
        engine.token_scorer = AsyncMock()
        engine.token_scorer.calculate_composite_score.return_value = {
            "total_score": 85,
            "components": {
                "technical": 80,
                "fundamental": 85,
                "social": 90
            }
        }
        
        opportunity = await engine._analyze_opportunity(mock_dex_api["new_pairs"][0])
        
        assert opportunity is not None
        assert opportunity.token == mock_dex_api["new_pairs"][0]["address"]
        assert opportunity.score > 0
        assert opportunity.confidence > 0
    
    @pytest.mark.asyncio
    async def test_execute_opportunity(self, engine, sample_trading_opportunity):
        """Test opportunity execution"""
        engine.risk_manager.calculate_position_size = AsyncMock(
            return_value=Decimal("1000")
        )
        engine.order_manager.create_order = AsyncMock(
            return_value="order_123"
        )
        engine.order_manager.execute_order = AsyncMock(
            return_value=True
        )
        engine.position_tracker.open_position = AsyncMock(
            return_value="pos_123"
        )
        
        result = await engine._execute_opportunity(sample_trading_opportunity)
        
        engine.order_manager.create_order.assert_called_once()
        engine.order_manager.execute_order.assert_called_once()
        engine.position_tracker.open_position.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitor_positions(self, engine, sample_position):
        """Test position monitoring"""
        engine.position_tracker.get_active_positions = AsyncMock(
            return_value=[sample_position]
        )
        engine.position_tracker.calculate_pnl = AsyncMock(
            return_value={"pnl": Decimal("100"), "pnl_percentage": Decimal("10")}
        )
        engine.position_tracker.check_stop_loss = AsyncMock(
            return_value=False
        )
        engine.position_tracker.check_take_profit = AsyncMock(
            return_value=True
        )
        engine._close_position = AsyncMock()
        
        await engine._monitor_existing_positions()
        
        engine._close_position.assert_called_once_with(
            sample_position, 
            "take_profit"
        )
    
    @pytest.mark.asyncio
    async def test_safety_checks(self, engine, sample_trading_opportunity):
        """Test final safety checks"""
        engine.risk_manager.check_position_limit = AsyncMock(return_value=True)
        engine.risk_manager.check_correlation_limit = AsyncMock(return_value=True)
        engine.risk_manager.emergency_stop_check = AsyncMock(return_value=False)
        
        # Should pass all checks
        result = await engine._final_safety_checks(sample_trading_opportunity)
        assert result == True
        
        # Test with emergency stop
        engine.risk_manager.emergency_stop_check = AsyncMock(return_value=True)
        result = await engine._final_safety_checks(sample_trading_opportunity)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_close_position(self, engine, sample_position):
        """Test position closing"""
        engine.order_manager.create_order = AsyncMock(return_value="order_456")
        engine.order_manager.execute_order = AsyncMock(return_value=True)
        engine.position_tracker.close_position = AsyncMock(
            return_value={"final_pnl": Decimal("100")}
        )
        engine.db_manager.update_position = AsyncMock(return_value=True)
        
        await engine._close_position(sample_position, "manual_close")
        
        engine.order_manager.create_order.assert_called_once()
        engine.position_tracker.close_position.assert_called_once()
        engine.db_manager.update_position.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in engine"""
        engine._monitor_new_pairs = AsyncMock(side_effect=Exception("API Error"))
        engine.alert_system.send_alert = AsyncMock()
        
        # Start engine with error
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)
        
        # Should handle error and alert
        engine.alert_system.send_alert.assert_called()
        
        await engine.stop()


