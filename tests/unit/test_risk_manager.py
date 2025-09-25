# tests/unit/test_risk_manager.py
"""
Unit tests for RiskManager
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from core.risk_manager import RiskManager

@pytest.mark.unit
class TestRiskManager:
    """Test cases for RiskManager"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        config = {
            "max_position_size_pct": Decimal("0.05"),
            "max_position_per_token_pct": Decimal("0.10"),
            "max_daily_loss_pct": Decimal("0.20"),
            "max_correlation": Decimal("0.70"),
            "var_confidence": Decimal("0.95"),
            "default_stop_loss_pct": Decimal("0.05")
        }
        return RiskManager(config)
    
    @pytest.mark.asyncio
    async def test_check_position_limit(self, risk_manager):
        """Test position limit checking"""
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_active_positions.return_value = [
            {"token": "0xabc", "value": Decimal("1000")},
            {"token": "0xdef", "value": Decimal("2000")}
        ]
        risk_manager.portfolio_value = Decimal("100000")
        
        # Should allow new token
        result = await risk_manager.check_position_limit("0x123")
        assert result == True
        
        # Should block if too many positions in same token
        risk_manager.db_manager.get_active_positions.return_value = [
            {"token": "0xabc", "value": Decimal("10000")}
        ]
        result = await risk_manager.check_position_limit("0xabc")
        assert result == False
    
    @pytest.mark.asyncio
    async def test_calculate_position_size(self, risk_manager, sample_trading_opportunity):
        """Test position size calculation"""
        risk_manager.portfolio_value = Decimal("100000")
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_active_positions.return_value = []
        
        size = await risk_manager.calculate_position_size(sample_trading_opportunity)
        
        # Should not exceed max position size
        assert size <= risk_manager.portfolio_value * risk_manager.max_position_size_pct
        assert size > 0
    
    @pytest.mark.asyncio
    async def test_set_stop_loss(self, risk_manager, sample_position):
        """Test stop loss calculation"""
        # Normal volatility
        risk_manager.calculate_volatility = AsyncMock(return_value=Decimal("0.02"))
        
        stop_loss = await risk_manager.set_stop_loss(sample_position)
        expected = sample_position["entry_price"] * (1 - risk_manager.default_stop_loss_pct)
        
        assert stop_loss <= expected
        assert stop_loss > 0
        
        # High volatility - wider stop
        risk_manager.calculate_volatility = AsyncMock(return_value=Decimal("0.10"))
        
        stop_loss_high_vol = await risk_manager.set_stop_loss(sample_position)
        assert stop_loss_high_vol < stop_loss  # Wider stop for high volatility
    
    @pytest.mark.asyncio
    async def test_calculate_var(self, risk_manager):
        """Test Value at Risk calculation"""
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_historical_returns.return_value = [
            Decimal("0.01"), Decimal("-0.02"), Decimal("0.03"),
            Decimal("-0.01"), Decimal("0.02"), Decimal("-0.03"),
            Decimal("0.01"), Decimal("-0.01"), Decimal("0.02")
        ]
        risk_manager.portfolio_value = Decimal("100000")
        
        var = await risk_manager.calculate_var(confidence=0.95)
        
        assert var > 0
        assert var < risk_manager.portfolio_value
    
    @pytest.mark.asyncio
    async def test_check_correlation_limit(self, risk_manager):
        """Test correlation limit checking"""
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_active_positions.return_value = [
            {"token": "0xabc", "chain": "ethereum"}
        ]
        
        risk_manager.calculate_correlation = AsyncMock(return_value=Decimal("0.60"))
        
        # Should allow if correlation below limit
        result = await risk_manager.check_correlation_limit("0xdef")
        assert result == True
        
        # Should block if correlation too high
        risk_manager.calculate_correlation = AsyncMock(return_value=Decimal("0.80"))
        result = await risk_manager.check_correlation_limit("0xdef")
        assert result == False
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, risk_manager):
        """Test emergency stop conditions"""
        risk_manager.db_manager = AsyncMock()
        
        # Normal conditions
        risk_manager.calculate_daily_pnl = AsyncMock(return_value=Decimal("-0.10"))
        risk_manager.portfolio_value = Decimal("100000")
        
        should_stop = await risk_manager.emergency_stop_check()
        assert should_stop == False
        
        # Trigger emergency stop - high daily loss
        risk_manager.calculate_daily_pnl = AsyncMock(return_value=Decimal("-0.25"))
        
        should_stop = await risk_manager.emergency_stop_check()
        assert should_stop == True
    
    @pytest.mark.asyncio
    async def test_portfolio_exposure(self, risk_manager):
        """Test portfolio exposure calculation"""
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_active_positions.return_value = [
            {"token": "0xabc", "chain": "ethereum", "value": Decimal("5000")},
            {"token": "0xdef", "chain": "ethereum", "value": Decimal("3000")},
            {"token": "0x123", "chain": "bsc", "value": Decimal("2000")}
        ]
        risk_manager.portfolio_value = Decimal("100000")
        
        exposure = await risk_manager.check_portfolio_exposure()
        
        assert "total_exposure" in exposure
        assert "chain_exposure" in exposure
        assert exposure["total_exposure"] == Decimal("10000")
        assert exposure["chain_exposure"]["ethereum"] == Decimal("8000")
        assert exposure["chain_exposure"]["bsc"] == Decimal("2000")
    
    @pytest.mark.asyncio
    async def test_risk_adjusted_returns(self, risk_manager):
        """Test risk-adjusted return calculations"""
        risk_manager.db_manager = AsyncMock()
        risk_manager.db_manager.get_historical_returns.return_value = [
            Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"),
            Decimal("0.03"), Decimal("0.01"), Decimal("-0.02")
        ]
        
        # Calculate Sharpe ratio
        sharpe = await risk_manager.calculate_sharpe_ratio()
        assert sharpe is not None
        assert isinstance(sharpe, Decimal)
        
        # Calculate Sortino ratio
        sortino = await risk_manager.calculate_sortino_ratio()
        assert sortino is not None
        assert isinstance(sortino, Decimal)