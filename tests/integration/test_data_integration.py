# tests/integration/test_data_integration.py
"""
Integration tests for data collection and storage
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import aiohttp

from data.collectors.dexscreener import DexScreenerCollector
from data.collectors.honeypot_checker import HoneypotChecker
from data.collectors.whale_tracker import WhaleTracker
from data.storage.database import DatabaseManager
from data.storage.cache import CacheManager

@pytest.mark.integration
@pytest.mark.requires_db
class TestDataIntegration:
    """Integration tests for data flow"""
    
    @pytest.mark.asyncio
    async def test_dexscreener_to_database(self, db_manager, mock_dex_api):
        """Test data flow from DexScreener to database"""
        collector = DexScreenerCollector()
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value=mock_dex_api)
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Collect data
            pairs = await collector.get_new_pairs("ethereum")
            
            # Save to database
            for pair in pairs:
                await db_manager.save_market_data({
                    "token": pair["address"],
                    "chain": "ethereum",
                    "price": pair["priceUsd"],
                    "volume": pair["volume24h"],
                    "liquidity": pair["liquidity"],
                    "timestamp": datetime.now()
                })
            
            # Verify data was saved
            historical = await db_manager.get_historical_data(
                pairs[0]["address"], 
                "1m"
            )
            assert len(historical) > 0
            assert historical[0]["token"] == pairs[0]["address"]
    
    @pytest.mark.asyncio
    async def test_honeypot_checker_caching(self, cache_manager, mock_dex_api):
        """Test honeypot checker with caching"""
        checker = HoneypotChecker()
        checker.cache_manager = cache_manager
        
        token = "0x1234567890123456789012345678901234567890"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(
                return_value=mock_dex_api["honeypot_check"]
            )
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # First call - should hit API
            result1 = await checker.check_token(token, "ethereum")
            assert result1["is_honeypot"] == False
            
            # Second call - should use cache
            result2 = await checker.check_token(token, "ethereum")
            assert result2 == result1
            
            # Verify only one API call was made
            assert mock_get.call_count == 1
    
    @pytest.mark.asyncio
    async def test_whale_tracker_integration(self, db_manager, cache_manager):
        """Test whale tracker with storage integration"""
        tracker = WhaleTracker()
        tracker.db_manager = db_manager
        tracker.cache_manager = cache_manager
        
        token = "0x1234567890123456789012345678901234567890"
        
        # Mock whale movements
        whale_data = {
            "movements": [
                {
                    "wallet": "0xwhale1",
                    "type": "buy",
                    "amount": Decimal("1000000"),
                    "timestamp": datetime.now()
                },
                {
                    "wallet": "0xwhale2",
                    "type": "sell",
                    "amount": Decimal("500000"),
                    "timestamp": datetime.now()
                }
            ]
        }
        
        with patch.object(tracker, 'fetch_whale_data', return_value=whale_data):
            movements = await tracker.track_whale_movements(token, "ethereum")
            
            # Save to database
            for movement in movements["movements"]:
                await db_manager.save_whale_activity({
                    "token": token,
                    "wallet": movement["wallet"],
                    "type": movement["type"],
                    "amount": movement["amount"],
                    "timestamp": movement["timestamp"]
                })
            
            # Cache the analysis
            await cache_manager.set(
                f"whale_analysis:{token}",
                {"score": 75, "trend": "accumulation"},
                ttl=300
            )
            
            # Verify storage
            cached = await cache_manager.get(f"whale_analysis:{token}")
            assert cached["score"] == 75
            assert cached["trend"] == "accumulation"
    
    @pytest.mark.asyncio
    async def test_data_aggregation_pipeline(self, db_manager, cache_manager):
        """Test complete data aggregation pipeline"""
        from data.processors.aggregator import DataAggregator
        
        aggregator = DataAggregator(db_manager, cache_manager)
        
        # Simulate multiple data sources
        token = "0x1234567890123456789012345678901234567890"
        
        # Mock data from various sources
        await db_manager.save_market_data({
            "token": token,
            "price": Decimal("0.001"),
            "volume": Decimal("50000"),
            "timestamp": datetime.now()
        })
        
        await cache_manager.set(f"honeypot:{token}", {
            "is_honeypot": False,
            "buy_tax": 1,
            "sell_tax": 1
        })
        
        await cache_manager.set(f"whale_analysis:{token}", {
            "score": 80,
            "trend": "accumulation"
        })
        
        # Aggregate all data
        aggregated = await aggregator.aggregate_token_data(token, "ethereum")
        
        assert aggregated["token"] == token
        assert "market_data" in aggregated
        assert "honeypot_status" in aggregated
        assert "whale_analysis" in aggregated
        assert aggregated["whale_analysis"]["score"] == 80
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, db_manager):
        """Test batch data processing"""
        # Create batch data
        batch_data = []
        base_time = datetime.now()
        
        for i in range(1000):
            batch_data.append({
                "token": f"0x{'0' * 38}{i:02x}",
                "price": Decimal("0.001") * (1 + i/1000),
                "volume": Decimal("1000") * (1 + i/100),
                "timestamp": base_time - timedelta(minutes=i)
            })
        
        # Test batch insert
        start_time = datetime.now()
        await db_manager.save_market_data_batch(batch_data)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Should be fast (under 2 seconds for 1000 records)
        assert duration < 2.0
        
        # Verify data was saved
        sample_token = batch_data[0]["token"]
        data = await db_manager.get_historical_data(sample_token, "1m")
        assert len(data) > 0
