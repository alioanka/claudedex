# tests/performance/test_performance.py
"""
Performance tests for trading bot
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile

from core.engine import TradingBotEngine
from data.storage.database import DatabaseManager
from data.storage.cache import CacheManager
from ml.models.rug_classifier import RugClassifier

@pytest.mark.performance
class TestPerformance:
    """Performance test suite"""
    
    @pytest.mark.asyncio
    async def test_database_write_performance(self, db_manager, benchmark_data):
        """Test database write performance"""
        start_time = time.time()
        
        # Test bulk insert performance
        await db_manager.save_market_data_batch(benchmark_data["bulk_trades"][:1000])
        
        duration = time.time() - start_time
        
        # Should handle 1000 records in under 1 second
        assert duration < 1.0
        
        # Calculate throughput
        throughput = 1000 / duration
        print(f"Database write throughput: {throughput:.2f} records/second")
        assert throughput > 1000  # At least 1000 records/second
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, cache_manager):
        """Test Redis cache performance"""
        # Test write performance
        start_time = time.time()
        
        for i in range(10000):
            await cache_manager.set(f"test_key_{i}", {"value": i}, ttl=60)
        
        write_duration = time.time() - start_time
        write_throughput = 10000 / write_duration
        
        print(f"Cache write throughput: {write_throughput:.2f} ops/second")
        assert write_throughput > 5000  # At least 5000 ops/second
        
        # Test read performance
        start_time = time.time()
        
        for i in range(10000):
            await cache_manager.get(f"test_key_{i}")
        
        read_duration = time.time() - start_time
        read_throughput = 10000 / read_duration
        
        print(f"Cache read throughput: {read_throughput:.2f} ops/second")
        assert read_throughput > 10000  # At least 10000 ops/second
    
    @pytest.mark.asyncio
    async def test_ml_model_performance(self, benchmark_data):
        """Test ML model inference performance"""
        classifier = RugClassifier()
        
        # Create test data
        test_tokens = []
        for i in range(1000):
            test_tokens.append({
                "liquidity": Decimal("10000") * (1 + i/100),
                "holders": 100 + i,
                "dev_holdings_pct": Decimal("5") + Decimal(i/100),
                "contract_verified": i % 2 == 0,
                "liquidity_locked": i % 3 == 0
            })
        
        # Test inference speed
        start_time = time.time()
        
        predictions = []
        for token in test_tokens:
            prob, analysis = classifier.predict(token)
            predictions.append(prob)
        
        duration = time.time() - start_time
        throughput = 1000 / duration
        
        print(f"ML inference throughput: {throughput:.2f} predictions/second")
        assert throughput > 100  # At least 100 predictions/second
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, db_manager, cache_manager):
        """Test concurrent request handling"""
        async def process_request(token_id: int):
            """Simulate processing a trading opportunity"""
            token = f"0x{'0' * 38}{token_id:02x}"
            
            # Database operations
            await db_manager.save_market_data({
                "token": token,
                "price": Decimal("0.001"),
                "volume": Decimal("10000"),
                "timestamp": datetime.now()
            })
            
            # Cache operations
            await cache_manager.set(f"token:{token}", {"processed": True}, ttl=60)
            
            return token
        
        # Test concurrent processing
        start_time = time.time()
        
        tasks = []
        for i in range(100):
            tasks.append(process_request(i))
        
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        print(f"Concurrent processing: {100/duration:.2f} requests/second")
        assert len(results) == 100
        assert duration < 5.0  # Should handle 100 concurrent requests in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, benchmark_data):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_data = []
        for i in range(100000):
            large_data.append({
                "id": i,
                "price": Decimal("0.001") * (1 + i/10000),
                "volume": Decimal("1000") * (1 + i/1000),
                "timestamp": datetime.now()
            })
        
        # Process data
        processed = []
        for item in large_data:
            processed.append(item["price"] * item["volume"])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB, Increase={memory_increase:.2f}MB")
        
        # Should not use more than 500MB for 100k records
        assert memory_increase < 500
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="order_execution")
    async def test_order_execution_latency(self, benchmark):
        """Benchmark order execution latency"""
        from trading.orders.order_manager import OrderManager
        
        order_manager = OrderManager()
        order_manager.executor = AsyncMock()
        order_manager.executor.execute_swap = AsyncMock(return_value={
            "status": "success",
            "tx_hash": "0xabc123"
        })
        
        order = {
            "token": "0x1234567890123456789012345678901234567890",
            "type": "buy",
            "amount": Decimal("1000"),
            "price": Decimal("0.001")
        }
        
        async def execute_order():
            order_id = await order_manager.create_order(order)
            await order_manager.execute_order(order_id)
        
        # Benchmark execution
        result = benchmark(execute_order)
        
        # Should execute in under 100ms
        assert benchmark.stats["mean"] < 0.1
