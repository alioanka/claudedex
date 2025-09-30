# tests/conftest.py
"""
Global pytest configuration and fixtures
"""
import asyncio
import pytest
import aioredis
import asyncpg
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock, MagicMock
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import TradingBotEngine
from core.risk_manager import RiskManager
from data.storage.database import DatabaseManager
from data.storage.cache import CacheManager
from config.config_manager import ConfigManager
from security.wallet_security import WalletSecurityManager
from security.audit_logger import AuditLogger

# Test configuration
TEST_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "tradingbot_test",
        "user": "test_user",
        "password": "test_password"
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 1
    },
    "trading": {
        "max_position_size": Decimal("1000"),
        "max_slippage": Decimal("0.05"),
        "min_liquidity": Decimal("50000")
    }
}

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def db_manager():
    """Database manager fixture"""
    manager = DatabaseManager(TEST_CONFIG["database"])
    await manager.connect()
    
    # Clean test database
    async with manager.pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE trades, positions, market_data, token_analysis CASCADE")
    
    yield manager
    await manager.disconnect()

@pytest.fixture
async def cache_manager():
    """Cache manager fixture"""
    manager = CacheManager(TEST_CONFIG["redis"])
    await manager.connect()
    
    # Clean test cache
    await manager.redis.flushdb()
    
    yield manager
    await manager.disconnect()

# Replace lines 88-92 with:
@pytest.fixture
async def config_manager(tmp_path):
    """Configuration manager fixture"""
    config_path = tmp_path / "configs"
    config_path.mkdir()
    
    manager = ConfigManager(config_dir=str(config_path))
    # Initialize with encryption key as a separate call
    await manager.initialize(encryption_key="test_key_32_bytes_long_for_test!")
    
    yield manager

@pytest.fixture
def risk_manager():
    """Risk manager fixture"""
    return RiskManager(TEST_CONFIG["trading"])

# Replace lines 99-103 with:
@pytest.fixture
async def wallet_security(config_manager):
    """Wallet security manager fixture"""
    # Pass config object, not empty call
    config = await config_manager.get_security_config()
    manager = WalletSecurityManager(config)
    await manager.initialize()
    yield manager
    await manager.cleanup()

# Replace lines 105-114 with:
@pytest.fixture
async def audit_logger(tmp_path, config_manager):
    """Audit logger fixture"""
    # Create config dict with required parameters
    config = {
        "log_dir": str(tmp_path / "audit"),
        "retention_days": 30,
        "buffer_size": 100,
        "compress_old_logs": False
    }
    
    logger = AuditLogger(config)
    await logger.initialize()
    yield logger
    await logger.cleanup()
@pytest.fixture
def mock_dex_api():
    """Mock DEX API responses"""
    return {
        "new_pairs": [
            {
                "address": "0x1234567890123456789012345678901234567890",
                "name": "TestToken",
                "symbol": "TEST",
                "liquidity": "100000",
                "volume24h": "50000",
                "priceUsd": "0.001",
                "chain": "ethereum"
            }
        ],
        "price_data": {
            "price": "0.001",
            "price_change_24h": "15.5",
            "volume_24h": "50000",
            "liquidity": "100000"
        },
        "honeypot_check": {
            "is_honeypot": False,
            "buy_tax": "1",
            "sell_tax": "1",
            "can_sell": True
        }
    }

@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        "token_data": {
            "address": "0x1234567890123456789012345678901234567890",
            "price": Decimal("0.001"),
            "volume_24h": Decimal("50000"),
            "liquidity": Decimal("100000"),
            "holders": 500,
            "market_cap": Decimal("1000000")
        },
        "price_history": [
            {"timestamp": datetime.now() - timedelta(hours=i), "price": Decimal("0.001") * (1 + i/100)}
            for i in range(24)
        ],
        "trades": [
            {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "type": "buy" if i % 2 == 0 else "sell",
                "amount": Decimal("1000"),
                "price": Decimal("0.001")
            }
            for i in range(100)
        ]
    }

@pytest.fixture
def sample_trading_opportunity():
    """Sample trading opportunity for testing"""
    return {
        "token": "0x1234567890123456789012345678901234567890",
        "chain": "ethereum",
        "score": 85.5,
        "confidence": 0.75,
        "expected_return": Decimal("0.25"),
        "risk_level": "medium",
        "entry_price": Decimal("0.001"),
        "target_price": Decimal("0.00125"),
        "stop_loss": Decimal("0.00095"),
        "position_size": Decimal("1000"),
        "signals": {
            "pump_probability": 0.65,
            "rug_probability": 0.10,
            "liquidity_score": 0.80
        }
    }

@pytest.fixture
def sample_position():
    """Sample position for testing"""
    return {
        "id": "pos_123456",
        "token": "0x1234567890123456789012345678901234567890",
        "chain": "ethereum",
        "entry_price": Decimal("0.001"),
        "current_price": Decimal("0.0011"),
        "quantity": Decimal("1000"),
        "entry_time": datetime.now(),
        "stop_loss": Decimal("0.00095"),
        "take_profit": Decimal("0.00125"),
        "pnl": Decimal("100"),
        "pnl_percentage": Decimal("10"),
        "status": "open"
    }

# Async fixtures for mocking external services
@pytest.fixture
async def mock_web3():
    """Mock Web3 connection"""
    web3 = AsyncMock()
    web3.eth.get_block_number = AsyncMock(return_value=1000000)
    web3.eth.get_gas_price = AsyncMock(return_value=30000000000)
    web3.eth.get_balance = AsyncMock(return_value=1000000000000000000)
    return web3

@pytest.fixture
async def mock_dex_client():
    """Mock DEX client"""
    client = AsyncMock()
    client.get_new_pairs = AsyncMock(return_value=[
        {"address": "0xabc", "liquidity": "100000", "volume24h": "50000"}
    ])
    client.get_token_price = AsyncMock(return_value=Decimal("0.001"))
    client.execute_swap = AsyncMock(return_value={"status": "success", "tx_hash": "0x123"})
    return client

# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Large dataset for performance testing"""
    return {
        "large_price_history": [
            {"timestamp": datetime.now() - timedelta(seconds=i), "price": Decimal("0.001") * (1 + i/10000)}
            for i in range(100000)
        ],
        "many_tokens": [
            f"0x{'0' * 38}{i:02x}" for i in range(1000)
        ],
        "bulk_trades": [
            {
                "id": f"trade_{i}",
                "token": f"0x{'0' * 38}{i % 100:02x}",
                "type": "buy" if i % 2 == 0 else "sell",
                "amount": Decimal("1000"),
                "price": Decimal("0.001") * (1 + i/1000)
            }
            for i in range(10000)
        ]
    }

# Helper functions for tests
def create_mock_token(address: str = None, **kwargs) -> Dict:
    """Create a mock token with default values"""
    default = {
        "address": address or "0x1234567890123456789012345678901234567890",
        "name": "TestToken",
        "symbol": "TEST",
        "decimals": 18,
        "total_supply": "1000000000",
        "liquidity": "100000",
        "volume24h": "50000",
        "price": "0.001",
        "holders": 500
    }
    default.update(kwargs)
    return default

def create_mock_trade(**kwargs) -> Dict:
    """Create a mock trade with default values"""
    default = {
        "id": "trade_123",
        "token": "0x1234567890123456789012345678901234567890",
        "type": "buy",
        "amount": Decimal("1000"),
        "price": Decimal("0.001"),
        "total": Decimal("1"),
        "timestamp": datetime.now(),
        "status": "completed",
        "tx_hash": "0xabc123"
    }
    default.update(kwargs)
    return default

async def populate_test_database(db_manager: DatabaseManager, num_records: int = 100):
    """Populate database with test data"""
    for i in range(num_records):
        await db_manager.save_trade(create_mock_trade(id=f"trade_{i}"))
        
        if i % 10 == 0:
            await db_manager.save_position({
                "id": f"pos_{i}",
                "token": f"0x{'0' * 38}{i:02x}",
                "entry_price": Decimal("0.001"),
                "quantity": Decimal("1000"),
                "status": "open" if i % 2 == 0 else "closed"
            })

# Cleanup fixture
@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after each test"""
    yield
    # Add any cleanup logic here