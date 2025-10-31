# Create test_patches.py
import asyncio
from core.engine import TradingBotEngine

async def test():
    config = {...}  # Your config
    engine = TradingBotEngine(config)
    await engine.initialize()
    
    # Check db_manager
    assert engine.trade_executor.db_manager is not None
    assert engine.order_manager.db_manager is not None
    
    # Test validation
    order = {
        'chain': 'ethereum',
        'token_address': '0x' + '0' * 40,
        'side': 'buy',
        'amount': 10.0
    }
    valid, error = engine.trade_executor.validate_order_params(order)
    assert valid, error
    
    print("✅ All tests passed!")

asyncio.run(test())