# test_dexscreener.py
import asyncio
from data.collectors.dexscreener import DexScreenerCollector

async def test():
    config = {
        'api_key': '',  # Empty = no authentication
        'chains': ['ethereum', 'base'],
        'rate_limit': 100,
        'min_liquidity': 10000
    }
    
    collector = DexScreenerCollector(config)
    await collector.initialize()
    
    try:
        print("Testing DexScreener without API key...")
        
        # Test 1: Get new pairs
        pairs = await collector.get_new_pairs('ethereum', limit=5)
        print(f"✓ Found {len(pairs)} new pairs")
        
        # Test 2: Get token price
        if pairs:
            token_addr = pairs[0]['token_address']
            price = await collector.get_token_price(token_addr)
            print(f"✓ Token price: ${price}")
        
        # Test 3: Search
        results = await collector.search_pairs('PEPE')
        print(f"✓ Search found {len(results)} results")
        
        print("\n✅ All tests passed! No API key needed.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await collector.close()

if __name__ == "__main__":
    asyncio.run(test())