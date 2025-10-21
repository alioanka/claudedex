#!/usr/bin/env python3
"""
Test Jupiter v1 Executor Integration
Run this inside the Docker container to verify the executor works
"""

import sys
import os
import asyncio
from decimal import Decimal

sys.path.insert(0, '/app')

from trading.chains.solana.jupiter_executor import JupiterExecutor
from trading.orders.order_manager import Order, OrderType


class MockOrder:
    """Mock order for testing"""
    def __init__(self):
        # BONK -> USDC swap (small amount for testing)
        self.symbol = "BONK"
        self.symbol_in = "BONK"
        self.symbol_out = "USDC"
        self.token_in = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # BONK
        self.token_out = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        self.amount_in = 1000000  # 1 BONK (5 decimals)
        self.order_type = OrderType.MARKET


async def test_jupiter_executor():
    """Test the Jupiter executor end-to-end"""
    
    print("=" * 60)
    print("Jupiter v1 Executor Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Configuration
    print("📋 Test 1: Configuration Loading")
    config = {
        'rpc_url': os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
        'private_key': os.getenv('SOLANA_PRIVATE_KEY', ''),
        'max_slippage_bps': 500,
        'dry_run': True  # Always use dry run for testing
    }
    
    print(f"   RPC URL: {config['rpc_url'][:50]}...")
    print(f"   Max Slippage: {config['max_slippage_bps']} bps")
    print(f"   Dry Run: {config['dry_run']}")
    print(f"   Private Key: {'Configured' if config['private_key'] else 'Not Set'}")
    print()
    
    # Test 2: Initialization
    print("🔧 Test 2: Executor Initialization")
    try:
        executor = JupiterExecutor(config)
        print("   ✅ Executor initialized")
        print(f"   Jupiter API: {executor.jupiter_api_url}")
        if executor.wallet_address:
            print(f"   Wallet: {executor.wallet_address[:8]}...{executor.wallet_address[-8:]}")
        else:
            print("   Wallet: Not configured (OK for paper trading)")
        print()
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False
    
    # Test 3: Session initialization
    print("🌐 Test 3: HTTP Session")
    try:
        await executor.initialize()
        print("   ✅ HTTP session initialized")
        print()
    except Exception as e:
        print(f"   ❌ Session init failed: {e}")
        return False
    
    # Test 4: Get quote
    print("💰 Test 4: Get Jupiter Quote (BONK -> USDC)")
    try:
        quote = await executor._get_quote(
            input_mint="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # BONK
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            amount=1000000,  # 1 BONK
            slippage_bps=500
        )
        
        if quote:
            print("   ✅ Quote received successfully")
            print(f"   Input Amount: {quote.get('inAmount')}")
            print(f"   Output Amount: {quote.get('outAmount')}")
            print(f"   Price Impact: {quote.get('priceImpactPct', 'N/A')}%")
            print(f"   Slippage: {quote.get('slippageBps')} bps")
            
            # Check required fields
            required_fields = ['inputMint', 'outputMint', 'inAmount', 'outAmount']
            missing_fields = [f for f in required_fields if f not in quote]
            
            if missing_fields:
                print(f"   ⚠️  Missing fields: {missing_fields}")
            else:
                print("   ✅ All required fields present")
            print()
        else:
            print("   ❌ Failed to get quote")
            return False
            
    except Exception as e:
        print(f"   ❌ Quote request failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Execute trade (dry run)
    print("🔥 Test 5: Execute Trade (Dry Run)")
    try:
        order = MockOrder()
        result = await executor.execute_trade(order)
        
        if result.get('success'):
            print("   ✅ Trade executed successfully (DRY RUN)")
            print(f"   Signature: {result.get('signature')}")
            print(f"   Execution Time: {result.get('execution_time')} seconds")
            print(f"   Gas Used: {result.get('gas_used')} SOL")
            
            if result.get('dry_run'):
                print("   ✅ Confirmed as dry run")
            print()
        else:
            print(f"   ❌ Trade execution failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Trade execution error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Stats
    print("📊 Test 6: Execution Statistics")
    try:
        stats = await executor.get_execution_stats()
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Successful: {stats['successful_trades']}")
        print(f"   Failed: {stats['failed_trades']}")
        
        if stats['total_trades'] > 0:
            print(f"   Success Rate: {stats.get('success_rate', 0) * 100:.2f}%")
        print()
    except Exception as e:
        print(f"   ⚠️  Stats unavailable: {e}")
        print()
    
    # Test 7: Cleanup
    print("🧹 Test 7: Cleanup")
    try:
        await executor.cleanup()
        print("   ✅ Cleanup successful")
        print()
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
        print()
    
    # Extra wait to ensure cleanup completes
    await asyncio.sleep(0.1)
    
    # Summary
    print("=" * 60)
    print("✅ All Tests Passed!")
    print("=" * 60)
    print()
    print("The Jupiter v1 executor is working correctly.")
    print("You can now enable Solana trading in your bot.")
    print()
    
    return True


async def test_api_only():
    """Quick test of just the Jupiter API"""
    import aiohttp
    
    print("🌐 Quick Jupiter v1 API Test")
    print("-" * 40)
    
    url = "https://lite-api.jup.ag/swap/v1/quote"
    params = {
        'inputMint': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',  # BONK
        'outputMint': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
        'amount': '1000000',
        'slippageBps': 500
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ API is accessible")
                    print(f"Response keys: {list(data.keys())}")
                    print(f"Input: {data.get('inAmount')}")
                    print(f"Output: {data.get('outAmount')}")
                else:
                    text = await response.text()
                    print(f"❌ API error: {text}")
                    
    except Exception as e:
        print(f"❌ Connection error: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Jupiter v1 Executor')
    parser.add_argument('--api-only', action='store_true', help='Test only API connectivity')
    args = parser.parse_args()
    
    if args.api_only:
        asyncio.run(test_api_only())
    else:
        success = asyncio.run(test_jupiter_executor())
        sys.exit(0 if success else 1)