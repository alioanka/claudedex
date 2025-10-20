#!/usr/bin/env python3
"""
Test script for Solana integration
Run this to verify Jupiter executor is working correctly
"""

import asyncio
import logging
from decimal import Decimal
from trading.chains.solana import JupiterExecutor, SolanaClient, COMMON_TOKENS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_jupiter_quote():
    """Test getting a quote from Jupiter"""
    logger.info("=" * 60)
    logger.info("TEST 1: Jupiter Quote")
    logger.info("=" * 60)
    
    config = {
        'jupiter_url': 'https://quote-api.jup.ag/v6',
        'solana_rpc_url': 'https://mainnet.helius-rpc.com/?api-key=a78df2b2-9cfb-421b-86cf-801dddff77c4',
        'solana_backup_rpcs': [
            'https://rpc.ankr.com/solana_devnet/4daecdbd46f7cc39b14e343e5ee0cc0be57e5f52faa2aff6baefe3826227064d',
            'https://solana-mainnet.g.alchemy.com/v2/p8TihQhCAIDgc5CdyPuX4'
        ],
        'max_slippage': 0.05,
        'rate_limit': 10,
        'priority_fee': 5000,
        'compute_unit_price': 1000,
        'compute_unit_limit': 200000
    }
    
    executor = JupiterExecutor(config)
    
    try:
        await executor.initialize()
        logger.info("✅ Jupiter executor initialized")
        
        # Test quote: 0.1 SOL → USDC
        logger.info("\n🔍 Getting quote for 0.1 SOL → USDC...")
        
        quote = await executor.get_quote(
            input_mint=COMMON_TOKENS['SOL'],
            output_mint=COMMON_TOKENS['USDC'],
            amount_lamports=100_000_000,  # 0.1 SOL
        )
        
        if quote:
            logger.info("✅ Quote received successfully!")
            logger.info(f"   Input: {quote.in_amount / 1e9:.6f} SOL")
            logger.info(f"   Output: {quote.out_amount / 1e6:.6f} USDC")
            logger.info(f"   Price Impact: {quote.price_impact_pct:.4f}%")
            logger.info(f"   Slippage: {quote.slippage_bps / 100:.2f}%")
            logger.info(f"   Route: {quote.route_type}")
            logger.info(f"   DEXes: {len(quote.route_plan)} hops")
            
            # Show route details
            logger.info("\n📍 Route Plan:")
            for i, hop in enumerate(quote.route_plan, 1):
                swap_info = hop.get('swapInfo', {})
                logger.info(f"   Hop {i}: {swap_info.get('label', 'Unknown DEX')}")
            
            return True
        else:
            logger.error("❌ Failed to get quote")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await executor.cleanup()


async def test_solana_rpc():
    """Test Solana RPC connection"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Solana RPC Connection")
    logger.info("=" * 60)
    
    rpc_urls = [
        'https://mainnet.helius-rpc.com/?api-key=a78df2b2-9cfb-421b-86cf-801dddff77c4',
        'https://rpc.ankr.com/solana_devnet/4daecdbd46f7cc39b14e343e5ee0cc0be57e5f52faa2aff6baefe3826227064d',
        'https://solana-mainnet.g.alchemy.com/v2/p8TihQhCAIDgc5CdyPuX4'
    ]
    
    client = SolanaClient(rpc_urls)
    
    try:
        await client.initialize()
        logger.info("✅ Solana client initialized")
        
        # Test getting current slot
        logger.info("\n🔍 Getting current slot...")
        slot = await client.get_slot()
        
        if slot:
            logger.info(f"✅ Current slot: {slot:,}")
            
            # Test getting recent blockhash
            logger.info("\n🔍 Getting recent blockhash...")
            blockhash = await client.get_recent_blockhash()
            
            if blockhash:
                logger.info(f"✅ Blockhash: {blockhash[:20]}...")
                return True
            else:
                logger.error("❌ Failed to get blockhash")
                return False
        else:
            logger.error("❌ Failed to get slot")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await client.cleanup()


async def test_wallet_balance():
    """Test checking wallet balance"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Wallet Balance Check")
    logger.info("=" * 60)
    
    # Test with a known wallet (Jupiter's wallet)
    test_wallet = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"
    
    client = SolanaClient(['https://api.mainnet-beta.solana.com'])
    
    try:
        await client.initialize()
        
        logger.info(f"\n🔍 Checking balance for: {test_wallet[:20]}...")
        balance = await client.get_balance(test_wallet)
        
        if balance is not None:
            logger.info(f"✅ SOL Balance: {balance:.9f} SOL")
            return True
        else:
            logger.error("❌ Failed to get balance")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await client.cleanup()


async def test_multiple_quotes():
    """Test getting quotes for multiple token pairs"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Multiple Token Pairs")
    logger.info("=" * 60)
    
    config = {
        'jupiter_url': 'https://quote-api.jup.ag/v6',
        'solana_rpc_url': 'https://api.mainnet-beta.solana.com',
        'max_slippage': 0.05
    }
    
    executor = JupiterExecutor(config)
    
    test_pairs = [
        ('SOL', 'USDC', 100_000_000),   # 0.1 SOL
        ('SOL', 'USDT', 50_000_000),    # 0.05 SOL
        ('SOL', 'RAY', 100_000_000),    # 0.1 SOL
    ]
    
    try:
        await executor.initialize()
        
        results = []
        
        for token_in, token_out, amount in test_pairs:
            logger.info(f"\n🔍 Testing {token_in} → {token_out}...")
            
            quote = await executor.get_quote(
                input_mint=COMMON_TOKENS[token_in],
                output_mint=COMMON_TOKENS[token_out],
                amount_lamports=amount
            )
            
            if quote:
                logger.info(f"   ✅ Quote received")
                logger.info(f"   Price Impact: {quote.price_impact_pct:.4f}%")
                results.append(True)
            else:
                logger.error(f"   ❌ Failed to get quote")
                results.append(False)
            
            # Rate limit
            await asyncio.sleep(0.5)
        
        success_rate = sum(results) / len(results) * 100
        logger.info(f"\n📊 Success Rate: {success_rate:.1f}% ({sum(results)}/{len(results)})")
        
        return success_rate == 100.0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await executor.cleanup()


async def test_execution_stats():
    """Test executor statistics tracking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Execution Statistics")
    logger.info("=" * 60)
    
    config = {
        'jupiter_url': 'https://quote-api.jup.ag/v6',
        'solana_rpc_url': 'https://api.mainnet-beta.solana.com',
        'max_slippage': 0.05
    }
    
    executor = JupiterExecutor(config)
    
    try:
        await executor.initialize()
        
        # Get initial stats
        stats = await executor.get_execution_stats()
        
        logger.info("✅ Statistics retrieved:")
        logger.info(f"   Total Trades: {stats['total_trades']}")
        logger.info(f"   Successful: {stats['successful_trades']}")
        logger.info(f"   Failed: {stats['failed_trades']}")
        logger.info(f"   Total Volume: {stats['total_volume_sol']} SOL")
        logger.info(f"   Total Fees: {stats['total_fees_sol']} SOL")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await executor.cleanup()


async def run_all_tests():
    """Run all Solana integration tests"""
    logger.info("\n" + "🔷" * 30)
    logger.info("SOLANA INTEGRATION TEST SUITE")
    logger.info("🔷" * 30 + "\n")
    
    tests = [
        ("Jupiter Quote", test_jupiter_quote),
        ("Solana RPC", test_solana_rpc),
        ("Wallet Balance", test_wallet_balance),
        ("Multiple Quotes", test_multiple_quotes),
        ("Execution Stats", test_execution_stats),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        # Pause between tests
        await asyncio.sleep(1)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    success_rate = (passed / total) * 100
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        logger.info("\n🎉 All tests passed! Solana integration is working correctly.")
    elif success_rate >= 80:
        logger.info("\n⚠️  Most tests passed. Check failures above.")
    else:
        logger.info("\n❌ Multiple failures detected. Review configuration.")
    
    return success_rate == 100


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
        exit(1)