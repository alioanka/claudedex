#!/usr/bin/env python3
"""
Test DexScreener Solana support
"""

import asyncio
import logging
from data.collectors.dexscreener import DexScreenerCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_solana_pairs():
    """Test fetching Solana pairs"""
    logger.info("=" * 60)
    logger.info("TEST 1: Fetch Solana Pairs")
    logger.info("=" * 60)
    
    config = {
        'api_key': '',
        'chains': ['solana'],
        'rate_limit': 10,
        'min_liquidity': 5000,
        'min_volume': 2000,
        'max_age_hours': 48
    }
    
    collector = DexScreenerCollector(config)
    await collector.initialize()
    
    try:
        # Test different chain name formats
        chain_formats = ['solana', 'SOLANA', 'sol', 'SOL']
        
        for chain_format in chain_formats:
            logger.info(f"\n🔍 Testing with chain format: '{chain_format}'")
            
            pairs = await collector.get_new_pairs(chain=chain_format, limit=5)
            
            if pairs:
                logger.info(f"✅ Found {len(pairs)} pairs for '{chain_format}'")
                
                # Show first pair details
                if len(pairs) > 0:
                    pair = pairs[0]
                    logger.info(f"\n📊 Sample Pair:")
                    logger.info(f"   Token: {pair.get('token_symbol', 'UNKNOWN')}")
                    logger.info(f"   Chain: {pair.get('chain', 'unknown')}")
                    logger.info(f"   Address: {pair.get('token_address', 'unknown')[:20]}...")
                    logger.info(f"   Price: ${pair.get('price_usd', 0):.8f}")
                    logger.info(f"   Liquidity: ${pair.get('liquidity_usd', 0):,.2f}")
                    logger.info(f"   Volume 24h: ${pair.get('volume_24h', 0):,.2f}")
                    logger.info(f"   DEX: {pair.get('dex', 'unknown')}")
                
                return True
            else:
                logger.warning(f"⚠️  No pairs found for '{chain_format}'")
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await collector.close()


async def test_solana_token_price():
    """Test getting Solana token prices"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Get Solana Token Prices")
    logger.info("=" * 60)
    
    config = {
        'api_key': '',
        'chains': ['solana']
    }
    
    collector = DexScreenerCollector(config)
    await collector.initialize()
    
    # Popular Solana tokens
    test_tokens = {
        'SOL': 'So11111111111111111111111111111111111111112',
        'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
        'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    }
    
    try:
        for symbol, address in test_tokens.items():
            logger.info(f"\n🔍 Getting price for {symbol}...")
            
            price = await collector.get_token_price(address, chain='solana')
            
            if price:
                logger.info(f"✅ {symbol} Price: ${price:.8f}")
            else:
                logger.warning(f"⚠️  Could not get price for {symbol}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await collector.close()


async def test_solana_chain_normalization():
    """Test chain name normalization"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Chain Name Normalization")
    logger.info("=" * 60)
    
    config = {'api_key': '', 'chains': ['solana']}
    collector = DexScreenerCollector(config)
    
    test_cases = [
        ('solana', 'solana'),
        ('SOLANA', 'solana'),
        ('sol', 'solana'),
        ('SOL', 'solana'),
        ('Solana', 'solana'),
        ('ethereum', 'ethereum'),
        ('ETH', 'ethereum'),
        ('bsc', 'bsc'),
        ('BNB', 'bsc'),
    ]
    
    all_passed = True
    
    for input_chain, expected_output in test_cases:
        result = collector._normalize_chain(input_chain)
        
        if result == expected_output:
            logger.info(f"✅ '{input_chain}' → '{result}' (expected: '{expected_output}')")
        else:
            logger.error(f"❌ '{input_chain}' → '{result}' (expected: '{expected_output}')")
            all_passed = False
    
    return all_passed


async def test_solana_filter_thresholds():
    """Test Solana-specific filter thresholds"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Solana Filter Thresholds")
    logger.info("=" * 60)
    
    config = {
        'api_key': '',
        'chains': ['solana'],
        'min_liquidity': 10000,  # Default
        'min_volume': 5000,      # Default
    }
    
    collector = DexScreenerCollector(config)
    await collector.initialize()
    
    try:
        # Fetch some Solana pairs and check if filters are applied correctly
        pairs = await collector.get_new_pairs(chain='solana', limit=10)
        
        logger.info(f"\n📊 Analyzing {len(pairs)} Solana pairs...")
        
        if pairs:
            liquidity_values = [p.get('liquidity_usd', 0) for p in pairs]
            volume_values = [p.get('volume_24h', 0) for p in pairs]
            
            logger.info(f"\n💧 Liquidity Range:")
            logger.info(f"   Min: ${min(liquidity_values):,.2f}")
            logger.info(f"   Max: ${max(liquidity_values):,.2f}")
            logger.info(f"   Avg: ${sum(liquidity_values)/len(liquidity_values):,.2f}")
            
            logger.info(f"\n📈 Volume Range:")
            logger.info(f"   Min: ${min(volume_values):,.2f}")
            logger.info(f"   Max: ${max(volume_values):,.2f}")
            logger.info(f"   Avg: ${sum(volume_values)/len(volume_values):,.2f}")
            
            # Check if Solana thresholds are being applied
            solana_min_liq = 5000  # From our patch
            solana_min_vol = 2000  # From our patch
            
            all_pass_liquidity = all(liq >= solana_min_liq for liq in liquidity_values)
            all_pass_volume = all(vol >= solana_min_vol for vol in volume_values)
            
            if all_pass_liquidity:
                logger.info(f"\n✅ All pairs meet Solana min liquidity (${solana_min_liq:,.0f})")
            else:
                logger.warning(f"\n⚠️  Some pairs below min liquidity")
            
            if all_pass_volume:
                logger.info(f"✅ All pairs meet Solana min volume (${solana_min_vol:,.0f})")
            else:
                logger.warning(f"⚠️  Some pairs below min volume")
            
            return all_pass_liquidity and all_pass_volume
        else:
            logger.warning("⚠️  No pairs found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        await collector.close()


async def run_all_tests():
    """Run all Solana DexScreener tests"""
    logger.info("\n" + "🔷" * 30)
    logger.info("DEXSCREENER SOLANA TEST SUITE")
    logger.info("🔷" * 30 + "\n")
    
    tests = [
        ("Fetch Solana Pairs", test_solana_pairs),
        ("Get Token Prices", test_solana_token_price),
        ("Chain Normalization", test_solana_chain_normalization),
        ("Filter Thresholds", test_solana_filter_thresholds),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        await asyncio.sleep(2)  # Rate limiting
    
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
        logger.info("\n🎉 All tests passed! Solana DexScreener integration working!")
    elif success_rate >= 75:
        logger.info("\n⚠️  Most tests passed. Check failures above.")
    else:
        logger.info("\n❌ Multiple failures. Review configuration.")
    
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