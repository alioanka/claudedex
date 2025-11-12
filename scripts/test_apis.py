# ============================================
# scripts/test_apis.py
# ============================================
"""Test API connections and keys"""
import asyncio
import aiohttp
import os
from datetime import datetime

async def test_apis():
    """Test all API connections"""
    
    results = {}
    
    async with aiohttp.ClientSession() as session:
        
        # Test DexScreener API
        print("🔍 Testing DexScreener API...")
        try:
            # Use environment variables for flexibility, with sensible defaults
            chain = os.getenv("TEST_CHAIN", "ethereum")
            pair_address = os.getenv("TEST_PAIR_ADDRESS", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640") # WETH/USDC on Ethereum

            # Use the correct, updated endpoint structure
            url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair_address}"

            print(f"   Querying URL: {url}")

            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('pair'):
                        results["DexScreener"] = f"✅ Connected (Pair: {data['pair']['baseToken']['symbol']}/{data['pair']['quoteToken']['symbol']})"
                    else:
                        results["DexScreener"] = f"❌ Error: Response OK, but no pair data found."
                else:
                    results["DexScreener"] = f"❌ Error: {resp.status}"
        except Exception as e:
            results["DexScreener"] = f"❌ Failed: {str(e)}"
        
        # Test Etherscan API
        print("🔍 Testing Etherscan API...")
        try:
            params = {
                "module": "stats",
                "action": "ethsupply",
                "apikey": os.getenv("ETHERSCAN_API_KEY", "")
            }
            async with session.get(
                "https://api.etherscan.io/api",
                params=params
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "1":
                        results["Etherscan"] = "✅ Connected"
                    else:
                        results["Etherscan"] = "❌ Invalid API key"
                else:
                    results["Etherscan"] = f"❌ Error: {resp.status}"
        except Exception as e:
            results["Etherscan"] = f"❌ Failed: {str(e)}"
        
        # Test RPC endpoints
        print("🔍 Testing RPC endpoints...")
        # Use a generic name for the RPC URL to test any chain
        rpc_url = os.getenv("TEST_RPC_URL", os.getenv("ETHEREUM_RPC_URLS", "").split(',')[0])
        if rpc_url:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber",
                    "params": [],
                    "id": 1
                }
                async with session.post(rpc_url, json=payload) as resp:
                    if resp.status == 200:
                        results["RPC Endpoint"] = "✅ Connected"
                    else:
                        results["RPC Endpoint"] = f"❌ Error: {resp.status}"
            except Exception as e:
                results["RPC Endpoint"] = f"❌ Failed: {str(e)}"
        
        # Test GoPlus API
        print("🔍 Testing GoPlus Security API...")
        try:
            # Note: GoPlus API requires an Authorization token, not a simple key.
            # Assuming the key is the token for this test.
            auth_token = os.getenv('GOPLUS_API_KEY', '')
            if auth_token:
                headers = {"Authorization": f"Bearer {auth_token}"}
                # Using a valid chain ID (1 for Ethereum) and a token address (UNI)
                goplus_url = "https://api.gopluslabs.io/api/v1/token_security/1?contract_addresses=0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
                async with session.get(goplus_url, headers=headers) as resp:
                    if resp.status == 200:
                        results["GoPlus"] = "✅ Connected"
                    else:
                        data = await resp.text()
                        results["GoPlus"] = f"❌ Error: {resp.status} - {data}"
            else:
                results["GoPlus"] = "⚠️  Skipped (GOPLUS_API_KEY not set)"
        except Exception as e:
            results["GoPlus"] = f"❌ Failed: {str(e)}"

    
    # Print results
    print("\n" + "="*50)
    print("API CONNECTION TEST RESULTS")
    print("="*50)
    for api, status in results.items():
        print(f"{api}: {status}")
    
    # Check if all critical APIs are working
    critical_apis = ["DexScreener", "RPC Endpoint"]
    critical_ok = all("✅" in results.get(api, "") for api in critical_apis)
    
    if critical_ok:
        print("\n✅ All critical APIs are working!")
    else:
        print("\n⚠️ Some critical APIs are not working. Please check your configuration.")
        return False
    
    return True

if __name__ == "__main__":
    # Load .env file for local testing
    try:
        from dotenv import load_dotenv
        print("Loading .env file for local testing...")
        load_dotenv()
    except ImportError:
        print("dotenv not installed, skipping .env load. Make sure environment variables are set.")

    asyncio.run(test_apis())
