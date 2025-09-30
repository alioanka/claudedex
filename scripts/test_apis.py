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
            headers = {"X-API-KEY": os.getenv("DEXSCREENER_API_KEY", "")}
            async with session.get(
                "https://api.dexscreener.com/latest/dex/tokens/0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    results["DexScreener"] = "✅ Connected"
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
        rpc_url = os.getenv("ETH_RPC_URL")
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
                        results["Ethereum RPC"] = "✅ Connected"
                    else:
                        results["Ethereum RPC"] = f"❌ Error: {resp.status}"
            except Exception as e:
                results["Ethereum RPC"] = f"❌ Failed: {str(e)}"
        
        # Test GoPlus API
        print("🔍 Testing GoPlus Security API...")
        try:
            headers = {"Authorization": f"Bearer {os.getenv('GOPLUS_API_KEY', '')}"}
            async with session.get(
                "https://api.gopluslabs.io/api/v1/token_security/1",
                params={"contract_addresses": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"},
                headers=headers
            ) as resp:
                if resp.status == 200:
                    results["GoPlus"] = "✅ Connected"
                else:
                    results["GoPlus"] = f"❌ Error: {resp.status}"
        except Exception as e:
            results["GoPlus"] = f"❌ Failed: {str(e)}"
    
    # Print results
    print("\n" + "="*50)
    print("API CONNECTION TEST RESULTS")
    print("="*50)
    for api, status in results.items():
        print(f"{api}: {status}")
    
    # Check if all critical APIs are working
    critical_apis = ["DexScreener", "Ethereum RPC"]
    critical_ok = all("✅" in results.get(api, "") for api in critical_apis)
    
    if critical_ok:
        print("\n✅ All critical APIs are working!")
    else:
        print("\n⚠️ Some critical APIs are not working. Please check your configuration.")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_apis())