#!/bin/bash
# Debug script to find why Solana executor isn't initializing

echo "=== Checking Solana Configuration ==="
docker compose exec trading-bot python3 << 'PYEOF'
import os
import sys
sys.path.insert(0, '/app')

print("\n1. Environment Variables:")
print(f"   SOLANA_ENABLED = {os.getenv('SOLANA_ENABLED')}")
print(f"   SOLANA_RPC_URL = {os.getenv('SOLANA_RPC_URL', 'NOT SET')}")
print(f"   SOLANA_PRIVATE_KEY = {'SET' if os.getenv('SOLANA_PRIVATE_KEY') else 'NOT SET'}")

print("\n2. Config Loading Test:")
try:
    from utils.config import load_config
    config = load_config()
    
    print(f"   config type: {type(config)}")
    print(f"   config keys: {list(config.keys())[:10]}")
    
    # Check Solana config structure
    if 'solana' in config:
        print(f"   config['solana']: {config['solana']}")
    
    # Check flat keys
    solana_keys = {k: v for k, v in config.items() if 'solana' in k.lower()}
    if solana_keys:
        print(f"   Flat Solana keys: {list(solana_keys.keys())}")
        for k, v in solana_keys.items():
            if 'key' not in k.lower():
                print(f"     {k} = {v}")
    
except Exception as e:
    print(f"   ERROR loading config: {e}")
    import traceback
    traceback.print_exc()

print("\n3. JupiterExecutor Import Test:")
try:
    from trading.chains.solana.jupiter_executor import JupiterExecutor
    print("   ✅ JupiterExecutor import successful")
except Exception as e:
    print(f"   ❌ JupiterExecutor import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. JupiterExecutor Initialization Test:")
try:
    from trading.chains.solana.jupiter_executor import JupiterExecutor
    
    # Try with flat config
    test_config = {
        'enabled': True,
        'rpc_url': os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
        'private_key': os.getenv('SOLANA_PRIVATE_KEY', 'dummy_key_for_test'),
        'max_slippage_bps': 500,
    }
    
    print(f"   Test config: {test_config}")
    executor = JupiterExecutor(test_config)
    print("   ✅ JupiterExecutor initialization successful")
except Exception as e:
    print(f"   ❌ JupiterExecutor initialization failed: {e}")
    import traceback
    traceback.print_exc()

PYEOF

echo ""
echo "=== Checking Logs for Initialization ==="
docker compose logs trading-bot | grep -E "(Solana|Jupiter|Failed to initialize)" | tail -20