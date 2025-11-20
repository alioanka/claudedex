#!/usr/bin/env python3
"""Test script to verify ABI loading from JSON files"""

import json
import os
import sys

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_abi_loading():
    """Test that ABIs can be loaded from JSON files"""

    abi_dir = os.path.join(os.path.dirname(__file__), 'trading', 'abis')

    print(f"Testing ABI loading from: {abi_dir}")
    print("=" * 60)

    # Test ERC20 ABI
    erc20_path = os.path.join(abi_dir, 'erc20.json')
    print(f"\n1. Loading ERC20 ABI from: {erc20_path}")

    if os.path.exists(erc20_path):
        with open(erc20_path, 'r') as f:
            erc20_abi = json.load(f)
        print(f"   ✓ Loaded {len(erc20_abi)} functions")
        print(f"   Functions: {', '.join([item['name'] for item in erc20_abi])}")
    else:
        print(f"   ✗ File not found!")
        return False

    # Test Uniswap V2 Router ABI
    router_path = os.path.join(abi_dir, 'uniswap_v2_router.json')
    print(f"\n2. Loading Uniswap V2 Router ABI from: {router_path}")

    if os.path.exists(router_path):
        with open(router_path, 'r') as f:
            router_abi = json.load(f)
        print(f"   ✓ Loaded {len(router_abi)} functions")
        print(f"   Functions: {', '.join([item['name'] for item in router_abi])}")
    else:
        print(f"   ✗ File not found!")
        return False

    print("\n" + "=" * 60)
    print("✓ All ABI files loaded successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_abi_loading()
    sys.exit(0 if success else 1)
