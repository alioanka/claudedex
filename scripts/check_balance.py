# ============================================
# scripts/check_balance.py
# ============================================
"""Check wallet balance and token holdings"""
import asyncio
from web3 import Web3
import os
from decimal import Decimal

async def check_balance():
    """Check wallet balance"""
    
    # Connect to Ethereum
    w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Ethereum network")
        return
    
    # Get wallet address from private key (encrypted)
    # In production, this should decrypt the private key
    wallet_address = os.getenv("WALLET_ADDRESS")
    
    if not wallet_address:
        print("❌ Wallet address not configured")
        return
    
    print(f"💼 Wallet: {wallet_address}")
    print("="*50)
    
    # Get ETH balance
    eth_balance = w3.eth.get_balance(wallet_address)
    eth_decimal = Web3.from_wei(eth_balance, 'ether')
    
    print(f"ETH Balance: {eth_decimal:.4f} ETH")
    
    # Get current ETH price (mock for now)
    eth_price = 2000  # In production, fetch from API
    usd_value = float(eth_decimal) * eth_price
    
    print(f"USD Value: ${usd_value:,.2f}")
    
    # Check gas price
    gas_price = w3.eth.gas_price
    gas_gwei = Web3.from_wei(gas_price, 'gwei')
    print(f"\nCurrent Gas Price: {gas_gwei:.1f} Gwei")
    
    # Calculate transaction costs
    typical_gas = 150000  # Typical swap gas
    tx_cost_wei = gas_price * typical_gas
    tx_cost_eth = Web3.from_wei(tx_cost_wei, 'ether')
    tx_cost_usd = float(tx_cost_eth) * eth_price
    
    print(f"Typical Swap Cost: {tx_cost_eth:.5f} ETH (${tx_cost_usd:.2f})")
    
    # Check if balance is sufficient
    min_balance = 0.1  # Minimum ETH for operations
    if float(eth_decimal) < min_balance:
        print(f"\n⚠️ Warning: Balance below minimum ({min_balance} ETH)")
        print("   Please add funds to continue trading")
    else:
        print(f"\n✅ Balance sufficient for trading")
    
    return {
        "address": wallet_address,
        "eth_balance": float(eth_decimal),
        "usd_value": usd_value,
        "gas_price_gwei": float(gas_gwei),
        "sufficient": float(eth_decimal) >= min_balance
    }

if __name__ == "__main__":
    asyncio.run(check_balance())
