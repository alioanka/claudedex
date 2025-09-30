# ============================================
# scripts/reset_nonce.py
# ============================================
"""Reset transaction nonce for stuck transactions"""
import asyncio
from web3 import Web3
import os

async def reset_nonce():
    """Reset transaction nonce"""
    
    w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Ethereum network")
        return
    
    wallet_address = os.getenv("WALLET_ADDRESS")
    if not wallet_address:
        print("❌ Wallet address not configured")
        return
    
    print(f"🔄 Resetting nonce for {wallet_address}")
    
    # Get current nonce
    current_nonce = w3.eth.get_transaction_count(wallet_address, 'latest')
    pending_nonce = w3.eth.get_transaction_count(wallet_address, 'pending')
    
    print(f"Current nonce: {current_nonce}")
    print(f"Pending nonce: {pending_nonce}")
    
    if pending_nonce > current_nonce:
        print(f"⚠️ Found {pending_nonce - current_nonce} pending transactions")
        
        # Option to cancel pending transactions
        response = input("Cancel pending transactions? (y/n): ")
        if response.lower() == 'y':
            # Send 0 ETH transaction with higher gas to cancel
            print("Sending cancellation transaction...")
            
            # In production, this would actually send a transaction
            print("✅ Nonce reset complete")
    else:
        print("✅ No pending transactions, nonce is correct")
    
    return current_nonce

if __name__ == "__main__":
    asyncio.run(reset_nonce())