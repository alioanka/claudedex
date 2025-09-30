"""Withdraw funds to safe wallet"""
import asyncio
import argparse
import os
from web3 import Web3
from eth_account import Account
from decimal import Decimal

async def withdraw_funds(to_address: str, amount: str = None, confirm: bool = False):
    """Withdraw ETH to specified address"""
    
    if not to_address:
        print("Error: Destination address required")
        print("Usage: python scripts/withdraw_funds.py --to YOUR_SAFE_WALLET")
        return False
    
    # Validate address
    if not Web3.is_address(to_address):
        print(f"Error: Invalid Ethereum address: {to_address}")
        return False
    
    to_address = Web3.to_checksum_address(to_address)
    
    print("WITHDRAW FUNDS")
    print("="*60)
    print(f"Destination: {to_address}")
    
    # Connect to Web3
    w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))
    if not w3.is_connected():
        print("Error: Cannot connect to Ethereum network")
        return False
    
    # Get wallet
    private_key = os.getenv("WALLET_PRIVATE_KEY")
    if not private_key:
        print("Error: WALLET_PRIVATE_KEY not set")
        return False
    
    account = Account.from_key(private_key)
    from_address = account.address
    
    print(f"From: {from_address}")
    
    # Get balance
    balance_wei = w3.eth.get_balance(from_address)
    balance_eth = Web3.from_wei(balance_wei, 'ether')
    
    print(f"Current balance: {balance_eth} ETH")
    
    # Determine amount to send
    if amount:
        if amount.lower() == 'all':
            # Calculate max amount after gas
            gas_price = w3.eth.gas_price
            gas_limit = 21000
            gas_cost_wei = gas_price * gas_limit
            amount_wei = balance_wei - gas_cost_wei
            
            if amount_wei <= 0:
                print("Error: Insufficient balance for gas")
                return False
        else:
            try:
                amount_eth = Decimal(amount)
                amount_wei = Web3.to_wei(amount_eth, 'ether')
                
                if amount_wei > balance_wei:
                    print(f"Error: Insufficient balance. Have {balance_eth} ETH, trying to send {amount_eth} ETH")
                    return False
            except Exception as e:
                print(f"Error: Invalid amount: {amount}")
                return False
    else:
        # Default: withdraw 90% of balance
        amount_wei = int(balance_wei * 0.9)
    
    amount_eth = Web3.from_wei(amount_wei, 'ether')
    
    print(f"Amount to send: {amount_eth} ETH")
    
    # Get gas price
    gas_price = w3.eth.gas_price
    gas_gwei = Web3.from_wei(gas_price, 'gwei')
    gas_cost_eth = Web3.from_wei(gas_price * 21000, 'ether')
    
    print(f"Gas price: {gas_gwei:.1f} Gwei")
    print(f"Estimated gas cost: {gas_cost_eth:.5f} ETH")
    
    remaining_eth = Web3.from_wei(balance_wei - amount_wei - (gas_price * 21000), 'ether')
    print(f"Remaining balance: {remaining_eth:.5f} ETH")
    
    # Confirm
    if not confirm:
        print("\nWARNING: This will transfer funds to the specified address!")
        print("Use --confirm flag to proceed")
        return False
    
    print("\nPreparing transaction...")
    
    try:
        # Build transaction
        nonce = w3.eth.get_transaction_count(from_address)
        
        transaction = {
            'nonce': nonce,
            'to': to_address,
            'value': amount_wei,
            'gas': 21000,
            'gasPrice': gas_price,
            'chainId': w3.eth.chain_id
        }
        
        # Sign transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
        
        # Send transaction
        print("Sending transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"Transaction sent: {tx_hash.hex()}")
        print("Waiting for confirmation...")
        
        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        
        if receipt['status'] == 1:
            print("\n" + "="*60)
            print("WITHDRAWAL SUCCESSFUL")
            print(f"Transaction: {tx_hash.hex()}")
            print(f"Amount: {amount_eth} ETH")
            print(f"To: {to_address}")
            print(f"Block: {receipt['blockNumber']}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            # Check new balance
            new_balance = Web3.from_wei(w3.eth.get_balance(from_address), 'ether')
            print(f"\nNew balance: {new_balance} ETH")
            
            return True
        else:
            print("\nTransaction failed!")
            return False
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Withdraw funds to safe wallet')
    parser.add_argument('--to', required=True, help='Destination address')
    parser.add_argument('--amount', help='Amount in ETH (or "all")')
    parser.add_argument('--confirm', action='store_true', help='Confirm withdrawal')
    args = parser.parse_args()
    
    result = asyncio.run(withdraw_funds(args.to, args.amount, args.confirm))
    exit(0 if result else 1)