#!/usr/bin/env python3
"""
Generate a new Solana wallet for ClaudeDex trading
Outputs private key in multiple formats for compatibility
"""

import sys

try:
    from solders.keypair import Keypair
    import base58
except ImportError:
    print("❌ Missing dependencies!")
    print("")
    print("Install with:")
    print("  pip install solders==0.18.1 base58==2.1.1")
    print("")
    sys.exit(1)

def generate_wallet():
    """Generate new Solana wallet"""
    
    # Generate keypair
    keypair = Keypair()
    
    # Get private key in different formats
    private_key_bytes = bytes(keypair)[:32]
    private_key_base58 = base58.b58encode(private_key_bytes).decode('utf-8')
    private_key_hex = private_key_bytes.hex()
    private_key_array = list(private_key_bytes)
    
    # Public address
    public_key = str(keypair.pubkey())
    
    # Display results
    print("")
    print("=" * 70)
    print("🔷 NEW SOLANA WALLET GENERATED".center(70))
    print("=" * 70)
    print("")
    
    print("📍 Public Address:")
    print(f"   {public_key}")
    print("")
    
    print("🔑 Private Key Formats:")
    print("")
    
    print("   Base58 (Recommended for .env):")
    print(f"   {private_key_base58}")
    print("")
    
    print("   Hexadecimal:")
    print(f"   0x{private_key_hex}")
    print("")
    
    print("   Byte Array (for config files):")
    print(f"   {private_key_array}")
    print("")
    
    print("=" * 70)
    print("⚙️  CONFIGURATION".center(70))
    print("=" * 70)
    print("")
    
    print("Add this to your .env file:")
    print("")
    print(f"SOLANA_PRIVATE_KEY={private_key_base58}")
    print("")
    
    print("=" * 70)
    print("💰 FUNDING YOUR WALLET".center(70))
    print("=" * 70)
    print("")
    print("Send SOL to this address:")
    print(f"   {public_key}")
    print("")
    print("Recommended amounts:")
    print("   • Testing: 0.1 SOL (~$15)")
    print("   • Light Trading: 0.5 SOL (~$75)")
    print("   • Active Trading: 2+ SOL (~$300+)")
    print("")
    print("Where to buy SOL:")
    print("   • Coinbase: https://www.coinbase.com/")
    print("   • Binance: https://www.binance.com/")
    print("   • Phantom Wallet: https://phantom.app/")
    print("")
    
    print("=" * 70)
    print("🔍 VERIFY YOUR WALLET".center(70))
    print("=" * 70)
    print("")
    print("View your wallet on Solscan:")
    print(f"   https://solscan.io/account/{public_key}")
    print("")
    print("Check balance after funding:")
    print(f"""
python -c "
import asyncio
from trading.chains.solana import SolanaClient

async def check():
    client = SolanaClient(['https://api.mainnet-beta.solana.com'])
    await client.initialize()
    balance = await client.get_balance('{public_key}')
    print(f'Balance: {{balance}} SOL')
    await client.cleanup()

asyncio.run(check())
"
""")
    
    print("=" * 70)
    print("⚠️  SECURITY WARNINGS".center(70))
    print("=" * 70)
    print("")
    print("🔒 NEVER share your private key with anyone!")
    print("🔒 Store it securely (password manager, encrypted file)")
    print("🔒 Don't commit .env file to git")
    print("🔒 Use a dedicated wallet for the bot (not your main wallet)")
    print("🔒 Start with small amounts for testing")
    print("")
    
    print("=" * 70)
    print("✅ NEXT STEPS".center(70))
    print("=" * 70)
    print("")
    print("1. ✅ Copy private key to .env")
    print("2. 💰 Fund wallet with SOL")
    print("3. 🧪 Run: python test_solana.py")
    print("4. 🚀 Start bot: python main.py")
    print("")
    print("=" * 70)
    print("")
    
    return {
        'public_key': public_key,
        'private_key_base58': private_key_base58,
        'private_key_hex': private_key_hex,
        'private_key_array': private_key_array
    }


def import_wallet():
    """Import existing wallet from private key"""
    
    print("")
    print("=" * 70)
    print("🔷 IMPORT EXISTING SOLANA WALLET".center(70))
    print("=" * 70)
    print("")
    print("Supported formats:")
    print("  1. Base58 string (88 characters)")
    print("  2. Hexadecimal (64 characters, with or without 0x)")
    print("  3. Byte array [1,2,3,...]")
    print("")
    
    private_key_input = input("Enter your private key: ").strip()
    
    try:
        # Try different formats
        if private_key_input.startswith('['):
            # Byte array format
            key_bytes = bytes(eval(private_key_input))
        elif private_key_input.startswith('0x'):
            # Hex format with 0x
            key_bytes = bytes.fromhex(private_key_input[2:])
        elif len(private_key_input) == 64:
            # Hex format without 0x
            key_bytes = bytes.fromhex(private_key_input)
        elif len(private_key_input) == 88:
            # Base58 format
            key_bytes = base58.b58decode(private_key_input)
        else:
            print("❌ Unrecognized private key format")
            return None
        
        # Create keypair
        keypair = Keypair.from_bytes(key_bytes[:32])
        
        # Show wallet info
        print("")
        print("✅ Wallet imported successfully!")
        print("")
        print(f"Public Address: {keypair.pubkey()}")
        print(f"Private Key (Base58): {base58.b58encode(key_bytes[:32]).decode('utf-8')}")
        print("")
        
        return {
            'public_key': str(keypair.pubkey()),
            'private_key': base58.b58encode(key_bytes[:32]).decode('utf-8')
        }
        
    except Exception as e:
        print(f"❌ Failed to import wallet: {e}")
        return None


def main():
    """Main function"""
    
    print("")
    print("🔷 Solana Wallet Manager for ClaudeDex 🔷")
    print("")
    print("Options:")
    print("  1. Generate new wallet")
    print("  2. Import existing wallet")
    print("  3. Exit")
    print("")
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == '1':
        result = generate_wallet()
        
        # Offer to save to file
        save = input("Save wallet info to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = "solana_wallet_backup.txt"
            with open(filename, 'w') as f:
                f.write("🔷 SOLANA WALLET BACKUP\n")
                f.write("=" * 70 + "\n")
                f.write(f"Public Address: {result['public_key']}\n")
                f.write(f"Private Key (Base58): {result['private_key_base58']}\n")
                f.write(f"Private Key (Hex): 0x{result['private_key_hex']}\n")
                f.write(f"Private Key (Array): {result['private_key_array']}\n")
                f.write("=" * 70 + "\n")
                f.write("⚠️  KEEP THIS FILE SECURE!\n")
            
            print(f"✅ Wallet info saved to: {filename}")
            print("⚠️  Remember to delete or encrypt this file after backing up!")
            
    elif choice == '2':
        import_wallet()
        
    elif choice == '3':
        print("Goodbye!")
        sys.exit(0)
        
    else:
        print("❌ Invalid choice")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)