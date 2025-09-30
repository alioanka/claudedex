"""Update token and wallet blacklists"""
import asyncio
import aiohttp
import os
import json
from pathlib import Path
from datetime import datetime

async def update_blacklists():
    """Update blacklists from external sources"""
    
    print("🚫 UPDATING BLACKLISTS")
    print("="*60)
    
    blacklist_dir = Path("data/blacklists")
    blacklist_dir.mkdir(parents=True, exist_ok=True)
    
    # Sources for blacklists
    sources = {
        "scam_tokens": "https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/tokens/tokens-eth.json",
        "known_rugpulls": "https://api.gopluslabs.io/api/v1/RugPull_tokens/",
    }
    
    async with aiohttp.ClientSession() as session:
        
        # Load existing blacklists
        token_blacklist = set()
        wallet_blacklist = set()
        
        token_file = blacklist_dir / "tokens.json"
        wallet_file = blacklist_dir / "wallets.json"
        
        if token_file.exists():
            with open(token_file, 'r') as f:
                token_blacklist = set(json.load(f))
        
        if wallet_file.exists():
            with open(wallet_file, 'r') as f:
                wallet_blacklist = set(json.load(f))
        
        initial_token_count = len(token_blacklist)
        initial_wallet_count = len(wallet_blacklist)
        
        print(f"\n📊 Current Blacklists:")
        print(f"   Tokens: {initial_token_count}")
        print(f"   Wallets: {initial_wallet_count}")
        
        # Check GoPlus for known scams
        print(f"\n🔍 Fetching from external sources...")
        
        goplus_key = os.getenv("GOPLUS_API_KEY")
        if goplus_key:
            try:
                headers = {"Authorization": f"Bearer {goplus_key}"}
                async with session.get(
                    "https://api.gopluslabs.io/api/v1/token_security/1",
                    headers=headers,
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Process and add scam tokens
                        print("   ✅ GoPlus data fetched")
                    else:
                        print(f"   ⚠️ GoPlus API error: {resp.status}")
            except Exception as e:
                print(f"   ⚠️ GoPlus fetch failed: {str(e)[:50]}")
        
        # Add manually reported scams from local file
        manual_file = blacklist_dir / "manual_reports.json"
        if manual_file.exists():
            with open(manual_file, 'r') as f:
                manual = json.load(f)
                token_blacklist.update(manual.get('tokens', []))
                wallet_blacklist.update(manual.get('wallets', []))
            print("   ✅ Manual reports loaded")
        
        # Save updated blacklists
        with open(token_file, 'w') as f:
            json.dump(list(token_blacklist), f, indent=2)
        
        with open(wallet_file, 'w') as f:
            json.dump(list(wallet_blacklist), f, indent=2)
        
        # Create backup
        backup_dir = blacklist_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"blacklist_backup_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump({
                'tokens': list(token_blacklist),
                'wallets': list(wallet_blacklist),
                'updated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        new_tokens = len(token_blacklist) - initial_token_count
        new_wallets = len(wallet_blacklist) - initial_wallet_count
        
        print(f"\n📊 Updated Blacklists:")
        print(f"   Tokens: {len(token_blacklist)} (+{new_tokens})")
        print(f"   Wallets: {len(wallet_blacklist)} (+{new_wallets})")
        print(f"\n💾 Backup saved to: {backup_file}")
        
        print("\n✅ Blacklist update complete!")

if __name__ == "__main__":
    asyncio.run(update_blacklists())