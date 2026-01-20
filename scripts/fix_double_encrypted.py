#!/usr/bin/env python3
"""
Fix Double-Encrypted Credentials

This script finds and fixes credentials that were accidentally encrypted twice.
Double encryption happens when a migration script encrypts an already-encrypted value.

Usage:
    docker exec -it trading-bot python scripts/fix_double_encrypted.py --dry-run
    docker exec -it trading-bot python scripts/fix_double_encrypted.py
"""
import asyncio
import os
from pathlib import Path

async def fix_double_encrypted():
    print("=" * 60)
    print("FIX DOUBLE-ENCRYPTED CREDENTIALS")
    print("=" * 60)

    # Load encryption key
    key_file = Path('.encryption_key')
    if not key_file.exists():
        print("❌ ERROR: .encryption_key not found!")
        return False

    key = key_file.read_bytes().strip()

    from cryptography.fernet import Fernet, InvalidToken
    fernet = Fernet(key)
    print(f"✅ Loaded encryption key")

    # Connect to database
    import asyncpg

    db_host = os.getenv('DB_HOST', 'postgres')

    # Read db credentials from secrets or env
    user_file = Path('/run/secrets/db_user')
    pass_file = Path('/run/secrets/db_password')

    if user_file.exists():
        db_user = user_file.read_text().strip()
        db_pass = pass_file.read_text().strip()
    else:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASSWORD')
        if not db_user or not db_pass:
            print("❌ ERROR: DB credentials not available")
            return False

    pool = await asyncpg.create_pool(
        host=db_host, port=5432, database='tradingbot',
        user=db_user, password=db_pass
    )
    print(f"✅ Connected to database")

    # Check for --dry-run
    import sys
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("\n🔶 DRY RUN MODE - No changes will be made\n")

    # Find all encrypted credentials
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, key_name, encrypted_value, is_encrypted
            FROM secure_credentials
            WHERE is_active = TRUE
            AND encrypted_value LIKE 'gAAAAAB%'
            ORDER BY key_name
        """)

    print(f"\nChecking {len(rows)} encrypted credentials...\n")

    fixed_count = 0
    for row in rows:
        key_name = row['key_name']
        encrypted_value = row['encrypted_value']

        # Try to decrypt
        try:
            decrypted = fernet.decrypt(encrypted_value.encode()).decode()
        except InvalidToken:
            print(f"⚠️  {key_name}: Cannot decrypt (key mismatch?)")
            continue
        except Exception as e:
            print(f"⚠️  {key_name}: Decryption error: {e}")
            continue

        # Check if result is STILL encrypted (double-encryption)
        if decrypted.startswith('gAAAAAB'):
            print(f"🔴 {key_name}: DOUBLE-ENCRYPTED detected!")

            # Try to decrypt the inner value
            try:
                # First, try with current key (same key used twice)
                inner_decrypted = fernet.decrypt(decrypted.encode()).decode()
                final_value = inner_decrypted
                print(f"   ✅ Successfully decrypted inner value (length: {len(final_value)})")
            except InvalidToken:
                # Inner encryption might use a different/old key
                print(f"   ⚠️  Inner value uses different key - keeping single-decrypted value")
                final_value = decrypted
            except Exception as e:
                print(f"   ❌ Inner decryption failed: {e}")
                continue

            # Check if we need to go deeper (triple encryption?)
            depth = 2
            while final_value.startswith('gAAAAAB') and depth < 5:
                try:
                    final_value = fernet.decrypt(final_value.encode()).decode()
                    depth += 1
                    print(f"   ✅ Decrypted layer {depth}")
                except:
                    break

            if final_value.startswith('gAAAAAB'):
                print(f"   ⚠️  Value still encrypted after {depth} layers - may need old key")
                continue

            # Re-encrypt with single layer
            new_encrypted = fernet.encrypt(final_value.encode()).decode()

            if dry_run:
                print(f"   [DRY RUN] Would fix: {key_name}")
                print(f"   Old length: {len(encrypted_value)}, New length: {len(new_encrypted)}")
            else:
                # Update database
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE secure_credentials
                        SET encrypted_value = $1, updated_at = NOW()
                        WHERE id = $2
                    """, new_encrypted, row['id'])
                print(f"   ✅ Fixed: {key_name}")

            fixed_count += 1
        else:
            # Not double-encrypted
            # Show first few chars for verification (mask sensitive data)
            preview = decrypted[:10] + "..." if len(decrypted) > 10 else decrypted
            if key_name in ['PRIVATE_KEY', 'SOLANA_PRIVATE_KEY', 'SOLANA_MODULE_PRIVATE_KEY']:
                preview = f"[{len(decrypted)} chars]"
            print(f"✅ {key_name}: OK (decrypts to: {preview})")

    await pool.close()

    print(f"\n" + "=" * 60)
    if dry_run:
        print(f"DRY RUN: Would fix {fixed_count} double-encrypted credentials")
        print("Run without --dry-run to apply fixes")
    else:
        print(f"Fixed {fixed_count} double-encrypted credentials")
        if fixed_count > 0:
            print("\n⚠️  Restart the trading bot to use the fixed credentials:")
            print("   docker-compose restart trading-bot")
    print("=" * 60)

    return True

if __name__ == '__main__':
    asyncio.run(fix_double_encrypted())
