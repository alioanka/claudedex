#!/usr/bin/env python3
"""
Diagnostic script to check encryption key status and test decryption

This script will:
1. Check if ENCRYPTION_KEY is set in environment
2. Verify it's a valid Fernet key
3. Test decrypting a sensitive config from database
4. Provide clear guidance on what's wrong and how to fix it
"""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.encryption import EncryptionManager
from cryptography.fernet import Fernet

async def main():
    print("=" * 80)
    print("ENCRYPTION KEY DIAGNOSTIC")
    print("=" * 80)
    print()

    # Step 1: Check ENCRYPTION_KEY
    print("1. Checking ENCRYPTION_KEY in environment...")
    encryption_key = os.getenv('ENCRYPTION_KEY')

    if not encryption_key:
        print("   ❌ ENCRYPTION_KEY not found in environment")
        print("   Solution: Add ENCRYPTION_KEY to your .env file")
        return False

    print(f"   ✅ ENCRYPTION_KEY found: {encryption_key[:20]}...{encryption_key[-10:]}")
    print()

    # Step 2: Validate it's a proper Fernet key
    print("2. Validating ENCRYPTION_KEY format...")
    try:
        test_fernet = Fernet(encryption_key.encode())
        print("   ✅ ENCRYPTION_KEY is a valid Fernet key")
    except Exception as e:
        print(f"   ❌ ENCRYPTION_KEY is invalid: {e}")
        print("   Solution: Run python scripts/generate_encryption_key.py")
        return False
    print()

    # Step 3: Initialize EncryptionManager
    print("3. Initializing EncryptionManager...")
    try:
        encryption_config = {'encryption_key': encryption_key}
        encryption_manager = EncryptionManager(encryption_config)
        print("   ✅ EncryptionManager initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    print()

    # Step 4: Connect to database
    print("4. Connecting to database...")
    database_url = os.getenv("DATABASE_URL")

    try:
        if database_url:
            conn = await asyncpg.connect(database_url)
        else:
            conn = await asyncpg.connect(
                host=os.getenv("DB_HOST", "postgres"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "tradingbot"),
                user=os.getenv("DB_USER", "bot_user"),
                password=os.getenv("DB_PASSWORD", "bot_password")
            )
        print("   ✅ Connected to database")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False
    print()

    # Step 5: Check sensitive configs in database
    print("5. Checking sensitive configs in database...")
    rows = await conn.fetch("""
        SELECT key, encrypted_value, description
        FROM config_sensitive
        WHERE is_active = TRUE
        LIMIT 5
    """)

    if not rows:
        print("   ⚠️  No sensitive configs found in database")
        print("   Solution: Run python scripts/import_env_secrets.py")
        await conn.close()
        return False

    print(f"   ✅ Found {len(rows)} sensitive configs (showing first 5)")
    print()

    # Step 6: Test decryption
    print("6. Testing decryption...")
    success_count = 0
    fail_count = 0

    for row in rows:
        key = row['key']
        encrypted_value = row['encrypted_value']

        try:
            decrypted = encryption_manager.decrypt_sensitive_data(encrypted_value)
            success_count += 1
            # Show first 10 chars of decrypted value for verification
            preview = str(decrypted)[:10] + "..." if len(str(decrypted)) > 10 else str(decrypted)
            print(f"   ✅ {key}: {preview}")
        except Exception as e:
            fail_count += 1
            print(f"   ❌ {key}: DECRYPTION FAILED - {type(e).__name__}")

    await conn.close()
    print()

    # Step 7: Summary and recommendations
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Successfully decrypted: {success_count}/{success_count + fail_count}")
    print(f"Failed to decrypt: {fail_count}/{success_count + fail_count}")
    print()

    if fail_count > 0:
        print("❌ PROBLEM IDENTIFIED:")
        print("   The data in the database was encrypted with a DIFFERENT key than")
        print(f"   your current ENCRYPTION_KEY ({encryption_key[:20]}...)")
        print()
        print("✅ SOLUTION:")
        print("   1. Keep your current ENCRYPTION_KEY in .env (DO NOT change it!)")
        print("   2. Clear and re-import sensitive configs:")
        print()
        print("      docker exec -it trading-bot python -c \"")
        print("      import asyncio, asyncpg, os")
        print("      async def clear():")
        print("          conn = await asyncpg.connect(os.getenv('DATABASE_URL'))")
        print("          await conn.execute('DELETE FROM config_sensitive')")
        print("          await conn.close()")
        print("      asyncio.run(clear())")
        print("      \"")
        print()
        print("   3. Re-import with your current key:")
        print("      docker exec -it trading-bot python scripts/import_env_secrets.py")
        print()
        print("   4. Rebuild:")
        print("      docker-compose up -d --build")
        print()
    else:
        print("✅ EVERYTHING LOOKS GOOD!")
        print("   All configs decrypt successfully with your current ENCRYPTION_KEY")
        print()
        print("   If Edit button still shows empty fields:")
        print("   1. Hard refresh your browser (Ctrl+Shift+R)")
        print("   2. Check browser console for errors (F12 → Console)")
        print("   3. Check docker logs: docker logs trading-bot --tail 50")

if __name__ == "__main__":
    asyncio.run(main())
