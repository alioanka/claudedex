#!/usr/bin/env python3
"""Debug script to diagnose credential decryption issues."""
import asyncio
import os
from pathlib import Path

async def debug_private_key():
    print("=" * 60)
    print("CREDENTIAL DECRYPTION DEBUG")
    print("=" * 60)

    # Test 1: Check encryption key
    key_file = Path('.encryption_key')
    if key_file.exists():
        key = key_file.read_bytes()
        print(f"\n1. Encryption key file: ✅ Found")
        print(f"   Size: {len(key)} bytes")
        print(f"   First 20 chars: {key[:20]}")
        print(f"   Stripped size: {len(key.strip())} bytes")
    else:
        print(f"\n1. Encryption key file: ❌ NOT FOUND at {key_file.absolute()}")
        return

    # Test 2: Create Fernet
    from cryptography.fernet import Fernet
    try:
        # Try with stripped key (what secrets_manager does)
        fernet = Fernet(key.strip())
        print(f"\n2. Fernet (stripped): ✅ Created successfully")
    except Exception as e:
        print(f"\n2. Fernet (stripped): ❌ ERROR: {e}")
        try:
            # Try without stripping
            fernet = Fernet(key)
            print(f"   Fernet (raw): ✅ Created successfully")
        except Exception as e2:
            print(f"   Fernet (raw): ❌ ERROR: {e2}")
            return

    # Test 3: Connect to database
    import asyncpg

    db_host = os.getenv('DB_HOST', 'postgres')

    # Read db credentials from secrets
    user_file = Path('/run/secrets/db_user')
    pass_file = Path('/run/secrets/db_password')

    if user_file.exists():
        db_user = user_file.read_text().strip()
        db_pass = pass_file.read_text().strip()
        print(f"\n3. Docker secrets: ✅ Found (user: {db_user})")
    else:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASSWORD')
        print(f"\n3. Docker secrets: ⚠️ Using env vars (user: {db_user})")

    try:
        pool = await asyncpg.create_pool(
            host=db_host, port=5432, database='tradingbot',
            user=db_user, password=db_pass
        )
        print(f"   Database: ✅ Connected to {db_host}")
    except Exception as e:
        print(f"   Database: ❌ ERROR: {e}")
        return

    # Test 4: Query PRIVATE_KEY
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT encrypted_value, is_encrypted, is_sensitive
                FROM secure_credentials
                WHERE key_name = 'PRIVATE_KEY' AND is_active = TRUE
            """)

        if not row:
            print(f"\n4. PRIVATE_KEY: ❌ Not found in database!")
            return

        print(f"\n4. Database row:")
        print(f"   is_encrypted: {row['is_encrypted']} (type: {type(row['is_encrypted']).__name__})")
        print(f"   is_sensitive: {row['is_sensitive']} (type: {type(row['is_sensitive']).__name__})")
        print(f"   encrypted_value starts with: {row['encrypted_value'][:30]}...")
        print(f"   encrypted_value length: {len(row['encrypted_value'])}")

    except Exception as e:
        print(f"\n4. Database query: ❌ ERROR: {e}")
        return

    # Test 5: Try to decrypt
    encrypted_value = row['encrypted_value']
    print(f"\n5. Decryption test:")

    try:
        decrypted = fernet.decrypt(encrypted_value.encode())
        print(f"   Result: ✅ SUCCESS!")
        print(f"   Decrypted length: {len(decrypted)} bytes")
        # Show first few chars (safe - it's a private key prefix, not the whole thing)
        print(f"   First 10 chars: {decrypted[:10].decode()}...")
    except Exception as e:
        print(f"   Result: ❌ FAILED: {type(e).__name__}: {e}")

        # Additional debug
        print(f"\n   Debug info:")
        print(f"   - Key used (first 20): {key.strip()[:20]}")
        print(f"   - Encrypted value (first 50): {encrypted_value[:50]}")

    # Test 6: Check secrets_manager state
    print(f"\n6. Secrets Manager state:")
    try:
        from security.secrets_manager import secrets
        print(f"   _fernet: {'✅ Set' if secrets._fernet else '❌ None'}")
        print(f"   _db_pool: {'✅ Set' if secrets._db_pool else '❌ None'}")
        print(f"   _initialized: {secrets._initialized}")
        print(f"   _bootstrap_mode: {secrets._bootstrap_mode}")

        # Initialize with db pool
        secrets.initialize(pool)
        print(f"   After initialize():")
        print(f"   _fernet: {'✅ Set' if secrets._fernet else '❌ None'}")
        print(f"   _db_pool: {'✅ Set' if secrets._db_pool else '❌ None'}")

        # Try get_async
        print(f"\n7. Testing get_async('PRIVATE_KEY'):")
        result = await secrets.get_async('PRIVATE_KEY', log_access=False)
        if result:
            print(f"   Result: ✅ Got value (length: {len(result)})")
            if result.startswith('gAAAAAB'):
                print(f"   ⚠️ WARNING: Value is still encrypted!")
            else:
                print(f"   First 10 chars: {result[:10]}...")
        else:
            print(f"   Result: ❌ None returned")

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

    await pool.close()
    print(f"\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(debug_private_key())
