#!/usr/bin/env python3
"""
Update a single credential in the database.

This script allows you to securely update a credential (like a private key)
by encrypting it with the current encryption key and storing it in the database.

Usage:
    # Interactive mode (prompts for value):
    docker exec -it trading-bot python scripts/update_credential.py PRIVATE_KEY

    # With value (be careful with shell history!):
    docker exec -it trading-bot python scripts/update_credential.py PRIVATE_KEY --value "your_private_key"

    # List available credentials:
    docker exec -it trading-bot python scripts/update_credential.py --list

Examples:
    # Update EVM private key:
    docker exec -it trading-bot python scripts/update_credential.py PRIVATE_KEY

    # Update Solana private key:
    docker exec -it trading-bot python scripts/update_credential.py SOLANA_MODULE_PRIVATE_KEY
"""
import asyncio
import os
import sys
import argparse
import getpass
from pathlib import Path

async def list_credentials():
    """List all credentials in the database."""
    import asyncpg

    db_host = os.getenv('DB_HOST', 'postgres')

    # Read db credentials
    user_file = Path('/run/secrets/db_user')
    pass_file = Path('/run/secrets/db_password')

    if user_file.exists():
        db_user = user_file.read_text().strip()
        db_pass = pass_file.read_text().strip()
    else:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASSWORD')
        if not db_user or not db_pass:
            print("ERROR: Database credentials not available")
            return

    pool = await asyncpg.create_pool(
        host=db_host, port=5432, database='tradingbot',
        user=db_user, password=db_pass
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT key_name, category,
                   CASE WHEN encrypted_value = 'PLACEHOLDER' THEN 'Not Set'
                        WHEN encrypted_value LIKE 'gAAAAAB%' THEN 'Encrypted'
                        ELSE 'Plain' END as status
            FROM secure_credentials
            WHERE is_active = TRUE
            ORDER BY category, key_name
        """)

    print("\n" + "=" * 60)
    print("AVAILABLE CREDENTIALS")
    print("=" * 60)

    current_category = None
    for row in rows:
        if row['category'] != current_category:
            current_category = row['category']
            print(f"\n[{current_category.upper()}]")

        status_icon = "✅" if row['status'] == 'Encrypted' else "⚪" if row['status'] == 'Not Set' else "⚠️"
        print(f"  {status_icon} {row['key_name']}: {row['status']}")

    print("\n" + "=" * 60)
    await pool.close()


async def update_credential(key_name: str, value: str):
    """Update a credential in the database."""
    from cryptography.fernet import Fernet
    import asyncpg

    # Load encryption key
    key_file = Path('.encryption_key')
    if not key_file.exists():
        print("ERROR: .encryption_key not found!")
        return False

    encryption_key = key_file.read_bytes().strip()
    fernet = Fernet(encryption_key)

    # IMPORTANT: Check if value is already encrypted - prevent double-encryption!
    if value.startswith('gAAAAAB'):
        print("ERROR: The value you provided appears to already be Fernet-encrypted!")
        print("       Please provide the PLAINTEXT value, not an encrypted one.")
        print("       This prevents double-encryption issues.")
        return False

    # Encrypt the value
    encrypted_value = fernet.encrypt(value.encode()).decode()

    # Verify we can decrypt it (sanity check)
    try:
        decrypted = fernet.decrypt(encrypted_value.encode()).decode()
        if decrypted != value:
            print("ERROR: Encryption verification failed!")
            return False
    except Exception as e:
        print(f"ERROR: Encryption verification failed: {e}")
        return False

    # Connect to database
    db_host = os.getenv('DB_HOST', 'postgres')

    user_file = Path('/run/secrets/db_user')
    pass_file = Path('/run/secrets/db_password')

    if user_file.exists():
        db_user = user_file.read_text().strip()
        db_pass = pass_file.read_text().strip()
    else:
        db_user = os.getenv('DB_USER')
        db_pass = os.getenv('DB_PASSWORD')
        if not db_user or not db_pass:
            print("ERROR: Database credentials not available")
            return False

    pool = await asyncpg.create_pool(
        host=db_host, port=5432, database='tradingbot',
        user=db_user, password=db_pass
    )

    # Check if credential exists
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("""
            SELECT id, key_name FROM secure_credentials
            WHERE key_name = $1 AND is_active = TRUE
        """, key_name)

        if not existing:
            print(f"ERROR: Credential '{key_name}' not found in database!")
            print("       Use --list to see available credentials.")
            await pool.close()
            return False

        # Update the credential
        await conn.execute("""
            UPDATE secure_credentials
            SET encrypted_value = $1,
                is_encrypted = TRUE,
                updated_at = NOW(),
                encryption_version = COALESCE(encryption_version, 0) + 1
            WHERE key_name = $2
        """, encrypted_value, key_name)

    await pool.close()

    print(f"\n✅ Successfully updated {key_name}")
    print(f"   Encrypted value length: {len(encrypted_value)}")
    print(f"\n⚠️  Remember to restart the trading bot to use the new value:")
    print(f"   docker-compose restart trading-bot")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Update a credential in the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all credentials:
    python scripts/update_credential.py --list

  Update EVM private key (interactive):
    python scripts/update_credential.py PRIVATE_KEY

  Update Solana private key:
    python scripts/update_credential.py SOLANA_MODULE_PRIVATE_KEY

  Update with value directly (careful with shell history!):
    python scripts/update_credential.py BINANCE_API_KEY --value "your_api_key"
        """
    )

    parser.add_argument('key_name', nargs='?', help='The credential key name to update')
    parser.add_argument('--value', help='The new value (if not provided, will prompt securely)')
    parser.add_argument('--list', action='store_true', help='List all available credentials')

    args = parser.parse_args()

    if args.list:
        asyncio.run(list_credentials())
        return

    if not args.key_name:
        parser.print_help()
        return

    # Get the value
    if args.value:
        value = args.value
    else:
        print(f"\nEnter new value for {args.key_name}")
        print("(input is hidden for security)")
        value = getpass.getpass("Value: ")

        if not value:
            print("ERROR: No value provided!")
            return

        # Confirm for sensitive keys
        if 'PRIVATE' in args.key_name or 'SECRET' in args.key_name:
            confirm = getpass.getpass("Confirm value: ")
            if value != confirm:
                print("ERROR: Values don't match!")
                return

    # Update the credential
    success = asyncio.run(update_credential(args.key_name, value))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
