#!/usr/bin/env python3
"""
Import secrets from .env file to config_sensitive database table
This script reads sensitive configuration from .env and encrypts them in the database
"""
import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet

# Add parent directory to path to import from security module
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.encryption import EncryptionManager

# List of sensitive keys to import from .env
SENSITIVE_KEYS = {
    'ENCRYPTION_KEY': 'Master encryption key for secure data',
    'PRIVATE_KEY': 'Ethereum wallet private key',
    'SOLANA_PRIVATE_KEY': 'Solana wallet private key',
    'WALLET_ADDRESS': 'Ethereum wallet address',
    'SOLANA_WALLET': 'Solana wallet address',
    'DEXSCREENER_API_KEY': 'DexScreener API key',
    'GOPLUS_API_KEY': 'GoPlus security API key',
    'TOKENSNIFFER_API_KEY': 'TokenSniffer API key',
    '1INCH_API_KEY': '1inch DEX aggregator API key',
    'PARASWAP_API_KEY': 'ParaSwap DEX aggregator API key',
    'TWITTER_API_KEY': 'Twitter API key',
    'TWITTER_API_SECRET': 'Twitter API secret',
    'TWITTER_BEARER_TOKEN': 'Twitter bearer token',
    'TELEGRAM_BOT_TOKEN': 'Telegram bot token for notifications',
    'TELEGRAM_CHAT_ID': 'Telegram chat ID for notifications',
    'DISCORD_WEBHOOK_URL': 'Discord webhook URL for notifications',
    'EMAIL_PASSWORD': 'Email password for notifications',
    'FLASHBOTS_SIGNING_KEY': 'Flashbots signing key for MEV protection',
    'DASHBOARD_API_KEY': 'Dashboard API authentication key',
    'JWT_SECRET': 'JWT token signing secret',
    'SESSION_SECRET': 'Session encryption secret',
    'SENTRY_DSN': 'Sentry error tracking DSN',
    'SOLANA_RPC_URL': 'Primary Solana RPC endpoint URL',
    'SOLANA_BACKUP_RPCS': 'Backup Solana RPC endpoint URLs',
    'ETHEREUM_RPC_URLS': 'Ethereum RPC endpoint URLs',
    'BSC_RPC_URLS': 'Binance Smart Chain RPC URLs',
    'POLYGON_RPC_URLS': 'Polygon RPC URLs',
    'ARBITRUM_RPC_URLS': 'Arbitrum RPC URLs',
    'BASE_RPC_URLS': 'Base chain RPC URLs',
    'WEB3_PROVIDER_URL': 'Primary Web3 provider URL',
    'WEB3_BACKUP_PROVIDER_1': 'Backup Web3 provider #1',
    'WEB3_BACKUP_PROVIDER_2': 'Backup Web3 provider #2',
    'FLASHBOTS_RPC': 'Flashbots RPC endpoint',
    'DATABASE_URL': 'PostgreSQL database connection URL',
    'DB_PASSWORD': 'Database password',
    'REDIS_URL': 'Redis cache connection URL',
    'REDIS_PASSWORD': 'Redis authentication password',
}

async def import_env_secrets():
    """Import .env secrets to config_sensitive table"""

    print("=" * 80)
    print("IMPORT .ENV SECRETS TO DATABASE")
    print("=" * 80)
    print()

    # Get encryption key
    encryption_key = os.getenv('ENCRYPTION_KEY')
    if not encryption_key:
        print("❌ ERROR: ENCRYPTION_KEY not found in environment")
        print("   Please set ENCRYPTION_KEY in your .env file")
        print()
        print("   To generate a new key, run:")
        print("   python scripts/generate_encryption_key.py")
        return False

    # Initialize encryption manager
    try:
        encryption_config = {'encryption_key': encryption_key}
        encryption_manager = EncryptionManager(encryption_config)
        print("✅ Encryption manager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize encryption manager: {e}")
        print()
        print("   Your ENCRYPTION_KEY might not be in the correct format.")
        print("   To generate a valid key, run:")
        print("   python scripts/generate_encryption_key.py")
        return False

    # Connect to database using Docker secrets or environment
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        print(f"Connecting using DATABASE_URL...")
        conn = await asyncpg.connect(database_url)
    else:
        # Try Docker secrets first
        try:
            from security.docker_secrets import get_db_credentials
            db_creds = get_db_credentials()
            print(f"Connecting using Docker secrets...")
            conn = await asyncpg.connect(
                host=db_creds['host'],
                port=int(db_creds['port']),
                database=db_creds['name'],
                user=db_creds['user'],
                password=db_creds['password']
            )
        except ImportError:
            print(f"Connecting using individual DB env vars...")
            conn = await asyncpg.connect(
                host=os.getenv("DB_HOST", "postgres"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "tradingbot"),
                user=os.getenv("DB_USER", "bot_user"),
                password=os.getenv("DB_PASSWORD", "")
            )

    print("✅ Connected to database")
    print()

    try:
        imported_count = 0
        skipped_count = 0
        updated_count = 0

        for key, description in SENSITIVE_KEYS.items():
            value = os.getenv(key)

            # Skip if not in .env or empty
            if not value or value in ('', 'null', 'None'):
                print(f"⊘ Skipping {key} (not set in .env)")
                skipped_count += 1
                continue

            # Check if already exists
            existing = await conn.fetchrow("""
                SELECT key, is_active FROM config_sensitive
                WHERE key = $1
            """, key)

            try:
                # Encrypt the value
                encrypted_value = encryption_manager.encrypt_sensitive_data(value)

                if existing:
                    # Update existing
                    await conn.execute("""
                        UPDATE config_sensitive
                        SET encrypted_value = $1,
                            description = $2,
                            last_rotated = NOW(),
                            is_active = TRUE,
                            updated_at = NOW()
                        WHERE key = $3
                    """, encrypted_value, description, key)
                    print(f"↻ Updated {key}")
                    updated_count += 1
                else:
                    # Insert new
                    await conn.execute("""
                        INSERT INTO config_sensitive (
                            key, encrypted_value, description,
                            last_rotated, is_active
                        ) VALUES ($1, $2, $3, NOW(), TRUE)
                    """, key, encrypted_value, description)
                    print(f"+ Imported {key}")
                    imported_count += 1

            except Exception as e:
                print(f"❌ Failed to process {key}: {e}")
                continue

        print()
        print("=" * 80)
        print(f"✅ Import completed!")
        print(f"   Imported: {imported_count}")
        print(f"   Updated:  {updated_count}")
        print(f"   Skipped:  {skipped_count}")
        print("=" * 80)
        print()
        print("🔒 All sensitive values are now encrypted in the database")
        print("📝 You can view them in the Sensitive Config tab in the dashboard")
        print()

        return True

    except Exception as e:
        print(f"❌ Error during import: {e}")
        return False

    finally:
        await conn.close()

if __name__ == "__main__":
    success = asyncio.run(import_env_secrets())
    sys.exit(0 if success else 1)
