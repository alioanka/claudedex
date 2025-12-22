#!/usr/bin/env python3
"""
Force Re-Import Credentials from .env to Database

This script will:
1. Read all credentials from .env
2. Re-encrypt them with the CURRENT encryption key
3. Update the database, overwriting existing encrypted values

Use this when:
- You changed your encryption key
- Previous migration used wrong encryption key
- You want to sync .env values to database

Usage:
    docker exec -it trading-bot python scripts/force_reimport_credentials.py
"""

import asyncio
import os
import sys
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Credential mappings (same as in credentials_routes.py)
CREDENTIAL_MAPPINGS = {
    # Security
    'ENCRYPTION_KEY': {'category': 'security', 'subcategory': 'encryption', 'is_sensitive': True, 'is_required': True, 'display_name': 'Master Encryption Key'},
    'JWT_SECRET': {'category': 'security', 'subcategory': 'auth', 'is_sensitive': True, 'display_name': 'JWT Secret'},
    'SESSION_SECRET': {'category': 'security', 'subcategory': 'auth', 'is_sensitive': True, 'display_name': 'Session Secret'},

    # Wallet
    'PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'evm', 'is_sensitive': True, 'is_required': True, 'display_name': 'EVM Private Key'},
    'WALLET_ADDRESS': {'category': 'wallet', 'subcategory': 'evm', 'is_sensitive': False, 'is_required': True, 'display_name': 'EVM Wallet Address'},
    'SOLANA_PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': True, 'display_name': 'Solana Private Key (DEX)'},
    'SOLANA_WALLET': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': False, 'display_name': 'Solana Wallet (DEX)'},
    'SOLANA_MODULE_PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': True, 'display_name': 'Solana Private Key (Module)'},
    'SOLANA_MODULE_WALLET': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': False, 'display_name': 'Solana Wallet (Module)'},
    'FLASHBOTS_SIGNING_KEY': {'category': 'wallet', 'subcategory': 'flashbots', 'is_sensitive': True, 'display_name': 'Flashbots Signing Key'},

    # Exchange
    'BINANCE_API_KEY': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True, 'display_name': 'Binance API Key (Mainnet)'},
    'BINANCE_API_SECRET': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True, 'display_name': 'Binance API Secret (Mainnet)'},
    'BINANCE_TESTNET_API_KEY': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True, 'display_name': 'Binance API Key (Testnet)'},
    'BINANCE_TESTNET_API_SECRET': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True, 'display_name': 'Binance API Secret (Testnet)'},
    'BYBIT_API_KEY': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True, 'display_name': 'Bybit API Key (Mainnet)'},
    'BYBIT_API_SECRET': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True, 'display_name': 'Bybit API Secret (Mainnet)'},
    'BYBIT_TESTNET_API_KEY': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True, 'display_name': 'Bybit API Key (Testnet)'},
    'BYBIT_TESTNET_API_SECRET': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True, 'display_name': 'Bybit API Secret (Testnet)'},

    # Database
    'DB_PASSWORD': {'category': 'database', 'subcategory': 'postgres', 'is_sensitive': True, 'is_required': True, 'display_name': 'Database Password'},
    'DATABASE_URL': {'category': 'database', 'subcategory': 'postgres', 'is_sensitive': True, 'is_required': True, 'display_name': 'Database URL'},
    'DB_URL': {'category': 'database', 'subcategory': 'postgres', 'is_sensitive': True, 'display_name': 'Database URL (alternate)'},
    'REDIS_PASSWORD': {'category': 'database', 'subcategory': 'redis', 'is_sensitive': True, 'display_name': 'Redis Password'},
    'REDIS_URL': {'category': 'database', 'subcategory': 'redis', 'is_sensitive': True, 'display_name': 'Redis URL'},

    # API
    'GOPLUS_API_KEY': {'category': 'api', 'subcategory': 'goplus', 'is_sensitive': True, 'display_name': 'GoPlus API Key'},
    '1INCH_API_KEY': {'category': 'api', 'subcategory': '1inch', 'is_sensitive': True, 'display_name': '1inch API Key'},
    'HELIUS_API_KEY': {'category': 'api', 'subcategory': 'helius', 'is_sensitive': True, 'display_name': 'Helius API Key'},
    'JUPITER_API_KEY': {'category': 'api', 'subcategory': 'jupiter', 'is_sensitive': True, 'display_name': 'Jupiter API Key'},
    'ETHERSCAN_API_KEY': {'category': 'api', 'subcategory': 'etherscan', 'is_sensitive': True, 'display_name': 'Etherscan API Key'},
    'OPENAI_API_KEY': {'category': 'api', 'subcategory': 'openai', 'is_sensitive': True, 'display_name': 'OpenAI API Key'},
    'ANTHROPIC_API_KEY': {'category': 'api', 'subcategory': 'anthropic', 'is_sensitive': True, 'display_name': 'Anthropic API Key'},

    # Notification
    'TELEGRAM_BOT_TOKEN': {'category': 'notification', 'subcategory': 'telegram', 'is_sensitive': True, 'display_name': 'Telegram Bot Token'},
    'TELEGRAM_CHAT_ID': {'category': 'notification', 'subcategory': 'telegram', 'is_sensitive': False, 'display_name': 'Telegram Chat ID'},
    'DISCORD_WEBHOOK_URL': {'category': 'notification', 'subcategory': 'discord', 'is_sensitive': True, 'display_name': 'Discord Webhook URL'},
    'TWITTER_API_KEY': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True, 'display_name': 'Twitter API Key'},
    'TWITTER_API_SECRET': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True, 'display_name': 'Twitter API Secret'},
    'TWITTER_BEARER_TOKEN': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True, 'display_name': 'Twitter Bearer Token'},
    'EMAIL_PASSWORD': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': True, 'display_name': 'Email Password'},
    'EMAIL_FROM': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': False, 'display_name': 'Email From Address'},
    'EMAIL_TO': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': False, 'display_name': 'Email To Address'},
}


def get_encryption_key():
    """Get encryption key from file or environment"""
    # Try file first (preferred for security)
    key_file = Path('.encryption_key')
    if key_file.exists():
        key = key_file.read_text().strip()
        logger.info(f"Using encryption key from file: {key_file}")
        return key

    # Fallback to environment
    key = os.getenv('ENCRYPTION_KEY')
    if key:
        logger.info("Using encryption key from environment")
        return key

    raise ValueError("No encryption key found! Set ENCRYPTION_KEY or create .encryption_key file")


async def force_reimport():
    """Force re-import all credentials from .env"""
    import asyncpg

    # Load .env
    load_dotenv()

    # Get encryption key
    encryption_key = get_encryption_key()

    # Setup Fernet encryption
    try:
        from cryptography.fernet import Fernet
        fernet = Fernet(encryption_key.encode())
        logger.info("✅ Fernet encryption initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Fernet: {e}")
        logger.error("   Make sure your encryption key is a valid Fernet key (32 url-safe base64 bytes)")
        return

    # Test decryption with a known value
    try:
        test_data = b"test"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        assert decrypted == test_data
        logger.info("✅ Encryption key verified working")
    except Exception as e:
        logger.error(f"❌ Encryption key validation failed: {e}")
        return

    # Connect to database using Docker secrets or environment
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL', 'postgresql://bot_user@postgres:5432/tradingbot')
    try:
        conn = await asyncpg.connect(db_url)
        logger.info("✅ Connected to database")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        return

    try:
        imported = 0
        skipped = 0
        errors = []

        for key_name, metadata in CREDENTIAL_MAPPINGS.items():
            value = os.getenv(key_name)

            # Skip if no value or placeholder
            if not value or value.startswith('your_') or value == 'PLACEHOLDER':
                logger.warning(f"⏭️  Skipping {key_name}: No value in .env")
                skipped += 1
                continue

            try:
                # Encrypt if sensitive
                if metadata.get('is_sensitive', True):
                    encrypted_value = fernet.encrypt(value.encode()).decode()
                    is_encrypted = True
                else:
                    encrypted_value = value
                    is_encrypted = False

                value_hash = hashlib.sha256(value.encode()).hexdigest()

                # Update or insert (upsert)
                await conn.execute("""
                    INSERT INTO secure_credentials
                    (key_name, display_name, encrypted_value, value_hash,
                     category, subcategory, is_sensitive, is_encrypted, is_required,
                     description, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, TRUE)
                    ON CONFLICT (key_name) DO UPDATE SET
                        encrypted_value = $3,
                        value_hash = $4,
                        is_encrypted = $8,
                        updated_at = NOW()
                """,
                    key_name,
                    metadata.get('display_name', key_name),
                    encrypted_value,
                    value_hash,
                    metadata.get('category', 'general'),
                    metadata.get('subcategory', 'general'),
                    metadata.get('is_sensitive', True),
                    is_encrypted,
                    metadata.get('is_required', False),
                    f"{metadata.get('display_name', key_name)} for {metadata.get('subcategory', 'general')}"
                )

                imported += 1
                status = "🔐 encrypted" if is_encrypted else "📝 plaintext"
                logger.info(f"✅ Imported {key_name} ({status})")

            except Exception as e:
                errors.append(f"{key_name}: {str(e)}")
                logger.error(f"❌ Failed to import {key_name}: {e}")

        # Summary
        logger.info("=" * 60)
        logger.info("IMPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Imported: {imported}")
        logger.info(f"⏭️  Skipped (no value): {skipped}")
        logger.info(f"❌ Errors: {len(errors)}")

        if errors:
            logger.info("\nErrors:")
            for error in errors:
                logger.info(f"  - {error}")

        logger.info("=" * 60)
        logger.info("✅ Re-import complete! Restart the bot to use new credentials.")

    finally:
        await conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("FORCE RE-IMPORT CREDENTIALS")
    print("=" * 60)
    print()
    print("This will re-encrypt ALL credentials from .env with the")
    print("current encryption key, overwriting existing database values.")
    print()

    confirm = input("Are you sure you want to continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        sys.exit(0)

    print()
    asyncio.run(force_reimport())
