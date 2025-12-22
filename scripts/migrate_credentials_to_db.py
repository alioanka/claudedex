#!/usr/bin/env python3
"""
Migrate Credentials from .env to Database

This script migrates sensitive credentials from the .env file to the
secure_credentials database table with encryption.

SECURITY ARCHITECTURE:
1. Reads credentials from .env
2. Encrypts sensitive values using ENCRYPTION_KEY
3. Stores in secure_credentials table
4. Validates the migration

Usage:
    # Inside Docker container:
    docker exec -it claudedex_dex python scripts/migrate_credentials_to_db.py

    # Or directly:
    python scripts/migrate_credentials_to_db.py --dry-run  # Preview changes
    python scripts/migrate_credentials_to_db.py            # Execute migration

Options:
    --dry-run       Preview what would be migrated without making changes
    --validate      Only validate existing credentials
    --export        Export current DB credentials to .env.migrated
"""

import os
import sys
import argparse
import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Try to import required packages
try:
    import asyncpg
    from cryptography.fernet import Fernet
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install asyncpg cryptography")
    DEPS_AVAILABLE = False


# Credential mappings with metadata
CREDENTIAL_MAPPINGS = {
    # Security & Encryption
    'ENCRYPTION_KEY': {
        'category': 'security',
        'subcategory': 'encryption',
        'display_name': 'Master Encryption Key',
        'description': 'Fernet key for encrypting/decrypting sensitive data',
        'is_sensitive': True,
        'is_required': True,
        'module': 'all'
    },
    'JWT_SECRET': {
        'category': 'security',
        'subcategory': 'auth',
        'display_name': 'JWT Secret',
        'description': 'Secret for signing JWT tokens',
        'is_sensitive': True,
        'module': 'all'
    },
    'SESSION_SECRET': {
        'category': 'security',
        'subcategory': 'auth',
        'display_name': 'Session Secret',
        'description': 'Secret for session management',
        'is_sensitive': True,
        'module': 'all'
    },

    # Wallet Credentials
    'PRIVATE_KEY': {
        'category': 'wallet',
        'subcategory': 'evm',
        'display_name': 'EVM Private Key',
        'description': 'Private key for EVM chains (Ethereum, BSC, etc.)',
        'is_sensitive': True,
        'is_required': True,
        'module': 'dex'
    },
    'WALLET_ADDRESS': {
        'category': 'wallet',
        'subcategory': 'evm',
        'display_name': 'EVM Wallet Address',
        'description': 'Public wallet address for EVM chains',
        'is_sensitive': False,
        'is_required': True,
        'module': 'dex'
    },
    'SOLANA_PRIVATE_KEY': {
        'category': 'wallet',
        'subcategory': 'solana',
        'display_name': 'Solana Private Key (DEX)',
        'description': 'Private key for Solana DEX trading',
        'is_sensitive': True,
        'module': 'dex'
    },
    'SOLANA_WALLET': {
        'category': 'wallet',
        'subcategory': 'solana',
        'display_name': 'Solana Wallet (DEX)',
        'description': 'Public wallet address for Solana DEX trading',
        'is_sensitive': False,
        'module': 'dex'
    },
    'SOLANA_MODULE_PRIVATE_KEY': {
        'category': 'wallet',
        'subcategory': 'solana',
        'display_name': 'Solana Private Key (Module)',
        'description': 'Private key for Solana Module',
        'is_sensitive': True,
        'module': 'solana'
    },
    'SOLANA_MODULE_WALLET': {
        'category': 'wallet',
        'subcategory': 'solana',
        'display_name': 'Solana Wallet (Module)',
        'description': 'Public wallet address for Solana Module',
        'is_sensitive': False,
        'module': 'solana'
    },
    'FLASHBOTS_SIGNING_KEY': {
        'category': 'wallet',
        'subcategory': 'flashbots',
        'display_name': 'Flashbots Signing Key',
        'description': 'Private key for signing Flashbots bundles',
        'is_sensitive': True,
        'module': 'dex'
    },

    # Exchange API Keys
    'BINANCE_API_KEY': {
        'category': 'exchange',
        'subcategory': 'binance',
        'display_name': 'Binance API Key (Mainnet)',
        'description': 'Binance mainnet API key',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BINANCE_API_SECRET': {
        'category': 'exchange',
        'subcategory': 'binance',
        'display_name': 'Binance API Secret (Mainnet)',
        'description': 'Binance mainnet API secret',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BINANCE_TESTNET_API_KEY': {
        'category': 'exchange',
        'subcategory': 'binance',
        'display_name': 'Binance API Key (Testnet)',
        'description': 'Binance testnet API key',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BINANCE_TESTNET_API_SECRET': {
        'category': 'exchange',
        'subcategory': 'binance',
        'display_name': 'Binance API Secret (Testnet)',
        'description': 'Binance testnet API secret',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BYBIT_API_KEY': {
        'category': 'exchange',
        'subcategory': 'bybit',
        'display_name': 'Bybit API Key (Mainnet)',
        'description': 'Bybit mainnet API key',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BYBIT_API_SECRET': {
        'category': 'exchange',
        'subcategory': 'bybit',
        'display_name': 'Bybit API Secret (Mainnet)',
        'description': 'Bybit mainnet API secret',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BYBIT_TESTNET_API_KEY': {
        'category': 'exchange',
        'subcategory': 'bybit',
        'display_name': 'Bybit API Key (Testnet)',
        'description': 'Bybit testnet API key',
        'is_sensitive': True,
        'module': 'futures'
    },
    'BYBIT_TESTNET_API_SECRET': {
        'category': 'exchange',
        'subcategory': 'bybit',
        'display_name': 'Bybit API Secret (Testnet)',
        'description': 'Bybit testnet API secret',
        'is_sensitive': True,
        'module': 'futures'
    },

    # Database & Cache
    'DB_PASSWORD': {
        'category': 'database',
        'subcategory': 'postgres',
        'display_name': 'Database Password',
        'description': 'PostgreSQL database password',
        'is_sensitive': True,
        'is_required': True,
        'module': 'all'
    },
    'REDIS_PASSWORD': {
        'category': 'database',
        'subcategory': 'redis',
        'display_name': 'Redis Password',
        'description': 'Redis cache password',
        'is_sensitive': True,
        'module': 'all'
    },

    # Third-Party APIs
    'GOPLUS_API_KEY': {
        'category': 'api',
        'subcategory': 'goplus',
        'display_name': 'GoPlus API Key',
        'description': 'API key for GoPlus token security checks',
        'is_sensitive': True,
        'module': 'dex'
    },
    '1INCH_API_KEY': {
        'category': 'api',
        'subcategory': '1inch',
        'display_name': '1inch API Key',
        'description': 'API key for 1inch DEX aggregator',
        'is_sensitive': True,
        'module': 'dex'
    },
    'HELIUS_API_KEY': {
        'category': 'api',
        'subcategory': 'helius',
        'display_name': 'Helius API Key',
        'description': 'API key for Helius Solana RPC',
        'is_sensitive': True,
        'module': 'solana'
    },
    'JUPITER_API_KEY': {
        'category': 'api',
        'subcategory': 'jupiter',
        'display_name': 'Jupiter API Key',
        'description': 'API key for Jupiter aggregator',
        'is_sensitive': True,
        'module': 'solana'
    },
    'ETHERSCAN_API_KEY': {
        'category': 'api',
        'subcategory': 'etherscan',
        'display_name': 'Etherscan API Key',
        'description': 'API key for Etherscan',
        'is_sensitive': True,
        'module': 'copytrading'
    },

    # Notification Credentials
    'TELEGRAM_BOT_TOKEN': {
        'category': 'notification',
        'subcategory': 'telegram',
        'display_name': 'Telegram Bot Token',
        'description': 'Token for Telegram notification bot',
        'is_sensitive': True,
        'module': 'all'
    },
    'TELEGRAM_CHAT_ID': {
        'category': 'notification',
        'subcategory': 'telegram',
        'display_name': 'Telegram Chat ID',
        'description': 'Chat ID for Telegram notifications',
        'is_sensitive': False,
        'module': 'all'
    },
    'DISCORD_WEBHOOK_URL': {
        'category': 'notification',
        'subcategory': 'discord',
        'display_name': 'Discord Webhook URL',
        'description': 'Webhook URL for Discord notifications',
        'is_sensitive': True,
        'module': 'all'
    },
    'TWITTER_API_KEY': {
        'category': 'notification',
        'subcategory': 'twitter',
        'display_name': 'Twitter API Key',
        'description': 'Twitter API key for sentiment analysis',
        'is_sensitive': True,
        'module': 'ai'
    },
    'TWITTER_API_SECRET': {
        'category': 'notification',
        'subcategory': 'twitter',
        'display_name': 'Twitter API Secret',
        'description': 'Twitter API secret',
        'is_sensitive': True,
        'module': 'ai'
    },
    'TWITTER_BEARER_TOKEN': {
        'category': 'notification',
        'subcategory': 'twitter',
        'display_name': 'Twitter Bearer Token',
        'description': 'Twitter bearer token',
        'is_sensitive': True,
        'module': 'ai'
    },
    'EMAIL_PASSWORD': {
        'category': 'notification',
        'subcategory': 'email',
        'display_name': 'Email Password',
        'description': 'Password for email notifications',
        'is_sensitive': True,
        'module': 'all'
    },
    'EMAIL_FROM': {
        'category': 'notification',
        'subcategory': 'email',
        'display_name': 'Email From Address',
        'description': 'Sender email address',
        'is_sensitive': False,
        'module': 'all'
    },
    'EMAIL_TO': {
        'category': 'notification',
        'subcategory': 'email',
        'display_name': 'Email To Address',
        'description': 'Recipient email address',
        'is_sensitive': False,
        'module': 'all'
    },
}


class CredentialMigrator:
    """Handles migration of credentials from .env to database"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fernet = None
        self.conn = None
        self.stats = {
            'migrated': 0,
            'skipped': 0,
            'errors': 0,
            'already_set': 0
        }

    async def connect(self) -> bool:
        """Connect to database"""
        db_url = os.getenv('DATABASE_URL') or os.getenv('DB_URL')
        if not db_url:
            logger.error("DATABASE_URL not set in environment")
            return False

        try:
            self.conn = await asyncpg.connect(db_url)
            logger.info("Connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def init_encryption(self) -> bool:
        """Initialize encryption"""
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            logger.warning("ENCRYPTION_KEY not set - values will be stored unencrypted")
            return False

        try:
            self.fernet = Fernet(encryption_key.encode())
            logger.info("Encryption initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return False

    async def run_migration(self):
        """Run the migration"""
        logger.info("="*60)
        logger.info("CREDENTIAL MIGRATION TO DATABASE")
        logger.info("="*60)

        if self.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")

        # Initialize encryption
        self.init_encryption()

        # Connect to database
        if not await self.connect():
            return False

        try:
            # Check if migration table exists
            await self._ensure_table_exists()

            # Process each credential
            for key_name, metadata in CREDENTIAL_MAPPINGS.items():
                await self._migrate_credential(key_name, metadata)

            # Print summary
            self._print_summary()

            return True

        finally:
            if self.conn:
                await self.conn.close()

    async def _ensure_table_exists(self):
        """Ensure secure_credentials table exists"""
        try:
            # Check if table exists
            exists = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'secure_credentials'
                )
            """)

            if not exists:
                logger.info("secure_credentials table not found - running migration...")
                # Read and execute migration file
                migration_file = Path(__file__).parent.parent / 'migrations' / '012_add_secure_credentials_table.sql'
                if migration_file.exists():
                    sql = migration_file.read_text()
                    await self.conn.execute(sql)
                    logger.info("Created secure_credentials table")
                else:
                    logger.error(f"Migration file not found: {migration_file}")
                    raise FileNotFoundError(f"Migration file not found: {migration_file}")

        except Exception as e:
            logger.error(f"Error ensuring table exists: {e}")
            raise

    async def _migrate_credential(self, key_name: str, metadata: dict):
        """Migrate a single credential"""
        # Get value from environment
        value = os.getenv(key_name)

        if not value or value.startswith('your_') or value == 'PLACEHOLDER':
            logger.debug(f"Skipping {key_name} - no value or placeholder")
            self.stats['skipped'] += 1
            return

        try:
            # Check if already set in database
            existing = await self.conn.fetchrow("""
                SELECT encrypted_value FROM secure_credentials WHERE key_name = $1
            """, key_name)

            if existing and existing['encrypted_value'] != 'PLACEHOLDER':
                logger.info(f"[SKIP] {key_name} - already configured in database")
                self.stats['already_set'] += 1
                return

            # Encrypt if sensitive
            encrypted_value = value
            is_encrypted = False
            if metadata.get('is_sensitive', True) and self.fernet:
                encrypted_value = self.fernet.encrypt(value.encode()).decode()
                is_encrypted = True

            # Create hash
            value_hash = hashlib.sha256(value.encode()).hexdigest()

            if self.dry_run:
                logger.info(f"[DRY RUN] Would migrate: {key_name} ({metadata.get('category')})")
            else:
                # Insert or update
                await self.conn.execute("""
                    INSERT INTO secure_credentials
                    (key_name, display_name, description, encrypted_value, value_hash,
                     category, subcategory, module, is_sensitive, is_encrypted, is_required)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (key_name) DO UPDATE SET
                        encrypted_value = $4,
                        value_hash = $5,
                        is_encrypted = $10,
                        updated_at = NOW()
                """,
                    key_name,
                    metadata.get('display_name', key_name),
                    metadata.get('description', ''),
                    encrypted_value,
                    value_hash,
                    metadata.get('category', 'general'),
                    metadata.get('subcategory'),
                    metadata.get('module', 'all'),
                    metadata.get('is_sensitive', True),
                    is_encrypted,
                    metadata.get('is_required', False)
                )
                logger.info(f"[OK] Migrated: {key_name}")

            self.stats['migrated'] += 1

        except Exception as e:
            logger.error(f"[ERROR] Failed to migrate {key_name}: {e}")
            self.stats['errors'] += 1

    def _print_summary(self):
        """Print migration summary"""
        logger.info("")
        logger.info("="*60)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"  Migrated:    {self.stats['migrated']}")
        logger.info(f"  Already Set: {self.stats['already_set']}")
        logger.info(f"  Skipped:     {self.stats['skipped']}")
        logger.info(f"  Errors:      {self.stats['errors']}")
        logger.info("="*60)

        if self.dry_run:
            logger.info("This was a DRY RUN. Run without --dry-run to execute.")
        else:
            logger.info("Migration complete!")
            logger.info("")
            logger.info("NEXT STEPS:")
            logger.info("1. Verify credentials in dashboard at /credentials")
            logger.info("2. Test that modules can access credentials")
            logger.info("3. Remove sensitive values from .env file")
            logger.info("4. Move ENCRYPTION_KEY to /secure/encryption.key or Docker secret")


async def validate_credentials():
    """Validate all credentials in database"""
    db_url = os.getenv('DATABASE_URL') or os.getenv('DB_URL')
    conn = await asyncpg.connect(db_url)

    try:
        rows = await conn.fetch("""
            SELECT key_name, encrypted_value, is_required, is_sensitive, is_encrypted
            FROM secure_credentials
            WHERE is_active = TRUE
        """)

        missing_required = []
        configured = []
        not_configured = []

        for row in rows:
            if row['encrypted_value'] == 'PLACEHOLDER':
                if row['is_required']:
                    missing_required.append(row['key_name'])
                not_configured.append(row['key_name'])
            else:
                configured.append(row['key_name'])

        logger.info("")
        logger.info("="*60)
        logger.info("CREDENTIAL VALIDATION")
        logger.info("="*60)
        logger.info(f"Configured:      {len(configured)}")
        logger.info(f"Not Configured:  {len(not_configured)}")
        logger.info(f"Missing Required: {len(missing_required)}")

        if missing_required:
            logger.warning("")
            logger.warning("MISSING REQUIRED CREDENTIALS:")
            for key in missing_required:
                logger.warning(f"  - {key}")

        logger.info("="*60)

    finally:
        await conn.close()


async def export_to_env():
    """Export credentials from database to .env.migrated file"""
    db_url = os.getenv('DATABASE_URL') or os.getenv('DB_URL')
    encryption_key = os.getenv('ENCRYPTION_KEY')

    fernet = None
    if encryption_key:
        try:
            fernet = Fernet(encryption_key.encode())
        except:
            pass

    conn = await asyncpg.connect(db_url)

    try:
        rows = await conn.fetch("""
            SELECT key_name, encrypted_value, is_encrypted
            FROM secure_credentials
            WHERE is_active = TRUE AND encrypted_value != 'PLACEHOLDER'
        """)

        output_file = Path('.env.migrated')
        with open(output_file, 'w') as f:
            f.write(f"# Exported from database at {datetime.now().isoformat()}\n")
            f.write("# WARNING: This file contains sensitive data - DO NOT COMMIT\n\n")

            for row in rows:
                value = row['encrypted_value']
                if row['is_encrypted'] and fernet:
                    try:
                        value = fernet.decrypt(value.encode()).decode()
                    except:
                        value = f"# ENCRYPTED: {value[:20]}..."

                f.write(f"{row['key_name']}={value}\n")

        os.chmod(output_file, 0o600)
        logger.info(f"Exported to {output_file}")

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Migrate credentials from .env to database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Preview migration:
    python scripts/migrate_credentials_to_db.py --dry-run

  Execute migration:
    python scripts/migrate_credentials_to_db.py

  Validate credentials:
    python scripts/migrate_credentials_to_db.py --validate

  Export to .env file:
    python scripts/migrate_credentials_to_db.py --export
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without executing')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing credentials')
    parser.add_argument('--export', action='store_true',
                        help='Export credentials to .env.migrated')

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        sys.exit(1)

    if args.validate:
        asyncio.run(validate_credentials())
    elif args.export:
        asyncio.run(export_to_env())
    else:
        migrator = CredentialMigrator(dry_run=args.dry_run)
        asyncio.run(migrator.run_migration())


if __name__ == '__main__':
    main()
