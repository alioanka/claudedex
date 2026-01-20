#!/usr/bin/env python3
"""
Encryption Key Rotation Script for ClaudeDex

This script safely rotates the encryption key used for storing credentials
in the secure_credentials database table.

IMPORTANT: This is a MANUAL migration script - NOT auto-migrated!
           Run this only when you intentionally want to rotate the encryption key.

WHAT THIS SCRIPT DOES:
1. Backs up the current encryption key
2. Reads all encrypted credentials from database
3. Decrypts them with the OLD key
4. Generates a NEW encryption key
5. Re-encrypts all credentials with the NEW key
6. Updates the database
7. Saves the new key to .encryption_key file

USAGE:
    # Method 1: Inside Docker container (if trading-bot is running):
    docker exec -it trading-bot python scripts/rotate_encryption_key.py

    # Method 2: From host with only postgres running (RECOMMENDED):
    # First stop the trading bot:
    docker-compose stop trading-bot

    # Then run from host with explicit credentials:
    DB_HOST=localhost DB_USER=<your_db_user> DB_PASSWORD=<your_db_password> \
        python scripts/rotate_encryption_key.py --dry-run

    # Options:
    python scripts/rotate_encryption_key.py --dry-run     # Preview changes only
    python scripts/rotate_encryption_key.py --backup-only # Just backup current key
    python scripts/rotate_encryption_key.py --verify      # Verify current encryption

REQUIREMENTS:
    - Database must be accessible (postgres container running)
    - .encryption_key file must exist with current key
    - trading-bot container should be STOPPED during rotation
    - Set DB_USER and DB_PASSWORD env vars (or use Docker secrets)

Author: ClaudeDex Security
Date: December 2024
"""

import os
import sys
import argparse
import asyncio
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import asyncpg
    from cryptography.fernet import Fernet, InvalidToken
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    DEPS_AVAILABLE = False


class EncryptionKeyRotator:
    """Handles encryption key rotation for secure_credentials"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.db_pool = None
        self.old_key = None
        self.new_key = None
        self.old_fernet = None
        self.new_fernet = None
        self.key_file = Path('.encryption_key')
        self.backup_dir = Path('backups/encryption_keys')

    async def connect_database(self):
        """Connect to PostgreSQL database"""
        # Get database connection info from Docker secrets or environment
        db_host = os.getenv('DB_HOST', 'postgres')
        db_port = int(os.getenv('DB_PORT', 5432))
        db_name = os.getenv('DB_NAME', 'tradingbot')  # Match docker-compose POSTGRES_DB

        # Try Docker secret first for db_user, then environment
        db_user = None
        user_secret_file = Path('/run/secrets/db_user')
        if user_secret_file.exists():
            db_user = user_secret_file.read_text().strip()
        else:
            db_user = os.getenv('DB_USER')
            if not db_user:
                raise ValueError(
                    "DB_USER not set. Either:\n"
                    "  1. Run inside trading-bot container (has Docker secrets), or\n"
                    "  2. Set DB_USER environment variable explicitly"
                )

        # Try Docker secret first for db_password, then environment
        db_password = None
        password_secret_file = Path('/run/secrets/db_password')
        if password_secret_file.exists():
            db_password = password_secret_file.read_text().strip()
        else:
            db_password = os.getenv('DB_PASSWORD')
            if not db_password:
                raise ValueError(
                    "DB_PASSWORD not set. Either:\n"
                    "  1. Run inside trading-bot container (has Docker secrets), or\n"
                    "  2. Set DB_PASSWORD environment variable explicitly"
                )

        self.db_pool = await asyncpg.create_pool(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            min_size=1,
            max_size=5
        )
        logger.info(f"Connected to database at {db_host}:{db_port}/{db_name}")

    def load_current_key(self) -> bool:
        """Load the current encryption key"""
        if not self.key_file.exists():
            logger.error(f"Encryption key file not found: {self.key_file}")
            return False

        self.old_key = self.key_file.read_bytes()
        try:
            self.old_fernet = Fernet(self.old_key)
            logger.info("Current encryption key loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Invalid encryption key: {e}")
            return False

    def backup_current_key(self) -> str:
        """Backup the current encryption key"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f'encryption_key_{timestamp}.bak'

        shutil.copy2(self.key_file, backup_file)
        os.chmod(backup_file, 0o600)

        logger.info(f"Current key backed up to: {backup_file}")
        return str(backup_file)

    def generate_new_key(self):
        """Generate a new Fernet encryption key"""
        self.new_key = Fernet.generate_key()
        self.new_fernet = Fernet(self.new_key)
        logger.info("New encryption key generated")

    async def get_all_credentials(self) -> list:
        """Get all encrypted credentials from database"""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, key_name, encrypted_value, is_encrypted, category
                FROM secure_credentials
                WHERE is_active = TRUE
                ORDER BY id
            """)
            return [dict(row) for row in rows]

    def decrypt_value(self, encrypted_value: str, key_name: str = "unknown") -> str:
        """
        Decrypt a value with the old key.

        IMPORTANT: This method handles double-encryption by recursively decrypting
        until the value is no longer encrypted (doesn't start with 'gAAAAAB').
        """
        if not encrypted_value or encrypted_value == 'PLACEHOLDER':
            return encrypted_value

        # Check if value looks encrypted (Fernet format)
        if not encrypted_value.startswith('gAAAAAB'):
            return encrypted_value  # Already plaintext

        try:
            decrypted = self.old_fernet.decrypt(encrypted_value.encode())
            result = decrypted.decode()

            # CRITICAL FIX: Check for double-encryption
            # If the decrypted value is STILL encrypted, decrypt again
            depth = 1
            while result.startswith('gAAAAAB') and depth < 5:
                try:
                    logger.warning(f"  ⚠️  {key_name}: Double-encryption detected (layer {depth}), decrypting inner value...")
                    decrypted = self.old_fernet.decrypt(result.encode())
                    result = decrypted.decode()
                    depth += 1
                except InvalidToken:
                    # Inner encryption uses different key - can't decrypt further
                    logger.error(f"  ❌ {key_name}: Inner encryption uses different key! Cannot fully decrypt.")
                    logger.error(f"     You may need to re-enter this credential manually.")
                    raise ValueError(f"{key_name} has inner encryption with different key")
                except Exception as e:
                    logger.error(f"  ❌ {key_name}: Error decrypting inner value: {e}")
                    raise

            if depth > 1:
                logger.info(f"  ✅ {key_name}: Successfully decrypted {depth} layers")

            return result
        except InvalidToken:
            logger.error(f"Failed to decrypt {key_name} - key mismatch or corrupted data")
            raise
        except Exception as e:
            logger.error(f"Decryption error for {key_name}: {e}")
            raise

    def encrypt_value(self, plaintext: str, key_name: str = "unknown") -> str:
        """
        Encrypt a value with the new key.

        IMPORTANT: This method checks for already-encrypted values to prevent
        double-encryption issues.
        """
        if not plaintext or plaintext == 'PLACEHOLDER':
            return plaintext

        # CRITICAL FIX: Prevent double-encryption
        if plaintext.startswith('gAAAAAB'):
            logger.error(f"  ❌ {key_name}: Refusing to encrypt already-encrypted value!")
            logger.error(f"     This would cause double-encryption. Value starts with: {plaintext[:20]}...")
            raise ValueError(f"Cannot encrypt {key_name}: value is already Fernet-encrypted")

        encrypted = self.new_fernet.encrypt(plaintext.encode())
        return encrypted.decode()

    async def update_credential(self, cred_id: int, new_encrypted_value: str):
        """Update a credential in the database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE secure_credentials
                SET encrypted_value = $1,
                    updated_at = NOW(),
                    last_rotated_at = NOW(),
                    encryption_version = encryption_version + 1
                WHERE id = $2
            """, new_encrypted_value, cred_id)

    def save_new_key(self):
        """Save the new encryption key to file"""
        with open(self.key_file, 'wb') as f:
            f.write(self.new_key)
        os.chmod(self.key_file, 0o600)
        logger.info(f"New encryption key saved to: {self.key_file}")

    async def rotate(self) -> dict:
        """Perform the full key rotation"""
        results = {
            'success': False,
            'credentials_rotated': 0,
            'credentials_failed': 0,
            'backup_file': None,
            'errors': []
        }

        try:
            # Step 1: Connect to database
            logger.info("=" * 60)
            logger.info("ENCRYPTION KEY ROTATION")
            logger.info("=" * 60)

            await self.connect_database()

            # Step 2: Load current key
            if not self.load_current_key():
                results['errors'].append("Failed to load current encryption key")
                return results

            # Step 3: Backup current key
            results['backup_file'] = self.backup_current_key()

            # Step 4: Generate new key
            self.generate_new_key()

            # Step 5: Get all credentials
            credentials = await self.get_all_credentials()
            logger.info(f"Found {len(credentials)} credentials to rotate")

            if self.dry_run:
                logger.info("\n[DRY RUN] Would rotate the following credentials:")

            # Step 6: Re-encrypt each credential
            for cred in credentials:
                cred_id = cred['id']
                key_name = cred['key_name']
                encrypted_value = cred['encrypted_value']
                is_encrypted = cred['is_encrypted']

                try:
                    # Skip non-encrypted values
                    if not is_encrypted and not encrypted_value.startswith('gAAAAAB'):
                        logger.info(f"  Skipping {key_name} (not encrypted)")
                        continue

                    # Decrypt with old key (handles double-encryption)
                    plaintext = self.decrypt_value(encrypted_value, key_name)

                    # Encrypt with new key (prevents double-encryption)
                    new_encrypted = self.encrypt_value(plaintext, key_name)

                    if self.dry_run:
                        logger.info(f"  [DRY RUN] Would rotate: {key_name}")
                    else:
                        # Update database
                        await self.update_credential(cred_id, new_encrypted)
                        logger.info(f"  ✅ Rotated: {key_name}")

                    results['credentials_rotated'] += 1

                except Exception as e:
                    results['credentials_failed'] += 1
                    results['errors'].append(f"{key_name}: {str(e)}")
                    logger.error(f"  ❌ Failed to rotate {key_name}: {e}")

            # Step 7: Save new key
            if not self.dry_run:
                self.save_new_key()

            results['success'] = results['credentials_failed'] == 0

            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("ROTATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Credentials rotated: {results['credentials_rotated']}")
            logger.info(f"Credentials failed: {results['credentials_failed']}")
            logger.info(f"Backup file: {results['backup_file']}")

            if self.dry_run:
                logger.info("\n[DRY RUN] No changes were made")
            else:
                logger.info("\n⚠️  IMPORTANT: Restart all modules to use the new key!")

        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Rotation failed: {e}")

        finally:
            if self.db_pool:
                await self.db_pool.close()

        return results

    async def verify_encryption(self) -> dict:
        """Verify all credentials can be decrypted and detect double-encryption"""
        results = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'double_encrypted': 0,
            'errors': []
        }

        try:
            await self.connect_database()

            if not self.load_current_key():
                results['errors'].append("Failed to load encryption key")
                return results

            credentials = await self.get_all_credentials()
            results['total'] = len(credentials)

            logger.info(f"Verifying {len(credentials)} credentials...")

            for cred in credentials:
                key_name = cred['key_name']
                encrypted_value = cred['encrypted_value']

                try:
                    if encrypted_value == 'PLACEHOLDER':
                        logger.info(f"  ⚪ {key_name}: PLACEHOLDER (not set)")
                        continue

                    if not encrypted_value.startswith('gAAAAAB'):
                        logger.info(f"  ⚪ {key_name}: Not encrypted (plaintext)")
                        results['valid'] += 1
                        continue

                    # Try to decrypt
                    decrypted = self.old_fernet.decrypt(encrypted_value.encode()).decode()

                    # Check for double-encryption
                    if decrypted.startswith('gAAAAAB'):
                        logger.warning(f"  🔴 {key_name}: DOUBLE-ENCRYPTED! Decrypted value is still encrypted.")
                        results['double_encrypted'] += 1
                        results['errors'].append(f"{key_name}: Double-encrypted (needs fix)")

                        # Try to decrypt the inner value
                        try:
                            inner_decrypted = self.old_fernet.decrypt(decrypted.encode()).decode()
                            if inner_decrypted.startswith('gAAAAAB'):
                                logger.warning(f"     Inner value also encrypted - may use different key")
                            else:
                                logger.info(f"     ✅ Can fix with same key (inner decrypts OK)")
                        except Exception:
                            logger.warning(f"     ⚠️  Inner encryption uses DIFFERENT key - needs manual fix")
                    else:
                        logger.info(f"  ✅ {key_name}: Valid")
                        results['valid'] += 1

                except Exception as e:
                    logger.error(f"  ❌ {key_name}: Invalid - {e}")
                    results['invalid'] += 1
                    results['errors'].append(f"{key_name}: {str(e)}")

            logger.info(f"\nVerification complete:")
            logger.info(f"  Valid: {results['valid']}")
            logger.info(f"  Invalid: {results['invalid']}")
            if results['double_encrypted'] > 0:
                logger.warning(f"  Double-encrypted: {results['double_encrypted']} (run fix_double_encrypted.py)")

        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Verification failed: {e}")

        finally:
            if self.db_pool:
                await self.db_pool.close()

        return results


async def main():
    parser = argparse.ArgumentParser(
        description='Rotate encryption key for secure_credentials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rotate_encryption_key.py --dry-run     # Preview changes
  python scripts/rotate_encryption_key.py               # Perform rotation
  python scripts/rotate_encryption_key.py --backup-only # Just backup current key
  python scripts/rotate_encryption_key.py --verify      # Verify encryption

IMPORTANT:
  - Stop all modules before rotating keys
  - Backup your database before rotation
  - Keep the old key backup in case of issues
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without making them')
    parser.add_argument('--backup-only', action='store_true',
                        help='Only backup current key, do not rotate')
    parser.add_argument('--verify', action='store_true',
                        help='Verify all credentials can be decrypted')

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available")
        sys.exit(1)

    rotator = EncryptionKeyRotator(dry_run=args.dry_run)

    if args.backup_only:
        if rotator.load_current_key():
            backup_file = rotator.backup_current_key()
            logger.info(f"Key backed up to: {backup_file}")
        sys.exit(0)

    if args.verify:
        results = await rotator.verify_encryption()
        sys.exit(0 if results['invalid'] == 0 else 1)

    # Confirm before rotating
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: ENCRYPTION KEY ROTATION")
        print("=" * 60)
        print("\nThis will:")
        print("  1. Backup the current encryption key")
        print("  2. Decrypt ALL credentials in the database")
        print("  3. Generate a NEW encryption key")
        print("  4. Re-encrypt ALL credentials with the new key")
        print("\nPrerequisites:")
        print("  - All trading modules should be STOPPED")
        print("  - Database should be backed up")
        print("\n")

        confirm = input("Type 'ROTATE' to proceed: ")
        if confirm != 'ROTATE':
            print("Rotation cancelled")
            sys.exit(0)

    results = await rotator.rotate()

    if results['success']:
        logger.info("\n✅ Key rotation completed successfully!")
        if not args.dry_run:
            logger.info("\nNEXT STEPS:")
            logger.info("  1. Restart all trading modules")
            logger.info("  2. Verify modules can connect and trade")
            logger.info(f"  3. Keep backup at {results['backup_file']} for 30 days")
    else:
        logger.error("\n❌ Key rotation failed!")
        if results['errors']:
            for error in results['errors']:
                logger.error(f"  - {error}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
