"""
Secure Secrets Manager for ClaudeDex Trading Bot

This module provides a centralized, secure way to access sensitive credentials.
It implements a layered approach:

1. Docker Secrets (highest priority in production)
2. External key file (/secure/encryption.key)
3. Database (encrypted credentials)
4. Environment variables (.env fallback)

SECURITY ARCHITECTURE:
- Encryption key is stored SEPARATELY from encrypted data
- All sensitive values are encrypted at rest in database
- Access is logged for audit compliance
- Fallback to .env allows gradual migration

Usage:
    from security.secrets_manager import secrets

    # Get a credential
    api_key = secrets.get('BINANCE_API_KEY')

    # Get with fallback
    api_key = secrets.get('BINANCE_API_KEY', default='')

    # Check if credential exists
    if secrets.has('BINANCE_API_KEY'):
        ...
"""

import os
import sys
import logging
import hashlib
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from functools import lru_cache
import threading

# Conditional imports for when database isn't available yet
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception

logger = logging.getLogger(__name__)


class SecureSecretsManager:
    """
    Centralized secrets management with multiple storage backends.

    Priority order for credential lookup:
    1. Docker secrets (/run/secrets/<key>)
    2. Memory cache (for performance)
    3. Database (encrypted)
    4. Environment variables (fallback)
    """

    _instance: Optional['SecureSecretsManager'] = None
    _lock = threading.Lock()

    # Credential categories
    CATEGORY_SECURITY = 'security'
    CATEGORY_WALLET = 'wallet'
    CATEGORY_EXCHANGE = 'exchange'
    CATEGORY_DATABASE = 'database'
    CATEGORY_API = 'api'
    CATEGORY_NOTIFICATION = 'notification'
    CATEGORY_GENERAL = 'general'

    # Keys that should NEVER be logged
    SENSITIVE_KEYS = {
        'ENCRYPTION_KEY', 'PRIVATE_KEY', 'SOLANA_PRIVATE_KEY',
        'SOLANA_MODULE_PRIVATE_KEY', 'FLASHBOTS_SIGNING_KEY',
        'BINANCE_API_SECRET', 'BINANCE_TESTNET_API_SECRET',
        'BYBIT_API_SECRET', 'BYBIT_TESTNET_API_SECRET',
        'JWT_SECRET', 'SESSION_SECRET', 'EMAIL_PASSWORD',
        'DB_PASSWORD', 'REDIS_PASSWORD', 'TWITTER_API_SECRET',
    }

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._db_pool = None
        self._fernet: Optional[Fernet] = None
        self._initialized = False
        self._initialization_lock = threading.Lock()

        # Paths for external key storage
        self._docker_secrets_path = Path('/run/secrets')
        self._external_key_path = Path(os.getenv('ENCRYPTION_KEY_FILE', '/secure/encryption.key'))

        # Bootstrap mode - use .env only until DB is available
        self._bootstrap_mode = True

        # Track which source provided each credential
        self._source_map: Dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> 'SecureSecretsManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self, db_pool=None) -> bool:
        """
        Initialize the secrets manager.

        Args:
            db_pool: Database connection pool (optional)

        Returns:
            bool: True if initialization successful
        """
        with self._initialization_lock:
            # Clear cache on re-initialization to remove stale encrypted values
            if db_pool and self._initialized:
                logger.info("Re-initializing secrets manager with database - clearing cache")
                self._cache.clear()
                self._source_map.clear()

            if self._initialized and not db_pool:
                return True

            try:
                # Step 1: Initialize encryption
                self._init_encryption()

                # Step 2: Set up database connection if provided
                if db_pool:
                    self._db_pool = db_pool
                    self._bootstrap_mode = False
                    logger.info("Secrets manager initialized with database backend")
                else:
                    self._bootstrap_mode = True
                    logger.info("Secrets manager initialized in bootstrap mode (.env fallback)")

                self._initialized = True
                return True

            except Exception as e:
                logger.error(f"Failed to initialize secrets manager: {e}")
                # Fall back to .env only mode
                self._bootstrap_mode = True
                self._initialized = True
                return True

    def _init_encryption(self) -> None:
        """Initialize encryption with external key"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography package not available - encryption disabled")
            return

        key = None
        key_source = None

        # Priority 1: Docker secret
        docker_key_path = self._docker_secrets_path / 'encryption_key'
        if docker_key_path.exists():
            try:
                key = docker_key_path.read_bytes().strip()
                key_source = 'docker_secret'
                logger.info("Loaded encryption key from Docker secret")
            except Exception as e:
                logger.warning(f"Failed to read Docker secret: {e}")

        # Priority 2: External key file
        if key is None and self._external_key_path.exists():
            try:
                key = self._external_key_path.read_bytes().strip()
                key_source = 'external_file'
                logger.info(f"Loaded encryption key from: {self._external_key_path}")
            except Exception as e:
                logger.warning(f"Failed to read external key file: {e}")

        # Priority 3: Project root .encryption_key file
        if key is None:
            project_key_path = Path('.encryption_key')
            if project_key_path.exists():
                try:
                    key = project_key_path.read_bytes().strip()
                    key_source = 'project_root'
                    logger.info("Loaded encryption key from .encryption_key")
                except Exception as e:
                    logger.warning(f"Failed to read .encryption_key: {e}")

        # Priority 4: Environment variable (fallback for migration)
        if key is None:
            env_key = os.getenv('ENCRYPTION_KEY')
            if env_key:
                key = env_key.encode() if isinstance(env_key, str) else env_key
                key_source = 'environment'
                logger.warning("Using encryption key from .env - migrate to external storage!")

        if key:
            try:
                self._fernet = Fernet(key)
                logger.debug(f"Encryption initialized from {key_source}")
            except Exception as e:
                logger.error(f"Invalid encryption key from {key_source}: {e}")
                self._fernet = None
        else:
            logger.warning("No encryption key found - encrypted credentials unavailable")

    def get(self, key: str, default: str = None, log_access: bool = True) -> Optional[str]:
        """
        Get a credential value.

        Args:
            key: Credential key name
            default: Default value if not found
            log_access: Whether to log this access

        Returns:
            str: The credential value or default
        """
        # Lazy initialization of encryption (in case get() is called before initialize())
        if self._fernet is None and CRYPTO_AVAILABLE:
            self._init_encryption()

        # Check cache first (but not if value is encrypted)
        if key in self._cache:
            cached = self._cache[key]
            # Don't return encrypted values from cache
            if not (cached and cached.startswith('gAAAAAB')):
                return cached

        value = None
        source = None

        # Priority 1: Docker secret
        value = self._get_from_docker_secret(key)
        if value:
            source = 'docker'

        # Priority 2: Database (if available)
        if value is None and self._db_pool and not self._bootstrap_mode:
            value = self._get_from_database_sync(key)
            if value:
                source = 'database'

        # Priority 3: Environment variable (fallback)
        if value is None:
            value = os.getenv(key)
            if value:
                source = 'env'
                # Decrypt if value is Fernet encrypted
                if value.startswith('gAAAAAB') and self._fernet:
                    try:
                        value = self._fernet.decrypt(value.encode()).decode()
                        logger.debug(f"Decrypted {key} from environment variable")
                    except Exception as e:
                        logger.warning(f"Failed to decrypt {key} from env: {e}")
                        # Don't cache failed decryption - might succeed later
                        value = None
                elif value.startswith('gAAAAAB'):
                    # Encrypted but no fernet available - don't use
                    logger.warning(f"Cannot decrypt {key} - encryption key not loaded")
                    value = None

        # Use default if not found
        if value is None:
            value = default
            source = 'default'

        # Cache the result (only if not encrypted)
        if value is not None and not (value and value.startswith('gAAAAAB')):
            self._cache[key] = value
            self._source_map[key] = source

        # Log access (but not sensitive values)
        if log_access and value is not None:
            if key in self.SENSITIVE_KEYS:
                logger.debug(f"Credential accessed: {key} (source: {source})")
            else:
                logger.debug(f"Credential accessed: {key}={self._mask_value(value)} (source: {source})")

        return value

    def _get_from_docker_secret(self, key: str) -> Optional[str]:
        """Get credential from Docker secret"""
        # Docker secret names are lowercase with underscores replaced
        secret_name = key.lower()
        secret_path = self._docker_secrets_path / secret_name

        if secret_path.exists():
            try:
                return secret_path.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read Docker secret {key}: {e}")

        return None

    def _get_from_database_sync(self, key: str) -> Optional[str]:
        """Get credential from database (synchronous wrapper)"""
        # The db_pool is bound to the event loop that created it.
        # We can only safely use it from that same loop.
        try:
            # Check if there's a running event loop in the current thread
            try:
                asyncio.get_running_loop()
                # There IS a running loop - the pool is attached to it.
                # We CANNOT use asyncio.run() or ThreadPoolExecutor because:
                # 1. asyncio.run() would try to create a new loop (fails)
                # 2. ThreadPoolExecutor creates a new loop in another thread,
                #    but the pool connections are bound to THIS loop
                #
                # The caller should use get_async() instead.
                # Skip database and rely on Docker/env fallback.
                logger.debug(f"Skipping database lookup for {key} - use get_async() in async context")
                return None
            except RuntimeError:
                # No event loop running, safe to use asyncio.run directly
                if self._db_pool:
                    return asyncio.run(self._get_from_database(key))
                return None
        except Exception as e:
            logger.warning(f"Failed to get credential from database: {e}")
            return None

    async def _get_from_database(self, key: str) -> Optional[str]:
        """Get credential from database"""
        if not self._db_pool:
            return None

        try:
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT encrypted_value, is_encrypted, is_sensitive
                    FROM secure_credentials
                    WHERE key_name = $1 AND is_active = TRUE
                """, key)

                if not row:
                    return None

                encrypted_value = row['encrypted_value']

                # Skip placeholder values
                if encrypted_value == 'PLACEHOLDER':
                    return None

                # Decrypt if needed
                if row['is_encrypted'] and self._fernet:
                    try:
                        decrypted = self._fernet.decrypt(encrypted_value.encode())
                        return decrypted.decode()
                    except InvalidToken:
                        logger.error(f"Failed to decrypt credential {key} - key mismatch?")
                        return None
                else:
                    return encrypted_value

        except Exception as e:
            logger.error(f"Database error getting credential {key}: {e}")
            return None

    async def get_async(self, key: str, default: str = None) -> Optional[str]:
        """Async version of get()"""
        # Lazy initialization of encryption (in case get_async() is called before initialize())
        if self._fernet is None and CRYPTO_AVAILABLE:
            self._init_encryption()

        # Check cache (but not if value is encrypted)
        if key in self._cache:
            cached = self._cache[key]
            if not (cached and cached.startswith('gAAAAAB')):
                return cached

        value = None

        # Docker secret
        value = self._get_from_docker_secret(key)

        # Database
        if value is None and self._db_pool:
            value = await self._get_from_database(key)

        # Environment (with decryption)
        if value is None:
            env_value = os.getenv(key)
            if env_value:
                # Decrypt if value is Fernet encrypted
                if env_value.startswith('gAAAAAB') and self._fernet:
                    try:
                        value = self._fernet.decrypt(env_value.encode()).decode()
                        logger.debug(f"Decrypted {key} from environment variable (async)")
                    except Exception as e:
                        logger.warning(f"Failed to decrypt {key} from env: {e}")
                        value = default
                elif env_value.startswith('gAAAAAB'):
                    # Encrypted but no fernet available
                    logger.warning(f"Cannot decrypt {key} - encryption key not loaded (async)")
                    value = default
                else:
                    value = env_value
            else:
                value = default

        # Cache (only if not encrypted)
        if value is not None and not (value and value.startswith('gAAAAAB')):
            self._cache[key] = value

        return value

    async def set(self, key: str, value: str, category: str = 'general',
                  is_sensitive: bool = True) -> bool:
        """
        Store a credential in the database.

        Args:
            key: Credential key name
            value: The value to store
            category: Category for organization
            is_sensitive: Whether to encrypt and mask in logs

        Returns:
            bool: True if successful
        """
        if not self._db_pool:
            logger.error("Cannot store credential - database not available")
            return False

        if not self._fernet and is_sensitive:
            logger.error("Cannot store sensitive credential - encryption not available")
            return False

        try:
            # Encrypt if sensitive
            if is_sensitive and self._fernet:
                encrypted_value = self._fernet.encrypt(value.encode()).decode()
            else:
                encrypted_value = value

            # Create hash for integrity checking
            value_hash = hashlib.sha256(value.encode()).hexdigest()

            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO secure_credentials
                    (key_name, encrypted_value, value_hash, category, is_sensitive, is_encrypted)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (key_name) DO UPDATE SET
                        encrypted_value = $2,
                        value_hash = $3,
                        category = $4,
                        is_sensitive = $5,
                        is_encrypted = $6,
                        updated_at = NOW()
                """, key, encrypted_value, value_hash, category, is_sensitive, is_sensitive)

                # Log the access
                await conn.execute("""
                    INSERT INTO credential_access_log
                    (credential_id, key_name, access_type, accessed_by, success)
                    SELECT id, $1, 'write', 'secrets_manager', TRUE
                    FROM secure_credentials WHERE key_name = $1
                """, key)

            # Update cache
            self._cache[key] = value
            self._source_map[key] = 'database'

            logger.info(f"Credential stored: {key} (category: {category})")
            return True

        except Exception as e:
            logger.error(f"Failed to store credential {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Soft delete a credential"""
        if not self._db_pool:
            return False

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE secure_credentials
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE key_name = $1
                """, key)

            # Remove from cache
            self._cache.pop(key, None)
            self._source_map.pop(key, None)

            return True
        except Exception as e:
            logger.error(f"Failed to delete credential {key}: {e}")
            return False

    def has(self, key: str) -> bool:
        """Check if a credential exists and has a value"""
        return self.get(key, log_access=False) is not None

    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a cached credential"""
        return self._source_map.get(key)

    def clear_cache(self) -> None:
        """Clear the credential cache"""
        self._cache.clear()
        self._source_map.clear()

    def _mask_value(self, value: str, visible_chars: int = 4) -> str:
        """Mask a sensitive value for logging"""
        if not value or len(value) <= visible_chars * 2:
            return '****'
        return value[:visible_chars] + '****' + value[-visible_chars:]

    async def get_all_credentials(self, category: str = None) -> List[Dict[str, Any]]:
        """
        Get all credentials for dashboard display.

        Returns masked values - NEVER returns actual secrets!
        """
        if not self._db_pool:
            return []

        try:
            async with self._db_pool.acquire() as conn:
                if category:
                    rows = await conn.fetch("""
                        SELECT id, key_name, display_name, description, category,
                               subcategory, module, is_sensitive, is_required,
                               is_active, last_accessed_at, access_count,
                               created_at, updated_at, last_rotated_at,
                               encrypted_value
                        FROM secure_credentials
                        WHERE category = $1
                        ORDER BY category, key_name
                    """, category)
                else:
                    rows = await conn.fetch("""
                        SELECT id, key_name, display_name, description, category,
                               subcategory, module, is_sensitive, is_required,
                               is_active, last_accessed_at, access_count,
                               created_at, updated_at, last_rotated_at,
                               encrypted_value
                        FROM secure_credentials
                        ORDER BY category, key_name
                    """)

                result = []
                for row in rows:
                    cred = dict(row)
                    # Check if value is set (not placeholder)
                    cred['has_value'] = cred['encrypted_value'] != 'PLACEHOLDER'
                    # Never return actual encrypted value
                    del cred['encrypted_value']

                    # Serialize datetimes
                    for key in ['last_accessed_at', 'created_at', 'updated_at', 'last_rotated_at']:
                        if cred.get(key):
                            cred[key] = cred[key].isoformat()

                    result.append(cred)

                return result

        except Exception as e:
            logger.error(f"Failed to get credentials: {e}")
            return []

    async def get_categories(self) -> List[Dict[str, Any]]:
        """Get all credential categories"""
        if not self._db_pool:
            return []

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT category, display_name, description, icon, color, sort_order
                    FROM credential_categories
                    ORDER BY sort_order
                """)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get credential statistics for dashboard"""
        if not self._db_pool:
            return {}

        try:
            async with self._db_pool.acquire() as conn:
                # Count by category
                by_category = await conn.fetch("""
                    SELECT category,
                           COUNT(*) as total,
                           COUNT(*) FILTER (WHERE encrypted_value != 'PLACEHOLDER') as configured
                    FROM secure_credentials
                    WHERE is_active = TRUE
                    GROUP BY category
                """)

                # Overall stats
                overall = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE encrypted_value != 'PLACEHOLDER') as configured,
                        COUNT(*) FILTER (WHERE is_required = TRUE) as required,
                        COUNT(*) FILTER (WHERE is_required = TRUE AND encrypted_value != 'PLACEHOLDER') as required_configured
                    FROM secure_credentials
                    WHERE is_active = TRUE
                """)

                return {
                    'by_category': [dict(row) for row in by_category],
                    'total': overall['total'],
                    'configured': overall['configured'],
                    'required': overall['required'],
                    'required_configured': overall['required_configured'],
                    'is_bootstrap_mode': self._bootstrap_mode,
                    'has_encryption': self._fernet is not None
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Singleton instance for easy import
secrets = SecureSecretsManager.get_instance()


# Convenience functions
def get_secret(key: str, default: str = None) -> Optional[str]:
    """Get a secret value"""
    return secrets.get(key, default)


def has_secret(key: str) -> bool:
    """Check if a secret exists"""
    return secrets.has(key)


async def set_secret(key: str, value: str, category: str = 'general',
                     is_sensitive: bool = True) -> bool:
    """Store a secret"""
    return await secrets.set(key, value, category, is_sensitive)
