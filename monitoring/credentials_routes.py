"""
Credentials Management Routes for ClaudeDex Dashboard

Provides API endpoints and page routes for managing secure credentials.
This module integrates with the SecureSecretsManager for encrypted credential storage.

Routes:
    GET  /credentials              - Credentials management page
    GET  /api/credentials          - List all credentials (masked values)
    POST /api/credentials          - Add new credential
    GET  /api/credentials/{key}    - Get credential details (not value)
    PUT  /api/credentials/{key}    - Update credential
    DELETE /api/credentials/{key}  - Remove credential value
    GET  /api/credentials/stats    - Get credential statistics
    GET  /api/credentials/categories - Get credential categories
    POST /api/credentials/import-env - Import from .env
    POST /api/credentials/validate   - Validate all credentials
"""

import os
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from aiohttp import web

logger = logging.getLogger(__name__)


class CredentialsRoutes:
    """Handles credential management API routes"""

    # Mapping of env vars to categories and metadata
    CREDENTIAL_MAPPINGS = {
        # Security
        'ENCRYPTION_KEY': {'category': 'security', 'subcategory': 'encryption', 'is_sensitive': True, 'is_required': True},
        'JWT_SECRET': {'category': 'security', 'subcategory': 'auth', 'is_sensitive': True},
        'SESSION_SECRET': {'category': 'security', 'subcategory': 'auth', 'is_sensitive': True},

        # Wallet
        'PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'evm', 'is_sensitive': True, 'is_required': True},
        'WALLET_ADDRESS': {'category': 'wallet', 'subcategory': 'evm', 'is_sensitive': False, 'is_required': True},
        'SOLANA_PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': True},
        'SOLANA_WALLET': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': False},
        'SOLANA_MODULE_PRIVATE_KEY': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': True},
        'SOLANA_MODULE_WALLET': {'category': 'wallet', 'subcategory': 'solana', 'is_sensitive': False},
        'FLASHBOTS_SIGNING_KEY': {'category': 'wallet', 'subcategory': 'flashbots', 'is_sensitive': True},

        # Exchange
        'BINANCE_API_KEY': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True},
        'BINANCE_API_SECRET': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True},
        'BINANCE_TESTNET_API_KEY': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True},
        'BINANCE_TESTNET_API_SECRET': {'category': 'exchange', 'subcategory': 'binance', 'is_sensitive': True},
        'BYBIT_API_KEY': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True},
        'BYBIT_API_SECRET': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True},
        'BYBIT_TESTNET_API_KEY': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True},
        'BYBIT_TESTNET_API_SECRET': {'category': 'exchange', 'subcategory': 'bybit', 'is_sensitive': True},

        # Database
        'DB_PASSWORD': {'category': 'database', 'subcategory': 'postgres', 'is_sensitive': True, 'is_required': True},
        'REDIS_PASSWORD': {'category': 'database', 'subcategory': 'redis', 'is_sensitive': True},

        # API
        'GOPLUS_API_KEY': {'category': 'api', 'subcategory': 'goplus', 'is_sensitive': True},
        '1INCH_API_KEY': {'category': 'api', 'subcategory': '1inch', 'is_sensitive': True},
        'HELIUS_API_KEY': {'category': 'api', 'subcategory': 'helius', 'is_sensitive': True},
        'JUPITER_API_KEY': {'category': 'api', 'subcategory': 'jupiter', 'is_sensitive': True},
        'ETHERSCAN_API_KEY': {'category': 'api', 'subcategory': 'etherscan', 'is_sensitive': True},

        # Notification
        'TELEGRAM_BOT_TOKEN': {'category': 'notification', 'subcategory': 'telegram', 'is_sensitive': True},
        'TELEGRAM_CHAT_ID': {'category': 'notification', 'subcategory': 'telegram', 'is_sensitive': False},
        'DISCORD_WEBHOOK_URL': {'category': 'notification', 'subcategory': 'discord', 'is_sensitive': True},
        'TWITTER_API_KEY': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True},
        'TWITTER_API_SECRET': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True},
        'TWITTER_BEARER_TOKEN': {'category': 'notification', 'subcategory': 'twitter', 'is_sensitive': True},
        'EMAIL_PASSWORD': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': True},
        'EMAIL_FROM': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': False},
        'EMAIL_TO': {'category': 'notification', 'subcategory': 'email', 'is_sensitive': False},
    }

    def __init__(self, app: web.Application, db_pool, jinja_env, secrets_manager=None):
        """
        Initialize credentials routes.

        Args:
            app: aiohttp web application
            db_pool: Database connection pool
            jinja_env: Jinja2 template environment
            secrets_manager: Optional SecureSecretsManager instance
        """
        self.app = app
        self.db_pool = db_pool
        self.jinja_env = jinja_env
        self.secrets_manager = secrets_manager

        # Try to import secrets manager if not provided
        if not self.secrets_manager:
            try:
                from security.secrets_manager import secrets
                self.secrets_manager = secrets
            except ImportError:
                logger.warning("SecureSecretsManager not available")

    def setup_routes(self):
        """Register all credential management routes"""
        logger.info("Setting up Credentials Management routes...")

        # Page route
        self.app.router.add_get('/credentials', self.credentials_page)

        # API routes
        self.app.router.add_get('/api/credentials', self.api_list_credentials)
        self.app.router.add_post('/api/credentials', self.api_add_credential)
        self.app.router.add_get('/api/credentials/stats', self.api_get_stats)
        self.app.router.add_get('/api/credentials/categories', self.api_get_categories)
        self.app.router.add_post('/api/credentials/import-env', self.api_import_from_env)
        self.app.router.add_post('/api/credentials/validate', self.api_validate_credentials)
        self.app.router.add_get('/api/credentials/{key}', self.api_get_credential)
        self.app.router.add_put('/api/credentials/{key}', self.api_update_credential)
        self.app.router.add_delete('/api/credentials/{key}', self.api_delete_credential)

        logger.info("Credentials Management routes configured")

    async def credentials_page(self, request: web.Request) -> web.Response:
        """Render credentials management page"""
        try:
            template = self.jinja_env.get_template('settings_credentials.html')
            html = template.render(
                page_title='Secure Credentials',
                page='credentials'
            )
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            logger.error(f"Error rendering credentials page: {e}")
            return web.Response(text=f"Error: {e}", status=500)

    async def api_list_credentials(self, request: web.Request) -> web.Response:
        """List all credentials with masked values"""
        try:
            logger.info(f"Listing credentials - db_pool: {self.db_pool is not None}, secrets_manager: {self.secrets_manager is not None}")

            # Always try database directly first for reliability
            if self.db_pool:
                return await self._list_credentials_from_db()

            # Fallback to secrets_manager if available
            if self.secrets_manager:
                if not getattr(self.secrets_manager, '_initialized', False) and self.db_pool:
                    self.secrets_manager.initialize(self.db_pool)

                credentials = await self.secrets_manager.get_all_credentials()
                if credentials:
                    return web.json_response(credentials)

            logger.warning("No database pool or secrets manager available for credentials")
            return web.json_response([])

        except Exception as e:
            logger.error(f"Error listing credentials: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def _list_credentials_from_db(self) -> web.Response:
        """List credentials directly from database"""
        if not self.db_pool:
            logger.warning("_list_credentials_from_db called but db_pool is None")
            return web.json_response([])

        try:
            logger.info("Fetching credentials from database...")
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, key_name, display_name, description, category,
                           subcategory, module, is_sensitive, is_required,
                           is_active, last_accessed_at, access_count,
                           created_at, updated_at, last_rotated_at,
                           encrypted_value
                    FROM secure_credentials
                    WHERE is_active = TRUE
                    ORDER BY category, key_name
                """)

                logger.info(f"Found {len(rows)} credentials in database")

                result = []
                for row in rows:
                    cred = dict(row)
                    cred['has_value'] = cred['encrypted_value'] != 'PLACEHOLDER'
                    del cred['encrypted_value']

                    for key in ['last_accessed_at', 'created_at', 'updated_at', 'last_rotated_at']:
                        if cred.get(key):
                            cred[key] = cred[key].isoformat()

                    result.append(cred)

                configured_count = sum(1 for c in result if c.get('has_value'))
                logger.info(f"Returning {len(result)} credentials ({configured_count} configured)")
                return web.json_response(result)

        except Exception as e:
            logger.error(f"Database error listing credentials: {e}")
            return web.json_response([], status=200)

    async def api_get_credential(self, request: web.Request) -> web.Response:
        """Get credential details (not the actual value)"""
        key = request.match_info['key']

        if not self.db_pool:
            return web.json_response({'error': 'Database not available'}, status=503)

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, key_name, display_name, description, category,
                           subcategory, module, is_sensitive, is_required,
                           is_active, access_count, created_at, updated_at
                    FROM secure_credentials
                    WHERE key_name = $1
                """, key)

                if not row:
                    return web.json_response({'error': 'Credential not found'}, status=404)

                cred = dict(row)
                for key in ['created_at', 'updated_at']:
                    if cred.get(key):
                        cred[key] = cred[key].isoformat()

                return web.json_response(cred)

        except Exception as e:
            logger.error(f"Error getting credential {key}: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_add_credential(self, request: web.Request) -> web.Response:
        """Add a new credential"""
        try:
            data = await request.json()
            key_name = data.get('key_name', '').upper()
            value = data.get('value', '')

            if not key_name or not value:
                return web.json_response({'error': 'key_name and value required'}, status=400)

            if self.secrets_manager and self.db_pool:
                if not self.secrets_manager._initialized:
                    self.secrets_manager.initialize(self.db_pool)

                success = await self.secrets_manager.set(
                    key_name,
                    value,
                    category=data.get('category', 'general'),
                    is_sensitive=data.get('is_sensitive', True)
                )

                if success:
                    return web.json_response({'success': True, 'key': key_name})
                else:
                    return web.json_response({'error': 'Failed to store credential'}, status=500)
            else:
                return await self._add_credential_to_db(data)

        except Exception as e:
            logger.error(f"Error adding credential: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def _add_credential_to_db(self, data: dict) -> web.Response:
        """Add credential directly to database"""
        if not self.db_pool:
            return web.json_response({'error': 'Database not available'}, status=503)

        try:
            # Get encryption key for encrypting
            encryption_key = os.getenv('ENCRYPTION_KEY')
            encrypted_value = data['value']

            if encryption_key and data.get('is_sensitive', True):
                try:
                    from cryptography.fernet import Fernet
                    fernet = Fernet(encryption_key.encode())
                    encrypted_value = fernet.encrypt(data['value'].encode()).decode()
                except Exception as e:
                    logger.warning(f"Could not encrypt value: {e}")

            value_hash = hashlib.sha256(data['value'].encode()).hexdigest()

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO secure_credentials
                    (key_name, display_name, encrypted_value, value_hash, category,
                     is_sensitive, is_encrypted)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (key_name) DO UPDATE SET
                        encrypted_value = $3,
                        value_hash = $4,
                        updated_at = NOW()
                """,
                    data['key_name'].upper(),
                    data.get('display_name', data['key_name']),
                    encrypted_value,
                    value_hash,
                    data.get('category', 'general'),
                    data.get('is_sensitive', True),
                    data.get('is_sensitive', True)
                )

            return web.json_response({'success': True, 'key': data['key_name']})

        except Exception as e:
            logger.error(f"Database error adding credential: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_update_credential(self, request: web.Request) -> web.Response:
        """Update an existing credential"""
        key = request.match_info['key']

        try:
            data = await request.json()

            if not self.db_pool:
                return web.json_response({'error': 'Database not available'}, status=503)

            # Build update query
            updates = []
            params = [key]
            param_idx = 2

            if 'display_name' in data:
                updates.append(f"display_name = ${param_idx}")
                params.append(data['display_name'])
                param_idx += 1

            if 'description' in data:
                updates.append(f"description = ${param_idx}")
                params.append(data['description'])
                param_idx += 1

            if 'category' in data:
                updates.append(f"category = ${param_idx}")
                params.append(data['category'])
                param_idx += 1

            if 'module' in data:
                updates.append(f"module = ${param_idx}")
                params.append(data['module'])
                param_idx += 1

            if 'is_required' in data:
                updates.append(f"is_required = ${param_idx}")
                params.append(data['is_required'])
                param_idx += 1

            # Handle value update separately (needs encryption)
            if 'value' in data and data['value']:
                encryption_key = os.getenv('ENCRYPTION_KEY')
                encrypted_value = data['value']

                if encryption_key:
                    try:
                        from cryptography.fernet import Fernet
                        fernet = Fernet(encryption_key.encode())
                        encrypted_value = fernet.encrypt(data['value'].encode()).decode()
                    except Exception as e:
                        logger.warning(f"Could not encrypt value: {e}")

                updates.append(f"encrypted_value = ${param_idx}")
                params.append(encrypted_value)
                param_idx += 1

                value_hash = hashlib.sha256(data['value'].encode()).hexdigest()
                updates.append(f"value_hash = ${param_idx}")
                params.append(value_hash)
                param_idx += 1

            if not updates:
                return web.json_response({'error': 'No updates provided'}, status=400)

            async with self.db_pool.acquire() as conn:
                await conn.execute(f"""
                    UPDATE secure_credentials
                    SET {', '.join(updates)}, updated_at = NOW()
                    WHERE key_name = $1
                """, *params)

            # Clear cache if using secrets manager
            if self.secrets_manager:
                self.secrets_manager.clear_cache()

            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error updating credential {key}: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_delete_credential(self, request: web.Request) -> web.Response:
        """Remove credential value (soft delete - sets to PLACEHOLDER)"""
        key = request.match_info['key']

        if not self.db_pool:
            return web.json_response({'error': 'Database not available'}, status=503)

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE secure_credentials
                    SET encrypted_value = 'PLACEHOLDER',
                        value_hash = NULL,
                        updated_at = NOW()
                    WHERE key_name = $1
                """, key)

            # Clear cache
            if self.secrets_manager:
                self.secrets_manager.clear_cache()

            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error deleting credential {key}: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_get_stats(self, request: web.Request) -> web.Response:
        """Get credential statistics"""
        logger.info(f"Getting stats - db_pool: {self.db_pool is not None}")

        if not self.db_pool:
            return web.json_response({
                'total': 0,
                'configured': 0,
                'required': 0,
                'required_configured': 0,
                'is_bootstrap_mode': True,
                'has_encryption': bool(os.getenv('ENCRYPTION_KEY'))
            })

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE encrypted_value != 'PLACEHOLDER') as configured,
                        COUNT(*) FILTER (WHERE is_required = TRUE) as required,
                        COUNT(*) FILTER (WHERE is_required = TRUE AND encrypted_value != 'PLACEHOLDER') as required_configured
                    FROM secure_credentials
                    WHERE is_active = TRUE
                """)

                return web.json_response({
                    'total': row['total'],
                    'configured': row['configured'],
                    'required': row['required'],
                    'required_configured': row['required_configured'],
                    'is_bootstrap_mode': self.secrets_manager is None or getattr(self.secrets_manager, '_bootstrap_mode', True),
                    'has_encryption': bool(os.getenv('ENCRYPTION_KEY'))
                })

        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)

    async def api_get_categories(self, request: web.Request) -> web.Response:
        """Get credential categories"""
        if not self.db_pool:
            # Return default categories
            return web.json_response([
                {'category': 'security', 'display_name': 'Security', 'icon': 'fa-shield-alt', 'color': '#dc3545'},
                {'category': 'wallet', 'display_name': 'Wallet', 'icon': 'fa-wallet', 'color': '#6f42c1'},
                {'category': 'exchange', 'display_name': 'Exchange', 'icon': 'fa-exchange-alt', 'color': '#fd7e14'},
                {'category': 'database', 'display_name': 'Database', 'icon': 'fa-database', 'color': '#20c997'},
                {'category': 'api', 'display_name': 'API', 'icon': 'fa-plug', 'color': '#0d6efd'},
                {'category': 'notification', 'display_name': 'Notification', 'icon': 'fa-bell', 'color': '#ffc107'},
            ])

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT category, display_name, description, icon, color
                    FROM credential_categories
                    ORDER BY sort_order
                """)
                return web.json_response([dict(row) for row in rows])

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return web.json_response([], status=200)

    async def api_import_from_env(self, request: web.Request) -> web.Response:
        """Import credentials from .env file to database"""
        if not self.db_pool:
            return web.json_response({'error': 'Database not available'}, status=503)

        encryption_key = os.getenv('ENCRYPTION_KEY')
        fernet = None
        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                fernet = Fernet(encryption_key.encode())
            except Exception as e:
                logger.warning(f"Encryption not available: {e}")

        imported = 0
        errors = []

        try:
            async with self.db_pool.acquire() as conn:
                for key_name, metadata in self.CREDENTIAL_MAPPINGS.items():
                    value = os.getenv(key_name)

                    if not value or value.startswith('your_'):
                        continue

                    try:
                        # Encrypt if sensitive
                        encrypted_value = value
                        if metadata.get('is_sensitive', True) and fernet:
                            encrypted_value = fernet.encrypt(value.encode()).decode()

                        value_hash = hashlib.sha256(value.encode()).hexdigest()

                        await conn.execute("""
                            UPDATE secure_credentials
                            SET encrypted_value = $2,
                                value_hash = $3,
                                is_encrypted = $4,
                                updated_at = NOW()
                            WHERE key_name = $1
                        """,
                            key_name,
                            encrypted_value,
                            value_hash,
                            bool(fernet and metadata.get('is_sensitive', True))
                        )

                        imported += 1
                        logger.info(f"Imported credential: {key_name}")

                    except Exception as e:
                        errors.append(f"{key_name}: {str(e)}")
                        logger.error(f"Failed to import {key_name}: {e}")

            # Clear cache
            if self.secrets_manager:
                self.secrets_manager.clear_cache()

            return web.json_response({
                'success': True,
                'imported': imported,
                'errors': errors
            })

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def api_validate_credentials(self, request: web.Request) -> web.Response:
        """Validate all credentials"""
        if not self.db_pool:
            return web.json_response({'error': 'Database not available'}, status=503)

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT key_name, encrypted_value, is_required
                    FROM secure_credentials
                    WHERE is_active = TRUE
                """)

                invalid = []
                for row in rows:
                    if row['is_required'] and row['encrypted_value'] == 'PLACEHOLDER':
                        invalid.append(row['key_name'])

                return web.json_response({
                    'all_valid': len(invalid) == 0,
                    'invalid_count': len(invalid),
                    'missing_required': invalid
                })

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return web.json_response({'error': str(e)}, status=500)


def setup_credentials_routes(app: web.Application, db_pool, jinja_env, secrets_manager=None):
    """Setup credentials routes on the application"""
    routes = CredentialsRoutes(app, db_pool, jinja_env, secrets_manager)
    routes.setup_routes()
    return routes
