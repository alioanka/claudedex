"""
Docker Secrets Reader

Reads secrets from Docker secrets (/run/secrets/) or falls back to environment variables.
This allows the same code to work both in Docker (with secrets) and locally (with .env).

Usage:
    from security.docker_secrets import get_secret

    db_password = get_secret('db_password')  # Reads from /run/secrets/db_password
    api_key = get_secret('api_key', 'default_value')  # With fallback
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SECRETS_DIR = Path('/run/secrets')


def get_secret(name: str, default: Optional[str] = None, env_var: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value from Docker secrets or environment.

    Priority:
    1. Docker secret file (/run/secrets/{name})
    2. Environment variable (env_var or name.upper())
    3. Default value

    Args:
        name: Secret name (e.g., 'db_password')
        default: Default value if not found
        env_var: Environment variable name (defaults to name.upper())

    Returns:
        Secret value or default
    """
    # 1. Try Docker secret file
    secret_file = SECRETS_DIR / name
    if secret_file.exists():
        try:
            value = secret_file.read_text().strip()
            if value:
                logger.debug(f"Secret '{name}' loaded from Docker secrets")
                return value
        except Exception as e:
            logger.warning(f"Failed to read Docker secret '{name}': {e}")

    # 2. Try environment variable
    env_name = env_var or name.upper()
    value = os.getenv(env_name)
    if value:
        logger.debug(f"Secret '{name}' loaded from environment ({env_name})")
        return value

    # Also try with common prefixes
    for prefix in ['', 'DB_', 'REDIS_']:
        value = os.getenv(f"{prefix}{env_name}")
        if value:
            logger.debug(f"Secret '{name}' loaded from environment ({prefix}{env_name})")
            return value

    # 3. Return default
    if default is not None:
        logger.debug(f"Secret '{name}' using default value")
        return default

    logger.warning(f"Secret '{name}' not found in Docker secrets or environment")
    return None


def get_db_credentials() -> dict:
    """
    Get database credentials from Docker secrets or environment.

    Returns:
        dict with 'user', 'password', 'host', 'port', 'name', 'url'
    """
    user = get_secret('db_user', env_var='DB_USER') or 'bot_user'
    password = get_secret('db_password', env_var='DB_PASSWORD') or ''
    host = os.getenv('DB_HOST', 'postgres')
    port = os.getenv('DB_PORT', '5432')
    name = os.getenv('DB_NAME', 'tradingbot')

    url = f"postgresql://{user}:{password}@{host}:{port}/{name}"

    return {
        'user': user,
        'password': password,
        'host': host,
        'port': port,
        'name': name,
        'url': url
    }


def get_redis_credentials() -> dict:
    """
    Get Redis credentials from Docker secrets or environment.

    Returns:
        dict with 'password', 'host', 'port', 'url'
    """
    password = get_secret('redis_password', env_var='REDIS_PASSWORD') or ''
    host = os.getenv('REDIS_HOST', 'redis')
    port = os.getenv('REDIS_PORT', '6379')
    db = os.getenv('REDIS_DB', '0')

    if password:
        url = f"redis://:{password}@{host}:{port}/{db}"
    else:
        url = f"redis://{host}:{port}/{db}"

    return {
        'password': password,
        'host': host,
        'port': port,
        'db': db,
        'url': url
    }


def get_database_url() -> str:
    """Get the full database URL."""
    return get_db_credentials()['url']


def get_redis_url() -> str:
    """Get the full Redis URL."""
    return get_redis_credentials()['url']
