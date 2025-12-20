"""
RPC/API Provider Utility

Provides easy access to RPC and API endpoints for all modules.
Wraps the Pool Engine with convenient methods and handles
synchronous/async access patterns.

Usage:
    from config.rpc_provider import RPCProvider

    # Async usage (recommended)
    rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')

    # Sync usage (for initialization)
    rpc_url = RPCProvider.get_rpc_sync('ETHEREUM_RPC')

    # With rate limit reporting
    rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')
    try:
        result = await make_request(rpc_url)
        await RPCProvider.report_success('ETHEREUM_RPC', rpc_url, latency_ms=100)
    except RateLimitError:
        await RPCProvider.report_rate_limit('ETHEREUM_RPC', rpc_url)
        rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')  # Get next available
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class RPCProvider:
    """
    Centralized RPC/API provider for all modules

    Provides static methods for easy access to RPC and API endpoints
    with automatic fallback to environment variables.
    """

    _pool_engine = None
    _initialized = False

    # =========================================================================
    # Async Methods (Preferred)
    # =========================================================================

    @classmethod
    async def get_rpc(cls, provider_type: str) -> Optional[str]:
        """
        Get an RPC URL asynchronously

        Args:
            provider_type: Type of provider (e.g., 'ETHEREUM_RPC', 'SOLANA_RPC')

        Returns:
            str: RPC URL or None if not available

        Examples:
            eth_rpc = await RPCProvider.get_rpc('ETHEREUM_RPC')
            sol_rpc = await RPCProvider.get_rpc('SOLANA_RPC')
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            url = await pool.get_endpoint(provider_type)
            if url:
                return url

        # Fallback to .env
        return cls._get_env_fallback(provider_type)

    @classmethod
    async def get_rpcs(cls, provider_type: str, max_count: int = 3) -> List[str]:
        """
        Get multiple RPC URLs for fallback

        Args:
            provider_type: Type of provider
            max_count: Maximum number of URLs to return

        Returns:
            List[str]: List of RPC URLs
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            urls = await pool.get_endpoint_with_fallbacks(provider_type, max_count)
            if urls:
                return urls

        # Fallback to .env
        fallback = cls._get_env_fallback(provider_type)
        if fallback:
            # Also try comma-separated env var
            multi_fallback = cls._get_env_multi_fallback(provider_type)
            if multi_fallback:
                return multi_fallback[:max_count]
            return [fallback]
        return []

    @classmethod
    async def get_api(cls, provider_type: str) -> Optional[str]:
        """
        Get an API key or URL asynchronously

        Args:
            provider_type: Type of provider (e.g., 'HELIUS_API', 'JUPITER_API')

        Returns:
            str: API key or URL
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            url = await pool.get_endpoint(provider_type)
            if url:
                return url

        # Fallback to .env
        return cls._get_env_api_fallback(provider_type)

    @classmethod
    async def get_ws(cls, provider_type: str) -> Optional[str]:
        """
        Get a WebSocket URL asynchronously

        Args:
            provider_type: Type of provider (e.g., 'SOLANA_WS', 'ETHEREUM_WS')

        Returns:
            str: WebSocket URL
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            url = await pool.get_endpoint(provider_type)
            if url:
                return url

        # Fallback to .env
        return cls._get_env_fallback(provider_type)

    # =========================================================================
    # Synchronous Methods (For initialization/compatibility)
    # =========================================================================

    @classmethod
    def get_rpc_sync(cls, provider_type: str) -> Optional[str]:
        """
        Get an RPC URL synchronously (uses .env fallback if pool not initialized)

        This method is for use during synchronous initialization.
        For async code, prefer get_rpc() instead.

        Args:
            provider_type: Type of provider

        Returns:
            str: RPC URL or None
        """
        # Try to get from pool if already initialized
        if cls._pool_engine and cls._pool_engine.initialized:
            # Can't call async from sync, but can check cache
            provider = cls._pool_engine.providers.get(provider_type)
            if provider:
                endpoint = provider.get_next_endpoint()
                if endpoint:
                    return endpoint.get_effective_url()

        # Fallback to .env
        return cls._get_env_fallback(provider_type)

    @classmethod
    def get_rpcs_sync(cls, provider_type: str, max_count: int = 3) -> List[str]:
        """
        Get multiple RPC URLs synchronously

        Args:
            provider_type: Type of provider
            max_count: Maximum number of URLs

        Returns:
            List[str]: List of RPC URLs
        """
        # Try pool first
        if cls._pool_engine and cls._pool_engine.initialized:
            provider = cls._pool_engine.providers.get(provider_type)
            if provider:
                available = provider.get_available_endpoints()
                return [e.get_effective_url() for e in available[:max_count]]

        # Fallback to .env
        multi = cls._get_env_multi_fallback(provider_type)
        if multi:
            return multi[:max_count]

        fallback = cls._get_env_fallback(provider_type)
        return [fallback] if fallback else []

    @classmethod
    def get_api_sync(cls, provider_type: str) -> Optional[str]:
        """
        Get an API key/URL synchronously

        Args:
            provider_type: Type of provider

        Returns:
            str: API key or URL
        """
        return cls._get_env_api_fallback(provider_type)

    # =========================================================================
    # Reporting Methods
    # =========================================================================

    @classmethod
    async def report_rate_limit(
        cls,
        provider_type: str,
        url: str,
        duration_seconds: int = 300
    ) -> None:
        """
        Report that an endpoint was rate limited

        Args:
            provider_type: Type of provider
            url: The URL that was rate limited
            duration_seconds: Duration until rate limit resets
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            await pool.report_rate_limit(provider_type, url, duration_seconds)
            logger.warning(f"Rate limit reported for {provider_type}: {url[:50]}...")

    @classmethod
    async def report_success(
        cls,
        provider_type: str,
        url: str,
        latency_ms: int = None
    ) -> None:
        """
        Report successful request to an endpoint

        Args:
            provider_type: Type of provider
            url: The URL used
            latency_ms: Request latency in milliseconds
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            await pool.report_success(provider_type, url, latency_ms)

    @classmethod
    async def report_failure(
        cls,
        provider_type: str,
        url: str,
        error_type: str = None,
        error_message: str = None
    ) -> None:
        """
        Report failed request to an endpoint

        Args:
            provider_type: Type of provider
            url: The URL used
            error_type: Type of error
            error_message: Error message
        """
        pool = await cls._get_pool()
        if pool and pool.initialized:
            await pool.report_failure(provider_type, url, error_type, error_message)

    # =========================================================================
    # Chain-Specific Convenience Methods
    # =========================================================================

    @classmethod
    async def get_ethereum_rpc(cls) -> Optional[str]:
        """Get Ethereum RPC URL"""
        return await cls.get_rpc('ETHEREUM_RPC')

    @classmethod
    async def get_solana_rpc(cls) -> Optional[str]:
        """Get Solana RPC URL"""
        return await cls.get_rpc('SOLANA_RPC')

    @classmethod
    async def get_bsc_rpc(cls) -> Optional[str]:
        """Get BSC RPC URL"""
        return await cls.get_rpc('BSC_RPC')

    @classmethod
    async def get_arbitrum_rpc(cls) -> Optional[str]:
        """Get Arbitrum RPC URL"""
        return await cls.get_rpc('ARBITRUM_RPC')

    @classmethod
    async def get_base_rpc(cls) -> Optional[str]:
        """Get Base RPC URL"""
        return await cls.get_rpc('BASE_RPC')

    @classmethod
    async def get_polygon_rpc(cls) -> Optional[str]:
        """Get Polygon RPC URL"""
        return await cls.get_rpc('POLYGON_RPC')

    @classmethod
    async def get_solana_ws(cls) -> Optional[str]:
        """Get Solana WebSocket URL"""
        return await cls.get_ws('SOLANA_WS')

    @classmethod
    async def get_helius_api(cls) -> Optional[str]:
        """Get Helius API key"""
        return await cls.get_api('HELIUS_API')

    @classmethod
    async def get_jupiter_api(cls) -> Optional[str]:
        """Get Jupiter API URL"""
        return await cls.get_api('JUPITER_API')

    # =========================================================================
    # Internal Methods
    # =========================================================================

    @classmethod
    async def _get_pool(cls):
        """Get Pool Engine instance"""
        if cls._pool_engine is None:
            try:
                from config.pool_engine import PoolEngine
                cls._pool_engine = await PoolEngine.get_instance()
            except Exception as e:
                logger.debug(f"Pool Engine not available: {e}")
        return cls._pool_engine

    @classmethod
    def set_pool_engine(cls, pool_engine):
        """Set the Pool Engine instance (for initialization)"""
        cls._pool_engine = pool_engine
        cls._initialized = True

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_env_fallback(provider_type: str) -> Optional[str]:
        """Get RPC URL fallback from environment variables"""
        env_mappings = {
            'ETHEREUM_RPC': ['ETHEREUM_RPC_URL', 'WEB3_PROVIDER_URL'],
            'BSC_RPC': ['BSC_RPC_URL'],
            'POLYGON_RPC': ['POLYGON_RPC_URL'],
            'ARBITRUM_RPC': ['ARBITRUM_RPC_URL'],
            'BASE_RPC': ['BASE_RPC_URL'],
            'SOLANA_RPC': ['SOLANA_RPC_URL'],
            'MONAD_RPC': ['MONAD_RPC_URL'],
            'PULSECHAIN_RPC': ['PULSECHAIN_RPC_URL'],
            'FANTOM_RPC': ['FANTOM_RPC_URL'],
            'CRONOS_RPC': ['CRONOS_RPC_URL'],
            'AVALANCHE_RPC': ['AVALANCHE_RPC_URL'],
            'SOLANA_WS': ['SOLANA_WS_URL'],
            'ETHEREUM_WS': ['ETHEREUM_WS_URL'],
            'BSC_WS': ['BSC_WS_URL'],
            'ARBITRUM_WS': ['ARBITRUM_WS_URL'],
        }

        env_vars = env_mappings.get(provider_type, [])
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value and value not in ('null', 'None', ''):
                return value.strip().strip('"').strip("'")
        return None

    @staticmethod
    def _get_env_multi_fallback(provider_type: str) -> List[str]:
        """Get multiple RPC URLs from comma-separated env var"""
        env_mappings = {
            'ETHEREUM_RPC': 'ETHEREUM_RPC_URLS',
            'BSC_RPC': 'BSC_RPC_URLS',
            'POLYGON_RPC': 'POLYGON_RPC_URLS',
            'ARBITRUM_RPC': 'ARBITRUM_RPC_URLS',
            'BASE_RPC': 'BASE_RPC_URLS',
            'SOLANA_RPC': 'SOLANA_RPC_URLS',
            'MONAD_RPC': 'MONAD_RPC_URLS',
            'PULSECHAIN_RPC': 'PULSECHAIN_RPC_URLS',
            'FANTOM_RPC': 'FANTOM_RPC_URLS',
            'CRONOS_RPC': 'CRONOS_RPC_URLS',
            'AVALANCHE_RPC': 'AVALANCHE_RPC_URLS',
        }

        env_var = env_mappings.get(provider_type)
        if env_var:
            value = os.getenv(env_var, '')
            if value:
                urls = [url.strip().strip('"').strip("'") for url in value.split(',')]
                return [url for url in urls if url and url not in ('null', 'None', '')]
        return []

    @staticmethod
    @lru_cache(maxsize=16)
    def _get_env_api_fallback(provider_type: str) -> Optional[str]:
        """Get API key/URL fallback from environment variables"""
        env_mappings = {
            'GOPLUS_API': 'GOPLUS_API_KEY',
            '1INCH_API': '1INCH_API_KEY',
            'HELIUS_API': 'HELIUS_API_KEY',
            'ETHERSCAN_API': 'ETHERSCAN_API_KEY',
            'JUPITER_API': 'JUPITER_API_URL',
        }

        env_var = env_mappings.get(provider_type)
        if env_var:
            value = os.getenv(env_var)
            if value and value not in ('null', 'None', ''):
                return value.strip()
        return None


# Convenience aliases
get_rpc = RPCProvider.get_rpc
get_rpcs = RPCProvider.get_rpcs
get_api = RPCProvider.get_api
get_rpc_sync = RPCProvider.get_rpc_sync
report_rate_limit = RPCProvider.report_rate_limit
report_success = RPCProvider.report_success
report_failure = RPCProvider.report_failure
