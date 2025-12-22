"""
RPC/API Pool Engine for Claudedex Trading Bot

Centralized management of RPC and API endpoints with:
- Smart endpoint selection based on health and priority
- Rate limit tracking and automatic rotation
- Periodic health checks
- .env fallback for initial startup
- Load balancing with weighted distribution

Usage:
    from config.pool_engine import PoolEngine

    pool = PoolEngine.get_instance()
    await pool.initialize(db_pool)

    # Get an endpoint
    rpc_url = await pool.get_endpoint('ETHEREUM_RPC')

    # Report rate limit
    await pool.report_rate_limit('ETHEREUM_RPC', rpc_url, duration_seconds=60)

    # Report success/failure
    await pool.report_success('ETHEREUM_RPC', rpc_url, latency_ms=150)
    await pool.report_failure('ETHEREUM_RPC', rpc_url, error='timeout')
"""

import os
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiohttp

# =========================================================================
# Dedicated Pool Engine Logging
# =========================================================================
log_dir = Path("logs/pool_engine")
log_dir.mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Pool Engine Logger
logger = logging.getLogger("PoolEngine")
logger.setLevel(logging.INFO)

# Main Log - all activity
main_handler = RotatingFileHandler(log_dir / 'pool_engine.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# Error Log - errors only
error_handler = RotatingFileHandler(log_dir / 'pool_engine_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# Full Activity Log - comprehensive logging of all operations
full_handler = RotatingFileHandler(log_dir / 'pool_engine_full.log', maxBytes=20*1024*1024, backupCount=5)
full_handler.setFormatter(log_formatter)
full_handler.setLevel(logging.DEBUG)
full_logger = logging.getLogger("PoolEngine.Full")
full_logger.setLevel(logging.DEBUG)
full_logger.addHandler(full_handler)

# Rate Limit Log - rate limit events specifically
rate_limit_handler = RotatingFileHandler(log_dir / 'pool_engine_rate_limits.log', maxBytes=5*1024*1024, backupCount=3)
rate_limit_handler.setFormatter(log_formatter)
rate_limit_logger = logging.getLogger("PoolEngine.RateLimit")
rate_limit_logger.setLevel(logging.INFO)
rate_limit_logger.addHandler(rate_limit_handler)

# Health Check Log - health check results
health_handler = RotatingFileHandler(log_dir / 'pool_engine_health.log', maxBytes=5*1024*1024, backupCount=3)
health_handler.setFormatter(log_formatter)
health_logger = logging.getLogger("PoolEngine.Health")
health_logger.setLevel(logging.INFO)
health_logger.addHandler(health_handler)

# Console output (shared across all pool engine loggers)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


class EndpointStatus(Enum):
    """Endpoint status enumeration"""
    ACTIVE = 'active'
    RATE_LIMITED = 'rate_limited'
    UNHEALTHY = 'unhealthy'
    DISABLED = 'disabled'


@dataclass
class Endpoint:
    """Represents a single RPC/API endpoint"""
    id: int
    provider_type: str
    name: str
    url: str
    api_key: Optional[str] = None
    status: EndpointStatus = EndpointStatus.ACTIVE
    is_enabled: bool = True
    priority: int = 100
    weight: int = 100
    rate_limit_until: Optional[datetime] = None
    rate_limit_count: int = 0
    last_rate_limit_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    last_health_check_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0
    health_score: float = 100
    consecutive_failures: int = 0
    chain: Optional[str] = None
    supports_ws: bool = False
    ws_url: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """Check if endpoint is available for use"""
        if not self.is_enabled:
            return False
        if self.status == EndpointStatus.DISABLED:
            return False
        if self.status == EndpointStatus.RATE_LIMITED:
            if self.rate_limit_until and datetime.now(self.rate_limit_until.tzinfo) < self.rate_limit_until:
                return False
        if self.status == EndpointStatus.UNHEALTHY and self.consecutive_failures >= 10:
            return False
        return True

    def get_effective_url(self) -> str:
        """Get URL with API key if applicable"""
        if self.api_key:
            # Handle different API key formats
            if '?' in self.url:
                return f"{self.url}&api-key={self.api_key}"
            else:
                return f"{self.url}?api-key={self.api_key}" if 'api-key' not in self.url else self.url
        return self.url


@dataclass
class ProviderEndpoints:
    """Container for all endpoints of a specific provider type"""
    provider_type: str
    endpoints: List[Endpoint] = field(default_factory=list)
    last_selected_index: int = 0

    def get_available_endpoints(self) -> List[Endpoint]:
        """Get all available endpoints sorted by priority and health"""
        available = [e for e in self.endpoints if e.is_available]
        # Sort by priority (lower is better), then by health_score (higher is better)
        return sorted(available, key=lambda e: (e.priority, -e.health_score))

    def get_next_endpoint(self) -> Optional[Endpoint]:
        """Get next available endpoint using weighted round-robin"""
        available = self.get_available_endpoints()
        if not available:
            return None

        # Use weighted random selection among top priority endpoints
        top_priority = available[0].priority
        top_tier = [e for e in available if e.priority == top_priority]

        if len(top_tier) == 1:
            return top_tier[0]

        # Weighted selection
        total_weight = sum(e.weight for e in top_tier)
        if total_weight == 0:
            return random.choice(top_tier)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for endpoint in top_tier:
            cumulative += endpoint.weight
            if r <= cumulative:
                return endpoint

        return top_tier[-1]


class PoolEngine:
    """
    Centralized RPC/API Pool Engine

    Manages all RPC and API endpoints with health tracking,
    rate limit handling, and smart selection.
    """

    _instance: Optional['PoolEngine'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.db_pool = None
        self.providers: Dict[str, ProviderEndpoints] = {}
        self.initialized = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 3600  # 1 hour
        self._env_fallback_used = False
        self._last_db_sync = None

        # Cache for quick lookups
        self._endpoint_cache: Dict[str, Endpoint] = {}  # url -> endpoint

        # Rate limit backoff settings
        self._default_rate_limit_duration = 300  # 5 minutes
        self._max_consecutive_failures = 10

        logger.info("PoolEngine initialized (not yet connected)")

    @classmethod
    async def get_instance(cls) -> 'PoolEngine':
        """Get singleton instance of PoolEngine"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> 'PoolEngine':
        """Get singleton instance synchronously (for startup)"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self, db_pool=None) -> bool:
        """
        Initialize the pool engine

        Args:
            db_pool: Database connection pool (optional, will use .env fallback if None)

        Returns:
            bool: True if initialization successful
        """
        try:
            self.db_pool = db_pool

            if db_pool:
                # Try to load from database
                success = await self._load_from_database()
                if success and self._has_endpoints():
                    logger.info(f"Loaded {self._count_endpoints()} endpoints from database")
                    self.initialized = True
                    self._start_health_check_task()
                    return True

            # Fall back to .env
            logger.info("Database empty or unavailable, loading from .env")
            await self._load_from_env()
            self._env_fallback_used = True
            self.initialized = True

            # Seed database if pool is available
            if db_pool and self._has_endpoints():
                await self._seed_database()

            self._start_health_check_task()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PoolEngine: {e}", exc_info=True)
            # Try .env fallback as last resort
            try:
                await self._load_from_env()
                self._env_fallback_used = True
                self.initialized = True
                return True
            except Exception as fallback_error:
                logger.error(f"Even .env fallback failed: {fallback_error}")
                return False

    async def _load_from_database(self) -> bool:
        """Load endpoints from database"""
        if not self.db_pool:
            return False

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, endpoint_type, provider_type, name, url, api_key,
                           status, is_enabled, priority, weight,
                           rate_limit_until, last_success_at, last_failure_at,
                           success_count, failure_count, avg_latency_ms,
                           health_score, consecutive_failures, chain,
                           supports_ws, ws_url
                    FROM rpc_api_pool
                    WHERE is_enabled = TRUE
                    ORDER BY provider_type, priority
                """)

                if not rows:
                    return False

                for row in rows:
                    endpoint = Endpoint(
                        id=row['id'],
                        provider_type=row['provider_type'],
                        name=row['name'],
                        url=row['url'],
                        api_key=row['api_key'],
                        status=EndpointStatus(row['status']),
                        is_enabled=row['is_enabled'],
                        priority=row['priority'],
                        weight=row['weight'],
                        rate_limit_until=row['rate_limit_until'],
                        last_success_at=row['last_success_at'],
                        last_failure_at=row['last_failure_at'],
                        success_count=row['success_count'],
                        failure_count=row['failure_count'],
                        avg_latency_ms=row['avg_latency_ms'] or 0,
                        health_score=row['health_score'] or 100,
                        consecutive_failures=row['consecutive_failures'],
                        chain=row['chain'],
                        supports_ws=row['supports_ws'],
                        ws_url=row['ws_url']
                    )

                    self._add_endpoint(endpoint)
                    self._endpoint_cache[row['url']] = endpoint

                self._last_db_sync = datetime.utcnow()
                return True

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            return False

    async def _load_from_env(self) -> None:
        """Load endpoints from environment variables as fallback"""
        logger.info("Loading RPC/API endpoints from .env")

        # Mapping of env vars to provider types
        env_mappings = {
            # EVM RPC URLs (comma-separated)
            'ETHEREUM_RPC_URLS': ('ETHEREUM_RPC', 'ethereum'),
            'BSC_RPC_URLS': ('BSC_RPC', 'bsc'),
            'POLYGON_RPC_URLS': ('POLYGON_RPC', 'polygon'),
            'ARBITRUM_RPC_URLS': ('ARBITRUM_RPC', 'arbitrum'),
            'BASE_RPC_URLS': ('BASE_RPC', 'base'),
            'MONAD_RPC_URLS': ('MONAD_RPC', 'monad'),
            'PULSECHAIN_RPC_URLS': ('PULSECHAIN_RPC', 'pulsechain'),
            'FANTOM_RPC_URLS': ('FANTOM_RPC', 'fantom'),
            'CRONOS_RPC_URLS': ('CRONOS_RPC', 'cronos'),
            'AVALANCHE_RPC_URLS': ('AVALANCHE_RPC', 'avalanche'),
            # Solana RPC URLs
            'SOLANA_RPC_URLS': ('SOLANA_RPC', 'solana'),
            'SOLANA_BACKUP_RPCS': ('SOLANA_RPC', 'solana'),
        }

        # Single RPC URL mappings
        single_rpc_mappings = {
            'ETHEREUM_RPC_URL': ('ETHEREUM_RPC', 'ethereum'),
            'WEB3_PROVIDER_URL': ('ETHEREUM_RPC', 'ethereum'),
            'WEB3_BACKUP_PROVIDER_1': ('ETHEREUM_RPC', 'ethereum'),
            'WEB3_BACKUP_PROVIDER_2': ('ETHEREUM_RPC', 'ethereum'),
            'BSC_RPC_URL': ('BSC_RPC', 'bsc'),
            'POLYGON_RPC_URL': ('POLYGON_RPC', 'polygon'),
            'ARBITRUM_RPC_URL': ('ARBITRUM_RPC', 'arbitrum'),
            'BASE_RPC_URL': ('BASE_RPC', 'base'),
            'MONAD_RPC_URL': ('MONAD_RPC', 'monad'),
            'PULSECHAIN_RPC_URL': ('PULSECHAIN_RPC', 'pulsechain'),
            'FANTOM_RPC_URL': ('FANTOM_RPC', 'fantom'),
            'CRONOS_RPC_URL': ('CRONOS_RPC', 'cronos'),
            'AVALANCHE_RPC_URL': ('AVALANCHE_RPC', 'avalanche'),
            'SOLANA_RPC_URL': ('SOLANA_RPC', 'solana'),
        }

        # WebSocket mappings
        ws_mappings = {
            'SOLANA_WS_URL': ('SOLANA_WS', 'solana'),
        }

        # API mappings
        api_mappings = {
            'GOPLUS_API_KEY': ('GOPLUS_API', None, 'https://api.gopluslabs.io'),
            '1INCH_API_KEY': ('1INCH_API', None, 'https://api.1inch.io'),
            'HELIUS_API_KEY': ('HELIUS_API', 'solana', None),  # URL built from key
            'ETHERSCAN_API_KEY': ('ETHERSCAN_API', 'ethereum', 'https://api.etherscan.io'),
            'JUPITER_API_KEY': ('JUPITER_API', 'solana', None),  # Uses JUPITER_API_URL
        }

        endpoint_id = 0

        # Load comma-separated RPC URLs
        for env_var, (provider_type, chain) in env_mappings.items():
            value = os.getenv(env_var, '')
            if value:
                urls = [url.strip().strip('"').strip("'") for url in value.split(',') if url.strip()]
                for i, url in enumerate(urls):
                    if url and url not in ('null', 'None', ''):
                        endpoint_id += 1
                        endpoint = Endpoint(
                            id=endpoint_id,
                            provider_type=provider_type,
                            name=f"{provider_type} #{i+1}",
                            url=url,
                            chain=chain,
                            priority=100 + i  # Later URLs have lower priority
                        )
                        self._add_endpoint(endpoint)
                        self._endpoint_cache[url] = endpoint

        # Load single RPC URLs
        for env_var, (provider_type, chain) in single_rpc_mappings.items():
            value = os.getenv(env_var, '')
            if value and value not in ('null', 'None', ''):
                # Check if already added
                if value not in self._endpoint_cache:
                    endpoint_id += 1
                    priority = 50 if 'PROVIDER' in env_var or 'BACKUP' not in env_var else 150
                    endpoint = Endpoint(
                        id=endpoint_id,
                        provider_type=provider_type,
                        name=env_var,
                        url=value,
                        chain=chain,
                        priority=priority
                    )
                    self._add_endpoint(endpoint)
                    self._endpoint_cache[value] = endpoint

        # Load WebSocket URLs
        for env_var, (provider_type, chain) in ws_mappings.items():
            value = os.getenv(env_var, '')
            if value and value not in ('null', 'None', ''):
                endpoint_id += 1
                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=env_var,
                    url=value,
                    chain=chain,
                    supports_ws=True
                )
                self._add_endpoint(endpoint)
                self._endpoint_cache[value] = endpoint

        # Load APIs
        for env_var, (provider_type, chain, base_url) in api_mappings.items():
            api_key = os.getenv(env_var, '')
            if api_key and api_key not in ('null', 'None', '', 'your_'):
                endpoint_id += 1

                # Determine URL
                if provider_type == 'HELIUS_API':
                    url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
                elif provider_type == 'JUPITER_API':
                    url = os.getenv('JUPITER_API_URL', 'https://lite-api.jup.ag')
                else:
                    url = base_url or ''

                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=env_var,
                    url=url,
                    api_key=api_key,
                    chain=chain
                )
                self._add_endpoint(endpoint)
                if url:
                    self._endpoint_cache[url] = endpoint

        logger.info(f"Loaded {self._count_endpoints()} endpoints from .env")

    async def _seed_database(self) -> None:
        """Seed database with endpoints loaded from .env"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                for provider_type, provider_endpoints in self.providers.items():
                    for endpoint in provider_endpoints.endpoints:
                        await conn.execute("""
                            INSERT INTO rpc_api_pool (
                                endpoint_type, provider_type, name, url, api_key,
                                status, is_enabled, priority, weight, chain,
                                supports_ws, ws_url
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            ON CONFLICT (provider_type, url) DO NOTHING
                        """,
                            'rpc' if 'RPC' in provider_type else ('ws' if 'WS' in provider_type else 'api'),
                            endpoint.provider_type,
                            endpoint.name,
                            endpoint.url,
                            endpoint.api_key,
                            endpoint.status.value,
                            endpoint.is_enabled,
                            endpoint.priority,
                            endpoint.weight,
                            endpoint.chain,
                            endpoint.supports_ws,
                            endpoint.ws_url
                        )

            logger.info("Seeded database with endpoints from .env")

        except Exception as e:
            logger.error(f"Failed to seed database: {e}")

    def _add_endpoint(self, endpoint: Endpoint) -> None:
        """Add endpoint to internal storage"""
        if endpoint.provider_type not in self.providers:
            self.providers[endpoint.provider_type] = ProviderEndpoints(
                provider_type=endpoint.provider_type
            )
        self.providers[endpoint.provider_type].endpoints.append(endpoint)

    def _has_endpoints(self) -> bool:
        """Check if any endpoints are loaded"""
        return any(len(p.endpoints) > 0 for p in self.providers.values())

    def _count_endpoints(self) -> int:
        """Count total endpoints"""
        return sum(len(p.endpoints) for p in self.providers.values())

    # =========================================================================
    # Public API - Endpoint Selection
    # =========================================================================

    async def get_endpoint(self, provider_type: str) -> Optional[str]:
        """
        Get the best available endpoint URL for a provider type

        Args:
            provider_type: Type of provider (e.g., 'ETHEREUM_RPC', 'SOLANA_RPC')

        Returns:
            str: Best available endpoint URL, or None if none available
        """
        if not self.initialized:
            logger.warning("PoolEngine not initialized, using .env fallback")
            return self._get_env_fallback(provider_type)

        provider = self.providers.get(provider_type)
        if not provider:
            logger.warning(f"No endpoints for provider type: {provider_type}")
            return self._get_env_fallback(provider_type)

        endpoint = provider.get_next_endpoint()
        if not endpoint:
            logger.warning(f"No available endpoints for: {provider_type}")
            return self._get_env_fallback(provider_type)

        # Log selection to full logger
        full_logger.debug(
            f"SELECTED: {provider_type} -> {endpoint.name} "
            f"(priority={endpoint.priority}, health={endpoint.health_score:.1f})"
        )

        return endpoint.get_effective_url()

    async def get_all_endpoints(self, provider_type: str) -> List[str]:
        """
        Get all available endpoint URLs for a provider type

        Args:
            provider_type: Type of provider

        Returns:
            List[str]: All available endpoint URLs
        """
        if not self.initialized:
            return []

        provider = self.providers.get(provider_type)
        if not provider:
            return []

        available = provider.get_available_endpoints()
        return [e.get_effective_url() for e in available]

    async def get_endpoint_with_fallbacks(self, provider_type: str, max_fallbacks: int = 3) -> List[str]:
        """
        Get endpoint with fallback options

        Args:
            provider_type: Type of provider
            max_fallbacks: Maximum number of fallback endpoints

        Returns:
            List[str]: List of endpoint URLs (primary + fallbacks)
        """
        endpoints = await self.get_all_endpoints(provider_type)
        return endpoints[:max_fallbacks] if endpoints else []

    def _get_env_fallback(self, provider_type: str) -> Optional[str]:
        """Get fallback from environment variable"""
        env_var_mappings = {
            'ETHEREUM_RPC': 'ETHEREUM_RPC_URL',
            'BSC_RPC': 'BSC_RPC_URL',
            'POLYGON_RPC': 'POLYGON_RPC_URL',
            'ARBITRUM_RPC': 'ARBITRUM_RPC_URL',
            'BASE_RPC': 'BASE_RPC_URL',
            'SOLANA_RPC': 'SOLANA_RPC_URL',
            'MONAD_RPC': 'MONAD_RPC_URL',
            'PULSECHAIN_RPC': 'PULSECHAIN_RPC_URL',
            'FANTOM_RPC': 'FANTOM_RPC_URL',
            'CRONOS_RPC': 'CRONOS_RPC_URL',
            'AVALANCHE_RPC': 'AVALANCHE_RPC_URL',
            'SOLANA_WS': 'SOLANA_WS_URL',
            'GOPLUS_API': 'GOPLUS_API_KEY',
            '1INCH_API': '1INCH_API_KEY',
            'HELIUS_API': 'HELIUS_API_KEY',
            'ETHERSCAN_API': 'ETHERSCAN_API_KEY',
            'JUPITER_API': 'JUPITER_API_URL',
        }

        env_var = env_var_mappings.get(provider_type)
        if env_var:
            value = os.getenv(env_var)
            if value and value not in ('null', 'None', ''):
                return value

        return None

    # =========================================================================
    # Public API - Status Reporting
    # =========================================================================

    async def report_rate_limit(
        self,
        provider_type: str,
        url: str,
        duration_seconds: int = None,
        error_message: str = None
    ) -> None:
        """
        Report that an endpoint has been rate limited

        Args:
            provider_type: Type of provider
            url: The endpoint URL that was rate limited
            duration_seconds: How long until rate limit resets (default: 5 min)
            error_message: Optional error message for logging
        """
        endpoint = self._find_endpoint(provider_type, url)
        if not endpoint:
            return

        duration = duration_seconds or self._default_rate_limit_duration
        endpoint.status = EndpointStatus.RATE_LIMITED
        endpoint.rate_limit_until = datetime.utcnow() + timedelta(seconds=duration)
        endpoint.rate_limit_count += 1
        endpoint.last_rate_limit_at = datetime.utcnow()
        endpoint.health_score = max(0, endpoint.health_score - 10)

        # Move to end of queue by increasing priority
        endpoint.priority = min(endpoint.priority + 50, 1000)

        rate_limit_msg = (
            f"Rate limited: {provider_type} - {endpoint.name} "
            f"(until {endpoint.rate_limit_until}, count={endpoint.rate_limit_count}, priority: {endpoint.priority})"
        )
        logger.warning(rate_limit_msg)
        rate_limit_logger.info(rate_limit_msg)
        full_logger.warning(rate_limit_msg)

        # Update database
        await self._update_endpoint_status(endpoint)

        # Log usage
        await self._log_usage(endpoint, False, error_type='rate_limit', error_message=error_message)

    async def report_success(
        self,
        provider_type: str,
        url: str,
        latency_ms: int = None
    ) -> None:
        """
        Report successful request to an endpoint

        Args:
            provider_type: Type of provider
            url: The endpoint URL
            latency_ms: Request latency in milliseconds
        """
        endpoint = self._find_endpoint(provider_type, url)
        if not endpoint:
            return

        endpoint.success_count += 1
        endpoint.last_success_at = datetime.utcnow()
        endpoint.consecutive_failures = 0

        # Reset status if it was rate limited and limit has passed
        if endpoint.status == EndpointStatus.RATE_LIMITED:
            if not endpoint.rate_limit_until or datetime.utcnow() >= endpoint.rate_limit_until:
                endpoint.status = EndpointStatus.ACTIVE
                endpoint.rate_limit_until = None
                full_logger.info(f"Endpoint recovered from rate limit: {endpoint.name}")

        # Reset status if it was unhealthy
        if endpoint.status == EndpointStatus.UNHEALTHY:
            endpoint.status = EndpointStatus.ACTIVE
            full_logger.info(f"Endpoint recovered from unhealthy: {endpoint.name}")

        # Update latency (rolling average)
        if latency_ms:
            if endpoint.avg_latency_ms == 0:
                endpoint.avg_latency_ms = latency_ms
            else:
                endpoint.avg_latency_ms = (endpoint.avg_latency_ms * 0.9) + (latency_ms * 0.1)

        # Update health score
        success_rate = endpoint.success_count / max(1, endpoint.success_count + endpoint.failure_count)
        latency_score = max(0, 100 - (endpoint.avg_latency_ms / 10))  # Lower latency = higher score
        endpoint.health_score = min(100, (success_rate * 70) + (latency_score * 0.3))

        # PRIORITY RECOVERY: If priority was penalized, gradually recover it
        # Base priority is typically 100, so if it's higher, we can improve it
        base_priority = 100
        if endpoint.priority > base_priority:
            # Recover 5 priority points on each success (min base_priority)
            old_priority = endpoint.priority
            endpoint.priority = max(base_priority, endpoint.priority - 5)
            if endpoint.priority != old_priority:
                full_logger.debug(
                    f"Priority improved for {endpoint.name}: {old_priority} -> {endpoint.priority}"
                )

        # Log to full activity log
        full_logger.debug(
            f"SUCCESS: {provider_type} - {endpoint.name} | "
            f"latency={latency_ms}ms | priority={endpoint.priority} | "
            f"health={endpoint.health_score:.1f}"
        )

        # Log usage (don't await to avoid blocking)
        asyncio.create_task(self._log_usage(endpoint, True, latency_ms=latency_ms))

    async def report_failure(
        self,
        provider_type: str,
        url: str,
        error_type: str = None,
        error_message: str = None
    ) -> None:
        """
        Report failed request to an endpoint

        Args:
            provider_type: Type of provider
            url: The endpoint URL
            error_type: Type of error (e.g., 'timeout', 'network_error')
            error_message: Error message for logging
        """
        endpoint = self._find_endpoint(provider_type, url)
        if not endpoint:
            return

        endpoint.failure_count += 1
        endpoint.last_failure_at = datetime.utcnow()
        endpoint.consecutive_failures += 1

        # Decrease health score
        endpoint.health_score = max(0, endpoint.health_score - 5)

        # PRIORITY PENALTY: Increase priority (lower = better, higher = worse)
        # Penalize by 10 points per failure (smaller than rate limit penalty of 50)
        old_priority = endpoint.priority
        endpoint.priority = min(endpoint.priority + 10, 500)

        # Log to full activity log
        full_logger.info(
            f"FAILURE: {provider_type} - {endpoint.name} | "
            f"error={error_type or 'unknown'} | priority: {old_priority} -> {endpoint.priority} | "
            f"consecutive_failures={endpoint.consecutive_failures}"
        )

        # Mark as unhealthy if too many consecutive failures
        if endpoint.consecutive_failures >= self._max_consecutive_failures:
            endpoint.status = EndpointStatus.UNHEALTHY
            # Additional priority penalty for unhealthy endpoints
            endpoint.priority = min(endpoint.priority + 100, 1000)
            unhealthy_msg = (
                f"Endpoint marked unhealthy: {provider_type} - {endpoint.name} "
                f"(consecutive failures: {endpoint.consecutive_failures}, priority: {endpoint.priority})"
            )
            logger.warning(unhealthy_msg)
            health_logger.warning(unhealthy_msg)

        # Persist changes to database
        await self._update_endpoint_status(endpoint)

        # Log usage
        await self._log_usage(endpoint, False, error_type=error_type, error_message=error_message)

    def _find_endpoint(self, provider_type: str, url: str) -> Optional[Endpoint]:
        """Find endpoint by provider type and URL"""
        # Try cache first
        endpoint = self._endpoint_cache.get(url)
        if endpoint and endpoint.provider_type == provider_type:
            return endpoint

        # Search in provider
        provider = self.providers.get(provider_type)
        if provider:
            for ep in provider.endpoints:
                if ep.url == url or ep.get_effective_url() == url:
                    return ep

        return None

    async def _update_endpoint_status(self, endpoint: Endpoint) -> None:
        """Update endpoint status in database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE rpc_api_pool SET
                        status = $2,
                        rate_limit_until = $3,
                        rate_limit_count = $4,
                        last_rate_limit_at = $5,
                        health_score = $6,
                        consecutive_failures = $7,
                        priority = $8,
                        last_success_at = $9,
                        last_failure_at = $10,
                        success_count = $11,
                        failure_count = $12,
                        avg_latency_ms = $13
                    WHERE id = $1
                """,
                    endpoint.id,
                    endpoint.status.value,
                    endpoint.rate_limit_until,
                    endpoint.rate_limit_count,
                    endpoint.last_rate_limit_at,
                    endpoint.health_score,
                    endpoint.consecutive_failures,
                    endpoint.priority,
                    endpoint.last_success_at,
                    endpoint.last_failure_at,
                    endpoint.success_count,
                    endpoint.failure_count,
                    endpoint.avg_latency_ms
                )
        except Exception as e:
            logger.error(f"Failed to update endpoint status in DB: {e}")

    async def _log_usage(
        self,
        endpoint: Endpoint,
        success: bool,
        latency_ms: int = None,
        error_type: str = None,
        error_message: str = None,
        module_name: str = None
    ) -> None:
        """Log endpoint usage to database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rpc_api_usage_history (
                        endpoint_id, module_name, success, latency_ms,
                        error_type, error_message
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    endpoint.id,
                    module_name,
                    success,
                    latency_ms,
                    error_type,
                    error_message
                )
        except Exception as e:
            logger.debug(f"Failed to log usage: {e}")

    # =========================================================================
    # Health Check
    # =========================================================================

    def _start_health_check_task(self) -> None:
        """Start background health check task"""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Started health check background task")

    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checks"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.run_health_checks()
            except asyncio.CancelledError:
                logger.info("Health check task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def run_health_checks(self) -> Dict[str, Any]:
        """
        Run health checks on all endpoints

        Returns:
            Dict with health check results
        """
        logger.info("Running health checks on all endpoints...")
        results = {
            'checked': 0,
            'healthy': 0,
            'unhealthy': 0,
            'rate_limited': 0,
            'recovered': 0
        }

        for provider_type, provider in self.providers.items():
            for endpoint in provider.endpoints:
                results['checked'] += 1

                # Check if rate limit has expired
                if endpoint.status == EndpointStatus.RATE_LIMITED:
                    if endpoint.rate_limit_until and datetime.utcnow() >= endpoint.rate_limit_until:
                        endpoint.status = EndpointStatus.ACTIVE
                        endpoint.rate_limit_until = None
                        results['recovered'] += 1
                        recovery_msg = f"Endpoint recovered from rate limit: {endpoint.name}"
                        logger.info(recovery_msg)
                        health_logger.info(recovery_msg)

                # For RPC endpoints, try a simple health check
                if 'RPC' in provider_type and endpoint.is_enabled:
                    try:
                        is_healthy = await self._check_rpc_health(endpoint)
                        if is_healthy:
                            results['healthy'] += 1
                            if endpoint.status == EndpointStatus.UNHEALTHY:
                                endpoint.status = EndpointStatus.ACTIVE
                                endpoint.consecutive_failures = 0
                                results['recovered'] += 1
                                health_logger.info(f"Endpoint recovered: {endpoint.name}")
                        else:
                            results['unhealthy'] += 1
                            # Mark endpoint as unhealthy if check fails
                            endpoint.consecutive_failures += 1
                            if endpoint.status != EndpointStatus.RATE_LIMITED:
                                endpoint.status = EndpointStatus.UNHEALTHY
                                endpoint.health_score = max(0, endpoint.health_score - 15)
                                health_logger.warning(f"Endpoint unhealthy: {endpoint.name} (consecutive failures: {endpoint.consecutive_failures})")
                    except Exception as e:
                        logger.debug(f"Health check failed for {endpoint.name}: {e}")
                        results['unhealthy'] += 1
                        endpoint.consecutive_failures += 1
                        if endpoint.status != EndpointStatus.RATE_LIMITED:
                            endpoint.status = EndpointStatus.UNHEALTHY
                else:
                    # Count non-RPC endpoints based on their current status
                    if endpoint.status == EndpointStatus.ACTIVE:
                        results['healthy'] += 1
                    elif endpoint.status == EndpointStatus.RATE_LIMITED:
                        results['rate_limited'] += 1
                    elif endpoint.status == EndpointStatus.UNHEALTHY:
                        results['unhealthy'] += 1

                # Update last health check time
                endpoint.last_health_check_at = datetime.utcnow()

        # Sync to database
        await self._sync_to_database()

        health_check_msg = (
            f"Health check complete: {results['checked']} checked, "
            f"{results['healthy']} healthy, {results['unhealthy']} unhealthy, "
            f"{results['recovered']} recovered"
        )
        logger.info(health_check_msg)
        health_logger.info(health_check_msg)

        return results

    async def _check_rpc_health(self, endpoint: Endpoint) -> bool:
        """Check health of an RPC endpoint"""
        try:
            # Different health checks for different chains
            if endpoint.chain == 'solana':
                payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
            else:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}

            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.get_effective_url(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency = int((time.time() - start) * 1000)
                    if response.status == 200:
                        data = await response.json()
                        if 'result' in data or 'error' not in data:
                            await self.report_success(endpoint.provider_type, endpoint.url, latency)
                            return True
                    return False

        except Exception as e:
            logger.debug(f"RPC health check failed for {endpoint.name}: {e}")
            return False

    async def _sync_to_database(self) -> None:
        """Sync current state to database"""
        if not self.db_pool:
            return

        try:
            for provider in self.providers.values():
                for endpoint in provider.endpoints:
                    await self._update_endpoint_status(endpoint)

            self._last_db_sync = datetime.utcnow()
            logger.debug("Synced endpoint status to database")

        except Exception as e:
            logger.error(f"Failed to sync to database: {e}")

    # =========================================================================
    # CRUD Operations for Dashboard
    # =========================================================================

    async def add_endpoint(
        self,
        provider_type: str,
        name: str,
        url: str,
        api_key: str = None,
        chain: str = None,
        priority: int = 100
    ) -> Optional[int]:
        """Add a new endpoint"""
        if not self.db_pool:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO rpc_api_pool (
                        endpoint_type, provider_type, name, url, api_key,
                        chain, priority, status, is_enabled
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', TRUE)
                    RETURNING id
                """,
                    'rpc' if 'RPC' in provider_type else ('ws' if 'WS' in provider_type else 'api'),
                    provider_type,
                    name,
                    url,
                    api_key,
                    chain,
                    priority
                )

                endpoint_id = row['id']

                # Add to in-memory cache
                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=name,
                    url=url,
                    api_key=api_key,
                    chain=chain,
                    priority=priority
                )
                self._add_endpoint(endpoint)
                self._endpoint_cache[url] = endpoint

                logger.info(f"Added endpoint: {name} ({provider_type})")
                return endpoint_id

        except Exception as e:
            logger.error(f"Failed to add endpoint: {e}")
            return None

    async def update_endpoint(
        self,
        endpoint_id: int,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing endpoint"""
        if not self.db_pool:
            return False

        try:
            # Build update query
            set_clauses = []
            values = [endpoint_id]
            param_num = 2

            allowed_fields = ['name', 'url', 'api_key', 'priority', 'weight', 'is_enabled', 'chain']
            for field in allowed_fields:
                if field in updates:
                    set_clauses.append(f"{field} = ${param_num}")
                    values.append(updates[field])
                    param_num += 1

            if not set_clauses:
                return False

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    f"UPDATE rpc_api_pool SET {', '.join(set_clauses)} WHERE id = $1",
                    *values
                )

            # Reload from database to sync
            await self._load_from_database()

            logger.info(f"Updated endpoint ID {endpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update endpoint: {e}")
            return False

    async def delete_endpoint(self, endpoint_id: int) -> bool:
        """Delete an endpoint"""
        if not self.db_pool:
            return False

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("DELETE FROM rpc_api_pool WHERE id = $1", endpoint_id)

            # Reload from database
            await self._load_from_database()

            logger.info(f"Deleted endpoint ID {endpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            return False

    async def get_all_endpoints_data(self) -> List[Dict[str, Any]]:
        """Get all endpoints for dashboard display"""
        if not self.db_pool:
            # Return from memory
            result = []
            for provider in self.providers.values():
                for ep in provider.endpoints:
                    result.append({
                        'id': ep.id,
                        'provider_type': ep.provider_type,
                        'name': ep.name,
                        'url': ep.url,
                        'status': ep.status.value,
                        'is_enabled': ep.is_enabled,
                        'priority': ep.priority,
                        'health_score': ep.health_score,
                        'success_count': ep.success_count,
                        'failure_count': ep.failure_count,
                        'chain': ep.chain
                    })
            return result

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, endpoint_type, provider_type, name, url,
                           status, is_enabled, priority, weight,
                           rate_limit_until, health_score,
                           success_count, failure_count, avg_latency_ms,
                           consecutive_failures, chain, last_success_at,
                           last_failure_at, last_health_check_at
                    FROM rpc_api_pool
                    ORDER BY provider_type, priority
                """)

                # Convert to list of dicts with datetime serialization
                result = []
                for row in rows:
                    d = dict(row)
                    # Serialize datetime objects to ISO strings
                    for key, val in d.items():
                        if isinstance(val, datetime):
                            d[key] = val.isoformat() if val else None
                    result.append(d)
                return result

        except Exception as e:
            logger.error(f"Failed to get endpoints: {e}")
            return []

    async def get_provider_types(self) -> List[Dict[str, Any]]:
        """Get all provider types for dashboard"""
        if not self.db_pool:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT provider_type, endpoint_type, chain, description,
                           default_priority, is_required
                    FROM rpc_api_provider_types
                    ORDER BY provider_type
                """)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get provider types: {e}")
            return []

    async def get_usage_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for dashboard"""
        from decimal import Decimal

        def serialize_value(val):
            """Convert Decimal and other non-JSON types to serializable values"""
            if isinstance(val, Decimal):
                return float(val)
            return val

        def serialize_dict(d):
            """Serialize all values in a dict"""
            return {k: serialize_value(v) for k, v in d.items()}

        if not self.db_pool:
            return {}

        try:
            async with self.db_pool.acquire() as conn:
                # Overall stats
                overall = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed,
                        AVG(latency_ms) as avg_latency,
                        COUNT(DISTINCT endpoint_id) as endpoints_used
                    FROM rpc_api_usage_history
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                """ % hours)

                # Per provider stats
                per_provider = await conn.fetch("""
                    SELECT
                        p.provider_type,
                        COUNT(*) as requests,
                        SUM(CASE WHEN h.success THEN 1 ELSE 0 END) as successful,
                        AVG(h.latency_ms) as avg_latency
                    FROM rpc_api_usage_history h
                    JOIN rpc_api_pool p ON p.id = h.endpoint_id
                    WHERE h.created_at > NOW() - INTERVAL '%s hours'
                    GROUP BY p.provider_type
                    ORDER BY requests DESC
                """ % hours)

                return {
                    'overall': serialize_dict(dict(overall)) if overall else {},
                    'per_provider': [serialize_dict(dict(row)) for row in per_provider],
                    'period_hours': hours
                }

        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def shutdown(self) -> None:
        """Shutdown the pool engine"""
        logger.info("Shutting down PoolEngine...")

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Final sync to database
        await self._sync_to_database()

        self.initialized = False
        logger.info("PoolEngine shutdown complete")


# Convenience function for getting the pool instance
async def get_pool() -> PoolEngine:
    """Get the PoolEngine singleton instance"""
    return await PoolEngine.get_instance()
