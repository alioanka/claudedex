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
from datetime import datetime, timedelta, timezone
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
# BUG 1: confine to its own file. These are children of "PoolEngine" and were
# propagating UP to the parent's console + main-file handlers, so every line
# was emitted three times (parent console, parent main.log, own file). Pinning
# propagate=False stops the storm while keeping each dedicated audit file.
full_logger.propagate = False

# Rate Limit Log - rate limit events specifically
rate_limit_handler = RotatingFileHandler(log_dir / 'pool_engine_rate_limits.log', maxBytes=5*1024*1024, backupCount=3)
rate_limit_handler.setFormatter(log_formatter)
rate_limit_logger = logging.getLogger("PoolEngine.RateLimit")
rate_limit_logger.setLevel(logging.INFO)
rate_limit_logger.addHandler(rate_limit_handler)
rate_limit_logger.propagate = False

# Health Check Log - health check results
health_handler = RotatingFileHandler(log_dir / 'pool_engine_health.log', maxBytes=5*1024*1024, backupCount=3)
health_handler.setFormatter(log_formatter)
health_logger = logging.getLogger("PoolEngine.Health")
health_logger.setLevel(logging.INFO)
health_logger.addHandler(health_handler)
health_logger.propagate = False

# Console output (shared across all pool engine loggers)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)


def _normalize_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize datetime to naive UTC for consistent comparisons.
    PostgreSQL with asyncpg can return timezone-aware datetimes,
    but we use datetime.utcnow() which is timezone-naive.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC and remove timezone info
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# Wave-F5 multi-key support: numbered env/secret variants (KEY, KEY_2 .. KEY_9)
# each become their OWN Endpoint row at equal priority so the round-robin
# selection in ProviderEndpoints.get_next_endpoint rotates across accounts.
MAX_NUMBERED_KEYS = 9


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    """Strip quotes/placeholders. Returns None for unset/placeholder values."""
    if not value:
        return None
    value = value.strip().strip('"').strip("'")
    if not value or value in ('null', 'None') or value.startswith('your_'):
        return None
    return value


def _numbered_env_values(env_var: str) -> List[Tuple[str, str]]:
    """[(name, value)] for VAR, VAR_2 .. VAR_9 — skips unset/placeholder slots.

    Lets the operator provision several accounts per provider (e.g. 3-4
    Helius keys) without touching the DB: HELIUS_API_KEY=..., HELIUS_API_KEY_2=...
    Fail-soft: zero extra vars set == exactly today's single-key behaviour.
    """
    out: List[Tuple[str, str]] = []
    for i in range(1, MAX_NUMBERED_KEYS + 1):
        name = env_var if i == 1 else f"{env_var}_{i}"
        value = _clean_env_value(os.getenv(name))
        if value:
            out.append((name, value))
    return out


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

    @property
    def is_soft_usable(self) -> bool:
        """
        True iff the endpoint is enabled and not hard-disabled/permanently-unhealthy.

        Used as a last-resort fallback when ALL endpoints in a pool are rate-limited
        or transiently unhealthy.  We pick the one whose rate-limit expires soonest
        rather than returning None and starving callers entirely.
        """
        if not self.is_enabled:
            return False
        if self.status == EndpointStatus.DISABLED:
            return False
        # Permanently unhealthy (>= 10 consecutive failures) is still hard-blocked
        if self.status == EndpointStatus.UNHEALTHY and self.consecutive_failures >= 10:
            return False
        return True

    @property
    def seconds_until_available(self) -> float:
        """Seconds until this endpoint becomes available (0 if already available)."""
        if self.is_available:
            return 0.0
        if self.status == EndpointStatus.RATE_LIMITED and self.rate_limit_until:
            now = datetime.now(self.rate_limit_until.tzinfo)
            remaining = (self.rate_limit_until - now).total_seconds()
            return max(0.0, remaining)
        return float('inf')

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
        """
        Get next available endpoint using weighted round-robin.

        If no endpoint is fully available (all are rate-limited / transiently
        unhealthy), fall back to the least-penalized soft-usable endpoint so
        callers never receive None when at least one recoverable endpoint exists.
        """
        available = self.get_available_endpoints()
        if not available:
            # Last-resort: pick the soft-usable endpoint with the soonest
            # rate-limit expiry rather than returning None and starving callers.
            return self._get_least_penalized_fallback()

        # BUG 3 — spread load across all equal-priority healthy endpoints
        # instead of hammering one. Weighted-RANDOM selection (the previous
        # behaviour) can return the SAME endpoint many times in a row, which
        # under a 33-wallet burst meant one Helius key got every call and
        # starved while sibling keys sat idle. We now ROUND-ROBIN through the
        # top tier so consecutive get_endpoint() calls rotate to the next
        # sibling first, only repeating a key after every sibling was handed
        # out once. Weight is preserved as a tie/ordering influence (higher
        # weight sorts earlier) but no longer lets one endpoint monopolise.
        top_priority = available[0].priority
        top_tier = [e for e in available if e.priority == top_priority]

        if len(top_tier) == 1:
            return top_tier[0]

        # Stable order within the tier: higher weight first, then by id so the
        # rotation is deterministic across calls.
        top_tier.sort(key=lambda e: (-e.weight, e.id))

        # Round-robin cursor advances every call; modulo the live tier size.
        idx = self.last_selected_index % len(top_tier)
        self.last_selected_index = (self.last_selected_index + 1) % max(1, len(top_tier))
        return top_tier[idx]

    def _get_least_penalized_fallback(self) -> Optional[Endpoint]:
        """
        When all endpoints are temporarily unavailable, return the one closest
        to recovery (smallest `seconds_until_available`).  Hard-disabled and
        permanently-unhealthy endpoints are excluded via `is_soft_usable`.
        """
        candidates = [e for e in self.endpoints if e.is_soft_usable]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.seconds_until_available)


class TokenBucket:
    """
    Async token-bucket rate limiter (FEATURE 2/3).

    Reusable, fail-soft pacing primitive shared via the PoolEngine so any
    module can space its outbound bursts instead of stampeding a single
    free-tier key. ``rate`` tokens are added per second up to ``capacity``;
    ``acquire()`` waits until a token is available, spreading an N-call burst
    over time rather than firing it all at once.

    Design notes:
    - Monotonic clock; immune to wall-clock jumps.
    - A single asyncio.Lock serialises refill+take so concurrent callers are
      naturally spaced (each waits its slice).
    - Conservative: never raises. A misuse (rate<=0) degrades to a no-op so a
      bad config can never wedge a module.
    """

    def __init__(self, rate: float, capacity: float):
        self.rate = max(0.0, float(rate))
        self.capacity = max(1.0, float(capacity))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them."""
        if self.rate <= 0:
            return  # no-op limiter (disabled by config)
        tokens = min(tokens, self.capacity)
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate if self.rate > 0 else 0.05
            # Sleep OUTSIDE the lock so other callers can refill-check too;
            # the lock-reacquire on the next loop re-serialises the take.
            await asyncio.sleep(max(0.001, wait))


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
        # endpoint.id -> last time report_success persisted it (throttle)
        self._last_success_persist: Dict[int, datetime] = {}

        # Rate limit backoff settings
        self._default_rate_limit_duration = 300  # 5 minutes
        self._max_consecutive_failures = 10

        # BUG 1 — rate-limit log-storm control.
        # Track which endpoints are CURRENTLY in the logged rate-limited state
        # so we emit exactly ONE concise WARNING per state transition (entered
        # rate-limit / recovered) instead of one line per 429 from three
        # loggers. Repeats within the window are demoted to DEBUG.
        # key = endpoint id; value = monotonic ts of last WARNING emitted.
        self._rl_logged_state: Dict[int, bool] = {}
        self._rl_last_warn_ts: Dict[int, float] = {}
        self._rl_suppressed_count: Dict[int, int] = {}
        self._rl_warn_throttle_s = 60.0  # re-WARN at most once per endpoint/min

        # FEATURE 4 — keep-alive liveness rotation.
        # Cheap getHealth/eth_blockNumber ping rotated slowly through EVERY
        # endpoint (including low-priority fallbacks) so provider keys that
        # auto-disable on idle (e.g. Ankr 30-day) stay warm. Index walks the
        # flattened endpoint list; interval is deliberately slow.
        self._keepalive_interval = 1800  # 30 min between single-endpoint pings
        self._keepalive_task: Optional[asyncio.Task] = None
        self._keepalive_cursor = 0

        # FEATURE 2/3 — per-provider outbound pacing (token bucket).
        # Reusable async rate limiters keyed by provider_type so any module
        # can space its bursts instead of stampeding a free-tier key. Lazily
        # created on first acquire(); conservative Helius-free-tier defaults.
        self._rate_limiters: Dict[str, 'TokenBucket'] = {}
        self._default_rps = 8.0   # ~8 req/s — under Helius free ~10 req/s
        self._default_burst = 8

        # Wave-F5 — env-fallback rotation cursors for get_api_key() so even
        # a DB-less bootstrap round-robins through numbered env keys.
        self._env_key_cursors: Dict[str, int] = {}

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
                    # Multi-key bootstrap: register any numbered API keys the
                    # operator added to secrets/.env since the pool was seeded.
                    try:
                        await self._load_keys_from_secrets()
                    except Exception as e:
                        logger.debug(f"secrets key probe skipped: {e}")
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

            # Multi-key bootstrap: DB-stored encrypted keys (fail-soft).
            try:
                await self._load_keys_from_secrets()
            except Exception as e:
                logger.debug(f"secrets key probe skipped: {e}")

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
                        # Normalize datetimes to naive UTC for consistent comparisons
                        rate_limit_until=_normalize_datetime(row['rate_limit_until']),
                        last_success_at=_normalize_datetime(row['last_success_at']),
                        last_failure_at=_normalize_datetime(row['last_failure_at']),
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

        # API mappings — every var also supports numbered _2.._9 variants,
        # one Endpoint per key at EQUAL priority (round-robin rotation).
        api_mappings = {
            'GOPLUS_API_KEY': ('GOPLUS_API', None, 'https://api.gopluslabs.io'),
            '1INCH_API_KEY': ('1INCH_API', None, 'https://api.1inch.io'),
            'HELIUS_API_KEY': ('HELIUS_API', 'solana', None),  # URL built from key
            'ETHERSCAN_API_KEY': ('ETHERSCAN_API', 'ethereum', 'https://api.etherscan.io'),
            'JUPITER_API_KEY': ('JUPITER_API', 'solana', None),  # Uses JUPITER_API_URL
            'BIRDEYE_API_KEY': ('BIRDEYE_API', 'solana', 'https://public-api.birdeye.so'),
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

        # Load single RPC URLs (numbered variants: ETHEREUM_RPC_URL_2 .. _9
        # register as equal-priority siblings so rotation spreads load).
        for env_var, (provider_type, chain) in single_rpc_mappings.items():
            base_priority = 50 if 'PROVIDER' in env_var or 'BACKUP' not in env_var else 150
            for name, value in _numbered_env_values(env_var):
                # Check if already added
                if value not in self._endpoint_cache:
                    endpoint_id += 1
                    endpoint = Endpoint(
                        id=endpoint_id,
                        provider_type=provider_type,
                        name=name,
                        url=value,
                        chain=chain,
                        priority=base_priority
                    )
                    self._add_endpoint(endpoint)
                    self._endpoint_cache[value] = endpoint

        # Load WebSocket URLs (numbered variants supported: SOLANA_WS_URL_2 ...)
        for env_var, (provider_type, chain) in ws_mappings.items():
            for name, value in _numbered_env_values(env_var):
                if value in self._endpoint_cache:
                    continue
                endpoint_id += 1
                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=name,
                    url=value,
                    chain=chain,
                    supports_ws=True
                )
                self._add_endpoint(endpoint)
                self._endpoint_cache[value] = endpoint

        # Load APIs — numbered variants (HELIUS_API_KEY_2 ...) each become
        # their own Endpoint at equal priority so get_next_endpoint round-robins
        # across accounts instead of pinning one key for the process lifetime.
        for env_var, (provider_type, chain, base_url) in api_mappings.items():
            for slot, (name, api_key) in enumerate(_numbered_env_values(env_var), start=1):
                if self._has_api_key(provider_type, api_key):
                    continue  # same key pasted into two slots
                endpoint_id += 1
                url = self._api_url_for_key(provider_type, api_key, slot, base_url)
                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=name,
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

    def _has_api_key(self, provider_type: str, api_key: str) -> bool:
        """True if this exact key is already registered for provider_type."""
        provider = self.providers.get(provider_type)
        if not provider:
            return False
        return any(e.api_key == api_key for e in provider.endpoints)

    @staticmethod
    def _api_url_for_key(provider_type: str, api_key: str, slot: int,
                         base_url: Optional[str]) -> str:
        """Endpoint URL for an API key. rpc_api_pool has UNIQUE(provider_type,
        url), so keys sharing one base URL (Etherscan/Birdeye slot 2+) get a
        harmless ?key_slot=N tag the upstream API ignores. Helius URLs embed
        the key and are naturally unique; slot 1 keeps today's bare URL so
        existing DB rows/seeds keep matching."""
        if provider_type == 'HELIUS_API':
            return f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        if provider_type == 'JUPITER_API':
            url = os.getenv('JUPITER_API_URL', 'https://lite-api.jup.ag')
        else:
            url = base_url or ''
        if slot > 1 and url:
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}key_slot={slot}"
        return url

    def _next_local_id(self) -> int:
        """Next synthetic endpoint id for DB-less operation."""
        ids = [e.id for p in self.providers.values() for e in p.endpoints]
        return (max(ids) + 1) if ids else 1

    # API-key secrets probed at initialize(): base name + _2.._9 variants.
    # Keys can live Fernet-encrypted in the secure_credentials DB table
    # instead of .env (secrets_manager get_async also falls back to env).
    _SECRET_API_KEY_MAP = {
        'HELIUS_API_KEY': ('HELIUS_API', 'solana', None),
        'ETHERSCAN_API_KEY': ('ETHERSCAN_API', 'ethereum', 'https://api.etherscan.io'),
        'BIRDEYE_API_KEY': ('BIRDEYE_API', 'solana', 'https://public-api.birdeye.so'),
        'GOPLUS_API_KEY': ('GOPLUS_API', None, 'https://api.gopluslabs.io'),
        '1INCH_API_KEY': ('1INCH_API', None, 'https://api.1inch.io'),
    }

    async def _load_keys_from_secrets(self) -> int:
        """Probe the secrets manager for numbered API keys and register any
        endpoint not already known (from DB or env). Idempotent + fail-soft:
        import failure, no DB, decryption failure or zero keys == no-op, so
        behaviour without extra keys is exactly today's.

        Returns the number of endpoints added.
        """
        try:
            from security.secrets_manager import secrets
        except Exception:
            return 0
        try:
            # Idempotent re-init is safe (Wave-13); ensures the DB-backed
            # encrypted-credentials path is available before we probe.
            if self.db_pool is not None and getattr(secrets, '_db_pool', None) is None:
                secrets.initialize(self.db_pool)
        except Exception as e:
            logger.debug(f"secrets re-init skipped: {e}")

        added = 0
        for env_var, (provider_type, chain, base_url) in self._SECRET_API_KEY_MAP.items():
            for slot in range(1, MAX_NUMBERED_KEYS + 1):
                name = env_var if slot == 1 else f"{env_var}_{slot}"
                try:
                    value = _clean_env_value(
                        await secrets.get_async(name, log_access=False)
                    )
                except Exception:
                    value = None
                if not value or self._has_api_key(provider_type, value):
                    continue
                url = self._api_url_for_key(provider_type, value, slot, base_url)
                if url and url in self._endpoint_cache:
                    continue
                endpoint_id = await self._insert_endpoint_row(
                    provider_type, name, url, value, chain
                )
                if endpoint_id is None:
                    continue
                endpoint = Endpoint(
                    id=endpoint_id,
                    provider_type=provider_type,
                    name=name,
                    url=url,
                    api_key=value,
                    chain=chain
                )
                self._add_endpoint(endpoint)
                if url:
                    self._endpoint_cache[url] = endpoint
                added += 1
        if added:
            logger.info(f"Registered {added} API-key endpoint(s) from secrets manager")
        return added

    async def _insert_endpoint_row(
        self, provider_type: str, name: str, url: str,
        api_key: Optional[str], chain: Optional[str]
    ) -> Optional[int]:
        """Idempotent insert into rpc_api_pool; returns the new row id.
        Returns None when the row already exists (e.g. operator-disabled —
        we respect that and do NOT resurrect it in memory). Without a DB
        pool, hands back a synthetic in-memory id."""
        if not self.db_pool:
            return self._next_local_id()
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO rpc_api_pool (
                        endpoint_type, provider_type, name, url, api_key,
                        status, is_enabled, priority, weight, chain
                    ) VALUES ($1, $2, $3, $4, $5, 'active', TRUE, 100, 100, $6)
                    ON CONFLICT (provider_type, url) DO NOTHING
                    RETURNING id
                """,
                    'rpc' if 'RPC' in provider_type else ('ws' if 'WS' in provider_type else 'api'),
                    provider_type, name, url, api_key, chain
                )
                return row['id'] if row else None
        except Exception as e:
            logger.debug(f"secret-key endpoint insert failed ({name}): {e}")
            return self._next_local_id()

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
            # All endpoints are hard-disabled or permanently unhealthy; .env is last resort.
            logger.warning(f"No available endpoints for: {provider_type} (all hard-blocked)")
            return self._get_env_fallback(provider_type)

        available_count = len(provider.get_available_endpoints())
        if available_count == 0:
            # Soft fallback: endpoint is rate-limited but will recover — log clearly.
            secs = endpoint.seconds_until_available
            logger.warning(
                f"All {provider_type} endpoints rate-limited; using least-penalized "
                f"'{endpoint.name}' (recovers in {secs:.0f}s)"
            )
            rate_limit_logger.warning(
                f"STARVED {provider_type}: returning '{endpoint.name}' "
                f"despite rate-limit (soonest recovery {secs:.0f}s)"
            )
        else:
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

    # =========================================================================
    # Public API - Rotated API keys (Wave-F5 multi-key)
    # =========================================================================

    def _select_endpoint(self, provider_type: str) -> Optional[Endpoint]:
        """Rotated healthy endpoint (same policy as get_endpoint, object form)."""
        provider = self.providers.get(provider_type)
        if not provider:
            return None
        return provider.get_next_endpoint()

    @staticmethod
    def _extract_api_key(endpoint: Endpoint) -> Optional[str]:
        """API key for an endpoint: the api_key column, else parsed from a
        key-in-URL style endpoint (Helius ?api-key=...)."""
        if endpoint.api_key:
            return endpoint.api_key
        if endpoint.url and 'api-key=' in endpoint.url:
            try:
                import urllib.parse as _up
                qs = _up.parse_qs(_up.urlparse(endpoint.url).query)
                return (qs.get('api-key') or qs.get('apikey') or [None])[0]
            except Exception:
                return None
        return None

    async def get_api_key(self, provider_type: str) -> Optional[Tuple[str, int]]:
        """
        CURRENT rotated healthy API key for a provider type.

        Consumers should call this per call-batch (not once at init) so a
        429-cooled key rotates out and sibling accounts share the load, then
        report the outcome via report_key_success / report_key_rate_limit /
        report_key_failure using the returned endpoint id.

        Returns:
            (api_key, endpoint_id) or None. endpoint_id -1 means the key came
            from the .env fallback (reports against it are no-ops).
        """
        if self.initialized:
            endpoint = self._select_endpoint(provider_type)
            if endpoint:
                key = self._extract_api_key(endpoint)
                if key:
                    full_logger.debug(
                        f"KEY SELECTED: {provider_type} -> {endpoint.name} (id={endpoint.id})"
                    )
                    return (key, endpoint.id)
        # .env fallback — rotate through numbered vars so even DB-less
        # bootstrap spreads load across configured keys.
        env_var = {
            'HELIUS_API': 'HELIUS_API_KEY',
            'ETHERSCAN_API': 'ETHERSCAN_API_KEY',
            'BIRDEYE_API': 'BIRDEYE_API_KEY',
            'GOPLUS_API': 'GOPLUS_API_KEY',
            '1INCH_API': '1INCH_API_KEY',
            'JUPITER_API': 'JUPITER_API_KEY',
        }.get(provider_type)
        if not env_var:
            return None
        values = [v for _, v in _numbered_env_values(env_var)]
        if not values:
            return None
        cursor = self._env_key_cursors.get(provider_type, 0)
        self._env_key_cursors[provider_type] = (cursor + 1) % len(values)
        return (values[cursor % len(values)], -1)

    def _find_endpoint_by_ref(self, ref) -> Optional[Endpoint]:
        """Resolve an endpoint by id (int), api key or URL (str)."""
        if ref is None:
            return None
        if isinstance(ref, int):
            if ref < 0:
                return None  # env-fallback sentinel
            for provider in self.providers.values():
                for ep in provider.endpoints:
                    if ep.id == ref:
                        return ep
            return None
        if isinstance(ref, str):
            ep = self._endpoint_cache.get(ref)
            if ep:
                return ep
            for provider in self.providers.values():
                for ep in provider.endpoints:
                    if ep.api_key == ref or ep.url == ref or self._extract_api_key(ep) == ref:
                        return ep
        return None

    async def report_key_success(self, ref, latency_ms: int = None) -> None:
        """report_success by endpoint id / api key (from get_api_key)."""
        endpoint = self._find_endpoint_by_ref(ref)
        if endpoint:
            await self.report_success(endpoint.provider_type, endpoint.url, latency_ms)

    async def report_key_rate_limit(self, ref, duration_seconds: int = None,
                                    error_message: str = None) -> None:
        """report_rate_limit by endpoint id / api key. Cools THIS key so the
        next get_api_key returns a sibling account. No-op for env-fallback
        sentinel (-1) or unknown refs — fail-soft."""
        endpoint = self._find_endpoint_by_ref(ref)
        if endpoint:
            await self.report_rate_limit(
                endpoint.provider_type, endpoint.url, duration_seconds, error_message
            )

    async def report_key_failure(self, ref, error_type: str = None,
                                 error_message: str = None) -> None:
        """report_failure by endpoint id / api key."""
        endpoint = self._find_endpoint_by_ref(ref)
        if endpoint:
            await self.report_failure(
                endpoint.provider_type, endpoint.url, error_type, error_message
            )

    # =========================================================================
    # Public API - Outbound Pacing (FEATURE 2/3)
    # =========================================================================

    def configure_rate_limiter(
        self, provider_type: str, rps: float, burst: Optional[float] = None
    ) -> None:
        """
        Configure (or reconfigure) the per-provider token-bucket limiter.

        Any module that fans out many calls to one provider (e.g. COPY polling
        33 Solana wallets through HELIUS_API) should call this once with a
        rate suited to that provider's free-tier ceiling, then await
        ``acquire_rate_limit(provider_type)`` before each outbound request.

        Fail-soft: invalid input is clamped, never raised.
        """
        try:
            rps = max(0.0, float(rps))
            cap = float(burst) if burst is not None else max(1.0, rps)
            existing = self._rate_limiters.get(provider_type)
            if existing is None:
                self._rate_limiters[provider_type] = TokenBucket(rps, cap)
            else:
                # Reconfigure in place so live tokens aren't reset to full.
                existing.rate = rps
                existing.capacity = max(1.0, cap)
            logger.debug(
                f"Rate limiter configured: {provider_type} -> {rps:.1f} req/s (burst {cap:.0f})"
            )
        except Exception as e:
            logger.debug(f"configure_rate_limiter({provider_type}) ignored: {e}")

    async def acquire_rate_limit(self, provider_type: str, tokens: float = 1.0) -> None:
        """
        Wait for outbound capacity on ``provider_type`` before a request.

        Lazily creates a conservative default limiter (``_default_rps``) the
        first time a provider is seen, so callers get sane pacing even without
        an explicit ``configure_rate_limiter`` call. Fail-soft: never raises.
        """
        try:
            limiter = self._rate_limiters.get(provider_type)
            if limiter is None:
                limiter = TokenBucket(self._default_rps, self._default_burst)
                self._rate_limiters[provider_type] = limiter
            await limiter.acquire(tokens)
        except Exception as e:
            logger.debug(f"acquire_rate_limit({provider_type}) no-op: {e}")

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
            'BIRDEYE_API': 'BIRDEYE_API_KEY',
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

        # Honour a caller-supplied duration exactly (e.g. parsed Retry-After);
        # otherwise exponential back-off so repeated 429s cool down (cap 30 min).
        if duration_seconds:
            duration = duration_seconds
        else:
            base = self._default_rate_limit_duration  # 300 s
            duration = min(base * (2 ** max(0, endpoint.rate_limit_count)), 1800)

        endpoint.status = EndpointStatus.RATE_LIMITED
        endpoint.rate_limit_until = datetime.utcnow() + timedelta(seconds=duration)
        endpoint.rate_limit_count += 1
        endpoint.last_rate_limit_at = datetime.utcnow()
        endpoint.health_score = max(0, endpoint.health_score - 10)

        # Move to end of queue by increasing priority
        endpoint.priority = min(endpoint.priority + 50, 1000)

        # BUG 1 — single concise log line per STATE TRANSITION, not per 429.
        # The previous code emitted the same WARNING from three loggers
        # (PoolEngine, PoolEngine.RateLimit, PoolEngine.Full) on EVERY hit,
        # flooding copy_trading/stderr.log under 33-wallet bursts. Now: one
        # WARNING when the endpoint ENTERS the rate-limited state (or once per
        # throttle window thereafter), and repeats are demoted to DEBUG with a
        # suppressed-count audited to the dedicated rate-limit log file.
        eid = endpoint.id
        now_mono = time.monotonic()
        was_logged = self._rl_logged_state.get(eid, False)
        last_warn = self._rl_last_warn_ts.get(eid, 0.0)
        rate_limit_msg = (
            f"Rate limited: {provider_type} - {endpoint.name} "
            f"(until {endpoint.rate_limit_until}, count={endpoint.rate_limit_count}, "
            f"duration={duration}s, priority={endpoint.priority})"
        )
        if not was_logged or (now_mono - last_warn) >= self._rl_warn_throttle_s:
            suppressed = self._rl_suppressed_count.pop(eid, 0)
            suffix = (
                f" [+{suppressed} further occurrences demoted to DEBUG]"
                if suppressed else
                f" (further occurrences demoted to DEBUG for {int(self._rl_warn_throttle_s)}s)"
            )
            logger.warning(rate_limit_msg + suffix)
            self._rl_logged_state[eid] = True
            self._rl_last_warn_ts[eid] = now_mono
        else:
            self._rl_suppressed_count[eid] = self._rl_suppressed_count.get(eid, 0) + 1
            logger.debug(rate_limit_msg + " (throttled)")
        # Audit trail (file-only; not console) keeps full forensic history.
        rate_limit_logger.info(rate_limit_msg)

        # Update database
        await self._update_endpoint_status(endpoint)

        # Log usage
        await self._log_usage(endpoint, False, error_type='rate_limit', error_message=error_message)

    def _note_rate_limit_recovery(self, endpoint: Endpoint) -> None:
        """
        BUG 1 — log exactly ONE recovery line per state transition and clear
        the storm-control bookkeeping so the NEXT rate-limit logs fresh.
        """
        eid = endpoint.id
        if self._rl_logged_state.get(eid):
            suppressed = self._rl_suppressed_count.pop(eid, 0)
            tail = f" ({suppressed} suppressed while limited)" if suppressed else ""
            logger.info(f"Recovered from rate limit: {endpoint.name}{tail}")
            health_logger.info(f"Endpoint recovered from rate limit: {endpoint.name}")
        self._rl_logged_state.pop(eid, None)
        self._rl_last_warn_ts.pop(eid, None)
        self._rl_suppressed_count.pop(eid, None)

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

        prev_status = endpoint.status
        endpoint.success_count += 1
        endpoint.last_success_at = datetime.utcnow()
        endpoint.consecutive_failures = 0

        # Reset status if it was rate limited and limit has passed
        if endpoint.status == EndpointStatus.RATE_LIMITED:
            if not endpoint.rate_limit_until or datetime.utcnow() >= endpoint.rate_limit_until:
                endpoint.status = EndpointStatus.ACTIVE
                endpoint.rate_limit_until = None
                self._note_rate_limit_recovery(endpoint)

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

        # Persist to rpc_api_pool so the dashboard (which reads the DB, not
        # this process's memory) reflects recovery/latency/score. Only the
        # failure/rate-limit paths persisted before, so a 'rate_limited'
        # status stuck in the DB forever and the Test button never updated
        # ping or health. Throttled: always on a status change (recovery),
        # otherwise at most once per 60s per endpoint — report_success fires
        # on every RPC call and must not become a DB write storm.
        if endpoint.id is not None and self.db_pool:
            now = datetime.utcnow()
            last = self._last_success_persist.get(endpoint.id)
            if (endpoint.status != prev_status or last is None
                    or (now - last).total_seconds() >= 60):
                self._last_success_persist[endpoint.id] = now
                asyncio.create_task(self._update_endpoint_status(endpoint))

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
        # FEATURE 4 — also start the slow keep-alive rotation alongside.
        self._start_keepalive_task()

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

    def _start_keepalive_task(self) -> None:
        """Start the slow keep-alive liveness rotation (FEATURE 4)."""
        if self._keepalive_interval <= 0:
            return
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            logger.info(
                f"Started keep-alive rotation (every {self._keepalive_interval}s, "
                "1 endpoint per tick)"
            )

    async def _keepalive_loop(self) -> None:
        """
        FEATURE 4 — keep provider keys warm.

        Providers like Ankr DISABLE keys idle for ~30 days. The full health
        check only probes RPC endpoints and runs hourly; low-priority/poor-ping
        fallbacks that selection never picks could still go cold. This loop
        rotates a SINGLE cheap liveness ping (Solana getHealth / EVM
        eth_blockNumber) through EVERY enabled endpoint — one per tick on a
        deliberately slow schedule — so each endpoint is touched several times
        per week regardless of selection. Adds negligible load and does not
        influence selection (we report success only; a failure here does not
        penalise an otherwise-unused fallback into the unhealthy state).
        """
        while True:
            try:
                await asyncio.sleep(self._keepalive_interval)
                endpoint = self._next_keepalive_endpoint()
                if endpoint is None:
                    continue
                await self._keepalive_ping(endpoint)
            except asyncio.CancelledError:
                logger.info("Keep-alive task cancelled")
                break
            except Exception as e:
                logger.debug(f"Keep-alive loop error: {e}")
                await asyncio.sleep(60)

    def _flatten_endpoints(self) -> List[Endpoint]:
        """Flatten all endpoints across providers into one ordered list."""
        flat: List[Endpoint] = []
        for provider in self.providers.values():
            flat.extend(provider.endpoints)
        return flat

    def _next_keepalive_endpoint(self) -> Optional[Endpoint]:
        """Pick the next endpoint to keep warm, walking the flattened list."""
        flat = [e for e in self._flatten_endpoints() if e.is_enabled]
        if not flat:
            return None
        idx = self._keepalive_cursor % len(flat)
        self._keepalive_cursor = (self._keepalive_cursor + 1) % len(flat)
        return flat[idx]

    async def _keepalive_ping(self, endpoint: Endpoint) -> None:
        """
        Cheap liveness ping for keep-alive. RPC endpoints get a real
        getHealth/eth_blockNumber; non-RPC (API) endpoints are skipped (no
        universal cheap probe). Success is reported (keeps last_success fresh,
        which is what providers watch for idle-disable); failure is logged at
        DEBUG only and does NOT mark the endpoint unhealthy — this is a warmth
        ping for fallbacks, not a selection-affecting health verdict.
        """
        try:
            if 'RPC' not in endpoint.provider_type and not endpoint.chain:
                return
            if endpoint.chain == 'solana' or 'solana' in endpoint.provider_type.lower():
                payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
            else:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.get_effective_url(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as response:
                    latency = int((time.time() - start) * 1000)
                    if response.status == 200:
                        await self.report_success(
                            endpoint.provider_type, endpoint.url, latency
                        )
                        health_logger.info(
                            f"Keep-alive OK: {endpoint.name} ({latency}ms)"
                        )
                        return
                    health_logger.debug(
                        f"Keep-alive non-200 for {endpoint.name}: {response.status}"
                    )
        except Exception as e:
            health_logger.debug(f"Keep-alive ping failed for {endpoint.name}: {e}")

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
                        self._note_rate_limit_recovery(endpoint)

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

        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
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
