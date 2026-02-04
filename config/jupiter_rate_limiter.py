"""
Global Jupiter API Rate Limiter

Provides a singleton rate limiter for coordinating Jupiter API access
across all modules (Solana Trading, Solana Arbitrage, etc.)

Jupiter lite-api.jup.ag has a 1 RPS limit. This shared limiter ensures
that combined module usage stays within that limit.
"""
import asyncio
import time
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GlobalJupiterRateLimiter:
    """
    Global rate limiter for Jupiter API access.

    Uses a singleton pattern to ensure all modules share the same limiter,
    preventing combined RPS from exceeding the API limit.

    Features:
    - Token bucket algorithm with configurable RPS
    - Adaptive backoff on 429 errors
    - Thread-safe with asyncio Lock
    """

    _instance: Optional['GlobalJupiterRateLimiter'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if GlobalJupiterRateLimiter._initialized:
            return

        # Jupiter lite-api.jup.ag limit is 1 RPS
        # Use 0.8 to stay safely under the limit (leaves margin for burst)
        self.rps = 0.8
        self.min_interval = 1.0 / self.rps
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

        # Adaptive backoff for 429 errors
        self.consecutive_429s = 0
        self.base_backoff = 2.0
        self.max_backoff = 60.0
        self._backoff_until: Optional[float] = None

        # Statistics for monitoring
        self._total_requests = 0
        self._total_429s = 0
        self._last_stats_log = time.time()

        GlobalJupiterRateLimiter._initialized = True
        logger.info(f"🚦 Global Jupiter rate limiter initialized: {self.rps} RPS")

    async def acquire(self, caller: str = "unknown") -> None:
        """
        Wait until we can make a request without exceeding rate limit.

        Args:
            caller: Identifier for the calling module (for logging)
        """
        async with self._lock:
            now = time.time()

            # Check if we're in a backoff period from 429 errors
            if self._backoff_until and now < self._backoff_until:
                wait_time = self._backoff_until - now
                logger.warning(f"⏳ [{caller}] Jupiter backoff: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._backoff_until = None
                now = time.time()

            # Enforce minimum interval between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()
            self._total_requests += 1

            # Log stats every 5 minutes
            if time.time() - self._last_stats_log > 300:
                self._log_stats()

    def record_429(self, caller: str = "unknown") -> float:
        """
        Record a 429 error and calculate backoff time.

        Args:
            caller: Identifier for the calling module

        Returns:
            Backoff time in seconds
        """
        self.consecutive_429s += 1
        self._total_429s += 1

        # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s (max)
        backoff = min(self.base_backoff * (2 ** (self.consecutive_429s - 1)), self.max_backoff)
        self._backoff_until = time.time() + backoff

        logger.warning(f"⏱️ [{caller}] Jupiter rate limit #{self.consecutive_429s}, backing off {backoff:.1f}s")
        return backoff

    def record_success(self) -> None:
        """Record a successful request and reset backoff."""
        if self.consecutive_429s > 0:
            logger.info(f"✅ Jupiter rate limit recovered after {self.consecutive_429s} 429s")
        self.consecutive_429s = 0

    def get_current_backoff(self) -> float:
        """Get remaining backoff time, if any."""
        if self._backoff_until:
            remaining = self._backoff_until - time.time()
            return max(0.0, remaining)
        return 0.0

    def _log_stats(self) -> None:
        """Log rate limiter statistics."""
        logger.info(f"📊 Jupiter Rate Limiter Stats: "
                   f"Total requests: {self._total_requests} | "
                   f"429 errors: {self._total_429s} | "
                   f"Current RPS: {self.rps}")
        self._last_stats_log = time.time()


# Singleton accessor
def get_jupiter_rate_limiter() -> GlobalJupiterRateLimiter:
    """Get the global Jupiter rate limiter instance."""
    return GlobalJupiterRateLimiter()


# Convenience function for modules that need a standalone rate limiter class
class SharedJupiterRateLimiter:
    """
    Wrapper class that delegates to the global rate limiter.

    Can be used as a drop-in replacement for module-specific rate limiters.
    """

    def __init__(self, caller: str = "unknown"):
        self.caller = caller
        self._global = get_jupiter_rate_limiter()

    async def acquire(self) -> None:
        """Wait for rate limit."""
        await self._global.acquire(self.caller)

    def record_429(self) -> float:
        """Record 429 error."""
        return self._global.record_429(self.caller)

    def record_success(self) -> None:
        """Record success."""
        self._global.record_success()

    def get_current_backoff(self) -> float:
        """Get current backoff time."""
        return self._global.get_current_backoff()
