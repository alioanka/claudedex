"""
Safety Engine for Solana Trading Module

Provides multiple layers of protection:
1. Honeypot detection - verify sell route exists before buying
2. Aggressive retry for failed closes with escalating slippage
3. Circuit breaker to pause trading on multiple failures
4. Emergency close mechanism for stuck positions
5. Quote caching for rate-limit-resilient emergency closes
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger('SolanaTradingEngine')


@dataclass
class CachedQuote:
    """Cached sell quote for emergency closes"""
    token_mint: str
    token_symbol: str
    quote_data: Dict[str, Any]
    estimated_out_lamports: int
    cached_at: float  # Unix timestamp
    slippage_bps: int
    route_label: str  # e.g., "Raydium CLMM"

    def is_fresh(self, max_age_seconds: int = 60) -> bool:
        """Check if quote is still fresh enough to use"""
        return (time.time() - self.cached_at) < max_age_seconds

    def get_age_seconds(self) -> float:
        """Get age of quote in seconds"""
        return time.time() - self.cached_at


class CloseFailureReason(Enum):
    """Reasons why a close might fail"""
    NO_JUPITER_ROUTE = "no_jupiter_route"
    SLIPPAGE_EXCEEDED = "slippage_exceeded"
    RATE_LIMITED = "rate_limited"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    RPC_ERROR = "rpc_error"
    HONEYPOT = "honeypot"
    UNKNOWN = "unknown"


@dataclass
class StuckPosition:
    """Tracks a position that failed to close"""
    token_mint: str
    token_symbol: str
    original_value_sol: float
    first_failure_time: datetime
    failure_count: int = 0
    last_failure_reason: CloseFailureReason = CloseFailureReason.UNKNOWN
    last_slippage_tried: int = 0
    next_retry_time: Optional[datetime] = None
    max_slippage_reached: bool = False


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms"""
    # Slippage settings
    default_slippage_bps: int = 50  # 0.5% for stable tokens
    pumpfun_slippage_bps: int = 500  # 5% for pump.fun tokens
    max_emergency_slippage_bps: int = 2000  # 20% max emergency slippage

    # Retry settings
    max_close_retries: int = 5
    retry_slippage_increment_bps: int = 200  # Increase by 2% each retry
    retry_delay_seconds: int = 5

    # Circuit breaker
    circuit_breaker_failures: int = 3  # Consecutive failures to trigger
    circuit_breaker_cooldown_seconds: int = 300  # 5 min cooldown

    # Honeypot detection
    min_sell_quote_ratio: float = 0.5  # Must get back at least 50% of input


class SafetyEngine:
    """
    Manages all safety mechanisms for trading.

    Key features:
    - Pre-buy sell route verification (honeypot detection)
    - Retry failed closes with escalating slippage
    - Circuit breaker to pause trading on failures
    - Track and retry stuck positions
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()

        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reset_time: Optional[datetime] = None

        # Stuck positions tracking
        self.stuck_positions: Dict[str, StuckPosition] = {}

        # Quote cache for emergency closes during rate limits
        self.quote_cache: Dict[str, CachedQuote] = {}
        self.quote_cache_max_age_seconds: int = 120  # 2 minute cache
        self.quote_cache_emergency_max_age_seconds: int = 300  # 5 min for emergencies

        # Stats
        self.total_close_attempts = 0
        self.total_close_failures = 0
        self.honeypots_detected = 0
        self.emergency_closes_from_cache = 0

    def get_slippage_for_strategy(self, strategy: str, is_close: bool = False,
                                   retry_count: int = 0) -> int:
        """
        Get appropriate slippage for a trade.

        Args:
            strategy: Trading strategy (pumpfun, jupiter, etc.)
            is_close: Whether this is a close/sell transaction
            retry_count: Number of previous failed attempts

        Returns:
            Slippage in basis points
        """
        # Base slippage depends on strategy
        if strategy.lower() == 'pumpfun':
            base_slippage = self.config.pumpfun_slippage_bps
        else:
            base_slippage = self.config.default_slippage_bps

        # For closes, use higher slippage and escalate on retries
        if is_close:
            # Minimum 2x for closes
            base_slippage = max(base_slippage, self.config.default_slippage_bps * 2)

            # Add increment for each retry
            escalation = retry_count * self.config.retry_slippage_increment_bps
            slippage = base_slippage + escalation

            # Cap at max emergency slippage
            slippage = min(slippage, self.config.max_emergency_slippage_bps)

            return slippage

        return base_slippage

    async def verify_sell_route(self, jupiter_client, token_mint: str,
                                 sol_mint: str, test_amount: int = 1000000,
                                 token_price_usd: float = 0.0,
                                 token_decimals: int = 6) -> Tuple[bool, str]:
        """
        Verify that a token can be sold back to SOL (honeypot detection).

        Args:
            jupiter_client: JupiterClient instance
            token_mint: Token mint address to check
            sol_mint: SOL mint address
            test_amount: Test amount in token units (fallback if price unknown)
            token_price_usd: Current token price in USD (for smart amount calculation)
            token_decimals: Token decimals (default 6)

        Returns:
            Tuple of (can_sell: bool, reason: str)
        """
        try:
            # Calculate a sensible test amount based on token price
            # We want to test with ~$1-5 worth of tokens to get meaningful quotes
            # without hitting minimum trade limits
            actual_test_amount = test_amount

            if token_price_usd > 0:
                # Calculate how many token units equal ~$2 USD
                target_usd_value = 2.0
                token_units_per_dollar = (10 ** token_decimals) / token_price_usd
                actual_test_amount = int(target_usd_value * token_units_per_dollar)
                # Ensure minimum of 1 token unit, max of 1B units
                actual_test_amount = max(1, min(actual_test_amount, 1_000_000_000))
            else:
                # Fallback: Use larger test amount for unknown tokens
                # 10M units covers most token decimal configurations
                actual_test_amount = 10_000_000

            # Try to get a sell quote (token -> SOL)
            sell_quote = await jupiter_client.get_quote(
                input_mint=token_mint,
                output_mint=sol_mint,
                amount=actual_test_amount
            )

            if not sell_quote:
                # Check if this might be a rate limit issue
                # Don't immediately flag as honeypot - could be temporary
                self.honeypots_detected += 1
                return False, "No sell route available via Jupiter - possible honeypot"

            # Check for error field in quote (indicates rate limit or other API error)
            if 'error' in sell_quote:
                error_msg = str(sell_quote.get('error', '')).lower()
                if '429' in error_msg or 'rate limit' in error_msg:
                    # Rate limited - don't flag as honeypot, allow trade
                    logger.warning(f"Jupiter rate limited during sell route verification - allowing trade")
                    return True, "Rate limited - verification skipped"
                # Other API errors - be conservative, allow trade
                return True, f"API error during verification: {sell_quote.get('error')}"

            # Check if output is reasonable (not 0 or extremely low)
            out_amount = int(sell_quote.get('outAmount', 0))
            if out_amount == 0:
                self.honeypots_detected += 1
                return False, "Sell quote returns 0 - likely honeypot"

            # Additional check: verify the route isn't suspiciously bad
            # We check if we'd get back a reasonable SOL amount
            # Minimum threshold: 100 lamports (0.0000001 SOL) to cover dust
            in_amount = int(sell_quote.get('inAmount', actual_test_amount))

            if in_amount > 0 and out_amount > 0:
                # For tokens with known price, validate the quote makes sense
                if token_price_usd > 0:
                    # Expected SOL output based on current prices
                    # Allow 50% slippage for low-liquidity pump.fun tokens
                    expected_usd = (in_amount / (10 ** token_decimals)) * token_price_usd
                    actual_sol = out_amount / 1e9  # Convert lamports to SOL
                    # Get approximate SOL price (assume ~$100 if unknown)
                    actual_usd = actual_sol * 100  # Rough estimate

                    # If we're getting less than 5% of expected value, flag it
                    if expected_usd > 0.001 and actual_usd < expected_usd * 0.05:
                        self.honeypots_detected += 1
                        return False, f"Sell quote too low ({actual_usd:.6f} vs expected {expected_usd:.6f}) - possible honeypot"
                else:
                    # For unknown price, just check we get SOME meaningful output
                    # Require at least 1000 lamports (0.000001 SOL) for any trade
                    if out_amount < 1000:
                        self.honeypots_detected += 1
                        return False, f"Sell quote too low ({out_amount} lamports) - possible honeypot"

            return True, "Sell route verified"

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit in exception message
            if '429' in error_str or 'rate limit' in error_str:
                logger.warning(f"Jupiter rate limited during verification - allowing trade")
                return True, "Rate limited - verification skipped"
            logger.warning(f"Error verifying sell route: {e}")
            # On error, be conservative and allow (might just be rate limit)
            return True, f"Could not verify (error: {e})"

    def check_circuit_breaker(self) -> Tuple[bool, Optional[int]]:
        """
        Check if circuit breaker is tripped.

        Returns:
            Tuple of (is_tripped: bool, seconds_until_reset: Optional[int])
        """
        if not self.circuit_breaker_tripped:
            return False, None

        if self.circuit_breaker_reset_time:
            now = datetime.now()
            if now >= self.circuit_breaker_reset_time:
                # Reset circuit breaker
                self.circuit_breaker_tripped = False
                self.circuit_breaker_reset_time = None
                self.consecutive_failures = 0
                logger.info("🟢 Circuit breaker reset - trading resumed")
                return False, None

            remaining = (self.circuit_breaker_reset_time - now).seconds
            return True, remaining

        return True, None

    def record_close_failure(self, token_mint: str, token_symbol: str,
                             value_sol: float, reason: CloseFailureReason,
                             slippage_used: int) -> bool:
        """
        Record a close failure and update circuit breaker.

        Args:
            token_mint: Token that failed to close
            token_symbol: Token symbol
            value_sol: Value in SOL
            reason: Failure reason
            slippage_used: Slippage that was tried

        Returns:
            True if should retry, False if max retries reached
        """
        self.total_close_failures += 1
        self.consecutive_failures += 1

        # Update or create stuck position record
        if token_mint not in self.stuck_positions:
            self.stuck_positions[token_mint] = StuckPosition(
                token_mint=token_mint,
                token_symbol=token_symbol,
                original_value_sol=value_sol,
                first_failure_time=datetime.now(),
                failure_count=1,
                last_failure_reason=reason,
                last_slippage_tried=slippage_used
            )
        else:
            stuck = self.stuck_positions[token_mint]
            stuck.failure_count += 1
            stuck.last_failure_reason = reason
            stuck.last_slippage_tried = slippage_used

        stuck = self.stuck_positions[token_mint]

        # Check if max retries reached
        if stuck.failure_count >= self.config.max_close_retries:
            stuck.max_slippage_reached = True
            logger.error(f"🚨 MAX RETRIES REACHED for {token_symbol}")
            logger.error(f"   Position is STUCK - manual intervention required")
            logger.error(f"   Token mint: {token_mint}")
            return False

        # Schedule next retry with delay
        stuck.next_retry_time = datetime.now() + timedelta(
            seconds=self.config.retry_delay_seconds * stuck.failure_count
        )

        # Check circuit breaker
        if self.consecutive_failures >= self.config.circuit_breaker_failures:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reset_time = datetime.now() + timedelta(
                seconds=self.config.circuit_breaker_cooldown_seconds
            )
            logger.error(f"🚨 CIRCUIT BREAKER TRIPPED - {self.consecutive_failures} consecutive failures")
            logger.error(f"   Trading paused for {self.config.circuit_breaker_cooldown_seconds}s")

        return True

    def record_close_success(self, token_mint: str):
        """Record a successful close."""
        self.consecutive_failures = 0

        if token_mint in self.stuck_positions:
            del self.stuck_positions[token_mint]

    def get_positions_to_retry(self) -> List[StuckPosition]:
        """Get stuck positions that are ready to retry."""
        ready = []
        now = datetime.now()

        for stuck in self.stuck_positions.values():
            if stuck.max_slippage_reached:
                continue
            if stuck.next_retry_time and now >= stuck.next_retry_time:
                ready.append(stuck)

        return ready

    def parse_error_reason(self, error_msg: str) -> CloseFailureReason:
        """Parse error message to determine failure reason."""
        error_lower = error_msg.lower()

        if '0x1771' in error_lower or 'slippage' in error_lower:
            return CloseFailureReason.SLIPPAGE_EXCEEDED
        elif '429' in error_lower or 'rate limit' in error_lower:
            return CloseFailureReason.RATE_LIMITED
        elif 'no quote' in error_lower or 'no route' in error_lower:
            return CloseFailureReason.NO_JUPITER_ROUTE
        elif 'insufficient' in error_lower or 'balance' in error_lower:
            return CloseFailureReason.INSUFFICIENT_BALANCE
        elif 'honeypot' in error_lower:
            return CloseFailureReason.HONEYPOT
        else:
            return CloseFailureReason.UNKNOWN

    def get_safety_stats(self) -> Dict:
        """Get safety statistics."""
        return {
            'total_close_attempts': self.total_close_attempts,
            'total_close_failures': self.total_close_failures,
            'failure_rate': self.total_close_failures / max(1, self.total_close_attempts),
            'honeypots_detected': self.honeypots_detected,
            'stuck_positions': len(self.stuck_positions),
            'circuit_breaker_tripped': self.circuit_breaker_tripped,
            'consecutive_failures': self.consecutive_failures,
            'cached_quotes': len(self.quote_cache),
            'emergency_closes_from_cache': self.emergency_closes_from_cache
        }

    # ==================== Quote Caching Methods ====================

    def cache_sell_quote(self, token_mint: str, token_symbol: str,
                         quote_data: Dict[str, Any], slippage_bps: int) -> bool:
        """
        Cache a sell quote for potential emergency close during rate limits.

        Should be called after successfully getting a quote for position monitoring.

        Args:
            token_mint: Token mint address
            token_symbol: Token symbol for logging
            quote_data: Raw Jupiter quote response
            slippage_bps: Slippage used for quote

        Returns:
            True if cached successfully
        """
        try:
            out_amount = int(quote_data.get('outAmount', 0))
            route_info = quote_data.get('routePlan', [{}])
            route_label = route_info[0].get('swapInfo', {}).get('label', 'Unknown') if route_info else 'Unknown'

            cached = CachedQuote(
                token_mint=token_mint,
                token_symbol=token_symbol,
                quote_data=quote_data,
                estimated_out_lamports=out_amount,
                cached_at=time.time(),
                slippage_bps=slippage_bps,
                route_label=route_label
            )

            self.quote_cache[token_mint] = cached
            logger.debug(f"📦 Cached sell quote for {token_symbol}: ~{out_amount / 1e9:.6f} SOL via {route_label}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache quote for {token_symbol}: {e}")
            return False

    def get_cached_quote(self, token_mint: str, emergency: bool = False) -> Optional[CachedQuote]:
        """
        Get a cached sell quote for a token.

        Args:
            token_mint: Token mint address
            emergency: If True, use longer max age (for rate-limited emergencies)

        Returns:
            CachedQuote if available and fresh, None otherwise
        """
        cached = self.quote_cache.get(token_mint)
        if not cached:
            return None

        max_age = (self.quote_cache_emergency_max_age_seconds if emergency
                   else self.quote_cache_max_age_seconds)

        if cached.is_fresh(max_age):
            return cached

        # Quote too old, remove it
        logger.debug(f"Cached quote for {cached.token_symbol} expired ({cached.get_age_seconds():.0f}s old)")
        del self.quote_cache[token_mint]
        return None

    def remove_cached_quote(self, token_mint: str):
        """Remove a cached quote (e.g., after position is closed)."""
        if token_mint in self.quote_cache:
            del self.quote_cache[token_mint]

    def cleanup_stale_quotes(self):
        """Remove all expired quotes from cache."""
        stale = []
        for mint, cached in self.quote_cache.items():
            # Use emergency max age for cleanup to keep quotes longer
            if not cached.is_fresh(self.quote_cache_emergency_max_age_seconds):
                stale.append(mint)

        for mint in stale:
            del self.quote_cache[mint]

        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale cached quotes")

    async def emergency_close_with_cache(self, token_mint: str, jupiter_client,
                                          wallet_keypair, token_balance: int,
                                          sol_mint: str) -> Tuple[bool, str]:
        """
        Attempt to close a position using cached quote when rate limited.

        This is a last-resort mechanism when fresh quotes cannot be obtained
        due to rate limiting. Uses previously cached route information.

        Args:
            token_mint: Token to sell
            jupiter_client: JupiterClient instance
            wallet_keypair: Wallet keypair for signing
            token_balance: Amount of tokens to sell
            sol_mint: SOL mint address

        Returns:
            Tuple of (success: bool, message: str)
        """
        cached = self.get_cached_quote(token_mint, emergency=True)

        if not cached:
            return False, "No cached quote available for emergency close"

        age_seconds = cached.get_age_seconds()
        logger.warning(f"⚠️ EMERGENCY CLOSE using {age_seconds:.0f}s old cached quote for {cached.token_symbol}")
        logger.warning(f"   Cached route: {cached.route_label}, slippage: {cached.slippage_bps}bps")

        try:
            # Increase slippage for emergency (prices may have moved)
            emergency_slippage = min(
                cached.slippage_bps + 500,  # Add 5% to cached slippage
                self.config.max_emergency_slippage_bps
            )

            # Try to get a fresh quote first (might work now)
            fresh_quote = None
            try:
                fresh_quote = await jupiter_client.get_quote(
                    input_mint=token_mint,
                    output_mint=sol_mint,
                    amount=token_balance,
                    slippage_bps=emergency_slippage
                )
            except Exception as e:
                logger.warning(f"Fresh quote failed during emergency close: {e}")

            quote_to_use = fresh_quote if fresh_quote else cached.quote_data

            # Execute the swap
            swap_result = await jupiter_client.execute_swap(
                quote=quote_to_use,
                wallet_keypair=wallet_keypair,
                slippage_bps=emergency_slippage
            )

            if swap_result and swap_result.get('success'):
                self.emergency_closes_from_cache += 1
                self.remove_cached_quote(token_mint)
                tx_sig = swap_result.get('signature', 'unknown')
                logger.info(f"✅ Emergency close successful for {cached.token_symbol}: {tx_sig}")
                return True, f"Emergency close successful: {tx_sig}"
            else:
                error = swap_result.get('error', 'Unknown error') if swap_result else 'Swap failed'
                return False, f"Emergency close failed: {error}"

        except Exception as e:
            logger.error(f"Emergency close exception for {cached.token_symbol}: {e}")
            return False, f"Emergency close exception: {e}"
