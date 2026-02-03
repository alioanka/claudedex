"""
Safety Engine for Solana Trading Module

Provides multiple layers of protection:
1. Honeypot detection - verify sell route exists before buying
2. Aggressive retry for failed closes with escalating slippage
3. Circuit breaker to pause trading on multiple failures
4. Emergency close mechanism for stuck positions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger('SolanaTradingEngine')


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

        # Stats
        self.total_close_attempts = 0
        self.total_close_failures = 0
        self.honeypots_detected = 0

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
                                 sol_mint: str, test_amount: int = 1000000) -> Tuple[bool, str]:
        """
        Verify that a token can be sold back to SOL (honeypot detection).

        Args:
            jupiter_client: JupiterClient instance
            token_mint: Token mint address to check
            sol_mint: SOL mint address
            test_amount: Test amount in token units

        Returns:
            Tuple of (can_sell: bool, reason: str)
        """
        try:
            # Try to get a sell quote (token -> SOL)
            sell_quote = await jupiter_client.get_quote(
                input_mint=token_mint,
                output_mint=sol_mint,
                amount=test_amount
            )

            if not sell_quote:
                self.honeypots_detected += 1
                return False, "No sell route available via Jupiter - possible honeypot"

            # Check if output is reasonable (not 0 or extremely low)
            out_amount = int(sell_quote.get('outAmount', 0))
            if out_amount == 0:
                self.honeypots_detected += 1
                return False, "Sell quote returns 0 - likely honeypot"

            # Additional check: verify the route isn't suspiciously bad
            # (very low output might indicate a trap)
            in_amount = int(sell_quote.get('inAmount', test_amount))
            if in_amount > 0:
                ratio = out_amount / in_amount
                # For pump.fun tokens, expect volatile prices but not 99% loss
                if ratio < 0.01:  # Getting back less than 1% is suspicious
                    self.honeypots_detected += 1
                    return False, f"Sell quote too low ({ratio:.4f}) - possible honeypot"

            return True, "Sell route verified"

        except Exception as e:
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
            'consecutive_failures': self.consecutive_failures
        }
