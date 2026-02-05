"""
Solana Arbitrage Engine

Features:
- Jupiter Aggregator integration (best routing across all Solana DEXs)
- Raydium AMM direct integration
- Orca Whirlpools integration
- Multi-token monitoring
- Jito MEV protection (Solana's Flashbots equivalent)
"""
import asyncio
import logging
import os
import json
import aiohttp
import base58
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger("SolanaArbitrageEngine")

# Solana Token Addresses (Mint addresses)
SOLANA_TOKENS = {
    # Major tokens
    'SOL': 'So11111111111111111111111111111111111111112',  # Wrapped SOL
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',

    # Popular DeFi
    'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',  # Raydium
    'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',  # Serum
    'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',  # Orca
    'MNGO': 'MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac',  # Mango

    # Meme tokens (high volatility = more arb opportunities)
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',  # Dogwifhat
    'POPCAT': '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',

    # Stablecoins
    'USDH': 'USDH1SM1ojwWUga67PGrgFWUHibbjqMvuMaDkRJTgkX',
    'USH': '9iLH8T7zoWhY7sBmj1WK9ENbWdS1nL8n9wAxaeRitTa6',

    # Liquid Staking
    'mSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # Marinade
    'stSOL': '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',  # Lido
    'jitoSOL': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',  # Jito
    'bSOL': 'bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1',  # BlazeStake

    # Jupiter ecosystem
    'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    'JTO': 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL',  # Jito

    # Other popular
    'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
    'W': '85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ',  # Wormhole
    'RENDER': 'rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof',
}

# Solana DEX pairs to monitor
SOLANA_ARB_PAIRS = [
    ('SOL', 'USDC'),
    ('SOL', 'USDT'),
    ('mSOL', 'SOL'),
    ('stSOL', 'SOL'),
    ('jitoSOL', 'SOL'),
    ('bSOL', 'SOL'),
    ('RAY', 'SOL'),
    ('ORCA', 'SOL'),
    ('JUP', 'SOL'),
    ('JTO', 'SOL'),
    ('BONK', 'SOL'),
    ('WIF', 'SOL'),
    ('PYTH', 'SOL'),
    ('W', 'SOL'),
    ('USDC', 'USDT'),
    ('mSOL', 'USDC'),
]

# DEX API endpoints
# Updated to use lite-api.jup.ag/swap/v1 (same as DEX module jupiter_executor.py)
JUPITER_API = "https://lite-api.jup.ag/swap/v1"
RAYDIUM_API = "https://api.raydium.io/v2"


class RateLimiter:
    """
    Simple rate limiter - DEPRECATED for Jupiter API.
    Use SharedJupiterRateLimiter from config.jupiter_rate_limiter instead.
    Kept for backwards compatibility with non-Jupiter APIs.
    """

    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._consecutive_429s = 0

    async def acquire(self):
        async with self._lock:
            now = datetime.now()
            if self.last_request_time:
                elapsed = (now - self.last_request_time).total_seconds()
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = datetime.now()

    def record_success(self) -> None:
        """Record a successful request and reset backoff counter."""
        self._consecutive_429s = 0

    def record_429(self) -> float:
        """Record a 429 error and return backoff time."""
        self._consecutive_429s += 1
        return min(2.0 * (2 ** (self._consecutive_429s - 1)), 60.0)


class JupiterClient:
    """Jupiter Aggregator API client with SHARED global rate limiting"""

    def __init__(self, rate_limit: float = 2.0):
        self.base_url = JUPITER_API
        self.session: Optional[aiohttp.ClientSession] = None

        # Use SHARED global rate limiter to coordinate with other modules
        # This prevents combined module RPS from exceeding Jupiter's 1 RPS limit
        try:
            from config.jupiter_rate_limiter import SharedJupiterRateLimiter
            self.rate_limiter = SharedJupiterRateLimiter(caller="SolanaArbitrage")
            logger.info("   🚦 Using shared global Jupiter rate limiter")
        except ImportError:
            # Fallback to local rate limiter (not recommended)
            logger.warning("   ⚠️ Global rate limiter unavailable, using local limiter")
            self.rate_limiter = RateLimiter(rate_limit)

        self._consecutive_errors = 0
        self._backoff_until: Optional[datetime] = None

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 150  # Increased from 50 to 150 bps (1.5%) for arbitrage
    ) -> Optional[Dict]:
        """
        Get quote from Jupiter aggregator.
        Jupiter finds the best route across all Solana DEXs.
        """
        # Check if we're in backoff period
        if self._backoff_until:
            if datetime.now() < self._backoff_until:
                remaining = (self._backoff_until - datetime.now()).seconds
                logger.debug(f"Jupiter in backoff period, {remaining}s remaining")
                return None
            self._backoff_until = None
            logger.info("✅ Jupiter backoff period ended")

        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': slippage_bps,
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }

            async with self.session.get(f"{self.base_url}/quote", params=params) as resp:
                if resp.status == 200:
                    self._consecutive_errors = 0
                    self.rate_limiter.record_success()
                    data = await resp.json()
                    # Check for error in response
                    if 'error' in data:
                        logger.warning(f"Jupiter API error: {data.get('error')}")
                        return None
                    # Add timestamp for quote freshness validation
                    data['_quote_timestamp'] = datetime.now().timestamp()
                    return data
                elif resp.status == 429:
                    # Rate limited - notify global rate limiter for coordinated backoff
                    self._consecutive_errors += 1
                    # Use the shared rate limiter's backoff mechanism
                    backoff_seconds = self.rate_limiter.record_429()
                    self._backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
                    logger.warning(f"⚠️ Jupiter rate limited (429) - backing off {backoff_seconds:.0f}s")
                    return None
                elif resp.status == 400:
                    # Bad request - likely invalid token pair or amount
                    error_text = await resp.text()
                    logger.warning(f"Jupiter bad request (400): {error_text[:200]}")
                    return None
                else:
                    error_text = await resp.text()
                    logger.warning(f"Jupiter quote error: HTTP {resp.status} - {error_text[:100]}")
                    return None

        except asyncio.TimeoutError:
            logger.warning("Jupiter quote timeout (10s) - API may be slow or unreachable")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Jupiter connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Jupiter quote unexpected error: {e}")
            return None

    async def get_swap_transaction(self, quote: Dict, user_public_key: str) -> Optional[Dict]:
        """Get swap transaction from Jupiter"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': user_public_key,
                'wrapAndUnwrapSol': True,
                'useSharedAccounts': True,
                'prioritizationFeeLamports': 'auto'
            }

            async with self.session.post(
                f"{self.base_url}/swap",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None

        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
            return None


class RaydiumClient:
    """Raydium AMM API client for direct pool queries"""

    def __init__(self):
        self.base_url = RAYDIUM_API
        self.session: Optional[aiohttp.ClientSession] = None
        self.pools_cache: Dict = {}
        self.cache_time: Optional[datetime] = None

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_pool_info(self, pool_id: str) -> Optional[Dict]:
        """Get pool info from Raydium"""
        try:
            async with self.session.get(f"{self.base_url}/ammV3/ammPools") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Find specific pool
                    for pool in data.get('data', []):
                        if pool.get('id') == pool_id:
                            return pool
                return None
        except Exception as e:
            logger.debug(f"Raydium pool error: {e}")
            return None

    async def get_price(self, base_mint: str, quote_mint: str) -> Optional[float]:
        """Get price from Raydium pools"""
        try:
            async with self.session.get(f"{self.base_url}/main/price") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    prices = data.get('data', {})
                    # Check if we have the price
                    if base_mint in prices:
                        return float(prices[base_mint])
                return None
        except Exception as e:
            logger.debug(f"Raydium price error: {e}")
            return None


class JitoClient:
    """Jito MEV protection client (Solana's Flashbots)"""

    # Regional Jito block engines for failover (from official docs)
    # https://docs.jito.wtf/lowlatencytxnsend/
    JITO_ENDPOINTS = [
        "https://mainnet.block-engine.jito.wtf",
        "https://amsterdam.mainnet.block-engine.jito.wtf",
        "https://dublin.mainnet.block-engine.jito.wtf",
        "https://frankfurt.mainnet.block-engine.jito.wtf",
        "https://london.mainnet.block-engine.jito.wtf",
        "https://ny.mainnet.block-engine.jito.wtf",
        "https://slc.mainnet.block-engine.jito.wtf",
        "https://singapore.mainnet.block-engine.jito.wtf",
        "https://tokyo.mainnet.block-engine.jito.wtf",
    ]

    # Official Jito tip accounts (from getTipAccounts API response)
    # https://docs.jito.wtf/lowlatencytxnsend/
    JITO_TIP_ACCOUNTS = [
        "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
        "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
        "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
        "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
        "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
        "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    ]

    # Global rate limit - Jito has strict limits across ALL endpoints
    # These are NOT independent - hitting one affects all
    # Free tier is approximately 1 bundle per 10-15 seconds
    _global_last_request_time = 0.0
    _global_backoff_until = 0.0
    _global_consecutive_429s = 0
    MIN_REQUEST_INTERVAL = 12.0  # Minimum 12 seconds between bundle requests (was 5s)

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.keypair = None  # Set during initialization
        # Use env var if set, otherwise use default
        self.primary_endpoint = os.getenv('JITO_BLOCK_ENGINE_URL', self.JITO_ENDPOINTS[0])
        self.current_endpoint_idx = 0
        self._last_429_time = 0
        self._backoff_seconds = 0

    @classmethod
    def is_available(cls) -> Tuple[bool, float]:
        """
        Check if Jito is available for bundle submission.

        Returns:
            Tuple of (is_available, seconds_until_available)
        """
        import time
        now = time.time()

        # Check global backoff period
        if cls._global_backoff_until > now:
            remaining = cls._global_backoff_until - now
            return False, remaining

        # Check minimum interval since last request
        time_since_last = now - cls._global_last_request_time
        if time_since_last < cls.MIN_REQUEST_INTERVAL:
            wait_time = cls.MIN_REQUEST_INTERVAL - time_since_last
            return False, wait_time

        return True, 0.0

    async def initialize(self, keypair=None):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self.keypair = keypair
        logger.info(f"   Jito endpoint: {self.primary_endpoint}")
        logger.info(f"   Jito rate limit: {self.MIN_REQUEST_INTERVAL}s between bundles")

    async def close(self):
        if self.session:
            await self.session.close()

    def _get_next_endpoint(self) -> str:
        """Rotate through Jito endpoints for load balancing"""
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.JITO_ENDPOINTS)
        return self.JITO_ENDPOINTS[self.current_endpoint_idx]

    def _handle_rate_limit(self, source: str):
        """
        Handle rate limit by setting global backoff.

        Jito's rate limits are GLOBAL across all endpoints, so we need to
        back off from ALL endpoints when any one returns 429.
        """
        import time

        JitoClient._global_consecutive_429s += 1

        # Exponential backoff: 10s, 20s, 40s, 60s (max)
        backoff_seconds = min(10 * (2 ** (JitoClient._global_consecutive_429s - 1)), 60)
        JitoClient._global_backoff_until = time.time() + backoff_seconds

        logger.warning(f"⚠️ Jito rate limited ({source}): "
                      f"429 #{JitoClient._global_consecutive_429s}, "
                      f"global backoff {backoff_seconds}s")

    async def send_bundle(self, transactions: List[str], tip_lamports: int = 10000) -> Optional[str]:
        """
        Send transaction bundle to Jito block engine with retry and failover.
        Protects against sandwich attacks and ensures atomic execution.

        Args:
            transactions: List of base64-encoded transactions (must be signed!)
            tip_lamports: Tip amount for Jito validators (default 10000 = 0.00001 SOL)

        Returns:
            Bundle ID if successful, None otherwise
        """
        import time

        now = time.time()

        # GLOBAL RATE LIMIT CHECK - Jito limits are SHARED across ALL endpoints
        # Check if we're in global backoff period
        if JitoClient._global_backoff_until > now:
            remaining = JitoClient._global_backoff_until - now
            logger.warning(f"⏳ Jito GLOBAL backoff: {remaining:.1f}s remaining (after {JitoClient._global_consecutive_429s} 429s)")
            return None

        # Enforce minimum interval between requests
        time_since_last = now - JitoClient._global_last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            wait_time = self.MIN_REQUEST_INTERVAL - time_since_last
            logger.info(f"⏳ Jito rate limit: waiting {wait_time:.1f}s before next bundle")
            await asyncio.sleep(wait_time)
            now = time.time()

        JitoClient._global_last_request_time = now

        # CRITICAL: Jito sendBundle requires encoding specification
        # Without this, Jito may fail to decode base64 transactions
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [
                transactions,
                {"encoding": "base64"}  # Explicit encoding for base64 transactions
            ]
        }

        # Try primary endpoint first, then ONE failover only
        # IMPORTANT: Jito endpoints share rate limits globally, so trying many endpoints
        # when rate limited just burns through all of them and extends the cooldown
        endpoints_to_try = [self.primary_endpoint, self._get_next_endpoint()]

        for i, endpoint in enumerate(endpoints_to_try):
            try:
                async with self.session.post(
                    f"{endpoint}/api/v1/bundles",
                    json=payload
                ) as resp:
                    response_text = await resp.text()

                    if resp.status == 200:
                        result = await resp.json()
                        if 'result' in result:
                            bundle_id = result.get('result')
                            logger.info(f"🛡️ Jito bundle submitted: {bundle_id}")

                            # Reset global backoff on success
                            JitoClient._global_consecutive_429s = 0
                            JitoClient._global_backoff_until = 0

                            # CRITICAL: Verify bundle was actually landed on-chain
                            # Bundle ID != transaction success - must check status
                            confirmed = await self._confirm_bundle(bundle_id, endpoint)
                            if confirmed:
                                logger.info(f"✅ Jito bundle CONFIRMED on-chain: {bundle_id}")
                                return bundle_id
                            else:
                                logger.warning(f"⚠️ Jito bundle NOT confirmed: {bundle_id} - bundle may have been dropped")
                                return None

                        elif 'error' in result:
                            error = result.get('error', {})
                            error_code = error.get('code', 'N/A')
                            error_msg = error.get('message', 'Unknown error')

                            # Check for rate limit error in response body
                            if error_code == -32097 or 'rate limit' in error_msg.lower():
                                self._handle_rate_limit("API error")
                                if i < len(endpoints_to_try) - 1:
                                    await asyncio.sleep(5)
                                    continue
                                return None

                            logger.error(f"❌ Jito bundle error: {error_msg}")
                            logger.error(f"   Error code: {error_code}")
                            return None

                    elif resp.status == 429:
                        # GLOBAL rate limit - ALL Jito endpoints share the same limit!
                        self._handle_rate_limit(f"HTTP 429 on {endpoint[:30]}")
                        # Don't try next endpoint immediately - they ALL share the limit
                        # Only try failover if this wasn't a repeated 429
                        if JitoClient._global_consecutive_429s <= 2 and i < len(endpoints_to_try) - 1:
                            await asyncio.sleep(5)  # Wait 5s before trying alternate
                            continue
                        return None

                    else:
                        # Log detailed error for debugging
                        try:
                            error_body = await resp.text()
                            logger.error(f"❌ Jito HTTP {resp.status} on {endpoint[:30]}...")
                            logger.error(f"   Response: {error_body[:500]}")
                        except Exception:
                            logger.error(f"❌ Jito HTTP {resp.status} on {endpoint[:30]}...")
                        continue  # Try next endpoint

            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Jito timeout on {endpoint[:30]}... trying next")
                continue
            except Exception as e:
                logger.warning(f"Jito error on {endpoint[:30]}...: {e}")
                continue

        logger.error("❌ All Jito endpoints failed or rate limited")
        return None

    async def _confirm_bundle(self, bundle_id: str, endpoint: str, timeout_seconds: int = 30) -> bool:
        """
        Confirm that a Jito bundle was actually landed on-chain.

        Jito's getBundleStatuses API returns the status of submitted bundles.
        A bundle is only successful if it shows 'Landed' status.

        Args:
            bundle_id: The bundle ID returned from sendBundle
            endpoint: The Jito endpoint to check
            timeout_seconds: How long to wait for confirmation

        Returns:
            True if bundle was confirmed landed, False otherwise
        """
        import time
        start_time = time.time()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBundleStatuses",
            "params": [[bundle_id]]
        }

        # Poll for bundle status with exponential backoff
        check_interval = 1.0  # Start with 1 second

        while (time.time() - start_time) < timeout_seconds:
            try:
                async with self.session.post(
                    f"{endpoint}/api/v1/bundles",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()

                        if 'result' in result and 'value' in result['result']:
                            statuses = result['result']['value']

                            if statuses and len(statuses) > 0:
                                bundle_status = statuses[0]

                                if bundle_status is None:
                                    # Bundle not found yet - keep waiting
                                    logger.debug(f"Bundle {bundle_id[:16]}... not found yet, waiting...")
                                else:
                                    confirmation_status = bundle_status.get('confirmation_status')
                                    err = bundle_status.get('err')

                                    if err:
                                        # Bundle failed with error
                                        logger.error(f"❌ Jito bundle failed: {err}")
                                        return False

                                    if confirmation_status == 'finalized' or confirmation_status == 'confirmed':
                                        # Bundle landed successfully!
                                        slot = bundle_status.get('slot', 'unknown')
                                        logger.info(f"   Bundle landed in slot {slot}")
                                        return True
                                    elif confirmation_status == 'processed':
                                        # Still processing - keep waiting
                                        logger.debug(f"Bundle {bundle_id[:16]}... processing...")
                                    else:
                                        logger.debug(f"Bundle status: {confirmation_status}")

            except asyncio.TimeoutError:
                logger.debug(f"Bundle status check timeout, retrying...")
            except Exception as e:
                logger.debug(f"Bundle status check error: {e}")

            # Wait before next check with exponential backoff (max 5 seconds)
            await asyncio.sleep(check_interval)
            check_interval = min(check_interval * 1.5, 5.0)

        # Timeout reached without confirmation
        logger.warning(f"⚠️ Bundle confirmation timeout after {timeout_seconds}s - bundle may have been dropped")
        return False

    def get_random_tip_account(self) -> str:
        """Get a random Jito tip account (or from env)"""
        import random
        env_tip = os.getenv('JITO_TIP_ACCOUNT')
        if env_tip:
            return env_tip
        return random.choice(self.JITO_TIP_ACCOUNTS)

    async def create_tip_transaction(
        self,
        payer_keypair,
        tip_lamports: int = 10000,
        recent_blockhash: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a signed tip transaction for Jito bundle.

        A tip is REQUIRED for bundles to be considered by Jito validators.
        The tip should be the LAST transaction in the bundle.

        Args:
            payer_keypair: Solders Keypair object for signing
            tip_lamports: Tip amount in lamports (min 1000, default 10000)
            recent_blockhash: Recent blockhash (will fetch if not provided)

        Returns:
            Base64-encoded signed transaction, or None on failure
        """
        try:
            import base64
            from solders.keypair import Keypair
            from solders.pubkey import Pubkey
            from solders.system_program import TransferParams, transfer
            from solders.message import Message
            from solders.transaction import Transaction
            from solders.hash import Hash

            if not payer_keypair:
                logger.error("No keypair provided for tip transaction")
                return None

            # Ensure minimum tip (1000 lamports per Jito docs)
            tip_lamports = max(tip_lamports, 1000)

            # Get random tip account
            tip_account_str = self.get_random_tip_account()
            tip_account = Pubkey.from_string(tip_account_str)
            payer_pubkey = payer_keypair.pubkey()

            logger.debug(f"Creating tip tx: {tip_lamports} lamports to {tip_account_str[:12]}...")

            # Fetch recent blockhash if not provided
            if not recent_blockhash:
                import aiohttp
                # Get RPC URL from environment
                rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
                try:
                    from config.rpc_provider import RPCProvider
                    rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC') or rpc_url
                except Exception:
                    pass

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "getLatestBlockhash",
                            "params": [{"commitment": "finalized"}]
                        }
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            recent_blockhash = data.get('result', {}).get('value', {}).get('blockhash')

            if not recent_blockhash:
                logger.error("Failed to get recent blockhash for tip transaction")
                return None

            # Create transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=payer_pubkey,
                    to_pubkey=tip_account,
                    lamports=tip_lamports
                )
            )

            # Create and sign transaction
            blockhash = Hash.from_string(recent_blockhash)
            message = Message.new_with_blockhash(
                [transfer_ix],
                payer_pubkey,
                blockhash
            )
            tx = Transaction.new_unsigned(message)
            tx.sign([payer_keypair], blockhash)

            # Encode to base64
            tx_bytes = bytes(tx)
            tx_b64 = base64.b64encode(tx_bytes).decode('utf-8')

            logger.info(f"✅ Tip transaction created: {tip_lamports} lamports to {tip_account_str[:12]}...")
            return tx_b64

        except Exception as e:
            logger.error(f"Failed to create tip transaction: {e}")
            return None


class SolanaPriceFetcher:
    """Fetch real-time prices from multiple sources"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"
    # Updated to use api.jup.ag/price/v3 (latest Jupiter API)
    JUPITER_PRICE_API = "https://api.jup.ag/price/v3"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 30  # 30 seconds cache

    async def get_price(self, symbol: str) -> float:
        """Get USD price for a token"""
        now = datetime.now()

        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (now - cached_at).total_seconds() < self._cache_ttl:
                return price

        # Try Jupiter Price API first (faster, Solana-native)
        try:
            mint = SOLANA_TOKENS.get(symbol.upper())
            if mint:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.JUPITER_PRICE_API}",
                        params={'ids': mint}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if 'data' in data and mint in data['data']:
                                price = float(data['data'][mint].get('price', 0))
                                if price > 0:
                                    self._cache[symbol] = (price, now)
                                    return price
        except Exception:
            pass

        # Fallback to CoinGecko
        try:
            symbol_map = {
                'sol': 'solana',
                'msol': 'marinade-staked-sol',
                'jitosol': 'jito-staked-sol',
                'ray': 'raydium',
                'orca': 'orca',
                'jup': 'jupiter-exchange-solana',
                'bonk': 'bonk',
                'wif': 'dogwifcoin',
            }
            coin_id = symbol_map.get(symbol.lower(), symbol.lower())

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.COINGECKO_API,
                    params={'ids': coin_id, 'vs_currencies': 'usd'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if coin_id in data and 'usd' in data[coin_id]:
                            price = float(data[coin_id]['usd'])
                            self._cache[symbol] = (price, now)
                            return price
        except Exception:
            pass

        # Return cached or default
        if symbol in self._cache:
            return self._cache[symbol][0]

        # SOL default
        return 100.0 if symbol.upper() == 'SOL' else 0.0


class SolanaArbitrageEngine:
    """
    Solana Arbitrage Engine with Jupiter, Raydium, and Jito integration.

    Strategy:
    1. Get quotes from Jupiter (aggregates all DEXs)
    2. Compare with direct Raydium quotes
    3. If spread > threshold, execute via Jito for MEV protection
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False

        # Solana RPC - use Pool Engine with fallback
        self.rpc_url = config.get('rpc_url')
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                self.rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC')
            except Exception:
                pass
        if not self.rpc_url:
            self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')

        self.private_key = None  # Loaded in initialize() from secrets manager
        self.wallet_address = config.get('wallet_address')  # Will be loaded from DB in initialize()

        # dry_run: Priority is database config > environment variable
        # This allows dashboard settings to override .env
        db_dry_run = config.get('dry_run')
        if db_dry_run is not None:
            self.dry_run = db_dry_run if isinstance(db_dry_run, bool) else str(db_dry_run).lower() in ('true', '1', 'yes')
        else:
            self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Settings from config (loaded from DB via settings page)
        # Threshold is stored as percentage (0.2 = 0.2%), convert to decimal (0.002)
        self.min_profit_threshold = config.get('sol_arb_threshold', 0.2) / 100.0
        self.trade_amount_sol = config.get('sol_trade_amount', 1.0)
        self.use_jito = config.get('use_jito', True)

        # Arbitrage slippage - higher than normal trading to account for price movement
        # during signing and bundle submission (default 150 bps = 1.5%)
        self.arb_slippage_bps = config.get('sol_arb_slippage_bps', 150)

        # Verbose logging (configurable via settings page)
        self.verbose_logging = config.get('sol_arb_verbose', False)

        # Rate limiting - cooldown from config
        self._last_opportunity_time: Dict[str, datetime] = {}
        self._opportunity_cooldown = config.get('sol_arb_cooldown', 300)  # 5 min default

        # Jupiter rate limiter from config
        jupiter_rps = config.get('jupiter_rps', 2.0)

        # Clients - pass rate limit to Jupiter
        self.jupiter = JupiterClient(rate_limit=float(jupiter_rps))
        self.raydium = RaydiumClient()
        self.jito = JitoClient()
        self.price_fetcher = SolanaPriceFetcher()
        self._pair_execution_count: Dict[str, int] = {}
        self._pair_execution_date: str = ""
        self._max_executions_per_pair_per_day = 10

        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }

    async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
        """
        Get decrypted private key from secrets manager or environment.

        Always checks if value is still encrypted and decrypts if needed.
        """
        try:
            value = None

            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                # Always re-initialize with db_pool if secrets was in bootstrap mode
                # or doesn't have a db_pool yet
                if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                    secrets.initialize(self.db_pool)
                    logger.debug(f"Re-initialized secrets manager with database pool for {key_name}")
                value = await secrets.get_async(key_name)
                if value:
                    logger.debug(f"Successfully loaded {key_name} from secrets manager")
            except Exception as e:
                logger.warning(f"Failed to get {key_name} from secrets manager: {e}")

            # Fallback to environment
            if not value:
                value = os.getenv(key_name)

            if not value:
                return None

            # Check if still encrypted (Fernet tokens start with gAAAAAB)
            if value.startswith('gAAAAAB'):
                from pathlib import Path
                encryption_key = None
                key_file = Path('.encryption_key')
                if key_file.exists():
                    encryption_key = key_file.read_text().strip()
                if not encryption_key:
                    encryption_key = os.getenv('ENCRYPTION_KEY')

                if encryption_key:
                    try:
                        from cryptography.fernet import Fernet
                        f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                        return f.decrypt(value.encode()).decode()
                    except Exception as e:
                        logger.error(f"Failed to decrypt {key_name}: {e}")
                        return None
                else:
                    logger.error(f"Cannot decrypt {key_name}: no encryption key found")
                    return None

            return value
        except Exception as e:
            logger.debug(f"Error getting {key_name}: {e}")
            return None

    async def initialize(self):
        logger.info("🌊 Initializing Solana Arbitrage Engine...")

        # Load private key from secrets manager - try multiple key names
        # Priority order: SOLANA_PRIVATE_KEY (DEX) > SOLANA_MODULE_PRIVATE_KEY (strategies)
        private_key_names = [
            'SOLANA_PRIVATE_KEY',         # DEX trading private key
            'SOLANA_MODULE_PRIVATE_KEY'   # Solana strategies private key
        ]
        for key_name in private_key_names:
            self.private_key = await self._get_decrypted_key(key_name)
            if self.private_key:
                logger.debug(f"Loaded private key from {key_name}")
                break

        # CRITICAL: Derive wallet address from private key to ensure they always match
        # This prevents signature verification failures from mismatched wallet/key pairs
        if self.private_key:
            import base58
            import json as json_module
            from solders.keypair import Keypair

            pk = self.private_key
            key_bytes = None

            # Format 1: JSON array (e.g., [1,2,3,...])
            if pk.startswith('['):
                try:
                    key_array = json_module.loads(pk)
                    key_bytes = bytes(key_array)
                except Exception:
                    pass

            # Format 2: Base58 encoded (most common)
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(pk)
                except Exception:
                    pass

            # Format 3: Hex encoded
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(pk)
                except Exception:
                    pass

            if key_bytes:
                if len(key_bytes) == 64:
                    keypair = Keypair.from_bytes(key_bytes)
                elif len(key_bytes) == 32:
                    keypair = Keypair.from_seed(key_bytes)
                else:
                    keypair = None
                    logger.error(f"Invalid private key length: {len(key_bytes)} bytes")

                if keypair:
                    self.wallet_address = str(keypair.pubkey())
                    logger.info(f"✅ Derived wallet from private key: {self.wallet_address[:8]}...{self.wallet_address[-8:]}")
            else:
                logger.error("Failed to parse private key for wallet derivation")

        await self.jupiter.initialize()
        await self.raydium.initialize()

        if self.private_key and not self.dry_run:
            await self.jito.initialize()
            logger.info("🛡️ Jito MEV protection enabled")

        # Log wallet configuration status
        if not self.wallet_address:
            logger.warning("⚠️ No Solana wallet address - check SOLANA_PRIVATE_KEY in database")

        logger.info(f"   RPC: {self.rpc_url[:40]}...")
        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Jito: {'Enabled' if self.use_jito and self.private_key else 'Disabled'}")
        logger.info(f"   Monitoring {len(SOLANA_ARB_PAIRS)} token pairs")
        logger.info(f"   Threshold: {self.min_profit_threshold:.2%} | Cooldown: {self._opportunity_cooldown}s")
        logger.info(f"   Verbose: {self.verbose_logging}")

    async def run(self):
        self.is_running = True
        logger.info("🌊 Solana Arbitrage Engine Started")

        pair_index = 0

        while self.is_running:
            try:
                self._stats['scans'] += 1

                # Get current pair
                token_in_symbol, token_out_symbol = SOLANA_ARB_PAIRS[pair_index]
                token_in = SOLANA_TOKENS.get(token_in_symbol)
                token_out = SOLANA_TOKENS.get(token_out_symbol)

                if token_in and token_out:
                    await self._check_arb_opportunity(
                        token_in, token_out,
                        token_in_symbol, token_out_symbol
                    )

                # Move to next pair
                pair_index = (pair_index + 1) % len(SOLANA_ARB_PAIRS)

                # Log stats
                await self._log_stats_if_needed()

                # RATE LIMIT AWARE SCANNING
                # Each pair check makes 2 Jupiter API calls (forward + reverse quote)
                # Jupiter free tier is ~1 RPS, so we need 2+ seconds per pair minimum
                # Use 2.5 seconds to stay safely under the limit
                await asyncio.sleep(2.5)

            except Exception as e:
                logger.error(f"Solana arb loop error: {e}")
                await asyncio.sleep(5)

    async def _log_stats_if_needed(self):
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:
            logger.info(f"📊 SOLANA ARB STATS (Last 5 min): "
                       f"Scans: {self._stats['scans']} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']}")

            self._stats = {
                'scans': 0,
                'opportunities_found': 0,
                'opportunities_executed': 0,
                'last_stats_log': now
            }

    async def _check_arb_opportunity(
        self,
        token_in: str,
        token_out: str,
        in_symbol: str,
        out_symbol: str
    ) -> bool:
        """Check for arbitrage between Jupiter aggregated route and direct DEX"""
        try:
            # Convert trade amount to lamports (SOL decimals = 9)
            if in_symbol == 'SOL':
                amount = int(self.trade_amount_sol * 1e9)
            elif in_symbol in ['USDC', 'USDT']:
                amount = int(self.trade_amount_sol * 100 * 1e6)  # Assuming ~$100 worth
            else:
                # For other tokens, use a standard amount
                amount = int(1e9)

            # Get Jupiter quote (best aggregated route)
            # Use higher slippage for arbitrage to account for price movement during execution
            jupiter_quote = await self.jupiter.get_quote(
                token_in, token_out, amount, slippage_bps=self.arb_slippage_bps
            )

            if not jupiter_quote:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: No forward quote from Jupiter")
                return False

            jupiter_out = int(jupiter_quote.get('outAmount', 0))
            if jupiter_out == 0:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: Jupiter returned 0 output")
                return False

            # Get reverse route quote to check round-trip profitability
            # Use higher slippage for arbitrage
            reverse_quote = await self.jupiter.get_quote(
                token_out, token_in, jupiter_out, slippage_bps=self.arb_slippage_bps
            )

            if not reverse_quote:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: No reverse quote from Jupiter")
                return False

            reverse_out = int(reverse_quote.get('outAmount', 0))

            # Calculate round-trip profit
            profit_pct = (reverse_out - amount) / amount

            # Log price check result (always log if verbose, or log good spreads)
            if self.verbose_logging or profit_pct > 0.0005:  # Log if > 0.05%
                route_info = jupiter_quote.get('routePlan', [])
                route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route_info[:3]])
                status = "✅ OPPORTUNITY" if profit_pct > self.min_profit_threshold else "📊 Below threshold"
                logger.info(f"🌊 {status} [{in_symbol}/{out_symbol}]: Spread: {profit_pct:.3%} | Threshold: {self.min_profit_threshold:.3%} | Route: {route_str}")

            if profit_pct > self.min_profit_threshold:
                self._stats['opportunities_found'] += 1

                # SANITY CHECK: Reject obvious false positives
                # Real arbitrage opportunities are typically 0.1-1%. Anything > 5% is almost
                # certainly stale/manipulated price data from low-liquidity DEXs.
                MAX_REALISTIC_PROFIT = 0.05  # 5% max realistic profit
                if profit_pct > MAX_REALISTIC_PROFIT:
                    route_info = jupiter_quote.get('routePlan', [])
                    route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route_info[:3]])
                    logger.warning(f"🚫 REJECTED [{in_symbol}/{out_symbol}]: {profit_pct:.2%} profit is unrealistic (max {MAX_REALISTIC_PROFIT:.0%})")
                    logger.warning(f"   Route: {route_str} - likely stale/manipulated price data")
                    return False

                # Note: We no longer skip opportunities when Jito is unavailable
                # Instead, we fall back to direct RPC execution in _execute_arbitrage

                # Check rate limiting
                opp_key = f"{in_symbol}_{out_symbol}"
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Reset daily counters
                if self._pair_execution_date != today:
                    self._pair_execution_count = {}
                    self._pair_execution_date = today

                # Check limits
                current_count = self._pair_execution_count.get(opp_key, 0)
                if current_count >= self._max_executions_per_pair_per_day:
                    return True

                # Check cooldown
                last_time = self._last_opportunity_time.get(opp_key)
                if last_time:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        return True

                # Log and execute
                self._last_opportunity_time[opp_key] = now
                self._pair_execution_count[opp_key] = current_count + 1

                route_info = jupiter_quote.get('routePlan', [])
                route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route_info[:3]])

                logger.info(f"🚨 SOLANA ARB [{in_symbol}/{out_symbol}]: {profit_pct:.2%} profit | Route: {route_str}")
                self._stats['opportunities_executed'] += 1

                await self._execute_arbitrage(
                    jupiter_quote, reverse_quote,
                    token_in, token_out,
                    in_symbol, out_symbol,
                    amount, profit_pct
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Solana arb check error [{in_symbol}/{out_symbol}]: {e}")
            return False

    def _is_quote_stale(self, quote: Dict, max_age_seconds: float = 10.0) -> bool:
        """
        Check if a quote is too old to execute safely.

        Args:
            quote: Quote dict with '_quote_timestamp' timestamp
            max_age_seconds: Maximum age in seconds (default: 10s for arbitrage)

        Returns:
            True if quote is stale, False if still fresh
        """
        import time
        # Check for the timestamp key (set in get_quote)
        timestamp = quote.get('_quote_timestamp', 0)
        if timestamp == 0:
            return True  # No timestamp means we can't verify freshness
        age = time.time() - timestamp
        return age > max_age_seconds

    async def _execute_arbitrage(
        self,
        quote1: Dict,
        quote2: Dict,
        token_in: str,
        token_out: str,
        in_symbol: str,
        out_symbol: str,
        amount: int,
        profit_pct: float
    ):
        """Execute arbitrage trade"""
        import time

        logger.info(f"⚡ Executing Solana Arbitrage [{in_symbol}/{out_symbol}] | Expected: +{profit_pct:.2%}")

        # QUOTE FRESHNESS VALIDATION
        # Jupiter quotes are valid for ~10-30 seconds, using 10s as safe threshold
        # Stale quotes cause 0x1789 slippage errors
        max_quote_age = 10.0  # 10 seconds max for arbitrage
        if self._is_quote_stale(quote1, max_quote_age) or self._is_quote_stale(quote2, max_quote_age):
            quote1_age = time.time() - quote1.get('_quote_timestamp', 0)
            quote2_age = time.time() - quote2.get('_quote_timestamp', 0)
            logger.warning(f"⚠️ Quotes are stale (age: {quote1_age:.1f}s, {quote2_age:.1f}s) - refreshing before execution")

            # Refresh quotes with proper slippage
            quote1 = await self.jupiter.get_quote(
                token_in, token_out, amount, slippage_bps=self.arb_slippage_bps
            )
            if not quote1:
                logger.error("❌ Failed to refresh forward quote - aborting arbitrage")
                return

            jupiter_out = int(quote1.get('outAmount', 0))
            if jupiter_out == 0:
                logger.error("❌ Fresh forward quote has 0 output - aborting arbitrage")
                return

            quote2 = await self.jupiter.get_quote(
                token_out, token_in, jupiter_out, slippage_bps=self.arb_slippage_bps
            )
            if not quote2:
                logger.error("❌ Failed to refresh reverse quote - aborting arbitrage")
                return

            # Recalculate profit with fresh quotes
            reverse_out = int(quote2.get('outAmount', 0))
            new_profit_pct = (reverse_out - amount) / amount if amount > 0 else 0

            # Verify opportunity still exists
            if new_profit_pct < self.min_profit_threshold:
                logger.warning(f"⚠️ Opportunity disappeared after refresh: {new_profit_pct:.3%} < {self.min_profit_threshold:.3%}")
                return

            profit_pct = new_profit_pct
            logger.info(f"✅ Quotes refreshed, profit: {profit_pct:.3%}")

        if self.dry_run:
            await asyncio.sleep(0.3)
            logger.info(f"✅ Solana Arb Executed (DRY RUN) [{in_symbol}/{out_symbol}]")
            await self._log_trade(
                in_symbol, out_symbol, token_in,
                amount, profit_pct, "DRY_RUN",
                quote1.get('routePlan', [])
            )
            return

        try:
            if not self.wallet_address:
                logger.error("No Solana wallet configured")
                return

            if not self.private_key:
                logger.error("No Solana private key configured - cannot sign transactions")
                return

            # Check SOL balance before execution
            # "Attempt to debit an account" error means insufficient SOL for tx fees + tip
            # Minimum required: 0.01 SOL (covers ~2 tx signatures + Jito tip + buffer)
            min_sol_required = 0.01
            sol_balance = await self._get_sol_balance()
            if sol_balance < min_sol_required:
                logger.warning(f"⏸️ Skipping Solana arbitrage - insufficient SOL for transaction fees")
                logger.warning(f"   Balance: {sol_balance:.6f} SOL | Required: {min_sol_required:.4f} SOL")
                return

            # Get swap transactions with increased slippage to handle 0x1789 errors
            # Note: Slippage is set in get_quote (150 bps = 1.5%)
            swap1 = await self.jupiter.get_swap_transaction(quote1, self.wallet_address)
            swap2 = await self.jupiter.get_swap_transaction(quote2, self.wallet_address)

            if not swap1 or not swap2:
                logger.error("Failed to get swap transactions")
                return

            # Check Jito availability BEFORE building bundle
            # If Jito is rate limited, skip directly to RPC to save time
            jito_available, jito_wait = JitoClient.is_available() if self.use_jito else (False, 0)
            use_jito_this_time = self.use_jito and self.jito.session and jito_available

            if self.use_jito and not jito_available:
                logger.info(f"⏩ Jito rate limited ({jito_wait:.0f}s wait) - using RPC directly")

            # If Jito is available, try bundle submission
            if use_jito_this_time:
                tx1_unsigned = swap1.get('swapTransaction')
                tx2_unsigned = swap2.get('swapTransaction')

                # Sign transactions before sending to Jito
                # Jito requires signed transactions in base64 format
                tx1_signed = await self._sign_transaction(tx1_unsigned)
                tx2_signed = await self._sign_transaction(tx2_unsigned)

                if not tx1_signed or not tx2_signed:
                    logger.error("Failed to sign transactions for Jito bundle")
                    return

                # CRITICAL: Create tip transaction for Jito
                # Jito requires a tip to consider the bundle (min 1000 lamports)
                # Tip should be the LAST transaction in the bundle
                keypair = self._get_keypair()
                if not keypair:
                    logger.error("Failed to get keypair for tip transaction")
                    return

                # Use configurable tip amount
                # NOTE: 10,000 lamports ($0.002) is TOO LOW for competitive arbitrage
                # Recommend 50,000-100,000 lamports ($0.01-0.02) minimum
                tip_lamports = self.config.get('jito_tip_lamports', 10000)
                if tip_lamports < 50000:
                    logger.warning(f"⚠️ Jito tip {tip_lamports} lamports is low - consider 50,000+ for better success")
                tip_tx = await self.jito.create_tip_transaction(keypair, tip_lamports)

                if not tip_tx:
                    logger.error("Failed to create tip transaction - bundle will likely be rejected")
                    # Continue anyway - sometimes bundles work without tips during low competition
                    bundle_txs = [tx1_signed, tx2_signed]
                else:
                    # Add tip as the last transaction in bundle
                    bundle_txs = [tx1_signed, tx2_signed, tip_tx]
                    logger.info(f"   💰 Jito tip: {tip_lamports} lamports ({tip_lamports/1e9:.6f} SOL)")

                bundle_id = await self.jito.send_bundle(bundle_txs)
                if bundle_id:
                    logger.info(f"✅ Solana Arb executed via Jito: {bundle_id}")
                    await self._log_trade(
                        in_symbol, out_symbol, token_in,
                        amount, profit_pct, bundle_id,
                        quote1.get('routePlan', [])
                    )
                else:
                    # FALLBACK: Jito failed, try direct RPC submission
                    # CRITICAL: After Jito attempts (can take 10+ seconds with rate limits),
                    # the original quotes/transactions are STALE and will fail with 0x1789.
                    # We MUST refresh quotes before attempting RPC fallback.
                    logger.warning("⚠️ Jito bundle failed - attempting RPC fallback with FRESH quotes")

                    # Refresh quotes before RPC fallback with HIGHER slippage for volatile tokens
                    # RPC execution is slower than Jito, so use 2x slippage to handle price movement
                    rpc_slippage = min(self.arb_slippage_bps * 2, 500)  # Max 5% slippage
                    fresh_quote1 = await self.jupiter.get_quote(
                        token_in, token_out, amount, slippage_bps=rpc_slippage
                    )
                    if not fresh_quote1:
                        logger.error(f"❌ [{in_symbol}/{out_symbol}] Failed to get fresh quote for RPC fallback")
                        return

                    fresh_out = int(fresh_quote1.get('outAmount', 0))
                    fresh_quote2 = await self.jupiter.get_quote(
                        token_out, token_in, fresh_out, slippage_bps=rpc_slippage
                    )
                    if not fresh_quote2:
                        logger.error(f"❌ [{in_symbol}/{out_symbol}] Failed to get reverse quote for RPC fallback")
                        return

                    # Verify still profitable
                    fresh_reverse_out = int(fresh_quote2.get('outAmount', 0))
                    fresh_profit = (fresh_reverse_out - amount) / amount
                    if fresh_profit < self.min_profit_threshold:
                        logger.warning(f"⚠️ [{in_symbol}/{out_symbol}] No longer profitable for RPC fallback ({fresh_profit:.2%})")
                        return

                    logger.info(f"   📊 Fresh quotes for RPC: {fresh_profit:.2%} profit")

                    # Create new swap transactions with fresh quotes
                    fresh_swap1 = await self.jupiter.get_swap_transaction(fresh_quote1, self.wallet_address)
                    fresh_swap2 = await self.jupiter.get_swap_transaction(fresh_quote2, self.wallet_address)
                    if not fresh_swap1 or not fresh_swap2:
                        logger.error("Failed to get fresh swap transactions for RPC fallback")
                        return

                    # Sign fresh transactions
                    fresh_tx1 = await self._sign_transaction(fresh_swap1.get('swapTransaction'))
                    fresh_tx2 = await self._sign_transaction(fresh_swap2.get('swapTransaction'))
                    if not fresh_tx1 or not fresh_tx2:
                        logger.error("Failed to sign fresh transactions for RPC fallback")
                        return

                    # Submit fresh transactions via RPC (front-run risk!)
                    logger.warning("   ⚠️ Submitting via RPC (front-run risk!)")
                    fallback_result = await self._submit_via_rpc(fresh_tx1, fresh_tx2)
                    if fallback_result:
                        logger.info(f"✅ Solana Arb executed via RPC fallback: {fallback_result}")
                        await self._log_trade(
                            in_symbol, out_symbol, token_in,
                            amount, fresh_profit, fallback_result,
                            fresh_quote1.get('routePlan', [])
                        )
                    else:
                        logger.error("❌ RPC fallback also failed")
            else:
                # Direct RPC execution (Jito disabled OR rate limited)
                # Use the quotes we already have since they're fresh (no Jito delay)
                reason = "rate limited" if (self.use_jito and not jito_available) else "disabled"
                logger.info(f"📡 Direct RPC execution (Jito {reason})")

                tx1_unsigned = swap1.get('swapTransaction')
                tx2_unsigned = swap2.get('swapTransaction')
                tx1_signed = await self._sign_transaction(tx1_unsigned)
                tx2_signed = await self._sign_transaction(tx2_unsigned)

                if tx1_signed and tx2_signed:
                    result = await self._submit_via_rpc(tx1_signed, tx2_signed)
                    if result:
                        logger.info(f"✅ Solana Arb executed via RPC: {result}")
                        await self._log_trade(
                            in_symbol, out_symbol, token_in,
                            amount, profit_pct, result,
                            quote1.get('routePlan', [])
                        )

        except Exception as e:
            logger.error(f"Solana arb execution error: {e}")

    def _get_keypair(self):
        """
        Get a Solders Keypair object from the configured private key.

        Supports multiple key formats: JSON array, base58, hex

        Returns:
            Keypair object or None on failure
        """
        try:
            import base58
            import json as json_module
            from solders.keypair import Keypair

            if not self.private_key:
                logger.error("No private key configured")
                return None

            key_bytes = None

            # Format 1: JSON array (e.g., [1,2,3,...])
            if self.private_key.startswith('['):
                try:
                    key_array = json_module.loads(self.private_key)
                    key_bytes = bytes(key_array)
                except Exception:
                    pass

            # Format 2: Base58 encoded (most common)
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(self.private_key)
                except Exception:
                    pass

            # Format 3: Hex encoded
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(self.private_key)
                except Exception:
                    pass

            if key_bytes is None:
                logger.error("Failed to parse private key")
                return None

            # Create keypair based on key length
            if len(key_bytes) == 64:
                return Keypair.from_bytes(key_bytes)
            elif len(key_bytes) == 32:
                return Keypair.from_seed(key_bytes)
            else:
                logger.error(f"Invalid key length: {len(key_bytes)} bytes")
                return None

        except Exception as e:
            logger.error(f"Failed to get keypair: {e}")
            return None

    async def _get_sol_balance(self) -> float:
        """
        Get SOL balance for the configured wallet.

        Returns:
            Balance in SOL, or 0.0 on failure
        """
        try:
            if not self.wallet_address:
                return 0.0

            import aiohttp

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [self.wallet_address]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.rpc_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if 'result' in result and 'value' in result['result']:
                            lamports = result['result']['value']
                            return lamports / 1e9  # Convert lamports to SOL
            return 0.0
        except Exception as e:
            logger.debug(f"Failed to get SOL balance: {e}")
            return 0.0

    async def _sign_transaction(self, transaction_b64: str) -> Optional[str]:
        """
        Sign a base64-encoded Solana transaction with the configured private key.

        Supports multiple key formats: JSON array, base58, hex

        Args:
            transaction_b64: Base64-encoded unsigned transaction from Jupiter

        Returns:
            Base64-encoded signed transaction, or None on failure
        """
        try:
            import base64
            import base58
            import json as json_module
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction

            if not self.private_key:
                logger.error("No private key available for signing")
                return None

            # Decode transaction
            tx_bytes = base64.b64decode(transaction_b64)
            tx = VersionedTransaction.from_bytes(tx_bytes)

            # Create keypair from private key (supports multiple formats)
            key_bytes = None
            format_used = None

            # Format 1: JSON array (e.g., [1,2,3,...])
            if self.private_key.startswith('['):
                try:
                    key_array = json_module.loads(self.private_key)
                    key_bytes = bytes(key_array)
                    format_used = "JSON array"
                except Exception:
                    pass

            # Format 2: Base58 encoded (most common)
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(self.private_key)
                    format_used = "base58"
                except Exception:
                    pass

            # Format 3: Hex encoded
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(self.private_key)
                    format_used = "hex"
                except Exception:
                    pass

            if key_bytes is None:
                logger.error("Failed to parse private key - tried JSON array, base58, and hex formats")
                return None

            # Create keypair based on key length
            if len(key_bytes) == 64:
                keypair = Keypair.from_bytes(key_bytes)
            elif len(key_bytes) == 32:
                keypair = Keypair.from_seed(key_bytes)
            else:
                logger.error(f"Invalid key length: {len(key_bytes)} bytes (expected 32 or 64)")
                return None

            # Get message and verify pubkey match
            message = tx.message
            our_pubkey = keypair.pubkey()

            # Get account keys
            account_keys = None
            if hasattr(message, 'account_keys') and message.account_keys:
                account_keys = list(message.account_keys)
            elif hasattr(message, 'static_account_keys'):
                account_keys = list(message.static_account_keys())

            # Verify fee payer matches our keypair
            if account_keys and len(account_keys) > 0:
                fee_payer = account_keys[0]
                if str(fee_payer) != str(our_pubkey):
                    logger.error(f"❌ PUBKEY MISMATCH! Transaction expects: {str(fee_payer)}")
                    logger.error(f"   But we have: {str(our_pubkey)}")
                    return None

            # Sign the transaction - handle multiple signature slots
            # CRITICAL: Transaction may require multiple signatures
            num_required_signatures = message.header.num_required_signatures
            logger.debug(f"Transaction requires {num_required_signatures} signature(s)")

            # Find our pubkey position in the required signers
            our_position = None
            our_pubkey_str = str(our_pubkey)
            if account_keys:
                for i in range(min(num_required_signatures, len(account_keys))):
                    if str(account_keys[i]) == our_pubkey_str:
                        our_position = i
                        break

            if our_position is None:
                logger.error(f"❌ Our pubkey not found in required signers!")
                return None

            # Sign the transaction using the CORRECT solders approach
            # Per solders docs: Pass keypairs directly to VersionedTransaction constructor
            # This properly handles MessageV0 with address lookup tables
            from solders.signature import Signature
            from solders.null_signer import NullSigner
            null_sig = Signature.default()

            # Check if there are existing non-null signatures we need to preserve
            has_existing_signatures = False
            for i, sig in enumerate(tx.signatures):
                if i != our_position and sig != null_sig:
                    has_existing_signatures = True
                    logger.debug(f"   Found existing signature at position {i}")

            if has_existing_signatures:
                # HYBRID APPROACH: Use constructor to get correct signature, then preserve others
                temp_signers = []
                for i in range(num_required_signatures):
                    if i == our_position:
                        temp_signers.append(keypair)
                    elif account_keys and i < len(account_keys):
                        temp_signers.append(NullSigner(account_keys[i]))
                    else:
                        logger.error(f"Missing account key for signer position {i}")
                        return None

                temp_tx = VersionedTransaction(message, temp_signers)
                our_computed_signature = temp_tx.signatures[our_position]

                # Build final signatures array preserving existing signatures
                final_signatures = []
                for i in range(num_required_signatures):
                    if i == our_position:
                        final_signatures.append(our_computed_signature)
                    elif i < len(tx.signatures) and tx.signatures[i] != null_sig:
                        final_signatures.append(tx.signatures[i])
                    else:
                        final_signatures.append(null_sig)

                signed_tx = VersionedTransaction.populate(message, final_signatures)
            else:
                # STANDARD APPROACH: No existing signatures to preserve
                signers = []
                for i in range(num_required_signatures):
                    if i == our_position:
                        signers.append(keypair)
                    elif account_keys and i < len(account_keys):
                        signers.append(NullSigner(account_keys[i]))
                    else:
                        logger.error(f"Missing account key for signer position {i}")
                        return None

                signed_tx = VersionedTransaction(message, signers)

            # Encode back to base64
            signed_bytes = bytes(signed_tx)
            signed_b64 = base64.b64encode(signed_bytes).decode('utf-8')

            logger.debug(f"Transaction signed ({format_used}, pubkey: {str(our_pubkey)[:12]}...)")
            return signed_b64

        except Exception as e:
            logger.error(f"Transaction signing failed: {e}")
            return None

    async def _wait_for_tx_confirmation(self, signature: str, session, timeout: float = None) -> bool:
        """
        Wait for a transaction to be confirmed with robust error handling.

        Args:
            signature: Transaction signature
            session: aiohttp session
            timeout: Maximum time to wait in seconds (default: 60s)

        Returns:
            True if confirmed, False otherwise
        """
        import time

        # Default 60s timeout for arbitrage (faster than trading)
        if timeout is None:
            timeout = 60.0

        start_time = time.time()
        poll_interval = 1.0  # Start with 1s polling
        max_poll_interval = 4.0  # Cap at 4s
        consecutive_errors = 0
        max_consecutive_errors = 5
        current_rpc = self.rpc_url

        while time.time() - start_time < timeout:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[signature], {"searchTransactionHistory": True}]
                }

                async with session.post(
                    current_rpc,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        statuses = result.get('result', {}).get('value', [])
                        if statuses and statuses[0]:
                            status = statuses[0]
                            confirmation = status.get('confirmationStatus')
                            if confirmation in ['confirmed', 'finalized']:
                                logger.debug(f"TX confirmed ({confirmation}): {signature[:16]}...")
                                return True
                            if status.get('err'):
                                logger.error(f"Transaction failed on-chain: {status['err']}")
                                return False
                            # Transaction found but not confirmed yet
                            if confirmation == 'processed':
                                logger.debug(f"TX processed, awaiting confirmation...")
                        consecutive_errors = 0  # Reset on success

                    elif resp.status == 429:
                        # RPC rate limited
                        logger.debug(f"RPC rate limited during confirmation")
                        consecutive_errors += 1
                        poll_interval = min(poll_interval * 1.5, max_poll_interval)

                    else:
                        logger.debug(f"RPC returned {resp.status}")
                        consecutive_errors += 1

            except asyncio.TimeoutError:
                logger.debug(f"RPC timeout during confirmation check")
                consecutive_errors += 1

            except Exception as e:
                logger.debug(f"Confirmation check error: {e}")
                consecutive_errors += 1

            # Try rotating RPC if too many errors
            if consecutive_errors >= max_consecutive_errors:
                try:
                    from config.rpc_provider import RPCProvider
                    new_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
                    if new_rpc and new_rpc != current_rpc:
                        logger.debug(f"Rotating RPC for confirmation...")
                        current_rpc = new_rpc
                        consecutive_errors = 0
                except Exception:
                    pass

            await asyncio.sleep(poll_interval)
            if consecutive_errors > 0:
                poll_interval = min(poll_interval * 1.2, max_poll_interval)

        elapsed = time.time() - start_time
        logger.warning(f"TX confirmation timeout after {elapsed:.1f}s: {signature[:16]}...")
        return False

    async def _submit_via_rpc(self, tx1_signed: str, tx2_signed: str) -> Optional[str]:
        """
        Submit signed transactions directly via RPC (fallback when Jito fails).

        WARNING: Direct RPC submission is vulnerable to front-running!
        Only use as fallback when Jito MEV protection is unavailable.

        IMPORTANT: This method waits for TX1 confirmation before submitting TX2
        to reduce the risk of partial execution causing losses.

        Args:
            tx1_signed: First signed transaction (base64)
            tx2_signed: Second signed transaction (base64)

        Returns:
            Second transaction signature on success, None on failure
        """
        try:
            import aiohttp
            import base64

            rpc_url = self.rpc_url

            # Submit first transaction
            payload1 = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    tx1_signed,
                    {
                        "encoding": "base64",
                        "skipPreflight": False,  # Enable preflight to catch errors early
                        "preflightCommitment": "confirmed",
                        "maxRetries": 3
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                # Send first swap
                async with session.post(rpc_url, json=payload1) as resp:
                    result1 = await resp.json()

                    if 'error' in result1:
                        error = result1.get('error', {})
                        error_msg = error.get('message', 'Unknown')
                        error_code = error.get('code', 'N/A')
                        logger.error(f"RPC tx1 failed [{error_code}]: {error_msg}")
                        # Log simulation logs if available
                        if isinstance(error.get('data'), dict):
                            logs = error['data'].get('logs', [])
                            if logs:
                                logger.error(f"   Simulation logs: {logs[-3:]}")

                        # Detect specific Jupiter errors for better diagnostics
                        error_str = str(error_msg) + str(logs) if logs else str(error_msg)
                        if '0x1789' in error_str or '6025' in error_str:
                            # 0x1789 = InvalidInputIndex - route structure changed
                            logger.error("   ⚠️ 0x1789 (InvalidInputIndex): Route is stale - pool state changed")
                            logger.error("   This often happens with volatile/low-liquidity pools")
                        elif '0x1771' in error_str or '6001' in error_str:
                            # 0x1771 = SlippageToleranceExceeded
                            logger.error("   ⚠️ 0x1771 (SlippageExceeded): Price moved beyond slippage tolerance")
                        elif '0x1772' in error_str or '6002' in error_str:
                            # 0x1772 = InsufficientFunds
                            logger.error("   ⚠️ 0x1772 (InsufficientFunds): Not enough tokens for swap")
                        return None

                    sig1 = result1.get('result')
                    logger.info(f"   TX1 submitted: {sig1[:20]}...")

                # CRITICAL: Wait for TX1 confirmation before sending TX2
                # This reduces the risk of TX1 failing after TX2 is sent
                logger.info(f"   ⏳ Waiting for TX1 confirmation...")
                tx1_confirmed = await self._wait_for_tx_confirmation(sig1, session, timeout=20.0)

                if not tx1_confirmed:
                    logger.error(f"❌ TX1 not confirmed - aborting TX2 to prevent partial execution")
                    logger.warning(f"   Check TX1 status manually: {sig1}")
                    return None

                logger.info(f"   ✅ TX1 confirmed, submitting TX2...")

                # Send second swap
                payload2 = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "sendTransaction",
                    "params": [
                        tx2_signed,
                        {
                            "encoding": "base64",
                            "skipPreflight": False,
                            "preflightCommitment": "confirmed",
                            "maxRetries": 3
                        }
                    ]
                }

                async with session.post(rpc_url, json=payload2) as resp:
                    result2 = await resp.json()

                    if 'error' in result2:
                        error = result2.get('error', {})
                        error_msg = error.get('message', 'Unknown')
                        error_code = error.get('code', 'N/A')
                        logger.error(f"RPC tx2 failed [{error_code}]: {error_msg}")
                        # CRITICAL: TX1 succeeded but TX2 failed - partial execution!
                        logger.error(f"🚨 PARTIAL EXECUTION: TX1 succeeded, TX2 failed!")
                        logger.error(f"   TX1 (confirmed): {sig1}")
                        logger.error(f"   You may need to manually reverse the first swap")
                        # Log simulation logs if available
                        if isinstance(error.get('data'), dict):
                            logs = error['data'].get('logs', [])
                            if logs:
                                logger.error(f"   TX2 simulation logs: {logs[-3:]}")
                        return None

                    sig2 = result2.get('result')
                    logger.info(f"   TX2 submitted: {sig2[:20]}...")

                    return sig2

        except Exception as e:
            logger.error(f"RPC submission failed: {e}")
            return None

    async def _log_trade(
        self,
        in_symbol: str,
        out_symbol: str,
        token_address: str,
        amount: int,
        profit_pct: float,
        tx_hash: str,
        route: List
    ):
        """Log trade to database"""
        if not self.db_pool:
            return

        try:
            import uuid

            # Get SOL price
            sol_price = await self.price_fetcher.get_price('SOL')

            # Calculate values
            if in_symbol == 'SOL':
                amount_sol = amount / 1e9
            else:
                amount_sol = self.trade_amount_sol

            entry_usd = amount_sol * sol_price
            profit_usd = entry_usd * profit_pct

            logger.info(f"💰 Solana Arb [{in_symbol}/{out_symbol}]: {amount_sol:.4f} SOL @ ${sol_price:.2f} = ${entry_usd:.2f} | Profit: ${profit_usd:.2f}")

            trade_id = f"sol_arb_{uuid.uuid4().hex[:12]}"

            # Get route as string
            route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route[:5]])

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO arbitrage_trades (
                        trade_id, token_address, chain, buy_dex, sell_dex,
                        side, entry_price, exit_price, amount, amount_eth,
                        entry_usd, exit_usd, profit_loss, profit_loss_pct, spread_pct,
                        status, is_simulated, entry_timestamp, exit_timestamp,
                        tx_hash, eth_price_at_trade, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                """,
                    trade_id,
                    token_address,
                    'solana',
                    'jupiter',
                    'jupiter',
                    'buy',
                    sol_price,
                    sol_price * (1 + profit_pct),
                    amount_sol,
                    amount_sol,
                    entry_usd,
                    entry_usd + profit_usd,
                    profit_usd,
                    profit_pct * 100,
                    profit_pct * 100,
                    'closed',
                    self.dry_run,
                    datetime.now(),
                    datetime.now(),
                    tx_hash,
                    sol_price,
                    json.dumps({
                        'chain': 'solana',
                        'in_symbol': in_symbol,
                        'out_symbol': out_symbol,
                        'route': route_str,
                        'dry_run': self.dry_run
                    })
                )
            logger.debug(f"💾 Logged Solana arb: {trade_id}")

        except Exception as e:
            logger.error(f"Error logging Solana trade: {e}")

    async def stop(self):
        self.is_running = False

        await self.jupiter.close()
        await self.raydium.close()
        await self.jito.close()

        logger.info("🛑 Solana Arbitrage Engine Stopped")
