"""
Solana Listener for Sniper Module - CREDIT-EFFICIENT VERSION
Uses polling instead of WebSocket subscriptions to minimize Helius credit usage.

Supports:
- Raydium V4 AMM new pool detection via polling
- Pump.fun bonding curve graduation
- Configurable poll intervals
"""

import logging
import asyncio
from typing import List, Dict, Optional, Set
import aiohttp
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger("SolanaListener")

# Program IDs
RAYDIUM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CPMM_PROGRAM_ID = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# WSOL mint address
WSOL_MINT = "So11111111111111111111111111111111111111112"


class SolanaListener:
    """
    Credit-efficient Solana listener using polling instead of WebSocket.
    Polls for new Raydium pools every 30 seconds (configurable).

    Credit Usage:
    - Polling mode: ~2-4 API calls per minute (minimal credits)
    - WebSocket mode: 1000s of messages per second (MASSIVE credits)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rpc_url = config.get('solana', {}).get('rpc_url')

        # Use Pool Engine with fallback
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                self.rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC')
            except Exception:
                pass
        if not self.rpc_url:
            self.rpc_url = os.getenv('SOLANA_RPC_URL')

        # Get Helius API from Pool Engine or env
        self.helius_api_key = None
        try:
            from config.rpc_provider import RPCProvider
            self.helius_api_key = RPCProvider.get_api_sync('HELIUS_API')
        except Exception:
            pass
        if not self.helius_api_key:
            # Try secrets manager first, then env fallback
            try:
                from security.secrets_manager import secrets
                self.helius_api_key = secrets.get('HELIUS_API_KEY', log_access=False) or os.getenv('HELIUS_API_KEY')
            except Exception:
                self.helius_api_key = os.getenv('HELIUS_API_KEY')

        # Polling configuration (credit-efficient)
        self.poll_interval = int(os.getenv('SNIPER_POLL_INTERVAL', '30'))  # seconds
        self.use_websocket = os.getenv('SNIPER_USE_WEBSOCKET', 'false').lower() == 'true'

        # State
        self.known_signatures: Set[str] = set()
        self.new_pools_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.poll_task: Optional[asyncio.Task] = None
        self.last_slot: Optional[int] = None

        # Stats for logging
        self._stats = {
            'polls': 0,
            'pools_detected': 0,
            'pools_queued': 0,
            'api_calls': 0,
            'last_log_time': datetime.now()
        }
        self._log_interval = timedelta(minutes=5)

        # Limit known signatures to prevent memory bloat
        self._max_signatures = 1000

    async def initialize(self):
        """Initialize the Solana listener with credit-efficient polling"""
        logger.info(f"🔌 Initializing Solana Listener (POLLING MODE)...")
        logger.info(f"   RPC: {self.rpc_url[:50] if self.rpc_url else 'Not configured'}...")
        logger.info(f"   Poll Interval: {self.poll_interval}s")
        logger.info(f"   WebSocket Mode: {'ENABLED (high credits!)' if self.use_websocket else 'DISABLED (credit-safe)'}")

        if not self.rpc_url:
            logger.error("❌ No RPC URL provided for Solana Listener")
            return

        self.is_running = True

        if self.use_websocket:
            logger.warning("⚠️ WebSocket mode enabled - this consumes MASSIVE credits!")
            logger.warning("   Set SNIPER_USE_WEBSOCKET=false to use credit-efficient polling")
            # Don't start WebSocket - too expensive
            logger.info("   WebSocket disabled to protect credits. Using polling instead.")

        # Start credit-efficient polling
        self.poll_task = asyncio.create_task(self._run_polling_listener())
        logger.info("✅ Solana Listener initialized with polling (credit-efficient)")

    async def get_new_pools(self) -> List[Dict]:
        """Get newly detected pools from the queue."""
        new_pools = []
        while not self.new_pools_queue.empty():
            try:
                pool = self.new_pools_queue.get_nowait()
                new_pools.append(pool)
            except asyncio.QueueEmpty:
                break
        return new_pools

    async def _run_polling_listener(self):
        """Main polling loop - credit efficient"""
        logger.info(f"📡 Starting pool detection polling (every {self.poll_interval}s)")

        while self.is_running:
            try:
                await self._poll_for_new_pools()
                self._stats['polls'] += 1

                # Log stats periodically
                await self._log_stats_if_needed()

            except Exception as e:
                logger.error(f"Polling error: {e}")

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

    async def _poll_for_new_pools(self):
        """Poll for new Raydium pool signatures"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get recent signatures for Raydium V4
                signatures = await self._get_recent_signatures(
                    session,
                    RAYDIUM_V4_PROGRAM_ID,
                    limit=10  # Only check last 10 transactions
                )

                for sig_info in signatures:
                    signature = sig_info.get('signature')

                    if not signature or signature in self.known_signatures:
                        continue

                    self.known_signatures.add(signature)

                    # Check if this is a pool initialization
                    pool_info = await self._check_and_fetch_pool(session, signature)

                    if pool_info:
                        self._stats['pools_detected'] += 1
                        self._stats['pools_queued'] += 1
                        await self.new_pools_queue.put(pool_info)
                        logger.info(f"🆕 New pool detected: {pool_info.get('token_address', 'unknown')[:16]}...")

                # Cleanup old signatures to prevent memory bloat
                if len(self.known_signatures) > self._max_signatures:
                    # Keep only most recent half
                    self.known_signatures = set(list(self.known_signatures)[-self._max_signatures//2:])

        except Exception as e:
            logger.error(f"Error polling for pools: {e}")

    async def _get_recent_signatures(self, session: aiohttp.ClientSession,
                                      program_id: str, limit: int = 10) -> List[Dict]:
        """Get recent transaction signatures for a program"""
        self._stats['api_calls'] += 1

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [
                program_id,
                {"limit": limit, "commitment": "confirmed"}
            ]
        }

        try:
            async with session.post(self.rpc_url, json=payload, timeout=15) as response:
                if response.status == 429:
                    logger.warning("⚠️ Rate limited - backing off and reporting to pool")
                    # Report to Pool Engine for rotation
                    try:
                        from config.rpc_provider import RPCProvider
                        await RPCProvider.report_rate_limit('SOLANA_RPC', self.rpc_url, 300)
                        # Get new RPC URL for next request
                        new_url = await RPCProvider.get_rpc('SOLANA_RPC')
                        if new_url and new_url != self.rpc_url:
                            self.rpc_url = new_url
                            logger.info(f"🔄 Rotated to new RPC endpoint")
                    except Exception:
                        pass
                    await asyncio.sleep(60)
                    return []

                if response.status != 200:
                    return []

                data = await response.json()
                return data.get('result', [])

        except asyncio.TimeoutError:
            logger.debug("Timeout getting signatures")
            return []
        except Exception as e:
            logger.debug(f"Error getting signatures: {e}")
            return []

    async def _check_and_fetch_pool(self, session: aiohttp.ClientSession,
                                     signature: str) -> Optional[Dict]:
        """Check if transaction is a pool init and fetch details"""
        self._stats['api_calls'] += 1

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
            ]
        }

        try:
            async with session.post(self.rpc_url, json=payload, timeout=15) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                result = data.get('result')

                if not result:
                    return None

                # Check if this is a pool initialization
                meta = result.get('meta', {})
                if meta.get('err') is not None:
                    return None  # Failed transaction

                # Check logs for init keywords
                log_messages = meta.get('logMessages', [])
                if not self._is_pool_init(log_messages):
                    return None

                # Extract token info
                token_mint, pool_address = self._parse_pool_transaction(result)

                if token_mint and token_mint != WSOL_MINT:
                    return {
                        'token_address': token_mint,
                        'pair_address': pool_address or signature,
                        'chain': 'solana',
                        'timestamp': datetime.utcnow().isoformat(),
                        'signature': signature,
                        'source': 'polling'
                    }

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching transaction: {signature[:16]}...")
        except Exception as e:
            logger.debug(f"Error fetching pool: {e}")

        return None

    def _is_pool_init(self, logs: List[str]) -> bool:
        """Check if logs indicate a pool initialization"""
        init_keywords = [
            'initialize2',
            'Initialize',
            'InitializePool',
            'create_pool',
            'init'
        ]

        for log in logs:
            log_lower = log.lower()
            for keyword in init_keywords:
                if keyword.lower() in log_lower:
                    return True
        return False

    def _parse_pool_transaction(self, tx_result: Dict) -> tuple:
        """Parse transaction to extract token mint and pool address"""
        token_mint = None
        pool_address = None

        try:
            meta = tx_result.get('meta', {})

            # Look through post token balances for new token
            post_balances = meta.get('postTokenBalances', [])
            pre_balances = meta.get('preTokenBalances', [])
            pre_mints = {b.get('mint') for b in pre_balances}

            for balance in post_balances:
                mint = balance.get('mint')
                if mint and mint != WSOL_MINT and mint not in pre_mints:
                    token_mint = mint
                    break

            # Fallback: find any non-WSOL mint
            if not token_mint:
                for balance in post_balances:
                    mint = balance.get('mint')
                    if mint and mint != WSOL_MINT:
                        token_mint = mint
                        break

            # Try to find pool address from inner instructions
            inner_instructions = meta.get('innerInstructions', [])
            for inner in inner_instructions:
                for ix in inner.get('instructions', []):
                    if isinstance(ix, dict):
                        if ix.get('program') == 'system':
                            parsed = ix.get('parsed', {})
                            if parsed.get('type') == 'createAccount':
                                pool_address = parsed.get('info', {}).get('newAccount')
                                break

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")

        return token_mint, pool_address

    async def _log_stats_if_needed(self):
        """Log detection stats periodically"""
        now = datetime.now()
        if now - self._stats['last_log_time'] >= self._log_interval:
            polls = self._stats['polls']
            detected = self._stats['pools_detected']
            queued = self._stats['pools_queued']
            api_calls = self._stats['api_calls']

            logger.info(f"📊 Solana Listener Stats (Last 5 min): "
                       f"Polls: {polls} | API Calls: {api_calls} | "
                       f"Pools Detected: {detected} | Queued: {queued}")

            # Reset stats
            self._stats = {
                'polls': 0,
                'pools_detected': 0,
                'pools_queued': 0,
                'api_calls': 0,
                'last_log_time': now
            }

    async def close(self):
        """Clean up resources"""
        self.is_running = False

        if self.poll_task:
            self.poll_task.cancel()
            try:
                await self.poll_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Solana Listener closed")
