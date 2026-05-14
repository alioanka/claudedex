"""
Solana Listener for Sniper Module - ROBUST MULTI-AMM VERSION

Supports detection from multiple sources:
- Raydium V4 AMM pools
- Raydium CPMM pools
- Pump.fun token launches and graduations
- Orca Whirlpools
- Meteora dynamic pools

Uses credit-efficient polling with smart filtering.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Set, Tuple
import aiohttp
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from config.rpc_provider import RPCProvider

logger = logging.getLogger("SolanaListener")


class PoolSource(Enum):
    """Source of pool detection"""
    RAYDIUM_V4 = "raydium_v4"
    RAYDIUM_CPMM = "raydium_cpmm"
    PUMP_FUN = "pump_fun"
    ORCA = "orca"
    METEORA = "meteora"
    UNKNOWN = "unknown"


# Program IDs for different AMMs
PROGRAM_IDS = {
    PoolSource.RAYDIUM_V4: "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    PoolSource.RAYDIUM_CPMM: "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",
    PoolSource.PUMP_FUN: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
    PoolSource.ORCA: "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    PoolSource.METEORA: "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",
}

# Pool initialization keywords by AMM
INIT_KEYWORDS = {
    PoolSource.RAYDIUM_V4: ["initialize2", "initialize", "initializepool", "create_pool", "init"],
    PoolSource.RAYDIUM_CPMM: ["initialize", "create_pool", "init_pool"],
    PoolSource.PUMP_FUN: ["create", "buy", "sell", "init"],  # Pump.fun uses different keywords
    PoolSource.ORCA: ["initializetick", "initializepool", "openposition"],
    PoolSource.METEORA: ["initialize", "create_pool", "init"],
}

# WSS-stage strict init tokens, per-source. Operator-instrumentation
# (commit 542bf51) showed INIT_KEYWORDS' bare "init" matches unrelated
# transactions that merely *mention* the Raydium program in their account
# list (DEX aggregators, generic SOL transfers, unknown programs). At the
# WSS pre-filter we require one of these canonical init signals instead.
# INIT_KEYWORDS is still used by _is_pool_init as the second-stage filter
# against canonical logMessages from getTransaction.
#
# Per-source design lets each AMM declare its own canonical token-creation
# marker. Pump.fun's "Buy"/"Sell" instructions fire orders of magnitude
# more often than "Create" and are NOT token launches — strict filter
# matches only the canonical create instruction. Each tuple's tokens are
# lowercased; logsNotification's log_blob is also lowercased before
# substring match.
STRICT_INIT_TOKENS_BY_SOURCE = {
    PoolSource.RAYDIUM_V4: ("initialize2", "init_pc_amount", "ray_log:"),
    PoolSource.PUMP_FUN: ("instruction: create",),
}

# Back-compat alias (some downstream tests reference this constant
# directly). Kept in sync with the Raydium V4 entry of the dict above.
STRICT_INIT_TOKENS = STRICT_INIT_TOKENS_BY_SOURCE[PoolSource.RAYDIUM_V4]

# Known mints to filter out (stablecoins, wrapped SOL)
WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

FILTERED_MINTS = {WSOL_MINT, USDC_MINT, USDT_MINT}

# Phase 1: Raydium V4 program ID (mainnet). Public constant.
RAYDIUM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Phase 1: WSS reconnect backoff (seconds). Capped exponential.
WSS_BACKOFF_INITIAL_SECONDS = 1.0
WSS_BACKOFF_MAX_SECONDS = 60.0


@dataclass
class DetectedPool:
    """Represents a newly detected pool"""
    token_address: str
    pair_address: str
    chain: str
    source: PoolSource
    timestamp: datetime
    signature: str
    base_liquidity: Optional[float] = None
    metadata: Optional[Dict] = None


class SolanaListener:
    """
    Robust Solana listener using polling with multi-AMM support.

    Features:
    - Polls multiple AMM programs for new pools
    - Smart filtering to avoid spam tokens
    - Rate limit handling with RPC rotation
    - Detailed logging for debugging
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rpc_url = self._get_rpc_url(config)
        self.helius_api_key = self._get_helius_key()

        # Polling configuration
        self.poll_interval = int(os.getenv('SNIPER_POLL_INTERVAL', '15'))  # Faster polling
        # Phase 1: SNIPER_LISTENER_MODE = 'polling' (default) | 'wss'
        # WSS mode opens a programSubscribe connection to Raydium V4 for
        # sub-second pool detection. Currently logs notifications only; pool
        # events are still emitted by the polling path (Phase 1.5 will wire
        # WSS into the downstream queue).
        self.listener_mode = os.getenv('SNIPER_LISTENER_MODE', 'polling').strip().lower()
        self.wss_url = os.getenv('SNIPER_SOLANA_WSS_URL', '').strip() or self._infer_wss_url()
        # Deprecated: SNIPER_USE_WEBSOCKET was unused. Kept readable for
        # back-compat detection so we can warn operators who set it.
        if os.getenv('SNIPER_USE_WEBSOCKET'):
            logger.warning(
                "SNIPER_USE_WEBSOCKET is deprecated; use "
                "SNIPER_LISTENER_MODE=wss instead. Currently ignored."
            )
        self.use_websocket = self.listener_mode == 'wss'

        # Which AMMs to monitor (configurable)
        self.enabled_sources = self._get_enabled_sources()

        # WSS-eligible sources: enabled AND we know their strict init tokens.
        # Other AMMs (orca/meteora/raydium_cpmm) can be added by extending
        # STRICT_INIT_TOKENS_BY_SOURCE.
        self._wss_sources = [
            src for src in self.enabled_sources
            if src in STRICT_INIT_TOKENS_BY_SOURCE
        ]

        # State
        self.known_signatures: Dict[PoolSource, Set[str]] = {
            source: set() for source in PoolSource
        }
        self.new_pools_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.poll_task: Optional[asyncio.Task] = None
        self.wss_task: Optional[asyncio.Task] = None

        # Stats
        self._stats = {
            'polls': 0,
            'pools_detected': 0,
            'pools_queued': 0,
            'api_calls': 0,
            'errors': 0,
            'last_log_time': datetime.now(),
            'by_source': {source.value: 0 for source in PoolSource},
            'wss_connects': 0,
            'wss_notifications': 0,
            'wss_pools_queued': 0,
            # _check_pool_transaction rejection-reason counters (cumulative,
            # never reset by the 5-min window — operator queries these via
            # sniper_runtime_stats to see the rejection profile over time).
            'rejected_rpc_error': 0,        # response.status != 200 or exception
            'rejected_no_result': 0,        # getTransaction returned null
            'rejected_tx_failed': 0,        # meta.err is not None
            'rejected_not_init': 0,         # _is_pool_init returned False
            'rejected_no_mint': 0,          # _parse_pool_transaction returned empty mint
            'rejected_filtered_mint': 0,    # mint in FILTERED_MINTS
            'rejected_bad_mint_format': 0,  # mint length not in [32, 44]
        }
        self._log_interval = timedelta(minutes=5)
        self._max_signatures = 500  # Per source

        # Session reuse for better performance
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_rpc_url(self, config: Dict) -> Optional[str]:
        """Get RPC URL from config, PoolEngine (sync), .env preserved as ultimate fallback"""
        return (
            config.get('solana', {}).get('rpc_url')
            or RPCProvider.get_rpc_sync('SOLANA_RPC')
            or os.getenv('SOLANA_RPC_URL')
        )

    def _infer_wss_url(self) -> Optional[str]:
        """Derive a WSS URL from the HTTP RPC URL by swapping the scheme.
        Helius / QuickNode / Triton all serve WSS on the same hostname.
        Returns None if rpc_url is unset or non-derivable."""
        if not self.rpc_url:
            return None
        if self.rpc_url.startswith('https://'):
            return 'wss://' + self.rpc_url[len('https://'):]
        if self.rpc_url.startswith('http://'):
            return 'ws://' + self.rpc_url[len('http://'):]
        return None

    def _get_helius_key(self) -> Optional[str]:
        """Get Helius API key"""
        try:
            from config.rpc_provider import RPCProvider
            key = RPCProvider.get_api_sync('HELIUS_API')
            if key:
                return key
        except Exception:
            pass

        try:
            from security.secrets_manager import secrets
            key = secrets.get('HELIUS_API_KEY', log_access=False)
            if key:
                return key
        except Exception:
            pass

        return os.getenv('HELIUS_API_KEY')

    def _get_enabled_sources(self) -> List[PoolSource]:
        """Get list of enabled AMM sources from config"""
        sources_str = os.getenv('SNIPER_AMM_SOURCES', 'raydium_v4,pump_fun')
        enabled = []

        for source_name in sources_str.split(','):
            source_name = source_name.strip().lower()
            try:
                enabled.append(PoolSource(source_name))
            except ValueError:
                logger.warning(f"Unknown AMM source: {source_name}")

        # Always include Raydium V4 as fallback
        if not enabled:
            enabled = [PoolSource.RAYDIUM_V4]

        return enabled

    async def initialize(self):
        """Initialize the Solana listener"""
        logger.info("🔌 Initializing Solana Listener (MULTI-AMM VERSION)...")
        logger.info(f"   RPC: {self.rpc_url[:50] if self.rpc_url else 'NOT CONFIGURED'}...")
        logger.info(f"   Poll Interval: {self.poll_interval}s")
        logger.info(f"   Enabled AMMs: {[s.value for s in self.enabled_sources]}")
        logger.info(f"   Listener Mode: {self.listener_mode.upper()}")
        if self.listener_mode == 'wss':
            logger.info(f"   WSS URL: {(self.wss_url or 'NONE')[:60]}...")

        if not self.rpc_url:
            logger.error("❌ No RPC URL provided - Solana detection DISABLED")
            logger.error("   Set SOLANA_RPC_URL in .env or configure Pool Engine")
            return

        # Verify RPC is working
        if not await self._verify_rpc():
            logger.error("❌ RPC verification failed - check your RPC URL")
            return

        self.is_running = True

        # Create session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # Start polling
        if self.listener_mode == 'wss':
            if not self.wss_url:
                logger.warning(
                    "SNIPER_LISTENER_MODE=wss but no WSS URL available "
                    "(set SNIPER_SOLANA_WSS_URL or use an https:// RPC); "
                    "falling back to polling"
                )
                self.listener_mode = 'polling'
                self.use_websocket = False
                self.poll_task = asyncio.create_task(self._run_polling_listener())
            else:
                logger.info(f"📡 Starting WSS listener: {self.wss_url[:60]}...")
                self.wss_task = asyncio.create_task(self._run_wss_listener())
                # Still run polling as a backstop until Phase 1.5 wires WSS
                # into the downstream queue.
                self.poll_task = asyncio.create_task(self._run_polling_listener())
        else:
            self.poll_task = asyncio.create_task(self._run_polling_listener())
        logger.info("✅ Solana Listener initialized successfully")

    async def _verify_rpc(self) -> bool:
        """Verify RPC connection is working"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                }
                async with session.post(self.rpc_url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('result') == 'ok':
                            logger.info("   RPC health check: OK")
                            return True
                        # Some RPCs don't support getHealth, try getSlot

                payload = {"jsonrpc": "2.0", "id": 1, "method": "getSlot"}
                async with session.post(self.rpc_url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('result'):
                            logger.info(f"   RPC verified (slot: {data['result']})")
                            return True

        except Exception as e:
            logger.error(f"   RPC verification failed: {e}")

        return False

    async def get_new_pools(self) -> List[Dict]:
        """Get newly detected pools from the queue"""
        new_pools = []
        while not self.new_pools_queue.empty():
            try:
                pool = self.new_pools_queue.get_nowait()
                new_pools.append(pool)
            except asyncio.QueueEmpty:
                break
        return new_pools

    async def _run_polling_listener(self):
        """Main polling loop"""
        logger.info(f"📡 Starting multi-AMM pool detection (every {self.poll_interval}s)")

        while self.is_running:
            try:
                # Poll each enabled source
                for source in self.enabled_sources:
                    if not self.is_running:
                        break
                    await self._poll_source(source)
                    # Small delay between sources to avoid rate limits
                    await asyncio.sleep(0.5)

                self._stats['polls'] += 1
                await self._log_stats_if_needed()

            except Exception as e:
                logger.error(f"Polling error: {e}")
                self._stats['errors'] += 1

            await asyncio.sleep(self.poll_interval)

    async def _run_wss_listener(self) -> None:
        """Phase 1 SNIPER WSS listener — connects to a Solana WSS endpoint,
        sends one logsSubscribe per WSS-eligible AMM source on the same
        connection, then routes incoming notifications back to the right
        source via the subscription-id map. Reconnects on disconnect with
        exponential backoff capped at 60s."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package not installed — WSS listener disabled. "
                "pip install websockets>=12.0"
            )
            return

        backoff = WSS_BACKOFF_INITIAL_SECONDS
        sub_id_counter = 0

        while self.is_running:
            try:
                async with websockets.connect(self.wss_url, ping_interval=20, ping_timeout=10) as ws:
                    self._stats['wss_connects'] += 1
                    backoff = WSS_BACKOFF_INITIAL_SECONDS  # reset on successful connect
                    sources_str = ", ".join(s.value for s in self._wss_sources) or "<none>"
                    logger.info(
                        f"🔌 WSS connected (connect #{self._stats['wss_connects']}); "
                        f"📡 WSS listener targeting {len(self._wss_sources)} source(s): {sources_str}"
                    )

                    # Subscribe to each WSS-eligible source separately so the
                    # subscription-id in notifications tells us which AMM
                    # emitted it. Solana RPC quirk: the request `id` echoed
                    # in the ack is different from the subscription id in
                    # `result` (which is what logsNotification.params.subscription
                    # carries) — we rebind once the ack arrives.
                    sub_id_to_source: Dict[int, PoolSource] = {}
                    for src in self._wss_sources:
                        sub_id_counter += 1
                        program_id = PROGRAM_IDS[src]
                        sub_msg = {
                            "jsonrpc": "2.0",
                            "id": sub_id_counter,
                            "method": "logsSubscribe",
                            "params": [
                                {"mentions": [program_id]},
                                {"commitment": "processed"},
                            ],
                        }
                        await ws.send(json.dumps(sub_msg))
                        sub_id_to_source[sub_id_counter] = src
                        logger.info(
                            f"🔌 WSS subscribing to {src.value} program {program_id}"
                        )

                    async for raw in ws:
                        if not self.is_running:
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        # Subscribe-ack: rebind request id -> actual subscription id.
                        if 'result' in msg and 'method' not in msg:
                            msg_id = msg.get('id')
                            sub_id_returned = msg.get('result')
                            if msg_id in sub_id_to_source and isinstance(sub_id_returned, int):
                                src = sub_id_to_source.pop(msg_id)
                                sub_id_to_source[sub_id_returned] = src
                                logger.info(
                                    f"✅ WSS subscription ack: {src.value} → sid={sub_id_returned}"
                                )
                            else:
                                logger.info(f"✅ WSS subscription ack (unmapped): {msg}")
                            continue

                        if msg.get('method') == 'logsNotification':
                            self._stats['wss_notifications'] += 1
                            params = msg.get('params') or {}
                            sub_id = params.get('subscription')
                            src = sub_id_to_source.get(sub_id)
                            if src is None:
                                # Notification for a subscription we don't recognize
                                # (e.g., ack hasn't landed yet). Skip silently.
                                continue

                            program_id = PROGRAM_IDS[src]
                            strict_tokens = STRICT_INIT_TOKENS_BY_SOURCE[src]

                            try:
                                value = params.get('result', {}).get('value', {})
                            except Exception:
                                continue
                            if not isinstance(value, dict) or value.get('err') is not None:
                                continue
                            signature = value.get('signature')
                            logs = value.get('logs') or []
                            if not signature or not isinstance(logs, list):
                                continue

                            # Dedup per source against polling-path known_signatures
                            sig_set = self.known_signatures.get(src)
                            if sig_set is None or signature in sig_set:
                                continue

                            # Strict two-gate filter — same shape as Raydium,
                            # source-tunable via STRICT_INIT_TOKENS_BY_SOURCE.
                            log_blob = " ".join(str(line).lower() for line in logs)
                            program_invoke_marker = f"program {program_id.lower()} invoke"
                            if program_invoke_marker not in log_blob:
                                continue
                            if not any(tok in log_blob for tok in strict_tokens):
                                continue

                            logger.info(
                                f"⚡ WSS init-candidate [{src.value}]: sig={signature[:16]}... "
                                f"({len(logs)} log lines)"
                            )

                            # Mark seen BEFORE the RPC call so concurrent
                            # retries don't double-fire if the call is slow.
                            sig_set.add(signature)
                            if len(sig_set) > self._max_signatures:
                                keep = list(sig_set)[-self._max_signatures:]
                                self.known_signatures[src] = set(keep)

                            try:
                                pool_info = await self._check_pool_transaction(signature, src)
                            except Exception as e:
                                logger.debug(
                                    f"WSS pool-check failed for {signature[:16]} [{src.value}]: {e}"
                                )
                                continue
                            if pool_info is None:
                                continue

                            pool_dict = {
                                'token_address': pool_info.token_address,
                                'pair_address': pool_info.pair_address,
                                'chain': 'solana',
                                'source': src.value,
                                # Wall-clock UTC ISO at emit-time — Phase 0
                                # timing's parse_iso_to_perf_counter reads
                                # this to anchor t_detect.
                                'timestamp': datetime.utcnow().isoformat(),
                                'signature': signature,
                                'base_liquidity': pool_info.base_liquidity,
                                'metadata': dict(pool_info.metadata or {}, **{'detection_path': 'wss'}),
                            }

                            self._stats['pools_detected'] += 1
                            self._stats['pools_queued'] += 1
                            self._stats['by_source'][src.value] += 1
                            self._stats['wss_pools_queued'] = self._stats.get('wss_pools_queued', 0) + 1

                            await self.new_pools_queue.put(pool_dict)
                            logger.info(
                                f"🆕 [WSS {src.value.upper()}] New pool: "
                                f"{pool_info.token_address[:12]}... "
                                f"(pair: {(pool_info.pair_address or 'N/A')[:12]}...)"
                            )

            except asyncio.CancelledError:
                logger.info("WSS listener cancelled")
                return
            except Exception as e:
                logger.warning(
                    f"WSS disconnect: {type(e).__name__}: {e}; "
                    f"reconnecting in {backoff:.1f}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * 2, WSS_BACKOFF_MAX_SECONDS)

    async def _poll_source(self, source: PoolSource):
        """Poll a specific AMM source for new pools"""
        program_id = PROGRAM_IDS.get(source)
        if not program_id:
            return

        try:
            # Get recent signatures
            signatures = await self._get_recent_signatures(program_id, limit=20)

            if not signatures:
                logger.debug(f"No signatures returned for {source.value}")
                return

            new_pools_found = 0

            for sig_info in signatures:
                signature = sig_info.get('signature')
                if not signature:
                    continue

                # Skip if already seen
                if signature in self.known_signatures[source]:
                    continue

                self.known_signatures[source].add(signature)

                # Check if this is a pool creation
                pool_info = await self._check_pool_transaction(signature, source)

                if pool_info:
                    new_pools_found += 1
                    self._stats['pools_detected'] += 1
                    self._stats['pools_queued'] += 1
                    self._stats['by_source'][source.value] += 1

                    # Convert to dict for queue
                    pool_dict = {
                        'token_address': pool_info.token_address,
                        'pair_address': pool_info.pair_address,
                        'chain': 'solana',
                        'source': source.value,
                        'timestamp': pool_info.timestamp.isoformat(),
                        'signature': signature,
                        'base_liquidity': pool_info.base_liquidity,
                        'metadata': dict(pool_info.metadata or {}, **{'detection_path': 'polling'}),
                    }

                    await self.new_pools_queue.put(pool_dict)

                    logger.info(
                        f"🆕 [{source.value.upper()}] New pool: "
                        f"{pool_info.token_address[:12]}... "
                        f"(pair: {pool_info.pair_address[:12] if pool_info.pair_address else 'N/A'}...)"
                    )

            if new_pools_found > 0:
                logger.info(f"   Found {new_pools_found} new pools from {source.value}")

            # Cleanup old signatures
            self._cleanup_signatures(source)

        except Exception as e:
            logger.error(f"Error polling {source.value}: {e}")
            self._stats['errors'] += 1

    async def _get_recent_signatures(
        self,
        program_id: str,
        limit: int = 20
    ) -> List[Dict]:
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
            if not self._session:
                self._session = aiohttp.ClientSession()

            async with self._session.post(
                self.rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:

                if response.status == 429:
                    await self._handle_rate_limit()
                    return []

                if response.status != 200:
                    logger.warning(f"RPC returned status {response.status}")
                    return []

                data = await response.json()

                if 'error' in data:
                    logger.warning(f"RPC error: {data['error']}")
                    return []

                return data.get('result', [])

        except asyncio.TimeoutError:
            logger.debug(f"Timeout getting signatures for {program_id[:12]}...")
            return []
        except Exception as e:
            logger.debug(f"Error getting signatures: {e}")
            return []

    async def _check_pool_transaction(
        self,
        signature: str,
        source: PoolSource
    ) -> Optional[DetectedPool]:
        """Check if a transaction is a pool creation"""
        self._stats['api_calls'] += 1

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": "confirmed",
                }
            ]
        }

        try:
            async with self._session.post(
                self.rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:

                if response.status != 200:
                    self._stats['rejected_rpc_error'] += 1
                    return None

                data = await response.json()
                result = data.get('result')

                if not result:
                    self._stats['rejected_no_result'] += 1
                    return None

                # Skip failed transactions
                meta = result.get('meta', {})
                if meta.get('err') is not None:
                    self._stats['rejected_tx_failed'] += 1
                    return None

                # Check logs for pool init
                log_messages = meta.get('logMessages', [])

                if not self._is_pool_init(log_messages, source):
                    # Sample the first 10 'not_init' rejections so the operator
                    # can paste an example of logMessages that the keyword
                    # pre-filter accepted but _is_pool_init rejected. Bump the
                    # counter AFTER reading the pre-bump value so the sample
                    # condition fires at #50, #100, ... #500 inclusive.
                    rej = self._stats.get('rejected_not_init', 0)
                    next_rej = rej + 1
                    if next_rej % 50 == 0 and next_rej <= 500:
                        sample = (log_messages[:5] if log_messages else ['<empty>'])
                        logger.info(
                            f"🔍 Rejection sample (not_init #{next_rej}): "
                            f"sig={signature[:16]}... "
                            f"log_count={len(log_messages)} "
                            f"logs={sample}"
                        )
                    self._stats['rejected_not_init'] = next_rej
                    return None

                # Extract token info
                token_mint, pool_address, liquidity = self._parse_pool_transaction(
                    result, source
                )

                # Filter out known mints and invalid tokens
                if not token_mint:
                    self._stats['rejected_no_mint'] += 1
                    return None
                if token_mint in FILTERED_MINTS:
                    self._stats['rejected_filtered_mint'] += 1
                    return None

                # Validate token address format (base58, 32-44 chars)
                if len(token_mint) < 32 or len(token_mint) > 44:
                    self._stats['rejected_bad_mint_format'] += 1
                    return None

                return DetectedPool(
                    token_address=token_mint,
                    pair_address=pool_address or signature,
                    chain='solana',
                    source=source,
                    timestamp=datetime.utcnow(),
                    signature=signature,
                    base_liquidity=liquidity,
                    metadata={
                        'program_id': PROGRAM_IDS.get(source),
                        'block_time': result.get('blockTime')
                    }
                )

        except asyncio.TimeoutError:
            self._stats['rejected_rpc_error'] += 1
            logger.debug(f"Timeout fetching transaction: {signature[:16]}...")
        except Exception as e:
            self._stats['rejected_rpc_error'] += 1
            logger.debug(f"Error checking pool: {e}")

        return None

    def _is_pool_init(self, logs: List[str], source: PoolSource) -> bool:
        """Check if logs indicate a pool initialization"""
        keywords = INIT_KEYWORDS.get(source, ['init', 'create'])

        for log in logs:
            log_lower = log.lower()

            # Check for program-specific keywords
            for keyword in keywords:
                if keyword.lower() in log_lower:
                    return True

            # Generic pool creation indicators
            if any(indicator in log_lower for indicator in [
                'pool created',
                'liquidity added',
                'new pool',
                'market created'
            ]):
                return True

        return False

    def _parse_pool_transaction(
        self,
        tx_result: Dict,
        source: PoolSource
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Parse transaction to extract token mint, pool address, and liquidity"""
        token_mint = None
        pool_address = None
        liquidity = None

        try:
            meta = tx_result.get('meta', {})

            # Find new token from post/pre balance difference
            post_balances = meta.get('postTokenBalances', [])
            pre_balances = meta.get('preTokenBalances', [])
            pre_mints = {b.get('mint') for b in pre_balances if b.get('mint')}

            # Look for new mints that appeared after tx
            for balance in post_balances:
                mint = balance.get('mint')
                if mint and mint not in FILTERED_MINTS and mint not in pre_mints:
                    token_mint = mint
                    break

            # Fallback: find any non-filtered mint
            if not token_mint:
                for balance in post_balances:
                    mint = balance.get('mint')
                    if mint and mint not in FILTERED_MINTS:
                        token_mint = mint
                        break

            # Try to extract liquidity from balance changes
            if token_mint:
                for balance in post_balances:
                    if balance.get('mint') == WSOL_MINT:
                        amount = balance.get('uiTokenAmount', {}).get('uiAmount')
                        if amount:
                            liquidity = float(amount)
                            break

            # Try to find pool address from account keys or inner instructions
            message = tx_result.get('transaction', {}).get('message', {})
            account_keys = message.get('accountKeys', [])

            # Pool address is often the 3rd-5th account in Raydium txs
            if source == PoolSource.RAYDIUM_V4 and len(account_keys) > 4:
                for i in range(2, min(6, len(account_keys))):
                    key = account_keys[i]
                    if isinstance(key, dict):
                        key = key.get('pubkey')
                    if key and key != token_mint and key != WSOL_MINT:
                        pool_address = key
                        break

            # Try inner instructions for createAccount
            inner_instructions = meta.get('innerInstructions', [])
            for inner in inner_instructions:
                for ix in inner.get('instructions', []):
                    if isinstance(ix, dict):
                        if ix.get('program') == 'system':
                            parsed = ix.get('parsed', {})
                            if parsed.get('type') == 'createAccount':
                                new_account = parsed.get('info', {}).get('newAccount')
                                if new_account:
                                    pool_address = new_account
                                    break

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")

        return token_mint, pool_address, liquidity

    def _cleanup_signatures(self, source: PoolSource):
        """Clean up old signatures to prevent memory bloat"""
        sigs = self.known_signatures[source]
        if len(sigs) > self._max_signatures:
            # Keep most recent half
            self.known_signatures[source] = set(list(sigs)[-self._max_signatures//2:])

    async def _handle_rate_limit(self):
        """Handle rate limiting by backing off and trying to rotate RPC"""
        logger.warning("⚠️ Rate limited - backing off...")

        try:
            from config.rpc_provider import RPCProvider
            await RPCProvider.report_rate_limit('SOLANA_RPC', self.rpc_url, 300)

            new_url = await RPCProvider.get_rpc('SOLANA_RPC')
            if new_url and new_url != self.rpc_url:
                self.rpc_url = new_url
                logger.info(f"🔄 Rotated to new RPC endpoint")
        except Exception:
            pass

        await asyncio.sleep(60)

    async def _log_stats_if_needed(self):
        """Log detection stats periodically"""
        now = datetime.now()
        if now - self._stats['last_log_time'] >= self._log_interval:
            polls = self._stats['polls']
            detected = self._stats['pools_detected']
            queued = self._stats['pools_queued']
            api_calls = self._stats['api_calls']
            errors = self._stats['errors']

            # Log by source
            source_stats = ", ".join([
                f"{s}: {c}" for s, c in self._stats['by_source'].items() if c > 0
            ]) or "No pools detected"

            logger.info(
                f"📊 Solana Listener Stats (Last 5 min):\n"
                f"   Polls: {polls} | API Calls: {api_calls} | Errors: {errors}\n"
                f"   Pools Detected: {detected} | Queued: {queued}\n"
                f"   WSS Rejections — "
                f"Init: {self._stats.get('rejected_not_init', 0)} | "
                f"NoMint: {self._stats.get('rejected_no_mint', 0)} | "
                f"FilteredMint: {self._stats.get('rejected_filtered_mint', 0)} | "
                f"BadMint: {self._stats.get('rejected_bad_mint_format', 0)} | "
                f"TxFailed: {self._stats.get('rejected_tx_failed', 0)} | "
                f"NoResult: {self._stats.get('rejected_no_result', 0)} | "
                f"RPCErr: {self._stats.get('rejected_rpc_error', 0)}\n"
                f"   By Source: {source_stats}"
            )

            if detected == 0:
                logger.warning("   ⚠️ No pools detected - verify RPC is working and AMMs are active")

            # Reset stats (WSS counters + rejection counters preserved across
            # windows so cumulative profile is queryable from sniper_runtime_stats)
            self._stats = {
                'polls': 0,
                'pools_detected': 0,
                'pools_queued': 0,
                'api_calls': 0,
                'errors': 0,
                'last_log_time': now,
                'by_source': {source.value: 0 for source in PoolSource},
                'wss_connects': self._stats.get('wss_connects', 0),
                'wss_notifications': self._stats.get('wss_notifications', 0),
                'wss_pools_queued': self._stats.get('wss_pools_queued', 0),
                'rejected_rpc_error': self._stats.get('rejected_rpc_error', 0),
                'rejected_no_result': self._stats.get('rejected_no_result', 0),
                'rejected_tx_failed': self._stats.get('rejected_tx_failed', 0),
                'rejected_not_init': self._stats.get('rejected_not_init', 0),
                'rejected_no_mint': self._stats.get('rejected_no_mint', 0),
                'rejected_filtered_mint': self._stats.get('rejected_filtered_mint', 0),
                'rejected_bad_mint_format': self._stats.get('rejected_bad_mint_format', 0),
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

        if self.wss_task and not self.wss_task.done():
            self.wss_task.cancel()
            try:
                await self.wss_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()

        logger.info("🛑 Solana Listener closed")
