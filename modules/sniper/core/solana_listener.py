"""
Solana Listener for Sniper Module
Listens for new Raydium/Pump.fun Liquidity Pools via WebSocket subscription.

Supports:
- Raydium V4 AMM new pool detection
- Pump.fun bonding curve graduation
- Real-time WebSocket subscription
"""

import logging
import asyncio
from typing import List, Dict, Optional, Set
import aiohttp
import json
import os
from datetime import datetime, timedelta
import base64
import struct

logger = logging.getLogger("SolanaListener")

# Program IDs
RAYDIUM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CPMM_PROGRAM_ID = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# WSOL mint address
WSOL_MINT = "So11111111111111111111111111111111111111112"

# Raydium pool initialization discriminator (first 8 bytes of instruction data)
RAYDIUM_INIT_DISCRIMINATOR = bytes([0x00])  # Simplified - real is more complex


class SolanaListener:
    """
    Real-time Solana listener using WebSocket subscriptions.
    Detects new pool creations on Raydium and Pump.fun.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rpc_url = config.get('solana', {}).get('rpc_url')
        self.ws_url = config.get('solana', {}).get('ws_url')

        if not self.rpc_url:
            self.rpc_url = os.getenv('SOLANA_RPC_URL')

        if not self.ws_url:
            # Convert HTTP URL to WebSocket URL
            if self.rpc_url:
                self.ws_url = self.rpc_url.replace('https://', 'wss://').replace('http://', 'ws://')

        self.helius_api_key = os.getenv('HELIUS_API_KEY')

        # State
        self.known_signatures: Set[str] = set()
        self.new_pools_queue: asyncio.Queue = asyncio.Queue()
        self.ws_connected = False
        self.ws_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Rate-limited logging to reduce log spam
        self._stats = {
            'potential_pools_detected': 0,
            'pools_queued': 0,
            'pools_failed_fetch': 0,
            'last_log_time': datetime.now()
        }
        self._log_interval = timedelta(minutes=5)  # Log stats every 5 minutes

    async def initialize(self):
        """Initialize the Solana listener"""
        logger.info(f"🔌 Initializing Solana Listener...")
        logger.info(f"   RPC: {self.rpc_url}")
        logger.info(f"   WebSocket: {self.ws_url}")

        if not self.rpc_url:
            logger.error("❌ No RPC URL provided for Solana Listener")
            return

        self.is_running = True

        # Start WebSocket subscription in background
        self.ws_task = asyncio.create_task(self._run_websocket_listener())
        logger.info("✅ Solana Listener initialized with WebSocket subscription")

    async def get_new_pools(self) -> List[Dict]:
        """
        Get newly detected pools from the queue.
        Non-blocking - returns empty list if no new pools.
        """
        new_pools = []

        # Drain the queue
        while not self.new_pools_queue.empty():
            try:
                pool = self.new_pools_queue.get_nowait()
                new_pools.append(pool)
            except asyncio.QueueEmpty:
                break

        return new_pools

    async def _run_websocket_listener(self):
        """Main WebSocket listener loop with reconnection logic"""
        reconnect_delay = 1

        while self.is_running:
            try:
                await self._connect_and_subscribe()
                reconnect_delay = 1  # Reset delay on successful connection
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.ws_connected = False

            if self.is_running:
                logger.info(f"Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff, max 60s

    async def _connect_and_subscribe(self):
        """Connect to WebSocket and subscribe to program logs"""
        if not self.ws_url:
            logger.error("No WebSocket URL configured")
            await asyncio.sleep(30)
            return

        logger.info(f"🔌 Connecting to Solana WebSocket...")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url, heartbeat=30) as ws:
                self.ws_connected = True
                logger.info("✅ WebSocket connected")

                # Subscribe to Raydium V4 program logs
                await self._subscribe_to_program(ws, RAYDIUM_V4_PROGRAM_ID, "raydium_v4")

                # Subscribe to Raydium CPMM program logs
                await self._subscribe_to_program(ws, RAYDIUM_CPMM_PROGRAM_ID, "raydium_cpmm")

                # Subscribe to Pump.fun program logs (optional)
                # await self._subscribe_to_program(ws, PUMP_FUN_PROGRAM_ID, "pump_fun")

                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_ws_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed")
                        break

    async def _subscribe_to_program(self, ws, program_id: str, label: str):
        """Subscribe to logs for a specific program"""
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": hash(program_id) % 10000,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [program_id]},
                {"commitment": "processed"}
            ]
        }

        await ws.send_str(json.dumps(subscribe_msg))
        logger.info(f"📡 Subscribed to {label} logs ({program_id[:8]}...)")

    async def _handle_ws_message(self, data: str):
        """Handle incoming WebSocket message"""
        try:
            msg = json.loads(data)

            # Check for subscription confirmation
            if 'result' in msg and isinstance(msg['result'], int):
                logger.debug(f"Subscription confirmed: {msg['result']}")
                return

            # Check for log notification
            if msg.get('method') == 'logsNotification':
                await self._process_log_notification(msg)

        except json.JSONDecodeError:
            logger.debug(f"Invalid JSON received: {data[:100]}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _process_log_notification(self, msg: Dict):
        """Process a log notification for potential new pool"""
        try:
            result = msg.get('params', {}).get('result', {})
            value = result.get('value', {})
            signature = value.get('signature')
            logs = value.get('logs', [])
            err = value.get('err')

            # Skip failed transactions
            if err is not None:
                return

            # Skip already processed signatures
            if signature in self.known_signatures:
                return

            # Check logs for pool initialization
            is_pool_init = self._check_for_pool_init(logs)

            if is_pool_init:
                # Increment stats counter instead of logging every detection
                self._stats['potential_pools_detected'] += 1
                self.known_signatures.add(signature)

                # Log at DEBUG level to reduce spam
                logger.debug(f"Potential new pool detected: {signature[:16]}...")

                # Fetch full transaction details to extract token addresses
                pool_info = await self._fetch_pool_details(signature)

                if pool_info:
                    self._stats['pools_queued'] += 1
                    await self.new_pools_queue.put(pool_info)
                    # Log at DEBUG level to reduce spam
                    logger.debug(f"New pool queued: {pool_info.get('token_address', 'unknown')[:16]}...")
                else:
                    self._stats['pools_failed_fetch'] += 1

                # Log summary stats periodically (every 5 minutes)
                await self._log_stats_if_needed()

        except Exception as e:
            logger.error(f"Error processing log notification: {e}")

    async def _log_stats_if_needed(self):
        """Log detection stats periodically to reduce log spam"""
        now = datetime.now()
        if now - self._stats['last_log_time'] >= self._log_interval:
            detected = self._stats['potential_pools_detected']
            queued = self._stats['pools_queued']
            failed = self._stats['pools_failed_fetch']

            if detected > 0:  # Only log if there was activity
                logger.info(f"📊 Solana Pool Detection (Last 5 min): "
                           f"Detected: {detected} | Queued: {queued} | Failed: {failed}")

            # Reset stats
            self._stats = {
                'potential_pools_detected': 0,
                'pools_queued': 0,
                'pools_failed_fetch': 0,
                'last_log_time': now
            }

    def _check_for_pool_init(self, logs: List[str]) -> bool:
        """Check if logs indicate a new pool initialization"""
        init_keywords = [
            'initialize2',       # Raydium V4
            'Initialize',        # Generic
            'InitializePool',    # Some DEXs
            'create_pool',       # Pump.fun style
            'Program log: init', # Generic init
        ]

        for log in logs:
            for keyword in init_keywords:
                if keyword.lower() in log.lower():
                    return True

        return False

    async def _fetch_pool_details(self, signature: str) -> Optional[Dict]:
        """Fetch transaction details and extract pool/token info"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTransaction",
                    "params": [
                        signature,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0,
                            "commitment": "confirmed"
                        }
                    ]
                }

                async with session.post(self.rpc_url, json=payload, timeout=10) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    result = data.get('result')

                    if not result:
                        return None

                    # Parse transaction to find token mints
                    token_mint, pool_address = self._parse_pool_transaction(result)

                    if token_mint and token_mint != WSOL_MINT:
                        return {
                            'token_address': token_mint,
                            'pair_address': pool_address or signature,
                            'chain': 'solana',
                            'timestamp': datetime.utcnow().isoformat(),
                            'signature': signature,
                            'source': 'raydium_websocket'
                        }

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching transaction: {signature}")
        except Exception as e:
            logger.debug(f"Error fetching pool details: {e}")

        return None

    def _parse_pool_transaction(self, tx_result: Dict) -> tuple:
        """Parse transaction to extract token mint and pool address"""
        token_mint = None
        pool_address = None

        try:
            meta = tx_result.get('meta', {})
            transaction = tx_result.get('transaction', {})
            message = transaction.get('message', {})

            # Check account keys for potential token mints
            account_keys = message.get('accountKeys', [])

            # Look through post token balances for new token
            post_balances = meta.get('postTokenBalances', [])
            pre_balances = meta.get('preTokenBalances', [])

            # Find mints that appear in post but not in pre (new positions)
            pre_mints = {b.get('mint') for b in pre_balances}

            for balance in post_balances:
                mint = balance.get('mint')
                # Skip WSOL
                if mint and mint != WSOL_MINT and mint not in pre_mints:
                    # This is likely the new token being added to pool
                    token_mint = mint
                    break

            # If we didn't find new token, look for non-WSOL mint in post balances
            if not token_mint:
                for balance in post_balances:
                    mint = balance.get('mint')
                    if mint and mint != WSOL_MINT:
                        token_mint = mint
                        break

            # Try to find pool address from inner instructions
            inner_instructions = meta.get('innerInstructions', [])
            for inner in inner_instructions:
                instructions = inner.get('instructions', [])
                for ix in instructions:
                    if isinstance(ix, dict):
                        program = ix.get('program')
                        if program == 'system' and ix.get('parsed', {}).get('type') == 'createAccount':
                            # This could be the pool account
                            pool_address = ix.get('parsed', {}).get('info', {}).get('newAccount')
                            break

        except Exception as e:
            logger.debug(f"Error parsing transaction: {e}")

        return token_mint, pool_address

    async def close(self):
        """Clean up resources"""
        self.is_running = False
        self.ws_connected = False

        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Solana Listener closed")


# Alternative: Helius-based listener for better reliability
class HeliusSolanaListener(SolanaListener):
    """
    Solana listener using Helius webhooks for more reliable detection.
    Requires HELIUS_API_KEY environment variable.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.helius_ws_url = None

        if self.helius_api_key:
            self.helius_ws_url = f"wss://atlas-mainnet.helius-rpc.com?api-key={self.helius_api_key}"

    async def initialize(self):
        """Initialize with Helius WebSocket"""
        if self.helius_ws_url:
            self.ws_url = self.helius_ws_url
            logger.info("🚀 Using Helius WebSocket for enhanced reliability")

        await super().initialize()
