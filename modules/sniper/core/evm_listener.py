"""
EVM Listener for Sniper Module
Listens for PairCreated events (Uniswap V2) and Mempool 'addLiquidity' transactions.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from web3 import Web3
from web3.contract import Contract
from eth_abi import decode
from eth_utils import event_abi_to_log_topic

from config.rpc_provider import RPCProvider

logger = logging.getLogger("EVMListener")

# Minimal ABI for Uniswap V2 Factory PairCreated event
FACTORY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "token0", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "token1", "type": "address"},
            {"indexed": False, "internalType": "address", "name": "pair", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "param3", "type": "uint256"}
        ],
        "name": "PairCreated",
        "type": "event"
    }
]

# Common Factory Addresses (Mainnet)
FACTORIES = {
    'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
    'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
    # Add Base/Arbitrum factories here as needed
}

# Well-known QUOTE tokens per chain. New pairs are almost always
# NEW_TOKEN / QUOTE_TOKEN; PairCreated emits token0/token1 sorted
# by address so the QUOTE side is sometimes token0. _select_target_token
# picks the non-quote side using this table.
#
# Chain IDs: 1 = Ethereum, 56 = BSC, 137 = Polygon, 8453 = Base,
# 42161 = Arbitrum, 10 = Optimism, 43114 = Avalanche.
EVM_QUOTE_TOKENS_BY_CHAIN = {
    1: frozenset([  # Ethereum mainnet
        '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',  # WETH
        '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
        '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
    ]),
    56: frozenset([  # BSC
        '0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c',  # WBNB
        '0x55d398326f99059ff775485246999027b3197955',  # USDT (BSC)
        '0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d',  # USDC (BSC)
        '0xe9e7cea3dedca5984780bafc599bd69add087d56',  # BUSD
    ]),
    137: frozenset([  # Polygon
        '0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270',  # WMATIC
        '0xc2132d05d31c914a87c6611c10748aeb04b58e8f',  # USDT (Polygon)
        '0x2791bca1f2de4661ed88a30c99a7a9449aa84174',  # USDC.e (Polygon)
        '0x3c499c542cef5e3811e1192ce70d8cc03d5c3359',  # USDC native (Polygon)
        '0x8f3cf7ad23cd3cadbd9735aff958023239c6a063',  # DAI (Polygon)
    ]),
    8453: frozenset([  # Base
        '0x4200000000000000000000000000000000000006',  # WETH (Base)
        '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC (Base)
        '0x50c5725949a6f0c72e6c4a641f24049a917db0cb',  # DAI (Base)
    ]),
    42161: frozenset([  # Arbitrum
        '0x82af49447d8a07e3bd95bd0d56f35241523fbab1',  # WETH (Arb)
        '0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9',  # USDT (Arb)
        '0xaf88d065e77c8cc2239327c5edb3a432268e5831',  # USDC native (Arb)
        '0xff970a61a04b1ca14834a43f5de4533ebddb5cc8',  # USDC.e (Arb)
        '0xda10009cbd5d07dd0cecc66161fc93d7c9000da1',  # DAI (Arb)
    ]),
    10: frozenset([  # Optimism
        '0x4200000000000000000000000000000000000006',  # WETH (OP)
        '0x94b008aa00579c1307b0ef2c499ad98a8ce58e58',  # USDT (OP)
        '0x0b2c639c533813f4aa9d7837caf62653d097ff85',  # USDC (OP)
        '0xda10009cbd5d07dd0cecc66161fc93d7c9000da1',  # DAI (OP)
    ]),
    43114: frozenset([  # Avalanche
        '0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7',  # WAVAX
        '0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7',  # USDT (Avax)
        '0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e',  # USDC native (Avax)
    ]),
}

# Default fallback (mainnet) used when the connected chain isn't in the dict.
_DEFAULT_QUOTE_TOKENS = EVM_QUOTE_TOKENS_BY_CHAIN[1]


def _select_target_token(token0: str, token1: str, chain_id: int = 1) -> str:
    """Pick the non-quote side of a PairCreated event using the per-chain
    quote-token set. Falls back to mainnet WETH/USDT/USDC/DAI if the chain
    isn't in our table (better than returning token0 blindly)."""
    quote_set = EVM_QUOTE_TOKENS_BY_CHAIN.get(chain_id, _DEFAULT_QUOTE_TOKENS)
    t0 = (token0 or '').lower()
    t1 = (token1 or '').lower()
    if t0 in quote_set and t1 not in quote_set:
        return token1
    if t1 in quote_set and t0 not in quote_set:
        return token0
    return token0  # fallback when both/neither are well-known

class EVMListener:
    def __init__(self, config: Dict):
        self.config = config
        self.w3: Optional[Web3] = None
        self.is_running = False
        self.is_configured = False  # Track if EVM is properly configured
        self.known_pairs = set()
        self.chain_id: Optional[int] = None

        # Get RPC URL from config, PoolEngine (sync ctor), .env preserved as ultimate fallback
        self.rpc_url = (
            config.get('web3', {}).get('provider_url')
            or RPCProvider.get_rpc_sync('ETHEREUM_RPC')
            or os.getenv('WEB3_PROVIDER_URL')
        )

        # Phase 1 EVM: SNIPER_EVM_LISTENER_MODE = 'polling' (default) | 'wss'
        # Separate from Solana's SNIPER_LISTENER_MODE so per-chain A/B works.
        self.listener_mode = os.getenv('SNIPER_EVM_LISTENER_MODE', 'polling').strip().lower()
        self.wss_url = os.getenv('SNIPER_EVM_WSS_URL', '').strip() or self._infer_wss_url()
        self._wss_queue: asyncio.Queue = asyncio.Queue()
        self.wss_task: Optional[asyncio.Task] = None
        self._stats = {
            'wss_connects': 0,
            'wss_log_notifications': 0,
            'wss_pairs_queued': 0,
            'polling_pairs_emitted': 0,
        }

    def _infer_wss_url(self) -> Optional[str]:
        """Derive WSS URL from HTTP RPC URL by swapping scheme. Most EVM
        RPC providers (Alchemy, Infura, QuickNode) serve WSS on the same
        host. Returns None if rpc_url is unset or non-derivable."""
        if not self.rpc_url:
            return None
        if self.rpc_url.startswith('https://'):
            return 'wss://' + self.rpc_url[len('https://'):]
        if self.rpc_url.startswith('http://'):
            return 'ws://' + self.rpc_url[len('http://'):]
        return None

    async def initialize(self):
        """Initialize Web3 connection"""
        logger.info("🔌 Initializing EVM Listener...")

        if not self.rpc_url:
            logger.warning("⚠️ EVM Listener: No RPC URL configured. Set WEB3_PROVIDER_URL in .env for EVM chain sniping.")
            logger.info("   Skipping EVM initialization - Solana-only mode active.")
            self.is_configured = False
            return

        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={'timeout': 10}))
            if self.w3.is_connected():
                chain_id = self.w3.eth.chain_id
                self.chain_id = chain_id  # stored for _select_target_token
                chain_name = {
                    1: 'Ethereum', 56: 'BSC', 137: 'Polygon', 8453: 'Base',
                    42161: 'Arbitrum', 10: 'Optimism', 43114: 'Avalanche',
                }.get(chain_id, f'Chain {chain_id}')
                logger.info(f"✅ Connected to EVM Node ({chain_name}): {self.rpc_url[:50]}...")
                self.is_configured = True
            else:
                logger.warning(f"⚠️ EVM Listener: Could not connect to RPC. Check WEB3_PROVIDER_URL.")
                logger.info("   Continuing without EVM support - Solana-only mode active.")
                self.is_configured = False
        except Exception as e:
            logger.warning(f"⚠️ EVM Listener initialization failed: {e}")
            logger.info("   Continuing without EVM support - Solana-only mode active.")
            self.is_configured = False

        if self.is_configured and self.listener_mode == 'wss':
            await self.start_wss_listener()

    async def start_wss_listener(self) -> None:
        """Spawn the WSS background task if listener_mode='wss' AND
        is_configured is True. No-op otherwise. Idempotent."""
        if self.listener_mode != 'wss':
            return
        if not self.is_configured:
            logger.warning("EVM listener not configured; WSS mode requires a working RPC connection first")
            return
        if not self.wss_url:
            logger.warning(
                "SNIPER_EVM_LISTENER_MODE=wss but no WSS URL available "
                "(set SNIPER_EVM_WSS_URL or use an https:// RPC); "
                "falling back to polling"
            )
            self.listener_mode = 'polling'
            return
        if self.wss_task and not self.wss_task.done():
            return
        self.is_running = True
        logger.info(f"📡 Starting EVM WSS listener: {self.wss_url[:60]}...")
        self.wss_task = asyncio.create_task(self._run_wss_listener())

    async def _run_wss_listener(self) -> None:
        """Phase 1 EVM WSS listener — eth_subscribe('logs') with PairCreated
        topic filter. Reconnects with capped exponential backoff (1s → 60s).
        Emits matching target dicts to self._wss_queue; get_new_pairs()
        drains the queue and (in WSS mode) also runs the polling backstop
        so missed events don't slip through during initial validation."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed — EVM WSS disabled")
            return

        backoff = 1.0
        backoff_max = 60.0
        sub_id_counter = 0
        # Pre-compute PairCreated topic hash once.
        try:
            if not self.w3:
                logger.error("EVM WSS: w3 instance not initialized")
                return
            topic_hash = self.w3.keccak(text="PairCreated(address,address,address,uint256)").hex()
            # eth_subscribe expects 0x-prefixed hash strings
            if not topic_hash.startswith('0x'):
                topic_hash = '0x' + topic_hash
        except Exception as e:
            logger.error(f"EVM WSS: failed to compute topic hash: {e}")
            return

        while self.is_running:
            try:
                async with websockets.connect(self.wss_url, ping_interval=20, ping_timeout=10) as ws:
                    self._stats['wss_connects'] += 1
                    backoff = 1.0
                    logger.info(
                        f"🔌 EVM WSS connected (#{self._stats['wss_connects']}); "
                        f"subscribing to PairCreated logs"
                    )

                    sub_id_counter += 1
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": sub_id_counter,
                        "method": "eth_subscribe",
                        "params": ["logs", {"topics": [topic_hash]}],
                    }))

                    async for raw in ws:
                        if not self.is_running:
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        if 'result' in msg and msg.get('id') == sub_id_counter:
                            logger.info(f"✅ EVM WSS subscription ack: {msg.get('result')}")
                            continue

                        if msg.get('method') != 'eth_subscription':
                            continue

                        self._stats['wss_log_notifications'] += 1
                        log = (msg.get('params') or {}).get('result')
                        if not isinstance(log, dict):
                            continue

                        # eth_subscribe 'logs' payload has the same shape as
                        # eth_getLogs items: topics, data, blockNumber, etc.
                        # Reuse _parse_log to extract pair/tokens.
                        try:
                            # Convert hex topics to bytes-like objects matching
                            # what _parse_log expects (HexBytes from web3.py).
                            from hexbytes import HexBytes
                            log_obj = dict(log)
                            log_obj['topics'] = [HexBytes(t) for t in (log.get('topics') or [])]
                            if 'data' in log_obj and isinstance(log_obj['data'], str):
                                log_obj['data'] = HexBytes(log_obj['data'])
                            parsed = self._parse_log(log_obj)
                        except Exception as e:
                            logger.debug(f"EVM WSS log parse failed: {e}")
                            continue

                        if not parsed:
                            continue
                        pair_addr = parsed.get('pair')
                        if not pair_addr or pair_addr in self.known_pairs:
                            continue
                        self.known_pairs.add(pair_addr)

                        bn_raw = log.get('blockNumber')
                        block_number = int(bn_raw, 16) if isinstance(bn_raw, str) else bn_raw
                        # EVM has no commitment-wait penalty (the log
                        # itself carries everything we need), but stamp
                        # rpc_receipt_perf anyway for cross-chain A/B
                        # symmetry with Solana.
                        rpc_receipt_perf = time.perf_counter()
                        target = {
                            'token_address': _select_target_token(parsed['token0'], parsed['token1'], self.chain_id or 1),
                            'pair_address': pair_addr,
                            'chain': 'ethereum',
                            'block_number': block_number,
                            'timestamp': datetime.utcnow().isoformat(),
                            'detection_path': 'wss',
                            'rpc_receipt_perf': rpc_receipt_perf,
                        }
                        self._stats['wss_pairs_queued'] += 1
                        try:
                            self._wss_queue.put_nowait(target)
                        except asyncio.QueueFull:
                            logger.warning("EVM WSS queue full; dropping pair")
                        logger.info(f"⚡ EVM WSS pair: {pair_addr[:16]}...")

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(
                    f"EVM WSS disconnect: {type(e).__name__}: {e}; reconnecting in {backoff:.1f}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * 2, backoff_max)

    async def get_new_pairs(self) -> List[Dict]:
        """Returns newly detected pair targets. In WSS mode, drains the
        WSS queue + still runs polling as backstop. In polling mode,
        polls eth.get_logs for the last 5 blocks."""
        if not self.is_configured or not self.w3 or not self.w3.is_connected():
            return []

        new_pairs: List[Dict] = []

        # WSS-mode: drain anything the background task has queued
        if self.listener_mode == 'wss':
            while True:
                try:
                    target = self._wss_queue.get_nowait()
                    new_pairs.append(target)
                except asyncio.QueueEmpty:
                    break

        # Polling path (always runs as backstop in WSS mode)
        try:
            current_block = self.w3.eth.block_number
            from_block = current_block - 5  # Scan last 5 blocks

            event_signature_hash = self.w3.keccak(text="PairCreated(address,address,address,uint256)").hex()
            logs = self.w3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': 'latest',
                'topics': [event_signature_hash]
            })
            # All logs in this batch share the same rpc_receipt: the
            # get_logs response moment.
            rpc_receipt_perf = time.perf_counter()

            for log in logs:
                pair_address = self._parse_log(log)
                if pair_address and pair_address['pair'] not in self.known_pairs:
                    self.known_pairs.add(pair_address['pair'])

                    target = {
                        'token_address': _select_target_token(pair_address['token0'], pair_address['token1'], self.chain_id or 1),
                        'pair_address': pair_address['pair'],
                        'chain': 'ethereum',
                        'block_number': log['blockNumber'],
                        'timestamp': datetime.utcnow().isoformat(),
                        'detection_path': 'polling',
                        'rpc_receipt_perf': rpc_receipt_perf,
                    }
                    self._stats['polling_pairs_emitted'] += 1
                    new_pairs.append(target)
                    logger.info(f"🆕 EVM polling pair: {target['pair_address']}")

        except Exception as e:
            logger.debug(f"Error polling EVM logs: {e}")  # Debug to avoid spamming if RPC is flaky

        return new_pairs

    def _parse_log(self, log):
        """Parse PairCreated log"""
        try:
            # Decode topics (indexed params)
            # topic0 is event sig, topic1 is token0, topic2 is token1
            if len(log['topics']) < 3:
                return None

            token0 = '0x' + log['topics'][1].hex()[-40:]
            token1 = '0x' + log['topics'][2].hex()[-40:]

            # Decode data (non-indexed params: pair address, uint)
            # pair address is first 32 bytes of data (padded), param3 is second
            data = log['data']
            # Using basic hex slicing as a fallback if ABI decoding fails or is overkill
            # eth_abi.decode(['address', 'uint256'], data) is the proper way

            # Minimal manual decode for robustness without heavy deps if needed,
            # but we use eth_abi here if installed
            try:
                decoded = decode(['address', 'uint256'], data)
                pair = decoded[0]
            except:
                # Fallback manual
                pair = '0x' + data.hex()[24:64]

            return {'token0': token0, 'token1': token1, 'pair': pair}
        except Exception as e:
            logger.error(f"Log parse error: {e}")
            return None
