"""
EVM Listener for Sniper Module
Listens for PairCreated events (Uniswap V2) and Mempool 'addLiquidity' transactions.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
from web3 import Web3
from web3.contract import Contract
from eth_abi import decode
from eth_utils import event_abi_to_log_topic

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

class EVMListener:
    def __init__(self, config: Dict):
        self.config = config
        self.w3: Optional[Web3] = None
        self.is_running = False
        self.is_configured = False  # Track if EVM is properly configured
        self.known_pairs = set()

        # Get RPC URL from config or env
        self.rpc_url = config.get('web3', {}).get('provider_url')
        if not self.rpc_url:
            import os
            self.rpc_url = os.getenv('WEB3_PROVIDER_URL')

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
                chain_name = {1: 'Ethereum', 56: 'BSC', 8453: 'Base', 42161: 'Arbitrum'}.get(chain_id, f'Chain {chain_id}')
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

    async def get_new_pairs(self) -> List[Dict]:
        """
        Polls for new PairCreated events.
        In a production environment, this would use WebSocket subscription.
        Here we poll logs for simplicity and compatibility with HTTP.
        """
        if not self.is_configured or not self.w3 or not self.w3.is_connected():
            return []

        new_pairs = []
        try:
            # Get current block
            current_block = self.w3.eth.block_number
            from_block = current_block - 5  # Scan last 5 blocks

            # Define event signature
            event_signature_hash = self.w3.keccak(text="PairCreated(address,address,address,uint256)").hex()

            # Filter logs
            logs = self.w3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': 'latest',
                'topics': [event_signature_hash]
            })

            for log in logs:
                pair_address = self._parse_log(log)
                if pair_address and pair_address not in self.known_pairs:
                    self.known_pairs.add(pair_address)

                    # Construct target object
                    target = {
                        'token_address': pair_address['token0'], # Simplified: assume interesting token is token0 for now or fetch symbols
                        'pair_address': pair_address['pair'],
                        'chain': 'ethereum', # Dynamic based on config
                        'block_number': log['blockNumber'],
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    new_pairs.append(target)
                    logger.info(f"🆕 New EVM Pair Detected: {target['pair_address']}")

        except Exception as e:
            logger.debug(f"Error polling EVM logs: {e}") # Debug to avoid spamming if RPC is flaky

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
