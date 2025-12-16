"""
Solana Listener for Sniper Module
Listens for new Raydium Liquidity Pools via RPC logs.
"""

import logging
import asyncio
from typing import List, Dict
import aiohttp
import json
from datetime import datetime

logger = logging.getLogger("SolanaListener")

# Raydium Liquidity Pool V4 Program ID
RAYDIUM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

class SolanaListener:
    def __init__(self, config: Dict):
        self.config = config
        self.rpc_url = config.get('solana', {}).get('rpc_url')
        if not self.rpc_url:
            import os
            self.rpc_url = os.getenv('SOLANA_RPC_URL')

        self.known_signatures = set()

    async def initialize(self):
        logger.info(f"🔌 Initializing Solana Listener (RPC: {self.rpc_url})...")
        if not self.rpc_url:
            logger.error("❌ No RPC URL provided for Solana Listener")

    async def get_new_pools(self) -> List[Dict]:
        """
        Polls logs for 'initialize2' instruction on Raydium V4 program.
        """
        if not self.rpc_url:
            return []

        new_pools = []
        try:
            # We fetch recent signatures for the Raydium program
            # Then we'd check transaction details for 'initialize2' log
            # This is a simplified "Poll for signatures" approach

            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [
                        RAYDIUM_V4_PROGRAM_ID,
                        {"limit": 10}
                    ]
                }

                async with session.post(self.rpc_url, json=payload) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    signatures = data.get('result', [])

                    for item in signatures:
                        sig = item.get('signature')
                        if sig and sig not in self.known_signatures:
                            self.known_signatures.add(sig)

                            # In a full implementation, we would now:
                            # 1. getTransaction(sig)
                            # 2. Parse logs for "initialize2"
                            # 3. Extract token mints

                            # For this implementation, we mark it as a candidate if it's new
                            # Real sniper would need deeper parsing here
                            if item.get('err') is None:
                                # Mocking extraction for demonstration of flow
                                # In reality, we'd parse the 'innerInstructions' or 'logMessages'
                                target = {
                                    'token_address': 'So11111111111111111111111111111111111111112', # Placeholder for parsed mint
                                    'pair_address': sig, # Using sig as unique ID for now
                                    'chain': 'solana',
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'signature': sig
                                }
                                # new_pools.append(target)
                                # Commented out to prevent spamming "new pools" for every Raydium tx in logs
                                # Uncomment when deep parsing is added
                                pass

        except Exception as e:
            logger.debug(f"Error polling Solana logs: {e}")

        return new_pools
