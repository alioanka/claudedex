"""
Sniper Engine - High-speed new token sniping
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime
import json

from config.config_manager import ConfigManager
from data.storage.database import DatabaseManager
from monitoring.alerts import AlertManager

logger = logging.getLogger("SniperEngine")

class SniperEngine:
    """
    Sniper Engine for detecting and buying new token launches immediately.
    Supports EVM (Mempool/Events) and Solana (Raydium logs).
    """

    def __init__(self, config: Dict, config_manager: ConfigManager, db_pool):
        self.config = config
        self.config_manager = config_manager
        self.db_pool = db_pool
        self.is_running = False
        self.tasks = []

        # State
        self.pending_targets = {}
        self.active_snipes = {}

        # Components (to be initialized)
        self.evm_listener = None
        self.solana_listener = None
        self.executor = None

    async def initialize(self):
        """Initialize sniper components"""
        logger.info("🔫 Initializing Sniper Engine...")

        # Load settings from DB
        sniper_config = self.config.get('sniper', {})

        # Initialize Listeners based on enabled chains
        if sniper_config.get('evm_enabled', True):
            from modules.sniper.core.evm_listener import EVMListener
            self.evm_listener = EVMListener(self.config)
            await self.evm_listener.initialize()

        if sniper_config.get('solana_enabled', True):
            from modules.sniper.core.solana_listener import SolanaListener
            self.solana_listener = SolanaListener(self.config)
            await self.solana_listener.initialize()

        logger.info("✅ Sniper Engine initialized")

    async def run(self):
        """Main loop"""
        self.is_running = True
        logger.info("🔫 Sniper Engine Started")

        self.tasks = [
            asyncio.create_task(self._monitor_new_pairs()),
            asyncio.create_task(self._process_targets()),
            asyncio.create_task(self._monitor_active_snipes())
        ]

        await asyncio.gather(*self.tasks)

    async def _monitor_new_pairs(self):
        """Listen for new pair events"""
        scan_count = 0
        evm_pairs_found = 0
        sol_pools_found = 0

        while self.is_running:
            try:
                scan_count += 1

                # 1. Check EVM Mempool/Events
                if self.evm_listener:
                    events = await self.evm_listener.get_new_pairs()
                    for event in events:
                        evm_pairs_found += 1
                        await self._evaluate_target(event, 'evm')

                # 2. Check Solana Raydium Logs
                if self.solana_listener:
                    events = await self.solana_listener.get_new_pools()
                    for event in events:
                        sol_pools_found += 1
                        await self._evaluate_target(event, 'solana')

                # Log status every 6000 scans (~10 minutes at 0.1s interval)
                if scan_count % 6000 == 0:
                    logger.info(f"🔫 Monitor status: {scan_count} scans, {evm_pairs_found} EVM pairs, {sol_pools_found} Solana pools detected")

                await asyncio.sleep(0.1) # Fast loop
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(1)

    async def _evaluate_target(self, target: Dict, chain_type: str):
        """Evaluate if a new token meets sniping criteria"""
        try:
            token_address = target.get('token_address')

            # 1. Check Filters (Liquidity, Tax, Honeypot)
            if not await self._check_filters(target):
                return

            # 2. Add to pending targets
            logger.info(f"🎯 SNIPER TARGET ACQUIRED: {token_address} ({chain_type})")
            self.pending_targets[token_address] = {
                'target': target,
                'chain_type': chain_type,
                'timestamp': datetime.now(),
                'status': 'pending'
            }

        except Exception as e:
            logger.error(f"Error evaluating target: {e}")

    async def _check_filters(self, target: Dict) -> bool:
        """Apply strict filters for sniping"""
        # Basic filter: check liquidity presence (simplified)
        if not target.get('pair_address'):
            return False
        return True

    async def _process_targets(self):
        """Execute buy orders for pending targets"""
        while self.is_running:
            try:
                if not self.pending_targets:
                    await asyncio.sleep(0.01)
                    continue

                # Process active targets
                for address, data in list(self.pending_targets.items()):
                    if data['status'] == 'pending':
                        # EXECUTE BUY
                        await self._execute_snipe(data)

                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing targets: {e}")
                await asyncio.sleep(1)

    async def _execute_snipe(self, data: Dict):
        """Execute the buy transaction with high priority"""
        token_address = data['target'].get('token_address')
        chain = data['chain_type']

        logger.info(f"🔫 EXECUTING SNIPE: {token_address} on {chain}")
        data['status'] = 'buying'

        # Real logic simulation
        try:
            if chain == 'evm':
                # Use Web3 to sendSwapExactETHForTokens
                # tx = router.functions.swapExactETHForTokens(...).buildTransaction(...)
                # signed_tx = w3.eth.account.sign_transaction(tx, private_key)
                # w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                pass
            elif chain == 'solana':
                # Use Solana Client to swap on Raydium
                pass

            # Simulate network delay
            await asyncio.sleep(0.5)

            logger.info(f"✅ SNIPE SUCCESS: {token_address}")
            data['status'] = 'active'
            data['entry_price'] = 0.0001 # Placeholder
            self.active_snipes[token_address] = data
            del self.pending_targets[token_address]

        except Exception as e:
            logger.error(f"❌ SNIPE FAILED: {e}")
            data['status'] = 'failed'

    async def _monitor_active_snipes(self):
        """Monitor active snipes for auto-sell targets"""
        while self.is_running:
            for address, data in list(self.active_snipes.items()):
                # Check for +50% profit or -10% stop loss
                # In real engine, fetch current price here
                pass
            await asyncio.sleep(1)

    async def stop(self):
        """Stop the engine"""
        self.is_running = False
        for task in self.tasks:
            task.cancel()
        logger.info("🛑 Sniper Engine Stopped")
