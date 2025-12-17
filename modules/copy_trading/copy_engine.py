"""
Copy Trading Engine - Wallet Tracking
"""
import asyncio
import logging
from typing import Dict, List
import aiohttp
import os
import ast

logger = logging.getLogger("CopyTradingEngine")

class CopyTradingEngine:
    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.targets = []  # Initialize empty, load from DB
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')

    async def run(self):
        self.is_running = True
        logger.info("👯 Copy Trading Engine Started")

        # Initial load of settings
        await self._load_settings()

        while self.is_running:
            try:
                # Reload settings periodically to catch updates
                await self._load_settings()

                await self._monitor_wallets()
                await asyncio.sleep(15) # Poll every 15s
            except Exception as e:
                logger.error(f"Copy loop error: {e}")
                await asyncio.sleep(15)

    async def _load_settings(self):
        """Load Copy Trading settings from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT key, value FROM config_settings WHERE config_type = 'copytrading_config'")

                targets_loaded = []
                for row in rows:
                    key = row['key']
                    val = row['value']

                    if key == 'target_wallets':
                        if val:
                            try:
                                # Try parsing as a list structure (e.g. "['0x1', '0x2']")
                                if val.strip().startswith('[') and val.strip().endswith(']'):
                                    parsed = ast.literal_eval(val)
                                    if isinstance(parsed, list):
                                        targets_loaded = [str(t).strip() for t in parsed if t]
                                    else:
                                        # Fallback if not list
                                        targets_loaded = [val.strip()]
                                else:
                                    # Fallback for comma/newline separated string
                                    targets_loaded = [t.strip() for t in val.replace(',', '\n').split('\n') if t.strip()]
                            except Exception as e:
                                logger.warning(f"Error parsing target_wallets: {e}")
                                # Final fallback
                                targets_loaded = [t.strip() for t in val.replace(',', '\n').split('\n') if t.strip()]

                if targets_loaded:
                    self.targets = targets_loaded
                    # Only log if changed to avoid spam
                    # logger.debug(f"Loaded {len(self.targets)} target wallets")

        except Exception as e:
            logger.warning(f"Failed to load Copy Trading settings: {e}")

    async def _monitor_wallets(self):
        """Check for new transactions from target wallets via Etherscan"""
        if not self.etherscan_api_key:
            if not getattr(self, '_etherscan_warning_logged', False):
                logger.warning("No ETHERSCAN_API_KEY found. Wallet monitoring disabled.")
                self._etherscan_warning_logged = True
            return

        if not self.targets:
            return

        for wallet in self.targets:
            try:
                url = f"https://api.etherscan.io/api?module=account&action=txlist&address={wallet}&startblock=0&endblock=99999999&sort=desc&apikey={self.etherscan_api_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                        if data['status'] == '1':
                            latest_tx = data['result'][0]
                            # Check if recent (last minute)
                            if int(latest_tx['timeStamp']) > asyncio.get_event_loop().time() - 60:
                                await self._analyze_and_copy(latest_tx)
            except Exception as e:
                logger.debug(f"Failed to check wallet {wallet}: {e}")

    async def _analyze_and_copy(self, tx):
        """Analyze transaction and execute copy"""
        # Filter for Swap methods
        method_id = tx['input'][:10]
        # Common Uniswap Router methods (swapExactETHForTokens, etc.)
        SWAP_METHODS = ['0x7ff36ab5', '0xb6f9de95']

        if method_id in SWAP_METHODS:
            logger.info(f"👯 COPY TRIGGER: Wallet {tx['from']} executed swap {method_id}")
            await self._execute_copy_trade(tx)

    async def _execute_copy_trade(self, source_tx):
        """Execute the same trade"""
        logger.info(f"🚀 Copying trade {source_tx['hash']}")
        # Simulated Execution
        await asyncio.sleep(1)
        logger.info("✅ Copy Trade Executed (Simulated)")

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Copy Trading Engine Stopped")
