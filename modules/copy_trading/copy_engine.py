"""
Copy Trading Engine - Wallet Tracking (EVM + Solana)
"""
import asyncio
import logging
import json
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
        self.solana_rpc_url = os.getenv('SOLANA_RPC_URL')
        self.helius_api_key = os.getenv('HELIUS_API_KEY')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Track known transactions to avoid duplicates
        self._known_tx_hashes = set()
        self._known_solana_sigs = set()

    async def run(self):
        self.is_running = True
        logger.info("👯 Copy Trading Engine Started")
        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   EVM monitoring: {'Enabled' if self.etherscan_api_key else 'Disabled (no ETHERSCAN_API_KEY)'}")
        logger.info(f"   Solana monitoring: {'Enabled' if self.solana_rpc_url else 'Disabled (no SOLANA_RPC_URL)'}")

        # Initial load of settings
        await self._load_settings()

        cycle_count = 0
        evm_trades_copied = 0
        sol_trades_copied = 0

        while self.is_running:
            try:
                cycle_count += 1
                # Reload settings periodically to catch updates
                await self._load_settings()

                if self.targets:
                    # Monitor EVM wallets
                    evm_copied = await self._monitor_evm_wallets()
                    evm_trades_copied += evm_copied

                    # Monitor Solana wallets
                    sol_copied = await self._monitor_solana_wallets()
                    sol_trades_copied += sol_copied

                # Log status every 20 cycles (5 minutes at 15s interval)
                if cycle_count % 20 == 0:
                    logger.info(f"👯 Status: {cycle_count} cycles, {len(self.targets)} wallets tracked, {evm_trades_copied} EVM + {sol_trades_copied} Solana trades copied")

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
                                # Try parsing as JSON first
                                parsed = json.loads(val)
                                if isinstance(parsed, list):
                                    targets_loaded = [str(t).strip() for t in parsed if t]
                                else:
                                    targets_loaded = [val.strip()] if val.strip() else []
                            except json.JSONDecodeError:
                                try:
                                    # Try parsing as a list structure (e.g. "['0x1', '0x2']")
                                    parsed = ast.literal_eval(val)
                                    if isinstance(parsed, list):
                                        targets_loaded = [str(t).strip() for t in parsed if t]
                                    else:
                                        targets_loaded = [val.strip()]
                                except (ValueError, SyntaxError):
                                    # Fallback for comma/newline separated string
                                    targets_loaded = [t.strip() for t in val.replace(',', '\n').split('\n') if t.strip()]

                if targets_loaded != self.targets:
                    self.targets = targets_loaded
                    if self.targets:
                        logger.info(f"👯 Loaded {len(self.targets)} target wallets")

        except Exception as e:
            logger.warning(f"Failed to load Copy Trading settings: {e}")

    def _is_solana_address(self, address: str) -> bool:
        """Check if address is a Solana address (base58, typically 32-44 chars)"""
        # Solana addresses are base58 encoded, 32-44 chars, no 0x prefix
        if address.startswith('0x'):
            return False
        if len(address) < 32 or len(address) > 44:
            return False
        # Basic base58 character check
        base58_chars = set('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')
        return all(c in base58_chars for c in address)

    async def _monitor_evm_wallets(self) -> int:
        """Check for new transactions from target EVM wallets via Etherscan"""
        trades_copied = 0

        if not self.etherscan_api_key:
            return 0

        evm_wallets = [w for w in self.targets if w.startswith('0x')]
        if not evm_wallets:
            return 0

        for wallet in evm_wallets:
            try:
                url = f"https://api.etherscan.io/api?module=account&action=txlist&address={wallet}&startblock=0&endblock=99999999&sort=desc&apikey={self.etherscan_api_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        data = await resp.json()
                        if data['status'] == '1' and data.get('result'):
                            latest_tx = data['result'][0]
                            tx_hash = latest_tx.get('hash')

                            # Skip if already processed
                            if tx_hash in self._known_tx_hashes:
                                continue

                            # Check if recent (last minute)
                            import time
                            if int(latest_tx['timeStamp']) > time.time() - 60:
                                self._known_tx_hashes.add(tx_hash)
                                if await self._analyze_and_copy_evm(latest_tx):
                                    trades_copied += 1
            except Exception as e:
                logger.debug(f"Failed to check EVM wallet {wallet}: {e}")

        return trades_copied

    async def _monitor_solana_wallets(self) -> int:
        """Check for new transactions from target Solana wallets"""
        trades_copied = 0

        if not self.solana_rpc_url:
            return 0

        sol_wallets = [w for w in self.targets if self._is_solana_address(w)]
        if not sol_wallets:
            return 0

        for wallet in sol_wallets:
            try:
                # Use getSignaturesForAddress to get recent transactions
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [wallet, {"limit": 5}]
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(self.solana_rpc_url, json=payload) as resp:
                        if resp.status != 200:
                            continue

                        data = await resp.json()
                        signatures = data.get('result', [])

                        for sig_info in signatures:
                            sig = sig_info.get('signature')
                            if not sig or sig in self._known_solana_sigs:
                                continue

                            # Check if transaction was successful and recent
                            if sig_info.get('err') is not None:
                                continue

                            # Check block time (within last 2 minutes)
                            block_time = sig_info.get('blockTime', 0)
                            import time
                            if block_time and block_time > time.time() - 120:
                                self._known_solana_sigs.add(sig)

                                # Analyze the transaction
                                if await self._analyze_and_copy_solana(wallet, sig):
                                    trades_copied += 1

            except Exception as e:
                logger.debug(f"Failed to check Solana wallet {wallet}: {e}")

        return trades_copied

    async def _analyze_and_copy_evm(self, tx) -> bool:
        """Analyze EVM transaction and execute copy if it's a swap"""
        try:
            input_data = tx.get('input', '')
            if len(input_data) < 10:
                return False

            method_id = input_data[:10]
            # Common DEX Router methods
            SWAP_METHODS = {
                '0x7ff36ab5': 'swapExactETHForTokens',
                '0xb6f9de95': 'swapExactETHForTokensSupportingFeeOnTransferTokens',
                '0x18cbafe5': 'swapExactTokensForETH',
                '0x38ed1739': 'swapExactTokensForTokens',
                '0x5c11d795': 'swapExactTokensForTokensSupportingFeeOnTransferTokens',
            }

            if method_id in SWAP_METHODS:
                method_name = SWAP_METHODS[method_id]
                logger.info(f"👯 EVM COPY TRIGGER: Wallet {tx['from']} executed {method_name}")
                await self._execute_evm_copy_trade(tx, method_name)
                return True

            return False
        except Exception as e:
            logger.error(f"Error analyzing EVM tx: {e}")
            return False

    async def _execute_evm_copy_trade(self, source_tx, method_name: str):
        """Execute the same trade on EVM"""
        tx_hash = source_tx.get('hash', 'unknown')
        logger.info(f"🚀 Copying EVM trade {tx_hash} ({method_name})")

        if self.dry_run:
            # Simulated Execution
            await asyncio.sleep(0.5)
            logger.info(f"✅ EVM Copy Trade Simulated: {tx_hash}")
        else:
            # TODO: Implement actual trade execution
            logger.warning("Live EVM copy trading not yet implemented")

    async def _analyze_and_copy_solana(self, wallet: str, signature: str) -> bool:
        """Analyze Solana transaction and execute copy if it's a swap"""
        try:
            # Get transaction details
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.solana_rpc_url, json=payload) as resp:
                    if resp.status != 200:
                        return False

                    data = await resp.json()
                    tx = data.get('result')
                    if not tx:
                        return False

                    # Check if it's a swap transaction
                    # Look for Jupiter, Raydium, or other DEX programs
                    DEX_PROGRAMS = [
                        'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',   # Jupiter v6
                        'JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB',   # Jupiter v4
                        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8', # Raydium V4
                        'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  # Raydium CPMM
                    ]

                    # Get instructions from transaction
                    message = tx.get('transaction', {}).get('message', {})
                    instructions = message.get('instructions', [])

                    is_swap = False
                    for instr in instructions:
                        program_id = instr.get('programId', '')
                        if program_id in DEX_PROGRAMS:
                            is_swap = True
                            break

                    if is_swap:
                        logger.info(f"👯 SOLANA COPY TRIGGER: Wallet {wallet} executed swap {signature[:20]}...")
                        await self._execute_solana_copy_trade(wallet, signature, tx)
                        return True

                    return False

        except Exception as e:
            logger.error(f"Error analyzing Solana tx: {e}")
            return False

    async def _execute_solana_copy_trade(self, wallet: str, signature: str, tx_data: dict):
        """Execute the same trade on Solana"""
        logger.info(f"🚀 Copying Solana trade {signature[:20]}...")

        if self.dry_run:
            # Simulated Execution
            await asyncio.sleep(0.5)
            logger.info(f"✅ Solana Copy Trade Simulated: {signature[:20]}...")
        else:
            # TODO: Implement actual Solana trade execution via Jupiter
            logger.warning("Live Solana copy trading not yet implemented")

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Copy Trading Engine Stopped")
