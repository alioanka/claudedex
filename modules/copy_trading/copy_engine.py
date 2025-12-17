"""
Copy Trading Engine - Wallet Tracking (EVM + Solana)

Features:
- Real-time wallet monitoring on EVM and Solana
- Real trade execution (when DRY_RUN=false)
- Jupiter aggregator for Solana swaps
- Uniswap/DEX router for EVM swaps
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional
import aiohttp
import os
import ast
from datetime import datetime

logger = logging.getLogger("CopyTradingEngine")

# Jupiter API
JUPITER_QUOTE_API = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_API = "https://quote-api.jup.ag/v6/swap"

# Common tokens
WSOL_MINT = "So11111111111111111111111111111111111111112"


class CopyTradeExecutor:
    """Trade executor for Copy Trading module"""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.session: Optional[aiohttp.ClientSession] = None

        # Solana credentials
        self.solana_rpc_url = os.getenv('SOLANA_RPC_URL')
        self.solana_private_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY')
        self.solana_wallet = os.getenv('SOLANA_MODULE_WALLET')

        # EVM credentials
        self.evm_private_key = os.getenv('PRIVATE_KEY')
        self.evm_wallet = os.getenv('WALLET_ADDRESS')
        self.web3_provider = os.getenv('WEB3_PROVIDER_URL')

    async def initialize(self):
        """Initialize executor"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"💱 Copy Trade Executor initialized ({mode})")

    async def close(self):
        """Close executor"""
        if self.session:
            await self.session.close()
            self.session = None

    async def copy_solana_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount_lamports: int,
        slippage_bps: int = 100
    ) -> Dict:
        """Copy a Solana swap via Jupiter"""
        if self.dry_run:
            return await self._simulate_solana_swap(input_mint, output_mint, amount_lamports)

        if not self.solana_wallet or not self.solana_private_key:
            return {'success': False, 'error': 'Solana wallet not configured'}

        try:
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(input_mint, output_mint, amount_lamports, slippage_bps)
            if not quote:
                return {'success': False, 'error': 'Failed to get Jupiter quote'}

            # Get swap transaction
            swap_tx = await self._get_jupiter_swap(quote)
            if not swap_tx:
                return {'success': False, 'error': 'Failed to get swap transaction'}

            # Sign and send
            tx_hash = await self._sign_and_send_solana(swap_tx)
            if tx_hash:
                return {
                    'success': True,
                    'tx_hash': tx_hash,
                    'input_mint': input_mint,
                    'output_mint': output_mint,
                    'amount': amount_lamports
                }
            else:
                return {'success': False, 'error': 'Failed to send transaction'}

        except Exception as e:
            logger.error(f"Solana copy swap error: {e}")
            return {'success': False, 'error': str(e)}

    async def copy_evm_swap(
        self,
        token_address: str,
        amount_wei: int,
        is_buy: bool = True,
        slippage: float = 10.0
    ) -> Dict:
        """Copy an EVM swap via Uniswap"""
        if self.dry_run:
            return await self._simulate_evm_swap(token_address, amount_wei, is_buy)

        if not self.evm_wallet or not self.evm_private_key:
            return {'success': False, 'error': 'EVM wallet not configured'}

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(self.web3_provider))
            if not w3.is_connected():
                return {'success': False, 'error': 'Failed to connect to Web3'}

            # Uniswap V2 Router
            ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
            WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"

            ROUTER_ABI = [{
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            }]

            router = w3.eth.contract(address=Web3.to_checksum_address(ROUTER), abi=ROUTER_ABI)

            if is_buy:
                path = [Web3.to_checksum_address(WETH), Web3.to_checksum_address(token_address)]
            else:
                path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(WETH)]

            deadline = int(datetime.now().timestamp()) + 120

            tx = router.functions.swapExactETHForTokens(
                0,  # Min output
                path,
                Web3.to_checksum_address(self.evm_wallet),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.evm_wallet),
                'value': amount_wei,
                'gas': 300000,
                'nonce': w3.eth.get_transaction_count(self.evm_wallet)
            })

            signed = w3.eth.account.sign_transaction(tx, self.evm_private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)

            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'token': token_address,
                'amount': amount_wei
            }

        except Exception as e:
            logger.error(f"EVM copy swap error: {e}")
            return {'success': False, 'error': str(e)}

    async def _get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict]:
        """Get Jupiter quote"""
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps)
            }
            async with self.session.get(JUPITER_QUOTE_API, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Jupiter quote error: {e}")
        return None

    async def _get_jupiter_swap(self, quote: Dict) -> Optional[str]:
        """Get Jupiter swap transaction"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': self.solana_wallet,
                'wrapAndUnwrapSol': True,
                'prioritizationFeeLamports': 5000
            }
            async with self.session.post(JUPITER_SWAP_API, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('swapTransaction')
        except Exception as e:
            logger.debug(f"Jupiter swap error: {e}")
        return None

    async def _sign_and_send_solana(self, swap_tx_base64: str) -> Optional[str]:
        """Sign and send Solana transaction"""
        try:
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            import base64
            import base58

            private_key_bytes = base58.b58decode(self.solana_private_key)
            keypair = Keypair.from_bytes(private_key_bytes)

            tx_bytes = base64.b64decode(swap_tx_base64)
            tx = VersionedTransaction.from_bytes(tx_bytes)
            tx.sign([keypair])

            async with AsyncClient(self.solana_rpc_url) as client:
                result = await client.send_transaction(tx)
                return str(result.value)

        except ImportError:
            logger.error("Solana libraries not installed")
        except Exception as e:
            logger.error(f"Solana transaction error: {e}")
        return None

    async def _simulate_solana_swap(self, input_mint: str, output_mint: str, amount: int) -> Dict:
        """Simulate Solana swap"""
        import hashlib
        fake_hash = hashlib.sha256(f"{input_mint}{output_mint}{datetime.now().timestamp()}".encode()).hexdigest()
        logger.info(f"🧪 [DRY RUN] Simulated Solana swap: {amount} lamports")
        return {
            'success': True,
            'tx_hash': f"DRY_RUN_{fake_hash[:16]}",
            'input_mint': input_mint,
            'output_mint': output_mint,
            'amount': amount
        }

    async def _simulate_evm_swap(self, token: str, amount: int, is_buy: bool) -> Dict:
        """Simulate EVM swap"""
        import hashlib
        fake_hash = hashlib.sha256(f"{token}{amount}{datetime.now().timestamp()}".encode()).hexdigest()
        side = "BUY" if is_buy else "SELL"
        logger.info(f"🧪 [DRY RUN] Simulated EVM {side}: {amount} wei")
        return {
            'success': True,
            'tx_hash': f"DRY_RUN_{fake_hash[:16]}",
            'token': token,
            'amount': amount
        }


class CopyTradingEngine:
    """
    Copy Trading Engine for tracking and copying wallet trades.

    Features:
    - EVM wallet monitoring via Etherscan
    - Solana wallet monitoring via RPC
    - Real trade execution with Jupiter/Uniswap
    - Configurable copy amount and ratio
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.targets = []  # Initialize empty, load from DB
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.solana_rpc_url = os.getenv('SOLANA_RPC_URL')
        self.helius_api_key = os.getenv('HELIUS_API_KEY')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Copy trading settings
        self.max_copy_amount = 100.0  # Max USD per copy
        self.copy_ratio = 10  # Copy 10% of original

        # Trade executor
        self.executor: Optional[CopyTradeExecutor] = None

        # Track known transactions to avoid duplicates
        self._known_tx_hashes = set()
        self._known_solana_sigs = set()

    async def run(self):
        self.is_running = True
        logger.info("👯 Copy Trading Engine Started")
        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   EVM monitoring: {'Enabled' if self.etherscan_api_key else 'Disabled (no ETHERSCAN_API_KEY)'}")
        logger.info(f"   Solana monitoring: {'Enabled' if self.solana_rpc_url else 'Disabled (no SOLANA_RPC_URL)'}")

        # Initialize executor
        self.executor = CopyTradeExecutor(self.dry_run)
        await self.executor.initialize()

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

        try:
            # Parse token from transaction
            # For swapExactETHForTokens, the token is in the path (input data)
            input_data = source_tx.get('input', '')
            original_value = int(source_tx.get('value', 0))

            # Calculate copy amount (ratio of original)
            copy_amount = min(
                original_value * self.copy_ratio // 100,
                int(self.max_copy_amount * 1e18)  # Max in wei
            )

            if copy_amount <= 0:
                logger.warning("Copy amount too small, skipping")
                return

            # Extract token address from input data (simplified)
            # In production, fully decode the ABI
            token_address = None
            if len(input_data) > 200:
                # Token is usually the last address in path
                # Path starts at offset 196 for swapExactETHForTokens
                try:
                    token_address = '0x' + input_data[-40:]
                except:
                    pass

            if not token_address:
                logger.warning("Could not extract token address from tx")
                return

            # Execute copy trade
            result = await self.executor.copy_evm_swap(
                token_address=token_address,
                amount_wei=copy_amount,
                is_buy='ForTokens' in method_name
            )

            if result.get('success'):
                logger.info(f"✅ EVM Copy Trade {'Executed' if not self.dry_run else 'Simulated'}: {result.get('tx_hash')}")

                # Log to database
                await self._log_copy_trade('ethereum', tx_hash, result)
            else:
                logger.error(f"❌ EVM Copy Trade Failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error executing EVM copy trade: {e}")

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

        try:
            # Extract token mints from transaction
            meta = tx_data.get('meta', {})
            post_balances = meta.get('postTokenBalances', [])
            pre_balances = meta.get('preTokenBalances', [])

            # Find the tokens involved
            input_mint = WSOL_MINT  # Default to SOL
            output_mint = None

            pre_mints = {b.get('mint') for b in pre_balances}

            for balance in post_balances:
                mint = balance.get('mint')
                if mint and mint != WSOL_MINT and mint not in pre_mints:
                    output_mint = mint
                    break

            if not output_mint:
                # Try to find any non-SOL token
                for balance in post_balances:
                    mint = balance.get('mint')
                    if mint and mint != WSOL_MINT:
                        output_mint = mint
                        break

            if not output_mint:
                logger.warning("Could not extract output token from tx")
                return

            # Calculate copy amount
            # Get SOL price and calculate based on max_copy_amount
            sol_price = 200  # Placeholder, in production fetch real price
            copy_lamports = int(min(self.max_copy_amount / sol_price, 0.1) * 1e9)

            # Execute copy trade
            result = await self.executor.copy_solana_swap(
                input_mint=input_mint,
                output_mint=output_mint,
                amount_lamports=copy_lamports
            )

            if result.get('success'):
                logger.info(f"✅ Solana Copy Trade {'Executed' if not self.dry_run else 'Simulated'}: {result.get('tx_hash')}")

                # Log to database
                await self._log_copy_trade('solana', signature, result)
            else:
                logger.error(f"❌ Solana Copy Trade Failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error executing Solana copy trade: {e}")

    async def _log_copy_trade(self, chain: str, source_tx: str, result: Dict):
        """Log copy trade to database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trades (
                        timestamp, symbol, side, price, quantity,
                        status, source, tx_hash, notes
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                    datetime.now(),
                    result.get('output_mint') or result.get('token', 'UNKNOWN'),
                    'BUY',
                    0,  # Price not easily available
                    result.get('amount', 0),
                    'FILLED' if result.get('success') else 'FAILED',
                    'copytrading',
                    result.get('tx_hash'),
                    json.dumps({
                        'chain': chain,
                        'source_tx': source_tx,
                        'dry_run': self.dry_run
                    })
                )
        except Exception as e:
            logger.error(f"Error logging copy trade: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False

        # Close executor
        if self.executor:
            await self.executor.close()

        logger.info("🛑 Copy Trading Engine Stopped")
