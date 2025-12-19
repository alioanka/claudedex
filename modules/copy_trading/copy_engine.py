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


class PriceFetcher:
    """Fetch real-time prices from CoinGecko"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {symbol: (price, timestamp)}
        self._cache_ttl = 60  # 1 minute cache

    async def get_price(self, symbol: str) -> float:
        """Get current USD price for a token (sol, ethereum, etc.)"""
        now = datetime.now()

        # Check cache first
        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (now - cached_at).total_seconds() < self._cache_ttl:
                return price

        # Fetch from CoinGecko
        try:
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'sol': 'solana',
                'solana': 'solana',
                'eth': 'ethereum',
                'ethereum': 'ethereum',
                'btc': 'bitcoin',
                'bitcoin': 'bitcoin',
            }
            coin_id = symbol_map.get(symbol.lower(), symbol.lower())

            async with aiohttp.ClientSession() as session:
                params = {'ids': coin_id, 'vs_currencies': 'usd'}
                async with session.get(self.COINGECKO_API, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if coin_id in data and 'usd' in data[coin_id]:
                            price = float(data[coin_id]['usd'])
                            self._cache[symbol] = (price, now)
                            return price
        except Exception as e:
            logger.debug(f"Price fetch error for {symbol}: {e}")

        # Fallback to cached or default
        if symbol in self._cache:
            return self._cache[symbol][0]

        # Last resort defaults
        defaults = {'sol': 200.0, 'solana': 200.0, 'eth': 2000.0, 'ethereum': 2000.0}
        return defaults.get(symbol.lower(), 1.0)


class CopyTradeExecutor:
    """Trade executor for Copy Trading module"""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_fetcher = PriceFetcher()

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

        # Rate limiting - cooldown per wallet (5 min)
        self._wallet_last_copy_time: Dict[str, datetime] = {}
        self._wallet_cooldown_seconds = 300  # 5 minutes

        # Statistics
        self._stats = {
            'cycles': 0,
            'evm_copies': 0,
            'sol_copies': 0,
            'last_stats_log': datetime.now()
        }

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

        while self.is_running:
            try:
                self._stats['cycles'] += 1

                # Reload settings periodically to catch updates
                await self._load_settings()

                if self.targets:
                    # Monitor EVM wallets
                    evm_copied = await self._monitor_evm_wallets()
                    self._stats['evm_copies'] += evm_copied

                    # Monitor Solana wallets
                    sol_copied = await self._monitor_solana_wallets()
                    self._stats['sol_copies'] += sol_copied

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                await asyncio.sleep(15)  # Poll every 15s
            except Exception as e:
                logger.error(f"Copy loop error: {e}")
                await asyncio.sleep(15)

    async def _log_stats_if_needed(self):
        """Log statistics every 5 minutes"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            logger.info(f"📊 COPY TRADING STATS (Last 5 min): "
                       f"Cycles: {self._stats['cycles']} | "
                       f"Wallets: {len(self.targets)} | "
                       f"EVM Copies: {self._stats['evm_copies']} | "
                       f"Solana Copies: {self._stats['sol_copies']}")

            # Reset stats
            self._stats = {
                'cycles': 0,
                'evm_copies': 0,
                'sol_copies': 0,
                'last_stats_log': now
            }

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
            wallet = tx.get('from', '')

            # Check wallet cooldown first
            if self._check_wallet_cooldown(wallet):
                return False  # Skip silently - wallet in cooldown

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

                # Update cooldown before executing
                self._update_wallet_cooldown(wallet)

                logger.info(f"👯 EVM COPY TRIGGER: Wallet {wallet[:16]}... executed {method_name}")
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

                # Log to database with source wallet
                source_wallet = source_tx.get('from', '')
                await self._log_copy_trade('ethereum', tx_hash, result, source_wallet)
            else:
                logger.error(f"❌ EVM Copy Trade Failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error executing EVM copy trade: {e}")

    def _check_wallet_cooldown(self, wallet: str) -> bool:
        """Check if wallet is in cooldown period. Returns True if we should skip."""
        now = datetime.now()
        last_copy = self._wallet_last_copy_time.get(wallet)

        if last_copy:
            elapsed = (now - last_copy).total_seconds()
            if elapsed < self._wallet_cooldown_seconds:
                return True  # Still in cooldown, skip

        return False  # Not in cooldown, proceed

    def _update_wallet_cooldown(self, wallet: str):
        """Update wallet's last copy time"""
        self._wallet_last_copy_time[wallet] = datetime.now()

    async def _analyze_and_copy_solana(self, wallet: str, signature: str) -> bool:
        """Analyze Solana transaction and execute copy if it's a swap"""
        try:
            # Check wallet cooldown first
            if self._check_wallet_cooldown(wallet):
                return False  # Skip silently - wallet in cooldown

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
                        # Update cooldown before executing
                        self._update_wallet_cooldown(wallet)

                        logger.info(f"👯 SOLANA COPY TRIGGER: Wallet {wallet[:16]}... executed swap {signature[:20]}...")
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

                # Log to database with source wallet
                await self._log_copy_trade('solana', signature, result, wallet)
            else:
                logger.error(f"❌ Solana Copy Trade Failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error executing Solana copy trade: {e}")

    async def _log_copy_trade(self, chain: str, source_tx: str, result: Dict, source_wallet: str = None):
        """Log copy trade to database"""
        if not self.db_pool:
            return

        try:
            import uuid
            now = datetime.now()

            # Get real prices from API
            amount = result.get('amount', 0)
            if chain == 'solana':
                # Convert lamports to SOL, get real price
                amount_native = amount / 1e9
                native_price = await self.executor.price_fetcher.get_price('sol')
                usd_value = amount_native * native_price
                entry_price = native_price
            else:
                # Convert wei to ETH, get real price
                amount_native = amount / 1e18
                native_price = await self.executor.price_fetcher.get_price('eth')
                usd_value = amount_native * native_price
                entry_price = native_price

            token_address = result.get('output_mint') or result.get('token', 'UNKNOWN')
            trade_id = f"copy_{uuid.uuid4().hex[:12]}"
            logger.info(f"💰 Trade value: {amount_native:.4f} {'SOL' if chain == 'solana' else 'ETH'} @ ${entry_price:.2f} = ${usd_value:.2f}")

            async with self.db_pool.acquire() as conn:
                # Insert into dedicated copytrading_trades table
                await conn.execute("""
                    INSERT INTO copytrading_trades (
                        trade_id, token_address, chain, source_wallet, source_tx,
                        side, entry_price, exit_price, amount,
                        entry_usd, exit_usd, profit_loss, profit_loss_pct,
                        status, is_simulated, entry_timestamp,
                        tx_hash, native_price_at_trade, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """,
                    trade_id,
                    token_address,
                    chain,
                    source_wallet or 'unknown',
                    source_tx,
                    'buy',
                    entry_price,
                    entry_price,  # Same as entry for now (position still open)
                    amount_native,
                    usd_value,
                    0.0,  # Exit USD - will be set on exit
                    0.0,  # No profit yet - will be calculated on exit
                    0.0,
                    'open' if result.get('success') else 'failed',
                    self.dry_run,
                    now,
                    result.get('tx_hash'),
                    native_price,
                    json.dumps({
                        'dry_run': self.dry_run
                    })
                )
                logger.debug(f"💾 Logged to copytrading_trades: {trade_id}")

                # Update wallet stats if source_wallet provided
                if source_wallet and result.get('success'):
                    await self._update_wallet_stats(conn, source_wallet)

        except Exception as e:
            logger.error(f"Error logging copy trade: {e}")

    async def _update_wallet_stats(self, conn, wallet: str):
        """Update wallet statistics after a copy trade"""
        try:
            # Count trades copied from this wallet (use dedicated table)
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM copytrading_trades
                WHERE source_wallet = $1
            """, wallet)

            # Update the wallet's copied trades count in config if tracking
            await conn.execute("""
                INSERT INTO config_settings (config_type, key, value)
                VALUES ('wallet_stats', $1, $2)
                ON CONFLICT (config_type, key) DO UPDATE SET value = $2
            """, wallet, str(count))

        except Exception as e:
            logger.debug(f"Error updating wallet stats: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False

        # Close executor
        if self.executor:
            await self.executor.close()

        logger.info("🛑 Copy Trading Engine Stopped")
