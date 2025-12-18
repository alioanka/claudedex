"""
Arbitrage Engine - Spatial Arbitrage (EVM)

Features:
- Multi-DEX price monitoring
- Aave flash loan integration
- Flashbots bundle submission for MEV protection
- Real trade execution (when DRY_RUN=false)
"""
import asyncio
import logging
import os
import json
import aiohttp
from web3 import Web3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from eth_account import Account
from eth_account.messages import encode_defunct

logger = logging.getLogger("ArbitrageEngine")


class PriceFetcher:
    """Fetch real-time prices from CoinGecko"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {symbol: (price, timestamp)}
        self._cache_ttl = 60  # 1 minute cache

    async def get_price(self, symbol: str) -> float:
        """Get current USD price for a token"""
        now = datetime.now()

        # Check cache first
        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (now - cached_at).total_seconds() < self._cache_ttl:
                return price

        # Fetch from CoinGecko
        try:
            symbol_map = {
                'eth': 'ethereum',
                'ethereum': 'ethereum',
                'weth': 'ethereum',
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

        return 2000.0  # Default ETH price

# Uniswap V2 Router ABI (Minimal)
ROUTER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsOut",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Aave V3 Pool ABI (Flash Loan)
AAVE_POOL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "receiverAddress", "type": "address"},
            {"internalType": "address[]", "name": "assets", "type": "address[]"},
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"},
            {"internalType": "uint256[]", "name": "interestRateModes", "type": "uint256[]"},
            {"internalType": "address", "name": "onBehalfOf", "type": "address"},
            {"internalType": "bytes", "name": "params", "type": "bytes"},
            {"internalType": "uint16", "name": "referralCode", "type": "uint16"}
        ],
        "name": "flashLoan",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Common Token Addresses (Ethereum Mainnet)
TOKENS = {
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'DAI': '0x6B175474E89094C44Da98b954EesvcZDECB80656',
}

# Routers
ROUTERS = {
    'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564'
}

# Aave V3 Pool (Mainnet)
AAVE_V3_POOL = '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2'

# Flashbots Relay
FLASHBOTS_RELAY_URL = 'https://relay.flashbots.net'
FLASHBOTS_GOERLI_URL = 'https://relay-goerli.flashbots.net'


class FlashLoanExecutor:
    """
    Flash Loan executor using Aave V3.
    Enables borrowing large amounts without collateral for arbitrage.
    """

    def __init__(self, w3: Web3, private_key: str, wallet_address: str):
        self.w3 = w3
        self.private_key = private_key
        self.wallet_address = wallet_address
        self.aave_pool = None

        if w3:
            self.aave_pool = w3.eth.contract(
                address=Web3.to_checksum_address(AAVE_V3_POOL),
                abi=AAVE_POOL_ABI
            )

    async def execute_flash_loan(
        self,
        assets: List[str],
        amounts: List[int],
        callback_data: bytes
    ) -> Optional[str]:
        """
        Execute Aave flash loan.

        Args:
            assets: List of token addresses to borrow
            amounts: List of amounts to borrow
            callback_data: ABI-encoded arbitrage params

        Returns:
            Transaction hash if successful
        """
        if not self.aave_pool:
            logger.error("Aave pool not initialized")
            return None

        try:
            # Build flash loan transaction
            tx = self.aave_pool.functions.flashLoan(
                Web3.to_checksum_address(self.wallet_address),  # receiver
                [Web3.to_checksum_address(a) for a in assets],
                amounts,
                [0] * len(assets),  # interest rate modes (0 = no debt)
                Web3.to_checksum_address(self.wallet_address),  # onBehalfOf
                callback_data,
                0  # referral code
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 500000,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            logger.info(f"⚡ Flash loan TX sent: {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Flash loan execution failed: {e}")
            return None


class FlashbotsExecutor:
    """
    Flashbots bundle executor for MEV protection.
    Sends transactions directly to block builders, bypassing the public mempool.
    """

    def __init__(self, w3: Web3, private_key: str, signing_key: str = None):
        self.w3 = w3
        self.private_key = private_key
        self.signing_key = signing_key or private_key  # Use separate key for signing
        self.session: Optional[aiohttp.ClientSession] = None
        self.relay_url = FLASHBOTS_RELAY_URL

    async def initialize(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_bundle(
        self,
        transactions: List[str],
        target_block: int
    ) -> Optional[Dict]:
        """
        Send a bundle to Flashbots relay.

        Args:
            transactions: List of signed transaction hex strings
            target_block: Target block number

        Returns:
            Bundle response if successful
        """
        try:
            # Create bundle payload
            params = [{
                'txs': transactions,
                'blockNumber': hex(target_block),
                'minTimestamp': 0,
                'maxTimestamp': int(datetime.now().timestamp()) + 120,
            }]

            # Sign the request
            body = json.dumps({
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'eth_sendBundle',
                'params': params
            })

            # Create signature
            message = encode_defunct(text=Web3.keccak(text=body).hex())
            signed = Account.sign_message(message, private_key=self.signing_key)
            signature = f"{Account.from_key(self.signing_key).address}:{signed.signature.hex()}"

            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': signature
            }

            async with self.session.post(self.relay_url, data=body, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"📦 Bundle sent to Flashbots: {result}")
                    return result
                else:
                    error = await response.text()
                    logger.error(f"Flashbots error: {response.status} - {error}")
                    return None

        except Exception as e:
            logger.error(f"Flashbots bundle send failed: {e}")
            return None

    async def simulate_bundle(
        self,
        transactions: List[str],
        block_number: int,
        state_block: str = 'latest'
    ) -> Optional[Dict]:
        """
        Simulate a bundle before sending.

        Returns:
            Simulation results with profit/loss info
        """
        try:
            params = [{
                'txs': transactions,
                'blockNumber': hex(block_number),
                'stateBlockNumber': state_block,
            }]

            body = json.dumps({
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'eth_callBundle',
                'params': params
            })

            message = encode_defunct(text=Web3.keccak(text=body).hex())
            signed = Account.sign_message(message, private_key=self.signing_key)
            signature = f"{Account.from_key(self.signing_key).address}:{signed.signature.hex()}"

            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': signature
            }

            async with self.session.post(self.relay_url, data=body, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"🔬 Bundle simulation: {result}")
                    return result
                else:
                    error = await response.text()
                    logger.error(f"Simulation error: {error}")
                    return None

        except Exception as e:
            logger.error(f"Bundle simulation failed: {e}")
            return None

class ArbitrageEngine:
    """
    Arbitrage Engine with flash loan and Flashbots support.

    Features:
    - Multi-DEX price monitoring
    - Aave flash loans for capital-efficient arb
    - Flashbots for MEV protection
    - Configurable profit thresholds
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.w3 = None
        self.rpc_url = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))
        self.private_key = os.getenv('PRIVATE_KEY')
        self.wallet_address = os.getenv('WALLET_ADDRESS')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        self.router_contracts = {}

        # Flash loan and Flashbots executors
        self.flash_loan_executor: Optional[FlashLoanExecutor] = None
        self.flashbots_executor: Optional[FlashbotsExecutor] = None

        # Settings
        self.min_profit_threshold = 0.005  # 0.5% minimum profit
        self.use_flash_loans = True
        self.use_flashbots = True
        self.flash_loan_amount = 10 * 10**18  # 10 ETH default

        # Price fetcher for real-time prices
        self.price_fetcher = PriceFetcher()

        # Rate limiting for logging and execution
        self._last_opportunity_time = None
        self._last_opportunity_key = None
        self._opportunity_cooldown = 300  # 5 minutes cooldown for same opportunity
        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }

    async def initialize(self):
        logger.info("⚖️ Initializing Arbitrage Engine...")

        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url.split(',')[0]))
            if self.w3.is_connected():
                logger.info("✅ Connected to Arbitrage RPC")

                # Initialize router contracts
                for name, address in ROUTERS.items():
                    try:
                        self.router_contracts[name] = self.w3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=ROUTER_ABI
                        )
                    except Exception as e:
                        logger.debug(f"Could not init {name} router: {e}")

                # Initialize Flash Loan executor
                if self.private_key and self.wallet_address and not self.dry_run:
                    self.flash_loan_executor = FlashLoanExecutor(
                        self.w3,
                        self.private_key,
                        self.wallet_address
                    )
                    logger.info("⚡ Flash Loan executor initialized")

                    # Initialize Flashbots executor
                    self.flashbots_executor = FlashbotsExecutor(
                        self.w3,
                        self.private_key
                    )
                    await self.flashbots_executor.initialize()
                    logger.info("📦 Flashbots executor initialized")

            else:
                logger.warning("⚠️ Failed to connect to Arbitrage RPC")

        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Flash Loans: {'Enabled' if self.flash_loan_executor else 'Disabled'}")
        logger.info(f"   Flashbots: {'Enabled' if self.flashbots_executor else 'Disabled'}")

    async def run(self):
        self.is_running = True
        logger.info("⚖️ Arbitrage Engine Started")

        if not self.w3:
            logger.error("RPC not connected, arbitrage disabled.")
            return

        while self.is_running:
            try:
                self._stats['scans'] += 1

                # Scan WETH/USDC
                await self._check_arb_opportunity(TOKENS['WETH'], TOKENS['USDC'])

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Arb loop error: {e}")
                await asyncio.sleep(5)

    async def _log_stats_if_needed(self):
        """Log statistics every 5 minutes"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            logger.info(f"📊 ARBITRAGE STATS (Last 5 min): "
                       f"Scans: {self._stats['scans']} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']}")

            # Reset stats
            self._stats = {
                'scans': 0,
                'opportunities_found': 0,
                'opportunities_executed': 0,
                'last_stats_log': now
            }

    async def _check_arb_opportunity(self, token_in, token_out) -> bool:
        """Check price difference between two DEXs. Returns True if opportunity found."""
        try:
            amount_in = self.flash_loan_amount  # Use configured flash loan amount

            prices = {}
            for name, contract in self.router_contracts.items():
                try:
                    amounts = contract.functions.getAmountsOut(amount_in, [token_in, token_out]).call()
                    prices[name] = amounts[1]
                except Exception as e:
                    logger.debug(f"Failed to get price from {name}: {e}")

            if len(prices) < 2:
                return False

            # Find best buy and sell
            best_buy_dex = min(prices, key=prices.get)
            best_sell_dex = max(prices, key=prices.get)

            buy_price = prices[best_buy_dex]
            sell_price = prices[best_sell_dex]

            spread = (sell_price - buy_price) / buy_price

            if spread > self.min_profit_threshold:
                self._stats['opportunities_found'] += 1

                # Create unique key for this opportunity
                opp_key = f"{best_buy_dex}_{best_sell_dex}_{token_in}"

                # Check cooldown - don't spam same opportunity
                now = datetime.now()
                if self._last_opportunity_key == opp_key and self._last_opportunity_time:
                    elapsed = (now - self._last_opportunity_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        # Same opportunity within cooldown, skip silently
                        return True

                # New opportunity or cooldown expired - log and execute
                self._last_opportunity_key = opp_key
                self._last_opportunity_time = now

                logger.info(f"🚨 ARBITRAGE OPPORTUNITY: Buy on {best_buy_dex}, Sell on {best_sell_dex}. Spread: {spread:.2%}")
                self._stats['opportunities_executed'] += 1

                # Execute arbitrage
                await self._execute_flash_swap(
                    buy_dex=best_buy_dex,
                    sell_dex=best_sell_dex,
                    token_in=token_in,
                    token_out=token_out,
                    amount=amount_in,
                    expected_profit=spread
                )
                return True
            return False

        except Exception as e:
            logger.error(f"Arb check failed: {e}")
            return False

    async def _execute_flash_swap(
        self,
        buy_dex: str,
        sell_dex: str,
        token_in: str,
        token_out: str,
        amount: int,
        expected_profit: float
    ):
        """Execute the arbitrage trade using flash loans and Flashbots"""
        logger.info(f"⚡ Executing Arbitrage: {buy_dex} -> {sell_dex} | Amount: {amount/1e18:.4f} ETH | Expected: +{expected_profit:.2%}")

        if self.dry_run:
            # Simulate execution
            await asyncio.sleep(0.5)
            logger.info("✅ Flash Swap Executed (DRY RUN)")
            await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, "DRY_RUN")
            return

        try:
            if self.use_flash_loans and self.flash_loan_executor:
                # Use flash loan for capital efficiency
                tx_hash = await self._execute_with_flash_loan(
                    buy_dex, sell_dex, token_in, token_out, amount
                )
            else:
                # Execute with own capital
                tx_hash = await self._execute_direct_swap(
                    buy_dex, sell_dex, token_in, token_out, amount
                )

            if tx_hash:
                logger.info(f"✅ Arbitrage executed: {tx_hash}")
                await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, tx_hash)
            else:
                logger.error("❌ Arbitrage execution failed")

        except Exception as e:
            logger.error(f"Arbitrage execution error: {e}")

    async def _execute_with_flash_loan(
        self,
        buy_dex: str,
        sell_dex: str,
        token_in: str,
        token_out: str,
        amount: int
    ) -> Optional[str]:
        """Execute arbitrage using Aave flash loan"""

        # Encode arbitrage parameters for the flash loan callback
        # In production, you'd have a deployed contract that implements IFlashLoanReceiver
        callback_data = self.w3.codec.encode_abi(
            ['address', 'address', 'address', 'address'],
            [
                ROUTERS.get(buy_dex, ROUTERS['uniswap_v2']),
                ROUTERS.get(sell_dex, ROUTERS['sushiswap']),
                token_in,
                token_out
            ]
        )

        # Build flash loan transactions
        tx_hash = await self.flash_loan_executor.execute_flash_loan(
            assets=[token_in],
            amounts=[amount],
            callback_data=callback_data
        )

        return tx_hash

    async def _execute_direct_swap(
        self,
        buy_dex: str,
        sell_dex: str,
        token_in: str,
        token_out: str,
        amount: int
    ) -> Optional[str]:
        """Execute arbitrage with own capital, optionally via Flashbots"""

        try:
            buy_router = self.router_contracts.get(buy_dex)
            sell_router = self.router_contracts.get(sell_dex)

            if not buy_router or not sell_router:
                logger.error("Router contracts not available")
                return None

            deadline = int(datetime.now().timestamp()) + 120

            # Build buy transaction
            buy_tx = buy_router.functions.swapExactTokensForTokens(
                amount,
                0,  # Min output
                [token_in, token_out],
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 300000,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
            })

            # Sign buy transaction
            signed_buy = self.w3.eth.account.sign_transaction(buy_tx, self.private_key)

            # Build sell transaction (nonce + 1)
            sell_tx = sell_router.functions.swapExactTokensForTokens(
                amount,  # Simplified - should use output from buy
                0,
                [token_out, token_in],
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 300000,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address) + 1
            })

            # Sign sell transaction
            signed_sell = self.w3.eth.account.sign_transaction(sell_tx, self.private_key)

            # Send via Flashbots if available
            if self.use_flashbots and self.flashbots_executor:
                current_block = self.w3.eth.block_number
                target_block = current_block + 1

                # First simulate
                sim_result = await self.flashbots_executor.simulate_bundle(
                    [signed_buy.rawTransaction.hex(), signed_sell.rawTransaction.hex()],
                    target_block
                )

                if sim_result and 'error' not in sim_result:
                    # Send bundle
                    bundle_result = await self.flashbots_executor.send_bundle(
                        [signed_buy.rawTransaction.hex(), signed_sell.rawTransaction.hex()],
                        target_block
                    )
                    if bundle_result:
                        return bundle_result.get('bundleHash', 'bundle_sent')
                    else:
                        logger.warning("Flashbots bundle rejected, falling back to public mempool")

            # Fallback: Send to public mempool
            tx_hash = self.w3.eth.send_raw_transaction(signed_buy.rawTransaction)
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Direct swap execution error: {e}")
            return None

    async def _log_arb_trade(
        self,
        buy_dex: str,
        sell_dex: str,
        token: str,
        amount: int,
        profit_pct: float,
        tx_hash: str
    ):
        """Log arbitrage trade to database"""
        if not self.db_pool:
            return

        try:
            import uuid
            amount_eth = amount / 1e18

            # Get real ETH price
            eth_price = await self.price_fetcher.get_price('eth')

            # Calculate real USD values
            entry_usd = amount_eth * eth_price
            profit_usd = entry_usd * profit_pct
            exit_usd = entry_usd + profit_usd

            # Use ETH price as entry, calculate exit based on profit
            entry_price = eth_price
            exit_price = eth_price * (1 + profit_pct)

            logger.info(f"💰 Arb value: {amount_eth:.4f} ETH @ ${eth_price:.2f} = ${entry_usd:.2f} | Profit: ${profit_usd:.2f}")

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trades (
                        trade_id, token_address, chain, side, entry_price, exit_price,
                        amount, usd_value, profit_loss, profit_loss_percentage,
                        status, strategy, entry_timestamp, exit_timestamp, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                    f"arb_{uuid.uuid4().hex[:12]}",
                    token,
                    'ethereum',
                    'buy',
                    entry_price,
                    exit_price,
                    amount_eth,
                    entry_usd,
                    profit_usd,
                    profit_pct * 100,
                    'closed',
                    'arbitrage',
                    datetime.now(),
                    datetime.now(),  # Exit timestamp same as entry for arb
                    json.dumps({
                        'tx_hash': tx_hash,
                        'buy_dex': buy_dex,
                        'sell_dex': sell_dex,
                        'eth_price': eth_price,
                        'dry_run': self.dry_run,
                        'spread_pct': profit_pct * 100
                    })
                )
        except Exception as e:
            logger.error(f"Error logging arb trade: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False

        # Close Flashbots executor
        if self.flashbots_executor:
            await self.flashbots_executor.close()

        logger.info("🛑 Arbitrage Engine Stopped")
