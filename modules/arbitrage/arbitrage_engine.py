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
from eth_abi import encode

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

        # Fallback to cached price if available
        if symbol in self._cache:
            return self._cache[symbol][0]

        # IMPORTANT: Do not use hardcoded fallback for P&L calculations
        # Return None to signal price unavailable - caller should handle appropriately
        logger.warning(f"⚠️ No price available for {symbol} - returning None")
        return None

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

# Aave V3 Pool ABI (Flash Loan Simple - single asset)
# This matches the FlashLoanArbitrage.sol contract which uses flashLoanSimple
AAVE_POOL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "receiverAddress", "type": "address"},
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bytes", "name": "params", "type": "bytes"},
            {"internalType": "uint16", "name": "referralCode", "type": "uint16"}
        ],
        "name": "flashLoanSimple",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# FlashLoanArbitrage contract ABI - this is what we actually call
# The contract initiates the flash loan internally (so initiator check passes)
FLASH_LOAN_CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "address", "name": "buyRouter", "type": "address"},
            {"internalType": "address", "name": "sellRouter", "type": "address"},
            {"internalType": "address", "name": "intermediateToken", "type": "address"}
        ],
        "name": "executeArbitrage",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "asset", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "address", "name": "buyRouter", "type": "address"},
            {"internalType": "address", "name": "sellRouter", "type": "address"},
            {"internalType": "address", "name": "intermediateToken", "type": "address"}
        ],
        "name": "checkArbitrage",
        "outputs": [{"internalType": "int256", "name": "profit", "type": "int256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-CHAIN ADDRESS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Chain IDs
CHAIN_IDS = {
    'ethereum': 1,
    'arbitrum': 42161,
    'optimism': 10,
    'base': 8453,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ETHEREUM MAINNET (Chain ID: 1)
# ═══════════════════════════════════════════════════════════════════════════════
TOKENS_ETHEREUM = {
    # Major tokens
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'DAI': '0x6B175474E89094C44Da98b954EeAdDcB80656c63',
    'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',

    # DeFi tokens
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
    'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
    'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
    'CRV': '0xD533a949740bb3306d119CC777fa900bA034cd52',
    'MKR': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2',
    'COMP': '0xc00e94Cb662C3520282E6f5717214004A7f26888',
    'SUSHI': '0x6B3595068778DD592e39A122f4f5a5cF09C90fE2',
    'SNX': '0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F',

    # Layer 2 / Scaling
    'MATIC': '0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0',
    'LDO': '0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32',
    'ARB': '0xB50721BCf8d664c30412Cfbc6cf7a15145234ad1',
    'OP': '0x4200000000000000000000000000000000000042',

    # Stablecoins
    'FRAX': '0x853d955aCEf822Db058eb8505911ED77F175b99e',
    'LUSD': '0x5f98805A4E8be255a32880FDeC7F6728C6568bA0',
    'TUSD': '0x0000000000085d4780B73119b644AE5ecd22b376',

    # Meme/Popular
    'SHIB': '0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE',
    'PEPE': '0x6982508145454Ce325dDbE47a25d4ec3d2311933',

    # Other DeFi
    'BAL': '0xba100000625a3754423978a60c9317c58a424e3D',
    'YFI': '0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e',
    'INCH': '0x111111111117dC0aa78b770fA6A738034120C302',
    'GRT': '0xc944E90C64B2c07662A292be6244BDf05Cda44a7',
    'ENS': '0xC18360217D8F7Ab5e7c516566761Ea12Ce7F9D72',
}

# ═══════════════════════════════════════════════════════════════════════════════
# ARBITRUM ONE (Chain ID: 42161)
# Lower gas costs (~95% cheaper than mainnet), same EVM compatibility
# ═══════════════════════════════════════════════════════════════════════════════
TOKENS_ARBITRUM = {
    # Major tokens - Arbitrum native/bridged addresses
    'WETH': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',  # Wrapped ETH on Arbitrum
    'USDC': '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',  # Native USDC (Circle)
    'USDC_BRIDGED': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',  # Bridged USDC.e
    'USDT': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',  # Tether USD
    'DAI': '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',  # DAI Stablecoin
    'WBTC': '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f',  # Wrapped BTC

    # Native Arbitrum tokens
    'ARB': '0x912CE59144191C1204E64559FE8253a0e49E6548',  # Arbitrum token
    'GMX': '0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a',  # GMX
    'MAGIC': '0x539bdE0d7Dbd336b79148AA742883198BBF60342',  # Magic (TreasureDAO)
    'RDNT': '0x3082CC23568eA640225c2467653dB90e9250AaA0',  # Radiant Capital
    'PENDLE': '0x0c880f6761F1af8d9Aa9C466984b80DAb9a8c9e8',  # Pendle

    # DeFi tokens on Arbitrum
    'UNI': '0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0',  # Uniswap
    'LINK': '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',  # Chainlink
    'AAVE': '0xba5DdD1f9d7F570dc94a51479a000E3BCE967196',  # Aave
    'CRV': '0x11cDb42B0EB46D95f990BeDD4695A6e3fA034978',  # Curve DAO
    'SUSHI': '0xd4d42F0b6DEF4CE0383636770eF773390d85c61A',  # SushiSwap

    # Stablecoins
    'FRAX': '0x17FC002b466eEc40DaE837Fc4bE5c67993ddBd6F',  # Frax
    'LUSD': '0x93b346b6BC2548dA6A1E7d98E9a421B42541425b',  # Liquity USD
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER ADDRESSES PER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
ROUTERS_ETHEREUM = {
    'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
}

ROUTERS_ARBITRUM = {
    'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',  # SushiSwap V2 Router
    'camelot': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d',    # Camelot DEX Router
    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564', # Uniswap V3 (same address)
}

# ═══════════════════════════════════════════════════════════════════════════════
# AAVE V3 POOL ADDRESSES PER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
AAVE_POOLS = {
    'ethereum': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
    'arbitrum': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
}

# ═══════════════════════════════════════════════════════════════════════════════
# ARBITRAGE PAIRS PER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
ARB_PAIRS_ETHEREUM = [
    ('WETH', 'USDC'),
    ('WETH', 'USDT'),
    ('WETH', 'DAI'),
    ('WBTC', 'WETH'),
    ('LINK', 'WETH'),
    ('UNI', 'WETH'),
    ('AAVE', 'WETH'),
    ('MATIC', 'WETH'),
    ('SUSHI', 'WETH'),
    ('CRV', 'WETH'),
    ('LDO', 'WETH'),
    ('SNX', 'WETH'),
    ('COMP', 'WETH'),
    ('MKR', 'WETH'),
    ('BAL', 'WETH'),
    ('YFI', 'WETH'),
    ('GRT', 'WETH'),
    ('ENS', 'WETH'),
    ('SHIB', 'WETH'),
    ('PEPE', 'WETH'),
]

ARB_PAIRS_ARBITRUM = [
    # High liquidity pairs
    ('WETH', 'USDC'),
    ('WETH', 'USDT'),
    ('WETH', 'DAI'),
    ('WBTC', 'WETH'),
    ('ARB', 'WETH'),       # Native ARB token - high volume
    ('ARB', 'USDC'),       # ARB/stablecoin

    # Native Arbitrum DeFi tokens
    ('GMX', 'WETH'),       # GMX - major Arbitrum protocol
    ('MAGIC', 'WETH'),     # TreasureDAO
    ('RDNT', 'WETH'),      # Radiant Capital
    ('PENDLE', 'WETH'),    # Pendle

    # Bridged DeFi tokens
    ('LINK', 'WETH'),
    ('UNI', 'WETH'),
    ('AAVE', 'WETH'),
    ('CRV', 'WETH'),
    ('SUSHI', 'WETH'),

    # Stablecoin pairs (good for low-risk arb)
    ('USDC', 'USDT'),
    ('USDC', 'DAI'),
    ('USDC', 'USDC_BRIDGED'),  # Native vs bridged USDC
]

# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN CONFIG AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════
CHAIN_CONFIGS = {
    1: {  # Ethereum Mainnet
        'name': 'ethereum',
        'tokens': TOKENS_ETHEREUM,
        'routers': ROUTERS_ETHEREUM,
        'aave_pool': AAVE_POOLS['ethereum'],
        'arb_pairs': ARB_PAIRS_ETHEREUM,
        'native_symbol': 'ETH',
        'min_gas_eth': 0.015,  # ~$30-50 for flash loan tx
    },
    42161: {  # Arbitrum One
        'name': 'arbitrum',
        'tokens': TOKENS_ARBITRUM,
        'routers': ROUTERS_ARBITRUM,
        'aave_pool': AAVE_POOLS['arbitrum'],
        'arb_pairs': ARB_PAIRS_ARBITRUM,
        'native_symbol': 'ETH',
        'min_gas_eth': 0.0005,  # ~$1 for flash loan tx (95% cheaper!)
    },
}

# Default to Ethereum for backwards compatibility
TOKENS = TOKENS_ETHEREUM

# Backwards compatibility - default to Ethereum
ARB_PAIRS = ARB_PAIRS_ETHEREUM
ROUTERS = ROUTERS_ETHEREUM
AAVE_V3_POOL = AAVE_POOLS['ethereum']

# Flashbots Relay
FLASHBOTS_RELAY_URL = 'https://relay.flashbots.net'
FLASHBOTS_GOERLI_URL = 'https://relay-goerli.flashbots.net'


class FlashLoanExecutor:
    """
    Flash Loan executor using FlashLoanArbitrage contract.

    IMPORTANT: We call the contract's executeArbitrage() function, NOT Aave directly!
    This ensures the initiator check passes (initiator == contract address).

    Flow:
    1. User wallet calls executeArbitrage() on FlashLoanArbitrage contract
    2. Contract calls flashLoanSimple() on Aave Pool
    3. Aave calls executeOperation() on contract with initiator = contract (passes check!)
    4. Contract executes swaps and repays loan
    """

    def __init__(self, w3: Web3, private_key: str, wallet_address: str, receiver_contract: str):
        self.w3 = w3
        self.private_key = private_key
        self.wallet_address = wallet_address  # EOA for signing/gas (must be contract owner)
        self.receiver_contract = receiver_contract  # FlashLoanArbitrage contract address
        self.flash_loan_contract = None
        self._pending_nonce = None  # Track nonce to avoid collisions
        self._last_tx_time = None

        if w3:
            self.flash_loan_contract = w3.eth.contract(
                address=Web3.to_checksum_address(receiver_contract),
                abi=FLASH_LOAN_CONTRACT_ABI
            )

    def _get_next_nonce(self) -> int:
        """Get the next nonce, accounting for pending transactions"""
        # Get both confirmed and pending nonce
        confirmed_nonce = self.w3.eth.get_transaction_count(self.wallet_address)
        pending_nonce = self.w3.eth.get_transaction_count(self.wallet_address, 'pending')

        # Use the higher of pending or our tracked nonce
        if self._pending_nonce is not None:
            next_nonce = max(confirmed_nonce, pending_nonce, self._pending_nonce)
        else:
            next_nonce = max(confirmed_nonce, pending_nonce)

        # Update tracked nonce for next call
        self._pending_nonce = next_nonce + 1
        return next_nonce

    async def check_arbitrage_profit(
        self,
        asset: str,
        amount: int,
        buy_router: str,
        sell_router: str,
        intermediate_token: str
    ) -> Optional[int]:
        """
        Simulate arbitrage to check if profitable (view function - no gas).

        Returns:
            Expected profit in wei, or None if simulation fails
        """
        if not self.flash_loan_contract:
            return None

        try:
            profit = self.flash_loan_contract.functions.checkArbitrage(
                Web3.to_checksum_address(asset),
                amount,
                Web3.to_checksum_address(buy_router),
                Web3.to_checksum_address(sell_router),
                Web3.to_checksum_address(intermediate_token)
            ).call()
            return profit
        except Exception as e:
            logger.debug(f"Arbitrage simulation failed: {e}")
            return None

    async def execute_arbitrage(
        self,
        asset: str,
        amount: int,
        buy_router: str,
        sell_router: str,
        intermediate_token: str
    ) -> Optional[str]:
        """
        Execute arbitrage via FlashLoanArbitrage contract.

        The contract will:
        1. Initiate flash loan from Aave (with itself as initiator)
        2. Execute swaps on buyRouter then sellRouter
        3. Repay loan + fee
        4. Keep profit in contract

        Args:
            asset: Token address to borrow (e.g., WETH)
            amount: Amount to borrow in wei
            buy_router: DEX router with lower price
            sell_router: DEX router with higher price
            intermediate_token: Token to swap through

        Returns:
            Transaction hash if successful
        """
        if not self.flash_loan_contract:
            logger.error("Flash loan contract not initialized")
            return None

        try:
            # First simulate to check profitability (saves gas on failures)
            expected_profit = await self.check_arbitrage_profit(
                asset, amount, buy_router, sell_router, intermediate_token
            )

            if expected_profit is not None and expected_profit <= 0:
                logger.warning(f"⚠️ Simulation shows no profit ({expected_profit}), skipping execution")
                return None

            # Get current gas price with buffer for faster inclusion
            gas_price = self.w3.eth.gas_price
            gas_price_with_buffer = int(gas_price * 1.2)  # 20% buffer for faster inclusion

            # CRITICAL: Verify profit exceeds gas cost + safety buffer
            # Flash loan reverts are caused by profit being consumed by gas or price movement
            # Realistic gas for flash loan arbitrage: ~450K (not 800K)
            gas_limit = 450000
            gas_cost_wei = gas_price_with_buffer * gas_limit
            # Require profit to be 30% higher than gas cost to account for:
            # - Price movement during block inclusion
            # - Slippage in actual execution vs simulation
            min_profit_required = int(gas_cost_wei * 1.3)

            if expected_profit is not None and expected_profit < min_profit_required:
                gas_cost_eth = gas_cost_wei / 1e18
                profit_eth = expected_profit / 1e18
                logger.warning(f"⚠️ Profit too low to cover gas + buffer, skipping execution")
                logger.warning(f"   Expected profit: {profit_eth:.6f} ETH | Gas cost: {gas_cost_eth:.6f} ETH")
                logger.warning(f"   Required: {min_profit_required/1e18:.6f} ETH (1.5x gas)")
                return None

            gas_price = gas_price_with_buffer

            # Rate limit: wait at least 12 seconds between transactions (1 block)
            if self._last_tx_time:
                elapsed = (datetime.now() - self._last_tx_time).total_seconds()
                if elapsed < 12:
                    wait_time = 12 - elapsed
                    logger.info(f"⏳ Waiting {wait_time:.1f}s for block confirmation...")
                    await asyncio.sleep(wait_time)

            # Build transaction to call contract's executeArbitrage function
            tx = self.flash_loan_contract.functions.executeArbitrage(
                Web3.to_checksum_address(asset),
                amount,
                Web3.to_checksum_address(buy_router),
                Web3.to_checksum_address(sell_router),
                Web3.to_checksum_address(intermediate_token)
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 800000,
                'gasPrice': gas_price,
                'nonce': self._get_next_nonce(),
                'chainId': self.w3.eth.chain_id
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            self._last_tx_time = datetime.now()

            logger.info(f"⚡ Flash loan TX sent: {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            error_msg = str(e)
            if 'replacement transaction underpriced' in error_msg:
                logger.warning(f"⚠️ Nonce collision detected, resetting nonce tracker")
                self._pending_nonce = None  # Reset to force fresh nonce fetch
            logger.error(f"Flash loan execution failed: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
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
    - Multi-chain support (Ethereum, Arbitrum)
    - Multi-DEX price monitoring
    - Aave flash loans for capital-efficient arb
    - Flashbots for MEV protection (Ethereum only)
    - Configurable profit thresholds
    """

    def __init__(self, config: Dict, db_pool, chain: str = 'ethereum'):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.w3 = None

        # Chain configuration
        self.chain_name = chain.lower()  # 'ethereum' or 'arbitrum'
        self.chain_id = None  # Set in initialize() from RPC
        self.chain_config = None  # Set in initialize() based on chain_id

        # Chain-specific addresses (set in initialize())
        self.tokens = TOKENS_ETHEREUM  # Default, updated based on chain
        self.routers = ROUTERS_ETHEREUM
        self.arb_pairs = ARB_PAIRS_ETHEREUM
        self.aave_pool = AAVE_POOLS['ethereum']

        # Get RPC URL from config - support chain-specific RPC keys
        self.rpc_url = config.get('rpc_url')
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                # Try chain-specific RPC first
                rpc_key = 'ARBITRUM_RPC' if chain == 'arbitrum' else 'ETHEREUM_RPC'
                self.rpc_url = RPCProvider.get_rpc_sync(rpc_key)
            except Exception:
                pass
        if not self.rpc_url:
            # Fallback to environment variables
            if chain == 'arbitrum':
                self.rpc_url = os.getenv('ARBITRUM_RPC_URL')
            else:
                self.rpc_url = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))

        self.private_key = None  # Loaded in initialize() from secrets manager
        self.wallet_address = None  # Loaded in initialize() from secrets manager

        # dry_run: Priority is database config > environment variable
        # This allows dashboard settings to override .env
        db_dry_run = config.get('dry_run')
        if db_dry_run is not None:
            self.dry_run = db_dry_run if isinstance(db_dry_run, bool) else str(db_dry_run).lower() in ('true', '1', 'yes')
        else:
            self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        self.router_contracts = {}

        # Flash loan and Flashbots executors
        self.flash_loan_executor: Optional[FlashLoanExecutor] = None
        self.flashbots_executor: Optional[FlashbotsExecutor] = None

        # Settings
        # IMPORTANT: Real DEX arbitrage opportunities are typically 0.1-0.5%
        # After costs (~0.55%): flash loan 0.05% + slippage ~0.5% = need ~0.6% raw spread
        self.min_profit_threshold = 0.003  # 0.3% minimum NET profit (after costs)
        self.use_flash_loans = True
        self.use_flashbots = True
        self.flash_loan_amount = 10 * 10**18  # 10 ETH default

        # Price fetcher for real-time prices
        self.price_fetcher = PriceFetcher()

        # Rate limiting for logging and execution
        self._last_opportunity_time = None
        self._last_opportunity_key = None
        self._opportunity_cooldown = 300  # 5 min cooldown for same opportunity (was 1 hour - too long)

        # Visibility logging - track spreads even when not executing
        self._last_spread_log_time = None
        self._best_spread_seen = -999.0  # Start negative so we track even negative spreads
        self._best_spread_pair = ""
        self._total_pairs_scanned = 0
        self._pairs_with_liquidity = 0

        # Per-pair daily execution limit (realistic arbitrage opportunities are rare)
        self._pair_execution_count: Dict[str, int] = {}  # pair_key -> count today
        self._pair_execution_date: str = ""  # Track which day we're on
        self._max_executions_per_pair_per_day = 5  # Max 5 arbitrage trades per DEX pair per day

        # Gas tracking
        self._last_gas_check_time: Optional[datetime] = None
        self._gas_check_interval = 60  # Check gas every 60 seconds
        self._cached_balance_eth: float = 0.0
        # Minimum ETH balance to attempt flash loan arbitrage
        # Actual cost is calculated dynamically, but we need at least this to try
        # Flash loan arb uses ~450K gas, at 30 gwei = 0.0135 ETH
        # Set minimum to 0.015 ETH to allow execution when gas is reasonable
        self._min_gas_eth: float = 0.015
        self._low_gas_warning_shown = False

        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }

    async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
        """
        Get decrypted private key from secrets manager or environment.

        Always checks if value is still encrypted and decrypts if needed.
        """
        try:
            value = None

            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                # Always re-initialize with db_pool if secrets was in bootstrap mode
                # or doesn't have a db_pool yet
                if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                    secrets.initialize(self.db_pool)
                    logger.debug(f"Re-initialized secrets manager with database pool for {key_name}")
                value = await secrets.get_async(key_name)
                if value:
                    logger.debug(f"Successfully loaded {key_name} from secrets manager")
            except Exception as e:
                logger.warning(f"Failed to get {key_name} from secrets manager: {e}")

            # Fallback to environment
            if not value:
                value = os.getenv(key_name)

            if not value:
                return None

            # Check if still encrypted (Fernet tokens start with gAAAAAB)
            # This handles cases where DB returned encrypted value without decrypting
            if value.startswith('gAAAAAB'):
                from pathlib import Path
                encryption_key = None
                key_file = Path('.encryption_key')
                if key_file.exists():
                    encryption_key = key_file.read_text().strip()
                if not encryption_key:
                    encryption_key = os.getenv('ENCRYPTION_KEY')

                if encryption_key:
                    try:
                        from cryptography.fernet import Fernet
                        f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                        return f.decrypt(value.encode()).decode()
                    except Exception as e:
                        logger.error(f"Failed to decrypt {key_name}: {e}")
                        return None
                else:
                    logger.error(f"Cannot decrypt {key_name}: no encryption key found")
                    return None

            return value
        except Exception as e:
            logger.debug(f"Error getting {key_name}: {e}")
            return None

    async def initialize(self):
        logger.info("⚖️ Initializing Arbitrage Engine...")

        # Load credentials from secrets manager (database)
        logger.info("Loading credentials from database...")
        self.private_key = await self._get_decrypted_key('PRIVATE_KEY')

        # IMPORTANT: Derive wallet address from private key to avoid mismatch issues
        # The stored WALLET_ADDRESS may not match the private key, so always derive it
        if self.private_key:
            try:
                from eth_account import Account
                # Ensure private key has 0x prefix
                pk = self.private_key if self.private_key.startswith('0x') else f'0x{self.private_key}'
                account = Account.from_key(pk)
                self.wallet_address = account.address
                masked_addr = self.wallet_address[:8] + "..." + self.wallet_address[-6:]
                logger.info(f"✅ Loaded EVM credentials from database (wallet: {masked_addr})")
                logger.debug(f"   Wallet derived from private key (ignoring stored WALLET_ADDRESS)")
            except Exception as e:
                logger.error(f"❌ Failed to derive wallet from private key: {e}")
                self.wallet_address = None
        else:
            logger.warning(f"⚠️ Missing PRIVATE_KEY in database - store via settings page")
            self.wallet_address = None

        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url.split(',')[0]))
            if self.w3.is_connected():
                # Detect chain from RPC
                self.chain_id = self.w3.eth.chain_id
                self.chain_config = CHAIN_CONFIGS.get(self.chain_id)

                if self.chain_config:
                    # Use chain-specific addresses
                    self.chain_name = self.chain_config['name']
                    self.tokens = self.chain_config['tokens']
                    self.routers = self.chain_config['routers']
                    self.arb_pairs = self.chain_config['arb_pairs']
                    self.aave_pool = self.chain_config['aave_pool']
                    self._min_gas_eth = self.chain_config['min_gas_eth']

                    logger.info(f"✅ Connected to {self.chain_name.upper()} RPC (chain_id: {self.chain_id})")
                    logger.info(f"   Tokens: {len(self.tokens)} | Routers: {len(self.routers)} | Pairs: {len(self.arb_pairs)}")
                    logger.info(f"   Min gas: {self._min_gas_eth} ETH")
                else:
                    # Unknown chain - use Ethereum defaults
                    logger.warning(f"⚠️ Unknown chain_id {self.chain_id} - using Ethereum defaults")
                    logger.info("✅ Connected to Arbitrage RPC")

                # Initialize router contracts using chain-specific addresses
                for name, address in self.routers.items():
                    try:
                        self.router_contracts[name] = self.w3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=ROUTER_ABI
                        )
                    except Exception as e:
                        logger.debug(f"Could not init {name} router: {e}")

                # Initialize Flash Loan executor
                # IMPORTANT: Flash loans require a deployed smart contract with IFlashLoanReceiver
                # The contract must implement executeOperation() callback
                # EOA wallets CANNOT receive flash loan callbacks - they will always revert
                flash_loan_contract = os.getenv('FLASH_LOAN_RECEIVER_CONTRACT')

                if self.private_key and self.wallet_address and not self.dry_run:
                    if flash_loan_contract:
                        # Convert to checksum address (web3.py requires this)
                        flash_loan_contract = Web3.to_checksum_address(flash_loan_contract)
                        # Initialize flash loan executor with BOTH wallet (for signing) AND contract (for receiver)
                        self.flash_loan_executor = FlashLoanExecutor(
                            self.w3,
                            self.private_key,
                            self.wallet_address,  # EOA wallet - signs TX and pays gas
                            flash_loan_contract   # Contract - receives flash loan callback
                        )
                        logger.info(f"⚡ Flash Loan executor initialized:")
                        logger.info(f"   Wallet (signer): {self.wallet_address[:10]}...")
                        logger.info(f"   Receiver contract: {flash_loan_contract[:10]}...")
                    else:
                        # No contract deployed - flash loans WILL NOT WORK with EOA
                        logger.warning("=" * 70)
                        logger.warning("⚠️ FLASH LOAN WARNING: No receiver contract deployed!")
                        logger.warning("   Aave flash loans require a smart contract that implements")
                        logger.warning("   IFlashLoanReceiver.executeOperation() callback.")
                        logger.warning("   EOA wallets CANNOT receive flash loan callbacks.")
                        logger.warning("")
                        logger.warning("   To enable flash loans:")
                        logger.warning("   1. Deploy a FlashLoanReceiver contract")
                        logger.warning("   2. Set FLASH_LOAN_RECEIVER_CONTRACT=<contract_address> in .env")
                        logger.warning("")
                        logger.warning("   ⚡ Flash loans DISABLED - using direct swaps only")
                        logger.warning("=" * 70)
                        self.flash_loan_executor = None
                        self.use_flash_loans = False

                    # Check wallet ETH balance for gas
                    try:
                        balance_wei = self.w3.eth.get_balance(self.wallet_address)
                        balance_eth = balance_wei / 1e18
                        if balance_eth < 0.01:
                            logger.error(f"❌ CRITICAL: Wallet has insufficient ETH for gas!")
                            logger.error(f"   Balance: {balance_eth:.6f} ETH")
                            logger.error(f"   Required: At least 0.01 ETH for flash loan gas")
                            logger.error(f"   Fund wallet: {self.wallet_address}")
                        else:
                            logger.info(f"   Wallet balance: {balance_eth:.4f} ETH")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not check wallet balance: {e}")

                    # Initialize Flashbots executor (Ethereum mainnet only)
                    # Flashbots is not available on L2s like Arbitrum
                    if self.chain_id == 1:  # Ethereum mainnet
                        self.flashbots_executor = FlashbotsExecutor(
                            self.w3,
                            self.private_key
                        )
                        await self.flashbots_executor.initialize()
                        logger.info("📦 Flashbots executor initialized")
                    else:
                        logger.info(f"ℹ️ Flashbots not available on {self.chain_name} - using direct RPC")
                        self.use_flashbots = False

            else:
                logger.warning("⚠️ Failed to connect to Arbitrage RPC")

        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Flash Loans: {'Enabled' if self.flash_loan_executor else 'Disabled'}")
        logger.info(f"   Flashbots: {'Enabled' if self.flashbots_executor else 'Disabled'}")

    async def run(self):
        self.is_running = True
        logger.info(f"⚖️ Arbitrage Engine Started [{self.chain_name.upper()}]")
        logger.info(f"   Monitoring {len(self.arb_pairs)} token pairs across {len(self.router_contracts)} DEXs")

        if not self.w3:
            logger.error("RPC not connected, arbitrage disabled.")
            return

        pair_index = 0  # Track which pair we're scanning

        while self.is_running:
            try:
                self._stats['scans'] += 1

                # Get current pair to scan (use chain-specific pairs and tokens)
                token_in_symbol, token_out_symbol = self.arb_pairs[pair_index]
                token_in = self.tokens.get(token_in_symbol)
                token_out = self.tokens.get(token_out_symbol)

                if token_in and token_out:
                    await self._check_arb_opportunity(token_in, token_out, token_in_symbol)

                # Move to next pair (round-robin)
                pair_index = (pair_index + 1) % len(self.arb_pairs)

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                # Sleep between scans (short delay to not overwhelm RPC)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Arb loop error: {e}")
                await asyncio.sleep(5)

    async def _log_stats_if_needed(self):
        """Log statistics every 5 minutes with spread visibility"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            # Enhanced logging with spread visibility
            logger.info(f"📊 ARBITRAGE STATS (Last 5 min): "
                       f"Scans: {self._stats['scans']} | "
                       f"Pairs w/Liquidity: {self._pairs_with_liquidity}/{self._total_pairs_scanned} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']}")

            # Log best spread seen (even if negative)
            if self._best_spread_seen > -999.0:  # -999 is initial value, means no spreads checked
                status = "✅ ABOVE" if self._best_spread_seen > self.min_profit_threshold else "❌ BELOW"
                sign = "+" if self._best_spread_seen >= 0 else ""
                logger.info(f"   Best spread: {sign}{self._best_spread_seen:.4%} on {self._best_spread_pair} ({status} threshold {self.min_profit_threshold:.2%})")
                if self._best_spread_seen < 0:
                    logger.info(f"   ⚠️ All spreads are NEGATIVE - arbitrage not profitable on current DEXs")
            else:
                logger.info(f"   No spreads calculated - check RPC connection and DEX liquidity")

            # Reset stats
            self._stats = {
                'scans': 0,
                'opportunities_found': 0,
                'opportunities_executed': 0,
                'last_stats_log': now
            }
            self._best_spread_seen = -999.0  # Start negative so we track even negative spreads
            self._best_spread_pair = ""
            self._total_pairs_scanned = 0
            self._pairs_with_liquidity = 0

    async def _check_arb_opportunity(self, token_in: str, token_out: str, token_symbol: str = "UNKNOWN") -> bool:
        """Check price difference between two DEXs. Returns True if opportunity found."""
        try:
            self._total_pairs_scanned += 1
            amount_in = self.flash_loan_amount  # Use configured flash loan amount

            # Ensure addresses are checksummed for web3.py compatibility
            token_in_checksum = Web3.to_checksum_address(token_in)
            token_out_checksum = Web3.to_checksum_address(token_out)

            # Query BOTH directions on all DEXs to find real arbitrage opportunities
            # Forward direction: token_in → token_out (first leg of arbitrage)
            forward_prices = {}
            forward_errors = {}
            for name, contract in self.router_contracts.items():
                try:
                    amounts = contract.functions.getAmountsOut(amount_in, [token_in_checksum, token_out_checksum]).call()
                    forward_prices[name] = amounts[1]
                except Exception as e:
                    # Track errors for diagnostic logging
                    forward_errors[name] = str(e)[:50]

            # Log diagnostic info periodically (every 60 scans = ~2 minutes)
            if self._total_pairs_scanned % 60 == 1:
                if forward_prices:
                    logger.debug(f"📊 [{token_symbol}] Forward prices: {len(forward_prices)} DEXs responded")
                else:
                    logger.warning(f"⚠️ [{token_symbol}] No forward prices from any DEX. Errors: {forward_errors}")

            if len(forward_prices) < 2:
                return False

            self._pairs_with_liquidity += 1

            # Find best forward DEX (gives most token_out for our token_in)
            best_buy_dex = max(forward_prices, key=forward_prices.get)
            forward_output = forward_prices[best_buy_dex]

            if forward_output == 0:
                return False

            # Now query reverse direction on OTHER DEXs: token_out → token_in
            # This is the actual second leg of the arbitrage
            reverse_outputs = {}
            for name, contract in self.router_contracts.items():
                if name == best_buy_dex:
                    continue  # Skip same DEX - no arbitrage within same DEX
                try:
                    amounts = contract.functions.getAmountsOut(forward_output, [token_out_checksum, token_in_checksum]).call()
                    reverse_outputs[name] = amounts[1]
                except Exception:
                    pass

            if not reverse_outputs:
                return False

            # Find best reverse DEX (gives most token_in back for our token_out)
            best_sell_dex = max(reverse_outputs, key=reverse_outputs.get)
            final_output = reverse_outputs[best_sell_dex]

            # Calculate actual profit: final_output - (borrowed_amount + flash_loan_fee)
            flash_loan_fee = amount_in * 5 // 10000  # 0.05% Aave fee
            amount_owed = amount_in + flash_loan_fee

            # Even if no profit, track the spread for visibility
            if final_output > 0:
                raw_spread = (final_output - amount_owed) / amount_in
                pair_key = f"{token_symbol} ({best_buy_dex}→{best_sell_dex})"

                # Track best spread seen (even negative ones)
                if raw_spread > self._best_spread_seen:
                    self._best_spread_seen = raw_spread
                    self._best_spread_pair = pair_key

            if final_output <= amount_owed:
                return False  # No profit possible

            profit = final_output - amount_owed
            raw_spread = profit / amount_in

            # Estimated additional costs (slippage, gas)
            estimated_costs = 0.005  # 0.5% for slippage and gas
            net_spread = raw_spread - estimated_costs

            if net_spread > self.min_profit_threshold:
                self._stats['opportunities_found'] += 1

                # Create unique key for this opportunity
                opp_key = f"{best_buy_dex}_{best_sell_dex}_{token_symbol}"
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Reset daily counters if new day
                if self._pair_execution_date != today:
                    self._pair_execution_count = {}
                    self._pair_execution_date = today
                    logger.info(f"📅 New day - reset arbitrage execution counters")

                # Check daily execution limit per pair
                current_count = self._pair_execution_count.get(opp_key, 0)
                if current_count >= self._max_executions_per_pair_per_day:
                    # Silently skip - already hit daily limit for this pair
                    return True

                # Check cooldown - don't spam same opportunity
                if self._last_opportunity_key == opp_key and self._last_opportunity_time:
                    elapsed = (now - self._last_opportunity_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        # Same opportunity within cooldown, skip silently
                        return True

                # New opportunity or cooldown expired - log and execute
                self._last_opportunity_key = opp_key
                self._last_opportunity_time = now

                # Update daily execution count
                self._pair_execution_count[opp_key] = current_count + 1
                remaining = self._max_executions_per_pair_per_day - (current_count + 1)

                logger.info(f"🚨 ARBITRAGE OPPORTUNITY [{token_symbol}]: Buy on {best_buy_dex}, Sell on {best_sell_dex}. Raw: {raw_spread:.2%}, Net: {net_spread:.2%} (#{current_count + 1} today, {remaining} remaining)")
                logger.info(f"   Path: {amount_in/1e18:.4f} {token_symbol} → {forward_output/1e18:.4f} intermediate → {final_output/1e18:.4f} {token_symbol} (profit: {profit/1e18:.4f})")
                self._stats['opportunities_executed'] += 1

                # Execute arbitrage
                await self._execute_flash_swap(
                    buy_dex=best_buy_dex,
                    sell_dex=best_sell_dex,
                    token_in=token_in,
                    token_out=token_out,
                    amount=amount_in,
                    expected_profit=net_spread,
                    token_symbol=token_symbol
                )
                return True
            return False

        except Exception as e:
            logger.error(f"Arb check failed for {token_symbol}: {e}")
            return False

    async def _check_gas_balance(self) -> Tuple[bool, float]:
        """
        Check if wallet has sufficient ETH for gas.

        Uses caching to avoid excessive RPC calls.

        Returns:
            Tuple of (has_sufficient_gas: bool, balance_eth: float)
        """
        now = datetime.now()

        # Use cached value if recent enough
        if self._last_gas_check_time:
            elapsed = (now - self._last_gas_check_time).total_seconds()
            if elapsed < self._gas_check_interval:
                return self._cached_balance_eth >= self._min_gas_eth, self._cached_balance_eth

        # Fetch fresh balance
        try:
            if not self.w3 or not self.wallet_address:
                return False, 0.0

            balance_wei = self.w3.eth.get_balance(self.wallet_address)
            self._cached_balance_eth = balance_wei / 1e18
            self._last_gas_check_time = now

            has_gas = self._cached_balance_eth >= self._min_gas_eth

            # Log warning only once when gas becomes insufficient
            if not has_gas and not self._low_gas_warning_shown:
                logger.warning(f"⚠️ Insufficient gas: {self._cached_balance_eth:.6f} ETH < {self._min_gas_eth} ETH minimum")
                logger.warning(f"   Fund wallet {self.wallet_address} to enable execution")
                self._low_gas_warning_shown = True
            elif has_gas and self._low_gas_warning_shown:
                logger.info(f"✅ Gas balance recovered: {self._cached_balance_eth:.6f} ETH")
                self._low_gas_warning_shown = False

            return has_gas, self._cached_balance_eth

        except Exception as e:
            logger.debug(f"Gas check failed: {e}")
            return False, 0.0

    async def _execute_flash_swap(
        self,
        buy_dex: str,
        sell_dex: str,
        token_in: str,
        token_out: str,
        amount: int,
        expected_profit: float,
        token_symbol: str = "UNKNOWN"
    ):
        """Execute the arbitrage trade using flash loans and Flashbots"""
        logger.info(f"⚡ Executing Arbitrage [{token_symbol}]: {buy_dex} -> {sell_dex} | Amount: {amount/1e18:.4f} ETH | Expected: +{expected_profit:.2%}")

        if self.dry_run:
            # Simulate execution
            await asyncio.sleep(0.5)
            logger.info(f"✅ Flash Swap Executed (DRY RUN) [{token_symbol}]")
            await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, "DRY_RUN", token_symbol)
            return

        # Validate credentials before live execution
        if not self.private_key or not self.wallet_address:
            logger.error(f"❌ Cannot execute - PRIVATE_KEY or WALLET_ADDRESS not configured in database")
            return

        # Check gas balance before execution
        has_gas, balance = await self._check_gas_balance()
        if not has_gas:
            logger.warning(f"⏸️ Skipping execution - insufficient gas ({balance:.6f} ETH < {self._min_gas_eth} ETH)")
            return

        # Additional check: estimate actual gas cost for flash loan transactions
        if self.use_flash_loans and self.flash_loan_executor and self.w3:
            try:
                gas_price = self.w3.eth.gas_price
                # Realistic gas limit for flash loan arbitrage:
                # - Flash loan borrow: ~100K gas
                # - DEX swap 1: ~150K gas
                # - DEX swap 2: ~150K gas
                # - Flash loan repay: ~50K gas
                # Total: ~450K gas (not 800K)
                gas_limit = 450000
                estimated_cost_eth = (gas_price * gas_limit) / 1e18
                # Add 30% buffer for gas price fluctuations and complex routes
                required_eth = estimated_cost_eth * 1.3

                if balance < required_eth:
                    logger.warning(f"⏸️ Skipping execution - insufficient ETH for gas cost")
                    logger.warning(f"   Balance: {balance:.6f} ETH | Required: {required_eth:.6f} ETH")
                    logger.warning(f"   Gas: {gas_price/1e9:.1f} gwei × {gas_limit:,} = {estimated_cost_eth:.6f} ETH + 30% buffer")
                    return
            except Exception as e:
                logger.debug(f"Gas estimation failed: {e}")

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
                logger.info(f"✅ Arbitrage executed [{token_symbol}]: {tx_hash}")
                await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, tx_hash, token_symbol)
            else:
                logger.error(f"❌ Arbitrage execution failed [{token_symbol}]")

        except Exception as e:
            logger.error(f"Arbitrage execution error [{token_symbol}]: {e}")

    async def _execute_with_flash_loan(
        self,
        buy_dex: str,
        sell_dex: str,
        token_in: str,
        token_out: str,
        amount: int
    ) -> Optional[str]:
        """
        Execute arbitrage via FlashLoanArbitrage contract.

        The contract's executeArbitrage() function will:
        1. Call Aave's flashLoanSimple (with contract as initiator - passes check!)
        2. Swap borrowed asset -> intermediate token on buyRouter
        3. Swap intermediate token -> borrowed asset on sellRouter
        4. Repay loan + fee to Aave
        5. Keep profit in contract (withdraw later)

        Args:
            buy_dex: DEX with lower price
            sell_dex: DEX with higher price
            token_in: Asset to borrow (e.g., WETH)
            token_out: Intermediate token to swap through (e.g., USDC)
            amount: Amount to borrow in wei
        """
        # Safety check: flash loan executor must be initialized with a contract address
        if not self.flash_loan_executor:
            logger.error("❌ Flash loan executor not initialized - no receiver contract deployed")
            logger.error("   Set FLASH_LOAN_RECEIVER_CONTRACT in .env and restart")
            return None

        # Get router addresses
        buy_router = ROUTERS.get(buy_dex, ROUTERS['uniswap_v2'])
        sell_router = ROUTERS.get(sell_dex, ROUTERS['sushiswap'])

        logger.info(f"   Flash loan params: borrow={token_in[:10]}..., intermediate={token_out[:10]}...")
        logger.info(f"   Route: {buy_dex} ({buy_router[:10]}...) -> {sell_dex} ({sell_router[:10]}...)")

        # Execute via contract's executeArbitrage function
        # This ensures initiator == contract address (passes the check!)
        tx_hash = await self.flash_loan_executor.execute_arbitrage(
            asset=token_in,
            amount=amount,
            buy_router=buy_router,
            sell_router=sell_router,
            intermediate_token=token_out
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

            # Get current gas price
            gas_price = self.w3.eth.gas_price
            # Add 10% buffer for faster inclusion
            gas_price = int(gas_price * 1.1)

            current_nonce = self.w3.eth.get_transaction_count(self.wallet_address)

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
                'gasPrice': gas_price,
                'nonce': current_nonce,
                'chainId': self.w3.eth.chain_id
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
                'gasPrice': gas_price,
                'nonce': current_nonce + 1,
                'chainId': self.w3.eth.chain_id
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
        tx_hash: str,
        token_symbol: str = "UNKNOWN"
    ):
        """Log arbitrage trade to database with REALISTIC P&L calculation"""
        if not self.db_pool:
            return

        try:
            import uuid
            amount_eth = amount / 1e18

            # Get real ETH price - CRITICAL: don't use fake prices
            eth_price = await self.price_fetcher.get_price('eth')
            if not eth_price:
                logger.warning(f"Cannot log trade - ETH price unavailable")
                return

            # Realistic cost deductions
            FLASH_LOAN_FEE_PCT = 0.0005  # 0.05% Aave fee
            SLIPPAGE_ESTIMATE_PCT = 0.006  # 0.3% x 2 swaps = 0.6%
            GAS_COST_USD = 15.0  # Estimated gas cost

            # Calculate gross profit
            entry_usd = amount_eth * eth_price
            gross_profit_pct = profit_pct

            # Deduct realistic costs for net profit
            flash_loan_cost = entry_usd * FLASH_LOAN_FEE_PCT
            slippage_cost = entry_usd * gross_profit_pct * 0.3  # Assume 30% of spread lost to slippage
            total_costs = flash_loan_cost + slippage_cost + GAS_COST_USD

            net_profit_usd = (entry_usd * gross_profit_pct) - total_costs
            net_profit_pct = net_profit_usd / entry_usd if entry_usd > 0 else 0
            exit_usd = entry_usd + net_profit_usd

            # Use ETH price as entry, calculate exit based on net profit
            entry_price = eth_price
            exit_price = eth_price * (1 + net_profit_pct)

            logger.info(
                f"💰 Arb value [{token_symbol}]: {amount_eth:.4f} ETH @ ${eth_price:.2f} = ${entry_usd:.2f} | "
                f"Gross: +{gross_profit_pct:.2%} | Costs: ${total_costs:.2f} | Net: ${net_profit_usd:.2f}"
            )

            trade_id = f"arb_{uuid.uuid4().hex[:12]}"

            async with self.db_pool.acquire() as conn:
                # Insert into dedicated arbitrage_trades table
                await conn.execute("""
                    INSERT INTO arbitrage_trades (
                        trade_id, token_address, chain, buy_dex, sell_dex,
                        side, entry_price, exit_price, amount, amount_eth,
                        entry_usd, exit_usd, profit_loss, profit_loss_pct, spread_pct,
                        status, is_simulated, entry_timestamp, exit_timestamp,
                        tx_hash, eth_price_at_trade, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                """,
                    trade_id,
                    token,
                    self.chain_name,  # Use chain from engine instance
                    buy_dex,
                    sell_dex,
                    'buy',
                    entry_price,
                    exit_price,
                    amount_eth,
                    amount_eth,
                    entry_usd,
                    exit_usd,
                    net_profit_usd,  # Use NET profit after costs
                    net_profit_pct * 100,  # Use NET profit % after costs
                    gross_profit_pct * 100,  # Keep gross spread for reference
                    'closed',
                    self.dry_run,
                    datetime.now(),
                    datetime.now(),
                    tx_hash,
                    eth_price,
                    json.dumps({
                        'dry_run': self.dry_run,
                        'chain': self.chain_name,
                        'chain_id': self.chain_id,
                        'token_symbol': token_symbol,
                        'gross_profit_pct': gross_profit_pct * 100,
                        'flash_loan_cost': flash_loan_cost,
                        'slippage_cost': slippage_cost,
                        'gas_cost': GAS_COST_USD,
                        'total_costs': total_costs
                    })
                )
            logger.debug(f"💾 Logged to arbitrage_trades: {trade_id} [{token_symbol}]")
        except Exception as e:
            logger.error(f"Error logging arb trade: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False

        # Close Flashbots executor
        if self.flashbots_executor:
            await self.flashbots_executor.close()

        logger.info("🛑 Arbitrage Engine Stopped")
