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
    # NOTE: Uniswap V3 SwapRouter (0xE592427A...) does NOT support getAmountsOut (V2 interface).
    # V3 requires the Quoter contract (0xb27308f9...) for price quotes.
    # Removed to prevent constant 'execution reverted' errors.
    # TODO: Add V3 Quoter integration for better price discovery.
}

ROUTERS_ARBITRUM = {
    # Note: All V2-compatible routers for getAmountsOut/swapExactTokensForTokens interface
    'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',  # SushiSwap V2 Router (works!)
    'camelot': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d',    # Camelot DEX Router (V2 interface)
    'zyberswap': '0x16e71B13fE6079B4312063F7E81F76d165Ad32Ad',  # Zyberswap V2 Router
}

ROUTERS_BASE = {
    # Note: Using V2-compatible routers for getAmountsOut/swapExactTokensForTokens interface
    # V3 routers require different interface (Quoter + SwapRouter)
    'sushiswap': '0x6BDED42c6DA8FBf0d2bA55B2fa120C5e0c8D7891',   # SushiSwap V2 Router (works!)
    'baseswap': '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86',    # BaseSwap V2 Router
    'swapbased': '0xaaa3b1F1bd7BCc97fD1917c18ADE665C5D31F066',   # SwapBased V2 Router
}

# ═══════════════════════════════════════════════════════════════════════════════
# AAVE V3 POOL ADDRESSES PER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
AAVE_POOLS = {
    'ethereum': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
    'arbitrum': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
    'base': '0xA238Dd80C259a72e81d7e4664a9801593F98d1c5',
}

# ═══════════════════════════════════════════════════════════════════════════════
# BASE (Chain ID: 8453)
# Coinbase L2, very low gas costs, growing DeFi ecosystem
# ═══════════════════════════════════════════════════════════════════════════════
TOKENS_BASE = {
    # Major tokens - Base native/bridged addresses
    'WETH': '0x4200000000000000000000000000000000000006',  # Wrapped ETH on Base
    'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',  # Native USDC (Circle)
    'USDbC': '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA', # Bridged USDC
    'DAI': '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',   # DAI Stablecoin

    # Native Base tokens
    'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631',  # Aerodrome
    'BRETT': '0x532f27101965dd16442E59d40670FaF5eBB142E4', # Brett (meme)
    'DEGEN': '0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed', # Degen
    'TOSHI': '0xAC1Bd2486aAf3B5C0fc3Fd868558b082a531B2B4', # Toshi

    # Bridged DeFi tokens
    'cbETH': '0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22', # Coinbase ETH
    'rETH': '0xB6fe221Fe9EeF5aBa221c348bA20A1Bf5e73624c',  # Rocket Pool ETH
}

# ═══════════════════════════════════════════════════════════════════════════════
# ARBITRAGE PAIRS PER CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
ARB_PAIRS_ETHEREUM = [
    # High liquidity pairs with verified V2 liquidity on Uniswap/SushiSwap
    ('WETH', 'USDC'),
    ('WETH', 'USDT'),
    ('WETH', 'DAI'),
    ('WBTC', 'WETH'),

    # Major DeFi tokens with good V2 liquidity
    ('LINK', 'WETH'),
    ('UNI', 'WETH'),
    ('AAVE', 'WETH'),
    ('SUSHI', 'WETH'),
    ('CRV', 'WETH'),

    # Stablecoin pairs (low volatility but consistent)
    ('USDC', 'USDT'),
    ('USDC', 'DAI'),
    ('DAI', 'USDT'),

    # NOTE: Removed low-liquidity or problematic pairs:
    # ('MATIC', 'WETH'),  # Most liquidity on Polygon now
    # ('SNX', 'WETH'),    # Low V2 liquidity
    # ('COMP', 'WETH'),   # Low V2 liquidity
    # ('MKR', 'WETH'),    # Low V2 liquidity
    # ('BAL', 'WETH'),    # Most liquidity on Balancer V2
    # ('YFI', 'WETH'),    # Low V2 liquidity
    # ('GRT', 'WETH'),    # Can have reverts on some DEXs
    # ('ENS', 'WETH'),    # Low V2 liquidity
    # ('SHIB', 'WETH'),   # High slippage, fee-on-transfer issues
    # ('PEPE', 'WETH'),   # High slippage
    # ('LDO', 'WETH'),    # Low V2 liquidity
]

ARB_PAIRS_ARBITRUM = [
    # High liquidity pairs (verified on SushiSwap, Camelot, Zyberswap)
    ('WETH', 'USDC'),
    ('WETH', 'USDT'),
    ('WETH', 'DAI'),
    ('WBTC', 'WETH'),
    ('ARB', 'WETH'),       # Native ARB token - high volume
    ('ARB', 'USDC'),       # ARB/stablecoin

    # Native Arbitrum DeFi tokens with verified V2 liquidity
    ('GMX', 'WETH'),       # GMX - major Arbitrum protocol

    # NOTE: Some tokens removed due to liquidity issues on V2 routers:
    # ('MAGIC', 'WETH'),   # Low V2 liquidity
    # ('RDNT', 'WETH'),    # Low V2 liquidity
    # ('PENDLE', 'WETH'),  # Low V2 liquidity

    # Bridged DeFi tokens with liquidity
    ('LINK', 'WETH'),

    # NOTE: These have low V2 liquidity on Arbitrum:
    # ('UNI', 'WETH'),
    # ('AAVE', 'WETH'),
    # ('CRV', 'WETH'),
    # ('SUSHI', 'WETH'),

    # Stablecoin pairs (good for low-risk arb)
    ('USDC', 'USDT'),
    ('USDC', 'DAI'),
]

ARB_PAIRS_BASE = [
    # High liquidity pairs (verified on SushiSwap, BaseSwap, SwapBased)
    ('WETH', 'USDC'),
    ('WETH', 'USDbC'),     # Bridged USDC
    ('WETH', 'DAI'),
    ('cbETH', 'WETH'),     # Coinbase ETH - good liquidity

    # NOTE: rETH has very low liquidity on Base V2 DEXs - removed to avoid errors
    # ('rETH', 'WETH'),    # Rocket Pool ETH - NO LIQUIDITY on Base V2 DEXs

    # Native Base tokens with verified liquidity
    ('AERO', 'WETH'),      # Aerodrome - major Base DEX token
    ('AERO', 'USDC'),

    # Meme tokens - only include if they have DEX liquidity
    # ('BRETT', 'WETH'),   # Check liquidity before enabling
    ('DEGEN', 'WETH'),     # DEGEN has liquidity on some DEXs
    # ('TOSHI', 'WETH'),   # Check liquidity before enabling

    # Stablecoin pairs (low volatility but consistent)
    ('USDC', 'USDbC'),     # Native vs bridged USDC - low risk arb
    ('USDC', 'DAI'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN CONFIG AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════
CHAIN_CONFIGS = {
    1: {  # Ethereum Mainnet
        'name': 'ethereum',
        'display_name': 'Ethereum',
        'tokens': TOKENS_ETHEREUM,
        'routers': ROUTERS_ETHEREUM,
        'aave_pool': AAVE_POOLS['ethereum'],
        'arb_pairs': ARB_PAIRS_ETHEREUM,
        'native_symbol': 'ETH',
        'min_gas_eth': 0.015,  # ~$30-50 for flash loan tx
        'flash_loan_env_key': 'FLASH_LOAN_RECEIVER_CONTRACT_ETH',
        'flash_loan_env_fallback': 'FLASH_LOAN_RECEIVER_CONTRACT',  # Backwards compat
    },
    42161: {  # Arbitrum One
        'name': 'arbitrum',
        'display_name': 'Arbitrum',
        'tokens': TOKENS_ARBITRUM,
        'routers': ROUTERS_ARBITRUM,
        'aave_pool': AAVE_POOLS['arbitrum'],
        'arb_pairs': ARB_PAIRS_ARBITRUM,
        'native_symbol': 'ETH',
        'min_gas_eth': 0.0005,  # ~$1 for flash loan tx (95% cheaper!)
        'flash_loan_env_key': 'FLASH_LOAN_RECEIVER_CONTRACT_ARB',
        'flash_loan_env_fallback': None,
    },
    8453: {  # Base
        'name': 'base',
        'display_name': 'Base',
        'tokens': TOKENS_BASE,
        'routers': ROUTERS_BASE,
        'aave_pool': AAVE_POOLS['base'],
        'arb_pairs': ARB_PAIRS_BASE,
        'native_symbol': 'ETH',
        'min_gas_eth': 0.0003,  # ~$0.50 for flash loan tx (even cheaper than Arbitrum!)
        'flash_loan_env_key': 'FLASH_LOAN_RECEIVER_CONTRACT_BASE',
        'flash_loan_env_fallback': None,
    },
}

# Default to Ethereum for backwards compatibility
TOKENS = TOKENS_ETHEREUM

# Backwards compatibility - default to Ethereum
ARB_PAIRS = ARB_PAIRS_ETHEREUM
ROUTERS = ROUTERS_ETHEREUM
AAVE_V3_POOL = AAVE_POOLS['ethereum']

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN DECIMALS - Required for correct amount scaling in spread calculations
# ═══════════════════════════════════════════════════════════════════════════════
TOKEN_DECIMALS = {
    # 18 decimals (standard ERC20)
    'WETH': 18, 'DAI': 18, 'FRAX': 18, 'LUSD': 18,
    'LINK': 18, 'UNI': 18, 'AAVE': 18, 'CRV': 18, 'SUSHI': 18,
    'SNX': 18, 'COMP': 18, 'MKR': 18, 'BAL': 18, 'YFI': 18,
    'INCH': 18, 'GRT': 18, 'ENS': 18, 'LDO': 18,
    'MATIC': 18, 'ARB': 18, 'OP': 18,
    'SHIB': 18, 'PEPE': 18,
    'GMX': 18, 'MAGIC': 18, 'RDNT': 18, 'PENDLE': 18,
    'AERO': 18, 'BRETT': 18, 'DEGEN': 18, 'TOSHI': 18,
    'cbETH': 18, 'rETH': 18,
    'TUSD': 18,
    # 8 decimals
    'WBTC': 8,
    # 6 decimals
    'USDC': 6, 'USDT': 6, 'USDbC': 6, 'USDC_BRIDGED': 6,
}

# Scan amounts per token (approximate ~$25k USD equivalent for consistent spread comparison)
# These are used for getAmountsOut queries, not for actual flash loan execution
# NOTE: These are BASE amounts at 10 ETH equivalent. They get scaled by the configured
# flash_loan_amount in the engine (e.g., if flash_loan_amount=1 ETH, amounts scale to ~$2.5k)
TOKEN_SCAN_AMOUNTS_BASE = {
    'WETH': 10 * 10**18,          # 10 ETH (base reference amount)
    'WBTC': 40 * 10**6,           # 0.4 WBTC (8 decimals) ≈ $25k
    'USDC': 25000 * 10**6,        # 25,000 USDC (6 decimals)
    'USDT': 25000 * 10**6,        # 25,000 USDT (6 decimals)
    'USDbC': 25000 * 10**6,       # 25,000 USDbC (6 decimals)
    'USDC_BRIDGED': 25000 * 10**6, # 25,000 USDC.e (6 decimals)
    'DAI': 25000 * 10**18,        # 25,000 DAI (18 decimals)
    'FRAX': 25000 * 10**18,       # 25,000 FRAX
    'LUSD': 25000 * 10**18,       # 25,000 LUSD
    'cbETH': 10 * 10**18,         # 10 cbETH (~10 ETH equivalent)
    'rETH': 10 * 10**18,          # 10 rETH (~10 ETH equivalent)
}
# Base reference: these amounts assume flash_loan_amount = 10 ETH
_BASE_FLASH_LOAN_ETH = 10

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

    def __init__(self, w3: Web3, private_key: str, wallet_address: str, receiver_contract: str, logger_instance=None):
        self.w3 = w3
        self.private_key = private_key
        self.wallet_address = wallet_address  # EOA for signing/gas (must be contract owner)
        self.receiver_contract = receiver_contract  # FlashLoanArbitrage contract address
        self.flash_loan_contract = None
        self._pending_nonce = None  # Track nonce to avoid collisions
        self._last_tx_time = None
        # Use passed logger or fall back to module logger
        self.logger = logger_instance or logger

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
            Contract returns -1 for no liquidity on buy DEX, -2 for no liquidity on sell DEX
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

            # Log detailed results for debugging
            if profit == -1:
                self.logger.warning(f"   ⚠️ Simulation: No liquidity on BUY router ({buy_router[:10]}...)")
            elif profit == -2:
                self.logger.warning(f"   ⚠️ Simulation: No liquidity on SELL router ({sell_router[:10]}...)")
            elif profit <= 0:
                self.logger.info(f"   📊 Simulation: Profit={profit/1e18:.6f} ETH (not profitable)")
            else:
                self.logger.info(f"   📊 Simulation: Profit={profit/1e18:.6f} ETH")

            return profit
        except Exception as e:
            self.logger.warning(f"   ⚠️ Simulation call failed: {str(e)[:100]}")
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
            self.logger.error("Flash loan contract not initialized")
            return None

        try:
            # First simulate to check profitability (saves gas on failures)
            expected_profit = await self.check_arbitrage_profit(
                asset, amount, buy_router, sell_router, intermediate_token
            )

            if expected_profit is not None and expected_profit <= 0:
                self.logger.warning(f"⚠️ Simulation shows no profit ({expected_profit}), skipping execution")
                return None

            # Get EIP-1559 gas pricing for better reliability
            # Using EIP-1559 prevents stuck transactions when base fee rises
            latest_block = self.w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', self.w3.eth.gas_price)

            # Priority fee (tip) - 2 gwei is usually enough for normal inclusion
            priority_fee = 2 * 10**9  # 2 gwei

            # Max fee: base fee + 50% buffer + priority fee (handles fee spikes)
            max_fee = int(base_fee * 1.5) + priority_fee

            self.logger.info(f"   Gas pricing: base={base_fee/1e9:.2f} gwei, maxFee={max_fee/1e9:.2f} gwei, priority={priority_fee/1e9:.1f} gwei")

            # CRITICAL: Verify profit exceeds gas cost + safety buffer
            # Flash loan reverts are caused by profit being consumed by gas or price movement
            # Realistic gas for flash loan arbitrage: ~450K (not 800K)
            gas_limit = 450000
            gas_cost_wei = max_fee * gas_limit
            # Require profit to be 30% higher than gas cost to account for:
            # - Price movement during block inclusion
            # - Slippage in actual execution vs simulation
            min_profit_required = int(gas_cost_wei * 1.3)

            if expected_profit is not None and expected_profit < min_profit_required:
                gas_cost_eth = gas_cost_wei / 1e18
                profit_eth = expected_profit / 1e18
                self.logger.warning(f"⚠️ Profit too low to cover gas + buffer, skipping execution")
                self.logger.warning(f"   Expected profit: {profit_eth:.6f} ETH | Gas cost: {gas_cost_eth:.6f} ETH")
                self.logger.warning(f"   Required: {min_profit_required/1e18:.6f} ETH (1.3x gas)")
                return None

            # Rate limit: wait at least 12 seconds between transactions (1 block)
            if self._last_tx_time:
                elapsed = (datetime.now() - self._last_tx_time).total_seconds()
                if elapsed < 12:
                    wait_time = 12 - elapsed
                    self.logger.info(f"⏳ Waiting {wait_time:.1f}s for block confirmation...")
                    await asyncio.sleep(wait_time)

            # Build transaction to call contract's executeArbitrage function
            # Use EIP-1559 gas parameters for better reliability
            tx = self.flash_loan_contract.functions.executeArbitrage(
                Web3.to_checksum_address(asset),
                amount,
                Web3.to_checksum_address(buy_router),
                Web3.to_checksum_address(sell_router),
                Web3.to_checksum_address(intermediate_token)
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 800000,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
                'nonce': self._get_next_nonce(),
                'chainId': self.w3.eth.chain_id
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            self._last_tx_time = datetime.now()

            self.logger.info(f"⚡ Flash loan TX sent: {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            error_msg = str(e)
            if 'replacement transaction underpriced' in error_msg:
                self.logger.warning(f"⚠️ Nonce collision detected, resetting nonce tracker")
                self._pending_nonce = None  # Reset to force fresh nonce fetch
            self.logger.error(f"Flash loan execution failed: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
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

class EVMArbitrageEngine:
    """
    Base EVM Arbitrage Engine with flash loan and Flashbots support.

    Features:
    - Multi-chain support (Ethereum, Arbitrum, Base)
    - Multi-DEX price monitoring
    - Aave flash loans for capital-efficient arb
    - Flashbots for MEV protection (Ethereum only)
    - Configurable profit thresholds
    - Chain-specific flash loan contracts

    Subclasses: ETHArbitrageEngine, ARBArbitrageEngine, BaseArbitrageEngine
    """

    # Override in subclasses for chain-specific configuration
    CHAIN_NAME = 'ethereum'
    EXPECTED_CHAIN_ID = 1
    RPC_PROVIDER_KEY = 'ETHEREUM_RPC'
    RPC_ENV_KEY = 'ETHEREUM_RPC_URL'
    RPC_ENV_FALLBACK = 'WEB3_PROVIDER_URL'
    LOGGER_NAME = 'ETHArbitrageEngine'

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.w3 = None

        # Chain-specific logger
        self.logger = logging.getLogger(self.LOGGER_NAME)

        # Chain configuration
        self.chain_name = self.CHAIN_NAME
        self.chain_id = None  # Set in initialize() from RPC
        self.chain_config = CHAIN_CONFIGS.get(self.EXPECTED_CHAIN_ID, {})

        # Chain-specific addresses from config
        self.tokens = self.chain_config.get('tokens', TOKENS_ETHEREUM)
        self.routers = self.chain_config.get('routers', ROUTERS_ETHEREUM)
        self.arb_pairs = self.chain_config.get('arb_pairs', ARB_PAIRS_ETHEREUM)
        self.aave_pool = self.chain_config.get('aave_pool', AAVE_POOLS['ethereum'])

        # Get RPC URL from config - support chain-specific RPC keys
        self.rpc_url = config.get('rpc_url')
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                self.rpc_url = RPCProvider.get_rpc_sync(self.RPC_PROVIDER_KEY)
            except Exception:
                pass
        if not self.rpc_url:
            # Fallback to environment variables
            self.rpc_url = os.getenv(self.RPC_ENV_KEY)
            if not self.rpc_url and self.RPC_ENV_FALLBACK:
                self.rpc_url = os.getenv(self.RPC_ENV_FALLBACK)

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

        # Flash loan amount: read from config (in ETH units), convert to wei
        # CRITICAL: 10 ETH ($25k) causes massive price impact on L2 V2 pools
        # Arbitrum/Base V2 pools often have only $50-200k liquidity
        flash_loan_eth = config.get('flash_loan_amount', 10)
        self.flash_loan_amount = int(flash_loan_eth * 10**18)
        self._flash_loan_eth = flash_loan_eth  # Store ETH units for scan amount scaling

        # Pre-compute scaled scan amounts based on configured flash loan size
        # Scale proportionally: if flash_loan=1 ETH (vs base 10), all amounts scale by 0.1x
        scale = flash_loan_eth / _BASE_FLASH_LOAN_ETH
        self._scan_amounts = {}
        for token, base_amount in TOKEN_SCAN_AMOUNTS_BASE.items():
            self._scan_amounts[token] = max(1, int(base_amount * scale))

        # Price fetcher for real-time prices
        self.price_fetcher = PriceFetcher()

        # Rate limiting for logging and execution
        self._last_opportunity_time = None
        self._last_opportunity_key = None
        self._opportunity_cooldown = 45  # 45s cooldown for same opportunity (was 300s=5min, too aggressive)

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

        # Dynamic liquidity blacklist - pairs that consistently fail
        # Format: {pair_key: (fail_count, last_fail_time)}
        self._liquidity_blacklist: Dict[str, Tuple[int, datetime]] = {}
        self._blacklist_threshold = 5  # Blacklist after 5 consecutive failures
        self._blacklist_duration = 3600  # Unblacklist after 1 hour (liquidity may return)

        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }

        # Telegram alerts - initialized in initialize() method
        self.telegram_alerts = None

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
        self.logger.info(f"⚖️ Initializing {self.LOGGER_NAME}...")

        # Load credentials from secrets manager (database)
        self.logger.info("Loading credentials from database...")
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
                self.logger.info(f"✅ Loaded EVM credentials from database (wallet: {masked_addr})")
                self.logger.debug(f"   Wallet derived from private key (ignoring stored WALLET_ADDRESS)")
            except Exception as e:
                self.logger.error(f"❌ Failed to derive wallet from private key: {e}")
                self.wallet_address = None
        else:
            self.logger.warning(f"⚠️ Missing PRIVATE_KEY in database - store via settings page")
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

                    self.logger.info(f"✅ Connected to {self.chain_name.upper()} RPC (chain_id: {self.chain_id})")
                    self.logger.info(f"   Tokens: {len(self.tokens)} | Routers: {len(self.routers)} | Pairs: {len(self.arb_pairs)}")
                    self.logger.info(f"   Min gas: {self._min_gas_eth} ETH")
                else:
                    # Unknown chain - use Ethereum defaults
                    self.logger.warning(f"⚠️ Unknown chain_id {self.chain_id} - using Ethereum defaults")
                    self.logger.info("✅ Connected to Arbitrage RPC")

                # Initialize router contracts using chain-specific addresses
                for name, address in self.routers.items():
                    try:
                        self.router_contracts[name] = self.w3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=ROUTER_ABI
                        )
                    except Exception as e:
                        self.logger.debug(f"Could not init {name} router: {e}")

                # Initialize Flash Loan executor
                # IMPORTANT: Flash loans require a deployed smart contract with IFlashLoanReceiver
                # The contract must implement executeOperation() callback
                # EOA wallets CANNOT receive flash loan callbacks - they will always revert
                #
                # Chain-specific flash loan contracts:
                # - FLASH_LOAN_RECEIVER_CONTRACT_ETH for Ethereum
                # - FLASH_LOAN_RECEIVER_CONTRACT_ARB for Arbitrum
                # - FLASH_LOAN_RECEIVER_CONTRACT_BASE for Base
                # - FLASH_LOAN_RECEIVER_CONTRACT as fallback (for backwards compat)
                flash_loan_env_key = self.chain_config.get('flash_loan_env_key', 'FLASH_LOAN_RECEIVER_CONTRACT')
                flash_loan_env_fallback = self.chain_config.get('flash_loan_env_fallback')
                flash_loan_contract = os.getenv(flash_loan_env_key)
                if not flash_loan_contract and flash_loan_env_fallback:
                    flash_loan_contract = os.getenv(flash_loan_env_fallback)

                if self.private_key and self.wallet_address and not self.dry_run:
                    if flash_loan_contract:
                        # Convert to checksum address (web3.py requires this)
                        flash_loan_contract = Web3.to_checksum_address(flash_loan_contract)
                        # Initialize flash loan executor with BOTH wallet (for signing) AND contract (for receiver)
                        self.flash_loan_executor = FlashLoanExecutor(
                            self.w3,
                            self.private_key,
                            self.wallet_address,  # EOA wallet - signs TX and pays gas
                            flash_loan_contract,  # Contract - receives flash loan callback
                            logger_instance=self.logger  # Use engine's logger for visibility
                        )
                        self.logger.info(f"⚡ Flash Loan executor initialized:")
                        self.logger.info(f"   Wallet (signer): {self.wallet_address[:10]}...")
                        self.logger.info(f"   Receiver contract: {flash_loan_contract[:10]}...")
                    else:
                        # No contract deployed - flash loans WILL NOT WORK with EOA
                        self.logger.warning("=" * 70)
                        self.logger.warning(f"⚠️ FLASH LOAN WARNING: No receiver contract for {self.chain_name.upper()}!")
                        self.logger.warning("   Aave flash loans require a smart contract that implements")
                        self.logger.warning("   IFlashLoanReceiver.executeOperation() callback.")
                        self.logger.warning("   EOA wallets CANNOT receive flash loan callbacks.")
                        self.logger.warning("")
                        self.logger.warning("   To enable flash loans:")
                        self.logger.warning(f"   1. Deploy FlashLoanReceiver contract on {self.chain_name.upper()}")
                        self.logger.warning(f"   2. Set {flash_loan_env_key}=<contract_address> in .env")
                        self.logger.warning("")
                        self.logger.warning("   ⚡ Flash loans DISABLED - using direct swaps only")
                        self.logger.warning("=" * 70)
                        self.flash_loan_executor = None
                        self.use_flash_loans = False

                    # Check wallet ETH balance for gas (use chain-specific minimum)
                    try:
                        balance_wei = self.w3.eth.get_balance(self.wallet_address)
                        balance_eth = balance_wei / 1e18
                        # Use chain-specific minimum gas requirement with 2x safety buffer
                        min_required = self._min_gas_eth * 2
                        if balance_eth < min_required:
                            self.logger.error(f"❌ CRITICAL: Wallet has insufficient ETH for gas!")
                            self.logger.error(f"   Balance: {balance_eth:.6f} ETH")
                            self.logger.error(f"   Required: At least {min_required:.4f} ETH for {self.chain_name} flash loan gas")
                            self.logger.error(f"   Fund wallet: {self.wallet_address}")
                        else:
                            self.logger.info(f"   Wallet balance: {balance_eth:.4f} ETH (min: {min_required:.4f} ETH)")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not check wallet balance: {e}")

                    # Initialize Flashbots executor (Ethereum mainnet only)
                    # Flashbots is not available on L2s like Arbitrum/Base
                    if self.chain_id == 1:  # Ethereum mainnet
                        self.flashbots_executor = FlashbotsExecutor(
                            self.w3,
                            self.private_key
                        )
                        await self.flashbots_executor.initialize()
                        self.logger.info("📦 Flashbots executor initialized")
                    else:
                        self.logger.info(f"ℹ️ Flashbots not available on {self.chain_name} - using direct RPC")
                        self.use_flashbots = False

            else:
                self.logger.warning("⚠️ Failed to connect to Arbitrage RPC")

        self.logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        self.logger.info(f"   Flash Loan Amount: {self._flash_loan_eth} ETH (scan scale: {self._flash_loan_eth/_BASE_FLASH_LOAN_ETH:.2f}x)")
        self.logger.info(f"   Flash Loans: {'Enabled' if self.flash_loan_executor else 'Disabled'}")
        self.logger.info(f"   Flashbots: {'Enabled' if self.flashbots_executor else 'Disabled'}")

        # Initialize Telegram alerts for arbitrage notifications
        try:
            from .arbitrage_alerts import ArbitrageTelegramAlerts, ArbitrageChain
            self.telegram_alerts = ArbitrageTelegramAlerts()
            if self.telegram_alerts.enabled:
                self.logger.info(f"📱 Telegram alerts enabled for {self.chain_name.upper()} arbitrage")
        except Exception as e:
            self.logger.warning(f"Telegram alerts not available: {e}")
            self.telegram_alerts = None

    async def run(self):
        self.is_running = True
        self.logger.info(f"⚖️ {self.LOGGER_NAME} Started [{self.chain_name.upper()}]")
        self.logger.info(f"   Monitoring {len(self.arb_pairs)} token pairs across {len(self.router_contracts)} DEXs")

        if not self.w3:
            self.logger.error("RPC not connected, arbitrage disabled.")
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
                    await self._check_arb_opportunity(token_in, token_out, token_in_symbol, token_out_symbol)

                # Move to next pair (round-robin)
                pair_index = (pair_index + 1) % len(self.arb_pairs)

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                # Sleep between scans - adaptive delay based on spread activity
                # In slow-scan mode (30min+ stale negative spreads), use longer delay
                # to conserve RPC quota on chains with no V2 pool activity
                scan_delay = 2
                if hasattr(self, '_slow_scan_active') and self._slow_scan_active:
                    scan_delay = 10  # 5x slower when spreads are static
                await asyncio.sleep(scan_delay)

            except Exception as e:
                self.logger.error(f"Arb loop error: {e}")
                await asyncio.sleep(5)

    async def _log_stats_if_needed(self):
        """Log statistics every 5 minutes with spread visibility"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            # Enhanced logging with spread visibility
            self.logger.info(f"📊 [{self.chain_name.upper()}] STATS (Last 5 min): "
                       f"Scans: {self._stats['scans']} | "
                       f"Pairs w/Liquidity: {self._pairs_with_liquidity}/{self._total_pairs_scanned} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']}")

            # Log best spread seen (even if negative)
            if self._best_spread_seen > -999.0:  # -999 is initial value, means no spreads checked
                status = "✅ ABOVE" if self._best_spread_seen > self.min_profit_threshold else "❌ BELOW"
                sign = "+" if self._best_spread_seen >= 0 else ""
                self.logger.info(f"   Best spread: {sign}{self._best_spread_seen:.4%} on {self._best_spread_pair} ({status} threshold {self.min_profit_threshold:.2%})")
                if self._best_spread_seen < 0:
                    self.logger.info(f"   ⚠️ All spreads are NEGATIVE - arbitrage not profitable on current DEXs")

                # Stale data detection: track if best spread hasn't changed
                if not hasattr(self, '_prev_best_spread'):
                    self._prev_best_spread = None
                    self._stale_spread_count = 0

                if self._prev_best_spread is not None:
                    spread_diff = abs(self._best_spread_seen - self._prev_best_spread)
                    if spread_diff < 0.0001:  # Less than 0.01% change
                        self._stale_spread_count += 1
                        if self._stale_spread_count >= 3:  # 15 minutes of identical spreads
                            self.logger.warning(
                                f"   ⚠️ STALE DATA: Best spread unchanged for {self._stale_spread_count * 5}min "
                                f"- RPC may be returning cached data or pools have no activity"
                            )
                            # After 20 minutes of stale data, check RPC liveness
                            if self._stale_spread_count >= 4 and self.w3:
                                try:
                                    block = self.w3.eth.block_number
                                    self.logger.info(f"   ℹ️ RPC is alive (block #{block}) - spreads are genuinely static on these DEXs")
                                except Exception as e:
                                    self.logger.error(f"   ❌ RPC connection may be dead: {e}")

                            # After 30 minutes of stale + negative spreads, switch to slow-scan mode
                            # Reduce RPC calls since pools clearly have no activity
                            if self._stale_spread_count >= 6 and self._best_spread_seen < 0:
                                self.logger.warning(
                                    f"   💤 SLOW-SCAN MODE: 30min+ stale negative spreads on {self.chain} "
                                    f"- reducing scan frequency to conserve RPC quota"
                                )
                                # Double the scan interval (tracked via attribute)
                                if not hasattr(self, '_slow_scan_active'):
                                    self._slow_scan_active = False
                                self._slow_scan_active = True
                    else:
                        self._stale_spread_count = 0
                        # Reset slow scan when spreads start moving again
                        if hasattr(self, '_slow_scan_active') and self._slow_scan_active:
                            self.logger.info(f"   ✅ Spreads moving again - resuming normal scan frequency")
                            self._slow_scan_active = False

                self._prev_best_spread = self._best_spread_seen
            else:
                self.logger.info(f"   No spreads calculated - check RPC connection and DEX liquidity")

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

    def _is_pair_blacklisted(self, pair_key: str) -> bool:
        """Check if a pair is currently blacklisted due to liquidity failures."""
        if pair_key not in self._liquidity_blacklist:
            return False

        fail_count, last_fail_time = self._liquidity_blacklist[pair_key]

        # Check if blacklist has expired
        elapsed = (datetime.now() - last_fail_time).total_seconds()
        if elapsed > self._blacklist_duration:
            # Blacklist expired - remove and allow scanning
            del self._liquidity_blacklist[pair_key]
            self.logger.info(f"🔄 [{pair_key}] Removed from blacklist after {self._blacklist_duration/60:.0f}min")
            return False

        return fail_count >= self._blacklist_threshold

    def _update_liquidity_blacklist(self, pair_key: str, has_liquidity: bool):
        """Update blacklist based on liquidity check result."""
        if has_liquidity:
            # Reset fail count on success
            if pair_key in self._liquidity_blacklist:
                del self._liquidity_blacklist[pair_key]
        else:
            # Increment fail count
            if pair_key in self._liquidity_blacklist:
                fail_count, _ = self._liquidity_blacklist[pair_key]
                self._liquidity_blacklist[pair_key] = (fail_count + 1, datetime.now())
            else:
                self._liquidity_blacklist[pair_key] = (1, datetime.now())

            # Log when pair gets blacklisted
            fail_count, _ = self._liquidity_blacklist[pair_key]
            if fail_count == self._blacklist_threshold:
                self.logger.warning(f"⛔ [{pair_key}] Blacklisted for {self._blacklist_duration/60:.0f}min (no liquidity)")

    async def _check_arb_opportunity(self, token_in: str, token_out: str, token_symbol: str = "UNKNOWN", token_out_symbol: str = "UNKNOWN") -> bool:
        """Check price difference between two DEXs. Returns True if opportunity found."""
        try:
            pair_key = f"{token_symbol}_liquidity"

            # Skip blacklisted pairs to avoid wasting RPC calls
            if self._is_pair_blacklisted(pair_key):
                return False

            self._total_pairs_scanned += 1

            # Use token-specific scan amount scaled to configured flash_loan_amount
            # CRITICAL: flash_loan_amount (10*10**18) only works for 18-decimal tokens
            # For USDC (6 dec), WBTC (8 dec), etc. we need decimal-appropriate amounts
            # Amounts are pre-scaled in __init__ based on flash_loan_amount setting
            amount_in = self._scan_amounts.get(token_symbol, self.flash_loan_amount)
            token_in_decimals = TOKEN_DECIMALS.get(token_symbol, 18)

            # Flash loan amount is in WETH (the borrow asset)
            borrow_amount = self.flash_loan_amount

            # Ensure addresses are checksummed for web3.py compatibility
            token_checksum = Web3.to_checksum_address(token_in)  # The token being arbitraged
            weth_checksum = Web3.to_checksum_address(token_out)  # WETH (borrow asset)

            # ============================================================
            # STEP 1: Query BUY direction - WETH → token (how much token can we buy?)
            # This matches the contract's buyRouter (asset → intermediateToken)
            # ============================================================
            buy_prices = {}
            buy_errors = {}
            for name, contract in self.router_contracts.items():
                try:
                    amounts = contract.functions.getAmountsOut(borrow_amount, [weth_checksum, token_checksum]).call()
                    buy_prices[name] = amounts[1]  # How many tokens we get for our WETH
                except Exception as e:
                    buy_errors[name] = str(e)[:50]

            # Log diagnostic info periodically (every 60 scans = ~2 minutes)
            if self._total_pairs_scanned % 60 == 1:
                if buy_prices:
                    self.logger.debug(f"📊 [{token_symbol}] Buy prices (WETH→token): {len(buy_prices)} DEXs responded")
                else:
                    self.logger.warning(f"⚠️ [{token_symbol}] No buy prices from any DEX. Errors: {buy_errors}")

            if len(buy_prices) < 2:
                # Need at least 2 DEXs for arbitrage
                self._update_liquidity_blacklist(pair_key, has_liquidity=False)
                return False

            # Liquidity found - clear from blacklist if present
            self._update_liquidity_blacklist(pair_key, has_liquidity=True)
            self._pairs_with_liquidity += 1

            # Find best BUY DEX (gives most tokens for our WETH = lowest token price)
            best_buy_dex = max(buy_prices, key=buy_prices.get)
            tokens_bought = buy_prices[best_buy_dex]

            if tokens_bought == 0:
                return False

            # ============================================================
            # STEP 2: Query SELL direction - token → WETH (how much WETH can we get back?)
            # This matches the contract's sellRouter (intermediateToken → asset)
            # ============================================================
            sell_prices = {}
            for name, contract in self.router_contracts.items():
                if name == best_buy_dex:
                    continue  # Skip same DEX - no arbitrage within same DEX
                try:
                    amounts = contract.functions.getAmountsOut(tokens_bought, [token_checksum, weth_checksum]).call()
                    sell_prices[name] = amounts[1]  # How much WETH we get back
                except Exception:
                    pass

            if not sell_prices:
                return False

            # Find best SELL DEX (gives most WETH for our tokens = highest token price)
            best_sell_dex = max(sell_prices, key=sell_prices.get)
            weth_returned = sell_prices[best_sell_dex]

            # ============================================================
            # STEP 3: Calculate profit (in WETH terms)
            # ============================================================
            flash_loan_fee = borrow_amount * 5 // 10000  # 0.05% Aave fee
            amount_owed = borrow_amount + flash_loan_fee

            # Track spread for visibility (even negative)
            if weth_returned > 0:
                raw_spread = (weth_returned - amount_owed) / borrow_amount
                spread_pair_key = f"{token_symbol} ({best_buy_dex}→{best_sell_dex})"

                if raw_spread > self._best_spread_seen:
                    self._best_spread_seen = raw_spread
                    self._best_spread_pair = spread_pair_key

            if weth_returned <= amount_owed:
                return False  # No profit possible

            profit = weth_returned - amount_owed
            raw_spread = profit / borrow_amount

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
                    self.logger.info(f"📅 New day - reset arbitrage execution counters")

                # Check daily execution limit per pair
                current_count = self._pair_execution_count.get(opp_key, 0)
                if current_count >= self._max_executions_per_pair_per_day:
                    return True  # Silently skip - already hit daily limit

                # Check cooldown - don't spam same opportunity
                if self._last_opportunity_key == opp_key and self._last_opportunity_time:
                    elapsed = (now - self._last_opportunity_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        return True  # Same opportunity within cooldown

                # New opportunity or cooldown expired - log and execute
                self._last_opportunity_key = opp_key
                self._last_opportunity_time = now

                # Update daily execution count
                self._pair_execution_count[opp_key] = current_count + 1
                remaining = self._max_executions_per_pair_per_day - (current_count + 1)

                # Use correct decimal divisor for logging
                in_divisor = 10 ** token_in_decimals
                out_decimals = TOKEN_DECIMALS.get(token_out_symbol, 18)
                out_divisor = 10 ** out_decimals

                self.logger.info(f"🚨 [{self.chain_name.upper()}] ARBITRAGE OPPORTUNITY [{token_symbol}/{token_out_symbol}]: Buy on {best_buy_dex}, Sell on {best_sell_dex}. Raw: {raw_spread:.2%}, Net: {net_spread:.2%} (#{current_count + 1} today, {remaining} remaining)")
                self.logger.info(f"   Path: {amount_in/in_divisor:.4f} {token_symbol} → {forward_output/out_divisor:.4f} {token_out_symbol} → {final_output/in_divisor:.4f} {token_symbol} (profit: {profit/in_divisor:.4f})")
                self._stats['opportunities_executed'] += 1

                # Execute arbitrage - now buy_dex and sell_dex match contract's expectations directly!
                # No router swapping needed because we scanned from WETH's perspective
                await self._execute_flash_swap(
                    buy_dex=best_buy_dex,   # Best for WETH → token (contract's buyRouter)
                    sell_dex=best_sell_dex, # Best for token → WETH (contract's sellRouter)
                    token_in=token_in,      # The token being arbitraged
                    token_out=token_out,    # WETH (borrow asset)
                    amount=borrow_amount,
                    expected_profit=net_spread,
                    token_symbol=token_symbol
                )
                return True
            return False

        except Exception as e:
            self.logger.error(f"Arb check failed for {token_symbol}: {e}")
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
                self.logger.warning(f"⚠️ [{self.chain_name.upper()}] Insufficient gas: {self._cached_balance_eth:.6f} ETH < {self._min_gas_eth} ETH minimum")
                self.logger.warning(f"   Fund wallet {self.wallet_address} to enable execution")
                self._low_gas_warning_shown = True
            elif has_gas and self._low_gas_warning_shown:
                self.logger.info(f"✅ [{self.chain_name.upper()}] Gas balance recovered: {self._cached_balance_eth:.6f} ETH")
                self._low_gas_warning_shown = False

            return has_gas, self._cached_balance_eth

        except Exception as e:
            self.logger.debug(f"Gas check failed: {e}")
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
        token_decimals = TOKEN_DECIMALS.get(token_symbol, 18)
        token_divisor = 10 ** token_decimals
        self.logger.info(f"⚡ [{self.chain_name.upper()}] Executing Arbitrage [{token_symbol}]: {buy_dex} -> {sell_dex} | Amount: {amount/token_divisor:.4f} {token_symbol} | Expected: +{expected_profit:.2%}")

        if self.dry_run:
            # Simulate execution
            await asyncio.sleep(0.5)
            self.logger.info(f"✅ [{self.chain_name.upper()}] Flash Swap Executed (DRY RUN) [{token_symbol}]")
            await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, "DRY_RUN", token_symbol)
            return

        # Validate credentials before live execution
        if not self.private_key or not self.wallet_address:
            self.logger.error(f"❌ Cannot execute - PRIVATE_KEY or WALLET_ADDRESS not configured in database")
            return

        # Check gas balance before execution
        has_gas, balance = await self._check_gas_balance()
        if not has_gas:
            self.logger.warning(f"⏸️ Skipping execution - insufficient gas ({balance:.6f} ETH < {self._min_gas_eth} ETH)")
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
                    self.logger.warning(f"⏸️ Skipping execution - insufficient ETH for gas cost")
                    self.logger.warning(f"   Balance: {balance:.6f} ETH | Required: {required_eth:.6f} ETH")
                    self.logger.warning(f"   Gas: {gas_price/1e9:.1f} gwei × {gas_limit:,} = {estimated_cost_eth:.6f} ETH + 30% buffer")
                    return
            except Exception as e:
                self.logger.debug(f"Gas estimation failed: {e}")

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
                self.logger.info(f"✅ [{self.chain_name.upper()}] Arbitrage executed [{token_symbol}]: {tx_hash}")
                await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, expected_profit, tx_hash, token_symbol)
            else:
                self.logger.error(f"❌ [{self.chain_name.upper()}] Arbitrage execution failed [{token_symbol}]")
                # Send Telegram error alert for failed execution
                await self._send_error_alert(
                    error_type="Execution Failed",
                    details=f"Flash swap failed for {token_symbol}\nBuy: {buy_dex} → Sell: {sell_dex}\nAmount: {amount/token_divisor:.4f} {token_symbol}",
                    token_symbol=token_symbol
                )

        except Exception as e:
            self.logger.error(f"Arbitrage execution error [{token_symbol}]: {e}")
            # Send Telegram error alert for exception
            await self._send_error_alert(
                error_type="Execution Error",
                details=f"Exception during arbitrage: {str(e)[:200]}",
                token_symbol=token_symbol
            )

    # Aave V3 supported flash loan assets (high liquidity pools)
    # These are the tokens Aave allows for flash loans on each chain
    AAVE_FLASHLOAN_ASSETS = {
        'ethereum': {
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
            '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',  # USDC
            '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
            '0x6B175474E89094C44Da98b954EeAdDcB80656c63',  # DAI
            '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',  # WBTC
        },
        'arbitrum': {
            '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',  # WETH
            '0xaf88d065e77c8cC2239327C5EDb3A432268e5831',  # USDC
            '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',  # USDT
            '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1',  # DAI
            '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f',  # WBTC
        },
        'base': {
            '0x4200000000000000000000000000000000000006',  # WETH
            '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',  # USDC
            '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA',  # USDbC
            '0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb',  # DAI
        },
    }

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

        IMPORTANT: The scanning function now works from WETH's perspective, so:
        - buy_dex = best for WETH → token (matches contract's buyRouter)
        - sell_dex = best for token → WETH (matches contract's sellRouter)

        The contract's executeArbitrage() function will:
        1. Call Aave's flashLoanSimple to borrow WETH
        2. Swap borrowed WETH → TOKEN on buyRouter (buy cheap)
        3. Swap TOKEN → WETH on sellRouter (sell expensive)
        4. Repay loan + fee to Aave
        5. Keep profit in contract (withdraw later)

        Args:
            buy_dex: DEX best for WETH → token (contract's buyRouter)
            sell_dex: DEX best for token → WETH (contract's sellRouter)
            token_in: The token being arbitraged (e.g., DEGEN, GRT)
            token_out: The borrow asset (WETH)
            amount: Amount to borrow in wei
        """
        # Safety check: flash loan executor must be initialized with a contract address
        if not self.flash_loan_executor:
            flash_loan_env_key = self.chain_config.get('flash_loan_env_key', 'FLASH_LOAN_RECEIVER_CONTRACT')
            self.logger.error(f"❌ Flash loan executor not initialized - no receiver contract for {self.chain_name.upper()}")
            self.logger.error(f"   Set {flash_loan_env_key} in .env and restart")
            return None

        # Get Aave-supported assets for this chain
        supported_assets = self.AAVE_FLASHLOAN_ASSETS.get(self.chain_name.lower(), set())

        # token_in = the token being arbitraged (e.g., DEGEN)
        # token_out = WETH (the borrow asset)
        token_checksum = Web3.to_checksum_address(token_in)
        weth_checksum = Web3.to_checksum_address(token_out)

        # Get router addresses
        buy_router_addr = self.routers.get(buy_dex, list(self.routers.values())[0])
        sell_router_addr = self.routers.get(sell_dex, list(self.routers.values())[-1])

        # WETH addresses on supported chains
        weth_addresses = {
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # Ethereum
            '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',  # Arbitrum
            '0x4200000000000000000000000000000000000006',  # Base
        }

        # Verify token_out is WETH (our borrow asset)
        if weth_checksum not in weth_addresses:
            self.logger.error(f"❌ Cannot flash loan - token_out is not WETH")
            self.logger.error(f"   token_out: {token_out[:10]}...")
            self.logger.error(f"   Only WETH-based pairs are supported for flash loans")
            return None

        # Set up flash loan parameters
        # Borrow WETH, use token as intermediate
        borrow_asset = weth_checksum
        intermediate_token = token_checksum

        # Router mapping is now direct - no swapping needed!
        # buy_dex was found to be best for WETH → token = contract's buyRouter
        # sell_dex was found to be best for token → WETH = contract's sellRouter
        contract_buy_router = buy_router_addr
        contract_sell_router = sell_router_addr

        self.logger.info(f"   Flash loan: Borrowing {amount/1e18:.4f} WETH")
        self.logger.info(f"   Flash loan params: borrow=WETH, intermediate={token_in[:10]}...")
        self.logger.info(f"   Route: {buy_dex} ({buy_router_addr[:10]}...) → {sell_dex} ({sell_router_addr[:10]}...)")

        # Execute via contract's executeArbitrage function
        # This ensures initiator == contract address (passes the check!)
        tx_hash = await self.flash_loan_executor.execute_arbitrage(
            asset=borrow_asset,
            amount=amount,
            buy_router=contract_buy_router,
            sell_router=contract_sell_router,
            intermediate_token=intermediate_token
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
                self.logger.error("Router contracts not available")
                return None

            deadline = int(datetime.now().timestamp()) + 120

            # Get EIP-1559 gas pricing for better reliability
            latest_block = self.w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas', self.w3.eth.gas_price)
            priority_fee = 2 * 10**9  # 2 gwei
            max_fee = int(base_fee * 1.5) + priority_fee

            current_nonce = self.w3.eth.get_transaction_count(self.wallet_address)

            # Build buy transaction with EIP-1559
            buy_tx = buy_router.functions.swapExactTokensForTokens(
                amount,
                0,  # Min output
                [token_in, token_out],
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 300000,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
                'nonce': current_nonce,
                'chainId': self.w3.eth.chain_id
            })

            # Sign buy transaction
            signed_buy = self.w3.eth.account.sign_transaction(buy_tx, self.private_key)

            # Build sell transaction (nonce + 1) with EIP-1559
            sell_tx = sell_router.functions.swapExactTokensForTokens(
                amount,  # Simplified - should use output from buy
                0,
                [token_out, token_in],
                Web3.to_checksum_address(self.wallet_address),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.wallet_address),
                'gas': 300000,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
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
                        bundle_hash = bundle_result.get('bundleHash', '')
                        if not bundle_hash:
                            self.logger.warning("Bundle sent but no bundleHash - treating as unconfirmed")
                        else:
                            self.logger.info(f"Bundle sent: {bundle_hash}")

                            # Verify bundle inclusion by checking first tx receipt
                            buy_tx_hash = self.w3.keccak(hexstr=signed_buy.rawTransaction.hex())
                            for wait_attempt in range(5):
                                await asyncio.sleep(3)
                                try:
                                    current = self.w3.eth.block_number
                                    if current >= target_block:
                                        try:
                                            receipt = self.w3.eth.get_transaction_receipt(buy_tx_hash)
                                            if receipt and receipt.status == 1:
                                                self.logger.info(f"✅ Bundle confirmed in block {receipt.blockNumber}")
                                                return receipt.transactionHash.hex()
                                            elif receipt and receipt.status == 0:
                                                self.logger.warning("Bundle tx reverted")
                                                return None
                                        except Exception:
                                            pass
                                        if current >= target_block + 2:
                                            break
                                except Exception:
                                    pass

                            self.logger.warning(f"Bundle {bundle_hash[:16]}... not confirmed - not included by builders")
                    else:
                        self.logger.warning("Flashbots bundle rejected, falling back to public mempool")

            # Fallback: Send to public mempool
            tx_hash = self.w3.eth.send_raw_transaction(signed_buy.rawTransaction)
            return tx_hash.hex()

        except Exception as e:
            self.logger.error(f"Direct swap execution error: {e}")
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
                self.logger.warning(f"Cannot log trade - ETH price unavailable")
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

            self.logger.info(
                f"💰 [{self.chain_name.upper()}] Arb value [{token_symbol}]: {amount_eth:.4f} ETH @ ${eth_price:.2f} = ${entry_usd:.2f} | "
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
            self.logger.debug(f"💾 Logged to arbitrage_trades: {trade_id} [{token_symbol}]")

            # Send Telegram alert for successful trade
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from .arbitrage_alerts import ArbitrageTradeAlert, ArbitrageChain

                    # Map chain name to ArbitrageChain enum
                    chain_map = {
                        'ethereum': ArbitrageChain.ETHEREUM,
                        'arbitrum': ArbitrageChain.ARBITRUM,
                        'base': ArbitrageChain.BASE,
                    }
                    chain = chain_map.get(self.chain_name, ArbitrageChain.ETHEREUM)

                    alert = ArbitrageTradeAlert(
                        chain=chain,
                        token_symbol=token_symbol,
                        buy_dex=buy_dex,
                        sell_dex=sell_dex,
                        amount=amount_eth,
                        amount_usd=entry_usd,
                        profit_pct=net_profit_pct * 100,
                        profit_amount=net_profit_usd / eth_price if eth_price > 0 else 0,
                        profit_usd=net_profit_usd,
                        tx_hash=tx_hash,
                        is_simulated=self.dry_run,
                        gas_cost_usd=GAS_COST_USD,
                        flash_loan_fee=flash_loan_cost,
                    )
                    await self.telegram_alerts.send_trade_alert(alert)
                except Exception as tg_err:
                    self.logger.debug(f"Telegram alert failed: {tg_err}")

        except Exception as e:
            self.logger.error(f"Error logging arb trade: {e}")

    async def _send_error_alert(
        self,
        error_type: str,
        details: str,
        token_symbol: Optional[str] = None,
        tx_hash: Optional[str] = None
    ):
        """Send Telegram error alert"""
        if not self.telegram_alerts or not self.telegram_alerts.enabled:
            return

        try:
            from .arbitrage_alerts import ArbitrageErrorAlert, ArbitrageChain

            # Map chain name to ArbitrageChain enum
            chain_map = {
                'ethereum': ArbitrageChain.ETHEREUM,
                'arbitrum': ArbitrageChain.ARBITRUM,
                'base': ArbitrageChain.BASE,
            }
            chain = chain_map.get(self.chain_name, ArbitrageChain.ETHEREUM)

            alert = ArbitrageErrorAlert(
                chain=chain,
                error_type=error_type,
                details=details,
                token_symbol=token_symbol,
                tx_hash=tx_hash
            )
            await self.telegram_alerts.send_error_alert(alert)
        except Exception as e:
            self.logger.debug(f"Failed to send error alert: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False

        # Close Flashbots executor
        if self.flashbots_executor:
            await self.flashbots_executor.close()

        self.logger.info(f"🛑 {self.LOGGER_NAME} Stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN-SPECIFIC ENGINE SUBCLASSES
# Each has its own logger for dedicated log output
# ═══════════════════════════════════════════════════════════════════════════════

class ETHArbitrageEngine(EVMArbitrageEngine):
    """Ethereum Mainnet Arbitrage Engine"""
    CHAIN_NAME = 'ethereum'
    EXPECTED_CHAIN_ID = 1
    RPC_PROVIDER_KEY = 'ETHEREUM_RPC'
    RPC_ENV_KEY = 'ETHEREUM_RPC_URL'
    RPC_ENV_FALLBACK = 'WEB3_PROVIDER_URL'
    LOGGER_NAME = 'ETHArbitrageEngine'


class ARBArbitrageEngine(EVMArbitrageEngine):
    """Arbitrum One Arbitrage Engine"""
    CHAIN_NAME = 'arbitrum'
    EXPECTED_CHAIN_ID = 42161
    RPC_PROVIDER_KEY = 'ARBITRUM_RPC'
    RPC_ENV_KEY = 'ARBITRUM_RPC_URL'
    RPC_ENV_FALLBACK = None
    LOGGER_NAME = 'ARBArbitrageEngine'


class BaseArbitrageEngine(EVMArbitrageEngine):
    """Base L2 Arbitrage Engine"""
    CHAIN_NAME = 'base'
    EXPECTED_CHAIN_ID = 8453
    RPC_PROVIDER_KEY = 'BASE_RPC'
    RPC_ENV_KEY = 'BASE_RPC_URL'
    RPC_ENV_FALLBACK = None
    LOGGER_NAME = 'BaseArbitrageEngine'


# Backwards compatibility alias
ArbitrageEngine = ETHArbitrageEngine
