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
import random
import aiohttp
from web3 import Web3
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
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
# WAVE-17: POOL-RESERVE FLOOR — UniV2 Factory + Pair ABIs
# getReserves() on the pair contract returns (reserve0, reserve1, blockTimestamp).
# reserve0/reserve1 are used to compute TVL in USD and skip pools below the
# configurable min_pool_tvl_usd floor (default $50k, DB-configurable via
# migration 053). A near-empty pool must never enter the spread calc.
# ═══════════════════════════════════════════════════════════════════════════════
UNIV2_FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"}
        ],
        "name": "getPair",
        "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]

UNIV2_PAIR_ABI = [
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "blockTimestampLast", "type": "uint32"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# UniV2-compatible factory addresses per (dex_name, chain_id).
# Key: dex_name matches ROUTERS_* keys; value: factory address (per chain_id).
UNIV2_FACTORIES: dict = {
    'uniswap_v2': {
        1: '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
    },
    'sushiswap': {
        1:     '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',
        42161: '0xc35DADB65012eC5796536bD9864eD8773aBc74C4',
        8453:  '0x71524B4f93c58fcbF659783284E38825f0622859',
    },
    'baseswap': {
        8453:  '0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB',
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE-17: UNISWAP V3 QUOTER (Ethereum mainnet proof-of-concept)
#
# QuoterV2 exposes quoteExactInputSingle as a non-reverting static call.
# Unlike the V3 SwapRouter it is safe to call view-only (no state change).
#
# Deployment addresses:
#   Ethereum:  0x61fFE014bA17989E743c5F6cB21bF9697530B21e  (QuoterV2)
#   Arbitrum:  0x61fFE014bA17989E743c5F6cB21bF9697530B21e  (same bytecode)
#   Base:      0x3d4e44Eb1374240CE5F1B136041212501e4a7901  (QuoterV2)
#
# Fee tiers probed: 500 (0.05%), 3000 (0.3%), 10000 (1%).
# The best tier is selected per quote.  Key in buy_prices: 'uniswap_v3'.
#
# Gated by v3_quoter_enabled (CHAIN_CONFIGS + DB override, default False for
# Arbitrum/Base until continuity plan executes; True for Ethereum).
# ═══════════════════════════════════════════════════════════════════════════════
UNIV3_QUOTER_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"}
                ],
                "internalType": "struct IQuoterV2.QuoteExactInputSingleParams",
                "name": "params",
                "type": "tuple"
            }
        ],
        "name": "quoteExactInputSingle",
        "outputs": [
            {"internalType": "uint256", "name": "amountOut", "type": "uint256"},
            {"internalType": "uint160", "name": "sqrtPriceX96After", "type": "uint160"},
            {"internalType": "uint32", "name": "initializedTicksCrossed", "type": "uint32"},
            {"internalType": "uint256", "name": "gasEstimate", "type": "uint256"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# QuoterV2 deployed addresses per chain_id.
UNIV3_QUOTER_ADDRESSES: dict = {
    1:     '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',  # Ethereum
    42161: '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',  # Arbitrum
    8453:  '0x3d4e44Eb1374240CE5F1B136041212501e4a7901',  # Base
}

# Fee tiers to probe per V3 quote.  The best amountOut across tiers is used.
UNIV3_FEE_TIERS = [500, 3000, 10000]  # 0.05%, 0.3%, 1%

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
    'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
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
    # WAVE-17: Only V2-compatible routers with verified TVL on Arbitrum.
    # REMOVED camelot — concentrated-liquidity V3 AMM; getAmountsOut uses
    #   constant-product math and returns wrong/partial results for CL pools,
    #   producing -2588 bps artifacts (root cause B, wave-17 diagnosis).
    # REMOVED zyberswap — exploited April 2023, TVL near zero; getAmountsOut
    #   on a ~$1k reserve pool returns garbage, producing -7033 bps artifacts
    #   (root cause A, wave-17 diagnosis).
    'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',  # SushiSwap V2 — live, real TVL
}

ROUTERS_BASE = {
    # WAVE-17: Only V2-compatible routers with verified TVL on Base.
    # REMOVED swapbased — thin liquidity, produced -25% spread artifacts.
    # Aerodrome "Basic" pools use constant-product V2 AMM; Universal Router does
    # NOT expose getAmountsOut. Aerodrome IRouter candidate noted below; add
    # after on-chain reserve verification of WETH/USDC Basic pool.
    'sushiswap': '0x6BDED42c6DA8FBf0d2bA55B2fa120C5e0c8D7891',   # SushiSwap V2 — live
    'baseswap': '0x327Df1E6de05895d2ab08513aaDD9313Fe505d86',    # BaseSwap V2 — live
    # CANDIDATE: 'aerodrome': '0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43'
    # Aerodrome V1 IRouter — same getAmountsOut signature (stable/volatile flag).
    # Verify Basic WETH/USDC pool reserves > $50k USD before enabling.
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
# WAVE-13 FIX: All pairs MUST have WETH as token_out (the borrow/repay asset).
# Spatial-arb model: borrow WETH -> buy token_in on cheapest DEX ->
# sell token_in for WETH on most expensive DEX -> repay WETH + Aave fee.
# Any pair where token_out != 'WETH' passed borrow_amount (ETH wei, ~10^18)
# as getAmountsOut amountIn for a 6-decimal asset (USDC/USDT), equivalent to
# querying 10^12 tokens worth, which reverts or returns 0 -> weth_returned ~= 0
# -> spread = (0 - borrow_amount) / borrow_amount = -1.0 = -10000 bps.
# This was the root cause of all -10005 / -10003 bps near-miss log entries.
# ('WETH','X') pairs are wrong direction: X would be borrow asset, not WETH.
# Fix: invert ('WETH','X') -> ('X','WETH'). Drop stablecoin-only pairs
# (USDC/USDT etc.) which require a USDC flash loan, not WETH.
ARB_PAIRS_ETHEREUM = [
    # Format: (intermediate_token, 'WETH') -- borrow WETH, arb intermediate
    ('USDC', 'WETH'),   # was ('WETH', 'USDC') - inverted to correct direction
    ('USDT', 'WETH'),   # was ('WETH', 'USDT') - inverted
    ('DAI', 'WETH'),    # was ('WETH', 'DAI') - inverted
    ('WBTC', 'WETH'),

    # Major DeFi tokens with good V2 liquidity
    ('LINK', 'WETH'),
    ('UNI', 'WETH'),
    ('AAVE', 'WETH'),
    ('SUSHI', 'WETH'),
    ('CRV', 'WETH'),

    # NOTE: Stablecoin-only pairs require USDC flash loan, not WETH -- removed:
    # ('USDC', 'USDT'), ('USDC', 'DAI'), ('DAI', 'USDT')
    # NOTE: Low-liquidity pairs excluded:
    # ('MATIC','WETH'), ('SNX','WETH'), ('COMP','WETH'), ('MKR','WETH'),
    # ('BAL','WETH'), ('YFI','WETH'), ('GRT','WETH'), ('ENS','WETH'),
    # ('SHIB','WETH'), ('PEPE','WETH'), ('LDO','WETH')
]

ARB_PAIRS_ARBITRUM = [
    # Format: (intermediate_token, 'WETH') -- borrow WETH, arb intermediate
    ('USDC', 'WETH'),   # was ('WETH', 'USDC') - inverted
    ('USDT', 'WETH'),   # was ('WETH', 'USDT') - inverted
    ('DAI', 'WETH'),    # was ('WETH', 'DAI') - inverted
    ('WBTC', 'WETH'),
    ('ARB', 'WETH'),    # Native ARB token - high volume

    # Native Arbitrum DeFi tokens with verified V2 liquidity
    ('GMX', 'WETH'),    # GMX - major Arbitrum protocol

    # Bridged DeFi tokens with liquidity
    ('LINK', 'WETH'),

    # NOTE: ('ARB','USDC') removed - USDC is not WETH, causes -100% spread bug
    # NOTE: ('USDC','USDT'), ('USDC','DAI') removed - stablecoin pairs need USDC flash loan
    # NOTE: ('MAGIC','WETH'), ('RDNT','WETH'), ('PENDLE','WETH') - low V2 liquidity
    # NOTE: ('UNI','WETH'), ('AAVE','WETH'), ('CRV','WETH'), ('SUSHI','WETH') - low liquidity
]

ARB_PAIRS_BASE = [
    # Format: (intermediate_token, 'WETH') -- borrow WETH, arb intermediate
    ('USDC', 'WETH'),    # was ('WETH', 'USDC') - inverted
    ('USDbC', 'WETH'),   # was ('WETH', 'USDbC') - inverted
    ('DAI', 'WETH'),     # was ('WETH', 'DAI') - inverted
    ('cbETH', 'WETH'),   # Coinbase ETH - good liquidity

    # NOTE: rETH removed - no liquidity on Base V2 DEXs

    # Native Base tokens with verified liquidity
    ('AERO', 'WETH'),    # Aerodrome - major Base DEX token

    # Meme tokens with liquidity
    ('DEGEN', 'WETH'),

    # NOTE: ('AERO','USDC') removed - USDC is not WETH, causes -100% spread bug
    # NOTE: ('BRETT','WETH'), ('TOSHI','WETH') - check liquidity before enabling
    # NOTE: ('USDC','USDbC'), ('USDC','DAI') removed - stablecoin pairs need USDC flash loan
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
        # A2-02/A2-03/A2-05: per-chain cost profile used by pre-execution gate
        # and persisted PnL accounting. Falls back to these constants only when
        # the live gas oracle is unreachable.
        'is_l2': False,
        'flash_loan_gas_limit': 450_000,
        'fallback_gas_gwei': 30,           # Reasonable Ethereum baseline
        'default_slippage_pct': 0.004,     # ~0.4% (0.2% per swap leg x 2)
        'flash_loan_fee_pct': 0.0005,      # Aave V3 fee
        # WAVE-16: minimum cross-DEX price divergence gate (bps). Skips the
        # sell-leg RPC calls and avoids logging confusing -70xx bps near-misses
        # that are actually price-impact artefacts, not spread calculation errors.
        'min_price_spread_bps': 10.0,
        # WAVE-17: upper sanity cap (bps). BUY-side divergence above this is a
        # thin-pool or wrong-ABI artifact, not a real spread. DB-configurable.
        'upper_spread_cap_bps': 500.0,
        # WAVE-17: pool-reserve floor. Skip a DEX for a pair when its on-chain
        # WETH leg TVL (2 * weth_reserve * eth_price) < this threshold.
        # Prevents garbage getAmountsOut results from near-empty pools.
        'min_pool_tvl_usd': 50_000.0,
        # WAVE-17: Uniswap V3 Quoter enabled (QuoterV2 proof-of-concept).
        # Ethereum mainnet: enabled by default — V3 is the dominant ETH liquidity
        # venue ($10B+ TVL across WETH/USDC, WETH/USDT, WETH/WBTC tiers).
        # DB-configurable via arb_v3_quoter_enabled_ethereum.
        'v3_quoter_enabled': True,
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
        'is_l2': True,
        'flash_loan_gas_limit': 1_200_000,  # ARB nitro charges effective gas higher
        'fallback_gas_gwei': 0.1,
        'default_slippage_pct': 0.005,      # L2 V2 pools shallower -> ~0.5%
        'flash_loan_fee_pct': 0.0005,
        'min_price_spread_bps': 10.0,
        'upper_spread_cap_bps': 500.0,
        'min_pool_tvl_usd': 50_000.0,
        # V3 Quoter: disabled on Arbitrum until V3->V2 execution path is
        # implemented (V3 price discovery with V2 flash-loan execution is a
        # mismatch; continuity plan: wave-18 adds V3 SwapRouter execution leg).
        'v3_quoter_enabled': False,
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
        'is_l2': True,
        'flash_loan_gas_limit': 800_000,
        'fallback_gas_gwei': 0.05,
        'default_slippage_pct': 0.005,
        'flash_loan_fee_pct': 0.0005,
        'min_price_spread_bps': 10.0,
        'upper_spread_cap_bps': 500.0,
        'min_pool_tvl_usd': 50_000.0,
        # V3 Quoter: disabled on Base — same wave-18 continuity plan applies.
        'v3_quoter_enabled': False,
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
        # WAVE-17: V3 QuoterV2 contract instance (None until initialize(); only set when
        # v3_quoter_enabled is True for the chain). Used in _quote_v3().
        self.v3_quoter_contract = None

        # Flash loan and Flashbots executors
        self.flash_loan_executor: Optional[FlashLoanExecutor] = None
        self.flashbots_executor: Optional[FlashbotsExecutor] = None

        # Settings
        # IMPORTANT: Real DEX arbitrage opportunities are typically 0.1-0.5%
        # After costs (~0.55%): flash loan 0.05% + slippage ~0.5% = need ~0.6% raw spread
        # A2-06: honor the dashboard "Base Profit Threshold (%)" knob
        # (settings_arbitrage.html -> min_profit_spread). Was previously a
        # hard constant; UI changes were silently ignored.
        try:
            cfg_min = config.get('min_profit_spread')
            if cfg_min is None:
                cfg_min = config.get('min_profit_threshold')
            if cfg_min is None:
                self.min_profit_threshold = 0.003
            else:
                # Accept either fraction (0.003) or percent (0.3 / 0.5) input
                cfg_val = float(cfg_min)
                self.min_profit_threshold = cfg_val / 100.0 if cfg_val >= 0.05 else cfg_val
        except Exception:
            self.min_profit_threshold = 0.003
        # Persist baseline for the adaptive curve (enhancement #4).
        self._min_profit_threshold_base = self.min_profit_threshold
        self.use_flash_loans = True
        self.use_flashbots = True

        # Flash loan amount: read from config (in ETH units), convert to wei.
        # When not explicitly configured in the DB, L2 chains use a smaller
        # default to avoid price impact that eats thin-but-real spreads.
        # Arbitrum/Base V2 pools typically carry $50-200k TVL; routing 10 ETH
        # (~$25k) through them produces ~0.5-1% one-way price impact, which
        # makes all raw spreads negative before the fee+slippage cost model even
        # runs.  L2 defaults: Arbitrum 1 ETH, Base 0.5 ETH.  The operator can
        # override either value via the DB `flash_loan_amount` config key.
        # ARB-W14-3: right-size L2 flash-loan scan amount.
        is_l2 = bool(self.chain_config.get('is_l2', False))
        if 'flash_loan_amount' in config:
            flash_loan_eth = config['flash_loan_amount']
        elif is_l2:
            if self.EXPECTED_CHAIN_ID == 8453:  # Base
                flash_loan_eth = 0.5
            else:  # Arbitrum and other L2s
                flash_loan_eth = 1.0
        else:
            flash_loan_eth = 10  # Ethereum mainnet: keep 10 ETH for USD gas frac visibility
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

        # A2-02/A2-03/A2-05: per-chain cost profile + live-gas cache.
        # Live gas is sampled at most once per second to keep RPC quota down.
        self._gas_cost_usd_cache: Optional[float] = None
        self._gas_cost_usd_cache_at: Optional[datetime] = None
        self._gas_cost_cache_ttl_s: int = 1
        # Rolling per-hour gas spend tracker (A2-07 / enhancement #3).
        self._gas_spend_usd_hour: float = 0.0
        self._gas_spend_window_start: datetime = datetime.now()
        # Default hourly budget can be overridden from DB config.
        self._gas_budget_usd_per_hour: float = float(config.get('gas_budget_usd_per_hour', 50.0))
        # Adaptive min-profit curve (enhancement #4): track recent gas spikes
        # vs baseline; raises the min-profit-bps threshold when gas is volatile.
        self._gas_spike_samples: List[Tuple[datetime, float]] = []  # (ts, gwei)
        self._gas_spike_window_s: int = 600  # 10-minute look-back

        # Dynamic liquidity blacklist - pairs that consistently fail
        # Format: {pair_key: (fail_count, last_fail_time)}
        self._liquidity_blacklist: Dict[str, Tuple[int, datetime]] = {}
        self._blacklist_threshold = 5  # Blacklist after 5 consecutive failures
        self._blacklist_duration = 3600  # Unblacklist after 1 hour (liquidity may return)

        # Wave-F5 RC-A1: RPC-infra failure state. HTTP 401/403 auth errors
        # used to be swallowed into buy_errors and blacklisted as "no
        # liquidity" (a dead Ankr key left ETH auth-blind for 11 straight
        # days while the dashboard said "Scanning"). The endpoint was also
        # pinned at startup; re-resolve from pool_engine every
        # arb_rpc_refresh_minutes (DB key, mig 142, default 15).
        try:
            self._rpc_refresh_minutes = float(config.get('arb_rpc_refresh_minutes', 15.0))
        except (TypeError, ValueError):
            self._rpc_refresh_minutes = 15.0
        self._rpc_last_refresh: datetime = datetime.now()
        self._rpc_infra_fail_count: int = 0
        self._rpc_infra_last_fail_at: Optional[datetime] = None
        self._rpc_infra_last_warn_at: Optional[datetime] = None
        self._rpc_infra_last_rotate_at: Optional[datetime] = None

        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }
        # Wave-5: rolling buffer of opportunities that were REJECTED so the
        # operator can see WHY no trades fired. Surfaced via
        # /api/arbitrage/diagnostics + "Why no trades?" dashboard panel.
        # Keep last 50 in-process; the API + persist snapshot ship the latest 20.
        self._near_misses: "deque[Dict]" = deque(maxlen=50)
        self._near_miss_counters: Dict[str, int] = {}

        # W6 commit 4/5 — subprocess health surface. _persist_runtime_stats
        # was only called every 5 minutes from _log_stats_if_needed, so any
        # crash before the first tick (or any loop wedged inside
        # _check_arb_opportunity) left the runtime_stats row unchanged for
        # hours, making /api/arbitrage/diagnostics report stale: true with
        # no way to tell "dead" from "just gated". These fields are stamped
        # by the run loop on every tick (in-memory only, no DB hit), and
        # persisted by the existing _persist_runtime_stats path so the
        # dashboard can render "engine alive 12s ago" or "engine crashed
        # 4h ago: <error>" without any new IPC.
        self._last_tick_at: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._last_error_at: Optional[datetime] = None

        # Wave-3 realized-slippage learning state. MUST be initialized here so
        # every chain subclass (ETH/ARB/Base) inherits sane defaults: these are
        # READ in get_stats, get_realized_slippage, the _refresh guard, and the
        # execute path BEFORE _refresh_realized_slippage first assigns them.
        # Missing init => AttributeError every tick (engine dead since Feb 7).
        # min_samples=5 matches the documented Wave-3 contract (ARB_WAVE3.md);
        # ttl=3600s matches the "hourly refresh" docstring on _refresh.
        self._realized_slip_cache: Dict[str, Tuple[float, float, int]] = {}
        self._realized_slip_refreshed_at: Optional[datetime] = None
        self._realized_slip_ttl_s: int = 3600
        self._realized_slip_min_samples: int = 5

        # ═════════════════════════════════════════════════════════════════
        # ECON (revival): shadow-mode + per-chain economics gates.
        # shadow_mode (default ON): record every detected opportunity with a
        #   simulated outcome to arbitrage_trades(is_simulated=true); never
        #   enter the execute path.
        # live_execution_enabled (default OFF): even with shadow_mode off and
        #   dry_run false, broadcast requires this explicit opt-in.
        # min_net_spread_bps_<chain>: operator floor on NET spread (migration 097).
        # ═════════════════════════════════════════════════════════════════
        self.shadow_mode = self._cfg_bool(config.get('shadow_mode'), True)
        self.live_execution_enabled = self._cfg_bool(
            config.get('live_execution_enabled'), False
        )
        try:
            _mns = config.get(f'min_net_spread_bps_{self.chain_name}')
            self._min_net_spread_frac = (
                float(_mns) / 10_000.0 if _mns is not None else 0.0
            )
        except (TypeError, ValueError):
            self._min_net_spread_frac = 0.0
        # Pair-level economic dormancy: when the rolling-24h p90 of observed
        # cross-DEX price divergence cannot clear full breakeven
        # (gas + flash fee + slippage buffer), stop scanning the pair for
        # pair_dormancy_minutes. Saves RPC quota and keeps the data honest.
        self._pair_spread_history: Dict[str, deque] = {}
        self._pair_dormant_until: Dict[str, datetime] = {}
        try:
            self._pair_dormancy_minutes = float(config.get('pair_dormancy_minutes', 240))
        except (TypeError, ValueError):
            self._pair_dormancy_minutes = 240.0
        try:
            self._dormancy_min_samples = int(config.get('dormancy_min_samples', 30))
        except (TypeError, ValueError):
            self._dormancy_min_samples = 30
        # Shadow-record throttle: at most one row per pair per this many seconds.
        try:
            self._shadow_record_interval_s = float(config.get('shadow_record_interval_s', 60))
        except (TypeError, ValueError):
            self._shadow_record_interval_s = 60.0
        self._shadow_last_recorded: Dict[str, datetime] = {}
        # LIVE receipt confirmation (mig 108): bounded wait for the tx receipt
        # before booking a broadcast trade as filled. 0 disables (book at
        # broadcast — the old fire-and-forget behavior).
        try:
            self._receipt_confirm_timeout_s = float(
                config.get('receipt_confirm_timeout_s', 90)
            )
        except (TypeError, ValueError):
            self._receipt_confirm_timeout_s = 90.0

        # Telegram alerts - initialized in initialize() method
        self.telegram_alerts = None

    @staticmethod
    def _cfg_bool(raw, default: bool) -> bool:
        """Parse a DB/env config value into bool with explicit default."""
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ('true', '1', 'yes', 'on')

        # P2#5: injected by orchestrator; consulted by P1-06 follow-up (validate_trade calls)
        self.risk_manager = None

    def set_risk_manager(self, risk_manager) -> None:
        """Inject a core.risk_manager.RiskManager. P1-06 will add validate_trade calls."""
        self.risk_manager = risk_manager

    # -----------------------------------------------------------------
    # A2-02 / A2-03 / A2-05 / A2-07: cost helpers + gas-budget tracker.
    # All consumers read these (no more 0.005 / $15 / 0.006 magic numbers).
    # -----------------------------------------------------------------
    def _current_gas_gwei(self) -> float:
        """Live gas price in gwei with chain-specific fallback. Best-effort."""
        try:
            if self.w3 is not None:
                return float(self.w3.eth.gas_price) / 1e9
        except Exception:
            pass
        return float(self.chain_config.get('fallback_gas_gwei', 30))

    def _record_gas_sample(self, gwei: float) -> None:
        """Track recent gas prices for the adaptive min-profit curve."""
        now = datetime.now()
        self._gas_spike_samples.append((now, gwei))
        cutoff = now.timestamp() - self._gas_spike_window_s
        self._gas_spike_samples = [s for s in self._gas_spike_samples if s[0].timestamp() >= cutoff]

    def _gas_spike_multiplier(self) -> float:
        """
        Returns 1.0 in calm gas, up to 2.5 during sustained spikes.
        Spike = current gwei > 1.5 * 10-min median.
        """
        if len(self._gas_spike_samples) < 5:
            return 1.0
        samples = sorted(s[1] for s in self._gas_spike_samples)
        median = samples[len(samples) // 2]
        if median <= 0:
            return 1.0
        latest = self._gas_spike_samples[-1][1]
        ratio = latest / median
        if ratio <= 1.2:
            return 1.0
        if ratio >= 3.0:
            return 2.5
        # Linear ramp 1.2 -> 3.0  ==>  1.0 -> 2.5
        return 1.0 + (ratio - 1.2) * (1.5 / 1.8)

    async def _gas_cost_usd_per_tx(self, eth_price_usd: Optional[float] = None) -> float:
        """
        Estimate full flash-loan-arb tx gas cost in USD using the chain
        profile and (optionally) a live ETH/USD price. Cached for 1s.
        Falls back to chain `fallback_gas_gwei` when w3 is unavailable.
        """
        now = datetime.now()
        if (
            self._gas_cost_usd_cache is not None
            and self._gas_cost_usd_cache_at is not None
            and (now - self._gas_cost_usd_cache_at).total_seconds() < self._gas_cost_cache_ttl_s
        ):
            return self._gas_cost_usd_cache

        gas_limit = int(self.chain_config.get('flash_loan_gas_limit', 450_000))
        gwei = self._current_gas_gwei()
        self._record_gas_sample(gwei)
        gas_cost_eth = (gwei * 1e9 * gas_limit) / 1e18

        if eth_price_usd is None:
            try:
                eth_price_usd = await self.price_fetcher.get_price('eth')
            except Exception:
                eth_price_usd = None
        # Conservative ETH price fallback if oracle unreachable; we do NOT
        # use a hardcoded $-price for PnL accounting (that path returns
        # early when oracle is down), but for the gas gate $2000 keeps us
        # erring on the side of skipping marginal trades.
        if not eth_price_usd or eth_price_usd <= 0:
            eth_price_usd = 2000.0

        usd = gas_cost_eth * float(eth_price_usd)
        self._gas_cost_usd_cache = usd
        self._gas_cost_usd_cache_at = now
        return usd

    def _gas_budget_check_and_charge(self, usd_cost: float) -> Tuple[bool, str]:
        """
        Hourly gas-budget gate. Returns (allowed, reason). Rolling window.
        """
        now = datetime.now()
        if (now - self._gas_spend_window_start).total_seconds() >= 3600:
            self._gas_spend_window_start = now
            self._gas_spend_usd_hour = 0.0
        projected = self._gas_spend_usd_hour + usd_cost
        if projected > self._gas_budget_usd_per_hour:
            return False, (
                f"hourly gas budget ${self._gas_budget_usd_per_hour:.2f} would be "
                f"exceeded (${self._gas_spend_usd_hour:.2f} spent + ${usd_cost:.2f} new)"
            )
        self._gas_spend_usd_hour = projected
        return True, "ok"

    def _adaptive_min_profit_threshold(self) -> float:
        """
        Adaptive min_profit_bps curve (enhancement #4). Multiplies the
        operator-configured baseline by the gas-spike multiplier so we
        raise the bar when gas is volatile. The per-chain operator floor
        (min_net_spread_bps_<chain>, migration 097) is applied on top.
        """
        adaptive = self._min_profit_threshold_base * self._gas_spike_multiplier()
        return max(adaptive, self._min_net_spread_frac)

    def _apply_db_chain_overrides(self) -> None:
        """Wire the per-chain DB knobs (migrations 053 + 090) into chain_config.
        Keys are `<key>_<chain_name>` under arbitrage_config; the manager loads
        them into self.config. Before this, migration 053's seeds were dormant —
        the hardcoded CHAIN_CONFIGS values were always effective."""
        for key in ('min_price_spread_bps', 'upper_spread_cap_bps', 'min_pool_tvl_usd'):
            raw = self.config.get(f'{key}_{self.chain_name}')
            if raw is None:
                continue
            try:
                self.chain_config[key] = float(raw)
            except (TypeError, ValueError):
                continue
        v3 = self.config.get(f'arb_v3_quoter_enabled_{self.chain_name}')
        if v3 is not None:
            self.chain_config['v3_quoter_enabled'] = self._cfg_bool(
                v3, bool(self.chain_config.get('v3_quoter_enabled', False))
            )

    async def _breakeven_frac(self) -> float:
        """Full round-trip breakeven as a fraction of the borrow notional.

        DEX LP fees + price impact at scan size are already embedded in the
        getAmountsOut round-trip quotes, so the residual breakeven is:
        gas (live, USD/notional) + Aave flash fee + execution-slippage buffer
        (realized p90 when learned, chain static default otherwise).
        """
        eth_price = None
        try:
            eth_price = await self.price_fetcher.get_price('eth')
        except Exception:
            pass
        if not eth_price or eth_price <= 0:
            eth_price = 2000.0  # conservative: overstates the gas fraction
        try:
            gas_usd = await self._gas_cost_usd_per_tx(eth_price_usd=eth_price)
        except Exception:
            gas_usd = 0.0
        notional_usd = (self.flash_loan_amount / 1e18) * eth_price
        gas_frac = gas_usd / notional_usd if notional_usd > 0 else 0.0
        fee_frac = float(self.chain_config.get('flash_loan_fee_pct', 0.0005))
        slip_frac = float(self.chain_config.get('default_slippage_pct', 0.005))
        return gas_frac + fee_frac + slip_frac

    def _record_near_miss(self, reason: str, **fields) -> None:
        """
        Wave-5 observability: log + remember any opportunity that was REJECTED
        by a gate (min-profit / gas-budget / cooldown / daily-cap / no-liquidity
        / negative-raw-spread). Operator-visible via /api/arbitrage/diagnostics.

        reason: short snake_case identifier (e.g. 'min_profit', 'gas_budget',
                'cooldown', 'daily_cap', 'raw_spread_negative', 'risk_manager').
        fields: arbitrary kwargs serialised into the buffer entry; common keys
                are pair, buy_dex, sell_dex, profit_bps, gas_usd, threshold_bps.
        Fail-soft - never raises into the trading loop.
        """
        try:
            entry: Dict = {
                'ts': datetime.now().isoformat(),
                'chain': self.chain_name,
                'reason': reason,
            }
            entry.update(fields)
            self._near_misses.append(entry)
            self._near_miss_counters[reason] = self._near_miss_counters.get(reason, 0) + 1
            # One-line structured log so a grep on the rotating file tells
            # the operator immediately why nothing is firing.
            # Wave-18: thin_pool_artifact fires hundreds of times per minute
            # on every pair (working-as-designed: absurdly wide spreads from
            # stale/low-TVL pools are correctly filtered).  Demote to DEBUG so
            # the INFO log is not flooded; the STATS summary still prints at INFO
            # every N minutes via _log_stats_if_needed.
            parts = [f"{k}={v}" for k, v in fields.items()
                     if v is not None and k in (
                         'pair', 'buy_dex', 'sell_dex', 'profit_bps',
                         'price_spread_bps', 'threshold_bps', 'gas_usd', 'detail'
                     )]
            log_msg = f"[arb-skip] reason={reason} " + " ".join(parts)
            if reason == 'thin_pool_artifact':
                self.logger.debug(log_msg)
            else:
                self.logger.info(log_msg)
        except Exception:
            pass

    def get_near_misses(self, limit: int = 20) -> List[Dict]:
        """Return the most-recent rejected opportunities (newest first)."""
        items = list(self._near_misses)[-limit:]
        items.reverse()
        return items

    async def _persist_runtime_stats(self) -> None:
        """
        Wave-3: snapshot the in-process gas-budget tracker (and a few
        adjacent counters) to arbitrage_runtime_stats so the standalone
        dashboard can render the /arbitrage/gas-spend tile without
        cross-process IPC. Single row per chain; UPSERT keyed by chain.
        Fail-soft - never blocks the trading loop.
        """
        if not self.db_pool:
            return
        try:
            snapshot = {
                'gas_spend_usd_hour': float(self._gas_spend_usd_hour),
                'gas_budget_usd_per_hour': float(self._gas_budget_usd_per_hour),
                'gas_budget_ratio': (
                    self._gas_spend_usd_hour / self._gas_budget_usd_per_hour
                    if self._gas_budget_usd_per_hour > 0 else 0.0
                ),
                'gas_window_start': self._gas_spend_window_start.isoformat(),
                'gas_window_age_s': (
                    datetime.now() - self._gas_spend_window_start
                ).total_seconds(),
                'gas_spike_multiplier': self._gas_spike_multiplier(),
                'min_profit_threshold_effective': self._adaptive_min_profit_threshold(),
                'min_profit_threshold_base': float(self._min_profit_threshold_base),
                # ECON revival surface: go/no-go telemetry for the dashboard.
                'shadow_mode': bool(self.shadow_mode),
                'live_execution_enabled': bool(self.live_execution_enabled),
                'min_net_spread_bps': round(self._min_net_spread_frac * 10_000, 2),
                'dormant_pairs': {
                    k: v.isoformat() for k, v in self._pair_dormant_until.items()
                },
                'chain_id': self.chain_id,
                'chain_name': self.chain_name,
                # Wallet identity (issue 15): surface the resolved public signer
                # address + chain so the dashboard can show the operator WHICH
                # wallet to fund per chain. Never the private key. None until
                # initialize() derives it from the PRIVATE_KEY secret.
                'wallet_address': self.wallet_address,
                'chain': self.chain_name,
                'realized_slip_keys': len(self._realized_slip_cache),
                'realized_slip_trusted_keys': sum(
                    1 for v in self._realized_slip_cache.values()
                    if v[2] >= self._realized_slip_min_samples
                ),
                'opportunities_found': int(self._stats.get('opportunities_found', 0)),
                'opportunities_executed': int(self._stats.get('opportunities_executed', 0)),
                'scans': int(self._stats.get('scans', 0)),
                # Wave-5: ship last 20 rejected opportunities + per-reason
                # counters so the dashboard "Why no trades?" panel can render
                # without an IPC channel back into the engine subprocess.
                'near_misses': self.get_near_misses(limit=20),
                'near_miss_counters': dict(self._near_miss_counters),
                # W6: subprocess health surface. Dashboard reads these to
                # render "engine alive 12s ago" or "engine crashed 4h ago".
                # last_tick_at is stamped at the top of every scan iteration,
                # last_error is set inside the run-loop except handler. Both
                # None on first persist (startup marker) before any tick.
                'last_tick_at': (
                    self._last_tick_at.isoformat() if self._last_tick_at else None
                ),
                'last_error': self._last_error,
                'last_error_at': (
                    self._last_error_at.isoformat() if self._last_error_at else None
                ),
                # Wave-F5 RC-A1: RPC health surface. auth_failing=true means an
                # infra (401/403) failure hit within the last 15 min — the chain
                # is quote-blind regardless of what "Scanning" says. Host only,
                # never the full URL (Ankr-style URLs embed the API key).
                'rpc_health': {
                    'endpoint_host': (
                        (self.rpc_url or '').split(',')[0]
                        .split('//')[-1].split('/')[0] or None
                    ),
                    'infra_fail_count': int(self._rpc_infra_fail_count),
                    'last_infra_fail_at': (
                        self._rpc_infra_last_fail_at.isoformat()
                        if self._rpc_infra_last_fail_at else None
                    ),
                    'auth_failing': bool(
                        self._rpc_infra_last_fail_at is not None
                        and (datetime.now() - self._rpc_infra_last_fail_at)
                        .total_seconds() < 900
                    ),
                    'last_refresh_at': self._rpc_last_refresh.isoformat(),
                    'refresh_minutes': self._rpc_refresh_minutes,
                },
            }
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO arbitrage_runtime_stats (chain, updated_at, stats)
                    VALUES ($1, NOW(), $2::jsonb)
                    ON CONFLICT (chain) DO UPDATE
                    SET updated_at = NOW(), stats = EXCLUDED.stats
                    """,
                    self.chain_name, json.dumps(snapshot, default=str),
                )
        except Exception as e:
            # Pure observability; never block trading.
            self.logger.debug(f"_persist_runtime_stats failed (non-fatal): {e}")

    # -----------------------------------------------------------------
    # Wave-3: per-(chain, dex_pair, pair_symbol) realized-slippage learning.
    # Replaces static CHAIN_CONFIGS[*]['default_slippage_pct'] once we have
    # >= _realized_slip_min_samples rows in the 7d window.
    # -----------------------------------------------------------------
    @staticmethod
    def _slip_key(buy_dex: str, sell_dex: str, pair_symbol: str) -> str:
        return f"{buy_dex}->{sell_dex}|{pair_symbol}"

    def get_realized_slippage(
        self,
        buy_dex: str,
        sell_dex: str,
        pair_symbol: str,
        *,
        use_p90: bool = False,
    ) -> Optional[float]:
        """
        Look up the median (default) or p90 realized slippage for this
        (chain, dex_pair, pair_symbol). Returns None when no sample exists
        or sample_count < _realized_slip_min_samples; caller falls back to
        the static CHAIN_CONFIGS default.

        Pair symbol normalisation: callers pass `f"{token_in}/{token_out}"`
        (e.g. "USDC/WETH") - same order as logged into arbitrage_trades by
        _log_arb_trade so the refresh aggregation lines up.
        """
        if not self._realized_slip_cache:
            return None
        entry = self._realized_slip_cache.get(
            self._slip_key(buy_dex, sell_dex, pair_symbol)
        )
        if not entry:
            return None
        median_pct, p90_pct, samples = entry
        if samples < self._realized_slip_min_samples:
            return None
        return p90_pct if use_p90 else median_pct

    async def _refresh_realized_slippage(self) -> None:
        """
        Hourly refresh: scan the last 7d of arbitrage_trades on this chain,
        compute median + p90 realized slippage per (dex_pair, pair_symbol),
        upsert into arb_realized_slippage, repopulate in-memory cache.
        Fail-soft: a bad refresh leaves the previous cache in place.

        Realized slippage attribution: for each closed row we back out
            realized_slippage = max(0, gross - net - flash_fee - gas_pct)
        where gross = spread_pct/100, net = profit_loss_pct/100,
        flash_fee = chain_config.flash_loan_fee_pct, and
        gas_pct = metadata.gas_cost / entry_usd (live gas USD from the
        trade row's own write, NOT a recomputed estimate).
        """
        if not self.db_pool:
            return
        now = datetime.now()
        if (
            self._realized_slip_refreshed_at is not None
            and (now - self._realized_slip_refreshed_at).total_seconds() < self._realized_slip_ttl_s
        ):
            return
        try:
            flash_fee_pct = float(self.chain_config.get('flash_loan_fee_pct', 0.0005))
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT buy_dex, sell_dex, spread_pct, profit_loss_pct,
                           entry_usd, metadata, entry_timestamp
                    FROM arbitrage_trades
                    WHERE chain = $1
                      AND entry_timestamp > NOW() - INTERVAL '7 days'
                      AND entry_usd > 0
                      AND spread_pct IS NOT NULL
                      AND profit_loss_pct IS NOT NULL
                    """,
                    self.chain_name,
                )
            # Bucket samples by (dex_pair, pair_symbol)
            buckets: Dict[str, List[Tuple[float, datetime]]] = {}
            for r in rows:
                try:
                    meta = r['metadata'] or {}
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    pair_symbol = meta.get('pair_symbol') or meta.get('token_symbol') or ''
                    if not pair_symbol:
                        continue
                    # Legacy labeling bug: pre-wave-13 rows were written as
                    # "WETH/WETH" (token_out_symbol was hardcoded while pairs
                    # were ('WETH', X)). Degenerate labels carry no usable
                    # slippage signal — drop them from the estimator.
                    parts = pair_symbol.split('/')
                    if len(parts) == 2 and parts[0] == parts[1]:
                        continue
                    gas_cost = float(meta.get('gas_cost') or 0.0)
                    entry_usd = float(r['entry_usd'] or 0.0)
                    if entry_usd <= 0:
                        continue
                    gas_pct = gas_cost / entry_usd
                    gross = float(r['spread_pct'] or 0.0) / 100.0
                    net = float(r['profit_loss_pct'] or 0.0) / 100.0
                    realized = gross - net - flash_fee_pct - gas_pct
                    # Clamp: negative would mean costs > gross which is a
                    # logging artifact (e.g. failed-tx gas-only row) not
                    # slippage. Drop those rather than skew the median.
                    if realized < 0:
                        continue
                    key = self._slip_key(r['buy_dex'], r['sell_dex'], pair_symbol)
                    buckets.setdefault(key, []).append((realized, r['entry_timestamp']))
                except Exception:
                    continue
            # Compute median + p90, upsert
            new_cache: Dict[str, Tuple[float, float, int]] = {}
            async with self.db_pool.acquire() as conn:
                for key, samples in buckets.items():
                    if not samples:
                        continue
                    vals = sorted(s[0] for s in samples)
                    n = len(vals)
                    median_pct = vals[n // 2]
                    p90_idx = min(n - 1, int(0.9 * n))
                    p90_pct = vals[p90_idx]
                    ts_list = [s[1] for s in samples]
                    win_start = min(ts_list)
                    win_end = max(ts_list)
                    new_cache[key] = (median_pct, p90_pct, n)
                    try:
                        dex_pair, pair_symbol = key.split('|', 1)
                    except ValueError:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO arb_realized_slippage (
                            chain, dex_pair, pair_symbol, sample_count,
                            median_pct, p90_pct, window_start, window_end, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        ON CONFLICT (chain, dex_pair, pair_symbol) DO UPDATE
                        SET sample_count = EXCLUDED.sample_count,
                            median_pct   = EXCLUDED.median_pct,
                            p90_pct      = EXCLUDED.p90_pct,
                            window_start = EXCLUDED.window_start,
                            window_end   = EXCLUDED.window_end,
                            updated_at   = NOW()
                        """,
                        self.chain_name, dex_pair, pair_symbol,
                        n, float(median_pct), float(p90_pct),
                        win_start, win_end,
                    )
            self._realized_slip_cache = new_cache
            self._realized_slip_refreshed_at = now
            if new_cache:
                self.logger.info(
                    f"[{self.chain_name.upper()}] Realized-slippage cache refreshed: "
                    f"{len(new_cache)} (dex_pair, pair) keys, "
                    f"trusted={sum(1 for v in new_cache.values() if v[2] >= self._realized_slip_min_samples)}"
                )
        except Exception as e:
            # Pure observability - never block the trading loop.
            self.logger.debug(f"_refresh_realized_slippage failed (non-fatal): {e}")

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
                    # Copy before applying per-chain DB overrides so we never
                    # mutate the shared CHAIN_CONFIGS module dict.
                    self.chain_config = dict(self.chain_config)
                    self._apply_db_chain_overrides()
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

                # WAVE-17: Initialize Uniswap V3 QuoterV2 when enabled for this chain.
                self.v3_quoter_contract = None
                if self.chain_config and self.chain_config.get('v3_quoter_enabled', False):
                    quoter_addr = UNIV3_QUOTER_ADDRESSES.get(self.chain_id)
                    if quoter_addr:
                        try:
                            self.v3_quoter_contract = self.w3.eth.contract(
                                address=Web3.to_checksum_address(quoter_addr),
                                abi=UNIV3_QUOTER_ABI,
                            )
                            self.logger.info(
                                f"   V3 Quoter initialized ({self.chain_name}): "
                                f"{quoter_addr[:10]}... fee-tiers={UNIV3_FEE_TIERS}"
                            )
                        except Exception as e:
                            self.logger.warning(f"Could not init V3 Quoter: {e}")

                # Initialize Flash Loan executor
                # IMPORTANT: Flash loans require a deployed smart contract with IFlashLoanReceiver
                # The contract must implement executeOperation() callback
                # EOA wallets CANNOT receive flash loan callbacks - they will always revert
                #
                # Chain-specific flash loan contracts. A2-04: resolve via
                # secrets_manager (DB-backed + Fernet) first, then fall back
                # to env. Lets the dashboard Credentials page manage these
                # addresses without editing .env on disk.
                # - FLASH_LOAN_RECEIVER_CONTRACT_ETH for Ethereum
                # - FLASH_LOAN_RECEIVER_CONTRACT_ARB for Arbitrum
                # - FLASH_LOAN_RECEIVER_CONTRACT_BASE for Base
                # - FLASH_LOAN_RECEIVER_CONTRACT as fallback (for backwards compat)
                flash_loan_env_key = self.chain_config.get('flash_loan_env_key', 'FLASH_LOAN_RECEIVER_CONTRACT')
                flash_loan_env_fallback = self.chain_config.get('flash_loan_env_fallback')
                flash_loan_contract = await self._get_decrypted_key(flash_loan_env_key)
                if not flash_loan_contract and flash_loan_env_fallback:
                    flash_loan_contract = await self._get_decrypted_key(flash_loan_env_fallback)

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

        # W6: persist a startup-marker snapshot before the first 5-min stats
        # tick. Without this, a newly-restarted engine is invisible to
        # /api/arbitrage/diagnostics for up to 5 minutes (the next
        # _log_stats_if_needed cadence). last_tick_at / last_error stay
        # None on this first write — caller can tell "alive but not yet
        # scanned a pair" from "alive and ticking".
        self._last_tick_at = None
        self._last_error = None
        await self._persist_runtime_stats()

        while self.is_running:
            try:
                self._stats['scans'] += 1
                # W6: stamp tick liveness BEFORE the per-pair work so even a
                # scan that raises inside _check_arb_opportunity leaves a
                # recent _last_tick_at — the dashboard then shows "alive but
                # crash-looping" rather than "stale".
                self._last_tick_at = datetime.now()

                # Get current pair to scan (use chain-specific pairs and tokens)
                token_in_symbol, token_out_symbol = self.arb_pairs[pair_index]
                token_in = self.tokens.get(token_in_symbol)
                token_out = self.tokens.get(token_out_symbol)

                if token_in and token_out:
                    await self._check_arb_opportunity(token_in, token_out, token_in_symbol, token_out_symbol)

                # W11 FIX 2: successful scan iter -> clear any prior 429 streak
                # so a transient rate-limit doesn't keep us in long-backoff mode
                # once the provider recovers.
                if getattr(self, '_rate_limit_streak', 0) > 0:
                    self._rate_limit_streak = 0

                # Move to next pair (round-robin)
                pair_index = (pair_index + 1) % len(self.arb_pairs)

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                # Wave-F5 RC-A1: periodic endpoint re-resolve from pool_engine.
                # The URL used to be pinned at startup for the process
                # lifetime, so a dead key was never swapped out. No-op when
                # pool_engine returns the same URL.
                if (
                    self._rpc_refresh_minutes > 0
                    and (datetime.now() - self._rpc_last_refresh).total_seconds()
                    >= self._rpc_refresh_minutes * 60
                ):
                    self._rpc_last_refresh = datetime.now()
                    await self._rotate_rpc_endpoint()

                # Sleep between scans - adaptive delay based on spread activity
                # In slow-scan mode (30min+ stale negative spreads), use longer delay
                # to conserve RPC quota on chains with no V2 pool activity
                scan_delay = 2
                if hasattr(self, '_slow_scan_active') and self._slow_scan_active:
                    scan_delay = 10  # 5x slower when spreads are static
                await asyncio.sleep(scan_delay)

            except Exception as e:
                # W6: capture last error for /api/arbitrage/diagnostics so the
                # dashboard can show the actual reason engines stop firing
                # (e.g. AttributeError, RPC connection drop, asyncpg pool
                # exhausted) without the operator having to ssh in and tail
                # the rotating log.
                self._last_error = f"{type(e).__name__}: {str(e)[:200]}"
                self._last_error_at = datetime.now()
                self.logger.error(f"Arb loop error: {e}")
                # Best-effort persist so the next /api poll surfaces the
                # error immediately rather than waiting 5min for the next
                # stats tick. Fail-soft inside _persist_runtime_stats.
                try:
                    await self._persist_runtime_stats()
                except Exception:
                    pass
                # W11 FIX 2: jittered exponential backoff on 429 / rate-limit
                # so we stop hammering low-tier RPCs (dRPC public Base tier
                # is the worst offender). Escalates 30s -> 60s -> 120s ->
                # 300s as consecutive 429s pile up; resets to baseline on
                # any successful loop iter. Also reports to pool_engine so
                # the endpoint is demoted in the rotation.
                err_str = str(e).lower()
                is_rate_limited = (
                    '429' in err_str
                    or 'too many requests' in err_str
                    or 'rate limit' in err_str
                )
                if is_rate_limited:
                    self._rate_limit_streak = getattr(self, '_rate_limit_streak', 0) + 1
                    backoff_ladder = [30, 60, 120, 300]
                    base_sleep = backoff_ladder[min(self._rate_limit_streak - 1, len(backoff_ladder) - 1)]
                    jitter = random.uniform(0, base_sleep * 0.2)
                    sleep_s = base_sleep + jitter
                    self.logger.warning(
                        f"⚠️ [{self.chain_name.upper()}] RPC 429 streak #{self._rate_limit_streak} "
                        f"- backing off {sleep_s:.0f}s. Consider provisioning a higher-tier "
                        f"RPC (Alchemy/Infura/Quicknode) for {self.chain_name} and adding it to "
                        f"{self.RPC_ENV_KEY}/{self.RPC_PROVIDER_KEY} in pool_engine."
                    )
                    # Best-effort: tell pool_engine the URL is rate-limited so
                    # rotation can demote it. Fail-soft on any import / lookup.
                    try:
                        from config.pool_engine import get_pool
                        pe = await get_pool()
                        if pe and self.rpc_url:
                            await pe.report_rate_limit(
                                self.RPC_PROVIDER_KEY,
                                self.rpc_url.split(',')[0],
                                duration_seconds=int(sleep_s),
                                error_message=str(e)[:200],
                            )
                    except Exception:
                        pass
                    await asyncio.sleep(sleep_s)
                    # Rotation contract: after penalizing the endpoint, pull a
                    # fresh one from pool_engine instead of re-hammering the
                    # same pinned URL forever.
                    await self._rotate_rpc_endpoint()
                elif self._is_rpc_infra_error(err_str):
                    # Wave-F5 RC-A1: 401/403/unauthorized escaping the scan is
                    # an infra failure too — report + rotate, then keep going.
                    await self._handle_rpc_infra_failure(str(e))
                    await asyncio.sleep(5)
                else:
                    # Reset streak on non-rate-limit errors so a single 429
                    # followed by an unrelated transient doesn't keep us in
                    # long-backoff mode.
                    if getattr(self, '_rate_limit_streak', 0) > 0:
                        self._rate_limit_streak = 0
                    await asyncio.sleep(5)

    @staticmethod
    def _is_rpc_infra_error(err: str) -> bool:
        """Wave-F5 RC-A1: HTTP auth/permission failures are RPC-INFRA
        problems, never market-liquidity signals."""
        e = err.lower()
        return (
            '401' in e
            or '403' in e
            or 'unauthorized' in e
            or 'forbidden' in e
            or 'api key' in e
        )

    async def _handle_rpc_infra_failure(self, error_msg: str) -> None:
        """Wave-F5 RC-A1: report auth/permission RPC failures to pool_engine
        and re-resolve the endpoint instead of poisoning the liquidity
        blacklist. Rotation debounced to once/60s (scan tick is 2s);
        operator-facing WARNING once/hour while the chain is auth-blind."""
        now = datetime.now()
        self._rpc_infra_fail_count += 1
        self._rpc_infra_last_fail_at = now
        last_warn = self._rpc_infra_last_warn_at
        if last_warn is None or (now - last_warn).total_seconds() >= 3600:
            self._rpc_infra_last_warn_at = now
            self.logger.warning(
                f"🔒 [{self.chain_name.upper()}] RPC auth/permission failure "
                f"(#{self._rpc_infra_fail_count}): {error_msg[:120]} — treating as "
                f"INFRA, not liquidity; rotating endpoint via pool_engine. "
                f"Check the {self.RPC_PROVIDER_KEY} key/quota."
            )
            self._record_near_miss('rpc_infra_error', detail=error_msg[:80])
        # Demote the endpoint in pool_engine (fail-soft).
        try:
            from config.pool_engine import get_pool
            pe = await get_pool()
            if pe and self.rpc_url:
                await pe.report_failure(
                    self.RPC_PROVIDER_KEY,
                    self.rpc_url.split(',')[0],
                    error_type='auth',
                    error_message=error_msg[:200],
                )
        except Exception:
            pass
        last_rot = self._rpc_infra_last_rotate_at
        if last_rot is None or (now - last_rot).total_seconds() >= 60:
            self._rpc_infra_last_rotate_at = now
            await self._rotate_rpc_endpoint()

    async def _rotate_rpc_endpoint(self) -> None:
        """Pull a fresh endpoint from pool_engine after a rate-limit/auth
        penalty (or the periodic re-resolve) and rebuild w3 + contract
        handles when the URL actually changed.
        Fail-soft: any error leaves the current connection in place."""
        try:
            from config.rpc_provider import RPCProvider
            new_url = await RPCProvider.get_rpc(self.RPC_PROVIDER_KEY)
            current = (self.rpc_url or '').split(',')[0]
            if not new_url or new_url == current:
                return
            w3 = Web3(Web3.HTTPProvider(new_url))
            if not w3.is_connected() or w3.eth.chain_id != self.chain_id:
                return  # wrong/unreachable endpoint — keep what we have
            self.rpc_url = new_url
            self.w3 = w3
            self.router_contracts = {}
            for name, address in self.routers.items():
                try:
                    self.router_contracts[name] = w3.eth.contract(
                        address=Web3.to_checksum_address(address), abi=ROUTER_ABI
                    )
                except Exception:
                    pass
            if self.v3_quoter_contract is not None:
                quoter_addr = UNIV3_QUOTER_ADDRESSES.get(self.chain_id)
                if quoter_addr:
                    self.v3_quoter_contract = w3.eth.contract(
                        address=Web3.to_checksum_address(quoter_addr),
                        abi=UNIV3_QUOTER_ABI,
                    )
            if self.flash_loan_executor:
                self.flash_loan_executor.w3 = w3
                self.flash_loan_executor.flash_loan_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(self.flash_loan_executor.receiver_contract),
                    abi=FLASH_LOAN_CONTRACT_ABI,
                )
            if self.flashbots_executor:
                self.flashbots_executor.w3 = w3
            self.logger.info(
                f"🔁 [{self.chain_name.upper()}] Rotated RPC endpoint via pool_engine"
            )
        except Exception as e:
            self.logger.debug(f"RPC rotation failed (non-fatal): {e}")

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

            # Wave-3: hourly refresh of per-(chain, dex_pair, pair) realized-
            # slippage cache. TTL-gated inside the method (no-op when fresh),
            # so a 5-min cadence here is safe.
            await self._refresh_realized_slippage()
            # Wave-3: snapshot runtime stats (gas-spend / budget / spike mult /
            # realized-slip cache size) for the dashboard tile. Fail-soft.
            await self._persist_runtime_stats()

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
                                    f"   💤 SLOW-SCAN MODE: 30min+ stale negative spreads on {self.chain_name} "
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

    def _pool_tvl_usd(
        self,
        dex_name: str,
        token_a: str,
        token_b: str,
        eth_price_usd: float,
    ) -> Optional[float]:
        """
        WAVE-17 pool-reserve floor helper.

        Fetches on-chain reserves for the (dex_name, token_a/token_b) UniV2
        pair and returns an approximate TVL in USD.  Returns None when the
        factory/pair address is unknown or the RPC call fails (caller treats
        None as "skip check, proceed").

        TVL approximation: we only know the WETH leg price reliably.
        If neither token is WETH we fall back to None (pass-through).
        If one leg is WETH: TVL = 2 * weth_reserve * eth_price_usd
        (both legs assumed ~equal value in a balanced pool).

        This is intentionally conservative — it is better to false-negative
        (skip a thin pool we could have quoted) than false-positive (allow
        a garbage quote from a $500 pool into the spread calculation).

        Result is NOT cached per-call (reserves can change each block).
        The liquidity blacklist handles repeated failures at the pair level.
        """
        if not self.w3 or not self.chain_id:
            return None
        factory_map = UNIV2_FACTORIES.get(dex_name, {})
        factory_addr = factory_map.get(self.chain_id) if isinstance(factory_map, dict) else None
        if not factory_addr:
            return None
        weth_addr = (self.tokens.get('WETH') or '').lower()
        try:
            factory = self.w3.eth.contract(
                address=Web3.to_checksum_address(factory_addr),
                abi=UNIV2_FACTORY_ABI,
            )
            pair_addr = factory.functions.getPair(
                Web3.to_checksum_address(token_a),
                Web3.to_checksum_address(token_b),
            ).call()
            zero = '0x0000000000000000000000000000000000000000'
            if pair_addr.lower() == zero:
                return 0.0  # No pool exists
            pair = self.w3.eth.contract(
                address=Web3.to_checksum_address(pair_addr),
                abi=UNIV2_PAIR_ABI,
            )
            r0, r1, _ = pair.functions.getReserves().call()
            # Identify which reserve is WETH so we can price it
            if token_a.lower() == weth_addr:
                weth_reserve_wei = r0
            elif token_b.lower() == weth_addr:
                weth_reserve_wei = r1
            else:
                # Neither side is WETH — can't price reliably; return None (pass-through)
                return None
            weth_reserve_eth = weth_reserve_wei / 1e18
            # Multiply by 2: both sides of the pool have equal USD value at equilibrium
            return 2.0 * weth_reserve_eth * eth_price_usd
        except Exception:
            return None

    def _quote_v3(self, token_in: str, token_out: str, amount_in: int) -> Optional[int]:
        """
        WAVE-17: Uniswap V3 QuoterV2 price discovery (Ethereum mainnet proof-of-concept).

        Calls quoteExactInputSingle across all UNIV3_FEE_TIERS and returns the
        best (highest) amountOut.  Returns None when the quoter is not initialized
        or all fee tiers revert (no pool for that pair/tier).

        NOTE: QuoterV2.quoteExactInputSingle is marked `nonpayable` in the ABI but
        behaves as a view call (it simulates the swap, reverts, and returns the
        output via the revert data — web3.py's `.call()` handles this correctly).
        It does NOT broadcast a transaction.

        The V3 price is added to buy_prices under key 'uniswap_v3' so the
        existing best-buy-dex selection logic picks it up automatically.
        The sell leg still uses V2 routers (the engine's flash-loan contract
        only supports V2 swapExactTokensForTokens).  This means:
        - V3 price discovery can identify a genuine cross-DEX spread
          (e.g. V3 pool is cheaper buy-side than V2 Sushi)
        - BUT execution is limited to V2 venues for both legs
        - If best_buy_dex == 'uniswap_v3', the execute path MUST fall back
          to the best V2 buy price for actual execution
        This constraint is documented as a wave-18 TODO: add V3 SwapRouter
        execution leg to the flash-loan contract so V3-discovered spreads
        can actually be executed.
        """
        if not self.v3_quoter_contract:
            return None
        best_out: Optional[int] = None
        for fee in UNIV3_FEE_TIERS:
            try:
                result = self.v3_quoter_contract.functions.quoteExactInputSingle(
                    (
                        Web3.to_checksum_address(token_in),
                        Web3.to_checksum_address(token_out),
                        amount_in,
                        fee,
                        0,  # sqrtPriceLimitX96 = 0 means no price limit
                    )
                ).call()
                amount_out = result[0]  # amountOut is first return value
                if amount_out > 0:
                    if best_out is None or amount_out > best_out:
                        best_out = amount_out
            except Exception:
                pass
        return best_out

    async def _check_arb_opportunity(self, token_in: str, token_out: str, token_symbol: str = "UNKNOWN", token_out_symbol: str = "UNKNOWN") -> bool:
        """Check price difference between two DEXs. Returns True if opportunity found."""
        try:
            pair_key = f"{token_symbol}_liquidity"

            # Skip blacklisted pairs to avoid wasting RPC calls
            if self._is_pair_blacklisted(pair_key):
                return False

            # ECON dormancy: pair proven uneconomic (rolling p90 spread below
            # breakeven) sleeps until its probe time. Re-probes on expiry.
            econ_key = f"{token_symbol}/{token_out_symbol}"
            dormant_until = self._pair_dormant_until.get(econ_key)
            if dormant_until is not None:
                if datetime.now() < dormant_until:
                    return False
                del self._pair_dormant_until[econ_key]

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

            # Identity guard: token_in == token_out (e.g. a misconfigured
            # ('WETH','WETH') pair) produces a degenerate quote AND poisons
            # arbitrage_trades / arb_realized_slippage with "WETH/WETH" rows.
            if token_checksum.lower() == weth_checksum.lower():
                self.logger.warning(
                    f"[arb-config] Pair {token_symbol}/{token_out_symbol} is degenerate "
                    f"(token_in == token_out) -- skipping"
                )
                return False

            # WAVE-13 GUARD: token_out MUST be the WETH address for this chain.
            # If it is not, borrow_amount (ETH wei) is compared against a non-WETH
            # return value, producing a nonsensical -100% spread. Reject early.
            chain_weth = self.tokens.get('WETH', '')
            if chain_weth and weth_checksum.lower() != chain_weth.lower():
                self.logger.warning(
                    f"[arb-config] Pair {token_symbol}/{token_out_symbol} has non-WETH "
                    f"borrow asset {token_out_symbol} -- skipping (requires USDC flash loan). "
                    f"Fix: use ('{token_symbol}', 'WETH') pair format."
                )
                return False

            # ============================================================
            # STEP 1: Query BUY direction - WETH → token (how much token can we buy?)
            # This matches the contract's buyRouter (asset → intermediateToken)
            # WAVE-17: pool-reserve floor — skip any DEX whose WETH/token pool
            # TVL is below min_pool_tvl_usd (default $50k).  A near-empty pool
            # returns a mathematically valid but economically garbage quote that
            # creates impossible-looking spreads when combined with the real price
            # from a liquid pool.  We need the ETH price for the TVL calculation;
            # fetch it once here and reuse below for gas cost estimation.
            # ============================================================
            _min_pool_tvl_usd = float(self.chain_config.get('min_pool_tvl_usd', 50_000.0))
            _reserve_eth_price: Optional[float] = None
            try:
                _reserve_eth_price = await self.price_fetcher.get_price('eth')
            except Exception:
                pass

            buy_prices = {}
            buy_errors = {}
            rpc_infra_error: Optional[str] = None
            for name, contract in self.router_contracts.items():
                # Pool-reserve floor: skip DEXes with insufficient liquidity
                if _reserve_eth_price and _reserve_eth_price > 0:
                    tvl = self._pool_tvl_usd(name, token_in, token_out, _reserve_eth_price)
                    if tvl is not None and tvl < _min_pool_tvl_usd:
                        buy_errors[name] = f"thin_pool tvl=${tvl:.0f}<${_min_pool_tvl_usd:.0f}"
                        continue
                try:
                    amounts = contract.functions.getAmountsOut(borrow_amount, [weth_checksum, token_checksum]).call()
                    buy_prices[name] = amounts[1]  # How many tokens we get for our WETH
                except Exception as e:
                    # Wave-F5 RC-A1: 401/403/unauthorized is an RPC-INFRA
                    # failure — never count it as a per-DEX quote error.
                    err = str(e)
                    if self._is_rpc_infra_error(err):
                        rpc_infra_error = err
                        continue
                    buy_errors[name] = err[:50]

            # WAVE-17: V3 Quoter price discovery (Ethereum mainnet proof-of-concept).
            # quoteExactInputSingle probes all fee tiers; best amountOut is added to
            # buy_prices under key 'uniswap_v3'.  See _quote_v3 docstring for the
            # important caveat: V3 discovery + V2 execution mismatch means the
            # 'uniswap_v3' key signals a price-discovery signal only, not an
            # executable buy-side.  Wave-18 adds the V3 execution leg.
            if self.v3_quoter_contract:
                v3_out = self._quote_v3(token_out, token_in, borrow_amount)  # WETH→token
                if v3_out and v3_out > 0:
                    buy_prices['uniswap_v3'] = v3_out

            # Log diagnostic info periodically (every 60 scans = ~2 minutes)
            if self._total_pairs_scanned % 60 == 1:
                if buy_prices:
                    v3_note = " (incl V3)" if 'uniswap_v3' in buy_prices else ""
                    self.logger.debug(f"📊 [{token_symbol}] Buy prices (WETH→token): {len(buy_prices)} DEXs{v3_note}")
                else:
                    self.logger.warning(f"⚠️ [{token_symbol}] No buy prices from any DEX. Errors: {buy_errors}")

            if len(buy_prices) < 2:
                if rpc_infra_error is not None:
                    # Wave-F5 RC-A1: auth failure — do NOT blacklist the pair
                    # as no-liquidity (that poisons 60min of scanning per pair
                    # for an infra problem). Report + rotate instead.
                    await self._handle_rpc_infra_failure(rpc_infra_error)
                    return False
                # Need at least 2 DEXs for arbitrage
                self._update_liquidity_blacklist(pair_key, has_liquidity=False)
                return False

            # Liquidity found - clear from blacklist if present
            self._update_liquidity_blacklist(pair_key, has_liquidity=True)
            self._pairs_with_liquidity += 1

            # Find best BUY DEX (gives most tokens for our WETH = lowest token price)
            best_buy_dex = max(buy_prices, key=buy_prices.get)
            tokens_bought = buy_prices[best_buy_dex]

            # WAVE-17 V3 execution caveat: if the best buy is from the V3 quoter,
            # V3 price discovery is valid for spread detection BUT the flash-loan
            # contract only supports V2 swapExactTokensForTokens.  Use the best V2
            # buy price for the sell-leg quote (tokens_bought) so the round-trip
            # P&L reflects what the V2 buy leg would actually deliver.
            # The execute path already guards: router_contracts.get('uniswap_v3')
            # returns None -> execute returns None -> no trade fires. This guard
            # ensures the sell-leg amount is realistic, not overstated.
            _v3_best_noted = False
            if best_buy_dex == 'uniswap_v3':
                v2_prices = {k: v for k, v in buy_prices.items() if k != 'uniswap_v3'}
                if v2_prices:
                    _v3_best_noted = True
                    # Use V2 tokens_bought for sell-leg (executable amount)
                    best_v2_buy_dex = max(v2_prices, key=v2_prices.get)
                    tokens_bought = v2_prices[best_v2_buy_dex]
                    # best_buy_dex stays 'uniswap_v3' for spread logging purposes
                else:
                    # No V2 prices at all — V3-only, not executable yet
                    return False

            if tokens_bought == 0:
                return False

            # ============================================================
            # WAVE-16 PRICE-SPREAD GATE: compute the true cross-DEX price divergence
            # before running the sell-leg queries.
            #
            # All buy_prices entries are getAmountsOut(borrow_amount, [WETH, token])
            # results in the same units (raw intermediate-token per borrow_amount WETH).
            # The ratio max/min is therefore dimensionless and decimal-agnostic
            # (WBTC 8-dec, USDC 6-dec, DAI 18-dec -- units cancel out).
            #
            # price_spread = (max_buy - min_buy) / min_buy
            # = pure cross-DEX price divergence, independent of trade size/pool depth.
            #
            # Without this gate, the round-trip P&L (profit_bps) includes 100% price
            # impact from near-empty SELL pools (-7000 to -9900 bps), which is
            # physically real but not "cross-DEX price spread".  The operator sees
            # -7177 bps on WBTC/WETH and (correctly) diagnoses a bug -- but the
            # number is the price-impact artefact, not a formula error.
            #
            # Empirical DEX price divergence on liquid pairs: 1-30 bps.
            # If max/min spread < min_price_spread_bps (default 10 bps), no arb possible
            # regardless of pool depth -- skip the sell-leg RPC calls entirely.
            # ============================================================
            min_buy_val = min(buy_prices.values())
            max_buy_val = max(buy_prices.values())
            # Guard against zero (empty pool edge case)
            if min_buy_val > 0:
                price_spread_bps = (max_buy_val - min_buy_val) / min_buy_val * 10_000
            else:
                price_spread_bps = 0.0

            pair_label_early = f"{token_symbol}/{token_out_symbol}"
            min_dex_name = min(buy_prices, key=buy_prices.get)

            # ECON: rolling-24h spread history per pair. Once we have enough
            # samples, a p90 below full breakeven means the pair structurally
            # cannot pay for gas + flash fee + slippage on this chain — refuse
            # to keep scanning it (dormant; re-probed after dormancy expires).
            _now_ts = datetime.now()
            _hist = self._pair_spread_history.setdefault(econ_key, deque(maxlen=2000))
            _hist.append((_now_ts, price_spread_bps))
            _cutoff = _now_ts.timestamp() - 86400
            while _hist and _hist[0][0].timestamp() < _cutoff:
                _hist.popleft()
            if len(_hist) >= self._dormancy_min_samples:
                _vals = sorted(s[1] for s in _hist)
                _p90 = _vals[min(len(_vals) - 1, int(0.9 * len(_vals)))]
                _breakeven_bps = (await self._breakeven_frac()) * 10_000
                if _p90 < _breakeven_bps:
                    self._pair_dormant_until[econ_key] = (
                        _now_ts + timedelta(minutes=self._pair_dormancy_minutes)
                    )
                    self._pair_spread_history[econ_key] = deque(maxlen=2000)
                    self._record_near_miss(
                        'pair_dormant_uneconomic',
                        pair=econ_key,
                        price_spread_bps=round(_p90, 2),
                        threshold_bps=round(_breakeven_bps, 2),
                        detail=(
                            f"p90 spread {_p90:.2f}bps < breakeven {_breakeven_bps:.2f}bps "
                            f"over {len(_vals)} samples; dormant "
                            f"{self._pair_dormancy_minutes:.0f}min"
                        ),
                    )
                    return False
            # Use a conservative 10 bps floor: pairs with <10 bps DEX price divergence
            # cannot produce net profit after Aave fee (5 bps) + slippage (~50 bps).
            # The threshold is deliberately loose so we don't filter marginal real opps.
            _min_price_spread_bps = float(self.chain_config.get('min_price_spread_bps', 10.0))
            if price_spread_bps < _min_price_spread_bps:
                if self._total_pairs_scanned % 120 == 1:
                    self._record_near_miss(
                        'price_spread_too_low',
                        pair=pair_label_early,
                        buy_dex=best_buy_dex, sell_dex=min_dex_name,
                        price_spread_bps=round(price_spread_bps, 2),
                        threshold_bps=round(_min_price_spread_bps, 2),
                    )
                return False

            # ============================================================
            # WAVE-17 UPPER-SPREAD SANITY CAP: any BUY-side price_spread_bps
            # above a sane ceiling is a data artifact, not an opportunity.
            # Physical reality: liquid V2 pools on the same assets can diverge
            # at most by ~200-400 bps before bots arbitrage them; 500 bps is a
            # conservative ceiling that still allows genuine large opps.
            # Values above this come from near-empty pools (e.g. 1 ETH TVL pool
            # returning extreme amounts) that slipped through the min-spread floor
            # because the other DEX quoted correctly. Record as thin_pool_artifact
            # instead of logging raw_spread_negative -7000 bps which confused
            # operators into thinking the spread formula was broken.
            # ============================================================
            _upper_spread_cap_bps = float(self.chain_config.get('upper_spread_cap_bps', 500.0))
            if price_spread_bps > _upper_spread_cap_bps:
                self._record_near_miss(
                    'thin_pool_artifact',
                    pair=pair_label_early,
                    buy_dex=best_buy_dex, sell_dex=min_dex_name,
                    price_spread_bps=round(price_spread_bps, 2),
                    cap_bps=round(_upper_spread_cap_bps, 2),
                )
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
                # Sample sparsely - a negative raw spread is the common case.
                if self._total_pairs_scanned % 120 == 1:
                    _negative_bps = (
                        (weth_returned - amount_owed) / borrow_amount * 10_000
                        if borrow_amount > 0 else 0.0
                    )
                    self._record_near_miss(
                        'raw_spread_negative',
                        pair=f"{token_symbol}/{token_out_symbol}",
                        buy_dex=best_buy_dex, sell_dex=best_sell_dex,
                        profit_bps=round(_negative_bps, 2),
                    )
                return False  # No profit possible

            profit = weth_returned - amount_owed
            raw_spread = profit / borrow_amount

            # A2-03: replace hardcoded 0.5% cost with chain-aware estimate.
            #   gas cost (USD)   -> converted to ETH via live ETH price,
            #                       then divided by borrow_amount (also ETH)
            #   slippage cost    -> per-chain default (Eth 0.4%, L2 0.5%)
            #   flash-loan fee   -> already deducted via amount_owed
            try:
                eth_price = await self.price_fetcher.get_price('eth')
                if eth_price and eth_price > 0:
                    gas_usd = await self._gas_cost_usd_per_tx(eth_price_usd=eth_price)
                    gas_eth = gas_usd / float(eth_price)
                    gas_frac = gas_eth / (borrow_amount / 1e18) if borrow_amount > 0 else 0.0
                else:
                    gas_frac = 0.0  # Cannot estimate USD-denominated gas; skip
            except Exception:
                gas_frac = 0.0
            # Wave-3: prefer the per-(chain, dex_pair, pair_symbol) realized
            # slippage when we have >= _realized_slip_min_samples in the 7d
            # window; otherwise fall back to the chain-level static default.
            # Use p90 here (not median) for the pre-execute gate to bias
            # toward skipping trades that are only marginal under typical
            # fills; median is fine for PnL accounting at log time.
            pair_symbol_key = f"{token_symbol}/{token_out_symbol}"
            slippage_frac = self.get_realized_slippage(
                best_buy_dex, best_sell_dex, pair_symbol_key, use_p90=True
            )
            if slippage_frac is None:
                slippage_frac = float(self.chain_config.get('default_slippage_pct', 0.005))
            estimated_costs = gas_frac + slippage_frac
            net_spread = raw_spread - estimated_costs

            # A2-06 / enhancement #4: gate by the adaptive threshold so we
            # raise the bar during gas spikes instead of executing thin trades.
            effective_threshold = self._adaptive_min_profit_threshold()
            pair_label = f"{token_symbol}/{token_out_symbol}"
            if net_spread > effective_threshold:
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
                    self._record_near_miss(
                        'daily_cap',
                        pair=pair_label,
                        buy_dex=best_buy_dex, sell_dex=best_sell_dex,
                        profit_bps=round(net_spread * 10_000, 2),
                        threshold_bps=round(effective_threshold * 10_000, 2),
                        detail=f"cap={self._max_executions_per_pair_per_day}",
                    )
                    return True  # Silently skip - already hit daily limit

                # Check cooldown - don't spam same opportunity
                if self._last_opportunity_key == opp_key and self._last_opportunity_time:
                    elapsed = (now - self._last_opportunity_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        self._record_near_miss(
                            'cooldown',
                            pair=pair_label,
                            buy_dex=best_buy_dex, sell_dex=best_sell_dex,
                            profit_bps=round(net_spread * 10_000, 2),
                            threshold_bps=round(effective_threshold * 10_000, 2),
                            detail=f"elapsed={elapsed:.0f}s/{self._opportunity_cooldown}s",
                        )
                        return True  # Same opportunity within cooldown

                # New opportunity or cooldown expired - log and execute
                self._last_opportunity_key = opp_key
                self._last_opportunity_time = now

                # Update daily execution count
                self._pair_execution_count[opp_key] = current_count + 1
                remaining = self._max_executions_per_pair_per_day - (current_count + 1)

                # Use correct decimal divisor for logging
                # Path is: borrow WETH (token_out) -> swap to token (token_in) -> swap back to WETH
                in_divisor = 10 ** token_in_decimals  # decimals of the intermediate token
                out_decimals = TOKEN_DECIMALS.get(token_out_symbol, 18)
                out_divisor = 10 ** out_decimals  # WETH decimals (18)

                # A2-01: prior refactor renamed forward_output -> tokens_bought
                # and final_output -> weth_returned but missed this log line,
                # which raised NameError on every real opportunity and silently
                # killed execution via the outer except.
                spike_mult = self._gas_spike_multiplier()
                self.logger.info(f"🚨 [{self.chain_name.upper()}] ARBITRAGE OPPORTUNITY [{token_symbol}/{token_out_symbol}]: Buy on {best_buy_dex}, Sell on {best_sell_dex}. Raw: {raw_spread:.2%}, Net: {net_spread:.2%} (threshold {effective_threshold:.2%}, gas-mult {spike_mult:.2f}x) (#{current_count + 1} today, {remaining} remaining)")
                self.logger.info(f"   Path: {borrow_amount/out_divisor:.4f} {token_out_symbol} → {tokens_bought/in_divisor:.4f} {token_symbol} → {weth_returned/out_divisor:.4f} {token_out_symbol} (profit: {profit/out_divisor:.6f} {token_out_symbol})")
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
                    token_symbol=token_symbol,
                    token_out_symbol=token_out_symbol,
                    # Pre-fee round-trip spread: _log_arb_trade deducts
                    # flash fee + slippage + gas from THIS value.
                    gross_spread=(weth_returned - borrow_amount) / borrow_amount,
                )
                return True
            # Net spread is positive but BELOW the effective gate. This is
            # the single most-common reason no trades fire on a busy chain
            # with thin spreads; log every Nth so we don't spam.
            if self._total_pairs_scanned % 40 == 1:
                gas_usd_now = self._gas_cost_usd_cache or 0.0
                self._record_near_miss(
                    'min_profit',
                    pair=pair_label,
                    buy_dex=best_buy_dex, sell_dex=best_sell_dex,
                    profit_bps=round(net_spread * 10_000, 2),
                    threshold_bps=round(effective_threshold * 10_000, 2),
                    gas_usd=round(gas_usd_now, 4),
                )
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
        token_symbol: str = "UNKNOWN",
        token_out_symbol: str = "WETH",
        gross_spread: Optional[float] = None,
    ):
        """Execute the arbitrage trade using flash loans and Flashbots.

        expected_profit = NET spread (post gas+slippage estimate, used for gating/logs).
        gross_spread    = round-trip spread BEFORE gas/slippage/flash-fee deductions;
                          this is what _log_arb_trade deducts costs from. Passing the
                          net value there double-counted gas+slippage in recorded PnL.
        """
        token_decimals = TOKEN_DECIMALS.get(token_symbol, 18)
        token_divisor = 10 ** token_decimals
        if gross_spread is None:
            gross_spread = expected_profit
        self.logger.info(f"⚡ [{self.chain_name.upper()}] Executing Arbitrage [{token_symbol}]: {buy_dex} -> {sell_dex} | Amount: {amount/token_divisor:.4f} {token_symbol} | Expected net: +{expected_profit:.2%}")

        # Kill-switch / pause / dry-run gate. Fail-safe: if the gate itself is
        # unreadable we must NOT broadcast.
        try:
            from core.dry_run import should_skip_live
            skip_live = should_skip_live(self.dry_run, module='arbitrage')
        except Exception:
            skip_live = True

        # shadow_mode (migration 097, default ON) is documented as 'never
        # enter the execute path' but was previously only surfaced in stats —
        # flipping live_execution_enabled alone could broadcast with shadow
        # still on. Enforce the documented shadow -> live ordering here.
        if not skip_live and self.shadow_mode:
            self.logger.info(
                "shadow_mode=true — recording simulated trade only"
            )
            skip_live = True

        # Defense-in-depth: broadcast requires the explicit live opt-in
        # (live_execution_enabled, migration 097 default false) even when
        # dry_run is off and no kill switch is set.
        if not skip_live and not self.live_execution_enabled:
            self.logger.info(
                "live_execution_enabled=false — recording simulated trade only"
            )
            skip_live = True

        if skip_live:
            # Simulate execution
            await asyncio.sleep(0.5)
            self.logger.info(f"✅ [{self.chain_name.upper()}] Flash Swap Executed (DRY RUN) [{token_symbol}]")
            await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, gross_spread, "DRY_RUN", token_symbol, token_out_symbol)
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

        # P1-06: pre-execute risk gate. DRY_RUN trades are NOT validated above (return at :1563).
        if self.risk_manager is not None:
            try:
                allowed, reason = await self.risk_manager.validate_trade(token_in, amount)
            except Exception as e:
                self.logger.warning(f"validate_trade raised: {e}; refusing execute")
                self._record_near_miss(
                    'risk_manager_error',
                    pair=token_symbol, buy_dex=buy_dex, sell_dex=sell_dex,
                    detail=str(e)[:120],
                )
                return
            if not allowed:
                self.logger.warning(f"⛔ Risk manager rejected EVM arb {token_in[:10]}: {reason}")
                self._record_near_miss(
                    'risk_manager',
                    pair=token_symbol, buy_dex=buy_dex, sell_dex=sell_dex,
                    detail=str(reason)[:120],
                )
                return

        # A2-07 / enhancement #3: hourly gas-budget tracker. Refuse new
        # broadcasts when the rolling 1-hour gas spend would exceed the
        # operator-configured USD budget. Charges optimistically; if the
        # tx reverts we keep the charge (worst case = we wait an hour).
        try:
            projected_gas_usd = await self._gas_cost_usd_per_tx()
        except Exception:
            projected_gas_usd = 0.0
        allowed_budget, reason = self._gas_budget_check_and_charge(projected_gas_usd)
        if not allowed_budget:
            self.logger.warning(f"⛔ Gas budget gate: {reason}; skipping execution")
            self._record_near_miss(
                'gas_budget',
                pair=token_symbol, buy_dex=buy_dex, sell_dex=sell_dex,
                gas_usd=round(projected_gas_usd, 4),
                detail=(
                    f"spend=${self._gas_spend_usd_hour:.2f}/"
                    f"${self._gas_budget_usd_per_hour:.2f}"
                ),
            )
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
                # LIVE honesty: a broadcast is not a fill. Wait (bounded) for
                # the receipt; a reverted flash loan is gas burnt with NO fill
                # and must not be booked as a profitable closed trade.
                receipt_ok = await self._confirm_receipt(tx_hash)
                if receipt_ok is False:
                    self.logger.error(
                        f"❌ [{self.chain_name.upper()}] Flash loan tx REVERTED "
                        f"[{token_symbol}]: {tx_hash} — gas burnt, no fill recorded"
                    )
                    self._record_near_miss(
                        'tx_reverted',
                        pair=token_symbol, buy_dex=buy_dex, sell_dex=sell_dex,
                        detail=tx_hash[:24],
                    )
                    await self._send_error_alert(
                        error_type="Tx Reverted",
                        details=f"Flash loan reverted on-chain for {token_symbol}\nTx: {tx_hash}",
                        token_symbol=token_symbol,
                        tx_hash=tx_hash,
                    )
                    return
                if receipt_ok is None:
                    # Punch-list #6: do NOT book an unconfirmed tx as a closed
                    # won fill. A receipt-timeout tx that later reverts would
                    # become a fake win in arbitrage_trades and poison live PnL
                    # + the realized-slippage learner. Record an UNVERIFIED
                    # near-miss + alert the operator with the hash to check;
                    # only a confirmed receipt.status==1 books a real fill.
                    self.logger.warning(
                        f"⚠️ [{self.chain_name.upper()}] Receipt UNCONFIRMED after "
                        f"{self._receipt_confirm_timeout_s:.0f}s [{token_symbol}] "
                        f"— NOT booking; VERIFY {tx_hash} on-chain"
                    )
                    self._record_near_miss(
                        'tx_unconfirmed',
                        pair=token_symbol, buy_dex=buy_dex, sell_dex=sell_dex,
                        detail=tx_hash[:24],
                    )
                    await self._send_error_alert(
                        error_type="Tx Unconfirmed",
                        details=(
                            f"Arbitrage tx for {token_symbol} not confirmed within "
                            f"{self._receipt_confirm_timeout_s:.0f}s — NOT booked as a "
                            f"fill. Verify on-chain.\nTx: {tx_hash}"
                        ),
                        token_symbol=token_symbol,
                        tx_hash=tx_hash,
                    )
                    return
                self.logger.info(f"✅ [{self.chain_name.upper()}] Arbitrage executed [{token_symbol}]: {tx_hash}")
                await self._log_arb_trade(buy_dex, sell_dex, token_in, amount, gross_spread, tx_hash, token_symbol, token_out_symbol)
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
            '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
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

        # Punch-list #3: hard-disabled, mirroring the MB-05 triangular gate.
        # Both legs below are built with amountOutMin=0 (unbounded slippage /
        # sandwich exposure) and the sell leg sells the INPUT amount instead of
        # the buy leg's actual output ("Simplified" in-code). It only failed
        # safe because Flashbots simulation + MB-04 rejected it. Refuse to
        # execute until rebuilt with real slippage floors on both legs, chained
        # leg outputs, and _get_next_nonce. Re-enabling requires a deliberate
        # code change, not a config flip — there is no flag that can reach the
        # unsafe body.
        self.logger.warning(
            "Direct-swap execution path is disabled (punch-list #3): "
            "amountOutMin=0 on both legs + sell leg uses input units. Deploy "
            f"the flash-loan receiver contract for {self.chain_name.upper()} "
            "or rebuild this path before use."
        )
        self._record_near_miss(
            'direct_swap_disabled',
            buy_dex=buy_dex, sell_dex=sell_dex,
            pair=f"{token_in[:10]}/{token_out[:10]}",
        )
        return None

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

            # MB-04: refuse one-legged fallback. Sending only the BUY leg to the
            # public mempool would acquire token_out with no atomic SELL - guaranteed
            # inventory leak. If Flashbots is unavailable, skip the opportunity.
            self.logger.warning(
                "Atomic execution unavailable (Flashbots failed/rejected/unconfirmed); "
                "refusing one-legged broadcast to avoid inventory leak"
            )
            return None

        except Exception as e:
            self.logger.error(f"Direct swap execution error: {e}")
            return None

    async def _confirm_receipt(self, tx_hash: str) -> Optional[bool]:
        """Bounded wait for a live tx receipt.

        Returns True=confirmed (status 1), False=reverted (status 0),
        None=unknown within the timeout (caller logs and books optimistically).
        Timeout 0 disables the wait entirely (returns None immediately).
        """
        timeout_s = float(getattr(self, '_receipt_confirm_timeout_s', 90.0) or 0.0)
        if timeout_s <= 0 or not self.w3:
            return None
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    return receipt.status == 1
            except Exception:
                # TransactionNotFound until mined; transient RPC errors retry.
                pass
            await asyncio.sleep(3.0)
        return None

    async def _log_arb_trade(
        self,
        buy_dex: str,
        sell_dex: str,
        token: str,
        amount: int,
        profit_pct: float,
        tx_hash: str,
        token_symbol: str = "UNKNOWN",
        token_out_symbol: str = "WETH",
    ):
        """Log arbitrage trade to database with REALISTIC P&L calculation"""
        if not self.db_pool:
            return

        try:
            import uuid
            amount_eth = amount / 1e18

            # Get real ETH price for USD P&L accounting.
            # In DRY_RUN mode a CoinGecko outage must not silently swallow the
            # DB record (which is the only signal that an opportunity cleared
            # all gates).  Use a conservative $2000 fallback so the row is
            # written; the `metadata.eth_price_fallback=true` flag lets the
            # operator identify rows whose USD figures are approximate.
            eth_price = await self.price_fetcher.get_price('eth')
            _eth_price_fallback = False
            if not eth_price:
                if self.dry_run:
                    eth_price = 2000.0
                    _eth_price_fallback = True
                    self.logger.warning(
                        "ETH price unavailable - using $2000 fallback for DRY_RUN record"
                    )
                else:
                    self.logger.warning("Cannot log trade - ETH price unavailable")
                    return

            # A2-02 / A2-05: chain-aware cost deductions. Previously the
            # method used GAS_COST_USD=15 and SLIPPAGE_ESTIMATE_PCT=0.006
            # regardless of chain - poisoned PnL on both ETH (under-counted)
            # and L2s (over-counted). Now driven by CHAIN_CONFIGS profile +
            # live gas oracle.
            flash_loan_fee_pct = float(self.chain_config.get('flash_loan_fee_pct', 0.0005))
            # Wave-3: prefer rolling-7d realized median for PnL accounting; the
            # pre-execute gate already uses p90. Falls back to chain static
            # default for cold-start (sample_count < min_samples).
            pair_symbol = f"{token_symbol}/{token_out_symbol}"
            realized = self.get_realized_slippage(
                buy_dex, sell_dex, pair_symbol, use_p90=False
            )
            slippage_estimate_pct = (
                realized
                if realized is not None
                else float(self.chain_config.get('default_slippage_pct', 0.005))
            )
            slippage_source = 'realized_median' if realized is not None else 'static_default'
            try:
                gas_cost_usd = await self._gas_cost_usd_per_tx(eth_price_usd=eth_price)
            except Exception:
                gas_cost_usd = 0.0

            # Calculate gross profit
            entry_usd = amount_eth * eth_price
            gross_profit_pct = profit_pct

            # Deduct realistic costs for net profit. Slippage is taken as a
            # fixed percent of notional (not of spread) so larger trades
            # bear proportionally more slippage cost.
            flash_loan_cost = entry_usd * flash_loan_fee_pct
            slippage_cost = entry_usd * slippage_estimate_pct
            total_costs = flash_loan_cost + slippage_cost + gas_cost_usd

            net_profit_usd = (entry_usd * gross_profit_pct) - total_costs
            net_profit_pct = net_profit_usd / entry_usd if entry_usd > 0 else 0
            exit_usd = entry_usd + net_profit_usd

            # Use ETH price as entry, calculate exit based on net profit
            entry_price = eth_price
            exit_price = eth_price * (1 + net_profit_pct)

            self.logger.info(
                f"💰 [{self.chain_name.upper()}] Arb value [{token_symbol}]: {amount_eth:.4f} ETH @ ${eth_price:.2f} = ${entry_usd:.2f} | "
                f"Gross: +{gross_profit_pct:.2%} | Costs: ${total_costs:.2f} "
                f"(gas ${gas_cost_usd:.2f} + slip ${slippage_cost:.2f} + fee ${flash_loan_cost:.2f}) | "
                f"Net: ${net_profit_usd:.2f}"
            )

            trade_id = f"arb_{uuid.uuid4().hex[:12]}"

            # Honest simulation flag: rows written via the skip path (tx_hash
            # 'DRY_RUN' — killswitch/pause/shadow/live-flag gate) are simulated
            # even when dry_run=false. Previously they were stamped
            # is_simulated=false, poisoning live PnL and the realized-slippage
            # learner.
            is_sim = bool(self.dry_run) or tx_hash == "DRY_RUN"

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
                    is_sim,
                    datetime.now(),
                    datetime.now(),
                    tx_hash,
                    eth_price,
                    json.dumps({
                        'dry_run': self.dry_run,
                        'chain': self.chain_name,
                        'chain_id': self.chain_id,
                        'token_symbol': token_symbol,
                        'pair_symbol': pair_symbol,
                        'gross_profit_pct': gross_profit_pct * 100,
                        'flash_loan_cost': flash_loan_cost,
                        'slippage_cost': slippage_cost,
                        'slippage_pct': slippage_estimate_pct,
                        'slippage_source': slippage_source,
                        'gas_cost': gas_cost_usd,
                        'total_costs': total_costs,
                        'eth_price_fallback': _eth_price_fallback,
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
                        is_simulated=is_sim,
                        gas_cost_usd=gas_cost_usd,
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
