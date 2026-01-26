"""
System-wide Constants for ClaudeDex Trading Bot
Centralized configuration for chains, DEXes, tokens, and trading parameters
"""

from decimal import Decimal
from enum import Enum, IntEnum
from typing import Dict, List, Set

# ============= Version Info =============
VERSION = "1.0.0"
BOT_NAME = "ClaudeDex"
PROJECT_NAME = "ClaudeDex Trading Bot"

# ============= Chain Configuration =============

class Chain(IntEnum):
    """Blockchain chain IDs"""
    ETHEREUM = 1
    BSC = 56
    POLYGON = 137
    ARBITRUM = 42161
    BASE = 8453
    OPTIMISM = 10
    AVALANCHE = 43114
    # New chains (added 2025)
    FANTOM = 250
    CRONOS = 25
    PULSECHAIN = 369
    MONAD = 41454  # Monad mainnet

CHAIN_NAMES = {
    Chain.ETHEREUM: "Ethereum",
    Chain.BSC: "BSC",
    Chain.POLYGON: "Polygon",
    Chain.ARBITRUM: "Arbitrum",
    Chain.BASE: "Base",
    Chain.OPTIMISM: "Optimism",
    Chain.AVALANCHE: "Avalanche",
    Chain.FANTOM: "Fantom",
    Chain.CRONOS: "Cronos",
    Chain.PULSECHAIN: "PulseChain",
    Chain.MONAD: "Monad"
}

CHAIN_SYMBOLS = {
    Chain.ETHEREUM: "ETH",
    Chain.BSC: "BNB",
    Chain.POLYGON: "MATIC",
    Chain.ARBITRUM: "ETH",
    Chain.BASE: "ETH",
    Chain.OPTIMISM: "ETH",
    Chain.AVALANCHE: "AVAX",
    Chain.FANTOM: "FTM",
    Chain.CRONOS: "CRO",
    Chain.PULSECHAIN: "PLS",
    Chain.MONAD: "MONAD"
}

# Static fallback RPC URLs (used when Pool Engine is not available)
# Using free public RPCs that don't require API keys
_STATIC_CHAIN_RPC_URLS = {
    Chain.ETHEREUM: [
        "https://eth.llamarpc.com",
        "https://ethereum.publicnode.com",
        "https://1rpc.io/eth",
        "https://rpc.mevblocker.io",
        "https://eth.drpc.org"
    ],
    Chain.BSC: [
        "https://bsc-dataseed1.binance.org",
        "https://bsc-dataseed2.binance.org",
        "https://bsc-dataseed3.binance.org",
        "https://bsc.publicnode.com",
        "https://bsc.drpc.org"
    ],
    Chain.POLYGON: [
        "https://polygon-rpc.com",
        "https://polygon.llamarpc.com",
        "https://polygon.publicnode.com",
        "https://1rpc.io/matic",
        "https://polygon.drpc.org"
    ],
    Chain.ARBITRUM: [
        "https://arb1.arbitrum.io/rpc",
        "https://arbitrum-one.publicnode.com",
        "https://arbitrum.llamarpc.com",
        "https://1rpc.io/arb",
        "https://arbitrum.drpc.org"
    ],
    Chain.BASE: [
        "https://mainnet.base.org",
        "https://base.publicnode.com",
        "https://base.llamarpc.com",
        "https://1rpc.io/base",
        "https://base.drpc.org"
    ],
    # New chains (added 2025)
    Chain.AVALANCHE: [
        "https://api.avax.network/ext/bc/C/rpc",
        "https://avalanche.publicnode.com",
        "https://avalanche.drpc.org",
        "https://1rpc.io/avax/c"
    ],
    Chain.FANTOM: [
        "https://rpc.fantom.network",
        "https://fantom.publicnode.com",
        "https://rpc.ftm.tools",
        "https://fantom-mainnet.public.blastapi.io",
        "https://fantom.drpc.org"
    ],
    Chain.CRONOS: [
        "https://evm.cronos.org",
        "https://cronos-evm.publicnode.com",
        "https://cronos.drpc.org"
    ],
    Chain.PULSECHAIN: [
        "https://rpc.pulsechain.com",
        "https://pulsechain.publicnode.com",
        "https://rpc-pulsechain.g4mm4.io"
    ],
    Chain.MONAD: [
        "https://monad.publicnode.com",
        "https://rpc.monad.xyz"
    ]
}

# Backwards compatibility alias
CHAIN_RPC_URLS = _STATIC_CHAIN_RPC_URLS

# Chain to provider type mapping
_CHAIN_TO_PROVIDER = {
    Chain.ETHEREUM: 'ETHEREUM_RPC',
    Chain.BSC: 'BSC_RPC',
    Chain.POLYGON: 'POLYGON_RPC',
    Chain.ARBITRUM: 'ARBITRUM_RPC',
    Chain.BASE: 'BASE_RPC',
    Chain.AVALANCHE: 'AVALANCHE_RPC',
    Chain.FANTOM: 'FANTOM_RPC',
    Chain.CRONOS: 'CRONOS_RPC',
    Chain.PULSECHAIN: 'PULSECHAIN_RPC',
    Chain.MONAD: 'MONAD_RPC',
}


def get_chain_rpc_urls(chain: Chain, max_count: int = 5) -> list:
    """
    Get RPC URLs for a chain from the Pool Engine

    This function should be preferred over direct CHAIN_RPC_URLS access
    as it provides intelligent load balancing and rate limit handling.

    Args:
        chain: The blockchain chain
        max_count: Maximum number of URLs to return

    Returns:
        list: List of RPC URLs

    Example:
        from utils.constants import Chain, get_chain_rpc_urls
        eth_rpcs = get_chain_rpc_urls(Chain.ETHEREUM)
    """
    try:
        from config.rpc_provider import RPCProvider
        provider_type = _CHAIN_TO_PROVIDER.get(chain)
        if provider_type:
            urls = RPCProvider.get_rpcs_sync(provider_type, max_count)
            if urls:
                return urls
    except Exception:
        pass

    # Fallback to static URLs
    return _STATIC_CHAIN_RPC_URLS.get(chain, [])[:max_count]


async def get_chain_rpc_urls_async(chain: Chain, max_count: int = 5) -> list:
    """
    Get RPC URLs for a chain from the Pool Engine (async version)

    Args:
        chain: The blockchain chain
        max_count: Maximum number of URLs to return

    Returns:
        list: List of RPC URLs
    """
    try:
        from config.rpc_provider import RPCProvider
        provider_type = _CHAIN_TO_PROVIDER.get(chain)
        if provider_type:
            urls = await RPCProvider.get_rpcs(provider_type, max_count)
            if urls:
                return urls
    except Exception:
        pass

    # Fallback to static URLs
    return _STATIC_CHAIN_RPC_URLS.get(chain, [])[:max_count]


def get_chain_rpc_url(chain: Chain) -> str:
    """
    Get a single RPC URL for a chain

    Args:
        chain: The blockchain chain

    Returns:
        str: Best available RPC URL
    """
    urls = get_chain_rpc_urls(chain, max_count=1)
    return urls[0] if urls else None


async def get_chain_rpc_url_async(chain: Chain) -> str:
    """
    Get a single RPC URL for a chain (async version)

    Args:
        chain: The blockchain chain

    Returns:
        str: Best available RPC URL
    """
    urls = await get_chain_rpc_urls_async(chain, max_count=1)
    return urls[0] if urls else None

BLOCK_EXPLORERS = {
    Chain.ETHEREUM: "https://etherscan.io",
    Chain.BSC: "https://bscscan.com",
    Chain.POLYGON: "https://polygonscan.com",
    Chain.ARBITRUM: "https://arbiscan.io",
    Chain.BASE: "https://basescan.org",
    Chain.AVALANCHE: "https://snowtrace.io",
    Chain.FANTOM: "https://ftmscan.com",
    Chain.CRONOS: "https://cronoscan.com",
    Chain.PULSECHAIN: "https://scan.pulsechain.com",
    Chain.MONAD: "https://explorer.monad.xyz"
}

# ============= DEX Configuration =============

class DEX(Enum):
    """Supported decentralized exchanges"""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    PANCAKESWAP = "pancakeswap"
    SUSHISWAP = "sushiswap"
    QUICKSWAP = "quickswap"
    TRADER_JOE = "trader_joe"
    TOXISOL = "toxisol"

DEX_ROUTERS = {
    DEX.UNISWAP_V2: {
        Chain.ETHEREUM: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    },
    DEX.UNISWAP_V3: {
        Chain.ETHEREUM: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        Chain.POLYGON: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        Chain.ARBITRUM: "0xE592427A0AEce92De3Edee1F18E0157C05861564"
    },
    DEX.PANCAKESWAP: {
        Chain.BSC: "0x10ED43C718714eb63d5aA57B78B54704E256024E"
    },
    DEX.SUSHISWAP: {
        Chain.ETHEREUM: "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
        Chain.POLYGON: "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506"
    }
}

DEX_FEES = {
    DEX.UNISWAP_V2: Decimal("0.003"),  # 0.3%
    DEX.UNISWAP_V3: Decimal("0.003"),  # Variable, using 0.3% as default
    DEX.PANCAKESWAP: Decimal("0.0025"),  # 0.25%
    DEX.SUSHISWAP: Decimal("0.003"),  # 0.3%
    DEX.QUICKSWAP: Decimal("0.003"),  # 0.3%
}

# ============= Token Standards =============

TOKEN_STANDARDS = {
    "ERC20": Chain.ETHEREUM,
    "BEP20": Chain.BSC,
    "ERC721": Chain.ETHEREUM,
    "ERC1155": Chain.ETHEREUM
}


SAFE_SOLANA_TOKENS = {
    # Major established tokens
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": "JUP",
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R": "RAY (Raydium)",
    "So11111111111111111111111111111111111111112": "SOL (Wrapped)",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL (Marinade)",
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": "stSOL (Lido)",
}

# Stablecoin addresses by chain
STABLECOINS = {
    Chain.ETHEREUM: {
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "BUSD": "0x4Fabb145d64652a948d72533023f6E7A623C7C53"
    },
    Chain.BSC: {
        "USDT": "0x55d398326f99059fF775485246999027B3197955",
        "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
        "BUSD": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
        "DAI": "0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3"
    },
    Chain.POLYGON: {
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
        "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "DAI": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"
    }
}

WRAPPED_NATIVE_TOKENS = {
    Chain.ETHEREUM: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    Chain.BSC: "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
    Chain.POLYGON: "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
    Chain.ARBITRUM: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH
    Chain.BASE: "0x4200000000000000000000000000000000000006"  # WETH
}

# ============= Trading Parameters =============

class TradingMode(Enum):
    """Trading modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

# Risk Management Defaults
RISK_PARAMETERS = {
    TradingMode.CONSERVATIVE: {
        "max_position_size": Decimal("0.02"),  # 2% of portfolio
        "stop_loss": Decimal("0.03"),  # 3%
        "take_profit": Decimal("0.05"),  # 5%
        "max_slippage": Decimal("0.01"),  # 1%
        "max_gas_price": 50,  # gwei
        "min_liquidity": 100000,  # USD
        "max_positions": 3
    },
    TradingMode.BALANCED: {
        "max_position_size": Decimal("0.05"),  # 5% of portfolio
        "stop_loss": Decimal("0.05"),  # 5%
        "take_profit": Decimal("0.10"),  # 10%
        "max_slippage": Decimal("0.02"),  # 2%
        "max_gas_price": 100,  # gwei
        "min_liquidity": 50000,  # USD
        "max_positions": 5
    },
    TradingMode.AGGRESSIVE: {
        "max_position_size": Decimal("0.10"),  # 10% of portfolio
        "stop_loss": Decimal("0.10"),  # 10%
        "take_profit": Decimal("0.25"),  # 25%
        "max_slippage": Decimal("0.03"),  # 3%
        "max_gas_price": 200,  # gwei
        "min_liquidity": 25000,  # USD
        "max_positions": 10
    }
}

# Position Limits
MAX_POSITION_SIZE_PERCENT = Decimal("0.10")  # 10% max per position
MAX_PORTFOLIO_EXPOSURE = Decimal("0.80")  # 80% max total exposure
MIN_POSITION_SIZE_USD = Decimal("50")  # Minimum $50 per trade
MAX_POSITION_SIZE_USD = Decimal("100000")  # Maximum $100k per trade

# Stop Loss & Take Profit
DEFAULT_STOP_LOSS = Decimal("0.05")  # 5%
MAX_STOP_LOSS = Decimal("0.20")  # 20%
DEFAULT_TAKE_PROFIT = Decimal("0.10")  # 10%
TRAILING_STOP_ACTIVATION = Decimal("0.05")  # Activate at 5% profit

# Slippage & Gas
DEFAULT_SLIPPAGE = Decimal("0.02")  # 2%
MAX_SLIPPAGE = Decimal("0.05")  # 5%
DEFAULT_GAS_LIMIT = 500000
MAX_GAS_PRICE_GWEI = 50  # FIXED: Was 500 (=$150-300/tx), now 50 (=$15-30/tx)
PRIORITY_FEE_GWEI = 2

# ============= ML Model Parameters =============

ML_CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for ML predictions
ML_ENSEMBLE_WEIGHTS = {
    "xgboost": 0.3,
    "lightgbm": 0.25,
    "random_forest": 0.2,
    "lstm": 0.15,
    "catboost": 0.1
}

ML_FEATURE_IMPORTANCE_THRESHOLD = 0.05  # Minimum feature importance

# ============= Honeypot & Security =============

HONEYPOT_CHECKS = {
    "honeypot_is": "https://honeypot.is/api",
    "tokensniffer": "https://tokensniffer.com/api",
    "goplus": "https://api.gopluslabs.io/api/v1"
}

HONEYPOT_THRESHOLDS = {
    "max_sell_tax": Decimal("0.10"),  # 10% max sell tax
    "max_buy_tax": Decimal("0.10"),  # 10% max buy tax
    "min_liquidity": 10000,  # $10k minimum liquidity
    "min_holders": 50,  # Minimum 50 holders
    "max_ownership": Decimal("0.10"),  # Max 10% single wallet ownership
}

BLACKLISTED_TOKENS: Set[str] = set()  # Dynamically populated
BLACKLISTED_CONTRACTS: Set[str] = set()  # Known scam contracts
BLACKLISTED_WALLETS: Set[str] = set()  # Known scammer wallets

# ============= API Rate Limits =============

API_RATE_LIMITS = {
    "dexscreener": {"calls": 300, "period": 60},  # 300 calls/minute
    "coingecko": {"calls": 50, "period": 60},  # 50 calls/minute
    "etherscan": {"calls": 5, "period": 1},  # 5 calls/second
    "bscscan": {"calls": 5, "period": 1},  # 5 calls/second
    "alchemy": {"calls": 100, "period": 1},  # 100 calls/second
    "infura": {"calls": 100, "period": 1},  # 100 calls/second
}

# ============= Time Constants =============

CACHE_TTL = {
    "token_info": 300,  # 5 minutes
    "price_data": 10,  # 10 seconds
    "liquidity_data": 60,  # 1 minute
    "whale_movements": 300,  # 5 minutes
    "gas_prices": 15,  # 15 seconds
}

TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800
}

# ============= Performance Metrics =============

PERFORMANCE_THRESHOLDS = {
    "min_sharpe_ratio": 1.0,
    "min_sortino_ratio": 1.5,
    "min_calmar_ratio": 1.0,
    "max_drawdown": Decimal("0.20"),  # 20%
    "min_win_rate": Decimal("0.40"),  # 40%
    "min_profit_factor": 1.2
}

# ============= Database Configuration =============

DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 40
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600

# Redis Configuration
REDIS_MAX_CONNECTIONS = 50
REDIS_DECODE_RESPONSES = True
REDIS_SOCKET_TIMEOUT = 5
REDIS_CONNECTION_TIMEOUT = 10

# ============= Monitoring & Alerts =============

ALERT_CHANNELS = ["telegram", "discord", "email", "webhook"]

ALERT_THRESHOLDS = {
    "portfolio_drawdown": Decimal("0.10"),  # 10% drawdown alert
    "position_loss": Decimal("0.05"),  # 5% position loss alert
    "gas_spike": 300,  # 300 gwei gas price alert
    "low_balance": Decimal("0.1"),  # 0.1 ETH low balance alert
    "high_slippage": Decimal("0.05"),  # 5% slippage alert
}

METRICS_COLLECTION_INTERVAL = 60  # seconds
HEALTH_CHECK_INTERVAL = 30  # seconds
PERFORMANCE_REPORT_INTERVAL = 3600  # 1 hour

# ============= Logging Configuration =============

LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# ============= Trading Signals =============

class SignalStrength(Enum):
    """Trading signal strength levels"""
    VERY_WEAK = 1
    WEAK = 2
    NEUTRAL = 3
    STRONG = 4
    VERY_STRONG = 5

SIGNAL_THRESHOLDS = {
    SignalStrength.VERY_WEAK: 0.2,
    SignalStrength.WEAK: 0.4,
    SignalStrength.NEUTRAL: 0.6,
    SignalStrength.STRONG: 0.8,
    SignalStrength.VERY_STRONG: 0.9
}

# ============= Pattern Detection =============

CHART_PATTERNS = [
    "head_and_shoulders",
    "double_top",
    "double_bottom",
    "ascending_triangle",
    "descending_triangle",
    "flag",
    "pennant",
    "wedge",
    "cup_and_handle"
]

CANDLESTICK_PATTERNS = [
    "doji",
    "hammer",
    "shooting_star",
    "engulfing",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows"
]

# ============= Market Conditions =============

class MarketCondition(Enum):
    """Market condition states"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

MARKET_INDICATORS = {
    "fear_greed_index": {
        "extreme_fear": (0, 20),
        "fear": (21, 40),
        "neutral": (41, 60),
        "greed": (61, 80),
        "extreme_greed": (81, 100)
    }
}

# ============= Order Types =============

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price

class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"

# ============= Error Messages =============

ERROR_MESSAGES = {
    "INSUFFICIENT_BALANCE": "Insufficient balance for trade",
    "HONEYPOT_DETECTED": "Token detected as potential honeypot",
    "HIGH_SLIPPAGE": "Slippage exceeds maximum allowed",
    "LOW_LIQUIDITY": "Insufficient liquidity for safe trading",
    "GAS_TOO_HIGH": "Gas price exceeds maximum limit",
    "POSITION_LIMIT": "Maximum position limit reached",
    "RISK_LIMIT": "Trade exceeds risk parameters",
    "NETWORK_ERROR": "Network connection error",
    "API_ERROR": "External API error",
    "CONTRACT_ERROR": "Smart contract interaction failed"
}

# ============= Success Messages =============

SUCCESS_MESSAGES = {
    "TRADE_EXECUTED": "Trade executed successfully",
    "POSITION_CLOSED": "Position closed successfully",
    "STOP_LOSS_SET": "Stop loss order placed",
    "TAKE_PROFIT_SET": "Take profit order placed",
    "WALLET_CONNECTED": "Wallet connected successfully",
    "BOT_STARTED": "Trading bot started successfully",
    "BOT_STOPPED": "Trading bot stopped safely"
}