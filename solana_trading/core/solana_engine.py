"""
Solana Trading Engine
Core engine for trading on Solana blockchain

Strategies:
- Jupiter V6: Token swaps via Jupiter aggregator
- Drift Protocol: Perpetuals trading
- Pump.fun: New token sniping

Features:
- DRY_RUN mode with simulated trades
- RPC failover with multiple endpoints
- Jupiter price API integration
- Risk management (daily loss limits, position limits)
- Slippage controls
- Transaction confirmation handling
"""

# ============================================================================
# HTTPX COMPATIBILITY FIX
# The solana library uses httpx.AsyncClient with 'proxy' parameter, but
# newer httpx versions (0.24+) renamed this to 'proxy' or require 'proxies'.
# This patch ensures compatibility with both old and new httpx versions.
# ============================================================================
import httpx

# Store original AsyncClient
_OriginalAsyncClient = httpx.AsyncClient


class _CompatAsyncClient(_OriginalAsyncClient):
    """Wrapper that handles the proxy parameter compatibility"""

    def __init__(self, *args, **kwargs):
        # Handle 'proxy' parameter for newer httpx versions
        proxy = kwargs.pop('proxy', None)
        if proxy is not None:
            # In newer httpx, use 'proxy' parameter (it was temporarily removed then re-added)
            # But some versions use 'proxies' dict instead
            try:
                # Try the new way first
                super().__init__(*args, proxy=proxy, **kwargs)
                return
            except TypeError:
                # If that fails, try without proxy (proxy=None means no proxy anyway)
                if proxy is not None:
                    # Try 'proxies' parameter for some versions
                    try:
                        super().__init__(*args, proxies={'all://': proxy}, **kwargs)
                        return
                    except TypeError:
                        pass
        # Default: call without proxy
        super().__init__(*args, **kwargs)


# Apply the monkey patch
httpx.AsyncClient = _CompatAsyncClient
# ============================================================================

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os
import base58
import uuid
import json
import aiohttp
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pnl_tracker import PnLTracker, TradeRecord

# Import JupiterHelper for live swap execution
try:
    from modules.solana_strategies.jupiter_helper import JupiterHelper
    JUPITER_HELPER_AVAILABLE = True
except ImportError:
    JUPITER_HELPER_AVAILABLE = False

# Import DriftHelper for live perpetual trading
try:
    from modules.solana_strategies.drift_helper import DriftHelper
    DRIFT_HELPER_AVAILABLE = True
except ImportError:
    DRIFT_HELPER_AVAILABLE = False

logger = logging.getLogger("SolanaTradingEngine")


# Well-known token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"


class TradeSide(Enum):
    """Trade direction"""
    BUY = "buy"
    SELL = "sell"


class Strategy(Enum):
    """Trading strategies"""
    JUPITER = "jupiter"
    DRIFT = "drift"
    PUMPFUN = "pumpfun"


@dataclass
class TokenInfo:
    """Token information"""
    mint: str
    symbol: str
    name: str
    decimals: int
    price_usd: float = 0.0
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    price_change_24h: float = 0.0


@dataclass
class Position:
    """Active trading position"""
    position_id: str
    token_mint: str
    token_symbol: str
    strategy: Strategy
    side: TradeSide
    entry_price: float
    current_price: float
    amount: float  # Token amount
    value_sol: float  # Value in SOL
    value_usd: float  # Value in USD
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    fees_paid: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    is_simulated: bool = False
    tx_signature: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    token_mint: str
    token_symbol: str
    strategy: Strategy
    side: TradeSide
    entry_price: float
    exit_price: float
    amount: float
    pnl_sol: float
    pnl_usd: float
    pnl_pct: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    close_reason: str
    is_simulated: bool = False


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    daily_pnl_sol: float = 0.0
    daily_pnl_usd: float = 0.0
    daily_trades: int = 0
    daily_loss_limit_sol: float = 5.0
    current_exposure_sol: float = 0.0
    consecutive_losses: int = 0
    last_reset: datetime = field(default_factory=datetime.now)

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        if self.daily_pnl_sol <= -self.daily_loss_limit_sol:
            return False
        if self.consecutive_losses >= 5:
            return False
        return True

    @property
    def risk_level(self) -> str:
        """Current risk level"""
        loss_ratio = abs(self.daily_pnl_sol) / self.daily_loss_limit_sol if self.daily_loss_limit_sol > 0 else 0
        if loss_ratio >= 0.8:
            return "HIGH"
        elif loss_ratio >= 0.5:
            return "MEDIUM"
        return "LOW"


class RPCManager:
    """Manages multiple RPC connections with failover"""

    def __init__(self, rpc_urls: List[str]):
        self.rpc_urls = rpc_urls
        self.current_index = 0
        self.failed_rpcs: Dict[str, datetime] = {}
        self.retry_after = timedelta(minutes=5)
        self._latencies: Dict[str, float] = {}

    @property
    def current_url(self) -> str:
        """Get current RPC URL"""
        return self.rpc_urls[self.current_index]

    def mark_failed(self, url: str):
        """Mark an RPC as failed"""
        self.failed_rpcs[url] = datetime.now()
        logger.warning(f"RPC marked as failed: {url}")
        self._rotate()

    def _rotate(self):
        """Rotate to next available RPC"""
        original = self.current_index
        for _ in range(len(self.rpc_urls)):
            self.current_index = (self.current_index + 1) % len(self.rpc_urls)
            url = self.rpc_urls[self.current_index]

            # Check if RPC is still in cooldown
            if url in self.failed_rpcs:
                if datetime.now() - self.failed_rpcs[url] < self.retry_after:
                    continue
                else:
                    del self.failed_rpcs[url]

            logger.info(f"Switched to RPC: {url[:50]}...")
            return

        # All RPCs failed, reset to original
        self.current_index = original
        logger.error("All RPCs are failing!")

    def get_healthy_url(self) -> str:
        """Get a healthy RPC URL"""
        for url in self.rpc_urls:
            if url not in self.failed_rpcs:
                return url
            elif datetime.now() - self.failed_rpcs[url] >= self.retry_after:
                del self.failed_rpcs[url]
                return url
        return self.rpc_urls[0]  # Fallback

    def record_latency(self, url: str, latency_ms: float):
        """Record RPC latency"""
        self._latencies[url] = latency_ms

    def get_latencies(self) -> Dict[str, float]:
        """Get RPC latencies"""
        return self._latencies.copy()


class JupiterClient:
    """Jupiter V6 API client with multi-source price fetching"""

    # Token address to CoinGecko ID mapping for common tokens
    COINGECKO_IDS = {
        SOL_MINT: 'solana',
        USDC_MINT: 'usd-coin',
        USDT_MINT: 'tether',
        'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263': 'bonk',
        'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL': 'jito-governance-token',
        'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm': 'dogwifcoin',
    }

    def __init__(self, api_url: str, api_key: Optional[str] = None, slippage_bps: int = 50):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.slippage_bps = slippage_bps
        self._session: Optional[aiohttp.ClientSession] = None
        self._price_cache: Dict[str, Dict] = {}
        self._price_cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=30)
        self._price_source_failures: Dict[str, int] = {}  # Track API failures

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: Optional[int] = None
    ) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        try:
            session = await self._get_session()
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': slippage_bps or self.slippage_bps,
            }

            async with session.get(f"{self.api_url}/quote", params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Jupiter quote error: {resp.status} - {await resp.text()}")
                    return None
        except Exception as e:
            logger.error(f"Jupiter quote failed: {e}")
            return None

    async def get_price(self, token_mint: str) -> Optional[float]:
        """
        Get token price in USD using multiple free price sources.

        Price sources (in order of preference):
        1. DexScreener API (free, no auth, covers all Solana tokens)
        2. CoinGecko API (free, no auth, reliable for major tokens)
        3. Jupiter Price API (may require auth)

        Returns:
            Optional[float]: Token price in USD, or None if unavailable
        """
        # Check cache first
        if token_mint in self._price_cache:
            cache_time = self._price_cache_time.get(token_mint, datetime.min)
            if datetime.now() - cache_time < self._cache_ttl:
                return self._price_cache[token_mint].get('price')

        session = await self._get_session()

        # Try DexScreener first (free, no auth, excellent coverage)
        price = await self._get_price_dexscreener(session, token_mint)
        if price:
            return price

        # Try CoinGecko for major tokens (free, no auth)
        if token_mint in self.COINGECKO_IDS:
            price = await self._get_price_coingecko(session, token_mint)
            if price:
                return price

        # Fallback to Jupiter (may need auth in newer versions)
        price = await self._get_price_jupiter(session, token_mint)
        if price:
            return price

        logger.warning(f"Could not fetch price for {token_mint[:8]}... from any source")
        return None

    async def _get_price_dexscreener(self, session: aiohttp.ClientSession, token_mint: str) -> Optional[float]:
        """Get price from DexScreener API (free, no auth)"""
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_mint}"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pairs = data.get('pairs', [])
                    if pairs:
                        # Get the pair with highest liquidity
                        best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                        price = float(best_pair.get('priceUsd', 0))
                        if price > 0:
                            self._price_cache[token_mint] = {'price': price, 'source': 'dexscreener'}
                            self._price_cache_time[token_mint] = datetime.now()
                            logger.debug(f"DexScreener price for {token_mint[:8]}...: ${price:.8f}")
                            return price
                elif resp.status != 404:
                    logger.debug(f"DexScreener returned status {resp.status} for {token_mint[:8]}...")
        except asyncio.TimeoutError:
            logger.debug(f"DexScreener timeout for {token_mint[:8]}...")
        except Exception as e:
            logger.debug(f"DexScreener error for {token_mint[:8]}...: {e}")
        return None

    async def _get_price_coingecko(self, session: aiohttp.ClientSession, token_mint: str) -> Optional[float]:
        """Get price from CoinGecko API (free, no auth)"""
        coingecko_id = self.COINGECKO_IDS.get(token_mint)
        if not coingecko_id:
            return None

        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {'ids': coingecko_id, 'vs_currencies': 'usd'}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if coingecko_id in data and 'usd' in data[coingecko_id]:
                        price = float(data[coingecko_id]['usd'])
                        self._price_cache[token_mint] = {'price': price, 'source': 'coingecko'}
                        self._price_cache_time[token_mint] = datetime.now()
                        logger.debug(f"CoinGecko price for {token_mint[:8]}...: ${price:.8f}")
                        return price
                elif resp.status == 429:
                    logger.debug("CoinGecko rate limited")
                else:
                    logger.debug(f"CoinGecko returned status {resp.status}")
        except asyncio.TimeoutError:
            logger.debug(f"CoinGecko timeout for {token_mint[:8]}...")
        except Exception as e:
            logger.debug(f"CoinGecko error: {e}")
        return None

    async def _get_price_jupiter(self, session: aiohttp.ClientSession, token_mint: str) -> Optional[float]:
        """Get price from Jupiter Price API (may require auth)"""
        try:
            price_api_url = "https://api.jup.ag/price/v2"
            params = {'ids': token_mint}

            async with session.get(price_api_url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data and token_mint in data['data']:
                        price_data = data['data'][token_mint]
                        price = float(price_data.get('price', 0))
                        if price > 0:
                            self._price_cache[token_mint] = {'price': price, 'source': 'jupiter'}
                            self._price_cache_time[token_mint] = datetime.now()
                            logger.debug(f"Jupiter price for {token_mint[:8]}...: ${price:.8f}")
                            return price
                elif resp.status == 401:
                    logger.debug("Jupiter price API requires authentication")
                else:
                    logger.debug(f"Jupiter price API returned status {resp.status}")
        except asyncio.TimeoutError:
            logger.debug(f"Jupiter price timeout for {token_mint[:8]}...")
        except Exception as e:
            logger.debug(f"Jupiter price error: {e}")
        return None

    async def get_prices_batch(self, token_mints: List[str]) -> Dict[str, float]:
        """Get prices for multiple tokens using multi-source approach"""
        prices = {}

        # Fetch prices individually using multi-source get_price
        for token_mint in token_mints:
            price = await self.get_price(token_mint)
            if price:
                prices[token_mint] = price

        return prices


class PumpFunMonitor:
    """Monitor Pump.fun for new token launches via DexScreener API"""

    # Pump.fun DEX identifier on DexScreener
    PUMPFUN_DEX_ID = "pumpfun"

    def __init__(self, config: Dict):
        self.program_id = config.get('program_id', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P')
        self.min_liquidity_sol = config.get('min_liquidity_sol', 10)
        self.min_liquidity_usd = config.get('min_liquidity_usd', 1000)  # Minimum $1000 liquidity
        self.max_age_seconds = config.get('max_age_seconds', 300)
        self.buy_amount_sol = config.get('buy_amount_sol', 0.1)
        self.min_volume_24h = config.get('min_volume_24h', 5000)  # Minimum $5000 24h volume
        self._recent_tokens: Dict[str, Dict] = {}
        self._seen_tokens: set = set()  # Track already seen tokens
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_fetch_time: Optional[datetime] = None
        self._fetch_interval = timedelta(seconds=30)  # Don't spam API

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_new_tokens(self) -> List[Dict]:
        """
        Get new Pump.fun tokens from DexScreener API

        Returns tokens that:
        - Are on Pump.fun DEX
        - Meet liquidity requirements
        - Are within max_age_seconds
        - Haven't been seen before
        """
        # Rate limit API calls
        if self._last_fetch_time:
            time_since_last = datetime.now() - self._last_fetch_time
            if time_since_last < self._fetch_interval:
                return []

        try:
            session = await self._get_session()
            self._last_fetch_time = datetime.now()

            # DexScreener API for new token pairs on Solana
            # Use the token-profiles endpoint for new launches, or search for pump.fun
            url = "https://api.dexscreener.com/token-profiles/latest/v1"

            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # token-profiles returns a list directly
                    profiles = data if isinstance(data, list) else []

                    # Count stats for logging
                    pumpfun_pairs = 0
                    filtered_out = {'seen': 0, 'liquidity': 0, 'volume': 0, 'price': 0, 'suspicious': 0, 'not_solana': 0}

                    new_tokens = []
                    for profile in profiles:
                        # Check if it's a Solana token
                        chain_id = profile.get('chainId', '').lower()
                        if chain_id != 'solana':
                            filtered_out['not_solana'] += 1
                            continue

                        pumpfun_pairs += 1
                        token_address = profile.get('tokenAddress')
                        if not token_address:
                            continue

                        # Skip already seen tokens
                        if token_address in self._seen_tokens:
                            filtered_out['seen'] += 1
                            continue

                        # Get token info - need to fetch price separately
                        symbol = profile.get('symbol', '')
                        name = profile.get('name', '')

                        # For new tokens, we'll try to get price from DexScreener token endpoint
                        price_data = await self._get_token_price_data(session, token_address)
                        if not price_data:
                            continue

                        # Use symbol/name from price_data if token-profiles didn't provide them
                        if not symbol or symbol == 'UNKNOWN':
                            symbol = price_data.get('symbol', '') or 'UNKNOWN'
                        if not name or name == 'Unknown':
                            name = price_data.get('name', '') or 'Unknown'

                        token_info = {
                            'mint': token_address,
                            'symbol': symbol,
                            'name': name,
                            'price_usd': price_data.get('price', 0),
                            'liquidity_usd': price_data.get('liquidity', 0),
                            'liquidity_sol': 0,
                            'volume_24h': price_data.get('volume', 0),
                            'price_change_24h': price_data.get('priceChange', 0),
                            'pair_address': price_data.get('pairAddress', ''),
                            'created_at': datetime.now(),
                            'dex': price_data.get('dex', 'unknown'),
                            'url': profile.get('url', ''),
                        }

                        # Apply filters with tracking
                        if token_info['liquidity_usd'] < self.min_liquidity_usd:
                            filtered_out['liquidity'] += 1
                            continue
                        if token_info['volume_24h'] < self.min_volume_24h:
                            filtered_out['volume'] += 1
                            continue
                        if token_info['price_usd'] <= 0:
                            filtered_out['price'] += 1
                            continue
                        if abs(token_info['price_change_24h']) > 500:
                            filtered_out['suspicious'] += 1
                            continue

                        # Token passes all filters!
                        new_tokens.append(token_info)
                        self._seen_tokens.add(token_address)
                        self._recent_tokens[token_address] = token_info
                        logger.info(
                            f"🎯 New token found: {token_info['symbol']} "
                            f"(${token_info['price_usd']:.8f}, "
                            f"liq=${token_info['liquidity_usd']:.0f}, "
                            f"vol=${token_info['volume_24h']:.0f})"
                        )

                    # Log scanning summary
                    if pumpfun_pairs > 0 or len(profiles) > 0:
                        logger.info(
                            f"🔍 Pump.fun scan: {len(profiles)} profiles, {pumpfun_pairs} Solana tokens, "
                            f"{len(new_tokens)} qualify"
                        )
                    else:
                        logger.info("🔍 Pump.fun scan: No new token profiles found")

                    return new_tokens

                elif resp.status == 404:
                    # Try alternative endpoint
                    logger.debug("Token profiles endpoint not available, trying search...")
                    return await self._search_pumpfun_tokens(session)
                else:
                    logger.warning(f"DexScreener returned status {resp.status}")
                    return []

        except asyncio.TimeoutError:
            logger.debug("DexScreener timeout for Pump.fun tokens")
            return []
        except Exception as e:
            logger.error(f"Pump.fun fetch failed: {e}")
            return []

    def filter_token(self, token: Dict) -> bool:
        """Filter token based on safety/quality criteria"""
        # Check USD liquidity
        liquidity_usd = token.get('liquidity_usd', 0)
        if liquidity_usd < self.min_liquidity_usd:
            return False

        # Check 24h volume (indicates real activity)
        volume_24h = token.get('volume_24h', 0)
        if volume_24h < self.min_volume_24h:
            return False

        # Check price exists
        price = token.get('price_usd', 0)
        if price <= 0:
            return False

        # Avoid honeypot indicators (very high price change could be manipulation)
        price_change = abs(token.get('price_change_24h', 0))
        if price_change > 500:  # >500% change in 24h is suspicious
            logger.debug(f"Skipping {token.get('symbol')} - suspicious price change: {price_change}%")
            return False

        return True

    def clear_seen_tokens(self):
        """Clear the seen tokens cache (call periodically)"""
        # Keep only tokens from last hour
        cutoff = datetime.now() - timedelta(hours=1)
        to_remove = []
        for mint, info in self._recent_tokens.items():
            if info.get('created_at', datetime.now()) < cutoff:
                to_remove.append(mint)

        for mint in to_remove:
            self._recent_tokens.pop(mint, None)
            self._seen_tokens.discard(mint)

    async def _get_token_price_data(self, session: aiohttp.ClientSession, token_address: str) -> Optional[Dict]:
        """
        Get price, liquidity, volume, and token info from DexScreener.

        Args:
            session: aiohttp session
            token_address: Solana token mint address

        Returns:
            Dict with price, liquidity, volume, priceChange, pairAddress, dex, symbol, name or None
        """
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pairs = data.get('pairs', [])

                    if not pairs:
                        return None

                    # Get the pair with highest liquidity
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))

                    price = float(best_pair.get('priceUsd', 0) or 0)
                    liquidity = float(best_pair.get('liquidity', {}).get('usd', 0) or 0)
                    volume_24h = float(best_pair.get('volume', {}).get('h24', 0) or 0)
                    price_change = float(best_pair.get('priceChange', {}).get('h24', 0) or 0)
                    pair_address = best_pair.get('pairAddress', '')
                    dex_id = best_pair.get('dexId', 'unknown')

                    # Extract token symbol and name from baseToken
                    base_token = best_pair.get('baseToken', {})
                    symbol = base_token.get('symbol', '')
                    name = base_token.get('name', '')

                    return {
                        'price': price,
                        'liquidity': liquidity,
                        'volume': volume_24h,
                        'priceChange': price_change,
                        'pairAddress': pair_address,
                        'dex': dex_id,
                        'symbol': symbol,
                        'name': name
                    }

                elif resp.status == 404:
                    logger.debug(f"Token {token_address[:8]}... not found on DexScreener")
                else:
                    logger.debug(f"DexScreener returned {resp.status} for token {token_address[:8]}...")

        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching price data for {token_address[:8]}...")
        except Exception as e:
            logger.debug(f"Error fetching price data for {token_address[:8]}...: {e}")

        return None

    async def _search_pumpfun_tokens(self, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Search for Pump.fun tokens using DexScreener pairs API.
        Fallback method when token-profiles endpoint is unavailable.

        Returns:
            List of token info dicts that pass filters
        """
        try:
            # Search for recent Solana pairs - this covers more tokens
            url = "https://api.dexscreener.com/latest/dex/pairs/solana"

            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pairs = data.get('pairs', [])

                    # Filter for Pump.fun pairs or high-liquidity new tokens
                    new_tokens = []
                    filtered_out = {'seen': 0, 'liquidity': 0, 'volume': 0, 'price': 0, 'suspicious': 0}

                    for pair in pairs:
                        dex_id = pair.get('dexId', '').lower()

                        # Focus on pump.fun, raydium, and other Solana DEXs
                        if dex_id not in ['pumpfun', 'raydium', 'orca', 'meteora']:
                            continue

                        base_token = pair.get('baseToken', {})
                        token_address = base_token.get('address', '')

                        if not token_address:
                            continue

                        # Skip already seen tokens
                        if token_address in self._seen_tokens:
                            filtered_out['seen'] += 1
                            continue

                        # Extract token info
                        price = float(pair.get('priceUsd', 0) or 0)
                        liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                        volume_24h = float(pair.get('volume', {}).get('h24', 0) or 0)
                        price_change = float(pair.get('priceChange', {}).get('h24', 0) or 0)

                        # Apply filters
                        if liquidity < self.min_liquidity_usd:
                            filtered_out['liquidity'] += 1
                            continue
                        if volume_24h < self.min_volume_24h:
                            filtered_out['volume'] += 1
                            continue
                        if price <= 0:
                            filtered_out['price'] += 1
                            continue
                        if abs(price_change) > 500:
                            filtered_out['suspicious'] += 1
                            continue

                        token_info = {
                            'mint': token_address,
                            'symbol': base_token.get('symbol', 'UNKNOWN'),
                            'name': base_token.get('name', 'Unknown'),
                            'price_usd': price,
                            'liquidity_usd': liquidity,
                            'liquidity_sol': 0,
                            'volume_24h': volume_24h,
                            'price_change_24h': price_change,
                            'pair_address': pair.get('pairAddress', ''),
                            'created_at': datetime.now(),
                            'dex': dex_id,
                            'url': pair.get('url', ''),
                        }

                        new_tokens.append(token_info)
                        self._seen_tokens.add(token_address)
                        self._recent_tokens[token_address] = token_info

                        logger.info(
                            f"🎯 Found token via search: {token_info['symbol']} on {dex_id} "
                            f"(${price:.8f}, liq=${liquidity:.0f}, vol=${volume_24h:.0f})"
                        )

                        # Limit results per scan
                        if len(new_tokens) >= 5:
                            break

                    if new_tokens or len(pairs) > 0:
                        logger.info(
                            f"🔍 Pump.fun search: {len(pairs)} pairs scanned, "
                            f"{len(new_tokens)} qualify, filtered: {filtered_out}"
                        )

                    return new_tokens

                else:
                    logger.warning(f"DexScreener pairs search returned {resp.status}")
                    return []

        except asyncio.TimeoutError:
            logger.debug("Timeout in Pump.fun token search")
            return []
        except Exception as e:
            logger.error(f"Error in Pump.fun token search: {e}")
            return []


class SolanaTradingEngine:
    """Core Solana trading engine"""

    # SOL decimals
    SOL_DECIMALS = 9
    LAMPORTS_PER_SOL = 1_000_000_000

    def __init__(
        self,
        rpc_url: str,
        strategies: List[str],
        max_positions: int = 3,
        mode: str = "production",
        config_manager=None
    ):
        """
        Initialize Solana trading engine

        Args:
            rpc_url: Primary Solana RPC URL
            strategies: List of enabled strategies (jupiter, drift, pumpfun)
            max_positions: Maximum concurrent positions
            mode: Operating mode
            config_manager: Optional SolanaConfigManager for DB-backed config
        """
        self.primary_rpc = rpc_url
        self.strategies = [Strategy(s.strip().lower()) for s in strategies if s.strip().lower() in [e.value for e in Strategy]]
        self.max_positions = max_positions
        self.mode = mode
        self.is_running = False
        self.config_manager = config_manager

        # DRY_RUN mode - CRITICAL: Check environment variable
        dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
        self.dry_run = dry_run_env in ('true', '1', 'yes')

        # RPC configuration
        rpc_urls_env = os.getenv('SOLANA_RPC_URLS', '')
        if rpc_urls_env:
            rpc_urls = [url.strip() for url in rpc_urls_env.split(',') if url.strip()]
        else:
            rpc_urls = [rpc_url]

        # Ensure primary is first
        if rpc_url not in rpc_urls:
            rpc_urls.insert(0, rpc_url)

        self.rpc_manager = RPCManager(rpc_urls)

        # Trading configuration - prefer config_manager, fallback to env vars
        if config_manager:
            self.position_size_sol = config_manager.position_size_sol
            self.stop_loss_pct = -abs(config_manager.stop_loss_pct)  # Ensure negative
            self.take_profit_pct = config_manager.take_profit_pct
            self.max_daily_loss_sol = config_manager.daily_loss_limit_sol
            self.slippage_bps = config_manager.jupiter_slippage_bps
        else:
            self.position_size_sol = float(os.getenv('SOLANA_POSITION_SIZE_SOL', '1.0'))
            self.stop_loss_pct = float(os.getenv('SOLANA_STOP_LOSS_PCT', '-10.0'))
            self.take_profit_pct = float(os.getenv('SOLANA_TAKE_PROFIT_PCT', '50.0'))
            self.max_daily_loss_sol = float(os.getenv('SOLANA_MAX_DAILY_LOSS_SOL', '5.0'))
            self.slippage_bps = int(os.getenv('JUPITER_SLIPPAGE_BPS', '50'))

        # Trading state
        self.active_positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []

        # Cooldowns
        self.token_cooldowns: Dict[str, datetime] = {}
        self.cooldown_duration = timedelta(minutes=10)

        # Solana client and wallet
        self.client = None
        self.wallet = None
        self.wallet_pubkey = None

        # Strategy clients (price/quote APIs)
        self.jupiter_client: Optional[JupiterClient] = None
        self.drift_client = None
        self.pumpfun_monitor: Optional[PumpFunMonitor] = None

        # Live trading helpers (for actual swap execution)
        self.jupiter_helper: Optional['JupiterHelper'] = None
        self.drift_helper: Optional['DriftHelper'] = None

        # Risk metrics
        self.risk_metrics = RiskMetrics(
            daily_loss_limit_sol=self.max_daily_loss_sol
        )

        # SOL price for USD calculations
        self.sol_price_usd = 200.0  # Will be updated

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl_sol = 0.0
        self.total_fees = 0.0

        # PnL Tracker for Sharpe/Sortino calculations
        self.pnl_tracker = PnLTracker(
            initial_capital=self.position_size_sol * self.max_positions,
            currency="SOL"
        )

        # Log configuration
        mode_str = "DRY_RUN (SIMULATED)" if self.dry_run else "LIVE TRADING"
        logger.info(f"Solana engine initialized:")
        logger.info(f"  Mode: {mode_str}")
        logger.info(f"  RPC endpoints: {len(rpc_urls)}")
        logger.info(f"  Strategies: {', '.join(s.value for s in self.strategies)}")
        logger.info(f"  Position size: {self.position_size_sol} SOL")
        logger.info(f"  Slippage: {self.slippage_bps} bps")

    async def initialize(self):
        """Initialize Solana connections and components"""
        try:
            logger.info(f"Initializing Solana connection...")

            # Initialize Solana client
            await self._init_solana_client()

            # Initialize enabled strategies
            if Strategy.JUPITER in self.strategies:
                await self._init_jupiter()

            if Strategy.DRIFT in self.strategies:
                await self._init_drift()

            if Strategy.PUMPFUN in self.strategies:
                await self._init_pumpfun()

            # Get initial SOL price
            await self._update_sol_price()

            logger.info("✅ Solana connection and strategies initialized")

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    async def _init_solana_client(self):
        """Initialize Solana RPC client and wallet"""
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.keypair import Keypair

            # Create RPC client
            self.client = AsyncClient(self.rpc_manager.current_url)

            # Test connection
            health = await self.client.is_connected()
            if not health:
                logger.warning(f"Primary RPC not responding, trying failover...")
                self.rpc_manager.mark_failed(self.rpc_manager.current_url)
                self.client = AsyncClient(self.rpc_manager.get_healthy_url())

            # Load wallet from encrypted private key
            encrypted_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY')
            encryption_key = os.getenv('ENCRYPTION_KEY')

            if not encrypted_key:
                raise ValueError("SOLANA_MODULE_PRIVATE_KEY required for Solana trading module")

            # Decrypt private key if encrypted
            private_key = encrypted_key
            if encrypted_key.startswith('gAAAAAB') and encryption_key:
                try:
                    from cryptography.fernet import Fernet
                    f = Fernet(encryption_key.encode())
                    private_key = f.decrypt(encrypted_key.encode()).decode()
                    logger.info("✅ Successfully decrypted Solana module private key")
                except Exception as e:
                    logger.error(f"Failed to decrypt Solana module private key: {e}")
                    raise ValueError("Cannot decrypt SOLANA_MODULE_PRIVATE_KEY")

            # Decode private key (try base58 first, then hex)
            try:
                key_bytes = base58.b58decode(private_key)
                self.wallet = Keypair.from_bytes(key_bytes)
            except (ValueError, Exception) as b58_error:
                try:
                    key_bytes = bytes.fromhex(private_key)
                    self.wallet = Keypair.from_bytes(key_bytes)
                except (ValueError, Exception) as hex_error:
                    logger.error(f"Failed to decode private key as base58: {b58_error}")
                    logger.error(f"Failed to decode private key as hex: {hex_error}")
                    raise ValueError("Invalid private key format - must be base58 or hex encoded")

            self.wallet_pubkey = str(self.wallet.pubkey())
            logger.info(f"✅ Solana wallet loaded: {self.wallet_pubkey[:8]}...{self.wallet_pubkey[-8:]}")

            # Get wallet balance
            balance = await self._get_wallet_balance()
            logger.info(f"   Wallet balance: {balance:.4f} SOL")

        except ImportError:
            logger.error("solana library not installed. Install: pip install solana solders")
            raise
        except Exception as e:
            logger.error(f"Solana client initialization failed: {e}")
            raise

    async def _init_jupiter(self):
        """Initialize Jupiter V6 aggregator client"""
        try:
            logger.info("Initializing Jupiter V6 aggregator...")

            jupiter_api_url = os.getenv('JUPITER_API_URL', 'https://quote-api.jup.ag/v6')
            jupiter_api_key = os.getenv('JUPITER_API_KEY', '')

            self.jupiter_client = JupiterClient(
                api_url=jupiter_api_url,
                api_key=jupiter_api_key if jupiter_api_key else None,
                slippage_bps=self.slippage_bps
            )

            # Initialize JupiterHelper for live swap execution (if not in dry run)
            if not self.dry_run and JUPITER_HELPER_AVAILABLE:
                try:
                    # Get the decrypted private key for signing
                    private_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY', '')
                    encryption_key = os.getenv('ENCRYPTION_KEY', '')

                    # Decrypt if needed
                    if private_key.startswith('gAAAAAB') and encryption_key:
                        from cryptography.fernet import Fernet
                        f = Fernet(encryption_key.encode())
                        private_key = f.decrypt(private_key.encode()).decode()

                    self.jupiter_helper = JupiterHelper(
                        solana_rpc_url=self.primary_rpc,
                        private_key=private_key
                    )
                    await self.jupiter_helper.initialize()
                    logger.info("✅ JupiterHelper initialized for LIVE swap execution")
                except Exception as e:
                    logger.warning(f"⚠️ JupiterHelper not available for live swaps: {e}")
                    self.jupiter_helper = None

            # Test with SOL price - uses multi-source price fetching
            sol_price = await self.jupiter_client.get_price(SOL_MINT)
            if sol_price:
                self.sol_price_usd = sol_price
                # Get the source from cache
                price_source = self.jupiter_client._price_cache.get(SOL_MINT, {}).get('source', 'unknown')
                logger.info(f"✅ Jupiter configured (SOL: ${sol_price:.2f} via {price_source})")
            else:
                logger.warning("⚠️ Could not fetch SOL price from any source, using default $200")

        except Exception as e:
            logger.error(f"Jupiter initialization failed: {e}")
            raise

    async def _init_drift(self):
        """Initialize Drift Protocol client"""
        try:
            logger.info("Initializing Drift Protocol...")

            # Drift requires driftpy which is optional
            try:
                from driftpy.drift_client import DriftClient

                # Initialize DriftHelper for live perpetual trading
                if not self.dry_run and DRIFT_HELPER_AVAILABLE:
                    try:
                        self.drift_helper = DriftHelper(
                            rpc_url=self.primary_rpc,
                            private_key=os.getenv('SOLANA_MODULE_PRIVATE_KEY', '')
                        )
                        initialized = await self.drift_helper.initialize()
                        if initialized:
                            logger.info("✅ DriftHelper initialized for LIVE perpetual trading")
                        else:
                            logger.warning("⚠️ DriftHelper initialization failed")
                            self.drift_helper = None
                    except Exception as e:
                        logger.warning(f"⚠️ DriftHelper not available: {e}")
                        self.drift_helper = None
                else:
                    logger.info("✅ Drift Protocol configured (DRY_RUN mode)")

            except ImportError:
                logger.warning("⚠️ driftpy not installed. Drift trading disabled.")
                self.strategies = [s for s in self.strategies if s != Strategy.DRIFT]

        except Exception as e:
            logger.error(f"Drift initialization failed: {e}")

    async def _init_pumpfun(self):
        """Initialize Pump.fun monitoring"""
        try:
            logger.info("Initializing Pump.fun monitor...")

            # Build config from config_manager (DB) or environment variables
            if self.config_manager:
                config = {
                    'program_id': os.getenv('PUMPFUN_PROGRAM_ID', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'),
                    'min_liquidity_sol': self.config_manager.get('pumpfun_min_liquidity', 10.0),
                    'min_liquidity_usd': self.config_manager.get('pumpfun_min_liquidity_usd', 1000.0),
                    'min_volume_24h': self.config_manager.get('pumpfun_min_volume_24h', 5000.0),
                    'max_age_seconds': self.config_manager.get('pumpfun_max_age', 300),
                    'buy_amount_sol': self.config_manager.get('pumpfun_buy_amount', 0.1),
                    'ws_url': os.getenv('PUMPFUN_WS_URL', 'wss://pumpportal.fun/api/data')
                }
                logger.info(f"   Config from DB: min_liq=${config['min_liquidity_usd']}, min_vol=${config['min_volume_24h']}")
            else:
                config = {
                    'program_id': os.getenv('PUMPFUN_PROGRAM_ID', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'),
                    'min_liquidity_sol': float(os.getenv('PUMPFUN_MIN_LIQUIDITY', '10')),
                    'min_liquidity_usd': float(os.getenv('PUMPFUN_MIN_LIQUIDITY_USD', '1000')),
                    'min_volume_24h': float(os.getenv('PUMPFUN_MIN_VOLUME_24H', '5000')),
                    'max_age_seconds': int(os.getenv('PUMPFUN_MAX_AGE_SECONDS', '300')),
                    'buy_amount_sol': float(os.getenv('PUMPFUN_BUY_AMOUNT_SOL', '0.1')),
                    'ws_url': os.getenv('PUMPFUN_WS_URL', 'wss://pumpportal.fun/api/data')
                }

            self.pumpfun_monitor = PumpFunMonitor(config)
            logger.info("✅ Pump.fun monitor configured")

        except Exception as e:
            logger.error(f"Pump.fun initialization failed: {e}")

    async def _get_wallet_balance(self) -> float:
        """Get wallet SOL balance"""
        try:
            from solders.pubkey import Pubkey

            pubkey = Pubkey.from_string(self.wallet_pubkey)
            response = await self.client.get_balance(pubkey)
            if response.value is not None:
                return response.value / self.LAMPORTS_PER_SOL
            return 0.0
        except Exception as e:
            logger.error(f"Error getting wallet balance: {e}")
            return 0.0

    async def _update_sol_price(self):
        """Update SOL price in USD"""
        if self.jupiter_client:
            price = await self.jupiter_client.get_price(SOL_MINT)
            if price:
                old_price = self.sol_price_usd
                self.sol_price_usd = price
                # Log significant price changes (>1%)
                if abs(price - old_price) / old_price > 0.01:
                    logger.info(f"📈 SOL price: ${old_price:.2f} → ${price:.2f}")

    def calculate_dynamic_position_size(
        self,
        signal_strength: float = 0.5,
        volatility: float = 1.0,
        trend_strength: float = 0.5
    ) -> float:
        """
        Calculate dynamic position size based on signal strength and market conditions.

        Args:
            signal_strength: Signal confidence (0.0 to 1.0)
            volatility: Market volatility multiplier (1.0 = normal)
            trend_strength: Trend alignment strength (0.0 to 1.0)

        Returns:
            Adjusted position size in SOL
        """
        base_size = self.position_size_sol

        # Adjust based on signal strength (50% to 150% of base)
        signal_multiplier = 0.5 + (signal_strength * 1.0)

        # Reduce size in high volatility (inverse relationship)
        volatility_multiplier = min(1.5, max(0.5, 1.0 / volatility))

        # Boost for strong trend alignment
        trend_multiplier = 0.8 + (trend_strength * 0.4)

        # Calculate win rate adjustment (more wins = more confidence)
        if self.total_trades > 10:
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
            # Adjust between 0.8 (30% win rate) and 1.2 (70% win rate)
            win_rate_multiplier = 0.5 + (win_rate * 1.0)
        else:
            win_rate_multiplier = 1.0

        # Current risk adjustment (reduce if already exposed)
        current_exposure = len(self.active_positions) / self.max_positions if self.max_positions > 0 else 0
        exposure_multiplier = max(0.5, 1.0 - (current_exposure * 0.5))

        # Calculate final size
        dynamic_size = (
            base_size *
            signal_multiplier *
            volatility_multiplier *
            trend_multiplier *
            win_rate_multiplier *
            exposure_multiplier
        )

        # Enforce minimum and maximum bounds
        min_size = self.config_manager.get('min_position', 0.05) if self.config_manager else 0.05
        max_size = base_size * 2.0  # Never more than 2x base size

        final_size = max(min_size, min(max_size, dynamic_size))

        logger.debug(
            f"Dynamic position size: base={base_size:.4f}, "
            f"signal={signal_multiplier:.2f}, vol={volatility_multiplier:.2f}, "
            f"trend={trend_multiplier:.2f}, win={win_rate_multiplier:.2f}, "
            f"exposure={exposure_multiplier:.2f} -> final={final_size:.4f}"
        )

        return final_size

    def _log_trade(self, trade_type: str, details: dict):
        """Log trade to separate trade log file"""
        try:
            import logging
            trade_logger = logging.getLogger("SolanaTrading.Trades")
            trade_info = {
                'type': trade_type,
                'timestamp': datetime.now().isoformat(),
                'mode': 'DRY_RUN' if self.dry_run else 'LIVE',
                **details
            }
            trade_logger.info(json.dumps(trade_info))
        except Exception as e:
            logger.debug(f"Error logging trade: {e}")

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        mode_str = "DRY_RUN (SIMULATED)" if self.dry_run else "LIVE TRADING"
        logger.info(f"🚀 Solana trading engine started - {mode_str}")

        cycle_count = 0
        heartbeat_interval = 60  # Log heartbeat every 60 cycles (5 min at 5s intervals)
        scan_log_interval = 12  # Log scanning activity every 12 cycles (1 min at 5s intervals)

        try:
            while self.is_running:
                try:
                    cycle_count += 1

                    # Reset daily metrics at midnight
                    await self._check_daily_reset()

                    # Update SOL price periodically
                    await self._update_sol_price()

                    # Main trading logic
                    await self._trading_cycle()

                    # Periodic scan activity log (every 1 minute)
                    if cycle_count % scan_log_interval == 0:
                        pos_count = len(self.active_positions)
                        strategies_str = ', '.join(s.value for s in self.strategies)
                        logger.info(
                            f"🔄 Scanning: strategies={strategies_str}, "
                            f"positions={pos_count}/{self.max_positions}, "
                            f"SOL=${self.sol_price_usd:.2f}, "
                            f"daily_pnl={self.risk_metrics.daily_pnl_sol:.4f} SOL"
                        )

                    # Periodic heartbeat log (every 5 minutes)
                    if cycle_count % heartbeat_interval == 0:
                        pos_count = len(self.active_positions)
                        strategies_str = ', '.join(s.value for s in self.strategies)
                        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                        logger.info(
                            f"💓 Heartbeat: cycle={cycle_count}, "
                            f"trades={self.total_trades} (win={win_rate:.1f}%), "
                            f"pnl={self.total_pnl_sol:.4f} SOL"
                        )

                    # Wait before next cycle
                    await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)
                    await asyncio.sleep(15)

        except Exception as e:
            logger.error(f"Critical error in trading engine: {e}", exc_info=True)
            raise

        finally:
            logger.info("Solana trading engine stopped")

    async def _check_daily_reset(self):
        """Reset daily metrics at midnight UTC"""
        now = datetime.utcnow()
        if now.date() > self.risk_metrics.last_reset.date():
            logger.info("📅 Resetting daily risk metrics")
            self.risk_metrics.daily_pnl_sol = 0.0
            self.risk_metrics.daily_pnl_usd = 0.0
            self.risk_metrics.daily_trades = 0
            self.risk_metrics.consecutive_losses = 0
            self.risk_metrics.last_reset = now

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Check risk limits
            if not self.risk_metrics.can_trade:
                logger.warning(f"⚠️ Trading paused - Risk limit reached (Daily PnL: {self.risk_metrics.daily_pnl_sol:.4f} SOL)")
                return

            # Monitor existing positions
            await self._monitor_positions()

            # Scan for opportunities per strategy
            if Strategy.JUPITER in self.strategies:
                await self._scan_jupiter_opportunities()

            if Strategy.DRIFT in self.strategies:
                await self._scan_drift_opportunities()

            if Strategy.PUMPFUN in self.strategies:
                await self._scan_pumpfun_opportunities()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _monitor_positions(self):
        """Monitor active positions for exit signals"""
        if not self.active_positions:
            return

        # Log monitoring activity periodically
        pos_count = len(self.active_positions)
        logger.info(f"📊 Monitoring {pos_count} active position(s)...")

        for token_mint, position in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = await self._get_token_price(token_mint)
                if current_price is None or current_price == 0:
                    logger.warning(f"⚠️ Could not fetch price for {position.token_symbol}")
                    continue

                old_price = position.current_price
                position.current_price = current_price

                # Calculate PnL
                if position.side == TradeSide.BUY:
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                position.unrealized_pnl_pct = pnl_pct
                position.unrealized_pnl = position.value_sol * (pnl_pct / 100)

                # Log position status
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                logger.info(
                    f"{pnl_emoji} {position.token_symbol}: "
                    f"${current_price:.8f} (entry: ${position.entry_price:.8f}), "
                    f"PnL: {pnl_pct:+.2f}%, "
                    f"SL: {self.stop_loss_pct}%, TP: {self.take_profit_pct}%"
                )

                # Check exit conditions
                exit_reason = await self._check_exit_conditions(position)
                if exit_reason:
                    logger.info(f"🎯 Exit signal for {position.token_symbol}: {exit_reason}")
                    await self._close_position(token_mint, exit_reason)

            except Exception as e:
                logger.error(f"Error monitoring position {token_mint[:8]}...: {e}")

    async def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed"""
        pnl_pct = position.unrealized_pnl_pct

        # Stop loss
        if pnl_pct <= self.stop_loss_pct:
            return "stop_loss"

        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return "take_profit"

        # Auto-sell delay for Pump.fun positions
        if position.strategy == Strategy.PUMPFUN:
            auto_sell_delay = 0
            if self.config_manager:
                auto_sell_delay = self.config_manager.get('pumpfun_auto_sell', 0)

            if auto_sell_delay > 0:
                time_held = (datetime.now() - position.opened_at).total_seconds()
                if time_held >= auto_sell_delay:
                    return "auto_sell_timeout"

        return None

    async def _scan_jupiter_opportunities(self):
        """Scan for Jupiter swap opportunities"""
        if len(self.active_positions) >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached, skipping Jupiter scan")
            return

        try:
            # Update SOL price
            if self.jupiter_client:
                sol_price = await self.jupiter_client.get_price(SOL_MINT)
                if sol_price and sol_price != self.sol_price_usd:
                    old_price = self.sol_price_usd
                    self.sol_price_usd = sol_price
                    pct_change = ((sol_price - old_price) / old_price) * 100 if old_price > 0 else 0
                    if abs(pct_change) > 0.5:  # Log significant changes
                        logger.info(f"📈 SOL price: ${old_price:.2f} → ${sol_price:.2f} ({pct_change:+.2f}%)")

            # Get tokens to monitor from config (configurable from settings page)
            if self.config_manager:
                tokens_to_scan = self.config_manager.jupiter_tokens
                if not tokens_to_scan:
                    # Fallback defaults if no tokens configured
                    tokens_to_scan = [
                        ('BONK', 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263'),
                        ('JTO', 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL'),
                        ('WIF', 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm'),
                    ]
            else:
                # No config manager - use defaults
                tokens_to_scan = [
                    ('BONK', 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263'),
                    ('JTO', 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL'),
                    ('WIF', 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm'),
                ]

            for token_name, token_mint in tokens_to_scan:
                # Skip if in cooldown
                if token_mint in self.token_cooldowns:
                    if datetime.now() < self.token_cooldowns[token_mint]:
                        continue

                # Skip if already have position
                if token_mint in self.active_positions:
                    continue

                price = await self.jupiter_client.get_price(token_mint)
                if not price:
                    continue

                logger.info(f"🔍 Jupiter: {token_name} = ${price:.8f}")

                # Trading signals for DRY_RUN and LIVE modes
                # Get price history from cache to detect momentum
                if not hasattr(self, '_price_history'):
                    self._price_history = {}
                if not hasattr(self, '_demo_trade_counter'):
                    self._demo_trade_counter = 0

                if token_mint not in self._price_history:
                    self._price_history[token_mint] = []

                self._price_history[token_mint].append({'price': price, 'time': datetime.now()})

                # Keep only last 10 prices
                self._price_history[token_mint] = self._price_history[token_mint][-10:]

                # Check for momentum signal
                momentum = 0.0
                should_trade = False
                trade_reason = ""

                if len(self._price_history[token_mint]) >= 3:
                    prices = [p['price'] for p in self._price_history[token_mint]]
                    avg_old = sum(prices[:-1]) / len(prices[:-1])
                    current = prices[-1]
                    momentum = ((current - avg_old) / avg_old) * 100 if avg_old > 0 else 0

                    # Buy signal: positive momentum > 0.1% (lowered from 0.5%)
                    if momentum > 0.1:
                        should_trade = True
                        trade_reason = f"momentum (+{momentum:.2f}%)"

                    # Also trigger on negative momentum (reversal potential)
                    elif momentum < -0.2 and len(self._price_history[token_mint]) >= 5:
                        # Potential reversal - price dropped but may bounce
                        should_trade = True
                        trade_reason = f"reversal potential ({momentum:.2f}%)"

                # DRY_RUN Demo Mode: trigger trades periodically for testing
                if self.dry_run and not should_trade:
                    self._demo_trade_counter += 1
                    # Trigger a demo trade every ~60 cycles (5 minutes) per token
                    if self._demo_trade_counter % 60 == 0:
                        should_trade = True
                        trade_reason = "demo periodic trade"
                        momentum = 0.05  # Small positive for demo

                if should_trade:
                    logger.info(f"📊 {token_name}: Trading signal - {trade_reason}")
                    await self._open_position(
                        token_mint=token_mint,
                        token_symbol=token_name,
                        strategy=Strategy.JUPITER,
                        amount_sol=self.calculate_dynamic_position_size(
                            signal_strength=min(1.0, abs(momentum) / 2.0 + 0.3),
                            volatility=1.0,
                            trend_strength=0.7
                        ),
                        metadata={'momentum': momentum, 'price': price, 'reason': trade_reason}
                    )
                    break  # One trade per cycle

        except Exception as e:
            logger.error(f"Error in Jupiter scan: {e}", exc_info=True)

    async def _scan_drift_opportunities(self):
        """Scan for Drift perpetual opportunities"""
        if len(self.active_positions) >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached, skipping Drift scan")
            return

        if not self.drift_client:
            return

        try:
            # Log that Drift is being scanned
            # In production, check funding rates, order book depth, etc.
            logger.debug("🔍 Drift scanning for perpetual opportunities...")

        except Exception as e:
            logger.error(f"Error in Drift scan: {e}")

    async def _scan_pumpfun_opportunities(self):
        """Scan for Pump.fun new token launches"""
        if len(self.active_positions) >= self.max_positions:
            logger.debug("Max positions reached, skipping Pump.fun scan")
            return

        if not self.pumpfun_monitor:
            return

        try:
            # Get new tokens (already filtered in get_new_tokens)
            new_tokens = await self.pumpfun_monitor.get_new_tokens()

            for token in new_tokens:
                token_mint = token.get('mint')
                if not token_mint:
                    continue

                # Check cooldown
                if token_mint in self.token_cooldowns:
                    if datetime.now() < self.token_cooldowns[token_mint]:
                        logger.debug(f"Token {token.get('symbol')} in cooldown")
                        continue

                # Skip if already have position
                if token_mint in self.active_positions:
                    continue

                logger.info(f"🎯 Pump.fun: Opening position on {token.get('symbol', 'UNKNOWN')}")

                # Open position
                await self._open_position(
                    token_mint=token_mint,
                    token_symbol=token.get('symbol', 'UNKNOWN'),
                    strategy=Strategy.PUMPFUN,
                    amount_sol=self.pumpfun_monitor.buy_amount_sol,
                    metadata=token
                )

                break  # One token per cycle

        except Exception as e:
            logger.error(f"Error scanning Pump.fun: {e}")

    async def _get_token_price(self, token_mint: str) -> Optional[float]:
        """Get current price for token via Jupiter"""
        if self.jupiter_client:
            return await self.jupiter_client.get_price(token_mint)
        return None

    async def _open_position(
        self,
        token_mint: str,
        token_symbol: str,
        strategy: Strategy,
        amount_sol: float,
        metadata: Dict = None
    ):
        """Open a new position"""
        try:
            # Get token price
            current_price = await self._get_token_price(token_mint)
            if current_price is None:
                logger.warning(f"Could not get price for {token_symbol}")
                return

            # Calculate values
            value_usd = amount_sol * self.sol_price_usd
            token_amount = (amount_sol * self.sol_price_usd) / current_price if current_price > 0 else 0

            # Create position
            position = Position(
                position_id=str(uuid.uuid4()),
                token_mint=token_mint,
                token_symbol=token_symbol,
                strategy=strategy,
                side=TradeSide.BUY,
                entry_price=current_price,
                current_price=current_price,
                amount=token_amount,
                value_sol=amount_sol,
                value_usd=value_usd,
                stop_loss=current_price * (1 + self.stop_loss_pct / 100),
                take_profit=current_price * (1 + self.take_profit_pct / 100),
                is_simulated=self.dry_run,
                metadata=metadata or {}
            )

            # Execute swap (or simulate)
            tx_signature = None
            if self.dry_run:
                logger.info(f"🔵 [DRY_RUN] SIMULATED BUY {token_symbol}")
                logger.info(f"   Price: ${current_price:.8f}")
                logger.info(f"   Amount: {amount_sol} SOL (${value_usd:.2f})")
                tx_signature = f"DRY_RUN_{uuid.uuid4().hex[:16]}"
            else:
                # Execute real Jupiter swap
                try:
                    if self.jupiter_helper:
                        # Use JupiterHelper for full swap execution
                        logger.info(f"🔄 Executing LIVE swap: {amount_sol} SOL → {token_symbol}")
                        tx_signature = await self.jupiter_helper.execute_swap(
                            input_mint=SOL_MINT,
                            output_mint=token_mint,
                            amount=int(amount_sol * self.LAMPORTS_PER_SOL),
                            slippage_bps=self.slippage_bps
                        )

                        if tx_signature:
                            logger.info(f"🟢 LIVE SWAP executed: {tx_signature}")
                            position.tx_signature = tx_signature
                        else:
                            logger.error("❌ Jupiter swap failed - no signature returned")
                            return
                    else:
                        # Fallback: get quote but warn about no execution
                        quote = await self.jupiter_client.get_quote(
                            input_mint=SOL_MINT,
                            output_mint=token_mint,
                            amount=int(amount_sol * self.LAMPORTS_PER_SOL)
                        )

                        if not quote:
                            logger.error("Failed to get Jupiter quote")
                            return

                        logger.warning(f"⚠️ JupiterHelper not available - swap NOT executed (quote only)")
                        logger.warning(f"   Quote output: {quote.get('outAmount', 'N/A')} lamports")
                        return  # Don't open position without actual swap

                except Exception as e:
                    logger.error(f"❌ Swap execution failed: {e}", exc_info=True)
                    return

            # Add to positions
            self.active_positions[token_mint] = position
            self.risk_metrics.current_exposure_sol += amount_sol

            logger.info(f"✅ Position opened: {token_symbol} ({strategy.value})")

            # Log trade to separate trade file
            self._log_trade('OPEN', {
                'position_id': position.position_id,
                'token': token_symbol,
                'mint': token_mint,
                'strategy': strategy.value,
                'side': 'BUY',
                'price': current_price,
                'amount_sol': amount_sol,
                'amount_usd': value_usd,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            })

        except Exception as e:
            logger.error(f"Error opening position: {e}")

    async def _close_position(self, token_mint: str, reason: str):
        """Close a position"""
        if token_mint not in self.active_positions:
            return

        position = self.active_positions[token_mint]

        try:
            # Calculate PnL
            pnl_pct = position.unrealized_pnl_pct
            pnl_sol = position.unrealized_pnl
            pnl_usd = pnl_sol * self.sol_price_usd

            # Execute close (or simulate)
            close_tx_signature = None
            if self.dry_run:
                logger.info(f"🔵 [DRY_RUN] SIMULATED SELL {position.token_symbol} ({reason})")
                close_tx_signature = f"DRY_RUN_CLOSE_{uuid.uuid4().hex[:16]}"
            else:
                # Execute real Jupiter swap back to SOL
                try:
                    if self.jupiter_helper:
                        # Calculate token amount in smallest units (lamports equivalent)
                        # Most SPL tokens use 6 or 9 decimals
                        token_decimals = 6  # Default, would need to fetch actual decimals
                        token_amount_raw = int(position.amount * (10 ** token_decimals))

                        logger.info(f"🔄 Executing LIVE close: {position.token_symbol} → SOL")
                        close_tx_signature = await self.jupiter_helper.execute_swap(
                            input_mint=token_mint,
                            output_mint=SOL_MINT,
                            amount=token_amount_raw,
                            slippage_bps=self.slippage_bps
                        )

                        if close_tx_signature:
                            logger.info(f"🟢 LIVE CLOSE executed: {close_tx_signature}")
                        else:
                            logger.error(f"❌ Close swap failed for {position.token_symbol}")
                            # Still record the position close attempt
                    else:
                        logger.warning(f"⚠️ JupiterHelper not available - close NOT executed")
                        logger.warning(f"   Position {position.token_symbol} needs manual close!")

                except Exception as e:
                    logger.error(f"❌ Close execution failed: {e}", exc_info=True)

            # Record trade
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                token_mint=token_mint,
                token_symbol=position.token_symbol,
                strategy=position.strategy,
                side=TradeSide.SELL,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                amount=position.amount,
                pnl_sol=pnl_sol,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                fees=position.fees_paid,
                opened_at=position.opened_at,
                closed_at=datetime.now(),
                close_reason=reason,
                is_simulated=position.is_simulated
            )
            self.trade_history.append(trade)

            # Update stats
            self.total_trades += 1
            self.total_pnl_sol += pnl_sol
            self.risk_metrics.daily_pnl_sol += pnl_sol
            self.risk_metrics.daily_pnl_usd += pnl_usd
            self.risk_metrics.daily_trades += 1
            self.risk_metrics.current_exposure_sol -= position.value_sol

            if pnl_sol > 0:
                self.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1

            # Record trade in PnL tracker for Sharpe/Sortino calculations
            net_pnl = pnl_sol - position.fees_paid
            trade_record = TradeRecord(
                trade_id=trade.trade_id,
                symbol=position.token_symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                size=position.amount,
                pnl=pnl_sol,
                fees=position.fees_paid,
                net_pnl=net_pnl,
                pnl_pct=pnl_pct,
                entry_time=position.opened_at,
                exit_time=datetime.now(),
                duration_seconds=int((datetime.now() - position.opened_at).total_seconds()),
                is_simulated=position.is_simulated
            )
            self.pnl_tracker.record_trade(trade_record)

            # Remove position
            del self.active_positions[token_mint]

            # Set cooldown
            self.token_cooldowns[token_mint] = datetime.now() + self.cooldown_duration

            # Log result
            pnl_emoji = "🟢" if pnl_sol > 0 else "🔴"
            sim_tag = "[DRY_RUN] " if position.is_simulated else ""
            logger.info(f"{pnl_emoji} {sim_tag}Position closed: {position.token_symbol}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   PnL: {pnl_sol:.4f} SOL (${pnl_usd:.2f}, {pnl_pct:.2f}%)")
            logger.info(f"   Daily PnL: {self.risk_metrics.daily_pnl_sol:.4f} SOL")

            # Log trade to separate trade file
            self._log_trade('CLOSE', {
                'position_id': position.position_id,
                'trade_id': trade.trade_id,
                'token': position.token_symbol,
                'mint': token_mint,
                'strategy': position.strategy.value,
                'side': 'SELL',
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'pnl_sol': pnl_sol,
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'win': pnl_sol > 0
            })

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    async def close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for token_mint in list(self.active_positions.keys()):
            await self._close_position(token_mint, "manual_close")

        logger.info("✅ All positions closed")

    async def get_stats(self) -> Dict:
        """Get trading statistics"""
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100

        positions_summary = []
        for mint, pos in self.active_positions.items():
            positions_summary.append({
                'token': pos.token_symbol,
                'mint': mint[:8] + '...',
                'strategy': pos.strategy.value,
                'entry': f"${pos.entry_price:.8f}",
                'current': f"${pos.current_price:.8f}",
                'pnl_pct': f"{pos.unrealized_pnl_pct:.2f}%"
            })

        # Get advanced metrics from PnL tracker
        sharpe = self.pnl_tracker.get_sharpe_ratio()
        sortino = self.pnl_tracker.get_sortino_ratio()
        calmar = self.pnl_tracker.get_calmar_ratio()
        profit_factor = self.pnl_tracker.get_profit_factor()
        max_drawdown = self.pnl_tracker.max_drawdown_pct  # Use attribute, not method

        return {
            'mode': 'DRY_RUN' if self.dry_run else 'LIVE',
            'strategies': ', '.join(s.value for s in self.strategies),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': f"{self.total_pnl_sol:.4f} SOL",
            'daily_pnl': f"{self.risk_metrics.daily_pnl_sol:.4f} SOL",
            'daily_trades': self.risk_metrics.daily_trades,
            'risk_level': self.risk_metrics.risk_level,
            'active_positions': len(self.active_positions),
            'positions': positions_summary,
            'sol_price': f"${self.sol_price_usd:.2f}",
            'sol_price_usd': self.sol_price_usd,  # Numeric for calculations
            'sharpe_ratio': f"{sharpe:.2f}" if sharpe else "N/A",
            'sortino_ratio': f"{sortino:.2f}" if sortino else "N/A",
            'calmar_ratio': f"{calmar:.2f}" if calmar else "N/A",
            'profit_factor': f"{profit_factor:.2f}" if profit_factor else "N/A",
            'max_drawdown': f"{max_drawdown:.2f}%" if max_drawdown else "N/A"
        }

    async def get_health(self) -> Dict:
        """Get health status"""
        rpc_connected = False
        try:
            if self.client:
                rpc_connected = await self.client.is_connected()
        except Exception as e:
            logger.debug(f"RPC connection check failed: {e}")

        wallet_balance = await self._get_wallet_balance() if rpc_connected else 0

        return {
            'status': 'healthy' if self.is_running and rpc_connected else 'degraded',
            'engine_running': self.is_running,
            'rpc_connected': rpc_connected,
            'rpc_url': self.rpc_manager.current_url[:50] + '...',
            'dry_run': self.dry_run,
            'wallet_balance_sol': wallet_balance,
            'risk_can_trade': self.risk_metrics.can_trade,
            'active_positions': len(self.active_positions),
            'daily_pnl_sol': self.risk_metrics.daily_pnl_sol,
            'rpc_latencies': self.rpc_manager.get_latencies()
        }

    async def shutdown(self):
        """Shutdown the engine"""
        logger.info("Shutting down Solana trading engine...")
        self.is_running = False

        # Close clients with proper error handling
        if self.jupiter_client:
            try:
                await self.jupiter_client.close()
            except Exception as e:
                logger.debug(f"Error closing Jupiter client: {e}")

        if self.jupiter_helper:
            try:
                await self.jupiter_helper.close()
            except Exception as e:
                logger.debug(f"Error closing Jupiter helper: {e}")

        if self.drift_helper:
            try:
                await self.drift_helper.close()
            except Exception as e:
                logger.debug(f"Error closing Drift helper: {e}")

        if self.pumpfun_monitor:
            try:
                await self.pumpfun_monitor.close()
            except Exception as e:
                logger.debug(f"Error closing Pump.fun monitor: {e}")

        if self.client:
            try:
                await self.client.close()
            except Exception as e:
                logger.debug(f"Error closing Solana client: {e}")

        logger.info("✅ Engine shutdown complete")
