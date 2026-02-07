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
import time
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

# Import SafetyEngine for honeypot detection and close retry logic
try:
    from modules.solana_trading.core.safety_engine import (
        SafetyEngine, SafetyConfig, CloseFailureReason
    )
    SAFETY_ENGINE_AVAILABLE = True
except ImportError:
    SAFETY_ENGINE_AVAILABLE = False

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
    opened_at: datetime = field(default_factory=datetime.utcnow)
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

    # Time-based consecutive loss blocking
    loss_block_start: Optional[datetime] = None  # When the block started
    loss_block_count: int = 0  # Number of times blocked (for progressive blocking)
    max_consecutive_losses: int = 5  # Threshold to trigger block
    block_duration_hours: float = 2.0  # Base block duration
    progressive_block: bool = True  # Increase block time after repeated blocks
    max_block_hours: float = 6.0  # Maximum block duration

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        if self.daily_pnl_sol <= -self.daily_loss_limit_sol:
            return False

        # Check consecutive losses with time-based reset
        if self.consecutive_losses >= self.max_consecutive_losses:
            # Check if we're in a timed block
            if self.loss_block_start is not None:
                block_duration = self._get_block_duration()
                elapsed = datetime.now() - self.loss_block_start
                if elapsed >= block_duration:
                    # Block expired - will be reset by check_and_reset_loss_block()
                    return True
                return False
            return False
        return True

    def _get_block_duration(self) -> timedelta:
        """Calculate block duration based on block count"""
        if not self.progressive_block or self.loss_block_count <= 1:
            return timedelta(hours=self.block_duration_hours)

        # Progressive: 2h -> 4h -> 6h (max)
        multiplier = min(self.loss_block_count, 3)  # Cap at 3x
        hours = min(self.block_duration_hours * multiplier, self.max_block_hours)
        return timedelta(hours=hours)

    def get_block_remaining_seconds(self) -> Optional[int]:
        """Get remaining seconds until block expires, or None if not blocked"""
        if self.consecutive_losses < self.max_consecutive_losses:
            return None
        if self.loss_block_start is None:
            return None

        block_duration = self._get_block_duration()
        elapsed = datetime.now() - self.loss_block_start
        remaining = block_duration - elapsed

        if remaining.total_seconds() <= 0:
            return 0
        return int(remaining.total_seconds())

    def trigger_loss_block(self):
        """Trigger a consecutive loss block"""
        if self.loss_block_start is None:
            self.loss_block_start = datetime.now()
            self.loss_block_count += 1
            block_hours = self._get_block_duration().total_seconds() / 3600
            logger.warning(f"🚫 CONSECUTIVE LOSS BLOCK TRIGGERED: {self.consecutive_losses} losses")
            logger.warning(f"   Block #{self.loss_block_count} for {block_hours:.1f} hours")

    def check_and_reset_loss_block(self) -> bool:
        """
        Check if loss block has expired and reset if so.

        Returns:
            True if block was reset, False otherwise
        """
        if self.loss_block_start is None:
            return False

        block_duration = self._get_block_duration()
        elapsed = datetime.now() - self.loss_block_start

        if elapsed >= block_duration:
            logger.info(f"✅ CONSECUTIVE LOSS BLOCK EXPIRED after {elapsed.total_seconds()/3600:.1f}h")
            logger.info(f"   Resetting consecutive losses from {self.consecutive_losses} to 0")
            self.consecutive_losses = 0
            self.loss_block_start = None
            return True
        return False

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
        # CRITICAL: Short cache for pump.fun tokens that can rug in seconds
        # 30s was too long - missed -20% SL because price went from +18% to -47% between refreshes
        self._cache_ttl = timedelta(seconds=5)
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
            price_api_url = "https://api.jup.ag/price/v3"
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
        self.buy_amount_sol = config.get('buy_amount_sol', 0.05)
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

                        # ENHANCED FILTERS: Avoid scams, dumps, and dead tokens
                        price_change = token_info.get('price_change_24h', 0)
                        liq = token_info['liquidity_usd']
                        vol = token_info['volume_24h']

                        # Reject tokens already in a dump (price dropping > 10%)
                        # Tightened from -15% to -10%: entering during a decline is high risk
                        if price_change < -10:
                            filtered_out['suspicious'] = filtered_out.get('suspicious', 0) + 1
                            continue

                        # Require minimum liquidity-to-volume ratio (avoid fake volume)
                        if liq > 0 and vol / liq > 50:
                            # Volume 50x liquidity = likely wash trading
                            filtered_out['suspicious'] = filtered_out.get('suspicious', 0) + 1
                            continue

                        # Require minimum $3000 liquidity for pump.fun (override if config is lower)
                        if liq < 3000:
                            filtered_out['liquidity'] += 1
                            continue

                        # Reject tokens with very low volume relative to liquidity
                        # (indicates the token is dying / no active trading)
                        if liq > 0 and vol / liq < 0.5:
                            filtered_out['suspicious'] = filtered_out.get('suspicious', 0) + 1
                            continue

                        # Reject tokens with very low liquidity-to-volume that suggests
                        # the token has already peaked and is winding down
                        # Tokens like GAS, THRT had high initial volume but declining interest
                        if vol < 10000 and liq < 20000:
                            # Low volume AND low liquidity = dead/dying token
                            filtered_out['suspicious'] = filtered_out.get('suspicious', 0) + 1
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
        config_manager=None,
        db_pool=None
    ):
        """
        Initialize Solana trading engine

        Args:
            rpc_url: Primary Solana RPC URL
            strategies: List of enabled strategies (jupiter, drift, pumpfun)
            max_positions: Maximum concurrent positions
            mode: Operating mode
            config_manager: Optional SolanaConfigManager for DB-backed config
            db_pool: Optional asyncpg database pool for trade persistence
        """
        self.primary_rpc = rpc_url
        self.strategies = [Strategy(s.strip().lower()) for s in strategies if s.strip().lower() in [e.value for e in Strategy]]
        self.max_positions = max_positions
        self.mode = mode
        self.is_running = False
        self.config_manager = config_manager
        self.db_pool = db_pool  # Database pool for trade persistence

        # DRY_RUN mode - CRITICAL: Check environment variable
        dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
        self.dry_run = dry_run_env in ('true', '1', 'yes')

        # RPC configuration - use Pool Engine with fallback to env vars
        rpc_urls = []
        try:
            from config.rpc_provider import RPCProvider
            rpc_urls = RPCProvider.get_rpcs_sync('SOLANA_RPC', max_count=5)
        except Exception:
            pass

        if not rpc_urls:
            rpc_urls_env = os.getenv('SOLANA_RPC_URLS', '')
            if rpc_urls_env:
                rpc_urls = [url.strip() for url in rpc_urls_env.split(',') if url.strip()]
            else:
                rpc_urls = [rpc_url]

        # Ensure primary is first
        if rpc_url and rpc_url not in rpc_urls:
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
            self.slippage_bps = int(os.getenv('JUPITER_SLIPPAGE_BPS', '200'))  # 2% default (was 50bps=0.5%)

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

        # State tracking
        self._last_pause_log_time = None

        # PnL Tracker for Sharpe/Sortino calculations
        self.pnl_tracker = PnLTracker(
            initial_capital=self.position_size_sol * self.max_positions,
            currency="SOL"
        )

        # Telegram alerts - will be initialized in async initialize() method
        # where we can properly load credentials from secrets manager
        self.telegram_alerts = None

        # Scam token blacklist - initialized in async initialize() method
        self.scam_blacklist = None

        # Safety engine - initialized in async initialize() method
        self.safety_engine = None

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
            # NOTE: Jupiter client is needed for both Jupiter AND Pumpfun strategies (for swaps)
            if Strategy.JUPITER in self.strategies or Strategy.PUMPFUN in self.strategies:
                await self._init_jupiter()

            if Strategy.DRIFT in self.strategies:
                await self._init_drift()

            if Strategy.PUMPFUN in self.strategies:
                await self._init_pumpfun()

            # Get initial SOL price
            await self._update_sol_price()

            # Load historical stats from database for accurate PnL tracking
            await self._load_historical_stats()

            # Initialize SafetyEngine for all strategies (slippage, circuit breaker, etc.)
            try:
                from modules.solana_trading.core.safety_engine import SafetyEngine, SafetyConfig
                pumpfun_slippage = self.config_manager.get('pumpfun_slippage', 800) if self.config_manager else 800
                safety_config = SafetyConfig(
                    default_slippage_bps=max(self.slippage_bps, 100),  # At least 1%
                    pumpfun_slippage_bps=pumpfun_slippage,
                    pumpfun_close_slippage_bps=1200,  # 12% base for pump.fun sells
                    max_emergency_slippage_bps=3500,  # 35% max for emergency exits
                    max_close_retries=7,  # More retries for stuck positions
                    retry_slippage_increment_bps=400,  # +4% each retry (was 2%)
                    circuit_breaker_failures=5,  # More tolerance before circuit break
                    circuit_breaker_cooldown_seconds=300
                )
                self.safety_engine = SafetyEngine(safety_config)
                logger.info(f"✅ SafetyEngine initialized (pump.fun slippage: {pumpfun_slippage}bps)")
            except Exception as e:
                logger.warning(f"⚠️ SafetyEngine not available: {e}")
                self.safety_engine = None

            # Initialize Telegram alerts with async credential loading
            try:
                from solana_trading.core.solana_alerts import SolanaTelegramAlerts
                from security.secrets_manager import secrets

                # Pre-load Telegram credentials asynchronously
                bot_token = await secrets.get_async('TELEGRAM_BOT_TOKEN', log_access=False)
                chat_id = await secrets.get_async('TELEGRAM_CHAT_ID', log_access=False)

                self.telegram_alerts = SolanaTelegramAlerts(
                    bot_token=bot_token,
                    chat_id=chat_id
                )
                if self.telegram_alerts.enabled:
                    logger.info("✅ Telegram alerts enabled for Solana module")
            except Exception as e:
                logger.warning(f"Solana Telegram alerts not available: {e}")

            # Initialize scam token blacklist
            try:
                from .scam_blacklist import initialize_blacklist
                self.scam_blacklist = await initialize_blacklist(self.db_pool)
                logger.info("✅ Scam token blacklist initialized")
            except Exception as e:
                logger.warning(f"Scam blacklist not available: {e}")

            logger.info("✅ Solana connection and strategies initialized")

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    async def _get_decrypted_private_key(self) -> str:
        """
        Get decrypted private key from secrets manager or database.

        Priority:
        1. Secrets manager (Docker secrets, database, env)
        2. Direct environment variable with decryption

        Always checks if the returned value is still encrypted and decrypts if needed.
        """
        try:
            private_key = None

            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                # Initialize with db_pool if available (re-init if in bootstrap mode)
                if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                    secrets.initialize(self.db_pool)

                private_key = await secrets.get_async('SOLANA_MODULE_PRIVATE_KEY')
                if private_key:
                    logger.debug("Got Solana private key from secrets manager")
            except Exception as e:
                logger.debug(f"Secrets manager not available: {e}")

            # Fallback: Get from environment
            if not private_key:
                private_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY')

            if not private_key:
                return None

            # Check if still encrypted (Fernet tokens start with gAAAAAB)
            # This can happen if DB returned encrypted value without decrypting
            if private_key.startswith('gAAAAAB'):
                logger.debug("Private key appears to be Fernet encrypted, decrypting...")
                # Get encryption key from file or environment
                encryption_key = None
                key_file = Path('.encryption_key')
                if key_file.exists():
                    encryption_key = key_file.read_text().strip()
                if not encryption_key:
                    encryption_key = os.getenv('ENCRYPTION_KEY')

                if not encryption_key:
                    logger.error("Cannot decrypt private key: no encryption key found")
                    return None

                try:
                    from cryptography.fernet import Fernet
                    f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                    decrypted_key = f.decrypt(private_key.encode()).decode()
                    logger.info("✅ Successfully decrypted Solana module private key")
                    return decrypted_key
                except Exception as e:
                    logger.error(f"Failed to decrypt Solana module private key: {e}")
                    return None
            else:
                # Not encrypted, return as-is
                logger.info("✅ Loaded Solana private key (already decrypted)")
                return private_key

        except Exception as e:
            logger.error(f"Error getting private key: {e}")
            return None

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

            # Load wallet from encrypted private key (using secrets manager)
            private_key = await self._get_decrypted_private_key()
            if not private_key:
                raise ValueError("SOLANA_MODULE_PRIVATE_KEY required for Solana trading module")

            # Decode private key (try multiple formats)
            key_bytes = None

            # Format 1: JSON array (e.g., [1,2,3,...])
            if private_key.startswith('['):
                try:
                    import json
                    key_array = json.loads(private_key)
                    key_bytes = bytes(key_array)
                    logger.debug("Parsed private key from JSON array format")
                except Exception as json_error:
                    logger.debug(f"Not JSON array format: {json_error}")

            # Format 2: Base58 encoded
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(private_key)
                    logger.debug("Parsed private key from base58 format")
                except (ValueError, Exception) as b58_error:
                    logger.debug(f"Not base58 format: {b58_error}")

            # Format 3: Hex encoded
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(private_key)
                    logger.debug("Parsed private key from hex format")
                except (ValueError, Exception) as hex_error:
                    logger.debug(f"Not hex format: {hex_error}")

            if key_bytes is None:
                raise ValueError("Invalid private key format - must be JSON array, base58, or hex encoded")

            self.wallet = Keypair.from_bytes(key_bytes)

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

            raw_url = os.getenv('JUPITER_API_URL', 'https://lite-api.jup.ag/swap/v1')

            # Normalize URL - ensure it has the proper API path suffix
            # If user sets just "https://lite-api.jup.ag", append "/swap/v1"
            if 'lite-api.jup.ag' in raw_url and not raw_url.endswith('/swap/v1'):
                if raw_url.endswith('/'):
                    jupiter_api_url = raw_url + 'swap/v1'
                else:
                    jupiter_api_url = raw_url + '/swap/v1'
            elif 'quote-api.jup.ag' in raw_url and not raw_url.endswith('/v6'):
                if raw_url.endswith('/'):
                    jupiter_api_url = raw_url + 'v6'
                else:
                    jupiter_api_url = raw_url + '/v6'
            else:
                jupiter_api_url = raw_url

            jupiter_api_key = os.getenv('JUPITER_API_KEY', '')

            self.jupiter_client = JupiterClient(
                api_url=jupiter_api_url,
                api_key=jupiter_api_key if jupiter_api_key else None,
                slippage_bps=self.slippage_bps
            )

            # Initialize JupiterHelper for live swap execution (if not in dry run)
            if not self.dry_run and JUPITER_HELPER_AVAILABLE:
                try:
                    # Get the decrypted private key for signing (reuse helper)
                    private_key = await self._get_decrypted_private_key()
                    if private_key:
                        self.jupiter_helper = JupiterHelper(
                            solana_rpc_url=self.primary_rpc,
                            private_key=private_key
                        )
                        await self.jupiter_helper.initialize()
                        logger.info("✅ JupiterHelper initialized for LIVE swap execution")
                    else:
                        logger.warning("⚠️ JupiterHelper not available - no private key")
                        self.jupiter_helper = None
                except Exception as e:
                    logger.warning(f"⚠️ JupiterHelper not available for live swaps: {e}")
                    self.jupiter_helper = None

            # SafetyEngine is now initialized in common initialize() method
            # for all strategies, not just Jupiter

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
                        # Get decrypted private key for signing
                        private_key = await self._get_decrypted_private_key()
                        if private_key:
                            self.drift_helper = DriftHelper(
                                rpc_url=self.primary_rpc,
                                private_key=private_key
                            )
                            initialized = await self.drift_helper.initialize()
                            if initialized:
                                logger.info("✅ DriftHelper initialized for LIVE perpetual trading")
                            else:
                                logger.warning("⚠️ DriftHelper initialization failed")
                                self.drift_helper = None
                        else:
                            logger.warning("⚠️ DriftHelper not available - no private key")
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
                    'buy_amount_sol': self.config_manager.get('pumpfun_buy_amount', 0.05),
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
                    'buy_amount_sol': float(os.getenv('PUMPFUN_BUY_AMOUNT_SOL', '0.05')),
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

    async def _get_token_balance(self, token_mint: str, decimals: int = 6) -> float:
        """
        Get actual SPL token balance from wallet.

        Args:
            token_mint: Token mint address
            decimals: Token decimals (default 6 for most SPL tokens)

        Returns:
            Token balance as float, or 0.0 if not found
        """
        try:
            from solders.pubkey import Pubkey
            from solana.rpc.types import TokenAccountOpts

            owner_pubkey = Pubkey.from_string(self.wallet_pubkey)
            mint_pubkey = Pubkey.from_string(token_mint)

            # Get token accounts for this mint
            response = await self.client.get_token_accounts_by_owner(
                owner_pubkey,
                TokenAccountOpts(mint=mint_pubkey)
            )

            if response.value:
                total_balance = 0
                for account in response.value:
                    # Parse the account data to get balance
                    account_data = account.account.data
                    if hasattr(account_data, 'parsed'):
                        # JSON parsed format
                        parsed = account_data.parsed
                        if 'info' in parsed and 'tokenAmount' in parsed['info']:
                            amount = int(parsed['info']['tokenAmount']['amount'])
                            total_balance += amount
                    else:
                        # Raw format - token balance is at offset 64, 8 bytes little endian
                        import base64
                        if isinstance(account_data, str):
                            data = base64.b64decode(account_data)
                        else:
                            data = bytes(account_data)
                        if len(data) >= 72:
                            amount = int.from_bytes(data[64:72], 'little')
                            total_balance += amount

                return total_balance / (10 ** decimals)

            return 0.0

        except Exception as e:
            logger.warning(f"Error getting token balance for {token_mint[:10]}...: {e}")
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
        trend_strength: float = 0.5,
        base_size: float = None
    ) -> float:
        """
        Calculate dynamic position size based on signal strength and market conditions.

        Args:
            signal_strength: Signal confidence (0.0 to 1.0)
            volatility: Market volatility multiplier (1.0 = normal)
            trend_strength: Trend alignment strength (0.0 to 1.0)
            base_size: Override base position size (None = use default)

        Returns:
            Adjusted position size in SOL
        """
        if base_size is None:
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

    async def _load_historical_stats(self):
        """
        Load historical trading stats from database on startup.
        This ensures accurate PnL tracking across restarts.
        """
        if not self.db_pool:
            logger.warning("⚠️ No database pool - historical stats NOT loaded (trades will reset)")
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Load aggregate stats from solana_trades table
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COALESCE(SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END), 0) as winning_trades,
                        COALESCE(SUM(CASE WHEN pnl_sol <= 0 THEN 1 ELSE 0 END), 0) as losing_trades,
                        COALESCE(SUM(pnl_sol), 0) as total_pnl_sol,
                        COALESCE(SUM(fees_sol), 0) as total_fees
                    FROM solana_trades
                """)

                if row:
                    self.total_trades = row['total_trades'] or 0
                    self.winning_trades = row['winning_trades'] or 0
                    self.losing_trades = row['losing_trades'] or 0
                    self.total_pnl_sol = float(row['total_pnl_sol'] or 0)
                    self.total_fees = float(row['total_fees'] or 0)

                    logger.info(f"📊 Loaded historical stats from DB:")
                    logger.info(f"   Total trades: {self.total_trades}")
                    logger.info(f"   Win/Loss: {self.winning_trades}/{self.losing_trades}")
                    logger.info(f"   Total PnL: {self.total_pnl_sol:.4f} SOL")

        except Exception as e:
            logger.warning(f"Could not load historical stats from DB: {e}")
            # Continue with zeroed stats - will accumulate from new trades

    async def _save_trade_to_db(self, trade: Trade):
        """
        Save a closed trade to the database for persistence across restarts.
        This is the CRITICAL method that ensures trades survive container restarts.
        """
        if not self.db_pool:
            logger.warning(f"⚠️ Trade NOT saved to DB (no db_pool): {trade.token_symbol}")
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO solana_trades (
                        trade_id, token_symbol, token_mint, strategy, side,
                        entry_price, exit_price, amount_sol, amount_tokens,
                        pnl_sol, pnl_usd, pnl_pct, fees_sol, exit_reason,
                        entry_time, exit_time, duration_seconds, is_simulated,
                        sol_price_usd, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """,
                    trade.trade_id,  # Fixed: was 'trade.id', correct attribute is 'trade_id'
                    trade.token_symbol,
                    trade.token_mint,
                    trade.strategy.value if hasattr(trade.strategy, 'value') else str(trade.strategy),
                    'long',  # Solana trades are always long
                    trade.entry_price,
                    trade.exit_price,
                    trade.amount,  # This is now value_sol (SOL amount used for trade)
                    getattr(trade, 'amount_tokens', None),  # Token amount if available
                    trade.pnl_sol,
                    trade.pnl_usd if hasattr(trade, 'pnl_usd') else (trade.pnl_sol * self.sol_price_usd),
                    trade.pnl_pct,
                    trade.fees if hasattr(trade, 'fees') else 0,
                    trade.close_reason,
                    trade.opened_at,
                    trade.closed_at,
                    int((trade.closed_at - trade.opened_at).total_seconds()),
                    trade.is_simulated,
                    self.sol_price_usd,
                    None  # metadata - can be extended later
                )
                logger.info(f"💾 Trade saved to DB: {trade.token_symbol} P&L: {trade.pnl_sol:.4f} SOL")
        except Exception as e:
            logger.error(f"Failed to save trade to DB: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
            # CRITICAL: Always monitor positions first, even when trading is paused
            # This ensures SL/TP checks, price updates, and time-based exits still work
            await self._monitor_positions()

            # CRITICAL: Retry stuck positions that failed to close
            # This runs even when trading is paused to ensure we can exit positions
            await self._retry_stuck_positions()

            # Check and reset time-based loss block (AUTO-RELEASE after block_duration_hours)
            if self.risk_metrics.check_and_reset_loss_block():
                logger.info("🔓 Trading resumed after time-based loss block expired")

            # Check risk limits for NEW trades only
            if not self.risk_metrics.can_trade:
                # Provide detailed reason for pause
                pause_reason = []
                if self.risk_metrics.daily_pnl_sol <= -self.risk_metrics.daily_loss_limit_sol:
                    pause_reason.append(f"daily loss limit ({self.risk_metrics.daily_pnl_sol:.4f} <= -{self.risk_metrics.daily_loss_limit_sol:.2f} SOL)")

                # Consecutive loss block with time remaining
                if self.risk_metrics.consecutive_losses >= self.risk_metrics.max_consecutive_losses:
                    remaining = self.risk_metrics.get_block_remaining_seconds()
                    if remaining is not None and remaining > 0:
                        hours = remaining // 3600
                        mins = (remaining % 3600) // 60
                        time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
                        pause_reason.append(f"consecutive losses ({self.risk_metrics.consecutive_losses} >= {self.risk_metrics.max_consecutive_losses}) - auto-reset in {time_str}")
                    else:
                        pause_reason.append(f"consecutive losses ({self.risk_metrics.consecutive_losses} >= {self.risk_metrics.max_consecutive_losses})")

                reason_str = " and ".join(pause_reason) if pause_reason else "unknown"

                # Log error only once every minute to avoid spam
                should_log = False
                if self._last_pause_log_time is None:
                    should_log = True
                elif (datetime.utcnow() - self._last_pause_log_time).total_seconds() > 60:
                    should_log = True

                if should_log:
                    logger.error(f"⚠️ Trading paused - {reason_str}. Positions still being monitored for SL/TP.")
                    self._last_pause_log_time = datetime.utcnow()

                return

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

                # Get strategy-specific SL/TP for display
                display_sl = self.stop_loss_pct
                display_tp = self.take_profit_pct
                if self.config_manager:
                    if position.strategy == Strategy.JUPITER:
                        display_sl = -abs(self.config_manager.jupiter_stop_loss_pct)
                        display_tp = self.config_manager.jupiter_take_profit_pct
                    elif position.strategy == Strategy.PUMPFUN:
                        display_sl = -abs(self.config_manager.pumpfun_stop_loss_pct)
                        display_tp = self.config_manager.pumpfun_take_profit_pct

                # Log position status with strategy-specific SL/TP
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                logger.info(
                    f"{pnl_emoji} {position.token_symbol}: "
                    f"${current_price:.8f} (entry: ${position.entry_price:.8f}), "
                    f"PnL: {pnl_pct:+.2f}%, "
                    f"SL: {display_sl}%, TP: {display_tp}%"
                )

                # Check exit conditions
                exit_reason = await self._check_exit_conditions(position)
                if exit_reason:
                    logger.info(f"🎯 Exit signal for {position.token_symbol}: {exit_reason}")
                    await self._close_position(token_mint, exit_reason)

            except Exception as e:
                logger.error(f"Error monitoring position {token_mint[:8]}...: {e}")

    async def _retry_stuck_positions(self):
        """
        Emergency close mechanism: Retry stuck positions that previously failed to close.

        This runs periodically to attempt closing positions that are tracked by the SafetyEngine
        as stuck (failed to close in previous attempts). Uses escalating slippage.
        """
        if not self.safety_engine:
            return

        # Get positions that are ready to retry
        positions_to_retry = self.safety_engine.get_positions_to_retry()

        if not positions_to_retry:
            return

        logger.info(f"🔄 EMERGENCY CLOSE: Retrying {len(positions_to_retry)} stuck position(s)")

        for stuck in positions_to_retry:
            try:
                token_mint = stuck.token_mint

                # Check if position is still in our active tracking
                # (might have been manually closed or sold externally)
                if token_mint not in self.active_positions:
                    # Check actual wallet balance
                    actual_balance = await self._get_token_balance(token_mint, 6)
                    if actual_balance <= 0:
                        logger.info(f"✅ {stuck.token_symbol} no longer in wallet - removing from stuck list")
                        self.safety_engine.record_close_success(token_mint)
                        continue
                    else:
                        # Token still in wallet but not in our tracking - add to active positions for close
                        logger.warning(f"⚠️ {stuck.token_symbol} found in wallet but not tracked - attempting emergency close")

                logger.info(f"🚨 EMERGENCY CLOSE attempt for {stuck.token_symbol}")
                logger.info(f"   Failure count: {stuck.failure_count}, Last reason: {stuck.last_failure_reason.value}")

                # Get escalating slippage based on failure count
                close_slippage = self.safety_engine.get_slippage_for_strategy(
                    'pumpfun',  # Use highest slippage settings for emergency closes
                    is_close=True,
                    retry_count=stuck.failure_count
                )

                # Get actual token balance
                actual_balance = await self._get_token_balance(token_mint, 6)

                if actual_balance <= 0:
                    logger.info(f"✅ {stuck.token_symbol} balance is 0 - position already closed")
                    self.safety_engine.record_close_success(token_mint)

                    # Also remove from active positions if present
                    if token_mint in self.active_positions:
                        del self.active_positions[token_mint]
                    continue

                token_amount_raw = int(actual_balance * (10 ** 6))

                logger.info(f"   Selling {actual_balance:.2f} tokens with {close_slippage}bps slippage")

                # Attempt the close
                close_tx_signature = await self.jupiter_helper.execute_swap(
                    input_mint=token_mint,
                    output_mint=SOL_MINT,
                    amount=token_amount_raw,
                    slippage_bps=close_slippage
                )

                if close_tx_signature:
                    logger.info(f"🟢 EMERGENCY CLOSE SUCCESS: {stuck.token_symbol}")
                    logger.info(f"   TX: {close_tx_signature}")

                    # Record success
                    self.safety_engine.record_close_success(token_mint)

                    # Remove from active positions if present
                    if token_mint in self.active_positions:
                        position = self.active_positions[token_mint]

                        # Record the trade (even if it's a loss, we want to track it)
                        pnl_pct = -50  # Assume worst case for emergency close
                        pnl_sol = -abs(position.value_sol * 0.5)  # Assume 50% loss

                        trade = Trade(
                            trade_id=str(uuid.uuid4()),
                            token_mint=token_mint,
                            token_symbol=position.token_symbol,
                            strategy=position.strategy,
                            side=TradeSide.SELL,
                            entry_price=position.entry_price,
                            exit_price=position.entry_price * 0.5,  # Estimate
                            amount=position.value_sol,
                            pnl_sol=pnl_sol,
                            pnl_usd=pnl_sol * self.sol_price_usd,
                            pnl_pct=pnl_pct,
                            fees=position.fees_paid,
                            opened_at=position.opened_at,
                            closed_at=datetime.utcnow(),
                            close_reason="emergency_close",
                            is_simulated=position.is_simulated
                        )
                        self.trade_history.append(trade)
                        await self._save_trade_to_db(trade)

                        # Update stats
                        self.total_trades += 1
                        self.total_pnl_sol += pnl_sol
                        self.losing_trades += 1

                        del self.active_positions[token_mint]
                else:
                    logger.warning(f"❌ Emergency close failed for {stuck.token_symbol}")

                    # Record another failure
                    failure_reason = self.safety_engine.parse_error_reason("no signature returned")
                    self.safety_engine.record_close_failure(
                        token_mint=token_mint,
                        token_symbol=stuck.token_symbol,
                        value_sol=stuck.original_value_sol,
                        reason=failure_reason,
                        slippage_used=close_slippage
                    )

            except Exception as e:
                logger.error(f"Error in emergency close for {stuck.token_symbol}: {e}")

                # Record the failure
                if self.safety_engine:
                    failure_reason = self.safety_engine.parse_error_reason(str(e))
                    self.safety_engine.record_close_failure(
                        token_mint=stuck.token_mint,
                        token_symbol=stuck.token_symbol,
                        value_sol=stuck.original_value_sol,
                        reason=failure_reason,
                        slippage_used=self.safety_engine.config.max_emergency_slippage_bps
                    )

            # Small delay between emergency close attempts to avoid rate limits
            await asyncio.sleep(2)

    async def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed using strategy-specific TP/SL with trailing support"""
        pnl_pct = position.unrealized_pnl_pct

        # Get strategy-specific TP/SL from config or use global defaults
        stop_loss_pct = self.stop_loss_pct
        take_profit_pct = self.take_profit_pct

        if self.config_manager:
            if position.strategy == Strategy.JUPITER:
                stop_loss_pct = -abs(self.config_manager.jupiter_stop_loss_pct)
                take_profit_pct = self.config_manager.jupiter_take_profit_pct
            elif position.strategy == Strategy.PUMPFUN:
                # Check if trailing strategy is enabled (default: True for moonbag strategy)
                use_trailing = self.config_manager.get('pumpfun_trailing_enabled', True)
                if use_trailing:
                    return await self._check_pumpfun_trailing_exit(position)
                else:
                    stop_loss_pct = -abs(self.config_manager.pumpfun_stop_loss_pct)
                    take_profit_pct = self.config_manager.pumpfun_take_profit_pct

        # Stop loss
        if pnl_pct <= stop_loss_pct:
            return "stop_loss"

        # Take profit
        if pnl_pct >= take_profit_pct:
            return "take_profit"

        # Time-based exits (use UTC for consistency with opened_at)
        time_held = (datetime.utcnow() - position.opened_at).total_seconds()

        # Jupiter time-based auto exit - IMPROVED: only exit if profitable or past max time
        if position.strategy == Strategy.JUPITER:
            jupiter_auto_exit = 0
            if self.config_manager:
                jupiter_auto_exit = self.config_manager.jupiter_auto_exit_seconds

            if jupiter_auto_exit > 0 and time_held >= jupiter_auto_exit:
                # Only time-exit if:
                # 1. Position is profitable (any amount)
                # 2. OR position has been held 3x the auto_exit time (force exit stale positions)
                max_hold_time = jupiter_auto_exit * 3
                if pnl_pct > 0 or time_held >= max_hold_time:
                    return "jupiter_time_exit"
                # If not profitable and not at max time, continue holding
                # Let the regular SL/TP handle the exit

        # Pump.fun auto-sell delay (only if trailing not enabled)
        if position.strategy == Strategy.PUMPFUN:
            auto_sell_delay = 0
            if self.config_manager:
                auto_sell_delay = self.config_manager.get('pumpfun_auto_sell', 0)

            if auto_sell_delay > 0 and time_held >= auto_sell_delay:
                return "auto_sell_timeout"

        # ============ STALE POSITION / MAX HOLD TIME PROTECTION ============
        # Force-close positions that have been held too long or have stale prices.
        # This prevents positions from blocking new trades indefinitely.

        # 1. Hard maximum hold time for pump.fun (2 hours default)
        if position.strategy == Strategy.PUMPFUN:
            max_hold_seconds = 7200  # 2 hours
            if self.config_manager:
                max_hold_seconds = self.config_manager.get('pumpfun_max_hold_seconds', 7200)
            if time_held >= max_hold_seconds:
                logger.warning(
                    f"⏰ MAX HOLD TIME: {position.token_symbol} held for "
                    f"{time_held/3600:.1f}h (max: {max_hold_seconds/3600:.1f}h) - force closing"
                )
                return "max_hold_time"

        # 2. Stale price detection: if price hasn't moved >1% in 30 minutes, force close
        if position.metadata and 'trailing' in position.metadata:
            trailing_meta = position.metadata['trailing']
            last_price = trailing_meta.get('last_price', 0)
            last_check_time = trailing_meta.get('last_check_time', 0)

            # Track price staleness
            if 'stale_since' not in trailing_meta:
                trailing_meta['stale_since'] = None
                trailing_meta['stale_ref_price'] = None

            if last_price and last_price > 0 and position.current_price > 0:
                price_change_pct = abs((position.current_price - last_price) / last_price) * 100

                # Consider price "stale" if it hasn't moved more than 1%
                if price_change_pct < 1.0:
                    if trailing_meta['stale_since'] is None:
                        trailing_meta['stale_since'] = time.time()
                        trailing_meta['stale_ref_price'] = position.current_price
                    else:
                        # Check if stale for too long (30 minutes default)
                        stale_timeout = 1800  # 30 minutes
                        if self.config_manager:
                            stale_timeout = self.config_manager.get('pumpfun_stale_timeout', 1800)

                        stale_duration = time.time() - trailing_meta['stale_since']
                        # Also check against reference price (not just last 5s)
                        ref_price = trailing_meta.get('stale_ref_price', position.current_price)
                        ref_change = abs((position.current_price - ref_price) / ref_price) * 100 if ref_price > 0 else 0

                        if stale_duration >= stale_timeout and ref_change < 2.0:
                            logger.warning(
                                f"💤 STALE POSITION: {position.token_symbol} price unchanged "
                                f"(~{ref_change:.2f}%) for {stale_duration/60:.0f}min - force closing "
                                f"(PnL: {pnl_pct:+.2f}%)"
                            )
                            return "stale_price_timeout"
                else:
                    # Price moved - reset staleness tracker
                    trailing_meta['stale_since'] = None
                    trailing_meta['stale_ref_price'] = None
        # ============ END STALE POSITION PROTECTION ============

        return None

    async def _check_pumpfun_trailing_exit(self, position: Position) -> Optional[str]:
        """
        Enhanced Pump.fun trailing stop strategy for capturing 10x-100x gains.

        IMPROVED Trailing Tiers (tighter stops, earlier profit-taking):
        - Tier 0 (0-20% gain): SL at -12% (tighter capital protection)
        - Tier 0.5 (20-40% gain): SL at +5% (lock small profit)
        - Tier 1 (40-80% gain): SL at +20%, partial exit 20%
        - Tier 2 (80-150% gain): SL at +40%, partial exit 20%
        - Tier 3 (150-400% gain): SL at 65% of peak, partial exit 20%
        - Tier 4 (400-1000% gain): SL at 75% of peak (25% trail), partial exit 20%
        - Tier 5 (1000%+ gain): SL at 80% of peak (20% trail), partial exit remaining
        """
        entry_price = position.entry_price
        current_price = position.current_price

        if not entry_price or entry_price <= 0:
            return None

        # ============ MAX HOLD TIME CHECK (before trailing logic) ============
        # Force-close pump.fun positions held beyond max time regardless of trailing state
        time_held = (datetime.utcnow() - position.opened_at).total_seconds()
        max_hold_seconds = 7200  # 2 hours default
        if self.config_manager:
            max_hold_seconds = self.config_manager.get('pumpfun_max_hold_seconds', 7200)
        if time_held >= max_hold_seconds:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            logger.warning(
                f"⏰ MAX HOLD TIME: {position.token_symbol} held for "
                f"{time_held/3600:.1f}h (max: {max_hold_seconds/3600:.1f}h), "
                f"PnL: {pnl_pct:+.2f}% - force closing"
            )
            return "max_hold_time"
        # ============ END MAX HOLD TIME CHECK ============

        # Initialize or get metadata
        if not position.metadata:
            position.metadata = {}

        meta = position.metadata
        if 'trailing' not in meta:
            # CRITICAL: Compute initial peak_gain_pct from current price vs entry
            # Previously hardcoded to 0, which meant trailing tiers never activated
            # when price was already above entry at first monitoring check
            initial_peak_gain = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            meta['trailing'] = {
                'peak_price': current_price,
                'peak_gain_pct': max(0, initial_peak_gain),
                'tier_reached': 0,
                'original_amount': position.amount,
                'remaining_pct': 100.0,
                'partial_exits': [],
                'total_realized_pnl': 0.0,
                'last_price': current_price,
                'last_check_time': time.time()
            }

        trailing = meta['trailing']

        # ============ RAPID DECLINE DETECTION (Emergency Exit) ============
        # Check for rapid price crash BEFORE normal trailing stop logic.
        # This catches fast-dying tokens that can drop 50%+ in seconds.
        last_price = trailing.get('last_price', current_price)
        last_check_time = trailing.get('last_check_time', time.time())
        time_since_last_check = time.time() - last_check_time

        # Calculate decline from last check
        if last_price > 0 and time_since_last_check > 0:
            decline_pct = ((last_price - current_price) / last_price) * 100

            # EMERGENCY EXIT CRITERIA:
            # 1. Price dropped >40% since last check (rapid crash)
            # 2. OR price dropped >30% AND we've held for <5 minutes (new position crashing fast)
            time_held = (datetime.utcnow() - position.opened_at).total_seconds()

            if decline_pct >= 40:
                logger.error(f"🚨 RAPID CRASH DETECTED: {position.token_symbol} dropped {decline_pct:.1f}% since last check!")
                logger.error(f"   Price: {last_price:.10f} → {current_price:.10f}")
                logger.error(f"   Triggering EMERGENCY EXIT")

                # Auto-blacklist this token to prevent future trades
                try:
                    if self.scam_blacklist and position.strategy == Strategy.PUMPFUN:
                        logger.info(f"   🚫 Adding {position.token_symbol} to scam blacklist...")
                        await self.scam_blacklist.on_rapid_crash_detected(
                            mint=position.token_mint,
                            symbol=position.token_symbol,
                            drop_pct=decline_pct,
                            time_seconds=int((datetime.utcnow() - position.opened_at).total_seconds())
                        )
                        # Force immediate database sync
                        await self.scam_blacklist._sync_to_db()
                        logger.info(f"   ✅ {position.token_symbol} blacklisted and synced to DB")
                    else:
                        if not self.scam_blacklist:
                            logger.warning(f"   ⚠️ Blacklist not initialized")
                        else:
                            logger.info(f"   ℹ️ Non-pumpfun strategy ({position.strategy}), not blacklisting")
                except Exception as e:
                    logger.error(f"   ❌ Failed to blacklist {position.token_symbol}: {e}")

                return "emergency_rapid_decline"

            if decline_pct >= 30 and time_held < 300:  # 5 minutes
                logger.error(f"🚨 NEW POSITION CRASHING: {position.token_symbol} dropped {decline_pct:.1f}% in {time_held:.0f}s!")
                logger.error(f"   Triggering EMERGENCY EXIT for new position rapid decline")

                # Auto-blacklist this token to prevent future trades
                try:
                    if self.scam_blacklist and position.strategy == Strategy.PUMPFUN:
                        logger.info(f"   🚫 Adding {position.token_symbol} to scam blacklist...")
                        await self.scam_blacklist.on_rapid_crash_detected(
                            mint=position.token_mint,
                            symbol=position.token_symbol,
                            drop_pct=decline_pct,
                            time_seconds=int(time_held)
                        )
                        # Force immediate database sync
                        await self.scam_blacklist._sync_to_db()
                        logger.info(f"   ✅ {position.token_symbol} blacklisted and synced to DB")
                    else:
                        if not self.scam_blacklist:
                            logger.warning(f"   ⚠️ Blacklist not initialized")
                        else:
                            logger.info(f"   ℹ️ Non-pumpfun strategy ({position.strategy}), not blacklisting")
                except Exception as e:
                    logger.error(f"   ❌ Failed to blacklist {position.token_symbol}: {e}")

                return "emergency_rapid_decline"

            # Also check decline from peak (not just last check)
            # If we've dropped more than 60% from peak, emergency exit
            peak_price = trailing.get('peak_price', current_price)
            if peak_price > 0:
                decline_from_peak = ((peak_price - current_price) / peak_price) * 100
                if decline_from_peak >= 60:
                    logger.error(f"🚨 SEVERE PEAK DECLINE: {position.token_symbol} down {decline_from_peak:.1f}% from peak!")
                    logger.error(f"   Peak: {peak_price:.10f} → Current: {current_price:.10f}")
                    return "emergency_peak_decline"

        # Update last price tracking for next check
        trailing['last_price'] = current_price
        trailing['last_check_time'] = time.time()
        # ============ END RAPID DECLINE DETECTION ============

        # ============ STALE PRICE DETECTION (for trailing-enabled positions) ============
        # If price hasn't moved significantly in 30+ minutes, the token is likely dead/abandoned.
        # Force close to free up position slots for new opportunities.
        if 'stale_since' not in trailing:
            trailing['stale_since'] = None
            trailing['stale_ref_price'] = None

        if last_price and last_price > 0 and current_price > 0:
            price_change_pct = abs((current_price - last_price) / last_price) * 100

            if price_change_pct < 1.0:
                if trailing['stale_since'] is None:
                    trailing['stale_since'] = time.time()
                    trailing['stale_ref_price'] = current_price
                else:
                    stale_timeout = 1800  # 30 minutes default
                    if self.config_manager:
                        stale_timeout = self.config_manager.get('pumpfun_stale_timeout', 1800)

                    stale_duration = time.time() - trailing['stale_since']
                    ref_price = trailing.get('stale_ref_price', current_price)
                    ref_change = abs((current_price - ref_price) / ref_price) * 100 if ref_price > 0 else 0

                    if stale_duration >= stale_timeout and ref_change < 2.0:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.warning(
                            f"💤 STALE POSITION: {position.token_symbol} price unchanged "
                            f"(~{ref_change:.2f}%) for {stale_duration/60:.0f}min - force closing "
                            f"(PnL: {pnl_pct:+.2f}%)"
                        )
                        return "stale_price_timeout"
            else:
                trailing['stale_since'] = None
                trailing['stale_ref_price'] = None
        # ============ END STALE PRICE DETECTION ============

        # Update peak price
        if current_price > trailing['peak_price']:
            trailing['peak_price'] = current_price
            trailing['peak_gain_pct'] = ((current_price - entry_price) / entry_price) * 100

        peak_price = trailing['peak_price']
        current_gain_pct = ((current_price - entry_price) / entry_price) * 100
        peak_gain_pct = trailing['peak_gain_pct']

        # Get configurable tier 0 stop loss from config (default -20% for pump.fun volatility)
        # TUNED: Pump.fun tokens can swing 15-20% before recovering
        tier0_sl_pct = 20.0  # Was 12%, increased to handle volatility
        if self.config_manager:
            tier0_sl_pct = self.config_manager.pumpfun_tier0_sl

        # Time-based stop widening: In first 2 minutes, use wider SL to handle initial volatility
        time_held = (datetime.utcnow() - position.opened_at).total_seconds()
        early_volatility_window = 120  # 2 minutes
        if time_held < early_volatility_window:
            # Use slightly wider SL in early period (add 5% buffer)
            effective_tier0_sl = tier0_sl_pct + 5.0
        else:
            effective_tier0_sl = tier0_sl_pct

        # Calculate dynamic trailing stop level - TUNED for pump.fun volatility
        if peak_gain_pct < 20:
            # Tier 0: Initial protection with time-based adjustment
            sl_price = entry_price * (1 - effective_tier0_sl / 100)
            sl_gain_pct = -effective_tier0_sl
        elif peak_gain_pct < 40:
            # Tier 0.5: Lock small profit early (+5%)
            sl_price = entry_price * 1.05
            sl_gain_pct = 5.0
        elif peak_gain_pct < 80:
            # Tier 1: Lock +20% (moved up from breakeven)
            sl_price = entry_price * 1.20
            sl_gain_pct = 20.0
        elif peak_gain_pct < 150:
            # Tier 2: Lock +40%
            sl_price = entry_price * 1.40
            sl_gain_pct = 40.0
        elif peak_gain_pct < 400:
            # Tier 3: Trail 35% below peak (tighter than 50%)
            sl_price = peak_price * 0.65
            sl_gain_pct = ((sl_price - entry_price) / entry_price) * 100
        elif peak_gain_pct < 1000:
            # Tier 4: Trail 25% below peak
            sl_price = peak_price * 0.75
            sl_gain_pct = ((sl_price - entry_price) / entry_price) * 100
        else:
            # Tier 5: Moon mode - trail only 20% below peak (tightest)
            sl_price = peak_price * 0.80
            sl_gain_pct = ((sl_price - entry_price) / entry_price) * 100

        # Store current SL level
        trailing['current_sl_price'] = sl_price
        trailing['current_sl_pct'] = sl_gain_pct

        # Check for trailing stop hit
        if current_price <= sl_price:
            tier = trailing['tier_reached']
            if tier == 0:
                return "trailing_stop_tier0"
            elif tier == 1:
                return "trailing_stop_tier1"
            elif tier == 2:
                return "trailing_stop_tier2"
            elif tier == 3:
                return "trailing_stop_tier3"
            elif tier == 4:
                return "trailing_stop_tier4"
            else:
                return "trailing_stop_moon"

        # Check for partial exit triggers (only trigger once per tier)
        # IMPROVED: Earlier and more granular partial exits to capture gains
        tier_reached = trailing['tier_reached']

        # Tier 1: First partial at 40% (moved from 50%)
        if peak_gain_pct >= 40 and tier_reached < 1:
            trailing['tier_reached'] = 1
            logger.info(f"🎯 {position.token_symbol}: Tier 1 reached (+40%), partial exit 20%")
            return "partial_exit_tier1"

        # Tier 2: Second partial at 80%
        if peak_gain_pct >= 80 and tier_reached < 2:
            trailing['tier_reached'] = 2
            logger.info(f"🚀 {position.token_symbol}: Tier 2 reached (+80%), partial exit 20%")
            return "partial_exit_tier2"

        # Tier 3: Third partial at 150%
        if peak_gain_pct >= 150 and tier_reached < 3:
            trailing['tier_reached'] = 3
            logger.info(f"💎 {position.token_symbol}: Tier 3 reached (+150%), partial exit 20%")
            return "partial_exit_tier3"

        # Tier 4: Fourth partial at 400% (5x)
        if peak_gain_pct >= 400 and tier_reached < 4:
            trailing['tier_reached'] = 4
            logger.info(f"🌙 {position.token_symbol}: Tier 4 reached (+400%), partial exit 20%")
            return "partial_exit_tier4"

        # Tier 5: Fifth partial at 1000% (10x) - MOON MODE
        if peak_gain_pct >= 1000 and tier_reached < 5:
            trailing['tier_reached'] = 5
            logger.info(f"🔥 {position.token_symbol}: TIER 5 MOON MODE (+1000%), final partial exit!")
            return "partial_exit_tier5"

        # Log trailing status periodically (every 30 seconds)
        time_held = (datetime.utcnow() - position.opened_at).total_seconds()
        if int(time_held) % 30 == 0 and current_gain_pct > 10:
            logger.info(
                f"📊 {position.token_symbol}: Gain={current_gain_pct:.1f}% Peak={peak_gain_pct:.1f}% "
                f"SL={sl_gain_pct:.1f}% Tier={tier_reached}"
            )

        return None

    async def _scan_jupiter_opportunities(self):
        """Scan for Jupiter swap opportunities"""
        # Use Jupiter-specific max positions if configured, else general max
        jupiter_max_pos = self.config_manager.jupiter_max_positions if self.config_manager else self.max_positions
        jupiter_positions = sum(1 for p in self.active_positions.values() if p.strategy == Strategy.JUPITER)

        if jupiter_positions >= jupiter_max_pos:
            logger.debug(f"Jupiter max positions ({jupiter_max_pos}) reached, skipping scan")
            return

        if len(self.active_positions) >= self.max_positions:
            logger.debug(f"Total max positions ({self.max_positions}) reached, skipping Jupiter scan")
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
                    # Use Jupiter-specific position size from config
                    jupiter_base_size = self.config_manager.jupiter_position_size_sol if self.config_manager else self.position_size_sol
                    await self._open_position(
                        token_mint=token_mint,
                        token_symbol=token_name,
                        strategy=Strategy.JUPITER,
                        amount_sol=self.calculate_dynamic_position_size(
                            signal_strength=min(1.0, abs(momentum) / 2.0 + 0.3),
                            volatility=1.0,
                            trend_strength=0.7,
                            base_size=jupiter_base_size
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
        # Count pump.fun positions separately from other strategies
        # This allows pump.fun to have dedicated position slots
        pumpfun_positions = sum(1 for p in self.active_positions.values() if p.strategy == Strategy.PUMPFUN)
        other_positions = len(self.active_positions) - pumpfun_positions

        # Pump.fun has its own limit from DB config (default 3 positions)
        pumpfun_max = 2  # Default fallback (reduced to limit exposure)
        if self.config_manager:
            pumpfun_max = self.config_manager.pumpfun_max_positions

        # Skip if pump.fun slots are full
        if pumpfun_positions >= pumpfun_max:
            logger.debug(f"Pump.fun position limit reached ({pumpfun_positions}/{pumpfun_max}), skipping scan")
            return

        # Also respect overall position limit but with some flexibility for pump.fun
        # Allow pump.fun if total is at limit but no pump.fun positions exist
        if len(self.active_positions) >= self.max_positions and pumpfun_positions > 0:
            logger.debug(f"Max positions reached ({len(self.active_positions)}/{self.max_positions}), skipping Pump.fun scan")
            return

        if not self.pumpfun_monitor:
            return

        try:
            # Get new tokens (already filtered in get_new_tokens)
            new_tokens = await self.pumpfun_monitor.get_new_tokens()

            # Track positions opened this scan
            positions_opened = 0
            max_per_scan = 3  # Max positions to open per scan cycle (prevent overwhelming)

            for token in new_tokens:
                # Check if we've hit limits
                current_pumpfun = sum(1 for p in self.active_positions.values() if p.strategy == Strategy.PUMPFUN)
                if current_pumpfun >= pumpfun_max:
                    logger.debug(f"Pump.fun max positions reached ({current_pumpfun}/{pumpfun_max})")
                    break

                if len(self.active_positions) >= self.max_positions:
                    logger.debug(f"Overall max positions reached ({len(self.active_positions)}/{self.max_positions})")
                    break

                if positions_opened >= max_per_scan:
                    logger.debug(f"Max positions per scan reached ({positions_opened}/{max_per_scan})")
                    break

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

                # Open position - returns True if successful
                success = await self._open_position(
                    token_mint=token_mint,
                    token_symbol=token.get('symbol', 'UNKNOWN'),
                    strategy=Strategy.PUMPFUN,
                    amount_sol=self.pumpfun_monitor.buy_amount_sol,
                    metadata=token
                )

                if success:
                    positions_opened += 1
                    logger.info(f"   ✅ Position {positions_opened} opened successfully")

            if positions_opened > 0:
                logger.info(f"📊 Opened {positions_opened} new position(s) this scan")

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
    ) -> bool:
        """
        Open a new position.

        Returns:
            True if position was successfully opened, False otherwise
        """
        try:
            # ============ PRE-BUY SAFETY CHECKS ============

            # 1. Block UNKNOWN tokens - high scam probability
            if not token_symbol or token_symbol.upper() == 'UNKNOWN':
                logger.warning(f"🚫 BLOCKED: Token has no symbol/UNKNOWN - likely scam")
                return False

            # 2. Block tokens with very suspicious symbols (single char, only numbers)
            if len(token_symbol) < 2 or token_symbol.isdigit():
                logger.warning(f"🚫 BLOCKED: {token_symbol} has suspicious symbol format")
                return False

            # 3. Check if token is blacklisted (scam/rug detected previously)
            if self.scam_blacklist and self.scam_blacklist.is_blacklisted(token_mint):
                reason = self.scam_blacklist.get_blacklist_reason(token_mint)
                logger.warning(f"🚫 BLOCKED: {token_symbol} is blacklisted - {reason}")
                return False

            # 4. Check token name for scam patterns (TRUMP, BIDEN, etc.)
            # Only check if scam detection is enabled in config
            scam_detection_enabled = True  # Default to enabled for safety
            if self.config_manager:
                scam_detection_enabled = self.config_manager.pumpfun_scam_detection

            if self.scam_blacklist and scam_detection_enabled:
                is_scam_name, pattern = self.scam_blacklist.check_name_pattern(
                    token_symbol,
                    metadata.get('name') if metadata else None
                )
                if is_scam_name:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} matches scam name pattern: {pattern}")
                    await self.scam_blacklist.add_to_blacklist(
                        mint=token_mint,
                        symbol=token_symbol,
                        reason=f"Scam name pattern: {pattern}",
                        metadata={'pattern': pattern, 'type': 'name_pattern'}
                    )
                    # Force immediate database sync to persist blacklist
                    await self.scam_blacklist._sync_to_db()
                    logger.info(f"   ✅ {token_symbol} blacklisted and synced to DB")
                    return False

            # 5. For pump.fun: Check holder count if available (require minimum holders)
            # Get configurable thresholds from config manager
            min_holders = 15  # Default
            max_dev_holding = 10.0  # Default
            if self.config_manager:
                min_holders = self.config_manager.pumpfun_min_holders
                max_dev_holding = self.config_manager.pumpfun_max_dev_holding

            if strategy == Strategy.PUMPFUN and metadata:
                holder_count = metadata.get('holder_count', metadata.get('holders', 0))
                if holder_count and holder_count < min_holders:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} has only {holder_count} holders (min: {min_holders})")
                    return False

                # Check dev holding percentage
                dev_holding = metadata.get('dev_holding_pct', metadata.get('creator_holdings', 0))
                if dev_holding and dev_holding > max_dev_holding:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} dev holds {dev_holding:.1f}% (max: {max_dev_holding}%)")
                    return False

                # ENHANCED: Check market cap - avoid extremely low mcap (likely scam) or absurdly high
                market_cap = metadata.get('market_cap', metadata.get('fdv', 0))
                if market_cap and market_cap < 5000:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} market cap ${market_cap:.0f} too low (min: $5000)")
                    return False
                if market_cap and market_cap > 10_000_000:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} market cap ${market_cap:,.0f} too high for pump.fun entry")
                    return False

                # ENHANCED: Check liquidity depth - require at least $3000
                liq_usd = metadata.get('liquidity_usd', metadata.get('liquidity', 0))
                if liq_usd and liq_usd < 3000:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} liquidity ${liq_usd:.0f} too low (min: $3000)")
                    return False

                # ENHANCED: Reject tokens with negative price momentum
                # Tightened from -20% to -10%: if price is already declining, avoid entry
                price_change = metadata.get('price_change_24h', metadata.get('priceChange', 0))
                if price_change and price_change < -10:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} price already dropping {price_change:.1f}% - avoiding downtrend")
                    return False

                # ENHANCED: Check volume relative to liquidity - require active trading
                vol = metadata.get('volume_24h', metadata.get('volume', 0))
                liq = metadata.get('liquidity_usd', metadata.get('liquidity', 0))
                if vol and liq and liq > 0 and vol / liq < 0.5:
                    logger.warning(f"🚫 BLOCKED: {token_symbol} vol/liq ratio {vol/liq:.2f} too low - token may be dying")
                    return False

            # ============ END PRE-BUY SAFETY CHECKS ============

            # Get token price - try multiple sources
            current_price = None

            # 1. First try to use price from metadata (already fetched from DexScreener/pump.fun)
            if metadata:
                current_price = metadata.get('price') or metadata.get('priceUsd') or metadata.get('price_usd')
                if current_price:
                    try:
                        current_price = float(current_price)
                        if current_price > 0:
                            logger.debug(f"   Using price from metadata: ${current_price:.10f}")
                    except (ValueError, TypeError):
                        current_price = None

            # 2. If no price from metadata, try DexScreener API
            if not current_price:
                try:
                    async with aiohttp.ClientSession() as session:
                        price_data = await self._get_token_price_data(session, token_mint)
                        if price_data and price_data.get('price'):
                            current_price = float(price_data['price'])
                            logger.debug(f"   Using price from DexScreener: ${current_price:.10f}")
                except Exception as e:
                    logger.debug(f"   DexScreener price fetch failed: {e}")

            # 3. Fall back to Jupiter for established tokens
            if not current_price:
                current_price = await self._get_token_price(token_mint)
                if current_price:
                    logger.debug(f"   Using price from Jupiter: ${current_price:.10f}")

            if current_price is None or current_price <= 0:
                logger.warning(f"Could not get price for {token_symbol} from any source")
                return False

            # Calculate values
            value_usd = amount_sol * self.sol_price_usd
            token_amount = (amount_sol * self.sol_price_usd) / current_price if current_price > 0 else 0

            # Get strategy-specific TP/SL from config or use global defaults
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct

            if self.config_manager:
                if strategy == Strategy.JUPITER:
                    stop_loss_pct = -abs(self.config_manager.jupiter_stop_loss_pct)
                    take_profit_pct = self.config_manager.jupiter_take_profit_pct
                    logger.info(f"   Jupiter TP/SL: TP={take_profit_pct}%, SL={stop_loss_pct}%")
                elif strategy == Strategy.PUMPFUN:
                    stop_loss_pct = -abs(self.config_manager.pumpfun_stop_loss_pct)
                    take_profit_pct = self.config_manager.pumpfun_take_profit_pct
                    logger.info(f"   Pump.fun TP/SL: TP={take_profit_pct}%, SL={stop_loss_pct}%")

            # Create position with strategy-specific TP/SL
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
                stop_loss=current_price * (1 + stop_loss_pct / 100),
                take_profit=current_price * (1 + take_profit_pct / 100),
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
                # Check wallet balance before executing swap
                try:
                    wallet_balance = await self._get_wallet_balance()
                    # Need enough SOL for swap amount plus gas (0.01 SOL buffer)
                    min_required = amount_sol + 0.01
                    if wallet_balance < min_required:
                        logger.error(f"❌ Insufficient balance: {wallet_balance:.4f} SOL < {min_required:.4f} SOL required")
                        logger.error(f"   Please fund your wallet before live trading")
                        return False
                except Exception as e:
                    logger.warning(f"⚠️ Could not check wallet balance: {e}")

                # Execute real swap based on strategy
                try:
                    # Check circuit breaker before any new trade
                    if self.safety_engine:
                        is_tripped, seconds_remaining = self.safety_engine.check_circuit_breaker()
                        if is_tripped:
                            logger.warning(f"🚨 Circuit breaker ACTIVE - skipping trade ({seconds_remaining}s remaining)")
                            return False

                    # Check if this is a pump.fun token that hasn't graduated
                    # Pump.fun tokens without Raydium routes can't be traded via Jupiter
                    is_pumpfun_only = False
                    if strategy == Strategy.PUMPFUN:
                        # Check if token has graduated by attempting a quote
                        if self.jupiter_client:
                            test_quote = await self.jupiter_client.get_quote(
                                input_mint=SOL_MINT,
                                output_mint=token_mint,
                                amount=int(0.001 * self.LAMPORTS_PER_SOL)  # Small test amount
                            )
                            if not test_quote:
                                is_pumpfun_only = True
                                logger.warning(f"⚠️ {token_symbol} is a pump.fun-only token (no Jupiter route)")
                                logger.warning(f"   Token has not graduated to Raydium yet")
                                logger.warning(f"   Pump.fun bonding curve trading not yet implemented")
                                logger.info(f"   Skipping live trade - monitor for graduation to Raydium")
                                return False

                    # CRITICAL: Verify SELL route exists before buying (honeypot detection)
                    # This prevents buying tokens that can't be sold back
                    if self.safety_engine and self.jupiter_client:
                        logger.info(f"🔍 Verifying sell route for {token_symbol}...")
                        # Get decimals from metadata or use default (9 for most modern tokens)
                        token_decimals = metadata.get('decimals', 9) if metadata else 9
                        can_sell, reason = await self.safety_engine.verify_sell_route(
                            self.jupiter_client,
                            token_mint,
                            SOL_MINT,
                            test_amount=1000000,  # Base test amount (may be adjusted by price)
                            token_price_usd=current_price,
                            token_decimals=token_decimals
                        )
                        if not can_sell:
                            logger.error(f"🚫 HONEYPOT DETECTED: {token_symbol}")
                            logger.error(f"   Reason: {reason}")
                            logger.error(f"   Token: {token_mint}")
                            logger.warning(f"   Skipping buy - cannot verify sell route")
                            return False
                        logger.info(f"✅ Sell route verified for {token_symbol}: {reason}")

                    # Get appropriate slippage for strategy
                    if self.safety_engine:
                        trade_slippage = self.safety_engine.get_slippage_for_strategy(
                            strategy.value, is_close=False
                        )
                    else:
                        trade_slippage = self.slippage_bps
                        # Fallback: use MUCH higher slippage for pump.fun
                        # 50 bps is far too low - causes constant 0x1771 SlippageToleranceExceeded errors
                        if strategy == Strategy.PUMPFUN:
                            trade_slippage = max(800, self.slippage_bps * 10)  # Min 8% for pump.fun buys
                        else:
                            trade_slippage = max(200, self.slippage_bps)  # Min 2% for other strategies

                    if self.jupiter_helper:
                        # Use JupiterHelper for full swap execution
                        logger.info(f"🔄 Executing LIVE swap: {amount_sol} SOL → {token_symbol} (slippage: {trade_slippage}bps)")
                        tx_signature = await self.jupiter_helper.execute_swap(
                            input_mint=SOL_MINT,
                            output_mint=token_mint,
                            amount=int(amount_sol * self.LAMPORTS_PER_SOL),
                            slippage_bps=trade_slippage
                        )

                        if tx_signature:
                            logger.info(f"🟢 LIVE SWAP executed: {tx_signature}")
                            position.tx_signature = tx_signature

                            # Update position.amount with actual tokens received
                            # This ensures accurate tracking for closes/partial exits
                            # CRITICAL: Retry with increasing delays - 1s was often too short
                            # for RPC to reflect new balance, causing phantom balance tracking
                            try:
                                actual_tokens = 0
                                for balance_retry in range(3):
                                    delay = 1.5 + balance_retry * 1.0  # 1.5s, 2.5s, 3.5s
                                    await asyncio.sleep(delay)
                                    actual_tokens = await self._get_token_balance(token_mint, 6)
                                    if actual_tokens > 0:
                                        break
                                    logger.debug(f"Balance check attempt {balance_retry+1}: 0 tokens, retrying...")

                                if actual_tokens > 0:
                                    if position.amount > 0:
                                        diff_pct = abs(actual_tokens - position.amount) / position.amount * 100
                                        if diff_pct > 5:
                                            logger.info(f"📊 Token amount adjusted: estimated {position.amount:.2f} → actual {actual_tokens:.2f} ({diff_pct:.1f}% diff)")
                                    position.amount = actual_tokens
                                else:
                                    logger.warning(f"⚠️ Could not verify token balance after 3 attempts - using estimated amount {position.amount:.2f}")
                            except Exception as e:
                                logger.debug(f"Could not verify token balance: {e}")
                        else:
                            logger.error("❌ Jupiter swap failed - check logs above for specific RPC error")
                            logger.error(f"   Token: {token_symbol} ({token_mint})")
                            logger.error(f"   Amount: {amount_sol} SOL, Slippage: {self.slippage_bps} bps")
                            if strategy == Strategy.PUMPFUN:
                                logger.error("   Pump.fun token may not have graduated to Raydium")
                            logger.error("   Common causes: insufficient SOL, high slippage, RPC issues")
                            return False
                    else:
                        # Fallback: get quote but warn about no execution
                        quote = await self.jupiter_client.get_quote(
                            input_mint=SOL_MINT,
                            output_mint=token_mint,
                            amount=int(amount_sol * self.LAMPORTS_PER_SOL)
                        )

                        if not quote:
                            logger.error("Failed to get Jupiter quote")
                            if strategy == Strategy.PUMPFUN:
                                logger.error("   Token may be pump.fun-only (not yet graduated to Raydium)")
                            return False

                        logger.warning(f"⚠️ JupiterHelper not available - swap NOT executed (quote only)")
                        logger.warning(f"   Quote output: {quote.get('outAmount', 'N/A')} lamports")
                        return False  # Don't open position without actual swap

                except Exception as e:
                    logger.error(f"❌ Swap execution failed: {e}", exc_info=True)
                    return False

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

            # Send Telegram entry alert
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from solana_trading.core.solana_alerts import SolanaTradeAlert
                    alert = SolanaTradeAlert(
                        token_symbol=token_symbol,
                        token_mint=token_mint,
                        strategy=strategy.value,
                        action='entry',
                        entry_price=current_price,
                        amount_sol=amount_sol,
                        token_amount=position.amount,
                        stop_loss_pct=abs(position.stop_loss) if position.stop_loss else None,
                        take_profit_pct=position.take_profit,
                        is_simulated=self.dry_run,
                        sol_price_usd=self.sol_price_usd
                    )
                    await self.telegram_alerts.send_entry_alert(alert)
                except Exception as e:
                    logger.debug(f"Telegram entry alert failed: {e}")

            return True  # Position successfully opened

        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False

    async def _close_position(self, token_mint: str, reason: str):
        """Close a position (full or partial)"""
        if token_mint not in self.active_positions:
            return

        position = self.active_positions[token_mint]

        try:
            # Determine if this is a partial exit
            is_partial = reason.startswith("partial_exit_")
            partial_pct = 0.20  # Default 20% partial exit per tier

            # Adjust partial percentage based on tier
            if reason == "partial_exit_tier5":
                # Final tier - exit remaining position
                partial_pct = 1.0  # Full remaining
                is_partial = False  # Treat as full close
            elif is_partial:
                # Tiers 1-4: 20% each
                partial_pct = 0.20

            # Calculate amounts based on partial or full exit
            if is_partial:
                # Calculate partial amounts
                close_pct = partial_pct
                close_amount = position.amount * close_pct
                close_value_sol = position.value_sol * close_pct
            else:
                # Full close
                close_pct = 1.0
                close_amount = position.amount
                close_value_sol = position.value_sol

            # Calculate PnL for the portion being closed
            pnl_pct = position.unrealized_pnl_pct
            pnl_sol = position.unrealized_pnl * close_pct
            pnl_usd = pnl_sol * self.sol_price_usd

            # Execute close (or simulate)
            close_tx_signature = None
            close_actually_executed = False  # Track if close actually succeeded

            if self.dry_run:
                partial_tag = f" ({close_pct*100:.0f}%)" if is_partial else ""
                logger.info(f"🔵 [DRY_RUN] SIMULATED SELL{partial_tag} {position.token_symbol} ({reason})")
                close_tx_signature = f"DRY_RUN_CLOSE_{uuid.uuid4().hex[:16]}"
                close_actually_executed = True
            else:
                # Execute real swap back to SOL with RETRY LOGIC
                if not self.jupiter_helper:
                    logger.warning(f"⚠️ JupiterHelper not available - close NOT executed")
                    logger.warning(f"   Position {position.token_symbol} needs manual close!")
                    return  # Don't record failed close

                # CRITICAL: Use ACTUAL wallet balance, not estimated position.amount!
                token_decimals = 6  # Default for most SPL tokens
                actual_balance = await self._get_token_balance(token_mint, token_decimals)

                if actual_balance <= 0:
                    logger.warning(f"⚠️ No tokens found in wallet for {position.token_symbol}")
                    logger.warning(f"   Expected: {close_amount:.6f}, Actual: {actual_balance:.6f}")
                    # This is a phantom position - close it in tracking only
                    close_actually_executed = True
                    close_tx_signature = "PHANTOM_POSITION_CLEANUP"
                else:
                    # Log discrepancy if significant (>5% difference)
                    if position.amount > 0:
                        discrepancy = abs(actual_balance - position.amount) / position.amount
                        if discrepancy > 0.05:
                            logger.info(f"📊 Balance discrepancy: tracked={position.amount:.2f}, actual={actual_balance:.2f}")

                    # For full close: sell ALL actual tokens
                    # For partial: sell percentage of actual balance
                    if is_partial:
                        sell_amount = actual_balance * close_pct
                        logger.info(f"🔄 Partial close {close_pct*100:.0f}%: selling {sell_amount:.2f} of {actual_balance:.2f} tokens")
                    else:
                        sell_amount = actual_balance  # Sell ALL tokens
                        logger.info(f"🔄 Full close: selling ALL {actual_balance:.2f} tokens")

                    # Update close_amount to actual amount being sold
                    close_amount = sell_amount
                    token_amount_raw = int(close_amount * (10 ** token_decimals))

                    # RETRY LOOP with escalating slippage
                    max_retries = 7 if self.safety_engine else 4  # More retries for stuck positions
                    last_error = None
                    program_mismatch_count = 0  # Track 0x1788 errors

                    # Detect emergency exits - use higher slippage immediately
                    is_emergency_exit = reason.startswith("emergency_")

                    for retry in range(max_retries):
                        try:
                            # If we've seen 3+ ProgramMismatch errors, token is likely unsellable
                            if program_mismatch_count >= 3:
                                logger.error(f"🚨 {position.token_symbol}: 3x ProgramMismatch (0x1788) - token likely unsellable via Jupiter")
                                logger.error(f"   Liquidity may have been removed (rug pull) or pool migrated")
                                last_error = "ProgramMismatch (0x1788) - token unsellable"
                                break

                            # Get slippage for this retry attempt
                            if self.safety_engine:
                                close_slippage = self.safety_engine.get_slippage_for_strategy(
                                    position.strategy.value,
                                    is_close=True,
                                    retry_count=retry,
                                    is_emergency=is_emergency_exit
                                )
                            else:
                                # Fallback: escalate slippage manually (increased defaults)
                                base_slippage = 1200 if position.strategy == Strategy.PUMPFUN else 200
                                if is_emergency_exit:
                                    base_slippage = int(base_slippage * 1.5)  # 50% more for emergencies
                                close_slippage = base_slippage + (retry * 400)
                                close_slippage = min(close_slippage, 3500)  # Cap at 35%

                            # After ProgramMismatch, use restricted intermediate tokens
                            # to force Jupiter to pick a different, safer route
                            use_restricted_routes = program_mismatch_count > 0

                            emoji = "🚨" if is_emergency_exit else "🔄"
                            route_tag = " [restricted routes]" if use_restricted_routes else ""
                            logger.info(f"{emoji} Close attempt {retry + 1}/{max_retries}: {position.token_symbol} → SOL (slippage: {close_slippage}bps{route_tag})")

                            close_tx_signature = await self.jupiter_helper.execute_swap(
                                input_mint=token_mint,
                                output_mint=SOL_MINT,
                                amount=token_amount_raw,
                                slippage_bps=close_slippage,
                                restrict_intermediate_tokens=use_restricted_routes
                            )

                            if close_tx_signature:
                                logger.info(f"🟢 LIVE CLOSE executed: {close_tx_signature}")

                                # CRITICAL: Verify tokens were actually sold by checking balance
                                # A tx can appear "confirmed" but fail on-chain (e.g., 0x1771 slippage error)
                                # The confirm_transaction fix catches most cases, but this is a safety net
                                # Use multiple balance checks with increasing delays (RPC can be slow to reflect)
                                sell_verified = False
                                for verify_attempt in range(3):
                                    wait = 2.0 + verify_attempt * 2.0  # 2s, 4s, 6s
                                    await asyncio.sleep(wait)
                                    remaining_balance = await self._get_token_balance(token_mint, token_decimals)

                                    if is_partial:
                                        # For partial close, check that balance decreased
                                        if remaining_balance <= actual_balance * 0.95:
                                            sell_verified = True
                                            break
                                    else:
                                        # For full close, tokens should be gone
                                        if remaining_balance <= actual_balance * 0.05:
                                            sell_verified = True
                                            break

                                    if verify_attempt < 2:
                                        logger.debug(f"Balance check {verify_attempt+1}: {remaining_balance:.2f} (was {actual_balance:.2f}), retrying...")

                                if not sell_verified:
                                    if is_partial:
                                        logger.error(f"❌ POST-SELL VERIFICATION FAILED for {position.token_symbol}")
                                        logger.error(f"   Balance unchanged: {remaining_balance:.2f} (was {actual_balance:.2f})")
                                        logger.error(f"   TX {close_tx_signature} likely FAILED on-chain!")
                                    else:
                                        logger.error(f"❌ POST-SELL VERIFICATION FAILED for {position.token_symbol}")
                                        logger.error(f"   Tokens still in wallet: {remaining_balance:.2f} (was {actual_balance:.2f})")
                                        logger.error(f"   TX {close_tx_signature} likely FAILED on-chain!")
                                    close_tx_signature = None
                                    last_error = "Post-sell balance verification failed - tokens not sold"
                                    continue  # Retry with higher slippage

                                logger.info(f"✅ Post-sell verification OK: balance {remaining_balance:.2f} (was {actual_balance:.2f})")
                                close_actually_executed = True

                                # Record success with safety engine
                                if self.safety_engine:
                                    self.safety_engine.record_close_success(token_mint)
                                break  # Success!
                            else:
                                last_error = "No transaction signature returned"
                                logger.warning(f"⚠️ Close attempt {retry + 1} failed - no signature")
                                # Detect ProgramMismatch (0x1788) from Jupiter's last error
                                # This error means the route's DEX program is invalid/removed
                                jup_error = getattr(self.jupiter_helper, 'last_swap_error', None)
                                if jup_error and '0x1788' in str(jup_error):
                                    program_mismatch_count += 1
                                    last_error = f"ProgramMismatch (0x1788): {jup_error}"
                                    logger.warning(f"   ⚠️ ProgramMismatch detected ({program_mismatch_count}x) - route/DEX invalid")

                        except Exception as e:
                            last_error = str(e)
                            logger.warning(f"⚠️ Close attempt {retry + 1} failed: {e}")
                            if '0x1788' in str(e):
                                program_mismatch_count += 1

                        # Wait before retry (exponential backoff)
                        if retry < max_retries - 1:
                            wait_time = (retry + 1) * 2  # 2s, 4s, 6s, 8s
                            logger.info(f"   Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)

                    # All retries failed
                    if not close_actually_executed:
                        logger.error(f"🚨 CRITICAL: All {max_retries} close attempts FAILED for {position.token_symbol}")
                        logger.error(f"   Last error: {last_error}")
                        logger.error(f"   Token mint: {token_mint}")
                        logger.error(f"   ⚠️ MANUAL ACTION REQUIRED")

                        # Record failure with safety engine
                        if self.safety_engine:
                            failure_reason = self.safety_engine.parse_error_reason(str(last_error))
                            should_keep_retrying = self.safety_engine.record_close_failure(
                                token_mint=token_mint,
                                token_symbol=position.token_symbol,
                                value_sol=position.value_sol,
                                reason=failure_reason,
                                slippage_used=close_slippage if 'close_slippage' in dir() else self.slippage_bps
                            )

                            if not should_keep_retrying:
                                logger.error(f"🚨 Position {position.token_symbol} marked as STUCK - manual intervention needed")

                        # DON'T record the trade or update position - keep it open for manual handling
                        return

            # Only proceed if close actually executed
            if not close_actually_executed:
                logger.error(f"Close not executed for {position.token_symbol} - keeping position open")
                return

            # Record trade - store value_sol for position size tracking
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                token_mint=token_mint,
                token_symbol=position.token_symbol,
                strategy=position.strategy,
                side=TradeSide.SELL,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                amount=close_value_sol,  # Use partial or full value_sol
                pnl_sol=pnl_sol,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                fees=position.fees_paid * close_pct,  # Proportional fees
                opened_at=position.opened_at,
                closed_at=datetime.utcnow(),  # Fixed: Use UTC for consistency with opened_at
                close_reason=reason,
                is_simulated=position.is_simulated
            )
            self.trade_history.append(trade)

            # Save trade to database for persistence across restarts
            await self._save_trade_to_db(trade)

            # Update stats
            self.total_trades += 1
            self.total_pnl_sol += pnl_sol
            self.risk_metrics.daily_pnl_sol += pnl_sol
            self.risk_metrics.daily_pnl_usd += pnl_usd
            self.risk_metrics.daily_trades += 1
            self.risk_metrics.current_exposure_sol -= close_value_sol

            if pnl_sol > 0:
                self.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
                # Reset block on winning trade (clear any pending block)
                self.risk_metrics.loss_block_start = None
            else:
                self.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1

                # Trigger time-based loss block when threshold is reached
                if self.risk_metrics.consecutive_losses >= self.risk_metrics.max_consecutive_losses:
                    self.risk_metrics.trigger_loss_block()

            # Record trade in PnL tracker for Sharpe/Sortino calculations
            net_pnl = pnl_sol - (position.fees_paid * close_pct)
            trade_record = TradeRecord(
                trade_id=trade.trade_id,
                symbol=position.token_symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                size=close_amount,
                pnl=pnl_sol,
                fees=position.fees_paid * close_pct,
                net_pnl=net_pnl,
                pnl_pct=pnl_pct,
                entry_time=position.opened_at,
                exit_time=datetime.now(),
                duration_seconds=int((datetime.now() - position.opened_at).total_seconds()),
                is_simulated=position.is_simulated
            )
            self.pnl_tracker.record_trade(trade_record)

            if is_partial:
                # Partial exit: Update position instead of removing
                # For LIVE trading, get actual remaining balance to keep tracking accurate
                if not self.dry_run:
                    try:
                        remaining_balance = await self._get_token_balance(token_mint, 6)
                        if remaining_balance > 0:
                            position.amount = remaining_balance  # Use actual remaining balance
                            logger.info(f"📊 Updated position tracking: {remaining_balance:.2f} tokens remaining")
                        else:
                            position.amount -= close_amount  # Fallback to calculated
                    except Exception:
                        position.amount -= close_amount  # Fallback if balance check fails
                else:
                    position.amount -= close_amount  # Dry run: use calculated values

                position.value_sol -= close_value_sol
                position.fees_paid -= position.fees_paid * close_pct  # Reduce tracked fees

                # Track realized PnL in trailing metadata
                if position.metadata and 'trailing' in position.metadata:
                    trailing = position.metadata['trailing']
                    trailing['partial_exits'].append({
                        'tier': reason,
                        'pnl_sol': pnl_sol,
                        'pnl_pct': pnl_pct,
                        'amount_sold': close_amount,
                        'value_sol_sold': close_value_sol,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    trailing['total_realized_pnl'] += pnl_sol
                    trailing['remaining_pct'] = (position.amount / trailing['original_amount']) * 100

                logger.info(f"📊 Partial exit recorded: {close_pct*100:.0f}% sold, {trailing.get('remaining_pct', 0):.0f}% remaining")
            else:
                # Full close: Remove position
                del self.active_positions[token_mint]

                # Set cooldown
                self.token_cooldowns[token_mint] = datetime.now() + self.cooldown_duration

            # Log result
            pnl_emoji = "🟢" if pnl_sol > 0 else "🔴"
            sim_tag = "[DRY_RUN] " if position.is_simulated else ""
            close_type = "Partial exit" if is_partial else "Position closed"
            logger.info(f"{pnl_emoji} {sim_tag}{close_type}: {position.token_symbol}")
            logger.info(f"   Reason: {reason}")
            if is_partial:
                logger.info(f"   Sold: {close_pct*100:.0f}% ({close_value_sol:.4f} SOL)")
            logger.info(f"   PnL: {pnl_sol:.4f} SOL (${pnl_usd:.2f}, {pnl_pct:.2f}%)")
            logger.info(f"   Daily PnL: {self.risk_metrics.daily_pnl_sol:.4f} SOL")

            # Calculate duration
            duration_seconds = int((datetime.utcnow() - position.opened_at).total_seconds())

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
                'win': pnl_sol > 0,
                'opened_at': position.opened_at.isoformat() + 'Z',
                'closed_at': datetime.utcnow().isoformat() + 'Z',
                'duration_seconds': duration_seconds
            })

            # Send Telegram exit alert
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from solana_trading.core.solana_alerts import SolanaTradeAlert
                    # Map reason to action type
                    action_map = {
                        'stop_loss': 'stop_loss',
                        'take_profit': 'take_profit',
                        'max_age': 'time_exit',
                        'time_exit': 'time_exit',
                        'manual_close': 'manual_close',
                    }
                    action = action_map.get(reason.lower(), 'exit')

                    alert = SolanaTradeAlert(
                        token_symbol=position.token_symbol,
                        token_mint=token_mint,
                        strategy=position.strategy.value,
                        action=action,
                        entry_price=position.entry_price,
                        exit_price=position.current_price,
                        amount_sol=position.value_sol,
                        token_amount=position.amount,
                        pnl_sol=pnl_sol,
                        pnl_pct=pnl_pct,
                        reason=reason,
                        is_simulated=position.is_simulated,
                        sol_price_usd=self.sol_price_usd
                    )
                    await self.telegram_alerts.send_exit_alert(alert)
                except Exception as e:
                    logger.debug(f"Telegram exit alert failed: {e}")

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
            # Get strategy-specific TP/SL percentages for display
            stop_loss_pct = self.stop_loss_pct
            take_profit_pct = self.take_profit_pct

            if self.config_manager:
                if pos.strategy == Strategy.JUPITER:
                    stop_loss_pct = -abs(self.config_manager.jupiter_stop_loss_pct)
                    take_profit_pct = self.config_manager.jupiter_take_profit_pct
                elif pos.strategy == Strategy.PUMPFUN:
                    stop_loss_pct = -abs(self.config_manager.pumpfun_stop_loss_pct)
                    take_profit_pct = self.config_manager.pumpfun_take_profit_pct

            positions_summary.append({
                'token_symbol': pos.token_symbol,
                'mint': mint,  # Full mint address for close button
                'strategy': pos.strategy.value,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl_percent': pos.unrealized_pnl_pct,
                'token_amount': pos.amount,
                'current_value_sol': pos.value_sol,
                'unrealized_pnl_usd': pos.unrealized_pnl * self.sol_price_usd if hasattr(pos, 'unrealized_pnl') else 0,
                'opened_at': pos.opened_at.isoformat() + 'Z' if pos.opened_at else None,  # Add 'Z' to indicate UTC
                'stop_loss': stop_loss_pct / 100,  # Convert to decimal for UI (e.g., -0.05 for -5%)
                'take_profit': take_profit_pct / 100,  # Convert to decimal for UI (e.g., 0.10 for 10%)
                'is_simulated': pos.is_simulated
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
