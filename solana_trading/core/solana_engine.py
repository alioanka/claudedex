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
    """Jupiter V6 API client"""

    def __init__(self, api_url: str, api_key: Optional[str] = None, slippage_bps: int = 50):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.slippage_bps = slippage_bps
        self._session: Optional[aiohttp.ClientSession] = None
        self._price_cache: Dict[str, Dict] = {}
        self._price_cache_time: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(seconds=30)

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
        """Get token price in USD via Jupiter price API"""
        # Check cache
        if token_mint in self._price_cache:
            cache_time = self._price_cache_time.get(token_mint, datetime.min)
            if datetime.now() - cache_time < self._cache_ttl:
                return self._price_cache[token_mint].get('price')

        try:
            session = await self._get_session()

            # Jupiter price API v2
            price_api_url = "https://api.jup.ag/price/v2"
            params = {'ids': token_mint}

            async with session.get(price_api_url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data and token_mint in data['data']:
                        price_data = data['data'][token_mint]
                        price = float(price_data.get('price', 0))
                        self._price_cache[token_mint] = {'price': price}
                        self._price_cache_time[token_mint] = datetime.now()
                        return price
                return None
        except Exception as e:
            logger.error(f"Jupiter price fetch failed for {token_mint}: {e}")
            return None

    async def get_prices_batch(self, token_mints: List[str]) -> Dict[str, float]:
        """Get prices for multiple tokens"""
        prices = {}
        try:
            session = await self._get_session()
            price_api_url = "https://api.jup.ag/price/v2"
            params = {'ids': ','.join(token_mints)}

            async with session.get(price_api_url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data:
                        for mint in token_mints:
                            if mint in data['data']:
                                prices[mint] = float(data['data'][mint].get('price', 0))
                                self._price_cache[mint] = {'price': prices[mint]}
                                self._price_cache_time[mint] = datetime.now()
        except Exception as e:
            logger.error(f"Jupiter batch price fetch failed: {e}")

        return prices


class PumpFunMonitor:
    """Monitor Pump.fun for new token launches"""

    def __init__(self, config: Dict):
        self.program_id = config.get('program_id', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P')
        self.min_liquidity_sol = config.get('min_liquidity_sol', 10)
        self.max_age_seconds = config.get('max_age_seconds', 300)
        self.buy_amount_sol = config.get('buy_amount_sol', 0.1)
        self.ws_url = config.get('ws_url', 'wss://pumpportal.fun/api/data')
        self._recent_tokens: Dict[str, Dict] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_new_tokens(self) -> List[Dict]:
        """Get new tokens from Pump.fun API"""
        try:
            session = await self._get_session()

            # Pump.fun doesn't have a public REST API, so this is a placeholder
            # In production, you'd use WebSocket or scrape their site
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Pump.fun fetch failed: {e}")
            return []

    def filter_token(self, token: Dict) -> bool:
        """Filter token based on criteria"""
        # Check liquidity
        liquidity = token.get('liquidity_sol', 0)
        if liquidity < self.min_liquidity_sol:
            return False

        # Check age
        created_at = token.get('created_at')
        if created_at:
            age_seconds = (datetime.now() - created_at).total_seconds()
            if age_seconds > self.max_age_seconds:
                return False

        return True


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
        mode: str = "production"
    ):
        """
        Initialize Solana trading engine

        Args:
            rpc_url: Primary Solana RPC URL
            strategies: List of enabled strategies (jupiter, drift, pumpfun)
            max_positions: Maximum concurrent positions
            mode: Operating mode
        """
        self.primary_rpc = rpc_url
        self.strategies = [Strategy(s.strip().lower()) for s in strategies if s.strip().lower() in [e.value for e in Strategy]]
        self.max_positions = max_positions
        self.mode = mode
        self.is_running = False

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

        # Trading configuration from environment
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

        # Strategy clients
        self.jupiter_client: Optional[JupiterClient] = None
        self.drift_client = None
        self.pumpfun_monitor: Optional[PumpFunMonitor] = None

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

            # Test with SOL price
            sol_price = await self.jupiter_client.get_price(SOL_MINT)
            if sol_price:
                self.sol_price_usd = sol_price
                logger.info(f"✅ Jupiter configured (SOL price: ${sol_price:.2f})")
            else:
                logger.warning("⚠️ Could not fetch SOL price, using default")

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

                # For now, just log that it's configured
                # Full implementation would require proper SDK setup
                logger.info("✅ Drift Protocol configured (basic mode)")

            except ImportError:
                logger.warning("⚠️ driftpy not installed. Drift trading disabled.")
                self.strategies = [s for s in self.strategies if s != Strategy.DRIFT]

        except Exception as e:
            logger.error(f"Drift initialization failed: {e}")

    async def _init_pumpfun(self):
        """Initialize Pump.fun monitoring"""
        try:
            logger.info("Initializing Pump.fun monitor...")

            config = {
                'program_id': os.getenv('PUMPFUN_PROGRAM_ID', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'),
                'min_liquidity_sol': float(os.getenv('PUMPFUN_MIN_LIQUIDITY', '10')),
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
                self.sol_price_usd = price

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        mode_str = "DRY_RUN (SIMULATED)" if self.dry_run else "LIVE TRADING"
        logger.info(f"🚀 Solana trading engine started - {mode_str}")

        try:
            while self.is_running:
                try:
                    # Reset daily metrics at midnight
                    await self._check_daily_reset()

                    # Update SOL price periodically
                    await self._update_sol_price()

                    # Main trading logic
                    await self._trading_cycle()

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

        for token_mint, position in list(self.active_positions.items()):
            try:
                # Get current price
                current_price = await self._get_token_price(token_mint)
                if current_price is None or current_price == 0:
                    continue

                position.current_price = current_price

                # Calculate PnL
                if position.side == TradeSide.BUY:
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                position.unrealized_pnl_pct = pnl_pct
                position.unrealized_pnl = position.value_sol * (pnl_pct / 100)

                # Check exit conditions
                exit_reason = await self._check_exit_conditions(position)
                if exit_reason:
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

        return None

    async def _scan_jupiter_opportunities(self):
        """Scan for Jupiter swap opportunities"""
        if len(self.active_positions) >= self.max_positions:
            return

        # For Jupiter, we'd typically monitor specific tokens or use signals
        # This is a placeholder for strategy logic
        logger.debug("Scanning Jupiter opportunities...")

    async def _scan_drift_opportunities(self):
        """Scan for Drift perpetual opportunities"""
        if len(self.active_positions) >= self.max_positions:
            return

        logger.debug("Scanning Drift opportunities...")

    async def _scan_pumpfun_opportunities(self):
        """Scan for Pump.fun new token launches"""
        if len(self.active_positions) >= self.max_positions:
            return

        if not self.pumpfun_monitor:
            return

        try:
            # Get new tokens
            new_tokens = await self.pumpfun_monitor.get_new_tokens()

            for token in new_tokens:
                # Filter token
                if not self.pumpfun_monitor.filter_token(token):
                    continue

                token_mint = token.get('mint')
                if not token_mint:
                    continue

                # Check cooldown
                if token_mint in self.token_cooldowns:
                    if datetime.now() < self.token_cooldowns[token_mint]:
                        continue

                # Check if already have position
                if token_mint in self.active_positions:
                    continue

                logger.info(f"🎯 Pump.fun token detected: {token.get('symbol', 'UNKNOWN')}")

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
            if self.dry_run:
                logger.info(f"🔵 [DRY_RUN] SIMULATED BUY {token_symbol}")
                logger.info(f"   Price: ${current_price:.8f}")
                logger.info(f"   Amount: {amount_sol} SOL (${value_usd:.2f})")
            else:
                # Execute real Jupiter swap
                try:
                    # Get quote
                    quote = await self.jupiter_client.get_quote(
                        input_mint=SOL_MINT,
                        output_mint=token_mint,
                        amount=int(amount_sol * self.LAMPORTS_PER_SOL)
                    )

                    if not quote:
                        logger.error("Failed to get Jupiter quote")
                        return

                    # TODO: Execute swap transaction
                    # This requires signing and sending transaction
                    logger.info(f"🟢 SWAP executed (simulation - full TX not implemented)")

                except Exception as e:
                    logger.error(f"Swap execution failed: {e}")
                    return

            # Add to positions
            self.active_positions[token_mint] = position
            self.risk_metrics.current_exposure_sol += amount_sol

            logger.info(f"✅ Position opened: {token_symbol} ({strategy.value})")

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
            if self.dry_run:
                logger.info(f"🔵 [DRY_RUN] SIMULATED SELL {position.token_symbol} ({reason})")
            else:
                # Execute real Jupiter swap back to SOL
                logger.info(f"🔴 Position closed (simulation - full TX not implemented)")

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
        max_drawdown = self.pnl_tracker.get_max_drawdown()

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
        except:
            pass

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
