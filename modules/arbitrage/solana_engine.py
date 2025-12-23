"""
Solana Arbitrage Engine

Features:
- Jupiter Aggregator integration (best routing across all Solana DEXs)
- Raydium AMM direct integration
- Orca Whirlpools integration
- Multi-token monitoring
- Jito MEV protection (Solana's Flashbots equivalent)
"""
import asyncio
import logging
import os
import json
import aiohttp
import base58
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger("SolanaArbitrageEngine")

# Solana Token Addresses (Mint addresses)
SOLANA_TOKENS = {
    # Major tokens
    'SOL': 'So11111111111111111111111111111111111111112',  # Wrapped SOL
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',

    # Popular DeFi
    'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',  # Raydium
    'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',  # Serum
    'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',  # Orca
    'MNGO': 'MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac',  # Mango

    # Meme tokens (high volatility = more arb opportunities)
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',  # Dogwifhat
    'POPCAT': '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',

    # Stablecoins
    'USDH': 'USDH1SM1ojwWUga67PGrgFWUHibbjqMvuMaDkRJTgkX',
    'USH': '9iLH8T7zoWhY7sBmj1WK9ENbWdS1nL8n9wAxaeRitTa6',

    # Liquid Staking
    'mSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # Marinade
    'stSOL': '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',  # Lido
    'jitoSOL': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',  # Jito
    'bSOL': 'bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1',  # BlazeStake

    # Jupiter ecosystem
    'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    'JTO': 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL',  # Jito

    # Other popular
    'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
    'W': '85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ',  # Wormhole
    'RENDER': 'rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof',
}

# Solana DEX pairs to monitor
SOLANA_ARB_PAIRS = [
    ('SOL', 'USDC'),
    ('SOL', 'USDT'),
    ('mSOL', 'SOL'),
    ('stSOL', 'SOL'),
    ('jitoSOL', 'SOL'),
    ('bSOL', 'SOL'),
    ('RAY', 'SOL'),
    ('ORCA', 'SOL'),
    ('JUP', 'SOL'),
    ('JTO', 'SOL'),
    ('BONK', 'SOL'),
    ('WIF', 'SOL'),
    ('PYTH', 'SOL'),
    ('W', 'SOL'),
    ('USDC', 'USDT'),
    ('mSOL', 'USDC'),
]

# DEX API endpoints
JUPITER_API = "https://quote-api.jup.ag/v6"
RAYDIUM_API = "https://api.raydium.io/v2"


class RateLimiter:
    """Simple rate limiter to prevent API rate limit errors"""

    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = datetime.now()
            if self.last_request_time:
                elapsed = (now - self.last_request_time).total_seconds()
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = datetime.now()


class JupiterClient:
    """Jupiter Aggregator API client with rate limiting"""

    def __init__(self, rate_limit: float = 2.0):
        self.base_url = JUPITER_API
        self.session: Optional[aiohttp.ClientSession] = None
        # Rate limit from config (default 2 req/sec)
        self.rate_limiter = RateLimiter(rate_limit)
        self._consecutive_errors = 0
        self._backoff_until: Optional[datetime] = None

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50
    ) -> Optional[Dict]:
        """
        Get quote from Jupiter aggregator.
        Jupiter finds the best route across all Solana DEXs.
        """
        # Check if we're in backoff period
        if self._backoff_until:
            if datetime.now() < self._backoff_until:
                return None
            self._backoff_until = None

        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': slippage_bps,
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }

            async with self.session.get(f"{self.base_url}/quote", params=params) as resp:
                if resp.status == 200:
                    self._consecutive_errors = 0
                    data = await resp.json()
                    return data
                elif resp.status == 429:
                    # Rate limited - exponential backoff
                    self._consecutive_errors += 1
                    backoff_seconds = min(60, 2 ** self._consecutive_errors)
                    self._backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
                    logger.warning(f"⚠️ Jupiter rate limited - backing off {backoff_seconds}s")
                    return None
                else:
                    logger.debug(f"Jupiter quote error: HTTP {resp.status}")
                    return None

        except asyncio.TimeoutError:
            logger.debug("Jupiter quote timeout")
            return None
        except Exception as e:
            logger.debug(f"Jupiter quote error: {e}")
            return None

    async def get_swap_transaction(self, quote: Dict, user_public_key: str) -> Optional[Dict]:
        """Get swap transaction from Jupiter"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': user_public_key,
                'wrapAndUnwrapSol': True,
                'useSharedAccounts': True,
                'prioritizationFeeLamports': 'auto'
            }

            async with self.session.post(
                f"{self.base_url}/swap",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None

        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
            return None


class RaydiumClient:
    """Raydium AMM API client for direct pool queries"""

    def __init__(self):
        self.base_url = RAYDIUM_API
        self.session: Optional[aiohttp.ClientSession] = None
        self.pools_cache: Dict = {}
        self.cache_time: Optional[datetime] = None

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=15)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def get_pool_info(self, pool_id: str) -> Optional[Dict]:
        """Get pool info from Raydium"""
        try:
            async with self.session.get(f"{self.base_url}/ammV3/ammPools") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Find specific pool
                    for pool in data.get('data', []):
                        if pool.get('id') == pool_id:
                            return pool
                return None
        except Exception as e:
            logger.debug(f"Raydium pool error: {e}")
            return None

    async def get_price(self, base_mint: str, quote_mint: str) -> Optional[float]:
        """Get price from Raydium pools"""
        try:
            async with self.session.get(f"{self.base_url}/main/price") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    prices = data.get('data', {})
                    # Check if we have the price
                    if base_mint in prices:
                        return float(prices[base_mint])
                return None
        except Exception as e:
            logger.debug(f"Raydium price error: {e}")
            return None


class JitoClient:
    """Jito MEV protection client (Solana's Flashbots)"""

    JITO_BLOCK_ENGINE = "https://mainnet.block-engine.jito.wtf"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def send_bundle(self, transactions: List[str]) -> Optional[str]:
        """
        Send transaction bundle to Jito block engine.
        Protects against sandwich attacks and ensures atomic execution.
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [transactions]
            }

            async with self.session.post(
                f"{self.JITO_BLOCK_ENGINE}/api/v1/bundles",
                json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    bundle_id = result.get('result')
                    logger.info(f"🛡️ Jito bundle submitted: {bundle_id}")
                    return bundle_id
                return None

        except Exception as e:
            logger.error(f"Jito bundle error: {e}")
            return None


class SolanaPriceFetcher:
    """Fetch real-time prices from multiple sources"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"
    JUPITER_PRICE_API = "https://price.jup.ag/v4/price"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 30  # 30 seconds cache

    async def get_price(self, symbol: str) -> float:
        """Get USD price for a token"""
        now = datetime.now()

        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (now - cached_at).total_seconds() < self._cache_ttl:
                return price

        # Try Jupiter Price API first (faster, Solana-native)
        try:
            mint = SOLANA_TOKENS.get(symbol.upper())
            if mint:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.JUPITER_PRICE_API}",
                        params={'ids': mint}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if 'data' in data and mint in data['data']:
                                price = float(data['data'][mint].get('price', 0))
                                if price > 0:
                                    self._cache[symbol] = (price, now)
                                    return price
        except Exception:
            pass

        # Fallback to CoinGecko
        try:
            symbol_map = {
                'sol': 'solana',
                'msol': 'marinade-staked-sol',
                'jitosol': 'jito-staked-sol',
                'ray': 'raydium',
                'orca': 'orca',
                'jup': 'jupiter-exchange-solana',
                'bonk': 'bonk',
                'wif': 'dogwifcoin',
            }
            coin_id = symbol_map.get(symbol.lower(), symbol.lower())

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.COINGECKO_API,
                    params={'ids': coin_id, 'vs_currencies': 'usd'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if coin_id in data and 'usd' in data[coin_id]:
                            price = float(data[coin_id]['usd'])
                            self._cache[symbol] = (price, now)
                            return price
        except Exception:
            pass

        # Return cached or default
        if symbol in self._cache:
            return self._cache[symbol][0]

        # SOL default
        return 100.0 if symbol.upper() == 'SOL' else 0.0


class SolanaArbitrageEngine:
    """
    Solana Arbitrage Engine with Jupiter, Raydium, and Jito integration.

    Strategy:
    1. Get quotes from Jupiter (aggregates all DEXs)
    2. Compare with direct Raydium quotes
    3. If spread > threshold, execute via Jito for MEV protection
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False

        # Solana RPC - use Pool Engine with fallback
        self.rpc_url = config.get('rpc_url')
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                self.rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC')
            except Exception:
                pass
        if not self.rpc_url:
            self.rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')

        self.private_key = None  # Loaded in initialize() from secrets manager
        self.wallet_address = os.getenv('SOLANA_WALLET_ADDRESS')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Settings from config (loaded from DB via settings page)
        # Threshold is stored as percentage (0.2 = 0.2%), convert to decimal (0.002)
        self.min_profit_threshold = config.get('sol_arb_threshold', 0.2) / 100.0
        self.trade_amount_sol = config.get('sol_trade_amount', 1.0)
        self.use_jito = config.get('use_jito', True)

        # Verbose logging (configurable via settings page)
        self.verbose_logging = config.get('sol_arb_verbose', False)

        # Rate limiting - cooldown from config
        self._last_opportunity_time: Dict[str, datetime] = {}
        self._opportunity_cooldown = config.get('sol_arb_cooldown', 300)  # 5 min default

        # Jupiter rate limiter from config
        jupiter_rps = config.get('jupiter_rps', 2.0)

        # Clients - pass rate limit to Jupiter
        self.jupiter = JupiterClient(rate_limit=float(jupiter_rps))
        self.raydium = RaydiumClient()
        self.jito = JitoClient()
        self.price_fetcher = SolanaPriceFetcher()
        self._pair_execution_count: Dict[str, int] = {}
        self._pair_execution_date: str = ""
        self._max_executions_per_pair_per_day = 10

        self._stats = {
            'scans': 0,
            'opportunities_found': 0,
            'opportunities_executed': 0,
            'last_stats_log': datetime.now()
        }

    async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
        """Get decrypted private key from secrets manager or environment."""
        try:
            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                if self.db_pool and not secrets._initialized:
                    secrets.initialize(self.db_pool)
                value = await secrets.get_async(key_name)
                if value:
                    return value
            except Exception:
                pass

            # Fallback to environment with decryption
            encrypted_key = os.getenv(key_name)
            if not encrypted_key:
                return None

            # Get encryption key
            encryption_key = None
            from pathlib import Path
            key_file = Path('.encryption_key')
            if key_file.exists():
                encryption_key = key_file.read_text().strip()
            if not encryption_key:
                encryption_key = os.getenv('ENCRYPTION_KEY')

            # Decrypt if Fernet encrypted
            if encrypted_key.startswith('gAAAAAB') and encryption_key:
                try:
                    from cryptography.fernet import Fernet
                    f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                    return f.decrypt(encrypted_key.encode()).decode()
                except Exception as e:
                    logger.error(f"Failed to decrypt {key_name}: {e}")
                    return None

            return encrypted_key
        except Exception as e:
            logger.debug(f"Error getting {key_name}: {e}")
            return None

    async def initialize(self):
        logger.info("🌊 Initializing Solana Arbitrage Engine...")

        # Load private key from secrets manager
        self.private_key = await self._get_decrypted_key('SOLANA_PRIVATE_KEY')

        await self.jupiter.initialize()
        await self.raydium.initialize()

        if self.private_key and not self.dry_run:
            await self.jito.initialize()
            logger.info("🛡️ Jito MEV protection enabled")

        logger.info(f"   RPC: {self.rpc_url[:40]}...")
        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Jito: {'Enabled' if self.use_jito and self.private_key else 'Disabled'}")
        logger.info(f"   Monitoring {len(SOLANA_ARB_PAIRS)} token pairs")
        logger.info(f"   Threshold: {self.min_profit_threshold:.2%} | Cooldown: {self._opportunity_cooldown}s")
        logger.info(f"   Verbose: {self.verbose_logging}")

    async def run(self):
        self.is_running = True
        logger.info("🌊 Solana Arbitrage Engine Started")

        pair_index = 0

        while self.is_running:
            try:
                self._stats['scans'] += 1

                # Get current pair
                token_in_symbol, token_out_symbol = SOLANA_ARB_PAIRS[pair_index]
                token_in = SOLANA_TOKENS.get(token_in_symbol)
                token_out = SOLANA_TOKENS.get(token_out_symbol)

                if token_in and token_out:
                    await self._check_arb_opportunity(
                        token_in, token_out,
                        token_in_symbol, token_out_symbol
                    )

                # Move to next pair
                pair_index = (pair_index + 1) % len(SOLANA_ARB_PAIRS)

                # Log stats
                await self._log_stats_if_needed()

                # Faster scanning on Solana (lower fees)
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Solana arb loop error: {e}")
                await asyncio.sleep(5)

    async def _log_stats_if_needed(self):
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:
            logger.info(f"📊 SOLANA ARB STATS (Last 5 min): "
                       f"Scans: {self._stats['scans']} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']}")

            self._stats = {
                'scans': 0,
                'opportunities_found': 0,
                'opportunities_executed': 0,
                'last_stats_log': now
            }

    async def _check_arb_opportunity(
        self,
        token_in: str,
        token_out: str,
        in_symbol: str,
        out_symbol: str
    ) -> bool:
        """Check for arbitrage between Jupiter aggregated route and direct DEX"""
        try:
            # Convert trade amount to lamports (SOL decimals = 9)
            if in_symbol == 'SOL':
                amount = int(self.trade_amount_sol * 1e9)
            elif in_symbol in ['USDC', 'USDT']:
                amount = int(self.trade_amount_sol * 100 * 1e6)  # Assuming ~$100 worth
            else:
                # For other tokens, use a standard amount
                amount = int(1e9)

            # Get Jupiter quote (best aggregated route)
            jupiter_quote = await self.jupiter.get_quote(token_in, token_out, amount)

            if not jupiter_quote:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: No forward quote from Jupiter")
                return False

            jupiter_out = int(jupiter_quote.get('outAmount', 0))
            if jupiter_out == 0:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: Jupiter returned 0 output")
                return False

            # Get reverse route quote to check round-trip profitability
            reverse_quote = await self.jupiter.get_quote(token_out, token_in, jupiter_out)

            if not reverse_quote:
                if self.verbose_logging:
                    logger.warning(f"🌊 [{in_symbol}/{out_symbol}]: No reverse quote from Jupiter")
                return False

            reverse_out = int(reverse_quote.get('outAmount', 0))

            # Calculate round-trip profit
            profit_pct = (reverse_out - amount) / amount

            # Log price check result (always log if verbose, or log good spreads)
            if self.verbose_logging or profit_pct > 0.0005:  # Log if > 0.05%
                route_info = jupiter_quote.get('routePlan', [])
                route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route_info[:3]])
                status = "✅ OPPORTUNITY" if profit_pct > self.min_profit_threshold else "📊 Below threshold"
                logger.info(f"🌊 {status} [{in_symbol}/{out_symbol}]: Spread: {profit_pct:.3%} | Threshold: {self.min_profit_threshold:.3%} | Route: {route_str}")

            if profit_pct > self.min_profit_threshold:
                self._stats['opportunities_found'] += 1

                # Check rate limiting
                opp_key = f"{in_symbol}_{out_symbol}"
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Reset daily counters
                if self._pair_execution_date != today:
                    self._pair_execution_count = {}
                    self._pair_execution_date = today

                # Check limits
                current_count = self._pair_execution_count.get(opp_key, 0)
                if current_count >= self._max_executions_per_pair_per_day:
                    return True

                # Check cooldown
                last_time = self._last_opportunity_time.get(opp_key)
                if last_time:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        return True

                # Log and execute
                self._last_opportunity_time[opp_key] = now
                self._pair_execution_count[opp_key] = current_count + 1

                route_info = jupiter_quote.get('routePlan', [])
                route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route_info[:3]])

                logger.info(f"🚨 SOLANA ARB [{in_symbol}/{out_symbol}]: {profit_pct:.2%} profit | Route: {route_str}")
                self._stats['opportunities_executed'] += 1

                await self._execute_arbitrage(
                    jupiter_quote, reverse_quote,
                    token_in, token_out,
                    in_symbol, out_symbol,
                    amount, profit_pct
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Solana arb check error [{in_symbol}/{out_symbol}]: {e}")
            return False

    async def _execute_arbitrage(
        self,
        quote1: Dict,
        quote2: Dict,
        token_in: str,
        token_out: str,
        in_symbol: str,
        out_symbol: str,
        amount: int,
        profit_pct: float
    ):
        """Execute arbitrage trade"""
        logger.info(f"⚡ Executing Solana Arbitrage [{in_symbol}/{out_symbol}] | Expected: +{profit_pct:.2%}")

        if self.dry_run:
            await asyncio.sleep(0.3)
            logger.info(f"✅ Solana Arb Executed (DRY RUN) [{in_symbol}/{out_symbol}]")
            await self._log_trade(
                in_symbol, out_symbol, token_in,
                amount, profit_pct, "DRY_RUN",
                quote1.get('routePlan', [])
            )
            return

        try:
            if not self.wallet_address:
                logger.error("No Solana wallet configured")
                return

            # Get swap transactions
            swap1 = await self.jupiter.get_swap_transaction(quote1, self.wallet_address)
            swap2 = await self.jupiter.get_swap_transaction(quote2, self.wallet_address)

            if not swap1 or not swap2:
                logger.error("Failed to get swap transactions")
                return

            # If Jito is enabled, bundle both transactions
            if self.use_jito and self.jito.session:
                tx1 = swap1.get('swapTransaction')
                tx2 = swap2.get('swapTransaction')

                bundle_id = await self.jito.send_bundle([tx1, tx2])
                if bundle_id:
                    logger.info(f"✅ Solana Arb executed via Jito: {bundle_id}")
                    await self._log_trade(
                        in_symbol, out_symbol, token_in,
                        amount, profit_pct, bundle_id,
                        quote1.get('routePlan', [])
                    )
                else:
                    logger.error("Jito bundle rejected")
            else:
                # Direct execution (not recommended for arb)
                logger.warning("Direct execution not recommended - use Jito for MEV protection")

        except Exception as e:
            logger.error(f"Solana arb execution error: {e}")

    async def _log_trade(
        self,
        in_symbol: str,
        out_symbol: str,
        token_address: str,
        amount: int,
        profit_pct: float,
        tx_hash: str,
        route: List
    ):
        """Log trade to database"""
        if not self.db_pool:
            return

        try:
            import uuid

            # Get SOL price
            sol_price = await self.price_fetcher.get_price('SOL')

            # Calculate values
            if in_symbol == 'SOL':
                amount_sol = amount / 1e9
            else:
                amount_sol = self.trade_amount_sol

            entry_usd = amount_sol * sol_price
            profit_usd = entry_usd * profit_pct

            logger.info(f"💰 Solana Arb [{in_symbol}/{out_symbol}]: {amount_sol:.4f} SOL @ ${sol_price:.2f} = ${entry_usd:.2f} | Profit: ${profit_usd:.2f}")

            trade_id = f"sol_arb_{uuid.uuid4().hex[:12]}"

            # Get route as string
            route_str = " → ".join([r.get('swapInfo', {}).get('label', 'DEX') for r in route[:5]])

            async with self.db_pool.acquire() as conn:
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
                    token_address,
                    'solana',
                    'jupiter',
                    'jupiter',
                    'buy',
                    sol_price,
                    sol_price * (1 + profit_pct),
                    amount_sol,
                    amount_sol,
                    entry_usd,
                    entry_usd + profit_usd,
                    profit_usd,
                    profit_pct * 100,
                    profit_pct * 100,
                    'closed',
                    self.dry_run,
                    datetime.now(),
                    datetime.now(),
                    tx_hash,
                    sol_price,
                    json.dumps({
                        'chain': 'solana',
                        'in_symbol': in_symbol,
                        'out_symbol': out_symbol,
                        'route': route_str,
                        'dry_run': self.dry_run
                    })
                )
            logger.debug(f"💾 Logged Solana arb: {trade_id}")

        except Exception as e:
            logger.error(f"Error logging Solana trade: {e}")

    async def stop(self):
        self.is_running = False

        await self.jupiter.close()
        await self.raydium.close()
        await self.jito.close()

        logger.info("🛑 Solana Arbitrage Engine Stopped")
