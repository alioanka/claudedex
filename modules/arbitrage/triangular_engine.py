"""
Triangular Arbitrage Engine

Finds profitable cycles: A → B → C → A

Examples:
- ETH → USDC → DAI → ETH
- WBTC → WETH → USDT → WBTC

Features:
- Dynamic gas-aware profit thresholds
- Multi-DEX route optimization
- Curve Finance integration for stablecoin swaps
- Balancer integration for multi-token pools
"""
import asyncio
import logging
import os
import json
import aiohttp
from web3 import Web3
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from itertools import permutations

logger = logging.getLogger("TriangularArbitrageEngine")


class RPCRateLimiter:
    """Rate limiter for Ethereum RPC calls to prevent 429 errors"""

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._consecutive_errors = 0
        self._backoff_until: Optional[datetime] = None

    async def acquire(self):
        async with self._lock:
            # Check backoff
            if self._backoff_until and datetime.now() < self._backoff_until:
                await asyncio.sleep((self._backoff_until - datetime.now()).total_seconds())
                self._backoff_until = None

            now = datetime.now()
            if self.last_call_time:
                elapsed = (now - self.last_call_time).total_seconds()
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self.last_call_time = datetime.now()

    def report_error(self, is_rate_limit: bool = False):
        """Report an error to trigger backoff"""
        if is_rate_limit:
            self._consecutive_errors += 1
            backoff_seconds = min(60, 2 ** self._consecutive_errors)
            self._backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
            logger.warning(f"⚠️ RPC rate limited - backing off {backoff_seconds}s")
        else:
            self._consecutive_errors = max(0, self._consecutive_errors - 1)

    def report_success(self):
        """Report success to reset error counter"""
        self._consecutive_errors = 0

# Expanded token list for triangular arb
TOKENS = {
    # Major
    'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
    'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
    'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    'DAI': '0x6B175474E89094C44Da98b954EeAdDcB80656c63',

    # DeFi
    'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
    'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
    'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
    'CRV': '0xD533a949740bb3306d119CC777fa900bA034cd52',
    'MKR': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2',

    # Stablecoins (good for stablecoin triangular arb)
    'FRAX': '0x853d955aCEf822Db058eb8505911ED77F175b99e',
    'LUSD': '0x5f98805A4E8be255a32880FDeC7F6728C6568bA0',
    'GUSD': '0x056Fd409E1d7A124BD7017459dFEa2F387b6d5Cd',
    'USDD': '0x0C10bF8FcB7Bf5412187A595ab97a3609160b5c6',
    'crvUSD': '0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E',

    # Liquid Staking
    'stETH': '0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84',
    'rETH': '0xae78736Cd615f374D3085123A210448E74Fc6393',
    'cbETH': '0xBe9895146f7AF43049ca1c1AE358B0541Ea49704',
    'frxETH': '0x5E8422345238F34275888049021821E8E08CAa1f',
}

# Triangular arbitrage cycles to check
# Format: (TokenA, TokenB, TokenC) - will check A→B→C→A
TRIANGULAR_CYCLES = [
    # Stablecoin triangles (often have small but consistent opportunities)
    ('USDC', 'USDT', 'DAI'),
    ('USDC', 'DAI', 'FRAX'),
    ('USDC', 'USDT', 'FRAX'),
    ('DAI', 'USDT', 'LUSD'),
    ('USDC', 'DAI', 'crvUSD'),

    # ETH-based triangles
    ('WETH', 'USDC', 'USDT'),
    ('WETH', 'USDC', 'DAI'),
    ('WETH', 'USDT', 'DAI'),
    ('WETH', 'WBTC', 'USDC'),
    ('WETH', 'WBTC', 'USDT'),
    ('WETH', 'LINK', 'USDC'),
    ('WETH', 'UNI', 'USDC'),
    ('WETH', 'AAVE', 'USDC'),

    # Liquid staking triangles
    ('WETH', 'stETH', 'USDC'),
    ('WETH', 'rETH', 'USDC'),
    ('WETH', 'cbETH', 'USDC'),
    ('stETH', 'rETH', 'WETH'),
    ('stETH', 'frxETH', 'WETH'),

    # DeFi token triangles
    ('CRV', 'WETH', 'USDC'),
    ('MKR', 'WETH', 'DAI'),
    ('UNI', 'WETH', 'USDT'),
]

# DEX Router addresses
ROUTERS = {
    'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
    'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
}

# Curve pools for stablecoin swaps
CURVE_POOLS = {
    '3pool': '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',  # DAI/USDC/USDT
    'frax3crv': '0xd632f22692FaC7611d2AA1C0D552930D43CAEd3B',  # FRAX/DAI/USDC/USDT
    'lusd3crv': '0xEd279fDD11cA84bEef15AF5D39BB4d4bEE23F0cA',  # LUSD/DAI/USDC/USDT
    'crvusd_usdc': '0x4DEcE678ceceb27446b35C672dC7d61F30bAD69E',  # crvUSD/USDC
    'steth': '0xDC24316b9AE028F1497c275EB9192a3Ea0f67022',  # ETH/stETH
}

# Uniswap V2 Router ABI
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
    }
]

# Curve pool ABI (for get_dy)
CURVE_POOL_ABI = [
    {
        "name": "get_dy",
        "outputs": [{"type": "uint256", "name": ""}],
        "inputs": [
            {"type": "int128", "name": "i"},
            {"type": "int128", "name": "j"},
            {"type": "uint256", "name": "dx"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "name": "coins",
        "outputs": [{"type": "address", "name": ""}],
        "inputs": [{"type": "uint256", "name": "i"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class GasOracle:
    """Dynamic gas price oracle for profit calculations"""

    ETHERSCAN_GAS_API = "https://api.etherscan.io/api"

    def __init__(self):
        self.current_gas_gwei = 30  # Default
        self._last_update = None
        self._cache_ttl = 60  # 1 minute

    async def get_gas_price(self) -> int:
        """Get current gas price in gwei"""
        now = datetime.now()

        if self._last_update:
            elapsed = (now - self._last_update).total_seconds()
            if elapsed < self._cache_ttl:
                return self.current_gas_gwei

        try:
            api_key = os.getenv('ETHERSCAN_API_KEY', '')
            async with aiohttp.ClientSession() as session:
                params = {
                    'module': 'gastracker',
                    'action': 'gasoracle',
                    'apikey': api_key
                }
                async with session.get(self.ETHERSCAN_GAS_API, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('status') == '1':
                            # Use fast gas price for arb
                            self.current_gas_gwei = int(data['result']['FastGasPrice'])
                            self._last_update = now
        except Exception:
            pass

        return self.current_gas_gwei

    def calculate_gas_cost_usd(self, gas_used: int, eth_price: float) -> float:
        """Calculate gas cost in USD"""
        gas_cost_eth = (self.current_gas_gwei * 1e-9) * gas_used
        return gas_cost_eth * eth_price


class TriangularArbitrageEngine:
    """
    Triangular Arbitrage Engine with gas-aware execution.

    Finds and executes profitable triangular cycles:
    A → B → C → A

    Features:
    - Dynamic gas threshold adjustment
    - Multi-DEX routing per hop
    - Curve integration for stablecoin swaps
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.w3 = None

        # Get RPC URL from config, Pool Engine, or env fallback
        self.rpc_url = config.get('rpc_url')
        if not self.rpc_url:
            try:
                from config.rpc_provider import RPCProvider
                self.rpc_url = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
            except Exception:
                pass
        if not self.rpc_url:
            self.rpc_url = os.getenv('ETHEREUM_RPC_URL', os.getenv('WEB3_PROVIDER_URL'))

        self.private_key = None  # Loaded in initialize() from secrets manager
        self.wallet_address = os.getenv('WALLET_ADDRESS')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        self.router_contracts = {}
        self.curve_contracts = {}

        # Gas oracle
        self.gas_oracle = GasOracle()

        # Settings from config (loaded from DB via settings page)
        # Threshold is stored as percentage (0.1 = 0.1%), convert to decimal (0.001)
        self.base_profit_threshold = config.get('tri_base_threshold', 0.1) / 100.0
        self.gas_buffer_multiplier = config.get('tri_gas_buffer', 1.1)
        self.trade_amount_eth = config.get('eth_trade_amount', 1.0)

        # Verbose logging (configurable via settings page)
        self.verbose_logging = config.get('tri_verbose_log', False)

        # RPC rate limiter from config
        rpc_rate = config.get('eth_rpc_calls_per_sec', 5)
        self.rpc_rate_limiter = RPCRateLimiter(float(rpc_rate))

        # Rate limiting - cooldown from config
        self._last_opportunity_time: Dict[str, datetime] = {}
        self._opportunity_cooldown = config.get('tri_cooldown', 300)  # 5 min default
        self._cycle_execution_count: Dict[str, int] = {}
        self._execution_date: str = ""
        self._max_executions_per_cycle_per_day = 5

        # ETH price cache
        self._eth_price = 2000.0
        self._eth_price_updated = None

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
        logger.info("🔺 Initializing Triangular Arbitrage Engine...")

        # Load private key from secrets manager
        self.private_key = await self._get_decrypted_key('PRIVATE_KEY')

        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url.split(',')[0]))
            if self.w3.is_connected():
                logger.info("✅ Connected to RPC")

                # Initialize DEX routers
                for name, address in ROUTERS.items():
                    try:
                        self.router_contracts[name] = self.w3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=ROUTER_ABI
                        )
                    except Exception as e:
                        logger.debug(f"Router init error {name}: {e}")

                # Initialize Curve pools
                for name, address in CURVE_POOLS.items():
                    try:
                        self.curve_contracts[name] = self.w3.eth.contract(
                            address=Web3.to_checksum_address(address),
                            abi=CURVE_POOL_ABI
                        )
                    except Exception as e:
                        logger.debug(f"Curve pool init error {name}: {e}")

                # Get initial gas price
                gas_price = await self.gas_oracle.get_gas_price()
                logger.info(f"   Current gas: {gas_price} gwei")

                # Initialize Flashbots executor for atomic bundle execution
                self.flashbots_executor = None
                if self.private_key and not self.dry_run:
                    try:
                        from .arbitrage_engine import FlashbotsExecutor
                        self.flashbots_executor = FlashbotsExecutor(
                            self.w3,
                            self.private_key
                        )
                        await self.flashbots_executor.initialize()
                        logger.info("⚡ Flashbots executor initialized for atomic triangular arb")
                    except Exception as e:
                        logger.warning(f"Flashbots initialization failed: {e}")
                        self.flashbots_executor = None

            else:
                logger.warning("⚠️ Failed to connect to RPC")
        else:
            logger.error("❌ No RPC URL configured")

        logger.info(f"   Mode: {'DRY_RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"   Cycles: {len(TRIANGULAR_CYCLES)} triangular routes")
        logger.info(f"   DEXs: {len(self.router_contracts)} routers")
        logger.info(f"   Curve: {len(self.curve_contracts)} pools")
        logger.info(f"   Threshold: {self.base_profit_threshold:.2%} base + gas buffer {self.gas_buffer_multiplier:.1f}x")
        logger.info(f"   Cooldown: {self._opportunity_cooldown}s | Verbose: {self.verbose_logging}")

    async def run(self):
        self.is_running = True
        logger.info("🔺 Triangular Arbitrage Engine Started")

        if not self.w3:
            logger.error("RPC not connected, triangular arb disabled")
            return

        cycle_index = 0

        while self.is_running:
            try:
                self._stats['scans'] += 1

                # Update ETH price periodically
                await self._update_eth_price()

                # Update gas price
                await self.gas_oracle.get_gas_price()

                # Get current cycle
                token_a, token_b, token_c = TRIANGULAR_CYCLES[cycle_index]

                await self._check_triangular_opportunity(token_a, token_b, token_c)

                # Move to next cycle
                cycle_index = (cycle_index + 1) % len(TRIANGULAR_CYCLES)

                # Log stats
                await self._log_stats_if_needed()

                # Sleep between scans
                await asyncio.sleep(3)

            except Exception as e:
                logger.error(f"Triangular loop error: {e}")
                await asyncio.sleep(5)

    async def _update_eth_price(self):
        """Update ETH price cache"""
        now = datetime.now()
        if self._eth_price_updated:
            elapsed = (now - self._eth_price_updated).total_seconds()
            if elapsed < 60:  # 1 minute cache
                return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.coingecko.com/api/v3/simple/price',
                    params={'ids': 'ethereum', 'vs_currencies': 'usd'}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._eth_price = float(data['ethereum']['usd'])
                        self._eth_price_updated = now
        except Exception:
            pass

    async def _log_stats_if_needed(self):
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:
            gas = self.gas_oracle.current_gas_gwei
            logger.info(f"📊 TRIANGULAR ARB STATS: "
                       f"Scans: {self._stats['scans']} | "
                       f"Opportunities: {self._stats['opportunities_found']} | "
                       f"Executed: {self._stats['opportunities_executed']} | "
                       f"Gas: {gas} gwei")

            self._stats = {
                'scans': 0,
                'opportunities_found': 0,
                'opportunities_executed': 0,
                'last_stats_log': now
            }

    def _calculate_dynamic_threshold(self, gas_used: int = 500000) -> float:
        """
        Calculate dynamic profit threshold based on gas costs.

        Formula: threshold = base_threshold + (gas_cost / trade_value) * buffer
        """
        gas_cost_usd = self.gas_oracle.calculate_gas_cost_usd(gas_used, self._eth_price)
        trade_value_usd = self.trade_amount_eth * self._eth_price

        # Gas cost as percentage of trade
        gas_pct = gas_cost_usd / trade_value_usd

        # Dynamic threshold = base + gas buffer
        threshold = self.base_profit_threshold + (gas_pct * self.gas_buffer_multiplier)

        return threshold

    async def _check_triangular_opportunity(
        self,
        token_a_symbol: str,
        token_b_symbol: str,
        token_c_symbol: str
    ) -> bool:
        """Check for profitable triangular arbitrage"""
        try:
            token_a = TOKENS.get(token_a_symbol)
            token_b = TOKENS.get(token_b_symbol)
            token_c = TOKENS.get(token_c_symbol)

            if not all([token_a, token_b, token_c]):
                return False

            # Determine trade amount based on first token
            if token_a_symbol in ['USDC', 'USDT', 'DAI', 'FRAX', 'LUSD']:
                # Stablecoin - use $1000 equivalent
                amount_in = int(1000 * 1e6) if token_a_symbol in ['USDC', 'USDT'] else int(1000 * 1e18)
            elif token_a_symbol == 'WETH':
                amount_in = int(self.trade_amount_eth * 1e18)
            elif token_a_symbol == 'WBTC':
                amount_in = int(0.05 * 1e8)  # 0.05 BTC
            else:
                amount_in = int(1 * 1e18)  # 1 token

            # Find best route for each hop
            cycle_key = f"{token_a_symbol}→{token_b_symbol}→{token_c_symbol}"

            # Hop 1: A → B
            best_hop1_out = 0
            best_hop1_dex = None
            hop1_errors = 0
            for dex_name, contract in self.router_contracts.items():
                try:
                    await self.rpc_rate_limiter.acquire()
                    amounts = contract.functions.getAmountsOut(
                        amount_in, [token_a, token_b]
                    ).call()
                    self.rpc_rate_limiter.report_success()
                    if amounts[1] > best_hop1_out:
                        best_hop1_out = amounts[1]
                        best_hop1_dex = dex_name
                except Exception as e:
                    hop1_errors += 1
                    # Check if it's a rate limit error
                    is_rate_limit = '429' in str(e) or 'rate' in str(e).lower()
                    self.rpc_rate_limiter.report_error(is_rate_limit)
                    if self.verbose_logging:
                        logger.debug(f"  {dex_name} {token_a_symbol}→{token_b_symbol}: {e}")

            if best_hop1_out == 0:
                if self.verbose_logging:
                    logger.warning(f"🔺 {cycle_key}: No quotes for hop1 ({hop1_errors} DEX errors)")
                return False

            # Hop 2: B → C
            best_hop2_out = 0
            best_hop2_dex = None
            hop2_errors = 0
            for dex_name, contract in self.router_contracts.items():
                try:
                    await self.rpc_rate_limiter.acquire()
                    amounts = contract.functions.getAmountsOut(
                        best_hop1_out, [token_b, token_c]
                    ).call()
                    self.rpc_rate_limiter.report_success()
                    if amounts[1] > best_hop2_out:
                        best_hop2_out = amounts[1]
                        best_hop2_dex = dex_name
                except Exception as e:
                    hop2_errors += 1
                    is_rate_limit = '429' in str(e) or 'rate' in str(e).lower()
                    self.rpc_rate_limiter.report_error(is_rate_limit)
                    if self.verbose_logging:
                        logger.debug(f"  {dex_name} {token_b_symbol}→{token_c_symbol}: {e}")

            if best_hop2_out == 0:
                if self.verbose_logging:
                    logger.warning(f"🔺 {cycle_key}: No quotes for hop2 ({hop2_errors} DEX errors)")
                return False

            # Hop 3: C → A
            best_hop3_out = 0
            best_hop3_dex = None
            hop3_errors = 0
            for dex_name, contract in self.router_contracts.items():
                try:
                    await self.rpc_rate_limiter.acquire()
                    amounts = contract.functions.getAmountsOut(
                        best_hop2_out, [token_c, token_a]
                    ).call()
                    self.rpc_rate_limiter.report_success()
                    if amounts[1] > best_hop3_out:
                        best_hop3_out = amounts[1]
                        best_hop3_dex = dex_name
                except Exception as e:
                    hop3_errors += 1
                    is_rate_limit = '429' in str(e) or 'rate' in str(e).lower()
                    self.rpc_rate_limiter.report_error(is_rate_limit)
                    if self.verbose_logging:
                        logger.debug(f"  {dex_name} {token_c_symbol}→{token_a_symbol}: {e}")

            if best_hop3_out == 0:
                if self.verbose_logging:
                    logger.warning(f"🔺 {cycle_key}: No quotes for hop3 ({hop3_errors} DEX errors)")
                return False

            # Calculate profit
            profit_pct = (best_hop3_out - amount_in) / amount_in

            # Calculate dynamic threshold
            threshold = self._calculate_dynamic_threshold()

            # Log price check result (always log if verbose, or log good spreads)
            if self.verbose_logging or profit_pct > 0.0005:  # Log if > 0.05%
                route = f"{token_a_symbol}({best_hop1_dex})→{token_b_symbol}({best_hop2_dex})→{token_c_symbol}({best_hop3_dex})→{token_a_symbol}"
                status = "✅ OPPORTUNITY" if profit_pct > threshold else "📊 Below threshold"
                logger.info(f"🔺 {status}: {route} | Spread: {profit_pct:.3%} | Threshold: {threshold:.3%}")

            if profit_pct > threshold:
                self._stats['opportunities_found'] += 1

                cycle_key = f"{token_a_symbol}_{token_b_symbol}_{token_c_symbol}"
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")

                # Reset daily counters
                if self._execution_date != today:
                    self._cycle_execution_count = {}
                    self._execution_date = today

                # Check limits
                current_count = self._cycle_execution_count.get(cycle_key, 0)
                if current_count >= self._max_executions_per_cycle_per_day:
                    return True

                # Check cooldown
                last_time = self._last_opportunity_time.get(cycle_key)
                if last_time:
                    elapsed = (now - last_time).total_seconds()
                    if elapsed < self._opportunity_cooldown:
                        return True

                # Execute
                self._last_opportunity_time[cycle_key] = now
                self._cycle_execution_count[cycle_key] = current_count + 1

                route = f"{token_a_symbol}→{token_b_symbol}({best_hop1_dex})→{token_c_symbol}({best_hop2_dex})→{token_a_symbol}({best_hop3_dex})"

                logger.info(f"🔺 TRIANGULAR ARB: {route} | Profit: {profit_pct:.2%} (threshold: {threshold:.2%})")
                self._stats['opportunities_executed'] += 1

                await self._execute_triangular(
                    cycle_key, route,
                    token_a, token_b, token_c,
                    token_a_symbol, token_b_symbol, token_c_symbol,
                    amount_in, best_hop3_out, profit_pct,
                    [best_hop1_dex, best_hop2_dex, best_hop3_dex]
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Triangular check error: {e}")
            return False

    async def _execute_triangular(
        self,
        cycle_key: str,
        route: str,
        token_a: str,
        token_b: str,
        token_c: str,
        symbol_a: str,
        symbol_b: str,
        symbol_c: str,
        amount_in: int,
        amount_out: int,
        profit_pct: float,
        dexes: List[str]
    ):
        """Execute triangular arbitrage"""
        logger.info(f"⚡ Executing Triangular Arb: {route}")

        if self.dry_run:
            await asyncio.sleep(0.5)
            logger.info(f"✅ Triangular Arb Executed (DRY RUN): {cycle_key}")
            await self._log_trade(
                cycle_key, route, token_a,
                amount_in, profit_pct, "DRY_RUN",
                dexes, symbol_a, symbol_b, symbol_c
            )
            return

        # Live execution of triangular arbitrage
        try:
            if not self.private_key or not self.wallet_address:
                logger.error("Private key or wallet address not available for live trading")
                return

            tx_hash = await self._execute_triangular_swap(
                token_a, token_b, token_c,
                amount_in, dexes
            )

            if tx_hash:
                logger.info(f"✅ Triangular Arb Executed: {tx_hash}")
                await self._log_trade(
                    cycle_key, route, token_a,
                    amount_in, profit_pct, tx_hash,
                    dexes, symbol_a, symbol_b, symbol_c
                )
            else:
                logger.error(f"❌ Triangular Arb Failed: {cycle_key}")

        except Exception as e:
            logger.error(f"Triangular arb execution error: {e}")

    async def _execute_triangular_swap(
        self,
        token_a: str,
        token_b: str,
        token_c: str,
        amount_in: int,
        dexes: List[str]
    ) -> Optional[str]:
        """
        Execute 3-hop triangular swap via Flashbots for atomic execution.

        Path: A → B → C → A
        Uses Flashbots bundle to ensure atomicity (all succeed or all fail).
        """
        try:
            if len(dexes) != 3:
                logger.error("Triangular arb requires exactly 3 DEXes")
                return None

            dex1, dex2, dex3 = dexes
            router1 = self.router_contracts.get(dex1)
            router2 = self.router_contracts.get(dex2)
            router3 = self.router_contracts.get(dex3)

            if not all([router1, router2, router3]):
                logger.error("Not all router contracts available")
                return None

            wallet = Web3.to_checksum_address(self.wallet_address)
            deadline = int(datetime.now().timestamp()) + 120
            base_nonce = self.w3.eth.get_transaction_count(wallet)

            # Build swap 1: A → B
            tx1 = router1.functions.swapExactTokensForTokens(
                amount_in,
                1,  # Min output (will rely on Flashbots simulation)
                [Web3.to_checksum_address(token_a), Web3.to_checksum_address(token_b)],
                wallet,
                deadline
            ).build_transaction({
                'from': wallet,
                'gas': 250000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': base_nonce
            })

            # Build swap 2: B → C
            tx2 = router2.functions.swapExactTokensForTokens(
                1,  # Will be filled by actual output from tx1
                1,
                [Web3.to_checksum_address(token_b), Web3.to_checksum_address(token_c)],
                wallet,
                deadline
            ).build_transaction({
                'from': wallet,
                'gas': 250000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': base_nonce + 1
            })

            # Build swap 3: C → A
            tx3 = router3.functions.swapExactTokensForTokens(
                1,  # Will be filled by actual output from tx2
                int(amount_in * 1.001),  # Min output: at least original + 0.1% profit
                [Web3.to_checksum_address(token_c), Web3.to_checksum_address(token_a)],
                wallet,
                deadline
            ).build_transaction({
                'from': wallet,
                'gas': 250000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': base_nonce + 2
            })

            # Sign all transactions
            signed_tx1 = self.w3.eth.account.sign_transaction(tx1, self.private_key)
            signed_tx2 = self.w3.eth.account.sign_transaction(tx2, self.private_key)
            signed_tx3 = self.w3.eth.account.sign_transaction(tx3, self.private_key)

            # Try Flashbots bundle for atomic execution
            if hasattr(self, 'flashbots_executor') and self.flashbots_executor:
                current_block = self.w3.eth.block_number
                target_block = current_block + 1

                bundle = [
                    signed_tx1.rawTransaction.hex(),
                    signed_tx2.rawTransaction.hex(),
                    signed_tx3.rawTransaction.hex()
                ]

                # Simulate bundle first
                sim_result = await self.flashbots_executor.simulate_bundle(bundle, target_block)

                if sim_result and 'error' not in sim_result:
                    logger.info(f"Flashbots simulation passed for triangular arb")

                    # Send bundle
                    bundle_result = await self.flashbots_executor.send_bundle(bundle, target_block)

                    if bundle_result:
                        bundle_hash = bundle_result.get('bundleHash', 'bundle_sent')
                        logger.info(f"Flashbots bundle sent: {bundle_hash}")
                        return bundle_hash
                    else:
                        logger.warning("Flashbots bundle rejected")
                else:
                    logger.warning(f"Flashbots simulation failed: {sim_result}")

            # Fallback: Sequential execution (RISKY - not atomic!)
            logger.warning("⚠️ Flashbots not available - executing sequentially (not atomic!)")

            # Only proceed if we have enough profit margin for gas
            gas_estimate = 250000 * 3 * self.w3.eth.gas_price
            if profit_pct < 0.01:  # Less than 1%
                logger.warning("Profit too low for non-atomic execution")
                return None

            # Execute sequentially
            tx1_hash = self.w3.eth.send_raw_transaction(signed_tx1.rawTransaction)
            logger.info(f"Swap 1 sent: {tx1_hash.hex()}")

            # Wait for tx1
            receipt1 = self.w3.eth.wait_for_transaction_receipt(tx1_hash, timeout=60)
            if receipt1.status != 1:
                logger.error("Swap 1 failed!")
                return None

            # Continue with tx2
            tx2_hash = self.w3.eth.send_raw_transaction(signed_tx2.rawTransaction)
            receipt2 = self.w3.eth.wait_for_transaction_receipt(tx2_hash, timeout=60)
            if receipt2.status != 1:
                logger.error("Swap 2 failed!")
                return None

            # Complete with tx3
            tx3_hash = self.w3.eth.send_raw_transaction(signed_tx3.rawTransaction)
            receipt3 = self.w3.eth.wait_for_transaction_receipt(tx3_hash, timeout=60)
            if receipt3.status != 1:
                logger.error("Swap 3 failed!")
                return None

            return tx3_hash.hex()

        except Exception as e:
            logger.error(f"Triangular swap execution error: {e}")
            return None

    async def _log_trade(
        self,
        cycle_key: str,
        route: str,
        token_a: str,
        amount_in: int,
        profit_pct: float,
        tx_hash: str,
        dexes: List[str],
        symbol_a: str,
        symbol_b: str,
        symbol_c: str
    ):
        """Log triangular arb trade to database"""
        if not self.db_pool:
            return

        try:
            import uuid

            # Calculate values
            amount_eth = amount_in / 1e18 if symbol_a not in ['USDC', 'USDT'] else amount_in / 1e6 / self._eth_price
            entry_usd = amount_eth * self._eth_price
            profit_usd = entry_usd * profit_pct

            logger.info(f"💰 Triangular [{cycle_key}]: ${entry_usd:.2f} | Profit: ${profit_usd:.2f}")

            trade_id = f"tri_arb_{uuid.uuid4().hex[:12]}"

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
                    token_a,
                    'ethereum',
                    dexes[0],  # First DEX
                    dexes[2],  # Last DEX
                    'triangular',
                    self._eth_price,
                    self._eth_price * (1 + profit_pct),
                    amount_eth,
                    amount_eth,
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
                    self._eth_price,
                    json.dumps({
                        'type': 'triangular',
                        'cycle': cycle_key,
                        'route': route,
                        'symbols': [symbol_a, symbol_b, symbol_c],
                        'dexes': dexes,
                        'gas_price': self.gas_oracle.current_gas_gwei,
                        'dry_run': self.dry_run
                    })
                )
            logger.debug(f"💾 Logged triangular arb: {trade_id}")

        except Exception as e:
            logger.error(f"Error logging triangular trade: {e}")

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Triangular Arbitrage Engine Stopped")
