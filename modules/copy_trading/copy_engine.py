"""
Copy Trading Engine - Wallet Tracking (EVM + Solana)

Features:
- Real-time wallet monitoring on EVM and Solana
- Real trade execution (when DRY_RUN=false)
- Jupiter aggregator for Solana swaps
- Uniswap/DEX router for EVM swaps
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional
import aiohttp
import os
import ast
from datetime import datetime

from modules.base_module import (
    BaseModule,
    ModuleConfig,
    ModuleMetrics,
    ModuleStatus,
    ModuleType,
)
from core.dry_run import should_skip_live
from config.rpc_provider import RPCProvider

logger = logging.getLogger("CopyTradingEngine")

# Jupiter API (using lite-api.jup.ag/swap/v1)
JUPITER_QUOTE_API = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_API = "https://lite-api.jup.ag/swap/v1/swap"

# Common tokens
WSOL_MINT = "So11111111111111111111111111111111111111112"

# Stablecoin + native-SOL mints that must NEVER be treated as the
# "traded token" of a copy-trade (2026-05-21 operator fix). If we see a
# leader's USDC balance rise, that's the OUTPUT leg of a SELL, not a
# BUY of USDC. WSOL is included so SOL receipts are treated as quote-
# side too. The 5 stuck copytrading_trades rows where token_address =
# EPjFWdd... (USDC) are direct evidence of the pre-fix bug.
STABLECOIN_MINTS = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC (Solana)
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT (Solana)
    "USDH1SM1ojwWUga67PGrgFWUHibbjqMvuMaDkRJTgkX",   # USDH
    WSOL_MINT,                                        # native SOL wrapper
}

# EVM analogue. Lowercased for case-insensitive comparison.
EVM_STABLECOIN_ADDRESSES = {
    a.lower() for a in [
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC mainnet
        "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT mainnet
        "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
        "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC Base
        "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC Arbitrum
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH mainnet
        "0x4200000000000000000000000000000000000006",  # WETH Base/OP
        "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
    ]
}

# Etherscan API V2 - Multi-chain support with single API key
# See: https://docs.etherscan.io/etherscan-v2
ETHERSCAN_V2_API = "https://api.etherscan.io/v2/api"

# Supported EVM chains with their chain IDs
EVM_CHAINS = {
    'ethereum': {'chain_id': 1, 'name': 'Ethereum', 'symbol': 'ETH', 'aliases': ['eth', 'mainnet']},
    'base': {'chain_id': 8453, 'name': 'Base', 'symbol': 'ETH', 'aliases': ['base']},
    'arbitrum': {'chain_id': 42161, 'name': 'Arbitrum One', 'symbol': 'ETH', 'aliases': ['arb', 'arbitrum-one']},
    'bsc': {'chain_id': 56, 'name': 'BNB Smart Chain', 'symbol': 'BNB', 'aliases': ['bnb', 'binance']},
    'polygon': {'chain_id': 137, 'name': 'Polygon', 'symbol': 'MATIC', 'aliases': ['matic', 'poly']},
    'optimism': {'chain_id': 10, 'name': 'Optimism', 'symbol': 'ETH', 'aliases': ['op']},
    'avalanche': {'chain_id': 43114, 'name': 'Avalanche C-Chain', 'symbol': 'AVAX', 'aliases': ['avax']},
}

# Default chain for EVM wallets without chain suffix
DEFAULT_EVM_CHAIN = 'ethereum'

# MB-24: per-chain DEX routing. Only V2-API-compatible AMMs are supported in
# this commit. Arbitrum/Optimism (Uniswap V3 only) and Avalanche (TraderJoe
# v2 - different API) raise unsupported errors; tracked as follow-ups.
EVM_DEX_ROUTING = {
    'ethereum': {
        'router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Uniswap V2
        'wrapped_native': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
        'rpc_key': 'ETHEREUM_RPC',
    },
    'bsc': {
        'router': '0x10ED43C718714eb63d5aA57B78B54704E256024E',  # PancakeSwap V2
        'wrapped_native': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',  # WBNB
        'rpc_key': 'BSC_RPC',
    },
    'polygon': {
        'router': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',  # QuickSwap V2
        'wrapped_native': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',  # WMATIC
        'rpc_key': 'POLYGON_RPC',
    },
    'base': {
        'router': '0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24',  # Uniswap V2 on Base
        'wrapped_native': '0x4200000000000000000000000000000000000006',  # WETH on Base
        'rpc_key': 'BASE_RPC',
    },
}


class PriceFetcher:
    """Fetch real-time prices from CoinGecko"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # {symbol: (price, timestamp)}
        self._cache_ttl = 60  # 1 minute cache

    async def get_price(self, symbol: str) -> float:
        """Get current USD price for a token (sol, ethereum, etc.)"""
        now = datetime.now()

        # Check cache first
        if symbol in self._cache:
            price, cached_at = self._cache[symbol]
            if (now - cached_at).total_seconds() < self._cache_ttl:
                return price

        # Fetch from CoinGecko
        try:
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'sol': 'solana',
                'solana': 'solana',
                'eth': 'ethereum',
                'ethereum': 'ethereum',
                'btc': 'bitcoin',
                'bitcoin': 'bitcoin',
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

        # Last resort defaults
        defaults = {'sol': 200.0, 'solana': 200.0, 'eth': 2000.0, 'ethereum': 2000.0}
        return defaults.get(symbol.lower(), 1.0)


class CopyTradeExecutor:
    """Trade executor for Copy Trading module"""

    def __init__(self, dry_run: bool = True, db_pool=None):
        self.dry_run = dry_run
        self.db_pool = db_pool
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_fetcher = PriceFetcher()
        # Cross-module risk gate. Injected via set_risk_manager() by the
        # outer CopyTradingEngine after construction. Optional so DRY_RUN
        # and tests can run without a fully wired RiskManager.
        self.risk_manager = None

        # Credentials will be loaded asynchronously in initialize()
        self.solana_rpc_url = None
        self.solana_private_key = None
        self.solana_wallet = None
        self.evm_private_key = None
        self.evm_wallet = None
        # PM Wave-11: surfaced to dashboard via _persist_execution_wallets
        # so the funding panel renders the same WARNING badge it already
        # shows for DEX when the stored WALLET_ADDRESS secret has gone
        # stale vs PRIVATE_KEY derivation. Defaults are safe (no warning).
        self.stored_evm_wallet = None
        self.wallet_address_secret_mismatch = False
        self.web3_provider = None

    def set_risk_manager(self, risk_manager) -> None:
        """Inject the cross-module RiskManager so broadcast paths can be
        gated by validate_trade(). Setter pattern matches arbitrage and
        solana engines."""
        self.risk_manager = risk_manager

    async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
        """Get decrypted private key from secrets manager or environment."""
        try:
            value = None

            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                # Always re-initialize with db_pool if secrets was in bootstrap mode
                if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                    secrets.initialize(self.db_pool)
                value = await secrets.get_async(key_name)
            except Exception as e:
                logger.debug(f"Failed to get {key_name} from secrets manager: {e}")

            # Fallback to environment
            if not value:
                value = os.getenv(key_name)

            if not value:
                return None

            # Check if still encrypted (Fernet tokens start with gAAAAAB)
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
        """Initialize executor and load credentials"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Load all credentials from secrets manager (database/Docker secrets)
        from security.secrets_manager import secrets
        # Ensure the secrets manager is out of bootstrap mode so the
        # DB-backed encrypted Helius key resolves (issue 17). Mirrors the
        # guard in _get_decrypted_key.
        if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
            try:
                secrets.initialize(self.db_pool)
            except Exception:
                pass

        # Resolve Solana RPC. Wave-7 (issue 17b): prefer the Helius
        # endpoint when a HELIUS_API_KEY is configured (it lives in the
        # encrypted DB, so resolve it via get_async AFTER db_pool init).
        # Public RPCs throttle the 15s monitor loop and spam
        # "Solana RPC rate limited - backing off". Helius (paid) does not.
        # Fall back: PoolEngine SOLANA_RPC -> .env SOLANA_RPC_URL.
        helius_key = None
        try:
            helius_key = await secrets.get_async('HELIUS_API_KEY', log_access=False)
        except Exception as e:
            logger.debug(f"Helius key lookup failed: {e}")
        if not helius_key:
            helius_key = os.getenv('HELIUS_API_KEY')
        self.solana_rpc_url = None
        if helius_key:
            self.solana_rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
            logger.info("   Solana RPC: using Helius endpoint")
        if not self.solana_rpc_url:
            self.solana_rpc_url = (
                await RPCProvider.get_rpc('SOLANA_RPC')
                or os.getenv('SOLANA_RPC_URL')
            )
            if self.solana_rpc_url:
                logger.info("   Solana RPC: using PoolEngine/.env fallback (no Helius key)")

        # Load Solana credentials from secrets manager.
        # Wave-13 FIX (copy_engine.py:288 punch-list): ALWAYS derive the
        # public wallet address from the decrypted SOLANA_MODULE_PRIVATE_KEY
        # so the Jupiter swap payload uses the correct signer pubkey even when
        # SOLANA_MODULE_WALLET is absent / stale in secrets DB. Mirrors the
        # EVM derivation block below and the pattern used by sniper/solana
        # modules. The main_copy.py post-init workaround is now redundant
        # for fresh starts but remains as a belt-and-suspenders guard for any
        # CopyTradeExecutor built outside the normal engine path.
        self.solana_private_key = await self._get_decrypted_key('SOLANA_MODULE_PRIVATE_KEY')
        _stored_sol_wallet = secrets.get('SOLANA_MODULE_WALLET') or os.getenv('SOLANA_MODULE_WALLET')
        _derived_sol_wallet = None
        if self.solana_private_key:
            try:
                from solders.keypair import Keypair as _SoldersKP
                import base58 as _b58
                import json as _jmod
                _pk_raw = self.solana_private_key
                _sol_bytes = None
                if _pk_raw.startswith('['):
                    try:
                        _sol_bytes = bytes(_jmod.loads(_pk_raw))
                    except Exception:
                        pass
                if _sol_bytes is None:
                    try:
                        _sol_bytes = _b58.b58decode(_pk_raw)
                    except Exception:
                        pass
                if _sol_bytes is None:
                    try:
                        _sol_bytes = bytes.fromhex(_pk_raw)
                    except Exception:
                        pass
                if _sol_bytes is not None:
                    _sol_kp = None
                    if len(_sol_bytes) == 64:
                        _sol_kp = _SoldersKP.from_bytes(_sol_bytes)
                    elif len(_sol_bytes) == 32:
                        _sol_kp = _SoldersKP.from_seed(_sol_bytes)
                    if _sol_kp is not None:
                        _derived_sol_wallet = str(_sol_kp.pubkey())
            except Exception as _sol_err:
                logger.debug(f"copy Solana wallet derivation in executor.initialize failed: {_sol_err}")

        if _derived_sol_wallet:
            self.solana_wallet = _derived_sol_wallet
            if _stored_sol_wallet and _stored_sol_wallet != _derived_sol_wallet:
                _s_mask = (_stored_sol_wallet[:6] + "..." + _stored_sol_wallet[-4:]) if len(_stored_sol_wallet) >= 10 else "***"
                _d_mask = _derived_sol_wallet[:6] + "..." + _derived_sol_wallet[-4:]
                logger.warning(
                    "COPY SOLANA_MODULE_WALLET secret mismatch: stored=%s derived=%s. "
                    "Using DERIVED (SOLANA_MODULE_PRIVATE_KEY is authoritative). "
                    "Update the stored secret to suppress this warning.",
                    _s_mask, _d_mask,
                )
            else:
                _d_mask = _derived_sol_wallet[:6] + "..." + _derived_sol_wallet[-4:]
                logger.info(f"   Solana execution wallet derived from PK: {_d_mask}")
        else:
            # Derivation unavailable (solders not installed, or no PK). Fall
            # back to the stored secret. copy_solana_swap will refuse live
            # txs if solana_wallet remains None.
            self.solana_wallet = _stored_sol_wallet
            if not self.solana_wallet:
                logger.warning(
                    "copy executor: Solana wallet could not be derived from PK "
                    "and SOLANA_MODULE_WALLET secret is not set — Solana copies disabled"
                )

        # Load EVM credentials from secrets manager. Wave-11 FIX 3
        # (mirrors DEX FIX 1): ALWAYS derive the public address from the
        # decrypted PRIVATE_KEY. The stored WALLET_ADDRESS secret is
        # reference-only — when the operator rotates the key but not the
        # address row, the stored value becomes stale and funding it in
        # LIVE mode would lose money. The derived address is the
        # authoritative funding identity for copy trades.
        self.evm_private_key = await self._get_decrypted_key('PRIVATE_KEY')
        stored_evm = secrets.get('WALLET_ADDRESS') or os.getenv('WALLET_ADDRESS')
        derived_evm = None
        if self.evm_private_key:
            try:
                from eth_account import Account
                _pk = self.evm_private_key if self.evm_private_key.startswith('0x') else f'0x{self.evm_private_key}'
                derived_evm = Account.from_key(_pk).address
            except Exception as e:
                logger.debug(f"copy EVM wallet derivation failed: {e}")
        # Remember stored address regardless of derivation outcome so the
        # dashboard can display BOTH addresses (derived + stale).
        self.stored_evm_wallet = stored_evm or None
        if derived_evm:
            self.evm_wallet = derived_evm
            if stored_evm and stored_evm.lower() != derived_evm.lower():
                # Stale stored secret. Surface masked-address CRITICAL
                # so the operator updates it; copies still broadcast
                # from the derived (correct) address.
                self.wallet_address_secret_mismatch = True
                stored_mask = (stored_evm[:6] + "..." + stored_evm[-4:]) if len(stored_evm) >= 10 else "***"
                derived_mask = derived_evm[:6] + "..." + derived_evm[-4:]
                logger.critical(
                    "COPY WALLET_ADDRESS secret mismatch: stored=%s vs derived=%s. "
                    "Using DERIVED (PRIVATE_KEY is authoritative). Update the stored "
                    "secret — funding the stored address would lose money.",
                    stored_mask, derived_mask,
                )
            else:
                self.wallet_address_secret_mismatch = False
        else:
            # No PRIVATE_KEY: last-resort fallback to stored address.
            # copy_evm_swap will refuse the trade without a private key,
            # so this is reference-only.
            self.evm_wallet = stored_evm

        # Load Web3 provider. RPCProvider is already imported module-level
        # at line 27; re-importing here would shadow it as a local var and
        # break the earlier `await RPCProvider.get_rpc('SOLANA_RPC')` call
        # at line 219 with UnboundLocalError.
        try:
            self.web3_provider = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
        except Exception:
            pass
        if not self.web3_provider:
            self.web3_provider = secrets.get('WEB3_PROVIDER_URL') or os.getenv('WEB3_PROVIDER_URL')

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"💱 Copy Trade Executor initialized ({mode})")

        # Log credential status
        if self.solana_private_key:
            logger.info("✅ Solana credentials loaded")
        else:
            logger.warning("⚠️ Solana credentials not found (Solana copy trading disabled)")

        if self.evm_private_key:
            logger.info("✅ EVM credentials loaded")
        else:
            logger.warning("⚠️ EVM credentials not found (EVM copy trading disabled)")

    async def close(self):
        """Close executor"""
        if self.session:
            await self.session.close()
            self.session = None

    async def copy_solana_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount_lamports: int,
        slippage_bps: int = 100
    ) -> Dict:
        """Copy a Solana swap via Jupiter"""
        if should_skip_live(self.dry_run, module='copy_trading', account=getattr(self, 'solana_wallet', None)):
            return await self._simulate_solana_swap(input_mint, output_mint, amount_lamports)

        if not self.solana_wallet or not self.solana_private_key:
            return {'success': False, 'error': 'Solana wallet not configured'}

        # P1 cross-module risk gate (matches arbitrage_engine.py and
        # SOLANA engine pattern). Only BUYs (SOL -> token) are gated;
        # SELLs (token -> SOL) close existing exposure and must always
        # be allowed. Amount sent to RiskManager is lamports->SOL.
        SOL_MINT = 'So11111111111111111111111111111111111111112'
        is_buy = (input_mint == SOL_MINT)
        if is_buy and self.risk_manager is not None:
            try:
                amount_sol_equiv = amount_lamports / 1_000_000_000
                allowed, reason = await self.risk_manager.validate_trade(
                    output_mint, amount_sol_equiv
                )
            except Exception as e:
                logger.warning(f"validate_trade raised: {e}; refusing copy_solana_swap")
                return {'success': False, 'error': f'risk gate raised: {e}'}
            if not allowed:
                logger.warning(
                    f"⛔ Risk manager rejected COPY solana swap "
                    f"{output_mint[:10]}: {reason}"
                )
                return {'success': False, 'error': f'risk gate rejected: {reason}'}

        try:
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(input_mint, output_mint, amount_lamports, slippage_bps)
            if not quote:
                return {'success': False, 'error': 'Failed to get Jupiter quote'}

            # Get swap transaction
            swap_tx = await self._get_jupiter_swap(quote)
            if not swap_tx:
                return {'success': False, 'error': 'Failed to get swap transaction'}

            # Sign and send
            tx_hash = await self._sign_and_send_solana(swap_tx)
            if tx_hash:
                return {
                    'success': True,
                    'tx_hash': tx_hash,
                    'input_mint': input_mint,
                    'output_mint': output_mint,
                    'amount': amount_lamports
                }
            else:
                return {'success': False, 'error': 'Failed to send transaction'}

        except Exception as e:
            logger.error(f"Solana copy swap error: {e}")
            return {'success': False, 'error': str(e)}

    async def copy_evm_swap(
        self,
        token_address: str,
        amount_wei: int,
        is_buy: bool = True,
        slippage: float = 10.0,
        chain: str = 'ethereum',
    ) -> Dict:
        """Copy an EVM swap via a V2-API DEX router on `chain`."""
        # MB-24: route per source-tx chain. Previously hardcoded to Uniswap V2
        # mainnet + WETH mainnet + self.web3_provider (ETHEREUM_RPC) - every
        # non-ETH copy reverted or hit a wrong-token address collision.
        routing = EVM_DEX_ROUTING.get(chain)
        if not routing:
            return {
                'success': False,
                'error': f"Chain '{chain}' not supported by EVM copy executor (V2-API only).",
            }

        if should_skip_live(self.dry_run, module='copy_trading', account=getattr(self, 'evm_wallet', None)):
            return await self._simulate_evm_swap(token_address, amount_wei, is_buy)

        if not self.evm_wallet or not self.evm_private_key:
            return {'success': False, 'error': 'EVM wallet not configured'}

        # P1 cross-module risk gate. Only BUYs are gated — SELLs close
        # existing exposure and must always be allowed. Amount sent to
        # RiskManager is in chain-native (ETH) — wei / 1e18 — matching
        # arbitrage's ETH input convention so the gate sees consistent
        # units across both modules.
        if is_buy and self.risk_manager is not None:
            try:
                amount_eth = amount_wei / 1_000_000_000_000_000_000
                allowed, reason = await self.risk_manager.validate_trade(
                    token_address, amount_eth
                )
            except Exception as e:
                logger.warning(f"validate_trade raised: {e}; refusing copy_evm_swap")
                return {'success': False, 'error': f'risk gate raised: {e}'}
            if not allowed:
                logger.warning(
                    f"⛔ Risk manager rejected COPY {chain} swap "
                    f"{token_address[:10]}: {reason}"
                )
                return {'success': False, 'error': f'risk gate rejected: {reason}'}

        try:
            from web3 import Web3
            from config.pool_engine import PoolEngine

            pool = await PoolEngine.get_instance()
            rpc_url = await pool.get_endpoint(routing['rpc_key'])
            if not rpc_url:
                return {'success': False, 'error': f"No RPC endpoint available for {chain}"}

            w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not w3.is_connected():
                return {'success': False, 'error': f'Failed to connect to Web3 on {chain}'}

            ROUTER = routing['router']
            WRAPPED_NATIVE = routing['wrapped_native']

            ROUTER_ABI = [{
                "inputs": [
                    {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"}
                ],
                "name": "swapExactETHForTokens",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "payable",
                "type": "function"
            }]

            router = w3.eth.contract(address=Web3.to_checksum_address(ROUTER), abi=ROUTER_ABI)

            if is_buy:
                path = [Web3.to_checksum_address(WRAPPED_NATIVE), Web3.to_checksum_address(token_address)]
            else:
                path = [Web3.to_checksum_address(token_address), Web3.to_checksum_address(WRAPPED_NATIVE)]

            deadline = int(datetime.now().timestamp()) + 120

            # CRITICAL: Calculate minOut to prevent sandwich attacks
            # Get quote first to determine expected output
            try:
                # Add getAmountsOut to ABI for quote
                quote_abi = [{
                    "inputs": [
                        {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                        {"internalType": "address[]", "name": "path", "type": "address[]"}
                    ],
                    "name": "getAmountsOut",
                    "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "view",
                    "type": "function"
                }]
                quote_router = w3.eth.contract(address=Web3.to_checksum_address(ROUTER), abi=quote_abi)
                amounts = quote_router.functions.getAmountsOut(amount_wei, path).call()
                expected_out = amounts[-1]
                # Apply slippage tolerance (e.g., 10% slippage = accept 90% of expected)
                min_out = int(expected_out * (100 - slippage) / 100)
                logger.info(f"EVM Swap: Expected {expected_out}, minOut {min_out} ({slippage}% slippage)")
            except Exception as quote_error:
                logger.warning(f"Could not get quote, using 0 minOut (RISKY): {quote_error}")
                min_out = 0  # Fallback - still risky but at least we tried

            tx = router.functions.swapExactETHForTokens(
                min_out,  # Apply slippage protection
                path,
                Web3.to_checksum_address(self.evm_wallet),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.evm_wallet),
                'value': amount_wei,
                'gas': 300000,
                'nonce': w3.eth.get_transaction_count(self.evm_wallet)
            })

            signed = w3.eth.account.sign_transaction(tx, self.evm_private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)

            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'token': token_address,
                'amount': amount_wei
            }

        except Exception as e:
            logger.error(f"EVM copy swap error: {e}")
            return {'success': False, 'error': str(e)}

    async def _get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict]:
        """Get Jupiter quote"""
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps)
            }
            async with self.session.get(JUPITER_QUOTE_API, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Jupiter quote error: {e}")
        return None

    async def _get_jupiter_swap(self, quote: Dict) -> Optional[str]:
        """Get Jupiter swap transaction"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': self.solana_wallet,
                'wrapAndUnwrapSol': True,
                'prioritizationFeeLamports': 5000
            }
            async with self.session.post(JUPITER_SWAP_API, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('swapTransaction')
        except Exception as e:
            logger.debug(f"Jupiter swap error: {e}")
        return None

    async def _sign_and_send_solana(self, swap_tx_base64: str) -> Optional[str]:
        """Sign and send Solana transaction (supports JSON array, base58, hex key formats)"""
        try:
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            import base64
            import base58
            import json as json_module

            # Parse private key (supports multiple formats)
            pk = self.solana_private_key
            key_bytes = None

            # Format 1: JSON array
            if pk.startswith('['):
                try:
                    key_array = json_module.loads(pk)
                    key_bytes = bytes(key_array)
                except Exception:
                    pass

            # Format 2: Base58
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(pk)
                except Exception:
                    pass

            # Format 3: Hex
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(pk)
                except Exception:
                    pass

            if key_bytes is None:
                logger.error("Failed to parse private key")
                return None

            # Create keypair based on key length
            if len(key_bytes) == 64:
                keypair = Keypair.from_bytes(key_bytes)
            elif len(key_bytes) == 32:
                keypair = Keypair.from_seed(key_bytes)
            else:
                logger.error(f"Invalid key length: {len(key_bytes)} bytes")
                return None

            # Decode transaction
            tx_bytes = base64.b64decode(swap_tx_base64)
            tx = VersionedTransaction.from_bytes(tx_bytes)

            # Get message and verify pubkey match
            message = tx.message
            our_pubkey = keypair.pubkey()

            # Verify fee payer matches
            if hasattr(message, 'account_keys') and len(message.account_keys) > 0:
                fee_payer = message.account_keys[0]
                if str(fee_payer) != str(our_pubkey):
                    logger.error(f"❌ PUBKEY MISMATCH! TX expects: {fee_payer}, we have: {our_pubkey}")
                    return None

            # Sign using VersionedTransaction.populate() pattern
            # Note: tx.sign([keypair]) does NOT work with solders VersionedTransaction
            signature = keypair.sign_message(bytes(message))
            signed_tx = VersionedTransaction.populate(message, [signature])

            async with AsyncClient(self.solana_rpc_url) as client:
                result = await client.send_transaction(signed_tx)
                return str(result.value)

        except ImportError:
            logger.error("Solana libraries not installed")
        except Exception as e:
            logger.error(f"Solana transaction error: {e}")
        return None

    async def _simulate_solana_swap(self, input_mint: str, output_mint: str, amount: int) -> Dict:
        """Simulate Solana swap.

        Wave-12 FIX 2 — when this simulator is called for a SELL
        (input_mint = the SPL token; output_mint = WSOL), the caller in
        `_execute_solana_copy_trade` passes `amount_lamports=1` as a
        placeholder because we hold ZERO real SPL in DRY_RUN. The
        previous simulator echoed that 1-lamport value back, and
        `_log_copy_trade` then computed `usd_value = 1/1e9 * sol_price ≈
        $2e-7`, producing a fabricated -100% PnL on every closed copy.

        Fix: detect the SELL direction (input_mint != WSOL_MINT) and
        compute realistic SOL proceeds by re-quoting the token at its
        current USD price (Jupiter Price v3, same source we use to
        populate metadata.tokens_received on BUY), multiplying by the
        tokens_received recorded on the matching open position, and
        converting USD -> SOL -> lamports. Returns that lamport value
        in the `amount` field so the rest of the logging pipeline
        computes a realistic exit USD without further surgery.

        Fail-soft hierarchy:
        1. Tokens_received known + live Jupiter price -> realistic.
        2. Live price unavailable but entry_usd known -> exit_usd =
           entry_usd (0% PnL placeholder, NEVER -100%). Marks
           `sim_sell_no_price=True` in the returned `metadata` so
           _log_copy_trade can stamp it for forensic clarity.
        3. Neither known -> return the original placeholder but with
           `sim_sell_no_price=True` so the closer skips the -100% trap.
        """
        import hashlib
        fake_hash = hashlib.sha256(f"{input_mint}{output_mint}{datetime.now().timestamp()}".encode()).hexdigest()

        is_sell = (input_mint != WSOL_MINT) and (output_mint == WSOL_MINT)
        simulated_amount = amount
        sim_meta: Dict = {'sim_sell_no_price': False, 'sim_price_source': None}

        if is_sell:
            token_mint = input_mint
            tokens_held: Optional[float] = None
            entry_usd: Optional[float] = None
            try:
                if self.db_pool is not None:
                    async with self.db_pool.acquire() as conn:
                        row = await conn.fetchrow(
                            "SELECT entry_usd, "
                            "       (metadata::jsonb->>'tokens_received')::float8 AS tokens_received "
                            "FROM copytrading_trades "
                            "WHERE token_address = $1 AND status = 'open' AND side = 'buy' "
                            "ORDER BY entry_timestamp ASC LIMIT 1",
                            token_mint,
                        )
                        if row:
                            entry_usd = float(row['entry_usd']) if row['entry_usd'] is not None else None
                            tokens_held = float(row['tokens_received']) if row['tokens_received'] is not None else None
            except Exception as e:
                logger.debug(f"sim_sell DB lookup failed for {token_mint[:10]}: {e}")

            # Resolve current per-token USD price. Same Jupiter Price v3
            # source the dashboard's "Now" column uses, with last-known
            # cached fallback inside PriceFetcher equivalent.
            current_usd_per_token: Optional[float] = None
            try:
                from urllib.parse import quote_plus
                timeout = aiohttp.ClientTimeout(total=5)
                url = f"https://api.jup.ag/price/v3?ids={quote_plus(token_mint)}"
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            payload = data.get('data') if isinstance(data, dict) and 'data' in data else data
                            row = payload.get(token_mint) if isinstance(payload, dict) else None
                            if isinstance(row, dict):
                                price_raw = row.get('usdPrice') or row.get('price') or row.get('usd') or 0
                                try:
                                    p = float(price_raw)
                                    if p > 0:
                                        current_usd_per_token = p
                                        sim_meta['sim_price_source'] = 'jupiter_v3'
                                except (TypeError, ValueError):
                                    pass
                        else:
                            logger.debug(
                                f"sim_sell Jupiter v3 status {resp.status} for {token_mint[:10]}"
                            )
            except Exception as e:
                logger.debug(f"sim_sell Jupiter v3 fetch failed for {token_mint[:10]}: {e}")

            # Resolve SOL USD price for the lamports conversion.
            sol_usd = 0.0
            try:
                sol_usd = await self.price_fetcher.get_price('sol')
            except Exception:
                sol_usd = 0.0

            exit_usd_sim: Optional[float] = None
            if tokens_held and current_usd_per_token and tokens_held > 0:
                exit_usd_sim = float(tokens_held) * float(current_usd_per_token)
            elif entry_usd is not None and entry_usd > 0:
                # No live price: fall back to entry value (0% PnL) so we
                # don't fabricate -100%. Flagged for forensics.
                exit_usd_sim = float(entry_usd)
                sim_meta['sim_sell_no_price'] = True
                sim_meta['sim_price_source'] = 'fallback_entry_usd'
            else:
                # Neither tokens nor entry — keep the placeholder but
                # tell the closer not to compute -100%.
                sim_meta['sim_sell_no_price'] = True

            if exit_usd_sim is not None and sol_usd > 0:
                simulated_amount = max(1, int(round(exit_usd_sim / sol_usd * 1e9)))
                logger.info(
                    f"🧪 [DRY RUN] Simulated Solana SELL: token={token_mint[:10]} "
                    f"exit_usd=${exit_usd_sim:.4f} -> {simulated_amount} lamports "
                    f"(src={sim_meta['sim_price_source']})"
                )
            else:
                logger.warning(
                    f"🧪 [DRY RUN] Simulated Solana SELL had NO price source for "
                    f"{token_mint[:10]}; closer will use entry-as-exit fail-soft "
                    f"(sim_sell_no_price=True)"
                )
        else:
            logger.info(f"🧪 [DRY RUN] Simulated Solana swap: {simulated_amount} lamports")

        return {
            'success': True,
            'tx_hash': f"DRY_RUN_{fake_hash[:16]}",
            'input_mint': input_mint,
            'output_mint': output_mint,
            'amount': simulated_amount,
            'sim_metadata': sim_meta,
        }

    async def _simulate_evm_swap(self, token: str, amount: int, is_buy: bool) -> Dict:
        """Simulate EVM swap.

        Wave-12 FIX 2 (EVM mirror) — same -100% trap exists on the EVM
        SELL path in principle (caller would pass a tiny placeholder
        amount because we hold no real ERC-20 in DRY_RUN). We don't have
        a free EVM price oracle on hand the way Jupiter v3 covers Solana,
        so the fail-soft path is "exit_usd = entry_usd" for any SELL with
        a matching open position. Live BUY simulator unchanged.
        """
        import hashlib
        fake_hash = hashlib.sha256(f"{token}{amount}{datetime.now().timestamp()}".encode()).hexdigest()
        side = "BUY" if is_buy else "SELL"
        simulated_amount = amount
        sim_meta: Dict = {'sim_sell_no_price': False, 'sim_price_source': None}

        if not is_buy:
            entry_usd: Optional[float] = None
            try:
                if self.db_pool is not None:
                    async with self.db_pool.acquire() as conn:
                        row = await conn.fetchrow(
                            "SELECT entry_usd FROM copytrading_trades "
                            "WHERE token_address = $1 AND status = 'open' AND side = 'buy' "
                            "ORDER BY entry_timestamp ASC LIMIT 1",
                            token,
                        )
                        if row and row['entry_usd'] is not None:
                            entry_usd = float(row['entry_usd'])
            except Exception as e:
                logger.debug(f"sim_sell DB lookup failed for {token[:10]}: {e}")

            eth_usd = 0.0
            try:
                eth_usd = await self.price_fetcher.get_price('eth')
            except Exception:
                eth_usd = 0.0

            if entry_usd is not None and entry_usd > 0 and eth_usd > 0:
                # Fail-soft entry-as-exit so PnL is 0% not -100%. We
                # don't have a free per-token EVM price source wired
                # here; flagging this for the dashboard.
                simulated_amount = max(1, int(round(entry_usd / eth_usd * 1e18)))
                sim_meta['sim_sell_no_price'] = True
                sim_meta['sim_price_source'] = 'fallback_entry_usd'
                logger.info(
                    f"🧪 [DRY RUN] Simulated EVM SELL: token={token[:10]} "
                    f"exit_usd≈${entry_usd:.4f} -> {simulated_amount} wei "
                    f"(src=fallback_entry_usd)"
                )
            else:
                sim_meta['sim_sell_no_price'] = True
                logger.warning(
                    f"🧪 [DRY RUN] Simulated EVM SELL had NO entry basis "
                    f"for {token[:10]}; closer will fail-soft (sim_sell_no_price=True)"
                )
        else:
            logger.info(f"🧪 [DRY RUN] Simulated EVM {side}: {amount} wei")

        return {
            'success': True,
            'tx_hash': f"DRY_RUN_{fake_hash[:16]}",
            'token': token,
            'amount': simulated_amount,
            'sim_metadata': sim_meta,
        }


class CopyTradingEngine(BaseModule):
    """
    Copy Trading Engine for tracking and copying wallet trades.

    Features:
    - EVM wallet monitoring via Etherscan
    - Solana wallet monitoring via RPC
    - Real trade execution with Jupiter/Uniswap
    - Configurable copy amount and ratio

    BaseModule-compliant: killswitch poller auto-starts via
    __init_subclass__ wrap (2b4c61f). Per-module pause via
    logs/.pause_copy_trading honored through should_skip_live
    (CopyTradeExecutor migrated in 2b82404).
    """

    def __init__(self, config: Dict, db_pool):
        module_config = ModuleConfig(
            name="copy_trading",
            module_type=ModuleType.COPY_TRADING,
            enabled=bool(config.get('copy_trading_enabled', True)),
            custom_settings=config,
        )
        super().__init__(module_config)
        self.db_pool = db_pool
        self.config_dict = config  # raw dict retained for legacy reads
        self.targets = []  # Initialize empty, load from DB

        # Use Pool Engine with secrets manager fallback (NOT os.getenv directly).
        # RPCProvider is already imported at module level (line 27); a local
        # re-import would shadow it as a local and break other RPCProvider
        # uses in this method.
        from security.secrets_manager import secrets
        try:
            self.etherscan_api_key = RPCProvider.get_api_sync('ETHERSCAN_API') or secrets.get('ETHERSCAN_API_KEY')
            self.solana_rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC') or secrets.get('SOLANA_RPC_URL')
            self.helius_api_key = RPCProvider.get_api_sync('HELIUS_API') or secrets.get('HELIUS_API_KEY')
        except Exception:
            self.etherscan_api_key = secrets.get('ETHERSCAN_API_KEY')
            self.solana_rpc_url = secrets.get('SOLANA_RPC_URL')
            self.helius_api_key = secrets.get('HELIUS_API_KEY')

        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Copy trading settings
        # MB-23: this is the per-copy cap in USD. The EVM path historically
        # (incorrectly) treated it as ETH and multiplied by 1e18 - at $2000 ETH
        # that was a 2000x overrun of operator intent. EVM path now uses the
        # live ETH price (see _execute_evm_copy_trade) to convert USD -> wei,
        # matching the Solana path's USD/sol_price -> lamports convention.
        self.max_copy_amount = 100.0  # Max USD per copy
        self.copy_ratio = 10  # Copy 10% of original
        # Global concurrent-position cap. Per-leader cooldown bounds the
        # per-leader rate, but with many leaders the count of open
        # copy_trades positions can still climb unbounded. Cap defaults
        # to 50; tune via config_settings.copy_trading_config.max_active_positions.
        self.max_active_positions = 50

        # Wave-2: Kelly-fraction sizing per leader (CT-Q-02). When
        # enabled, _get_leader_kelly reads copy_leader_scores
        # (migration 023) and uses the persisted kelly_fraction as a
        # multiplier on max_copy_amount. Default OFF so existing
        # operator configs see no behavior change.
        #
        # Tunables (all reloaded by _load_settings from
        # config_settings.copytrading_config):
        #   kelly_sizing_enabled      — bool, default False
        #   kelly_probation_fraction  — fraction used for unscored /
        #                               stale-scored leaders (default 0.05
        #                               = 5% of cap; small but non-zero
        #                               so new leaders still trade)
        #   kelly_staleness_days      — score age before fallback to
        #                               probation (default 7d)
        self.kelly_sizing_enabled = False
        self.kelly_probation_fraction = 0.05
        self.kelly_staleness_days = 7

        # Wave-3 (CT-W3-01): slippage-decay tracker. Records
        # (leader_fill_price, our_fill_price, delta_ms) per mirrored
        # trade into copy_slippage_observations (migration 026).
        # Default ON because it's a side-channel write -- failure is
        # fail-soft, never breaks the trade pipeline. Operator can
        # disable via config_settings.copytrading_config.slippage_tracking_enabled.
        self.slippage_tracking_enabled = True

        # Wave-4 (CT-Q-09): per-leader probation gate. Consults the
        # `on_probation` + `probation_until` columns on copy_leader_scores
        # (migration 026) and refuses BUYs from leaders that are benched.
        # SELLs are NEVER gated -- they only reduce exposure. Probation
        # is set automatically when (a) a leader's composite score drops
        # below `probation_score_threshold` OR (b) a mirrored trade
        # closes worse than `-probation_loss_pct_threshold` percent.
        # All tunables reloaded from `copytrading_config`. Default ON.
        self.probation_gate_enabled = True
        self.probation_score_threshold = 30.0
        self.probation_loss_pct_threshold = 25.0   # absolute %; trigger on PnL < -25%
        self.probation_days = 7

        # Wave-4 (CT-Q-12): cross-module exposure cap. Per-token open
        # USD across DEX + SNIPER + SOLANA + COPY + AI. When the
        # intended buy would push the per-token total above
        # `cross_module_exposure_cap_usd`, the BUY is refused. Default
        # ON because the operator explicitly flagged this as desirable
        # (the 5-module fanout is otherwise un-bounded per-token).
        self.cross_module_exposure_check_enabled = True
        self.cross_module_exposure_cap_usd = 5000.0

        # Trade executor
        self.executor: Optional[CopyTradeExecutor] = None

        # Track known transactions to avoid duplicates. Bounded — the
        # sets grew unboundedly across weeks of leader monitoring,
        # eventually costing 10s of MB. Capped at _known_max with FIFO
        # eviction (oldest insertion drops first). 5000 covers ~17 days
        # of typical leader traffic at the 5-min cooldown rate; tune via
        # SNIPER doesn't apply, this is a copy-trading internal.
        self._known_tx_hashes = set()
        self._known_solana_sigs = set()
        self._known_tx_order: list = []   # insertion order for eviction
        self._known_sig_order: list = []
        self._known_max = 5000

        # Rate limiting - cooldown per wallet (5 min)
        self._wallet_last_copy_time: Dict[str, datetime] = {}
        self._wallet_cooldown_seconds = 300  # 5 minutes

        # Statistics
        self._stats = {
            'cycles': 0,
            'evm_copies': 0,
            'sol_copies': 0,
            'last_stats_log': datetime.now()
        }

        # 2026-05-21 operator fix: counter of BUYs refused because the
        # detector picked a stablecoin/WSOL mint. Read by
        # /api/copytrading/stats (`stablecoin_refusals` key).
        self._stablecoin_refusals = 0

    async def initialize(self) -> bool:
        """Initialize executor and load settings. Idempotent."""
        try:
            self.status = ModuleStatus.INITIALIZING
            # Initialize executor with db_pool for secrets manager access
            self.executor = CopyTradeExecutor(self.dry_run, db_pool=self.db_pool)

            # P1 cross-module risk gate. Construct a shared RiskManager
            # and inject into the executor so copy_solana_swap and
            # copy_evm_swap can validate before broadcast. Fail-soft.
            try:
                from core.risk_manager import RiskManager
                risk_manager = RiskManager(config=self.config_dict or {})
                self.executor.set_risk_manager(risk_manager)
                self.logger.info("✅ RiskManager wired into Copy Trading executor")
            except Exception as e:
                self.logger.warning(
                    f"RiskManager init failed (executor will run without cross-module gate): {e}"
                )

            await self.executor.initialize()
            # Adopt the executor's async-resolved Solana RPC (issue 17b):
            # the engine's __init__ resolves solana_rpc_url synchronously
            # BEFORE the db_pool/secrets bootstrap, so it can't see the
            # DB-stored Helius key. The executor resolves it correctly in
            # its async initialize(); the engine's _monitor_solana_wallets
            # reads self.solana_rpc_url, so mirror the resolved value here.
            if getattr(self.executor, 'solana_rpc_url', None):
                self.solana_rpc_url = self.executor.solana_rpc_url
            # Surface the resolved PUBLIC execution wallet addresses to the
            # dashboard (issue 15). Public addresses only; fail-soft.
            await self._persist_execution_wallets()
            await self._load_settings()
            return True
        except Exception as e:
            self.logger.error(f"Copy Trading initialize failed: {e}")
            self.error_message = str(e)
            self.status = ModuleStatus.ERROR
            return False

    async def start(self) -> bool:
        """Run the copy-trading monitor loop. Returns True on clean exit."""
        if self.executor is None:
            if not await self.initialize():
                return False
        self._running = True
        self.status = ModuleStatus.RUNNING
        self.start_time = datetime.now()
        logger.info("👯 Copy Trading Engine Started")
        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        if self.etherscan_api_key:
            supported_chains = ', '.join([info['name'] for info in EVM_CHAINS.values()])
            logger.info(f"   EVM monitoring: Enabled (Etherscan V2 API)")
            logger.info(f"   Supported chains: {supported_chains}")
            logger.info(f"   Wallet format: 0x...@chain (e.g., 0x1234...@base, 0x5678...@arb)")
        else:
            logger.info(f"   EVM monitoring: Disabled (no ETHERSCAN_API_KEY)")
        logger.info(f"   Solana monitoring: {'Enabled' if self.solana_rpc_url else 'Disabled (no SOLANA_RPC_URL)'}")

        while self._running:
            try:
                self._stats['cycles'] += 1

                # Reload settings periodically to catch updates
                await self._load_settings()

                if self.targets:
                    # Monitor EVM wallets
                    evm_copied = await self._monitor_evm_wallets()
                    self._stats['evm_copies'] += evm_copied

                    # Monitor Solana wallets
                    sol_copied = await self._monitor_solana_wallets()
                    self._stats['sol_copies'] += sol_copied

                # Operator manual-close requests via flag-file IPC.
                # Dashboard's POST /api/copytrading/positions/<id>/close
                # writes logs/.close_copy_<trade_id>; we honor it here.
                try:
                    await self._process_close_flag_files()
                except Exception as e:
                    logger.error(f"close-flag processor failed: {e}")

                # Log stats every 5 minutes
                await self._log_stats_if_needed()

                await asyncio.sleep(15)  # Poll every 15s
            except Exception as e:
                logger.error(f"Copy loop error: {e}")
                await asyncio.sleep(15)
        return True

    async def process_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """COPY discovers opportunities internally via wallet monitoring."""
        return None

    async def get_metrics(self) -> ModuleMetrics:
        """Return current ModuleMetrics snapshot (filled by update_metrics)."""
        try:
            await self.update_metrics()
        except Exception as e:
            self.logger.debug(f"update_metrics failed: {e}")
        return self.metrics

    async def _log_stats_if_needed(self):
        """Log statistics every 5 minutes"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            logger.info(f"📊 COPY TRADING STATS (Last 5 min): "
                       f"Cycles: {self._stats['cycles']} | "
                       f"Wallets: {len(self.targets)} | "
                       f"EVM Copies: {self._stats['evm_copies']} | "
                       f"Solana Copies: {self._stats['sol_copies']}")

            # Reset stats
            self._stats = {
                'cycles': 0,
                'evm_copies': 0,
                'sol_copies': 0,
                'last_stats_log': now
            }

    async def _load_settings(self):
        """Load Copy Trading settings from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT key, value FROM config_settings WHERE config_type = 'copytrading_config'")

                targets_loaded = []
                for row in rows:
                    key = row['key']
                    val = row['value']

                    if key == 'target_wallets':
                        if val:
                            try:
                                # Try parsing as JSON first
                                parsed = json.loads(val)
                                if isinstance(parsed, list):
                                    targets_loaded = [str(t).strip() for t in parsed if t]
                                else:
                                    targets_loaded = [val.strip()] if val.strip() else []
                            except json.JSONDecodeError:
                                try:
                                    # Try parsing as a list structure (e.g. "['0x1', '0x2']")
                                    parsed = ast.literal_eval(val)
                                    if isinstance(parsed, list):
                                        targets_loaded = [str(t).strip() for t in parsed if t]
                                    else:
                                        targets_loaded = [val.strip()]
                                except (ValueError, SyntaxError):
                                    # Fallback for comma/newline separated string
                                    targets_loaded = [t.strip() for t in val.replace(',', '\n').split('\n') if t.strip()]
                    elif key == 'max_active_positions':
                        try:
                            self.max_active_positions = int(val) if val else 50
                        except (TypeError, ValueError):
                            pass
                    elif key == 'max_copy_amount':
                        try:
                            self.max_copy_amount = float(val) if val else 100.0
                        except (TypeError, ValueError):
                            pass
                    elif key == 'kelly_sizing_enabled':
                        # Accept '1' / 'true' / 'yes' (DB values are
                        # often stringified booleans).
                        self.kelly_sizing_enabled = str(val).strip().lower() in (
                            '1', 'true', 'yes', 'on',
                        )
                    elif key == 'kelly_probation_fraction':
                        try:
                            v = float(val) if val else 0.05
                            # Clamp to a sane band so a fat-finger value
                            # can't size beyond quarter-Kelly.
                            self.kelly_probation_fraction = max(0.0, min(0.25, v))
                        except (TypeError, ValueError):
                            pass
                    elif key == 'kelly_staleness_days':
                        try:
                            self.kelly_staleness_days = max(
                                1.0, float(val) if val else 7.0,
                            )
                        except (TypeError, ValueError):
                            pass
                    elif key == 'slippage_tracking_enabled':
                        # Wave-3 CT-W3-01: default ON; operator can
                        # disable to skip the side-channel write.
                        self.slippage_tracking_enabled = str(val).strip().lower() in (
                            '1', 'true', 'yes', 'on',
                        )
                    elif key in ('probation_gate_enabled', 'copy_probation_gate_enabled'):
                        # Wave-4 CT-Q-09. Accept both the bare key and the
                        # `copy_`-prefixed alias seeded in migration 030.
                        self.probation_gate_enabled = str(val).strip().lower() in (
                            '1', 'true', 'yes', 'on',
                        )
                    elif key in ('probation_score_threshold', 'copy_probation_score_threshold'):
                        try:
                            v = float(val) if val else 30.0
                            # Clamp to the score domain (0..100).
                            self.probation_score_threshold = max(0.0, min(100.0, v))
                        except (TypeError, ValueError):
                            pass
                    elif key in ('probation_loss_pct_threshold', 'copy_probation_loss_pct_threshold'):
                        try:
                            v = float(val) if val else 25.0
                            # Always positive (we trigger on PnL < -v%).
                            # Cap at 95 so a fat-finger can't disable the gate.
                            self.probation_loss_pct_threshold = max(0.0, min(95.0, abs(v)))
                        except (TypeError, ValueError):
                            pass
                    elif key in ('probation_days', 'copy_probation_days'):
                        try:
                            v = float(val) if val else 7.0
                            # Min 1 day so an accidental 0 doesn't lift
                            # probation on the same tick it was set.
                            self.probation_days = max(1.0, min(365.0, v))
                        except (TypeError, ValueError):
                            pass
                    elif key in (
                        'cross_module_exposure_check_enabled',
                        'copy_cross_module_exposure_check_enabled',
                    ):
                        # Wave-4 CT-Q-12.
                        self.cross_module_exposure_check_enabled = str(val).strip().lower() in (
                            '1', 'true', 'yes', 'on',
                        )
                    elif key in (
                        'cross_module_exposure_cap_usd',
                        'copy_cross_module_exposure_cap_usd',
                    ):
                        try:
                            v = float(val) if val else 5000.0
                            # Clamp to a sane band -- 0 disables, but
                            # we cap the upper end at 10M USD to prevent
                            # a fat-finger from neutralising the gate.
                            self.cross_module_exposure_cap_usd = max(0.0, min(1e7, v))
                        except (TypeError, ValueError):
                            pass

                if targets_loaded != self.targets:
                    self.targets = targets_loaded
                    if self.targets:
                        logger.info(f"👯 Loaded {len(self.targets)} target wallets")

        except Exception as e:
            logger.warning(f"Failed to load Copy Trading settings: {e}")

    def _is_solana_address(self, address: str) -> bool:
        """Check if address is a Solana address (base58, typically 32-44 chars)"""
        # Strip any chain suffix first
        clean_addr = address.split('@')[0] if '@' in address else address
        # Solana addresses are base58 encoded, 32-44 chars, no 0x prefix
        if clean_addr.startswith('0x'):
            return False
        if len(clean_addr) < 32 or len(clean_addr) > 44:
            return False
        # Basic base58 character check
        base58_chars = set('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')
        return all(c in base58_chars for c in clean_addr)

    def _parse_evm_wallet(self, wallet: str) -> tuple:
        """
        Parse EVM wallet with optional chain suffix.
        Format: 0x...@chain (e.g., 0x1234...@base)
        Returns: (address, chain_name, chain_id)
        """
        if '@' in wallet:
            address, chain_hint = wallet.split('@', 1)
            chain_hint = chain_hint.lower().strip()
        else:
            address = wallet
            chain_hint = DEFAULT_EVM_CHAIN

        # Find chain by name or alias
        for chain_name, chain_info in EVM_CHAINS.items():
            if chain_hint == chain_name or chain_hint in chain_info['aliases']:
                return (address, chain_name, chain_info['chain_id'])

        # Default to Ethereum if chain not found
        logger.warning(f"Unknown chain '{chain_hint}' for {address[:10]}..., defaulting to Ethereum")
        return (address, 'ethereum', 1)

    async def _monitor_evm_wallets(self) -> int:
        """Check for new transactions from target EVM wallets via Etherscan V2 API.

        Supports multiple chains: Ethereum, Base, Arbitrum, BSC, Polygon, Optimism, Avalanche.
        Wallet format: 0x...@chain (e.g., 0x1234...@base, 0x5678...@arb)
        Default chain is Ethereum if no suffix specified.
        """
        trades_copied = 0

        if not self.etherscan_api_key:
            return 0

        # Filter EVM wallets (may include chain suffix like @base)
        evm_wallets = [w for w in self.targets if w.startswith('0x') or (w.split('@')[0].startswith('0x') if '@' in w else False)]
        if not evm_wallets:
            return 0

        # Group wallets by chain for logging
        chain_counts = {}
        for wallet in evm_wallets:
            _, chain_name, _ = self._parse_evm_wallet(wallet)
            chain_counts[chain_name] = chain_counts.get(chain_name, 0) + 1

        import time
        async with aiohttp.ClientSession() as session:
            for wallet_with_chain in evm_wallets:
                try:
                    # Parse wallet address and chain
                    address, chain_name, chain_id = self._parse_evm_wallet(wallet_with_chain)
                    chain_info = EVM_CHAINS.get(chain_name, EVM_CHAINS['ethereum'])

                    # Use Etherscan V2 API with chain ID
                    url = (
                        f"{ETHERSCAN_V2_API}?chainid={chain_id}"
                        f"&module=account&action=txlist"
                        f"&address={address}"
                        f"&startblock=0&endblock=99999999"
                        f"&page=1&offset=5"  # Only get last 5 txs
                        f"&sort=desc"
                        f"&apikey={self.etherscan_api_key}"
                    )

                    async with session.get(url) as resp:
                        # Handle rate limiting
                        if resp.status == 429:
                            logger.warning(f"⚠️ Etherscan rate limited on {chain_name} - backing off")
                            try:
                                await RPCProvider.report_rate_limit('ETHERSCAN_API', 'etherscan.io', 300)
                            except Exception:
                                pass
                            await asyncio.sleep(30)
                            continue

                        data = await resp.json()

                        # V2 API returns different status format
                        if data.get('status') == '1' and data.get('result'):
                            for tx in data['result']:
                                tx_hash = tx.get('hash')

                                # Skip if already processed
                                if tx_hash in self._known_tx_hashes:
                                    continue

                                # Check if recent (last minute)
                                if int(tx['timeStamp']) > time.time() - 60:
                                    self._remember_tx_hash(tx_hash)
                                    # Add chain info to tx for analysis
                                    tx['_chain'] = chain_name
                                    tx['_chain_id'] = chain_id
                                    tx['_chain_symbol'] = chain_info['symbol']
                                    if await self._analyze_and_copy_evm(tx):
                                        trades_copied += 1
                                        logger.info(f"📋 Copied trade on {chain_info['name']}: {tx_hash[:16]}...")

                        elif data.get('message') and 'rate limit' in data.get('message', '').lower():
                            logger.warning(f"⚠️ Etherscan V2 rate limited on {chain_name}")
                            await asyncio.sleep(5)

                except Exception as e:
                    logger.debug(f"Failed to check EVM wallet {wallet_with_chain}: {e}")

        return trades_copied

    async def _monitor_solana_wallets(self) -> int:
        """Check for new transactions from target Solana wallets"""
        trades_copied = 0

        if not self.solana_rpc_url:
            return 0

        sol_wallets = [w for w in self.targets if self._is_solana_address(w)]
        if not sol_wallets:
            return 0

        async with aiohttp.ClientSession() as session:
            for wallet in sol_wallets:
                try:
                    # Use getSignaturesForAddress to get recent transactions
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getSignaturesForAddress",
                        "params": [wallet, {"limit": 5}]
                    }

                    async with session.post(self.solana_rpc_url, json=payload) as resp:
                        # Handle rate limiting
                        if resp.status == 429:
                            logger.warning("⚠️ Solana RPC rate limited - backing off")
                            try:
                                # Don't rotate AWAY from Helius to a public
                                # endpoint — Helius is the high-rate provider
                                # (issue 17b). A public fallback would just
                                # 429 again. Only rotate when we're already on
                                # a non-Helius endpoint.
                                on_helius = 'helius' in (self.solana_rpc_url or '').lower()
                                if not on_helius:
                                    await RPCProvider.report_rate_limit('SOLANA_RPC', self.solana_rpc_url, 300)
                                    new_url = await RPCProvider.get_rpc('SOLANA_RPC')
                                    if new_url and new_url != self.solana_rpc_url:
                                        self.solana_rpc_url = new_url
                                        logger.info("🔄 Rotated to new Solana RPC")
                            except Exception:
                                pass
                            await asyncio.sleep(30)
                            continue

                        if resp.status != 200:
                            continue

                        data = await resp.json()
                        signatures = data.get('result', [])

                        for sig_info in signatures:
                            sig = sig_info.get('signature')
                            if not sig or sig in self._known_solana_sigs:
                                continue

                            # Check if transaction was successful and recent
                            if sig_info.get('err') is not None:
                                continue

                            # Check block time (within last 2 minutes)
                            block_time = sig_info.get('blockTime', 0)
                            import time
                            if block_time and block_time > time.time() - 120:
                                self._remember_sol_sig(sig)

                                # Analyze the transaction
                                if await self._analyze_and_copy_solana(wallet, sig):
                                    trades_copied += 1

                except Exception as e:
                    logger.debug(f"Failed to check Solana wallet {wallet}: {e}")

        return trades_copied

    def _log_replay_decision(
        self,
        *,
        chain: str,
        wallet: str,
        tx_hash: str,
        decision: str,
        reason: str,
        extra: Optional[Dict] = None,
    ) -> None:
        """Wave-2 replay diagnostics. Records WHY each detected leader
        tx was or was not mirrored. Operators previously had no way to
        tell the difference between a leader being silently skipped
        (cooldown / cap / risk-gate / unsupported chain) and the
        leader simply not trading.

        Stored in an in-memory ring buffer (capped at 500 entries) AND
        emitted at INFO level with a structured prefix so logs/<module>/
        is greppable. The dashboard surfaces the ring via the future
        GET /api/copytrading/replay endpoint (T1/T2 wave).

        decision in {'copied', 'skipped', 'rejected', 'simulated', 'error'}
        reason is a short tag like 'cooldown' / 'position_cap' /
            'unsupported_chain' / 'risk_gate' / 'no_token_extracted' /
            'amount_too_small' / 'no_executor' / 'not_a_swap' /
            'tx_failed'.
        """
        try:
            if not hasattr(self, "_replay_decisions"):
                self._replay_decisions = []
            entry = {
                "ts": datetime.now().isoformat(),
                "chain": chain,
                "wallet": (wallet or "")[:64],
                "tx_hash": (tx_hash or "")[:80],
                "decision": decision,
                "reason": reason,
            }
            if extra:
                entry["extra"] = extra
            self._replay_decisions.append(entry)
            # Bounded ring — 500 is enough for a few hours of dashboard
            # backscroll without eating memory across week-long runs.
            if len(self._replay_decisions) > 500:
                self._replay_decisions.pop(0)
            logger.info(
                f"[replay] {chain} wallet={wallet[:10]} "
                f"tx={(tx_hash or '')[:14]} decision={decision} reason={reason}"
                + (f" extra={extra}" if extra else "")
            )
        except Exception as e:
            logger.debug(f"_log_replay_decision failed (non-fatal): {e}")

    def get_replay_decisions(self, limit: int = 100) -> List[Dict]:
        """Return the most-recent replay-decision entries for the
        dashboard / Test Runner. Read-only — safe to call from any
        async or sync context."""
        ring = getattr(self, "_replay_decisions", [])
        if limit and limit > 0:
            return ring[-limit:]
        return list(ring)

    async def _get_leader_kelly(self, chain: str, wallet: str) -> float:
        """Wave-2 enhancement: fetch this leader's Kelly fraction from
        copy_leader_scores (migration 023).

        Returns 0.0 when:
          * leader has never been scored (no row, no history)
          * leader's score is below quarter-Kelly breakeven
          * the kelly_fraction column is NULL
          * the kelly_fraction is stale (last_scored_at > 7 days)
          * the DB lookup itself fails (fail-soft)

        Returns a value in (0, 0.25]. The engine uses this as a
        *multiplier* on the operator-set max_copy_amount cap. A leader
        that has been validated as profitable gets full size; a leader
        with no history or a bad score gets fractional / zero size.

        When the kelly-sizing feature is disabled (operator-tunable
        config_settings.copytrading_config.kelly_sizing_enabled), this
        returns 1.0 so the existing static cap behavior is preserved.
        """
        if not getattr(self, "kelly_sizing_enabled", False):
            return 1.0
        if not self.db_pool or not chain or not wallet:
            return 0.0
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT kelly_fraction, score, last_scored_at
                    FROM copy_leader_scores
                    WHERE chain = $1
                      AND lower(wallet_address) = lower($2)
                    LIMIT 1
                    """,
                    chain, wallet,
                )
        except Exception as e:
            logger.debug(f"_get_leader_kelly DB lookup failed: {e}")
            return 0.0
        if not row:
            # Unscored leader. Operator-set policy: 0.0 means "do not
            # size beyond a probationary fraction" — we return the
            # configured probation size instead of 0.0 so completely
            # new leaders still get a tiny mirror trade.
            return float(getattr(self, "kelly_probation_fraction", 0.05))
        kelly = row["kelly_fraction"]
        if kelly is None:
            return float(getattr(self, "kelly_probation_fraction", 0.05))
        # Staleness gate: scores older than kelly_staleness_days fall back
        # to probation. Operators tune via DB; default 7 days.
        try:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            staleness_days = float(getattr(self, "kelly_staleness_days", 7))
            last = row["last_scored_at"]
            if last is not None:
                if last.tzinfo is None:
                    last = last.replace(tzinfo=_tz.utc)
                if (_dt.now(_tz.utc) - last) > _td(days=staleness_days):
                    return float(getattr(self, "kelly_probation_fraction", 0.05))
        except Exception:
            pass
        return float(max(0.0, min(0.25, float(kelly))))

    async def _is_leader_on_probation(self, chain: str, wallet: str) -> tuple:
        """Wave-4 CT-Q-09 probation gate.

        Returns (on_probation_active, reason) where on_probation_active is
        True iff the leader's score row has `on_probation = TRUE` AND
        `probation_until > NOW()`. Re-entry is automatic on expiry: a
        stale `probation_until` value is treated as "not on probation"
        without needing the row to be cleared (cheap & idempotent).

        Fail-soft: any DB error, or the gate being globally disabled,
        returns (False, ''). We NEVER block a BUY because the gate
        itself failed -- the cap / risk-manager / position-cap gates
        remain in place as a safety net.
        """
        if not getattr(self, "probation_gate_enabled", True):
            return (False, '')
        if not self.db_pool or not chain or not wallet:
            return (False, '')
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT on_probation, probation_until, probation_reason
                    FROM copy_leader_scores
                    WHERE chain = $1
                      AND lower(wallet_address) = lower($2)
                    LIMIT 1
                    """,
                    chain, wallet,
                )
        except Exception as e:
            logger.debug(f"_is_leader_on_probation DB lookup failed: {e}")
            return (False, '')
        if not row or not row["on_probation"]:
            return (False, '')
        until = row["probation_until"]
        if until is None:
            return (False, '')
        try:
            from datetime import datetime as _dt, timezone as _tz
            now_utc = _dt.now(_tz.utc)
            if until.tzinfo is None:
                until = until.replace(tzinfo=_tz.utc)
            if until <= now_utc:
                return (False, '')  # expired -> auto re-entry
        except Exception:
            return (False, '')
        return (True, str(row["probation_reason"] or 'benched'))

    async def _maybe_set_probation(
        self,
        *,
        chain: str,
        wallet: str,
        reason: str,
        days: Optional[float] = None,
    ) -> bool:
        """UPSERT the probation columns on copy_leader_scores. Called
        from the closing leg of a losing mirrored trade and from the
        leader-scorer when a score drops below threshold.

        Idempotent: re-calling extends `probation_until` only if the
        new expiry is later than the existing one (so the longest
        cooldown wins). Fail-soft: returns False on any DB error.
        """
        if not self.db_pool or not chain or not wallet:
            return False
        d = float(days if days is not None else getattr(self, "probation_days", 7.0))
        d = max(1.0, min(365.0, d))
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO copy_leader_scores (
                        chain, wallet_address, source,
                        on_probation, probation_until, probation_reason,
                        probation_set_at, last_scored_at, discovered_at
                    )
                    VALUES ($1, lower($2), 'auto_probation',
                            TRUE, NOW() + ($3 || ' days')::INTERVAL,
                            $4, NOW(), NOW(), NOW())
                    ON CONFLICT (chain, wallet_address) DO UPDATE
                       SET on_probation     = TRUE,
                           probation_until  = GREATEST(
                               COALESCE(copy_leader_scores.probation_until, NOW()),
                               NOW() + ($3 || ' days')::INTERVAL
                           ),
                           probation_reason = EXCLUDED.probation_reason,
                           probation_set_at = NOW()
                    """,
                    chain, wallet, str(int(d)), reason[:120],
                )
            logger.warning(
                f"[copy] leader {wallet[:10]}... on {chain} placed on probation "
                f"for {int(d)}d (reason={reason})"
            )
            return True
        except Exception as e:
            logger.debug(f"_maybe_set_probation failed (fail-soft): {e}")
            return False

    async def _check_cross_module_exposure(
        self,
        *,
        chain: str,
        token_address: str,
        intended_buy_usd: float,
    ) -> tuple:
        """Wave-4 CT-Q-12 cross-module exposure gate.

        Returns (allow, existing_exposure_usd, breakdown). `allow` is
        False iff (existing + intended) > cap. Fail-soft: any error
        returns (True, 0.0, {}) -- the per-module caps remain in place
        as the safety net so we never block trading on a gate failure.

        The breakdown is only computed when we're about to refuse, to
        keep the success path cheap (single SUM per module instead of
        two queries per module).
        """
        if not getattr(self, "cross_module_exposure_check_enabled", True):
            return (True, 0.0, {})
        if not self.db_pool or not chain or not token_address:
            return (True, 0.0, {})
        cap = float(getattr(self, "cross_module_exposure_cap_usd", 5000.0))
        if cap <= 0:
            return (True, 0.0, {})
        try:
            from modules.copy_trading.exposure_aggregator import (
                get_exposure_usd, get_exposure_breakdown_usd,
            )
            existing = await get_exposure_usd(chain, token_address, self.db_pool)
            projected = float(existing or 0.0) + float(max(0.0, intended_buy_usd))
            if projected <= cap:
                return (True, float(existing or 0.0), {})
            # Over cap -- pay the cost of a per-module breakdown so the
            # operator can see in the replay log WHICH module is holding
            # the bulk of the existing exposure.
            breakdown = await get_exposure_breakdown_usd(
                chain, token_address, self.db_pool,
            )
            return (False, float(existing or 0.0), breakdown)
        except Exception as e:
            logger.debug(f"_check_cross_module_exposure failed (fail-soft): {e}")
            return (True, 0.0, {})

    async def _record_slippage_observation(
        self,
        *,
        chain: str,
        leader_wallet: str,
        token_address: str,
        side: str,
        leader_tx_hash: Optional[str],
        our_tx_hash: Optional[str],
        leader_fill_price_usd: Optional[float],
        our_fill_price_usd: Optional[float],
        leader_fill_ts: Optional[datetime],
        our_fill_ts: Optional[datetime],
        notes: Optional[Dict] = None,
    ) -> None:
        """Wave-3 CT-W3-01 slippage-decay recorder. Fail-soft -- never
        raises. Skipped silently when the operator disables tracking or
        the db_pool is unavailable. Called from _log_copy_trade so it
        sees the same leader / our-fill snapshot."""
        if not getattr(self, "slippage_tracking_enabled", True):
            return
        if not self.db_pool:
            return
        try:
            from modules.copy_trading.slippage_tracker import (
                SlippageObservation, persist_observation,
            )
            obs = SlippageObservation(
                chain=chain,
                leader_wallet=leader_wallet or "unknown",
                token_address=token_address or "unknown",
                side=side,
                leader_tx_hash=leader_tx_hash,
                our_tx_hash=our_tx_hash,
                leader_fill_price_usd=leader_fill_price_usd,
                our_fill_price_usd=our_fill_price_usd,
                leader_fill_ts=leader_fill_ts,
                our_fill_ts=our_fill_ts,
                is_simulated=bool(self.dry_run),
                notes=notes,
            )
            await persist_observation(self.db_pool, obs)
        except Exception as e:
            # Fail-soft: tracker errors must never break a mirrored trade.
            logger.debug(f"_record_slippage_observation failed: {e}")

    async def _analyze_and_copy_evm(self, tx) -> bool:
        """Analyze EVM transaction and execute copy if it's a swap"""
        try:
            wallet = tx.get('from', '')
            tx_hash = tx.get('hash', '')
            chain = tx.get('_chain', 'ethereum')

            # Check wallet cooldown first
            if self._check_wallet_cooldown(wallet):
                self._log_replay_decision(
                    chain=chain, wallet=wallet, tx_hash=tx_hash,
                    decision='skipped', reason='cooldown',
                )
                return False  # Skip silently - wallet in cooldown

            input_data = tx.get('input', '')
            if len(input_data) < 10:
                self._log_replay_decision(
                    chain=chain, wallet=wallet, tx_hash=tx_hash,
                    decision='skipped', reason='not_a_swap',
                    extra={'input_len': len(input_data)},
                )
                return False

            method_id = input_data[:10]
            # Common DEX Router methods
            SWAP_METHODS = {
                '0x7ff36ab5': 'swapExactETHForTokens',
                '0xb6f9de95': 'swapExactETHForTokensSupportingFeeOnTransferTokens',
                '0x18cbafe5': 'swapExactTokensForETH',
                '0x38ed1739': 'swapExactTokensForTokens',
                '0x5c11d795': 'swapExactTokensForTokensSupportingFeeOnTransferTokens',
            }

            if method_id in SWAP_METHODS:
                method_name = SWAP_METHODS[method_id]

                # Update cooldown before executing
                self._update_wallet_cooldown(wallet)

                logger.info(f"👯 EVM COPY TRIGGER: Wallet {wallet[:16]}... executed {method_name}")
                await self._execute_evm_copy_trade(tx, method_name)
                return True

            self._log_replay_decision(
                chain=chain, wallet=wallet, tx_hash=tx_hash,
                decision='skipped', reason='not_a_swap',
                extra={'method_id': method_id},
            )
            return False
        except Exception as e:
            logger.error(f"Error analyzing EVM tx: {e}")
            return False

    async def _execute_evm_copy_trade(self, source_tx, method_name: str):
        """Execute the same trade on EVM - handles both BUY and SELL"""
        tx_hash = source_tx.get('hash', 'unknown')
        logger.info(f"🚀 Copying EVM trade {tx_hash} ({method_name})")

        # MB-24: derive chain from monitor-stamped source_tx['_chain_id']; bail
        # out cleanly for chains where we have no V2-API router configured
        # (Arbitrum/Optimism = V3 only, Avalanche = TraderJoe v2 different API).
        chain_id = source_tx.get('_chain_id')
        chain_name = None
        for name, info in EVM_CHAINS.items():
            if info['chain_id'] == chain_id:
                chain_name = name
                break
        if not chain_name:
            logger.warning(f"Unknown chain_id {chain_id} on source tx {tx_hash} - skipping")
            self._log_replay_decision(
                chain=str(chain_id), wallet=source_tx.get('from', ''),
                tx_hash=tx_hash, decision='skipped',
                reason='unknown_chain_id', extra={'chain_id': chain_id},
            )
            return
        if chain_name not in EVM_DEX_ROUTING:
            logger.info(f"Skip copy on {chain_name}: no V2-API router configured (tx {tx_hash})")
            self._log_replay_decision(
                chain=chain_name, wallet=source_tx.get('from', ''),
                tx_hash=tx_hash, decision='skipped',
                reason='unsupported_chain',
                extra={'note': 'no V2-API router configured'},
            )
            return

        try:
            # Parse token from transaction
            # For swapExactETHForTokens, the token is in the path (input data)
            input_data = source_tx.get('input', '')
            original_value = int(source_tx.get('value', 0))

            # Detect trade direction based on method name
            # BUY: swapExactETHForTokens, swapETHForExactTokens
            # SELL: swapExactTokensForETH, swapTokensForExactETH
            is_buy = 'ForTokens' in method_name
            side = 'buy' if is_buy else 'sell'

            # Calculate copy amount (ratio of original).
            # MB-23 fix: the cap is in USD - convert to wei via live ETH price.
            # Previously `int(self.max_copy_amount * 1e18)` treated USD as ETH:
            # at ~$2000 ETH the cap was ~$200,000 (2000x operator intent).
            eth_price = (
                await self.executor.price_fetcher.get_price('eth')
                if self.executor else 3000
            )
            if not eth_price or eth_price <= 0:
                logger.error("Invalid ETH price - skipping copy to avoid bad sizing")
                return
            max_copy_wei = int((self.max_copy_amount / eth_price) * 1e18)
            # Wave-2 enhancement: per-leader Kelly multiplier on the
            # operator-set cap. When kelly_sizing_enabled is False
            # this returns 1.0 and behavior is unchanged.
            kelly_mult = await self._get_leader_kelly(
                chain_name, source_tx.get('from', '')
            )
            copy_amount = min(
                original_value * self.copy_ratio // 100,
                int(max_copy_wei * kelly_mult),
            )

            if copy_amount <= 0:
                logger.warning("Copy amount too small, skipping")
                self._log_replay_decision(
                    chain=chain_name, wallet=source_tx.get('from', ''),
                    tx_hash=tx_hash, decision='skipped',
                    reason='amount_too_small',
                    extra={
                        'copy_amount_wei': int(copy_amount),
                        'leader_value_wei': int(original_value),
                    },
                )
                return

            # Extract token address from input data (simplified)
            # In production, fully decode the ABI
            token_address = None
            if len(input_data) > 200:
                # Token is usually the last address in path
                # Path starts at offset 196 for swapExactETHForTokens
                try:
                    token_address = '0x' + input_data[-40:]
                except:
                    pass

            if not token_address:
                logger.warning("Could not extract token address from tx")
                self._log_replay_decision(
                    chain=chain_name, wallet=source_tx.get('from', ''),
                    tx_hash=tx_hash, decision='skipped',
                    reason='no_token_extracted',
                )
                return

            # 2026-05-21 operator fix: never mirror a BUY whose target is
            # a stablecoin / wrapped-native. The EVM extractor takes the
            # last 20 bytes of input_data as the path's terminal token --
            # for SELL methods (swapExactTokensForETH) the terminal is
            # WETH, which we'd otherwise be tempted to "buy".
            if is_buy and token_address.lower() in EVM_STABLECOIN_ADDRESSES:
                logger.error(
                    f"REFUSING BUY on stablecoin/WETH {token_address[:10]}... "
                    f"on {chain_name} (method={method_name})."
                )
                self._log_replay_decision(
                    chain=chain_name, wallet=source_tx.get('from', ''),
                    tx_hash=tx_hash, decision='skipped',
                    reason='stablecoin_not_tradeable',
                    extra={'token': token_address[:12], 'method': method_name},
                )
                self._bump_stablecoin_refusals()
                return

            logger.info(f"👯 Detected {side.upper()} trade for token {token_address[:20]}...")

            # Wave-4 CT-Q-09 probation gate. SELLs always allowed --
            # they reduce exposure -- but BUYs from a benched leader
            # are refused until `probation_until` expires.
            if is_buy:
                on_prob, prob_reason = await self._is_leader_on_probation(
                    chain_name, source_tx.get('from', ''),
                )
                if on_prob:
                    self._log_replay_decision(
                        chain=chain_name, wallet=source_tx.get('from', ''),
                        tx_hash=tx_hash, decision='skipped',
                        reason='probation',
                        extra={'probation_reason': prob_reason[:64]},
                    )
                    return

                # Wave-4 CT-Q-12 cross-module exposure cap. Sums per-token
                # open USD across DEX/SNIPER/SOLANA/COPY/AI; refuses if
                # existing + intended_buy > cap. intended_buy USD is
                # already known (we sized in USD before converting to wei).
                intended_usd = float(copy_amount) / 1e18 * float(eth_price)
                allow, existing_usd, breakdown = await self._check_cross_module_exposure(
                    chain=chain_name,
                    token_address=token_address,
                    intended_buy_usd=intended_usd,
                )
                if not allow:
                    self._log_replay_decision(
                        chain=chain_name, wallet=source_tx.get('from', ''),
                        tx_hash=tx_hash, decision='skipped',
                        reason='cross_module_cap',
                        extra={
                            'existing_usd': round(existing_usd, 2),
                            'intended_usd': round(intended_usd, 2),
                            'cap_usd': float(self.cross_module_exposure_cap_usd),
                            'breakdown': {k: round(v, 2) for k, v in breakdown.items()},
                        },
                    )
                    return

            # Global open-position cap; bounded SQL count so a fanout
            # of leaders can't blow past the operator's exposure budget.
            if await self._at_position_cap():
                self._log_replay_decision(
                    chain=chain_name, wallet=source_tx.get('from', ''),
                    tx_hash=tx_hash, decision='skipped',
                    reason='position_cap',
                    extra={'cap': self.max_active_positions},
                )
                return

            # Execute copy trade
            result = await self.executor.copy_evm_swap(
                token_address=token_address,
                amount_wei=copy_amount,
                is_buy=is_buy,
                chain=chain_name,
            )

            if result.get('success'):
                logger.info(f"✅ EVM Copy Trade {'Executed' if not self.dry_run else 'Simulated'} on {chain_name}: {result.get('tx_hash')}")

                # Log to database with source wallet and proper side
                source_wallet = source_tx.get('from', '')
                self._log_replay_decision(
                    chain=chain_name, wallet=source_wallet,
                    tx_hash=tx_hash,
                    decision='simulated' if self.dry_run else 'copied',
                    reason='success',
                    extra={'side': side, 'copy_wei': int(copy_amount)},
                )
                # Wave-3: leader fill timestamp from Etherscan tx
                # (`timeStamp` = unix epoch seconds). Threaded through to
                # the slippage tracker so delta_ms can be computed.
                leader_ts = None
                try:
                    ts_raw = source_tx.get('timeStamp')
                    if ts_raw is not None:
                        leader_ts = datetime.utcfromtimestamp(int(ts_raw))
                except Exception:
                    leader_ts = None
                await self._log_copy_trade(
                    chain_name, tx_hash, result, source_wallet,
                    side=side, token_address=token_address,
                    leader_fill_ts=leader_ts,
                )
            else:
                logger.error(f"❌ EVM Copy Trade Failed: {result.get('error')}")
                self._log_replay_decision(
                    chain=chain_name, wallet=source_tx.get('from', ''),
                    tx_hash=tx_hash, decision='error',
                    reason=str(result.get('error', 'unknown'))[:120],
                )

        except Exception as e:
            logger.error(f"Error executing EVM copy trade: {e}")

    def _check_wallet_cooldown(self, wallet: str) -> bool:
        """Check if wallet is in cooldown period. Returns True if we should skip."""
        now = datetime.now()
        last_copy = self._wallet_last_copy_time.get(wallet)

        if last_copy:
            elapsed = (now - last_copy).total_seconds()
            if elapsed < self._wallet_cooldown_seconds:
                return True  # Still in cooldown, skip

        return False  # Not in cooldown, proceed

    def _remember_tx_hash(self, tx_hash: str) -> None:
        """Track tx_hash as seen, evicting oldest entry when bounded by
        _known_max. Prevents unbounded set growth under continuous
        leader-monitoring traffic."""
        if tx_hash in self._known_tx_hashes:
            return
        self._known_tx_hashes.add(tx_hash)
        self._known_tx_order.append(tx_hash)
        if len(self._known_tx_order) > self._known_max:
            oldest = self._known_tx_order.pop(0)
            self._known_tx_hashes.discard(oldest)

    def _remember_sol_sig(self, sig: str) -> None:
        """Solana-signature variant of _remember_tx_hash."""
        if sig in self._known_solana_sigs:
            return
        self._known_solana_sigs.add(sig)
        self._known_sig_order.append(sig)
        if len(self._known_sig_order) > self._known_max:
            oldest = self._known_sig_order.pop(0)
            self._known_solana_sigs.discard(oldest)

    async def _at_position_cap(self) -> bool:
        """Return True if the global open-copy-trade count is at or above
        max_active_positions. Per-leader cooldown bounds per-leader rate;
        this gate bounds GLOBAL exposure across all leaders so a
        many-leader config can't fan out unbounded simultaneous positions.
        Fail-soft: a DB error returns False (don't block trading if the
        cap-check itself fails)."""
        if not self.db_pool or self.max_active_positions <= 0:
            return False
        try:
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM copytrading_trades WHERE status = 'open'"
                )
            count = int(count or 0)
            if count >= self.max_active_positions:
                logger.warning(
                    f"🛑 COPY CAP: {count}/{self.max_active_positions} open "
                    f"copy_trades positions — refusing new copy"
                )
                return True
            return False
        except Exception as e:
            logger.debug(f"COPY position-cap check failed (fail-soft): {e}")
            return False

    async def _has_open_copy_position(self, chain: str, token_address: str) -> bool:
        """Return True iff we already have at least one row in
        copytrading_trades with status='open' for the given chain+token.
        Used by the SELL replay path to refuse mirroring an exit on a
        token we never bought. Fail-soft: returns False on DB error."""
        if not self.db_pool or not token_address:
            return False
        try:
            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM copytrading_trades "
                    "WHERE chain = $1 AND token_address = $2 "
                    "AND status = 'open'",
                    chain, token_address,
                )
            return int(count or 0) > 0
        except Exception as e:
            logger.debug(f"_has_open_copy_position fail-soft: {e}")
            return False

    def _bump_stablecoin_refusals(self) -> None:
        """Increment in-memory counter of BUYs refused because the
        detector picked a stablecoin/WSOL mint. Exposed via
        /api/copytrading/stats. The [replay] log line is the
        durable forensic record."""
        try:
            self._stablecoin_refusals = getattr(
                self, '_stablecoin_refusals', 0
            ) + 1
        except Exception:
            self._stablecoin_refusals = 1

    def _update_wallet_cooldown(self, wallet: str):
        """Update wallet's last copy time"""
        self._wallet_last_copy_time[wallet] = datetime.now()

    async def _analyze_and_copy_solana(self, wallet: str, signature: str) -> bool:
        """Analyze Solana transaction and execute copy if it's a swap"""
        try:
            # Check wallet cooldown first
            if self._check_wallet_cooldown(wallet):
                self._log_replay_decision(
                    chain='solana', wallet=wallet, tx_hash=signature,
                    decision='skipped', reason='cooldown',
                )
                return False  # Skip silently - wallet in cooldown

            # Get transaction details
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.solana_rpc_url, json=payload) as resp:
                    if resp.status != 200:
                        return False

                    data = await resp.json()
                    tx = data.get('result')
                    if not tx:
                        return False

                    # Check if it's a swap transaction
                    # Look for Jupiter, Raydium, or other DEX programs
                    DEX_PROGRAMS = [
                        'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',   # Jupiter v6
                        'JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB',   # Jupiter v4
                        '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8', # Raydium V4
                        'CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK',  # Raydium CPMM
                    ]

                    # Get instructions from transaction
                    message = tx.get('transaction', {}).get('message', {})
                    instructions = message.get('instructions', [])

                    is_swap = False
                    for instr in instructions:
                        program_id = instr.get('programId', '')
                        if program_id in DEX_PROGRAMS:
                            is_swap = True
                            break

                    if is_swap:
                        # Update cooldown before executing
                        self._update_wallet_cooldown(wallet)

                        logger.info(f"👯 SOLANA COPY TRIGGER: Wallet {wallet[:16]}... executed swap {signature[:20]}...")
                        await self._execute_solana_copy_trade(wallet, signature, tx)
                        return True

                    return False

        except Exception as e:
            logger.error(f"Error analyzing Solana tx: {e}")
            return False

    async def _get_solana_token_balance(self, mint: str) -> tuple:
        """
        Fetch our Solana wallet's raw SPL balance for `mint`.
        Returns (raw_amount, decimals). (0, 0) on any failure / no account.
        """
        wallet = getattr(self.executor, 'solana_wallet', None) if self.executor else None
        rpc_url = self.solana_rpc_url
        if not wallet or not rpc_url:
            return (0, 0)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
        }
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(rpc_url, json=payload) as resp:
                    if resp.status != 200:
                        return (0, 0)
                    data = await resp.json()
            accounts = (data.get('result') or {}).get('value') or []
            if not accounts:
                return (0, 0)
            info = accounts[0]['account']['data']['parsed']['info']['tokenAmount']
            return (int(info.get('amount', 0)), int(info.get('decimals', 0)))
        except Exception as e:
            logger.error(f"Error fetching SPL balance for {mint[:8]}...: {e}")
            return (0, 0)

    async def _execute_solana_copy_trade(self, wallet: str, signature: str, tx_data: dict):
        """Execute the same trade on Solana - handles both BUY and SELL.

        2026-05-21 operator bugfix: previous logic iterated postTokenBalances
        and stopped at the FIRST non-WSOL mint with a delta. When a leader
        SOLD a token for USDC, USDC iterated first and was misread as the
        traded token -> we entered a "BUY USDC" position. The 5 stuck
        copytrading_trades rows where token_address=EPjFWdd... are
        direct evidence.

        New algorithm:
          1. Compute per-mint NET delta = sum(post.uiAmount) - sum(pre).
          2. Partition: base (mint not in STABLECOIN_MINTS) vs quote.
          3. The traded token = base mint with largest |delta|.
             STABLECOIN_MINTS members are never returned.
          4. is_buy = base_delta > 0 (leader received tokens).
          5. Quote-total is used as a cross-check warning log.
        """
        logger.info(f"🚀 Analyzing Solana trade {signature[:20]}...")

        try:
            # Extract token mints from transaction
            meta = tx_data.get('meta', {})
            post_balances = meta.get('postTokenBalances', [])
            pre_balances = meta.get('preTokenBalances', [])

            def _ui_amount(b):
                try:
                    return float(
                        (b.get('uiTokenAmount') or {}).get('uiAmount') or 0
                    )
                except (TypeError, ValueError):
                    return 0.0

            # Per-mint net delta (post - pre), summed across owner rows.
            delta_by_mint: dict = {}
            for b in pre_balances:
                m = b.get('mint')
                if m:
                    delta_by_mint[m] = delta_by_mint.get(m, 0.0) - _ui_amount(b)
            for b in post_balances:
                m = b.get('mint')
                if m:
                    delta_by_mint[m] = delta_by_mint.get(m, 0.0) + _ui_amount(b)

            # Drop near-zero dust deltas.
            DUST_EPSILON = 1e-9
            delta_by_mint = {
                m: d for m, d in delta_by_mint.items()
                if abs(d) > DUST_EPSILON
            }

            base_deltas = {
                m: d for m, d in delta_by_mint.items()
                if m not in STABLECOIN_MINTS
            }
            quote_deltas = {
                m: d for m, d in delta_by_mint.items()
                if m in STABLECOIN_MINTS
            }

            # Native-SOL fallback (closed WSOL accounts).
            sol_delta = 0.0
            try:
                pre_sol = meta.get('preBalances') or []
                post_sol = meta.get('postBalances') or []
                if pre_sol and post_sol:
                    sol_delta = (int(post_sol[0]) - int(pre_sol[0])) / 1e9
            except Exception:
                sol_delta = 0.0

            token_mint = None
            base_delta = 0.0
            if base_deltas:
                token_mint = max(
                    base_deltas.keys(), key=lambda m: abs(base_deltas[m])
                )
                base_delta = base_deltas[token_mint]

            if not token_mint:
                logger.warning(
                    "No non-stablecoin token in tx -- likely stable<->stable "
                    "swap. Skipping."
                )
                self._log_replay_decision(
                    chain='solana', wallet=wallet, tx_hash=signature,
                    decision='skipped', reason='no_token_extracted',
                    extra={
                        'mints_seen': list(delta_by_mint.keys())[:6],
                        'sol_delta': round(sol_delta, 6),
                    },
                )
                return

            # Defense-in-depth: refuse outright if detector picked stable.
            if token_mint in STABLECOIN_MINTS:
                logger.error(
                    f"REFUSING: detector picked stablecoin "
                    f"{token_mint[:8]}... as traded token."
                )
                self._log_replay_decision(
                    chain='solana', wallet=wallet, tx_hash=signature,
                    decision='skipped', reason='stablecoin_not_tradeable',
                    extra={'token': token_mint[:16]},
                )
                self._bump_stablecoin_refusals()
                return

            is_buy = base_delta > 0
            side = 'buy' if is_buy else 'sell'

            quote_total = sum(quote_deltas.values()) + sol_delta
            if quote_total != 0 and ((quote_total > 0) == is_buy):
                logger.warning(
                    f"BUY/SELL cross-check disagrees for {signature[:16]}: "
                    f"base={base_delta:+.4f} quote={quote_total:+.4f} -- "
                    f"trusting base ({side})"
                )

            logger.info(
                f"👯 Detected {side.upper()} for {token_mint[:16]}... "
                f"(base_delta={base_delta:+.6f}, "
                f"quote_total={quote_total:+.4f})"
            )

            # Calculate copy amount
            sol_price = await self.executor.price_fetcher.get_price('sol') if self.executor else 200
            # Wave-2 enhancement: per-leader Kelly multiplier on the
            # operator-set cap (Solana path). The hard 0.1-SOL ceiling
            # is preserved as a final fuse so a mis-scored leader can
            # never blow past the operator-visible per-trade limit.
            kelly_mult = await self._get_leader_kelly('solana', wallet)
            usd_cap = self.max_copy_amount * kelly_mult
            copy_lamports = int(min(usd_cap / sol_price, 0.1) * 1e9)

            if is_buy:
                # Wave-4 CT-Q-09 probation gate. SELLs are NEVER gated
                # (they only close exposure); only BUYs from a benched
                # leader are refused until `probation_until` expires.
                on_prob, prob_reason = await self._is_leader_on_probation(
                    'solana', wallet,
                )
                if on_prob:
                    self._log_replay_decision(
                        chain='solana', wallet=wallet, tx_hash=signature,
                        decision='skipped', reason='probation',
                        extra={'probation_reason': prob_reason[:64]},
                    )
                    return

                # Wave-4 CT-Q-12 cross-module exposure cap. Same shape
                # as the EVM path. intended_buy USD = lamports / 1e9 *
                # sol_price (mirrors the dashboard convention).
                intended_usd = float(copy_lamports) / 1e9 * float(sol_price)
                allow, existing_usd, breakdown = await self._check_cross_module_exposure(
                    chain='solana',
                    token_address=token_mint,
                    intended_buy_usd=intended_usd,
                )
                if not allow:
                    self._log_replay_decision(
                        chain='solana', wallet=wallet, tx_hash=signature,
                        decision='skipped', reason='cross_module_cap',
                        extra={
                            'existing_usd': round(existing_usd, 2),
                            'intended_usd': round(intended_usd, 2),
                            'cap_usd': float(self.cross_module_exposure_cap_usd),
                            'breakdown': {k: round(v, 2) for k, v in breakdown.items()},
                        },
                    )
                    return

                # Global open-position cap — only gates BUYs because
                # SELLs close existing exposure and should never be
                # blocked by the cap.
                if await self._at_position_cap():
                    self._log_replay_decision(
                        chain='solana', wallet=wallet, tx_hash=signature,
                        decision='skipped', reason='position_cap',
                        extra={'cap': self.max_active_positions},
                    )
                    return
                # Execute BUY copy trade
                result = await self.executor.copy_solana_swap(
                    input_mint=WSOL_MINT,
                    output_mint=token_mint,
                    amount_lamports=copy_lamports
                )
            else:
                # SELL: only mirror if WE hold the token. Check BOTH
                # on-chain SPL balance (LIVE mode) AND open copytrading
                # _trades rows (DRY_RUN mode, where SPL=0).
                raw_balance, _decimals = await self._get_solana_token_balance(token_mint)
                has_db_position = await self._has_open_copy_position(
                    'solana', token_mint,
                )
                if raw_balance <= 0 and not has_db_position:
                    logger.warning(
                        f"⚠️ Leader SELL for {token_mint[:8]}... but we "
                        f"hold no position (on-chain=0, db_open=False)."
                    )
                    self._log_replay_decision(
                        chain='solana', wallet=wallet, tx_hash=signature,
                        decision='skipped',
                        reason='leader_sold_we_dont_hold',
                        extra={
                            'token': token_mint[:16],
                            'on_chain_raw': raw_balance,
                            'db_open': has_db_position,
                        },
                    )
                    return
                # DRY_RUN: SPL=0 because no real buy happened; pass
                # placeholder so executor can simulate the exit.
                exit_amount = raw_balance if raw_balance > 0 else 1
                result = await self.executor.copy_solana_swap(
                    input_mint=token_mint,
                    output_mint=WSOL_MINT,
                    amount_lamports=exit_amount,
                    slippage_bps=300,  # memecoin-tolerant exit
                )

            if result.get('success'):
                action = 'Tracked' if result.get('is_sell_tracking') else ('Executed' if not self.dry_run else 'Simulated')
                logger.info(f"✅ Solana Copy Trade {action}: {result.get('tx_hash')}")
                self._log_replay_decision(
                    chain='solana', wallet=wallet, tx_hash=signature,
                    decision='simulated' if self.dry_run else 'copied',
                    reason='success',
                    extra={'side': side, 'token': token_mint[:16]},
                )

                # Wave-3: leader fill timestamp from Solana RPC
                # (tx_data['blockTime'] = unix epoch seconds at top level).
                leader_ts = None
                try:
                    bt = tx_data.get('blockTime') if tx_data else None
                    if bt is not None:
                        leader_ts = datetime.utcfromtimestamp(int(bt))
                except Exception:
                    leader_ts = None
                # Log to database with proper side (BUY or SELL)
                await self._log_copy_trade(
                    'solana', signature, result, wallet,
                    side=side, token_address=token_mint,
                    leader_fill_ts=leader_ts,
                )
            else:
                logger.error(f"❌ Solana Copy Trade Failed: {result.get('error')}")
                err = result.get('error', 'unknown')
                # risk-gate rejections come back as 'risk gate rejected: ...'
                reason = 'risk_gate' if 'risk gate' in str(err) else str(err)[:120]
                self._log_replay_decision(
                    chain='solana', wallet=wallet, tx_hash=signature,
                    decision='rejected' if 'risk' in str(err) else 'error',
                    reason=reason,
                )

        except Exception as e:
            logger.error(f"Error executing Solana copy trade: {e}")

    async def _log_copy_trade(
        self,
        chain: str,
        source_tx: str,
        result: Dict,
        source_wallet: str = None,
        side: str = 'buy',
        token_address: str = None,
        leader_fill_ts: Optional[datetime] = None,
    ):
        """Log copy trade to database with P&L calculation for sells.

        Wave-3: `leader_fill_ts` is the leader's on-chain fill time
        (EVM `timeStamp`, Solana `blockTime`). Threaded through so the
        slippage tracker can compute `delta_ms = our_fill - leader_fill`.
        """
        if not self.db_pool:
            return

        try:
            import uuid
            now = datetime.now()

            # Get real prices from API
            amount = result.get('amount', 0)
            if chain == 'solana':
                # Convert lamports to SOL, get real price
                amount_native = amount / 1e9
                native_price = await self.executor.price_fetcher.get_price('sol') if self.executor else 200
                usd_value = amount_native * native_price
            else:
                # Convert wei to ETH, get real price
                amount_native = amount / 1e18
                native_price = await self.executor.price_fetcher.get_price('eth') if self.executor else 3000
                usd_value = amount_native * native_price

            token_addr = token_address or result.get('output_mint') or result.get('token', 'UNKNOWN')
            trade_id = f"copy_{uuid.uuid4().hex[:12]}"

            logger.info(f"💰 {side.upper()} Trade: {amount_native:.4f} {'SOL' if chain == 'solana' else 'ETH'} @ ${native_price:.2f} = ${usd_value:.2f}")

            async with self.db_pool.acquire() as conn:
                if side == 'sell':
                    # SELL: Find matching open BUY position and close it with P&L
                    open_trade = await conn.fetchrow("""
                        SELECT trade_id, entry_usd, entry_price, amount, entry_timestamp
                        FROM copytrading_trades
                        WHERE source_wallet = $1
                          AND token_address = $2
                          AND status = 'open'
                          AND side = 'buy'
                        ORDER BY entry_timestamp ASC
                        LIMIT 1
                    """, source_wallet or 'unknown', token_addr)

                    if open_trade:
                        # Found matching open position - close it with P&L.
                        # Wave-12 FIX 2: when the DRY_RUN SELL simulator
                        # could not source a real per-token price (flagged
                        # via result.sim_metadata.sim_sell_no_price), we
                        # fail SOFT — use entry_usd as exit_usd so the
                        # dashboard records a 0% PnL row labelled
                        # `sim_sell_no_price=true` instead of a fabricated
                        # -100%. This was the operator-reported root cause
                        # of every closed copy showing $0 exit / -100%.
                        entry_usd = float(open_trade['entry_usd'])
                        sim_meta = result.get('sim_metadata') or {}
                        sim_no_price = bool(sim_meta.get('sim_sell_no_price'))
                        if self.dry_run and sim_no_price and (usd_value <= 0 or usd_value < entry_usd * 0.001):
                            # Fail-soft path: pretend we got entry value back.
                            exit_usd = entry_usd
                            logger.info(
                                f"🧪 [DRY RUN] Sim SELL had no price feed for "
                                f"{token_addr[:16]}; recording entry-as-exit "
                                f"(0% PnL placeholder, NOT a real -100%)."
                            )
                        else:
                            exit_usd = usd_value
                        profit_loss = exit_usd - entry_usd
                        profit_loss_pct = ((exit_usd / entry_usd) - 1) * 100 if entry_usd > 0 else 0

                        # Update the existing trade to closed status with P&L
                        await conn.execute("""
                            UPDATE copytrading_trades
                            SET status = 'closed',
                                exit_price = $1,
                                exit_usd = $2,
                                profit_loss = $3,
                                profit_loss_pct = $4,
                                exit_timestamp = $5
                            WHERE trade_id = $6
                        """, native_price, exit_usd, profit_loss, profit_loss_pct, now, open_trade['trade_id'])

                        pnl_emoji = "📈" if profit_loss >= 0 else "📉"
                        logger.info(f"{pnl_emoji} Position CLOSED: P&L ${profit_loss:.2f} ({profit_loss_pct:.1f}%) for {token_addr[:16]}...")
                        logger.debug(f"💾 Updated trade {open_trade['trade_id']} to closed status")

                        # Wave-4 CT-Q-09 probation trigger. If this
                        # close was a "mirrored trade goes >X% negative"
                        # event, bench the leader for `probation_days`.
                        # We compare against the absolute threshold and
                        # only trigger on the loss side (positive pct =
                        # profit, no action).
                        try:
                            loss_thr = float(getattr(self, "probation_loss_pct_threshold", 25.0))
                            if (
                                source_wallet
                                and profit_loss_pct is not None
                                and float(profit_loss_pct) <= -abs(loss_thr)
                            ):
                                await self._maybe_set_probation(
                                    chain=chain,
                                    wallet=source_wallet,
                                    reason=f"loss_{float(profit_loss_pct):.1f}pct",
                                )
                        except Exception as e:
                            logger.debug(f"probation-trigger eval failed: {e}")
                    else:
                        # No matching open position - log as a standalone sell
                        logger.info(f"⚠️ No matching open position found for {token_addr[:16]}... - logging as standalone sell")
                        await conn.execute("""
                            INSERT INTO copytrading_trades (
                                trade_id, token_address, chain, source_wallet, source_tx,
                                side, entry_price, exit_price, amount,
                                entry_usd, exit_usd, profit_loss, profit_loss_pct,
                                status, is_simulated, entry_timestamp, exit_timestamp,
                                tx_hash, native_price_at_trade, metadata
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                        """,
                            trade_id, token_addr, chain, source_wallet or 'unknown', source_tx,
                            'sell', native_price, native_price, amount_native,
                            0.0, usd_value, 0.0, 0.0,
                            'closed', self.dry_run, now, now,
                            result.get('tx_hash'), native_price,
                            json.dumps({'dry_run': self.dry_run, 'no_matching_buy': True})
                        )
                else:
                    # BUY: Record new open position.
                    # SCHEMA SEMANTICS (operator-reported bug):
                    #   entry_price column = USD per TOKEN (not native price)
                    #   amount column      = TOKEN count (not SOL amount)
                    #   entry_usd column   = USD spent on the position
                    #   native_price_at_trade column = SOL/ETH USD price at trade time
                    # Previously the engine wrote native_price + amount_native,
                    # which caused the dashboard to show '-100% loss on USDC'
                    # because (token_price - SOL_price) × SOL_amount is
                    # meaningless. Stash tokens_received in metadata AND use
                    # it to compute proper token-level entry_price + amount.
                    # Fail-soft: if we can't get the price, fall back to the
                    # legacy native_price + amount_native and skip the metadata
                    # field so the dashboard knows to show 'PnL pending'.
                    tokens_received = None
                    try:
                        if chain == 'solana' and usd_value and usd_value > 0:
                            tokens_received = await self._estimate_tokens_received(
                                token_addr, float(usd_value)
                            )
                    except Exception as e:
                        logger.debug(f"_estimate_tokens_received failed: {e}")
                    meta = {'dry_run': self.dry_run}
                    if tokens_received and tokens_received > 0:
                        meta['tokens_received'] = tokens_received
                        token_entry_price_usd = float(usd_value) / tokens_received
                        amount_for_db = tokens_received
                    else:
                        # Legacy fallback — preserve the pre-fix behavior on
                        # rows where we can't get a quote (e.g. brand-new
                        # token that Jupiter doesn't index yet).
                        token_entry_price_usd = native_price
                        amount_for_db = amount_native
                    await conn.execute("""
                        INSERT INTO copytrading_trades (
                            trade_id, token_address, chain, source_wallet, source_tx,
                            side, entry_price, exit_price, amount,
                            entry_usd, exit_usd, profit_loss, profit_loss_pct,
                            status, is_simulated, entry_timestamp,
                            tx_hash, native_price_at_trade, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    """,
                        trade_id, token_addr, chain, source_wallet or 'unknown', source_tx,
                        'buy', token_entry_price_usd, 0.0, amount_for_db,
                        usd_value, 0.0, 0.0, 0.0,
                        'open' if result.get('success') else 'failed', self.dry_run, now,
                        result.get('tx_hash'), native_price,
                        json.dumps(meta)
                    )
                    logger.debug(f"💾 Logged BUY to copytrading_trades: {trade_id}")

                # Update wallet stats if source_wallet provided
                if source_wallet and result.get('success'):
                    await self._update_wallet_stats(conn, source_wallet)

            # Wave-3 CT-W3-01: record slippage observation (fail-soft).
            # leader_fill_price == our_fill_price == native_price is a
            # best-effort proxy when both fills are within seconds; the
            # operationally interesting metric here is delta_ms. When a
            # caller passes a token-level USD spot from a finer source
            # (e.g. Jupiter quote), this hook can be extended without
            # touching the trade pipeline.
            if result.get('success'):
                try:
                    await self._record_slippage_observation(
                        chain=chain,
                        leader_wallet=source_wallet or 'unknown',
                        token_address=token_addr,
                        side=side,
                        leader_tx_hash=source_tx,
                        our_tx_hash=result.get('tx_hash'),
                        leader_fill_price_usd=float(native_price)
                            if native_price else None,
                        our_fill_price_usd=float(native_price)
                            if native_price else None,
                        leader_fill_ts=leader_fill_ts,
                        our_fill_ts=now,
                        notes={
                            'amount_native': float(amount_native or 0),
                            'usd_value': float(usd_value or 0),
                            'dry_run': bool(self.dry_run),
                        },
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error logging copy trade: {e}")

    async def _update_wallet_stats(self, conn, wallet: str):
        """Update wallet statistics after a copy trade"""
        try:
            # Count trades copied from this wallet (use dedicated table)
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM copytrading_trades
                WHERE source_wallet = $1
            """, wallet)

            # Update the wallet's copied trades count in config if tracking
            await conn.execute("""
                INSERT INTO config_settings (config_type, key, value)
                VALUES ('wallet_stats', $1, $2)
                ON CONFLICT (config_type, key) DO UPDATE SET value = $2
            """, wallet, str(count))

        except Exception as e:
            logger.debug(f"Error updating wallet stats: {e}")

    def get_execution_wallets(self) -> Dict[str, Optional[str]]:
        """Return the bot's OWN execution wallet PUBLIC addresses per chain
        (issue 15 wallet identity). These are the wallets that broadcast the
        mirrored trades — distinct from the leader `targets` being copied.

        NEVER returns private keys/keypairs — public addresses only. Values
        are None until executor.initialize() resolves them from secrets.

        Sources:
          - EVM:    secrets key WALLET_ADDRESS / private key PRIVATE_KEY
          - Solana: secrets key SOLANA_MODULE_WALLET / keypair
                    SOLANA_MODULE_PRIVATE_KEY
        """
        evm = getattr(self.executor, 'evm_wallet', None) if self.executor else None
        sol = getattr(self.executor, 'solana_wallet', None) if self.executor else None
        return {
            'evm': {
                'address': evm,
                'address_secret_key': 'WALLET_ADDRESS',
                'private_key_secret_key': 'PRIVATE_KEY',
            },
            'solana': {
                'address': sol,
                'address_secret_key': 'SOLANA_MODULE_WALLET',
                'private_key_secret_key': 'SOLANA_MODULE_PRIVATE_KEY',
            },
        }

    async def _persist_execution_wallets(self) -> None:
        """Persist the resolved PUBLIC execution addresses to
        config_settings(config_type='copytrading_diagnostics') so the
        dashboard can show the operator WHICH wallet funds copy trades on
        each chain. Public addresses only — never the key. Fail-soft.

        Wave-11 FIX 3: error logging promoted from debug -> warning so a
        DB error (the reason the operator's funding panel showed
        "Copy Trading: wallets not resolved yet") is visible in
        logs/copy_trading/. Also adds a masked-address confirmation log
        line on success so the operator can verify both wallets
        surfaced. Idempotent on the SQL side (ON CONFLICT DO UPDATE)."""
        if not self.db_pool:
            logger.warning(
                "_persist_execution_wallets: db_pool=None, copy wallets "
                "will NOT be visible on the dashboard funding panel"
            )
            return
        wallets = self.get_execution_wallets()
        evm_addr = wallets['evm']['address']
        sol_addr = wallets['solana']['address']
        # PM Wave-11: also surface the stored-vs-derived mismatch so the
        # dashboard's funding panel renders the same WARNING badge it
        # already shows for DEX. Dashboard reads these two extra keys at
        # monitoring/enhanced_dashboard.py:~14673 (see comment "if the
        # copy_engine ever starts persisting it the same way DEX does").
        # Set on the executor by Wave-11 commit 6fb1f50 wallet-derivation
        # block — read defensively in case those attrs are absent.
        _exec = getattr(self, 'executor', None)
        stored_evm = getattr(_exec, 'stored_evm_wallet', None) if _exec else None
        evm_mismatch = bool(getattr(_exec, 'wallet_address_secret_mismatch', False)) if _exec else False
        try:
            async with self.db_pool.acquire() as conn:
                for key, addr in (
                    ('evm_execution_wallet', evm_addr),
                    ('solana_execution_wallet', sol_addr),
                    ('evm_wallet_address_stored', stored_evm or ''),
                    ('wallet_address_secret_mismatch', 'true' if evm_mismatch else 'false'),
                ):
                    await conn.execute(
                        "INSERT INTO config_settings (config_type, key, value) "
                        "VALUES ('copytrading_diagnostics', $1, $2) "
                        "ON CONFLICT (config_type, key) DO UPDATE SET value = $2",
                        key, addr or '',
                    )
            evm_mask = (evm_addr[:6] + "..." + evm_addr[-4:]) if evm_addr else "none"
            sol_mask = (sol_addr[:6] + "..." + sol_addr[-4:]) if sol_addr else "none"
            logger.info(
                f"🔑 Copy execution wallets surfaced: evm={evm_mask} solana={sol_mask}"
            )
        except Exception as e:
            # Explicit warning so the operator sees this in logs when the
            # dashboard funding panel reads "wallets not resolved yet".
            logger.warning(
                f"_persist_execution_wallets failed: {e} — dashboard funding "
                "panel will show 'wallets not resolved yet' until next call"
            )

    async def get_positions(self) -> List[Dict]:
        """Return open copy-trade positions from the DB.

        Each row is the operator-facing view, NOT a full BaseModule.Position
        object - that conversion lives in the eventual BaseModule wrapper.
        """
        if not self.db_pool:
            return []
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT trade_id, chain, source_wallet, token_address,
                           entry_price, entry_usd, amount, entry_timestamp
                    FROM copytrading_trades
                    WHERE status = 'open' AND side = 'buy'
                """)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"COPY.get_positions failed: {e}")
            return []

    async def _get_evm_token_balance(self, token_address: str, chain: str) -> int:
        """Return raw on-chain ERC-20 balance of self.executor.evm_wallet on chain."""
        if not self.executor or not self.executor.evm_wallet:
            return 0
        routing = EVM_DEX_ROUTING.get(chain)
        if not routing:
            return 0
        from config.pool_engine import PoolEngine
        pool = await PoolEngine.get_instance()
        rpc_url = await pool.get_endpoint(routing['rpc_key'])
        if not rpc_url:
            return 0
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        erc20_abi = [{
            "constant": True, "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function",
        }]
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(token_address), abi=erc20_abi
        )
        return contract.functions.balanceOf(
            Web3.to_checksum_address(self.executor.evm_wallet)
        ).call()

    async def _estimate_tokens_received(self, mint: str, usd_spent: float) -> Optional[float]:
        """Estimate the human-readable token count bought for `usd_spent`
        USD on Solana. Used to populate metadata.tokens_received on BUY
        so the dashboard can show correct live PnL.

        Reads the token's current USD price from Jupiter Price v3
        (free tier), divides usd_spent by that price. Returns None on
        any failure — caller treats None as "skip the field".
        """
        if not mint or usd_spent <= 0:
            return None
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5)
            url = f"https://api.jup.ag/price/v3?ids={mint}"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
        except Exception:
            return None
        # v3 shape: {<mint>: {"usdPrice": "...", ...}}; legacy v2:
        # {"data": {<mint>: {"price": ...}}}
        payload = data.get('data') if isinstance(data, dict) and 'data' in data else data
        if not isinstance(payload, dict):
            return None
        row = payload.get(mint)
        if not isinstance(row, dict):
            return None
        price_raw = row.get('usdPrice') or row.get('price') or row.get('usd') or 0
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return None
        if price <= 0:
            return None
        return usd_spent / price

    async def _process_close_flag_files(self) -> None:
        """Pick up logs/.close_copy_<trade_id> files dropped by the
        dashboard's manual-close endpoint and execute the swap. After a
        successful (or terminally-failed) attempt the flag file is
        deleted so we don't retry forever. Cross-subprocess IPC pattern
        mirrors logs/.killswitch.
        """
        from pathlib import Path
        log_dir = Path('logs')
        if not log_dir.is_dir():
            return
        for flag in log_dir.glob('.close_copy_*'):
            trade_id = flag.name[len('.close_copy_'):]
            if not trade_id:
                try:
                    flag.unlink()
                except Exception:
                    pass
                continue
            position = None
            from_trades_table = False
            if self.db_pool:
                try:
                    async with self.db_pool.acquire() as conn:
                        # First check copytrading_positions (legacy table).
                        row = await conn.fetchrow(
                            "SELECT trade_id, chain, token_address, amount "
                            "FROM copytrading_positions "
                            "WHERE trade_id = $1 AND status IN ('open','closing')",
                            trade_id,
                        )
                        if row:
                            position = dict(row)
                        else:
                            # OPERATOR-REPORTED BUG: positions are tracked in
                            # copytrading_trades (no separate positions row)
                            # for the current copy_engine path. Fall back to
                            # that table so the close button actually does
                            # something. Use entry_usd as the USD basis.
                            row = await conn.fetchrow(
                                "SELECT trade_id, chain, token_address, amount, entry_usd, "
                                "       (metadata::jsonb->>'tokens_received')::float8 AS tokens_received "
                                "FROM copytrading_trades "
                                "WHERE trade_id = $1 AND status = 'open'",
                                trade_id,
                            )
                            if row:
                                position = dict(row)
                                from_trades_table = True
                except Exception as e:
                    logger.error(f"close-flag DB lookup failed for {trade_id}: {e}")
            if not position:
                logger.warning(f"close-flag {trade_id} found but no open position; deleting flag")
                try:
                    flag.unlink()
                except Exception:
                    pass
                continue
            result = await self.close_position(position)
            ok = bool(result and result.get('success'))
            logger.info(
                f"manual close {trade_id}: success={ok} "
                f"tx={result.get('tx_hash') if result else None} "
                f"note={result.get('note') if result else None}"
            )
            # On success mark the position closed in DB so the UI reflects
            # the change immediately (engine's own SELL audit may also
            # do this on the next tick, but we don't want a race).
            # If we resolved from copytrading_trades, update that table too —
            # otherwise the operator's positions page keeps showing the row.
            if ok and self.db_pool:
                try:
                    async with self.db_pool.acquire() as conn:
                        if from_trades_table:
                            await conn.execute(
                                "UPDATE copytrading_trades "
                                "SET status='closed', exit_timestamp=NOW() "
                                "WHERE trade_id = $1",
                                trade_id,
                            )
                        await conn.execute(
                            "UPDATE copytrading_positions "
                            "SET status='closed', closed_at=NOW(), updated_at=NOW() "
                            "WHERE trade_id = $1",
                            trade_id,
                        )
                except Exception as e:
                    logger.error(f"close-flag DB update failed for {trade_id}: {e}")
            try:
                flag.unlink()
            except Exception:
                pass

    async def close_position(self, position) -> Dict:
        """Best-effort close. Routes via the executor based on position['chain'].

        Caller is responsible for any audit-trail / _log_copy_trade accounting -
        the dashboard emergency-exit handler wants to record its own outcome,
        so this method stays pure (no DB writes).
        """
        if not self.executor:
            return {'success': False, 'error': 'executor not initialised'}
        chain = position.get('chain')
        token = position.get('token_address')
        if not chain or not token:
            return {'success': False, 'error': 'position missing chain or token_address'}

        if chain == 'solana':
            # Reuse the MB-22 SELL path: actual on-chain balance via Jupiter.
            balance_raw, _decimals = await self._get_solana_token_balance(token)
            if balance_raw <= 0:
                return {'success': True, 'tx_hash': None, 'note': 'no on-chain balance to close'}
            return await self.executor.copy_solana_swap(
                input_mint=token,
                output_mint=WSOL_MINT,
                amount_lamports=balance_raw,
                slippage_bps=300,  # memecoin-tolerant exit, matches MB-22
            )

        # EVM: prefer actual on-chain balance; fall back to DB `amount` if the
        # RPC read fails so a panic-exit isn't blocked by a flaky endpoint.
        try:
            amount_wei = await self._get_evm_token_balance(token, chain)
        except Exception:
            amount_wei = int(position.get('amount') or 0)
        if amount_wei <= 0:
            return {'success': True, 'tx_hash': None, 'note': 'no on-chain balance to close'}
        return await self.executor.copy_evm_swap(
            token_address=token,
            amount_wei=amount_wei,
            is_buy=False,
            chain=chain,
        )

    async def stop(self) -> bool:
        """Stop the engine. Returns True on clean shutdown."""
        self._running = False
        self.status = ModuleStatus.STOPPING

        # Close executor
        try:
            if self.executor:
                await self.executor.close()
        except Exception as e:
            self.logger.warning(f"executor.close() failed: {e}")

        self.status = ModuleStatus.STOPPED
        self.stop_time = datetime.now()
        logger.info("🛑 Copy Trading Engine Stopped")
        return True
