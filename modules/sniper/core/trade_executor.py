"""
Trade Executor for Sniper Module
Handles real trade execution on both EVM chains and Solana.

EVM: Uses Uniswap V2/V3 Router
Solana: Uses Jupiter Aggregator API for best pricing

IMPORTANT: Only executes real trades when DRY_RUN=false
"""

import asyncio
import hashlib
import logging
import random
import aiohttp
import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from core.dry_run import should_skip_live
from core.units import to_raw_evm

# Import RPCProvider for centralized RPC management
try:
    from config.rpc_provider import RPCProvider
except ImportError:
    RPCProvider = None

logger = logging.getLogger("TradeExecutor")

# Jupiter API endpoints - use lite-api.jup.ag/swap/v1 (proven to work)
# Resolved lazily via secrets_manager (DB-encrypted) with .env fallback so
# that secrets.initialize(db_pool) can engage before first use.
_JUPITER_DEFAULT = 'https://lite-api.jup.ag/swap/v1'
_jupiter_base_cache: Optional[str] = None


def _resolve_jupiter_base() -> str:
    """Lazily resolve Jupiter base URL via secrets_manager
    (DB-encrypted) with .env fallback. Cached after first call."""
    global _jupiter_base_cache
    if _jupiter_base_cache is not None:
        return _jupiter_base_cache
    val = None
    try:
        from security.secrets_manager import secrets
        val = secrets.get('JUPITER_API_URL', default=None,
                          log_access=False)
    except Exception:
        val = None
    base = val or os.getenv('JUPITER_API_URL', _JUPITER_DEFAULT)
    # Normalize URL
    if 'lite-api.jup.ag' in base and not base.endswith('/swap/v1'):
        base = base.rstrip('/') + '/swap/v1'
    elif 'quote-api.jup.ag' in base and not base.endswith('/v6'):
        base = base.rstrip('/') + '/v6'
    _jupiter_base_cache = base
    return _jupiter_base_cache


def _quote_api() -> str:
    return f"{_resolve_jupiter_base()}/quote"


def _swap_api() -> str:
    return f"{_resolve_jupiter_base()}/swap"

# Common token addresses
WSOL_ADDRESS = "So11111111111111111111111111111111111111112"
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # Mainnet

# Uniswap V2 Router (Mainnet)
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"

# Uniswap V2 Router ABI (minimal)
ROUTER_ABI = [
    {
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
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForETH",
        "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
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


@dataclass
class TradeResult:
    """Result of a trade execution"""
    success: bool
    chain: str
    token_address: str
    amount_in: float
    amount_out: float
    tx_hash: Optional[str]
    gas_used: Optional[int]
    error: Optional[str]
    timestamp: datetime


class TradeExecutor:
    """
    Trade Executor for Sniper Module.
    Supports both EVM (Uniswap) and Solana (Jupiter) execution.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.w3 = None

        # Wallet credentials (loaded from env)
        self.evm_private_key = None
        self.evm_wallet = None
        self.solana_private_key = None
        self.solana_wallet = None

        # Settings
        self.dry_run = True

    async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
        """
        Get decrypted private key from secrets manager or environment.

        Priority:
        1. Secrets manager (Docker secrets, database)
        2. Environment variable with decryption

        Always checks if value is still encrypted and decrypts if needed.
        """
        try:
            value = None

            # Try secrets manager first
            try:
                from security.secrets_manager import secrets
                value = await secrets.get_async(key_name)
            except Exception:
                pass

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
        """Initialize the executor"""
        # HTTP session for API calls
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Load credentials from secrets manager (Docker secrets, database, or env)
        self.evm_private_key = await self._get_decrypted_key('PRIVATE_KEY') or await self._get_decrypted_key('EVM_PRIVATE_KEY')
        self.solana_private_key = await self._get_decrypted_key('SOLANA_MODULE_PRIVATE_KEY')

        # Get STORED wallet addresses from secrets manager (reference-only;
        # the derived addresses below are the authoritative funding identity).
        try:
            from security.secrets_manager import secrets
            stored_evm = secrets.get('WALLET_ADDRESS', log_access=False) or secrets.get('EVM_WALLET_ADDRESS', log_access=False) or os.getenv('WALLET_ADDRESS') or os.getenv('EVM_WALLET_ADDRESS')
            stored_sol = secrets.get('SOLANA_MODULE_WALLET', log_access=False) or os.getenv('SOLANA_MODULE_WALLET')
        except Exception:
            stored_evm = os.getenv('WALLET_ADDRESS') or os.getenv('EVM_WALLET_ADDRESS')
            stored_sol = os.getenv('SOLANA_MODULE_WALLET')
        self.stored_evm_wallet = stored_evm or None
        self.stored_solana_wallet = stored_sol or None

        # Wave-12 FIX 1: ALWAYS derive both public addresses from the
        # decrypted private keys (mirrors DEX FIX 1 / Copy FIX 3 Wave-11
        # pattern). The stored *_WALLET secrets were silently None for
        # operators who funded only the PK, leaving `solana_wallet` /
        # `evm_wallet` falsy and the sniper runtime-stats snapshot showing
        # `solana=none evm=none` on the dashboard funding panel. The
        # derived address is the authoritative funding identity; the
        # stored secret is compared and a CRITICAL masked-address warning
        # is logged on mismatch (funding the stale stored address in LIVE
        # mode would lose money).
        derived_evm = None
        if self.evm_private_key:
            try:
                from eth_account import Account
                _pk = self.evm_private_key if self.evm_private_key.startswith('0x') else f'0x{self.evm_private_key}'
                derived_evm = Account.from_key(_pk).address
            except Exception as e:
                logger.debug(f"sniper EVM wallet derivation failed: {e}")
        derived_sol = None
        if self.solana_private_key:
            try:
                from solders.keypair import Keypair
                import base58
                pk = self.solana_private_key
                key_bytes = None
                if pk.startswith('['):
                    try:
                        key_bytes = bytes(json.loads(pk))
                    except Exception:
                        pass
                if key_bytes is None:
                    try:
                        key_bytes = base58.b58decode(pk)
                    except Exception:
                        pass
                if key_bytes is None:
                    try:
                        key_bytes = bytes.fromhex(pk)
                    except Exception:
                        pass
                if key_bytes is not None:
                    if len(key_bytes) == 64:
                        kp = Keypair.from_bytes(key_bytes)
                    elif len(key_bytes) == 32:
                        kp = Keypair.from_seed(key_bytes)
                    else:
                        kp = None
                    if kp is not None:
                        derived_sol = str(kp.pubkey())
            except Exception as e:
                logger.debug(f"sniper Solana wallet derivation failed: {e}")

        # Mismatch detection (logged at CRITICAL with masked addresses).
        self.wallet_address_secret_mismatch = False
        self.solana_wallet_secret_mismatch = False
        if derived_evm:
            self.evm_wallet = derived_evm
            if stored_evm and stored_evm.lower() != derived_evm.lower():
                self.wallet_address_secret_mismatch = True
                stored_mask = (stored_evm[:6] + "..." + stored_evm[-4:]) if len(stored_evm) >= 10 else "***"
                derived_mask = derived_evm[:6] + "..." + derived_evm[-4:]
                logger.critical(
                    "SNIPER WALLET_ADDRESS mismatch: stored=%s vs derived=%s. "
                    "Using DERIVED (PRIVATE_KEY authoritative). Update stored "
                    "secret — funding the stored address would lose money.",
                    stored_mask, derived_mask,
                )
        else:
            self.evm_wallet = stored_evm or None
        if derived_sol:
            self.solana_wallet = derived_sol
            if stored_sol and stored_sol != derived_sol:
                self.solana_wallet_secret_mismatch = True
                stored_mask = (stored_sol[:6] + "..." + stored_sol[-4:]) if len(stored_sol) >= 10 else "***"
                derived_mask = derived_sol[:6] + "..." + derived_sol[-4:]
                logger.critical(
                    "SNIPER SOLANA_MODULE_WALLET mismatch: stored=%s vs derived=%s. "
                    "Using DERIVED (SOLANA_MODULE_PRIVATE_KEY authoritative). "
                    "Update stored secret — funding the stored address would lose money.",
                    stored_mask, derived_mask,
                )
        else:
            self.solana_wallet = stored_sol or None

        # DRY_RUN check
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Initialize Web3 if EVM credentials available
        if self.evm_private_key and not should_skip_live(self.dry_run, module='sniper', account=getattr(self, 'evm_wallet', None)):
            try:
                from web3 import Web3
                # PoolEngine first (sync ctor), .env preserved as ultimate fallback
                rpc_url = (
                    (RPCProvider.get_rpc_sync('ETHEREUM_RPC') if RPCProvider else None)
                    or os.getenv('ETHEREUM_RPC_URL')
                    or os.getenv('WEB3_PROVIDER_URL')
                )
                if rpc_url:
                    self.w3 = Web3(Web3.HTTPProvider(rpc_url))
                    if self.w3.is_connected():
                        logger.info("✅ Web3 connected for trade execution")
            except Exception as e:
                logger.error(f"Web3 initialization error: {e}")

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"💱 Trade Executor initialized ({mode})")

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def execute_buy(
        self,
        token_address: str,
        chain: str,
        amount_in: float,
        slippage: float = 10.0,
        priority_fee: int = 5000
    ) -> TradeResult:
        """
        Execute a buy order for a token.

        Args:
            token_address: Token to buy
            chain: 'solana', 'ethereum', 'bsc', etc.
            amount_in: Amount to spend (SOL/ETH)
            slippage: Max slippage percentage
            priority_fee: Priority fee (Gwei for EVM, Lamports for Solana)

        Returns:
            TradeResult with execution details
        """
        logger.info(f"🛒 Executing BUY: {token_address} on {chain}")
        logger.info(f"   Amount: {amount_in} | Slippage: {slippage}% | Priority: {priority_fee}")

        _account = getattr(self, 'solana_wallet', None) if chain == 'solana' else getattr(self, 'evm_wallet', None)
        if should_skip_live(self.dry_run, module='sniper', account=_account):
            return await self._simulate_buy(token_address, chain, amount_in)

        if chain == 'solana':
            return await self._execute_solana_buy(token_address, amount_in, slippage, priority_fee)
        else:
            return await self._execute_evm_buy(token_address, chain, amount_in, slippage, priority_fee)

    async def execute_sell(
        self,
        token_address: str,
        chain: str,
        amount_in: float,
        slippage: float = 10.0,
        priority_fee: int = 5000
    ) -> TradeResult:
        """
        Execute a sell order for a token.

        Args:
            token_address: Token to sell
            chain: 'solana', 'ethereum', 'bsc', etc.
            amount_in: Amount of tokens to sell
            slippage: Max slippage percentage
            priority_fee: Priority fee (Gwei for EVM, Lamports for Solana)

        Returns:
            TradeResult with execution details
        """
        logger.info(f"💰 Executing SELL: {token_address} on {chain}")
        logger.info(f"   Amount: {amount_in} | Slippage: {slippage}% | Priority: {priority_fee}")

        _account = getattr(self, 'solana_wallet', None) if chain == 'solana' else getattr(self, 'evm_wallet', None)
        if should_skip_live(self.dry_run, module='sniper', account=_account):
            return await self._simulate_sell(token_address, chain, amount_in)

        if chain == 'solana':
            return await self._execute_solana_sell(token_address, amount_in, slippage, priority_fee)
        else:
            return await self._execute_evm_sell(token_address, chain, amount_in, slippage, priority_fee)

    # ===== SOLANA EXECUTION (Jupiter) =====

    async def _execute_solana_buy(
        self,
        token_address: str,
        amount_in: float,
        slippage: float,
        priority_fee: int
    ) -> TradeResult:
        """Execute buy on Solana using Jupiter"""
        try:
            if not self.solana_wallet or not self.solana_private_key:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Solana wallet not configured", timestamp=datetime.now()
                )

            # 1. Get quote from Jupiter
            amount_lamports = int(amount_in * 1e9)  # Convert SOL to lamports
            slippage_bps = int(slippage * 100)  # Convert % to basis points

            quote = await self._get_jupiter_quote(
                input_mint=WSOL_ADDRESS,
                output_mint=token_address,
                amount=amount_lamports,
                slippage_bps=slippage_bps
            )

            if not quote:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to get Jupiter quote", timestamp=datetime.now()
                )

            expected_output = int(quote.get('outAmount', 0))
            logger.info(f"📊 Jupiter quote: {amount_lamports} lamports -> {expected_output} tokens")

            # 2. Get swap transaction from Jupiter
            swap_tx = await self._get_jupiter_swap_tx(
                quote=quote,
                user_public_key=self.solana_wallet,
                priority_fee=priority_fee
            )

            if not swap_tx:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to get Jupiter swap transaction", timestamp=datetime.now()
                )

            # 3. Sign and send transaction
            tx_hash = await self._sign_and_send_solana_tx(swap_tx)

            if tx_hash:
                return TradeResult(
                    success=True, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=expected_output / 1e6,  # Assume 6 decimals
                    tx_hash=tx_hash, gas_used=priority_fee,
                    error=None, timestamp=datetime.now()
                )
            else:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to send transaction", timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"Solana buy execution error: {e}")
            return TradeResult(
                success=False, chain='solana', token_address=token_address,
                amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                error=str(e), timestamp=datetime.now()
            )

    async def _execute_solana_sell(
        self,
        token_address: str,
        amount_in: float,
        slippage: float,
        priority_fee: int
    ) -> TradeResult:
        """Execute sell on Solana using Jupiter"""
        try:
            if not self.solana_wallet or not self.solana_private_key:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Solana wallet not configured", timestamp=datetime.now()
                )

            # 1. Get quote from Jupiter (selling token for SOL)
            amount_tokens = int(amount_in * 1e6)  # Assume 6 decimals
            slippage_bps = int(slippage * 100)

            quote = await self._get_jupiter_quote(
                input_mint=token_address,
                output_mint=WSOL_ADDRESS,
                amount=amount_tokens,
                slippage_bps=slippage_bps
            )

            if not quote:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to get Jupiter quote", timestamp=datetime.now()
                )

            expected_output = int(quote.get('outAmount', 0))
            logger.info(f"📊 Jupiter quote: {amount_tokens} tokens -> {expected_output} lamports")

            # 2. Get swap transaction from Jupiter
            swap_tx = await self._get_jupiter_swap_tx(
                quote=quote,
                user_public_key=self.solana_wallet,
                priority_fee=priority_fee
            )

            if not swap_tx:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to get Jupiter swap transaction", timestamp=datetime.now()
                )

            # 3. Sign and send transaction
            tx_hash = await self._sign_and_send_solana_tx(swap_tx)

            if tx_hash:
                return TradeResult(
                    success=True, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=expected_output / 1e9,  # Convert lamports to SOL
                    tx_hash=tx_hash, gas_used=priority_fee,
                    error=None, timestamp=datetime.now()
                )
            else:
                return TradeResult(
                    success=False, chain='solana', token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="Failed to send transaction", timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"Solana sell execution error: {e}")
            return TradeResult(
                success=False, chain='solana', token_address=token_address,
                amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                error=str(e), timestamp=datetime.now()
            )

    async def _get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int
    ) -> Optional[Dict]:
        """Get quote from Jupiter API"""
        try:
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }

            async with self.session.get(_quote_api(), params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter quote error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Jupiter quote API error: {e}")
            return None

    async def _get_jupiter_swap_tx(
        self,
        quote: Dict,
        user_public_key: str,
        priority_fee: int
    ) -> Optional[str]:
        """Get swap transaction from Jupiter API"""
        try:
            payload = {
                'quoteResponse': quote,
                'userPublicKey': user_public_key,
                'wrapAndUnwrapSol': True,
                'prioritizationFeeLamports': priority_fee,
                'dynamicComputeUnitLimit': True
            }

            async with self.session.post(_swap_api(), json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('swapTransaction')
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter swap error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Jupiter swap API error: {e}")
            return None

    async def _sign_and_send_solana_tx(self, swap_tx_base64: str) -> Optional[str]:
        """Sign and send Solana transaction (supports JSON array, base58, hex key formats)"""
        try:
            # Import Solana libraries
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solders.signature import Signature
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

            # Preserve any pre-existing co-signer slots (Jupiter setup/ATA/advanced routes).
            num_required = message.header.num_required_signatures
            account_keys = message.account_keys
            existing_sigs = list(tx.signatures)
            our_index = None
            for i in range(num_required):
                if str(account_keys[i]) == str(our_pubkey):
                    our_index = i
                    break
            if our_index is None:
                logger.error(f"❌ Our pubkey {our_pubkey} not in required signers")
                return None

            our_sig = keypair.sign_message(bytes(message))
            zero_sig = Signature.default()
            final_sigs = []
            for i in range(num_required):
                if i == our_index:
                    final_sigs.append(our_sig)
                elif i < len(existing_sigs) and existing_sigs[i] != zero_sig:
                    final_sigs.append(existing_sigs[i])
                else:
                    logger.error(f"❌ Missing co-signer for slot {i} ({account_keys[i]})")
                    return None
            signed_tx = VersionedTransaction.populate(message, final_sigs)

            # Send transaction - PoolEngine first (async ctx), .env preserved as fallback
            rpc_url = (
                (await RPCProvider.get_rpc('SOLANA_RPC') if RPCProvider else None)
                or os.getenv('SOLANA_RPC_URL')
            )
            async with AsyncClient(rpc_url) as client:
                result = await client.send_transaction(signed_tx)
                tx_hash = str(result.value)
                logger.info(f"✅ Solana TX sent: {tx_hash}")
                return tx_hash

        except ImportError as e:
            logger.error(f"Solana libraries not installed: {e}")
            logger.error("Install with: pip install solana solders base58")
            return None
        except Exception as e:
            logger.error(f"Solana transaction error: {e}")
            return None

    # ===== EVM EXECUTION (Uniswap) =====

    async def _execute_evm_buy(
        self,
        token_address: str,
        chain: str,
        amount_in: float,
        slippage: float,
        priority_fee: int
    ) -> TradeResult:
        """Execute buy on EVM using Uniswap V2"""
        try:
            if not self.w3 or not self.evm_private_key or not self.evm_wallet:
                return TradeResult(
                    success=False, chain=chain, token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="EVM wallet not configured", timestamp=datetime.now()
                )

            from web3 import Web3

            # Router contract
            router = self.w3.eth.contract(
                address=Web3.to_checksum_address(UNISWAP_V2_ROUTER),
                abi=ROUTER_ABI
            )

            # Convert amount to Wei
            amount_wei = Web3.to_wei(amount_in, 'ether')

            # Path: WETH -> Token
            path = [
                Web3.to_checksum_address(WETH_ADDRESS),
                Web3.to_checksum_address(token_address)
            ]

            # MB-11: quote expected output and apply slippage haircut so we
            # never broadcast amount_out_min=0 (guaranteed-sandwich).
            try:
                amounts_out = router.functions.getAmountsOut(amount_wei, path).call()
                expected_out = int(amounts_out[-1])
            except Exception as quote_err:
                logger.error(f"getAmountsOut failed; aborting buy to avoid 0-min: {quote_err}")
                return TradeResult(
                    success=False, chain=chain, token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error=f"Quote failed: {quote_err}", timestamp=datetime.now()
                )
            # Primary path: caller passes slippage (already DB-plumbed via
            # SniperEngine.slippage). Fallback only fires for direct
            # callers that omit the arg; operator can override the floor
            # via SNIPER_SLIPPAGE_FALLBACK_PCT env without redeploying.
            if slippage is not None:
                slippage_frac = max(float(slippage), 0.0) / 100.0
            else:
                slippage_frac = max(float(os.getenv('SNIPER_SLIPPAGE_FALLBACK_PCT', '3.0')), 0.0) / 100.0
            amount_out_min = int(expected_out * (1 - slippage_frac))

            # Deadline: 2 minutes from now
            deadline = int(datetime.now().timestamp()) + 120

            # Build transaction
            tx = router.functions.swapExactETHForTokens(
                amount_out_min,
                path,
                Web3.to_checksum_address(self.evm_wallet),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.evm_wallet),
                'value': amount_wei,
                'gas': 300000,
                'maxPriorityFeePerGas': Web3.to_wei(priority_fee, 'gwei'),
                'maxFeePerGas': Web3.to_wei(priority_fee + 50, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.evm_wallet)
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.evm_private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(f"✅ EVM TX sent: {tx_hash_hex}")

            # Wait for confirmation (optional, can be async)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            return TradeResult(
                success=receipt['status'] == 1,
                chain=chain,
                token_address=token_address,
                amount_in=amount_in,
                amount_out=0,  # Would parse from logs in production
                tx_hash=tx_hash_hex,
                gas_used=receipt['gasUsed'],
                error=None if receipt['status'] == 1 else "Transaction reverted",
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"EVM buy execution error: {e}")
            return TradeResult(
                success=False, chain=chain, token_address=token_address,
                amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                error=str(e), timestamp=datetime.now()
            )

    async def _execute_evm_sell(
        self,
        token_address: str,
        chain: str,
        amount_in: float,
        slippage: float,
        priority_fee: int
    ) -> TradeResult:
        """Execute sell on EVM using Uniswap V2"""
        try:
            if not self.w3 or not self.evm_private_key or not self.evm_wallet:
                return TradeResult(
                    success=False, chain=chain, token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error="EVM wallet not configured", timestamp=datetime.now()
                )

            from web3 import Web3

            # First, approve router to spend tokens (if not already approved)
            # In production, check allowance first

            # Router contract
            router = self.w3.eth.contract(
                address=Web3.to_checksum_address(UNISWAP_V2_ROUTER),
                abi=ROUTER_ABI
            )

            # Path: Token -> WETH
            path = [
                Web3.to_checksum_address(token_address),
                Web3.to_checksum_address(WETH_ADDRESS)
            ]

            # MB-11: convert input using on-chain decimals, not hardcoded 1e18.
            amount_tokens = await to_raw_evm(chain or 'ethereum', token_address, Decimal(str(amount_in)))

            # MB-11: quote expected ETH out and apply slippage haircut.
            try:
                amounts_out = router.functions.getAmountsOut(amount_tokens, path).call()
                expected_out = int(amounts_out[-1])
            except Exception as quote_err:
                logger.error(f"getAmountsOut failed; aborting sell to avoid 0-min: {quote_err}")
                return TradeResult(
                    success=False, chain=chain, token_address=token_address,
                    amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                    error=f"Quote failed: {quote_err}", timestamp=datetime.now()
                )
            # Primary path: caller passes slippage (already DB-plumbed via
            # SniperEngine.slippage). Fallback only fires for direct
            # callers that omit the arg; operator can override the floor
            # via SNIPER_SLIPPAGE_FALLBACK_PCT env without redeploying.
            if slippage is not None:
                slippage_frac = max(float(slippage), 0.0) / 100.0
            else:
                slippage_frac = max(float(os.getenv('SNIPER_SLIPPAGE_FALLBACK_PCT', '3.0')), 0.0) / 100.0
            amount_out_min = int(expected_out * (1 - slippage_frac))

            # Deadline
            deadline = int(datetime.now().timestamp()) + 120

            # Build transaction
            tx = router.functions.swapExactTokensForETH(
                amount_tokens,
                amount_out_min,
                path,
                Web3.to_checksum_address(self.evm_wallet),
                deadline
            ).build_transaction({
                'from': Web3.to_checksum_address(self.evm_wallet),
                'gas': 300000,
                'maxPriorityFeePerGas': Web3.to_wei(priority_fee, 'gwei'),
                'maxFeePerGas': Web3.to_wei(priority_fee + 50, 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.evm_wallet)
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.evm_private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_hash_hex = tx_hash.hex()

            logger.info(f"✅ EVM TX sent: {tx_hash_hex}")

            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            return TradeResult(
                success=receipt['status'] == 1,
                chain=chain,
                token_address=token_address,
                amount_in=amount_in,
                amount_out=0,
                tx_hash=tx_hash_hex,
                gas_used=receipt['gasUsed'],
                error=None if receipt['status'] == 1 else "Transaction reverted",
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"EVM sell execution error: {e}")
            return TradeResult(
                success=False, chain=chain, token_address=token_address,
                amount_in=amount_in, amount_out=0, tx_hash=None, gas_used=None,
                error=str(e), timestamp=datetime.now()
            )

    # ===== DRY RUN SIMULATION =====

    async def _simulate_buy(
        self,
        token_address: str,
        chain: str,
        amount_in: float
    ) -> TradeResult:
        """Simulate a buy order (DRY RUN)"""
        logger.info(f"🧪 [DRY RUN] Simulating BUY: {amount_in} {chain.upper()} -> {token_address}")

        # Simulate network delay
        await asyncio.sleep(0.5)

        # Generate fake transaction hash
        fake_hash = hashlib.sha256(f"{token_address}{datetime.now().timestamp()}".encode()).hexdigest()

        # Simulate some output amount
        simulated_output = amount_in * 1000000  # Fake multiplier

        logger.info(f"🧪 [DRY RUN] Simulated BUY complete: {simulated_output} tokens")

        return TradeResult(
            success=True,
            chain=chain,
            token_address=token_address,
            amount_in=amount_in,
            amount_out=simulated_output,
            tx_hash=f"DRY_RUN_{fake_hash[:16]}",
            gas_used=0,
            error=None,
            timestamp=datetime.now()
        )

    async def _simulate_sell(
        self,
        token_address: str,
        chain: str,
        amount_in: float
    ) -> TradeResult:
        """Simulate a sell order (DRY RUN).

        Wave-12 FIX 3 — the operator surfaced `+696226.17%` /
        `+471681.10%` / `+762621.15%` TAKE PROFIT log lines on fresh
        Pump.fun mints. Root cause is upstream of this function: the
        engine's monitor (`SniperEngine._monitor_active_snipes`) divides
        a non-zero but stale `current_price` by an `entry_price` derived
        from a fabricated 1e6 buy quote and produces an unbounded
        `pnl_pct`, which is then logged BEFORE control reaches us. We
        don't own `sniper_engine.py` this wave (concurrent agent), so we
        cannot touch that log line or the monitor's comparison itself.
        What we DO own and have hardened here:

        1. Hard cap the simulated `move` factor at +200% (`move <= 3.0`)
           and -99% (`move >= 0.01`) so the `amount_out` we return —
           and therefore the `profit_loss_pct` that `_log_exit_to_db`
           later computes for the DB row — can never exceed ±200%
           regardless of how broken the engine's monitor-side
           comparison was. This is the DB-side sanity guard.
        2. The DB row PnL is `(move - 1) * 100`% by construction because
           the buy leg minted `amount_in_native * 1e6` tokens at
           `entry_price = sol_price / 1e6` and the sell returns
           `amount_in_native * move` SOL; so clamping `move` ∈
           [0.01, 3.0] guarantees DB `profit_loss_pct` ∈ [-99%, +200%].
        3. Wave-7 distribution (55% loss / 30% chop / 15% winner)
           preserved — the winner bucket's upper tail is the only
           thing trimmed (was +250% -> now +200%).

        This is the "missing code path" for the Wave-7 cap: the engine-
        side monitor reads its current_price from Pyth/Jupiter/Birdeye
        and computes `pnl_pct` directly from that, bypassing
        `_model_dry_run_exit_pct` whenever the upstream price source
        returns *any* non-zero value (even a wildly stale one). The
        clean fix lives in `sniper_engine.py`'s monitor (treat
        `|pnl_pct| > 200` as a phantom-price signal and reroute through
        the synthetic-close path); that change is the concurrent agent's
        scope. Until they ship it, this clamp ensures the DB row stays
        honest even if the LOG line is briefly noisy. See
        `modules/sniper/CLAUDE.md` for the full Wave-12 note.
        """
        logger.info(f"🧪 [DRY RUN] Simulating SELL: {amount_in} tokens -> {chain.upper()}")

        await asyncio.sleep(0.5)

        fake_hash = hashlib.sha256(f"{token_address}{datetime.now().timestamp()}".encode()).hexdigest()

        # The buy leg minted tokens at amount_in_native * 1e6 (see _simulate_buy),
        # so a flat /1e6 round-trips to break-even. A constant *1.1 made EVERY
        # simulated win identical (the "$0.84 on every trade" the operator saw).
        # Model a realistic price move seeded by the token mint so the round-trip
        # P&L varies per trade and re-runs are reproducible: ~55% lose, ~30%
        # chop, ~15% pump. This is a MODEL for DRY_RUN data collection, not a
        # measured exit price.
        seed = int(hashlib.sha256(("exitmult:" + token_address).encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        roll = rng.random()
        if roll < 0.55:
            move = 1.0 - rng.uniform(0.10, 0.85)   # loss: -10%..-85%
        elif roll < 0.85:
            move = 1.0 + rng.uniform(-0.08, 0.12)  # chop
        else:
            # Winner: +20%..+200%. Was +20%..+250% pre-Wave-12; tightened
            # so the DB-side `profit_loss_pct` cannot exceed +200% even
            # if the engine's monitor produced a phantom-price TAKE PROFIT.
            move = 1.0 + rng.uniform(0.20, 2.00)
        # Belt-and-suspenders ±200%/-99% clamp on the simulated move.
        # `_log_exit_to_db` computes `profit_loss_pct = (exit_usd/entry_usd - 1)*100`
        # and `exit_usd ≈ entry_usd * move` for DRY_RUN, so this directly
        # bounds the DB row's profit_loss_pct to [-99%, +200%]. Hard cap
        # exists as a sanity guard for the operator-reported +696226%
        # phantom-price case.
        if move > 3.0:
            logger.warning(
                f"🧪 [DRY RUN] sim_sell move {move:.2f} > 3.0 cap "
                f"({token_address[:8]}); clamping to +200%."
            )
            move = 3.0
        if move < 0.01:
            logger.warning(
                f"🧪 [DRY RUN] sim_sell move {move:.4f} < 0.01 floor "
                f"({token_address[:8]}); clamping to -99%."
            )
            move = 0.01
        simulated_output = amount_in / 1000000 * move

        logger.info(f"🧪 [DRY RUN] Simulated SELL complete: {simulated_output} {chain.upper()}")

        return TradeResult(
            success=True,
            chain=chain,
            token_address=token_address,
            amount_in=amount_in,
            amount_out=simulated_output,
            tx_hash=f"DRY_RUN_{fake_hash[:16]}",
            gas_used=0,
            error=None,
            timestamp=datetime.now()
        )
