"""
Trade Executor for Sniper Module
Handles real trade execution on both EVM chains and Solana.

EVM: Uses Uniswap V2/V3 Router
Solana: Uses Jupiter Aggregator API for best pricing

IMPORTANT: Only executes real trades when DRY_RUN=false
"""

import asyncio
import logging
import aiohttp
import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

# Import RPCProvider for centralized RPC management
try:
    from config.rpc_provider import RPCProvider
except ImportError:
    RPCProvider = None

logger = logging.getLogger("TradeExecutor")

# Jupiter API endpoints - use lite-api.jup.ag/swap/v1 (proven to work)
# Can be overridden via JUPITER_API_URL environment variable
_jupiter_base = os.getenv('JUPITER_API_URL', 'https://lite-api.jup.ag/swap/v1')
# Normalize URL
if 'lite-api.jup.ag' in _jupiter_base and not _jupiter_base.endswith('/swap/v1'):
    _jupiter_base = _jupiter_base.rstrip('/') + '/swap/v1'
elif 'quote-api.jup.ag' in _jupiter_base and not _jupiter_base.endswith('/v6'):
    _jupiter_base = _jupiter_base.rstrip('/') + '/v6'
JUPITER_QUOTE_API = f"{_jupiter_base}/quote"
JUPITER_SWAP_API = f"{_jupiter_base}/swap"

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

        # Get wallet addresses from secrets manager
        try:
            from security.secrets_manager import secrets
            self.evm_wallet = secrets.get('WALLET_ADDRESS', log_access=False) or secrets.get('EVM_WALLET_ADDRESS', log_access=False) or os.getenv('WALLET_ADDRESS') or os.getenv('EVM_WALLET_ADDRESS')
            self.solana_wallet = secrets.get('SOLANA_MODULE_WALLET', log_access=False) or os.getenv('SOLANA_MODULE_WALLET')
        except Exception:
            self.evm_wallet = os.getenv('WALLET_ADDRESS') or os.getenv('EVM_WALLET_ADDRESS')
            self.solana_wallet = os.getenv('SOLANA_MODULE_WALLET')

        # DRY_RUN check
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Initialize Web3 if EVM credentials available
        if self.evm_private_key and not self.dry_run:
            try:
                from web3 import Web3
                # Get RPC from Pool Engine with fallback
                rpc_url = None
                try:
                    from config.rpc_provider import RPCProvider
                    rpc_url = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
                except Exception:
                    pass
                if not rpc_url:
                    rpc_url = os.getenv('WEB3_PROVIDER_URL') or os.getenv('ETHEREUM_RPC_URL')
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

        if self.dry_run:
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

        if self.dry_run:
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

            async with self.session.get(JUPITER_QUOTE_API, params=params) as response:
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

            async with self.session.post(JUPITER_SWAP_API, json=payload) as response:
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
        """Sign and send Solana transaction"""
        try:
            # Import Solana libraries
            from solders.keypair import Keypair
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            import base64
            import base58

            # Decode private key
            private_key_bytes = base58.b58decode(self.solana_private_key)
            keypair = Keypair.from_bytes(private_key_bytes)

            # Decode transaction
            tx_bytes = base64.b64decode(swap_tx_base64)
            tx = VersionedTransaction.from_bytes(tx_bytes)

            # Sign transaction using correct VersionedTransaction.populate() pattern
            # The .sign([keypair]) method doesn't work with solders VersionedTransaction
            message = tx.message
            signature = keypair.sign_message(bytes(message))
            signed_tx = VersionedTransaction.populate(message, [signature])

            # Send transaction - use Pool Engine for RPC
            if RPCProvider:
                rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC')
            else:
                rpc_url = os.getenv('SOLANA_RPC_URL')
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

            # Calculate minimum output with slippage
            # Note: In production, you'd get expected output from router.getAmountsOut
            amount_out_min = 0  # Accept any amount (risky, but for speed)

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

            # Convert amount (assuming 18 decimals)
            amount_tokens = int(amount_in * 1e18)

            # Minimum output with slippage
            amount_out_min = 0

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
        import hashlib
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
        """Simulate a sell order (DRY RUN)"""
        logger.info(f"🧪 [DRY RUN] Simulating SELL: {amount_in} tokens -> {chain.upper()}")

        await asyncio.sleep(0.5)

        import hashlib
        fake_hash = hashlib.sha256(f"{token_address}{datetime.now().timestamp()}".encode()).hexdigest()

        simulated_output = amount_in / 1000000 * 1.1  # Slight profit

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
