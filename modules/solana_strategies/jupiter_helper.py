"""
Jupiter Aggregator Helper
Handles swap routing, quote fetching, and transaction signing for Jupiter v6 API
"""
import os
import asyncio
import logging
from typing import Dict, Optional, List
import aiohttp
import base58
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.pubkey import Pubkey

# Import RPCProvider for centralized RPC management
try:
    from config.rpc_provider import RPCProvider
except ImportError:
    RPCProvider = None

logger = logging.getLogger(__name__)


class JupiterHelper:
    """
    Jupiter Aggregator integration helper

    Provides:
    - Quote fetching
    - Route optimization
    - Transaction creation
    - Transaction signing
    - Swap execution
    """

    def __init__(self, solana_rpc_url: str = None, private_key: str = None):
        """
        Initialize Jupiter helper

        Args:
            solana_rpc_url: Solana RPC URL
            private_key: Base58-encoded private key for transaction signing
        """
        # Jupiter API URL - supports different plans:
        # - Lite (Free): https://lite-api.jup.ag/swap/v1 (1 RPS) - DEFAULT
        # - Public V6: https://quote-api.jup.ag/v6 (standard rate limits)
        # - Ultra (Premium): https://api.jup.ag/ultra (dynamic scaling)
        # Set JUPITER_API_URL in .env to your subscribed plan
        # NOTE: lite-api.jup.ag/swap/v1 is proven to work in arbitrage module
        raw_url = os.getenv('JUPITER_API_URL', 'https://lite-api.jup.ag/swap/v1')

        # Normalize URL - ensure it has the proper API path suffix
        # If user sets just "https://lite-api.jup.ag", append "/swap/v1"
        if 'lite-api.jup.ag' in raw_url and not raw_url.endswith('/swap/v1'):
            if raw_url.endswith('/'):
                self.api_url = raw_url + 'swap/v1'
            else:
                self.api_url = raw_url + '/swap/v1'
        elif 'quote-api.jup.ag' in raw_url and not raw_url.endswith('/v6'):
            if raw_url.endswith('/'):
                self.api_url = raw_url + 'v6'
            else:
                self.api_url = raw_url + '/v6'
        else:
            self.api_url = raw_url

        # Use Pool Engine via RPCProvider for centralized RPC management
        if solana_rpc_url:
            self.solana_rpc = solana_rpc_url
        elif RPCProvider:
            self.solana_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
        else:
            self.solana_rpc = os.getenv('SOLANA_RPC_URL')

        logger.info(f"🔗 Jupiter API endpoint: {self.api_url}")

        # Load keypair for signing - use secrets manager (database/Docker secrets)
        if private_key:
            try:
                self.keypair = Keypair.from_base58_string(private_key)
                logger.info(f"✅ Jupiter keypair loaded (pubkey: {str(self.keypair.pubkey())[:12]}...)")
            except Exception as e:
                logger.error(f"❌ Failed to load keypair from provided private_key: {e}")
                self.keypair = None
        else:
            # Get private key from secrets manager
            try:
                from security.secrets_manager import secrets
                pk_value = secrets.get('SOLANA_PRIVATE_KEY', log_access=False)
                if pk_value:
                    self.keypair = self._load_keypair_from_value(pk_value)
                else:
                    self.keypair = None
                    logger.warning("No Solana private key provided, signing will fail")
            except Exception:
                # Fallback to environment
                if os.getenv('SOLANA_PRIVATE_KEY'):
                    self.keypair = self._load_keypair_from_env()
                else:
                    self.keypair = None
                    logger.warning("No Solana private key provided, signing will fail")

        self.session: Optional[aiohttp.ClientSession] = None

    def _load_keypair_from_value(self, pk_str: str) -> Optional[Keypair]:
        """
        Load keypair from a string value (from secrets manager or env)

        Args:
            pk_str: Private key string (base58 encoded or encrypted)

        Returns:
            Optional[Keypair]: Loaded keypair or None
        """
        try:
            if not pk_str:
                return None

            # Check if value is encrypted (Fernet tokens start with gAAAAAB)
            if pk_str.startswith('gAAAAAB'):
                logger.debug("Private key appears to be Fernet encrypted, decrypting...")
                pk_str = self._decrypt_value(pk_str)
                if not pk_str:
                    logger.error("Failed to decrypt Solana private key")
                    return None
                logger.info("✅ Successfully decrypted Solana private key")

            # Try to parse as base58
            try:
                return Keypair.from_base58_string(pk_str)
            except (ValueError, Exception) as e:
                logger.warning(f"Private key parse failed: {e}")
                return None

        except Exception as e:
            logger.error(f"Error loading keypair: {e}")
            return None

    def _decrypt_value(self, encrypted_value: str) -> Optional[str]:
        """
        Decrypt a Fernet-encrypted value

        Args:
            encrypted_value: Fernet encrypted string (starts with gAAAAAB)

        Returns:
            Optional[str]: Decrypted value or None
        """
        try:
            from pathlib import Path

            # Get encryption key from file or environment
            encryption_key = None
            key_file = Path('.encryption_key')
            if key_file.exists():
                encryption_key = key_file.read_text().strip()
            if not encryption_key:
                encryption_key = os.getenv('ENCRYPTION_KEY')

            if not encryption_key:
                logger.error("Cannot decrypt: no encryption key found (.encryption_key or ENCRYPTION_KEY env)")
                return None

            from cryptography.fernet import Fernet
            f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
            return f.decrypt(encrypted_value.encode()).decode()

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None

    def _load_keypair_from_env(self) -> Optional[Keypair]:
        """
        Load keypair from environment variable (fallback)

        Returns:
            Optional[Keypair]: Loaded keypair or None
        """
        pk_str = os.getenv('SOLANA_PRIVATE_KEY')
        return self._load_keypair_from_value(pk_str)

    def _parse_quote_error(self, status_code: int, error_text: str, input_mint: str, output_mint: str) -> str:
        """
        Parse Jupiter quote error for better diagnostics

        Args:
            status_code: HTTP status code
            error_text: Error response text
            input_mint: Input token mint
            output_mint: Output token mint

        Returns:
            Human-readable error reason
        """
        import json

        # Known pump.fun token patterns (bonding curve program)
        PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

        error_lower = error_text.lower()

        # Parse JSON error if available
        try:
            error_json = json.loads(error_text)
            error_msg = error_json.get('error', error_json.get('message', ''))
            if error_msg:
                error_lower = error_msg.lower()
        except (json.JSONDecodeError, TypeError):
            pass

        # Check for specific error patterns
        if status_code == 400:
            if 'no route' in error_lower or 'no routes found' in error_lower:
                return f"No swap route available - token may be pump.fun-only or have no liquidity on Jupiter"
            if 'invalid' in error_lower and 'mint' in error_lower:
                return f"Invalid token mint address"
            if 'amount' in error_lower:
                return f"Invalid swap amount"
            return f"Bad request: {error_text[:100]}"
        elif status_code == 404:
            return f"Token not found on Jupiter - may be pump.fun-only token"
        elif status_code == 429:
            return f"Rate limited - too many requests to Jupiter API"
        elif status_code == 500:
            return f"Jupiter API server error - try again later"
        elif status_code == 503:
            return f"Jupiter API unavailable - temporary outage"
        else:
            return f"HTTP {status_code}: {error_text[:100]}"

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        only_direct_routes: bool = False
    ) -> Optional[Dict]:
        """
        Get swap quote from Jupiter

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
            only_direct_routes: Only use direct routes (no intermediate swaps)

        Returns:
            Optional[Dict]: Quote data or None if failed
        """
        try:
            if not self.session:
                await self.initialize()

            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'onlyDirectRoutes': str(only_direct_routes).lower(),
            }

            url = f"{self.api_url}/quote"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    quote = await response.json()
                    logger.info(
                        f"✅ Jupiter quote: {input_mint[:10]}... → {output_mint[:10]}... "
                        f"Amount: {amount}, Output: {quote.get('outAmount', 0)}"
                    )
                    return quote
                else:
                    error_text = await response.text()
                    # Parse Jupiter error response for better diagnostics
                    error_reason = self._parse_quote_error(response.status, error_text, input_mint, output_mint)
                    logger.warning(f"⚠️ Jupiter quote failed: {error_reason}")
                    return None

        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}", exc_info=True)
            return None

    async def get_swap_transaction(
        self,
        quote: Dict,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        as_legacy_transaction: bool = False
    ) -> Optional[Dict]:
        """
        Get swap transaction from quote

        Args:
            quote: Quote from get_quote()
            user_public_key: User's wallet public key
            wrap_unwrap_sol: Auto wrap/unwrap SOL
            as_legacy_transaction: Use legacy transaction format

        Returns:
            Optional[Dict]: Transaction data or None
        """
        try:
            if not self.session:
                await self.initialize()

            url = f"{self.api_url}/swap"

            payload = {
                'quoteResponse': quote,
                'userPublicKey': user_public_key,
                'wrapAndUnwrapSol': wrap_unwrap_sol,
                'asLegacyTransaction': as_legacy_transaction,
            }

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    swap_data = await response.json()
                    logger.debug("✅ Got swap transaction from Jupiter")
                    return swap_data
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter swap transaction error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Error getting swap transaction: {e}", exc_info=True)
            return None

    def sign_transaction(self, transaction_data: str) -> Optional[str]:
        """
        Sign a transaction with the loaded keypair

        Args:
            transaction_data: Base64-encoded transaction

        Returns:
            Optional[str]: Signed transaction as base64 or None
        """
        try:
            if not self.keypair:
                logger.error("No keypair available for signing")
                logger.error("   Ensure SOLANA_MODULE_PRIVATE_KEY is set in database secrets")
                return None

            # Decode the transaction
            import base64
            logger.debug(f"Decoding transaction ({len(transaction_data)} chars)...")
            tx_bytes = base64.b64decode(transaction_data)
            logger.debug(f"Transaction decoded ({len(tx_bytes)} bytes)")

            # Parse as versioned transaction
            logger.debug("Parsing versioned transaction...")
            tx = VersionedTransaction.from_bytes(tx_bytes)
            logger.debug(f"Transaction parsed, message signatures required: {len(tx.message.account_keys) if hasattr(tx.message, 'account_keys') else 'N/A'}")

            # Sign the transaction message and create signed transaction
            # Note: solders VersionedTransaction doesn't have a .sign() method
            # We need to sign the message and use VersionedTransaction.populate()
            logger.debug(f"Signing with keypair (pubkey: {str(self.keypair.pubkey())[:12]}...)")
            message = tx.message
            signature = self.keypair.sign_message(bytes(message))
            signed_tx = VersionedTransaction.populate(message, [signature])

            # Encode back to base64
            signed_tx_bytes = bytes(signed_tx)
            signed_tx_b64 = base64.b64encode(signed_tx_bytes).decode('utf-8')

            logger.info("✅ Transaction signed successfully")
            return signed_tx_b64

        except Exception as e:
            logger.error(f"Error signing transaction: {e}", exc_info=True)
            return None

    async def send_transaction(self, signed_transaction: str) -> Optional[str]:
        """
        Send signed transaction to Solana network

        Args:
            signed_transaction: Base64-encoded signed transaction

        Returns:
            Optional[str]: Transaction signature or None
        """
        try:
            if not self.solana_rpc:
                logger.error("No Solana RPC URL configured")
                return None

            if not self.session:
                await self.initialize()

            # Send via RPC
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    signed_transaction,
                    {
                        "encoding": "base64",
                        "skipPreflight": False,
                        "preflightCommitment": "confirmed",
                        "maxRetries": 3
                    }
                ]
            }

            async with self.session.post(self.solana_rpc, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'result' in result:
                        signature = result['result']
                        logger.info(f"✅ Transaction sent: {signature}")
                        return signature
                    elif 'error' in result:
                        logger.error(f"RPC error: {result['error']}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Send transaction error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Error sending transaction: {e}", exc_info=True)
            return None

    async def confirm_transaction(
        self,
        signature: str,
        timeout: int = 60,
        commitment: str = "confirmed"
    ) -> bool:
        """
        Confirm transaction on Solana

        Args:
            signature: Transaction signature
            timeout: Timeout in seconds
            commitment: Commitment level

        Returns:
            bool: True if confirmed, False otherwise
        """
        try:
            import time
            if not self.solana_rpc or not self.session:
                return False

            start_time = time.monotonic()

            while (time.monotonic() - start_time) < timeout:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[signature], {"searchTransactionHistory": True}]
                }

                async with self.session.post(self.solana_rpc, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        value = result.get('result', {}).get('value', [])

                        if value and value[0]:
                            status = value[0]
                            if status.get('confirmationStatus') == commitment:
                                logger.info(f"✅ Transaction confirmed: {signature}")
                                return True

                            if status.get('err'):
                                logger.error(f"Transaction failed: {status['err']}")
                                return False

                # Wait before checking again
                await asyncio.sleep(2)

            logger.warning(f"Transaction confirmation timeout: {signature}")
            return False

        except Exception as e:
            logger.error(f"Error confirming transaction: {e}", exc_info=True)
            return False

    async def execute_swap(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        user_public_key: str = None
    ) -> Optional[str]:
        """
        Execute complete swap: quote → transaction → sign → send → confirm

        Args:
            input_mint: Input token mint
            output_mint: Output token mint
            amount: Amount to swap
            slippage_bps: Slippage tolerance
            user_public_key: User public key (defaults to loaded keypair)

        Returns:
            Optional[str]: Transaction signature or None
        """
        # Use module-level logger that matches SolanaTradingEngine
        import logging
        swap_logger = logging.getLogger("SolanaTradingEngine")

        try:
            # Pre-check: Verify keypair is loaded before attempting swap
            if not self.keypair:
                swap_logger.error("❌ Jupiter: No keypair loaded - cannot sign transactions")
                swap_logger.error("   Check SOLANA_PRIVATE_KEY in database/secrets and .encryption_key file")
                return None

            # Use keypair's public key if not provided
            if not user_public_key:
                user_public_key = str(self.keypair.pubkey())
                swap_logger.debug(f"Using wallet pubkey: {user_public_key[:12]}...")

            if not user_public_key:
                swap_logger.error("❌ Jupiter: No user public key available for signing")
                return None

            # Ensure session is initialized
            if not self.session:
                await self.initialize()

            # 1. Get quote
            swap_logger.info(f"   📊 Jupiter: Getting quote for {input_mint[:8]}... → {output_mint[:8]}... amount={amount}")
            quote = await self.get_quote(input_mint, output_mint, amount, slippage_bps)
            if not quote:
                swap_logger.error(f"❌ Jupiter: Failed to get quote - token may be pump.fun-only or have no Jupiter routes")
                return None

            swap_logger.info(f"   ✅ Quote received: out={quote.get('outAmount', 'N/A')}")

            # 2. Get swap transaction
            swap_logger.info(f"   📝 Jupiter: Creating swap transaction...")
            swap_data = await self.get_swap_transaction(quote, user_public_key)
            if not swap_data:
                swap_logger.error("❌ Jupiter: Failed to get swap transaction")
                return None

            transaction_data = swap_data.get('swapTransaction')
            if not transaction_data:
                swap_logger.error("❌ Jupiter: No transaction data in response")
                return None

            # 3. Sign transaction
            swap_logger.info(f"   ✍️ Jupiter: Signing transaction...")
            signed_tx = self.sign_transaction(transaction_data)
            if not signed_tx:
                swap_logger.error("❌ Jupiter: Failed to sign transaction")
                return None

            # 4. Send transaction
            swap_logger.info(f"   📤 Jupiter: Sending transaction to network...")
            signature = await self.send_transaction(signed_tx)
            if not signature:
                swap_logger.error("❌ Jupiter: Failed to send transaction")
                return None

            # 5. Confirm transaction
            swap_logger.info(f"   ⏳ Jupiter: Waiting for confirmation...")
            confirmed = await self.confirm_transaction(signature)

            if confirmed:
                swap_logger.info(f"   ✅ Jupiter: Swap completed: {signature}")
                return signature
            else:
                swap_logger.warning(f"   ⚠️ Jupiter: Sent but unconfirmed: {signature}")
                return signature

        except Exception as e:
            swap_logger.error(f"❌ Jupiter swap error: {e}", exc_info=True)
            return None
