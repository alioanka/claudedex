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
            # Use the multi-format parser for provided private key
            self.keypair = self._load_keypair_from_value(private_key)
            if self.keypair:
                logger.info(f"✅ Jupiter keypair loaded (pubkey: {str(self.keypair.pubkey())[:12]}...)")
            else:
                logger.error("❌ Failed to load keypair from provided private_key")
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

        Supports multiple key formats:
        - JSON array: [1, 2, 3, ...] (64 bytes)
        - Base58 encoded: string of base58 characters
        - Hex encoded: hexadecimal string

        Args:
            pk_str: Private key string (base58 encoded, JSON array, hex, or encrypted)

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

            key_bytes = None
            format_used = None

            # Format 1: JSON array (e.g., [1,2,3,...])
            if pk_str.startswith('['):
                try:
                    import json
                    key_array = json.loads(pk_str)
                    key_bytes = bytes(key_array)
                    format_used = "JSON array"
                    logger.debug(f"Parsed private key from JSON array format ({len(key_bytes)} bytes)")
                except Exception as json_error:
                    logger.debug(f"Not JSON array format: {json_error}")

            # Format 2: Base58 encoded (most common)
            if key_bytes is None:
                try:
                    key_bytes = base58.b58decode(pk_str)
                    format_used = "base58"
                    logger.debug(f"Parsed private key from base58 format ({len(key_bytes)} bytes)")
                except (ValueError, Exception) as b58_error:
                    logger.debug(f"Not base58 format: {b58_error}")

            # Format 3: Hex encoded
            if key_bytes is None:
                try:
                    key_bytes = bytes.fromhex(pk_str)
                    format_used = "hex"
                    logger.debug(f"Parsed private key from hex format ({len(key_bytes)} bytes)")
                except (ValueError, Exception) as hex_error:
                    logger.debug(f"Not hex format: {hex_error}")

            if key_bytes is None:
                logger.error("Failed to parse private key - tried JSON array, base58, and hex formats")
                return None

            # Verify key length (should be 64 bytes: 32 secret + 32 public)
            if len(key_bytes) == 64:
                keypair = Keypair.from_bytes(key_bytes)
            elif len(key_bytes) == 32:
                # Some keys are just the 32-byte seed - create keypair from seed
                keypair = Keypair.from_seed(key_bytes)
            else:
                logger.error(f"Invalid key length: {len(key_bytes)} bytes (expected 32 or 64)")
                return None

            logger.info(f"✅ Keypair loaded from {format_used} format (pubkey: {str(keypair.pubkey())[:12]}...)")
            return keypair

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

            # Get the message for signing
            message = tx.message
            our_pubkey = self.keypair.pubkey()
            our_pubkey_str = str(our_pubkey)

            # CRITICAL: Verify that the fee payer (first account) matches our keypair
            # This catches pubkey mismatches that would cause signature verification failure
            # Handle both legacy Message and MessageV0 formats
            fee_payer = None
            try:
                # Try direct attribute access (works for most message types)
                if hasattr(message, 'account_keys') and message.account_keys:
                    fee_payer = message.account_keys[0]
                # Alternative: some versions use static_account_keys()
                elif hasattr(message, 'static_account_keys'):
                    keys = message.static_account_keys()
                    if keys:
                        fee_payer = keys[0]
            except Exception as e:
                logger.warning(f"Could not extract fee payer from message: {e}")

            if fee_payer:
                fee_payer_str = str(fee_payer)
                logger.info(f"   🔑 Fee payer: {fee_payer_str[:12]}... | Our key: {our_pubkey_str[:12]}...")

                if fee_payer_str != our_pubkey_str:
                    logger.error(f"❌ PUBKEY MISMATCH! Transaction expects fee payer: {fee_payer_str}")
                    logger.error(f"   But we are signing with pubkey: {our_pubkey_str}")
                    logger.error("   This will cause 'Transaction signature verification failure'")
                    logger.error("   Check that SOLANA_MODULE_PRIVATE_KEY matches the wallet pubkey")
                    return None
                logger.info("   ✓ Fee payer matches our keypair")
            else:
                logger.warning(f"   ⚠️ Could not verify fee payer, signing with: {our_pubkey_str[:12]}...")

            # Sign the transaction message and create signed transaction
            # Note: solders VersionedTransaction doesn't have a .sign() method
            # We need to sign the message and use VersionedTransaction.populate()
            logger.debug(f"Signing with keypair (pubkey: {str(our_pubkey)[:12]}...)")
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
        # Use SolanaTradingEngine logger for visibility in main logs
        tx_logger = logging.getLogger("SolanaTradingEngine")

        try:
            if not self.solana_rpc:
                tx_logger.error("   ❌ No Solana RPC URL configured")
                return None

            if not self.session:
                await self.initialize()

            # Log the RPC being used
            tx_logger.info(f"   📡 Using RPC: {self.solana_rpc[:50]}...")

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
                        tx_logger.info(f"   ✅ Transaction sent: {signature}")
                        return signature
                    elif 'error' in result:
                        error = result['error']
                        error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                        error_code = error.get('code', 'N/A') if isinstance(error, dict) else 'N/A'
                        # Log detailed error that will be visible in main logs
                        tx_logger.error(f"   ❌ RPC error [{error_code}]: {error_msg}")
                        # Also log any additional error data (like transaction simulation logs)
                        if isinstance(error, dict) and 'data' in error:
                            data = error['data']
                            if isinstance(data, dict):
                                logs = data.get('logs', [])
                                if logs:
                                    tx_logger.error(f"   📋 Simulation logs: {logs[-3:]}")
                            else:
                                tx_logger.error(f"   📋 Error data: {str(data)[:200]}")
                        return None
                else:
                    error_text = await response.text()
                    tx_logger.error(f"   ❌ HTTP {response.status}: {error_text[:200]}")
                    return None

        except Exception as e:
            tx_logger.error(f"   ❌ Exception sending transaction: {e}", exc_info=True)
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
