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
        self.api_url = os.getenv('JUPITER_API_URL', 'https://quote-api.jup.ag/v6')
        self.solana_rpc = solana_rpc_url or os.getenv('SOLANA_RPC_URL')

        # Load keypair for signing
        if private_key:
            self.keypair = Keypair.from_base58_string(private_key)
        elif os.getenv('SOLANA_PRIVATE_KEY'):
            # Load from environment (may be encrypted)
            self.keypair = self._load_keypair_from_env()
        else:
            self.keypair = None
            logger.warning("No Solana private key provided, signing will fail")

        self.session: Optional[aiohttp.ClientSession] = None

    def _load_keypair_from_env(self) -> Optional[Keypair]:
        """
        Load keypair from environment variable

        Returns:
            Optional[Keypair]: Loaded keypair or None
        """
        try:
            # This assumes the private key in .env is the base58-encoded secret key
            # If it's encrypted, you'd need to decrypt it first
            pk_str = os.getenv('SOLANA_PRIVATE_KEY')
            if not pk_str:
                return None

            # Try to parse as base58
            try:
                return Keypair.from_base58_string(pk_str)
            except:
                # If that fails, it might be encrypted - would need decryption logic
                logger.warning("Private key appears to be encrypted, implement decryption")
                return None

        except Exception as e:
            logger.error(f"Error loading keypair: {e}")
            return None

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
                    logger.error(f"Jupiter quote error: {response.status} - {error_text}")
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
                return None

            # Decode the transaction
            import base64
            tx_bytes = base64.b64decode(transaction_data)

            # Parse as versioned transaction
            tx = VersionedTransaction.from_bytes(tx_bytes)

            # Sign the transaction
            tx.sign([self.keypair])

            # Encode back to base64
            signed_tx_bytes = bytes(tx)
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
            if not self.solana_rpc or not self.session:
                return False

            start_time = asyncio.get_event_loop().time()

            while (asyncio.get_event_loop().time() - start_time) < timeout:
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
        try:
            # Use keypair's public key if not provided
            if not user_public_key and self.keypair:
                user_public_key = str(self.keypair.pubkey())

            if not user_public_key:
                logger.error("No user public key available")
                return None

            # 1. Get quote
            logger.info(f"🔄 Getting quote for swap...")
            quote = await self.get_quote(input_mint, output_mint, amount, slippage_bps)
            if not quote:
                return None

            # 2. Get swap transaction
            logger.info(f"📝 Creating swap transaction...")
            swap_data = await self.get_swap_transaction(quote, user_public_key)
            if not swap_data:
                return None

            transaction_data = swap_data.get('swapTransaction')
            if not transaction_data:
                logger.error("No transaction data in response")
                return None

            # 3. Sign transaction
            logger.info(f"✍️ Signing transaction...")
            signed_tx = self.sign_transaction(transaction_data)
            if not signed_tx:
                return None

            # 4. Send transaction
            logger.info(f"📤 Sending transaction...")
            signature = await self.send_transaction(signed_tx)
            if not signature:
                return None

            # 5. Confirm transaction
            logger.info(f"⏳ Waiting for confirmation...")
            confirmed = await self.confirm_transaction(signature)

            if confirmed:
                logger.info(f"✅ Swap completed successfully: {signature}")
                return signature
            else:
                logger.warning(f"⚠️ Swap transaction sent but confirmation uncertain: {signature}")
                return signature

        except Exception as e:
            logger.error(f"❌ Error executing swap: {e}", exc_info=True)
            return None
