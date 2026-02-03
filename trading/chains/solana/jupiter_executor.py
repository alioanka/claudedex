# trading/chains/solana/jupiter_executor.py
"""
Jupiter Aggregator v1 API Integration for Solana Trading
Updated to use lite-api.jup.ag/swap/v1 endpoints
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import base58
import base64
import os

from monitoring.logger import log_trade_entry

try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from solders.message import Message as TransactionMessage
except ImportError:
    # Fallback if solders not available
    Keypair = None
    Pubkey = None
    VersionedTransaction = None
    TransactionMessage = None

from trading.executors.base_executor import BaseExecutor

logger = logging.getLogger(__name__)


class JupiterExecutor(BaseExecutor):
    """
    Jupiter Aggregator v1 API integration for Solana DEX trading
    
    Features:
    - Best price routing across all Solana DEXes
    - Auto slippage calculation
    - Versioned transactions support
    - Automatic retry with fallback RPCs
    """
    
    def __init__(self, config: Dict[str, Any], db_manager=None):
        """
        Initialize Jupiter executor
        
        Args:
            config: Flat config dict with keys like:
                - rpc_url: Solana RPC endpoint
                - private_key: Base58 encoded private key
                - max_slippage_bps: Maximum slippage in basis points (default: 50 = 0.5%)
                - enabled: Whether Solana trading is enabled
        """
        super().__init__(config, db_manager)

        # ✅ CRITICAL: DRY_RUN mode check (standardize key name)
        self.dry_run = config.get('DRY_RUN', True) or config.get('dry_run', True)
        if self.dry_run:
            logger.warning("🔶 JUPITER EXECUTOR IN DRY RUN MODE - NO REAL TRANSACTIONS 🔶")
        else:
            logger.critical("🔥 JUPITER EXECUTOR IN LIVE MODE - REAL MONEY AT RISK 🔥")
        
        # Jupiter v1 API Configuration
        self.jupiter_api_url = "https://lite-api.jup.ag/swap/v1"
        
        # Solana Configuration
        self.rpc_url = config.get('rpc_url') or config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')
        # CRITICAL FIX: Reduced from 500 bps (5%) to 50 bps (0.5%) to prevent MEV sandwich attacks
        self.max_slippage_bps = int(config.get('max_slippage_bps', 50))  # 0.5% default (was 5%)
        
        # Initialize keypair from private key
        private_key = config.get('solana_private_key') or config.get('private_key')
        # ✅ ADD DECRYPTION LOGIC:
        if private_key and private_key.startswith('gAAAAA'):  # Encrypted format
            try:
                from cryptography.fernet import Fernet
                encryption_key = config.get('encryption_key') or os.getenv('ENCRYPTION_KEY')
                if encryption_key:
                    cipher = Fernet(encryption_key.encode())
                    private_key = cipher.decrypt(private_key.encode()).decode()
                    logger.info("✅ Solana private key decrypted")
            except Exception as e:
                logger.error(f"Failed to decrypt Solana private key: {e}")
                raise

        if private_key and Keypair:
            try:
                # ✅ IMPROVED: Validate and handle multiple formats
                if isinstance(private_key, str):
                    # Try base58 format first (most common, 88 chars)
                    if len(private_key) == 88:
                        try:
                            private_key_bytes = base58.b58decode(private_key)
                            if len(private_key_bytes) != 64:
                                raise ValueError(f"Invalid key length: {len(private_key_bytes)}, expected 64")
                            self.keypair = Keypair.from_bytes(private_key_bytes)
                            logger.info("✅ Loaded Solana keypair from base58 string")
                        
                        except Exception as e:
                            raise ValueError(f"Invalid base58 Solana private key: {e}")
                    
                    # Try hex format (128 chars)
                    elif len(private_key) == 128:
                        try:
                            private_key_bytes = bytes.fromhex(private_key)
                            self.keypair = Keypair.from_bytes(private_key_bytes)
                            logger.info("✅ Loaded Solana keypair from hex string")
                        except Exception as e:
                            raise ValueError(f"Invalid hex Solana private key: {e}")
                    
                    # Try JSON array format [1,2,3,...]
                    elif private_key.startswith('['):
                        try:
                            import json
                            key_array = json.loads(private_key)
                            private_key_bytes = bytes(key_array)
                            if len(private_key_bytes) != 64:
                                raise ValueError(f"Invalid key length: {len(private_key_bytes)}")
                            self.keypair = Keypair.from_bytes(private_key_bytes)
                            logger.info("✅ Loaded Solana keypair from JSON array")
                        except Exception as e:
                            raise ValueError(f"Invalid JSON array Solana private key: {e}")
                    
                    else:
                        raise ValueError(
                            f"Unsupported Solana private key format. "
                            f"Expected base58 (88 chars), hex (128 chars), or JSON array. "
                            f"Got length: {len(private_key)}"
                        )
                
                # Byte array format
                elif isinstance(private_key, (list, bytes, bytearray)):
                    private_key_bytes = bytes(private_key)
                    if len(private_key_bytes) != 64:
                        raise ValueError(f"Invalid key length: {len(private_key_bytes)}, expected 64")
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                    logger.info("✅ Loaded Solana keypair from bytes")
                
                else:
                    raise ValueError(f"Invalid private key type: {type(private_key)}")
                
                # Validate keypair was created successfully
                if not self.keypair:
                    raise ValueError("Keypair creation failed")
                
                self.wallet_address = str(self.keypair.pubkey())
                logger.info(f"🟣 Jupiter executor initialized for wallet: {self.wallet_address[:8]}...")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Solana keypair: {e}")
                raise  # ✅ CRITICAL: Raise instead of silently continuing
                
        else:
            if not Keypair:
                logger.error("❌ solders library not available - cannot initialize Solana trading")
            if not private_key:
                logger.error("❌ No Solana private key provided")
            raise ValueError("Cannot initialize Jupiter executor without valid private key")
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_fees_sol': Decimal('0'),
            'total_slippage_bps': 0
        }
        
        # Known token mints (for reference)
        self.known_tokens = {
            'SOL': 'So11111111111111111111111111111111111111112',  # Wrapped SOL
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
            'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
            'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
            'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R'
        }
        
        logger.info(f"Jupiter Executor initialized with max slippage: {self.max_slippage_bps} bps")
    
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Jupiter executor session initialized")
    
    async def execute_trade(self, order) -> Dict[str, Any]:
        """
        Execute a trade using Jupiter aggregator
        
        Args:
            order: Order object with token details
            
        Returns:
            Dict with execution results
        """
        if not self.session:
            await self.initialize()

            # ✅ CRITICAL: DRY_RUN CHECK AT TOP LEVEL
            if self.dry_run:
                logger.info(f"🔶 DRY RUN: Simulating Jupiter trade for {getattr(order, 'symbol', 'unknown')}")
                return await self._simulate_jupiter_trade(order)
        
        try:
            logger.info(f"🟣 Executing Solana trade for {getattr(order, 'symbol', 'unknown')}")
            
            # Step 1: Get quote from Jupiter v1 API
            quote = await self._get_quote(
                input_mint=order.token_in,
                output_mint=order.token_out,
                amount=order.amount_in,
                slippage_bps=self.max_slippage_bps
            )
            
            if not quote:
                logger.error("Failed to get Jupiter quote")
                return {'success': False, 'error': 'Failed to get quote'}
            
            logger.info(f"📊 Jupiter quote received:")
            logger.info(f"   Input: {quote['inAmount']} ({getattr(order, 'symbol_in', 'unknown')})")
            logger.info(f"   Output: {quote['outAmount']} ({getattr(order, 'symbol_out', 'unknown')})")
            logger.info(f"   Price Impact: {quote.get('priceImpactPct', 'N/A')}%")
            
            # Step 2: Get swap transaction
            swap_result = await self._execute_swap(quote, order)
            
            if not swap_result.get('success'):
                logger.error(f"Swap execution failed: {swap_result.get('error')}")
                return swap_result
            
            # Step 3: Update stats
            self.execution_stats['total_trades'] += 1
            self.execution_stats['successful_trades'] += 1
            
            logger.info(f"✅ Trade executed successfully: {swap_result.get('signature')}")

            # 🆕 PATCH: Structured Solana trade logging
            try:
                log_trade_entry(
                    chain='SOLANA',
                    symbol=getattr(order, 'symbol_out', 'UNKNOWN'),
                    token_address=order.token_out,
                    trade_id=swap_result.get('signature', 'no_sig'),
                    entry_price=float(quote['outAmount']) / float(quote['inAmount']),
                    amount=float(quote['outAmount']) / 1e9,  # Convert from lamports
                    size_usd=float(quote['inAmount']) / 1e9,  # Convert from lamports
                    reason="jupiter_swap"
                )
            except Exception as log_err:
                logger.warning(f"Failed to log Solana trade entry: {log_err}")

            
            # ✅ FIXED: Standardized return format
            return {
                'success': True,
                'tx_hash': swap_result.get('signature'),  # Add tx_hash alias
                'signature': swap_result.get('signature'),
                'quote': quote,
                'execution_time': swap_result.get('execution_time'),
                'gas_used': swap_result.get('gas_used'),
                'execution_price': int(quote['outAmount']) / int(quote['inAmount']),
                'amount': int(quote['inAmount']),
                'token_amount': int(quote['outAmount']),
                'slippage_actual': float(quote.get('priceImpactPct', 0)),
                'route': 'jupiter',
                'metadata': {
                    'chain': 'solana',
                    'dex': 'jupiter',
                    'price_impact': quote.get('priceImpactPct')
                }
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}", exc_info=True)
            self.execution_stats['failed_trades'] += 1
            return {
                'success': False,
                'error': str(e)
            }

    async def get_balance(self, token_mint: Optional[str] = None) -> float:
            """
            Get SOL or token balance for the wallet
            
            Args:
                token_mint: Optional token mint address. If None, returns SOL balance
                
            Returns:
                Balance as float (in SOL or token units)
            """
            if not self.session:
                await self.initialize()
            
            try:
                if not self.keypair or not self.wallet_address:
                    logger.error("❌ Wallet not initialized")
                    return 0.0
                
                # Get SOL balance by default
                if token_mint is None or token_mint == 'So11111111111111111111111111111111111111112':
                    # Query Solana RPC for SOL balance
                    async with self.session.post(
                        self.rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "getBalance",
                            "params": [self.wallet_address]
                        },
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'result' in data and 'value' in data['result']:
                                # Convert lamports to SOL (1 SOL = 1e9 lamports)
                                lamports = data['result']['value']
                                sol_balance = lamports / 1e9
                                logger.debug(f"SOL balance: {sol_balance:.4f} SOL")
                                return sol_balance
                            else:
                                logger.error(f"Unexpected RPC response: {data}")
                                return 0.0
                        else:
                            logger.error(f"RPC request failed with status {response.status}")
                            return 0.0
                
                # Get token balance
                else:
                    async with self.session.post(
                        self.rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "getTokenAccountsByOwner",
                            "params": [
                                self.wallet_address,
                                {"mint": token_mint},
                                {"encoding": "jsonParsed"}
                            ]
                        },
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'result' in data and 'value' in data['result'] and data['result']['value']:
                                # Get token balance from first account
                                account_data = data['result']['value'][0]['account']['data']['parsed']['info']
                                token_amount = float(account_data['tokenAmount']['uiAmount'])
                                logger.debug(f"Token balance: {token_amount}")
                                return token_amount
                            else:
                                # No token account found - balance is 0
                                return 0.0
                        else:
                            logger.error(f"RPC request failed with status {response.status}")
                            return 0.0
            
            except Exception as e:
                logger.error(f"Error getting Solana balance: {e}", exc_info=True)
                return 0.0


    async def _simulate_jupiter_trade(self, order) -> Dict[str, Any]:
        """Simulate Jupiter trade for paper trading"""
        try:
            import random
            
            # Get real quote for simulation
            quote = await self._get_quote(
                input_mint=order.token_in,
                output_mint=order.token_out,
                amount=order.amount_in,
                slippage_bps=self.max_slippage_bps
            )
            
            if not quote:
                return {
                    'success': False,
                    'error': 'No quotes available for simulation',
                    'tx_hash': None
                }
            
            # Simulate execution delay
            await asyncio.sleep(random.uniform(1.0, 2.5))
            
            # Calculate simulated values
            in_amount = int(quote['inAmount'])
            out_amount = int(quote['outAmount'])
            price = out_amount / in_amount if in_amount > 0 else 0
            
            logger.info(f"✅ DRY RUN Jupiter swap:")
            logger.info(f"   Input: {in_amount} ({getattr(order, 'symbol_in', 'unknown')})")
            logger.info(f"   Output: {out_amount} ({getattr(order, 'symbol_out', 'unknown')})")
            logger.info(f"   Price Impact: {quote.get('priceImpactPct', 'N/A')}%")
            
            # Update simulation stats
            self.execution_stats['total_trades'] += 1
            self.execution_stats['successful_trades'] += 1
            
            return {
                'success': True,
                'tx_hash': f"0xDRYRUN_SOLANA_{int(time.time())}{random.randint(1000, 9999)}",
                'signature': f"DRY_RUN_SOL_{int(time.time())}{random.randint(1000, 9999)}",
                'quote': quote,
                'execution_time': random.uniform(1.0, 2.5),
                'gas_used': 0.0005,  # Estimated SOL fee
                'execution_price': price,
                'amount': in_amount,
                'token_amount': out_amount,
                'slippage_actual': float(quote.get('priceImpactPct', 0)),
                'route': 'jupiter_simulated',
                'metadata': {
                    'dry_run': True,
                    'chain': 'solana',
                    'dex': 'jupiter'
                }
            }
            
        except Exception as e:
            logger.error(f"Jupiter simulation error: {e}", exc_info=True)
            self.execution_stats['failed_trades'] += 1
            return {
                'success': False,
                'error': f"Simulation failed: {str(e)}",
                'tx_hash': None
            }
    
    async def get_quote(
        self,
        token_in: str,
        token_out: str,
        amount_in: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get quote for swap (BaseExecutor abstract method implementation)
        
        Args:
            token_in: Input token mint address
            token_out: Output token mint address
            amount_in: Input amount in smallest units
            
        Returns:
            Quote dict or None
        """
        return await self._get_quote(
            input_mint=token_in,
            output_mint=token_out,
            amount=amount_in,
            slippage_bps=self.max_slippage_bps
        )
    
    async def validate_order(self, order) -> bool:
        """
        Validate order parameters (BaseExecutor abstract method implementation)
        
        Args:
            order: Order object to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate token addresses
            if not getattr(order, 'token_in', None) or not getattr(order, 'token_out', None):
                logger.error("Missing token_in or token_out")
                return False
            
            # Validate amount
            amount_in = getattr(order, 'amount_in', None)
            if not amount_in or amount_in <= 0:
                logger.error("Invalid amount_in")
                return False
            
            # Validate slippage
            if hasattr(order, 'max_slippage_bps'):
                if order.max_slippage_bps < 0 or order.max_slippage_bps > 10000:
                    logger.error("Invalid slippage (must be 0-10000 bps)")
                    return False

            # ✅ NEW: Check position size limits
            max_position_size = self.config.get('MAX_POSITION_SIZE_USD', 50)
            
            # Convert amount_in to approximate USD (simplified)
            # In production, you'd want to get real price
            if amount_in > max_position_size * 1_000_000_000:  # Rough lamports estimate
                logger.error(f"Position size exceeds limit")
                return False
            
            # ✅ NEW: Check if we have sufficient balance (only if not dry run)
            if not self.dry_run and self.wallet_address:
                balance = await self.get_token_balance(order.token_in)
                # Convert to smallest units for comparison
                if int(balance * 1_000_000_000) < amount_in:
                    logger.error(f"Insufficient balance: {balance} < {amount_in / 1_000_000_000}")
                    return False

            # Check if we can get a quote (validates tokens exist)
            quote = await self.get_quote(
                token_in=order.token_in,
                token_out=order.token_out,
                amount_in=amount_in
            )
            
            if not quote:
                logger.error("Unable to get quote for order validation")
                return False
            
            logger.debug(f"Order validation passed for {getattr(order, 'symbol', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    def _is_quote_expired(self, quote: Dict[str, Any], max_age_seconds: int = 10) -> bool:
        """
        CRITICAL FIX (P1): Check if quote is too old to execute safely.

        Jupiter quotes can become stale quickly in volatile markets.
        Executing stale quotes leads to:
        - Unexpected slippage
        - Failed transactions
        - MEV sandwich attacks

        Args:
            quote: Quote dict with '_fetched_at' timestamp
            max_age_seconds: Maximum age in seconds (default: 10)

        Returns:
            True if quote is expired, False if still fresh
        """
        if '_fetched_at' not in quote:
            logger.warning("Quote missing timestamp - treating as expired")
            return True

        age = time.time() - quote['_fetched_at']
        is_expired = age > max_age_seconds

        if is_expired:
            logger.warning(
                f"⚠️ Quote expired: {age:.1f}s old (max: {max_age_seconds}s) - "
                "prices may have changed significantly"
            )

        return is_expired

    async def _get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 500
    ) -> Optional[Dict[str, Any]]:
        """
        Get swap quote from Jupiter v1 API
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit (raw amount before decimals)
            slippage_bps: Slippage tolerance in basis points
            
        Returns:
            Quote dict or None if failed
        """
        try:
            url = f"{self.jupiter_api_url}/quote"
            
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'swapMode': 'ExactIn',
                'onlyDirectRoutes': 'false',  # Must be string, not boolean
                'asLegacyTransaction': 'false',  # Must be string, not boolean
                'maxAccounts': '64',
            }
            
            logger.debug(f"Requesting Jupiter quote with params: {params}")
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Jupiter quote request failed: {response.status} - {error_text}")
                    return None
                
                quote_data = await response.json()

                # Validate required fields
                required_fields = ['inputMint', 'outputMint', 'inAmount', 'outAmount']
                if not all(field in quote_data for field in required_fields):
                    logger.error(f"Quote missing required fields: {quote_data.keys()}")
                    return None

                # CRITICAL FIX (P1): Add timestamp for expiration checking
                quote_data['_fetched_at'] = time.time()

                return quote_data
                
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}", exc_info=True)
            return None
    
    async def _execute_swap(
        self,
        quote: Dict[str, Any],
        order
    ) -> Dict[str, Any]:
        """
        Execute swap using Jupiter v1 API
        
        Args:
            quote: Quote dict from _get_quote
            order: Original order object
            
        Returns:
            Dict with execution results
        """
        try:
            # CRITICAL FIX (P1): Reject stale quotes to prevent executing at outdated prices
            if self._is_quote_expired(quote):
                logger.error("❌ Quote expired - refusing to execute swap with stale price data")
                return {
                    'success': False,
                    'error': 'Quote expired (>10 seconds old)',
                    'quote_age': time.time() - quote.get('_fetched_at', 0)
                }

            # Check if we're in dry run mode
            if self.dry_run:  # Use instance variable instead of config
                logger.info("🔸 DRY RUN MODE - Simulating swap execution")
                return {
                    'success': True,
                    'signature': f"DRY_RUN_{int(time.time())}",
                    'execution_time': 0.5,
                    'gas_used': 0.0005,  # Estimated SOL for transaction fee
                    'dry_run': True
                }
            
            if not self.keypair:
                logger.error("Cannot execute real swap: No keypair configured")
                return {
                    'success': False,
                    'error': 'No keypair configured for signing'
                }
            
            # Request swap transaction from Jupiter
            url = f"{self.jupiter_api_url}/swap"
            
            payload = {
                'quoteResponse': quote,
                'userPublicKey': self.wallet_address,
                'wrapAndUnwrapSol': True,
                'dynamicComputeUnitLimit': True,
                'prioritizationFeeLamports': 'auto'
            }
            
            logger.debug(f"Requesting swap transaction from Jupiter")
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Jupiter swap request failed: {response.status} - {error_text}")
                    return {
                        'success': False,
                        'error': f'Swap request failed: {error_text}'
                    }
                
                swap_data = await response.json()
            
            # Extract serialized transaction
            swap_transaction_str = swap_data.get('swapTransaction')
            if not swap_transaction_str:
                logger.error("No swap transaction in response")
                return {
                    'success': False,
                    'error': 'No transaction in swap response'
                }
            
            # Deserialize and sign transaction
            transaction_bytes = base64.b64decode(swap_transaction_str)
            transaction = VersionedTransaction.from_bytes(transaction_bytes)

            # Get message and verify pubkey match before signing
            message = transaction.message
            our_pubkey = self.keypair.pubkey()

            # CRITICAL: Verify fee payer matches our keypair
            if hasattr(message, 'account_keys') and len(message.account_keys) > 0:
                fee_payer = message.account_keys[0]
                if str(fee_payer) != str(our_pubkey):
                    logger.error(f"❌ PUBKEY MISMATCH! TX expects: {fee_payer}, we have: {our_pubkey}")
                    return {
                        'success': False,
                        'error': f'Pubkey mismatch: expected {fee_payer}'
                    }

            # Sign transaction using correct VersionedTransaction.populate() pattern
            # The .sign() method doesn't work with solders VersionedTransaction
            signature = self.keypair.sign_message(bytes(message))
            signed_transaction = VersionedTransaction.populate(message, [signature])

            # Send transaction to Solana network (with retry)
            serialized_tx = base64.b64encode(bytes(signed_transaction)).decode('utf-8')
            
            start_time = time.time()
            
            # ✅ FIXED: Use retry method instead of single attempt
            send_result = await self._send_transaction_with_retry(serialized_tx, max_retries=3)
            
            execution_time = time.time() - start_time
            
            if not send_result['success']:
                logger.error(f"Transaction send failed: {send_result['error']}")
                return {
                    'success': False,
                    'error': send_result['error']
                }
            
            signature = send_result['signature']
            logger.info(f"✅ Transaction sent: {signature}")
            
            # Wait for confirmation
            confirmed = await self._wait_for_confirmation(signature)
            
            return {
                'success': True,
                'signature': signature,
                'execution_time': execution_time,
                'confirmed': confirmed,
                'gas_used': 0.0005  # Estimate, actual from transaction details
            }

        # ✅ ADD THESE LINES:
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _wait_for_confirmation(
        self,
        signature: str,
        max_wait_seconds: int = 60
    ) -> bool:
        """
        Wait for transaction confirmation
        
        Args:
            signature: Transaction signature
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if confirmed, False otherwise
        """
        try:
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": True}]
            }
            
            start_time = time.time()
            
            while time.time() - start_time < max_wait_seconds:
                async with self.session.post(self.rpc_url, json=rpc_payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'result' in result and 'value' in result['result']:
                            statuses = result['result']['value']
                            if statuses and statuses[0]:
                                status = statuses[0]
                                
                                # Check if confirmed
                                if status.get('confirmationStatus') in ['confirmed', 'finalized']:
                                    logger.info(f"✅ Transaction confirmed: {signature}")
                                    return True
                                
                                # Check for errors
                                if status.get('err'):
                                    logger.error(f"Transaction failed: {status['err']}")
                                    return False
                
                # Wait before next check
                await asyncio.sleep(2)
            
            logger.warning(f"Transaction confirmation timeout: {signature}")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for confirmation: {e}")
            return False


    async def _send_transaction_with_retry(
        self,
        serialized_tx: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Send transaction to Solana with retry logic
        
        Args:
            serialized_tx: Base64 encoded transaction
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with success status and signature
        """
        for attempt in range(max_retries):
            try:
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "sendTransaction",
                    "params": [
                        serialized_tx,
                        {
                            "encoding": "base64",
                            "skipPreflight": False,
                            "preflightCommitment": "confirmed",
                            "maxRetries": 3
                        }
                    ]
                }
                
                async with self.session.post(self.rpc_url, json=rpc_payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'error' in result:
                            error_msg = result['error'].get('message', 'Unknown error')
                            
                            # Don't retry certain errors
                            if 'already been processed' in error_msg:
                                logger.warning("Transaction already processed")
                                return {'success': False, 'error': error_msg}
                            
                            if attempt < max_retries - 1:
                                logger.warning(f"RPC error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            else:
                                return {'success': False, 'error': error_msg}
                        
                        signature = result.get('result')
                        if signature:
                            return {'success': True, 'signature': signature}
                        else:
                            return {'success': False, 'error': 'No signature in response'}
                    else:
                        error_text = await response.text()
                        if attempt < max_retries - 1:
                            logger.warning(f"HTTP {response.status} (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            return {'success': False, 'error': f'HTTP {response.status}: {error_text}'}
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Send error (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Max retries exceeded'}

    async def get_token_balance(self, token_mint: str) -> Decimal:
        """
        Get SPL token balance for wallet
        
        Args:
            token_mint: Token mint address
            
        Returns:
            Token balance as Decimal
        """
        if not self.wallet_address:
            return Decimal('0')
        
        try:
            # Special case for SOL
            if token_mint == self.known_tokens['SOL']:
                return await self._get_sol_balance()
            
            # Get SPL token balance
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    self.wallet_address,
                    {"mint": token_mint},
                    {"encoding": "jsonParsed"}
                ]
            }
            
            async with self.session.post(self.rpc_url, json=rpc_payload) as response:
                if response.status != 200:
                    return Decimal('0')
                
                result = await response.json()
                
                if 'result' in result and 'value' in result['result']:
                    accounts = result['result']['value']
                    if accounts:
                        token_amount = accounts[0]['account']['data']['parsed']['info']['tokenAmount']
                        return Decimal(token_amount['uiAmount'])
            
            return Decimal('0')
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return Decimal('0')
    
    async def _get_sol_balance(self) -> Decimal:
        """Get SOL balance for wallet"""
        try:
            rpc_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [self.wallet_address]
            }
            
            async with self.session.post(self.rpc_url, json=rpc_payload) as response:
                if response.status != 200:
                    return Decimal('0')
                
                result = await response.json()
                
                if 'result' in result and 'value' in result['result']:
                    lamports = result['result']['value']
                    return Decimal(lamports) / Decimal('1000000000')  # Convert lamports to SOL
            
            return Decimal('0')
            
        except Exception as e:
            logger.error(f"Error getting SOL balance: {e}")
            return Decimal('0')
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_trades = self.execution_stats['total_trades']
        
        if total_trades == 0:
            return self.execution_stats
        
        return {
            **self.execution_stats,
            'success_rate': self.execution_stats['successful_trades'] / total_trades,
            'average_fee_sol': float(self.execution_stats['total_fees_sol'] / total_trades)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        
        logger.info("Jupiter executor cleaned up")
    
    def __del__(self):
        """Destructor"""
        if self.session and not self.session.closed:
            try:
                asyncio.get_event_loop().run_until_complete(self.cleanup())
            except:
                pass