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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Jupiter executor
        
        Args:
            config: Flat config dict with keys like:
                - rpc_url: Solana RPC endpoint
                - private_key: Base58 encoded private key
                - max_slippage_bps: Maximum slippage in basis points (default: 500 = 5%)
                - enabled: Whether Solana trading is enabled
        """
        super().__init__(config)
        
        # Jupiter v1 API Configuration
        self.jupiter_api_url = "https://lite-api.jup.ag/swap/v1"
        
        # Solana Configuration
        self.rpc_url = config.get('rpc_url') or config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')
        self.max_slippage_bps = int(config.get('max_slippage_bps', 500))  # 5% default
        
        # Initialize keypair from private key
        private_key = config.get('private_key') or config.get('solana_private_key')
        if private_key and Keypair:
            try:
                # Handle base58 encoded private key
                if isinstance(private_key, str):
                    private_key_bytes = base58.b58decode(private_key)
                    self.keypair = Keypair.from_bytes(private_key_bytes)
                else:
                    self.keypair = Keypair.from_bytes(bytes(private_key))
                    
                self.wallet_address = str(self.keypair.pubkey())
                logger.info(f"🟣 Jupiter executor initialized for wallet: {self.wallet_address[:8]}...")
            except Exception as e:
                logger.error(f"Failed to initialize Solana keypair: {e}")
                self.keypair = None
                self.wallet_address = None
        else:
            if not Keypair:
                logger.warning("solders library not available - limited functionality")
            if not private_key:
                logger.warning("No Solana private key provided - executor will not be able to sign transactions")
            self.keypair = None
            self.wallet_address = None
        
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
            
            return {
                'success': True,
                'signature': swap_result.get('signature'),
                'quote': quote,
                'execution_time': swap_result.get('execution_time'),
                'gas_used': swap_result.get('gas_used')
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}", exc_info=True)
            self.execution_stats['failed_trades'] += 1
            return {
                'success': False,
                'error': str(e)
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
            # Check if we're in dry run mode
            if self.config.get('dry_run', True):
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
            
            # Sign transaction
            signed_tx = self.keypair.sign_message(bytes(transaction.message))
            transaction.signatures[0] = signed_tx
            
            # Send transaction to Solana network
            serialized_tx = base64.b64encode(bytes(transaction)).decode('utf-8')
            
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
            
            start_time = time.time()
            
            async with self.session.post(self.rpc_url, json=rpc_payload) as rpc_response:
                if rpc_response.status != 200:
                    error_text = await rpc_response.text()
                    logger.error(f"RPC sendTransaction failed: {rpc_response.status} - {error_text}")
                    return {
                        'success': False,
                        'error': f'RPC call failed: {error_text}'
                    }
                
                result = await rpc_response.json()
            
            execution_time = time.time() - start_time
            
            if 'error' in result:
                logger.error(f"Transaction error: {result['error']}")
                return {
                    'success': False,
                    'error': result['error'].get('message', 'Unknown error')
                }
            
            signature = result.get('result')
            
            if not signature:
                logger.error("No signature in RPC response")
                return {
                    'success': False,
                    'error': 'No signature returned'
                }
            
            logger.info(f"✅ Transaction sent: {signature}")
            
            # Wait for confirmation (optional - can be done async)
            confirmed = await self._wait_for_confirmation(signature)
            
            return {
                'success': True,
                'signature': signature,
                'execution_time': execution_time,
                'confirmed': confirmed,
                'gas_used': 0.0005  # Estimate, actual value would come from transaction details
            }
            
        except Exception as e:
            logger.error(f"Error executing swap: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _wait_for_confirmation(
        self,
        signature: str,
        max_wait_seconds: int = 30
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
        if self.session:
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