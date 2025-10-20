# trading/chains/solana/jupiter_executor.py
"""
Jupiter Aggregator Integration for Solana Trading
Provides DEX aggregation across Solana ecosystem
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import base58
import base64

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.hash import Hash

from trading.orders.order_manager import Order, OrderType, OrderStatus
from trading.executors.base_executor import BaseExecutor
from utils.helpers import retry_async, measure_time

logger = logging.getLogger(__name__)

class JupiterRoute(Enum):
    """Jupiter routing preferences"""
    BEST_PRICE = "bestPrice"           # Best output amount
    LEAST_SLIPPAGE = "leastSlippage"   # Minimum slippage
    FASTEST = "fastest"                 # Fastest execution
    
@dataclass
class JupiterQuote:
    """Quote from Jupiter API"""
    route_plan: List[Dict]
    in_amount: int
    out_amount: int
    price_impact_pct: float
    market_infos: List[Dict]
    slippage_bps: int
    other_amount_threshold: int
    swap_mode: str
    fees: Dict[str, int]
    expiry: datetime
    route_type: str

class JupiterExecutor(BaseExecutor):
    """
    Jupiter Aggregator integration for Solana DEX trading
    Features:
    - Best price routing across all Solana DEXes
    - Auto slippage calculation
    - Priority fee optimization
    - Versioned transactions (v0)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Jupiter API Configuration
        self.jupiter_url = config.get('jupiter_url', 'https://quote-api.jup.ag/v6')
        
        # Solana Configuration
        self.rpc_url = config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')
        self.backup_rpcs = config.get('solana_backup_rpcs', [
            'https://solana-api.projectserum.com',
            'https://rpc.ankr.com/solana'
        ])
        
        # Wallet Configuration
        self.wallet_private_key = config.get('solana_private_key')
        self.wallet_keypair: Optional[Keypair] = None
        
        # Trading Parameters
        self.max_slippage_bps = int(config.get('max_slippage', 0.05) * 10000)  # 5% = 500 bps
        self.quote_validity_seconds = config.get('quote_validity', 20)
        self.priority_fee_lamports = config.get('priority_fee', 5000)  # Micro-lamports
        
        # Transaction Settings
        self.compute_unit_price = config.get('compute_unit_price', 1000)
        self.compute_unit_limit = config.get('compute_unit_limit', 200000)
        
        # Rate Limiting
        self.rate_limit = config.get('rate_limit', 10)
        self.last_request_time = 0
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance Tracking
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume_sol': Decimal('0'),
            'total_fees_sol': Decimal('0'),
            'average_slippage_bps': 0,
        }
        
        self.active_orders = {}
        
        logger.info("Jupiter Executor initialized for Solana")
        
    async def initialize(self) -> None:
        """Initialize Jupiter and Solana connection"""
        try:
            # Initialize wallet keypair
            if self.wallet_private_key:
                # Handle different key formats
                if self.wallet_private_key.startswith('['):
                    # Array format: [1,2,3,...]
                    key_bytes = bytes(json.loads(self.wallet_private_key))
                elif len(self.wallet_private_key) == 88:
                    # Base58 format
                    key_bytes = base58.b58decode(self.wallet_private_key)
                else:
                    # Hex format
                    key_bytes = bytes.fromhex(self.wallet_private_key.replace('0x', ''))
                    
                self.wallet_keypair = Keypair.from_bytes(key_bytes[:32])
                logger.info(f"Wallet initialized: {self.wallet_keypair.pubkey()}")
            
            # Create HTTP session
            headers = {
                'User-Agent': 'ClaudeDex/1.0',
                'Accept': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            # Test Jupiter API
            await self._test_jupiter_connection()
            
            # Test Solana RPC
            await self._test_solana_connection()
            
            logger.info("Jupiter Executor fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jupiter Executor: {e}")
            raise
            
    async def _test_jupiter_connection(self) -> bool:
        """Test Jupiter API connection"""
        try:
            async with self.session.get(f"{self.jupiter_url}/health") as response:
                if response.status == 200:
                    logger.info("Jupiter API connection OK")
                    return True
                else:
                    logger.warning(f"Jupiter API health check returned: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Jupiter connection test failed: {e}")
            return False
            
    async def _test_solana_connection(self) -> bool:
        """Test Solana RPC connection"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getHealth"
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('result') == 'ok':
                        logger.info("Solana RPC connection OK")
                        return True
                        
            logger.warning("Solana RPC health check failed")
            return False
            
        except Exception as e:
            logger.error(f"Solana connection test failed: {e}")
            return False
            
    async def _rate_limit(self) -> None:
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    @retry_async(max_retries=3, delay=1.0)
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount_lamports: int,
        slippage_bps: Optional[int] = None,
        route_type: JupiterRoute = JupiterRoute.BEST_PRICE
    ) -> Optional[JupiterQuote]:
        """Get quote from Jupiter API"""
        try:
            await self._rate_limit()
            
            # Prepare parameters
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount_lamports),
                'slippageBps': slippage_bps or self.max_slippage_bps,
                'onlyDirectRoutes': False,
                'asLegacyTransaction': False  # Use versioned transactions
            }
            
            # Get quote
            async with self.session.get(
                f"{self.jupiter_url}/quote",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse quote
                    quote = JupiterQuote(
                        route_plan=data['routePlan'],
                        in_amount=int(data['inAmount']),
                        out_amount=int(data['outAmount']),
                        price_impact_pct=float(data.get('priceImpactPct', 0)),
                        market_infos=data.get('marketInfos', []),
                        slippage_bps=int(data.get('slippageBps', slippage_bps or self.max_slippage_bps)),
                        other_amount_threshold=int(data['otherAmountThreshold']),
                        swap_mode=data.get('swapMode', 'ExactIn'),
                        fees=data.get('platformFee', {}),
                        expiry=datetime.now() + timedelta(seconds=self.quote_validity_seconds),
                        route_type=data.get('routePlan', [{}])[0].get('swapInfo', {}).get('ammKey', 'unknown')
                    )
                    
                    logger.info(
                        f"Got Jupiter quote: {quote.in_amount / 1e9:.6f} SOL -> "
                        f"{quote.out_amount} tokens (impact: {quote.price_impact_pct:.2f}%)"
                    )
                    return quote
                    
                elif response.status == 400:
                    error_data = await response.json()
                    logger.error(f"Jupiter quote error: {error_data.get('error', 'Unknown error')}")
                    return None
                else:
                    logger.error(f"Jupiter quote failed with status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
            
    @measure_time
    async def execute_trade(
        self,
        order: Order,
        quote: Optional[JupiterQuote] = None
    ) -> Dict[str, Any]:
        """Execute trade through Jupiter"""
        try:
            # Validate order
            if not self.validate_order(order):
                return {
                    'success': False,
                    'error': 'Order validation failed'
                }
                
            # Get quote if not provided
            if not quote:
                # Convert order amount to lamports
                amount_lamports = int(float(order.amount) * 1e9)  # Assuming SOL input
                
                quote = await self.get_quote(
                    input_mint=order.token_in,
                    output_mint=order.token_out,
                    amount_lamports=amount_lamports,
                    slippage_bps=int(float(order.slippage or self.max_slippage_bps / 10000) * 10000)
                )
                
            if not quote:
                return {
                    'success': False,
                    'error': 'Failed to get quote'
                }
                
            # Check quote validity
            if datetime.now() > quote.expiry:
                return {
                    'success': False,
                    'error': 'Quote expired'
                }
                
            # Check price impact
            if quote.price_impact_pct > 10.0:  # 10% price impact threshold
                logger.warning(f"High price impact: {quote.price_impact_pct:.2f}%")
                if not order.force_execution:
                    return {
                        'success': False,
                        'error': f'Price impact too high: {quote.price_impact_pct:.2f}%'
                    }
                    
            # Build swap transaction
            swap_transaction = await self._build_swap_transaction(quote, order)
            
            if not swap_transaction:
                return {
                    'success': False,
                    'error': 'Failed to build swap transaction'
                }
                
            # Sign and send transaction
            result = await self._send_transaction(swap_transaction)
            
            # Update statistics
            if result['success']:
                self.execution_stats['successful_trades'] += 1
                self.execution_stats['total_volume_sol'] += Decimal(str(quote.in_amount / 1e9))
                
                # Calculate fees
                tx_fee = Decimal(str(result.get('fee', 5000) / 1e9))
                self.execution_stats['total_fees_sol'] += tx_fee
                
            else:
                self.execution_stats['failed_trades'] += 1
                
            self.execution_stats['total_trades'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Jupiter trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _build_swap_transaction(
        self,
        quote: JupiterQuote,
        order: Order
    ) -> Optional[str]:
        """Build swap transaction from Jupiter quote"""
        try:
            # Prepare swap request
            swap_request = {
                'quoteResponse': {
                    'inputMint': order.token_in,
                    'outputMint': order.token_out,
                    'inAmount': str(quote.in_amount),
                    'outAmount': str(quote.out_amount),
                    'otherAmountThreshold': str(quote.other_amount_threshold),
                    'swapMode': quote.swap_mode,
                    'slippageBps': quote.slippage_bps,
                    'priceImpactPct': str(quote.price_impact_pct),
                    'routePlan': quote.route_plan,
                    'platformFee': quote.fees
                },
                'userPublicKey': str(self.wallet_keypair.pubkey()),
                'wrapAndUnwrapSol': True,
                'useSharedAccounts': True,
                'prioritizationFeeLamports': self.priority_fee_lamports,
                'asLegacyTransaction': False,
                'dynamicComputeUnitLimit': True
            }
            
            # Request swap transaction
            async with self.session.post(
                f"{self.jupiter_url}/swap",
                json=swap_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Return serialized transaction
                    return data['swapTransaction']
                else:
                    error_data = await response.json()
                    logger.error(f"Failed to build swap transaction: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error building swap transaction: {e}")
            return None
            
    async def _send_transaction(
        self,
        serialized_transaction: str
    ) -> Dict[str, Any]:
        """Sign and send transaction to Solana"""
        try:
            # Deserialize transaction
            tx_bytes = base64.b64decode(serialized_transaction)
            versioned_tx = VersionedTransaction.from_bytes(tx_bytes)
            
            # Sign transaction
            signed_tx = VersionedTransaction(
                versioned_tx.message,
                [self.wallet_keypair.sign_message(bytes(versioned_tx.message))]
            )
            
            # Serialize signed transaction
            signed_tx_b64 = base64.b64encode(bytes(signed_tx)).decode('utf-8')
            
            # Send transaction
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    signed_tx_b64,
                    {
                        "encoding": "base64",
                        "skipPreflight": False,
                        "preflightCommitment": "confirmed",
                        "maxRetries": 3
                    }
                ]
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'result' in data:
                        tx_signature = data['result']
                        logger.info(f"Transaction sent: {tx_signature}")
                        
                        # Wait for confirmation
                        confirmed = await self._wait_for_confirmation(tx_signature)
                        
                        if confirmed:
                            return {
                                'success': True,
                                'signature': tx_signature,
                                'explorer_url': f"https://solscan.io/tx/{tx_signature}"
                            }
                        else:
                            return {
                                'success': False,
                                'error': 'Transaction not confirmed',
                                'signature': tx_signature
                            }
                    else:
                        error = data.get('error', {})
                        return {
                            'success': False,
                            'error': error.get('message', 'Unknown RPC error')
                        }
                else:
                    return {
                        'success': False,
                        'error': f'RPC request failed: {response.status}'
                    }
                    
        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _wait_for_confirmation(
        self,
        signature: str,
        timeout: int = 60
    ) -> bool:
        """Wait for transaction confirmation"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[signature]]
                }
                
                async with self.session.post(self.rpc_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        statuses = data.get('result', {}).get('value', [])
                        
                        if statuses and statuses[0]:
                            status = statuses[0]
                            
                            if status.get('confirmationStatus') in ['confirmed', 'finalized']:
                                logger.info(f"Transaction confirmed: {signature}")
                                return True
                            elif status.get('err'):
                                logger.error(f"Transaction failed: {status['err']}")
                                return False
                                
                await asyncio.sleep(2)
                
            logger.warning(f"Transaction confirmation timeout: {signature}")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for confirmation: {e}")
            return False
            
    async def validate_order(self, order: Order) -> bool:
        """Validate Solana order"""
        try:
            # Check required fields
            if not order.token_in or not order.token_out:
                logger.error("Missing token mints")
                return False
                
            if order.amount <= 0:
                logger.error("Invalid amount")
                return False
                
            # Validate Solana addresses (base58, 32-44 chars)
            if not (32 <= len(order.token_in) <= 44) or not (32 <= len(order.token_out) <= 44):
                logger.error("Invalid Solana token addresses")
                return False
                
            # Check wallet initialized
            if not self.wallet_keypair:
                logger.error("Wallet not initialized")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
            
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
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
        logger.info("Jupiter Executor cleaned up")