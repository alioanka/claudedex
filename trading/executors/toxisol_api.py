# trading/executors/toxisol_api.py
"""
ToxiSol API Integration for Advanced Trading
Provides high-speed execution through ToxiSol's infrastructure
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from web3 import Web3
from eth_account import Account
from eth_account.datastructures import SignedTransaction

from trading.orders.order_manager import Order, OrderType, OrderStatus
from trading.executors.base_executor import BaseExecutor
from utils.helpers import retry_async, measure_time
from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class ToxiSolRoute(Enum):
    """ToxiSol routing options"""
    FAST = "fast"           # Fastest execution, higher fees
    OPTIMAL = "optimal"     # Balance speed and cost
    CHEAP = "cheap"         # Lowest cost, slower execution
    PRIVATE = "private"     # MEV protected route
    
@dataclass
class ToxiSolQuote:
    """Quote from ToxiSol API"""
    route_id: str
    token_in: str
    token_out: str
    amount_in: Decimal
    amount_out: Decimal
    price_impact: float
    gas_estimate: int
    execution_price: Decimal
    slippage: float
    route_path: List[Dict]
    expiry: datetime
    confidence: float

class ToxiSolAPIExecutor(BaseExecutor):
    """
    ToxiSol API integration for advanced DEX trading
    Features:
    - Multi-DEX aggregation
    - MEV protection
    - Smart routing
    - Gas optimization
    - Flash loan integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # API Configuration
        self.api_key = config.get('toxisol_api_key')
        self.api_secret = config.get('toxisol_api_secret')
        self.base_url = config.get('toxisol_url', 'https://api.toxisol.com/v1')
        
        # Execution parameters
        self.max_slippage = Decimal(str(config.get('max_slippage', 0.05)))
        self.gas_price_multiplier = config.get('gas_multiplier', 1.2)
        self.quote_validity_seconds = config.get('quote_validity', 30)
        self.enable_flash_loans = config.get('enable_flash_loans', False)
        
        # MEV Protection settings
        self.use_flashbots = config.get('use_flashbots', True)
        self.private_mempool = config.get('private_mempool', True)
        self.bundle_transactions = config.get('bundle_transactions', True)
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 10)  # requests per second
        self.last_request_time = 0
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.encryption = EncryptionManager(config.get('encryption', {}))
        
        # Performance tracking
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': Decimal('0'),
            'total_fees_saved': Decimal('0'),
            'average_slippage': 0,
            'mev_attacks_prevented': 0
        }

        self.active_orders = {}  # Track active orders
        
        logger.info("ToxiSol API Executor initialized")
        
    async def initialize(self) -> None:
        """Initialize ToxiSol connection"""
        try:
            # Create HTTP session with custom headers
            headers = {
                'X-API-Key': self.api_key,
                'User-Agent': 'ClaudeDex/1.0',
                'Accept': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            # Test connection
            await self._test_connection()
            
            # Subscribe to websocket for real-time updates
            if self.config.get('enable_websocket', True):
                asyncio.create_task(self._maintain_websocket())
                
            logger.info("ToxiSol API connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize ToxiSol API: {e}")
            raise
            
    async def _test_connection(self) -> bool:
        """Test API connection"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"ToxiSol API status: {data.get('status')}")
                    return True
                else:
                    logger.error(f"ToxiSol API health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
            
    def _generate_signature(self, payload: str, timestamp: int) -> str:
        """Generate HMAC signature for authenticated requests"""
        message = f"{timestamp}{payload}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
        
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
        token_in: str,
        token_out: str,
        amount: Decimal,
        chain: str = 'ethereum',
        route_type: ToxiSolRoute = ToxiSolRoute.OPTIMAL
    ) -> Optional[ToxiSolQuote]:
        """Get quote from ToxiSol API"""
        try:
            await self._rate_limit()
            
            # Prepare request
            params = {
                'tokenIn': token_in,
                'tokenOut': token_out,
                'amount': str(int(amount * 10**18)),  # Convert to wei
                'chain': chain,
                'route': route_type.value,
                'slippage': float(self.max_slippage),
                'includeGas': True,
                'includeDexes': 'all'
            }
            
            # Add MEV protection if enabled
            if self.private_mempool:
                params['protection'] = 'mev'
                
            # Request quote
            async with self.session.get(
                f"{self.base_url}/quote",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse quote
                    quote = ToxiSolQuote(
                        route_id=data['routeId'],
                        token_in=token_in,
                        token_out=token_out,
                        amount_in=Decimal(data['amountIn']) / 10**18,
                        amount_out=Decimal(data['amountOut']) / 10**18,
                        price_impact=data['priceImpact'],
                        gas_estimate=data['gasEstimate'],
                        execution_price=Decimal(data['executionPrice']),
                        slippage=data['estimatedSlippage'],
                        route_path=data['route'],
                        expiry=datetime.now() + timedelta(seconds=self.quote_validity_seconds),
                        confidence=data.get('confidence', 0.95)
                    )
                    
                    logger.info(f"Got quote: {quote.amount_in} {token_in} -> {quote.amount_out} {token_out}")
                    return quote
                    
                else:
                    logger.error(f"Quote request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None
            
    @measure_time
    async def execute_trade(
        self,
        order: Order,
        quote: Optional[ToxiSolQuote] = None
    ) -> Dict[str, Any]:
        """Execute trade through ToxiSol"""
        try:
            # Validate order
            if not self.validate_order(order):
                return {
                    'success': False,
                    'error': 'Order validation failed'
                }
                
            # Get quote if not provided
            if not quote:
                quote = await self.get_quote(
                    order.token_in,
                    order.token_out,
                    order.amount,
                    order.chain,
                    ToxiSolRoute.OPTIMAL if order.order_type == OrderType.MARKET else ToxiSolRoute.PRIVATE
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
                
            # Final safety checks
            if quote.price_impact > 0.10:  # 10% price impact
                logger.warning(f"High price impact: {quote.price_impact:.2%}")
                if not order.force_execution:
                    return {
                        'success': False,
                        'error': f'Price impact too high: {quote.price_impact:.2%}'
                    }
                    
            # Build transaction
            transaction = await self._build_toxisol_transaction(order, quote)  # Changed from _build_transaction
            
            # Sign transaction
            signed_tx = await self._sign_transaction(transaction)
            
            # Execute based on protection level
            if self.private_mempool and self.use_flashbots:
                result = await self._execute_flashbots(signed_tx)
            else:
                result = await self._execute_standard(signed_tx)
                
            # Update statistics
            if result['success']:
                self.execution_stats['successful_trades'] += 1
                self.execution_stats['total_volume'] += order.amount
                
                # Calculate fees saved
                standard_gas = quote.gas_estimate * 1.5
                actual_gas = result.get('gasUsed', quote.gas_estimate)
                gas_saved = max(0, standard_gas - actual_gas)
                self.execution_stats['total_fees_saved'] += Decimal(str(gas_saved * result.get('gasPrice', 0) / 10**18))
                
            else:
                self.execution_stats['failed_trades'] += 1
                
            self.execution_stats['total_trades'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    # In ToxiSolAPIExecutor class, rename the first occurrence to be more specific:

    async def _build_toxisol_transaction(
        self,
        order: Order,
        quote: ToxiSolQuote
    ) -> Dict[str, Any]:
        """Build transaction from ToxiSol quote (renamed from _build_transaction)"""
        try:
            # Prepare transaction request
            payload = {
                'routeId': quote.route_id,
                'walletAddress': order.wallet_address,
                'slippage': float(order.slippage or self.max_slippage),
                'deadline': int((datetime.now() + timedelta(minutes=20)).timestamp()),
                'recipient': order.recipient or order.wallet_address
            }
            
            # Add MEV protection parameters
            if self.private_mempool:
                payload['protection'] = {
                    'type': 'flashbots',
                    'maxPriorityFee': str(int(order.max_priority_fee * 10**9)) if order.max_priority_fee else 'auto',
                    'bundleTimeout': 25  # blocks
                }
                
            # Generate signature
            timestamp = int(time.time())
            signature = self._generate_signature(json.dumps(payload), timestamp)
            
            # Request transaction data
            headers = {
                'X-Timestamp': str(timestamp),
                'X-Signature': signature
            }
            
            async with self.session.post(
                f"{self.base_url}/build-transaction",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    tx_data = await response.json()
                    
                    # Add gas parameters
                    tx_data['gas'] = int(quote.gas_estimate * self.gas_price_multiplier)
                    
                    # Add current gas prices
                    gas_price = await self._get_optimal_gas_price(order.chain)
                    tx_data['gasPrice'] = gas_price
                    
                    return tx_data
                else:
                    raise Exception(f"Failed to build transaction: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise
            
    async def _execute_flashbots(
        self,
        signed_tx: SignedTransaction
    ) -> Dict[str, Any]:
        """Execute transaction through Flashbots"""
        try:
            # Bundle transaction
            bundle = {
                'txs': [signed_tx.rawTransaction.hex()],
                'blockNumber': await self._get_block_number() + 1,
                'minTimestamp': int(time.time()),
                'maxTimestamp': int(time.time()) + 120
            }
            
            # Send to Flashbots
            async with self.session.post(
                'https://relay.flashbots.net/relay/v1/bundles',
                json=bundle,
                headers={
                    'X-Flashbots-Signature': self._sign_flashbots_request(bundle)
                }
            ) as response:
                if response.status == 200:
                    bundle_hash = (await response.json())['bundleHash']
                    
                    # Wait for inclusion
                    result = await self._wait_for_bundle(bundle_hash)
                    
                    if result['included']:
                        self.execution_stats['mev_attacks_prevented'] += 1
                        
                    return result
                else:
                    logger.error(f"Flashbots submission failed: {response.status}")
                    # Fallback to standard execution
                    return await self._execute_standard(signed_tx)
                    
        except Exception as e:
            logger.error(f"Flashbots execution failed: {e}")
            return await self._execute_standard(signed_tx)
            
    async def _execute_standard(
        self,
        signed_tx: SignedTransaction
    ) -> Dict[str, Any]:
        """Execute transaction through standard mempool"""
        try:
            # Send transaction
            tx_hash = await self._send_transaction(signed_tx)
            
            # Wait for confirmation
            receipt = await self._wait_for_confirmation(tx_hash)
            
            return {
                'success': receipt['status'] == 1,
                'transactionHash': tx_hash,
                'blockNumber': receipt['blockNumber'],
                'gasUsed': receipt['gasUsed'],
                'gasPrice': receipt['effectiveGasPrice'],
                'error': None if receipt['status'] == 1 else 'Transaction reverted'
            }
            
        except Exception as e:
            logger.error(f"Standard execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_trades = self.execution_stats['total_trades']
        
        if total_trades == 0:
            return self.execution_stats
            
        return {
            **self.execution_stats,
            'success_rate': self.execution_stats['successful_trades'] / total_trades,
            'average_slippage': self.execution_stats['average_slippage'],
            'mev_protection_rate': self.execution_stats['mev_attacks_prevented'] / total_trades
        }
        
    async def estimate_gas(
        self,
        token_in: str,
        token_out: str,
        amount: Decimal,
        chain: str = 'ethereum'
    ) -> Optional[int]:
        """Estimate gas for trade"""
        quote = await self.get_quote(token_in, token_out, amount, chain)
        return quote.gas_estimate if quote else None
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
        logger.info("ToxiSol API Executor cleaned up")

    # Missing methods for trading/executors classes
    # These methods should be added to the respective executor classes

    # ============================================
    # For DirectDEXExecutor and ToxiSolAPIExecutor
    # ============================================

    async def validate_order(self, order: Order) -> bool:
        """
        Validate order before execution
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid
        """
        try:
            # Check required fields
            if not order.token_in or not order.token_out:
                logger.error("Missing token addresses")
                return False
                
            if order.amount <= 0:
                logger.error("Invalid amount")
                return False
                
            if not order.wallet_address:
                logger.error("Missing wallet address")
                return False
                
            # Check token addresses are valid
            if not Web3.isAddress(order.token_in) or not Web3.isAddress(order.token_out):
                logger.error("Invalid token addresses")
                return False
                
            # Check chain is supported
            if order.chain not in self.w3_connections:  # For DirectDEX
            # if order.chain not in ['ethereum', 'bsc', 'polygon']:  # For ToxiSol
                logger.error(f"Unsupported chain: {order.chain}")
                return False
                
            # Validate slippage
            if order.slippage and (order.slippage < 0 or order.slippage > 1):
                logger.error(f"Invalid slippage: {order.slippage}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status
        """
        try:
            # Check if we have the order in tracking
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Check transaction status on-chain
                if order.tx_hash:
                    receipt = await self._get_transaction_receipt(order.tx_hash)
                    
                    if receipt:
                        if receipt['status'] == 1:
                            return OrderStatus.COMPLETED
                        else:
                            return OrderStatus.FAILED
                    else:
                        # Still pending
                        return OrderStatus.EXECUTING
                else:
                    return order.status
            else:
                logger.warning(f"Order {order_id} not found")
                return OrderStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return OrderStatus.ERROR

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (if possible)
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.error(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # Can only cancel if not yet executed
            if order.status in [OrderStatus.PENDING, OrderStatus.QUEUED]:
                order.status = OrderStatus.CANCELLED
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                logger.info(f"Order {order_id} cancelled")
                return True
                
            elif order.status == OrderStatus.EXECUTING:
                # Try to cancel on-chain (replace with higher gas price tx doing nothing)
                # This is complex and chain-specific
                logger.warning(f"Cannot cancel executing order {order_id}")
                return False
                
            else:
                logger.warning(f"Order {order_id} in state {order.status} cannot be cancelled")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def modify_order(
        self, 
        order_id: str,
        modifications: Dict[str, Any]
    ) -> bool:
        """
        Modify an existing order (if not yet executed)
        
        Args:
            order_id: Order ID to modify
            modifications: Dictionary of fields to modify
            
        Returns:
            True if modified successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.error(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # Can only modify pending orders
            if order.status != OrderStatus.PENDING:
                logger.error(f"Cannot modify order in status {order.status}")
                return False
                
            # Apply modifications
            allowed_fields = ['amount', 'slippage', 'gas_price', 'max_priority_fee']
            
            for field, value in modifications.items():
                if field in allowed_fields:
                    setattr(order, field, value)
                    logger.info(f"Modified {field} to {value} for order {order_id}")
                else:
                    logger.warning(f"Field {field} cannot be modified")
                    
            # Re-validate order
            if not await self.validate_order(order):
                logger.error("Modified order is invalid")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False

    # ============================================
    # Helper methods that might be needed
    # ============================================

    async def _get_transaction_receipt(self, tx_hash: str) -> Optional[Dict]:
        """
        Get transaction receipt from blockchain
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Receipt dictionary or None
        """
        try:
            # For DirectDEX
            if hasattr(self, 'w3_connections'):
                for chain, w3 in self.w3_connections.items():
                    try:
                        receipt = w3.eth.get_transaction_receipt(tx_hash)
                        if receipt:
                            return dict(receipt)
                    except:
                        continue
                        
            # For MEVProtectionLayer and ToxiSol
            elif hasattr(self, 'w3'):
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                return dict(receipt) if receipt else None
                
            return None
            
        except Exception as e:
            logger.debug(f"Could not get receipt for {tx_hash}: {e}")
            return None

    async def _check_bundle_status(self, bundle_id: str) -> str:
        """
        Check Flashbots bundle status
        
        Args:
            bundle_id: Bundle hash
            
        Returns:
            Status string: 'pending', 'included', 'failed'
        """
        try:
            if not self.session:
                return 'unknown'
                
            async with self.session.get(
                f"{self.flashbots_relay}/relay/v1/bundle",
                params={'bundleHash': bundle_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('isIncluded'):
                        return 'included'
                    elif data.get('isFailed'):
                        return 'failed'
                    else:
                        return 'pending'
                        
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error checking bundle status: {e}")
            return 'unknown'

    # Also add this helper method for building base transactions:
    async def _build_base_transaction(self, order: Order) -> Dict[str, Any]:
        """
        Build base transaction from order
        
        Args:
            order: Order to build transaction from
            
        Returns:
            Transaction dictionary
        """
        return {
            'from': order.wallet_address,
            'to': order.token_out,  # This would be router address in reality
            'value': 0,
            'data': '0x',  # Would be actual swap data
            'gas': 300000,  # Estimate
            'gasPrice': self.w3.eth.gas_price if hasattr(self, 'w3') else 50 * 10**9,
            'nonce': self.w3.eth.get_transaction_count(order.wallet_address) if hasattr(self, 'w3') else 0,
            'chainId': 1  # Would get from order.chain
        }

    # Format helper function (add to direct_dex.py):
    def format_token_amount(amount: Decimal, decimals: int) -> str:
        """Format token amount for display"""
        return f"{amount:.{decimals}f}".rstrip('0').rstrip('.')