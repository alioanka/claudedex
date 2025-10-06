"""
Order Manager for DexScreener Trading Bot
Handles complete order lifecycle from creation to settlement
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class ExecutionStrategy(Enum):
    """Execution strategies"""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price
    ICEBERG = "iceberg"  # Hidden size
    SNIPER = "sniper"  # MEV-protected

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    token_address: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal]  # For limit orders
    stop_price: Optional[Decimal]  # For stop orders
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    
    # Execution details
    execution_strategy: ExecutionStrategy
    slippage_tolerance: float
    gas_price: Decimal
    gas_limit: int
    deadline: Optional[datetime]
    
    # Risk management
    take_profit: Optional[List[Decimal]]
    stop_loss: Optional[Decimal]
    trailing_stop_distance: Optional[float]
    trailing_stop_activated: bool = False
    
    # Tracking
    filled_amount: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    gas_used: int = 0
    
    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Transaction details
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    confirmations: int = 0

# Add this AFTER the Order class definition:
def build_order(
    token_address: str,
    side: OrderSide,
    amount: Decimal,
    order_type: OrderType = OrderType.MARKET,
    **kwargs
) -> Order:
    """
    Helper to build Order object from parameters
    
    Args:
        token_address: Token address
        side: Buy/Sell
        amount: Order amount
        order_type: Order type
        **kwargs: Additional parameters
        
    Returns:
        Order object
    """
    return Order(
        order_id=str(uuid.uuid4()),
        token_address=token_address,
        side=side,
        order_type=order_type,
        amount=amount,
        price=kwargs.get('price'),
        stop_price=kwargs.get('stop_price'),
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        execution_strategy=kwargs.get('execution_strategy', ExecutionStrategy.IMMEDIATE),
        slippage_tolerance=kwargs.get('slippage_tolerance', 0.005),
        gas_price=kwargs.get('gas_price', Decimal('0')),
        gas_limit=kwargs.get('gas_limit', 300000),
        deadline=kwargs.get('deadline'),
        take_profit=kwargs.get('take_profit'),
        stop_loss=kwargs.get('stop_loss'),
        trailing_stop_distance=kwargs.get('trailing_stop_distance'),
        strategy_id=kwargs.get('strategy_id'),
        signal_id=kwargs.get('signal_id'),
        parent_order_id=kwargs.get('parent_order_id'),
        metadata=kwargs.get('metadata', {})
    )


@dataclass
class OrderBook:
    """Local order book representation"""
    token_address: str
    timestamp: datetime
    bids: List[Dict[str, float]]  # [{"price": x, "size": y}, ...]
    asks: List[Dict[str, float]]
    last_trade_price: Decimal
    last_trade_size: Decimal
    spread: float
    mid_price: Decimal

@dataclass
class Fill:
    """Order fill information"""
    fill_id: str
    order_id: str
    timestamp: datetime
    price: Decimal
    amount: Decimal
    fee: Decimal
    tx_hash: str
    block_number: int

class OrderManager:
    """
    Manages order lifecycle from creation to settlement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize order manager"""
        self.config = config or self._default_config()
        self.orders: Dict[str, Order] = {}
        self.active_orders: Set[str] = set()
        self.order_history: List[Order] = []
        self.fills: Dict[str, List[Fill]] = defaultdict(list)
        self.order_books: Dict[str, OrderBook] = {}
        
        # Performance tracking
        self.metrics = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "average_slippage": 0.0,
            "total_fees": Decimal("0"),
            "total_gas": 0
        }
        
        # Initialize components
        self._initialize_components()
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # Order defaults
            "default_slippage": 0.005,  # 0.5%
            "default_gas_multiplier": 1.1,
            "max_gas_price": 500,  # Gwei
            "order_timeout": 300,  # 5 minutes
            "confirmation_blocks": 2,
            
            # Execution parameters
            "max_retries": 3,
            "retry_delay": 5,  # seconds
            "partial_fill_threshold": 0.95,  # Accept 95% fills
            
            # Risk limits
            "max_order_value": 10000,  # USD
            "max_orders_per_token": 3,
            "max_pending_orders": 10,
            
            # TWAP/VWAP settings
            "twap_intervals": 5,
            "vwap_buckets": 10,
            "iceberg_show_ratio": 0.2,  # Show 20% of order
            
            # MEV Protection
            "use_flashbots": True,
            "private_mempool": True,
            "bundle_timeout": 25,  # seconds
            
            # Order book settings
            "orderbook_depth": 20,
            "orderbook_update_interval": 1,  # seconds
            
            # Monitoring
            "health_check_interval": 30,
            "cleanup_interval": 300  # 5 minutes
        }
    
    def _initialize_components(self):
        """Initialize order management components"""
        self.execution_engine = ExecutionEngine(self.config)
        self.risk_monitor = OrderRiskMonitor(self.config)
        self.settlement_processor = SettlementProcessor(self.config)
        
        # Start background tasks
        asyncio.create_task(self._monitor_orders())
        asyncio.create_task(self._cleanup_expired_orders())


    # Fixes for trading/orders module signature mismatches

    # ============================================
    # FIX 1: OrderManager.create_order signature
    # ============================================
    # Expected: async def create_order(order: Order) -> str
    # Current: async def create_order(token_address, side, amount, ...) -> Order

    # Add this wrapper method to OrderManager class:

    async def create_order(self, order: Order) -> str:
        """
        Create order from Order object (API-compliant signature)
        
        Args:
            order: Pre-built Order object
            
        Returns:
            Order ID string
        """
        # Store the order directly
        self.orders[order.order_id] = order
        self.active_orders.add(order.order_id)
        self.metrics["total_orders"] += 1
        
        # Log order creation
        logger.info(f"Created order {order.order_id}")
        
        return order.order_id

    # ============================================
    # FIX 2: OrderManager.execute_order (missing)
    # ============================================
    # Add this method to OrderManager:

    async def execute_order(self, order_id: str) -> bool:
        """
        Execute an order (API-compliant signature)
        
        Args:
            order_id: Order ID to execute
            
        Returns:
            True if execution started successfully
        """
        try:
            result = await self.submit_order(order_id)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Error executing order {order_id}: {e}")
            return False

    async def create_order_from_params(
        self,
        token_address: str,
        side: OrderSide,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.IMMEDIATE,
        **kwargs
    ) -> Order:
        """
        Create a new order
        
        Args:
            token_address: Token contract address
            side: Buy or sell
            amount: Order amount
            order_type: Type of order
            price: Limit price (for limit orders)
            execution_strategy: How to execute the order
            **kwargs: Additional order parameters
            
        Returns:
            Created Order object
        """
        try:
            # Validate order parameters
            await self._validate_order_params(
                token_address, side, amount, order_type, price
            )
            
            # Check risk limits
            if not await self.risk_monitor.check_order_risk(
                token_address, side, amount, self.orders
            ):
                raise ValueError("Order exceeds risk limits")
            
            # Create order object
            order = Order(
                order_id=str(uuid.uuid4()),
                token_address=token_address,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                stop_price=kwargs.get("stop_price"),
                status=OrderStatus.PENDING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                execution_strategy=execution_strategy,
                slippage_tolerance=kwargs.get("slippage_tolerance", self.config["default_slippage"]),
                gas_price=Decimal(str(kwargs.get("gas_price", 0))),
                gas_limit=kwargs.get("gas_limit", 300000),
                deadline=kwargs.get("deadline"),
                take_profit=kwargs.get("take_profit"),
                stop_loss=kwargs.get("stop_loss"),
                trailing_stop_distance=kwargs.get("trailing_stop_distance"),
                strategy_id=kwargs.get("strategy_id"),
                signal_id=kwargs.get("signal_id"),
                parent_order_id=kwargs.get("parent_order_id"),
                metadata=kwargs.get("metadata", {})
            )
            
            # Store order
            self.orders[order.order_id] = order
            self.active_orders.add(order.order_id)
            self.metrics["total_orders"] += 1
            
            # Log order creation
            logger.info(f"Created order {order.order_id}: {side.value} {amount} of {token_address}")
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    async def submit_order(self, order_id: str) -> Dict[str, Any]:
        """
        Submit order for execution
        
        Args:
            order_id: Order ID to submit
            
        Returns:
            Submission result
        """
        try:
            order = self.orders.get(order_id)
            if not order:
                raise ValueError(f"Order {order_id} not found")
            
            if order.status != OrderStatus.PENDING:
                raise ValueError(f"Order {order_id} is not pending")
            
            # Update order book before execution
            await self._update_order_book(order.token_address)
            
            # Choose execution path based on strategy
            if order.execution_strategy == ExecutionStrategy.IMMEDIATE:
                result = await self._execute_immediate(order)
            elif order.execution_strategy == ExecutionStrategy.TWAP:
                result = await self._execute_twap(order)
            elif order.execution_strategy == ExecutionStrategy.VWAP:
                result = await self._execute_vwap(order)
            elif order.execution_strategy == ExecutionStrategy.ICEBERG:
                result = await self._execute_iceberg(order)
            elif order.execution_strategy == ExecutionStrategy.SNIPER:
                result = await self._execute_sniper(order)
            else:
                raise ValueError(f"Unknown execution strategy: {order.execution_strategy}")
            
            # Update order status
            if result["success"]:
                order.status = OrderStatus.SUBMITTED
                order.tx_hash = result.get("tx_hash")
                order.updated_at = datetime.utcnow()
                
                # Start monitoring for confirmation
                asyncio.create_task(self._monitor_order_confirmation(order))
            else:
                order.status = OrderStatus.FAILED
                order.metadata["failure_reason"] = result.get("error")
                self.metrics["failed_orders"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting order {order_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Success boolean
        """
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                return False
            
            # Attempt to cancel on-chain if submitted
            if order.status == OrderStatus.SUBMITTED and order.tx_hash:
                cancel_result = await self.execution_engine.cancel_transaction(
                    order.tx_hash,
                    order.gas_price * Decimal("1.2")  # Higher gas to ensure cancellation
                )
                
                if not cancel_result:
                    logger.warning(f"Failed to cancel on-chain transaction for order {order_id}")
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.utcnow()
            
            # Remove from active orders
            self.active_orders.discard(order_id)
            
            # Cancel child orders if any
            for child_id in order.child_order_ids:
                await self.cancel_order(child_id)
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def modify_order(
        self,
        order_id: str,
        modifications: Dict[str, Any]
    ) -> bool:
        """
        Modify a pending order
        
        Args:
            order_id: Order ID to modify
            modifications: Dictionary of modifications
            
        Returns:
            Success boolean
        """
        try:
            order = self.orders.get(order_id)
            if not order:
                return False
            
            if order.status != OrderStatus.PENDING:
                logger.warning(f"Cannot modify non-pending order {order_id}")
                return False
            
            # Apply modifications
            allowed_mods = [
                "price", "amount", "stop_price", "take_profit",
                "stop_loss", "slippage_tolerance", "deadline"
            ]
            
            for key, value in modifications.items():
                if key in allowed_mods:
                    setattr(order, key, value)
            
            order.updated_at = datetime.utcnow()
            
            logger.info(f"Modified order {order_id}: {modifications}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get detailed order status
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary
        """
        order = self.orders.get(order_id)
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "side": order.side.value,
            "amount": str(order.amount),
            "filled_amount": str(order.filled_amount),
            "average_fill_price": str(order.average_fill_price),
            "fees_paid": str(order.fees_paid),
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "tx_hash": order.tx_hash,
            "confirmations": order.confirmations,
            "fills": [
                {
                    "price": str(fill.price),
                    "amount": str(fill.amount),
                    "timestamp": fill.timestamp.isoformat()
                }
                for fill in self.fills.get(order_id, [])
            ]
        }
    
    async def _validate_order_params(
        self,
        token_address: str,
        side: OrderSide,
        amount: Decimal,
        order_type: OrderType,
        price: Optional[Decimal]
    ):
        """Validate order parameters"""
        # Check token address format
        if not token_address or len(token_address) != 42:
            raise ValueError("Invalid token address")
        
        # Check amount
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        # Check order value
        if price:
            order_value = float(amount * price)
            if order_value > self.config["max_order_value"]:
                raise ValueError(f"Order value {order_value} exceeds maximum")
        
        # Check limit price for limit orders
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not price:
            raise ValueError("Limit price required for limit orders")
        
        # Check pending orders limit
        if len(self.active_orders) >= self.config["max_pending_orders"]:
            raise ValueError("Too many pending orders")
        
        # Check per-token limit
        token_orders = sum(
            1 for o in self.orders.values()
            if o.token_address == token_address and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
        )
        if token_orders >= self.config["max_orders_per_token"]:
            raise ValueError(f"Too many orders for token {token_address}")
    
    async def _execute_immediate(self, order: Order) -> Dict[str, Any]:
        """Execute order immediately"""
        try:
            # Prepare transaction
            tx_params = await self._prepare_transaction(order)
            
            # Use MEV protection if configured
            if self.config["use_flashbots"] and order.metadata.get("mev_protect", True):
                result = await self.execution_engine.submit_flashbots_bundle(
                    tx_params,
                    order.gas_price
                )
            else:
                result = await self.execution_engine.submit_transaction(tx_params)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing immediate order: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_twap(self, order: Order) -> Dict[str, Any]:
        """Execute order using TWAP strategy"""
        try:
            intervals = self.config["twap_intervals"]
            interval_amount = order.amount / intervals
            interval_duration = (
                (order.deadline - datetime.utcnow()).total_seconds() / intervals
                if order.deadline
                else 60  # Default 1 minute intervals
            )
            
            # Create child orders for each interval
            child_orders = []
            for i in range(intervals):
                child = await self.create_order(
                    token_address=order.token_address,
                    side=order.side,
                    amount=interval_amount,
                    order_type=OrderType.MARKET,
                    execution_strategy=ExecutionStrategy.IMMEDIATE,
                    parent_order_id=order.order_id,
                    metadata={"twap_interval": i}
                )
                child_orders.append(child)
                order.child_order_ids.append(child.order_id)
            
            # Schedule execution
            asyncio.create_task(
                self._execute_twap_intervals(child_orders, interval_duration)
            )
            
            return {"success": True, "child_orders": len(child_orders)}
            
        except Exception as e:
            logger.error(f"Error executing TWAP: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_twap_intervals(
        self,
        child_orders: List[Order],
        interval_duration: float
    ):
        """Execute TWAP child orders at intervals"""
        for child in child_orders:
            try:
                await self.submit_order(child.order_id)
                await asyncio.sleep(interval_duration)
            except Exception as e:
                logger.error(f"Error executing TWAP interval: {e}")
    
    async def _execute_vwap(self, order: Order) -> Dict[str, Any]:
        """Execute order using VWAP strategy"""
        try:
            # Analyze volume profile
            volume_profile = await self._get_volume_profile(order.token_address)
            
            # Create weighted child orders
            child_orders = []
            remaining_amount = order.amount
            
            for bucket in volume_profile:
                bucket_amount = order.amount * Decimal(str(bucket["weight"]))
                if bucket_amount > remaining_amount:
                    bucket_amount = remaining_amount
                
                child = await self.create_order(
                    token_address=order.token_address,
                    side=order.side,
                    amount=bucket_amount,
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(bucket["target_price"])),
                    execution_strategy=ExecutionStrategy.IMMEDIATE,
                    parent_order_id=order.order_id,
                    metadata={"vwap_bucket": bucket["index"]}
                )
                child_orders.append(child)
                order.child_order_ids.append(child.order_id)
                remaining_amount -= bucket_amount
                
                if remaining_amount <= 0:
                    break
            
            # Execute child orders
            for child in child_orders:
                asyncio.create_task(self.submit_order(child.order_id))
            
            return {"success": True, "child_orders": len(child_orders)}
            
        except Exception as e:
            logger.error(f"Error executing VWAP: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_iceberg(self, order: Order) -> Dict[str, Any]:
        """Execute iceberg order (hidden size)"""
        try:
            show_amount = order.amount * Decimal(str(self.config["iceberg_show_ratio"]))
            hidden_amount = order.amount - show_amount
            
            # Create visible order
            visible_order = await self.create_order(
                token_address=order.token_address,
                side=order.side,
                amount=show_amount,
                order_type=order.order_type,
                price=order.price,
                execution_strategy=ExecutionStrategy.IMMEDIATE,
                parent_order_id=order.order_id,
                metadata={"iceberg_visible": True}
            )
            
            order.child_order_ids.append(visible_order.order_id)
            
            # Submit visible portion
            result = await self.submit_order(visible_order.order_id)
            
            # Monitor and refill
            if result["success"]:
                asyncio.create_task(
                    self._monitor_iceberg_refill(order, visible_order, hidden_amount)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing iceberg order: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_sniper(self, order: Order) -> Dict[str, Any]:
        """Execute order with MEV protection"""
        try:
            # Prepare private transaction
            tx_params = await self._prepare_transaction(order)
            
            # Add MEV protection parameters
            tx_params["max_priority_fee"] = str(order.gas_price * Decimal("1.5"))
            tx_params["private"] = True
            
            # Submit to private mempool
            if self.config["private_mempool"]:
                result = await self.execution_engine.submit_private_transaction(
                    tx_params,
                    order.gas_price
                )
            else:
                # Fallback to flashbots
                result = await self.execution_engine.submit_flashbots_bundle(
                    tx_params,
                    order.gas_price
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing sniper order: {e}")
            return {"success": False, "error": str(e)}
    
    async def _prepare_transaction(self, order: Order) -> Dict[str, Any]:
        """Prepare transaction parameters"""
        # Get current market data
        market_data = await self._get_market_data(order.token_address)
        
        # Calculate execution price with slippage
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                exec_price = market_data["ask"] * (1 + order.slippage_tolerance)
            else:
                exec_price = market_data["bid"] * (1 - order.slippage_tolerance)
        else:
            exec_price = order.price
        
        # Build transaction
        tx_params = {
            "token_address": order.token_address,
            "side": order.side.value,
            "amount": str(order.amount),
            "price": str(exec_price),
            "slippage": order.slippage_tolerance,
            "gas_price": str(order.gas_price),
            "gas_limit": order.gas_limit,
            "deadline": int(order.deadline.timestamp()) if order.deadline else None
        }
        
        return tx_params
    
    async def _monitor_order_confirmation(self, order: Order):
        """Monitor order for confirmation"""
        try:
            max_wait = self.config["order_timeout"]
            start_time = datetime.utcnow()
            
            while (datetime.utcnow() - start_time).total_seconds() < max_wait:
                # Check transaction status
                tx_status = await self.execution_engine.get_transaction_status(
                    order.tx_hash
                )
                
                if tx_status["confirmed"]:
                    order.confirmations = tx_status["confirmations"]
                    order.block_number = tx_status["block_number"]
                    
                    if order.confirmations >= self.config["confirmation_blocks"]:
                        # Process fill
                        await self._process_order_fill(order, tx_status)
                        break
                
                elif tx_status.get("failed"):
                    order.status = OrderStatus.FAILED
                    order.metadata["failure_reason"] = tx_status.get("error")
                    self.metrics["failed_orders"] += 1
                    break
                
                await asyncio.sleep(5)
            
            else:
                # Timeout reached
                order.status = OrderStatus.EXPIRED
                logger.warning(f"Order {order.order_id} expired waiting for confirmation")
            
            # Remove from active orders
            self.active_orders.discard(order.order_id)
            
        except Exception as e:
            logger.error(f"Error monitoring order confirmation: {e}")
    
    async def _process_order_fill(self, order: Order, tx_status: Dict):
        """Process confirmed order fill"""
        try:
            # Extract fill details from transaction
            fill_amount = Decimal(str(tx_status.get("amount", order.amount)))
            fill_price = Decimal(str(tx_status.get("price", order.price or 0)))
            fee = Decimal(str(tx_status.get("fee", 0)))
            
            # Create fill record
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                timestamp=datetime.utcnow(),
                price=fill_price,
                amount=fill_amount,
                fee=fee,
                tx_hash=order.tx_hash,
                block_number=order.block_number
            )
            
            self.fills[order.order_id].append(fill)
            
            # Update order
            order.filled_amount += fill_amount
            order.fees_paid += fee
            order.gas_used = tx_status.get("gas_used", 0)
            
            # Calculate average fill price
            if order.filled_amount > 0:
                total_value = sum(
                    f.price * f.amount for f in self.fills[order.order_id]
                )
                order.average_fill_price = total_value / order.filled_amount
            
            # Update status
            if order.filled_amount >= order.amount * Decimal(str(self.config["partial_fill_threshold"])):
                order.status = OrderStatus.FILLED
                self.metrics["successful_orders"] += 1
                
                # Trigger any follow-up orders (TP/SL)
                await self._trigger_follow_up_orders(order)
            else:
                order.status = OrderStatus.PARTIAL
            
            order.updated_at = datetime.utcnow()
            
            # Calculate slippage
            if order.price:
                slippage = float(abs(order.average_fill_price - order.price) / order.price)
                self._update_slippage_metric(slippage)
            
            # Update fee metrics
            self.metrics["total_fees"] += fee
            self.metrics["total_gas"] += order.gas_used
            
            logger.info(
                f"Order {order.order_id} filled: "
                f"{fill_amount} @ {fill_price} (fee: {fee})"
            )
            
        except Exception as e:
            logger.error(f"Error processing order fill: {e}")
    
    async def _trigger_follow_up_orders(self, order: Order):
        """Trigger take profit and stop loss orders"""
        try:
            # Create take profit orders
            if order.take_profit:
                for i, tp_price in enumerate(order.take_profit):
                    tp_amount = order.filled_amount / len(order.take_profit)
                    
                    tp_order = await self.create_order(
                        token_address=order.token_address,
                        side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                        amount=tp_amount,
                        order_type=OrderType.LIMIT,
                        price=tp_price,
                        execution_strategy=ExecutionStrategy.IMMEDIATE,
                        parent_order_id=order.order_id,
                        metadata={"take_profit_level": i + 1}
                    )
                    
                    asyncio.create_task(self.submit_order(tp_order.order_id))
            
            # Create stop loss order
            if order.stop_loss:
                sl_order = await self.create_order(
                    token_address=order.token_address,
                    side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                    amount=order.filled_amount,
                    order_type=OrderType.STOP,
                    stop_price=order.stop_loss,
                    execution_strategy=ExecutionStrategy.SNIPER,  # Fast execution for stops
                    parent_order_id=order.order_id,
                    metadata={"stop_loss": True}
                )
                
                asyncio.create_task(self._monitor_stop_order(sl_order))
            
            # Setup trailing stop if configured
            if order.trailing_stop_distance:
                asyncio.create_task(self._monitor_trailing_stop(order))
            
        except Exception as e:
            logger.error(f"Error triggering follow-up orders: {e}")
    
    async def _monitor_stop_order(self, stop_order: Order):
        """Monitor stop order for trigger"""
        try:
            while stop_order.order_id in self.active_orders:
                # Get current price
                market_data = await self._get_market_data(stop_order.token_address)
                current_price = Decimal(str(market_data["price"]))
                
                # Check if stop triggered
                triggered = False
                if stop_order.side == OrderSide.SELL and current_price <= stop_order.stop_price:
                    triggered = True
                elif stop_order.side == OrderSide.BUY and current_price >= stop_order.stop_price:
                    triggered = True
                
                if triggered:
                    # Convert to market order and execute
                    stop_order.order_type = OrderType.MARKET
                    await self.submit_order(stop_order.order_id)
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error monitoring stop order: {e}")
    
    async def _monitor_trailing_stop(self, order: Order):
        """Monitor and adjust trailing stop"""
        try:
            best_price = order.average_fill_price
            trail_distance = Decimal(str(order.trailing_stop_distance))
            
            while order.order_id in self.active_orders:
                # Get current price
                market_data = await self._get_market_data(order.token_address)
                current_price = Decimal(str(market_data["price"]))
                
                # Update best price and stop loss
                if order.side == OrderSide.BUY:
                    if current_price > best_price:
                        best_price = current_price
                        new_stop = best_price * (Decimal("1") - trail_distance)
                        
                        if not order.stop_loss or new_stop > order.stop_loss:
                            order.stop_loss = new_stop
                            order.trailing_stop_activated = True
                            logger.info(f"Trailing stop updated for {order.order_id}: {new_stop}")
                
                else:  # SELL order
                    if current_price < best_price:
                        best_price = current_price
                        new_stop = best_price * (Decimal("1") + trail_distance)
                        
                        if not order.stop_loss or new_stop < order.stop_loss:
                            order.stop_loss = new_stop
                            order.trailing_stop_activated = True
                            logger.info(f"Trailing stop updated for {order.order_id}: {new_stop}")
                
                # Check if stop hit
                if order.stop_loss:
                    if (order.side == OrderSide.BUY and current_price <= order.stop_loss) or \
                       (order.side == OrderSide.SELL and current_price >= order.stop_loss):
                        # Trigger stop loss
                        await self._execute_stop_loss(order)
                        break
                
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error monitoring trailing stop: {e}")
    
    async def _execute_stop_loss(self, order: Order):
        """Execute stop loss order"""
        try:
            sl_order = await self.create_order(
                token_address=order.token_address,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                amount=order.filled_amount,
                order_type=OrderType.MARKET,
                execution_strategy=ExecutionStrategy.SNIPER,
                parent_order_id=order.order_id,
                metadata={"stop_loss_triggered": True}
            )
            
            await self.submit_order(sl_order.order_id)
            
        except Exception as e:
            logger.error(f"Error executing stop loss: {e}")
    
    async def _update_order_book(self, token_address: str):
        """Update local order book cache"""
        try:
            # Fetch latest order book data
            ob_data = await self.execution_engine.get_order_book(token_address)
            
            self.order_books[token_address] = OrderBook(
                token_address=token_address,
                timestamp=datetime.utcnow(),
                bids=ob_data["bids"],
                asks=ob_data["asks"],
                last_trade_price=Decimal(str(ob_data.get("last_price", 0))),
                last_trade_size=Decimal(str(ob_data.get("last_size", 0))),
                spread=ob_data.get("spread", 0),
                mid_price=Decimal(str(ob_data.get("mid_price", 0)))
            )
            
        except Exception as e:
            logger.error(f"Error updating order book: {e}")
    
    async def _get_market_data(self, token_address: str) -> Dict:
        """Get current market data for token"""
        # Fetch from order book cache or external source
        if token_address in self.order_books:
            ob = self.order_books[token_address]
            return {
                "price": float(ob.mid_price),
                "bid": float(ob.bids[0]["price"]) if ob.bids else 0,
                "ask": float(ob.asks[0]["price"]) if ob.asks else 0,
                "spread": ob.spread
            }
        
        # Fallback to fetching fresh data
        await self._update_order_book(token_address)
        return await self._get_market_data(token_address)
    
    async def _get_volume_profile(self, token_address: str) -> List[Dict]:
        """Get volume profile for VWAP execution"""
        # This would analyze historical volume patterns
        # For now, return a simple distribution
        buckets = []
        for i in range(self.config["vwap_buckets"]):
            buckets.append({
                "index": i,
                "weight": 1.0 / self.config["vwap_buckets"],
                "target_price": 1.0  # Would be calculated from volume analysis
            })
        return buckets
    
    async def _monitor_iceberg_refill(
        self,
        parent_order: Order,
        visible_order: Order,
        hidden_amount: Decimal
    ):
        """Monitor and refill iceberg order"""
        try:
            remaining_hidden = hidden_amount
            
            while remaining_hidden > 0 and parent_order.order_id in self.active_orders:
                # Wait for visible order to fill
                visible_status = await self.get_order_status(visible_order.order_id)
                
                if visible_status["status"] == "filled":
                    # Refill with next chunk
                    refill_amount = min(
                        remaining_hidden,
                        parent_order.amount * Decimal(str(self.config["iceberg_show_ratio"]))
                    )
                    
                    new_visible = await self.create_order(
                        token_address=parent_order.token_address,
                        side=parent_order.side,
                        amount=refill_amount,
                        order_type=parent_order.order_type,
                        price=parent_order.price,
                        execution_strategy=ExecutionStrategy.IMMEDIATE,
                        parent_order_id=parent_order.order_id,
                        metadata={"iceberg_refill": True}
                    )
                    
                    parent_order.child_order_ids.append(new_visible.order_id)
                    await self.submit_order(new_visible.order_id)
                    
                    remaining_hidden -= refill_amount
                    visible_order = new_visible
                
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error monitoring iceberg refill: {e}")
    
    def _update_slippage_metric(self, slippage: float):
        """Update average slippage metric"""
        if self.metrics["successful_orders"] > 0:
            # Running average
            prev_avg = self.metrics["average_slippage"]
            n = self.metrics["successful_orders"]
            self.metrics["average_slippage"] = (prev_avg * (n - 1) + slippage) / n
        else:
            self.metrics["average_slippage"] = slippage
    
    async def _monitor_orders(self):
        """Background task to monitor all active orders"""
        while True:
            try:
                for order_id in list(self.active_orders):
                    order = self.orders.get(order_id)
                    if not order:
                        continue
                    
                    # Check for expired orders
                    if order.deadline and datetime.utcnow() > order.deadline:
                        order.status = OrderStatus.EXPIRED
                        self.active_orders.discard(order_id)
                        logger.info(f"Order {order_id} expired")
                    
                    # Update order status from chain
                    if order.tx_hash and order.status == OrderStatus.SUBMITTED:
                        tx_status = await self.execution_engine.get_transaction_status(
                            order.tx_hash
                        )
                        if tx_status.get("confirmed"):
                            order.confirmations = tx_status["confirmations"]
                
                await asyncio.sleep(self.config.get("health_check_interval", 30))
                
            except Exception as e:
                logger.error(f"Error in order monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_expired_orders(self):
        """Background task to cleanup old orders"""
        while True:
            try:
                await asyncio.sleep(self.config.get("cleanup_interval", 300))
                
                current_time = datetime.utcnow()
                expired_count = 0
                
                for order_id in list(self.orders.keys()):
                    order = self.orders[order_id]
                    
                    # Remove old completed orders from memory
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
                        age = (current_time - order.updated_at).total_seconds()
                        if age > 3600:  # 1 hour old
                            self.order_history.append(order)
                            del self.orders[order_id]
                            self.active_orders.discard(order_id)
                            expired_count += 1
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} old orders")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def get_metrics(self) -> Dict:
        """Get order manager metrics"""
        return {
            **self.metrics,
            "active_orders": len(self.active_orders),
            "total_orders_in_memory": len(self.orders),
            "order_history_size": len(self.order_history)
        }
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active orders"""
        active = []
        for order_id in self.active_orders:
            order = self.orders.get(order_id)
            if order:
                active.append({
                    "order_id": order.order_id,
                    "token": order.token_address,
                    "side": order.side.value,
                    "amount": str(order.amount),
                    "status": order.status.value,
                    "created_at": order.created_at.isoformat()
                })
        return active

    # PATCHES FOR order_manager.py
    # Add these methods to the OrderManager class:

    async def execute_immediate(self, order: Order) -> Dict:
        """
        Public interface for immediate order execution
        
        Args:
            order: Order to execute immediately
            
        Returns:
            Execution result dictionary
        """
        return await self._execute_immediate(order)

    async def execute_twap(self, order: Order, duration: int, slices: int) -> Dict:
        """
        Execute order using TWAP strategy
        
        Args:
            order: Order to execute
            duration: Total duration in seconds
            slices: Number of time slices
            
        Returns:
            Execution result with child orders
        """
        # Store original parameters
        order.metadata["twap_duration"] = duration
        order.metadata["twap_slices"] = slices
        
        # Update config temporarily if needed
        original_intervals = self.config.get("twap_intervals", 5)
        self.config["twap_intervals"] = slices
        
        try:
            result = await self._execute_twap(order)
            return result
        finally:
            # Restore config
            self.config["twap_intervals"] = original_intervals

    async def execute_sniper(self, order: Order, trigger_price: Decimal) -> Dict:
        """
        Execute sniper order with MEV protection
        
        Args:
            order: Order to execute
            trigger_price: Price trigger for execution
            
        Returns:
            Execution result
        """
        # Set trigger price
        order.metadata["trigger_price"] = str(trigger_price)
        
        # Monitor for trigger
        asyncio.create_task(self._monitor_sniper_trigger(order, trigger_price))
        
        return {
            "status": "monitoring",
            "order_id": order.order_id,
            "trigger_price": str(trigger_price)
        }

    async def _monitor_sniper_trigger(self, order: Order, trigger_price: Decimal):
        """Monitor price for sniper execution"""
        try:
            while order.order_id in self.active_orders:
                # Get current market price
                market_data = await self._get_market_data(order.token_address)
                current_price = Decimal(str(market_data["price"]))
                
                # Check trigger based on order side
                triggered = False
                if order.side == OrderSide.BUY and current_price <= trigger_price:
                    triggered = True
                elif order.side == OrderSide.SELL and current_price >= trigger_price:
                    triggered = True
                
                if triggered:
                    # Execute with MEV protection
                    result = await self._execute_sniper(order)
                    logger.info(f"Sniper triggered for {order.order_id} at {current_price}")
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error monitoring sniper trigger: {e}")

    async def get_open_orders(self) -> List[Order]:
        """
        Get all open orders
        
        Returns:
            List of open Order objects
        """
        open_orders = []
        
        for order_id in self.active_orders:
            order = self.orders.get(order_id)
            if order and order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
                open_orders.append(order)
        
        return open_orders


class ExecutionEngine:
    """Handles actual transaction execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def submit_transaction(self, tx_params: Dict) -> Dict:
        """Submit transaction to blockchain"""
        # Implementation would interact with Web3
        pass
    
    async def submit_flashbots_bundle(self, tx_params: Dict, gas_price: Decimal) -> Dict:
        """Submit transaction via Flashbots"""
        # Implementation would use Flashbots API
        pass
    
    async def submit_private_transaction(self, tx_params: Dict, gas_price: Decimal) -> Dict:
        """Submit transaction to private mempool"""
        # Implementation would use private mempool services
        pass
    
    async def cancel_transaction(self, tx_hash: str, gas_price: Decimal) -> bool:
        """Cancel pending transaction"""
        # Implementation would submit cancellation transaction
        pass
    
    async def get_transaction_status(self, tx_hash: str) -> Dict:
        """Get transaction status from blockchain"""
        # Implementation would check transaction receipt
        pass
    
    async def get_order_book(self, token_address: str) -> Dict:
        """Fetch order book data"""
        # Implementation would fetch from DEX APIs
        pass


class OrderRiskMonitor:
    """Monitor and enforce order risk limits"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def check_order_risk(
        self,
        token_address: str,
        side: OrderSide,
        amount: Decimal,
        existing_orders: Dict[str, Order]
    ) -> bool:
        """Check if order passes risk checks"""
        # Implementation would check various risk metrics
        return True


class SettlementProcessor:
    """Process order settlements and reconciliation"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def process_settlement(self, order: Order, fill: Fill) -> bool:
        """Process order settlement"""
        # Implementation would handle settlement logic
        return True