"""
Jupiter Limit Orders Strategy

Integrates with Jupiter's limit order functionality for:
- Better execution prices
- Reduced slippage
- Off-hours trading
- Automated DCA strategies
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import aiohttp
import base64
import json

from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Jupiter limit order types"""
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    DCA = "dca"  # Dollar cost averaging


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class LimitOrder:
    """Jupiter limit order"""
    id: str
    token_address: str
    order_type: OrderType
    side: str  # buy/sell
    amount: Decimal
    limit_price: Decimal
    current_price: Decimal
    status: OrderStatus
    filled_amount: Decimal
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]


class JupiterLimitOrdersStrategy(BaseStrategy):
    """
    Advanced strategy using Jupiter's limit order functionality

    Features:
    - Smart limit orders based on support/resistance levels
    - Automated stop loss and take profit orders
    - DCA (Dollar Cost Averaging) strategies
    - Range trading with limit orders
    - Market making with limit orders
    - Reduced slippage vs market orders

    Use Cases:
    1. Enter positions at better prices
    2. Exit positions with limit orders
    3. Automated DCA into positions
    4. Range trading strategies
    5. Passive market making
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Jupiter limit orders strategy"""
        super().__init__(config)

        # Jupiter API configuration
        self.jupiter_api = config.get("jupiter_api", "https://quote-api.jup.ag/v6")
        self.jupiter_limit_api = config.get(
            "jupiter_limit_api",
            "https://api.jup.ag/limit/v1"
        )

        # Strategy configuration
        self.use_limit_orders = config.get("use_limit_orders", True)
        self.limit_order_offset_pct = config.get("limit_order_offset_pct", 0.01)  # 1% better price
        self.max_order_age = config.get("max_order_age", 3600)  # 1 hour
        self.partial_fill_enabled = config.get("partial_fill_enabled", True)
        self.auto_cancel_unfilled = config.get("auto_cancel_unfilled", True)

        # DCA configuration
        self.dca_enabled = config.get("dca_enabled", False)
        self.dca_intervals = config.get("dca_intervals", [300, 600, 900])  # 5, 10, 15 min
        self.dca_amounts = config.get("dca_amounts", [0.4, 0.3, 0.3])  # 40%, 30%, 30%

        # Range trading
        self.range_trading_enabled = config.get("range_trading_enabled", False)
        self.range_levels = config.get("range_levels", 5)  # Number of price levels
        self.range_spread = config.get("range_spread", 0.02)  # 2% between levels

        # Order tracking
        self.active_orders: Dict[str, LimitOrder] = {}
        self.order_history: List[LimitOrder] = []

        # Support/Resistance levels cache
        self.support_resistance: Dict[str, Dict[str, List[Decimal]]] = {}

        logger.info("Jupiter Limit Orders Strategy initialized")

    async def analyze(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Analyze market and create limit order signal

        Args:
            market_data: Current market data

        Returns:
            Trading signal with limit order parameters
        """
        try:
            token_address = market_data.get("token_address")
            if not token_address:
                return None

            # Only for Solana tokens
            if market_data.get("chain", "").lower() != "solana":
                return None

            # Calculate support/resistance levels
            levels = await self._calculate_support_resistance(token_address, market_data)

            # Determine optimal entry price
            current_price = Decimal(str(market_data.get("price", 0)))
            optimal_entry = await self._find_optimal_entry(
                current_price,
                levels,
                market_data
            )

            if not optimal_entry:
                return None

            # Calculate confidence based on price distance to support/resistance
            confidence = await self._calculate_limit_order_confidence(
                current_price,
                optimal_entry,
                levels,
                market_data
            )

            if confidence < self.min_confidence:
                return None

            # Create limit order signal
            signal = await self._create_limit_order_signal(
                token_address,
                optimal_entry,
                current_price,
                levels,
                confidence,
                market_data
            )

            return signal

        except Exception as e:
            logger.error(f"Error analyzing for Jupiter limit orders: {e}", exc_info=True)
            return None

    async def _calculate_support_resistance(
        self,
        token_address: str,
        market_data: Dict
    ) -> Dict[str, List[Decimal]]:
        """
        Calculate support and resistance levels

        Args:
            token_address: Token address
            market_data: Market data with price history

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Check cache
            if token_address in self.support_resistance:
                cached = self.support_resistance[token_address]
                # Cache valid for 5 minutes
                if (datetime.now() - cached.get("timestamp", datetime.min)).seconds < 300:
                    return cached

            # Get price history
            price_history = market_data.get("price_history", [])
            if not price_history or len(price_history) < 20:
                # Fallback: generate levels around current price
                current_price = Decimal(str(market_data.get("price", 0)))
                return self._generate_basic_levels(current_price)

            # Find support levels (local minima)
            support_levels = self._find_local_minima(price_history)

            # Find resistance levels (local maxima)
            resistance_levels = self._find_local_maxima(price_history)

            levels = {
                "support": sorted(support_levels)[:5],  # Top 5 support levels
                "resistance": sorted(resistance_levels, reverse=True)[:5],  # Top 5 resistance
                "timestamp": datetime.now()
            }

            # Cache levels
            self.support_resistance[token_address] = levels

            logger.debug(
                f"Calculated S/R levels for {token_address[:10]}...: "
                f"Support={len(levels['support'])}, Resistance={len(levels['resistance'])}"
            )

            return levels

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {"support": [], "resistance": [], "timestamp": datetime.now()}

    def _generate_basic_levels(self, current_price: Decimal) -> Dict[str, List[Decimal]]:
        """Generate basic support/resistance levels around current price"""
        support = []
        resistance = []

        for i in range(1, 6):
            # Support levels below current price (1%, 2%, 3%, 4%, 5%)
            support.append(current_price * (Decimal("1") - Decimal(str(i * 0.01))))

            # Resistance levels above current price
            resistance.append(current_price * (Decimal("1") + Decimal(str(i * 0.01))))

        return {
            "support": support,
            "resistance": resistance,
            "timestamp": datetime.now()
        }

    def _find_local_minima(self, prices: List[float], window: int = 5) -> List[Decimal]:
        """Find local minimum price levels"""
        minima = []

        for i in range(window, len(prices) - window):
            left_window = prices[i-window:i]
            right_window = prices[i+1:i+window+1]

            if all(prices[i] <= p for p in left_window) and all(prices[i] <= p for p in right_window):
                minima.append(Decimal(str(prices[i])))

        return minima

    def _find_local_maxima(self, prices: List[float], window: int = 5) -> List[Decimal]:
        """Find local maximum price levels"""
        maxima = []

        for i in range(window, len(prices) - window):
            left_window = prices[i-window:i]
            right_window = prices[i+1:i+window+1]

            if all(prices[i] >= p for p in left_window) and all(prices[i] >= p for p in right_window):
                maxima.append(Decimal(str(prices[i])))

        return maxima

    async def _find_optimal_entry(
        self,
        current_price: Decimal,
        levels: Dict[str, List[Decimal]],
        market_data: Dict
    ) -> Optional[Decimal]:
        """
        Find optimal entry price based on support/resistance

        Args:
            current_price: Current market price
            levels: Support/resistance levels
            market_data: Market data

        Returns:
            Optimal entry price or None
        """
        support_levels = levels.get("support", [])

        if not support_levels:
            # Default: place limit order slightly below current price
            return current_price * (Decimal("1") - Decimal(str(self.limit_order_offset_pct)))

        # Find nearest support level below current price
        support_below = [s for s in support_levels if s < current_price]

        if not support_below:
            # No support found, use offset from current price
            return current_price * (Decimal("1") - Decimal(str(self.limit_order_offset_pct)))

        # Use the nearest support level
        nearest_support = max(support_below)

        # Place limit order slightly above support (better fill probability)
        optimal_entry = nearest_support * Decimal("1.002")  # 0.2% above support

        # Ensure it's below current price
        if optimal_entry >= current_price:
            optimal_entry = current_price * (Decimal("1") - Decimal(str(self.limit_order_offset_pct)))

        # Ensure minimum price improvement
        min_improvement = current_price * Decimal("0.005")  # 0.5%
        if current_price - optimal_entry < min_improvement:
            return None

        return optimal_entry

    async def _calculate_limit_order_confidence(
        self,
        current_price: Decimal,
        entry_price: Decimal,
        levels: Dict[str, List[Decimal]],
        market_data: Dict
    ) -> float:
        """Calculate confidence for limit order execution"""

        scores = []

        # Price improvement score
        price_improvement = (current_price - entry_price) / current_price
        if price_improvement > Decimal("0.02"):  # >2% improvement
            improvement_score = 1.0
        elif price_improvement > Decimal("0.01"):  # >1%
            improvement_score = 0.8
        elif price_improvement > Decimal("0.005"):  # >0.5%
            improvement_score = 0.6
        else:
            improvement_score = 0.4
        scores.append(improvement_score * 0.30)  # 30% weight

        # Support level proximity score
        support_levels = levels.get("support", [])
        if support_levels:
            nearest_support = max([s for s in support_levels if s < current_price], default=entry_price)
            distance_to_support = abs(entry_price - nearest_support) / current_price
            if distance_to_support < Decimal("0.005"):  # Within 0.5%
                support_score = 1.0
            elif distance_to_support < Decimal("0.01"):
                support_score = 0.8
            elif distance_to_support < Decimal("0.02"):
                support_score = 0.6
            else:
                support_score = 0.4
            scores.append(support_score * 0.25)  # 25% weight

        # Volume score (higher volume = better fill probability)
        volume = market_data.get("volume_24h", 0)
        if volume > 100000:
            volume_score = 1.0
        elif volume > 50000:
            volume_score = 0.8
        elif volume > 10000:
            volume_score = 0.6
        else:
            volume_score = 0.4
        scores.append(volume_score * 0.20)  # 20% weight

        # Volatility score (moderate volatility is good for limit orders)
        volatility = market_data.get("volatility", 0)
        if 0.02 < volatility < 0.05:  # 2-5% volatility
            vol_score = 1.0
        elif volatility < 0.02 or volatility < 0.08:
            vol_score = 0.7
        else:
            vol_score = 0.5
        scores.append(vol_score * 0.15)  # 15% weight

        # Liquidity score
        liquidity = market_data.get("liquidity", 0)
        if liquidity > 100000:
            liq_score = 1.0
        elif liquidity > 50000:
            liq_score = 0.8
        elif liquidity > 10000:
            liq_score = 0.6
        else:
            liq_score = 0.4
        scores.append(liq_score * 0.10)  # 10% weight

        total_confidence = sum(scores)

        logger.debug(f"Limit order confidence: {total_confidence:.2f}")

        return total_confidence

    async def _create_limit_order_signal(
        self,
        token_address: str,
        entry_price: Decimal,
        current_price: Decimal,
        levels: Dict[str, List[Decimal]],
        confidence: float,
        market_data: Dict
    ) -> TradingSignal:
        """Create limit order trading signal"""

        # Determine signal strength
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate stop loss (below nearest support)
        support_levels = levels.get("support", [])
        if support_levels:
            support_below_entry = [s for s in support_levels if s < entry_price]
            if support_below_entry:
                stop_loss = max(support_below_entry) * Decimal("0.98")  # 2% below support
            else:
                stop_loss = entry_price * (Decimal("1") - self.default_stop_loss)
        else:
            stop_loss = entry_price * (Decimal("1") - self.default_stop_loss)

        # Calculate take profit (at nearest resistance)
        resistance_levels = levels.get("resistance", [])
        if resistance_levels:
            resistance_above_entry = [r for r in resistance_levels if r > entry_price]
            if resistance_above_entry:
                target_price = min(resistance_above_entry) * Decimal("0.98")  # Just below resistance
            else:
                risk = entry_price - stop_loss
                target_price = entry_price + (risk * Decimal("2"))  # 2:1 RR
        else:
            risk = entry_price - stop_loss
            target_price = entry_price + (risk * Decimal("2"))

        # Set expiration
        expires_at = datetime.now() + timedelta(seconds=self.max_order_age)

        signal = TradingSignal(
            strategy_name=self.name,
            signal_type=SignalType.BUY,
            strength=strength,
            token_address=token_address,
            chain="solana",
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            timeframe="limit_order",
            expires_at=expires_at,
            indicators={
                "current_price": float(current_price),
                "entry_price": float(entry_price),
                "price_improvement": float((current_price - entry_price) / current_price * 100),
                "support_levels": [float(s) for s in levels.get("support", [])[:3]],
                "resistance_levels": [float(r) for r in levels.get("resistance", [])[:3]]
            },
            metadata={
                "strategy_type": "jupiter_limit_order",
                "order_type": "limit",
                "current_market_price": float(current_price),
                "limit_price": float(entry_price),
                "partial_fill_enabled": self.partial_fill_enabled,
                "auto_cancel": self.auto_cancel_unfilled,
                "max_age": self.max_order_age,
                "use_jupiter": True
            }
        )

        logger.info(
            f"Created Jupiter limit order signal: {token_address[:10]}... "
            f"Limit=${entry_price:.6f}, Market=${current_price:.6f}, "
            f"Improvement={(current_price-entry_price)/current_price*100:.2f}%"
        )

        return signal

    async def place_jupiter_limit_order(
        self,
        token_address: str,
        input_mint: str,
        output_mint: str,
        amount: Decimal,
        limit_price: Decimal,
        wallet_address: str
    ) -> Optional[Dict]:
        """
        Place limit order on Jupiter

        Args:
            token_address: Token to trade
            input_mint: Input token mint
            output_mint: Output token mint
            amount: Amount to trade
            limit_price: Limit price
            wallet_address: Wallet address

        Returns:
            Order details or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Create limit order via Jupiter API
                url = f"{self.jupiter_limit_api}/createOrder"

                payload = {
                    "maker": wallet_address,
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "makingAmount": str(int(amount * Decimal("1e9"))),  # Convert to lamports
                    "takingAmount": str(int(amount * limit_price * Decimal("1e9"))),
                    "expiredAt": int((datetime.now() + timedelta(seconds=self.max_order_age)).timestamp())
                }

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Jupiter limit order placed: {data.get('orderKey')}")
                        return data
                    else:
                        error = await response.text()
                        logger.error(f"Failed to place Jupiter limit order: {error}")
                        return None

        except Exception as e:
            logger.error(f"Error placing Jupiter limit order: {e}", exc_info=True)
            return None

    async def get_open_orders(self, wallet_address: str) -> List[Dict]:
        """Get open Jupiter limit orders"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.jupiter_limit_api}/orders"
                params = {"wallet": wallet_address, "status": "open"}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("orders", [])
                    return []

        except Exception as e:
            logger.error(f"Error fetching Jupiter orders: {e}")
            return []

    async def cancel_order(self, order_key: str) -> bool:
        """Cancel Jupiter limit order"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.jupiter_limit_api}/cancelOrder"
                payload = {"orderKey": order_key}

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Order cancelled: {order_key}")
                        return True
                    return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def calculate_indicators(
        self,
        price_data: List[float],
        volume_data: List[float]
    ) -> Dict[str, Any]:
        """Calculate indicators for limit order strategy"""
        if not price_data:
            return {}

        return {
            "support_levels": self._find_local_minima(price_data),
            "resistance_levels": self._find_local_maxima(price_data),
            "current_price": price_data[-1] if price_data else 0,
            "avg_volume": sum(volume_data) / len(volume_data) if volume_data else 0
        }

    def validate_signal(
        self,
        signal: TradingSignal,
        market_data: Dict[str, Any]
    ) -> bool:
        """Validate limit order signal"""

        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            return False

        # Ensure limit price is better than market price
        current_price = Decimal(str(market_data.get("price", 0)))
        limit_price = signal.entry_price

        if signal.signal_type == SignalType.BUY:
            if limit_price >= current_price:
                logger.warning("Limit buy price not below market price")
                return False
        else:
            if limit_price <= current_price:
                logger.warning("Limit sell price not above market price")
                return False

        # Check minimum price improvement
        improvement = abs(current_price - limit_price) / current_price
        if improvement < Decimal("0.005"):  # 0.5% minimum
            logger.debug("Insufficient price improvement for limit order")
            return False

        return True

    def get_module_info(self) -> Dict:
        """Get strategy information"""
        return {
            "name": self.name,
            "type": "jupiter_limit_orders",
            "description": "Jupiter limit order strategy for better execution",
            "version": "1.0.0",
            "features": [
                "Support/Resistance level detection",
                "Smart limit order placement",
                "Automated stop loss/take profit orders",
                "DCA strategies",
                "Range trading",
                "Partial fill support"
            ],
            "parameters": {
                "limit_offset": f"{self.limit_order_offset_pct*100}%",
                "max_order_age": f"{self.max_order_age}s",
                "partial_fills": self.partial_fill_enabled,
                "dca_enabled": self.dca_enabled
            }
        }
