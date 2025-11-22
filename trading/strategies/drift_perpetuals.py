"""
Drift Protocol Perpetuals Strategy

Integrates with Drift Protocol for perpetual futures trading on Solana
Complementary to Binance Futures but Solana-native
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


class DriftMarketType(Enum):
    """Drift market types"""
    PERPETUAL = "perpetual"
    SPOT = "spot"


class DriftPositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class DriftPosition:
    """Drift Protocol position"""
    market_index: int
    market_symbol: str
    side: DriftPositionSide
    base_asset_amount: Decimal
    quote_asset_amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    liquidation_price: Decimal
    unrealized_pnl: Decimal
    leverage: float
    margin: Decimal
    last_funding_rate: float
    created_at: datetime


@dataclass
class DriftMarket:
    """Drift Protocol market"""
    market_index: int
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: DriftMarketType
    oracle_price: Decimal
    mark_price: Decimal
    index_price: Decimal
    funding_rate: float
    open_interest: Decimal
    volume_24h: Decimal
    max_leverage: int
    min_order_size: Decimal
    tick_size: Decimal


class DriftPerpetualsStrategy(BaseStrategy):
    """
    Strategy for trading perpetual futures on Drift Protocol

    Features:
    - Solana-native perpetuals trading
    - Up to 20x leverage (configurable)
    - Funding rate arbitrage
    - Basis trading (cash-and-carry)
    - Trend following on perpetuals
    - Integration with Drift's cross-margin system

    Advantages over CEX perpetuals:
    - Non-custodial
    - On-chain transparency
    - Lower fees
    - Solana speed
    - Composability with other DeFi protocols

    Use Cases:
    1. Directional trading (long/short)
    2. Hedging DEX positions
    3. Funding rate arbitrage
    4. Basis trading
    5. Delta-neutral strategies
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Drift perpetuals strategy"""
        super().__init__(config)

        # Drift configuration
        self.drift_program_id = config.get(
            "drift_program_id",
            "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH"  # Mainnet
        )
        self.drift_rpc = config.get("drift_rpc", "https://api.mainnet-beta.solana.com")

        # Trading configuration
        self.max_leverage = config.get("max_leverage", 5)  # Conservative default
        self.min_leverage = config.get("min_leverage", 1)
        self.use_cross_margin = config.get("use_cross_margin", True)
        self.max_positions = config.get("max_positions", 3)

        # Funding rate strategy
        self.funding_rate_enabled = config.get("funding_rate_enabled", True)
        self.funding_threshold = config.get("funding_threshold", 0.01)  # 0.01% per 8h
        self.funding_arb_min_apr = config.get("funding_arb_min_apr", 0.20)  # 20% APR

        # Risk management (more conservative for perpetuals)
        self.perp_stop_loss_pct = config.get("perp_stop_loss_pct", 0.08)  # 8% including leverage
        self.perp_take_profit_pct = config.get("perp_take_profit_pct", 0.15)  # 15%
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.20)  # 20%
        self.liquidation_buffer_pct = config.get("liquidation_buffer_pct", 0.25)  # 25% buffer

        # Position management
        self.use_trailing_stop = config.get("use_trailing_stop", True)
        self.trailing_distance_pct = config.get("trailing_distance_pct", 0.03)  # 3%
        self.partial_tp_enabled = config.get("partial_tp_enabled", True)
        self.tp_levels = config.get("tp_levels", [0.05, 0.10, 0.15, 0.20])
        self.tp_quantities = config.get("tp_quantities", [0.25, 0.25, 0.25, 0.25])

        # Market selection
        self.preferred_markets = config.get(
            "preferred_markets",
            ["SOL-PERP", "BTC-PERP", "ETH-PERP"]  # Most liquid markets
        )

        # Position tracking
        self.active_drift_positions: Dict[str, DriftPosition] = {}
        self.drift_markets: Dict[str, DriftMarket] = {}

        logger.info(
            f"Drift Perpetuals Strategy initialized: "
            f"max_leverage={self.max_leverage}x, markets={len(self.preferred_markets)}"
        )

    async def analyze(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Analyze market for Drift perpetuals opportunity

        Args:
            market_data: Market data

        Returns:
            Trading signal for Drift perpetuals
        """
        try:
            token_address = market_data.get("token_address")
            symbol = market_data.get("symbol", "")

            # Check if this is a Drift-supported market
            if not self._is_drift_market(symbol):
                return None

            # Get Drift market data
            drift_market = await self._get_drift_market(symbol)
            if not drift_market:
                return None

            # Check position limits
            if len(self.active_drift_positions) >= self.max_positions:
                logger.debug("Max Drift positions reached")
                return None

            # Analyze market conditions
            direction, confidence = await self._analyze_perp_opportunity(
                drift_market,
                market_data
            )

            if not direction or confidence < self.min_confidence:
                return None

            # Check funding rate strategy
            if self.funding_rate_enabled:
                funding_signal = await self._check_funding_rate_opportunity(drift_market)
                if funding_signal:
                    # Funding rate arbitrage opportunity takes precedence
                    return funding_signal

            # Create directional signal
            signal = await self._create_drift_signal(
                drift_market,
                direction,
                confidence,
                market_data
            )

            return signal

        except Exception as e:
            logger.error(f"Error analyzing Drift perpetuals: {e}", exc_info=True)
            return None

    def _is_drift_market(self, symbol: str) -> bool:
        """Check if symbol is supported on Drift"""
        if not symbol:
            return False

        # Check if it's in preferred markets
        if symbol in self.preferred_markets:
            return True

        # Check if it ends with -PERP
        if symbol.endswith("-PERP"):
            return True

        return False

    async def _get_drift_market(self, symbol: str) -> Optional[DriftMarket]:
        """Get Drift market data"""
        try:
            # Check cache
            if symbol in self.drift_markets:
                cached = self.drift_markets[symbol]
                # Cache valid for 30 seconds
                age = (datetime.now() - getattr(cached, '_cached_at', datetime.min)).seconds
                if age < 30:
                    return cached

            # TODO: Implement actual Drift SDK integration
            # For now, create mock market data
            # In production, this would call Drift Protocol SDK

            market_index = self._get_market_index(symbol)

            # Simulate market data
            mock_market = DriftMarket(
                market_index=market_index,
                symbol=symbol,
                base_asset=symbol.replace("-PERP", ""),
                quote_asset="USDC",
                market_type=DriftMarketType.PERPETUAL,
                oracle_price=Decimal("100.0"),  # Would fetch from oracle
                mark_price=Decimal("100.05"),
                index_price=Decimal("100.02"),
                funding_rate=0.0001,  # 0.01% per 8h
                open_interest=Decimal("1000000"),
                volume_24h=Decimal("5000000"),
                max_leverage=20,
                min_order_size=Decimal("0.01"),
                tick_size=Decimal("0.01")
            )

            # Cache market
            setattr(mock_market, '_cached_at', datetime.now())
            self.drift_markets[symbol] = mock_market

            return mock_market

        except Exception as e:
            logger.error(f"Error getting Drift market: {e}")
            return None

    def _get_market_index(self, symbol: str) -> int:
        """Get Drift market index for symbol"""
        # Drift market indices (would be fetched from SDK)
        indices = {
            "SOL-PERP": 0,
            "BTC-PERP": 1,
            "ETH-PERP": 2,
            "BONK-PERP": 3,
            "WIF-PERP": 4
        }
        return indices.get(symbol, 0)

    async def _analyze_perp_opportunity(
        self,
        market: DriftMarket,
        market_data: Dict
    ) -> tuple[Optional[str], float]:
        """
        Analyze perpetual market for trading opportunity

        Returns:
            (direction, confidence) where direction is 'LONG' or 'SHORT'
        """
        scores = []
        direction = None

        # Trend analysis
        price_change_24h = market_data.get("price_change_24h", 0)
        if price_change_24h > 0.05:  # +5%
            direction = "LONG"
            trend_score = 0.8
        elif price_change_24h < -0.05:  # -5%
            direction = "SHORT"
            trend_score = 0.8
        elif price_change_24h > 0.02:
            direction = "LONG"
            trend_score = 0.6
        elif price_change_24h < -0.02:
            direction = "SHORT"
            trend_score = 0.6
        else:
            trend_score = 0.3

        scores.append(trend_score * 0.30)  # 30% weight

        # Funding rate analysis
        funding_rate = market.funding_rate
        if direction == "LONG" and funding_rate < -0.0005:
            # Negative funding favors longs
            funding_score = 0.9
        elif direction == "SHORT" and funding_rate > 0.0005:
            # Positive funding favors shorts
            funding_score = 0.9
        elif abs(funding_rate) < 0.0002:
            # Neutral funding
            funding_score = 0.7
        else:
            # Funding against position
            funding_score = 0.5

        scores.append(funding_score * 0.20)  # 20% weight

        # Volume analysis
        volume_24h = float(market.volume_24h)
        if volume_24h > 10000000:  # >10M
            volume_score = 1.0
        elif volume_24h > 1000000:  # >1M
            volume_score = 0.8
        elif volume_24h > 100000:  # >100K
            volume_score = 0.6
        else:
            volume_score = 0.4

        scores.append(volume_score * 0.15)  # 15% weight

        # Open interest analysis
        oi = float(market.open_interest)
        if oi > 1000000:
            oi_score = 1.0
        elif oi > 500000:
            oi_score = 0.8
        elif oi > 100000:
            oi_score = 0.6
        else:
            oi_score = 0.4

        scores.append(oi_score * 0.15)  # 15% weight

        # Basis analysis (mark vs index)
        basis = float((market.mark_price - market.index_price) / market.index_price)
        if abs(basis) < 0.001:  # <0.1% basis
            basis_score = 1.0
        elif abs(basis) < 0.003:  # <0.3%
            basis_score = 0.8
        else:
            basis_score = 0.6

        scores.append(basis_score * 0.10)  # 10% weight

        # Volatility score
        volatility = market_data.get("volatility", 0.03)
        if 0.02 < volatility < 0.06:  # 2-6% ideal for perps
            vol_score = 1.0
        elif volatility < 0.02 or volatility < 0.10:
            vol_score = 0.7
        else:
            vol_score = 0.5

        scores.append(vol_score * 0.10)  # 10% weight

        total_confidence = sum(scores)

        logger.debug(
            f"Drift perp analysis: {market.symbol} "
            f"direction={direction}, confidence={total_confidence:.2f}, "
            f"funding={funding_rate:.4f}%"
        )

        return direction, total_confidence

    async def _check_funding_rate_opportunity(
        self,
        market: DriftMarket
    ) -> Optional[TradingSignal]:
        """
        Check for funding rate arbitrage opportunity

        Strategy: If funding rate is very high/low, take opposite position
        to collect funding payments
        """
        try:
            funding_rate = market.funding_rate
            funding_rate_8h = funding_rate * 3  # Convert to 8h rate

            # Calculate annualized funding rate
            funding_apr = funding_rate_8h * 365 * 3  # 3 periods per day

            if abs(funding_apr) < self.funding_arb_min_apr:
                return None

            # High positive funding = short to collect funding
            # High negative funding = long to collect funding
            if funding_apr > self.funding_arb_min_apr:
                direction = "SHORT"
                reason = "collecting_positive_funding"
            elif funding_apr < -self.funding_arb_min_apr:
                direction = "LONG"
                reason = "collecting_negative_funding"
            else:
                return None

            # Create funding arbitrage signal
            signal = TradingSignal(
                strategy_name=self.name,
                signal_type=SignalType.BUY if direction == "LONG" else SignalType.SELL,
                strength=SignalStrength.MODERATE,
                token_address=market.symbol,
                chain="solana",
                entry_price=market.mark_price,
                target_price=market.mark_price,  # Funding arb, not directional
                stop_loss=market.mark_price * (
                    Decimal("1.03") if direction == "LONG" else Decimal("0.97")
                ),
                confidence=0.7,
                timeframe="funding",
                indicators={
                    "funding_rate_8h": funding_rate_8h,
                    "funding_apr": funding_apr,
                    "mark_price": float(market.mark_price)
                },
                metadata={
                    "strategy_type": "drift_funding_arb",
                    "direction": direction,
                    "reason": reason,
                    "market_index": market.market_index,
                    "leverage": 1,  # Low leverage for funding arb
                    "expected_apr": abs(funding_apr)
                }
            )

            logger.info(
                f"Funding arbitrage opportunity: {market.symbol} "
                f"Direction={direction}, APR={abs(funding_apr):.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error checking funding rate: {e}")
            return None

    async def _create_drift_signal(
        self,
        market: DriftMarket,
        direction: str,
        confidence: float,
        market_data: Dict
    ) -> TradingSignal:
        """Create Drift perpetuals trading signal"""

        # Determine signal strength
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate leverage based on confidence and volatility
        volatility = market_data.get("volatility", 0.03)
        leverage = self._calculate_optimal_leverage(confidence, volatility)

        # Entry price
        entry_price = market.mark_price

        # Calculate stop loss and take profit
        if direction == "LONG":
            stop_loss = entry_price * (Decimal("1") - Decimal(str(self.perp_stop_loss_pct / leverage)))
            target_price = entry_price * (Decimal("1") + Decimal(str(self.perp_take_profit_pct / leverage)))
        else:  # SHORT
            stop_loss = entry_price * (Decimal("1") + Decimal(str(self.perp_stop_loss_pct / leverage)))
            target_price = entry_price * (Decimal("1") - Decimal(str(self.perp_take_profit_pct / leverage)))

        # Calculate liquidation price estimate
        liquidation_price = self._estimate_liquidation_price(
            entry_price,
            direction,
            leverage
        )

        signal = TradingSignal(
            strategy_name=self.name,
            signal_type=SignalType.BUY if direction == "LONG" else SignalType.SELL,
            strength=strength,
            token_address=market.symbol,
            chain="solana",
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            timeframe="perp",
            indicators={
                "mark_price": float(market.mark_price),
                "index_price": float(market.index_price),
                "funding_rate": market.funding_rate,
                "open_interest": float(market.open_interest),
                "volume_24h": float(market.volume_24h)
            },
            metadata={
                "strategy_type": "drift_perpetuals",
                "direction": direction,
                "leverage": leverage,
                "market_index": market.market_index,
                "market_symbol": market.symbol,
                "liquidation_price": float(liquidation_price),
                "use_cross_margin": self.use_cross_margin,
                "trailing_stop_enabled": self.use_trailing_stop,
                "trailing_distance_pct": self.trailing_distance_pct,
                "tp_levels": self.tp_levels,
                "tp_quantities": self.tp_quantities,
                "protocol": "drift"
            }
        )

        logger.info(
            f"Created Drift perpetuals signal: {market.symbol} "
            f"{direction} {leverage}x, Entry=${entry_price:.2f}, "
            f"Target=${target_price:.2f}, SL=${stop_loss:.2f}"
        )

        return signal

    def _calculate_optimal_leverage(self, confidence: float, volatility: float) -> int:
        """Calculate optimal leverage based on confidence and volatility"""

        # Start with max leverage
        base_leverage = self.max_leverage

        # Reduce leverage for lower confidence
        if confidence < 0.7:
            base_leverage = max(self.min_leverage, base_leverage - 2)
        elif confidence < 0.75:
            base_leverage = max(self.min_leverage, base_leverage - 1)

        # Reduce leverage for high volatility
        if volatility > 0.08:  # >8% volatility
            base_leverage = max(self.min_leverage, base_leverage - 2)
        elif volatility > 0.05:  # >5%
            base_leverage = max(self.min_leverage, base_leverage - 1)

        return int(max(self.min_leverage, min(base_leverage, self.max_leverage)))

    def _estimate_liquidation_price(
        self,
        entry_price: Decimal,
        direction: str,
        leverage: int
    ) -> Decimal:
        """Estimate liquidation price"""

        # Simplified liquidation calculation
        # Actual calculation depends on Drift's margin requirements

        liquidation_distance = Decimal("1") / Decimal(str(leverage))

        if direction == "LONG":
            liquidation_price = entry_price * (Decimal("1") - liquidation_distance)
        else:  # SHORT
            liquidation_price = entry_price * (Decimal("1") + liquidation_distance)

        return liquidation_price

    async def calculate_indicators(
        self,
        price_data: List[float],
        volume_data: List[float]
    ) -> Dict[str, Any]:
        """Calculate indicators for Drift perpetuals"""
        if not price_data:
            return {}

        return {
            "current_price": price_data[-1] if price_data else 0,
            "volatility": float(Decimal(str(max(price_data) - min(price_data))) / Decimal(str(price_data[-1]))),
            "avg_volume": sum(volume_data) / len(volume_data) if volume_data else 0
        }

    def validate_signal(
        self,
        signal: TradingSignal,
        market_data: Dict[str, Any]
    ) -> bool:
        """Validate Drift perpetuals signal"""

        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            return False

        # Check leverage is within limits
        leverage = signal.metadata.get("leverage", 1)
        if leverage < self.min_leverage or leverage > self.max_leverage:
            logger.warning(f"Leverage {leverage}x outside limits")
            return False

        # Check liquidation price is safe
        liquidation_price = Decimal(str(signal.metadata.get("liquidation_price", 0)))
        entry_price = signal.entry_price

        liquidation_distance = abs(entry_price - liquidation_price) / entry_price
        if liquidation_distance < Decimal(str(self.liquidation_buffer_pct)):
            logger.warning("Liquidation price too close to entry")
            return False

        return True

    def get_module_info(self) -> Dict:
        """Get strategy information"""
        return {
            "name": self.name,
            "type": "drift_perpetuals",
            "description": "Drift Protocol perpetual futures trading",
            "version": "1.0.0",
            "features": [
                "Solana-native perpetuals",
                "Up to 20x leverage",
                "Funding rate arbitrage",
                "Cross-margin support",
                "Trailing stop loss",
                "Partial take profits",
                "Non-custodial trading"
            ],
            "parameters": {
                "max_leverage": f"{self.max_leverage}x",
                "cross_margin": self.use_cross_margin,
                "funding_arb": self.funding_rate_enabled,
                "trailing_stop": self.use_trailing_stop,
                "preferred_markets": self.preferred_markets
            }
        }
