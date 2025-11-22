"""
Pump.fun Launch Trading Strategy

Monitors and trades new token launches on Pump.fun platform
Focuses on early entry opportunities with strict risk management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import aiohttp

from .base_strategy import BaseStrategy, TradingSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


class LaunchPhase(Enum):
    """Token launch phases"""
    PRE_LAUNCH = "pre_launch"  # Before bonding curve complete
    EARLY = "early"  # 0-5 minutes
    MID = "mid"  # 5-30 minutes
    LATE = "late"  # 30+ minutes
    GRADUATED = "graduated"  # Migrated to Raydium


@dataclass
class LaunchMetrics:
    """Metrics for token launch analysis"""
    age_seconds: int
    bonding_progress: float  # 0-100% progress to Raydium migration
    holder_count: int
    holder_growth_rate: float
    buy_sell_ratio: float
    initial_liquidity: Decimal
    current_liquidity: Decimal
    developer_holding_pct: float
    top_10_holding_pct: float
    transaction_count: int
    unique_buyers: int
    avg_buy_size: Decimal
    max_buy_size: Decimal
    sniper_count: int  # Wallets that bought in first 10 seconds
    is_dev_sold: bool
    is_bundled: bool  # Suspicious bundled launch
    social_score: float


class PumpFunLaunchStrategy(BaseStrategy):
    """
    Advanced strategy for trading Pump.fun token launches

    Features:
    - Real-time launch monitoring
    - Early entry detection
    - Bonding curve analysis
    - Developer behavior tracking
    - Sniper detection
    - Graduated token monitoring (post-Raydium migration)

    Risk Controls:
    - Maximum 2% portfolio per launch
    - Quick stop loss (15-20%)
    - Graduated exits (25% at each level)
    - Maximum hold time enforcement
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Pump.fun launch strategy"""
        super().__init__(config)

        # Strategy-specific config
        self.max_token_age = config.get("max_token_age", 1800)  # 30 minutes default
        self.min_bonding_progress = config.get("min_bonding_progress", 10.0)
        self.max_bonding_progress = config.get("max_bonding_progress", 85.0)
        self.min_holder_count = config.get("min_holder_count", 20)
        self.max_dev_holding = config.get("max_dev_holding", 5.0)  # 5%
        self.max_top_10_holding = config.get("max_top_10_holding", 25.0)  # 25%
        self.min_buy_sell_ratio = config.get("min_buy_sell_ratio", 1.5)
        self.max_sniper_pct = config.get("max_sniper_pct", 15.0)

        # Exit strategy
        self.quick_stop_loss_pct = config.get("quick_stop_loss_pct", 0.15)  # 15%
        self.tp_levels = config.get("tp_levels", [0.30, 0.50, 1.00, 2.00])  # 30%, 50%, 100%, 200%
        self.tp_quantities = config.get("tp_quantities", [0.25, 0.25, 0.25, 0.25])
        self.max_hold_time = config.get("max_hold_time", 3600)  # 1 hour max

        # Graduated token settings
        self.trade_graduated = config.get("trade_graduated", True)
        self.graduated_min_liquidity = config.get("graduated_min_liquidity", 50000)

        # Monitoring
        self.launch_cache: Dict[str, LaunchMetrics] = {}
        self.blacklisted_devs: set = set()

        # API endpoints
        self.pumpfun_api = config.get("pumpfun_api", "https://frontend-api.pump.fun")
        self.pumpfun_ws = config.get("pumpfun_ws", "wss://frontend-api.pump.fun/socket")

        logger.info(f"Pump.fun Launch Strategy initialized: max_age={self.max_token_age}s")

    async def analyze(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Analyze Pump.fun launch for trading opportunity

        Args:
            market_data: Market data including launch metrics

        Returns:
            Trading signal if opportunity found
        """
        try:
            token_address = market_data.get("token_address")
            if not token_address:
                return None

            # Verify this is a Pump.fun token
            if not market_data.get("is_pumpfun", False):
                return None

            # Get or fetch launch metrics
            launch_metrics = await self._get_launch_metrics(token_address, market_data)
            if not launch_metrics:
                return None

            # Determine launch phase
            phase = self._determine_launch_phase(launch_metrics)

            # Check if we should trade based on phase
            if phase == LaunchPhase.LATE:
                logger.debug(f"Token too old: {token_address[:10]}...")
                return None

            if phase == LaunchPhase.GRADUATED and not self.trade_graduated:
                logger.debug(f"Graduated token trading disabled: {token_address[:10]}...")
                return None

            # Run safety checks
            if not await self._pass_safety_checks(launch_metrics, token_address):
                return None

            # Calculate entry confidence
            confidence = await self._calculate_launch_confidence(
                launch_metrics,
                phase,
                market_data
            )

            if confidence < self.min_confidence:
                return None

            # Create trading signal
            signal = await self._create_launch_signal(
                token_address,
                market_data,
                launch_metrics,
                phase,
                confidence
            )

            return signal

        except Exception as e:
            logger.error(f"Error analyzing Pump.fun launch: {e}", exc_info=True)
            return None

    async def _get_launch_metrics(
        self,
        token_address: str,
        market_data: Dict
    ) -> Optional[LaunchMetrics]:
        """Fetch or calculate launch metrics"""
        try:
            # Check cache first
            if token_address in self.launch_cache:
                cached = self.launch_cache[token_address]
                # Refresh if older than 30 seconds
                if (datetime.now() - market_data.get("timestamp", datetime.now())).seconds < 30:
                    return cached

            # Fetch from Pump.fun API
            async with aiohttp.ClientSession() as session:
                url = f"{self.pumpfun_api}/coins/{token_address}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        metrics = self._parse_launch_data(data, market_data)

                        # Cache metrics
                        self.launch_cache[token_address] = metrics
                        return metrics

            # Fallback to market_data
            return self._parse_launch_data(market_data, market_data)

        except Exception as e:
            logger.error(f"Error fetching launch metrics: {e}")
            return None

    def _parse_launch_data(self, data: Dict, market_data: Dict) -> LaunchMetrics:
        """Parse launch data into metrics"""
        now = datetime.now()
        launch_time = data.get("created_timestamp")
        if launch_time:
            if isinstance(launch_time, (int, float)):
                age = int(now.timestamp() - launch_time)
            else:
                age = int((now - datetime.fromisoformat(launch_time)).total_seconds())
        else:
            age = 0

        return LaunchMetrics(
            age_seconds=age,
            bonding_progress=float(data.get("bonding_progress", 0)),
            holder_count=int(data.get("holder_count", 0)),
            holder_growth_rate=float(data.get("holder_growth_rate", 0)),
            buy_sell_ratio=float(data.get("buy_sell_ratio", 1.0)),
            initial_liquidity=Decimal(str(data.get("initial_liquidity", 0))),
            current_liquidity=Decimal(str(data.get("liquidity", market_data.get("liquidity", 0)))),
            developer_holding_pct=float(data.get("dev_holding_pct", 0)),
            top_10_holding_pct=float(data.get("top_10_holding_pct", 0)),
            transaction_count=int(data.get("txn_count", 0)),
            unique_buyers=int(data.get("unique_buyers", 0)),
            avg_buy_size=Decimal(str(data.get("avg_buy", 0))),
            max_buy_size=Decimal(str(data.get("max_buy", 0))),
            sniper_count=int(data.get("sniper_count", 0)),
            is_dev_sold=bool(data.get("dev_sold", False)),
            is_bundled=bool(data.get("is_bundled", False)),
            social_score=float(data.get("social_score", 0))
        )

    def _determine_launch_phase(self, metrics: LaunchMetrics) -> LaunchPhase:
        """Determine current phase of token launch"""
        if metrics.bonding_progress >= 100:
            return LaunchPhase.GRADUATED
        elif metrics.age_seconds < 300:  # 5 minutes
            return LaunchPhase.EARLY
        elif metrics.age_seconds < 1800:  # 30 minutes
            return LaunchPhase.MID
        else:
            return LaunchPhase.LATE

    async def _pass_safety_checks(
        self,
        metrics: LaunchMetrics,
        token_address: str
    ) -> bool:
        """Run safety checks on launch"""

        # Check age
        if metrics.age_seconds > self.max_token_age:
            logger.debug(f"Token too old: {metrics.age_seconds}s")
            return False

        # Check bonding curve progress
        if metrics.bonding_progress < self.min_bonding_progress:
            logger.debug(f"Bonding progress too low: {metrics.bonding_progress}%")
            return False

        if metrics.bonding_progress > self.max_bonding_progress:
            logger.debug(f"Bonding progress too high: {metrics.bonding_progress}%")
            return False

        # Check holder metrics
        if metrics.holder_count < self.min_holder_count:
            logger.debug(f"Not enough holders: {metrics.holder_count}")
            return False

        # Check concentration
        if metrics.developer_holding_pct > self.max_dev_holding:
            logger.warning(f"Dev holding too high: {metrics.developer_holding_pct}%")
            return False

        if metrics.top_10_holding_pct > self.max_top_10_holding:
            logger.warning(f"Top 10 holding too high: {metrics.top_10_holding_pct}%")
            return False

        # Check buy/sell ratio
        if metrics.buy_sell_ratio < self.min_buy_sell_ratio:
            logger.debug(f"Buy/sell ratio too low: {metrics.buy_sell_ratio}")
            return False

        # Check for rug signs
        if metrics.is_dev_sold:
            logger.warning(f"Developer already sold: {token_address[:10]}...")
            self.blacklisted_devs.add(token_address)
            return False

        if metrics.is_bundled:
            logger.warning(f"Suspicious bundled launch: {token_address[:10]}...")
            return False

        # Check sniper percentage
        if metrics.holder_count > 0:
            sniper_pct = (metrics.sniper_count / metrics.holder_count) * 100
            if sniper_pct > self.max_sniper_pct:
                logger.warning(f"Too many snipers: {sniper_pct:.1f}%")
                return False

        return True

    async def _calculate_launch_confidence(
        self,
        metrics: LaunchMetrics,
        phase: LaunchPhase,
        market_data: Dict
    ) -> float:
        """Calculate confidence score for launch trade"""

        scores = []

        # Bonding progress score (sweet spot: 20-70%)
        if 20 <= metrics.bonding_progress <= 70:
            bonding_score = 1.0
        elif 10 <= metrics.bonding_progress < 20:
            bonding_score = 0.7
        elif 70 < metrics.bonding_progress <= 85:
            bonding_score = 0.6
        else:
            bonding_score = 0.3
        scores.append(bonding_score * 0.25)  # 25% weight

        # Holder growth score
        if metrics.holder_growth_rate > 0.5:  # 50% growth
            holder_score = 1.0
        elif metrics.holder_growth_rate > 0.2:
            holder_score = 0.7
        else:
            holder_score = 0.4
        scores.append(holder_score * 0.15)  # 15% weight

        # Buy/sell ratio score
        if metrics.buy_sell_ratio > 3.0:
            bs_score = 1.0
        elif metrics.buy_sell_ratio > 2.0:
            bs_score = 0.8
        elif metrics.buy_sell_ratio > 1.5:
            bs_score = 0.6
        else:
            bs_score = 0.3
        scores.append(bs_score * 0.20)  # 20% weight

        # Concentration score (lower is better)
        concentration = metrics.top_10_holding_pct
        if concentration < 15:
            conc_score = 1.0
        elif concentration < 20:
            conc_score = 0.8
        elif concentration < 25:
            conc_score = 0.6
        else:
            conc_score = 0.3
        scores.append(conc_score * 0.15)  # 15% weight

        # Phase score
        phase_scores = {
            LaunchPhase.EARLY: 1.0,
            LaunchPhase.MID: 0.7,
            LaunchPhase.GRADUATED: 0.6,
            LaunchPhase.LATE: 0.3
        }
        scores.append(phase_scores.get(phase, 0.5) * 0.10)  # 10% weight

        # Social score
        social_score = min(metrics.social_score / 100, 1.0)
        scores.append(social_score * 0.15)  # 15% weight

        total_confidence = sum(scores)

        logger.debug(
            f"Launch confidence: {total_confidence:.2f} "
            f"(bonding={bonding_score:.2f}, holders={holder_score:.2f}, "
            f"bs_ratio={bs_score:.2f}, phase={phase.value})"
        )

        return total_confidence

    async def _create_launch_signal(
        self,
        token_address: str,
        market_data: Dict,
        metrics: LaunchMetrics,
        phase: LaunchPhase,
        confidence: float
    ) -> TradingSignal:
        """Create trading signal for launch"""

        current_price = Decimal(str(market_data.get("price", 0)))

        # Determine signal strength
        if confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            strength = SignalStrength.STRONG
        elif confidence >= 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Calculate stop loss (tighter for launches)
        stop_loss = current_price * (Decimal("1") - Decimal(str(self.quick_stop_loss_pct)))

        # Calculate take profit targets
        tp_targets = [
            current_price * (Decimal("1") + Decimal(str(tp_pct)))
            for tp_pct in self.tp_levels
        ]

        # Set expiration based on max hold time
        expires_at = datetime.now() + timedelta(seconds=self.max_hold_time)

        signal = TradingSignal(
            strategy_name=self.name,
            signal_type=SignalType.BUY,
            strength=strength,
            token_address=token_address,
            chain="solana",
            entry_price=current_price,
            target_price=tp_targets[0],  # First TP level
            stop_loss=stop_loss,
            confidence=confidence,
            timeframe="1m",
            expires_at=expires_at,
            indicators={
                "bonding_progress": metrics.bonding_progress,
                "holder_count": metrics.holder_count,
                "buy_sell_ratio": metrics.buy_sell_ratio,
                "age_seconds": metrics.age_seconds,
                "phase": phase.value
            },
            metadata={
                "strategy_type": "pumpfun_launch",
                "launch_phase": phase.value,
                "tp_levels": [float(tp) for tp in tp_targets],
                "tp_quantities": self.tp_quantities,
                "max_hold_time": self.max_hold_time,
                "bonding_progress": metrics.bonding_progress,
                "is_graduated": phase == LaunchPhase.GRADUATED
            }
        )

        logger.info(
            f"Created Pump.fun launch signal: {token_address[:10]}... "
            f"Phase={phase.value}, Confidence={confidence:.2f}, "
            f"Entry=${current_price:.6f}"
        )

        return signal

    async def calculate_indicators(
        self,
        price_data: List[float],
        volume_data: List[float]
    ) -> Dict[str, Any]:
        """Calculate indicators for launch strategy"""
        # Launch strategy relies more on launch metrics than traditional indicators
        if not price_data or not volume_data:
            return {}

        return {
            "price_velocity": self._calculate_velocity(price_data),
            "volume_trend": self._calculate_velocity(volume_data),
            "volatility": float(np.std(price_data)) if len(price_data) > 1 else 0.0
        }

    def _calculate_velocity(self, data: List[float]) -> float:
        """Calculate rate of change"""
        if len(data) < 2:
            return 0.0

        recent = data[-5:] if len(data) >= 5 else data
        if len(recent) < 2:
            return 0.0

        changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        return float(np.mean(changes))

    def validate_signal(
        self,
        signal: TradingSignal,
        market_data: Dict[str, Any]
    ) -> bool:
        """Validate launch trading signal"""

        # Check minimum confidence
        if signal.confidence < self.min_confidence:
            return False

        # Check if token is blacklisted
        if signal.token_address in self.blacklisted_devs:
            logger.warning(f"Token is blacklisted: {signal.token_address[:10]}...")
            return False

        # Verify liquidity for graduated tokens
        if signal.metadata.get("is_graduated", False):
            liquidity = market_data.get("liquidity", 0)
            if liquidity < self.graduated_min_liquidity:
                logger.debug(f"Insufficient liquidity for graduated token: ${liquidity}")
                return False

        return True

    async def _check_custom_exit_conditions(
        self,
        position: TradingSignal,
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Check Pump.fun specific exit conditions"""

        current_price = Decimal(str(market_data.get("price", 0)))

        # Check if bonding curve completed (graduated to Raydium)
        if market_data.get("is_graduated", False) and not position.metadata.get("was_graduated", False):
            logger.info(f"Token graduated to Raydium: {position.token_address[:10]}...")
            # Could continue holding or exit - configurable
            if not self.trade_graduated:
                return self._create_exit_signal(position, "graduated", current_price)

        # Check for dev dump
        if market_data.get("dev_sold", False):
            logger.warning(f"Developer dumped: {position.token_address[:10]}...")
            self.blacklisted_devs.add(position.token_address)
            return self._create_exit_signal(position, "dev_dump", current_price)

        # Check partial TP levels
        tp_levels = position.metadata.get("tp_levels", [])
        tp_quantities = position.metadata.get("tp_quantities", [])

        for i, (tp_price, tp_qty) in enumerate(zip(tp_levels, tp_quantities)):
            if current_price >= Decimal(str(tp_price)):
                # Mark this TP level as hit
                hit_key = f"tp_{i}_hit"
                if not position.metadata.get(hit_key, False):
                    position.metadata[hit_key] = True
                    logger.info(
                        f"TP level {i+1} hit: {position.token_address[:10]}... "
                        f"Selling {tp_qty*100:.0f}% at ${current_price:.6f}"
                    )
                    # Return partial exit signal
                    exit_signal = self._create_exit_signal(position, f"tp_{i+1}", current_price)
                    exit_signal.metadata["partial_exit"] = True
                    exit_signal.metadata["exit_percentage"] = tp_qty
                    return exit_signal

        return None

    def get_module_info(self) -> Dict:
        """Get strategy information"""
        return {
            "name": self.name,
            "type": "pumpfun_launch",
            "description": "Pump.fun token launch trading strategy",
            "version": "1.0.0",
            "features": [
                "Real-time launch monitoring",
                "Bonding curve analysis",
                "Developer behavior tracking",
                "Graduated token support",
                "Multi-level take profits",
                "Quick stop loss protection"
            ],
            "parameters": {
                "max_token_age": self.max_token_age,
                "bonding_range": f"{self.min_bonding_progress}-{self.max_bonding_progress}%",
                "min_holders": self.min_holder_count,
                "stop_loss": f"{self.quick_stop_loss_pct*100}%",
                "tp_levels": self.tp_levels,
                "max_hold_time": f"{self.max_hold_time}s"
            }
        }
