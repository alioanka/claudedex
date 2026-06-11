"""
Futures Trading Engine
Core engine for futures trading on Binance and Bybit

Features:
- Multi-exchange support (Binance, Bybit)
- DRY_RUN mode with simulated trades
- Testnet support
- Leverage management with per-symbol setup
- Position tracking with real-time PnL
- Technical indicator-based strategies (RSI, MACD, Volume)
- Risk management (daily loss limits, position limits, liquidation protection)
- Fee and slippage modeling
- Per-symbol cooldowns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import os
import uuid
import json
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dry_run import resolve_dry_run_env, should_skip_live
from core.pnl_tracker import PnLTracker, TradeRecord

logger = logging.getLogger("FuturesTradingEngine")


class TradeSide(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class SignalStrength(Enum):
    """Signal strength levels"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class TechnicalSignals:
    """Technical indicator signals"""
    rsi: float = 50.0
    rsi_signal: SignalStrength = SignalStrength.NEUTRAL
    macd: float = 0.0
    macd_signal_line: float = 0.0
    macd_histogram: float = 0.0
    macd_signal: SignalStrength = SignalStrength.NEUTRAL
    volume_ratio: float = 1.0  # Current volume / Average volume
    volume_signal: SignalStrength = SignalStrength.NEUTRAL
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_middle: float = 0.0
    bb_signal: SignalStrength = SignalStrength.NEUTRAL
    bb_position: str = "middle"  # above_upper, below_lower, middle
    # EMA Crossover
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_signal: SignalStrength = SignalStrength.NEUTRAL
    ema_crossover: str = "none"  # golden_cross, death_cross, none
    # Price action
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    trend: str = "sideways"  # uptrend, downtrend, sideways
    support_level: float = 0.0
    resistance_level: float = 0.0
    # FUT-RM-06: Average True Range (14-period). Same price units as the
    # underlying (quote currency for USDT perps). Zero when there isn't
    # enough history. Used by _calculate_position_size for risk-parity
    # sizing across symbols of different volatility.
    atr: float = 0.0
    atr_pct: float = 0.0  # atr / last_close (decimal, e.g. 0.025 = 2.5%)

    @property
    def overall_signal(self) -> SignalStrength:
        """Calculate overall signal from all indicators"""
        # Now includes 5 indicators: RSI, MACD, Volume, Bollinger Bands, EMA
        score = (
            self.rsi_signal.value +
            self.macd_signal.value +
            self.volume_signal.value +
            self.bb_signal.value +
            self.ema_signal.value
        )
        # Adjusted thresholds for 5 indicators (max score now -10 to +10)
        if score >= 5:
            return SignalStrength.STRONG_BUY
        elif score >= 3:
            return SignalStrength.BUY
        elif score <= -5:
            return SignalStrength.STRONG_SELL
        elif score <= -3:
            return SignalStrength.SELL
        return SignalStrength.NEUTRAL


@dataclass
class Position:
    """Active trading position"""
    position_id: str
    symbol: str
    side: TradeSide
    entry_price: float
    current_price: float
    size: float  # Position size in base currency (e.g., BTC)
    notional_value: float  # Position value in USDT
    leverage: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    fees_paid: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    is_simulated: bool = False  # True if DRY_RUN
    metadata: Dict = field(default_factory=dict)
    # Multiple TP levels: [{'level': 1, 'price': 1.05, 'pct': 2.0, 'size_pct': 25.0, 'hit': False}, ...]
    tp_levels: List[Dict] = field(default_factory=list)
    # Original size to track partial closes
    original_size: float = 0.0
    # Trailing stop tracking
    trailing_stop_price: Optional[float] = None
    trailing_stop_active: bool = False  # Activated after TP2 is hit
    highest_price: Optional[float] = None  # For LONG: track highest since entry
    lowest_price: Optional[float] = None   # For SHORT: track lowest since entry


@dataclass
class Trade:
    """Completed trade record"""
    trade_id: str
    symbol: str
    side: TradeSide
    entry_price: float
    exit_price: float
    size: float
    notional_value: float
    leverage: int
    pnl: float
    pnl_pct: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    close_reason: str  # stop_loss, take_profit, manual, signal
    is_simulated: bool = False


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_loss_limit: float = 500.0  # From FUTURES_MAX_DAILY_LOSS_USD
    current_exposure: float = 0.0  # Total notional value
    max_exposure: float = 0.0  # Max allowed exposure
    consecutive_losses: int = 0
    # Wave-25: was hard-coded 5 inside can_trade, ignoring the DB-configured
    # futures_risk.max_consecutive_losses (default 4). Now injected by the
    # engine from FuturesRiskConfig.
    max_consecutive_losses: int = 5
    drawdown_pct: float = 0.0
    peak_balance: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Daily loss limit check
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        # Consecutive losses circuit breaker (any-symbol streak; clears on a
        # winning close or at the midnight-UTC daily reset)
        if self.consecutive_losses >= max(1, self.max_consecutive_losses):
            return False
        return True

    @property
    def pause_reason(self) -> Optional[str]:
        """Wave-25: honest, specific pause reason for logging/status surfaces.
        The old log claimed 'daily risk limit reached' even when the actual
        trigger was the consecutive-loss breaker (observed: paused with a
        POSITIVE +$20.68 daily PnL). Returns None when trading is allowed."""
        if self.daily_pnl <= -self.daily_loss_limit:
            return (
                f"daily LOSS limit reached (Daily PnL ${self.daily_pnl:.2f} "
                f"<= -${self.daily_loss_limit:.2f})"
            )
        if self.consecutive_losses >= max(1, self.max_consecutive_losses):
            return (
                f"consecutive-loss circuit breaker "
                f"({self.consecutive_losses} losses in a row >= "
                f"{self.max_consecutive_losses}; clears on a winning close "
                f"or at the midnight-UTC daily reset; Daily PnL "
                f"${self.daily_pnl:.2f})"
            )
        return None

    @property
    def risk_level(self) -> str:
        """Current risk level"""
        loss_ratio = abs(self.daily_pnl) / self.daily_loss_limit if self.daily_loss_limit > 0 else 0
        if loss_ratio >= 0.8:
            return "HIGH"
        elif loss_ratio >= 0.5:
            return "MEDIUM"
        return "LOW"


class FuturesTradingEngine:
    """
    Core futures trading engine with full implementation

    Configuration Architecture:
    - Receives config from FuturesConfigManager (database-backed)
    - Only sensitive data (API keys) comes from .env via config manager
    - Trading parameters are loaded from database and can be changed without restart
    """

    # Trading fees (maker/taker)
    BINANCE_MAKER_FEE = 0.0002  # 0.02%
    BINANCE_TAKER_FEE = 0.0004  # 0.04%
    BYBIT_MAKER_FEE = 0.0001   # 0.01%
    BYBIT_TAKER_FEE = 0.0006   # 0.06%

    # Default slippage estimate
    DEFAULT_SLIPPAGE = 0.0005  # 0.05%

    def __init__(
        self,
        config_manager=None,
        mode: str = "production",
        db_pool=None
    ):
        """
        Initialize futures trading engine with database-backed configuration

        Args:
            config_manager: FuturesConfigManager instance (loads from database)
            mode: Operating mode
            db_pool: asyncpg connection pool for trade persistence
        """
        self.config_manager = config_manager
        self.mode = mode
        self.is_running = False
        self.db_pool = db_pool  # Database connection pool for trade persistence

        # MB-16: resolve DRY_RUN BEFORE testnet-safety branch reads self.dry_run.
        # Safe default True so a missing/typo DRY_RUN env var never goes live.
        self.dry_run = resolve_dry_run_env('DRY_RUN', default=True)

        # Load configuration from config manager (database)
        if config_manager:
            general_config = config_manager.get_general()
            position_config = config_manager.get_position()
            leverage_config = config_manager.get_leverage()
            risk_config = config_manager.get_risk()
            pairs_config = config_manager.get_pairs()
            strategy_config = config_manager.get_strategy()

            # General settings
            self.exchange = general_config.exchange.lower()

            # Testnet/Mainnet determination - IMPORTANT precedence:
            # 1. FUTURES_TESTNET env var (explicit override)
            # 2. If DRY_RUN=false (live trading), default to MAINNET for safety
            # 3. Fall back to database config (only in dry run mode)
            testnet_env = os.getenv('FUTURES_TESTNET', '').strip().lower()
            testnet_source = None

            if testnet_env in ('true', '1', 'yes'):
                self.testnet = True
                testnet_source = "FUTURES_TESTNET env var"
            elif testnet_env in ('false', '0', 'no'):
                self.testnet = False
                testnet_source = "FUTURES_TESTNET env var"
            elif not self.dry_run:
                # SAFETY: Live trading (DRY_RUN=false) defaults to MAINNET
                # This prevents accidental testnet trading when live trading is enabled
                self.testnet = False
                testnet_source = "DRY_RUN=false safety default (use FUTURES_TESTNET=true to override)"
            else:
                # In dry run mode, use database setting
                self.testnet = general_config.testnet
                testnet_source = f"database config (general_config.testnet={general_config.testnet})"

            logger.info(f"   Testnet: {self.testnet} (source: {testnet_source})")

            # Position settings
            self.max_positions = position_config.max_positions
            self.capital_allocation = position_config.capital_allocation

            # Dynamic position sizing settings
            self.dynamic_position_sizing = getattr(position_config, 'dynamic_position_sizing', True)
            self.min_position_pct = getattr(position_config, 'min_position_pct', 5.0)
            self.max_position_pct = getattr(position_config, 'max_position_pct', 20.0)
            self.max_position_usd = getattr(position_config, 'max_position_usd', 500.0)
            self.static_position_pct = getattr(position_config, 'static_position_pct', 15.0)
            self.position_size_usd = position_config.position_size_usd  # Legacy fallback
            self.min_trade_size = getattr(position_config, 'min_trade_size', 10.0)
            # FUT-RM-06: ATR-based sizing toggles
            self.atr_sizing_enabled = getattr(position_config, 'atr_sizing_enabled', False)
            self.atr_risk_pct = getattr(position_config, 'atr_risk_pct', 1.0)
            self.atr_stop_multiplier = getattr(position_config, 'atr_stop_multiplier', 1.5)

            # Leverage settings
            self.leverage = leverage_config.default_leverage
            self.max_leverage = getattr(leverage_config, 'max_leverage', 20)
            # FUT-RM-07: post-fill ISOLATED-margin verification toggle
            self.enforce_isolated_margin = getattr(
                leverage_config, 'enforce_isolated_margin', True
            )
            # FUT-RM-07b (Wave 4): emit a high-priority Telegram alert when
            # the FUT-RM-07 path fires an emergency-close. Fail-soft if
            # Telegram is not configured (just logs).
            self.telegram_emergency_close_enabled = getattr(
                leverage_config, 'telegram_emergency_close_enabled', True
            )

            # Risk settings - SL/TP as price percentages
            self.stop_loss_pct = abs(risk_config.stop_loss_pct)  # Store as positive
            self.take_profit_pct = abs(risk_config.take_profit_pct)  # Legacy single TP

            # Multiple Take Profit settings
            self.tp1_pct = getattr(risk_config, 'tp1_pct', 2.0)
            self.tp2_pct = getattr(risk_config, 'tp2_pct', 4.0)
            self.tp3_pct = getattr(risk_config, 'tp3_pct', 6.0)
            self.tp4_pct = getattr(risk_config, 'tp4_pct', 10.0)
            self.tp1_size_pct = getattr(risk_config, 'tp1_size_pct', 25.0)
            self.tp2_size_pct = getattr(risk_config, 'tp2_size_pct', 25.0)
            self.tp3_size_pct = getattr(risk_config, 'tp3_size_pct', 25.0)
            self.tp4_size_pct = getattr(risk_config, 'tp4_size_pct', 25.0)

            # Trailing stop
            self.trailing_stop_enabled = getattr(risk_config, 'trailing_stop_enabled', True)
            self.trailing_stop_distance = getattr(risk_config, 'trailing_stop_distance', 1.0)

            # Market condition filters (relaxed defaults for live trading)
            self.require_trend_confirmation = getattr(risk_config, 'require_trend_confirmation', False)
            self.min_volume_multiplier = getattr(risk_config, 'min_volume_multiplier', 0.8)
            # FUT-RM-21 (Wave 7): hard-block counter-trend entries by regime.
            self.block_counter_trend_entries = bool(getattr(
                risk_config, 'block_counter_trend_entries', True))

            # FUT-RM-10 (Wave 3): auto-deleverage on drawdown.
            self.auto_deleverage_enabled = bool(getattr(
                risk_config, 'auto_deleverage_enabled', False))
            self.auto_deleverage_cooldown_seconds = int(getattr(
                risk_config, 'auto_deleverage_cooldown_seconds', 600))
            self._auto_deleverage_last_at = None  # datetime of last trigger

            # FUT-RM-16 (Wave 5): ATR-scaled SL/TP. When enabled, the engine
            # uses signals.atr_pct at entry time to size SL/TP distances per
            # symbol's actual volatility (see _open_position).
            self.atr_dynamic_sl_tp_enabled = bool(getattr(
                risk_config, 'atr_dynamic_sl_tp_enabled', True))
            self.atr_sl_multiplier = float(getattr(
                risk_config, 'atr_sl_multiplier', 1.5))
            self.atr_sl_min_pct = float(getattr(
                risk_config, 'atr_sl_min_pct', 1.5))
            self.atr_tp_rr_ratio = float(getattr(
                risk_config, 'atr_tp_rr_ratio', 2.0))

            # FUT-RM-23 (Wave 14): intraday max-hold cap (0 = disabled).
            self.max_hold_minutes = int(getattr(risk_config, 'max_hold_minutes', 240))
            # FUT-RM-24 (Wave 14): signal-reversal score threshold for early exit.
            self.signal_reversal_threshold = int(getattr(risk_config, 'signal_reversal_threshold', 4))

            # Calculate max_daily_loss_usd from percentage and capital
            # If max_daily_loss_pct is set (from UI), use that. Otherwise use max_daily_loss_usd directly.
            if risk_config.max_daily_loss_pct and risk_config.max_daily_loss_pct > 0:
                self.max_daily_loss = self.capital_allocation * (risk_config.max_daily_loss_pct / 100)
                logger.info(f"Daily loss limit: ${self.max_daily_loss:.2f} ({risk_config.max_daily_loss_pct}% of ${self.capital_allocation})")
            else:
                self.max_daily_loss = risk_config.max_daily_loss_usd
                logger.info(f"Daily loss limit: ${self.max_daily_loss:.2f} (fixed USD)")

            # Wave-25: consecutive-loss circuit breaker count — was hard-coded
            # to 5 in RiskMetrics.can_trade while the DB default says 4.
            self.max_consecutive_losses = int(getattr(
                risk_config, 'max_consecutive_losses', 5) or 5)

            self.cooldown_duration = timedelta(minutes=strategy_config.cooldown_minutes)

            # Pairs settings
            self.symbols = pairs_config.pairs_list

            # Strategy settings
            self.signal_timeframe = strategy_config.signal_timeframe
            self.scan_interval_seconds = strategy_config.scan_interval_seconds
            self.rsi_oversold = strategy_config.rsi_oversold
            self.rsi_overbought = strategy_config.rsi_overbought
            self.rsi_weak_oversold = strategy_config.rsi_weak_oversold
            self.rsi_weak_overbought = strategy_config.rsi_weak_overbought
            self.min_signal_score = strategy_config.min_signal_score
            # FUT-RM-15 (Wave 5): multi-indicator confluence gate count
            self.min_signal_confluence_count = int(getattr(
                strategy_config, 'min_signal_confluence_count', 2
            ))
            # FUT-RM-19 (Wave 7): fee + funding aware minimum-edge gate
            self.min_edge_gate_enabled = bool(getattr(
                strategy_config, 'min_edge_gate_enabled', True))
            self.min_net_edge_pct = float(getattr(
                strategy_config, 'min_net_edge_pct', 0.30))
            self.edge_slippage_pct = float(getattr(
                strategy_config, 'edge_slippage_pct', 0.05))
            self.edge_funding_fallback_pct = float(getattr(
                strategy_config, 'edge_funding_fallback_pct', 0.05))
            # FUT-RM-20 (Wave 7): one-entry-per-candle throttle
            self.one_entry_per_candle = bool(getattr(
                strategy_config, 'one_entry_per_candle', True))
            # FUT-RM-26 (Wave 24): hard kill of NEW entries (neutralization).
            # When set, the engine monitors/exits existing positions but opens
            # no new momentum or carry positions. See migration 066 + report.
            self.entries_suppressed = bool(getattr(
                strategy_config, 'entries_suppressed', False))
            self.verbose_signals = strategy_config.verbose_signals

            # FUT-RM-25 (Wave 14): funding-rate carry config.
            # Optional; falls back to disabled if FuturesFundingConfig is absent.
            try:
                funding_config = config_manager.get_funding()
                self.funding_carry_enabled = bool(getattr(
                    funding_config, 'funding_carry_enabled', False))
                self.carry_min_funding_bps = float(getattr(
                    funding_config, 'carry_min_funding_bps', 8.0))
                self.carry_exit_funding_bps = float(getattr(
                    funding_config, 'carry_exit_funding_bps', 3.0))
                self.carry_max_positions = int(getattr(
                    funding_config, 'carry_max_positions', 2))
                self.carry_max_hold_minutes = int(getattr(
                    funding_config, 'carry_max_hold_minutes', 960))
                # FUT-QC-01 (carry v2): bidirectional stability-gated carry.
                # Separate flag — when false, ZERO behavior change.
                self.funding_carry_v2_enabled = bool(getattr(
                    funding_config, 'futures_funding_carry_enabled', False))
                self.carry_min_abs_funding_bps = float(getattr(
                    funding_config, 'carry_min_abs_funding_bps', 10.0))
                self.carry_funding_stability_window = int(getattr(
                    funding_config, 'carry_funding_stability_window', 3))
                self.carry_max_carry_positions = int(getattr(
                    funding_config, 'carry_max_carry_positions', 3))
            except Exception:
                # FuturesFundingConfig absent in very old DB seeds; disable carry.
                self.funding_carry_enabled = False
                self.carry_min_funding_bps = 8.0
                self.carry_exit_funding_bps = 3.0
                self.carry_max_positions = 2
                self.carry_max_hold_minutes = 960
                self.funding_carry_v2_enabled = False
                self.carry_min_abs_funding_bps = 10.0
                self.carry_funding_stability_window = 3
                self.carry_max_carry_positions = 3

        else:
            # Fallback to defaults if no config manager (should not happen in production)
            logger.warning("⚠️ No config manager provided, using defaults")
            self.exchange = "binance"
            self.testnet = True
            self.max_positions = 5
            self.position_size_usd = 100.0
            # FUT-RM-18 (Wave 5): default lowered from 10x to 5x; see migration 031.
            self.leverage = 5
            # FUT-RM-06 / FUT-RM-07 fallback defaults
            self.atr_sizing_enabled = False
            self.atr_risk_pct = 1.0
            self.atr_stop_multiplier = 1.5
            self.enforce_isolated_margin = True
            self.telegram_emergency_close_enabled = True
            # FUT-RM-16 (Wave 5): ATR-scaled SL/TP fallback defaults
            self.atr_dynamic_sl_tp_enabled = True
            self.atr_sl_multiplier = 1.5
            self.atr_sl_min_pct = 1.5
            self.atr_tp_rr_ratio = 2.0
            self.stop_loss_pct = -5.0
            self.take_profit_pct = 10.0
            self.max_daily_loss = 500.0
            self.max_consecutive_losses = 5
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
            self.signal_timeframe = "15m"  # 15-minute candles for faster signals
            self.scan_interval_seconds = 30  # Scan every 30 seconds
            self.rsi_oversold = 30.0
            self.rsi_overbought = 70.0
            self.rsi_weak_oversold = 40.0
            self.rsi_weak_overbought = 60.0
            self.min_signal_score = 3  # Lower threshold for more signals
            # FUT-RM-15 (Wave 5): multi-indicator confluence gate (fallback)
            self.min_signal_confluence_count = 2
            # FUT-RM-19/20 (Wave 7) fallback defaults
            self.min_edge_gate_enabled = True
            self.min_net_edge_pct = 0.30
            self.edge_slippage_pct = 0.05
            self.edge_funding_fallback_pct = 0.05
            self.one_entry_per_candle = True
            # FUT-RM-26 (Wave 24) fallback default: entries enabled unless DB says otherwise
            self.entries_suppressed = False
            # FUT-RM-21 (Wave 7) fallback default
            self.block_counter_trend_entries = True
            self.require_trend_confirmation = False
            self.min_volume_multiplier = 0.8
            self.verbose_signals = True
            self.cooldown_duration = timedelta(minutes=5)
            # FUT-RM-23/24 fallback defaults
            self.max_hold_minutes = 240
            self.signal_reversal_threshold = 4
            # FUT-RM-25 fallback defaults
            self.funding_carry_enabled = False
            self.carry_min_funding_bps = 8.0
            self.carry_exit_funding_bps = 3.0
            self.carry_max_positions = 2
            self.carry_max_hold_minutes = 960
            # FUT-QC-01 (carry v2) fallback defaults — OFF
            self.funding_carry_v2_enabled = False
            self.carry_min_abs_funding_bps = 10.0
            self.carry_funding_stability_window = 3
            self.carry_max_carry_positions = 3

        # DRY_RUN already resolved at top of __init__ via core.dry_run.resolve_dry_run_env

        # Trading state
        self.active_positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.trade_history: List[Trade] = []
        # FUT-RM-25: track which symbols currently hold a carry-strategy position
        # so the carry scan can exit when funding falls and not double-enter.
        self._carry_position_symbols: set = set()

        # FUT-QC-01 (carry v2): rolling-window planner + per-symbol tracking.
        # Planner state is in-memory only; after a restart the stability
        # window must refill before a new v2 entry can arm (fail-safe: no
        # evidence -> no trade). Exits never depend on the window.
        from modules.futures_trading.strategies.funding_carry import (
            FundingCarryPlanner,
        )
        self._carry_v2_symbols: set = set()
        self._carry_planner = FundingCarryPlanner(
            min_abs_funding_bps=float(getattr(
                self, 'carry_min_abs_funding_bps', 10.0)),
            min_samples=max(2, int(getattr(
                self, 'carry_funding_stability_window', 3))),
            exit_abs_funding_bps=float(getattr(
                self, 'carry_exit_funding_bps', 3.0)),
        )

        # Reconcile observability (set by _sync_positions on startup)
        self.last_reconcile_at: Optional[datetime] = None
        self.last_reconcile_count: int = 0

        # Cooldowns (symbol -> next_trade_time)
        self.symbol_cooldowns: Dict[str, datetime] = {}

        # FUT-RM-20 (Wave 7): one-entry-per-candle throttle.
        # symbol -> the candle-open timestamp we last entered on. Prevents the
        # 30s scan loop from firing repeatedly into the same 15m bar.
        self._last_entry_candle: Dict[str, datetime] = {}

        # Exchange client
        self.exchange_client = None
        self.price_client = None  # Mainnet client for accurate prices in DRY_RUN mode

        # MB-17: optional cross-module risk validator (orchestrator wiring).
        # When None, no validation is performed (legacy/backward-compatible).
        self.risk_manager: Optional[Any] = None

        # Risk metrics
        self.risk_metrics = RiskMetrics(
            daily_loss_limit=self.max_daily_loss,
            max_consecutive_losses=getattr(self, 'max_consecutive_losses', 5),
            last_reset=datetime.now()
        )

        # Wave-25 log-spam throttles (operator complaint: FUT-RM-17 refusals
        # every ~35s per symbol for hours; pause warning every 5 min).
        # Pause: WARNING once per state transition, INFO reminder per 30 min.
        self._pause_state_key: Optional[str] = None
        self._pause_last_log_at: Optional[datetime] = None
        # Cool-off refusals: symbol -> {'expires_at': str, 'last_logged': dt}.
        self._cooloff_log_state: Dict[str, Dict] = {}
        self._risk_log_reminder_seconds: int = 1800  # 30 min

        # Price cache for simulated trading
        self._price_cache: Dict[str, Dict] = {}
        self._last_price_update: Dict[str, datetime] = {}

        # FUT-RM-05: funding-rate cache for directional gate.
        # symbol -> (fraction_rate, fetched_at). Pulled via ccxt
        # fetch_funding_rate; gated on staleness in _get_funding_rate_cached.
        self._funding_cache: Dict[str, tuple] = {}
        # Configurable in seconds; ccxt funding endpoints are heavy so
        # we re-poll every 5 minutes max (well below the 8h interval).
        self._funding_cache_ttl_seconds: int = 300
        # Wave-18: per-symbol throttle for "Funding gate refused entry"
        # WARNING. The scan loop runs every ~35s and the same symbol
        # re-triggers the gate every iteration while its funding rate
        # remains out of range. Throttle to at most once per 5 minutes
        # per symbol; subsequent refusals within the window are DEBUG.
        self._funding_gate_warned_at: Dict[str, float] = {}
        self._funding_gate_warn_interval_s: float = 300.0  # 5 minutes

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0

        # PnL Tracker for Sharpe/Sortino calculations
        self.pnl_tracker = PnLTracker(
            initial_capital=self.position_size_usd * self.max_positions,
            currency="USD"
        )

        # Telegram alerts - will be initialized in async initialize() method
        # where we can properly load credentials from secrets manager
        self.telegram_alerts = None

        # ISSUE-15 (Wave 7): non-sensitive identity of the API key in use, so
        # the operator can tell WHICH account is live without exposing secrets.
        # Set by _set_api_key_fingerprint() during exchange init. Format:
        # "****<last4>" (never the full key) or None when no key is loaded.
        self.api_key_fingerprint: Optional[str] = None

        # Logging mode info
        mode_str = "DRY_RUN (SIMULATED)" if self.dry_run else "LIVE TRADING"
        net_str = "TESTNET" if self.testnet else "MAINNET"
        logger.info(f"Futures engine initialized:")
        logger.info(f"  Exchange: {self.exchange.upper()}")
        logger.info(f"  Mode: {mode_str}")
        logger.info(f"  Network: {net_str}")
        logger.info(f"  Leverage: {self.leverage}x")
        logger.info(f"  Position size: ${self.position_size_usd}")
        logger.info(f"  Symbols: {', '.join(self.symbols)}")
        logger.info(f"  Strategy Settings:")
        logger.info(f"    Signal Timeframe: {self.signal_timeframe}")
        logger.info(f"    Scan Interval: {self.scan_interval_seconds}s")
        logger.info(f"    RSI Oversold (STRONG_BUY): < {self.rsi_oversold}")
        logger.info(f"    RSI Overbought (STRONG_SELL): > {self.rsi_overbought}")
        logger.info(f"    RSI Weak Oversold (BUY): < {self.rsi_weak_oversold}")
        logger.info(f"    RSI Weak Overbought (SELL): > {self.rsi_weak_overbought}")
        logger.info(f"    Min Signal Score: {self.min_signal_score} (4=STRONG only, 2=BUY/SELL)")
        logger.info(f"    Verbose Signals: {self.verbose_signals}")

    def set_risk_manager(self, risk_manager: Any) -> None:
        """Inject a FuturesRiskManager. Entry path will call validate_new_position
        before opening a position; rejection logs the reason and aborts entry."""
        self.risk_manager = risk_manager

    def _set_api_key_fingerprint(self, api_key: Optional[str]) -> None:
        """ISSUE-15: stash a NON-SENSITIVE fingerprint of the active API key
        (last 4 chars only) so the diagnostics/health surface can show which
        account is in use. Never stores or returns the full key."""
        try:
            if api_key and len(api_key) >= 4:
                self.api_key_fingerprint = f"****{api_key[-4:]}"
            elif api_key:
                self.api_key_fingerprint = "****"
            else:
                self.api_key_fingerprint = None
        except Exception:
            self.api_key_fingerprint = None

    async def initialize(self):
        """Initialize exchange connections and components"""
        try:
            logger.info(f"Initializing {self.exchange.upper()} connection...")

            # Initialize exchange client
            if self.exchange == "binance":
                await self._init_binance()
            elif self.exchange == "bybit":
                await self._init_bybit()
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange}")

            # Verify connection and set leverage for all symbols
            await self._setup_symbols()

            # Load existing positions from exchange (if any)
            await self._sync_positions()

            logger.info("✅ Exchange connection initialized")

            # Initialize Telegram alerts with async credential loading
            try:
                from futures_trading.core.futures_alerts import FuturesTelegramAlerts
                from security.secrets_manager import secrets

                # Pre-load Telegram credentials asynchronously
                bot_token = await secrets.get_async('TELEGRAM_BOT_TOKEN', log_access=False)
                chat_id = await secrets.get_async('TELEGRAM_CHAT_ID', log_access=False)

                self.telegram_alerts = FuturesTelegramAlerts(
                    bot_token=bot_token,
                    chat_id=chat_id
                )
                if self.telegram_alerts.enabled:
                    logger.info("✅ Telegram alerts enabled for Futures module")
            except Exception as e:
                logger.warning(f"Telegram alerts not available: {e}")

            # Load historical trade stats from database
            await self._load_stats_from_db()

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    async def _load_stats_from_db(self):
        """Load historical trade statistics from database"""
        if not self.db_pool:
            logger.debug("No database pool available, skipping stats load")
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Load trade statistics
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COALESCE(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END), 0) as winning_trades,
                        COALESCE(SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END), 0) as losing_trades,
                        COALESCE(SUM(net_pnl), 0) as total_pnl,
                        COALESCE(SUM(fees), 0) as total_fees
                    FROM futures_trades
                    WHERE is_simulated = $1
                      AND exchange = $2
                      AND network = $3
                """, self.dry_run, self.exchange, 'testnet' if self.testnet else 'mainnet')

                if row and row['total_trades'] > 0:
                    self.total_trades = row['total_trades']
                    self.winning_trades = row['winning_trades']
                    self.losing_trades = row['losing_trades']
                    self.total_pnl = float(row['total_pnl'])
                    self.total_fees = float(row['total_fees'])
                    logger.info(f"📊 Loaded trade history from DB: {self.total_trades} trades, {self.winning_trades} wins, ${self.total_pnl:.2f} P&L")

                # Load today's stats for risk management
                today_row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as daily_trades,
                        COALESCE(SUM(net_pnl), 0) as daily_pnl
                    FROM futures_trades
                    WHERE is_simulated = $1
                      AND exchange = $2
                      AND network = $3
                      AND DATE(exit_time) = CURRENT_DATE
                """, self.dry_run, self.exchange, 'testnet' if self.testnet else 'mainnet')

                if today_row:
                    self.risk_metrics.daily_trades = today_row['daily_trades']
                    self.risk_metrics.daily_pnl = float(today_row['daily_pnl'])

        except Exception as e:
            logger.warning(f"Could not load stats from DB: {e}")

    async def warm_symbol_gate_from_db(self):
        """FUT-RM-27 (Wave 25): seed the rolling per-symbol performance gate
        from persisted trades so it is effective immediately after a restart
        (otherwise every symbol gets rolling_gate_min_trades free trades).
        Called by main_futures AFTER set_risk_manager. Fail-soft."""
        if not self.db_pool or self.risk_manager is None or not hasattr(
            self.risk_manager, 'seed_symbol_history'
        ):
            return
        try:
            window = int(getattr(self.risk_manager, 'rolling_gate_window', 20))
            if window <= 0:
                return
            async with self.db_pool.acquire() as conn:
                # Last `window` closed trades per symbol, replayed oldest-
                # first so the deque ends in live order. Same identity
                # filters as _load_stats_from_db.
                rows = await conn.fetch("""
                    SELECT symbol, net_pnl FROM (
                        SELECT symbol, net_pnl, exit_time,
                               ROW_NUMBER() OVER (
                                   PARTITION BY symbol
                                   ORDER BY exit_time DESC
                               ) AS rn
                        FROM futures_trades
                        WHERE is_simulated = $1
                          AND exchange = $2
                          AND network = $3
                    ) t
                    WHERE rn <= $4
                    ORDER BY symbol, exit_time ASC
                """, self.dry_run, self.exchange,
                    'testnet' if self.testnet else 'mainnet', window)
            by_symbol: Dict[str, List[float]] = {}
            for row in rows:
                by_symbol.setdefault(row['symbol'], []).append(
                    float(row['net_pnl'])
                )
            for symbol, pnls in by_symbol.items():
                self.risk_manager.seed_symbol_history(symbol, pnls)
            if by_symbol:
                logger.info(
                    f"FUT-RM-27 rolling gate warmed from DB: "
                    f"{len(by_symbol)} symbols, window={window}"
                )
        except Exception as e:
            logger.warning(f"warm_symbol_gate_from_db failed (non-fatal): {e}")

    async def _save_trade_to_db(self, trade: Trade):
        """Save a closed trade to the database"""
        if not self.db_pool:
            logger.debug("No database pool available, trade not persisted")
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO futures_trades (
                        symbol, side, entry_price, exit_price, size, notional_value,
                        leverage, pnl, pnl_pct, fees, net_pnl, exit_reason,
                        entry_time, exit_time, duration_seconds, is_simulated,
                        exchange, network
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                    trade.symbol,
                    trade.side.value,
                    trade.entry_price,
                    trade.exit_price,
                    trade.size,
                    trade.notional_value,
                    trade.leverage,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.fees,
                    trade.pnl - trade.fees,  # net_pnl
                    trade.close_reason,
                    trade.opened_at,
                    trade.closed_at,
                    int((trade.closed_at - trade.opened_at).total_seconds()),
                    trade.is_simulated,
                    self.exchange,
                    'testnet' if self.testnet else 'mainnet'
                )
                logger.debug(f"💾 Trade saved to DB: {trade.symbol} {trade.side.value} P&L: ${trade.pnl:.2f}")
        except Exception as e:
            logger.error(f"Failed to save trade to DB: {e}")

    @staticmethod
    def _ccxt_transient_error_types() -> tuple:
        """Return ccxt transient-error base classes, importing lazily.

        Wave-15: used by _load_markets_with_retry to distinguish network
        hiccups (retryable) from auth errors (fail-fast).
        """
        try:
            import ccxt.base.errors as _ccxt_errs
            return (
                _ccxt_errs.RequestTimeout,
                _ccxt_errs.NetworkError,
                _ccxt_errs.ExchangeNotAvailable,
                _ccxt_errs.DDoSProtection,
            )
        except Exception:
            return ()

    @staticmethod
    async def _load_markets_with_retry(client, label: str, max_attempts: int = 5) -> None:
        """Call client.load_markets() with exponential backoff on transient errors.

        Wave-15 fix: Binance fapi/v1/exchangeInfo can time out during brief
        network hiccups on startup. Retry sequence: 2 s, 4 s, 8 s, 16 s, 32 s
        (62 s total wait before giving up).

        Only ccxt timeout / network errors are retried. Auth errors
        (InvalidNonce, AuthenticationError, etc.) propagate immediately so the
        operator sees the real problem quickly.
        """
        transient = FuturesTradingEngine._ccxt_transient_error_types()
        delays = [2, 4, 8, 16, 32]
        last_exc: Exception = RuntimeError("load_markets never attempted")
        for attempt in range(1, max_attempts + 1):
            try:
                await client.load_markets()
                logger.info(
                    f"Loaded {len(client.markets)} markets [{label}]"
                    + (f" (attempt {attempt}/{max_attempts})" if attempt > 1 else "")
                )
                return
            except Exception as exc:
                last_exc = exc
                if transient and isinstance(exc, transient):
                    wait = delays[min(attempt - 1, len(delays) - 1)]
                    logger.warning(
                        f"Binance load_markets transient error [{label}] "
                        f"(attempt {attempt}/{max_attempts}): "
                        f"{type(exc).__name__}: {exc} — retrying in {wait}s"
                    )
                    if attempt < max_attempts:
                        await asyncio.sleep(wait)
                else:
                    # Non-transient (auth, bad config): fail immediately.
                    logger.error(
                        f"Binance load_markets non-transient error [{label}]: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise
        # All retries exhausted.
        logger.error(
            f"Binance load_markets failed after {max_attempts} attempts [{label}]: {last_exc}"
        )
        raise last_exc

    async def _init_binance(self):
        """Initialize Binance Futures client"""
        try:
            import ccxt.async_support as ccxt

            # Get API credentials from config manager (reads from .env)
            if self.config_manager:
                credentials = self.config_manager.get_api_credentials('binance', self.testnet)
                api_key = credentials.get('api_key')
                api_secret = credentials.get('api_secret')
            else:
                # Fallback to secrets manager then direct env read
                # Use get_async() since we're in async context
                try:
                    from security.secrets_manager import secrets
                    if self.testnet:
                        api_key = await secrets.get_async('BINANCE_TESTNET_API_KEY', log_access=False) or os.getenv('BINANCE_TESTNET_API_KEY')
                        api_secret = await secrets.get_async('BINANCE_TESTNET_API_SECRET', log_access=False) or os.getenv('BINANCE_TESTNET_API_SECRET')
                    else:
                        api_key = await secrets.get_async('BINANCE_API_KEY', log_access=False) or os.getenv('BINANCE_API_KEY')
                        api_secret = await secrets.get_async('BINANCE_API_SECRET', log_access=False) or os.getenv('BINANCE_API_SECRET')
                except Exception:
                    if self.testnet:
                        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
                        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
                    else:
                        api_key = os.getenv('BINANCE_API_KEY')
                        api_secret = os.getenv('BINANCE_API_SECRET')

            sandbox_mode = self.testnet
            if self.testnet:
                logger.info("Using Binance TESTNET")
            else:
                logger.info("Using Binance MAINNET")

            if not api_key or not api_secret:
                raise ValueError("BINANCE API keys required")

            # ISSUE-15: record masked fingerprint of the active key.
            self._set_api_key_fingerprint(api_key)

            # Wave-15: raise ccxt timeout from default 10 s to 30 s.
            # Datacenter -> Binance fapi round-trips can exceed 10 s under load.
            self.exchange_client = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })

            # Enable sandbox/testnet mode
            if sandbox_mode:
                self.exchange_client.set_sandbox_mode(True)

            # Wave-15: load_markets with retry + exponential backoff.
            await self._load_markets_with_retry(self.exchange_client, label='primary')

            # Create mainnet price client for accurate prices (especially in DRY_RUN mode)
            # This ensures we always get live prices from mainnet, regardless of testnet setting
            if should_skip_live(self.dry_run, module='futures', account=self.exchange) or self.testnet:
                try:
                    # Use mainnet credentials if available (via secrets manager), otherwise create public client
                    # Use get_async() since we're in async context
                    try:
                        from security.secrets_manager import secrets
                        mainnet_key = await secrets.get_async('BINANCE_API_KEY', log_access=False) or os.getenv('BINANCE_API_KEY')
                        mainnet_secret = await secrets.get_async('BINANCE_API_SECRET', log_access=False) or os.getenv('BINANCE_API_SECRET')
                    except Exception:
                        mainnet_key = os.getenv('BINANCE_API_KEY')
                        mainnet_secret = os.getenv('BINANCE_API_SECRET')

                    if mainnet_key and mainnet_secret:
                        self.price_client = ccxt.binance({
                            'apiKey': mainnet_key,
                            'secret': mainnet_secret,
                            'enableRateLimit': True,
                            'timeout': 30000,
                            'options': {
                                'defaultType': 'future',
                                'adjustForTimeDifference': True,
                            }
                        })
                    else:
                        # Public client without credentials (can still fetch prices)
                        self.price_client = ccxt.binance({
                            'enableRateLimit': True,
                            'timeout': 30000,
                            'options': {
                                'defaultType': 'future',
                            }
                        })

                    # Wave-15: also use retry for the price client.
                    await self._load_markets_with_retry(self.price_client, label='price-client')
                    logger.info("Mainnet price client initialized for accurate live prices")
                except Exception as e:
                    logger.warning(f"Could not initialize mainnet price client: {e}")
                    self.price_client = None

        except ImportError:
            logger.error("ccxt library not installed. Install: pip install ccxt")
            raise
        except Exception as e:
            logger.error(f"Binance initialization failed: {e}")
            raise

    async def _init_bybit(self):
        """Initialize Bybit Futures client"""
        try:
            import ccxt.async_support as ccxt

            # Get API credentials from config manager (reads from .env)
            if self.config_manager:
                credentials = self.config_manager.get_api_credentials('bybit', self.testnet)
                api_key = credentials.get('api_key')
                api_secret = credentials.get('api_secret')
            else:
                # Fallback to secrets manager then direct env read
                # Use get_async() since we're in async context
                try:
                    from security.secrets_manager import secrets
                    if self.testnet:
                        api_key = await secrets.get_async('BYBIT_TESTNET_API_KEY', log_access=False) or os.getenv('BYBIT_TESTNET_API_KEY')
                        api_secret = await secrets.get_async('BYBIT_TESTNET_API_SECRET', log_access=False) or os.getenv('BYBIT_TESTNET_API_SECRET')
                    else:
                        api_key = await secrets.get_async('BYBIT_API_KEY', log_access=False) or os.getenv('BYBIT_API_KEY')
                        api_secret = await secrets.get_async('BYBIT_API_SECRET', log_access=False) or os.getenv('BYBIT_API_SECRET')
                except Exception:
                    if self.testnet:
                        api_key = os.getenv('BYBIT_TESTNET_API_KEY')
                        api_secret = os.getenv('BYBIT_TESTNET_API_SECRET')
                    else:
                        api_key = os.getenv('BYBIT_API_KEY')
                        api_secret = os.getenv('BYBIT_API_SECRET')

            sandbox_mode = self.testnet
            if self.testnet:
                logger.info("Using Bybit TESTNET")
            else:
                logger.info("Using Bybit MAINNET")

            if not api_key or not api_secret:
                raise ValueError("BYBIT API keys required")

            # ISSUE-15: record masked fingerprint of the active key.
            self._set_api_key_fingerprint(api_key)

            self.exchange_client = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear',  # USDT perpetuals
                }
            })

            if sandbox_mode:
                self.exchange_client.set_sandbox_mode(True)

            await self.exchange_client.load_markets()
            logger.info(f"✅ Loaded {len(self.exchange_client.markets)} markets")

        except ImportError:
            logger.error("ccxt library not installed. Install: pip install ccxt")
            raise
        except Exception as e:
            logger.error(f"Bybit initialization failed: {e}")
            raise

    async def _setup_symbols(self):
        """Setup leverage and margin mode for all trading symbols"""
        for symbol in self.symbols:
            try:
                if symbol not in self.exchange_client.markets:
                    logger.warning(f"Symbol {symbol} not found on exchange, skipping")
                    continue

                # Set leverage (only in non-dry-run mode or testnet)
                if not should_skip_live(self.dry_run, module='futures', account=self.exchange) or self.testnet:
                    try:
                        await self.exchange_client.set_leverage(self.leverage, symbol)
                        logger.info(f"✅ Set {symbol} leverage to {self.leverage}x")
                    except Exception as e:
                        # Some exchanges don't support setting leverage via API
                        logger.warning(f"Could not set leverage for {symbol}: {e}")

            except Exception as e:
                logger.error(f"Error setting up {symbol}: {e}")

    async def _sync_positions(self):
        """Sync positions from exchange"""
        if should_skip_live(self.dry_run, module='futures', account=self.exchange):
            self.last_reconcile_at = datetime.now()
            self.last_reconcile_count = 0
            logger.info("DRY_RUN mode: position reconcile skipped (last_reconcile_at set; active_positions left empty)")
            return

        seeded = 0
        filtered_zero = 0
        try:
            positions = await self.exchange_client.fetch_positions()
            for pos in positions:
                try:
                    if not pos.get('contracts') or float(pos['contracts']) <= 0:
                        filtered_zero += 1
                        continue
                    symbol = pos['symbol']
                    side = TradeSide.LONG if pos['side'] == 'long' else TradeSide.SHORT

                    position = Position(
                        position_id=str(uuid.uuid4()),
                        symbol=symbol,
                        side=side,
                        entry_price=float(pos['entryPrice']),
                        current_price=float(pos['markPrice']),
                        size=float(pos['contracts']),
                        notional_value=float(pos['notional']),
                        leverage=int(pos['leverage']),
                        unrealized_pnl=float(pos['unrealizedPnl']),
                        liquidation_price=float(pos['liquidationPrice']) if pos['liquidationPrice'] else None,
                        is_simulated=False
                    )
                    self.active_positions[symbol] = position
                    seeded += 1
                    logger.info(f"📊 Synced position: {symbol} {side.value} @ {position.entry_price}")
                except Exception as per_entry_err:
                    logger.warning(f"Skipping malformed position entry during reconcile: {per_entry_err}")
                    continue

            self.last_reconcile_at = datetime.now()
            self.last_reconcile_count = seeded
            logger.info(
                f"📊 Position reconcile: {seeded} seeded, {filtered_zero} flat-filtered, "
                f"last_reconcile_at={self.last_reconcile_at.isoformat()}"
            )

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            return

        # Restart-time over-cap detection (fail-soft).
        if self.risk_manager is not None and seeded > 0:
            try:
                current_positions_list = [
                    {'notional_value': p.notional_value, 'symbol': p.symbol}
                    for p in self.active_positions.values()
                ]
                capacity = self.risk_manager.check_reconciled_capacity(current_positions_list)
                if capacity.get('over_cap'):
                    logger.error(
                        f"🚨 RESTART OVER-CAP: reconciled {capacity['count']} positions but "
                        f"max_positions={capacity['max_positions']}. Engine will refuse new entries "
                        f"until count drops."
                    )
                elif capacity.get('at_cap'):
                    logger.warning(
                        f"⚠️ RESTART AT-CAP: reconciled {capacity['count']}/{capacity['max_positions']} "
                        f"positions. No room for new entries."
                    )
            except Exception as e:
                logger.warning(f"check_reconciled_capacity failed (non-fatal): {e}")

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        mode_str = "DRY_RUN (SIMULATED)" if self.dry_run else "LIVE TRADING"
        logger.info(f"🚀 Futures trading engine started - {mode_str}")

        try:
            while self.is_running:
                try:
                    # Reset daily metrics at midnight
                    await self._check_daily_reset()

                    # Main trading logic
                    await self._trading_cycle()

                    # Wait before next cycle (configurable scan interval)
                    await asyncio.sleep(self.scan_interval_seconds)

                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}", exc_info=True)
                    await asyncio.sleep(self.scan_interval_seconds * 3)  # Longer wait on error

        except Exception as e:
            logger.error(f"Critical error in trading engine: {e}", exc_info=True)
            raise

        finally:
            logger.info("Futures trading engine stopped")

    async def _check_daily_reset(self):
        """Reset daily metrics at midnight UTC"""
        now = datetime.utcnow()
        if now.date() > self.risk_metrics.last_reset.date():
            logger.info("📅 Resetting daily risk metrics")
            self.risk_metrics.daily_pnl = 0.0
            self.risk_metrics.daily_trades = 0
            self.risk_metrics.consecutive_losses = 0
            self.risk_metrics.last_reset = now

    def _log_pause_state(self, pause_reason: Optional[str]) -> None:
        """Wave-25 log-spam fix for the risk pause.

        Old behavior: 'Trading paused — daily risk limit reached' at WARNING
        every 5 min for the rest of the UTC day — and the text was WRONG when
        the actual trigger was the consecutive-loss breaker (observed paused
        at +$20.68 daily PnL). New behavior: WARNING once per state
        transition (with the true reason), INFO reminder every 30 min, and
        an INFO 'resumed' line when the pause clears."""
        try:
            now = datetime.now()
            if pause_reason is None:
                if self._pause_state_key is not None:
                    logger.info(
                        "✅ Trading entries resumed — risk limits back "
                        "within bounds"
                    )
                    self._pause_state_key = None
                    self._pause_last_log_at = None
                return
            # Coarse key so e.g. the streak count growing while paused does
            # not re-fire the transition WARNING.
            key = 'daily_loss' if 'daily LOSS' in pause_reason else 'breaker'
            if key != self._pause_state_key:
                logger.warning(
                    f"⏸️ Trading entries paused — {pause_reason}. Open "
                    f"positions are still monitored/exited. Reminders every "
                    f"{self._risk_log_reminder_seconds // 60} min at INFO."
                )
                self._pause_state_key = key
                self._pause_last_log_at = now
            elif (
                self._pause_last_log_at is None
                or (now - self._pause_last_log_at).total_seconds()
                >= self._risk_log_reminder_seconds
            ):
                logger.info(f"⏸️ Trading entries still paused — {pause_reason}")
                self._pause_last_log_at = now
        except Exception as e:
            logger.debug(f"_log_pause_state error (non-fatal): {e}")

    def _log_cooloff_refusal(self, symbol: str, cgate: Dict) -> None:
        """Wave-25 log-spam fix for FUT-RM-17 refusals: the 30s scan re-hits
        an armed cool-off every cycle, repeating the same WARNING for hours.
        Now: INFO once per cool-off instance (keyed on expires_at), INFO
        reminder every 30 min, DEBUG otherwise. The 'cool-off ARMED' WARNING
        in FuturesRiskManager is unchanged — that is the true transition."""
        reason = cgate.get('reason')
        try:
            now = datetime.now()
            key = str(cgate.get('expires_at') or '')
            prev = self._cooloff_log_state.get(symbol)
            if prev is None or prev.get('expires_at') != key:
                self._cooloff_log_state[symbol] = {
                    'expires_at': key, 'last_logged': now,
                }
                logger.info(
                    f"⏭️  FUT-RM-17 cool-off refused entry for {symbol}: "
                    f"{reason} (further refusals at DEBUG; "
                    f"{self._risk_log_reminder_seconds // 60}-min reminders "
                    f"at INFO)"
                )
            elif (
                now - prev['last_logged']
            ).total_seconds() >= self._risk_log_reminder_seconds:
                prev['last_logged'] = now
                logger.info(
                    f"⏭️  FUT-RM-17 cool-off still active for {symbol}: "
                    f"{reason}"
                )
            else:
                logger.debug(
                    f"FUT-RM-17 cool-off refused entry for {symbol}: {reason}"
                )
        except Exception:
            logger.debug(
                f"FUT-RM-17 cool-off refused entry for {symbol}: {reason}"
            )

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # 1. Check risk limits — ENTRY-ONLY pause (Wave-25 fix).
            # Pre-Wave-25 this early-returned BEFORE _monitor_positions, so a
            # daily-limit/consecutive-loss pause also stopped SL/TP/time-exit
            # monitoring of OPEN positions for the rest of the UTC day — the
            # one moment unmanaged exposure is most dangerous. Now a pause
            # only suppresses new entries + pending-order processing; open
            # positions are always monitored and exited.
            pause_reason = self.risk_metrics.pause_reason \
                if not self.risk_metrics.can_trade else None
            self._log_pause_state(pause_reason)

            # 2. Monitor existing positions (runs even while paused)
            await self._monitor_positions()

            if pause_reason is not None:
                return

            # FUT-RM-26 (Wave 24): entry suppression / neutralization gate.
            # When set, the engine still monitors and exits existing positions
            # (step 2 above and step 5 below) but opens NO new momentum or carry
            # positions. This is the futures equivalent of the wave-18 budget=0
            # neutralization for sniper/arb — see migration 066 + the wave-24
            # strategy review. Reversible: flip strategy.entries_suppressed back
            # to false (or delete the row) to re-enable entries.
            if getattr(self, 'entries_suppressed', False):
                _now = datetime.now()
                _last_supp = getattr(self, '_last_suppressed_log_at', None)
                if _last_supp is None or (_now - _last_supp).total_seconds() >= 300:
                    logger.warning(
                        "FUT-RM-26: new-entry suppression ACTIVE "
                        "(strategy.entries_suppressed=true) — monitoring/exiting "
                        "existing positions only, no new entries"
                    )
                    self._last_suppressed_log_at = _now
            else:
                # 3. Check for new opportunities
                await self._scan_opportunities()

                # 4. FUT-RM-25: funding-carry scan (parallel strategy, separate cap)
                if getattr(self, 'funding_carry_enabled', False):
                    await self._scan_funding_carry_opportunities()

                # 4b. FUT-QC-01: carry v2 — bidirectional, stability-gated.
                # Separate DB flag (futures_funding_carry_enabled); when off
                # this branch never runs and behavior is unchanged.
                if getattr(self, 'funding_carry_v2_enabled', False):
                    await self._scan_funding_carry_v2()

            # 5. Execute pending orders
            await self._process_orders()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _monitor_positions(self):
        """Monitor active positions for exit signals and partial TPs"""
        if not self.active_positions:
            return

        # FUT-RM-09: refresh the funding-cost snapshot for the widget.
        # Cheap: UNIQUE constraint upserts within the current hour bucket.
        try:
            await self._record_funding_snapshot()
        except Exception as snap_err:
            logger.debug(f"funding snapshot skipped: {snap_err}")

        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = await self._get_decision_price(symbol)
                if current_price is None:
                    continue
                position.current_price = current_price

                # Update high/low tracking for trailing stop
                if position.side == TradeSide.LONG:
                    if position.highest_price is None or current_price > position.highest_price:
                        position.highest_price = current_price
                        # Update trailing stop if active
                        if position.trailing_stop_active and self.trailing_stop_enabled:
                            new_trailing_stop = position.highest_price * (1 - self.trailing_stop_distance / 100)
                            # Only move trailing stop UP (never down)
                            if position.trailing_stop_price is None or new_trailing_stop > position.trailing_stop_price:
                                position.trailing_stop_price = new_trailing_stop
                                logger.info(f"📈 TSL updated {position.symbol}: ${new_trailing_stop:.4f} (high: ${position.highest_price:.4f})")
                else:  # SHORT
                    if position.lowest_price is None or current_price < position.lowest_price:
                        position.lowest_price = current_price
                        # Update trailing stop if active
                        if position.trailing_stop_active and self.trailing_stop_enabled:
                            new_trailing_stop = position.lowest_price * (1 + self.trailing_stop_distance / 100)
                            # Only move trailing stop DOWN (never up)
                            if position.trailing_stop_price is None or new_trailing_stop < position.trailing_stop_price:
                                position.trailing_stop_price = new_trailing_stop
                                logger.info(f"📉 TSL updated {position.symbol}: ${new_trailing_stop:.4f} (low: ${position.lowest_price:.4f})")

                # Calculate PnL
                if position.side == TradeSide.LONG:
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:  # SHORT
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                # Apply leverage to PnL percentage
                leveraged_pnl_pct = pnl_pct * position.leverage
                position.unrealized_pnl_pct = leveraged_pnl_pct

                # Calculate PnL in USD
                position.unrealized_pnl = position.notional_value * (pnl_pct / 100)

                # Check for partial TP hits first
                tp_hit = await self._check_tp_levels(position, current_price)
                if tp_hit:
                    # Partial close was executed, position may still be open
                    # Use tolerance for floating-point comparison
                    if position.size <= 1e-8:
                        # All TPs hit, fully closed - remove from active positions
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        continue

                # Check exit conditions (SL, liquidation protection, signal reversal)
                exit_reason = await self._check_exit_conditions(position)
                if exit_reason:
                    await self._close_position(symbol, exit_reason)

            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")

    async def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed and return reason"""
        leveraged_pnl_pct = position.unrealized_pnl_pct  # This is leveraged PnL %

        # Un-leverage to get actual price change percentage for SL/TP comparison
        # stop_loss_pct and take_profit_pct are price movement percentages (not leveraged)
        actual_price_change_pct = leveraged_pnl_pct / position.leverage if position.leverage > 0 else leveraged_pnl_pct

        # TRAILING STOP CHECK (if active and set)
        if position.trailing_stop_price is not None:
            if position.side == TradeSide.LONG:
                # For LONG: exit if price falls below trailing stop
                if position.current_price <= position.trailing_stop_price:
                    # Distinguish exit reasons:
                    # - TSL Hit: Active trailing stop triggered
                    # - Breakeven SL: Stop at entry price (after TP1)
                    # - SL Hit: Other stop levels
                    if position.trailing_stop_active:
                        return "TSL Hit"
                    elif abs(position.trailing_stop_price - position.entry_price) < 0.0001:
                        return "Breakeven SL"
                    else:
                        return "SL Hit"
            else:  # SHORT
                # For SHORT: exit if price rises above trailing stop
                if position.current_price >= position.trailing_stop_price:
                    if position.trailing_stop_active:
                        return "TSL Hit"
                    elif abs(position.trailing_stop_price - position.entry_price) < 0.0001:
                        return "Breakeven SL"
                    else:
                        return "SL Hit"

        # Standard stop loss check (if no trailing stop is set)
        # stop_loss_pct is stored as positive (e.g., 2.0)
        # SL triggers when price moves against position by >= stop_loss_pct
        # Example: if stop_loss_pct=2.0, SL triggers when price drops 2% (for LONG)
        # FUT-RM-16 (Wave 5): when ATR-scaled SL was applied at entry, the
        # position carries its own sl_pct in metadata so we respect the
        # per-symbol level instead of the engine-wide static one.
        effective_sl_pct = position.metadata.get('dynamic_sl_pct') or self.stop_loss_pct
        if position.trailing_stop_price is None and actual_price_change_pct <= -effective_sl_pct:
            return "SL Hit"

        # Take profit check (legacy single TP - only if NOT using multiple tp_levels)
        # Skip this check if position has tp_levels (handled by partial TPs)
        # TP triggers when price moves in favor by >= take_profit_pct
        if not position.tp_levels and actual_price_change_pct >= self.take_profit_pct:
            return "take_profit"

        # FUT-RM-25: carry-specific funding-drop exit.
        # A carry position's primary edge is funding; when funding drops below
        # the exit threshold there is no longer a reason to hold the SHORT and
        # the position should be exited to free capital for the next signal.
        carry_exit_bps = position.metadata.get('carry_exit_bps')
        if carry_exit_bps is not None and position.metadata.get('carry_trade'):
            try:
                rate = await self._get_funding_rate_cached(position.symbol)
                if rate is not None:
                    rate_bps = float(rate) * 10000.0
                    # FUT-QC-01: v2 carry is bidirectional. A LONG carry
                    # collects NEGATIVE funding, so its edge is gone when the
                    # rate rises above -exit_bps (the v1 `< exit_bps` test
                    # would close a LONG carry instantly). SHORT carry keeps
                    # the original v1 semantics.
                    if (
                        position.metadata.get('carry_v2')
                        and position.side == TradeSide.LONG
                    ):
                        if rate_bps > -float(carry_exit_bps):
                            logger.info(
                                f"[carry-v2] funding-decay exit: "
                                f"{position.symbol} LONG carry, funding "
                                f"{rate_bps:+.2f} bps > -{float(carry_exit_bps):.2f} bps"
                            )
                            return "carry_funding_dropped"
                    elif rate_bps < float(carry_exit_bps):
                        logger.info(
                            f"FUT-RM-25 carry funding-drop exit: {position.symbol} "
                            f"funding {rate_bps:.2f} bps < exit threshold "
                            f"{carry_exit_bps:.2f} bps"
                        )
                        return "carry_funding_dropped"
            except Exception:
                pass  # Fail open — keep position if rate unavailable

        # Liquidation protection (close at 80% of liquidation price)
        if position.liquidation_price:
            if position.side == TradeSide.LONG:
                liq_threshold = position.entry_price - (position.entry_price - position.liquidation_price) * 0.8
                if position.current_price <= liq_threshold:
                    return "liquidation_protection"
            else:
                liq_threshold = position.entry_price + (position.liquidation_price - position.entry_price) * 0.8
                if position.current_price >= liq_threshold:
                    return "liquidation_protection"

        # Check for signal reversal (only after minimum hold time and strong reversal)
        # Minimum hold time prevents premature exits on noise
        min_hold_seconds = 300  # 5 minutes minimum hold
        position_age = (datetime.now() - position.opened_at).total_seconds()

        # FUT-RM-23 (Wave 14): intraday max-hold cap.
        # 15m-signal trades drifting for hours bleed funding + fees with no
        # incremental edge. Exit via time_limit when max_hold_minutes is set,
        # the cap has elapsed, and the position has NOT yet hit TP1 (indicated
        # by trailing_stop_price still being None — TP1 hit moves stop to BE).
        # This is DRY_RUN-safe: _close_position simulates the close in paper mode.
        # FUT-RM-25: carry positions use carry_max_hold_minutes from metadata
        # instead of the global cap; carry DOES benefit from longer holds.
        carry_meta = position.metadata.get('carry_max_hold_minutes')
        if carry_meta is not None:
            max_hold_min = int(carry_meta)
        else:
            max_hold_min = int(getattr(self, 'max_hold_minutes', 240))
        if max_hold_min > 0:
            hold_elapsed_min = position_age / 60.0
            if hold_elapsed_min >= max_hold_min:
                # Only time-exit if TP1 has NOT been hit yet (no breakeven stop set)
                if position.trailing_stop_price is None:
                    logger.info(
                        f"FUT-RM-23 max-hold cap: {position.symbol} held "
                        f"{hold_elapsed_min:.0f}min >= {max_hold_min}min limit, "
                        f"no TP hit yet — time_limit exit"
                    )
                    return "time_limit"

        if position_age >= min_hold_seconds:
            signals = await self._get_technical_signals(position.symbol)
            if signals:
                # Calculate signal strength for reversal
                reversal_score = (
                    signals.rsi_signal.value +
                    signals.macd_signal.value +
                    signals.volume_signal.value +
                    signals.bb_signal.value +
                    signals.ema_signal.value
                )

                # FUT-RM-24 (Wave 14): configurable reversal threshold.
                # Pre-Wave-14 required score <= -6 which was practically
                # impossible (needed all 5 indicators at STRONG_BUY/SELL).
                # signal_reversal_threshold defaults to 4 = 2 strong reversals,
                # above the entry bar of 3 to avoid whipsaw exits.
                rev_thr = int(getattr(self, 'signal_reversal_threshold', 4))
                if position.side == TradeSide.LONG and reversal_score <= -rev_thr:
                    # Additional check: only if we're in profit or at breakeven
                    if leveraged_pnl_pct >= -1.0:  # Allow small loss
                        return "Signal"
                elif position.side == TradeSide.SHORT and reversal_score >= rev_thr:
                    if leveraged_pnl_pct >= -1.0:  # Allow small loss
                        return "Signal"

        return None

    async def _scan_opportunities(self):
        """Scan for new trading opportunities"""
        if len(self.active_positions) >= self.max_positions:
            if self.verbose_signals:
                logger.debug(f"⏸️ At max positions ({self.max_positions}), skipping scan")
            return  # Already at max positions

        if self.verbose_signals:
            logger.info(f"🔍 Scanning {len(self.symbols)} symbols for opportunities: {', '.join(self.symbols)}")

        for symbol in self.symbols:
            try:
                # Skip if already have position
                if symbol in self.active_positions:
                    if self.verbose_signals:
                        logger.debug(f"  {symbol}: Skipped - already have open position")
                    continue

                # Check cooldown
                if symbol in self.symbol_cooldowns:
                    if datetime.now() < self.symbol_cooldowns[symbol]:
                        remaining = (self.symbol_cooldowns[symbol] - datetime.now()).seconds
                        if self.verbose_signals:
                            logger.debug(f"  {symbol}: Skipped - cooldown ({remaining}s remaining)")
                        continue

                # FUT-RM-27 (Wave 25): per-symbol tier / rolling-gate skip.
                # Cheap, BEFORE the OHLCV fetch + signal math. The
                # authoritative gate (with transition logging) lives in
                # _open_position; this one just saves the scan work.
                if self.risk_manager is not None and hasattr(
                    self.risk_manager, 'should_skip_for_symbol'
                ):
                    try:
                        sgate = self.risk_manager.should_skip_for_symbol(symbol)
                        if sgate.get('skip'):
                            if self.verbose_signals:
                                logger.debug(
                                    f"  {symbol}: Skipped - FUT-RM-27 "
                                    f"{sgate.get('reason')}"
                                )
                            continue
                    except Exception as e:
                        logger.debug(
                            f"symbol gate non-fatal error for {symbol}: {e}"
                        )

                # FUT-RM-20 (Wave 7): one-entry-per-candle throttle. The 30s
                # scan loop re-evaluates the same 15m bar ~30 times; without
                # this we can re-enter the same chop repeatedly. Skip if we
                # already entered this symbol within the current candle.
                if getattr(self, 'one_entry_per_candle', True):
                    cur_candle = self._current_candle_open()
                    last_candle = self._last_entry_candle.get(symbol)
                    if last_candle is not None and last_candle == cur_candle:
                        if self.verbose_signals:
                            logger.debug(
                                f"  {symbol}: Skipped - already entered this candle "
                                f"({self.signal_timeframe})"
                            )
                        continue

                # Check if symbol exists on exchange
                if symbol not in self.exchange_client.markets:
                    if self.verbose_signals:
                        logger.warning(f"  {symbol}: Skipped - not found on exchange")
                    continue

                # Get technical signals
                signals = await self._get_technical_signals(symbol)
                if not signals:
                    if self.verbose_signals:
                        logger.debug(f"  {symbol}: Skipped - could not calculate signals")
                    continue

                # Calculate combined signal score (now 5 indicators)
                signal_score = (
                    signals.rsi_signal.value +
                    signals.macd_signal.value +
                    signals.volume_signal.value +
                    signals.bb_signal.value +
                    signals.ema_signal.value
                )

                # Log detailed signal analysis
                if self.verbose_signals:
                    logger.info(f"  📊 {symbol} Signal Analysis:")
                    logger.info(f"     RSI: {signals.rsi:.1f} → {signals.rsi_signal.name} ({signals.rsi_signal.value:+d})")
                    logger.info(f"     MACD: {signals.macd_histogram:.6f} → {signals.macd_signal.name} ({signals.macd_signal.value:+d})")
                    logger.info(f"     Volume: {signals.volume_ratio:.2f}x → {signals.volume_signal.name} ({signals.volume_signal.value:+d})")
                    logger.info(f"     Bollinger: {signals.bb_position} → {signals.bb_signal.name} ({signals.bb_signal.value:+d})")
                    logger.info(f"     EMA: {signals.ema_crossover} → {signals.ema_signal.name} ({signals.ema_signal.value:+d})")
                    logger.info(f"     Trend: {signals.trend}, 1h: {signals.price_change_1h:+.2f}%, 24h: {signals.price_change_24h:+.2f}%")
                    logger.info(f"     Combined Score: {signal_score:+d} (min required: ±{self.min_signal_score})")

                # Determine entry based on signals using configurable min_signal_score
                entry_side = None

                # Additional quality filters
                trend_ok = True
                volume_ok = True

                if self.require_trend_confirmation:
                    # For LONG: require uptrend
                    # For SHORT: require downtrend
                    if signal_score > 0 and signals.trend != "uptrend":
                        trend_ok = False
                        if self.verbose_signals:
                            logger.info(f"     ⚠️ Trend filter: Bullish signal but trend is {signals.trend}")
                    elif signal_score < 0 and signals.trend != "downtrend":
                        trend_ok = False
                        if self.verbose_signals:
                            logger.info(f"     ⚠️ Trend filter: Bearish signal but trend is {signals.trend}")

                # Wave-13 FUT-RM-22: volume_ok is DIAGNOSTIC ONLY — not a gate.
                # Live data shows volume_ratio of 0.17x–0.76x across 18 symbols;
                # a hard 0.80x block kills 100% of signals. Volume is already
                # factored into the signed score via volume_signal (+2/0/-2).
                # We log below-threshold events but never block entry.
                if hasattr(self, 'min_volume_multiplier') and self.min_volume_multiplier > 0:
                    if signals.volume_ratio < self.min_volume_multiplier:
                        volume_ok = False  # informational only — NOT blocking
                        if self.verbose_signals:
                            logger.info(
                                f"     ℹ️  Volume below threshold: {signals.volume_ratio:.2f}x "
                                f"(ref {self.min_volume_multiplier:.2f}x) — "
                                f"diagnostic only, entry not blocked"
                            )

                # FUT-RM-21 (Wave 7): regime gate. The signal stack mixes
                # mean-reversion (RSI extremes) with trend-following (BB
                # breakout, EMA) additively, so a bullish RSI bounce can clear
                # the score in a clear downtrend — catching a falling knife.
                # Hard-block counter-trend entries: no LONG in a downtrend
                # regime, no SHORT in an uptrend regime. `sideways` stays
                # tradeable both ways (range mean-reversion is legitimate).
                regime_ok = True
                if getattr(self, 'block_counter_trend_entries', True):
                    if signal_score > 0 and signals.trend == "downtrend":
                        regime_ok = False
                        if self.verbose_signals:
                            logger.info("     ⚠️ Regime gate: bullish signal but regime is DOWNTREND — blocked")
                    elif signal_score < 0 and signals.trend == "uptrend":
                        regime_ok = False
                        if self.verbose_signals:
                            logger.info("     ⚠️ Regime gate: bearish signal but regime is UPTREND — blocked")

                # MOMENTUM CONFIRMATION: Recent price must move in signal direction
                momentum_ok = True
                if signal_score > 0 and signals.price_change_1h < -0.5:
                    # Bullish signal but price falling - weak setup
                    momentum_ok = False
                    if self.verbose_signals:
                        logger.info(f"     ⚠️ Momentum filter: Bullish signal but 1h change {signals.price_change_1h:+.2f}%")
                elif signal_score < 0 and signals.price_change_1h > 0.5:
                    # Bearish signal but price rising - weak setup
                    momentum_ok = False
                    if self.verbose_signals:
                        logger.info(f"     ⚠️ Momentum filter: Bearish signal but 1h change {signals.price_change_1h:+.2f}%")

                # CONFLICTING SIGNALS CHECK: RSI and MACD must not strongly disagree
                signals_aligned = True
                # Strong RSI oversold + Strong MACD bearish = RSI says buy but MACD says sell
                if signals.rsi_signal == SignalStrength.STRONG_BUY and signals.macd_signal == SignalStrength.STRONG_SELL:
                    signals_aligned = False
                    if self.verbose_signals:
                        logger.info(f"     ⚠️ Conflicting signals: RSI oversold but MACD strongly bearish")
                # Strong RSI overbought + Strong MACD bullish = RSI says sell but MACD says buy
                elif signals.rsi_signal == SignalStrength.STRONG_SELL and signals.macd_signal == SignalStrength.STRONG_BUY:
                    signals_aligned = False
                    if self.verbose_signals:
                        logger.info(f"     ⚠️ Conflicting signals: RSI overbought but MACD strongly bullish")

                # FUT-RM-15 (Wave 5): multi-indicator CONFLUENCE gate.
                # Count how many of the 4 directional indicators
                # (RSI / MACD / Bollinger / EMA — volume is a confirmer, not
                # a direction-giver) agree with the prospective entry side.
                # We require >= self.min_signal_confluence_count agreement,
                # in addition to the existing signed signal_score >=
                # self.min_signal_score. This blocks single-indicator entries
                # that just happen to round above the score bar.
                confluence_min = int(getattr(self, 'min_signal_confluence_count', 0) or 0)
                directional = (
                    signals.rsi_signal.value,
                    signals.macd_signal.value,
                    signals.bb_signal.value,
                    signals.ema_signal.value,
                )
                bullish_confluence = sum(1 for v in directional if v > 0)
                bearish_confluence = sum(1 for v in directional if v < 0)
                confluence_ok = True
                if confluence_min > 0:
                    if signal_score > 0 and bullish_confluence < confluence_min:
                        confluence_ok = False
                        if self.verbose_signals:
                            logger.info(
                                f"     ⚠️ Confluence filter: only {bullish_confluence}/4 "
                                f"bullish indicators agree (min {confluence_min})"
                            )
                    elif signal_score < 0 and bearish_confluence < confluence_min:
                        confluence_ok = False
                        if self.verbose_signals:
                            logger.info(
                                f"     ⚠️ Confluence filter: only {bearish_confluence}/4 "
                                f"bearish indicators agree (min {confluence_min})"
                            )

                # All filters must pass — volume_ok intentionally excluded (diagnostic only)
                all_filters_ok = (
                    trend_ok and momentum_ok
                    and signals_aligned and confluence_ok and regime_ok
                )

                if signal_score >= self.min_signal_score and all_filters_ok:
                    entry_side = TradeSide.LONG
                    if self.verbose_signals:
                        logger.info(f"     ✅ LONG signal triggered (score {signal_score} >= {self.min_signal_score})")
                elif signal_score <= -self.min_signal_score and all_filters_ok:
                    entry_side = TradeSide.SHORT
                    if self.verbose_signals:
                        logger.info(f"     ✅ SHORT signal triggered (score {signal_score} <= -{self.min_signal_score})")
                else:
                    if self.verbose_signals:
                        if not all_filters_ok:
                            logger.info(
                                f"     ❌ REJECTED: Quality filters failed "
                                f"(trend={trend_ok}, momentum={momentum_ok}, "
                                f"aligned={signals_aligned}, confluence={confluence_ok}, regime={regime_ok})"
                                f" | volume_ratio={signals.volume_ratio:.2f}x [diagnostic]"
                            )
                        elif signal_score > 0:
                            logger.info(f"     ❌ REJECTED: Bullish but weak (score {signal_score} < {self.min_signal_score})")
                        elif signal_score < 0:
                            logger.info(f"     ❌ REJECTED: Bearish but weak (score {signal_score} > -{self.min_signal_score})")
                        else:
                            logger.info(f"     ❌ REJECTED: Neutral market conditions (score = 0)")

                if entry_side:
                    logger.info(f"📈 Signal detected: {symbol} {entry_side.value.upper()}")
                    logger.info(f"   RSI: {signals.rsi:.1f}, MACD: {signals.macd_histogram:.6f}, Volume: {signals.volume_ratio:.2f}x")

                    await self._open_position(symbol, entry_side, signals)

                    # Only open one position per cycle
                    break

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

    async def _scan_funding_carry_opportunities(self) -> None:
        """FUT-RM-25 (Wave 14): funding-rate carry strategy scan.

        Purpose
        -------
        Scan each configured symbol for extreme funding rates. When the
        per-interval rate exceeds carry_min_funding_bps, enter SHORT to
        collect the funding payment from long-side payers. This is a separate
        profit motive from the momentum/signal stack: we are not predicting
        price direction, we are harvesting a risk premium that exists because
        over-leveraged longs must pay to maintain their positions.

        Edge formula (working-rule per-interval):
            carry_edge = funding_rate * notional
            net_carry   = carry_edge - taker_fees_round_trip - expected_slippage
        At 8 bps: net_carry on $100 notional ≈ $0.08 - $0.06 - $0.05 = -$0.03
        per interval; the carry becomes positive after ~2–3 intervals. This
        is why the hold cap is 2 × funding period (960 min default).

        Risk controls
        -------------
        - Separate cap: carry_max_positions independent of max_positions so
          carry does not crowd out the momentum book.
        - Carries still go through the standard _open_position path so all
          existing risk gates (funding gate, risk validator, margin verify)
          apply; a synthetic `signals` object with neutral indicators is
          passed so ATR sizing and TP levels work normally.
        - Exit when funding drops below carry_exit_funding_bps: the carry
          advantage is gone and holding a naked SHORT purely on hope is not
          the strategy. This check runs during the position monitor via the
          metadata 'carry_exit_bps' key read in _check_exit_conditions.
        - DRY_RUN-safe: _open_position and _close_position both check
          should_skip_live before any live order.

        Operator instructions
        ---------------------
        Default OFF. Enable via DB: funding_carry_enabled=true in
        futures_funding config. Observe the dashboard funding-forecast widget
        (FUT-RM-09b) for at least 48h before enabling to confirm carry_min
        threshold is achievable in your market. Start with carry_max_positions=1.
        """
        try:
            carry_enabled = getattr(self, 'funding_carry_enabled', False)
            if not carry_enabled:
                return

            carry_max = int(getattr(self, 'carry_max_positions', 2))
            carry_min_bps = float(getattr(self, 'carry_min_funding_bps', 8.0))
            carry_exit_bps = float(getattr(self, 'carry_exit_funding_bps', 3.0))
            carry_max_hold = int(getattr(self, 'carry_max_hold_minutes', 960))

            # Count active carry positions
            current_carry_count = len(self._carry_position_symbols & set(self.active_positions.keys()))

            for symbol in self.symbols:
                try:
                    # 1. Exit check: if we hold a carry position and funding has dropped,
                    #    close it. This runs regardless of the carry-new-entry cap.
                    if symbol in self._carry_position_symbols and symbol in self.active_positions:
                        rate = await self._get_funding_rate_cached(symbol)
                        if rate is not None:
                            rate_bps = float(rate) * 10000.0  # fraction -> bps
                            if rate_bps < carry_exit_bps:
                                logger.info(
                                    f"FUT-RM-25 carry exit: {symbol} funding "
                                    f"{rate_bps:.2f} bps < exit threshold "
                                    f"{carry_exit_bps:.2f} bps — closing"
                                )
                                await self._close_position(symbol, "carry_funding_dropped")
                                self._carry_position_symbols.discard(symbol)
                        continue

                    # 2. Skip if already holding a non-carry position on this symbol
                    if symbol in self.active_positions:
                        continue

                    # 3. Carry cap check
                    if current_carry_count >= carry_max:
                        if self.verbose_signals:
                            logger.debug(
                                f"FUT-RM-25: at carry cap "
                                f"({current_carry_count}/{carry_max}), skipping {symbol}"
                            )
                        break

                    # 4. Fetch live funding rate
                    rate = await self._get_funding_rate_cached(symbol)
                    if rate is None:
                        continue
                    rate_bps = float(rate) * 10000.0  # fraction -> bps

                    if rate_bps < carry_min_bps:
                        if self.verbose_signals:
                            logger.debug(
                                f"FUT-RM-25: {symbol} funding {rate_bps:.2f} bps "
                                f"< entry threshold {carry_min_bps:.2f} bps — skip"
                            )
                        continue

                    # 5. Positive funding: LONG pays SHORT. Enter SHORT.
                    logger.info(
                        f"FUT-RM-25 carry signal: {symbol} funding "
                        f"{rate_bps:.2f} bps >= {carry_min_bps:.2f} bps — "
                        f"entering SHORT to collect funding"
                    )

                    # Build a neutral TechnicalSignals so _open_position sizing
                    # and TP/SL logic work without indicator data. The carry
                    # edge is funding, not price direction — all indicator scores
                    # are intentionally neutral.
                    carry_signals = TechnicalSignals()
                    # Tag the position so _check_exit_conditions can carry-exit it
                    # and so _close_position updates _carry_position_symbols.
                    carry_signals_meta = {
                        'carry_trade': True,
                        'entry_funding_bps': rate_bps,
                        'carry_exit_bps': carry_exit_bps,
                        'carry_max_hold_minutes': carry_max_hold,
                    }

                    # Open the carry position. _open_position handles all gates:
                    # risk validator, margin verify, DRY_RUN, FUT-RM-19 edge gate.
                    # We temporarily stash carry metadata so _open_position can
                    # inject it into position.metadata via the signals object.
                    carry_signals.carry_metadata = carry_signals_meta  # type: ignore[attr-defined]
                    await self._open_position(symbol, TradeSide.SHORT, carry_signals)

                    # Mark this symbol as a carry position if it opened
                    if symbol in self.active_positions:
                        pos = self.active_positions[symbol]
                        pos.metadata['carry_trade'] = True
                        pos.metadata['entry_funding_bps'] = rate_bps
                        pos.metadata['carry_exit_bps'] = carry_exit_bps
                        # Override max_hold for carry positions if configured
                        if carry_max_hold > 0:
                            pos.metadata['carry_max_hold_minutes'] = carry_max_hold
                        self._carry_position_symbols.add(symbol)
                        current_carry_count += 1
                        logger.info(
                            f"FUT-RM-25: carry position opened for {symbol} "
                            f"(carry positions: {current_carry_count}/{carry_max})"
                        )

                except Exception as sym_err:
                    logger.error(f"FUT-RM-25 error scanning {symbol}: {sym_err}")

        except Exception as e:
            logger.error(f"FUT-RM-25 _scan_funding_carry_opportunities error: {e}")

    async def _scan_funding_carry_v2(self) -> None:
        """FUT-QC-01 (carry v2): bidirectional, stability-gated funding carry.

        Delegates ALL entry/exit math to the pure-logic FundingCarryPlanner
        (modules/futures_trading/strategies/funding_carry.py — edge thesis,
        cost model, and self-test live there). This method only does I/O:

        1. Record one funding sample per symbol per cycle (the planner's
           300s spacing gate dedupes the 300s-TTL cached rate, so a sample
           is only appended when the cache has actually refreshed).
        2. Exit: when a v2-held position's live rate no longer pays our side
           by exit_abs_funding_bps (bidirectional — unlike the v1 metadata
           check which is SHORT-only).
        3. Entry: when the planner's stability gate arms (>= min_samples
           samples, >= 80% window span, same sign, EVERY sample beyond
           carry_min_abs_funding_bps), enter the collecting side via
           _open_position — which runs every existing risk gate unchanged:
           FUT-RM-27 symbol tiering + rolling gate, FUT-RM-19 edge gate,
           FUT-RM-17 cool-off, FUT-RM-05 funding gate (which always favors
           the carry direction), FuturesRiskManager.validate_new_position,
           and should_skip_live (DRY_RUN / killswitch / pause).

        Caps + holds: carry_max_carry_positions bounds the v2 book
        (independent of v1 + momentum caps); carry_max_hold_minutes is
        stamped into position metadata and honored by FUT-RM-23/25 in
        _check_exit_conditions. DRY_RUN measurability: every entry logs a
        `[carry-v2]` line with the expected net carry at 1 and 2 intervals.
        """
        try:
            carry_max = int(getattr(self, 'carry_max_carry_positions', 3))
            carry_max_hold = int(getattr(self, 'carry_max_hold_minutes', 960))
            planner = self._carry_planner
            now = datetime.utcnow()

            current_count = len(
                self._carry_v2_symbols & set(self.active_positions.keys()))

            for symbol in self.symbols:
                try:
                    rate = await self._get_funding_rate_cached(symbol)
                    rate_bps = float(rate) * 10000.0 if rate is not None else None

                    # 1. Always feed the window (even at cap / while holding)
                    # so the stability evidence stays warm.
                    if rate_bps is not None:
                        planner.record(symbol, rate_bps, now)

                    # 2. Exit check for v2-held positions (bidirectional).
                    if symbol in self._carry_v2_symbols and symbol in self.active_positions:
                        pos = self.active_positions[symbol]
                        side_name = pos.side.value.upper() if hasattr(
                            pos.side, 'value') else str(pos.side).upper()
                        do_exit, why = planner.should_exit(side_name, rate_bps)
                        if do_exit:
                            logger.info(
                                f"[carry-v2] exit {symbol} ({side_name}): {why}")
                            await self._close_position(
                                symbol, "carry_v2_funding_decayed")
                            self._carry_v2_symbols.discard(symbol)
                            current_count = max(0, current_count - 1)
                        continue

                    # 3. Skip symbols already held by momentum or v1 carry.
                    if symbol in self.active_positions:
                        continue
                    if symbol in self._carry_position_symbols:
                        continue

                    # 4. v2 book cap.
                    if current_count >= carry_max:
                        continue

                    # 5. Stability-gated entry decision (pure logic).
                    decision = planner.evaluate_entry(symbol, now)
                    if not decision.enter:
                        if self.verbose_signals and decision.samples > 0:
                            logger.debug(
                                f"[carry-v2] {symbol} not armed: {decision.reason}")
                        continue

                    side = TradeSide.SHORT if decision.side == 'SHORT' else TradeSide.LONG
                    net_1 = planner.expected_net_carry_bps(decision.min_abs_bps, 1.0)
                    net_2 = planner.expected_net_carry_bps(decision.min_abs_bps, 2.0)
                    logger.info(
                        f"[carry-v2] ENTRY signal {symbol} {decision.side}: "
                        f"{decision.reason} (mean {decision.mean_bps:+.2f} bps, "
                        f"{decision.samples} samples / "
                        f"{decision.span_minutes:.0f} min span; expected net "
                        f"carry {net_1:+.1f} bps @1 interval, "
                        f"{net_2:+.1f} bps @2 intervals)"
                    )

                    # Neutral indicator object — the edge is funding, not
                    # price direction. _open_position runs all risk gates.
                    carry_signals = TechnicalSignals()
                    await self._open_position(symbol, side, carry_signals)

                    if symbol in self.active_positions:
                        pos = self.active_positions[symbol]
                        pos.metadata['carry_trade'] = True
                        pos.metadata['carry_v2'] = True
                        pos.metadata['carry_side'] = decision.side
                        pos.metadata['entry_funding_bps'] = decision.mean_bps
                        pos.metadata['carry_exit_bps'] = float(
                            planner.exit_abs_funding_bps)
                        if carry_max_hold > 0:
                            pos.metadata['carry_max_hold_minutes'] = carry_max_hold
                        self._carry_v2_symbols.add(symbol)
                        current_count += 1
                        logger.info(
                            f"[carry-v2] position opened {symbol} {decision.side} "
                            f"({current_count}/{carry_max} v2 carry positions)"
                        )

                except Exception as sym_err:
                    logger.error(f"[carry-v2] error scanning {symbol}: {sym_err}")

        except Exception as e:
            logger.error(f"[carry-v2] _scan_funding_carry_v2 error: {e}")

    async def _get_technical_signals(self, symbol: str) -> Optional[TechnicalSignals]:
        """Calculate technical indicators for a symbol using configurable timeframe"""
        try:
            # Use mainnet price_client for accurate live prices (especially in DRY_RUN/testnet mode)
            # This ensures we always get real market data, not testnet data
            client = self.price_client if self.price_client else self.exchange_client

            # Fetch OHLCV data using configured timeframe (default: 15m for faster signals)
            ohlcv = await client.fetch_ohlcv(symbol, self.signal_timeframe, limit=100)
            if len(ohlcv) < 50:
                return None

            closes = [candle[4] for candle in ohlcv]
            highs = [candle[2] for candle in ohlcv]
            lows = [candle[3] for candle in ohlcv]
            volumes = [candle[5] for candle in ohlcv]

            signals = TechnicalSignals()

            # Calculate RSI (14-period) using configurable thresholds
            rsi = self._calculate_rsi(closes, 14)
            signals.rsi = rsi
            if rsi < self.rsi_oversold:
                signals.rsi_signal = SignalStrength.STRONG_BUY
            elif rsi < self.rsi_weak_oversold:
                signals.rsi_signal = SignalStrength.BUY
            elif rsi > self.rsi_overbought:
                signals.rsi_signal = SignalStrength.STRONG_SELL
            elif rsi > self.rsi_weak_overbought:
                signals.rsi_signal = SignalStrength.SELL

            # Calculate MACD
            macd, signal_line, histogram = self._calculate_macd(closes)
            signals.macd = macd
            signals.macd_signal_line = signal_line
            signals.macd_histogram = histogram

            # Wave-7: score the histogram as a PERCENTAGE of price, not in raw
            # price units. The old absolute 0.001/0.005 thresholds were
            # price-scale dependent — they fired on almost every BTC bar and
            # almost no cheap-alt bar. hist_pct makes the same threshold mean
            # the same momentum across all symbols. 0.02% = weak cross,
            # 0.08% = strong momentum (tuned to the prior majors behaviour).
            last_close_macd = closes[-1] if closes else 0.0
            hist_pct = (histogram / last_close_macd * 100.0) if last_close_macd > 0 else 0.0
            macd_weak_thr = 0.02   # % of price for BUY/SELL
            macd_strong_thr = 0.08  # % of price for STRONG_BUY/STRONG_SELL

            if histogram > 0 and macd > signal_line:
                signals.macd_signal = SignalStrength.BUY if hist_pct > macd_weak_thr else SignalStrength.NEUTRAL
            elif histogram < 0 and macd < signal_line:
                signals.macd_signal = SignalStrength.SELL if hist_pct < -macd_weak_thr else SignalStrength.NEUTRAL

            # Amplify MACD signal for strong momentum
            if abs(hist_pct) > macd_strong_thr:
                if histogram > 0 and macd > signal_line:
                    signals.macd_signal = SignalStrength.STRONG_BUY
                elif histogram < 0 and macd < signal_line:
                    signals.macd_signal = SignalStrength.STRONG_SELL

            # Calculate Volume ratio
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            signals.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            if signals.volume_ratio > 2.0:
                # High volume confirms the trend
                if signals.macd_signal.value > 0:
                    signals.volume_signal = SignalStrength.BUY
                elif signals.macd_signal.value < 0:
                    signals.volume_signal = SignalStrength.SELL

            # Calculate Bollinger Bands (20-period, 2 std dev)
            bb_period = 20
            if len(closes) >= bb_period:
                bb_sma = sum(closes[-bb_period:]) / bb_period
                variance = sum((x - bb_sma) ** 2 for x in closes[-bb_period:]) / bb_period
                bb_std = variance ** 0.5
                signals.bb_middle = bb_sma
                signals.bb_upper = bb_sma + (2 * bb_std)
                signals.bb_lower = bb_sma - (2 * bb_std)

                current_price = closes[-1]
                # Determine Bollinger position and signal
                # TREND-FOLLOWING logic (breakout/breakdown confirmation):
                # - Price breaking above upper band = strong momentum UP (bullish)
                # - Price breaking below lower band = strong momentum DOWN (bearish)
                # - Price above middle = bullish continuation
                # - Price below middle = bearish continuation
                if current_price >= signals.bb_upper:
                    signals.bb_position = "above_upper"
                    signals.bb_signal = SignalStrength.STRONG_BUY  # Breakout, strong bullish momentum
                elif current_price <= signals.bb_lower:
                    signals.bb_position = "below_lower"
                    signals.bb_signal = SignalStrength.STRONG_SELL  # Breakdown, strong bearish momentum
                elif current_price > signals.bb_middle:
                    signals.bb_position = "upper_half"
                    signals.bb_signal = SignalStrength.BUY  # Above middle, bullish continuation
                elif current_price < signals.bb_middle:
                    signals.bb_position = "lower_half"
                    signals.bb_signal = SignalStrength.SELL  # Below middle, bearish continuation
                else:
                    signals.bb_position = "middle"
                    signals.bb_signal = SignalStrength.NEUTRAL

            # Calculate EMA Crossover (9 and 21 period)
            if len(closes) >= 21:
                signals.ema_9 = self._calculate_ema(closes, 9)
                signals.ema_21 = self._calculate_ema(closes, 21)

                # Also calculate previous values to detect crossover
                prev_ema_9 = self._calculate_ema(closes[:-1], 9)
                prev_ema_21 = self._calculate_ema(closes[:-1], 21)

                # Detect crossover
                if signals.ema_9 > signals.ema_21 and prev_ema_9 <= prev_ema_21:
                    signals.ema_crossover = "golden_cross"
                    signals.ema_signal = SignalStrength.STRONG_BUY
                elif signals.ema_9 < signals.ema_21 and prev_ema_9 >= prev_ema_21:
                    signals.ema_crossover = "death_cross"
                    signals.ema_signal = SignalStrength.STRONG_SELL
                elif signals.ema_9 > signals.ema_21:
                    signals.ema_crossover = "bullish"
                    signals.ema_signal = SignalStrength.BUY
                elif signals.ema_9 < signals.ema_21:
                    signals.ema_crossover = "bearish"
                    signals.ema_signal = SignalStrength.SELL
                else:
                    signals.ema_crossover = "neutral"
                    signals.ema_signal = SignalStrength.NEUTRAL

            # Price changes
            signals.price_change_1h = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 else 0
            signals.price_change_24h = ((closes[-1] - closes[-24]) / closes[-24]) * 100 if len(closes) >= 24 else 0

            # FUT-RM-06: ATR (14-period) for per-symbol risk-parity sizing.
            # Simple mean of TRs over the last 14 bars (close enough to
            # Wilder for sizing — Wilder undershoots SMA by <5% steady-state).
            atr_period = 14
            if len(closes) >= atr_period + 1:
                trs = []
                for i in range(1, len(closes)):
                    h = highs[i]
                    l = lows[i]
                    pc = closes[i-1]
                    tr = max(h - l, abs(h - pc), abs(l - pc))
                    trs.append(tr)
                signals.atr = sum(trs[-atr_period:]) / atr_period if trs else 0.0
                last_close = closes[-1]
                signals.atr_pct = (signals.atr / last_close) if last_close > 0 else 0.0

            # Trend detection
            sma_20 = sum(closes[-20:]) / 20
            sma_50 = sum(closes[-50:]) / 50
            if closes[-1] > sma_20 > sma_50:
                signals.trend = "uptrend"
            elif closes[-1] < sma_20 < sma_50:
                signals.trend = "downtrend"
            else:
                signals.trend = "sideways"

            # Support/Resistance (simple implementation)
            signals.support_level = min(lows[-20:])
            signals.resistance_level = max(highs[-20:])

            return signals

        except Exception as e:
            logger.error(f"Error calculating signals for {symbol}: {e}")
            return None

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(closes) < period + 1:
            return 50.0

        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, closes: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD (12, 26, 9) with a REAL 9-period signal line.

        Pre-Wave-7 this used signal_line = macd_line * 0.9, which made
        histogram = macd_line * 0.1 — i.e. the histogram was just a fixed
        fraction of the MACD line, not the MACD-minus-signal crossover the
        downstream scoring assumes. That broke MACD as a momentum signal:
        the histogram never reflected an actual signal-line cross, and its
        magnitude scaled with raw price (huge on BTC, tiny on a $0.50 alt),
        so the absolute 0.001/0.005 thresholds in _get_technical_signals
        fired almost-always on majors and almost-never on cheap alts.

        Fix: build the full MACD-line series across the window, then take a
        true 9-period EMA of it as the signal line. histogram = macd - signal.
        """
        def ema_series(data: List[float], period: int) -> List[float]:
            """Return the EMA value at each step (same length as data)."""
            if not data:
                return []
            multiplier = 2 / (period + 1)
            seed = min(period, len(data))
            ema_value = sum(data[:seed]) / seed
            out: List[float] = [ema_value]
            for price in data[seed:]:
                ema_value = (price - ema_value) * multiplier + ema_value
                out.append(ema_value)
            return out

        if len(closes) < 26:
            # Not enough history for a meaningful MACD; report flat.
            return 0.0, 0.0, 0.0

        ema_12_series = ema_series(closes, 12)
        ema_26_series = ema_series(closes, 26)
        # Align the two series on their shared tail so each MACD point uses
        # the 12- and 26-EMA computed at the same bar.
        n = min(len(ema_12_series), len(ema_26_series))
        macd_series = [
            ema_12_series[-n + i] - ema_26_series[-n + i] for i in range(n)
        ]
        macd_line = macd_series[-1]

        # Real 9-period signal line = EMA of the MACD-line series.
        signal_series = ema_series(macd_series, 9)
        signal_line = signal_series[-1] if signal_series else macd_line
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return sum(data) / len(data) if data else 0
        multiplier = 2 / (period + 1)
        ema_value = sum(data[:period]) / period
        for price in data[period:]:
            ema_value = (price - ema_value) * multiplier + ema_value
        return ema_value

    def _calculate_position_size(self, signals: TechnicalSignals) -> float:
        """
        Calculate position size based on dynamic or static settings

        Dynamic sizing: Adjusts position size based on signal strength
        - Stronger signals → larger positions (up to max_position_pct)
        - Weaker signals → smaller positions (down to min_position_pct)

        Static sizing: Uses fixed percentage of capital

        ATR sizing (FUT-RM-06): Per-symbol risk-parity. The capped dollar
        risk per trade is held constant; position size scales inversely
        with ATR. A 5% ATR symbol gets 1/5 the notional of a 1% ATR symbol
        — same realized $ loss when the price moves stop-multiplier * ATR.

        Returns: Position size in USD (notional value before leverage)
        """
        # FUT-RM-06: ATR-based sizing path (highest priority when enabled
        # AND we have a usable ATR reading; otherwise fall through to the
        # existing static / dynamic paths).
        if (
            getattr(self, 'atr_sizing_enabled', False)
            and signals is not None
            and getattr(signals, 'atr_pct', 0.0) > 0.0
        ):
            risk_amount = self.capital_allocation * (self.atr_risk_pct / 100.0)
            stop_distance_pct = signals.atr_pct * float(self.atr_stop_multiplier)
            if stop_distance_pct > 0:
                position_margin = risk_amount / stop_distance_pct
                notional = position_margin * float(self.leverage)
                logger.debug(
                    f"ATR sizing: risk=${risk_amount:.2f} "
                    f"ATR%={signals.atr_pct*100:.3f} stopMult={self.atr_stop_multiplier} "
                    f"-> margin=${position_margin:.2f} × {self.leverage}x "
                    f"= ${notional:.2f}"
                )
                if notional > self.max_position_usd:
                    notional = self.max_position_usd
                    logger.debug(f"ATR sizing capped at ${self.max_position_usd:.2f}")
                return notional
            logger.debug("ATR sizing skipped: stop_distance_pct=0; falling through")

        if not self.dynamic_position_sizing:
            # Static position sizing: Capital × Position%
            # Example: $300 × 15% = $45 margin
            position_margin = self.capital_allocation * (self.static_position_pct / 100)
            notional = position_margin * self.leverage  # Apply leverage for notional
            logger.debug(f"Static sizing: ${position_margin:.2f} margin × {self.leverage}x = ${notional:.2f} notional")
        else:
            # Dynamic position sizing based on signal strength
            # Signal score ranges from 1-6 (higher = stronger signal)
            signal_score = getattr(signals, 'combined_score', 3) if signals else 3

            # Map signal score to position percentage
            # Score 1-2: min_position_pct
            # Score 3-4: mid-range
            # Score 5-6: max_position_pct
            min_pct = self.min_position_pct
            max_pct = self.max_position_pct

            # Linear interpolation based on score (1-6 range)
            score_normalized = max(0, min(1, (signal_score - 1) / 5))  # 0 to 1
            position_pct = min_pct + (max_pct - min_pct) * score_normalized

            position_margin = self.capital_allocation * (position_pct / 100)
            notional = position_margin * self.leverage

            logger.debug(f"Dynamic sizing: Signal={signal_score}, {position_pct:.1f}% = ${position_margin:.2f} margin × {self.leverage}x = ${notional:.2f} notional")

        # Apply max USD cap
        if notional > self.max_position_usd:
            notional = self.max_position_usd
            logger.debug(f"Position capped at max ${self.max_position_usd:.2f}")

        return notional

    def _calculate_tp_levels(self, entry_price: float, side: TradeSide) -> List[Dict]:
        """
        Calculate multiple take profit levels

        Returns list of TP levels with price and size percentage:
        [{'price': 1.05, 'pct': 2.0, 'size_pct': 25.0}, ...]
        """
        tp_levels = []

        # Define TP configs: (price_pct, size_pct)
        tp_configs = [
            (self.tp1_pct, self.tp1_size_pct),
            (self.tp2_pct, self.tp2_size_pct),
            (self.tp3_pct, self.tp3_size_pct),
            (self.tp4_pct, self.tp4_size_pct),
        ]

        cumulative_size = 0.0

        for i, (price_pct, size_pct) in enumerate(tp_configs):
            # Skip if size is 0 or we've already allocated 100%
            if size_pct <= 0 or cumulative_size >= 100:
                continue

            # Calculate price based on side
            if side == TradeSide.LONG:
                tp_price = entry_price * (1 + price_pct / 100)
            else:
                tp_price = entry_price * (1 - price_pct / 100)

            tp_levels.append({
                'level': i + 1,
                'price': tp_price,
                'pct': price_pct,
                'size_pct': size_pct,
                'hit': False
            })

            cumulative_size += size_pct

        return tp_levels

    async def _check_tp_levels(self, position: Position, current_price: float) -> bool:
        """
        Check if any TP level has been hit and execute partial close

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            bool: True if a TP was hit (partial close executed)
        """
        if not position.tp_levels:
            return False

        for tp in position.tp_levels:
            if tp['hit']:
                continue  # Already hit this TP

            tp_price = tp['price']

            # Check if TP is hit based on side
            if position.side == TradeSide.LONG:
                tp_hit = current_price >= tp_price
            else:  # SHORT
                tp_hit = current_price <= tp_price

            if tp_hit:
                # Execute partial close
                close_size_pct = tp['size_pct']
                close_size = position.original_size * (close_size_pct / 100)

                # Don't close more than remaining size
                close_size = min(close_size, position.size)

                if close_size > 0:
                    logger.info(f"🎯 TP{tp['level']} hit for {position.symbol}! Price: ${current_price:.4f} >= TP: ${tp_price:.4f}")
                    logger.info(f"   Closing {close_size_pct}% of position ({close_size:.6f} units)")

                    await self._partial_close_position(position, close_size, f"take_profit_{tp['level']}")

                    # Mark as hit
                    tp['hit'] = True

                    # TRAILING STOP ACTIVATION:
                    # After TP1: Move stop loss to breakeven (entry price) - always enabled for safety
                    # After TP2+: Activate trailing stop (only if trailing_stop_enabled)
                    if tp['level'] == 1:
                        # TP1 hit - move stop to true breakeven INCLUDING
                        # round-trip fees (FUT-RM-14). Stopping exactly at
                        # entry_price still pays entry+exit fees, so the
                        # operator nets a small loss on a "breakeven" stop.
                        # Buffer = 2 × taker_fee + tiny slippage cushion.
                        fee_rate = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
                        be_buffer_pct = (2 * fee_rate) + 0.0001  # +1bp cushion
                        if position.side == TradeSide.LONG:
                            position.trailing_stop_price = position.entry_price * (1 + be_buffer_pct)
                        else:  # SHORT
                            position.trailing_stop_price = position.entry_price * (1 - be_buffer_pct)
                        logger.info(
                            f"🔒 {position.symbol}: Stop moved to fee-adjusted breakeven "
                            f"${position.trailing_stop_price:.4f} (entry ${position.entry_price:.4f}, "
                            f"buffer {be_buffer_pct*100:.3f}%)"
                        )
                    elif tp['level'] >= 2 and not position.trailing_stop_active and self.trailing_stop_enabled:
                        # TP2+ hit - activate trailing stop (ONLY if trailing stop is enabled in settings)
                        position.trailing_stop_active = True
                        # Initialize trailing stop based on current peak
                        if position.side == TradeSide.LONG:
                            peak = position.highest_price or current_price
                            position.trailing_stop_price = peak * (1 - self.trailing_stop_distance / 100)
                        else:  # SHORT
                            peak = position.lowest_price or current_price
                            position.trailing_stop_price = peak * (1 + self.trailing_stop_distance / 100)
                        logger.info(f"🎚️ {position.symbol}: Trailing stop activated at ${position.trailing_stop_price:.4f}")
                    elif tp['level'] >= 2 and not position.trailing_stop_active and not self.trailing_stop_enabled:
                        # TP2+ hit but trailing stop disabled - update stop to current TP level for protection
                        position.trailing_stop_price = tp['price'] * (0.99 if position.side == TradeSide.LONG else 1.01)
                        logger.info(f"🔒 {position.symbol}: Stop updated to TP{tp['level']} level ${position.trailing_stop_price:.4f} (TSL disabled)")

                    return True

        return False

    async def _partial_close_position(self, position: Position, close_size: float, reason: str):
        """
        Execute a partial close of a position

        Args:
            position: Position to partially close
            close_size: Size to close (in base currency)
            reason: Reason for the partial close
        """
        symbol = position.symbol

        try:
            # Calculate close value
            close_notional = close_size * position.current_price
            close_pct = (close_size / position.original_size) * 100

            # Calculate PnL for this portion
            if position.side == TradeSide.LONG:
                pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - position.current_price) / position.entry_price) * 100

            leveraged_pnl_pct = pnl_pct * position.leverage
            pnl_usd = close_notional * (pnl_pct / 100)

            # Calculate fees for this portion
            fee_rate = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
            exit_fee = close_notional * fee_rate
            net_pnl = pnl_usd - exit_fee

            # Execute partial close (or simulate)
            if should_skip_live(self.dry_run, module='futures', account=self.exchange):
                logger.info(f"🔵 [DRY_RUN] Partial close {symbol} ({reason})")
                logger.info(f"   Closed: {close_size:.6f} ({close_pct:.1f}%), PnL: ${net_pnl:.2f}")
            else:
                # Execute real partial close order
                close_side = 'sell' if position.side == TradeSide.LONG else 'buy'
                try:
                    order = await self.exchange_client.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=close_size,
                        params={'reduceOnly': True}
                    )
                    logger.info(f"🎯 Partial close {symbol} - Order ID: {order['id']}")
                except Exception as e:
                    logger.error(f"❌ Partial close order failed: {e}")
                    return

            # Update position size
            position.size -= close_size
            position.notional_value = position.size * position.entry_price

            # Update stats
            self.total_pnl += net_pnl
            self.total_fees += exit_fee
            self.risk_metrics.daily_pnl += net_pnl

            if net_pnl > 0:
                self.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1

            # Record partial trade in history
            partial_trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                size=close_size,
                notional_value=close_notional,
                leverage=position.leverage,
                pnl=net_pnl,
                pnl_pct=leveraged_pnl_pct,
                fees=exit_fee,
                opened_at=position.opened_at,
                closed_at=datetime.now(),
                close_reason=reason,
                is_simulated=position.is_simulated
            )
            self.trade_history.append(partial_trade)
            await self._save_trade_to_db(partial_trade)

            # Send Telegram alert for partial TP
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from futures_trading.core.futures_alerts import FuturesTradeAlert
                    alert = FuturesTradeAlert(
                        symbol=symbol,
                        side=position.side.value,
                        action=reason,
                        entry_price=position.entry_price,
                        exit_price=position.current_price,
                        size=close_size,
                        leverage=position.leverage,
                        pnl=net_pnl,
                        pnl_pct=leveraged_pnl_pct,
                        reason=f"Partial close ({close_pct:.0f}%)",
                        is_simulated=position.is_simulated,
                        exchange=self.exchange
                    )
                    await self.telegram_alerts.send_exit_alert(alert)
                except Exception as e:
                    logger.warning(f"Failed to send Telegram partial close alert: {e}")

            # Log result
            pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
            remaining_pct = (position.size / position.original_size) * 100 if position.original_size > 0 else 0
            logger.info(f"{pnl_emoji} Partial close: {symbol} {reason}")
            logger.info(f"   Closed: {close_pct:.1f}%, Remaining: {remaining_pct:.1f}%")
            logger.info(f"   PnL: ${net_pnl:.2f} ({leveraged_pnl_pct:.2f}%)")

            # Check if fully closed (use tolerance for floating-point comparison)
            if position.size <= 1e-8:
                logger.info(f"✅ Position {symbol} fully closed through TPs")
                self.total_trades += 1
                self.symbol_cooldowns[symbol] = datetime.now() + self.cooldown_duration
                # Clean up position from active positions to prevent double-close
                if symbol in self.active_positions:
                    del self.active_positions[symbol]

        except Exception as e:
            logger.error(f"Error in partial close {symbol}: {e}")

    async def _open_position(self, symbol: str, side: TradeSide, signals: TechnicalSignals):
        """Open a new position with dynamic sizing and multiple TPs"""
        try:
            # Get current price
            ticker = await self._get_ticker(symbol)
            if not ticker:
                return

            current_price = float(ticker['last'])

            # FUT-RM-27 (Wave 25): per-symbol tier / rolling-gate. This is
            # the AUTHORITATIVE check (the scan-loop one is just an
            # optimization) so it also covers the FUT-RM-25 carry path,
            # which enters via _open_position directly. Fail-open.
            symbol_weight = 1.0
            if self.risk_manager is not None:
                try:
                    if hasattr(self.risk_manager, 'should_skip_for_symbol'):
                        sgate = self.risk_manager.should_skip_for_symbol(symbol)
                        if sgate.get('skip'):
                            logger.debug(
                                f"FUT-RM-27 symbol gate refused entry for "
                                f"{symbol}: {sgate.get('reason')}"
                            )
                            return
                    if hasattr(self.risk_manager, 'resolve_symbol_size_weight'):
                        symbol_weight = float(
                            self.risk_manager.resolve_symbol_size_weight(symbol)
                        )
                except Exception as e:
                    logger.debug(f"symbol gate non-fatal error for {symbol}: {e}")
                    symbol_weight = 1.0

            # Calculate position size based on settings
            notional = self._calculate_position_size(signals)
            # FUT-RM-27: apply the per-symbol tier/probation size multiplier.
            if symbol_weight != 1.0:
                logger.info(
                    f"FUT-RM-27 size weight {symbol_weight:.2f}x on {symbol}: "
                    f"${notional:.2f} -> ${notional * symbol_weight:.2f}"
                )
                notional *= symbol_weight
            size = notional / current_price

            # Ensure minimum trade size
            if notional < self.min_trade_size:
                logger.debug(f"Position size ${notional:.2f} below minimum ${self.min_trade_size}, skipping")
                return

            # Calculate fees (estimate)
            fee_rate = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
            estimated_fees = notional * fee_rate * 2  # Entry + exit

            # Calculate stop loss price (SL% is price move %)
            sl_pct = abs(self.stop_loss_pct)

            # FUT-RM-16 (Wave 5): ATR-scaled SL/TP per symbol. Overrides the
            # static sl_pct with max(atr_sl_min_pct, atr_sl_multiplier × ATR%)
            # and rescales TP1..TP4 so TP1 = atr_tp_rr_ratio × SL distance.
            # TP2/TP3/TP4 keep their relative proportions to TP1 so the
            # front-loaded size_pct distribution still makes sense.
            atr_dyn = (
                getattr(self, 'atr_dynamic_sl_tp_enabled', False)
                and signals is not None
                and getattr(signals, 'atr_pct', 0.0) > 0.0
            )
            _orig_sl_pct = self.stop_loss_pct
            _orig_tps = (self.tp1_pct, self.tp2_pct, self.tp3_pct, self.tp4_pct)
            if atr_dyn:
                atr_pct = float(signals.atr_pct) * 100.0  # decimal -> %
                dyn_sl_pct = max(
                    float(self.atr_sl_min_pct),
                    float(self.atr_sl_multiplier) * atr_pct,
                )
                dyn_tp1_pct = dyn_sl_pct * float(self.atr_tp_rr_ratio)
                tp1_ratio = dyn_tp1_pct / self.tp1_pct if self.tp1_pct > 0 else 1.0
                self.stop_loss_pct = dyn_sl_pct
                self.tp1_pct = dyn_tp1_pct
                self.tp2_pct = _orig_tps[1] * tp1_ratio
                self.tp3_pct = _orig_tps[2] * tp1_ratio
                self.tp4_pct = _orig_tps[3] * tp1_ratio
                sl_pct = dyn_sl_pct
                logger.info(
                    f"FUT-RM-16 ATR SL/TP {symbol}: ATR%={atr_pct:.2f} -> "
                    f"SL={dyn_sl_pct:.2f}% TP1={dyn_tp1_pct:.2f}% "
                    f"(R:R={self.atr_tp_rr_ratio:.1f})"
                )

            # Calculate multiple take profit levels
            tp_levels = self._calculate_tp_levels(current_price, side)

            # FUT-RM-16: restore static settings so the next entry recomputes
            # from the operator's baseline; the position carries its own
            # tp_levels + stop_loss_price for live monitoring.
            if atr_dyn:
                self.stop_loss_pct = _orig_sl_pct
                self.tp1_pct, self.tp2_pct, self.tp3_pct, self.tp4_pct = _orig_tps

            # FUT-RM-19 (Wave 7): fee + funding aware minimum-edge gate.
            # Refuse entries whose first realistic target (TP1 distance) does
            # not clear round-trip costs by min_net_edge_pct. This is the
            # working-rule edge formula: TP1 - 2*taker_fee - slippage -
            # funding_drag. Cheap; runs before the heavier validator. No order
            # is placed if it fails, so it is inherently DRY_RUN-safe.
            if getattr(self, 'min_edge_gate_enabled', True):
                tp1_dist_pct = float(tp_levels[0]['pct']) if tp_levels else float(self.take_profit_pct)
                edge = await self._compute_net_edge_pct(symbol, side, tp1_dist_pct)
                if edge['net_edge_pct'] < float(self.min_net_edge_pct):
                    msg = (
                        f"⏭️  FUT-RM-19 edge gate refused {side.value.upper()} "
                        f"{symbol}: net_edge={edge['net_edge_pct']:.3f}% < "
                        f"min {self.min_net_edge_pct:.3f}% "
                        f"(TP1={tp1_dist_pct:.2f}% fees={edge['fee_pct']:.3f}% "
                        f"slip={edge['slippage_pct']:.3f}% "
                        f"funding={edge['funding_pct']:+.3f}%)"
                    )
                    logger.warning(msg)
                    self._alert_edge_gate_breach(symbol, side, edge, tp1_dist_pct)
                    return

            if side == TradeSide.LONG:
                stop_loss_price = current_price * (1 - sl_pct / 100)
                take_profit_price = tp_levels[0]['price'] if tp_levels else current_price * (1 + self.take_profit_pct / 100)
                liquidation_price = current_price * (1 - 0.9 / self.leverage)
            else:
                stop_loss_price = current_price * (1 + sl_pct / 100)
                take_profit_price = tp_levels[0]['price'] if tp_levels else current_price * (1 - self.take_profit_pct / 100)
                liquidation_price = current_price * (1 + 0.9 / self.leverage)

            # Log detailed entry with SL and TP levels for trade log (cleaner format)
            tp_str = ""
            if tp_levels:
                tp_str = f"TP1:${tp_levels[0]['price']:.2f} | TP2:${tp_levels[1]['price']:.2f} | TP3:${tp_levels[2]['price']:.2f} | TP4:${tp_levels[3]['price']:.2f}"
            else:
                tp_str = f"${take_profit_price:.2f}"

            # FUT-RM-17 (Wave 5): per-symbol consecutive-loss cool-off.
            # Cheap pre-validator skip; fails open on missing method or error.
            if self.risk_manager is not None and hasattr(
                self.risk_manager, 'should_skip_for_cooloff'
            ):
                try:
                    cgate = self.risk_manager.should_skip_for_cooloff(symbol)
                    if cgate.get('skip'):
                        # Wave-25: throttled — once per cool-off instance +
                        # 30-min INFO reminders (was WARNING every ~35s).
                        self._log_cooloff_refusal(symbol, cgate)
                        return
                except Exception as e:
                    logger.debug(f"cooloff gate non-fatal error for {symbol}: {e}")

            # FUT-RM-05: funding-rate directional gate. Cheap, runs before
            # the heavier validator. None rate -> gate fails open.
            if self.risk_manager is not None and hasattr(
                self.risk_manager, 'should_skip_for_funding'
            ):
                try:
                    fund_rate = await self._get_funding_rate_cached(symbol)
                    fgate = self.risk_manager.should_skip_for_funding(
                        side=side.value.upper() if hasattr(side, 'value') else str(side),
                        funding_rate=fund_rate,
                    )
                    if fgate.get('skip'):
                        import time as _ft
                        _now_fw = _ft.time()
                        _last_fw = self._funding_gate_warned_at.get(symbol, 0.0)
                        _interval_fw = getattr(
                            self, '_funding_gate_warn_interval_s', 300.0
                        )
                        if _now_fw - _last_fw >= _interval_fw:
                            logger.warning(
                                f"⏭️  Funding gate refused entry for {symbol}: "
                                f"{fgate.get('reason')} "
                                f"(further refusals demoted to DEBUG for "
                                f"{int(_interval_fw)}s)"
                            )
                            self._funding_gate_warned_at[symbol] = _now_fw
                        else:
                            logger.debug(
                                f"funding gate refused entry for {symbol}: "
                                f"{fgate.get('reason')} (throttled)"
                            )
                        return
                except Exception as e:
                    logger.debug(f"funding gate non-fatal error for {symbol}: {e}")

            # MB-17: cross-module risk gate — refuse to open if validator rejects.
            if self.risk_manager is not None:
                try:
                    current_positions = [
                        {'notional_value': p.notional_value}
                        for p in self.active_positions.values()
                    ]
                    validation = self.risk_manager.validate_new_position(
                        symbol=symbol,
                        side=side.value.upper() if hasattr(side, 'value') else str(side),
                        size_usd=notional,
                        leverage=self.leverage,
                        current_positions=current_positions,
                        available_capital=self.capital_allocation,
                    )
                except Exception as e:
                    logger.error(f"Risk validator raised: {e}; refusing entry")
                    return
                if not validation.get('allowed', True):
                    logger.info(
                        f"     ⏭️  Risk gate: {symbol} rejected — "
                        f"{validation.get('reason', 'no reason given')}"
                    )
                    return

            # Create position object
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=symbol,
                side=side,
                entry_price=current_price,
                current_price=current_price,
                size=size,
                notional_value=notional,
                leverage=self.leverage,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                liquidation_price=liquidation_price,
                fees_paid=estimated_fees / 2,  # Entry fee
                is_simulated=self.dry_run,
                metadata={
                    'signals': {
                        'rsi': signals.rsi,
                        'macd': signals.macd_histogram,
                        'volume_ratio': signals.volume_ratio,
                        'trend': signals.trend
                    },
                    # FUT-RM-16: stash the SL pct actually used at entry so
                    # _check_exit_conditions uses the per-symbol level even
                    # after the engine reverts self.stop_loss_pct for the
                    # next entry. None when ATR sizing was off.
                    'dynamic_sl_pct': sl_pct if atr_dyn else None,
                },
                tp_levels=tp_levels,
                original_size=size,
                highest_price=current_price if side == TradeSide.LONG else None,
                lowest_price=current_price if side == TradeSide.SHORT else None
            )

            # Execute order (or simulate)
            if should_skip_live(self.dry_run, module='futures', account=self.exchange):
                logger.info(f"🔵 [DRY_RUN] SIMULATED {side.value.upper()} {symbol}")
                logger.info(f"   Entry: ${current_price:.2f}, Size: {size:.6f}, Notional: ${notional:.2f}")
                logger.info(f"   SL: ${stop_loss_price:.2f}, TP: ${take_profit_price:.2f}")
            else:
                # Execute real order via the ISOLATED-margin + leverage-set helpers
                try:
                    if side == TradeSide.LONG:
                        order = await self.exchange_client.open_long(
                            symbol=symbol,
                            quantity=size,
                            leverage=self.leverage,
                        )
                    else:
                        order = await self.exchange_client.open_short(
                            symbol=symbol,
                            quantity=size,
                            leverage=self.leverage,
                        )
                    if not order:
                        logger.error(f"❌ Order execution returned empty result for {symbol}")
                        return
                    logger.info(f"🟢 EXECUTED {side.value.upper()} {symbol}")
                    logger.info(f"   Order ID: {order.get('id', order.get('orderId', 'unknown'))}")
                    logger.info(f"   Entry: ${current_price:.2f}, Size: {size:.6f}")

                    # Update entry price from actual fill
                    if order.get('average'):
                        position.entry_price = float(order['average'])

                    # FUT-RM-07: defense-in-depth on MB-17. set_margin_type
                    # is called inside open_long/open_short, but a stale
                    # account-level setting or a Bybit 110026/110043
                    # idempotency false-positive could land us with a
                    # CROSS-margin fill. Verify by re-reading the position
                    # immediately and close on mismatch.
                    if getattr(self, 'enforce_isolated_margin', True):
                        await self._verify_isolated_or_close(symbol, side)

                except Exception as e:
                    logger.error(f"❌ Order execution failed: {e}")
                    return

            # Add to active positions
            self.active_positions[symbol] = position
            self.risk_metrics.current_exposure += notional
            # FUT-RM-20 (Wave 7): mark this symbol as entered for the current
            # candle so the one-entry-per-candle throttle won't re-fire on it.
            self._last_entry_candle[symbol] = self._current_candle_open()

            # Log trade entry with SL/TP details (captured by TradeLogFilter for futures_trades.log)
            logger.info(f"✅ Position opened: {symbol} {side.value.upper()}")
            logger.info(f"   Entry: ${current_price:.4f}, Notional: ${notional:.2f}, Leverage: {self.leverage}x")
            logger.info(f"   🛑 Stop Loss: ${stop_loss_price:.4f} ({sl_pct:.1f}%)")
            logger.info(f"   📈 Take Profit: {tp_str}")
            logger.info(f"   Active positions: {len(self.active_positions)}/{self.max_positions}")

            # Send Telegram entry alert
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from futures_trading.core.futures_alerts import FuturesTradeAlert
                    alert = FuturesTradeAlert(
                        symbol=symbol,
                        side=side.value,
                        action='entry',
                        entry_price=position.entry_price,
                        size=position.size,
                        leverage=position.leverage,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        trailing_stop=position.metadata.get('trailing_stop'),
                        is_simulated=position.is_simulated,
                        exchange=self.exchange
                    )
                    await self.telegram_alerts.send_entry_alert(alert)
                except Exception as e:
                    logger.warning(f"Failed to send Telegram entry alert: {e}")

        except Exception as e:
            logger.error(f"Error opening position {symbol}: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        if symbol not in self.active_positions:
            return

        position = self.active_positions[symbol]

        try:
            # Calculate final PnL
            if position.side == TradeSide.LONG:
                pnl_pct = ((position.current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - position.current_price) / position.entry_price) * 100

            leveraged_pnl_pct = pnl_pct * position.leverage
            pnl_usd = position.notional_value * (pnl_pct / 100)

            # Calculate fees
            fee_rate = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
            exit_fee = position.notional_value * fee_rate
            total_fees = position.fees_paid + exit_fee

            # Net PnL after fees
            net_pnl = pnl_usd - total_fees

            # Execute close order (or simulate)
            if should_skip_live(self.dry_run, module='futures', account=self.exchange):
                logger.info(f"🔵 [DRY_RUN] SIMULATED CLOSE {symbol} ({reason})")
            else:
                # Execute real close order
                close_side = 'sell' if position.side == TradeSide.LONG else 'buy'
                try:
                    order = await self.exchange_client.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=position.size,
                        params={'reduceOnly': True}
                    )
                    logger.info(f"🔴 CLOSED {symbol} - Order ID: {order['id']}")
                except Exception as e:
                    logger.error(f"❌ Close order failed: {e}")
                    return

            # Record trade
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                symbol=symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                size=position.size,
                notional_value=position.notional_value,
                leverage=position.leverage,
                pnl=net_pnl,
                pnl_pct=leveraged_pnl_pct,
                fees=total_fees,
                opened_at=position.opened_at,
                closed_at=datetime.now(),
                close_reason=reason,
                is_simulated=position.is_simulated
            )
            self.trade_history.append(trade)

            # Record in PnL tracker for Sharpe/Sortino calculations
            trade_record = TradeRecord(
                trade_id=trade.trade_id,
                symbol=symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=position.current_price,
                size=position.size,
                pnl=pnl_usd,
                fees=total_fees,
                net_pnl=net_pnl,
                pnl_pct=leveraged_pnl_pct,
                entry_time=position.opened_at,
                exit_time=datetime.now(),
                duration_seconds=int((datetime.now() - position.opened_at).total_seconds()),
                is_simulated=position.is_simulated
            )
            self.pnl_tracker.record_trade(trade_record)

            # Save trade to database for persistence across restarts
            await self._save_trade_to_db(trade)

            # Update stats
            self.total_trades += 1
            self.total_pnl += net_pnl
            self.total_fees += total_fees
            self.risk_metrics.daily_pnl += net_pnl
            self.risk_metrics.daily_trades += 1
            self.risk_metrics.current_exposure -= position.notional_value

            if net_pnl > 0:
                self.winning_trades += 1
                self.risk_metrics.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.risk_metrics.consecutive_losses += 1

            # FUT-RM-17 (Wave 5): notify risk manager so per-symbol cool-off
            # arms after N consecutive losses on this pair. Best-effort.
            if self.risk_manager is not None and hasattr(
                self.risk_manager, 'update_on_trade_close'
            ):
                try:
                    self.risk_manager.update_on_trade_close(net_pnl, symbol=symbol)
                except Exception as e:
                    logger.debug(f"risk_manager.update_on_trade_close failed: {e}")

            # Remove from active positions
            del self.active_positions[symbol]
            # FUT-RM-25: clean up carry tracking on any close path.
            self._carry_position_symbols.discard(symbol)
            # FUT-QC-01: same for the carry-v2 book.
            self._carry_v2_symbols.discard(symbol)

            # Set cooldown
            self.symbol_cooldowns[symbol] = datetime.now() + self.cooldown_duration

            # Log result with detailed information
            pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
            sim_tag = "[DRY_RUN] " if position.is_simulated else ""
            logger.info(f"{pnl_emoji} {sim_tag}Position closed: {symbol}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Entry: ${position.entry_price:.4f} → Exit: ${position.current_price:.4f}")
            logger.info(f"   Price Change: {pnl_pct:+.4f}%")
            logger.info(f"   PnL (with {position.leverage}x leverage): ${net_pnl:.2f} ({leveraged_pnl_pct:+.2f}%)")
            if reason == "stop_loss":
                logger.info(f"   SL Trigger: Price moved {pnl_pct:.2f}% <= -{self.stop_loss_pct}% threshold")
            logger.info(f"   Fees: ${total_fees:.2f}")
            logger.info(f"   Daily PnL: ${self.risk_metrics.daily_pnl:.2f}")

            # Send Telegram exit alert
            if self.telegram_alerts and self.telegram_alerts.enabled:
                try:
                    from futures_trading.core.futures_alerts import FuturesTradeAlert
                    alert = FuturesTradeAlert(
                        symbol=symbol,
                        side=position.side.value,
                        action=reason,  # 'take_profit', 'stop_loss', 'manual_close', etc.
                        entry_price=position.entry_price,
                        exit_price=position.current_price,
                        size=position.size,
                        leverage=position.leverage,
                        pnl=net_pnl,
                        pnl_pct=leveraged_pnl_pct,
                        reason=reason,
                        is_simulated=position.is_simulated,
                        exchange=self.exchange
                    )
                    await self.telegram_alerts.send_exit_alert(alert)
                except Exception as e:
                    logger.warning(f"Failed to send Telegram exit alert: {e}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    async def _process_orders(self):
        """Process pending orders"""
        # Currently not implementing limit orders - using market orders only
        pass

    async def _get_decision_price(self, symbol: str) -> Optional[float]:
        """Return the exchange's mark price if available; fall back to ticker last.
        Mark price is what drives liquidation — use it for SL/TP/liq checks."""
        if hasattr(self.exchange_client, 'get_mark_price'):
            try:
                mp = await self.exchange_client.get_mark_price(symbol)
                if mp is not None and mp > 0:
                    return float(mp)
            except Exception as e:
                logger.debug(f"get_mark_price failed for {symbol}: {e}")
        ticker = await self._get_ticker(symbol)
        if ticker and 'last' in ticker:
            return float(ticker['last'])
        return None

    async def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker for symbol - uses mainnet price client when available"""
        try:
            # Use mainnet price client for accurate prices (especially in testnet/DRY_RUN mode)
            client = self.price_client if self.price_client else self.exchange_client
            ticker = await client.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            # Fallback to exchange_client if price_client fails
            if self.price_client and client == self.price_client:
                try:
                    ticker = await self.exchange_client.fetch_ticker(symbol)
                    return ticker
                except Exception as e2:
                    logger.error(f"Error fetching ticker for {symbol} from both clients: {e}, {e2}")
                    return None
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    async def _get_funding_rate_cached(self, symbol: str) -> Optional[float]:
        """FUT-RM-05: Return the latest per-interval funding rate as a
        FRACTION (e.g. 0.0005 = 5 bps) for `symbol`, cached for
        self._funding_cache_ttl_seconds. Returns None on any failure —
        the gate fails open on missing data.

        Uses the mainnet price_client when available so DRY_RUN/testnet
        sessions see real funding numbers (testnet funding is fictional).
        """
        try:
            now = datetime.now()
            cached = self._funding_cache.get(symbol)
            if cached:
                rate, fetched_at = cached
                age = (now - fetched_at).total_seconds()
                if age < self._funding_cache_ttl_seconds:
                    return rate
            client = self.price_client if self.price_client else self.exchange_client
            if not client or not hasattr(client, 'fetch_funding_rate'):
                return None
            data = await client.fetch_funding_rate(symbol)
            # ccxt normalizes to {'fundingRate': float, ...}
            rate = data.get('fundingRate') if isinstance(data, dict) else None
            if rate is None:
                return None
            rate = float(rate)
            self._funding_cache[symbol] = (rate, now)
            return rate
        except Exception as e:
            logger.debug(f"funding rate fetch failed for {symbol}: {e}")
            return None

    def _current_candle_open(self) -> datetime:
        """FUT-RM-20 (Wave 7): floor `now` to the open of the current
        signal-timeframe candle. Used by the one-entry-per-candle throttle so
        all scans within a single bar map to the same key."""
        tf = str(getattr(self, 'signal_timeframe', '15m')).strip().lower()
        unit = 'm'
        qty = 15
        # Only parse if it's a well-formed <int><unit> string; anything else
        # falls back to a safe 15m bucket (avoids a degenerate giant bucket).
        if len(tf) >= 2 and tf[-1] in ('m', 'h', 'd') and tf[:-1].isdigit():
            unit = tf[-1]
            qty = int(tf[:-1])
        seconds = qty * {'m': 60, 'h': 3600, 'd': 86400}[unit]
        seconds = max(60, seconds)
        now = datetime.now()
        epoch = now.timestamp()
        floored = epoch - (epoch % seconds)
        return datetime.fromtimestamp(floored)

    async def _compute_net_edge_pct(
        self, symbol: str, side: TradeSide, tp1_dist_pct: float
    ) -> Dict[str, float]:
        """FUT-RM-19 (Wave 7): expected net edge of a prospective entry, in
        price-% terms, after subtracting all the costs that ate the operator's
        -$59.99 @ 39% book.

            net_edge_pct = tp1_dist_pct
                           - 2 * taker_fee_pct     (round-trip taker fees)
                           - slippage_pct          (modeled, both legs)
                           - funding_drag_pct      (adverse only; favorable=0)

        funding_drag is the per-interval funding rate (as a price %) only when
        it works AGAINST the position direction — a LONG pays positive funding,
        a SHORT pays negative funding. Favorable funding is floored at 0 here
        (we do not credit it as edge; that would encourage funding-chasing
        entries). When the live rate is unavailable we use the conservative
        edge_funding_fallback_pct so the gate never fails open into free
        trading. Note tp1_dist_pct is the *price* move to TP1, NOT leveraged —
        fees/funding are also charged on notional, so comparing in price-%
        terms is apples-to-apples (leverage scales both sides equally).
        """
        # Round-trip taker fee as a price percentage of notional.
        taker = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
        fee_pct = taker * 2 * 100.0
        slippage_pct = float(getattr(self, 'edge_slippage_pct', 0.05))

        # Funding drag: only count it when it works against us.
        funding_pct = float(getattr(self, 'edge_funding_fallback_pct', 0.05))
        try:
            rate = await self._get_funding_rate_cached(symbol)
            if rate is not None:
                rate_pct = float(rate) * 100.0  # per-interval, as price %
                if side == TradeSide.LONG:
                    # LONG pays when funding > 0; favorable when < 0 -> 0 drag.
                    funding_pct = max(0.0, rate_pct)
                else:
                    # SHORT pays when funding < 0.
                    funding_pct = max(0.0, -rate_pct)
        except Exception as e:
            logger.debug(f"edge funding lookup failed for {symbol}: {e}")

        net_edge_pct = tp1_dist_pct - fee_pct - slippage_pct - funding_pct
        return {
            'tp1_dist_pct': tp1_dist_pct,
            'fee_pct': fee_pct,
            'slippage_pct': slippage_pct,
            'funding_pct': funding_pct,
            'net_edge_pct': net_edge_pct,
        }

    def _alert_edge_gate_breach(
        self, symbol: str, side: TradeSide, edge: Dict[str, float], tp1_dist_pct: float
    ) -> None:
        """FUT-RM-19: surface an edge-gate rejection via monitoring/alerts.py.
        Best-effort + lazy import so the engine import stays clean and a
        missing/uninitialized alerts module never blocks the gate (which has
        already logged + returned). Uses AlertManager.send_alert(alert_type,
        message, priority) — fired on the running loop via ensure_future so
        the _open_position caller is not awaited-on here."""
        try:
            from monitoring.alerts import AlertManager
            mgr = getattr(self, '_alert_manager', None)
            if mgr is None:
                mgr = AlertManager()
                self._alert_manager = mgr  # cache one instance
            side_str = side.value.upper() if hasattr(side, 'value') else str(side)
            detail = (
                f"[futures] FUT-RM-19 edge gate: {side_str} {symbol} rejected — "
                f"net_edge={edge['net_edge_pct']:.3f}% < min {self.min_net_edge_pct:.3f}% "
                f"(TP1={tp1_dist_pct:.2f}% fee={edge['fee_pct']:.3f}% "
                f"slip={edge['slippage_pct']:.3f}% funding={edge['funding_pct']:+.3f}%)"
            )
            asyncio.ensure_future(
                mgr.send_alert('futures_edge_gate', detail, 'low')
            )
        except Exception as e:
            logger.debug(f"edge-gate alert dispatch failed (non-fatal): {e}")

    async def _auto_deleverage_if_needed(self) -> bool:
        """FUT-RM-10 (Wave 3): wire the unused should_auto_deleverage() check.

        Called from the monitor loop. When the drawdown threshold trips
        AND the feature flag is on AND we're outside cooldown, halve the
        WORST unrealized-PnL open position (close 50% at market).

        Returns True if a deleverage fired this call. False otherwise.

        Safety:
          - Feature-flagged off by default (auto_deleverage_enabled).
          - Cooldown so a single drawdown event doesn't repeat-halve.
          - Skips when no risk_manager or no positions.
          - Best-effort: failure to close one position doesn't crash the
            loop; logged at ERROR for the operator.
          - Each trigger logs a clearly-labeled WARNING that the
            futures_trades.log filter picks up.
        """
        try:
            if not getattr(self, 'auto_deleverage_enabled', False):
                return False
            if self.risk_manager is None or not hasattr(
                self.risk_manager, 'should_auto_deleverage'
            ):
                return False
            if not self.active_positions:
                return False
            # Cooldown gate.
            now = datetime.now()
            last = getattr(self, '_auto_deleverage_last_at', None)
            cooldown = max(0, int(getattr(self, 'auto_deleverage_cooldown_seconds', 600)))
            if last and (now - last).total_seconds() < cooldown:
                return False
            # Compose total PnL (realized + unrealized) over a capital base.
            unrealized = sum(
                float(getattr(p, 'unrealized_pnl', 0) or 0)
                for p in self.active_positions.values()
            )
            realized = float(getattr(self, 'total_pnl', 0) or 0)
            total_pnl = realized + unrealized
            capital = float(getattr(self, 'capital_allocation', 0) or 0)
            if capital <= 0:
                return False
            should_dl = bool(self.risk_manager.should_auto_deleverage(
                total_pnl=total_pnl,
                total_capital=capital,
            ))
            if not should_dl:
                return False
            # Pick worst-PnL open position.
            worst_symbol = None
            worst_pnl = float('inf')
            worst_pos = None
            for sym, pos in self.active_positions.items():
                p = float(getattr(pos, 'unrealized_pnl', 0) or 0)
                if p < worst_pnl:
                    worst_pnl = p
                    worst_symbol = sym
                    worst_pos = pos
            if worst_pos is None or worst_symbol is None:
                return False
            close_size = max(0.0, float(getattr(worst_pos, 'size', 0)) / 2.0)
            if close_size <= 0:
                return False
            logger.warning(
                "🛑 FUT-RM-10 AUTO-DELEVERAGE triggered "
                f"(total_pnl=${total_pnl:.2f} on capital=${capital:.2f}); "
                f"halving worst position {worst_symbol} "
                f"(unrealized_pnl=${worst_pnl:.2f}, size {worst_pos.size:.6f} -> {close_size:.6f})"
            )
            self._auto_deleverage_last_at = now
            try:
                await self._partial_close_position(
                    worst_pos, close_size, 'fut_rm_10_auto_deleverage'
                )
                return True
            except Exception as e:
                logger.error(
                    f"FUT-RM-10 partial close failed for {worst_symbol}: {e}"
                )
                return False
        except Exception as e:
            logger.debug(f"_auto_deleverage_if_needed errored: {e}")
            return False

    async def _record_funding_snapshot(self) -> None:
        """FUT-RM-09 (Wave 3): write a per-hour funding-cost snapshot to
        futures_funding_payments for the dashboard widget.

        For each active position we compute:
            predicted_usd = funding_rate(symbol) × notional × side_sign

        side_sign is +1 when the book is *paying* funding (LONG with
        positive funding, SHORT with negative funding) and -1 when the
        book is *receiving*. So predicted_usd > 0 means cost to the book.

        The UNIQUE(hour_bucket, symbol, side, exchange, network, source)
        constraint upserts within an hour bucket so calling this every
        cycle just refreshes the latest snapshot. The dashboard reads the
        trailing 24h and sums by hour_bucket.

        Realized is left at 0 here — populating it requires reading the
        exchange income history (Binance /fapi/v1/income type=FUNDING_FEE,
        Bybit /v5/account/transaction-log type=Funding). That's gated to
        a follow-up commit so this one stays in the LoC budget.

        Best-effort: any failure is logged at debug and never blocks the
        trading loop. No DB writes in DRY_RUN unless db_pool is available
        (the dashboard widget works for paper trading too).
        """
        try:
            if not self.db_pool or not self.active_positions:
                return
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc).replace(minute=0, second=0, microsecond=0)
            network = 'testnet' if getattr(self, 'testnet', False) else 'mainnet'
            exch = getattr(self, 'exchange', 'binance') or 'binance'
            rows = []
            for symbol, pos in list(self.active_positions.items()):
                try:
                    rate = await self._get_funding_rate_cached(symbol)
                    if rate is None:
                        continue
                    notional = float(getattr(pos, 'notional_value', 0) or 0)
                    if notional <= 0:
                        continue
                    side_val = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                    side_u = side_val.upper()
                    # LONG pays when funding > 0; SHORT pays when funding < 0.
                    if side_u == 'LONG':
                        side_sign = 1.0 if rate > 0 else -1.0
                    else:
                        side_sign = 1.0 if rate < 0 else -1.0
                    predicted = abs(float(rate)) * notional * side_sign
                    rows.append((now, symbol, side_u, notional, predicted, 0.0,
                                 exch, network, 'engine'))
                except Exception as inner:
                    logger.debug(f"funding snapshot row failed for {symbol}: {inner}")
            if not rows:
                return
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO futures_funding_payments (
                        hour_bucket, symbol, side, notional_usd,
                        predicted_usd, realized_usd, exchange, network, source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (hour_bucket, symbol, side, exchange, network, source)
                    DO UPDATE SET
                        notional_usd  = EXCLUDED.notional_usd,
                        predicted_usd = EXCLUDED.predicted_usd
                    """,
                    rows,
                )
        except Exception as e:
            logger.debug(f"funding snapshot write failed (non-fatal): {e}")

    async def _verify_isolated_or_close(self, symbol: str, side: TradeSide) -> None:
        """FUT-RM-07: defense-in-depth on MB-17.

        Re-reads the position from the exchange right after a fill and
        confirms margin_type == ISOLATED. If the readback shows CROSS
        (stale account-level setting, prior session leak, or a
        110026/110043 false-positive from Bybit's "idempotency" branch)
        we immediately close the position to bound risk to one trade's
        margin requirement instead of the whole account.

        No-op when:
          - executor lacks get_position (cannot verify; logs warning)
          - DRY_RUN (no real fill)
          - margin_type reads as None (unknown — fail-open, log)

        Bounded to ~1s of extra latency per entry: a single REST read.
        """
        try:
            executor = self.exchange_client
            if executor is None or not hasattr(executor, 'get_position'):
                logger.warning(
                    f"FUT-RM-07: executor has no get_position(); skipping isolated verify for {symbol}"
                )
                return
            pos = await executor.get_position(symbol)
            if not pos:
                # No position came back — fill failed silently or fully closed.
                # Don't open a second order, just log.
                logger.warning(
                    f"FUT-RM-07: get_position returned empty for {symbol} immediately "
                    f"after fill — fill may have failed; verify externally"
                )
                return
            # Normalize so we have margin_type regardless of source.
            try:
                from modules.futures_trading.exchanges import normalize_position
                src = getattr(self, 'exchange', '') or ''
                normalized = normalize_position(pos, src) or pos
            except Exception:
                normalized = pos
            margin_type = (
                normalized.get('margin_type')
                if isinstance(normalized, dict) else None
            )
            if margin_type is None:
                logger.info(
                    f"FUT-RM-07: margin_type unavailable for {symbol}; "
                    f"skipping enforcement (fail-open)"
                )
                return
            if str(margin_type).upper() != 'ISOLATED':
                logger.error(
                    f"🚨 FUT-RM-07: {symbol} fill landed with margin_type="
                    f"{margin_type!r} but ISOLATED is required. Closing immediately."
                )
                # Capture position size BEFORE the close so the alert payload
                # carries the actual notional that breached.
                breach_size = None
                try:
                    raw_size = (
                        normalized.get('contracts')
                        or normalized.get('size')
                        or normalized.get('positionAmt')
                        or normalized.get('qty')
                        if isinstance(normalized, dict) else None
                    )
                    breach_size = float(raw_size) if raw_size is not None else None
                except (TypeError, ValueError):
                    breach_size = None
                close_ok = True
                close_err_msg = None
                # Best-effort close — same path the SL/TP uses.
                try:
                    await self._close_position(symbol, "fut_rm_07_cross_margin_detected")
                except Exception as close_err:
                    close_ok = False
                    close_err_msg = str(close_err)
                    logger.error(
                        f"FUT-RM-07: emergency close after CROSS detect failed for "
                        f"{symbol}: {close_err}. Operator MUST intervene."
                    )
                # FUT-RM-07b: high-priority Telegram alert. Fail-soft.
                if getattr(self, 'telegram_emergency_close_enabled', True):
                    try:
                        await self._notify_fut_rm_07_emergency_close(
                            symbol=symbol,
                            side=side,
                            actual_margin=str(margin_type),
                            position_size=breach_size,
                            close_ok=close_ok,
                            close_error=close_err_msg,
                        )
                    except Exception as notify_err:
                        logger.warning(
                            f"FUT-RM-07b: Telegram alert dispatch failed "
                            f"(non-fatal) for {symbol}: {notify_err}"
                        )
            else:
                logger.debug(f"FUT-RM-07: {symbol} margin_type verified ISOLATED")
        except Exception as e:
            logger.warning(f"FUT-RM-07 verify errored for {symbol}: {e}")

    async def _notify_fut_rm_07_emergency_close(
        self,
        symbol: str,
        side: 'TradeSide',
        actual_margin: str,
        position_size: Optional[float],
        close_ok: bool,
        close_error: Optional[str],
    ) -> None:
        """FUT-RM-07b (Wave 4): high-priority Telegram alert when the
        FUT-RM-07 verify path detects a CROSS-margin fill and fires the
        emergency-close. Reaches the shared TelegramBotController
        singleton (initialized by main_futures.py at startup). Fail-soft
        on every path — never raises into the caller.
        """
        # Lazy import — telegram_bot has its own optional aiohttp dep and
        # we don't want futures_engine import-time coupling.
        try:
            from monitoring.telegram_bot import get_telegram_controller
        except Exception as imp_err:
            logger.info(
                f"FUT-RM-07b: telegram_bot import unavailable ({imp_err}); "
                f"skipping alert for {symbol}"
            )
            return
        controller = get_telegram_controller()
        # Treat missing bot_token/chat_id as "Telegram not configured" —
        # log a warning and continue; do NOT block the emergency flow.
        if not getattr(controller, 'bot_token', None) or not getattr(
            controller, 'chat_id', None
        ):
            logger.warning(
                f"FUT-RM-07b: Telegram not configured "
                f"(no bot_token/chat_id) — emergency-close alert for "
                f"{symbol} ({actual_margin}) only logged, not pushed."
            )
            return
        side_str = getattr(side, 'value', str(side)).upper()
        size_str = (
            f"{position_size:.6f}"
            if isinstance(position_size, (int, float))
            else "unknown"
        )
        close_status = (
            "CLOSED"
            if close_ok
            else f"CLOSE FAILED: {close_error}"
        )
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        msg = (
            f"FUT-RM-07 EMERGENCY CLOSE\n"
            f"Symbol: {symbol}\n"
            f"Side: {side_str}\n"
            f"Margin (intended -> actual): ISOLATED -> {actual_margin}\n"
            f"Position size: {size_str}\n"
            f"Exchange: {getattr(self, 'exchange', 'unknown')}\n"
            f"Status: {close_status}\n"
            f"Timestamp: {ts}\n"
            f"Operator action: verify the position is flat and "
            f"investigate the CROSS-margin slip."
        )
        try:
            await controller.notify(msg, priority="critical")
        except Exception as notify_err:
            logger.warning(
                f"FUT-RM-07b: controller.notify() failed (non-fatal) "
                f"for {symbol}: {notify_err}"
            )

    async def close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for symbol in list(self.active_positions.keys()):
            await self._close_position(symbol, "manual_close")

        logger.info("✅ All positions closed")

    async def get_stats(self) -> Dict:
        """Get trading statistics"""
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100

        positions_summary = []
        for symbol, pos in self.active_positions.items():
            positions_summary.append({
                'symbol': symbol,
                'side': pos.side.value,
                'entry': pos.entry_price,
                'current': pos.current_price,
                'pnl_pct': f"{pos.unrealized_pnl_pct:.2f}%",
                'pnl_usd': f"${pos.unrealized_pnl:.2f}"
            })

        # Get advanced metrics from PnL tracker
        pnl_snapshot = self.pnl_tracker.get_snapshot()

        # Calculate unrealized PnL for all open positions
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        self.pnl_tracker.update_unrealized_pnl(unrealized_pnl)

        return {
            'mode': 'DRY_RUN' if self.dry_run else 'LIVE',
            'network': 'TESTNET' if self.testnet else 'MAINNET',
            'exchange': self.exchange.upper(),
            'leverage': f"{self.leverage}x",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': f"${self.total_pnl:.2f}",
            'total_fees': f"${self.total_fees:.2f}",
            'net_pnl': f"${pnl_snapshot.net_pnl:.2f}",
            'unrealized_pnl': f"${unrealized_pnl:.2f}",
            'daily_pnl': f"${self.risk_metrics.daily_pnl:.2f}",
            'daily_trades': self.risk_metrics.daily_trades,
            'risk_level': self.risk_metrics.risk_level,
            # Advanced metrics from PnL tracker
            'sharpe_ratio': pnl_snapshot.sharpe_ratio,
            'sortino_ratio': pnl_snapshot.sortino_ratio,
            'calmar_ratio': pnl_snapshot.calmar_ratio,
            'profit_factor': pnl_snapshot.profit_factor,
            'max_drawdown': f"${pnl_snapshot.max_drawdown:.2f}",
            'max_drawdown_pct': f"{pnl_snapshot.max_drawdown_pct:.1f}%",
            'avg_win': f"${pnl_snapshot.avg_win:.2f}",
            'avg_loss': f"${pnl_snapshot.avg_loss:.2f}",
            'best_trade': f"${pnl_snapshot.best_trade:.2f}",
            'worst_trade': f"${pnl_snapshot.worst_trade:.2f}",
            'current_streak': pnl_snapshot.current_streak,
            'max_win_streak': pnl_snapshot.max_win_streak,
            'max_loss_streak': pnl_snapshot.max_loss_streak,
            'active_positions': len(self.active_positions),
            'positions': positions_summary
        }

    async def get_health(self) -> Dict:
        """Get health status for monitoring"""
        exchange_connected = False
        try:
            if self.exchange_client:
                await self.exchange_client.fetch_time()
                exchange_connected = True
        except:
            pass

        # ISSUE-15: non-sensitive account identity so the dashboard/operator
        # can tell WHICH exchange + account is in use. The api_key_secret_name
        # is the env/secret KEY NAME (not the value) the active key was read
        # from; api_key_fingerprint is the masked last-4. Never the full key.
        exch_u = (self.exchange or '').upper()
        api_key_secret_name = (
            f"{exch_u}_TESTNET_API_KEY" if self.testnet else f"{exch_u}_API_KEY"
        )

        return {
            'status': 'healthy' if self.is_running and exchange_connected else 'degraded',
            'engine_running': self.is_running,
            'exchange_connected': exchange_connected,
            'dry_run': self.dry_run,
            'testnet': self.testnet,
            # Account identity (non-sensitive)
            'exchange': self.exchange,
            'network': 'testnet' if self.testnet else 'mainnet',
            'api_key_secret_name': api_key_secret_name,
            'api_key_fingerprint': getattr(self, 'api_key_fingerprint', None),
            'risk_can_trade': self.risk_metrics.can_trade,
            'active_positions': len(self.active_positions),
            'daily_pnl': self.risk_metrics.daily_pnl,
            'daily_loss_limit': self.risk_metrics.daily_loss_limit,
            'consecutive_losses': self.risk_metrics.consecutive_losses
        }

    async def shutdown(self):
        """Shutdown the engine"""
        logger.info("Shutting down futures trading engine...")
        self.is_running = False

        # Close exchange connection
        if self.exchange_client:
            try:
                await self.exchange_client.close()
            except:
                pass

        logger.info("✅ Engine shutdown complete")
