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
    drawdown_pct: float = 0.0
    peak_balance: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Daily loss limit check
        if self.daily_pnl <= -self.daily_loss_limit:
            return False
        # Consecutive losses check (pause after 5 consecutive losses)
        if self.consecutive_losses >= 5:
            return False
        return True

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
        mode: str = "production"
    ):
        """
        Initialize futures trading engine with database-backed configuration

        Args:
            config_manager: FuturesConfigManager instance (loads from database)
            mode: Operating mode
        """
        self.config_manager = config_manager
        self.mode = mode
        self.is_running = False

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
            self.testnet = general_config.testnet

            # Position settings
            self.max_positions = position_config.max_positions
            self.position_size_usd = position_config.position_size_usd

            # Leverage settings
            self.leverage = leverage_config.default_leverage

            # Risk settings
            self.stop_loss_pct = -abs(risk_config.stop_loss_pct)  # Ensure negative
            self.take_profit_pct = abs(risk_config.take_profit_pct)  # Ensure positive
            self.max_daily_loss = risk_config.max_daily_loss_usd
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
            self.verbose_signals = strategy_config.verbose_signals

        else:
            # Fallback to defaults if no config manager (should not happen in production)
            logger.warning("⚠️ No config manager provided, using defaults")
            self.exchange = "binance"
            self.testnet = True
            self.max_positions = 5
            self.position_size_usd = 100.0
            self.leverage = 10
            self.stop_loss_pct = -5.0
            self.take_profit_pct = 10.0
            self.max_daily_loss = 500.0
            self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
            self.signal_timeframe = "15m"  # 15-minute candles for faster signals
            self.scan_interval_seconds = 30  # Scan every 30 seconds
            self.rsi_oversold = 30.0
            self.rsi_overbought = 70.0
            self.rsi_weak_oversold = 40.0
            self.rsi_weak_overbought = 60.0
            self.min_signal_score = 3  # Lower threshold for more signals
            self.verbose_signals = True
            self.cooldown_duration = timedelta(minutes=5)

        # DRY_RUN mode - CRITICAL: Check environment variable (global safety setting)
        dry_run_env = os.getenv('DRY_RUN', 'true').strip().lower()
        self.dry_run = dry_run_env in ('true', '1', 'yes')

        # Trading state
        self.active_positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.trade_history: List[Trade] = []

        # Cooldowns (symbol -> next_trade_time)
        self.symbol_cooldowns: Dict[str, datetime] = {}

        # Exchange client
        self.exchange_client = None

        # Risk metrics
        self.risk_metrics = RiskMetrics(
            daily_loss_limit=self.max_daily_loss,
            last_reset=datetime.now()
        )

        # Price cache for simulated trading
        self._price_cache: Dict[str, Dict] = {}
        self._last_price_update: Dict[str, datetime] = {}

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

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

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
                # Fallback to direct env read
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

            self.exchange_client = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })

            # Enable sandbox/testnet mode
            if sandbox_mode:
                self.exchange_client.set_sandbox_mode(True)

            # Load markets
            await self.exchange_client.load_markets()
            logger.info(f"✅ Loaded {len(self.exchange_client.markets)} markets")

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
                # Fallback to direct env read
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
                if not self.dry_run or self.testnet:
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
        if self.dry_run:
            logger.info("DRY_RUN mode: Skipping position sync from exchange")
            return

        try:
            positions = await self.exchange_client.fetch_positions()
            for pos in positions:
                if pos['contracts'] and float(pos['contracts']) > 0:
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
                    logger.info(f"📊 Synced position: {symbol} {side.value} @ {position.entry_price}")

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")

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

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # 1. Check risk limits
            if not self.risk_metrics.can_trade:
                logger.warning(f"⚠️ Trading paused - Risk limit reached (Daily PnL: ${self.risk_metrics.daily_pnl:.2f})")
                return

            # 2. Monitor existing positions
            await self._monitor_positions()

            # 3. Check for new opportunities
            await self._scan_opportunities()

            # 4. Execute pending orders
            await self._process_orders()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _monitor_positions(self):
        """Monitor active positions for exit signals"""
        if not self.active_positions:
            return

        for symbol, position in list(self.active_positions.items()):
            try:
                # Get current price
                ticker = await self._get_ticker(symbol)
                if not ticker or 'last' not in ticker:
                    continue

                current_price = float(ticker['last'])
                position.current_price = current_price

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

                # Check exit conditions
                exit_reason = await self._check_exit_conditions(position)
                if exit_reason:
                    await self._close_position(symbol, exit_reason)

            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {e}")

    async def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be closed and return reason"""
        pnl_pct = position.unrealized_pnl_pct

        # Stop loss check
        if pnl_pct <= self.stop_loss_pct:
            return "stop_loss"

        # Take profit check
        if pnl_pct >= self.take_profit_pct:
            return "take_profit"

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

        # Check for signal reversal (if we have strong opposite signal)
        signals = await self._get_technical_signals(position.symbol)
        if signals:
            if position.side == TradeSide.LONG and signals.overall_signal == SignalStrength.STRONG_SELL:
                return "signal_reversal"
            elif position.side == TradeSide.SHORT and signals.overall_signal == SignalStrength.STRONG_BUY:
                return "signal_reversal"

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
                if signal_score >= self.min_signal_score:
                    entry_side = TradeSide.LONG
                    if self.verbose_signals:
                        logger.info(f"     ✅ LONG signal triggered (score {signal_score} >= {self.min_signal_score})")
                elif signal_score <= -self.min_signal_score:
                    entry_side = TradeSide.SHORT
                    if self.verbose_signals:
                        logger.info(f"     ✅ SHORT signal triggered (score {signal_score} <= -{self.min_signal_score})")
                else:
                    if self.verbose_signals:
                        if signal_score > 0:
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

    async def _get_technical_signals(self, symbol: str) -> Optional[TechnicalSignals]:
        """Calculate technical indicators for a symbol using configurable timeframe"""
        try:
            # Fetch OHLCV data using configured timeframe (default: 15m for faster signals)
            ohlcv = await self.exchange_client.fetch_ohlcv(symbol, self.signal_timeframe, limit=100)
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

            if histogram > 0 and macd > signal_line:
                signals.macd_signal = SignalStrength.BUY if histogram > 0.001 else SignalStrength.NEUTRAL
            elif histogram < 0 and macd < signal_line:
                signals.macd_signal = SignalStrength.SELL if histogram < -0.001 else SignalStrength.NEUTRAL

            # Amplify MACD signal for strong momentum
            if abs(histogram) > 0.005:
                if histogram > 0:
                    signals.macd_signal = SignalStrength.STRONG_BUY
                else:
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
                if current_price <= signals.bb_lower:
                    signals.bb_position = "below_lower"
                    signals.bb_signal = SignalStrength.STRONG_BUY  # Oversold, expect bounce
                elif current_price >= signals.bb_upper:
                    signals.bb_position = "above_upper"
                    signals.bb_signal = SignalStrength.STRONG_SELL  # Overbought, expect pullback
                elif current_price < signals.bb_middle:
                    signals.bb_position = "lower_half"
                    signals.bb_signal = SignalStrength.BUY  # Below middle, slight bullish
                elif current_price > signals.bb_middle:
                    signals.bb_position = "upper_half"
                    signals.bb_signal = SignalStrength.SELL  # Above middle, slight bearish
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
        """Calculate MACD (12, 26, 9)"""
        def ema(data: List[float], period: int) -> float:
            if len(data) < period:
                return sum(data) / len(data)
            multiplier = 2 / (period + 1)
            ema_value = sum(data[:period]) / period
            for price in data[period:]:
                ema_value = (price - ema_value) * multiplier + ema_value
            return ema_value

        ema_12 = ema(closes, 12)
        ema_26 = ema(closes, 26)
        macd_line = ema_12 - ema_26

        # For signal line, we'd need historical MACD values
        # Simplified: use recent EMA as approximation
        signal_line = macd_line * 0.9  # Simplified approximation
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

    async def _open_position(self, symbol: str, side: TradeSide, signals: TechnicalSignals):
        """Open a new position"""
        try:
            # Get current price
            ticker = await self._get_ticker(symbol)
            if not ticker:
                return

            current_price = float(ticker['last'])

            # Calculate position size
            size = self.position_size_usd / current_price
            notional = self.position_size_usd

            # Calculate fees (estimate)
            fee_rate = self.BINANCE_TAKER_FEE if self.exchange == 'binance' else self.BYBIT_TAKER_FEE
            estimated_fees = notional * fee_rate * 2  # Entry + exit

            # Calculate stop loss and take profit prices
            if side == TradeSide.LONG:
                stop_loss_price = current_price * (1 + self.stop_loss_pct / 100 / self.leverage)
                take_profit_price = current_price * (1 + self.take_profit_pct / 100 / self.leverage)
                # Liquidation estimate (simplified)
                liquidation_price = current_price * (1 - 0.9 / self.leverage)
            else:
                stop_loss_price = current_price * (1 - self.stop_loss_pct / 100 / self.leverage)
                take_profit_price = current_price * (1 - self.take_profit_pct / 100 / self.leverage)
                liquidation_price = current_price * (1 + 0.9 / self.leverage)

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
                metadata={'signals': {
                    'rsi': signals.rsi,
                    'macd': signals.macd_histogram,
                    'volume_ratio': signals.volume_ratio,
                    'trend': signals.trend
                }}
            )

            # Execute order (or simulate)
            if self.dry_run:
                logger.info(f"🔵 [DRY_RUN] SIMULATED {side.value.upper()} {symbol}")
                logger.info(f"   Entry: ${current_price:.2f}, Size: {size:.6f}, Notional: ${notional:.2f}")
                logger.info(f"   SL: ${stop_loss_price:.2f}, TP: ${take_profit_price:.2f}")
            else:
                # Execute real order
                order_side = 'buy' if side == TradeSide.LONG else 'sell'
                try:
                    order = await self.exchange_client.create_market_order(
                        symbol=symbol,
                        side=order_side,
                        amount=size,
                        params={'leverage': self.leverage}
                    )
                    logger.info(f"🟢 EXECUTED {side.value.upper()} {symbol}")
                    logger.info(f"   Order ID: {order['id']}")
                    logger.info(f"   Entry: ${current_price:.2f}, Size: {size:.6f}")

                    # Update entry price from actual fill
                    if order.get('average'):
                        position.entry_price = float(order['average'])

                except Exception as e:
                    logger.error(f"❌ Order execution failed: {e}")
                    return

            # Add to active positions
            self.active_positions[symbol] = position
            self.risk_metrics.current_exposure += notional

            logger.info(f"✅ Position opened: {symbol} {side.value.upper()}")
            logger.info(f"   Active positions: {len(self.active_positions)}/{self.max_positions}")

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
            if self.dry_run:
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

            # Remove from active positions
            del self.active_positions[symbol]

            # Set cooldown
            self.symbol_cooldowns[symbol] = datetime.now() + self.cooldown_duration

            # Log result
            pnl_emoji = "🟢" if net_pnl > 0 else "🔴"
            sim_tag = "[DRY_RUN] " if position.is_simulated else ""
            logger.info(f"{pnl_emoji} {sim_tag}Position closed: {symbol}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Entry: ${position.entry_price:.2f} → Exit: ${position.current_price:.2f}")
            logger.info(f"   PnL: ${net_pnl:.2f} ({leveraged_pnl_pct:.2f}%)")
            logger.info(f"   Fees: ${total_fees:.2f}")
            logger.info(f"   Daily PnL: ${self.risk_metrics.daily_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    async def _process_orders(self):
        """Process pending orders"""
        # Currently not implementing limit orders - using market orders only
        pass

    async def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get current ticker for symbol"""
        try:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

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

        return {
            'status': 'healthy' if self.is_running and exchange_connected else 'degraded',
            'engine_running': self.is_running,
            'exchange_connected': exchange_connected,
            'dry_run': self.dry_run,
            'testnet': self.testnet,
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
