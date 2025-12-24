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
            self.capital_allocation = position_config.capital_allocation

            # Dynamic position sizing settings
            self.dynamic_position_sizing = getattr(position_config, 'dynamic_position_sizing', True)
            self.min_position_pct = getattr(position_config, 'min_position_pct', 5.0)
            self.max_position_pct = getattr(position_config, 'max_position_pct', 20.0)
            self.max_position_usd = getattr(position_config, 'max_position_usd', 500.0)
            self.static_position_pct = getattr(position_config, 'static_position_pct', 15.0)
            self.position_size_usd = position_config.position_size_usd  # Legacy fallback
            self.min_trade_size = getattr(position_config, 'min_trade_size', 10.0)

            # Leverage settings
            self.leverage = leverage_config.default_leverage
            self.max_leverage = getattr(leverage_config, 'max_leverage', 20)

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
            self.trailing_stop_distance = getattr(risk_config, 'trailing_stop_distance', 1.5)

            # Calculate max_daily_loss_usd from percentage and capital
            # If max_daily_loss_pct is set (from UI), use that. Otherwise use max_daily_loss_usd directly.
            if risk_config.max_daily_loss_pct and risk_config.max_daily_loss_pct > 0:
                self.max_daily_loss = self.capital_allocation * (risk_config.max_daily_loss_pct / 100)
                logger.info(f"Daily loss limit: ${self.max_daily_loss:.2f} ({risk_config.max_daily_loss_pct}% of ${self.capital_allocation})")
            else:
                self.max_daily_loss = risk_config.max_daily_loss_usd
                logger.info(f"Daily loss limit: ${self.max_daily_loss:.2f} (fixed USD)")

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
        self.price_client = None  # Mainnet client for accurate prices in DRY_RUN mode

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

        # Telegram alerts
        self.telegram_alerts = None
        try:
            from futures_trading.core.futures_alerts import FuturesTelegramAlerts
            self.telegram_alerts = FuturesTelegramAlerts()
            if self.telegram_alerts.enabled:
                logger.info("✅ Telegram alerts enabled for Futures module")
        except ImportError as e:
            logger.warning(f"Telegram alerts not available: {e}")

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

            # Create mainnet price client for accurate prices (especially in DRY_RUN mode)
            # This ensures we always get live prices from mainnet, regardless of testnet setting
            if self.dry_run or self.testnet:
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
                            'options': {
                                'defaultType': 'future',
                                'adjustForTimeDifference': True,
                            }
                        })
                    else:
                        # Public client without credentials (can still fetch prices)
                        self.price_client = ccxt.binance({
                            'enableRateLimit': True,
                            'options': {
                                'defaultType': 'future',
                            }
                        })

                    await self.price_client.load_markets()
                    logger.info("✅ Mainnet price client initialized for accurate live prices")
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
        """Monitor active positions for exit signals and partial TPs"""
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
        if position.trailing_stop_price is None and actual_price_change_pct <= -self.stop_loss_pct:
            return "SL Hit"

        # Take profit check (legacy single TP - only if NOT using multiple tp_levels)
        # Skip this check if position has tp_levels (handled by partial TPs)
        # TP triggers when price moves in favor by >= take_profit_pct
        if not position.tp_levels and actual_price_change_pct >= self.take_profit_pct:
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

    def _calculate_position_size(self, signals: TechnicalSignals) -> float:
        """
        Calculate position size based on dynamic or static settings

        Dynamic sizing: Adjusts position size based on signal strength
        - Stronger signals → larger positions (up to max_position_pct)
        - Weaker signals → smaller positions (down to min_position_pct)

        Static sizing: Uses fixed percentage of capital

        Returns: Position size in USD (notional value before leverage)
        """
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
                        # TP1 hit - move stop to breakeven (always enabled for capital protection)
                        position.trailing_stop_price = position.entry_price
                        logger.info(f"🔒 {position.symbol}: Stop moved to breakeven ${position.entry_price:.4f}")
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
            if self.dry_run:
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

            # Calculate position size based on settings
            notional = self._calculate_position_size(signals)
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

            # Calculate multiple take profit levels
            tp_levels = self._calculate_tp_levels(current_price, side)

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
                }},
                tp_levels=tp_levels,
                original_size=size,
                highest_price=current_price if side == TradeSide.LONG else None,
                lowest_price=current_price if side == TradeSide.SHORT else None
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

            # Remove from active positions
            del self.active_positions[symbol]

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
