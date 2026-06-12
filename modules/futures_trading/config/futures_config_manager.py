"""
Futures Trading Module Configuration Manager
Database-backed configuration with .env integration for sensitive data

Architecture:
- All trading configuration is stored in the database (config_settings table)
- Only sensitive data (API keys, private keys) is read from .env
- Settings page writes/reads from database via API endpoints
- Hot-reload support for live configuration changes
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, ClassVar, Set
import os
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class FuturesConfigType(Enum):
    """Futures configuration types - maps to config_type in database"""
    GENERAL = "futures_general"
    POSITION = "futures_position"
    LEVERAGE = "futures_leverage"
    RISK = "futures_risk"
    PAIRS = "futures_pairs"
    STRATEGY = "futures_strategy"
    FUNDING = "futures_funding"


# Pydantic models for validation

class FuturesGeneralConfig(BaseModel):
    """General futures trading configuration"""
    enabled: bool = False
    exchange: str = "binance"  # binance or bybit
    # CRITICAL: Default to mainnet (False) for live trading safety
    # Set testnet=True in database settings if you want to use testnet
    # Or set FUTURES_TESTNET=true in environment
    testnet: bool = False
    contract_type: str = "perpetual"


class FuturesPositionConfig(BaseModel):
    """Position sizing configuration"""
    capital_allocation: float = 300.0
    # Dynamic position sizing
    dynamic_position_sizing: bool = True  # Enable/disable dynamic sizing
    min_position_pct: float = 5.0  # Minimum position size as % of capital
    max_position_pct: float = 20.0  # Maximum position size as % of capital
    max_position_usd: float = 500.0  # Maximum position size in USD (cap)
    # Static position sizing (when dynamic is disabled)
    static_position_pct: float = 15.0  # Fixed position size as % of capital
    position_size_usd: float = 100.0  # Legacy: fixed USD position size (deprecated)
    max_positions: int = 5
    min_trade_size: float = 10.0

    # FUT-RM-06: ATR-based per-symbol sizing.
    # When enabled, the engine sizes positions so that an `atr_stop_multiplier`
    # × ATR move costs `atr_risk_pct` of capital_allocation per trade.
    # Result: a volatile BTC trade and a quiet ALGO trade risk the same $.
    # Multiplies cleanly with leverage — notional = (risk_$ / (ATR * stop_mult))
    #   × price × leverage. Then capped by max_position_usd + min_trade_size.
    atr_sizing_enabled: bool = False
    atr_risk_pct: float = 1.0   # % of capital_allocation risked per trade
    atr_stop_multiplier: float = 1.5  # SL distance in ATR units


class FuturesLeverageConfig(BaseModel):
    """Leverage configuration"""
    # FUT-RM-18 (Wave 5): lowered from 10x to 5x. 10x left too little room
    # before the static 2% SL triggered. Migration 031 lowers the seeded
    # DB value to match. max_leverage stays 20x for opt-in per-trade
    # aggression via the per-symbol override table (FUT-RM-08).
    default_leverage: int = 5
    max_leverage: int = 20
    margin_mode: str = "isolated"  # isolated or cross
    # FUT-RM-07: defense-in-depth on MB-17. When True, the engine verifies
    # margin_type=ISOLATED via a position-read AFTER placing the entry
    # order and immediately closes if a CROSS-margin fill is detected.
    # No-op in DRY_RUN.
    enforce_isolated_margin: bool = True
    # FUT-RM-07b (Wave 4): high-priority Telegram alert on the
    # FUT-RM-07 emergency-close path. Default True so an operator
    # always learns when a CROSS-margin fill slipped through the
    # MB-17 set_margin_type call and got force-closed. Fail-soft:
    # if Telegram is not configured the engine just logs.
    telegram_emergency_close_enabled: bool = True
    # FUT-RM-08 (Wave 3): per-symbol leverage cap overrides. Operator may
    # want different caps per pair (e.g. max 5x on PEPE/USDT but 10x on
    # BTC/USDT). When the validator runs, override > global max_leverage.
    # Keys are exchange-native symbols (e.g. "BTC/USDT" or "BTCUSDT");
    # the resolver normalizes case + slash before lookup. Empty dict
    # means "use global max_leverage for every pair" (current behavior).
    max_leverage_overrides: Dict[str, int] = Field(default_factory=dict)


class FuturesRiskConfig(BaseModel):
    """Risk management configuration

    IMPORTANT: For profitability, TP1 must be > SL and position sizing must be front-loaded.

    Risk/Reward Math Example (with defaults below):
    - SL at 1.2%: Max loss = 1.2% of position
    - TP1 at 1.8% (40% closed): Lock in 0.72% profit
    - TP2 at 3.5% (30% closed): Lock in 1.05% more
    - After TP2: Stop moves to breakeven, risk-free ride to TP3/TP4

    With 54% win rate and these settings:
    - Expected TP1 profit: 0.54 × 1.8% × 40% = 0.39% per trade (partial)
    - Expected SL loss: 0.46 × 1.2% × 100% = 0.55% per trade
    - But TP2+ adds extra profit making overall EV positive
    """
    # CRITICAL: SL must be LESS than TP1 for positive expectancy
    stop_loss_pct: float = 1.2  # Tighter stop - cut losses quickly (was 2.0)

    # Multiple Take Profits - Front-loaded for early profit capture
    # TP1 should be wider than SL to ensure R:R > 1
    tp1_pct: float = 1.8  # First TP at 1.8% (wider than SL) - was 2.0
    tp2_pct: float = 3.5  # Second TP at 3.5% - was 4.0
    tp3_pct: float = 6.0  # Third TP at 6% (unchanged)
    tp4_pct: float = 10.0  # Fourth TP at 10% (unchanged)

    # Position size distribution - FRONT-LOADED for early profit locking
    # Take more profit early to ensure wins outweigh losses
    tp1_size_pct: float = 40.0  # Close 40% at TP1 (was 25%) - lock in profits early
    tp2_size_pct: float = 30.0  # Close 30% at TP2 (was 25%) - capture momentum
    tp3_size_pct: float = 20.0  # Close 20% at TP3 (was 25%)
    tp4_size_pct: float = 10.0  # Close remaining 10% at TP4 (was 25%) - let runners run

    # Legacy single TP (deprecated, use tp1_pct instead)
    take_profit_pct: float = 4.0

    # Daily loss limits
    max_daily_loss_usd: float = 500.0
    max_daily_loss_pct: float = 5.0
    liquidation_buffer: float = 20.0

    # Trailing stop - tighter for better profit protection
    trailing_stop_enabled: bool = True
    trailing_stop_distance: float = 1.0  # Tighter trailing (was 1.5)

    # Risk controls
    max_consecutive_losses: int = 4  # Reduced from 5 to pause earlier

    # FUT-RM-10 (Wave 3): auto-deleverage on drawdown. When enabled, the
    # monitor loop checks should_auto_deleverage(total_pnl, capital) every
    # cycle. On trigger, the position with the worst unrealized PnL is
    # halved (close 50% at market). Default OFF — operator must opt in
    # via the settings page after canary.
    auto_deleverage_enabled: bool = False
    # Throttle so a single drawdown event doesn't fire on every cycle.
    auto_deleverage_cooldown_seconds: int = 600   # 10 min between triggers

    # Market condition filters
    # Relaxed for live trading - strict filters were rejecting all trades in sideways markets
    require_trend_confirmation: bool = False  # Allow trading in sideways markets (was True)
    min_volume_multiplier: float = 0.8  # Allow 80% of average volume (was 1.2)

    # LIVE-flip safety (mig 109): when live orders are blocked (killswitch /
    # pause / DRY_RUN flipped back on) the engine refuses to paper-close a
    # LIVE position (which would orphan real exchange exposure). Set True to
    # instead allow REAL reduce-only closes in that state (risk-reducing
    # orders only). Default False = no orders while blocked.
    reduce_only_close_when_paused: bool = False

    # FUT-RM-21 (Wave 7): regime gate. The signal stack mixes mean-reversion
    # (RSI extremes scored as STRONG_BUY/SELL) with trend-following (Bollinger
    # breakout, EMA cross) and sums them additively — so the engine happily
    # buys a downtrend on an RSI bounce. That is the classic "catching a
    # falling knife" loss the -$59.99 book kept paying. This gate is stricter
    # than require_trend_confirmation: it HARD-BLOCKS counter-trend entries
    # (no LONG when SMA20<SMA50 downtrend; no SHORT in an uptrend) while still
    # allowing both directions in a `sideways` regime (range mean-reversion is
    # legitimate there). Default ON. Set False to revert to the additive-only
    # behaviour.
    block_counter_trend_entries: bool = True

    # FUT-RM-16 (Wave 5): ATR-scaled SL/TP per symbol. The static 1.2% / 1.8%
    # SL/TP that ships above is the right number for a quiet majors book but
    # gets stopped out instantly on a vol-name (FIL, NEAR, AAVE all moved 4%+
    # against the operator at 10x). When enabled, SL becomes
    #   max(atr_sl_min_pct, atr_sl_multiplier * ATR_pct)
    # and TP becomes atr_tp_rr_ratio * SL distance (i.e. enforced R:R).
    # Defaults are tuned to keep the existing 1.2% / 1.8%≈1.5R behavior for
    # quiet symbols (ATR_pct ~0.8%) while widening for volatile ones. Set
    # atr_dynamic_sl_tp_enabled=False to revert to the static SL/TP above.
    atr_dynamic_sl_tp_enabled: bool = True
    atr_sl_multiplier: float = 1.5     # SL = max(atr_sl_min_pct, 1.5 × ATR%)
    atr_sl_min_pct: float = 1.5        # Floor on SL distance (price %)
    atr_tp_rr_ratio: float = 2.0       # TP1 = 2 × SL distance

    # FUT-RM-17 (Wave 5): per-symbol consecutive-loss cool-off. After
    # `post_loss_cooloff_threshold` losses in a row on the same symbol,
    # FuturesRiskManager refuses new entries on that pair for
    # `post_loss_cooloff_minutes` minutes. A winning trade resets the
    # per-symbol counter and clears any active cool-off.
    post_loss_cooloff_threshold: int = 2
    post_loss_cooloff_minutes: int = 240

    # FUT-RM-23 (Wave 14): intraday max-hold cap.
    # 15m-signal trades that drift unclosed for hours bleed funding + fees
    # with no incremental edge. When a position exceeds max_hold_minutes AND
    # has not hit TP1 yet (trailing_stop_price == None means stop is still at
    # entry SL, not yet moved to breakeven), it is time-exited.
    # Wave-14 DRY_RUN data: avg_hold 231-1825 min caused net -$113 on 18
    # symbols in 21h. 0 = disabled. Default 240 min = 16 × the 15m signal bar.
    max_hold_minutes: int = 240

    # FUT-RM-27 (Wave 25): per-symbol tiering + rolling performance gate.
    # Driven by a week of live data (442 trades, +$31.05, 55.7% WR overall):
    # the winner tier (BCH/AAVE/ETH/ADA/DOGE) earned ~+$74 while the loser
    # tier (SUI/NEAR/ZEC/DOT/AVAX/FIL) burned ~-$83 — cutting the loser tier
    # roughly triples PnL. Two layers:
    #   1. symbol_size_weights: operator-curated per-symbol size multiplier
    #      (0 = disabled, 0<w<1 = reduced, missing = 1.0). Migration 088
    #      seeds the loser tier at 0.
    #   2. Rolling gate: auto-BENCH a symbol when its trailing
    #      rolling_gate_window trades have net PnL < rolling_gate_max_net_pnl_usd
    #      AND win rate < rolling_gate_max_win_rate; auto-UNBENCH after
    #      rolling_gate_bench_minutes into a probation window at
    #      rolling_gate_probation_weight × size for rolling_gate_min_trades
    #      trades. Keeps the tiering current without manual curation (ZEC
    #      re-entered 42×/week through the expiring 4h cool-off).
    symbol_tiering_enabled: bool = True
    symbol_size_weights: Dict[str, float] = Field(default_factory=dict)
    rolling_gate_enabled: bool = True
    rolling_gate_window: int = 20          # trailing-N trades per symbol
    rolling_gate_min_trades: int = 10      # min closes before gate can fire
    rolling_gate_max_net_pnl_usd: float = -5.0   # bench when net < this...
    rolling_gate_max_win_rate: float = 0.45      # ...AND win rate < this
    rolling_gate_bench_minutes: int = 1440       # 24h bench
    rolling_gate_probation_weight: float = 0.5   # post-unbench size factor

    # FUT-RM-24 (Wave 14): signal-reversal threshold for early exit.
    # Pre-Wave-14 code required reversal_score <= -6 (all 5 indicators
    # strongly reversed) which is practically impossible — max score is
    # ±10 and 6 requires 3 STRONG + any 1 other fully aligned. Lowering
    # to 4 = 2 strong reversals aligns with entry threshold and allows
    # the path to actually fire. Minimum 3 to avoid whipsaw exits on noise.
    signal_reversal_threshold: int = 4


class FuturesPairsConfig(BaseModel):
    """Trading pairs configuration"""
    allowed_pairs: str = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT"
    both_directions: bool = True
    preferred_direction: str = "both"  # long, short, both

    @property
    def pairs_list(self) -> List[str]:
        """Get allowed pairs as a list"""
        return [p.strip() for p in self.allowed_pairs.split(',')]


class FuturesStrategyConfig(BaseModel):
    """Strategy parameters configuration

    Signal Quality:
    - 5 indicators (RSI, MACD, Volume, Bollinger, EMA) each score -2 to +2
    - Total score range: -10 to +10
    - Higher min_signal_score = fewer but higher quality trades
    - Recommended: 4-5 for balanced, 6+ for conservative
    """
    signal_timeframe: str = "15m"  # Timeframe for signal analysis: 1m, 5m, 15m, 30m, 1h, 4h
    scan_interval_seconds: int = 30  # How often to scan for opportunities

    # RSI thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_weak_oversold: float = 40.0
    rsi_weak_overbought: float = 60.0

    # Signal quality threshold - CRITICAL for profitability
    # Higher = fewer trades but better win rate
    min_signal_score: int = 4  # Increased from 3 for better entries

    # FUT-RM-15 (Wave 5): multi-indicator CONFLUENCE gate. The aggregate
    # signal_score above only checks SIGNED magnitude — a single very-strong
    # indicator (e.g. STRONG_BUY RSI alone, +2) plus weak agreement can clear
    # the +4 bar after generous rounding. After Wave-5 audit of 4 losing trades
    # (AAVE / FIL / NEAR shorts, all hit SL at -20% on 10x), we now ALSO
    # require at least N of the 4 directional indicators
    # {RSI extreme, MACD cross, Bollinger touch, EMA alignment} to point the
    # same way before opening. Volume is excluded — it's a confirmer, not a
    # direction-giver. Default 2 keeps reasonable trade frequency while
    # rejecting single-indicator setups. Set 0 to disable.
    min_signal_confluence_count: int = 2

    # FUT-RM-19 (Wave 7): fee + funding aware minimum-edge gate.
    # The -$59.99 @ 39% loss was dominated by fee/funding bleed: at 39% win
    # rate the strategy churns trades whose first realistic target (TP1) does
    # not clear round-trip taker fees (Bybit 0.06% × 2 = 0.12%) + slippage +
    # adverse funding. Before opening, the engine computes:
    #   net_edge_pct = TP1_distance_pct
    #                  - 2*taker_fee_pct - slippage_pct - funding_drag_pct
    # and refuses entry unless net_edge_pct >= min_net_edge_pct. This directly
    # subtracts costs from the expected move (the working-rule edge formula:
    # funding*notional - taker_fees*2 - slippage - liquidation_premium). Set
    # min_net_edge_pct=0 to disable the gate (NOT recommended for live).
    min_edge_gate_enabled: bool = True
    min_net_edge_pct: float = 0.30        # TP1 must beat costs by >= 0.30%
    edge_slippage_pct: float = 0.05       # modeled round-trip slippage (price %)
    # funding_drag_pct is computed live from the current funding rate when the
    # price_client exposes it; this is the conservative fallback used when the
    # funding rate is unavailable (per-interval, expressed as price %).
    edge_funding_fallback_pct: float = 0.05

    # FUT-RM-20 (Wave 7): one-entry-per-candle throttle. Scanning every 30s on
    # a 15m candle re-evaluates the SAME bar ~30 times; without this the engine
    # can fire repeatedly into the same chop. When enabled, a symbol that was
    # scanned/entered within the current signal-timeframe candle is skipped.
    one_entry_per_candle: bool = True

    # Wave-24 FUT-RM-26: hard kill of NEW entries (neutralization switch).
    # When true the engine opens NO new momentum or carry positions — existing
    # positions are still monitored and exited normally (SL/TP/time/manual).
    # This is the futures equivalent of the wave-18 budget=0 neutralization for
    # sniper/arb: budget_usd_futures=0 alone is a no-op here because the
    # allocation guard treats 0 as "unlimited", so suppression must be an
    # explicit engine gate. Default False (no behavior change on existing DBs);
    # migration 066 seeds it True after the wave-24 strategy review concluded
    # the momentum stack is structurally unprofitable. Flip back to false (or
    # delete the row) to re-enable entries.
    entries_suppressed: bool = False

    # Additional filters for trade quality
    require_trend_alignment: bool = True  # Trade only in direction of trend
    require_volume_confirmation: bool = True  # Require above-average volume

    verbose_signals: bool = True
    cooldown_minutes: int = 10  # Increased from 5 to avoid overtrading after losses


class FuturesFundingConfig(BaseModel):
    """Funding rate settings.

    Funding economics:
    - Perps converge to spot via funding payments. Positive funding => longs
      pay shorts every funding interval (8h on Binance/Bybit USDT perps).
    - Annualized: APR ~ funding_rate * 3 * 365 = funding_rate * 1095.
    - 10 bps per 8h = ~109% APR — at that point a fresh long is paying more
      in funding than most strategies can earn in price drift, so we gate it.
    """
    funding_arbitrage_enabled: bool = False
    max_funding_rate: float = 0.1

    # FUT-RM-05: funding-rate gate for directional entries.
    # When current funding > skip_long_funding_bps, refuse new LONG entries
    # (longs pay funding). When funding < -skip_short_funding_bps, refuse
    # new SHORT entries. Units: basis points of the per-interval rate
    # (1 bp = 0.0001). Zero disables the gate on that side.
    skip_long_funding_bps: float = 5.0   # ~55% APR ceiling for longs
    skip_short_funding_bps: float = 5.0  # symmetric for shorts
    # Stale funding rate is worse than no funding rate — if the rate older
    # than this many seconds, skip the gate rather than gate on stale data.
    max_funding_age_seconds: int = 900   # 15 min

    # FUT-RM-25 (Wave 14): funding-rate carry strategy.
    # When a symbol's per-interval funding rate exceeds carry_min_funding_bps,
    # enter the SHORT side to collect funding payments as primary edge.
    # The carry edge = funding_rate * notional * intervals_held. At 8 bps/8h
    # that is ~87% APR on notional — net carry after 1 interval (8h) on a
    # $100 notional = 100 * 0.0008 = $0.08. At 5x leverage that is $0.40
    # net vs Bybit taker round-trip ~$0.12; carry breakeven is ~1.5 intervals.
    # Exit when funding drops below carry_exit_funding_bps OR when the
    # carry position hits its normal SL/TP/max-hold.
    # Default OFF — operator must opt in after observing funding data in the
    # dashboard funding-forecast widget (FUT-RM-09b) for several days.
    funding_carry_enabled: bool = False
    # Entry threshold: per-interval rate in bps (0.01% = 1 bp).
    # 8 bps ≈ 87%/yr APR. Default 8 to ensure carry clears fees in ≤2 intervals.
    carry_min_funding_bps: float = 8.0
    # Exit threshold: close carry positions when funding drops below this.
    # Below 3 bps the carry APR (~33%/yr) no longer compensates for position
    # risk; the normal SL/TP/max-hold would close first anyway.
    carry_exit_funding_bps: float = 3.0
    # Cap on simultaneous carry positions to bound the carry-specific book.
    carry_max_positions: int = 2
    # Max hold for carry positions (minutes). Funding is 8h-periodic; cap at
    # 2 intervals (960 min) so the strategy does not outlast the signal.
    # 0 = use the engine's global max_hold_minutes.
    carry_max_hold_minutes: int = 960

    # FUT-QC-01 (carry v2, Wave-26): bidirectional, STABILITY-GATED funding
    # carry — see modules/futures_trading/strategies/funding_carry.py for the
    # full edge/cost model. Differences from FUT-RM-25 above:
    #   - bidirectional (negative funding -> LONG perp, shorts pay longs)
    #   - entry requires PERSISTENT funding (every sample in a rolling window
    #     beyond the threshold, same sign) instead of a single snapshot
    # Default OFF; when off, ZERO behavior change. Both carry generations can
    # run independently; v2 skips symbols already held by v1 and vice versa.
    futures_funding_carry_enabled: bool = False
    # Entry threshold on |funding| in bps per interval, applied to the WEAKEST
    # sample in the stability window (10 bps ~ breakeven in <2 intervals vs
    # 17 bps round-trip taker+slippage cost).
    carry_min_abs_funding_bps: float = 10.0
    # Minimum number of spaced samples (engine records at most one per 300s
    # funding-cache TTL) required inside the planner's stability window
    # before an entry can arm. Window-span coverage is additionally enforced
    # by the planner itself (min_span_fraction).
    carry_funding_stability_window: int = 3
    # Cap on simultaneous carry-v2 positions (independent of the v1
    # carry_max_positions cap and the momentum max_positions cap).
    carry_max_carry_positions: int = 3


class FuturesConfigManager:
    """
    Configuration manager for Futures Trading Module

    Follows the same pattern as DEX config_manager.py:
    - Database-backed storage for all trading configuration
    - .env integration for sensitive data ONLY (API keys, secrets)
    - Pydantic validation
    - Hot-reload support
    """

    # Sensitive env keys that should consult the encrypted secrets_manager
    # before falling back to .env. O(1) membership test. Mirrors SOLANA pattern.
    SENSITIVE_KEYS: ClassVar[Set[str]] = {
        # Binance
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'BINANCE_TESTNET_API_KEY',
        'BINANCE_TESTNET_API_SECRET',
        # Bybit
        'BYBIT_API_KEY',
        'BYBIT_API_SECRET',
        'BYBIT_TESTNET_API_KEY',
        'BYBIT_TESTNET_API_SECRET',
    }

    def __init__(self, db_pool=None):
        """
        Initialize Futures configuration manager

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool

        # Configuration models
        self.config_models = {
            FuturesConfigType.GENERAL: FuturesGeneralConfig,
            FuturesConfigType.POSITION: FuturesPositionConfig,
            FuturesConfigType.LEVERAGE: FuturesLeverageConfig,
            FuturesConfigType.RISK: FuturesRiskConfig,
            FuturesConfigType.PAIRS: FuturesPairsConfig,
            FuturesConfigType.STRATEGY: FuturesStrategyConfig,
            FuturesConfigType.FUNDING: FuturesFundingConfig,
        }

        # Loaded configs
        self.configs: Dict[FuturesConfigType, BaseModel] = {}

        # Environment config (sensitive data ONLY from .env)
        self._env_config = self._load_environment_config()

        # Load default configs
        for config_type, model_class in self.config_models.items():
            self.configs[config_type] = model_class()

        logger.info("FuturesConfigManager initialized")

    def _load_environment_config(self) -> Dict[str, Any]:
        """
        Load ONLY sensitive configuration from secrets manager or .env

        Priority:
        1. Secrets manager (Docker secrets, encrypted database)
        2. Environment variables (.env fallback)

        Returns:
            Dict: Environment configuration (sensitive data only)
        """
        env_config = {}
        _placeholders = ('null', 'None', '', 'your_testnet_api_key', 'your_mainnet_api_key', 'PLACEHOLDER')

        for var in self.SENSITIVE_KEYS:
            value = None
            try:
                from security.secrets_manager import secrets
                # log_access=True only for *_API_SECRET (the truly sensitive half of the pair)
                value = secrets.get(var, log_access=var.endswith('_API_SECRET'))
            except Exception:
                pass
            if not value:
                value = os.getenv(var)
            if value and value not in _placeholders:
                env_config[var] = value

        return env_config

    async def initialize(self) -> None:
        """Initialize and load all configurations from database"""
        try:
            # Reload sensitive credentials from secrets manager (async)
            # This is needed because __init__ uses sync secrets.get() which skips DB in async context
            await self._reload_sensitive_credentials()
            await self._load_all_configs()
            logger.info("✅ Futures configuration loaded from database")
        except Exception as e:
            logger.error(f"Failed to load Futures config from database: {e}", exc_info=True)
            logger.info("Using default configuration")

    async def set_db_pool(self, db_pool) -> None:
        """
        Set database pool and reload configs

        Args:
            db_pool: Database connection pool
        """
        self.db_pool = db_pool
        if db_pool:
            logger.info("Database pool set, reloading Futures configs...")
            # Reload environment config with secrets manager now that DB is available
            await self._reload_sensitive_credentials()
            await self._load_all_configs()

    async def _reload_sensitive_credentials(self) -> None:
        """Reload API credentials from secrets manager after DB pool is available"""
        try:
            from security.secrets_manager import secrets

            # Initialize secrets manager with db_pool (re-init if in bootstrap mode)
            if self.db_pool and (not secrets._initialized or secrets._db_pool is None or secrets._bootstrap_mode):
                secrets.initialize(self.db_pool)

            sensitive_keys = [
                'BINANCE_API_KEY', 'BINANCE_API_SECRET',
                'BINANCE_TESTNET_API_KEY', 'BINANCE_TESTNET_API_SECRET',
                'BYBIT_API_KEY', 'BYBIT_API_SECRET',
                'BYBIT_TESTNET_API_KEY', 'BYBIT_TESTNET_API_SECRET',
            ]

            for var in sensitive_keys:
                value = await secrets.get_async(var)
                if value and value not in ('null', 'None', '', 'PLACEHOLDER', 'your_testnet_api_key', 'your_mainnet_api_key'):
                    # Check if value is encrypted and needs decryption
                    if value.startswith('gAAAAAB'):
                        decrypted = await self._decrypt_value(value)
                        if decrypted:
                            self._env_config[var] = decrypted
                            logger.debug(f"Loaded and decrypted {var} from secrets manager")
                    else:
                        self._env_config[var] = value
                        logger.debug(f"Loaded {var} from secrets manager")

            logger.info(f"✅ Reloaded {len(self._env_config)} API credentials from secrets manager")

        except Exception as e:
            logger.warning(f"Failed to reload credentials from secrets manager: {e}")

    async def _decrypt_value(self, encrypted_value: str) -> Optional[str]:
        """Decrypt a Fernet-encrypted value"""
        try:
            from pathlib import Path
            from cryptography.fernet import Fernet

            encryption_key = None
            key_file = Path('.encryption_key')
            if key_file.exists():
                encryption_key = key_file.read_text().strip()
            if not encryption_key:
                encryption_key = os.getenv('ENCRYPTION_KEY')

            if encryption_key:
                f = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
                return f.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
        return None

    async def _load_all_configs(self) -> None:
        """Load all configuration types from database"""
        if not self.db_pool:
            logger.warning("No database pool, using default configs")
            return

        for config_type, model_class in self.config_models.items():
            try:
                config = await self._load_config_from_db(config_type)
                if config:
                    self.configs[config_type] = model_class(**config)
                    logger.debug(f"Loaded {config_type.value} from database")
            except Exception as e:
                logger.error(f"Error loading {config_type.value}: {e}")

    async def _load_config_from_db(self, config_type: FuturesConfigType) -> Optional[Dict]:
        """
        Load configuration from database config_settings table

        Args:
            config_type: Configuration type to load

        Returns:
            Optional[Dict]: Configuration data or None
        """
        if not self.db_pool:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT key, value, value_type
                    FROM config_settings
                    WHERE config_type = $1
                    AND is_editable = TRUE
                """, config_type.value)

                if not rows:
                    return None

                config_data = {}
                for row in rows:
                    key = row['key']
                    value = row['value']
                    value_type = row['value_type']

                    # Convert value based on type
                    if value_type == 'int':
                        config_data[key] = int(value)
                    elif value_type == 'float':
                        config_data[key] = float(value)
                    elif value_type == 'bool':
                        config_data[key] = value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == 'json':
                        import json
                        config_data[key] = json.loads(value)
                    else:
                        config_data[key] = value

                return config_data

        except Exception as e:
            logger.error(f"Database error loading {config_type.value}: {e}")
            return None

    async def save_config(self, config_type: FuturesConfigType, config_data: Dict[str, Any],
                         user_id: int = None, reason: str = None) -> bool:
        """
        Save configuration to database

        Args:
            config_type: Configuration type
            config_data: Configuration data to save
            user_id: User ID making the change
            reason: Reason for the change

        Returns:
            bool: True if saved successfully
        """
        if not self.db_pool:
            logger.warning("No database pool, cannot save config")
            return False

        try:
            # Validate with Pydantic model
            model_class = self.config_models[config_type]
            validated_config = model_class(**config_data)

            async with self.db_pool.acquire() as conn:
                for key, value in validated_config.dict().items():
                    # Determine value type
                    if isinstance(value, bool):
                        value_type = 'bool'
                        value_str = str(value).lower()
                    elif isinstance(value, int):
                        value_type = 'int'
                        value_str = str(value)
                    elif isinstance(value, float):
                        value_type = 'float'
                        value_str = str(value)
                    elif isinstance(value, (dict, list)):
                        # FUT-RM-08: store dict/list as JSON so the loader's
                        # value_type=='json' branch round-trips correctly.
                        import json as _json
                        value_type = 'json'
                        value_str = _json.dumps(value)
                    else:
                        value_type = 'string'
                        value_str = str(value)

                    # Get old value for history
                    old_row = await conn.fetchrow("""
                        SELECT value FROM config_settings
                        WHERE config_type = $1 AND key = $2
                    """, config_type.value, key)
                    old_value = old_row['value'] if old_row else None

                    # Update or insert config
                    await conn.execute("""
                        INSERT INTO config_settings (config_type, key, value, value_type, updated_by)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (config_type, key) DO UPDATE
                        SET value = $3,
                            value_type = $4,
                            updated_by = $5,
                            updated_at = NOW()
                    """, config_type.value, key, value_str, value_type, user_id)

                    # Log to history
                    if old_value != value_str:
                        await conn.execute("""
                            INSERT INTO config_history (
                                config_type, key, old_value, new_value,
                                change_source, changed_by, reason
                            )
                            VALUES ($1, $2, $3, $4, 'api', $5, $6)
                        """, config_type.value, key, old_value, value_str, user_id, reason)

            # Update in-memory config
            self.configs[config_type] = validated_config

            logger.info(f"✅ Saved {config_type.value} configuration")
            return True

        except Exception as e:
            logger.error(f"Error saving {config_type.value}: {e}", exc_info=True)
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (can include category prefix like "risk.stop_loss_pct")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        # Check environment config first for sensitive data
        env_key = key.upper().replace('.', '_')
        if env_key in self._env_config:
            return self._env_config[env_key]

        # Parse key (e.g., "risk.stop_loss_pct" or just "stop_loss_pct")
        parts = key.split('.', 1)

        if len(parts) == 2:
            category, field = parts
            # Find config type matching category
            for config_type, config_obj in self.configs.items():
                if category in config_type.value:
                    return getattr(config_obj, field, default)

        # Search all configs for the key
        for config_obj in self.configs.values():
            if hasattr(config_obj, key):
                return getattr(config_obj, key)

        return default

    def get_general(self) -> FuturesGeneralConfig:
        """Get general configuration"""
        return self.configs.get(FuturesConfigType.GENERAL, FuturesGeneralConfig())

    def get_position(self) -> FuturesPositionConfig:
        """Get position configuration"""
        return self.configs.get(FuturesConfigType.POSITION, FuturesPositionConfig())

    def get_leverage(self) -> FuturesLeverageConfig:
        """Get leverage configuration"""
        return self.configs.get(FuturesConfigType.LEVERAGE, FuturesLeverageConfig())

    def get_risk(self) -> FuturesRiskConfig:
        """Get risk configuration"""
        return self.configs.get(FuturesConfigType.RISK, FuturesRiskConfig())

    def get_pairs(self) -> FuturesPairsConfig:
        """Get pairs configuration"""
        return self.configs.get(FuturesConfigType.PAIRS, FuturesPairsConfig())

    def get_strategy(self) -> FuturesStrategyConfig:
        """Get strategy configuration"""
        return self.configs.get(FuturesConfigType.STRATEGY, FuturesStrategyConfig())

    def get_funding(self) -> FuturesFundingConfig:
        """Get funding configuration"""
        return self.configs.get(FuturesConfigType.FUNDING, FuturesFundingConfig())

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a flat dictionary for the settings page

        Returns:
            Dict: All settings with prefixed keys
        """
        all_settings = {}

        for config_type, config_obj in self.configs.items():
            for key, value in config_obj.dict().items():
                # Use the key directly (settings page uses flat keys)
                all_settings[f"futures_{key}"] = value

        # Add flags for API key availability (don't expose actual values)
        all_settings['_has_binance_api'] = bool(self._env_config.get('BINANCE_API_KEY') or
                                                 self._env_config.get('BINANCE_TESTNET_API_KEY'))
        all_settings['_has_bybit_api'] = bool(self._env_config.get('BYBIT_API_KEY') or
                                               self._env_config.get('BYBIT_TESTNET_API_KEY'))

        return all_settings

    async def update_from_settings_page(self, settings: Dict[str, Any], user_id: int = None) -> bool:
        """
        Update configuration from settings page submission

        Args:
            settings: Dictionary of settings from the page (with futures_ prefix)
            user_id: User making the change

        Returns:
            bool: True if all updates succeeded
        """
        # Map settings to their config types
        config_map = {
            'enabled': FuturesConfigType.GENERAL,
            'exchange': FuturesConfigType.GENERAL,
            'testnet': FuturesConfigType.GENERAL,
            'trading_mode': FuturesConfigType.GENERAL,  # alias for testnet
            'contract_type': FuturesConfigType.GENERAL,
            # Position settings
            'capital': FuturesConfigType.POSITION,  # alias for capital_allocation
            'capital_allocation': FuturesConfigType.POSITION,
            'position_size_usd': FuturesConfigType.POSITION,
            'max_position_pct': FuturesConfigType.POSITION,
            'max_positions': FuturesConfigType.POSITION,
            'min_trade_size': FuturesConfigType.POSITION,
            # Dynamic position sizing
            'dynamic_position_sizing': FuturesConfigType.POSITION,
            'min_position_pct': FuturesConfigType.POSITION,
            'max_position_usd': FuturesConfigType.POSITION,
            'static_position_pct': FuturesConfigType.POSITION,
            # FUT-RM-06: ATR-based sizing
            'atr_sizing_enabled': FuturesConfigType.POSITION,
            'atr_risk_pct': FuturesConfigType.POSITION,
            'atr_stop_multiplier': FuturesConfigType.POSITION,
            # Leverage settings
            'leverage': FuturesConfigType.LEVERAGE,  # alias for default_leverage
            'default_leverage': FuturesConfigType.LEVERAGE,
            'max_leverage': FuturesConfigType.LEVERAGE,
            'margin_mode': FuturesConfigType.LEVERAGE,
            'enforce_isolated_margin': FuturesConfigType.LEVERAGE,  # FUT-RM-07
            'telegram_emergency_close_enabled': FuturesConfigType.LEVERAGE,  # FUT-RM-07b
            'max_leverage_overrides': FuturesConfigType.LEVERAGE,   # FUT-RM-08
            # Risk settings - SL
            'stop_loss': FuturesConfigType.RISK,  # alias
            'stop_loss_pct': FuturesConfigType.RISK,
            # Risk settings - Legacy single TP
            'take_profit': FuturesConfigType.RISK,  # alias
            'take_profit_pct': FuturesConfigType.RISK,
            # Risk settings - Multiple TPs
            'tp1_pct': FuturesConfigType.RISK,
            'tp2_pct': FuturesConfigType.RISK,
            'tp3_pct': FuturesConfigType.RISK,
            'tp4_pct': FuturesConfigType.RISK,
            'tp1_size_pct': FuturesConfigType.RISK,
            'tp2_size_pct': FuturesConfigType.RISK,
            'tp3_size_pct': FuturesConfigType.RISK,
            'tp4_size_pct': FuturesConfigType.RISK,
            # Risk settings - Daily loss
            'daily_loss_limit': FuturesConfigType.RISK,  # alias
            'max_daily_loss_usd': FuturesConfigType.RISK,
            'max_daily_loss_pct': FuturesConfigType.RISK,
            'max_consecutive_losses': FuturesConfigType.RISK,
            'liquidation_buffer': FuturesConfigType.RISK,
            # FUT-RM-10 auto-deleverage
            'auto_deleverage_enabled': FuturesConfigType.RISK,
            'auto_deleverage_cooldown_seconds': FuturesConfigType.RISK,
            # Risk settings - Trailing stop
            'trailing_stop': FuturesConfigType.RISK,  # alias
            'trailing_stop_enabled': FuturesConfigType.RISK,
            'trailing_distance': FuturesConfigType.RISK,  # alias
            'trailing_stop_distance': FuturesConfigType.RISK,
            # Pairs settings
            'allowed_pairs': FuturesConfigType.PAIRS,
            'both_directions': FuturesConfigType.PAIRS,
            'preferred_direction': FuturesConfigType.PAIRS,
            # Strategy settings
            'signal_timeframe': FuturesConfigType.STRATEGY,
            'timeframe': FuturesConfigType.STRATEGY,  # alias
            'scan_interval_seconds': FuturesConfigType.STRATEGY,
            'scan_interval': FuturesConfigType.STRATEGY,  # alias
            'rsi_oversold': FuturesConfigType.STRATEGY,
            'rsi_overbought': FuturesConfigType.STRATEGY,
            'rsi_weak_oversold': FuturesConfigType.STRATEGY,
            'rsi_weak_overbought': FuturesConfigType.STRATEGY,
            'min_signal_score': FuturesConfigType.STRATEGY,
            'verbose_signals': FuturesConfigType.STRATEGY,
            'cooldown_minutes': FuturesConfigType.STRATEGY,
            'require_trend_alignment': FuturesConfigType.STRATEGY,
            'require_volume_confirmation': FuturesConfigType.STRATEGY,
            'min_signal_confluence_count': FuturesConfigType.STRATEGY,
            # FUT-RM-26 (Wave 24): hard entry-suppression / neutralization switch
            'entries_suppressed': FuturesConfigType.STRATEGY,
            # FUT-RM-19/20 (Wave 7): edge gate + per-candle throttle
            'min_edge_gate_enabled': FuturesConfigType.STRATEGY,
            'min_net_edge_pct': FuturesConfigType.STRATEGY,
            'edge_slippage_pct': FuturesConfigType.STRATEGY,
            'edge_funding_fallback_pct': FuturesConfigType.STRATEGY,
            'one_entry_per_candle': FuturesConfigType.STRATEGY,
            # Risk - new market filters
            'require_trend_confirmation': FuturesConfigType.RISK,
            'min_volume_multiplier': FuturesConfigType.RISK,
            # FUT-RM-21 (Wave 7): regime / counter-trend gate
            'block_counter_trend_entries': FuturesConfigType.RISK,
            # FUT-RM-27 (Wave 25): per-symbol tiering + rolling gate
            'symbol_tiering_enabled': FuturesConfigType.RISK,
            'symbol_size_weights': FuturesConfigType.RISK,
            'rolling_gate_enabled': FuturesConfigType.RISK,
            'rolling_gate_window': FuturesConfigType.RISK,
            'rolling_gate_min_trades': FuturesConfigType.RISK,
            'rolling_gate_max_net_pnl_usd': FuturesConfigType.RISK,
            'rolling_gate_max_win_rate': FuturesConfigType.RISK,
            'rolling_gate_bench_minutes': FuturesConfigType.RISK,
            'rolling_gate_probation_weight': FuturesConfigType.RISK,
            # Funding settings
            'funding_arb': FuturesConfigType.FUNDING,  # alias
            'funding_arbitrage_enabled': FuturesConfigType.FUNDING,
            'max_funding_rate': FuturesConfigType.FUNDING,
            # FUT-RM-05 directional funding gate
            'skip_long_funding_bps': FuturesConfigType.FUNDING,
            'skip_short_funding_bps': FuturesConfigType.FUNDING,
            'max_funding_age_seconds': FuturesConfigType.FUNDING,
            # FUT-RM-25 (v1 carry) settings
            'funding_carry_enabled': FuturesConfigType.FUNDING,
            'carry_min_funding_bps': FuturesConfigType.FUNDING,
            'carry_exit_funding_bps': FuturesConfigType.FUNDING,
            'carry_max_positions': FuturesConfigType.FUNDING,
            'carry_max_hold_minutes': FuturesConfigType.FUNDING,
            # FUT-QC-01 (carry v2) settings. NOTE: the master-flag FIELD name
            # deliberately matches the DB key `futures_funding_carry_enabled`;
            # the settings page round-trips it as
            # `futures_futures_funding_carry_enabled` and the single-leading-
            # prefix strip below resolves it back. Do NOT post the bare key
            # `futures_funding_carry_enabled` from a form — after the strip it
            # collides with the v1 `funding_carry_enabled` field.
            'futures_funding_carry_enabled': FuturesConfigType.FUNDING,
            'carry_min_abs_funding_bps': FuturesConfigType.FUNDING,
            'carry_funding_stability_window': FuturesConfigType.FUNDING,
            'carry_max_carry_positions': FuturesConfigType.FUNDING,
        }

        # Group settings by config type
        grouped: Dict[FuturesConfigType, Dict[str, Any]] = {}

        for key, value in settings.items():
            # Remove ONE leading futures_ prefix if present. Must be a single
            # leading strip (not str.replace, which removes every occurrence):
            # FUT-QC-01's `futures_funding_carry_enabled` field round-trips
            # from get_all_settings as `futures_futures_funding_carry_enabled`
            # and a global replace would collapse it onto the unrelated v1
            # `funding_carry_enabled` field.
            clean_key = key[len('futures_'):] if key.startswith('futures_') else key

            config_type = config_map.get(clean_key)
            if config_type:
                if config_type not in grouped:
                    # Start with current config values
                    grouped[config_type] = self.configs[config_type].dict()

                # Handle aliases
                actual_key = self._resolve_alias(clean_key)
                grouped[config_type][actual_key] = value

        # Save each config type
        success = True
        for config_type, config_data in grouped.items():
            if not await self.save_config(config_type, config_data, user_id, "Settings page update"):
                success = False

        return success

    def _resolve_alias(self, key: str) -> str:
        """Resolve setting aliases to actual field names"""
        aliases = {
            'trading_mode': 'testnet',  # paper = testnet=true, live = testnet=false
            'capital': 'capital_allocation',
            'leverage': 'default_leverage',
            'stop_loss': 'stop_loss_pct',
            'take_profit': 'take_profit_pct',
            'daily_loss_limit': 'max_daily_loss_usd',
            'trailing_stop': 'trailing_stop_enabled',
            'trailing_distance': 'trailing_stop_distance',
            'funding_arb': 'funding_arbitrage_enabled',
            'timeframe': 'signal_timeframe',
            'scan_interval': 'scan_interval_seconds',
        }
        return aliases.get(key, key)

    def get_api_credentials(self, exchange: str = None, testnet: bool = None) -> Dict[str, str]:
        """
        Get API credentials for the specified exchange

        Args:
            exchange: Exchange name (binance or bybit), defaults to config
            testnet: Whether to use testnet, defaults to config

        Returns:
            Dict with 'api_key' and 'api_secret'
        """
        if exchange is None:
            exchange = self.get_general().exchange
        if testnet is None:
            testnet = self.get_general().testnet

        exchange = exchange.upper()

        if testnet:
            api_key = self._env_config.get(f'{exchange}_TESTNET_API_KEY', '')
            api_secret = self._env_config.get(f'{exchange}_TESTNET_API_SECRET', '')
        else:
            api_key = self._env_config.get(f'{exchange}_API_KEY', '')
            api_secret = self._env_config.get(f'{exchange}_API_SECRET', '')

        return {
            'api_key': api_key,
            'api_secret': api_secret
        }
