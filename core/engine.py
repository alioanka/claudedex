"""
Core Trading Engine - Orchestrates all bot operations
"""

import asyncio
import uuid
import logging  # ADD THIS LINE
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import time
from dataclasses import dataclass, field
import json
from enum import Enum
import numpy as np
from trading.chains.solana.jupiter_executor import JupiterExecutor
from trading.chains.solana.solana_client import SolanaClient

from core.risk_manager import RiskManager, RiskScore
from core.pattern_analyzer import PatternAnalyzer
from core.decision_maker import DecisionMaker
from core.portfolio_manager import PortfolioManager
from core.event_bus import EventBus, Event, EventType

from data.collectors.dexscreener import DexScreenerCollector
from data.collectors.chain_data import ChainDataCollector
from data.collectors.social_data import SocialDataCollector
from data.collectors.mempool_monitor import MempoolMonitor
from data.collectors.whale_tracker import WhaleTracker
from data.collectors.honeypot_checker import HoneypotChecker

from ml.models.ensemble_model import EnsemblePredictor
from ml.feature_builder import build_ml_feature_dict
from ml.optimization.hyperparameter import HyperparameterOptimizer
from ml.optimization.reinforcement import RLOptimizer

from trading.executors.base_executor import TradeExecutor
from trading.strategies import StrategyManager
from trading.orders.order_manager import OrderManager
from trading.orders.position_tracker import PositionTracker

# ✅ PATCH 8: Import order helper functions and enums
from trading.orders.order_manager import (
    build_order,
    OrderSide,
    OrderType,
    OrderStatus,
    ExecutionStrategy
)
from decimal import Decimal

from monitoring.alerts import AlertManager
from monitoring.performance import PerformanceTracker
from monitoring.logger import StructuredLogger  # Add import at top
from monitoring.logger import log_trade_entry, log_trade_exit

from security.wallet_security import WalletSecurityManager

class BotState(Enum):
    """Bot operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

logger = logging.getLogger(__name__)


def _clamp_retries(config: Dict) -> int:
    """trading.live_max_execute_retries (migration 105), clamped to 1..5.
    Fail-soft to the legacy executor default of 3."""
    try:
        v = int((config.get('trading', {}) or {}).get('live_max_execute_retries', 3) or 3)
    except (TypeError, ValueError):
        return 3
    return min(max(v, 1), 5)


@dataclass
class TradingOpportunity:
    """Represents a potential trading opportunity"""
    token_address: str
    pair_address: str
    chain: str
    price: float
    liquidity: float
    volume_24h: float
    risk_score: RiskScore
    ml_confidence: float
    pump_probability: float
    rug_probability: float
    expected_return: float
    recommended_position_size: float
    entry_strategy: str
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Calculate overall opportunity score"""
        return (
            self.ml_confidence * 0.3 +
            self.pump_probability * 0.25 +
            (1 - self.rug_probability) * 0.25 +
            min(self.expected_return / 100, 1) * 0.2
        )

    @property
    def volatility(self) -> float:
        """
        Calculate volatility for strategy selection

        Uses price change metrics from metadata if available,
        otherwise estimates from volume/liquidity ratio
        """
        # Try to get from metadata first
        if 'volatility' in self.metadata:
            return self.metadata['volatility']

        # Estimate from volume/liquidity ratio
        # High volume relative to liquidity indicates high volatility
        if self.liquidity > 0:
            vol_estimate = (self.volume_24h / self.liquidity) * 0.1
            return min(vol_estimate, 1.0)  # Cap at 100%

        return 0.0

    @property
    def spread(self) -> float:
        """
        Calculate bid-ask spread for strategy selection

        Uses spread from metadata if available,
        otherwise estimates from liquidity (high liquidity = low spread)
        """
        # Try to get from metadata first
        if 'spread' in self.metadata:
            return self.metadata['spread']

        # Estimate from liquidity (inverse relationship)
        # High liquidity = tight spread, low liquidity = wide spread
        if self.liquidity > 100000:  # >$100k liquidity
            return 0.003  # 0.3% spread
        elif self.liquidity > 50000:  # >$50k liquidity
            return 0.007  # 0.7% spread
        elif self.liquidity > 10000:  # >$10k liquidity
            return 0.015  # 1.5% spread
        else:
            return 0.030  # 3.0% spread (high)

        return 0.01  # Default 1%

def _as_utc(dt):
    """Normalize a datetime to UTC-aware. Naive values are assumed UTC.

    Wave-11: positions restored by _load_state come from TIMESTAMPTZ (aware),
    while in-process datetime.now() values are naive. Mixing them in
    subtraction raises TypeError, so every (now - entry_time) arithmetic on
    DB-sourced datetimes must route both sides through this helper.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _jsonable_metadata(value):
    """Recursively coerce numpy scalars to plain Python for DB jsonb writes.

    Wave-F5 RC-D3: position metadata carried numpy.float64 (ML scores),
    which broke orjson in database.update_trade and left closed trades
    stuck OPEN. The DB boundary also converts (belt+braces).
    """
    if isinstance(value, dict):
        return {k: _jsonable_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metadata(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ClosedPositionRecord:
    """Track recently closed positions for cooldown"""
    token_address: str
    closed_at: datetime
    reason: str
    pnl: float

    def is_cooled_down(self, cooldown_minutes: int = 60) -> bool:
        """Check if cooldown period has elapsed"""
        elapsed = (datetime.now(timezone.utc) - _as_utc(self.closed_at)).total_seconds() / 60
        return elapsed >= cooldown_minutes

class TradingBotEngine:
    """Main orchestration engine for the trading bot"""
    
    def __init__(self, config: Dict, config_manager, chain_rpc_urls: Dict, mode: str = "production"):
        """
        Initialize the trading engine
        
        Args:
            config: Configuration dictionary
            config_manager: The main ConfigManager instance
            chain_rpc_urls: Dictionary of chain-specific RPC URLs
            mode: Operating mode
        """
        self.config = config
        self.config_manager = config_manager
        self.chain_rpc_urls = chain_rpc_urls
        self.mode = mode
        self.state = BotState.INITIALIZING
        
        # Core components
        self.event_bus = EventBus()
        self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
        self.risk_manager = RiskManager(
            config['risk_management'], 
            portfolio_manager=self.portfolio_manager,
            config_manager=self.config_manager,
            chain_rpc_urls=self.chain_rpc_urls
        )
        self.pattern_analyzer = PatternAnalyzer()
        self.decision_maker = DecisionMaker(config)

        # RiskManager.validate_trade calls wallet_manager.get_available_balance(),
        # which WalletSecurityManager does not implement — every validation
        # would fail-closed with an AttributeError in LIVE mode. Back the call
        # with the portfolio manager (USD available balance) until the
        # risk-manager owner ships a real implementation. 0.0 on error keeps
        # the check fail-closed.
        if not hasattr(self.risk_manager.wallet_manager, 'get_available_balance'):
            _pm = self.portfolio_manager

            async def _available_balance_usd() -> float:
                try:
                    return float(_pm.get_available_balance())
                except Exception:
                    return 0.0

            self.risk_manager.wallet_manager.get_available_balance = _available_balance_usd

        # Data collectors
        self.dex_collector = DexScreenerCollector(
            config.get('data_sources', {}).get('dexscreener', {})
        )
        self.chain_collector = ChainDataCollector(config['web3'])
        self.social_collector = SocialDataCollector(config['data_sources']['social'])
        self.mempool_monitor = MempoolMonitor(config['web3'])
        self.whale_tracker = WhaleTracker(config['web3'])

        # --- FIX: Initialize honeypot checker with config manager and RPC URLs ---
        self.honeypot_checker = HoneypotChecker(self.config_manager, self.chain_rpc_urls)
        
        # ML components
        self.ensemble_predictor = EnsemblePredictor()
        self.hyperparam_optimizer = HyperparameterOptimizer()
        self.rl_optimizer = RLOptimizer()
        
        # Trading components

        executor_config = {
            # Resolved module dry-run flag (set by main_dex from
            # resolve_module_dry_run). Without this key the executor fell
            # back to its safe default True and could never broadcast live.
            'DRY_RUN': config.get('dry_run', True),
            # Enables the executor's wallet-mismatch safety check.
            'wallet_address': config.get('wallet_address'),
            'web3_provider_url': config.get('web3', {}).get('provider_url'),
            'private_key': config.get('security', {}).get('private_key'),
            'chain_id': config.get('web3', {}).get('chain_id', 1),
            'max_gas_price': config.get('web3', {}).get('max_gas_price', 50),  # FIXED: Was 500
            'gas_limit': config.get('web3', {}).get('gas_limit', 500000),
            # trading.live_max_execute_retries (migration 105) — clamped 1..5.
            # WARNING: a receipt-timeout retry can RE-BROADCAST the swap;
            # 1 is the safe LIVE setting (default 3 = legacy behavior).
            'max_retries': _clamp_retries(config),
            'retry_delay': 1,
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Mainnet router
            '1inch_api_key': config.get('api', {}).get('1inch_api_key'),
            'paraswap_api_key': config.get('api', {}).get('paraswap_api_key'),
        }

        # Database connection for logging
        from data.storage.database import DatabaseManager
        self.db = DatabaseManager(config.get('database', {}))  

        # FIXED: Strategies moved to top-level config, use conversion method for proper nesting
        from config.config_manager import ConfigType
        strategies_config = self.config_manager.get_config(ConfigType.STRATEGIES)
        self.strategy_manager = StrategyManager(
            strategies_config.to_strategy_manager_dict(),
            db_pool=self.db,
        )
        self.order_manager = OrderManager(config, db_manager=self.db)  # 🆕 ADD db_manager
        self.position_tracker = PositionTracker()
        self.trade_executor = TradeExecutor(executor_config, db_manager=self.db)  # 🆕 ADD db_manager
        

        # ✅ PATCH 1: Connect OrderManager to actual execution engine
        logger.info("🔗 Connecting OrderManager to TradeExecutor...")
        self.order_manager.execution_engine = self.trade_executor

        # ✅ PATCH 1B: Inject position tracker into risk monitor
        logger.info("🔗 Connecting OrderManager to PositionTracker...")
        self.order_manager.risk_monitor.position_tracker = self.position_tracker
        self.order_manager.risk_monitor.portfolio_manager = self.portfolio_manager

        logger.info("✅ OrderManager integrations complete")

        # Find this section in __init__:
        # self.trade_executor = DirectDEXExecutor(config)
        # OR
        # self.trade_executor = ToxiSolAPIExecutor(config)

        # Add AFTER the existing executor initialization:

        # Initialize Solana executor if enabled
        self.solana_executor = None

        # Correctly check if Solana is enabled from the chain configuration
        chain_config = config.get('chain', {})
        solana_enabled = chain_config.get('solana_enabled', False)

        if solana_enabled:
            try:
                solana_config = config.get('solana', {})
                security_config = config.get('security', {})

                # Assemble the config for JupiterExecutor by combining solana and security configs
                executor_solana_config = {
                    **solana_config,
                    'solana_private_key': security_config.get('solana_private_key'),
                    'encryption_key': security_config.get('encryption_key'),
                    'dry_run': config.get('trading', {}).get('dry_run', True)
                }

                self.solana_executor = JupiterExecutor(executor_solana_config)
                logger.info("✅ Solana Jupiter Executor initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Solana executor: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("ℹ️ Solana trading disabled")
        
        # Monitoring
        # Wave-19: pass module='dex' so AlertManager routes trade/error
        # notifications through TelegramNotificationEngine with the DEX caption.
        self.alert_manager = AlertManager(config['notifications'], module='dex')
        self.performance_tracker = PerformanceTracker()

        self.structured_logger = StructuredLogger("TradingBot", config.get('logging', {}))
        
        # Security
        self.wallet_manager = WalletSecurityManager(config['security'])
        
        # Internal state
        self.active_positions: Dict[str, Any] = {}
        self.pending_opportunities: List[TradingOpportunity] = []
        self.blacklisted_tokens: set = set()
        self.blacklisted_devs: set = set()

        # Position tracking
        self.open_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        self.positions_lock = asyncio.Lock()
        self.active_trades: Dict[str, Dict] = {}

        # Cooldown tracking
        self.recently_closed: Dict[str, ClosedPositionRecord] = {}  # token_address -> record
        self.cooldown_minutes = config.get('risk_management', {}).get('position_cooldown_minutes', 60)

        # Wave-26: per-UTC-day ENTRY budget (trading.max_trades_per_day,
        # seeded by migration 095 — previously had NO consumer). Counter is
        # lazily re-seeded from the trades table on the first entry attempt
        # of each UTC day so it survives subprocess restarts. ENTRY-only:
        # exits never pass through _execute_opportunity.
        self._entry_budget_day: Optional[str] = None      # 'YYYY-MM-DD' (UTC)
        self._entries_today: int = 0
        self._entry_budget_logged_day: Optional[str] = None  # log-once guard
        
      
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0,
            'tokens_analyzed': 0,
            'opportunities_found': 0,
            'start_time': datetime.now()
        }
        
        # Tasks
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize all components"""
        try:
            self.state = BotState.INITIALIZING
            
            # Load blacklists
            await self._load_blacklists()
            
            # Initialize components in order (only if they have initialize methods)
            if hasattr(self.wallet_manager, 'initialize'):
                await self.wallet_manager.initialize()

            # Initialize database
            if hasattr(self.db, 'connect'):
                await self.db.connect()
                logger.info("✅ Database connected")

            # Set portfolio manager dependencies (AFTER db is connected)
            self.portfolio_manager.set_dependencies(self.db, self.alert_manager)
            await self.portfolio_manager.load_block_state()
            logger.info("✅ Portfolio manager block state loaded")
            
            if hasattr(self.risk_manager, 'initialize'):
                await self.risk_manager.initialize()

            # ✅ ADD THIS - Initialize honeypot checker
            if hasattr(self.honeypot_checker, 'initialize'):
                await self.honeypot_checker.initialize()
            
            # ADD THIS - Initialize data collectors!
            if hasattr(self.dex_collector, 'initialize'):
                await self.dex_collector.initialize()
                
            if hasattr(self.chain_collector, 'initialize'):
                await self.chain_collector.initialize()
                
            if hasattr(self.social_collector, 'initialize'):
                await self.social_collector.initialize()
            
            if hasattr(self.ensemble_predictor, 'load_models'):
                await self.ensemble_predictor.load_models()
            
            if hasattr(self.strategy_manager, 'initialize'):
                await self.strategy_manager.initialize()
            
            if hasattr(self.order_manager, 'initialize'):
                await self.order_manager.initialize()

            # Initialize Solana executor
            if self.solana_executor:
                try:
                    await self.solana_executor.initialize()
                    logger.info("Solana trading enabled via Jupiter")
                except Exception as e:
                    logger.error(f"Solana executor initialization failed: {e}")
                    self.solana_executor = None
                        
            # Rest of initialization...

            # Start event bus processing
            await self.event_bus.start()
            logger.info("✅ Event bus started")

            # Setup event handlers
            self._setup_event_handlers()
            
            # Load saved state if exists
            await self._load_state()
            
            # Warm up data collectors
            await self._warmup_collectors()
            
            self.state = BotState.RUNNING
            
        except Exception as e:
            self.state = BotState.ERROR
            raise Exception(f"Failed to initialize engine: {e}")
            
    def _setup_event_handlers(self):
        """Setup event bus handlers"""
        # Use subscribe_sync() for synchronous subscription with EventType enum
        self.event_bus.subscribe_sync(EventType.NEW_PAIR_DETECTED, self._handle_new_pair)
        self.event_bus.subscribe_sync(EventType.WHALE_MOVEMENT, self._handle_whale_movement)
        self.event_bus.subscribe_sync(EventType.UNUSUAL_VOLUME, self._handle_unusual_volume)
        self.event_bus.subscribe_sync(EventType.RUG_PULL_DETECTED, self._handle_rug_pull)
        self.event_bus.subscribe_sync(EventType.POSITION_OPENED, self._handle_position_opened)
        self.event_bus.subscribe_sync(EventType.POSITION_CLOSED, self._handle_position_closed)
        
    async def run(self):
        """Main engine loop"""
        try:
            # Create concurrent tasks with names for debugging hangs
            self.tasks = [
                asyncio.create_task(self._monitor_new_pairs(), name="monitor_new_pairs"),
                asyncio.create_task(self._monitor_existing_positions(), name="monitor_positions"),
                asyncio.create_task(self._process_opportunities(), name="process_opportunities"),
                asyncio.create_task(self._monitor_mempool(), name="monitor_mempool"),
                asyncio.create_task(self._track_whales(), name="track_whales"),
                asyncio.create_task(self._optimize_strategies(), name="optimize_strategies"),
                asyncio.create_task(self._retrain_models(), name="retrain_models"),
                asyncio.create_task(self._update_blacklists(), name="update_blacklists"),
                asyncio.create_task(self._monitor_performance(), name="monitor_performance"),
                asyncio.create_task(self._health_check(), name="health_check"),
                asyncio.create_task(self._monitor_wallet_balances(), name="monitor_wallets"),
                asyncio.create_task(self._monitor_positions_with_engine(), name="monitor_positions_engine"),
                asyncio.create_task(self._watchdog(), name="watchdog"),


            ]
            
            # Wait for tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            await self.alert_manager.send_critical(f"Engine error: {e}")
            raise
            
    # ============================================================================
    # FIX #5: core/engine.py - Fix chain configuration
    # Around line 248 in _monitor_new_pairs method
    # ============================================================================

    async def _monitor_new_pairs(self):
        """Continuously monitor for new trading pairs"""
        logger.info("🔍 Starting new pairs monitoring loop...")
        
        # ✅ FIX: Simplified chain configuration reading
        # Try multiple config paths
        enabled_chains = None
        
        # Method 1: Direct from config
        if 'enabled_chains' in self.config:
            chains_str = self.config['enabled_chains']
            if isinstance(chains_str, str):
                enabled_chains = [c.strip() for c in chains_str.split(',') if c.strip()]
            elif isinstance(chains_str, list):
                enabled_chains = chains_str
        
        # Method 2: From chains.enabled
        if not enabled_chains and 'chains' in self.config:
            enabled_chains = self.config['chains'].get('enabled')
        
        # Method 3: From data_sources.dexscreener.chains
        if not enabled_chains and 'data_sources' in self.config:
            enabled_chains = self.config.get('data_sources', {}).get('dexscreener', {}).get('chains')
        
        # Default fallback
        if not enabled_chains:
            enabled_chains = ['ethereum', 'bsc', 'base', 'arbitrum', 'polygon']
            logger.warning(f"⚠️ Using default chains: {enabled_chains}")
        
        # Get other settings
        chain_config = self.config.get('chain', {})
        max_pairs_per_chain = chain_config.get('max_pairs_per_chain', 50)
        discovery_interval = chain_config.get('discovery_interval_seconds', 300)
        
        logger.info(f"🌐 Multi-chain mode: {len(enabled_chains)} chains enabled")
        logger.info(f"  Chains: {', '.join(enabled_chains)}")
        logger.info(f"  Max pairs per chain: {max_pairs_per_chain}")
        logger.info(f"  Discovery interval: {discovery_interval}s")
        
        discovery_count = 0

        while self.state == BotState.RUNNING:
            try:
                discovery_count += 1
                all_opportunities = []
                chain_stats = {}
                
                logger.info(f"🌐 Discovery cycle #{discovery_count} across {len(enabled_chains)} chains...")
                cycle_start = asyncio.get_event_loop().time()
                
                # ✅ CRITICAL: Loop through each enabled chain
                for chain in enabled_chains:
                    try:
                        chain_start = asyncio.get_event_loop().time()
                        
                        # Get chain-specific settings
                        chain_config = self.config.get('chain', {})
                        min_liquidity = chain_config.get(f'{chain}_min_liquidity', 10000)
                        
                        logger.info(f"  🔗 Scanning {chain.upper()}... (min liquidity: ${min_liquidity:,.0f})")
                        
                        # ✅ CRITICAL: Pass chain parameter to get_new_pairs
                        pairs = await self.dex_collector.get_new_pairs(
                            chain=chain, 
                            limit=max_pairs_per_chain
                        )
                        
                        if pairs:
                            logger.info(f"    ✅ Found {len(pairs)} pairs on {chain.upper()}")

                            # ADD THIS DEBUG CODE:
                            if len(pairs) > 0:
                                sample_pair = pairs[0]
                                logger.info(f"    🔍 DEBUG: Sample pair keys: {list(sample_pair.keys())}")
                                logger.info(f"    🔍 DEBUG: Sample pair data:")
                                logger.info(f"       token_symbol: {sample_pair.get('token_symbol')}")
                                logger.info(f"       liquidity_usd: {sample_pair.get('liquidity_usd')}")
                                logger.info(f"       liquidity: {sample_pair.get('liquidity')}")
                                logger.info(f"       volume_24h: {sample_pair.get('volume_24h')}")
                                logger.info(f"       price: {sample_pair.get('price')}")
                                logger.info(f"       price_usd: {sample_pair.get('price_usd')}")
                            
                            # ============================================================================
                            # FIX #4: core/engine.py - Fix liquidity key mismatch
                            # Around line 300 in _monitor_new_pairs method
                            # ============================================================================

                            # FIND THIS CODE (around line 300):
                            for pair in pairs:
                                try:
                                    self.stats['tokens_analyzed'] += 1
                                    
                                    # Normalize token address
                                    token_address = pair.get('token_address', '').lower()
                                    
                                    # Check blacklist
                                    if token_address in self.blacklisted_tokens:
                                        logger.debug(f"⛔ Token {pair.get('token_symbol')} is blacklisted - SKIPPING")
                                        continue
                                    
                                    # Quick filter checks
                                    if self._is_blacklisted(pair):
                                        logger.debug(f"⛔ Pair {pair.get('pair_address', 'unknown')} is blacklisted")
                                        continue
                                    
                                    # REPLACE THIS SECTION:
                                    # OLD (BROKEN):
                                    # if pair.get('liquidity_usd', 0) < min_liquidity:
                                    #     continue
                                    
                                    # NEW (FIXED) - Check both possible key names:
                                    liquidity = pair.get('liquidity_usd') or pair.get('liquidity') or 0
                                    
                                    # ADD DEBUG LOG
                                    logger.info(f"  🔍 Checking {pair.get('token_symbol', 'UNKNOWN')}: liq=${liquidity:,.0f}, min=${min_liquidity:,.0f}")
                                    
                                    if liquidity < min_liquidity:
                                        logger.debug(f"    ❌ Rejected: Liquidity ${liquidity:,.0f} < ${min_liquidity:,.0f}")
                                        continue
                                    
                                    logger.info(f"  ✅ Passed liquidity filter: {pair.get('token_symbol', 'UNKNOWN')}")
                                    
                                    # Analyze opportunity
                                    logger.debug(f"Analyzing pair: {pair.get('token_symbol', 'UNKNOWN')} on {chain}")
                                    opportunity = await self._analyze_opportunity(pair)
                                    
                                    if opportunity:
                                        min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.25)
                                        logger.debug(f"  Score: {opportunity.score:.3f} (min: {min_score})")
                                        
                                        if opportunity.score > min_score:
                                            opportunity.chain = chain
                                            all_opportunities.append(opportunity)
                                            self.stats['opportunities_found'] += 1
                                            logger.info(f"🎯 OPPORTUNITY: {pair.get('token_symbol')} on {chain.upper()} - Score: {opportunity.score:.3f}")
                                            
                                            # Emit event
                                            await self.event_bus.emit(Event(
                                                event_type=EventType.OPPORTUNITY_FOUND,
                                                data=opportunity
                                            ))
                                        else:
                                            logger.info(f"  ❌ Score too low: {opportunity.score:.3f} < {min_score}")
                                            
                                except Exception as e:
                                    logger.debug(f"Error analyzing pair on {chain}: {e}")
                                    continue
                        else:
                            logger.warning(f"    ⚠️  No pairs found on {chain.upper()}")
                        
                        # Track stats
                        chain_elapsed = asyncio.get_event_loop().time() - chain_start
                        chain_stats[chain] = {
                            'pairs_found': len(pairs) if pairs else 0,
                            'opportunities': len([o for o in all_opportunities if o.chain == chain]),
                            'scan_time': chain_elapsed
                        }
                        
                        # Brief delay between chains to avoid rate limits
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"❌ Error scanning {chain}: {e}")
                        chain_stats[chain] = {'pairs_found': 0, 'opportunities': 0, 'error': str(e)}
                        continue
                
                # Log summary
                cycle_elapsed = asyncio.get_event_loop().time() - cycle_start
                total_pairs = sum(s.get('pairs_found', 0) for s in chain_stats.values())
                total_opps = len(all_opportunities)
                
                logger.info(f"📊 Discovery #{discovery_count} Complete ({cycle_elapsed:.1f}s):")
                logger.info(f"   • Total pairs scanned: {total_pairs}")
                logger.info(f"   • Opportunities found: {total_opps}")
                for chain, stats in chain_stats.items():
                    logger.info(f"   • {chain.upper()}: {stats.get('pairs_found', 0)} pairs → "
                            f"{stats.get('opportunities', 0)} opportunities "
                            f"({stats.get('scan_time', 0):.1f}s)")
                
                # Add opportunities to pending queue
                if all_opportunities:
                    # Sort by score
                    all_opportunities.sort(key=lambda x: x.score, reverse=True)
                    
                    # Add to pending
                    self.pending_opportunities.extend(all_opportunities[:10])  # Top 10
                    
                    logger.info(f"📋 Added {min(len(all_opportunities), 10)} opportunities to pending queue")
                else:
                    logger.info("⏸️  No qualifying opportunities in this cycle")
                
                # Wait before next discovery cycle
                logger.info(f"⏳ Waiting {discovery_interval}s until next discovery cycle...")
                await asyncio.sleep(discovery_interval)
                
            except asyncio.CancelledError:
                logger.info("Discovery loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in multi-chain discovery: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error
                
    # ============================================================================
    # FIX #2: core/engine.py
    # Add more detailed logging to see WHY pairs are being filtered out
    # ============================================================================

    # Find the _analyze_opportunity method in core/engine.py (around line 402)
    # Replace the scoring section with this enhanced version:



    async def _monitor_wallet_balances(self):
        """Monitor wallet balances and alert if low"""
        while self.state == BotState.RUNNING:
            try:
                for chain in ['ethereum', 'bsc', 'base', 'arbitrum', 'polygon']:
                    if not self.config.get(f'{chain}_enabled', False):
                        continue
                    
                    # Check balance
                    balance = await self._get_chain_balance(chain)
                    min_balance = 0.05  # 0.05 ETH/BNB minimum
                    
                    if balance < min_balance:
                        await self.alert_manager.send_warning(
                            f"⚠️ LOW BALANCE WARNING\n"
                            f"Chain: {chain.upper()}\n"
                            f"Balance: {balance:.4f}\n"
                            f"Minimum: {min_balance}\n"
                            f"Please top up wallet!"
                        )
                
                # Check Solana
                if self.config.get('solana_enabled'):
                    sol_balance = await self.solana_executor.get_balance()
                    if sol_balance < 0.1:
                        await self.alert_manager.send_warning(
                            f"⚠️ LOW SOLANA BALANCE\n"
                            f"Balance: {sol_balance:.4f} SOL\n"
                            f"Please top up wallet!"
                        )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Balance monitoring error: {e}")
                await asyncio.sleep(3600)

    async def _get_chain_balance(self, chain: str) -> float:
        """
        Get native token balance for a chain
        
        Args:
            chain: Chain name ('ethereum', 'bsc', 'base', etc.)
            
        Returns:
            Balance in native token (ETH, BNB, etc.)
        """
        try:
            # Get the appropriate executor for the chain
            if chain == 'solana':
                if self.solana_executor:
                    return await self.solana_executor.get_balance()
                return 0.0
            
            # For EVM chains, use the trade_executor's web3 instance
            wallet_address = self.config.get('wallet_address')
            if not wallet_address:
                logger.warning(f"No wallet address configured for {chain}")
                return 0.0
            
            # Get balance from web3
            balance_wei = self.trade_executor.w3.eth.get_balance(wallet_address)
            balance_eth = self.trade_executor.w3.from_wei(balance_wei, 'ether')
            
            return float(balance_eth)
            
        except Exception as e:
            logger.error(f"Error getting {chain} balance: {e}")
            return 0.0

    def _build_ml_feature_dict(self, pair: Dict, risk_score, patterns) -> Dict:
        """Map the data we already gathered in _analyze_opportunity into the
        nested-dict schema EnsemblePredictor.extract_features() consumes.

        Wave-10: the mapping body now lives in the shared, pure
        `ml.feature_builder.build_ml_feature_dict` so the offline trainer
        (`scripts/train_ensemble.py`) builds train-time features with the SAME
        code path — guaranteeing train==inference feature parity by
        construction (no copy that can drift). This wrapper preserves the
        original method signature/behaviour exactly (zero logic change).
        """
        return build_ml_feature_dict(pair, risk_score, patterns)

    @staticmethod
    def _is_trustworthy_ml_result(result: Dict) -> bool:
        """A prediction is trustworthy only if the ensemble produced it without
        error AND it is not the degenerate untrained/neutral output.

        EnsemblePredictor.predict_decoupled() returns a {0.5, 0.5, ...} dict
        with an 'error' key when the (unfitted) RobustScaler.transform raises —
        which is exactly the state in any environment with no trained model
        artifacts on disk. We must NOT treat that as a real signal: doing so
        would re-introduce the fabricated-confidence bug from the other side.
        """
        if not isinstance(result, dict):
            return False
        if result.get('error'):
            return False
        pump = result.get('pump_probability')
        rug = result.get('rug_probability')
        conf = result.get('confidence', 0)
        if pump is None or rug is None:
            return False
        # All-neutral 0.5 with floor confidence == untrained passthrough.
        if abs(pump - 0.5) < 1e-9 and abs(rug - 0.5) < 1e-9 and conf <= 0.1:
            return False
        return True

    async def _ml_predict_opportunity(
        self, pair: Dict, risk_score, patterns, heuristic_score: float
    ) -> Dict:
        """Consult EnsemblePredictor for REAL ml_confidence/pump/rug.

        Fail-soft + HONEST labeling (Wave-8 DEFECT 2): if the ensemble is
        unavailable / untrained / errors, fall back to the heuristic but mark
        ml_source='heuristic_fallback' and DO NOT fabricate an optimistic
        rug_probability. The heuristic fallback derives rug_probability from
        the heuristic score (low score -> higher implied rug risk) instead of
        the old flat 0.2 constant that silently passed the 0.5 rug gate.

        Returns a dict with: ml_confidence, pump_probability, rug_probability,
        expected_return, ml_source, ml_meta.
        """
        token = pair.get('token_address', '')
        chain = pair.get('chain', 'ethereum')
        try:
            predictor = getattr(self, 'ensemble_predictor', None)
            if predictor is not None and hasattr(predictor, 'predict_decoupled'):
                feat = self._build_ml_feature_dict(pair, risk_score, patterns)
                result = await predictor.predict_decoupled(token, chain, feat)
                if self._is_trustworthy_ml_result(result):
                    return {
                        'ml_confidence': float(result.get('confidence', 0.0)),
                        'pump_probability': float(result['pump_probability']),
                        'rug_probability': float(result['rug_probability']),
                        'expected_return': float(result.get('expected_return', 0.0)),
                        'ml_source': 'ensemble',
                        'ml_meta': {
                            'model_agreements': result.get('model_agreements', {}),
                            'risk_adjusted_score': result.get('risk_adjusted_score'),
                        },
                    }
                logger.info(
                    "   ℹ️ ML ensemble unavailable/untrained for "
                    f"{pair.get('token_symbol', token[:10])} "
                    f"(reason={result.get('error', 'neutral-passthrough')}) "
                    "— using HEURISTIC fallback (not a real ML signal)"
                )
        except Exception as e:
            logger.warning(f"   ⚠️ ML ensemble prediction failed: {e} — heuristic fallback")

        # ---- honest heuristic fallback -----------------------------------
        # rug_probability is NOT a flat optimistic constant: a weak heuristic
        # score implies more uncertainty, so map it inversely and clamp so a
        # genuinely strong heuristic still has to clear the 0.5 rug gate on its
        # own merit rather than being handed a free 0.2.
        hs = max(0.0, min(1.0, float(heuristic_score)))
        rug_prob = max(0.25, min(0.6, 0.6 - 0.4 * hs))
        return {
            'ml_confidence': hs,
            'pump_probability': hs * 0.8,
            'rug_probability': rug_prob,
            'expected_return': hs * 100,
            'ml_source': 'heuristic_fallback',
            'ml_meta': {'note': 'no trained ensemble available in this environment'},
        }

    async def _analyze_opportunity(self, pair: Dict) -> Optional[TradingOpportunity]:
        """
        Comprehensive analysis of a trading opportunity
        """
        try:
            # Get token symbol for logging
            token_symbol = pair.get('token_symbol', 'UNKNOWN')
            token_address = pair.get('token_address', '')
            
            # ADD THIS LOG
            logger.info(f"🔬 Analyzing {token_symbol} on {pair.get('chain', 'unknown')}")
            logger.info(f"   Price: ${pair.get('price_usd', 0):.8f}")
            logger.info(f"   Liquidity: ${pair.get('liquidity_usd', 0):,.2f}")
            logger.info(f"   Volume 24h: ${pair.get('volume_24h', 0):,.2f}")
            logger.info(f"   Age: {pair.get('age_hours', 999):.1f}h")
            
            # Parallel analysis tasks
            results = await asyncio.gather(
                self.risk_manager.analyze_token(pair.get('token_address', '')),
                self.pattern_analyzer.analyze_patterns(pair),
                self.chain_collector.get_token_info(pair.get('token_address', '')),
                self._check_developer_reputation(pair.get('creator_address', '')),
                self._analyze_liquidity_depth(pair),
                self._check_smart_contract(pair.get('token_address', '')),
                self._analyze_holder_distribution(pair.get('token_address', '')),
                return_exceptions=True
            )
            
            # Unpack results
            risk_score, patterns, token_info, dev_reputation, \
            liquidity_depth, contract_safety, holder_dist = results
            
            sentiment = None
            
            # Handle exceptions
            if isinstance(risk_score, Exception):
                logger.debug(f"Risk analysis failed: {risk_score}")
                risk_score = None
            if isinstance(patterns, Exception):
                logger.debug(f"Pattern analysis failed: {patterns}")
                patterns = None
            if isinstance(token_info, Exception):
                logger.debug(f"Token info failed: {token_info}")
                token_info = None
            
            # Calculate overall score
            score = self._calculate_opportunity_score(
                pair=pair,
                risk_score=risk_score,
                patterns=patterns,
                sentiment=sentiment,
                liquidity=liquidity_depth,
                contract_safety=contract_safety
            )
            
            # ADD THIS DETAILED LOG
            min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.25)
            logger.info(f"   📊 Score: {score:.4f} (min required: {min_score})")
            
            if score < min_score:
                # ADD THIS LOG TO SEE WHY IT FAILED
                logger.info(f"   ❌ REJECTED: Score {score:.4f} < {min_score}")
                return None
            
            # If we get here, score is good enough!
            logger.info(f"   ✅ PASSED: Score {score:.4f} >= {min_score}")

            # ✅ Calculate position size from config (Kelly Criterion)
            position_size = await self._calculate_position_size(
                risk_score=risk_score,
                opportunity_score=score
            )

            # 🆕 Wave-8 DEFECT 2: consult the REAL ML ensemble for
            # ml_confidence/pump/rug instead of fabricating them from the
            # heuristic. Fail-soft + honestly labeled (ml_source).
            ml = await self._ml_predict_opportunity(
                pair=pair, risk_score=risk_score, patterns=patterns,
                heuristic_score=score,
            )
            logger.info(
                f"   🤖 ML[{ml['ml_source']}] conf={ml['ml_confidence']:.3f} "
                f"pump={ml['pump_probability']:.3f} rug={ml['rug_probability']:.3f}"
            )

            # 🆕 CRITICAL FIX: Create opportunity FIRST (needed for strategy selection)
            # Create a temporary opportunity for strategy selection
            temp_opportunity = TradingOpportunity(
                token_address=pair.get('token_address', ''),
                pair_address=pair.get('pair_address', ''),
                chain=pair.get('chain', 'ethereum'),
                price=pair.get('price_usd', 0),
                liquidity=pair.get('liquidity_usd', 0),
                volume_24h=pair.get('volume_24h', 0),
                # SAFETY FIX (Wave-8 DEFECT 1): worst-case risk on a missing
                # assessment, NOT a neutral 0.5. (The scorer above already
                # rejects when risk_score is falsy, so this branch is normally
                # unreachable — but the old `RiskScore(overall_risk=0.5)` call
                # was itself a latent bug: `overall_risk` is a read-only @property,
                # not a constructor arg, so it would have raised TypeError. We
                # build a real all-1.0 worst-case RiskScore here instead.)
                risk_score=risk_score if risk_score else RiskScore(
                    liquidity_risk=1.0, developer_risk=1.0, contract_risk=1.0,
                    volume_risk=1.0, holder_risk=1.0, social_risk=1.0,
                    technical_risk=1.0, market_risk=1.0, confidence=0.0,
                ),
                ml_confidence=ml['ml_confidence'],
                pump_probability=ml['pump_probability'],
                rug_probability=ml['rug_probability'],
                expected_return=ml['expected_return'],
                recommended_position_size=position_size,
                entry_strategy='momentum',  # Temporary, will be updated
                metadata={
                    'pair': pair,
                    'risk_score': risk_score,
                    'patterns': patterns,
                    'sentiment': sentiment,
                    'liquidity_depth': liquidity_depth,
                    'contract_safety': contract_safety,
                    'holder_distribution': holder_dist,
                    'token_symbol': token_symbol,
                    # Wave-8 DEFECT 2: honest provenance of the ML numbers above
                    # so DRY_RUN audit can tell a real ensemble signal from a
                    # heuristic fallback. NEVER reports a fake high ml_confidence.
                    'ml_source': ml['ml_source'],
                    'ml_meta': ml['ml_meta'],
                    'heuristic_score': score,
                },
                timestamp=datetime.utcnow()
            )

            # 🆕 CRITICAL FIX: Use StrategyManager to select appropriate strategy
            selected_strategy = self.strategy_manager.select_strategy(temp_opportunity)

            # Log strategy selection with details
            logger.info(
                f"   📊 STRATEGY SELECTION for {token_symbol}:\n"
                f"      Pump Probability: {temp_opportunity.pump_probability:.2%}\n"
                f"      Volatility: {temp_opportunity.volatility:.2%}\n"
                f"      Spread: {temp_opportunity.spread:.2%}\n"
                f"      Liquidity: ${temp_opportunity.liquidity:,.0f}\n"
                f"      ➜ Selected: {selected_strategy.upper()}"
            )

            # Update the opportunity with the selected strategy
            opportunity = temp_opportunity
            opportunity.entry_strategy = selected_strategy
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}", exc_info=True)
            return None
            
    async def _process_opportunities(self):
        """Process pending opportunities and execute trades"""
        while self.state == BotState.RUNNING:
            try:
                if not self.pending_opportunities:
                    await asyncio.sleep(0.1)
                    continue
                
                # ADD THIS DEBUG LOG:
                logger.info(f"📋 Processing {len(self.pending_opportunities)} pending opportunities...")
                    
                # Sort by score
                self.pending_opportunities.sort(key=lambda x: x.score, reverse=True)
                
                # Process top opportunities
                for opportunity in self.pending_opportunities[:5]:  # Process top 5
                    # Check if we can take more positions
                    # FIX: Added missing await - can_open_position() is async
                    can_open = await self.portfolio_manager.can_open_position()

                    # ADD THIS DEBUG LOG:
                    logger.info(f"   Can open position? {can_open}")
                    
                    if not can_open:
                        # Rate limit this warning to once per minute
                        if not hasattr(self, '_last_no_positions_warning'):
                            self._last_no_positions_warning = 0

                        now = time.time()
                        if now - self._last_no_positions_warning > 60:  # Once per minute
                            logger.warning("❌ Portfolio manager says NO to new positions")
                            self._last_no_positions_warning = now
                        break
                        
                    # Final checks before execution
                    logger.info(f"   Running final safety checks for {opportunity.token_address[:10]}...")
                    
                    if await self._final_safety_checks(opportunity):
                        logger.info(f"   ✅ Safety checks passed! Executing trade...")
                        await self._execute_opportunity(opportunity)
                    else:
                        logger.warning(f"   ❌ Safety checks failed for {opportunity.token_address[:10]}")
                        
                    if opportunity in self.pending_opportunities:
                        self.pending_opportunities.remove(opportunity)
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing opportunities: {e}", exc_info=True)
                await self.alert_manager.send_error(f"Error processing opportunities: {e}")
                await asyncio.sleep(5)
                
    # ============================================================================
    # Wave-26: entry-side consumers for the migration-095 tunables that
    # previously had NO consumer (trading.chain_weights /
    # trading.max_trades_per_day) + DB-authoritative position SL/TP.
    # ============================================================================

    def _get_chain_weight(self, chain: str) -> float:
        """Consumer of `trading.chain_weights` (seeded by migration 095).

        0.0 disables NEW entries on the chain (measured negative edge —
        week-1: solana avg -2.24/trade, monad -16.64); (0,1) scales position
        size; chains absent from the map default to 1.0. Fail-soft: any
        malformed value yields 1.0 (behavior identical to pre-Wave-26).
        """
        try:
            weights = (self.config.get('trading', {}) or {}).get('chain_weights') or {}
            if isinstance(weights, str):
                weights = json.loads(weights)
            if not isinstance(weights, dict):
                return 1.0
            w = float(weights.get((chain or '').lower(), 1.0))
            if w != w:  # NaN guard
                return 1.0
            return min(max(w, 0.0), 1.0)
        except Exception:
            return 1.0

    def _position_sl_tp(self) -> Tuple[float, float]:
        """DB-configured position stop-loss / take-profit fractions.

        Reads the authoritative `risk_management.stop_loss_pct` /
        `take_profit_pct` rows (seeded 0.12 / 0.24 by migration 095) so the
        position dicts agree with the sizing math in
        _calculate_position_size (which already uses stop_loss_pct).
        Fail-soft to the legacy hardcoded constants (0.1 / 0.3) if unset or
        malformed.
        """
        risk_cfg = self.config.get('risk_management', {}) or {}
        try:
            sl = float(risk_cfg.get('stop_loss_pct') or 0.1)
        except (TypeError, ValueError):
            sl = 0.1
        try:
            tp = float(risk_cfg.get('take_profit_pct') or 0.3)
        except (TypeError, ValueError):
            tp = 0.3
        if not (0.0 < sl < 1.0):
            sl = 0.1
        if not (0.0 < tp < 5.0):
            tp = 0.3
        return sl, tp

    def _effective_dry_run(self) -> bool:
        """Engine-side broadcast gate: module dry-run OR global kill-switch OR
        pause flag (`core.dry_run.should_skip_live`). The executors apply the
        same gate at the tx-write boundary; mirroring it here keeps the engine
        from labeling a gated (simulated) fill as a real one. Fail-safe: any
        error in the gate means NO live broadcast."""
        try:
            from core.dry_run import should_skip_live
            return should_skip_live(
                bool(self.config.get('dry_run', True)),
                module='dex',
                account=self.config.get('wallet_address'),
            )
        except Exception as e:
            logger.error(f"dry-run gate check failed (failing safe to DRY_RUN): {e}")
            return True

    # Wrapped-native token per chain — used to price the native currency in
    # USD when converting a USD position size into the executor's native-unit
    # BUY amount. Mirrors base_executor._get_weth_address.
    _WRAPPED_NATIVE = {
        'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'bsc': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',
        'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
        'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
        'base': '0x4200000000000000000000000000000000000006',
        'optimism': '0x4200000000000000000000000000000000000006',
        'avalanche': '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7',
    }

    @staticmethod
    def _to_checksum(addr: str) -> str:
        """EIP-55 checksum for EVM addresses (web3 contract calls reject
        lowercase). Fail-soft: returns the input unchanged on error."""
        try:
            from web3 import Web3
            return Web3.to_checksum_address(addr)
        except Exception:
            return addr

    def _live_slippage(self, key: str, default: float = 0.05) -> float:
        """trading.live_entry_slippage_pct / live_exit_slippage_pct (migration
        105). Fail-soft to the legacy hardcoded 0.05; sanity-clamped to
        (0, 0.5]."""
        try:
            v = float((self.config.get('trading', {}) or {}).get(key, default) or default)
        except (TypeError, ValueError):
            return default
        return v if 0.0 < v <= 0.5 else default

    async def _native_usd_price(self, chain: str) -> Optional[float]:
        """USD price of the chain's native token via the wrapped-native pair.
        Returns None when unresolvable — live callers must fail CLOSED (skip
        the trade) rather than guess."""
        try:
            wrapped = self._WRAPPED_NATIVE.get((chain or '').lower())
            if not wrapped:
                return None
            price = await self.dex_collector.get_token_price(
                token_address=wrapped, chain=chain
            )
            price = float(price) if price else 0.0
            return price if price > 0 else None
        except Exception as e:
            logger.error(f"native USD price lookup failed for {chain}: {e}")
            return None

    async def _check_daily_entry_budget(self) -> bool:
        """Consumer of `trading.max_trades_per_day` (seeded by migration 095).

        Returns True when a NEW entry is allowed today (UTC day). 0 or
        negative = unlimited. Opportunities are processed best-score-first,
        so the budget keeps the highest-conviction trades. ENTRY-only —
        exits never pass through _execute_opportunity, so they are never
        blocked. Fail-soft: any error allows the entry (never blocks on a
        broken config/DB read). Cap-reached is logged ONCE per UTC day.
        """
        try:
            cap = int((self.config.get('trading', {}) or {}).get('max_trades_per_day', 100) or 0)
        except (TypeError, ValueError):
            cap = 100
        if cap <= 0:
            return True
        try:
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if self._entry_budget_day != today:
                self._entry_budget_day = today
                self._entries_today = await self._count_entries_today_db()
            if self._entries_today >= cap:
                if self._entry_budget_logged_day != today:
                    self._entry_budget_logged_day = today
                    logger.warning(
                        f"📅 Daily entry budget reached ({self._entries_today}/{cap}, "
                        f"trading.max_trades_per_day) — skipping NEW entries until the "
                        f"next UTC day. Exits/position monitoring unaffected."
                    )
                return False
            return True
        except Exception as e:
            logger.debug(f"Daily entry budget check failed (fail-soft, allowing entry): {e}")
            return True

    async def _count_entries_today_db(self) -> int:
        """Restart-safe seed for the daily entry counter (UTC midnight).

        Counts today's BUY rows in the trades table. The column may be
        TIMESTAMP (naive) or TIMESTAMPTZ depending on deployment age, so try
        naive-UTC first, then tz-aware; fall back to 0 (in-memory counting
        still applies from that point on).
        """
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        query = "SELECT COUNT(*) AS n FROM trades WHERE side = 'buy' AND entry_timestamp >= $1"
        for ts in (midnight.replace(tzinfo=None), midnight):
            try:
                row = await self.db.fetch_one(query, ts)
                if row is not None:
                    return int(row['n'] or 0)
            except Exception:
                continue
        return 0

    def _register_entry(self) -> None:
        """Increment the per-UTC-day entry counter after a successful entry."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._entry_budget_day != today:
            self._entry_budget_day = today
            self._entries_today = 0
        self._entries_today += 1

    # ============================================================================
    # FIX 4: engine.py - Add duplicate position check in _execute_opportunity
    # Add this check at the beginning of _execute_opportunity method (around line 620)
    # ============================================================================

    async def _execute_opportunity(self, opportunity: TradingOpportunity):
        """Execute a trading opportunity"""
        try:
            token_symbol = opportunity.metadata.get('token_symbol', 'UNKNOWN')
            token_address = opportunity.token_address.lower()

            # ✅ CHECK 0 (Wave-26): per-chain weight gate + per-UTC-day entry
            # budget — the two migration-095 tunables that previously had NO
            # consumer. ENTRY-only: exits never pass through this method, so
            # they can never be blocked here.
            entry_chain = (opportunity.chain or 'ethereum').lower()
            chain_weight = self._get_chain_weight(entry_chain)
            if chain_weight <= 0.0:
                logger.info(
                    f"⚖️ Chain '{entry_chain}' weighted 0 in trading.chain_weights "
                    f"— skipping NEW entry {token_symbol} (measured negative edge)"
                )
                return
            if not await self._check_daily_entry_budget():
                return  # cap-reached warning already logged once per UTC day
            if chain_weight < 1.0:
                # (0,1) scales capital on chains with a weak measured edge.
                scaled = float(opportunity.recommended_position_size) * chain_weight
                logger.info(
                    f"⚖️ Chain weight {chain_weight:.2f} for '{entry_chain}': position "
                    f"${float(opportunity.recommended_position_size):.2f} → ${scaled:.2f}"
                )
                opportunity.recommended_position_size = scaled

            # DB-authoritative SL/TP for the position dicts below (was
            # hardcoded 0.1/0.3, contradicting risk_management.stop_loss_pct
            # 0.12 / take_profit_pct 0.24 used by sizing and the DB watchdog).
            sl_pct, tp_pct = self._position_sl_tp()

            # ✅ CHECK 1: Already have active position?
            # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
            async with self.positions_lock:
                if token_address in self.active_positions:
                    logger.warning(f"⚠️ Already have position in {token_symbol} - SKIPPING")
                    return
            
            # ✅ CHECK 2: Position on cooldown?
            if token_address in self.recently_closed:
                record = self.recently_closed[token_address]
                if not record.is_cooled_down(self.cooldown_minutes):
                    elapsed = (datetime.now(timezone.utc) - _as_utc(record.closed_at)).total_seconds() / 60
                    remaining = self.cooldown_minutes - elapsed
                    logger.warning(
                        f"❄️ COOLDOWN ACTIVE for {token_symbol}: "
                        f"closed {elapsed:.1f}min ago (reason: {record.reason}), "
                        f"{remaining:.1f}min remaining"
                    )
                    return
                else:
                    logger.info(f"♻️ Cooldown expired for {token_symbol}, can re-enter")
                    del self.recently_closed[token_address]
            
            # ✅ CHECK 3: Portfolio limits
            max_positions = self.config.get('portfolio', {}).get('max_positions', 40)
            # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
            async with self.positions_lock:
                num_positions = len(self.active_positions)

            if num_positions >= max_positions:
                logger.warning(f"⚠️ Max positions reached ({num_positions}) - SKIPPING")
                return

            # ✅ CHECK 4: CIRCUIT BREAKERS
            try:
                metrics = await self.risk_manager._get_current_metrics()
                breaker_ok, breaker_reason = self.risk_manager.check_circuit_breakers(metrics)
                
                if not breaker_ok:
                    logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {breaker_reason}")
                    logger.error(f"   Token: {token_symbol}")
                    logger.error(f"   All trading HALTED until conditions improve")
                    
                    await self.alert_manager.send_critical(
                        f"🚨 CIRCUIT BREAKER TRIPPED\n\n"
                        f"Reason: {breaker_reason}\n"
                        f"Token blocked: {token_symbol}\n\n"
                        f"All trading has been HALTED.\n"
                        f"Metrics:\n"
                        f"  • Error rate: {metrics.error_rate_pct:.1f}%\n"
                        f"  • Slippage: {metrics.realized_slippage_bps:.0f} bps\n"
                        f"  • Consecutive losses: {metrics.consecutive_losses}\n"
                        f"  • Drawdown: {metrics.drawdown_pct:.1f}%\n"
                        f"  • Daily loss: {metrics.daily_loss_pct:.1f}%"
                    )
                    
                    self.stats['circuit_breaker_trips'] = self.stats.get('circuit_breaker_trips', 0) + 1
                    return
                
                logger.debug(f"✅ Circuit breakers OK - proceeding with trade")
                
            except Exception as e:
                logger.error(f"❌ Circuit breaker check failed: {e}")
                logger.warning(f"⚠️  Aborting trade due to circuit breaker check failure")
                return
            #chain = opportunity.chain.lower()
            # ✅ Define all variables at the beginning
            chain = opportunity.chain.lower()
            position_value = opportunity.recommended_position_size
            eth_price = 4000.0  # Rough estimate for gas calculation

            # Engine-side broadcast gate: module DRY_RUN, global kill-switch
            # or pause flag all force the simulated branch — so a gated fill
            # is recorded as simulated, never as live.
            effective_dry_run = self._effective_dry_run()

            # Check wallet balance before trading (skip in DRY_RUN mode)
            if not effective_dry_run:  # ✅ Only check balance in LIVE trading
                if chain == 'solana':
                    balance = await self.solana_executor.get_balance()
                    required = position_value + 0.01  # Position + fees
                else:
                    w3 = self.trade_executor.w3
                    balance_wei = w3.eth.get_balance(self.trade_executor.wallet_address)
                    balance = float(w3.from_wei(balance_wei, 'ether'))
                    required = position_value / eth_price + 0.01  # Convert to ETH + gas
                
                if balance < required:
                    logger.error(f"❌ Insufficient balance: {balance} < {required}")
                    await self.alert_manager.send_critical(
                        f"⚠️ INSUFFICIENT FUNDS\n"
                        f"Chain: {chain}\n"
                        f"Required: ${required:.2f}\n"
                        f"Available: ${balance:.2f}"
                    )
                    return
            else:
                # ✅ DRY_RUN mode - log but don't block
                logger.info(f"🎯 DRY_RUN: Skipping balance check (would need ${position_value:.2f})")
            
            # Select appropriate executor
            if chain == 'solana':
                executor = self.solana_executor
                if not executor:
                    logger.error("❌ Solana executor not available")
                    return
                logger.info(f"🔷 Using Jupiter executor for Solana")
            else:
                # EVM chains use existing executor
                executor = self.trade_executor
                logger.info(f"🔶 Using EVM executor for {chain}")
            
            # Check if in DRY_RUN mode (module flag OR kill-switch OR pause)
            if effective_dry_run:
                logger.info(f"🎯 DRY RUN - SIMULATING TRADE:")
                logger.info(f"   Token: {token_symbol}")
                logger.info(f"   Address: {opportunity.token_address}")
                logger.info(f"   Chain: {chain.upper()}")
                logger.info(f"   Executor: {'Jupiter' if chain == 'solana' else 'EVM'}")
                logger.info(f"   Price: ${opportunity.price:.8f}")
                logger.info(f"   Score: {opportunity.score:.3f}")
                
                # Create simulated position (same as before)
                # ✅ Create simulated position with calculated size
                from decimal import Decimal
                import uuid
                
                # Use the recommended_position_size from opportunity (already calculated)
                position_value = Decimal(str(opportunity.recommended_position_size))
                simulated_amount = position_value / Decimal(str(opportunity.price))
                trade_id = str(uuid.uuid4())
                
                logger.info(f"   Position Size: ${position_value} ({simulated_amount:.4f} tokens)")
                
                position = {
                    'id': trade_id,
                    'position_id': f"DRY-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{token_symbol}",
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'entry_price': Decimal(str(opportunity.price)),
                    'amount': Decimal(str(simulated_amount)),
                    'entry_value': position_value,  # ✅ FROM CALCULATED SIZE!
                    'entry_time': datetime.now(),
                    'chain': opportunity.chain,
                    'strategy': {'name': opportunity.entry_strategy},
                    'risk_score': opportunity.risk_score,
                    'stop_loss_percentage': sl_pct,
                    'take_profit_percentage': tp_pct,
                    'max_hold_time': 60,
                    'metadata': opportunity.metadata,
                    'is_dry_run': True,
                    'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
                }

                # Add to active positions
                # CRITICAL FIX (P1): Protect with lock to prevent race conditions
                async with self.positions_lock:
                    self.active_positions[token_address] = position

                # Wave-26: count this entry against trading.max_trades_per_day
                self._register_entry()

                # ⚡ CRITICAL: Schedule immediate first check for this position
                # Don't wait for the monitoring loop - check within 5 seconds
                asyncio.create_task(
                    self._immediate_position_check(token_address, token_symbol),
                    name=f"immediate_check_{token_symbol}"
                )

                # ✅ NEW: Add to portfolio manager
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        # Calculate stop loss and take profit levels
                        stop_loss_price = float(opportunity.price) * (1 - position.get('stop_loss_percentage', sl_pct))
                        take_profit_price = float(opportunity.price) * (1 + position.get('take_profit_percentage', tp_pct))
                        
                        await self.portfolio_manager.update_portfolio({
                            'token_address': token_address,
                            'pair_address': opportunity.pair_address if hasattr(opportunity, 'pair_address') else token_address,  # ✅ FIXED
                            'chain': chain,
                            'side': 'buy',
                            'price': float(opportunity.price),  # ✅ FIXED
                            'amount': float(simulated_amount),  # ✅ FIXED
                            'cost': float(position_value),  # ✅ FIXED
                            'stop_loss': stop_loss_price,  # ✅ FIXED
                            'take_profits': [take_profit_price],  # ✅ FIXED
                            'strategy': opportunity.entry_strategy,  # ✅ FIXED
                            'id': position['position_id'],  # ✅ FIXED
                            'symbol': opportunity.metadata.get('token_symbol', 'UNKNOWN')  # ✅ Get from metadata
                        })
                        logger.debug(f"✅ Position added to portfolio manager")
                    except Exception as e:
                        logger.error(f"Error adding position to portfolio manager: {e}")
                
                # ✅ LOG TO DATABASE
                try:
                    trade_data = {
                        'trade_id': trade_id,
                        'token_address': token_address,
                        'chain': opportunity.chain,
                        'side': 'buy',
                        'entry_price': float(opportunity.price),
                        'exit_price': None,
                        'amount': float(simulated_amount),
                        'usd_value': float(position_value),
                        'gas_fee': 0.0,
                        'slippage': 0.0,
                        'profit_loss': None,
                        'profit_loss_percentage': None,
                        'strategy': opportunity.entry_strategy,
                        'risk_score': float(opportunity.risk_score.overall_risk) if opportunity.risk_score else None,
                        'ml_confidence': float(opportunity.ml_confidence),
                        'entry_timestamp': datetime.now(),
                        'exit_timestamp': None,
                        'status': 'open',
                        'metadata': {
                            'token_symbol': token_symbol,
                            'is_dry_run': True,
                            'opportunity_score': float(opportunity.score),
                            'executor_type': 'Jupiter' if chain == 'solana' else 'EVM',
                            # Persist entry-time SL/TP so _load_state restores
                            # the SAME stop after a subprocess restart.
                            'stop_loss_percentage': sl_pct,
                            'take_profit_percentage': tp_pct
                        }
                    }
                    await self.db.save_trade(trade_data)
                    logger.info(f"✅ Trade logged to database: {trade_id}")

                    # 🆕 PATCH: Structured trade entry logging
                    try:
                        chain = opportunity.chain if hasattr(opportunity, 'chain') else 'unknown'
                        log_trade_entry(
                            chain=chain,
                            symbol=token_symbol,
                            token_address=opportunity.token_address,
                            trade_id=trade_id,
                            entry_price=float(opportunity.price),
                            amount=float(simulated_amount),
                            size_usd=float(position_value),
                            reason="opportunity_signal"
                        )
                    except Exception as log_err:
                        logger.warning(f"Failed to log trade entry: {log_err}")

                except Exception as e:
                    logger.error(f"❌ Failed to log trade to database: {e}")
                
                # Update stats. successful_trades means "closed at profit"
                # and is incremented ONLY at close — counting it here too
                # made successful > total (58 trades / 84 successful),
                # corrupting win_rate-driven sizing.
                self.stats['total_trades'] += 1

                # Update circuit breaker metrics for successful real trade
                # Update circuit breaker metrics for simulated trade
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': 0,  # Entry only, no P&L yet
                    'slippage_bps': 0  # Simulated, no real slippage in dry run
                })
                
                # Send alert
                executor_emoji = "🔷" if chain == 'solana' else "🔶"
                await self.alert_manager.send_trade_alert(
                    f"📝 DRY RUN - OPENED POSITION: {token_symbol}\n"
                    f"Chain: {chain.upper()} {executor_emoji}\n"
                    f"Executor: {'Jupiter' if chain == 'solana' else 'EVM DEX'}\n"
                    f"Entry: ${opportunity.price:.8f}\n"
                    f"Amount: {simulated_amount:.2f} tokens\n"
                    f"Value: ${position_value:.2f}\n"
                    f"Score: {opportunity.score:.3f}\n"
                    f"Stop Loss: -{sl_pct * 100:.0f}% | Take Profit: +{tp_pct * 100:.0f}%\n"
                    f"Max Hold: 60 minutes"
                )
                
                logger.info(f"✅ DRY RUN position added to tracking: {token_symbol}")
                # Update circuit breaker metrics for successful entry
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': 0,  # Entry only, no P&L yet
                    'slippage_bps': 0  # Simulated, no real slippage
                })
                return
            
            # REAL EXECUTION
            logger.info(f"💰 EXECUTING REAL TRADE for {token_symbol} on {chain.upper()}")

            from trading.orders.order_manager import Order, OrderType
            from trading.executors.base_executor import TradeOrder

            position_value_usd = float(opportunity.recommended_position_size)
            trade_id = str(uuid.uuid4())  # generated BEFORE broadcast

            # ✅ RISK GATE (fail-closed): RiskManager.validate_trade before any
            # live broadcast — circuit breakers, FRESH honeypot/liquidity/dev
            # risk, position-size cap and the cross-module allocation guard.
            try:
                risk_ok, risk_reason = await self.risk_manager.validate_trade(
                    token_address, position_value_usd
                )
            except Exception as e:
                risk_ok, risk_reason = False, f"validate_trade error: {e}"
            if not risk_ok:
                logger.error(f"🛑 RiskManager BLOCKED live entry {token_symbol}: {risk_reason}")
                await self.alert_manager.send_warning(
                    f"🛑 Live entry blocked by RiskManager: {token_symbol}\n{risk_reason}"
                )
                return

            # ✅ FINAL SAFETY CHECKS before real execution
            logger.info(f"🔍 FINAL SAFETY CHECKS for {token_symbol}...")

            # 1. Position size against portfolio limits
            max_position = self.portfolio_manager.get_max_position_size(opportunity.chain)
            if position_value_usd > max_position:
                logger.error(
                    f"❌ POSITION SIZE EXCEEDS LIMIT: "
                    f"${position_value_usd:.2f} > ${max_position:.2f}"
                )
                await self.alert_manager.send_error(
                    f"Trade cancelled - Position size too large\n"
                    f"Token: {token_symbol}\n"
                    f"Requested: ${position_value_usd:.2f}\n"
                    f"Max allowed: ${max_position:.2f}"
                )
                return
            logger.info(f"  ✅ Position size check passed")

            if chain == 'solana':
                # Jupiter executor consumes the order_manager Order shape.
                order = Order(
                    order_id=trade_id,
                    token_in="So11111111111111111111111111111111111111112",
                    token_out=opportunity.token_address,
                    amount=Decimal(str(position_value_usd / opportunity.price)),
                    order_type=OrderType.MARKET,
                    slippage=self._live_slippage('live_entry_slippage_pct'),
                    chain=opportunity.chain,
                    wallet_address=str(self.solana_executor.wallet_keypair.pubkey()),
                    metadata={
                        'opportunity_id': opportunity.metadata.get('opportunity_id'),
                        'token_symbol': token_symbol,
                        'score': opportunity.score,
                        'executor_type': 'Jupiter'
                    }
                )
            else:
                # EVM executor BUY amount is in NATIVE units. Convert the USD
                # position size via the wrapped-native price; fail CLOSED when
                # the price cannot be resolved (never guess with live funds).
                native_usd = await self._native_usd_price(chain)
                if not native_usd:
                    logger.error(
                        f"❌ Cannot resolve native USD price for {chain} — "
                        f"skipping live entry {token_symbol} (fail-closed)"
                    )
                    return
                amount_native = position_value_usd / native_usd

                # 2. Native balance must cover size + gas headroom
                try:
                    w3 = self.trade_executor.w3
                    balance_native = float(w3.from_wei(
                        w3.eth.get_balance(self.trade_executor.wallet_address), 'ether'
                    ))
                except Exception as e:
                    logger.error(f"❌ Balance check failed: {e}")
                    return
                if balance_native < amount_native * 1.1:  # 10% gas buffer
                    logger.error(
                        f"❌ INSUFFICIENT BALANCE: have {balance_native:.6f}, "
                        f"need {amount_native * 1.1:.6f} native"
                    )
                    await self.alert_manager.send_error(
                        f"Trade cancelled - Insufficient balance\n"
                        f"Token: {token_symbol}\n"
                        f"Have: {balance_native:.6f}\n"
                        f"Need: {amount_native * 1.1:.6f} (native, incl. gas buffer)"
                    )
                    return
                logger.info(f"  ✅ Balance check passed: {balance_native:.6f} native")

                order = TradeOrder(
                    token_address=self._to_checksum(opportunity.token_address),
                    side='buy',
                    amount=amount_native,
                    slippage=self._live_slippage('live_entry_slippage_pct'),
                    deadline=300,
                    use_mev_protection=True,
                    metadata={
                        'opportunity_id': opportunity.metadata.get('opportunity_id'),
                        'token_symbol': token_symbol,
                        'score': opportunity.score,
                        'trade_id': trade_id,
                        'executor_type': 'EVM'
                    }
                )

            # Re-assert the broadcast gate right before send: the kill-switch
            # may have tripped between the entry checks and this point.
            if self._effective_dry_run():
                logger.warning(
                    f"🛑 Broadcast gate closed (kill-switch/pause) — dropping "
                    f"live entry {token_symbol}"
                )
                return

            logger.warning(
                f"⚠️  ABOUT TO EXECUTE REAL TRADE: {token_symbol} "
                f"${position_value_usd:.2f} on {chain}"
            )

            # Both executors confirm the receipt internally and return a Dict
            # (success only when receipt.status == 1) — no second wait needed.
            result = await executor.execute_trade(order)
            if not isinstance(result, dict):
                logger.error(
                    f"❌ Executor returned unexpected result type "
                    f"{type(result).__name__} — treating as failed trade"
                )
                result = {'success': False, 'error': 'unexpected executor result type'}

            # A fill simulated at the executor boundary (kill-switch race /
            # executor-level dry-run) must NEVER be recorded as a live one.
            tx_hash = result.get('tx_hash') or result.get('signature') or result.get('transactionHash')
            if result.get('success'):
                if ((result.get('metadata') or {}).get('dry_run')
                        or str(tx_hash or '').startswith('0xDRYRUN')):
                    logger.warning(
                        f"🛑 Executor SIMULATED the fill for {token_symbol} "
                        f"(kill-switch/pause) — not recording a live position"
                    )
                    return
                if not tx_hash:
                    logger.error(
                        f"❌ Executor reported success WITHOUT a tx hash for "
                        f"{token_symbol} — refusing to record position"
                    )
                    result = {'success': False, 'error': 'success without tx hash'}

            if result['success']:
                # Track position. token_amount fallback: estimated from the
                # USD size and signal price (NOT order.amount, which is in
                # native units for the EVM leg).
                est_token_amount = position_value_usd / opportunity.price if opportunity.price > 0 else 0.0
                position = {
                    'token_address': opportunity.token_address,
                    'token_symbol': token_symbol,
                    # Decimal like the DRY/restored paths — the monitoring /
                    # exit arithmetic mixes these with Decimal and a float
                    # here silently breaks every PnL/trailing computation.
                    'entry_price': Decimal(str(result.get('execution_price') or opportunity.price)),
                    'amount': Decimal(str(result.get('token_amount') or est_token_amount)),
                    'entry_value': opportunity.recommended_position_size,
                    'tx_hash': tx_hash,
                    'chain': opportunity.chain,
                    'strategy': {'name': opportunity.entry_strategy},
                    'risk_score': opportunity.risk_score,
                    'entry_time': datetime.now(),
                    'stop_loss_percentage': sl_pct,
                    'take_profit_percentage': tp_pct,
                    'max_hold_time': 60,
                    'is_dry_run': False,
                    'trade_id': trade_id,
                    'metadata': {
                        **opportunity.metadata,
                        'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
                    }
                }

                # CRITICAL FIX (P1): Protect with lock to prevent race conditions
                async with self.positions_lock:
                    self.active_positions[opportunity.token_address] = position

                # Wave-26: count this entry against trading.max_trades_per_day
                self._register_entry()

                # ⚡ CRITICAL: Schedule immediate first check for this position
                asyncio.create_task(
                    self._immediate_position_check(opportunity.token_address, token_symbol),
                    name=f"immediate_check_{token_symbol}"
                )

                # successful_trades means "closed at profit" — incremented
                # only at close (see _close_position). Entry counts only
                # total_trades so win_rate stays wins/total.
                self.stats['total_trades'] += 1

                # Update circuit breaker metrics for successful real trade
                actual_slippage = result.get('slippage_bps', 0)
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': 0,  # Entry only, no P&L yet
                    'slippage_bps': actual_slippage
                })

                # ✅ ADD COMPREHENSIVE LOGGING (NEW)
                execution_price = float(result.get('execution_price') or opportunity.price)
                token_amount = float(result.get('token_amount') or est_token_amount)
                actual_value = float(execution_price * token_amount)
                slippage_pct = ((execution_price - opportunity.price) / opportunity.price * 100) if opportunity.price > 0 else 0
                
                logger.info(f"✅ REAL TRADE EXECUTED SUCCESSFULLY:")
                logger.info(f"   Token: {token_symbol}")
                logger.info(f"   Chain: {chain.upper()}")
                logger.info(f"   Executor: {'Jupiter' if chain == 'solana' else 'EVM DEX'}")
                logger.info(f"   Expected Price: ${opportunity.price:.8f}")
                logger.info(f"   Execution Price: ${execution_price:.8f}")
                logger.info(f"   Slippage: {slippage_pct:+.2f}%")
                logger.info(f"   Amount: {token_amount:.4f} tokens")
                logger.info(f"   Target Value: ${opportunity.recommended_position_size:.2f}")
                logger.info(f"   Actual Value: ${actual_value:.2f}")
                logger.info(f"   TX Hash: {tx_hash or 'N/A'}")
                logger.info(f"   Gas Used: ${result.get('gas_fee', 0):.4f}")

                # ✅ LOG REAL TRADE TO DATABASE — trade_id was generated
                # BEFORE broadcast so the position and the row always agree.
                try:

                    trade_data = {
                        'trade_id': trade_id,
                        'token_address': token_address,
                        'chain': opportunity.chain,
                        'side': 'buy',
                        'entry_price': float(execution_price),
                        'exit_price': None,
                        'amount': float(token_amount),
                        'usd_value': actual_value,
                        'gas_fee': float(result.get('gas_fee', 0)),
                        # Executors report 'slippage_actual' (fraction); some
                        # legacy paths report 'slippage_bps'.
                        'slippage': (float(result.get('slippage_bps')) / 10000
                                     if result.get('slippage_bps') is not None
                                     else float(result.get('slippage_actual') or 0)),
                        'profit_loss': None,
                        'profit_loss_percentage': None,
                        'strategy': opportunity.entry_strategy,
                        'risk_score': float(opportunity.risk_score.overall_risk) if opportunity.risk_score else None,
                        'ml_confidence': float(opportunity.ml_confidence),
                        'entry_timestamp': datetime.now(),
                        'exit_timestamp': None,
                        'status': 'open',
                        'metadata': {
                            'token_symbol': token_symbol,
                            'is_dry_run': False,  # ✅ REAL TRADE
                            'opportunity_score': float(opportunity.score),
                            'executor_type': 'Jupiter' if chain == 'solana' else 'EVM',
                            'tx_hash': tx_hash,
                            'expected_price': float(opportunity.price),
                            'execution_price': float(execution_price),
                            'slippage_bps': result.get('slippage_bps', 0),
                            # Persist entry-time SL/TP so _load_state restores
                            # the SAME stop after a subprocess restart.
                            'stop_loss_percentage': sl_pct,
                            'take_profit_percentage': tp_pct
                        }
                    }
                    
                    await self.db.save_trade(trade_data)
                    logger.info(f"✅ Real trade logged to database: {trade_id}")

                except Exception as e:
                    # The position IS live on-chain: keep tracking it in
                    # memory and alert loudly that the DB row is missing.
                    logger.error(f"❌ Failed to log real trade to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    await self.alert_manager.send_critical(
                        f"🚨 LIVE position OPENED ON-CHAIN but DB write FAILED: "
                        f"{token_symbol} (tx {tx_hash}). Position tracked "
                        f"in-memory only — restart will orphan it."
                    )

                # Send success alert
                tx_link = result.get('explorer_url', f"Transaction: {str(tx_hash or 'N/A')[:10]}...")
                slippage_pct = ((execution_price - opportunity.price) / opportunity.price * 100) if opportunity.price > 0 else 0
                
                await self.alert_manager.send_trade_alert(
                    f"✅ OPENED REAL POSITION: {token_symbol}\n"
                    f"Chain: {chain.upper()}\n"
                    f"Executor: {'Jupiter 🔷' if chain == 'solana' else 'EVM DEX 🔶'}\n"
                    f"Expected: ${opportunity.price:.8f}\n"
                    f"Executed: ${execution_price:.8f}\n"
                    f"Slippage: {slippage_pct:+.2f}%\n"
                    f"Amount: {token_amount:.4f} tokens\n"
                    f"Target: ${opportunity.recommended_position_size:.2f}\n"
                    f"Actual: ${actual_value:.2f}\n"
                    f"Gas: ${result.get('gas_fee', 0):.4f}\n"
                    f"Score: {opportunity.score:.3f}\n"
                    f"Tx: {tx_link}"
                )
                
            else:
                self.stats['failed_trades'] += 1
                
                await self.alert_manager.send_warning(
                    f"❌ TRADE FAILED: {token_symbol}\n"
                    f"Chain: {chain.upper()}\n"
                    f"Error: {result.get('error', 'Unknown error')}"
                )

                # Update circuit breaker metrics for failed trade
                self.risk_manager.update_trade_metrics({
                    'success': False,
                    'profit': 0,
                    'slippage_bps': 0
                })
                
        except Exception as e:
            logger.error(f"Error executing opportunity: {e}", exc_info=True)
            await self.alert_manager.send_error(f"Trade execution error: {e}")
            
    # ============================================================================
    # FIX 1: engine.py - Fix position monitoring and tracking
    # Replace the entire _monitor_existing_positions method (around line 750)
    # ============================================================================

    async def _monitor_existing_positions(self):
        """Monitor and manage existing positions"""
        logger.info("📊 Starting position monitoring loop...")
        
        while self.state == BotState.RUNNING:
            # ✅ FIX: Initialize at outer scope
            token_address = None
            position = None
            
            try:
                # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
                async with self.positions_lock:
                    has_positions = len(self.active_positions) > 0

                if not has_positions:
                    await asyncio.sleep(5)
                    continue
                
                # CRITICAL FIX (P1): Protect with lock to prevent race conditions during iteration
                async with self.positions_lock:
                    num_positions = len(self.active_positions)
                    positions_snapshot = list(self.active_positions.items())

                logger.info(f"📊 Monitoring {num_positions} active positions...")

                # ✅ STEP 1: Collect all prices first (REFACTORED FOR RELIABILITY)
                price_data = {}
                for token_address, position in positions_snapshot:
                    try:
                        chain = position.get('chain', 'ethereum')
                        token_symbol = position.get('token_symbol', 'UNKNOWN')
                        
                        # Ensure the pair address from the original opportunity is available
                        pair_address = position.get('metadata', {}).get('pair', {}).get('pair_address')

                        if not pair_address:
                            logger.warning(f"⚠️ Missing pair_address in metadata for {token_symbol}. Falling back to token address lookup.")
                            # Fallback to the less reliable method if pair_address is missing
                            current_price = await self.dex_collector.get_token_price(token_address=token_address, chain=chain)
                            if current_price:
                                price_data[token_address] = float(current_price)
                            else:
                                logger.warning(f"⚠️ Could not get price for {token_symbol} using fallback.")
                            continue

                        # Use the more reliable get_pair_data method
                        pair_data = await self.dex_collector.get_pair_data(
                            pair_address=pair_address,
                            chain=chain
                        )
                        
                        if pair_data and 'price' in pair_data:
                            price_data[token_address] = float(pair_data['price'])
                        else:
                            logger.warning(
                                f"⚠️ Could not get price for {token_symbol} "
                                f"(Pair: {pair_address[:10]}...) on {chain}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching price for {token_address}: {e}", exc_info=True)
                        continue
                
                # ✅ STEP 2: Bulk update portfolio manager
                if price_data and hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        updated_count = await self.portfolio_manager.update_all_positions(price_data)
                        logger.debug(f"📊 Updated {updated_count} positions in portfolio manager")
                    except Exception as e:
                        logger.error(f"Error updating portfolio positions: {e}")
                
                # ✅ STEP 3: Update active_positions dict (for backward compatibility)
                # CRITICAL FIX (P1): Take snapshot under lock to prevent race conditions
                async with self.positions_lock:
                    positions_items = list(self.active_positions.items())

                for token_address, position in positions_items:
                    try:
                        position_symbol = position.get('token_symbol', 'UNKNOWN')

                        if token_address in price_data:
                            current_price = price_data[token_address]

                            # Reset failed price fetch counter on success
                            position['_price_fetch_failures'] = 0
                            position['_last_price_update'] = datetime.now()

                            from decimal import Decimal
                            position['current_price'] = Decimal(str(current_price))
                            position['current_value'] = Decimal(str(current_price)) * position['amount']

                            # Calculate P&L
                            entry_value = position['entry_price'] * position['amount']
                            current_value = position['current_value']
                            position['pnl'] = current_value - entry_value
                            position['pnl_percentage'] = float((current_value - entry_value) / entry_value * 100)

                            # Calculate holding time
                            holding_time = (datetime.now(timezone.utc) - _as_utc(position['entry_time'])).total_seconds() / 60

                            logger.info(
                                f"  📈 {position_symbol} - "
                                f"Entry: ${position['entry_price']:.8f}, Current: ${current_price:.8f}, "
                                f"P&L: {position['pnl_percentage']:.2f}% (${position['pnl']:.2f}), "
                                f"Time: {holding_time:.1f}min"
                            )

                            # Check exit conditions
                            should_exit, reason = await self._check_exit_conditions(position)

                            if should_exit:
                                logger.info(
                                    f"  🚪 EXIT SIGNAL for {position_symbol}: {reason}",
                                    extra={
                                        'token_address': token_address,
                                        'symbol': position_symbol,
                                        'reason': reason
                                    }
                                )
                                await self._close_position(position, reason)
                            else:
                                # Update position in tracker
                                if hasattr(self, 'position_tracker') and self.position_tracker:
                                    await self.position_tracker.update_position(
                                        position.get('tracker_id', ''),
                                        {'current_price': current_price}
                                    )
                        else:
                            # ⚠️ CRITICAL: Price fetch failed - track failures for honeypot detection
                            failures = position.get('_price_fetch_failures', 0) + 1
                            position['_price_fetch_failures'] = failures

                            holding_time = (datetime.now(timezone.utc) - _as_utc(position['entry_time'])).total_seconds() / 60
                            last_price_update = position.get('_last_price_update', position['entry_time'])
                            time_since_price = (datetime.now(timezone.utc) - _as_utc(last_price_update)).total_seconds() / 60

                            logger.warning(
                                f"  ⚠️ PRICE FETCH FAILED for {position_symbol} "
                                f"(failures: {failures}, time since last price: {time_since_price:.1f}min, holding: {holding_time:.1f}min)"
                            )

                            # EMERGENCY CLOSE: After 3 consecutive price failures OR 5+ minutes without price
                            # This protects against honeypot tokens that become untradeable
                            max_price_failures = self.config.get('max_price_failures', 3)
                            max_time_without_price_mins = self.config.get('max_time_without_price_mins', 5)

                            if failures >= max_price_failures or time_since_price >= max_time_without_price_mins:
                                logger.error(
                                    f"  🚨 EMERGENCY CLOSE for {position_symbol} - "
                                    f"Cannot get price data (possible honeypot/rug). "
                                    f"Failures: {failures}, Time without price: {time_since_price:.1f}min"
                                )
                                # Mark as honeypot for future reference
                                position['_suspected_honeypot'] = True
                                position['_honeypot_reason'] = f"Price unavailable after {failures} attempts"
                                await self._close_position(position, "emergency_no_price")

                    except Exception as e:
                        # ✅ FIX: Now all variables are guaranteed to be defined
                        logger.error(
                            f"Error monitoring position {token_address[:10] if token_address else 'unknown'} "
                            f"({position_symbol or 'UNKNOWN'}): {e}",
                            extra={
                                'token_address': token_address,
                                'symbol': position_symbol,
                                'error': str(e)
                            },
                            exc_info=True
                        )
                        continue
                
                # Log portfolio summary
                # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
                async with self.positions_lock:
                    positions_snapshot = list(self.active_positions.values())

                total_value = sum(p.get('current_value', 0) for p in positions_snapshot)
                total_pnl = sum(p.get('pnl', 0) for p in positions_snapshot)
                
                # ✅ NEW: Get per-chain breakdown
                chain_summary = ""
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        chain_metrics = self.portfolio_manager.get_chain_metrics()
                        if chain_metrics:
                            chain_details = []
                            for chain, metrics in chain_metrics.items():
                                chain_details.append(
                                    f"{chain.upper()}: {metrics['positions']} pos, "
                                    f"${metrics['value']:.2f} ({metrics['roi']:.1f}%)"
                                )
                            chain_summary = " | " + " | ".join(chain_details)
                    except Exception as e:
                        logger.debug(f"Could not get chain metrics: {e}")
                
                logger.info(
                    f"💼 Portfolio: {len(positions_snapshot)} positions, "
                    f"Value: ${total_value:.2f}, P&L: ${total_pnl:.2f}{chain_summary}"
                )

                # 🆕 Log strategy statistics every 10 iterations (to avoid spam)
                if hasattr(self, '_position_monitor_iteration'):
                    self._position_monitor_iteration += 1
                else:
                    self._position_monitor_iteration = 1

                if self._position_monitor_iteration % 10 == 0:
                    try:
                        self.strategy_manager.log_strategy_stats()
                    except Exception as e:
                        logger.debug(f"Could not log strategy stats: {e}")

                # CRITICAL: Use dynamic interval based on position age
                # New positions need faster monitoring to catch rapid price drops
                base_interval = self.config.get('position_update_interval_seconds', 10)

                # Check if any position is "new" (less than 5 minutes old)
                # Note: positions_snapshot here contains just position dicts (values), not (key, value) tuples
                has_new_positions = False
                for pos in positions_snapshot:
                    entry_time = pos.get('entry_time')
                    if entry_time:
                        holding_mins = (datetime.now(timezone.utc) - _as_utc(entry_time)).total_seconds() / 60
                        if holding_mins < 5:  # Position is less than 5 minutes old
                            has_new_positions = True
                            break

                # Use rapid interval (3 sec) for new positions, normal interval (10 sec) otherwise
                if has_new_positions:
                    update_interval = 3  # Check every 3 seconds for new positions
                    logger.debug(f"⚡ Rapid monitoring mode (new positions exist) - next check in {update_interval}s")
                else:
                    update_interval = base_interval

                await asyncio.sleep(update_interval)
                
            except Exception as e:
                # ✅ FIX: Safe variable access in outer exception
                logger.error(
                    f"Error in position monitoring loop for {token_address[:10] if token_address else 'unknown'}: {e}",
                    exc_info=True
                )
                await asyncio.sleep(30)

    async def _immediate_position_check(self, token_address: str, token_symbol: str):
        """
        ⚡ CRITICAL: Immediate first check for new positions
        This runs independently of the main monitoring loop to catch rapid price drops
        """
        try:
            # Wait 5 seconds to allow price data to propagate
            await asyncio.sleep(5)

            # Check every 5 seconds for the first 3 minutes
            for check_num in range(36):  # 36 checks x 5 seconds = 3 minutes
                # Check if position still exists
                async with self.positions_lock:
                    if token_address not in self.active_positions:
                        logger.info(f"⚡ Immediate check: {token_symbol} no longer in active positions")
                        return
                    position = self.active_positions[token_address]

                # Get current price
                chain = position.get('chain', 'ethereum')
                pair_address = position.get('metadata', {}).get('pair', {}).get('pair_address')

                current_price = None
                if pair_address:
                    pair_data = await self.dex_collector.get_pair_data(pair_address=pair_address, chain=chain)
                    if pair_data and 'price' in pair_data:
                        current_price = float(pair_data['price'])

                if not current_price:
                    # Fallback
                    current_price = await self.dex_collector.get_token_price(token_address=token_address, chain=chain)

                if current_price:
                    # Calculate P&L
                    from decimal import Decimal
                    entry_price = float(position['entry_price'])
                    pnl_percentage = ((current_price - entry_price) / entry_price) * 100

                    # Check stop loss (immediate trigger if hit)
                    stop_loss_pct = -position.get('stop_loss_percentage', 0.12) * 100

                    if check_num == 0 or check_num % 6 == 0:  # Log every 30 seconds
                        logger.info(
                            f"⚡ Rapid check #{check_num+1} for {token_symbol}: "
                            f"P&L {pnl_percentage:+.2f}% (stop @ {stop_loss_pct:.0f}%)"
                        )

                    if pnl_percentage <= stop_loss_pct:
                        logger.warning(
                            f"🚨 RAPID STOP LOSS TRIGGERED for {token_symbol}! "
                            f"P&L: {pnl_percentage:.2f}% at check #{check_num+1}"
                        )
                        # Update position with current price before closing
                        position['current_price'] = Decimal(str(current_price))
                        position['pnl_percentage'] = pnl_percentage
                        await self._close_position(position, "stop_loss_rapid")
                        return

                    # Check take profit too
                    take_profit_pct = position.get('take_profit_percentage', 0.3) * 100
                    if pnl_percentage >= take_profit_pct:
                        logger.info(
                            f"✅ RAPID TAKE PROFIT for {token_symbol}! "
                            f"P&L: {pnl_percentage:.2f}% at check #{check_num+1}"
                        )
                        position['current_price'] = Decimal(str(current_price))
                        position['pnl_percentage'] = pnl_percentage
                        await self._close_position(position, "take_profit_rapid")
                        return
                else:
                    logger.warning(f"⚡ Rapid check #{check_num+1}: Could not get price for {token_symbol}")

                await asyncio.sleep(5)

            logger.info(f"⚡ Rapid monitoring complete for {token_symbol} (position handed to main loop)")

        except Exception as e:
            logger.error(f"Error in immediate position check for {token_symbol}: {e}", exc_info=True)


    async def _calculate_position_size(
        self,
        risk_score: Optional[RiskScore] = None,
        opportunity_score: float = 0.7
    ) -> float:
        """
        Calculates a dynamic position size based on portfolio value, risk, and opportunity.
        """
        try:
            portfolio_config = self.config.get('portfolio', {})
            risk_config = self.config.get('risk_management', {})

            # Get current total portfolio value
            portfolio_balance = float(await self.portfolio_manager.get_portfolio_value())
            
            # Get risk parameters from config
            risk_per_trade_pct = risk_config.get('risk_per_trade_pct', 0.02)  # Risk 2% of portfolio by default
            stop_loss_pct = risk_config.get('stop_loss_pct', 0.12) # Use the tuned stop-loss

            # Calculate the total capital to risk on this trade
            capital_at_risk = portfolio_balance * risk_per_trade_pct

            # Kelly Criterion Calculation
            # K% = W - (1-W)/R
            # W = Win Rate (default 0.6 if unknown)
            # R = Reward/Risk Ratio (default 2.0)
            # Bounded [0,1]: successful_trades = closed-at-profit only.
            win_rate = min(1.0, self.stats['successful_trades'] / max(1, self.stats['total_trades'])) if self.stats['total_trades'] > 10 else 0.6
            profit_factor = 2.0 # Target 2:1

            kelly_pct = win_rate - (1 - win_rate) / profit_factor

            # Use Half-Kelly for safety
            kelly_fraction = max(0.0, kelly_pct * 0.5)

            # Combine Base Risk with Kelly
            # If Kelly suggests higher risk and opportunity score is high, boost size
            opportunity_multiplier = 0.8 + (opportunity_score * 0.4) # Range [0.8, 1.2]

            # If Kelly is high (high confidence), allow up to 2x standard risk
            if kelly_fraction > risk_per_trade_pct:
                risk_per_trade_pct = min(kelly_fraction, risk_per_trade_pct * 2)

            adjusted_capital_at_risk = portfolio_balance * risk_per_trade_pct * opportunity_multiplier
            
            # Calculate position size based on stop-loss
            # Position Size = Capital at Risk / Stop-Loss Percentage
            if stop_loss_pct <= 0:
                logger.warning("Stop loss percentage is zero or negative. Using a safe default of 10%.")
                stop_loss_pct = 0.10
                
            position_size = adjusted_capital_at_risk / stop_loss_pct

            # Get min/max position size limits from config
            min_size = portfolio_config.get('min_position_size_usd', 5.0)
            max_size = portfolio_config.get('max_position_size_usd', 100.0) # Increased max cap
            
            # Enforce absolute min/max limits
            final_size = max(min_size, min(position_size, max_size))

            logger.info(f"💰 Dynamic Position Sizing:")
            logger.info(f"   Portfolio Value: ${portfolio_balance:,.2f}")
            logger.info(f"   Base Capital at Risk ({risk_per_trade_pct:.1%}): ${capital_at_risk:,.2f}")
            logger.info(f"   Opportunity Score: {opportunity_score:.2f} (Multiplier: {opportunity_multiplier:.2f})")
            logger.info(f"   Adjusted Capital at Risk: ${adjusted_capital_at_risk:,.2f}")
            logger.info(f"   Stop-Loss Pct: {stop_loss_pct:.1%}")
            logger.info(f"   Calculated Size (Risk/SL): ${position_size:,.2f}")
            logger.info(f"   Clamped Size (Min: ${min_size}, Max: ${max_size}): ${final_size:,.2f}")

            return float(final_size)
            
        except Exception as e:
            logger.error(f"Error in dynamic position size calculation: {e}", exc_info=True)
            # Fallback to a safe, fixed position size on error
            return portfolio_config.get('min_position_size_usd', 10.0)
                
    # ============================================================================
    # FIX 2: engine.py - Fix _check_exit_conditions method (around line 820)
    # ============================================================================

    async def _check_exit_conditions(self, position: Dict) -> tuple[bool, str]:
        """Check if position should be closed"""
        try:
            # Get position details
            pnl_percentage = position.get('pnl_percentage', 0)
            holding_time = (datetime.now(timezone.utc) - _as_utc(position['entry_time'])).total_seconds() / 60  # minutes
            
            # 1. Take profit hit (default 30%)
            take_profit = position.get('take_profit_percentage', 0.3) * 100
            if pnl_percentage >= take_profit:
                logger.info(f"  ✅ Take profit hit: {pnl_percentage:.2f}% >= {take_profit:.2f}%")
                return True, "take_profit"
            
            # 2. Stop loss hit
            if 'stop_loss_price' in position:
                if position['current_price'] <= position['stop_loss_price']:
                    logger.info(f"  🛑 Custom stop loss hit: ${position['current_price']:.8f} <= ${position['stop_loss_price']:.8f}")
                    return True, "stop_loss"
            else:
                stop_loss_pct = -position.get('stop_loss_percentage', 0.12) * 100
                if pnl_percentage <= stop_loss_pct:
                    logger.info(f"  🛑 Percentage stop loss hit: {pnl_percentage:.2f}% <= {stop_loss_pct:.2f}%")
                    return True, "stop_loss"
            
            # 3. Time-based exit (default 60 minutes for scalping)
            max_hold_time = position.get('max_hold_time', 60)  # minutes
            if holding_time > max_hold_time:
                logger.info(f"  ⏰ Max hold time reached: {holding_time:.1f}min > {max_hold_time}min")
                return True, "time_limit"
            
            # 4. Break-even stop-loss
            if pnl_percentage >= 10 and not position.get('is_break_even', False):
                position['stop_loss_price'] = position['entry_price']
                position['is_break_even'] = True
                logger.info(f"  🛡️ Break-even stop-loss activated for {position.get('token_symbol', 'UNKNOWN')} at ${position['entry_price']:.8f}")

            # 5. Ratchet Trailing Stop (Advanced)
            if pnl_percentage > 10: # Activates after 10% profit
                max_profit = position.get('max_profit', pnl_percentage)
                position['max_profit'] = max(max_profit, pnl_percentage)

                trailing_stop_pct = 6 # Default 6% trail

                # Ratchet Logic
                if max_profit >= 50:
                    trailing_stop_pct = 2 # Extremely tight at 50%+ gain
                elif max_profit >= 30:
                    trailing_stop_pct = 3 # Tighten to 3%
                elif max_profit >= 15:
                    trailing_stop_pct = 5 # Standard trail
                
                # Calculate the trailing stop price
                trailing_stop_price = position['entry_price'] * (Decimal(1) + (Decimal(str(max_profit)) - Decimal(str(trailing_stop_pct))) / Decimal(100))

                if position['current_price'] < trailing_stop_price:
                    logger.info(f"  📉 Ratchet Trailing Stop Hit: Price ${position['current_price']:.8f} < Trail ${trailing_stop_price:.8f}")
                    logger.info(f"     (Max Profit: {max_profit:.2f}%, Trail: {trailing_stop_pct}%)")
                    return True, "trailing_stop"
            
            # 5. Volatility exit (sudden price movement)
            if 'last_price' in position:
                price_change = abs((position['current_price'] - position['last_price']) / position['last_price']) * 100
                if price_change > 20:  # 20% sudden move
                    logger.info(f"  ⚡ High volatility exit: {price_change:.2f}% price change")
                    return True, "high_volatility"
            
            position['last_price'] = position.get('current_price', position['entry_price'])
            
            # 6. ML-based exit signal (if available)
            if self.config.get('position_management', {}).get('use_ml_exits', False):
                exit_signal = await self._check_ml_exit_signal(position)
                if exit_signal['should_exit']:
                    return True, f"ml_signal_{exit_signal['reason']}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            # On error, hold position
            return False, ""
        
    # ============================================================================
    # FIX 3: engine.py - Fix _close_position method (around line 880)
    # ============================================================================

    async def _close_position(self, position: Dict, reason: str):
        """Close a trading position and add to cooldown"""
        # ✅ Get token_address from position parameter FIRST
        token_address = position.get('token_address')
        token_symbol = position.get('token_symbol', 'UNKNOWN')
        
        if not token_address:
            logger.error(f"❌ Cannot close position - missing token_address")
            return False
        
        # ✅ Verify position exists in active_positions
        # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
        async with self.positions_lock:
            if token_address not in self.active_positions:
                logger.error(f"❌ Cannot close position - not in active positions: {token_symbol}")
                return False
        
        try:
            logger.info(f"💰 CLOSING POSITION: {token_symbol} ({token_address[:10]}...)")
            logger.info(f"   Reason: {reason}")

            chain = (position.get('chain') or 'ethereum').lower()
            module_dry_run = bool(self.config.get('dry_run', True))
            position_is_real = not (module_dry_run or position.get('is_dry_run', False))
            if position_is_real and self._effective_dry_run():
                # Kill-switch / pause while LIVE: never fake-close a REAL
                # position — a DB-only close would desync the row from actual
                # on-chain holdings. The position stays open and the monitor
                # retries once the gate clears.
                logger.error(
                    f"🛑 LIVE close of {token_symbol} BLOCKED by kill-switch/pause — "
                    f"refusing simulated close of a REAL position (will retry)"
                )
                return False

            is_dry_run = module_dry_run or position.get('is_dry_run', False)

            if is_dry_run:
                # Calculate P&L
                current_price = position.get('current_price', position['entry_price'])
                entry_price = position['entry_price']
                amount = position['amount']
                
                final_pnl = (current_price - entry_price) * amount
                pnl_percentage = float((current_price - entry_price) / entry_price * 100)
                holding_time = (datetime.now(timezone.utc) - _as_utc(position['entry_time'])).total_seconds() / 60
                
                logger.info(f"📝 DRY RUN - CLOSING POSITION:")
                logger.info(f"   Token: {token_symbol}")
                logger.info(f"   Entry: ${entry_price:.8f}")
                logger.info(f"   Exit: ${current_price:.8f}")
                logger.info(f"   P&L: ${final_pnl:.2f} ({pnl_percentage:+.2f}%)")
                logger.info(f"   Holding Time: {holding_time:.1f} minutes")
                logger.info(f"   Reason: {reason}")
                
                # Update stats
                self.stats['total_profit'] += float(final_pnl)
                if final_pnl > 0:
                    self.stats['successful_trades'] += 1
                else:
                    self.stats['failed_trades'] += 1
                
                # ✅ UPDATE DATABASE - FIXED VERSION
                try:
                    trade_id = position.get('trade_id')
                    
                    if not trade_id:
                        query = """
                        SELECT id FROM trades 
                        WHERE token_address = $1 
                        AND status = 'open'
                        ORDER BY entry_timestamp DESC 
                        LIMIT 1
                        """
                        trade_id = await self.db.pool.fetchval(query, token_address)

                    if trade_id:
                        updated_metadata = _jsonable_metadata({
                            **position.get('metadata', {}),
                            'close_reason': reason,
                            'holding_time_minutes': holding_time,
                            'close_details': {
                                'entry_price': float(entry_price),
                                'exit_price': float(current_price),
                                'amount': float(amount),
                                'final_pnl': float(final_pnl),
                                'pnl_percentage': float(pnl_percentage)
                            }
                        })
                        
                        await self.db.update_trade(trade_id, {
                            'exit_price': float(current_price),
                            'exit_timestamp': datetime.now(),
                            'profit_loss': float(final_pnl),
                            'profit_loss_percentage': float(pnl_percentage),
                            'status': 'closed',
                            'metadata': updated_metadata
                        })
                        
                        logger.info(f"✅ Trade {trade_id} closed in database")

                        # 🆕 PATCH: Structured trade exit logging
                        try:
                            chain = position.get('chain', 'unknown')
                            log_trade_exit(
                                chain=chain,
                                symbol=token_symbol,
                                trade_id=str(trade_id),
                                entry_price=float(entry_price),
                                exit_price=float(current_price),
                                profit_loss=float(final_pnl),
                                pnl_pct=float(pnl_percentage),
                                reason=reason,
                                hold_time_minutes=int(holding_time)
                            )
                        except Exception as log_err:
                            logger.warning(f"Failed to log trade exit: {log_err}")

                    else:
                        logger.warning(f"⚠️  Could not find open trade_id for {token_symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to update trade in database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                # Feature-store outcome backfill (best-effort; never blocks close).
                # Threads back to the row written by AIStrategy._extract_features.
                try:
                    ai_strategy = self.strategy_manager.strategies.get('ai') if hasattr(self, 'strategy_manager') else None
                    if ai_strategy is not None and hasattr(ai_strategy, 'get_last_feature_row_id'):
                        feature_row_id = ai_strategy.get_last_feature_row_id(
                            token_address, entry_time=position.get('entry_time')
                        )
                        if feature_row_id is not None:
                            from ml.feature_store import update_outcome
                            await update_outcome(
                                self.db.pool,
                                row_id=feature_row_id,
                                outcome={
                                    'pnl_pct': float(pnl_percentage),
                                    'won': bool(float(final_pnl) > 0),
                                    'exit_reason': reason,
                                    'token_symbol': token_symbol,
                                },
                            )
                            ai_strategy.clear_last_feature_row_id(
                                token_address, entry_time=position.get('entry_time')
                            )
                except Exception as e:
                    logger.debug(f"feature-store outcome backfill failed (non-fatal): {e}")

                # ✅ Update circuit breaker metrics ONCE at the end
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': float(final_pnl),
                    'slippage_bps': 0
                })
                
                # ✅ ADD TO COOLDOWN TRACKING
                self.recently_closed[token_address] = ClosedPositionRecord(
                    token_address=token_address,
                    closed_at=datetime.now(),
                    reason=reason,
                    pnl=float(final_pnl)
                )
                logger.info(f"🕐❄️ {token_symbol} added to cooldown for {self.cooldown_minutes} minutes")
                
                logger.info(f"📊 Total profit so far: ${self.stats.get('total_profit', 0):.2f}")

                # Remove from active positions
                # CRITICAL FIX (P1): Protect with lock to prevent race conditions
                async with self.positions_lock:
                    if token_address in self.active_positions:
                        del self.active_positions[token_address]

                # ✅ NEW: Update portfolio manager
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        # Find position by token address in portfolio manager
                        pm_position = self.portfolio_manager.get_position(token_address=token_address)
                        if pm_position:
                            result = await self.portfolio_manager.close_position(pm_position.id)
                            if result.get('success'):
                                logger.info(
                                    f"✅ Portfolio manager updated: "
                                    f"P&L ${result.get('pnl', 0):.2f}"
                                )
                            else:
                                logger.warning(f"⚠️ Portfolio manager close failed: {result.get('error')}")
                        else:
                            logger.warning(f"⚠️ Position not found in portfolio manager: {token_address}")
                    except Exception as e:
                        logger.error(f"Error updating portfolio manager on close: {e}")


                # Send alert
                emoji = "💰" if final_pnl > 0 else "💸"
                await self.alert_manager.send_trade_alert(
                    f"{emoji} DRY RUN - Position Closed: {token_symbol}\n"
                    f"Entry: ${entry_price:.8f}\n"
                    f"Exit: ${current_price:.8f}\n"
                    f"P&L: ${final_pnl:.2f} ({pnl_percentage:+.2f}%)\n"
                    f"Holding Time: {holding_time:.1f}min\n"
                    f"Reason: {reason}\n"
                    f"Cooldown: {self.cooldown_minutes}min"
                )
                
                logger.info(f"✅ DRY RUN position closed and added to cooldown")
                return True
            
            # REAL EXECUTION (EVM only — the executor confirms the receipt
            # internally and returns success only when receipt.status == 1,
            # so no second confirmation wait is needed here).
            from trading.executors.base_executor import TradeOrder

            if chain == 'solana':
                logger.error(
                    f"❌ LIVE close for solana position {token_symbol} is not "
                    f"supported by the DEX module (EVM executor only) — position kept open"
                )
                return False

            # Sell leg needs RAW token units: the executor consumes
            # order.token_amount as on-chain integer units (no decimals
            # scaling). A human-unit value would sell dust. Fail-closed.
            try:
                from core.units import to_raw_evm
                raw_token_amount = await to_raw_evm(chain, token_address, position['amount'])
            except Exception as e:
                logger.error(
                    f"❌ Cannot resolve token decimals for LIVE SELL of "
                    f"{token_symbol} — aborting close (will retry): {e}"
                )
                return False
            if raw_token_amount <= 0:
                logger.error(f"❌ LIVE SELL of {token_symbol}: zero raw amount — aborting close")
                return False

            order = TradeOrder(
                token_address=self._to_checksum(token_address),
                side='sell',
                amount=float(position['amount']),
                token_amount=raw_token_amount,
                slippage=self._live_slippage('live_exit_slippage_pct'),
                deadline=300,
                gas_price_multiplier=1.5 if 'rug' in reason else 1.2,
                use_mev_protection=True,
                urgency='high' if reason in ['stop_loss', 'rug_pull_detected'] else 'normal',
                metadata={
                    'position_id': position.get('id'),
                    'token_symbol': token_symbol,
                    'exit_reason': reason
                }
            )

            result = await self.trade_executor.execute(order)

            # Executor-level kill-switch/pause race: a simulated fill must
            # never be recorded as a real on-chain close.
            if result.success and (result.metadata or {}).get('dry_run'):
                logger.error(
                    f"🛑 LIVE close of {token_symbol} was SIMULATED by the executor "
                    f"(kill-switch/pause) — keeping position open (will retry)"
                )
                return False

            if result.success:
                # All-float PnL math: execution_price is a float while the
                # position carries Decimals (float-Decimal ops raise).
                exit_price = float(result.execution_price)
                entry_price_f = float(position['entry_price'])
                amount_f = float(position['amount'])
                final_pnl = (exit_price - entry_price_f) * amount_f
                pnl_percentage = ((exit_price - entry_price_f) / entry_price_f) * 100 if entry_price_f else 0.0

                self.stats['total_profit'] += final_pnl
                if final_pnl > 0:
                    self.stats['successful_trades'] += 1
                else:
                    self.stats['failed_trades'] += 1
                
                # ✅ Update circuit breaker metrics for real trade
                actual_slippage = getattr(result, 'slippage_bps', 0)
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': float(final_pnl),
                    'slippage_bps': actual_slippage
                })
                
                # Add to cooldown
                self.recently_closed[token_address] = ClosedPositionRecord(
                    token_address=token_address,
                    closed_at=datetime.now(),
                    reason=reason,
                    pnl=float(final_pnl)
                )

                # Backfill DB + ML outcome on real-execution close (mirrors dry-run branch)
                holding_time = (datetime.now(timezone.utc) - _as_utc(position['entry_time'])).total_seconds() / 60
                try:
                    trade_id = position.get('trade_id')
                    if not trade_id:
                        query = """
                        SELECT id FROM trades
                        WHERE token_address = $1
                        AND status = 'open'
                        ORDER BY entry_timestamp DESC
                        LIMIT 1
                        """
                        trade_id = await self.db.pool.fetchval(query, token_address)

                    if trade_id:
                        updated_metadata = _jsonable_metadata({
                            **position.get('metadata', {}),
                            'close_reason': reason,
                            'holding_time_minutes': holding_time,
                            'close_details': {
                                'entry_price': float(position['entry_price']),
                                'exit_price': float(exit_price),
                                'amount': float(position['amount']),
                                'final_pnl': float(final_pnl),
                                'pnl_percentage': float(pnl_percentage)
                            }
                        })
                        await self.db.update_trade(trade_id, {
                            'exit_price': float(exit_price),
                            'exit_timestamp': datetime.now(),
                            'profit_loss': float(final_pnl),
                            'profit_loss_percentage': float(pnl_percentage),
                            'status': 'closed',
                            'metadata': updated_metadata
                        })
                        logger.info(f"✅ Trade {trade_id} closed in database")

                        try:
                            log_trade_exit(
                                chain=position.get('chain', 'unknown'),
                                symbol=token_symbol,
                                trade_id=str(trade_id),
                                entry_price=float(position['entry_price']),
                                exit_price=float(exit_price),
                                profit_loss=float(final_pnl),
                                pnl_pct=float(pnl_percentage),
                                reason=reason,
                                hold_time_minutes=int(holding_time)
                            )
                        except Exception as log_err:
                            logger.warning(f"Failed to log trade exit: {log_err}")
                    else:
                        logger.warning(f"⚠️  Could not find open trade_id for {token_symbol}")
                except Exception as e:
                    logger.error(f"❌ Failed to update trade in database: {e}")

                try:
                    ai_strategy = self.strategy_manager.strategies.get('ai') if hasattr(self, 'strategy_manager') else None
                    if ai_strategy is not None and hasattr(ai_strategy, 'get_last_feature_row_id'):
                        feature_row_id = ai_strategy.get_last_feature_row_id(
                            token_address, entry_time=position.get('entry_time')
                        )
                        if feature_row_id is not None:
                            from ml.feature_store import update_outcome
                            await update_outcome(
                                self.db.pool,
                                row_id=feature_row_id,
                                outcome={
                                    'pnl_pct': float(pnl_percentage),
                                    'won': bool(float(final_pnl) > 0),
                                    'exit_reason': reason,
                                    'token_symbol': token_symbol,
                                },
                            )
                            ai_strategy.clear_last_feature_row_id(
                                token_address, entry_time=position.get('entry_time')
                            )
                except Exception as e:
                    logger.debug(f"feature-store outcome backfill failed (non-fatal): {e}")

                if hasattr(self, 'position_tracker') and position.get('tracker_id'):
                    await self.position_tracker.close_position(position['tracker_id'])

                # CRITICAL FIX (P1): Protect with lock to prevent race conditions
                async with self.positions_lock:
                    del self.active_positions[token_address]

                emoji = "💰" if final_pnl > 0 else "💸"
                await self.alert_manager.send_trade_alert(
                    f"{emoji} Position Closed: {token_symbol}\n"
                    f"Entry: ${position['entry_price']:.8f}\n"
                    f"Exit: ${exit_price:.8f}\n"
                    f"P&L: ${final_pnl:.2f} ({pnl_percentage:.2f}%)\n"
                    f"Reason: {reason}\n"
                    f"Tx: {(result.tx_hash or 'N/A')[:10]}..."
                )

                logger.info(f"✅ Successfully closed position: {token_symbol}")
                return True
            else:
                # Surface repeated live-close failures loudly: the position
                # stays in active_positions so the monitor retries, but an
                # unsellable token (honeypot / drained pool) must not fail
                # silently forever.
                failures = position.get('_close_failures', 0) + 1
                position['_close_failures'] = failures
                logger.error(
                    f"❌ Failed to close position ({failures} attempt(s)): {result.error}"
                )
                if failures == 5:
                    await self.alert_manager.send_critical(
                        f"🚨 LIVE close FAILING repeatedly: {token_symbol}\n"
                        f"Attempts: {failures}\n"
                        f"Last error: {result.error}\n"
                        f"Possible honeypot / drained pool — manual action needed."
                    )

                # Update circuit breaker for failed trade
                self.risk_manager.update_trade_metrics({
                    'success': False,
                    'profit': 0,
                    'slippage_bps': 0
                })
                
                await self.alert_manager.send_warning(
                    f"⚠️ Failed to close {token_symbol}\n"
                    f"Error: {result.error}\n"
                    f"Will retry..."
                )
                return False
                
                
        except Exception as e:
            logger.error(f"Error closing position {token_symbol}: {e}", exc_info=True)
            await self.alert_manager.send_critical(f"Critical error closing position: {e}")
            return False

    async def _monitor_mempool(self):
        """Monitor mempool for relevant transactions"""
        while self.state == BotState.RUNNING:
            try:
                # Get pending transactions
                pending_txs = await self.mempool_monitor.get_pending_transactions()

                # CRITICAL FIX (P1): Take snapshot of positions to avoid race conditions
                async with self.positions_lock:
                    active_tokens = set(self.active_positions.keys())

                for tx in pending_txs:
                    # Check if it affects our positions
                    if tx['to'] in active_tokens:
                        await self._analyze_mempool_tx(tx)
                        
                    # Detect sandwich attacks
                    if await self.mempool_monitor.detect_sandwich_attack(tx):
                        await self.event_bus.emit(Event(
                            event_type=EventType.SANDWICH_DETECTED,
                            data=tx
                        ))
                        
                await asyncio.sleep(0.1)  # Fast mempool monitoring
                
            except Exception as e:
                await asyncio.sleep(1)
                
    async def _track_whales(self):
        """Track whale wallet movements"""
        while self.state == BotState.RUNNING:
            try:
                # Get whale movements
                movements = await self.whale_tracker.get_recent_movements()

                # CRITICAL FIX (P1): Take snapshot of positions to avoid race conditions
                async with self.positions_lock:
                    active_tokens = set(self.active_positions.keys())

                for movement in movements:
                    # Check if it affects our tokens
                    if movement['token'] in active_tokens:
                        await self._handle_whale_movement(movement)
                        
                    # Check for whale accumulation patterns
                    if movement['type'] == 'accumulation':
                        await self._analyze_whale_accumulation(movement)
                        
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                await asyncio.sleep(10)
                
    # ============================================================================
    # FIX #1: core/engine.py (Line ~878)
    # Replace the _optimize_strategies method
    # ============================================================================

    async def _optimize_strategies(self):
        """Continuously optimize trading strategies"""
        while self.state == BotState.RUNNING:
            try:
                # Wait for enough data
                await asyncio.sleep(3600)  # Optimize every hour
                
                # FIX: Use the correct method name from PerformanceTracker
                # The actual method is get_performance_report(), not get_recent_performance()
                try:
                    performance_data = self.performance_tracker.get_performance_report(period="daily")
                except AttributeError as e:
                    logger.warning(f"Performance tracking method not available: {e}")
                    continue  # Skip this optimization cycle
                except Exception as e:
                    logger.error(f"Error getting performance data: {e}")
                    continue
                
                # Check if we have valid performance data
                if not performance_data or 'error' in performance_data:
                    logger.info("No sufficient performance data for optimization")
                    continue
                
                # Only optimize if we have enough trades
                if performance_data.get('summary', {}).get('total_trades', 0) < 10:
                    logger.info("Not enough trades for meaningful optimization (need 10+)")
                    continue
                
                # Run hyperparameter optimization
                try:
                    new_params = await self.hyperparam_optimizer.optimize(
                        performance_data,
                        current_params=self.strategy_manager.get_parameters()
                    )
                    
                    # Validate new parameters
                    if await self._validate_new_parameters(new_params):
                        await self.strategy_manager.update_parameters(new_params)
                        logger.info("Strategy parameters optimized successfully")
                        await self.alert_manager.send_info("Strategy parameters optimized")
                except Exception as e:
                    logger.error(f"Hyperparameter optimization failed: {e}")
                
                # RL optimization
                try:
                    await self.rl_optimizer.update_policy(performance_data)
                except Exception as e:
                    logger.error(f"RL optimization failed: {e}")
                
            except asyncio.CancelledError:
                logger.info("Strategy optimization loop cancelled")
                break
            except Exception as e:
                logger.error(f"Strategy optimization error: {e}", exc_info=True)
                # Don't send Telegram alert here - it's too spammy
                # The error is already logged
                
    async def _retrain_models(self):
        """Periodically retrain ML models via scripts/train_ensemble.py.

        Wave-13: the previous loop called _collect_training_data() (returns {})
        and _should_retrain() (returns False) — so it NEVER retrained. This
        replacement invokes the canonical offline trainer as a subprocess so:
        - the training pipeline is tested end-to-end independently of the engine
        - artifacts land in models/ where load_models() reads them
        - only the DB version row is written (no trading-state mutation)
        - the engine reloads the new models without a full restart

        Config keys (all under ml_models in config_settings):
          ml_retrain_enabled       bool  default True   — master toggle
          ml_retrain_interval_hours int  default 24     — hours between runs
          ml_retrain_min_trades    int  default 50      — skip if fewer closed trades
          ml_retrain_days          int  default 90      — training window
        """
        import subprocess
        from pathlib import Path

        while self.state == BotState.RUNNING:
            try:
                ml_cfg = self.config.get('ml_models', {})
                interval_h = int(ml_cfg.get('ml_retrain_interval_hours', 24))
                await asyncio.sleep(interval_h * 3600)

                if not ml_cfg.get('ml_retrain_enabled', True):
                    logger.info("Auto-retrain disabled via ml_models.ml_retrain_enabled")
                    continue

                days = int(ml_cfg.get('ml_retrain_days', 90))
                min_trades = int(ml_cfg.get('ml_retrain_min_trades', 50))

                logger.info(
                    f"Auto-retrain: launching scripts/train_ensemble.py "
                    f"--days {days} (min_trades={min_trades})"
                )

                repo_root = str(Path(__file__).parent.parent)
                script = str(Path(repo_root) / "scripts" / "train_ensemble.py")

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["python", script, "--days", str(days)],
                        capture_output=True, text=True,
                        cwd=repo_root, timeout=1800,  # 30-min hard cap
                    )
                )

                if result.returncode == 0:
                    logger.info("Auto-retrain: SUCCEEDED — reloading model artifacts")
                    # Reload the freshly trained models without a full restart.
                    try:
                        await self.ensemble_predictor.load_models()
                        logger.info("Auto-retrain: model artifacts reloaded into ensemble_predictor")
                    except Exception as reload_err:
                        logger.warning(f"Auto-retrain: reload failed (non-fatal): {reload_err}")
                    await self.alert_manager.send_info(
                        f"DEX ensemble retrained ({days}d window). "
                        "ML[ensemble] now active."
                    )
                elif result.returncode == 3:
                    logger.info(
                        f"Auto-retrain: skipped — fewer than {min_trades} closed "
                        "trades in window (normal early-run behaviour)"
                    )
                else:
                    logger.warning(
                        f"Auto-retrain: exit {result.returncode}\n"
                        f"stdout: {result.stdout[-2000:]}\n"
                        f"stderr: {result.stderr[-1000:]}"
                    )

            except Exception as e:
                logger.error(f"Auto-retrain loop error: {e}", exc_info=True)
                await self.alert_manager.send_warning(f"Model retraining error: {e}")
                
    async def _update_blacklists(self):
        """Update blacklists based on learned patterns"""
        while self.state == BotState.RUNNING:
            try:
                await asyncio.sleep(1800)  # Update every 30 minutes
                
                # Get recent rug pulls and scams
                recent_rugs = await self._get_recent_rug_pulls()
                
                for rug in recent_rugs:
                    # Add token to blacklist
                    self.blacklisted_tokens.add(rug['token_address'])
                    
                    # Add developer to blacklist
                    if rug.get('developer_address'):
                        self.blacklisted_devs.add(rug['developer_address'])
                        
                # Save updated blacklists
                await self._save_blacklists()
                
                # Also fetch community blacklists
                await self._update_community_blacklists()
                
            except Exception as e:
                await asyncio.sleep(1800)
                
    async def _monitor_performance(self):
        """Monitor and report performance metrics"""
        while self.state == BotState.RUNNING:
            try:
                # CRITICAL FIX (P1): Protect read with lock to prevent race conditions
                async with self.positions_lock:
                    num_active_positions = len(self.active_positions)

                # Calculate metrics
                metrics = {
                    'total_trades': self.stats['total_trades'],
                    'win_rate': min(1.0, self.stats['successful_trades'] / max(self.stats['total_trades'], 1)),
                    'total_pnl': self.stats['total_profit'],
                    'active_positions': num_active_positions,
                    'opportunities_found': self.stats['opportunities_found'],
                    'tokens_analyzed': self.stats['tokens_analyzed'],
                    'uptime': (datetime.now() - self.stats['start_time']).total_seconds()
                }
                
                # Track performance
                await self.performance_tracker.record_metrics(metrics)
                
                logger.info(f"📊 Blacklist size: {len(self.blacklisted_tokens)} tokens")
                # Send daily report
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    await self._send_daily_report()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await asyncio.sleep(60)
                
    async def _health_check(self):
        """Perform system health checks"""
        while self.state == BotState.RUNNING:
            try:
                checks = {
                    'database': await self._check_database_health(),
                    'web3': await self._check_web3_health(),
                    'apis': await self._check_api_health(),
                    'memory': self._check_memory_usage(),
                    'cpu': self._check_cpu_usage()
                }
                
                # Alert if any issues
                for component, status in checks.items():
                    if not status['healthy']:
                        await self.alert_manager.send_warning(
                            f"Health check failed for {component}: {status['message']}"
                        )
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                await asyncio.sleep(300)

    # Add these methods to existing TradingBotEngine class
    
    async def start(self):
        """Start the trading engine"""
        try:
            await self.initialize()
            self.state = BotState.RUNNING
            await self.run()
        except Exception as e:
            self.state = BotState.ERROR
            raise Exception(f"Failed to start engine: {e}")
    
    # ============================================
    # ALSO UPDATE: Modify stop() method to call cleanup()
    # Find the stop() method (around line 950) and update it:
    # ============================================

    async def stop(self):
        """Stop the trading engine"""
        try:
            logger.info("🛑 Stopping trading engine...")
            self.state = BotState.STOPPING
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            logger.info("✅ All tasks cancelled")
            
            # Close positions if configured
            if self.config.get('close_on_stop', False):
                logger.info("💰 Closing all positions...")
                # CRITICAL FIX (P1): Take snapshot under lock to prevent race conditions
                async with self.positions_lock:
                    positions_to_close = list(self.active_positions.values())

                for position in positions_to_close:
                    await self._close_position(position, "engine_stopped")
            
            # Save state before cleanup
            await self._save_state()
            logger.info("✅ State saved")
            
            # ⭐ NEW: Call cleanup method
            await self.cleanup()
            
            self.state = BotState.STOPPED
            logger.info("✅ Trading engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping engine: {e}", exc_info=True)
            await self.alert_manager.send_critical(f"Error stopping engine: {e}")
            raise


    # ============================================
    # PATCH: Add cleanup() method to TradingBotEngine class
    # Add this method after the stop() method in engine.py (around line 1000)
    # ============================================

    async def cleanup(self):
        """
        Cleanup all resources and connections
        Called during shutdown or after stop()
        """
        try:
            logger.info("🧹 Starting engine cleanup...")

            # Stop event bus processing
            try:
                await self.event_bus.stop()
                logger.info("✅ Event bus stopped")
            except Exception as e:
                logger.error(f"Error stopping event bus: {e}")

            # 1. Cleanup Solana executor
            if self.solana_executor:
                try:
                    await self.solana_executor.cleanup()
                    logger.info("✅ Solana executor cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up Solana executor: {e}")
            
            # 2. Cleanup EVM executor
            if hasattr(self.trade_executor, 'cleanup'):
                try:
                    await self.trade_executor.cleanup()
                    logger.info("✅ Trade executor cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up trade executor: {e}")
            
            # 3. Cleanup data collectors
            collectors = [
                ('DexScreener', self.dex_collector),
                ('Chain Data', self.chain_collector),
                ('Social Data', self.social_collector),
                ('Mempool Monitor', self.mempool_monitor),
                ('Whale Tracker', self.whale_tracker),
                ('Honeypot Checker', self.honeypot_checker),
            ]
            
            for name, collector in collectors:
                cleanup_method = None
                if hasattr(collector, 'cleanup'):
                    cleanup_method = collector.cleanup
                elif hasattr(collector, 'close'):
                    cleanup_method = collector.close

                if cleanup_method:
                    try:
                        await cleanup_method()
                        logger.info(f"✅ {name} collector cleaned up")
                    except Exception as e:
                        logger.debug(f"Error cleaning up {name} collector: {e}")
            
            # 4. Cleanup database connection
            if hasattr(self.db, 'disconnect'):
                try:
                    await self.db.disconnect()
                    logger.info("✅ Database disconnected")
                except Exception as e:
                    logger.error(f"Error disconnecting database: {e}")
            
            # 5. Cleanup alert manager
            if hasattr(self.alert_manager, 'cleanup'):
                try:
                    await self.alert_manager.cleanup()
                    logger.info("✅ Alert manager cleaned up")
                except Exception as e:
                    logger.debug(f"Error cleaning up alert manager: {e}")
            
            # 6. Save final state
            try:
                await self._save_state()
                logger.info("✅ Final state saved")
            except Exception as e:
                logger.error(f"Error saving final state: {e}")
            
            logger.info("✅ Engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    
    async def _final_safety_checks(self, opportunity: TradingOpportunity) -> bool:
        """
        Perform final safety checks before executing trade
        
        Args:
            opportunity: Trading opportunity to check
            
        Returns:
            True if all checks pass, False otherwise
        """
        try:
            # Get symbol from metadata
            token_symbol = opportunity.metadata.get('token_symbol', 'UNKNOWN')
            
            logger.info(f"🔍 Starting safety checks for {token_symbol} ({opportunity.token_address[:10]}...)")
            
            # 1. Check honeypot status
            logger.info(f"   Checking honeypot status...")
            
            # ✅ FIX: Assign result to variable
            honeypot_result = await self.honeypot_checker.check_token(
                opportunity.token_address,
                opportunity.chain
            )
            
            # ✅ NOW this works:
            if honeypot_result.get('is_honeypot', False):
                logger.warning(f"   ❌ HONEYPOT DETECTED: {token_symbol}")
                
                # ✅ ADD TO BLACKLIST
                self.blacklisted_tokens.add(opportunity.token_address.lower())
                await self._save_blacklists()
                
                logger.info(f"   🚫 Added {token_symbol} ({opportunity.token_address}) to blacklist")
                
                return False
                
            logger.info(f"   ✅ Not a honeypot")
            
            # 2. Verify liquidity is still sufficient
            logger.info(f"   Checking liquidity...")
            current_liquidity = opportunity.liquidity
            chain_name = opportunity.chain.lower() if hasattr(opportunity, 'chain') else 'ethereum'
            chain_config = self.config.get('chain', {})
            min_liquidity = chain_config.get(f'{chain_name}_min_liquidity', 10000)
            logger.info(f"   Current liquidity: ${current_liquidity:,.2f}, Min required: ${min_liquidity:,.2f}")
            
            if current_liquidity < min_liquidity:
                logger.warning(f"   ❌ INSUFFICIENT LIQUIDITY: ${current_liquidity:,.2f} < ${min_liquidity:,.2f}")
                return False
            logger.info(f"   ✅ Liquidity sufficient")
            
            # 3. Check if token is blacklisted
            logger.info(f"   Checking blacklist...")
            if self._is_blacklisted({'token_address': opportunity.token_address}):
                logger.warning(f"   ❌ TOKEN BLACKLISTED: {token_symbol}")
                return False
            logger.info(f"   ✅ Not blacklisted")
            
            # 4. Verify rug pull probability is acceptable
            logger.info(f"   Checking rug probability...")
            max_rug_prob = 0.5  # 50% max
            logger.info(f"   Rug probability: {opportunity.rug_probability:.2%}, Max allowed: {max_rug_prob:.2%}")
            
            if opportunity.rug_probability > max_rug_prob:
                logger.warning(f"   ❌ HIGH RUG RISK: {opportunity.rug_probability:.2%} > {max_rug_prob:.2%}")
                return False
            logger.info(f"   ✅ Rug risk acceptable")
            
            # 5. Check recent price action isn't too volatile
            logger.info(f"   Checking volatility...")
            price_change = opportunity.metadata.get('pair', {}).get('price_change_5m', 0)
            if price_change and abs(price_change) > 50:  # 50% move in 5min
                logger.warning(f"   ❌ EXCESSIVE VOLATILITY: {price_change:+.1f}% in 5min")
                return False
            logger.info(f"   ✅ Volatility acceptable")
            
            # 6. Contract-verification gate (Wave-F6, adjudication #15).
            # Wave-9 made this honest (absent result != verified) but left it
            # WARN-only, claiming the binding gate was RiskScore.contract_risk
            # in the scorer. Verified this wave: that is only a WEIGHTED score
            # component, not a hard gate — an unverifiable contract could still
            # enter on volume/liquidity alone. Per the F5 principle (an
            # unverifiable safety signal must not pass), the gate fails CLOSED
            # in LIVE mode. PM adjudication (Wave-F6 final gate): the gate is
            # DRY_RUN-AWARE — the engine-level verifier is an always-unverified
            # stub, so a mode-blind hard gate would block 100% of DRY_RUN DEX
            # entries and end the operator's data collection. Semantics:
            #   * DRY_RUN (module flag / kill-switch / pause): WARN-only —
            #     entries proceed so paper data keeps flowing; every miss is
            #     logged for the future real-verifier rollout.
            #   * LIVE: fail-closed — unverifiable contract REJECTED.
            #   * DB knob `trading.block_unverified_contracts` (mig 147,
            #     default true) stays the operator override: 'false' restores
            #     warn-only in BOTH modes (including LIVE — explicit opt-out).
            # A positive signal from EITHER source passes: the engine-level
            # contract_safety metadata (currently the honest stub) or
            # RiskScore.verified_contract (real chain-collector signal).
            logger.info(f"   Checking contract verification...")
            contract_safety = opportunity.metadata.get('contract_safety', {}) or {}
            rs = getattr(opportunity, 'risk_score', None)
            is_verified = bool(
                contract_safety.get('verified', False)
                or getattr(rs, 'verified_contract', False)
            )
            if not is_verified:
                block_unverified = str(
                    (self.config.get('trading', {}) or {}).get(
                        'block_unverified_contracts', True
                    )
                ).strip().lower() not in ('false', '0', 'no', 'off')
                if block_unverified and not self._effective_dry_run():
                    logger.warning(
                        f"   ❌ CONTRACT NOT VERIFIED: {token_symbol} — REJECTING "
                        f"(LIVE fail-closed; set trading.block_unverified_contracts"
                        f"=false to restore warn-only)"
                    )
                    return False
                logger.warning(
                    f"   ⚠️  Contract not verified — proceeding "
                    f"({'DRY_RUN warn-only (LIVE would reject)' if block_unverified else 'trading.block_unverified_contracts=false'})"
                )
            
            logger.info(f"✅ All safety checks PASSED for {token_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in safety checks for {opportunity.token_address}: {e}", exc_info=True)
            return False  # Fail safe - reject on error

    # Add these methods to TradingBotEngine class in engine.py

    async def _load_blacklists(self):
        """Load token and developer blacklists"""
        try:
            # Try to load from file
            import os
            blacklist_file = 'data/blacklists.json'
            
            if os.path.exists(blacklist_file):
                with open(blacklist_file, 'r') as f:
                    data = json.load(f)
                    self.blacklisted_tokens = set(data.get('tokens', []))
                    self.blacklisted_devs = set(data.get('developers', []))
            else:
                # Initialize empty blacklists
                self.blacklisted_tokens = set()
                self.blacklisted_devs = set()
                
        except Exception as e:
            # If loading fails, start with empty blacklists
            self.blacklisted_tokens = set()
            self.blacklisted_devs = set()

    # EVM/Solana DEX chains this engine trades. Mirrors
    # modules/dex_trading/position_service._DEX_CHAINS so _load_state restores
    # exactly the rows position_service manages (no orphaned / cross-module rows).
    _DEX_STATE_CHAINS = (
        'ethereum', 'bsc', 'polygon', 'arbitrum', 'base',
        'optimism', 'avalanche', 'solana',
    )

    async def _load_state(self):
        """Restore OPEN positions from the DB into self.active_positions
        (Wave-8 DEFECT 3).

        Previously a no-op, so after a subprocess restart self.active_positions
        was empty: in-engine exit logic, the rug-probability exit gate and
        get_stats() all behaved as if there were no open positions, and
        DexPositionService (DB-first) was masking only the PRICE-refresh symptom.

        Coexistence with modules/dex_trading/position_service.py:
        - We restore the SAME rows position_service reads
          (trades WHERE status='open' AND side='buy' AND chain IN dex_chains).
        - Each restored position carries `trade_id` = the INTEGER trades.id, so
          the engine's close path (db.update_trade(trade_id, ...), which matches
          the `id` column for ints) writes to the EXACT row position_service
          manages — no duplicate/orphan rows, last-writer-wins on metadata which
          converges since both write the same fresh quote.
        - Restored positions are tagged metadata.restored_from_db=True for audit
          and to distinguish them from positions opened in-process this session.

        Fail-soft: any error logs and leaves active_positions as-is — a bad load
        must never crash startup.
        """
        try:
            from decimal import Decimal as _Dec

            pool = getattr(getattr(self, 'db', None), 'pool', None)
            if pool is None:
                logger.info("   _load_state: no DB pool — starting with empty active_positions")
                return

            # Wave-26: restored rows without persisted SL/TP fall back to the
            # DB-configured values (was hardcoded 0.1/0.3).
            sl_pct, tp_pct = self._position_sl_tp()

            chains = ', '.join(f"'{c}'" for c in self._DEX_STATE_CHAINS)
            query = f"""
                SELECT id, trade_id, token_address, chain, entry_price, amount,
                       usd_value, entry_timestamp, metadata
                FROM trades
                WHERE status = 'open' AND side = 'buy'
                  AND chain IN ({chains})
                ORDER BY entry_timestamp DESC
                LIMIT 200
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)

            restored = 0
            async with self.positions_lock:
                for row in rows:
                    try:
                        token_address = row['token_address']
                        if not token_address:
                            continue

                        # metadata may be jsonb(dict) or text(json) depending on driver.
                        raw_meta = row['metadata']
                        if isinstance(raw_meta, dict):
                            metadata = dict(raw_meta)
                        elif raw_meta:
                            try:
                                metadata = json.loads(raw_meta)
                            except Exception:
                                metadata = {}
                        else:
                            metadata = {}

                        # Match the in-memory shape used by the live entry path
                        # (lines ~1220/1300): entry_price + amount as Decimal so
                        # the monitoring/exit arithmetic stays type-correct.
                        try:
                            entry_price = _Dec(str(row['entry_price'] or 0))
                            amount = _Dec(str(row['amount'] or 0))
                        except Exception:
                            continue

                        entry_time = row['entry_timestamp']
                        if not isinstance(entry_time, datetime):
                            entry_time = datetime.now()

                        position = {
                            'token_address': token_address,
                            'token_symbol': metadata.get('token_symbol', 'UNKNOWN'),
                            'entry_price': entry_price,
                            'amount': amount,
                            'entry_value': float(row['usd_value'] or 0),
                            'chain': (row['chain'] or 'ethereum'),
                            'strategy': {'name': metadata.get('entry_strategy', 'momentum')},
                            'entry_time': entry_time,
                            'stop_loss_percentage': metadata.get('stop_loss_percentage', sl_pct),
                            'take_profit_percentage': metadata.get('take_profit_percentage', tp_pct),
                            # INTEGER trades.id -> engine close writes target this exact row.
                            'trade_id': row['id'],
                            'metadata': {**metadata, 'restored_from_db': True},
                        }
                        self.active_positions[token_address] = position
                        restored += 1
                    except Exception as row_err:
                        logger.debug(f"   _load_state: skipped a row: {row_err}")
                        continue

            logger.info(
                f"   _load_state: restored {restored} open position(s) into "
                f"active_positions (DB-first refresh still owned by "
                f"DexPositionService)"
            )

            # LIVE only: verify restored REAL fills against on-chain holdings
            # (logging + metadata flag only — never auto-closes).
            if restored and not bool(self.config.get('dry_run', True)):
                if (self.config.get('trading', {}) or {}).get('live_reconcile_enabled', True):
                    asyncio.create_task(
                        self._reconcile_live_positions(),
                        name='reconcile_live_positions'
                    )
        except Exception as e:
            # Fail-soft: never block startup on a bad state load.
            logger.warning(f"   _load_state failed (non-fatal): {e}")

    # chain name -> EVM chain id (reconcile guard: the single-chain
    # TradeExecutor can only read balances on its own chain).
    _CHAIN_IDS = {
        'ethereum': 1, 'bsc': 56, 'polygon': 137, 'arbitrum': 42161,
        'base': 8453, 'optimism': 10, 'avalanche': 43114,
    }

    async def _reconcile_live_positions(self):
        """One-shot startup reconcile (LIVE mode, observability-only).

        For every restored REAL position (metadata.is_dry_run == False) on
        the executor's chain, compare the DB amount against the on-chain
        ERC20 balance. A shortfall means the row no longer reflects holdings
        (sold elsewhere / failed buy recorded open) — exits fired from it
        would revert and burn gas. Flags `metadata.reconcile_onchain_deficit`
        and alerts; NEVER auto-closes. Fail-soft per position."""
        try:
            executor = self.trade_executor
            exec_chain_id = getattr(executor, 'chain_id', None)
            async with self.positions_lock:
                snapshot = list(self.active_positions.items())
            checked = flagged = 0
            for token_address, pos in snapshot:
                try:
                    meta = pos.get('metadata') or {}
                    if meta.get('is_dry_run') is not False:
                        continue  # only REAL fills are reconcilable
                    chain = (pos.get('chain') or 'ethereum').lower()
                    if self._CHAIN_IDS.get(chain) != exec_chain_id:
                        continue
                    balance = float(await executor._get_token_balance(
                        self._to_checksum(token_address)
                    ))
                    checked += 1
                    db_amount = float(pos.get('amount') or 0)
                    if db_amount <= 0:
                        continue
                    if balance < db_amount * 0.95:  # 5% tolerance (fees/dust)
                        flagged += 1
                        symbol = pos.get('token_symbol', 'UNKNOWN')
                        logger.critical(
                            f"🚨 RECONCILE MISMATCH {symbol}: DB amount "
                            f"{db_amount:.6f} vs on-chain {balance:.6f} — row "
                            f"does not reflect holdings; live exits from it "
                            f"would fail. Manual review needed."
                        )
                        pos.setdefault('metadata', {})['reconcile_onchain_deficit'] = True
                        trade_id = pos.get('trade_id')
                        if trade_id:
                            try:
                                await self.db.update_trade(trade_id, {'metadata': {
                                    **meta,
                                    'reconcile_onchain_deficit': True,
                                    'reconcile_onchain_balance': balance,
                                    'reconcile_checked_at':
                                        datetime.now(timezone.utc).isoformat(),
                                }})
                            except Exception as db_err:
                                logger.error(f"reconcile flag persist failed: {db_err}")
                        await self.alert_manager.send_critical(
                            f"🚨 DEX reconcile mismatch: {symbol}\n"
                            f"DB amount: {db_amount:.6f}\n"
                            f"On-chain: {balance:.6f}\n"
                            f"Row flagged reconcile_onchain_deficit — review before LIVE exits."
                        )
                except Exception as pos_err:
                    logger.warning(f"reconcile skipped for {token_address}: {pos_err}")
                    continue
            logger.info(
                f"🔎 Startup reconcile: {checked} live position(s) checked, "
                f"{flagged} mismatch(es) flagged"
            )
        except Exception as e:
            logger.warning(f"startup reconcile failed (non-fatal): {e}")

    async def _warmup_collectors(self):
        """Warm up data collectors"""
        try:
            # Test connections
            await self.dex_collector.test_connection()
        except Exception:
            pass

    def _is_blacklisted(self, pair: Dict) -> bool:
        """Check if token or developer is blacklisted"""
    #    token_address = address.lower()  # Always normalize to lowercase!
        return (
            pair.get('token_address') in self.blacklisted_tokens or
            pair.get('creator_address') in self.blacklisted_devs
        )

    async def _check_developer_reputation(self, dev_address: str) -> float:
        """Developer reputation score.

        UNIMPLEMENTED no-op (Wave-9 quant audit). The REAL developer-risk
        signal is already produced by RiskManager.analyze_token() →
        RiskScore.developer_risk (the 20% risk weight in
        _calculate_opportunity_score). This parallel score is NOT consumed by
        any gate or scorer — it is only stored in opportunity.metadata for
        audit. Returns a NEUTRAL 0.5 (no optimism injected); do NOT wire this
        into a score without first replacing it with a real on-chain
        dev-history source (analysis/dev_analyzer.py).
        """
        # Neutral placeholder — feeds no live gate/score; see RiskScore.developer_risk.
        return 0.5

    async def _analyze_liquidity_depth(self, pair: Dict) -> Dict:
        """Analyze liquidity depth"""
        return {'depth': pair.get('liquidity', 0)}

    async def _check_smart_contract(self, token_address: str) -> Dict:
        """Smart-contract verification check.

        HONEST UNIMPLEMENTED stub (Wave-9 quant audit). It previously returned
        `verified=True` unconditionally — a FALSE positive safety signal that
        claimed every contract was source-verified. No real verifier is wired
        into this engine path (analysis/smart_contract_analyzer.py exists but is
        not instantiated here; the REAL contract-risk signal is
        RiskScore.contract_risk via RiskManager.analyze_token, which DOES feed
        the scorer). We now return `verified=False`/`status='unknown'` so the
        downstream gate in _final_safety_checks logs a caution instead of
        silently asserting a verification we never performed.
        """
        # NEVER assert a positive safety signal we cannot verify.
        return {'verified': False, 'status': 'unknown', 'issues': [],
                'note': 'engine-level contract check unimplemented; '
                        'real signal is RiskScore.contract_risk'}

    async def _analyze_holder_distribution(self, token_address: str) -> Dict:
        """Holder-distribution analysis.

        UNIMPLEMENTED no-op (Wave-9 quant audit). Not consumed by any gate or
        scorer — only stored in opportunity.metadata for audit. The REAL
        holder-concentration signal is RiskScore.holder_risk via
        RiskManager.analyze_token. Returns a NEUTRAL/UNKNOWN value (no optimism:
        does NOT assert `concentrated=False`, which would have been an
        unverified "looks safe" claim).
        """
        return {'concentrated': None, 'status': 'unknown',
                'note': 'engine-level holder check unimplemented; '
                        'real signal is RiskScore.holder_risk'}

    # NOTE (Wave-9 quant audit): the former `_extract_features(data)` that
    # returned `np.random.rand(10)` was DELETED. It had ZERO callers in this
    # engine — the live ML path builds features via `_build_ml_feature_dict`
    # → `EnsemblePredictor.extract_features` (a deterministic 95-dim vector),
    # never this method. Returning RANDOM features from a live path is the
    # worst possible fabrication, so the dead stub is removed outright rather
    # than left as a foot-gun for a future caller.

    def _calculate_pnl(self, position: Dict) -> float:
        """Calculate position P&L"""
        return (position.get('current_price', 0) - position['entry_price']) * position['amount']

    def _calculate_pnl_percentage(self, position: Dict) -> float:
        """Calculate position P&L percentage"""
        return ((position.get('current_price', 0) - position['entry_price']) / position['entry_price']) * 100

    async def _check_ml_exit_signal(self, position: Dict) -> Dict:
        """Check if ML model signals exit"""
        return {'should_exit': False, 'reason': ''}

    async def _detect_rug_pull_signs(self, token_address: str) -> bool:
        """Detect signs of rug pull"""
        return False

    async def _check_liquidity_crisis(self, token_address: str) -> bool:
        """Check for liquidity crisis"""
        return False

    async def _set_exit_orders(self, position: Dict, strategy: Dict):
        """Set stop loss and take profit orders"""
        pass

    async def _should_dca(self, position: Dict) -> bool:
        """Check if should do dollar cost averaging"""
        return False

    async def _execute_dca(self, position: Dict):
        """Execute DCA for position"""
        pass

    async def _update_trailing_stop(self, position: Dict, current_price: float):
        """Update trailing stop loss"""
        pass

    def _calculate_slippage(self, opportunity: TradingOpportunity) -> float:
        """Calculate appropriate slippage tolerance"""
        return 0.02  # 2% default

    def _calculate_gas_multiplier(self, opportunity: TradingOpportunity) -> float:
        """Calculate gas price multiplier"""
        return 1.2  # 20% above base

    async def _apply_mev_protection(self, order: Dict, opportunity: TradingOpportunity) -> Dict:
        """Apply MEV protection to order"""
        return order

    async def _learn_from_trade(self, position: Dict, pnl: float, reason: str):
        """Learn from completed trade"""
        pass

    async def _analyze_mempool_tx(self, tx: Dict):
        """Analyze mempool transaction"""
        pass

    async def _handle_whale_movement(self, movement: Dict):
        """Handle whale movement event"""
        pass

    async def _analyze_whale_accumulation(self, movement: Dict):
        """Analyze whale accumulation pattern"""
        pass

    async def _validate_new_parameters(self, params: Dict) -> bool:
        """Validate new strategy parameters"""
        return True

    async def _collect_training_data(self) -> Dict:
        """Legacy stub — superseded by the subprocess call in _retrain_models."""
        return {}

    def _should_retrain(self, data: Dict) -> bool:
        """Legacy stub — superseded by the subprocess call in _retrain_models."""
        return False

    async def _validate_models(self, models: Dict) -> bool:
        """Legacy stub — superseded by the subprocess call in _retrain_models."""
        return True

    async def _get_recent_rug_pulls(self) -> List[Dict]:
        """Get recent rug pull incidents"""
        return []

    async def _save_blacklists(self):
        """Save blacklists to file"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            with open('data/blacklists.json', 'w') as f:
                json.dump({
                    'tokens': list(self.blacklisted_tokens),
                    'developers': list(self.blacklisted_devs)
                }, f)
        except Exception:
            pass

    async def _update_community_blacklists(self):
        """Update from community blacklist sources"""
        pass

    async def _send_daily_report(self):
        """Send daily performance report"""
        await self.alert_manager.send_performance_summary(
            'daily',
            self.stats
        )

    async def _check_database_health(self) -> Dict:
        """Check database health"""
        return {'healthy': True, 'message': 'OK'}

    async def _check_web3_health(self) -> Dict:
        """Check Web3 connection health"""
        return {'healthy': True, 'message': 'OK'}

    async def _check_api_health(self) -> Dict:
        """Check API connections health"""
        return {'healthy': True, 'message': 'OK'}

    def _check_memory_usage(self) -> Dict:
        """Check memory usage"""
        return {'healthy': True, 'message': 'OK'}

    def _check_cpu_usage(self) -> Dict:
        """Check CPU usage"""
        return {'healthy': True, 'message': 'OK'}

    async def _save_state(self):
        """Save bot state"""
        pass

    async def emergency_close_all_positions(self):
        """Emergency close all open positions"""
        # CRITICAL FIX (P1): Take snapshot under lock to prevent race conditions
        async with self.positions_lock:
            positions_to_close = list(self.active_positions.values())

        for position in positions_to_close:
            await self._close_position(position, "emergency_shutdown")

    # ============================================
    # ALSO UPDATE: Modify shutdown() method to call cleanup()
    # Find the shutdown() method (around line 1450) and update it:
    # ============================================

    async def shutdown(self):
        """Shutdown the engine gracefully"""
        try:
            logger.info("🔴 Initiating graceful shutdown...")
            
            # Stop the engine (which now includes cleanup)
            await self.stop()
            
            # Send shutdown notification
            await self.alert_manager.send_info("🔴 Trading bot has been shut down")
            
            logger.info("✅ Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def get_stats(self) -> Dict:
        """Get engine statistics"""
        return self.stats

    async def save_state(self):
        """Save current state"""
        await self._save_state()

    async def _watchdog(self):
        """Watchdog task to detect and log hanging tasks"""
        logger.info("🐕 Watchdog started - monitoring for hangs every 60s")

        while self.state == BotState.RUNNING:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Log task status
                alive_tasks = [t for t in self.tasks if not t.done()]
                done_tasks = [t for t in self.tasks if t.done()]

                if done_tasks:
                    logger.warning(f"⚠️ Watchdog: {len(done_tasks)} tasks have stopped!")
                    for task in done_tasks:
                        task_name = task.get_name()
                        if task.exception():
                            logger.error(f"  ❌ Task {task_name} crashed: {task.exception()}")
                        else:
                            logger.warning(f"  ⏹️ Task {task_name} completed unexpectedly")

                logger.info(f"🐕 Watchdog: {len(alive_tasks)} tasks alive, {len(done_tasks)} stopped")

                # Log event bus stats
                event_stats = self.event_bus.get_statistics()
                logger.info(f"  📊 EventBus: {event_stats['events_processed']} processed, "
                           f"{event_stats['queue_size']} queued, "
                           f"{event_stats['events_failed']} failed")

            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(60)

    async def _handle_new_pair(self, event: Event):
        """Handle new pair detected event"""
        pass

    async def _handle_position_opened(self, event: Event):
        """Handle position opened event"""
        pass

    async def _handle_position_closed(self, event: Event):
        """Handle position closed event"""
        pass

    async def _handle_unusual_volume(self, event: Event):
        """Handle unusual volume event"""
        pass

    async def _handle_rug_pull(self, event: Event):
        """Handle rug pull detection event"""
        # Add token to blacklist
        if 'token_address' in event.data:
            self.blacklisted_tokens.add(event.data['token_address'])
            await self._save_blacklists()


    def _calculate_target_price(self, pair: Dict, score: float) -> float:
        """Calculate target price based on score"""
        current_price = pair.get('price_usd', 0)
        # Higher score = higher target (5-20% profit target)
        profit_target = 0.05 + (score * 0.15)
        return current_price * (1 + profit_target)

    def _calculate_stop_loss(self, pair: Dict) -> float:
        """Calculate stop loss price"""
        current_price = pair.get('price_usd', 0)
        # 5% stop loss by default
        return current_price * 0.95

    def _determine_risk_level(self, risk_score: Optional[RiskScore]) -> str:
        """Determine risk level from risk score"""
        if not risk_score or not hasattr(risk_score, 'overall_risk'):
            return 'MEDIUM'
        
        risk = risk_score.overall_risk
        if risk < 0.3:
            return 'LOW'
        elif risk < 0.6:
            return 'MEDIUM'
        elif risk < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'


    async def _monitor_positions_with_engine(self):
        """
        Monitor positions and update with real prices from data collectors
        This runs in the engine to provide position_tracker with market data
        """
        try:
            while self.state == BotState.RUNNING:
                # Get all open positions
                if not self.position_tracker.positions:
                    await asyncio.sleep(30)
                    continue
                
                logger.debug(
                    f"Monitoring {len(self.position_tracker.positions)} positions"
                )
                
                # Update each position with current price
                for position_id, position in list(
                    self.position_tracker.positions.items()
                ):
                    try:
                        # Get chain from position metadata
                        chain = position.metadata.get('chain', 'ethereum')
                        
                        # Fetch current price from DexScreener
                        pair_data = await self.dex_collector.get_pair_data(
                            position.token_address,
                            chain=chain
                        )
                        
                        if pair_data and 'price_usd' in pair_data:
                            current_price = Decimal(str(pair_data['price_usd']))
                            
                            # Update position with current price
                            actions = await self.position_tracker.update_position_with_price(
                                position_id=position_id,
                                current_price=current_price
                            )
                            
                            # Handle any required actions
                            if actions and "close" in actions:
                                reason = actions["close"].get("reason", "unknown")
                                logger.warning(
                                    f"⚠️ Stop-loss/Take-profit hit for {position.token_symbol}: "
                                    f"{reason}"
                                )
                                
                                # Execute close through order manager
                                await self._execute_position_close(
                                    position_id=position_id,
                                    reason=reason,
                                    current_price=current_price
                                )
                            
                            elif actions and "partial_close" in actions:
                                reason = actions["partial_close"].get("reason", "partial_tp")
                                logger.info(
                                    f"📊 Partial take-profit for {position.token_symbol}"
                                )
                                # Handle partial close
                                await self._execute_partial_close(
                                    position_id=position_id,
                                    action=actions["partial_close"]
                                )
                        
                        else:
                            logger.warning(
                                f"⚠️ Could not fetch price for {position.token_symbol} "
                                f"({position.token_address})"
                            )
                    
                    except Exception as e:
                        logger.error(
                            f"Error monitoring position {position_id}: {e}",
                            exc_info=True
                        )
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}", exc_info=True)

    async def _execute_position_close(
        self,
        position_id: str,
        reason: str,
        current_price: Decimal
    ):
        """Execute position close due to stop-loss or take-profit"""
        try:
            position = self.position_tracker.positions.get(position_id)
            if not position:
                logger.error(f"Position {position_id} not found")
                return
            
            logger.info(
                f"🔴 CLOSING POSITION: {position.token_symbol} "
                f"Reason: {reason} | Price: ${current_price} | "
                f"P&L: ${position.unrealized_pnl}"
            )
            
            # Get chain info
            chain = position.metadata.get('chain', 'ethereum')
            
            # ✅ PATCH 4: Build Order object first
            from trading.orders.order_manager import build_order, OrderSide, OrderType
            from decimal import Decimal

            logger.info(
                f"🔴 Creating SELL order to close position {position.token_symbol}"
            )

            # Build the order object with correct signature
            order_obj = build_order(
                token_address=position.token_address,
                side=OrderSide.SELL,
                amount=Decimal(str(position.entry_amount)),
                order_type=OrderType.MARKET,
                chain=chain,
                slippage_tolerance=0.02,  # 2% slippage for exits
                gas_limit=500000,
                metadata={
                    'reason': reason,
                    'position_id': position_id,
                    'auto_close': True,
                    'urgent': reason in ['stop_loss_hit', 'rug_pull_detected']
                }
            )

            logger.info(
                f"📝 Order created: {order_obj.order_id}\n"
                f"   Amount: {order_obj.amount}\n"
                f"   Slippage: {order_obj.slippage_tolerance:.1%}"
            )

            # Create order using API-compliant signature
            order_id = await self.order_manager.create_order(order_obj)

            logger.info(f"✅ Order {order_id} submitted to order manager")

            # Execute the order
            success = await self.order_manager.execute_order(order_id)

            if success:
                # Get order details for result
                sell_order = self.order_manager.orders.get(order_id)
                logger.info(f"✅ Order execution started successfully")
            else:
                logger.error(f"❌ Order execution failed")
                sell_order = None
            
            if sell_order:
                # Close position in tracker
                result = await self.position_tracker.close_position_with_details(
                    position_id=position_id,
                    exit_price=current_price,
                    order_ids=[sell_order.order_id],
                    reason=reason
                )
                
                if result:
                    logger.info(
                        f"✅ Position closed successfully: {position.token_symbol} | "
                        f"Realized P&L: ${result.realized_pnl}"
                    )
                    
                    # Send alert
                    await self.alert_manager.send_trade_alert({
                        'action': 'CLOSE',
                        'symbol': position.token_symbol,
                        'reason': reason,
                        'pnl': str(result.realized_pnl),
                        'roi': f"{result.roi:.2%}"
                    })
            else:
                logger.error(
                    f"❌ Failed to create sell order for {position.token_symbol}"
                )
        
        except Exception as e:
            logger.error(
                f"Error executing position close for {position_id}: {e}",
                exc_info=True
            )

    async def _execute_partial_close(
        self,
        position_id: str,
        action: Dict
    ):
        """Execute partial position close for take-profit"""
        try:
            position = self.position_tracker.positions.get(position_id)
            if not position:
                return
            
            exit_amount = action.get('amount', position.entry_amount / 3)
            
            logger.info(
                f"📊 Partial close: {position.token_symbol} | "
                f"Amount: {exit_amount}"
            )
            
            # Create partial sell order
            chain = position.metadata.get('chain', 'ethereum')
            
            # ✅ PATCH 5: Build Order object for partial close
            from trading.orders.order_manager import build_order, OrderSide, OrderType
            from decimal import Decimal

            logger.info(
                f"📊 Creating partial SELL order for {position.token_symbol}\n"
                f"   Amount: {exit_amount}"
            )

            # Build the order object
            order_obj = build_order(
                token_address=position.token_address,
                side=OrderSide.SELL,
                amount=Decimal(str(exit_amount)),
                order_type=OrderType.MARKET,
                chain=chain,
                slippage_tolerance=0.015,  # 1.5% for partial exits
                metadata={
                    'reason': action.get('reason', 'partial_tp'),
                    'position_id': position_id,
                    'partial': True,
                    'partial_close': True,
                    'take_profit_level': action.get('take_profit_level')
                }
            )

            # Create and execute order
            order_id = await self.order_manager.create_order(order_obj)
            success = await self.order_manager.execute_order(order_id)

            if success:
                sell_order = self.order_manager.orders.get(order_id)
                logger.info(f"✅ Partial close order executing")
            else:
                logger.error(f"❌ Partial close order failed")
                sell_order = None
            
            if sell_order:
                # Partial close in tracker
                await self.position_tracker.close_position_with_details(
                    position_id=position_id,
                    exit_price=action['price'],
                    exit_amount=exit_amount,
                    order_ids=[sell_order.order_id],
                    reason=action.get('reason')
                )
                
                logger.info(
                    f"✅ Partial close executed: {position.token_symbol}"
                )
        
        except Exception as e:
            logger.error(f"Error in partial close: {e}", exc_info=True)

    # ============================================================================
    # FINAL FIX: core/engine.py - Fix liquidity key in _calculate_opportunity_score
    # Around line 1450 in _calculate_opportunity_score method
    # ============================================================================

    def _calculate_opportunity_score(
        self,
        pair: Dict,
        risk_score: Optional[RiskScore],
        patterns: Optional[Dict],
        sentiment: Optional[Dict],
        liquidity: Optional[Dict],
        contract_safety: Optional[Dict]
    ) -> float:
        """
        Calculate overall opportunity score with detailed logging
        Returns: Score between 0 and 1
        """
        try:
            age_minutes = pair.get('age_minutes', 9999)
            if age_minutes < 15:
                logger.info(f"   ❌ REJECTED: Token is too new ({age_minutes} minutes old)")
                return 0.0

            score = 0.0
            weights = 0.0
            score_breakdown = {}
            
            # Volume score (30% weight) - Max score at $250k volume
            volume_24h = pair.get('volume_24h', 0)
            if volume_24h > 0:
                volume_score = min(volume_24h / 250000, 1.0)
                score += volume_score * 0.30
                weights += 0.30
                score_breakdown['volume'] = {
                    'score': volume_score, 'weight': 0.30, 'contribution': volume_score * 0.30, 'raw_value': volume_24h
                }
            
            # Liquidity score (35% weight) - Increased weight, max score at $50k
            liquidity_usd = pair.get('liquidity_usd') or pair.get('liquidity') or 0
            if liquidity_usd > 0:
                volume_to_liq_ratio = volume_24h / liquidity_usd if liquidity_usd > 0 else 0

                # Wave-13 fix: the previous 2.0x hard-reject killed ~100% of real
                # opportunities. Typical active DEX pairs run 0.1x-1.0x vol/liq;
                # 2.0x means the full pool turns over twice per day (extreme).
                # We now use a CONFIG-tunable soft threshold (default 0.05x = 5%
                # daily turnover) to reject ghost pools with zero activity. Active
                # pairs are scored on the ratio continuously, blended into liq bucket.
                min_vol_liq_ratio = self.config.get('trading', {}).get(
                    'min_vol_liq_ratio', 0.05
                )
                if volume_to_liq_ratio < min_vol_liq_ratio:
                    logger.info(
                        f"   ❌ REJECTED: Ghost pool — vol/liq ratio "
                        f"({volume_to_liq_ratio:.3f}x) < min {min_vol_liq_ratio:.3f}x "
                        f"(vol=${volume_24h:,.0f}, liq=${liquidity_usd:,.0f})"
                    )
                    return 0.0

                # Score: liq depth (60%) + turnover activity (40%), both [0,1].
                # Turnover saturates at 2.0x (extremely active).
                liq_depth_score = min(liquidity_usd / 50000, 1.0)
                turnover_score = min(volume_to_liq_ratio / 2.0, 1.0)
                liq_score = 0.6 * liq_depth_score + 0.4 * turnover_score

                score += liq_score * 0.35
                weights += 0.35
                score_breakdown['liquidity'] = {
                    'score': liq_score, 'weight': 0.35, 'contribution': liq_score * 0.35,
                    'raw_value': liquidity_usd,
                    'vol_liq_ratio': round(volume_to_liq_ratio, 3),
                    'liq_depth_score': round(liq_depth_score, 3),
                    'turnover_score': round(turnover_score, 3),
                }
            
            # Price change score (10% weight) - Reduced weight, less emphasis on initial pump
            price_change_5m = pair.get('price_change_5m', 0)
            if price_change_5m is not None:
                # Normalize score: a 10% change gives a full score, cap at 20%
                price_score = min(max(price_change_5m / 10, 0), 2.0) / 2.0
                score += price_score * 0.10
                weights += 0.10
                score_breakdown['price_change'] = {
                    'score': price_score, 'weight': 0.10, 'contribution': price_score * 0.10, 'raw_value': price_change_5m
                }
            
            # Risk score (20% weight) - Adjusted to balance weights
            # SAFETY FIX (Wave-8 DEFECT 1): a missing/failed risk assessment is
            # NOT "absent" — it means we could not rule out a honeypot/rug. The
            # previous code only added the risk term `if risk_score`, which
            # DROPPED the 0.20 weight from the denominator on failure, so a token
            # we could not safety-check normalized HIGHER than a token with known
            # moderate risk. We now treat unknown risk as WORST-CASE and reject
            # outright: in DRY_RUN data-collection mode an unverifiable safety
            # signal must bias the decision to "do not enter", never reward it.
            if risk_score and hasattr(risk_score, 'overall_risk'):
                risk_component = 1.0 - risk_score.overall_risk
                score += risk_component * 0.20
                weights += 0.20
                score_breakdown['risk'] = {
                    'score': risk_component, 'weight': 0.20, 'contribution': risk_component * 0.20, 'raw_value': risk_score.overall_risk
                }
            else:
                # Wave-13 fix: a missing/failed risk assessment is NOT absent — it
                # means we could not rule out a honeypot/rug. The previous code
                # dropped the 0.20 weight from the denominator on failure, so an
                # un-safety-checkable token NORMALIZED HIGHER than a token with
                # known moderate risk. We now reject outright: an unverifiable
                # honeypot/rug signal must never raise the score.
                logger.warning(
                    "      ❌ REJECTED: risk assessment unavailable/failed "
                    "(treated as worst-case — cannot verify token is not a "
                    "honeypot/rug)"
                )
                return 0.0
            
            # Age bonus (5% weight) - Remains the same
            age_hours = pair.get('age_hours', 999)
            age_score = 0.0
            if age_hours < 24:
                age_score = 1.0 - (age_hours / 24)

            score += age_score * 0.05
            weights += 0.05
            score_breakdown['age'] = {
                'score': age_score, 'weight': 0.05, 'contribution': age_score * 0.05, 'raw_value': age_hours
            }
            
            # Normalize by total weights used
            if weights > 0:
                final_score = score / weights
            else:
                final_score = 0.3
            
            # Detailed logging
            token_symbol = pair.get('token_symbol', 'UNKNOWN')
            logger.info(f"      📊 Scoring breakdown for {token_symbol}:")
            
            for component, data in score_breakdown.items():
                logger.info(
                    f"         • {component.upper()}: "
                    f"score={data['score']:.3f}, "
                    f"weight={data['weight']:.0%}, "
                    f"contribution={data['contribution']:.4f} "
                    f"(raw={data['raw_value']})"
                )
            
            logger.info(f"      📈 Total weights: {weights:.2f}")
            logger.info(f"      🎯 Final normalized score: {final_score:.4f}")
            
            # Check what's missing
            missing_components = []
            if 'volume' not in score_breakdown:
                missing_components.append('volume')
            if 'liquidity' not in score_breakdown:
                missing_components.append('liquidity')
            if 'price_change' not in score_breakdown:
                missing_components.append('price_change')
            if 'risk' not in score_breakdown:
                missing_components.append('risk')
            if 'age' not in score_breakdown:
                missing_components.append('age')
            
            if missing_components:
                logger.warning(f"      ⚠️  Missing score components: {', '.join(missing_components)}")
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}", exc_info=True)
            return 0.3