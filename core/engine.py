"""
Core Trading Engine - Orchestrates all bot operations
"""

import asyncio
import uuid
import logging  # ADD THIS LINE
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
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

@dataclass
class ClosedPositionRecord:
    """Track recently closed positions for cooldown"""
    token_address: str
    closed_at: datetime
    reason: str
    pnl: float
    
    def is_cooled_down(self, cooldown_minutes: int = 60) -> bool:
        """Check if cooldown period has elapsed"""
        elapsed = (datetime.now() - self.closed_at).total_seconds() / 60
        return elapsed >= cooldown_minutes

class TradingBotEngine:
    """Main orchestration engine for the trading bot"""
    
    def __init__(self, config: Dict, mode: str = "production"):
        """
        Initialize the trading engine
        
        Args:
            config: Configuration dictionary
            mode: Operating mode
        """
        self.config = config
        self.mode = mode
        self.state = BotState.INITIALIZING
        
        # Core components
        self.event_bus = EventBus()
        self.portfolio_manager = PortfolioManager(config.get('portfolio', {}))
        self.risk_manager = RiskManager(
            config['risk_management'], 
            portfolio_manager=self.portfolio_manager
        )
        self.pattern_analyzer = PatternAnalyzer()
        self.decision_maker = DecisionMaker(config)

        # Data collectors
        # Use:
        self.dex_collector = DexScreenerCollector(
            config.get('data_sources', {}).get('dexscreener', {})
        )
        self.chain_collector = ChainDataCollector(config['web3'])
        self.social_collector = SocialDataCollector(config['data_sources']['social'])
        self.mempool_monitor = MempoolMonitor(config['web3'])
        self.whale_tracker = WhaleTracker(config['web3'])

        # ✅ ADD THIS LINE - Initialize honeypot checker
        self.honeypot_checker = HoneypotChecker(config.get('security', {}))
        
        # ML components
        self.ensemble_predictor = EnsemblePredictor()
        self.hyperparam_optimizer = HyperparameterOptimizer()
        self.rl_optimizer = RLOptimizer()
        
        # Trading components

        executor_config = {
            'web3_provider_url': config.get('web3', {}).get('provider_url'),
            'private_key': config.get('security', {}).get('private_key'),
            'chain_id': config.get('web3', {}).get('chain_id', 1),
            'max_gas_price': config.get('web3', {}).get('max_gas_price', 500),
            'gas_limit': config.get('web3', {}).get('gas_limit', 500000),
            'max_retries': 3,
            'retry_delay': 1,
            'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Mainnet router
            '1inch_api_key': config.get('api', {}).get('1inch_api_key'),
            'paraswap_api_key': config.get('api', {}).get('paraswap_api_key'),
        }

        # Database connection for logging
        from data.storage.database import DatabaseManager
        self.db = DatabaseManager(config.get('database', {}))  

        self.strategy_manager = StrategyManager(config['trading']['strategies'])
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

        # ✅ FIXED: Check both nested and flat config structures
        solana_config = config.get('solana', {})
        solana_enabled = (
            solana_config.get('enabled', False) or 
            config.get('solana_enabled', False)
        )

        if solana_enabled:
            try:
                # Try nested config first, fallback to flat
                if solana_config:
                    self.solana_executor = JupiterExecutor(solana_config)
                else:
                    # Build config from flat structure
                    solana_config = {
                        'enabled': config.get('solana_enabled', False),
                        'rpc_url': config.get('solana_rpc_url'),
                        'solana_private_key': config.get('solana_private_key'),
                        'encryption_key': config.get('encryption_key'),
                        'max_slippage_bps': config.get('jupiter_max_slippage_bps', 500),
                        'dry_run': config.get('dry_run', True),
                    }
                    self.solana_executor = JupiterExecutor(solana_config)
                
                logger.info("✅ Solana Jupiter Executor initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Solana executor: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("ℹ️ Solana trading disabled")
        
        # Monitoring
        self.alert_manager = AlertManager(config['notifications'])
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
        self.cooldown_minutes = config.get('trading', {}).get('position_cooldown_minutes', 60)
        
      
        
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
        self.event_bus.subscribe(EventType.NEW_PAIR_DETECTED, self._handle_new_pair)
        self.event_bus.subscribe(EventType.WHALE_MOVEMENT, self._handle_whale_movement)
        self.event_bus.subscribe(EventType.UNUSUAL_VOLUME, self._handle_unusual_volume)
        self.event_bus.subscribe(EventType.RUG_PULL_DETECTED, self._handle_rug_pull)
        self.event_bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self.event_bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        
    async def run(self):
        """Main engine loop"""
        try:
            # Create concurrent tasks
            self.tasks = [
                asyncio.create_task(self._monitor_new_pairs()),
                asyncio.create_task(self._monitor_existing_positions()),
                asyncio.create_task(self._process_opportunities()),
                asyncio.create_task(self._monitor_mempool()),
                asyncio.create_task(self._track_whales()),
                asyncio.create_task(self._optimize_strategies()),
                asyncio.create_task(self._retrain_models()),
                asyncio.create_task(self._update_blacklists()),
                asyncio.create_task(self._monitor_performance()),
                asyncio.create_task(self._health_check()),
                asyncio.create_task(self._monitor_wallet_balances()),
                asyncio.create_task(self._monitor_positions_with_engine()),

                
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
        max_pairs_per_chain = self.config.get('chains', {}).get('max_pairs_per_chain', 50)
        discovery_interval = self.config.get('chains', {}).get('discovery_interval', 300)
        
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
                        chain_config = self.config.get('chains', {}).get(chain, {})
                        min_liquidity = chain_config.get('min_liquidity', 10000)
                        
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
                                        min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.7)
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
            min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.7)
            logger.info(f"   📊 Score: {score:.4f} (min required: {min_score})")
            
            if score < min_score:
                # ADD THIS LOG TO SEE WHY IT FAILED
                logger.info(f"   ❌ REJECTED: Score {score:.4f} < {min_score}")
                return None
            
            # If we get here, score is good enough!
            logger.info(f"   ✅ PASSED: Score {score:.4f} >= {min_score}")

            # ✅ Calculate position size from config
            position_size = await self._calculate_position_size(
                risk_score=risk_score,
                opportunity_score=score
            )

            # Create opportunity
            opportunity = TradingOpportunity(
                token_address=pair.get('token_address', ''),
                pair_address=pair.get('pair_address', ''),
                chain=pair.get('chain', 'ethereum'),
                price=pair.get('price_usd', 0),
                liquidity=pair.get('liquidity_usd', 0),
                volume_24h=pair.get('volume_24h', 0),
                risk_score=risk_score if risk_score else RiskScore(overall_risk=0.5),
                ml_confidence=score,
                pump_probability=score * 0.8,
                rug_probability=0.2,
                expected_return=score * 100,
                recommended_position_size=position_size,  # ✅ FROM CONFIG!
                entry_strategy='momentum',
                metadata={
                    'pair': pair,
                    'risk_score': risk_score,
                    'patterns': patterns,
                    'sentiment': sentiment,
                    'liquidity_depth': liquidity_depth,
                    'contract_safety': contract_safety,
                    'holder_distribution': holder_dist,
                    'token_symbol': token_symbol
                },
                timestamp=datetime.utcnow()
            )
            
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
                    can_open = self.portfolio_manager.can_open_position()
                    
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
                        
                    self.pending_opportunities.remove(opportunity)
                    
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing opportunities: {e}", exc_info=True)
                await self.alert_manager.send_error(f"Error processing opportunities: {e}")
                await asyncio.sleep(5)
                
    # ============================================================================
    # FIX 4: engine.py - Add duplicate position check in _execute_opportunity
    # Add this check at the beginning of _execute_opportunity method (around line 620)
    # ============================================================================

    async def _execute_opportunity(self, opportunity: TradingOpportunity):
        """Execute a trading opportunity"""
        try:
            token_symbol = opportunity.metadata.get('token_symbol', 'UNKNOWN')
            token_address = opportunity.token_address.lower()
            
            # ✅ CHECK 1: Already have active position?
            if token_address in self.active_positions:
                logger.warning(f"⚠️ Already have position in {token_symbol} - SKIPPING")
                return
            
            # ✅ CHECK 2: Position on cooldown?
            if token_address in self.recently_closed:
                record = self.recently_closed[token_address]
                if not record.is_cooled_down(self.cooldown_minutes):
                    elapsed = (datetime.now() - record.closed_at).total_seconds() / 60
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
            max_positions = self.config.get('trading', {}).get('max_positions', 40)
            if len(self.active_positions) >= max_positions:
                logger.warning(f"⚠️ Max positions reached ({len(self.active_positions)}) - SKIPPING")
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
            
            
            # Check wallet balance before trading (skip in DRY_RUN mode)
            if not self.config.get('dry_run', True):  # ✅ Only check balance in LIVE trading
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
            
            # Check if in DRY_RUN mode
            if self.config.get('dry_run', True):
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
                    'stop_loss_percentage': 0.1,
                    'take_profit_percentage': 0.3,
                    'max_hold_time': 60,
                    'metadata': opportunity.metadata,
                    'is_dry_run': True,
                    'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
                }
                
                # Add to active positions
                self.active_positions[token_address] = position

                # ✅ NEW: Add to portfolio manager
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        # Calculate stop loss and take profit levels
                        stop_loss_price = float(opportunity.price) * (1 - position.get('stop_loss_percentage', 0.1))
                        take_profit_price = float(opportunity.price) * (1 + position.get('take_profit_percentage', 0.3))
                        
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
                            'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
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
                
                # Update stats
                self.stats['total_trades'] += 1
                self.stats['successful_trades'] += 1

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
                    f"Stop Loss: -10% | Take Profit: +30%\n"
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
            
            # Create trade order (chain-agnostic)
            from trading.orders.order_manager import Order, OrderType
            
            # For Solana, token addresses are base58 mints
            # For EVM, they're hex addresses
            order = Order(
                order_id=str(uuid.uuid4()),
                token_in="So11111111111111111111111111111111111111112" if chain == 'solana' else self.config.get('weth_address'),  # SOL or WETH
                token_out=opportunity.token_address,
                amount=Decimal(str(opportunity.recommended_position_size / opportunity.price)),
                order_type=OrderType.MARKET,
                slippage=0.05,
                chain=opportunity.chain,
                wallet_address=str(self.solana_executor.wallet_keypair.pubkey()) if chain == 'solana' else self.config.get('wallet_address'),
                metadata={
                    'opportunity_id': opportunity.metadata.get('opportunity_id'),
                    'token_symbol': token_symbol,
                    'score': opportunity.score,
                    'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
                }
            )
            
            # ✅ CRITICAL: Final safety checks before real execution
            is_dry_run = self.config.get('dry_run', True)           # ✅ DEFINE IT!
            if not is_dry_run:
                logger.info(f"🔍 FINAL SAFETY CHECKS for {token_symbol}...")
                
                # 1. Verify balance
                try:
                    balance = await executor.get_balance(order.token_in)
                    required = order.amount * Decimal('1.1')  # Need 10% buffer for gas
                    
                    if balance < required:
                        logger.error(f"❌ INSUFFICIENT BALANCE: Have {balance}, need {required}")
                        await self.alert_manager.send_error(
                            f"Trade cancelled - Insufficient balance\n"
                            f"Token: {token_symbol}\n"
                            f"Have: {balance:.4f}\n"
                            f"Need: {required:.4f}"
                        )
                        return
                    logger.info(f"  ✅ Balance check passed: {balance:.4f} >= {required:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Balance check failed: {e}")
                    return
                
                # 2. Verify position size against risk limits
                max_position = self.portfolio_manager.get_max_position_size(opportunity.chain)
                if opportunity.recommended_position_size > max_position:
                    logger.error(
                        f"❌ POSITION SIZE EXCEEDS LIMIT: "
                        f"${opportunity.recommended_position_size:.2f} > ${max_position:.2f}"
                    )
                    await self.alert_manager.send_error(
                        f"Trade cancelled - Position size too large\n"
                        f"Token: {token_symbol}\n"
                        f"Requested: ${opportunity.recommended_position_size:.2f}\n"
                        f"Max allowed: ${max_position:.2f}"
                    )
                    return
                logger.info(f"  ✅ Position size check passed")
                
                # 3. Final confirmation prompt (optional - remove in production)
                logger.warning(f"⚠️  ABOUT TO EXECUTE REAL TRADE")
                logger.warning(f"   Token: {token_symbol}")
                logger.warning(f"   Amount: ${opportunity.recommended_position_size:.2f}")
                logger.warning(f"   Chain: {chain}")
                
            # Now execute the trade
            result = await executor.execute_trade(order)
            
            # Wait for transaction confirmation
            if not self.config.get('dry_run', True):
                try:
                    if chain == 'solana':
                        # Solana confirmation
                        confirmation = await self.solana_executor.wait_for_confirmation(
                            result.tx_hash,
                            max_wait=30
                        )
                        if not confirmation or confirmation.get('err'):
                            logger.error(f"❌ Solana transaction FAILED: {result.tx_hash}")
                            await self.db.update_trade(trade_id, {'status': 'failed'})
                            return
                    else:
                        # EVM confirmation
                        w3 = self.trade_executor.w3
                        receipt = w3.eth.wait_for_transaction_receipt(
                            result.tx_hash,
                            timeout=120
                        )
                        
                        if receipt.status != 1:
                            logger.error(f"❌ Transaction FAILED: {result.tx_hash}")
                            # Update database with failed status
                            await self.db.update_trade(trade_id, {'status': 'failed'})
                            return
                        
                        logger.info(f"✅ Transaction CONFIRMED: {result.tx_hash}")
                        
                except Exception as e:
                    logger.error(f"❌ Transaction confirmation failed: {e}")
                    return


            if result['success']:
                # Track position
                position = {
                    'token_address': opportunity.token_address,
                    'token_symbol': token_symbol,
                    'entry_price': result.get('execution_price', opportunity.price),
                    'amount': result.get('token_amount', order.amount),
                    'entry_value': opportunity.recommended_position_size,
                    'tx_hash': result.get('signature' if chain == 'solana' else 'transactionHash'),
                    'chain': opportunity.chain,
                    'strategy': {'name': opportunity.entry_strategy},
                    'risk_score': opportunity.risk_score,
                    'entry_time': datetime.now(),
                    'stop_loss_percentage': 0.1,
                    'take_profit_percentage': 0.3,
                    'metadata': {
                        **opportunity.metadata,
                        'executor_type': 'Jupiter' if chain == 'solana' else 'EVM'
                    }
                }
                
                self.active_positions[opportunity.token_address] = position
                self.stats['total_trades'] += 1
                self.stats['successful_trades'] += 1

                # Update circuit breaker metrics for successful real trade
                actual_slippage = result.get('slippage_bps', 0)
                self.risk_manager.update_trade_metrics({
                    'success': True,
                    'profit': 0,  # Entry only, no P&L yet
                    'slippage_bps': actual_slippage
                })

                # ✅ ADD COMPREHENSIVE LOGGING (NEW)
                execution_price = result.get('execution_price', opportunity.price)
                token_amount = result.get('token_amount', order.amount)
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
                logger.info(f"   TX Hash: {result.get('signature' if chain == 'solana' else 'transactionHash', 'N/A')}")
                logger.info(f"   Gas Used: ${result.get('gas_fee', 0):.4f}")
                
                # Send success alert

                # ✅ LOG REAL TRADE TO DATABASE (NEW)
                try:
                    trade_id = str(uuid.uuid4())
                    execution_price = result.get('execution_price', opportunity.price)
                    token_amount = result.get('token_amount', order.amount)
                    actual_value = float(execution_price * token_amount)
                    
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
                        'slippage': float(result.get('slippage_bps', 0)) / 10000,  # Convert bps to decimal
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
                            'tx_hash': result.get('signature' if chain == 'solana' else 'transactionHash'),
                            'expected_price': float(opportunity.price),
                            'execution_price': float(execution_price),
                            'slippage_bps': result.get('slippage_bps', 0)
                        }
                    }
                    
                    # Add trade_id to position for later reference
                    position['trade_id'] = trade_id
                    
                    await self.db.save_trade(trade_data)
                    logger.info(f"✅ Real trade logged to database: {trade_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to log real trade to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Send success alert
                tx_link = result.get('explorer_url', f"Transaction: {result.get('signature' or 'transactionHash', 'N/A')[:10]}...")
                
                # Build better alert with actual values
                execution_price = result.get('execution_price', opportunity.price)
                token_amount = result.get('token_amount', order.amount)
                actual_value = float(execution_price * token_amount)
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
                if not self.active_positions:
                    await asyncio.sleep(5)
                    continue
                
                logger.info(f"📊 Monitoring {len(self.active_positions)} active positions...")
                
                # ✅ STEP 1: Collect all prices first
                price_data = {}
                for token_address, position in list(self.active_positions.items()):
                    try:
                        chain = position.get('chain', 'ethereum')
                        token_symbol = position.get('token_symbol', 'UNKNOWN')
                        
                        # Get current price from DexScreener
                        current_price = await self.dex_collector.get_token_price(
                            token_address=token_address,
                            chain=chain
                        )
                        
                        if current_price:
                            price_data[token_address] = float(current_price)
                        else:
                            logger.warning(
                                f"⚠️ Could not get price for {token_symbol} "
                                f"({token_address[:10]}...) on {chain}"
                            )
                    except Exception as e:
                        logger.error(f"Error fetching price for {token_address}: {e}")
                        continue
                
                # ✅ STEP 2: Bulk update portfolio manager
                if price_data and hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    try:
                        updated_count = await self.portfolio_manager.update_all_positions(price_data)
                        logger.debug(f"📊 Updated {updated_count} positions in portfolio manager")
                    except Exception as e:
                        logger.error(f"Error updating portfolio positions: {e}")
                
                # ✅ STEP 3: Update active_positions dict (for backward compatibility)
                for token_address, position in list(self.active_positions.items()):
                    try:
                        if token_address in price_data:
                            current_price = price_data[token_address]
                            
                            # ✅ FIX: Get token_symbol for THIS position (not reused from outer loop)
                            position_symbol = position.get('token_symbol', 'UNKNOWN')
                            
                            from decimal import Decimal
                            position['current_price'] = Decimal(str(current_price))
                            position['current_value'] = Decimal(str(current_price)) * position['amount']
                            
                            # Calculate P&L
                            entry_value = position['entry_price'] * position['amount']
                            current_value = position['current_value']
                            position['pnl'] = current_value - entry_value
                            position['pnl_percentage'] = float((current_value - entry_value) / entry_value * 100)
                            
                            # Calculate holding time
                            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 60
                            
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
                                    f"  🚪 EXIT SIGNAL for {token_symbol}: {reason}",
                                    extra={
                                        'token_address': token_address,
                                        'symbol': token_symbol,
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
                        
                    except Exception as e:
                        # ✅ FIX: Now all variables are guaranteed to be defined
                        logger.error(
                            f"Error monitoring position {token_address[:10] if token_address else 'unknown'} "
                            f"({token_symbol or 'UNKNOWN'}): {e}",
                            extra={
                                'token_address': token_address,
                                'symbol': token_symbol,
                                'chain': chain,
                                'error': str(e)
                            },
                            exc_info=True
                        )
                        continue
                
                # Log portfolio summary
                total_value = sum(p.get('current_value', 0) for p in self.active_positions.values())
                total_pnl = sum(p.get('pnl', 0) for p in self.active_positions.values())
                
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
                    f"💼 Portfolio: {len(self.active_positions)} positions, "
                    f"Value: ${total_value:.2f}, P&L: ${total_pnl:.2f}{chain_summary}"
                )
                
                # Get interval from config
                update_interval = self.config.get('position_update_interval_seconds', 10)
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                # ✅ FIX: Safe variable access in outer exception
                logger.error(
                    f"Error in position monitoring loop for {token_address[:10] if token_address else 'unknown'}: {e}",
                    exc_info=True
                )
                await asyncio.sleep(30)


    async def _calculate_position_size(
        self,
        risk_score: Optional[float] = None,
        opportunity_score: float = 0.7
    ) -> float:
        try:
            # Get config values
            from config.config_manager import PortfolioConfig
            
            portfolio_config = PortfolioConfig()
            
            # ✅ Get both values from PortfolioConfig
            portfolio_balance = portfolio_config.initial_balance
            max_positions = portfolio_config.max_positions  # ✅ NOW FROM PORTFOLIO CONFIG!
            base_position_size = portfolio_balance / max_positions
            
            logger.debug(
                f"📊 Position sizing: Portfolio=${portfolio_balance}, "
                f"Max positions={max_positions}, Base=${base_position_size}"
            )
            
            # Apply risk adjustment if provided
            if risk_score and hasattr(risk_score, 'overall_risk'):
                # Reduce size for high risk (max 30% reduction)
                risk_multiplier = 1.0 - (risk_score.overall_risk * 0.3)
                adjusted_size = base_position_size * risk_multiplier
                
                logger.debug(
                    f"   Risk: {risk_score.overall_risk:.2f}, "
                    f"Multiplier: {risk_multiplier:.2f}, "
                    f"Adjusted: ${adjusted_size:.2f}"
                )
            else:
                adjusted_size = base_position_size
            
            # Ensure within bounds
            min_size = portfolio_config.min_position_size or 5
            max_size = base_position_size * 1.5  # Allow up to 1.5x for great opportunities
            
            final_size = max(min_size, min(adjusted_size, max_size))
            
            logger.info(f"💰 Calculated position size: ${final_size:.2f}")
            
            return float(final_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 10.0  # Safe fallback
                
    # ============================================================================
    # FIX 2: engine.py - Fix _check_exit_conditions method (around line 820)
    # ============================================================================

    async def _check_exit_conditions(self, position: Dict) -> tuple[bool, str]:
        """Check if position should be closed"""
        try:
            # Get position details
            pnl_percentage = position.get('pnl_percentage', 0)
            holding_time = (datetime.now() - position['entry_time']).total_seconds() / 60  # minutes
            
            # 1. Take profit hit (default 30%)
            take_profit = position.get('take_profit_percentage', 0.3) * 100
            if pnl_percentage >= take_profit:
                logger.info(f"  ✅ Take profit hit: {pnl_percentage:.2f}% >= {take_profit:.2f}%")
                return True, "take_profit"
            
            # 2. Stop loss hit (default -10%)
            stop_loss = -position.get('stop_loss_percentage', 0.1) * 100
            if pnl_percentage <= stop_loss:
                logger.info(f"  🛑 Stop loss hit: {pnl_percentage:.2f}% <= {stop_loss:.2f}%")
                return True, "stop_loss"
            
            # 3. Time-based exit (default 60 minutes for scalping)
            max_hold_time = position.get('max_hold_time', 60)  # minutes
            if holding_time > max_hold_time:
                logger.info(f"  ⏰ Max hold time reached: {holding_time:.1f}min > {max_hold_time}min")
                return True, "time_limit"
            
            # 4. Trailing stop (if profit > 10%, exit if drops back to 5%)
            if pnl_percentage > 10:
                max_profit = position.get('max_profit', pnl_percentage)
                position['max_profit'] = max(max_profit, pnl_percentage)
                
                # If dropped more than 50% from peak
                if pnl_percentage < (position['max_profit'] * 0.5):
                    logger.info(f"  📉 Trailing stop: dropped from {position['max_profit']:.2f}% to {pnl_percentage:.2f}%")
                    return True, "trailing_stop"
            
            # 5. Volatility exit (sudden price movement)
            if 'last_price' in position:
                price_change = abs((position['current_price'] - position['last_price']) / position['last_price']) * 100
                if price_change > 20:  # 20% sudden move
                    logger.info(f"  ⚡ High volatility exit: {price_change:.2f}% price change")
                    return True, "high_volatility"
            
            position['last_price'] = position.get('current_price', position['entry_price'])
            
            # 6. ML-based exit signal (if available)
            if self.config.get('use_ml_exits', False):
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
        if token_address not in self.active_positions:
            logger.error(f"❌ Cannot close position - not in active positions: {token_symbol}")
            return False
        
        try:
            logger.info(f"💰 CLOSING POSITION: {token_symbol} ({token_address[:10]}...)")
            logger.info(f"   Reason: {reason}")
            
            is_dry_run = self.config.get('dry_run', True) or position.get('is_dry_run', False)
            
            if is_dry_run:
                # Calculate P&L
                current_price = position.get('current_price', position['entry_price'])
                entry_price = position['entry_price']
                amount = position['amount']
                
                final_pnl = (current_price - entry_price) * amount
                pnl_percentage = float((current_price - entry_price) / entry_price * 100)
                holding_time = (datetime.now() - position['entry_time']).total_seconds() / 60
                
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
                        updated_metadata = {
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
                        }
                        
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
            
            # REAL EXECUTION
            from trading.executors.base_executor import TradeOrder
            
            order = TradeOrder(
                token_address=token_address,
                side='sell',
                amount=position['amount'],
                slippage=0.05,
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


            # Wait for transaction confirmation
            if not self.config.get('dry_run', True):
                try:
                    if chain == 'solana':
                        # Solana confirmation
                        confirmation = await self.solana_executor.wait_for_confirmation(
                            result.tx_hash,
                            max_wait=30
                        )
                    else:
                        # EVM confirmation
                        w3 = self.trade_executor.w3
                        receipt = w3.eth.wait_for_transaction_receipt(
                            result.tx_hash,
                            timeout=120
                        )
                        
                        if receipt.status != 1:
                            logger.error(f"❌ Transaction FAILED: {result.tx_hash}")
                            # Update database with failed status
                            await self.db.update_trade(trade_id, {'status': 'failed'})
                            return
                        
                        logger.info(f"✅ Transaction CONFIRMED: {result.tx_hash}")
                        
                except Exception as e:
                    logger.error(f"❌ Transaction confirmation failed: {e}")
                    return

            
            if result.success:
                exit_price = result.execution_price
                final_pnl = (exit_price - position['entry_price']) * position['amount']
                pnl_percentage = ((exit_price - position['entry_price']) / position['entry_price']) * 100
                
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
                
                if hasattr(self, 'position_tracker') and position.get('tracker_id'):
                    await self.position_tracker.close_position(position['tracker_id'])
                
                del self.active_positions[token_address]
                
                emoji = "💰" if final_pnl > 0 else "💸"
                await self.alert_manager.send_trade_alert(
                    f"{emoji} Position Closed: {token_symbol}\n"
                    f"Entry: ${position['entry_price']:.8f}\n"
                    f"Exit: ${exit_price:.8f}\n"
                    f"P&L: ${final_pnl:.2f} ({pnl_percentage:.2f}%)\n"
                    f"Reason: {reason}\n"
                    f"Tx: {result.tx_hash[:10]}..."
                )
                
                logger.info(f"✅ Successfully closed position: {token_symbol}")
                return True
            else:
                logger.error(f"❌ Failed to close position: {result.error}")
                
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
                
                for tx in pending_txs:
                    # Check if it affects our positions
                    if tx['to'] in self.active_positions:
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
                
                for movement in movements:
                    # Check if it affects our tokens
                    if movement['token'] in self.active_positions:
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
        """Periodically retrain ML models"""
        while self.state == BotState.RUNNING:
            try:
                # Wait for retrain interval
                await asyncio.sleep(self.config['ml']['retrain_interval_hours'] * 3600)
                
                # Collect training data
                training_data = await self._collect_training_data()
                
                # Check if retraining is needed
                if self._should_retrain(training_data):
                    # Train new models
                    new_models = await self.ensemble_predictor.retrain(training_data)
                    
                    # Validate on test set
                    if await self._validate_models(new_models):
                        await self.ensemble_predictor.update_models(new_models)
                        await self.alert_manager.send_info("ML models retrained successfully")
                        
            except Exception as e:
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
                # Calculate metrics
                metrics = {
                    'total_trades': self.stats['total_trades'],
                    'win_rate': self.stats['successful_trades'] / max(self.stats['total_trades'], 1),
                    'total_pnl': self.stats['total_profit'],
                    'active_positions': len(self.active_positions),
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
                for position in list(self.active_positions.values()):
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
                if hasattr(collector, 'cleanup'):
                    try:
                        await collector.cleanup()
                        logger.info(f"✅ {name} collector cleaned up")
                    except Exception as e:
                        logger.debug(f"Error cleaning up {name}: {e}")
            
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
 #           min_liquidity = self.config.get('trading', {}).get('min_liquidity_threshold', 50000)
            chain_name = opportunity.chain.lower() if hasattr(opportunity, 'chain') else 'ethereum'
            chain_config = self.config.get('chains', {}).get(chain_name, {})
            min_liquidity = chain_config.get('min_liquidity', 50000)          
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
            
            # 6. Verify contract is verified (if available)
            logger.info(f"   Checking contract verification...")
            contract_safety = opportunity.metadata.get('contract_safety', {})
            if not contract_safety.get('verified', True):
                logger.warning(f"   ⚠️  Contract not verified - proceeding with caution")
            
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

    async def _load_state(self):
        """Load saved bot state"""
        try:
            # Load from database or file
            pass
        except Exception:
            pass

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
        """Check developer reputation score"""
        # Placeholder - returns neutral score
        return 0.5

    async def _analyze_liquidity_depth(self, pair: Dict) -> Dict:
        """Analyze liquidity depth"""
        return {'depth': pair.get('liquidity', 0)}

    async def _check_smart_contract(self, token_address: str) -> Dict:
        """Check smart contract for vulnerabilities"""
        return {'verified': True, 'issues': []}

    async def _analyze_holder_distribution(self, token_address: str) -> Dict:
        """Analyze token holder distribution"""
        return {'concentrated': False}

    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract ML features from data"""
        # Placeholder - return dummy features
        return np.random.rand(10)

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
        """Collect data for model training"""
        return {}

    def _should_retrain(self, data: Dict) -> bool:
        """Check if models should be retrained"""
        return False

    async def _validate_models(self, models: Dict) -> bool:
        """Validate new models"""
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
        for position in list(self.active_positions.values()):
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
                            position.token_address
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
            score = 0.0
            weights = 0.0
            score_breakdown = {}
            
            # Volume score (30% weight)
            volume_24h = pair.get('volume_24h', 0)
            if volume_24h > 0:
                volume_score = min(volume_24h / 20000, 1.0)
                score += volume_score * 0.3
                weights += 0.3
                score_breakdown['volume'] = {
                    'score': volume_score,
                    'weight': 0.3,
                    'contribution': volume_score * 0.3,
                    'raw_value': volume_24h
                }
            
            # Liquidity score (25% weight) - FIX THIS SECTION:
            # OLD (BROKEN):
            # liquidity_usd = pair.get('liquidity_usd', 0)
            
            # NEW (FIXED) - Check both possible keys:
            liquidity_usd = pair.get('liquidity_usd') or pair.get('liquidity') or 0
            
            if liquidity_usd > 0:
                # Lowered threshold: $10k = max (was $50k)
                liq_score = min(liquidity_usd / 3000, 1.0)
                score += liq_score * 0.25
                weights += 0.25
                score_breakdown['liquidity'] = {
                    'score': liq_score,
                    'weight': 0.25,
                    'contribution': liq_score * 0.25,
                    'raw_value': liquidity_usd
                }
            
            # Price change score (20% weight)
            price_change_5m = pair.get('price_change_5m', 0)
            if price_change_5m or price_change_5m == 0:
                price_score = min(max(price_change_5m / 10, 0), 1.0)
                score += price_score * 0.2
                weights += 0.2
                score_breakdown['price_change'] = {
                    'score': price_score,
                    'weight': 0.2,
                    'contribution': price_score * 0.2,
                    'raw_value': price_change_5m
                }
            
            # Risk score (15% weight)
            if risk_score and hasattr(risk_score, 'overall_risk'):
                risk_component = 1.0 - risk_score.overall_risk
                score += risk_component * 0.15
                weights += 0.15
                score_breakdown['risk'] = {
                    'score': risk_component,
                    'weight': 0.15,
                    'contribution': risk_component * 0.15,
                    'raw_value': risk_score.overall_risk
                }
            
            # Age bonus (10% weight)
            age_hours = pair.get('age_hours', 999)
            if age_hours < 24:  # Keep at 24h for now
                age_score = 1.0 - (age_hours / 72)
                score += age_score * 0.1
                weights += 0.1
                score_breakdown['age'] = {
                    'score': age_score,
                    'weight': 0.1,
                    'contribution': age_score * 0.1,
                    'raw_value': age_hours
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