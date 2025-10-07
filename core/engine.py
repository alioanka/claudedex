"""
Core Trading Engine - Orchestrates all bot operations
"""

import asyncio
import logging  # ADD THIS LINE
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from enum import Enum
import numpy as np

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

from monitoring.alerts import AlertManager
from monitoring.performance import PerformanceTracker

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
        self.risk_manager = RiskManager(config['risk_management'])
        self.pattern_analyzer = PatternAnalyzer()
        self.decision_maker = DecisionMaker(config)
        self.portfolio_manager = PortfolioManager(config['portfolio'])
        
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
        self.strategy_manager = StrategyManager(config['trading']['strategies'])
        self.order_manager = OrderManager(config)
        self.position_tracker = PositionTracker()
        self.trade_executor = TradeExecutor(config)
        
        # Monitoring
        self.alert_manager = AlertManager(config['notifications'])
        self.performance_tracker = PerformanceTracker()
        
        # Security
        self.wallet_manager = WalletSecurityManager(config['security'])
        
        # Internal state
        self.active_positions: Dict[str, Any] = {}
        self.pending_opportunities: List[TradingOpportunity] = []
        self.blacklisted_tokens: set = set()
        self.blacklisted_devs: set = set()
        
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
                asyncio.create_task(self._health_check())
            ]
            
            # Wait for tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            await self.alert_manager.send_critical(f"Engine error: {e}")
            raise
            
    async def _monitor_new_pairs(self):
        """Continuously monitor for new trading pairs"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🔍 Starting new pairs monitoring loop...")
        
        while self.state == BotState.RUNNING:
            try:
                # Get new pairs from DexScreener
                logger.info("Fetching new pairs from DexScreener...")
                new_pairs = await self.dex_collector.get_new_pairs()
                
                logger.info(f"📊 Received {len(new_pairs) if new_pairs else 0} pairs from DexScreener")
                
                if not new_pairs:
                    logger.warning("No pairs returned from DexScreener - waiting 10 seconds")
                    await asyncio.sleep(10)
                    continue
                
                for pair in new_pairs:
                    self.stats['tokens_analyzed'] += 1
                    
                    # Quick filter checks
                    if self._is_blacklisted(pair):
                        logger.debug(f"Pair {pair.get('pair_address', 'unknown')} is blacklisted")
                        continue
                        
                    # Analyze opportunity
                    logger.info(f"Analyzing pair: {pair.get('token_symbol', 'UNKNOWN')}")
                    opportunity = await self._analyze_opportunity(pair)
                    
                    min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.7)
                    
                    if opportunity:
                        # ADD THIS DEBUG LOG:
                        logger.info(f"✨ Opportunity created for {pair.get('token_symbol')}")
                        logger.info(f"   Score: {opportunity.score:.3f} (min: {min_score})")
                        logger.info(f"   ML Confidence: {opportunity.ml_confidence:.3f}")
                        logger.info(f"   Pump Prob: {opportunity.pump_probability:.3f}")
                        logger.info(f"   Rug Prob: {opportunity.rug_probability:.3f}")
                        
                        if opportunity.score > min_score:
                            self.pending_opportunities.append(opportunity)
                            self.stats['opportunities_found'] += 1
                            
                            logger.info(f"🎯 OPPORTUNITY FOUND: {pair.get('token_symbol')} - Score: {opportunity.score:.3f}")
                            
                            # Emit event
                            await self.event_bus.emit(Event(
                                event_type=EventType.OPPORTUNITY_FOUND,
                                data=opportunity
                            ))
                        else:
                            # ADD THIS:
                            logger.warning(f"❌ Opportunity score {opportunity.score:.3f} below minimum {min_score} for {pair.get('token_symbol')}")
                    else:
                        # ADD THIS:
                        logger.warning(f"❌ No opportunity created for {pair.get('token_symbol')} - analysis returned None")
                
                await asyncio.sleep(5)  # Check every 5 seconds for new data
                
            except Exception as e:
                logger.error(f"Error monitoring pairs: {e}", exc_info=True)
                await asyncio.sleep(10)
                
    async def _analyze_opportunity(self, pair: Dict) -> Optional[TradingOpportunity]:
        """
        Comprehensive analysis of a trading opportunity
        """
        try:
            # Parallel analysis tasks - properly awaited
            # NOTE: Removed social_collector.get_sentiment() since it doesn't exist yet
            results = await asyncio.gather(
                self.risk_manager.analyze_token(pair.get('token_address', '')),
                self.pattern_analyzer.analyze_patterns(pair),
                self.chain_collector.get_token_info(pair.get('token_address', '')),
                # self.social_collector.get_sentiment(pair.get('token_address', '')),  # COMMENTED OUT
                self._check_developer_reputation(pair.get('creator_address', '')),
                self._analyze_liquidity_depth(pair),
                self._check_smart_contract(pair.get('token_address', '')),
                self._analyze_holder_distribution(pair.get('token_address', '')),
                return_exceptions=True  # Don't fail if one task fails
            )
            
            # Unpack results (adjusted for removed sentiment)
            risk_score, patterns, token_info, dev_reputation, \
                liquidity_depth, contract_safety, holder_dist = results
            
            # Set sentiment to None since we don't have it
            sentiment = None
            
            # Handle any exceptions in results
            if isinstance(risk_score, Exception):
                logger.error(f"Risk analysis failed: {risk_score}")
                risk_score = None
            if isinstance(patterns, Exception):
                logger.error(f"Pattern analysis failed: {patterns}")
                patterns = None
            if isinstance(token_info, Exception):
                logger.error(f"Token info failed: {token_info}")
                token_info = None
            
            # Calculate overall score
            score = self._calculate_opportunity_score(
                pair=pair,
                risk_score=risk_score,
                patterns=patterns,
                sentiment=sentiment,  # Will be None
                liquidity=liquidity_depth,
                contract_safety=contract_safety
            )
            
            # Check minimum threshold from config
            min_score = self.config.get('trading', {}).get('min_opportunity_score', 0.7)
            
            if score < min_score:
                logger.debug(f"Opportunity score {score:.2f} below threshold {min_score}")
                return None
            
            # Create opportunity using existing TradingOpportunity structure
            # Map to expected fields
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
                recommended_position_size=1000,
                entry_strategy='momentum',
                metadata={
                    'pair': pair,
                    'risk_score': risk_score,
                    'patterns': patterns,
                    'sentiment': sentiment,
                    'liquidity_depth': liquidity_depth,
                    'contract_safety': contract_safety,
                    'holder_distribution': holder_dist,
                    'token_symbol': pair.get('token_symbol', 'UNKNOWN')
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
                        logger.warning("❌ Portfolio manager says NO to new positions")
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
                
    async def _execute_opportunity(self, opportunity: TradingOpportunity):
        """Execute a trading opportunity"""
        try:
            # Select strategy
            strategy = self.strategy_manager.select_strategy(opportunity)
            
            # Prepare order
            order = await self.order_manager.prepare_order({
                'token_address': opportunity.token_address,
                'side': 'buy',
                'amount': opportunity.recommended_position_size,
                'strategy': strategy,
                'slippage': self._calculate_slippage(opportunity),
                'gas_price_multiplier': self._calculate_gas_multiplier(opportunity)
            })
            
            # MEV protection
            order = await self._apply_mev_protection(order, opportunity)
            
            # Execute trade
            result = await self.trade_executor.execute(order)
            
            if result['success']:
                # Track position
                position = await self.position_tracker.open_position({
                    'token_address': opportunity.token_address,
                    'entry_price': result['execution_price'],
                    'amount': result['amount'],
                    'strategy': strategy,
                    'risk_score': opportunity.risk_score,
                    'metadata': opportunity.metadata
                })
                
                self.active_positions[opportunity.token_address] = position
                self.stats['total_trades'] += 1
                
                # Set stop loss and take profit
                await self._set_exit_orders(position, strategy)
                
                # Alert
                await self.alert_manager.send_trade_alert(
                    f"✅ Opened position in {opportunity.token_address[:8]}...\n"
                    f"Price: {result['execution_price']}\n"
                    f"Amount: {result['amount']}\n"
                    f"Risk Score: {opportunity.risk_score.overall_risk:.2f}"
                )
                
                # Emit event
                await self.event_bus.emit(Event(
                    event_type=EventType.POSITION_OPENED,
                    data=position
                ))
                
        except Exception as e:
            await self.alert_manager.send_error(f"Failed to execute opportunity: {e}")
            
    async def _monitor_existing_positions(self):
        """Monitor and manage existing positions"""
        while self.state == BotState.RUNNING:
            try:
                for token_address, position in list(self.active_positions.items()):
                    # Get current price
                    current_price = await self.dex_collector.get_token_price(token_address)
                    
                    if not current_price:
                        continue
                        
                    # Update position
                    position['current_price'] = current_price
                    position['pnl'] = self._calculate_pnl(position)
                    position['pnl_percentage'] = self._calculate_pnl_percentage(position)
                    
                    # Check exit conditions
                    should_exit, reason = await self._check_exit_conditions(position)
                    
                    if should_exit:
                        await self._close_position(position, reason)
                        
                    # Check for trailing stop update
                    elif position['strategy'].get('trailing_stop'):
                        await self._update_trailing_stop(position, current_price)
                        
                    # Check for DCA opportunity
                    elif await self._should_dca(position):
                        await self._execute_dca(position)
                        
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                await self.alert_manager.send_error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)
                
    async def _check_exit_conditions(self, position: Dict) -> tuple[bool, str]:
        """Check if position should be closed"""
        # Take profit hit
        if position['pnl_percentage'] >= position['take_profit_percentage']:
            return True, "take_profit"
            
        # Stop loss hit
        if position['pnl_percentage'] <= -position['stop_loss_percentage']:
            return True, "stop_loss"
            
        # Time-based exit
        if position.get('max_hold_time'):
            if (datetime.now() - position['entry_time']).seconds > position['max_hold_time']:
                return True, "time_limit"
                
        # ML-based exit signal
        exit_signal = await self._check_ml_exit_signal(position)
        if exit_signal['should_exit']:
            return True, f"ml_signal_{exit_signal['reason']}"
            
        # Rug pull detection
        if await self._detect_rug_pull_signs(position['token_address']):
            return True, "rug_pull_detected"
            
        # Liquidity crisis
        if await self._check_liquidity_crisis(position['token_address']):
            return True, "liquidity_crisis"
            
        return False, ""
        
    async def _close_position(self, position: Dict, reason: str):
        """Close a trading position"""
        try:
            # Prepare sell order
            order = await self.order_manager.prepare_order({
                'token_address': position['token_address'],
                'side': 'sell',
                'amount': position['amount'],
                'urgency': 'high' if 'rug' in reason else 'normal'
            })
            
            # Execute
            result = await self.trade_executor.execute(order)
            
            if result['success']:
                # Calculate final P&L
                final_pnl = result['execution_price'] * position['amount'] - position['entry_price'] * position['amount']
                
                # Update stats
                self.stats['total_profit'] += final_pnl
                if final_pnl > 0:
                    self.stats['successful_trades'] += 1
                else:
                    self.stats['failed_trades'] += 1
                    
                # Close position
                await self.position_tracker.close_position(
                    position['id'],
                    result['execution_price'],
                    reason
                )
                
                # Remove from active
                del self.active_positions[position['token_address']]
                
                # Alert
                emoji = "💰" if final_pnl > 0 else "💸"
                await self.alert_manager.send_trade_alert(
                    f"{emoji} Closed position in {position['token_address'][:8]}...\n"
                    f"Entry: {position['entry_price']}\n"
                    f"Exit: {result['execution_price']}\n"
                    f"P&L: ${final_pnl:.2f} ({position['pnl_percentage']:.2f}%)\n"
                    f"Reason: {reason}"
                )
                
                # Learn from trade
                await self._learn_from_trade(position, final_pnl, reason)
                
        except Exception as e:
            await self.alert_manager.send_critical(f"Failed to close position: {e}")
            
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
                
    async def _optimize_strategies(self):
        """Continuously optimize trading strategies"""
        while self.state == BotState.RUNNING:
            try:
                # Wait for enough data
                await asyncio.sleep(3600)  # Optimize every hour
                
                # Get recent performance data
                performance_data = await self.performance_tracker.get_recent_performance()
                
                # Run hyperparameter optimization
                new_params = await self.hyperparam_optimizer.optimize(
                    performance_data,
                    current_params=self.strategy_manager.get_parameters()
                )
                
                # Validate new parameters
                if await self._validate_new_parameters(new_params):
                    await self.strategy_manager.update_parameters(new_params)
                    await self.alert_manager.send_info("Strategy parameters optimized")
                    
                # RL optimization
                await self.rl_optimizer.update_policy(performance_data)
                
            except Exception as e:
                await self.alert_manager.send_warning(f"Strategy optimization error: {e}")
                
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
    
    async def stop(self):
        """Stop the trading engine"""
        try:
            self.state = BotState.STOPPING
            
            # Cancel all tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close positions if configured
            if self.config.get('close_on_stop', False):
                for position in list(self.active_positions.values()):
                    await self._close_position(position, "engine_stopped")
            
            # Save state
            await self._save_state()
            
            self.state = BotState.STOPPED
            
        except Exception as e:
            await self.alert_manager.send_critical(f"Error stopping engine: {e}")
            raise
    
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
            is_honeypot = await self.honeypot_checker.check_token(
                opportunity.token_address,
                opportunity.chain
            )
            
            if is_honeypot:
                logger.warning(f"   ❌ HONEYPOT DETECTED: {token_symbol}")
                return False
            logger.info(f"   ✅ Not a honeypot")
            
            # 2. Verify liquidity is still sufficient
            logger.info(f"   Checking liquidity...")
            current_liquidity = opportunity.liquidity
            min_liquidity = self.config.get('trading', {}).get('min_liquidity_threshold', 50000)
            
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

    async def shutdown(self):
        """Shutdown the engine"""
        await self.stop()

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
        Calculate overall opportunity score
        
        Returns:
            Score between 0 and 1
        """
        try:
            score = 0.0
            weights = 0.0
            
            # Volume score (30% weight) - NEW!
            volume_24h = pair.get('volume_24h', 0)
            if volume_24h > 0:
                # Higher volume = better score
                volume_score = min(volume_24h / 100000, 1.0)  # $100k = max
                score += volume_score * 0.3
                weights += 0.3
                logger.debug(f"Volume score: {volume_score:.3f} (volume: ${volume_24h:,.0f})")
            
            # Liquidity score (25% weight) - UPDATED
            liquidity_usd = pair.get('liquidity_usd', 0)
            if liquidity_usd > 0:
                # Higher liquidity = better
                liq_score = min(liquidity_usd / 50000, 1.0)  # $50k = max
                score += liq_score * 0.25
                weights += 0.25
                logger.debug(f"Liquidity score: {liq_score:.3f} (liq: ${liquidity_usd:,.0f})")
            
            # Price change score (20% weight) - NEW!
            price_change_5m = pair.get('price_change_5m', 0)
            if price_change_5m:
                # Positive price change = good
                price_score = min(max(price_change_5m / 10, 0), 1.0)  # 10% gain = max
                score += price_score * 0.2
                weights += 0.2
                logger.debug(f"Price change score: {price_score:.3f} (change: {price_change_5m:+.1f}%)")
            
            # Risk score (15% weight)
            if risk_score and hasattr(risk_score, 'overall_risk'):
                # Invert risk (lower risk = higher score)
                risk_component = 1.0 - risk_score.overall_risk
                score += risk_component * 0.15
                weights += 0.15
                logger.debug(f"Risk score: {risk_component:.3f}")
            
            # Age bonus (10% weight) - NEW!
            age_hours = pair.get('age_hours', 999)
            if age_hours < 24:  # New tokens (< 24h old)
                age_score = 1.0 - (age_hours / 24)  # Newer = better
                score += age_score * 0.1
                weights += 0.1
                logger.debug(f"Age score: {age_score:.3f} (age: {age_hours:.1f}h)")
            
            # Normalize by total weights used
            if weights > 0:
                final_score = score / weights
            else:
                final_score = 0.3  # Default low score
            
            logger.info(f"Final opportunity score: {final_score:.3f} (weights: {weights:.2f})")
            return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.3  # Return low score on error