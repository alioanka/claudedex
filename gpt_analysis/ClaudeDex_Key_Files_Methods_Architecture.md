# ClaudeDex — Key Files, Methods & Architecture Reference

## Most-Referenced Internal Modules (top 40)
- utils.helpers
- data.storage.database
- utils.constants
- trading.orders.order_manager
- data.storage.cache
- ml.models.ensemble_model
- config.config_manager
- data.collectors.dexscreener
- security.encryption
- core.engine
- core.risk_manager
- security.wallet_security
- core.event_bus
- trading.executors.base_executor
- ml.models.rug_classifier
- monitoring.alerts
- ml.models.pump_predictor
- data.collectors.honeypot_checker
- trading.chains.solana.jupiter_executor
- core.portfolio_manager
- monitoring.logger
- trading.chains.solana
- analysis.market_analyzer
- core.pattern_analyzer
- data.collectors.chain_data
- data.collectors.whale_tracker
- trading.chains.solana.solana_client
- trading.orders.position_tracker
- security.audit_logger
- monitoring.enhanced_dashboard
- analysis.liquidity_monitor
- analysis.rug_detector
- core.decision_maker
- data.collectors.mempool_monitor
- data.collectors.social_data
- ml.optimization.hyperparameter
- ml.optimization.reinforcement
- monitoring.performance
- trading.strategies
- ml.models.volume_validator

## Directory Overview
- **core/** — Engine, decision-making, portfolio & risk.
- **trading/** — Executors, orders, strategies per chain.
- **data/** — Collectors, processors, storage models & DB.
- **ml/** — Models and learning components.
- **analysis/** — Token/liquidity/rug/pump analysis.
- **monitoring/** — Logging, dashboards, perf metrics.
- **dashboard/** — Static assets & templates for UI.
- **config/** — Config manager, settings, validation.
- **security/** — Key management, encryption, audit.
- **utils/** — Shared constants & helpers.
- **scripts/** — Admin/ops scripts: deploy, health, backup.
- **tests/** — Unit, integration, performance, security tests.

## Key Classes & Functions (by file)
### core/engine.py
- Classes: BotState, TradingOpportunity, ClosedPositionRecord, TradingBotEngine

### core/decision_maker.py
- Classes: TradingDecision, RiskScore, TradingOpportunity, StrategyType, DecisionMaker

### core/risk_manager.py
- Classes: RiskLevel, RiskScore, TradingOpportunity, Position, RiskManager

### core/portfolio_manager.py
- Classes: Position, PortfolioMetrics, AllocationStrategy, PortfolioManager

### trading/executors/base_executor.py
- Classes: BaseExecutor, TradeOrder, ExecutionResult, ExecutionRoute, TradeExecutor

### trading/executors/direct_dex.py
- Classes: DEXQuote, DirectDEXExecutor

### trading/executors/mev_protection.py
- Classes: MEVProtectionLevel, AttackType, MEVThreat, ProtectedTransaction, MEVProtectionLayer

### trading/orders/order_manager.py
- Classes: OrderStatus, OrderType, OrderSide, ExecutionStrategy, Order, OrderBook, Fill, OrderManager, ExecutionEngine, OrderRiskMonitor, SettlementProcessor
- Functions: build_order, create_solana_order, create_evm_order

### trading/orders/position_tracker.py
- Classes: PositionStatus, PositionType, RiskLevel, Position, PortfolioSnapshot, PerformanceMetrics, PositionTracker

### trading/strategies/base_strategy.py
- Classes: SignalType, SignalStrength, StrategyState, TradingSignal, StrategyPerformance, BaseStrategy

### trading/strategies/scalping.py
- Classes: ScalpingSignal, ScalpingOpportunity, ScalpingStrategy

### trading/strategies/momentum.py
- Classes: MomentumType, TimeFrame, MomentumSignal, MomentumMetrics, MomentumStrategy

### trading/chains/solana/solana_client.py
- Classes: SolanaClient

### trading/chains/solana/jupiter_executor.py
- Classes: JupiterExecutor

### data/collectors/mempool_monitor.py
- Classes: PendingTransaction, SandwichRisk, MEVOpportunity, MempoolMonitor

### data/collectors/dexscreener.py
- Classes: TokenPair, DexScreenerCollector

### data/storage/models.py
- Classes: TradeSide, TradeStatus, PositionStatus, AlertPriority, Trade, Position, MarketData, Alert, TokenAnalysis, PerformanceMetrics, WhaleWallet, MEVTransaction, SystemLog
- Functions: create_all_tables, drop_all_tables

### monitoring/enhanced_dashboard.py
- Classes: DashboardEndpoints

### monitoring/logger.py
- Classes: MetricType, TradeRecord, PerformanceSnapshot, PerformanceTracker, StructuredLogger, JsonFormatter, ColoredFormatter, StandardFormatter, TradeFormatter

### config/config_manager.py
- Classes: ConfigType, ConfigSource, ConfigChange, TradingConfig, SecurityConfig, DatabaseConfig, APIConfig, MonitoringConfig, MLModelsConfig, RiskManagementConfig, PortfolioConfig, ConfigManager

### config/settings.py
- Classes: Environment, ChainConfig, Settings

### config/validation.py
- Classes: ValidationResult, ConfigValidator
- Functions: validate_trading_config, validate_security_config

### security/wallet_security.py
- Classes: WalletType, SecurityLevel, WalletConfig, TransactionApproval, WalletSecurityManager

### utils/helpers.py
- Classes: TTLCache
- Functions: retry_async, measure_time, rate_limit, is_valid_address, normalize_address, wei_to_ether, ether_to_wei, format_token_amount, to_base_unit, calculate_percentage_change, calculate_slippage, round_to_significant_digits, calculate_profit_loss, calculate_moving_average, calculate_ema, get_timestamp, get_timestamp_ms, format_timestamp, parse_timeframe, is_market_hours, format_number, format_currency, truncate_string, safe_json_loads, deep_merge_dicts, validate_token_symbol, validate_chain_id, sanitize_input, generate_signature, hash_data, mask_sensitive_data, chunk_list

### utils/constants.py
- Classes: Chain, DEX, TradingMode, SignalStrength, MarketCondition, OrderType, OrderStatus

## Environment Variables Referenced
- DB_USER ×16
- DB_HOST ×15
- DB_NAME ×15
- DB_PASSWORD ×15
- DB_PORT ×15
- ETH_RPC_URL ×6
- DEXSCREENER_API_KEY ×3
- MAX_SLIPPAGE ×3
- TELEGRAM_BOT_TOKEN ×3
- SOLANA_PRIVATE_KEY ×2
- SOLANA_RPC_URL ×2
- DISCORD_WEBHOOK_URL ×2
- ENABLED_CHAINS ×2
- ENCRYPTION_KEY ×2
- LOG_LEVEL ×2
- MAX_POSITION_SIZE_PERCENT ×2
- REDIS_URL ×2
- TELEGRAM_CHAT_ID ×2
- TWITTER_API_KEY ×2
- WALLET_ADDRESS ×2
- REDIS_HOST ×2
- REDIS_PORT ×2
- GOPLUS_API_KEY ×2
- ARBITRUM_MIN_LIQUIDITY ×1
- BACKTEST_INITIAL_BALANCE ×1
- BASE_MIN_LIQUIDITY ×1
- BSC_MIN_LIQUIDITY ×1
- CHAIN_ID ×1
- DEFAULT_CHAIN ×1
- DISCOVERY_INTERVAL_SECONDS ×1
- DRY_RUN ×1
- ETHEREUM_MIN_LIQUIDITY ×1
- JUPITER_MAX_SLIPPAGE_BPS ×1
- MAX_DAILY_LOSS_PERCENT ×1
- MAX_DRAWDOWN_PERCENT ×1
- MAX_GAS_PRICE ×1
- MAX_PAIRS_PER_CHAIN ×1
- MIN_TRADE_SIZE_USD ×1
- ML_MIN_CONFIDENCE ×1
- ML_RETRAIN_INTERVAL_HOURS ×1
- POLYGON_MIN_LIQUIDITY ×1
- PRIORITY_GAS_MULTIPLIER ×1
- PRIVATE_KEY ×1
- SOLANA_ENABLED ×1
- SOLANA_MAX_AGE_HOURS ×1
- SOLANA_MIN_LIQUIDITY ×1
- SOLANA_MIN_VOLUME ×1
- TWITTER_API_SECRET ×1
- WEB3_BACKUP_PROVIDER_1 ×1
- WEB3_BACKUP_PROVIDER_2 ×1
