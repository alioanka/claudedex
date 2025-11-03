# ClaudeDex: Key Files, Classes & Methods Map


## analysis/__init__.py

- **Classes**: —
- **Functions**: —

## analysis/dev_analyzer.py

- **Classes**: DeveloperRisk, ProjectStatus, DeveloperProfile, ProjectAnalysis, DeveloperAnalyzer
- **Functions**: —

## analysis/liquidity_monitor.py

- **Classes**: LiquidityEvent, SlippageEstimate, LiquidityMonitor
- **Functions**: —

## analysis/market_analyzer.py

- **Classes**: MarketCondition, TrendAnalysis, CorrelationMatrix, MarketAnalyzer
- **Functions**: —

## analysis/pump_predictor.py

- **Classes**: PumpSignal, PumpPattern, PumpPredictorAnalysis
- **Functions**: —

## analysis/rug_detector.py

- **Classes**: RugDetector
- **Functions**: —

## analysis/smart_contract_analyzer.py

- **Classes**: VulnerabilityLevel, ContractType, Vulnerability, ContractAnalysis, SmartContractAnalyzer
- **Functions**: —

## analysis/token_scorer.py

- **Classes**: TokenScore, ScoringWeights, TokenScorer
- **Functions**: —

## config/__init__.py

- **Classes**: —
- **Functions**: —

## config/config_manager.py

- **Classes**: ConfigType, ConfigSource, ConfigChange, TradingConfig, SecurityConfig, DatabaseConfig, APIConfig, MonitoringConfig, MLModelsConfig, RiskManagementConfig, PortfolioConfig, ConfigManager
- **Functions**: —

## config/consecutive_losses_config.py

- **Classes**: —
- **Functions**: get_block_duration, get_position_size_multiplier

## config/settings.py

- **Classes**: Environment, ChainConfig, Settings
- **Functions**: —

## config/validation.py

- **Classes**: ValidationResult, ConfigValidator
- **Functions**: validate_trading_config, validate_security_config, validate_config_at_startup

## core/__init__.py

- **Classes**: —
- **Functions**: —

## core/decision_maker.py

- **Classes**: TradingDecision, RiskScore, TradingOpportunity, StrategyType, DecisionMaker
- **Functions**: —

## core/engine.py

- **Classes**: BotState, TradingOpportunity, ClosedPositionRecord, TradingBotEngine
- **Functions**: —

## core/event_bus.py

- **Classes**: EventType, Event, EventSubscription, EventBus, EventLogger, EventAggregator
- **Functions**: —

## core/pattern_analyzer.py

- **Classes**: PatternType, Pattern, TrendInfo, SupportResistance, PatternAnalyzer
- **Functions**: —

## core/portfolio_manager.py

- **Classes**: Position, PortfolioMetrics, AllocationStrategy, PortfolioManager
- **Functions**: —

## core/risk_manager.py

- **Classes**: RiskLevel, RiskScore, CircuitBreakerMetrics, TradingOpportunity, Position, RiskManager
- **Functions**: —

## create_tree.py

- **Classes**: —
- **Functions**: create_tree, main

## data/__init__.py

- **Classes**: —
- **Functions**: —

## data/collectors/__init__.py

- **Classes**: —
- **Functions**: —

## data/collectors/chain_data.py

- **Classes**: TokenInfo, ContractInfo, LiquidityInfo, TransactionInfo, ChainDataCollector
- **Functions**: —

## data/collectors/dexscreener.py

- **Classes**: TokenPair, DexScreenerCollector
- **Functions**: —

## data/collectors/honeypot_checker.py

- **Classes**: HoneypotChecker
- **Functions**: is_valid_solana_address

## data/collectors/mempool_monitor.py

- **Classes**: PendingTransaction, SandwichRisk, MEVOpportunity, MempoolMonitor
- **Functions**: —

## data/collectors/social_data.py

- **Classes**: SentimentLevel, SocialPlatform, SocialMetrics, InfluencerMention, SocialDataCollector
- **Functions**: —

## data/collectors/token_sniffer.py

- **Classes**: TokenRisk, TokenFlag, TokenAnalysis, TokenMetrics, TokenSniffer
- **Functions**: —

## data/collectors/volume_analyzer.py

- **Classes**: VolumePattern, VolumeProfile, TradeCluster, VolumeAnalyzer
- **Functions**: —

## data/collectors/whale_tracker.py

- **Classes**: WhaleTracker
- **Functions**: —

## data/processors/__init__.py

- **Classes**: —
- **Functions**: —

## data/processors/aggregator.py

- **Classes**: DataAggregator
- **Functions**: —

## data/processors/feature_extractor.py

- **Classes**: FeatureExtractor
- **Functions**: —

## data/processors/normalizer.py

- **Classes**: DataType, NormalizationConfig, DataNormalizer
- **Functions**: —

## data/processors/validator.py

- **Classes**: ValidationLevel, DataField, ValidationResult, ValidationRule, DataValidator
- **Functions**: —

## data/storage/__init__.py

- **Classes**: —
- **Functions**: —

## data/storage/cache.py

- **Classes**: CacheManager
- **Functions**: —

## data/storage/database.py

- **Classes**: DatabaseManager
- **Functions**: —

## data/storage/models.py

- **Classes**: TradeSide, TradeStatus, PositionStatus, AlertPriority, Trade, Position, MarketData, Alert, TokenAnalysis, PerformanceMetrics, WhaleWallet, MEVTransaction, SystemLog
- **Functions**: create_all_tables, drop_all_tables

## generate_file_tree.py

- **Classes**: —
- **Functions**: list_tree_lines, write_tree_file, main

## main.py

- **Classes**: HealthChecker, TradingBotApplication
- **Functions**: setup_multichain_config, setup_logger, parse_arguments

## main_log_analyzer.py

- **Classes**: —
- **Functions**: analyze_main_log

## ml/__init__.py

- **Classes**: —
- **Functions**: —

## ml/models/__init__.py

- **Classes**: —
- **Functions**: —

## ml/models/ensemble_model.py

- **Classes**: PredictionResult, LSTMPricePredictor, TransformerPredictor, EnsemblePredictor
- **Functions**: —

## ml/models/pump_predictor.py

- **Classes**: PumpPredictor
- **Functions**: —

## ml/models/rug_classifier.py

- **Classes**: RugClassifier
- **Functions**: —

## ml/models/volume_validator.py

- **Classes**: VolumeValidationResult, VolumeValidatorML
- **Functions**: —

## ml/optimization/__init__.py

- **Classes**: —
- **Functions**: —

## ml/optimization/hyperparameter.py

- **Classes**: HyperparameterSpace, HyperparameterOptimizer
- **Functions**: —

## ml/optimization/reinforcement.py

- **Classes**: State, Action, Experience, RLOptimizer
- **Functions**: —

## ml/training/__init__.py

- **Classes**: —
- **Functions**: —

## monitoring/__init__.py

- **Classes**: —
- **Functions**: —

## monitoring/alerts.py

- **Classes**: AlertPriority, AlertType, NotificationChannel, Alert, AlertRule, ChannelConfig, AlertsSystem, AlertManager
- **Functions**: escape_markdown

## monitoring/dashboard.py

- **Classes**: DashboardSection, DashboardData, ChartData, Dashboard, DashboardManager
- **Functions**: —

## monitoring/enhanced_dashboard.py

- **Classes**: DashboardEndpoints
- **Functions**: —

## monitoring/logger.py

- **Classes**: IgnorePortScannersFilter, MetricType, TradeRecord, PerformanceSnapshot, PerformanceTracker, StructuredLogger, JsonFormatter, ColoredFormatter, StandardFormatter, TradeFormatter
- **Functions**: get_logger, log_trade_entry, log_trade_exit, log_portfolio_update

## monitoring/performance.py

- **Classes**: MetricType, TradeRecord, PerformanceSnapshot, PerformanceTracker, StructuredLogger, JsonFormatter, ColoredFormatter, StandardFormatter, TradeFormatter
- **Functions**: —

## scripts/analyze_strategy.py

- **Classes**: —
- **Functions**: —

## scripts/check_balance.py

- **Classes**: —
- **Functions**: —

## scripts/close_all_positions.py

- **Classes**: —
- **Functions**: —

## scripts/daily_report.py

- **Classes**: —
- **Functions**: —

## scripts/dev_autofix_imports.py

- **Classes**: —
- **Functions**: find_py_files, get_module_path, parse_imports, collect_annotation_names, build_local_name_index, insertion_index, ensure_future_annotations, ensure_import_lines, find_external_modules, is_local_module, normalize_req_line, main

## scripts/emergency_stop.py

- **Classes**: —
- **Functions**: —

## scripts/export_trades.py

- **Classes**: —
- **Functions**: —

## scripts/fix_illegal_relatives.py

- **Classes**: Replacement
- **Functions**: iter_py_files, path_to_module, module_exists, top_level_package, compute_absolute_from_relative, rewrite_import_line, fix_file, main

## scripts/generate_report.py

- **Classes**: —
- **Functions**: —

## scripts/generate_solana_wallet.py

- **Classes**: —
- **Functions**: generate_wallet, import_wallet, main

## scripts/health_check.py

- **Classes**: —
- **Functions**: —

## scripts/init_config.py

- **Classes**: —
- **Functions**: init_config

## scripts/migrate_database.py

- **Classes**: —
- **Functions**: —

## scripts/optimize_db.py

- **Classes**: —
- **Functions**: —

## scripts/overnight_summary.py

- **Classes**: —
- **Functions**: —

## scripts/post_update_check.py

- **Classes**: —
- **Functions**: —

## scripts/reset_db_sequences.py

- **Classes**: —
- **Functions**: —

## scripts/reset_nonce.py

- **Classes**: —
- **Functions**: —

## scripts/retrain_models.py

- **Classes**: —
- **Functions**: —

## scripts/run_tests.py

- **Classes**: TestRunner
- **Functions**: main

## scripts/security_audit.py

- **Classes**: —
- **Functions**: —

## scripts/setup_database.py

- **Classes**: —
- **Functions**: —

## scripts/solana_wallet_balance.py

- **Classes**: —
- **Functions**: —

## scripts/strategy_analysis.py

- **Classes**: —
- **Functions**: —

## scripts/test_alerts.py

- **Classes**: —
- **Functions**: —

## scripts/test_apis.py

- **Classes**: —
- **Functions**: —

## scripts/test_solana_setup.py

- **Classes**: —
- **Functions**: —

## scripts/train_models.py

- **Classes**: —
- **Functions**: —

## scripts/update_blacklists.py

- **Classes**: —
- **Functions**: —

## scripts/update_models.py

- **Classes**: —
- **Functions**: —

## scripts/verify_claudedex_plus.py

- **Classes**: ExpectedItem, FoundMethod, FileReport, _CallCollector, _AstCollector
- **Functions**: _read_text, _norm, _ann_to_str, _params_from_args, _visibility_from_name, _resolve_filename_under_root, parse_expectations_from_md, _parse_tree_block_lines, parse_structure_md, collect_file_symbols, _compare_signature, _check_visibility, _check_wrapper, generate_reports, render_markdown, main

## scripts/verify_claudedex_plus2.py

- **Classes**: ExpectedItem, FoundMethod, FileReport, _CallCollector, _AstCollector
- **Functions**: _read_text, _norm, _ann_to_str, _params_from_args, _visibility_from_name, _resolve_filename_under_root, parse_expectations_from_md, _parse_tree_block_lines, parse_structure_md, collect_file_symbols, _compare_signature, _check_visibility, _check_wrapper, _enforce_interfaces, generate_reports, render_markdown, main

## scripts/verify_claudedex_plus3.py

- **Classes**: ExpectedItem, FoundMethod, FileReport, _CallCollector, _AstCollector
- **Functions**: _read_text, _norm, _ann_to_str, _params_from_args, _visibility_from_name, _resolve_filename_under_root, _split_params_safely, parse_expectations_from_md, parse_structure_md, collect_file_symbols, _compare_signature, _enforce_interfaces, _closest, generate_reports, main

## scripts/weekly_report.py

- **Classes**: —
- **Functions**: —

## scripts/withdraw_funds.py

- **Classes**: —
- **Functions**: —

## security/__init__.py

- **Classes**: —
- **Functions**: —

## security/api_security.py

- **Classes**: APISecurityManager
- **Functions**: —

## security/audit_logger.py

- **Classes**: AuditEventType, AuditSeverity, AuditStatus, AuditEvent, AuditLogger
- **Functions**: —

## security/encryption.py

- **Classes**: EncryptionManager
- **Functions**: —

## security/wallet_security.py

- **Classes**: WalletType, SecurityLevel, WalletConfig, TransactionApproval, WalletSecurityManager
- **Functions**: —

## setup.py

- **Classes**: —
- **Functions**: read_requirements

## setup_env_keys.py

- **Classes**: —
- **Functions**: generate_encryption_key, generate_jwt_secret, encrypt_private_key, main

## tests/__init__.py

- **Classes**: —
- **Functions**: —

## tests/conftest.py

- **Classes**: —
- **Functions**: event_loop, risk_manager, mock_dex_api, mock_market_data, sample_trading_opportunity, sample_position, benchmark_data, create_mock_token, create_mock_trade

## tests/fixtures/mock_data.py

- **Classes**: MockDataGenerator
- **Functions**: —

## tests/fixtures/test_helpers.py

- **Classes**: TestHelpers
- **Functions**: —

## tests/integration/test_data_integration.py

- **Classes**: TestDataIntegration
- **Functions**: —

## tests/integration/test_dexscreener.py

- **Classes**: —
- **Functions**: —

## tests/integration/test_ml_integration.py

- **Classes**: TestMLIntegration
- **Functions**: —

## tests/integration/test_trading_integration.py

- **Classes**: TestTradingIntegration
- **Functions**: —

## tests/performance/test_performance.py

- **Classes**: TestPerformance
- **Functions**: —

## tests/security/test_security.py

- **Classes**: TestSecurity
- **Functions**: —

## tests/smoke/test_smoke.py

- **Classes**: TestSmoke
- **Functions**: —

## tests/test_all.py

- **Classes**: —
- **Functions**: run_all_tests

## tests/unit/test_engine.py

- **Classes**: TestTradingBotEngine
- **Functions**: —

## tests/unit/test_risk_manager.py

- **Classes**: TestRiskManager
- **Functions**: —

## trade_analyzer.py

- **Classes**: —
- **Functions**: analyze_trades

## trading/__init__.py

- **Classes**: —
- **Functions**: —

## trading/chains/__init__.py

- **Classes**: —
- **Functions**: —

## trading/chains/solana/__init__.py

- **Classes**: —
- **Functions**: —

## trading/chains/solana/jupiter_executor.py

- **Classes**: JupiterExecutor
- **Functions**: —

## trading/chains/solana/solana_client.py

- **Classes**: SolanaClient
- **Functions**: —

## trading/chains/solana/spl_token_handler.py

- **Classes**: SPLTokenHandler
- **Functions**: get_token_decimals

## trading/executors/__init__.py

- **Classes**: —
- **Functions**: —

## trading/executors/base_executor.py

- **Classes**: BaseExecutor, TradeOrder, ExecutionResult, ExecutionRoute, TradeExecutor
- **Functions**: —

## trading/executors/direct_dex.py

- **Classes**: DEXQuote, DirectDEXExecutor
- **Functions**: —

## trading/executors/mev_protection.py

- **Classes**: MEVProtectionLevel, AttackType, MEVThreat, ProtectedTransaction, MEVProtectionLayer
- **Functions**: —

## trading/executors/toxisol_api.py

- **Classes**: ToxiSolRoute, ToxiSolQuote, ToxiSolAPIExecutor
- **Functions**: —

## trading/orders/__init__.py

- **Classes**: —
- **Functions**: —

## trading/orders/order_manager.py

- **Classes**: OrderStatus, OrderType, OrderSide, ExecutionStrategy, Order, OrderBook, Fill, OrderManager, ExecutionEngine, OrderRiskMonitor, SettlementProcessor
- **Functions**: build_order, create_solana_order, create_evm_order

## trading/orders/position_tracker.py

- **Classes**: PositionStatus, PositionType, RiskLevel, Position, PortfolioSnapshot, PerformanceMetrics, PositionTracker
- **Functions**: —

## trading/strategies/__init__.py

- **Classes**: StrategyManager
- **Functions**: —

## trading/strategies/ai_strategy.py

- **Classes**: AIStrategy
- **Functions**: —

## trading/strategies/base_strategy.py

- **Classes**: SignalType, SignalStrength, StrategyState, TradingSignal, StrategyPerformance, BaseStrategy
- **Functions**: —

## trading/strategies/momentum.py

- **Classes**: MomentumType, TimeFrame, MomentumMetrics, MomentumStrategy
- **Functions**: —

## trading/strategies/scalping copy.py

- **Classes**: —
- **Functions**: —

## trading/strategies/scalping.py

- **Classes**: ScalpingSignal, ScalpingOpportunity, ScalpingStrategy
- **Functions**: —

## utils/__init__.py

- **Classes**: —
- **Functions**: —

## utils/constants.py

- **Classes**: Chain, DEX, TradingMode, SignalStrength, MarketCondition, OrderType, OrderStatus
- **Functions**: —

## utils/errors.py

- **Classes**: NetworkError, RPCError, QuoteError, APIRateLimitError, ExecutionError, InsufficientBalanceError, SlippageExceededError, ConfirmationTimeout, NonceError, ContractError, ABIError, DecodeError, HoneypotDetected, ConfigurationError, ValidationError, RiskLimitError, CircuitBreakerTripped, PositionLimitError, DatabaseError, DataIntegrityError, AnalysisError, ModelError
- **Functions**: —

## utils/helpers.py

- **Classes**: TTLCache
- **Functions**: retry_async, measure_time, rate_limit, is_valid_address, normalize_address, wei_to_ether, ether_to_wei, format_token_amount, to_base_unit, calculate_percentage_change, calculate_slippage, round_to_significant_digits, calculate_profit_loss, calculate_moving_average, calculate_ema, get_timestamp, get_timestamp_ms, format_timestamp, parse_timeframe, is_market_hours, format_number, format_currency, truncate_string, safe_json_loads, deep_merge_dicts, validate_token_symbol, validate_chain_id, sanitize_input, generate_signature, hash_data, mask_sensitive_data, chunk_list

## verify_references.py

- **Classes**: DefInfo, ClassInfo, ModuleInfo, CallSite, Problem, PyIndexer, CallCollector, CollMap, Analyzer
- **Functions**: is_py, is_cfg, to_module, file_for_module, extract_sig, arg_summary, dotted_attr, read_file, resolve_relative, pretty_sig, parse_args, main

## xref_symbol_db.py

- **Classes**: Problem
- **Functions**: is_py, to_module, read_file, dotted_attr, arg_summary, scan_files, build_file_tree_md, extract_sig, build_symbol_db, write_symbol_md, import_table, ensure_in_db, class_has_method, synth_init_sig, get_sig, pretty_sig, check_sig, target_kind_from_sig, verify_with_db, parse_args, main

## Parser Notes
