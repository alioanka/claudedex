# ClaudeDex Verifier Report (PLUS)

## Summary
- Files checked: **76**
- Total missing items: **9**
- Files with in-file duplicates: **3**

## Cross-file duplicate top-level functions
- `main`: main.py, scripts\health_check.py, scripts\run_tests.py

## Per-file details
### `analysis\dev_analyzer.py` — ✅ OK
- Exists: **True**
- Classes: DeveloperAnalyzer, DeveloperProfile, DeveloperRisk, ProjectAnalysis, ProjectStatus
- Methods: DeveloperAnalyzer.__init__, DeveloperAnalyzer._analyze_code_quality, DeveloperAnalyzer._analyze_community_metrics, DeveloperAnalyzer._analyze_previous_projects, DeveloperAnalyzer._calculate_project_score, DeveloperAnalyzer._calculate_reputation, DeveloperAnalyzer._calculate_transparency_score, DeveloperAnalyzer._check_liquidity_status, DeveloperAnalyzer._check_social_presence, DeveloperAnalyzer._create_rugger_profile, DeveloperAnalyzer._create_trusted_profile, DeveloperAnalyzer._determine_project_status, DeveloperAnalyzer._determine_risk_level, DeveloperAnalyzer._find_aliases, DeveloperAnalyzer._find_associated_wallets

### `analysis\liquidity_monitor.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\analysis\liquidity_monitor.py: unexpected indent (<unknown>, line 498)

### `analysis\market_analyzer.py` — ✅ OK
- Exists: **True**
- Classes: CorrelationMatrix, MarketAnalyzer, MarketCondition, TrendAnalysis
- Methods: MarketAnalyzer.__init__, MarketAnalyzer._calculate_market_metrics, MarketAnalyzer._calculate_market_risk, MarketAnalyzer._calculate_market_volatility, MarketAnalyzer._detect_volume_spikes, MarketAnalyzer._determine_market_trend, MarketAnalyzer._get_regime_recommendation, MarketAnalyzer._identify_correlation_clusters, MarketAnalyzer.analyze_market, MarketAnalyzer.analyze_market_conditions, MarketAnalyzer.analyze_volume_patterns, MarketAnalyzer.calculate_correlations, MarketAnalyzer.calculate_market_sentiment, MarketAnalyzer.detect_market_regime, MarketAnalyzer.find_market_inefficiencies

### `analysis\pump_predictor.py` — ✅ OK
- Exists: **True**
- Classes: PumpPattern, PumpPredictorAnalysis, PumpSignal
- Methods: PumpPredictorAnalysis.__init__, PumpPredictorAnalysis._analyze_pump_patterns, PumpPredictorAnalysis._assess_pump_risk, PumpPredictorAnalysis._calculate_entry_exit_points, PumpPredictorAnalysis._calculate_expected_metrics, PumpPredictorAnalysis._combine_signals, PumpPredictorAnalysis._detect_accumulation_phase, PumpPredictorAnalysis._detect_price_spike, PumpPredictorAnalysis._detect_volume_spike, PumpPredictorAnalysis._determine_pump_status, PumpPredictorAnalysis._gather_prediction_data, PumpPredictorAnalysis._get_default_signal, PumpPredictorAnalysis._get_signal_strength, PumpPredictorAnalysis._identify_pump_type, PumpPredictorAnalysis._process_significant_signal

### `analysis\rug_detector.py` — ✅ OK
- Exists: **True**
- Classes: RugDetector
- Methods: RugDetector.__init__, RugDetector._calculate_contract_risk_score, RugDetector._calculate_holder_risk_score, RugDetector._check_lock_provider, RugDetector._check_ownership, RugDetector._check_proxy_pattern, RugDetector._check_risky_functions, RugDetector._check_verification, RugDetector._get_token_holders, RugDetector._initialize_web3, RugDetector.analyze_contract, RugDetector.analyze_holder_distribution, RugDetector.analyze_token, RugDetector.calculate_rug_score, RugDetector.check_contract_vulnerabilities

### `analysis\smart_contract_analyzer.py` — ✅ OK
- Exists: **True**
- Classes: ContractAnalysis, ContractType, SmartContractAnalyzer, Vulnerability, VulnerabilityLevel
- Methods: SmartContractAnalyzer.__init__, SmartContractAnalyzer._analyze_fees, SmartContractAnalyzer._analyze_permissions, SmartContractAnalyzer._analyze_proxy_pattern, SmartContractAnalyzer._analyze_storage, SmartContractAnalyzer._analyze_vulnerabilities, SmartContractAnalyzer._calculate_risk_score, SmartContractAnalyzer._check_upgrade_mechanism, SmartContractAnalyzer._check_verification, SmartContractAnalyzer._create_malicious_analysis, SmartContractAnalyzer._detect_changes, SmartContractAnalyzer._extract_events, SmartContractAnalyzer._extract_functions, SmartContractAnalyzer._extract_modifiers, SmartContractAnalyzer._fetch_contract_data

### `analysis\token_scorer.py` — ✅ OK
- Exists: **True**
- Classes: ScoringWeights, TokenScore, TokenScorer
- Methods: TokenScorer.__init__, TokenScorer._adjust_score_for_risk, TokenScorer._calculate_category_scores, TokenScorer._calculate_confidence, TokenScorer._calculate_innovation_score, TokenScorer._calculate_opportunity_score, TokenScorer._calculate_risk_score, TokenScorer._calculate_weighted_score, TokenScorer._determine_grade, TokenScorer._gather_scoring_data, TokenScorer._generate_comparison_recommendation, TokenScorer._generate_recommendation, TokenScorer._get_default_score, TokenScorer._get_developer_activity, TokenScorer._get_holder_distribution

### `config\__init__.py` — ✅ OK
- Exists: **True**

### `config\config_manager.py` — ✅ OK
- Exists: **True**
- Classes: APIConfig, ConfigChange, ConfigManager, ConfigSource, ConfigType, DatabaseConfig, MLModelsConfig, MonitoringConfig, RiskManagementConfig, SecurityConfig
- Methods: ConfigManager.__init__, ConfigManager._auto_reload_loop, ConfigManager._check_for_changes, ConfigManager._decrypt_sensitive_values, ConfigManager._encrypt_sensitive_values, ConfigManager._is_float, ConfigManager._load_all_configs, ConfigManager._load_config, ConfigManager._load_config_from_database, ConfigManager._load_config_from_env, ConfigManager._load_config_from_file, ConfigManager._load_default_config, ConfigManager._notify_config_watchers, ConfigManager._persist_config, ConfigManager._reload_config

### `config\settings.py` — ✅ OK
- Exists: **True**
- Classes: ChainConfig, Environment, Settings
- Methods: ChainConfig.__init__, Settings.get_chain_config, Settings.get_current_features, Settings.get_dex_config, Settings.get_environment_info, Settings.get_gas_config, Settings.get_stablecoin_address, Settings.is_feature_enabled, Settings.is_token_blacklisted, Settings.validate_environment

### `config\validation.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\config\validation.py: unterminated string literal (detected at line 368) (<unknown>, line 368)

### `core\__init__.py` — ✅ OK
- Exists: **True**

### `core\decision_maker.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\core\decision_maker.py: invalid syntax (<unknown>, line 736)

### `core\engine.py` — ✅ OK
- Exists: **True**
- Classes: BotState, TradingBotEngine, TradingOpportunity
- Methods: TradingBotEngine.__init__, TradingBotEngine._analyze_opportunity, TradingBotEngine._check_exit_conditions, TradingBotEngine._close_position, TradingBotEngine._execute_opportunity, TradingBotEngine._final_safety_checks, TradingBotEngine._health_check, TradingBotEngine._monitor_existing_positions, TradingBotEngine._monitor_mempool, TradingBotEngine._monitor_new_pairs, TradingBotEngine._monitor_performance, TradingBotEngine._optimize_strategies, TradingBotEngine._process_opportunities, TradingBotEngine._retrain_models, TradingBotEngine._setup_event_handlers

### `core\event_bus.py` — ✅ OK
- Exists: **True**
- **Duplicates** (2): class:EventBus::subscribe x2; class:EventBus::unsubscribe x2
- Classes: Event, EventAggregator, EventBus, EventLogger, EventSubscription, EventType
- Methods: Event.to_dict, Event.to_json, EventAggregator.__init__, EventAggregator.add_event, EventAggregator.flush, EventBus.__init__, EventBus._add_to_history, EventBus._apply_filters, EventBus._handle_event, EventBus._process_events, EventBus.add_global_filter, EventBus.callback, EventBus.clear_dead_letter_queue, EventBus.create_event_stream, EventBus.emit

### `core\pattern_analyzer.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\core\pattern_analyzer.py: expected 'except' or 'finally' block (<unknown>, line 446)

### `core\portfolio_manager.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\core\portfolio_manager.py: expected 'except' or 'finally' block (<unknown>, line 275)

### `core\risk_manager.py` — ✅ OK
- Exists: **True**
- Classes: RiskLevel, RiskManager, RiskScore
- Methods: RiskManager.__init__, RiskManager._analyze_contract_risk, RiskManager._analyze_developer_risk, RiskManager._analyze_holder_risk, RiskManager._analyze_liquidity_risk, RiskManager._analyze_market_risk, RiskManager._analyze_social_risk, RiskManager._analyze_technical_risk, RiskManager._analyze_volume_risk, RiskManager.analyze_token, RiskManager.calculate_position_size, RiskManager.calculate_sharpe_ratio, RiskManager.calculate_sortino_ratio, RiskManager.calculate_stop_loss, RiskManager.calculate_take_profit

### `data\collectors\chain_data.py` — ✅ OK
- Exists: **True**
- Classes: ChainDataCollector, ContractInfo, LiquidityInfo, TokenInfo, TransactionInfo
- Methods: ChainDataCollector.__init__, ChainDataCollector._check_function_in_code, ChainDataCollector._check_liquidity_lock, ChainDataCollector._check_renounced_ownership, ChainDataCollector._get_contract_creation, ChainDataCollector._safe_call, ChainDataCollector._setup_connections, ChainDataCollector.analyze_contract, ChainDataCollector.check_honeypot_onchain, ChainDataCollector.estimate_transaction_cost, ChainDataCollector.get_block_number, ChainDataCollector.get_gas_price, ChainDataCollector.get_holder_distribution, ChainDataCollector.get_liquidity_info, ChainDataCollector.get_recent_transactions

### `data\collectors\dexscreener.py` — ✅ OK
- Exists: **True**
- Classes: DexScreenerCollector, TokenPair
- Methods: DexScreenerCollector.__init__, DexScreenerCollector._detect_changes, DexScreenerCollector._filter_pair, DexScreenerCollector._make_request, DexScreenerCollector._pair_to_dict, DexScreenerCollector._parse_pair, DexScreenerCollector._rate_limit, DexScreenerCollector.calculate_metrics, DexScreenerCollector.close, DexScreenerCollector.get_boosts, DexScreenerCollector.get_gainers_losers, DexScreenerCollector.get_new_pairs, DexScreenerCollector.get_pair_data, DexScreenerCollector.get_price_history, DexScreenerCollector.get_stats

### `data\collectors\honeypot_checker.py` — ✅ OK
- Exists: **True**
- Classes: HoneypotChecker
- Methods: HoneypotChecker.__init__, HoneypotChecker._calculate_verdict, HoneypotChecker._check_contract_verification, HoneypotChecker._check_dextools, HoneypotChecker._check_erc20_interface, HoneypotChecker._check_goplus, HoneypotChecker._check_honeypot_is, HoneypotChecker._check_tokensniffer, HoneypotChecker._get_chain_id, HoneypotChecker._is_blacklisted, HoneypotChecker._setup_web3_connections, HoneypotChecker.analyze_contract_code, HoneypotChecker.batch_check, HoneypotChecker.check_liquidity_locks, HoneypotChecker.check_multiple_apis

### `data\collectors\mempool_monitor.py` — ✅ OK
- Exists: **True**
- Classes: MEVOpportunity, MempoolMonitor, PendingTransaction, SandwichRisk
- Methods: MempoolMonitor.__init__, MempoolMonitor._cleanup_old_txs, MempoolMonitor._decode_swap, MempoolMonitor._detect_arbitrage, MempoolMonitor._detect_liquidations, MempoolMonitor._load_mev_bots, MempoolMonitor._process_pending_tx, MempoolMonitor._setup_connections, MempoolMonitor.add_mev_bot, MempoolMonitor.analyze_pending_tx, MempoolMonitor.check_sandwich_risk, MempoolMonitor.detect_frontrun_risk, MempoolMonitor.detect_mev_opportunities, MempoolMonitor.get_gas_prices, MempoolMonitor.get_mempool_stats

### `data\collectors\social_data.py` — ✅ OK
- Exists: **True**
- Classes: InfluencerMention, SentimentLevel, SocialDataCollector, SocialMetrics, SocialPlatform
- Methods: SocialDataCollector.__init__, SocialDataCollector._calculate_fear_index, SocialDataCollector._calculate_fomo_index, SocialDataCollector._calculate_overall_sentiment, SocialDataCollector._calculate_trending_score, SocialDataCollector._classify_sentiment, SocialDataCollector._collect_reddit_data, SocialDataCollector._collect_telegram_data, SocialDataCollector._collect_twitter_data, SocialDataCollector._count_influential_mentions, SocialDataCollector._count_negative_keywords, SocialDataCollector._get_previous_metrics, SocialDataCollector._get_trending_tokens, SocialDataCollector._load_influencers, SocialDataCollector._store_metrics

### `data\collectors\token_sniffer.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\data\collectors\token_sniffer.py: invalid syntax (<unknown>, line 716)

### `data\collectors\volume_analyzer.py` — ✅ OK
- Exists: **True**
- Classes: TradeCluster, VolumeAnalyzer, VolumePattern, VolumeProfile
- Methods: VolumeAnalyzer.__init__, VolumeAnalyzer._analyze_address_patterns, VolumeAnalyzer._analyze_dex_distribution, VolumeAnalyzer._analyze_price_impact, VolumeAnalyzer._analyze_smart_money, VolumeAnalyzer._analyze_time_distribution, VolumeAnalyzer._analyze_trade_sizes, VolumeAnalyzer._analyze_trade_timing, VolumeAnalyzer._analyze_trader_types, VolumeAnalyzer._calculate_avg_trade_size, VolumeAnalyzer._calculate_confidence, VolumeAnalyzer._calculate_time_coverage, VolumeAnalyzer._calculate_trade_similarity, VolumeAnalyzer._calculate_velocity, VolumeAnalyzer._detect_accumulation_distribution

### `data\collectors\whale_tracker.py` — ✅ OK
- Exists: **True**
- Classes: WhaleTracker
- Methods: WhaleTracker.__init__, WhaleTracker._calculate_risk_level, WhaleTracker._generate_recommendation, WhaleTracker._get_chain_id, WhaleTracker._get_erc20_abi, WhaleTracker._get_token_decimals, WhaleTracker._get_token_price, WhaleTracker._get_top_holders, WhaleTracker._get_wallet_history, WhaleTracker._load_known_whales, WhaleTracker._setup_web3_connections, WhaleTracker._track_recent_transfers, WhaleTracker.analyze_whale_behavior, WhaleTracker.close, WhaleTracker.get_statistics

### `data\processors\aggregator.py` — ✅ OK
- Exists: **True**
- Classes: DataAggregator
- Methods: DataAggregator.__init__, DataAggregator._aggregate_holder_metrics, DataAggregator._aggregate_liquidity, DataAggregator._aggregate_prices, DataAggregator._aggregate_social_metrics, DataAggregator._aggregate_technical_indicators, DataAggregator._aggregate_volumes, DataAggregator._calculate_quality_score, DataAggregator._clean_source_data, DataAggregator._detect_anomalies, DataAggregator._record_conflict, DataAggregator._validate_source_data, DataAggregator.aggregate_market_data, DataAggregator.aggregate_token_data, DataAggregator.get_conflict_report

### `data\processors\feature_extractor.py` — ✅ OK
- Exists: **True**
- Classes: FeatureExtractor
- Methods: FeatureExtractor.__init__, FeatureExtractor._detect_doji, FeatureExtractor._detect_double_bottom, FeatureExtractor._detect_double_top, FeatureExtractor._detect_engulfing, FeatureExtractor._detect_flag, FeatureExtractor._detect_hammer, FeatureExtractor._detect_head_shoulders, FeatureExtractor._detect_triangle, FeatureExtractor.create_feature_vector, FeatureExtractor.extract_all_features, FeatureExtractor.extract_chain_features, FeatureExtractor.extract_derived_features, FeatureExtractor.extract_interaction_features, FeatureExtractor.extract_market_features

### `data\processors\normalizer.py` — ✅ OK
- Exists: **True**
- Classes: DataNormalizer, DataType, NormalizationConfig
- Methods: DataNormalizer.__init__, DataNormalizer._create_record_key, DataNormalizer._default_config, DataNormalizer.export_schema, DataNormalizer.merge_normalized_data, DataNormalizer.normalize_address, DataNormalizer.normalize_batch, DataNormalizer.normalize_chain, DataNormalizer.normalize_dataframe, DataNormalizer.normalize_dex, DataNormalizer.normalize_market_cap, DataNormalizer.normalize_percentage, DataNormalizer.normalize_price, DataNormalizer.normalize_record, DataNormalizer.normalize_symbol

### `data\processors\validator.py` — ✅ OK
- Exists: **True**
- Classes: DataField, DataValidator, ValidationLevel, ValidationResult, ValidationRule
- Methods: DataValidator.__init__, DataValidator._apply_rule, DataValidator._calculate_quality_score, DataValidator._check_consistency, DataValidator._check_required_fields, DataValidator._determine_field_type, DataValidator._generate_suggestions, DataValidator._initialize_rules, DataValidator._summarize_errors, DataValidator._summarize_warnings, DataValidator._validate_address, DataValidator._validate_liquidity, DataValidator._validate_percentage, DataValidator._validate_price, DataValidator._validate_schema

### `data\storage\cache.py` — ✅ OK
- Exists: **True**
- Classes: CacheManager
- Methods: CacheManager.__init__, CacheManager._setup_keyspace_notifications, CacheManager.acquire_lock, CacheManager.connect, CacheManager.decr, CacheManager.delete, CacheManager.disconnect, CacheManager.exists, CacheManager.expire, CacheManager.get, CacheManager.get_many, CacheManager.get_stats, CacheManager.hget, CacheManager.hgetall, CacheManager.hset

### `data\storage\database.py` — ✅ OK
- Exists: **True**
- Classes: DatabaseManager
- Methods: DatabaseManager.__init__, DatabaseManager._create_tables, DatabaseManager._initialize_timescaledb, DatabaseManager.acquire, DatabaseManager.cleanup_old_data, DatabaseManager.connect, DatabaseManager.disconnect, DatabaseManager.get_active_positions, DatabaseManager.get_historical_data, DatabaseManager.get_recent_trades, DatabaseManager.get_statistics, DatabaseManager.get_token_analysis, DatabaseManager.save_market_data, DatabaseManager.save_market_data_batch, DatabaseManager.save_performance_metrics

### `data\storage\models.py` — ✅ OK
- Exists: **True**
- Classes: Alert, AlertPriority, MEVTransaction, MarketData, PerformanceMetrics, Position, PositionStatus, SystemLog, TokenAnalysis, Trade
- Methods: Alert.to_dict, PerformanceMetrics.to_dict, Position.to_dict, TokenAnalysis.to_dict, Trade.to_dict, Trade.validate_side, Trade.validate_status, create_all_tables, drop_all_tables

### `main.py` — ✅ OK
- Exists: **True**
- Classes: TradingBotApplication
- Methods: TradingBotApplication.__init__, TradingBotApplication._check_apis, TradingBotApplication._check_database, TradingBotApplication._check_models, TradingBotApplication._check_redis, TradingBotApplication._check_wallet, TradingBotApplication._check_web3, TradingBotApplication._perform_system_checks, TradingBotApplication._shutdown_monitor, TradingBotApplication._signal_handler, TradingBotApplication._status_reporter, TradingBotApplication._validate_environment, TradingBotApplication.initialize, TradingBotApplication.run, TradingBotApplication.shutdown

### `ml\models\ensemble_model.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\ml\models\ensemble_model.py: closing parenthesis ']' does not match opening parenthesis '(' (<unknown>, line 744)

### `ml\models\pump_predictor.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\ml\models\pump_predictor.py: unexpected indent (<unknown>, line 432)

### `ml\models\rug_classifier.py` — ✅ OK
- Exists: **True**
- Classes: RugClassifier
- Methods: RugClassifier.__init__, RugClassifier._calculate_confidence, RugClassifier._calculate_ensemble_importance, RugClassifier._calculate_liquidity_removal_risk, RugClassifier._ensemble_predict_proba, RugClassifier._generate_warnings, RugClassifier._get_recommendation, RugClassifier._identify_red_flags, RugClassifier._initialize_models, RugClassifier.analyze_token, RugClassifier.extract_features, RugClassifier.load_model, RugClassifier.predict, RugClassifier.save_model, RugClassifier.train

### `ml\models\volume_validator.py` — ✅ OK
- Exists: **True**
- Classes: VolumeValidationResult, VolumeValidatorML
- Methods: VolumeValidatorML.__init__, VolumeValidatorML._ensemble_predict, VolumeValidatorML._estimate_real_volume, VolumeValidatorML._generate_recommendation, VolumeValidatorML._get_feature_importance, VolumeValidatorML._heuristic_validation, VolumeValidatorML._identify_risk_factors, VolumeValidatorML.extract_features, VolumeValidatorML.get_model_performance, VolumeValidatorML.load_model, VolumeValidatorML.needs_retraining, VolumeValidatorML.predict, VolumeValidatorML.save_model, VolumeValidatorML.train

### `monitoring\alerts.py` — ✅ OK
- Exists: **True**
- Classes: Alert, AlertPriority, AlertRule, AlertType, AlertsSystem, ChannelConfig, NotificationChannel
- Methods: AlertsSystem.__init__, AlertsSystem._aggregate_alerts, AlertsSystem._check_rate_limit, AlertsSystem._default_config, AlertsSystem._format_aggregated_alerts, AlertsSystem._format_arbitrage_opportunity, AlertsSystem._format_correlation_warning, AlertsSystem._format_daily_summary, AlertsSystem._format_drawdown_alert, AlertsSystem._format_margin_call, AlertsSystem._format_performance_metrics, AlertsSystem._format_position_closed, AlertsSystem._format_position_opened, AlertsSystem._format_risk_warning, AlertsSystem._format_signal

### `monitoring\dashboard.py` — ✅ OK
- Exists: **True**
- Classes: ChartData, Dashboard, DashboardData, DashboardManager, DashboardSection
- Methods: Dashboard.__init__, Dashboard._broadcast_updates, Dashboard._cancel_order, Dashboard._close_position, Dashboard._default_config, Dashboard._execute_action, Dashboard._modify_order, Dashboard._modify_position, Dashboard._serialize_dashboard_data, Dashboard._setup_routes, Dashboard._setup_socketio_handlers, Dashboard._toggle_strategy, Dashboard._update_dashboard_data, Dashboard.add_alert, Dashboard.add_chart

### `monitoring\logger.py` — ✅ OK
- Exists: **True**
- **Duplicates** (1): class:StructuredLogger::setup_logging x2
- Classes: ColoredFormatter, JsonFormatter, MetricType, PerformanceSnapshot, PerformanceTracker, StandardFormatter, StructuredLogger, TradeFormatter, TradeRecord
- Methods: ColoredFormatter.format, JsonFormatter.format, PerformanceTracker.__init__, PerformanceTracker._calculate_snapshots, PerformanceTracker._create_snapshot, PerformanceTracker._default_config, PerformanceTracker._init_database, PerformanceTracker._initialize_metrics, PerformanceTracker._log_trade, PerformanceTracker._persist_data, PerformanceTracker._update_metrics, PerformanceTracker.get_performance_report, PerformanceTracker.record_trade, StandardFormatter.__init__, StructuredLogger.__init__

### `monitoring\performance.py` — ✅ OK
- Exists: **True**
- Classes: ColoredFormatter, JsonFormatter, MetricType, PerformanceSnapshot, PerformanceTracker, StandardFormatter, StructuredLogger, TradeFormatter, TradeRecord
- Methods: ColoredFormatter.format, JsonFormatter.format, PerformanceTracker.__init__, PerformanceTracker._calculate_snapshots, PerformanceTracker._calculate_var, PerformanceTracker._create_snapshot, PerformanceTracker._default_config, PerformanceTracker._init_database, PerformanceTracker._initialize_metrics, PerformanceTracker._log_trade, PerformanceTracker._persist_data, PerformanceTracker._update_metrics, PerformanceTracker.calculate_max_drawdown, PerformanceTracker.calculate_metrics, PerformanceTracker.calculate_sharpe_ratio

### `scripts\health_check.py` — ✅ OK
- Exists: **True**
- Methods: check_health, main

### `scripts\init_config.py` — ✅ OK
- Exists: **True**
- Methods: init_config

### `scripts\run_tests.py` — ✅ OK
- Exists: **True**
- Classes: TestRunner
- Methods: TestRunner.__init__, TestRunner.check_redis, TestRunner.check_test_database, TestRunner.cleanup, TestRunner.create_test_database, TestRunner.generate_report, TestRunner.parse_results, TestRunner.run_tests, TestRunner.setup_environment, TestRunner.start_redis, main

### `scripts\setup_database.py` — ✅ OK
- Exists: **True**
- Methods: setup_database

### `security\__init__.py` — ✅ OK
- Exists: **True**

### `security\api_security.py` — ✅ OK
- Exists: **True**
- Classes: APISecurityManager
- Methods: APISecurityManager.__init__, APISecurityManager._check_ip_access, APISecurityManager._check_permissions, APISecurityManager._load_api_keys, APISecurityManager._log_request, APISecurityManager._save_api_keys, APISecurityManager._validate_api_key, APISecurityManager._validate_signature, APISecurityManager.add_ip_blacklist, APISecurityManager.add_ip_whitelist, APISecurityManager.check_rate_limit, APISecurityManager.generate_api_key, APISecurityManager.generate_jwt_token, APISecurityManager.revoke_api_key, APISecurityManager.validate_jwt_token

### `security\audit_logger.py` — ✅ OK
- Exists: **True**
- Classes: AuditEvent, AuditEventType, AuditLogger, AuditSeverity, AuditStatus
- Methods: AuditLogger.__init__, AuditLogger._calculate_checksum, AuditLogger._calculate_risk_score, AuditLogger._check_compliance_rules, AuditLogger._check_risk_patterns, AuditLogger._cleanup_old_logs, AuditLogger._cleanup_old_logs_periodically, AuditLogger._compress_log_file, AuditLogger._flush_buffer, AuditLogger._flush_buffer_periodically, AuditLogger._get_current_log_file, AuditLogger._handle_compliance_violation, AuditLogger._load_compliance_rules, AuditLogger._load_risk_patterns, AuditLogger._send_immediate_alert

### `security\encryption.py` — ✅ OK
- Exists: **True**
- Classes: EncryptionManager
- Methods: EncryptionManager.__init__, EncryptionManager._generate_key_id, EncryptionManager._get_or_create_master_key, EncryptionManager.decrypt_api_key, EncryptionManager.decrypt_sensitive_data, EncryptionManager.decrypt_wallet_key, EncryptionManager.encrypt_api_key, EncryptionManager.encrypt_sensitive_data, EncryptionManager.encrypt_wallet_key, EncryptionManager.rotate_encryption_key, EncryptionManager.secure_delete

### `security\wallet_security.py` — ✅ OK
- Exists: **True**
- Classes: SecurityLevel, TransactionApproval, WalletConfig, WalletSecurityManager, WalletType
- Methods: WalletSecurityManager.__init__, WalletSecurityManager._check_rate_limits, WalletSecurityManager._check_transaction_limits, WalletSecurityManager._get_wallet_data, WalletSecurityManager._initialize_web3_connections, WalletSecurityManager._load_security_policies, WalletSecurityManager._load_wallet_configurations, WalletSecurityManager._lock_wallet, WalletSecurityManager._log_transaction, WalletSecurityManager._notify_emergency_contacts, WalletSecurityManager._perform_security_checks, WalletSecurityManager._save_wallet_data, WalletSecurityManager._setup_hardware_wallet_address, WalletSecurityManager._setup_hardware_wallets, WalletSecurityManager._sign_hardware_wallet_transaction

### `setup.py` — ✅ OK
- Exists: **True**
- Methods: read_requirements

### `tests\conftest.py` — ✅ OK
- Exists: **True**
- Methods: audit_logger, benchmark_data, cache_manager, cleanup, config_manager, create_mock_token, create_mock_trade, db_manager, event_loop, mock_dex_api, mock_dex_client, mock_market_data, mock_web3, populate_test_database, risk_manager

### `tests\fixtures\mock_data.py` — ✅ OK
- Exists: **True**
- Classes: MockDataGenerator
- Methods: MockDataGenerator.generate_market_conditions, MockDataGenerator.generate_price_history, MockDataGenerator.generate_token, MockDataGenerator.generate_token_address, MockDataGenerator.generate_trades, MockDataGenerator.generate_whale_movement

### `tests\fixtures\test_helpers.py` — ✅ OK
- Exists: **True**
- Classes: TestHelpers
- Methods: TestHelpers.assert_position_valid, TestHelpers.assert_trade_valid, TestHelpers.cleanup_test_data, TestHelpers.compare_decimals, TestHelpers.create_mock_web3_contract, TestHelpers.generate_mock_config, TestHelpers.wait_for_condition

### `tests\integration\test_data_integration.py` — ✅ OK
- Exists: **True**
- Classes: TestDataIntegration
- Methods: TestDataIntegration.test_batch_processing, TestDataIntegration.test_data_aggregation_pipeline, TestDataIntegration.test_dexscreener_to_database, TestDataIntegration.test_honeypot_checker_caching, TestDataIntegration.test_whale_tracker_integration

### `tests\integration\test_ml_integration.py` — ✅ OK
- Exists: **True**
- Classes: TestMLIntegration
- Methods: TestMLIntegration.test_ensemble_model, TestMLIntegration.test_pump_predictor_training, TestMLIntegration.test_rug_classifier_training, TestMLIntegration.training_data

### `tests\integration\test_trading_integration.py` — ✅ OK
- Exists: **True**
- Classes: TestTradingIntegration
- Methods: TestTradingIntegration.test_order_execution_flow, TestTradingIntegration.test_position_monitoring_and_exit, TestTradingIntegration.test_strategy_signal_execution

### `tests\performance\test_performance.py` — ✅ OK
- Exists: **True**
- Classes: TestPerformance
- Methods: TestPerformance.execute_order, TestPerformance.process_request, TestPerformance.test_cache_performance, TestPerformance.test_concurrent_request_handling, TestPerformance.test_database_write_performance, TestPerformance.test_memory_usage, TestPerformance.test_ml_model_performance, TestPerformance.test_order_execution_latency

### `tests\security\test_security.py` — ✅ OK
- Exists: **True**
- Classes: TestSecurity
- Methods: TestSecurity.test_api_authentication, TestSecurity.test_api_security_rate_limiting, TestSecurity.test_audit_logging, TestSecurity.test_encryption_manager, TestSecurity.test_input_validation, TestSecurity.test_private_key_protection, TestSecurity.test_secure_random_generation, TestSecurity.test_sql_injection_protection, TestSecurity.test_wallet_security

### `tests\smoke\test_smoke.py` — ✅ OK
- Exists: **True**
- Classes: TestSmoke
- Methods: TestSmoke.test_api_endpoints_available, TestSmoke.test_cache_connection, TestSmoke.test_config_loading, TestSmoke.test_database_connection, TestSmoke.test_engine_startup, TestSmoke.test_ml_model_loading

### `tests\unit\test_engine.py` — ✅ OK
- Exists: **True**
- Classes: TestTradingBotEngine
- Methods: TestTradingBotEngine.engine, TestTradingBotEngine.test_analyze_opportunity, TestTradingBotEngine.test_close_position, TestTradingBotEngine.test_engine_initialization, TestTradingBotEngine.test_error_handling, TestTradingBotEngine.test_execute_opportunity, TestTradingBotEngine.test_monitor_positions, TestTradingBotEngine.test_safety_checks, TestTradingBotEngine.test_start_stop

### `tests\unit\test_risk_manager.py` — ✅ OK
- Exists: **True**
- Classes: TestRiskManager
- Methods: TestRiskManager.risk_manager, TestRiskManager.test_calculate_position_size, TestRiskManager.test_calculate_var, TestRiskManager.test_check_correlation_limit, TestRiskManager.test_check_position_limit, TestRiskManager.test_emergency_stop, TestRiskManager.test_portfolio_exposure, TestRiskManager.test_risk_adjusted_returns, TestRiskManager.test_set_stop_loss

### `trading\executors\base_executor.py` — ✅ OK
- Exists: **True**
- Classes: ExecutionResult, ExecutionRoute, TradeExecutor, TradeOrder
- Methods: TradeExecutor.__init__, TradeExecutor._apply_mev_protection, TradeExecutor._approve_token, TradeExecutor._execute_1inch, TradeExecutor._execute_uniswap_v2, TradeExecutor._execute_with_retry, TradeExecutor._get_1inch_quote, TradeExecutor._get_all_quotes, TradeExecutor._get_token_balance, TradeExecutor._get_uniswap_v2_quote, TradeExecutor._get_weth_address, TradeExecutor._load_abi, TradeExecutor._post_execution_processing, TradeExecutor._pre_execution_checks, TradeExecutor._select_best_route

### `trading\executors\direct_dex.py` — ⚠️ Issues
- Exists: **True**
- **Missing** (1): [ParseError] AST parse error in C:\Users\HP\Desktop\ClaudeDex\trading\executors\direct_dex.py: expected 'except' or 'finally' block (<unknown>, line 170)

### `trading\executors\mev_protection.py` — ✅ OK
- Exists: **True**
- Classes: AttackType, MEVProtectionLayer, MEVProtectionLevel, MEVThreat, ProtectedTransaction
- Methods: MEVProtectionLayer.__init__, MEVProtectionLayer._analyze_attack_patterns, MEVProtectionLayer._analyze_mev_risk, MEVProtectionLayer._apply_commit_reveal, MEVProtectionLayer._apply_dynamic_routing, MEVProtectionLayer._apply_gas_randomization, MEVProtectionLayer._apply_time_delays, MEVProtectionLayer._calculate_tx_hash, MEVProtectionLayer._check_attacker_activity, MEVProtectionLayer._check_mempool_congestion, MEVProtectionLayer._create_flashbots_bundle, MEVProtectionLayer._detect_mev_threat, MEVProtectionLayer._estimate_mev_savings, MEVProtectionLayer._execute_flashbots_bundle, MEVProtectionLayer._execute_private_mempool

### `trading\executors\toxisol_api.py` — ✅ OK
- Exists: **True**
- Classes: ToxiSolAPIExecutor, ToxiSolQuote, ToxiSolRoute
- Methods: ToxiSolAPIExecutor.__init__, ToxiSolAPIExecutor._build_transaction, ToxiSolAPIExecutor._execute_flashbots, ToxiSolAPIExecutor._execute_standard, ToxiSolAPIExecutor._generate_signature, ToxiSolAPIExecutor._rate_limit, ToxiSolAPIExecutor._test_connection, ToxiSolAPIExecutor.cleanup, ToxiSolAPIExecutor.estimate_gas, ToxiSolAPIExecutor.execute_trade, ToxiSolAPIExecutor.get_execution_stats, ToxiSolAPIExecutor.get_quote, ToxiSolAPIExecutor.initialize

### `trading\orders\order_manager.py` — ✅ OK
- Exists: **True**
- Classes: ExecutionEngine, ExecutionStrategy, Fill, Order, OrderBook, OrderManager, OrderRiskMonitor, OrderSide, OrderStatus, OrderType
- Methods: ExecutionEngine.__init__, ExecutionEngine.cancel_transaction, ExecutionEngine.get_order_book, ExecutionEngine.get_transaction_status, ExecutionEngine.submit_flashbots_bundle, ExecutionEngine.submit_private_transaction, ExecutionEngine.submit_transaction, OrderManager.__init__, OrderManager._cleanup_expired_orders, OrderManager._default_config, OrderManager._execute_iceberg, OrderManager._execute_immediate, OrderManager._execute_sniper, OrderManager._execute_stop_loss, OrderManager._execute_twap

### `trading\orders\position_tracker.py` — ✅ OK
- Exists: **True**
- Classes: PerformanceMetrics, PortfolioSnapshot, Position, PositionStatus, PositionTracker, PositionType, RiskLevel
- Methods: PositionTracker.__init__, PositionTracker._calculate_correlation, PositionTracker._calculate_portfolio_beta, PositionTracker._calculate_portfolio_metrics, PositionTracker._calculate_portfolio_risk, PositionTracker._calculate_portfolio_volatility, PositionTracker._calculate_position_risk, PositionTracker._calculate_value_at_risk, PositionTracker._check_portfolio_alerts, PositionTracker._check_position_limits, PositionTracker._check_position_rules, PositionTracker._default_config, PositionTracker._determine_risk_level, PositionTracker._execute_position_actions, PositionTracker._get_current_price

### `trading\strategies\ai_strategy.py` — ✅ OK
- Exists: **True**
- Classes: AIStrategy
- Methods: AIStrategy.__init__, AIStrategy._assess_feature_quality, AIStrategy._calculate_bb_position, AIStrategy._calculate_hurst_exponent, AIStrategy._calculate_kurtosis, AIStrategy._calculate_log_returns, AIStrategy._calculate_macd_signal, AIStrategy._calculate_model_agreement, AIStrategy._calculate_momentum, AIStrategy._calculate_order_imbalance, AIStrategy._calculate_resistance_distance, AIStrategy._calculate_returns, AIStrategy._calculate_rsi, AIStrategy._calculate_skewness, AIStrategy._calculate_support_distance

### `trading\strategies\base_strategy.py` — ✅ OK
- Exists: **True**
- Classes: BaseStrategy, SignalStrength, SignalType, StrategyPerformance, StrategyState, TradingSignal
- Methods: BaseStrategy.__init__, BaseStrategy._calculate_kelly_position, BaseStrategy._check_custom_exit_conditions, BaseStrategy._create_exit_signal, BaseStrategy._create_order_from_signal, BaseStrategy._get_strength_multiplier, BaseStrategy._on_position_closed, BaseStrategy._on_position_failed, BaseStrategy._on_position_opened, BaseStrategy._update_performance, BaseStrategy.analyze, BaseStrategy.backtest, BaseStrategy.calculate_indicators, BaseStrategy.calculate_max_drawdown, BaseStrategy.calculate_position_size

### `trading\strategies\momentum.py` — ✅ OK
- Exists: **True**
- Classes: MomentumMetrics, MomentumSignal, MomentumStrategy, MomentumType, TimeFrame
- Methods: MomentumStrategy.__init__, MomentumStrategy._analyze_volume_pattern, MomentumStrategy._calculate_average_strength, MomentumStrategy._calculate_breakout_probability, MomentumStrategy._calculate_breakout_stop_loss, MomentumStrategy._calculate_breakout_targets, MomentumStrategy._calculate_ma_trend, MomentumStrategy._calculate_macd_momentum, MomentumStrategy._calculate_momentum_metrics, MomentumStrategy._calculate_position_size, MomentumStrategy._calculate_price_velocity, MomentumStrategy._calculate_range, MomentumStrategy._calculate_rsi_momentum, MomentumStrategy._calculate_smart_money_score, MomentumStrategy._calculate_smart_money_stop_loss

### `trading\strategies\scalping.py` — ✅ OK
- Exists: **True**
- Classes: ScalpingOpportunity, ScalpingSignal, ScalpingStrategy
- Methods: ScalpingStrategy.__init__, ScalpingStrategy._analyze_volume_profile, ScalpingStrategy._assess_risk, ScalpingStrategy._calculate_bollinger_bands, ScalpingStrategy._calculate_confidence, ScalpingStrategy._calculate_entry_price, ScalpingStrategy._calculate_expected_profit, ScalpingStrategy._calculate_indicators, ScalpingStrategy._calculate_macd, ScalpingStrategy._calculate_momentum, ScalpingStrategy._calculate_position_size, ScalpingStrategy._calculate_rsi, ScalpingStrategy._calculate_stop_loss, ScalpingStrategy._calculate_target_price, ScalpingStrategy._close_position

### `utils\__init__.py` — ✅ OK
- Exists: **True**

### `utils\constants.py` — ✅ OK
- Exists: **True**
- Classes: Chain, DEX, MarketCondition, OrderStatus, OrderType, SignalStrength, TradingMode

### `utils\helpers.py` — ✅ OK
- Exists: **True**
- **Duplicates** (1): func:decorator::wrapper x2
- Classes: TTLCache
- Methods: TTLCache.__init__, TTLCache.clear, TTLCache.get, TTLCache.set, batch_request, batch_request.fetch_with_semaphore, calculate_ema, calculate_moving_average, calculate_percentage_change, calculate_profit_loss, calculate_slippage, chunk_list, decorator.wrapper, deep_merge_dicts, ether_to_wei
