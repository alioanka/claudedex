# Symbol Index (Mini DB)

## analysis
- File: `analysis\__init__.py`

## analysis.dev_analyzer
- File: `analysis\dev_analyzer.py`

### Classes & Methods
- **DeveloperAnalyzer**
  - `__init__(config)`
  - `_analyze_code_quality(project_address, chain)`
  - `_analyze_community_metrics(project_address)`
  - `_analyze_previous_projects(history)`
  - `_calculate_project_score(code_quality, community_score, transparency, red_flags_count, green_flags_count)`
  - `_calculate_reputation(projects, social, history)`
  - `_calculate_transparency_score(project_data, liquidity)`
  - `_check_liquidity_status(project_address, chain)`
  - `_check_social_presence(address)`
  - `_create_rugger_profile(address)`
  - `_create_trusted_profile(address)`
  - `_determine_project_status(project_data, liquidity, holders)`
  - `_determine_risk_level(reputation, projects)`
  - `_find_aliases(address, associated_wallets)`
  - `_find_associated_wallets(address, chain)`
  - `_find_similar_projects(project_address, chain)`
  - `_get_developer_history(address, chain)`
  - `_get_holder_statistics(project_address, chain)`
  - `_get_project_data(project_address, chain)`
  - `_get_project_developer(project_address, chain)`
  - `_get_recommendation(profile)`
  - `_identify_green_flags(project_data, liquidity, holders, code_quality)`
  - `_identify_red_flags(project_data, liquidity, holders, developer)`
  - `_load_developer_database()`
  - `_log_developer_summary(profile)`
  - `analyze_developer(address, chain)`
  - `analyze_project(project_address, chain)`
  - `batch_analyze_developers(developers)`
  - `check_developer_reputation(address, chain)`
  - `get_risk_summary(profile)`
  - `initialize()`
  - `monitor_developer(address, chain, interval, callback)`
- **DeveloperProfile**
- **DeveloperRisk**
- **ProjectAnalysis**
- **ProjectStatus**

## analysis.liquidity_monitor
- File: `analysis\liquidity_monitor.py`

### Classes & Methods
- **LiquidityEvent**
- **LiquidityMonitor**
  - `__init__(db_manager, cache_manager, event_bus, alerts, config)`
  - `_calculate_amm_slippage(liquidity_data, amount, is_buy)`
  - `_calculate_liquidity_changes(current, history)`
  - `_calculate_liquidity_health(liquidity)`
  - `_calculate_momentum_indicators(data)`
  - `_calculate_orderbook_slippage(liquidity_data, amount, is_buy)`
  - `_calculate_price_impact(liquidity_data, amount, is_buy)`
  - `_calculate_pump_metrics(data, price_spike, volume_spike)`
  - `_calculate_removal_risk_score(analysis)`
  - `_check_lock_platform(token, chain, platform)`
  - `_detect_liquidity_anomalies(changes, history)`
  - `_detect_rug_patterns(token, chain, analysis)`
  - `_determine_slippage_risk(slippage)`
  - `_emit_liquidity_event(token, chain, changes)`
  - `_estimate_remaining_potential(data, metrics)`
  - `_get_current_liquidity(token, chain)`
  - `_get_exit_recommendation(metrics, potential)`
  - `_get_liquidity_depth(token, chain)`
  - `_get_liquidity_transactions(token, chain)`
  - `_get_lp_token_holders(lp_token, chain)`
  - `_get_pool_data(token, chain)`
  - `_get_realtime_data(token, chain)`
  - `_get_recent_price_volume(token, chain, minutes)`
  - `_get_removal_recommendation(analysis)`
  - `_get_slippage_recommendation(slippage, price_impact, liquidity_depth)`
  - `_is_lp_locked(address)`
  - `_is_suspicious_wallet(wallet)`
  - `_monitor_token_liquidity(token)`
  - `_send_liquidity_alert(token, chain, analysis, severity)`
  - `analyze_liquidity_locks(token, chain)`
  - `calculate_slippage(amount, liquidity_data)`
  - `calculate_slippage_async(token, chain, amount, is_buy)`
  - `calculate_slippage_sync(amount, liquidity_data)`
  - `detect_liquidity_removal(token, chain)`
  - `get_liquidity_depth(pair_address)`
  - `get_liquidity_depth_extended(pair_address, chain)`
  - `get_liquidity_providers(token, chain)`
  - `monitor_liquidity(token, chain)`
  - `monitor_liquidity_changes(token, chain)`
  - `start_monitoring(tokens)`
  - `stop_monitoring(token)`
  - `track_liquidity_changes(token)`
  - `track_liquidity_changes_extended(token, chain)`
- **SlippageEstimate**

## analysis.market_analyzer
- File: `analysis\market_analyzer.py`

### Classes & Methods
- **CorrelationMatrix**
- **MarketAnalyzer**
  - `__init__(db_manager, cache_manager, event_bus, ml_model, config)`
  - `_calculate_market_metrics(market_data)`
  - `_calculate_market_risk(volatility, liquidity, sentiment)`
  - `_calculate_market_volatility(metrics)`
  - `_detect_volume_spikes(df)`
  - `_determine_market_trend(metrics)`
  - `_get_regime_recommendation(regime)`
  - `_identify_correlation_clusters(correlation_matrix, threshold)`
  - `analyze_market(chain)`
  - `analyze_market_conditions(tokens, chain)`
  - `analyze_volume_patterns(token, chain)`
  - `calculate_correlations(tokens, chain, period)`
  - `calculate_market_sentiment()`
  - `detect_market_regime(token, chain)`
  - `find_market_inefficiencies(tokens, chain)`
  - `get_market_indicators()`
  - `identify_market_trends()`
  - `identify_trends(token, chain, timeframe)`
- **MarketCondition**
- **TrendAnalysis**

## analysis.pump_predictor
- File: `analysis\pump_predictor.py`

### Classes & Methods
- **PumpPattern**
- **PumpPredictorAnalysis**
  - `__init__(db_manager, cache_manager, event_bus, ml_predictor, market_analyzer, config)`
  - `_analyze_pump_patterns(data)`
  - `_assess_pump_risk(probability, pump_type, data)`
  - `_calculate_entry_exit_points(data, expected_metrics)`
  - `_calculate_expected_metrics(data, probability, pump_type)`
  - `_combine_signals(ml_probability, pattern_signals, accumulation, whale_signals, social_signals)`
  - `_detect_accumulation_phase(data)`
  - `_detect_price_spike(data)`
  - `_detect_volume_spike(data)`
  - `_determine_pump_status(price_change, expected_magnitude, duration, expected_duration, targets_hit)`
  - `_gather_prediction_data(token, chain)`
  - `_get_default_signal(token, chain)`
  - `_get_signal_strength(probability)`
  - `_identify_pump_type(pattern_signals, whale_signals, social_signals)`
  - `_process_significant_signal(signal)`
  - `analyze_pump_history(token, chain, days)`
  - `analyze_volume_patterns(volume_data)`
  - `calculate_pump_probability(indicators)`
  - `detect_accumulation_phase(price_data)`
  - `detect_accumulation_phase_sync(price_data)`
  - `detect_ongoing_pump(token, chain)`
  - `monitor_pump_completion(signal)`
  - `predict_pump(token, chain)`
- **PumpSignal**

## analysis.rug_detector
- File: `analysis\rug_detector.py`

### Classes & Methods
- **RugDetector**
  - `__init__(config)`
  - `_calculate_contract_risk_score(checks)`
  - `_calculate_holder_risk_score(distribution)`
  - `_check_lock_provider(token, chain, provider)`
  - `_check_ownership(address, chain)`
  - `_check_proxy_pattern(address, chain)`
  - `_check_risky_functions(address, chain, code)`
  - `_check_verification(address, chain)`
  - `_get_token_holders(token, chain)`
  - `_initialize_web3()`
  - `analyze_contract(address, chain)`
  - `analyze_holder_distribution(token, chain)`
  - `analyze_token(token, chain)`
  - `calculate_rug_score(factors)`
  - `check_contract_vulnerabilities(contract_code)`
  - `check_liquidity_lock(token, chain)`
  - `check_liquidity_removal_risk(liquidity_data)`
  - `check_ownership_concentration(holder_data)`
  - `comprehensive_analysis(token, chain)`

## analysis.smart_contract_analyzer
- File: `analysis\smart_contract_analyzer.py`

### Classes & Methods
- **ContractAnalysis**
- **ContractType**
- **SmartContractAnalyzer**
  - `__init__(config)`
  - `_analyze_fees(contract_data, chain)`
  - `_analyze_permissions(contract_data, chain)`
  - `_analyze_proxy_pattern(address, contract_data, chain)`
  - `_analyze_storage(address, chain)`
  - `_analyze_vulnerabilities(source_code)`
  - `_calculate_risk_score(vulnerabilities, is_verified, is_proxy, owner, permissions, fees, upgrade_mechanism)`
  - `_check_upgrade_mechanism(contract_data, is_proxy)`
  - `_check_verification(address, chain)`
  - `_create_malicious_analysis(address, chain)`
  - `_detect_changes(old_analysis, new_analysis)`
  - `_extract_events(contract_data)`
  - `_extract_functions(contract_data)`
  - `_extract_modifiers(contract_data)`
  - `_fetch_contract_data(address, chain)`
  - `_fetch_from_etherscan(address, chain)`
  - `_get_owner(address, contract_data, chain)`
  - `_identify_contract_type(contract_data, chain)`
  - `_load_function_signatures()`
  - `_load_malicious_contracts()`
  - `_load_vulnerability_patterns()`
  - `_log_analysis_summary(analysis)`
  - `analyze_contract(address, chain, deep_analysis)`
  - `batch_analyze(contracts, deep_analysis)`
  - `check_renounced_ownership(address, chain)`
  - `estimate_gas_usage(address, chain, function_name, params)`
  - `get_contract_creation_info(address, chain)`
  - `get_risk_summary(analysis)`
  - `initialize()`
  - `monitor_contract(address, chain, interval, callback)`
  - `verify_token_safety(token_address, chain)`
- **Vulnerability**
- **VulnerabilityLevel**

## analysis.token_scorer
- File: `analysis\token_scorer.py`

### Classes & Methods
- **ScoringWeights**
- **TokenScore**
- **TokenScorer**
  - `__init__(db_manager, cache_manager, event_bus, rug_detector, liquidity_monitor, market_analyzer, ml_model, config)`
  - `_adjust_score_for_risk(composite_score, risk_score)`
  - `_calculate_category_scores(data)`
  - `_calculate_confidence(data)`
  - `_calculate_innovation_score(data)`
  - `_calculate_opportunity_score(data)`
  - `_calculate_risk_score(data)`
  - `_calculate_weighted_score(category_scores)`
  - `_determine_grade(score)`
  - `_gather_scoring_data(token, chain)`
  - `_generate_comparison_recommendation(score1, score2)`
  - `_generate_recommendation(composite_score, risk_score, opportunity_score, data)`
  - `_get_default_score(token, chain)`
  - `_get_developer_activity(token, chain)`
  - `_get_holder_distribution(token, chain)`
  - `_get_rank_value(score, criteria)`
  - `_get_social_metrics(token)`
  - `_get_token_data(token, chain)`
  - `_identify_strengths(category_scores, data)`
  - `_identify_weaknesses(category_scores, data)`
  - `_is_cache_valid(timestamp)`
  - `_is_undervalued(data)`
  - `calculate_composite_score(token, chain)`
  - `calculate_fundamental_score(token_data)`
  - `calculate_social_score(social_data)`
  - `calculate_technical_score(price_data)`
  - `compare_tokens(token1, token2, chain)`
  - `get_overall_score(scores)`
  - `get_top_opportunities(chain, limit, min_score)`
  - `rank_tokens(tokens, chain, criteria)`
  - `score_token(token, chain)`

## config
- File: `config\__init__.py`

## config.config_manager
- File: `config\config_manager.py`

### Classes & Methods
- **APIConfig**
- **ConfigChange**
- **ConfigManager**
  - `__init__(config_dir)`
  - `_auto_reload_loop()`
  - `_check_for_changes()`
  - `_decrypt_sensitive_values(config_data)`
  - `_encrypt_sensitive_values(config_data)`
  - `_is_float(value)`
  - `_load_all_configs()`
  - `_load_config(config_type)`
  - `_load_config_from_database(config_type)`
  - `_load_config_from_env(config_type)`
  - `_load_config_from_file(config_type)`
  - `_load_default_config(config_type)`
  - `_notify_config_watchers(config_type, new_config)`
  - `_persist_config(config_type, config)`
  - `_reload_config(config_type)`
  - `_start_auto_reload()`
  - `_track_config_changes(config_type, old_config, new_config)`
  - `backup_configs(backup_path)`
  - `cleanup()`
  - `get_api_config()`
  - `get_change_history(config_type, limit)`
  - `get_config(key)`
  - `get_config_internal(config_type)`
  - `get_config_status()`
  - `get_database_config()`
  - `get_ml_models_config()`
  - `get_monitoring_config()`
  - `get_risk_management_config()`
  - `get_security_config()`
  - `get_trading_config()`
  - `initialize(encryption_key)`
  - `load_config(env)`
  - `register_watcher(config_type, callback)`
  - `reload_config()`
  - `restore_configs(backup_path)`
  - `unregister_watcher(config_type, callback)`
  - `update_config(key, value)`
  - `update_config_internal(config_type, updates, user, reason, persist)`
  - `validate_config(config)`
  - `validate_config_internal(config_type, config_data)`
- **ConfigSource**
- **ConfigType**
- **DatabaseConfig**
- **MLModelsConfig**
- **MonitoringConfig**
- **RiskManagementConfig**
  - `validate_portfolio_risk(cls, v)`
- **SecurityConfig**
  - `validate_rate_limit(cls, v)`
- **TradingConfig**
  - `validate_position_size(cls, v)`

## config.settings
- File: `config\settings.py`

### Classes & Methods
- **ChainConfig**
  - `__init__(name, chain_id, rpc_url, explorer_url, native_token, is_testnet)`
- **Environment**
- **Settings**
  - `get_chain_config(cls, chain)`
  - `get_chain_config_exMethod(cls, chain_name)`
  - `get_current_features(cls)`
  - `get_database_url(cls)`
  - `get_dex_config(cls, dex_name)`
  - `get_environment_info(cls)`
  - `get_gas_config(cls, chain_name, speed)`
  - `get_redis_url(cls)`
  - `get_stablecoin_address(cls, chain, symbol)`
  - `is_feature_enabled(cls, feature)`
  - `is_token_blacklisted(cls, token_address)`
  - `validate_environment(cls)`

## config.validation
- File: `config\validation.py`

### Functions
- `validate_security_config(config)`
- `validate_trading_config(config)`

### Classes & Methods
- **ConfigValidator**
  - `__init__()`
  - `_is_valid_email(email)`
  - `_is_valid_hostname(hostname)`
  - `_is_valid_ip(ip)`
  - `_is_valid_url(url)`
  - `_validate_api_config(config, result)`
  - `_validate_cross_config(config_type, config, result)`
  - `_validate_database_config(config, result)`
  - `_validate_ml_models_config(config, result)`
  - `_validate_monitoring_config(config, result)`
  - `_validate_risk_management_config(config, result)`
  - `_validate_security_config(config, result)`
  - `_validate_trading_config(config, result)`
  - `check_required_fields(config)`
  - `generate_validation_report(configs)`
  - `validate_api_keys(keys)`
  - `validate_config(config_type, config_data)`
  - `validate_decimal_string(value)`
  - `validate_ethereum_address(address)`
  - `validate_json_config(config_str)`
  - `validate_percentage(value, min_val, max_val)`
  - `validate_private_key(private_key)`
  - `validate_yaml_config(config_str)`
- **ValidationResult**
  - `__init__()`
  - `add_error(message)`
  - `add_recommendation(message)`
  - `add_warning(message)`
  - `merge(other)`
  - `to_dict()`

## core
- File: `core\__init__.py`

## core.decision_maker
- File: `core\decision_maker.py`

### Classes & Methods
- **DecisionMaker**
  - `__init__(config)`
  - `_adjust_strategy_weights()`
  - `_calculate_confidence(risk_score, ml_predictions, patterns, sentiment)`
  - `_calculate_position_parameters(data, strategy, confidence)`
  - `_classify_market_conditions(market_data)`
  - `_evaluate_ai_hybrid(data)`
  - `_evaluate_breakout(data)`
  - `_evaluate_mean_reversion(data)`
  - `_evaluate_momentum(data)`
  - `_evaluate_scalping(data)`
  - `_evaluate_strategies(analysis_data)`
  - `_evaluate_swing(data)`
  - `_generate_reasoning(data, strategy, confidence)`
  - `_kelly_criterion(confidence, risk_score, balance)`
  - `_select_best_strategy(scores)`
  - `_should_trade(confidence, risk_score, ml_predictions, liquidity)`
  - `calculate_confidence_score(signals)`
  - `determine_action(scores)`
  - `evaluate_opportunity(opportunity)`
  - `make_decision(analysis)`
  - `update_performance(decision_id, outcome)`
  - `validate_decision(decision)`
- **RiskScore**
  - `__post_init__()`
- **StrategyType**
- **TradingDecision**
- **TradingOpportunity**

## core.engine
- File: `core\engine.py`

### Classes & Methods
- **BotState**
- **TradingBotEngine**
  - `__init__(config, mode)`
  - `_analyze_opportunity(pair)`
  - `_check_exit_conditions(position)`
  - `_close_position(position, reason)`
  - `_execute_opportunity(opportunity)`
  - `_final_safety_checks(opportunity)`
  - `_health_check()`
  - `_monitor_existing_positions()`
  - `_monitor_mempool()`
  - `_monitor_new_pairs()`
  - `_monitor_performance()`
  - `_optimize_strategies()`
  - `_process_opportunities()`
  - `_retrain_models()`
  - `_setup_event_handlers()`
  - `_track_whales()`
  - `_update_blacklists()`
  - `initialize()`
  - `run()`
  - `start()`
  - `stop()`
- **TradingOpportunity**
  - `score()`

## core.event_bus
- File: `core\event_bus.py`

### Classes & Methods
- **Event**
  - `__init__(event_type, data)`
  - `to_dict()`
  - `to_json()`
- **EventAggregator**
  - `__init__(batch_size, timeout)`
  - `add_event(event)`
  - `flush()`
- **EventBus**
  - `__init__(config)`
  - `_add_to_history(event)`
  - `_apply_filters(event, filters)`
  - `_handle_event(event)`
  - `_process_events()`
  - `add_global_filter(filter_func)`
  - `clear_dead_letter_queue()`
  - `create_event_stream(event_types)`
  - `emit(event)`
  - `get_history(event_type, limit)`
  - `get_statistics()`
  - `process_events()`
  - `publish(event_type, data)`
  - `remove_global_filter(filter_func)`
  - `replay_events(events)`
  - `start()`
  - `stop()`
  - `subscribe(event_type, handler)`
  - `subscribe_sync(event_type, callback, subscriber_id, filters, priority)`
  - `unsubscribe(event_type, handler)`
  - `unsubscribe_sync(subscriber_id, event_type)`
  - `wait_for(event_type, timeout, filters)`
- **EventLogger**
  - `__init__(log_file)`
  - `log_event(event)`
  - `read_events(limit)`
- **EventSubscription**
- **EventType**

## core.pattern_analyzer
- File: `core\pattern_analyzer.py`

### Classes & Methods
- **Pattern**
- **PatternAnalyzer**
  - `__init__(config)`
  - `_analyze_momentum(df)`
  - `_analyze_trend(df)`
  - `_analyze_volume_profile(df)`
  - `_calculate_indicators(df)`
  - `_calculate_pattern_score(patterns, trend, indicators, momentum)`
  - `_calculate_risk_reward(df, sr_levels)`
  - `_calculate_trend_duration(closes, direction)`
  - `_cluster_levels(levels, tolerance)`
  - `_detect_breakouts(df)`
  - `_detect_candlestick_patterns(df)`
  - `_detect_chart_patterns(df)`
  - `_detect_cup_handle(closes, volumes)`
  - `_detect_double_patterns(closes, highs, lows)`
  - `_detect_flags(closes, volumes)`
  - `_detect_head_shoulders(closes, highs, lows)`
  - `_detect_triangles(closes, highs, lows)`
  - `_find_support_resistance(df)`
  - `_generate_signal(patterns, trend, indicators)`
  - `_prepare_dataframe(data)`
  - `analyze_patterns(price_data)`
  - `calculate_support_resistance(prices)`
  - `detect_candlestick_patterns(ohlc_data)`
  - `detect_chart_patterns(prices)`
  - `identify_trend(prices)`
- **PatternType**
- **SupportResistance**
- **TrendInfo**

## core.portfolio_manager
- File: `core\portfolio_manager.py`

### Classes & Methods
- **AllocationStrategy**
- **PortfolioManager**
  - `__init__(config)`
  - `_equal_weight_allocation(opportunities, available)`
  - `_risk_parity_allocation(opportunities, available)`
  - `allocate_capital(opportunities)`
  - `calculate_allocation(token)`
  - `can_open_position()`
  - `check_diversification()`
  - `get_available_balance()`
  - `get_portfolio_metrics()`
  - `get_portfolio_value()`
  - `rebalance_portfolio()`
  - `update_portfolio(trade)`
- **PortfolioMetrics**
- **Position**
  - `age()`
  - `unrealized_pnl()`
  - `value()`

## core.risk_manager
- File: `core\risk_manager.py`

### Classes & Methods
- **Position**
- **RiskLevel**
- **RiskManager**
  - `__init__(config)`
  - `_analyze_contract_risk(token_address)`
  - `_analyze_developer_risk(token_address)`
  - `_analyze_holder_risk(token_address)`
  - `_analyze_liquidity_risk(token_address)`
  - `_analyze_market_risk(token_address)`
  - `_analyze_social_risk(token_address)`
  - `_analyze_technical_risk(token_address)`
  - `_analyze_volume_risk(token_address)`
  - `analyze_token(token_address, force_refresh)`
  - `calculate_position_size(opportunity)`
  - `calculate_sharpe_ratio()`
  - `calculate_sortino_ratio()`
  - `calculate_stop_loss(risk_score)`
  - `calculate_take_profit(risk_score, market_conditions)`
  - `calculate_var(confidence)`
  - `check_correlation_limit(token)`
  - `check_portfolio_exposure()`
  - `check_position_limit(token)`
  - `emergency_stop_check()`
  - `initialize()`
  - `set_stop_loss(position)`
  - `validate_trade(token_address, amount)`
- **RiskScore**
  - `overall_risk()`
  - `risk_level()`
- **TradingOpportunity**

## count_loc
- File: `count_loc.py`

### Functions
- `count_loc_in_file(file_path)`
- `count_project_loc(root_dir)`

## data
- File: `data\__init__.py`

## data.collectors
- File: `data\collectors\__init__.py`

## data.collectors.chain_data
- File: `data\collectors\chain_data.py`

### Classes & Methods
- **ChainDataCollector**
  - `__init__(config)`
  - `_check_function_in_code(code, function_name)`
  - `_check_liquidity_lock(pair_address, chain)`
  - `_check_renounced_ownership(address, chain)`
  - `_get_contract_creation(address, chain)`
  - `_safe_call(func)`
  - `_setup_connections()`
  - `analyze_contract(contract_address, chain)`
  - `check_honeypot_onchain(token_address, chain)`
  - `estimate_transaction_cost(from_addr, to_addr, value, chain)`
  - `get_block_number(chain)`
  - `get_gas_price(chain)`
  - `get_holder_distribution(token_address, chain)`
  - `get_liquidity_info(pair_address, chain)`
  - `get_recent_transactions(address, chain, limit)`
  - `get_token_balance(address, token, chain)`
  - `get_token_info(token_address, chain)`
  - `get_transaction(tx_hash, chain)`
  - `get_web3(chain)`
  - `monitor_mempool(chain)`
- **ContractInfo**
- **LiquidityInfo**
- **TokenInfo**
- **TransactionInfo**

## data.collectors.dexscreener
- File: `data\collectors\dexscreener.py`

### Functions
- `test_api_connection()`

### Classes & Methods
- **DexScreenerCollector**
  - `__init__(config)`
  - `_detect_changes(previous, current)`
  - `_filter_pair(pair)`
  - `_make_request(endpoint, params)`
  - `_pair_to_dict(pair)`
  - `_parse_pair(data)`
  - `_rate_limit()`
  - `calculate_metrics(pair_address)`
  - `close()`
  - `get_boosts()`
  - `get_gainers_losers(chain, period)`
  - `get_new_pairs(chain)`
  - `get_pair_data(pair_address)`
  - `get_price_history(address, chain, interval)`
  - `get_stats()`
  - `get_token_info(address, chain)`
  - `get_token_pairs(token_address)`
  - `get_token_price(token_address)`
  - `get_trending_pairs()`
  - `get_trending_tokens(chain)`
  - `initialize()`
  - `monitor_pair(address, chain)`
  - `monitor_pair_with_callback(pair_address, callback)`
  - `search_pairs(query)`
  - `stop_monitoring(pair_address)`
- **TokenPair**
  - `age_hours()`
  - `buy_sell_ratio()`

## data.collectors.honeypot_checker
- File: `data\collectors\honeypot_checker.py`

### Classes & Methods
- **HoneypotChecker**
  - `__init__(config)`
  - `_calculate_verdict(checks)`
  - `_check_contract_verification(address, chain)`
  - `_check_dextools(address, chain)`
  - `_check_erc20_interface(w3, address)`
  - `_check_goplus(address, chain)`
  - `_check_honeypot_is(address, chain)`
  - `_check_tokensniffer(address, chain)`
  - `_get_chain_id(chain)`
  - `_is_blacklisted(address)`
  - `_setup_web3_connections()`
  - `analyze_contract_code(address, chain)`
  - `batch_check(tokens)`
  - `check_liquidity_locks(address, chain)`
  - `check_multiple_apis(address, chain)`
  - `check_token(address, chain)`
  - `close()`
  - `get_risk_score(check_result)`
  - `get_statistics()`
  - `initialize()`
  - `monitor_token(address, chain, interval)`
  - `remove_from_blacklist(address)`
  - `update_blacklist(address, reason)`

## data.collectors.mempool_monitor
- File: `data\collectors\mempool_monitor.py`

### Classes & Methods
- **MEVOpportunity**
- **MempoolMonitor**
  - `__init__(config)`
  - `_cleanup_old_txs()`
  - `_decode_swap(input_data, router_address)`
  - `_detect_arbitrage()`
  - `_detect_liquidations()`
  - `_load_mev_bots()`
  - `_process_pending_tx(tx_hash, chain)`
  - `_setup_connections()`
  - `add_mev_bot(address)`
  - `analyze_pending_tx(tx_hash)`
  - `check_sandwich_risk(tx_hash)`
  - `detect_frontrun_risk(transaction)`
  - `detect_mev_opportunities()`
  - `get_gas_prices(chain)`
  - `get_mempool_stats(chain)`
  - `monitor_mempool(chain)`
  - `monitor_token_mempool(token_address, chain)`
  - `start_monitoring(chain)`
  - `stop_monitoring(chain)`
- **PendingTransaction**
- **SandwichRisk**

## data.collectors.social_data
- File: `data\collectors\social_data.py`

### Classes & Methods
- **InfluencerMention**
- **SentimentLevel**
- **SocialDataCollector**
  - `__init__(config)`
  - `_calculate_fear_index(sentiment, negative_count)`
  - `_calculate_fomo_index(mentions, sentiment, engagement)`
  - `_calculate_overall_sentiment(messages)`
  - `_calculate_trending_score(mentions, engagement, unique_users)`
  - `_classify_sentiment(score)`
  - `_collect_reddit_data(symbol)`
  - `_collect_telegram_data(symbol)`
  - `_collect_twitter_data(symbol)`
  - `_count_influential_mentions(twitter_data, reddit_data, telegram_data)`
  - `_count_negative_keywords(messages)`
  - `_get_previous_metrics(token_address)`
  - `_get_trending_tokens()`
  - `_load_influencers()`
  - `_store_metrics(metrics)`
  - `_test_connections()`
  - `analyze_social_velocity(token_address, symbol)`
  - `calculate_social_score(metrics)`
  - `collect_social_metrics(token_address, symbol, chain)`
  - `get_influencer_mentions(token_symbol, hours)`
  - `initialize()`
  - `monitor_trending()`
- **SocialMetrics**
- **SocialPlatform**

## data.collectors.token_sniffer
- File: `data\collectors\token_sniffer.py`

### Classes & Methods
- **TokenAnalysis**
- **TokenFlag**
- **TokenMetrics**
- **TokenRisk**
- **TokenSniffer**
  - `__init__(config)`
  - `_analyze_onchain(token_address, chain)`
  - `_calculate_confidence(sources_checked)`
  - `_calculate_risk(data)`
  - `_check_dextools(token_address, chain)`
  - `_check_goplus(token_address, chain)`
  - `_check_honeypot(token_address, chain)`
  - `_check_token_sniffer(token_address, chain)`
  - `_combine_analysis_data()`
  - `_create_blacklisted_result(token_address, chain)`
  - `_create_error_result(token_address, chain, error)`
  - `_find_similar_scams(name, symbol)`
  - `_generate_recommendations(risk_level, flags)`
  - `_get_chain_id(chain)`
  - `_get_dextools_chain(chain)`
  - `_load_known_scams()`
  - `_load_lists()`
  - `_parse_dextools_response(data)`
  - `_parse_goplus_response(data, token_address)`
  - `_parse_honeypot_response(data)`
  - `_parse_token_sniffer_response(data)`
  - `analyze_token(token_address, chain, deep_scan)`
  - `batch_analyze(tokens, deep_scan)`
  - `cleanup()`
  - `clear_cache()`
  - `export_analysis(analysis, format)`
  - `get_cache_stats()`
  - `get_risk_summary(analysis)`
  - `initialize()`
  - `monitor_token(token_address, chain, interval, callback)`
  - `update_blacklist(token_address, reason)`
  - `update_whitelist(token_address)`

## data.collectors.volume_analyzer
- File: `data\collectors\volume_analyzer.py`

### Classes & Methods
- **TradeCluster**
- **VolumeAnalyzer**
  - `__init__(config)`
  - `_analyze_address_patterns(trades)`
  - `_analyze_dex_distribution(trades)`
  - `_analyze_price_impact(trades)`
  - `_analyze_smart_money(trades)`
  - `_analyze_time_distribution(trades)`
  - `_analyze_trade_sizes(trades)`
  - `_analyze_trade_timing(trades)`
  - `_analyze_trader_types(trades)`
  - `_calculate_avg_trade_size(trades)`
  - `_calculate_confidence(trades_count, unique_traders, time_coverage)`
  - `_calculate_time_coverage(trades, expected_hours)`
  - `_calculate_trade_similarity(trade1, trade2)`
  - `_calculate_velocity(trades)`
  - `_detect_accumulation_distribution(trades)`
  - `_detect_bot_trading(trades)`
  - `_detect_breakout(trades)`
  - `_detect_circular_trades(trades)`
  - `_detect_pump_pattern(trades)`
  - `_detect_wash_trading(trades)`
  - `_empty_profile(token_address, chain)`
  - `_fetch_trades(token_address, chain, hours)`
  - `_find_trade_clusters(trades)`
  - `_identify_patterns(trades)`
  - `_load_bot_addresses()`
  - `_load_known_wash_traders()`
  - `_separate_volume(trades, clusters)`
  - `analyze_volume(token_address, chain, time_window)`
  - `initialize()`
  - `monitor_volume_health(token_address, chain, callback)`
- **VolumePattern**
- **VolumeProfile**

## data.collectors.whale_tracker
- File: `data\collectors\whale_tracker.py`

### Classes & Methods
- **WhaleTracker**
  - `__init__(config)`
  - `_calculate_risk_level(impact_score, movements)`
  - `_generate_recommendation(impact_score, movements)`
  - `_get_chain_id(chain)`
  - `_get_erc20_abi()`
  - `_get_token_decimals(contract)`
  - `_get_token_price(token, chain_id)`
  - `_get_top_holders(token, chain_id)`
  - `_get_wallet_history(wallet, chain_id)`
  - `_load_known_whales()`
  - `_setup_web3_connections()`
  - `_track_recent_transfers(token, whale_wallets, chain_id)`
  - `analyze_whale_behavior(wallet, chain)`
  - `close()`
  - `get_statistics()`
  - `get_whale_impact_score(movements)`
  - `identify_whale_wallets(token, chain)`
  - `initialize()`
  - `monitor_whale_alerts(token, chain, callback)`
  - `track_whale_movements(token, chain)`

## data.processors
- File: `data\processors\__init__.py`

## data.processors.aggregator
- File: `data\processors\aggregator.py`

### Classes & Methods
- **DataAggregator**
  - `__init__(config)`
  - `_aggregate_holder_metrics(sources)`
  - `_aggregate_liquidity(sources)`
  - `_aggregate_prices(sources)`
  - `_aggregate_social_metrics(sources)`
  - `_aggregate_technical_indicators(sources)`
  - `_aggregate_volumes(sources)`
  - `_calculate_quality_score(sources)`
  - `_clean_source_data(data)`
  - `_detect_anomalies(aggregated, sources)`
  - `_record_conflict(field, sources)`
  - `_validate_source_data(data)`
  - `aggregate_market_data(chain, data_sources)`
  - `aggregate_token_data(token_address, chain, data_sources)`
  - `get_conflict_report()`
  - `get_reliability_report()`
  - `merge_time_series(series_list, method)`
  - `update_source_reliability(source, accuracy_score)`

## data.processors.feature_extractor
- File: `data\processors\feature_extractor.py`

### Classes & Methods
- **FeatureExtractor**
  - `__init__(config)`
  - `_detect_doji(data)`
  - `_detect_double_bottom(prices)`
  - `_detect_double_top(prices)`
  - `_detect_engulfing(data)`
  - `_detect_flag(prices)`
  - `_detect_hammer(data)`
  - `_detect_head_shoulders(prices)`
  - `_detect_triangle(prices)`
  - `create_feature_vector(features, feature_list)`
  - `extract_all_features(data)`
  - `extract_chain_features(data)`
  - `extract_derived_features(features)`
  - `extract_interaction_features(features)`
  - `extract_market_features(data)`
  - `extract_pattern_features(data)`
  - `extract_price_features(data)`
  - `extract_risk_features(data)`
  - `extract_social_features(data)`
  - `extract_technical_indicators(data)`
  - `extract_volume_features(data)`
  - `get_feature_importance(model, feature_list)`

## data.processors.normalizer
- File: `data\processors\normalizer.py`

### Classes & Methods
- **DataNormalizer**
  - `__init__(config)`
  - `_create_record_key(record)`
  - `_default_config()`
  - `export_schema(schema)`
  - `merge_normalized_data()`
  - `normalize_address(address)`
  - `normalize_batch(data, schema)`
  - `normalize_chain(chain)`
  - `normalize_dataframe(df, schema)`
  - `normalize_dex(dex)`
  - `normalize_market_cap(market_cap)`
  - `normalize_percentage(percentage)`
  - `normalize_price(price)`
  - `normalize_record(record, schema)`
  - `normalize_symbol(symbol)`
  - `normalize_timestamp(timestamp)`
  - `normalize_transaction(tx)`
  - `normalize_value(value, data_type)`
  - `normalize_volume(volume)`
  - `validate_normalized_data(data, schema)`
- **DataType**
- **NormalizationConfig**

## data.processors.validator
- File: `data\processors\validator.py`

### Classes & Methods
- **DataField**
- **DataValidator**
  - `__init__(config)`
  - `_apply_rule(rule, field, value)`
  - `_calculate_quality_score(errors, warnings, info)`
  - `_check_consistency(data)`
  - `_check_required_fields(data)`
  - `_determine_field_type(field)`
  - `_generate_suggestions(errors, warnings)`
  - `_initialize_rules()`
  - `_summarize_errors(results)`
  - `_summarize_warnings(results)`
  - `_validate_address(value)`
  - `_validate_liquidity(value)`
  - `_validate_percentage(value)`
  - `_validate_price(value)`
  - `_validate_schema(data, schema)`
  - `_validate_symbol(value)`
  - `_validate_timestamp(value)`
  - `_validate_volume(value)`
  - `add_rule(rule)`
  - `export_rules()`
  - `get_validation_stats()`
  - `import_rules(rules_data)`
  - `register_custom_validator(name, validator_func)`
  - `remove_rule(field, rule_type)`
  - `reset_stats()`
  - `validate(data, schema)`
  - `validate_batch(data_list, schema)`
- **ValidationLevel**
- **ValidationResult**
- **ValidationRule**

## data.storage
- File: `data\storage\__init__.py`

## data.storage.cache
- File: `data\storage\cache.py`

### Classes & Methods
- **CacheManager**
  - `__init__(config)`
  - `_setup_keyspace_notifications()`
  - `acquire_lock(resource, timeout, blocking)`
  - `clear(pattern)`
  - `connect()`
  - `decr(key, amount)`
  - `delete(key)`
  - `disconnect()`
  - `exists(key)`
  - `expire(key, seconds)`
  - `get(key)`
  - `get_many(keys)`
  - `get_stats()`
  - `get_with_options(key, default, decode_json)`
  - `hget(name, key)`
  - `hgetall(name)`
  - `hset(name, key, value)`
  - `incr(key, amount)`
  - `invalidate(pattern)`
  - `lpush(key, value)`
  - `lrange(key, start, stop)`
  - `publish(channel, message)`
  - `release_lock(lock)`
  - `rpop(key)`
  - `set(key, value, ttl, cache_type)`
  - `set_many(mapping, ttl)`
  - `subscribe()`
  - `ttl(key)`
  - `warm_cache(data_type, data)`
  - `zadd(key, mapping)`
  - `zrange(key, start, stop, withscores)`

## data.storage.database
- File: `data\storage\database.py`

### Classes & Methods
- **DatabaseManager**
  - `__init__(config)`
  - `_create_tables()`
  - `_initialize_timescaledb()`
  - `acquire()`
  - `cleanup_old_data(days)`
  - `connect()`
  - `disconnect()`
  - `get_active_positions()`
  - `get_historical_data(token, timeframe)`
  - `get_historical_data_extended(token_address, timeframe, limit, chain)`
  - `get_recent_trades(limit, status)`
  - `get_statistics()`
  - `get_token_analysis(token_address, chain, hours_back)`
  - `save_market_data(data)`
  - `save_market_data_batch(data_points)`
  - `save_performance_metrics(metrics)`
  - `save_position(position)`
  - `save_token_analysis(analysis)`
  - `save_trade(trade)`
  - `update_position(position_id, updates)`
  - `update_trade(trade_id, updates)`

## data.storage.models
- File: `data\storage\models.py`

### Functions
- `create_all_tables(engine)`
- `drop_all_tables(engine)`

### Classes & Methods
- **Alert**
  - `to_dict()`
- **AlertPriority**
- **MEVTransaction**
- **MarketData**
- **PerformanceMetrics**
  - `to_dict()`
- **Position**
  - `to_dict()`
- **PositionStatus**
- **SystemLog**
- **TokenAnalysis**
  - `to_dict()`
- **Trade**
  - `to_dict()`
  - `validate_side(key, value)`
  - `validate_status(key, value)`
- **TradeSide**
- **TradeStatus**
- **WhaleWallet**

## main
- File: `main.py`

### Functions
- `check_wallet_balance()`
- `close_all_connections()`
- `main()`
- `parse_arguments()`
- `setup_logger(name, mode)`
- `test_api_connection()`
- `test_connection()`
- `test_redis_connection()`
- `test_web3_connection()`
- `verify_models_loaded()`

### Classes & Methods
- **HealthChecker**
  - `__init__(engine)`
  - `monitor()`
- **TradingBotApplication**
  - `__init__(config_path, mode)`
  - `_check_apis()`
  - `_check_database()`
  - `_check_models()`
  - `_check_redis()`
  - `_check_wallet()`
  - `_check_web3()`
  - `_perform_system_checks()`
  - `_shutdown_monitor()`
  - `_signal_handler(signum, frame)`
  - `_status_reporter()`
  - `_validate_environment()`
  - `initialize()`
  - `run()`
  - `shutdown()`

## ml
- File: `ml\__init__.py`

## ml.models
- File: `ml\models\__init__.py`

## ml.models.ensemble_model
- File: `ml\models\ensemble_model.py`

### Classes & Methods
- **EnsemblePredictor**
  - `__init__(model_dir)`
  - `_calculate_weighted_average(values, weights)`
  - `_create_gradient_boosting_model()`
  - `_create_lightgbm_model()`
  - `_create_random_forest_model()`
  - `_create_xgboost_model()`
  - `_get_feature_importance(features)`
  - `_predict_from_features(features)`
  - `_train_lstm(X, y, epochs)`
  - `_train_transformer(X, y, epochs)`
  - `calculate_weighted_score(scores, weights)`
  - `combine_predictions(predictions)`
  - `extract_features(data)`
  - `get_confidence_level(predictions)`
  - `load_models()`
  - `predict(token, chain)`
  - `predict_from_token(token, chain)`
  - `retrain(training_data)`
  - `save_models()`
  - `update_models(new_models)`
  - `update_weights(performance_data)`
- **LSTMPricePredictor**
  - `__init__(input_dim, hidden_dim, num_layers, dropout)`
  - `forward(x)`
- **PredictionResult**
  - `__post_init__()`
- **TransformerPredictor**
  - `__init__(input_dim, d_model, nhead, num_layers)`
  - `forward(x)`

## ml.models.pump_predictor
- File: `ml\models\pump_predictor.py`

### Classes & Methods
- **PumpPredictor**
  - `__init__(config)`
  - `_build_lstm_model()`
  - `_calculate_adx(data, period)`
  - `_calculate_bollinger_bands(prices, period, std_dev)`
  - `_calculate_ema(data, period)`
  - `_calculate_macd(prices, fast, slow, signal)`
  - `_calculate_market_features(data)`
  - `_calculate_obv(data)`
  - `_calculate_rsi(prices, period)`
  - `_calculate_stochastic(data, period, smooth)`
  - `_calculate_technical_indicators(data)`
  - `_calculate_vwap(data)`
  - `_detect_accumulation(data)`
  - `_detect_breakout(data)`
  - `_generate_pump_signals(data, pump_probability)`
  - `_identify_pump_patterns(data)`
  - `_initialize_models()`
  - `backtest(historical_data)`
  - `extract_features(market_data)`
  - `load_model(version)`
  - `predict_pump_probability(features)`
  - `predict_pump_probability_detailed(current_data)`
  - `prepare_features(market_data)`
  - `prepare_sequences(price_data, lookback)`
  - `save_model(version)`
  - `train(price_history, pump_labels)`
  - `train_lstm(sequences, targets)`

## ml.models.rug_classifier
- File: `ml\models\rug_classifier.py`

### Classes & Methods
- **RugClassifier**
  - `__init__(config)`
  - `_calculate_confidence(model_predictions)`
  - `_calculate_ensemble_importance()`
  - `_calculate_liquidity_removal_risk(token_data)`
  - `_ensemble_predict_proba(X)`
  - `_generate_warnings(token_data, rug_probability)`
  - `_get_recommendation(rug_probability)`
  - `_identify_red_flags(token_data)`
  - `_initialize_models()`
  - `analyze_token(token_data)`
  - `extract_features(token_data)`
  - `load_model(version)`
  - `predict(token_features)`
  - `save_model(version)`
  - `train(historical_data, labels)`
  - `update_model(new_data, new_labels)`

## ml.models.volume_validator
- File: `ml\models\volume_validator.py`

### Classes & Methods
- **VolumeValidationResult**
- **VolumeValidatorML**
  - `__init__(config)`
  - `_ensemble_predict(features)`
  - `_estimate_real_volume(volume_data, is_genuine, confidence)`
  - `_generate_recommendation(is_genuine, confidence, anomaly_score, risk_factors)`
  - `_get_feature_importance()`
  - `_heuristic_validation(volume_data)`
  - `_identify_risk_factors(volume_data, anomaly_score)`
  - `extract_features(volume_data)`
  - `get_model_performance()`
  - `load_model(filepath)`
  - `needs_retraining(days_threshold)`
  - `predict(volume_data)`
  - `save_model(filepath)`
  - `train(training_data, labels, validation_split)`

## ml.optimization
- File: `ml\optimization\__init__.py`

## ml.training
- File: `ml\training\__init__.py`

## monitoring
- File: `monitoring\__init__.py`

## monitoring.alerts
- File: `monitoring\alerts.py`

### Classes & Methods
- **Alert**
- **AlertManager**
  - `__init__(config)`
  - `send_critical(message)`
  - `send_error(message)`
  - `send_info(message)`
  - `send_trade_alert(message)`
  - `send_warning(message)`
- **AlertPriority**
- **AlertRule**
- **AlertType**
- **AlertsSystem**
  - `__init__(config)`
  - `_aggregate_alerts()`
  - `_check_rate_limit(channel)`
  - `_default_config()`
  - `_format_aggregated_alerts(alerts)`
  - `_format_arbitrage_opportunity(data)`
  - `_format_correlation_warning(data)`
  - `_format_daily_summary(metrics)`
  - `_format_drawdown_alert(data)`
  - `_format_margin_call(data)`
  - `_format_performance_metrics(metrics)`
  - `_format_position_closed(position, pnl)`
  - `_format_position_opened(position)`
  - `_format_risk_warning(data)`
  - `_format_signal(data, confidence)`
  - `_format_stop_loss(position, loss)`
  - `_format_take_profit(position, profit)`
  - `_format_volume_surge(data)`
  - `_format_weekly_report(metrics)`
  - `_format_whale_movement(data)`
  - `_generate_alert_id()`
  - `_get_channels_for_priority(priority)`
  - `_get_color_for_priority(priority)`
  - `_get_emoji(alert_type)`
  - `_initialize_channels()`
  - `_notify_subscribers(alert)`
  - `_process_alerts()`
  - `_send_alert_to_channels(alert)`
  - `_send_discord(alert)`
  - `_send_email(alert)`
  - `_send_pushover(alert)`
  - `_send_slack(alert)`
  - `_send_sms(alert)`
  - `_send_telegram(alert)`
  - `_send_to_channel(alert, channel)`
  - `_send_webhook(alert)`
  - `_should_send_alert(alert)`
  - `add_rule(name, condition, alert_type, priority, channels, cooldown)`
  - `check_rules(data)`
  - `format_alert(alert_type, data)`
  - `get_alert_stats()`
  - `send_alert(alert_type, message, data)`
  - `send_alert_internal(alert_type, title, message, priority, data, channels)`
  - `send_discord(message)`
  - `send_email(subject, body)`
  - `send_opportunity_alert(opportunity_type, data, confidence)`
  - `send_performance_summary(period, metrics)`
  - `send_risk_alert(risk_type, severity, data)`
  - `send_telegram(message)`
  - `send_trading_alert(event_type, position)`
  - `set_channel_enabled(channel, enabled)`
  - `subscribe_to_alerts(alert_type, callback)`
  - `update_channel_config(channel, config)`
- **ChannelConfig**
- **NotificationChannel**

## monitoring.dashboard
- File: `monitoring\dashboard.py`

### Classes & Methods
- **ChartData**
- **Dashboard**
  - `__init__(host, port, config)`
  - `_broadcast_updates()`
  - `_cancel_order(order_id)`
  - `_close_position(position_id)`
  - `_default_config()`
  - `_execute_action(action, params)`
  - `_modify_order(order_id, mods)`
  - `_modify_position(position_id, mods)`
  - `_serialize_dashboard_data()`
  - `_setup_routes()`
  - `_setup_socketio_handlers()`
  - `_toggle_strategy(strategy, enabled)`
  - `_update_dashboard_data()`
  - `add_alert(alert)`
  - `add_chart(chart_id, chart_type, title, data, options, update_interval)`
  - `alerts_handler(request)`
  - `cancel_order_handler(request)`
  - `chart_handler(request)`
  - `close_position_handler(request)`
  - `create_portfolio_chart()`
  - `create_positions_chart()`
  - `create_risk_gauge()`
  - `create_volume_chart()`
  - `create_win_rate_chart()`
  - `dashboard_handler(request)`
  - `dismiss_alert_handler(request)`
  - `generate_charts(data)`
  - `get_dashboard_data()`
  - `index_handler(request)`
  - `modify_order_handler(request)`
  - `modify_position_handler(request)`
  - `orders_handler(request)`
  - `performance_handler(request)`
  - `positions_handler(request)`
  - `risk_handler(request)`
  - `sse_handler(request)`
  - `start()`
  - `start_dashboard(port)`
  - `status_handler(request)`
  - `toggle_strategy_handler(request)`
  - `update_chart_data(chart_id, data)`
  - `update_dashboard_data(portfolio_value, cash_balance, positions_value, daily_pnl, total_pnl, open_positions, pending_orders, win_rate, sharpe_ratio, max_drawdown, active_alerts)`
  - `update_metrics(metrics)`
  - `update_orders(orders)`
  - `update_performance(performance)`
  - `update_positions(positions)`
  - `update_risk(risk)`
- **DashboardData**
- **DashboardManager**
  - `__init__(dashboard)`
  - `_update_alerts_data()`
  - `_update_orders_data()`
  - `_update_performance_data()`
  - `_update_portfolio_data()`
  - `_update_positions_data()`
  - `_update_risk_data()`
  - `connect_to_trading_system(engine, portfolio_manager, order_manager, alerts_system)`
- **DashboardSection**

## monitoring.logger
- File: `monitoring\logger.py`

### Classes & Methods
- **ColoredFormatter**
  - `format(record)`
- **JsonFormatter**
  - `format(record)`
- **MetricType**
- **PerformanceSnapshot**
- **PerformanceTracker**
  - `__init__(config)`
  - `_calculate_snapshots()`
  - `_create_snapshot(period)`
  - `_default_config()`
  - `_init_database()`
  - `_initialize_metrics()`
  - `_log_trade(trade)`
  - `_persist_data()`
  - `_update_metrics(trade)`
  - `get_performance_report(period)`
  - `record_trade(trade)`
- **StandardFormatter**
  - `__init__()`
- **StructuredLogger**
  - `__init__(name, config)`
  - `_default_config()`
  - `_get_formatter(output_type)`
  - `_save_error_to_file(error_data)`
  - `log_error(error, context)`
  - `log_performance(metrics)`
  - `log_trade(trade)`
  - `setup_logging(config)`
- **TradeFormatter**
  - `format(record)`
- **TradeRecord**

## monitoring.performance
- File: `monitoring\performance.py`

### Classes & Methods
- **ColoredFormatter**
  - `format(record)`
- **JsonFormatter**
  - `format(record)`
- **MetricType**
- **PerformanceSnapshot**
- **PerformanceTracker**
  - `__init__(config)`
  - `_calculate_snapshots()`
  - `_calculate_var(confidence_level)`
  - `_create_snapshot(period)`
  - `_default_config()`
  - `_init_database()`
  - `_initialize_metrics()`
  - `_log_trade(trade)`
  - `_persist_data()`
  - `_update_metrics(trade)`
  - `calculate_max_drawdown(equity_curve)`
  - `calculate_metrics()`
  - `calculate_sharpe_ratio(returns)`
  - `generate_report()`
  - `get_performance_report(period)`
  - `record_trade(trade)`
  - `track_trade(trade)`
- **StandardFormatter**
  - `__init__()`
- **StructuredLogger**
  - `__init__(name, config)`
  - `_default_config()`
  - `_get_formatter(output_type)`
  - `setup_logging()`
- **TradeFormatter**
  - `format(record)`
- **TradeRecord**

## scaffold
- File: `scaffold.py`

### Functions
- `ensure_dir(path)`
- `ensure_file(path)`

## scripts.health_check
- File: `scripts\health_check.py`

### Functions
- `check_health()`
- `main()`

## scripts.init_config
- File: `scripts\init_config.py`

### Functions
- `init_config()`

## scripts.run_tests
- File: `scripts\run_tests.py`

### Functions
- `main()`

### Classes & Methods
- **TestRunner**
  - `__init__()`
  - `check_redis()`
  - `check_test_database()`
  - `cleanup()`
  - `create_test_database()`
  - `generate_report()`
  - `parse_results(result)`
  - `run_tests(test_type, coverage, parallel, verbose, markers)`
  - `setup_environment()`
  - `start_redis()`

## scripts.setup_database
- File: `scripts\setup_database.py`

### Functions
- `setup_database()`

## scripts.verify_claudedex_plus
- File: `scripts\verify_claudedex_plus.py`

### Functions
- `_ann_to_str(node)`
- `_check_visibility(expected, found_variants, file_path)`
- `_check_wrapper(expected, found_variants, file_path)`
- `_compare_signature(expected, found_variants, file_path)`
- `_norm(s)`
- `_params_from_args(args)`
- `_parse_tree_block_lines(text)`
- `_read_text(path)`
- `_resolve_filename_under_root(root, rel_path)`
- `_visibility_from_name(name)`
- `collect_file_symbols(py_path)`
- `generate_reports(root, expected_map, all_files_to_check, include_cross_file_duplicates)`
- `main()`
- `parse_expectations_from_md(md_path)`
- `parse_structure_md(md_path)`
- `render_markdown(reports, cross_dups)`

### Classes & Methods
- **ExpectedItem**
- **FileReport**
- **FoundMethod**
- **_AstCollector**
  - `__init__()`
  - `_visit_func(node, is_async)`
  - `visit_AsyncFunctionDef(node)`
  - `visit_ClassDef(node)`
  - `visit_FunctionDef(node)`
- **_CallCollector**
  - `__init__()`
  - `visit_Call(node)`

## scripts.verify_claudedex_plus2
- File: `scripts\verify_claudedex_plus2.py`

### Functions
- `_ann_to_str(node)`
- `_check_visibility(expected, found_variants, file_path)`
- `_check_wrapper(expected, found_variants, file_path)`
- `_compare_signature(expected, found_variants, file_path)`
- `_enforce_interfaces(class_bases, methods)`
- `_norm(s)`
- `_params_from_args(args)`
- `_parse_tree_block_lines(text)`
- `_read_text(path)`
- `_resolve_filename_under_root(root, rel_path)`
- `_visibility_from_name(name)`
- `collect_file_symbols(py_path)`
- `generate_reports(root, expected_map, all_files_to_check, include_cross_file_duplicates)`
- `main()`
- `parse_expectations_from_md(md_path)`
- `parse_structure_md(md_path)`
- `render_markdown(reports, cross_dups)`

### Classes & Methods
- **ExpectedItem**
- **FileReport**
- **FoundMethod**
- **_AstCollector**
  - `__init__()`
  - `_visit_func(node, is_async)`
  - `visit_AsyncFunctionDef(node)`
  - `visit_ClassDef(node)`
  - `visit_FunctionDef(node)`
- **_CallCollector**
  - `__init__()`
  - `visit_Call(node)`

## scripts.verify_claudedex_plus3
- File: `scripts\verify_claudedex_plus3.py`

### Functions
- `_ann_to_str(node)`
- `_closest(names, target, n, cutoff)`
- `_compare_signature(expected, found_variants, file_path)`
- `_enforce_interfaces(class_bases, methods)`
- `_norm(s)`
- `_params_from_args(args)`
- `_read_text(path)`
- `_resolve_filename_under_root(root, rel_path)`
- `_split_params_safely(raw)`
- `_visibility_from_name(name)`
- `collect_file_symbols(py_path)`
- `generate_reports(root, expected_map, all_files_to_check)`
- `main()`
- `parse_expectations_from_md(md_path)`
- `parse_structure_md(md_path)`

### Classes & Methods
- **ExpectedItem**
- **FileReport**
- **FoundMethod**
- **_AstCollector**
  - `__init__()`
  - `_visit_func(node, is_async)`
  - `visit_AsyncFunctionDef(node)`
  - `visit_ClassDef(node)`
  - `visit_FunctionDef(node)`
- **_CallCollector**
  - `__init__()`
  - `visit_Call(node)`

## security
- File: `security\__init__.py`

## security.api_security
- File: `security\api_security.py`

### Classes & Methods
- **APISecurityManager**
  - `__init__(config)`
  - `_check_ip_access(ip)`
  - `_check_permissions(api_key, endpoint, method)`
  - `_load_api_keys()`
  - `_log_request(api_key, endpoint, success)`
  - `_save_api_keys()`
  - `_validate_api_key(api_key)`
  - `_validate_signature(request_data, signature, api_key)`
  - `add_ip_blacklist(ip)`
  - `add_ip_whitelist(ip)`
  - `check_rate_limit(identifier, endpoint_type)`
  - `generate_api_key(user_id, permissions, expires_days)`
  - `generate_jwt(payload)`
  - `generate_jwt_token(user_id, expires_minutes)`
  - `rate_limit_check(ip)`
  - `revoke_api_key(api_key)`
  - `validate_api_key(key)`
  - `validate_jwt_token(token)`
  - `validate_request(request_data, endpoint, method)`
  - `verify_jwt(token)`

## security.audit_logger
- File: `security\audit_logger.py`

### Classes & Methods
- **AuditEvent**
- **AuditEventType**
- **AuditLogger**
  - `__init__(config)`
  - `_calculate_checksum(event)`
  - `_calculate_risk_score(event_type, severity, source, action, details, ip_address)`
  - `_check_compliance_rules(event)`
  - `_check_risk_patterns(event_type, source, details, ip_address)`
  - `_cleanup_old_logs()`
  - `_cleanup_old_logs_periodically()`
  - `_compress_log_file(log_file)`
  - `_flush_buffer()`
  - `_flush_buffer_periodically()`
  - `_get_current_log_file()`
  - `_handle_compliance_violation(event, rule)`
  - `_load_compliance_rules()`
  - `_load_risk_patterns()`
  - `_send_immediate_alert(event)`
  - `_start_background_tasks()`
  - `_verify_checksum(event)`
  - `cleanup()`
  - `generate_compliance_report(start_date, end_date, report_type)`
  - `get_audit_trail(filters)`
  - `get_metrics()`
  - `initialize()`
  - `log_access(user, resource, action)`
  - `log_event(event_type, severity, status, source, action, resource, details, user_id, session_id, ip_address, user_agent, correlation_id, parent_event_id, tags)`
  - `log_security_event(event)`
  - `log_transaction(tx_data)`
  - `search_events(start_time, end_time, event_type, severity, user_id, source, ip_address, correlation_id, limit)`
- **AuditSeverity**
- **AuditStatus**

## security.encryption
- File: `security\encryption.py`

### Classes & Methods
- **EncryptionManager**
  - `__init__(config)`
  - `_generate_key_id(key)`
  - `_get_or_create_master_key()`
  - `decrypt_api_key(encrypted_data)`
  - `decrypt_data(encrypted, key)`
  - `decrypt_sensitive_data(encrypted_data)`
  - `decrypt_wallet_key(encrypted_data)`
  - `encrypt_api_key(api_key)`
  - `encrypt_data(data, key)`
  - `encrypt_sensitive_data(data)`
  - `encrypt_wallet_key(private_key)`
  - `generate_key()`
  - `hash_password(password)`
  - `rotate_encryption_key()`
  - `secure_delete(data)`
  - `verify_password(password, hash)`

## security.wallet_security
- File: `security\wallet_security.py`

### Classes & Methods
- **SecurityLevel**
- **TransactionApproval**
- **WalletConfig**
- **WalletSecurityManager**
  - `__init__(config)`
  - `_check_rate_limits(wallet_id)`
  - `_check_transaction_limits(wallet_id, transaction_data)`
  - `_get_wallet_data(wallet_id)`
  - `_initialize_web3_connections()`
  - `_load_security_policies()`
  - `_load_wallet_configurations()`
  - `_lock_wallet(wallet_id, reason)`
  - `_log_transaction(wallet_id, transaction_data, signed_tx)`
  - `_notify_emergency_contacts(reason)`
  - `_perform_security_checks(wallet_id, transaction_data)`
  - `_save_wallet_data(wallet_id, wallet_data)`
  - `_setup_hardware_wallet_address(wallet_type, derivation_path)`
  - `_setup_hardware_wallets()`
  - `_sign_hardware_wallet_transaction(wallet_id, transaction_data, w3)`
  - `_sign_hot_wallet_transaction(wallet_id, transaction_data, w3)`
  - `_sign_multisig_transaction(wallet_id, transaction_data, w3)`
  - `_verify_hardware_wallet(wallet_id, wallet_config)`
  - `backup_wallet(wallet_id, backup_location)`
  - `cleanup()`
  - `create_wallet(wallet_type, security_level, derivation_path)`
  - `create_wallet_simple(wallet_type, security_level)`
  - `emergency_stop(reason)`
  - `encrypt_private_key(private_key)`
  - `get_wallet_status(wallet_id)`
  - `initialize()`
  - `rotate_keys(wallet_id)`
  - `sign_transaction(wallet_id, transaction_data, chain)`
- **WalletType**

## setup
- File: `setup.py`

### Functions
- `read_requirements(file)`

## tests
- File: `tests\__init__.py`

## tests.conftest
- File: `tests\conftest.py`

### Functions
- `audit_logger(tmp_path)`
- `benchmark_data()`
- `cache_manager()`
- `cleanup()`
- `config_manager(tmp_path)`
- `create_mock_token(address)`
- `create_mock_trade()`
- `db_manager()`
- `event_loop()`
- `mock_dex_api()`
- `mock_dex_client()`
- `mock_market_data()`
- `mock_web3()`
- `populate_test_database(db_manager, num_records)`
- `risk_manager()`
- `sample_position()`
- `sample_trading_opportunity()`
- `wallet_security()`

## tests.fixtures.mock_data
- File: `tests\fixtures\mock_data.py`

### Classes & Methods
- **MockDataGenerator**
  - `generate_market_conditions()`
  - `generate_price_history(days, interval, include_pump)`
  - `generate_token(is_rug, is_honeypot, liquidity)`
  - `generate_token_address()`
  - `generate_trades(count)`
  - `generate_whale_movement()`

## tests.fixtures.test_helpers
- File: `tests\fixtures\test_helpers.py`

### Classes & Methods
- **TestHelpers**
  - `assert_position_valid(position)`
  - `assert_trade_valid(trade)`
  - `cleanup_test_data(db_manager, cache_manager)`
  - `compare_decimals(a, b, tolerance)`
  - `create_mock_web3_contract()`
  - `generate_mock_config()`
  - `wait_for_condition(condition_func, timeout, interval)`

## tests.integration.test_data_integration
- File: `tests\integration\test_data_integration.py`

### Classes & Methods
- **TestDataIntegration**
  - `test_batch_processing(db_manager)`
  - `test_data_aggregation_pipeline(db_manager, cache_manager)`
  - `test_dexscreener_to_database(db_manager, mock_dex_api)`
  - `test_honeypot_checker_caching(cache_manager, mock_dex_api)`
  - `test_whale_tracker_integration(db_manager, cache_manager)`

## tests.integration.test_ml_integration
- File: `tests\integration\test_ml_integration.py`

### Classes & Methods
- **TestMLIntegration**
  - `test_ensemble_model(training_data, db_manager)`
  - `test_pump_predictor_training(db_manager)`
  - `test_rug_classifier_training(training_data)`
  - `training_data(db_manager)`

## tests.integration.test_trading_integration
- File: `tests\integration\test_trading_integration.py`

### Classes & Methods
- **TestTradingIntegration**
  - `test_order_execution_flow(db_manager)`
  - `test_position_monitoring_and_exit(db_manager)`
  - `test_strategy_signal_execution(db_manager)`

## tests.performance.test_performance
- File: `tests\performance\test_performance.py`

### Classes & Methods
- **TestPerformance**
  - `test_cache_performance(cache_manager)`
  - `test_concurrent_request_handling(db_manager, cache_manager)`
  - `test_database_write_performance(db_manager, benchmark_data)`
  - `test_memory_usage(benchmark_data)`
  - `test_ml_model_performance(benchmark_data)`
  - `test_order_execution_latency(benchmark)`

## tests.security.test_security
- File: `tests\security\test_security.py`

### Classes & Methods
- **TestSecurity**
  - `test_api_authentication()`
  - `test_api_security_rate_limiting()`
  - `test_audit_logging(audit_logger)`
  - `test_encryption_manager()`
  - `test_input_validation()`
  - `test_private_key_protection()`
  - `test_secure_random_generation()`
  - `test_sql_injection_protection(db_manager)`
  - `test_wallet_security(wallet_security)`

## tests.smoke.test_smoke
- File: `tests\smoke\test_smoke.py`

### Classes & Methods
- **TestSmoke**
  - `test_api_endpoints_available()`
  - `test_cache_connection(cache_manager)`
  - `test_config_loading(config_manager)`
  - `test_database_connection(db_manager)`
  - `test_engine_startup(risk_manager)`
  - `test_ml_model_loading()`

## tests.test_all
- File: `tests\test_all.py`

### Functions
- `run_all_tests()`

## tests.unit.test_engine
- File: `tests\unit\test_engine.py`

### Classes & Methods
- **TestTradingBotEngine**
  - `engine(risk_manager)`
  - `test_analyze_opportunity(engine, mock_dex_api)`
  - `test_close_position(engine, sample_position)`
  - `test_engine_initialization(engine)`
  - `test_error_handling(engine)`
  - `test_execute_opportunity(engine, sample_trading_opportunity)`
  - `test_monitor_positions(engine, sample_position)`
  - `test_safety_checks(engine, sample_trading_opportunity)`
  - `test_start_stop(engine)`

## tests.unit.test_risk_manager
- File: `tests\unit\test_risk_manager.py`

### Classes & Methods
- **TestRiskManager**
  - `risk_manager()`
  - `test_calculate_position_size(risk_manager, sample_trading_opportunity)`
  - `test_calculate_var(risk_manager)`
  - `test_check_correlation_limit(risk_manager)`
  - `test_check_position_limit(risk_manager)`
  - `test_emergency_stop(risk_manager)`
  - `test_portfolio_exposure(risk_manager)`
  - `test_risk_adjusted_returns(risk_manager)`
  - `test_set_stop_loss(risk_manager, sample_position)`

## trading
- File: `trading\__init__.py`

## trading.executors
- File: `trading\executors\__init__.py`

## trading.executors.base_executor
- File: `trading\executors\base_executor.py`

### Functions
- `test_web3_connection()`

### Classes & Methods
- **ExecutionResult**
- **ExecutionRoute**
- **TradeExecutor**
  - `__init__(config)`
  - `_apply_mev_protection(order)`
  - `_approve_token(token_address, spender, amount)`
  - `_execute_1inch(order)`
  - `_execute_uniswap_v2(order)`
  - `_execute_with_retry(order, route)`
  - `_get_1inch_quote(order)`
  - `_get_all_quotes(order)`
  - `_get_token_balance(token_address)`
  - `_get_uniswap_v2_quote(order)`
  - `_get_weth_address()`
  - `_load_abi(contract_type)`
  - `_post_execution_processing(order, result)`
  - `_pre_execution_checks(order)`
  - `_select_best_route(order)`
  - `_send_private_transaction(signed_tx)`
  - `_verify_token_contract(token_address)`
  - `cancel_order(order_id)`
  - `emergency_sell_all()`
  - `execute(order)`
  - `execute_trade(order)`
  - `get_order_status(order_id)`
  - `get_stats()`
  - `modify_order(order_id, updates)`
  - `validate_order(order)`
- **TradeOrder**

## trading.executors.direct_dex
- File: `trading\executors\direct_dex.py`

### Classes & Methods
- **DEXQuote**
- **DirectDEXExecutor**
  - `__init__(config)`
  - `_apply_commit_reveal(tx)`
  - `_build_swap_transaction(order, quote)`
  - `_build_transaction(order)`
  - `_calculate_fee(dex, amount)`
  - `_check_bundle_status(bundle_id)`
  - `_encode_v3_path(tokens)`
  - `_estimate_gas(dex, path, amount, chain)`
  - `_estimate_price_impact(dex, path, amount, chain)`
  - `_find_path(dex, token_in, token_out, chain)`
  - `_get_dex_quote(dex_name, contract, token_in, token_out, amount, chain)`
  - `_get_liquidity(token_a, token_b, chain)`
  - `_get_optimal_gas_price(chain)`
  - `_get_transaction_receipt(tx_hash)`
  - `_initialize_dex_contracts(chain, w3)`
  - `_path_has_liquidity(path, chain)`
  - `_quote_v3(contract, path, amount)`
  - `_randomize_gas_price(tx)`
  - `_send_transaction_with_retry(signed_tx, chain, max_retries)`
  - `_simulate_swap(dex, path, amount, chain)`
  - `_wait_for_confirmation(tx_hash, chain, timeout)`
  - `cancel_order(order_id)`
  - `cleanup()`
  - `execute_trade(order)`
  - `format_token_amount(amount, decimals)`
  - `get_best_quote(token_in, token_out, amount, chain)`
  - `get_order_status(order_id)`
  - `initialize()`
  - `modify_order(order_id, modifications)`
  - `validate_order(order)`

## trading.executors.mev_protection
- File: `trading\executors\mev_protection.py`

### Classes & Methods
- **AttackType**
- **MEVProtectionLayer**
  - `__init__(config)`
  - `_analyze_attack_patterns()`
  - `_analyze_mev_risk(order, transaction)`
  - `_apply_commit_reveal(transaction)`
  - `_apply_dynamic_routing(transaction)`
  - `_apply_gas_randomization(transaction)`
  - `_apply_time_delays(transaction)`
  - `_build_transaction(order)`
  - `_calculate_tx_hash(transaction)`
  - `_check_attacker_activity()`
  - `_check_bundle_status(bundle_id)`
  - `_check_mempool_congestion()`
  - `_create_flashbots_bundle(transaction)`
  - `_detect_mev_threat(transaction)`
  - `_estimate_mev_savings(original_tx, protected_tx, risk_score)`
  - `_execute_flashbots_bundle(bundle_id)`
  - `_execute_private_mempool(transaction)`
  - `_execute_standard(transaction)`
  - `_get_transaction_receipt(tx_hash)`
  - `_is_frontrun_attempt(transaction)`
  - `_is_sandwich_attack(transaction)`
  - `_monitor_mempool()`
  - `_respond_to_threat(threat)`
  - `_route_private_mempool(transaction)`
  - `_send_decoy_transactions(order)`
  - `_sign_flashbots_bundle(bundle)`
  - `cancel_order(order_id)`
  - `cleanup()`
  - `execute_protected_trade(order)`
  - `execute_trade(order)`
  - `format_token_amount(amount, decimals)`
  - `get_order_status(order_id)`
  - `get_protection_stats()`
  - `initialize()`
  - `modify_order(order_id, modifications)`
  - `protect_transaction(order, transaction)`
  - `validate_order(order)`
- **MEVProtectionLevel**
- **MEVThreat**
- **ProtectedTransaction**

## trading.executors.toxisol_api
- File: `trading\executors\toxisol_api.py`

### Classes & Methods
- **ToxiSolAPIExecutor**
  - `__init__(config)`
  - `_build_base_transaction(order)`
  - `_build_toxisol_transaction(order, quote)`
  - `_check_bundle_status(bundle_id)`
  - `_execute_flashbots(signed_tx)`
  - `_execute_standard(signed_tx)`
  - `_generate_signature(payload, timestamp)`
  - `_get_transaction_receipt(tx_hash)`
  - `_rate_limit()`
  - `_test_connection()`
  - `cancel_order(order_id)`
  - `cleanup()`
  - `estimate_gas(token_in, token_out, amount, chain)`
  - `execute_trade(order, quote)`
  - `format_token_amount(amount, decimals)`
  - `get_execution_stats()`
  - `get_order_status(order_id)`
  - `get_quote(token_in, token_out, amount, chain, route_type)`
  - `initialize()`
  - `modify_order(order_id, modifications)`
  - `validate_order(order)`
- **ToxiSolQuote**
- **ToxiSolRoute**

## trading.orders
- File: `trading\orders\__init__.py`

## trading.orders.order_manager
- File: `trading\orders\order_manager.py`

### Classes & Methods
- **ExecutionEngine**
  - `__init__(config)`
  - `cancel_transaction(tx_hash, gas_price)`
  - `get_order_book(token_address)`
  - `get_transaction_status(tx_hash)`
  - `submit_flashbots_bundle(tx_params, gas_price)`
  - `submit_private_transaction(tx_params, gas_price)`
  - `submit_transaction(tx_params)`
- **ExecutionStrategy**
- **Fill**
- **Order**
  - `build_order(token_address, side, amount, order_type)`
- **OrderBook**
- **OrderManager**
  - `__init__(config)`
  - `_cleanup_expired_orders()`
  - `_default_config()`
  - `_execute_iceberg(order)`
  - `_execute_immediate(order)`
  - `_execute_sniper(order)`
  - `_execute_stop_loss(order)`
  - `_execute_twap(order)`
  - `_execute_twap_intervals(child_orders, interval_duration)`
  - `_execute_vwap(order)`
  - `_get_market_data(token_address)`
  - `_get_volume_profile(token_address)`
  - `_initialize_components()`
  - `_monitor_iceberg_refill(parent_order, visible_order, hidden_amount)`
  - `_monitor_order_confirmation(order)`
  - `_monitor_orders()`
  - `_monitor_sniper_trigger(order, trigger_price)`
  - `_monitor_stop_order(stop_order)`
  - `_monitor_trailing_stop(order)`
  - `_prepare_transaction(order)`
  - `_process_order_fill(order, tx_status)`
  - `_trigger_follow_up_orders(order)`
  - `_update_order_book(token_address)`
  - `_update_slippage_metric(slippage)`
  - `_validate_order_params(token_address, side, amount, order_type, price)`
  - `cancel_order(order_id)`
  - `create_order(order)`
  - `create_order_from_params(token_address, side, amount, order_type, price, execution_strategy)`
  - `execute_immediate(order)`
  - `execute_order(order_id)`
  - `execute_sniper(order, trigger_price)`
  - `execute_twap(order, duration, slices)`
  - `get_active_orders()`
  - `get_metrics()`
  - `get_open_orders()`
  - `get_order_status(order_id)`
  - `modify_order(order_id, modifications)`
  - `submit_order(order_id)`
- **OrderRiskMonitor**
  - `__init__(config)`
  - `check_order_risk(token_address, side, amount, existing_orders)`
- **OrderSide**
- **OrderStatus**
- **OrderType**
- **SettlementProcessor**
  - `__init__(config)`
  - `process_settlement(order, fill)`

## trading.orders.position_tracker
- File: `trading\orders\position_tracker.py`

### Classes & Methods
- **PerformanceMetrics**
- **PortfolioSnapshot**
- **Position**
  - `build_position(token_address, token_symbol, position_type, entry_price, entry_amount)`
- **PositionStatus**
- **PositionTracker**
  - `__init__(config)`
  - `_calculate_correlation(token1, token2)`
  - `_calculate_portfolio_beta()`
  - `_calculate_portfolio_metrics()`
  - `_calculate_portfolio_risk()`
  - `_calculate_portfolio_volatility()`
  - `_calculate_position_risk(position)`
  - `_calculate_value_at_risk()`
  - `_check_portfolio_alerts()`
  - `_check_position_limits(token_address, amount)`
  - `_check_position_rules(position)`
  - `_default_config()`
  - `_determine_risk_level(risk_score)`
  - `_execute_position_actions(position, actions)`
  - `_get_current_price(token_address)`
  - `_initialize_performance_metrics()`
  - `_monitor_positions()`
  - `_update_correlation_matrix()`
  - `_update_performance_metrics(closed_position)`
  - `_update_portfolio_value()`
  - `calculate_pnl(position_id)`
  - `check_stop_loss(position_id)`
  - `close_position(position_id)`
  - `close_position_with_details(position_id, exit_price, exit_amount, order_ids, reason)`
  - `get_active_positions()`
  - `get_open_positions()`
  - `get_performance_report()`
  - `get_portfolio_summary()`
  - `open_position(position)`
  - `open_position_from_params(token_address, token_symbol, position_type, entry_price, entry_amount, order_ids)`
  - `update_position(position_id, updates)`
  - `update_position_with_price(position_id, current_price)`
- **PositionType**
- **RiskLevel**

## trading.strategies
- File: `trading\strategies\__init__.py`

### Classes & Methods
- **StrategyManager**
  - `__init__(config)`
  - `get_parameters()`
  - `initialize()`
  - `select_strategy(opportunity)`
  - `update_parameters(new_params)`

## trading.strategies.ai_strategy
- File: `trading\strategies\ai_strategy.py`

### Classes & Methods
- **AIStrategy**
  - `__init__(config)`
  - `_assess_feature_quality(market_data)`
  - `_calculate_bb_position(prices, period)`
  - `_calculate_hurst_exponent(prices)`
  - `_calculate_kurtosis(prices)`
  - `_calculate_log_returns(prices)`
  - `_calculate_macd_signal(prices)`
  - `_calculate_model_agreement(predictions)`
  - `_calculate_momentum(data, period)`
  - `_calculate_order_imbalance(volumes)`
  - `_calculate_resistance_distance(prices)`
  - `_calculate_returns(prices)`
  - `_calculate_rsi(prices, period)`
  - `_calculate_skewness(prices)`
  - `_calculate_support_distance(prices)`
  - `_calculate_technical_score(features, market_data)`
  - `_calculate_trend(prices)`
  - `_calculate_volatility(prices, window)`
  - `_calculate_volume_ratio(volumes, period)`
  - `_calculate_vwap(prices, volumes)`
  - `_check_retrain()`
  - `_check_rug_probability(features, market_data)`
  - `_detect_pattern_strength(prices)`
  - `_determine_signal_strength(score)`
  - `_estimate_spread(prices)`
  - `_extract_features(market_data)`
  - `_generate_signal_from_predictions(predictions, market_data, rug_probability)`
  - `_get_ml_predictions(features, market_data)`
  - `_load_models()`
  - `_prepare_pump_features(features, market_data)`
  - `_record_prediction(signal, predictions)`
  - `_retrain_models(training_data)`
  - `analyze(market_data)`
  - `calculate_indicators(price_data, volume_data)`
  - `explain_prediction(signal)`
  - `get_feature_importance()`
  - `initialize()`
  - `update_prediction_outcome(signal_id, outcome)`
  - `validate_signal(signal, market_data)`

## trading.strategies.base_strategy
- File: `trading\strategies\base_strategy.py`

### Classes & Methods
- **BaseStrategy**
  - `__init__(config)`
  - `_calculate_kelly_position(confidence, account_balance)`
  - `_check_custom_exit_conditions(position, market_data)`
  - `_create_exit_signal(position, reason, current_price)`
  - `_create_order_from_signal(signal)`
  - `_get_strength_multiplier(strength)`
  - `_on_position_closed(position, exit_price, reason)`
  - `_on_position_failed(signal, result)`
  - `_on_position_opened(signal, result)`
  - `_update_performance(pnl_amount, pnl_percent, duration)`
  - `analyze(market_data)`
  - `backtest(historical_data, initial_balance)`
  - `calculate_indicators(price_data, volume_data)`
  - `calculate_max_drawdown(equity_curve)`
  - `calculate_position_size(signal, account_balance, current_price)`
  - `calculate_sharpe_ratio(returns, risk_free_rate)`
  - `calculate_stop_loss(entry_price, signal)`
  - `calculate_take_profit(entry_price, signal)`
  - `check_exit_conditions(position, market_data)`
  - `execute(signal, order_manager)`
  - `get_active_positions()`
  - `get_performance_summary()`
  - `get_signal_history(limit)`
  - `get_trade_history(limit)`
  - `reset()`
  - `should_analyze(token_address, market_data)`
  - `start()`
  - `stop()`
  - `validate_signal(signal, market_data)`
- **SignalStrength**
- **SignalType**
- **StrategyPerformance**
- **StrategyState**
- **TradingSignal**

## trading.strategies.momentum
- File: `trading\strategies\momentum.py`

### Classes & Methods
- **MomentumMetrics**
- **MomentumSignal**
- **MomentumStrategy**
  - `__init__(config)`
  - `_analyze_volume_pattern(market_data)`
  - `_calculate_average_strength()`
  - `_calculate_breakout_probability(market_data)`
  - `_calculate_breakout_stop_loss(entry_price, market_data)`
  - `_calculate_breakout_targets(entry_price, resistance_levels, metrics)`
  - `_calculate_ma_trend(ma_data)`
  - `_calculate_macd_momentum(macd)`
  - `_calculate_momentum_metrics(market_data)`
  - `_calculate_position_size(confidence, risk_score)`
  - `_calculate_price_velocity(price_history)`
  - `_calculate_range(candles)`
  - `_calculate_rsi_momentum(rsi)`
  - `_calculate_smart_money_score(smart_money)`
  - `_calculate_smart_money_stop_loss(entry_price, whale_activity)`
  - `_calculate_smart_money_targets(entry_price, smart_money, smart_score)`
  - `_calculate_smart_money_velocity(smart_money)`
  - `_calculate_success_rate()`
  - `_calculate_trend_stop_loss(entry_price, market_data)`
  - `_calculate_trend_strength(market_data)`
  - `_calculate_trend_targets(entry_price, trend_strength, market_data)`
  - `_calculate_volume_ratio(volume_data)`
  - `_calculate_volume_targets(entry_price, volume_ratio, volume_pattern)`
  - `_check_breakout_momentum(market_data, metrics)`
  - `_check_ma_alignment(market_data)`
  - `_check_reversal_signals(market_data)`
  - `_check_smart_money_momentum(market_data, metrics)`
  - `_check_trend_momentum(market_data, metrics)`
  - `_check_volume_momentum(market_data, metrics)`
  - `_cluster_price_levels(levels, threshold)`
  - `_default_config()`
  - `_detect_consolidation(candles)`
  - `_detect_range_tightening(candles)`
  - `_detect_volume_buildup(candles)`
  - `_enhance_signal(signal, market_data)`
  - `_find_resistance_levels(market_data)`
  - `_get_signal_distribution()`
  - `_initialize_parameters()`
  - `_select_best_signal(signals)`
  - `_validate_signal(signal, market_data)`
  - `analyze(market_data)`
  - `calculate_momentum(prices)`
  - `calculate_position_size(signal)`
  - `execute(signal)`
  - `get_performance_metrics()`
  - `identify_entry_points(momentum_data)`
  - `update_signal(token_address, market_data)`
- **MomentumType**
- **TimeFrame**

## trading.strategies.scalping
- File: `trading\strategies\scalping.py`

### Classes & Methods
- **ScalpingOpportunity**
- **ScalpingSignal**
- **ScalpingStrategy**
  - `__init__(config)`
  - `_analyze_volume_profile(volumes, prices)`
  - `_assess_risk(market_data, indicators, volume_profile)`
  - `_calculate_bollinger_bands(prices, period, std_dev)`
  - `_calculate_confidence(indicators, volume_profile, market_data)`
  - `_calculate_entry_price(current_price, signal, market_data)`
  - `_calculate_expected_profit(entry_price, target_price, stop_loss, confidence)`
  - `_calculate_indicators(prices, volumes)`
  - `_calculate_macd(prices)`
  - `_calculate_momentum(prices, period)`
  - `_calculate_position_size(opportunity)`
  - `_calculate_rsi(prices, period)`
  - `_calculate_stop_loss(entry_price, signal)`
  - `_calculate_target_price(entry_price, signal, indicators)`
  - `_close_position(order_id, reason)`
  - `_determine_time_window(indicators, volume_profile)`
  - `_identify_signal(indicators, market_data)`
  - `_identify_support_resistance(prices)`
  - `_monitor_position(order_id, opportunity)`
  - `_update_statistics()`
  - `_validate_market_conditions(market_data)`
  - `analyze(market_data)`
  - `execute(opportunity, order_manager)`
  - `get_statistics()`

## utils
- File: `utils\__init__.py`

## utils.constants
- File: `utils\constants.py`

### Classes & Methods
- **Chain**
- **DEX**
- **MarketCondition**
- **OrderStatus**
- **OrderType**
- **SignalStrength**
- **TradingMode**

## utils.helpers
- File: `utils\helpers.py`

### Functions
- `batch_request(urls, max_concurrent)`
- `calculate_ema(values, period)`
- `calculate_moving_average(values, window)`
- `calculate_percentage_change(old_value, new_value)`
- `calculate_profit_loss(entry_price, exit_price, amount, fees)`
- `calculate_slippage(expected_price, actual_price)`
- `chunk_list(lst, chunk_size)`
- `deep_merge_dicts(dict1, dict2)`
- `ether_to_wei(ether)`
- `fetch_json(url, headers, timeout)`
- `format_currency(value, symbol, decimals)`
- `format_number(value, decimals)`
- `format_timestamp(timestamp, fmt)`
- `format_token_amount(amount, decimals)`
- `generate_signature(message, secret)`
- `get_timestamp()`
- `get_timestamp_ms()`
- `hash_data(data)`
- `is_market_hours(timezone_str)`
- `is_valid_address(address)`
- `mask_sensitive_data(data, visible_chars)`
- `measure_time(func)`
- `normalize_address(address)`
- `parse_timeframe(timeframe)`
- `process_in_chunks(items, processor, chunk_size)`
- `rate_limit(calls, period)`
- `retry_async(max_retries, delay, exponential_backoff)`
- `round_to_significant_digits(value, sig_digits)`
- `safe_json_loads(json_str, default)`
- `sanitize_input(value, max_length)`
- `to_base_unit(amount, decimals)`
- `truncate_string(text, max_length, suffix)`
- `validate_chain_id(chain_id)`
- `validate_token_symbol(symbol)`
- `wei_to_ether(wei)`

### Classes & Methods
- **TTLCache**
  - `__init__(ttl)`
  - `clear()`
  - `get(key)`
  - `set(key, value)`

## verify_references
- File: `verify_references.py`

### Functions
- `arg_summary(node)`
- `dotted_attr(node)`
- `extract_sig(fn, is_method)`
- `file_for_module(root, module)`
- `is_cfg(path)`
- `is_py(path)`
- `main()`
- `parse_args()`
- `pretty_sig(sig)`
- `read_file(p)`
- `resolve_relative(module, level, imported)`
- `to_module(root, file_path)`

### Classes & Methods
- **Analyzer**
  - `__init__(root, exclude, exclude_glob, strict_config, project_only)`
  - `_check_sig(defsig, given_pos, given_kw)`
  - `_is_glob_excluded(path)`
  - `_target_kind_from_sig(sig, fq)`
  - `analyze()`
  - `ensure_indexed(module)`
  - `index(files)`
  - `locate(fq)`
  - `locate_def(fq)`
  - `report_data()`
  - `report_md(res)`
  - `resolve_call(c)`
  - `resolve_self_chain(call)`
  - `scan_configs(cfg_files, strict)`
  - `walk()`
- **CallCollector**
  - `__init__(mi)`
  - `visit_AsyncFunctionDef(node)`
  - `visit_Call(node)`
  - `visit_ClassDef(node)`
  - `visit_FunctionDef(node)`
- **CallSite**
- **ClassInfo**
- **CollMap**
- **DefInfo**
- **ModuleInfo**
- **Problem**
- **PyIndexer**
  - `__init__(root, filepath, module)`
  - `visit_Assign(node)`
  - `visit_AsyncFunctionDef(node)`
  - `visit_ClassDef(node)`
  - `visit_FunctionDef(node)`
  - `visit_Import(node)`
  - `visit_ImportFrom(node)`

## xref_symbol_db
- File: `xref_symbol_db.py`

### Functions
- `arg_summary(node)`
- `build_file_tree_md(root, files)`
- `build_symbol_db(root, files)`
- `check_sig(sig, given_pos, given_kw)`
- `class_has_method(db, module, cls, method)`
- `dotted_attr(node)`
- `ensure_in_db(db, module)`
- `extract_sig(fn, is_method)`
- `get_sig(db, module, fq)`
- `import_table(tree, cur_module)`
- `is_py(path)`
- `main()`
- `parse_args()`
- `pretty_sig(sig)`
- `read_file(p)`
- `scan_files(root, exclude_dirs, exclude_glob)`
- `synth_init_sig()`
- `target_kind_from_sig(sig, fq)`
- `to_module(root, file_path)`
- `verify_with_db(root, files, db)`
- `write_symbol_md(root, db, out_md)`

### Classes & Methods
- **Problem**

---
## Collisions (public names)
- Class name collisions: 27
- Function name collisions: 16
