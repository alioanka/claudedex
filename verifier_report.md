### setup.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### main.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/__init__.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/engine.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/risk_manager.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/pattern_analyzer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/decision_maker.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/portfolio_manager.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/event_bus.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/dexscreener.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/chain_data.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/honeypot_checker.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/whale_tracker.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/mempool_monitor.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/social_data.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/volume_analyzer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/collectors/token_sniffer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/processors/normalizer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/processors/feature_extractor.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/processors/aggregator.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/processors/validator.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/storage/database.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### data/storage/cache.py
- Missing: []
- Sig mismatches: ["data\\storage\\cache.py:176 CacheManager.set params mismatch expected=['key', 'value', 'ttl'] found=['key', 'value', 'ttl', 'cache_type']"]
- Duplicates: []

### data/storage/models.py
- Missing: ['class Performance (closest: PerformanceMetrics)']
- Sig mismatches: []
- Duplicates: []

### analysis/rug_detector.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/pump_predictor.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/liquidity_monitor.py
- Missing: []
- Sig mismatches: ["analysis\\liquidity_monitor.py:518 LiquidityMonitor.get_liquidity_depth params mismatch expected=['pair_address'] found=['pair_address', 'chain']", "analysis\\liquidity_monitor.py:228 LiquidityMonitor.calculate_slippage params mismatch expected=['amount', 'liquidity_data'] found=['token', 'chain', 'amount', 'is_buy']", "analysis\\liquidity_monitor.py:568 LiquidityMonitor.track_liquidity_changes params mismatch expected=['token'] found=['token', 'chain']"]
- Duplicates: []

### analysis/market_analyzer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/token_scorer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/dev_analyzer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/smart_contract_analyzer.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### ml/models/ensemble_model.py
- Missing: []
- Sig mismatches: ["ml\\models\\ensemble_model.py:503 EnsemblePredictor.predict params mismatch expected=['token', 'chain'] found=['features']"]
- Duplicates: []

### ml/models/rug_classifier.py
- Missing: []
- Sig mismatches: ["ml\\models\\rug_classifier.py:245 RugClassifier.train params mismatch expected=['historical_data', 'labels'] found=['historical_data', 'labels', 'validation_split']"]
- Duplicates: []

### ml/models/pump_predictor.py
- Missing: []
- Sig mismatches: ["ml\\models\\pump_predictor.py:415 PumpPredictor.predict_pump_probability params mismatch expected=['features'] found=['current_data']"]
- Duplicates: []

### ml/models/volume_validator.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/executors/base_executor.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/executors/toxisol_api.py
- Missing: ['def cancel_order()', 'def get_order_status()', 'def modify_order()', 'def validate_order()']
- Sig mismatches: []
- Duplicates: []

### trading/executors/direct_dex.py
- Missing: ['def cancel_order()', 'def get_order_status()', 'def modify_order()', 'def validate_order()']
- Sig mismatches: []
- Duplicates: []

### trading/executors/mev_protection.py
- Missing: ['def cancel_order()', 'def execute_trade()', 'def get_order_status()', 'def modify_order()', 'def validate_order()']
- Sig mismatches: []
- Duplicates: []

### trading/strategies/base_strategy.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/strategies/momentum.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/strategies/scalping.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/strategies/ai_strategy.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### trading/orders/order_manager.py
- Missing: ['def execute_order() (closest: execute_sniper, create_order, _execute_sniper)']
- Sig mismatches: ["trading\\orders\\order_manager.py:195 OrderManager.create_order params mismatch expected=['order'] found=['token_address', 'side', 'amount', 'order_type', 'price', 'execution_strategy']"]
- Duplicates: []

### trading/orders/position_tracker.py
- Missing: []
- Sig mismatches: ["trading\\orders\\position_tracker.py:233 PositionTracker.open_position params mismatch expected=['position'] found=['token_address', 'token_symbol', 'position_type', 'entry_price', 'entry_amount', 'order_ids']", "trading\\orders\\position_tracker.py:404 PositionTracker.close_position params mismatch expected=['position_id'] found=['position_id', 'exit_price', 'exit_amount', 'order_ids', 'reason']", "trading\\orders\\position_tracker.py:333 PositionTracker.update_position params mismatch expected=['position_id', 'updates'] found=['position_id', 'current_price']"]
- Duplicates: []

### monitoring/alerts.py
- Missing: ['def send_telegram() (closest: _send_telegram, send_alert)', 'def send_discord() (closest: _send_discord, _send_pushover, _send_slack)', 'def send_email() (closest: _send_email, send_alert, _send_sms)', 'def format_alert() (closest: _format_drawdown_alert, _format_signal, _format_aggregated_alerts)']
- Sig mismatches: ["monitoring\\alerts.py:269 AlertsSystem.send_alert params mismatch expected=['alert_type', 'message', 'data'] found=['alert_type', 'title', 'message', 'priority', 'data', 'channels']"]
- Duplicates: []

### monitoring/dashboard.py
- Missing: ['def start_dashboard() (closest: update_dashboard_data, _update_dashboard_data, _serialize_dashboard_data)', 'def update_metrics() (closest: update_risk, update_orders, _update_risk_data)', 'def get_dashboard_data() (closest: update_dashboard_data, _update_dashboard_data, _serialize_dashboard_data)', 'def generate_charts() (closest: create_win_rate_chart)']
- Sig mismatches: []
- Duplicates: []

### monitoring/performance.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### monitoring/logger.py
- Missing: []
- Sig mismatches: ["monitoring\\logger.py:710 StructuredLogger.setup_logging params mismatch expected=['config'] found=[]"]
- Duplicates: ['class:StructuredLogger::setup_logging x2']

### security/__init__.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### security/encryption.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### security/api_security.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### security/wallet_security.py
- Missing: []
- Sig mismatches: ["security\\wallet_security.py:210 WalletSecurityManager.create_wallet params mismatch expected=['wallet_type', 'security_level'] found=['wallet_type', 'security_level', 'derivation_path']"]
- Duplicates: []

### security/audit_logger.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### config/__init__.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### config/config_manager.py
- Missing: ['def load_config() (closest: _load_config, _reload_config, _load_all_configs)', 'def reload_config() (closest: _reload_config, _load_config, _load_all_configs)']
- Sig mismatches: ["config\\config_manager.py:757 ConfigManager.validate_config params mismatch expected=['config'] found=['config_type', 'config_data']", "config\\config_manager.py:557 ConfigManager.update_config params mismatch expected=['key', 'value'] found=['config_type', 'updates', 'user', 'reason', 'persist']", "config\\config_manager.py:525 ConfigManager.get_config params mismatch expected=['key'] found=['config_type']"]
- Duplicates: []

### config/settings.py
- Missing: []
- Sig mismatches: ["config\\settings.py:419 Settings.get_chain_config params mismatch expected=['chain'] found=['chain_name']"]
- Duplicates: []

### config/validation.py
- Missing: ['def validate_trading_config() (closest: _validate_trading_config, _validate_monitoring_config, _validate_api_config)', 'def validate_security_config() (closest: _validate_security_config, _validate_cross_config, validate_yaml_config)']
- Sig mismatches: []
- Duplicates: []

### utils/__init__.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### utils/helpers.py
- Missing: []
- Sig mismatches: []
- Duplicates: ['func:decorator::wrapper x2']

### utils/constants.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/conftest.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/unit/test_engine.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/unit/test_risk_manager.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/integration/test_data_integration.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/integration/test_ml_integration.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/integration/test_trading_integration.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/performance/test_performance.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/security/test_security.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/smoke/test_smoke.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/fixtures/mock_data.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### tests/fixtures/test_helpers.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### scripts/setup_database.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### scripts/init_config.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### scripts/health_check.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### scripts/run_tests.py
- Missing: []
- Sig mismatches: []
- Duplicates: []
