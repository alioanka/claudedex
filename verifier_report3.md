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
- Sig mismatches: ["core\\risk_manager.py:630 RiskManager.calculate_position_size params mismatch expected=['opportunity'] found=['risk_score', 'available_balance', 'ml_confidence', 'expected_return']"]
- Duplicates: []

### core/pattern_analyzer.py
- Missing: []
- Sig mismatches: ["core\\pattern_analyzer.py:100 PatternAnalyzer.analyze_patterns params mismatch expected=['price_data'] found=['data']"]
- Duplicates: []

### core/decision_maker.py
- Missing: ['def make_decision()', 'def evaluate_opportunity()', 'def calculate_confidence_score()', 'def determine_action()', 'def validate_decision()', "[ParseError] AST parse error in C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\decision_maker.py: expected an indented block after 'except' statement on line 823 (<unknown>, line 827)"]
- Sig mismatches: []
- Duplicates: []

### core/portfolio_manager.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### core/event_bus.py
- Missing: []
- Sig mismatches: ["core\\event_bus.py:195 EventBus.subscribe params mismatch expected=['event_type', 'handler'] found=['event_type', 'callback', 'subscriber_id', 'filters', 'priority']", "core\\event_bus.py:234 EventBus.unsubscribe params mismatch expected=['event_type', 'handler'] found=['subscriber_id', 'event_type']"]
- Duplicates: ['class:EventBus::subscribe x2', 'class:EventBus::unsubscribe x2']

### data/collectors/dexscreener.py
- Missing: []
- Sig mismatches: ["data\\collectors\\dexscreener.py:165 DexScreenerCollector.get_new_pairs params mismatch expected=['chain'] found=['limit']", "data\\collectors\\dexscreener.py:595 DexScreenerCollector.get_price_history params mismatch expected=['address', 'chain', 'interval'] found=['pair_address', 'interval', 'limit']", "data\\collectors\\dexscreener.py:371 DexScreenerCollector.monitor_pair params mismatch expected=['address', 'chain'] found=['pair_address', 'callback']"]
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
- Sig mismatches: ["data\\collectors\\mempool_monitor.py:512 MempoolMonitor.get_mempool_stats params mismatch expected=['chain'] found=[]"]
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
- Sig mismatches: ["data\\storage\\database.py:433 DatabaseManager.get_historical_data params mismatch expected=['token', 'timeframe'] found=['token_address', 'timeframe', 'limit', 'chain']"]
- Duplicates: []

### data/storage/cache.py
- Missing: ['def clear()']
- Sig mismatches: ["data\\storage\\cache.py:98 CacheManager.get params mismatch expected=['key'] found=['key', 'default', 'decode_json']", "data\\storage\\cache.py:125 CacheManager.set params mismatch expected=['key', 'value', 'ttl'] found=['key', 'value', 'ttl', 'cache_type']"]
- Duplicates: []

### data/storage/models.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### analysis/rug_detector.py
- Missing: ['def check_liquidity_removal_risk()']
- Sig mismatches: []
- Duplicates: []

### analysis/pump_predictor.py
- Missing: ['def detect_accumulation_phase()']
- Sig mismatches: []
- Duplicates: []

### analysis/liquidity_monitor.py
- Missing: ['def monitor_liquidity()', 'def get_liquidity_depth()', 'def calculate_slippage()', 'def track_liquidity_changes()', '[ParseError] AST parse error in C:\\Users\\HP\\Desktop\\ClaudeDex\\analysis\\liquidity_monitor.py: unexpected indent (<unknown>, line 498)']
- Sig mismatches: []
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
- Missing: ['def prepare_features()', 'def train_lstm()', 'def predict_pump_probability()', 'def backtest()', '[ParseError] AST parse error in C:\\Users\\HP\\Desktop\\ClaudeDex\\ml\\models\\pump_predictor.py: unexpected indent (<unknown>, line 432)']
- Sig mismatches: []
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
- Missing: ['def initialize()', 'def get_best_quote()', 'def execute_trade()', '[ParseError] AST parse error in C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\executors\\direct_dex.py: unexpected indent (<unknown>, line 515)']
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
- Missing: ['def execute_order()']
- Sig mismatches: ["trading\\orders\\order_manager.py:195 OrderManager.create_order params mismatch expected=['order'] found=['token_address', 'side', 'amount', 'order_type', 'price', 'execution_strategy']"]
- Duplicates: []

### trading/orders/position_tracker.py
- Missing: []
- Sig mismatches: ["trading\\orders\\position_tracker.py:233 PositionTracker.open_position params mismatch expected=['position'] found=['token_address', 'token_symbol', 'position_type', 'entry_price', 'entry_amount', 'order_ids']", "trading\\orders\\position_tracker.py:404 PositionTracker.close_position params mismatch expected=['position_id'] found=['position_id', 'exit_price', 'exit_amount', 'order_ids', 'reason']", "trading\\orders\\position_tracker.py:333 PositionTracker.update_position params mismatch expected=['position_id', 'updates'] found=['position_id', 'current_price']"]
- Duplicates: []

### monitoring/alerts.py
- Missing: ['def send_telegram()', 'def send_discord()', 'def send_email()', 'def format_alert()']
- Sig mismatches: ["monitoring\\alerts.py:269 AlertsSystem.send_alert params mismatch expected=['alert_type', 'message', 'data'] found=['alert_type', 'title', 'message', 'priority', 'data', 'channels']"]
- Duplicates: []

### monitoring/dashboard.py
- Missing: ['def start_dashboard()', 'def update_metrics()', 'def get_dashboard_data()', 'def generate_charts()']
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
- Missing: ['def encrypt_data()', 'def decrypt_data()', 'def generate_key()', 'def hash_password()', 'def verify_password()']
- Sig mismatches: []
- Duplicates: []

### security/api_security.py
- Missing: ['def validate_api_key()', 'def rate_limit_check()', 'def generate_jwt()', 'def verify_jwt()']
- Sig mismatches: []
- Duplicates: []

### security/wallet_security.py
- Missing: ['def encrypt_private_key()']
- Sig mismatches: ["security\\wallet_security.py:210 WalletSecurityManager.create_wallet params mismatch expected=['wallet_type', 'security_level'] found=['wallet_type', 'security_level', 'derivation_path']"]
- Duplicates: []

### security/audit_logger.py
- Missing: ['def log_security_event()', 'def log_access()', 'def log_transaction()', 'def get_audit_trail()']
- Sig mismatches: []
- Duplicates: []

### config/__init__.py
- Missing: []
- Sig mismatches: []
- Duplicates: []

### config/config_manager.py
- Missing: ['def load_config()', 'def reload_config()']
- Sig mismatches: ["config\\config_manager.py:757 ConfigManager.validate_config params mismatch expected=['config'] found=['config_type', 'config_data']", "config\\config_manager.py:557 ConfigManager.update_config params mismatch expected=['key', 'value'] found=['config_type', 'updates', 'user', 'reason', 'persist']", "config\\config_manager.py:525 ConfigManager.get_config params mismatch expected=['key'] found=['config_type']"]
- Duplicates: []

### config/settings.py
- Missing: ['def get_database_url()', 'def get_redis_url()']
- Sig mismatches: ["config\\settings.py:419 Settings.get_chain_config params mismatch expected=['chain'] found=['chain_name']"]
- Duplicates: []

### config/validation.py
- Missing: ['def validate_trading_config()', 'def validate_security_config()', 'def validate_api_keys()', 'def check_required_fields()', '[ParseError] AST parse error in C:\\Users\\HP\\Desktop\\ClaudeDex\\config\\validation.py: unterminated string literal (detected at line 368) (<unknown>, line 368)']
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
