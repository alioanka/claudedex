# XRef Verification Report (Mini-DB)
- Root: `C:\Users\HP\Desktop\ClaudeDex`
- Files scanned: `99`
- Skipped external: `3463`


## Missing Definitions (12)
- **Missing definition: monitoring.logger.setup_logger**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:49
  - expr: `setup_logger`
  - target: `monitoring.logger.setup_logger`
- **Missing definition: config.load_config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:71
  - expr: `load_config`
  - target: `config.load_config`
- **Missing definition: security.encryption.SecurityManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:78
  - expr: `SecurityManager`
  - target: `security.encryption.SecurityManager`
- **Missing definition: data.storage.database.test_connection**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:155
  - expr: `test_connection`
  - target: `data.storage.database.test_connection`
- **Missing definition: data.storage.cache.test_redis_connection**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:160
  - expr: `test_redis_connection`
  - target: `data.storage.cache.test_redis_connection`
- **Missing definition: trading.executors.direct_dex.test_web3_connection**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:165
  - expr: `test_web3_connection`
  - target: `trading.executors.direct_dex.test_web3_connection`
- **Missing definition: ml.models.verify_models_loaded**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:175
  - expr: `verify_models_loaded`
  - target: `ml.models.verify_models_loaded`
- **Missing definition: data.storage.database.close_all_connections**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:262
  - expr: `close_all_connections`
  - target: `data.storage.database.close_all_connections`
- **Missing definition: trading.strategies.StrategyManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:113
  - expr: `StrategyManager`
  - target: `trading.strategies.StrategyManager`
- **Missing definition: monitoring.alerts.AlertManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:119
  - expr: `AlertManager`
  - target: `monitoring.alerts.AlertManager`
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:120
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:84
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`

## Missing Methods (0)
- None

## Signature Mismatches (26)
- **verify_references.extract_sig: Missing required positional args: required 2, given 1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\verify_references.py`:184
  - called_expr: `extract_sig`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['is_method']`
- **verify_references.extract_sig: Missing required positional args: required 2, given 1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\verify_references.py`:226
  - called_expr: `extract_sig`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['is_method']`
- **verify_references.extract_sig: Missing required positional args: required 2, given 1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\verify_references.py`:234
  - called_expr: `extract_sig`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['is_method']`
- **xref_symbol_db.extract_sig: Missing required positional args: required 2, given 1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\xref_symbol_db.py`:135
  - called_expr: `extract_sig`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['is_method']`
- **xref_symbol_db.extract_sig: Missing required positional args: required 2, given 1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\xref_symbol_db.py`:151
  - called_expr: `extract_sig`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['is_method']`
- **core.event_bus.Event: Too many positional args: 1>0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:230
  - called_expr: `Event`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['data']`
- **core.event_bus.Event: Too many positional args: 1>0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:404
  - called_expr: `Event`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['data']`
- **core.event_bus.Event: Too many positional args: 1>0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:543
  - called_expr: `Event`
  - args: `{'positional': 1, 'keyword': 1}`
  - keywords: `['data']`
- **config.config_manager.ConfigManager: Unknown keyword(s): encryption_key**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:88
  - called_expr: `ConfigManager`
  - args: `{'positional': 0, 'keyword': 2}`
  - keywords: `['config_dir', 'encryption_key']`
- **security.wallet_security.WalletSecurityManager: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:104
  - called_expr: `WalletSecurityManager`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **security.audit_logger.AuditLogger: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:114
  - called_expr: `AuditLogger`
  - args: `{'positional': 0, 'keyword': 2}`
  - keywords: `['log_dir', 'retention_days']`
- **data.collectors.dexscreener.DexScreenerCollector: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_data_integration.py`:26
  - called_expr: `DexScreenerCollector`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **data.processors.aggregator.DataAggregator: Too many positional args: 2>1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_data_integration.py`:140
  - called_expr: `DataAggregator`
  - args: `{'positional': 2, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.rug_classifier.RugClassifier: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:52
  - called_expr: `RugClassifier`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.pump_predictor.PumpPredictor: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:81
  - called_expr: `PumpPredictor`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.rug_classifier.RugClassifier: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:123
  - called_expr: `RugClassifier`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.pump_predictor.PumpPredictor: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:124
  - called_expr: `PumpPredictor`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.rug_classifier.RugClassifier: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\performance\test_performance.py`:71
  - called_expr: `RugClassifier`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **security.encryption.EncryptionManager: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:25
  - called_expr: `EncryptionManager`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **security.api_security.APISecurityManager: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:49
  - called_expr: `APISecurityManager`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **security.api_security.APISecurityManager: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:73
  - called_expr: `APISecurityManager`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **security.wallet_security.WalletSecurityManager: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:209
  - called_expr: `WalletSecurityManager`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **ml.models.rug_classifier.RugClassifier: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\smoke\test_smoke.py`:56
  - called_expr: `RugClassifier`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **core.engine.TradingBotEngine: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\smoke\test_smoke.py`:94
  - called_expr: `TradingBotEngine`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **core.engine.TradingBotEngine: Missing required positional args: required 1, given 0**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\unit\test_engine.py`:22
  - called_expr: `TradingBotEngine`
  - args: `{'positional': 0, 'keyword': 0}`
  - keywords: `[]`
- **trading.strategies.base_strategy.TradingSignal: Unknown keyword(s): chain, confidence, entry_price, indicators, metadata, signal_type, strategy_name, strength, timeframe, token_address**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:409
  - called_expr: `TradingSignal`
  - args: `{'positional': 0, 'keyword': 10}`
  - keywords: `['chain', 'confidence', 'entry_price', 'indicators', 'metadata', 'signal_type', 'strategy_name', 'strength', 'timeframe', 'token_address']`

## Import Failures (Parse) (0)
- None

## Import Name Shadowing (0)
- None

## Duplicate Classes (in module) (0)
- None

## Duplicate Functions (in module) (0)
- None

## Duplicate Methods (in class) (2)
- **Duplicate method in class: monitoring.logger.StructuredLogger.setup_logging**  
  File: `C:\Users\HP\Desktop\ClaudeDex\monitoring\logger.py`:905
  - lines: `[710, 905]`
- **Duplicate method in class: trading.executors.toxisol_api.ToxiSolAPIExecutor._build_transaction**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\toxisol_api.py`:727
  - lines: `[320, 727]`

## Class Name Collisions (across modules) (27)
- **Class name 'Problem' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 75], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 205]]`
- **Class name 'MarketCondition' appears in multiple modules**  
  File: ``:1
  - definitions: `[['analysis.market_analyzer', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\analysis\\market_analyzer.py', 24], ['utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 381]]`
- **Class name 'ValidationResult' appears in multiple modules**  
  File: ``:1
  - definitions: `[['config.validation', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\config\\validation.py', 20], ['data.processors.validator', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\processors\\validator.py', 38]]`
- **Class name 'RiskScore' appears in multiple modules**  
  File: ``:1
  - definitions: `[['core.decision_maker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\decision_maker.py', 31], ['core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 28]]`
- **Class name 'TradingOpportunity' appears in multiple modules**  
  File: ``:1
  - definitions: `[['core.decision_maker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\decision_maker.py', 44], ['core.engine', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\engine.py', 50], ['core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 137]]`
- **Class name 'Position' appears in multiple modules**  
  File: ``:1
  - definitions: `[['core.portfolio_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\portfolio_manager.py', 16], ['core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 154], ['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 149], ['trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 40]]`
- **Class name 'RiskLevel' appears in multiple modules**  
  File: ``:1
  - definitions: `[['core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 18], ['trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 32]]`
- **Class name 'TokenAnalysis' appears in multiple modules**  
  File: ``:1
  - definitions: `[['data.collectors.token_sniffer', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\collectors\\token_sniffer.py', 49], ['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 357]]`
- **Class name 'PositionStatus' appears in multiple modules**  
  File: ``:1
  - definitions: `[['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 37], ['trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 19]]`
- **Class name 'AlertPriority' appears in multiple modules**  
  File: ``:1
  - definitions: `[['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 45], ['monitoring.alerts', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\alerts.py', 21]]`
- **Class name 'Alert' appears in multiple modules**  
  File: ``:1
  - definitions: `[['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 296], ['monitoring.alerts', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\alerts.py', 79]]`
- **Class name 'PerformanceMetrics' appears in multiple modules**  
  File: ``:1
  - definitions: `[['data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 446], ['trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 169]]`
- **Class name 'MetricType' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 33], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 32]]`
- **Class name 'TradeRecord' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 43], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 42]]`
- **Class name 'PerformanceSnapshot' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 82], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 81]]`
- **Class name 'PerformanceTracker' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 123], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 122]]`
- **Class name 'StructuredLogger' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 686], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 904]]`
- **Class name 'JsonFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 946], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 990]]`
- **Class name 'ColoredFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 969], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1013]]`
- **Class name 'StandardFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 990], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1034]]`
- **Class name 'TradeFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 1000], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1044]]`
- **Class name 'ExpectedItem' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 27], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 16], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 26]]`
- **Class name 'FoundMethod' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 40], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 29], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 39]]`
- **Class name 'FileReport' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 55], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 44], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 54]]`
- **Class name 'OrderStatus' appears in multiple modules**  
  File: ``:1
  - definitions: `[['trading.orders.order_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\order_manager.py', 19], ['utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 410]]`
- **Class name 'OrderType' appears in multiple modules**  
  File: ``:1
  - definitions: `[['trading.orders.order_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\order_manager.py', 29], ['utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 400]]`
- **Class name 'SignalStrength' appears in multiple modules**  
  File: ``:1
  - definitions: `[['trading.strategies.base_strategy', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\strategies\\base_strategy.py', 35], ['utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 338]]`

## Function Name Collisions (across modules) (13)
- **Function name 'main' appears in multiple modules**  
  File: ``:1
  - definitions: `[['main', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\main.py', 314], ['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 730], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 477], ['scripts.health_check', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\health_check.py', 58], ['scripts.run_tests', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\run_tests.py', 201], ['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 579], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 490], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 461]]`
- **Function name 'is_py' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 88], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 29]]`
- **Function name 'to_module' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 91], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 31]]`
- **Function name 'extract_sig' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 107], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 93]]`
- **Function name 'arg_summary' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 118], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 56]]`
- **Function name 'dotted_attr' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 123], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 47]]`
- **Function name 'read_file' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 132], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 40]]`
- **Function name 'parse_args' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 719], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 466]]`
- **Function name 'parse_expectations_from_md' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 118], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 98], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 155]]`
- **Function name 'parse_structure_md' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 228], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 185], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 226]]`
- **Function name 'collect_file_symbols' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 333], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 279], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 324]]`
- **Function name 'generate_reports' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 400], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 346], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 377]]`
- **Function name 'render_markdown' appears in multiple modules**  
  File: ``:1
  - definitions: `[['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 490], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 419]]`

---
**Total findings:** 80
