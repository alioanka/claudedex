# XRef Verification Report (Mini-DB)
- Root: `C:\Users\HP\Desktop\ClaudeDex`
- Files scanned: `123`
- Skipped external: `3689`


## Missing Definitions (2)
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:120
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\ml\models\ensemble_model.py`
  - target_kind: `class_or_constructor`
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:84
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\ml\models\ensemble_model.py`
  - target_kind: `class_or_constructor`

## Missing Methods (0)
- None

## Signature Mismatches (17)
- **config.config_manager.ConfigManager: Unknown keywords: encryption_key**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:88
  - called_expr: `ConfigManager`
  - called_args: `{'positional': 0, 'keyword': 2}`
  - declared_signature: `__init__(config_dir) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\config\config_manager.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['encryption_key']`
  - allowed_positionals: `1`
- **security.wallet_security.WalletSecurityManager: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:104
  - called_expr: `WalletSecurityManager`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\wallet_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.audit_logger.AuditLogger: Missing: config; Unknown keywords: log_dir, retention_days**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:114
  - called_expr: `AuditLogger`
  - called_args: `{'positional': 0, 'keyword': 2}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\audit_logger.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `['log_dir', 'retention_days']`
  - allowed_positionals: `1`
- **data.collectors.dexscreener.DexScreenerCollector: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_data_integration.py`:26
  - called_expr: `DexScreenerCollector`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\dexscreener.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **data.processors.aggregator.DataAggregator: Too many positional args: 2>1**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_data_integration.py`:140
  - called_expr: `DataAggregator`
  - called_args: `{'positional': 2, 'keyword': 0}`
  - declared_signature: `__init__(config) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\data\processors\aggregator.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:52
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.pump_predictor.PumpPredictor: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:81
  - called_expr: `PumpPredictor`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\pump_predictor.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:123
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.pump_predictor.PumpPredictor: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:124
  - called_expr: `PumpPredictor`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\pump_predictor.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\performance\test_performance.py`:71
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.encryption.EncryptionManager: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:25
  - called_expr: `EncryptionManager`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\encryption.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.api_security.APISecurityManager: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:49
  - called_expr: `APISecurityManager`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\api_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.api_security.APISecurityManager: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:73
  - called_expr: `APISecurityManager`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\api_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.wallet_security.WalletSecurityManager: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:209
  - called_expr: `WalletSecurityManager`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\wallet_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\smoke\test_smoke.py`:56
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **core.engine.TradingBotEngine: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\smoke\test_smoke.py`:94
  - called_expr: `TradingBotEngine`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config, mode) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `2`
- **core.engine.TradingBotEngine: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\unit\test_engine.py`:22
  - called_expr: `TradingBotEngine`
  - called_args: `{'positional': 0, 'keyword': 0}`
  - declared_signature: `__init__(config, mode) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `2`

## Import Failures (Parse) (0)
- None

## Import Name Shadowing (0)
- None

## Duplicate Classes (in module) (0)
- None

## Duplicate Functions (in module) (0)
- None

## Duplicate Methods (in class) (0)
- None

## Class Name Collisions (across modules) (27)
- **Class name 'Problem' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 76], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 204]]`
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
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 947], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 990]]`
- **Class name 'ColoredFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 970], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1013]]`
- **Class name 'StandardFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 991], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1034]]`
- **Class name 'TradeFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[['monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 1001], ['monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1044]]`
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

## Function Name Collisions (across modules) (16)
- **Function name 'test_web3_connection' appears in multiple modules**  
  File: ``:1
  - definitions: `[['main', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\main.py', 52], ['trading.executors.base_executor', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\executors\\base_executor.py', 1118]]`
- **Function name 'test_api_connection' appears in multiple modules**  
  File: ``:1
  - definitions: `[['main', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\main.py', 57], ['data.collectors.dexscreener', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\collectors\\dexscreener.py', 933]]`
- **Function name 'main' appears in multiple modules**  
  File: ``:1
  - definitions: `[['main', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\main.py', 372], ['setup_env_keys', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\setup_env_keys.py', 32], ['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 778], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 516], ['scripts.health_check', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\health_check.py', 58], ['scripts.run_tests', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\run_tests.py', 201], ['scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 579], ['scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 490], ['scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 461]]`
- **Function name 'is_py' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 89], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 28]]`
- **Function name 'to_module' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 92], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 30]]`
- **Function name 'extract_sig' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 108], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 92]]`
- **Function name 'arg_summary' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 119], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 55]]`
- **Function name 'dotted_attr' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 124], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 46]]`
- **Function name 'read_file' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 133], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 39]]`
- **Function name 'pretty_sig' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 148], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 265]]`
- **Function name 'parse_args' appears in multiple modules**  
  File: ``:1
  - definitions: `[['verify_references', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\verify_references.py', 767], ['xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 505]]`
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
**Total findings:** 62
