# Cross-file Reference & Signature Verification Report

- Root: `C:\Users\HP\Desktop\ClaudeDex`
- Python modules indexed: `97`
- Calls analyzed: `13723`
- Skipped (external): `3666`
- Skipped (unknown/duck-typed): `9845`


## Missing Definitions (7)

- **Missing definition: monitoring.logger.setup_logger**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:49
  - expr: `setup_logger`
  - target: `monitoring.logger.setup_logger`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\monitoring\logger.py`
  - target_kind: `method_or_attr`
- **Missing definition: config.load_config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:71
  - expr: `load_config`
  - target: `config.load_config`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\config\__init__.py`
  - target_kind: `function`
- **Missing definition: security.encryption.SecurityManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:78
  - expr: `SecurityManager`
  - target: `security.encryption.SecurityManager`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\security\encryption.py`
  - target_kind: `method_or_attr`
- **Missing definition: trading.strategies.StrategyManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:113
  - expr: `StrategyManager`
  - target: `trading.strategies.StrategyManager`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\__init__.py`
  - target_kind: `method_or_attr`
- **Missing definition: monitoring.alerts.AlertManager**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:119
  - expr: `AlertManager`
  - target: `monitoring.alerts.AlertManager`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\monitoring\alerts.py`
  - target_kind: `method_or_attr`
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:120
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\ml\models\ensemble_model.py`
  - target_kind: `method_or_attr`
- **Missing definition: ml.models.ensemble_model.EnsembleModel**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:84
  - expr: `EnsembleModel`
  - target: `ml.models.ensemble_model.EnsembleModel`
  - target_module_file: `C:\Users\HP\Desktop\ClaudeDex\ml\models\ensemble_model.py`
  - target_kind: `method_or_attr`


## Missing Methods (0)

- None


## Signature Mismatches (18)

- **core.event_bus.Event.__init__: Too many positional args: 1>0; Unknown keywords: data**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:230
  - called_expr: `Event`
  - called_args: `{'positional': 1, 'keywords': ['data']}`
  - declared_signature: `__init__()`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\event_bus.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['data']`
  - allowed_positionals: `0`
- **core.event_bus.Event.__init__: Too many positional args: 1>0; Unknown keywords: data**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:404
  - called_expr: `Event`
  - called_args: `{'positional': 1, 'keywords': ['data']}`
  - declared_signature: `__init__()`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\event_bus.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['data']`
  - allowed_positionals: `0`
- **core.event_bus.Event.__init__: Too many positional args: 1>0; Unknown keywords: data**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:543
  - called_expr: `Event`
  - called_args: `{'positional': 1, 'keywords': ['data']}`
  - declared_signature: `__init__()`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\event_bus.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['data']`
  - allowed_positionals: `0`
- **config.config_manager.ConfigManager.__init__: Unknown keywords: encryption_key**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:88
  - called_expr: `ConfigManager`
  - called_args: `{'positional': 0, 'keywords': ['config_dir', 'encryption_key']}`
  - declared_signature: `__init__(config_dir) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\config\config_manager.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['encryption_key']`
  - allowed_positionals: `1`
- **security.wallet_security.WalletSecurityManager.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:104
  - called_expr: `WalletSecurityManager`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\wallet_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.audit_logger.AuditLogger.__init__: Missing: config; Unknown keywords: log_dir, retention_days**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\conftest.py`:114
  - called_expr: `AuditLogger`
  - called_args: `{'positional': 0, 'keywords': ['log_dir', 'retention_days']}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\audit_logger.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `['log_dir', 'retention_days']`
  - allowed_positionals: `1`
- **data.collectors.dexscreener.DexScreenerCollector.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_data_integration.py`:26
  - called_expr: `DexScreenerCollector`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\dexscreener.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:52
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.pump_predictor.PumpPredictor.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:81
  - called_expr: `PumpPredictor`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\pump_predictor.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:123
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.pump_predictor.PumpPredictor.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:124
  - called_expr: `PumpPredictor`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\pump_predictor.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **ml.models.rug_classifier.RugClassifier.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\performance\test_performance.py`:71
  - called_expr: `RugClassifier`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\ml\models\rug_classifier.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.encryption.EncryptionManager.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:25
  - called_expr: `EncryptionManager`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\encryption.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.api_security.APISecurityManager.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:49
  - called_expr: `APISecurityManager`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\api_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.api_security.APISecurityManager.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:73
  - called_expr: `APISecurityManager`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\api_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **security.wallet_security.WalletSecurityManager.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\security\test_security.py`:209
  - called_expr: `WalletSecurityManager`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config)`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\security\wallet_security.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `1`
- **core.engine.TradingBotEngine.__init__: Missing: config**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\unit\test_engine.py`:22
  - called_expr: `TradingBotEngine`
  - called_args: `{'positional': 0, 'keywords': []}`
  - declared_signature: `__init__(config, mode) [defaults=1]`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`
  - target_kind: `method`
  - missing_params: `['config']`
  - unknown_keywords: `[]`
  - allowed_positionals: `2`
- **trading.strategies.base_strategy.TradingSignal.__init__: Unknown keywords: chain, confidence, entry_price, indicators, metadata, signal_type, strategy_name, strength, timeframe, token_address**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:409
  - called_expr: `TradingSignal`
  - called_args: `{'positional': 0, 'keywords': ['chain', 'confidence', 'entry_price', 'indicators', 'metadata', 'signal_type', 'strategy_name', 'strength', 'timeframe', 'token_address']}`
  - declared_signature: `__init__()`
  - declared_in: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\base_strategy.py`
  - target_kind: `method`
  - missing_params: `[]`
  - unknown_keywords: `['chain', 'confidence', 'entry_price', 'indicators', 'metadata', 'signal_type', 'strategy_name', 'strength', 'timeframe', 'token_address']`
  - allowed_positionals: `0`


## Import Failures (Parse) (0)

- None


## Import Object Missing (39)

- **from monitoring.logger import setup_logger -> setup_logger not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:1
  - imported_as: `setup_logger`
- **from config import load_config -> load_config not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:1
  - imported_as: `load_config`
- **from security.encryption import SecurityManager -> SecurityManager not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\main.py`:1
  - imported_as: `SecurityManager`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\dev_analyzer.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from utils.constants import BLOCK_EXPLORERS -> BLOCK_EXPLORERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\dev_analyzer.py`:1
  - imported_as: `BLOCK_EXPLORERS`
- **from ml.models.ensemble_model import EnsembleModel -> EnsembleModel not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\market_analyzer.py`:1
  - imported_as: `EnsembleModel`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\smart_contract_analyzer.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from utils.constants import BLOCK_EXPLORERS -> BLOCK_EXPLORERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\smart_contract_analyzer.py`:1
  - imported_as: `BLOCK_EXPLORERS`
- **from ml.models.ensemble_model import EnsembleModel -> EnsembleModel not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\analysis\token_scorer.py`:1
  - imported_as: `EnsembleModel`
- **from trading.strategies import StrategyManager -> StrategyManager not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:1
  - imported_as: `StrategyManager`
- **from monitoring.alerts import AlertManager -> AlertManager not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\core\engine.py`:1
  - imported_as: `AlertManager`
- **from utils.constants import HONEYPOT_CHECKS -> HONEYPOT_CHECKS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `HONEYPOT_CHECKS`
- **from utils.constants import HONEYPOT_THRESHOLDS -> HONEYPOT_THRESHOLDS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `HONEYPOT_THRESHOLDS`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from utils.constants import BLACKLISTED_TOKENS -> BLACKLISTED_TOKENS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `BLACKLISTED_TOKENS`
- **from utils.constants import BLACKLISTED_CONTRACTS -> BLACKLISTED_CONTRACTS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `BLACKLISTED_CONTRACTS`
- **from utils.constants import BLACKLISTED_WALLETS -> BLACKLISTED_WALLETS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\honeypot_checker.py`:1
  - imported_as: `BLACKLISTED_WALLETS`
- **from utils.constants import API_RATE_LIMITS -> API_RATE_LIMITS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\social_data.py`:1
  - imported_as: `API_RATE_LIMITS`
- **from utils.constants import CHAIN_NAMES -> CHAIN_NAMES not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\token_sniffer.py`:1
  - imported_as: `CHAIN_NAMES`
- **from utils.constants import HONEYPOT_THRESHOLDS -> HONEYPOT_THRESHOLDS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\token_sniffer.py`:1
  - imported_as: `HONEYPOT_THRESHOLDS`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\volume_analyzer.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from utils.constants import DEX_ROUTERS -> DEX_ROUTERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\volume_analyzer.py`:1
  - imported_as: `DEX_ROUTERS`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\whale_tracker.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from utils.constants import BLOCK_EXPLORERS -> BLOCK_EXPLORERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\whale_tracker.py`:1
  - imported_as: `BLOCK_EXPLORERS`
- **from utils.constants import WRAPPED_NATIVE_TOKENS -> WRAPPED_NATIVE_TOKENS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\whale_tracker.py`:1
  - imported_as: `WRAPPED_NATIVE_TOKENS`
- **from utils.constants import STABLECOINS -> STABLECOINS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\data\collectors\whale_tracker.py`:1
  - imported_as: `STABLECOINS`
- **from ml.models.ensemble_model import EnsembleModel -> EnsembleModel not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\tests\integration\test_ml_integration.py`:1
  - imported_as: `EnsembleModel`
- **from trading.executors.base_executor import BaseExecutor -> BaseExecutor not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\direct_dex.py`:1
  - imported_as: `BaseExecutor`
- **from utils.constants import DEX_ROUTERS -> DEX_ROUTERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\direct_dex.py`:1
  - imported_as: `DEX_ROUTERS`
- **from utils.constants import CHAIN_RPC_URLS -> CHAIN_RPC_URLS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\direct_dex.py`:1
  - imported_as: `CHAIN_RPC_URLS`
- **from trading.executors.base_executor import BaseExecutor -> BaseExecutor not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\mev_protection.py`:1
  - imported_as: `BaseExecutor`
- **from trading.executors.base_executor import BaseExecutor -> BaseExecutor not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\executors\toxisol_api.py`:1
  - imported_as: `BaseExecutor`
- **from ml.models.ensemble_model import EnsembleModel -> EnsembleModel not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\ai_strategy.py`:1
  - imported_as: `EnsembleModel`
- **from utils.constants import MAX_POSITION_SIZE_PERCENT -> MAX_POSITION_SIZE_PERCENT not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\base_strategy.py`:1
  - imported_as: `MAX_POSITION_SIZE_PERCENT`
- **from utils.constants import DEFAULT_STOP_LOSS -> DEFAULT_STOP_LOSS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\base_strategy.py`:1
  - imported_as: `DEFAULT_STOP_LOSS`
- **from utils.constants import RISK_PARAMETERS -> RISK_PARAMETERS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\base_strategy.py`:1
  - imported_as: `RISK_PARAMETERS`
- **from utils.constants import DEFAULT_SLIPPAGE -> DEFAULT_SLIPPAGE not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\scalping.py`:1
  - imported_as: `DEFAULT_SLIPPAGE`
- **from utils.constants import MAX_SLIPPAGE -> MAX_SLIPPAGE not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\scalping.py`:1
  - imported_as: `MAX_SLIPPAGE`
- **from utils.constants import SIGNAL_THRESHOLDS -> SIGNAL_THRESHOLDS not found in project module**  
  File: `C:\Users\HP\Desktop\ClaudeDex\trading\strategies\scalping.py`:1
  - imported_as: `SIGNAL_THRESHOLDS`


## Import Name Shadowing (0)

- None


## Duplicate Classes (in module) (0)

- None


## Duplicate Functions (in module) (0)

- None


## Duplicate Methods (in class) (0)

- None


## Class Name Collisions (across modules) (26)

- **Class name 'MarketCondition' appears in multiple modules**  
  File: ``:1
  - definitions: `[('analysis.market_analyzer', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\analysis\\market_analyzer.py', 24), ('utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 381)]`
- **Class name 'ValidationResult' appears in multiple modules**  
  File: ``:1
  - definitions: `[('config.validation', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\config\\validation.py', 20), ('data.processors.validator', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\processors\\validator.py', 38)]`
- **Class name 'RiskScore' appears in multiple modules**  
  File: ``:1
  - definitions: `[('core.decision_maker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\decision_maker.py', 31), ('core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 28)]`
- **Class name 'TradingOpportunity' appears in multiple modules**  
  File: ``:1
  - definitions: `[('core.decision_maker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\decision_maker.py', 44), ('core.engine', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\engine.py', 50), ('core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 137)]`
- **Class name 'Position' appears in multiple modules**  
  File: ``:1
  - definitions: `[('core.portfolio_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\portfolio_manager.py', 16), ('core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 154), ('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 149), ('trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 40)]`
- **Class name 'RiskLevel' appears in multiple modules**  
  File: ``:1
  - definitions: `[('core.risk_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\core\\risk_manager.py', 18), ('trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 32)]`
- **Class name 'TokenAnalysis' appears in multiple modules**  
  File: ``:1
  - definitions: `[('data.collectors.token_sniffer', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\collectors\\token_sniffer.py', 49), ('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 357)]`
- **Class name 'PositionStatus' appears in multiple modules**  
  File: ``:1
  - definitions: `[('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 37), ('trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 19)]`
- **Class name 'AlertPriority' appears in multiple modules**  
  File: ``:1
  - definitions: `[('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 45), ('monitoring.alerts', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\alerts.py', 21)]`
- **Class name 'Alert' appears in multiple modules**  
  File: ``:1
  - definitions: `[('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 296), ('monitoring.alerts', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\alerts.py', 79)]`
- **Class name 'PerformanceMetrics' appears in multiple modules**  
  File: ``:1
  - definitions: `[('data.storage.models', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\data\\storage\\models.py', 446), ('trading.orders.position_tracker', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\position_tracker.py', 169)]`
- **Class name 'MetricType' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 33), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 32)]`
- **Class name 'TradeRecord' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 43), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 42)]`
- **Class name 'PerformanceSnapshot' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 82), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 81)]`
- **Class name 'PerformanceTracker' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 123), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 122)]`
- **Class name 'StructuredLogger' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 686), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 904)]`
- **Class name 'JsonFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 947), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 990)]`
- **Class name 'ColoredFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 970), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1013)]`
- **Class name 'StandardFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 991), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1034)]`
- **Class name 'TradeFormatter' appears in multiple modules**  
  File: ``:1
  - definitions: `[('monitoring.logger', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\logger.py', 1001), ('monitoring.performance', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\monitoring\\performance.py', 1044)]`
- **Class name 'ExpectedItem' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 27), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 16), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 26)]`
- **Class name 'FoundMethod' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 40), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 29), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 39)]`
- **Class name 'FileReport' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 55), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 44), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 54)]`
- **Class name 'OrderStatus' appears in multiple modules**  
  File: ``:1
  - definitions: `[('trading.orders.order_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\order_manager.py', 19), ('utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 410)]`
- **Class name 'OrderType' appears in multiple modules**  
  File: ``:1
  - definitions: `[('trading.orders.order_manager', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\orders\\order_manager.py', 29), ('utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 400)]`
- **Class name 'SignalStrength' appears in multiple modules**  
  File: ``:1
  - definitions: `[('trading.strategies.base_strategy', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\trading\\strategies\\base_strategy.py', 35), ('utils.constants', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\utils\\constants.py', 338)]`


## Function Name Collisions (across modules) (6)

- **Function name 'main' appears in multiple modules**  
  File: ``:1
  - definitions: `[('main', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\main.py', 314), ('xref_symbol_db', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\xref_symbol_db.py', 516), ('scripts.health_check', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\health_check.py', 58), ('scripts.run_tests', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\run_tests.py', 201), ('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 579), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 490), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 461)]`
- **Function name 'parse_expectations_from_md' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 118), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 98), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 155)]`
- **Function name 'parse_structure_md' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 228), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 185), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 226)]`
- **Function name 'collect_file_symbols' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 333), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 279), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 324)]`
- **Function name 'generate_reports' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 400), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 346), ('scripts.verify_claudedex_plus3', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus3.py', 377)]`
- **Function name 'render_markdown' appears in multiple modules**  
  File: ``:1
  - definitions: `[('scripts.verify_claudedex_plus', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus.py', 490), ('scripts.verify_claudedex_plus2', 'C:\\Users\\HP\\Desktop\\ClaudeDex\\scripts\\verify_claudedex_plus2.py', 419)]`


## Unresolved Config References (0)

- None


---
**Total findings:** 96

> Notes:
> - External/stdlib/3rd-party calls are skipped.
> - Duck-typed/unknown receivers are skipped to avoid noise.
> - Collisions are informational.
