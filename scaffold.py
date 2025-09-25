import os

# All directories (relative to ClaudeDex)
dirs = [
    "core",
    "data/collectors",
    "data/processors",
    "data/storage/migrations",
    "analysis",
    "ml/models",
    "ml/training",
    "ml/optimization",
    "trading/executors",
    "trading/strategies",
    "trading/orders",
    "monitoring",
    "security",
    "config/blacklists",
    "config/filters",
    "config/strategies",
    "utils",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "scripts"
]

# All files (relative to ClaudeDex)
files = [
    # top-level
    "requirements.txt", "docker-compose.yml", "Dockerfile",
    ".env.example", "README.md", "setup.py", "main.py",

    # core
    "core/__init__.py", "core/engine.py", "core/risk_manager.py",
    "core/pattern_analyzer.py", "core/decision_maker.py",
    "core/portfolio_manager.py", "core/event_bus.py",

    # data
    "data/__init__.py",
    "data/collectors/__init__.py", "data/collectors/dexscreener.py",
    "data/collectors/chain_data.py", "data/collectors/social_data.py",
    "data/collectors/volume_analyzer.py", "data/collectors/mempool_monitor.py",
    "data/collectors/whale_tracker.py", "data/collectors/honeypot_checker.py",

    "data/processors/__init__.py", "data/processors/normalizer.py",
    "data/processors/feature_extractor.py", "data/processors/aggregator.py",
    "data/processors/validator.py",

    "data/storage/__init__.py", "data/storage/database.py",
    "data/storage/cache.py", "data/storage/models.py",

    # analysis
    "analysis/__init__.py", "analysis/rug_detector.py",
    "analysis/pump_predictor.py", "analysis/dev_analyzer.py",
    "analysis/liquidity_monitor.py", "analysis/smart_contract_analyzer.py",
    "analysis/token_scorer.py", "analysis/market_analyzer.py",

    # ml
    "ml/__init__.py",
    "ml/models/__init__.py", "ml/models/rug_classifier.py",
    "ml/models/pump_predictor.py", "ml/models/volume_validator.py",
    "ml/models/sentiment_analyzer.py", "ml/models/ensemble_model.py",

    "ml/training/__init__.py", "ml/training/trainer.py",
    "ml/training/evaluator.py", "ml/training/backtester.py",
    "ml/training/dataset_builder.py",

    "ml/optimization/__init__.py", "ml/optimization/hyperparameter.py",
    "ml/optimization/genetic_optimizer.py", "ml/optimization/reinforcement.py",

    # trading
    "trading/__init__.py",
    "trading/executors/__init__.py", "trading/executors/base_executor.py",
    "trading/executors/toxisol_api.py", "trading/executors/direct_dex.py",
    "trading/executors/mev_protection.py", "trading/executors/sniper.py",
    "trading/executors/arbitrage.py",

    "trading/strategies/__init__.py", "trading/strategies/base_strategy.py",
    "trading/strategies/scalping.py", "trading/strategies/momentum.py",
    "trading/strategies/mean_reversion.py", "trading/strategies/grid_trading.py",
    "trading/strategies/ai_strategy.py",

    "trading/orders/__init__.py", "trading/orders/order_manager.py",
    "trading/orders/position_tracker.py", "trading/orders/pnl_calculator.py",

    # monitoring
    "monitoring/__init__.py", "monitoring/alerts.py",
    "monitoring/dashboard.py", "monitoring/performance.py",
    "monitoring/health_checker.py", "monitoring/logger.py",

    # security
    "security/__init__.py", "security/encryption.py",
    "security/api_manager.py", "security/wallet_manager.py",
    "security/audit_logger.py",

    # config
    "config/__init__.py", "config/settings.yaml",
    "config/blacklists/tokens.json", "config/blacklists/developers.json",
    "config/blacklists/contracts.json", "config/blacklists/updater.py",
    "config/filters/entry_rules.yaml", "config/filters/exit_rules.yaml",
    "config/filters/risk_rules.yaml",
    "config/strategies/strategy_configs.yaml",

    # utils
    "utils/__init__.py", "utils/constants.py", "utils/helpers.py",
    "utils/validators.py", "utils/formatters.py", "utils/math_utils.py",

    # tests
    "tests/__init__.py",

    # scripts
    "scripts/setup_database.py", "scripts/train_models.py",
    "scripts/backtest.py", "scripts/deploy.sh",
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_file(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith("__init__.py"):
                f.write("# Package marker\n")

# Create all directories
for d in dirs:
    ensure_dir(d)

# Create all files
for f in files:
    ensure_dir(os.path.dirname(f) or ".")
    ensure_file(f)

print("✅ Project structure created under", os.getcwd())
