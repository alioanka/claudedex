#!/usr/bin/env python3
"""
AI Analysis Module - Entry Point
"""
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load env
load_dotenv()

# Setup Logging
log_dir = Path("logs/ai_analysis")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger
logger = logging.getLogger("AIModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'ai.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'ai_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.ai_analysis.core.sentiment_engine import SentimentEngine, AITradeExecutor
from config.config_manager import ConfigManager

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller
except ImportError:
    get_telegram_controller = None

# Also configure logging for the engine classes
for engine_name in ["SentimentEngine", "AITradeExecutor"]:
    engine_logger = logging.getLogger(engine_name)
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(main_handler)
    engine_logger.addHandler(error_handler)
    engine_logger.addHandler(console)

# Also add OpenAI API log
openai_log_handler = RotatingFileHandler(log_dir / 'openai_api.log', maxBytes=10*1024*1024, backupCount=5)
openai_log_handler.setFormatter(log_formatter)
openai_logger = logging.getLogger("OpenAI_API")
openai_logger.setLevel(logging.INFO)
openai_logger.addHandler(openai_log_handler)
openai_logger.addHandler(console)

# Also add Claude API log
claude_log_handler = RotatingFileHandler(log_dir / 'claude_api.log', maxBytes=10*1024*1024, backupCount=5)
claude_log_handler.setFormatter(log_formatter)
claude_logger = logging.getLogger("Claude_API")
claude_logger.setLevel(logging.INFO)
claude_logger.addHandler(claude_log_handler)
claude_logger.addHandler(console)

def _build_risk_config(config_manager) -> dict:
    """Build a POPULATED core.risk_manager.RiskManager config from the
    operator's DB-backed ConfigManager (live-safety item #5: the AI live
    risk gate previously ran on an EMPTY config, i.e. code defaults, while
    DEX feeds nested ConfigManager model dumps and Solana passes the
    config_manager through).

    Two layers, mirroring what RiskManager.__init__ / check_circuit_breakers
    / _get_current_metrics actually read:
      1) Nested model dumps keyed by config_type value — RiskManager reads
         config['portfolio']['initial_balance'] for the drawdown/daily-loss
         circuit-breaker baselines. Only risk-relevant, non-sensitive types
         are included (NOT 'security'/'database'/'api' — those dumps carry
         private keys and the AI gate never reads them).
      2) Flat aliases — RiskManager reads flat keys ('max_position_size_pct',
         'breaker.error_rate_max', ...) that don't match the Pydantic field
         names ('max_position_size_percent', 'breaker_error_rate_max', ...),
         so the operator's DB values are mapped onto the exact keys the
         risk gate consumes. Units match: both sides are percent ints.

    Fail-soft by design: any error at any layer skips that value so the
    RiskManager code default applies; raising is left to the caller's
    try/except which falls back to an empty config. Never blocks startup.
    """
    from config.config_manager import ConfigType

    def _dump(model):
        # Pydantic v1/v2 compatible model -> dict (same helper as main_dex).
        if hasattr(model, 'model_dump'):
            return model.model_dump()
        return model.dict()

    cfg: dict = {}
    # Risk-relevant config types only (see docstring for the exclusions).
    wanted = {'risk_management', 'portfolio', 'trading', 'position_management'}
    for config_type in ConfigType:
        if config_type.value not in wanted:
            continue
        try:
            model = config_manager.get_config(config_type)
            if model is not None:
                cfg[config_type.value] = _dump(model)
        except Exception:
            continue  # fail-soft: missing section -> RiskManager defaults

    risk = cfg.get('risk_management') or {}
    portfolio = cfg.get('portfolio') or {}

    # (RiskManager key) <- (source section, ConfigManager field). All are
    # positive numeric thresholds; non-positive/missing values are skipped
    # so the RiskManager code default survives a fat-finger DB row.
    flat_aliases = {
        'max_position_size_pct': (risk, 'max_position_size_percent'),
        'max_position_size_usd': (portfolio, 'max_position_size_usd'),
        'max_positions': (portfolio, 'max_positions'),
        'breaker.error_rate_max': (risk, 'breaker_error_rate_max'),
        'breaker.slippage_realized_bps_max': (risk, 'breaker_slippage_realized_bps_max'),
        'breaker.max_consecutive_losses': (risk, 'breaker_max_consecutive_losses'),
        'breaker.max_drawdown_pct': (risk, 'breaker_max_drawdown_pct'),
        'breaker.max_daily_loss_pct': (risk, 'breaker_max_daily_loss_pct'),
    }
    for dest, (src, key) in flat_aliases.items():
        try:
            val = src.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                cfg[dest] = val
        except Exception:
            continue
    return cfg


async def main():
    logger.info("🧠 AI Analysis Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")
    # Per-module DRY_RUN override (Phase 3 A8). AI_DRY_RUN env beats
    # DRY_RUN. Mirror into DRY_RUN so sentiment_engine's
    # should_skip_live() picks up the resolved value before the LLM
    # signal pipeline starts.
    try:
        from core.dry_run import resolve_module_dry_run
        ai_dry = resolve_module_dry_run('ai', default=True)
        os.environ['DRY_RUN'] = 'true' if ai_dry else 'false'
        logger.info(f"   DRY_RUN (resolved per-module): {ai_dry}")
    except Exception as e:
        logger.warning(f"   Could not resolve per-module DRY_RUN: {e}")

    # Wave-6 fix: API keys live in the encrypted `secure_credentials` DB
    # table, NOT in .env. The previous code read `secrets.get(...)` BEFORE
    # the db_pool was connected — secrets_manager was still in bootstrap
    # mode, so it silently fell back to os.getenv which always returned
    # None. Subprocess then started with no keys, sentiment_engine logged
    # "No AI API keys" and the run-loop produced zero signals for ~4 months.
    #
    # Resolution order (matches copy_trading / dex pattern):
    #   1) secrets manager (Docker secret -> DB -> env, post-db-pool init)
    #   2) os.getenv fallback (legacy .env operators)

    # Init DB FIRST so the secrets manager can decrypt DB-stored keys.
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')

    if not db_url:
        logger.error("No database credentials found (Docker secrets or DATABASE_URL)")
        return

    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    # Now initialize secrets manager WITH the db_pool so the DB-backed
    # encrypted credentials path is available (otherwise it stays in
    # bootstrap mode and skips the DB lookup entirely).
    openai_key = None
    anthropic_key = None
    try:
        from security.secrets_manager import secrets as _secrets
        _secrets.initialize(db_pool)
        # Use get_async so the DB path is taken (sync get() short-circuits
        # when called from inside a running event loop — see
        # _get_from_database_sync's "use get_async() in async context"
        # debug log).
        openai_key = await _secrets.get_async('OPENAI_API_KEY', log_access=False)
        anthropic_key = await _secrets.get_async('ANTHROPIC_API_KEY', log_access=False)
    except Exception as e:
        logger.warning(f"   secrets_manager lookup failed: {e}; falling back to env")

    # .env fallback (gradual-migration support)
    if not openai_key:
        openai_key = os.getenv('OPENAI_API_KEY')
    if not anthropic_key:
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if openai_key:
        logger.info(f"   OpenAI API Key: {openai_key[:20]}... (len={len(openai_key)})")
    else:
        logger.warning("⚠️ No OPENAI_API_KEY found in secrets_manager or env")

    if anthropic_key:
        logger.info(f"   Anthropic API Key: {anthropic_key[:20]}... (len={len(anthropic_key)})")
    else:
        logger.warning("⚠️ No ANTHROPIC_API_KEY found in secrets_manager or env")

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Engine
    config = {
        'openai_api_key': openai_key,
        'anthropic_api_key': anthropic_key
    }
    # P2#5: construct core.risk_manager and thread through to executor.
    # Live-safety #5: the gate used to be built with config={} so LIVE
    # validate_trade ran on code defaults instead of the operator's DB risk
    # profile. Build a populated config from the already-initialized
    # ConfigManager (DEX nested-dump pattern + the flat aliases RiskManager
    # actually reads). Fail-soft at every layer:
    #   DB/config read fails -> empty config (= previous code defaults);
    #   populated construction fails -> retry with empty config;
    #   both fail -> risk_manager stays None and the executor fail-closes
    #   LIVE entries (see AITradeExecutor's risk_manager-is-None guard).
    risk_manager = None
    try:
        from core.risk_manager import RiskManager
        risk_config = {}
        try:
            risk_config = _build_risk_config(config_manager)
            logger.info(
                "   RiskManager config loaded from DB: sections=%s flat_overrides=%s"
                % (sorted(k for k, v in risk_config.items() if isinstance(v, dict)),
                   sorted(k for k, v in risk_config.items() if not isinstance(v, dict)))
            )
        except Exception as e:
            logger.warning(f"risk config load failed: {e}; using RiskManager code defaults")
            risk_config = {}
        try:
            risk_manager = RiskManager(config=risk_config,
                                       portfolio_manager=None,
                                       config_manager=config_manager)
        except Exception as e:
            logger.warning(f"populated RiskManager init failed: {e}; retrying with code defaults")
            risk_manager = RiskManager(config={}, portfolio_manager=None, config_manager=None)
    except Exception as e:
        logger.warning(f"core.risk_manager wiring failed: {e}; AI will run without it")
    engine = SentimentEngine(config, db_pool, risk_manager=risk_manager)

    try:
        await engine.initialize()
        logger.info("✅ Sentiment Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Engine initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Initialize Telegram controller for remote control (credentials from secrets manager)
    telegram_controller = None
    if get_telegram_controller:
        try:
            # Wave-11 FIX 2: tag with module_name so start_polling honors TELEGRAM_POLL_OWNER.
            telegram_controller = get_telegram_controller(db_pool, module_name='ai')
            if await telegram_controller.initialize():
                telegram_controller.register_module(
                    name='ai_analysis',
                    engine=engine,
                    start_method='run',
                    stop_method='stop',
                    positions_attr='active_signals'
                )
                await telegram_controller.start_polling()
                logger.info("📱 Telegram remote control enabled")
                await telegram_controller.notify("AI Analysis Module started. Send /help for commands.", priority="normal", category="lifecycle")
        except Exception as e:
            logger.warning(f"Telegram controller failed to initialize: {e}")

    try:
        await engine.run()
    except KeyboardInterrupt:
        if telegram_controller:
            await telegram_controller.notify("AI Analysis module shutting down...", priority="high")
            await telegram_controller.stop_polling()
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Engine error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if telegram_controller:
            await telegram_controller.stop_polling()
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
