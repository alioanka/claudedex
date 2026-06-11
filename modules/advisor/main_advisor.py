#!/usr/bin/env python3
"""
Financial Advisor Module — Main Entry Point

ADVICE-ONLY. No order execution. No trading keys.
Generates Long/Short/Neutral signals for CRYPTO, US equities,
BIST, FX/metals, and Turkish Midas funds for operator review.

Enabled via: ADVISOR_MODULE_ENABLED=true in .env (default FALSE)
Health server: http://0.0.0.0:8086  (env override: ADVISOR_HEALTH_PORT)

Configuration (DB-backed, config_type='advisor_config'):
  advisor_anthropic_model    — LLM for rationale generation
  enabled_markets            — comma-sep Market values
  run_interval_minutes       — advice cycle cadence (default 60)
  See modules/advisor/config/advisor_config.py for full inventory.

Kill switch:
  Global: logs/.killswitch (polled by BaseModule via start_killswitch_poller)
  Per-module: logs/.pause_advisor

Logs: logs/advisor/
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (before any module imports that might emit logs)
# ---------------------------------------------------------------------------

log_dir = Path("logs/advisor")
log_dir.mkdir(parents=True, exist_ok=True)

_log_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

_root = logging.getLogger()
_root.setLevel(logging.INFO)

_main_handler = RotatingFileHandler(
    log_dir / "advisor.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_main_handler.setFormatter(_log_fmt)
_main_handler.setLevel(logging.INFO)
_root.addHandler(_main_handler)

_error_handler = RotatingFileHandler(
    log_dir / "advisor_errors.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_error_handler.setFormatter(_log_fmt)
_error_handler.setLevel(logging.WARNING)
_root.addHandler(_error_handler)

_console = logging.StreamHandler()
_console.setFormatter(_log_fmt)
_root.addHandler(_console)

logger = logging.getLogger("AdvisorModule")

# ---------------------------------------------------------------------------
# Module imports (after path setup)
# ---------------------------------------------------------------------------

from aiohttp import web
import asyncpg

from modules.advisor.config.advisor_config import AdvisorConfigManager
from modules.advisor.core.advice_engine import AdviceEngine
from modules.advisor.core.kronos_forecaster import KronosForecaster
from modules.advisor.core.models import Market
from modules.advisor.core.portfolio_engine import AdvisorPortfolioEngine
from modules.advisor.core.risk_engine import AdvisorRiskEngine
from modules.advisor.core.telegram_notifier import AdvisorTelegramBot

# Analyzers
from modules.advisor.core.analyzers.crypto import CryptoAnalyzer
from modules.advisor.core.analyzers.us_equities import USEquitiesAnalyzer
from modules.advisor.core.analyzers.bist import BISTAnalyzer
from modules.advisor.core.analyzers.fx import FXAnalyzer
from modules.advisor.core.analyzers.midas_funds import MidasFundsAnalyzer


# ---------------------------------------------------------------------------
# Health server
# ---------------------------------------------------------------------------

class AdvisorHealthServer:
    """
    Lightweight HTTP health / status surface for the advisor module.
    Port: ADVISOR_HEALTH_PORT (default 8086).

    Endpoints:
      GET /health   — liveness + last cycle time + analyzer statuses
      GET /status   — full diagnostics dict
      GET /advice   — last N advice results (query param: n=10)
    """

    def __init__(self, app: "AdvisorApplication", host: str = "0.0.0.0", port: int = 8086):
        self.app = app
        self.host = host
        self.port = port
        self._runner = None

    async def start(self) -> None:
        web_app = web.Application()
        web_app.router.add_get("/health", self._health)
        web_app.router.add_get("/status", self._status)
        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"[advisor] Health server running at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, request: web.Request) -> web.Response:
        engine = self.app.engine
        diag = engine.get_diagnostics() if engine else {}
        body = {
            "status": "running" if self.app.running else "stopped",
            "module": "advisor",
            "advice_only": True,
            "last_cycle_at": diag.get("last_cycle_at"),
            "cycle_count": diag.get("cycle_count", 0),
            "registered_markets": diag.get("registered_markets", []),
            "kronos": diag.get("kronos_health"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        return web.Response(
            text=json.dumps(body), content_type="application/json"
        )

    async def _status(self, request: web.Request) -> web.Response:
        engine = self.app.engine
        body = engine.get_diagnostics() if engine else {"error": "engine not started"}
        return web.Response(
            text=json.dumps(body), content_type="application/json"
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AdvisorApplication:
    def __init__(self):
        self.running = False
        self.db_pool = None
        self.config_mgr = None
        self.engine: AdviceEngine = None
        self.health_server: AdvisorHealthServer = None
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        logger.info("=== Financial Advisor Module Starting ===")
        logger.info(f"  Working dir: {Path.cwd()}")
        logger.info("  ADVICE-ONLY mode — no trade execution anywhere.")

        # Resolve dry-run (advisors always run in advisory mode; dry-run
        # only affects whether sim positions are opened)
        try:
            from core.dry_run import resolve_module_dry_run, start_killswitch_poller
            advisor_dry = resolve_module_dry_run("advisor", default=True)
            os.environ["DRY_RUN"] = "true" if advisor_dry else "false"
            logger.info(f"  DRY_RUN (per-module): {advisor_dry}")
            start_killswitch_poller()
        except Exception as exc:
            logger.warning(f"  Could not resolve DRY_RUN or start poller: {exc}")

        # DB connection
        try:
            from security.docker_secrets import get_database_url
            db_url = get_database_url()
        except ImportError:
            db_url = os.getenv("DATABASE_URL")

        if not db_url:
            logger.error("No DATABASE_URL — cannot connect to DB. Exiting.")
            return

        try:
            self.db_pool = await asyncpg.create_pool(db_url)
            logger.info("  DB connected.")
        except Exception as exc:
            logger.error(f"  DB connection failed: {exc}. Exiting.")
            return

        # Load config
        self.config_mgr = AdvisorConfigManager(self.db_pool)
        config = await self.config_mgr.load()

        # Resolve advisor API keys from Secure Credentials (encrypted DB) so the
        # operator can manage them in the dashboard's Secure Credentials panel.
        # The analyzers + rationale helper read these from the config dict first
        # (then fall back to os.getenv / .env), so inject any secret that resolves
        # under the config key they look for. Fail-soft: missing/unavailable keys
        # leave the config untouched and the feature stays degraded/off.
        try:
            from security.secrets_manager import secrets as _secrets
            _secrets.initialize(self.db_pool)
            for _secret_key, _config_key in (
                ("ADVISOR_ANTHROPIC_API_KEY", "advisor_anthropic_api_key"),
                ("ADVISOR_OPENAI_API_KEY", "advisor_openai_api_key"),
                ("ADVISOR_FX_ALPHAVANTAGE_KEY", "advisor_fx_alphavantage_key"),
                ("ADVISOR_BIST_API_KEY", "advisor_bist_api_key"),
                ("ADVISOR_FONOLOJI_API_KEY", "advisor_fonoloji_api_key"),
            ):
                if config.get(_config_key):
                    continue
                try:
                    _val = await _secrets.get_async(_secret_key, log_access=False)
                except Exception:
                    _val = None
                if _val:
                    config[_config_key] = _val
        except Exception as exc:
            logger.warning(
                f"  Could not resolve advisor API keys from Secure Credentials: {exc}"
            )

        # Turkish-stack source ACTIVATION diagnostics + yfinance noise filter.
        # install_yfinance_noise_filter demotes the EXPECTED yfinance
        # "$TICKER: possibly delisted; no price data found" ERROR flood (non-
        # equity KAP tickers) to DEBUG so advisor_errors.log stays signal.
        # resolve_sources logs which source each Turkish path resolves to RIGHT
        # NOW (Fonoloji auto-prefer rule, migration 083). Fail-soft.
        try:
            from modules.advisor.core.data import activation as _activation
            _activation.install_yfinance_noise_filter()
            _sources = _activation.resolve_sources(config)
            logger.info(
                "  Turkish-stack sources: bist=%s midas=%s universe=%s kap_prices=%s",
                _sources.get("bist"), _sources.get("midas"),
                _sources.get("universe"), _sources.get("kap_prices"),
            )
        except Exception as exc:
            logger.warning(
                f"  Source-activation diagnostics failed (fail-soft): {exc}"
            )

        # Build analyzers
        analyzers = {
            Market.CRYPTO: CryptoAnalyzer(config, self.db_pool),
            Market.US_EQUITIES: USEquitiesAnalyzer(config, self.db_pool),
            Market.BIST: BISTAnalyzer(config, self.db_pool),
            Market.FX: FXAnalyzer(config, self.db_pool),
            Market.MIDAS_FUNDS: MidasFundsAnalyzer(config, self.db_pool),
        }

        # Build Kronos forecaster
        kronos = None
        if str(config.get("advisor_kronos_enabled", "false")).lower() == "true":
            kronos = KronosForecaster(
                variant=config.get("advisor_kronos_variant", "Kronos-mini"),
                device=config.get("advisor_kronos_device", "cpu"),
            )
            loaded = await kronos.initialize()
            logger.info(
                f"  Kronos: {'loaded' if loaded else 'weights not available (fail-soft)'}"
            )

        # Build engines. Config MUST be passed: without it the portfolio engine
        # silently ignored sim_horizon_days_* and sim_target_fill_policy keys
        # and always used hard-coded defaults.
        portfolio = AdvisorPortfolioEngine(self.db_pool, config)
        risk = AdvisorRiskEngine(config)
        telegram = AdvisorTelegramBot(config)
        await telegram.initialize(self.db_pool)

        self.engine = AdviceEngine(
            config=config,
            analyzers=analyzers,
            portfolio=portfolio,
            risk=risk,
            kronos=kronos,
            telegram=telegram,
            db_pool=self.db_pool,
        )

        # Health server
        health_port = int(os.getenv("ADVISOR_HEALTH_PORT", "8086"))
        self.health_server = AdvisorHealthServer(self, port=health_port)
        await self.health_server.start()

        # Log analyzer data source statuses at startup
        for market, analyzer in analyzers.items():
            status = analyzer.data_source_status()
            logger.info(
                f"  Analyzer [{market.value}]: data_source_status={status.value}"
            )

        self.running = True
        run_interval = int(config.get("run_interval_minutes", 60))
        logger.info(
            f"  Run interval: {run_interval} minutes. "
            f"Enabled markets: {config.get('enabled_markets')}."
        )

        # KAP ingestion (listener + return accumulator) -- Wave-23
        # Fail-soft: import error or disabled config -> log + continue
        _kap_tasks = []
        _kap_enabled = str(config.get("advisor_kap_enabled", "false")).lower() == "true"
        if _kap_enabled:
            try:
                from modules.advisor.core.kap.kap_listener import KapListener
                from modules.advisor.core.kap.forward_return_accumulator import ReturnAccumulator
                from modules.advisor.core.kap.classifier_worker import KapClassifierWorker
                _kap_listener = KapListener(config=config, db_pool=self.db_pool)
                _kap_accumulator = ReturnAccumulator(config=config, db_pool=self.db_pool)
                _kap_classifier = KapClassifierWorker(
                    config=config, db_pool=self.db_pool, telegram=telegram,
                    # KAP-driven sims (strong polarity -> dry-run BIST sim in the
                    # 'kap' channel). Reuses the SAME portfolio engine + BIST
                    # analyzer; fully fail-soft (never breaks the worker).
                    portfolio=portfolio,
                    bist_analyzer=analyzers.get(Market.BIST),
                )
                _kap_tasks.append(asyncio.create_task(_kap_listener.run(), name="kap_listener"))
                _kap_tasks.append(asyncio.create_task(_kap_accumulator.run_daily(), name="kap_accumulator"))
                _kap_tasks.append(asyncio.create_task(_kap_classifier.run(), name="kap_classifier"))
                logger.info("[advisor] KAP listener + return accumulator + classifier worker started.")
            except Exception as exc:
                logger.warning(
                    "[advisor] KAP ingestion failed to start (non-fatal): %s. "
                    "Set advisor_kap_enabled=false to silence or fix the error.", exc
                )
        else:
            logger.info(
                "[advisor] KAP ingestion disabled (advisor_kap_enabled=false). "
                "Set advisor_kap_enabled=true in advisor_config to enable."
            )

        # Main advice loop
        while not self._shutdown.is_set():
            try:
                logger.info(f"[advisor] Starting advice cycle...")
                await self.engine.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[advisor] Cycle error: {exc}", exc_info=True)

            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=run_interval * 60,
                )
            except asyncio.TimeoutError:
                pass  # normal — next cycle

        logger.info("[advisor] Shutting down...")
        # Cancel KAP background tasks cleanly
        for _t in _kap_tasks:
            if not _t.done():
                _t.cancel()
        if _kap_tasks:
            await asyncio.gather(*_kap_tasks, return_exceptions=True)
        await self.health_server.stop()
        if self.db_pool:
            await self.db_pool.close()
        self.running = False
        logger.info("[advisor] Stopped.")

    def _signal_handler(self, sig, frame):
        logger.warning(f"[advisor] Received signal {sig}. Initiating shutdown.")
        self._shutdown.set()


async def main() -> None:
    app = AdvisorApplication()
    signal.signal(signal.SIGINT, app._signal_handler)
    signal.signal(signal.SIGTERM, app._signal_handler)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
