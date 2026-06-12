#!/usr/bin/env python3
"""options_vol Module — crypto options hedging advisor (Deribit, shadow-first).

Data source: Deribit PUBLIC REST API (free, no key). Each cycle it:
  1. reads the fleet's net directional exposure from the open-position tables,
  2. builds a transparent BTC/ETH vol surface (IV recomputed from marks, never
     trusting feed greeks) + trailing realized vol,
  3. emits a HEDGING advisory — protective put, or collar when IV is rich vs
     RV — sized against fleet net delta with a hard premium cap, recorded
     SIMULATED to options_vol_suggestions (is_simulated=true).

LIVE execution is OFF by default and gated (see executor.py):
shadow_mode=false AND live_execution_enabled=true AND not should_skip_live
AND RiskManager.validate_trade AND monthly premium budget AND Deribit creds.
Only BUY legs can ever go live; SELL legs are record-only.

Launched by main.py when OPTIONS_VOL_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8097 (env override: OPTIONS_VOL_HEALTH_PORT).
Config: DB-backed config_type='options_vol' (migration 125).
Kill switches: logs/.killswitch (global), logs/.pause_options_vol (per-module).
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from aiohttp import web  # noqa: E402

# Setup logging
log_dir = Path("logs/options_vol")
log_dir.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("OptionsVolModule")
logger.setLevel(logging.INFO)
main_handler = RotatingFileHandler(log_dir / 'options_vol.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)
error_handler = RotatingFileHandler(log_dir / 'options_vol_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)
for sub in ("OptionsVolModule.Deribit", "OptionsVolModule.Executor",
            "OptionsVolModule.Exposure"):
    sub_logger = logging.getLogger(sub)
    sub_logger.setLevel(logging.INFO)
    sub_logger.addHandler(main_handler)
    sub_logger.addHandler(error_handler)
    sub_logger.addHandler(console)

from modules.options_vol.deribit_client import DeribitPublicClient, DEFAULT_BASE_URL  # noqa: E402
from modules.options_vol.hedge_advisor import (  # noqa: E402
    enrich_chain, build_surface, build_hedge_advisory, atm_iv_for_window,
)
from modules.options_vol.exposure_reader import get_fleet_net_delta_usd  # noqa: E402
from modules.options_vol.executor import OptionsVolExecutor, new_structure_id  # noqa: E402
from modules.options_vol.vol_math import realized_vol, ivrv_ratio  # noqa: E402


async def load_config(db_pool) -> dict:
    """Load config_type='options_vol' rows (mirrors polymarket loader)."""
    settings = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'options_vol'"
            )
        for row in rows:
            val = row['value']
            if isinstance(val, str):
                low = val.lower()
                if low in ('true', 'false'):
                    val = low == 'true'
                else:
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
            settings[row['key']] = val
        logger.info(f"Loaded {len(settings)} options_vol settings from database")
    except Exception as e:
        logger.error(f"Failed to load options_vol settings from DB: {e}")
    return settings


class OptionsVolEngine:
    """Poll Deribit -> surface + IV/RV -> hedge advisory -> record (simulated)."""

    def __init__(self, db_pool, config: dict, risk_manager, module_dry_run: bool):
        self.db_pool = db_pool
        self.config = config
        self.module_dry_run = module_dry_run
        self.deribit = DeribitPublicClient(
            str(config.get('deribit_base_url', DEFAULT_BASE_URL)),
            max_requests_per_minute=int(config.get('max_requests_per_minute', 20)),
        )
        self.executor = OptionsVolExecutor(db_pool, config, risk_manager)
        self.running = False
        self._last_advisory_ts: dict = {}     # currency -> monotonic ts
        self._last_snapshot_ts: dict = {}     # currency -> monotonic ts
        self.stats = {
            'cycles': 0, 'advisories': 0, 'surface_snapshots': 0,
            'last_net_delta_usd': None, 'last_ivrv': {}, 'last_cycle_at': None,
            'last_error': None,
        }

    def _throttled(self, bucket: dict, key: str, interval_s: float) -> bool:
        now = time.monotonic()
        last = bucket.get(key)
        if last is not None and (now - last) < interval_s:
            return True
        bucket[key] = now
        return False

    async def _snapshot_surface(self, currency: str, surface: list,
                                index_price: float, rv) -> None:
        if not self.db_pool or not surface:
            return
        try:
            async with self.db_pool.acquire() as conn:
                for s in surface:
                    await conn.execute(
                        """
                        INSERT INTO options_vol_surface
                            (currency, expiry, tenor_days, atm_strike, atm_iv,
                             rv, ivrv_ratio, index_price, n_strikes)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        """,
                        currency, s['expiry'], s['tenor_days'], s['atm_strike'],
                        s['atm_iv'], rv, ivrv_ratio(s['atm_iv'], rv),
                        index_price, s['n_strikes'],
                    )
            self.stats['surface_snapshots'] += 1
        except Exception as e:
            logger.error(f"options_vol_surface insert failed: {e}")

    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused('options_vol'):
            logger.info("Module paused (logs/.pause_options_vol) — idling")
            return

        self.config.update(await load_config(self.db_pool))
        self.executor.config = self.config

        # 1. Fleet exposure (the reason this module exists)
        exposure = await get_fleet_net_delta_usd(
            self.db_pool, fleet_beta=float(self.config.get('fleet_beta', 1.0)))
        net_delta = exposure['net_delta_usd'] if exposure else None
        self.stats['last_net_delta_usd'] = net_delta

        now = datetime.now(timezone.utc)
        currencies = [c.strip().upper() for c in
                      str(self.config.get('currencies', 'BTC,ETH')).split(',')
                      if c.strip()]
        hedge_currency = str(self.config.get('hedge_currency', 'BTC')).upper()
        rv_hours = int(self.config.get('rv_window_hours', 720))

        for currency in currencies:
            index_price = await self.deribit.get_index_price(currency)
            if not index_price:
                logger.warning(f"[{currency}] no index price "
                               f"(last_error={self.deribit.last_error}) — skipping")
                continue
            summaries = await self.deribit.get_option_book_summaries(currency)
            if not summaries:
                logger.warning(f"[{currency}] empty option chain — skipping")
                continue
            closes = await self.deribit.get_hourly_closes(currency, rv_hours)
            rv = realized_vol(closes, 8760.0)

            chain = enrich_chain(summaries, index_price, now)
            surface = build_surface(chain, index_price)
            atm_iv = atm_iv_for_window(
                surface, float(self.config.get('tenor_min_days', 5)),
                float(self.config.get('tenor_max_days', 21)))
            ratio = ivrv_ratio(atm_iv, rv)
            self.stats['last_ivrv'][currency] = ratio
            logger.info(
                f"[{currency}] index={index_price:.0f} chain={len(chain)} "
                f"expiries={len(surface)} atm_iv={atm_iv} rv={rv} ivrv={ratio}")

            if not self._throttled(
                    self._last_snapshot_ts, currency,
                    float(self.config.get('surface_snapshot_interval_s', 3600))):
                await self._snapshot_surface(currency, surface, index_price, rv)

            # 2. Hedge advisory — only on the configured hedge currency.
            if currency != hedge_currency or net_delta is None:
                continue
            if self._throttled(
                    self._last_advisory_ts, currency,
                    float(self.config.get('shadow_record_interval_s', 3600))):
                continue
            advisory = build_hedge_advisory(
                chain=chain, surface=surface, index_price=index_price,
                fleet_net_delta_usd=net_delta, rv=rv, cfg=self.config)
            if advisory is None:
                logger.info(f"[{currency}] no hedge needed "
                            f"(net_delta=${net_delta:.0f}, threshold="
                            f"${float(self.config.get('hedge_delta_threshold_usd', 1000)):.0f})")
                continue
            self.stats['advisories'] += 1
            structure_id = new_structure_id()
            for leg in advisory['legs']:
                suggestion_type = (advisory['structure'] if len(advisory['legs']) == 1
                                   else f"collar_{leg['leg']}_leg")
                result = await self.executor.execute_leg(
                    structure_id=structure_id, suggestion_type=suggestion_type,
                    currency=currency, leg=leg, signal=advisory,
                    module_dry_run=self.module_dry_run)
                logger.info(
                    f"[hedge:{advisory['structure']}] {leg['side']} "
                    f"{leg['instrument_name']} x{leg['contracts']} "
                    f"(~${abs(leg['premium_usd']):.2f}) hedges "
                    f"${advisory['hedged_delta_usd']:.0f} of "
                    f"${advisory['fleet_net_delta_usd']:.0f} net delta "
                    f"-> {result['status']}"
                    f"{' (' + str(result['skip_reason']) + ')' if result['skip_reason'] else ''}")

    async def run(self) -> None:
        self.running = True
        logger.info("options_vol engine loop starting "
                    f"(shadow_mode={self.config.get('shadow_mode', True)}, "
                    f"live_execution_enabled={self.config.get('live_execution_enabled', False)})")
        while self.running:
            try:
                await self._cycle()
                self.stats['cycles'] += 1
                self.stats['last_cycle_at'] = datetime.now(timezone.utc).isoformat()
                self.stats['last_error'] = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # fail-soft: the subprocess never dies on a cycle error
                self.stats['last_error'] = f"{type(e).__name__}: {e}"
                logger.error(f"Cycle error (continuing): {self.stats['last_error']}")
            await asyncio.sleep(float(self.config.get('poll_interval_s', 300)))

    async def stop(self) -> None:
        self.running = False
        await self.executor.close()


class OptionsVolHealthServer:
    """GET /health (liveness) + GET /status (stats). Port: OPTIONS_VOL_HEALTH_PORT (8097)."""

    def __init__(self, engine: OptionsVolEngine, host: str = "0.0.0.0", port: int = 8097):
        self.engine = engine
        self.host = host
        self.port = port
        self._runner = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/status", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Health server running at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, request: web.Request) -> web.Response:
        body = {
            "status": "running" if self.engine.running else "stopped",
            "module": "options_vol",
            "shadow_mode": bool(self.engine.config.get('shadow_mode', True)),
            "live_execution_enabled": bool(self.engine.config.get('live_execution_enabled', False)),
            "dry_run": self.engine.module_dry_run,
            "deribit_last_error": self.engine.deribit.last_error,
            "live_orders": self.engine.executor.live_orders,
            "simulated_records": self.engine.executor.simulated_records,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body, default=str), content_type="application/json")


async def main():
    logger.info("=== options_vol Module Starting (hedging-first, shadow-first) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Per-module DRY_RUN (OPTIONS_VOL_DRY_RUN beats DRY_RUN; safe default True).
    module_dry_run = True
    try:
        from core.dry_run import resolve_module_dry_run, start_killswitch_poller
        module_dry_run = resolve_module_dry_run('options_vol', default=True)
        logger.info(f"   DRY_RUN (resolved per-module): {module_dry_run}")
        start_killswitch_poller()
    except Exception as e:
        logger.warning(f"   Could not resolve DRY_RUN / start killswitch poller: {e}")

    # DB connection
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("No database credentials (Docker secrets or DATABASE_URL). Exiting.")
        return
    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url)
        logger.info("   Database connected")
    except Exception as e:
        logger.error(f"   Database connection failed: {e}. Exiting.")
        return

    # Secrets manager (only needed if the operator later enables LIVE).
    try:
        from security.secrets_manager import secrets
        secrets.initialize(db_pool)
    except Exception as e:
        logger.warning(f"   Secrets manager init failed (live path degraded): {e}")

    config = await load_config(db_pool)
    # DB dry_run row (if present) takes precedence over env resolution.
    if 'dry_run' in config:
        try:
            from core.dry_run import resolve_module_dry_run
            module_dry_run = resolve_module_dry_run(
                'options_vol', db_row_value=str(config['dry_run']), default=True)
            logger.info(f"   DRY_RUN (DB override): {module_dry_run}")
        except Exception:
            pass

    risk_manager = None
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager(config={}, portfolio_manager=None, config_manager=None)
        logger.info("   core.risk_manager wired (validate_trade gates any live order)")
    except Exception as e:
        logger.warning(f"   RiskManager unavailable ({e}) — live path will refuse to fire")

    engine = OptionsVolEngine(db_pool, config, risk_manager, module_dry_run)
    health_port = int(os.getenv('OPTIONS_VOL_HEALTH_PORT', '8097'))
    health = OptionsVolHealthServer(engine, port=health_port)
    try:
        await health.start()
    except Exception as e:
        logger.warning(f"   Health server failed to start (continuing): {e}")

    try:
        await engine.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")
    finally:
        await engine.stop()
        await health.stop()
        await db_pool.close()
        logger.info("options_vol module stopped")


if __name__ == "__main__":
    asyncio.run(main())
