#!/usr/bin/env python3
"""
Basis Desk Module — cross-venue funding/basis delta-neutral carry ADVISOR.

Polls FREE public perp-funding + spot prices (Bybit V5, Binance) per symbol,
costs out the COMPLETE hedged structure (perp leg + the equal-base-quantity
spot hedge leg — the piece futures funding-carry v2 leaves to a manual
operator action), and shadow-records actionable suggestions to
basis_carry_suggestions (is_simulated=true). ADVISORY/SHADOW ONLY by
default; the live order path is gated (shadow_mode + live_execution_enabled
+ should_skip_live + RiskManager) AND intentionally unwired — see
modules/basis_desk/executor.py.

Launched by main.py when BASIS_DESK_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8103 (env override: BASIS_DESK_HEALTH_PORT).
Config: DB-backed config_type='basis_desk' (migration 131).
Kill switches: logs/.killswitch (global), logs/.pause_basis_desk (per-module).
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
log_dir = Path("logs/basis_desk")
log_dir.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("BasisDeskModule")
logger.setLevel(logging.INFO)
main_handler = RotatingFileHandler(log_dir / 'basis_desk.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)
error_handler = RotatingFileHandler(log_dir / 'basis_desk_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)
for sub in ("BasisDeskModule.Venues", "BasisDeskModule.Executor"):
    sub_logger = logging.getLogger(sub)
    sub_logger.setLevel(logging.INFO)
    sub_logger.addHandler(main_handler)
    sub_logger.addHandler(error_handler)
    sub_logger.addHandler(console)

from modules.basis_desk.carry_math import (  # noqa: E402
    ConfirmTracker, FeeModel, plan_carry,
)
from modules.basis_desk.venues import build_clients  # noqa: E402
from modules.basis_desk.executor import BasisDeskExecutor  # noqa: E402


async def load_config(db_pool) -> dict:
    """Load config_type='basis_desk' rows (mirrors the polymarket loader)."""
    settings = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'basis_desk'"
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
        logger.info(f"Loaded {len(settings)} basis_desk settings from database")
    except Exception as e:
        logger.error(f"Failed to load basis_desk settings from DB: {e}")
    return settings


class BasisDeskEngine:
    """Poll venues -> cost hedged carry -> confirm -> shadow-record advice."""

    def __init__(self, db_pool, config: dict, risk_manager, module_dry_run: bool):
        self.db_pool = db_pool
        self.config = config
        self.module_dry_run = module_dry_run
        self.executor = BasisDeskExecutor(db_pool, config, risk_manager)
        self.confirm = ConfirmTracker(int(config.get('confirm_polls', 2)))
        self.running = False
        self._last_recorded: dict = {}  # (venue, symbol) -> monotonic ts
        self.stats = {
            'cycles': 0, 'venues_ok': 0, 'quotes_evaluated': 0,
            'actionable': 0, 'suggestions_recorded': 0,
            'best_last_cycle': None, 'last_cycle_at': None, 'last_error': None,
        }

    def _throttled(self, venue: str, symbol: str) -> bool:
        interval = float(self.config.get('suggest_interval_s', 1800))
        key = (venue, symbol)
        now = time.monotonic()
        last = self._last_recorded.get(key)
        if last is not None and (now - last) < interval:
            return True
        self._last_recorded[key] = now
        if len(self._last_recorded) > 2000:
            self._last_recorded.clear()
        return False

    def _fees(self) -> FeeModel:
        c = self.config
        return FeeModel(
            perp_taker_fee_bps=float(c.get('perp_taker_fee_bps', 6.0)),
            spot_taker_fee_bps=float(c.get('spot_taker_fee_bps', 10.0)),
            perp_slippage_bps=float(c.get('perp_slippage_bps', 5.0)),
            spot_slippage_bps=float(c.get('spot_slippage_bps', 5.0)),
            liquidation_premium_bps=float(c.get('liquidation_premium_bps', 2.0)),
        )

    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused('basis_desk'):
            logger.info("Module paused (logs/.pause_basis_desk) — idling")
            return

        self.config.update(await load_config(self.db_pool))
        self.executor.config = self.config
        self.confirm.confirm_polls = max(1, int(self.config.get('confirm_polls', 2)))

        symbols = [s.strip().upper() for s in str(
            self.config.get('symbols',
                            'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT')
        ).split(',') if s.strip()]
        clients = build_clients(self.config)
        if not clients or not symbols:
            logger.warning("No venues/symbols configured — idle cycle")
            return

        results = await asyncio.gather(
            *(c.fetch_quotes(symbols) for c in clients.values()),
            return_exceptions=True,
        )
        self.stats['venues_ok'] = sum(
            1 for r in results if isinstance(r, dict) and r)

        fees = self._fees()
        best = None
        for res in results:
            if not isinstance(res, dict):
                continue
            for q in res.values():
                plan = plan_carry(
                    symbol=q.symbol, venue=q.venue,
                    funding_bps=q.funding_bps,
                    perp_price=q.perp_price, spot_price=q.spot_price,
                    fees=fees,
                    funding_interval_hours=float(
                        self.config.get('funding_interval_hours', 8.0)),
                    horizon_intervals=float(
                        self.config.get('horizon_intervals', 6.0)),
                    min_net_carry_bps=float(
                        self.config.get('min_net_carry_bps', 10.0)),
                    max_breakeven_intervals=float(
                        self.config.get('max_breakeven_intervals', 3.0)),
                    allow_short_spot=bool(
                        self.config.get('allow_short_spot', False)),
                )
                if plan is None:
                    continue
                self.stats['quotes_evaluated'] += 1
                if best is None or (plan.net_carry_bps_at_horizon
                                    > best.net_carry_bps_at_horizon):
                    best = plan
                confirmed = self.confirm.update(
                    f'{plan.venue}:{plan.symbol}', plan.actionable,
                    plan.direction)
                if not plan.actionable:
                    logger.debug(f"[skip] {plan.venue}:{plan.symbol} {plan.reason}")
                    continue
                self.stats['actionable'] += 1
                if not confirmed:
                    logger.info(
                        f"[confirming] {plan.venue}:{plan.symbol} "
                        f"{plan.direction} f={plan.funding_bps:+.2f}bps "
                        f"net@{plan.horizon_intervals:.0f}iv="
                        f"{plan.net_carry_bps_at_horizon:.1f}bps — awaiting "
                        f"{self.confirm.confirm_polls} consecutive polls")
                    continue
                if self._throttled(plan.venue, plan.symbol):
                    continue
                result = await self.executor.advise(
                    plan, module_dry_run=self.module_dry_run)
                self.stats['suggestions_recorded'] += 1
                logger.info(
                    f"[advise] {plan.venue}:{plan.symbol} {plan.direction} "
                    f"perp {plan.perp_side} / spot {plan.spot_side} "
                    f"f={plan.funding_bps:+.2f}bps basis={plan.basis_bps:+.1f}bps "
                    f"cost={plan.total_cost_bps:.1f}bps "
                    f"breakeven={plan.breakeven_intervals:.2f}iv "
                    f"net@{plan.horizon_intervals:.0f}iv="
                    f"{plan.net_carry_bps_at_horizon:.1f}bps "
                    f"aprGross={plan.apr_gross_pct:.1f}% "
                    f"-> {result['status']} ({result['skip_reason']})")

        if best is not None:
            self.stats['best_last_cycle'] = {
                'venue': best.venue, 'symbol': best.symbol,
                'direction': best.direction,
                'funding_bps': round(best.funding_bps, 3),
                'basis_bps': round(best.basis_bps, 2),
                'net_carry_bps_at_horizon': round(
                    best.net_carry_bps_at_horizon, 2),
                'actionable': best.actionable, 'reason': best.reason,
            }

    async def run(self) -> None:
        self.running = True
        logger.info("Basis desk engine loop starting "
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


class BasisDeskHealthServer:
    """GET /health (liveness) + GET /status (stats). Port: BASIS_DESK_HEALTH_PORT (8103)."""

    def __init__(self, engine: BasisDeskEngine, host: str = "0.0.0.0", port: int = 8103):
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
            "module": "basis_desk",
            "shadow_mode": bool(self.engine.config.get('shadow_mode', True)),
            "live_execution_enabled": bool(self.engine.config.get('live_execution_enabled', False)),
            "dry_run": self.engine.module_dry_run,
            "live_orders": self.engine.executor.live_orders,
            "suggestions_recorded_total": self.engine.executor.suggestions_recorded,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body), content_type="application/json")


async def main():
    logger.info("=== Basis Desk Module Starting (advisory/shadow-first) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Per-module DRY_RUN (BASIS_DESK_DRY_RUN beats DRY_RUN; safe default True).
    module_dry_run = True
    try:
        from core.dry_run import resolve_module_dry_run, start_killswitch_poller
        module_dry_run = resolve_module_dry_run('basis_desk', default=True)
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
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        logger.info("   Database connected")
    except Exception as e:
        logger.error(f"   Database connection failed: {e}. Exiting.")
        return

    config = await load_config(db_pool)
    # DB dry_run row (if present) takes precedence over env resolution.
    if 'dry_run' in config:
        try:
            from core.dry_run import resolve_module_dry_run
            module_dry_run = resolve_module_dry_run(
                'basis_desk', db_row_value=str(config['dry_run']), default=True)
            logger.info(f"   DRY_RUN (DB override): {module_dry_run}")
        except Exception:
            pass

    risk_manager = None
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager(config={}, portfolio_manager=None, config_manager=None)
        logger.info("   core.risk_manager wired (gates the — unwired — live path)")
    except Exception as e:
        logger.warning(f"   RiskManager unavailable ({e}) — live gate chain will refuse earlier")

    engine = BasisDeskEngine(db_pool, config, risk_manager, module_dry_run)
    health_port = int(os.getenv('BASIS_DESK_HEALTH_PORT', '8103'))
    health = BasisDeskHealthServer(engine, port=health_port)
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
        logger.info("Basis desk module stopped")


if __name__ == "__main__":
    asyncio.run(main())
