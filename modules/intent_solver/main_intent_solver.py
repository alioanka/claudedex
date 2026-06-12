#!/usr/bin/env python3
"""INTENT_SOLVER module — SHADOW-ONLY scaffold entry point (EXPERIMENTAL).

Doc verdict (docs/agents/NEW_MODULE_IDEAS.md #11): PARK IT. This scaffold only
proves the data path: poll open CoW / UniswapX intents, compare each intent's
limit to a reference DEX quote minus gas, and record simulated fill
opportunities (is_simulated=true) to intent_fill_opportunities. There is NO
live path anywhere in this module — no keys, no signing, no settlement, no
bonding. Solving for real requires a staked CoW solver bond and sub-100ms
quoting infra the bot does not have.

Launched by main.py when INTENT_SOLVER_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8102 (env override: INTENT_SOLVER_HEALTH_PORT).
Config: DB-backed config_type='intent_solver' (migration 130).
Kill switches: logs/.killswitch (global), logs/.pause_intent_solver (per-module).
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

log_dir = Path("logs/intent_solver")
log_dir.mkdir(parents=True, exist_ok=True)
_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("IntentSolverModule")
logger.setLevel(logging.INFO)
main_h = RotatingFileHandler(log_dir / 'intent_solver.log',
                             maxBytes=10 * 1024 * 1024, backupCount=5)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)
err_h = RotatingFileHandler(log_dir / 'intent_solver_errors.log',
                            maxBytes=5 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)
console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)
for _sub in ("IntentSolverModule.Clients",):
    _sl = logging.getLogger(_sub)
    _sl.setLevel(logging.INFO)
    for _h in (main_h, err_h, console):
        _sl.addHandler(_h)

from modules.intent_solver.clients import (  # noqa: E402
    CowClient, UniswapXClient, DexScreenerPriceClient,
    DEFAULT_COW_BASE_URL, DEFAULT_UNISWAPX_BASE_URL,
    DEFAULT_DEXSCREENER_BASE_URL,
)
from modules.intent_solver.evaluator import (  # noqa: E402
    evaluate_fill, normalize_cow_order, normalize_uniswapx_order,
)

# chainId -> CoW orderbook slug (used to quote UniswapX orders too).
CHAIN_ID_TO_COW = {1: "mainnet", 100: "xdai", 42161: "arbitrum_one", 8453: "base"}
# Canonical wrapped-native per CoW chain (for native->USD pricing).
NATIVE_TOKEN = {
    "mainnet": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "xdai": "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d",
    "arbitrum_one": "0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
    "base": "0x4200000000000000000000000000000000000006",
}


async def load_config(db_pool) -> dict:
    """Load config_type='intent_solver' rows (mirrors the polymarket loader)."""
    settings = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'intent_solver'"
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
    except Exception as e:
        logger.error(f"Failed to load intent_solver settings from DB: {e}")
    return settings


class IntentSolverEngine:
    """Poll intents -> shadow-evaluate fillability -> record simulated rows."""

    def __init__(self, db_pool, config: dict):
        self.db_pool = db_pool
        self.config = config
        rpm = int(config.get('max_requests_per_minute', 30))
        self.cow = CowClient(str(config.get('cow_base_url', DEFAULT_COW_BASE_URL)),
                             max_requests_per_minute=rpm)
        self.uniswapx = UniswapXClient(
            str(config.get('uniswapx_base_url', DEFAULT_UNISWAPX_BASE_URL)),
            max_requests_per_minute=rpm)
        self.prices = DexScreenerPriceClient(
            str(config.get('dexscreener_base_url', DEFAULT_DEXSCREENER_BASE_URL)),
            max_requests_per_minute=rpm)
        self.running = False
        self._last_recorded: dict = {}  # order_uid -> monotonic ts
        self.stats = {
            'cycles': 0, 'orders_seen': 0, 'quotes_requested': 0,
            'opportunities_recorded': 0, 'last_cycle_at': None, 'last_error': None,
        }

    def _throttled(self, order_uid: str) -> bool:
        interval = float(self.config.get('shadow_record_interval_s', 300))
        now = time.monotonic()
        last = self._last_recorded.get(order_uid)
        if last is not None and (now - last) < interval:
            return True
        self._last_recorded[order_uid] = now
        if len(self._last_recorded) > 10000:
            self._last_recorded.clear()
        return False

    async def _gather_intents(self, now_ts: float) -> list:
        """Fetch + normalize open intents from every enabled source. Fail-soft."""
        cap = int(self.config.get('max_orders_per_poll', 200))
        intents = []
        if bool(self.config.get('cow_enabled', True)):
            chains = [c.strip() for c in
                      str(self.config.get('cow_chains', 'mainnet')).split(',') if c.strip()]
            for chain in chains:
                raw = await self.cow.fetch_open_orders(chain, cap)
                if not raw and self.cow.last_error:
                    logger.warning(f"CoW {chain}: {self.cow.last_error}")
                for r in raw:
                    o = normalize_cow_order(r, chain)
                    if o:
                        intents.append(o)
        if bool(self.config.get('uniswapx_enabled', False)):
            chain_ids = [int(c) for c in
                         str(self.config.get('uniswapx_chain_ids', '1')).split(',')
                         if str(c).strip().isdigit()]
            for cid in chain_ids:
                if cid not in CHAIN_ID_TO_COW:
                    continue  # no quote source for this chain — skip honestly
                raw = await self.uniswapx.fetch_open_orders(cid, cap)
                if not raw and self.uniswapx.last_error:
                    logger.warning(f"UniswapX {cid}: {self.uniswapx.last_error}")
                for r in raw:
                    o = normalize_uniswapx_order(r, CHAIN_ID_TO_COW[cid], now_ts)
                    if o:
                        intents.append(o)
        return intents

    async def _usd_context(self, chain: str, order: dict) -> tuple:
        """(notional_usd|None, gas_cost_usd) — best-effort, fail-soft to None.

        CoW native_price is in atom space (atoms-native per atom-token), so
        notional_wei = limit_atoms * price needs NO token-decimals lookup.
        """
        native_addr = NATIVE_TOKEN.get(chain)
        native_usd = None
        if native_addr:
            native_usd = await self.prices.price_usd(native_addr)
        if not native_usd:
            native_usd = float(self.config.get('native_usd_fallback', 3000))
        gas_cost_usd = (float(self.config.get('settlement_gas_units', 350000))
                        * float(self.config.get('gas_price_gwei', 5))
                        * 1e-9 * native_usd)
        limit_token = (order['buy_token'] if order['kind'] == 'sell'
                       else order['sell_token'])
        limit_amount = (order['buy_amount'] if order['kind'] == 'sell'
                        else order['sell_amount'])
        atom_price = await self.cow.fetch_native_price(chain, limit_token)
        notional_usd = None
        if atom_price and atom_price > 0:
            notional_usd = limit_amount * atom_price / 1e18 * native_usd
        return notional_usd, gas_cost_usd

    async def _save_opportunity(self, order: dict, quote_amount: int, ev: dict) -> None:
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO intent_fill_opportunities
                        (source, chain, order_uid, order_kind, sell_token, buy_token,
                         sell_amount, buy_amount, quote_amount, gross_edge_bps,
                         gas_bps, buffer_bps, net_edge_bps, notional_usd, edge_usd,
                         gas_cost_usd, partially_fillable, valid_to, details,
                         is_simulated)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            $16,$17,$18,$19,TRUE)
                    """,
                    order['source'], order['chain'], order['order_uid'],
                    order['kind'], order['sell_token'], order['buy_token'],
                    str(order['sell_amount']), str(order['buy_amount']),
                    str(quote_amount), ev['gross_edge_bps'], ev['gas_bps'],
                    ev['buffer_bps'], ev['net_edge_bps'], ev['notional_usd'],
                    ev['edge_usd'], ev['gas_cost_usd'],
                    bool(order.get('partially_fillable')),
                    datetime.fromtimestamp(order['valid_to'], tz=timezone.utc)
                    if order.get('valid_to') else None,
                    json.dumps({'order_class': order.get('order_class')}, default=str),
                )
        except Exception as e:
            logger.error(f"intent_fill_opportunities insert failed: {e}")

    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused('intent_solver'):
            logger.info("Module paused (logs/.pause_intent_solver) — idling")
            return

        self.config.update(await load_config(self.db_pool))
        now_ts = time.time()
        intents = await self._gather_intents(now_ts)
        self.stats['orders_seen'] += len(intents)
        if not intents:
            return

        min_validity = float(self.config.get('min_validity_s', 30))
        quote_budget = int(self.config.get('max_quotes_per_cycle', 10))
        from_addr = str(self.config.get(
            'quote_from_address', '0x1111111111111111111111111111111111111111'))

        for order in intents:
            if quote_budget <= 0:
                break
            if order['sell_token'] == order['buy_token']:
                continue
            if order.get('valid_to') and order['valid_to'] - now_ts < min_validity:
                continue
            if self._throttled(order['order_uid']):
                continue
            quote_amount = await self.cow.fetch_quote(
                order['chain'], kind=order['kind'],
                sell_token=order['sell_token'], buy_token=order['buy_token'],
                amount=(order['sell_amount'] if order['kind'] == 'sell'
                        else order['buy_amount']),
                from_address=from_addr)
            quote_budget -= 1
            self.stats['quotes_requested'] += 1
            if not quote_amount:
                continue
            notional_usd, gas_cost_usd = await self._usd_context(order['chain'], order)
            ev = evaluate_fill(
                order, quote_amount, now_ts=now_ts,
                gas_cost_usd=gas_cost_usd, notional_usd=notional_usd,
                safety_buffer_bps=float(self.config.get('safety_buffer_bps', 20)),
                min_edge_bps=float(self.config.get('min_edge_bps', 30)),
                min_edge_usd=float(self.config.get('min_edge_usd', 5)),
                fallback_gas_bps=float(self.config.get('fallback_gas_bps', 50)))
            if ev['recordable']:
                self.stats['opportunities_recorded'] += 1
                await self._save_opportunity(order, quote_amount, ev)
                logger.info(
                    f"[opportunity] {order['source']}/{order['chain']} "
                    f"{order['order_uid'][:18]} kind={order['kind']} "
                    f"net={ev['net_edge_bps']}bps edge_usd={ev['edge_usd']} (simulated)")

    async def run(self) -> None:
        self.running = True
        logger.info("Intent-solver SHADOW loop starting (no live path exists; "
                    "every row is is_simulated=true)")
        while self.running:
            try:
                await self._cycle()
                self.stats['cycles'] += 1
                self.stats['last_cycle_at'] = datetime.now(timezone.utc).isoformat()
                self.stats['last_error'] = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # fail-soft: subprocess never dies on a cycle error
                self.stats['last_error'] = f"{type(e).__name__}: {e}"
                logger.error(f"Cycle error (continuing): {self.stats['last_error']}")
            await asyncio.sleep(float(self.config.get('poll_interval_s', 60)))

    async def stop(self) -> None:
        self.running = False


class IntentSolverHealthServer:
    """GET /health + /status. Port: INTENT_SOLVER_HEALTH_PORT (default 8102)."""

    def __init__(self, engine: IntentSolverEngine, host: str = "0.0.0.0",
                 port: int = 8102):
        self.engine = engine
        self.host = host
        self.port = port
        self._runner = None

    async def start(self) -> None:
        from aiohttp import web
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

    async def _health(self, request) -> "object":
        from aiohttp import web
        body = {
            "status": "running" if self.engine.running else "stopped",
            "module": "intent_solver",
            "shadow_only": True,
            "live_path_exists": False,
            "cow_last_error": self.engine.cow.last_error,
            "uniswapx_last_error": self.engine.uniswapx.last_error,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body), content_type="application/json")


async def main():
    logger.info("=== Intent Solver Module Starting (SHADOW-ONLY scaffold, "
                "doc verdict: PARK IT) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # No live path exists, but honor the killswitch poll so the dashboard can
    # stop the subprocess like every other module.
    try:
        from core.dry_run import start_killswitch_poller
        start_killswitch_poller()
    except Exception as e:
        logger.warning(f"   Could not start killswitch poller: {e}")

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
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)
        logger.info("   Database connected")
    except Exception as e:
        logger.error(f"   Database connection failed: {e}. Exiting.")
        return

    config = await load_config(db_pool)
    logger.info(f"   Loaded {len(config)} intent_solver settings from database")

    engine = IntentSolverEngine(db_pool, config)
    health_port = int(os.getenv('INTENT_SOLVER_HEALTH_PORT', '8102'))
    health = IntentSolverHealthServer(engine, port=health_port)
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
        logger.info("Intent solver module stopped")


if __name__ == "__main__":
    asyncio.run(main())
