"""Shadow CLMM engine: poll free pool data, propose ranges, mark IL honestly.

Per cycle: (1) snapshot each candidate pool (price/volume/TVL), feeding an
in-process realized-vol estimator; (2) propose a position when the transparent
net-of-IL APR clears the floor (recorded simulated via ClmmExecutor); (3)
mark-to-market every open shadow position against the HODL benchmark — IL is
booked every cycle, never hidden — and simulate the re-range close (with its
cost) when price exits the range or the position ages out.
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.clmm_lp import fee_il_math as m
from modules.clmm_lp.executor import ClmmExecutor
from modules.clmm_lp.pool_data import PoolDataClient, DEFAULT_DEXSCREENER_BASE

logger = logging.getLogger("ClmmLpModule.Engine")

_MAX_VOL_SAMPLES = 4000


async def load_config(db_pool) -> dict:
    """Load config_type='clmm_lp' rows (mirrors the polymarket loader)."""
    settings: dict = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'clmm_lp'"
            )
        for row in rows:
            val = row['value']
            if isinstance(val, str):
                low = val.lower()
                if low in ('true', 'false'):
                    val = low == 'true'
                elif not val.startswith(('[', '{')):
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
            settings[row['key']] = val
    except Exception as e:
        logger.error("Failed to load clmm_lp settings from DB: %s", e)
    return settings


def parse_candidate_pools(raw: Any) -> List[Dict[str, Any]]:
    """Validate the candidate_pools JSON config row. Bad entries are dropped."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            logger.error("candidate_pools is not valid JSON — no candidates")
            return []
    if not isinstance(raw, list):
        return []
    pools = []
    for p in raw:
        if (isinstance(p, dict) and p.get('chain') and p.get('pool_address')
                and isinstance(p.get('fee_rate_bps'), (int, float))
                and p['fee_rate_bps'] > 0):
            pools.append(p)
        else:
            logger.warning("Dropping malformed candidate pool entry: %s", p)
    return pools


class ClmmLpEngine:
    def __init__(self, db_pool, config: dict, risk_manager, module_dry_run: bool):
        self.db_pool = db_pool
        self.config = config
        self.module_dry_run = module_dry_run
        self.executor = ClmmExecutor(db_pool, config, risk_manager)
        self.pool_data = PoolDataClient(
            str(config.get('dexscreener_base_url', DEFAULT_DEXSCREENER_BASE)))
        self.running = False
        self._vol_samples: Dict[str, List] = {}      # pool key -> [(ts, price)]
        self._last_proposed: Dict[str, float] = {}   # pool key -> monotonic ts
        self.stats = {'cycles': 0, 'proposals': 0, 'rejections': 0,
                      'open_positions': 0, 'closed_positions': 0,
                      'last_cycle_at': None, 'last_error': None}

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _key(pool: Dict[str, Any]) -> str:
        return f"{pool['chain']}:{pool['pool_address']}"

    def _annual_vol(self, key: str) -> Dict[str, Any]:
        vol = m.realized_annual_vol(
            self._vol_samples.get(key, []),
            min_samples=int(self.config.get('min_vol_samples', 12)))
        if vol is None:
            return {'vol': float(self.config.get('default_annual_vol', 0.8)),
                    'source': 'default'}
        # Never trust a realized vol BELOW the default with sparse data —
        # underestimating vol overstates net APR (the doc's measurement trap).
        floor = float(self.config.get('default_annual_vol', 0.8))
        # Cap absurd readings: a noisy/erroneous price series (e.g. a thin Solana
        # pool whose DexScreener price jumps) can yield vol > 20 (2000%+),
        # producing nonsense net-APR like -192417%. Clamp to a sane ceiling so
        # the rejection reason stays interpretable. Default 5.0 = 500% annualized
        # (already extreme); operator-tunable via max_annual_vol.
        ceiling = float(self.config.get('max_annual_vol', 5.0))
        if len(self._vol_samples.get(key, [])) < 100 and vol < floor:
            return {'vol': floor, 'source': 'default_floor'}
        if vol > ceiling:
            return {'vol': ceiling, 'source': 'realized_capped'}
        return {'vol': vol, 'source': 'realized'}

    async def _open_count(self) -> int:
        try:
            async with self.db_pool.acquire() as conn:
                return await conn.fetchval(
                    "SELECT COUNT(*) FROM clmm_shadow_positions WHERE status = 'open'")
        except Exception as e:
            logger.error("open-count query failed: %s", e)
            return 10**9  # fail-safe: behave as if at cap

    # ── proposal side ────────────────────────────────────────────────────
    async def _consider_pool(self, pool: Dict[str, Any],
                             snapshot: Dict[str, Any]) -> None:
        key = self._key(pool)
        interval = float(self.config.get('shadow_record_interval_s', 3600))
        last = self._last_proposed.get(key)
        if last is not None and (time.monotonic() - last) < interval:
            return
        try:
            async with self.db_pool.acquire() as conn:
                already_open = await conn.fetchval(
                    "SELECT COUNT(*) FROM clmm_shadow_positions "
                    "WHERE status = 'open' AND chain = $1 AND pool_address = $2",
                    pool['chain'], pool['pool_address'])
        except Exception as e:
            logger.error("dup-check failed (%s) — skipping proposal", e)
            return
        if already_open:
            return
        if await self._open_count() >= int(self.config.get('max_open_positions', 2)):
            return

        # On-chain fee verification (EVM, cached; None = unverified, allowed).
        onchain_bps = await self.pool_data.verify_evm_fee_bps(
            pool['chain'], pool['pool_address'])
        if onchain_bps is not None and onchain_bps != int(pool['fee_rate_bps']):
            logger.warning("Fee-tier mismatch for %s: config=%sbps on-chain=%sbps "
                           "— pool excluded", key, pool['fee_rate_bps'], onchain_bps)
            return

        price = snapshot['price']
        width = float(self.config.get('range_width_pct', 10)) / 100.0
        pa, pb = price * (1.0 - width), price * (1.0 + width)
        vol_info = self._annual_vol(key)
        breakdown = m.expected_net_apr(
            price=price, price_lower=pa, price_upper=pb,
            volume_24h_usd=snapshot['volume_24h_usd'],
            fee_rate=float(pool['fee_rate_bps']) / 10000.0,
            tvl_usd=snapshot['tvl_usd'],
            annual_vol=vol_info['vol'],
            fee_decay_factor=float(self.config.get('fee_decay_factor', 0.7)),
            rebalance_cost_frac=float(self.config.get('rebalance_cost_bps', 30)) / 10000.0,
        )
        min_net = float(self.config.get('min_net_apr_pct', 10)) / 100.0
        if breakdown['net_apr'] < min_net:
            self.stats['rejections'] += 1
            logger.info("[reject] %s net_apr=%.1f%% < %.1f%% (fee=%.1f%% il=%.1f%% "
                        "rebal=%.1f%% vol=%.2f/%s)", key,
                        breakdown['net_apr'] * 100, min_net * 100,
                        breakdown['fee_apr'] * 100, breakdown['il_apr'] * 100,
                        breakdown['rebalance_apr'] * 100,
                        vol_info['vol'], vol_info['source'])
            return

        size_usd = float(self.config.get('max_position_size_usd', 100))
        liq = m.liquidity_for_value(size_usd, price, pa, pb)
        a0, a1 = m.position_amounts(liq, price, pa, pb)
        self._last_proposed[key] = time.monotonic()
        result = await self.executor.open_position(proposal={
            'chain': pool['chain'], 'pool_address': pool['pool_address'],
            'label': pool.get('label', ''), 'fee_rate_bps': pool['fee_rate_bps'],
            'entry_price': price, 'price_lower': pa, 'price_upper': pb,
            'size_usd': size_usd, 'amount0': a0, 'amount1': a1, 'liquidity': liq,
            'expected_fee_apr': breakdown['fee_apr'],
            'expected_il_apr': breakdown['il_apr'],
            'expected_rebalance_apr': breakdown['rebalance_apr'],
            'expected_net_apr': breakdown['net_apr'],
            'annual_vol_used': vol_info['vol'], 'vol_source': vol_info['source'],
            'details': {'snapshot': snapshot,
                        'concentration_factor': breakdown['concentration_factor'],
                        'onchain_fee_bps': onchain_bps},
        }, module_dry_run=self.module_dry_run)
        self.stats['proposals'] += 1
        logger.info("[propose] %s net_apr=%.1f%% range=[%.6g, %.6g] $%.0f -> %s (%s)",
                    key, breakdown['net_apr'] * 100, pa, pb, size_usd,
                    result['status'], result['skip_reason'])

    # ── mark-to-market side (the IL honesty loop) ────────────────────────
    async def _mark_positions(self, snapshots: Dict[str, Dict[str, Any]],
                              pools_by_key: Dict[str, Dict[str, Any]]) -> None:
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM clmm_shadow_positions WHERE status = 'open'")
        except Exception as e:
            logger.error("open-positions fetch failed: %s", e)
            return
        self.stats['open_positions'] = len(rows)
        now = datetime.now(timezone.utc)
        rebal_frac = float(self.config.get('rebalance_cost_bps', 30)) / 10000.0
        for row in rows:
            key = f"{row['chain']}:{row['pool_address']}"
            snap = snapshots.get(key)
            if snap is None:
                continue  # no fresh data this cycle; mark next time
            price = snap['price']
            liq, pa, pb = row['liquidity'], row['price_lower'], row['price_upper']
            il = m.il_usd(liq, row['entry_price'], price, pa, pb)
            in_range = pa <= price <= pb
            # Fee accrual only while in range, at the CURRENT pool fee run-rate.
            fees = float(row['fees_usd'] or 0)
            last_mark = row['last_marked_at'] or row['created_at']
            dt_years = max(0.0, (now - last_mark).total_seconds()) / m.YEAR_SECONDS
            pool_cfg = pools_by_key.get(key)
            if in_range and pool_cfg is not None and dt_years > 0:
                conc = m.concentration_factor(
                    min(max(price, pa * 1.000001), pb * 0.999999), pa, pb)
                fee_apr_now = (conc
                               * m.pool_fee_apr(snap['volume_24h_usd'],
                                                float(pool_cfg['fee_rate_bps']) / 10000.0,
                                                snap['tvl_usd'])
                               * float(self.config.get('fee_decay_factor', 0.7)))
                fees += float(row['size_usd']) * fee_apr_now * dt_years
            net = fees + il

            buffer_frac = float(self.config.get('out_of_range_exit_buffer_pct', 2)) / 100.0
            age_h = (now - row['created_at']).total_seconds() / 3600.0
            close_reason = None
            if price < pa * (1.0 - buffer_frac) or price > pb * (1.0 + buffer_frac):
                close_reason = 'out_of_range'
            elif age_h > float(self.config.get('max_position_age_hours', 168)):
                close_reason = 'max_age'
            try:
                async with self.db_pool.acquire() as conn:
                    if close_reason:
                        # Booking the simulated re-range/burn cost on close.
                        net -= float(row['size_usd']) * rebal_frac
                        await conn.execute(
                            "UPDATE clmm_shadow_positions SET current_price=$2, "
                            "fees_usd=$3, il_usd=$4, net_usd=$5, last_marked_at=$6, "
                            "status='closed', closed_at=$6, close_reason=$7 "
                            "WHERE id=$1",
                            row['id'], price, fees, il, net, now, close_reason)
                        self.stats['closed_positions'] += 1
                        logger.info("[close:%s] id=%s fees=$%.2f il=$%.2f net=$%.2f",
                                    close_reason, row['id'], fees, il, net)
                    else:
                        await conn.execute(
                            "UPDATE clmm_shadow_positions SET current_price=$2, "
                            "fees_usd=$3, il_usd=$4, net_usd=$5, last_marked_at=$6 "
                            "WHERE id=$1",
                            row['id'], price, fees, il, net, now)
            except Exception as e:
                logger.error("position %s mark/close failed: %s", row['id'], e)

    # ── loop ─────────────────────────────────────────────────────────────
    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused('clmm_lp'):
            logger.info("Module paused (logs/.pause_clmm_lp) — idling")
            return
        self.config.update(await load_config(self.db_pool))
        self.executor.config = self.config

        pools = parse_candidate_pools(self.config.get('candidate_pools', '[]'))
        snapshots: Dict[str, Dict[str, Any]] = {}
        pools_by_key: Dict[str, Dict[str, Any]] = {}
        for pool in pools:
            key = self._key(pool)
            pools_by_key[key] = pool
            snap = await self.pool_data.fetch_pool_snapshot(
                pool['chain'], pool['pool_address'])
            if snap is None:
                logger.warning("No snapshot for %s (%s)", key, self.pool_data.last_error)
                continue
            snapshots[key] = snap
            samples = self._vol_samples.setdefault(key, [])
            samples.append((time.time(), snap['price']))
            if len(samples) > _MAX_VOL_SAMPLES:
                del samples[: len(samples) - _MAX_VOL_SAMPLES]
            await self._consider_pool(pool, snap)
        await self._mark_positions(snapshots, pools_by_key)

    async def run(self) -> None:
        self.running = True
        logger.info("CLMM LP engine loop starting (shadow_mode=%s, "
                    "live_execution_enabled=%s)",
                    self.config.get('shadow_mode', True),
                    self.config.get('live_execution_enabled', False))
        import asyncio
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
                logger.error("Cycle error (continuing): %s", self.stats['last_error'])
            await asyncio.sleep(float(self.config.get('poll_interval_s', 300)))

    async def stop(self) -> None:
        self.running = False
