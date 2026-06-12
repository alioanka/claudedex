"""DEX position service — DB-first price refresh + flag-file close.

Runs inside the DEX subprocess. Two independent background loops:

1. price_refresh_loop: keeps every OPEN row in the shared `trades` table
   priced. The shared engine only refreshes positions held in its in-memory
   `active_positions` dict, and `engine._load_state()` is a no-op, so after a
   restart NO open position gets a price update and its PnL freezes at entry
   (operator's "stuck GRAIL position, ~5 days, no update"). This loop is the
   source of truth for OPEN-position price/PnL on the `/dex/*` dashboard,
   which reads `metadata.current_price`.

2. close_flag_loop: honours `logs/.close_dex_<id>` files dropped by the
   dashboard's manual-close button (issues 5 + 6). Mirrors the copy-trading
   `logs/.close_copy_<trade_id>` IPC pattern. `<id>` is the integer SERIAL
   `trades.id` (the value the dashboard already returns as `position['id']`).

Both loops are DB-first so they work even when the position is not in the
engine's in-memory dict (the common case after a subprocess restart).

Wave-15 (week-1 tuning): price_refresh_loop doubles as a DB-first EXIT
WATCHDOG. Week-1 DRY_RUN data (1638 trades, 42.1% WR) showed realized
stop-losses averaging -14.51% against a 12% configured stop — polling
latency plus restart-orphaned rows that NOTHING exits (the 5-day GRAIL
zombie). The watchdog enforces, per refresh pass:

  * stop-loss with early-fire buffer: triggers at
    -(stop_loss_pct - sl_trigger_buffer_pct) so the REALIZED loss lands
    near the configured stop instead of overshooting it;
  * take-profit at risk_management.take_profit_pct (the DB value becomes
    the authoritative operator surface);
  * ratchet trailing stop mirroring the in-engine tiers (6% base, 5% at
    15%+ peak, 3% at 30%+, 2% at 50%+);
  * max-hold time-limit with a grace margin over the in-engine cap, so the
    engine (when alive) always exits first — this is the zombie-row sweep;
  * fast polling (watchdog_fast_poll_seconds) while any position sits
    within sl_fast_poll_band_pct above the stop trigger.

SAFETY: watchdog exits are SIMULATED closes and run ONLY when the module
is in DRY_RUN — in live mode the engine owns exits (a DB-only close would
desync the row from real on-chain holdings). Kill-switch / pause flags
suppress enforcement (price refresh keeps running — pure observability).
All thresholds are DB-configurable (migration 095); no new
config_manager.py fields are introduced.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Module.DexTrading.PositionService")

# EVM/Solana DEX chains DEX module trades on. Mirrors dex_module supported set
# minus chains DexScreener cannot price.
_DEX_CHAINS = (
    'ethereum', 'bsc', 'polygon', 'arbitrum', 'base',
    'optimism', 'avalanche', 'solana',
)

_LOG_DIR = Path('logs')
_CLOSE_FLAG_PREFIX = '.close_dex_'
_KILLSWITCH_FLAG = _LOG_DIR / '.killswitch'

# Exit-policy defaults — mirror config_manager model defaults exactly so the
# watchdog behaves identically with or without DB rows (migration 095 seeds
# the same values). Keys map 1:1 to RiskManagementConfig /
# PositionManagementConfig fields.
_POLICY_DEFAULTS = {
    'stop_loss_pct': 0.12,
    'sl_trigger_buffer_pct': 0.02,
    'sl_fast_poll_band_pct': 0.04,
    'take_profit_pct': 0.24,
    'max_hold_minutes': 60.0,
    'trailing_enabled': True,
    'trailing_pct': 6.0,
    'trailing_activation_pct': 10.0,
}
# dex_module config_settings keys (migration 095) — read directly from the DB
# because they are module-local and must not become config_manager fields.
_WATCHDOG_DEFAULTS = {
    'db_exit_watchdog_enabled': True,
    'watchdog_fast_poll_seconds': 5,
    'watchdog_max_hold_grace_pct': 25,
}
_WATCHDOG_KEYS_TTL_S = 300.0
# Fast polling refetches every open position; above this count we stay at the
# normal interval to respect DexScreener rate limits (300 req/min).
_FAST_POLL_MAX_POSITIONS = 25

# A position with no successful price fetch for this many seconds is flagged
# unroutable/illiquid in metadata (UI shows it instead of a frozen price).
_STALE_AFTER_SECONDS = 30 * 60
# Consecutive failures before we flag a position stale even sooner.
_STALE_AFTER_FAILURES = 5


class DexPositionService:
    """DB-first price refresh + manual-close poller for OPEN DEX positions."""

    def __init__(self, db_manager, dex_collector, *, dry_run: bool,
                 refresh_interval: int = 30, close_poll_interval: int = 15,
                 config_manager=None, engine=None):
        self.db = db_manager
        self.collector = dex_collector
        self.dry_run = dry_run
        self.refresh_interval = refresh_interval
        self.close_poll_interval = close_poll_interval
        # ConfigManager (DB-backed) — source of the exit-policy thresholds.
        self.config_manager = config_manager
        # Shared TradingBotEngine — when the watchdog closes a row the engine
        # also holds in memory, we pop it from engine.active_positions so the
        # in-engine monitor does not double-close the same trades row.
        self.engine = engine
        self._stop = asyncio.Event()
        # token_address -> consecutive price-fetch failure count
        self._fail_counts: Dict[str, int] = {}
        # dex_module watchdog keys cache (direct config_settings reads).
        self._watchdog_keys = dict(_WATCHDOG_DEFAULTS)
        self._watchdog_keys_loaded_at = 0.0
        # Set during a refresh pass when any position is near its stop.
        self._fast_poll_armed = False
        self._live_watchdog_notice_logged = False

    def stop(self) -> None:
        self._stop.set()

    # ---- DB helpers -----------------------------------------------------

    @property
    def _pool(self):
        return getattr(self.db, 'pool', None)

    async def _fetch_open_positions(self) -> List[Dict]:
        if not self._pool:
            return []
        chains = ', '.join(f"'{c}'" for c in _DEX_CHAINS)
        query = f"""
            SELECT id, trade_id, token_address, chain, entry_price, amount,
                   usd_value, entry_timestamp, metadata, status
            FROM trades
            WHERE status = 'open' AND side = 'buy'
              AND chain IN ({chains})
            ORDER BY entry_timestamp DESC
            LIMIT 200
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"open-position fetch failed: {e}")
            return []

    @staticmethod
    def _parse_metadata(raw) -> Dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # ---- exit policy (Wave-15 DB-first watchdog) ------------------------

    def _load_policy(self) -> Dict:
        """Exit-policy thresholds from ConfigManager (DB-backed, migration
        095 seeds). Falls back field-by-field to _POLICY_DEFAULTS; never
        raises. Cheap (in-memory model attrs) — called once per pass."""
        policy = dict(_POLICY_DEFAULTS)
        cm = self.config_manager
        if cm is None:
            return policy
        try:
            from config.config_manager import ConfigType
            rm = cm.get_config(ConfigType.RISK_MANAGEMENT)
            if rm is not None:
                policy['stop_loss_pct'] = float(getattr(rm, 'stop_loss_pct', policy['stop_loss_pct']))
                policy['sl_trigger_buffer_pct'] = float(getattr(rm, 'sl_trigger_buffer_pct', policy['sl_trigger_buffer_pct']))
                policy['sl_fast_poll_band_pct'] = float(getattr(rm, 'sl_fast_poll_band_pct', policy['sl_fast_poll_band_pct']))
                policy['take_profit_pct'] = float(getattr(rm, 'take_profit_pct', policy['take_profit_pct']))
            pm = cm.get_config(ConfigType.POSITION_MANAGEMENT)
            if pm is not None:
                policy['max_hold_minutes'] = float(getattr(pm, 'max_hold_time_minutes', policy['max_hold_minutes']))
                policy['trailing_enabled'] = bool(getattr(pm, 'trailing_stop_enabled', policy['trailing_enabled']))
                policy['trailing_pct'] = float(getattr(pm, 'trailing_stop_percent', policy['trailing_pct']))
                policy['trailing_activation_pct'] = float(getattr(pm, 'trailing_stop_activation', policy['trailing_activation_pct']))
        except Exception as e:
            logger.debug(f"exit-policy load failed, using defaults: {e}")
        # Sanity clamps: the buffer must never invert the stop, thresholds
        # must stay positive — a fat-fingered DB row must not disable exits.
        policy['stop_loss_pct'] = max(0.01, policy['stop_loss_pct'])
        policy['sl_trigger_buffer_pct'] = min(
            max(0.0, policy['sl_trigger_buffer_pct']), policy['stop_loss_pct'] * 0.5
        )
        policy['take_profit_pct'] = max(0.01, policy['take_profit_pct'])
        policy['max_hold_minutes'] = max(1.0, policy['max_hold_minutes'])
        policy['trailing_pct'] = max(0.5, policy['trailing_pct'])
        return policy

    async def _load_watchdog_keys(self) -> Dict:
        """dex_module rows from config_settings (migration 095), cached for
        _WATCHDOG_KEYS_TTL_S. Direct DB read by design: module-local keys
        must not become config_manager.py fields (campaign rule)."""
        now = time.monotonic()
        if now - self._watchdog_keys_loaded_at < _WATCHDOG_KEYS_TTL_S:
            return self._watchdog_keys
        self._watchdog_keys_loaded_at = now
        if not self._pool:
            return self._watchdog_keys
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value FROM config_settings "
                    "WHERE config_type = 'dex_module'"
                )
            keys = dict(_WATCHDOG_DEFAULTS)
            for row in rows:
                k, v = row['key'], row['value']
                if k == 'db_exit_watchdog_enabled':
                    keys[k] = str(v).strip().lower() in ('true', '1', 'yes', 'on')
                elif k in ('watchdog_fast_poll_seconds', 'watchdog_max_hold_grace_pct'):
                    try:
                        keys[k] = max(1, int(float(v)))
                    except (TypeError, ValueError):
                        pass
            self._watchdog_keys = keys
        except Exception as e:
            logger.debug(f"watchdog key load failed, keeping previous: {e}")
        return self._watchdog_keys

    def _enforcement_active(self) -> bool:
        """Watchdog exits run ONLY under module DRY_RUN (simulated closes).
        In live mode the engine owns exits — a DB-only close would desync
        the row from real on-chain holdings. Kill-switch / pause suppress
        enforcement too; price refresh continues either way."""
        if not self.dry_run:
            if not self._live_watchdog_notice_logged:
                self._live_watchdog_notice_logged = True
                logger.info(
                    "DB exit watchdog inactive: module is LIVE — exits are "
                    "engine-owned (watchdog closes are simulation-only)"
                )
            return False
        if not self._watchdog_keys.get('db_exit_watchdog_enabled', True):
            return False
        try:
            if _KILLSWITCH_FLAG.exists():
                return False
            from core.dry_run import is_module_paused
            if is_module_paused('dex'):
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _ratchet_trail_pct(max_profit_pct: float, base_trail_pct: float) -> float:
        """Mirror of the in-engine ratchet tiers (core/engine.py
        _check_exit_conditions): tighten the trail as the peak grows."""
        if max_profit_pct >= 50:
            return min(base_trail_pct, 2.0)
        if max_profit_pct >= 30:
            return min(base_trail_pct, 3.0)
        if max_profit_pct >= 15:
            return min(base_trail_pct, 5.0)
        return base_trail_pct

    def _evaluate_exit(self, pnl_pct: float, metadata: Dict,
                       entry_timestamp, policy: Dict,
                       grace_pct: int) -> Tuple[Optional[str], bool]:
        """Returns (exit_reason | None, near_stop). Pure function of the
        refreshed PnL + policy; mutates metadata only to ratchet
        max_profit_pct (persisted by the caller in the same write)."""
        stop_trigger_pct = -(policy['stop_loss_pct'] - policy['sl_trigger_buffer_pct']) * 100.0
        # 1. Stop-loss with early-fire buffer (week-1: realized -14.51% vs
        # -12% config — fire early by the measured overshoot).
        if pnl_pct <= stop_trigger_pct:
            return 'stop_loss_db_watchdog', False
        # 2. Take-profit at the CONFIGURED level (DB row is authoritative).
        if pnl_pct >= policy['take_profit_pct'] * 100.0:
            return 'take_profit_db_watchdog', False
        # 3. Ratchet trailing stop (peak tracked in metadata across passes).
        if policy['trailing_enabled']:
            prev_peak = float(metadata.get('max_profit_pct') or 0.0)
            peak = max(prev_peak, pnl_pct)
            if peak > 0:
                metadata['max_profit_pct'] = round(peak, 4)
            if peak >= policy['trailing_activation_pct']:
                trail = self._ratchet_trail_pct(peak, policy['trailing_pct'])
                if pnl_pct <= peak - trail:
                    return 'trailing_stop_db_watchdog', False
        # 4. Max-hold with grace over the in-engine cap (zombie-row sweep —
        # the engine, when alive, exits first at max_hold_minutes).
        entry_dt = self._parse_dt(entry_timestamp)
        if entry_dt is not None:
            age_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60.0
            if age_min > policy['max_hold_minutes'] * (1.0 + grace_pct / 100.0):
                return 'time_limit_db_watchdog', False
        # 5. Near-stop band -> arm fast polling for the next pass.
        near_stop = pnl_pct <= stop_trigger_pct + policy['sl_fast_poll_band_pct'] * 100.0
        return None, near_stop

    async def _purge_engine_position(self, token_address: str) -> None:
        """Remove a watchdog-closed position from the engine's in-memory
        dict so the in-engine monitor does not double-close the row."""
        eng = self.engine
        if eng is None or not token_address:
            return
        try:
            active = getattr(eng, 'active_positions', None)
            if not isinstance(active, dict):
                return
            lock = getattr(eng, 'positions_lock', None)
            if lock is not None:
                async with lock:
                    active.pop(token_address, None)
                    active.pop(token_address.lower(), None)
            else:
                active.pop(token_address, None)
                active.pop(token_address.lower(), None)
        except Exception as e:
            logger.debug(f"engine position purge failed for {token_address}: {e}")

    # ---- price refresh --------------------------------------------------

    async def _fetch_price(self, token_address: str, chain: str,
                           metadata: Dict) -> Optional[float]:
        """Live price via the position's pair (reliable) then token fallback."""
        pair_address = (
            (metadata.get('pair') or {}).get('pair_address')
            or metadata.get('pair_address')
        )
        if pair_address:
            try:
                pair_data = await self.collector.get_pair_data(
                    pair_address=pair_address, chain=chain
                )
                if pair_data and pair_data.get('price'):
                    return float(pair_data['price'])
            except Exception as e:
                logger.debug(f"pair price fetch failed for {token_address}: {e}")
        try:
            price = await self.collector.get_token_price(
                token_address=token_address, chain=chain
            )
            if price:
                return float(price)
        except Exception as e:
            logger.debug(f"token price fetch failed for {token_address}: {e}")
        return None

    async def _refresh_one(self, pos: Dict, *, policy: Optional[Dict] = None,
                           grace_pct: int = 25, enforce: bool = False) -> None:
        token = pos['token_address']
        chain = (pos.get('chain') or 'ethereum').lower()
        metadata = self._parse_metadata(pos.get('metadata'))
        try:
            entry_price = float(pos.get('entry_price') or 0)
            amount = float(pos.get('amount') or 0)
        except (TypeError, ValueError):
            return

        price = await self._fetch_price(token, chain, metadata)
        now = datetime.now(timezone.utc)

        if price and price > 0:
            self._fail_counts[token] = 0
            pnl = (price - entry_price) * amount
            pnl_pct = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
            metadata['current_price'] = price
            metadata['_last_price_update'] = now.isoformat()
            metadata.pop('price_stale', None)
            metadata.pop('price_stale_reason', None)

            # Wave-15 DB-first exit watchdog (DRY_RUN only — see
            # _enforcement_active). Evaluated on the FRESH quote so the
            # simulated fill is honest.
            if enforce and policy and entry_price > 0:
                reason, near_stop = self._evaluate_exit(
                    pnl_pct, metadata, pos.get('entry_timestamp'), policy, grace_pct
                )
                if reason:
                    await self._close_row(
                        pos['id'], token, entry_price, amount, metadata,
                        exit_price=price, reason=reason,
                    )
                    return
                if near_stop:
                    self._fast_poll_armed = True

            await self._update_trade(pos['id'], {
                'profit_loss': float(pnl),
                'profit_loss_percentage': float(pnl_pct),
                'metadata': metadata,
            })
            return

        # price fetch failed — track + flag stale rather than freeze forever
        failures = self._fail_counts.get(token, 0) + 1
        self._fail_counts[token] = failures
        last_update = self._parse_dt(metadata.get('_last_price_update')) or self._parse_dt(pos.get('entry_timestamp'))
        age_s = (now - last_update).total_seconds() if last_update else 0

        # Wave-15: max-hold backstop also fires on UNPRICEABLE rows — the
        # exact zombie case (5-day GRAIL position with no routable price).
        # Closes at the last-known price, flagged exit_price_stale for audit.
        if enforce and policy and entry_price > 0:
            entry_dt = self._parse_dt(pos.get('entry_timestamp'))
            if entry_dt is not None:
                age_min = (now - entry_dt).total_seconds() / 60.0
                if age_min > policy['max_hold_minutes'] * (1.0 + grace_pct / 100.0):
                    fallback_price = float(metadata.get('current_price') or entry_price)
                    metadata['exit_price_stale'] = True
                    await self._close_row(
                        pos['id'], token, entry_price, amount, metadata,
                        exit_price=fallback_price, reason='time_limit_db_watchdog',
                    )
                    return

        if failures >= _STALE_AFTER_FAILURES or age_s >= _STALE_AFTER_SECONDS:
            metadata['price_stale'] = True
            metadata['price_stale_reason'] = (
                f"no routable price after {failures} attempts "
                f"({int(age_s / 60)}m since last update) — token may be "
                f"unroutable/illiquid"
            )
            await self._update_trade(pos['id'], {'metadata': metadata})
            logger.warning(
                f"⚠️ DEX position {pos['id']} ({metadata.get('token_symbol', token[:10])}) "
                f"flagged stale: {metadata['price_stale_reason']}"
            )

    @staticmethod
    def _parse_dt(value) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(str(value))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    async def _update_trade(self, trade_id: int, updates: Dict) -> None:
        try:
            await self.db.update_trade(int(trade_id), updates)
        except Exception as e:
            logger.error(f"trade {trade_id} update failed: {e}")

    async def _close_row(self, pid: int, token_address: str,
                         entry_price: float, amount: float, metadata: Dict,
                         *, exit_price: float, reason: str) -> Dict:
        """Mark one trades row closed (simulated under DRY_RUN) and purge it
        from the engine's in-memory dict. Shared by the manual-close path and
        the Wave-15 exit watchdog."""
        pnl = (exit_price - entry_price) * amount
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        simulated = self.dry_run
        metadata['close_reason'] = reason
        metadata['closed_simulated'] = simulated

        await self._update_trade(pid, {
            'exit_price': float(exit_price),
            'exit_timestamp': datetime.now(timezone.utc),
            'profit_loss': float(pnl),
            'profit_loss_percentage': float(pnl_pct),
            'status': 'closed',
            'metadata': metadata,
        })
        await self._purge_engine_position(token_address)
        logger.info(
            f"{'📝 DRY-RUN ' if simulated else '💰 '}DEX position {pid} closed "
            f"[{reason}] (exit=${exit_price:.8f}, pnl=${pnl:.2f} / {pnl_pct:+.2f}%)"
        )
        return {'success': True, 'simulated': simulated,
                'exit_price': exit_price, 'profit_loss': pnl}

    async def price_refresh_loop(self) -> None:
        logger.info("📊 DEX DB-first price-refresh + exit-watchdog loop started")
        while not self._stop.is_set():
            interval = self.refresh_interval
            try:
                self._fast_poll_armed = False
                keys = await self._load_watchdog_keys()
                enforce = self._enforcement_active()
                policy = self._load_policy() if enforce else None
                grace_pct = int(keys.get('watchdog_max_hold_grace_pct', 25))

                positions = await self._fetch_open_positions()
                for pos in positions:
                    if self._stop.is_set():
                        break
                    await self._refresh_one(
                        pos, policy=policy, grace_pct=grace_pct, enforce=enforce
                    )
                if positions:
                    logger.debug(f"📊 Refreshed {len(positions)} open DEX positions")

                # Fast polling while any position sits within the
                # sl_fast_poll_band above its stop trigger — cuts the
                # stop-detection latency that drove realized SL to -14.5%.
                # Skipped above _FAST_POLL_MAX_POSITIONS to respect API limits.
                if (self._fast_poll_armed
                        and len(positions) <= _FAST_POLL_MAX_POSITIONS):
                    interval = min(
                        interval,
                        int(keys.get('watchdog_fast_poll_seconds', 5)),
                    )
                    logger.debug(
                        f"⚡ near-stop fast poll armed — next pass in {interval}s"
                    )
            except Exception as e:
                logger.error(f"price-refresh loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ---- manual close (flag-file IPC) -----------------------------------

    async def close_position(self, position_id) -> Dict:
        """Close one OPEN DEX position by integer trades.id. DB-first so it
        works post-restart. Under DRY_RUN (or kill-switch/pause) this is a
        simulated close: marks the row closed at last-known price."""
        if not self._pool:
            return {'success': False, 'error': 'db pool unavailable'}
        try:
            pid = int(position_id)
        except (TypeError, ValueError):
            return {'success': False, 'error': f'bad position id {position_id!r}'}

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, token_address, chain, entry_price, amount, metadata, status "
                "FROM trades WHERE id = $1",
                pid,
            )
        if not row:
            return {'success': False, 'error': f'no trade id {pid}'}
        if row['status'] != 'open':
            return {'success': True, 'note': f'already {row["status"]}'}

        metadata = self._parse_metadata(row['metadata'])

        # LIVE module + REAL fill: a DB-only close would mark the row closed
        # while the tokens stay on-chain. Route through the engine's real
        # on-chain sell instead; REFUSE the DB-only close when that is not
        # possible (never silently desync DB from holdings). DRY_RUN rows
        # (metadata.is_dry_run, the engine writes it on every entry) keep the
        # simulated-close behavior unchanged.
        row_is_real_fill = metadata.get('is_dry_run') is False
        if not self.dry_run and row_is_real_fill:
            eng = self.engine
            token = row['token_address'] or ''
            position = None
            if eng is not None:
                active = getattr(eng, 'active_positions', None) or {}
                position = active.get(token) or active.get(token.lower())
            if position is not None:
                try:
                    ok = await eng._close_position(position, 'manual_dashboard')
                    return {'success': bool(ok), 'live': True}
                except Exception as e:
                    logger.error(f"live manual close via engine failed for {pid}: {e}")
                    return {'success': False, 'error': f'live close failed: {e}'}
            logger.error(
                f"manual close {pid}: LIVE fill not present in engine "
                f"active_positions — refusing DB-only close (tokens would "
                f"remain on-chain). Restart the module or sell manually."
            )
            return {'success': False,
                    'error': 'live position not tracked by engine; DB-only close refused'}

        try:
            entry_price = float(row['entry_price'] or 0)
            amount = float(row['amount'] or 0)
        except (TypeError, ValueError):
            entry_price, amount = 0.0, 0.0

        # Best-effort fresh exit price; fall back to last-known then entry.
        exit_price = await self._fetch_price(
            row['token_address'], (row['chain'] or 'ethereum').lower(), metadata
        )
        if not exit_price:
            exit_price = float(metadata.get('current_price') or entry_price)

        return await self._close_row(
            pid, row['token_address'], entry_price, amount, metadata,
            exit_price=float(exit_price), reason='manual_dashboard',
        )

    async def _process_close_flags(self) -> None:
        if not _LOG_DIR.is_dir():
            return
        for flag in _LOG_DIR.glob(f'{_CLOSE_FLAG_PREFIX}*'):
            raw_id = flag.name[len(_CLOSE_FLAG_PREFIX):]
            if not raw_id:
                self._unlink(flag)
                continue
            try:
                result = await self.close_position(raw_id)
                logger.info(
                    f"manual close dex {raw_id}: success={result.get('success')} "
                    f"note={result.get('note')} error={result.get('error')}"
                )
            except Exception as e:
                logger.error(f"close-flag {raw_id} failed: {e}")
            finally:
                self._unlink(flag)

    @staticmethod
    def _unlink(flag: Path) -> None:
        try:
            flag.unlink()
        except Exception:
            pass

    async def close_flag_loop(self) -> None:
        logger.info("🚪 DEX close-flag poller started (logs/.close_dex_<id>)")
        while not self._stop.is_set():
            try:
                await self._process_close_flags()
            except Exception as e:
                logger.error(f"close-flag loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.close_poll_interval)
            except asyncio.TimeoutError:
                pass
