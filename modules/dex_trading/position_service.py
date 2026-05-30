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
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("Module.DexTrading.PositionService")

# EVM/Solana DEX chains DEX module trades on. Mirrors dex_module supported set
# minus chains DexScreener cannot price.
_DEX_CHAINS = (
    'ethereum', 'bsc', 'polygon', 'arbitrum', 'base',
    'optimism', 'avalanche', 'solana',
)

_LOG_DIR = Path('logs')
_CLOSE_FLAG_PREFIX = '.close_dex_'

# A position with no successful price fetch for this many seconds is flagged
# unroutable/illiquid in metadata (UI shows it instead of a frozen price).
_STALE_AFTER_SECONDS = 30 * 60
# Consecutive failures before we flag a position stale even sooner.
_STALE_AFTER_FAILURES = 5


class DexPositionService:
    """DB-first price refresh + manual-close poller for OPEN DEX positions."""

    def __init__(self, db_manager, dex_collector, *, dry_run: bool,
                 refresh_interval: int = 30, close_poll_interval: int = 15):
        self.db = db_manager
        self.collector = dex_collector
        self.dry_run = dry_run
        self.refresh_interval = refresh_interval
        self.close_poll_interval = close_poll_interval
        self._stop = asyncio.Event()
        # token_address -> consecutive price-fetch failure count
        self._fail_counts: Dict[str, int] = {}

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

    async def _refresh_one(self, pos: Dict) -> None:
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

    async def price_refresh_loop(self) -> None:
        logger.info("📊 DEX DB-first price-refresh loop started")
        while not self._stop.is_set():
            try:
                positions = await self._fetch_open_positions()
                for pos in positions:
                    if self._stop.is_set():
                        break
                    await self._refresh_one(pos)
                if positions:
                    logger.debug(f"📊 Refreshed {len(positions)} open DEX positions")
            except Exception as e:
                logger.error(f"price-refresh loop error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.refresh_interval)
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

        pnl = (exit_price - entry_price) * amount
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        simulated = self.dry_run
        metadata['close_reason'] = 'manual_dashboard'
        metadata['closed_simulated'] = simulated

        await self._update_trade(pid, {
            'exit_price': float(exit_price),
            'exit_timestamp': datetime.now(timezone.utc),
            'profit_loss': float(pnl),
            'profit_loss_percentage': float(pnl_pct),
            'status': 'closed',
            'metadata': metadata,
        })
        logger.info(
            f"{'📝 DRY-RUN ' if simulated else '💰 '}DEX position {pid} closed "
            f"(exit=${exit_price:.8f}, pnl=${pnl:.2f} / {pnl_pct:+.2f}%)"
        )
        return {'success': True, 'simulated': simulated, 'exit_price': exit_price,
                'profit_loss': pnl}

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
