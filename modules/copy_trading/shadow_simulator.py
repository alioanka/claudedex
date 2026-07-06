"""
COPY v3 shadow-copy simulator — paper-copies candidate/leader wallets to
measure their COPY-performance (not their raw wallet performance) before any
live promotion.

What it models
--------------
For every observed leader swap event we book a PAPER fill at the leader's
at-event price adjusted for our modeled execution cost:

    buy  fill = leader_price * (1 + (slippage_bps + fee_bps)/10000)
    sell fill = leader_price * (1 - (slippage_bps + fee_bps)/10000)

BUYs open/add a fixed-notional paper position (``sim_notional_usd``, add cap
``max_adds_per_position``). SELLs close the fraction of OUR position equal to
the fraction of the LEADER's tracked open qty being sold (we mirror exits
proportionally). Realized PnL is booked on sells; unrealized is marked at the
last observed event price (honest limitation: between events the mark is
stale).

Persistence (migration 135; every row is_simulated=TRUE):
    copy_shadow_fills      — every paper fill, idempotent on
                             (chain, wallet, token, side, source_ref)
    copy_shadow_positions  — current open paper positions
    copy_shadow_equity     — per-wallet equity snapshots after each tick
    copy_shadow_cursors    — per-wallet ingest cursor (no reprocess/skip)

Event sources (read-only):
    * smart_money_wallet_events (mig 133) — EVM swaps with at-event USD price.
    * copytrading detected swaps would need an engine hook; v3 deliberately
      avoids touching the hot path — target wallets are usually also visible
      via smart_money events on EVM. Solana candidate coverage is provided by
      discovery_v3's Helius enrichment writing events through `record_events`.

No-look-ahead: events are processed strictly in (event_time, id) order from a
persisted cursor; an event with event_time > now is refused. The pure core
(``ShadowBook``) is self-tested offline below.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("CopyShadowSim")

DUST_QTY = 1e-9


@dataclass
class ShadowConfig:
    sim_notional_usd: float = 100.0
    fee_bps: float = 30.0
    slippage_bps: float = 50.0
    max_adds_per_position: int = 3
    max_wallets: int = 40


@dataclass
class ShadowPosition:
    token: str
    qty: float = 0.0            # OUR paper qty
    cost_usd: float = 0.0       # OUR remaining cost basis
    leader_qty: float = 0.0     # leader's tracked open qty
    last_price_usd: float = 0.0
    adds_count: int = 0
    opened_at: Optional[datetime] = None
    token_symbol: Optional[str] = None


@dataclass
class ShadowFill:
    token: str
    side: str
    leader_price_usd: float
    fill_price_usd: float
    qty: float
    notional_usd: float
    fee_usd: float
    realized_pnl_usd: Optional[float]
    source_ref: str
    event_source: str
    event_time: datetime
    token_symbol: Optional[str] = None


@dataclass
class ShadowBook:
    """Pure per-(chain, leader-wallet) paper book. No I/O."""
    cfg: ShadowConfig
    positions: Dict[str, ShadowPosition] = field(default_factory=dict)
    realized_pnl_usd: float = 0.0
    fills_count: int = 0
    _last_event_time: Optional[datetime] = None

    def _cost_rate(self) -> float:
        return (self.cfg.fee_bps + self.cfg.slippage_bps) / 10000.0

    def apply_event(self, event: Dict, *, now: Optional[datetime] = None) -> Optional[ShadowFill]:
        """Apply one leader swap event. Returns the paper fill, or None when
        the event produces no fill (add-cap hit, sell with no position, dup
        ordering violation, malformed).

        Event shape: {token, side('buy'|'sell'), qty, price_usd, ts,
                      source_ref, event_source?, token_symbol?}
        """
        ts = event.get("ts")
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if not isinstance(ts, datetime):
            return None
        now = now or datetime.now(timezone.utc)
        if ts > now:
            # No-look-ahead: refuse to book a fill for a future event.
            return None
        if self._last_event_time and ts < self._last_event_time:
            # Out-of-order replay — refuse (cursor discipline upstream).
            return None
        side = event.get("side")
        try:
            leader_qty = float(event.get("qty") or 0.0)
            price = float(event.get("price_usd") or 0.0)
        except (TypeError, ValueError):
            return None
        if side not in ("buy", "sell") or leader_qty <= 0 or price <= 0:
            return None
        token = str(event.get("token") or "?")
        source_ref = str(event.get("source_ref") or "")
        event_source = str(event.get("event_source") or "smart_money")
        symbol = event.get("token_symbol")
        cost_rate = self._cost_rate()
        self._last_event_time = ts

        pos = self.positions.get(token)
        if side == "buy":
            if pos is not None and pos.adds_count >= self.cfg.max_adds_per_position:
                pos.leader_qty += leader_qty
                pos.last_price_usd = price
                return None
            fill_price = price * (1.0 + cost_rate)
            notional = self.cfg.sim_notional_usd
            qty = notional / fill_price
            fee = notional * (self.cfg.fee_bps / 10000.0)
            if pos is None:
                pos = ShadowPosition(token=token, opened_at=ts, token_symbol=symbol)
                self.positions[token] = pos
            pos.qty += qty
            pos.cost_usd += notional
            pos.leader_qty += leader_qty
            pos.last_price_usd = price
            pos.adds_count += 1
            self.fills_count += 1
            return ShadowFill(
                token=token, side="buy", leader_price_usd=price,
                fill_price_usd=fill_price, qty=qty, notional_usd=notional,
                fee_usd=fee, realized_pnl_usd=None, source_ref=source_ref,
                event_source=event_source, event_time=ts, token_symbol=symbol,
            )

        # sell
        if pos is None or pos.qty <= DUST_QTY:
            return None
        frac = min(1.0, leader_qty / pos.leader_qty) if pos.leader_qty > DUST_QTY else 1.0
        close_qty = pos.qty * frac
        if close_qty <= DUST_QTY:
            return None
        fill_price = price * (1.0 - cost_rate)
        proceeds = close_qty * fill_price
        basis = pos.cost_usd * (close_qty / pos.qty)
        realized = proceeds - basis
        fee = proceeds * (self.cfg.fee_bps / 10000.0)
        pos.qty -= close_qty
        pos.cost_usd -= basis
        pos.leader_qty = max(0.0, pos.leader_qty - leader_qty)
        pos.last_price_usd = price
        self.realized_pnl_usd += realized
        self.fills_count += 1
        if pos.qty <= DUST_QTY:
            del self.positions[token]
        return ShadowFill(
            token=token, side="sell", leader_price_usd=price,
            fill_price_usd=fill_price, qty=close_qty, notional_usd=proceeds,
            fee_usd=fee, realized_pnl_usd=realized, source_ref=source_ref,
            event_source=event_source, event_time=ts, token_symbol=symbol,
        )

    def unrealized_pnl_usd(self) -> float:
        total = 0.0
        for p in self.positions.values():
            if p.qty > DUST_QTY and p.last_price_usd > 0:
                total += p.qty * p.last_price_usd - p.cost_usd
        return total

    def equity_usd(self) -> float:
        return self.realized_pnl_usd + self.unrealized_pnl_usd()


# -------------------------------------------------------------------------
# DB runner (asyncpg pool; fail-soft, read-only on source tables)
# -------------------------------------------------------------------------

async def _load_book(db_pool, chain: str, wallet: str, cfg: ShadowConfig) -> ShadowBook:
    book = ShadowBook(cfg=cfg)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT token, token_symbol, qty, cost_usd, leader_qty, "
            "       last_price_usd, adds_count, opened_at "
            "FROM copy_shadow_positions "
            "WHERE chain = $1 AND wallet_address = $2 AND qty > 0",
            chain, wallet,
        )
        agg = await conn.fetchrow(
            "SELECT COALESCE(SUM(realized_pnl_usd), 0) AS r, COUNT(*) AS n "
            "FROM copy_shadow_fills WHERE chain = $1 AND wallet_address = $2",
            chain, wallet,
        )
    for r in rows:
        book.positions[r["token"]] = ShadowPosition(
            token=r["token"], token_symbol=r["token_symbol"],
            qty=float(r["qty"]), cost_usd=float(r["cost_usd"]),
            leader_qty=float(r["leader_qty"]),
            last_price_usd=float(r["last_price_usd"] or 0),
            adds_count=int(r["adds_count"]), opened_at=r["opened_at"],
        )
    if agg:
        book.realized_pnl_usd = float(agg["r"] or 0)
        book.fills_count = int(agg["n"] or 0)
    return book


async def _persist(db_pool, chain: str, wallet: str, book: ShadowBook,
                   fills: Sequence[ShadowFill], cursor_time, cursor_id: int) -> None:
    async with db_pool.acquire() as conn:
        for f in fills:
            await conn.execute(
                """
                INSERT INTO copy_shadow_fills (
                    chain, wallet_address, token, token_symbol, side,
                    leader_price_usd, fill_price_usd, qty, notional_usd,
                    fee_usd, realized_pnl_usd, source_ref, event_source,
                    event_time, is_simulated
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,TRUE)
                ON CONFLICT (chain, wallet_address, token, side, source_ref)
                DO NOTHING
                """,
                chain, wallet, f.token, f.token_symbol, f.side,
                f.leader_price_usd, f.fill_price_usd, f.qty, f.notional_usd,
                f.fee_usd, f.realized_pnl_usd, f.source_ref, f.event_source,
                f.event_time,
            )
        # Positions: replace this wallet's rows with the book's state.
        await conn.execute(
            "DELETE FROM copy_shadow_positions WHERE chain=$1 AND wallet_address=$2",
            chain, wallet,
        )
        for p in book.positions.values():
            await conn.execute(
                """
                INSERT INTO copy_shadow_positions (
                    chain, wallet_address, token, token_symbol, qty, cost_usd,
                    leader_qty, last_price_usd, adds_count, opened_at,
                    updated_at, is_simulated
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,NOW(),TRUE)
                ON CONFLICT (chain, wallet_address, token) DO UPDATE SET
                    qty=EXCLUDED.qty, cost_usd=EXCLUDED.cost_usd,
                    leader_qty=EXCLUDED.leader_qty,
                    last_price_usd=EXCLUDED.last_price_usd,
                    adds_count=EXCLUDED.adds_count, updated_at=NOW()
                """,
                chain, wallet, p.token, p.token_symbol, p.qty, p.cost_usd,
                p.leader_qty, p.last_price_usd, p.adds_count,
                p.opened_at or datetime.now(timezone.utc),
            )
        unreal = book.unrealized_pnl_usd()
        await conn.execute(
            """
            INSERT INTO copy_shadow_equity (
                chain, wallet_address, realized_pnl_usd, unrealized_pnl_usd,
                equity_usd, open_positions, fills_count, is_simulated
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE)
            """,
            chain, wallet, book.realized_pnl_usd, unreal,
            book.realized_pnl_usd + unreal, len(book.positions),
            book.fills_count,
        )
        await conn.execute(
            """
            INSERT INTO copy_shadow_cursors (chain, wallet_address,
                last_event_time, last_event_id, updated_at)
            VALUES ($1,$2,$3,$4,NOW())
            ON CONFLICT (chain, wallet_address) DO UPDATE SET
                last_event_time=EXCLUDED.last_event_time,
                last_event_id=EXCLUDED.last_event_id, updated_at=NOW()
            """,
            chain, wallet, cursor_time, cursor_id,
        )


async def record_events(db_pool, chain: str, wallet: str,
                        events: Sequence[Dict], cfg: ShadowConfig) -> int:
    """Feed externally-sourced events (e.g. discovery_v3 Helius enrichment)
    through the wallet's paper book. Events must carry a stable source_ref
    (tx hash) — the fills table dedupes on it, so re-feeding is safe.
    Returns the number of fills booked."""
    if db_pool is None or not events:
        return 0
    try:
        book = await _load_book(db_pool, chain, wallet, cfg)
        async with db_pool.acquire() as conn:
            cur = await conn.fetchrow(
                "SELECT last_event_time, last_event_id FROM copy_shadow_cursors "
                "WHERE chain=$1 AND wallet_address=$2", chain, wallet)
        last_t = cur["last_event_time"] if cur else None
        ordered = sorted(
            (e for e in events if isinstance(e.get("ts"), datetime)),
            key=lambda e: e["ts"],
        )
        fills: List[ShadowFill] = []
        now = datetime.now(timezone.utc)
        max_t = last_t
        for e in ordered:
            ts = e["ts"] if e["ts"].tzinfo else e["ts"].replace(tzinfo=timezone.utc)
            if last_t and ts <= last_t:
                continue  # already ingested (cursor); fills table dedupes too
            f = book.apply_event({**e, "ts": ts}, now=now)
            if f:
                fills.append(f)
            if max_t is None or ts > max_t:
                max_t = ts
        if fills or max_t != last_t:
            await _persist(db_pool, chain, wallet, book, fills, max_t,
                           int(cur["last_event_id"]) if cur else 0)
        return len(fills)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"record_events({wallet[:8]}): {e}")
        return 0


async def _fetch_solana_prices(mints: Sequence[str]) -> Dict[str, float]:
    """Best-effort Jupiter Price v3 lookup (free, no key) for Solana mints.
    Returns {mint: usd_price}; empty dict on any failure (caller keeps the
    last observed mark). Bounded to 100 mints per call."""
    mints = [m for m in dict.fromkeys(mints) if m]
    if not mints:
        return {}
    try:
        import aiohttp  # local import: module stays importable without aiohttp
    except ImportError:
        return {}
    url = "https://lite-api.jup.ag/price/v3?ids=" + ",".join(mints[:100])
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json(content_type=None)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"jupiter price fetch failed: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    # v3 shape: {mint: {usdPrice: x}}; some deployments nest under 'data'.
    root = data.get("data") if isinstance(data.get("data"), dict) else data
    out: Dict[str, float] = {}
    for mint, rec in (root or {}).items():
        if not isinstance(rec, dict):
            continue
        try:
            px = float(rec.get("usdPrice") or rec.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if px > 0:
            out[mint] = px
    return out


async def mark_to_market(db_pool, cfg: ShadowConfig) -> int:
    """AI-Trader port: continuously re-mark OPEN shadow positions and write a
    fresh copy_shadow_equity snapshot per wallet, so equity curves are honest
    BETWEEN leader events (previously the mark was stale until the next event).
    Solana marks are refreshed via Jupiter (free); other chains keep the last
    observed price (documented limitation). Read-only on fills; fail-soft.
    Returns the number of wallets re-marked."""
    if db_pool is None:
        return 0
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chain, wallet_address, token, qty, cost_usd, last_price_usd "
                "FROM copy_shadow_positions WHERE qty > 0")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"mark_to_market load failed: {e}")
        return 0
    if not rows:
        return 0
    sol_mints = [r["token"] for r in rows if r["chain"] == "solana"]
    prices = await _fetch_solana_prices(sol_mints)
    by_wallet: Dict[tuple, List] = {}
    for r in rows:
        by_wallet.setdefault((r["chain"], r["wallet_address"]), []).append(r)
    marked = 0
    for (chain, wallet), plist in by_wallet.items():
        try:
            unreal = 0.0
            async with db_pool.acquire() as conn:
                for r in plist:
                    px = prices.get(r["token"])
                    if px and px > 0:
                        await conn.execute(
                            "UPDATE copy_shadow_positions SET last_price_usd=$1, "
                            "updated_at=NOW() WHERE chain=$2 AND wallet_address=$3 "
                            "AND token=$4", px, chain, wallet, r["token"])
                    else:
                        px = float(r["last_price_usd"] or 0)
                    if px > 0:
                        unreal += float(r["qty"]) * px - float(r["cost_usd"])
                agg = await conn.fetchrow(
                    "SELECT COALESCE(SUM(realized_pnl_usd),0) AS r, COUNT(*) AS n "
                    "FROM copy_shadow_fills WHERE chain=$1 AND wallet_address=$2 "
                    "AND is_simulated", chain, wallet)
                realized = float(agg["r"] or 0) if agg else 0.0
                fills_n = int(agg["n"] or 0) if agg else 0
                await conn.execute(
                    """
                    INSERT INTO copy_shadow_equity (
                        chain, wallet_address, realized_pnl_usd, unrealized_pnl_usd,
                        equity_usd, open_positions, fills_count, is_simulated
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE)
                    """,
                    chain, wallet, realized, unreal, realized + unreal,
                    len(plist), fills_n)
            marked += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"mark_to_market {wallet[:8]}: {e}")
    if marked:
        logger.info(f"[shadow-sim] marked-to-market {marked} wallet(s) with open positions")
    return marked


async def _shadow_watchlist(db_pool, max_wallets: int) -> List[Dict]:
    """Wallets worth shadow-copying: top discovered + pending candidates.
    Active targets are EVM/Solana leader wallets already mirrored DRY/LIVE;
    they're included via copy_discovered_wallets (the onchain source seeds
    them) rather than re-parsed here."""
    rows: List[Dict] = []
    async with db_pool.acquire() as conn:
        r1 = await conn.fetch(
            "SELECT chain, wallet_address FROM copy_leader_candidates "
            "WHERE status = 'pending' ORDER BY score DESC NULLS LAST LIMIT $1",
            max_wallets,
        )
        r2 = await conn.fetch(
            "SELECT chain, wallet_address FROM copy_discovered_wallets "
            "WHERE status IN ('discovered', 'proposed') "
            "ORDER BY score DESC NULLS LAST LIMIT $1",
            max_wallets,
        )
    seen = set()
    for r in list(r1) + list(r2):
        key = (r["chain"], r["wallet_address"])
        if key not in seen:
            seen.add(key)
            rows.append({"chain": r["chain"], "wallet_address": r["wallet_address"]})
    return rows[:max_wallets]


async def tick(db_pool, cfg: ShadowConfig) -> Dict:
    """One simulator pass: for every watchlist wallet, ingest NEW
    smart_money_wallet_events past the cursor and book paper fills.
    Read-only on smart_money tables; fail-soft per wallet."""
    if db_pool is None:
        return {"wallets": 0, "fills": 0}
    total_fills = 0
    wallets = 0
    try:
        watch = await _shadow_watchlist(db_pool, cfg.max_wallets)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"shadow watchlist failed: {e}")
        return {"wallets": 0, "fills": 0}
    for w in watch:
        chain, wallet = w["chain"], w["wallet_address"]
        try:
            async with db_pool.acquire() as conn:
                cur = await conn.fetchrow(
                    "SELECT last_event_time, last_event_id FROM copy_shadow_cursors "
                    "WHERE chain=$1 AND wallet_address=$2", chain, wallet)
                last_id = int(cur["last_event_id"]) if cur else 0
                evs = await conn.fetch(
                    """
                    SELECT id, token, token_symbol, side, amount_usd,
                           price_usd_at_event, tx_hash, log_index, block_time
                    FROM smart_money_wallet_events
                    WHERE chain = $1 AND wallet = $2 AND id > $3
                      AND block_time <= NOW()
                    ORDER BY block_time ASC, id ASC
                    LIMIT 500
                    """,
                    chain, wallet, last_id,
                )
            if not evs:
                continue
            book = await _load_book(db_pool, chain, wallet, cfg)
            fills: List[ShadowFill] = []
            now = datetime.now(timezone.utc)
            max_t = cur["last_event_time"] if cur else None
            max_id = last_id
            for r in evs:
                price = float(r["price_usd_at_event"] or 0)
                usd = float(r["amount_usd"] or 0)
                if price <= 0 or usd <= 0:
                    max_id = max(max_id, int(r["id"]))
                    continue
                f = book.apply_event({
                    "token": r["token"], "token_symbol": r["token_symbol"],
                    "side": r["side"], "qty": usd / price, "price_usd": price,
                    "ts": r["block_time"],
                    "source_ref": f"{r['tx_hash']}:{r['log_index']}",
                    "event_source": "smart_money",
                }, now=now)
                if f:
                    fills.append(f)
                max_id = max(max_id, int(r["id"]))
                if max_t is None or r["block_time"] > max_t:
                    max_t = r["block_time"]
            await _persist(db_pool, chain, wallet, book, fills, max_t, max_id)
            total_fills += len(fills)
            wallets += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"shadow tick skipped {wallet[:8]}: {e}")
    if total_fills:
        logger.info(f"[shadow-sim] booked {total_fills} paper fills across {wallets} wallets")
    # AI-Trader port: re-mark open positions each cycle so equity curves stay
    # honest between leader events (fail-soft, does not affect fills).
    marked = 0
    try:
        marked = await mark_to_market(db_pool, cfg)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"mark_to_market pass failed: {e}")
    return {"wallets": wallets, "fills": total_fills, "marked": marked}


# -------------------------------------------------------------------------
# Self-test (pure core; offline). Run:
#   python -m modules.copy_trading.shadow_simulator
# -------------------------------------------------------------------------
def _self_test() -> None:
    now = datetime(2026, 6, 13, tzinfo=timezone.utc)
    cfg = ShadowConfig(sim_notional_usd=100.0, fee_bps=30.0, slippage_bps=50.0,
                       max_adds_per_position=2)
    book = ShadowBook(cfg=cfg)
    cost = (30.0 + 50.0) / 10000.0  # 0.8% per side

    # 1. BUY books fixed notional at the slipped price.
    f1 = book.apply_event({"token": "T", "side": "buy", "qty": 1000,
                           "price_usd": 1.0, "ts": now - timedelta(hours=3),
                           "source_ref": "tx1"}, now=now)
    assert f1 and abs(f1.fill_price_usd - 1.0 * (1 + cost)) < 1e-12
    assert abs(f1.qty - 100.0 / (1 + cost)) < 1e-9
    assert abs(book.positions["T"].cost_usd - 100.0) < 1e-9

    # 2. Leader sells HALF -> we close half, realized = round-trip cost drag.
    f2 = book.apply_event({"token": "T", "side": "sell", "qty": 500,
                           "price_usd": 1.0, "ts": now - timedelta(hours=2),
                           "source_ref": "tx2"}, now=now)
    assert f2 and f2.side == "sell"
    expected_realized = 50.0 * ((1 - cost) / (1 + cost) - 1.0)  # flat price, cost both sides
    assert abs(f2.realized_pnl_usd - expected_realized) < 1e-9, (f2.realized_pnl_usd, expected_realized)
    assert abs(book.positions["T"].qty - f1.qty / 2) < 1e-9

    # 3. Leader exits the rest at 2x -> profit despite costs; position closed.
    f3 = book.apply_event({"token": "T", "side": "sell", "qty": 500,
                           "price_usd": 2.0, "ts": now - timedelta(hours=1),
                           "source_ref": "tx3"}, now=now)
    assert f3 and f3.realized_pnl_usd > 40.0
    assert "T" not in book.positions
    assert abs(book.equity_usd() - book.realized_pnl_usd) < 1e-9

    # 4. NO LOOK-AHEAD: future event refused.
    assert book.apply_event({"token": "U", "side": "buy", "qty": 1,
                             "price_usd": 1.0, "ts": now + timedelta(seconds=1),
                             "source_ref": "txF"}, now=now) is None

    # 5. Out-of-order replay refused (cursor discipline).
    assert book.apply_event({"token": "U", "side": "buy", "qty": 1,
                             "price_usd": 1.0, "ts": now - timedelta(days=1),
                             "source_ref": "txOld"}, now=now) is None

    # 6. SELL with no position is a no-op (no fabricated basis).
    assert book.apply_event({"token": "NOPE", "side": "sell", "qty": 10,
                             "price_usd": 1.0, "ts": now - timedelta(minutes=30),
                             "source_ref": "tx4"}, now=now) is None

    # 7. Add cap: third buy tracks leader qty but books no fill.
    b2 = ShadowBook(cfg=cfg)
    t0 = now - timedelta(hours=5)
    for i in range(3):
        r = b2.apply_event({"token": "A", "side": "buy", "qty": 100,
                            "price_usd": 1.0, "ts": t0 + timedelta(minutes=i),
                            "source_ref": f"a{i}"}, now=now)
        assert (r is not None) == (i < 2)
    assert abs(b2.positions["A"].leader_qty - 300.0) < 1e-9
    assert abs(b2.positions["A"].cost_usd - 200.0) < 1e-9

    # 8. Unrealized marks at last seen price.
    b2.apply_event({"token": "A", "side": "sell", "qty": 1,  # tiny leader trim
                    "price_usd": 3.0, "ts": t0 + timedelta(minutes=10),
                    "source_ref": "a3"}, now=now)
    assert b2.unrealized_pnl_usd() > 0  # marked at 3.0 vs ~1.008 basis

    print("shadow_simulator self-test OK "
          f"(rt1={f2.realized_pnl_usd:.4f} rt2={f3.realized_pnl_usd:.2f} "
          f"equity={book.equity_usd():.2f})")


if __name__ == "__main__":
    _self_test()
