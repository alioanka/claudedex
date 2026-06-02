"""
AdvisorPortfolioEngine — operator-reported holdings tracker + sim lifecycle.

Stores and queries the operator's declared portfolio (what they hold on
Midas or elsewhere). This is READ/WRITE but advice-only: no execution.
The operator updates holdings via the advisor dashboard; this engine
reads them to provide contextual advice ("you already hold 20% in AAPL,
adding more increases concentration risk").

Sim position tracking (advisor_sim_positions) also lives here:
  - open_sim_position(advice_result)           -> sim_id (int)
  - mark_to_market(sim_id, price)              -> updated SimPosition
  - close_sim_position(sim_id, price, reason)  -> final SimPosition with pnl
  - auto_close_expired(analyzer_fn)            -> count of auto-closed sims
  - list_open_sims()                           -> List[SimPosition]
  - backtest(symbol, horizon, direction, ...)  -> BacktestResult

Wave-21 additions
  - horizon_end_date stored on open (from horizon_days config key).
  - auto_close_expired(): closes sims where horizon end has passed or
    target/stop hit; called by AdviceEngine._mark_to_market_open_sims().
  - backtest(): replay historical prices via yfinance for a past advice and
    compute what PnL would have been. Free data, no exchange key required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from modules.advisor.core.models import (
    AdviceResult,
    Direction,
    Horizon,
    Market,
    PortfolioHolding,
    SimPosition,
)

logger = logging.getLogger("advisor.portfolio_engine")

# ---------------------------------------------------------------------------
# Horizon-to-days map (used when opening a sim to set expected close date)
# ---------------------------------------------------------------------------
_HORIZON_DAYS: Dict[str, int] = {
    Horizon.SHORT.value: 7,     # short  = up to 1 week
    Horizon.MID.value:   90,    # mid    = up to 3 months
    Horizon.LONG.value:  365,   # long   = up to 1 year (conservative end)
}


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Outcome of replaying a historical advice against recorded price data.

    All monetary figures are in USD.  pnl_pct and pnl_usd reflect the
    full notional from entry to final exit price.
    """
    symbol: str
    direction: str
    horizon: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    target_price: Optional[float]
    stop_price: Optional[float]
    notional_usd: float
    pnl_pct: float
    pnl_usd: float
    close_reason: str              # target_hit | stop_hit | expired
    price_series_len: int          # how many price points were replayed
    data_source: str               # yfinance | ccxt | mock
    error: Optional[str] = None    # non-None if data fetch failed


class AdvisorPortfolioEngine:
    """
    Manages operator holdings + sim position lifecycle.

    All persistence is via asyncpg (advisor_portfolio, advisor_sim_positions
    tables defined in migration 058). Pass db_pool=None for unit-test mode
    (in-memory only).
    """

    def __init__(self, db_pool=None, config: Optional[dict] = None):
        self.db_pool = db_pool
        self.config = config or {}
        self._sim_cache: dict[int, SimPosition] = {}  # in-memory mirror

    # ------------------------------------------------------------------
    # Sim position management
    # ------------------------------------------------------------------

    def _horizon_days(self, horizon: Horizon) -> int:
        """Return the operator-configured max holding days for a horizon."""
        key = f"sim_horizon_days_{horizon.value}"
        return int(self.config.get(key, _HORIZON_DAYS.get(horizon.value, 90)))

    async def open_sim_position(self, result: AdviceResult) -> Optional[int]:
        """
        Seed a new sim position from an AdviceResult.
        Returns the DB-generated sim_id, or None on failure.

        The horizon_end_date is computed from the advice horizon and stored
        so that auto_close_expired() can close positions at deadline.
        """
        if result.entry_low is None and result.entry_high is None:
            logger.warning(
                "[portfolio] Cannot open sim for %s: "
                "no entry price range in AdviceResult",
                result.symbol,
            )
            return None

        entry_price = (
            ((result.entry_low or 0) + (result.entry_high or 0)) / 2
            if result.entry_low and result.entry_high
            else result.entry_low or result.entry_high
        )

        days = self._horizon_days(result.horizon)
        horizon_end = datetime.now(timezone.utc) + timedelta(days=days)

        sim = SimPosition(
            symbol=result.symbol,
            market=result.market,
            direction=result.direction,
            horizon=result.horizon,
            entry_price=entry_price,
            target_price=result.target_price,
            stop_price=result.stop_price,
            notional_usd=result.sim_amount_usd,
            advice_id=result.extra.get("advice_id"),
        )

        if self.db_pool is None:
            # In-memory fallback (test mode)
            sim_id = len(self._sim_cache) + 1
            self._sim_cache[sim_id] = sim
            return sim_id

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO advisor_sim_positions
                      (advice_id, symbol, market, direction, horizon,
                       entry_price, target_price, stop_price,
                       notional_usd, status, opened_at, horizon_end_date)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'open',NOW(),$10)
                    RETURNING id
                    """,
                    sim.advice_id,
                    sim.symbol,
                    sim.market.value,
                    sim.direction.value,
                    sim.horizon.value,
                    sim.entry_price,
                    sim.target_price,
                    sim.stop_price,
                    sim.notional_usd,
                    horizon_end,
                )
                sim_id = row["id"]
                self._sim_cache[sim_id] = sim
                logger.info(
                    "[portfolio] Opened sim #%d: %s %s @ %.4f notional=$%.2f "
                    "horizon_end=%s",
                    sim_id, sim.direction.value, sim.symbol,
                    sim.entry_price, sim.notional_usd,
                    horizon_end.date().isoformat(),
                )
                return sim_id
        except Exception as exc:
            logger.error("[portfolio] Failed to open sim for %s: %s", result.symbol, exc)
            return None

    async def mark_to_market(self, sim_id: int, current_price: float) -> Optional[SimPosition]:
        """
        Update a sim position with the current market price.
        Calculates unrealised PnL and persists to DB.
        Does NOT auto-close — call auto_close_expired() for that.
        """
        sim = self._sim_cache.get(sim_id)
        if sim is None:
            if self.db_pool:
                sim = await self._load_sim_from_db(sim_id)
            if sim is None:
                return None

        if sim.direction == Direction.LONG:
            pnl_pct = (current_price - sim.entry_price) / sim.entry_price * 100
        elif sim.direction == Direction.SHORT:
            pnl_pct = (sim.entry_price - current_price) / sim.entry_price * 100
        else:
            pnl_pct = 0.0

        pnl_usd = sim.notional_usd * (pnl_pct / 100)

        sim.pnl_pct = pnl_pct
        sim.pnl_usd = pnl_usd
        sim.exit_price = current_price  # current price, not final exit

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE advisor_sim_positions
                        SET current_price=$1, pnl_pct=$2, pnl_usd=$3,
                            updated_at=NOW()
                        WHERE id=$4
                        """,
                        current_price, pnl_pct, pnl_usd, sim_id,
                    )
            except Exception as exc:
                logger.warning(
                    "[portfolio] mark_to_market DB error sim#%d: %s", sim_id, exc
                )

        return sim

    async def close_sim_position(
        self, sim_id: int, exit_price: float, reason: str = "manual"
    ) -> Optional[SimPosition]:
        """
        Close a sim position at exit_price. Calculates final PnL.

        close_reason values: "manual" | "target_hit" | "stop_hit" |
                             "expired" | "operator"
        """
        sim = self._sim_cache.get(sim_id)
        if sim is None:
            if self.db_pool:
                sim = await self._load_sim_from_db(sim_id)
            if sim is None:
                return None

        if sim.direction == Direction.LONG:
            pnl_pct = (exit_price - sim.entry_price) / sim.entry_price * 100
        elif sim.direction == Direction.SHORT:
            pnl_pct = (sim.entry_price - exit_price) / sim.entry_price * 100
        else:
            pnl_pct = 0.0

        pnl_usd = sim.notional_usd * (pnl_pct / 100)

        sim.exit_price = exit_price
        sim.pnl_pct = pnl_pct
        sim.pnl_usd = pnl_usd
        sim.status = "closed"
        sim.closed_at = datetime.now(timezone.utc)

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE advisor_sim_positions
                        SET exit_price=$1, pnl_pct=$2, pnl_usd=$3,
                            status='closed', closed_at=NOW(),
                            close_reason=$4, updated_at=NOW()
                        WHERE id=$5
                        """,
                        exit_price, pnl_pct, pnl_usd, reason, sim_id,
                    )
            except Exception as exc:
                logger.warning(
                    "[portfolio] close_sim DB error sim#%d: %s", sim_id, exc
                )

        logger.info(
            "[portfolio] Closed sim #%d: %s pnl=%+.2f%% ($%+.2f) reason=%s",
            sim_id, sim.symbol, pnl_pct, pnl_usd, reason,
        )
        return sim

    async def auto_close_expired(
        self,
        price_fetcher: Optional[Callable] = None,
    ) -> int:
        """
        Scan all open sim positions and auto-close those where:
          - horizon_end_date has passed  (close_reason='expired')
          - target_price hit             (close_reason='target_hit')
          - stop_price hit               (close_reason='stop_hit')

        Parameters
        ----------
        price_fetcher : async callable(symbol, market) -> Optional[float]
            Used to get the current price for target/stop evaluation.
            If None, only expired-by-date positions are closed.

        Returns
        -------
        int : number of positions auto-closed this call.
        """
        closed_count = 0
        now = datetime.now(timezone.utc)

        # Load open sims from DB (authoritative) if pool is available.
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, symbol, market, direction, entry_price,
                               target_price, stop_price, notional_usd,
                               horizon_end_date, current_price
                        FROM advisor_sim_positions
                        WHERE status='open'
                        """
                    )
            except Exception as exc:
                logger.error("[portfolio] auto_close_expired DB read error: %s", exc)
                return 0
        else:
            rows = [
                _sim_to_row(sid, s)
                for sid, s in self._sim_cache.items()
                if s.status == "open"
            ]

        for row in rows:
            sim_id = row["id"]
            symbol = row["symbol"]
            direction = row["direction"]
            entry_price = float(row["entry_price"])
            target = float(row["target_price"]) if row["target_price"] else None
            stop = float(row["stop_price"]) if row["stop_price"] else None
            horizon_end = row.get("horizon_end_date")

            # Fetch current price if we have a fetcher.
            current_price: Optional[float] = None
            if price_fetcher is not None:
                try:
                    current_price = await price_fetcher(symbol, row["market"])
                except Exception as exc:
                    logger.debug(
                        "[portfolio] price_fetcher failed for %s: %s", symbol, exc
                    )

            if current_price is None:
                # Fall back to last stored current_price from DB.
                current_price = (
                    float(row["current_price"]) if row.get("current_price") else None
                )

            close_reason: Optional[str] = None

            # Check target/stop (requires a current price).
            if current_price is not None:
                if direction == Direction.LONG.value or direction == "long":
                    if target and current_price >= target:
                        close_reason = "target_hit"
                    elif stop and current_price <= stop:
                        close_reason = "stop_hit"
                elif direction == Direction.SHORT.value or direction == "short":
                    if target and current_price <= target:
                        close_reason = "target_hit"
                    elif stop and current_price >= stop:
                        close_reason = "stop_hit"

            # Check horizon expiry.
            if close_reason is None and horizon_end is not None:
                if isinstance(horizon_end, datetime):
                    he = horizon_end if horizon_end.tzinfo else horizon_end.replace(tzinfo=timezone.utc)
                    if now >= he:
                        close_reason = "expired"

            if close_reason and current_price is not None:
                await self.close_sim_position(sim_id, current_price, close_reason)
                closed_count += 1
            elif close_reason == "expired" and current_price is None:
                # No price available; close at entry as zero-PnL fallback.
                await self.close_sim_position(sim_id, entry_price, "expired_no_price")
                closed_count += 1

        if closed_count:
            logger.info(
                "[portfolio] auto_close_expired: closed %d position(s).", closed_count
            )
        return closed_count

    async def list_open_sims(self) -> List[SimPosition]:
        """Return all currently open sim positions."""
        if self.db_pool is None:
            return [s for s in self._sim_cache.values() if s.status == "open"]

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM advisor_sim_positions WHERE status='open'"
                )
            return [_row_to_sim(r) for r in rows]
        except Exception as exc:
            logger.error("[portfolio] list_open_sims error: %s", exc)
            return []

    async def count_open_sims(self) -> int:
        sims = await self.list_open_sims()
        return len(sims)

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    async def backtest(
        self,
        symbol: str,
        market: Market,
        direction: Direction,
        horizon: Horizon,
        entry_date: datetime,
        entry_price: float,
        notional_usd: float = 1000.0,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> BacktestResult:
        """
        Replay historical prices to compute what an advice WOULD have returned.

        Given a past advice (symbol, direction, horizon, entry_date,
        entry_price) this method fetches historical OHLCV from yfinance
        (free, no key) and walks forward day-by-day to check if the target
        or stop would have been hit before the horizon expired.

        This is how the operator measures advice quality in DRY_RUN before
        trusting it with real Midas capital.  The dashboard can call this
        for any closed or historical sim position.

        Parameters
        ----------
        symbol       : Ticker in yfinance or ccxt format.
        market       : Market enum (determines data source format).
        direction    : LONG / SHORT.
        horizon      : Horizon for max holding period.
        entry_date   : UTC datetime when the advice was issued.
        entry_price  : Price at advice time.
        notional_usd : Simulated position size in USD.
        target_price : Optional take-profit level.
        stop_price   : Optional stop-loss level.

        Returns
        -------
        BacktestResult with pnl_pct, pnl_usd, close_reason, and metadata.
        Note: if yfinance data is unavailable, returns a result with
        error!=None and pnl=0 (fail-soft).
        """
        days = self._horizon_days(horizon)
        exit_date = entry_date + timedelta(days=days)
        now = datetime.now(timezone.utc)
        # Cannot backtest into the future.
        if exit_date > now:
            exit_date = now

        yf_symbol = _to_yfinance_symbol(symbol, market)
        price_series: List[tuple] = []  # (date, close)
        data_source = "yfinance"

        try:
            import yfinance as yf  # type: ignore[import]
            start = entry_date.strftime("%Y-%m-%d")
            end = (exit_date + timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(
                yf_symbol, start=start, end=end,
                progress=False, auto_adjust=True,
            )
            if df is not None and not df.empty and "Close" in df.columns:
                for ts, row in df["Close"].items():
                    price_series.append((ts.to_pydatetime(), float(row)))
        except ImportError:
            logger.warning("[portfolio] backtest: yfinance not installed; cannot replay")
            return BacktestResult(
                symbol=symbol, direction=direction.value, horizon=horizon.value,
                entry_date=entry_date, exit_date=exit_date,
                entry_price=entry_price, exit_price=entry_price,
                target_price=target_price, stop_price=stop_price,
                notional_usd=notional_usd,
                pnl_pct=0.0, pnl_usd=0.0, close_reason="no_data",
                price_series_len=0, data_source=data_source,
                error="yfinance not installed",
            )
        except Exception as exc:
            logger.warning("[portfolio] backtest fetch error for %s: %s", symbol, exc)
            return BacktestResult(
                symbol=symbol, direction=direction.value, horizon=horizon.value,
                entry_date=entry_date, exit_date=exit_date,
                entry_price=entry_price, exit_price=entry_price,
                target_price=target_price, stop_price=stop_price,
                notional_usd=notional_usd,
                pnl_pct=0.0, pnl_usd=0.0, close_reason="no_data",
                price_series_len=0, data_source=data_source,
                error=str(exc)[:200],
            )

        if not price_series:
            return BacktestResult(
                symbol=symbol, direction=direction.value, horizon=horizon.value,
                entry_date=entry_date, exit_date=exit_date,
                entry_price=entry_price, exit_price=entry_price,
                target_price=target_price, stop_price=stop_price,
                notional_usd=notional_usd,
                pnl_pct=0.0, pnl_usd=0.0, close_reason="no_data",
                price_series_len=0, data_source=data_source,
                error="No price data returned",
            )

        # Walk forward: check target/stop each day.
        final_price = price_series[-1][1]
        final_date = price_series[-1][0]
        close_reason = "expired"

        for ts, price in price_series:
            if direction == Direction.LONG:
                if target_price and price >= target_price:
                    final_price, final_date, close_reason = price, ts, "target_hit"
                    break
                if stop_price and price <= stop_price:
                    final_price, final_date, close_reason = price, ts, "stop_hit"
                    break
            elif direction == Direction.SHORT:
                if target_price and price <= target_price:
                    final_price, final_date, close_reason = price, ts, "target_hit"
                    break
                if stop_price and price >= stop_price:
                    final_price, final_date, close_reason = price, ts, "stop_hit"
                    break

        if direction == Direction.LONG:
            pnl_pct = (final_price - entry_price) / entry_price * 100
        elif direction == Direction.SHORT:
            pnl_pct = (entry_price - final_price) / entry_price * 100
        else:
            pnl_pct = 0.0

        pnl_usd = notional_usd * (pnl_pct / 100)

        return BacktestResult(
            symbol=symbol,
            direction=direction.value,
            horizon=horizon.value,
            entry_date=entry_date,
            exit_date=final_date if isinstance(final_date, datetime) else datetime.now(timezone.utc),
            entry_price=entry_price,
            exit_price=final_price,
            target_price=target_price,
            stop_price=stop_price,
            notional_usd=notional_usd,
            pnl_pct=round(pnl_pct, 4),
            pnl_usd=round(pnl_usd, 2),
            close_reason=close_reason,
            price_series_len=len(price_series),
            data_source=data_source,
        )

    # ------------------------------------------------------------------
    # Operator holdings
    # ------------------------------------------------------------------

    async def list_holdings(self) -> List[PortfolioHolding]:
        """Return all operator-reported portfolio holdings."""
        if self.db_pool is None:
            return []
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM advisor_portfolio")
            return [_row_to_holding(r) for r in rows]
        except Exception as exc:
            logger.error("[portfolio] list_holdings error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    async def _load_sim_from_db(self, sim_id: int) -> Optional[SimPosition]:
        if not self.db_pool:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM advisor_sim_positions WHERE id=$1", sim_id
                )
            if row:
                sim = _row_to_sim(row)
                self._sim_cache[sim_id] = sim
                return sim
        except Exception as exc:
            logger.warning("[portfolio] _load_sim_from_db(%d) error: %s", sim_id, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_sim(row) -> SimPosition:
    return SimPosition(
        symbol=row["symbol"],
        market=Market(row["market"]),
        direction=Direction(row["direction"]),
        horizon=Horizon(row["horizon"]),
        entry_price=float(row["entry_price"]),
        target_price=float(row["target_price"]) if row.get("target_price") else None,
        stop_price=float(row["stop_price"]) if row.get("stop_price") else None,
        notional_usd=float(row["notional_usd"]),
        advice_id=row.get("advice_id"),
        opened_at=row["opened_at"],
        closed_at=row.get("closed_at"),
        exit_price=float(row["exit_price"]) if row.get("exit_price") else None,
        pnl_pct=float(row["pnl_pct"]) if row.get("pnl_pct") else None,
        pnl_usd=float(row["pnl_usd"]) if row.get("pnl_usd") else None,
        status=row.get("status", "open"),
    )


def _row_to_holding(row) -> PortfolioHolding:
    return PortfolioHolding(
        symbol=row["symbol"],
        market=Market(row["market"]),
        quantity=float(row["quantity"]),
        avg_cost=float(row["avg_cost"]),
        current_price=float(row["current_price"]) if row.get("current_price") else None,
        updated_at=row["updated_at"],
    )


def _sim_to_row(sim_id: int, sim: SimPosition) -> dict:
    """Convert an in-memory SimPosition to a dict mimicking a DB row."""
    return {
        "id": sim_id,
        "symbol": sim.symbol,
        "market": sim.market.value,
        "direction": sim.direction.value,
        "entry_price": sim.entry_price,
        "target_price": sim.target_price,
        "stop_price": sim.stop_price,
        "notional_usd": sim.notional_usd,
        "horizon_end_date": sim.closed_at,   # None for in-memory test sims
        "current_price": sim.exit_price,
    }


def _to_yfinance_symbol(symbol: str, market: Market) -> str:
    """
    Map a market-native symbol to a yfinance-compatible ticker.

    - US equities: already yfinance format (AAPL, TSLA, ...).
    - BIST: append .IS if not already present (THYAO -> THYAO.IS).
    - FX/metals: already yfinance format (EURUSD=X, XAUUSD=X).
    - Crypto: strip the slash (BTC/USDT -> BTC-USD); approximate.
    - Midas funds: not supported by yfinance; return symbol as-is.
    """
    if market == Market.BIST:
        return symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    if market == Market.CRYPTO:
        # ccxt format BTC/USDT -> yfinance BTC-USD approximation
        base = symbol.split("/")[0] if "/" in symbol else symbol
        return f"{base}-USD"
    # US_EQUITIES, FX, MIDAS_FUNDS: return as-is
    return symbol
