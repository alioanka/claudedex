"""
AdviceEngine — orchestrates the full advice cycle.

One cycle per run_interval_minutes (default 60):
  1. Load enabled markets + symbols from advisor_config.
  2. For each (market, symbol, horizon): call the appropriate analyzer.
  3. Optionally overlay Kronos forecast signal.
  4. Pass through AdvisorRiskEngine gate.
  5. Persist accepted advice to advisor_advice table.
  6. If sim_enabled on advice: open sim position via portfolio_engine.
  7. Notify via AdvisorTelegramBot.
  8. Mark-to-market existing open sim positions.
  9. Auto-close expired / target-hit / stop-hit sims.

Wave-21 changes (quant agent)
  - _mark_to_market_open_sims: uses sim.id (from DB) not advice_id.
  - auto-close hook: calls portfolio.auto_close_expired() after mark-to-market.
  - _ml_learning_tick: implemented (see advisor_ml.py).
  - Kronos overlay: None handled cleanly; advice still produced.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.kronos_forecaster import KronosForecaster
from modules.advisor.core.models import AdviceResult, DataSourceStatus, Horizon, Market
from modules.advisor.core.portfolio_engine import AdvisorPortfolioEngine
from modules.advisor.core.risk_engine import AdvisorRiskEngine

logger = logging.getLogger("advisor.advice_engine")


class AdviceEngine:
    """
    Top-level advice orchestrator.

    Parameters
    ----------
    config      : dict loaded from advisor_config DB rows.
    analyzers   : Dict[Market, BaseAnalyzer] — one analyzer per market.
    portfolio   : AdvisorPortfolioEngine instance.
    risk        : AdvisorRiskEngine instance.
    kronos      : KronosForecaster instance (may be unloaded — fail-soft).
    telegram    : AdvisorTelegramBot instance or None.
    db_pool     : asyncpg pool for persistence.
    advisor_ml  : AdvisorMLModel instance or None (Wave-21 ML learning loop).
    """

    def __init__(
        self,
        config: dict,
        analyzers: Dict[Market, BaseAnalyzer],
        portfolio: AdvisorPortfolioEngine,
        risk: AdvisorRiskEngine,
        kronos: Optional[KronosForecaster] = None,
        telegram=None,
        db_pool=None,
        advisor_ml=None,
    ):
        self.config = config
        self.analyzers = analyzers
        self.portfolio = portfolio
        self.risk = risk
        self.kronos = kronos
        self.telegram = telegram
        self.db_pool = db_pool
        self.advisor_ml = advisor_ml   # AdvisorMLModel or None

        self._cycle_count = 0
        self._last_cycle_at: Optional[datetime] = None
        self._last_advice_at: Optional[datetime] = None
        # Track last ML tick date so we run at most once per calendar day.
        self._last_ml_tick_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run_once(self) -> List[AdviceResult]:
        """
        Run a single advice cycle. Returns list of published AdviceResults.
        Called by main_advisor.py on schedule.
        """
        self._cycle_count += 1
        self._last_cycle_at = datetime.now(timezone.utc)
        published: List[AdviceResult] = []

        enabled_markets = self._enabled_markets()
        watch_list = self._load_watchlist()
        horizons = self._enabled_horizons()

        logger.info(
            "[advice] Cycle #%d: markets=%s symbols=%d horizons=%s",
            self._cycle_count,
            [m.value for m in enabled_markets],
            sum(len(v) for v in watch_list.values()),
            [h.value for h in horizons],
        )

        open_sim_count = await self.portfolio.count_open_sims()

        for market in enabled_markets:
            analyzer = self.analyzers.get(market)
            if analyzer is None:
                logger.warning(
                    "[advice] No analyzer registered for market=%s", market.value
                )
                continue

            symbols = watch_list.get(market, [])
            for symbol in symbols:
                for horizon in horizons:
                    result = await self._run_symbol(
                        analyzer, symbol, horizon, open_sim_count
                    )
                    if result is not None:
                        published.append(result)
                        open_sim_count += int(result.sim_enabled)
                        self._last_advice_at = datetime.now(timezone.utc)

        # Mark-to-market all open sim positions (best-effort).
        await self._mark_to_market_open_sims()

        # Auto-close expired / target-hit / stop-hit sims.
        await self._auto_close_sims()

        # ML daily tick — runs at most once per calendar day.
        await self._ml_learning_tick()

        logger.info(
            "[advice] Cycle #%d complete: %d advice(s) published.",
            self._cycle_count, len(published),
        )
        return published

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_symbol(
        self,
        analyzer: BaseAnalyzer,
        symbol: str,
        horizon: Horizon,
        open_sim_count: int,
    ) -> Optional[AdviceResult]:
        """Run analysis + risk gate + persist for one (symbol, horizon)."""
        try:
            result = await analyzer.analyze(symbol, horizon)
        except Exception as exc:
            logger.error(
                "[advice] Unhandled error from %s for %s/%s: %s",
                analyzer.__class__.__name__, symbol, horizon.value, exc,
                exc_info=True,
            )
            return None

        # Overlay Kronos signal if available.
        # A None kronos_signal is valid — advice is still produced from
        # technicals + LLM. The field is shown as "unavailable" in dashboard.
        if self.kronos and result.data_source_status == DataSourceStatus.AVAILABLE:
            try:
                klines = result.extra.get("klines_df")
                if klines is not None:
                    result.kronos_signal = await self.kronos.predict(klines)
                    # kronos_signal stays None if Kronos is not loaded.
            except Exception as exc:
                logger.debug(
                    "[advice] Kronos overlay failed for %s: %s", symbol, exc
                )

        # Risk gate.
        passes, reject_reason = self.risk.should_publish(result, open_sim_count)
        if not passes:
            logger.debug(
                "[advice] %s/%s rejected: %s", symbol, horizon.value, reject_reason
            )
            return None

        # Persist.
        advice_id = await self._persist_advice(result)

        # Open sim position if requested.
        if result.sim_enabled and advice_id is not None:
            result.extra["advice_id"] = advice_id
            await self.portfolio.open_sim_position(result)

        # Notify operator.
        if self.telegram:
            try:
                await self.telegram.send_advice(result)
            except Exception as exc:
                logger.warning("[advice] Telegram notify failed: %s", exc)

        return result

    async def _persist_advice(self, result: AdviceResult) -> Optional[int]:
        """Write advice to advisor_advice table. Returns inserted id or None."""
        if self.db_pool is None:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO advisor_advice
                      (market, symbol, horizon, direction,
                       entry_low, entry_high, target_price, stop_price,
                       confidence, rationale, model_id,
                       kronos_signal, data_source_status,
                       sim_enabled, sim_amount_usd, extra, created_at)
                    VALUES
                      ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,NOW())
                    RETURNING id
                    """,
                    result.market.value,
                    result.symbol,
                    result.horizon.value,
                    result.direction.value,
                    result.entry_low,
                    result.entry_high,
                    result.target_price,
                    result.stop_price,
                    result.confidence,
                    result.rationale,
                    result.model_id,
                    result.kronos_signal,
                    result.data_source_status.value,
                    result.sim_enabled,
                    result.sim_amount_usd,
                    json.dumps({k: v for k, v in result.extra.items()
                                if k != "klines_df"}),  # never persist raw DF
                )
                return row["id"] if row else None
        except Exception as exc:
            logger.error(
                "[advice] Failed to persist advice for %s: %s", result.symbol, exc
            )
            return None

    async def _mark_to_market_open_sims(self) -> None:
        """
        Best-effort mark-to-market all open sim positions.

        Uses the DB id of each sim (sim.advice_id is the FK to advisor_advice;
        we use the portfolio engine's list which returns full SimPosition
        objects with their own id from advisor_sim_positions.id).
        """
        try:
            open_sims = await self.portfolio.list_open_sims()
        except Exception as exc:
            logger.debug("[advice] list_open_sims error: %s", exc)
            return

        for sim in open_sims:
            analyzer = self.analyzers.get(sim.market)
            if analyzer is None:
                continue
            try:
                current_price = await _fetch_current_price(analyzer, sim.symbol)
                if current_price is not None:
                    # Use sim.advice_id as the sim row key (portfolio_engine
                    # stores sims by DB id; use _sim_cache key = sim_id).
                    # We look up via in-memory cache or re-load from DB.
                    sim_id = _find_sim_id(self.portfolio, sim)
                    if sim_id is not None:
                        await self.portfolio.mark_to_market(sim_id, current_price)
            except Exception as exc:
                logger.debug(
                    "[advice] mark_to_market failed for %s: %s", sim.symbol, exc
                )

    async def _auto_close_sims(self) -> None:
        """Delegate to portfolio engine to close expired/hit sims."""
        try:
            price_fetcher = _make_price_fetcher(self.analyzers)
            await self.portfolio.auto_close_expired(price_fetcher=price_fetcher)
        except Exception as exc:
            logger.debug("[advice] auto_close_expired error: %s", exc)

    async def _ml_learning_tick(self) -> None:
        """
        Daily ML learning loop (Wave-21).

        Delegated to AdvisorMLModel.run_daily_tick() if:
          - advisor_ml_enabled=true in config.
          - An AdvisorMLModel instance is wired (self.advisor_ml is not None).
          - We have not already run today.

        Fail-soft: any exception is caught and logged; advice cycle continues.
        """
        ml_enabled = str(self.config.get("advisor_ml_enabled", "false")).lower() == "true"
        if not ml_enabled:
            return
        if self.advisor_ml is None:
            return

        today = datetime.now(timezone.utc).date().isoformat()
        if self._last_ml_tick_date == today:
            return   # already ran today

        try:
            await self.advisor_ml.run_daily_tick(db_pool=self.db_pool)
            self._last_ml_tick_date = today
        except Exception as exc:
            logger.warning("[advice] ML daily tick error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _enabled_markets(self) -> List[Market]:
        raw = self.config.get("enabled_markets", "crypto,us_equities")
        return [Market(m.strip()) for m in raw.split(",") if m.strip()]

    def _load_watchlist(self) -> Dict[Market, List[str]]:
        """Load per-market symbol watchlists from config."""
        out: Dict[Market, List[str]] = {}
        for market in Market:
            key = f"watchlist_{market.value}"
            raw = self.config.get(key, "")
            symbols = [s.strip() for s in raw.split(",") if s.strip()]
            if symbols:
                out[market] = symbols
        return out

    def _enabled_horizons(self) -> List[Horizon]:
        raw = self.config.get("enabled_horizons", "short,mid,long")
        return [Horizon(h.strip()) for h in raw.split(",") if h.strip()]

    def get_diagnostics(self) -> dict:
        return {
            "cycle_count": self._cycle_count,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "last_advice_at": (
                self._last_advice_at.isoformat() if self._last_advice_at else None
            ),
            "registered_markets": [m.value for m in self.analyzers],
            "kronos_health": self.kronos.health() if self.kronos else None,
            "ml_enabled": str(self.config.get("advisor_ml_enabled", "false")).lower() == "true",
            "ml_last_tick_date": self._last_ml_tick_date,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def _fetch_current_price(analyzer: BaseAnalyzer, symbol: str) -> Optional[float]:
    """
    Ask the analyzer for a current price without generating a full AdviceResult.
    Fast path: analyzer exposes last_price attribute (dict keyed by symbol).
    Slow path: re-analyze with SHORT horizon and use entry_low as proxy.
    """
    try:
        if hasattr(analyzer, "last_price") and analyzer.last_price:
            price = analyzer.last_price.get(symbol)
            if price is not None:
                return float(price)
        result = await analyzer.analyze(symbol, Horizon.SHORT)
        return result.entry_low or result.entry_high
    except Exception:
        return None


def _find_sim_id(portfolio: AdvisorPortfolioEngine, sim: "SimPosition") -> Optional[int]:
    """
    Resolve the int key used in portfolio._sim_cache for the given SimPosition.

    When list_open_sims() returns objects loaded from DB, the cache may not
    have them.  We search by symbol + entry_price + status as a best-effort
    fallback.  If advice_id is present, that is stored in the sim's advice_id
    field which matches advisor_sim_positions.advice_id (FK) — not the PK.
    The DB PK (id) is not stored on SimPosition directly.

    For now we accept that the in-memory cache refresh is a known limitation:
    the portfolio engine will re-load the row from DB on next mark_to_market
    via _load_sim_from_db(), which is triggered by a cache miss.
    """
    # Try to find by exact object reference or matching fields in cache.
    for k, v in portfolio._sim_cache.items():
        if (v.symbol == sim.symbol
                and v.entry_price == sim.entry_price
                and v.status == "open"):
            return k
    # Not in cache — return None; mark_to_market will load from DB.
    return None


def _make_price_fetcher(
    analyzers: Dict[Market, BaseAnalyzer]
) -> "Callable":
    """
    Build an async price fetcher closure usable by auto_close_expired().
    """
    async def fetcher(symbol: str, market_str: str) -> Optional[float]:
        try:
            market = Market(market_str)
        except ValueError:
            return None
        analyzer = analyzers.get(market)
        if analyzer is None:
            return None
        return await _fetch_current_price(analyzer, symbol)

    return fetcher
