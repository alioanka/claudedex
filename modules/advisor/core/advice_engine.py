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

Stub status (Wave-20): scaffolded; LLM rationale generation deferred to
the advice specialist agent (Wave-21).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
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
    ):
        self.config = config
        self.analyzers = analyzers
        self.portfolio = portfolio
        self.risk = risk
        self.kronos = kronos
        self.telegram = telegram
        self.db_pool = db_pool

        self._cycle_count = 0
        self._last_cycle_at: Optional[datetime] = None
        self._last_advice_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run_once(self) -> List[AdviceResult]:
        """
        Run a single advice cycle. Returns list of published AdviceResults.
        Called by main_advisor.py on schedule.
        """
        self._cycle_count += 1
        self._last_cycle_at = datetime.utcnow()
        published: List[AdviceResult] = []

        enabled_markets = self._enabled_markets()
        watch_list = self._load_watchlist()
        horizons = self._enabled_horizons()

        logger.info(
            f"[advice] Cycle #{self._cycle_count}: "
            f"markets={[m.value for m in enabled_markets]} "
            f"symbols={len(watch_list)} horizons={[h.value for h in horizons]}"
        )

        open_sim_count = await self.portfolio.count_open_sims()

        for market in enabled_markets:
            analyzer = self.analyzers.get(market)
            if analyzer is None:
                logger.warning(f"[advice] No analyzer registered for market={market.value}")
                continue

            symbols = watch_list.get(market, [])
            for symbol in symbols:
                for horizon in horizons:
                    result = await self._run_symbol(analyzer, symbol, horizon, open_sim_count)
                    if result is not None:
                        published.append(result)
                        open_sim_count += int(result.sim_enabled)
                        self._last_advice_at = datetime.utcnow()

        # Mark-to-market all open sim positions (best-effort)
        await self._mark_to_market_open_sims()

        # Daily ML learning tick (stub — ML agent fills in Wave-22)
        await self._ml_learning_tick()

        logger.info(
            f"[advice] Cycle #{self._cycle_count} complete: "
            f"{len(published)} advice(s) published."
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
                f"[advice] Unhandled error from {analyzer.__class__.__name__} "
                f"for {symbol}/{horizon.value}: {exc}",
                exc_info=True,
            )
            return None

        # Overlay Kronos signal if available
        if self.kronos and result.data_source_status == DataSourceStatus.AVAILABLE:
            try:
                klines = result.extra.get("klines_df")
                if klines is not None:
                    result.kronos_signal = await self.kronos.predict(klines)
            except Exception as exc:
                logger.debug(f"[advice] Kronos overlay failed for {symbol}: {exc}")

        # Risk gate
        passes, reject_reason = self.risk.should_publish(result, open_sim_count)
        if not passes:
            logger.debug(
                f"[advice] {symbol}/{horizon.value} rejected: {reject_reason}"
            )
            return None

        # Persist
        advice_id = await self._persist_advice(result)

        # Open sim position if requested
        if result.sim_enabled and advice_id is not None:
            result.extra["advice_id"] = advice_id
            await self.portfolio.open_sim_position(result)

        # Notify operator
        if self.telegram:
            try:
                await self.telegram.send_advice(result)
            except Exception as exc:
                logger.warning(f"[advice] Telegram notify failed: {exc}")

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
            logger.error(f"[advice] Failed to persist advice for {result.symbol}: {exc}")
            return None

    async def _mark_to_market_open_sims(self) -> None:
        """Best-effort mark-to-market all open sim positions."""
        sims = await self.portfolio.list_open_sims()
        for sim in sims:
            # Analyzer for this sim's market should know current price.
            analyzer = self.analyzers.get(sim.market)
            if analyzer is None:
                continue
            try:
                current_price = await _fetch_current_price(analyzer, sim.symbol)
                if current_price is not None:
                    await self.portfolio.mark_to_market(
                        sim.advice_id or 0, current_price
                    )
            except Exception as exc:
                logger.debug(f"[advice] mark_to_market failed for {sim.symbol}: {exc}")

    async def _ml_learning_tick(self) -> None:
        """
        STUB: Daily ML learning loop.
        ML specialist agent (Wave-22) implements this.
        For now: no-op.
        """
        pass

    def _enabled_markets(self) -> List[Market]:
        raw = self.config.get("enabled_markets", "crypto,us_equities")
        return [Market(m.strip()) for m in raw.split(",") if m.strip()]

    def _load_watchlist(self) -> Dict[Market, List[str]]:
        """
        Load per-market symbol watchlists from config.
        Keys: watchlist_crypto, watchlist_us_equities, etc.
        """
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
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def _fetch_current_price(analyzer: BaseAnalyzer, symbol: str) -> Optional[float]:
    """
    Ask the analyzer for a current price without generating a full AdviceResult.
    Analyzers may store the last-fetched price in their cache; otherwise
    a fresh SHORT-horizon analysis is used and entry_low is taken as proxy.
    """
    try:
        # Fast path: analyzer exposes current_price attribute
        if hasattr(analyzer, "last_price") and analyzer.last_price:
            return analyzer.last_price.get(symbol)
        # Slow path: re-analyze with SHORT horizon
        result = await analyzer.analyze(symbol, Horizon.SHORT)
        return result.entry_low or result.entry_high
    except Exception:
        return None
