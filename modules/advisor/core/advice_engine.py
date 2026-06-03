"""
AdviceEngine — orchestrates the full advice cycle.

One cycle per run_interval_minutes (default 60):
  1. Load enabled markets + symbols from advisor_config.
  2. For each (market, symbol, horizon): call the appropriate analyzer.
  3. Optionally overlay Kronos forecast signal.
  4. Compute multi-layer composite signal (signal_engine) and store in extra['signal'].
  5. Pass through AdvisorRiskEngine gate.
  6. Persist accepted advice to advisor_advice table.
  7. If sim_enabled on advice: open sim position via portfolio_engine.
  8. Notify via AdvisorTelegramBot.
  9. Mark-to-market existing open sim positions.
  10. Auto-close expired / target-hit / stop-hit sims.

Wave-21 changes (quant agent)
  - _mark_to_market_open_sims: uses sim.id (from DB) not advice_id.
  - auto-close hook: calls portfolio.auto_close_expired() after mark-to-market.
  - _ml_learning_tick: implemented (see advisor_ml.py).
  - Kronos overlay: None handled cleanly; advice still produced.

Wave-22 changes (quant agent)
  - _overlay_signal: compute_composite_signal called after Kronos overlay;
    result stored in AdviceResult.extra['signal']. Fail-soft.
  - build_rationale now accepts extra= kwarg to receive dual-advice artefacts
    (rationale_anthropic, rationale_openai, providers_disagree) when
    advisor_dual_advice_mode != 'off'. Analyzers pass result.extra through.
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
from modules.advisor.core.signal_engine import compute_composite_signal

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
        # Discovery cadence: in-memory last-run timestamp (seeded from the
        # persisted config row on first use so a restart respects the cadence).
        self._last_discovery_at: Optional[datetime] = None

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

        # PER-MARKET sim caps (issue #13): max_sim_positions is the cap PER
        # market/strategy (e.g. 10 crypto + 10 BIST independently), NOT one
        # global cap across all markets. We track the open count per market and
        # pass the relevant market's count to the risk gate.
        open_sim_by_market = await self.portfolio.count_open_sims_by_market()

        for market in enabled_markets:
            analyzer = self.analyzers.get(market)
            if analyzer is None:
                logger.warning(
                    "[advice] No analyzer registered for market=%s", market.value
                )
                continue

            market_open = open_sim_by_market.get(market.value, 0)
            symbols = watch_list.get(market, [])
            for symbol in symbols:
                for horizon in horizons:
                    result = await self._run_symbol(
                        analyzer, symbol, horizon, market_open
                    )
                    if result is not None:
                        published.append(result)
                        market_open += int(result.sim_enabled)
                        self._last_advice_at = datetime.now(timezone.utc)
            open_sim_by_market[market.value] = market_open

        # GATED discovery pass — surfaces NEW symbols beyond the watchlist on a
        # slow cadence (advisor_discovery_refresh_hours). Default OFF. Fail-soft:
        # any error here NEVER affects the watchlist advice already published.
        try:
            disc = await self._run_discovery_pass(horizons, open_sim_by_market)
            published.extend(disc)
        except Exception as exc:
            logger.warning("[advice] discovery pass error (fail-soft): %s", exc)

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
        origin: str = "watchlist",
    ) -> Optional[AdviceResult]:
        """Run analysis + risk gate + persist for one (symbol, horizon).

        `origin` marks the advice provenance: 'watchlist' (default) or
        'discovery' (surfaced by the New Gems discovery layer). It is stored on
        the AdviceResult (extra['origin']) and persisted to advisor_advice.origin.
        """
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

        # Multi-layer composite signal (Wave-22). Fail-soft: any error leaves
        # extra['signal'] absent; advice cycle is unaffected.
        await self._overlay_composite_signal(result)

        # KAP disclosure context (BIST only, advice-only). Surfaces recent
        # classified disclosures as CONTEXT in extra['kap_context']; does NOT
        # mutate the numeric score. Fail-soft.
        if result.market == Market.BIST:
            await self._overlay_kap_context(result)

        # Dual-advice: if advisor_dual_advice_mode != 'off' and an OpenAI key
        # is available, fetch the OpenAI rationale and store artefacts.
        # The primary result.rationale (from the analyzer / Anthropic) is
        # preserved; OpenAI text goes into result.extra['rationale_openai'].
        await self._overlay_dual_advice(result, symbol, horizon)

        # Provenance marker (watchlist vs discovery) — for persistence + dashboard.
        result.extra["origin"] = origin

        # Risk gate.
        passes, reject_reason = self.risk.should_publish(result, open_sim_count)
        if not passes:
            logger.debug(
                "[advice] %s/%s rejected: %s", symbol, horizon.value, reject_reason
            )
            return None

        # Persist.
        advice_id = await self._persist_advice(result, origin=origin)

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

    async def _persist_advice(
        self, result: AdviceResult, origin: str = "watchlist"
    ) -> Optional[int]:
        """Write advice to advisor_advice table. Returns inserted id or None.

        `origin` is persisted to advisor_advice.origin (migration 076). The
        INSERT is resilient to a DB that has not yet run migration 076: on an
        UndefinedColumn error it retries without the origin column (fail-soft).
        """
        if self.db_pool is None:
            return None
        extra_json = json.dumps({k: v for k, v in result.extra.items()
                                 if k != "klines_df"})  # never persist raw DF
        try:
            async with self.db_pool.acquire() as conn:
                try:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO advisor_advice
                          (market, symbol, horizon, direction,
                           entry_low, entry_high, target_price, stop_price,
                           confidence, rationale, model_id,
                           kronos_signal, data_source_status,
                           sim_enabled, sim_amount_usd, extra, origin, created_at)
                        VALUES
                          ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,NOW())
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
                        extra_json,
                        origin,
                    )
                except Exception as col_exc:
                    # Pre-migration-076 DB has no origin column — fall back.
                    if "origin" not in str(col_exc).lower():
                        raise
                    logger.debug(
                        "[advice] origin column missing (run migration 076); "
                        "persisting without it."
                    )
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
                        extra_json,
                    )
                return row["id"] if row else None
        except Exception as exc:
            logger.error(
                "[advice] Failed to persist advice for %s: %s", result.symbol, exc
            )
            return None

    async def _overlay_composite_signal(self, result: AdviceResult) -> None:
        """
        Wave-22: compute multi-layer composite signal and store in extra['signal'].

        Reads the klines DataFrame from extra['klines_df'] (populated by the
        analyzer). Fail-soft: any error is caught and logged at DEBUG level;
        the advice cycle continues without the composite signal.
        """
        try:
            klines = result.extra.get("klines_df")
            if klines is None or len(klines) < 20:
                return
            signal = compute_composite_signal(
                ohlcv_df=klines,
                market=result.market.value,
                config=self.config,
                kronos_signal=result.kronos_signal,
            )
            result.extra["signal"] = signal
            logger.debug(
                "[advice] signal overlay for %s: action=%s confidence=%d regime=%s",
                result.symbol,
                signal.get("action", "?"),
                signal.get("confidence", 0),
                signal.get("market_regime", "?"),
            )
        except Exception as exc:
            logger.debug(
                "[advice] _overlay_composite_signal failed for %s: %s",
                result.symbol, exc,
            )

    async def _overlay_kap_context(self, result: AdviceResult) -> None:
        """
        Attach recent KAP disclosure context to a BIST AdviceResult.

        ADVICE-ONLY context overlay. Looks up the most recent CLASSIFIED
        disclosures for this ticker within advisor_kap_lookback_days (default 7)
        and stores them in result.extra['kap_context'].

        IMPORTANT — this is CONTEXT for the operator. It does NOT flip the
        action and does NOT mutate the numeric confidence/score. base_polarity
        is a documented PRIOR, not a quantitative impact estimate (impact stats
        require months of forward-return accumulation in kap_returns).

        No-op if KAP is disabled, db_pool is None, or no classified disclosures
        exist for the ticker. Fail-soft: any error is caught and logged at DEBUG.
        """
        if self.db_pool is None:
            return
        if str(self.config.get("advisor_kap_enabled", "false")).lower() != "true":
            return
        try:
            from modules.advisor.core.kap.kap_store import (
                get_recent_classified_for_ticker,
            )

            try:
                lookback_days = int(self.config.get("advisor_kap_lookback_days", 7))
            except (ValueError, TypeError):
                lookback_days = 7

            rows = await get_recent_classified_for_ticker(
                self.db_pool, result.symbol, lookback_days=lookback_days, limit=5
            )
            if not rows:
                return

            disclosures = []
            for r in rows:
                disclosed_at = r.get("disclosed_at")
                disclosures.append({
                    "event_type":       r.get("event_type"),
                    "base_polarity":    r.get("base_polarity"),
                    "classifier_stage": r.get("classifier_stage"),
                    "confidence":       float(r["confidence"]) if r.get("confidence") is not None else None,
                    "subject":          (r.get("subject") or "")[:200],
                    "disclosed_at":     disclosed_at.isoformat() if disclosed_at else None,
                    "url":              r.get("url") or "",
                })

            result.extra["kap_context"] = {
                "lookback_days": lookback_days,
                "count": len(disclosures),
                "disclosures": disclosures,
                "disclaimer": (
                    "KAP base_polarity is a documented PRIOR, NOT a price "
                    "prediction or market-impact score. Context only — the "
                    "advice action and confidence were NOT modified by KAP."
                ),
            }
            logger.debug(
                "[advice] KAP context overlay for %s: %d recent disclosure(s)",
                result.symbol, len(disclosures),
            )
        except Exception as exc:
            logger.debug(
                "[advice] _overlay_kap_context failed for %s: %s",
                result.symbol, exc,
            )

    async def _overlay_dual_advice(
        self,
        result: AdviceResult,
        symbol: str,
        horizon: Horizon,
    ) -> None:
        """
        Wave-22: fetch OpenAI second opinion if advisor_dual_advice_mode != 'off'.

        The Anthropic rationale is already in result.rationale (produced by the
        analyzer). This method:
          1. Checks advisor_dual_advice_mode in config.
          2. If 'both' or 'consensus': calls OpenAI and stores:
               extra['rationale_anthropic'] = result.rationale (copy)
               extra['rationale_openai']    = OpenAI text
               extra['dual_advice_mode']    = mode
          3. In 'consensus' mode: compares direction keywords extracted from
             each rationale; if they disagree:
               extra['providers_disagree'] = True
             The dashboard uses this flag to show a disagreement warning.

        Fail-soft: any error silently skips dual-advice; primary rationale unchanged.
        """
        from modules.advisor.core.rationale_helper import (
            _resolve_openai_key,
            _DUAL_MODE_OFF,
            _DUAL_MODE_BOTH,
            _DUAL_MODE_CONSENSUS,
            _build_signal_summary,
            _build_prompt,
            _MARKET_LABEL,
            _call_openai,
            _extract_direction_from_text,
            build_rule_based_rationale,
        )
        from modules.advisor.core.models import Direction

        dual_mode = str(
            self.config.get("advisor_dual_advice_mode", _DUAL_MODE_OFF)
        ).lower().strip()

        if dual_mode == _DUAL_MODE_OFF:
            return

        openai_key = _resolve_openai_key(self.config)
        if not openai_key:
            return

        try:
            # Re-extract signals from extra for the prompt (best-effort).
            # The analyzer may not have put a clean signal dict at extra['signals'],
            # so we build a minimal one from the existing klines or stored values.
            stored_signals = result.extra.get("signals", {})
            # Fallback: reconstruct a minimal signals dict from result fields.
            if not stored_signals:
                stored_signals = {
                    "close":      result.entry_low or 0,
                    "sma20":      0,
                    "sma50":      0,
                    "rsi":        50,
                    "sma_signal": 1 if result.direction.value == "long" else -1,
                }

            from modules.advisor.core.models import Horizon as _Horizon
            openai_model  = self.config.get("advisor_openai_model", "gpt-4o")
            mkt_label     = _MARKET_LABEL.get(result.market, result.market.value)
            signal_summary = _build_signal_summary(
                symbol, result.market, horizon, stored_signals, result.direction
            )
            prompt = _build_prompt(signal_summary, mkt_label)
            fallback = result.rationale  # use existing rationale as fallback

            openai_text = await _call_openai(
                prompt, openai_model, openai_key, fallback, logger, symbol
            )

            result.extra["dual_advice_mode"]     = dual_mode
            result.extra["rationale_anthropic"]  = result.rationale
            result.extra["rationale_openai"]     = openai_text

            if dual_mode == _DUAL_MODE_CONSENSUS:
                dir_a = _extract_direction_from_text(result.rationale)
                dir_o = _extract_direction_from_text(openai_text)
                disagree = (
                    dir_a is not None
                    and dir_o is not None
                    and dir_a != dir_o
                )
                result.extra["providers_disagree"] = disagree
                if disagree:
                    # A provider disagreement should LOWER confidence — it is a
                    # real signal of uncertainty. The analyzer computed confidence
                    # before dual-advice ran, so apply the penalty here once.
                    try:
                        penalty = float(
                            self.config.get("levels_conf_disagree_penalty", 0.20)
                        )
                    except (TypeError, ValueError):
                        penalty = 0.20
                    try:
                        conf_floor = float(self.config.get("levels_conf_floor", 0.05))
                    except (TypeError, ValueError):
                        conf_floor = 0.05
                    before = result.confidence
                    result.confidence = max(conf_floor, result.confidence - penalty)
                    result.extra["confidence_pre_disagree"] = before
                    logger.info(
                        "[advice] Dual-advice DISAGREE for %s: "
                        "anthropic=%s openai=%s; confidence %.3f -> %.3f",
                        symbol, dir_a, dir_o, before, result.confidence,
                    )

        except Exception as exc:
            logger.debug(
                "[advice] _overlay_dual_advice failed for %s: %s", symbol, exc
            )

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

    # ------------------------------------------------------------------
    # Discovery ("New Gems") pass
    # ------------------------------------------------------------------

    async def _run_discovery_pass(
        self,
        horizons: List[Horizon],
        open_sim_by_market: Dict[str, int],
    ) -> List[AdviceResult]:
        """
        GATED discovery pass. Surfaces NEW symbols beyond the watchlist and runs
        them through the SAME analyzer pipeline, flagged origin='discovery'.

        Gates (all must hold):
          - advisor_discovery_enabled == true
          - cadence: at least advisor_discovery_refresh_hours since last run

        COST DISCIPLINE: discovery candidates default to rule-based rationale
        (NO LLM) — the analyzer config is shallow-copied with
        _advisor_force_rule_based=true. Only the top advisor_discovery_llm_max
        candidates per pass get LLM narration, and that path still flows through
        the global daily paid-LLM budget (core/llm_budget) — discovery cannot
        blow the budget.

        Fail-soft throughout: any error -> empty list, watchlist advice intact.
        """
        if str(self.config.get("advisor_discovery_enabled", "false")).lower() != "true":
            return []
        if not self._discovery_due():
            return []

        try:
            from modules.advisor.core.discovery import discover_candidates
        except Exception as exc:
            logger.warning("[discovery] import failed (fail-soft): %s", exc)
            return []

        markets = [
            m.strip() for m in
            str(self.config.get("advisor_discovery_markets", "crypto,us_equities,bist")).split(",")
            if m.strip()
        ]
        try:
            llm_max = int(float(self.config.get("advisor_discovery_llm_max", 0)))
        except (TypeError, ValueError):
            llm_max = 0
        llm_budget_remaining = max(0, llm_max)

        # Build per-market exclusion data once.
        watch_list = self._load_watchlist()
        open_symbols_by_market = await self._open_sim_symbols_by_market()
        blocklist = [
            s.strip() for s in str(self.config.get("blocked_symbols", "")).split(",")
            if s.strip()
        ]

        # Rule-based config clone for the cost-free discovery analysis path.
        rule_based_cfg = dict(self.config)
        rule_based_cfg["_advisor_force_rule_based"] = "true"

        published: List[AdviceResult] = []
        total_new = 0
        for market_str in markets:
            try:
                market = Market(market_str)
            except ValueError:
                logger.debug("[discovery] unknown market '%s' skipped.", market_str)
                continue
            analyzer = self.analyzers.get(market)
            if analyzer is None:
                continue

            wl = watch_list.get(market, [])
            open_syms = open_symbols_by_market.get(market.value, [])
            try:
                candidates = await discover_candidates(
                    market.value, self.config,
                    watchlist=wl, open_sims=open_syms, blocklist=blocklist,
                )
            except Exception as exc:
                logger.warning("[discovery] %s candidate fetch failed: %s", market.value, exc)
                continue

            if not candidates:
                continue

            logger.info(
                "[discovery] %s: %d new candidate(s): %s",
                market.value, len(candidates),
                ", ".join(c.symbol for c in candidates),
            )
            total_new += len(candidates)
            market_open = open_sim_by_market.get(market.value, 0)

            for cand in candidates:
                # Pick the rationale config: LLM-narrate only while budget allows.
                if llm_budget_remaining > 0:
                    cfg_for_call = self.config
                    llm_budget_remaining -= 1
                else:
                    cfg_for_call = rule_based_cfg
                for horizon in horizons:
                    result = await self._run_discovery_symbol(
                        analyzer, cand, horizon, market_open, cfg_for_call
                    )
                    if result is not None:
                        published.append(result)
                        market_open += int(result.sim_enabled)
                        self._last_advice_at = datetime.now(timezone.utc)
            open_sim_by_market[market.value] = market_open

        # Record the run so cadence is honoured (in-memory + best-effort DB).
        await self._mark_discovery_ran()
        logger.info(
            "[discovery] pass complete: %d candidate(s) across %d market(s); "
            "%d discovery advice(s) published.",
            total_new, len(markets), len(published),
        )
        return published

    async def _run_discovery_symbol(
        self,
        analyzer: BaseAnalyzer,
        cand,
        horizon: Horizon,
        open_sim_count: int,
        cfg_for_call: dict,
    ) -> Optional[AdviceResult]:
        """
        Analyze one discovery candidate with the (possibly rule-based) config,
        then publish with origin='discovery'. Temporarily swaps the analyzer's
        config so the rationale-helper cost short-circuit applies, then restores.
        """
        original_cfg = analyzer.config
        try:
            analyzer.config = cfg_for_call
            result = await self._run_symbol(
                analyzer, cand.symbol, horizon, open_sim_count, origin="discovery"
            )
            if result is not None:
                # Attach the discovery score/metrics for the dashboard.
                result.extra["discovery"] = {
                    "score": getattr(cand, "score", None),
                    "source": getattr(cand, "source", ""),
                    "change_pct_24h": getattr(cand, "change_pct_24h", None),
                    "quote_volume": getattr(cand, "quote_volume", None),
                }
            return result
        except Exception as exc:
            logger.debug("[discovery] analyze failed for %s: %s", cand.symbol, exc)
            return None
        finally:
            analyzer.config = original_cfg

    def _discovery_due(self) -> bool:
        """True if the discovery cadence window has elapsed since last run."""
        try:
            refresh_hours = float(self.config.get("advisor_discovery_refresh_hours", 12))
        except (TypeError, ValueError):
            refresh_hours = 12.0
        if refresh_hours <= 0:
            return True  # 0 = run every cycle (not recommended)

        last = self._last_discovery_at
        if last is None:
            # Seed from the persisted config row so a restart respects cadence.
            last = self._parse_iso(self.config.get("advisor_discovery_last_run_at", ""))
            self._last_discovery_at = last
        if last is None:
            return True
        elapsed_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600.0
        return elapsed_h >= refresh_hours

    async def _mark_discovery_ran(self) -> None:
        """Persist the discovery last-run timestamp (in-memory + best-effort DB)."""
        now = datetime.now(timezone.utc)
        self._last_discovery_at = now
        if self.db_pool is None:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE config_settings
                       SET value = $1, updated_at = NOW()
                     WHERE config_type = 'advisor_config'
                       AND key = 'advisor_discovery_last_run_at'
                    """,
                    now.isoformat(),
                )
        except Exception as exc:
            logger.debug("[discovery] last-run persist failed (soft): %s", exc)

    async def _open_sim_symbols_by_market(self) -> Dict[str, List[str]]:
        """Return open-sim symbols grouped by market value (for dedupe)."""
        out: Dict[str, List[str]] = {}
        try:
            for s in await self.portfolio.list_open_sims():
                out.setdefault(s.market.value, []).append(s.symbol)
        except Exception as exc:
            logger.debug("[discovery] open-sim symbol load failed (soft): %s", exc)
        return out

    @staticmethod
    def _parse_iso(raw) -> Optional[datetime]:
        """Parse an ISO-8601 timestamp; tz-aware UTC. None on any failure."""
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(str(raw))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

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
            "discovery_enabled": str(self.config.get("advisor_discovery_enabled", "false")).lower() == "true",
            "discovery_last_run_at": (
                self._last_discovery_at.isoformat() if self._last_discovery_at else None
            ),
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
