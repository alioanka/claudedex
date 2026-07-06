"""
classifier_worker.py — periodic KAP classification worker.

ADVICE-ONLY. Pulls unclassified disclosures, runs the two-stage classifier
(rule + optional LLM fallback), persists the result, and (optionally) fires a
Telegram polarity-prior alert for non-NEUTRAL high-confidence disclosures.

Lifecycle
---------
    worker = KapClassifierWorker(config=config, db_pool=pool, telegram=tg)
    asyncio.create_task(worker.run())

The worker:
  - Does nothing if advisor_kap_enabled != 'true' (logs once, returns).
  - Every advisor_kap_classify_interval_s (default 60), pulls a batch of
    unclassified disclosures via kap_store.get_unclassified() and classifies
    each one with classifier.classify().
  - Persists each result via kap_store.store_classification() (fail-soft:
    the rule-stage / UNCLASSIFIED fallback always persists even if the LLM
    stage errors — classify() never raises).
  - Fires a polarity-prior Telegram alert ONLY for disclosures whose
    base_polarity is non-NEUTRAL and confidence >= advisor_kap_alert_min_confidence
    (default 0.5), capped at advisor_kap_alert_max_per_cycle (default 10) per
    cycle so a backfill burst cannot spam the chat. Excess alerts are logged,
    not sent.

HONESTY NOTE
------------
The classifier emits an event_type + a documented base_polarity PRIOR only.
There is NO market-impact score. Forward-return impact statistics accumulate
over months in kap_returns and are not produced here.

Cancellation
------------
Stops cleanly on asyncio.CancelledError (mirrors KapListener.run()).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from modules.advisor.core.kap.classifier import classify
from modules.advisor.core.kap.kap_store import get_unclassified, store_classification
from modules.advisor.core.kap.taxonomy import BasePolarity

logger = logging.getLogger("advisor.kap.classifier_worker")

_DEFAULT_INTERVAL_S = 60
_DEFAULT_BATCH_LIMIT = 100
_DEFAULT_ALERT_MIN_CONF = 0.5
_DEFAULT_ALERT_MAX_PER_CYCLE = 10
# Wave-F5 fix 7: re-queue budget-starved UNCLASSIFIED stamps after this many
# hours (0 disables). Bounded: kap_store returns never-classified rows FIRST,
# the batch LIMIT caps the cycle, and every attempt re-stamps classified_at so
# a row retries at most once per window. llm_budget.try_consume remains the
# hard cost backstop — a budget-less retry is a cheap local no-op (no API call).
_DEFAULT_RECLASSIFY_AFTER_HOURS = 24.0


class KapClassifierWorker:
    """
    Periodic worker that classifies unclassified KAP disclosures and emits
    optional polarity-prior Telegram alerts.

    Parameters
    ----------
    config   : advisor config dict (advisor_config DB rows).
    db_pool  : asyncpg pool. Worker is a no-op if None.
    telegram : AdvisorTelegramBot instance or None (alerts skipped if None).
    """

    def __init__(self, config: dict, db_pool=None, telegram=None,
                 portfolio=None, bist_analyzer=None):
        self.config = config
        self.db_pool = db_pool
        self.telegram = telegram
        # Optional: enable KAP-driven BIST sims (strong polarity -> dry-run sim
        # in the 'kap' channel). Both must be provided AND advisor_kap_sim_enabled
        # must be true; otherwise the hook is a no-op. Fail-soft throughout.
        self.portfolio = portfolio
        self.bist_analyzer = bist_analyzer
        self._running = False
        # Disclosures already CLASSIFY-attempted in this process run. Guards
        # against a runaway re-classification loop: if store_classification()
        # fails (e.g. the kap_classifications table is missing) the disclosure
        # would otherwise stay "unclassified" and be re-LLM'd every cycle —
        # which once burned a full day of paid API credit in ~3 hours. We never
        # re-attempt the same disclosure within a run; a process restart retries
        # once. The global llm_budget cap is the second, hard backstop.
        self._attempted_ids: set = set()
        # Wave-F5 fix 7: aged-UNCLASSIFIED rows re-surfaced by the re-queue may
        # bypass _attempted_ids ONCE per process run (a second in-run retry is
        # only possible if the re-stamp failed to persist — exactly the store-
        # failure loop the guard exists for, so it stays blocked).
        self._requeue_attempted_ids: set = set()
        self._store_failure_warned = False

    # ------------------------------------------------------------------
    # Config accessors
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return str(self.config.get("advisor_kap_enabled", "false")).lower() == "true"

    @property
    def interval_s(self) -> int:
        try:
            return int(self.config.get("advisor_kap_classify_interval_s", _DEFAULT_INTERVAL_S))
        except (ValueError, TypeError):
            return _DEFAULT_INTERVAL_S

    @property
    def alert_min_confidence(self) -> float:
        try:
            return float(self.config.get("advisor_kap_alert_min_confidence", _DEFAULT_ALERT_MIN_CONF))
        except (ValueError, TypeError):
            return _DEFAULT_ALERT_MIN_CONF

    @property
    def alert_max_per_cycle(self) -> int:
        try:
            return int(self.config.get("advisor_kap_alert_max_per_cycle", _DEFAULT_ALERT_MAX_PER_CYCLE))
        except (ValueError, TypeError):
            return _DEFAULT_ALERT_MAX_PER_CYCLE

    @property
    def reclassify_after_hours(self) -> float:
        """Hours after which a stamped-UNCLASSIFIED row is retried (Wave-F5
        fix 7). advisor_kap_reclassify_after_hours, default 24; 0 disables."""
        try:
            return max(0.0, float(self.config.get(
                "advisor_kap_reclassify_after_hours",
                _DEFAULT_RECLASSIFY_AFTER_HOURS)))
        except (ValueError, TypeError):
            return _DEFAULT_RECLASSIFY_AFTER_HOURS

    @property
    def kap_sim_enabled(self) -> bool:
        """
        Whether strong-polarity disclosures open a KAP-driven BIST sim ('kap'
        channel). Requires a portfolio engine + BIST analyzer to be wired AND
        advisor_kap_sim_enabled='true' (default true). Fail-soft if the hook errors.
        """
        if self.portfolio is None or self.bist_analyzer is None:
            return False
        return str(self.config.get("advisor_kap_sim_enabled", "true")).lower() == "true"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop. Classifies on a fixed interval until CancelledError."""
        if not self.enabled:
            logger.info(
                "[kap.classifier_worker] advisor_kap_enabled=false -- worker not started."
            )
            return
        if self.db_pool is None:
            logger.warning(
                "[kap.classifier_worker] No db_pool -- worker cannot run. Returning."
            )
            return

        logger.info(
            "[kap.classifier_worker] Starting (interval=%ds, alert_min_conf=%.2f, "
            "alert_max_per_cycle=%d).",
            self.interval_s, self.alert_min_confidence, self.alert_max_per_cycle,
        )
        self._running = True
        try:
            while True:
                try:
                    await self._classify_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "[kap.classifier_worker] Classify cycle error (will retry): %s", exc
                    )
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            logger.info("[kap.classifier_worker] Stopped (cancelled).")
        finally:
            self._running = False

    async def _classify_cycle(self) -> None:
        """One cycle: pull unclassified disclosures, classify, persist, alert."""
        rows = await get_unclassified(
            self.db_pool, limit=_DEFAULT_BATCH_LIMIT,
            reclassify_after_hours=self.reclassify_after_hours,
        )
        if not rows:
            return

        classified = 0
        alerts_sent = 0
        alerts_suppressed = 0
        min_conf = self.alert_min_confidence
        max_alerts = self.alert_max_per_cycle

        for row in rows:
            row_id = row.get("id")

            # Runaway-loop hard-stop: never re-attempt a disclosure already
            # classified this run. Without this, a persistent store failure
            # re-LLM's the same rows every cycle (the cost runaway).
            # EXCEPTION (Wave-F5 fix 7): rows flagged requeued=True came back
            # through the aged-UNCLASSIFIED re-queue — their previous stamp DID
            # persist (store works), and kap_store only re-surfaces them after
            # advisor_kap_reclassify_after_hours. Honor the retry AT MOST ONCE
            # per process run (each attempt re-stamps classified_at, so the DB
            # window bounds retries across runs).
            is_requeue = bool(row.get("requeued"))
            if row_id in self._attempted_ids:
                if not is_requeue or row_id in self._requeue_attempted_ids:
                    continue
            if is_requeue:
                self._requeue_attempted_ids.add(row_id)

            # The classifier reads disclosure['id'] / 'subject' / 'text'.
            # kap_disclosures stores body text under 'full_text'.
            disclosure = {
                "id":      row_id,
                "subject": row.get("subject", ""),
                "text":    row.get("full_text", "") or row.get("summary", ""),
            }

            # classify() never raises: LLM failure -> rule/UNCLASSIFIED fallback.
            result = classify(disclosure, config=self.config, caller_logger=logger)
            # Mark attempted REGARDLESS of store success below — a failed store
            # must not cause the same disclosure to be re-classified next cycle.
            self._attempted_ids.add(row_id)

            # disclosure_id column in kap_classifications is the BIGINT PK
            # (kap_disclosures.id), not the KAP string index.
            ok = await store_classification(
                self.db_pool,
                disclosure_id=row.get("id"),
                event_type=result.event_type.value,
                sentiment=result.base_polarity.value,
                confidence=result.confidence,
                classifier_model=result.classifier_stage,
                extra={
                    "params": result.params,
                    "raw_subject": result.raw_subject,
                    "kap_disclosure_id": row.get("disclosure_id"),
                },
            )
            if ok:
                classified += 1
            elif not self._store_failure_warned:
                self._store_failure_warned = True
                logger.warning(
                    "[kap.classifier_worker] store_classification FAILED (is the "
                    "kap_classifications table present? run migrations). "
                    "Classified disclosures will NOT be re-LLM'd this run "
                    "(loop-guard active), but classifications are not persisting."
                )

            # Polarity-prior alert (non-NEUTRAL, high-confidence only).
            if (
                result.base_polarity != BasePolarity.NEUTRAL
                and result.confidence >= min_conf
            ):
                if alerts_sent < max_alerts:
                    if await self._fire_alert(row, result):
                        alerts_sent += 1
                else:
                    alerts_suppressed += 1

            # KAP-driven BIST sim (strong polarity only). Fully fail-soft: any
            # error here is swallowed and NEVER affects classification/alerts.
            if self.kap_sim_enabled:
                await self._maybe_open_kap_sim(row, result)

        if classified:
            logger.info(
                "[kap.classifier_worker] Cycle complete: %d classified, "
                "%d alert(s) sent%s.",
                classified, alerts_sent,
                f", {alerts_suppressed} suppressed (per-cycle cap)" if alerts_suppressed else "",
            )

    async def _maybe_open_kap_sim(self, row: dict, result) -> None:
        """
        Open a KAP-driven BIST sim for a strong-polarity disclosure (delegates to
        kap_sim.maybe_open_kap_sim). Resolves the BIST ticker from the disclosure
        row. Fully fail-soft: never raises into the classify cycle.
        """
        try:
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                return
            from modules.advisor.core.kap.kap_sim import maybe_open_kap_sim
            await maybe_open_kap_sim(
                ticker=ticker,
                base_polarity=result.base_polarity.value,
                confidence=result.confidence,
                config=self.config,
                portfolio=self.portfolio,
                bist_analyzer=self.bist_analyzer,
            )
        except Exception as exc:
            logger.warning(
                "[kap.classifier_worker] KAP-driven sim hook error (fail-soft): %s", exc
            )

    async def _fire_alert(self, row: dict, result) -> bool:
        """Fire a Telegram polarity-prior alert. Fail-soft; returns send status."""
        if self.telegram is None:
            return False
        try:
            return await self.telegram.send_kap_alert(
                ticker=row.get("ticker") or "",
                company=row.get("company_name") or "",
                event_type=result.event_type.value,
                base_polarity=result.base_polarity.value,
                classifier_stage=result.classifier_stage,
                confidence=result.confidence,
                subject=row.get("subject") or result.raw_subject,
                url=row.get("url") or "",
            )
        except Exception as exc:
            logger.warning("[kap.classifier_worker] send_kap_alert failed: %s", exc)
            return False
