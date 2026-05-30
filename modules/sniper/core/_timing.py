"""Per-opportunity timing instrumentation for SNIPER latency tracking.

Stamps monotonic markers (time.perf_counter, not wall-clock so NTP
jumps don't poison deltas) at each lifecycle stage. Emits one
structured log line per executed snipe so latency budget is
observable end-to-end.

Phase-0 deliverable from docs/agents/reports/SNIPER_LATENCY_PLAN.md.
Always-on; pure observability; no behavior change."""

from dataclasses import dataclass, field
from typing import Optional
import time
import logging

logger = logging.getLogger("SniperTiming")


@dataclass
class SnipeTimingContext:
    """Latency markers for a single snipe opportunity. All times are
    time.perf_counter() floats; subtract to get monotonic deltas."""
    token_address: str = ""
    chain: str = ""
    # Stage 1: detection — when the block was produced. Parsed from
    # target['timestamp'] (ISO wall-clock anchored on on-chain
    # block_time when available). Approximates "block production
    # time" in perf_counter space via parse_iso_to_perf_counter.
    t_detect: Optional[float] = None
    # Stage 1b: rpc_receipt — wall-clock when this process first SAW
    # the event (logsSubscribe notification arrival for WSS, or
    # getSignaturesForAddress response for polling). Stamped by the
    # listener BEFORE any getTransaction confirmation wait, so the
    # delta (t_rpc_receipt - t_detect) reflects pure detection
    # staleness and isolates the WSS-vs-polling structural advantage
    # from the commitment-wait penalty.
    t_rpc_receipt: Optional[float] = None
    # Stage 2: evaluate start — sniper_engine._evaluate_target entry
    t_eval_start: float = field(default_factory=time.perf_counter)
    # Stage 3: safety check entry / exit
    t_safety_start: Optional[float] = None
    t_safety_done: Optional[float] = None
    # Stage 4: broadcast (execute_buy) entry / exit
    t_broadcast_start: Optional[float] = None
    t_broadcast_done: Optional[float] = None
    # Outcome: 'success', 'failed', 'rejected_filter', 'rejected_safety'
    outcome: str = "pending"
    # Idempotency guard so we never double-log a single opportunity.
    _emitted: bool = False

    def stamp(self, marker: str) -> None:
        """Stamp the given marker name with the current monotonic time."""
        if hasattr(self, marker):
            setattr(self, marker, time.perf_counter())

    def emit(self) -> None:
        """Emit one structured log line. Safe to call multiple times
        (idempotent); typically called from _execute_snipe at the end
        of the lifecycle."""
        if self._emitted:
            return
        self._emitted = True
        try:
            def _ms(start: Optional[float], end: Optional[float]) -> str:
                if start is None or end is None:
                    return "—"
                return f"{(end - start) * 1000:.1f}ms"

            # Detection-staleness metric: block-production → process-receipt.
            # The headline number for the WSS-vs-polling A/B; isolated
            # from the getTransaction commitment wait that previously
            # dominated total_ms.
            detect_to_rpc = _ms(self.t_detect, self.t_rpc_receipt)
            rpc_to_eval = _ms(self.t_rpc_receipt, self.t_eval_start)
            detect_to_eval = _ms(self.t_detect, self.t_eval_start)
            eval_to_safety = _ms(self.t_eval_start, self.t_safety_start)
            safety_dur = _ms(self.t_safety_start, self.t_safety_done)
            safety_to_broadcast = _ms(self.t_safety_done, self.t_broadcast_start)
            broadcast_dur = _ms(self.t_broadcast_start, self.t_broadcast_done)
            total = _ms(self.t_detect or self.t_eval_start,
                        self.t_broadcast_done)

            token_disp = (self.token_address[:16] + "...") if self.token_address else "—"
            logger.info(
                "⏱️ SNIPE TIMING %s %s outcome=%s | "
                "detect→rpc=%s rpc→eval=%s detect→eval=%s eval→safety=%s "
                "safety=%s safety→broadcast=%s broadcast=%s | total=%s",
                self.chain, token_disp, self.outcome,
                detect_to_rpc, rpc_to_eval,
                detect_to_eval, eval_to_safety, safety_dur,
                safety_to_broadcast, broadcast_dur, total,
            )
        except Exception as e:
            # Never let instrumentation break trading.
            logger.debug(f"timing emit failed (non-fatal): {e}")

    def to_metadata_dict(self) -> dict:
        """Return timing deltas (in milliseconds) as a JSON-friendly dict
        for persistence in sniper_trades.metadata. None for unstamped stages.
        Enables historical P50/P95 dashboards beyond the per-event log line."""
        def _delta_ms(start: Optional[float], end: Optional[float]):
            if start is None or end is None:
                return None
            return round((end - start) * 1000.0, 2)

        return {
            'outcome': self.outcome,
            # Detection-staleness: block production → process receipt.
            # Headline WSS-vs-polling A/B metric.
            'detect_to_rpc_receipt_ms': _delta_ms(self.t_detect, self.t_rpc_receipt),
            # In-process queue cost from receipt to engine pickup.
            'rpc_receipt_to_eval_ms': _delta_ms(self.t_rpc_receipt, self.t_eval_start),
            'detect_to_eval_ms': _delta_ms(self.t_detect, self.t_eval_start),
            'eval_to_safety_ms': _delta_ms(self.t_eval_start, self.t_safety_start),
            'safety_ms': _delta_ms(self.t_safety_start, self.t_safety_done),
            'safety_to_broadcast_ms': _delta_ms(self.t_safety_done, self.t_broadcast_start),
            'broadcast_ms': _delta_ms(self.t_broadcast_start, self.t_broadcast_done),
            'total_ms': _delta_ms(self.t_detect or self.t_eval_start, self.t_broadcast_done),
        }


def parse_iso_to_perf_counter(iso_ts: str) -> Optional[float]:
    """Convert an ISO wall-clock timestamp to an approximate
    perf_counter value. We anchor on 'now' so the delta from
    detect→eval is approximately (now_perf - (now_wall - iso_wall)).

    Only meaningful for sub-second resolution; do not trust beyond ~5s
    accuracy because perf_counter and wall-clock drift differently."""
    if not iso_ts:
        return None
    try:
        from datetime import datetime, timezone
        iso_dt = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        if iso_dt.tzinfo is not None:
            now_wall = datetime.now(timezone.utc)
        else:
            now_wall = datetime.utcnow()
        delta_seconds = (now_wall - iso_dt).total_seconds()
        return time.perf_counter() - delta_seconds
    except Exception:
        return None
