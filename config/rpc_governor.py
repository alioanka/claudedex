"""
RPC Governor — smart, self-regulating per-(module, provider) request governor.

Wave-F7. Problem (docs/agents/wave-f6/01_rate_limiting.md): the pool engine
had ONE TokenBucket per provider_type shared by ALL modules, so a single
spamming module (sniper tx-enrich ~10/s, copy polling) drained the bucket and
every module got throttled — with no attribution, no adaptive backoff, no
priority, no coalescing.

This module is the PURE core (stdlib only — no aiohttp, no DB, no imports
from pool_engine; pool_engine imports *us*). It provides:

1. MODULE ATTRIBUTION — a ``current_module`` contextvar plus process-name
   inference (every module runs as its own subprocess: ``main_sniper.py`` /
   ``modules/copy_trading/...`` → 'sniper' / 'copy_trading'), so ZERO call
   sites need editing. ``set_calling_module()`` / ``governed_call()`` allow
   explicit overrides. Best-effort: attribution can never crash a call.

2. ROLLING ACCOUNTING — per-(module, provider) sliding windows (10s / 60s /
   1h) of requests, successes, 429s and method-weighted CREDIT cost
   (Helius enhanced-tx=100, getHealth=1, default=1; DB-overridable table).

3. ADAPTIVE AIMD LIMITER — each (module, provider) pair owns a token bucket.
   A 429 attributed to the pair MULTIPLICATIVELY decreases its rps
   (× ``decrease_factor``, floor ``min_rps``); a sustained clean window
   ADDITIVELY recovers (+``recover_step`` rps/s up to the provider ceiling).
   A spammy module therefore converges to a trickle while quiet modules keep
   their rate — the more 429s you cause, the faster you decay.

4. SPAM CLAMP — a module whose 10s request rate or 60s credit burn exceeds
   ``module_max_rps`` / ``module_credit_budget_per_min`` is hard-clamped to
   the floor with ONE named WARN per minute
   ("governor: throttling sniper on HELIUS_API 41.0→0.2 rps ...").

5. PRIORITY CLASSES — 'execution' calls (priority kwarg or method allowlist)
   draw from the FULL provider ceiling with a short wait cap, while
   'background' calls draw from a bucket sized at ceiling×(1−exec_reserve).
   A scan can exhaust only the background share; a live send always has the
   reserved headroom.

6. REQUEST COALESCING — ``coalesce(key, factory)`` shares one in-flight
   result (plus a short TTL micro-cache) among identical concurrent
   idempotent reads. Opt-in per call site.

FAIL-SOFT CONTRACT: the governor only ever DELAYS requests (never blocks
forever — every wait is capped), never raises into a caller, and any internal
error degrades to a no-op so the legacy shared limiter in pool_engine remains
the safety net. It never touches trade logic.

Self-test (offline, no network/DB):  python -m config.rpc_governor
"""

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("PoolEngine.Governor")

# =========================================================================
# Module attribution (contextvar + process inference)
# =========================================================================

current_module: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "rpc_governor_current_module", default=None
)
current_priority: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "rpc_governor_current_priority", default=None
)

_process_module: Optional[str] = None


def _infer_process_module() -> str:
    """Best-effort module name for THIS process.

    Every trading module runs as its own subprocess launched by main.py as
    ``python modules/<dir>/main_<x>.py`` — the script path is the module
    identity, so attribution needs zero edits in module mains. Cached.
    """
    global _process_module
    if _process_module is not None:
        return _process_module
    name = "unknown"
    try:
        argv0 = (sys.argv[0] or "").replace("\\", "/")
        m = re.search(r"/modules/([A-Za-z0-9_]+)/", argv0)
        if m:
            name = m.group(1)
        else:
            base = argv0.rsplit("/", 1)[-1]
            m = re.match(r"main_([A-Za-z0-9_]+)\.py$", base)
            if m:
                name = m.group(1)
            elif base == "main.py":
                name = "orchestrator"
            elif base.endswith(".py"):
                name = base[:-3] or "unknown"
    except Exception:
        name = "unknown"
    _process_module = name
    return name


def set_calling_module(name: Optional[str]) -> None:
    """Set the attributed module for the current async context. Never raises."""
    try:
        current_module.set(str(name) if name else None)
    except Exception:
        pass


def get_calling_module() -> str:
    """Attributed module: contextvar → process inference → 'unknown'."""
    try:
        val = current_module.get()
        if val:
            return val
    except Exception:
        pass
    return _infer_process_module()


@contextlib.contextmanager
def governed_call(module: Optional[str] = None, priority: Optional[str] = None):
    """Attribute + prioritise every pool call inside the block::

        with governed_call(priority='execution'):
            url = await pool.get_endpoint('SOLANA_RPC')
    """
    tok_m = tok_p = None
    try:
        if module is not None:
            tok_m = current_module.set(module)
        if priority is not None:
            tok_p = current_priority.set(priority)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            if tok_m is not None:
                current_module.reset(tok_m)
            if tok_p is not None:
                current_priority.reset(tok_p)
        except Exception:
            pass


# =========================================================================
# Primitives
# =========================================================================

class WindowCounter:
    """Bucketed sliding-window counter (bounded memory, monotonic clock)."""

    __slots__ = ("span", "bucket_s", "_buckets")

    def __init__(self, span_s: float, bucket_s: float):
        self.span = float(span_s)
        self.bucket_s = max(0.05, float(bucket_s))
        self._buckets: deque = deque()  # (bucket_start_mono, value)

    def add(self, value: float = 1.0, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        start = now - (now % self.bucket_s)
        if self._buckets and self._buckets[-1][0] == start:
            self._buckets[-1][1] += value
        else:
            self._buckets.append([start, value])
        self._evict(now)

    def _evict(self, now: float) -> None:
        horizon = now - self.span
        while self._buckets and self._buckets[0][0] < horizon:
            self._buckets.popleft()

    def total(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        self._evict(now)
        return sum(v for _, v in self._buckets)

    def rate(self, now: Optional[float] = None) -> float:
        """Average events/second over the window span."""
        return self.total(now) / self.span if self.span > 0 else 0.0


class MonoBucket:
    """Minimal monotonic token bucket with mutable rate. Never raises."""

    __slots__ = ("rate", "capacity", "_tokens", "_last", "_lock")

    def __init__(self, rate: float, capacity: Optional[float] = None):
        self.rate = max(0.0, float(rate))
        self.capacity = max(1.0, float(capacity if capacity is not None else max(1.0, rate)))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def set_rate(self, rate: float, capacity: Optional[float] = None) -> None:
        self.rate = max(0.0, float(rate))
        if capacity is not None:
            self.capacity = max(1.0, float(capacity))
            self._tokens = min(self._tokens, self.capacity)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    async def acquire(self, tokens: float = 1.0, max_wait: float = 15.0) -> float:
        """Wait for tokens, capped at ``max_wait`` seconds (fail-soft: after
        the cap the call proceeds anyway — the governor paces, it never
        starves). Returns the seconds actually waited."""
        if self.rate <= 0:
            return 0.0
        tokens = min(tokens, self.capacity)
        waited = 0.0
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                need = deficit / self.rate if self.rate > 0 else 0.05
            step = min(max(0.001, need), max(0.0, max_wait - waited))
            if step <= 0:
                # Wait budget exhausted — proceed unpaid (fail-soft).
                return waited
            await asyncio.sleep(step)
            waited += step


# =========================================================================
# Per-pair / per-provider state
# =========================================================================

@dataclass
class PairStats:
    """Sliding-window accounting for one (module, provider) pair."""
    req_10s: WindowCounter = field(default_factory=lambda: WindowCounter(10, 1))
    req_60s: WindowCounter = field(default_factory=lambda: WindowCounter(60, 5))
    req_1h: WindowCounter = field(default_factory=lambda: WindowCounter(3600, 300))
    credit_60s: WindowCounter = field(default_factory=lambda: WindowCounter(60, 5))
    credit_1h: WindowCounter = field(default_factory=lambda: WindowCounter(3600, 300))
    ok_60s: WindowCounter = field(default_factory=lambda: WindowCounter(60, 5))
    r429_60s: WindowCounter = field(default_factory=lambda: WindowCounter(60, 5))
    r429_1h: WindowCounter = field(default_factory=lambda: WindowCounter(3600, 300))


@dataclass
class PairState:
    module: str
    provider: str
    rps: float                       # current AIMD rate
    bucket: MonoBucket
    stats: PairStats = field(default_factory=PairStats)
    clamped: bool = False
    last_429_mono: float = 0.0
    last_increase_mono: float = 0.0
    last_warn_mono: float = 0.0
    total_throttle_events: int = 0


@dataclass
class ProviderState:
    provider: str
    ceiling_rps: float
    exec_bucket: MonoBucket          # full ceiling — execution class
    bg_bucket: MonoBucket            # ceiling × (1 − exec_reserve) — background


# =========================================================================
# Governor
# =========================================================================

_EXEC_PRIORITIES = ("execution", "exec", "quote", "critical")

_DEFAULT_METHOD_COSTS = {
    "default": 1,
    "gethealth": 1,
    "getslot": 1,
    "eth_blocknumber": 1,
    "getbalance": 1,
    "gettransaction": 10,
    "getsignaturesforaddress": 10,
    "getparsedtransaction": 50,
    "getprogramaccounts": 25,
    "eth_getlogs": 25,
    "enhanced_tx": 100,
    "parse_transactions": 100,
}

_DEFAULT_EXEC_METHODS = (
    "sendtransaction", "sendrawtransaction", "eth_sendrawtransaction",
    "simulatetransaction", "eth_estimategas", "quote", "swap",
)


class RpcGovernor:
    """The self-regulating per-(module, provider) request governor.

    One instance per process (each module is its own subprocess, so the pair
    key is effectively (this-process-module, provider) plus any explicit
    ``governed_call`` overrides). Cross-process fairness emerges from the
    AIMD feedback: when the aggregate exceeds a provider's real quota, the
    modules actually causing 429s report more of them and decay fastest.
    """

    def __init__(self):
        self.enabled = True
        # AIMD knobs (all DB-overridable via configure())
        self.min_rps = 0.2
        self.decrease_factor = 0.5
        self.recover_step = 0.25          # rps added per clean second
        self.clean_window_s = 20.0        # no-429 window before recovery
        # Spam clamp
        self.module_max_rps = 10.0
        self.module_credit_budget_per_min = 3000.0
        # Priority split
        self.exec_reserve_pct = 30.0
        self.max_wait_s = 15.0            # background wait cap
        self.exec_max_wait_s = 2.0        # execution wait cap
        # Provider ceiling default (matches pool_engine._default_rps)
        self.default_ceiling_rps = 8.0
        # Coalescing
        self.coalesce_ttl_s = 2.0

        self.method_costs: Dict[str, float] = dict(_DEFAULT_METHOD_COSTS)
        self.exec_methods: Tuple[str, ...] = _DEFAULT_EXEC_METHODS

        self._pairs: Dict[Tuple[str, str], PairState] = {}
        self._providers: Dict[str, ProviderState] = {}
        self._coalesce_map: Dict[str, Tuple[asyncio.Task, float]] = {}
        self._warn_throttle_s = 60.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    _FLOAT_KEYS = {
        "governor_min_rps": "min_rps",
        "governor_decrease_factor": "decrease_factor",
        "governor_recover_step": "recover_step",
        "governor_clean_window_s": "clean_window_s",
        "governor_module_max_rps": "module_max_rps",
        "governor_module_credit_budget_per_min": "module_credit_budget_per_min",
        "governor_exec_reserve_pct": "exec_reserve_pct",
        "governor_max_wait_s": "max_wait_s",
        "governor_exec_max_wait_s": "exec_max_wait_s",
        "governor_default_ceiling_rps": "default_ceiling_rps",
        "governor_coalesce_ttl_s": "coalesce_ttl_s",
    }

    def configure(self, settings: Dict[str, Any]) -> None:
        """Apply string-valued knobs (e.g. straight from config_settings).
        Unknown keys are ignored; bad values are ignored; never raises."""
        if not settings:
            return
        try:
            for key, value in settings.items():
                try:
                    if key == "governor_enabled":
                        self.enabled = str(value).strip().lower() in ("true", "1", "yes")
                    elif key in self._FLOAT_KEYS:
                        setattr(self, self._FLOAT_KEYS[key], max(0.0, float(value)))
                    elif key == "governor_method_costs":
                        table = json.loads(value) if isinstance(value, str) else dict(value)
                        merged = dict(_DEFAULT_METHOD_COSTS)
                        merged.update({str(k).lower(): float(v) for k, v in table.items()})
                        self.method_costs = merged
                    elif key == "governor_exec_methods":
                        seq = json.loads(value) if isinstance(value, str) else list(value)
                        self.exec_methods = tuple(str(m).lower() for m in seq)
                except Exception as e:
                    logger.debug(f"governor: ignored bad knob {key}={value!r}: {e}")
            # Sanity floors so a bad config can't wedge everything.
            self.min_rps = max(0.05, self.min_rps)
            self.decrease_factor = min(0.95, max(0.1, self.decrease_factor or 0.5))
            self.exec_reserve_pct = min(90.0, self.exec_reserve_pct)
            # Re-derive provider buckets under the new reserve split.
            for ps in self._providers.values():
                self._apply_provider_rates(ps, ps.ceiling_rps)
            logger.info(
                "governor configured: enabled=%s min_rps=%.2f decrease=%.2f "
                "recover=+%.2f/s clean=%.0fs clamp=%.1frps/%.0fcr/min "
                "exec_reserve=%.0f%%",
                self.enabled, self.min_rps, self.decrease_factor,
                self.recover_step, self.clean_window_s, self.module_max_rps,
                self.module_credit_budget_per_min, self.exec_reserve_pct,
            )
        except Exception as e:
            logger.debug(f"governor configure failed (defaults kept): {e}")

    def set_provider_ceiling(self, provider: str, rps: float,
                             burst: Optional[float] = None) -> None:
        """Provider quota ceiling (fed by pool_engine.configure_rate_limiter).
        Pairs on this provider recover up to (and start at) this ceiling."""
        try:
            rps = max(self.min_rps, float(rps))
            ps = self._providers.get(provider)
            if ps is None:
                ps = self._make_provider(provider, rps, burst)
            else:
                self._apply_provider_rates(ps, rps, burst)
            for pair in self._pairs.values():
                if pair.provider == provider and pair.rps > rps:
                    pair.rps = rps
                    pair.bucket.set_rate(rps)
        except Exception as e:
            logger.debug(f"set_provider_ceiling({provider}) ignored: {e}")

    def _make_provider(self, provider: str, ceiling: Optional[float] = None,
                       burst: Optional[float] = None) -> ProviderState:
        ceiling = float(ceiling if ceiling is not None else self.default_ceiling_rps)
        ps = ProviderState(
            provider=provider,
            ceiling_rps=ceiling,
            exec_bucket=MonoBucket(ceiling, burst),
            bg_bucket=MonoBucket(ceiling, burst),
        )
        self._apply_provider_rates(ps, ceiling, burst)
        self._providers[provider] = ps
        return ps

    def _apply_provider_rates(self, ps: ProviderState, ceiling: float,
                              burst: Optional[float] = None) -> None:
        ps.ceiling_rps = ceiling
        bg_share = max(0.0, 1.0 - self.exec_reserve_pct / 100.0)
        ps.exec_bucket.set_rate(ceiling, burst)
        ps.bg_bucket.set_rate(max(self.min_rps, ceiling * bg_share),
                              burst if burst is None else max(1.0, burst * bg_share))

    # ------------------------------------------------------------------
    # Pair lookup / cost / priority
    # ------------------------------------------------------------------

    def _pair(self, module: str, provider: str) -> PairState:
        key = (module, provider)
        pair = self._pairs.get(key)
        if pair is None:
            ps = self._providers.get(provider) or self._make_provider(provider)
            pair = PairState(
                module=module, provider=provider,
                rps=ps.ceiling_rps,
                bucket=MonoBucket(ps.ceiling_rps),
            )
            self._pairs[key] = pair
        return pair

    def cost_for(self, method: Optional[str]) -> float:
        if not method:
            return self.method_costs.get("default", 1.0)
        return float(self.method_costs.get(str(method).lower(),
                                           self.method_costs.get("default", 1.0)))

    def _is_execution(self, priority: Optional[str], method: Optional[str]) -> bool:
        try:
            pr = priority or current_priority.get()
        except Exception:
            pr = priority
        if pr and str(pr).lower() in _EXEC_PRIORITIES:
            return True
        if method and str(method).lower() in self.exec_methods:
            return True
        return False

    # ------------------------------------------------------------------
    # The pacing seam
    # ------------------------------------------------------------------

    async def acquire(self, provider: str, module: Optional[str] = None,
                      priority: Optional[str] = None, method: Optional[str] = None,
                      tokens: Optional[float] = None) -> float:
        """Pace one outbound request. Returns seconds waited. NEVER raises
        and NEVER waits beyond the class wait cap (fail-soft pacing)."""
        try:
            if not self.enabled:
                return 0.0
            module = module or get_calling_module()
            pair = self._pair(module, provider)
            ps = self._providers[provider]
            cost = float(tokens) if tokens is not None else self.cost_for(method)
            is_exec = self._is_execution(priority, method)

            # Accounting first — the clamp must see the spam even if pacing
            # is bypassed by the wait cap.
            now = time.monotonic()
            pair.stats.req_10s.add(1, now)
            pair.stats.req_60s.add(1, now)
            pair.stats.req_1h.add(1, now)
            pair.stats.credit_60s.add(cost, now)
            pair.stats.credit_1h.add(cost, now)

            self._maybe_clamp(pair, now, is_exec)
            self._maybe_recover(pair, now)

            wait_cap = self.exec_max_wait_s if is_exec else self.max_wait_s
            waited = await pair.bucket.acquire(cost, max_wait=wait_cap)
            remaining = max(0.0, wait_cap - waited)
            cls_bucket = ps.exec_bucket if is_exec else ps.bg_bucket
            waited += await cls_bucket.acquire(cost, max_wait=remaining)
            return waited
        except Exception as e:
            logger.debug(f"governor.acquire({provider}) no-op: {e}")
            return 0.0

    def _maybe_clamp(self, pair: PairState, now: float, is_exec: bool) -> None:
        """Hard-clamp a spamming module and WARN (once/min, named)."""
        req_rate = pair.stats.req_10s.rate(now)
        credit_min = pair.stats.credit_60s.total(now)
        over = (req_rate > self.module_max_rps
                or credit_min > self.module_credit_budget_per_min)
        if over and not is_exec:
            old_rps = pair.rps
            if not pair.clamped or pair.rps > self.min_rps:
                pair.clamped = True
                pair.rps = self.min_rps
                pair.bucket.set_rate(pair.rps)
                pair.total_throttle_events += 1
            if now - pair.last_warn_mono >= self._warn_throttle_s:
                pair.last_warn_mono = now
                r429 = self._rate429_pct(pair, now)
                logger.warning(
                    "governor: throttling %s on %s %.1f→%.1f rps "
                    "(req-rate %.1f/s, credits %.0f/min, 429-rate %.0f%%)",
                    pair.module, pair.provider, old_rps, pair.rps,
                    req_rate, credit_min, r429,
                )
        elif pair.clamped and not over:
            # Under budget again — release the clamp; AIMD recovery takes
            # over from the floor (rps stays low until the clean window).
            pair.clamped = False

    def _maybe_recover(self, pair: PairState, now: float) -> None:
        """Additive-increase after a sustained clean (no-429) window."""
        if pair.clamped:
            return
        ceiling = self._providers[pair.provider].ceiling_rps
        if pair.rps >= ceiling:
            return
        if now - pair.last_429_mono < self.clean_window_s:
            return
        if now - pair.last_increase_mono < 1.0:
            return
        pair.last_increase_mono = now
        pair.rps = min(ceiling, pair.rps + self.recover_step)
        pair.bucket.set_rate(pair.rps)

    # ------------------------------------------------------------------
    # Outcome feed (from pool_engine report_* seam)
    # ------------------------------------------------------------------

    def on_rate_limit(self, provider: str, module: Optional[str] = None) -> None:
        """A 429/quota event attributed to (module, provider): AIMD decrease."""
        try:
            module = module or get_calling_module()
            pair = self._pair(module, provider)
            now = time.monotonic()
            pair.stats.r429_60s.add(1, now)
            pair.stats.r429_1h.add(1, now)
            pair.last_429_mono = now
            old = pair.rps
            pair.rps = max(self.min_rps, pair.rps * self.decrease_factor)
            if pair.rps != old:
                pair.bucket.set_rate(pair.rps)
                pair.total_throttle_events += 1
                logger.info(
                    "governor: AIMD decrease %s on %s %.2f→%.2f rps (429)",
                    module, provider, old, pair.rps,
                )
        except Exception as e:
            logger.debug(f"governor.on_rate_limit({provider}) no-op: {e}")

    def on_success(self, provider: str, module: Optional[str] = None) -> None:
        try:
            module = module or get_calling_module()
            pair = self._pair(module, provider)
            now = time.monotonic()
            pair.stats.ok_60s.add(1, now)
            self._maybe_recover(pair, now)
        except Exception as e:
            logger.debug(f"governor.on_success({provider}) no-op: {e}")

    # ------------------------------------------------------------------
    # Request coalescing (idempotent reads only — opt-in per call site)
    # ------------------------------------------------------------------

    async def coalesce(self, key: str, factory, ttl: Optional[float] = None):
        """Share one in-flight result among identical concurrent reads.

        ``factory`` is a zero-arg callable returning a coroutine. The first
        caller creates the task; concurrent (and TTL-window) callers await
        the SAME task. Errors are propagated to every waiter and NOT cached.
        Safe ONLY for idempotent reads (getLogs/getTransaction/price fetch).
        """
        ttl = self.coalesce_ttl_s if ttl is None else float(ttl)
        now = time.monotonic()
        try:
            entry = self._coalesce_map.get(key)
            if entry is not None:
                task, expiry = entry
                if now < expiry and not (task.done() and task.exception() is not None):
                    return await asyncio.shield(task)
                self._coalesce_map.pop(key, None)
            task = asyncio.ensure_future(factory())
            self._coalesce_map[key] = (task, now + max(0.05, ttl))
            # Opportunistic sweep so the map stays bounded.
            if len(self._coalesce_map) > 512:
                self._coalesce_map = {
                    k: (t, e) for k, (t, e) in self._coalesce_map.items()
                    if e > now and not t.done()
                }
            try:
                return await asyncio.shield(task)
            except BaseException:
                self._coalesce_map.pop(key, None)
                raise
        except (asyncio.CancelledError,):
            raise
        except Exception:
            # Governor bookkeeping error — run the factory directly (fail-soft).
            if key in self._coalesce_map:
                self._coalesce_map.pop(key, None)
            return await factory()

    # ------------------------------------------------------------------
    # Introspection (dashboard)
    # ------------------------------------------------------------------

    def _rate429_pct(self, pair: PairState, now: float) -> float:
        n429 = pair.stats.r429_60s.total(now)
        nok = pair.stats.ok_60s.total(now)
        denom = n429 + nok
        return (n429 / denom * 100.0) if denom > 0 else 0.0

    def snapshot(self) -> List[Dict[str, Any]]:
        """Per-pair rows for the dashboard/DB flush. Never raises."""
        rows: List[Dict[str, Any]] = []
        try:
            now = time.monotonic()
            for pair in self._pairs.values():
                ps = self._providers.get(pair.provider)
                state = ("CLAMPED" if pair.clamped
                         else "THROTTLED" if ps and pair.rps < ps.ceiling_rps - 1e-9
                         else "OK")
                rows.append({
                    "module": pair.module,
                    "provider_type": pair.provider,
                    "requests_per_min": round(pair.stats.req_60s.total(now), 1),
                    "requests_per_hour": round(pair.stats.req_1h.total(now), 1),
                    "credits_per_min": round(pair.stats.credit_60s.total(now), 1),
                    "rate429_pct": round(self._rate429_pct(pair, now), 1),
                    "current_rps": round(pair.rps, 2),
                    "ceiling_rps": round(ps.ceiling_rps, 2) if ps else None,
                    "state": state,
                    "clamped": pair.clamped,
                    "throttle_events": pair.total_throttle_events,
                })
        except Exception as e:
            logger.debug(f"governor.snapshot failed: {e}")
        return rows


# =========================================================================
# Singleton access
# =========================================================================

_governor: Optional[RpcGovernor] = None


def get_governor() -> RpcGovernor:
    global _governor
    if _governor is None:
        _governor = RpcGovernor()
    return _governor


# =========================================================================
# Offline self-test:  python -m config.rpc_governor
# =========================================================================

async def _selftest() -> int:
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = ""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("rpc_governor self-test (offline, no network/DB)")

    # 1. AIMD multiplicative decrease on 429 ------------------------------
    g = RpcGovernor()
    g.set_provider_ceiling("TEST_API", 100.0)
    await g.acquire("TEST_API", module="modA")
    g.on_rate_limit("TEST_API", module="modA")
    g.on_rate_limit("TEST_API", module="modA")
    pair = g._pair("modA", "TEST_API")
    check("AIMD decrease on 429", abs(pair.rps - 25.0) < 1e-6,
          f"100→{pair.rps} rps after two 429s (×0.5 each)")
    # Quiet sibling module keeps its full rate.
    pair_b = g._pair("modB", "TEST_API")
    check("quiet module unaffected", pair_b.rps == 100.0,
          f"modB still {pair_b.rps} rps")

    # 2. Additive recovery on sustained clean window ----------------------
    g.clean_window_s = 0.15
    g.recover_step = 5.0
    pair.last_increase_mono = 0.0
    await asyncio.sleep(0.2)
    g.on_success("TEST_API", module="modA")
    rps_after_one = pair.rps
    check("additive recovery on clean window", abs(rps_after_one - 30.0) < 1e-6,
          f"25→{rps_after_one} rps (+5)")

    # 3. Spam clamp trips + names the module ------------------------------
    g2 = RpcGovernor()
    g2.module_max_rps = 5.0
    g2.set_provider_ceiling("SPAM_API", 1000.0)
    for _ in range(80):  # 80 reqs inside a 10s window → 8/s > 5/s
        await g2.acquire("SPAM_API", module="spammy")
    sp = g2._pair("spammy", "SPAM_API")
    check("spam clamp trips", sp.clamped and sp.rps == g2.min_rps,
          f"clamped={sp.clamped}, rps={sp.rps} (floor {g2.min_rps})")
    snap = {(r["module"], r["provider_type"]): r for r in g2.snapshot()}
    check("snapshot reports CLAMPED",
          snap[("spammy", "SPAM_API")]["state"] == "CLAMPED")
    # Execution-class calls are exempt from the clamp *enforcement*.
    ex_before = sp.rps
    await g2.acquire("SPAM_API", module="spammy", priority="execution")
    check("execution exempt from clamp path", sp.rps == ex_before)

    # 4. Execution priority preempts background ---------------------------
    g3 = RpcGovernor()
    g3.exec_reserve_pct = 50.0
    g3.max_wait_s = 1.0
    g3.exec_max_wait_s = 1.0
    g3.set_provider_ceiling("PRI_API", 2.0, burst=1.0)
    # Drain the background share.
    for _ in range(3):
        await g3.acquire("PRI_API", module="scanner", method="eth_getlogs")
    t0 = time.monotonic()
    await g3.acquire("PRI_API", module="trader", priority="execution")
    exec_wait = time.monotonic() - t0
    t0 = time.monotonic()
    await g3.acquire("PRI_API", module="scanner")
    bg_wait = time.monotonic() - t0
    check("execution preempts background", exec_wait < bg_wait,
          f"exec waited {exec_wait*1000:.0f}ms vs background {bg_wait*1000:.0f}ms")

    # 5. Wait cap — never blocks forever ----------------------------------
    g4 = RpcGovernor()
    g4.max_wait_s = 0.3
    g4.set_provider_ceiling("SLOW_API", 0.5, burst=1.0)
    t0 = time.monotonic()
    for _ in range(4):
        await g4.acquire("SLOW_API", module="modX")
    capped = time.monotonic() - t0
    check("wait cap bounds delay", capped < 4 * 0.3 + 0.5,
          f"4 acquires at 0.5rps took {capped:.2f}s (cap 0.3s each)")

    # 6. Coalescing dedups identical concurrent reads ---------------------
    g5 = RpcGovernor()
    calls = {"n": 0}

    async def fetch():
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return "payload"

    results = await asyncio.gather(
        *[g5.coalesce("getLogs:0xabc:blk100", fetch) for _ in range(10)]
    )
    check("coalescing dedups", calls["n"] == 1 and all(r == "payload" for r in results),
          f"10 concurrent callers → {calls['n']} upstream call")
    await g5.coalesce("other:key", fetch)
    check("distinct keys not coalesced", calls["n"] == 2)

    # 7. Method-cost weighting --------------------------------------------
    g6 = RpcGovernor()
    check("cost table weights", g6.cost_for("enhanced_tx") == 100
          and g6.cost_for("getHealth") == 1 and g6.cost_for("unknown_method") == 1)

    # 8. Attribution inference + governed_call ----------------------------
    with governed_call(module="explicit_mod", priority="execution"):
        check("contextvar attribution", get_calling_module() == "explicit_mod")
    check("process inference fallback", get_calling_module() not in (None, ""))

    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures}")
        return 1
    print("SELF-TEST PASSED (8 groups)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_selftest()))
