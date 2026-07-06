"""
core/llm_budget.py — HARD, bot-wide daily cap on paid LLM API calls, now with
per-module RESERVATIONS so one module cannot starve another.

ONE global counter shared by every module that calls a paid LLM
(Anthropic / OpenAI): the Advisor (KAP classifier + advice rationale) and the
AI Analysis module (sentiment). No combination of bugs, tight loops, retries, or
cycle frequency can exceed a fixed number of paid calls per UTC day. When the
budget is exhausted, every caller fails soft (rule-based / neutral) and makes NO
API call until the UTC day rolls over.

Why this exists: a runaway KAP re-classification loop (a 60s worker re-LLM'ing
the same un-stored disclosures every cycle, and retrying even after Anthropic
returned "credit balance too low") burned a full day of paid API credit in
~3 hours. This module is the single backstop that makes that impossible.

Wave-F5 redesign (the starvation fix): a single global counter let the Advisor
drain the whole day's budget within ~25 min of midnight UTC, leaving the AI
Analysis module with ~1 stale reading/day for 20 days. The fix keeps the hard
global ceiling but carves it into per-module buckets:

  * RESERVATIONS — a module's first N calls are GUARANTEED (drawn from its own
    reservation), so nobody can starve it:  ai=40, kap_classify=30.
  * CEILINGS      — a module may use AT MOST N calls:  advisor(advice*)=60.
  * SHARED POOL   — the remainder (cap - sum(reservations)) is drawn by any
    module after it exhausts its own reservation, first-come-first-served,
    still subject to each module's ceiling.

The default global cap is raised 25 -> 150 (measured cost ~$0.001/call ⇒ the
whole 150 is ~$0.15/day). The env override still wins, so ops can pin it.

Enforcement is an in-process, lock-protected counter keyed on the UTC date,
persisted to a small JSON file so a process restart / crash-loop cannot reset
the count and re-spend within the same day. Each module runs as its own
subprocess, so the JSON file (default logs/.llm_budget.json) is the shared,
cross-process source of truth. Concurrent writers use an atomic replace; an
occasional lost increment is acceptable for an approximate safety cap.

Global-cap resolution (first hit wins):
  1. explicit `cap=` argument
  2. env  BOT_LLM_DAILY_MAX_CALLS
  3. config['llm_daily_max_calls'] or config['advisor_llm_daily_max_calls']
  4. default 150
A cap <= 0 means UNLIMITED (operator explicitly disabled the gate).

Reservation/ceiling resolution is fail-soft: missing knobs fall back to the
built-in defaults below. Optional config overrides:
  config['llm_reserve_ai'], config['llm_reserve_kap_classify'],
  config['llm_cap_advisor'].

Persistence: the JSON file stays readable by old code — top-level {"day",
"count"} are preserved (count == total across all buckets); new code adds a
{"counts": {bucket: n}} map. An old-format file (no "counts") is migrated on
read by attributing its legacy count to a shared "_legacy" bucket.

Self-check: ``python -m core.llm_budget``
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_DEFAULT_DAILY_MAX = 150  # Wave-F5: raised 25 -> 150 (~$0.15/day at measured cost)
_ENV_KEY = "BOT_LLM_DAILY_MAX_CALLS"
_CONFIG_KEYS = ("llm_daily_max_calls", "advisor_llm_daily_max_calls")
_BUDGET_FILE = Path(os.getenv("LLM_BUDGET_FILE", "logs/.llm_budget.json"))

# Per-module GUARANTEED reservations (a module's first N calls always succeed).
_DEFAULT_RESERVATIONS: Dict[str, int] = {"ai": 40, "kap_classify": 30}
# Per-module CEILINGS (a module may use at most N calls total per day).
_DEFAULT_CEILINGS: Dict[str, int] = {"advisor": 60}
# config-key overrides -> bucket (fail-soft; missing -> default above)
_RESERVE_CONFIG_KEYS = {"ai": "llm_reserve_ai", "kap_classify": "llm_reserve_kap_classify"}
_CEILING_CONFIG_KEYS = {"advisor": "llm_cap_advisor"}
_LEGACY_BUCKET = "_legacy"

_lock = threading.Lock()
_logger = logging.getLogger("llm_budget")

_state = {"day": "", "count": 0, "counts": {}}  # type: ignore[var-annotated]
_warned: Dict[str, str] = {}  # bucket -> day the "budget reached" WARNING fired


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_from_disk() -> dict:
    try:
        with _BUDGET_FILE.open("r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _counts_for_today(disk: dict, today: str) -> Dict[str, int]:
    """Return the per-bucket counts map for `today`, migrating an old-format
    file (no "counts") by attributing its legacy count to the shared bucket."""
    if disk.get("day") != today:
        return {}
    counts = disk.get("counts")
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items() if _is_int(v)}
    # Old format: only a scalar "count". Preserve it as legacy shared spend.
    legacy = disk.get("count", 0)
    return {_LEGACY_BUCKET: int(legacy)} if _is_int(legacy) and int(legacy) > 0 else {}


def _is_int(v) -> bool:
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


def _save_to_disk(day: str, counts: Dict[str, int]) -> None:
    try:
        _BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _BUDGET_FILE.with_suffix(".tmp")
        total = int(sum(counts.values()))
        with tmp.open("w") as fh:
            # `count` is kept for backward-compat with old readers.
            json.dump({"day": day, "count": total, "counts": counts}, fh)
        tmp.replace(_BUDGET_FILE)  # atomic
    except Exception:
        pass  # best-effort; the in-memory counter still enforces within the run


def _resolve_cap(config: Optional[dict], explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    env = os.getenv(_ENV_KEY)
    if env not in (None, ""):
        try:
            return int(env)
        except (TypeError, ValueError):
            pass
    if config:
        for k in _CONFIG_KEYS:
            if k in config:
                try:
                    return int(config[k])
                except (TypeError, ValueError):
                    pass
    return _DEFAULT_DAILY_MAX


def _resolve_int(config: Optional[dict], key: str, default: int) -> int:
    """Fail-soft int resolver for reservation/ceiling overrides."""
    if config and key in config:
        try:
            return max(0, int(config[key]))
        except (TypeError, ValueError):
            pass
    return default


def _bucket_for(kind: str, module: Optional[str]) -> str:
    """Map an explicit module or a `kind` label to a budget bucket.

    Callers pass kinds like ai_anthropic / ai_legacy_openai (module=ai),
    kap_classify (module=kap_classify), advice_rationale / advice_openai
    (module=advisor). Unknown kinds share the pool as bucket 'other'.
    """
    if module:
        return str(module)
    k = (kind or "").lower()
    if k.startswith("ai"):
        return "ai"
    if k.startswith("kap"):
        return "kap_classify"
    if k.startswith("advice") or k.startswith("advisor"):
        return "advisor"
    return "other"


def try_consume(
    config: Optional[dict] = None,
    *,
    kind: str = "llm",
    module: Optional[str] = None,
    cap: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Atomically reserve one paid-LLM call against today's budget, honoring the
    per-module reservation/ceiling rules described in the module docstring.

    Returns
    -------
    True  : permitted; the counter was incremented.
    False : denied — the caller MUST fall back to rule-based / neutral output
            and make NO paid API call. Denial reasons: the module's ceiling is
            reached, or its reservation is spent AND the shared pool is empty.

    Backward-compatible: `kind` is unchanged; `module` is optional and derived
    from `kind` when omitted. cap <= 0 disables the gate (always True).
    """
    lg = log or _logger
    resolved_cap = _resolve_cap(config, cap)
    if resolved_cap <= 0:
        return True  # gate disabled by operator

    bucket = _bucket_for(kind, module)
    reservations = {
        b: _resolve_int(config, _RESERVE_CONFIG_KEYS.get(b, ""), d)
        for b, d in _DEFAULT_RESERVATIONS.items()
    }
    ceilings = {
        b: _resolve_int(config, _CEILING_CONFIG_KEYS.get(b, ""), d)
        for b, d in _DEFAULT_CEILINGS.items()
    }

    with _lock:
        today = _today()
        disk = _load_from_disk()
        counts = _counts_for_today(disk, today)
        if disk.get("day") != today:
            _warned.clear()  # new UTC day -> allow warnings to fire again

        total = int(sum(counts.values()))
        used = int(counts.get(bucket, 0))

        # If reservations don't fit under a tiny operator cap, degrade to a
        # simple global cap (ignore reservations/ceilings) — never over-spend.
        reservations_fit = sum(reservations.values()) <= resolved_cap

        allowed = False
        deny_reason = ""
        if not reservations_fit:
            allowed = total < resolved_cap
            deny_reason = "global cap"
        else:
            ceiling = ceilings.get(bucket)
            if ceiling is not None and used >= ceiling:
                allowed = False
                deny_reason = f"module ceiling {ceiling}"
            elif used < reservations.get(bucket, 0):
                allowed = True  # within this module's guaranteed reservation
            else:
                # Beyond reservation (or no reservation): draw from shared pool.
                shared_size = resolved_cap - sum(reservations.values())
                shared_used = sum(
                    max(0, c - reservations.get(b, 0)) for b, c in counts.items()
                )
                if shared_used < shared_size:
                    allowed = True
                else:
                    allowed = False
                    deny_reason = "shared pool exhausted"

        if not allowed:
            counts.setdefault(bucket, used)
            _state.update({"day": today, "count": total, "counts": dict(counts)})
            if _warned.get(bucket) != today:
                lg.warning(
                    "[llm_budget] Daily paid-LLM budget denied for module=%s "
                    "(kind=%s): %s. total=%d/%d used_by_module=%d for %s; "
                    "falling back to rule-based/neutral output — NO paid API "
                    "call. Set env %s (<=0 disables the cap).",
                    bucket, kind, deny_reason, total, resolved_cap, used, today,
                    _ENV_KEY,
                )
                _warned[bucket] = today
            return False

        counts[bucket] = used + 1
        _save_to_disk(today, counts)
        _state.update({
            "day": today,
            "count": int(sum(counts.values())),
            "counts": dict(counts),
        })
        return True


def snapshot() -> dict:
    """Return {'day', 'count', 'counts'} for diagnostics / dashboard surfacing."""
    with _lock:
        return {
            "day": _state.get("day", ""),
            "count": _state.get("count", 0),
            "counts": dict(_state.get("counts", {})),
        }


if __name__ == "__main__":  # tiny self-check (no network, no DB)
    import tempfile
    logging.basicConfig(level=logging.INFO)
    _BUDGET_FILE = Path(tempfile.mkdtemp()) / ".llm_budget.json"

    # cap=0 disables the gate.
    assert try_consume(cap=0) is True

    # Small cap, reservations don't fit -> simple global cap of 3.
    r = [try_consume(cap=3, kind="ai_anthropic") for _ in range(5)]
    assert r == [True, True, True, False, False], r

    # Reservation isolation: a fresh day/file, tiny reserved buckets.
    _BUDGET_FILE = Path(tempfile.mkdtemp()) / ".llm_budget.json"
    cfg = {
        "llm_daily_max_calls": 10,
        "llm_reserve_ai": 4,
        "llm_reserve_kap_classify": 2,
        "llm_cap_advisor": 3,
    }
    # Advisor drains everything it can (ceiling 3, shared pool = 10-6 = 4).
    adv = [try_consume(cfg, kind="advice_rationale") for _ in range(5)]
    assert adv == [True, True, True, False, False], adv  # ceiling stops at 3
    # AI still has its full guaranteed reservation of 4 despite advisor spend.
    ai = [try_consume(cfg, kind="ai_anthropic") for _ in range(4)]
    assert all(ai), ai
    # kap_classify still has its guaranteed 2.
    kap = [try_consume(cfg, kind="kap_classify") for _ in range(2)]
    assert all(kap), kap
    print("core.llm_budget self-check OK; snapshot=", snapshot())
