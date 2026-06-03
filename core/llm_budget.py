"""
core/llm_budget.py — HARD, bot-wide daily cap on paid LLM API calls.

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

Enforcement is an in-process, lock-protected counter keyed on the UTC date,
persisted to a small JSON file so a process restart / crash-loop cannot reset
the count and re-spend within the same day. Each module runs as its own
subprocess, so the JSON file (default logs/.llm_budget.json) is the shared,
cross-process source of truth. Concurrent writers use an atomic replace; an
occasional lost increment is acceptable for an approximate safety cap.

Cap resolution (first hit wins):
  1. explicit `cap=` argument
  2. env  BOT_LLM_DAILY_MAX_CALLS
  3. config['llm_daily_max_calls'] or config['advisor_llm_daily_max_calls']
  4. default 25
A cap <= 0 means UNLIMITED (operator explicitly disabled the gate).

Self-check: ``python -m core.llm_budget``
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DAILY_MAX = 25
_ENV_KEY = "BOT_LLM_DAILY_MAX_CALLS"
_CONFIG_KEYS = ("llm_daily_max_calls", "advisor_llm_daily_max_calls")
_BUDGET_FILE = Path(os.getenv("LLM_BUDGET_FILE", "logs/.llm_budget.json"))

_lock = threading.Lock()
_logger = logging.getLogger("llm_budget")

_state = {"day": "", "count": 0}
_loaded = False
_warned_day: Optional[str] = None  # so the "budget reached" WARNING fires once/day


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_from_disk() -> dict:
    try:
        with _BUDGET_FILE.open("r") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_to_disk(day: str, count: int) -> None:
    try:
        _BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _BUDGET_FILE.with_suffix(".tmp")
        with tmp.open("w") as fh:
            json.dump({"day": day, "count": count}, fh)
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


def try_consume(
    config: Optional[dict] = None,
    *,
    kind: str = "llm",
    cap: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Atomically reserve one paid-LLM call against today's global budget.

    Returns
    -------
    True  : permitted; the daily counter was incremented.
    False : the daily cap is already reached. The caller MUST fall back to
            rule-based / neutral output and make NO paid API call.

    cap <= 0 disables the gate (always True). The "budget reached" message is
    logged at WARNING exactly once per UTC day. On cross-process use the JSON
    file is re-read each call so a sibling process's spend is respected.
    """
    global _warned_day
    lg = log or _logger
    resolved_cap = _resolve_cap(config, cap)
    if resolved_cap <= 0:
        return True  # gate disabled by operator

    with _lock:
        today = _today()
        # Re-read disk each call so sibling subprocesses share one budget.
        disk = _load_from_disk()
        count = int(disk.get("count", 0)) if disk.get("day") == today else 0
        if disk.get("day") != today:
            _warned_day = None  # new UTC day -> allow the warning to fire again

        if count >= resolved_cap:
            _state["day"], _state["count"] = today, count
            if _warned_day != today:
                lg.warning(
                    "[llm_budget] Daily paid-LLM budget reached (%d/%d for %s, "
                    "last kind=%s); falling back to rule-based/neutral output — "
                    "NO further paid API calls until UTC rollover. Set env %s "
                    "(<=0 disables the cap).",
                    count, resolved_cap, today, kind, _ENV_KEY,
                )
                _warned_day = today
            return False

        count += 1
        _save_to_disk(today, count)
        _state["day"], _state["count"] = today, count
        return True


def snapshot() -> dict:
    """Return {'day', 'count'} for diagnostics / dashboard surfacing."""
    with _lock:
        return dict(_state)


if __name__ == "__main__":  # tiny self-check (no network, no DB)
    import tempfile
    logging.basicConfig(level=logging.INFO)
    _BUDGET_FILE = Path(tempfile.mkdtemp()) / ".llm_budget.json"
    results = [try_consume(cap=3, kind="selftest") for _ in range(5)]
    assert results == [True, True, True, False, False], results
    assert try_consume(cap=0) is True  # disabled
    print("core.llm_budget self-check OK:", results, "snapshot=", snapshot())
