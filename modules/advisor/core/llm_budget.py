"""
modules/advisor/core/llm_budget.py — hard global daily cap on paid LLM API calls.

Shared by BOTH the KAP classifier (`classifier._call_llm_classify`) and the
market-advice rationale generator (`rationale_helper.build_rationale`) so that
NO combination of bugs, tight loops, or cycle frequency can exceed a fixed
number of paid Anthropic/OpenAI calls per UTC day. When the budget is exhausted,
callers MUST fail soft to rule-based output and make NO API call.

This exists because a runaway KAP re-classification loop (a 60s worker that
re-LLM'd the same un-stored disclosures every cycle) burned a full day's API
credit in ~3 hours. The per-key rationale cache (issue #12) reduces advice spam
but is NOT a hard ceiling and does nothing for the KAP path — this module is the
backstop that makes a cost runaway impossible regardless of upstream bugs.

Enforcement is an in-process, lock-protected counter keyed on the UTC date,
persisted to a small JSON file under logs/advisor/ so a process restart (or a
crash-loop) cannot reset the count and re-spend within the same day.

Config (config_type='advisor_config'):
  advisor_llm_daily_max_calls : int, default 50.  <= 0 means UNLIMITED (gate off).

Self-check: ``python -m modules.advisor.core.llm_budget``
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DAILY_MAX = 50
_CONFIG_KEY = "advisor_llm_daily_max_calls"
_BUDGET_FILE = Path(
    os.getenv("ADVISOR_LLM_BUDGET_FILE", "logs/advisor/.llm_budget.json")
)

_lock = threading.Lock()
_logger = logging.getLogger("advisor.llm_budget")

# In-memory state is the enforced source of truth within a process; the JSON
# file makes it survive restarts within the same UTC day.
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


def _resolve_cap(config: Optional[dict]) -> int:
    try:
        return int((config or {}).get(_CONFIG_KEY, _DEFAULT_DAILY_MAX))
    except (TypeError, ValueError):
        return _DEFAULT_DAILY_MAX


def try_consume(
    config: Optional[dict] = None,
    *,
    kind: str = "llm",
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Atomically reserve one paid-LLM call against today's budget.

    Returns
    -------
    True  : the call is permitted; the daily counter has been incremented.
    False : the daily cap is already reached. The caller MUST fall back to
            rule-based output and make NO paid API call.

    Behaviour:
      * cap <= 0  -> always True (operator explicitly disabled the gate).
      * UTC day rollover resets the counter.
      * The "budget reached" message is logged at WARNING exactly once per day.
    """
    global _loaded, _warned_day
    lg = log or _logger
    cap = _resolve_cap(config)
    if cap <= 0:
        return True  # gate disabled by operator

    with _lock:
        today = _today()

        # First call in this process: seed from disk if it's still the same day.
        if not _loaded:
            disk = _load_from_disk()
            if disk.get("day") == today:
                _state["day"], _state["count"] = today, int(disk.get("count", 0))
            else:
                _state["day"], _state["count"] = today, 0
            _loaded = True

        if _state["day"] != today:  # UTC rollover during a long-lived process
            _state["day"], _state["count"] = today, 0
            _warned_day = None

        if _state["count"] >= cap:
            if _warned_day != today:
                lg.warning(
                    "[llm_budget] Daily paid-LLM budget reached (%d/%d for %s, "
                    "last kind=%s); falling back to rule-based output — NO further "
                    "paid API calls until UTC rollover. Raise/lower '%s' in "
                    "advisor_config to tune (<=0 disables the cap).",
                    _state["count"], cap, today, kind, _CONFIG_KEY,
                )
                _warned_day = today
            return False

        _state["count"] += 1
        _save_to_disk(today, _state["count"])
        return True


def snapshot() -> dict:
    """Return {'day', 'count', 'cap'?} for diagnostics / dashboard surfacing."""
    with _lock:
        return dict(_state)


if __name__ == "__main__":  # tiny self-check (no network, no DB)
    logging.basicConfig(level=logging.INFO)
    cfg = {"advisor_llm_daily_max_calls": 3}
    results = [try_consume(cfg, kind="selftest") for _ in range(5)]
    assert results[:3] == [True, True, True], results
    assert results[3:] == [False, False], results
    assert try_consume({"advisor_llm_daily_max_calls": 0}) is True  # disabled
    print("llm_budget self-check OK:", results, "snapshot=", snapshot())
