"""Centralized DRY_RUN / kill-switch gating.

Every module's send/order/sign/transfer path should call
should_skip_live(...) before executing live writes. Returns True iff the
call must be simulated. The orchestrator can also flip a process-wide
kill switch via set_global_kill_switch(True) without touching modules.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

_GLOBAL_KILL_SWITCH = False

_logger = logging.getLogger("core.dry_run")

_POLLER_TASK: Optional["asyncio.Task"] = None
_POLLER_PATH: Optional[Path] = None  # Track current path for idempotency check

_TRUTHY = {"true", "1", "yes", "on"}
_FALSY = {"false", "0", "no", "off"}


def set_global_kill_switch(value: bool) -> None:
    global _GLOBAL_KILL_SWITCH
    _GLOBAL_KILL_SWITCH = bool(value)


def is_global_kill_switch() -> bool:
    return _GLOBAL_KILL_SWITCH


def resolve_dry_run_env(env_var: str = "DRY_RUN", default: bool = True) -> bool:
    """Parse env. Default True (safe). Accepts true/1/yes/on / false/0/no/off.
    Default=True ensures a missing/typo env var does NOT flip the bot live."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return default


def should_skip_live(
    module_dry_run: bool,
    *,
    module: str = "",
    account: Optional[str] = None,
) -> bool:
    """Returns True if this call must NOT hit live exchanges/chains.
    True iff: global kill-switch ON, OR module_dry_run=True."""
    if _GLOBAL_KILL_SWITCH:
        return True
    return bool(module_dry_run)


async def _killswitch_poll_loop(path: Path, interval: float) -> None:
    """Watch path; when it appears, flip the global kill switch and keep polling.

    Continues running after the switch is set so future logs note that the file
    is still present (or vanished). The switch never auto-clears; once tripped
    only a process restart resets it.
    """
    tripped = False
    while True:
        try:
            exists = path.exists()
            if exists and not tripped:
                set_global_kill_switch(True)
                tripped = True
                try:
                    payload = path.read_text(errors="replace")[:512]
                except Exception:
                    payload = "<unreadable>"
                _logger.critical(
                    "KILL SWITCH TRIPPED via flag file %s: %s", path, payload
                )
        except Exception as e:  # poller MUST NOT die — log and continue
            _logger.error("killswitch poll error: %s", e)
        await asyncio.sleep(interval)


def start_killswitch_poller(
    path: "str | Path" = "logs/.killswitch",
    interval: float = 1.0,
) -> "asyncio.Task":
    """Start (or return the already-running) flag-file poller.

    Idempotent: subsequent calls with the same path return the existing task.
    Calling with a different path while a poller is already running is a
    programmer error — the existing task is returned and a warning logged.

    Must be called from within a running event loop. Each process needs its
    own poller; this is a per-process singleton.
    """
    global _POLLER_TASK, _POLLER_PATH
    target = Path(path)
    if _POLLER_TASK is not None and not _POLLER_TASK.done():
        if _POLLER_PATH != target:
            _logger.warning(
                "start_killswitch_poller(%s) called but poller already running "
                "on %s; ignoring new path", target, _POLLER_PATH
            )
        return _POLLER_TASK
    _POLLER_PATH = target
    _POLLER_TASK = asyncio.create_task(
        _killswitch_poll_loop(target, interval),
        name="killswitch-poller",
    )
    _logger.info("killswitch poller started: path=%s interval=%ss", target, interval)
    return _POLLER_TASK


def stop_killswitch_poller() -> None:
    """Cancel the poller. Mainly for tests / clean shutdown."""
    global _POLLER_TASK, _POLLER_PATH
    if _POLLER_TASK is not None and not _POLLER_TASK.done():
        _POLLER_TASK.cancel()
    _POLLER_TASK = None
    _POLLER_PATH = None
