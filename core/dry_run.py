"""Centralized DRY_RUN / kill-switch gating.

Every module's send/order/sign/transfer path should call
should_skip_live(...) before executing live writes. Returns True iff the
call must be simulated. The orchestrator can also flip a process-wide
kill switch via set_global_kill_switch(True) without touching modules.
"""

import os
from typing import Optional

_GLOBAL_KILL_SWITCH = False

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
