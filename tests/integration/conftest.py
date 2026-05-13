"""Integration-suite conftest.

Skips the whole sub-tree gracefully when infra dependencies (asyncpg,
aiohttp, numpy, etc.) are absent. Without this, missing deps cause
hard collection errors instead of clean SKIPs - making CI runs in
slim environments unrunnable.

Heavyweight production-stack imports remain lazy (see /tests/conftest.py)
so a test that doesn't need them can still run.

NOTE: We do NOT call pytest.importorskip() at module top level here.
Raising Skipped during initial conftest load causes pytest to abort
with exit=1 rather than emit per-test SKIPs. Instead we set
collect_ignore so the affected files are dropped from collection,
which yields a clean run.
"""

import importlib.util

# Required infra deps for the integration suite. Add to this list when
# a new integration test introduces a hard infra dep that should also
# be skip-gated.
_REQUIRED = ("asyncpg",)

collect_ignore_glob: list[str] = []

_missing = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
if _missing:
    # Drop the entire integration sub-tree from collection. Pytest will
    # report this as "no tests ran" rather than a collection error.
    collect_ignore_glob.append("test_*.py")
