#!/usr/bin/env python3
"""Emergency stop: flip kill-switch in-process AND notify the dashboard.

Usage:
    python scripts/emergency_stop.py [--reason TEXT] [--no-http]
                                      [--url http://localhost:8080]
                                      [--no-flag-file]

Effects (in order):
    1. Writes a flag file at logs/.killswitch (touched with reason+timestamp)
       so long-running subprocesses that poll the flag will halt.
    2. POSTs to {url}/api/bot/emergency-exit (the dash variant, which is
       canonical post-MB-31). If unreachable, exit code 2 but still considered
       partial success because the flag file is set.
    3. Calls core.dry_run.set_global_kill_switch(True) in this Python process
       (no-op outside the orchestrator process; cheap defense in depth).

Exit codes:
    0 - all requested channels succeeded
    2 - HTTP failed but flag-file succeeded (partial success)
    3 - flag-file write failed (most serious - no cross-process effect)
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root on sys.path so `core.dry_run` import works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiohttp  # noqa: E402

FLAG_PATH = _REPO_ROOT / "logs" / ".killswitch"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] emergency_stop: {msg}", file=sys.stderr, flush=True)


def write_flag_file(reason: str) -> bool:
    """Write logs/.killswitch with reason+timestamp+pid. Returns True on success."""
    try:
        FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
        }
        FLAG_PATH.write_text(json.dumps(payload))
        _log(f"flag file written: {FLAG_PATH}")
        return True
    except Exception as e:
        _log(f"FAILED to write flag file {FLAG_PATH}: {e}")
        return False


async def post_emergency_exit(url: str, reason: str) -> bool:
    """POST to /api/bot/emergency-exit. Returns True on HTTP 2xx."""
    endpoint = url.rstrip("/") + "/api/bot/emergency-exit"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json={"reason": reason}) as resp:
                body = await resp.text()
                if 200 <= resp.status < 300:
                    _log(f"HTTP POST {endpoint} -> {resp.status}: {body[:200]}")
                    return True
                _log(f"HTTP POST {endpoint} -> {resp.status}: {body[:200]}")
                return False
    except Exception as e:
        _log(f"HTTP POST {endpoint} failed: {e}")
        return False


def flip_inprocess_killswitch() -> None:
    """No-op outside orchestrator process. Defense in depth."""
    try:
        from core.dry_run import set_global_kill_switch
        set_global_kill_switch(True)
        _log("in-process kill switch SET (only affects this process)")
    except Exception as e:
        _log(f"in-process kill switch flip failed (non-fatal): {e}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Emergency stop CLI")
    parser.add_argument("--reason", default="manual emergency stop")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--no-http", action="store_true", default=False)
    parser.add_argument("--no-flag-file", action="store_true", default=False)
    args = parser.parse_args()

    _log(f"reason={args.reason!r} url={args.url} "
         f"http={'off' if args.no_http else 'on'} "
         f"flag_file={'off' if args.no_flag_file else 'on'}")

    flag_ok = True
    if not args.no_flag_file:
        flag_ok = write_flag_file(args.reason)

    http_ok = True
    if not args.no_http:
        http_ok = await post_emergency_exit(args.url, args.reason)

    flip_inprocess_killswitch()

    if not args.no_flag_file and not flag_ok:
        _log("EXIT 3 - flag-file write failed (no cross-process effect)")
        return 3
    if not args.no_http and not http_ok:
        _log("EXIT 2 - HTTP failed but flag-file succeeded")
        return 2
    _log("EXIT 0 - all requested channels succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
