#!/usr/bin/env python3
"""Best-effort: close every open position across all modules.

Does NOT flip the kill switch. If you want both, also run
scripts/emergency_stop.py first.

Usage:
    python scripts/close_all_positions.py [--url http://localhost:8080]
                                          [--dry-run]
                                          [--module MODULE_KEY]

Strategy:
    POST {url}/api/bot/emergency-exit -- the canonical handler (unified
    in MB-31) already iterates modules and calls close_position(). The
    server-side endpoint flips the kill switch + flattens + stops modules
    in one shot today; a pure-flatten endpoint (no kill switch, no module
    stop) is a future addition.

    --module is accepted for forward compatibility but currently logs a
    warning that single-module flatten is not yet supported server-side.

Exit codes:
    0 - success
    1 - partial (response body indicated module-level errors)
    2 - dashboard unreachable / non-2xx
"""
# TODO: when a pure-flatten endpoint exists (e.g. /api/bot/flatten that
# does NOT flip the kill switch and does NOT call module.stop()), point
# this script at it instead of /api/bot/emergency-exit. Likewise wire up
# --module once per-module flatten is supported server-side.
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root on sys.path for parity with emergency_stop.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiohttp  # noqa: E402


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] close_all_positions: {msg}", file=sys.stderr, flush=True)


async def post_flatten(url: str) -> tuple[int, dict | None]:
    """POST to /api/bot/emergency-exit. Returns (http_status, parsed_body_or_None)."""
    endpoint = url.rstrip("/") + "/api/bot/emergency-exit"
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json={"reason": "close_all_positions CLI"}) as resp:
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = None
                return resp.status, body
    except Exception as e:
        _log(f"HTTP POST {endpoint} failed: {e}")
        return 0, None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Best-effort position flattener CLI")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--module", default=None,
                        help="(reserved) single-module key; not yet supported server-side")
    args = parser.parse_args()

    if args.module:
        _log(f"WARNING: --module={args.module!r} ignored - single-module flatten "
             f"not yet supported by /api/bot/emergency-exit")

    endpoint = args.url.rstrip("/") + "/api/bot/emergency-exit"
    if args.dry_run:
        _log(f"DRY-RUN: would POST {endpoint} with reason='close_all_positions CLI'")
        _log("DRY-RUN: no HTTP call made; exit 0")
        return 0

    _log(f"POSTing to {endpoint}")
    status, body = await post_flatten(args.url)

    if status == 0:
        _log("EXIT 2 - dashboard unreachable")
        return 2
    if not (200 <= status < 300):
        _log(f"EXIT 2 - dashboard returned HTTP {status}: {body!r}")
        return 2

    if not isinstance(body, dict):
        _log(f"EXIT 1 - unexpected response body (not a dict): {body!r}")
        return 1

    closed = body.get("closed_positions", 0)
    message = body.get("message", "")
    success = bool(body.get("success", False))

    _log(f"server: success={success} closed_positions={closed} message={message!r}")

    if not success:
        _log(f"EXIT 1 - server reported success=False: {body.get('error', '')}")
        return 1

    _log(f"EXIT 0 - flattened {closed} position(s) across all modules")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
