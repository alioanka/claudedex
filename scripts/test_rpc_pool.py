#!/usr/bin/env python3
"""
RPC/API Pool connectivity tester (FEATURE 6).

Loads every endpoint from the DB-backed RPC/API pool and tests each one:
  * connectivity + HTTP status
  * round-trip latency
  * one sample chain call:
      - Solana endpoints: getHealth (then getSlot as a deeper liveness probe)
      - EVM endpoints:    eth_blockNumber
      - API endpoints:    a plain GET to the base URL

Prints a per-endpoint PASS / FAIL / RATE-LIMITED report with latency, grouped
by provider, plus a summary tally and a non-zero exit code if any endpoint
failed (so it is usable in CI / cron health checks).

Operator-runnable:
    docker exec trading-bot python scripts/test_rpc_pool.py
    docker exec trading-bot python scripts/test_rpc_pool.py --provider HELIUS_API
    docker exec trading-bot python scripts/test_rpc_pool.py --timeout 8 --concurrency 8

It is READ-ONLY against live state in spirit: it does call the pool engine's
report_success/report_rate_limit/report_failure so the health tracker benefits
from the probe, exactly like the dashboard "Test All" button. Pass --no-report
to skip that and leave pool health untouched.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Dict, List, Optional

# Make the repo root importable when invoked as `python scripts/test_rpc_pool.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import aiohttp
except ImportError:  # pragma: no cover - aiohttp is a hard dep of the app
    print("ERROR: aiohttp is required (pip install aiohttp)", file=sys.stderr)
    sys.exit(2)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("test_rpc_pool")


# --------------------------------------------------------------------------- #
# DB / pool bootstrap (mirrors scripts/refresh_copy_leaders.py resolver)
# --------------------------------------------------------------------------- #
async def _open_db_pool():
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except Exception:
        db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("No DATABASE_URL configured; falling back to .env endpoints")
        return None
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg not installed; falling back to .env endpoints")
        return None
    try:
        return await asyncpg.create_pool(db_url, min_size=1, max_size=4)
    except Exception as e:
        logger.warning(f"asyncpg.create_pool failed ({e}); falling back to .env endpoints")
        return None


# --------------------------------------------------------------------------- #
# Per-endpoint probe
# --------------------------------------------------------------------------- #
def _is_solana(endpoint: Dict) -> bool:
    chain = (endpoint.get("chain") or "").lower()
    ptype = (endpoint.get("provider_type") or "").lower()
    url = (endpoint.get("url") or "").lower()
    return "solana" in chain or "solana" in ptype or "helius" in ptype or "helius" in url


def _is_rpc(endpoint: Dict) -> bool:
    ptype = (endpoint.get("provider_type") or "").upper()
    return "RPC" in ptype or "WS" in ptype or "HELIUS" in ptype


async def _probe(
    session: aiohttp.ClientSession, endpoint: Dict, timeout: float
) -> Dict:
    url = endpoint.get("effective_url") or endpoint.get("url")
    provider_type = endpoint.get("provider_type", "")
    result = {
        "id": endpoint.get("id"),
        "name": endpoint.get("name", ""),
        "provider_type": provider_type,
        "chain": endpoint.get("chain"),
        "success": False,
        "latency_ms": 0,
        "error": None,
        "rate_limited": False,
        "sample": None,
    }
    if not url:
        result["error"] = "no url"
        return result

    t0 = time.time()
    try:
        if _is_rpc(endpoint):
            if _is_solana(endpoint):
                payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
            else:
                payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                result["latency_ms"] = int((time.time() - t0) * 1000)
                if resp.status == 429:
                    result["rate_limited"] = True
                    result["error"] = "HTTP 429 rate limited"
                    return result
                if resp.status != 200:
                    result["error"] = f"HTTP {resp.status}"
                    return result
                data = await resp.json(content_type=None)
                if "result" in data:
                    result["success"] = True
                    result["sample"] = str(data["result"])[:40]
                    # Deeper liveness for Solana: pull a slot number too.
                    if _is_solana(endpoint):
                        try:
                            slot_payload = {"jsonrpc": "2.0", "id": 2, "method": "getSlot"}
                            async with session.post(
                                url, json=slot_payload,
                                timeout=aiohttp.ClientTimeout(total=timeout),
                            ) as sresp:
                                if sresp.status == 200:
                                    sdata = await sresp.json(content_type=None)
                                    if "result" in sdata:
                                        result["sample"] = f"slot={sdata['result']}"
                        except Exception:
                            pass
                elif "error" in data:
                    err = data["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    code = err.get("code") if isinstance(err, dict) else 0
                    if code in (-32005, -32097, -32098, -32099):
                        result["rate_limited"] = True
                    result["error"] = msg or "rpc error"
                else:
                    result["success"] = True
        else:
            # API endpoint — plain GET liveness.
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                result["latency_ms"] = int((time.time() - t0) * 1000)
                if resp.status == 429:
                    result["rate_limited"] = True
                    result["error"] = "HTTP 429 rate limited"
                elif resp.status in (200, 201, 400, 401, 403, 404):
                    # Reaching the API host (even a 4xx auth/route response)
                    # proves connectivity; the API-key validity is out of scope.
                    result["success"] = resp.status in (200, 201)
                    if not result["success"]:
                        result["error"] = f"reachable, HTTP {resp.status}"
                else:
                    result["error"] = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        result["latency_ms"] = int((time.time() - t0) * 1000)
        result["error"] = "timeout"
    except aiohttp.ClientError as e:
        result["error"] = f"connection error: {e}"
    except Exception as e:
        result["error"] = str(e)
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def run(args) -> int:
    from config.pool_engine import PoolEngine

    db_pool = await _open_db_pool()
    pool = await PoolEngine.get_instance()
    await pool.initialize(db_pool)

    endpoints: List[Dict] = await pool.get_all_endpoints_data()

    # Attach an effective_url (URL + api-key) so probes hit the real target.
    by_id = {}
    for prov in pool.providers.values():
        for ep in prov.endpoints:
            by_id[ep.id] = ep
    for ep_dict in endpoints:
        live = by_id.get(ep_dict.get("id"))
        if live is not None:
            ep_dict["effective_url"] = live.get_effective_url()

    if args.provider:
        endpoints = [e for e in endpoints if e.get("provider_type") == args.provider]

    if not endpoints:
        print("No endpoints found (DB empty and no .env fallback configured).")
        return 1

    print(f"Testing {len(endpoints)} endpoint(s) "
          f"(timeout={args.timeout}s, concurrency={args.concurrency})...\n")

    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(session, ep):
        async with sem:
            res = await _probe(session, ep, args.timeout)
            if not args.no_report:
                try:
                    url = ep.get("url")
                    pt = ep.get("provider_type")
                    if res["success"]:
                        await pool.report_success(pt, url, res["latency_ms"])
                    elif res["rate_limited"]:
                        await pool.report_rate_limit(pt, url, duration_seconds=300)
                    else:
                        await pool.report_failure(pt, url, "pool_test", res["error"])
                except Exception:
                    pass
            return res

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_bounded(session, e) for e in endpoints])

    # Group + print
    by_provider: Dict[str, List[Dict]] = {}
    for r in results:
        by_provider.setdefault(r["provider_type"], []).append(r)

    passed = failed = limited = 0
    for provider in sorted(by_provider):
        print(f"== {provider} ==")
        for r in sorted(by_provider[provider], key=lambda x: x.get("name") or ""):
            if r["success"]:
                state = "PASS"
                passed += 1
            elif r["rate_limited"]:
                state = "RATE-LIMITED"
                limited += 1
            else:
                state = "FAIL"
                failed += 1
            lat = f"{r['latency_ms']}ms" if r["latency_ms"] else "-"
            detail = ""
            if r["success"] and r.get("sample"):
                detail = f"  [{r['sample']}]"
            elif r.get("error"):
                detail = f"  ({r['error']})"
            print(f"  [{state:>12}] {r['name'][:40]:<40} {lat:>8}{detail}")
        print()

    total = len(results)
    print(f"Summary: {passed}/{total} PASS, {failed} FAIL, {limited} RATE-LIMITED")

    await pool.shutdown()
    if db_pool is not None:
        await db_pool.close()

    # Non-zero exit if anything failed (rate-limited is not a hard failure).
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Test all RPC/API pool endpoints.")
    parser.add_argument("--provider", help="Only test this provider_type (e.g. HELIUS_API)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent probes")
    parser.add_argument("--no-report", action="store_true",
                        help="Do not report results back to the pool health tracker")
    args = parser.parse_args()
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
