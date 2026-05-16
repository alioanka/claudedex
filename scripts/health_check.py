# scripts/health_check.py
"""
Lightweight health check used by Docker / external monitoring.

Hits the dashboard's /health endpoint and reports its findings. The
dashboard already probes the DB internally, so we don't duplicate that
work here (and we don't carry stale 'trading:trading123' credentials,
which never worked against the secrets-mounted postgres).

Exits 0 if /health returns 200 AND status == 'healthy'.
Exits 1 otherwise.
"""
import asyncio
import sys
import aiohttp


async def check_health(url: str = "http://localhost:8080/health") -> int:
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"❌ /health returned HTTP {resp.status}")
                    return 1
                data = await resp.json()
        status = data.get('status', 'unknown')
        sha = data.get('git_sha') or 'unknown'
        db_state = data.get('db') or 'unknown'
        print(f"Health Status: {status}")
        print(f"  service: {data.get('service', '?')}")
        print(f"  build:   {sha}")
        print(f"  db:      {db_state}")
        return 0 if status == 'healthy' else 1
    except aiohttp.ClientError as e:
        print(f"❌ /health unreachable: {type(e).__name__}: {e}")
        return 1
    except Exception as e:
        print(f"❌ unexpected error: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    rc = asyncio.run(check_health())
    sys.exit(rc)
