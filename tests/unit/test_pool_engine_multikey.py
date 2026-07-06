"""Offline tests for Wave-F5 pool_engine multi-key rotation.

No network, no DB: endpoints are bootstrapped from monkeypatched numbered
env vars and exercised entirely in memory.

Proves:
  1. numbered env keys (KEY, KEY_2, KEY_3) create N distinct endpoints;
  2. consecutive get_api_key calls round-robin across different keys;
  3. report_key_rate_limit cools a key and the next call returns a sibling;
  4. env-fallback rotation works even before initialize().
"""
import asyncio

import pytest

from config.pool_engine import PoolEngine


HELIUS_KEYS = ["helius-key-aaa", "helius-key-bbb", "helius-key-ccc"]
ETHERSCAN_KEYS = ["etherscan-key-one", "etherscan-key-two"]


@pytest.fixture
def multikey_env(monkeypatch):
    # Clear any ambient single-key config that could leak into the test.
    for var in list(HELIUS_KEYS) + ["HELIUS_API_KEY", "ETHERSCAN_API_KEY",
                                    "BIRDEYE_API_KEY", "GOPLUS_API_KEY",
                                    "1INCH_API_KEY", "JUPITER_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    for i in range(2, 10):
        monkeypatch.delenv(f"HELIUS_API_KEY_{i}", raising=False)
        monkeypatch.delenv(f"ETHERSCAN_API_KEY_{i}", raising=False)

    monkeypatch.setenv("HELIUS_API_KEY", HELIUS_KEYS[0])
    monkeypatch.setenv("HELIUS_API_KEY_2", HELIUS_KEYS[1])
    monkeypatch.setenv("HELIUS_API_KEY_3", HELIUS_KEYS[2])
    monkeypatch.setenv("ETHERSCAN_API_KEY", ETHERSCAN_KEYS[0])
    monkeypatch.setenv("ETHERSCAN_API_KEY_2", ETHERSCAN_KEYS[1])
    monkeypatch.setenv("BIRDEYE_API_KEY", "birdeye-key-solo")
    return monkeypatch


def _make_pool():
    """Fresh, non-singleton engine loaded ONLY from env (no DB, no tasks)."""
    pool = PoolEngine()
    asyncio.run(pool._load_from_env())
    pool.initialized = True
    return pool


def test_numbered_env_keys_create_n_endpoints(multikey_env):
    pool = _make_pool()

    helius = pool.providers.get("HELIUS_API")
    assert helius is not None
    assert len(helius.endpoints) == 3
    assert {e.api_key for e in helius.endpoints} == set(HELIUS_KEYS)
    # Each key gets a distinct URL (key embedded) at EQUAL priority.
    assert len({e.url for e in helius.endpoints}) == 3
    assert len({e.priority for e in helius.endpoints}) == 1

    etherscan = pool.providers.get("ETHERSCAN_API")
    assert etherscan is not None
    assert len(etherscan.endpoints) == 2
    assert {e.api_key for e in etherscan.endpoints} == set(ETHERSCAN_KEYS)
    # Base-URL providers keep slot 1 bare (back-compat with existing DB
    # rows) and tag slot 2+ to satisfy UNIQUE(provider_type, url).
    urls = sorted(e.url for e in etherscan.endpoints)
    assert urls[0] == "https://api.etherscan.io"
    assert "key_slot=2" in urls[1]

    birdeye = pool.providers.get("BIRDEYE_API")
    assert birdeye is not None and len(birdeye.endpoints) == 1


def test_get_api_key_round_robins(multikey_env):
    pool = _make_pool()

    async def run():
        seen = []
        for _ in range(6):
            result = await pool.get_api_key("HELIUS_API")
            assert result is not None
            key, endpoint_id = result
            assert endpoint_id > 0  # real endpoint, not env fallback
            seen.append(key)
        return seen

    seen = asyncio.run(run())
    # Consecutive calls rotate — never the same key twice in a row, and all
    # three siblings are handed out within one full rotation.
    assert all(a != b for a, b in zip(seen, seen[1:]))
    assert set(seen[:3]) == set(HELIUS_KEYS)


def test_rate_limited_key_rotates_out(multikey_env):
    pool = _make_pool()

    async def run():
        key, endpoint_id = await pool.get_api_key("HELIUS_API")
        # Report a 429 against exactly what was used (by endpoint id).
        await pool.report_key_rate_limit(endpoint_id, duration_seconds=300)
        # The cooled key must not be selected while limited.
        for _ in range(6):
            next_key, next_id = await pool.get_api_key("HELIUS_API")
            assert next_key != key
            assert next_id != endpoint_id
        # Reporting by the KEY STRING must resolve the same endpoint (this is
        # what consumers without an id in hand use, e.g. discovery URLs).
        key2, id2 = await pool.get_api_key("HELIUS_API")
        await pool.report_key_rate_limit(key2, duration_seconds=300)
        # With 2 of 3 keys cooled, only the last healthy sibling is served.
        for _ in range(4):
            last_key, _ = await pool.get_api_key("HELIUS_API")
            assert last_key not in (key, key2)
        return True

    assert asyncio.run(run())


def test_env_fallback_rotates_before_initialize(multikey_env):
    pool = PoolEngine()  # NOT initialized, no endpoints loaded

    async def run():
        seen = set()
        for _ in range(3):
            result = await pool.get_api_key("HELIUS_API")
            assert result is not None
            key, endpoint_id = result
            assert endpoint_id == -1  # env-fallback sentinel
            seen.add(key)
        return seen

    assert asyncio.run(run()) == set(HELIUS_KEYS)


def test_report_by_unknown_ref_is_noop(multikey_env):
    pool = _make_pool()

    async def run():
        # Must never raise: env sentinel, unknown id, unknown key.
        await pool.report_key_rate_limit(-1)
        await pool.report_key_rate_limit(999999)
        await pool.report_key_success("no-such-key")
        await pool.report_key_failure(None)
        return True

    assert asyncio.run(run())


if __name__ == "__main__":
    # Standalone self-test (repo pattern): the sandbox/pytest plugins are not
    # always available, so allow `python -m tests.unit.test_pool_engine_multikey`.
    import os

    class _EnvPatch:
        def setenv(self, k, v):
            os.environ[k] = v

        def delenv(self, k, raising=True):
            os.environ.pop(k, None)

    env = _EnvPatch()
    for var in ["HELIUS_API_KEY", "ETHERSCAN_API_KEY", "BIRDEYE_API_KEY"]:
        env.delenv(var)
    for i in range(2, 10):
        env.delenv(f"HELIUS_API_KEY_{i}")
        env.delenv(f"ETHERSCAN_API_KEY_{i}")
    env.setenv("HELIUS_API_KEY", HELIUS_KEYS[0])
    env.setenv("HELIUS_API_KEY_2", HELIUS_KEYS[1])
    env.setenv("HELIUS_API_KEY_3", HELIUS_KEYS[2])
    env.setenv("ETHERSCAN_API_KEY", ETHERSCAN_KEYS[0])
    env.setenv("ETHERSCAN_API_KEY_2", ETHERSCAN_KEYS[1])
    env.setenv("BIRDEYE_API_KEY", "birdeye-key-solo")

    test_numbered_env_keys_create_n_endpoints(env)
    test_get_api_key_round_robins(env)
    test_rate_limited_key_rotates_out(env)
    test_env_fallback_rotates_before_initialize(env)
    test_report_by_unknown_ref_is_noop(env)
    print("OK: all pool_engine multi-key rotation self-tests passed")
