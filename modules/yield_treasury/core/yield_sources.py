"""READ-ONLY yield/APR readers — free public RPC + free public APIs only.

- Aave v3 USDC supply APR: one eth_call to Pool.getReserveData(asset);
  currentLiquidityRate is word 2 of the returned struct, RAY-scaled (1e27).
- Lido stETH APR: free Lido API (percent).
- jitoSOL APY: free Sanctum API (fraction), shape-tolerant parse.

Venue allowlist is HARD-CODED here on purpose (blue-chip only, per the
module thesis) — adding a venue is a code review, not a config edit.
No key material, no signing, no state-changing calls.
Self-test: python -m modules.yield_treasury.core.yield_sources
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("yield_treasury")

_HTTP_TIMEOUT_S = 10
_RAY = 10 ** 27

# keccak256('getReserveData(address)')[:4] — Aave v3 Pool
_GET_RESERVE_DATA_SELECTOR = "0x35ea6a75"

# ── HARD-CODED venue allowlist (blue-chip only; never config-extendable) ──────
AAVE_V3_POOLS = {
    "ethereum": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
    "arbitrum": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
    "base": "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5",
}
AAVE_V3_USDC = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
}
LIDO_APR_URL_DEFAULT = "https://eth-api.lido.fi/v1/protocol/steth/apr/last"
JITOSOL_APY_URL_DEFAULT = "https://extra-api.sanctum.so/v1/apy/latest?lst=jitoSOL"


# ───────────────────────── pure helpers (self-tested) ─────────────────────────

def encode_get_reserve_data(asset: str) -> str:
    """ABI-encode getReserveData(asset) calldata."""
    addr = asset.lower()
    if addr.startswith("0x"):
        addr = addr[2:]
    return _GET_RESERVE_DATA_SELECTOR + addr.rjust(64, "0")

def decode_aave_supply_apr(hex_result: Optional[str]) -> Optional[float]:
    """eth_call result of getReserveData -> supply APR fraction (e.g. 0.043).

    Returned struct words: 0=configuration, 1=liquidityIndex,
    2=currentLiquidityRate (RAY), ... Sanity-clamped: None outside [0, 50%].
    """
    if not hex_result or not hex_result.startswith("0x"):
        return None
    data = hex_result[2:]
    if len(data) < 3 * 64:
        return None
    try:
        rate = int(data[2 * 64:3 * 64], 16)
    except ValueError:
        return None
    apr = rate / _RAY
    return apr if 0.0 <= apr <= 0.5 else None

def extract_rate(payload, keys=("apy", "apr")) -> Optional[float]:
    """Find the first numeric value under an apy/apr-named key in a JSON
    payload (shape-tolerant: handles Lido {'data':{'apr':3.1}} and Sanctum
    {'apys':{'jitoSOL':0.072}}), normalized to a fraction: values > 1.5 are
    treated as percent. Sanity-clamped to [0, 50%]."""
    def first_number(node):
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return float(node)
        if isinstance(node, dict):
            for v in node.values():
                r = first_number(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = first_number(v)
                if r is not None:
                    return r
        return None

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if any(t in str(k).lower() for t in keys):
                    r = first_number(v)
                    if r is not None:
                        return r
            for v in node.values():
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = walk(v)
                if r is not None:
                    return r
        return None

    raw = walk(payload)
    if raw is None:
        return None
    frac = raw / 100.0 if raw > 1.5 else raw
    return frac if 0.0 <= frac <= 0.5 else None


# ───────────────────────── network readers (fail-soft -> None) ────────────────

class RateLimited(Exception):
    """HTTP 429 — caller must report_rate_limit to pool_engine for RPC reads."""

async def aave_v3_usdc_supply_apr(session, rpc_url: str, chain: str) -> Optional[float]:
    """Supply APR for canonical USDC on one allowlisted Aave v3 market.
    Raises RateLimited on 429 (so the caller reports it); other failures raise."""
    pool_addr = AAVE_V3_POOLS.get(chain)
    asset = AAVE_V3_USDC.get(chain)
    if not pool_addr or not asset:
        return None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": pool_addr,
                           "data": encode_get_reserve_data(asset)}, "latest"]}
    async with session.post(rpc_url, json=payload, timeout=_HTTP_TIMEOUT_S) as resp:
        if resp.status == 429:
            raise RateLimited("HTTP 429 from eth_call getReserveData")
        resp.raise_for_status()
        body = await resp.json(content_type=None)
    if "error" in body:
        raise RuntimeError(f"getReserveData RPC error: {body['error']}")
    return decode_aave_supply_apr(body.get("result"))

async def fetch_json_rate(session, url: str) -> Optional[float]:
    """GET a free public APY/APR endpoint -> fraction. Fail-soft -> None."""
    try:
        async with session.get(url, timeout=_HTTP_TIMEOUT_S) as resp:
            if resp.status != 200:
                logger.warning("rate endpoint %s -> HTTP %d", url, resp.status)
                return None
            body = await resp.json(content_type=None)
        return extract_rate(body)
    except Exception as exc:
        logger.warning("rate endpoint %s fail-soft: %s", url, exc)
        return None


# ───────────────────────────── self-test ──────────────────────────────────────

if __name__ == "__main__":
    cd = encode_get_reserve_data(AAVE_V3_USDC["ethereum"])
    assert cd == ("0x35ea6a75"
                  "000000000000000000000000"
                  "a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"), cd
    assert len(cd) == 2 + 8 + 64

    # 4.3% APR in RAY at word 2; words 0/1 nonzero noise; trailing words present
    rate_ray = int(0.043 * _RAY)
    words = ["ab" * 32, "01" * 32, hex(rate_ray)[2:].rjust(64, "0")] + ["00" * 32] * 12
    apr = decode_aave_supply_apr("0x" + "".join(words))
    assert apr is not None and abs(apr - 0.043) < 1e-9, apr
    assert decode_aave_supply_apr(None) is None
    assert decode_aave_supply_apr("0x") is None
    assert decode_aave_supply_apr("0xzz") is None
    big = hex(int(0.9 * _RAY))[2:].rjust(64, "0")          # 90% -> implausible
    assert decode_aave_supply_apr("0x" + "00" * 64 + big) is None

    # Lido shape (percent) and Sanctum shape (fraction)
    assert abs(extract_rate({"data": {"apr": 3.1}}) - 0.031) < 1e-12
    assert abs(extract_rate({"apys": {"jitoSOL": 0.0721}}) - 0.0721) < 1e-12
    assert abs(extract_rate({"data": [{"apy": 7.4}]}) - 0.074) < 1e-12
    assert extract_rate({"data": {"value": "n/a"}}) is None
    assert extract_rate({"apr": 99.0}) is None              # 99% -> implausible
    assert extract_rate({"apr": True}) is None              # bool is not a rate
    print("yield_sources self-test OK")
