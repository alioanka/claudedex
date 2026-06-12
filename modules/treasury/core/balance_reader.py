"""READ-ONLY balance readers (EVM JSON-RPC + Solana JSON-RPC).

Only balance queries — eth_getBalance / eth_call(balanceOf) /
getBalance / getTokenAccountsByOwner. No key material, no signing,
no state-changing calls. Self-test: python -m modules.treasury.core.balance_reader
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("treasury")

_RPC_TIMEOUT_S = 10

# keccak256('balanceOf(address)')[:4]
_BALANCE_OF_SELECTOR = "0x70a08231"


# ───────────────────────── pure helpers (self-tested) ─────────────────────────

def encode_balance_of(wallet: str) -> str:
    """ABI-encode balanceOf(wallet) calldata."""
    addr = wallet.lower()
    if addr.startswith("0x"):
        addr = addr[2:]
    return _BALANCE_OF_SELECTOR + addr.rjust(64, "0")


def hex_to_units(hex_value: Optional[str], decimals: int) -> Optional[float]:
    """JSON-RPC hex quantity -> human units. None/empty -> None."""
    if not hex_value or hex_value in ("0x", "0X"):
        return None
    try:
        return int(hex_value, 16) / (10 ** decimals)
    except (ValueError, TypeError):
        return None


# ───────────────────────── RPC readers (fail-soft -> None) ────────────────────

async def _json_rpc(session, url: str, method: str, params: list):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    async with session.post(url, json=payload, timeout=_RPC_TIMEOUT_S) as resp:
        if resp.status == 429:
            raise RateLimited(f"HTTP 429 from {method}")
        resp.raise_for_status()
        body = await resp.json(content_type=None)
    if "error" in body:
        raise RuntimeError(f"{method} RPC error: {body['error']}")
    return body.get("result")


class RateLimited(Exception):
    """Raised on HTTP 429 so the caller can report_rate_limit to pool_engine."""


async def evm_native_balance(session, rpc_url: str, wallet: str) -> Optional[float]:
    """Native balance in whole units (e.g. ETH). None on bad result."""
    result = await _json_rpc(session, rpc_url, "eth_getBalance", [wallet, "latest"])
    return hex_to_units(result, 18)


async def evm_erc20_balance(session, rpc_url: str, token_address: str,
                            wallet: str, decimals: int) -> Optional[float]:
    """ERC20 balanceOf(wallet) in human units. None on bad result."""
    result = await _json_rpc(
        session, rpc_url, "eth_call",
        [{"to": token_address, "data": encode_balance_of(wallet)}, "latest"],
    )
    return hex_to_units(result, int(decimals))


async def solana_native_balance(session, rpc_url: str, wallet: str) -> Optional[float]:
    """SOL balance in whole SOL. None on bad result."""
    result = await _json_rpc(session, rpc_url, "getBalance", [wallet])
    if not isinstance(result, dict) or "value" not in result:
        return None
    return int(result["value"]) / 1e9


async def solana_token_balance(session, rpc_url: str, wallet: str,
                               mint: str) -> Optional[float]:
    """Sum of ui token balances for one mint owned by wallet. None on bad result."""
    result = await _json_rpc(
        session, rpc_url, "getTokenAccountsByOwner",
        [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
    )
    if not isinstance(result, dict):
        return None
    total = 0.0
    for acc in result.get("value") or []:
        try:
            info = acc["account"]["data"]["parsed"]["info"]
            total += float(info["tokenAmount"]["uiAmount"] or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
    return total


# ───────────────────────────── self-test ──────────────────────────────────────

if __name__ == "__main__":
    cd = encode_balance_of("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")
    assert cd == ("0x70a08231"
                  "000000000000000000000000"
                  "a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"), cd
    assert len(cd) == 2 + 8 + 64
    assert hex_to_units("0xde0b6b3a7640000", 18) == 1.0          # 1 ETH in wei
    assert hex_to_units("0xf4240", 6) == 1.0                     # 1 USDC (6 dec)
    assert hex_to_units("0x0", 18) == 0.0
    assert hex_to_units("0x", 18) is None
    assert hex_to_units(None, 18) is None
    assert hex_to_units("zz", 18) is None
    print("balance_reader self-test OK")
