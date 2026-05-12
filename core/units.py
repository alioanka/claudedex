"""Cached on-chain decimals + unit conversions for EVM and Solana.

Single source of truth for human <-> raw on-chain unit conversions across
every trading path. Decimals are immutable on-chain, so the per-token
cache has no TTL.

RPC endpoints are resolved through ``config.pool_engine.PoolEngine`` -
this module must never read ``*_RPC_URL`` env vars directly.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Dict, Union

NATIVE_ETH_DECIMALS = 18
NATIVE_SOL_DECIMALS = 9

Number = Union[int, float, Decimal, str]


class UnitsError(Exception):
    """Raised when a decimals lookup or conversion fails.

    We intentionally do NOT fall back to a default decimals value: a
    silent fallback (the bug fixed by MB-01/06/23) builds wrong-sized
    orders and either reverts or trades blind.
    """


# ---------------------------------------------------------------------------
# Caches. Decimals are immutable on-chain, so we accept the rare race where
# two callers hit RPC concurrently for the same key - both will write the
# same value. No lock needed.
# ---------------------------------------------------------------------------
_evm_decimals_cache: Dict[str, int] = {}  # key: f"{chain}:{checksum_addr}"
_spl_decimals_cache: Dict[str, int] = {}  # key: mint string


# ---------------------------------------------------------------------------
# Chain -> pool_engine provider_type
# ---------------------------------------------------------------------------
_CHAIN_TO_PROVIDER: Dict[str, str] = {
    "ethereum": "ETHEREUM_RPC",
    "bsc": "BSC_RPC",
    "polygon": "POLYGON_RPC",
    "arbitrum": "ARBITRUM_RPC",
    "base": "BASE_RPC",
    "monad": "MONAD_RPC",
    "pulsechain": "PULSECHAIN_RPC",
    "fantom": "FANTOM_RPC",
    "cronos": "CRONOS_RPC",
    "avalanche": "AVALANCHE_RPC",
}

# Minimal ERC-20 decimals() ABI
_ERC20_DECIMALS_ABI = json.loads(
    '[{"constant":true,"inputs":[],"name":"decimals",'
    '"outputs":[{"name":"","type":"uint8"}],'
    '"payable":false,"stateMutability":"view","type":"function"}]'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_decimal(amount: Number) -> Decimal:
    """Coerce amount to Decimal without losing precision from floats."""
    if isinstance(amount, Decimal):
        return amount
    # Route through str() so float drift doesn't leak into raw integer math.
    return Decimal(str(amount))


async def _get_evm_rpc_url(chain: str) -> str:
    """Resolve an EVM RPC URL for ``chain`` via pool_engine."""
    provider = _CHAIN_TO_PROVIDER.get(chain.lower())
    if not provider:
        raise UnitsError(f"Unknown EVM chain: {chain!r}")

    from config.pool_engine import PoolEngine

    pool = await PoolEngine.get_instance()
    url = await pool.get_endpoint(provider)
    if not url:
        raise UnitsError(f"No RPC endpoint available for {provider}")
    return url


async def _get_solana_rpc_url() -> str:
    """Resolve a Solana RPC URL via pool_engine."""
    from config.pool_engine import PoolEngine

    pool = await PoolEngine.get_instance()
    url = await pool.get_endpoint("SOLANA_RPC")
    if not url:
        raise UnitsError("No RPC endpoint available for SOLANA_RPC")
    return url


# ---------------------------------------------------------------------------
# EVM
# ---------------------------------------------------------------------------
async def get_evm_decimals(chain: str, token_address: str) -> int:
    """Fetch ERC-20 ``decimals()`` for ``token_address`` on ``chain``.

    Result is cached per ``(chain, checksum_address)`` for the process
    lifetime - decimals never change.
    """
    from web3 import Web3

    try:
        checksum = Web3.to_checksum_address(token_address)
    except Exception as exc:
        raise UnitsError(f"Invalid EVM address {token_address!r}: {exc}") from exc

    cache_key = f"{chain.lower()}:{checksum}"
    cached = _evm_decimals_cache.get(cache_key)
    if cached is not None:
        return cached

    url = await _get_evm_rpc_url(chain)
    try:
        w3 = Web3(Web3.HTTPProvider(url))
        contract = w3.eth.contract(address=checksum, abi=_ERC20_DECIMALS_ABI)
        decimals = int(contract.functions.decimals().call())
    except Exception as exc:
        raise UnitsError(
            f"decimals() call failed for {checksum} on {chain}: {exc}"
        ) from exc

    if decimals < 0 or decimals > 36:
        raise UnitsError(
            f"Implausible decimals={decimals} for {checksum} on {chain}"
        )

    _evm_decimals_cache[cache_key] = decimals
    return decimals


async def to_raw_evm(chain: str, token_address: str, amount: Number) -> int:
    """Convert a human-readable amount to raw on-chain units."""
    decimals = await get_evm_decimals(chain, token_address)
    return int(_to_decimal(amount) * (Decimal(10) ** decimals))


async def from_raw_evm(chain: str, token_address: str, raw: int) -> Decimal:
    """Convert raw on-chain units to a human-readable Decimal."""
    decimals = await get_evm_decimals(chain, token_address)
    return Decimal(int(raw)) / (Decimal(10) ** decimals)


# ---------------------------------------------------------------------------
# Solana
# ---------------------------------------------------------------------------
async def get_spl_decimals(token_mint: str) -> int:
    """Fetch SPL mint decimals via solana-py + pool_engine.

    Cached per mint for the process lifetime.
    """
    if not token_mint:
        raise UnitsError("Empty token_mint")

    cached = _spl_decimals_cache.get(token_mint)
    if cached is not None:
        return cached

    url = await _get_solana_rpc_url()
    try:
        from solana.rpc.async_api import AsyncClient
        from solders.pubkey import Pubkey

        client = AsyncClient(url)
        try:
            resp = await client.get_token_supply(Pubkey.from_string(token_mint))
        finally:
            await client.close()

        value = getattr(resp, "value", None)
        decimals = getattr(value, "decimals", None) if value is not None else None
        if decimals is None:
            raise UnitsError(f"Mint {token_mint} returned no decimals")
        decimals = int(decimals)
    except UnitsError:
        raise
    except Exception as exc:
        raise UnitsError(
            f"get_token_supply failed for mint {token_mint}: {exc}"
        ) from exc

    if decimals < 0 or decimals > 36:
        raise UnitsError(f"Implausible decimals={decimals} for mint {token_mint}")

    _spl_decimals_cache[token_mint] = decimals
    return decimals


async def to_raw_spl(token_mint: str, amount: Number) -> int:
    """Convert human-readable SPL amount to raw mint units."""
    decimals = await get_spl_decimals(token_mint)
    return int(_to_decimal(amount) * (Decimal(10) ** decimals))


async def from_raw_spl(token_mint: str, raw: int) -> Decimal:
    """Convert raw SPL mint units to a human-readable Decimal."""
    decimals = await get_spl_decimals(token_mint)
    return Decimal(int(raw)) / (Decimal(10) ** decimals)


# ---------------------------------------------------------------------------
# Native shortcuts (constants only - no RPC)
# ---------------------------------------------------------------------------
def to_wei(amount: Number) -> int:
    """Convert ETH-denominated amount to wei (18 decimals)."""
    return int(_to_decimal(amount) * (Decimal(10) ** NATIVE_ETH_DECIMALS))


def from_wei(raw: int) -> Decimal:
    """Convert wei to ETH-denominated Decimal."""
    return Decimal(int(raw)) / (Decimal(10) ** NATIVE_ETH_DECIMALS)


def to_lamports(sol_amount: Number) -> int:
    """Convert SOL-denominated amount to lamports (9 decimals)."""
    return int(_to_decimal(sol_amount) * (Decimal(10) ** NATIVE_SOL_DECIMALS))


def from_lamports(raw: int) -> Decimal:
    """Convert lamports to SOL-denominated Decimal."""
    return Decimal(int(raw)) / (Decimal(10) ** NATIVE_SOL_DECIMALS)
