# tests/unit/test_units.py
"""Smoke tests for core.units."""
from decimal import Decimal

import pytest

from core import units


@pytest.mark.unit
def test_native_shortcuts():
    assert units.to_wei(1) == 10 ** 18
    assert units.to_wei(Decimal("0.5")) == 5 * 10 ** 17
    assert units.from_wei(10 ** 18) == Decimal(1)

    assert units.to_lamports(1) == 10 ** 9
    assert units.to_lamports(Decimal("0.001")) == 10 ** 6
    assert units.from_lamports(10 ** 9) == Decimal(1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_to_raw_evm_roundtrip_mocked(monkeypatch):
    """With decimals=6 (USDC), 1.5 -> 1_500_000 raw."""
    USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

    async def fake_get_evm_decimals(chain, token_address):
        return 6

    monkeypatch.setattr(units, "get_evm_decimals", fake_get_evm_decimals)

    raw = await units.to_raw_evm("ethereum", USDC, Decimal("1.5"))
    assert raw == 1_500_000

    back = await units.from_raw_evm("ethereum", USDC, raw)
    assert back == Decimal("1.5")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_to_raw_spl_roundtrip_mocked(monkeypatch):
    """BONK is 5 decimals: 2.5 -> 250_000 raw."""
    BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

    async def fake_get_spl_decimals(mint):
        return 5

    monkeypatch.setattr(units, "get_spl_decimals", fake_get_spl_decimals)

    raw = await units.to_raw_spl(BONK, Decimal("2.5"))
    assert raw == 250_000

    back = await units.from_raw_spl(BONK, raw)
    assert back == Decimal("2.5")


@pytest.mark.unit
def test_units_error_is_exception():
    with pytest.raises(units.UnitsError):
        raise units.UnitsError("boom")
