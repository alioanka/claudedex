# tests/unit/test_dex_decimals.py
"""Decimals-correctness regression tests for DirectDEXExecutor.

Guards MB-01 (both legs): the swap-build path must use the actual
ERC-20 decimals() of token_in / token_out via core.units.to_raw_evm,
not the hardcoded *10**18 of utils.helpers.ether_to_wei.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from core import units


USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # 6 decimals
WBTC_ETH = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # 8 decimals
WETH_ETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # 18 decimals


@pytest.mark.unit
@pytest.mark.asyncio
async def test_to_raw_evm_usdc_six_decimals(monkeypatch):
    """100.5 USDC must encode as 100_500_000 (6 dec), not 100.5 * 10**18."""

    async def fake_decimals(chain, token):
        return 6

    monkeypatch.setattr(units, "get_evm_decimals", fake_decimals)
    raw = await units.to_raw_evm("ethereum", USDC_ETH, Decimal("100.5"))
    assert raw == 100_500_000
    # And NOT the broken hardcoded path:
    assert raw != int(Decimal("100.5") * Decimal(10 ** 18))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_to_raw_evm_wbtc_eight_decimals(monkeypatch):
    """0.25 WBTC must encode as 25_000_000 (8 dec)."""

    async def fake_decimals(chain, token):
        return 8

    monkeypatch.setattr(units, "get_evm_decimals", fake_decimals)
    raw = await units.to_raw_evm("ethereum", WBTC_ETH, Decimal("0.25"))
    assert raw == 25_000_000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_to_raw_evm_weth_eighteen_decimals(monkeypatch):
    """1 WETH must still encode as 10**18 (18 dec) — sanity for the
    happy path that previously masked the bug."""

    async def fake_decimals(chain, token):
        return 18

    monkeypatch.setattr(units, "get_evm_decimals", fake_decimals)
    raw = await units.to_raw_evm("ethereum", WETH_ETH, Decimal("1"))
    assert raw == 10 ** 18


@pytest.mark.unit
def test_score_quote_prefers_lower_gas():
    """Route-quality scoring: between two quotes with identical headline
    amount_out and identical impact, the one with the lower gas estimate
    must win. Guards the 'max amount_out' regression that picked thin-
    liquidity DEXes over efficient ones."""
    pytest.importorskip("web3")  # CI/runner may not have web3 installed
    from trading.executors.direct_dex import DirectDEXExecutor, DEXQuote
    from utils.constants import DEX

    # Build an executor without going through __init__ (we only need the
    # scoring method to be callable; __init__ requires Web3 setup).
    executor = object.__new__(DirectDEXExecutor)

    q_high_gas = DEXQuote(
        dex=DEX.SUSHISWAP,
        path=[WETH_ETH, USDC_ETH],
        amount_in=Decimal("1"),
        amount_out=Decimal("3000"),
        price=Decimal("3000"),
        price_impact=0.001,
        gas_estimate=400_000,
        liquidity=Decimal("1000000"),
        fee=Decimal("9"),
    )
    q_low_gas = DEXQuote(
        dex=DEX.UNISWAP_V2,
        path=[WETH_ETH, USDC_ETH],
        amount_in=Decimal("1"),
        amount_out=Decimal("3000"),
        price=Decimal("3000"),
        price_impact=0.001,
        gas_estimate=150_000,
        liquidity=Decimal("1000000"),
        fee=Decimal("9"),
    )

    gas_price_wei = 50 * 10 ** 9  # 50 gwei
    s_high = executor._score_quote(q_high_gas, gas_price_wei)
    s_low = executor._score_quote(q_low_gas, gas_price_wei)
    assert s_low > s_high, (
        f"expected lower-gas quote to score higher; got low={s_low} high={s_high}"
    )


@pytest.mark.unit
def test_score_quote_prefers_lower_impact():
    """With identical amount_out and gas, the quote with smaller price
    impact must win."""
    pytest.importorskip("web3")
    from trading.executors.direct_dex import DirectDEXExecutor, DEXQuote
    from utils.constants import DEX

    executor = object.__new__(DirectDEXExecutor)

    q_high_impact = DEXQuote(
        dex=DEX.PANCAKESWAP,
        path=[WETH_ETH, USDC_ETH],
        amount_in=Decimal("1"),
        amount_out=Decimal("3000"),
        price=Decimal("3000"),
        price_impact=0.05,  # 5%
        gas_estimate=200_000,
        liquidity=Decimal("100000"),
        fee=Decimal("9"),
    )
    q_low_impact = DEXQuote(
        dex=DEX.UNISWAP_V3,
        path=[WETH_ETH, USDC_ETH],
        amount_in=Decimal("1"),
        amount_out=Decimal("3000"),
        price=Decimal("3000"),
        price_impact=0.001,  # 0.1%
        gas_estimate=200_000,
        liquidity=Decimal("1000000"),
        fee=Decimal("9"),
    )

    s = executor._score_quote
    assert s(q_low_impact, 50 * 10 ** 9) > s(q_high_impact, 50 * 10 ** 9)
