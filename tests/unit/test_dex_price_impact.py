# tests/unit/test_dex_price_impact.py
"""Unit tests for Wave-4 `_estimate_price_impact` chunked-quoter round-trip.

The new math:
    eff_tiny   = amount_out_tiny   / (amount * 0.01)
    eff_actual = amount_out_actual / amount
    price_impact = (eff_tiny - eff_actual) / eff_tiny

Tests mock `_simulate_swap` per-amount so the result is deterministic
and chain-/contract-independent (we are only validating the math and
the refusal gate in `get_best_quote`).
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest


def _make_executor():
    pytest.importorskip("web3")
    import sys
    if 'eth_abi' in sys.modules and not hasattr(sys.modules['eth_abi'], 'encode_abi'):
        sys.modules['eth_abi'].encode_abi = lambda *a, **kw: b''
    from trading.executors.direct_dex import DirectDEXExecutor
    ex = object.__new__(DirectDEXExecutor)
    ex.config = {}
    ex.w3_connections = {}
    ex.dex_contracts = {}
    ex.max_price_impact_bps = 200
    return ex


def _install_simulate_swap(executor, outputs_by_amount):
    """Patch `_simulate_swap` to return canned outputs keyed by amount.

    `outputs_by_amount` is a dict {amount_in: amount_out}. The function
    matches by exact amount so we can verify the impact estimator
    queries with both the tiny (1% of size) and actual amounts.
    """
    async def _fake_simulate_swap(dex, path, amount, chain):
        return outputs_by_amount.get(int(amount), 0)
    executor._simulate_swap = _fake_simulate_swap


@pytest.mark.unit
@pytest.mark.asyncio
async def test_price_impact_zero_when_linear_pool():
    """If amount_out scales linearly with amount_in, impact must be 0."""
    ex = _make_executor()
    amount = 1_000_000  # 1 USDC in 6-dec
    tiny = amount // 100  # 10_000
    # Linear pool: 1 USDC -> 0.0003 WETH at every size
    _install_simulate_swap(ex, {tiny: 3, amount: 300})
    impact = await ex._estimate_price_impact('uniswap_v3', ['A', 'B'], amount, 'ethereum')
    assert impact == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_price_impact_500_bps_concave_pool():
    """A 5% degradation between tiny and actual size = 500 bps impact."""
    ex = _make_executor()
    amount = 1_000_000
    tiny = amount // 100  # 10_000
    # tiny:  10_000 in -> 10_000 out  => eff_tiny = 1.0
    # full:  1_000_000 in -> 950_000 out => eff_actual = 0.95
    # impact = (1.0 - 0.95) / 1.0 = 0.05 = 500 bps
    _install_simulate_swap(ex, {tiny: 10_000, amount: 950_000})
    impact = await ex._estimate_price_impact('uniswap_v3', ['A', 'B'], amount, 'ethereum')
    assert impact == pytest.approx(0.05, rel=1e-6)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_price_impact_returns_zero_when_no_quote():
    """If either leg returns 0, impact must be 0 (caller can't refuse on noise)."""
    ex = _make_executor()
    amount = 1_000_000
    tiny = amount // 100
    _install_simulate_swap(ex, {tiny: 0, amount: 950_000})
    impact = await ex._estimate_price_impact('uniswap_v3', ['A', 'B'], amount, 'ethereum')
    assert impact == 0.0
    # Both zero
    _install_simulate_swap(ex, {})
    impact = await ex._estimate_price_impact('uniswap_v3', ['A', 'B'], amount, 'ethereum')
    assert impact == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_price_impact_clamps_negative_to_zero():
    """If actual leg returns more per unit than tiny (numerical noise on
    very deep pools), the formula could go negative — clamp to 0."""
    ex = _make_executor()
    amount = 1_000_000
    tiny = amount // 100  # 10_000
    # tiny: 10_000 in -> 10_000 out  => eff_tiny = 1.0
    # full: 1_000_000 in -> 1_000_001 out => eff_actual > 1.0
    _install_simulate_swap(ex, {tiny: 10_000, amount: 1_000_001})
    impact = await ex._estimate_price_impact('uniswap_v3', ['A', 'B'], amount, 'ethereum')
    assert impact == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_best_quote_refuses_when_impact_exceeds_cap():
    """`get_best_quote` must drop any DEXQuote whose price_impact > cap
    (200 bps default) and return None when nothing passes."""
    pytest.importorskip("web3")
    import sys
    if 'eth_abi' in sys.modules and not hasattr(sys.modules['eth_abi'], 'encode_abi'):
        sys.modules['eth_abi'].encode_abi = lambda *a, **kw: b''
    from trading.executors.direct_dex import DirectDEXExecutor, DEXQuote
    from utils.constants import DEX

    ex = object.__new__(DirectDEXExecutor)
    ex.config = {}
    ex.w3_connections = {}
    # Pretend two dexes are wired so the loop runs twice
    ex.dex_contracts = {'ethereum': {'UNISWAP_V2': MagicMock(), 'UNISWAP_V3': MagicMock()}}
    ex.max_price_impact_bps = 200  # 2%

    async def _fake_get_dex_quote(dex_name, contract, t_in, t_out, amount, chain):
        # Both DEXes return a quote with impact = 5% (way over cap)
        return DEXQuote(
            dex=DEX[dex_name],
            path=[t_in, t_out],
            amount_in=amount,
            amount_out=Decimal('0.95'),
            price=Decimal('0.95'),
            price_impact=0.05,  # 500 bps
            gas_estimate=180_000,
            liquidity=Decimal('1000000'),
            fee=Decimal('0.003'),
        )

    async def _fake_gas_price(chain):
        return 30 * 10 ** 9

    ex._get_dex_quote = _fake_get_dex_quote
    ex._get_optimal_gas_price = _fake_gas_price

    out = await ex.get_best_quote('TKN_A', 'TKN_B', Decimal('1'), 'ethereum')
    assert out is None, "high-impact quotes must be refused"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_best_quote_keeps_quote_under_cap():
    """A quote at exactly the cap (200 bps) must still be returned."""
    pytest.importorskip("web3")
    import sys
    if 'eth_abi' in sys.modules and not hasattr(sys.modules['eth_abi'], 'encode_abi'):
        sys.modules['eth_abi'].encode_abi = lambda *a, **kw: b''
    from trading.executors.direct_dex import DirectDEXExecutor, DEXQuote
    from utils.constants import DEX

    ex = object.__new__(DirectDEXExecutor)
    ex.config = {}
    ex.w3_connections = {}
    ex.dex_contracts = {'ethereum': {'UNISWAP_V3': MagicMock()}}
    ex.max_price_impact_bps = 200

    async def _fake_get_dex_quote(dex_name, contract, t_in, t_out, amount, chain):
        return DEXQuote(
            dex=DEX[dex_name],
            path=[t_in, t_out],
            amount_in=amount,
            amount_out=Decimal('0.98'),
            price=Decimal('0.98'),
            price_impact=0.02,  # exactly 200 bps
            gas_estimate=180_000,
            liquidity=Decimal('1000000'),
            fee=Decimal('0.003'),
        )

    async def _fake_gas_price(chain):
        return 30 * 10 ** 9

    ex._get_dex_quote = _fake_get_dex_quote
    ex._get_optimal_gas_price = _fake_gas_price

    out = await ex.get_best_quote('TKN_A', 'TKN_B', Decimal('1'), 'ethereum')
    assert out is not None
    assert out.price_impact == 0.02
