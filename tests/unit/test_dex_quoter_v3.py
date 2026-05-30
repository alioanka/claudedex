# tests/unit/test_dex_quoter_v3.py
"""Unit tests for the Uniswap V3 QuoterV2 binding in DirectDEXExecutor.

Guards Wave-3 fix: `_quote_v3` was returning `int(amount * 0.997)`
(static 0.3% haircut), so V3 quotes were always wrong AND always tied
with whatever V3 path was first tried. After Wave-3 it:

- Calls `quoteExactInputSingle` on QuoterV2 per fee tier {100, 500,
  3000, 10000} for single-hop pairs.
- Picks the best `amountOut` across fee tiers.
- Returns 0 (NOT the placeholder ratio) when the quoter has no pool.
- Falls back to `quoteExactInput(bytes,uint256)` for multi-hop.
"""

from unittest.mock import MagicMock

import pytest


USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"


def _make_executor():
    pytest.importorskip("web3")  # CI/runner may not have web3 installed
    # Avoid the module-import-time `from eth_abi import encode_abi` that
    # broke in eth-abi 5.x: patch in a stub before importing.
    import sys
    if 'eth_abi' in sys.modules and not hasattr(sys.modules['eth_abi'], 'encode_abi'):
        sys.modules['eth_abi'].encode_abi = lambda *a, **kw: b''
    from trading.executors.direct_dex import DirectDEXExecutor
    # Bypass __init__ — it pulls in BaseExecutor + Order which require
    # full app config. We only need _quote_v3 / _get_v3_quoter callable.
    ex = object.__new__(DirectDEXExecutor)
    ex.config = {}
    ex.w3_connections = {}
    return ex


def _install_mock_quoter(executor, chain, single_outs, multi_out=None):
    """Wire a fake QuoterV2 contract for `chain` that returns
    `single_outs[fee]` from quoteExactInputSingle and `multi_out` from
    quoteExactInput. `single_outs[fee]=None` means raise (no pool).
    """

    class _FakeQuoteSingle:
        def __init__(self, fee):
            self._fee = fee

        def call(self):
            out = single_outs.get(self._fee)
            if out is None:
                raise Exception(f"no pool at fee {self._fee}")
            return (out, 0, 0, 0)

    class _FakeQuoteMulti:
        def call(self):
            if multi_out is None:
                raise Exception("no multi-hop")
            return (multi_out, [], [], 0)

    class _FakeFuncs:
        def quoteExactInputSingle(self, params):
            fee = params[3]
            return _FakeQuoteSingle(fee)

        def quoteExactInput(self, path_bytes, amount_in):
            return _FakeQuoteMulti()

    fake = MagicMock()
    fake.functions = _FakeFuncs()
    executor._v3_quoter_cache = {chain: fake}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quote_v3_picks_best_fee_tier():
    """Across fee tiers {100, 500, 3000, 10000}, _quote_v3 must pick max."""
    ex = _make_executor()
    # 500-bps tier has the deepest liquidity, returns the most output
    _install_mock_quoter(
        ex,
        'ethereum',
        {100: 990_000, 500: 999_000, 3000: 997_000, 10000: 990_000},
    )
    contract_unused = MagicMock()
    out = await ex._quote_v3(contract_unused, [USDC, WETH], 1_000_000, 'ethereum')
    assert out == 999_000, f"expected best-tier 999000, got {out}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quote_v3_returns_zero_when_no_pool():
    """If no fee tier has a pool, _quote_v3 returns 0 — NOT the
    legacy `int(amount * 0.997)` placeholder.
    """
    ex = _make_executor()
    from trading.executors.direct_dex import UNISWAP_V3_FEE_TIERS
    _install_mock_quoter(
        ex,
        'ethereum',
        {fee: None for fee in UNISWAP_V3_FEE_TIERS},
    )
    contract_unused = MagicMock()
    amount = 1_000_000
    out = await ex._quote_v3(contract_unused, [USDC, WETH], amount, 'ethereum')
    assert out == 0
    # Must NOT silently report the old placeholder ratio
    assert out != int(amount * 0.997)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quote_v3_skips_missing_tiers():
    """Pools may exist at 3000 but not 100/500 — must still return the
    3000-tier output, not raise.
    """
    ex = _make_executor()
    _install_mock_quoter(
        ex,
        'ethereum',
        {100: None, 500: None, 3000: 997_000, 10000: None},
    )
    contract_unused = MagicMock()
    out = await ex._quote_v3(contract_unused, [USDC, WETH], 1_000_000, 'ethereum')
    assert out == 997_000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quote_v3_multi_hop_fallback():
    """3-token paths must fall back to quoteExactInput(bytes,uint256)."""
    ex = _make_executor()
    _install_mock_quoter(ex, 'ethereum', single_outs={}, multi_out=995_000)
    contract_unused = MagicMock()
    out = await ex._quote_v3(
        contract_unused, [USDC, WETH, DAI], 1_000_000, 'ethereum'
    )
    assert out == 995_000


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quote_v3_no_quoter_address_returns_zero():
    """For a chain with no QuoterV2 deployment, return 0 not placeholder."""
    ex = _make_executor()
    contract_unused = MagicMock()
    out = await ex._quote_v3(contract_unused, [USDC, WETH], 1_000_000, 'fantom')
    assert out == 0


@pytest.mark.unit
def test_quoter_v2_addresses_cover_major_chains():
    """Sanity: we have QuoterV2 addresses for every EVM chain DEX trades."""
    pytest.importorskip("web3")
    from trading.executors.direct_dex import UNISWAP_V3_QUOTER_V2_ADDRESSES
    required = {'ethereum', 'polygon', 'arbitrum', 'base', 'optimism', 'bsc'}
    have = set(UNISWAP_V3_QUOTER_V2_ADDRESSES.keys())
    assert required.issubset(have), f"missing chains: {required - have}"
