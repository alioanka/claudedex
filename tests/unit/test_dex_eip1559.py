# tests/unit/test_dex_eip1559.py
"""Unit tests for the EIP-1559 fee logic in DirectDEXExecutor.

Guards Wave-3 fix #2: `_get_optimal_gas_price` returned legacy
`gasPrice` only, so post-London Ethereum txs were Type-0 and bid the
node's current base price instead of the much-cheaper tip-only model.

After Wave-3:
- `_chain_supports_1559(chain)` returns True for ETH/Polygon/Arb/Base/Op,
  False for BSC.
- `_get_eip1559_fees(chain)` computes `(maxPriorityFeePerGas,
  maxFeePerGas)` from `eth_feeHistory(5,'latest',[50])` — median of
  50th-percentile tips across the 5-block window, maxFee = next_base*2
  + tip, both clamped at per-chain gwei ceiling.
- Falls back to `w3.eth.max_priority_fee` if feeHistory fails.
- `_build_swap_transaction` writes Type-2 fields (type=2, maxFeePerGas,
  maxPriorityFeePerGas) on supported chains, plain `gasPrice` on BSC.
"""

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
    ex.max_gas_price = 50  # legacy gwei ceiling fallback
    return ex


def _install_fake_w3(executor, chain, *, base_fees, rewards, max_priority=None, gas_price=None):
    """Inject a fake Web3 with eth.fee_history returning the supplied
    history, and (optional) eth.max_priority_fee / eth.gas_price for
    the fallback path.
    """
    fake_eth = MagicMock()
    fake_eth.fee_history = MagicMock(
        return_value={'baseFeePerGas': base_fees, 'reward': rewards}
    )
    if max_priority is not None:
        # PropertyMock-ish via plain attribute
        fake_eth.max_priority_fee = max_priority
    if gas_price is not None:
        fake_eth.gas_price = gas_price
    fake_w3 = MagicMock()
    fake_w3.eth = fake_eth
    executor.w3_connections[chain] = fake_w3
    return fake_w3


@pytest.mark.unit
def test_chain_supports_1559_defaults():
    """Default 1559 support map: BSC off, ETH/Polygon/Arb/Base/Op on."""
    ex = _make_executor()
    assert ex._chain_supports_1559('ethereum') is True
    assert ex._chain_supports_1559('polygon') is True
    assert ex._chain_supports_1559('arbitrum') is True
    assert ex._chain_supports_1559('base') is True
    assert ex._chain_supports_1559('optimism') is True
    assert ex._chain_supports_1559('bsc') is False


@pytest.mark.unit
def test_chain_supports_1559_config_override():
    """Operator can flip BSC on via config override (e.g. validator upgrade)."""
    ex = _make_executor()
    ex.config = {'chain_supports_1559': {'bsc': True, 'ethereum': False}}
    assert ex._chain_supports_1559('bsc') is True
    assert ex._chain_supports_1559('ethereum') is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_eip1559_fees_median_tip():
    """Median of 50th-percentile tips across feeHistory window — tip
    must equal the median of the supplied tip series, NOT just the
    last block (which would be vulnerable to one-block spikes).
    """
    ex = _make_executor()
    # 5 blocks: tips [1, 2, 3, 4, 5] gwei → median = 3 gwei
    _install_fake_w3(
        ex,
        'ethereum',
        base_fees=[10 * 10**9, 10 * 10**9, 10 * 10**9, 10 * 10**9, 10 * 10**9, 10 * 10**9],
        rewards=[[i * 10**9] for i in (1, 2, 3, 4, 5)],
    )
    out = await ex._get_eip1559_fees('ethereum')
    assert out is not None
    assert out['maxPriorityFeePerGas'] == 3 * 10**9, out
    # maxFee = next_base * 2 + tip = 10 * 2 + 3 = 23 gwei
    assert out['maxFeePerGas'] == 23 * 10**9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_eip1559_fees_clamped_at_ceiling():
    """If computed maxFee exceeds per-chain gwei cap, it must be clamped."""
    ex = _make_executor()
    # 5 blocks, tip 100 gwei, base 100 gwei → maxFee would be 300 gwei,
    # but ETH ceiling is 80 gwei → must clamp to 80 gwei.
    _install_fake_w3(
        ex,
        'ethereum',
        base_fees=[100 * 10**9] * 6,
        rewards=[[100 * 10**9]] * 5,
    )
    out = await ex._get_eip1559_fees('ethereum')
    assert out is not None
    assert out['maxFeePerGas'] == 80 * 10**9
    assert out['maxPriorityFeePerGas'] == 80 * 10**9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_eip1559_fees_fallback_to_max_priority_fee():
    """If feeHistory raises, fall back to w3.eth.max_priority_fee + 2x base."""
    ex = _make_executor()
    fake_eth = MagicMock()
    fake_eth.fee_history = MagicMock(side_effect=Exception("not supported"))
    fake_eth.max_priority_fee = 2 * 10**9   # 2 gwei
    fake_eth.gas_price = 30 * 10**9         # 30 gwei
    fake_w3 = MagicMock()
    fake_w3.eth = fake_eth
    ex.w3_connections['ethereum'] = fake_w3

    out = await ex._get_eip1559_fees('ethereum')
    assert out is not None
    # tip = 2 gwei, max = 30*2 + 2 = 62 gwei (below 80 cap)
    assert out['maxPriorityFeePerGas'] == 2 * 10**9
    assert out['maxFeePerGas'] == 62 * 10**9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_eip1559_fees_no_w3_returns_none():
    """No web3 for the chain → return None so caller falls back to legacy."""
    ex = _make_executor()
    out = await ex._get_eip1559_fees('unknown-chain')
    assert out is None
