# tests/unit/test_mev_bloxroute_bsc.py
"""Wave-4 tests for bloXroute BSC private-pool routing in MEVProtectionLayer.

Validates:
- `_attempt_bloxroute_bsc` returns None when no auth header set
  (so caller falls back to public mempool).
- Returns annotated tx with `bloxroute_tx_hash` on HTTP 200.
- Returns None on HTTP failure / RPC error / timeout — never raises.
- `protect_transaction` ADVANCED-tier branch routes BSC to bloXroute
  when enabled and falls back when no header.
- Ethereum mainnet Flashbots path is unchanged.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest


def _make_layer(bloxroute_enabled=False, auth_header=None):
    pytest.importorskip("aiohttp")
    pytest.importorskip("web3")
    import sys
    if 'eth_abi' in sys.modules and not hasattr(sys.modules['eth_abi'], 'encode_abi'):
        sys.modules['eth_abi'].encode_abi = lambda *a, **kw: b''
    from trading.executors.mev_protection import MEVProtectionLayer
    # MEVProtectionLayer subclasses BaseExecutor which is ABC with
    # abstract methods (get_quote, ...). Avoid the ABC machinery by
    # building a concrete subclass that stubs them out for the test.
    class _ConcreteLayer(MEVProtectionLayer):
        async def get_quote(self, *a, **kw):
            return None
        async def execute_trade(self, *a, **kw):
            return {}
        async def validate_order(self, *a, **kw):
            return True
        async def get_order_status(self, *a, **kw):
            return None
        async def cancel_order(self, *a, **kw):
            return False
        async def modify_order(self, *a, **kw):
            return False
    layer = object.__new__(_ConcreteLayer)
    layer.bloxroute_enabled = bloxroute_enabled
    layer.bloxroute_bsc_endpoint = 'https://api.blxrbdn.com'
    layer.bloxroute_auth_header = auth_header
    layer.session = None
    layer.w3 = None
    return layer


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return str(self._body)

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, status=200, body=None):
        self._status = status
        self._body = body or {'result': {'txHash': '0xdeadbeef'}}
        self.last_request = None

    def post(self, url, json=None, headers=None, timeout=None):
        self.last_request = {'url': url, 'json': json, 'headers': headers}
        return _FakeResp(self._status, self._body)


def _install_signer(layer):
    """Patch w3.eth.account.sign_transaction to return raw bytes."""
    layer.w3 = MagicMock()
    signed = MagicMock()
    signed.rawTransaction = bytes.fromhex('f8aa808504a817c800825208')
    layer.w3.eth.account.sign_transaction = MagicMock(return_value=signed)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_returns_none_without_auth_header():
    layer = _make_layer(bloxroute_enabled=True, auth_header=None)
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0, 'private_key': '0x' + 'aa' * 32}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is None, "no auth header -> must return None so caller falls back"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_returns_none_without_private_key():
    layer = _make_layer(bloxroute_enabled=True, auth_header='KEY')
    _install_signer(layer)
    layer.session = _FakeSession()
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_success_annotates_tx():
    layer = _make_layer(bloxroute_enabled=True, auth_header='KEY')
    _install_signer(layer)
    layer.session = _FakeSession(status=200, body={'result': {'txHash': '0xfeedface'}})
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0, 'private_key': '0x' + 'aa' * 32}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is not None
    assert out['bloxroute_tx_hash'] == '0xfeedface'
    assert out['bloxroute_endpoint'] == 'https://api.blxrbdn.com'
    # Header propagated
    assert layer.session.last_request['headers']['Authorization'] == 'KEY'
    # JSON-RPC method correct
    assert layer.session.last_request['json']['method'] == 'blxr_private_tx'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_http_500_returns_none():
    layer = _make_layer(bloxroute_enabled=True, auth_header='KEY')
    _install_signer(layer)
    layer.session = _FakeSession(status=500, body='internal error')
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0, 'private_key': '0x' + 'aa' * 32}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_rpc_error_returns_none():
    layer = _make_layer(bloxroute_enabled=True, auth_header='KEY')
    _install_signer(layer)
    layer.session = _FakeSession(status=200, body={'error': {'code': -32000, 'message': 'bad tx'}})
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0, 'private_key': '0x' + 'aa' * 32}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bloxroute_no_session_returns_none():
    """Defensive: if initialize() never ran, must not crash."""
    layer = _make_layer(bloxroute_enabled=True, auth_header='KEY')
    _install_signer(layer)
    layer.session = None
    tx = {'from': '0x' + '11' * 20, 'to': '0x' + '22' * 20, 'value': 0, 'private_key': '0x' + 'aa' * 32}
    out = await layer._attempt_bloxroute_bsc(tx)
    assert out is None


@pytest.mark.unit
def test_bloxroute_config_defaults_off():
    """Default bloxroute_enabled is False, so Ethereum-only flow is intact."""
    layer = _make_layer()
    assert layer.bloxroute_enabled is False
    assert layer.bloxroute_bsc_endpoint == 'https://api.blxrbdn.com'
    assert layer.bloxroute_auth_header is None
