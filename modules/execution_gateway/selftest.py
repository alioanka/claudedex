"""Offline self-test of the ExecutionGateway send pipeline. All RPC mocked —
NO real broadcasts. Run: python -m modules.execution_gateway.selftest"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dry_run import set_global_kill_switch
from modules.execution_gateway.gateway import (
    ExecutionGateway, GatewayRPCError, TxIntent,
)

PRIVATE_URL = 'https://private.test/rpc'
PUBLIC_URL = 'https://public.test/rpc'


class FakeGateway(ExecutionGateway):
    def __init__(self, cfg: dict, *, private_down: bool = False):
        super().__init__('selftest')
        self._cfg = dict(cfg)
        self._cfg_loaded_at = time.monotonic() + 3600  # pin injected config
        self.private_down = private_down
        self.calls = []
        self.reported = []

    async def _public_rpc(self, chain):
        return PUBLIC_URL

    async def _report_public(self, chain, url, ok, rate_limited=False, error=None):
        self.reported.append((url, ok))

    async def _rpc_call(self, url, method, params, timeout=15.0):
        self.calls.append((url, method))
        if method == 'eth_getBlockByNumber':
            return {'baseFeePerGas': hex(20 * 10**9)}
        if method == 'eth_getTransactionCount':
            return hex(5)
        if method == 'eth_sendRawTransaction':
            if url == PRIVATE_URL and self.private_down:
                raise GatewayRPCError('private relay down')
            return '0x' + ('ab' if url == PRIVATE_URL else 'cd') * 32


def _sign(tx):
    assert 'nonce' in tx and ('maxFeePerGas' in tx or 'gasPrice' in tx)
    return '0x' + 'ff' * 100


ETH_ON = {'private_send_enabled_ethereum': 'true',
          'private_rpc_url_ethereum': PRIVATE_URL}


def _intent(**kw):
    base = dict(chain='ethereum', tx={'to': '0x' + '11' * 20, 'value': 1},
                sign_fn=_sign, sender='0x' + '22' * 20, notional_usd=1000.0)
    base.update(kw)
    return TxIntent(**base)


async def main():
    # 1. DRY_RUN short-circuits before any RPC traffic
    gw = FakeGateway(ETH_ON)
    r = await gw.send(_intent(), dry_run=True)
    assert r.ok and r.simulated and not gw.calls and r.route == 'private'

    # 2. killswitch forces simulation even with dry_run=False
    set_global_kill_switch(True)
    r = await gw.send(_intent(), dry_run=False)
    assert r.simulated and not gw.calls
    set_global_kill_switch(False)

    # 3. live unsigned send: gas + nonce filled, signed, routed private
    r = await gw.send(_intent(), dry_run=False)
    assert r.ok and not r.simulated and r.route == 'private'
    assert r.tx_hash == '0x' + 'ab' * 32 and r.nonce == 5
    assert r.gas is not None and r.gas.max_fee_per_gas > 0
    assert (PRIVATE_URL, 'eth_sendRawTransaction') in gw.calls
    assert not any(u == PUBLIC_URL and m == 'eth_sendRawTransaction'
                   for u, m in gw.calls)
    # nonce advanced for the next send
    r2 = await gw.send(_intent(), dry_run=False)
    assert r2.nonce == 6 and gw.calls.count((PUBLIC_URL, 'eth_getTransactionCount')) == 1

    # 4. private down -> public fallback, pool_engine success reported
    gw = FakeGateway(ETH_ON, private_down=True)
    r = await gw.send(_intent(), dry_run=False)
    assert r.ok and r.route == 'public' and r.fallback_used
    assert r.tx_hash == '0x' + 'cd' * 32 and (PUBLIC_URL, True) in gw.reported
    assert gw.counters['fallbacks'] == 1

    # 5. fallback disabled -> clear error, nonce rolled back for reuse
    gw = FakeGateway({**ETH_ON, 'public_fallback_enabled': 'false'},
                     private_down=True)
    r = await gw.send(_intent(), dry_run=False)
    assert not r.ok and 'no fallback' in r.error and r.nonce == 5
    r2 = await gw.send(_intent(prefer_private=False), dry_run=False)
    assert r2.ok and r2.route == 'public' and r2.nonce == 5  # nonce reused

    # 6. caller opt-out routes public without touching the private relay
    gw = FakeGateway(ETH_ON)
    r = await gw.send(_intent(prefer_private=False), dry_run=False)
    assert r.ok and r.route == 'public'
    assert (PRIVATE_URL, 'eth_sendRawTransaction') not in gw.calls

    # 7. pre-signed raw passthrough: no gas/nonce work, just broadcast
    gw = FakeGateway({})
    r = await gw.send(TxIntent(chain='ethereum', signed_raw='0x' + 'ee' * 100),
                      dry_run=False)
    assert r.ok and r.route == 'public' and r.gas is None and r.nonce is None
    assert gw.calls == [(PUBLIC_URL, 'eth_sendRawTransaction')]

    # 8. unsigned intent without signer is a clear error, not a crash
    r = await gw.send(TxIntent(chain='ethereum', tx={'to': '0x0'}), dry_run=False)
    assert not r.ok and 'sign_fn' in r.error

    print('execution_gateway selftest OK')


if __name__ == '__main__':
    asyncio.run(main())
