"""Pure EVM send-route selection. No I/O — self-tested via __main__."""

from dataclasses import dataclass
from typing import Optional

ROUTE_PRIVATE = 'private'
ROUTE_PUBLIC = 'public'

# Chains the gateway knows how to send on (JSON-RPC eth_sendRawTransaction).
EVM_CHAINS = {
    'ethereum', 'bsc', 'polygon', 'arbitrum', 'base',
    'avalanche', 'fantom', 'cronos', 'pulsechain', 'monad',
}

# Sequencer-ordered chains: no public mempool to sandwich from, so private
# order flow buys little. Informational only — flags still decide.
SEQUENCER_ORDERED = {'arbitrum', 'base'}


@dataclass(frozen=True)
class RoutePlan:
    route: str                 # ROUTE_PRIVATE | ROUTE_PUBLIC
    fallback: Optional[str]    # ROUTE_PUBLIC | None (only set on private plans)
    reason: str


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('true', '1', 'yes', 'on')


def select_route(
    chain: str,
    cfg: dict,
    *,
    direction: str = 'entry',
    notional_usd: Optional[float] = None,
    prefer_private: Optional[bool] = None,
    private_url: Optional[str] = None,
) -> RoutePlan:
    """Decide private-order-flow vs public-mempool send for one tx intent.

    prefer_private is the caller's per-module/per-direction override:
    False = hard opt-out (e.g. sniper entries where latency IS the edge),
    True = request private even below the notional floor, None = policy.
    """
    chain = (chain or '').strip().lower()
    fallback = ROUTE_PUBLIC if _truthy(cfg.get('public_fallback_enabled', True)) else None

    if chain not in EVM_CHAINS:
        return RoutePlan(ROUTE_PUBLIC, None, f'non-EVM or unknown chain {chain!r}')
    if prefer_private is False:
        return RoutePlan(ROUTE_PUBLIC, None, 'caller opted out (prefer_private=False)')
    if not private_url:
        return RoutePlan(ROUTE_PUBLIC, None, f'no private RPC configured for {chain}')
    if not _truthy(cfg.get(f'private_send_enabled_{chain}', False)):
        return RoutePlan(ROUTE_PUBLIC, None, f'private_send_enabled_{chain} is off')
    if prefer_private is not True:
        try:
            floor = float(cfg.get('private_min_notional_usd', 0) or 0)
        except (TypeError, ValueError):
            floor = 0.0
        if floor > 0 and (notional_usd is None or notional_usd < floor):
            return RoutePlan(
                ROUTE_PUBLIC, None,
                f'notional {notional_usd} below private floor {floor}')
    note = ' (sequencer-ordered chain; private gain limited)' if chain in SEQUENCER_ORDERED else ''
    return RoutePlan(ROUTE_PRIVATE, fallback,
                     f'private order flow enabled for {chain} [{direction}]{note}')


if __name__ == '__main__':
    on = {'private_send_enabled_ethereum': 'true'}
    url = 'https://rpc.flashbots.net/fast'

    # default-safe: everything public with empty cfg
    assert select_route('ethereum', {}, private_url=url).route == ROUTE_PUBLIC
    # enabled + url -> private with public fallback
    p = select_route('ethereum', on, private_url=url)
    assert p.route == ROUTE_PRIVATE and p.fallback == ROUTE_PUBLIC
    # fallback can be disabled
    p2 = select_route('ethereum', {**on, 'public_fallback_enabled': 'false'}, private_url=url)
    assert p2.route == ROUTE_PRIVATE and p2.fallback is None
    # no url -> public even when enabled
    assert select_route('ethereum', on, private_url=None).route == ROUTE_PUBLIC
    # caller hard opt-out beats everything
    assert select_route('ethereum', on, private_url=url,
                        prefer_private=False).route == ROUTE_PUBLIC
    # notional floor gates, prefer_private=True bypasses the floor only
    floored = {**on, 'private_min_notional_usd': '500'}
    assert select_route('ethereum', floored, private_url=url,
                        notional_usd=100).route == ROUTE_PUBLIC
    assert select_route('ethereum', floored, private_url=url,
                        notional_usd=900).route == ROUTE_PRIVATE
    assert select_route('ethereum', floored, private_url=url, notional_usd=100,
                        prefer_private=True).route == ROUTE_PRIVATE
    # prefer_private=True never forces private when the chain flag is off
    assert select_route('ethereum', {}, private_url=url,
                        prefer_private=True).route == ROUTE_PUBLIC
    # non-EVM -> public
    assert select_route('solana', on, private_url=url).route == ROUTE_PUBLIC
    assert select_route('', on, private_url=url).route == ROUTE_PUBLIC
    print('route_policy self-test OK')
