"""Pure EIP-1559 gas/priority-fee policy. No I/O — self-tested via __main__."""

from dataclasses import dataclass
from typing import Optional, Tuple

GWEI = 10 ** 9

# Chains that still use legacy gasPrice txs in this stack.
LEGACY_GAS_CHAINS = {'bsc', 'cronos', 'pulsechain'}

URGENCY_PRIORITY_MULT = {'normal': 1.0, 'fast': 2.0, 'rescue': 4.0}


@dataclass(frozen=True)
class GasQuote:
    max_fee_per_gas: int               # wei
    max_priority_fee_per_gas: int      # wei
    legacy_gas_price: Optional[int]    # wei; set only for LEGACY_GAS_CHAINS
    capped: bool                       # True when the ceiling clamped us


def _cfg_float(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def compute_gas(
    base_fee_wei: int,
    cfg: dict,
    *,
    chain: str = 'ethereum',
    urgency: str = 'normal',
    node_priority_wei: Optional[int] = None,
) -> GasQuote:
    """Fee quote: priority = max(configured floor, node hint) * urgency,
    max_fee = base_fee * max_fee_multiplier + priority, both clamped to
    gas_ceiling_gwei. A capped quote may be slow to include — that is the
    honest outcome of the ceiling, not hidden."""
    chain = (chain or 'ethereum').strip().lower()
    base_fee_wei = max(0, int(base_fee_wei or 0))

    prio_gwei = _cfg_float(cfg, f'priority_fee_gwei_{chain}',
                           _cfg_float(cfg, 'priority_fee_gwei_default', 1.5))
    priority = int(prio_gwei * GWEI)
    if node_priority_wei:
        priority = max(priority, int(node_priority_wei))
    priority = int(priority * URGENCY_PRIORITY_MULT.get(urgency, 1.0))

    mult = max(1.0, _cfg_float(cfg, 'max_fee_multiplier', 2.0))
    max_fee = int(base_fee_wei * mult) + priority

    ceiling = int(_cfg_float(cfg, 'gas_ceiling_gwei', 150.0) * GWEI)
    capped = False
    if ceiling > 0 and max_fee > ceiling:
        max_fee, capped = ceiling, True
    priority = min(priority, max_fee)  # invariant: priority <= max_fee

    legacy = None
    if chain in LEGACY_GAS_CHAINS:
        legacy = min(max(base_fee_wei + priority, priority), max_fee) or priority
    return GasQuote(max_fee, priority, legacy, capped)


def bump_for_replacement(
    prev_max_fee_wei: int,
    prev_priority_wei: int,
    cfg: dict,
) -> Tuple[int, int]:
    """Replacement fees for a stuck tx: >= node minimum +12.5%, default +15%."""
    pct = max(12.5, _cfg_float(cfg, 'replacement_bump_pct', 15.0))
    factor = 1.0 + pct / 100.0
    new_fee = max(int(prev_max_fee_wei * factor) + 1, prev_max_fee_wei + 1)
    new_prio = max(int(prev_priority_wei * factor) + 1, prev_priority_wei + 1)
    return new_fee, min(new_prio, new_fee)


if __name__ == '__main__':
    base = 20 * GWEI

    q = compute_gas(base, {})
    assert q.max_priority_fee_per_gas == int(1.5 * GWEI)
    assert q.max_fee_per_gas == 2 * base + q.max_priority_fee_per_gas
    assert not q.capped and q.legacy_gas_price is None

    # per-chain priority key beats the default key
    q2 = compute_gas(base, {'priority_fee_gwei_default': '3',
                            'priority_fee_gwei_ethereum': '0.5'})
    assert q2.max_priority_fee_per_gas == int(0.5 * GWEI)

    # node hint raises the floor; urgency multiplies
    q3 = compute_gas(base, {}, node_priority_wei=4 * GWEI, urgency='rescue')
    assert q3.max_priority_fee_per_gas == 16 * GWEI

    # ceiling clamps and keeps priority <= max_fee
    q4 = compute_gas(200 * GWEI, {'gas_ceiling_gwei': '50'}, urgency='rescue')
    assert q4.capped and q4.max_fee_per_gas == 50 * GWEI
    assert q4.max_priority_fee_per_gas <= q4.max_fee_per_gas

    # legacy chains get a gasPrice within the ceiling
    q5 = compute_gas(5 * GWEI, {}, chain='bsc')
    assert q5.legacy_gas_price is not None
    assert q5.legacy_gas_price <= q5.max_fee_per_gas

    # replacement bump is strictly increasing and >= 12.5%
    f, p = bump_for_replacement(q.max_fee_per_gas, q.max_priority_fee_per_gas, {})
    assert f > q.max_fee_per_gas * 1.125 and p > q.max_priority_fee_per_gas
    f0, p0 = bump_for_replacement(0, 0, {'replacement_bump_pct': '1'})  # floor at 12.5
    assert f0 >= 1 and p0 >= 1

    # garbage config falls back to defaults, never raises
    q6 = compute_gas(base, {'max_fee_multiplier': 'oops', 'gas_ceiling_gwei': None})
    assert q6.max_fee_per_gas > 0
    print('gas_policy self-test OK')
