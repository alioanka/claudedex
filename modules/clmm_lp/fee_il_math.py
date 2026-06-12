"""Pure CLMM fee-vs-IL math. No I/O, no imports beyond stdlib math.

Conventions: price P = quote (assumed USD-stable) per 1 unit of token0.
Range [pa, pb], pa < P < pb at entry. All APRs are fractions/year (0.2 = 20%).

The honest model (every term is auditable):
  M(P,pa,pb)   = 2*sqrt(P) / (2*sqrt(P) - P/sqrt(pb) - sqrt(pa))
                 capital-efficiency (concentration) multiplier vs full range.
  fee_apr_pos  = M * (volume_24h * fee_rate / tvl * 365) * fee_decay
  il_apr_pos   = M * sigma^2 / 8
                 LVR (loss-versus-rebalancing) run-rate: an in-range CLMM
                 position is short gamma; its adverse-selection loss scales
                 with the SAME M as its fees. Narrow ranges are not free yield.
  rebal_apr    = rebalances_per_year * rebalance_cost_frac
                 rebalances_per_year = sigma^2 / (a*b), a=ln(P/pa), b=ln(pb/P)
                 (expected Brownian exit time from a log-price interval).
  net_apr      = fee_apr_pos - il_apr_pos - rebal_apr
"""
import math
from typing import Dict, List, Optional, Tuple

YEAR_SECONDS = 365.0 * 24 * 3600


def position_amounts(liquidity: float, p: float, pa: float, pb: float) -> Tuple[float, float]:
    """(amount0, amount1) held by a position of `liquidity` at price p."""
    if pa <= 0 or pb <= pa or liquidity < 0:
        raise ValueError("invalid range or liquidity")
    sa, sb = math.sqrt(pa), math.sqrt(pb)
    if p <= pa:
        return liquidity * (1.0 / sa - 1.0 / sb), 0.0
    if p >= pb:
        return 0.0, liquidity * (sb - sa)
    sp = math.sqrt(p)
    return liquidity * (1.0 / sp - 1.0 / sb), liquidity * (sp - sa)


def position_value(liquidity: float, p: float, pa: float, pb: float) -> float:
    """USD value of the position at price p (quote token assumed USD)."""
    a0, a1 = position_amounts(liquidity, p, pa, pb)
    return a0 * p + a1


def liquidity_for_value(value_usd: float, p: float, pa: float, pb: float) -> float:
    """Liquidity purchasable with value_usd at in-range price p."""
    if not (pa < p < pb):
        raise ValueError("entry price must be inside the range")
    denom = 2.0 * math.sqrt(p) - p / math.sqrt(pb) - math.sqrt(pa)
    if denom <= 0:
        raise ValueError("degenerate range")
    return value_usd / denom


def hodl_value(amount0_entry: float, amount1_entry: float, p: float) -> float:
    """Value of just HOLDING the entry token amounts at price p (the benchmark)."""
    return amount0_entry * p + amount1_entry


def il_usd(liquidity: float, p_entry: float, p_now: float, pa: float, pb: float) -> float:
    """Impermanent loss in USD vs HODL of the entry inventory. <= 0 always."""
    a0, a1 = position_amounts(liquidity, p_entry, pa, pb)
    return position_value(liquidity, p_now, pa, pb) - hodl_value(a0, a1, p_now)


def concentration_factor(p: float, pa: float, pb: float) -> float:
    """Capital efficiency vs a full-range position, exact at price p."""
    if not (0 < pa < p < pb):
        raise ValueError("need 0 < pa < p < pb")
    sp = math.sqrt(p)
    denom = 2.0 * sp - p / math.sqrt(pb) - math.sqrt(pa)
    return 2.0 * sp / denom


def pool_fee_apr(volume_24h_usd: float, fee_rate: float, tvl_usd: float) -> float:
    """Pro-rata pool fee APR: daily fees / TVL, annualized."""
    if tvl_usd <= 0:
        return 0.0
    return max(0.0, volume_24h_usd) * fee_rate / tvl_usd * 365.0


def lvr_apr(annual_vol: float) -> float:
    """Full-range LVR run-rate sigma^2/8 (Milionis et al.)."""
    return annual_vol * annual_vol / 8.0


def rebalances_per_year(annual_vol: float, p: float, pa: float, pb: float) -> float:
    """1 / expected-exit-time of driftless log-price BM from [ln pa, ln pb]."""
    a = math.log(p / pa)
    b = math.log(pb / p)
    if a <= 0 or b <= 0:
        return float('inf')
    return annual_vol * annual_vol / (a * b)


def expected_net_apr(
    *,
    price: float,
    price_lower: float,
    price_upper: float,
    volume_24h_usd: float,
    fee_rate: float,
    tvl_usd: float,
    annual_vol: float,
    fee_decay_factor: float = 0.7,
    rebalance_cost_frac: float = 0.003,
) -> Dict[str, float]:
    """Transparent net-of-IL expected APR breakdown for a proposed position."""
    m = concentration_factor(price, price_lower, price_upper)
    fee_pos = m * pool_fee_apr(volume_24h_usd, fee_rate, tvl_usd) * fee_decay_factor
    il_pos = m * lvr_apr(annual_vol)
    rebal = rebalances_per_year(annual_vol, price, price_lower, price_upper) * rebalance_cost_frac
    return {
        'concentration_factor': m,
        'fee_apr': fee_pos,
        'il_apr': il_pos,
        'rebalance_apr': rebal,
        'net_apr': fee_pos - il_pos - rebal,
    }


def realized_annual_vol(
    samples: List[Tuple[float, float]], min_samples: int = 12
) -> Optional[float]:
    """Annualized vol from (unix_ts, price) samples via quadratic variation.

    Robust to irregular sampling: sqrt(sum(r_i^2) / total_dt_years).
    Returns None when fewer than min_samples usable returns exist.
    """
    rets, dt_total = [], 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        if p0 > 0 and p1 > 0 and t1 > t0:
            rets.append(math.log(p1 / p0))
            dt_total += (t1 - t0) / YEAR_SECONDS
    if len(rets) < min_samples or dt_total <= 0:
        return None
    return math.sqrt(sum(r * r for r in rets) / dt_total)


def _self_test() -> None:
    # Concentration factor at geometric center matches 1/(1 - rho^-1/4).
    p, rho = 100.0, 4.0
    pa, pb = p / math.sqrt(rho), p * math.sqrt(rho)
    m = concentration_factor(p, pa, pb)
    assert abs(m - 1.0 / (1.0 - rho ** -0.25)) < 1e-9, m
    assert m > 1.0
    # Narrower range -> strictly larger M.
    assert concentration_factor(p, 95, 105) > concentration_factor(p, 80, 125)

    # Liquidity/value round-trip and IL properties.
    liq = liquidity_for_value(1000.0, p, pa, pb)
    assert abs(position_value(liq, p, pa, pb) - 1000.0) < 1e-6
    assert abs(il_usd(liq, p, p, pa, pb)) < 1e-9            # zero at entry
    for p_now in (60.0, 90.0, 110.0, 160.0):                # negative both ways
        assert il_usd(liq, p, p_now, pa, pb) < 0.0, p_now
    # Value continuous at range boundaries.
    for edge in (pa, pb):
        lo = position_value(liq, edge - 1e-7, pa, pb)
        hi = position_value(liq, edge + 1e-7, pa, pb)
        assert abs(lo - hi) < 1e-3, (edge, lo, hi)
    # Out-of-range below: all token0; above: all quote.
    a0, a1 = position_amounts(liq, pa / 2, pa, pb)
    assert a1 == 0.0 and a0 > 0
    a0, a1 = position_amounts(liq, pb * 2, pa, pb)
    assert a0 == 0.0 and a1 > 0

    # Exit-time symmetry and monotonicity: narrower -> more rebalances.
    r_wide = rebalances_per_year(0.8, p, pa, pb)
    r_narrow = rebalances_per_year(0.8, p, 95, 105.3)
    assert r_narrow > r_wide > 0
    a = math.log(pb / p)
    assert abs(rebalances_per_year(0.8, p, pa, pb) - 0.64 / (a * a)) < 1e-9

    # Net APR: decreasing in vol; fee and IL both scale with M (no free lunch).
    kw = dict(price=p, volume_24h_usd=5_000_000, fee_rate=0.0005,
              tvl_usd=10_000_000, fee_decay_factor=0.7, rebalance_cost_frac=0.003)
    lo_vol = expected_net_apr(price_lower=pa, price_upper=pb, annual_vol=0.4, **kw)
    hi_vol = expected_net_apr(price_lower=pa, price_upper=pb, annual_vol=1.2, **kw)
    assert lo_vol['net_apr'] > hi_vol['net_apr']
    assert abs(lo_vol['il_apr'] / lvr_apr(0.4) - lo_vol['concentration_factor']) < 1e-9
    # Known numeric case: M=3.4142, pool fee apr = 9.125%, vol=0.8.
    res = expected_net_apr(price_lower=pa, price_upper=pb, annual_vol=0.8, **kw)
    assert abs(res['concentration_factor'] - 3.41421356) < 1e-6
    assert abs(res['fee_apr'] - 3.41421356 * 0.09125 * 0.7) < 1e-6
    assert abs(res['il_apr'] - 3.41421356 * 0.08) < 1e-6

    # Realized vol estimator: constant prices -> 0; too few samples -> None.
    flat = [(i * 300.0, 100.0) for i in range(20)]
    assert realized_annual_vol(flat) == 0.0
    assert realized_annual_vol(flat[:5]) is None
    # 1% move every 300s should annualize to sqrt(1e-4 * YEAR/300) ~ 3.24.
    alt = [(i * 300.0, 100.0 * (1.01 if i % 2 else 1.0)) for i in range(40)]
    v = realized_annual_vol(alt)
    assert v is not None and 2.5 < v < 3.6, v
    print("fee_il_math self-test OK")


if __name__ == "__main__":
    _self_test()
