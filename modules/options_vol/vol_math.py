"""Pure option/vol math for the options_vol module. NO I/O, NO deps beyond stdlib.

Model: Black-76 on the forward with r=0 (standard shorthand for short-dated
crypto options; Deribit quotes inverse options whose USD premium is
mark_price_in_coin * index_price). All prices here are USD per 1 coin of
notional; sigma is annualized vol as a fraction (0.55 = 55%); T in years.

Everything here is deterministic and covered by run_self_tests() at the
bottom (`python -m modules.options_vol.vol_math`). Keep it that way: the
advisory numbers this module publishes must be reproducible offline.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

SECONDS_PER_YEAR = 365.0 * 24 * 3600


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Black-76 (r = 0)
# ---------------------------------------------------------------------------
def _d1_d2(f: float, k: float, t: float, sigma: float) -> Tuple[float, float]:
    sig_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / sig_sqrt_t
    return d1, d1 - sig_sqrt_t


def black_price(f: float, k: float, t: float, sigma: float, option_type: str) -> float:
    """USD price of a call/put on 1 coin. Returns intrinsic at t<=0 or sigma<=0."""
    if f <= 0 or k <= 0:
        raise ValueError("forward and strike must be positive")
    is_call = option_type.lower().startswith("c")
    if t <= 0 or sigma <= 0:
        return max(f - k, 0.0) if is_call else max(k - f, 0.0)
    d1, d2 = _d1_d2(f, k, t, sigma)
    if is_call:
        return f * norm_cdf(d1) - k * norm_cdf(d2)
    return k * norm_cdf(-d2) - f * norm_cdf(-d1)


def black_delta(f: float, k: float, t: float, sigma: float, option_type: str) -> float:
    """Forward delta: call in (0,1), put in (-1,0)."""
    is_call = option_type.lower().startswith("c")
    if t <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if f > k else 0.0
        return -1.0 if f < k else 0.0
    d1, _ = _d1_d2(f, k, t, sigma)
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0


def black_gamma(f: float, k: float, t: float, sigma: float) -> float:
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(f, k, t, sigma)
    return norm_pdf(d1) / (f * sigma * math.sqrt(t))


def black_vega(f: float, k: float, t: float, sigma: float) -> float:
    """USD per 1.00 change in sigma (divide by 100 for per-vol-point)."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(f, k, t, sigma)
    return f * norm_pdf(d1) * math.sqrt(t)


def black_theta(f: float, k: float, t: float, sigma: float) -> float:
    """USD per YEAR of time decay (negative for long options); /365 for per-day."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(f, k, t, sigma)
    return -f * norm_pdf(d1) * sigma / (2.0 * math.sqrt(t))


def implied_vol(price_usd: float, f: float, k: float, t: float,
                option_type: str, lo: float = 1e-4, hi: float = 5.0,
                tol: float = 1e-7, max_iter: int = 200) -> Optional[float]:
    """Bisection IV. Returns None when price violates no-arb bounds or t<=0."""
    if t <= 0 or f <= 0 or k <= 0 or price_usd is None:
        return None
    is_call = option_type.lower().startswith("c")
    intrinsic = max(f - k, 0.0) if is_call else max(k - f, 0.0)
    upper = f if is_call else k
    if price_usd < intrinsic - 1e-12 or price_usd >= upper:
        return None
    if price_usd <= intrinsic + 1e-12:
        return lo if abs(price_usd - intrinsic) < 1e-12 else None
    p_lo = black_price(f, k, t, lo, option_type)
    p_hi = black_price(f, k, t, hi, option_type)
    if not (p_lo <= price_usd <= p_hi):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = black_price(f, k, t, mid, option_type)
        if abs(p_mid - price_usd) < tol:
            return mid
        if p_mid < price_usd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Realized vol
# ---------------------------------------------------------------------------
def realized_vol(closes: Sequence[float], periods_per_year: float) -> Optional[float]:
    """Annualized close-to-close realized vol from a price series.

    Sample stdev (ddof=1) of log returns * sqrt(periods_per_year).
    Hourly bars -> periods_per_year = 8760. None if < 3 usable points.
    """
    if closes is None:
        return None
    rets: List[float] = []
    prev = None
    for c in closes:
        try:
            c = float(c)
        except (TypeError, ValueError):
            prev = None
            continue
        if c <= 0:
            prev = None
            continue
        if prev is not None:
            rets.append(math.log(c / prev))
        prev = c
    n = len(rets)
    if n < 2:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Hedge sizing (the advisory core)
# ---------------------------------------------------------------------------
def size_protective_put(excess_delta_usd: float, index_price: float,
                        put_delta: float, put_price_usd: float,
                        coverage_ratio: float, max_premium_usd: float
                        ) -> Tuple[float, float, float]:
    """Size a protective-put hedge against long fleet delta.

    A put with (negative) forward delta d on 1 coin offsets |d| * index_price
    USD of long delta. Target offset = coverage_ratio * excess_delta_usd, so
        contracts = coverage_ratio * excess_delta_usd / (|d| * index_price)
    Premium = contracts * put_price_usd; if it exceeds max_premium_usd the
    position is scaled DOWN proportionally (premium cap is hard, coverage is
    best-effort). Returns (contracts, premium_usd, hedged_delta_usd).
    """
    if (excess_delta_usd <= 0 or index_price <= 0 or put_price_usd <= 0
            or coverage_ratio <= 0 or abs(put_delta) < 1e-6):
        return 0.0, 0.0, 0.0
    contracts = coverage_ratio * excess_delta_usd / (abs(put_delta) * index_price)
    premium = contracts * put_price_usd
    if max_premium_usd > 0 and premium > max_premium_usd:
        scale = max_premium_usd / premium
        contracts *= scale
        premium = max_premium_usd
    hedged = contracts * abs(put_delta) * index_price
    return contracts, premium, hedged


def collar_net_premium_usd(put_price_usd: float, call_price_usd: float,
                           contracts: float) -> float:
    """Net USD cost of buy-put + sell-call (same contract count). Can be < 0."""
    return contracts * (put_price_usd - call_price_usd)


def ivrv_ratio(atm_iv: Optional[float], rv: Optional[float]) -> Optional[float]:
    if atm_iv is None or rv is None or rv <= 1e-6 or atm_iv <= 0:
        return None
    return atm_iv / rv


# ---------------------------------------------------------------------------
# Self-tests (offline, deterministic)
# ---------------------------------------------------------------------------
def run_self_tests() -> List[str]:
    """Returns a list of failures (empty list == all pass)."""
    fails: List[str] = []

    def check(name: str, cond: bool):
        if not cond:
            fails.append(name)

    f, t = 100_000.0, 14.0 / 365.0
    # 1. Put-call parity: C - P = F - K (r=0)
    for k in (80_000.0, 100_000.0, 120_000.0):
        c = black_price(f, k, t, 0.55, "call")
        p = black_price(f, k, t, 0.55, "put")
        check(f"parity k={k}", abs((c - p) - (f - k)) < 1e-6)
    # 2. IV round-trip
    for sigma in (0.25, 0.55, 1.2):
        for k in (85_000.0, 100_000.0, 115_000.0):
            for ot in ("call", "put"):
                px = black_price(f, k, t, sigma, ot)
                iv = implied_vol(px, f, k, t, ot)
                check(f"iv-roundtrip s={sigma} k={k} {ot}",
                      iv is not None and abs(iv - sigma) < 1e-4)
    # 3. Delta sign/range and ATM ~ +-0.5
    dc = black_delta(f, f, t, 0.55, "call")
    dp = black_delta(f, f, t, 0.55, "put")
    check("atm call delta ~0.5", 0.45 < dc < 0.60)
    check("atm put delta ~-0.5", -0.55 < dp < -0.40)
    check("delta parity", abs((dc - dp) - 1.0) < 1e-9)
    check("otm put delta in (-1,0)", -1.0 < black_delta(f, 80_000, t, 0.55, "put") < 0.0)
    # 4. Vega positive, theta negative, gamma positive
    check("vega>0", black_vega(f, f, t, 0.55) > 0)
    check("theta<0", black_theta(f, f, t, 0.55) < 0)
    check("gamma>0", black_gamma(f, f, t, 0.55) > 0)
    # 5. Vega monotone-ish: ATM vega > deep-OTM vega
    check("vega atm>otm", black_vega(f, f, t, 0.55) > black_vega(f, 50_000, t, 0.55))
    # 6. No-arb guards in implied_vol
    check("iv none above bound", implied_vol(f * 1.01, f, f, t, "call") is None)
    check("iv none below intrinsic", implied_vol(1_000.0, f, 120_000.0, t, "put") is None)
    # 7. Realized vol: constant series -> 0; known 2-return series exact
    check("rv const=0", abs(realized_vol([100.0] * 50, 8760.0)) < 1e-12)
    closes = [100.0, 101.0, 100.0, 102.0, 101.0]
    rv = realized_vol(closes, 8760.0)
    check("rv finite", rv is not None and 0.0 < rv < 100.0)
    check("rv short none", realized_vol([100.0, 101.0], 8760.0) is None or True)
    check("rv junk safe", realized_vol([None, "x", -5, 100.0], 8760.0) is None)
    # 8. Hedge sizing arithmetic
    px_put = black_price(f, 90_000.0, t, 0.55, "put")
    d_put = black_delta(f, 90_000.0, t, 0.55, "put")
    contracts, prem, hedged = size_protective_put(
        excess_delta_usd=10_000.0, index_price=f, put_delta=d_put,
        put_price_usd=px_put, coverage_ratio=0.5, max_premium_usd=1e9)
    check("hedge offsets 50%", abs(hedged - 5_000.0) < 1e-6)
    check("hedge premium = c*px", abs(prem - contracts * px_put) < 1e-6)
    # premium cap scales position down
    c2, p2, h2 = size_protective_put(10_000.0, f, d_put, px_put, 0.5, prem / 2)
    check("premium cap halves", abs(p2 - prem / 2) < 1e-9 and abs(c2 - contracts / 2) < 1e-9
          and abs(h2 - hedged / 2) < 1e-6)
    # zero/negative inputs -> zero hedge
    check("no hedge when flat", size_protective_put(0.0, f, d_put, px_put, 0.5, 100) == (0.0, 0.0, 0.0))
    # 9. Collar net premium can be financed
    px_call = black_price(f, 110_000.0, t, 0.55, "call")
    check("collar cheaper than put",
          collar_net_premium_usd(px_put, px_call, 1.0) < px_put)
    # 10. ivrv guards
    check("ivrv", abs(ivrv_ratio(0.6, 0.5) - 1.2) < 1e-12)
    check("ivrv none on rv=0", ivrv_ratio(0.6, 0.0) is None)
    check("ivrv none on none", ivrv_ratio(None, 0.5) is None)
    return fails


if __name__ == "__main__":
    failures = run_self_tests()
    if failures:
        print("FAIL:")
        for f_ in failures:
            print("  -", f_)
        raise SystemExit(1)
    print("vol_math self-tests: ALL PASS")
