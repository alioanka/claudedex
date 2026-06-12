"""PURE spread / cointegration / z-score / half-life math for stat_arb.

Stdlib-only (no numpy, no I/O, no DB) so it is offline self-testable:

    python -m modules.stat_arb.core.pair_math

Conventions
-----------
- All price series are OLDEST-FIRST lists of floats (closes of CLOSED bars
  only — the caller must never feed a forming bar, that would be look-ahead).
- The spread is  s_t = ln(y_t) - beta * ln(x_t) - alpha  with (alpha, beta)
  from OLS of ln(y) on ln(x) over the lookback window.
- z-score uses the rolling mean/std of the SAME window the OLS was fit on,
  i.e. everything is computed from data available at decision time.

Mean-reversion gate (honesty note)
----------------------------------
`ar1_tstat` regresses  Δs_t = a + b * s_{t-1}  and returns (b, t-stat of b).
Requiring t-stat <= adf_tstat_max (default -2.9, the approximate 5% ADF
critical value, no-trend case) is an Engle-Granger-STYLE test, not a full ADF
with lag selection. It is deliberately conservative and transparent; crypto
cointegration is regime-fragile regardless, which is why the engine pairs this
gate with a HARD z-stop and per-pair cooldown (no averaging down, ever).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Basic regression / stats primitives
# ---------------------------------------------------------------------------

def ols(x: Sequence[float], y: Sequence[float]) -> Optional[Tuple[float, float]]:
    """OLS fit y = alpha + beta*x. Returns (alpha, beta) or None if degenerate."""
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0.0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    beta = sxy / sxx
    alpha = my - beta * mx
    return alpha, beta


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or None if degenerate."""
    n = len(a)
    if n < 3 or n != len(b):
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((ai - ma) ** 2 for ai in a)
    vb = sum((bi - mb) ** 2 for bi in b)
    if va <= 0.0 or vb <= 0.0:
        return None
    cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
    return cov / math.sqrt(va * vb)


def log_returns(closes: Sequence[float]) -> List[float]:
    """ln(p_t / p_{t-1}); skips non-positive prices defensively."""
    out: List[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0.0 and cur > 0.0:
            out.append(math.log(cur / prev))
    return out


# ---------------------------------------------------------------------------
# Spread construction + diagnostics
# ---------------------------------------------------------------------------

def fit_spread(closes_y: Sequence[float],
               closes_x: Sequence[float]) -> Optional[dict]:
    """Fit ln(y) = alpha + beta*ln(x); return spread series + params.

    Returns dict(alpha, beta, spread: List[float]) or None when the fit is
    impossible or beta is non-positive (a negative-beta 'pair' is not the
    long-cheap/short-rich trade this module makes).
    """
    n = min(len(closes_y), len(closes_x))
    if n < 30:
        return None
    ys = [math.log(p) for p in closes_y[-n:] if p > 0.0]
    xs = [math.log(p) for p in closes_x[-n:] if p > 0.0]
    if len(ys) != n or len(xs) != n:
        return None
    fit = ols(xs, ys)
    if fit is None:
        return None
    alpha, beta = fit
    if beta <= 0.0:
        return None
    spread = [yi - beta * xi - alpha for xi, yi in zip(xs, ys)]
    return {"alpha": alpha, "beta": beta, "spread": spread}


def zscore(series: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    """(z of the LAST point, mean, std) over the whole series; None if flat."""
    n = len(series)
    if n < 10:
        return None
    mean = sum(series) / n
    var = sum((s - mean) ** 2 for s in series) / (n - 1)
    std = math.sqrt(var)
    if std <= 1e-12:
        return None
    return (series[-1] - mean) / std, mean, std


def ar1_tstat(spread: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Regress Δs_t = a + b*s_{t-1}; return (b, t-stat of b). See module doc."""
    n = len(spread)
    if n < 30:
        return None
    lagged = list(spread[:-1])
    diffs = [spread[i + 1] - spread[i] for i in range(n - 1)]
    fit = ols(lagged, diffs)
    if fit is None:
        return None
    a, b = fit
    m = len(lagged)
    resid = [d - (a + b * lx) for lx, d in zip(lagged, diffs)]
    dof = m - 2
    if dof <= 0:
        return None
    s2 = sum(r * r for r in resid) / dof
    mlx = sum(lagged) / m
    sxx = sum((lx - mlx) ** 2 for lx in lagged)
    if sxx <= 0.0 or s2 <= 0.0:
        return None
    se_b = math.sqrt(s2 / sxx)
    if se_b <= 0.0:
        return None
    return b, b / se_b


def half_life(spread: Sequence[float]) -> Optional[float]:
    """Mean-reversion half-life in bars from the AR(1) coefficient.

    half_life = -ln(2) / ln(1 + b), valid only for -1 < b < 0
    (b >= 0 means no reversion; b <= -1 means oscillation, both -> None).
    """
    res = ar1_tstat(spread)
    if res is None:
        return None
    b, _ = res
    if not (-1.0 < b < 0.0):
        return None
    return -math.log(2.0) / math.log(1.0 + b)


# ---------------------------------------------------------------------------
# Pair evaluation + trade decision (the only two entry points the engine uses)
# ---------------------------------------------------------------------------

def evaluate_pair(closes_y: Sequence[float], closes_x: Sequence[float], *,
                  min_correlation: float = 0.6,
                  adf_tstat_max: float = -2.9,
                  min_half_life_bars: float = 4.0,
                  max_half_life_bars: float = 120.0) -> dict:
    """Full pair diagnostic. Never raises.

    Returns a dict with keys: tradable (bool), reason (str — first failed
    gate, or 'ok'), and whichever of correlation/alpha/beta/spread/
    spread_mean/spread_std/zscore/ar1_b/ar1_tstat/half_life_bars were
    computable (absent ones are None).
    """
    out = {"tradable": False, "reason": "insufficient_data",
           "correlation": None, "alpha": None, "beta": None,
           "spread": None, "spread_mean": None, "spread_std": None,
           "zscore": None, "ar1_b": None, "ar1_tstat": None,
           "half_life_bars": None}

    corr = pearson(log_returns(closes_y), log_returns(closes_x))
    out["correlation"] = corr
    if corr is None:
        return out
    if corr < min_correlation:
        out["reason"] = "low_correlation"
        return out

    fitted = fit_spread(closes_y, closes_x)
    if fitted is None:
        out["reason"] = "spread_fit_failed"
        return out
    out["alpha"] = fitted["alpha"]
    out["beta"] = fitted["beta"]
    spread = fitted["spread"]
    out["spread"] = spread[-1]

    z = zscore(spread)
    if z is None:
        out["reason"] = "flat_spread"
        return out
    out["zscore"], out["spread_mean"], out["spread_std"] = z

    ar1 = ar1_tstat(spread)
    if ar1 is None:
        out["reason"] = "ar1_fit_failed"
        return out
    out["ar1_b"], out["ar1_tstat"] = ar1
    if out["ar1_tstat"] > adf_tstat_max:
        out["reason"] = "not_mean_reverting"
        return out

    hl = half_life(spread)
    out["half_life_bars"] = hl
    if hl is None or not (min_half_life_bars <= hl <= max_half_life_bars):
        out["reason"] = "half_life_out_of_band"
        return out

    out["tradable"] = True
    out["reason"] = "ok"
    return out


def decide(z: Optional[float], in_position: bool, position_side: Optional[str],
           *, entry_z: float = 2.0, exit_z: float = 0.5,
           stop_z: float = 4.0) -> str:
    """Trade decision from the current z. Pure; enforces the tail-risk rules.

    position_side: None, 'short_spread' (entered at z>0: short y / long x)
    or 'long_spread' (entered at z<0: long y / short x).

    Returns one of:
      ENTER_SHORT_SPREAD / ENTER_LONG_SPREAD   (only when flat)
      EXIT_MEAN_REVERT / EXIT_HARD_STOP        (only when in position)
      HOLD / NONE

    Invariants enforced here:
    - never enter when |z| >= stop_z (do not step into a blown-out spread);
    - the HARD stop fires the moment |z| >= stop_z AND z has moved AGAINST the
      position — there is no averaging down and no stop widening by design;
    - config must satisfy stop_z > entry_z > exit_z >= 0 or we refuse (NONE).
    """
    if z is None or not (stop_z > entry_z > exit_z >= 0.0):
        return "NONE"

    if not in_position:
        if abs(z) >= stop_z:
            return "NONE"  # blown-out spread: not an entry, a pair break
        if z >= entry_z:
            return "ENTER_SHORT_SPREAD"   # spread rich: short y, long x
        if z <= -entry_z:
            return "ENTER_LONG_SPREAD"    # spread cheap: long y, short x
        return "NONE"

    # In position: hard stop only when z moved against the entry direction.
    if position_side == "short_spread":
        if z >= stop_z:
            return "EXIT_HARD_STOP"
        if z <= exit_z:           # reverted to the band OR overshot past it
            return "EXIT_MEAN_REVERT"
        return "HOLD"
    if position_side == "long_spread":
        if z <= -stop_z:
            return "EXIT_HARD_STOP"
        if z >= -exit_z:
            return "EXIT_MEAN_REVERT"
        return "HOLD"
    return "NONE"


def simulated_pair_pnl_usd(*, notional_per_leg_usd: float,
                           long_entry: float, long_exit: float,
                           short_entry: float, short_exit: float,
                           taker_fee_bps: float = 5.5,
                           slippage_bps: float = 3.0) -> Optional[float]:
    """Dollar-neutral two-leg PnL net of modeled costs (4 fills total).

    Costs: (taker_fee + slippage) bps on EACH of the 4 fills' notional.
    Returns None on non-positive prices/notional (refuses to fabricate PnL).
    """
    if (notional_per_leg_usd <= 0 or long_entry <= 0 or long_exit <= 0
            or short_entry <= 0 or short_exit <= 0):
        return None
    gross = notional_per_leg_usd * ((long_exit / long_entry - 1.0)
                                    - (short_exit / short_entry - 1.0))
    cost = 4.0 * notional_per_leg_usd * (taker_fee_bps + slippage_bps) / 10000.0
    return gross - cost


# ---------------------------------------------------------------------------
# Self-test (offline, deterministic)
# ---------------------------------------------------------------------------

def _self_test() -> None:  # pragma: no cover - exercised via __main__
    import random
    rng = random.Random(42)

    # 1) Synthetic cointegrated pair: x random walk, ln(y)=0.1+0.8*ln(x)+OU(s).
    n = 400
    lx = [math.log(100.0)]
    for _ in range(n - 1):
        lx.append(lx[-1] + rng.gauss(0.0, 0.01))
    s, spread_true = 0.0, []
    for _ in range(n):
        s = 0.9 * s + rng.gauss(0.0, 0.004)   # AR(1), phi=0.9 -> HL ~ 6.6 bars
        spread_true.append(s)
    ly = [0.1 + 0.8 * xi + si for xi, si in zip(lx, spread_true)]
    x = [math.exp(v) for v in lx]
    y = [math.exp(v) for v in ly]

    ev = evaluate_pair(y, x)
    assert ev["tradable"], f"cointegrated pair must be tradable: {ev['reason']}"
    assert abs(ev["beta"] - 0.8) < 0.1, f"beta off: {ev['beta']}"
    assert 3.0 < ev["half_life_bars"] < 20.0, f"HL off: {ev['half_life_bars']}"
    assert ev["ar1_tstat"] <= -2.9

    # 2) Independent random walks must NOT be tradable.
    lz = [math.log(50.0)]
    for _ in range(n - 1):
        lz.append(lz[-1] + rng.gauss(0.0, 0.012))
    z_prices = [math.exp(v) for v in lz]
    ev2 = evaluate_pair(z_prices, x)
    assert not ev2["tradable"], "independent walks must fail a gate"

    # 3) Decision logic invariants.
    assert decide(2.5, False, None) == "ENTER_SHORT_SPREAD"
    assert decide(-2.5, False, None) == "ENTER_LONG_SPREAD"
    assert decide(1.0, False, None) == "NONE"
    assert decide(4.5, False, None) == "NONE", "never enter past the hard stop"
    assert decide(4.0, True, "short_spread") == "EXIT_HARD_STOP"
    assert decide(-4.0, True, "long_spread") == "EXIT_HARD_STOP"
    assert decide(-4.0, True, "short_spread") == "EXIT_MEAN_REVERT", \
        "favorable overshoot past the mean is a take-profit, not a hold"
    assert decide(0.3, True, "short_spread") == "EXIT_MEAN_REVERT"
    assert decide(1.5, True, "short_spread") == "HOLD"
    assert decide(-1.5, True, "long_spread") == "HOLD"
    assert decide(0.3, True, "long_spread") == "EXIT_MEAN_REVERT"
    assert decide(2.0, False, None, entry_z=2.0, exit_z=2.5,
                  stop_z=4.0) == "NONE", "bad config must refuse"
    assert decide(None, False, None) == "NONE"

    # 4) PnL: symmetric flat round-trip loses exactly the cost; sanity signs.
    flat = simulated_pair_pnl_usd(notional_per_leg_usd=100.0,
                                  long_entry=10, long_exit=10,
                                  short_entry=20, short_exit=20)
    assert flat is not None and abs(flat - (-4 * 100 * 8.5 / 10000)) < 1e-9
    win = simulated_pair_pnl_usd(notional_per_leg_usd=100.0,
                                 long_entry=10, long_exit=11,
                                 short_entry=20, short_exit=19)
    assert win is not None and win > 0
    assert simulated_pair_pnl_usd(notional_per_leg_usd=0, long_entry=1,
                                  long_exit=1, short_entry=1,
                                  short_exit=1) is None

    # 5) Half-life numerical check: AR(1) phi=0.9 -> b=-0.1 -> HL=6.58.
    hl = -math.log(2.0) / math.log(0.9)
    assert abs(hl - 6.5788) < 0.01

    print("pair_math self-test OK "
          f"(coint pair: beta={ev['beta']:.3f} HL={ev['half_life_bars']:.1f} "
          f"t={ev['ar1_tstat']:.2f} z={ev['zscore']:.2f}; "
          f"independent pair rejected: {ev2['reason']})")


if __name__ == "__main__":
    _self_test()
