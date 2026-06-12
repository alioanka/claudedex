"""Pure TCA cost math — no DB, no I/O, no clock. Self-tested.

Run the self-test: python -m modules.execution_quality.core.cost_model

Cost convention: every metric is a COST in basis points of trade notional;
positive = money lost to execution, negative = price improvement. The model
only sums components it can actually measure (`total_cost_bps` is the sum of
KNOWN components) and reports coverage honestly via `quote_covered` — a trade
without a captured quote gets NULL slippage, never a guessed one.

Formulas (every number operator-recomputable):
  fee_bps   = fee_usd   / notional_usd * 10000
  gas_bps   = gas_usd   / notional_usd * 10000
  extra_bps = extra_usd / notional_usd * 10000          # flash-loan fee, tips,
                                                        # modeled slippage already
                                                        # deducted upstream
  entry_slippage_bps (long buy / short sell entry):
      buy leg : (realized - quoted) / quoted * 10000    # paid more = cost
      sell leg: (quoted - realized) / quoted * 10000    # received less = cost
  exit leg uses the opposite direction of the entry leg.
  recorded_entry_slippage_bps is used as a FALLBACK entry slippage when the
  source table recorded a slippage figure but no quote was captured (flagged
  in components, quote_covered stays False).
  total_cost_usd  = fee + gas + extra + known slippage legs (bps -> usd)
  total_cost_bps  = total_cost_usd / notional_usd * 10000
  cost_to_gross_pct = total_cost_usd / |gross_pnl_usd| * 100
  mev_suspect = entry slippage measured AND >= sandwich_suspect_bps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BPS = 10_000.0
_EPS = 1e-12


@dataclass
class TradeCostInputs:
    """Normalized view of one closed trade from any module's table."""
    module: str
    trade_ref: str
    venue: str = ""
    side: str = "buy"                 # buy/long vs sell/short decides leg signs
    is_simulated: bool = True
    notional_usd: float = 0.0
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quoted_entry_price: Optional[float] = None   # intent capture (metadata)
    quoted_exit_price: Optional[float] = None
    recorded_entry_slippage_bps: Optional[float] = None  # source-recorded, no quote
    fee_usd: float = 0.0
    gas_usd: float = 0.0
    extra_cost_usd: float = 0.0       # flash-loan fee / tip / modeled slippage
    gross_pnl_usd: Optional[float] = None
    net_pnl_usd: Optional[float] = None
    trade_time: Any = None
    components: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeCostResult:
    module: str
    trade_ref: str
    venue: str
    side: str
    is_simulated: bool
    notional_usd: float
    fee_bps: Optional[float]
    gas_bps: Optional[float]
    extra_bps: Optional[float]
    entry_slippage_bps: Optional[float]
    exit_slippage_bps: Optional[float]
    total_cost_bps: Optional[float]
    total_cost_usd: float
    gross_pnl_usd: Optional[float]
    net_pnl_usd: Optional[float]
    cost_to_gross_pct: Optional[float]
    quote_covered: bool
    mev_suspect: bool
    trade_time: Any
    components: Dict[str, Any]


def _bps(part_usd: float, notional_usd: float) -> Optional[float]:
    if notional_usd is None or notional_usd <= _EPS:
        return None
    return part_usd / notional_usd * BPS


def leg_slippage_bps(quoted: Optional[float], realized: Optional[float],
                     is_buy_leg: bool) -> Optional[float]:
    """Signed slippage cost of one leg in bps. None when unmeasurable."""
    if quoted is None or realized is None:
        return None
    if quoted <= _EPS or realized <= _EPS:
        return None
    raw = (realized - quoted) / quoted * BPS
    return raw if is_buy_leg else -raw


def compute_trade_cost(inp: TradeCostInputs,
                       sandwich_suspect_bps: float = 150.0) -> TradeCostResult:
    """Decompose one trade's execution cost. Pure; never raises on bad data —
    unmeasurable components come back None and are excluded from the sum."""
    notional = float(inp.notional_usd or 0.0)
    fee_usd = max(float(inp.fee_usd or 0.0), 0.0)
    gas_usd = max(float(inp.gas_usd or 0.0), 0.0)
    extra_usd = max(float(inp.extra_cost_usd or 0.0), 0.0)

    entry_is_buy = str(inp.side).lower() not in ("sell", "short")
    entry_slip = leg_slippage_bps(inp.quoted_entry_price, inp.entry_price,
                                  is_buy_leg=entry_is_buy)
    exit_slip = leg_slippage_bps(inp.quoted_exit_price, inp.exit_price,
                                 is_buy_leg=not entry_is_buy)
    quote_covered = entry_slip is not None or exit_slip is not None

    components = dict(inp.components)
    if entry_slip is None and inp.recorded_entry_slippage_bps is not None:
        # Source-recorded slippage figure, but no captured quote: use it as a
        # fallback measurement and say so. Coverage stays False.
        try:
            entry_slip = float(inp.recorded_entry_slippage_bps)
            components["entry_slippage_source"] = "recorded_no_quote"
        except (TypeError, ValueError):
            entry_slip = None

    slip_usd = 0.0
    if notional > _EPS:
        for s in (entry_slip, exit_slip):
            if s is not None:
                slip_usd += max(s, 0.0) / BPS * notional  # only count adverse slip as cost

    total_cost_usd = fee_usd + gas_usd + extra_usd + slip_usd

    gross = None if inp.gross_pnl_usd is None else float(inp.gross_pnl_usd)
    cost_to_gross = None
    if gross is not None and abs(gross) > _EPS:
        cost_to_gross = total_cost_usd / abs(gross) * 100.0

    mev_suspect = (
        entry_slip is not None
        and components.get("entry_slippage_source") != "recorded_no_quote"
        and entry_slip >= float(sandwich_suspect_bps)
    )

    return TradeCostResult(
        module=inp.module,
        trade_ref=inp.trade_ref,
        venue=inp.venue,
        side=inp.side,
        is_simulated=bool(inp.is_simulated),
        notional_usd=notional,
        fee_bps=_bps(fee_usd, notional),
        gas_bps=_bps(gas_usd, notional),
        extra_bps=_bps(extra_usd, notional),
        entry_slippage_bps=entry_slip,
        exit_slippage_bps=exit_slip,
        total_cost_bps=_bps(total_cost_usd, notional),
        total_cost_usd=total_cost_usd,
        gross_pnl_usd=gross,
        net_pnl_usd=None if inp.net_pnl_usd is None else float(inp.net_pnl_usd),
        cost_to_gross_pct=cost_to_gross,
        quote_covered=quote_covered,
        mev_suspect=mev_suspect,
        trade_time=inp.trade_time,
        components=components,
    )


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def aggregate_scorecard(results: List[TradeCostResult],
                        thresholds: Optional[Dict[str, float]] = None,
                        min_trades_for_breaches: int = 3) -> Dict[str, Any]:
    """Fold per-trade results into one module scorecard dict.

    thresholds keys (all optional): fee_warn_bps, gas_warn_bps,
    slippage_warn_bps, total_cost_warn_bps. Breaches are only evaluated when
    len(results) >= min_trades_for_breaches — thin data never raises a flag.
    """
    th = thresholds or {}
    n = len(results)
    fee = [r.fee_bps for r in results if r.fee_bps is not None]
    gas = [r.gas_bps for r in results if r.gas_bps is not None]
    extra = [r.extra_bps for r in results if r.extra_bps is not None]
    eslip = [r.entry_slippage_bps for r in results if r.entry_slippage_bps is not None]
    tcost = [r.total_cost_bps for r in results if r.total_cost_bps is not None]
    gross_known = [r.gross_pnl_usd for r in results if r.gross_pnl_usd is not None]
    net_known = [r.net_pnl_usd for r in results if r.net_pnl_usd is not None]
    total_cost_usd = sum(r.total_cost_usd for r in results)
    gross_sum = sum(gross_known) if gross_known else None

    card: Dict[str, Any] = {
        "trades_scored": n,
        "quote_coverage_pct": (
            100.0 * sum(1 for r in results if r.quote_covered) / n if n else None
        ),
        "avg_fee_bps": _mean(fee),
        "avg_gas_bps": _mean(gas),
        "avg_extra_bps": _mean(extra),
        "avg_entry_slippage_bps": _mean(eslip),
        "median_entry_slippage_bps": _median(eslip),
        "avg_total_cost_bps": _mean(tcost),
        "median_total_cost_bps": _median(tcost),
        "total_cost_usd": total_cost_usd,
        "gross_pnl_usd": gross_sum,
        "net_pnl_usd": sum(net_known) if net_known else None,
        "cost_to_gross_pct": (
            total_cost_usd / abs(gross_sum) * 100.0
            if gross_sum is not None and abs(gross_sum) > _EPS else None
        ),
        "mev_suspect_count": sum(1 for r in results if r.mev_suspect),
        "breaches": [],
    }

    if n >= max(int(min_trades_for_breaches), 1):
        checks = [
            ("avg_fee_bps", "fee_warn_bps"),
            ("avg_gas_bps", "gas_warn_bps"),
            ("avg_entry_slippage_bps", "slippage_warn_bps"),
            ("avg_total_cost_bps", "total_cost_warn_bps"),
        ]
        for metric, key in checks:
            limit = th.get(key)
            val = card.get(metric)
            if limit is not None and val is not None and val > float(limit):
                card["breaches"].append(
                    {"metric": metric, "value": round(val, 2),
                     "threshold": float(limit)}
                )
    return card


# ─────────────────────────────── self-test ───────────────────────────────

def _self_test() -> None:
    approx = lambda a, b, tol=1e-6: a is not None and abs(a - b) <= tol

    # 1. Buy leg slippage: quoted 100, filled 101 -> +100 bps cost.
    assert approx(leg_slippage_bps(100.0, 101.0, True), 100.0)
    # 2. Sell leg: quoted 100, filled 99 -> +100 bps cost (received less).
    assert approx(leg_slippage_bps(100.0, 99.0, False), 100.0)
    # 3. Price improvement on a buy -> negative cost.
    assert leg_slippage_bps(100.0, 99.5, True) < 0
    # 4. Unmeasurable legs -> None, never a guess.
    assert leg_slippage_bps(None, 101.0, True) is None
    assert leg_slippage_bps(0.0, 101.0, True) is None

    # 5. Full decomposition: $1000 notional, $1 fee (10bps), $2 gas (20bps),
    #    quoted 100 / filled 101 buy entry (100bps -> $10 slip cost).
    r = compute_trade_cost(TradeCostInputs(
        module="dex", trade_ref="t1", side="buy", notional_usd=1000.0,
        entry_price=101.0, quoted_entry_price=100.0,
        fee_usd=1.0, gas_usd=2.0, gross_pnl_usd=50.0, net_pnl_usd=37.0,
    ))
    assert approx(r.fee_bps, 10.0) and approx(r.gas_bps, 20.0)
    assert approx(r.entry_slippage_bps, 100.0)
    assert approx(r.total_cost_usd, 13.0)            # 1 + 2 + 10
    assert approx(r.total_cost_bps, 130.0)
    assert approx(r.cost_to_gross_pct, 26.0)         # 13 / 50 * 100
    assert r.quote_covered and not r.mev_suspect

    # 6. Short entry is a sell leg: quoted 100, filled 99 -> +100 bps cost.
    r = compute_trade_cost(TradeCostInputs(
        module="futures", trade_ref="t2", side="short", notional_usd=1000.0,
        entry_price=99.0, quoted_entry_price=100.0,
    ))
    assert approx(r.entry_slippage_bps, 100.0)

    # 7. Favorable slippage is reported but NOT charged as cost.
    r = compute_trade_cost(TradeCostInputs(
        module="dex", trade_ref="t3", side="buy", notional_usd=1000.0,
        entry_price=99.0, quoted_entry_price=100.0, fee_usd=1.0,
    ))
    assert r.entry_slippage_bps < 0 and approx(r.total_cost_usd, 1.0)

    # 8. MEV flag fires at the threshold, only on quote-measured slippage.
    r = compute_trade_cost(TradeCostInputs(
        module="sniper", trade_ref="t4", side="buy", notional_usd=500.0,
        entry_price=103.0, quoted_entry_price=100.0,
    ), sandwich_suspect_bps=150.0)
    assert r.mev_suspect
    r = compute_trade_cost(TradeCostInputs(
        module="dex", trade_ref="t5", side="buy", notional_usd=500.0,
        recorded_entry_slippage_bps=400.0,
    ))
    assert not r.mev_suspect and not r.quote_covered
    assert r.components.get("entry_slippage_source") == "recorded_no_quote"

    # 9. Zero notional -> bps None, usd still summed, nothing raises.
    r = compute_trade_cost(TradeCostInputs(
        module="ai", trade_ref="t6", notional_usd=0.0, fee_usd=3.0,
    ))
    assert r.fee_bps is None and approx(r.total_cost_usd, 3.0)

    # 10. Aggregation + breaches + thin-data guard.
    results = [
        compute_trade_cost(TradeCostInputs(
            module="dex", trade_ref=f"a{i}", side="buy", notional_usd=1000.0,
            entry_price=101.0, quoted_entry_price=100.0,
            fee_usd=5.0, gas_usd=1.0, gross_pnl_usd=20.0,
        )) for i in range(4)
    ]
    card = aggregate_scorecard(
        results,
        thresholds={"fee_warn_bps": 30.0, "slippage_warn_bps": 50.0,
                    "total_cost_warn_bps": 1000.0},
        min_trades_for_breaches=3,
    )
    assert card["trades_scored"] == 4
    assert approx(card["quote_coverage_pct"], 100.0)
    assert approx(card["avg_fee_bps"], 50.0)
    assert approx(card["avg_entry_slippage_bps"], 100.0)
    breached = {b["metric"] for b in card["breaches"]}
    assert breached == {"avg_fee_bps", "avg_entry_slippage_bps"}
    thin = aggregate_scorecard(results[:2],
                               thresholds={"fee_warn_bps": 1.0},
                               min_trades_for_breaches=3)
    assert thin["breaches"] == []

    # 11. Median: odd and even counts.
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert _median([]) is None

    print("cost_model self-test: ALL PASSED")


if __name__ == "__main__":
    _self_test()
