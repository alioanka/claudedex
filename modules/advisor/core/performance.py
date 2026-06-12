"""
Advisor sim performance metrics — PURE helpers, no I/O, no DB, no imports
beyond the stdlib. ADVICE-ONLY bookkeeping: these summarize dry-run sim rows.

Every function takes plain row dicts shaped like advisor_sim_positions rows
(the dashboard API row mapping or asyncpg Records converted with dict(r)):

    {id, symbol, market, channel, direction, horizon,
     entry_price, current_price, exit_price, notional_usd,
     pnl_pct, pnl_usd, status, close_reason, opened_at, closed_at}

Conventions (kept consistent with /api/advisor/simulations):
  - TERMINAL  = status in ('closed', 'expired').
  - DECIDED   = terminal AND pnl_usd IS NOT NULL (sims closed flat with no
                usable price carry NULL pnl and are excluded from win-rate
                denominators so they don't drag WR toward zero).
  - win       = decided and pnl_usd > 0; loss = pnl_usd < 0; flat = == 0.
  - win_rate  = wins / decided (flats count in the denominator).
  - avg_loss_* values are SIGNED (negative); the UI decides presentation.
  - All outputs are None-safe: empty input -> zeroed/None metrics, never a
    raise. The dashboard renders an empty panel instead of a 500.

Self-test (no DB):  python -m modules.advisor.core.performance
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

TERMINAL_STATUSES: Tuple[str, ...] = ("closed", "expired")


# ---------------------------------------------------------------------------
# Small null-safe primitives
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    """Coerce DB numerics/strings to float; None/garbage -> None (0.0 is kept)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # reject NaN


def _ts_iso(v) -> Optional[str]:
    """Normalize a datetime or ISO string to an ISO string (sortable)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def is_terminal(row: dict) -> bool:
    return str(row.get("status") or "").lower() in TERMINAL_STATUSES


def is_decided(row: dict) -> bool:
    """Terminal AND has a usable pnl_usd (see module docstring)."""
    return is_terminal(row) and _num(row.get("pnl_usd")) is not None


def split_rows(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split into (open_rows, terminal_rows). Unknown statuses are dropped."""
    open_rows: List[dict] = []
    terminal_rows: List[dict] = []
    for r in rows or []:
        status = str(r.get("status") or "").lower()
        if status == "open":
            open_rows.append(r)
        elif status in TERMINAL_STATUSES:
            terminal_rows.append(r)
    return open_rows, terminal_rows


def _row_brief(row: dict) -> dict:
    """Compact row summary for best/worst slots."""
    return {
        "id": row.get("id"),
        "symbol": row.get("symbol"),
        "channel": row.get("channel") or row.get("market"),
        "direction": row.get("direction"),
        "horizon": row.get("horizon"),
        "pnl_pct": _num(row.get("pnl_pct")),
        "pnl_usd": _num(row.get("pnl_usd")),
        "close_reason": row.get("close_reason"),
        "closed_at": _ts_iso(row.get("closed_at")),
    }


# ---------------------------------------------------------------------------
# Core aggregate
# ---------------------------------------------------------------------------

def compute_performance(rows: List[dict]) -> dict:
    """
    Aggregate performance over a set of sim rows (any mix of open/terminal).

    Returns a dict with: open_count, terminal_count, decided_count, wins,
    losses, flats, win_rate (percent, None if nothing decided), avg_win_pct,
    avg_loss_pct, avg_win_usd, avg_loss_usd (signed), expectancy_usd,
    expectancy_pct (means over decided sims), profit_factor (gross profit /
    gross loss; None if no losing trades yet), realized_pnl_usd,
    unrealized_pnl_usd, total_pnl_usd, best, worst (row briefs or None).
    """
    open_rows, terminal_rows = split_rows(rows)

    decided = [r for r in terminal_rows if _num(r.get("pnl_usd")) is not None]
    win_rows = [r for r in decided if _num(r.get("pnl_usd")) > 0]
    loss_rows = [r for r in decided if _num(r.get("pnl_usd")) < 0]
    flats = len(decided) - len(win_rows) - len(loss_rows)

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        vs = [v for v in vals if v is not None]
        return (sum(vs) / len(vs)) if vs else None

    avg_win_usd = _mean([_num(r.get("pnl_usd")) for r in win_rows])
    avg_loss_usd = _mean([_num(r.get("pnl_usd")) for r in loss_rows])
    avg_win_pct = _mean([_num(r.get("pnl_pct")) for r in win_rows])
    avg_loss_pct = _mean([_num(r.get("pnl_pct")) for r in loss_rows])

    expectancy_usd = _mean([_num(r.get("pnl_usd")) for r in decided])
    expectancy_pct = _mean(
        [_num(r.get("pnl_pct")) for r in decided if _num(r.get("pnl_pct")) is not None]
    )

    gross_profit = sum(_num(r.get("pnl_usd")) for r in win_rows) if win_rows else 0.0
    gross_loss = -sum(_num(r.get("pnl_usd")) for r in loss_rows) if loss_rows else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    realized = sum(_num(r.get("pnl_usd")) or 0.0 for r in terminal_rows)
    unrealized = sum(_num(r.get("pnl_usd")) or 0.0 for r in open_rows)

    best = max(decided, key=lambda r: _num(r.get("pnl_usd")), default=None)
    worst = min(decided, key=lambda r: _num(r.get("pnl_usd")), default=None)

    win_rate = (
        round(len(win_rows) / len(decided) * 100, 1) if decided else None
    )

    def _r(v, nd=4):
        return round(v, nd) if v is not None else None

    return {
        "open_count": len(open_rows),
        "terminal_count": len(terminal_rows),
        "decided_count": len(decided),
        "wins": len(win_rows),
        "losses": len(loss_rows),
        "flats": flats,
        "win_rate": win_rate,
        "avg_win_pct": _r(avg_win_pct),
        "avg_loss_pct": _r(avg_loss_pct),
        "avg_win_usd": _r(avg_win_usd, 2),
        "avg_loss_usd": _r(avg_loss_usd, 2),
        "expectancy_usd": _r(expectancy_usd, 2),
        "expectancy_pct": _r(expectancy_pct),
        "profit_factor": _r(profit_factor, 2),
        "realized_pnl_usd": round(realized, 2),
        "unrealized_pnl_usd": round(unrealized, 2),
        "total_pnl_usd": round(realized + unrealized, 2),
        "best": _row_brief(best) if best is not None else None,
        "worst": _row_brief(worst) if worst is not None else None,
    }


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

def hit_rate_by_horizon(rows: List[dict]) -> Dict[str, dict]:
    """
    Per-horizon hit rate over DECIDED sims:
      {horizon: {decided, wins, hit_rate (pct or None), expectancy_usd}}.
    Horizons with zero decided sims are absent.
    """
    buckets: Dict[str, List[dict]] = {}
    for r in rows or []:
        if not is_decided(r):
            continue
        h = str(r.get("horizon") or "unknown").lower()
        buckets.setdefault(h, []).append(r)

    out: Dict[str, dict] = {}
    for h, rs in buckets.items():
        wins = sum(1 for r in rs if _num(r.get("pnl_usd")) > 0)
        pnls = [_num(r.get("pnl_usd")) for r in rs]
        out[h] = {
            "decided": len(rs),
            "wins": wins,
            "hit_rate": round(wins / len(rs) * 100, 1) if rs else None,
            "expectancy_usd": round(sum(pnls) / len(pnls), 2) if pnls else None,
        }
    return out


def performance_by_channel(rows: List[dict]) -> Dict[str, dict]:
    """
    Full compute_performance() per channel (fallback: market value when the
    channel field is missing — pre-077 rows). Channels with no rows are absent.
    """
    buckets: Dict[str, List[dict]] = {}
    for r in rows or []:
        ch = str(r.get("channel") or r.get("market") or "unknown").lower()
        buckets.setdefault(ch, []).append(r)
    return {ch: compute_performance(rs) for ch, rs in buckets.items()}


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def equity_curve(rows: List[dict], max_points: int = 500) -> List[dict]:
    """
    Cumulative REALIZED PnL over time from decided sims, ordered by closed_at.

    Returns [{t: iso str, equity: cumulative usd, pnl: trade pnl, symbol}].
    Rows without a closed_at timestamp are skipped (cannot be placed on the
    time axis). Downsampled by stride to <= max_points, always keeping the
    final point so the end-state equity is exact.
    """
    pts: List[Tuple[str, float, Optional[str]]] = []
    for r in rows or []:
        if not is_decided(r):
            continue
        t = _ts_iso(r.get("closed_at"))
        if not t:
            continue
        pts.append((t, _num(r.get("pnl_usd")), r.get("symbol")))

    pts.sort(key=lambda p: p[0])

    curve: List[dict] = []
    equity = 0.0
    for t, pnl, symbol in pts:
        equity += pnl
        curve.append(
            {"t": t, "equity": round(equity, 2), "pnl": round(pnl, 2), "symbol": symbol}
        )

    if max_points and max_points > 1 and len(curve) > max_points:
        stride = -(-len(curve) // max_points)  # ceil division
        sampled = curve[::stride]
        if sampled[-1] is not curve[-1]:
            sampled.append(curve[-1])
        curve = sampled
    return curve


# ---------------------------------------------------------------------------
# Self-test (pure, no DB):  python -m modules.advisor.core.performance
# ---------------------------------------------------------------------------

def _selftest() -> None:
    def row(i, status, pnl_usd, pnl_pct=None, horizon="short", channel="crypto",
            closed_at=None, symbol=None):
        return {
            "id": i, "symbol": symbol or f"SYM{i}", "market": "crypto",
            "channel": channel, "direction": "long", "horizon": horizon,
            "entry_price": 100.0, "notional_usd": 1000.0,
            "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "status": status,
            "close_reason": "target_hit" if status != "open" else None,
            "closed_at": closed_at,
        }

    # 1. Empty input never raises and yields a renderable zero-state.
    p0 = compute_performance([])
    assert p0["decided_count"] == 0 and p0["win_rate"] is None
    assert p0["total_pnl_usd"] == 0.0 and p0["best"] is None
    assert hit_rate_by_horizon([]) == {}
    assert performance_by_channel([]) == {}
    assert equity_curve([]) == []

    # 2. Mixed book: 2 wins, 1 loss, 1 flat, 1 NULL-pnl expired (NOT decided),
    #    1 open with unrealized PnL.
    rows = [
        row(1, "closed", 100.0, 10.0, closed_at="2026-06-01T00:00:00+00:00"),
        row(2, "closed", 50.0, 5.0, horizon="mid",
            closed_at="2026-06-02T00:00:00+00:00"),
        row(3, "closed", -50.0, -5.0, closed_at="2026-06-03T00:00:00+00:00"),
        row(4, "expired", 0.0, 0.0, closed_at="2026-06-04T00:00:00+00:00"),
        row(5, "expired", None, None, closed_at="2026-06-05T00:00:00+00:00"),
        row(6, "open", 25.0, 2.5),
    ]
    p = compute_performance(rows)
    assert p["open_count"] == 1 and p["terminal_count"] == 5
    assert p["decided_count"] == 4, p          # NULL-pnl expired excluded
    assert p["wins"] == 2 and p["losses"] == 1 and p["flats"] == 1
    assert p["win_rate"] == 50.0               # 2/4, flat in denominator
    assert p["avg_win_usd"] == 75.0 and p["avg_loss_usd"] == -50.0
    assert p["avg_win_pct"] == 7.5 and p["avg_loss_pct"] == -5.0
    assert p["expectancy_usd"] == 25.0         # (100+50-50+0)/4
    assert p["profit_factor"] == 3.0           # 150 gross win / 50 gross loss
    assert p["realized_pnl_usd"] == 100.0      # NULL pnl counts as 0
    assert p["unrealized_pnl_usd"] == 25.0
    assert p["total_pnl_usd"] == 125.0
    assert p["best"]["id"] == 1 and p["worst"]["id"] == 3

    # 3. Hit-rate by horizon: short has 3 decided (w/l/flat), mid 1 win.
    hr = hit_rate_by_horizon(rows)
    assert hr["short"]["decided"] == 3 and hr["short"]["wins"] == 1
    assert hr["short"]["hit_rate"] == 33.3
    assert hr["mid"] == {"decided": 1, "wins": 1, "hit_rate": 100.0,
                         "expectancy_usd": 50.0}

    # 4. Per-channel breakdown + market fallback when channel missing.
    rows2 = rows + [row(7, "closed", 10.0, 1.0, channel=None,
                        closed_at="2026-06-06T00:00:00+00:00")]
    by_ch = performance_by_channel(rows2)
    assert by_ch["crypto"]["decided_count"] == 5   # fallback row joins crypto
    assert set(by_ch) == {"crypto"}

    # 5. Equity curve: ordered, cumulative, NULL-pnl/opens skipped.
    curve = equity_curve(rows)
    assert [c["equity"] for c in curve] == [100.0, 150.0, 100.0, 100.0]
    assert curve[0]["t"] < curve[-1]["t"]
    # datetime closed_at also accepted.
    rows_dt = [row(8, "closed", 5.0, 0.5,
                   closed_at=datetime(2026, 6, 7, tzinfo=None))]
    assert equity_curve(rows_dt)[0]["equity"] == 5.0

    # 6. Downsampling keeps the exact final equity.
    big = [row(100 + i, "closed", 1.0, 0.1,
               closed_at=f"2026-05-{(i % 28) + 1:02d}T{i % 24:02d}:00:00+00:00")
           for i in range(1000)]
    ds = equity_curve(big, max_points=50)
    assert len(ds) <= 51 and ds[-1]["equity"] == 1000.0

    # 7. SHORT-sim sign is whatever pnl_usd says (helpers never re-derive
    #    direction math); a short with positive pnl_usd is a win.
    srow = row(9, "closed", 20.0, 2.0, closed_at="2026-06-08T00:00:00+00:00")
    srow["direction"] = "short"
    assert compute_performance([srow])["wins"] == 1

    print("performance self-test OK: aggregates, horizon hit-rate, "
          "channel breakdown, equity curve")


if __name__ == "__main__":
    _selftest()
