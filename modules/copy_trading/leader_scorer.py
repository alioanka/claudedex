"""
Copy-trading leader scorer.

Pure-function module: takes a list of closed-trade rows for a leader
wallet and returns a normalised score (0..100) plus the per-component
metrics that drove it. Persistence is handled by wallet_discovery /
copy_engine via the copy_leader_scores table (migration 023).

Design notes
------------
* No look-ahead. The scorer only consumes trades whose `exit_timestamp`
  is strictly less than `as_of` (passed in, defaults to "now"). The
  walk-forward window is a parameter, default 30 days.
* No survivorship bias on the scorer side. The discovery layer is
  responsible for the leader-universe definition. The scorer trusts
  the leader list passed in.
* Sharpe is computed from daily-bucketed realized USD PnL, annualised
  with sqrt(365) — crypto trades 24/7. The Phase-1 quant audit
  (COPY_TRADING_quant.md) calls out this exact requirement.
* Score is a clipped linear blend of five normalised z-equivalents.
  Weights are exposed in DEFAULT_WEIGHTS so they can be DB-tuned later
  via config_settings.copytrading_config.leader_score_weights.
* Composite score is **bounded** to [0, 100] and gracefully degrades
  on missing inputs (a leader with only 3 trades returns a low score
  rather than a NaN).
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence


# --- public weights (DB-overridable) -----------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "pnl": 0.35,        # realized 30d PnL (z-scaled vs. cohort)
    "sharpe": 0.25,     # annualised Sharpe of daily PnL
    "hit_rate": 0.15,   # closed-trade win ratio (sample-size adjusted)
    "hold": 0.10,       # avg hold-time fit (penalise scalps + bag-holders)
    "drawdown": 0.15,   # 1 - max_drawdown (so smaller DD scores higher)
}

# Walk-forward window. The scorer trusts only trades whose exit was
# < as_of - SAMPLE_WINDOW_DAYS days ago. 30d is the operator default;
# 90d is computed in parallel for trend confirmation.
DEFAULT_SAMPLE_WINDOW_DAYS = 30

# Minimum closed trades for full credit on hit_rate. Below this we
# Bayesian-shrink toward the cohort prior of 0.5 so a 1-for-1 wallet
# doesn't dominate a 200-for-300 wallet.
HIT_RATE_PRIOR_N = 20
HIT_RATE_PRIOR_P = 0.5

# Ideal hold seconds — bell curve maxed at ~6h (memecoin sweet spot
# in the operator's universe). Penalty curve uses log-distance.
IDEAL_HOLD_SECONDS = 6 * 3600

# Kelly cap. Quarter-Kelly is the operator-default in
# core/decision_maker.py:712; we mirror that here.
KELLY_CAP = 0.25


@dataclass
class LeaderMetrics:
    """Computed metrics for one leader wallet over the score window."""
    chain: str
    wallet_address: str
    sample_window_days: int = DEFAULT_SAMPLE_WINDOW_DAYS
    realized_pnl_usd_30d: float = 0.0
    realized_pnl_usd_90d: float = 0.0
    sharpe_30d: float = 0.0
    hit_rate: float = 0.0           # 0..1, raw closed-trade ratio
    hit_rate_shrunk: float = 0.5    # Bayesian-shrunk toward prior
    avg_hold_seconds: float = 0.0
    max_drawdown_pct: float = 0.0   # 0..1
    trade_count_30d: int = 0
    avg_trade_size_usd: float = 0.0
    # Composite score 0..100 — computed by compute_score.
    score: Optional[float] = None
    kelly_fraction: Optional[float] = None
    components: Dict[str, float] = field(default_factory=dict)

    def to_db_row(self) -> Dict[str, object]:
        """Coerce to types matching copy_leader_scores columns."""
        d = asdict(self)
        # Coerce floats to None where 0 means "no data" so the dashboard
        # can show "n/a" instead of misleading zeros.
        if self.trade_count_30d == 0:
            for k in (
                "realized_pnl_usd_30d", "sharpe_30d", "hit_rate",
                "avg_hold_seconds", "max_drawdown_pct", "avg_trade_size_usd",
            ):
                d[k] = None
        return d


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value) -> Optional[datetime]:
    """Defensive timestamp parser — DB layer returns datetimes, but
    third-party JSON often returns ISO strings or epoch seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _filter_window(
    trades: Sequence[Dict],
    as_of: datetime,
    window_days: int,
) -> List[Dict]:
    """Keep only trades whose exit_timestamp falls in (as_of - window, as_of].

    Trades without exit (still open) are excluded — Sharpe / DD need
    realized outcomes.
    """
    cutoff = as_of - timedelta(days=window_days)
    out: List[Dict] = []
    for t in trades:
        exit_ts = _parse_ts(t.get("exit_timestamp"))
        if not exit_ts:
            continue
        if cutoff < exit_ts <= as_of:
            out.append(t)
    return out


def _daily_pnl_series(trades: Sequence[Dict]) -> List[float]:
    """Bucket realized USD PnL by exit-date (UTC). Returns a list ordered
    by date — used for Sharpe and drawdown."""
    buckets: Dict[str, float] = defaultdict(float)
    for t in trades:
        exit_ts = _parse_ts(t.get("exit_timestamp"))
        if not exit_ts:
            continue
        bucket = exit_ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
        try:
            pnl = float(t.get("profit_loss") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        buckets[bucket] += pnl
    return [buckets[k] for k in sorted(buckets.keys())]


def _annualised_sharpe(daily_pnl: Sequence[float]) -> float:
    """Sharpe of daily PnL series, annualised sqrt(365). Returns 0 when
    fewer than 3 samples or zero-variance.

    Risk-free rate is approximated to 0 — for memecoin time-horizons
    the funding rate / treasury yield is a rounding error.
    """
    if len(daily_pnl) < 3:
        return 0.0
    mean = statistics.fmean(daily_pnl)
    try:
        stdev = statistics.pstdev(daily_pnl)
    except statistics.StatisticsError:
        return 0.0
    if stdev <= 0:
        return 0.0
    return (mean / stdev) * math.sqrt(365)


def _max_drawdown(daily_pnl: Sequence[float]) -> float:
    """Peak-to-trough drawdown of the cumulative-PnL curve, expressed
    as a fraction of running peak. Returns 0 when the curve never
    drops below its starting point.

    Note: pnl curve can start at any value, so we normalise against
    the running max-equity, not absolute equity. A leader that goes
    +1k -> -500 has a 1.5k drawdown from peak (DD = 1.5 / 1.0 = 1.5
    which we clip to 1.0).
    """
    if not daily_pnl:
        return 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in daily_pnl:
        cum += pnl
        if cum > peak:
            peak = cum
        if peak <= 0:
            # Still underwater from the start; DD is open-ended.
            # Use abs(cum) over a nominal $1k risk unit so this still
            # contributes to the DD penalty without dividing by zero.
            dd = min(1.0, abs(cum) / 1000.0)
        else:
            dd = (peak - cum) / peak
        if dd > max_dd:
            max_dd = dd
    return min(1.0, max(0.0, max_dd))


def _avg_hold_seconds(trades: Sequence[Dict]) -> float:
    holds: List[float] = []
    for t in trades:
        entry = _parse_ts(t.get("entry_timestamp"))
        exit_ts = _parse_ts(t.get("exit_timestamp"))
        if not entry or not exit_ts:
            continue
        delta = (exit_ts - entry).total_seconds()
        if delta > 0:
            holds.append(delta)
    if not holds:
        return 0.0
    return statistics.fmean(holds)


def _hit_rate(trades: Sequence[Dict]) -> tuple[float, float]:
    """Return (raw, Bayesian-shrunk) hit rate.

    Shrinkage prevents a 1-for-1 leader from scoring above a
    100-for-200 leader. Posterior with Beta(α=10, β=10) prior:
        p_shrunk = (wins + α) / (n + α + β)
    """
    if not trades:
        return (0.0, HIT_RATE_PRIOR_P)
    wins = 0
    for t in trades:
        try:
            pnl = float(t.get("profit_loss") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl > 0:
            wins += 1
    raw = wins / len(trades) if trades else 0.0
    alpha = HIT_RATE_PRIOR_N * HIT_RATE_PRIOR_P
    beta = HIT_RATE_PRIOR_N * (1 - HIT_RATE_PRIOR_P)
    shrunk = (wins + alpha) / (len(trades) + alpha + beta)
    return (raw, shrunk)


def _hold_score(avg_hold_seconds: float) -> float:
    """Bell-curve hold-time score. 1.0 at IDEAL_HOLD_SECONDS, decaying
    log-symmetrically. A 2-second scalp scores ~0.05; a 7-day bag-hold
    scores ~0.10. Zero hold → 0."""
    if avg_hold_seconds <= 0:
        return 0.0
    # log-distance from ideal in dex (log10) units.
    dex = abs(math.log10(avg_hold_seconds) - math.log10(IDEAL_HOLD_SECONDS))
    # 1 dex = 10x off ideal → score 0.5. 2 dex → 0.25. Asymptote 0.
    return float(max(0.0, min(1.0, 1.0 / (1.0 + dex * dex))))


def compute_metrics(
    chain: str,
    wallet_address: str,
    trades: Sequence[Dict],
    *,
    as_of: Optional[datetime] = None,
    window_days: int = DEFAULT_SAMPLE_WINDOW_DAYS,
) -> LeaderMetrics:
    """Pure: take closed-trade rows, return LeaderMetrics (no score yet).

    Each `trades` row must expose at minimum:
        entry_timestamp, exit_timestamp, profit_loss (USD), entry_usd

    All keys are optional; missing values gracefully degrade the
    output rather than raise.
    """
    as_of = as_of or _now_utc()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    window = _filter_window(trades, as_of, window_days)
    window_90 = _filter_window(trades, as_of, 90)

    daily = _daily_pnl_series(window)
    pnl_30 = sum(float(t.get("profit_loss") or 0.0) for t in window)
    pnl_90 = sum(float(t.get("profit_loss") or 0.0) for t in window_90)
    sharpe = _annualised_sharpe(daily)
    raw_hit, shrunk_hit = _hit_rate(window)
    hold = _avg_hold_seconds(window)
    dd = _max_drawdown(daily)
    sizes = [
        float(t.get("entry_usd") or 0.0)
        for t in window
        if (t.get("entry_usd") or 0)
    ]
    avg_size = statistics.fmean(sizes) if sizes else 0.0

    return LeaderMetrics(
        chain=chain,
        wallet_address=wallet_address,
        sample_window_days=window_days,
        realized_pnl_usd_30d=float(pnl_30),
        realized_pnl_usd_90d=float(pnl_90),
        sharpe_30d=float(sharpe),
        hit_rate=float(raw_hit),
        hit_rate_shrunk=float(shrunk_hit),
        avg_hold_seconds=float(hold),
        max_drawdown_pct=float(dd),
        trade_count_30d=int(len(window)),
        avg_trade_size_usd=float(avg_size),
    )


def _zscale_component(value: float, *, mid: float, span: float) -> float:
    """Map an unbounded input to [0, 1] with sigmoid-ish curve.
        score = 1 / (1 + exp(-(value - mid) / span))
    Picks span so 1 span above mid = ~0.73, 2 spans = ~0.88, etc.
    """
    try:
        return float(1.0 / (1.0 + math.exp(-(value - mid) / span)))
    except OverflowError:
        return 0.0 if value < mid else 1.0


def compute_score(
    metrics: LeaderMetrics,
    weights: Optional[Dict[str, float]] = None,
) -> LeaderMetrics:
    """Populate `metrics.score` and `metrics.kelly_fraction` in-place
    and return the same object.

    Components (each in [0, 1] before weighting):
      pnl       — sigmoid of 30d PnL USD, mid $500, span $750
      sharpe    — sigmoid of Sharpe, mid 1.0, span 1.0
      hit_rate  — Bayesian-shrunk closed-trade ratio
      hold      — bell-curve fit around IDEAL_HOLD_SECONDS
      drawdown  — (1 - max_dd) so smaller DD = higher score

    Weights default to DEFAULT_WEIGHTS but operator can override via
    config_settings.copytrading_config.leader_score_weights.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w and isinstance(v, (int, float)) and v >= 0:
                w[k] = float(v)
    # Normalise weights so they always sum to 1 (operator might submit 35/25/15
    # in any units; we don't enforce a sum-constraint).
    total_w = sum(w.values()) or 1.0
    for k in list(w.keys()):
        w[k] /= total_w

    # Penalise low-sample leaders: shrink composite score toward 0 if
    # trade_count_30d < HIT_RATE_PRIOR_N. This avoids a 1-trade wallet
    # outscoring a 100-trade one merely because its single trade won.
    sample_credit = min(1.0, metrics.trade_count_30d / float(HIT_RATE_PRIOR_N))

    components = {
        "pnl": _zscale_component(metrics.realized_pnl_usd_30d, mid=500.0, span=750.0),
        "sharpe": _zscale_component(metrics.sharpe_30d, mid=1.0, span=1.0),
        "hit_rate": float(metrics.hit_rate_shrunk),
        "hold": _hold_score(metrics.avg_hold_seconds),
        "drawdown": max(0.0, 1.0 - float(metrics.max_drawdown_pct)),
    }

    composite = sum(components[k] * w[k] for k in components)
    composite *= sample_credit
    score = round(float(max(0.0, min(1.0, composite))) * 100.0, 2)

    # Kelly fraction from hit_rate + payoff ratio (avg-win / avg-loss).
    # When data is sparse, default kelly = 0 so engine refuses to size.
    metrics.score = score
    metrics.components = components
    metrics.kelly_fraction = _kelly_from_metrics(metrics)
    return metrics


def _kelly_from_metrics(m: LeaderMetrics) -> float:
    """Capped fractional-Kelly from the shrunk hit-rate and an assumed
    payoff ratio of 1.5 (a common memecoin take-profit / stop ratio).

        f* = (p * b - q) / b
        f  = clamp(f* * KELLY_CAP, 0, KELLY_CAP)

    Uses the Bayesian-shrunk hit rate so 1-trade wallets get 0, not 1.
    Returns 0 if hit_rate_shrunk is at or below break-even (1/(1+b)).
    """
    if m.trade_count_30d < 5:
        return 0.0
    p = float(m.hit_rate_shrunk)
    q = 1.0 - p
    b = 1.5  # avg-win / avg-loss assumption
    breakeven = 1.0 / (1.0 + b)
    if p <= breakeven:
        return 0.0
    f_star = (p * b - q) / b
    f = max(0.0, min(KELLY_CAP, f_star * KELLY_CAP))
    return round(float(f), 4)


def score_leader(
    chain: str,
    wallet_address: str,
    trades: Sequence[Dict],
    *,
    as_of: Optional[datetime] = None,
    window_days: int = DEFAULT_SAMPLE_WINDOW_DAYS,
    weights: Optional[Dict[str, float]] = None,
) -> LeaderMetrics:
    """Convenience: compute_metrics + compute_score in one call."""
    m = compute_metrics(
        chain, wallet_address, trades,
        as_of=as_of, window_days=window_days,
    )
    return compute_score(m, weights=weights)


def rank_leaders(
    leaders: Iterable[LeaderMetrics],
    *,
    min_trades: int = 5,
    top_n: Optional[int] = None,
) -> List[LeaderMetrics]:
    """Filter then sort by composite score descending. Leaders with
    score == None or trade_count_30d < min_trades are dropped.

    NB: rank_leaders does not mutate; it returns a new list.
    """
    filtered = [
        m for m in leaders
        if m.score is not None and m.trade_count_30d >= min_trades
    ]
    filtered.sort(key=lambda m: (m.score or 0.0), reverse=True)
    return filtered[: top_n] if top_n else filtered


# ---------------------------------------------------------------------
# Persistence helpers (thin wrappers; copy_engine.py / wallet_discovery
# call these so the SQL stays in one place)
# ---------------------------------------------------------------------

UPSERT_SQL = """
    INSERT INTO copy_leader_scores (
        chain, wallet_address, label, source,
        realized_pnl_usd_30d, realized_pnl_usd_90d, sharpe_30d,
        hit_rate, avg_hold_seconds, max_drawdown_pct, trade_count_30d,
        avg_trade_size_usd, score, kelly_fraction,
        sample_window_days, last_scored_at, raw_metrics
    ) VALUES (
        $1, $2, $3, $4,
        $5, $6, $7,
        $8, $9, $10, $11,
        $12, $13, $14,
        $15, NOW(), $16
    )
    ON CONFLICT (chain, wallet_address) DO UPDATE SET
        label = COALESCE(EXCLUDED.label, copy_leader_scores.label),
        source = EXCLUDED.source,
        realized_pnl_usd_30d = EXCLUDED.realized_pnl_usd_30d,
        realized_pnl_usd_90d = EXCLUDED.realized_pnl_usd_90d,
        sharpe_30d = EXCLUDED.sharpe_30d,
        hit_rate = EXCLUDED.hit_rate,
        avg_hold_seconds = EXCLUDED.avg_hold_seconds,
        max_drawdown_pct = EXCLUDED.max_drawdown_pct,
        trade_count_30d = EXCLUDED.trade_count_30d,
        avg_trade_size_usd = EXCLUDED.avg_trade_size_usd,
        score = EXCLUDED.score,
        kelly_fraction = EXCLUDED.kelly_fraction,
        sample_window_days = EXCLUDED.sample_window_days,
        last_scored_at = NOW(),
        raw_metrics = EXCLUDED.raw_metrics
"""


async def upsert_score(
    db_pool,
    m: LeaderMetrics,
    *,
    label: Optional[str] = None,
    source: str = "manual",
    raw_metrics: Optional[Dict] = None,
) -> None:
    """Persist a LeaderMetrics row to copy_leader_scores."""
    if db_pool is None:
        return
    import json as _json
    async with db_pool.acquire() as conn:
        await conn.execute(
            UPSERT_SQL,
            m.chain, m.wallet_address, label, source,
            m.realized_pnl_usd_30d if m.trade_count_30d else None,
            m.realized_pnl_usd_90d if m.trade_count_30d else None,
            m.sharpe_30d if m.trade_count_30d else None,
            m.hit_rate if m.trade_count_30d else None,
            int(m.avg_hold_seconds) if m.avg_hold_seconds else None,
            m.max_drawdown_pct if m.trade_count_30d else None,
            int(m.trade_count_30d),
            m.avg_trade_size_usd if m.trade_count_30d else None,
            m.score,
            m.kelly_fraction,
            int(m.sample_window_days),
            _json.dumps(raw_metrics or {}),
        )


async def fetch_top_leaders(
    db_pool,
    *,
    chain: Optional[str] = None,
    limit: int = 25,
    min_score: float = 0.0,
) -> List[Dict]:
    """Return top-N rows from copy_leader_scores ordered by score DESC."""
    if db_pool is None:
        return []
    if chain:
        sql = (
            "SELECT * FROM copy_leader_scores "
            "WHERE chain = $1 AND COALESCE(score, 0) >= $2 "
            "ORDER BY score DESC NULLS LAST LIMIT $3"
        )
        args = (chain, min_score, limit)
    else:
        sql = (
            "SELECT * FROM copy_leader_scores "
            "WHERE COALESCE(score, 0) >= $1 "
            "ORDER BY score DESC NULLS LAST LIMIT $2"
        )
        args = (min_score, limit)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) for r in rows]
