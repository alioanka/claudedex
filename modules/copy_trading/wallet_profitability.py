"""
COPY v3 wallet-profitability scorer — PURE math, no I/O.

Takes a wallet's raw swap events (buys/sells with price + time), FIFO-matches
them into REALIZED round-trips, and scores the wallet 0..100 on trailing
realized performance only.

No-look-ahead guarantee
-----------------------
* Every event with ``ts > as_of`` is dropped before any computation
  (``_filter_events`` raises in strict mode if one is passed — the self-test
  asserts this).
* All metrics are computed from round-trips whose SELL leg (realization
  time) falls inside ``(as_of - window_days, as_of]``. Nothing is ever
  scored on un-elapsed forward returns.
* Open lots (buys without a matching sell) contribute NOTHING to the score
  — unrealized PnL is reported separately by the shadow simulator, never
  blended into this ranking.

Anti-gaming penalties
---------------------
* ``wash_penalty`` — share of round-trips that look like wash/self trading:
  hold < ``WASH_HOLD_SECONDS`` AND |pnl| < ``WASH_PNL_EPS_PCT`` of notional.
* ``lucky_penalty`` — single-trade PnL concentration: when the best trade
  contributes more than ``LUCKY_SHARE_THRESHOLD`` of total positive PnL the
  composite is shrunk proportionally (a 1-moonshot wallet is luck until
  proven otherwise).
* ``sample_credit`` — composite shrinks toward 0 below MIN_FULL_CREDIT_TRADES
  closed round-trips (mirrors leader_scorer's Bayesian posture).

Event shape (dict; missing keys degrade gracefully, never raise):
    {token: str, side: 'buy'|'sell', qty: float, price_usd: float, ts: datetime}
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

# --- tunables (operator overrides flow through discovery config) --------
DEFAULT_WINDOW_DAYS = 30
MIN_FULL_CREDIT_TRADES = 20      # round-trips for full sample credit
HIT_RATE_PRIOR_N = 20            # Beta prior strength (mirrors leader_scorer)
HIT_RATE_PRIOR_P = 0.5
PROFIT_FACTOR_CAP = 10.0
IDEAL_HOLD_SECONDS = 6 * 3600    # memecoin sweet spot (leader_scorer parity)
WASH_HOLD_SECONDS = 60.0         # round-trip faster than this is suspect
WASH_PNL_EPS_PCT = 0.5           # ...when |pnl| < 0.5% of notional
LUCKY_SHARE_THRESHOLD = 0.6      # best trade > 60% of gross wins = lucky
DIVERSIFICATION_FULL_TOKENS = 5  # distinct tokens for full diversification
CROWDING_FOLLOWERS_SOFTCAP = 5   # followers-per-entry before alpha decays


def crowding_penalty_from_followers(avg_followers: float) -> float:
    """Soft-cap crowding penalty 0..0.5 (AI-Trader port, INVERTED).

    ``avg_followers`` = mean distinct wallets that bought the same token shortly
    AFTER the leader's entry. On-chain, crowding == alpha decay: a publicly
    rankable wallet attracts copiers who front-run and fade the edge (our own
    CLAUDE.md honesty note). No penalty up to the soft cap, then a bounded shave
    — mirrors the smart_money soft-cap pattern rather than a hard cliff."""
    try:
        avg_followers = float(avg_followers)
    except (TypeError, ValueError):
        return 0.0
    if avg_followers <= CROWDING_FOLLOWERS_SOFTCAP:
        return 0.0
    excess = avg_followers - CROWDING_FOLLOWERS_SOFTCAP
    return min(0.5, excess / (excess + CROWDING_FOLLOWERS_SOFTCAP))

DEFAULT_WEIGHTS: Dict[str, float] = {
    "pnl": 0.30,            # realized PnL USD (sigmoid)
    "win_rate": 0.15,       # Bayesian-shrunk
    "profit_factor": 0.15,  # gross wins / gross losses (sigmoid)
    "consistency": 0.15,    # daily realized-PnL Sharpe (sigmoid)
    "drawdown": 0.10,       # 1 - max DD of cumulative realized PnL
    "hold": 0.05,           # bell curve around IDEAL_HOLD_SECONDS
    "recency": 0.05,        # active recently, not a dormant legend
    "diversification": 0.05,
}


@dataclass
class RoundTrip:
    """One FIFO-matched realized trade."""
    token: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: datetime
    exit_ts: datetime

    @property
    def pnl_usd(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def notional_usd(self) -> float:
        return self.entry_price * self.qty

    @property
    def hold_seconds(self) -> float:
        return max(0.0, (self.exit_ts - self.entry_ts).total_seconds())


@dataclass
class WalletScore:
    chain: str
    wallet_address: str
    window_days: int = DEFAULT_WINDOW_DAYS
    realized_pnl_usd: float = 0.0
    win_rate: float = 0.0
    win_rate_shrunk: float = HIT_RATE_PRIOR_P
    profit_factor: float = 0.0
    trade_count: int = 0
    max_drawdown_pct: float = 0.0
    avg_hold_seconds: float = 0.0
    consistency: float = 0.0
    diversification: float = 0.0
    wash_penalty: float = 0.0
    lucky_penalty: float = 0.0
    crowding_penalty: float = 0.0
    recency: float = 0.0
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _filter_events(events: Sequence[Dict], as_of: datetime, *, strict: bool = False) -> List[Dict]:
    """Drop malformed events and ANY event after as_of (no look-ahead).

    strict=True raises AssertionError on a future event instead of dropping
    — used by the self-test to prove the guard exists.
    """
    out: List[Dict] = []
    for e in events:
        ts = _utc(e.get("ts"))
        side = e.get("side")
        try:
            qty = float(e.get("qty") or 0.0)
            price = float(e.get("price_usd") or 0.0)
        except (TypeError, ValueError):
            continue
        if ts is None or side not in ("buy", "sell") or qty <= 0 or price <= 0:
            continue
        if ts > as_of:
            if strict:
                raise AssertionError(
                    f"look-ahead event at {ts.isoformat()} > as_of {as_of.isoformat()}"
                )
            continue
        out.append({**e, "ts": ts, "qty": qty, "price_usd": price})
    out.sort(key=lambda e: e["ts"])
    return out


def fifo_round_trips(events: Sequence[Dict], as_of: datetime, *, strict: bool = False) -> List[RoundTrip]:
    """FIFO-match buys against sells per token → realized round-trips.

    Sells with no matching open lot (history starts mid-position) are
    skipped — we refuse to fabricate a cost basis.
    """
    evs = _filter_events(events, as_of, strict=strict)
    lots: Dict[str, List[Tuple[float, float, datetime]]] = defaultdict(list)  # token -> [(qty, price, ts)]
    trips: List[RoundTrip] = []
    for e in evs:
        token = str(e.get("token") or "?")
        if e["side"] == "buy":
            lots[token].append((e["qty"], e["price_usd"], e["ts"]))
            continue
        # sell: consume FIFO lots
        remaining = e["qty"]
        open_lots = lots[token]
        while remaining > 1e-12 and open_lots:
            lot_qty, lot_price, lot_ts = open_lots[0]
            take = min(remaining, lot_qty)
            trips.append(RoundTrip(
                token=token, qty=take,
                entry_price=lot_price, exit_price=e["price_usd"],
                entry_ts=lot_ts, exit_ts=e["ts"],
            ))
            remaining -= take
            if take >= lot_qty - 1e-12:
                open_lots.pop(0)
            else:
                open_lots[0] = (lot_qty - take, lot_price, lot_ts)
        # leftover sell qty with no lot: ignored (unknown basis).
    return trips


def _window_trips(trips: Sequence[RoundTrip], as_of: datetime, window_days: int) -> List[RoundTrip]:
    cutoff = as_of - timedelta(days=window_days)
    return [t for t in trips if cutoff < t.exit_ts <= as_of]


def _daily_pnl(trips: Sequence[RoundTrip]) -> List[float]:
    buckets: Dict[str, float] = defaultdict(float)
    for t in trips:
        buckets[t.exit_ts.astimezone(timezone.utc).strftime("%Y-%m-%d")] += t.pnl_usd
    return [buckets[k] for k in sorted(buckets)]


def _max_drawdown(daily: Sequence[float]) -> float:
    cum = peak = max_dd = 0.0
    for p in daily:
        cum += p
        peak = max(peak, cum)
        dd = min(1.0, abs(cum) / 1000.0) if peak <= 0 else (peak - cum) / peak
        max_dd = max(max_dd, dd)
    return min(1.0, max(0.0, max_dd))


def _sigmoid(value: float, mid: float, span: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-(value - mid) / span))
    except OverflowError:
        return 0.0 if value < mid else 1.0


def _hold_score(avg_hold: float) -> float:
    if avg_hold <= 0:
        return 0.0
    dex = abs(math.log10(avg_hold) - math.log10(IDEAL_HOLD_SECONDS))
    return max(0.0, min(1.0, 1.0 / (1.0 + dex * dex)))


def score_wallet(
    chain: str,
    wallet_address: str,
    events: Sequence[Dict],
    *,
    as_of: Optional[datetime] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    weights: Optional[Dict[str, float]] = None,
    strict: bool = False,
    crowding: float = 0.0,
) -> WalletScore:
    """Full pipeline: events → FIFO round-trips → trailing-window metrics →
    bounded 0..100 composite with wash/lucky/crowding penalties.

    ``crowding`` (0..1) is an OPTIONAL, caller-supplied soft-cap penalty
    (see ``crowding_penalty_from_followers``); default 0 keeps the scorer pure
    and backward-compatible. The discovery layer populates it from smart_money
    follower density; it can only SHRINK the composite, never inflate it."""
    as_of = _utc(as_of) or datetime.now(timezone.utc)
    trips = _window_trips(fifo_round_trips(events, as_of, strict=strict), as_of, window_days)
    s = WalletScore(chain=chain, wallet_address=wallet_address, window_days=window_days)
    s.trade_count = len(trips)
    if not trips:
        return s

    pnls = [t.pnl_usd for t in trips]
    s.realized_pnl_usd = float(sum(pnls))
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    s.win_rate = len(wins) / len(trips)
    alpha = HIT_RATE_PRIOR_N * HIT_RATE_PRIOR_P
    beta = HIT_RATE_PRIOR_N * (1 - HIT_RATE_PRIOR_P)
    s.win_rate_shrunk = (len(wins) + alpha) / (len(trips) + alpha + beta)
    gross_win = sum(wins)
    gross_loss = sum(losses)
    s.profit_factor = (
        min(PROFIT_FACTOR_CAP, gross_win / gross_loss) if gross_loss > 0
        else (PROFIT_FACTOR_CAP if gross_win > 0 else 0.0)
    )

    daily = _daily_pnl(trips)
    s.max_drawdown_pct = _max_drawdown(daily)
    if len(daily) >= 3:
        mean = statistics.fmean(daily)
        stdev = statistics.pstdev(daily)
        s.consistency = _sigmoid((mean / stdev) * math.sqrt(365), 1.0, 1.0) if stdev > 0 else 0.0
    s.avg_hold_seconds = statistics.fmean([t.hold_seconds for t in trips])

    # Recency: share of round-trips realized in the most recent 25% of the
    # window. 0 = all activity is old; sigmoid mid at the uniform rate.
    recent_cut = as_of - timedelta(days=window_days * 0.25)
    recent_share = sum(1 for t in trips if t.exit_ts > recent_cut) / len(trips)
    s.recency = _sigmoid(recent_share, 0.25, 0.10)

    # Diversification: token-notional HHI inverted, scaled by token count.
    notional_by_token: Dict[str, float] = defaultdict(float)
    for t in trips:
        notional_by_token[t.token] += t.notional_usd
    total_notional = sum(notional_by_token.values()) or 1.0
    hhi = sum((v / total_notional) ** 2 for v in notional_by_token.values())
    s.diversification = (1.0 - hhi) * min(1.0, len(notional_by_token) / DIVERSIFICATION_FULL_TOKENS)

    # Wash penalty: fast near-zero-PnL round-trip share.
    washy = sum(
        1 for t in trips
        if t.hold_seconds < WASH_HOLD_SECONDS
        and abs(t.pnl_usd) < (WASH_PNL_EPS_PCT / 100.0) * max(1e-9, t.notional_usd)
    )
    s.wash_penalty = washy / len(trips)

    # Lucky penalty: best single trade's share of gross wins above threshold.
    if gross_win > 0 and wins:
        top_share = max(wins) / gross_win
        if top_share > LUCKY_SHARE_THRESHOLD:
            s.lucky_penalty = min(
                1.0, (top_share - LUCKY_SHARE_THRESHOLD) / (1.0 - LUCKY_SHARE_THRESHOLD)
            )

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w and isinstance(v, (int, float)) and v >= 0:
                w[k] = float(v)
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    components = {
        "pnl": _sigmoid(s.realized_pnl_usd, 500.0, 750.0),
        "win_rate": s.win_rate_shrunk,
        "profit_factor": _sigmoid(s.profit_factor, 1.5, 1.0),
        "consistency": s.consistency,
        "drawdown": max(0.0, 1.0 - s.max_drawdown_pct),
        "hold": _hold_score(s.avg_hold_seconds),
        "recency": s.recency,
        "diversification": s.diversification,
    }
    s.crowding_penalty = max(0.0, min(1.0, float(crowding or 0.0)))

    composite = sum(components[k] * w[k] for k in components)
    sample_credit = min(1.0, len(trips) / float(MIN_FULL_CREDIT_TRADES))
    composite *= (sample_credit * (1.0 - s.wash_penalty)
                  * (1.0 - s.lucky_penalty) * (1.0 - s.crowding_penalty))
    s.components = components
    s.score = round(max(0.0, min(1.0, composite)) * 100.0, 2)
    return s


def trades_to_events(rows: Sequence[Dict]) -> List[Dict]:
    """Adapter: copytrading_trades-style CLOSED rows (entry/exit timestamp +
    profit_loss + entry_usd) → synthetic buy/sell event pairs so mirrored
    history feeds the same FIFO scorer. Price levels are synthetic (basis 1.0,
    exit 1.0 + pnl/notional) — PnL, hold and timing are exact; per-token
    diversification uses the row's token when present."""
    events: List[Dict] = []
    for i, r in enumerate(rows):
        entry_ts = _utc(r.get("entry_timestamp"))
        exit_ts = _utc(r.get("exit_timestamp"))
        if not entry_ts or not exit_ts:
            continue
        try:
            notional = float(r.get("entry_usd") or 0.0)
            pnl = float(r.get("profit_loss") or 0.0)
        except (TypeError, ValueError):
            continue
        if notional <= 0:
            continue
        token = str(r.get("token_address") or r.get("token") or f"trade_{i}")
        qty = notional  # basis price 1.0 → qty == notional
        exit_price = 1.0 + (pnl / notional)
        if exit_price <= 0:
            exit_price = 1e-9  # -100% floor
        events.append({"token": token, "side": "buy", "qty": qty, "price_usd": 1.0, "ts": entry_ts})
        events.append({"token": token, "side": "sell", "qty": qty, "price_usd": exit_price, "ts": exit_ts})
    return events


# -------------------------------------------------------------------------
# Self-test (offline, asserts the no-look-ahead guard). Run:
#   python -m modules.copy_trading.wallet_profitability
# -------------------------------------------------------------------------
def _self_test() -> None:
    now = datetime(2026, 6, 13, tzinfo=timezone.utc)

    def ev(day, side, qty, price, token="TKN"):
        return {"token": token, "side": side, "qty": qty, "price_usd": price,
                "ts": now - timedelta(days=day)}

    # 1. FIFO realized PnL exactness.
    trips = fifo_round_trips(
        [ev(10, "buy", 100, 1.0), ev(9, "buy", 100, 2.0), ev(8, "sell", 150, 3.0)], now)
    assert len(trips) == 2
    assert abs(trips[0].pnl_usd - 200.0) < 1e-9   # 100 @ 1.0 -> 3.0
    assert abs(trips[1].pnl_usd - 50.0) < 1e-9    # 50 @ 2.0 -> 3.0

    # 2. Sell without basis is ignored, never fabricated.
    assert fifo_round_trips([ev(5, "sell", 10, 1.0)], now) == []

    # 3. NO LOOK-AHEAD: a future event must be excluded (and raise in strict).
    future = {"token": "TKN", "side": "sell", "qty": 100, "price_usd": 99.0,
              "ts": now + timedelta(days=1)}
    past_buy = ev(3, "buy", 100, 1.0)
    assert fifo_round_trips([past_buy, future], now) == []  # dropped silently
    try:
        fifo_round_trips([past_buy, future], now, strict=True)
        raise SystemExit("FAIL: look-ahead event not rejected in strict mode")
    except AssertionError:
        pass

    # 4. Window: a trip realized 40d ago is outside a 30d window.
    old = fifo_round_trips([ev(45, "buy", 10, 1.0), ev(40, "sell", 10, 2.0)], now)
    assert _window_trips(old, now, 30) == [] and _window_trips(old, now, 90) != []

    # 5. Lucky penalty: one moonshot among dust scores below a steady grinder.
    lucky_evs, steady_evs = [], []
    lucky_evs += [ev(20, "buy", 1000, 1.0, "MOON"), ev(19, "sell", 1000, 3.0, "MOON")]
    for i in range(10):
        lucky_evs += [ev(15 - i * 0.5, "buy", 10, 1.0, f"T{i}"),
                      ev(14.7 - i * 0.5, "sell", 10, 1.01, f"T{i}")]
    for i in range(20):
        steady_evs += [ev(25 - i, "buy", 100, 1.0, f"S{i % 6}"),
                       ev(24.7 - i, "sell", 100, 1.10, f"S{i % 6}")]
    lucky = score_wallet("solana", "LUCKY", lucky_evs, as_of=now)
    steady = score_wallet("solana", "STEADY", steady_evs, as_of=now)
    assert lucky.lucky_penalty > 0.3, lucky.lucky_penalty
    assert steady.lucky_penalty == 0.0
    assert steady.score > lucky.score, (steady.score, lucky.score)

    # 6. Wash penalty: sub-minute zero-PnL churn is penalized.
    wash_evs = []
    for i in range(20):
        t0 = now - timedelta(days=5, seconds=i * 120)
        wash_evs.append({"token": "W", "side": "buy", "qty": 100, "price_usd": 1.0, "ts": t0})
        wash_evs.append({"token": "W", "side": "sell", "qty": 100, "price_usd": 1.0001,
                         "ts": t0 + timedelta(seconds=10)})
    wash = score_wallet("ethereum", "0xWASH", wash_evs, as_of=now)
    assert wash.wash_penalty > 0.9, wash.wash_penalty
    assert wash.score < 5.0, wash.score

    # 7. Sample credit: 2 great trades cannot beat 20 good ones.
    two = score_wallet("solana", "TWO", [
        ev(5, "buy", 100, 1.0, "A"), ev(4.7, "sell", 100, 2.0, "A"),
        ev(3, "buy", 100, 1.0, "B"), ev(2.7, "sell", 100, 2.0, "B"),
    ], as_of=now)
    assert steady.score > two.score

    # 8. trades_to_events adapter preserves PnL exactly.
    rows = [{"entry_timestamp": now - timedelta(days=2),
             "exit_timestamp": now - timedelta(days=1),
             "profit_loss": 37.5, "entry_usd": 150.0, "token_address": "X"}]
    s = score_wallet("solana", "ADAPT", trades_to_events(rows), as_of=now)
    assert abs(s.realized_pnl_usd - 37.5) < 1e-6 and s.trade_count == 1

    # 9. Bounds: score always within [0, 100].
    for w in (lucky, steady, wash, two, s):
        assert 0.0 <= w.score <= 100.0

    # 10. Crowding penalty: soft-cap curve + it can only SHRINK the score.
    assert crowding_penalty_from_followers(CROWDING_FOLLOWERS_SOFTCAP) == 0.0
    assert 0.0 < crowding_penalty_from_followers(20) <= 0.5
    crowded = score_wallet("solana", "STEADY", steady_evs, as_of=now,
                           crowding=crowding_penalty_from_followers(30))
    assert crowded.crowding_penalty > 0.0 and crowded.score < steady.score

    print("wallet_profitability self-test OK "
          f"(steady={steady.score} lucky={lucky.score} wash={wash.score} two={two.score})")


if __name__ == "__main__":
    _self_test()
