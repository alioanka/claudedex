"""SMART_MONEY pure math: wallet forward-return scoring + accumulation clustering.

NO LOOK-AHEAD GUARANTEE: an event only contributes to a wallet's score when
``ts + max(horizon)*60 <= now_ts`` — i.e. every scoring horizon has FULLY
elapsed in wall-clock time. This is enforced here regardless of what marks the
caller supplies, so a buggy/poisoned mark can never leak future information.

Pure functions only (no I/O, no DB, no network). Self-test:
    python -m modules.smart_money.core.cluster_scorer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

SIDE_BUY = "buy"
SIDE_SELL = "sell"

# Logistic squash scale: a +10% avg forward return maps to ~0.73.
_RETURN_SQUASH_SCALE_PCT = 10.0


@dataclass(frozen=True)
class WalletEvent:
    wallet: str
    chain: str
    token: str
    side: str                       # 'buy' | 'sell'
    amount_usd: float
    price_usd: float                # token price at observation time
    ts: float                       # epoch seconds (event observation time)
    # horizon_minutes -> realized fwd return pct, marked AFTER the horizon
    # elapsed (late-marked, never early). Missing key = not marked yet.
    fwd_returns_pct: Dict[int, float] = field(default_factory=dict)


@dataclass
class WalletScore:
    wallet: str
    chain: str
    events_scored: int
    total_buy_usd: float
    avg_fwd_return_pct: float       # decay-weighted mean across marked horizons
    hit_rate: float                 # decay-weighted share of events with >0 blended return
    confidence: float               # 0..1, saturates at n_target events
    score: float                    # 0..1 final


@dataclass
class Cluster:
    chain: str
    token: str
    wallets: List[str]
    total_buy_usd: float
    first_ts: float
    last_ts: float


@dataclass
class SignalCandidate:
    chain: str
    token: str
    strength: float                 # 0..1
    cluster_wallets: int
    smart_wallets: int
    avg_wallet_score: float
    avg_fwd_return_pct: float       # historical, of the smart wallets
    crowding_factor: float
    total_buy_usd: float
    smart_wallet_list: List[str]


def decay_weight(event_ts: float, now_ts: float, half_life_days: float) -> float:
    """Exponential age decay; weight halves every half_life_days."""
    age_days = max(0.0, (now_ts - event_ts) / 86400.0)
    if half_life_days <= 0:
        return 1.0
    return 2.0 ** (-age_days / half_life_days)


def _squash_return_pct(pct: float) -> float:
    """Map a return pct to (0,1); 0% -> 0.5."""
    return 1.0 / (1.0 + math.exp(-pct / _RETURN_SQUASH_SCALE_PCT))


def crowding_factor(n_wallets: int, soft_cap: int) -> float:
    """1.0 up to soft_cap participants, then decays — crowded accumulation is
    late accumulation; the edge decays with the crowd."""
    if n_wallets <= 0:
        return 0.0
    if soft_cap <= 0 or n_wallets <= soft_cap:
        return 1.0
    return soft_cap / float(n_wallets)


def score_wallet(events: Sequence[WalletEvent], *, now_ts: float,
                 horizons_minutes: Sequence[int], half_life_days: float,
                 min_events: int, n_target: int) -> Optional[WalletScore]:
    """Score one wallet by realized FORWARD return of its past BUY events.

    An event is scorable only if (a) it is a buy, (b) its LONGEST horizon has
    fully elapsed (``ts + max_h*60 <= now_ts`` — the no-look-ahead gate), and
    (c) at least one horizon mark is present. Returns None below min_events.
    """
    if not horizons_minutes:
        return None
    max_h_secs = max(horizons_minutes) * 60.0
    scorable: List[WalletEvent] = []
    for e in events:
        if e.side != SIDE_BUY:
            continue
        if e.ts + max_h_secs > now_ts:      # horizon not elapsed: NEVER scored
            continue
        marked = [h for h in horizons_minutes if h in e.fwd_returns_pct]
        if not marked:
            continue
        scorable.append(e)
    if len(scorable) < max(1, min_events):
        return None

    w_sum = 0.0
    ret_sum = 0.0
    hit_sum = 0.0
    total_buy = 0.0
    for e in scorable:
        w = decay_weight(e.ts, now_ts, half_life_days)
        marked = [e.fwd_returns_pct[h] for h in horizons_minutes
                  if h in e.fwd_returns_pct]
        blended = sum(marked) / len(marked)
        w_sum += w
        ret_sum += w * blended
        hit_sum += w * (1.0 if blended > 0 else 0.0)
        total_buy += e.amount_usd
    if w_sum <= 0:
        return None
    avg_ret = ret_sum / w_sum
    hit_rate = hit_sum / w_sum
    confidence = min(1.0, len(scorable) / float(max(1, n_target)))
    raw = 0.5 * hit_rate + 0.5 * _squash_return_pct(avg_ret)
    first = scorable[0]
    return WalletScore(
        wallet=first.wallet, chain=first.chain, events_scored=len(scorable),
        total_buy_usd=total_buy, avg_fwd_return_pct=avg_ret, hit_rate=hit_rate,
        confidence=confidence, score=raw * confidence,
    )


def detect_clusters(events: Sequence[WalletEvent], *, now_ts: float,
                    window_minutes: int, min_wallets: int) -> List[Cluster]:
    """Group recent BUY events into per-(chain, token) accumulation clusters
    with at least min_wallets DISTINCT wallets in the trailing window."""
    cutoff = now_ts - window_minutes * 60.0
    grouped: Dict[tuple, List[WalletEvent]] = {}
    for e in events:
        if e.side != SIDE_BUY or e.ts < cutoff or e.ts > now_ts:
            continue
        grouped.setdefault((e.chain, e.token), []).append(e)
    out: List[Cluster] = []
    for (chain, token), evs in grouped.items():
        wallets = sorted({e.wallet for e in evs})
        if len(wallets) < max(1, min_wallets):
            continue
        out.append(Cluster(
            chain=chain, token=token, wallets=wallets,
            total_buy_usd=sum(e.amount_usd for e in evs),
            first_ts=min(e.ts for e in evs), last_ts=max(e.ts for e in evs),
        ))
    out.sort(key=lambda c: c.total_buy_usd, reverse=True)
    return out


def build_signals(clusters: Sequence[Cluster],
                  wallet_scores: Dict[tuple, WalletScore], *,
                  min_wallet_score: float, min_smart_wallets: int,
                  min_forward_return_pct: float,
                  crowding_soft_cap: int) -> List[SignalCandidate]:
    """Emit an ADVISORY accumulation candidate per cluster whose participants
    include enough historically-profitable ('smart') wallets. Pure filter —
    can only drop clusters, never invent them."""
    out: List[SignalCandidate] = []
    for c in clusters:
        smart = [wallet_scores[(c.chain, w)] for w in c.wallets
                 if (c.chain, w) in wallet_scores
                 and wallet_scores[(c.chain, w)].score >= min_wallet_score]
        if len(smart) < max(1, min_smart_wallets):
            continue
        avg_ret = sum(s.avg_fwd_return_pct for s in smart) / len(smart)
        if avg_ret < min_forward_return_pct:
            continue
        avg_score = sum(s.score for s in smart) / len(smart)
        crowd = crowding_factor(len(c.wallets), crowding_soft_cap)
        out.append(SignalCandidate(
            chain=c.chain, token=c.token,
            strength=min(1.0, avg_score * crowd),
            cluster_wallets=len(c.wallets), smart_wallets=len(smart),
            avg_wallet_score=avg_score, avg_fwd_return_pct=avg_ret,
            crowding_factor=crowd, total_buy_usd=c.total_buy_usd,
            smart_wallet_list=[s.wallet for s in smart],
        ))
    out.sort(key=lambda s: s.strength, reverse=True)
    return out


# ───────────────────────────── self-test ────────────────────────────────────

def _selftest() -> None:
    now = 1_000_000_000.0
    horizons = [60, 360]
    day = 86400.0

    def ev(wallet, token, ts, fwd, side=SIDE_BUY, usd=5000.0):
        return WalletEvent(wallet=wallet, chain="ethereum", token=token,
                           side=side, amount_usd=usd, price_usd=1.0, ts=ts,
                           fwd_returns_pct=fwd)

    # 1. Winner wallet (buys before rises) outscores loser (buys before dumps).
    winner = [ev("w1", "TOKA", now - (i + 2) * day, {60: 8.0, 360: 15.0})
              for i in range(4)]
    loser = [ev("w2", "TOKB", now - (i + 2) * day, {60: -6.0, 360: -12.0})
             for i in range(4)]
    common = dict(now_ts=now, horizons_minutes=horizons, half_life_days=14,
                  min_events=3, n_target=10)
    s_win = score_wallet(winner, **common)
    s_lose = score_wallet(loser, **common)
    assert s_win is not None and s_lose is not None
    assert s_win.score > s_lose.score, "winner must outscore loser"
    assert s_win.hit_rate == 1.0 and s_lose.hit_rate == 0.0

    # 2. NO LOOK-AHEAD: a recent event whose longest horizon has NOT elapsed
    #    is excluded even if a (bogus) mark claims a future return.
    poisoned = ev("w1", "TOKA", now - 60.0, {60: 500.0, 360: 500.0})
    s_with = score_wallet(winner + [poisoned], **common)
    assert s_with is not None
    assert s_with.events_scored == s_win.events_scored
    assert abs(s_with.score - s_win.score) < 1e-12, "unelapsed event leaked"

    # 3. Unmarked + sell events never score; min_events floor returns None.
    unmarked = [ev("w3", "TOKA", now - 5 * day, {}) for _ in range(5)]
    sells = [ev("w3", "TOKA", now - 5 * day, {60: 9.0, 360: 9.0}, side=SIDE_SELL)
             for _ in range(5)]
    assert score_wallet(unmarked + sells, **common) is None
    assert score_wallet(winner[:2], **common) is None  # below min_events

    # 4. Decay: identical returns, fresher events -> higher weight.
    assert decay_weight(now - day, now, 14) > decay_weight(now - 20 * day, now, 14)

    # 5. Clusters: 3 distinct wallets in-window -> cluster; stale buys don't count.
    recent = [ev(f"w{i}", "TOKC", now - 600.0, {}) for i in range(3)]
    stale = [ev("w9", "TOKD", now - 7 * day, {})]
    cl = detect_clusters(recent + stale, now_ts=now, window_minutes=45,
                         min_wallets=3)
    assert len(cl) == 1 and cl[0].token == "TOKC" and len(cl[0].wallets) == 3
    assert not detect_clusters(recent[:2], now_ts=now, window_minutes=45,
                               min_wallets=3)

    # 6. Crowding: flat up to the soft cap, then monotone decreasing.
    assert crowding_factor(5, 10) == 1.0
    assert crowding_factor(20, 10) == 0.5
    assert crowding_factor(40, 10) == 0.25

    # 7. Signals: cluster of scored-smart wallets emits; unknown wallets don't.
    scores = {("ethereum", f"w{i}"): WalletScore(
        wallet=f"w{i}", chain="ethereum", events_scored=5, total_buy_usd=2e4,
        avg_fwd_return_pct=9.0, hit_rate=0.8, confidence=0.8, score=0.7)
        for i in range(3)}
    sig = build_signals(cl, scores, min_wallet_score=0.55, min_smart_wallets=2,
                        min_forward_return_pct=3.0, crowding_soft_cap=12)
    assert len(sig) == 1 and sig[0].smart_wallets == 3
    assert 0.0 < sig[0].strength <= 1.0
    assert not build_signals(cl, {}, min_wallet_score=0.55, min_smart_wallets=2,
                             min_forward_return_pct=3.0, crowding_soft_cap=12)

    print("cluster_scorer selftest OK: "
          f"winner={s_win.score:.3f} loser={s_lose.score:.3f} "
          f"signal_strength={sig[0].strength:.3f}")


if __name__ == "__main__":
    _selftest()
