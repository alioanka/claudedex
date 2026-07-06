"""Polymarket shadow strategies — pure functions, no I/O, no side effects.

(a) detect_risk_free_arb: YES + NO bought together cost < $1 minus a fee/gas
    buffer -> guaranteed $1 redemption edge. NOTE: Gamma outcomePrices are
    mid/last prices, NOT executable asks — shadow edges are an upper bound and
    the live path MUST re-quote against the CLOB order book before any order.
(b) score_momentum: large short-window price moves on liquid markets, plus
    brand-new liquid markets. ADVICE/shadow only; transparent score formula.

Self-test (offline): python -m modules.polymarket.strategies
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

MarketDict = Dict[str, Any]

# Markets priced this close to resolution are dead books, not signals
# (yes=0.9995 "arb"/"momentum" is noise; see wave-F5 BUG-5).
NEAR_RESOLUTION_MIN = 0.02
NEAR_RESOLUTION_MAX = 0.98


def _parse_iso_ts(value: Any) -> Optional[float]:
    """ISO-8601 string (Gamma createdAt/startDate) -> epoch seconds, or None."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def detect_risk_free_arb(
    markets: List[MarketDict],
    *,
    min_edge_bps: float = 100.0,
    fee_gas_buffer_bps: float = 100.0,
    min_liquidity_usd: float = 1000.0,
    max_edge_bps: float = 500.0,
) -> List[Dict[str, Any]]:
    """Flag markets where yes_price + no_price < 1 - buffer.

    edge_bps = (1 - (yes + no)) * 10000 - fee_gas_buffer_bps, per $1 of the
    YES+NO pair. Only edges >= min_edge_bps are returned, best first.

    Book-honesty gates (wave-F5 BUG-1: the only firing in 20 days was a dead
    tennis book with drifting mids):
      * liquidity and 24h volume must be present and >= min_liquidity_usd / >0
      * best_bid/best_ask must be present, positive and uncrossed — a market
        with no live order book cannot host an executable arb
      * gross edge > max_edge_bps is rejected as a stale-book false positive:
        a real YES+NO book does not sit >max_edge_bps below $1 for a full
        poll cycle; that is a settled/dead market, not free money.
    """
    signals: List[Dict[str, Any]] = []
    for m in markets:
        yes = m.get("yes_price")
        no = m.get("no_price")
        if yes is None or no is None or yes <= 0.0 or no <= 0.0:
            continue  # a zero leg means a dead/settled book, not free money
        liquidity = m.get("liquidity") or 0.0
        volume = m.get("volume_24h") or 0.0
        if liquidity < max(min_liquidity_usd, 1e-9) or volume <= 0.0:
            continue  # no order-book-backed liquidity/volume -> no signal
        best_bid = m.get("best_bid")
        best_ask = m.get("best_ask")
        if not best_bid or not best_ask or best_bid <= 0.0 or best_ask <= 0.0:
            continue  # no live book quotes -> mid prices are untradeable
        if best_bid > best_ask:
            continue  # crossed book = stale/garbage data
        gross_bps = (1.0 - (yes + no)) * 10000.0
        if gross_bps > max_edge_bps:
            continue  # too good to be true = stale book, not an arb
        edge_bps = gross_bps - fee_gas_buffer_bps
        if edge_bps < min_edge_bps:
            continue
        signals.append(
            {
                "signal_type": "risk_free_arb",
                "market_id": m["market_id"],
                "question": m.get("question", ""),
                "direction": "BOTH",
                "yes_price": yes,
                "no_price": no,
                "edge_bps": round(edge_bps, 2),
                "score": round(min(edge_bps / 1000.0, 1.0), 3),
                "details": {
                    "gross_edge_bps": round(gross_bps, 2),
                    "fee_gas_buffer_bps": fee_gas_buffer_bps,
                    "pair_cost": round(yes + no, 6),
                    "yes_token_id": m.get("yes_token_id"),
                    "no_token_id": m.get("no_token_id"),
                    "liquidity": liquidity,
                    "volume_24h": volume,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "price_source": "gamma_mid",  # honesty: not executable asks
                },
            }
        )
    signals.sort(key=lambda s: s["edge_bps"], reverse=True)
    return signals


def score_momentum(
    markets: List[MarketDict],
    prev_yes_prices: Dict[str, float],
    prev_seen_at: Optional[float] = None,
    *,
    min_liquidity_usd: float = 10000.0,
    min_volume_24h_usd: float = 5000.0,
    min_move_frac: float = 0.05,
    min_score: float = 0.3,
    exclude_categories: Optional[Set[str]] = None,
    new_market_max_age_hours: float = 24.0,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Directional ADVICE signals from short-window YES-price moves + new markets.

    Transparent score in [0, 1), asymptotic — a score of exactly 1.0 is
    impossible, so scores rank (wave-F5 BUG-4: the old clamped formula pinned
    70.6% of signals at 1.0):
        move_ratio  = |delta| / min_move_frac          (>= 1 by the gate)
        move_term   = move_ratio / (move_ratio + 1)    (0.5 at threshold)
        vol_ratio   = volume_24h / min_volume_24h_usd  (>= 1 by the gate)
        volume_term = vol_ratio / (vol_ratio + 4)      (0.2 at threshold)
        score       = 0.7 * move_term + 0.3 * volume_term
    'new_market' (score = 0.3 + 0.6 * volume_term) is only emitted for
    markets whose Gamma created_at is younger than new_market_max_age_hours —
    an old market re-entering the volume window is NOT new (BUG-5).
    Near-resolution markets (yes outside [0.02, 0.98]) never signal.
    exclude_categories drops whole categories (e.g. in-play sports noise).
    """
    now = now if now is not None else time.time()
    window_s = (now - prev_seen_at) if prev_seen_at else None
    excluded = {c.lower() for c in (exclude_categories or set())}
    signals: List[Dict[str, Any]] = []
    for m in markets:
        yes = m.get("yes_price")
        if yes is None:
            continue
        if yes < NEAR_RESOLUTION_MIN or yes > NEAR_RESOLUTION_MAX:
            continue  # effectively resolved — advice on a 0.9995 book is noise
        if excluded and str(m.get("category", "")).lower() in excluded:
            continue
        liquidity = m.get("liquidity") or 0.0
        volume = m.get("volume_24h") or 0.0
        if liquidity < min_liquidity_usd or volume < min_volume_24h_usd:
            continue
        vol_ratio = volume / max(min_volume_24h_usd, 1e-9)
        volume_term = vol_ratio / (vol_ratio + 4.0)
        prev = prev_yes_prices.get(m["market_id"])
        if prev is None:
            if not prev_yes_prices:
                continue  # first poll ever: everything is "new", skip the wave
            created_ts = _parse_iso_ts(m.get("created_at"))
            if created_ts is None:
                continue  # cannot verify it is actually new — do not mislabel
            age_hours = (now - created_ts) / 3600.0
            if age_hours > new_market_max_age_hours:
                continue  # old market re-entering the volume window, not new
            score = round(0.3 + 0.6 * volume_term, 3)
            if score < min_score:
                continue
            signals.append(
                {
                    "signal_type": "new_market",
                    "market_id": m["market_id"],
                    "question": m.get("question", ""),
                    "direction": "WATCH",
                    "yes_price": yes,
                    "no_price": m.get("no_price"),
                    "edge_bps": None,
                    "score": score,
                    "details": {
                        "liquidity": liquidity,
                        "volume_24h": volume,
                        "volume_term": round(volume_term, 3),
                        "age_hours": round(age_hours, 2),
                        "category": m.get("category", ""),
                    },
                }
            )
            continue
        delta = yes - prev
        if abs(delta) < min_move_frac:
            continue
        move_ratio = abs(delta) / max(min_move_frac, 1e-9)
        move_term = move_ratio / (move_ratio + 1.0)
        score = round(0.7 * move_term + 0.3 * volume_term, 3)
        if score < min_score:
            continue
        signals.append(
            {
                "signal_type": "momentum",
                "market_id": m["market_id"],
                "question": m.get("question", ""),
                "direction": "YES" if delta > 0 else "NO",
                "yes_price": yes,
                "no_price": m.get("no_price"),
                "edge_bps": None,
                "score": score,
                "details": {
                    "prev_yes_price": prev,
                    "delta": round(delta, 6),
                    "window_s": round(window_s, 1) if window_s else None,
                    "move_term": round(move_term, 3),
                    "volume_term": round(volume_term, 3),
                    "liquidity": liquidity,
                    "volume_24h": volume,
                    "category": m.get("category", ""),
                },
            }
        )
    signals.sort(key=lambda s: s["score"], reverse=True)
    return signals


def _self_test() -> int:
    fresh_iso = datetime.now(timezone.utc).isoformat()
    mk = lambda i, y, n, liq=50000.0, vol=20000.0, bid=None, ask=None, cat="", created=None: {  # noqa: E731,E501
        "market_id": i, "question": f"Q{i}", "yes_price": y, "no_price": n,
        "liquidity": liq, "volume_24h": vol,
        "best_bid": bid if bid is not None else min(y, n) - 0.01,
        "best_ask": ask if ask is not None else min(y, n) + 0.01,
        "category": cat, "created_at": created,
        "yes_token_id": f"{i}-yes", "no_token_id": f"{i}-no",
    }
    # Arb: 0.46+0.51=0.97 -> gross 300bps, buffer 100 -> net 200 >= 100. Fires.
    arbs = detect_risk_free_arb(
        [mk("a", 0.46, 0.51), mk("b", 0.50, 0.50), mk("c", 0.0, 0.90)],
        min_edge_bps=100.0, fee_gas_buffer_bps=100.0,
    )
    assert len(arbs) == 1 and arbs[0]["market_id"] == "a", arbs
    assert abs(arbs[0]["edge_bps"] - 200.0) < 1e-6
    # Buffer kills a thin edge: 0.495+0.495 -> gross 100, net 0 < 100.
    assert not detect_risk_free_arb([mk("d", 0.495, 0.495)],
                                    min_edge_bps=100.0, fee_gas_buffer_bps=100.0)
    # BUG-1 gates: no book quotes / no liquidity / no volume -> no signal.
    assert not detect_risk_free_arb([mk("e", 0.46, 0.51, bid=0.0)])
    assert not detect_risk_free_arb([mk("f", 0.46, 0.51, liq=0.0)])
    assert not detect_risk_free_arb([mk("g", 0.46, 0.51, vol=0.0)])
    # Stale-book cap: 0.30+0.40=0.70 -> gross 3000bps > max 500 -> rejected.
    assert not detect_risk_free_arb([mk("h", 0.30, 0.40)], max_edge_bps=500.0)
    # Momentum: +0.10 move (2x threshold), high volume; score < 1.0 always.
    prev = {"a": 0.36, "b": 0.50}
    moms = score_momentum([mk("a", 0.46, 0.51), mk("b", 0.51, 0.49)], prev,
                          now - 60 if (now := time.time()) else None)
    assert len(moms) == 1 and moms[0]["direction"] == "YES", moms
    assert 0.3 <= moms[0]["score"] < 1.0, moms[0]["score"]
    # Category exclusion drops in-play sports noise.
    assert not score_momentum([mk("a", 0.46, 0.51, cat="Sports")], prev,
                              time.time() - 60, exclude_categories={"sports"})
    # Near-resolution book never signals.
    assert not score_momentum([mk("a", 0.995, 0.005)], {"a": 0.90}, time.time() - 60)
    # new_market requires a verifiably fresh created_at (BUG-5).
    news = score_momentum([mk("z", 0.40, 0.60, created=fresh_iso)], prev, time.time() - 60)
    assert len(news) == 1 and news[0]["signal_type"] == "new_market"
    assert news[0]["score"] < 1.0
    assert not score_momentum([mk("z2", 0.40, 0.60)], prev, time.time() - 60)  # no created_at
    assert not score_momentum(
        [mk("z3", 0.40, 0.60, created="2020-01-01T00:00:00Z")], prev, time.time() - 60)
    assert not score_momentum([mk("z", 0.40, 0.60, created=fresh_iso)], {}, None)  # first poll
    # Illiquid markets are ignored entirely.
    assert not score_momentum([mk("a", 0.46, 0.51, liq=100.0)], prev, None)
    print("strategies self-test OK (arb book gates + stale cap, momentum "
          "normalization, category/near-resolution filters, new-market age)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
