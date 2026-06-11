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
from typing import Any, Dict, List, Optional

MarketDict = Dict[str, Any]


def detect_risk_free_arb(
    markets: List[MarketDict],
    *,
    min_edge_bps: float = 100.0,
    fee_gas_buffer_bps: float = 100.0,
) -> List[Dict[str, Any]]:
    """Flag markets where yes_price + no_price < 1 - buffer.

    edge_bps = (1 - (yes + no)) * 10000 - fee_gas_buffer_bps, per $1 of the
    YES+NO pair. Only edges >= min_edge_bps are returned, best first.
    """
    signals: List[Dict[str, Any]] = []
    for m in markets:
        yes = m.get("yes_price")
        no = m.get("no_price")
        if yes is None or no is None or yes <= 0.0 or no <= 0.0:
            continue  # a zero leg means a dead/settled book, not free money
        gross_bps = (1.0 - (yes + no)) * 10000.0
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
                    "liquidity": m.get("liquidity"),
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
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Directional ADVICE signals from short-window YES-price moves + new markets.

    Transparent score in [0, 1]:
        move_term   = clamp(|delta| / (2 * min_move_frac), 0, 1)
        volume_term = clamp(volume_24h / (4 * min_volume_24h_usd), 0, 1)
        score       = 0.7 * move_term + 0.3 * volume_term
    New liquid markets (not in the previous snapshot) get signal_type
    'new_market' with score = 0.3 + 0.7 * volume_term.
    """
    now = now if now is not None else time.time()
    window_s = (now - prev_seen_at) if prev_seen_at else None
    signals: List[Dict[str, Any]] = []
    for m in markets:
        yes = m.get("yes_price")
        if yes is None:
            continue
        liquidity = m.get("liquidity") or 0.0
        volume = m.get("volume_24h") or 0.0
        if liquidity < min_liquidity_usd or volume < min_volume_24h_usd:
            continue
        volume_term = min(volume / (4.0 * min_volume_24h_usd), 1.0)
        prev = prev_yes_prices.get(m["market_id"])
        if prev is None:
            if not prev_yes_prices:
                continue  # first poll ever: everything is "new", skip the wave
            score = round(0.3 + 0.7 * volume_term, 3)
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
                    },
                }
            )
            continue
        delta = yes - prev
        if abs(delta) < min_move_frac or yes in (0.0, 1.0):
            continue
        move_term = min(abs(delta) / (2.0 * min_move_frac), 1.0)
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
                },
            }
        )
    signals.sort(key=lambda s: s["score"], reverse=True)
    return signals


def _self_test() -> int:
    mk = lambda i, y, n, liq=50000.0, vol=20000.0: {  # noqa: E731
        "market_id": i, "question": f"Q{i}", "yes_price": y, "no_price": n,
        "liquidity": liq, "volume_24h": vol,
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
    # Momentum: +0.10 move (2x threshold -> move_term 1.0), volume_term 1.0.
    prev = {"a": 0.36, "b": 0.50}
    moms = score_momentum([mk("a", 0.46, 0.51), mk("b", 0.51, 0.49)], prev,
                          now - 60 if (now := time.time()) else None)
    assert len(moms) == 1 and moms[0]["direction"] == "YES", moms
    assert abs(moms[0]["score"] - 1.0) < 1e-6
    # New liquid market detected only once a baseline snapshot exists.
    news = score_momentum([mk("z", 0.40, 0.60)], prev, time.time() - 60)
    assert len(news) == 1 and news[0]["signal_type"] == "new_market"
    assert not score_momentum([mk("z", 0.40, 0.60)], {}, None)  # first poll: quiet
    # Illiquid markets are ignored entirely.
    assert not score_momentum([mk("a", 0.46, 0.51, liq=100.0)], prev, None)
    print("strategies self-test OK (arb edge math, buffer gate, momentum, new-market)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
