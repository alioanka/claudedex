"""Pure fill-evaluation math for the intent_solver shadow scaffold.

No I/O, no network, no DB. Self-test (offline, fixture-backed):
    python -m modules.intent_solver.evaluator
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "intent_orders_fixture.json"


# ── normalization (pure parsing of raw API payloads) ──────────────────────

def normalize_cow_order(raw: dict, chain: str) -> Optional[dict]:
    """CoW orderbook order -> normalized intent dict (None if malformed)."""
    try:
        kind = str(raw["kind"]).lower()
        if kind not in ("sell", "buy"):
            return None
        sell_amount = int(raw["sellAmount"])
        buy_amount = int(raw["buyAmount"])
        if sell_amount <= 0 or buy_amount <= 0:
            return None
        return {
            "source": "cow",
            "chain": chain,
            "order_uid": str(raw["uid"]),
            "kind": kind,
            "sell_token": str(raw["sellToken"]).lower(),
            "buy_token": str(raw["buyToken"]).lower(),
            "sell_amount": sell_amount,
            "buy_amount": buy_amount,
            "partially_fillable": bool(raw.get("partiallyFillable", False)),
            "valid_to": int(raw.get("validTo", 0)),
            "order_class": str(raw.get("class", "")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def uniswapx_decayed_amount(start: int, end: int, decay_start: int,
                            decay_end: int, now_ts: float) -> int:
    """Linear Dutch-decay interpolation of a UniswapX amount at now_ts."""
    if now_ts <= decay_start or decay_end <= decay_start:
        return start
    if now_ts >= decay_end:
        return end
    frac = (now_ts - decay_start) / (decay_end - decay_start)
    return int(start + (end - start) * frac)


def normalize_uniswapx_order(raw: dict, chain: str, now_ts: float) -> Optional[dict]:
    """UniswapX open Dutch order -> normalized intent (exact-input == sell kind).

    Input amount is fixed; the required output decays start->end. The amount a
    filler must deliver RIGHT NOW is the decayed output (limit at now_ts).
    """
    try:
        inp = raw["input"]
        outs = raw.get("outputs") or []
        if not outs:
            return None
        decay_start = int(raw.get("decayStartTime", 0))
        decay_end = int(raw.get("decayEndTime", 0))
        sell_amount = int(inp["startAmount"])  # exact-input: start == end
        # Sum all outputs in the same token (swapper amount + fee outputs).
        out_token = str(outs[0]["token"]).lower()
        buy_amount = 0
        for o in outs:
            if str(o["token"]).lower() != out_token:
                return None  # mixed output tokens — skip, can't price as one leg
            buy_amount += uniswapx_decayed_amount(
                int(o["startAmount"]), int(o["endAmount"]),
                decay_start, decay_end, now_ts)
        if sell_amount <= 0 or buy_amount <= 0:
            return None
        return {
            "source": "uniswapx",
            "chain": chain,
            "order_uid": str(raw.get("orderHash", "")),
            "kind": "sell",
            "sell_token": str(inp["token"]).lower(),
            "buy_token": out_token,
            "sell_amount": sell_amount,
            "buy_amount": buy_amount,
            "partially_fillable": False,
            "valid_to": decay_end,
            "order_class": "dutch",
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


# ── fill evaluation ────────────────────────────────────────────────────────

def evaluate_fill(order: dict, quote_amount: int, *, now_ts: float,
                  gas_cost_usd: float, notional_usd: Optional[float],
                  safety_buffer_bps: float, min_edge_bps: float,
                  min_edge_usd: float, fallback_gas_bps: float) -> dict:
    """Could the bot have filled this intent profitably? (shadow math only)

    sell kind: quote_amount = buy_token obtainable on DEXes for sell_amount
               (net of route fees); limit = order min buy_amount.
               gross_edge = quote/limit - 1            (decimals cancel)
    buy kind:  quote_amount = sell_token needed on DEXes to source buy_amount;
               limit = order max sell_amount.
               gross_edge = 1 - quote/limit            (decimals cancel)
    net_edge_bps = gross_edge_bps - gas_bps - safety_buffer_bps
      gas_bps = gas_cost_usd / notional_usd (or fallback_gas_bps if unpriced)
    recordable iff net_edge_bps >= min_edge_bps
                AND (edge_usd unknown OR edge_usd >= min_edge_usd).
    """
    res = {
        "order_uid": order.get("order_uid"), "fillable": False,
        "recordable": False, "gross_edge_bps": None, "gas_bps": None,
        "net_edge_bps": None, "edge_usd": None, "notional_usd": notional_usd,
        "gas_cost_usd": gas_cost_usd, "buffer_bps": safety_buffer_bps,
        "skip_reason": None,
    }
    kind = order.get("kind")
    limit = order.get("buy_amount") if kind == "sell" else order.get("sell_amount")
    if kind not in ("sell", "buy") or not limit or limit <= 0 or quote_amount <= 0:
        res["skip_reason"] = "malformed"
        return res
    valid_to = order.get("valid_to") or 0
    if valid_to and now_ts >= valid_to:
        res["skip_reason"] = "expired"
        return res

    if kind == "sell":
        gross_frac = quote_amount / limit - 1.0
    else:
        gross_frac = 1.0 - quote_amount / limit
    gross_bps = gross_frac * 10_000.0

    if notional_usd and notional_usd > 0:
        gas_bps = (gas_cost_usd / notional_usd) * 10_000.0
    else:
        gas_bps = fallback_gas_bps
    net_bps = gross_bps - gas_bps - safety_buffer_bps

    res["gross_edge_bps"] = round(gross_bps, 4)
    res["gas_bps"] = round(gas_bps, 4)
    res["net_edge_bps"] = round(net_bps, 4)
    if notional_usd and notional_usd > 0:
        res["edge_usd"] = round(net_bps / 10_000.0 * notional_usd, 4)
    res["fillable"] = net_bps > 0
    res["recordable"] = (
        net_bps >= min_edge_bps
        and (res["edge_usd"] is None or res["edge_usd"] >= min_edge_usd)
    )
    if not res["recordable"] and res["skip_reason"] is None:
        res["skip_reason"] = "below_min_edge"
    return res


# ── offline fixture self-test ──────────────────────────────────────────────

def _self_test() -> int:
    fx = json.loads(FIXTURE_PATH.read_text())
    now = float(fx["now_ts"])
    kw = dict(now_ts=now, safety_buffer_bps=20.0, min_edge_bps=30.0,
              min_edge_usd=5.0, fallback_gas_bps=50.0)

    # 1. CoW sell order: 1 WETH -> min 2900 USDC, quote 3000 USDC.
    o1 = normalize_cow_order(fx["cow_orders"][0], "mainnet")
    assert o1 and o1["kind"] == "sell" and o1["sell_amount"] == 10**18
    r1 = evaluate_fill(o1, 3000_000000, gas_cost_usd=3.0, notional_usd=3000.0, **kw)
    assert abs(r1["gross_edge_bps"] - 344.8276) < 0.001, r1   # 3000/2900-1
    assert abs(r1["gas_bps"] - 10.0) < 1e-9
    assert abs(r1["net_edge_bps"] - 314.8276) < 0.001
    assert abs(r1["edge_usd"] - 94.4483) < 0.001 and r1["recordable"], r1

    # 2. CoW buy order: max 3100 USDC for 1 WETH, quote needs 3000 USDC.
    o2 = normalize_cow_order(fx["cow_orders"][1], "mainnet")
    assert o2 and o2["kind"] == "buy"
    r2 = evaluate_fill(o2, 3000_000000, gas_cost_usd=3.0, notional_usd=3100.0, **kw)
    assert abs(r2["gross_edge_bps"] - 322.5806) < 0.001 and r2["recordable"], r2

    # 3. Expired order is never fillable.
    o3 = normalize_cow_order(fx["cow_orders"][2], "mainnet")
    r3 = evaluate_fill(o3, 3000_000000, gas_cost_usd=3.0, notional_usd=3000.0, **kw)
    assert not r3["fillable"] and r3["skip_reason"] == "expired", r3

    # 4. Thin edge: quote 2905 vs limit 2900 -> ~17.2bps gross, net < 0.
    r4 = evaluate_fill(o1, 2905_000000, gas_cost_usd=3.0, notional_usd=2905.0, **kw)
    assert not r4["recordable"] and r4["skip_reason"] == "below_min_edge", r4

    # 5. Unpriced notional -> fallback gas bps, edge_usd None, bps gate only.
    r5 = evaluate_fill(o1, 3000_000000, gas_cost_usd=3.0, notional_usd=None, **kw)
    assert r5["edge_usd"] is None and abs(r5["gas_bps"] - 50.0) < 1e-9
    assert r5["recordable"], r5

    # 6. UniswapX Dutch decay: midpoint of 2950->2850 window = 2900.
    ux = normalize_uniswapx_order(fx["uniswapx_orders"][0], "1", now)
    assert ux and ux["kind"] == "sell" and ux["buy_amount"] == 2900_000000, ux
    rx = evaluate_fill(ux, 3000_000000, gas_cost_usd=4.0, notional_usd=3000.0, **kw)
    assert abs(rx["gross_edge_bps"] - 344.8276) < 0.001 and rx["recordable"], rx
    # Decay clamps at the ends.
    assert uniswapx_decayed_amount(100, 50, 1000, 2000, 500) == 100
    assert uniswapx_decayed_amount(100, 50, 1000, 2000, 9999) == 50

    # 7. Malformed payloads normalize to None, never raise.
    assert normalize_cow_order({"kind": "weird"}, "mainnet") is None
    assert normalize_cow_order({}, "mainnet") is None
    assert normalize_uniswapx_order({}, "1", now) is None

    print("intent_solver evaluator self-test OK "
          "(sell/buy edge math, expiry, min-edge gate, unpriced fallback, dutch decay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
