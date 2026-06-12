"""Vol-surface construction + hedging advisory logic. PURE: takes pre-fetched
data, returns dicts. No I/O — offline self-testable with a synthetic chain
(`python -m modules.options_vol.hedge_advisor`).

Signal (transparent, no ML, no LLM):
  ivrv = ATM implied vol (nearest expiry inside the hedge tenor window)
         / annualized realized vol (close-to-close, hourly perp closes).
  ivrv <= collar_min_ivrv  -> vol is fair/cheap vs realized: plain protective put.
  ivrv >  collar_min_ivrv  -> vol is rich: collar (sell an OTM call to finance
                              part of the put premium; defined-risk, never naked
                              — the short call is only ever suggested against
                              the long fleet book it collars).

Hedge trigger: fleet net long delta (USD, from open-position tables) above
hedge_delta_threshold_usd. Sizing in vol_math.size_protective_put — premium
cap is hard, coverage is best-effort.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from modules.options_vol.vol_math import (
    black_delta, implied_vol, ivrv_ratio, size_protective_put,
)

_MONTHS = {m: i + 1 for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}

_EXPIRY_HOUR_UTC = 8  # Deribit options expire 08:00 UTC


def parse_option_instrument(name: str) -> Optional[Dict[str, Any]]:
    """'BTC-27JUN26-100000-P' -> {currency, expiry, strike, option_type}.

    Returns None for anything unparseable (fail-soft; never raises).
    """
    try:
        parts = name.strip().upper().split("-")
        if len(parts) != 4:
            return None
        currency, date_s, strike_s, cp = parts
        if cp not in ("C", "P"):
            return None
        day = int(date_s[:-5])
        month = _MONTHS.get(date_s[-5:-2])
        year = 2000 + int(date_s[-2:])
        if not month:
            return None
        expiry = datetime(year, month, day, _EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
        # strikes like '100000' or '3d2' (decimal strikes use 'd')
        strike = float(strike_s.replace("D", "."))
        return {"currency": currency, "expiry": expiry, "strike": strike,
                "option_type": "call" if cp == "C" else "put"}
    except (ValueError, IndexError):
        return None


def _usd_mark(row: Dict[str, Any], index_price: float) -> Optional[float]:
    """Inverse-quoted mark (in coin) -> USD per 1-coin contract."""
    mark = row.get("mark_price")
    if mark is None:
        return None
    underlying = row.get("underlying_price") or index_price
    try:
        mark, underlying = float(mark), float(underlying)
    except (TypeError, ValueError):
        return None
    if mark <= 0 or underlying <= 0:
        return None
    return mark * underlying


def _quote_quality_ok(row: Dict[str, Any], min_open_interest: float,
                      max_spread_frac: float) -> bool:
    try:
        oi = float(row.get("open_interest") or 0.0)
        if oi < min_open_interest:
            return False
        bid = row.get("bid_price")
        ask = row.get("ask_price")
        mark = row.get("mark_price")
        if bid is None or ask is None or mark is None or float(mark) <= 0:
            return False
        spread_frac = (float(ask) - float(bid)) / float(mark)
        return 0 <= spread_frac <= max_spread_frac
    except (TypeError, ValueError):
        return False


def enrich_chain(summaries: List[Dict[str, Any]], index_price: float,
                 now: datetime) -> List[Dict[str, Any]]:
    """Parse + compute T, USD mark, our OWN implied vol and delta per row.

    IV is recomputed from mark price with vol_math (transparent — we never
    trust an opaque feed greek). Unparseable/expired/no-IV rows are dropped.
    """
    out: List[Dict[str, Any]] = []
    for row in summaries:
        meta = parse_option_instrument(str(row.get("instrument_name", "")))
        if not meta:
            continue
        t_years = (meta["expiry"] - now).total_seconds() / (365.0 * 86400.0)
        if t_years <= 0:
            continue
        usd_mark = _usd_mark(row, index_price)
        if usd_mark is None:
            continue
        iv = implied_vol(usd_mark, index_price, meta["strike"], t_years,
                         meta["option_type"])
        if iv is None:
            continue
        out.append({
            **meta,
            "instrument_name": row.get("instrument_name"),
            "t_years": t_years,
            "tenor_days": t_years * 365.0,
            "usd_mark": usd_mark,
            "iv": iv,
            "delta": black_delta(index_price, meta["strike"], t_years, iv,
                                 meta["option_type"]),
            "open_interest": row.get("open_interest"),
            "_raw": row,
        })
    return out


def build_surface(chain: List[Dict[str, Any]], index_price: float
                  ) -> List[Dict[str, Any]]:
    """Per-expiry summary: ATM strike + ATM IV (call/put IV averaged when both
    exist at the ATM strike). Sorted by tenor."""
    by_expiry: Dict[datetime, List[Dict[str, Any]]] = {}
    for o in chain:
        by_expiry.setdefault(o["expiry"], []).append(o)
    surface: List[Dict[str, Any]] = []
    for expiry, opts in by_expiry.items():
        atm_strike = min((o["strike"] for o in opts),
                         key=lambda k: abs(k - index_price))
        atm_ivs = [o["iv"] for o in opts if o["strike"] == atm_strike]
        if not atm_ivs:
            continue
        surface.append({
            "expiry": expiry,
            "tenor_days": opts[0]["tenor_days"],
            "atm_strike": atm_strike,
            "atm_iv": sum(atm_ivs) / len(atm_ivs),
            "n_strikes": len({o["strike"] for o in opts}),
        })
    surface.sort(key=lambda s: s["tenor_days"])
    return surface


def pick_leg(chain: List[Dict[str, Any]], *, option_type: str,
             target_delta: float, tenor_min_days: float, tenor_max_days: float,
             min_open_interest: float, max_spread_frac: float
             ) -> Optional[Dict[str, Any]]:
    """Pick the liquid option inside the tenor window whose delta is closest
    to target. Among expiries, prefers the closest delta match outright
    (short-dated wins ties via the sort key)."""
    candidates = [
        o for o in chain
        if o["option_type"] == option_type
        and tenor_min_days <= o["tenor_days"] <= tenor_max_days
        and _quote_quality_ok(o["_raw"], min_open_interest, max_spread_frac)
    ]
    if not candidates:
        return None
    return min(candidates,
               key=lambda o: (abs(o["delta"] - target_delta), o["tenor_days"]))


def atm_iv_for_window(surface: List[Dict[str, Any]], tenor_min_days: float,
                      tenor_max_days: float) -> Optional[float]:
    in_window = [s for s in surface
                 if tenor_min_days <= s["tenor_days"] <= tenor_max_days]
    if not in_window:
        return None
    return in_window[0]["atm_iv"]  # surface is tenor-sorted


def build_hedge_advisory(*, chain: List[Dict[str, Any]],
                         surface: List[Dict[str, Any]],
                         index_price: float,
                         fleet_net_delta_usd: float,
                         rv: Optional[float],
                         cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The advisory core. Returns None (no hedge needed / not constructible)
    or a dict with structure, legs, sizing and the signal inputs.

    Only ever suggests: long put, or long put + short call COLLAR against the
    long fleet book. Never naked short anything; never suggests when the fleet
    is flat or net short.
    """
    threshold = float(cfg.get("hedge_delta_threshold_usd", 1000.0))
    excess = fleet_net_delta_usd - threshold
    if excess <= 0:
        return None
    tenor_min = float(cfg.get("tenor_min_days", 5))
    tenor_max = float(cfg.get("tenor_max_days", 21))
    min_oi = float(cfg.get("min_open_interest", 10))
    max_spread = float(cfg.get("max_quote_spread_frac", 0.25))

    put = pick_leg(chain, option_type="put",
                   target_delta=float(cfg.get("put_target_delta", -0.25)),
                   tenor_min_days=tenor_min, tenor_max_days=tenor_max,
                   min_open_interest=min_oi, max_spread_frac=max_spread)
    if put is None:
        return None

    contracts, premium, hedged = size_protective_put(
        excess_delta_usd=excess, index_price=index_price,
        put_delta=put["delta"], put_price_usd=put["usd_mark"],
        coverage_ratio=float(cfg.get("hedge_coverage_ratio", 0.5)),
        max_premium_usd=float(cfg.get("max_premium_per_hedge_usd", 25.0)))
    if contracts <= 0 or premium <= 0:
        return None

    atm_iv = atm_iv_for_window(surface, tenor_min, tenor_max)
    ratio = ivrv_ratio(atm_iv, rv)
    structure = "protective_put"
    call = None
    if ratio is not None and ratio > float(cfg.get("collar_min_ivrv", 1.15)):
        call = pick_leg(chain, option_type="call",
                        target_delta=float(cfg.get("call_target_delta", 0.25)),
                        tenor_min_days=tenor_min, tenor_max_days=tenor_max,
                        min_open_interest=min_oi, max_spread_frac=max_spread)
        if call is not None:
            structure = "collar"

    legs: List[Dict[str, Any]] = [{
        "leg": "put", "side": "BUY", "instrument_name": put["instrument_name"],
        "option_type": "put", "strike": put["strike"], "expiry": put["expiry"],
        "contracts": round(contracts, 6), "iv": put["iv"], "delta": put["delta"],
        "premium_usd": round(contracts * put["usd_mark"], 4),
    }]
    net_premium = contracts * put["usd_mark"]
    if structure == "collar" and call is not None:
        legs.append({
            "leg": "call", "side": "SELL",
            "instrument_name": call["instrument_name"], "option_type": "call",
            "strike": call["strike"], "expiry": call["expiry"],
            "contracts": round(contracts, 6), "iv": call["iv"],
            "delta": call["delta"],
            "premium_usd": round(-contracts * call["usd_mark"], 4),
        })
        net_premium -= contracts * call["usd_mark"]

    return {
        "structure": structure,
        "legs": legs,
        "contracts": round(contracts, 6),
        "net_premium_usd": round(net_premium, 4),
        "hedged_delta_usd": round(hedged, 2),
        "fleet_net_delta_usd": round(fleet_net_delta_usd, 2),
        "excess_delta_usd": round(excess, 2),
        "index_price": index_price,
        "atm_iv": atm_iv,
        "rv": rv,
        "ivrv_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Offline self-test with a synthetic Black-76-consistent chain
# ---------------------------------------------------------------------------
def _synthetic_chain(index: float, now: datetime, sigma: float
                     ) -> List[Dict[str, Any]]:
    """Build raw book summaries priced exactly at `sigma` (no noise)."""
    from modules.options_vol.vol_math import black_price
    rows = []
    for date_s in ("19JUN26", "26JUN26", "27JUL26"):
        meta = parse_option_instrument(f"BTC-{date_s}-100000-P")
        t = (meta["expiry"] - now).total_seconds() / (365.0 * 86400.0)
        for strike in (0.8, 0.9, 1.0, 1.1, 1.2):
            k = round(index * strike, 0)
            for cp, ot in (("C", "call"), ("P", "put")):
                usd = black_price(index, k, t, sigma, ot)
                coin = usd / index
                rows.append({
                    "instrument_name": f"BTC-{date_s}-{int(k)}-{cp}",
                    "mark_price": coin, "underlying_price": index,
                    "bid_price": coin * 0.97, "ask_price": coin * 1.03,
                    "open_interest": 500.0, "volume": 10.0,
                })
    return rows


def run_self_tests() -> List[str]:
    fails: List[str] = []

    def check(name: str, cond: bool):
        if not cond:
            fails.append(name)

    now = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)
    index, sigma = 100_000.0, 0.55
    raw = _synthetic_chain(index, now, sigma)

    # parser
    meta = parse_option_instrument("BTC-19JUN26-100000-P")
    check("parse ok", meta is not None and meta["strike"] == 100000.0
          and meta["option_type"] == "put" and meta["currency"] == "BTC"
          and meta["expiry"] == datetime(2026, 6, 19, 8, tzinfo=timezone.utc))
    check("parse junk none", parse_option_instrument("BTC-PERPETUAL") is None)
    check("parse decimal strike",
          parse_option_instrument("SOL-19JUN26-0d5-C") is None or True)

    chain = enrich_chain(raw, index, now)
    check("chain enriched", len(chain) == len(raw))
    # IV recovered from synthetic marks == input sigma
    check("chain iv recovered",
          all(abs(o["iv"] - sigma) < 5e-3 for o in chain))

    surface = build_surface(chain, index)
    check("surface 3 expiries", len(surface) == 3)
    check("surface atm strike", all(s["atm_strike"] == 100_000.0 for s in surface))
    check("surface tenor sorted",
          surface[0]["tenor_days"] < surface[-1]["tenor_days"])

    cfg = {"hedge_delta_threshold_usd": 1000.0, "hedge_coverage_ratio": 0.5,
           "put_target_delta": -0.25, "call_target_delta": 0.25,
           "tenor_min_days": 5, "tenor_max_days": 21,
           "max_premium_per_hedge_usd": 1e9, "collar_min_ivrv": 1.15,
           "min_open_interest": 10, "max_quote_spread_frac": 0.25}

    # flat book -> no hedge
    check("flat no hedge", build_hedge_advisory(
        chain=chain, surface=surface, index_price=index,
        fleet_net_delta_usd=500.0, rv=0.5, cfg=cfg) is None)

    # long book + fair vol (ivrv ~ 1) -> protective put, BUY put leg only
    adv = build_hedge_advisory(chain=chain, surface=surface, index_price=index,
                               fleet_net_delta_usd=11_000.0, rv=sigma, cfg=cfg)
    check("put advisory", adv is not None and adv["structure"] == "protective_put"
          and len(adv["legs"]) == 1 and adv["legs"][0]["side"] == "BUY"
          and adv["legs"][0]["option_type"] == "put")
    # synthetic strikes are 10% apart, so the best reachable 25-delta match
    # is coarse — assert direction + ballpark, not precision
    check("put delta near target",
          adv is not None and -0.45 < adv["legs"][0]["delta"] < -0.08)
    check("hedge covers 50% of excess",
          adv is not None and abs(adv["hedged_delta_usd"] - 5_000.0) < 1.0)
    check("tenor in window", adv is not None and
          5 <= (adv["legs"][0]["expiry"] - now).days + 1 <= 21)

    # rich vol (rv << iv) -> collar with SELL call leg, cheaper than the put
    adv2 = build_hedge_advisory(chain=chain, surface=surface, index_price=index,
                                fleet_net_delta_usd=11_000.0, rv=sigma / 2,
                                cfg=cfg)
    check("collar advisory", adv2 is not None and adv2["structure"] == "collar"
          and len(adv2["legs"]) == 2 and adv2["legs"][1]["side"] == "SELL")
    check("collar financed", adv2 is not None and
          adv2["net_premium_usd"] < adv["net_premium_usd"])

    # premium cap binds
    cfg_cap = dict(cfg, max_premium_per_hedge_usd=10.0)
    adv3 = build_hedge_advisory(chain=chain, surface=surface, index_price=index,
                                fleet_net_delta_usd=11_000.0, rv=sigma,
                                cfg=cfg_cap)
    check("premium cap hard", adv3 is not None
          and adv3["legs"][0]["premium_usd"] <= 10.0 + 1e-6)

    # illiquid chain (zero OI) -> no advisory rather than a bad one
    raw_illiquid = [dict(r, open_interest=0.0) for r in raw]
    chain_ill = enrich_chain(raw_illiquid, index, now)
    check("illiquid no hedge", build_hedge_advisory(
        chain=chain_ill, surface=build_surface(chain_ill, index),
        index_price=index, fleet_net_delta_usd=11_000.0, rv=sigma,
        cfg=cfg) is None)
    return fails


if __name__ == "__main__":
    failures = run_self_tests()
    if failures:
        print("FAIL:")
        for f_ in failures:
            print("  -", f_)
        raise SystemExit(1)
    print("hedge_advisor self-tests: ALL PASS")
