"""Pure parsing/normalization for the catalyst_calendar module.

NO network, NO database, NO side effects in this file — every function takes a
raw payload (plus an explicit `now`) and returns a list of normalized event
dicts. That keeps the parsing deterministic and self-testable offline.

Normalized event shape (one dict per catalyst):
    {
        "source":     str,   # 'defillama_unlocks' | 'binance_announcements' |
                             # 'static_macro' | 'fmp_macro'
        "symbol":     str,   # upper-cased token/asset symbol, or '*' = market-wide
        "event_type": str,   # 'token_unlock' | 'exchange_listing' |
                             # 'exchange_delisting' | 'macro_fomc' | 'macro_cpi' |
                             # 'macro_other'
        "event_time": datetime (tz-aware UTC),
        "title":      str,
        "magnitude":  float | None,  # unlocks: fraction of max supply (0..1);
                                     # other sources: None
        "confidence": str,   # 'high' | 'medium' | 'low' — source-trust label
        "url":        str | None,
        "raw":        dict,  # small source-specific extras for the operator
    }

Dedup identity (matches the DB unique key): (source, symbol, event_type,
event_time).

Source-trust honesty:
- DefiLlama emissions is an UNOFFICIAL aggregation of vesting schedules; the
  endpoint schema is undocumented and may change — parsing is defensive and a
  schema change degrades to []. Confidence: medium.
- Binance announcement parsing is title-regex over a CMS endpoint (scrape-class,
  fragile). The announcement timestamp is used as event_time because the actual
  listing go-live time inside the title is not reliably parseable.
  Confidence: low.
- The static macro schedule ships the official published FOMC 2026 decision
  dates only. It EXPIRES at STATIC_MACRO_EXPIRES; past that it emits nothing
  (a stale calendar is worse than none). Confidence: high.
- FMP macro requires a key (gated; degrades to empty without one).
  Confidence: high.

Self-test (offline, fixture-backed — no network):
    python -m modules.catalyst_calendar.core.normalizer
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

SOURCE_DEFILLAMA = "defillama_unlocks"
SOURCE_BINANCE = "binance_announcements"
SOURCE_STATIC_MACRO = "static_macro"
SOURCE_FMP = "fmp_macro"

MARKET_WIDE_SYMBOL = "*"

# Official FOMC 2026 meeting schedule — decision day (second day of each
# meeting), statement at 19:00 UTC (14:00 ET). Operator must refresh this list
# (or set the `macro_static_override` config key) before the expiry below.
STATIC_MACRO_2026: List[Dict[str, str]] = [
    {"date": "2026-01-28T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-03-18T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-04-29T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-06-17T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-07-29T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-09-16T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-10-28T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
    {"date": "2026-12-09T19:00:00", "event_type": "macro_fomc", "title": "FOMC rate decision"},
]
STATIC_MACRO_EXPIRES = date(2026, 12, 31)

_LISTING_RE = re.compile(r"\bwill\s+list\b", re.IGNORECASE)
_DELISTING_RE = re.compile(r"\bwill\s+delist\b", re.IGNORECASE)
_SYMBOL_PAREN_RE = re.compile(r"\(([A-Z0-9]{1,15})\)")


# ───────────────────────── coercion helpers ─────────────────────────

def _as_float(value: Any) -> Optional[float]:
    """Defensive numeric coercion; NaN/inf -> None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _epoch_to_dt(value: Any) -> Optional[datetime]:
    """Epoch seconds OR milliseconds OR ISO string -> tz-aware UTC datetime."""
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    f = _as_float(value)
    if f is None or f <= 0:
        return None
    if f > 1e12:  # milliseconds
        f /= 1000.0
    if f > 4102444800:  # past year 2100 — garbage
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _clean_symbol(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip().upper()
    if s.startswith("COINGECKO:"):
        s = s.split(":", 1)[1].strip().upper()
    s = s[:32]
    return s or None


def _finalize(source: str, symbol: str, event_type: str, event_time: datetime,
              title: str, magnitude: Optional[float], confidence: str,
              url: Optional[str], raw: Optional[dict] = None) -> Dict[str, Any]:
    if magnitude is not None:
        magnitude = min(max(float(magnitude), 0.0), 1.0)
    return {
        "source": source,
        "symbol": symbol,
        "event_type": event_type,
        "event_time": event_time,
        "title": str(title)[:300],
        "magnitude": magnitude,
        "confidence": confidence,
        "url": (str(url)[:500] if url else None),
        "raw": raw or {},
    }


def dedup_key(event: Dict[str, Any]) -> tuple:
    """In-memory dedup identity — mirrors the DB unique constraint."""
    return (event["source"], event["symbol"], event["event_type"], event["event_time"])


# ───────────────────────── DefiLlama unlocks ────────────────────────

def parse_defillama_emissions(payload: Any, *, now: datetime,
                              lookahead_days: int) -> List[Dict[str, Any]]:
    """Normalize the (unofficial) api.llama.fi/emissions payload.

    Best-effort: accepts a list of protocol dicts (or {'protocols': [...]}),
    reads each protocol's future unlock events within the lookahead window.
    magnitude = unlocked tokens / max supply when both are derivable, else None.
    Any unrecognized shape degrades to [] — never raises, never guesses.
    """
    if isinstance(payload, dict):
        payload = payload.get("protocols") or payload.get("data")
    if not isinstance(payload, list):
        return []
    horizon = now + timedelta(days=max(1, int(lookahead_days)))
    out: List[Dict[str, Any]] = []
    seen = set()
    for proto in payload:
        if not isinstance(proto, dict):
            continue
        symbol = (_clean_symbol(proto.get("tokenSymbol"))
                  or _clean_symbol(proto.get("symbol"))
                  or _clean_symbol(proto.get("token"))
                  or _clean_symbol(proto.get("name")))
        if not symbol:
            continue
        max_supply = _as_float(proto.get("maxSupply")) or _as_float(proto.get("totalSupply"))
        events = proto.get("events")
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            nxt = proto.get("nextEvent")
            events = [nxt] if isinstance(nxt, dict) else []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            when = _epoch_to_dt(ev.get("timestamp") if ev.get("timestamp") is not None
                                else ev.get("date"))
            if when is None or when <= now or when > horizon:
                continue
            tokens_raw = ev.get("noOfTokens")
            if isinstance(tokens_raw, list):
                tokens = sum(t for t in (_as_float(x) for x in tokens_raw) if t and t > 0) or None
            else:
                tokens = _as_float(tokens_raw)
            magnitude = None
            if tokens and max_supply and max_supply > 0:
                magnitude = tokens / max_supply
            title = str(ev.get("description") or f"{symbol} token unlock")
            event = _finalize(
                SOURCE_DEFILLAMA, symbol, "token_unlock", when, title, magnitude,
                confidence="medium", url=None,
                raw={"tokens": tokens, "max_supply": max_supply,
                     "protocol": str(proto.get("name") or "")[:100]},
            )
            k = dedup_key(event)
            if k not in seen:
                seen.add(k)
                out.append(event)
    return out


# ───────────────────────── Binance announcements ────────────────────

def parse_binance_announcements(payload: Any, *, now: datetime,
                                recent_window_hours: int = 72) -> List[Dict[str, Any]]:
    """Normalize the Binance CMS announcement payload (scrape-class, fragile).

    Keeps only 'Will List' / 'Will Delist' titles released within the last
    `recent_window_hours` (an announcement IS the catalyst; the go-live time in
    the title is not reliably parseable, so the release timestamp is used).
    One event per (SYMBOL) found in the title; '*' if no symbol parseable.
    Confidence: low — title-regex over an undocumented endpoint.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    articles: List[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("articles"), list):
            articles = data["articles"]
        elif isinstance(data.get("catalogs"), list):
            for cat in data["catalogs"]:
                if isinstance(cat, dict) and isinstance(cat.get("articles"), list):
                    articles.extend(cat["articles"])
    if not articles:
        return []
    floor = now - timedelta(hours=max(1, int(recent_window_hours)))
    out: List[Dict[str, Any]] = []
    seen = set()
    for art in articles:
        if not isinstance(art, dict):
            continue
        title = str(art.get("title") or "").strip()
        if not title:
            continue
        if _DELISTING_RE.search(title):
            event_type = "exchange_delisting"
        elif _LISTING_RE.search(title):
            event_type = "exchange_listing"
        else:
            continue
        when = _epoch_to_dt(art.get("releaseDate"))
        if when is None or when < floor or when > now + timedelta(days=14):
            continue
        code = str(art.get("code") or "").strip()
        url = f"https://www.binance.com/en/support/announcement/{code}" if code else None
        symbols = _SYMBOL_PAREN_RE.findall(title) or [MARKET_WIDE_SYMBOL]
        for sym in symbols[:5]:
            event = _finalize(
                SOURCE_BINANCE, sym, event_type, when, title, None,
                confidence="low", url=url, raw={"exchange": "binance"},
            )
            k = dedup_key(event)
            if k not in seen:
                seen.add(k)
                out.append(event)
    return out


# ───────────────────────── static macro schedule ────────────────────

def parse_static_macro(schedule: Optional[List[Dict[str, str]]] = None, *,
                       now: datetime, lookahead_days: int) -> List[Dict[str, Any]]:
    """Forward macro events from the built-in (or operator-override) schedule.

    Returns [] once `now` is past STATIC_MACRO_EXPIRES when using the built-in
    list — a stale macro calendar must not masquerade as coverage.
    """
    using_builtin = schedule is None
    if using_builtin:
        if now.date() > STATIC_MACRO_EXPIRES:
            return []
        schedule = STATIC_MACRO_2026
    if not isinstance(schedule, list):
        return []
    horizon = now + timedelta(days=max(1, int(lookahead_days)))
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in schedule:
        if not isinstance(item, dict):
            continue
        when = _epoch_to_dt(item.get("date"))
        if when is None or when <= now or when > horizon:
            continue
        event_type = str(item.get("event_type") or "macro_other")
        if event_type not in ("macro_fomc", "macro_cpi", "macro_other"):
            event_type = "macro_other"
        event = _finalize(
            SOURCE_STATIC_MACRO, MARKET_WIDE_SYMBOL, event_type, when,
            str(item.get("title") or event_type), None,
            confidence="high", url=None,
            raw={"schedule": "builtin" if using_builtin else "override"},
        )
        k = dedup_key(event)
        if k not in seen:
            seen.add(k)
            out.append(event)
    return out


def static_macro_is_stale(now: datetime) -> bool:
    """True when the built-in macro schedule has expired (operator must refresh)."""
    return now.date() > STATIC_MACRO_EXPIRES


# ───────────────────────── FMP macro (key-gated) ────────────────────

def parse_fmp_macro(payload: Any, *, now: datetime,
                    lookahead_days: int) -> List[Dict[str, Any]]:
    """Normalize an FMP economic-calendar payload (key-gated upstream).

    Keeps only High-impact US events. FMP dates are 'YYYY-MM-DD HH:MM:SS'
    (documented as UTC). event_type derived from the event name.
    """
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        return []
    horizon = now + timedelta(days=max(1, int(lookahead_days)))
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        impact = str(item.get("impact") or "").strip().lower()
        country = str(item.get("country") or "").strip().upper()
        if impact != "high" or country not in ("US", "USA", "UNITED STATES"):
            continue
        when = _epoch_to_dt(str(item.get("date") or "").replace(" ", "T"))
        if when is None or when <= now or when > horizon:
            continue
        name = str(item.get("event") or "macro event")
        low = name.lower()
        if "cpi" in low or "consumer price" in low:
            event_type = "macro_cpi"
        elif "fomc" in low or "fed interest rate" in low or "federal funds" in low:
            event_type = "macro_fomc"
        else:
            event_type = "macro_other"
        event = _finalize(
            SOURCE_FMP, MARKET_WIDE_SYMBOL, event_type, when, name, None,
            confidence="high", url=None, raw={"country": country},
        )
        k = dedup_key(event)
        if k not in seen:
            seen.add(k)
            out.append(event)
    return out


# ───────────────────────── offline self-test ────────────────────────

def _self_test() -> int:
    import json
    from pathlib import Path

    fixture_path = (Path(__file__).parent.parent / "fixtures"
                    / "catalyst_sources_fixture.json")
    fx = json.loads(fixture_path.read_text())
    now = datetime(2026, 6, 12, 0, 0, 0, tzinfo=timezone.utc)

    # DefiLlama: 1 future in-window event; past + beyond-horizon + broken skipped.
    unlocks = parse_defillama_emissions(fx["defillama_emissions"], now=now,
                                        lookahead_days=30)
    assert len(unlocks) == 1, f"expected 1 unlock, got {len(unlocks)}"
    u = unlocks[0]
    assert u["symbol"] == "TPT" and u["event_type"] == "token_unlock"
    assert u["magnitude"] is not None and abs(u["magnitude"] - 0.05) < 1e-9
    assert u["confidence"] == "medium" and u["event_time"] > now

    # Binance: listing + delisting kept (recent), maintenance + stale skipped.
    anns = parse_binance_announcements(fx["binance_announcements"], now=now,
                                       recent_window_hours=72)
    assert len(anns) == 2, f"expected 2 announcements, got {len(anns)}"
    by_type = {a["event_type"]: a for a in anns}
    assert by_type["exchange_listing"]["symbol"] == "TPT"
    assert by_type["exchange_delisting"]["symbol"] == "DEAD"
    assert all(a["confidence"] == "low" and a["url"] for a in anns)

    # Static macro: only the 2026-06-17 FOMC is inside a 30d window from Jun 12.
    macro = parse_static_macro(now=now, lookahead_days=30)
    assert len(macro) == 1 and macro[0]["event_type"] == "macro_fomc"
    assert macro[0]["symbol"] == MARKET_WIDE_SYMBOL
    assert macro[0]["event_time"] == datetime(2026, 6, 17, 19, 0,
                                              tzinfo=timezone.utc)
    # Expiry honesty: a stale built-in schedule emits NOTHING.
    stale_now = datetime(2027, 2, 1, tzinfo=timezone.utc)
    assert parse_static_macro(now=stale_now, lookahead_days=30) == []
    assert static_macro_is_stale(stale_now) and not static_macro_is_stale(now)

    # FMP: High-impact US kept; Low impact + non-US skipped.
    fmp = parse_fmp_macro(fx["fmp_macro"], now=now, lookahead_days=30)
    assert len(fmp) == 1 and fmp[0]["event_type"] == "macro_cpi"
    assert fmp[0]["confidence"] == "high"

    # Malformed payloads degrade to [] — never raise.
    assert parse_defillama_emissions("garbage", now=now, lookahead_days=30) == []
    assert parse_defillama_emissions({"unexpected": True}, now=now, lookahead_days=30) == []
    assert parse_binance_announcements([1, 2], now=now) == []
    assert parse_binance_announcements({"data": {"catalogs": "nope"}}, now=now) == []
    assert parse_fmp_macro(None, now=now, lookahead_days=30) == []
    assert parse_static_macro("broken", now=now, lookahead_days=30) == []

    # Dedup identity is stable.
    assert dedup_key(u) == (SOURCE_DEFILLAMA, "TPT", "token_unlock", u["event_time"])

    print("normalizer self-test OK "
          f"(unlocks={len(unlocks)}, announcements={len(anns)}, "
          f"macro={len(macro)}, fmp={len(fmp)}; malformed inputs -> [])")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
