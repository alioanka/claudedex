"""Pure normalization + dedup logic for the market-data warehouse.

No I/O, no imports beyond stdlib — every function here is deterministic and
self-tested (run: python -m modules.market_data_warehouse.core.normalizer).

Canonical forms
  symbol     'BASE/QUOTE' uppercase, e.g. 'BTC/USDT' (exchange-agnostic)
  timeframe  one of TIMEFRAME_SECONDS keys ('1m','5m','15m','30m','1h','4h','1d')
  ts         epoch SECONDS (int), floored to the timeframe boundary
Dedup key
  candles: (source, symbol, timeframe, ts)   series: (source, symbol, metric, ts)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Canonical timeframes -> seconds. Single source of truth.
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Aliases seen across Binance ('1m','1h'), Bybit v5 ('1','60','D'), ccxt.
_TIMEFRAME_ALIASES: Dict[str, str] = {
    "1": "1m", "1min": "1m", "60s": "1m",
    "5": "5m", "5min": "5m",
    "15": "15m", "15min": "15m",
    "30": "30m", "30min": "30m",
    "60": "1h", "60m": "1h", "1hr": "1h",
    "240": "4h", "240m": "4h",
    "d": "1d", "1day": "1d", "24h": "1d", "1440": "1d",
}

# Quote currencies recognized when splitting a concatenated pair like
# 'BTCUSDT'. Longest-first so 'USDT' wins over 'USD'.
_KNOWN_QUOTES: Tuple[str, ...] = (
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI",
    "USD", "EUR", "TRY", "BTC", "ETH", "SOL", "BNB",
)


def normalize_timeframe(raw: str) -> Optional[str]:
    """Any known timeframe spelling -> canonical key, else None."""
    if not raw:
        return None
    tf = str(raw).strip().lower()
    if tf in TIMEFRAME_SECONDS:
        return tf
    return _TIMEFRAME_ALIASES.get(tf)


def timeframe_seconds(timeframe: str) -> Optional[int]:
    tf = normalize_timeframe(timeframe)
    return TIMEFRAME_SECONDS.get(tf) if tf else None


def floor_ts(ts_seconds: float, timeframe: str) -> Optional[int]:
    """Floor an epoch-seconds timestamp to the timeframe boundary."""
    secs = timeframe_seconds(timeframe)
    if secs is None or ts_seconds is None or ts_seconds <= 0:
        return None
    return int(ts_seconds // secs) * secs


def to_epoch_seconds(ts) -> Optional[int]:
    """Accept epoch seconds OR milliseconds (int/float/str) -> epoch seconds.
    Heuristic: values >= 1e12 are milliseconds (1e12 s ~= year 33658)."""
    try:
        v = float(ts)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    if v >= 1e12:
        v = v / 1000.0
    return int(v)


def normalize_symbol(raw: str) -> Optional[str]:
    """Any common pair spelling -> canonical 'BASE/QUOTE', else None.

    Accepts 'BTC/USDT', 'btc-usdt', 'BTC_USDT', 'BTCUSDT', 'BTC/USDT:USDT'
    (ccxt perp suffix is dropped)."""
    if not raw:
        return None
    s = str(raw).strip().upper()
    if ":" in s:  # ccxt linear-perp suffix, e.g. 'BTC/USDT:USDT'
        s = s.split(":", 1)[0]
    for sep in ("/", "-", "_"):
        if sep in s:
            parts = [p for p in s.split(sep) if p]
            if len(parts) == 2 and parts[0] and parts[1]:
                return f"{parts[0]}/{parts[1]}"
            return None
    # Concatenated form: split on a known quote suffix (longest-first order).
    for q in _KNOWN_QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return f"{s[:-len(q)]}/{q}"
    return None


def to_exchange_symbol(symbol: str, source: str = "") -> Optional[str]:
    """Canonical 'BASE/QUOTE' -> concatenated 'BASEQUOTE' (Binance/Bybit REST
    spelling). Unknown/invalid input -> None."""
    canon = normalize_symbol(symbol)
    if canon is None:
        return None
    return canon.replace("/", "")


@dataclass(frozen=True)
class Candle:
    source: str
    symbol: str          # canonical 'BASE/QUOTE'
    timeframe: str       # canonical
    ts: int              # epoch seconds, floored to timeframe boundary
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return (self.source, self.symbol, self.timeframe, self.ts)


@dataclass(frozen=True)
class SeriesPoint:
    source: str
    symbol: str          # canonical 'BASE/QUOTE'
    metric: str          # e.g. 'funding_rate', 'mark_price', 'open_interest'
    ts: int              # epoch seconds
    value: float

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return (self.source, self.symbol, self.metric, self.ts)


def _finite_positive(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in vals)


def make_candle(source: str, symbol: str, timeframe: str, ts,
                o, h, l, c, v, quote_volume=None) -> Optional[Candle]:
    """Validate + canonicalize one raw candle. Returns None (drop) when any
    field is unusable — fail-soft normalization, never raises on bad rows."""
    canon_sym = normalize_symbol(symbol)
    canon_tf = normalize_timeframe(timeframe)
    secs = to_epoch_seconds(ts)
    if not source or canon_sym is None or canon_tf is None or secs is None:
        return None
    floored = floor_ts(secs, canon_tf)
    if floored is None:
        return None
    try:
        o, h, l, c = float(o), float(h), float(l), float(c)
        v = float(v) if v is not None else 0.0
        qv = float(quote_volume) if quote_volume is not None else None
    except (TypeError, ValueError):
        return None
    if not _finite_positive(o, h, l, c):
        return None
    if h < l or v < 0 or not math.isfinite(v):
        return None
    if not (l <= o <= h and l <= c <= h):
        return None
    if qv is not None and (not math.isfinite(qv) or qv < 0):
        qv = None
    return Candle(str(source), canon_sym, canon_tf, floored, o, h, l, c, v, qv)


def make_series_point(source: str, symbol: str, metric: str, ts, value) -> Optional[SeriesPoint]:
    """Validate + canonicalize one scalar series point (funding/vol/price)."""
    canon_sym = normalize_symbol(symbol)
    secs = to_epoch_seconds(ts)
    if not source or not metric or canon_sym is None or secs is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return SeriesPoint(str(source), canon_sym, str(metric), secs, val)


def dedup_candles(candles: Iterable[Optional[Candle]]) -> List[Candle]:
    """Drop Nones, dedup on (source,symbol,timeframe,ts) keeping the LAST
    occurrence (later fetches refresh the in-progress candle), sort by ts."""
    seen: Dict[Tuple[str, str, str, int], Candle] = {}
    for c in candles:
        if c is not None:
            seen[c.key] = c
    return sorted(seen.values(), key=lambda c: (c.symbol, c.timeframe, c.ts))


def dedup_series(points: Iterable[Optional[SeriesPoint]]) -> List[SeriesPoint]:
    """Same contract as dedup_candles for scalar series points."""
    seen: Dict[Tuple[str, str, str, int], SeriesPoint] = {}
    for p in points:
        if p is not None:
            seen[p.key] = p
    return sorted(seen.values(), key=lambda p: (p.symbol, p.metric, p.ts))


# ───────────────────────────── self-test ────────────────────────────────────

def _self_test() -> None:
    # timeframe normalization
    assert normalize_timeframe("1m") == "1m"
    assert normalize_timeframe("60") == "1h"
    assert normalize_timeframe("D") == "1d"
    assert normalize_timeframe("240") == "4h"
    assert normalize_timeframe("7w") is None
    assert timeframe_seconds("1h") == 3600

    # ts handling: ms + s + flooring
    assert to_epoch_seconds(1718000000) == 1718000000
    assert to_epoch_seconds(1718000000123) == 1718000000
    assert to_epoch_seconds("not-a-ts") is None
    assert floor_ts(1718000000, "1h") == 1718000000 // 3600 * 3600
    assert floor_ts(-5, "1m") is None

    # symbol normalization
    assert normalize_symbol("BTCUSDT") == "BTC/USDT"
    assert normalize_symbol("btc/usdt") == "BTC/USDT"
    assert normalize_symbol("SOL-USDC") == "SOL/USDC"
    assert normalize_symbol("ETH/USDT:USDT") == "ETH/USDT"
    assert normalize_symbol("1000PEPEUSDT") == "1000PEPE/USDT"
    assert normalize_symbol("") is None
    assert normalize_symbol("XYZ") is None
    assert to_exchange_symbol("BTC/USDT") == "BTCUSDT"

    # candle validation
    good = make_candle("binance", "BTCUSDT", "60", 1718000000123,
                       "100", "110", "90", "105", "12.5", "1300")
    assert good is not None
    assert good.symbol == "BTC/USDT" and good.timeframe == "1h"
    assert good.ts % 3600 == 0
    assert good.quote_volume == 1300.0
    assert make_candle("binance", "BTCUSDT", "1h", 1718000000,
                       100, 90, 110, 105, 1) is None          # high < low
    assert make_candle("binance", "BTCUSDT", "1h", 1718000000,
                       100, 110, 90, 0, 1) is None            # close <= 0
    assert make_candle("binance", "BTCUSDT", "1h", 1718000000,
                       100, 110, 90, 120, 1) is None          # close > high
    assert make_candle("binance", "BTCUSDT", "1h", 1718000000,
                       "oops", 110, 90, 105, 1) is None       # non-numeric
    assert make_candle("binance", "???", "1h", 1718000000,
                       100, 110, 90, 105, 1) is None          # bad symbol
    nan = float("nan")
    assert make_candle("binance", "BTCUSDT", "1h", 1718000000,
                       nan, 110, 90, 105, 1) is None          # NaN

    # series validation (funding can be negative — only NaN/inf rejected)
    fp = make_series_point("bybit", "BTCUSDT", "funding_rate",
                           1718000000000, "-0.0001")
    assert fp is not None and fp.value == -0.0001 and fp.symbol == "BTC/USDT"
    assert make_series_point("bybit", "BTCUSDT", "funding_rate",
                             1718000000, float("inf")) is None

    # dedup: last occurrence wins, Nones dropped, sorted output
    base = dict(source="binance", symbol="BTC/USDT", timeframe="1h",
                ts=1718000000 // 3600 * 3600,
                open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
    c1 = Candle(**base)
    c2 = Candle(**{**base, "close": 1.9})                      # same key, newer
    c3 = Candle(**{**base, "ts": base["ts"] - 3600})           # earlier candle
    out = dedup_candles([c3, None, c1, c2])
    assert len(out) == 2
    assert out[0].ts < out[1].ts
    assert out[1].close == 1.9                                  # last won

    p1 = SeriesPoint("bybit", "BTC/USDT", "funding_rate", 1000, 0.0001)
    p2 = SeriesPoint("bybit", "BTC/USDT", "funding_rate", 1000, 0.0002)
    assert dedup_series([p1, p2, None])[0].value == 0.0002

    print("normalizer self-test OK")


if __name__ == "__main__":
    _self_test()
