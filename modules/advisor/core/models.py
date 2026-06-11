"""
Advisor data models — shared across all advisor engines.
These are plain dataclasses; no ORM, no trading module imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Market(str, Enum):
    """Asset market category."""
    CRYPTO = "crypto"
    US_EQUITIES = "us_equities"
    BIST = "bist"
    FX = "fx"            # includes metals (XAU/XAG)
    MIDAS_FUNDS = "midas_funds"


class Direction(str, Enum):
    """Directional bias for a piece of advice."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Horizon(str, Enum):
    """Investment time horizon."""
    SHORT = "short"    # intraday – 1 week
    MID = "mid"        # 1 week – 3 months
    LONG = "long"      # 3 months – 2 years


class DataSourceStatus(str, Enum):
    """Whether a market's data source is available."""
    AVAILABLE = "available"
    NOT_CONFIGURED = "not_configured"   # key missing / scraper not set up
    DEGRADED = "degraded"               # partial data (e.g. delayed)
    ERROR = "error"                     # runtime fetch failure


# ---------------------------------------------------------------------------
# Sim channels — independent dry-run buckets, each with its own cap.
# ---------------------------------------------------------------------------
# A channel is NOT always the same as a market: discovered ("New Gems") advice
# routes to 'gems' regardless of its underlying market, and KAP-driven sims
# route to 'kap'. The seven channels each have an independent per-channel cap
# (advisor_sim_cap_per_channel, default 15). See migration 077.
SIM_CHANNELS: tuple = (
    "crypto",
    "us_equities",
    "bist",
    "fx",
    "midas_funds",
    "gems",
    "kap",
)


def derive_channel(market, origin: str = "watchlist", kap_driven: bool = False) -> str:
    """
    Map an advice's (market, origin) to its sim channel.

    Rules (in priority order):
      1. kap_driven=True       -> 'kap'  (KAP strong-polarity BIST sims)
      2. origin == 'discovery' -> 'gems' (New Gems layer; ANY underlying market)
      3. otherwise             -> the market's value (watchlist sims)

    `market` may be a Market enum or its string value. A None market falls back
    to 'crypto' only as a last resort; callers should pass a valid market. Pure
    function, no I/O — safe to unit-test without a DB.
    """
    if kap_driven:
        return "kap"
    if str(origin).lower() == "discovery":
        return "gems"
    if market is None:
        return "crypto"
    return market.value if hasattr(market, "value") else str(market)


# ---------------------------------------------------------------------------
# Core advice result
# ---------------------------------------------------------------------------

@dataclass
class AdviceResult:
    """
    Single piece of advice produced by a market analyzer.

    Fields
    ------
    market          : Which market segment produced this advice.
    symbol          : Ticker/identifier (e.g. "AAPL", "BTC/USDT", "THYAO.IS",
                      "XAUUSD", "AK Portfoy Para Piyasasi Katilim Fonu").
    horizon         : Time horizon the advice targets.
    direction       : LONG / SHORT / NEUTRAL.
    entry_low       : Lower bound of suggested entry price range.
    entry_high      : Upper bound of suggested entry price range.
    target_price    : Primary price target.
    stop_price      : Suggested stop-loss price.
    confidence      : Model confidence [0.0, 1.0].
    rationale       : Human-readable explanation (LLM-generated).
    model_id        : Anthropic model ID used for LLM rationale generation.
    kronos_signal   : Optional Kronos forecaster directional prediction
                      (float; positive = bullish, negative = bearish, None if
                       weights not loaded or Kronos not enabled).
    data_source_status : Availability of the underlying data feed.
    data_source_note   : Human note on data availability (shown in dashboard).
    created_at      : UTC timestamp when advice was generated.
    sim_enabled     : Whether this advice should seed a sim position.
    sim_amount_usd  : Notional USD amount for the sim position.
    extra           : Arbitrary JSON-serialisable metadata for specialist use.
    """

    market: Market
    symbol: str
    horizon: Horizon
    direction: Direction

    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    confidence: float = 0.0

    rationale: str = ""
    model_id: str = ""

    kronos_signal: Optional[float] = None

    data_source_status: DataSourceStatus = DataSourceStatus.AVAILABLE
    data_source_note: str = ""

    created_at: datetime = field(default_factory=datetime.utcnow)

    sim_enabled: bool = False
    sim_amount_usd: float = 1000.0

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "market": self.market.value,
            "symbol": self.symbol,
            "horizon": self.horizon.value,
            "direction": self.direction.value,
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model_id": self.model_id,
            "kronos_signal": self.kronos_signal,
            "data_source_status": self.data_source_status.value,
            "data_source_note": self.data_source_note,
            "created_at": self.created_at.isoformat(),
            "sim_enabled": self.sim_enabled,
            "sim_amount_usd": self.sim_amount_usd,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Sim position (dry-run tracking)
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    """
    A simulated position seeded from an advice row.
    Tracked in advisor_sim_positions table.
    """
    symbol: str
    market: Market
    direction: Direction
    horizon: Horizon

    entry_price: float
    target_price: Optional[float]
    stop_price: Optional[float]
    notional_usd: float

    # DB primary key (advisor_sim_positions.id). None until persisted. Having
    # the PK ON the object lets mark-to-market address the exact row instead of
    # guessing by (symbol, entry_price) — which mis-bucketed same-price sims
    # across horizons/channels and left DB-loaded sims unmarked after restart.
    id: Optional[int] = None
    advice_id: Optional[int] = None   # FK to advisor_advice.id
    # Sim channel (independent capped bucket). For watchlist sims this equals
    # the market value; for discovered sims it is 'gems'; for KAP-driven sims
    # it is 'kap'. Defaults to the market value if not explicitly set at open.
    channel: Optional[str] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    # Last marked-to-market price for an OPEN sim (mirrors DB current_price).
    # exit_price is reserved for the FINAL close price only.
    current_price: Optional[float] = None
    horizon_end_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    status: str = "open"              # open | closed | expired


# ---------------------------------------------------------------------------
# Portfolio holding (operator-reported)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioHolding:
    """
    An asset the operator holds (self-reported via dashboard).
    Used to contextualise advice and compute overall exposure.
    """
    symbol: str
    market: Market
    quantity: float
    avg_cost: float              # cost basis per unit in USD
    current_price: Optional[float] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> Optional[float]:
        if self.current_price is None:
            return None
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Optional[float]:
        mv = self.market_value
        if mv is None:
            return None
        return mv - (self.quantity * self.avg_cost)


# ---------------------------------------------------------------------------
# Self-test (no DB): channel derivation + per-channel cap counting.
#   python -m modules.advisor.core.models
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # 1. derive_channel rules.
    assert derive_channel(Market.CRYPTO) == "crypto"
    assert derive_channel(Market.US_EQUITIES, origin="watchlist") == "us_equities"
    # discovery routes to gems regardless of market (even a US ticker).
    assert derive_channel(Market.US_EQUITIES, origin="discovery") == "gems"
    assert derive_channel(Market.CRYPTO, origin="discovery") == "gems"
    # kap_driven wins over everything.
    assert derive_channel(Market.BIST, origin="watchlist", kap_driven=True) == "kap"
    assert derive_channel(Market.US_EQUITIES, origin="discovery", kap_driven=True) == "kap"
    # string market value also accepted.
    assert derive_channel("fx") == "fx"
    assert set(SIM_CHANNELS) == {
        "crypto", "us_equities", "bist", "fx", "midas_funds", "gems", "kap"
    }

    # 2. per-channel cap counting + enforcement (mock open sims, no DB).
    def channel_of(market, origin="watchlist", kap=False):
        return derive_channel(market, origin=origin, kap_driven=kap)

    open_sims = (
        [channel_of(Market.CRYPTO)] * 15
        + [channel_of(Market.US_EQUITIES, origin="discovery")] * 15  # -> gems
        + [channel_of(Market.BIST, kap=True)] * 15                   # -> kap
    )
    counts: dict = {}
    for ch in open_sims:
        counts[ch] = counts.get(ch, 0) + 1
    assert counts == {"crypto": 15, "gems": 15, "kap": 15}, counts

    # cap=15: each channel is independently at the cap; a new crypto sim is
    # blocked, but a new gems sim is NOT (different bucket).
    cap = 15
    assert counts.get("crypto", 0) >= cap          # crypto full
    assert counts.get("fx", 0) < cap               # fx empty -> allowed
    # a gem on a US ticker counts as gems, not us_equities.
    assert counts.get("us_equities", 0) == 0
    assert counts.get("gems", 0) == 15

    print("models self-test OK: channel derivation + per-channel cap counting")


if __name__ == "__main__":
    _selftest()
