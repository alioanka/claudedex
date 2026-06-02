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

    advice_id: Optional[int] = None   # FK to advisor_advice.id
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
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
