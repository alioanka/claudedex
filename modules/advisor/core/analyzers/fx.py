"""
FXAnalyzer — FX pairs and metals (gold XAU/USD, silver XAG/USD) analyzer.

Data source options (ordered by recommendation):
  1. Alpha Vantage — free tier: 25 requests/day, 500/month.
     Covers major FX pairs + precious metals.
     Key needed: ADVISOR_FX_ALPHAVANTAGE_KEY (operator-provided, free signup).
     URL: https://www.alphavantage.co/support/#api-key

  2. ExchangeRate-API (exchangerate-api.com) — free tier 1500 req/month.
     FX pairs only; metals NOT covered on free tier.
     Key needed: ADVISOR_FX_EXCHANGERATE_KEY.

  3. Stooq.com — free, no key, but no SLA. CSV endpoint:
     https://stooq.com/q/d/l/?s=eurusd&i=d
     Metals available as XAU (gold), XAG (silver) with .US suffix.
     Coverage: major pairs + metals. No auth required.
     Risk: no SLA, may go down.

  4. yfinance — covers major FX via "EURUSD=X" format and metals
     via "GC=F" (gold futures), "SI=F" (silver futures) or "GLD" ETF.
     Free, no key. Data may be delayed ~15 min for futures.

DEFAULT: use yfinance (free, no key). Operator may override to
Alpha Vantage for more reliable / real-time data.

Required operator config (advisor_config DB):
  advisor_fx_data_source     : "yfinance" (default) | "alphavantage" | "stooq"
  ADVISOR_FX_ALPHAVANTAGE_KEY: Secure Credentials (only if alphavantage selected)

Watchlist format (watchlist_fx config key):
  yfinance:     "EURUSD=X,GBPUSD=X,XAUUSD=X,XAGUSD=X"
  alphavantage: "EUR/USD,GBP/USD,XAU/USD"
  metals only:  "XAU/USD,XAG/USD" (always normalised internally)

STUB STATUS (Wave-20): scaffold only. Data fetch and signal computation
deferred to Wave-21 (quant-algo agent).
"""

from __future__ import annotations

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import AdviceResult, DataSourceStatus, Horizon, Market


_NOT_CONFIGURED_NOTE = (
    "FX data source not configured. "
    "Recommended free option: set advisor_fx_data_source='yfinance' in advisor_config "
    "(no key needed; covers major pairs + metals via GC=F / SI=F). "
    "For real-time data: set advisor_fx_data_source='alphavantage' and add "
    "ADVISOR_FX_ALPHAVANTAGE_KEY to Secure Credentials (free signup at alphavantage.co). "
    "Watchlist key: watchlist_fx (e.g. 'EURUSD=X,XAUUSD=X')."
)


class FXAnalyzer(BaseAnalyzer):
    """
    FX pairs and metals (XAU/USD, XAG/USD) analyzer.

    STUB — implement in Wave-21 (quant-algo agent).
    See module docstring for data source options and required keys.
    """

    market = Market.FX

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_fx_data_source", "")
        if source == "yfinance":
            try:
                import yfinance  # noqa: F401
                return DataSourceStatus.AVAILABLE
            except ImportError:
                return DataSourceStatus.NOT_CONFIGURED
        if source == "alphavantage":
            key = self.config.get("advisor_fx_alphavantage_key") or ""
            return DataSourceStatus.AVAILABLE if key else DataSourceStatus.NOT_CONFIGURED
        if source == "stooq":
            return DataSourceStatus.DEGRADED  # no SLA
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        STUB. Wave-21 quant agent: implement FX/metals data fetch.

        Steps:
          1. Check advisor_fx_data_source.
          2a. yfinance: Ticker(symbol).history(...) — works for EURUSD=X, GC=F.
          2b. alphavantage: GET /query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD
          2c. stooq: fetch CSV from stooq.com URL.
          3. Compute SMA/RSI/BB (same pipeline as us_equities).
          4. Return AdviceResult with appropriate DataSourceStatus.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)
        return self._not_configured(
            symbol, horizon,
            "FXAnalyzer data fetch not yet implemented (Wave-21 task)."
        )
