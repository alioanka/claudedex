"""
BISTAnalyzer — Borsa Istanbul equities analyzer.

Data source: NO FREE API EXISTS as of 2026-06-02.

Options and their status:
  1. investing.com/tr — HTML scraping only; no official API.
     Requires: requests + BeautifulSoup + regular CSS selector maintenance.
     Risk: site may change structure or block scrapers without notice.
     Key needed: none (but fragile; not production-grade).

  2. IsYatirim / Matriks / Rasyonet — Turkish financial data vendors.
     Require paid subscription + API token.
     Key needed: ADVISOR_BIST_API_KEY (operator-provided).

  3. Yahoo Finance (yfinance) with .IS suffix — PARTIAL.
     Some BIST stocks are available (e.g. "THYAO.IS"), but:
     - Coverage is incomplete (mid/small caps often missing).
     - Data may be delayed 15-20 minutes.
     - Mutual funds (Midas Funds) are NOT covered.
     Key needed: none (yfinance).

DEFAULT BEHAVIOR: this analyzer checks for ADVISOR_BIST_DATA_SOURCE in
advisor_config. If set to "yfinance", it uses yfinance with .IS suffix
as a degraded-but-free option. Otherwise returns NOT_CONFIGURED.

Required operator actions:
  - Set advisor_config key advisor_bist_data_source = "yfinance" for the
    free/degraded path (incomplete coverage, no funds).
  - OR set advisor_bist_data_source = "matriks" and provide
    ADVISOR_BIST_API_KEY in Secure Credentials for production coverage.
  - Watchlist: watchlist_bist = "THYAO.IS,EREGL.IS,GARAN.IS" (yfinance format)
    OR "THYAO,EREGL,GARAN" (Matriks format).

STUB STATUS (Wave-20): scaffold only. Data wiring deferred to Wave-21
(backend/data agent). Both the yfinance-degraded path and the Matriks
paid path need implementation.
"""

from __future__ import annotations

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import AdviceResult, DataSourceStatus, Horizon, Market


_NOT_CONFIGURED_NOTE = (
    "BIST data source not configured. "
    "Set advisor_bist_data_source='yfinance' in advisor_config for degraded "
    "free coverage (incomplete, delayed). "
    "For production: set advisor_bist_data_source='matriks' and provide "
    "ADVISOR_BIST_API_KEY in Secure Credentials (Settings > Credentials). "
    "Paid providers: Matriks (matriks.com), Rasyonet, IsYatirim."
)


class BISTAnalyzer(BaseAnalyzer):
    """
    Borsa Istanbul analyzer.

    STUB — implement in Wave-21 (backend/data agent).
    See module docstring for data source options and required keys.
    """

    market = Market.BIST

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_bist_data_source", "")
        if source == "yfinance":
            try:
                import yfinance  # noqa: F401
                return DataSourceStatus.DEGRADED  # partial coverage
            except ImportError:
                return DataSourceStatus.NOT_CONFIGURED
        if source == "matriks":
            api_key = self.config.get("advisor_bist_api_key") or ""
            return DataSourceStatus.AVAILABLE if api_key else DataSourceStatus.NOT_CONFIGURED
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        STUB. Wave-21 data agent: implement BIST data fetch + technicals.

        Steps to implement:
          1. Check advisor_bist_data_source config key.
          2a. If 'yfinance': fetch via yfinance Ticker(symbol + ".IS") if .IS
              not already appended. Mirror us_equities._fetch_and_compute.
          2b. If 'matriks': call Matriks REST API with ADVISOR_BIST_API_KEY.
          3. Compute SMA/RSI/BB (same pipeline as us_equities).
          4. Return AdviceResult with DataSourceStatus.DEGRADED (yfinance)
             or AVAILABLE (Matriks).
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)
        # Degraded or Available but not yet implemented
        return self._not_configured(
            symbol, horizon,
            "BISTAnalyzer data fetch not yet implemented (Wave-21 task)."
        )
