"""
MidasFundsAnalyzer — Turkish mutual fund (fon) analyzer for Midas exchange.

Target funds (examples):
  - Tera Portfoy Para Piyasasi Fonu
  - Ak Portfoy Para Piyasasi Katilim Fonu
  - Various AI-themed Turkish funds (e.g. Fidelity/AK AI fonu)

Data source: NO FREE CLEAN API EXISTS as of 2026-06-02.

Options and their status:
  1. Tefas.gov.tr (Turkish Electronic Fund Distribution Platform):
     - Official source for all Turkish mutual fund NAV data.
     - Has a public-facing website but NO documented REST API.
     - Can be scraped: POST to https://www.tefas.gov.tr/api/DB/BindHistoryInfo
       with form data (FONKODU, BASTARIH, BITTARIH).
     - Risk: undocumented endpoint, may change at any time.
     - Key needed: none (POST with CSRF token).
     - Implementation: requests session with cookie + CSRF extraction.

  2. Midas Exchange (midasplatform.com.tr):
     - No public API documented.
     - Fund prices visible on website; scraping required.
     - Key needed: none (public pricing) or ADVISOR_MIDAS_SESSION_COOKIE.

  3. Fintables.com / Fonanaliz.com:
     - Third-party aggregators with some fund data.
     - No official API; HTML scraping only.

  4. Manual entry via advisor dashboard:
     - Operator enters NAV manually each day.
     - Stored in advisor_portfolio table as "holdings".
     - This is the SAFEST fallback; technical analysis limited (no OHLCV).

DEFAULT BEHAVIOR: returns NOT_CONFIGURED with instructions.
When advisor_midas_data_source='tefas_scrape', attempts the Tefas.gov.tr
scrape (undocumented POST endpoint — fragile but free).

Required operator config (advisor_config DB):
  advisor_midas_data_source : "manual" | "tefas_scrape" (default "manual")
  watchlist_midas_funds     : comma-sep fund codes (Tefas FONKODU format)
                              e.g. "TPP,AKP,MAC" (Tera Para Piyasasi, AK
                              Para Piyasasi Katilim, etc.)

For manual entry:
  No key needed. Operator enters current NAV via /advisor/portfolio page.
  Technical analysis not possible without historical NAV series.

For tefas_scrape:
  No key needed. The scraper must handle CSRF + session cookie rotation.
  Implementation: backend/data agent, Wave-21.

STUB STATUS (Wave-20): scaffold only. Both paths deferred to Wave-21.
"""

from __future__ import annotations

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import AdviceResult, DataSourceStatus, Horizon, Market


_NOT_CONFIGURED_NOTE = (
    "Midas Funds data source not configured. "
    "No free clean API exists for Turkish mutual funds (2026-06-02). "
    "Options: "
    "(1) Set advisor_midas_data_source='tefas_scrape' — scrapes tefas.gov.tr "
    "for historical NAV (undocumented POST endpoint, fragile but free). "
    "(2) Set advisor_midas_data_source='manual' — operator enters NAV manually "
    "via the /advisor/portfolio dashboard page. "
    "Fund codes (watchlist_midas_funds): use Tefas FONKODU format "
    "(e.g. 'TPP' for Tera Portfoy Para Piyasasi Fonu, "
    "'AKP' for AK Portfoy Para Piyasasi Katilim Fonu). "
    "Look up FONKODU at https://www.tefas.gov.tr."
)


class MidasFundsAnalyzer(BaseAnalyzer):
    """
    Turkish mutual fund (fon) analyzer for Midas exchange.

    STUB — implement in Wave-21 (backend/data agent).
    See module docstring for data source options and setup instructions.
    """

    market = Market.MIDAS_FUNDS

    # Known Tefas fund codes → human-readable names (expand as needed)
    KNOWN_FUNDS = {
        "TPP": "Tera Portfoy Para Piyasasi Fonu",
        "AKP": "Ak Portfoy Para Piyasasi Katilim Fonu",
        "MAC": "Midas AI Cathie Wood Fonu",  # placeholder — verify code
    }

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_midas_data_source", "")
        if source == "manual":
            return DataSourceStatus.DEGRADED   # no OHLCV, analysis limited
        if source == "tefas_scrape":
            return DataSourceStatus.DEGRADED   # works but fragile/undocumented
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        STUB. Wave-21 data agent: implement Tefas scrape or manual-entry path.

        Steps for tefas_scrape path:
          1. POST to https://www.tefas.gov.tr/api/DB/BindHistoryInfo with
             FONKODU=symbol, BASTARIH=<start>, BITTARIH=<today>.
             Extract CSRF token from session cookie first.
          2. Parse JSON response: list of {TARIH, FIYAT} (date, NAV).
          3. Build pandas DataFrame with close=FIYAT.
          4. Compute SMA/RSI (no OHLCV → single-price series; skip BB/vol).
          5. Return AdviceResult with DataSourceStatus.DEGRADED.

        Steps for manual path:
          1. Query advisor_portfolio table for holdings with market='midas_funds'.
          2. Return NEUTRAL advice with current_value from the last manual entry.
          3. Rationale: "Manual NAV entry; no technical analysis available."
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)
        return self._not_configured(
            symbol, horizon,
            "MidasFundsAnalyzer data fetch not yet implemented (Wave-21 task)."
        )
