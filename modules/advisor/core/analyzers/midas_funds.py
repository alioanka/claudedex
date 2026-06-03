"""
MidasFundsAnalyzer -- Turkish mutual fund (fon) analyzer for Midas exchange.

Target funds (examples):
  - Tera Portfoy Para Piyasasi Fonu (TPP)
  - Ak Portfoy Para Piyasasi Katilim Fonu (AKP)
  - Other TEFAS-listed funds as FONKODU codes

Data source priority (2026-06-02):

  1. tefas-crawler (PRIMARY) -- AVAILABLE
     Python library adapted to the 2026 tefas.gov.tr/api/funds/ JSON API.
     Per-fund NAV history up to 5 years, documented endpoint.
     Install: pip install tefas-crawler
     Import-guarded: if not installed, falls through to tefasfon/scrape fallback.
     advisor_midas_data_source='tefas_crawler'  -- explicit
     advisor_midas_data_source=''               -- auto-detect: tefas-crawler first
     Watchlist: watchlist_midas_funds='TPP,AKP,MAC' (FONKODU format)
     data_source_status = AVAILABLE when lib importable and returns data.

  2. tefasfon (SECONDARY FALLBACK) -- DEGRADED
     Alternative lightweight Python wrapper for tefas.gov.tr.
     Install: pip install tefasfon
     Falls back automatically when tefas-crawler is absent/fails.
     data_source_status = DEGRADED.

  3. tefas.gov.tr POST scrape (LEGACY FALLBACK) -- DEGRADED
     Undocumented POST API. Fragile (no SLA, CSRF dependent).
     advisor_midas_data_source='tefas_scrape'  -- explicit legacy mode
     data_source_status = DEGRADED.

  4. manual entry -- DEGRADED
     Operator enters current NAV via /advisor/portfolio dashboard page.
     No historical series -- no technicals, NEUTRAL direction only.
     advisor_midas_data_source='manual'
     data_source_status = DEGRADED.

data_source_status resolution:
  tefas-crawler importable (any config incl. empty)           -> AVAILABLE
  tefas-crawler absent, tefasfon importable                   -> DEGRADED
  tefas_scrape / manual explicit                              -> DEGRADED
  nothing importable, no config                               -> NOT_CONFIGURED

Technical analysis:
  tefas-crawler / tefasfon / tefas_scrape: SMA + RSI only
    (single-price NAV series; no OHLCV, no volume).
  manual: NEUTRAL direction, no technical signals possible.
  Bollinger Bands omitted -- requires O/H/L which funds don't have.
  Confidence capped at 0.8 (no OHLCV validation available).

LLM rationale: shared build_rationale() -- works with partial signal dict.

Import guards: tefas-crawler / tefasfon absent -> NEVER crashes the advice loop.

ADVICE-ONLY. analyze() never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import (
    AdviceResult,
    DataSourceStatus,
    Direction,
    Horizon,
    Market,
)
from modules.advisor.core.rationale_helper import build_rationale
from modules.advisor.core.analyzers.levels import (
    horizon_levels,
    signal_confidence,
)

logger = logging.getLogger("advisor.analyzer.midas_funds")

_LOOKBACK: dict = {
    Horizon.SHORT: 30,
    Horizon.MID: 180,
    Horizon.LONG: 730,
}

_NOT_CONFIGURED_NOTE = (
    "Midas Funds data source not configured. "
    "Recommended (free, AVAILABLE): pip install tefas-crawler  then leave "
    "advisor_midas_data_source empty or set to 'tefas_crawler'. "
    "Watchlist: watchlist_midas_funds='TPP,AKP,MAC' (Tefas FONKODU format). "
    "Fallback (DEGRADED): pip install tefasfon (alternative wrapper). "
    "Legacy fallback: set advisor_midas_data_source='tefas_scrape' "
    "(undocumented POST endpoint, fragile). "
    "Manual fallback: set advisor_midas_data_source='manual' and enter "
    "NAV via /advisor/portfolio dashboard page. "
    "Look up FONKODU at https://www.tefas.gov.tr."
)

# Tefas legacy scrape endpoint (fallback only)
_TEFAS_BASE = "https://www.tefas.gov.tr"
_TEFAS_HISTORY_API = f"{_TEFAS_BASE}/api/DB/BindHistoryInfo"
_TEFAS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{_TEFAS_BASE}/TarihselVeriler.aspx",
}


def _tefas_crawler_available() -> bool:
    """Return True if tefas-crawler (package name: tefas) can be imported."""
    try:
        import tefas  # noqa: F401  (tefas-crawler installs as 'tefas')
        return True
    except ImportError:
        return False


def _tefasfon_available() -> bool:
    """Return True if tefasfon can be imported."""
    try:
        import tefasfon  # noqa: F401
        return True
    except ImportError:
        return False


class MidasFundsAnalyzer(BaseAnalyzer):
    """
    Turkish mutual fund (fon) analyzer for Midas exchange.

    Primary   : tefas-crawler (2026 API, up to 5y history)     -> AVAILABLE
    Secondary : tefasfon (alternative wrapper)                   -> DEGRADED
    Legacy    : tefas_scrape (undocumented POST, fragile)        -> DEGRADED
    Manual    : operator-entered NAV, no historical series       -> DEGRADED
    None      : NOT_CONFIGURED
    """

    market = Market.MIDAS_FUNDS

    # Known Tefas fund codes -> human-readable names (expand as needed)
    KNOWN_FUNDS = {
        "TPP": "Tera Portfoy Para Piyasasi Fonu",
        "AKP": "Ak Portfoy Para Piyasasi Katilim Fonu",
        "MAC": "Midas AI Cathie Wood Fonu",   # verify FONKODU on tefas.gov.tr
        "TI2": "Tacirler Portfoy BIST-100 Endeks Fonu",
        "TTE": "Tacirler Portfoy Teknoloji Yabanci HISSE Fonu",
    }

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self.last_price: dict = {}

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_midas_data_source", "")

        # Auto-detect: tefas-crawler (AVAILABLE) preferred
        if source in ("tefas_crawler", "") and _tefas_crawler_available():
            return DataSourceStatus.AVAILABLE

        # Auto-detect: tefasfon (DEGRADED) second choice
        if source in ("tefasfon", "") and _tefasfon_available():
            return DataSourceStatus.DEGRADED

        # Explicit legacy/manual paths
        if source in ("tefas_scrape", "manual"):
            return DataSourceStatus.DEGRADED

        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a Turkish mutual fund (FONKODU).
        Never raises -- _error_result or _not_configured on any failure.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)

        source = self.config.get("advisor_midas_data_source", "")
        try:
            # Explicit manual path
            if source == "manual":
                return await self._analyze_manual(symbol, horizon)

            # Explicit legacy scrape
            if source == "tefas_scrape":
                return await self._analyze_tefas_scrape(symbol, horizon)

            # tefas-crawler (AVAILABLE) -- primary
            if _tefas_crawler_available() and source in ("tefas_crawler", ""):
                result = await self._analyze_tefas_crawler(symbol, horizon)
                if result.data_source_status != DataSourceStatus.NOT_CONFIGURED:
                    return result
                self.logger.warning(
                    "[midas_funds] tefas-crawler returned no data for %s. "
                    "Trying tefasfon fallback.",
                    symbol,
                )

            # tefasfon (DEGRADED) -- secondary fallback
            if _tefasfon_available() and source in ("tefasfon", ""):
                result = await self._analyze_tefasfon(symbol, horizon)
                if result.data_source_status != DataSourceStatus.NOT_CONFIGURED:
                    return result
                self.logger.warning(
                    "[midas_funds] tefasfon returned no data for %s. "
                    "Trying legacy scrape fallback.",
                    symbol,
                )

            # Legacy scrape last resort (when source='' and both libs fail)
            return await self._analyze_tefas_scrape(symbol, horizon)

        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # tefas-crawler path (AVAILABLE, primary)
    # ------------------------------------------------------------------

    async def _analyze_tefas_crawler(
        self, symbol: str, horizon: Horizon
    ) -> AdviceResult:
        """
        Fetch NAV history via tefas-crawler (pip install tefas-crawler).

        tefas-crawler installs as the 'tefas' package and talks to the
        2026 tefas.gov.tr/api/funds/ JSON API (per-fund history, up to 5y).

        tefas-crawler API (as of 2026-06-02):
          from tefas import Crawler
          crawler = Crawler()
          df = crawler.fetch(symbol, start_date='2024-01-01', end_date='2026-06-01',
                             columns=['date', 'price'])
          # Returns DataFrame with columns: date (index), price (float)
          # Some versions also expose: title, code, price, number_of_shares,
          #   number_of_investors, total_value

        Import-guarded: if not installed, caller falls back to tefasfon.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_tefas_crawler, symbol, horizon
        )

        if signals is None:
            # Propagate NOT_CONFIGURED so caller can try next fallback
            return self._not_configured(
                symbol, horizon,
                (
                    f"tefas-crawler returned no NAV data for fund '{symbol}'. "
                    "Possible causes: invalid FONKODU, tefas.gov.tr unreachable, "
                    "or API change. Trying tefasfon/legacy fallback. "
                    "Look up FONKODU at https://www.tefas.gov.tr."
                ),
            )

        return self._build_nav_result(
            symbol, horizon,
            signals=signals,
            actual_status=DataSourceStatus.AVAILABLE,
            data_source_tag="tefas_crawler",
            note=(
                f"NAV via tefas-crawler (tefas.gov.tr 2026 API). "
                f"Fund: {self.KNOWN_FUNDS.get(symbol.upper(), symbol)}. "
                "No OHLCV -- SMA+RSI only."
            ),
        )

    def _fetch_tefas_crawler(
        self, symbol: str, horizon: Horizon
    ) -> Optional[dict]:
        """
        Sync fetch via tefas-crawler. Runs in executor.
        Returns signals dict or None on any failure.
        """
        try:
            from tefas import Crawler  # import guard (pip install tefas-crawler)
        except ImportError:
            logger.debug(
                "[midas_funds] tefas-crawler not installed. "
                "pip install tefas-crawler to enable AVAILABLE data source."
            )
            return None

        try:
            import pandas as pd
        except ImportError:
            return None

        lookback = _LOOKBACK[horizon]
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 60)

        try:
            crawler = Crawler()
            df = crawler.fetch(
                symbol.upper(),
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                columns=["date", "price"],
            )
        except Exception as exc:
            logger.warning(
                "[midas_funds] tefas-crawler.fetch(%r) failed: %s",
                symbol, exc,
            )
            return None

        if df is None or df.empty or len(df) < 10:
            logger.debug(
                "[midas_funds] tefas-crawler insufficient data for %r (rows=%d).",
                symbol, 0 if df is None else len(df),
            )
            return None

        # Normalise columns (tefas-crawler may use 'price' or 'fiyat')
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        col_map = {}
        for c in df.columns:
            if c in ("price", "fiyat", "nav"):
                col_map[c] = "close"
            elif c == "date":
                col_map[c] = "date"
        df = df.rename(columns=col_map)

        # If date is a column (not index), set as index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")

        if "close" not in df.columns:
            logger.warning(
                "[midas_funds] tefas-crawler response missing price column "
                "for %r. Columns: %s",
                symbol, list(df.columns),
            )
            return None

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_index()

        if len(df) < 10:
            return None

        return _compute_fund_signals(df, lookback)

    # ------------------------------------------------------------------
    # tefasfon path (DEGRADED, secondary fallback)
    # ------------------------------------------------------------------

    async def _analyze_tefasfon(
        self, symbol: str, horizon: Horizon
    ) -> AdviceResult:
        """
        Fetch NAV history via tefasfon (pip install tefasfon).

        tefasfon is an alternative lightweight wrapper for tefas.gov.tr.
        Used as a secondary fallback when tefas-crawler is absent or fails.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_tefasfon, symbol, horizon
        )

        if signals is None:
            return self._not_configured(
                symbol, horizon,
                (
                    f"tefasfon returned no NAV data for fund '{symbol}'. "
                    "Possible causes: invalid FONKODU or tefas.gov.tr unreachable. "
                    "Set advisor_midas_data_source='manual' as last resort."
                ),
            )

        return self._build_nav_result(
            symbol, horizon,
            signals=signals,
            actual_status=DataSourceStatus.DEGRADED,
            data_source_tag="tefasfon",
            note=(
                f"NAV via tefasfon (tefas.gov.tr fallback wrapper). "
                f"Fund: {self.KNOWN_FUNDS.get(symbol.upper(), symbol)}. "
                "No OHLCV -- SMA+RSI only. "
                "For AVAILABLE status: pip install tefas-crawler."
            ),
        )

    def _fetch_tefasfon(
        self, symbol: str, horizon: Horizon
    ) -> Optional[dict]:
        """
        Sync fetch via tefasfon. Runs in executor.
        Returns signals dict or None on any failure.

        tefasfon API (pip install tefasfon):
          from tefasfon import Tefas
          t = Tefas()
          df = t.get_fund(symbol, start_date='YYYY-MM-DD', end_date='YYYY-MM-DD')
          # DataFrame with columns: date, price (or similar)
        """
        try:
            from tefasfon import Tefas  # import guard (pip install tefasfon)
        except ImportError:
            logger.debug(
                "[midas_funds] tefasfon not installed. "
                "pip install tefasfon for a secondary TEFAS data source."
            )
            return None

        try:
            import pandas as pd
        except ImportError:
            return None

        lookback = _LOOKBACK[horizon]
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 60)

        try:
            t = Tefas()
            df = t.get_fund(
                symbol.upper(),
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logger.warning(
                "[midas_funds] tefasfon.get_fund(%r) failed: %s",
                symbol, exc,
            )
            return None

        if df is None or df.empty or len(df) < 10:
            return None

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        # Map price-like columns to 'close'
        for candidate in ("price", "fiyat", "nav", "close"):
            if candidate in df.columns and candidate != "close":
                df = df.rename(columns={candidate: "close"})
                break

        if "close" not in df.columns:
            logger.warning(
                "[midas_funds] tefasfon response missing price column for %r. "
                "Columns: %s",
                symbol, list(df.columns),
            )
            return None

        # Ensure date index
        for candidate in ("date", "tarih"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
                df = df.dropna(subset=[candidate])
                df = df.set_index(candidate)
                break

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"]).sort_index()

        if len(df) < 10:
            return None

        return _compute_fund_signals(df, lookback)

    # ------------------------------------------------------------------
    # Legacy tefas.gov.tr POST scrape (DEGRADED, last free-lib fallback)
    # ------------------------------------------------------------------

    async def _analyze_tefas_scrape(
        self, symbol: str, horizon: Horizon
    ) -> AdviceResult:
        import asyncio

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_tefas_scrape, symbol, horizon
        )

        if signals is None:
            return self._not_configured(
                symbol,
                horizon,
                (
                    f"Tefas scrape returned no NAV data for fund '{symbol}'. "
                    "Possible causes: invalid FONKODU, tefas.gov.tr unreachable, "
                    "or CSRF/API change. "
                    "Recommended fix: pip install tefas-crawler (AVAILABLE source). "
                    "Last resort: set advisor_midas_data_source='manual' "
                    "and enter NAV via /advisor/portfolio."
                ),
            )

        return self._build_nav_result(
            symbol, horizon,
            signals=signals,
            actual_status=DataSourceStatus.DEGRADED,
            data_source_tag="tefas_scrape",
            note=(
                f"NAV via tefas.gov.tr POST scrape (fragile/undocumented). "
                f"Fund: {self.KNOWN_FUNDS.get(symbol.upper(), symbol)}. "
                "No OHLCV -- SMA+RSI only. "
                "Upgrade: pip install tefas-crawler for stable AVAILABLE source."
            ),
        )

    def _fetch_tefas_scrape(
        self, symbol: str, horizon: Horizon
    ) -> Optional[dict]:
        """
        Scrape historical NAV from tefas.gov.tr (undocumented POST endpoint).
        Runs in executor (sync). Legacy fallback.

        Endpoint: POST /api/DB/BindHistoryInfo
        Form: FONKODU=<code>&BASTARIH=<DD.MM.YYYY>&BITTARIH=<DD.MM.YYYY>
        CSRF: extracted from homepage cookie on initial GET.
        """
        try:
            import requests
            import pandas as pd
        except ImportError:
            raise RuntimeError("requests/pandas not installed")

        lookback = _LOOKBACK[horizon]
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 60)

        date_fmt = "%d.%m.%Y"
        start_str = start_dt.strftime(date_fmt)
        end_str = end_dt.strftime(date_fmt)

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; advisor)"})

        try:
            home_resp = session.get(
                f"{_TEFAS_BASE}/TarihselVeriler.aspx", timeout=10
            )
            home_resp.raise_for_status()
        except Exception as exc:
            self.logger.warning("[midas_funds] Tefas homepage GET failed: %s", exc)
            return None

        csrf_token = _extract_tefas_csrf(home_resp.text)

        form_data = {
            "FONKODU": symbol.upper(),
            "BASTARIH": start_str,
            "BITTARIH": end_str,
        }
        if csrf_token:
            form_data["__RequestVerificationToken"] = csrf_token

        try:
            api_resp = session.post(
                _TEFAS_HISTORY_API,
                data=form_data,
                headers=_TEFAS_HEADERS,
                timeout=15,
            )
            api_resp.raise_for_status()
            payload = api_resp.json()
        except Exception as exc:
            # Tefas POST commonly 404s (undocumented endpoint changes); 10 funds
            # x retries flooded advisor_errors.log. Log ONE warning per 5 min,
            # demote the rest to debug. If Tefas stays down, disable midas_funds
            # in enabled_markets — it produces no advice anyway.
            import time as _t
            now = _t.monotonic()
            if now - getattr(self, "_tefas_warn_ts", 0.0) >= 300.0:
                self._tefas_warn_ts = now
                self.logger.warning(
                    "[midas_funds] Tefas API unavailable (e.g. %s: %s) — further "
                    "occurrences suppressed 5m. Disable midas_funds in "
                    "enabled_markets if Tefas stays down.", symbol, exc
                )
            else:
                self.logger.debug(
                    "[midas_funds] Tefas API POST failed for %s: %s", symbol, exc
                )
            return None

        records = payload if isinstance(payload, list) else payload.get("data", [])
        if not records:
            self.logger.debug(
                "[midas_funds] Tefas scrape returned empty data for %s", symbol
            )
            return None

        try:
            df = pd.DataFrame(records)
            col_map = {}
            for c in df.columns:
                cu = c.upper()
                if cu in ("TARIH", "DATE"):
                    col_map[c] = "date"
                elif cu in ("FIYAT", "PRICE", "NAV"):
                    col_map[c] = "close"
            df = df.rename(columns=col_map)
            if "date" not in df.columns or "close" not in df.columns:
                self.logger.warning(
                    "[midas_funds] Unexpected Tefas scrape columns: %s",
                    list(df.columns),
                )
                return None
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            df = df.set_index("date").sort_index()
        except Exception as exc:
            self.logger.warning(
                "[midas_funds] Tefas scrape parse failed for %s: %s", symbol, exc
            )
            return None

        if len(df) < 10:
            return None

        return _compute_fund_signals(df, lookback)

    # ------------------------------------------------------------------
    # Manual entry path (DEGRADED, last resort)
    # ------------------------------------------------------------------

    async def _analyze_manual(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Manual NAV entry path. Operator enters current NAV via
        /advisor/portfolio dashboard page. No historical series -> NEUTRAL.
        """
        fund_name = self.KNOWN_FUNDS.get(symbol.upper(), symbol)
        note = (
            f"Manual NAV entry mode for {fund_name} ({symbol}). "
            "No historical series available -- technical analysis not possible. "
            "Enter current NAV and notes via /advisor/portfolio. "
            "Upgrade: pip install tefas-crawler for AVAILABLE data source."
        )

        last_nav = await self._read_last_manual_nav(symbol)
        if last_nav:
            self.last_price[symbol] = last_nav

        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=Direction.NEUTRAL,
            entry_low=round(last_nav * 0.999, 4) if last_nav else None,
            entry_high=round(last_nav * 1.001, 4) if last_nav else None,
            confidence=0.0,
            rationale=(
                f"Manual NAV entry for {fund_name}. "
                "No technical analysis without historical price data. "
                "Review fund performance on tefas.gov.tr and enter notes "
                "in the portfolio page."
            ),
            model_id="",
            data_source_status=DataSourceStatus.DEGRADED,
            data_source_note=note,
            sim_enabled=False,  # can't open a sim without price history
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra={"data_source": "manual", "last_nav": last_nav},
        )

    async def _read_last_manual_nav(self, symbol: str) -> Optional[float]:
        """
        Try to read the last operator-entered NAV from advisor_portfolio table.
        Returns None if DB unavailable or no entry exists.
        """
        if self.db_pool is None:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT current_price FROM advisor_portfolio
                    WHERE symbol = $1 AND market = 'midas_funds'
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    symbol.upper(),
                )
                if row and row["current_price"] is not None:
                    return float(row["current_price"])
        except Exception as exc:
            self.logger.debug(
                "[midas_funds] Could not read manual NAV for %s: %s", symbol, exc
            )
        return None

    # ------------------------------------------------------------------
    # Shared AdviceResult builder for NAV-series paths
    # ------------------------------------------------------------------

    async def _build_nav_result(
        self,
        symbol: str,
        horizon: Horizon,
        signals: dict,
        actual_status: DataSourceStatus,
        data_source_tag: str,
        note: str,
    ) -> AdviceResult:
        """Build AdviceResult from a NAV signals dict (async, awaits rationale)."""
        direction = _signals_to_direction_fund(signals)
        # Funds: only SMA + RSI vote (no BB on NAV-only series) -> n_votes=2.
        confidence = signal_confidence(
            signals, horizon, config=self.config, n_votes=2
        )
        last_nav = signals.get("close", 0)

        self.last_price[symbol] = last_nav

        rationale = await build_rationale(
            symbol=symbol,
            market=self.market,
            horizon=horizon,
            signals=signals,
            direction=direction,
            config=self.config,
            caller_logger=self.logger,
        )

        model_id = self.config.get("advisor_anthropic_model", "claude-opus-4-5")
        extra = {"klines_df": signals.get("df"), "data_source": data_source_tag}

        entry_low, entry_high, target, stop = horizon_levels(
            signals, direction, horizon, config=self.config, price_decimals=4
        )

        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=direction,
            entry_low=entry_low,
            entry_high=entry_high,
            target_price=target,
            stop_price=stop,
            confidence=confidence,
            rationale=rationale,
            model_id=model_id,
            data_source_status=actual_status,
            data_source_note=note,
            sim_enabled=str(self.config.get("sim_default_enabled", "false")).lower() == "true",
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Fund signal computation (no BB / vol_ratio -- single-price NAV series)
# ---------------------------------------------------------------------------

def _compute_fund_signals(df, lookback: int) -> Optional[dict]:
    """
    Compute SMA + RSI from a NAV price DataFrame.
    bb_signal and vol_ratio are zeroed/neutral (no OHLCV for funds).
    """
    close = df["close"]

    sma20 = close.rolling(min(20, len(close))).mean()
    sma50 = close.rolling(min(50, len(close))).mean()
    sma_signal = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1

    rsi_period = min(14, len(close) - 1)
    if rsi_period < 3:
        rsi_signal = 0
        rsi_val = 50.0
    else:
        rsi_val = _rsi(close, rsi_period).iloc[-1]
        if rsi_val > 70:
            rsi_signal = -1
        elif rsi_val < 30:
            rsi_signal = 1
        else:
            rsi_signal = 0

    last_close = close.iloc[-1]
    return_bars = min(lookback, len(df))

    # NAV daily-return stdev as the volatility unit (no OHLCV -> no true ATR).
    # This lets horizon_levels scale fund target/stop by the fund's own vol.
    atr_pct = _nav_vol_pct(close)

    return {
        "close": last_close,
        "sma20": sma20.iloc[-1],
        "sma50": sma50.iloc[-1],
        "rsi": rsi_val,
        # No BB for single-price series (no OHLCV)
        "sma_signal": sma_signal,
        "rsi_signal": rsi_signal,
        "bb_signal": 0,       # neutral -- no Bollinger on NAV-only series
        "vol_ratio": 1.0,     # no volume for mutual funds
        "atr_pct": atr_pct,
        "df": df[["close"]].tail(return_bars),
    }


def _nav_vol_pct(close, window: int = 20) -> Optional[float]:
    """
    Daily NAV-return standard deviation (fractional) over the last `window` bars.
    Serves as the volatility unit for funds (which have no OHLCV/ATR). None on
    insufficient/NaN data so levels.py falls back to the configured vol floor.
    """
    try:
        import math as _math
        rets = close.pct_change().dropna()
        if len(rets) < 5:
            return None
        vol = float(rets.tail(window).std())
        if vol is None or _math.isnan(vol) or vol <= 0:
            return None
        return vol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fund-specific helpers (pure functions)
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _signals_to_direction_fund(signals: dict) -> Direction:
    """
    For funds: only SMA + RSI vote (no reliable BB on NAV-only series).
    Tie (0 votes) = NEUTRAL.
    """
    votes = signals.get("sma_signal", 0) + signals.get("rsi_signal", 0)
    if votes > 0:
        return Direction.LONG
    elif votes < 0:
        return Direction.SHORT
    return Direction.NEUTRAL


def _extract_tefas_csrf(html: str) -> Optional[str]:
    """
    Extract __RequestVerificationToken from Tefas HTML.
    ASP.NET anti-CSRF token in a hidden input field.
    Returns None if not found (many requests succeed without it).
    """
    import re
    match = re.search(
        r'<input[^>]+name="__RequestVerificationToken"[^>]+value="([^"]+)"',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None
