"""
MidasFundsAnalyzer — Turkish mutual fund (fon) analyzer for Midas exchange.

Target funds (examples):
  - Tera Portfoy Para Piyasasi Fonu (TPP)
  - Ak Portfoy Para Piyasasi Katilim Fonu (AKP)
  - Other TEFAS-listed funds as FONKODU codes

Data source options:

  1. tefas.gov.tr POST scrape (tefas_scrape) — DEGRADED
     Official NAV source for all Turkish mutual funds. Undocumented POST API.
     Endpoint: POST https://www.tefas.gov.tr/api/DB/BindHistoryInfo
     Form fields: FONKODU=<code>&BASTARIH=<DD.MM.YYYY>&BITTARIH=<DD.MM.YYYY>
     CSRF token is extracted from a fresh GET to the homepage.
     Risk: undocumented endpoint; may break without notice.
     Returns: JSON list of {TARIH, FIYAT} (date, NAV price).
     data_source_status = DEGRADED (fragile / no SLA).

  2. manual entry — DEGRADED
     Operator enters current NAV manually via /advisor/portfolio dashboard page.
     Stored in advisor_portfolio table. No historical series → no technicals.
     Returns NEUTRAL advice with current NAV from last manual entry.
     data_source_status = DEGRADED (limited analysis possible).

Default: NOT_CONFIGURED (returns instructions to operator).
Activate: set advisor_midas_data_source='tefas_scrape' or 'manual'
          in advisor_config DB.
Watchlist: watchlist_midas_funds='TPP,AKP,MAC' (Tefas FONKODU format).
           Look up codes at https://www.tefas.gov.tr.

Technical analysis:
  tefas_scrape: SMA + RSI only (single-price NAV series; no OHLCV, no volume).
  manual      : NEUTRAL direction; no technical signals possible.
  BB is omitted for single-price series (requires O/H/L which funds don't have).

LLM rationale: shared build_rationale() — works with partial signal dict.

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

logger = logging.getLogger("advisor.analyzer.midas_funds")

_LOOKBACK: dict = {
    Horizon.SHORT: 30,
    Horizon.MID: 180,
    Horizon.LONG: 730,
}

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
    "'AKP' for Ak Portfoy Para Piyasasi Katilim Fonu). "
    "Look up FONKODU at https://www.tefas.gov.tr."
)

# Tefas undocumented endpoint details
_TEFAS_BASE = "https://www.tefas.gov.tr"
_TEFAS_HISTORY_API = f"{_TEFAS_BASE}/api/DB/BindHistoryInfo"
_TEFAS_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{_TEFAS_BASE}/TarihselVeriler.aspx",
}


class MidasFundsAnalyzer(BaseAnalyzer):
    """
    Turkish mutual fund (fon) analyzer for Midas exchange.

    tefas_scrape : DEGRADED — fragile undocumented POST scrape.
    manual       : DEGRADED — operator-entered NAV, no historical technicals.
    not set      : NOT_CONFIGURED.
    """

    market = Market.MIDAS_FUNDS

    # Known Tefas fund codes → human-readable names (expand as needed)
    KNOWN_FUNDS = {
        "TPP": "Tera Portfoy Para Piyasasi Fonu",
        "AKP": "Ak Portfoy Para Piyasasi Katilim Fonu",
        "MAC": "Midas AI Cathie Wood Fonu",   # verify FONKODU on tefas.gov.tr
    }

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self.last_price: dict = {}

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_midas_data_source", "")
        if source in ("tefas_scrape", "manual"):
            return DataSourceStatus.DEGRADED
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a Turkish mutual fund (FONKODU).
        Never raises — _error_result or _not_configured on any failure.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)

        source = self.config.get("advisor_midas_data_source", "")
        try:
            if source == "tefas_scrape":
                return await self._analyze_tefas(symbol, horizon)
            if source == "manual":
                return await self._analyze_manual(symbol, horizon)
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)
        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # tefas_scrape path
    # ------------------------------------------------------------------

    async def _analyze_tefas(self, symbol: str, horizon: Horizon) -> AdviceResult:
        import asyncio

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_tefas, symbol, horizon
        )

        if signals is None:
            # Scraper failed — fall through to NOT_CONFIGURED note so operator
            # understands manual entry is the reliable fallback.
            return self._not_configured(
                symbol,
                horizon,
                (
                    f"Tefas scrape returned no NAV data for fund '{symbol}'. "
                    "Possible causes: invalid FONKODU, tefas.gov.tr unreachable, "
                    "or CSRF/API change. Set advisor_midas_data_source='manual' "
                    "and enter NAV via /advisor/portfolio as fallback."
                ),
            )

        direction = _signals_to_direction_fund(signals)
        confidence = _compute_confidence_fund(signals)
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
        fund_name = self.KNOWN_FUNDS.get(symbol.upper(), symbol)
        note = (
            f"NAV via tefas.gov.tr scrape (fragile/undocumented). "
            f"Fund: {fund_name}. No OHLCV — SMA+RSI only."
        )
        extra = {"klines_df": signals.get("df"), "data_source": "tefas_scrape"}

        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=direction,
            entry_low=round(last_nav * 0.999, 4),
            entry_high=round(last_nav * 1.001, 4),
            target_price=_target_fund(signals, direction),
            stop_price=_stop_fund(signals, direction),
            confidence=confidence,
            rationale=rationale,
            model_id=model_id,
            data_source_status=DataSourceStatus.DEGRADED,
            data_source_note=note,
            sim_enabled=bool(self.config.get("sim_default_enabled", "false") == "true"),
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    def _fetch_tefas(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Scrape historical NAV from tefas.gov.tr.
        Runs in executor (sync).

        Endpoint: POST /api/DB/BindHistoryInfo
        Form: FONKODU=<code>&BASTARIH=<DD.MM.YYYY>&BITTARIH=<DD.MM.YYYY>
        CSRF: extracted from homepage cookie on initial GET.

        Returns a signal dict with SMA+RSI (no BB — single-price series).
        Returns None on any network/parse failure (caller falls back gracefully).
        """
        try:
            import requests
            import pandas as pd
        except ImportError:
            raise RuntimeError("requests/pandas not installed")

        lookback = _LOOKBACK[horizon]
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 60)

        # Tefas date format: DD.MM.YYYY
        date_fmt = "%d.%m.%Y"
        start_str = start_dt.strftime(date_fmt)
        end_str = end_dt.strftime(date_fmt)

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; advisor)"})

        # Step 1: GET homepage to acquire session cookie + CSRF token
        try:
            home_resp = session.get(
                f"{_TEFAS_BASE}/TarihselVeriler.aspx", timeout=10
            )
            home_resp.raise_for_status()
        except Exception as exc:
            self.logger.warning(f"[midas_funds] Tefas homepage GET failed: {exc}")
            return None

        # Extract __RequestVerificationToken if present (ASP.NET CSRF)
        csrf_token = _extract_tefas_csrf(home_resp.text)

        # Step 2: POST to history API
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
            self.logger.warning(
                f"[midas_funds] Tefas API POST failed for {symbol}: {exc}"
            )
            return None

        # Step 3: parse response
        # Response format: {"data": [{"TARIH": "02.06.2026", "FIYAT": 1.234567}, ...]}
        # or directly a list — handle both
        records = payload if isinstance(payload, list) else payload.get("data", [])
        if not records:
            self.logger.warning(
                f"[midas_funds] Tefas returned empty data for {symbol}"
            )
            return None

        try:
            df = pd.DataFrame(records)
            # Normalise column names — Tefas uses Turkish uppercase
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
                    f"[midas_funds] Unexpected Tefas columns: {list(df.columns)}"
                )
                return None
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            df = df.set_index("date").sort_index()
        except Exception as exc:
            self.logger.warning(
                f"[midas_funds] Tefas data parse failed for {symbol}: {exc}"
            )
            return None

        if len(df) < 10:
            return None

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

        return {
            "close": last_close,
            "sma20": sma20.iloc[-1],
            "sma50": sma50.iloc[-1],
            "rsi": rsi_val,
            # No BB for single-price series (no OHLCV)
            "sma_signal": sma_signal,
            "rsi_signal": rsi_signal,
            "bb_signal": 0,       # neutral — no Bollinger on NAV-only series
            "vol_ratio": 1.0,     # no volume for mutual funds
            "df": df[["close"]].tail(return_bars),
        }

    # ------------------------------------------------------------------
    # manual entry path
    # ------------------------------------------------------------------

    async def _analyze_manual(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Manual NAV entry path. Operator enters current NAV via
        /advisor/portfolio dashboard page. No historical series → NEUTRAL.
        """
        fund_name = self.KNOWN_FUNDS.get(symbol.upper(), symbol)
        note = (
            f"Manual NAV entry mode for {fund_name} ({symbol}). "
            "No historical series available — technical analysis not possible. "
            "Enter current NAV and notes via /advisor/portfolio."
        )

        # Attempt to read last manual price from db_pool
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
                "No technical analysis is available without historical price data. "
                "Operator should review fund performance on tefas.gov.tr and "
                "enter notes in the portfolio page."
            ),
            model_id="",
            data_source_status=DataSourceStatus.DEGRADED,
            data_source_note=note,
            sim_enabled=False,  # can't open a sim without a price history
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
                f"[midas_funds] Could not read manual NAV for {symbol}: {exc}"
            )
        return None


# ---------------------------------------------------------------------------
# Fund-specific signal helpers (no BB / vol_ratio — single-price NAV series)
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


def _compute_confidence_fund(signals: dict) -> float:
    """Confidence based on 2-signal agreement (lower ceiling than OHLCV)."""
    abs_vote = abs(signals.get("sma_signal", 0) + signals.get("rsi_signal", 0))
    return min(abs_vote / 2.0 * 0.8, 1.0)  # cap at 0.8 — no OHLCV validation


def _target_fund(signals: dict, direction: Direction) -> Optional[float]:
    close = signals.get("close", 0)
    if not close:
        return None
    if direction == Direction.LONG:
        return round(close * 1.03, 4)   # 3% target for fund NAV
    elif direction == Direction.SHORT:
        return round(close * 0.97, 4)
    return None


def _stop_fund(signals: dict, direction: Direction) -> Optional[float]:
    close = signals.get("close", 0)
    if not close:
        return None
    if direction == Direction.LONG:
        return round(close * 0.98, 4)   # 2% stop for fund NAV
    elif direction == Direction.SHORT:
        return round(close * 1.02, 4)
    return None


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
