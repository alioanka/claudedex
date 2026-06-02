"""
BaseAnalyzer — abstract contract every market analyzer must implement.

Rules
-----
- No auto-trade logic anywhere in subclasses.
- Each analyzer is responsible for fetching its own data; it must NOT call
  pool_engine, risk_manager, or any other trading-module engine.
- A "not configured" data source MUST return AdviceResult with
  data_source_status=NOT_CONFIGURED rather than raising. Fail-soft.
- analyze() is async; I/O (yfinance, ccxt, HTTP) goes here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from modules.advisor.core.models import (
    AdviceResult,
    DataSourceStatus,
    Direction,
    Horizon,
    Market,
)


class BaseAnalyzer(ABC):
    """
    Abstract market analyzer.

    Subclasses implement analyze() for a specific market segment.
    The advice_engine calls analyze() once per cycle per symbol per horizon.
    """

    #: Market segment this analyzer covers (set in subclass class body).
    market: Market

    def __init__(self, config: dict, db_pool=None):
        """
        Parameters
        ----------
        config  : dict of advisor_config keys relevant to this analyzer.
        db_pool : asyncpg pool for DB reads (e.g. reading config overrides).
                  May be None when running in unit-test mode.
        """
        self.config = config
        self.db_pool = db_pool
        self.logger = logging.getLogger(
            f"advisor.analyzer.{self.__class__.__name__}"
        )

    @abstractmethod
    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce a single AdviceResult for (symbol, horizon).

        Must never raise — catch all exceptions and return an AdviceResult
        with data_source_status=ERROR and a descriptive data_source_note.

        Parameters
        ----------
        symbol  : Ticker string in the market's native format
                  (e.g. "AAPL", "BTC/USDT", "THYAO.IS", "XAUUSD",
                   "Tera Portfoy Para Piyasasi Fonu").
        horizon : SHORT / MID / LONG.

        Returns
        -------
        AdviceResult — always, never None.
        """

    @abstractmethod
    def data_source_status(self) -> DataSourceStatus:
        """
        Return the current availability of the underlying data source.
        Called at module startup and exposed on the health endpoint.
        Subclasses should check required API keys / connectivity.
        """

    def _not_configured(
        self,
        symbol: str,
        horizon: Horizon,
        note: str,
    ) -> AdviceResult:
        """
        Convenience factory for a NOT_CONFIGURED result stub.
        Use this in subclasses that lack data source credentials.
        """
        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            rationale="Data source not configured. See data_source_note.",
            data_source_status=DataSourceStatus.NOT_CONFIGURED,
            data_source_note=note,
        )

    def _error_result(
        self,
        symbol: str,
        horizon: Horizon,
        exc: Exception,
    ) -> AdviceResult:
        """
        Convenience factory for a runtime ERROR result.
        """
        self.logger.error(
            f"Analyzer error for {symbol}/{horizon.value}: {exc}",
            exc_info=True,
        )
        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            rationale="Data fetch error. Retry next cycle.",
            data_source_status=DataSourceStatus.ERROR,
            data_source_note=str(exc)[:500],
        )

    async def health_check(self) -> dict:
        """
        Returns a health dict for the dashboard's /api/advisor/health endpoint.
        Override in subclasses that have richer connectivity checks.
        """
        return {
            "analyzer": self.__class__.__name__,
            "market": self.market.value,
            "data_source_status": self.data_source_status().value,
        }
