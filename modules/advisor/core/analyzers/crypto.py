"""
CryptoAnalyzer — crypto market analyzer using ccxt (free, no key for public data).

Data source: ccxt with public REST endpoints (no API key for read-only OHLCV).
Supported exchanges: Binance, Bybit, Coinbase (configurable via
advisor_config key: advisor_crypto_exchange, default "binance").

Required operator config (advisor_config DB):
  advisor_crypto_exchange   : exchange id for ccxt (default "binance")
  watchlist_crypto          : comma-separated pairs, e.g. "BTC/USDT,ETH/USDT"

Data source status: AVAILABLE (ccxt public endpoints, no key required).

STUB STATUS (Wave-20): scaffold only. Technical signal computation and LLM
rationale generation to be implemented by the quant-algo specialist agent
in Wave-21. The analyze() method returns a NOT_CONFIGURED placeholder
until the quant agent wires the ccxt OHLCV fetch + signal logic.
"""

from __future__ import annotations

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import AdviceResult, DataSourceStatus, Horizon, Market


class CryptoAnalyzer(BaseAnalyzer):
    """
    Crypto market advisor using ccxt public OHLCV.

    STUB — implement in Wave-21 (quant-algo agent).
    Interface contract: see base_analyzer.BaseAnalyzer.
    Reference implementation: us_equities.USEquitiesAnalyzer.
    """

    market = Market.CRYPTO

    def data_source_status(self) -> DataSourceStatus:
        try:
            import ccxt  # noqa: F401
            return DataSourceStatus.AVAILABLE
        except ImportError:
            return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        STUB. Wave-21 quant agent: implement ccxt OHLCV fetch + technicals.

        Steps to implement:
          1. Load ccxt exchange from advisor_config.advisor_crypto_exchange.
          2. Fetch OHLCV bars: exchange.fetch_ohlcv(symbol, "1d", limit=200).
          3. Compute SMA/RSI/BB signals (mirror us_equities pattern).
          4. Call Anthropic for rationale (reuse _llm_rationale pattern).
          5. Return AdviceResult with data_source_status=AVAILABLE.
        """
        return self._not_configured(
            symbol, horizon,
            "CryptoAnalyzer not yet implemented (Wave-21 task). "
            "Install ccxt and implement analyze() in crypto.py."
        )
