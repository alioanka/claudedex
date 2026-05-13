"""
Futures Exchange Executors

Supported exchanges:
- Binance Futures (primary)
- Bybit Futures (secondary)
- OKX Futures (future implementation)

Executors are lazy-loaded (PEP 562) so importing the normalizers
doesn't transitively pull in aiohttp / web3 / etc.
"""

from ._normalizers import normalize_position, normalize_balance

__all__ = [
    'BinanceFuturesExecutor',
    'BybitFuturesExecutor',
    'normalize_position',
    'normalize_balance',
]


def __getattr__(name: str):
    if name == 'BinanceFuturesExecutor':
        from .binance_futures import BinanceFuturesExecutor
        return BinanceFuturesExecutor
    if name == 'BybitFuturesExecutor':
        from .bybit_futures import BybitFuturesExecutor
        return BybitFuturesExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
