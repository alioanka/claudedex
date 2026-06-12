"""Free public market-data clients for the basis_desk module.

Read-only, key-less REST endpoints only (no signing, no account access):
  - Bybit V5:  /v5/market/tickers?category=linear  (perp funding + mark)
               /v5/market/tickers?category=spot    (spot last price)
  - Binance:   fapi /fapi/v1/premiumIndex          (perp funding + mark)
               spot /api/v3/ticker/price           (spot last price)

Each client returns {symbol: VenueQuote} for the requested symbol set and is
fail-soft: any HTTP/parse error returns {} and stashes `last_error` (the
engine continues with the other venue). Funding rates from both APIs are
fractions per funding interval; we convert to bps (rate * 1e4).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import aiohttp

logger = logging.getLogger("BasisDeskModule.Venues")

_TIMEOUT = aiohttp.ClientTimeout(total=15)


@dataclass(frozen=True)
class VenueQuote:
    venue: str
    symbol: str
    funding_bps: float   # signed, per funding interval
    perp_price: float    # mark price
    spot_price: float    # spot last price


async def _get_json(url: str, params: Optional[dict] = None):
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()


def _f(val, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class BybitPublicClient:
    def __init__(self, base_url: str = 'https://api.bybit.com'):
        self.base_url = base_url.rstrip('/')
        self.last_error: Optional[str] = None

    async def fetch_quotes(self, symbols: Iterable[str]) -> Dict[str, VenueQuote]:
        wanted = {s.upper() for s in symbols}
        try:
            perp = await _get_json(f'{self.base_url}/v5/market/tickers',
                                   {'category': 'linear'})
            spot = await _get_json(f'{self.base_url}/v5/market/tickers',
                                   {'category': 'spot'})
        except Exception as e:
            self.last_error = f'{type(e).__name__}: {e}'
            logger.warning('bybit fetch failed (fail-soft): %s', self.last_error)
            return {}
        self.last_error = None

        spot_px = {}
        for t in (spot.get('result') or {}).get('list') or []:
            if t.get('symbol') in wanted:
                px = _f(t.get('lastPrice'))
                if px and px > 0:
                    spot_px[t['symbol']] = px

        out: Dict[str, VenueQuote] = {}
        for t in (perp.get('result') or {}).get('list') or []:
            sym = t.get('symbol')
            if sym not in wanted or sym not in spot_px:
                continue
            rate = _f(t.get('fundingRate'))
            mark = _f(t.get('markPrice')) or _f(t.get('lastPrice'))
            if rate is None or not mark or mark <= 0:
                continue
            out[sym] = VenueQuote('bybit', sym, rate * 10000.0, mark,
                                  spot_px[sym])
        return out


class BinancePublicClient:
    def __init__(self,
                 fapi_base_url: str = 'https://fapi.binance.com',
                 spot_base_url: str = 'https://api.binance.com'):
        self.fapi_base_url = fapi_base_url.rstrip('/')
        self.spot_base_url = spot_base_url.rstrip('/')
        self.last_error: Optional[str] = None

    async def fetch_quotes(self, symbols: Iterable[str]) -> Dict[str, VenueQuote]:
        wanted = {s.upper() for s in symbols}
        try:
            perp = await _get_json(f'{self.fapi_base_url}/fapi/v1/premiumIndex')
            spot = await _get_json(f'{self.spot_base_url}/api/v3/ticker/price')
        except Exception as e:
            self.last_error = f'{type(e).__name__}: {e}'
            logger.warning('binance fetch failed (fail-soft): %s', self.last_error)
            return {}
        self.last_error = None

        spot_px = {}
        for t in spot if isinstance(spot, list) else []:
            if t.get('symbol') in wanted:
                px = _f(t.get('price'))
                if px and px > 0:
                    spot_px[t['symbol']] = px

        out: Dict[str, VenueQuote] = {}
        for t in perp if isinstance(perp, list) else []:
            sym = t.get('symbol')
            if sym not in wanted or sym not in spot_px:
                continue
            rate = _f(t.get('lastFundingRate'))
            mark = _f(t.get('markPrice'))
            if rate is None or not mark or mark <= 0:
                continue
            out[sym] = VenueQuote('binance', sym, rate * 10000.0, mark,
                                  spot_px[sym])
        return out


def build_clients(config: dict) -> Dict[str, object]:
    """Instantiate the clients named in config 'venues' (csv). Fail-soft:
    an unknown venue name is logged and skipped."""
    names = [v.strip().lower()
             for v in str(config.get('venues', 'bybit,binance')).split(',')
             if v.strip()]
    clients: Dict[str, object] = {}
    for name in names:
        if name == 'bybit':
            clients['bybit'] = BybitPublicClient(
                str(config.get('bybit_base_url', 'https://api.bybit.com')))
        elif name == 'binance':
            clients['binance'] = BinancePublicClient(
                str(config.get('binance_fapi_base_url',
                               'https://fapi.binance.com')),
                str(config.get('binance_spot_base_url',
                               'https://api.binance.com')))
        else:
            logger.warning("unknown venue '%s' in config — skipped", name)
    return clients
