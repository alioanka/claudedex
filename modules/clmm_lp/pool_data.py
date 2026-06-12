"""Free, read-only pool data for CLMM candidates. Fail-soft everywhere.

Primary source: DexScreener public REST (no key) — price, 24h volume, TVL,
for both EVM (Uniswap v3 pool address == pair address) and Solana (Orca).
Optional: one-time on-chain fee-tier verification for EVM pools via
config.pool_engine.PoolEngine endpoints (READ-ONLY eth_call, never sends).
"""
import logging
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("ClmmLpModule.PoolData")

DEFAULT_DEXSCREENER_BASE = "https://api.dexscreener.com"

# DexScreener chain slug -> pool_engine provider type (EVM only).
_EVM_PROVIDER_TYPES = {
    'ethereum': 'ETHEREUM_RPC',
    'base': 'BASE_RPC',
    'arbitrum': 'ARBITRUM_RPC',
    'polygon': 'POLYGON_RPC',
    'bsc': 'BSC_RPC',
    'optimism': 'OPTIMISM_RPC',
}
_UNIV3_FEE_SELECTOR = "0xddca3f43"  # fee()


class PoolDataClient:
    def __init__(self, base_url: str = DEFAULT_DEXSCREENER_BASE, timeout_s: float = 15.0):
        self.base_url = base_url.rstrip('/')
        self.timeout_s = timeout_s
        self.last_error: Optional[str] = None
        self._fee_verified: Dict[str, Optional[int]] = {}  # pool addr -> on-chain bps

    async def fetch_pool_snapshot(self, chain: str, pool_address: str) -> Optional[Dict[str, Any]]:
        """{'price','volume_24h_usd','tvl_usd'} from DexScreener, or None."""
        url = f"{self.base_url}/latest/dex/pairs/{chain}/{pool_address}"
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        self.last_error = f"dexscreener http {resp.status}"
                        return None
                    data = await resp.json()
        except Exception as e:
            self.last_error = f"dexscreener: {type(e).__name__}: {e}"
            return None
        pairs = data.get('pairs') or ([data['pair']] if data.get('pair') else [])
        if not pairs:
            self.last_error = "dexscreener: no pair data"
            return None
        pair = pairs[0]
        try:
            price = float(pair.get('priceUsd') or 0)
            volume = float((pair.get('volume') or {}).get('h24') or 0)
            tvl = float((pair.get('liquidity') or {}).get('usd') or 0)
        except (TypeError, ValueError) as e:
            self.last_error = f"dexscreener parse: {e}"
            return None
        if price <= 0 or tvl <= 0:
            self.last_error = f"dexscreener: degenerate snapshot price={price} tvl={tvl}"
            return None
        self.last_error = None
        return {'price': price, 'volume_24h_usd': volume, 'tvl_usd': tvl,
                'base_symbol': (pair.get('baseToken') or {}).get('symbol', '?'),
                'quote_symbol': (pair.get('quoteToken') or {}).get('symbol', '?')}

    async def verify_evm_fee_bps(self, chain: str, pool_address: str) -> Optional[int]:
        """On-chain Uniswap v3 fee() in bps via pool_engine (read-only, cached).

        Returns None when unverifiable (non-EVM chain, no endpoint, RPC error)
        — the caller must treat None as 'unverified', never as a mismatch.
        """
        key = f"{chain}:{pool_address.lower()}"
        if key in self._fee_verified:
            return self._fee_verified[key]
        provider_type = _EVM_PROVIDER_TYPES.get(chain.lower())
        if provider_type is None:
            self._fee_verified[key] = None
            return None
        endpoint = None
        try:
            from config.pool_engine import PoolEngine
            pool_engine = await PoolEngine.get_instance()
            endpoint = await pool_engine.get_endpoint(provider_type)
        except Exception as e:
            logger.warning("pool_engine unavailable for %s (%s) — fee unverified", chain, e)
            return None
        if not endpoint:
            return None
        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                   "params": [{"to": pool_address, "data": _UNIV3_FEE_SELECTOR}, "latest"]}
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload) as resp:
                    if resp.status == 429:
                        await pool_engine.report_rate_limit(provider_type, endpoint)
                        return None
                    body = await resp.json()
            result = body.get('result')
            if not result or result == '0x':
                await pool_engine.report_failure(provider_type, endpoint,
                                                 error_type='empty_result')
                return None
            fee_units = int(result, 16)            # Uniswap units: hundredths of a bp
            await pool_engine.report_success(provider_type, endpoint)
            fee_bps = fee_units // 100
            self._fee_verified[key] = fee_bps
            return fee_bps
        except Exception as e:
            try:
                await pool_engine.report_failure(provider_type, endpoint,
                                                 error_type=type(e).__name__,
                                                 error_message=str(e)[:200])
            except Exception:
                pass
            logger.warning("fee() eth_call failed for %s: %s", pool_address, e)
            return None
