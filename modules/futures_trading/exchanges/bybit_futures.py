"""
Bybit Futures Executor - Integration with Bybit Derivatives API

Handles:
- USDT perpetuals
- Inverse perpetuals
- Position management
- Leverage control
"""

import asyncio
import json
import logging
import hmac
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
import aiohttp


class BybitFuturesExecutor:
    """
    Bybit Futures API executor

    Note: This is a simplified implementation.
    For production, consider using the official pybit library.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        max_leverage: int = 3
    ):
        """
        Initialize Bybit Futures executor

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: Use testnet if True
            max_leverage: Maximum leverage
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_leverage = max_leverage

        # API endpoints
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self.logger = logging.getLogger("BybitFutures")
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> bool:
        """Initialize the executor"""
        try:
            self.session = aiohttp.ClientSession()
            self.logger.info("✅ Bybit Futures initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit Futures: {e}")
            return False

    async def close(self):
        """Close the executor"""
        if self.session:
            await self.session.close()

    def _generate_signature(self, params: str) -> str:
        """Generate HMAC SHA256 signature (legacy helper, unused on V5 path)."""
        signature = hmac.new(
            self.api_secret.encode(),
            params.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        body: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """V5 authenticated request. Returns parsed result dict on success, None on failure."""
        if not self.session:
            self.logger.error("Bybit session not initialised")
            return None
        body = body or {}
        # Bybit V5 signs the exact raw JSON body bytes — must match what we send on the wire.
        raw_body = json.dumps(body, separators=(',', ':')) if body else ""
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        pre_sign = f"{ts}{self.api_key}{recv_window}{raw_body}"
        signature = hmac.new(
            self.api_secret.encode(), pre_sign.encode(), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.request(
                method, url, data=raw_body or None, headers=headers
            ) as resp:
                data = await resp.json()
                ret_code = data.get("retCode")
                if ret_code == 0:
                    return data.get("result") or {}
                # Idempotency: leverage/margin already at requested value — treat as success.
                if ret_code in (110043, 110026):
                    self.logger.debug(
                        f"Bybit no-op response {ret_code}: {data.get('retMsg')}"
                    )
                    return data.get("result") or {}
                self.logger.error(
                    f"Bybit {method} {endpoint} failed: {ret_code} {data.get('retMsg')}"
                )
                return None
        except Exception as e:
            self.logger.error(f"Bybit {method} {endpoint} exception: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set buy/sell leverage for a linear (USDT perp) symbol."""
        leverage = min(int(leverage), self.max_leverage)
        body = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        result = await self._request("POST", "/v5/position/set-leverage", body)
        if result is not None:
            self.logger.info(f"Bybit set {symbol} leverage to {leverage}x")
            return True
        return False

    async def set_margin_type(
        self,
        symbol: str,
        margin_type: str = "ISOLATED",
        leverage: Optional[int] = None,
    ) -> bool:
        """Switch a symbol to ISOLATED (tradeMode=1) or CROSS (tradeMode=0) margin."""
        trade_mode = 1 if margin_type.upper() == "ISOLATED" else 0
        lev = str(min(int(leverage or self.max_leverage), self.max_leverage))
        body = {
            "category": "linear",
            "symbol": symbol,
            "tradeMode": trade_mode,
            "buyLeverage": lev,
            "sellLeverage": lev,
        }
        result = await self._request("POST", "/v5/position/switch-isolated", body)
        if result is not None:
            self.logger.info(f"Bybit set {symbol} margin to {margin_type}")
            return True
        return False

    async def get_balance(self) -> Optional[Dict]:
        """Get USDT balance (placeholder — separate ticket)."""
        self.logger.info("Bybit balance check (placeholder)")
        return {'balance': 0.0, 'available': 0.0}

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position (placeholder — separate ticket)."""
        return None

    async def open_long(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 3,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Open a long (Buy) market position with ISOLATED margin + leverage cap."""
        try:
            await self.set_leverage(symbol, leverage)
            await self.set_margin_type(symbol, "ISOLATED", leverage=leverage)
            body = {
                "category": "linear",
                "symbol": symbol,
                "side": "Buy",
                "orderType": "Market",
                "qty": str(quantity),
                "orderLinkId": f"cd-{int(time.time()*1000)}-{symbol[:6]}",
            }
            if reduce_only:
                body["reduceOnly"] = True
            result = await self._request("POST", "/v5/order/create", body)
            if result:
                self.logger.info(
                    f"✅ Opened LONG {symbol}: {quantity} @ {leverage}x leverage"
                )
                return result
            return None
        except Exception as e:
            self.logger.error(f"Bybit open_long error: {e}")
            return None

    async def open_short(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 3,
        reduce_only: bool = False,
    ) -> Optional[Dict]:
        """Open a short (Sell) market position with ISOLATED margin + leverage cap."""
        try:
            await self.set_leverage(symbol, leverage)
            await self.set_margin_type(symbol, "ISOLATED", leverage=leverage)
            body = {
                "category": "linear",
                "symbol": symbol,
                "side": "Sell",
                "orderType": "Market",
                "qty": str(quantity),
                "orderLinkId": f"cd-{int(time.time()*1000)}-{symbol[:6]}",
            }
            if reduce_only:
                body["reduceOnly"] = True
            result = await self._request("POST", "/v5/order/create", body)
            if result:
                self.logger.info(
                    f"✅ Opened SHORT {symbol}: {quantity} @ {leverage}x leverage"
                )
                return result
            return None
        except Exception as e:
            self.logger.error(f"Bybit open_short error: {e}")
            return None

    async def close_position(self, symbol: str) -> Optional[Dict]:
        """Close position (placeholder — separate ticket)."""
        self.logger.info(f"Bybit close {symbol} (placeholder)")
        return None

    async def get_all_positions(self) -> List[Dict]:
        """Get all positions (placeholder — separate ticket)."""
        return []
