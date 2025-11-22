"""
Binance Futures Executor - Integration with Binance Futures API

Handles:
- USDT-M perpetual futures
- COIN-M perpetual futures
- Position management (long/short)
- Leverage control
- Liquidation monitoring
- Funding rate tracking
"""

import asyncio
import logging
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import aiohttp


class BinanceFuturesExecutor:
    """
    Binance Futures API executor

    Supports:
    - USDT-M perpetuals (e.g., BTCUSDT, ETHUSDT)
    - COIN-M perpetuals (e.g., BTCUSD_PERP)
    - Long and short positions
    - Leverage up to 125x (configurable max)
    - Isolated and cross margin
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        max_leverage: int = 3
    ):
        """
        Initialize Binance Futures executor

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            max_leverage: Maximum leverage (default 3x for safety)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_leverage = max_leverage

        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"

        self.logger = logging.getLogger("BinanceFutures")
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    async def initialize(self) -> bool:
        """Initialize the executor"""
        try:
            self.session = aiohttp.ClientSession()

            # Test connection
            account_info = await self.get_account_info()
            if account_info:
                self.logger.info("✅ Binance Futures connected successfully")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to initialize Binance Futures: {e}")
            return False

    async def close(self):
        """Close the executor"""
        if self.session:
            await self.session.close()

    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Optional[Dict]:
        """
        Make API request

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request

        Returns:
            Optional[Dict]: Response data or None
        """
        try:
            # Rate limiting
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)

            params = params or {}
            headers = {"X-MBX-APIKEY": self.api_key}

            if signed:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._generate_signature(params)

            url = f"{self.base_url}{endpoint}"

            async with self.session.request(
                method,
                url,
                params=params,
                headers=headers
            ) as response:
                self.last_request_time = time.time()

                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"API error {response.status}: {error_text}")
                    return None

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

    async def get_account_info(self) -> Optional[Dict]:
        """Get account information and balances"""
        return await self._request('GET', '/fapi/v2/account', signed=True)

    async def get_balance(self) -> Optional[Dict]:
        """Get USDT balance and available margin"""
        try:
            account = await self.get_account_info()
            if not account:
                return None

            # Extract USDT balance
            for asset in account.get('assets', []):
                if asset['asset'] == 'USDT':
                    return {
                        'asset': 'USDT',
                        'balance': float(asset['walletBalance']),
                        'available': float(asset['availableBalance']),
                        'unrealized_pnl': float(asset['unrealizedProfit']),
                        'margin_balance': float(asset['marginBalance'])
                    }

            return None

        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            leverage: Leverage multiplier (1-125)

        Returns:
            bool: True if successful
        """
        try:
            # Clamp to max leverage
            leverage = min(leverage, self.max_leverage)

            result = await self._request(
                'POST',
                '/fapi/v1/leverage',
                params={'symbol': symbol, 'leverage': leverage},
                signed=True
            )

            if result:
                self.logger.info(f"Set {symbol} leverage to {leverage}x")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a symbol

        Args:
            symbol: Trading pair
            margin_type: 'ISOLATED' or 'CROSSED'

        Returns:
            bool: True if successful
        """
        try:
            result = await self._request(
                'POST',
                '/fapi/v1/marginType',
                params={'symbol': symbol, 'marginType': margin_type},
                signed=True
            )

            if result:
                self.logger.info(f"Set {symbol} margin type to {margin_type}")
                return True

            return False

        except Exception as e:
            # Margin type might already be set
            self.logger.debug(f"Margin type setting: {e}")
            return True

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol

        Returns:
            Optional[Dict]: Position info or None
        """
        try:
            positions = await self._request(
                'GET',
                '/fapi/v2/positionRisk',
                params={'symbol': symbol},
                signed=True
            )

            if positions and len(positions) > 0:
                pos = positions[0]
                position_amt = float(pos['positionAmt'])

                if position_amt == 0:
                    return None

                return {
                    'symbol': pos['symbol'],
                    'position_amt': position_amt,
                    'entry_price': float(pos['entryPrice']),
                    'mark_price': float(pos['markPrice']),
                    'unrealized_pnl': float(pos['unRealizedProfit']),
                    'leverage': int(pos['leverage']),
                    'liquidation_price': float(pos['liquidationPrice']),
                    'margin_type': pos['marginType'],
                    'side': 'LONG' if position_amt > 0 else 'SHORT',
                    'notional_value': abs(position_amt * float(pos['markPrice']))
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting position: {e}")
            return None

    async def open_long(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 3,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Open a long position (buy)

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            quantity: Amount to buy
            leverage: Leverage to use
            reduce_only: Only reduce existing position

        Returns:
            Optional[Dict]: Order result
        """
        try:
            # Set leverage
            await self.set_leverage(symbol, leverage)

            # Set isolated margin (safer)
            await self.set_margin_type(symbol, 'ISOLATED')

            # Place market order
            params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': quantity
            }

            if reduce_only:
                params['reduceOnly'] = 'true'

            result = await self._request(
                'POST',
                '/fapi/v1/order',
                params=params,
                signed=True
            )

            if result:
                self.logger.info(
                    f"✅ Opened LONG {symbol}: {quantity} @ {leverage}x leverage"
                )
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error opening long: {e}")
            return None

    async def open_short(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 3,
        reduce_only: bool = False
    ) -> Optional[Dict]:
        """
        Open a short position (sell)

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            quantity: Amount to sell
            leverage: Leverage to use
            reduce_only: Only reduce existing position

        Returns:
            Optional[Dict]: Order result
        """
        try:
            # Set leverage
            await self.set_leverage(symbol, leverage)

            # Set isolated margin (safer)
            await self.set_margin_type(symbol, 'ISOLATED')

            # Place market order
            params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': quantity
            }

            if reduce_only:
                params['reduceOnly'] = 'true'

            result = await self._request(
                'POST',
                '/fapi/v1/order',
                params=params,
                signed=True
            )

            if result:
                self.logger.info(
                    f"✅ Opened SHORT {symbol}: {quantity} @ {leverage}x leverage"
                )
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error opening short: {e}")
            return None

    async def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Close entire position for a symbol

        Args:
            symbol: Trading pair

        Returns:
            Optional[Dict]: Close order result
        """
        try:
            # Get current position
            position = await self.get_position(symbol)
            if not position:
                self.logger.warning(f"No position to close for {symbol}")
                return None

            # Determine close side (opposite of position)
            close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            quantity = abs(position['position_amt'])

            # Close with reduce-only market order
            result = await self._request(
                'POST',
                '/fapi/v1/order',
                params={
                    'symbol': symbol,
                    'side': close_side,
                    'type': 'MARKET',
                    'quantity': quantity,
                    'reduceOnly': 'true'
                },
                signed=True
            )

            if result:
                self.logger.info(f"✅ Closed position {symbol}: PnL ${position['unrealized_pnl']}")
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Get current and predicted funding rate

        Args:
            symbol: Trading pair

        Returns:
            Optional[Dict]: Funding rate info
        """
        try:
            result = await self._request(
                'GET',
                '/fapi/v1/fundingRate',
                params={'symbol': symbol, 'limit': 1}
            )

            if result and len(result) > 0:
                rate = result[0]
                return {
                    'symbol': rate['symbol'],
                    'funding_rate': float(rate['fundingRate']),
                    'funding_time': datetime.fromtimestamp(rate['fundingTime'] / 1000),
                    'mark_price': float(rate.get('markPrice', 0))
                }

            return None

        except Exception as e:
            self.logger.error(f"Error getting funding rate: {e}")
            return None

    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = await self._request(
                'GET',
                '/fapi/v2/positionRisk',
                signed=True
            )

            if not positions:
                return []

            # Filter only positions with non-zero amount
            active_positions = []
            for pos in positions:
                position_amt = float(pos['positionAmt'])
                if position_amt != 0:
                    active_positions.append({
                        'symbol': pos['symbol'],
                        'position_amt': position_amt,
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'leverage': int(pos['leverage']),
                        'liquidation_price': float(pos['liquidationPrice']),
                        'margin_type': pos['marginType'],
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'notional_value': abs(position_amt * float(pos['markPrice']))
                    })

            return active_positions

        except Exception as e:
            self.logger.error(f"Error getting all positions: {e}")
            return []

    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        side: str
    ) -> Optional[Dict]:
        """
        Set stop loss for a position

        Args:
            symbol: Trading pair
            stop_price: Price to trigger stop loss
            side: 'BUY' for short positions, 'SELL' for long positions

        Returns:
            Optional[Dict]: Order result
        """
        try:
            result = await self._request(
                'POST',
                '/fapi/v1/order',
                params={
                    'symbol': symbol,
                    'side': side,
                    'type': 'STOP_MARKET',
                    'stopPrice': stop_price,
                    'closePosition': 'true'
                },
                signed=True
            )

            if result:
                self.logger.info(f"✅ Set stop loss for {symbol} at ${stop_price}")
                return result

            return None

        except Exception as e:
            self.logger.error(f"Error setting stop loss: {e}")
            return None

    async def get_mark_price(self, symbol: str) -> Optional[float]:
        """Get current mark price for a symbol"""
        try:
            result = await self._request(
                'GET',
                '/fapi/v1/premiumIndex',
                params={'symbol': symbol}
            )

            if result:
                return float(result['markPrice'])

            return None

        except Exception as e:
            self.logger.error(f"Error getting mark price: {e}")
            return None
