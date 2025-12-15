#!/usr/bin/env python3
"""
Cross-Exchange Arbitrage Strategy
"""

import asyncio
import logging
from typing import Dict, Any

from core.global_event_bus import GlobalEventBus

logger = logging.getLogger(__name__)

class ArbitrageStrategy:
    """
    Identifies and acts on arbitrage opportunities between DEX and CEX markets.
    """

    def __init__(self, global_event_bus: GlobalEventBus, config: Dict[str, Any]):
        self.bus = global_event_bus
        self.config = config
        self.dex_prices: Dict[str, float] = {}
        self.cex_prices: Dict[str, float] = {}
        self.min_profit_threshold = config.get("min_profit_threshold_pct", 0.01) # Default 1%

    async def initialize(self):
        """
        Subscribe to the necessary event channels for price updates.
        """
        await self.bus.subscribe("DEX_PRICE_UPDATE", self.handle_dex_price_update)
        await self.bus.subscribe("CEX_PRICE_UPDATE", self.handle_cex_price_update)
        logger.info("Arbitrage Strategy initialized and subscribed to price update channels.")

    async def handle_dex_price_update(self, data: Dict[str, Any]):
        """
        Callback for when a new DEX price is received.
        """
        symbol = data.get("symbol")
        price = data.get("price")
        if symbol and price:
            self.dex_prices[symbol] = float(price)
            await self.check_for_opportunity(symbol)

    async def handle_cex_price_update(self, data: Dict[str, Any]):
        """
        Callback for when a new CEX price is received.
        """
        symbol = data.get("symbol")
        price = data.get("price")
        if symbol and price:
            self.cex_prices[symbol] = float(price)
            await self.check_for_opportunity(symbol)

    async def check_for_opportunity(self, symbol: str):
        """
        Check for an arbitrage opportunity for a given symbol.
        """
        dex_price = self.dex_prices.get(symbol)
        cex_price = self.cex_prices.get(symbol)

        if dex_price and cex_price:
            # Opportunity: Buy on DEX, Sell on CEX
            if cex_price > dex_price:
                profit_pct = (cex_price - dex_price) / dex_price
                if profit_pct >= self.min_profit_threshold:
                    logger.info(f"Arbitrage opportunity found for {symbol}: Buy DEX @ {dex_price}, Sell CEX @ {cex_price} (Profit: {profit_pct:.2%})")
                    await self.bus.publish("ARBITRAGE_OPPORTUNITY", {
                        "symbol": symbol,
                        "buy_market": "DEX",
                        "sell_market": "CEX",
                        "buy_price": dex_price,
                        "sell_price": cex_price,
                        "profit_pct": profit_pct
                    })

            # Opportunity: Buy on CEX, Sell on DEX
            elif dex_price > cex_price:
                profit_pct = (dex_price - cex_price) / cex_price
                if profit_pct >= self.min_profit_threshold:
                    logger.info(f"Arbitrage opportunity found for {symbol}: Buy CEX @ {cex_price}, Sell DEX @ {dex_price} (Profit: {profit_pct:.2%})")
                    await self.bus.publish("ARBITRAGE_OPPORTUNITY", {
                        "symbol": symbol,
                        "buy_market": "CEX",
                        "sell_market": "DEX",
                        "buy_price": cex_price,
                        "sell_price": dex_price,
                        "profit_pct": profit_pct
                    })
