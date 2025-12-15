#!/usr/bin/env python3
"""
Portfolio-Level Risk Manager
Provides a holistic view of risk across all trading modules.
"""

import asyncio
import logging
from typing import Dict, Any

from core.global_event_bus import GlobalEventBus

logger = logging.getLogger(__name__)

class PortfolioLevelRiskManager:
    """
    Analyzes risk across all active positions in all modules.
    """

    def __init__(self, global_event_bus: GlobalEventBus, config: Dict[str, Any]):
        self.bus = global_event_bus
        self.config = config
        self.positions: Dict[str, Dict[str, Any]] = {} # module -> {symbol -> position_data}
        self.max_total_exposure = config.get("max_total_exposure_usd", 1000.0)
        self.max_exposure_per_asset = config.get("max_exposure_per_asset_usd", 200.0)

    async def initialize(self):
        """
        Subscribe to position update events from all modules.
        """
        await self.bus.subscribe("POSITION_UPDATE", self.handle_position_update)
        await self.bus.subscribe("POSITION_CLOSE", self.handle_position_close)
        logger.info("Portfolio-Level Risk Manager initialized and subscribed to position channels.")
        asyncio.create_task(self.monitor_risk())

    def handle_position_update(self, data: Dict[str, Any]):
        """
        Callback for when a position is opened or updated.
        """
        module = data.get("module")
        symbol = data.get("symbol")
        if module and symbol:
            if module not in self.positions:
                self.positions[module] = {}
            self.positions[module][symbol] = data

    def handle_position_close(self, data: Dict[str, Any]):
        """
        Callback for when a position is closed.
        """
        module = data.get("module")
        symbol = data.get("symbol")
        if module and symbol and module in self.positions and symbol in self.positions[module]:
            del self.positions[module][symbol]

    def get_total_exposure(self) -> float:
        """
        Calculate the total notional exposure across all modules.
        """
        total_exposure = 0.0
        for module_positions in self.positions.values():
            for position in module_positions.values():
                total_exposure += position.get("notional_value", 0.0)
        return total_exposure

    def get_exposure_per_asset(self) -> Dict[str, float]:
        """
        Calculate the total notional exposure for each asset across all modules.
        """
        exposure_per_asset: Dict[str, float] = {}
        for module_positions in self.positions.values():
            for position in module_positions.values():
                symbol = position.get("symbol")
                notional = position.get("notional_value", 0.0)
                if symbol:
                    exposure_per_asset[symbol] = exposure_per_asset.get(symbol, 0.0) + notional
        return exposure_per_asset

    async def monitor_risk(self):
        """
        Periodically monitor portfolio-level risk and take action if limits are breached.
        """
        while True:
            await asyncio.sleep(self.config.get("risk_check_interval_seconds", 60))

            total_exposure = self.get_total_exposure()
            if total_exposure > self.max_total_exposure:
                logger.warning(f"Total exposure ({total_exposure}) exceeds limit ({self.max_total_exposure}). Publishing risk mitigation event.")
                await self.bus.publish("RISK_MITIGATION", {
                    "type": "TOTAL_EXPOSURE_BREACH",
                    "current_exposure": total_exposure,
                    "limit": self.max_total_exposure
                })

            exposure_per_asset = self.get_exposure_per_asset()
            for symbol, exposure in exposure_per_asset.items():
                if exposure > self.max_exposure_per_asset:
                    logger.warning(f"Exposure for {symbol} ({exposure}) exceeds limit ({self.max_exposure_per_asset}). Publishing risk mitigation event.")
                    await self.bus.publish("RISK_MITIGATION", {
                        "type": "ASSET_EXPOSURE_BREACH",
                        "symbol": symbol,
                        "current_exposure": exposure,
                        "limit": self.max_exposure_per_asset
                    })
