"""
DEX Trading Module - Implementation

Wraps existing DEX trading functionality into a modular component
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from modules.base_module import (
    BaseModule,
    ModuleConfig,
    ModuleType,
    ModuleStatus,
    ModuleMetrics
)


class DexTradingModule(BaseModule):
    """
    DEX Trading Module

    Handles spot trading on decentralized exchanges:
    - Solana: Jupiter, Raydium, Orca
    - EVM: Uniswap, SushiSwap

    Strategies:
    - Momentum Trading
    - Scalping
    - AI-powered decision making
    """

    def __init__(
        self,
        config: ModuleConfig,
        trading_engine=None,
        db_manager=None,
        cache_manager=None,
        alert_manager=None
    ):
        """
        Initialize DEX Trading Module

        Args:
            config: Module configuration
            trading_engine: Reference to main trading engine
            db_manager: Database manager instance
            cache_manager: Cache manager instance
            alert_manager: Alert manager instance
        """
        super().__init__(config, db_manager, cache_manager, alert_manager)

        self.engine = trading_engine
        self.logger = logging.getLogger("Module.DexTrading")

        # DEX-specific components (will be set during initialization)
        self.portfolio_manager = None
        self.strategy_manager = None
        self.position_tracker = None
        self.order_manager = None

    async def initialize(self) -> bool:
        """
        Initialize DEX trading module

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing DEX Trading Module...")
            self.status = ModuleStatus.INITIALIZING

            # Get references from engine if available
            if self.engine:
                self.portfolio_manager = self.engine.portfolio_manager
                self.strategy_manager = self.engine.strategy_manager
                self.position_tracker = self.engine.position_tracker
                self.order_manager = self.engine.order_manager

            # Validate wallet configuration
            if 'solana' not in self.config.wallet_addresses:
                self.logger.warning("No Solana wallet configured for DEX trading")

            # Set up DEX-specific settings
            self._setup_dex_settings()

            self.logger.info("DEX Trading Module initialized successfully")
            self.status = ModuleStatus.STOPPED
            return True

        except Exception as e:
            self.logger.error(f"DEX Trading Module initialization failed: {e}", exc_info=True)
            self._handle_error(e)
            return False

    async def start(self) -> bool:
        """
        Start DEX trading operations

        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting DEX Trading Module...")

            if not self.is_enabled:
                self.logger.warning("Module is disabled, cannot start")
                return False

            self.status = ModuleStatus.RUNNING
            self._running = True
            self.start_time = datetime.now()

            # Start DEX-specific tasks
            # Note: The actual trading is handled by the main engine
            # This module primarily manages DEX-specific configuration and metrics

            self.logger.info("DEX Trading Module started successfully")
            return True

        except Exception as e:
            self.logger.error(f"DEX Trading Module start failed: {e}", exc_info=True)
            self._handle_error(e)
            return False

    async def stop(self) -> bool:
        """
        Stop DEX trading operations

        Returns:
            bool: True if stopped successfully
        """
        try:
            self.logger.info("Stopping DEX Trading Module...")

            self._running = False
            self.status = ModuleStatus.STOPPING

            # Cancel any module-specific tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks.clear()

            self.status = ModuleStatus.STOPPED
            self.stop_time = datetime.now()

            self.logger.info("DEX Trading Module stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"DEX Trading Module stop failed: {e}", exc_info=True)
            return False

    async def process_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """
        Process a DEX trading opportunity

        Args:
            opportunity: Trading opportunity data

        Returns:
            Optional[Dict]: Trade result or None
        """
        try:
            if not self.is_running:
                return None

            # Validate opportunity is for DEX trading
            chain = opportunity.get('chain', '').lower()
            if chain not in ['solana', 'ethereum', 'polygon', 'bsc', 'arbitrum', 'base']:
                return None

            # Check if we have capital available
            if self.metrics.capital_used >= self.config.capital_allocation:
                self.logger.warning("DEX module capital fully allocated")
                return None

            # Check position limits
            if self.metrics.active_positions >= self.config.max_positions:
                self.logger.warning("DEX module max positions reached")
                return None

            # Opportunity will be processed by the main engine
            # This is just validation at module level
            self.logger.debug(f"DEX opportunity validated: {opportunity.get('token_address')}")

            return opportunity

        except Exception as e:
            self.logger.error(f"Error processing DEX opportunity: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        """
        Get current DEX trading positions

        Returns:
            List[Dict]: List of open DEX positions
        """
        try:
            if not self.position_tracker:
                return []

            # Get all positions and filter for DEX chains
            all_positions = self.position_tracker.get_open_positions()

            dex_positions = []
            for pos in all_positions:
                chain = pos.chain.lower()
                if chain in ['solana', 'ethereum', 'polygon', 'bsc', 'arbitrum', 'base']:
                    dex_positions.append({
                        'id': pos.id,
                        'token_address': pos.token_address,
                        'pair_address': pos.pair_address,
                        'chain': pos.chain,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'size': pos.size,
                        'cost': pos.cost,
                        'pnl': pos.pnl,
                        'pnl_percentage': pos.pnl_percentage,
                        'strategy': pos.strategy,
                        'entry_time': pos.entry_time.isoformat(),
                        'age_seconds': pos.age.total_seconds(),
                        'status': pos.status
                    })

            return dex_positions

        except Exception as e:
            self.logger.error(f"Error getting DEX positions: {e}")
            return []

    async def get_metrics(self) -> ModuleMetrics:
        """
        Get DEX trading module metrics

        Returns:
            ModuleMetrics: Current metrics
        """
        try:
            # Update positions count
            positions = await self.get_positions()
            self.metrics.active_positions = len(positions)

            # Calculate capital used
            self.metrics.capital_used = sum(p['cost'] for p in positions)

            # Get trade history from database if available
            if self.db:
                # Query DEX trades
                trades = await self._get_trade_history()
                self.metrics.total_trades = len(trades)

                # Calculate win/loss
                winning = [t for t in trades if t.get('pnl', 0) > 0]
                losing = [t for t in trades if t.get('pnl', 0) < 0]

                self.metrics.winning_trades = len(winning)
                self.metrics.losing_trades = len(losing)

                # Calculate PnL
                self.metrics.realized_pnl = sum(t.get('pnl', 0) for t in trades)

                # Calculate unrealized PnL
                self.metrics.unrealized_pnl = sum(p.get('pnl', 0) for p in positions)
                self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl

            # Update uptime
            if self.start_time:
                self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            return self.metrics

        except Exception as e:
            self.logger.error(f"Error getting DEX metrics: {e}")
            return self.metrics

    async def _get_trade_history(self) -> List[Dict]:
        """Get trade history for DEX trades"""
        try:
            if not self.db:
                return []

            # Query trades for DEX chains
            query = """
                SELECT * FROM trades
                WHERE chain IN ('solana', 'ethereum', 'polygon', 'bsc', 'arbitrum', 'base')
                ORDER BY entry_timestamp DESC
                LIMIT 1000
            """

            result = await self.db.fetch_all(query)
            return [dict(row) for row in result]

        except Exception as e:
            self.logger.error(f"Error fetching DEX trade history: {e}")
            return []

    def _setup_dex_settings(self):
        """Setup DEX-specific settings"""
        try:
            # Get DEX-specific settings from config
            custom = self.config.custom_settings

            # Slippage settings
            self.max_slippage_bps = custom.get('max_slippage_bps', 50)  # 0.5%

            # Gas settings
            self.max_gas_price = custom.get('max_gas_price', 50)  # 50 Gwei

            # MEV protection
            self.mev_protection_enabled = custom.get('mev_protection', True)

            # Route optimization
            self.use_jupiter_routing = custom.get('jupiter_routing', True)

            # Supported DEXs
            self.supported_dexs = custom.get('supported_dexs', [
                'raydium', 'orca', 'jupiter',  # Solana
                'uniswap', 'sushiswap'  # EVM
            ])

            self.logger.info(
                f"DEX settings configured: "
                f"slippage={self.max_slippage_bps}bps, "
                f"max_gas={self.max_gas_price}gwei"
            )

        except Exception as e:
            self.logger.error(f"Error setting up DEX settings: {e}")

    async def get_supported_chains(self) -> List[str]:
        """Get list of supported chains for DEX trading"""
        return ['solana', 'ethereum', 'polygon', 'bsc', 'arbitrum', 'base']

    async def get_supported_dexs(self) -> List[str]:
        """Get list of supported DEXs"""
        return self.supported_dexs

    def get_module_info(self) -> Dict:
        """Get detailed module information"""
        return {
            'name': self.name,
            'type': self.module_type.value,
            'version': '1.0.0',
            'description': 'DEX spot trading module',
            'supported_chains': ['solana', 'ethereum', 'polygon', 'bsc', 'arbitrum', 'base'],
            'supported_dexs': self.supported_dexs,
            'strategies': self.config.strategies,
            'features': [
                'Jupiter aggregator integration',
                'MEV protection',
                'Dynamic route optimization',
                'Multi-chain support',
                'Momentum trading',
                'Scalping strategy',
                'AI-powered decisions'
            ]
        }
