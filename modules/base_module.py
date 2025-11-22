"""
Base Module - Abstract base class for all trading modules

Each trading module (DEX, Futures, Arbitrage) inherits from this base class
and implements its own trading logic while sharing common infrastructure.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ModuleType(Enum):
    """Types of trading modules"""
    DEX_TRADING = "dex_trading"
    FUTURES_TRADING = "futures_trading"
    ARBITRAGE = "arbitrage"
    LIQUIDITY_PROVISION = "liquidity_provision"
    SOLANA_STRATEGIES = "solana_strategies"
    CUSTOM = "custom"


class ModuleStatus(Enum):
    """Module operational status"""
    DISABLED = "disabled"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ModuleMetrics:
    """Performance metrics for a module"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    active_positions: int = 0
    capital_allocated: float = 0.0
    capital_used: float = 0.0
    last_trade_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    errors_count: int = 0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'active_positions': self.active_positions,
            'capital_allocated': self.capital_allocated,
            'capital_used': self.capital_used,
            'capital_available': self.capital_allocated - self.capital_used,
            'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'uptime_seconds': self.uptime_seconds,
            'errors_count': self.errors_count
        }


@dataclass
class ModuleConfig:
    """Configuration for a trading module"""
    name: str
    module_type: ModuleType
    enabled: bool = True
    capital_allocation: float = 0.0
    max_positions: int = 5
    max_position_size: float = 0.0
    risk_per_trade: float = 0.02  # 2% risk per trade
    stop_loss_pct: float = 0.05   # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    wallet_addresses: Dict[str, str] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    strategies: List[str] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'name': self.name,
            'module_type': self.module_type.value,
            'enabled': self.enabled,
            'capital_allocation': self.capital_allocation,
            'max_positions': self.max_positions,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'wallet_addresses': self.wallet_addresses,
            'strategies': self.strategies,
            'custom_settings': self.custom_settings
        }


class BaseModule(ABC):
    """
    Abstract base class for all trading modules

    Each module must implement:
    - initialize(): Setup module resources
    - start(): Start module operations
    - stop(): Stop module operations
    - process_opportunity(): Handle trading opportunities
    - get_positions(): Return current positions
    - get_metrics(): Return performance metrics
    """

    def __init__(
        self,
        config: ModuleConfig,
        db_manager=None,
        cache_manager=None,
        alert_manager=None
    ):
        """
        Initialize base module

        Args:
            config: Module configuration
            db_manager: Database manager instance
            cache_manager: Cache manager instance
            alert_manager: Alert manager instance
        """
        self.config = config
        self.db = db_manager
        self.cache = cache_manager
        self.alerts = alert_manager

        self.logger = logging.getLogger(f"Module.{config.name}")
        self.status = ModuleStatus.DISABLED
        self.metrics = ModuleMetrics(capital_allocated=config.capital_allocation)

        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        """Check if module is enabled"""
        return self.config.enabled

    @property
    def is_running(self) -> bool:
        """Check if module is running"""
        return self._running and self.status == ModuleStatus.RUNNING

    @property
    def name(self) -> str:
        """Get module name"""
        return self.config.name

    @property
    def module_type(self) -> ModuleType:
        """Get module type"""
        return self.config.module_type

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize module resources

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    async def start(self) -> bool:
        """
        Start module operations

        Returns:
            bool: True if started successfully
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop module operations

        Returns:
            bool: True if stopped successfully
        """
        pass

    @abstractmethod
    async def process_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        """
        Process a trading opportunity

        Args:
            opportunity: Trading opportunity data

        Returns:
            Optional[Dict]: Trade result or None
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """
        Get current open positions

        Returns:
            List[Dict]: List of open positions
        """
        pass

    @abstractmethod
    async def get_metrics(self) -> ModuleMetrics:
        """
        Get module performance metrics

        Returns:
            ModuleMetrics: Current metrics
        """
        pass

    async def update_metrics(self):
        """Update module metrics (called periodically)"""
        try:
            positions = await self.get_positions()
            self.metrics.active_positions = len(positions)

            # Calculate capital used
            capital_used = sum(p.get('cost', 0) for p in positions)
            self.metrics.capital_used = capital_used

            # Update uptime
            if self.start_time:
                self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            # Calculate win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Calculate profit factor
            total_wins = self.metrics.winning_trades * abs(self.metrics.realized_pnl / max(self.metrics.total_trades, 1))
            total_losses = self.metrics.losing_trades * abs(self.metrics.realized_pnl / max(self.metrics.total_trades, 1))
            if total_losses > 0:
                self.metrics.profit_factor = total_wins / total_losses

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    async def enable(self):
        """Enable the module"""
        async with self._lock:
            self.config.enabled = True
            self.logger.info(f"Module {self.name} enabled")

    async def disable(self):
        """Disable the module"""
        async with self._lock:
            self.config.enabled = False
            if self.is_running:
                await self.stop()
            self.logger.info(f"Module {self.name} disabled")

    async def pause(self):
        """Pause module operations"""
        async with self._lock:
            if self.status == ModuleStatus.RUNNING:
                self.status = ModuleStatus.PAUSED
                self.logger.info(f"Module {self.name} paused")

    async def resume(self):
        """Resume module operations"""
        async with self._lock:
            if self.status == ModuleStatus.PAUSED:
                self.status = ModuleStatus.RUNNING
                self.logger.info(f"Module {self.name} resumed")

    def get_status(self) -> Dict:
        """
        Get module status information

        Returns:
            Dict: Status information
        """
        return {
            'name': self.name,
            'module_type': self.module_type.value,
            'status': self.status.value,
            'enabled': self.is_enabled,
            'running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stop_time': self.stop_time.isoformat() if self.stop_time else None,
            'error_message': self.error_message,
            'metrics': self.metrics.to_dict(),
            'config': self.config.to_dict()
        }

    async def health_check(self) -> bool:
        """
        Perform health check on module

        Returns:
            bool: True if healthy
        """
        try:
            # Basic health checks
            if not self.is_enabled:
                return True  # Disabled modules are considered healthy

            if self.status == ModuleStatus.ERROR:
                return False

            # Check if module is responsive
            if self.is_running:
                positions = await asyncio.wait_for(self.get_positions(), timeout=5.0)
                return True

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def _handle_error(self, error: Exception):
        """
        Handle module error

        Args:
            error: Exception that occurred
        """
        self.status = ModuleStatus.ERROR
        self.error_message = str(error)
        self.metrics.errors_count += 1
        self.logger.error(f"Module error: {error}", exc_info=True)

        if self.alerts:
            asyncio.create_task(self.alerts.send_alert(
                title=f"Module Error: {self.name}",
                message=str(error),
                severity="high"
            ))
