"""
Module Manager - Orchestrates all trading modules

Manages lifecycle, coordination, and communication between trading modules:
- DEX Trading
- Futures Trading
- Arbitrage
- Future modules

Features:
- Module enable/disable
- Health monitoring
- Inter-module communication
- Shared risk management
- Unified portfolio view
- Capital allocation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from modules.base_module import BaseModule, ModuleType, ModuleStatus, ModuleConfig


class ModuleManager:
    """
    Module Manager - Central orchestration for all trading modules

    Responsibilities:
    - Module registration and lifecycle management
    - Health monitoring and auto-restart
    - Risk coordination across modules
    - Capital allocation management
    - Inter-module communication
    - Unified metrics aggregation
    """

    def __init__(
        self,
        config: Dict,
        db_manager=None,
        cache_manager=None,
        alert_manager=None,
        risk_manager=None
    ):
        """
        Initialize module manager

        Args:
            config: Configuration dictionary
            db_manager: Database manager instance
            cache_manager: Cache manager instance
            alert_manager: Alert manager instance
            risk_manager: Shared risk manager instance
        """
        self.config = config
        self.db = db_manager
        self.cache = cache_manager
        self.alerts = alert_manager
        self.risk_manager = risk_manager

        self.logger = logging.getLogger("ModuleManager")

        # Module registry
        self.modules: Dict[str, BaseModule] = {}
        self.module_configs: Dict[str, ModuleConfig] = {}

        # State tracking
        self.total_capital = config.get('total_capital', 1000.0)
        self.allocated_capital = 0.0
        self.available_capital = self.total_capital

        # Monitoring
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_update_task: Optional[asyncio.Task] = None

        self._lock = asyncio.Lock()

        self.logger.info("ModuleManager initialized")

    async def initialize(self) -> bool:
        """
        Initialize module manager

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing ModuleManager...")

            # Load module configurations
            await self._load_module_configs()

            # Initialize registered modules
            for module_name, module in self.modules.items():
                if module.is_enabled:
                    self.logger.info(f"Initializing module: {module_name}")
                    success = await module.initialize()
                    if not success:
                        self.logger.error(f"Failed to initialize module: {module_name}")
                        return False

            self.logger.info("ModuleManager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"ModuleManager initialization failed: {e}", exc_info=True)
            return False

    async def start(self) -> bool:
        """
        Start module manager and all enabled modules

        Returns:
            bool: True if started successfully
        """
        try:
            self.logger.info("Starting ModuleManager...")
            self._running = True

            # Start all enabled modules
            for module_name, module in self.modules.items():
                if module.is_enabled:
                    self.logger.info(f"Starting module: {module_name}")
                    success = await module.start()
                    if not success:
                        self.logger.error(f"Failed to start module: {module_name}")
                    else:
                        self.logger.info(f"Module started: {module_name}")

            # Start monitoring tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_update_task = asyncio.create_task(self._metrics_update_loop())

            self.logger.info("ModuleManager started successfully")
            return True

        except Exception as e:
            self.logger.error(f"ModuleManager start failed: {e}", exc_info=True)
            return False

    async def stop(self) -> bool:
        """
        Stop module manager and all modules

        Returns:
            bool: True if stopped successfully
        """
        try:
            self.logger.info("Stopping ModuleManager...")
            self._running = False

            # Cancel monitoring tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._metrics_update_task:
                self._metrics_update_task.cancel()

            # Stop all modules
            for module_name, module in self.modules.items():
                if module.is_running:
                    self.logger.info(f"Stopping module: {module_name}")
                    await module.stop()

            self.logger.info("ModuleManager stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"ModuleManager stop failed: {e}", exc_info=True)
            return False

    def register_module(self, module: BaseModule):
        """
        Register a trading module

        Args:
            module: Module instance to register
        """
        module_name = module.name
        if module_name in self.modules:
            self.logger.warning(f"Module {module_name} already registered, replacing")

        self.modules[module_name] = module
        self.module_configs[module_name] = module.config

        # Update capital allocation
        self.allocated_capital += module.config.capital_allocation
        self.available_capital = self.total_capital - self.allocated_capital

        self.logger.info(
            f"Module registered: {module_name} "
            f"(type: {module.module_type.value}, "
            f"capital: ${module.config.capital_allocation})"
        )

    def unregister_module(self, module_name: str):
        """
        Unregister a trading module

        Args:
            module_name: Name of module to unregister
        """
        if module_name in self.modules:
            module = self.modules[module_name]

            # Free up capital
            self.allocated_capital -= module.config.capital_allocation
            self.available_capital = self.total_capital - self.allocated_capital

            del self.modules[module_name]
            del self.module_configs[module_name]

            self.logger.info(f"Module unregistered: {module_name}")

    async def enable_module(self, module_name: str) -> bool:
        """
        Enable a module

        Args:
            module_name: Name of module to enable

        Returns:
            bool: True if enabled successfully
        """
        try:
            if module_name not in self.modules:
                self.logger.error(f"Module not found: {module_name}")
                return False

            module = self.modules[module_name]
            await module.enable()

            # Start module if manager is running
            if self._running:
                await module.initialize()
                await module.start()

            self.logger.info(f"Module enabled: {module_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable module {module_name}: {e}")
            return False

    async def disable_module(self, module_name: str) -> bool:
        """
        Disable a module

        Args:
            module_name: Name of module to disable

        Returns:
            bool: True if disabled successfully
        """
        try:
            if module_name not in self.modules:
                self.logger.error(f"Module not found: {module_name}")
                return False

            module = self.modules[module_name]
            await module.disable()

            self.logger.info(f"Module disabled: {module_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to disable module {module_name}: {e}")
            return False

    async def pause_module(self, module_name: str) -> bool:
        """Pause a module"""
        if module_name in self.modules:
            await self.modules[module_name].pause()
            return True
        return False

    async def resume_module(self, module_name: str) -> bool:
        """Resume a module"""
        if module_name in self.modules:
            await self.modules[module_name].resume()
            return True
        return False

    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """
        Get a module by name

        Args:
            module_name: Name of module

        Returns:
            Optional[BaseModule]: Module instance or None
        """
        return self.modules.get(module_name)

    def get_all_modules(self) -> List[BaseModule]:
        """
        Get all registered modules

        Returns:
            List[BaseModule]: List of all modules
        """
        return list(self.modules.values())

    def get_active_modules(self) -> List[BaseModule]:
        """
        Get all active (running) modules

        Returns:
            List[BaseModule]: List of active modules
        """
        return [m for m in self.modules.values() if m.is_running]

    def get_modules_by_type(self, module_type: ModuleType) -> List[BaseModule]:
        """
        Get all modules of a specific type

        Returns:
            List[BaseModule]: List of modules
        """
        return [m for m in self.modules.values() if m.module_type == module_type]

    async def get_all_positions(self) -> List[Dict]:
        """
        Get positions from all modules

        Returns:
            List[Dict]: Combined positions from all modules
        """
        all_positions = []
        for module in self.modules.values():
            try:
                positions = await module.get_positions()
                # Add module info to each position
                for pos in positions:
                    pos['module'] = module.name
                    pos['module_type'] = module.module_type.value
                all_positions.extend(positions)
            except Exception as e:
                self.logger.error(f"Error getting positions from {module.name}: {e}")

        return all_positions

    async def get_aggregated_metrics(self) -> Dict:
        """
        Get aggregated metrics across all modules

        Returns:
            Dict: Aggregated metrics
        """
        metrics = {
            'total_capital': self.total_capital,
            'allocated_capital': self.allocated_capital,
            'available_capital': self.available_capital,
            'modules_count': len(self.modules),
            'active_modules': len(self.get_active_modules()),
            'total_positions': 0,
            'total_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'overall_win_rate': 0.0,
            'modules': {}
        }

        # Aggregate metrics from all modules
        for module_name, module in self.modules.items():
            try:
                module_metrics = await module.get_metrics()
                metrics['modules'][module_name] = module_metrics.to_dict()

                # Aggregate totals
                metrics['total_positions'] += module_metrics.active_positions
                metrics['total_pnl'] += module_metrics.total_pnl
                metrics['unrealized_pnl'] += module_metrics.unrealized_pnl
                metrics['realized_pnl'] += module_metrics.realized_pnl
                metrics['total_trades'] += module_metrics.total_trades
                metrics['winning_trades'] += module_metrics.winning_trades
                metrics['losing_trades'] += module_metrics.losing_trades

            except Exception as e:
                self.logger.error(f"Error getting metrics from {module_name}: {e}")

        # Calculate overall win rate
        if metrics['total_trades'] > 0:
            metrics['overall_win_rate'] = metrics['winning_trades'] / metrics['total_trades']

        return metrics

    async def reallocate_capital(self, allocations: Dict[str, float]) -> bool:
        """
        Reallocate capital across modules

        Args:
            allocations: Dict mapping module_name to new capital allocation

        Returns:
            bool: True if reallocation successful
        """
        try:
            async with self._lock:
                # Validate total allocation
                total_allocation = sum(allocations.values())
                if total_allocation > self.total_capital:
                    self.logger.error(
                        f"Total allocation ${total_allocation} exceeds "
                        f"total capital ${self.total_capital}"
                    )
                    return False

                # Update allocations
                for module_name, allocation in allocations.items():
                    if module_name in self.modules:
                        module = self.modules[module_name]
                        old_allocation = module.config.capital_allocation
                        module.config.capital_allocation = allocation
                        module.metrics.capital_allocated = allocation

                        self.logger.info(
                            f"Module {module_name} capital: "
                            f"${old_allocation} -> ${allocation}"
                        )

                # Update totals
                self.allocated_capital = total_allocation
                self.available_capital = self.total_capital - self.allocated_capital

                self.logger.info(f"Capital reallocation complete")
                return True

        except Exception as e:
            self.logger.error(f"Capital reallocation failed: {e}")
            return False

    async def _health_check_loop(self):
        """Health check monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for module_name, module in self.modules.items():
                    if not module.is_enabled:
                        continue

                    healthy = await module.health_check()
                    if not healthy:
                        self.logger.error(f"Health check failed for module: {module_name}")

                        if self.alerts:
                            await self.alerts.send_alert(
                                title=f"Module Health Check Failed",
                                message=f"Module {module_name} failed health check",
                                severity="high"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")

    async def _metrics_update_loop(self):
        """Metrics update loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Update every minute

                for module in self.modules.values():
                    if module.is_running:
                        await module.update_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics update loop error: {e}")

    async def _load_module_configs(self):
        """Load module configurations from config files"""
        try:
            # This will be implemented to load from config/modules/*.yaml
            # For now, we'll use defaults from main config
            pass
        except Exception as e:
            self.logger.error(f"Error loading module configs: {e}")

    def get_status_summary(self) -> Dict:
        """
        Get status summary of all modules

        Returns:
            Dict: Status summary
        """
        return {
            'manager_running': self._running,
            'total_modules': len(self.modules),
            'enabled_modules': len([m for m in self.modules.values() if m.is_enabled]),
            'running_modules': len(self.get_active_modules()),
            'total_capital': self.total_capital,
            'allocated_capital': self.allocated_capital,
            'available_capital': self.available_capital,
            'modules': {
                name: module.get_status()
                for name, module in self.modules.items()
            }
        }
