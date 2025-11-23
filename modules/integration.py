"""
Module Integration Helper

Helps integrate the modular architecture with the existing trading engine
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

from core.module_manager import ModuleManager
from modules.base_module import ModuleConfig, ModuleType

# Conditional imports for modules that may not exist yet
try:
    from modules.dex_trading import DexTradingModule
except ImportError:
    DexTradingModule = None

try:
    from modules.futures_trading import FuturesTradingModule
except ImportError:
    FuturesTradingModule = None

try:
    from modules.solana_strategies import SolanaStrategiesModule
except ImportError:
    SolanaStrategiesModule = None


logger = logging.getLogger(__name__)


class ModuleIntegration:
    """
    Helper class to integrate modules with the trading bot

    Usage in main.py:
        integration = ModuleIntegration(engine, db, cache, alerts)
        module_manager = await integration.setup_modules()
        engine.module_manager = module_manager
    """

    def __init__(
        self,
        trading_engine=None,
        db_manager=None,
        cache_manager=None,
        alert_manager=None,
        risk_manager=None,
        config_dir: str = "config"
    ):
        """
        Initialize module integration

        Args:
            trading_engine: TradingBotEngine instance
            db_manager: Database manager
            cache_manager: Cache manager
            alert_manager: Alert manager
            risk_manager: Risk manager
            config_dir: Configuration directory
        """
        self.engine = trading_engine
        self.db = db_manager
        self.cache = cache_manager
        self.alerts = alert_manager
        self.risk = risk_manager
        self.config_dir = Path(config_dir)

        self.logger = logging.getLogger("ModuleIntegration")

    async def setup_modules(self) -> ModuleManager:
        """
        Setup and configure all trading modules

        Returns:
            ModuleManager: Configured module manager
        """
        try:
            self.logger.info("Setting up trading modules...")

            # Load module configurations
            module_configs = self._load_module_configs()

            # Create module manager
            manager_config = {
                'total_capital': self._get_total_capital(module_configs)
            }

            module_manager = ModuleManager(
                config=manager_config,
                db_manager=self.db,
                cache_manager=self.cache,
                alert_manager=self.alerts,
                risk_manager=self.risk
            )

            # Initialize module manager
            await module_manager.initialize()

            # Register DEX trading module
            if 'dex_trading' in module_configs and DexTradingModule is not None:
                dex_config = module_configs['dex_trading']
                dex_module = self._create_dex_module(dex_config)
                module_manager.register_module(dex_module)
                self.logger.info("DEX trading module registered")
            elif 'dex_trading' in module_configs:
                self.logger.warning("DEX trading module configured but not available (not installed)")

            # Register Futures trading module
            if 'futures_trading' in module_configs and FuturesTradingModule is not None:
                futures_config = module_configs['futures_trading']
                futures_module = self._create_futures_module(futures_config)
                module_manager.register_module(futures_module)
                self.logger.info("Futures trading module registered")
            elif 'futures_trading' in module_configs:
                self.logger.warning("Futures trading module configured but not available (not installed)")

            # Register Solana Strategies module
            if 'solana_strategies' in module_configs and SolanaStrategiesModule is not None:
                solana_config = module_configs['solana_strategies']
                solana_module = self._create_solana_module(solana_config)
                module_manager.register_module(solana_module)
                self.logger.info("Solana strategies module registered")
            elif 'solana_strategies' in module_configs:
                self.logger.warning("Solana strategies module configured but not available (not installed)")

            self.logger.info(
                f"Module setup complete. "
                f"Registered {len(module_manager.modules)} modules"
            )

            return module_manager

        except Exception as e:
            self.logger.error(f"Failed to setup modules: {e}", exc_info=True)
            raise

    def _load_module_configs(self) -> Dict:
        """Load module configurations from YAML files"""
        configs = {}

        modules_dir = self.config_dir / "modules"
        if not modules_dir.exists():
            self.logger.warning(f"Modules config directory not found: {modules_dir}")
            return configs

        # Load each module config file
        for config_file in modules_dir.glob("*.yaml"):
            try:
                module_name = config_file.stem
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'module' in config_data:
                    configs[module_name] = config_data
                    self.logger.info(f"Loaded config for module: {module_name}")

            except Exception as e:
                self.logger.error(f"Error loading config {config_file}: {e}")

        return configs

    def _create_dex_module(self, config_data: Dict) -> DexTradingModule:
        """
        Create DEX trading module from configuration

        Args:
            config_data: Configuration dictionary

        Returns:
            DexTradingModule: Configured DEX module
        """
        try:
            module_cfg = config_data['module']
            capital_cfg = config_data.get('capital', {})
            positions_cfg = config_data.get('positions', {})
            risk_cfg = config_data.get('risk', {})
            strategies_cfg = config_data.get('strategies', {})
            dex_cfg = config_data.get('dex', {})

            # Load wallet addresses
            wallet_addresses = self._load_wallet_addresses('dex_trading')

            # Create module configuration
            config = ModuleConfig(
                name=module_cfg['name'],
                module_type=ModuleType.DEX_TRADING,
                enabled=module_cfg.get('enabled', True),
                capital_allocation=capital_cfg.get('allocation', 500.0),
                max_positions=positions_cfg.get('max_open', 5),
                max_position_size=capital_cfg.get('max_position_size', 100.0),
                risk_per_trade=capital_cfg.get('risk_per_trade', 0.02),
                stop_loss_pct=risk_cfg.get('stop_loss_pct', 0.05),
                take_profit_pct=risk_cfg.get('take_profit_pct', 0.10),
                wallet_addresses=wallet_addresses,
                strategies=strategies_cfg.get('enabled', []),
                custom_settings={
                    'max_slippage_bps': dex_cfg.get('max_slippage_bps', 50),
                    'max_gas_price': dex_cfg.get('max_gas_price', 50),
                    'mev_protection': dex_cfg.get('mev_protection', True),
                    'jupiter_routing': dex_cfg.get('jupiter_routing', True),
                    'supported_chains': dex_cfg.get('chains', []),
                    'supported_dexs': dex_cfg.get('dexs', [])
                }
            )

            # Create module instance
            module = DexTradingModule(
                config=config,
                trading_engine=self.engine,
                db_manager=self.db,
                cache_manager=self.cache,
                alert_manager=self.alerts
            )

            return module

        except Exception as e:
            self.logger.error(f"Error creating DEX module: {e}", exc_info=True)
            raise

    def _create_futures_module(self, config_data: Dict) -> FuturesTradingModule:
        """Create Futures trading module from configuration"""
        try:
            module_cfg = config_data['module']
            capital_cfg = config_data.get('capital', {})
            positions_cfg = config_data.get('positions', {})
            risk_cfg = config_data.get('risk', {})

            # Load wallet addresses
            wallet_addresses = self._load_wallet_addresses('futures_trading')

            # Create module configuration
            config = ModuleConfig(
                name=module_cfg['name'],
                module_type=ModuleType.FUTURES_TRADING,
                enabled=module_cfg.get('enabled', True),
                capital_allocation=capital_cfg.get('allocation', 300.0),
                max_positions=positions_cfg.get('max_open', 5),
                max_position_size=capital_cfg.get('max_position_size', 60.0),
                risk_per_trade=capital_cfg.get('risk_per_trade', 0.02),
                stop_loss_pct=risk_cfg.get('stop_loss_pct', 0.05),
                take_profit_pct=risk_cfg.get('take_profit_pct', 0.10),
                wallet_addresses=wallet_addresses,
                strategies=config_data.get('strategies', {}).get('enabled', []),
                custom_settings=config_data
            )

            # Create module instance
            # Note: ml_model and pattern_analyzer are optional and can be added later
            module = FuturesTradingModule(
                config=config,
                trading_engine=self.engine,
                ml_model=None,  # TODO: Connect ML model if available
                pattern_analyzer=None,  # TODO: Connect pattern analyzer if available
                db_manager=self.db,
                cache_manager=self.cache,
                alert_manager=self.alerts,
                risk_manager=self.risk
            )

            return module

        except Exception as e:
            self.logger.error(f"Error creating Futures module: {e}", exc_info=True)
            raise

    def _create_solana_module(self, config_data: Dict) -> SolanaStrategiesModule:
        """Create Solana strategies module from configuration"""
        try:
            module_cfg = config_data['module']
            capital_cfg = config_data.get('capital', {})
            positions_cfg = config_data.get('positions', {})
            risk_cfg = config_data.get('risk', {})

            # Load wallet addresses
            wallet_addresses = self._load_wallet_addresses('solana_strategies')

            # Create module configuration
            config = ModuleConfig(
                name=module_cfg['name'],
                module_type=ModuleType.CUSTOM,
                enabled=module_cfg.get('enabled', True),
                capital_allocation=capital_cfg.get('allocation', 400.0),
                max_positions=positions_cfg.get('max_open', 8),
                max_position_size=capital_cfg.get('max_position_size', 80.0),
                risk_per_trade=capital_cfg.get('risk_per_trade', 0.02),
                stop_loss_pct=risk_cfg.get('stop_loss_pct', 0.10),
                take_profit_pct=risk_cfg.get('take_profit_pct', 0.20),
                wallet_addresses=wallet_addresses,
                strategies=list(config_data.get('strategies', {}).keys()),
                custom_settings=config_data
            )

            # Create module instance
            module = SolanaStrategiesModule(
                config=config,
                trading_engine=self.engine,
                db_manager=self.db,
                cache_manager=self.cache,
                alert_manager=self.alerts
            )

            return module

        except Exception as e:
            self.logger.error(f"Error creating Solana module: {e}", exc_info=True)
            raise

    def _load_wallet_addresses(self, module_name: str) -> Dict[str, str]:
        """
        Load wallet addresses for a module

        Args:
            module_name: Name of the module

        Returns:
            Dict: Wallet addresses by chain
        """
        try:
            wallets_file = self.config_dir / "wallets.yaml"
            if not wallets_file.exists():
                self.logger.warning(f"Wallets config not found: {wallets_file}")
                return {}

            with open(wallets_file, 'r') as f:
                wallets_data = yaml.safe_load(f)

            if not wallets_data or module_name not in wallets_data:
                return {}

            module_wallets = wallets_data[module_name]

            # Extract addresses (handle environment variable placeholders)
            addresses = {}
            for chain, wallet_info in module_wallets.items():
                if isinstance(wallet_info, dict) and 'address' in wallet_info:
                    addresses[chain] = wallet_info['address']

            return addresses

        except Exception as e:
            self.logger.error(f"Error loading wallet addresses: {e}")
            return {}

    def _get_total_capital(self, module_configs: Dict) -> float:
        """
        Calculate total capital from module configs

        Args:
            module_configs: Dictionary of module configurations

        Returns:
            float: Total capital
        """
        total = 0.0
        for config_data in module_configs.values():
            capital = config_data.get('capital', {}).get('allocation', 0.0)
            total += capital

        # Add some buffer for unallocated capital
        total = max(total, 1000.0)

        return total

    def integrate_with_dashboard(self, dashboard, module_manager):
        """
        Integrate module manager with dashboard

        Args:
            dashboard: DashboardEndpoints instance
            module_manager: ModuleManager instance
        """
        try:
            from monitoring.module_routes import ModuleRoutes

            # Create module routes
            module_routes = ModuleRoutes(
                module_manager=module_manager,
                jinja_env=dashboard.jinja_env
            )

            # Setup routes
            module_routes.setup_routes(dashboard.app)

            self.logger.info("Module routes integrated with dashboard")

        except Exception as e:
            self.logger.error(f"Error integrating with dashboard: {e}", exc_info=True)


async def setup_modular_architecture(
    engine,
    db_manager,
    cache_manager,
    alert_manager,
    risk_manager,
    dashboard=None
) -> ModuleManager:
    """
    Convenience function to setup the modular architecture

    Args:
        engine: Trading engine
        db_manager: Database manager
        cache_manager: Cache manager
        alert_manager: Alert manager
        risk_manager: Risk manager
        dashboard: Dashboard instance (optional)

    Returns:
        ModuleManager: Configured module manager
    """
    integration = ModuleIntegration(
        trading_engine=engine,
        db_manager=db_manager,
        cache_manager=cache_manager,
        alert_manager=alert_manager,
        risk_manager=risk_manager
    )

    # Setup modules
    module_manager = await integration.setup_modules()

    # Integrate with dashboard if provided
    if dashboard:
        integration.integrate_with_dashboard(dashboard, module_manager)

    return module_manager
