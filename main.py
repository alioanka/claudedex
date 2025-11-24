#!/usr/bin/env python3
"""
Trading Bot Orchestrator - Multi-Module Manager
Manages DEX, Futures, and Solana trading modules as separate processes

Architecture:
- DEX Trading: main_dex.py (EVM chains)
- Futures Trading: futures_trading/main_futures.py (Binance/Bybit)
- Solana Trading: solana_trading/main_solana.py (Jupiter/Drift/Pump.fun)
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import subprocess
from typing import Dict, List, Optional
import psutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")


class ModuleProcess:
    """Represents a trading module process"""

    def __init__(self, name: str, script_path: str, enabled_env_var: str):
        self.name = name
        self.script_path = script_path
        self.enabled_env_var = enabled_env_var
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.max_restarts = 3

    def is_enabled(self) -> bool:
        """Check if module is enabled via environment variable"""
        enabled = os.getenv(self.enabled_env_var, 'false').lower()
        return enabled in ('true', '1', 'yes')

    async def start(self):
        """Start the module process"""
        if not self.is_enabled():
            logger.info(f"⊘ {self.name} module is disabled ({self.enabled_env_var}=false)")
            return False

        if not Path(self.script_path).exists():
            logger.error(f"❌ {self.name} script not found: {self.script_path}")
            return False

        try:
            logger.info(f"🚀 Starting {self.name} module...")
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            logger.info(f"✅ {self.name} started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start {self.name}: {e}")
            return False

    def is_running(self) -> bool:
        """Check if process is running"""
        if self.process is None:
            return False
        return self.process.poll() is None

    async def stop(self):
        """Stop the module process gracefully"""
        if self.process is None:
            return

        try:
            logger.info(f"Stopping {self.name}...")

            # Try graceful shutdown first
            self.process.terminate()

            # Wait up to 30 seconds for graceful shutdown
            try:
                self.process.wait(timeout=30)
                logger.info(f"✅ {self.name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning(f"⚠️  {self.name} not responding, forcing shutdown...")
                self.process.kill()
                self.process.wait()
                logger.info(f"✅ {self.name} force stopped")

        except Exception as e:
            logger.error(f"Error stopping {self.name}: {e}")

    async def restart(self):
        """Restart the module"""
        if self.restart_count >= self.max_restarts:
            logger.error(f"❌ {self.name} exceeded max restarts ({self.max_restarts})")
            return False

        logger.info(f"♻️  Restarting {self.name}...")
        await self.stop()
        await asyncio.sleep(5)  # Wait before restart

        self.restart_count += 1
        return await self.start()


class TradingBotOrchestrator:
    """Orchestrates multiple trading modules"""

    def __init__(self):
        self.modules: Dict[str, ModuleProcess] = {}
        self.shutdown_event = asyncio.Event()
        self.health_check_interval = 60  # seconds

        # Define modules
        self.modules['dex'] = ModuleProcess(
            name="DEX Trading",
            script_path="main_dex.py",
            enabled_env_var="DEX_MODULE_ENABLED"
        )

        self.modules['futures'] = ModuleProcess(
            name="Futures Trading",
            script_path="futures_trading/main_futures.py",
            enabled_env_var="FUTURES_MODULE_ENABLED"
        )

        self.modules['solana'] = ModuleProcess(
            name="Solana Trading",
            script_path="solana_trading/main_solana.py",
            enabled_env_var="SOLANA_MODULE_ENABLED"
        )

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def start_modules(self):
        """Start all enabled modules"""
        logger.info("=" * 80)
        logger.info("🚀 Trading Bot Orchestrator Starting")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info("=" * 80)
        logger.info("")

        started_count = 0
        for name, module in self.modules.items():
            if await module.start():
                started_count += 1

        if started_count == 0:
            logger.error("❌ No modules started!")
            return False

        logger.info("")
        logger.info(f"✅ Started {started_count}/{len(self.modules)} module(s)")
        logger.info("=" * 80)
        return True

    async def health_monitor(self):
        """Monitor health of all modules"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)

                for name, module in self.modules.items():
                    if not module.is_enabled():
                        continue

                    if not module.is_running():
                        logger.warning(f"⚠️  {module.name} is not running!")

                        # Attempt restart
                        if module.restart_count < module.max_restarts:
                            await module.restart()
                        else:
                            logger.error(f"❌ {module.name} has failed permanently")

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def status_reporter(self):
        """Periodically report status of all modules"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Report every 5 minutes

                logger.info("=" * 80)
                logger.info("📊 MODULE STATUS REPORT")
                logger.info("=" * 80)

                for name, module in self.modules.items():
                    if not module.is_enabled():
                        logger.info(f"  {module.name:20s} - DISABLED")
                        continue

                    if module.is_running():
                        try:
                            proc = psutil.Process(module.process.pid)
                            cpu = proc.cpu_percent(interval=1)
                            mem = proc.memory_info().rss / 1024 / 1024  # MB
                            logger.info(
                                f"  {module.name:20s} - RUNNING "
                                f"(PID: {module.process.pid}, "
                                f"CPU: {cpu:.1f}%, "
                                f"MEM: {mem:.0f}MB, "
                                f"Restarts: {module.restart_count})"
                            )
                        except:
                            logger.info(f"  {module.name:20s} - RUNNING")
                    else:
                        logger.info(f"  {module.name:20s} - STOPPED")

                logger.info("=" * 80)

            except Exception as e:
                logger.error(f"Error in status reporter: {e}")

    async def run(self):
        """Main orchestrator loop"""
        try:
            # Start all modules
            if not await self.start_modules():
                return

            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self.health_monitor()),
                asyncio.create_task(self.status_reporter()),
                asyncio.create_task(self._shutdown_monitor())
            ]

            # Wait for shutdown signal or task failure
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def _shutdown_monitor(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()

    async def shutdown(self):
        """Gracefully shutdown all modules"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("🛑 Initiating Orchestrator Shutdown")
        logger.info("=" * 80)

        # Stop all modules
        for name, module in self.modules.items():
            if module.is_running():
                await module.stop()

        logger.info("✅ All modules stopped")
        logger.info("=" * 80)


async def main():
    """Main entry point"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    orchestrator = TradingBotOrchestrator()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)

    asyncio.run(main())
