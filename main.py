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
from typing import Dict, List, Optional, Tuple
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


class RotatingLogFile:
    """
    A file-like object that rotates logs based on size.
    Can be used as stdout/stderr for subprocess.Popen.

    Note: subprocess writes directly to the file descriptor, bypassing write().
    Use check_and_rotate() periodically to enforce size limits.
    """
    def __init__(self, filepath: Path, max_bytes: int = 10*1024*1024, backup_count: int = 3):
        self.filepath = Path(filepath)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._file = None
        # Check and rotate at startup if file is already too large
        self._check_and_rotate_if_needed()
        self._open()

    def _open(self):
        """Open or reopen the log file"""
        if self._file:
            try:
                self._file.close()
            except:
                pass
        self._file = open(self.filepath, 'a', encoding='utf-8', buffering=1)  # Line buffered

    def _check_and_rotate_if_needed(self):
        """Check file size and rotate if needed - call this periodically"""
        try:
            if self.filepath.exists() and self.filepath.stat().st_size >= self.max_bytes:
                self._do_rollover_external()
                return True
        except:
            pass
        return False

    def _do_rollover_external(self):
        """Perform log rotation when file is not open (or before opening)"""
        try:
            # Rotate existing backups: .3 -> .4, .2 -> .3, .1 -> .2
            for i in range(self.backup_count - 1, 0, -1):
                src = Path(f"{self.filepath}.{i}")
                dst = Path(f"{self.filepath}.{i + 1}")
                if src.exists():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
            # Move current to .1
            backup_1 = Path(f"{self.filepath}.1")
            if self.filepath.exists():
                if backup_1.exists():
                    backup_1.unlink()
                self.filepath.rename(backup_1)
            logger.info(f"Rotated log file: {self.filepath}")
        except Exception as e:
            logger.warning(f"Log rotation failed for {self.filepath}: {e}")

    def check_and_rotate(self):
        """
        Check file size and rotate if needed.
        Call this periodically since subprocess writes bypass write().
        Returns True if rotation occurred.
        """
        try:
            if self.filepath.exists() and self.filepath.stat().st_size >= self.max_bytes:
                # Close file, rotate, reopen
                if self._file:
                    self._file.flush()
                    self._file.close()
                self._do_rollover_external()
                self._open()
                return True
        except Exception as e:
            logger.warning(f"Error checking log rotation for {self.filepath}: {e}")
        return False

    def write(self, data):
        """Write data to file"""
        if data:
            try:
                if isinstance(data, bytes):
                    data = data.decode('utf-8', errors='replace')
                self._file.write(data)
                self._file.flush()
            except Exception as e:
                try:
                    self._open()
                    self._file.write(data)
                except:
                    pass
        return len(data) if data else 0

    def flush(self):
        """Flush the file buffer"""
        try:
            if self._file:
                self._file.flush()
        except:
            pass

    def fileno(self):
        """Return file descriptor for subprocess compatibility"""
        return self._file.fileno()

    def close(self):
        """Close the file"""
        try:
            if self._file:
                self._file.close()
                self._file = None
        except:
            pass


class ModuleProcess:
    """Represents a trading module process"""

    # Required secrets per module
    REQUIRED_SECRETS = {
        'dex': [
            ('PRIVATE_KEY', 'EVM private key for DEX trading'),
            ('ENCRYPTION_KEY', 'Encryption key for decrypting secrets'),
        ],
        'futures': [
            ('ENCRYPTION_KEY', 'Encryption key for decrypting secrets'),
            # Binance OR Bybit keys required (checked dynamically)
        ],
        'solana': [
            ('SOLANA_MODULE_PRIVATE_KEY', 'Solana module private key'),
            ('ENCRYPTION_KEY', 'Encryption key for decrypting secrets'),
        ],
        'sniper': [],
        'ai_analysis': [('OPENAI_API_KEY', 'OpenAI API Key for Sentiment Analysis')]
    }

    def __init__(self, name: str, script_path: str, enabled_env_var: str, module_key: str = None):
        self.name = name
        self.script_path = script_path
        self.enabled_env_var = enabled_env_var
        self.module_key = module_key or name.lower().replace(' ', '_').replace('trading', '').strip('_')
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.max_restarts = 3
        # File handles for subprocess output (prevents PIPE buffer deadlock)
        self._stdout_file = None
        self._stderr_file = None
        # Validation state
        self.validation_errors: List[str] = []
        self.is_validated = False

    def is_enabled(self) -> bool:
        """Check if module is enabled via environment variable"""
        enabled = os.getenv(self.enabled_env_var, 'false').lower()
        return enabled in ('true', '1', 'yes')

    def validate_secrets(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required secrets are present for this module.
        Returns (is_valid, list_of_missing_secrets)
        """
        errors = []
        module_key = self.module_key

        # Get base required secrets
        required = self.REQUIRED_SECRETS.get(module_key, [])

        for secret_name, description in required:
            value = os.getenv(secret_name)
            if not value or value.strip() == '':
                errors.append(f"Missing {secret_name}: {description}")

        # Special validation for futures module (needs Binance OR Bybit keys)
        if module_key == 'futures':
            exchange = os.getenv('FUTURES_EXCHANGE', 'binance').lower()
            testnet = os.getenv('FUTURES_TESTNET', 'true').lower() in ('true', '1', 'yes')

            if exchange == 'binance':
                if testnet:
                    if not os.getenv('BINANCE_TESTNET_API_KEY'):
                        errors.append("Missing BINANCE_TESTNET_API_KEY: Required for Binance testnet")
                    if not os.getenv('BINANCE_TESTNET_API_SECRET'):
                        errors.append("Missing BINANCE_TESTNET_API_SECRET: Required for Binance testnet")
                else:
                    if not os.getenv('BINANCE_API_KEY'):
                        errors.append("Missing BINANCE_API_KEY: Required for Binance mainnet")
                    if not os.getenv('BINANCE_API_SECRET'):
                        errors.append("Missing BINANCE_API_SECRET: Required for Binance mainnet")
            elif exchange == 'bybit':
                if testnet:
                    if not os.getenv('BYBIT_TESTNET_API_KEY'):
                        errors.append("Missing BYBIT_TESTNET_API_KEY: Required for Bybit testnet")
                    if not os.getenv('BYBIT_TESTNET_API_SECRET'):
                        errors.append("Missing BYBIT_TESTNET_API_SECRET: Required for Bybit testnet")
                else:
                    if not os.getenv('BYBIT_API_KEY'):
                        errors.append("Missing BYBIT_API_KEY: Required for Bybit mainnet")
                    if not os.getenv('BYBIT_API_SECRET'):
                        errors.append("Missing BYBIT_API_SECRET: Required for Bybit mainnet")

        # Special validation for Solana module
        if module_key == 'solana':
            if not os.getenv('SOLANA_RPC_URL') and not os.getenv('SOLANA_RPC_URLS'):
                errors.append("Missing SOLANA_RPC_URL: Required for Solana trading")

        self.validation_errors = errors
        self.is_validated = len(errors) == 0
        return self.is_validated, errors

    async def start(self):
        """Start the module process"""
        if not self.is_enabled():
            logger.info(f"⊘ {self.name} module is disabled ({self.enabled_env_var}=false)")
            return False

        if not Path(self.script_path).exists():
            logger.error(f"❌ {self.name} script not found: {self.script_path}")
            return False

        # Validate required secrets before starting
        is_valid, errors = self.validate_secrets()
        if not is_valid:
            logger.error(f"❌ {self.name} module BLOCKED - Missing required secrets:")
            for error in errors:
                logger.error(f"   • {error}")
            logger.error(f"   ⚠️  Configure the missing secrets in .env and restart")
            return False

        try:
            logger.info(f"🚀 Starting {self.name} module...")

            # Create log directory for module output
            log_dir = Path("logs") / self.name.lower().replace(" ", "_")
            log_dir.mkdir(parents=True, exist_ok=True)

            # Open rotating log files for subprocess stdout/stderr
            # CRITICAL: Do NOT use subprocess.PIPE without reading from it!
            # If the pipe buffer fills up (~64KB), the subprocess will block/freeze.
            # Using RotatingLogFile to prevent logs from growing to 100MB+
            stdout_file = RotatingLogFile(
                log_dir / "stdout.log",
                max_bytes=10*1024*1024,  # 10MB max
                backup_count=3
            )
            stderr_file = RotatingLogFile(
                log_dir / "stderr.log",
                max_bytes=10*1024*1024,  # 10MB max
                backup_count=3
            )

            self.process = subprocess.Popen(
                [sys.executable, "-u", self.script_path],  # -u for unbuffered stdout
                stdout=stdout_file,
                stderr=stderr_file,
                env=os.environ.copy()
            )

            # Store file handles for cleanup
            self._stdout_file = stdout_file
            self._stderr_file = stderr_file

            logger.info(f"✅ {self.name} started (PID: {self.process.pid})")
            logger.info(f"   Logs: {log_dir}/stdout.log, {log_dir}/stderr.log (rotating, max 10MB)")
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
        finally:
            # Close log file handles
            if hasattr(self, '_stdout_file') and self._stdout_file:
                try:
                    self._stdout_file.close()
                except:
                    pass
            if hasattr(self, '_stderr_file') and self._stderr_file:
                try:
                    self._stderr_file.close()
                except:
                    pass

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
            script_path="modules/dex_trading/main_dex.py",
            enabled_env_var="DEX_MODULE_ENABLED",
            module_key="dex"
        )

        self.modules['futures'] = ModuleProcess(
            name="Futures Trading",
            script_path="modules/futures_trading/main_futures.py",
            enabled_env_var="FUTURES_MODULE_ENABLED",
            module_key="futures"
        )

        self.modules['solana'] = ModuleProcess(
            name="Solana Trading",
            script_path="modules/solana_trading/main_solana.py",
            enabled_env_var="SOLANA_MODULE_ENABLED",
            module_key="solana"
        )

        self.modules['sniper'] = ModuleProcess(
            name="Sniper Module",
            script_path="modules/sniper/main_sniper.py",
            enabled_env_var="SNIPER_MODULE_ENABLED",
            module_key="sniper"
        )

        self.modules['ai_analysis'] = ModuleProcess(
            name="AI Analysis",
            script_path="modules/ai_analysis/main_ai.py",
            enabled_env_var="AI_MODULE_ENABLED",
            module_key="ai_analysis"
        )

        self.modules['arbitrage'] = ModuleProcess(
            name="Arbitrage Module",
            script_path="modules/arbitrage/main_arbitrage.py",
            enabled_env_var="ARBITRAGE_MODULE_ENABLED",
            module_key="arbitrage"
        )

        self.modules['copy_trading'] = ModuleProcess(
            name="Copy Trading Module",
            script_path="modules/copy_trading/main_copy.py",
            enabled_env_var="COPY_TRADING_MODULE_ENABLED",
            module_key="copy_trading"
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
        """Monitor health of all modules and rotate logs"""
        log_rotation_counter = 0
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                log_rotation_counter += 1

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

                    # Check log rotation every 5 health checks (~5 minutes)
                    if log_rotation_counter % 5 == 0:
                        if hasattr(module, '_stdout_file') and module._stdout_file:
                            if hasattr(module._stdout_file, 'check_and_rotate'):
                                module._stdout_file.check_and_rotate()
                        if hasattr(module, '_stderr_file') and module._stderr_file:
                            if hasattr(module._stderr_file, 'check_and_rotate'):
                                module._stderr_file.check_and_rotate()

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
