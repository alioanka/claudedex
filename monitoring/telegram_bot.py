"""
Telegram Bot Controller - Full Remote Control for Trading Bot

Commands:
    /status - Get overall bot status and module states
    /stop_all - Emergency stop: halt all modules and close all positions
    /stop <module> - Stop specific module (solana, arbitrage, futures, etc.)
    /start <module> - Start specific module
    /pause - Pause all trading (keeps monitoring)
    /resume - Resume trading
    /positions - View all open positions across all modules
    /close_all - Close all open positions
    /close <token> - Close specific position by token symbol
    /balance - Check wallet balances (SOL, ETH, etc.)
    /stats - Get trading statistics
    /pnl - Get P&L summary
    /config <key> <value> - Update config setting
    /help - Show all commands
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json

logger = logging.getLogger("TelegramBot")


class ModuleState(Enum):
    """State of a trading module"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RegisteredModule:
    """Represents a registered trading module"""
    name: str
    engine: Any  # The actual engine object
    state: ModuleState = ModuleState.STOPPED
    start_method: str = "run"  # Method to call to start
    stop_method: str = "stop"  # Method to call to stop
    positions_attr: str = "active_positions"  # Attribute containing positions
    task: Optional[asyncio.Task] = None
    last_error: Optional[str] = None
    started_at: Optional[datetime] = None


class TelegramBotController:
    """
    Telegram Bot Controller for full remote control of the trading bot.

    This controller allows you to:
    - Start/stop individual modules
    - Emergency stop all trading
    - Close all positions
    - View status, positions, and statistics
    - Update configuration remotely
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.admin_chat_ids: List[str] = []  # Authorized admin chat IDs

        # Parse admin chat IDs from env
        admin_ids = os.getenv('TELEGRAM_ADMIN_IDS', '')
        if admin_ids:
            self.admin_chat_ids = [id.strip() for id in admin_ids.split(',')]
        if self.chat_id:
            self.admin_chat_ids.append(self.chat_id)

        # Registered modules
        self.modules: Dict[str, RegisteredModule] = {}

        # Global state
        self.is_running = False
        self.trading_paused = False
        self.emergency_stop_active = False

        # Rate limiting
        self._last_command_time: Dict[str, datetime] = {}
        self._command_cooldown = 2  # seconds

        # Polling
        self._polling_task: Optional[asyncio.Task] = None
        self._last_update_id = 0

        # Command handlers
        self._commands: Dict[str, Callable] = {
            'status': self._cmd_status,
            'stop_all': self._cmd_stop_all,
            'stop': self._cmd_stop_module,
            'start': self._cmd_start_module,
            'pause': self._cmd_pause,
            'resume': self._cmd_resume,
            'positions': self._cmd_positions,
            'close_all': self._cmd_close_all,
            'close': self._cmd_close_position,
            'balance': self._cmd_balance,
            'stats': self._cmd_stats,
            'pnl': self._cmd_pnl,
            'config': self._cmd_config,
            'help': self._cmd_help,
            'emergency': self._cmd_emergency,
        }

    def register_module(
        self,
        name: str,
        engine: Any,
        start_method: str = "run",
        stop_method: str = "stop",
        positions_attr: str = "active_positions"
    ):
        """
        Register a trading module for remote control.

        Args:
            name: Module name (e.g., 'solana', 'arbitrage', 'futures')
            engine: The engine object
            start_method: Method name to call to start the engine
            stop_method: Method name to call to stop the engine
            positions_attr: Attribute name containing active positions dict
        """
        self.modules[name.lower()] = RegisteredModule(
            name=name,
            engine=engine,
            start_method=start_method,
            stop_method=stop_method,
            positions_attr=positions_attr,
            state=ModuleState.STOPPED
        )
        logger.info(f"Registered module: {name}")

    async def initialize(self):
        """Initialize the Telegram bot controller"""
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set - Telegram commands disabled")
            return False

        if not self.chat_id and not self.admin_chat_ids:
            logger.warning("TELEGRAM_CHAT_ID not set - Telegram commands disabled")
            return False

        # Test connection
        try:
            me = await self._api_call('getMe')
            if me:
                logger.info(f"Telegram bot connected: @{me.get('username', 'unknown')}")
                await self._send_message("Bot controller started. Send /help for commands.")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")

        return False

    async def start_polling(self):
        """Start polling for Telegram commands"""
        if not self.bot_token:
            return

        self.is_running = True
        self._polling_task = asyncio.create_task(self._poll_updates())
        logger.info("Telegram command polling started")

    async def stop_polling(self):
        """Stop polling for commands"""
        self.is_running = False
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        logger.info("Telegram command polling stopped")

    async def _poll_updates(self):
        """Poll Telegram for updates"""
        while self.is_running:
            try:
                updates = await self._api_call(
                    'getUpdates',
                    {'offset': self._last_update_id + 1, 'timeout': 30}
                )

                if updates:
                    for update in updates:
                        self._last_update_id = update.get('update_id', self._last_update_id)
                        await self._handle_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)

    async def _handle_update(self, update: Dict):
        """Handle a Telegram update"""
        message = update.get('message', {})
        text = message.get('text', '')
        chat_id = str(message.get('chat', {}).get('id', ''))

        if not text or not text.startswith('/'):
            return

        # Check authorization
        if chat_id not in self.admin_chat_ids:
            logger.warning(f"Unauthorized command from chat_id: {chat_id}")
            await self._send_message("Unauthorized. Your chat ID is not in the admin list.", chat_id)
            return

        # Parse command
        parts = text.split()
        command = parts[0][1:].lower().split('@')[0]  # Remove / and @botname
        args = parts[1:] if len(parts) > 1 else []

        # Rate limiting
        now = datetime.utcnow()
        last_time = self._last_command_time.get(chat_id)
        if last_time and (now - last_time).total_seconds() < self._command_cooldown:
            return
        self._last_command_time[chat_id] = now

        # Execute command
        handler = self._commands.get(command)
        if handler:
            try:
                await handler(chat_id, args)
            except Exception as e:
                logger.error(f"Command error: {e}", exc_info=True)
                await self._send_message(f"Error: {str(e)}", chat_id)
        else:
            await self._send_message(f"Unknown command: /{command}\nUse /help for available commands.", chat_id)

    async def _api_call(self, method: str, data: Dict = None) -> Optional[Any]:
        """Make Telegram API call"""
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data or {}, timeout=aiohttp.ClientTimeout(total=35)) as resp:
                    result = await resp.json()
                    if result.get('ok'):
                        return result.get('result')
                    else:
                        logger.error(f"Telegram API error: {result}")
                        return None
        except Exception as e:
            logger.error(f"Telegram API call failed: {e}")
            return None

    async def _send_message(self, text: str, chat_id: str = None):
        """Send message to Telegram"""
        target_chat = chat_id or self.chat_id
        if not target_chat:
            return

        # Telegram message limit is 4096 characters
        if len(text) > 4000:
            text = text[:3990] + "\n...(truncated)"

        await self._api_call('sendMessage', {
            'chat_id': target_chat,
            'text': text,
            'parse_mode': 'Markdown'
        })

    # ========== COMMAND HANDLERS ==========

    async def _cmd_help(self, chat_id: str, args: List[str]):
        """Show help message"""
        help_text = """
*Trading Bot Remote Control*

*Module Control:*
/status - Show all module states
/stop\\_all - EMERGENCY: Stop all & close positions
/stop <module> - Stop specific module
/start <module> - Start specific module
/pause - Pause trading (keep monitoring)
/resume - Resume trading

*Position Management:*
/positions - View all open positions
/close\\_all - Close all positions
/close <token> - Close specific position

*Information:*
/balance - Check wallet balances
/stats - Trading statistics
/pnl - P&L summary

*Configuration:*
/config <key> <value> - Update setting

*Emergency:*
/emergency - Full emergency shutdown

*Modules:* """ + ", ".join(self.modules.keys()) if self.modules else "None registered"

        await self._send_message(help_text, chat_id)

    async def _cmd_status(self, chat_id: str, args: List[str]):
        """Get status of all modules"""
        status_lines = ["*Bot Status*\n"]

        # Global state
        status_lines.append(f"Trading Paused: {'YES' if self.trading_paused else 'NO'}")
        status_lines.append(f"Emergency Stop: {'ACTIVE' if self.emergency_stop_active else 'NO'}")
        status_lines.append("")

        # Module states
        if not self.modules:
            status_lines.append("No modules registered")
        else:
            for name, module in self.modules.items():
                state_emoji = {
                    ModuleState.RUNNING: "",
                    ModuleState.PAUSED: "",
                    ModuleState.STOPPED: "",
                    ModuleState.STOPPING: "",
                    ModuleState.STARTING: "",
                    ModuleState.ERROR: "",
                }.get(module.state, "")

                status_lines.append(f"{state_emoji} *{name}*: {module.state.value}")

                # Position count
                if hasattr(module.engine, module.positions_attr):
                    positions = getattr(module.engine, module.positions_attr, {})
                    if positions:
                        status_lines.append(f"   Positions: {len(positions)}")

                if module.last_error:
                    status_lines.append(f"   Error: {module.last_error[:50]}")

        await self._send_message("\n".join(status_lines), chat_id)

    async def _cmd_stop_all(self, chat_id: str, args: List[str]):
        """Emergency stop all modules and close all positions"""
        await self._send_message("EMERGENCY STOP initiated...", chat_id)
        self.emergency_stop_active = True
        self.trading_paused = True

        results = []

        # First, close all positions
        results.append("*Closing all positions...*")
        for name, module in self.modules.items():
            try:
                if hasattr(module.engine, module.positions_attr):
                    positions = getattr(module.engine, module.positions_attr, {})
                    if positions:
                        results.append(f"  {name}: {len(positions)} positions to close")

                        # Try to close positions
                        if hasattr(module.engine, '_close_position'):
                            for token_mint in list(positions.keys()):
                                try:
                                    await module.engine._close_position(token_mint, "emergency_stop")
                                    results.append(f"    Closed: {token_mint[:8]}...")
                                except Exception as e:
                                    results.append(f"    Failed: {token_mint[:8]}... - {str(e)[:30]}")
            except Exception as e:
                results.append(f"  {name}: Error closing positions - {e}")

        # Then stop all modules
        results.append("\n*Stopping modules...*")
        for name, module in self.modules.items():
            try:
                if module.state == ModuleState.RUNNING:
                    module.state = ModuleState.STOPPING

                    if hasattr(module.engine, module.stop_method):
                        stop_func = getattr(module.engine, module.stop_method)
                        if asyncio.iscoroutinefunction(stop_func):
                            await stop_func()
                        else:
                            stop_func()

                    if hasattr(module.engine, 'is_running'):
                        module.engine.is_running = False

                    if module.task and not module.task.done():
                        module.task.cancel()

                    module.state = ModuleState.STOPPED
                    results.append(f"  {name}: STOPPED")
            except Exception as e:
                module.state = ModuleState.ERROR
                module.last_error = str(e)
                results.append(f"  {name}: FAILED - {e}")

        results.append("\nEMERGENCY STOP COMPLETE")
        await self._send_message("\n".join(results), chat_id)

    async def _cmd_stop_module(self, chat_id: str, args: List[str]):
        """Stop a specific module"""
        if not args:
            await self._send_message("Usage: /stop <module>\nAvailable: " + ", ".join(self.modules.keys()), chat_id)
            return

        module_name = args[0].lower()
        if module_name not in self.modules:
            await self._send_message(f"Unknown module: {module_name}\nAvailable: " + ", ".join(self.modules.keys()), chat_id)
            return

        module = self.modules[module_name]

        if module.state == ModuleState.STOPPED:
            await self._send_message(f"{module_name} is already stopped", chat_id)
            return

        try:
            module.state = ModuleState.STOPPING
            await self._send_message(f"Stopping {module_name}...", chat_id)

            if hasattr(module.engine, module.stop_method):
                stop_func = getattr(module.engine, module.stop_method)
                if asyncio.iscoroutinefunction(stop_func):
                    await stop_func()
                else:
                    stop_func()

            if hasattr(module.engine, 'is_running'):
                module.engine.is_running = False

            if module.task and not module.task.done():
                module.task.cancel()

            module.state = ModuleState.STOPPED
            await self._send_message(f"{module_name} STOPPED", chat_id)

        except Exception as e:
            module.state = ModuleState.ERROR
            module.last_error = str(e)
            await self._send_message(f"Failed to stop {module_name}: {e}", chat_id)

    async def _cmd_start_module(self, chat_id: str, args: List[str]):
        """Start a specific module"""
        if not args:
            await self._send_message("Usage: /start <module>\nAvailable: " + ", ".join(self.modules.keys()), chat_id)
            return

        module_name = args[0].lower()
        if module_name not in self.modules:
            await self._send_message(f"Unknown module: {module_name}\nAvailable: " + ", ".join(self.modules.keys()), chat_id)
            return

        if self.emergency_stop_active:
            await self._send_message("Cannot start - emergency stop is active. Use /resume first.", chat_id)
            return

        module = self.modules[module_name]

        if module.state == ModuleState.RUNNING:
            await self._send_message(f"{module_name} is already running", chat_id)
            return

        try:
            module.state = ModuleState.STARTING
            await self._send_message(f"Starting {module_name}...", chat_id)

            if hasattr(module.engine, module.start_method):
                start_func = getattr(module.engine, module.start_method)
                if asyncio.iscoroutinefunction(start_func):
                    module.task = asyncio.create_task(start_func())
                else:
                    start_func()

            module.state = ModuleState.RUNNING
            module.started_at = datetime.utcnow()
            await self._send_message(f"{module_name} STARTED", chat_id)

        except Exception as e:
            module.state = ModuleState.ERROR
            module.last_error = str(e)
            await self._send_message(f"Failed to start {module_name}: {e}", chat_id)

    async def _cmd_pause(self, chat_id: str, args: List[str]):
        """Pause all trading"""
        self.trading_paused = True

        # Set pause flag on all modules
        for name, module in self.modules.items():
            if hasattr(module.engine, 'trading_paused'):
                module.engine.trading_paused = True
            if hasattr(module.engine, 'risk_metrics') and hasattr(module.engine.risk_metrics, 'can_trade'):
                module.engine.risk_metrics.can_trade = False

        await self._send_message("Trading PAUSED on all modules.\nPosition monitoring continues.\nUse /resume to resume trading.", chat_id)

    async def _cmd_resume(self, chat_id: str, args: List[str]):
        """Resume trading"""
        self.trading_paused = False
        self.emergency_stop_active = False

        # Clear pause flag on all modules
        for name, module in self.modules.items():
            if hasattr(module.engine, 'trading_paused'):
                module.engine.trading_paused = False
            if hasattr(module.engine, 'risk_metrics') and hasattr(module.engine.risk_metrics, 'can_trade'):
                module.engine.risk_metrics.can_trade = True

        await self._send_message("Trading RESUMED on all modules.", chat_id)

    async def _cmd_positions(self, chat_id: str, args: List[str]):
        """View all open positions"""
        lines = ["*Open Positions*\n"]
        total_positions = 0

        for name, module in self.modules.items():
            if not hasattr(module.engine, module.positions_attr):
                continue

            positions = getattr(module.engine, module.positions_attr, {})
            if not positions:
                continue

            lines.append(f"*{name}:*")

            for token_mint, position in positions.items():
                total_positions += 1
                symbol = getattr(position, 'token_symbol', token_mint[:8])
                pnl = getattr(position, 'unrealized_pnl_pct', 0)
                value = getattr(position, 'value_sol', 0)

                pnl_emoji = "" if pnl >= 0 else ""
                lines.append(f"  {pnl_emoji} {symbol}: {pnl:+.2f}% ({value:.4f} SOL)")

        if total_positions == 0:
            lines.append("No open positions")
        else:
            lines.append(f"\n*Total: {total_positions} positions*")

        await self._send_message("\n".join(lines), chat_id)

    async def _cmd_close_all(self, chat_id: str, args: List[str]):
        """Close all open positions"""
        await self._send_message("Closing all positions...", chat_id)

        results = []
        closed_count = 0

        for name, module in self.modules.items():
            if not hasattr(module.engine, module.positions_attr):
                continue

            positions = getattr(module.engine, module.positions_attr, {})
            if not positions:
                continue

            if hasattr(module.engine, '_close_position'):
                for token_mint in list(positions.keys()):
                    try:
                        await module.engine._close_position(token_mint, "telegram_close_all")
                        symbol = positions.get(token_mint, {})
                        if hasattr(symbol, 'token_symbol'):
                            symbol = symbol.token_symbol
                        else:
                            symbol = token_mint[:8]
                        results.append(f" {symbol}")
                        closed_count += 1
                    except Exception as e:
                        results.append(f" {token_mint[:8]}: {str(e)[:30]}")

        if closed_count > 0:
            results.insert(0, f"*Closed {closed_count} positions:*")
        else:
            results = ["No positions to close"]

        await self._send_message("\n".join(results), chat_id)

    async def _cmd_close_position(self, chat_id: str, args: List[str]):
        """Close a specific position"""
        if not args:
            await self._send_message("Usage: /close <token\\_symbol>", chat_id)
            return

        target_symbol = args[0].upper()
        found = False

        for name, module in self.modules.items():
            if not hasattr(module.engine, module.positions_attr):
                continue

            positions = getattr(module.engine, module.positions_attr, {})

            for token_mint, position in positions.items():
                symbol = getattr(position, 'token_symbol', '').upper()
                if symbol == target_symbol or token_mint.upper().startswith(target_symbol):
                    found = True
                    try:
                        if hasattr(module.engine, '_close_position'):
                            await module.engine._close_position(token_mint, "telegram_manual_close")
                            await self._send_message(f"Closed position: {symbol}", chat_id)
                        else:
                            await self._send_message(f"Module {name} doesn't support closing positions", chat_id)
                    except Exception as e:
                        await self._send_message(f"Failed to close {symbol}: {e}", chat_id)
                    break

            if found:
                break

        if not found:
            await self._send_message(f"Position not found: {target_symbol}", chat_id)

    async def _cmd_balance(self, chat_id: str, args: List[str]):
        """Check wallet balances"""
        lines = ["*Wallet Balances*\n"]

        # Check SOL balance
        for name, module in self.modules.items():
            if hasattr(module.engine, 'sol_balance_native'):
                balance = module.engine.sol_balance_native
                lines.append(f"SOL ({name}): {balance:.4f}")
            if hasattr(module.engine, 'wallet_address'):
                addr = module.engine.wallet_address
                if addr:
                    lines.append(f"Wallet: {addr[:8]}...{addr[-6:]}")
                    break

        # Try to get ETH balance
        for name, module in self.modules.items():
            if hasattr(module.engine, 'wallet_address') and hasattr(module.engine, 'w3'):
                try:
                    w3 = module.engine.w3
                    addr = module.engine.wallet_address
                    if w3 and addr:
                        balance_wei = w3.eth.get_balance(addr)
                        balance_eth = balance_wei / 1e18
                        lines.append(f"ETH ({name}): {balance_eth:.4f}")
                except:
                    pass

        if len(lines) == 1:
            lines.append("No balance information available")

        await self._send_message("\n".join(lines), chat_id)

    async def _cmd_stats(self, chat_id: str, args: List[str]):
        """Get trading statistics"""
        lines = ["*Trading Statistics*\n"]

        for name, module in self.modules.items():
            engine = module.engine

            stats_added = False
            if hasattr(engine, 'total_trades'):
                if not stats_added:
                    lines.append(f"*{name}:*")
                    stats_added = True

                lines.append(f"  Trades: {engine.total_trades}")

                if hasattr(engine, 'winning_trades'):
                    win_rate = (engine.winning_trades / engine.total_trades * 100) if engine.total_trades > 0 else 0
                    lines.append(f"  Win Rate: {win_rate:.1f}%")

                if hasattr(engine, 'total_pnl_sol'):
                    lines.append(f"  PnL: {engine.total_pnl_sol:+.4f} SOL")

        if len(lines) == 1:
            lines.append("No statistics available")

        await self._send_message("\n".join(lines), chat_id)

    async def _cmd_pnl(self, chat_id: str, args: List[str]):
        """Get P&L summary"""
        lines = ["*P&L Summary*\n"]
        total_pnl_sol = 0
        total_pnl_usd = 0

        for name, module in self.modules.items():
            engine = module.engine

            if hasattr(engine, 'total_pnl_sol'):
                pnl_sol = engine.total_pnl_sol
                total_pnl_sol += pnl_sol

                pnl_usd = 0
                if hasattr(engine, 'sol_price_usd'):
                    pnl_usd = pnl_sol * engine.sol_price_usd
                    total_pnl_usd += pnl_usd

                emoji = "" if pnl_sol >= 0 else ""
                lines.append(f"{emoji} *{name}*: {pnl_sol:+.4f} SOL (${pnl_usd:+.2f})")

                if hasattr(engine, 'risk_metrics') and hasattr(engine.risk_metrics, 'daily_pnl_sol'):
                    daily = engine.risk_metrics.daily_pnl_sol
                    daily_emoji = "" if daily >= 0 else ""
                    lines.append(f"   Today: {daily_emoji} {daily:+.4f} SOL")

        if total_pnl_sol != 0:
            lines.append(f"\n*Total: {total_pnl_sol:+.4f} SOL (${total_pnl_usd:+.2f})*")
        else:
            lines.append("No P&L data available")

        await self._send_message("\n".join(lines), chat_id)

    async def _cmd_config(self, chat_id: str, args: List[str]):
        """Update configuration"""
        if len(args) < 2:
            await self._send_message("Usage: /config <key> <value>\nExample: /config stop\\_loss\\_pct -15", chat_id)
            return

        key = args[0]
        value = " ".join(args[1:])

        # Try to parse value type
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
            if '.' in value:
                value = float(value)
            else:
                value = int(value)

        # Update in database if available
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO config_settings (config_type, key, value)
                        VALUES ('telegram_update', $1, $2)
                        ON CONFLICT (config_type, key) DO UPDATE SET value = $2
                    """, key, str(value))
                await self._send_message(f"Config updated: {key} = {value}", chat_id)
            except Exception as e:
                await self._send_message(f"Failed to update config: {e}", chat_id)
        else:
            await self._send_message("Database not available for config updates", chat_id)

    async def _cmd_emergency(self, chat_id: str, args: List[str]):
        """Full emergency shutdown"""
        await self._send_message("EMERGENCY SHUTDOWN INITIATED\nThis will stop all modules and close all positions.", chat_id)
        await self._cmd_stop_all(chat_id, args)

    # ========== NOTIFICATION METHODS ==========

    async def notify(self, message: str, priority: str = "normal"):
        """
        Send notification to Telegram.

        Args:
            message: Message to send
            priority: 'low', 'normal', 'high', 'critical'
        """
        if not self.bot_token or not self.chat_id:
            return

        emoji = {
            'low': '',
            'normal': '',
            'high': '',
            'critical': ''
        }.get(priority, '')

        await self._send_message(f"{emoji} {message}")

    async def notify_trade(self, action: str, symbol: str, amount: float, price: float, pnl: float = None):
        """Send trade notification"""
        if pnl is not None:
            emoji = "" if pnl >= 0 else ""
            msg = f"{emoji} *{action}* {symbol}\nAmount: {amount:.4f}\nPrice: ${price:.6f}\nPnL: {pnl:+.2f}%"
        else:
            msg = f" *{action}* {symbol}\nAmount: {amount:.4f}\nPrice: ${price:.6f}"

        await self._send_message(msg)

    async def notify_error(self, error: str, module: str = None):
        """Send error notification"""
        prefix = f"[{module}] " if module else ""
        await self._send_message(f" *Error* {prefix}\n{error}")

    async def notify_position_stuck(self, symbol: str, reason: str):
        """Notify about stuck position"""
        await self._send_message(f" *Stuck Position*\n{symbol}: {reason}\n\nUse /close {symbol} to force close")


# Singleton instance
_telegram_controller: Optional[TelegramBotController] = None


def get_telegram_controller(db_pool=None) -> TelegramBotController:
    """Get or create the Telegram controller singleton"""
    global _telegram_controller
    if _telegram_controller is None:
        _telegram_controller = TelegramBotController(db_pool)
    return _telegram_controller
