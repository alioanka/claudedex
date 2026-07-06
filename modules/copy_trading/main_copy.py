#!/usr/bin/env python3
"""
Copy Trading Module - Entry Point
"""
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import RPCProvider for centralized RPC management
try:
    from config.rpc_provider import RPCProvider
except ImportError:
    RPCProvider = None

# Load env
load_dotenv()

# Setup Logging
log_dir = Path("logs/copy_trading")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
trade_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Root logger
logger = logging.getLogger("CopyTradingModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'copy_trading.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'copy_trading_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# 3. Trades Log
trade_logger = logging.getLogger("CopyTradingModule.Trades")
trade_logger.setLevel(logging.INFO)
trade_logger.propagate = False
trade_handler = RotatingFileHandler(log_dir / 'copy_trading_trades.log', maxBytes=10*1024*1024, backupCount=5)
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.copy_trading.copy_engine import CopyTradingEngine, CopyTradeExecutor
from config.config_manager import ConfigManager

# Import Telegram controller for remote control
try:
    from monitoring.telegram_bot import get_telegram_controller
except ImportError:
    get_telegram_controller = None

# Also configure logging for engine classes
for engine_name in ["CopyTradingEngine", "CopyTradeExecutor"]:
    engine_logger = logging.getLogger(engine_name)
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(main_handler)
    engine_logger.addHandler(error_handler)
    engine_logger.addHandler(console)

async def main():
    logger.info("👯 Copy Trading Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")
    # Per-module DRY_RUN override (Phase 3 A4). COPY_TRADING_DRY_RUN
    # env beats DRY_RUN. Mirror into DRY_RUN so copy_engine's
    # should_skip_live() picks up the resolved value.
    try:
        from core.dry_run import resolve_module_dry_run
        copy_dry = resolve_module_dry_run('copy_trading', default=True)
        os.environ['DRY_RUN'] = 'true' if copy_dry else 'false'
        logger.info(f"   DRY_RUN (resolved per-module): {copy_dry}")
    except Exception as e:
        logger.warning(f"   Could not resolve per-module DRY_RUN: {e}")

    # Wave-7 fix (issue 17a, mirrors AI commit cca8d94): the copy module's
    # API keys (ETHERSCAN_API_KEY / HELIUS_API_KEY) live in the encrypted
    # `secure_credentials` DB table, NOT in .env. The previous code read
    # `secrets.get(...)` HERE — BEFORE the db_pool was created and BEFORE
    # `secrets.initialize(db_pool)`. The secrets manager was still in
    # bootstrap mode, so it fell back to os.getenv (None for DB-only ops)
    # and logged "ETHERSCAN_API_KEY: NOT SET" / "HELIUS_API_KEY: Not set"
    # even though both were configured. The Solana RPC was likewise
    # resolved via the public .env fallback (hence the constant
    # "Solana RPC rate limited" spam) because the Helius key wasn't
    # available to build the Helius endpoint.
    #
    # Resolution: connect the DB FIRST, initialize the secrets manager
    # with the pool, THEN resolve the keys via get_async (the sync get()
    # short-circuits inside a running event loop — see secrets_manager
    # _get_from_database_sync). Only after the Helius key resolves do we
    # pick the Solana RPC, preferring Helius over any public endpoint.

    # Init DB FIRST - Use Docker secrets or environment
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')

    if not db_url:
        logger.error("No database credentials found (Docker secrets or DATABASE_URL)")
        return

    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    # Initialize secrets manager with database pool (before any config managers)
    try:
        from security.secrets_manager import secrets
        secrets.initialize(db_pool)
        logger.info("✅ Secrets manager initialized with database")
    except Exception as e:
        logger.warning(f"Could not initialize secrets manager: {e}")

    # Now resolve API keys AFTER the secrets manager has the db_pool, so
    # the DB-backed encrypted credentials path is taken instead of the
    # bootstrap-mode os.getenv fallback.
    etherscan_key = None
    helius_key = None
    try:
        from security.secrets_manager import secrets as _secrets
        etherscan_key = await _secrets.get_async('ETHERSCAN_API_KEY', log_access=False)
        helius_key = await _secrets.get_async('HELIUS_API_KEY', log_access=False)
    except Exception as e:
        logger.warning(f"   secrets_manager lookup failed: {e}; falling back to env")
    # .env fallback (gradual-migration support)
    if not etherscan_key:
        etherscan_key = os.getenv('ETHERSCAN_API_KEY')
    if not helius_key:
        helius_key = os.getenv('HELIUS_API_KEY')

    # Wave-F5 multi-key: count numbered sibling keys (KEY_2 .. KEY_9 in
    # secrets/env). pool_engine registers each as its own rotating endpoint
    # at engine init; this is an honest startup log of what's configured.
    async def _count_sibling_keys(base: str, have_base: bool) -> int:
        n = 1 if have_base else 0
        for i in range(2, 10):
            name = f"{base}_{i}"
            v = None
            try:
                from security.secrets_manager import secrets as _s
                v = await _s.get_async(name, log_access=False)
            except Exception:
                pass
            if not v:
                v = os.getenv(name)
            if v and v not in ('null', 'None') and not v.startswith('your_'):
                n += 1
        return n

    helius_key_count = await _count_sibling_keys('HELIUS_API_KEY', bool(helius_key))
    etherscan_key_count = await _count_sibling_keys('ETHERSCAN_API_KEY', bool(etherscan_key))

    # Resolve the Solana RPC AFTER the Helius key is known. Prefer Helius
    # (paid, high-rate) over any public endpoint so the run-loop stops
    # getting 429-throttled. This boot-time value is DIAGNOSTIC + last-resort
    # only: the engine re-resolves its RPC/keys per cycle via pool_engine
    # (with multi-key rotation), so it is never pinned to this URL.
    solana_rpc = None
    if helius_key:
        solana_rpc = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
    if not solana_rpc and RPCProvider:
        solana_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
    if not solana_rpc:
        solana_rpc = os.getenv('SOLANA_RPC_URL')

    rpc_is_helius = bool(helius_key) and bool(solana_rpc) and 'helius' in solana_rpc.lower()
    logger.info(
        f"   ETHERSCAN_API_KEY: "
        f"{f'SET ({etherscan_key_count} key(s), pool-rotated)' if etherscan_key else 'NOT SET - EVM monitoring disabled'}"
    )
    logger.info(
        f"   HELIUS_API_KEY: "
        f"{f'SET ({helius_key_count} key(s), pool-rotated)' if helius_key else 'Not set (optional)'}"
    )
    logger.info(
        f"   SOLANA_RPC: {'SET (Helius)' if rpc_is_helius else ('SET (public/fallback)' if solana_rpc else 'NOT SET - Solana monitoring disabled')}"
    )

    if not etherscan_key and not solana_rpc:
        logger.warning("⚠️ No API keys configured - Copy Trading will not monitor any wallets")

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Engine
    config = {'copy_trading_enabled': True}
    engine = CopyTradingEngine(config, db_pool)

    # Initialize Telegram controller for remote control (credentials from secrets manager)
    telegram_controller = None
    if get_telegram_controller:
        try:
            # Wave-11 FIX 2: tag with module_name so start_polling honors TELEGRAM_POLL_OWNER.
            telegram_controller = get_telegram_controller(db_pool, module_name='copy')
            if await telegram_controller.initialize():
                telegram_controller.register_module(
                    name='copy_trading',
                    engine=engine,
                    start_method='start',
                    stop_method='stop',
                    positions_attr='active_copies'
                )
                await telegram_controller.start_polling()
                logger.info("📱 Telegram remote control enabled")
                await telegram_controller.notify("Copy Trading Module started. Send /help for commands.", priority="normal", category="lifecycle")
        except Exception as e:
            logger.warning(f"Telegram controller failed to initialize: {e}")

    try:
        # NOTE: the previous "✅ initialized successfully" log here
        # was misleading — main_copy.py constructs the engine object
        # but does NOT call engine.initialize(). The real init runs
        # inside engine.start() (line 714: if self.executor is None →
        # await self.initialize()). When initialize() fails, start()
        # returns False, main() returns, the subprocess exits silently,
        # and the orchestrator restarts it in a loop until max_restarts
        # is reached (the "Copy Trading has failed permanently" log).
        # Surface the failure here so the operator can read the actual
        # error from logs/copy_trading/copy_trading.log instead of
        # spelunking the engine's own logger.

        # Wave-12 FIX 2: derive the Solana execution pubkey from
        # SOLANA_MODULE_PRIVATE_KEY BEFORE engine.start() so
        # _persist_execution_wallets surfaces it on the dashboard funding
        # panel instead of "solana=none". The same class of bug as DEX /
        # Sniper / Copy EVM (Wave-11 FIX 1 / 3 / Wave-12 FIX 1): the
        # stored SOLANA_MODULE_WALLET secret was None for the operator,
        # so copy_engine line 288 (`secrets.get('SOLANA_MODULE_WALLET') or
        # os.getenv(...)`) resolved to None even with a valid PK. We can't
        # edit copy_engine.py (concurrent agent owns it), so we run
        # engine.initialize() here (idempotent), derive the pubkey from
        # the executor's already-decrypted solana_private_key, write it
        # into executor.solana_wallet, then re-call
        # _persist_execution_wallets so the dashboard reads the resolved
        # address. engine.start() will skip re-init since executor is
        # non-None. Public address only — keypair never logged. Mask
        # logged addresses to first6...last4.
        if not await engine.initialize():
            err = getattr(engine, 'error_message', None) or 'unknown (check engine.initialize)'
            logger.error(f"❌ Copy Trading engine.initialize() returned False: {err}")
            logger.error("   The subprocess will exit. Check upstream config / db / secrets.")
            return
        try:
            _exec = getattr(engine, 'executor', None)
            _pk = getattr(_exec, 'solana_private_key', None) if _exec else None
            _existing = getattr(_exec, 'solana_wallet', None) if _exec else None
            if _exec is not None and _pk and not _existing:
                derived_sol = None
                try:
                    from solders.keypair import Keypair
                    import base58
                    key_bytes = None
                    if _pk.startswith('['):
                        try:
                            import json as _json
                            key_bytes = bytes(_json.loads(_pk))
                        except Exception:
                            pass
                    if key_bytes is None:
                        try:
                            key_bytes = base58.b58decode(_pk)
                        except Exception:
                            pass
                    if key_bytes is None:
                        try:
                            key_bytes = bytes.fromhex(_pk)
                        except Exception:
                            pass
                    if key_bytes is not None:
                        if len(key_bytes) == 64:
                            kp = Keypair.from_bytes(key_bytes)
                        elif len(key_bytes) == 32:
                            kp = Keypair.from_seed(key_bytes)
                        else:
                            kp = None
                        if kp is not None:
                            derived_sol = str(kp.pubkey())
                except Exception as e:
                    logger.debug(f"copy Solana wallet derivation failed: {e}")
                if derived_sol:
                    _exec.solana_wallet = derived_sol
                    mask = derived_sol[:6] + "..." + derived_sol[-4:]
                    logger.info(f"🔑 Copy Solana wallet derived from PK: {mask}")
                    # Re-surface to dashboard so funding panel updates.
                    try:
                        await engine._persist_execution_wallets()
                    except Exception as e:
                        logger.warning(f"re-persist execution wallets failed: {e}")
        except Exception as e:
            logger.warning(f"Copy Solana wallet derivation block failed (non-fatal): {e}")

        logger.info("🚀 Copy Trading Engine starting — entering main loop")
        result = await engine.start()
        if not result:
            err = getattr(engine, 'error_message', None) or 'unknown (check engine.initialize)'
            logger.error(f"❌ Copy Trading engine.start() returned False: {err}")
            logger.error("   The subprocess will exit. Check upstream config / db / secrets.")
    except KeyboardInterrupt:
        if telegram_controller:
            await telegram_controller.notify("Copy Trading module shutting down...", priority="high")
            await telegram_controller.stop_polling()
        await engine.stop()
    except Exception as e:
        # Catch-all so the operator sees the traceback in
        # logs/copy_trading/ instead of a silent exit.
        import traceback
        logger.error(f"❌ Copy Trading main loop crashed: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"❌ Engine error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if telegram_controller:
            await telegram_controller.stop_polling()
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
