"""
Sniper Engine - High-speed new token sniping
"""

import asyncio
import logging
import hashlib
import random
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
import os
import aiohttp

from config.config_manager import ConfigManager
from core.dry_run import should_skip_live
from data.storage.database import DatabaseManager
from modules.sniper.core._timing import SnipeTimingContext, parse_iso_to_perf_counter
from monitoring.alerts import AlertManager

logger = logging.getLogger("SniperEngine")


class PriceFetcher:
    """Fetch real-time cryptocurrency prices from CoinGecko"""

    COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

    # Token symbol to CoinGecko ID mapping
    TOKEN_IDS = {
        'sol': 'solana',
        'eth': 'ethereum',
        'bnb': 'binancecoin',
        'matic': 'matic-network',
        'avax': 'avalanche-2',
        'arb': 'arbitrum',
        'base': 'base-protocol'
    }

    # Fallback prices (only used if API fails completely)
    FALLBACK_PRICES = {
        'sol': 200.0,
        'eth': 3500.0,
        'bnb': 600.0,
        'matic': 0.85,
        'avax': 35.0,
        'arb': 1.0,
        'base': 1.0
    }

    def __init__(self):
        self._price_cache: Dict[str, float] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=1)  # 1-minute cache

    async def get_price(self, symbol: str) -> float:
        """Get current USD price for a token symbol"""
        symbol = symbol.lower()

        # Check cache first
        if symbol in self._price_cache:
            cache_age = datetime.now() - self._cache_time.get(symbol, datetime.min)
            if cache_age < self._cache_duration:
                return self._price_cache[symbol]

        # Get CoinGecko ID
        coingecko_id = self.TOKEN_IDS.get(symbol)
        if not coingecko_id:
            return self.FALLBACK_PRICES.get(symbol, 0.0)

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'ids': coingecko_id,
                    'vs_currencies': 'usd'
                }
                async with session.get(self.COINGECKO_API, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get(coingecko_id, {}).get('usd', 0)
                        if price > 0:
                            # Update cache
                            self._price_cache[symbol] = price
                            self._cache_time[symbol] = datetime.now()
                            return price
        except Exception as e:
            logger.debug(f"Price fetch error for {symbol}: {e}")

        # Return cached price if available, otherwise fallback
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        return self.FALLBACK_PRICES.get(symbol, 0.0)

class SniperEngine:
    """
    Sniper Engine for detecting and buying new token launches immediately.
    Supports EVM (Mempool/Events) and Solana (Raydium logs).
    """

    def __init__(self, config: Dict, config_manager: ConfigManager, db_pool):
        self.config = config
        self.config_manager = config_manager
        self.db_pool = db_pool
        self.is_running = False
        self.tasks = []

        # State
        self.pending_targets = {}
        self.active_snipes = {}

        # Components (to be initialized)
        self.evm_listener = None
        self.solana_listener = None
        self.token_safety = None
        self.executor = None
        # Wave-13: cross-module RiskManager gate (injected from main_sniper.py).
        # None when not wired -- engine runs without the gate (DRY_RUN, tests).
        # Only the entry leg is gated; exits are always allowed.
        self.risk_manager = None

        # Settings (loaded from DB)
        self.dry_run = True
        self.trade_amount = 0.1
        self.slippage = 10.0
        self.priority_fee = 5000
        self.max_buy_tax = 15.0
        self.max_sell_tax = 15.0
        self.min_liquidity = 1000.0
        self.test_mode_min_liquidity = 10.0  # Very relaxed for test mode
        self.safety_check_enabled = True
        self.test_mode = False  # Relaxed safety for testing
        # Emergency brake against runaway position accumulation
        # (DRY_RUN stress test hit 10k+ positions in 22h).
        self.max_active_positions = 500
        # Per-cap-slot gas/fee buffer (USD) used by the funding recommendation
        # surfaced for the operator. Conservative default; DB-overridable.
        self.gas_buffer_usd = 0.50
        # SNIPE-RM-12: per-position max hold. 0 = disabled (keep
        # back-compat with existing deployments that have no
        # max_hold_minutes row in config_settings).
        self.max_hold_minutes = 0
        # Wave-3: Pyth Hermes blue-chip price feed. Default TRUE
        # because Pyth is free, independent of Jupiter/Birdeye, and
        # publishes sub-second for the mints in
        # SOLANA_MINT_TO_PYTH_FEED_ID. Pump.fun mints have no feed-id
        # so resolution falls through unaffected.
        self.sniper_pyth_feeds_enabled = True

        # Wave-14: per-chain EVM enable switch + tighter EVM-specific filters.
        # 21h DRY_RUN: EVM 12.7% WR / -0.09% avg (negative expectancy) vs
        # Solana 50.8% WR / +2.81% avg. Default FALSE so EVM is gated off
        # until an operator confirms a profitable sub-segment exists (higher
        # safety score + liquidity floors likely required).
        # Flip sniper_evm_enabled=true in config_settings to re-enable without
        # a code deploy.
        self.sniper_evm_enabled: bool = False
        # EVM-specific filter overrides. Applied on top of (not instead of)
        # global min_liquidity. 0 = inherit global value unchanged.
        # Seeded to tighter-than-global defaults in migration 041.
        self.evm_min_liquidity: float = 0.0   # USD; 0 = use global min_liquidity
        self.evm_min_safety_score: int = 0    # 0-100; 0 = no extra score gate

        # Wave-15: entry quality filters (all DB-configurable via migration 044)
        # Defaults are conservative so the next DRY_RUN window measures the
        # reworked strategy without requiring a manual DB update.
        self.sniper_min_holder_count: int = 10        # 0 = disabled
        self.sniper_min_token_age_seconds: int = 30   # 0 = disabled
        self.sniper_max_dev_holding_pct: float = 30.0 # 100 = disabled
        self.sniper_min_buy_sell_ratio: float = 1.5   # 0.0 = disabled (fail-open)
        self.sniper_min_safety_score: int = 40        # 0 = disabled

        # Wave-16: decouple W15 quality gates from the honeypot-safety-API toggle.
        # When true (default), the W15 entry-quality heuristics run regardless of
        # safety_check_enabled.  Gates that require a SafetyReport (holder count,
        # dev holding, safety score) run only when the safety API call is made;
        # gates that don't need a report (token age, buy-sell ratio) run even when
        # safety_check_enabled=false.  Set false only to disable quality filtering
        # entirely (not recommended for LIVE).  Seeded by migration 052.
        self.sniper_quality_gates_enabled: bool = True

        # Wave-15: partial-take exit (all DB-configurable via migration 044)
        # partial_take_pct  - fire partial exit at this P&L% (0 = disabled)
        # partial_take_size - fraction of position (%) to sell at partial take
        # trail_after_partial - use trailing stop on remainder after partial take
        self.sniper_partial_take_pct: float = 20.0
        self.sniper_partial_take_size_pct: float = 50.0
        self.sniper_trail_after_partial: bool = True

        # Wave-15 risk-gate fix (migration 047): controls whether the sniper
        # runs new-launch targets through the DEX-oriented RiskManager.validate_trade
        # (which rejects <$10k liquidity, blocking 100% of fresh memecoins).
        # FALSE (default): skip DEX liquidity/honeypot/token-risk analysis; only
        #   run capital-protection checks (circuit breakers + allocation guard).
        #   The sniper's own _check_filters + TokenSafetyChecker remain the memecoin
        #   risk layer. Healthy new launches that pass _check_filters are ALLOWED.
        # TRUE: restore old full validate_trade behavior (useful for mature-token
        #   or EVM sniping where DEX liquidity analysis is meaningful).
        self.sniper_use_dex_risk_manager: bool = False

        # Statistics tracking for rate-limited logging
        self._stats = {
            'tokens_analyzed': 0,
            'honeypots_detected': 0,
            'danger_ratings': 0,
            'high_tax_rejected': 0,
            'low_liquidity_rejected': 0,
            'low_holder_rejected': 0,
            'too_young_rejected': 0,
            'high_dev_holding_rejected': 0,
            'low_buy_sell_ratio_rejected': 0,
            'low_score_rejected': 0,
            'passed_safety': 0,
            'partial_takes_fired': 0,
            'positions_synthetic_closed': 0,
            'capped_rejections': 0,
            'last_capped_log': datetime.now(),
            'last_stats_log': datetime.now()
        }

        # Cooldown tracking for rejected/analyzed tokens
        self._rejected_cache: Dict[str, datetime] = {}  # Track recently rejected tokens
        self._cooldown_duration = timedelta(minutes=5)  # 5-minute cooldown per token

        # Price fetcher for real USD values
        self.price_fetcher = PriceFetcher()

        # Per-mint price cache for the Jupiter quote fallback. Monitor loop
        # ticks every second; without a cache we would issue a quote per
        # active position per tick.
        self._mint_price_cache: Dict[str, tuple] = {}  # token -> (price, ts)
        self._mint_price_ttl = timedelta(seconds=15)

    def set_risk_manager(self, risk_manager) -> None:
        """Inject a core.risk_manager.RiskManager instance (Wave-13).

        Called from main_sniper.py after the DB pool is ready. Pattern matches
        arbitrage_engine.py and solana_engine.py. If construction fails the
        caller logs a warning and leaves self.risk_manager=None so the engine
        runs without the cross-module gate (fail-soft, consistent with SOLANA /
        ARB / COPY usage).
        """
        self.risk_manager = risk_manager

    async def initialize(self):
        """Initialize sniper components"""
        logger.info("🔫 Initializing Sniper Engine...")

        # Load settings from DB
        await self._load_settings()
        sniper_config = self.config.get('sniper', {})

        # Initialize Token Safety Checker
        from modules.sniper.core.token_safety import TokenSafetyChecker
        self.token_safety = TokenSafetyChecker(self.config)
        await self.token_safety.initialize()

        # Initialize Trade Executor
        from modules.sniper.core.trade_executor import TradeExecutor
        self.executor = TradeExecutor(self.config)
        await self.executor.initialize()

        # Initialize Listeners based on enabled chains.
        # EVM listener is further gated by sniper_evm_enabled (DB key loaded in
        # _load_settings). sniper_evm_enabled defaults FALSE because 21h DRY_RUN
        # showed EVM at 12.7% WR / -0.09% avg (negative expectancy). Set
        # sniper_evm_enabled=true in config_settings to re-enable when EVM
        # filter thresholds are tightened enough to restore positive expectancy.
        if sniper_config.get('evm_enabled', True) and self.sniper_evm_enabled:
            from modules.sniper.core.evm_listener import EVMListener
            self.evm_listener = EVMListener(self.config)
            await self.evm_listener.initialize()
        elif sniper_config.get('evm_enabled', True) and not self.sniper_evm_enabled:
            logger.warning(
                "EVM sniping DISABLED by sniper_evm_enabled=false (DB config). "
                "Solana path is unaffected. To re-enable: UPDATE config_settings "
                "SET value='true' WHERE config_type='sniper_config' "
                "AND key='sniper_evm_enabled';"
            )

        if sniper_config.get('solana_enabled', True):
            from modules.sniper.core.solana_listener import SolanaListener
            self.solana_listener = SolanaListener(self.config)
            await self.solana_listener.initialize()

        logger.info("✅ Sniper Engine initialized")

        # Wave-11 FIX 2: surface wallets to the dashboard funding panel
        # IMMEDIATELY at the end of initialize(), not only on the first
        # _log_stats_if_needed window flip (which can be 1+ min away on a
        # quiet listener) and not only on the first trade. The
        # TradeExecutor.initialize() above resolved evm_wallet /
        # solana_wallet from secrets; persist that snapshot now so the
        # dashboard stops showing "no wallet in runtime stats" the
        # instant the subprocess is up. Idempotent (UPSERT id=1); the
        # run()-loop seed at line 319 then overwrites with full counters.
        try:
            await self._persist_runtime_stats()
            _exec = getattr(self, 'executor', None)
            sol = getattr(_exec, 'solana_wallet', None) if _exec else None
            evm = getattr(_exec, 'evm_wallet', None) if _exec else None
            sol_mask = (sol[:6] + "..." + sol[-4:]) if sol else "none"
            evm_mask = (evm[:6] + "..." + evm[-4:]) if evm else "none"
            logger.info(f"🔑 Sniper wallets surfaced: solana={sol_mask} evm={evm_mask}")
        except Exception as e:
            logger.debug(f"initial wallet surfacing failed (non-fatal): {e}")

    async def _load_settings(self):
        """Load settings from database"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT key, value FROM config_settings WHERE config_type = 'sniper_config'"
                    )
                    for row in rows:
                        key = row['key']
                        val = row['value']
                        if key == 'enabled':
                            pass  # Handled by orchestrator
                        elif key == 'trade_amount':
                            self.trade_amount = float(val) if val else 0.1
                        elif key == 'slippage':
                            self.slippage = float(val) if val else 10.0
                        elif key == 'priority_fee':
                            self.priority_fee = int(val) if val else 5000
                        elif key == 'max_buy_tax':
                            self.max_buy_tax = float(val) if val else 15.0
                        elif key == 'max_sell_tax':
                            self.max_sell_tax = float(val) if val else 15.0
                        elif key == 'min_liquidity':
                            self.min_liquidity = float(val) if val else 1000.0
                        elif key == 'test_mode_min_liquidity':
                            self.test_mode_min_liquidity = float(val) if val else 10.0
                        elif key == 'safety_check_enabled':
                            self.safety_check_enabled = val.lower() in ('true', '1', 'yes') if val else True
                        elif key == 'test_mode':
                            self.test_mode = val.lower() in ('true', '1', 'yes') if val else False
                        elif key == 'chain':
                            self.target_chain = val if val else 'solana'  # 'all', 'solana', 'ethereum', etc.
                        elif key == 'take_profit_pct':
                            self.take_profit_pct = float(val) if val else 50.0
                        elif key == 'stop_loss_pct':
                            self.stop_loss_pct = float(val) if val else 20.0
                        elif key == 'max_active_positions':
                            self.max_active_positions = int(val) if val else 500
                        elif key == 'max_hold_minutes':
                            self.max_hold_minutes = int(val) if val else 0
                        elif key == 'gas_buffer_usd':
                            self.gas_buffer_usd = float(val) if val else 0.50
                        elif key == 'sniper_pyth_feeds_enabled':
                            self.sniper_pyth_feeds_enabled = (
                                val.lower() in ('true', '1', 'yes') if val else True
                            )
                        # Wave-14: per-chain EVM enable + tighter EVM filters
                        elif key == 'sniper_evm_enabled':
                            self.sniper_evm_enabled = (
                                val.lower() in ('true', '1', 'yes') if val else False
                            )
                        elif key == 'evm_min_liquidity':
                            self.evm_min_liquidity = float(val) if val else 0.0
                        elif key == 'evm_min_safety_score':
                            self.evm_min_safety_score = int(val) if val else 0
                        # Wave-15: entry quality gates
                        elif key == 'sniper_min_holder_count':
                            self.sniper_min_holder_count = int(val) if val else 10
                        elif key == 'sniper_min_token_age_seconds':
                            self.sniper_min_token_age_seconds = int(val) if val else 30
                        elif key == 'sniper_max_dev_holding_pct':
                            self.sniper_max_dev_holding_pct = float(val) if val else 30.0
                        elif key == 'sniper_min_buy_sell_ratio':
                            self.sniper_min_buy_sell_ratio = float(val) if val else 1.5
                        elif key == 'sniper_min_safety_score':
                            self.sniper_min_safety_score = int(val) if val else 40
                        # Wave-15: exit rework
                        elif key == 'sniper_partial_take_pct':
                            self.sniper_partial_take_pct = float(val) if val else 20.0
                        elif key == 'sniper_partial_take_size_pct':
                            self.sniper_partial_take_size_pct = float(val) if val else 50.0
                        elif key == 'sniper_trail_after_partial':
                            self.sniper_trail_after_partial = (
                                val.lower() in ('true', '1', 'yes') if val else True
                            )
                        # Wave-15 risk-gate fix (migration 047)
                        elif key == 'sniper_use_dex_risk_manager':
                            self.sniper_use_dex_risk_manager = (
                                val.lower() in ('true', '1', 'yes') if val else False
                            )
                        # Wave-16: quality gates independent of safety_check_enabled
                        # (migration 052)
                        elif key == 'sniper_quality_gates_enabled':
                            self.sniper_quality_gates_enabled = (
                                val.lower() in ('true', '1', 'yes') if val else True
                            )

            # Check for DRY_RUN mode
            self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

            # Log mode info
            mode_info = []
            if self.dry_run:
                mode_info.append("DRY_RUN (no real trades)")
            else:
                mode_info.append("LIVE TRADING")
            if self.test_mode:
                mode_info.append("TEST_MODE (relaxed safety)")

            target = getattr(self, 'target_chain', 'solana')
            logger.info(f"📋 Sniper settings loaded: trade_amount={self.trade_amount}, slippage={self.slippage}%")
            logger.info(f"   Mode: {' | '.join(mode_info)}")
            logger.info(f"   Target Chain: {target.upper()} {'(All Chains)' if target == 'all' else ''}")
            if self.test_mode:
                liq_msg = "DISABLED (set to $0)" if self.test_mode_min_liquidity <= 0 else f"${self.test_mode_min_liquidity}"
                logger.info(f"   Safety (TEST MODE): max_tax=50%, min_liq={liq_msg}")
            else:
                logger.info(f"   Safety: max_tax={self.max_buy_tax}%, min_liq=${self.min_liquidity}")
            logger.info(f"   Exit: TP={getattr(self, 'take_profit_pct', 50)}%, SL={getattr(self, 'stop_loss_pct', 20)}%")
            evm_liq_msg = (
                f"${self.evm_min_liquidity:,.0f}" if self.evm_min_liquidity > 0
                else f"inherit global (${self.min_liquidity:,.0f})"
            )
            evm_score_msg = (
                f">={self.evm_min_safety_score}" if self.evm_min_safety_score > 0
                else "no extra gate"
            )
            logger.info(
                f"   EVM sniping: {'ENABLED' if self.sniper_evm_enabled else 'DISABLED'} "
                f"(sniper_evm_enabled={self.sniper_evm_enabled}) | "
                f"evm_min_liq={evm_liq_msg} | evm_min_score={evm_score_msg}"
            )
            logger.info(
                f"   W15 exit: PartialTake={self.sniper_partial_take_pct}%"
                f"@{self.sniper_partial_take_size_pct}% trail={self.sniper_trail_after_partial}"
            )
            quality_gate_mode = (
                "ACTIVE (always-on)" if self.sniper_quality_gates_enabled
                else "DISABLED (sniper_quality_gates_enabled=false)"
            )
            logger.info(
                f"   W15/W16 entry gates [{quality_gate_mode}]: "
                f"min_holders={self.sniper_min_holder_count} "
                f"min_age={self.sniper_min_token_age_seconds}s "
                f"max_dev_holding={self.sniper_max_dev_holding_pct}% "
                f"min_buy_sell={self.sniper_min_buy_sell_ratio} "
                f"min_score={self.sniper_min_safety_score} "
                f"| safety_api={'ON' if self.safety_check_enabled else 'OFF'}"
            )
            logger.info(
                f"   W15 risk gate: sniper_use_dex_risk_manager="
                f"{self.sniper_use_dex_risk_manager} "
                f"({'full DEX validate_trade' if self.sniper_use_dex_risk_manager else 'sniper-own gates only (capital checks preserved)'})"
            )

        except Exception as e:
            logger.error(f"Error loading sniper settings: {e}")

        # LIVE-trading safety guard: refuse to start sniping with the
        # safety filter disabled while not in DRY_RUN. Phase 2 turned
        # safety_check_enabled off to measure raw volume; leaving it
        # off on a LIVE flip would buy honeypots indiscriminately.
        # Kept OUTSIDE the try/except so the RuntimeError actually
        # propagates and halts the subprocess instead of being logged.
        if not self.dry_run and not self.safety_check_enabled:
            logger.critical(
                "🛑 REFUSING TO RUN: safety_check_enabled=false while DRY_RUN=false. "
                "Run this in DB to re-enable before going live: "
                "UPDATE config_settings SET value='true' WHERE config_type='sniper_config' "
                "AND key='safety_check_enabled';"
            )
            raise RuntimeError(
                "Sniper refused to start: safety_check_enabled=false in LIVE mode"
            )

        # SNIPE-RM-19: test_mode + LIVE is unsafe. test_mode relaxes
        # the tax/liquidity gates (allows DANGER-rated tokens for
        # measurement); combining it with LIVE means buying tokens
        # that would normally be filtered. Refuse to start.
        if not self.dry_run and getattr(self, 'test_mode', False):
            logger.critical(
                "🛑 REFUSING TO RUN: test_mode=true while DRY_RUN=false. "
                "test_mode bypasses honeypot/tax gates and is only safe in DRY_RUN. "
                "Disable in DB before going live: "
                "UPDATE config_settings SET value='false' WHERE config_type='sniper_config' "
                "AND key='test_mode';"
            )
            raise RuntimeError(
                "Sniper refused to start: test_mode=true in LIVE mode"
            )

    async def run(self):
        """Main loop"""
        self.is_running = True
        logger.info("🔫 Sniper Engine Started")

        # Seed the runtime-stats snapshot so the standalone dashboard
        # has data immediately instead of waiting ~5 min for the first
        # periodic emission. Fail-soft.
        try:
            await self._persist_runtime_stats()
        except Exception as e:
            logger.debug(f"initial _persist_runtime_stats failed (non-fatal): {e}")

        self.tasks = [
            asyncio.create_task(self._monitor_new_pairs()),
            asyncio.create_task(self._process_targets()),
            asyncio.create_task(self._monitor_active_snipes())
        ]

        await asyncio.gather(*self.tasks)

    async def _monitor_new_pairs(self):
        """Listen for new pair events"""
        scan_count = 0
        evm_pairs_found = 0
        sol_pools_found = 0

        while self.is_running:
            try:
                scan_count += 1

                # 1. Check EVM Mempool/Events
                if self.evm_listener:
                    events = await self.evm_listener.get_new_pairs()
                    for event in events:
                        evm_pairs_found += 1
                        await self._evaluate_target(event, 'evm')

                # 2. Check Solana Raydium Logs
                if self.solana_listener:
                    events = await self.solana_listener.get_new_pools()
                    for event in events:
                        sol_pools_found += 1
                        await self._evaluate_target(event, 'solana')

                # Log status every 6000 scans (~10 minutes at 0.1s interval)
                if scan_count % 6000 == 0:
                    logger.info(f"🔫 Monitor status: {scan_count} scans, {evm_pairs_found} EVM pairs, {sol_pools_found} Solana pools detected")

                await asyncio.sleep(0.1) # Fast loop
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(1)

    async def _effective_active_count(self) -> int:
        """Return the count to evaluate against max_active_positions.

        max(in-memory, DB) so the cap honors both:
          - In-memory dict: positions THIS process has opened (resets to
            0 on restart, so alone it's worthless after a crash).
          - DB sniper_trades WHERE status='open': true active across
            restarts, including orphan rows from prior crashes.

        Prevents the dashboard "Active Positions: 1089 / cap: 500"
        skew that surfaced when len(self.active_snipes)=0 after a
        restart while the DB still carried 1089 orphans.

        Fail-soft: a DB error falls back to in-memory count rather
        than blocking trading.
        """
        in_mem = len(self.active_snipes)
        if not self.db_pool:
            return in_mem
        try:
            async with self.db_pool.acquire() as conn:
                db_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM sniper_trades WHERE status = 'open'"
                )
            return max(in_mem, int(db_count or 0))
        except Exception as e:
            logger.debug(f"_effective_active_count DB fallback: {e}")
            return in_mem

    async def _evaluate_target(self, target: Dict, chain_type: str):
        """Evaluate if a new token meets sniping criteria"""
        token_address = target.get('token_address', '')
        # SNIPE-RM-18: dedupe before doing any work. The same address
        # can arrive on multiple listener paths (polling + WSS race,
        # or repeat block scans), and without this gate we'd burn
        # safety-check API calls + log duplicate "TARGET ACQUIRED"
        # lines for each. Cheap O(1) check against the live pending +
        # active sets.
        if token_address and (
            token_address in self.pending_targets
            or token_address in self.active_snipes
        ):
            return
        # t_rpc_receipt is stamped by the listener BEFORE getTransaction
        # commitment-wait. Read from either top-level (EVM listener) or
        # nested metadata (Solana listener) so both detection paths
        # surface the WSS-vs-polling staleness delta.
        rpc_receipt_perf = (
            target.get('rpc_receipt_perf')
            or target.get('metadata', {}).get('rpc_receipt_perf')
        )
        timing = SnipeTimingContext(
            token_address=token_address or '',
            chain=chain_type,
            t_detect=parse_iso_to_perf_counter(target.get('timestamp', '')),
            t_rpc_receipt=rpc_receipt_perf,
        )
        target['_timing'] = timing
        try:
            # 0. Active-positions cap (emergency brake against runaway accumulation).
            # Gate here so we don't pay safety-check cost when already at cap.
            # Uses _effective_active_count() which prefers DB count over the
            # in-memory dict — len(self.active_snipes) resets to 0 on restart
            # while DB orphans accumulate, leaving the cap unenforced.
            effective = await self._effective_active_count()
            if effective >= self.max_active_positions:
                self._stats['capped_rejections'] = self._stats.get('capped_rejections', 0) + 1
                now = datetime.now()
                if now - self._stats.get('last_capped_log', now) >= timedelta(minutes=1):
                    logger.warning(
                        f"🛑 SNIPER CAP: {effective}/{self.max_active_positions} "
                        f"active positions (in-mem={len(self.active_snipes)}, "
                        f"db-open={effective}) — rejected {self._stats['capped_rejections']} "
                        f"candidates in last minute"
                    )
                    self._stats['last_capped_log'] = now
                    self._stats['capped_rejections'] = 0
                timing.outcome = 'rejected_capped'
                try:
                    timing.emit()
                except Exception:
                    pass
                return

            # 1. Check Filters (Liquidity, Tax, Honeypot, Safety)
            if not await self._check_filters(target, chain_type):
                # _check_filters sets timing.outcome (rejected_filter
                # or rejected_safety) before returning False. Emit
                # here so the rejection's safety-stage cost is logged.
                if timing.outcome == 'pending':
                    timing.outcome = 'rejected_filter'
                try:
                    timing.emit()
                except Exception:
                    pass
                return

            # 2. Add to pending targets (demoted to DEBUG — thousands/day in DRY_RUN)
            logger.debug(f"🎯 SNIPER TARGET ACQUIRED: {token_address} ({chain_type})")
            self.pending_targets[token_address] = {
                'target': target,
                'chain_type': chain_type,
                'timestamp': datetime.now(),
                'status': 'pending'
            }

        except Exception as e:
            logger.error(f"Error evaluating target: {e}")

    async def _check_filters(self, target: Dict, chain_type: str) -> bool:
        """Apply strict filters for sniping including token safety checks"""
        # Basic filter: check liquidity presence
        if not target.get('pair_address'):
            return False

        token_address = target.get('token_address')
        if not token_address:
            return False

        # COOLDOWN CHECK: Skip if we recently analyzed/rejected this token
        if token_address in self._rejected_cache:
            last_check = self._rejected_cache[token_address]
            if datetime.now() - last_check < self._cooldown_duration:
                # Still in cooldown, skip silently
                return False
            else:
                # Cooldown expired, remove from cache
                del self._rejected_cache[token_address]

        # Update stats
        self._stats['tokens_analyzed'] += 1

        # Log stats periodically (every 1 minute)
        await self._log_stats_if_needed()

        timing = target.get('_timing')

        # Wave-16: quality gates that don't require a SafetyReport run here,
        # BEFORE the safety_check_enabled guard.  This means they fire even
        # when the honeypot-API check is disabled (TEST_MODE / DRY_RUN with
        # safety_check_enabled=false).  Gate 2 (token age) and Gate 5 (buy-
        # sell ratio) are cheap heuristics — no external API call needed.
        if self.sniper_quality_gates_enabled and not self.test_mode:
            # Gate 2 (early): token age window.
            # pool block_time stamped by listener into target['block_time']
            # or target['metadata']['pool_block_time']. Fail-open when absent.
            min_age = self.sniper_min_token_age_seconds
            if min_age > 0:
                pool_block_time = (
                    target.get('block_time')
                    or target.get('metadata', {}).get('pool_block_time')
                )
                if pool_block_time is not None:
                    try:
                        from datetime import timezone as _tz
                        if isinstance(pool_block_time, (int, float)):
                            bt = datetime.fromtimestamp(pool_block_time, tz=_tz.utc)
                        else:
                            bt = pool_block_time
                        now_utc = datetime.now(tz=_tz.utc)
                        age_secs = (now_utc - bt).total_seconds()
                        if age_secs < min_age:
                            self._stats['too_young_rejected'] = (
                                self._stats.get('too_young_rejected', 0) + 1
                            )
                            self._rejected_cache[token_address] = datetime.now()
                            logger.debug(
                                f"Token too young: {token_address[:16]}... "
                                f"({age_secs:.1f}s < {min_age}s)"
                            )
                            if timing:
                                timing.outcome = 'rejected_quality'
                            return False
                    except Exception as _age_err:
                        logger.debug(
                            f"token age gate parse error (fail-open): {_age_err}"
                        )

            # Gate 5 (early): buy/sell pressure ratio (fail-open if unavailable).
            # Solana only — calls Birdeye trade-stats asynchronously; any
            # error allows candidate through (Birdeye outage != reject all).
            min_bsr = self.sniper_min_buy_sell_ratio
            if min_bsr > 0 and chain_type == 'solana':
                bsr = await self._get_buy_sell_ratio(token_address)
                if bsr is not None and bsr < min_bsr:
                    self._stats['low_buy_sell_ratio_rejected'] = (
                        self._stats.get('low_buy_sell_ratio_rejected', 0) + 1
                    )
                    self._rejected_cache[token_address] = datetime.now()
                    logger.debug(
                        f"Buy/sell ratio too low: {token_address[:16]}... "
                        f"({bsr:.2f} < {min_bsr})"
                    )
                    if timing:
                        timing.outcome = 'rejected_quality'
                    return False

        # Skip safety API check if disabled. Quality gates that require a
        # SafetyReport (holder count, dev holding, safety score) are gated
        # inside the safety block below and won't run when the safety API
        # is disabled — they fail-open in that case, which is intentional.
        if not self.safety_check_enabled:
            logger.debug(f"Safety API disabled, skipping honeypot/tax/liq check for {token_address}")
            self._stats['passed_safety'] += 1
            return True

        # Test mode uses relaxed thresholds
        if self.test_mode:
            max_buy_tax = 50.0     # Allow up to 50% in test mode
            max_sell_tax = 50.0
            min_liquidity = self.test_mode_min_liquidity  # Use configurable test mode liquidity (default $10)
            allow_caution = True   # Allow CAUTION rated tokens
        else:
            max_buy_tax = self.max_buy_tax
            max_sell_tax = self.max_sell_tax
            min_liquidity = self.min_liquidity
            allow_caution = False

        # Wave-14: EVM-specific filter overrides applied on top of global gates.
        # evm_min_liquidity>0 raises the liquidity floor for EVM-only candidates.
        # evm_min_safety_score>0 adds a hard score floor (0-100 subtractive scale;
        # higher score = safer token). Both ignored in test_mode.
        if chain_type == 'evm' and not self.test_mode:
            if self.evm_min_liquidity > 0:
                min_liquidity = max(min_liquidity, self.evm_min_liquidity)
            evm_score_floor = self.evm_min_safety_score  # 0 means gate disabled
        else:
            evm_score_floor = 0

        # Perform comprehensive safety check (timing already captured above)
        try:
            if self.token_safety:
                from modules.sniper.core.token_safety import SafetyRating

                if timing:
                    timing.stamp('t_safety_start')
                report = await self.token_safety.check_token(token_address, chain_type)
                if timing:
                    timing.stamp('t_safety_done')

                # Log the safety report
                target['safety_report'] = {
                    'rating': report.rating.value,
                    'score': report.score,
                    'is_honeypot': report.is_honeypot,
                    'buy_tax': report.buy_tax,
                    'sell_tax': report.sell_tax,
                    'liquidity_usd': report.liquidity_usd,
                    'warnings': report.warnings[:5]
                }

                # Check if safe to snipe (rate-limited logging - only log at DEBUG level)
                if report.is_honeypot:
                    self._stats['honeypots_detected'] += 1
                    self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                    logger.debug(f"🍯 HONEYPOT: {token_address[:16]}...")
                    if timing:
                        timing.outcome = 'rejected_safety'
                    return False

                if report.rating == SafetyRating.DANGER:
                    self._stats['danger_ratings'] += 1
                    # In test mode, allow DANGER tokens but log warning
                    if self.test_mode:
                        logger.warning(f"⚠️ TEST MODE: Allowing DANGER token {token_address[:16]}...")
                    else:
                        self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                        logger.debug(f"🚨 DANGER: {token_address[:16]}... (Score: {report.score})")
                        if timing:
                            timing.outcome = 'rejected_safety'
                        return False

                if report.rating == SafetyRating.CAUTION and not allow_caution and not self.test_mode:
                    logger.debug(f"⚠️ CAUTION: {token_address[:16]}... (Score: {report.score})")
                    # Allow CAUTION tokens in production (they often have minor issues)

                if report.buy_tax > max_buy_tax or report.sell_tax > max_sell_tax:
                    self._stats['high_tax_rejected'] += 1
                    self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                    logger.debug(f"⚠️ High tax: {token_address[:16]}... (Buy: {report.buy_tax:.1f}%, Sell: {report.sell_tax:.1f}%)")
                    if timing:
                        timing.outcome = 'rejected_safety'
                    return False

                # Skip liquidity check if min_liquidity is 0 (useful for devnet testing)
                if min_liquidity > 0 and report.liquidity_usd < min_liquidity:
                    self._stats['low_liquidity_rejected'] += 1
                    self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                    logger.debug(f"⚠️ Low liquidity: {token_address[:16]}... (${report.liquidity_usd:,.0f} < ${min_liquidity:,.0f})")
                    if timing:
                        timing.outcome = 'rejected_safety'
                    return False

                # Wave-14: EVM-specific safety score floor. score is a
                # subtractive-penalty integer (0=DANGER, 100=perfect). Tokens
                # below evm_score_floor are rejected even if rating is not
                # DANGER — targeting the CAUTION band that drives EVM losses.
                if evm_score_floor > 0 and report.score < evm_score_floor:
                    self._rejected_cache[token_address] = datetime.now()
                    logger.debug(
                        f"EVM score floor: {token_address[:16]}... "
                        f"(score {report.score} < floor {evm_score_floor})"
                    )
                    if timing:
                        timing.outcome = 'rejected_safety'
                    return False

                # Wave-15/W16 entry quality gates requiring a SafetyReport.
                # Gates 2 (age) and 5 (buy-sell ratio) already ran above because
                # they don't need a report and must fire even when safety_check_enabled=false.
                # Gates 1, 3, 4 need report data; they run here when quality gates
                # are enabled and we're not in test_mode.
                if self.sniper_quality_gates_enabled and not self.test_mode:
                    # Gate 1: minimum holder count.
                    # report.holder_count populated by GoPlus for EVM; Solana
                    # defaults to 0 when unavailable -> fail-open (don't mass-reject
                    # Solana candidates on missing data).
                    min_h = self.sniper_min_holder_count
                    if min_h > 0 and report.holder_count > 0 and report.holder_count < min_h:
                        self._stats['low_holder_rejected'] = (
                            self._stats.get('low_holder_rejected', 0) + 1
                        )
                        self._rejected_cache[token_address] = datetime.now()
                        logger.debug(
                            f"Low holder count: {token_address[:16]}... "
                            f"({report.holder_count} < {min_h})"
                        )
                        if timing:
                            timing.outcome = 'rejected_quality'
                        return False

                    # Gate 3: dev / top-holder concentration.
                    # report.top_holder_percentage from GoPlus (EVM) + RugCheck (Solana).
                    # Fail-open when 0 (data missing).
                    max_dev = self.sniper_max_dev_holding_pct
                    if max_dev < 100 and report.top_holder_percentage > 0:
                        if report.top_holder_percentage > max_dev:
                            self._stats['high_dev_holding_rejected'] = (
                                self._stats.get('high_dev_holding_rejected', 0) + 1
                            )
                            self._rejected_cache[token_address] = datetime.now()
                            logger.debug(
                                f"Dev holding too high: {token_address[:16]}... "
                                f"({report.top_holder_percentage:.1f}% > {max_dev}%)"
                            )
                            if timing:
                                timing.outcome = 'rejected_quality'
                            return False

                    # Gate 4: minimum safety score floor (subtractive-penalty).
                    # score=0 = DANGER, 100 = perfect. Floor at 40 rejects
                    # low-CAUTION tokens that pass the DANGER/HONEYPOT check.
                    min_score = self.sniper_min_safety_score
                    if min_score > 0 and report.score < min_score:
                        self._stats['low_score_rejected'] = (
                            self._stats.get('low_score_rejected', 0) + 1
                        )
                        self._rejected_cache[token_address] = datetime.now()
                        logger.debug(
                            f"Safety score below floor: {token_address[:16]}... "
                            f"(score {report.score} < floor {min_score})"
                        )
                        if timing:
                            timing.outcome = 'rejected_quality'
                        return False

                # Token passed all checks - log this at INFO level
                self._stats['passed_safety'] += 1
                mode_tag = "[TEST] " if self.test_mode else ""
                logger.info(
                    f"✅ {mode_tag}Safety PASSED: {token_address} "
                    f"(Score: {report.score}/100, Liq: ${report.liquidity_usd:,.0f}, "
                    f"Holders: {report.holder_count})"
                )
                return True

        except Exception as e:
            # R2: log + cooldown the token + bump a counter so the
            # dashboard can surface persistent API outages. Without
            # the cooldown the same token would retry every poll tick
            # against GoPlus/Honeypot.is and burn rate-limit budget.
            logger.error(f"Error during safety check for {token_address}: {e}")
            self._stats['safety_check_errors'] = (
                self._stats.get('safety_check_errors', 0) + 1
            )
            self._rejected_cache[token_address] = datetime.now()
            if timing:
                timing.outcome = 'rejected_safety_error'
            # Fail safe - don't snipe if safety check errors
            return False

        return True

    async def _get_buy_sell_ratio(self, token_address: str) -> Optional[float]:
        """Wave-15: fetch buy/sell volume ratio from Birdeye trade-stats.

        Returns the ratio buys/sells over the last ~50 recent swaps, or None
        when data is unavailable (fail-open — caller treats None as pass).
        Uses a 30s TTL stored in the rejected_cache namespace to avoid
        hammering Birdeye on the hot filter path.
        """
        _cache_key = f"bsr:{token_address}"
        cached = self._rejected_cache.get(_cache_key)
        if cached and isinstance(cached, tuple):
            ts, val = cached
            if (datetime.now() - ts).total_seconds() < 30:
                return val
        try:
            url = (
                f"https://public-api.birdeye.so/defi/txs/token"
                f"?address={token_address}&tx_type=swap&offset=0&limit=50"
            )
            headers = {'X-Chain': 'solana', 'accept': 'application/json'}
            try:
                from security.secrets_manager import secrets
                api_key = secrets.get('BIRDEYE_API_KEY', default=None, log_access=False)
            except Exception:
                api_key = None
            if api_key:
                headers['X-API-KEY'] = api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=4) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    items = (data.get('data') or {}).get('items') or []
                    if not items:
                        return None
                    buys = sum(1 for tx in items if tx.get('side') in ('buy', 'Buy'))
                    sells = sum(1 for tx in items if tx.get('side') in ('sell', 'Sell'))
                    if sells == 0:
                        ratio: Optional[float] = float('inf') if buys > 0 else None
                    else:
                        ratio = buys / sells
                    self._rejected_cache[_cache_key] = (datetime.now(), ratio)
                    return ratio
        except Exception as e:
            logger.debug(f"buy_sell_ratio fetch error for {token_address}: {e}")
        return None

    async def _log_stats_if_needed(self):
        """Log filter statistics every 1 minute (was 5; tightened for
        faster dashboard refresh + Phase 2 iteration loop)."""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 60:  # 1 minute
            total = self._stats['tokens_analyzed']
            passed = self._stats['passed_safety']
            pass_rate = (passed / total * 100) if total > 0 else 0

            logger.info(
                f"SNIPER STATS (Last 1 min): "
                f"Analyzed: {total} | Passed: {passed} ({pass_rate:.1f}%) | "
                f"Honeypots: {self._stats['honeypots_detected']} | "
                f"Danger: {self._stats['danger_ratings']} | "
                f"HighTax: {self._stats['high_tax_rejected']} | "
                f"LowLiq: {self._stats['low_liquidity_rejected']} | "
                f"LowHolders: {self._stats.get('low_holder_rejected', 0)} | "
                f"TooYoung: {self._stats.get('too_young_rejected', 0)} | "
                f"HighDev: {self._stats.get('high_dev_holding_rejected', 0)} | "
                f"LowBSR: {self._stats.get('low_buy_sell_ratio_rejected', 0)} | "
                f"LowScore: {self._stats.get('low_score_rejected', 0)} | "
                f"PartialTakes: {self._stats.get('partial_takes_fired', 0)}"
            )

            # Persist a snapshot for the standalone dashboard before
            # resetting the rolling window. Fail-soft.
            await self._persist_runtime_stats()

            # Reset stats. safety_check_errors + jupiter_quote_fallback_hits
            # are cumulative-by-design counters surfaced to the dashboard
            # via _persist_runtime_stats; preserving them across the window
            # flip prevents the "always 0" trap that hid R3-class issues.
            self._stats = {
                'tokens_analyzed': 0,
                'honeypots_detected': 0,
                'danger_ratings': 0,
                'high_tax_rejected': 0,
                'low_liquidity_rejected': 0,
                'low_holder_rejected': 0,
                'too_young_rejected': 0,
                'high_dev_holding_rejected': 0,
                'low_buy_sell_ratio_rejected': 0,
                'low_score_rejected': 0,
                'passed_safety': 0,
                'partial_takes_fired': 0,
                'positions_synthetic_closed': 0,
                'capped_rejections': 0,
                'safety_check_errors': self._stats.get('safety_check_errors', 0),
                'jupiter_quote_fallback_hits': self._stats.get('jupiter_quote_fallback_hits', 0),
                'birdeye_fallback_hits': self._stats.get('birdeye_fallback_hits', 0),
                'pyth_fallback_hits': self._stats.get('pyth_fallback_hits', 0),
                'last_capped_log': now,
                'last_stats_log': now
            }

    async def _persist_runtime_stats(self) -> None:
        """Snapshot current in-process stats to sniper_runtime_stats so the
        standalone dashboard can read counters that otherwise only exist
        in this subprocess. Fail-soft; never breaks the trading loop."""
        if not self.db_pool:
            return
        try:
            # Merge engine + listener stats into a single dict
            snapshot = dict(self._stats) if hasattr(self, '_stats') else {}
            # Pull listener stats if available
            if hasattr(self, 'solana_listener') and self.solana_listener is not None:
                try:
                    sl_stats = getattr(self.solana_listener, '_stats', {}) or {}
                    snapshot['solana_listener'] = dict(sl_stats)
                except Exception:
                    pass
            if hasattr(self, 'evm_listener') and self.evm_listener is not None:
                try:
                    el_stats = getattr(self.evm_listener, '_stats', {}) or {}
                    snapshot['evm_listener'] = dict(el_stats)
                    # known_pairs is a set — surface its size as a counter
                    kp = getattr(self.evm_listener, 'known_pairs', None)
                    if kp is not None:
                        try:
                            snapshot['evm_listener']['known_pairs_total'] = len(kp)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Derive the 4 dashboard counters from merged snapshot.
            # pools_passed uses passed_safety directly (incremented only
            # when a candidate clears ALL gates) instead of the old
            # evaluated-minus-rejected formula, which under-counted
            # quality-gate rejections and showed Passed>0 even when
            # safety_check_enabled=false short-circuited all checks.
            snapshot['pools_detected'] = (
                (snapshot.get('solana_listener', {}).get('pools_detected') or 0)
                + (snapshot.get('evm_listener', {}).get('known_pairs_total') or 0)
            )
            snapshot['pools_evaluated'] = snapshot.get('tokens_analyzed', 0)
            snapshot['pools_rejected'] = (
                snapshot.get('honeypots_detected', 0)
                + snapshot.get('danger_ratings', 0)
                + snapshot.get('high_tax_rejected', 0)
                + snapshot.get('low_liquidity_rejected', 0)
                + snapshot.get('low_holder_rejected', 0)
                + snapshot.get('too_young_rejected', 0)
                + snapshot.get('high_dev_holding_rejected', 0)
                + snapshot.get('low_buy_sell_ratio_rejected', 0)
                + snapshot.get('low_score_rejected', 0)
            )
            snapshot['pools_passed'] = snapshot.get('passed_safety', 0)
            # Active-positions cap visibility for the dashboard. Surface
            # BOTH in-memory (what THIS process tracks) and the effective
            # count (max of in-mem vs DB open rows) so operators can spot
            # orphan accumulation immediately.
            snapshot['active_positions'] = len(self.active_snipes)
            snapshot['max_active_positions'] = self.max_active_positions
            try:
                snapshot['active_positions_effective'] = await self._effective_active_count()
            except Exception:
                snapshot['active_positions_effective'] = snapshot['active_positions']

            # Funding guidance (issue 12). Surface the USD notional already
            # committed to open positions plus a recommended wallet balance so
            # the operator gets a concrete number to fund for LIVE. Math is
            # documented in modules/sniper/CLAUDE.md.
            try:
                funding = await self._compute_funding_recommendation()
                snapshot.update(funding)
            except Exception as e:
                logger.debug(f"funding recommendation failed (non-fatal): {e}")

            # Wallet identity (issue 15). Public address only — NEVER the key.
            # Sniper Solana shares SOLANA_MODULE_WALLET with the solana_trading
            # module; EVM uses WALLET_ADDRESS / EVM_WALLET_ADDRESS.
            #
            # Wave-11 FIX 2: the wallet addresses live on `self.executor`
            # (TradeExecutor.{solana_wallet,evm_wallet}), NOT on the engine
            # itself — the previous `getattr(self, 'solana_wallet', ...)`
            # always returned None and the dashboard funding panel showed
            # "Sniper: no wallet in runtime stats". Read through executor.
            _exec = getattr(self, 'executor', None)
            sol_wallet = getattr(_exec, 'solana_wallet', None) if _exec else None
            evm_wallet = getattr(_exec, 'evm_wallet', None) if _exec else None
            snapshot['wallet_address'] = sol_wallet or evm_wallet or None
            snapshot['solana_wallet_address'] = sol_wallet or None
            snapshot['evm_wallet_address'] = evm_wallet or None

            import json as _json
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sniper_runtime_stats (id, updated_at, stats)
                    VALUES (1, NOW(), $1::jsonb)
                    ON CONFLICT (id) DO UPDATE
                    SET updated_at = NOW(), stats = EXCLUDED.stats
                """, _json.dumps(snapshot, default=str))
        except Exception as e:
            # Pure observability — never block trading
            logger.debug(f"_persist_runtime_stats failed (non-fatal): {e}")

    async def _compute_funding_recommendation(self) -> Dict:
        """Compute open notional + recommended wallet funding (issue 12).

        open_notional_usd       = SUM(entry_usd) over open sniper_trades rows.
        recommended_funding_usd = open_notional_usd
                                  + headroom to fill the remaining cap slots
                                    at the average open-position entry size
                                  + a flat per-slot gas buffer.

        Gas buffer is per *cap slot* (max_active_positions), not per current
        open count, because the operator must fund for the worst case where
        the cap fully fills. Solana priority-fee + base-fee is tiny (~$0.01),
        EVM snipe gas is larger; we use a conservative SNIPER_GAS_BUFFER_USD
        per slot (default $0.50, DB-overridable). Fail-soft to zeros.
        """
        result = {
            'open_notional_usd': 0.0,
            'avg_entry_usd': 0.0,
            'recommended_funding_usd': 0.0,
            'gas_buffer_usd_per_slot': float(getattr(self, 'gas_buffer_usd', 0.50)),
        }
        if not self.db_pool:
            return result
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) AS n, COALESCE(SUM(entry_usd), 0) AS total, "
                    "COALESCE(AVG(entry_usd), 0) AS avg FROM sniper_trades "
                    "WHERE status = 'open'"
                )
            open_count = int(row['n'] or 0)
            open_notional = float(row['total'] or 0)
            avg_entry = float(row['avg'] or 0)
            # If no open rows yet, size headroom off the configured trade size.
            if avg_entry <= 0:
                avg_entry = float(self.trade_amount)
            cap = int(self.max_active_positions)
            remaining_slots = max(0, cap - open_count)
            gas_per_slot = float(getattr(self, 'gas_buffer_usd', 0.50))
            recommended = (
                open_notional
                + remaining_slots * avg_entry
                + cap * gas_per_slot
            )
            result['open_notional_usd'] = round(open_notional, 2)
            result['avg_entry_usd'] = round(avg_entry, 4)
            result['recommended_funding_usd'] = round(recommended, 2)
        except Exception as e:
            logger.debug(f"_compute_funding_recommendation DB error: {e}")
        return result

    async def _process_targets(self):
        """Execute buy orders for pending targets"""
        while self.is_running:
            try:
                if not self.pending_targets:
                    await asyncio.sleep(0.01)
                    continue

                # Process active targets
                for address, data in list(self.pending_targets.items()):
                    if data['status'] == 'pending':
                        # EXECUTE BUY
                        await self._execute_snipe(data)

                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing targets: {e}")
                await asyncio.sleep(1)

    async def _execute_snipe(self, data: Dict):
        """Execute the buy transaction with high priority"""
        token_address = data['target'].get('token_address')
        chain = data['chain_type']
        timing = data.get('target', {}).get('_timing')
        if timing:
            timing.stamp('t_broadcast_start')

        # Belt-and-suspenders cap check: pending_targets can fill up between
        # the _evaluate_target gate and broadcast. Skip silently (no DB log,
        # no timing emit) — _evaluate_target already accounted the rejection.
        if await self._effective_active_count() >= self.max_active_positions:
            data['status'] = 'failed'
            data['error'] = 'capped'
            if timing:
                timing.stamp('t_broadcast_done')
                timing.outcome = 'rejected_capped'
                try:
                    timing.emit()
                except Exception:
                    pass
            self.pending_targets.pop(token_address, None)
            return

        # Per-trade hot path (63k+ trades in a DRY_RUN window). DEBUG to keep
        # logs/sniper/ small; failures below are still ERROR.
        logger.debug(f"🔫 EXECUTING SNIPE: {token_address} on {chain}")
        data['status'] = 'buying'

        try:
            if not self.executor:
                logger.error("Trade executor not initialized")
                data['status'] = 'failed'
                if timing:
                    timing.stamp('t_broadcast_done')
                    timing.outcome = 'failed'
                return

            # Entry size. LIVE always uses the exact configured trade_amount.
            # DRY_RUN jitters ±20% (seeded by token so it's reproducible) so
            # collected data shows realistic size variation instead of an
            # identical notional on every simulated trade.
            entry_amount = self.trade_amount
            if self.dry_run and token_address:
                seed = int(hashlib.sha256(("size:" + token_address).encode()).hexdigest()[:16], 16)
                entry_amount = self.trade_amount * (0.8 + (random.Random(seed).random() * 0.4))

            # Wave-15 risk-gate fix: sniper-specific capital-protection check.
            #
            # When sniper_use_dex_risk_manager=False (default, seeded by migration 047):
            #   Skip the DEX-oriented liquidity/honeypot/token-risk analysis inside
            #   validate_trade — which always rejects new-launch targets with <$10k
            #   liquidity. The sniper's _check_filters + TokenSafetyChecker (already
            #   run above in _evaluate_target) are the correct memecoin risk layer.
            #   Only the capital-protection half is preserved: circuit breakers (loss
            #   rate, drawdown, consecutive losses) and the allocation guard (per-module
            #   USD budget cap). Both are read from the risk_manager instance directly.
            #
            # When sniper_use_dex_risk_manager=True:
            #   Full validate_trade is called, restoring Wave-13 behavior.
            #
            # Skipped entirely when risk_manager is None (tests or failed construction).
            if self.risk_manager is not None:
                try:
                    if self.sniper_use_dex_risk_manager:
                        # Full DEX-oriented gate (Wave-13 behavior).
                        allowed, reason = await self.risk_manager.validate_trade(
                            token_address, entry_amount
                        )
                    else:
                        # Sniper-safe capital-protection gate only:
                        # 1. Circuit breakers (loss rate / drawdown / consecutive losses).
                        metrics = await self.risk_manager._get_current_metrics()
                        cb_ok, cb_reason = self.risk_manager.check_circuit_breakers(metrics)
                        if not cb_ok:
                            allowed, reason = False, f"Circuit breaker: {cb_reason}"
                        else:
                            # 2. Allocation guard (per-module USD budget cap).
                            allowed, reason = True, "Capital checks passed"
                            guard = getattr(self.risk_manager, '_allocation_guard', None)
                            if guard is not None:
                                try:
                                    guard_module = getattr(
                                        self.risk_manager, '_module_name', 'sniper'
                                    )
                                    g_ok, g_reason = await guard.check(
                                        guard_module, float(entry_amount)
                                    )
                                    if not g_ok:
                                        allowed, reason = False, f"Allocation guard: {g_reason}"
                                except Exception as guard_exc:
                                    logger.debug(
                                        "allocation guard error (fail-soft): %s", guard_exc
                                    )

                    if not allowed:
                        logger.warning(
                            f"⛔ Sniper entry blocked by RiskManager: "
                            f"{token_address} amount={entry_amount:.4f} reason={reason}"
                        )
                        data['status'] = 'failed'
                        data['error'] = f'risk_manager:{reason}'
                        if timing:
                            timing.stamp('t_broadcast_done')
                            timing.outcome = 'rejected_risk_manager'
                            try:
                                timing.emit()
                            except Exception:
                                pass
                        self.pending_targets.pop(token_address, None)
                        return
                except Exception as rm_exc:
                    logger.warning(
                        f"RiskManager gate raised: {rm_exc}; continuing without gate"
                    )

            # Execute buy using the trade executor
            result = await self.executor.execute_buy(
                token_address=token_address,
                chain=chain,
                amount_in=entry_amount,
                slippage=self.slippage,
                priority_fee=self.priority_fee
            )

            if result.success:
                if timing:
                    timing.stamp('t_broadcast_done')
                    timing.outcome = 'success'
                logger.debug(f"✅ SNIPE SUCCESS: {token_address}")
                logger.debug(f"   TX: {result.tx_hash} | Amount: {result.amount_out}")

                # Determine native token and get real USD price. Use the
                # actual amount spent (result.amount_in) — in DRY_RUN this is
                # the jittered entry size, so entry_usd / entry_price vary
                # per trade instead of being a fixed constant.
                native_token = 'sol' if chain == 'solana' else 'eth'
                native_price = await self.price_fetcher.get_price(native_token)
                entry_usd = result.amount_in * native_price

                data['status'] = 'active'
                data['entry_price'] = result.amount_in / result.amount_out if result.amount_out > 0 else 0
                data['amount_bought'] = result.amount_out
                data['tx_hash'] = result.tx_hash
                data['entry_time'] = result.timestamp
                data['entry_usd'] = entry_usd  # Store USD value for accurate exit PnL
                data['native_price_at_entry'] = native_price

                logger.debug(f"   Entry value: ${entry_usd:.2f} ({result.amount_in:.4f} {native_token.upper()} @ ${native_price:.2f})")

                self.active_snipes[token_address] = data
                del self.pending_targets[token_address]

                # Log to database
                await self._log_snipe_to_db(data, result)
            else:
                if timing:
                    timing.stamp('t_broadcast_done')
                    timing.outcome = 'failed'
                logger.error(f"❌ SNIPE FAILED: {result.error}")
                data['status'] = 'failed'
                data['error'] = result.error

        except Exception as e:
            if timing:
                timing.stamp('t_broadcast_done')
                timing.outcome = 'failed'
            logger.error(f"❌ SNIPE FAILED: {e}")
            data['status'] = 'failed'
            data['error'] = str(e)
        finally:
            # Always emit one timing line per snipe attempt. Wrapped
            # fail-soft so an instrumentation bug never breaks trading.
            if timing:
                try:
                    timing.emit()
                except Exception:
                    pass

    async def _log_snipe_to_db(self, data: Dict, result):
        """Log snipe trade to dedicated sniper_trades table with real USD values"""
        try:
            if self.db_pool:
                import uuid

                # Determine native token based on chain
                chain_type = data.get('chain_type', 'solana')
                native_token = 'sol' if chain_type == 'solana' else 'eth'

                # Fetch real native token price for USD conversion
                native_price = await self.price_fetcher.get_price(native_token)
                trade_amount_native = result.amount_in  # Amount spent in native token
                usd_value = trade_amount_native * native_price

                # Get safety report if available
                safety_report = data.get('target', {}).get('safety_report', {})
                trade_id = f"snipe_{uuid.uuid4().hex[:12]}"

                async with self.db_pool.acquire() as conn:
                    # Insert into dedicated sniper_trades table
                    await conn.execute("""
                        INSERT INTO sniper_trades (
                            trade_id, token_address, chain, side, entry_price, amount,
                            entry_usd, native_token, native_price_at_entry, trade_amount_native,
                            safety_score, safety_rating, is_honeypot, buy_tax, sell_tax, liquidity_usd,
                            status, is_simulated, entry_timestamp, entry_tx_hash, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    """,
                        trade_id,
                        data['target'].get('token_address', 'UNKNOWN'),
                        chain_type,
                        'buy',
                        data.get('entry_price', 0),
                        result.amount_out,
                        usd_value,
                        native_token.upper(),
                        native_price,
                        trade_amount_native,
                        safety_report.get('score'),
                        safety_report.get('rating'),
                        safety_report.get('is_honeypot', False),
                        safety_report.get('buy_tax'),
                        safety_report.get('sell_tax'),
                        safety_report.get('liquidity_usd'),
                        'open' if result.success else 'failed',
                        should_skip_live(self.dry_run, module='sniper'),
                        result.timestamp,
                        result.tx_hash,
                        json.dumps({
                            'warnings': safety_report.get('warnings', []),
                            'detection_path': (
                                # EVM listener stamps at target top level;
                                # Solana listener nests under target['metadata'].
                                # Read either location so both paths show up
                                # in /api/sniper/timing groupings.
                                data.get('target', {}).get('detection_path')
                                or data.get('target', {}).get('metadata', {}).get('detection_path')
                            ),
                            # Propagate block-time anchoring flag so the DB
                            # is SQL-filterable. Listener already stamps it
                            # into target['metadata'] on WSS detection.
                            'block_time_anchored': bool(
                                data.get('target', {}).get('metadata', {}).get('block_time_anchored')
                                or data.get('target', {}).get('block_time_anchored')
                            ),
                            'timing': (data.get('target', {}).get('_timing').to_metadata_dict()
                                       if data.get('target', {}).get('_timing') is not None
                                       else None),
                        })
                    )
                # Store trade_id in data for linking exit trade
                data['db_trade_id'] = trade_id
                logger.debug(f"💾 Logged to sniper_trades: {trade_id} ${usd_value:.2f}")
        except Exception as e:
            logger.error(f"Error logging snipe to DB: {e}")

    async def _monitor_active_snipes(self):
        """Monitor active snipes for auto-sell targets (take profit / stop loss)"""
        # Use DB-configured thresholds. stop_loss_pct is stored as a positive
        # "loss threshold"; negate at use because pnl_pct is signed.
        take_profit_pct = self.take_profit_pct
        stop_loss_pct = -abs(self.stop_loss_pct)

        check_count = 0
        while self.is_running:
            check_count += 1

            for address, data in list(self.active_snipes.items()):
                try:
                    if data['status'] != 'active':
                        continue

                    chain = data.get('chain_type', 'solana')
                    entry_price = data.get('entry_price', 0)
                    amount_held = data.get('amount_bought', 0)

                    if entry_price <= 0 or amount_held <= 0:
                        continue

                    # entry_price is stored as native-per-token (SOL/ETH per
                    # token) but _get_token_price returns USD per token.
                    # Comparing them directly produces absurd PnL%
                    # (the "+594089%" lines in Wave-12 logs). Derive a
                    # USD-denominated entry price from entry_usd / amount_held
                    # so both legs are in the same unit.
                    entry_usd_total = data.get('entry_usd', 0) or 0
                    entry_price_usd = (
                        entry_usd_total / amount_held
                        if entry_usd_total > 0 and amount_held > 0
                        else 0
                    )

                    # Get current price (USD per token)
                    current_price = await self._get_token_price(address, chain)

                    if current_price is None or current_price <= 0:
                        # In DRY_RUN, new Pump.fun / freshly-launched mints have no
                        # Jupiter/Birdeye/CoinGecko price yet, so the monitor loop
                        # can never decide TP/SL and positions accumulate forever.
                        # Force-retire so Phase 2 data accumulates. Live path is
                        # unchanged: production should fix the price-fetch root cause
                        # separately (Jupiter route quote or pool-derived price).
                        dry = self.dry_run or os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')
                        if dry:
                            await self._close_position_synthetic(
                                data, reason='dry_run_no_price_feed'
                            )
                        continue

                    # Calculate P&L using USD-denominated prices (same unit on
                    # both legs). Falls back to native-unit comparison only when
                    # entry_usd is missing (shouldn't happen post-fix).
                    if entry_price_usd > 0:
                        pnl_pct = ((current_price - entry_price_usd) / entry_price_usd) * 100
                    else:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    # Phantom-price guard: |pnl_pct| > 200 means the price
                    # source returned a stale or wrong-unit value. In DRY_RUN,
                    # route through the modeled synthetic close instead of
                    # triggering a false TP/SL. In LIVE, log and skip — do NOT
                    # act on bad data.
                    _PHANTOM_THRESHOLD = 200.0
                    if abs(pnl_pct) > _PHANTOM_THRESHOLD:
                        dry = self.dry_run or os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')
                        if dry:
                            logger.warning(
                                f"phantom-price {address[:8]}... "
                                f"entry_usd={entry_price_usd:.8f} "
                                f"current={current_price:.8f} "
                                f"pnl={pnl_pct:+.2f}%; synthetic close"
                            )
                            await self._close_position_synthetic(
                                data, reason='phantom_price_dry_run'
                            )
                        else:
                            logger.warning(
                                f"LIVE phantom-price skipped {address[:8]}... "
                                f"pnl={pnl_pct:+.2f}% > +/-{_PHANTOM_THRESHOLD}% "
                                f"— price source may be wrong unit or stale"
                            )
                        continue

                    # Log status periodically
                    if check_count % 60 == 0:  # Every minute
                        logger.info(
                            f"Position: {address[:8]}... | "
                            f"Entry: ${entry_price_usd:.8f} | "
                            f"Current: ${current_price:.8f} | P&L: {pnl_pct:+.2f}%"
                        )

                    # Wave-15: update high-watermark for trailing stop
                    ref_price = entry_price_usd if entry_price_usd > 0 else entry_price
                    hw = data.get('high_watermark_price', ref_price)
                    if current_price > hw:
                        data['high_watermark_price'] = current_price
                        hw = current_price

                    already_partial = data.get('partial_taken', False)

                    # Wave-15: partial take — fire when enabled and not yet fired
                    partial_take_pct = self.sniper_partial_take_pct
                    if (
                        partial_take_pct > 0
                        and not already_partial
                        and pnl_pct >= partial_take_pct
                    ):
                        partial_size_pct = self.sniper_partial_take_size_pct
                        amount_held = data.get('amount_bought', 0)
                        partial_amount = amount_held * (partial_size_pct / 100.0)
                        remaining_amount = amount_held - partial_amount
                        logger.info(
                            f"PARTIAL TAKE for {address[:8]}... "
                            f"(+{pnl_pct:.1f}% >= +{partial_take_pct:.1f}%) "
                            f"selling {partial_size_pct:.0f}% = {partial_amount:.4f} tokens"
                        )
                        await self._partial_exit(data, partial_amount, 'PARTIAL_TAKE')
                        data['amount_bought'] = remaining_amount
                        data['partial_taken'] = True
                        data['partial_take_price'] = current_price
                        data['high_watermark_price'] = current_price
                        self._stats['partial_takes_fired'] = (
                            self._stats.get('partial_takes_fired', 0) + 1
                        )

                    # Check take profit (remaining position)
                    elif pnl_pct >= take_profit_pct:
                        logger.info(f"TAKE PROFIT triggered for {address[:8]}... ({pnl_pct:+.2f}%)")
                        await self._exit_position(data, 'TAKE_PROFIT')

                    # Stop loss: trailing after partial take, or fixed otherwise
                    elif already_partial and self.sniper_trail_after_partial:
                        stop_loss_pct_raw = abs(self.stop_loss_pct)
                        trail_stop_price = hw * (1 - stop_loss_pct_raw / 100.0)
                        if current_price <= trail_stop_price:
                            logger.warning(
                                f"TRAIL STOP for {address[:8]}... "
                                f"(price {current_price:.8f} <= trail {trail_stop_price:.8f})"
                            )
                            await self._exit_position(data, 'TRAIL_STOP')
                    elif pnl_pct <= stop_loss_pct:
                        logger.warning(f"STOP LOSS triggered for {address[:8]}... ({pnl_pct:.2f}%)")
                        await self._exit_position(data, 'STOP_LOSS')

                    # SNIPE-RM-12: time-stop. Some snipes neither hit
                    # TP nor SL but just sit at -10% for hours, tying
                    # up the position-cap slot. If max_hold_minutes is
                    # configured, exit at market regardless of P&L.
                    elif self.max_hold_minutes > 0:
                        entry_time = data.get('entry_time')
                        if entry_time is not None:
                            try:
                                hold_minutes = (datetime.now() - entry_time).total_seconds() / 60.0
                                if hold_minutes >= self.max_hold_minutes:
                                    logger.info(
                                        f"TIME STOP triggered for {address[:8]}... "
                                        f"(held {hold_minutes:.0f}m, cap {self.max_hold_minutes}m, P&L {pnl_pct:+.2f}%)"
                                    )
                                    await self._exit_position(data, 'TIME_STOP')
                            except (TypeError, AttributeError):
                                # entry_time wasn't a datetime — skip
                                # the check rather than crash the loop.
                                pass

                except Exception as e:
                    logger.error(f"Error monitoring snipe {address}: {e}")

            await asyncio.sleep(1)

    def _model_dry_run_exit_pct(self, token_address: str) -> float:
        """Model a realistic snipe exit %P&L when no live price feed exists.

        DRY_RUN snipes new Pump.fun / freshly-launched mints that have no
        Jupiter/Birdeye/Pyth price yet, so the monitor loop can never compute
        a real exit. Previously the position was retired flat (exit==entry,
        P&L=0), which made EVERY DRY_RUN trade look identical and inflated the
        win rate to a constant. That is fabricated data, not collected data.

        Instead, model a distribution that matches observed memecoin-snipe
        reality: most fresh mints bleed out or rug, a minority pump. The draw
        is seeded by token_address so a given mint always exits the same way
        (reproducible across re-runs / restarts) while the population spans a
        believable spread. TP/SL caps from DB config bound the outcome so the
        modeled exit respects the same take_profit_pct / stop_loss_pct the
        live path enforces.

        IMPORTANT: this is a MODEL, not a measured price. It is only used in
        DRY_RUN and only for mints with no obtainable price. The live path
        never calls this. Documented in modules/sniper/CLAUDE.md.
        """
        seed = int(hashlib.sha256(token_address.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        roll = rng.random()
        # Outcome buckets (memecoin snipe realities):
        #   ~55% bleed/rug to a loss, ~30% small chop, ~15% pump.
        if roll < 0.55:
            # Loss: clustered toward the stop-loss floor.
            pct = -rng.uniform(abs(self.stop_loss_pct) * 0.4, abs(self.stop_loss_pct))
        elif roll < 0.85:
            # Chop: small move either side, well inside TP/SL.
            pct = rng.uniform(-abs(self.stop_loss_pct) * 0.3, self.take_profit_pct * 0.3)
        else:
            # Winner: up toward (and sometimes capped at) take-profit.
            pct = rng.uniform(self.take_profit_pct * 0.5, self.take_profit_pct)
        # Bound to live-path exit caps so modeled P&L never exceeds what the
        # real TP/SL gates would have realized.
        return max(-abs(self.stop_loss_pct), min(self.take_profit_pct, pct))

    async def _close_position_synthetic(self, data: Dict, reason: str) -> None:
        """Retire a DRY_RUN position with a MODELED exit when no price feed
        is available.

        Pump.fun / freshly-launched mints have no Jupiter/Birdeye/Pyth price
        yet, so the monitor loop cannot compute a real exit. Rather than retire
        flat (which fabricated a constant P&L and an unrealistic win rate), we
        draw a realistic exit %P&L from `_model_dry_run_exit_pct`, derive the
        exit price/USD from it, and persist that. Exit price now differs from
        entry, P&L varies, and the win rate reflects the modeled distribution.
        Only used in DRY_RUN; the live path computes real P&L in _exit_position.
        """
        try:
            target = data.get('target') or {}
            token_address = target.get('token_address') if isinstance(target, dict) else None
            if not token_address:
                return

            entry_price = data.get('entry_price', 0) or 0
            entry_usd = data.get('entry_usd', 0) or 0
            trade_id = data.get('db_trade_id')
            now = datetime.now()

            # Model a realistic exit instead of a flat zero.
            pnl_pct = self._model_dry_run_exit_pct(token_address)
            exit_price = entry_price * (1 + pnl_pct / 100.0) if entry_price else 0
            exit_usd = entry_usd * (1 + pnl_pct / 100.0) if entry_usd else 0
            pnl_usd = exit_usd - entry_usd

            # Mark in-memory before DB write so subsequent ticks skip it.
            data['status'] = 'closed'
            data['exit_price'] = exit_price
            data['exit_reason'] = reason
            data['exit_time'] = now

            if trade_id and self.db_pool:
                try:
                    async with self.db_pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE sniper_trades SET
                                status = 'closed',
                                exit_price = $1,
                                exit_usd = $2,
                                exit_timestamp = $3,
                                profit_loss = $4,
                                profit_loss_pct = $5,
                                exit_reason = $6
                            WHERE trade_id = $7
                            """,
                            exit_price, exit_usd, now, pnl_usd, pnl_pct, reason, trade_id,
                        )
                except Exception as e:
                    logger.debug(f"synthetic close db update failed: {e}")

            # Retire from active_snipes so the monitor loop stops iterating it.
            self.active_snipes.pop(token_address, None)

            self._stats['positions_synthetic_closed'] = (
                self._stats.get('positions_synthetic_closed', 0) + 1
            )
            logger.debug(
                f"🧹 Modeled-exit {token_address[:8]}... reason={reason} "
                f"P&L {pnl_pct:+.1f}% (${pnl_usd:+.2f}; DRY_RUN, no live price)"
            )
        except Exception as e:
            logger.error(f"_close_position_synthetic error: {e}")

    async def _get_token_price(self, token_address: str, chain: str) -> float:
        """Get current token price (simplified).

        Wave-3: Solana resolution order is now
            Pyth Hermes (if mint has a feed-id) → Jupiter Price v2 →
            Jupiter /quote → Birdeye /defi/price

        Pyth is preferred for blue-chips because it is independent of
        Jupiter/Birdeye and sub-second fresh, so a Jupiter brown-out
        cannot synthetically-close blue-chip positions. Pump.fun mints
        have no Pyth feed-id; get_pyth_feed_id returns None and the
        chain falls through unaffected.

        The Pyth lookup is gated by `self.sniper_pyth_feeds_enabled`
        (default True; DB-overridable via config_settings).
        """
        try:
            cached = self._mint_price_cache.get(token_address)
            if cached:
                price, ts = cached
                if datetime.now() - ts < self._mint_price_ttl:
                    return price

            import aiohttp

            if chain == 'solana':
                # 0) Pyth Hermes — blue-chip mints only. Free + independent
                # of Jupiter/Birdeye, so this layer breaks the
                # "Jupiter brown-out cascades through every blue-chip SL/TP"
                # failure mode. No-op for Pump.fun (no feed-id).
                if getattr(self, 'sniper_pyth_feeds_enabled', True):
                    pyth_price = await self._get_token_price_via_pyth(token_address)
                    if pyth_price > 0:
                        self._mint_price_cache[token_address] = (pyth_price, datetime.now())
                        return pyth_price

                # 1) Jupiter Price API v2 — fast when indexed
                url = f"https://api.jup.ag/price/v2?ids={token_address}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            price_info = data.get('data', {}).get(token_address)
                            if price_info:
                                price = float(price_info.get('price') or 0)
                                if price > 0:
                                    self._mint_price_cache[token_address] = (price, datetime.now())
                                    return price

                # 2) Jupiter /quote fallback — works for fresh mints that
                # Price v2 has not yet indexed but already have a pool.
                quote_price = await self._get_token_price_via_jupiter_quote(token_address)
                if quote_price > 0:
                    self._mint_price_cache[token_address] = (quote_price, datetime.now())
                    return quote_price

                # 3) Birdeye fallback — independent of Jupiter so a
                # Jupiter brown-out doesn't synthetically-close every
                # active position simultaneously.
                bird_price = await self._get_token_price_via_birdeye(token_address)
                if bird_price > 0:
                    self._mint_price_cache[token_address] = (bird_price, datetime.now())
                    return bird_price
            else:
                # For EVM, use DexScreener or similar
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            pairs = data.get('pairs', [])
                            if pairs:
                                price = float(pairs[0].get('priceNative', 0))
                                if price > 0:
                                    self._mint_price_cache[token_address] = (price, datetime.now())
                                    return price

        except Exception as e:
            logger.debug(f"Error fetching price for {token_address}: {e}")

        return 0

    async def _get_token_price_via_pyth(self, token_address: str) -> float:
        """Wave-3: Pyth Hermes blue-chip price lookup.

        Returns 0 for any mint without a mapped feed-id (Pump.fun and
        the long tail of new memecoins). Returns 0 for any HTTP or
        parse failure — caller falls through to Jupiter.

        Uses the module-level `pyth_client` singleton so cache hits +
        rate-limit state are shared across all positions.
        """
        try:
            from modules.sniper.core.pyth_feed import pyth_client
            from modules.sniper.core.pyth_feed_ids import get_pyth_feed_id

            feed_id = get_pyth_feed_id(token_address)
            if not feed_id:
                return 0
            price = await pyth_client.get_price(feed_id)
            if price and price > 0:
                self._stats['pyth_fallback_hits'] = (
                    self._stats.get('pyth_fallback_hits', 0) + 1
                )
                return float(price)
        except Exception as e:
            logger.debug(f"Pyth feed lookup error for {token_address}: {e}")
        return 0

    async def _get_token_price_via_jupiter_quote(self, token_address: str) -> float:
        """Derive USD price per whole token from a Jupiter quote.

        Sends a small SOL → token quote and converts outAmount into a
        USD price. Trade executor normalizes amount_out at /1e6 (i.e.
        assumes 6 decimals) so we match that convention here for unit
        consistency with entry_price. Returns 0 on failure.
        """
        try:
            import aiohttp
            SOL_MINT = 'So11111111111111111111111111111111111111112'
            in_lamports = 10_000_000  # 0.01 SOL probe
            url = (
                'https://lite-api.jup.ag/swap/v1/quote'
                f'?inputMint={SOL_MINT}&outputMint={token_address}'
                f'&amount={in_lamports}&slippageBps=500&onlyDirectRoutes=false'
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status != 200:
                        return 0
                    data = await response.json()
                    out_amount_raw = int(data.get('outAmount') or 0)
                    if out_amount_raw <= 0:
                        return 0

            sol_usd = await self.price_fetcher.get_price('sol')
            if sol_usd <= 0:
                return 0

            # Match trade_executor's 6-decimal convention: tokens_received
            # is out_amount_raw / 1e6.
            in_sol = in_lamports / 1e9
            tokens_received = out_amount_raw / 1e6
            if tokens_received <= 0:
                return 0
            price_usd = (in_sol * sol_usd) / tokens_received
            self._stats['jupiter_quote_fallback_hits'] = (
                self._stats.get('jupiter_quote_fallback_hits', 0) + 1
            )
            return float(price_usd)
        except Exception as e:
            logger.debug(f"Jupiter quote fallback error for {token_address}: {e}")
            return 0

    async def _get_token_price_via_birdeye(self, token_address: str) -> float:
        """R5: Birdeye /defi/price tertiary fallback for Solana mints.

        Independent of Jupiter. Free tier requires no API key for the
        public price endpoint but applies a soft ~1 req/s rate limit
        — bounded by the 15s per-mint cache in _get_token_price.

        Returns price in USD per whole token, or 0 on any failure.
        """
        try:
            import aiohttp
            url = f"https://public-api.birdeye.so/defi/price?address={token_address}"
            headers = {'X-Chain': 'solana', 'accept': 'application/json'}
            try:
                from security.secrets_manager import secrets
                api_key = secrets.get('BIRDEYE_API_KEY', default=None, log_access=False)
            except Exception:
                api_key = None
            if api_key:
                headers['X-API-KEY'] = api_key
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status != 200:
                        return 0
                    data = await response.json()
                    # Birdeye response shape: {success: bool, data: {value: float}}
                    inner = (data.get('data') or {}) if isinstance(data, dict) else {}
                    price = float(inner.get('value') or 0)
                    if price > 0:
                        self._stats['birdeye_fallback_hits'] = (
                            self._stats.get('birdeye_fallback_hits', 0) + 1
                        )
                        return price
        except Exception as e:
            logger.debug(f"Birdeye fallback error for {token_address}: {e}")
        return 0

    async def _partial_exit(self, data: Dict, partial_amount: float, reason: str):
        """Wave-15: Sell a fraction of the position (partial take).

        Executes execute_sell for partial_amount tokens, logs the partial exit
        to sniper_trades as a supplementary 'sell' row (does NOT close the parent
        open row — the parent remains open until the remaining position exits).
        Fail-soft: any error is logged but does not crash the monitor loop.
        """
        token_address = data['target'].get('token_address')
        chain = data.get('chain_type', 'solana')
        try:
            if not self.executor or partial_amount <= 0:
                return
            result = await self.executor.execute_sell(
                token_address=token_address,
                chain=chain,
                amount_in=partial_amount,
                slippage=self.slippage,
                priority_fee=self.priority_fee
            )
            if result.success:
                native_token = 'sol' if chain == 'solana' else 'eth'
                native_price = await self.price_fetcher.get_price(native_token)
                exit_usd = result.amount_out * native_price
                full_amount = data.get('amount_bought', partial_amount) + partial_amount
                fraction = partial_amount / full_amount if full_amount > 0 else 0
                entry_usd_partial = data.get('entry_usd', 0) * fraction
                pnl_usd = exit_usd - entry_usd_partial
                pnl_pct = (pnl_usd / entry_usd_partial * 100) if entry_usd_partial > 0 else 0
                logger.info(
                    f"PARTIAL EXIT ok: {token_address[:8]}... "
                    f"{partial_amount:.4f} tokens -> ${exit_usd:.2f} "
                    f"(P&L ${pnl_usd:+.2f} / {pnl_pct:+.1f}%)"
                )
                if self.db_pool:
                    try:
                        import uuid as _uuid
                        trade_id = f"snipe_pt_{_uuid.uuid4().hex[:12]}"
                        parent_id = data.get('db_trade_id')
                        async with self.db_pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO sniper_trades (
                                    trade_id, token_address, chain, side,
                                    entry_price, exit_price, amount,
                                    entry_usd, exit_usd,
                                    profit_loss, profit_loss_pct,
                                    status, exit_reason, is_simulated,
                                    entry_timestamp, exit_timestamp,
                                    exit_tx_hash, metadata
                                ) VALUES (
                                    $1,$2,$3,'sell',
                                    $4,$5,$6,
                                    $7,$8,
                                    $9,$10,
                                    'closed',$11,$12,
                                    $13,$14,
                                    $15,$16
                                )
                            """,
                                trade_id, token_address, chain,
                                data.get('entry_price', 0),
                                (result.amount_out / partial_amount
                                 if partial_amount else 0),
                                partial_amount,
                                entry_usd_partial, exit_usd,
                                pnl_usd, pnl_pct,
                                reason,
                                should_skip_live(self.dry_run, module='sniper'),
                                data.get('entry_time') or result.timestamp,
                                result.timestamp,
                                result.tx_hash,
                                json.dumps({'partial_of': parent_id, 'reason': reason})
                            )
                    except Exception as db_e:
                        logger.debug(f"partial exit DB log failed (non-fatal): {db_e}")
            else:
                logger.warning(
                    f"PARTIAL EXIT failed for {token_address}: {result.error}"
                )
        except Exception as e:
            logger.error(f"_partial_exit error for {token_address}: {e}")

    async def _exit_position(self, data: Dict, reason: str):
        """Exit a position (sell tokens)"""
        token_address = data['target'].get('token_address')
        chain = data.get('chain_type', 'solana')
        amount = data.get('amount_bought', 0)

        logger.debug(f"💰 Exiting position: {token_address} | Reason: {reason}")

        try:
            if not self.executor or amount <= 0:
                logger.error("Cannot exit: executor not ready or no tokens")
                return

            result = await self.executor.execute_sell(
                token_address=token_address,
                chain=chain,
                amount_in=amount,
                slippage=self.slippage,
                priority_fee=self.priority_fee
            )

            if result.success:
                logger.debug(f"✅ EXIT SUCCESS: {token_address}")
                logger.debug(f"   TX: {result.tx_hash} | Received: {result.amount_out}")

                data['status'] = 'closed'
                data['exit_price'] = result.amount_out / amount if amount > 0 else 0
                data['exit_reason'] = reason
                data['exit_tx'] = result.tx_hash
                data['exit_time'] = result.timestamp

                # Remove from active snipes
                if token_address in self.active_snipes:
                    del self.active_snipes[token_address]

                # Log exit to database
                await self._log_exit_to_db(data, result, reason)
            else:
                logger.error(f"❌ EXIT FAILED: {result.error}")

        except Exception as e:
            logger.error(f"Error exiting position: {e}")

    async def _log_exit_to_db(self, data: Dict, result, reason: str):
        """Update sniper_trades record with exit data"""
        try:
            if self.db_pool:
                chain_type = data.get('chain_type', 'solana')
                native_token = 'sol' if chain_type == 'solana' else 'eth'

                # Fetch real native token price for USD conversion
                native_price = await self.price_fetcher.get_price(native_token)

                # Exit price per token (in native token)
                exit_price = data.get('exit_price', 0)

                # Amount of native token received
                native_received = result.amount_out

                # Calculate USD values
                entry_usd = data.get('entry_usd', self.trade_amount * native_price)
                exit_usd = native_received * native_price

                # Calculate real PnL in USD
                pnl_usd = exit_usd - entry_usd
                pnl_pct = ((exit_usd - entry_usd) / entry_usd * 100) if entry_usd > 0 else 0

                trade_id = data.get('db_trade_id')

                async with self.db_pool.acquire() as conn:
                    if trade_id:
                        # UPDATE the existing sniper_trades record
                        await conn.execute("""
                            UPDATE sniper_trades SET
                                exit_price = $1,
                                exit_usd = $2,
                                profit_loss = $3,
                                profit_loss_pct = $4,
                                native_price_at_exit = $5,
                                status = $6,
                                exit_reason = $7,
                                exit_timestamp = $8,
                                exit_tx_hash = $9
                            WHERE trade_id = $10
                        """,
                            exit_price,
                            exit_usd,
                            pnl_usd,
                            pnl_pct,
                            native_price,
                            'closed' if result.success else 'failed',
                            reason,
                            result.timestamp,
                            result.tx_hash,
                            trade_id
                        )
                        logger.debug(f"💾 Updated sniper_trades: {trade_id} exit ${exit_usd:.2f} (PnL: ${pnl_usd:.2f} / {pnl_pct:.1f}%)")
                    else:
                        # Fallback: insert new record if no trade_id (shouldn't happen normally)
                        import uuid
                        token_address = data['target'].get('token_address', 'UNKNOWN')
                        await conn.execute("""
                            INSERT INTO sniper_trades (
                                trade_id, token_address, chain, side, exit_price,
                                exit_usd, profit_loss, profit_loss_pct, native_price_at_exit,
                                status, exit_reason, is_simulated, exit_timestamp, exit_tx_hash
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        """,
                            f"snipe_exit_{uuid.uuid4().hex[:12]}",
                            token_address,
                            chain_type,
                            'sell',
                            exit_price,
                            exit_usd,
                            pnl_usd,
                            pnl_pct,
                            native_price,
                            'closed' if result.success else 'failed',
                            reason,
                            should_skip_live(self.dry_run, module='sniper'),
                            result.timestamp,
                            result.tx_hash
                        )
                        logger.debug(f"💾 Logged exit to sniper_trades: ${exit_usd:.2f} (PnL: ${pnl_usd:.2f})")
        except Exception as e:
            logger.error(f"Error logging exit to DB: {e}")

    async def stop(self):
        """Stop the engine"""
        self.is_running = False
        for task in self.tasks:
            task.cancel()
        if self.token_safety:
            await self.token_safety.close()
        if self.executor:
            await self.executor.close()
        logger.info("🛑 Sniper Engine Stopped")
