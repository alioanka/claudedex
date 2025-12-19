"""
Sniper Engine - High-speed new token sniping
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
import os
import aiohttp

from config.config_manager import ConfigManager
from data.storage.database import DatabaseManager
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

        # Settings (loaded from DB)
        self.dry_run = True
        self.trade_amount = 0.1
        self.slippage = 10.0
        self.priority_fee = 5000
        self.max_buy_tax = 15.0
        self.max_sell_tax = 15.0
        self.min_liquidity = 1000.0
        self.safety_check_enabled = True
        self.test_mode = False  # Relaxed safety for testing

        # Statistics tracking for rate-limited logging
        self._stats = {
            'tokens_analyzed': 0,
            'honeypots_detected': 0,
            'danger_ratings': 0,
            'high_tax_rejected': 0,
            'low_liquidity_rejected': 0,
            'passed_safety': 0,
            'last_stats_log': datetime.now()
        }

        # Cooldown tracking for rejected/analyzed tokens
        self._rejected_cache: Dict[str, datetime] = {}  # Track recently rejected tokens
        self._cooldown_duration = timedelta(minutes=5)  # 5-minute cooldown per token

        # Price fetcher for real USD values
        self.price_fetcher = PriceFetcher()

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

        # Initialize Listeners based on enabled chains
        if sniper_config.get('evm_enabled', True):
            from modules.sniper.core.evm_listener import EVMListener
            self.evm_listener = EVMListener(self.config)
            await self.evm_listener.initialize()

        if sniper_config.get('solana_enabled', True):
            from modules.sniper.core.solana_listener import SolanaListener
            self.solana_listener = SolanaListener(self.config)
            await self.solana_listener.initialize()

        logger.info("✅ Sniper Engine initialized")

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
                        elif key == 'safety_check_enabled':
                            self.safety_check_enabled = val.lower() in ('true', '1', 'yes') if val else True
                        elif key == 'test_mode':
                            self.test_mode = val.lower() in ('true', '1', 'yes') if val else False

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

            logger.info(f"📋 Sniper settings loaded: trade_amount={self.trade_amount}, slippage={self.slippage}%")
            logger.info(f"   Mode: {' | '.join(mode_info)}")
            logger.info(f"   Safety: max_tax={self.max_buy_tax}%, min_liq=${self.min_liquidity}")

        except Exception as e:
            logger.error(f"Error loading sniper settings: {e}")

    async def run(self):
        """Main loop"""
        self.is_running = True
        logger.info("🔫 Sniper Engine Started")

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

    async def _evaluate_target(self, target: Dict, chain_type: str):
        """Evaluate if a new token meets sniping criteria"""
        try:
            token_address = target.get('token_address')

            # 1. Check Filters (Liquidity, Tax, Honeypot, Safety)
            if not await self._check_filters(target, chain_type):
                return

            # 2. Add to pending targets
            logger.info(f"🎯 SNIPER TARGET ACQUIRED: {token_address} ({chain_type})")
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

        # Log stats periodically (every 5 minutes)
        await self._log_stats_if_needed()

        # Skip safety check if disabled
        if not self.safety_check_enabled:
            logger.debug(f"Safety check disabled, allowing {token_address}")
            return True

        # Test mode uses relaxed thresholds
        if self.test_mode:
            max_buy_tax = 50.0     # Allow up to 50% in test mode
            max_sell_tax = 50.0
            min_liquidity = 100.0  # Allow lower liquidity in test mode
            allow_caution = True   # Allow CAUTION rated tokens
        else:
            max_buy_tax = self.max_buy_tax
            max_sell_tax = self.max_sell_tax
            min_liquidity = self.min_liquidity
            allow_caution = False

        # Perform comprehensive safety check
        try:
            if self.token_safety:
                from modules.sniper.core.token_safety import SafetyRating

                report = await self.token_safety.check_token(token_address, chain_type)

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
                    return False

                if report.rating == SafetyRating.DANGER:
                    self._stats['danger_ratings'] += 1
                    # In test mode, allow DANGER tokens but log warning
                    if self.test_mode:
                        logger.warning(f"⚠️ TEST MODE: Allowing DANGER token {token_address[:16]}...")
                    else:
                        self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                        logger.debug(f"🚨 DANGER: {token_address[:16]}... (Score: {report.score})")
                        return False

                if report.rating == SafetyRating.CAUTION and not allow_caution and not self.test_mode:
                    logger.debug(f"⚠️ CAUTION: {token_address[:16]}... (Score: {report.score})")
                    # Allow CAUTION tokens in production (they often have minor issues)

                if report.buy_tax > max_buy_tax or report.sell_tax > max_sell_tax:
                    self._stats['high_tax_rejected'] += 1
                    self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                    logger.debug(f"⚠️ High tax: {token_address[:16]}... (Buy: {report.buy_tax:.1f}%, Sell: {report.sell_tax:.1f}%)")
                    return False

                if report.liquidity_usd < min_liquidity:
                    self._stats['low_liquidity_rejected'] += 1
                    self._rejected_cache[token_address] = datetime.now()  # Add to cooldown
                    logger.debug(f"⚠️ Low liquidity: {token_address[:16]}... (${report.liquidity_usd:,.0f})")
                    return False

                # Token passed all checks - log this at INFO level
                self._stats['passed_safety'] += 1
                mode_tag = "[TEST] " if self.test_mode else ""
                logger.info(f"✅ {mode_tag}Safety PASSED: {token_address} (Score: {report.score}/100, Liq: ${report.liquidity_usd:,.0f})")
                return True

        except Exception as e:
            logger.error(f"Error during safety check: {e}")
            # Fail safe - don't snipe if safety check errors
            return False

        return True

    async def _log_stats_if_needed(self):
        """Log filter statistics every 5 minutes"""
        now = datetime.now()
        elapsed = (now - self._stats['last_stats_log']).total_seconds()

        if elapsed >= 300:  # 5 minutes
            total = self._stats['tokens_analyzed']
            passed = self._stats['passed_safety']
            pass_rate = (passed / total * 100) if total > 0 else 0

            logger.info(f"📊 SNIPER STATS (Last 5 min): "
                       f"Analyzed: {total} | "
                       f"Passed: {passed} ({pass_rate:.1f}%) | "
                       f"Honeypots: {self._stats['honeypots_detected']} | "
                       f"Danger: {self._stats['danger_ratings']} | "
                       f"High Tax: {self._stats['high_tax_rejected']} | "
                       f"Low Liq: {self._stats['low_liquidity_rejected']}")

            # Reset stats
            self._stats = {
                'tokens_analyzed': 0,
                'honeypots_detected': 0,
                'danger_ratings': 0,
                'high_tax_rejected': 0,
                'low_liquidity_rejected': 0,
                'passed_safety': 0,
                'last_stats_log': now
            }

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

        logger.info(f"🔫 EXECUTING SNIPE: {token_address} on {chain}")
        data['status'] = 'buying'

        try:
            if not self.executor:
                logger.error("Trade executor not initialized")
                data['status'] = 'failed'
                return

            # Execute buy using the trade executor
            result = await self.executor.execute_buy(
                token_address=token_address,
                chain=chain,
                amount_in=self.trade_amount,
                slippage=self.slippage,
                priority_fee=self.priority_fee
            )

            if result.success:
                logger.info(f"✅ SNIPE SUCCESS: {token_address}")
                logger.info(f"   TX: {result.tx_hash} | Amount: {result.amount_out}")

                # Determine native token and get real USD price
                native_token = 'sol' if chain == 'solana' else 'eth'
                native_price = await self.price_fetcher.get_price(native_token)
                entry_usd = self.trade_amount * native_price

                data['status'] = 'active'
                data['entry_price'] = self.trade_amount / result.amount_out if result.amount_out > 0 else 0
                data['amount_bought'] = result.amount_out
                data['tx_hash'] = result.tx_hash
                data['entry_time'] = result.timestamp
                data['entry_usd'] = entry_usd  # Store USD value for accurate exit PnL
                data['native_price_at_entry'] = native_price

                logger.info(f"   Entry value: ${entry_usd:.2f} ({self.trade_amount:.4f} {native_token.upper()} @ ${native_price:.2f})")

                self.active_snipes[token_address] = data
                del self.pending_targets[token_address]

                # Log to database
                await self._log_snipe_to_db(data, result)
            else:
                logger.error(f"❌ SNIPE FAILED: {result.error}")
                data['status'] = 'failed'
                data['error'] = result.error

        except Exception as e:
            logger.error(f"❌ SNIPE FAILED: {e}")
            data['status'] = 'failed'
            data['error'] = str(e)

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
                        True,  # is_simulated - update based on dry_run mode
                        result.timestamp,
                        result.tx_hash,
                        json.dumps({
                            'warnings': safety_report.get('warnings', [])
                        })
                    )
                # Store trade_id in data for linking exit trade
                data['db_trade_id'] = trade_id
                logger.debug(f"💾 Logged to sniper_trades: {trade_id} ${usd_value:.2f}")
        except Exception as e:
            logger.error(f"Error logging snipe to DB: {e}")

    async def _monitor_active_snipes(self):
        """Monitor active snipes for auto-sell targets (take profit / stop loss)"""
        # Exit settings
        take_profit_pct = 50.0  # +50% profit target
        stop_loss_pct = -20.0   # -20% stop loss

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

                    # Get current price (simplified - in production use DEX price feeds)
                    current_price = await self._get_token_price(address, chain)

                    if current_price <= 0:
                        continue

                    # Calculate P&L
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    # Log status periodically
                    if check_count % 60 == 0:  # Every minute
                        logger.info(f"📊 Position: {address[:8]}... | Entry: {entry_price:.8f} | Current: {current_price:.8f} | P&L: {pnl_pct:+.2f}%")

                    # Check take profit
                    if pnl_pct >= take_profit_pct:
                        logger.info(f"🎯 TAKE PROFIT triggered for {address[:8]}... (+{pnl_pct:.2f}%)")
                        await self._exit_position(data, 'TAKE_PROFIT')

                    # Check stop loss
                    elif pnl_pct <= stop_loss_pct:
                        logger.warning(f"🛑 STOP LOSS triggered for {address[:8]}... ({pnl_pct:.2f}%)")
                        await self._exit_position(data, 'STOP_LOSS')

                except Exception as e:
                    logger.error(f"Error monitoring snipe {address}: {e}")

            await asyncio.sleep(1)

    async def _get_token_price(self, token_address: str, chain: str) -> float:
        """Get current token price (simplified)"""
        try:
            import aiohttp

            if chain == 'solana':
                # Use Jupiter for price (SOL per token)
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            price_info = data.get('data', {}).get(token_address, {})
                            return float(price_info.get('price', 0))
            else:
                # For EVM, use DexScreener or similar
                url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()
                            pairs = data.get('pairs', [])
                            if pairs:
                                return float(pairs[0].get('priceNative', 0))

        except Exception as e:
            logger.debug(f"Error fetching price for {token_address}: {e}")

        return 0

    async def _exit_position(self, data: Dict, reason: str):
        """Exit a position (sell tokens)"""
        token_address = data['target'].get('token_address')
        chain = data.get('chain_type', 'solana')
        amount = data.get('amount_bought', 0)

        logger.info(f"💰 Exiting position: {token_address} | Reason: {reason}")

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
                logger.info(f"✅ EXIT SUCCESS: {token_address}")
                logger.info(f"   TX: {result.tx_hash} | Received: {result.amount_out}")

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
                            True,
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
