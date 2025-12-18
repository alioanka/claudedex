"""
Sentiment Engine - AI-powered market sentiment analysis

Features:
- LLM-powered sentiment analysis using OpenAI
- Real trade execution (when DRY_RUN=false)
- Exit strategy with take profit and stop loss
- Position tracking and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import aiohttp
import os

logger = logging.getLogger("SentimentEngine")
openai_logger = logging.getLogger("OpenAI_API")


class AITradeExecutor:
    """
    Trade executor for AI module.
    Supports both DEX (for crypto) and CEX (for futures) execution.
    """

    def __init__(self, config: Dict, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.session: Optional[aiohttp.ClientSession] = None

        # Binance API for futures trading
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_API_SECRET')

    async def initialize(self):
        """Initialize executor"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"💱 AI Trade Executor initialized ({mode})")

    async def close(self):
        """Close executor"""
        if self.session:
            await self.session.close()
            self.session = None

    async def execute_trade(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount_usd: float,
        price: float = 0
    ) -> Dict:
        """
        Execute a trade.

        Returns:
            Dict with trade result
        """
        if self.dry_run:
            return await self._simulate_trade(symbol, side, amount_usd, price)

        # Real execution - Binance Futures
        if self.binance_api_key and self.binance_secret:
            return await self._execute_binance_futures(symbol, side, amount_usd)

        logger.error("No exchange credentials configured for live trading")
        return {'success': False, 'error': 'No exchange configured'}

    async def _simulate_trade(
        self,
        symbol: str,
        side: str,
        amount_usd: float,
        price: float
    ) -> Dict:
        """Simulate a trade for DRY RUN mode"""
        # Fetch current price if not provided
        if price <= 0:
            price = await self._get_current_price(symbol)

        if price <= 0:
            return {'success': False, 'error': 'Could not fetch price'}

        amount = amount_usd / price

        logger.info(f"🧪 [DRY RUN] Simulated {side.upper()}: {amount:.6f} {symbol} @ ${price:,.2f}")

        import hashlib
        fake_hash = hashlib.sha256(f"{symbol}{datetime.now().timestamp()}".encode()).hexdigest()

        return {
            'success': True,
            'order_id': f"DRY_RUN_{fake_hash[:12]}",
            'symbol': symbol,
            'side': side,
            'price': price,
            'amount': amount,
            'amount_usd': amount_usd,
            'timestamp': datetime.now()
        }

    async def _execute_binance_futures(
        self,
        symbol: str,
        side: str,
        amount_usd: float
    ) -> Dict:
        """Execute trade on Binance Futures"""
        try:
            import hmac
            import hashlib
            import time

            # Get current price
            price = await self._get_current_price(symbol)
            if price <= 0:
                return {'success': False, 'error': 'Could not fetch price'}

            # Calculate quantity
            quantity = round(amount_usd / price, 3)

            # Binance Futures API
            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/order"

            timestamp = int(time.time() * 1000)

            # Map symbol to Binance format
            binance_symbol = f"{symbol}USDT"

            params = {
                'symbol': binance_symbol,
                'side': 'BUY' if side == 'buy' else 'SELL',
                'type': 'MARKET',
                'quantity': quantity,
                'timestamp': timestamp
            }

            # Create signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.binance_secret.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()

            params['signature'] = signature

            headers = {'X-MBX-APIKEY': self.binance_api_key}

            async with self.session.post(
                f"{base_url}{endpoint}",
                params=params,
                headers=headers
            ) as response:
                data = await response.json()

                if response.status == 200:
                    logger.info(f"✅ Binance Futures order placed: {data.get('orderId')}")
                    return {
                        'success': True,
                        'order_id': data.get('orderId'),
                        'symbol': symbol,
                        'side': side,
                        'price': float(data.get('avgPrice', price)),
                        'amount': float(data.get('executedQty', quantity)),
                        'amount_usd': amount_usd,
                        'timestamp': datetime.now()
                    }
                else:
                    logger.error(f"Binance error: {data}")
                    return {'success': False, 'error': data.get('msg', 'Unknown error')}

        except Exception as e:
            logger.error(f"Binance execution error: {e}")
            return {'success': False, 'error': str(e)}

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Use Binance public API
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('price', 0))
        except Exception as e:
            logger.debug(f"Error fetching price for {symbol}: {e}")
        return 0


class SentimentEngine:
    """
    Analyzes market sentiment using LLMs (OpenAI) and public news feeds.
    Provides signals to other trading modules.

    Features:
    - Real-time sentiment analysis
    - Automated trade execution
    - Take profit / Stop loss exit strategy
    - Position tracking
    """

    def __init__(self, config: Dict, db_pool):
        self.config = config
        self.db_pool = db_pool
        self.is_running = False
        self.openai_api_key = config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')

        # Trading settings (loaded from DB/Config)
        self.direct_trading = False
        self.confidence_threshold = 0.85
        self.trade_amount_usd = 50.0
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() in ('true', '1', 'yes')

        # Exit strategy settings
        self.take_profit_pct = 5.0   # +5% take profit
        self.stop_loss_pct = -3.0    # -3% stop loss
        self.max_hold_hours = 24     # Maximum position hold time

        # Trade executor
        self.executor: Optional[AITradeExecutor] = None

        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}

    async def initialize(self):
        logger.info("🧠 Initializing Sentiment Engine...")

        # Load settings from DB if available
        await self._load_settings()

        # Initialize trade executor
        self.executor = AITradeExecutor(self.config, self.dry_run)
        await self.executor.initialize()

        # Load active positions from DB
        await self._load_active_positions()

        if not self.openai_api_key:
            logger.warning("⚠️ No OpenAI API Key found. AI analysis will be skipped.")
        else:
            logger.info("✅ OpenAI API Key loaded.")

        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Direct Trading: {'Enabled' if self.direct_trading else 'Disabled'}")
        logger.info(f"   Exit Strategy: TP={self.take_profit_pct}%, SL={self.stop_loss_pct}%")
        logger.info(f"   Active Positions: {len(self.active_positions)}")

    async def _load_settings(self):
        """Load AI settings from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT key, value FROM config_settings WHERE config_type = 'ai_config'")
                for row in rows:
                    key = row['key']
                    val = row['value']

                    if key == 'direct_trading':
                        self.direct_trading = val.lower() == 'true'
                    elif key == 'confidence_threshold':
                        self.confidence_threshold = float(val) / 100.0 if float(val) > 1 else float(val)
                    elif key == 'trade_amount_usd':
                        self.trade_amount_usd = float(val)

            logger.info(f"Loaded AI Settings: direct_trading={self.direct_trading}, threshold={self.confidence_threshold}")
        except Exception as e:
            logger.warning(f"Failed to load AI settings: {e}")

    async def run(self):
        """Main analysis loop"""
        self.is_running = True
        logger.info("🧠 Sentiment Engine Started")

        # Start position monitor in background
        position_task = asyncio.create_task(self._monitor_positions())

        cycle_count = 0
        trades_executed = 0

        while self.is_running:
            try:
                cycle_count += 1
                # Reload settings occasionally
                await self._load_settings()

                logger.info(f"🧠 Cycle {cycle_count}: Fetching market news...")

                # 1. Fetch Market News
                news_data = await self._fetch_news()

                if news_data and self.openai_api_key:
                    logger.info(f"🧠 Retrieved {len(news_data)} headlines, analyzing...")

                    # 2. Analyze Sentiment with LLM
                    sentiment_score = await self._analyze_with_llm(news_data)
                    logger.info(f"🧠 Market Sentiment Score: {sentiment_score:.2f}")

                    # 3. Store Result in DB
                    await self._store_sentiment(sentiment_score)

                    # 4. Execute Trade if conditions met
                    if self.direct_trading and abs(sentiment_score) >= self.confidence_threshold:
                        await self._execute_trade(sentiment_score)
                        trades_executed += 1
                elif not news_data:
                    logger.info("🧠 No news data retrieved, skipping analysis")
                elif not self.openai_api_key:
                    logger.debug("🧠 No OpenAI API key, skipping LLM analysis")

                logger.info(f"🧠 Cycle {cycle_count} complete. Positions: {len(self.active_positions)}. Total trades: {trades_executed}")
                await asyncio.sleep(900)  # Run every 15 minutes to save API credits
            except Exception as e:
                logger.error(f"Error in sentiment loop: {e}")
                await asyncio.sleep(60)

        # Clean up
        position_task.cancel()
        try:
            await position_task
        except asyncio.CancelledError:
            pass

    async def _fetch_news(self) -> List[str]:
        """Fetch latest crypto news headlines from public API"""
        headlines = []
        try:
            # Using CryptoCompare News API (public free tier) as an example
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get('Data', [])[:10] # Get top 10
                        headlines = [a.get('title') for a in articles]
        except Exception as e:
            logger.debug(f"Failed to fetch news: {e}")

        return headlines

    async def _analyze_with_llm(self, texts: List[str]) -> float:
        """Send headlines to OpenAI and get a sentiment score (-1 to 1)"""
        try:
            prompt = (
                "Analyze the sentiment of the following crypto news headlines. "
                "Return a single float number between -1.0 (extremely bearish) and 1.0 (extremely bullish). "
                "Only return the number.\n\n" + "\n".join(texts)
            )

            # Log the OpenAI API request
            openai_logger.info("=" * 80)
            openai_logger.info(f"🤖 OpenAI API Request at {datetime.now().isoformat()}")
            openai_logger.info(f"   Model: gpt-3.5-turbo")
            openai_logger.info(f"   Headlines count: {len(texts)}")
            for i, headline in enumerate(texts[:5], 1):  # Log first 5 headlines
                openai_logger.info(f"   [{i}] {headline[:100]}...")
            if len(texts) > 5:
                openai_logger.info(f"   ... and {len(texts) - 5} more headlines")
            openai_logger.info("-" * 40)

            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-3.5-turbo", # Cost effective
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }

            async with aiohttp.ClientSession() as session:
                start_time = datetime.now()
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                    elapsed = (datetime.now() - start_time).total_seconds()

                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content'].strip()
                        usage = data.get('usage', {})

                        # Log the response
                        openai_logger.info(f"✅ OpenAI API Response (Status: 200)")
                        openai_logger.info(f"   Response time: {elapsed:.2f}s")
                        openai_logger.info(f"   Raw response: {content}")
                        openai_logger.info(f"   Tokens used: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}, total={usage.get('total_tokens', 'N/A')}")

                        try:
                            score = float(content)
                            openai_logger.info(f"   Parsed sentiment score: {score:.4f}")

                            # Store detailed log in database
                            await self._store_openai_log(texts, content, score, usage, elapsed)

                            return score
                        except ValueError:
                            openai_logger.error(f"❌ Could not parse response as float: {content}")
                            return 0.0
                    else:
                        error_text = await resp.text()
                        openai_logger.error(f"❌ OpenAI API Error: {resp.status}")
                        openai_logger.error(f"   Response: {error_text[:500]}")
                        logger.error(f"OpenAI API Error: {resp.status}")
                        return 0.0
        except Exception as e:
            openai_logger.error(f"❌ LLM analysis failed: {e}")
            logger.error(f"LLM analysis failed: {e}")
            return 0.0

    async def _store_openai_log(self, headlines: List[str], response: str, score: float, usage: Dict, elapsed: float):
        """Store detailed OpenAI API log in database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Check if table exists, create if not
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_analysis_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        headlines JSONB,
                        raw_response TEXT,
                        sentiment_score FLOAT,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        response_time_sec FLOAT,
                        model VARCHAR(50)
                    )
                """)

                await conn.execute("""
                    INSERT INTO ai_analysis_logs (
                        headlines, raw_response, sentiment_score,
                        prompt_tokens, completion_tokens, total_tokens,
                        response_time_sec, model
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    json.dumps(headlines),
                    response,
                    score,
                    usage.get('prompt_tokens', 0),
                    usage.get('completion_tokens', 0),
                    usage.get('total_tokens', 0),
                    elapsed,
                    'gpt-3.5-turbo'
                )
        except Exception as e:
            openai_logger.error(f"Failed to store OpenAI log: {e}")

    async def _store_sentiment(self, score: float):
        """Log sentiment score to database"""
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO sentiment_logs (score, source, timestamp)
                        VALUES ($1, $2, $3)
                    """, score, 'openai_news_analysis', datetime.now())
            except Exception as e:
                logger.error(f"Failed to store sentiment: {e}")

    async def _execute_trade(self, score: float):
        """Execute or simulate a trade based on sentiment"""
        side = "buy" if score > 0 else "sell"
        # For AI module, we focus on major assets like ETH/BTC for sentiment trading
        symbol = "ETH"
        action_type = "LONG" if side == "buy" else "SHORT"

        logger.info(f"🤖 AI Signal Triggered: {action_type} {symbol} (Score: {score:.2f})")

        # Check if we already have a position in this symbol
        if symbol in self.active_positions:
            logger.info(f"⚠️ Already have position in {symbol}, skipping new entry")
            return

        trade_id = f"ai_{int(datetime.now().timestamp())}"

        try:
            # Execute trade using executor
            result = await self.executor.execute_trade(
                symbol=symbol,
                side=side,
                amount_usd=self.trade_amount_usd
            )

            if not result.get('success'):
                logger.error(f"Trade execution failed: {result.get('error')}")
                return

            entry_price = result.get('price', 0)
            amount = result.get('amount', 0)

            # Track position
            self.active_positions[symbol] = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'amount': amount,
                'amount_usd': self.trade_amount_usd,
                'entry_time': datetime.now(),
                'sentiment_score': score,
                'order_id': result.get('order_id')
            }

            # Log trade to database
            if self.db_pool:
                metadata = json.dumps({
                    'token_symbol': symbol,
                    'sentiment_score': score,
                    'is_simulated': self.dry_run,
                    'reason': f"Sentiment score {score:.2f} >= {self.confidence_threshold}",
                    'order_id': result.get('order_id')
                })

                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO trades (
                            trade_id, token_address, chain, strategy,
                            side, entry_price, amount, usd_value, status,
                            entry_timestamp, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                        trade_id,
                        "0x0000000000000000000000000000000000000000",
                        "ethereum",
                        "ai_analysis",
                        side,
                        entry_price,
                        amount,
                        self.trade_amount_usd,
                        "open",
                        datetime.now(),
                        metadata
                    )

            logger.info(f"✅ AI Trade {'Simulated' if self.dry_run else 'Executed'}: {action_type} {symbol} @ ${entry_price:,.2f}")

        except Exception as e:
            logger.error(f"Failed to execute AI trade: {e}")

    async def _load_active_positions(self):
        """Load active AI positions from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT trade_id, side, entry_price, amount, usd_value, entry_timestamp, metadata
                    FROM trades
                    WHERE strategy = 'ai_analysis' AND status = 'open'
                    ORDER BY entry_timestamp DESC
                """)

                for row in rows:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    symbol = metadata.get('token_symbol', 'ETH')

                    self.active_positions[symbol] = {
                        'trade_id': row['trade_id'],
                        'symbol': symbol,
                        'side': row['side'],
                        'entry_price': float(row['entry_price']),
                        'amount': float(row['amount']),
                        'amount_usd': float(row['usd_value']),
                        'entry_time': row['entry_timestamp'],
                        'sentiment_score': metadata.get('sentiment_score', 0)
                    }

                logger.info(f"Loaded {len(self.active_positions)} active AI positions")

        except Exception as e:
            logger.warning(f"Error loading active positions: {e}")

    async def _monitor_positions(self):
        """Monitor active positions for exit conditions (TP/SL/Time)"""
        logger.info("📊 Position monitor started")

        while self.is_running:
            try:
                for symbol, position in list(self.active_positions.items()):
                    await self._check_exit_conditions(symbol, position)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)

    async def _check_exit_conditions(self, symbol: str, position: Dict):
        """Check if position should be closed"""
        try:
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            side = position['side']

            # Get current price
            current_price = await self.executor._get_current_price(symbol)
            if current_price <= 0:
                return

            # Calculate P&L
            if side == 'buy':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            # Check time-based exit
            hold_time = datetime.now() - entry_time
            hours_held = hold_time.total_seconds() / 3600

            exit_reason = None

            # Check take profit
            if pnl_pct >= self.take_profit_pct:
                exit_reason = f"TAKE_PROFIT (+{pnl_pct:.2f}%)"

            # Check stop loss
            elif pnl_pct <= self.stop_loss_pct:
                exit_reason = f"STOP_LOSS ({pnl_pct:.2f}%)"

            # Check max hold time
            elif hours_held >= self.max_hold_hours:
                exit_reason = f"TIME_EXIT ({hours_held:.1f}h)"

            if exit_reason:
                logger.info(f"🚪 Exit triggered for {symbol}: {exit_reason}")
                await self._close_position(symbol, position, exit_reason, current_price, pnl_pct)

        except Exception as e:
            logger.error(f"Error checking exit for {symbol}: {e}")

    async def _close_position(
        self,
        symbol: str,
        position: Dict,
        exit_reason: str,
        exit_price: float,
        pnl_pct: float
    ):
        """Close an active position"""
        try:
            trade_id = position['trade_id']
            side = position['side']
            amount = position['amount']

            # Execute closing trade (opposite side)
            close_side = 'sell' if side == 'buy' else 'buy'

            result = await self.executor.execute_trade(
                symbol=symbol,
                side=close_side,
                amount_usd=amount * exit_price,
                price=exit_price
            )

            if result.get('success'):
                logger.info(f"✅ Position closed: {symbol} | P&L: {pnl_pct:+.2f}% | Reason: {exit_reason}")

                # Update database
                if self.db_pool:
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE trades
                            SET status = 'closed',
                                exit_price = $1,
                                exit_timestamp = $2,
                                pnl = $3,
                                metadata = metadata || $4::jsonb
                            WHERE trade_id = $5
                        """,
                            exit_price,
                            datetime.now(),
                            (exit_price - position['entry_price']) * amount,
                            json.dumps({'exit_reason': exit_reason, 'exit_order_id': result.get('order_id')}),
                            trade_id
                        )

                # Remove from active positions
                del self.active_positions[symbol]

            else:
                logger.error(f"Failed to close position: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    async def stop(self):
        """Stop the sentiment engine"""
        self.is_running = False

        # Close all positions if emergency exit
        for symbol, position in list(self.active_positions.items()):
            try:
                current_price = await self.executor._get_current_price(symbol)
                if current_price > 0:
                    await self._close_position(symbol, position, "ENGINE_STOP", current_price, 0)
            except Exception as e:
                logger.error(f"Error closing position on stop: {e}")

        # Close executor
        if self.executor:
            await self.executor.close()

        logger.info("🛑 Sentiment Engine Stopped")
