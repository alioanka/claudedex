"""
Sentiment Engine - AI-powered market sentiment analysis
"""

import asyncio
import logging
from typing import Dict, List
import json
from datetime import datetime
import aiohttp
import os

logger = logging.getLogger("SentimentEngine")

class SentimentEngine:
    """
    Analyzes market sentiment using LLMs (OpenAI) and public news feeds.
    Provides signals to other trading modules.
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

    async def initialize(self):
        logger.info("🧠 Initializing Sentiment Engine...")

        # Load settings from DB if available
        await self._load_settings()

        if not self.openai_api_key:
            logger.warning("⚠️ No OpenAI API Key found. AI analysis will be skipped.")
        else:
            logger.info("✅ OpenAI API Key loaded.")

        logger.info(f"   Mode: {'DRY_RUN (Simulated)' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"   Direct Trading: {'Enabled' if self.direct_trading else 'Disabled'}")

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

                logger.info(f"🧠 Cycle {cycle_count} complete. Next analysis in 15 minutes. Total trades: {trades_executed}")
                await asyncio.sleep(900) # Run every 15 minutes to save API credits
            except Exception as e:
                logger.error(f"Error in sentiment loop: {e}")
                await asyncio.sleep(60)

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
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content'].strip()
                        return float(content)
                    else:
                        logger.error(f"OpenAI API Error: {resp.status}")
                        return 0.0
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return 0.0

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

        trade_id = f"ai_{int(datetime.now().timestamp())}"

        # Log trade to database
        if self.db_pool:
            try:
                # In dry run, we simulate the trade entry
                # In live mode, this would call an executor (not implemented in this scope)

                # Mock prices for simulation
                entry_price = 3500.0 # Placeholder
                amount = self.trade_amount_usd / entry_price

                async with self.db_pool.acquire() as conn:
                    # Note: 'token_symbol' column does not exist in 'trades' table in some schemas.
                    # We rely on metadata to store the symbol.
                    # Also 'token_address' is required.
                    metadata = json.dumps({
                        'token_symbol': symbol,
                        'sentiment_score': score,
                        'is_simulated': self.dry_run,
                        'reason': f"Sentiment score {score:.2f} >= {self.confidence_threshold}"
                    })

                    await conn.execute("""
                        INSERT INTO trades (
                            trade_id, token_address, chain, strategy,
                            side, entry_price, amount, usd_value, status,
                            entry_timestamp, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                        trade_id,
                        "0x0000000000000000000000000000000000000000", # Dummy address
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

                logger.info(f"✅ AI Trade {'Simulated' if self.dry_run else 'Logged'}: {action_type} {symbol}")

            except Exception as e:
                logger.error(f"Failed to execute AI trade: {e}")

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Sentiment Engine Stopped")
