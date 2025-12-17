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

    async def initialize(self):
        logger.info("🧠 Initializing Sentiment Engine...")
        if not self.openai_api_key:
            logger.warning("⚠️ No OpenAI API Key found. AI analysis will be skipped.")
        else:
            logger.info("✅ OpenAI API Key loaded.")

    async def run(self):
        """Main analysis loop"""
        self.is_running = True
        logger.info("🧠 Sentiment Engine Started")

        while self.is_running:
            try:
                # 1. Fetch Market News
                news_data = await self._fetch_news()

                if news_data and self.openai_api_key:
                    # 2. Analyze Sentiment with LLM
                    sentiment_score = await self._analyze_with_llm(news_data)
                    logger.info(f"🧠 Market Sentiment Score: {sentiment_score:.2f}")

                    # 3. Store Result in DB
                    await self._store_sentiment(sentiment_score)

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

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Sentiment Engine Stopped")
