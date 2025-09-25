"""
Social Data Collector - Social media sentiment analysis for ClaudeDex Trading Bot

This module collects and analyzes social media data for sentiment analysis.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import hashlib

import aiohttp
from loguru import logger
import numpy as np
from textblob import TextBlob

from ...utils.helpers import retry_async, measure_time, TTLCache
from ...utils.constants import API_RATE_LIMITS


class SentimentLevel(Enum):
    """Sentiment classification levels"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class SocialPlatform(Enum):
    """Social media platforms"""
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"


@dataclass
class SocialMetrics:
    """Social media metrics for a token"""
    token_address: str
    symbol: str
    mentions_24h: int
    mentions_change: float  # Percentage change
    sentiment_score: float  # -1 to 1
    sentiment_level: SentimentLevel
    engagement_rate: float
    unique_users: int
    influential_mentions: int
    platforms: Dict[str, Dict[str, Any]]
    trending_score: float
    fomo_index: float
    fear_index: float
    social_volume: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InfluencerMention:
    """Influencer mention data"""
    username: str
    platform: SocialPlatform
    followers: int
    message: str
    sentiment: float
    engagement: int  # likes + retweets + comments
    timestamp: datetime
    url: str
    verified: bool = False


class SocialDataCollector:
    """Collects and analyzes social media data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API configurations
        self.twitter_bearer = config.get("twitter_bearer_token")
        self.reddit_client_id = config.get("reddit_client_id")
        self.reddit_secret = config.get("reddit_secret")
        self.telegram_api_key = config.get("telegram_api_key")
        
        # Rate limiting
        self.rate_limits = {
            SocialPlatform.TWITTER: 300,  # requests per 15 min
            SocialPlatform.REDDIT: 60,    # requests per minute
            SocialPlatform.TELEGRAM: 30,  # requests per minute
        }
        
        # Caching
        self.cache = TTLCache(ttl=300)  # 5 minute cache
        self.sentiment_cache = TTLCache(ttl=600)  # 10 minute cache
        
        # Tracking
        self.tracked_tokens: Dict[str, Dict] = {}
        self.influencers: Dict[str, Dict] = {}
        self.trending_tokens: List[str] = []
        
        # Sentiment analysis
        self.positive_keywords = {
            "moon", "bullish", "pump", "buy", "long", "hold", "diamond hands",
            "to the moon", "lfg", "wagmi", "100x", "gem", "alpha"
        }
        
        self.negative_keywords = {
            "dump", "bearish", "sell", "short", "rug", "scam", "avoid",
            "red flag", "stay away", "ponzi", "honeypot", "exit"
        }
        
        # Influencer thresholds
        self.influencer_thresholds = {
            SocialPlatform.TWITTER: 10000,
            SocialPlatform.REDDIT: 5000,
            SocialPlatform.TELEGRAM: 1000
        }
        
    async def initialize(self) -> None:
        """Initialize social data collector"""
        logger.info("Initializing Social Data Collector...")
        
        # Load influencer list
        await self._load_influencers()
        
        # Test API connections
        await self._test_connections()
        
        logger.info("Social Data Collector initialized")
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def collect_social_metrics(
        self,
        token_address: str,
        symbol: str,
        chain: str
    ) -> SocialMetrics:
        """
        Collect comprehensive social metrics for a token
        
        Args:
            token_address: Token contract address
            symbol: Token symbol
            chain: Blockchain network
            
        Returns:
            Social metrics data
        """
        try:
            # Check cache
            cache_key = f"social_{token_address}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            # Collect from all platforms
            twitter_data = await self._collect_twitter_data(symbol)
            reddit_data = await self._collect_reddit_data(symbol)
            telegram_data = await self._collect_telegram_data(symbol)
            
            # Aggregate metrics
            total_mentions = (
                twitter_data.get("mentions", 0) +
                reddit_data.get("mentions", 0) +
                telegram_data.get("mentions", 0)
            )
            
            # Calculate sentiment
            overall_sentiment = await self._calculate_overall_sentiment(
                twitter_data.get("messages", []) +
                reddit_data.get("messages", []) +
                telegram_data.get("messages", [])
            )
            
            # Calculate engagement
            total_engagement = (
                twitter_data.get("engagement", 0) +
                reddit_data.get("engagement", 0) +
                telegram_data.get("engagement", 0)
            )
            
            engagement_rate = (
                total_engagement / total_mentions if total_mentions > 0 else 0
            )
            
            # Count unique users
            unique_users = len(set(
                twitter_data.get("users", []) +
                reddit_data.get("users", []) +
                telegram_data.get("users", [])
            ))
            
            # Count influential mentions
            influential_mentions = await self._count_influential_mentions(
                twitter_data, reddit_data, telegram_data
            )
            
            # Calculate indices
            fomo_index = self._calculate_fomo_index(
                total_mentions, overall_sentiment, engagement_rate
            )
            
            fear_index = self._calculate_fear_index(
                overall_sentiment, 
                self._count_negative_keywords(
                    twitter_data.get("messages", []) +
                    reddit_data.get("messages", [])
                )
            )
            
            # Calculate trending score
            trending_score = self._calculate_trending_score(
                total_mentions,
                engagement_rate,
                unique_users
            )
            
            # Previous period comparison
            prev_metrics = await self._get_previous_metrics(token_address)
            mentions_change = 0
            if prev_metrics and prev_metrics.get("mentions_24h", 0) > 0:
                mentions_change = (
                    (total_mentions - prev_metrics["mentions_24h"]) / 
                    prev_metrics["mentions_24h"] * 100
                )
            
            # Create metrics object
            metrics = SocialMetrics(
                token_address=token_address,
                symbol=symbol,
                mentions_24h=total_mentions,
                mentions_change=mentions_change,
                sentiment_score=overall_sentiment,
                sentiment_level=self._classify_sentiment(overall_sentiment),
                engagement_rate=engagement_rate,
                unique_users=unique_users,
                influential_mentions=influential_mentions,
                platforms={
                    "twitter": twitter_data,
                    "reddit": reddit_data,
                    "telegram": telegram_data
                },
                trending_score=trending_score,
                fomo_index=fomo_index,
                fear_index=fear_index,
                social_volume=total_mentions
            )
            
            # Cache result
            self.cache.set(cache_key, metrics)
            
            # Store for historical tracking
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect social metrics: {e}")
            raise
    
    async def _collect_twitter_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data from Twitter"""
        if not self.twitter_bearer:
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
        
        try:
            headers = {"Authorization": f"Bearer {self.twitter_bearer}"}
            
            # Search for token mentions
            query = f"${symbol} OR #{symbol} -is:retweet"
            params = {
                "query": query,
                "max_results": 100,
                "tweet.fields": "public_metrics,author_id,created_at",
                "user.fields": "public_metrics,verified"
            }
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.twitter.com/2/tweets/search/recent"
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tweets = data.get("data", [])
                        
                        messages = []
                        users = []
                        engagement = 0
                        
                        for tweet in tweets:
                            messages.append(tweet.get("text", ""))
                            users.append(tweet.get("author_id", ""))
                            
                            metrics = tweet.get("public_metrics", {})
                            engagement += (
                                metrics.get("like_count", 0) +
                                metrics.get("retweet_count", 0) +
                                metrics.get("reply_count", 0)
                            )
                        
                        return {
                            "mentions": len(tweets),
                            "messages": messages,
                            "users": users,
                            "engagement": engagement,
                            "raw_data": tweets
                        }
            
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
            
        except Exception as e:
            logger.error(f"Twitter collection failed: {e}")
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
    
    async def _collect_reddit_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data from Reddit"""
        if not self.reddit_client_id or not self.reddit_secret:
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
        
        try:
            # Get Reddit access token
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_secret)
            headers = {"User-Agent": "ClaudeDex/1.0"}
            
            async with aiohttp.ClientSession() as session:
                # Get access token
                token_url = "https://www.reddit.com/api/v1/access_token"
                token_data = {"grant_type": "client_credentials"}
                
                async with session.post(
                    token_url, 
                    auth=auth, 
                    data=token_data, 
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
                    
                    token = (await resp.json())["access_token"]
                
                # Search for mentions
                headers["Authorization"] = f"Bearer {token}"
                search_url = "https://oauth.reddit.com/search"
                params = {
                    "q": symbol,
                    "sort": "new",
                    "limit": 100,
                    "type": "link,comment"
                }
                
                async with session.get(
                    search_url,
                    headers=headers,
                    params=params
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])
                        
                        messages = []
                        users = []
                        engagement = 0
                        
                        for post in posts:
                            post_data = post.get("data", {})
                            
                            # Get text content
                            if "selftext" in post_data:
                                messages.append(post_data["selftext"])
                            elif "body" in post_data:
                                messages.append(post_data["body"])
                            
                            users.append(post_data.get("author", ""))
                            engagement += post_data.get("score", 0)
                        
                        return {
                            "mentions": len(posts),
                            "messages": messages,
                            "users": users,
                            "engagement": engagement,
                            "raw_data": posts
                        }
            
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
            
        except Exception as e:
            logger.error(f"Reddit collection failed: {e}")
            return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
    
    async def _collect_telegram_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data from Telegram"""
        # Telegram collection would require specific channel monitoring
        # This is a placeholder implementation
        return {"mentions": 0, "messages": [], "users": [], "engagement": 0}
    
    async def _calculate_overall_sentiment(
        self,
        messages: List[str]
    ) -> float:
        """Calculate overall sentiment from messages"""
        if not messages:
            return 0.0
        
        sentiments = []
        
        for message in messages:
            # Use TextBlob for basic sentiment
            blob = TextBlob(message)
            sentiment = blob.sentiment.polarity  # -1 to 1
            
            # Adjust for crypto-specific keywords
            message_lower = message.lower()
            
            # Positive keywords boost
            for keyword in self.positive_keywords:
                if keyword in message_lower:
                    sentiment += 0.1
            
            # Negative keywords penalty
            for keyword in self.negative_keywords:
                if keyword in message_lower:
                    sentiment -= 0.1
            
            # Clamp to [-1, 1]
            sentiment = max(-1, min(1, sentiment))
            sentiments.append(sentiment)
        
        return np.mean(sentiments)
    
    def _classify_sentiment(self, score: float) -> SentimentLevel:
        """Classify sentiment score into levels"""
        if score >= 0.5:
            return SentimentLevel.VERY_BULLISH
        elif score >= 0.2:
            return SentimentLevel.BULLISH
        elif score >= -0.2:
            return SentimentLevel.NEUTRAL
        elif score >= -0.5:
            return SentimentLevel.BEARISH
        else:
            return SentimentLevel.VERY_BEARISH
    
    async def _count_influential_mentions(
        self,
        twitter_data: Dict,
        reddit_data: Dict,
        telegram_data: Dict
    ) -> int:
        """Count mentions from influential accounts"""
        count = 0
        
        # Check Twitter influencers
        # This would check against known influencer list
        # For now, returning placeholder
        
        return count
    
    def _calculate_fomo_index(
        self,
        mentions: int,
        sentiment: float,
        engagement: float
    ) -> float:
        """Calculate FOMO (Fear of Missing Out) index"""
        # Normalize inputs
        mention_score = min(mentions / 1000, 1.0)  # Cap at 1000 mentions
        sentiment_score = (sentiment + 1) / 2  # Convert to 0-1
        
        # FOMO increases with positive sentiment and high engagement
        fomo = (
            mention_score * 0.3 +
            sentiment_score * 0.4 +
            engagement * 0.3
        )
        
        return min(1.0, fomo)
    
    def _calculate_fear_index(
        self,
        sentiment: float,
        negative_count: int
    ) -> float:
        """Calculate fear index"""
        # Fear increases with negative sentiment
        fear_from_sentiment = max(0, -sentiment)
        
        # Add fear from negative keyword count
        fear_from_keywords = min(negative_count / 50, 1.0)
        
        fear = (fear_from_sentiment * 0.7 + fear_from_keywords * 0.3)
        
        return min(1.0, fear)
    
    def _calculate_trending_score(
        self,
        mentions: int,
        engagement: float,
        unique_users: int
    ) -> float:
        """Calculate trending score"""
        # Normalize factors
        mention_score = min(mentions / 500, 1.0)
        user_score = min(unique_users / 100, 1.0)
        
        trending = (
            mention_score * 0.4 +
            engagement * 0.3 +
            user_score * 0.3
        )
        
        return min(1.0, trending)
    
    def _count_negative_keywords(self, messages: List[str]) -> int:
        """Count negative keywords in messages"""
        count = 0
        for message in messages:
            message_lower = message.lower()
            for keyword in self.negative_keywords:
                if keyword in message_lower:
                    count += 1
        return count
    
    async def _get_previous_metrics(
        self,
        token_address: str
    ) -> Optional[Dict[str, Any]]:
        """Get previous period metrics for comparison"""
        # This would fetch from database
        # For now, returning None
        return None
    
    async def _store_metrics(self, metrics: SocialMetrics) -> None:
        """Store metrics for historical tracking"""
        # This would store in database
        pass
    
    async def _load_influencers(self) -> None:
        """Load list of known crypto influencers"""
        # This would load from database or config
        self.influencers = {
            # Twitter influencers
            "elonmusk": {"platform": "twitter", "followers": 150000000},
            "VitalikButerin": {"platform": "twitter", "followers": 5000000},
            # Add more influencers
        }
    
    async def _test_connections(self) -> None:
        """Test API connections"""
        if self.twitter_bearer:
            try:
                # Test Twitter API
                headers = {"Authorization": f"Bearer {self.twitter_bearer}"}
                async with aiohttp.ClientSession() as session:
                    url = "https://api.twitter.com/2/tweets/search/recent"
                    params = {"query": "test", "max_results": 10}
                    async with session.get(url, headers=headers, params=params) as resp:
                        if resp.status == 200:
                            logger.info("Twitter API connected")
                        else:
                            logger.warning(f"Twitter API error: {resp.status}")
            except Exception as e:
                logger.error(f"Twitter connection test failed: {e}")
    
    async def monitor_trending(self) -> AsyncGenerator[List[str], None]:
        """Monitor trending tokens"""
        while True:
            try:
                # Get trending topics from various platforms
                trending = await self._get_trending_tokens()
                
                if trending != self.trending_tokens:
                    self.trending_tokens = trending
                    yield trending
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Trending monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _get_trending_tokens(self) -> List[str]:
        """Get list of trending tokens"""
        trending = []
        
        # Aggregate from different sources
        # This would query APIs for trending topics
        
        return trending
    
    async def get_influencer_mentions(
        self,
        token_symbol: str,
        hours: int = 24
    ) -> List[InfluencerMention]:
        """Get recent influencer mentions for a token"""
        mentions = []
        
        # Search for mentions by known influencers
        # This would query social platforms
        
        return mentions
    
    async def analyze_social_velocity(
        self,
        token_address: str,
        symbol: str
    ) -> Dict[str, Any]:
        """Analyze rate of change in social metrics"""
        # Get historical data points
        history = []  # Would fetch from database
        
        if len(history) < 2:
            return {
                "velocity": 0,
                "acceleration": 0,
                "trend": "stable"
            }
        
        # Calculate velocity (rate of change)
        recent = history[-1]
        previous = history[-2]
        
        time_diff = (recent["timestamp"] - previous["timestamp"]).total_seconds() / 3600
        mention_diff = recent["mentions"] - previous["mentions"]
        
        velocity = mention_diff / time_diff if time_diff > 0 else 0
        
        # Calculate acceleration if enough data
        acceleration = 0
        if len(history) >= 3:
            older = history[-3]
            prev_velocity = (previous["mentions"] - older["mentions"]) / time_diff
            acceleration = velocity - prev_velocity
        
        # Determine trend
        if velocity > 10:
            trend = "rapidly_increasing"
        elif velocity > 5:
            trend = "increasing"
        elif velocity < -10:
            trend = "rapidly_decreasing"
        elif velocity < -5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "trend": trend
        }
    
    def calculate_social_score(self, metrics: SocialMetrics) -> float:
        """Calculate overall social score for trading decisions"""
        score = 0.5  # Start neutral
        
        # Sentiment contribution (30%)
        sentiment_contribution = (metrics.sentiment_score + 1) / 2 * 0.3
        score += sentiment_contribution - 0.15  # Center around neutral
        
        # Volume contribution (25%)
        if metrics.mentions_24h > 1000:
            score += 0.15
        elif metrics.mentions_24h > 100:
            score += 0.10
        elif metrics.mentions_24h < 10:
            score -= 0.10
        
        # Trending contribution (20%)
        score += metrics.trending_score * 0.2
        
        # FOMO/Fear balance (15%)
        emotion_balance = metrics.fomo_index - metrics.fear_index
        score += emotion_balance * 0.15
        
        # Influencer contribution (10%)
        if metrics.influential_mentions > 0:
            score += min(metrics.influential_mentions / 10, 0.1)
        
        return max(0, min(1, score))