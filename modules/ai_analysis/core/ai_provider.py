"""
AI Provider Manager - Enterprise-grade LLM integration

Supports:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic Claude (Claude 3.5 Sonnet, Claude 3.5 Haiku)
- Multi-provider fallback and load balancing
- Cost tracking and rate limiting
- Response validation and caching
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import hashlib

logger = logging.getLogger("AIProvider")


class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COMBINED = "combined"  # Use both and average


class AIModel(Enum):
    """Supported AI models with their costs per 1K tokens"""
    # OpenAI models - costs in USD per 1K tokens (input/output)
    GPT4O = ("gpt-4o", 0.0025, 0.01)
    GPT4O_MINI = ("gpt-4o-mini", 0.00015, 0.0006)
    GPT4_TURBO = ("gpt-4-turbo", 0.01, 0.03)

    # Anthropic models
    CLAUDE_35_SONNET = ("claude-3-5-sonnet-latest", 0.003, 0.015)
    CLAUDE_35_HAIKU = ("claude-3-5-haiku-latest", 0.001, 0.005)
    CLAUDE_3_OPUS = ("claude-3-opus-latest", 0.015, 0.075)

    def __init__(self, model_id: str, input_cost: float, output_cost: float):
        self.model_id = model_id
        self.input_cost = input_cost
        self.output_cost = output_cost


@dataclass
class APICallMetrics:
    """Metrics for a single API call"""
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cached: bool = False


@dataclass
class ProviderConfig:
    """Configuration for an AI provider"""
    provider: AIProvider
    api_key: str
    model: AIModel
    enabled: bool = True
    priority: int = 1  # Lower = higher priority
    rate_limit_rpm: int = 60  # Requests per minute
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout_seconds: int = 30


class ResponseCache:
    """Simple in-memory cache for LLM responses"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def _make_key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[Any]:
        key = self._make_key(prompt, model)
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, prompt: str, model: str, response: Any):
        key = self._make_key(prompt, model)
        self.cache[key] = (response, datetime.utcnow())

    def clear(self):
        self.cache.clear()


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            now = datetime.utcnow()
            elapsed = (now - self.last_refill).total_seconds()

            # Refill tokens based on elapsed time
            refill = int(elapsed * self.rpm / 60)
            if refill > 0:
                self.tokens = min(self.rpm, self.tokens + refill)
                self.last_refill = now

            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self, timeout: float = 60.0):
        start = datetime.utcnow()
        while True:
            if await self.acquire():
                return True
            if (datetime.utcnow() - start).total_seconds() > timeout:
                return False
            await asyncio.sleep(0.1)


class AIProviderManager:
    """
    Enterprise-grade AI provider manager with:
    - Multi-provider support (OpenAI, Claude)
    - Automatic fallback on failures
    - Cost tracking and budgeting
    - Rate limiting
    - Response caching
    - Comprehensive logging
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self.providers: Dict[AIProvider, ProviderConfig] = {}
        self.rate_limiters: Dict[AIProvider, RateLimiter] = {}
        self.cache = ResponseCache(ttl_seconds=300)

        # Cost tracking
        self.session_costs: Dict[str, float] = {}
        self.daily_budget_usd: float = 10.0
        self.daily_spent_usd: float = 0.0
        self.budget_reset_time: datetime = datetime.utcnow()

        # Metrics
        self.call_history: List[APICallMetrics] = []
        self._max_history = 1000

        # HTTP session
        self._session = None

    async def initialize(self):
        """Initialize the provider manager with credentials"""
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )

        # Load credentials from secrets manager
        openai_key = await self._get_api_key('OPENAI_API_KEY')
        anthropic_key = await self._get_api_key('ANTHROPIC_API_KEY')

        # Configure OpenAI if available
        if openai_key:
            self.providers[AIProvider.OPENAI] = ProviderConfig(
                provider=AIProvider.OPENAI,
                api_key=openai_key,
                model=AIModel.GPT4O_MINI,
                priority=2
            )
            self.rate_limiters[AIProvider.OPENAI] = RateLimiter(60)
            logger.info("✅ OpenAI provider configured (GPT-4o-mini)")
        else:
            logger.warning("⚠️ OpenAI API key not found")

        # Configure Anthropic if available
        if anthropic_key:
            self.providers[AIProvider.ANTHROPIC] = ProviderConfig(
                provider=AIProvider.ANTHROPIC,
                api_key=anthropic_key,
                model=AIModel.CLAUDE_35_HAIKU,
                priority=1  # Prefer Claude
            )
            self.rate_limiters[AIProvider.ANTHROPIC] = RateLimiter(60)
            logger.info("✅ Anthropic provider configured (Claude 3.5 Haiku)")
        else:
            logger.warning("⚠️ Anthropic API key not found")

        # Load budget from config
        await self._load_config()

        if not self.providers:
            logger.error("❌ No AI providers configured - AI analysis disabled")
            return False

        return True

    async def _get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from secrets manager or environment"""
        try:
            from security.secrets_manager import secrets
            key = secrets.get(key_name, log_access=False)
            if key:
                return key
        except Exception:
            pass

        return os.getenv(key_name)

    async def _load_config(self):
        """Load configuration from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT key, value, value_type FROM config_settings
                    WHERE config_type = 'ai_config'
                """)

                for row in rows:
                    key = row['key']
                    value = row['value']

                    if key == 'daily_budget_usd':
                        self.daily_budget_usd = float(value)
                    elif key == 'openai_model':
                        if AIProvider.OPENAI in self.providers:
                            self.providers[AIProvider.OPENAI].model = self._get_model(value)
                    elif key == 'claude_model':
                        if AIProvider.ANTHROPIC in self.providers:
                            self.providers[AIProvider.ANTHROPIC].model = self._get_model(value)

                logger.info(f"📊 AI Config: Daily budget ${self.daily_budget_usd}")
        except Exception as e:
            logger.warning(f"Could not load AI config: {e}")

    def _get_model(self, model_id: str) -> AIModel:
        """Get AIModel enum from model ID string"""
        for model in AIModel:
            if model.model_id == model_id:
                return model
        return AIModel.GPT4O_MINI  # Default

    async def analyze_sentiment(
        self,
        text: str,
        context: str = "",
        use_cache: bool = True,
        preferred_provider: Optional[AIProvider] = None
    ) -> Dict:
        """
        Analyze sentiment using configured AI providers.

        Returns:
            Dict with:
            - sentiment: float (-1.0 to 1.0)
            - confidence: float (0.0 to 1.0)
            - reasoning: str
            - provider: str
            - model: str
            - cached: bool
            - cost_usd: float
        """
        # Check daily budget
        if self.daily_spent_usd >= self.daily_budget_usd:
            logger.warning("⚠️ Daily AI budget exceeded")
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'reasoning': 'Budget exceeded',
                'provider': 'none',
                'model': 'none',
                'cached': False,
                'cost_usd': 0.0,
                'error': 'daily_budget_exceeded'
            }

        # Build sentiment analysis prompt
        prompt = self._build_sentiment_prompt(text, context)

        # Try cache first
        if use_cache:
            for provider in self._get_ordered_providers(preferred_provider):
                config = self.providers[provider]
                cached = self.cache.get(prompt, config.model.model_id)
                if cached:
                    logger.debug(f"Cache hit for {provider.value}")
                    cached['cached'] = True
                    return cached

        # Try each provider in priority order
        last_error = None
        for provider in self._get_ordered_providers(preferred_provider):
            if provider not in self.providers:
                continue

            config = self.providers[provider]
            if not config.enabled:
                continue

            # Check rate limit
            limiter = self.rate_limiters.get(provider)
            if limiter and not await limiter.wait_for_token(timeout=10):
                logger.warning(f"Rate limit reached for {provider.value}")
                continue

            try:
                result = await self._call_provider(provider, prompt)

                # Cache successful result
                if result.get('success') and use_cache:
                    self.cache.set(prompt, config.model.model_id, result)

                if result.get('success'):
                    return result

                last_error = result.get('error')
            except Exception as e:
                logger.error(f"Provider {provider.value} failed: {e}")
                last_error = str(e)

        # All providers failed
        return {
            'sentiment': 0.0,
            'confidence': 0.0,
            'reasoning': f'All providers failed: {last_error}',
            'provider': 'none',
            'model': 'none',
            'cached': False,
            'cost_usd': 0.0,
            'error': last_error
        }

    def _get_ordered_providers(self, preferred: Optional[AIProvider] = None) -> List[AIProvider]:
        """Get providers ordered by priority, with preferred first"""
        providers = list(self.providers.keys())
        providers.sort(key=lambda p: self.providers[p].priority)

        if preferred and preferred in providers:
            providers.remove(preferred)
            providers.insert(0, preferred)

        return providers

    def _build_sentiment_prompt(self, text: str, context: str = "") -> str:
        """Build the sentiment analysis prompt"""
        base_prompt = """You are a cryptocurrency market sentiment analyst. Analyze the following market information and provide a sentiment score.

INSTRUCTIONS:
1. Analyze the text for bullish or bearish signals
2. Consider market impact, credibility, and timing
3. Return a JSON response with exactly these fields:
   - sentiment: float from -1.0 (extremely bearish) to 1.0 (extremely bullish)
   - confidence: float from 0.0 (no confidence) to 1.0 (very confident)
   - reasoning: brief explanation of your analysis (max 100 words)

IMPORTANT:
- Be objective and data-driven
- Consider both short-term and medium-term implications
- Weight major news more heavily than minor updates
- Consider the source credibility

"""
        if context:
            base_prompt += f"\nCONTEXT: {context}\n"

        base_prompt += f"\nTEXT TO ANALYZE:\n{text}\n\nRespond with valid JSON only."

        return base_prompt

    async def _call_provider(self, provider: AIProvider, prompt: str) -> Dict:
        """Call a specific provider and return parsed result"""
        config = self.providers[provider]
        start_time = datetime.utcnow()

        try:
            if provider == AIProvider.OPENAI:
                result = await self._call_openai(config, prompt)
            elif provider == AIProvider.ANTHROPIC:
                result = await self._call_anthropic(config, prompt)
            else:
                return {'success': False, 'error': f'Unknown provider: {provider}'}

            # Calculate cost
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            cost = self._calculate_cost(
                config.model,
                result.get('input_tokens', 0),
                result.get('output_tokens', 0)
            )

            # Track metrics
            metrics = APICallMetrics(
                provider=provider.value,
                model=config.model.model_id,
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0),
                cost_usd=cost,
                latency_ms=latency,
                success=result.get('success', False),
                error=result.get('error')
            )
            self._record_metrics(metrics)

            # Update daily spending
            self.daily_spent_usd += cost

            # Add metadata to result
            result['provider'] = provider.value
            result['model'] = config.model.model_id
            result['cost_usd'] = cost
            result['latency_ms'] = latency
            result['cached'] = False

            # Log the call
            await self._log_api_call(metrics, prompt, result.get('raw_response', ''))

            return result

        except Exception as e:
            logger.error(f"Error calling {provider.value}: {e}")
            return {'success': False, 'error': str(e)}

    async def _call_openai(self, config: ProviderConfig, prompt: str) -> Dict:
        """Call OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.model.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "response_format": {"type": "json_object"}
        }

        async with self._session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenAI error: {error_text}")
                return {'success': False, 'error': f'API error: {response.status}'}

            data = await response.json()

            # Extract tokens
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

            # Parse response
            content = data['choices'][0]['message']['content']
            parsed = self._parse_sentiment_response(content)

            parsed['input_tokens'] = input_tokens
            parsed['output_tokens'] = output_tokens
            parsed['raw_response'] = content

            return parsed

    async def _call_anthropic(self, config: ProviderConfig, prompt: str) -> Dict:
        """Call Anthropic Claude API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.model.model_id,
            "max_tokens": config.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        async with self._session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic error: {error_text}")
                return {'success': False, 'error': f'API error: {response.status}'}

            data = await response.json()

            # Extract tokens
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

            # Parse response
            content = data['content'][0]['text']
            parsed = self._parse_sentiment_response(content)

            parsed['input_tokens'] = input_tokens
            parsed['output_tokens'] = output_tokens
            parsed['raw_response'] = content

            return parsed

    def _parse_sentiment_response(self, content: str) -> Dict:
        """Parse the LLM response into structured sentiment data"""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])

            data = json.loads(content)

            # Validate and extract fields
            sentiment = float(data.get('sentiment', 0))
            confidence = float(data.get('confidence', 0))
            reasoning = str(data.get('reasoning', ''))

            # Clamp values to valid ranges
            sentiment = max(-1.0, min(1.0, sentiment))
            confidence = max(0.0, min(1.0, confidence))

            return {
                'success': True,
                'sentiment': sentiment,
                'confidence': confidence,
                'reasoning': reasoning
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")

            # Try to extract sentiment from text
            content_lower = content.lower()
            if 'bullish' in content_lower or 'positive' in content_lower:
                return {
                    'success': True,
                    'sentiment': 0.5,
                    'confidence': 0.3,
                    'reasoning': 'Extracted bullish sentiment from unstructured response'
                }
            elif 'bearish' in content_lower or 'negative' in content_lower:
                return {
                    'success': True,
                    'sentiment': -0.5,
                    'confidence': 0.3,
                    'reasoning': 'Extracted bearish sentiment from unstructured response'
                }

            return {
                'success': False,
                'sentiment': 0.0,
                'confidence': 0.0,
                'reasoning': 'Failed to parse response',
                'error': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'sentiment': 0.0,
                'confidence': 0.0,
                'reasoning': f'Parse error: {e}',
                'error': str(e)
            }

    def _calculate_cost(self, model: AIModel, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of an API call"""
        input_cost = (input_tokens / 1000) * model.input_cost
        output_cost = (output_tokens / 1000) * model.output_cost
        return round(input_cost + output_cost, 6)

    def _record_metrics(self, metrics: APICallMetrics):
        """Record API call metrics"""
        self.call_history.append(metrics)

        # Trim history if too long
        if len(self.call_history) > self._max_history:
            self.call_history = self.call_history[-self._max_history:]

    async def _log_api_call(self, metrics: APICallMetrics, prompt: str, response: str):
        """Log API call to database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_analysis_logs (
                        provider, model, prompt_preview, response_preview,
                        input_tokens, output_tokens, cost_usd, latency_ms,
                        success, error, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    metrics.provider,
                    metrics.model,
                    prompt[:500],  # Preview only
                    response[:500],
                    metrics.input_tokens,
                    metrics.output_tokens,
                    metrics.cost_usd,
                    metrics.latency_ms,
                    metrics.success,
                    metrics.error,
                    metrics.timestamp
                )
        except Exception as e:
            logger.debug(f"Could not log API call: {e}")

    def get_stats(self) -> Dict:
        """Get provider statistics"""
        stats = {
            'providers': {},
            'daily_spent_usd': self.daily_spent_usd,
            'daily_budget_usd': self.daily_budget_usd,
            'budget_remaining': self.daily_budget_usd - self.daily_spent_usd,
            'total_calls': len(self.call_history),
            'cache_size': len(self.cache.cache)
        }

        for provider, config in self.providers.items():
            provider_calls = [m for m in self.call_history if m.provider == provider.value]
            stats['providers'][provider.value] = {
                'enabled': config.enabled,
                'model': config.model.model_id,
                'calls': len(provider_calls),
                'total_cost': sum(m.cost_usd for m in provider_calls),
                'avg_latency_ms': sum(m.latency_ms for m in provider_calls) / len(provider_calls) if provider_calls else 0,
                'success_rate': sum(1 for m in provider_calls if m.success) / len(provider_calls) if provider_calls else 0
            }

        return stats

    async def close(self):
        """Close the provider manager"""
        if self._session:
            await self._session.close()
            self._session = None

        self.cache.clear()
        logger.info("🛑 AI Provider Manager closed")
