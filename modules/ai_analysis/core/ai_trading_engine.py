"""
AI Trading Engine - Enterprise-Grade Multi-Chain Trading System

This module provides:
- Multi-chain trading (ETH, Base, ARB, BSC, SOL)
- DEX and Futures integration
- AI-driven strategy generation
- Self-improvement through performance analysis
- Market condition monitoring via LLM
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import aiohttp

logger = logging.getLogger(__name__)


class Chain(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    BASE = "base"
    ARBITRUM = "arbitrum"
    BSC = "bsc"
    SOLANA = "solana"


class TradeType(Enum):
    """Trade execution types"""
    SPOT_DEX = "spot_dex"
    FUTURES_LONG = "futures_long"
    FUTURES_SHORT = "futures_short"


class SignalStrength(Enum):
    """AI signal confidence levels"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


@dataclass
class MarketCondition:
    """Current market condition analysis"""
    trend: str  # bullish, bearish, sideways
    volatility: str  # low, medium, high, extreme
    sentiment: float  # -1.0 to 1.0
    fear_greed_index: int  # 0-100
    recommended_action: str  # buy, sell, hold, reduce_exposure
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_to_buy: List[str] = field(default_factory=list)
    tokens_to_sell: List[str] = field(default_factory=list)


@dataclass
class AITradeSignal:
    """AI-generated trade signal"""
    symbol: str
    chain: Chain
    trade_type: TradeType
    direction: str  # long or short
    entry_price: float
    target_price: float
    stop_loss: float
    position_size_pct: float  # % of portfolio
    confidence: float
    strength: SignalStrength
    reasoning: str
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AIStrategy:
    """AI-generated trading strategy"""
    name: str
    description: str
    chain: Chain
    trade_type: TradeType
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_params: Dict[str, float]
    performance_score: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    is_active: bool = True


class ChainConfig:
    """Chain-specific configuration"""

    CONFIGS = {
        Chain.ETHEREUM: {
            "rpc_env": "ETHEREUM_RPC_URL",
            "dexes": ["uniswap_v2", "uniswap_v3", "sushiswap", "curve"],
            "futures_exchanges": ["binance", "bybit", "dydx"],
            "native_token": "ETH",
            "wrapped_native": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "gas_multiplier": 1.2,
            "min_trade_usd": 50,
            "explorer": "https://etherscan.io"
        },
        Chain.BASE: {
            "rpc_env": "BASE_RPC_URL",
            "dexes": ["aerodrome", "baseswap", "uniswap_v3"],
            "futures_exchanges": [],
            "native_token": "ETH",
            "wrapped_native": "0x4200000000000000000000000000000000000006",
            "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "gas_multiplier": 1.1,
            "min_trade_usd": 10,
            "explorer": "https://basescan.org"
        },
        Chain.ARBITRUM: {
            "rpc_env": "ARBITRUM_RPC_URL",
            "dexes": ["camelot", "uniswap_v3", "sushiswap", "gmx"],
            "futures_exchanges": ["gmx"],
            "native_token": "ETH",
            "wrapped_native": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            "usdc": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            "gas_multiplier": 1.1,
            "min_trade_usd": 20,
            "explorer": "https://arbiscan.io"
        },
        Chain.BSC: {
            "rpc_env": "BSC_RPC_URL",
            "dexes": ["pancakeswap_v2", "pancakeswap_v3", "biswap"],
            "futures_exchanges": ["binance"],
            "native_token": "BNB",
            "wrapped_native": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
            "usdc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
            "gas_multiplier": 1.1,
            "min_trade_usd": 10,
            "explorer": "https://bscscan.com"
        },
        Chain.SOLANA: {
            "rpc_env": "SOLANA_RPC_URL",
            "dexes": ["jupiter", "raydium", "orca"],
            "futures_exchanges": ["drift"],
            "native_token": "SOL",
            "wrapped_native": "So11111111111111111111111111111111111111112",
            "usdc": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "gas_multiplier": 1.0,
            "min_trade_usd": 5,
            "explorer": "https://solscan.io"
        }
    }

    @classmethod
    def get(cls, chain: Chain) -> Dict:
        return cls.CONFIGS.get(chain, {})


class AIMarketAnalyzer:
    """
    Uses AI to analyze market conditions and recommend actions.
    Queries LLM frequently about crypto market status.
    """

    def __init__(self, ai_provider_manager):
        self.ai_manager = ai_provider_manager
        self.cache: Dict[str, Tuple[MarketCondition, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def analyze_market(self, force_refresh: bool = False) -> MarketCondition:
        """Get comprehensive market analysis from AI"""
        cache_key = "global_market"

        if not force_refresh and cache_key in self.cache:
            cached, ts = self.cache[cache_key]
            if (datetime.now() - ts).total_seconds() < self.cache_ttl:
                return cached

        # Fetch market data
        market_data = await self._fetch_market_data()

        prompt = f"""Analyze the current cryptocurrency market conditions based on this data:

{json.dumps(market_data, indent=2)}

Provide a JSON response with these exact fields:
{{
    "trend": "bullish" | "bearish" | "sideways",
    "volatility": "low" | "medium" | "high" | "extreme",
    "sentiment": <float between -1.0 and 1.0>,
    "fear_greed_index": <int 0-100>,
    "recommended_action": "buy" | "sell" | "hold" | "reduce_exposure",
    "reasoning": "<brief explanation>",
    "confidence": <float 0-1>,
    "tokens_to_buy": ["<token1>", "<token2>", ...],
    "tokens_to_sell": ["<token1>", "<token2>", ...]
}}

Be specific about which tokens show opportunity based on current conditions."""

        try:
            result = await self.ai_manager.analyze_sentiment(
                text=prompt,
                context="Crypto market analysis for trading decisions",
                use_cache=False
            )

            # Parse AI response
            response_text = result.get('reasoning', '{}')

            # Try to extract JSON from response
            parsed = self._extract_json(response_text)

            condition = MarketCondition(
                trend=parsed.get('trend', 'sideways'),
                volatility=parsed.get('volatility', 'medium'),
                sentiment=float(parsed.get('sentiment', 0)),
                fear_greed_index=int(parsed.get('fear_greed_index', 50)),
                recommended_action=parsed.get('recommended_action', 'hold'),
                reasoning=parsed.get('reasoning', 'No analysis available'),
                confidence=float(parsed.get('confidence', 0.5)),
                tokens_to_buy=parsed.get('tokens_to_buy', []),
                tokens_to_sell=parsed.get('tokens_to_sell', [])
            )

            self.cache[cache_key] = (condition, datetime.now())
            return condition

        except Exception as e:
            logger.error(f"AI market analysis failed: {e}")
            return MarketCondition(
                trend="sideways",
                volatility="medium",
                sentiment=0.0,
                fear_greed_index=50,
                recommended_action="hold",
                reasoning=f"Analysis failed: {str(e)}",
                confidence=0.0
            )

    async def analyze_token(self, symbol: str, chain: Chain) -> Dict:
        """Get AI analysis for a specific token"""
        prompt = f"""Analyze {symbol} on {chain.value} chain for trading opportunity.

Consider:
1. Recent price action and volume
2. On-chain activity and whale movements
3. Social sentiment and news
4. Technical indicators (RSI, MACD, moving averages)
5. Fundamental factors (TVL, revenue, token utility)

Provide JSON response:
{{
    "recommendation": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "confidence": <float 0-1>,
    "entry_zone": {{"low": <price>, "high": <price>}},
    "targets": [<price1>, <price2>, <price3>],
    "stop_loss": <price>,
    "risk_reward": <float>,
    "timeframe": "short_term" | "medium_term" | "long_term",
    "reasoning": "<detailed analysis>"
}}"""

        try:
            result = await self.ai_manager.analyze_sentiment(
                text=prompt,
                context=f"Token analysis for {symbol}",
                use_cache=True
            )
            return self._extract_json(result.get('reasoning', '{}'))
        except Exception as e:
            logger.error(f"Token analysis failed for {symbol}: {e}")
            return {"recommendation": "hold", "confidence": 0, "reasoning": str(e)}

    async def _fetch_market_data(self) -> Dict:
        """Fetch current market data from various sources"""
        data = {
            "btc_price": 0,
            "eth_price": 0,
            "total_market_cap": 0,
            "btc_dominance": 0,
            "top_gainers": [],
            "top_losers": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with aiohttp.ClientSession() as session:
                # CoinGecko global data
                async with session.get(
                    "https://api.coingecko.com/api/v3/global",
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        global_data = result.get('data', {})
                        data['total_market_cap'] = global_data.get('total_market_cap', {}).get('usd', 0)
                        data['btc_dominance'] = global_data.get('market_cap_percentage', {}).get('btc', 0)

                # BTC/ETH prices
                async with session.get(
                    "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        prices = await resp.json()
                        data['btc_price'] = prices.get('bitcoin', {}).get('usd', 0)
                        data['btc_24h_change'] = prices.get('bitcoin', {}).get('usd_24h_change', 0)
                        data['eth_price'] = prices.get('ethereum', {}).get('usd', 0)
                        data['eth_24h_change'] = prices.get('ethereum', {}).get('usd_24h_change', 0)

        except Exception as e:
            logger.warning(f"Failed to fetch market data: {e}")

        return data

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from AI response text"""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            pass

        # Try to find JSON block
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return {}


class AIStrategyGenerator:
    """
    Generates and evolves trading strategies using AI.
    Implements self-improvement through performance analysis.
    """

    def __init__(self, ai_provider_manager, db_pool=None):
        self.ai_manager = ai_provider_manager
        self.db_pool = db_pool
        self.strategies: Dict[str, AIStrategy] = {}
        self.performance_history: List[Dict] = []

    async def generate_strategy(
        self,
        chain: Chain,
        trade_type: TradeType,
        market_condition: MarketCondition
    ) -> AIStrategy:
        """Generate a new trading strategy based on current conditions"""

        prompt = f"""Create a trading strategy for {chain.value} chain, {trade_type.value} trades.

Current market conditions:
- Trend: {market_condition.trend}
- Volatility: {market_condition.volatility}
- Sentiment: {market_condition.sentiment}
- Fear/Greed: {market_condition.fear_greed_index}

Generate a JSON strategy:
{{
    "name": "<unique strategy name>",
    "description": "<what this strategy does>",
    "entry_conditions": [
        "<condition 1>",
        "<condition 2>",
        ...
    ],
    "exit_conditions": [
        "<take profit condition>",
        "<stop loss condition>",
        "<time-based exit>"
    ],
    "risk_params": {{
        "max_position_pct": <0-100>,
        "stop_loss_pct": <negative number>,
        "take_profit_pct": <positive number>,
        "max_drawdown_pct": <negative number>,
        "risk_reward_min": <positive number>
    }},
    "preferred_tokens": ["<token1>", "<token2>", ...],
    "avoid_tokens": ["<token1>", ...]
}}

Make the strategy specific and actionable for current conditions."""

        try:
            result = await self.ai_manager.analyze_sentiment(
                text=prompt,
                context="Strategy generation",
                use_cache=False
            )

            parsed = self._extract_json(result.get('reasoning', '{}'))

            strategy = AIStrategy(
                name=parsed.get('name', f'AI_Strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                description=parsed.get('description', 'AI-generated strategy'),
                chain=chain,
                trade_type=trade_type,
                entry_conditions=parsed.get('entry_conditions', []),
                exit_conditions=parsed.get('exit_conditions', []),
                risk_params=parsed.get('risk_params', {
                    'max_position_pct': 5,
                    'stop_loss_pct': -3,
                    'take_profit_pct': 6,
                    'max_drawdown_pct': -10,
                    'risk_reward_min': 2.0
                })
            )

            # Store strategy
            self.strategies[strategy.name] = strategy
            await self._save_strategy(strategy)

            logger.info(f"Generated new strategy: {strategy.name}")
            return strategy

        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            # Return default strategy
            return AIStrategy(
                name=f'Default_{chain.value}_{trade_type.value}',
                description='Default fallback strategy',
                chain=chain,
                trade_type=trade_type,
                entry_conditions=['sentiment > 0.3', 'trend confirmation'],
                exit_conditions=['take profit at +5%', 'stop loss at -3%', 'max hold 24h'],
                risk_params={
                    'max_position_pct': 5,
                    'stop_loss_pct': -3,
                    'take_profit_pct': 5,
                    'max_drawdown_pct': -10,
                    'risk_reward_min': 1.5
                }
            )

    async def evolve_strategy(
        self,
        strategy: AIStrategy,
        performance_data: Dict
    ) -> AIStrategy:
        """Improve strategy based on performance data"""

        prompt = f"""Analyze and improve this trading strategy based on its performance:

Strategy: {strategy.name}
Description: {strategy.description}
Current Entry Conditions: {json.dumps(strategy.entry_conditions)}
Current Exit Conditions: {json.dumps(strategy.exit_conditions)}
Current Risk Params: {json.dumps(strategy.risk_params)}

Performance Data:
{json.dumps(performance_data, indent=2)}

Suggest improvements as JSON:
{{
    "improved_entry_conditions": [...],
    "improved_exit_conditions": [...],
    "improved_risk_params": {{...}},
    "reasoning": "<why these changes will improve performance>",
    "expected_improvement": "<percentage improvement expected>"
}}"""

        try:
            result = await self.ai_manager.analyze_sentiment(
                text=prompt,
                context="Strategy evolution",
                use_cache=False
            )

            parsed = self._extract_json(result.get('reasoning', '{}'))

            # Update strategy with improvements
            if parsed.get('improved_entry_conditions'):
                strategy.entry_conditions = parsed['improved_entry_conditions']
            if parsed.get('improved_exit_conditions'):
                strategy.exit_conditions = parsed['improved_exit_conditions']
            if parsed.get('improved_risk_params'):
                strategy.risk_params.update(parsed['improved_risk_params'])

            await self._save_strategy(strategy)
            logger.info(f"Evolved strategy {strategy.name}: {parsed.get('reasoning', 'No details')}")

            return strategy

        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}")
            return strategy

    async def _save_strategy(self, strategy: AIStrategy):
        """Persist strategy to database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_strategies (
                        name, description, chain, trade_type,
                        entry_conditions, exit_conditions, risk_params,
                        performance_score, total_trades, win_rate,
                        is_active, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (name) DO UPDATE SET
                        entry_conditions = $5,
                        exit_conditions = $6,
                        risk_params = $7,
                        performance_score = $8,
                        total_trades = $9,
                        win_rate = $10
                """,
                    strategy.name,
                    strategy.description,
                    strategy.chain.value,
                    strategy.trade_type.value,
                    json.dumps(strategy.entry_conditions),
                    json.dumps(strategy.exit_conditions),
                    json.dumps(strategy.risk_params),
                    strategy.performance_score,
                    strategy.total_trades,
                    strategy.win_rate,
                    strategy.is_active,
                    strategy.created_at
                )
        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text"""
        try:
            return json.loads(text)
        except:
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return {}


class AITradingEngine:
    """
    Main AI Trading Engine - Orchestrates all trading operations.
    Supports multi-chain DEX and Futures trading with AI-driven decisions.
    """

    def __init__(
        self,
        ai_provider_manager,
        db_pool=None,
        config: Dict = None
    ):
        self.ai_manager = ai_provider_manager
        self.db_pool = db_pool
        self.config = config or {}

        # Components
        self.market_analyzer = AIMarketAnalyzer(ai_provider_manager)
        self.strategy_generator = AIStrategyGenerator(ai_provider_manager, db_pool)

        # State
        self.active_positions: Dict[str, Dict] = {}
        self.pending_signals: List[AITradeSignal] = []
        self.daily_stats = {
            'trades': 0,
            'pnl': 0.0,
            'wins': 0,
            'losses': 0
        }

        # Configuration
        self.enabled_chains: List[Chain] = [
            Chain.ETHEREUM,
            Chain.BASE,
            Chain.ARBITRUM,
            Chain.BSC,
            Chain.SOLANA
        ]
        self.max_positions = config.get('max_positions', 5)
        self.max_position_size_pct = config.get('max_position_size_pct', 10)
        self.analysis_interval = config.get('analysis_interval_minutes', 15)
        self.dry_run = config.get('dry_run', True)

        # Performance tracking
        self.performance_tracker = AIPerformanceTracker(db_pool)

        logger.info(f"AI Trading Engine initialized - Chains: {[c.value for c in self.enabled_chains]}")

    async def run_analysis_cycle(self) -> List[AITradeSignal]:
        """Run a complete analysis cycle and generate trade signals"""
        signals = []

        try:
            # 1. Get market conditions
            market = await self.market_analyzer.analyze_market()
            logger.info(f"Market: {market.trend}, {market.volatility} volatility, sentiment={market.sentiment:.2f}")

            # 2. Check if we should trade
            if market.recommended_action == 'reduce_exposure' and len(self.active_positions) > 0:
                logger.warning("AI recommends reducing exposure - checking exits")
                await self._check_position_exits(market)

            # 3. Generate signals for recommended tokens
            if market.tokens_to_buy and market.recommended_action in ['buy', 'strong_buy']:
                for token in market.tokens_to_buy[:3]:  # Limit to top 3
                    signal = await self._generate_buy_signal(token, market)
                    if signal:
                        signals.append(signal)

            # 4. Check existing strategies for opportunities
            for strategy in self.strategy_generator.strategies.values():
                if strategy.is_active:
                    strategy_signals = await self._evaluate_strategy(strategy, market)
                    signals.extend(strategy_signals)

            # 5. Filter and prioritize signals
            signals = self._prioritize_signals(signals)

            # 6. Execute top signals (in DRY_RUN or LIVE mode)
            if signals and len(self.active_positions) < self.max_positions:
                await self._execute_signals(signals[:self.max_positions - len(self.active_positions)])

            return signals

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
            return []

    async def _generate_buy_signal(
        self,
        token: str,
        market: MarketCondition
    ) -> Optional[AITradeSignal]:
        """Generate a buy signal for a specific token"""

        # Determine best chain for this token
        chain = await self._determine_chain(token)
        if not chain:
            return None

        # Get token analysis
        analysis = await self.market_analyzer.analyze_token(token, chain)

        if analysis.get('recommendation') not in ['buy', 'strong_buy']:
            return None

        confidence = float(analysis.get('confidence', 0))
        if confidence < 0.6:
            return None

        # Calculate position size based on confidence
        base_size = self.max_position_size_pct
        adjusted_size = base_size * confidence

        return AITradeSignal(
            symbol=token,
            chain=chain,
            trade_type=TradeType.SPOT_DEX,
            direction='long',
            entry_price=analysis.get('entry_zone', {}).get('high', 0),
            target_price=analysis.get('targets', [0])[0] if analysis.get('targets') else 0,
            stop_loss=analysis.get('stop_loss', 0),
            position_size_pct=adjusted_size,
            confidence=confidence,
            strength=self._get_signal_strength(confidence),
            reasoning=analysis.get('reasoning', ''),
            strategy_name='AI_Market_Analysis',
            metadata={'analysis': analysis, 'market': market.__dict__}
        )

    async def _determine_chain(self, token: str) -> Optional[Chain]:
        """Determine the best chain to trade a token on"""
        token_upper = token.upper()

        # Known token mappings
        if token_upper in ['SOL', 'BONK', 'JTO', 'JUP', 'WIF']:
            return Chain.SOLANA
        elif token_upper in ['BNB', 'CAKE']:
            return Chain.BSC
        elif token_upper in ['ARB', 'GMX']:
            return Chain.ARBITRUM
        else:
            # Default to Ethereum for major tokens
            return Chain.ETHEREUM

    async def _evaluate_strategy(
        self,
        strategy: AIStrategy,
        market: MarketCondition
    ) -> List[AITradeSignal]:
        """Evaluate a strategy and generate signals if conditions match"""
        signals = []

        # Check if market conditions align with strategy
        if strategy.chain not in self.enabled_chains:
            return signals

        # Use AI to evaluate if strategy conditions are met
        prompt = f"""Evaluate if this trading strategy should generate signals now:

Strategy: {strategy.name}
Entry Conditions: {json.dumps(strategy.entry_conditions)}
Market Conditions:
- Trend: {market.trend}
- Volatility: {market.volatility}
- Sentiment: {market.sentiment}

Respond with JSON:
{{
    "should_trade": true/false,
    "tokens": ["<token1>", "<token2>", ...],
    "confidence": <0-1>,
    "reasoning": "<explanation>"
}}"""

        try:
            result = await self.ai_manager.analyze_sentiment(
                text=prompt,
                context=f"Strategy evaluation: {strategy.name}",
                use_cache=True
            )

            parsed = self._extract_json(result.get('reasoning', '{}'))

            if parsed.get('should_trade') and parsed.get('tokens'):
                for token in parsed['tokens'][:2]:
                    signal = AITradeSignal(
                        symbol=token,
                        chain=strategy.chain,
                        trade_type=strategy.trade_type,
                        direction='long' if strategy.trade_type != TradeType.FUTURES_SHORT else 'short',
                        entry_price=0,  # Will be filled on execution
                        target_price=0,
                        stop_loss=0,
                        position_size_pct=strategy.risk_params.get('max_position_pct', 5),
                        confidence=float(parsed.get('confidence', 0.5)),
                        strength=self._get_signal_strength(parsed.get('confidence', 0.5)),
                        reasoning=parsed.get('reasoning', ''),
                        strategy_name=strategy.name
                    )
                    signals.append(signal)

        except Exception as e:
            logger.error(f"Strategy evaluation failed for {strategy.name}: {e}")

        return signals

    def _prioritize_signals(self, signals: List[AITradeSignal]) -> List[AITradeSignal]:
        """Sort and filter signals by priority"""
        # Remove duplicates
        seen = set()
        unique = []
        for s in signals:
            key = f"{s.symbol}_{s.chain.value}"
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Sort by confidence * strength
        unique.sort(key=lambda s: s.confidence * s.strength.value, reverse=True)

        return unique

    async def _execute_signals(self, signals: List[AITradeSignal]):
        """Execute trade signals"""
        for signal in signals:
            try:
                if self.dry_run:
                    await self._execute_dry_run(signal)
                else:
                    await self._execute_live(signal)
            except Exception as e:
                logger.error(f"Failed to execute signal for {signal.symbol}: {e}")

    async def _execute_dry_run(self, signal: AITradeSignal):
        """Simulate trade execution"""
        position_id = f"sim_{signal.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get current price (simulated or from API)
        price = await self._get_current_price(signal.symbol, signal.chain)

        position = {
            'position_id': position_id,
            'symbol': signal.symbol,
            'chain': signal.chain.value,
            'trade_type': signal.trade_type.value,
            'direction': signal.direction,
            'entry_price': price,
            'current_price': price,
            'size_pct': signal.position_size_pct,
            'stop_loss': price * (1 - 0.03),  # 3% SL
            'take_profit': price * (1 + 0.06),  # 6% TP
            'pnl': 0,
            'pnl_pct': 0,
            'strategy': signal.strategy_name,
            'confidence': signal.confidence,
            'entry_time': datetime.now().isoformat(),
            'is_simulated': True
        }

        self.active_positions[position_id] = position
        self.daily_stats['trades'] += 1

        logger.info(f"[DRY_RUN] Opened {signal.direction.upper()} {signal.symbol} @ ${price:.4f} "
                   f"(confidence: {signal.confidence:.2f}, strategy: {signal.strategy_name})")

        # Save to database
        await self._save_trade(position, 'open')

    async def _execute_live(self, signal: AITradeSignal):
        """Execute live trade - integrates with DEX/Exchange APIs"""
        # This would integrate with the actual trading infrastructure
        # For now, log the intent
        logger.info(f"[LIVE] Would execute {signal.direction.upper()} {signal.symbol} on {signal.chain.value}")

        # TODO: Integrate with:
        # - DEX module for spot trades
        # - Futures module for perp trades
        # - Solana module for SOL chain trades

    async def _get_current_price(self, symbol: str, chain: Chain) -> float:
        """Fetch current price for a symbol"""
        try:
            async with aiohttp.ClientSession() as session:
                # Use CoinGecko for price
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get(symbol.lower(), {}).get('usd', 100)
        except:
            pass
        return 100.0  # Fallback

    async def _check_position_exits(self, market: MarketCondition):
        """Check and close positions based on market conditions"""
        for pos_id, pos in list(self.active_positions.items()):
            # Update current price
            price = await self._get_current_price(pos['symbol'], Chain(pos['chain']))
            pos['current_price'] = price

            # Calculate P&L
            if pos['direction'] == 'long':
                pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            else:
                pnl_pct = (pos['entry_price'] - price) / pos['entry_price'] * 100

            pos['pnl_pct'] = pnl_pct

            # Check exit conditions
            should_close = False
            close_reason = ""

            if pnl_pct >= 6:  # Take profit
                should_close = True
                close_reason = "take_profit"
            elif pnl_pct <= -3:  # Stop loss
                should_close = True
                close_reason = "stop_loss"
            elif market.recommended_action == 'reduce_exposure' and pnl_pct > 0:
                should_close = True
                close_reason = "ai_recommendation"

            if should_close:
                await self._close_position(pos_id, close_reason)

    async def _close_position(self, position_id: str, reason: str):
        """Close a position"""
        if position_id not in self.active_positions:
            return

        pos = self.active_positions.pop(position_id)

        # Update stats
        self.daily_stats['pnl'] += pos.get('pnl_pct', 0)
        if pos.get('pnl_pct', 0) > 0:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1

        logger.info(f"Closed {pos['symbol']}: {pos['pnl_pct']:.2f}% ({reason})")

        # Save to database
        pos['exit_time'] = datetime.now().isoformat()
        pos['exit_reason'] = reason
        await self._save_trade(pos, 'closed')

    async def _save_trade(self, trade: Dict, status: str):
        """Save trade to database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_trades (
                        trade_id, token_symbol, chain, trade_type, side,
                        entry_price, exit_price, amount, sentiment_score,
                        confidence_score, ai_provider, status, is_simulated,
                        entry_timestamp, exit_timestamp, profit_loss, profit_loss_pct,
                        exit_reason, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    ON CONFLICT (trade_id) DO UPDATE SET
                        exit_price = EXCLUDED.exit_price,
                        profit_loss_pct = EXCLUDED.profit_loss_pct,
                        exit_reason = EXCLUDED.exit_reason,
                        exit_timestamp = EXCLUDED.exit_timestamp,
                        status = EXCLUDED.status
                """,
                    trade['position_id'],
                    trade['symbol'],
                    trade['chain'],
                    trade['trade_type'],
                    trade['direction'],
                    trade['entry_price'],
                    trade.get('current_price', trade['entry_price']),
                    trade['size_pct'],
                    0.0,  # sentiment_score
                    trade['confidence'],
                    'ai_trading_engine',
                    status,
                    trade.get('is_simulated', True),
                    trade.get('entry_time'),
                    trade.get('exit_time'),
                    0.0,  # profit_loss USD
                    trade.get('pnl_pct', 0),
                    trade.get('exit_reason'),
                    json.dumps(trade)
                )
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def _get_signal_strength(self, confidence: float) -> SignalStrength:
        """Convert confidence to signal strength"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.75:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        elif confidence >= 0.4:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text"""
        try:
            return json.loads(text)
        except:
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return {}

    async def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'enabled_chains': [c.value for c in self.enabled_chains],
            'active_positions': len(self.active_positions),
            'positions': list(self.active_positions.values()),
            'strategies_count': len(self.strategy_generator.strategies),
            'daily_stats': self.daily_stats,
            'dry_run': self.dry_run,
            'analysis_interval': self.analysis_interval
        }


class AIPerformanceTracker:
    """
    Tracks AI trading performance and provides improvement recommendations.
    Implements self-learning through performance analysis.
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self.metrics_cache: Dict = {}

    async def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days"""
        if not self.db_pool:
            return {}

        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE profit_loss_pct > 0) as wins,
                        COUNT(*) FILTER (WHERE profit_loss_pct <= 0) as losses,
                        COALESCE(SUM(profit_loss_pct), 0) as total_pnl_pct,
                        COALESCE(AVG(profit_loss_pct), 0) as avg_pnl_pct,
                        COALESCE(MAX(profit_loss_pct), 0) as best_trade_pct,
                        COALESCE(MIN(profit_loss_pct), 0) as worst_trade_pct
                    FROM ai_trades
                    WHERE entry_timestamp >= NOW() - INTERVAL '%s days'
                    AND status = 'closed'
                """ % days)

                if result:
                    total = result['total_trades'] or 0
                    wins = result['wins'] or 0

                    return {
                        'total_trades': total,
                        'wins': wins,
                        'losses': result['losses'] or 0,
                        'win_rate': (wins / total * 100) if total > 0 else 0,
                        'total_pnl_pct': float(result['total_pnl_pct']),
                        'avg_pnl_pct': float(result['avg_pnl_pct']),
                        'best_trade_pct': float(result['best_trade_pct']),
                        'worst_trade_pct': float(result['worst_trade_pct'])
                    }
        except Exception as e:
            logger.error(f"Performance query failed: {e}")

        return {}

    async def analyze_and_suggest_improvements(self, ai_manager) -> Dict:
        """Use AI to analyze performance and suggest improvements"""
        perf = await self.get_performance_summary()

        if not perf or perf.get('total_trades', 0) < 5:
            return {'message': 'Insufficient data for analysis'}

        prompt = f"""Analyze this AI trading bot performance and suggest specific improvements:

Performance Data (Last 30 Days):
{json.dumps(perf, indent=2)}

Provide actionable recommendations as JSON:
{{
    "overall_assessment": "excellent" | "good" | "needs_improvement" | "poor",
    "key_issues": ["<issue 1>", "<issue 2>", ...],
    "recommendations": [
        {{
            "area": "<risk_management | entry_timing | exit_timing | position_sizing | strategy_selection>",
            "current_state": "<what's wrong>",
            "suggestion": "<specific improvement>",
            "expected_impact": "<percentage improvement expected>"
        }},
        ...
    ],
    "parameter_adjustments": {{
        "stop_loss_pct": <recommended value>,
        "take_profit_pct": <recommended value>,
        "confidence_threshold": <recommended value>,
        "position_size_pct": <recommended value>
    }},
    "reasoning": "<summary of analysis>"
}}"""

        try:
            result = await ai_manager.analyze_sentiment(
                text=prompt,
                context="Trading performance analysis",
                use_cache=False
            )

            return self._extract_json(result.get('reasoning', '{}'))
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}

    def _extract_json(self, text: str) -> Dict:
        try:
            return json.loads(text)
        except:
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return {}
