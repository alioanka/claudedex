"""
Decision Maker - ML-based trading decision engine
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from decimal import Decimal

@dataclass
class TradingDecision:
    """Represents a trading decision"""
    should_trade: bool
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    strategy: str
    position_size: float
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    expected_return: float
    risk_reward_ratio: float
    reasoning: List[str]
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RiskScore:
    """Risk score components"""
    overall_risk: float
    liquidity_risk: float
    contract_risk: float
    developer_risk: float
    holder_risk: float
    total_risk: float = field(init=False)
    
    def __post_init__(self):
        self.total_risk = self.overall_risk

@dataclass
class TradingOpportunity:
    """Represents a trading opportunity for evaluation"""
    token_address: str
    chain: str
    price: float
    volume: float
    liquidity: float
    risk_score: RiskScore
    ml_confidence: float
    pump_probability: float
    rug_probability: float
    expected_return: float
    signals: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
class StrategyType(Enum):
    """Available trading strategies"""
    SCALPING = "scalping"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    GRID = "grid"
    DCA = "dca"  # Dollar Cost Averaging
    AI_HYBRID = "ai_hybrid"

class DecisionMaker:
    """Advanced decision-making engine"""
    
    def __init__(self, config: Dict):
        """
        Initialize decision maker
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Decision thresholds
        self.min_confidence = config.get('min_confidence', 0.65)
        self.min_risk_reward = config.get('min_risk_reward', 2.0)
        self.max_risk_score = config.get('max_risk_score', 0.7)
        
        # Strategy weights
        self.strategy_weights = {
            StrategyType.SCALPING: 0.15,
            StrategyType.MOMENTUM: 0.25,
            StrategyType.MEAN_REVERSION: 0.15,
            StrategyType.BREAKOUT: 0.20,
            StrategyType.SWING: 0.10,
            StrategyType.AI_HYBRID: 0.15
        }
        
        # Market condition classifiers
        self.market_conditions = {
            'bull': False,
            'bear': False,
            'sideways': False,
            'volatile': False,
            'stable': False
        }
        
        # Decision history for learning
        self.decision_history = []
        self.performance_tracker = {}
        
    async def make_decision(self, analysis: Dict) -> TradingDecision:
        """
        Make a comprehensive trading decision
        
        Args:
            analysis: All analysis data including risk, ML predictions, patterns, etc.
            
        Returns:
            TradingDecision object
        """
        try:
            # Extract components
            risk_score = analysis.get('risk_score')
            ml_predictions = analysis.get('ml_predictions', {})
            patterns = analysis.get('patterns', [])
            sentiment = analysis.get('sentiment', {})
            market_conditions = analysis.get('market_conditions', {})
            liquidity = analysis.get('liquidity', {})
            
            # Update market conditions
            await self._classify_market_conditions(market_conditions)
            
            # Parallel strategy evaluation
            strategy_scores = await self._evaluate_strategies(analysis)
            
            # Select best strategy
            best_strategy = self._select_best_strategy(strategy_scores)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(
                risk_score, ml_predictions, patterns, sentiment
            )
            
            # Check if we should trade
            should_trade = await self._should_trade(
                confidence, risk_score, ml_predictions, liquidity
            )
            
            if not should_trade:
                return TradingDecision(
                    should_trade=False,
                    action='hold',
                    confidence=confidence,
                    strategy='none',
                    position_size=0,
                    entry_price=0,
                    stop_loss=0,
                    take_profits=[],
                    expected_return=0,
                    risk_reward_ratio=0,
                    reasoning=['Conditions not met for trading']
                )
            
            # Calculate position parameters
            position_params = await self._calculate_position_parameters(
                analysis, best_strategy, confidence
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                analysis, best_strategy, confidence
            )
            
            decision = TradingDecision(
                should_trade=True,
                action='buy',
                confidence=confidence,
                strategy=best_strategy.value,
                position_size=position_params['position_size'],
                entry_price=position_params['entry_price'],
                stop_loss=position_params['stop_loss'],
                take_profits=position_params['take_profits'],
                expected_return=position_params['expected_return'],
                risk_reward_ratio=position_params['risk_reward_ratio'],
                reasoning=reasoning,
                metadata={
                    'strategy_scores': strategy_scores,
                    'market_conditions': self.market_conditions,
                    'ml_predictions': ml_predictions
                }
            )
            
            # Store decision for learning
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            print(f"Decision making error: {e}")
            return TradingDecision(
                should_trade=False,
                action='hold',
                confidence=0,
                strategy='error',
                position_size=0,
                entry_price=0,
                stop_loss=0,
                take_profits=[],
                expected_return=0,
                risk_reward_ratio=0,
                reasoning=[f'Error in decision making: {e}']
            )
            
    async def _evaluate_strategies(self, analysis_data: Dict) -> Dict[StrategyType, float]:
        """Evaluate all strategies and return scores"""
        scores = {}
        
        # Parallel evaluation
        tasks = [
            self._evaluate_scalping(analysis_data),
            self._evaluate_momentum(analysis_data),
            self._evaluate_mean_reversion(analysis_data),
            self._evaluate_breakout(analysis_data),
            self._evaluate_swing(analysis_data),
            self._evaluate_ai_hybrid(analysis_data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        strategies = [
            StrategyType.SCALPING,
            StrategyType.MOMENTUM,
            StrategyType.MEAN_REVERSION,
            StrategyType.BREAKOUT,
            StrategyType.SWING,
            StrategyType.AI_HYBRID
        ]
        
        for strategy, result in zip(strategies, results):
            if isinstance(result, Exception):
                scores[strategy] = 0
            else:
                scores[strategy] = result
                
        return scores
        
    async def _evaluate_scalping(self, data: Dict) -> float:
        """Evaluate scalping strategy suitability"""
        score = 0.0
        
        try:
            # High liquidity required
            liquidity = data.get('liquidity', {})
            if liquidity.get('value', 0) > 1000000:  # $1M+
                score += 0.3
                
            # Low spread
            if data.get('spread', 1) < 0.005:  # <0.5%
                score += 0.2
                
            # High volume
            volume_ratio = data.get('volume_ratio', 0)
            if volume_ratio > 2:
                score += 0.2
                
            # Technical indicators
            indicators = data.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            
            # Look for quick reversals
            if 30 < rsi < 70:
                score += 0.15
                
            # Volatility check
            if indicators.get('atr', 0) > 0:
                volatility_score = min(indicators['atr'] / data.get('price', 1), 0.15)
                score += volatility_score
                
            # Market conditions
            if self.market_conditions['volatile']:
                score *= 1.2
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Scalping evaluation error: {e}")
            return 0.0
            
    async def _evaluate_momentum(self, data: Dict) -> float:
        """Evaluate momentum strategy suitability"""
        score = 0.0
        
        try:
            # Check momentum indicators
            momentum = data.get('momentum', {})
            if momentum.get('direction') == 'bullish':
                score += momentum.get('momentum_score', 0) * 0.3
                
            # Strong trend required
            trend = data.get('trend', {})
            if trend and trend.get('direction') == 'uptrend':
                score += trend.get('strength', 0) * 0.25
                
            # Volume confirmation
            volume_momentum = momentum.get('volume_momentum', 1)
            if volume_momentum > 1.5:
                score += 0.2
                
            # Pattern confirmation
            patterns = data.get('patterns', [])
            bullish_patterns = [p for p in patterns if 'bull' in p.pattern_type.value.lower()]
            if bullish_patterns:
                score += 0.15
                
            # MACD positive
            indicators = data.get('indicators', {})
            if indicators.get('macd_histogram', 0) > 0:
                score += 0.1
                
            # Market conditions boost
            if self.market_conditions['bull']:
                score *= 1.3
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Momentum evaluation error: {e}")
            return 0.0
            
    async def _evaluate_mean_reversion(self, data: Dict) -> float:
        """Evaluate mean reversion strategy suitability"""
        score = 0.0
        
        try:
            indicators = data.get('indicators', {})
            
            # Oversold/overbought conditions
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                score += 0.3
            elif rsi > 70:
                score += 0.25  # Can short
                
            # Bollinger bands
            price = data.get('price', 0)
            bb_lower = indicators.get('bb_lower', 0)
            bb_upper = indicators.get('bb_upper', 0)
            
            if bb_lower > 0 and price < bb_lower:
                score += 0.25
            elif bb_upper > 0 and price > bb_upper:
                score += 0.2
                
            # Range-bound market preferred
            trend = data.get('trend', {})
            if trend and trend.get('direction') == 'sideways':
                score += 0.2
                
            # Low ATR (consolidation)
            atr = indicators.get('atr', 0)
            if atr > 0 and atr < price * 0.02:  # Less than 2% ATR
                score += 0.15
                
            # Support/resistance levels
            sr_levels = data.get('support_resistance', {})
            if sr_levels and sr_levels.get('supports'):
                if abs(price - sr_levels['supports'][0]) / price < 0.02:
                    score += 0.15
                    
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Mean reversion evaluation error: {e}")
            return 0.0
            
    async def _evaluate_breakout(self, data: Dict) -> float:
        """Evaluate breakout strategy suitability"""
        score = 0.0
        
        try:
            # Check for breakout patterns
            patterns = data.get('patterns', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.pattern_type.value.lower()]
            if breakout_patterns:
                score += max([p.confidence for p in breakout_patterns]) * 0.3
                
            # Volume surge
            volume_ratio = data.get('volume_ratio', 1)
            if volume_ratio > 2:
                score += 0.25
                
            # Price above resistance
            sr_levels = data.get('support_resistance', {})
            price = data.get('price', 0)
            
            if sr_levels and sr_levels.get('resistances'):
                nearest_resistance = sr_levels['resistances'][0]
                if price > nearest_resistance * 0.99:
                    score += 0.2
                    
            # Volatility expansion
            indicators = data.get('indicators', {})
            bb_width = indicators.get('bb_width', 0)
            if bb_width > 0.05:  # 5% band width
                score += 0.15
                
            # ADX for trend strength
            adx = indicators.get('adx', 0)
            if adx > 25:
                score += 0.1
                
            # Triangle patterns
            triangle_patterns = [p for p in patterns if 'triangle' in p.pattern_type.value.lower()]
            if triangle_patterns:
                score += 0.15
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Breakout evaluation error: {e}")
            return 0.0
            
    async def _evaluate_swing(self, data: Dict) -> float:
        """Evaluate swing trading strategy suitability"""
        score = 0.0
        
        try:
            # Medium-term trend
            trend = data.get('trend', {})
            if trend and trend.get('duration', 0) > 10:  # Trend lasting >10 periods
                score += 0.25
                
            # Clear support/resistance
            sr_levels = data.get('support_resistance', {})
            if sr_levels and len(sr_levels.get('supports', [])) >= 2:
                score += 0.2
                
            # Pattern reliability
            patterns = data.get('patterns', [])
            reliable_patterns = [p for p in patterns if p.confidence > 0.7]
            if reliable_patterns:
                score += 0.2
                
            # Moderate volatility preferred
            indicators = data.get('indicators', {})
            atr = indicators.get('atr', 0)
            price = data.get('price', 1)
            
            if 0.02 < atr/price < 0.05:  # 2-5% ATR
                score += 0.15
                
            # MACD crossover
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if abs(macd - macd_signal) < 0.001:  # Near crossover
                score += 0.15
                
            # Volume consistency
            volume_ratio = data.get('volume_ratio', 1)
            if 0.8 < volume_ratio < 1.5:  # Stable volume
                score += 0.1
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Swing evaluation error: {e}")
            return 0.0
            
    async def _evaluate_ai_hybrid(self, data: Dict) -> float:
        """Evaluate AI hybrid strategy suitability"""
        score = 0.0
        
        try:
            # High ML confidence required
            ml_predictions = data.get('ml_predictions', {})
            ml_confidence = ml_predictions.get('confidence', 0)
            
            if ml_confidence > 0.8:
                score += 0.35
            elif ml_confidence > 0.7:
                score += 0.2
                
            # Multiple confirmations
            confirmations = 0
            
            # Pattern confirmation
            if data.get('patterns'):
                confirmations += 1
                
            # Trend confirmation
            trend = data.get('trend', {})
            if trend and trend.get('strength', 0) > 0.5:
                confirmations += 1
                
            # Momentum confirmation
            momentum = data.get('momentum', {})
            if momentum.get('momentum_score', 0) > 0.05:
                confirmations += 1
                
            # Volume confirmation
            if data.get('volume_ratio', 1) > 1.5:
                confirmations += 1
                
            score += confirmations * 0.15
            
            # Low risk preference
            risk_score = data.get('risk_score')
            if risk_score and risk_score.total_risk < 0.5:
                score += 0.15
                
            # Feature richness (more data = better AI prediction)
            if 'features' in ml_predictions and len(ml_predictions['features']) > 100:
                score += 0.1
                
            return min(score, 1.0)
            
        except Exception as e:
            print(f"AI hybrid evaluation error: {e}")
            return 0.0
            
    def _select_best_strategy(self, scores: Dict[StrategyType, float]) -> StrategyType:
        """Select the best strategy based on scores and weights"""
        weighted_scores = {}
        
        for strategy, score in scores.items():
            weighted_scores[strategy] = score * self.strategy_weights.get(strategy, 0.1)
            
        # Add randomness for exploration (10%)
        for strategy in weighted_scores:
            weighted_scores[strategy] *= np.random.uniform(0.9, 1.1)
            
        return max(weighted_scores.items(), key=lambda x: x[1])[0]
        
    async def _calculate_confidence(self, risk_score, ml_predictions, patterns, sentiment) -> float:
        """Calculate overall trading confidence"""
        confidence = 0.0
        weights_sum = 0
        
        try:
            # ML confidence (40% weight)
            if ml_predictions:
                ml_conf = ml_predictions.get('confidence', 0)
                confidence += ml_conf * 0.4
                weights_sum += 0.4
                
            # Risk score (30% weight) - inverted
            if risk_score:
                risk_conf = 1 - risk_score.total_risk
                confidence += risk_conf * 0.3
                weights_sum += 0.3
                
            # Pattern confidence (20% weight)
            if patterns:
                pattern_confs = [p.confidence for p in patterns]
                if pattern_confs:
                    avg_pattern_conf = np.mean(pattern_confs)
                    confidence += avg_pattern_conf * 0.2
                    weights_sum += 0.2
                    
            # Sentiment (10% weight)
            if sentiment:
                sentiment_score = sentiment.get('score', 0)
                # Convert sentiment to 0-1 scale (assuming -1 to 1 input)
                normalized_sentiment = (sentiment_score + 1) / 2
                confidence += normalized_sentiment * 0.1
                weights_sum += 0.1
                
            # Normalize if not all components present
            if weights_sum > 0:
                confidence = confidence / weights_sum
                
            return min(max(confidence, 0), 1)
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.0
            
    async def _should_trade(self, confidence: float, risk_score, ml_predictions, liquidity) -> bool:
        """Determine if conditions are met for trading"""
        try:
            # Basic confidence check
            if confidence < self.min_confidence:
                return False
                
            # Risk check
            if risk_score and risk_score.total_risk > self.max_risk_score:
                return False
                
            # Rug pull check
            if ml_predictions and ml_predictions.get('rug_probability', 0) > 0.3:
                return False
                
            # Liquidity check
            if liquidity and liquidity.get('value', 0) < 100000:  # Min $100k liquidity
                return False
                
            # Honeypot check
            if risk_score and risk_score.contract_risk > 0.7:
                return False
                
            return True
            
        except Exception as e:
            print(f"Trade decision error: {e}")
            return False
            
    async def _calculate_position_parameters(self, data: Dict, strategy: StrategyType, 
                                            confidence: float) -> Dict:
        """Calculate position size, stops, and targets"""
        try:
            price = data.get('price', 0)
            balance = data.get('balance', 0)
            risk_score = data.get('risk_score')
            
            # Base position size (Kelly Criterion)
            base_size = self._kelly_criterion(confidence, risk_score, balance)
            
            # Adjust for strategy
            if strategy == StrategyType.SCALPING:
                position_size = base_size * 1.5  # Larger for quick trades
                stop_loss = 0.02  # 2% stop
                take_profits = [0.01, 0.015, 0.02]  # 1%, 1.5%, 2%
                
            elif strategy == StrategyType.MOMENTUM:
                position_size = base_size
                stop_loss = 0.05  # 5% stop
                take_profits = [0.1, 0.15, 0.25]  # 10%, 15%, 25%
                
            elif strategy == StrategyType.MEAN_REVERSION:
                position_size = base_size * 0.8
                stop_loss = 0.03  # 3% stop
                # Targets at mean
                indicators = data.get('indicators', {})
                sma = indicators.get('sma_20', price)
                target_pct = abs(sma - price) / price
                take_profits = [target_pct * 0.5, target_pct, target_pct * 1.5]
                
            elif strategy == StrategyType.BREAKOUT:
                position_size = base_size * 1.2
                stop_loss = 0.04  # 4% stop
                # Use pattern targets if available
                patterns = data.get('patterns', [])
                breakout_patterns = [p for p in patterns if 'breakout' in p.pattern_type.value.lower()]
                if breakout_patterns and breakout_patterns[0].price_target:
                    target = breakout_patterns[0].price_target
                    target_pct = (target - price) / price
                    take_profits = [target_pct * 0.5, target_pct, target_pct * 1.5]
                else:
                    take_profits = [0.07, 0.12, 0.20]
                    
            elif strategy == StrategyType.SWING:
                position_size = base_size * 0.7
                stop_loss = 0.06  # 6% stop
                take_profits = [0.15, 0.25, 0.40]  # 15%, 25%, 40%
                
            else:  # AI_HYBRID or default
                position_size = base_size
                ml_predictions = data.get('ml_predictions', {})
                expected_return = ml_predictions.get('expected_return', 0.1)
                stop_loss = 0.04
                take_profits = [expected_return * 0.5, expected_return, expected_return * 1.5]
                
            # Calculate risk/reward
            avg_target = np.mean(take_profits) if take_profits else 0
            risk_reward_ratio = avg_target / stop_loss if stop_loss > 0 else 0
            
            # Ensure minimum risk/reward
            if risk_reward_ratio < self.min_risk_reward:
                # Adjust targets
                take_profits = [stop_loss * self.min_risk_reward * (i+1)/3 for i in range(3)]
                risk_reward_ratio = self.min_risk_reward
                
            return {
                'position_size': position_size,
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'expected_return': avg_target,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception as e:
            print(f"Position calculation error: {e}")
            return {
                'position_size': 0,
                'entry_price': 0,
                'stop_loss': 0.05,
                'take_profits': [0.1],
                'expected_return': 0,
                'risk_reward_ratio': 0
            }
            
    def _kelly_criterion(self, confidence: float, risk_score, balance: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Win probability
            p = confidence
            # Loss probability  
            q = 1 - p
            # Win/loss ratio (simplified)
            b = 2.5  # Average risk/reward ratio
            
            # Kelly formula: f = (p * b - q) / b
            kelly_fraction = (p * b - q) / b
            
            # Apply safety factor (25% of Kelly)
            safe_fraction = kelly_fraction * 0.25
            
            # Risk adjustment
            if risk_score:
                risk_factor = 1 - risk_score.total_risk
                safe_fraction *= risk_factor
                
            # Max position size limits
            max_position = balance * self.config.get('max_position_pct', 0.05)  # 5% max
            
            position_size = min(balance * safe_fraction, max_position)
            
            # Minimum position size
            min_position = self.config.get('min_position_size', 100)
            
            return max(position_size, min_position)
            
        except Exception as e:
            print(f"Kelly calculation error: {e}")
            return balance * 0.01  # Default 1%
            
    def _generate_reasoning(self, data: Dict, strategy: StrategyType, 
                          confidence: float) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []
        
        try:
            # Confidence reasoning
            reasoning.append(f"Overall confidence: {confidence:.1%}")
            
            # Strategy reasoning
            reasoning.append(f"Selected strategy: {strategy.value}")
            
            # ML reasoning
            ml_predictions = data.get('ml_predictions', {})
            if ml_predictions:
                pump_prob = ml_predictions.get('pump_probability', 0)
                rug_prob = ml_predictions.get('rug_probability', 0)
                reasoning.append(f"ML predictions - Pump: {pump_prob:.1%}, Rug: {rug_prob:.1%}")
                
            # Risk reasoning
            risk_score = data.get('risk_score')
            if risk_score:
                reasoning.append(f"Risk assessment: {risk_score.total_risk:.1%}")
                if risk_score.liquidity_risk > 0.5:
                    reasoning.append("⚠️ Elevated liquidity risk")
                if risk_score.developer_risk > 0.5:
                    reasoning.append("⚠️ Developer reputation concerns")
                    
            # Pattern reasoning
            patterns = data.get('patterns', [])
            if patterns:
                pattern_names = [p.pattern_type.value for p in patterns[:3]]
                reasoning.append(f"Patterns detected: {', '.join(pattern_names)}")
                
            # Trend reasoning
            trend = data.get('trend', {})
            if trend:
                reasoning.append(f"Market trend: {trend.get('direction', 'unknown')} "
                               f"(strength: {trend.get('strength', 0):.1%})")
                
            # Volume reasoning
            volume_ratio = data.get('volume_ratio', 1)
            if volume_ratio > 2:
                reasoning.append(f"📈 High volume detected ({volume_ratio:.1f}x average)")
            elif volume_ratio < 0.5:
                reasoning.append(f"📉 Low volume warning ({volume_ratio:.1f}x average)")
                
            # Market conditions
            if self.market_conditions['volatile']:
                reasoning.append("🌊 Volatile market conditions")
            if self.market_conditions['bull']:
                reasoning.append("🐂 Bullish market sentiment")
                
            return reasoning
            
        except Exception as e:
            print(f"Reasoning generation error: {e}")
            return ["Error generating reasoning"]
            
    async def _classify_market_conditions(self, market_data: Dict):
        """Classify current market conditions"""
        try:
            # Reset conditions
            for key in self.market_conditions:
                self.market_conditions[key] = False
                
            # Bull/Bear classification
            market_trend = market_data.get('trend', {})
            if market_trend.get('direction') == 'uptrend' and market_trend.get('strength', 0) > 0.5:
                self.market_conditions['bull'] = True
            elif market_trend.get('direction') == 'downtrend' and market_trend.get('strength', 0) > 0.5:
                self.market_conditions['bear'] = True
            else:
                self.market_conditions['sideways'] = True
                
            # Volatility classification
            volatility = market_data.get('volatility', {})
            if volatility.get('atr_ratio', 1) > 1.5:
                self.market_conditions['volatile'] = True
            else:
                self.market_conditions['stable'] = True
                
        except Exception as e:
            print(f"Market classification error: {e}")
            
    async def update_performance(self, decision_id: str, outcome: Dict):
        """Update performance tracking for learning"""
        try:
            if decision_id not in self.performance_tracker:
                self.performance_tracker[decision_id] = {}
                
            self.performance_tracker[decision_id].update({
                'outcome': outcome,
                'timestamp': datetime.now(),
                'profit': outcome.get('profit', 0),
                'success': outcome.get('success', False)
            })
            
            # Adjust strategy weights based on performance
            await self._adjust_strategy_weights()
            
        except Exception as e:
            print(f"Performance update error: {e}")
            
    async def _adjust_strategy_weights(self):
        """Dynamically adjust strategy weights based on performance"""
        try:
            # Calculate performance by strategy
            strategy_performance = {}
            
            for decision in self.decision_history[-100:]:  # Last 100 decisions
                strategy = StrategyType[decision.strategy.upper()]
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'wins': 0, 'losses': 0}
                    
                # Check if we have outcome for this decision
                # This would need to be tracked separately
                # For now, using placeholder logic
                
            # Update weights using exponential moving average
            # This would be implemented with actual performance tracking
            
        except Exception as e:
            print(f"Weight adjustment error: {e}")
    
    # API-required methods
    
    async def evaluate_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """Evaluate if opportunity should be taken"""
        try:
            # Check risk score
            if opportunity.risk_score.overall_risk > self.max_risk_score:
                return False
            
            # Check ML confidence
            if opportunity.ml_confidence < self.min_confidence:
                return False
            
            # Check expected return vs risk
            risk_reward = opportunity.expected_return / (opportunity.risk_score.overall_risk + 0.1)
            if risk_reward < self.min_risk_reward:
                return False
            
            # Check pump probability
            if opportunity.pump_probability < 0.4:
                return False
            
            # Check rug probability
            if opportunity.rug_probability > 0.3:
                return False
            
            return True
            
        except Exception as e:
            print(f"Opportunity evaluation error: {e}")
            return False
    
    def calculate_confidence_score(self, signals: Dict) -> float:
        """Calculate confidence score from signals"""
        try:
            confidence = 0.0
            weights = {
                'technical': 0.3,
                'ml': 0.3,
                'volume': 0.2,
                'social': 0.1,
                'fundamental': 0.1
            }
            
            for signal_type, weight in weights.items():
                if signal_type in signals:
                    signal_strength = signals[signal_type].get('strength', 0)
                    confidence += signal_strength * weight
            
            return min(confidence, 1.0)
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.0
    
    def determine_action(self, scores: Dict) -> str:
        """Determine action based on scores"""
        try:
            # Calculate composite score
            buy_score = scores.get('buy_score', 0)
            sell_score = scores.get('sell_score', 0)
            hold_score = scores.get('hold_score', 0.5)
            
            # Determine action
            if buy_score > sell_score and buy_score > hold_score:
                if buy_score > 0.7:
                    return 'strong_buy'
                else:
                    return 'buy'
            elif sell_score > buy_score and sell_score > hold_score:
                if sell_score > 0.7:
                    return 'strong_sell'
                else:
                    return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            print(f"Action determination error: {e}")
            return 'hold'
    
    async def validate_decision(self, decision: TradingDecision) -> bool:
        """Validate trading decision"""
        try:
            # Check confidence
            if decision.confidence < self.min_confidence:
                return False
            
            # Check risk/reward
            if decision.risk_reward_ratio < self.min_risk_reward:
                return False
            
            # Check position size
            if decision.position_size <= 0:
                return False
            
            # Check stop loss
            if decision.stop_loss <= 0:
                return False
            
            # Check take profits
            if not decision.take_profits or len(decision.take_profits) == 0:
                return False
            
            # Validate strategy
            valid_strategies = [s.value for s in StrategyType]
            if decision.strategy not in valid_strategies:
                return False
            
            return True
            
        except Exception as e:
            print(f"Decision validation error: {e}")
            return False