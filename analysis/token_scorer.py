# analysis/token_scorer.py

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

from ..core.event_bus import EventBus
from ..data.storage.database import DatabaseManager
from ..data.storage.cache import CacheManager
from ..ml.models.ensemble_model import EnsembleModel
from .rug_detector import RugDetector
from .liquidity_monitor import LiquidityMonitor
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class TokenScore:
    """Comprehensive token score"""
    token: str
    chain: str
    composite_score: float  # 0-100
    category_scores: Dict[str, float]
    risk_score: float  # 0-100, lower is better
    opportunity_score: float  # 0-100, higher is better
    confidence: float  # 0-1
    grade: str  # 'A+', 'A', 'B', 'C', 'D', 'F'
    recommendation: str
    strengths: List[str]
    weaknesses: List[str]
    timestamp: datetime

@dataclass
class ScoringWeights:
    """Weights for different scoring factors"""
    liquidity: float = 0.20
    volume: float = 0.15
    holder_distribution: float = 0.15
    developer_activity: float = 0.10
    contract_safety: float = 0.15
    price_action: float = 0.10
    social_sentiment: float = 0.05
    market_correlation: float = 0.05
    innovation: float = 0.05

class TokenScorer:
    """Multi-factor token scoring and ranking system"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_manager: CacheManager,
        event_bus: EventBus,
        rug_detector: RugDetector,
        liquidity_monitor: LiquidityMonitor,
        market_analyzer: MarketAnalyzer,
        ml_model: Optional[EnsembleModel],
        config: Dict[str, Any]
    ):
        self.db = db_manager
        self.cache = cache_manager
        self.event_bus = event_bus
        self.rug_detector = rug_detector
        self.liquidity_monitor = liquidity_monitor
        self.market_analyzer = market_analyzer
        self.ml_model = ml_model
        self.config = config
        
        # Scoring weights (customizable)
        self.weights = ScoringWeights(**config.get('scoring_weights', {}))
        
        # Scoring thresholds
        self.thresholds = {
            'A+': 90,
            'A': 80,
            'B': 70,
            'C': 60,
            'D': 50,
            'F': 0
        }
        
        # Risk factors
        self.risk_factors = {
            'high_concentration': 0.3,  # High holder concentration
            'low_liquidity': 0.25,
            'contract_issues': 0.2,
            'suspicious_activity': 0.15,
            'high_volatility': 0.1
        }
        
        # Cache for scores
        self._score_cache: Dict[str, TokenScore] = {}
        self._cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
    async def calculate_composite_score(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> TokenScore:
        """Calculate comprehensive token score"""
        try:
            # Check cache
            cache_key = f"token_score:{chain}:{token}"
            cached = await self.cache.get(cache_key)
            
            if cached and self._is_cache_valid(cached.get('timestamp')):
                return TokenScore(**cached)
                
            # Gather all scoring data
            scoring_data = await self._gather_scoring_data(token, chain)
            
            # Calculate individual scores
            category_scores = await self._calculate_category_scores(scoring_data)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(scoring_data)
            
            # Calculate opportunity score
            opportunity_score = await self._calculate_opportunity_score(scoring_data)
            
            # Calculate composite score
            composite_score = await self._calculate_weighted_score(category_scores)
            
            # Adjust for risk
            adjusted_score = self._adjust_score_for_risk(composite_score, risk_score)
            
            # Determine grade
            grade = self._determine_grade(adjusted_score)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                adjusted_score, risk_score, opportunity_score, scoring_data
            )
            
            # Identify strengths and weaknesses
            strengths = self._identify_strengths(category_scores, scoring_data)
            weaknesses = self._identify_weaknesses(category_scores, scoring_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(scoring_data)
            
            # Create score object
            token_score = TokenScore(
                token=token,
                chain=chain,
                composite_score=adjusted_score,
                category_scores=category_scores,
                risk_score=risk_score,
                opportunity_score=opportunity_score,
                confidence=confidence,
                grade=grade,
                recommendation=recommendation,
                strengths=strengths,
                weaknesses=weaknesses,
                timestamp=datetime.utcnow()
            )
            
            # Cache and emit
            await self.cache.set(cache_key, token_score.__dict__, ttl=self._cache_ttl)
            await self.event_bus.emit('token.scored', token_score.__dict__)
            
            # Store in database
            await self.db.save_token_score({
                'token': token,
                'chain': chain,
                'composite_score': adjusted_score,
                'risk_score': risk_score,
                'opportunity_score': opportunity_score,
                'grade': grade,
                'category_scores': category_scores,
                'timestamp': datetime.utcnow()
            })
            
            return token_score
            
        except Exception as e:
            logger.error(f"Error calculating composite score for {token}: {e}")
            return self._get_default_score(token, chain)
            
    async def rank_tokens(
        self,
        tokens: List[str],
        chain: str = 'ethereum',
        criteria: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Rank multiple tokens based on scores"""
        try:
            # Calculate scores for all tokens
            scores = []
            
            for token in tokens:
                try:
                    score = await self.calculate_composite_score(token, chain)
                    scores.append({
                        'token': token,
                        'score': score,
                        'rank_value': self._get_rank_value(score, criteria)
                    })
                except Exception as e:
                    logger.error(f"Error scoring token {token}: {e}")
                    continue
                    
            # Sort by ranking criteria
            if criteria == 'risk':
                scores.sort(key=lambda x: x['score'].risk_score)
            elif criteria == 'opportunity':
                scores.sort(key=lambda x: x['score'].opportunity_score, reverse=True)
            elif criteria == 'liquidity':
                scores.sort(key=lambda x: x['score'].category_scores.get('liquidity', 0), reverse=True)
            else:  # Default to composite score
                scores.sort(key=lambda x: x['score'].composite_score, reverse=True)
                
            # Add rank numbers
            ranked = []
            for i, item in enumerate(scores, 1):
                ranked.append({
                    'rank': i,
                    'token': item['token'],
                    'composite_score': item['score'].composite_score,
                    'risk_score': item['score'].risk_score,
                    'opportunity_score': item['score'].opportunity_score,
                    'grade': item['score'].grade,
                    'recommendation': item['score'].recommendation,
                    'confidence': item['score'].confidence
                })
                
            return ranked
            
        except Exception as e:
            logger.error(f"Error ranking tokens: {e}")
            return []
            
    async def compare_tokens(
        self,
        token1: str,
        token2: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Detailed comparison between two tokens"""
        try:
            # Get scores for both tokens
            score1 = await self.calculate_composite_score(token1, chain)
            score2 = await self.calculate_composite_score(token2, chain)
            
            # Compare categories
            category_comparison = {}
            for category in score1.category_scores.keys():
                diff = score1.category_scores[category] - score2.category_scores.get(category, 0)
                category_comparison[category] = {
                    'token1': score1.category_scores[category],
                    'token2': score2.category_scores.get(category, 0),
                    'difference': diff,
                    'winner': token1 if diff > 0 else token2
                }
                
            # Overall comparison
            comparison = {
                'tokens': {
                    'token1': token1,
                    'token2': token2
                },
                'composite_scores': {
                    'token1': score1.composite_score,
                    'token2': score2.composite_score,
                    'difference': score1.composite_score - score2.composite_score
                },
                'risk_scores': {
                    'token1': score1.risk_score,
                    'token2': score2.risk_score,
                    'lower_risk': token1 if score1.risk_score < score2.risk_score else token2
                },
                'opportunity_scores': {
                    'token1': score1.opportunity_score,
                    'token2': score2.opportunity_score,
                    'higher_opportunity': token1 if score1.opportunity_score > score2.opportunity_score else token2
                },
                'grades': {
                    'token1': score1.grade,
                    'token2': score2.grade
                },
                'category_comparison': category_comparison,
                'overall_winner': token1 if score1.composite_score > score2.composite_score else token2,
                'recommendation': self._generate_comparison_recommendation(score1, score2)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing tokens {token1} and {token2}: {e}")
            return {}
            
    async def get_top_opportunities(
        self,
        chain: str = 'ethereum',
        limit: int = 10,
        min_score: float = 70
    ) -> List[Dict[str, Any]]:
        """Get top scoring tokens as opportunities"""
        try:
            # Get recently scored tokens from database
            recent_scores = await self.db.get_recent_token_scores(
                chain=chain,
                hours=24,
                min_score=min_score
            )
            
            # Filter and sort
            opportunities = []
            
            for score_data in recent_scores:
                if score_data['composite_score'] >= min_score:
                    # Get full token data
                    token_data = await self._get_token_data(
                        score_data['token'],
                        chain
                    )
                    
                    opportunities.append({
                        'token': score_data['token'],
                        'score': score_data['composite_score'],
                        'risk': score_data['risk_score'],
                        'opportunity': score_data['opportunity_score'],
                        'grade': score_data['grade'],
                        'price': token_data.get('price', 0),
                        'volume_24h': token_data.get('volume_24h', 0),
                        'liquidity': token_data.get('liquidity', 0),
                        'price_change_24h': token_data.get('price_change_24h', 0),
                        'recommendation': score_data.get('recommendation', ''),
                        'timestamp': score_data['timestamp']
                    })
                    
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity'], reverse=True)
            
            return opportunities[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top opportunities: {e}")
            return []
            
    # Private helper methods
    
    async def _gather_scoring_data(self, token: str, chain: str) -> Dict[str, Any]:
        """Gather all data needed for scoring"""
        data = {}
        
        # Get basic token data
        data['token_info'] = await self._get_token_data(token, chain)
        
        # Get liquidity data
        data['liquidity'] = await self.liquidity_monitor.monitor_liquidity_changes(token, chain)
        
        # Get rug detection analysis
        data['rug_analysis'] = await self.rug_detector.comprehensive_analysis(token, chain)
        
        # Get market analysis
        data['market'] = await self.market_analyzer.identify_trends(token, chain)
        
        # Get holder data
        data['holders'] = await self._get_holder_distribution(token, chain)
        
        # Get volume data
        data['volume'] = await self.market_analyzer.analyze_volume_patterns(token, chain)
        
        # Get social data if available
        data['social'] = await self._get_social_metrics(token)
        
        # Get developer activity
        data['developer'] = await self._get_developer_activity(token, chain)
        
        return data
        
    async def _calculate_category_scores(self, data: Dict) -> Dict[str, float]:
        """Calculate scores for each category"""
        scores = {}
        
        # Liquidity score (0-100)
        liquidity_usd = data.get('liquidity', {}).get('current_liquidity', {}).get('usd_value', 0)
        scores['liquidity'] = min(100, (liquidity_usd / 1_000_000) * 100) if liquidity_usd else 0
        
        # Volume score (0-100)
        volume_24h = data.get('volume', {}).get('metrics', {}).get('current_volume', 0)
        relative_volume = data.get('volume', {}).get('relative_volume', 0)
        scores['volume'] = min(100, relative_volume * 20) if relative_volume else 0
        
        # Holder distribution score (0-100)
        holder_data = data.get('holders', {})
        concentration = holder_data.get('top_10_percentage', 100)
        scores['holder_distribution'] = max(0, 100 - concentration)
        
        # Developer activity score (0-100)
        dev_data = data.get('developer', {})
        scores['developer_activity'] = dev_data.get('activity_score', 50)
        
        # Contract safety score (0-100)
        rug_analysis = data.get('rug_analysis', {})
        safety_score = 100 * (1 - rug_analysis.get('risk_score', 0.5))
        scores['contract_safety'] = safety_score
        
        # Price action score (0-100)
        market_data = data.get('market', {})
        trend_strength = market_data.get('trend_strength', 0)
        scores['price_action'] = min(100, trend_strength * 100)
        
        # Social sentiment score (0-100)
        social_data = data.get('social', {})
        scores['social_sentiment'] = social_data.get('sentiment_score', 50)
        
        # Market correlation score (0-100)
        # Lower correlation is better (more independent)
        correlation = abs(data.get('token_info', {}).get('market_correlation', 0.5))
        scores['market_correlation'] = max(0, 100 * (1 - correlation))
        
        # Innovation score (0-100)
        scores['innovation'] = await self._calculate_innovation_score(data)
        
        return scores
        
    async def _calculate_risk_score(self, data: Dict) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        risk_components = []
        
        # Liquidity risk
        liquidity_usd = data.get('liquidity', {}).get('current_liquidity', {}).get('usd_value', 0)
        if liquidity_usd < 50000:
            risk_components.append(('low_liquidity', 30))
        elif liquidity_usd < 100000:
            risk_components.append(('low_liquidity', 15))
            
        # Holder concentration risk
        holder_data = data.get('holders', {})
        top_10_percentage = holder_data.get('top_10_percentage', 100)
        if top_10_percentage > 50:
            risk_components.append(('high_concentration', 25))
        elif top_10_percentage > 30:
            risk_components.append(('high_concentration', 10))
            
        # Contract risk
        rug_analysis = data.get('rug_analysis', {})
        contract_risk = rug_analysis.get('risk_score', 0.5) * 40
        if contract_risk > 0:
            risk_components.append(('contract_issues', contract_risk))
            
        # Volatility risk
        volatility = data.get('market', {}).get('volatility', 0)
        if volatility > 0.1:
            risk_components.append(('high_volatility', min(20, volatility * 100)))
            
        # Wash trading risk
        wash_score = data.get('volume', {}).get('wash_trading_score', 0)
        if wash_score > 0.5:
            risk_components.append(('suspicious_activity', wash_score * 20))
            
        # Calculate total risk
        total_risk = sum(risk for _, risk in risk_components)
        
        return min(100, total_risk)
        
    async def _calculate_opportunity_score(self, data: Dict) -> float:
        """Calculate opportunity score (0-100, higher is better)"""
        opportunity_factors = []
        
        # Momentum opportunity
        trend_data = data.get('market', {})
        if trend_data.get('primary_trend') == 'bullish':
            opportunity_factors.append(trend_data.get('trend_strength', 0) * 30)
            
        # Volume surge opportunity
        relative_volume = data.get('volume', {}).get('relative_volume', 1)
        if relative_volume > 2:
            opportunity_factors.append(min(25, (relative_volume - 1) * 10))
            
        # Breakout opportunity
        breakout_prob = trend_data.get('breakout_probability', 0)
        opportunity_factors.append(breakout_prob * 20)
        
        # Undervaluation opportunity
        if await self._is_undervalued(data):
            opportunity_factors.append(15)
            
        # Social momentum
        social_momentum = data.get('social', {}).get('momentum_score', 0)
        opportunity_factors.append(social_momentum * 10)
        
        return min(100, sum(opportunity_factors))
        
    async def _calculate_weighted_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        weighted_sum = 0
        total_weight = 0
        
        weight_map = {
            'liquidity': self.weights.liquidity,
            'volume': self.weights.volume,
            'holder_distribution': self.weights.holder_distribution,
            'developer_activity': self.weights.developer_activity,
            'contract_safety': self.weights.contract_safety,
            'price_action': self.weights.price_action,
            'social_sentiment': self.weights.social_sentiment,
            'market_correlation': self.weights.market_correlation,
            'innovation': self.weights.innovation
        }
        
        for category, score in category_scores.items():
            weight = weight_map.get(category, 0)
            weighted_sum += score * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 50.0  # Default middle score
            
    def _adjust_score_for_risk(self, composite_score: float, risk_score: float) -> float:
        """Adjust composite score based on risk"""
        # Risk adjustment factor (0.5 to 1.0)
        risk_factor = 1.0 - (risk_score / 200)  # Max 50% reduction
        
        return composite_score * risk_factor
        
    def _determine_grade(self, score: float) -> str:
        """Determine letter grade based on score"""
        for grade, threshold in self.thresholds.items():
            if score >= threshold:
                return grade
        return 'F'
        
    async def _generate_recommendation(
        self,
        composite_score: float,
        risk_score: float,
        opportunity_score: float,
        data: Dict
    ) -> str:
        """Generate trading recommendation"""
        if composite_score >= 80 and risk_score < 30:
            return "STRONG BUY - Excellent fundamentals with low risk"
        elif composite_score >= 70 and opportunity_score >= 70:
            return "BUY - Good opportunity with favorable risk/reward"
        elif composite_score >= 60 and risk_score < 40:
            return "MODERATE BUY - Decent opportunity, monitor closely"
        elif risk_score > 70:
            return "AVOID - High risk detected"
        elif composite_score < 50:
            return "AVOID - Weak fundamentals"
        else:
            return "HOLD/WATCH - Needs further analysis"
            
    def _identify_strengths(self, category_scores: Dict[str, float], data: Dict) -> List[str]:
        """Identify token strengths"""
        strengths = []
        
        # Check each category for high scores
        for category, score in category_scores.items():
            if score >= 80:
                if category == 'liquidity':
                    strengths.append("Strong liquidity depth")
                elif category == 'volume':
                    strengths.append("High trading volume")
                elif category == 'holder_distribution':
                    strengths.append("Well-distributed ownership")
                elif category == 'contract_safety':
                    strengths.append("Secure smart contract")
                elif category == 'price_action':
                    strengths.append("Positive price momentum")
                elif category == 'developer_activity':
                    strengths.append("Active development team")
                    
        # Check for special conditions
        if data.get('liquidity', {}).get('lock_data', {}).get('is_locked'):
            strengths.append("Liquidity locked")
            
        if data.get('rug_analysis', {}).get('contract_verified'):
            strengths.append("Verified contract")
            
        return strengths[:5]  # Top 5 strengths
        
    def _identify_weaknesses(self, category_scores: Dict[str, float], data: Dict) -> List[str]:
        """Identify token weaknesses"""
        weaknesses = []
        
        # Check each category for low scores
        for category, score in category_scores.items():
            if score < 40:
                if category == 'liquidity':
                    weaknesses.append("Low liquidity")
                elif category == 'volume':
                    weaknesses.append("Low trading volume")
                elif category == 'holder_distribution':
                    weaknesses.append("Concentrated ownership")
                elif category == 'contract_safety':
                    weaknesses.append("Contract safety concerns")
                elif category == 'price_action':
                    weaknesses.append("Weak price action")
                elif category == 'developer_activity':
                    weaknesses.append("Low developer activity")
                    
        # Check for risk factors
        if data.get('volume', {}).get('wash_trading_score', 0) > 0.5:
            weaknesses.append("Possible wash trading")
            
        if not data.get('liquidity', {}).get('lock_data', {}).get('is_locked'):
            weaknesses.append("Liquidity not locked")
            
        return weaknesses[:5]  # Top 5 weaknesses
        
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence in the scoring"""
        confidence_factors = []
        
        # Data completeness
        required_fields = ['liquidity', 'holders', 'volume', 'market', 'rug_analysis']
        completeness = sum(1 for field in required_fields if field in data and data[field]) / len(required_fields)
        confidence_factors.append(completeness)
        
        # Data freshness
        latest_timestamp = None
        for key, value in data.items():
            if isinstance(value, dict) and 'timestamp' in value:
                ts = value['timestamp']
                if isinstance(ts, datetime):
                    if latest_timestamp is None or ts > latest_timestamp:
                        latest_timestamp = ts
                        
        if latest_timestamp:
            age_minutes = (datetime.utcnow() - latest_timestamp).total_seconds() / 60
            freshness = max(0, 1 - (age_minutes / 60))  # Decay over 1 hour
            confidence_factors.append(freshness)
        else:
            confidence_factors.append(0.5)
            
        # Analysis quality
        rug_confidence = data.get('rug_analysis', {}).get('confidence', 0.5)
        confidence_factors.append(rug_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
        
    async def _calculate_innovation_score(self, data: Dict) -> float:
        """Calculate innovation/uniqueness score"""
        score = 50  # Base score
        
        # Check for unique features
        token_info = data.get('token_info', {})
        
        # Novel tokenomics
        if token_info.get('has_burn_mechanism'):
            score += 10
        if token_info.get('has_reflection_rewards'):
            score += 10
        if token_info.get('has_auto_liquidity'):
            score += 10
            
        # Technical innovation
        if token_info.get('uses_advanced_amm'):
            score += 15
        if token_info.get('cross_chain_enabled'):
            score += 15
            
        return min(100, score)
        
    async def _is_undervalued(self, data: Dict) -> bool:
        """Check if token appears undervalued"""
        # Simple heuristic - can be enhanced with more sophisticated valuation models
        
        market_cap = data.get('token_info', {}).get('market_cap', 0)
        volume = data.get('volume', {}).get('metrics', {}).get('current_volume', 0)
        liquidity = data.get('liquidity', {}).get('current_liquidity', {}).get('usd_value', 0)
        
        if market_cap > 0:
            # Volume to market cap ratio
            vol_mc_ratio = volume / market_cap if market_cap > 0 else 0
            
            # Liquidity to market cap ratio
            liq_mc_ratio = liquidity / market_cap if market_cap > 0 else 0
            
            # High volume and liquidity relative to market cap might indicate undervaluation
            if vol_mc_ratio > 0.5 and liq_mc_ratio > 0.2:
                return True
                
        return False
        
    def _get_rank_value(self, score: TokenScore, criteria: Optional[str]) -> float:
        """Get ranking value based on criteria"""
        if criteria == 'risk':
            return score.risk_score
        elif criteria == 'opportunity':
            return score.opportunity_score
        elif criteria == 'liquidity':
            return score.category_scores.get('liquidity', 0)
        else:
            return score.composite_score
            
    def _generate_comparison_recommendation(self, score1: TokenScore, score2: TokenScore) -> str:
        """Generate recommendation for token comparison"""
        diff = score1.composite_score - score2.composite_score
        
        if abs(diff) < 5:
            return "Both tokens are similarly rated. Choose based on your risk preference."
        elif diff > 20:
            return f"{score1.token} is significantly better rated."
        elif diff > 0:
            return f"{score1.token} has a moderately better score."
        elif diff < -20:
            return f"{score2.token} is significantly better rated."
        else:
            return f"{score2.token} has a moderately better score."
            
    def _is_cache_valid(self, timestamp: Optional[datetime]) -> bool:
        """Check if cached data is still valid"""
        if not timestamp:
            return False
            
        age_seconds = (datetime.utcnow() - timestamp).total_seconds()
        return age_seconds < self._cache_ttl
        
    def _get_default_score(self, token: str, chain: str) -> TokenScore:
        """Return default score when calculation fails"""
        return TokenScore(
            token=token,
            chain=chain,
            composite_score=0.0,
            category_scores={},
            risk_score=100.0,
            opportunity_score=0.0,
            confidence=0.0,
            grade='F',
            recommendation='Unable to score - insufficient data',
            strengths=[],
            weaknesses=['Scoring failed'],
            timestamp=datetime.utcnow()
        )
        
    async def _get_token_data(self, token: str, chain: str) -> Dict[str, Any]:
        """Get basic token data"""
        # This would fetch from your data collectors
        # Placeholder implementation
        return await self.db.get_token_info(token, chain) or {}
        
    async def _get_holder_distribution(self, token: str, chain: str) -> Dict[str, Any]:
        """Get holder distribution data"""
        # This would fetch holder data
        # Placeholder implementation
        return await self.db.get_holder_distribution(token, chain) or {}
        
    async def _get_social_metrics(self, token: str) -> Dict[str, Any]:
        """Get social media metrics"""
        # This would fetch social data
        # Placeholder implementation
        return {}
        
    async def _get_developer_activity(self, token: str, chain: str) -> Dict[str, Any]:
        """Get developer activity metrics"""
        # This would analyze GitHub, contract updates, etc.
        # Placeholder implementation
        return {'activity_score': 50}  # Default middle score

    # ============================================================================
    # PATCH FOR: token_scorer.py
    # Add these wrapper methods to the TokenScorer class
    # ============================================================================

    async def score_token(self, token: str, chain: str = 'ethereum') -> Dict:
        """
        Score a token (wrapper for calculate_composite_score)
        
        Args:
            token: Token address
            chain: Blockchain network
            
        Returns:
            Token score dictionary
        """
        # Use existing calculate_composite_score
        score = await self.calculate_composite_score(token, chain)
        
        # Convert to expected format
        return {
            'token': token,
            'chain': chain,
            'composite_score': score.composite_score,
            'fundamental_score': score.category_scores.get('liquidity', 0) * 0.5 + 
                                score.category_scores.get('holder_distribution', 0) * 0.5,
            'technical_score': score.category_scores.get('price_action', 0),
            'social_score': score.category_scores.get('social_sentiment', 0),
            'risk_score': score.risk_score,
            'opportunity_score': score.opportunity_score,
            'grade': score.grade,
            'recommendation': score.recommendation,
            'confidence': score.confidence,
            'timestamp': score.timestamp
        }

    def calculate_fundamental_score(self, token_data: Dict) -> float:
        """
        Calculate fundamental score for a token
        
        Args:
            token_data: Token fundamental data
            
        Returns:
            Fundamental score (0-100)
        """
        score = 50.0  # Base score
        
        # Market cap factor
        market_cap = token_data.get('market_cap', 0)
        if market_cap > 10_000_000:
            score += 15
        elif market_cap > 1_000_000:
            score += 10
        elif market_cap > 100_000:
            score += 5
        
        # Liquidity factor
        liquidity = token_data.get('liquidity', 0)
        if liquidity > 1_000_000:
            score += 15
        elif liquidity > 100_000:
            score += 10
        elif liquidity > 10_000:
            score += 5
        
        # Holder count factor
        holders = token_data.get('holder_count', 0)
        if holders > 1000:
            score += 10
        elif holders > 500:
            score += 7
        elif holders > 100:
            score += 3
        
        # Age factor
        age_days = token_data.get('age_days', 0)
        if age_days > 365:
            score += 10
        elif age_days > 90:
            score += 7
        elif age_days > 30:
            score += 3
        
        return min(100.0, max(0.0, score))

    def calculate_technical_score(self, price_data: Dict) -> float:
        """
        Calculate technical analysis score
        
        Args:
            price_data: Price and technical indicator data
            
        Returns:
            Technical score (0-100)
        """
        score = 50.0  # Base score
        
        # Trend factor
        trend = price_data.get('trend', 'neutral')
        if trend == 'bullish':
            score += 15
        elif trend == 'bearish':
            score -= 15
        
        # Momentum indicators
        rsi = price_data.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 5  # Neutral zone
        elif 60 < rsi <= 70:
            score += 10  # Bullish
        elif rsi > 70:
            score -= 5  # Overbought
        elif 30 <= rsi < 40:
            score -= 5  # Bearish
        elif rsi < 30:
            score -= 10  # Oversold
        
        # Moving average position
        if price_data.get('above_ma50', False):
            score += 10
        if price_data.get('above_ma200', False):
            score += 10
        
        # Volume confirmation
        if price_data.get('volume_increasing', False):
            score += 5
        
        # Support/Resistance
        if price_data.get('near_support', False):
            score += 10
        elif price_data.get('near_resistance', False):
            score -= 5
        
        return min(100.0, max(0.0, score))

    def calculate_social_score(self, social_data: Dict) -> float:
        """
        Calculate social sentiment score
        
        Args:
            social_data: Social media metrics
            
        Returns:
            Social score (0-100)
        """
        score = 50.0  # Base score
        
        # Sentiment factor
        sentiment = social_data.get('sentiment', 0)
        score += sentiment * 20  # -1 to 1 scaled to -20 to +20
        
        # Mention volume
        mentions = social_data.get('mentions_24h', 0)
        if mentions > 1000:
            score += 10
        elif mentions > 100:
            score += 5
        elif mentions < 10:
            score -= 10
        
        # Engagement rate
        engagement = social_data.get('engagement_rate', 0)
        score += engagement * 10
        
        # Influencer mentions
        if social_data.get('influencer_mentions', 0) > 0:
            score += 10
        
        # Community growth
        growth_rate = social_data.get('community_growth_rate', 0)
        if growth_rate > 0.1:  # 10% growth
            score += 10
        elif growth_rate > 0.05:  # 5% growth
            score += 5
        elif growth_rate < -0.05:  # 5% decline
            score -= 10
        
        return min(100.0, max(0.0, score))

    def get_overall_score(self, scores: Dict) -> float:
        """
        Calculate overall score from component scores
        
        Args:
            scores: Dictionary of component scores
            
        Returns:
            Overall score (0-100)
        """
        # Define weights for each component
        weights = {
            'fundamental': 0.35,
            'technical': 0.25,
            'social': 0.15,
            'risk': 0.25  # Negative weight
        }
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key == 'risk':
                # Risk is negative - lower is better
                weighted_sum += (100 - scores.get(key, 50)) * weight
            else:
                weighted_sum += scores.get(key, 50) * weight
            total_weight += abs(weight)
        
        # Return weighted average
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 50.0  # Default middle score