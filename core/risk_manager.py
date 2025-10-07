"""
Advanced Risk Management System
Multi-layer risk assessment and position sizing
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
from enum import Enum
import json

from data.collectors.chain_data import ChainDataCollector
from data.collectors.honeypot_checker import HoneypotChecker
from security.wallet_security import WalletSecurityManager

class RiskLevel(Enum):
    """Risk level classifications"""
    ULTRA_SAFE = "ultra_safe"
    SAFE = "safe"
    MODERATE = "moderate"
    RISKY = "risky"
    EXTREME = "extreme"
    UNTRADEABLE = "untradeable"

@dataclass
class RiskScore:
    """Comprehensive risk assessment"""
    # Individual risk scores (0-1, where 1 is highest risk)
    liquidity_risk: float
    developer_risk: float
    contract_risk: float
    volume_risk: float
    holder_risk: float
    social_risk: float
    technical_risk: float
    market_risk: float
    
    # Detailed sub-scores
    liquidity_locked: bool = False
    liquidity_lock_duration: int = 0  # days
    ownership_renounced: bool = False
    honeypot_risk: float = 0.0
    mint_function_present: bool = False
    pause_function_present: bool = False
    blacklist_function_present: bool = False
    proxy_contract: bool = False
    verified_contract: bool = False
    
    # Holder analysis
    top_10_holders_percentage: float = 0.0
    whale_concentration: float = 0.0
    unique_holders: int = 0
    holder_growth_rate: float = 0.0
    
    # Volume analysis
    buy_sell_ratio: float = 0.0
    unique_buyers_24h: int = 0
    wash_trading_score: float = 0.0
    
    # Developer analysis
    dev_wallet_balance: float = 0.0
    dev_previous_projects: int = 0
    dev_rug_history: bool = False
    dev_reputation_score: float = 0.0
    
    # Market conditions
    market_volatility: float = 0.0
    correlation_with_major_tokens: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    @property
    def overall_risk(self) -> float:
        """Calculate weighted overall risk score"""
        weights = {
            'liquidity': 0.25,
            'developer': 0.20,
            'contract': 0.20,
            'volume': 0.10,
            'holder': 0.10,
            'social': 0.05,
            'technical': 0.05,
            'market': 0.05
        }
        
        scores = {
            'liquidity': self.liquidity_risk,
            'developer': self.developer_risk,
            'contract': self.contract_risk,
            'volume': self.volume_risk,
            'holder': self.holder_risk,
            'social': self.social_risk,
            'technical': self.technical_risk,
            'market': self.market_risk
        }
        
        # Apply weights
        weighted_score = sum(scores[k] * weights[k] for k in weights)
        
        # Apply penalties for critical issues
        if self.honeypot_risk > 0.5:
            weighted_score = min(weighted_score + 0.3, 1.0)
        if self.dev_rug_history:
            weighted_score = min(weighted_score + 0.4, 1.0)
        if not self.liquidity_locked:
            weighted_score = min(weighted_score + 0.1, 1.0)
        if self.mint_function_present and not self.ownership_renounced:
            weighted_score = min(weighted_score + 0.2, 1.0)
            
        return weighted_score
    
    @property
    def risk_level(self) -> RiskLevel:
        """Get risk level classification"""
        score = self.overall_risk
        
        if score < 0.2:
            return RiskLevel.ULTRA_SAFE
        elif score < 0.4:
            return RiskLevel.SAFE
        elif score < 0.6:
            return RiskLevel.MODERATE
        elif score < 0.8:
            return RiskLevel.RISKY
        elif score < 0.95:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.UNTRADEABLE

# Add these class definitions at the beginning of risk_manager.py after the existing imports and dataclasses

@dataclass
class TradingOpportunity:
    """Trading opportunity for evaluation"""
    token_address: str
    chain: str
    price: float
    volume: float
    liquidity: float
    risk_score: 'RiskScore'
    ml_confidence: float
    pump_probability: float
    rug_probability: float
    expected_return: float
    signals: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class Position:
    """Active trading position"""
    position_id: str
    token_address: str
    chain: str
    entry_price: float
    current_price: float
    amount: float
    value: float
    stop_loss: Optional[float] = None
    take_profit: Optional[List[float]] = None
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        self.chain_collector = ChainDataCollector(config.get('web3', {}))
        self.honeypot_checker = HoneypotChecker()
        self.wallet_manager = WalletSecurityManager(config.get('security', {}))
        
        # Risk thresholds
        self.thresholds = config.get('risk_levels', {
            'ultra_safe': {'max_risk': 0.2, 'position_multiplier': 1.5},
            'safe': {'max_risk': 0.4, 'position_multiplier': 1.0},
            'moderate': {'max_risk': 0.6, 'position_multiplier': 0.7},
            'risky': {'max_risk': 0.8, 'position_multiplier': 0.3},
            'extreme': {'max_risk': 0.95, 'position_multiplier': 0.1}
        })
        
        # Position sizing parameters
        self.max_position_size_percent = config.get('max_position_size_percent', 5)
        self.max_portfolio_risk_percent = config.get('max_portfolio_risk_percent', 20)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Conservative Kelly
        
        # Cache for risk assessments
        self.risk_cache: Dict[str, RiskScore] = {}
        self.cache_duration = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize risk manager components"""
        # Only initialize components that have initialize methods
        if hasattr(self.chain_collector, 'initialize'):
            await self.chain_collector.initialize()
        
        if hasattr(self.honeypot_checker, 'initialize'):
            await self.honeypot_checker.initialize()
        
        # Initialize internal state
        self.positions = {}
        self.balance = Decimal("1.0")  # Default starting balance
        self.peak_balance = self.balance
        self.consecutive_losses = 0
        self.returns_history = []
        self.max_positions = self.config.get('max_positions', 10)
        
    async def analyze_token(self, token_address: str, force_refresh: bool = False) -> RiskScore:
        """
        Comprehensive risk analysis of a token
        
        Args:
            token_address: Token contract address
            force_refresh: Force fresh analysis ignoring cache
            
        Returns:
            Comprehensive risk score
        """
        # Check cache
        if not force_refresh and token_address in self.risk_cache:
            cached = self.risk_cache[token_address]
            if (datetime.now() - cached.timestamp).seconds < self.cache_duration:
                return cached
                
        try:
            # Run parallel risk assessments
            tasks = [
                self._analyze_liquidity_risk(token_address),
                self._analyze_developer_risk(token_address),
                self._analyze_contract_risk(token_address),
                self._analyze_volume_risk(token_address),
                self._analyze_holder_risk(token_address),
                self._analyze_social_risk(token_address),
                self._analyze_technical_risk(token_address),
                self._analyze_market_risk(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = 1.0  # Maximum risk on error
                    
            # Create risk score
            risk_score = RiskScore(
                liquidity_risk=results[0]['score'] if isinstance(results[0], dict) else results[0],
                developer_risk=results[1]['score'] if isinstance(results[1], dict) else results[1],
                contract_risk=results[2]['score'] if isinstance(results[2], dict) else results[2],
                volume_risk=results[3]['score'] if isinstance(results[3], dict) else results[3],
                holder_risk=results[4]['score'] if isinstance(results[4], dict) else results[4],
                social_risk=results[5]['score'] if isinstance(results[5], dict) else results[5],
                technical_risk=results[6]['score'] if isinstance(results[6], dict) else results[6],
                market_risk=results[7]['score'] if isinstance(results[7], dict) else results[7]
            )
            
            # Add detailed attributes from results
            for result in results:
                if isinstance(result, dict) and 'details' in result:
                    for key, value in result['details'].items():
                        if hasattr(risk_score, key):
                            setattr(risk_score, key, value)
                            
            # Cache the result
            self.risk_cache[token_address] = risk_score
            
            return risk_score
            
        except Exception as e:
            # Return maximum risk on error
            return RiskScore(
                liquidity_risk=1.0,
                developer_risk=1.0,
                contract_risk=1.0,
                volume_risk=1.0,
                holder_risk=1.0,
                social_risk=1.0,
                technical_risk=1.0,
                market_risk=1.0,
                confidence=0.1
            )
            
    async def _analyze_liquidity_risk(self, token_address: str) -> Dict:
        """Analyze liquidity-related risks"""
        try:
            liquidity_data = await self.chain_collector.get_liquidity_info(token_address)
            
            risk_score = 0.0
            details = {}
            
            # Check liquidity amount
            total_liquidity = liquidity_data.get('total_liquidity_usd', 0)
            if total_liquidity < 10000:
                risk_score += 0.3
            elif total_liquidity < 50000:
                risk_score += 0.15
            elif total_liquidity < 100000:
                risk_score += 0.05
                
            details['total_liquidity'] = total_liquidity
            
            # Check if liquidity is locked
            lock_info = liquidity_data.get('lock_info', {})
            details['liquidity_locked'] = lock_info.get('is_locked', False)
            
            if not details['liquidity_locked']:
                risk_score += 0.25
            else:
                # Check lock duration
                lock_duration = lock_info.get('lock_duration_days', 0)
                details['liquidity_lock_duration'] = lock_duration
                
                if lock_duration < 30:
                    risk_score += 0.15
                elif lock_duration < 180:
                    risk_score += 0.05
                    
            # Check liquidity concentration
            lp_holders = liquidity_data.get('lp_holders', [])
            if lp_holders:
                top_holder_percentage = lp_holders[0].get('percentage', 0)
                if top_holder_percentage > 50:
                    risk_score += 0.2
                elif top_holder_percentage > 30:
                    risk_score += 0.1
                    
            # Check liquidity to market cap ratio
            market_cap = liquidity_data.get('market_cap', 0)
            if market_cap > 0:
                liq_to_mcap = total_liquidity / market_cap
                if liq_to_mcap < 0.05:
                    risk_score += 0.2
                elif liq_to_mcap < 0.1:
                    risk_score += 0.1
                    
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 1.0, 'details': {}}
            
    async def _analyze_developer_risk(self, token_address: str) -> Dict:
        """Analyze developer-related risks"""
        try:
            dev_data = await self.chain_collector.get_developer_info(token_address)
            
            risk_score = 0.0
            details = {}
            
            # Check developer wallet balance
            dev_balance = dev_data.get('dev_wallet_balance', 0)
            details['dev_wallet_balance'] = dev_balance
            
            if dev_balance > 20:  # More than 20% of supply
                risk_score += 0.3
            elif dev_balance > 10:
                risk_score += 0.15
            elif dev_balance > 5:
                risk_score += 0.05
                
            # Check previous projects
            previous_projects = dev_data.get('previous_projects', [])
            details['dev_previous_projects'] = len(previous_projects)
            
            # Analyze previous project outcomes
            rug_count = sum(1 for p in previous_projects if p.get('outcome') == 'rug_pull')
            successful_count = sum(1 for p in previous_projects if p.get('outcome') == 'successful')
            
            if rug_count > 0:
                risk_score += 0.4
                details['dev_rug_history'] = True
            elif len(previous_projects) == 0:
                risk_score += 0.2  # Unknown developer
            elif successful_count / len(previous_projects) < 0.5:
                risk_score += 0.15
                
            # Check if developer is doxxed
            is_doxxed = dev_data.get('is_doxxed', False)
            if not is_doxxed:
                risk_score += 0.1
                
            # Check team token vesting
            vesting_info = dev_data.get('vesting_info', {})
            if not vesting_info.get('has_vesting', False):
                risk_score += 0.15
                
            # Calculate reputation score
            details['dev_reputation_score'] = max(0, 1 - risk_score)
            
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 0.8, 'details': {}}  # High risk for unknown developer
            
    async def _analyze_contract_risk(self, token_address: str) -> Dict:
        """Analyze smart contract risks"""
        try:
            # Check for honeypot
            honeypot_result = await self.honeypot_checker.check(token_address)
            
            risk_score = 0.0
            details = {}
            
            # Honeypot risk
            details['honeypot_risk'] = honeypot_result.get('risk_score', 0)
            risk_score += details['honeypot_risk'] * 0.4
            
            # Get contract analysis
            contract_data = await self.chain_collector.analyze_contract(token_address)
            
            # Check for dangerous functions
            functions = contract_data.get('functions', [])
            
            details['mint_function_present'] = 'mint' in functions
            if details['mint_function_present']:
                risk_score += 0.2
                
            details['pause_function_present'] = 'pause' in functions
            if details['pause_function_present']:
                risk_score += 0.15
                
            details['blacklist_function_present'] = 'blacklist' in functions
            if details['blacklist_function_present']:
                risk_score += 0.15
                
            # Check ownership
            details['ownership_renounced'] = contract_data.get('ownership_renounced', False)
            if not details['ownership_renounced'] and details['mint_function_present']:
                risk_score += 0.2
                
            # Check if proxy contract
            details['proxy_contract'] = contract_data.get('is_proxy', False)
            if details['proxy_contract']:
                risk_score += 0.1  # Proxy contracts can be upgraded
                
            # Check if verified
            details['verified_contract'] = contract_data.get('is_verified', False)
            if not details['verified_contract']:
                risk_score += 0.1
                
            # Check contract age
            contract_age_days = contract_data.get('age_days', 0)
            if contract_age_days < 1:
                risk_score += 0.15
            elif contract_age_days < 7:
                risk_score += 0.05
                
            # Check for known vulnerabilities
            vulnerabilities = contract_data.get('vulnerabilities', [])
            if vulnerabilities:
                risk_score += min(len(vulnerabilities) * 0.1, 0.3)
                
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 1.0, 'details': {}}
            
    async def _analyze_volume_risk(self, token_address: str) -> Dict:
        """Analyze volume-related risks"""
        try:
            volume_data = await self.chain_collector.get_volume_analysis(token_address)
            
            risk_score = 0.0
            details = {}
            
            # Check 24h volume
            volume_24h = volume_data.get('volume_24h_usd', 0)
            if volume_24h < 10000:
                risk_score += 0.2
            elif volume_24h < 50000:
                risk_score += 0.1
                
            # Check buy/sell ratio
            buy_volume = volume_data.get('buy_volume_24h', 0)
            sell_volume = volume_data.get('sell_volume_24h', 0)
            
            if buy_volume + sell_volume > 0:
                details['buy_sell_ratio'] = buy_volume / (buy_volume + sell_volume)
                
                # Suspicious if too skewed
                if details['buy_sell_ratio'] > 0.9 or details['buy_sell_ratio'] < 0.1:
                    risk_score += 0.25
                elif details['buy_sell_ratio'] > 0.8 or details['buy_sell_ratio'] < 0.2:
                    risk_score += 0.1
                    
            # Check unique buyers
            details['unique_buyers_24h'] = volume_data.get('unique_buyers_24h', 0)
            if details['unique_buyers_24h'] < 50:
                risk_score += 0.15
            elif details['unique_buyers_24h'] < 100:
                risk_score += 0.05
                
            # Wash trading detection
            wash_trading_indicators = volume_data.get('wash_trading_indicators', {})
            details['wash_trading_score'] = wash_trading_indicators.get('score', 0)
            risk_score += details['wash_trading_score'] * 0.3
            
            # Check for volume spikes
            volume_history = volume_data.get('volume_history', [])
            if volume_history:
                avg_volume = np.mean(volume_history)
                if volume_24h > avg_volume * 10:  # Suspicious spike
                    risk_score += 0.15
                    
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 0.7, 'details': {}}
            
    async def _analyze_holder_risk(self, token_address: str) -> Dict:
        """Analyze holder distribution risks"""
        try:
            holder_data = await self.chain_collector.get_holder_analysis(token_address)
            
            risk_score = 0.0
            details = {}
            
            # Check top holder concentration
            top_holders = holder_data.get('top_holders', [])
            if top_holders:
                top_10_percentage = sum(h.get('percentage', 0) for h in top_holders[:10])
                details['top_10_holders_percentage'] = top_10_percentage
                
                if top_10_percentage > 70:
                    risk_score += 0.3
                elif top_10_percentage > 50:
                    risk_score += 0.15
                elif top_10_percentage > 30:
                    risk_score += 0.05
                    
            # Check whale concentration
            whale_holders = [h for h in top_holders if h.get('percentage', 0) > 5]
            details['whale_concentration'] = len(whale_holders)
            
            if details['whale_concentration'] > 5:
                risk_score += 0.2
            elif details['whale_concentration'] > 3:
                risk_score += 0.1
                
            # Check unique holder count
            details['unique_holders'] = holder_data.get('total_holders', 0)
            if details['unique_holders'] < 100:
                risk_score += 0.25
            elif details['unique_holders'] < 500:
                risk_score += 0.1
            elif details['unique_holders'] < 1000:
                risk_score += 0.05
                
            # Check holder growth rate
            holder_history = holder_data.get('holder_history', [])
            if len(holder_history) >= 2:
                recent_holders = holder_history[-1].get('count', 0)
                previous_holders = holder_history[-2].get('count', 0)
                
                if previous_holders > 0:
                    details['holder_growth_rate'] = (recent_holders - previous_holders) / previous_holders
                    
                    # Negative growth is bad
                    if details['holder_growth_rate'] < 0:
                        risk_score += 0.2
                        
            # Check for snipers
            snipers = holder_data.get('snipers', [])
            if len(snipers) > 10:
                risk_score += 0.15
                
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 0.8, 'details': {}}
            
    async def _analyze_social_risk(self, token_address: str) -> Dict:
        """Analyze social sentiment and activity risks"""
        try:
            # This would integrate with social data collector
            # For now, return moderate risk
            return {'score': 0.5, 'details': {}}
            
        except Exception as e:
            return {'score': 0.6, 'details': {}}
            
    async def _analyze_technical_risk(self, token_address: str) -> Dict:
        """Analyze technical indicators and price action risks"""
        try:
            technical_data = await self.chain_collector.get_technical_indicators(token_address)
            
            risk_score = 0.0
            
            # Check volatility
            volatility = technical_data.get('volatility_24h', 0)
            if volatility > 100:  # More than 100% daily volatility
                risk_score += 0.3
            elif volatility > 50:
                risk_score += 0.15
            elif volatility > 25:
                risk_score += 0.05
                
            # Check for pump patterns
            price_history = technical_data.get('price_history', [])
            if price_history:
                recent_high = max(price_history[-24:]) if len(price_history) >= 24 else max(price_history)
                current_price = price_history[-1]
                
                # Check if dumping after pump
                if current_price < recent_high * 0.5:
                    risk_score += 0.25
                elif current_price < recent_high * 0.7:
                    risk_score += 0.1
                    
            return {
                'score': min(risk_score, 1.0),
                'details': {'volatility': volatility}
            }
            
        except Exception as e:
            return {'score': 0.5, 'details': {}}
            
    async def _analyze_market_risk(self, token_address: str) -> Dict:
        """Analyze overall market condition risks"""
        try:
            market_data = await self.chain_collector.get_market_conditions()
            
            risk_score = 0.0
            details = {}
            
            # Check overall market volatility
            details['market_volatility'] = market_data.get('volatility_index', 0)
            if details['market_volatility'] > 80:  # High fear index
                risk_score += 0.2
            elif details['market_volatility'] > 50:
                risk_score += 0.1
                
            # Check correlation with major tokens
            correlations = market_data.get('correlations', {})
            btc_correlation = correlations.get('btc', 0)
            eth_correlation = correlations.get('eth', 0)
            
            details['correlation_with_major_tokens'] = max(abs(btc_correlation), abs(eth_correlation))
            
            # Low correlation might mean higher risk (not following market)
            if details['correlation_with_major_tokens'] < 0.3:
                risk_score += 0.1
                
            return {
                'score': min(risk_score, 1.0),
                'details': details
            }
            
        except Exception as e:
            return {'score': 0.3, 'details': {}}
            
    # Replace the existing calculate_position_size method with this corrected version
    # The API expects: async def calculate_position_size(opportunity: TradingOpportunity) -> Decimal

    async def calculate_position_size(self, opportunity: 'TradingOpportunity') -> Decimal:
        """
        Calculate optimal position size using multiple methods
        
        Args:
            opportunity: Trading opportunity to evaluate
            
        Returns:
            Recommended position size in base currency as Decimal
        """
        try:
            # Extract necessary data from opportunity
            risk_score = opportunity.risk_score
            ml_confidence = opportunity.ml_confidence
            expected_return = opportunity.expected_return
            
            # Get available balance
            available_balance = await self.wallet_manager.get_available_balance()
            
            # Base position size (percentage of portfolio)
            base_position_percent = self.max_position_size_percent
            
            # Adjust based on risk level
            risk_level = risk_score.risk_level
            if risk_level == RiskLevel.UNTRADEABLE:
                return Decimal("0")
                
            risk_multiplier = self.thresholds[risk_level.value]['position_multiplier']
            
            # Apply ML confidence adjustment
            confidence_multiplier = 0.5 + (ml_confidence * 0.5)  # Range: 0.5-1.0
            
            # Kelly Criterion calculation
            if expected_return > 0:
                # Simplified Kelly: f = (p*b - q) / b
                # where p = probability of win, b = odds, q = probability of loss
                win_probability = ml_confidence
                loss_probability = 1 - win_probability
                odds = expected_return / 0.1  # Assuming 10% stop loss
                
                kelly_fraction = (win_probability * odds - loss_probability) / odds
                kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
                
                # Apply conservative Kelly fraction
                kelly_position = kelly_fraction * self.kelly_fraction * available_balance
            else:
                kelly_position = 0
                
            # Calculate final position size
            position_size = min(
                base_position_percent * risk_multiplier * confidence_multiplier * available_balance / 100,
                kelly_position if kelly_position > 0 else float('inf'),
                available_balance * self.max_position_size_percent / 100
            )
            
            # Additional safety checks
            if risk_score.honeypot_risk > 0.3:
                position_size *= 0.5
                
            if not risk_score.liquidity_locked:
                position_size *= 0.7
                
            if risk_score.dev_rug_history:
                position_size *= 0.3
                
            # Ensure minimum viable position
            min_position = available_balance * 0.001  # 0.1% minimum
            if position_size < min_position:
                return Decimal("0")
                
            return Decimal(str(round(position_size, 2)))
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return Decimal("0")
        
    def calculate_stop_loss(self, risk_score: RiskScore) -> float:
        """
        Calculate stop loss percentage based on risk
        
        Args:
            risk_score: Token risk assessment
            
        Returns:
            Stop loss percentage (e.g., 0.1 for 10%)
        """
        risk_level = risk_score.risk_level
        
        # Base stop loss percentages by risk level
        stop_loss_map = {
            RiskLevel.ULTRA_SAFE: 0.15,  # 15% stop loss
            RiskLevel.SAFE: 0.12,
            RiskLevel.MODERATE: 0.10,
            RiskLevel.RISKY: 0.08,
            RiskLevel.EXTREME: 0.05,
            RiskLevel.UNTRADEABLE: 0.0
        }
        
        base_stop_loss = stop_loss_map[risk_level]
        
        # Adjust based on specific risks
        if risk_score.honeypot_risk > 0.5:
            base_stop_loss *= 0.5  # Tighter stop loss
            
        if risk_score.liquidity_risk > 0.7:
            base_stop_loss *= 0.7  # Account for slippage
            
        return base_stop_loss
        
    def calculate_take_profit(self, risk_score: RiskScore, market_conditions: Dict = None) -> List[float]:
        """
        Calculate take profit targets based on risk
        
        Args:
            risk_score: Token risk assessment
            market_conditions: Current market conditions
            
        Returns:
            List of take profit percentages
        """
        risk_level = risk_score.risk_level
        
        # Base take profit targets by risk level
        take_profit_map = {
            RiskLevel.ULTRA_SAFE: [0.20, 0.40, 0.60],  # 20%, 40%, 60%
            RiskLevel.SAFE: [0.25, 0.50, 0.75],
            RiskLevel.MODERATE: [0.30, 0.60, 1.00],
            RiskLevel.RISKY: [0.40, 0.80, 1.50],
            RiskLevel.EXTREME: [0.50, 1.00, 2.00],
            RiskLevel.UNTRADEABLE: []
        }
        
        targets = take_profit_map[risk_level].copy()
        
        # Adjust based on market conditions
        if market_conditions:
            if market_conditions.get('trend') == 'bullish':
                targets = [t * 1.2 for t in targets]
            elif market_conditions.get('trend') == 'bearish':
                targets = [t * 0.8 for t in targets]
                
        return targets
        
    async def validate_trade(self, token_address: str, amount: float) -> Tuple[bool, str]:
        """
        Final validation before executing trade
        
        Args:
            token_address: Token to trade
            amount: Trade amount
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Get fresh risk assessment
            risk_score = await self.analyze_token(token_address, force_refresh=True)
            
            # Check if untradeable
            if risk_score.risk_level == RiskLevel.UNTRADEABLE:
                return False, "Token risk level is untradeable"
                
            # Check honeypot
            if risk_score.honeypot_risk > 0.7:
                return False, "High honeypot risk detected"
                
            # Check liquidity
            if risk_score.liquidity_risk > 0.8:
                return False, "Insufficient liquidity"
                
            # Check developer risk
            if risk_score.dev_rug_history:
                return False, "Developer has rug pull history"
                
            # Check position size limits
            available_balance = await self.wallet_manager.get_available_balance()
            max_position = available_balance * self.max_position_size_percent / 100
            
            if amount > max_position:
                return False, f"Position size exceeds maximum ({max_position})"
                
            return True, "Trade validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    # Add these methods to existing RiskManager class
    
    async def check_position_limit(self, token: str) -> bool:
        """Check if position limit allows new position"""
        try:
            # Check if token already in portfolio
            existing_positions = [p for p in self.positions if p.token_address == token]
            if existing_positions:
                return False  # Already have position in this token
            
            # Check total position count
            if len(self.positions) >= self.max_positions:
                return False
            
            return True
            
        except Exception as e:
            print(f"Position limit check error: {e}")
            return False
    
    async def set_stop_loss(self, position: Position) -> Decimal:
        """Set stop loss for a position"""
        try:
            # Get risk score for the token
            risk_score = await self.analyze_token(position.token_address)
            
            # Calculate stop loss
            stop_loss_pct = self.calculate_stop_loss(risk_score)
            stop_loss_price = Decimal(str(position.entry_price * (1 - stop_loss_pct)))
            
            # Update position
            position.stop_loss = float(stop_loss_price)
            
            return stop_loss_price
            
        except Exception as e:
            print(f"Stop loss setting error: {e}")
            return Decimal(str(position.entry_price * 0.9))  # Default 10% stop
    
    async def check_portfolio_exposure(self) -> Dict:
        """Check overall portfolio exposure"""
        try:
            total_value = sum(p.value for p in self.positions.values())
            available = self.balance - total_value
            
            exposure = {
                'total_exposure': total_value,
                'available_balance': available,
                'exposure_percentage': (total_value / self.balance) * 100 if self.balance > 0 else 0,
                'positions_count': len(self.positions),
                'max_position_reached': len(self.positions) >= self.max_positions,
                'risk_level': self._calculate_portfolio_risk_level()
            }
            
            # Add per-chain exposure
            chain_exposure = {}
            for position in self.positions.values():
                chain = position.chain
                if chain not in chain_exposure:
                    chain_exposure[chain] = 0
                chain_exposure[chain] += position.value
            
            exposure['chain_exposure'] = chain_exposure
            
            return exposure
            
        except Exception as e:
            print(f"Portfolio exposure check error: {e}")
            return {'error': str(e)}
    
    async def calculate_var(self, confidence: float = 0.95) -> Decimal:
        """Calculate Value at Risk"""
        try:
            if not self.positions:
                return Decimal("0")
            
            # Get historical returns for positions
            returns = []
            for position in self.positions.values():
                # Fetch historical data
                history = await self.chain_collector.get_price_history(
                    position.token_address,
                    periods=100
                )
                if history:
                    # Calculate returns
                    prices = [h['price'] for h in history]
                    for i in range(1, len(prices)):
                        returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if not returns:
                return Decimal("0")
            
            # Calculate VaR
            returns_sorted = sorted(returns)
            index = int(len(returns_sorted) * (1 - confidence))
            var_value = abs(returns_sorted[index]) * sum(p.value for p in self.positions.values())
            
            return Decimal(str(var_value))
            
        except Exception as e:
            print(f"VaR calculation error: {e}")
            return Decimal("0")
    
    async def check_correlation_limit(self, token: str) -> bool:
        """Check if token correlation with portfolio is within limits"""
        try:
            if not self.positions:
                return True
            
            # Get token price history
            token_history = await self.chain_collector.get_price_history(token)
            if not token_history:
                return False
            
            token_returns = self._calculate_returns([h['price'] for h in token_history])
            
            # Check correlation with each position
            high_correlations = 0
            for position in self.positions.values():
                pos_history = await self.chain_collector.get_price_history(position.token_address)
                if pos_history:
                    pos_returns = self._calculate_returns([h['price'] for h in pos_history])
                    
                    # Calculate correlation
                    if len(token_returns) == len(pos_returns):
                        correlation = np.corrcoef(token_returns, pos_returns)[0, 1]
                        if abs(correlation) > 0.7:  # High correlation threshold
                            high_correlations += 1
            
            # Reject if too many high correlations
            return high_correlations < len(self.positions) * 0.3
            
        except Exception as e:
            print(f"Correlation check error: {e}")
            return True
    
    async def emergency_stop_check(self) -> bool:
        """Check if emergency stop conditions are met"""
        try:
            # Check portfolio drawdown
            current_value = sum(p.value for p in self.positions.values()) + self.balance
            drawdown = (self.peak_balance - current_value) / self.peak_balance
            
            if drawdown > 0.25:  # 25% drawdown
                return True
            
            # Check consecutive losses
            if self.consecutive_losses > 5:
                return True
            
            # Check daily loss
            daily_loss = self._calculate_daily_loss()
            if daily_loss > 0.15:  # 15% daily loss
                return True
            
            # Check system health
            system_healthy = await self._check_system_health()
            if not system_healthy:
                return True
            
            return False
            
        except Exception as e:
            print(f"Emergency stop check error: {e}")
            return True  # Default to stop on error
    
    async def calculate_sharpe_ratio(self) -> Decimal:
        """Calculate portfolio Sharpe ratio"""
        try:
            if not self.returns_history or len(self.returns_history) < 2:
                return Decimal("0")
            
            returns = np.array(self.returns_history)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return Decimal("0")
            
            # Assuming risk-free rate of 0.02 (2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            
            sharpe = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
            
            return Decimal(str(sharpe))
            
        except Exception as e:
            print(f"Sharpe ratio calculation error: {e}")
            return Decimal("0")
    
    async def calculate_sortino_ratio(self) -> Decimal:
        """Calculate portfolio Sortino ratio"""
        try:
            if not self.returns_history or len(self.returns_history) < 2:
                return Decimal("0")
            
            returns = np.array(self.returns_history)
            avg_return = np.mean(returns)
            
            # Downside deviation (only negative returns)
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return Decimal("999")  # No downside
            
            downside_std = np.std(negative_returns)
            
            if downside_std == 0:
                return Decimal("999")
            
            # Assuming risk-free rate of 0.02 (2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            
            sortino = (avg_return - risk_free_rate) / downside_std * np.sqrt(252)
            
            return Decimal(str(sortino))
            
        except Exception as e:
            print(f"Sortino ratio calculation error: {e}")
            return Decimal("0")
