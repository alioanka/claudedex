"""
Trading Strategies Package
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StrategyManager:
    """Manages and coordinates trading strategies"""
    
    def __init__(self, config: Dict):
        """
        Initialize strategy manager
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.strategies = {}
        self.parameters = {}
        self.active_strategies = set()
        
    async def initialize(self):
        """Initialize all strategies"""
        # Load strategies based on config
        if self.config.get('momentum_enabled', True):
            from .momentum import MomentumStrategy
            self.strategies['momentum'] = MomentumStrategy(self.config.get('momentum', {}))
            
        if self.config.get('scalping_enabled', True):
            from .scalping import ScalpingStrategy
            self.strategies['scalping'] = ScalpingStrategy(self.config.get('scalping', {}))
            
        # CRITICAL FIX: Enable AI strategy by default (was False, now True)
        if self.config.get('ai_enabled', True):  # Changed from False to True
            from .ai_strategy import AIStrategy
            # Lower thresholds for production testing
            ai_config = self.config.get('ai', {})
            ai_config.setdefault('ml_confidence_threshold', 0.65)  # Lowered from 0.75
            ai_config.setdefault('min_pump_probability', 0.50)      # Lowered from 0.60
            self.strategies['ai'] = AIStrategy(ai_config)
            
        # Initialize active strategies (only if they have initialize method)
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'initialize'):
                await strategy.initialize()
            self.active_strategies.add(name)
            
        logger.info(f"Initialized {len(self.strategies)} strategies")
        
    def select_strategy(self, opportunity: Any) -> str:
        """
        Select best strategy for opportunity (FIXED: Multi-strategy support)

        Args:
            opportunity: Trading opportunity

        Returns:
            Strategy name
        """
        # CRITICAL FIX: Improved strategy selection logic

        # Fix scalping config conflict: use volatility instead of liquidity
        # Old bug: selected scalping for liquidity < 100k, but scalping requires >= 100k
        # New: select scalping for high volatility + tight spreads
        if hasattr(opportunity, 'volatility') and hasattr(opportunity, 'spread'):
            if opportunity.volatility > 0.05 and opportunity.spread < 0.01:
                if 'scalping' in self.active_strategies:
                    return 'scalping'

        # Use AI for medium confidence opportunities (if available)
        if 0.50 <= opportunity.pump_probability <= 0.75:
            if 'ai' in self.active_strategies:
                return 'ai'

        # Use momentum for high confidence opportunities
        if opportunity.pump_probability > 0.70:
            return 'momentum'

        # Default to AI if available, otherwise momentum
        return 'ai' if 'ai' in self.active_strategies else 'momentum'

    def select_strategies_multi(self, opportunity: Any) -> List[str]:
        """
        Select MULTIPLE strategies for opportunity (ENHANCED FEATURE)

        This allows running multiple strategies in parallel for better coverage.

        Args:
            opportunity: Trading opportunity

        Returns:
            List of strategy names to apply
        """
        selected = []

        # Always use momentum for high pump probability
        if opportunity.pump_probability > 0.70:
            if 'momentum' in self.active_strategies:
                selected.append('momentum')

        # Add scalping for high volatility
        if hasattr(opportunity, 'volatility') and opportunity.volatility > 0.05:
            if 'scalping' in self.active_strategies:
                selected.append('scalping')

        # Add AI for risk assessment
        if 'ai' in self.active_strategies:
            selected.append('ai')

        # Always return at least one strategy
        if not selected and 'momentum' in self.active_strategies:
            selected.append('momentum')

        return selected
            
    def get_parameters(self) -> Dict:
        """Get current strategy parameters"""
        return self.parameters.copy()
        
    async def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        self.parameters.update(new_params)
        
        # Update individual strategies (only if they have the method)
        for name, strategy in self.strategies.items():
            if name in new_params and hasattr(strategy, 'update_parameters'):
                await strategy.update_parameters(new_params[name])

# Export main classes
__all__ = ['StrategyManager']