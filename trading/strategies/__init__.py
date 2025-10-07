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
            
        if self.config.get('ai_enabled', False):
            from .ai_strategy import AIStrategy
            self.strategies['ai'] = AIStrategy(self.config.get('ai', {}))
            
        # Initialize active strategies (only if they have initialize method)
        for name, strategy in self.strategies.items():
            if hasattr(strategy, 'initialize'):
                await strategy.initialize()
            self.active_strategies.add(name)
            
        logger.info(f"Initialized {len(self.strategies)} strategies")
        
    def select_strategy(self, opportunity: Any) -> str:
        """
        Select best strategy for opportunity
        
        Args:
            opportunity: Trading opportunity
            
        Returns:
            Strategy name
        """
        # Simple selection logic - can be enhanced
        if opportunity.pump_probability > 0.7:
            return 'momentum'
        elif opportunity.liquidity < 100000:
            return 'scalping'
        else:
            return 'ai' if 'ai' in self.active_strategies else 'momentum'
            
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