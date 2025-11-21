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

        # 🆕 Track strategy usage statistics
        self.strategy_stats = {
            'momentum': 0,
            'scalping': 0,
            'ai': 0
        }
        
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
        Select best strategy for opportunity (IMPROVED: Balanced multi-strategy support)

        Args:
            opportunity: Trading opportunity

        Returns:
            Strategy name
        """
        # 🆕 IMPROVED STRATEGY SELECTION LOGIC
        # Now uses a priority system with better balance across all strategies

        selected_strategy = None

        # Priority 1: Scalping for HIGH volatility + TIGHT spreads + GOOD liquidity
        # Scalping works best with: rapid price moves, low slippage, deep liquidity
        if (hasattr(opportunity, 'volatility') and hasattr(opportunity, 'spread') and
            hasattr(opportunity, 'liquidity')):

            # Scalping conditions (more balanced):
            # - High volatility (>3% for quick gains)
            # - Tight spread (<1% for low cost)
            # - Sufficient liquidity (>$50k for execution)
            if (opportunity.volatility > 0.03 and
                opportunity.spread < 0.01 and
                opportunity.liquidity > 50000):

                if 'scalping' in self.active_strategies:
                    selected_strategy = 'scalping'
                    logger.debug(
                        f"Scalping selected: vol={opportunity.volatility:.2%}, "
                        f"spread={opportunity.spread:.2%}, liq=${opportunity.liquidity:,.0f}"
                    )

        # Priority 2: AI for MEDIUM confidence (sweet spot for ML models)
        # AI works best when probability is uncertain but positive
        if selected_strategy is None:
            # Wider range for AI: 40%-80% (catches more opportunities)
            if 0.40 <= opportunity.pump_probability <= 0.80:
                if 'ai' in self.active_strategies:
                    selected_strategy = 'ai'
                    logger.debug(
                        f"AI selected: pump_prob={opportunity.pump_probability:.2%} "
                        f"(medium confidence range)"
                    )

        # Priority 3: Momentum for VERY HIGH confidence
        # Momentum works best for clear trends with strong signals
        if selected_strategy is None:
            # Only use momentum for VERY high confidence (>75%)
            if opportunity.pump_probability > 0.75:
                if 'momentum' in self.active_strategies:
                    selected_strategy = 'momentum'
                    logger.debug(
                        f"Momentum selected: pump_prob={opportunity.pump_probability:.2%} "
                        f"(high confidence)"
                    )

        # Priority 4: Default strategy based on availability
        if selected_strategy is None:
            # Prefer AI for lower confidence, then momentum, then scalping
            if 'ai' in self.active_strategies:
                selected_strategy = 'ai'
                logger.debug("AI selected (default for low-medium confidence)")
            elif 'momentum' in self.active_strategies:
                selected_strategy = 'momentum'
                logger.debug("Momentum selected (default fallback)")
            elif 'scalping' in self.active_strategies:
                selected_strategy = 'scalping'
                logger.debug("Scalping selected (only available)")
            else:
                selected_strategy = 'momentum'  # Ultimate fallback
                logger.warning("No strategies active, using momentum as ultimate fallback")

        # 🆕 Track statistics
        if selected_strategy in self.strategy_stats:
            self.strategy_stats[selected_strategy] += 1

        return selected_strategy

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

    def get_strategy_stats(self) -> Dict:
        """
        Get strategy usage statistics

        Returns:
            Dict with strategy usage counts and percentages
        """
        total = sum(self.strategy_stats.values())

        if total == 0:
            return {
                'total_opportunities': 0,
                'strategies': {name: {'count': 0, 'percentage': 0.0}
                              for name in self.strategy_stats.keys()}
            }

        stats = {
            'total_opportunities': total,
            'strategies': {}
        }

        for name, count in self.strategy_stats.items():
            stats['strategies'][name] = {
                'count': count,
                'percentage': (count / total) * 100
            }

        return stats

    def log_strategy_stats(self):
        """Log current strategy usage statistics"""
        stats = self.get_strategy_stats()

        if stats['total_opportunities'] == 0:
            logger.info("📊 Strategy Stats: No opportunities processed yet")
            return

        logger.info(
            f"📊 STRATEGY USAGE STATISTICS (Total: {stats['total_opportunities']} opportunities):\n"
            f"   🎯 Momentum:  {stats['strategies']['momentum']['count']:3d} "
            f"({stats['strategies']['momentum']['percentage']:5.1f}%)\n"
            f"   ⚡ Scalping:  {stats['strategies']['scalping']['count']:3d} "
            f"({stats['strategies']['scalping']['percentage']:5.1f}%)\n"
            f"   🤖 AI:        {stats['strategies']['ai']['count']:3d} "
            f"({stats['strategies']['ai']['percentage']:5.1f}%)"
        )

# Export main classes
__all__ = ['StrategyManager']