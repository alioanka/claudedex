"""
Futures Trading Strategies

Available strategies:
- Trend Following: Follow confirmed trends (long/short)
- Hedge Strategy: Hedge DEX positions with futures
- Funding Arbitrage: Exploit funding rate differences
"""

from .trend_following import TrendFollowingStrategy
from .hedge_strategy import HedgeStrategy
from .funding_arbitrage import FundingArbitrageStrategy
from .funding_carry import FundingCarryPlanner, CarryDecision

__all__ = [
    'TrendFollowingStrategy',
    'HedgeStrategy',
    'FundingArbitrageStrategy',
    'FundingCarryPlanner',
    'CarryDecision',
]
