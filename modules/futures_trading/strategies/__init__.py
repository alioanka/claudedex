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

__all__ = [
    'TrendFollowingStrategy',
    'HedgeStrategy',
    'FundingArbitrageStrategy',
    'FundingCarryPlanner',
    'CarryDecision',
]


def __getattr__(name):
    # Lazy export (PEP 562): funding_carry carries a __main__ self-test, so
    # eager-importing it here would trigger a runpy double-import warning on
    # `python -m ...strategies.funding_carry`. The engine imports the submodule
    # directly; package-level consumers still resolve these names on first use.
    if name in ('FundingCarryPlanner', 'CarryDecision'):
        from .funding_carry import FundingCarryPlanner, CarryDecision
        globals().update(
            FundingCarryPlanner=FundingCarryPlanner, CarryDecision=CarryDecision
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
