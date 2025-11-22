"""
Funding Rate Arbitrage Strategy

Exploits funding rate differences between spot and perpetual futures:

Positive Funding (Long pays Short):
- Long spot, Short perp → Earn funding
- Risk-neutral position

Negative Funding (Short pays Long):
- Short spot (or avoid), Long perp → Earn funding

Funding rates typically paid every 8 hours.
"""

import logging
from typing import Dict, Optional


class FundingArbitrageStrategy:
    """
    Funding rate arbitrage strategy

    Profit from funding rate payments without directional exposure

    Entry:
    - Funding rate > threshold
    - Open opposite position (long spot + short perp, or vice versa)

    Exit:
    - Funding rate normalizes
    - Position held for minimum duration (capture funding)
    """

    def __init__(self, config: Dict):
        """
        Initialize funding arbitrage strategy

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger("FundingArbitrage")

        # Strategy parameters
        self.min_funding_rate = config.get('min_funding_rate', 0.01)  # 0.01% = 0.0001
        self.min_apr = config.get('min_apr', 0.10)  # 10% APR minimum
        self.min_hold_hours = config.get('min_hold_hours', 8)  # Hold for at least 1 funding

    def analyze_opportunity(
        self,
        symbol: str,
        spot_price: float,
        perp_price: float,
        funding_rate: float
    ) -> Optional[Dict]:
        """
        Analyze funding arbitrage opportunity

        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            perp_price: Current perpetual price
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)

        Returns:
            Optional[Dict]: Opportunity signal or None
        """
        try:
            # Calculate annualized funding rate (assuming 8-hour funding)
            # APR = funding_rate * (365 * 24 / 8) = funding_rate * 1095
            funding_apr = abs(funding_rate) * 1095

            # Check if funding rate is significant enough
            if abs(funding_rate) < self.min_funding_rate:
                return None

            if funding_apr < self.min_apr:
                return None

            # Calculate price difference (should be minimal for arb)
            price_diff_pct = abs(perp_price - spot_price) / spot_price

            # Price difference should be < 0.5% for true arbitrage
            if price_diff_pct > 0.005:
                self.logger.warning(
                    f"Price difference too high for arbitrage: {price_diff_pct*100:.2f}%"
                )
                return None

            # Determine position direction
            if funding_rate > 0:
                # Positive funding: Longs pay Shorts
                # Strategy: Long spot + Short perp (earn funding)
                return {
                    'action': 'FUNDING_ARB',
                    'symbol': symbol,
                    'spot_action': 'BUY',
                    'perp_action': 'SHORT',
                    'funding_rate': funding_rate,
                    'funding_apr': funding_apr,
                    'spot_price': spot_price,
                    'perp_price': perp_price,
                    'price_diff_pct': price_diff_pct,
                    'reason': f'Positive funding {funding_rate*100:.4f}% (APR: {funding_apr*100:.1f}%)',
                    'expected_profit_per_period': funding_rate,
                    'min_hold_hours': self.min_hold_hours
                }

            else:
                # Negative funding: Shorts pay Longs
                # Strategy: Short spot + Long perp (earn funding)
                # Note: Shorting spot is harder, so might just long perp
                return {
                    'action': 'FUNDING_ARB',
                    'symbol': symbol,
                    'spot_action': 'SKIP',  # Hard to short spot
                    'perp_action': 'LONG',
                    'funding_rate': funding_rate,
                    'funding_apr': funding_apr,
                    'spot_price': spot_price,
                    'perp_price': perp_price,
                    'price_diff_pct': price_diff_pct,
                    'reason': f'Negative funding {funding_rate*100:.4f}% (APR: {funding_apr*100:.1f}%)',
                    'expected_profit_per_period': abs(funding_rate),
                    'min_hold_hours': self.min_hold_hours
                }

        except Exception as e:
            self.logger.error(f"Error analyzing funding opportunity: {e}")
            return None

    def should_exit(
        self,
        position: Dict,
        current_funding_rate: float,
        hours_held: float
    ) -> Optional[Dict]:
        """
        Check if funding arbitrage position should be exited

        Args:
            position: Current arb position
            current_funding_rate: Current funding rate
            hours_held: Hours position has been held

        Returns:
            Optional[Dict]: Exit signal or None
        """
        try:
            # Must hold for at least one funding period
            if hours_held < self.min_hold_hours:
                return None

            # Exit if funding rate normalizes (becomes unprofitable)
            if abs(current_funding_rate) < self.min_funding_rate / 2:
                return {
                    'action': 'CLOSE',
                    'reason': f'Funding rate normalized ({current_funding_rate*100:.4f}%)',
                    'hours_held': hours_held
                }

            # Exit if funding rate reverses significantly
            entry_rate = position.get('funding_rate', 0)
            if entry_rate > 0 and current_funding_rate < 0:
                return {
                    'action': 'CLOSE',
                    'reason': 'Funding rate reversed',
                    'hours_held': hours_held
                }
            elif entry_rate < 0 and current_funding_rate > 0:
                return {
                    'action': 'CLOSE',
                    'reason': 'Funding rate reversed',
                    'hours_held': hours_held
                }

            # Continue holding if still profitable
            return None

        except Exception as e:
            self.logger.error(f"Error checking exit condition: {e}")
            return None

    def calculate_expected_profit(
        self,
        position_size_usd: float,
        funding_rate: float,
        funding_periods: int = 1
    ) -> Dict:
        """
        Calculate expected profit from funding arbitrage

        Args:
            position_size_usd: Size of position in USD
            funding_rate: Funding rate per period
            funding_periods: Number of funding periods to hold

        Returns:
            Dict: Profit calculations
        """
        try:
            # Profit per funding period
            profit_per_period = position_size_usd * abs(funding_rate)

            # Total expected profit
            total_profit = profit_per_period * funding_periods

            # APR
            apr = abs(funding_rate) * 1095  # 3 fundings/day * 365 days

            return {
                'position_size_usd': position_size_usd,
                'funding_rate': funding_rate,
                'funding_periods': funding_periods,
                'profit_per_period': profit_per_period,
                'total_expected_profit': total_profit,
                'apr': apr,
                'daily_profit': profit_per_period * 3  # 3 fundings per day
            }

        except Exception as e:
            self.logger.error(f"Error calculating expected profit: {e}")
            return {}
