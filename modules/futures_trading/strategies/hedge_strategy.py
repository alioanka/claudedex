"""
Hedge Strategy for Futures

Automatically hedges DEX spot positions with futures shorts
to reduce downside risk during market downturns.

Hedge Sizing:
- Full hedge: 100% of DEX position size
- Partial hedge: 50-80% of DEX position size
- Dynamic: Adjusts based on market conditions
"""

import logging
from typing import Dict, Optional, List


class HedgeStrategy:
    """
    Hedge strategy to protect DEX positions

    Triggers:
    - DEX position in drawdown > threshold
    - Market showing weakness
    - High volatility periods

    Hedge Ratio:
    - 1:1 for full protection
    - 0.5:1 for partial protection
    - Dynamic based on correlation
    """

    def __init__(self, config: Dict):
        """
        Initialize hedge strategy

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger("HedgeStrategy")

        # Hedge parameters
        self.drawdown_threshold = config.get('drawdown_threshold', 0.10)  # 10% drawdown
        self.hedge_ratio = config.get('hedge_ratio', 0.8)  # 80% hedge
        self.min_position_size = config.get('min_position_size', 10.0)  # $10 min

    def should_hedge(
        self,
        dex_positions: List[Dict],
        market_conditions: Dict
    ) -> Optional[Dict]:
        """
        Determine if DEX positions should be hedged

        Args:
            dex_positions: List of open DEX positions
            market_conditions: Current market conditions

        Returns:
            Optional[Dict]: Hedge signal or None
        """
        try:
            if not dex_positions:
                return None

            # Calculate total DEX exposure
            total_exposure = sum(
                p.get('cost', 0) for p in dex_positions
            )

            # Calculate total unrealized PnL
            total_pnl = sum(
                p.get('pnl', 0) for p in dex_positions
            )

            # Calculate drawdown percentage
            if total_exposure > 0:
                drawdown_pct = total_pnl / total_exposure
            else:
                return None

            # Check if drawdown exceeds threshold
            if drawdown_pct < -self.drawdown_threshold:
                # Calculate hedge size
                hedge_size = total_exposure * self.hedge_ratio

                if hedge_size < self.min_position_size:
                    return None

                return {
                    'action': 'HEDGE',
                    'hedge_size_usd': hedge_size,
                    'dex_exposure': total_exposure,
                    'current_drawdown': drawdown_pct,
                    'hedge_ratio': self.hedge_ratio,
                    'reason': f'DEX drawdown {drawdown_pct*100:.1f}% exceeds threshold',
                    'positions_to_hedge': len(dex_positions)
                }

            # Check market conditions
            volatility = market_conditions.get('volatility', 0)
            if volatility > 100:  # High volatility
                hedge_size = total_exposure * 0.5  # Partial hedge

                if hedge_size >= self.min_position_size:
                    return {
                        'action': 'HEDGE',
                        'hedge_size_usd': hedge_size,
                        'dex_exposure': total_exposure,
                        'current_drawdown': drawdown_pct,
                        'hedge_ratio': 0.5,
                        'reason': f'High volatility ({volatility:.1f}%)',
                        'positions_to_hedge': len(dex_positions)
                    }

            return None

        except Exception as e:
            self.logger.error(f"Error determining hedge: {e}")
            return None

    def calculate_hedge_effectiveness(
        self,
        dex_pnl: float,
        futures_pnl: float,
        total_capital: float
    ) -> float:
        """
        Calculate hedge effectiveness

        Args:
            dex_pnl: DEX P&L
            futures_pnl: Futures P&L
            total_capital: Total capital

        Returns:
            float: Hedge effectiveness (0-1, 1 = perfect hedge)
        """
        try:
            if total_capital == 0:
                return 0.0

            # Perfect hedge: dex_pnl + futures_pnl ≈ 0
            combined_pnl = dex_pnl + futures_pnl
            hedge_effectiveness = 1 - abs(combined_pnl / total_capital)

            return max(0.0, min(1.0, hedge_effectiveness))

        except Exception as e:
            self.logger.error(f"Error calculating hedge effectiveness: {e}")
            return 0.0

    def should_remove_hedge(
        self,
        hedge_position: Dict,
        dex_positions: List[Dict],
        market_conditions: Dict
    ) -> bool:
        """
        Determine if hedge should be removed

        Args:
            hedge_position: Current futures hedge position
            dex_positions: DEX positions
            market_conditions: Market conditions

        Returns:
            bool: True if hedge should be removed
        """
        try:
            if not dex_positions:
                # No DEX positions, remove hedge
                return True

            # Calculate current DEX PnL
            total_pnl = sum(p.get('pnl', 0) for p in dex_positions)
            total_cost = sum(p.get('cost', 0) for p in dex_positions)

            if total_cost > 0:
                pnl_pct = total_pnl / total_cost
            else:
                return True

            # Remove hedge if:
            # 1. DEX positions recovered (positive PnL)
            # 2. Market conditions improved
            # 3. Volatility normalized

            if pnl_pct > 0:
                self.logger.info(f"Removing hedge: DEX positions recovered ({pnl_pct*100:.1f}%)")
                return True

            volatility = market_conditions.get('volatility', 0)
            if volatility < 30:  # Low volatility
                self.logger.info(f"Removing hedge: Low volatility ({volatility:.1f}%)")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking hedge removal: {e}")
            return False
