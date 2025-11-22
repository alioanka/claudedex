"""
Futures Risk Manager

Specialized risk management for leverage trading:
- Liquidation price monitoring
- Leverage limits
- Position size validation
- Cross-position exposure tracking
- Auto-deleveraging on drawdown
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime


class FuturesRiskManager:
    """
    Risk management for futures trading

    Key features:
    - Prevent over-leverage
    - Monitor liquidation risk
    - Enforce position limits
    - Track total exposure (long + short)
    - Auto-reduce leverage on losses
    """

    def __init__(self, config: Dict):
        """
        Initialize futures risk manager

        Args:
            config: Risk configuration
        """
        self.config = config
        self.logger = logging.getLogger("FuturesRiskManager")

        # Risk parameters
        self.max_leverage = config.get('max_leverage', 3)
        self.max_positions = config.get('max_positions', 3)
        self.max_total_exposure = config.get('max_total_exposure', 500.0)  # USD
        self.liquidation_buffer = config.get('liquidation_buffer', 0.20)  # 20% from liq price
        self.max_drawdown = config.get('max_drawdown', 0.10)  # 10%

        # State tracking
        self.consecutive_losses = 0
        self.total_realized_pnl = 0.0

    def validate_new_position(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        leverage: int,
        current_positions: List[Dict],
        available_capital: float
    ) -> Dict:
        """
        Validate if new position can be opened

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            size_usd: Position size in USD
            leverage: Requested leverage
            current_positions: List of current positions
            available_capital: Available capital

        Returns:
            Dict: Validation result with 'allowed' and 'reason'
        """
        try:
            # Check position count
            if len(current_positions) >= self.max_positions:
                return {
                    'allowed': False,
                    'reason': f'Max positions reached ({self.max_positions})'
                }

            # Check leverage limit
            if leverage > self.max_leverage:
                return {
                    'allowed': False,
                    'reason': f'Leverage {leverage}x exceeds max {self.max_leverage}x',
                    'suggested_leverage': self.max_leverage
                }

            # Check capital availability
            required_margin = size_usd / leverage
            if required_margin > available_capital:
                return {
                    'allowed': False,
                    'reason': f'Insufficient capital (need ${required_margin:.2f}, have ${available_capital:.2f})'
                }

            # Check total exposure
            current_exposure = sum(
                abs(p.get('notional_value', 0)) for p in current_positions
            )
            total_exposure = current_exposure + size_usd

            if total_exposure > self.max_total_exposure:
                return {
                    'allowed': False,
                    'reason': f'Total exposure ${total_exposure:.2f} exceeds limit ${self.max_total_exposure:.2f}'
                }

            # All checks passed
            return {
                'allowed': True,
                'reason': 'Position validated',
                'required_margin': required_margin,
                'total_exposure_after': total_exposure
            }

        except Exception as e:
            self.logger.error(f"Error validating position: {e}")
            return {
                'allowed': False,
                'reason': f'Validation error: {str(e)}'
            }

    def check_liquidation_risk(
        self,
        position: Dict,
        current_price: float
    ) -> Dict:
        """
        Check if position is at risk of liquidation

        Args:
            position: Position info with liquidation_price
            current_price: Current market price

        Returns:
            Dict: Risk assessment
        """
        try:
            liq_price = position.get('liquidation_price', 0)
            if liq_price == 0:
                return {'risk_level': 'unknown'}

            side = position.get('side', 'LONG')

            # Calculate distance to liquidation
            if side == 'LONG':
                # For longs, liquidation price is below current
                distance_pct = (current_price - liq_price) / current_price
            else:
                # For shorts, liquidation price is above current
                distance_pct = (liq_price - current_price) / current_price

            # Assess risk level
            if distance_pct < 0:
                risk_level = 'CRITICAL'  # Already past liquidation
            elif distance_pct < 0.05:
                risk_level = 'EXTREME'  # < 5% from liquidation
            elif distance_pct < 0.10:
                risk_level = 'HIGH'  # < 10% from liquidation
            elif distance_pct < self.liquidation_buffer:
                risk_level = 'MEDIUM'  # < buffer from liquidation
            else:
                risk_level = 'LOW'  # Safe distance

            return {
                'risk_level': risk_level,
                'distance_pct': distance_pct * 100,
                'current_price': current_price,
                'liquidation_price': liq_price,
                'should_reduce': risk_level in ['CRITICAL', 'EXTREME', 'HIGH']
            }

        except Exception as e:
            self.logger.error(f"Error checking liquidation risk: {e}")
            return {'risk_level': 'unknown'}

    def should_auto_deleverage(
        self,
        total_pnl: float,
        total_capital: float
    ) -> bool:
        """
        Check if positions should be auto-deleveraged due to losses

        Args:
            total_pnl: Total realized + unrealized PnL
            total_capital: Total capital

        Returns:
            bool: True if should deleverage
        """
        try:
            if total_capital == 0:
                return False

            # Calculate drawdown
            drawdown_pct = abs(total_pnl / total_capital)

            # Auto-deleverage if drawdown exceeds threshold
            if drawdown_pct > self.max_drawdown:
                self.logger.warning(
                    f"⚠️ Auto-deleveraging triggered: "
                    f"Drawdown {drawdown_pct*100:.1f}% > {self.max_drawdown*100:.1f}%"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking auto-deleverage: {e}")
            return False

    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        leverage: int,
        stop_loss_pct: float
    ) -> float:
        """
        Calculate appropriate position size

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade (e.g., 0.02 = 2%)
            leverage: Leverage to use
            stop_loss_pct: Stop loss percentage (e.g., 0.03 = 3%)

        Returns:
            float: Position size in USD
        """
        try:
            # Risk amount
            risk_amount = capital * risk_per_trade

            # Position size = risk / stop_loss_pct
            position_size = risk_amount / stop_loss_pct

            # Account for leverage (can open larger position with same margin)
            leveraged_size = position_size * leverage

            # Cap at max exposure
            leveraged_size = min(leveraged_size, self.max_total_exposure)

            return leveraged_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def update_on_trade_close(self, pnl: float):
        """
        Update risk state after trade closes

        Args:
            pnl: Trade profit/loss
        """
        try:
            self.total_realized_pnl += pnl

            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Log if multiple consecutive losses
            if self.consecutive_losses >= 3:
                self.logger.warning(
                    f"⚠️ {self.consecutive_losses} consecutive losses in futures"
                )

        except Exception as e:
            self.logger.error(f"Error updating trade close: {e}")

    def get_adjusted_leverage(
        self,
        base_leverage: int,
        volatility: float,
        win_rate: float
    ) -> int:
        """
        Adjust leverage based on market conditions and performance

        Args:
            base_leverage: Base leverage to use
            volatility: Current market volatility (%)
            win_rate: Current win rate (0-1)

        Returns:
            int: Adjusted leverage
        """
        try:
            adjusted = base_leverage

            # Reduce leverage in high volatility
            if volatility > 100:
                adjusted = max(1, adjusted - 1)
            elif volatility > 50:
                # Keep same
                pass
            else:
                # Can slightly increase in low volatility
                adjusted = min(self.max_leverage, adjusted + 1)

            # Reduce leverage if poor performance
            if win_rate < 0.4:
                adjusted = max(1, adjusted - 1)

            # Reduce leverage if consecutive losses
            if self.consecutive_losses >= 3:
                adjusted = 1  # Drop to 1x after 3 losses

            return max(1, min(adjusted, self.max_leverage))

        except Exception as e:
            self.logger.error(f"Error adjusting leverage: {e}")
            return 1

    def get_risk_summary(self, positions: List[Dict]) -> Dict:
        """
        Get overall risk summary

        Args:
            positions: List of current positions

        Returns:
            Dict: Risk summary
        """
        try:
            total_long = sum(
                p.get('notional_value', 0)
                for p in positions
                if p.get('side') == 'LONG'
            )

            total_short = sum(
                abs(p.get('notional_value', 0))
                for p in positions
                if p.get('side') == 'SHORT'
            )

            net_exposure = total_long - total_short
            total_exposure = total_long + total_short

            # Count positions at risk
            positions_at_risk = sum(
                1 for p in positions
                if self.check_liquidation_risk(p, p.get('mark_price', 0))['risk_level'] in ['HIGH', 'EXTREME', 'CRITICAL']
            )

            return {
                'total_positions': len(positions),
                'total_long_exposure': total_long,
                'total_short_exposure': total_short,
                'net_exposure': net_exposure,
                'total_exposure': total_exposure,
                'positions_at_risk': positions_at_risk,
                'consecutive_losses': self.consecutive_losses,
                'total_realized_pnl': self.total_realized_pnl
            }

        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
