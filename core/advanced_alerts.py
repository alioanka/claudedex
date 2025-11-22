"""
Advanced Alert System

Monitors trading performance and sends intelligent alerts based on:
- Performance thresholds
- Risk metrics
- Anomaly detection
- Strategy performance
- Portfolio health
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("AdvancedAlerts")


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE = "performance"
    RISK = "risk"
    DRAWDOWN = "drawdown"
    LOSS_STREAK = "loss_streak"
    WIN_STREAK = "win_streak"
    EXPOSURE = "exposure"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    ANOMALY = "anomaly"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    type: AlertType
    module: str
    title: str
    message: str
    timestamp: datetime
    data: Dict
    acknowledged: bool = False


class AdvancedAlertSystem:
    """
    Advanced Alert System

    Monitors performance and risk metrics, generates intelligent alerts
    """

    def __init__(
        self,
        analytics_engine=None,
        alert_manager=None,
        module_manager=None
    ):
        """
        Initialize Advanced Alert System

        Args:
            analytics_engine: Analytics engine
            alert_manager: Basic alert manager
            module_manager: Module manager
        """
        self.analytics = analytics_engine
        self.alert_manager = alert_manager
        self.module_manager = module_manager

        self.logger = logger

        # Alert configuration
        self.config = {
            # Performance thresholds
            'max_drawdown_pct': 0.15,  # 15%
            'min_win_rate': 0.40,  # 40%
            'max_loss_streak': 5,
            'min_sharpe_ratio': 0.5,

            # Risk thresholds
            'max_exposure_pct': 0.80,  # 80% of capital
            'max_leverage': 10,
            'max_position_concentration': 0.30,  # 30% in single position
            'var_99_threshold': 1000,  # $1000 VaR

            # Volatility thresholds
            'max_daily_volatility': 0.10,  # 10%
            'anomaly_std_threshold': 3.0,  # 3 standard deviations

            # Liquidity thresholds
            'min_liquidity_score': 50,
            'max_low_liquidity_positions': 2,

            # Check intervals
            'check_interval': 300,  # 5 minutes
            'anomaly_detection_window': 30  # 30 data points
        }

        # Alert history
        self.alerts: List[Alert] = []
        self.alert_count: Dict[str, int] = {}

        # Cooldowns to prevent spam
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(hours=1)

        # Monitoring state
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Historical data for anomaly detection
        self.performance_history: Dict[str, List[float]] = {}

        self.logger.info("Advanced Alert System initialized")

    async def start(self):
        """Start alert monitoring"""
        try:
            self._running = True

            # Start monitoring tasks
            task = asyncio.create_task(self._monitoring_loop())
            self._tasks.append(task)

            self.logger.info("Advanced Alert System started")

        except Exception as e:
            self.logger.error(f"Error starting alert system: {e}", exc_info=True)

    async def stop(self):
        """Stop alert monitoring"""
        try:
            self._running = False

            # Cancel tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()

            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks.clear()

            self.logger.info("Advanced Alert System stopped")

        except Exception as e:
            self.logger.error(f"Error stopping alert system: {e}", exc_info=True)

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_all_modules()
                await asyncio.sleep(self.config['check_interval'])

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _check_all_modules(self):
        """Check all modules for alert conditions"""
        try:
            if not self.module_manager:
                return

            for module_name in self.module_manager.modules.keys():
                await self._check_module(module_name)

        except Exception as e:
            self.logger.error(f"Error checking modules: {e}", exc_info=True)

    async def _check_module(self, module_name: str):
        """Check a module for alert conditions"""
        try:
            if not self.analytics:
                return

            # Get performance metrics
            from core.analytics_engine import TimeFrame
            perf = await self.analytics.get_module_performance(
                module_name,
                TimeFrame.HOUR_24
            )

            # Get risk metrics
            risk = await self.analytics.get_risk_metrics(module_name)

            # Check various conditions
            await self._check_drawdown(module_name, perf)
            await self._check_win_rate(module_name, perf)
            await self._check_loss_streak(module_name, perf)
            await self._check_sharpe_ratio(module_name, perf)
            await self._check_exposure(module_name, risk)
            await self._check_leverage(module_name, risk)
            await self._check_concentration(module_name, risk)
            await self._check_volatility(module_name, risk)
            await self._check_liquidity(module_name, risk)
            await self._check_anomalies(module_name, perf)

            # Store performance data for anomaly detection
            self._store_performance_data(module_name, perf)

        except Exception as e:
            self.logger.error(f"Error checking module {module_name}: {e}", exc_info=True)

    async def _check_drawdown(self, module_name: str, perf):
        """Check for excessive drawdown"""
        try:
            if perf.current_drawdown > self.config['max_drawdown_pct']:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.DRAWDOWN,
                    module=module_name,
                    title="Excessive Drawdown Detected",
                    message=f"Current drawdown of {perf.current_drawdown*100:.1f}% exceeds threshold of {self.config['max_drawdown_pct']*100:.1f}%",
                    data={
                        'current_drawdown': perf.current_drawdown,
                        'max_drawdown': perf.max_drawdown,
                        'threshold': self.config['max_drawdown_pct']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")

    async def _check_win_rate(self, module_name: str, perf):
        """Check for low win rate"""
        try:
            if perf.total_trades >= 10 and perf.win_rate < self.config['min_win_rate']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.PERFORMANCE,
                    module=module_name,
                    title="Low Win Rate",
                    message=f"Win rate of {perf.win_rate*100:.1f}% is below threshold of {self.config['min_win_rate']*100:.1f}%",
                    data={
                        'win_rate': perf.win_rate,
                        'total_trades': perf.total_trades,
                        'threshold': self.config['min_win_rate']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking win rate: {e}")

    async def _check_loss_streak(self, module_name: str, perf):
        """Check for prolonged loss streak"""
        try:
            if perf.current_streak < 0 and abs(perf.current_streak) >= self.config['max_loss_streak']:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.LOSS_STREAK,
                    module=module_name,
                    title="Prolonged Loss Streak",
                    message=f"Current loss streak of {abs(perf.current_streak)} trades exceeds threshold of {self.config['max_loss_streak']}",
                    data={
                        'current_streak': perf.current_streak,
                        'max_loss_streak': perf.max_loss_streak,
                        'threshold': self.config['max_loss_streak']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking loss streak: {e}")

    async def _check_sharpe_ratio(self, module_name: str, perf):
        """Check for poor Sharpe ratio"""
        try:
            if perf.total_trades >= 20 and perf.sharpe_ratio < self.config['min_sharpe_ratio']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.PERFORMANCE,
                    module=module_name,
                    title="Poor Risk-Adjusted Returns",
                    message=f"Sharpe ratio of {perf.sharpe_ratio:.2f} is below threshold of {self.config['min_sharpe_ratio']}",
                    data={
                        'sharpe_ratio': perf.sharpe_ratio,
                        'threshold': self.config['min_sharpe_ratio']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking Sharpe ratio: {e}")

    async def _check_exposure(self, module_name: str, risk):
        """Check for excessive exposure"""
        try:
            # Get module capital allocation
            if self.module_manager:
                module = self.module_manager.modules.get(module_name)
                if module:
                    capital = module.config.capital_allocation
                    exposure_pct = float(risk.total_exposure / Decimal(str(capital)))

                    if exposure_pct > self.config['max_exposure_pct']:
                        await self._create_alert(
                            level=AlertLevel.WARNING,
                            type=AlertType.EXPOSURE,
                            module=module_name,
                            title="High Exposure",
                            message=f"Exposure of {exposure_pct*100:.1f}% exceeds threshold of {self.config['max_exposure_pct']*100:.1f}%",
                            data={
                                'exposure': float(risk.total_exposure),
                                'exposure_pct': exposure_pct,
                                'threshold': self.config['max_exposure_pct']
                            }
                        )

        except Exception as e:
            self.logger.error(f"Error checking exposure: {e}")

    async def _check_leverage(self, module_name: str, risk):
        """Check for excessive leverage"""
        try:
            if risk.max_leverage > self.config['max_leverage']:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    type=AlertType.RISK,
                    module=module_name,
                    title="Excessive Leverage",
                    message=f"Maximum leverage of {risk.max_leverage:.1f}x exceeds threshold of {self.config['max_leverage']}x",
                    data={
                        'max_leverage': risk.max_leverage,
                        'avg_leverage': risk.avg_leverage,
                        'threshold': self.config['max_leverage']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking leverage: {e}")

    async def _check_concentration(self, module_name: str, risk):
        """Check for position concentration"""
        try:
            concentration_pct = risk.largest_position_pct / 100

            if concentration_pct > self.config['max_position_concentration']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.RISK,
                    module=module_name,
                    title="High Position Concentration",
                    message=f"Largest position at {risk.largest_position_pct:.1f}% exceeds threshold of {self.config['max_position_concentration']*100:.1f}%",
                    data={
                        'largest_position_pct': risk.largest_position_pct,
                        'top_5_positions_pct': risk.top_5_positions_pct,
                        'threshold': self.config['max_position_concentration'] * 100
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking concentration: {e}")

    async def _check_volatility(self, module_name: str, risk):
        """Check for high volatility"""
        try:
            if risk.daily_volatility > self.config['max_daily_volatility']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.VOLATILITY,
                    module=module_name,
                    title="High Volatility",
                    message=f"Daily volatility of {risk.daily_volatility*100:.1f}% exceeds threshold of {self.config['max_daily_volatility']*100:.1f}%",
                    data={
                        'daily_volatility': risk.daily_volatility,
                        'annual_volatility': risk.annual_volatility,
                        'threshold': self.config['max_daily_volatility']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking volatility: {e}")

    async def _check_liquidity(self, module_name: str, risk):
        """Check for liquidity issues"""
        try:
            if risk.low_liquidity_positions > self.config['max_low_liquidity_positions']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    type=AlertType.LIQUIDITY,
                    module=module_name,
                    title="Low Liquidity Positions",
                    message=f"{risk.low_liquidity_positions} positions have low liquidity (threshold: {self.config['max_low_liquidity_positions']})",
                    data={
                        'low_liquidity_positions': risk.low_liquidity_positions,
                        'avg_liquidity_score': risk.avg_liquidity_score,
                        'threshold': self.config['max_low_liquidity_positions']
                    }
                )

        except Exception as e:
            self.logger.error(f"Error checking liquidity: {e}")

    async def _check_anomalies(self, module_name: str, perf):
        """Check for anomalous performance"""
        try:
            history = self.performance_history.get(module_name, [])

            if len(history) >= self.config['anomaly_detection_window']:
                import numpy as np

                # Calculate z-score for current PnL
                current_pnl = float(perf.net_pnl)
                mean = np.mean(history)
                std = np.std(history)

                if std > 0:
                    z_score = abs((current_pnl - mean) / std)

                    if z_score > self.config['anomaly_std_threshold']:
                        await self._create_alert(
                            level=AlertLevel.WARNING,
                            type=AlertType.ANOMALY,
                            module=module_name,
                            title="Anomalous Performance Detected",
                            message=f"Current performance deviates {z_score:.1f} standard deviations from mean",
                            data={
                                'current_pnl': current_pnl,
                                'mean_pnl': mean,
                                'z_score': z_score,
                                'threshold': self.config['anomaly_std_threshold']
                            }
                        )

        except Exception as e:
            self.logger.error(f"Error checking anomalies: {e}")

    def _store_performance_data(self, module_name: str, perf):
        """Store performance data for anomaly detection"""
        try:
            if module_name not in self.performance_history:
                self.performance_history[module_name] = []

            self.performance_history[module_name].append(float(perf.net_pnl))

            # Keep only recent data
            max_window = self.config['anomaly_detection_window']
            if len(self.performance_history[module_name]) > max_window:
                self.performance_history[module_name] = self.performance_history[module_name][-max_window:]

        except Exception as e:
            self.logger.error(f"Error storing performance data: {e}")

    async def _create_alert(
        self,
        level: AlertLevel,
        type: AlertType,
        module: str,
        title: str,
        message: str,
        data: Dict
    ):
        """Create and send an alert"""
        try:
            # Check cooldown
            cooldown_key = f"{module}:{type.value}"
            if cooldown_key in self.alert_cooldowns:
                last_alert = self.alert_cooldowns[cooldown_key]
                if datetime.now() - last_alert < self.cooldown_period:
                    return  # Skip alert (cooldown)

            # Create alert
            alert = Alert(
                id=f"{module}:{type.value}:{datetime.now().timestamp()}",
                level=level,
                type=type,
                module=module,
                title=title,
                message=message,
                timestamp=datetime.now(),
                data=data
            )

            # Store alert
            self.alerts.append(alert)

            # Update cooldown
            self.alert_cooldowns[cooldown_key] = datetime.now()

            # Update count
            if module not in self.alert_count:
                self.alert_count[module] = 0
            self.alert_count[module] += 1

            # Send alert through alert manager
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    title=title,
                    message=message,
                    severity=level.value,
                    metadata=data
                )

            self.logger.warning(
                f"Alert created: [{level.value.upper()}] {module} - {title}"
            )

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}", exc_info=True)

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        return sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def get_alerts_by_module(self, module_name: str, limit: int = 10) -> List[Alert]:
        """Get alerts for a specific module"""
        module_alerts = [a for a in self.alerts if a.module == module_name]
        return sorted(module_alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        summary = {
            'total_alerts': len(self.alerts),
            'by_level': {
                'info': sum(1 for a in self.alerts if a.level == AlertLevel.INFO),
                'warning': sum(1 for a in self.alerts if a.level == AlertLevel.WARNING),
                'critical': sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL)
            },
            'by_type': {},
            'by_module': self.alert_count.copy(),
            'unacknowledged': sum(1 for a in self.alerts if not a.acknowledged)
        }

        # Count by type
        for type_enum in AlertType:
            count = sum(1 for a in self.alerts if a.type == type_enum)
            if count > 0:
                summary['by_type'][type_enum.value] = count

        return summary
