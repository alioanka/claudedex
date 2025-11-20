"""
Alerts System for DexScreener Trading Bot
Multi-channel notification system for critical events and performance updates
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import json
import aiohttp
from collections import deque, defaultdict
import hashlib
import hmac
from utils.helpers import retry_async

logger = logging.getLogger(__name__)

# Add this function at the top of alerts.py
def escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts"""
    # Trading alerts
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    TRAILING_STOP_UPDATED = "trailing_stop_updated"
    
    # Opportunity alerts
    HIGH_CONFIDENCE_SIGNAL = "high_confidence_signal"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    WHALE_MOVEMENT = "whale_movement"
    VOLUME_SURGE = "volume_surge"
    
    # Risk alerts
    HIGH_RISK_WARNING = "high_risk_warning"
    DRAWDOWN_ALERT = "drawdown_alert"
    CORRELATION_WARNING = "correlation_warning"
    POSITION_REVIEW = "position_review"
    MARGIN_CALL = "margin_call"
    
    # System alerts
    SYSTEM_ERROR = "system_error"
    CONNECTION_LOST = "connection_lost"
    API_LIMIT_WARNING = "api_limit_warning"
    LOW_BALANCE = "low_balance"
    
    # Performance alerts
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_REPORT = "weekly_report"
    MILESTONE_REACHED = "milestone_reached"
    NEW_HIGH_SCORE = "new_high_score"
    
    # Security alerts
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    HONEYPOT_DETECTED = "honeypot_detected"
    RUG_PULL_WARNING = "rug_pull_warning"
    CONTRACT_VULNERABILITY = "contract_vulnerability"

class NotificationChannel(Enum):
    """Notification channels"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    PUSHOVER = "pushover"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    sent_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: Callable
    alert_type: AlertType
    priority: AlertPriority
    channels: List[NotificationChannel]
    cooldown: int  # seconds
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChannelConfig:
    """Channel configuration"""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]
    rate_limit: int  # messages per minute
    last_sent: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0

class AlertsSystem:
    """
    Comprehensive alerts and notification system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize alerts system"""
        self.config = config or self._default_config()
        self.alerts_queue: asyncio.Queue = asyncio.Queue()
        self.alerts_history: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.channel_configs: Dict[NotificationChannel, ChannelConfig] = {}
        self.alert_subscribers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert aggregation
        self.alert_buffer: Dict[str, List[Alert]] = defaultdict(list)
        self.aggregation_enabled = self.config.get("aggregation_enabled", True)
        
        # Performance tracking
        self.metrics = {
            "total_alerts": 0,
            "sent_alerts": 0,
            "failed_alerts": 0,
            "by_type": defaultdict(int),
            "by_channel": defaultdict(int),
            "by_priority": defaultdict(int)
        }
        
        # Initialize channels
        self._initialize_channels()
        
        # Start alert processor
        asyncio.create_task(self._process_alerts())
        asyncio.create_task(self._aggregate_alerts())
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # General settings
            "enabled": True,
            "max_retries": 3,
            "retry_delay": 30,  # seconds
            "aggregation_window": 60,  # seconds
            "aggregation_enabled": True,
            
            # Priority settings
            "priority_cooldowns": {
                AlertPriority.LOW: 300,  # 5 minutes
                AlertPriority.MEDIUM: 60,  # 1 minute
                AlertPriority.HIGH: 10,  # 10 seconds
                AlertPriority.CRITICAL: 0  # No cooldown
            },
            
            # Channel priorities (which channels for which priority)
            "channel_priorities": {
                AlertPriority.LOW: [NotificationChannel.DISCORD],
                AlertPriority.MEDIUM: [NotificationChannel.TELEGRAM, NotificationChannel.DISCORD],
                AlertPriority.HIGH: [NotificationChannel.TELEGRAM, NotificationChannel.DISCORD, NotificationChannel.EMAIL],
                AlertPriority.CRITICAL: [NotificationChannel.TELEGRAM, NotificationChannel.SMS, NotificationChannel.WEBHOOK]
            },
            
            # Rate limits per channel (messages per minute)
            "rate_limits": {
                NotificationChannel.TELEGRAM: 30,
                NotificationChannel.DISCORD: 60,
                NotificationChannel.EMAIL: 10,
                NotificationChannel.SMS: 5,
                NotificationChannel.WEBHOOK: 100,
                NotificationChannel.SLACK: 30,
                NotificationChannel.PUSHOVER: 30
            },
            
            # Alert formatting
            "use_emoji": True,
            "use_markdown": True,
            "include_timestamp": True,
            "include_metrics": True,
            
            # Alert rules
            "default_rules": {
                "high_profit": {
                    "condition": lambda pnl: pnl > 100,
                    "alert_type": AlertType.MILESTONE_REACHED,
                    "priority": AlertPriority.HIGH
                },
                "high_loss": {
                    "condition": lambda pnl: pnl < -50,
                    "alert_type": AlertType.STOP_LOSS_HIT,
                    "priority": AlertPriority.HIGH
                },
                "drawdown": {
                    "condition": lambda dd: dd > 0.1,
                    "alert_type": AlertType.DRAWDOWN_ALERT,
                    "priority": AlertPriority.HIGH
                }
            }
        }
    
    def _initialize_channels(self):
        """Initialize notification channels"""
        # Get rate limits from config with fallback to defaults
        rate_limits = self.config.get("rate_limits", self._default_config()["rate_limits"])
        
        # Telegram
        if self.config.get("telegram"):
            self.channel_configs[NotificationChannel.TELEGRAM] = ChannelConfig(
                channel=NotificationChannel.TELEGRAM,
                enabled=True,
                config=self.config["telegram"],
                rate_limit=rate_limits[NotificationChannel.TELEGRAM]  # Use local variable
            )
        
        # Discord
        if self.config.get("discord"):
            self.channel_configs[NotificationChannel.DISCORD] = ChannelConfig(
                channel=NotificationChannel.DISCORD,
                enabled=True,
                config=self.config["discord"],
                rate_limit=rate_limits[NotificationChannel.DISCORD]
            )
        
        # Email
        if self.config.get("email"):
            self.channel_configs[NotificationChannel.EMAIL] = ChannelConfig(
                channel=NotificationChannel.EMAIL,
                enabled=True,
                config=self.config["email"],
                rate_limit=self.config["rate_limits"][NotificationChannel.EMAIL]
            )
        
        # Webhook
        if self.config.get("webhook"):
            self.channel_configs[NotificationChannel.WEBHOOK] = ChannelConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=True,
                config=self.config["webhook"],
                rate_limit=self.config["rate_limits"][NotificationChannel.WEBHOOK]
            )
        
        # SMS
        if self.config.get("sms"):
            self.channel_configs[NotificationChannel.SMS] = ChannelConfig(
                channel=NotificationChannel.SMS,
                enabled=True,
                config=self.config["sms"],
                rate_limit=self.config["rate_limits"][NotificationChannel.SMS]
            )

    ## alerts.py fixes:

    
    # Add to AlertsSystem class:

    async def send_alert(self, alert_type: str, message: str, data: Dict) -> None:
        """
        Send alert matching API specification
        
        Args:
            alert_type: Type of alert as string
            message: Alert message
            data: Additional alert data
        """
        # Map string alert_type to enum
        try:
            alert_type_enum = AlertType[alert_type.upper()]
        except KeyError:
            alert_type_enum = AlertType.SYSTEM_ERROR
        
        # Determine priority based on alert type
        priority = AlertPriority.MEDIUM
        if "error" in alert_type.lower() or "critical" in alert_type.lower():
            priority = AlertPriority.HIGH
        elif "warning" in alert_type.lower():
            priority = AlertPriority.MEDIUM
        
        # Call the existing send_alert with title derived from message
        title = message[:50] if len(message) > 50 else message
        
        await self.send_alert_internal(
            alert_type=alert_type_enum,
            title=title,
            message=message,
            priority=priority,
            data=data,
            channels=None  # Use default channels
        )

    async def send_alert_internal(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        data: Optional[Dict] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> str:
        """
        Send an alert
        
        Args:
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            priority: Alert priority
            data: Additional data
            channels: Specific channels to use
            
        Returns:
            Alert ID
        """
        try:
            # Create alert
            alert = Alert(
                alert_id=self._generate_alert_id(),
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                timestamp=datetime.utcnow(),
                data=data or {},
                channels=channels or self._get_channels_for_priority(priority)
            )
            
            # Check if should be sent (cooldown, deduplication)
            if not self._should_send_alert(alert):
                logger.debug(f"Alert {alert.alert_id} skipped due to cooldown/dedup")
                return alert.alert_id
            
            # Add to queue
            await self.alerts_queue.put(alert)
            
            # Update metrics
            self.metrics["total_alerts"] += 1
            self.metrics["by_type"][alert_type] += 1
            self.metrics["by_priority"][priority] += 1
            
            # Trigger subscribers
            await self._notify_subscribers(alert)
            
            return alert.alert_id
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return ""
    
    async def send_trading_alert(
        self,
        event_type: str,
        position: Dict,
        **kwargs
    ):
        """Send trading-specific alerts"""
        try:
            # Use send_alert_internal directly with simple message
            message = position.get('message', '')
            if not message:
                # Build message from position data
                message = f"Trade Alert: {event_type}"
                if 'token_symbol' in position:
                    message += f" - {position['token_symbol']}"
            
            await self.send_alert_internal(
                alert_type=AlertType.POSITION_OPENED,
                title=f"Trade: {event_type}",
                message=message,
                priority=AlertPriority.MEDIUM,
                data=position
            )
        except Exception as e:
            logger.error(f"Error sending trading alert: {e}")
    
    async def send_opportunity_alert(
        self,
        opportunity_type: str,
        data: Dict,
        confidence: float = 0.5
    ):
        """Send opportunity alerts"""
        try:
            # Determine priority based on confidence
            if confidence > 0.8:
                priority = AlertPriority.HIGH
            elif confidence > 0.6:
                priority = AlertPriority.MEDIUM
            else:
                priority = AlertPriority.LOW
            
            # Format based on opportunity type
            if opportunity_type == "arbitrage":
                alert_type = AlertType.ARBITRAGE_OPPORTUNITY
                title = "💎 Arbitrage Opportunity"
                message = self._format_arbitrage_opportunity(data)
            elif opportunity_type == "whale":
                alert_type = AlertType.WHALE_MOVEMENT
                title = "🐋 Whale Movement Detected"
                message = self._format_whale_movement(data)
            elif opportunity_type == "volume":
                alert_type = AlertType.VOLUME_SURGE
                title = "📈 Volume Surge Detected"
                message = self._format_volume_surge(data)
            elif opportunity_type == "signal":
                alert_type = AlertType.HIGH_CONFIDENCE_SIGNAL
                title = "🎯 High Confidence Signal"
                message = self._format_signal(data, confidence)
            else:
                alert_type = AlertType.HIGH_CONFIDENCE_SIGNAL
                title = "📊 Trading Opportunity"
                message = json.dumps(data, default=str)
            
            # Send alert
            await self.send_alert(
                alert_type=alert_type,
                title=title,
                message=message,
                priority=priority,
                data=data
            )
            
        except Exception as e:
            logger.error(f"Error sending opportunity alert: {e}")
    
    async def send_risk_alert(
        self,
        risk_type: str,
        severity: str,
        data: Dict
    ):
        """Send risk management alerts"""
        try:
            # Map severity to priority
            priority_map = {
                "low": AlertPriority.LOW,
                "medium": AlertPriority.MEDIUM,
                "high": AlertPriority.HIGH,
                "critical": AlertPriority.CRITICAL
            }
            priority = priority_map.get(severity, AlertPriority.MEDIUM)
            
            # Format based on risk type
            if risk_type == "drawdown":
                alert_type = AlertType.DRAWDOWN_ALERT
                title = "⚠️ Drawdown Alert"
                message = self._format_drawdown_alert(data)
            elif risk_type == "correlation":
                alert_type = AlertType.CORRELATION_WARNING
                title = "🔗 High Correlation Warning"
                message = self._format_correlation_warning(data)
            elif risk_type == "high_risk":
                alert_type = AlertType.HIGH_RISK_WARNING
                title = "🚨 High Risk Warning"
                message = self._format_risk_warning(data)
            elif risk_type == "margin":
                alert_type = AlertType.MARGIN_CALL
                title = "💰 Margin Call Warning"
                message = self._format_margin_call(data)
                priority = AlertPriority.CRITICAL
            else:
                alert_type = AlertType.HIGH_RISK_WARNING
                title = "⚠️ Risk Alert"
                message = json.dumps(data, default=str)
            
            # Send alert
            await self.send_alert(
                alert_type=alert_type,
                title=title,
                message=message,
                priority=priority,
                data=data
            )
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
    
    async def send_performance_summary(
        self,
        period: str,
        metrics: Dict
    ):
        """Send performance summary alerts"""
        try:
            if period == "daily":
                alert_type = AlertType.DAILY_SUMMARY
                title = "📊 Daily Performance Summary"
                message = self._format_daily_summary(metrics)
                priority = AlertPriority.LOW
            elif period == "weekly":
                alert_type = AlertType.WEEKLY_REPORT
                title = "📈 Weekly Performance Report"
                message = self._format_weekly_report(metrics)
                priority = AlertPriority.MEDIUM
            else:
                alert_type = AlertType.DAILY_SUMMARY
                title = f"📊 {period.title()} Summary"
                message = self._format_performance_metrics(metrics)
                priority = AlertPriority.LOW
            
            # Send alert
            await self.send_alert(
                alert_type=alert_type,
                title=title,
                message=message,
                priority=priority,
                data=metrics
            )
            
        except Exception as e:
            logger.error(f"Error sending performance summary: {e}")
    
    async def _process_alerts(self):
        """Background task to process alert queue"""
        while True:
            try:
                # Get alert from queue
                alert = await self.alerts_queue.get()
                
                # Check if aggregation is enabled
                if self.aggregation_enabled and alert.priority == AlertPriority.LOW:
                    # Add to buffer for aggregation
                    key = f"{alert.alert_type}_{alert.priority}"
                    self.alert_buffer[key].append(alert)
                else:
                    # Send immediately
                    await self._send_alert_to_channels(alert)
                
                # Add to history
                self.alerts_history.append(alert)
                
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                await asyncio.sleep(1)
    
    async def _aggregate_alerts(self):
        """Aggregate low-priority alerts"""
        while True:
            try:
                await asyncio.sleep(self.config.get("aggregation_window", 60))
                
                if not self.alert_buffer:
                    continue
                
                # Process each buffer
                for key, alerts in list(self.alert_buffer.items()):
                    if not alerts:
                        continue
                    
                    # Create aggregated alert
                    alert_type = alerts[0].alert_type
                    priority = alerts[0].priority
                    
                    title = f"📦 {len(alerts)} {alert_type.value} Alerts"
                    message = self._format_aggregated_alerts(alerts)
                    
                    aggregated = Alert(
                        alert_id=self._generate_alert_id(),
                        alert_type=alert_type,
                        priority=priority,
                        title=title,
                        message=message,
                        timestamp=datetime.utcnow(),
                        data={"alerts": [a.data for a in alerts]},
                        channels=alerts[0].channels,
                        metadata={"aggregated": True, "count": len(alerts)}
                    )
                    
                    # Send aggregated alert
                    await self._send_alert_to_channels(aggregated)
                    
                    # Clear buffer
                    self.alert_buffer[key] = []
                
            except Exception as e:
                logger.error(f"Error aggregating alerts: {e}")
    
    async def _send_alert_to_channels(self, alert: Alert):
        """Send alert to configured channels"""
        try:
            success_count = 0
            
            for channel in alert.channels:
                if channel not in self.channel_configs:
                    continue
                
                config = self.channel_configs[channel]
                
                if not config.enabled:
                    continue
                
                # Check rate limit
                if not self._check_rate_limit(channel):
                    logger.warning(f"Rate limit exceeded for {channel.value}")
                    continue
                
                # Send to channel
                success = await self._send_to_channel(alert, channel)
                
                if success:
                    success_count += 1
                    config.message_count += 1
                    config.last_sent = datetime.utcnow()
                    self.metrics["by_channel"][channel] += 1
                else:
                    config.error_count += 1
                    
                    # Retry if configured
                    if alert.retry_count < self.config["max_retries"]:
                        alert.retry_count += 1
                        await asyncio.sleep(self.config["retry_delay"])
                        await self.alerts_queue.put(alert)
            
            # Update alert status
            if success_count > 0:
                alert.sent = True
                alert.sent_at = datetime.utcnow()
                self.metrics["sent_alerts"] += 1
            else:
                self.metrics["failed_alerts"] += 1
                
        except Exception as e:
            logger.error(f"Error sending alert to channels: {e}")
    
    async def _send_to_channel(
        self,
        alert: Alert,
        channel: NotificationChannel
    ) -> bool:
        """Send alert to specific channel"""
        try:
            if channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram(alert)
            elif channel == NotificationChannel.DISCORD:
                return await self._send_discord(alert)
            elif channel == NotificationChannel.EMAIL:
                return await self._send_email(alert)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(alert)
            elif channel == NotificationChannel.SMS:
                return await self._send_sms(alert)
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack(alert)
            elif channel == NotificationChannel.PUSHOVER:
                return await self._send_pushover(alert)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error sending to {channel.value}: {e}")
            return False

    # Rename existing send_alert to send_alert_internal
    # Then add public methods for channels:

    async def send_telegram(self, message: str) -> bool:
        """
        Send message to Telegram
        
        Args:
            message: Message to send
            
        Returns:
            Success status
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.SYSTEM_ERROR,
            priority=AlertPriority.MEDIUM,
            title="Telegram Message",
            message=message,
            timestamp=datetime.utcnow(),
            channels=[NotificationChannel.TELEGRAM]
        )
        return await self._send_telegram(alert)

    async def send_discord(self, message: str) -> bool:
        """
        Send message to Discord
        
        Args:
            message: Message to send
            
        Returns:
            Success status
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.SYSTEM_ERROR,
            priority=AlertPriority.MEDIUM,
            title="Discord Message",
            message=message,
            timestamp=datetime.utcnow(),
            channels=[NotificationChannel.DISCORD]
        )
        return await self._send_discord(alert)

    async def send_email(self, subject: str, body: str) -> bool:
        """
        Send email
        
        Args:
            subject: Email subject
            body: Email body
            
        Returns:
            Success status
        """
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.SYSTEM_ERROR,
            priority=AlertPriority.MEDIUM,
            title=subject,
            message=body,
            timestamp=datetime.utcnow(),
            channels=[NotificationChannel.EMAIL]
        )
        return await self._send_email(alert)

    def format_alert(self, alert_type: str, data: Dict) -> str:
        """
        Format alert message
        
        Args:
            alert_type: Type of alert
            data: Alert data
            
        Returns:
            Formatted message
        """
        # Use appropriate formatter based on alert type
        if alert_type == "position_opened":
            return self._format_position_opened(data)
        elif alert_type == "position_closed":
            return self._format_position_closed(data, data.get('pnl'))
        elif alert_type == "stop_loss":
            return self._format_stop_loss(data, data.get('loss'))
        elif alert_type == "take_profit":
            return self._format_take_profit(data, data.get('profit'))
        elif alert_type == "arbitrage":
            return self._format_arbitrage_opportunity(data)
        elif alert_type == "whale":
            return self._format_whale_movement(data)
        elif alert_type == "volume":
            return self._format_volume_surge(data)
        elif alert_type == "signal":
            return self._format_signal(data, data.get('confidence', 0.5))
        elif alert_type == "drawdown":
            return self._format_drawdown_alert(data)
        elif alert_type == "risk":
            return self._format_risk_warning(data)
        elif alert_type == "daily_summary":
            return self._format_daily_summary(data)
        else:
            # Generic formatting
            lines = [f"Alert: {alert_type}"]
            for key, value in data.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines)

    @retry_async(max_retries=3, delay=5, exponential_backoff=True)
    async def _send_telegram(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        try:
            config = self.channel_configs.get(NotificationChannel.TELEGRAM)
            if not config or not config.enabled:
                return False
            
            telegram_config = config.config
            bot_token = telegram_config.get('telegram_bot_token') or telegram_config.get('bot_token')
            chat_id = telegram_config.get('telegram_chat_id') or telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                logger.warning("Telegram credentials not configured. Skipping alert.")
                return False
            
            # Build message with proper escaping
            timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            emoji_map = {
                AlertPriority.LOW: "ℹ️",
                AlertPriority.MEDIUM: "⚠️",
                AlertPriority.HIGH: "🚨",
                AlertPriority.CRITICAL: "🔥"
            }
            
            emoji = emoji_map.get(alert.priority, "📢")
            
            # Escape BEFORE adding markdown formatting
            title_escaped = escape_markdown(alert.title)
            message_escaped = escape_markdown(alert.message)
            timestamp_escaped = escape_markdown(timestamp_str)
            
            # Now build message with MarkdownV2 formatting
            message = f"{emoji} *{title_escaped}*\n"
            message += f"⏰ {timestamp_escaped}\n"
            message += f"📝 {message_escaped}\n"
            
            if alert.metadata:
                message += "\n_Details:_\n"
                for key, value in list(alert.metadata.items())[:5]:
                    key_escaped = escape_markdown(str(key))
                    value_escaped = escape_markdown(str(value))
                    message += f"• {key_escaped}: {value_escaped}\n"
            
            # Send to Telegram with MarkdownV2
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'MarkdownV2'  # Changed to MarkdownV2
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False
                        
        except aiohttp.ClientError as e:
            logger.error(f"Telegram send error (network issue): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Telegram send error (unexpected): {e}", exc_info=True)
            return False
    
    async def _send_discord(self, alert: Alert) -> bool:
        """Send alert to Discord"""
        try:
            config = self.channel_configs[NotificationChannel.DISCORD].config
            webhook_url = config.get("webhook_url")
            
            if not webhook_url:
                return False
            
            # Create embed
            embed = {
                "title": alert.title,
                "description": alert.message,
                "color": self._get_color_for_priority(alert.priority),
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": f"Alert Type: {alert.alert_type.value}"}
            }
            
            # Add fields from data
            if alert.data:
                fields = []
                for key, value in list(alert.data.items())[:5]:  # Limit to 5 fields
                    fields.append({
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:100],  # Limit length
                        "inline": True
                    })
                embed["fields"] = fields
            
            payload = {"embeds": [embed]}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 204
                    
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False
    
    async def _send_email(self, alert: Alert) -> bool:
        """Send alert via email"""
        # Implementation would use SMTP or email service API
        return False
    
    async def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to webhook"""
        try:
            config = self.channel_configs[NotificationChannel.WEBHOOK].config
            url = config.get("url")
            secret = config.get("secret")
            
            if not url:
                return False
            
            # Prepare payload
            payload = {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "priority": alert.priority.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data
            }
            
            # Add signature if secret provided
            headers = {}
            if secret:
                signature = hmac.new(
                    secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Signature"] = signature
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    return response.status in [200, 201, 202, 204]
                    
        except Exception as e:
            logger.error(f"Webhook send error: {e}")
            return False
    
    async def _send_sms(self, alert: Alert) -> bool:
        """Send alert via SMS"""
        # Implementation would use Twilio or similar service
        return False
    
    async def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        # Implementation would use Slack webhook
        return False
    
    async def _send_pushover(self, alert: Alert) -> bool:
        """Send alert via Pushover"""
        # Implementation would use Pushover API
        return False
    
    def _get_emoji(self, alert_type: AlertType) -> str:
        """Get emoji for alert type"""
        # Safe config access with default
        if not self.config.get("use_emoji", True):
            return ""
        
        emoji_map = {
            AlertType.POSITION_OPENED: "🟢",
            AlertType.POSITION_CLOSED: "🔴",
            AlertType.STOP_LOSS_HIT: "⛔",
            AlertType.TAKE_PROFIT_HIT: "✅",
            AlertType.HIGH_CONFIDENCE_SIGNAL: "🎯",
            AlertType.ARBITRAGE_OPPORTUNITY: "💎",
            AlertType.WHALE_MOVEMENT: "🐋",
            AlertType.VOLUME_SURGE: "📈",
            AlertType.HIGH_RISK_WARNING: "⚠️",
            AlertType.DRAWDOWN_ALERT: "📉",
            AlertType.SYSTEM_ERROR: "❌",
            AlertType.DAILY_SUMMARY: "📊",
            AlertType.MILESTONE_REACHED: "🏆",
            AlertType.HONEYPOT_DETECTED: "🍯",
            AlertType.RUG_PULL_WARNING: "🚨"
        }
        
        return emoji_map.get(alert_type, "📢")
    
    def _get_color_for_priority(self, priority: AlertPriority) -> int:
        """Get color code for Discord embed"""
        color_map = {
            AlertPriority.LOW: 0x808080,     # Gray
            AlertPriority.MEDIUM: 0x0099ff,   # Blue
            AlertPriority.HIGH: 0xffaa00,     # Orange
            AlertPriority.CRITICAL: 0xff0000  # Red
        }
        return color_map.get(priority, 0x808080)
    
    def _get_channels_for_priority(
        self,
        priority: AlertPriority
    ) -> List[NotificationChannel]:
        """Get appropriate channels for priority level"""
        try:
            # Get channel priorities from config
            channel_priorities = self.config.get("channel_priorities")
            
            # Handle different config formats
            if channel_priorities is None:
                logger.warning("No channel_priorities configured")
                return []
            
            # If it's a list (misconfigured), return empty
            if isinstance(channel_priorities, list):
                logger.warning(f"channel_priorities is a list, expected dict. Returning empty channels.")
                return []
            
            # If it's not a dict, return empty
            if not isinstance(channel_priorities, dict):
                logger.warning(f"Invalid channel_priorities type: {type(channel_priorities)}")
                return []
            
            # Get channels for this priority
            channels = channel_priorities.get(priority, [])
            
            # Ensure channels is a list
            if not isinstance(channels, list):
                logger.warning(f"Channels for {priority} is not a list: {type(channels)}")
                return []
            
            # Filter to only enabled and configured channels
            enabled_channels = []
            for ch in channels:
                if ch in self.channel_configs and self.channel_configs[ch].enabled:
                    enabled_channels.append(ch)
            
            return enabled_channels
            
        except Exception as e:
            logger.error(f"Error getting channels for priority {priority}: {e}")
            return []
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent (cooldown, dedup)"""
        try:
            # Check cooldown for this alert type with safe fallback
            priority_cooldowns = self.config.get("priority_cooldowns", {})
            cooldown = priority_cooldowns.get(alert.priority, 0)
            
            if cooldown > 0:
                # Check last sent time for this type
                for past_alert in reversed(self.alerts_history):
                    if past_alert.alert_type == alert.alert_type and past_alert.sent:
                        time_diff = (alert.timestamp - past_alert.timestamp).total_seconds()
                        if time_diff < cooldown:
                            return False
                        break
            
            return True
            
        except Exception as e:
            logger.error(f"Error in _should_send_alert: {e}")
            return True  # Allow sending on error to avoid blocking alerts
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if channel rate limit allows sending"""
        config = self.channel_configs.get(channel)
        if not config:
            return False
        
        # Get recent sends for this channel
        key = f"{channel.value}_rate"
        recent_sends = self.rate_limiters[key]
        
        # Clean old entries (older than 1 minute)
        current_time = datetime.utcnow()
        while recent_sends and (current_time - recent_sends[0]).total_seconds() > 60:
            recent_sends.popleft()
        
        # Check if under limit
        if len(recent_sends) >= config.rate_limit:
            return False
        
        # Add current send
        recent_sends.append(current_time)
        return True
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"ALERT-{timestamp}"
    
    def _format_position_opened(self, position: Dict) -> str:
        """Format position opened message"""
        return (
            f"**Token:** {position.get('token_symbol', 'Unknown')}\n"
            f"**Type:** {position.get('position_type', 'Unknown')}\n"
            f"**Entry Price:** ${position.get('entry_price', 0):.6f}\n"
            f"**Amount:** {position.get('entry_amount', 0):.4f}\n"
            f"**Value:** ${position.get('entry_value', 0):.2f}\n"
            f"**Stop Loss:** ${position.get('stop_loss', 0):.6f}\n"
            f"**Strategy:** {position.get('strategy_name', 'Manual')}\n"
            f"**Confidence:** {position.get('confidence_score', 0):.1%}"
        )
    
    def _format_position_closed(self, position: Dict, pnl: Optional[Decimal]) -> str:
        """Format position closed message"""
        pnl_str = f"${pnl:.2f}" if pnl else "Unknown"
        pnl_emoji = "🟢" if pnl and pnl > 0 else "🔴"
        
        return (
            f"**Token:** {position.get('token_symbol', 'Unknown')}\n"
            f"**Exit Price:** ${position.get('exit_price', 0):.6f}\n"
            f"**P&L:** {pnl_emoji} {pnl_str}\n"
            f"**ROI:** {position.get('roi', 0):.2%}\n"
            f"**Duration:** {position.get('holding_period', 'Unknown')}\n"
            f"**Exit Reason:** {position.get('exit_reason', 'Manual')}"
        )
    
    def _format_stop_loss(self, position: Dict, loss: Optional[Decimal]) -> str:
        """Format stop loss hit message"""
        return (
            f"⛔ **STOP LOSS TRIGGERED** ⛔\n\n"
            f"**Token:** {position.get('token_symbol', 'Unknown')}\n"
            f"**Entry:** ${position.get('entry_price', 0):.6f}\n"
            f"**Stop:** ${position.get('stop_loss', 0):.6f}\n"
            f"**Loss:** -${abs(loss) if loss else 0:.2f}\n"
            f"**Reason:** Protecting capital from further losses"
        )
    
    def _format_take_profit(self, position: Dict, profit: Optional[Decimal]) -> str:
        """Format take profit hit message"""
        return (
            f"✅ **TAKE PROFIT REACHED** ✅\n\n"
            f"**Token:** {position.get('token_symbol', 'Unknown')}\n"
            f"**Entry:** ${position.get('entry_price', 0):.6f}\n"
            f"**Target:** ${position.get('take_profit', 0):.6f}\n"
            f"**Profit:** +${profit if profit else 0:.2f}\n"
            f"**Level:** {position.get('tp_level', 1)}"
        )
    
    def _format_arbitrage_opportunity(self, data: Dict) -> str:
        """Format arbitrage opportunity message"""
        return (
            f"💎 **ARBITRAGE DETECTED** 💎\n\n"
            f"**Token:** {data.get('token', 'Unknown')}\n"
            f"**Buy DEX:** {data.get('buy_dex', 'Unknown')} @ ${data.get('buy_price', 0):.6f}\n"
            f"**Sell DEX:** {data.get('sell_dex', 'Unknown')} @ ${data.get('sell_price', 0):.6f}\n"
            f"**Spread:** {data.get('spread', 0):.2%}\n"
            f"**Est. Profit:** ${data.get('estimated_profit', 0):.2f}\n"
            f"**Gas Cost:** ${data.get('gas_cost', 0):.2f}\n"
            f"**Net Profit:** ${data.get('net_profit', 0):.2f}"
        )
    
    def _format_whale_movement(self, data: Dict) -> str:
        """Format whale movement message"""
        action = "BUYING" if data.get('action') == 'buy' else "SELLING"
        return (
            f"🐋 **WHALE {action}** 🐋\n\n"
            f"**Token:** {data.get('token', 'Unknown')}\n"
            f"**Amount:** ${data.get('amount', 0):,.2f}\n"
            f"**Price Impact:** {data.get('price_impact', 0):.2%}\n"
            f"**Wallet:** `{data.get('wallet', 'Unknown')[:10]}...`\n"
            f"**Whale Score:** {data.get('whale_score', 0):.1f}/10\n"
            f"**Historical Performance:** {data.get('performance', 0):.1%}"
        )
    
    def _format_volume_surge(self, data: Dict) -> str:
        """Format volume surge message"""
        return (
            f"📈 **VOLUME SURGE** 📈\n\n"
            f"**Token:** {data.get('token', 'Unknown')}\n"
            f"**Current Volume:** ${data.get('current_volume', 0):,.0f}\n"
            f"**24h Avg:** ${data.get('avg_volume', 0):,.0f}\n"
            f"**Surge Ratio:** {data.get('surge_ratio', 0):.1f}x\n"
            f"**Price Change:** {data.get('price_change', 0):+.2%}\n"
            f"**Buy/Sell Ratio:** {data.get('buy_sell_ratio', 0):.2f}"
        )
    
    def _format_signal(self, data: Dict, confidence: float) -> str:
        """Format trading signal message"""
        return (
            f"🎯 **HIGH CONFIDENCE SIGNAL** 🎯\n\n"
            f"**Token:** {data.get('token', 'Unknown')}\n"
            f"**Signal Type:** {data.get('signal_type', 'Unknown')}\n"
            f"**Action:** {data.get('action', 'BUY')}\n"
            f"**Entry:** ${data.get('entry_price', 0):.6f}\n"
            f"**Target:** ${data.get('target_price', 0):.6f}\n"
            f"**Stop:** ${data.get('stop_loss', 0):.6f}\n"
            f"**Confidence:** {confidence:.1%}\n"
            f"**Risk/Reward:** {data.get('risk_reward', 0):.2f}"
        )
    
    def _format_drawdown_alert(self, data: Dict) -> str:
        """Format drawdown alert message"""
        return (
            f"📉 **DRAWDOWN WARNING** 📉\n\n"
            f"**Current Drawdown:** {data.get('drawdown', 0):.2%}\n"
            f"**Peak Value:** ${data.get('peak_value', 0):,.2f}\n"
            f"**Current Value:** ${data.get('current_value', 0):,.2f}\n"
            f"**Loss:** -${data.get('loss_amount', 0):,.2f}\n"
            f"**Recovery Needed:** {data.get('recovery_needed', 0):.2%}\n"
            f"**Recommendation:** {data.get('recommendation', 'Review positions')}"
        )
    
    def _format_correlation_warning(self, data: Dict) -> str:
        """Format correlation warning message"""
        return (
            f"🔗 **HIGH CORRELATION** 🔗\n\n"
            f"**Tokens:** {data.get('token1', '')} ↔️ {data.get('token2', '')}\n"
            f"**Correlation:** {data.get('correlation', 0):.2f}\n"
            f"**Combined Exposure:** ${data.get('exposure', 0):,.2f}\n"
            f"**Risk Level:** {data.get('risk_level', 'Medium')}\n"
            f"**Suggestion:** Consider reducing position in one token"
        )
    
    def _format_risk_warning(self, data: Dict) -> str:
        """Format risk warning message"""
        return (
            f"⚠️ **RISK WARNING** ⚠️\n\n"
            f"**Risk Score:** {data.get('risk_score', 0):.1f}/100\n"
            f"**VaR (95%):** ${data.get('var_95', 0):,.2f}\n"
            f"**Portfolio Beta:** {data.get('beta', 0):.2f}\n"
            f"**Volatility:** {data.get('volatility', 0):.2%}\n"
            f"**Exposed Capital:** ${data.get('exposed', 0):,.2f}\n"
            f"**Safe Capital:** ${data.get('safe', 0):,.2f}"
        )
    
    def _format_margin_call(self, data: Dict) -> str:
        """Format margin call message"""
        return (
            f"🚨 **MARGIN CALL** 🚨\n\n"
            f"**URGENT ACTION REQUIRED**\n\n"
            f"**Current Margin:** {data.get('margin_level', 0):.2%}\n"
            f"**Required Margin:** {data.get('required_margin', 0):.2%}\n"
            f"**Deficit:** ${data.get('deficit', 0):,.2f}\n"
            f"**Time to Act:** {data.get('time_limit', 'Immediate')}\n"
            f"**Options:**\n"
            f"1. Add ${data.get('deposit_needed', 0):,.2f} funds\n"
            f"2. Close positions worth ${data.get('close_needed', 0):,.2f}"
        )
    
    def _format_daily_summary(self, metrics: Dict) -> str:
        """Format daily summary message"""
        pnl = metrics.get('daily_pnl', 0)
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        return (
            f"📊 **DAILY SUMMARY** 📊\n"
            f"_{datetime.utcnow().strftime('%Y-%m-%d')}_\n\n"
            f"**P&L:** {pnl_emoji} ${pnl:+,.2f} ({metrics.get('roi', 0):+.2%})\n"
            f"**Trades:** {metrics.get('total_trades', 0)}\n"
            f"**Win Rate:** {metrics.get('win_rate', 0):.1%}\n"
            f"**Best Trade:** ${metrics.get('best_trade', 0):+,.2f}\n"
            f"**Worst Trade:** ${metrics.get('worst_trade', 0):+,.2f}\n"
            f"**Open Positions:** {metrics.get('open_positions', 0)}\n"
            f"**Portfolio Value:** ${metrics.get('portfolio_value', 0):,.2f}\n"
            f"**Daily Volume:** ${metrics.get('volume', 0):,.2f}\n"
            f"**Sharpe Ratio:** {metrics.get('sharpe', 0):.2f}"
        )
    
    def _format_weekly_report(self, metrics: Dict) -> str:
        """Format weekly report message"""
        return (
            f"📈 **WEEKLY REPORT** 📈\n"
            f"_Week of {(datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')}_\n\n"
            f"**Performance**\n"
            f"• Total P&L: ${metrics.get('weekly_pnl', 0):+,.2f}\n"
            f"• ROI: {metrics.get('weekly_roi', 0):+.2%}\n"
            f"• Sharpe: {metrics.get('sharpe', 0):.2f}\n\n"
            f"**Trading Stats**\n"
            f"• Total Trades: {metrics.get('total_trades', 0)}\n"
            f"• Win Rate: {metrics.get('win_rate', 0):.1%}\n"
            f"• Avg Win: ${metrics.get('avg_win', 0):.2f}\n"
            f"• Avg Loss: ${metrics.get('avg_loss', 0):.2f}\n"
            f"• Profit Factor: {metrics.get('profit_factor', 0):.2f}\n\n"
            f"**Risk Metrics**\n"
            f"• Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
            f"• VaR (95%): ${metrics.get('var_95', 0):,.2f}\n"
            f"• Beta: {metrics.get('beta', 0):.2f}\n\n"
            f"**Top Performers**\n"
            + "\n".join([f"• {t['symbol']}: {t['pnl']:+.2%}" 
                        for t in metrics.get('top_trades', [])[:3]])
        )
    
    def _format_performance_metrics(self, metrics: Dict) -> str:
        """Format generic performance metrics"""
        lines = ["**Performance Metrics**\n"]
        for key, value in metrics.items():
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, (int, float)):
                if "percent" in key or "rate" in key or "roi" in key:
                    lines.append(f"• {formatted_key}: {value:.2%}")
                elif "ratio" in key:
                    lines.append(f"• {formatted_key}: {value:.2f}")
                else:
                    lines.append(f"• {formatted_key}: ${value:,.2f}")
            else:
                lines.append(f"• {formatted_key}: {value}")
        return "\n".join(lines)
    
    def _format_aggregated_alerts(self, alerts: List[Alert]) -> str:
        """Format aggregated alerts"""
        if not alerts:
            return "No alerts"
        
        # Group by type
        by_type = defaultdict(list)
        for alert in alerts:
            by_type[alert.alert_type].append(alert)
        
        lines = [f"**{len(alerts)} Aggregated Alerts**\n"]
        
        for alert_type, type_alerts in by_type.items():
            lines.append(f"\n**{alert_type.value} ({len(type_alerts)})**")
            for alert in type_alerts[:3]:  # Show max 3 per type
                lines.append(f"• {alert.title}: {alert.message[:50]}...")
        
        return "\n".join(lines)
    
    async def _notify_subscribers(self, alert: Alert):
        """Notify alert subscribers"""
        subscribers = self.alert_subscribers.get(alert.alert_type, [])
        for callback in subscribers:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe_to_alerts(
        self,
        alert_type: AlertType,
        callback: Callable
    ):
        """Subscribe to specific alert type"""
        self.alert_subscribers[alert_type].append(callback)
    
    def add_rule(
        self,
        name: str,
        condition: Callable,
        alert_type: AlertType,
        priority: AlertPriority = AlertPriority.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None,
        cooldown: int = 60
    ) -> str:
        """Add custom alert rule"""
        rule = AlertRule(
            rule_id=f"RULE-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            name=name,
            condition=condition,
            alert_type=alert_type,
            priority=priority,
            channels=channels or self._get_channels_for_priority(priority),
            cooldown=cooldown
        )
        
        self.alert_rules[rule.rule_id] = rule
        return rule.rule_id
    
    async def check_rules(self, data: Dict):
        """Check alert rules against data"""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if rule.last_triggered:
                    elapsed = (datetime.utcnow() - rule.last_triggered).total_seconds()
                    if elapsed < rule.cooldown:
                        continue
                
                # Check condition
                if rule.condition(data):
                    # Trigger alert
                    await self.send_alert(
                        alert_type=rule.alert_type,
                        title=f"Rule Triggered: {rule.name}",
                        message=f"Condition met for rule {rule.name}",
                        priority=rule.priority,
                        data=data,
                        channels=rule.channels
                    )
                    
                    # Update rule
                    rule.last_triggered = datetime.utcnow()
                    rule.trigger_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking rule {rule.name}: {e}")
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        return {
            "metrics": self.metrics,
            "channels": {
                channel.value: {
                    "enabled": config.enabled,
                    "messages_sent": config.message_count,
                    "errors": config.error_count,
                    "last_sent": config.last_sent.isoformat() if config.last_sent else None
                }
                for channel, config in self.channel_configs.items()
            },
            "rules": {
                rule_id: {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "triggers": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule_id, rule in self.alert_rules.items()
            },
            "recent_alerts": [
                {
                    "id": alert.alert_id,
                    "type": alert.alert_type.value,
                    "priority": alert.priority.value,
                    "title": alert.title,
                    "sent": alert.sent,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in list(self.alerts_history)[-10:]
            ]
        }
    
    def set_channel_enabled(
        self,
        channel: NotificationChannel,
        enabled: bool
    ):
        """Enable or disable a notification channel"""
        if channel in self.channel_configs:
            self.channel_configs[channel].enabled = enabled
    
    def update_channel_config(
        self,
        channel: NotificationChannel,
        config: Dict
    ):
        """Update channel configuration"""
        if channel in self.channel_configs:
            self.channel_configs[channel].config.update(config)
        else:
            self.channel_configs[channel] = ChannelConfig(
                channel=channel,
                enabled=True,
                config=config,
                rate_limit=self.config["rate_limits"].get(channel, 30)
            )

# Add this to monitoring/alerts.py if AlertManager class is missing:

# Add this corrected AlertManager class to monitoring/alerts.py 
# (Replace the existing incomplete AlertManager at the end of the file)

class AlertManager:
    """Alert manager wrapper"""
    
    def __init__(self, config: Dict = None):
        """Initialize with optional config"""
        self.config = config or {}
        self.alerts_system = AlertsSystem(config)
    
    async def initialize(self):
        """Initialize alert manager"""
        # Any initialization needed
        pass
        
    async def send_alert(self, alert_type: str, message: str, priority: str = 'medium'):
        """Send generic alert"""
        priority_map = {
            'low': AlertPriority.LOW,
            'medium': AlertPriority.MEDIUM,
            'high': AlertPriority.HIGH,
            'critical': AlertPriority.CRITICAL
        }
        
        await self.alerts_system.send_alert_internal(
            alert_type=AlertType.SYSTEM_ERROR,
            title=message[:50],
            message=message,
            priority=priority_map.get(priority, AlertPriority.MEDIUM),
            data={}
        )
    
    async def send_critical(self, message: str):
        """Send critical alert"""
        await self.send_alert('critical', message, 'critical')
    
    async def send_error(self, message: str):
        """Send error alert"""
        await self.send_alert('error', message, 'high')
    
    async def send_warning(self, message: str):
        """Send warning alert"""
        await self.send_alert('warning', message, 'medium')
    
    async def send_info(self, message: str):
        """Send info alert"""
        await self.send_alert('info', message, 'low')
    
    async def send_trade_alert(self, message: str):
        """Send trade alert"""
        await self.alerts_system.send_trading_alert(
            event_type="position_opened",
            position={'message': message}
        )
    
    async def send_performance_summary(self, period: str, metrics: Dict):
        """Send performance summary"""
        await self.alerts_system.send_performance_summary(period, metrics)