"""
Security Audit Logger for DexScreener Trading Bot
Comprehensive logging system for security events, compliance, and forensic analysis
"""

import os
import json
import asyncio
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import gzip
import uuid
from pathlib import Path

import aiofiles
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TRANSACTION = "transaction"
    WALLET_OPERATION = "wallet_operation"
    API_ACCESS = "api_access"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_INCIDENT = "security_incident"
    SYSTEM_EVENT = "system_event"
    COMPLIANCE = "compliance"
    ERROR = "error"
    WARNING = "warning"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(Enum):
    """Status of audit events"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    BLOCKED = "blocked"

@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    status: AuditStatus
    source: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    risk_score: float
    tags: List[str]
    correlation_id: Optional[str]
    parent_event_id: Optional[str]
    checksum: Optional[str]

class AuditLogger:
    """
    Comprehensive security audit logging system with:
    - Tamper-proof logging with checksums
    - Real-time event streaming
    - Compliance reporting
    - Forensic analysis capabilities
    - Performance monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_directory = Path(config.get('log_directory', './logs/audit'))
        self.max_log_size = config.get('max_log_size', 100 * 1024 * 1024)  # 100MB
        self.retention_days = config.get('retention_days', 365)
        self.compression_enabled = config.get('compression_enabled', True)
        self.real_time_alerts = config.get('real_time_alerts', True)
        
        # Security settings
        self.hmac_key = config.get('hmac_key', os.urandom(32))
        self.encrypt_logs = config.get('encrypt_logs', True)
        
        # Performance settings
        self.batch_size = config.get('batch_size', 100)
        self.flush_interval = config.get('flush_interval', 5)  # seconds
        
        # Initialize logging components
        self.event_buffer: List[AuditEvent] = []
        self.event_handlers: Dict[AuditEventType, List[callable]] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.risk_patterns: Dict[str, Dict] = {}
        self.compliance_rules: List[Dict] = []
        
        # Metrics tracking
        self.metrics = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'failed_events': 0,
            'last_flush_time': datetime.utcnow()
        }
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Background tasks
        self._flush_task = None
        self._cleanup_task = None
        
        logger.info("AuditLogger initialized")

    async def initialize(self) -> None:
        """Initialize audit logger with background tasks"""
        try:
            await self._load_compliance_rules()
            await self._load_risk_patterns()
            await self._start_background_tasks()
            
            # Log initialization
            await self.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.INFO,
                status=AuditStatus.SUCCESS,
                source="audit_logger",
                action="initialize",
                resource="audit_system",
                details={"version": "1.0", "config": self.config}
            )
            
            logger.info("Audit logger initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit logger: {e}")
            raise

    async def _load_compliance_rules(self) -> None:
        """Load compliance rules for automated checking"""
        self.compliance_rules = [
            {
                'rule_id': 'financial_transaction_limit',
                'description': 'Monitor large financial transactions',
                'condition': lambda event: event.event_type == AuditEventType.TRANSACTION and 
                           event.details.get('amount', 0) > 10000,
                'action': 'alert_compliance_team'
            },
            {
                'rule_id': 'failed_authentication_pattern',
                'description': 'Monitor repeated authentication failures',
                'condition': lambda event: event.event_type == AuditEventType.AUTHENTICATION and 
                           event.status == AuditStatus.FAILURE,
                'action': 'track_failed_attempts'
            },
            {
                'rule_id': 'unusual_api_access',
                'description': 'Monitor unusual API access patterns',
                'condition': lambda event: event.event_type == AuditEventType.API_ACCESS and 
                           event.risk_score > 0.7,
                'action': 'security_review'
            }
        ]

    async def _load_risk_patterns(self) -> None:
        """Load risk detection patterns"""
        self.risk_patterns = {
            'suspicious_login': {
                'pattern': 'multiple_failed_logins',
                'threshold': 5,
                'time_window': 300,  # 5 minutes
                'risk_score': 0.8
            },
            'unusual_transaction_volume': {
                'pattern': 'high_volume_transactions',
                'threshold': 100000,  # $100k
                'time_window': 3600,  # 1 hour
                'risk_score': 0.9
            },
            'off_hours_activity': {
                'pattern': 'activity_outside_business_hours',
                'business_hours': (9, 17),  # 9 AM to 5 PM
                'risk_score': 0.6
            }
        }

    async def _start_background_tasks(self) -> None:
        """Start background tasks for log flushing and cleanup"""
        self._flush_task = asyncio.create_task(self._flush_buffer_periodically())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_logs_periodically())

    async def _flush_buffer_periodically(self) -> None:
        """Periodically flush event buffer to disk"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush buffer task: {e}")

    async def _cleanup_old_logs_periodically(self) -> None:
        """Periodically cleanup old log files"""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Daily cleanup
                await self._cleanup_old_logs()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def log_event(self,
                       event_type: AuditEventType,
                       severity: AuditSeverity,
                       status: AuditStatus,
                       source: str,
                       action: str,
                       resource: str,
                       details: Dict[str, Any],
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       correlation_id: Optional[str] = None,
                       parent_event_id: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> str:
        """
        Log a security audit event
        
        Returns:
            Event ID for correlation
        """
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(
                event_type, severity, source, action, details, ip_address
            )
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                status=status,
                source=source,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                action=action,
                resource=resource,
                details=details,
                risk_score=risk_score,
                tags=tags or [],
                correlation_id=correlation_id,
                parent_event_id=parent_event_id,
                checksum=None  # Will be calculated later
            )
            
            # Calculate checksum for tamper detection
            audit_event.checksum = self._calculate_checksum(audit_event)
            
            # Add to buffer
            self.event_buffer.append(audit_event)
            
            # Update metrics
            self.metrics['total_events'] += 1
            self.metrics['events_by_type'][event_type.value] = \
                self.metrics['events_by_type'].get(event_type.value, 0) + 1
            self.metrics['events_by_severity'][severity.value] = \
                self.metrics['events_by_severity'].get(severity.value, 0) + 1
            
            # Check compliance rules
            await self._check_compliance_rules(audit_event)
            
            # Check for immediate alerts
            if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                await self._send_immediate_alert(audit_event)
            
            # Flush buffer if it's getting large
            if len(self.event_buffer) >= self.batch_size:
                await self._flush_buffer()
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            self.metrics['failed_events'] += 1
            raise

    async def _calculate_risk_score(self,
                                  event_type: AuditEventType,
                                  severity: AuditSeverity,
                                  source: str,
                                  action: str,
                                  details: Dict[str, Any],
                                  ip_address: Optional[str]) -> float:
        """Calculate risk score for the event"""
        base_score = 0.0
        
        # Base score from severity
        severity_scores = {
            AuditSeverity.INFO: 0.1,
            AuditSeverity.LOW: 0.3,
            AuditSeverity.MEDIUM: 0.5,
            AuditSeverity.HIGH: 0.7,
            AuditSeverity.CRITICAL: 0.9
        }
        base_score = severity_scores.get(severity, 0.5)
        
        # Adjust based on event type
        type_multipliers = {
            AuditEventType.SECURITY_INCIDENT: 1.5,
            AuditEventType.AUTHENTICATION: 1.2,
            AuditEventType.TRANSACTION: 1.3,
            AuditEventType.WALLET_OPERATION: 1.4,
            AuditEventType.API_ACCESS: 1.1
        }
        base_score *= type_multipliers.get(event_type, 1.0)
        
        # Check for risk patterns
        pattern_score = await self._check_risk_patterns(event_type, source, details, ip_address)
        
        # Combine scores
        final_score = min(1.0, base_score + pattern_score)
        
        return round(final_score, 3)

    async def _check_risk_patterns(self,
                                 event_type: AuditEventType,
                                 source: str,
                                 details: Dict[str, Any],
                                 ip_address: Optional[str]) -> float:
        """Check for known risk patterns"""
        risk_score = 0.0
        
        # Check for off-hours activity
        current_hour = datetime.utcnow().hour
        business_start, business_end = self.risk_patterns['off_hours_activity']['business_hours']
        if not (business_start <= current_hour <= business_end):
            risk_score += 0.2
        
        # Check for high transaction amounts
        if event_type == AuditEventType.TRANSACTION:
            amount = details.get('amount', 0)
            if amount > self.risk_patterns['unusual_transaction_volume']['threshold']:
                risk_score += 0.3
        
        # Check for repeated failures from same IP
        if ip_address and event_type == AuditEventType.AUTHENTICATION:
            # This would check recent failed attempts from the same IP
            risk_score += 0.1  # Simplified
        
        return min(1.0, risk_score)

    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate HMAC checksum for tamper detection"""
        # Create a copy without checksum for calculation
        event_dict = asdict(event)
        event_dict.pop('checksum', None)
        
        # Convert to JSON string
        event_string = json.dumps(event_dict, sort_keys=True, default=str)
        
        # Calculate HMAC
        h = HMAC(self.hmac_key, hashes.SHA256(), backend=default_backend())
        h.update(event_string.encode())
        
        return h.finalize().hex()

    def _verify_checksum(self, event: AuditEvent) -> bool:
        """Verify event checksum"""
        stored_checksum = event.checksum
        event.checksum = None
        calculated_checksum = self._calculate_checksum(event)
        event.checksum = stored_checksum
        
        return stored_checksum == calculated_checksum

    async def _check_compliance_rules(self, event: AuditEvent) -> None:
        """Check event against compliance rules"""
        for rule in self.compliance_rules:
            try:
                if rule['condition'](event):
                    await self._handle_compliance_violation(event, rule)
                    
            except Exception as e:
                logger.error(f"Error checking compliance rule {rule['rule_id']}: {e}")

    async def _handle_compliance_violation(self, event: AuditEvent, rule: Dict) -> None:
        """Handle compliance rule violation"""
        violation_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.COMPLIANCE,
            severity=AuditSeverity.HIGH,
            status=AuditStatus.SUCCESS,
            source="compliance_engine",
            user_id=event.user_id,
            session_id=event.session_id,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            action="compliance_rule_triggered",
            resource=rule['rule_id'],
            details={
                'rule_id': rule['rule_id'],
                'rule_description': rule['description'],
                'triggering_event_id': event.event_id,
                'action_required': rule['action']
            },
            risk_score=0.8,
            tags=['compliance', 'violation'],
            correlation_id=event.correlation_id,
            parent_event_id=event.event_id,
            checksum=None
        )
        
        violation_event.checksum = self._calculate_checksum(violation_event)
        self.event_buffer.append(violation_event)
        
        logger.warning(f"Compliance violation: {rule['rule_id']} triggered by event {event.event_id}")

    async def _send_immediate_alert(self, event: AuditEvent) -> None:
        """Send immediate alert for high-severity events"""
        if not self.real_time_alerts:
            return
        
        try:
            alert_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'severity': event.severity.value,
                'event_type': event.event_type.value,
                'source': event.source,
                'action': event.action,
                'resource': event.resource,
                'risk_score': event.risk_score,
                'details': event.details
            }
            
            # This would integrate with your alerting system
            logger.critical(f"SECURITY ALERT: {json.dumps(alert_data)}")
            
        except Exception as e:
            logger.error(f"Failed to send immediate alert: {e}")

    async def _flush_buffer(self) -> None:
        """Flush event buffer to disk"""
        if not self.event_buffer:
            return
        
        try:
            # Get current log file
            log_file = await self._get_current_log_file()
            
            # Write events to file
            async with aiofiles.open(log_file, 'a', encoding='utf-8') as f:
                for event in self.event_buffer:
                    event_json = json.dumps(asdict(event), default=str)
                    await f.write(event_json + '\n')
            
            logger.debug(f"Flushed {len(self.event_buffer)} events to {log_file}")
            
            # Clear buffer
            self.event_buffer.clear()
            self.metrics['last_flush_time'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to flush event buffer: {e}")
            self.metrics['failed_events'] += len(self.event_buffer)

    async def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        log_file = self.log_directory / f"audit_{date_str}.log"
        
        # Check if file needs rotation
        if log_file.exists() and log_file.stat().st_size > self.max_log_size:
            # Rotate log file
            timestamp = datetime.utcnow().strftime('%H%M%S')
            rotated_file = self.log_directory / f"audit_{date_str}_{timestamp}.log"
            log_file.rename(rotated_file)
            
            # Compress rotated file if enabled
            if self.compression_enabled:
                await self._compress_log_file(rotated_file)
        
        return log_file

    async def _compress_log_file(self, log_file: Path) -> None:
        """Compress log file"""
        try:
            compressed_file = log_file.with_suffix('.log.gz')
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            log_file.unlink()
            
            logger.info(f"Compressed log file: {compressed_file}")
            
        except Exception as e:
            logger.error(f"Failed to compress log file {log_file}: {e}")

    async def _cleanup_old_logs(self) -> None:
        """Cleanup old log files based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            for log_file in self.log_directory.glob("audit_*.log*"):
                # Extract date from filename
                try:
                    parts = log_file.stem.split('_')
                    if len(parts) >= 2:
                        date_str = parts[1]
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        
                        if file_date < cutoff_date:
                            log_file.unlink()
                            logger.info(f"Deleted old log file: {log_file}")
                            
                except ValueError:
                    # Skip files that don't match expected pattern
                    continue
            
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")

    async def search_events(self,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_type: Optional[AuditEventType] = None,
                          severity: Optional[AuditSeverity] = None,
                          user_id: Optional[str] = None,
                          source: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          correlation_id: Optional[str] = None,
                          limit: int = 100) -> List[AuditEvent]:
        """Search audit events with various filters"""
        try:
            events = []
            
            # Define search criteria
            def matches_criteria(event: AuditEvent) -> bool:
                if start_time and event.timestamp < start_time:
                    return False
                if end_time and event.timestamp > end_time:
                    return False
                if event_type and event.event_type != event_type:
                    return False
                if severity and event.severity != severity:
                    return False
                if user_id and event.user_id != user_id:
                    return False
                if source and event.source != source:
                    return False
                if ip_address and event.ip_address != ip_address:
                    return False
                if correlation_id and event.correlation_id != correlation_id:
                    return False
                return True
            
            # Search through log files
            log_files = sorted(self.log_directory.glob("audit_*.log*"))
            
            for log_file in log_files:
                if len(events) >= limit:
                    break
                
                try:
                    # Handle compressed files
                    if log_file.suffix == '.gz':
                        file_obj = gzip.open(log_file, 'rt', encoding='utf-8')
                    else:
                        file_obj = open(log_file, 'r', encoding='utf-8')
                    
                    with file_obj as f:
                        for line in f:
                            if len(events) >= limit:
                                break
                            
                            try:
                                event_data = json.loads(line.strip())
                                event = AuditEvent(**event_data)
                                
                                # Verify checksum
                                if not self._verify_checksum(event):
                                    logger.warning(f"Checksum verification failed for event {event.event_id}")
                                    continue
                                
                                if matches_criteria(event):
                                    events.append(event)
                                    
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning(f"Failed to parse log line: {e}")
                                continue
                
                except Exception as e:
                    logger.error(f"Failed to search log file {log_file}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to search events: {e}")
            return []

    async def generate_compliance_report(self,
                                       start_date: datetime,
                                       end_date: datetime,
                                       report_type: str = "security") -> Dict[str, Any]:
        """Generate compliance report for specified period"""
        try:
            # Search for relevant events
            events = await self.search_events(start_time=start_date, end_time=end_date)
            
            # Analyze events
            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': datetime.utcnow().isoformat(),
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'report_type': report_type,
                'summary': {
                    'total_events': len(events),
                    'events_by_type': {},
                    'events_by_severity': {},
                    'high_risk_events': 0,
                    'compliance_violations': 0
                },
                'detailed_analysis': {
                    'authentication_events': [],
                    'transaction_events': [],
                    'security_incidents': [],
                    'compliance_events': []
                },
                'recommendations': []
            }
            
            # Analyze events
            for event in events:
                # Count by type
                event_type = event.event_type.value
                report['summary']['events_by_type'][event_type] = \
                    report['summary']['events_by_type'].get(event_type, 0) + 1
                
                # Count by severity
                severity = event.severity.value
                report['summary']['events_by_severity'][severity] = \
                    report['summary']['events_by_severity'].get(severity, 0) + 1
                
                # Count high-risk events
                if event.risk_score > 0.7:
                    report['summary']['high_risk_events'] += 1
                
                # Count compliance violations
                if event.event_type == AuditEventType.COMPLIANCE:
                    report['summary']['compliance_violations'] += 1
                
                # Categorize for detailed analysis
                if event.event_type == AuditEventType.AUTHENTICATION:
                    report['detailed_analysis']['authentication_events'].append(asdict(event))
                elif event.event_type == AuditEventType.TRANSACTION:
                    report['detailed_analysis']['transaction_events'].append(asdict(event))
                elif event.event_type == AuditEventType.SECURITY_INCIDENT:
                    report['detailed_analysis']['security_incidents'].append(asdict(event))
                elif event.event_type == AuditEventType.COMPLIANCE:
                    report['detailed_analysis']['compliance_events'].append(asdict(event))
            
            # Generate recommendations
            if report['summary']['high_risk_events'] > 10:
                report['recommendations'].append("Review security policies due to high number of high-risk events")
            
            if report['summary']['compliance_violations'] > 0:
                report['recommendations'].append("Address compliance violations identified in the period")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get audit logger metrics"""
        return {
            'metrics': self.metrics.copy(),
            'buffer_size': len(self.event_buffer),
            'active_sessions': len(self.active_sessions),
            'log_directory': str(self.log_directory),
            'retention_days': self.retention_days,
            'compression_enabled': self.compression_enabled
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop background tasks"""
        try:
            # Cancel background tasks
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Final flush
            await self._flush_buffer()
            
            # Log shutdown
            await self.log_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.INFO,
                status=AuditStatus.SUCCESS,
                source="audit_logger",
                action="shutdown",
                resource="audit_system",
                details={"reason": "normal_shutdown"}
            )
            
            # Final flush after shutdown log
            await self._flush_buffer()
            
            logger.info("Audit logger cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during audit logger cleanup: {e}")

    async def log_security_event(self, event: Dict) -> None:
        """
        Log security event - wrapper for documented signature
        """
        await self.log_event(
            event_type=AuditEventType.SECURITY_INCIDENT,
            severity=AuditSeverity(event.get('severity', 'medium')),
            status=AuditStatus(event.get('status', 'success')),
            source=event.get('source', 'unknown'),
            action=event.get('action', 'unknown'),
            resource=event.get('resource', 'unknown'),
            details=event
        )

    async def log_access(self, user: str, resource: str, action: str) -> None:
        """
        Log access event
        """
        await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            status=AuditStatus.SUCCESS,
            source="access_control",
            user_id=user,
            action=action,
            resource=resource,
            details={'user': user, 'resource': resource, 'action': action}
        )

    async def log_transaction(self, tx_data: Dict) -> None:
        """
        Log transaction event
        """
        await self.log_event(
            event_type=AuditEventType.TRANSACTION,
            severity=AuditSeverity.MEDIUM,
            status=AuditStatus(tx_data.get('status', 'pending')),
            source="transaction_manager",
            action="execute_transaction",
            resource=tx_data.get('to', 'unknown'),
            details=tx_data
        )

    async def get_audit_trail(self, filters: Dict) -> List[Dict]:
        """
        Get audit trail - wrapper for documented signature
        """
        events = await self.search_events(
            start_time=filters.get('start_time'),
            end_time=filters.get('end_time'),
            event_type=filters.get('event_type'),
            severity=filters.get('severity'),
            user_id=filters.get('user_id'),
            source=filters.get('source'),
            ip_address=filters.get('ip_address'),
            correlation_id=filters.get('correlation_id'),
            limit=filters.get('limit', 100)
        )
        
        # Convert to dict format
        return [asdict(event) for event in events]