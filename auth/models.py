"""
Authentication Models
User, Session, and related data structures
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"      # Full access including sensitive settings
    OPERATOR = "operator"  # Can execute trades and modify non-sensitive settings
    VIEWER = "viewer"    # Read-only access


@dataclass
class User:
    """User model"""
    id: int
    username: str
    password_hash: str
    role: UserRole
    email: Optional[str] = None
    is_active: bool = True
    require_2fa: bool = False
    totp_secret: Optional[str] = None
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding sensitive data)"""
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role.value,
            'email': self.email,
            'is_active': self.is_active,
            'require_2fa': self.require_2fa,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class Session:
    """Session model"""
    session_id: str
    user_id: int
    ip_address: str
    user_agent: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_active': self.is_active
        }


@dataclass
class AuditLog:
    """Audit log for tracking important actions"""
    id: int
    user_id: Optional[int]
    username: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    old_value: Optional[str]
    new_value: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.username,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }
