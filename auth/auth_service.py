"""
Authentication Service
Handles user authentication, session management, and security features
"""
import secrets
import hashlib
import pyotp
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import logging
from asyncpg import Pool

from .models import User, Session, UserRole, AuditLog

logger = logging.getLogger(__name__)


class AuthService:
    """
    Comprehensive authentication service with:
    - Password hashing with bcrypt
    - Session management
    - Rate limiting
    - 2FA support (TOTP)
    - Audit logging
    """

    def __init__(self, db_pool: Pool, session_timeout: int = 3600, max_failed_attempts: int = 5):
        self.db_pool = db_pool
        self.session_timeout = session_timeout  # seconds
        self.max_failed_attempts = max_failed_attempts

    # ==================== User Management ====================

    async def create_user(self,username: str,
                         password: str,
                         role: UserRole = UserRole.VIEWER,
                         email: Optional[str] = None,
                         require_2fa: bool = False,
                         created_by: Optional[int] = None) -> Optional[User]:
        """Create a new user with hashed password"""
        try:
            # Hash password with bcrypt
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Generate TOTP secret if 2FA is required
            totp_secret = pyotp.random_base32() if require_2fa else None

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO users (username, password_hash, role, email, require_2fa, totp_secret, is_active, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, TRUE, NOW(), NOW())
                    RETURNING id, username, password_hash, role, email, is_active, require_2fa, totp_secret,
                              failed_login_attempts, last_login, created_at, updated_at
                """, username, password_hash, role.value, email, require_2fa, totp_secret)

                if row:
                    user = User(
                        id=row['id'],
                        username=row['username'],
                        password_hash=row['password_hash'],
                        role=UserRole(row['role']),
                        email=row['email'],
                        is_active=row['is_active'],
                        require_2fa=row['require_2fa'],
                        totp_secret=row['totp_secret'],
                        failed_login_attempts=row['failed_login_attempts'],
                        last_login=row['last_login'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )

                    logger.info(f"Created user: {username} with role {role.value}")
                    return user

        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            return None

    async def authenticate(self,
                          username: str,
                          password: str,
                          totp_code: Optional[str] = None,
                          ip_address: str = "unknown",
                          user_agent: str = "unknown") -> Tuple[bool, Optional[User], Optional[str]]:
        """
        Authenticate user with password and optional 2FA
        Returns: (success, user, error_message)
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get user
                row = await conn.fetchrow("""
                    SELECT id, username, password_hash, role, email, is_active, require_2fa, totp_secret,
                           failed_login_attempts, last_login, created_at, updated_at
                    FROM users
                    WHERE username = $1
                """, username)

                if not row:
                    logger.warning(f"Authentication failed: User {username} not found")
                    await self._log_audit(conn, None, username, "login_failed", "user", None,
                                         None, None, ip_address, user_agent, False, "User not found")
                    return False, None, "Invalid username or password"

                user = User(
                    id=row['id'],
                    username=row['username'],
                    password_hash=row['password_hash'],
                    role=UserRole(row['role']),
                    email=row['email'],
                    is_active=row['is_active'],
                    require_2fa=row['require_2fa'],
                    totp_secret=row['totp_secret'],
                    failed_login_attempts=row['failed_login_attempts'],
                    last_login=row['last_login'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                # Check if user is active
                if not user.is_active:
                    logger.warning(f"Authentication failed: User {username} is inactive")
                    await self._log_audit(conn, user.id, username, "login_failed", "user", str(user.id),
                                         None, None, ip_address, user_agent, False, "User is inactive")
                    return False, None, "Account is disabled"

                # Check rate limiting
                if user.failed_login_attempts >= self.max_failed_attempts:
                    logger.warning(f"Authentication failed: User {username} exceeded max failed attempts")
                    await self._log_audit(conn, user.id, username, "login_failed", "user", str(user.id),
                                         None, None, ip_address, user_agent, False, "Too many failed attempts")
                    return False, None, "Account locked due to too many failed login attempts. Contact administrator."

                # Verify password
                if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                    # Increment failed attempts
                    await conn.execute("""
                        UPDATE users
                        SET failed_login_attempts = failed_login_attempts + 1, updated_at = NOW()
                        WHERE id = $1
                    """, user.id)

                    logger.warning(f"Authentication failed: Invalid password for user {username}")
                    await self._log_audit(conn, user.id, username, "login_failed", "user", str(user.id),
                                         None, None, ip_address, user_agent, False, "Invalid password")
                    return False, None, "Invalid username or password"

                # Verify 2FA if required
                if user.require_2fa:
                    if not totp_code:
                        return False, None, "2FA code required"

                    if not self._verify_totp(user.totp_secret, totp_code):
                        await conn.execute("""
                            UPDATE users
                            SET failed_login_attempts = failed_login_attempts + 1, updated_at = NOW()
                            WHERE id = $1
                        """, user.id)

                        logger.warning(f"Authentication failed: Invalid 2FA code for user {username}")
                        await self._log_audit(conn, user.id, username, "login_failed", "user", str(user.id),
                                             None, None, ip_address, user_agent, False, "Invalid 2FA code")
                        return False, None, "Invalid 2FA code"

                # Success - reset failed attempts and update last login
                await conn.execute("""
                    UPDATE users
                    SET failed_login_attempts = 0, last_login = NOW(), updated_at = NOW()
                    WHERE id = $1
                """, user.id)

                logger.info(f"User {username} authenticated successfully")
                await self._log_audit(conn, user.id, username, "login_success", "user", str(user.id),
                                     None, None, ip_address, user_agent, True, None)

                return True, user, None

        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}", exc_info=True)
            return False, None, "Authentication error"

    # ==================== Session Management ====================

    async def create_session(self, user_id: int, ip_address: str, user_agent: str) -> Optional[str]:
        """Create a new session for authenticated user"""
        try:
            session_id = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sessions (session_id, user_id, ip_address, user_agent, created_at, expires_at, last_activity, is_active)
                    VALUES ($1, $2, $3, $4, NOW(), $5, NOW(), TRUE)
                """, session_id, user_id, ip_address, user_agent, expires_at)

                logger.info(f"Created session for user_id {user_id}")
                return session_id

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None

    async def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user if valid"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT s.user_id, s.expires_at, s.is_active,
                           u.id, u.username, u.password_hash, u.role, u.email, u.is_active as user_active,
                           u.require_2fa, u.totp_secret, u.failed_login_attempts, u.last_login, u.created_at, u.updated_at
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.session_id = $1
                """, session_id)

                if not row:
                    return None

                # Check if session expired
                if row['expires_at'] < datetime.utcnow():
                    await self.invalidate_session(session_id)
                    return None

                # Check if session is active
                if not row['is_active']:
                    return None

                # Check if user is active
                if not row['user_active']:
                    return None

                # Update last activity
                new_expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
                await conn.execute("""
                    UPDATE sessions
                    SET last_activity = NOW(), expires_at = $2
                    WHERE session_id = $1
                """, session_id, new_expires_at)

                user = User(
                    id=row['id'],
                    username=row['username'],
                    password_hash=row['password_hash'],
                    role=UserRole(row['role']),
                    email=row['email'],
                    is_active=row['user_active'],
                    require_2fa=row['require_2fa'],
                    totp_secret=row['totp_secret'],
                    failed_login_attempts=row['failed_login_attempts'],
                    last_login=row['last_login'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                return user

        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate (logout) a session"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sessions
                    SET is_active = FALSE
                    WHERE session_id = $1
                """, session_id)

                logger.info(f"Invalidated session {session_id[:10]}...")
                return True

        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
            return False

    async def invalidate_all_user_sessions(self, user_id: int) -> bool:
        """Invalidate all sessions for a user (force logout everywhere)"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sessions
                    SET is_active = FALSE
                    WHERE user_id = $1 AND is_active = TRUE
                """, user_id)

                logger.info(f"Invalidated all sessions for user_id {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to invalidate user sessions: {e}")
            return False

    # ==================== 2FA Functions ====================

    def _verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)  # Allow 1 time step tolerance
        except:
            return False

    def get_totp_qr_uri(self, user: User, issuer: str = "ClaudeDex Bot") -> str:
        """Get QR code URI for 2FA setup"""
        totp = pyotp.TOTP(user.totp_secret)
        return totp.provisioning_uri(user.username, issuer_name=issuer)

    # ==================== Audit Logging ====================

    async def _log_audit(self, conn, user_id: Optional[int], username: str, action: str,
                        resource_type: str, resource_id: Optional[str], old_value: Optional[str],
                        new_value: Optional[str], ip_address: str, user_agent: str,
                        success: bool, error_message: Optional[str]):
        """Log audit event"""
        try:
            await conn.execute("""
                INSERT INTO audit_logs (user_id, username, action, resource_type, resource_id,
                                       old_value, new_value, ip_address, user_agent,
                                       timestamp, success, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), $10, $11)
            """, user_id, username, action, resource_type, resource_id, old_value, new_value,
               ip_address, user_agent, success, error_message)
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    async def log_config_change(self, user: User, config_type: str, key: str,
                               old_value: Any, new_value: Any, ip_address: str, user_agent: str):
        """Log configuration change"""
        try:
            async with self.db_pool.acquire() as conn:
                await self._log_audit(
                    conn, user.id, user.username, "config_update", config_type, key,
                    str(old_value) if old_value else None,
                    str(new_value) if new_value else None,
                    ip_address, user_agent, True, None
                )
        except Exception as e:
            logger.error(f"Failed to log config change: {e}")

    # ==================== User Management ====================

    async def get_all_users(self) -> list[User]:
        """Get all users (for admin panel)"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, username, password_hash, role, email, is_active, require_2fa, totp_secret,
                           failed_login_attempts, last_login, created_at, updated_at
                    FROM users
                    ORDER BY created_at DESC
                """)

                users = []
                for row in rows:
                    users.append(User(
                        id=row['id'],
                        username=row['username'],
                        password_hash=row['password_hash'],
                        role=UserRole(row['role']),
                        email=row['email'],
                        is_active=row['is_active'],
                        require_2fa=row['require_2fa'],
                        totp_secret=row['totp_secret'],
                        failed_login_attempts=row['failed_login_attempts'],
                        last_login=row['last_login'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    ))

                return users

        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            return []

    async def update_user_role(self, user_id: int, new_role: UserRole) -> bool:
        """Update user role"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE users
                    SET role = $2, updated_at = NOW()
                    WHERE id = $1
                """, user_id, new_role.value)

                logger.info(f"Updated user {user_id} role to {new_role.value}")
                return True

        except Exception as e:
            logger.error(f"Failed to update user role: {e}")
            return False

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE users
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE id = $1
                """, user_id)

                # Invalidate all sessions
                await self.invalidate_all_user_sessions(user_id)

                logger.info(f"Deactivated user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to deactivate user: {e}")
            return False

    async def change_password(self, user_id: int, new_password: str) -> bool:
        """Change user password"""
        try:
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE users
                    SET password_hash = $2, updated_at = NOW()
                    WHERE id = $1
                """, user_id, password_hash)

                logger.info(f"Changed password for user {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to change password: {e}")
            return False
