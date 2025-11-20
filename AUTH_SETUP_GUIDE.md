# Authentication System Setup Guide

## 🔐 Overview

This guide will help you set up the secure authentication system for the ClaudeDex Trading Bot dashboard. The system includes:

- ✅ Secure login with bcrypt password hashing
- ✅ Session management with automatic expiration
- ✅ Role-Based Access Control (RBAC): Admin, Operator, Viewer
- ✅ Optional 2FA (TOTP-based)
- ✅ Rate limiting to prevent brute force attacks
- ✅ Comprehensive audit logging
- ✅ Secure configuration management from dashboard

## 📋 Prerequisites

- PostgreSQL database running
- Python 3.8+ with required packages
- Dashboard application configured

## 🚀 Quick Setup

### Step 1: Install Required Packages

```bash
pip install bcrypt pyotp asyncpg
```

### Step 2: Initialize Auth System

Run the initialization script to create auth tables and default admin user:

```bash
python scripts/init_auth.py
```

This will:
- Create `users`, `sessions`, and `audit_logs` tables
- Create default admin user (username: `admin`, password: `admin123`)
- ⚠️ **IMPORTANT**: Change the default password immediately after first login!

### Step 3: Integrate with Dashboard

Add the following to your `monitoring/enhanced_dashboard.py`:

```python
from auth.auth_service import AuthService
from auth.middleware import auth_middleware_factory
from monitoring.auth_routes import AuthRoutes
import asyncpg

class DashboardEndpoints:
    def __init__(self, app, engine=None, db=None):
        self.app = app
        self.engine = engine
        self.db = db

        # ✅ ADD: Initialize auth service
        self.auth_service = None

        self._setup_routes()

    async def initialize_auth(self):
        """Initialize authentication system"""
        if self.db and hasattr(self.db, 'pool'):
            self.auth_service = AuthService(
                db_pool=self.db.pool,
                session_timeout=3600,  # 1 hour
                max_failed_attempts=5
            )

            # Store in app for middleware access
            self.app['auth_service'] = self.auth_service

            # Setup auth routes
            AuthRoutes(self.app, self.auth_service)

            # Add auth middleware
            self.app.middlewares.insert(0, auth_middleware_factory)

            logger.info("✅ Authentication system initialized")

    def _setup_routes(self):
        """Setup all routes"""
        # ... existing route setup ...

        # Note: Auth routes will be added by AuthRoutes class
```

### Step 4: Start Dashboard

Start your dashboard application. The auth system will be active.

### Step 5: Login

1. Navigate to `http://localhost:8080/login`
2. Use default credentials:
   - Username: `admin`
   - Password: `admin123`
3. **IMMEDIATELY** change your password after first login!

## 🔒 Security Features

### Password Security
- Passwords hashed with bcrypt (cost factor 12)
- Minimum password requirements can be added
- Password change forces logout from all devices

### Session Security
- Sessions stored in database
- Automatic expiration (default: 1 hour)
- HTTPOnly cookies (prevents XSS attacks)
- SameSite=Lax (prevents CSRF)
- IP address and user agent tracking

### Rate Limiting
- Max 5 failed login attempts before account lock
- Account automatically unlocks after successful password reset
- Admin can manually unlock accounts

### Audit Logging
- All login attempts logged (success/failure)
- All configuration changes logged
- Includes IP address, user agent, timestamps
- Cannot be modified or deleted by users

## 👥 User Roles

### Admin
- Full access to all features
- Can manage users (create, edit, delete)
- Can edit sensitive configuration (API keys, private keys)
- Can view audit logs

### Operator
- Can execute trades
- Can modify non-sensitive settings
- Cannot access sensitive configuration
- Cannot manage users

### Viewer
- Read-only access
- Can view dashboard, trades, positions
- Cannot modify any settings
- Cannot execute trades

## 🔑 User Management

### Creating Users (Admin Only)

1. Login as admin
2. Navigate to `/users` page
3. Click "Create New User"
4. Fill in details:
   - Username
   - Password
   - Role (admin/operator/viewer)
   - Email (optional)
   - Enable 2FA (optional)

### Changing Passwords

Users can change their own password:

```bash
# Via API
POST /api/auth/change-password
{
  "old_password": "current_password",
  "new_password": "new_secure_password"
}
```

Admins can reset any user's password via user management page.

### Deactivating Users

Admins can deactivate user accounts, which:
- Prevents login
- Invalidates all active sessions
- Preserves audit trail

## 🛡️ 2FA Setup

### Enable 2FA for User

1. Admin enables 2FA when creating/editing user
2. User receives TOTP secret
3. User scans QR code with authenticator app (Google Authenticator, Authy, etc.)
4. User enters 6-digit code on next login

### Recommended for:
- Admin accounts (highly recommended)
- Production environments
- Accounts with access to sensitive data

## 📊 Configuration Management

### Non-Sensitive Settings
Located in `config/` directory as YAML files:
- trading.yaml
- risk_management.yaml
- portfolio.yaml
- etc.

### Sensitive Settings (.env only)
**Never expose these in dashboard:**
- PRIVATE_KEY
- SOLANA_PRIVATE_KEY
- API keys (initially)
- Database passwords

**Expose with admin-only access:**
- API keys (for trading APIs)
- RPC URLs
- Notification tokens

### Accessing Settings from Dashboard

Admin users can edit all settings through `/settings` page:
- Trading parameters
- Risk management
- API configurations
- Monitoring settings

All changes are:
- Validated before saving
- Logged in audit trail
- Applied immediately (hot-reload)

## 🔄 Session Management

### Session Lifecycle
1. User logs in → Session created
2. Session valid for 1 hour (configurable)
3. Activity extends session
4. Logout → Session invalidated
5. Expiration → Auto invalidate

### Force Logout
Admin can invalidate all sessions for a user:
```python
await auth_service.invalidate_all_user_sessions(user_id)
```

## 📝 Audit Trail

All security-relevant events are logged:

```sql
SELECT * FROM audit_logs
WHERE action IN ('login_success', 'login_failed', 'config_update')
ORDER BY timestamp DESC
LIMIT 100;
```

Includes:
- User ID and username
- Action type
- Resource affected
- Old/new values (for config changes)
- IP address
- User agent
- Timestamp
- Success/failure

## 🚨 Security Best Practices

### For Admins

1. **Change Default Password Immediately**
   - Default: `admin/admin123`
   - Change on first login

2. **Enable 2FA for Admin Accounts**
   - Adds extra layer of security
   - Required for production

3. **Regular Password Rotation**
   - Change passwords every 90 days
   - Use strong, unique passwords

4. **Monitor Audit Logs**
   - Review failed login attempts
   - Check for suspicious activity
   - Monitor config changes

5. **Limit Admin Accounts**
   - Only create admin accounts when necessary
   - Use operator/viewer roles for most users

6. **Secure .env File**
   - Set proper file permissions (600)
   - Never commit to git
   - Use encryption for backups

### For Deployment

1. **Enable HTTPS**
   - Required for production
   - Update cookie secure flag: `secure=True`

2. **Configure Firewall**
   - Limit dashboard access to trusted IPs
   - Use VPN for remote access

3. **Database Security**
   - Use strong database password
   - Limit database access
   - Enable SSL for database connections

4. **Regular Backups**
   - Backup database (includes audit logs)
   - Test backup restoration

5. **Update Dependencies**
   - Keep bcrypt, pyotp up to date
   - Monitor security advisories

## 🔧 Troubleshooting

### Can't Login
1. Check database is running
2. Verify auth tables exist: `\dt users` in psql
3. Check logs for error messages
4. Verify password hasn't been changed

### Account Locked
- Too many failed attempts
- Admin can reset failed_login_attempts:
```sql
UPDATE users SET failed_login_attempts = 0 WHERE username = 'username';
```

### Session Expired Immediately
- Check session_timeout setting
- Verify system time is correct
- Check database timestamps

### 2FA Not Working
- Verify time synchronization (TOTP is time-based)
- Check TOTP secret is correct
- Try with 1-step tolerance window

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review audit_logs table
3. Check database connectivity
4. Verify Python package versions

## 🔄 Migration from No-Auth Setup

If you have an existing dashboard without auth:

1. Backup your database
2. Run init_auth.py to add tables
3. Update enhanced_dashboard.py with auth integration
4. Restart dashboard
5. Login with default admin credentials
6. Create user accounts for team members
7. Test thoroughly before production use

## ✅ Verification Checklist

- [ ] Auth tables created successfully
- [ ] Can login with default credentials
- [ ] Default password changed
- [ ] Created additional user accounts
- [ ] Tested different user roles
- [ ] 2FA working (if enabled)
- [ ] Sessions expire correctly
- [ ] Audit logs recording events
- [ ] Settings page accessible
- [ ] Config changes persist
- [ ] Logout working correctly
- [ ] Password change working
- [ ] User management working (admin)

## 📚 API Reference

### Authentication Endpoints

```
POST /api/auth/login
  Body: { username, password, totp_code? }
  Returns: { success, user }
  Sets: session_id cookie

POST /api/auth/logout
  Auth: Required
  Returns: { success }
  Clears: session_id cookie

GET /api/auth/session
  Auth: Required
  Returns: { success, user }

GET /api/auth/users
  Auth: Admin only
  Returns: { success, users[] }

POST /api/auth/users
  Auth: Admin only
  Body: { username, password, role, email?, require_2fa? }
  Returns: { success, user }

PUT /api/auth/users/{id}
  Auth: Admin only
  Body: { role?, password? }
  Returns: { success }

DELETE /api/auth/users/{id}
  Auth: Admin only
  Returns: { success }

POST /api/auth/change-password
  Auth: Required
  Body: { old_password, new_password }
  Returns: { success }
```

## 🎯 Next Steps

1. Complete the integration in enhanced_dashboard.py
2. Test all authentication features
3. Create user accounts for your team
4. Enable 2FA for admin accounts
5. Review and customize session timeout
6. Set up monitoring for failed login attempts
7. Configure backup strategy for audit logs

---

**Security Note**: This authentication system is designed for internal dashboards. For public-facing applications, consider additional security measures such as rate limiting at the network level, DDoS protection, and regular security audits.
