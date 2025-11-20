# Quick Integration Instructions

## Step-by-Step Integration of Auth System

### 1. Install Required Packages

```bash
pip install bcrypt pyotp
```

### 2. Run Database Migration

```bash
python scripts/init_auth.py
```

This creates auth tables and default admin user.

### 3. Update `monitoring/enhanced_dashboard.py`

Add these imports at the top:

```python
from auth.auth_service import AuthService
from auth.middleware import auth_middleware_factory, require_auth, require_admin
from monitoring.auth_routes import AuthRoutes
```

In the `__init__` method, add:

```python
def __init__(self, app, engine=None, db=None):
    self.app = app
    self.engine = engine
    self.db = db

    # ✅ ADD THIS:
    self.auth_service = None

    self._setup_routes()
```

Add this new method after `__init__`:

```python
async def initialize_auth(self):
    """Initialize authentication system"""
    try:
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

            # Add auth middleware (must be first)
            self.app.middlewares.insert(0, auth_middleware_factory)

            logger.info("✅ Authentication system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize auth: {e}", exc_info=True)
```

### 4. Call `initialize_auth()` during startup

In your main startup code (probably `main.py` or where you initialize the dashboard):

```python
# After creating dashboard instance
dashboard = DashboardEndpoints(app, engine=engine, db=db)

# ✅ ADD THIS: Initialize auth system
await dashboard.initialize_auth()

# Then start the app
web.run_app(app, host='0.0.0.0', port=8080)
```

### 5. Protect Sensitive Routes (Optional but Recommended)

To protect specific routes, add decorators:

```python
# Example: Protect bot control endpoints
@require_auth
async def api_bot_start(self, request):
    # ... existing code ...

# Example: Admin-only endpoints
@require_auth
@require_admin
async def api_emergency_exit(self, request):
    # ... existing code ...
```

Or protect by checking in the handler:

```python
async def api_bot_stop(self, request):
    # Check if user is authenticated
    if 'user' not in request:
        return web.json_response({'error': 'Authentication required'}, status=401)

    user = request['user']

    # Check if admin
    if user.role != UserRole.ADMIN:
        return web.json_response({'error': 'Admin access required'}, status=403)

    # ... rest of your code ...
```

### 6. Test the System

1. Start your dashboard:
```bash
python main.py  # or however you start it
```

2. Navigate to: `http://localhost:8080/login`

3. Login with:
   - Username: `admin`
   - Password: `admin123`

4. **IMMEDIATELY** change the password!

### 7. Create Additional Users

After logging in as admin:
- Navigate to `/users` (not yet implemented in UI, use API)
- Or use the API directly:

```bash
curl -X POST http://localhost:8080/api/auth/users \
  -H "Content-Type: application/json" \
  -b "session_id=YOUR_SESSION_COOKIE" \
  -d '{
    "username": "trader1",
    "password": "SecurePass123!",
    "role": "operator",
    "email": "trader@example.com"
  }'
```

## Configuration Files Structure

After integration, your config structure will be:

```
.env                          # Sensitive credentials ONLY
├── PRIVATE_KEY
├── SOLANA_PRIVATE_KEY
├── DATABASE_URL
├── API keys for external services
└── Notification tokens

config/                       # Non-sensitive settings (managed from dashboard)
├── trading.yaml
├── risk_management.yaml
├── portfolio.yaml
├── chains.yaml
└── ... other config files
```

## Protected Routes

By default, all dashboard pages will require authentication:
- `/` - Dashboard (requires auth)
- `/trades` - Trade history (requires auth)
- `/positions` - Open positions (requires auth)
- `/settings` - Settings (requires auth)
- `/reports` - Reports (requires auth)

Public routes (no auth required):
- `/login` - Login page
- `/static/*` - Static files (CSS, JS, images)

## User Roles Explained

**Admin:**
- Full access to everything
- Can create/manage users
- Can access sensitive settings (API keys, private keys)
- Can execute trades and modify configs

**Operator:**
- Can execute trades
- Can modify non-sensitive settings
- Cannot manage users
- Cannot access sensitive credentials

**Viewer:**
- Read-only access
- Can view dashboard, trades, positions
- Cannot execute trades
- Cannot modify settings

## Quick Commands

```bash
# Initialize auth system
python scripts/init_auth.py

# Check if auth tables exist
psql -U bot_user -d tradingbot -c "\dt users"

# View users
psql -U bot_user -d tradingbot -c "SELECT id, username, role, is_active FROM users;"

# Reset admin password (if locked out)
psql -U bot_user -d tradingbot -c "UPDATE users SET password_hash = '\$2b\$12\$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Lfw99hhm1qJYT8sFm', failed_login_attempts = 0 WHERE username = 'admin';"
# This resets admin password back to 'admin123'

# View audit logs
psql -U bot_user -d tradingbot -c "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10;"

# Clear expired sessions
psql -U bot_user -d tradingbot -c "DELETE FROM sessions WHERE expires_at < NOW();"
```

## Troubleshooting

**"Authentication system not available"**
- Auth service not initialized
- Make sure you called `await dashboard.initialize_auth()`

**"Cannot connect to database"**
- Check DATABASE_URL in .env
- Verify PostgreSQL is running
- Check database credentials

**"Auth tables don't exist"**
- Run `python scripts/init_auth.py`
- Check database logs for errors

**Sessions expire immediately**
- Check system time is correct
- Verify session_timeout setting
- Check database timestamps

## Next Steps

1. ✅ Integration complete
2. Test all features
3. Create user accounts for team
4. Configure role-based access
5. Enable 2FA for admin accounts
6. Set up monitoring for failed logins
7. Review and customize session timeout
8. Update .env.example with new structure

---

**That's it!** Your dashboard now has enterprise-grade authentication and security.
