# DexScreener Trading Bot - Complete Deployment Checklist

## 📋 Pre-Deployment Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify Python version (3.9+)
python --version
```

### 2. Directory Structure Verification

```bash
# Create all required directories
mkdir -p dashboard/templates
mkdir -p dashboard/static/{css,js,img}
mkdir -p config
mkdir -p logs
mkdir -p data

# Verify structure
tree -L 2 dashboard/
```

Expected structure:
```
dashboard/
├── static/
│   ├── css/
│   │   ├── main.css
│   │   ├── dashboard.css
│   │   └── charts.css
│   ├── js/
│   │   ├── main.js
│   │   ├── websocket.js
│   │   └── dashboard.js
│   └── img/
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── trades.html
    ├── positions.html
    ├── performance.html
    ├── settings.html
    ├── reports.html
    └── backtest.html
```

### 3. File Placement

Copy all provided files to their locations:

```bash
# Backend
cp enhanced_dashboard.py monitoring/enhanced_dashboard.py

# Templates
cp base.html dashboard/templates/
cp dashboard.html dashboard/templates/
cp trades.html dashboard/templates/
cp positions.html dashboard/templates/
cp performance.html dashboard/templates/
cp settings.html dashboard/templates/
cp reports.html dashboard/templates/
cp backtest.html dashboard/templates/

# CSS
cp main.css dashboard/static/css/
cp dashboard.css dashboard/static/css/
cp charts.css dashboard/static/css/

# JavaScript
cp main.js dashboard/static/js/
cp websocket.js dashboard/static/js/
cp dashboard.js dashboard/static/js/
```

### 4. Dependencies Installation

Update `requirements.txt`:

```txt
# Existing dependencies...

# Dashboard dependencies
aiohttp==3.9.1
aiohttp-cors==0.7.0
aiohttp-sse==2.2.0
python-socketio==5.10.0
jinja2==3.1.2
pandas==2.1.4
openpyxl==3.1.2
reportlab==4.0.7
```

Install:
```bash
pip install -r requirements.txt
```

### 5. Environment Variables

Verify your `.env` file has:

```bash
# Dashboard
DASHBOARD_PORT=8080

# All your existing variables
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
WEB3_PROVIDER_URL=https://...
PRIVATE_KEY=...
ENCRYPTION_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK_URL=...
```

### 6. Database Setup

```bash
# Initialize database
python scripts/setup_database.py

# Run migrations if needed
python scripts/migrate_database.py

# Verify connection
python -c "from data.storage.database import DatabaseManager; import asyncio; asyncio.run(DatabaseManager({'database_url': 'your_url'}).connect())"
```

### 7. Redis Setup

```bash
# Start Redis (if not running)
redis-server

# Or with Docker
docker run -d -p 6379:6379 redis:7-alpine

# Test connection
redis-cli ping
```

## 🚀 Deployment Steps

### Step 1: Integrate Dashboard into main.py

Add the integration code from `main_py_integration.py` to your `main.py`:

1. Add imports at top
2. Initialize dashboard in `__init__()`
3. Start dashboard in `run()` method
4. Handle shutdown properly

### Step 2: Test Locally

```bash
# Dry run mode (no real trades)
python main.py --dry-run --mode development

# Check dashboard is accessible
curl http://localhost:8080/api/bot/status
```

Expected output:
```json
{
  "success": true,
  "data": {
    "running": false,
    "uptime": "N/A",
    "mode": "development"
  }
}
```

### Step 3: Access Dashboard

Open browser: `http://localhost:8080`

You should see:
- ✅ Dashboard loads without errors
- ✅ Sidebar navigation works
- ✅ Real-time connection indicator shows status
- ✅ Charts render (may be empty without data)

### Step 4: Verify All Pages

Check each page loads:
- `http://localhost:8080/dashboard` ✅
- `http://localhost:8080/trades` ✅
- `http://localhost:8080/positions` ✅
- `http://localhost:8080/performance` ✅
- `http://localhost:8080/settings` ✅
- `http://localhost:8080/reports` ✅
- `http://localhost:8080/backtest` ✅

### Step 5: Test Bot Controls

From dashboard, test:
1. ✅ Start Bot button
2. ✅ Stop Bot button
3. ✅ Bot status updates in real-time
4. ✅ Emergency Exit (be careful!)

### Step 6: Test Settings Management

1. Go to Settings page
2. Modify a setting
3. Click Save
4. Verify change applied without restart
5. Test Revert functionality

### Step 7: Generate a Report

1. Go to Reports page
2. Click "Generate Daily Report"
3. Verify report displays
4. Test export (CSV, Excel, PDF)

## 🐳 Docker Deployment

### Update docker-compose.yml

Add dashboard port mapping:

```yaml
services:
  trading-bot:
    ports:
      - "8080:8080"  # Dashboard
      - "5432:5432"  # PostgreSQL (if needed)
```

### Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f trading-bot

# Verify dashboard
curl http://localhost:8080/api/bot/status
```

### Access Dashboard

If running on server:
```
http://your-server-ip:8080
```

## 🔒 Production Security (IMPORTANT)

### 1. Add Authentication

Create `dashboard/middleware.py`:

```python
from aiohttp import web

async def auth_middleware(request, handler):
    """Simple API key authentication"""
    api_key = request.headers.get('X-API-Key')
    expected_key = os.getenv('DASHBOARD_API_KEY')
    
    if not expected_key or api_key != expected_key:
        # Allow access to static files
        if request.path.startswith('/static'):
            return await handler(request)
        return web.json_response(
            {'error': 'Unauthorized'}, 
            status=401
        )
    
    return await handler(request)
```

Add to dashboard initialization:
```python
self.app.middlewares.append(auth_middleware)
```

### 2. Setup Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 3. SSL Certificate

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 4. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable

# Dashboard should NOT be directly accessible
# Only through nginx reverse proxy
```

## ✅ Post-Deployment Verification

### 1. Health Checks

```bash
# Bot status
curl http://localhost:8080/api/bot/status

# Dashboard summary
curl http://localhost:8080/api/dashboard/summary

# Recent trades
curl http://localhost:8080/api/trades/recent?limit=10
```

### 2. Monitor Logs

```bash
# Application logs
tail -f logs/trading_bot.log

# Dashboard logs (if separate)
tail -f logs/dashboard.log

# Docker logs
docker-compose logs -f
```

### 3. Test Real-time Updates

1. Open dashboard
2. Open browser console (F12)
3. Look for WebSocket connection: `WebSocket connection to 'ws://localhost:8080/socket.io/' opened`
4. Verify updates appearing every 1-2 seconds

### 4. Performance Check

Monitor resource usage:
```bash
# CPU and Memory
htop

# Docker stats
docker stats

# Network connections
netstat -an | grep 8080
```

## 🐛 Troubleshooting

### Dashboard Won't Start

**Issue**: Port 8080 already in use
```bash
# Find process using port
lsof -i :8080
# or
netstat -an | grep 8080

# Kill process or change DASHBOARD_PORT
export DASHBOARD_PORT=8081
```

**Issue**: Template not found
```bash
# Verify templates directory
ls -la dashboard/templates/

# Check permissions
chmod -R 755 dashboard/
```

### WebSocket Not Connecting

**Issue**: CORS errors
- Check aiohttp-cors configuration
- Verify Socket.IO versions match (client & server)
- Check browser console for errors

**Issue**: Connection refused
- Verify dashboard started successfully
- Check firewall rules
- Test with: `curl http://localhost:8080/socket.io/`

### Charts Not Displaying

**Issue**: Chart.js not loading
- Check browser console for 404 errors
- Verify CDN accessible
- Check internet connection

**Issue**: No data in charts
- Verify API endpoints returning data
- Check browser console for API errors
- Confirm database has data

### Settings Not Saving

**Issue**: Permission denied
```bash
# Check config directory permissions
chmod -R 755 config/
```

**Issue**: Changes not persisting
- Verify config_manager initialized correctly
- Check database write permissions
- Review logs for errors

## 📊 Monitoring & Maintenance

### Daily Checks

```bash
# Check bot status
curl http://localhost:8080/api/bot/status

# Check disk space
df -h

# Check logs for errors
grep -i error logs/trading_bot.log | tail -20
```

### Weekly Tasks

- Review performance reports
- Check for software updates
- Verify backups working
- Review and optimize database

### Monthly Tasks

- Security audit
- Update dependencies
- Review and optimize strategies
- Generate comprehensive reports

## 📝 Quick Reference Commands

```bash
# Start bot (development)
python main.py --mode development --dry-run

# Start bot (production)
python main.py --mode production

# Start with Docker
docker-compose up -d

# Stop bot
docker-compose down

# View logs
docker-compose logs -f trading-bot

# Restart bot
docker-compose restart trading-bot

# Emergency stop (close all positions)
curl -X POST http://localhost:8080/api/bot/emergency_exit

# Generate report
curl -X POST http://localhost:8080/api/reports/generate \
  -H "Content-Type: application/json" \
  -d '{"period":"daily","metrics":["all"]}'
```

## 🎉 Success Criteria

Your deployment is successful when:

- ✅ Dashboard accessible at http://localhost:8080
- ✅ All pages load without errors
- ✅ Real-time updates working (WebSocket connected)
- ✅ Bot controls functional (start/stop/restart)
- ✅ Settings can be modified and saved
- ✅ Reports generate and export correctly
- ✅ Charts display data properly
- ✅ No errors in logs
- ✅ Performance acceptable (< 2s page load)
- ✅ Mobile responsive (test on phone)

## 📞 Support & Resources

- **Logs**: `logs/trading_bot.log`
- **Config**: `config/`
- **Database**: Check PostgreSQL logs
- **Redis**: `redis-cli` for debugging
- **Browser Console**: F12 for frontend errors

## 🚨 Emergency Procedures

### If Bot Malfunctions

```bash
# 1. Stop everything immediately
docker-compose down
# or
pkill -f "python main.py"

# 2. Emergency exit all positions (if needed)
python scripts/emergency_stop.py

# 3. Check logs
tail -100 logs/trading_bot.log

# 4. Review recent trades
python scripts/check_balance.py
```

### If Dashboard Crashes

```bash
# Dashboard crash doesn't affect trading
# Bot continues running independently

# Restart only dashboard
# (requires separate dashboard service - optional enhancement)
```

---

## ✅ Final Checklist

Before going live:

- [ ] All files in correct locations
- [ ] Dependencies installed
- [ ] Database connected and initialized
- [ ] Redis connected
- [ ] Environment variables set
- [ ] Dashboard accessible locally
- [ ] All pages load correctly
- [ ] WebSocket connection working
- [ ] Bot controls functional
- [ ] Settings management working
- [ ] Report generation working
- [ ] Tested in dry-run mode
- [ ] Logs monitoring setup
- [ ] Backups configured
- [ ] Security measures implemented
- [ ] Documentation reviewed
- [ ] Emergency procedures understood

**YOU'RE READY TO GO LIVE! 🚀**