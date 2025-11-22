# Phase 1: Modular Architecture - Complete Testing Guide

**Version:** 1.0
**Date:** 2025-11-22
**Prerequisites:** Docker, Docker Compose, PostgreSQL, Redis

---

## Table of Contents

1. [Pre-Testing Checklist](#pre-testing-checklist)
2. [Fresh Installation Testing](#fresh-installation-testing)
3. [Existing Database Migration Testing](#existing-database-migration-testing)
4. [Docker Restart Testing](#docker-restart-testing)
5. [Module Functionality Testing](#module-functionality-testing)
6. [Dashboard Testing](#dashboard-testing)
7. [Integration Verification](#integration-verification)
8. [Troubleshooting](#troubleshooting)

---

## Pre-Testing Checklist

### ✅ Files Modified/Created

Verify these files exist with Phase 1 integration:

- [x] `main.py` - Module manager integration added
- [x] `modules/base_module.py` - Base module class (FIXED: added ModuleType.CUSTOM)
- [x] `modules/integration.py` - Module setup (FIXED: conditional imports)
- [x] `core/module_manager.py` - Module lifecycle management
- [x] `modules/dex_trading/dex_module.py` - DEX trading module
- [x] `monitoring/module_routes.py` - API routes for modules
- [x] `dashboard/templates/modules.html` - Module management UI
- [x] `migrations/002_add_module_support.sql` - Database migration
- [x] `config/modules/dex_trading.yaml` - DEX module config
- [x] `config/wallets.yaml` - Wallet configuration

### ✅ Environment Variables

Ensure your `.env` file has:

```bash
# Existing variables (keep as is)
DATABASE_URL=postgresql://bot_user:bot_password@postgres:5432/tradingbot
REDIS_URL=redis://:your_password@redis:6379/0

# Phase 1: Optional module-specific variables
MODULE_MANAGER_ENABLED=true  # Optional, defaults to true
```

---

## Fresh Installation Testing

### Scenario 1: Brand New Installation (No Existing Database)

**Purpose:** Verify Phase 1 works with completely fresh setup

#### Step 1: Clean Start

```bash
# Stop all containers
docker-compose down -v  # -v removes volumes too

# Remove old data (CAUTION: This deletes everything!)
rm -rf postgres_data redis_data

# Verify clean state
docker ps -a  # Should show no containers
```

#### Step 2: Start Services

```bash
# Start PostgreSQL and Redis first
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready (30 seconds)
sleep 30

# Check PostgreSQL is running
docker-compose logs postgres | tail -20
# Look for: "database system is ready to accept connections"
```

#### Step 3: First Bot Startup

```bash
# Start the bot
docker-compose up trading-bot

# Or if running locally:
python main.py --mode development
```

#### Step 4: Verify Migration

Watch the logs for these key messages:

```
✅ EXPECTED LOG OUTPUT:
==================================================
🚀 DexScreener Trading Bot Starting...
Mode: production
==================================================
Loading configuration...
Connecting to database...
Running database migrations...
📝 Applying migration: 001_initial_schema.sql
✅ Applied migration: 001_initial_schema.sql
📝 Applying migration: 002_add_module_support.sql
✅ Applied migration: 002_add_module_support.sql
✅ Database is up to date
✅ Config manager now reading from database
Initializing trading engine...
Setting up modular architecture...
✅ Module manager started with 0 module(s)  ← Phase 1 working!
Enhanced dashboard initialized
✅ Initialization complete!
```

#### Step 5: Verify Database Tables

```bash
# Connect to PostgreSQL
docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot

# Check migration was applied
SELECT * FROM migrations;
# Should show: 001_initial_schema and 002_add_module_support

# Check module tables exist
\dt modules*
# Should show:
#   modules
#   module_metrics
#   module_positions

# Check initial modules
SELECT name, module_type, enabled, status, capital_allocation FROM modules;
# Should show:
#   dex_trading       | dex_trading       | f | stopped | 500.00
#   solana_strategies | solana_strategies | f | stopped | 400.00
#   futures_trading   | futures_trading   | f | stopped | 300.00

# Exit psql
\q
```

#### Step 6: Access Dashboard

```bash
# Open browser to:
http://localhost:8080

# Navigate to Modules page:
http://localhost:8080/modules

# You should see:
# - Module management interface
# - Capital allocation chart
# - Three module cards (all disabled initially)
```

---

## Existing Database Migration Testing

### Scenario 2: Existing Database (Upgrading from Previous Version)

**Purpose:** Verify Phase 1 migrates cleanly without breaking existing data

#### Step 1: Backup Existing Database

```bash
# CRITICAL: Always backup before migration!
docker exec claudedex_postgres_1 pg_dump -U bot_user tradingbot > backup_before_phase1.sql

# Verify backup
ls -lh backup_before_phase1.sql
```

#### Step 2: Check Current State

```bash
# Connect to database
docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot

# Check existing migrations
SELECT * FROM migrations ORDER BY version;
# Note the latest version

# Check if module tables already exist
\dt modules*
# If they don't exist, migration will create them
# If they DO exist, migration will skip (using IF NOT EXISTS)

# Exit
\q
```

#### Step 3: Apply Migration

```bash
# Stop the bot if running
docker-compose stop trading-bot

# Start fresh
docker-compose up trading-bot

# Watch logs
docker-compose logs -f trading-bot
```

#### Step 4: Verify Migration Success

```bash
# Check logs for:
✅ Applied migration: 002_add_module_support
✅ Database is up to date

# Connect to database
docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot

# Verify migration
SELECT * FROM migrations WHERE version = '002';
# Should show: 002 | Add Module Support - Phase 1 Modular Architecture

# Verify tables
SELECT table_name FROM information_schema.tables
WHERE table_name LIKE 'module%';
# Should show: modules, module_metrics, module_positions

# Verify views
SELECT table_name FROM information_schema.views
WHERE table_name LIKE 'v_%module%';
# Should show: v_active_modules, v_module_performance_24h

# IMPORTANT: Check existing data is intact
SELECT COUNT(*) FROM trades;  # Should match pre-migration count
SELECT COUNT(*) FROM positions;  # Should match pre-migration count

# Exit
\q
```

#### Step 5: Verify Existing Functionality

```bash
# Restart bot
docker-compose restart trading-bot

# Check dashboard still works
curl http://localhost:8080/health
# Should return: {"status": "healthy"}

# Check existing trades endpoint
curl http://localhost:8080/api/trades | jq '.success'
# Should return: true
```

---

## Docker Restart Testing

### Scenario 3: Verify Persistence Across Restarts

**Purpose:** Ensure module state persists and restores correctly

#### Step 1: Enable a Module

```bash
# Access dashboard
http://localhost:8080/modules

# Click "Enable" on DEX Trading module

# Verify in logs:
Module dex_trading enabled
Module dex_trading started successfully
✅ Module manager started with 1 module(s)
```

#### Step 2: Check Database State

```bash
docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot

# Check module status
SELECT name, enabled, status FROM modules WHERE name = 'dex_trading';
# Should show: dex_trading | t | running

# Exit
\q
```

#### Step 3: Restart Container

```bash
# Soft restart
docker-compose restart trading-bot

# Watch logs
docker-compose logs -f trading-bot
```

#### Step 4: Verify Module State Restored

```bash
# In logs, look for:
Setting up modular architecture...
Loaded config for module: dex_trading
DEX trading module registered
Initializing module: dex_trading
Starting module: dex_trading
✅ Module manager started with 1 module(s)

# Check dashboard
http://localhost:8080/modules
# DEX Trading should still show as "RUNNING"
```

#### Step 5: Hard Restart (Stop/Start)

```bash
# Complete stop
docker-compose stop trading-bot

# Wait 10 seconds
sleep 10

# Start again
docker-compose start trading-bot

# Verify same behavior as soft restart
docker-compose logs trading-bot | grep -i "module"
```

#### Step 6: Full Stack Restart

```bash
# Stop everything
docker-compose down

# Start everything
docker-compose up -d

# Wait for services
sleep 30

# Check module restored
docker-compose logs trading-bot | tail -50
```

---

## Module Functionality Testing

### Test 1: Module Lifecycle

#### Enable Module

```bash
# Via Dashboard:
1. Go to http://localhost:8080/modules
2. Find "DEX Trading" module
3. Click "✅ Enable" button
4. Wait 2 seconds
5. Module status should change to "RUNNING"

# Via API:
curl -X POST http://localhost:8080/api/modules/dex_trading/enable
# Response: {"success": true, "message": "Module dex_trading enabled"}

# Verify in logs:
Module dex_trading enabled
Initializing module: dex_trading
Starting module: dex_trading
Module dex_trading started successfully
```

#### Pause Module

```bash
# Via Dashboard:
1. Click "⏸️ Pause" button
2. Status changes to "PAUSED"

# Via API:
curl -X POST http://localhost:8080/api/modules/dex_trading/pause
# Response: {"success": true, "message": "Module dex_trading paused"}

# Verify in logs:
Module dex_trading paused
```

#### Resume Module

```bash
# Via Dashboard:
1. Click "▶️ Resume" button
2. Status changes back to "RUNNING"

# Via API:
curl -X POST http://localhost:8080/api/modules/dex_trading/resume
# Response: {"success": true, "message": "Module dex_trading resumed"}

# Verify in logs:
Module dex_trading resumed
```

#### Disable Module

```bash
# Via Dashboard:
1. Click "🛑 Disable" button
2. Confirm dialog
3. Status changes to "STOPPED"

# Via API:
curl -X POST http://localhost:8080/api/modules/dex_trading/disable
# Response: {"success": true, "message": "Module dex_trading disabled"}

# Verify in logs:
Module dex_trading disabled
Stopping module: dex_trading
Module dex_trading stopped successfully
```

### Test 2: Module Metrics

```bash
# Get module metrics
curl http://localhost:8080/api/modules/dex_trading/metrics | jq

# Expected response:
{
  "success": true,
  "data": {
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "total_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "realized_pnl": 0.0,
    "win_rate": 0.0,
    "profit_factor": 0.0,
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "active_positions": 0,
    "capital_allocated": 500.0,
    "capital_used": 0.0,
    "capital_available": 500.0,
    "uptime_seconds": 120,
    "errors_count": 0
  }
}
```

### Test 3: Module Status

```bash
# Get all modules status
curl http://localhost:8080/api/modules | jq

# Expected response:
{
  "success": true,
  "data": {
    "manager_running": true,
    "total_modules": 1,
    "enabled_modules": 1,
    "running_modules": 1,
    "total_capital": 500.0,
    "allocated_capital": 500.0,
    "available_capital": 0.0,
    "modules": {
      "dex_trading": {
        "name": "dex_trading",
        "module_type": "dex_trading",
        "status": "running",
        "enabled": true,
        "running": true,
        ...
      }
    }
  }
}
```

### Test 4: Health Checks

```bash
# Module manager runs health checks every 30 seconds
# Watch logs:
docker-compose logs -f trading-bot | grep -i health

# Should see:
# (Every 30 seconds)
Health check passed for module: dex_trading
```

---

## Dashboard Testing

### Test 1: Modules Page

```bash
# Access: http://localhost:8080/modules

✅ Verify you see:
1. Summary statistics at top:
   - Total Capital: $500.00
   - Allocated Capital: $500.00
   - Active Modules: 0/1
   - Total PnL: $0.00

2. Capital Allocation visualization:
   - Bar chart showing allocation per module
   - Available capital shown in gray

3. Module cards showing:
   - Module name (DEX_TRADING)
   - Status badge (RUNNING/STOPPED/etc)
   - Metrics (Capital, Positions, Trades, PnL, Uptime)
   - Action buttons (Enable/Disable/Pause/Configure/Details)
```

### Test 2: Auto-Refresh

```bash
# The page auto-refreshes every 30 seconds

# To test:
1. Keep page open
2. Enable a module via API:
   curl -X POST http://localhost:8080/api/modules/dex_trading/enable
3. Wait 30 seconds
4. Page should refresh and show updated status
```

### Test 3: Module Detail View

```bash
# Click "📊 Details" button on any module

# Should redirect to:
http://localhost:8080/modules/dex_trading

# Should show (JSON for now):
{
  "module": {...},
  "metrics": {...},
  "positions": []
}
```

---

## Integration Verification

### Test 1: Module Manager Integration

```bash
# Check main.py integration
grep -n "module_manager" main.py

# Should see:
# 34: from modules.integration import setup_modular_architecture
# 159: self.module_manager = None
# 431: self.module_manager = await setup_modular_architecture(
# 457: module_manager=self.module_manager
# 615: await self.module_manager.stop()
```

### Test 2: Dashboard Integration

```bash
# Check dashboard accepts module_manager
grep -n "module_manager" monitoring/enhanced_dashboard.py

# Should see it's passed as parameter and used in setup
```

### Test 3: Database Integration

```bash
# Verify module data persists

# Connect to DB
docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot

# Check modules table has data
SELECT * FROM modules;

# Check views work
SELECT * FROM v_active_modules;

# Exit
\q
```

---

## Troubleshooting

### Issue 1: "ModuleManager initialization failed"

**Symptoms:**
```
WARNING: Module manager setup failed (non-critical): ...
WARNING: Continuing without modular architecture...
```

**Solution:**
1. Check config files exist:
   ```bash
   ls -la config/modules/
   ```

2. Check YAML syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config/modules/dex_trading.yaml'))"
   ```

3. Check logs for specific error
4. Module manager failure is non-critical - bot continues without it

---

### Issue 2: "Module tables don't exist"

**Symptoms:**
```
ERROR: relation "modules" does not exist
```

**Solution:**
1. Check migration ran:
   ```bash
   docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot \
     -c "SELECT * FROM migrations WHERE version = '002';"
   ```

2. If not found, manually run migration:
   ```bash
   docker exec -i claudedex_postgres_1 psql -U bot_user -d tradingbot \
     < migrations/002_add_module_support.sql
   ```

3. Restart bot:
   ```bash
   docker-compose restart trading-bot
   ```

---

### Issue 3: "Module not loading"

**Symptoms:**
```
Module dex_trading configured but not available (not installed)
```

**Solution:**
1. Check module files exist:
   ```bash
   ls -la modules/dex_trading/
   ```

2. Check Python imports work:
   ```bash
   python -c "from modules.dex_trading import DexTradingModule; print('OK')"
   ```

3. Check for import errors in logs

---

### Issue 4: "Dashboard shows 404 for /modules"

**Symptoms:**
- Dashboard loads but /modules returns 404

**Solution:**
1. Verify module_manager passed to dashboard:
   ```bash
   grep "module_manager=" main.py
   ```

2. Check module routes setup:
   ```bash
   grep "_setup_module_routes" monitoring/enhanced_dashboard.py
   ```

3. Restart dashboard:
   ```bash
   docker-compose restart trading-bot
   ```

---

### Issue 5: "Module state not persisting"

**Symptoms:**
- Enable module
- Restart
- Module shows disabled

**Solution:**
1. Check database connection:
   ```bash
   docker-compose logs postgres | grep "connection"
   ```

2. Verify module status saved:
   ```bash
   docker exec -it claudedex_postgres_1 psql -U bot_user -d tradingbot \
     -c "SELECT name, enabled, status FROM modules;"
   ```

3. Check for errors during save:
   ```bash
   docker-compose logs trading-bot | grep -i "module.*error"
   ```

---

## Success Criteria

### ✅ Phase 1 is working correctly when:

1. **Fresh Installation:**
   - Migration 002 runs automatically
   - Module tables created
   - Default modules inserted
   - Bot starts successfully

2. **Existing Database:**
   - Migration runs without errors
   - Existing data preserved
   - New tables added
   - Bot functions normally

3. **Docker Restart:**
   - Module state persists
   - Enabled modules auto-start
   - Configuration restored
   - No errors in logs

4. **Module Functionality:**
   - Can enable/disable modules
   - Can pause/resume modules
   - Metrics update correctly
   - Health checks run

5. **Dashboard:**
   - /modules page loads
   - Shows all modules
   - Buttons work
   - Auto-refresh works

6. **Integration:**
   - Module manager in logs
   - Dashboard accepts module_manager
   - Database queries work
   - No regressions in existing features

---

## Next Steps After Successful Testing

Once Phase 1 testing passes:

1. ✅ Deploy to production with minimal capital
2. ✅ Monitor for 24-48 hours
3. ✅ Verify module lifecycle works in production
4. ⏭️ Consider Phase 3 integration (Solana strategies)
5. ⏭️ Consider Phase 4 integration (Analytics) after security fixes

---

## Test Report Template

```
PHASE 1 TEST REPORT
==================
Date: YYYY-MM-DD
Tester: [Your Name]
Environment: [Development/Staging/Production]

Fresh Installation: [PASS/FAIL]
- Migration ran: [YES/NO]
- Tables created: [YES/NO]
- Bot started: [YES/NO]
- Notes:

Existing Database: [PASS/FAIL]
- Migration applied: [YES/NO]
- Data preserved: [YES/NO]
- No errors: [YES/NO]
- Notes:

Docker Restart: [PASS/FAIL]
- State persisted: [YES/NO]
- Modules restored: [YES/NO]
- No errors: [YES/NO]
- Notes:

Module Functionality: [PASS/FAIL]
- Enable/disable: [YES/NO]
- Pause/resume: [YES/NO]
- Metrics update: [YES/NO]
- Health checks: [YES/NO]
- Notes:

Dashboard: [PASS/FAIL]
- Page loads: [YES/NO]
- Modules visible: [YES/NO]
- Buttons work: [YES/NO]
- Auto-refresh: [YES/NO]
- Notes:

Overall Result: [PASS/FAIL]
Production Ready: [YES/NO]

Issues Found:
1.
2.

Recommendations:
1.
2.
```

---

**Testing Complete!** 🎉

For issues or questions, check the troubleshooting section or review the detailed logs.
