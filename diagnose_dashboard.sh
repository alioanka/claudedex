#!/bin/bash
# diagnose_dashboard.sh - Diagnose dashboard issues

echo "🔍 Dashboard Issue Diagnosis"
echo "============================"
echo ""

# 1. Check if dashboard is running
echo "1️⃣ Dashboard Service Status:"
if docker compose ps | grep -q "dashboard.*Up"; then
    echo "   ✅ Dashboard is running"
    port=$(docker compose ps dashboard | grep -o "0.0.0.0:[0-9]*" | cut -d: -f2)
    echo "   Port: $port"
else
    echo "   ❌ Dashboard is NOT running!"
fi
echo ""

# 2. Check database stats
echo "2️⃣ Database Statistics:"
docker compose exec postgres psql -U claudedex_user -d claudedex_db -c "
SELECT 
    COUNT(*) as total_trades,
    COUNT(CASE WHEN status = 'open' THEN 1 END) as open_positions,
    COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed_trades,
    TO_CHAR(SUM(pnl_usd), 'FM$999,999.99') as total_pnl
FROM trades;
"
echo ""

# 3. Sample trade data
echo "3️⃣ Sample Trade Data (Last 3):"
docker compose exec postgres psql -U claudedex_user -d claudedex_db -c "
SELECT 
    id,
    symbol,
    chain,
    status,
    entry_price,
    current_price,
    pnl_usd,
    created_at
FROM trades 
ORDER BY created_at DESC 
LIMIT 3;
" -x
echo ""

# 4. Check dashboard logs for errors
echo "4️⃣ Dashboard Error Logs (Last 20):"
docker compose logs dashboard --tail=50 | grep -iE "(error|exception|failed)" | head -20
echo ""

# 5. Check API endpoints
echo "5️⃣ Testing Dashboard API Endpoints:"
port=$(docker compose port dashboard 8080 2>/dev/null | cut -d: -f2)
if [ -n "$port" ]; then
    # Test summary endpoint
    echo "   Testing /api/summary..."
    curl -s "http://localhost:$port/api/summary" | python3 -m json.tool 2>/dev/null | head -30
    echo ""
    
    echo "   Testing /api/positions..."
    curl -s "http://localhost:$port/api/positions" | python3 -m json.tool 2>/dev/null | head -20
else
    echo "   ⚠️  Cannot determine dashboard port"
fi
echo ""

# 6. Check which files exist
echo "6️⃣ Dashboard File Structure:"
echo "Templates:"
ls -la monitoring/templates/*.html 2>/dev/null | awk '{print "   ", $9}' || echo "   ⚠️  No templates found"
echo ""
echo "Static files:"
ls -la monitoring/static/js/*.js 2>/dev/null | awk '{print "   ", $9}' || echo "   ⚠️  No JS files found"
ls -la monitoring/static/css/*.css 2>/dev/null | awk '{print "   ", $9}' || echo "   ⚠️  No CSS files found"
echo ""

echo "============================"
echo "📋 What I need from you:"
echo ""
echo "Please share these files:"
echo "1. monitoring/enhanced_dashboard.py (the main dashboard backend)"
echo "2. monitoring/templates/dashboard.html (main page)"
echo "3. monitoring/templates/trades.html"
echo "4. monitoring/templates/positions.html"
echo "5. monitoring/static/js/dashboard.js"
echo ""
echo "Or run these commands:"
echo "  head -100 monitoring/enhanced_dashboard.py"
echo "  grep -n 'def.*summary' monitoring/enhanced_dashboard.py"
echo "  grep -n 'def.*positions' monitoring/enhanced_dashboard.py"
echo ""