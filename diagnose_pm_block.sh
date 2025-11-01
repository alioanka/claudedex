#!/bin/bash
# DIAGNOSTIC: Why Portfolio Manager Blocks Trades

set -e

echo "=========================================="
echo "DIAGNOSTIC: Portfolio Manager Block Reason"
echo "=========================================="
echo ""

# Query to check portfolio manager state
cat > /tmp/check_pm_state.sql << 'SQL'
-- Check trade statistics
SELECT 
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE status = 'closed') as closed_trades,
    COUNT(*) FILTER (WHERE status = 'open') as open_trades,
    COUNT(*) FILTER (WHERE status = 'closed' AND (exit_price - entry_price) > 0) as winning_trades,
    COUNT(*) FILTER (WHERE status = 'closed' AND (exit_price - entry_price) < 0) as losing_trades,
    SUM(CASE WHEN status = 'closed' THEN (exit_price - entry_price) * amount ELSE 0 END) as cumulative_pnl
FROM trades;

\echo ''
\echo 'Last 10 trades (most recent first):'
SELECT 
    trade_id,
    token_symbol,
    status,
    entry_price,
    exit_price,
    CASE 
        WHEN exit_price IS NOT NULL 
        THEN ROUND(((exit_price - entry_price) / entry_price * 100)::numeric, 2) 
        ELSE NULL 
    END as pnl_pct,
    entry_timestamp,
    exit_timestamp
FROM trades
ORDER BY entry_timestamp DESC
LIMIT 10;

\echo ''
\echo 'Consecutive losses (last 10 closed trades):'
WITH last_trades AS (
    SELECT 
        token_symbol,
        CASE WHEN (exit_price - entry_price) > 0 THEN 'WIN' ELSE 'LOSS' END as result,
        exit_timestamp
    FROM trades
    WHERE status = 'closed'
    ORDER BY exit_timestamp DESC
    LIMIT 10
)
SELECT * FROM last_trades;
SQL

echo "Checking database state..."
docker-compose exec -T postgres psql -U bot_user -d tradingbot < /tmp/check_pm_state.sql

echo ""
echo "=========================================="
echo "ANALYSIS"
echo "=========================================="
echo ""
echo "Portfolio Manager blocks when:"
echo "1. Available balance < $5 (min position size)"
echo "2. Consecutive losses >= 5"
echo "3. Daily loss > $40 or 10%"
echo "4. Total risk > 25%"
echo ""
echo "From your logs:"
echo "  - Last trade: ROCK exit at 00:46:24"
echo "  - Lost: -$1.42 (-19.7%)"
echo "  - Then: NO NEW POSITIONS for 20 hours"
echo ""
echo "Most likely causes:"
echo "  1. Consecutive losses limit hit (5 losses in a row)"
echo "  2. Daily loss limit hit (-$40 total)"
echo "  3. Portfolio manager balance is wrong (thinks $0 available)"
echo ""

read -p "Press Enter to see config values..."

echo ""
echo "Current config (.env):"
grep -E "MAX_POSITION|CONSECUTIVE_LOSSES|DAILY_LOSS|BALANCE" .env | grep -v "^#"

echo ""
echo "=========================================="
echo "RECOMMENDED FIXES"
echo "=========================================="
echo ""
echo "Fix 1: Reset consecutive losses counter"
echo "  - Add logging to see actual count"
echo "  - Consider reducing limit from 5 to 3"
echo ""
echo "Fix 2: Fix portfolio manager balance calculation"
echo "  - Currently uses self.positions (empty)"
echo "  - Should use engine.active_positions"
echo ""
echo "Fix 3: Add debug logging"
echo "  - Log WHY can_open_position returns False"
echo "  - Show which condition failed"
echo ""