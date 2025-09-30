# ============================================
# scripts/daily_report.py
# ============================================
"""Generate daily trading report"""
import asyncio
import asyncpg
from datetime import datetime, timedelta
import os
from decimal import Decimal
import json

async def daily_report():
    """Generate daily performance report"""
    
    # Connect to database
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Get today's date range
        today = datetime.now().date()
        start_time = datetime.combine(today, datetime.min.time())
        end_time = datetime.combine(today, datetime.max.time())
        
        print(f"📊 DAILY TRADING REPORT - {today}")
        print("="*60)
        
        # Get trading summary
        trades = await conn.fetch("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_trades,
                COUNT(CASE WHEN type = 'buy' THEN 1 END) as buy_trades,
                COUNT(CASE WHEN type = 'sell' THEN 1 END) as sell_trades,
                SUM(CASE WHEN type = 'sell' THEN total - 
                    (SELECT total FROM trades t2 
                     WHERE t2.token = trades.token 
                     AND t2.type = 'buy' 
                     AND t2.created_at < trades.created_at 
                     ORDER BY created_at DESC LIMIT 1)
                    ELSE 0 END) as total_pnl
            FROM trades
            WHERE created_at >= $1 AND created_at <= $2
        """, start_time, end_time)
        
        if trades[0]['total_trades'] > 0:
            print(f"📈 Trades Executed: {trades[0]['total_trades']}")
            print(f"   ✅ Successful: {trades[0]['successful_trades']}")
            print(f"   🟢 Buys: {trades[0]['buy_trades']}")
            print(f"   🔴 Sells: {trades[0]['sell_trades']}")
            
            pnl = trades[0]['total_pnl'] or 0
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            print(f"\n{pnl_emoji} Total P&L: ${pnl:,.2f}")
        else:
            print("No trades executed today")
        
        # Get position summary
        positions = await conn.fetch("""
            SELECT 
                COUNT(*) as open_positions,
                SUM(pnl) as unrealized_pnl
            FROM positions
            WHERE status = 'open'
        """)
        
        print(f"\n📊 Open Positions: {positions[0]['open_positions']}")
        if positions[0]['unrealized_pnl']:
            print(f"   Unrealized P&L: ${positions[0]['unrealized_pnl']:,.2f}")
        
        # Get top performers
        top_trades = await conn.fetch("""
            SELECT 
                token,
                pnl_percentage,
                pnl
            FROM positions
            WHERE exit_time >= $1 AND exit_time <= $2
            ORDER BY pnl DESC
            LIMIT 3
        """, start_time, end_time)
        
        if top_trades:
            print("\n🏆 Top Performers:")
            for i, trade in enumerate(top_trades, 1):
                print(f"   {i}. {trade['token'][:10]}... +{trade['pnl_percentage']:.1f}% (${trade['pnl']:.2f})")
        
        # Get risk metrics
        risk_metrics = await conn.fetchrow("""
            SELECT 
                MAX(pnl_percentage) as max_gain,
                MIN(pnl_percentage) as max_loss,
                AVG(pnl_percentage) as avg_return,
                STDDEV(pnl_percentage) as volatility
            FROM positions
            WHERE exit_time >= $1 AND exit_time <= $2
        """, start_time, end_time)
        
        if risk_metrics['avg_return']:
            print("\n📊 Risk Metrics:")
            print(f"   Max Gain: {risk_metrics['max_gain']:.1f}%")
            print(f"   Max Loss: {risk_metrics['max_loss']:.1f}%")
            print(f"   Avg Return: {risk_metrics['avg_return']:.1f}%")
            print(f"   Volatility: {risk_metrics['volatility']:.1f}%")
        
        # Get alerts summary
        alerts = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_alerts,
                COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical,
                COUNT(CASE WHEN severity = 'high' THEN 1 END) as high,
                COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium,
                COUNT(CASE WHEN severity = 'low' THEN 1 END) as low
            FROM audit_logs
            WHERE timestamp >= $1 AND timestamp <= $2
            AND event_type LIKE 'alert_%'
        """, start_time, end_time)
        
        if alerts['total_alerts'] > 0:
            print(f"\n🔔 Alerts Today: {alerts['total_alerts']}")
            print(f"   🔴 Critical: {alerts['critical']}")
            print(f"   🟠 High: {alerts['high']}")
            print(f"   🟡 Medium: {alerts['medium']}")
            print(f"   🟢 Low: {alerts['low']}")
        
        print("\n" + "="*60)
        print("Report generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(daily_report())