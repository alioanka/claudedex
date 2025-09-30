"""Generate overnight trading summary"""
import asyncio
import asyncpg
import os
from datetime import datetime, timedelta

async def overnight_summary():
    """Generate summary of overnight activity"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        now = datetime.now()
        last_night = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        print(f"🌙 OVERNIGHT SUMMARY - {last_night.date()}")
        print("="*60)
        
        # Trading activity
        trades = await conn.fetch("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN type = 'buy' THEN 1 END) as buys,
                COUNT(CASE WHEN type = 'sell' THEN 1 END) as sells,
                SUM(total) as volume
            FROM trades
            WHERE created_at >= $1
        """, last_night)
        
        if trades[0]['total'] > 0:
            print(f"\n📊 Trading Activity:")
            print(f"   Total Trades: {trades[0]['total']}")
            print(f"   Buys: {trades[0]['buys']}")
            print(f"   Sells: {trades[0]['sells']}")
            print(f"   Volume: ${trades[0]['volume'] or 0:,.2f}")
        else:
            print("\n📊 No trading activity overnight")
        
        # Position changes
        positions = await conn.fetch("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'open' AND created_at >= $1) as opened,
                COUNT(*) FILTER (WHERE status = 'closed' AND exit_time >= $1) as closed,
                SUM(pnl) FILTER (WHERE status = 'closed' AND exit_time >= $1) as pnl
            FROM positions
        """, last_night)
        
        print(f"\n💼 Position Changes:")
        print(f"   Opened: {positions[0]['opened']}")
        print(f"   Closed: {positions[0]['closed']}")
        if positions[0]['pnl']:
            pnl_emoji = "📈" if positions[0]['pnl'] >= 0 else "📉"
            print(f"   {pnl_emoji} P&L: ${positions[0]['pnl']:,.2f}")
        
        # Alerts
        alerts = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE severity = 'critical') as critical,
                COUNT(*) FILTER (WHERE severity = 'high') as high
            FROM audit_logs
            WHERE event_type LIKE 'alert_%'
            AND timestamp >= $1
        """, last_night)
        
        if alerts['total'] > 0:
            print(f"\n🔔 Alerts:")
            print(f"   Total: {alerts['total']}")
            if alerts['critical'] > 0:
                print(f"   🔴 Critical: {alerts['critical']}")
            if alerts['high'] > 0:
                print(f"   🟠 High: {alerts['high']}")
        
        # System health
        errors = await conn.fetchval("""
            SELECT COUNT(*)
            FROM audit_logs
            WHERE event_type = 'error'
            AND timestamp >= $1
        """, last_night)
        
        print(f"\n🖥️  System Health:")
        print(f"   Errors: {errors}")
        
        # Current portfolio
        portfolio = await conn.fetchrow("""
            SELECT 
                COUNT(*) as positions,
                SUM(unrealized_pnl) as unrealized_pnl
            FROM positions
            WHERE status = 'open'
        """)
        
        print(f"\n💰 Current Portfolio:")
        print(f"   Open Positions: {portfolio['positions']}")
        if portfolio['unrealized_pnl']:
            print(f"   Unrealized P&L: ${portfolio['unrealized_pnl']:,.2f}")
        
        print("\n" + "="*60)
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(overnight_summary())