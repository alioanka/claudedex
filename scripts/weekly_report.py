"""Generate weekly trading report"""
import asyncio
import asyncpg
import os
from datetime import datetime, timedelta
import numpy as np

async def weekly_report():
    """Generate comprehensive weekly report"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"📊 WEEKLY REPORT")
        print(f"{start_date.date()} to {end_date.date()}")
        print("="*60)
        
        # Trading summary
        trades = await conn.fetch("""
            SELECT 
                type,
                COUNT(*) as count,
                SUM(total) as volume,
                AVG(gas_price) as avg_gas
            FROM trades
            WHERE created_at >= $1 AND created_at <= $2
            GROUP BY type
        """, start_date, end_date)
        
        buy_count = sell_count = 0
        buy_volume = sell_volume = 0
        
        for t in trades:
            if t['type'] == 'buy':
                buy_count = t['count']
                buy_volume = t['volume'] or 0
            else:
                sell_count = t['count']
                sell_volume = t['volume'] or 0
        
        print(f"\n📈 Trading Summary:")
        print(f"   Total Trades: {buy_count + sell_count}")
        print(f"   Buy Orders: {buy_count} (${buy_volume:,.2f})")
        print(f"   Sell Orders: {sell_count} (${sell_volume:,.2f})")
        
        # Performance metrics
        positions = await conn.fetch("""
            SELECT 
                pnl,
                pnl_percentage,
                exit_time - entry_time as duration
            FROM positions
            WHERE status = 'closed'
            AND exit_time >= $1 AND exit_time <= $2
        """, start_date, end_date)
        
        if positions:
            winning = [p for p in positions if p['pnl'] and p['pnl'] > 0]
            losing = [p for p in positions if p['pnl'] and p['pnl'] < 0]
            
            total_pnl = sum(p['pnl'] or 0 for p in positions)
            win_rate = (len(winning) / len(positions)) * 100 if positions else 0
            
            avg_win = np.mean([p['pnl'] for p in winning]) if winning else 0
            avg_loss = np.mean([p['pnl'] for p in losing]) if losing else 0
            
            print(f"\n💰 Performance:")
            print(f"   Total P&L: ${total_pnl:,.2f}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Win: ${avg_win:,.2f}")
            print(f"   Avg Loss: ${avg_loss:,.2f}")
            
            # Calculate Sharpe ratio
            returns = [p['pnl_percentage'] for p in positions if p['pnl_percentage']]
            if len(returns) > 1:
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(52)
                print(f"   Sharpe Ratio: {sharpe:.2f}")
        
        # Daily breakdown
        print(f"\n📅 Daily Breakdown:")
        for i in range(7):
            day = start_date + timedelta(days=i)
            day_pnl = await conn.fetchval("""
                SELECT COALESCE(SUM(pnl), 0)
                FROM positions
                WHERE status = 'closed'
                AND exit_time >= $1 AND exit_time < $2
            """, day, day + timedelta(days=1))
            
            emoji = "📈" if day_pnl >= 0 else "📉"
            print(f"   {day.strftime('%a %m/%d')}: {emoji} ${day_pnl:,.2f}")
        
        # Top tokens
        top_tokens = await conn.fetch("""
            SELECT 
                token_address,
                SUM(pnl) as total_pnl,
                COUNT(*) as trades
            FROM positions
            WHERE status = 'closed'
            AND exit_time >= $1 AND exit_time <= $2
            GROUP BY token_address
            ORDER BY total_pnl DESC
            LIMIT 5
        """, start_date, end_date)
        
        if top_tokens:
            print(f"\n🏆 Top Performers:")
            for i, token in enumerate(top_tokens, 1):
                print(f"   {i}. {token['token_address'][:10]}... ${token['total_pnl']:,.2f} ({token['trades']} trades)")
        
        # Alert summary
        alerts = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE severity = 'critical') as critical,
                COUNT(*) FILTER (WHERE severity = 'high') as high
            FROM audit_logs
            WHERE event_type LIKE 'alert_%'
            AND timestamp >= $1 AND timestamp <= $2
        """, start_date, end_date)
        
        print(f"\n🔔 Alerts:")
        print(f"   Total: {alerts['total']}")
        print(f"   Critical: {alerts['critical']}")
        print(f"   High: {alerts['high']}")
        
        # Current status
        open_positions = await conn.fetchval("""
            SELECT COUNT(*)
            FROM positions
            WHERE status = 'open'
        """)
        
        print(f"\n📊 Current Status:")
        print(f"   Open Positions: {open_positions}")
        
        print("\n" + "="*60)
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(weekly_report())