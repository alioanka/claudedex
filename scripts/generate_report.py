# ============================================
# scripts/generate_report.py
# ============================================
"""Generate comprehensive trading report"""
import asyncio
import asyncpg
from datetime import datetime, timedelta
import argparse
import pandas as pd
import json
from pathlib import Path

async def generate_report(days: int = 30, format: str = "text"):
    """Generate comprehensive trading report"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch all trades
        trades_data = await conn.fetch("""
            SELECT * FROM trades
            WHERE created_at >= $1 AND created_at <= $2
            ORDER BY created_at DESC
        """, start_date, end_date)
        
        # Fetch all positions
        positions_data = await conn.fetch("""
            SELECT * FROM positions
            WHERE created_at >= $1 AND created_at <= $2
            ORDER BY created_at DESC
        """, start_date, end_date)
        
        # Calculate metrics
        total_trades = len(trades_data)
        total_positions = len(positions_data)
        
        # Calculate P&L
        total_pnl = sum(p['pnl'] or 0 for p in positions_data if p['status'] == 'closed')
        winning_trades = sum(1 for p in positions_data if p['pnl'] and p['pnl'] > 0)
        losing_trades = sum(1 for p in positions_data if p['pnl'] and p['pnl'] < 0)
        
        win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
        
        # Create report
        report = {
            "period": f"{days} days",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "summary": {
                "total_trades": total_trades,
                "total_positions": total_positions,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": f"{win_rate:.1f}%",
                "total_pnl": float(total_pnl)
            },
            "daily_breakdown": [],
            "top_performers": [],
            "worst_performers": []
        }
        
        # Daily breakdown
        for day in range(days):
            date = start_date + timedelta(days=day)
            day_trades = [t for t in trades_data if t['created_at'].date() == date.date()]
            day_pnl = sum(p['pnl'] or 0 for p in positions_data 
                         if p['exit_time'] and p['exit_time'].date() == date.date())
            
            report["daily_breakdown"].append({
                "date": date.date().isoformat(),
                "trades": len(day_trades),
                "pnl": float(day_pnl)
            })
        
        # Top performers
        top_positions = sorted(
            [p for p in positions_data if p['pnl']],
            key=lambda x: x['pnl'],
            reverse=True
        )[:5]
        
        for pos in top_positions:
            report["top_performers"].append({
                "token": pos['token'],
                "pnl": float(pos['pnl']),
                "pnl_percentage": float(pos['pnl_percentage'] or 0),
                "entry_time": pos['entry_time'].isoformat()
            })
        
        # Output report
        if format == "json":
            output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")
            
        elif format == "csv":
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(report["daily_breakdown"])
            output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"Report saved to {output_file}")
            
        else:  # text format
            print(f"\n📊 TRADING REPORT - Last {days} Days")
            print("="*60)
            print(f"Period: {start_date.date()} to {end_date.date()}")
            print(f"\n📈 Summary:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total P&L: ${total_pnl:,.2f}")
            
            if report["top_performers"]:
                print(f"\n🏆 Top Performers:")
                for i, pos in enumerate(report["top_performers"], 1):
                    print(f"   {i}. {pos['token'][:10]}... +${pos['pnl']:.2f} ({pos['pnl_percentage']:.1f}%)")
        
        return report
        
    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trading report')
    parser.add_argument('--days', type=int, default=30, help='Number of days to report')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], default='text', help='Output format')
    args = parser.parse_args()
    
    asyncio.run(generate_report(args.days, args.format))