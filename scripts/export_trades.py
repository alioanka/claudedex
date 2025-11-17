# ============================================
# scripts/export_trades.py
# ============================================
"""Export trades to various formats"""
import asyncio
import asyncpg
import pandas as pd
import argparse
from datetime import datetime
import json
import os

async def export_trades(format: str = "csv", output: str = None):
    """Export trades to file"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Fetch all trades
        trades = await conn.fetch("""
            SELECT 
                id,
                token,
                chain,
                type,
                amount,
                price,
                total,
                gas_price,
                gas_used,
                tx_hash,
                status,
                created_at
            FROM trades
            ORDER BY created_at DESC
        """)
        
        # Convert to list of dicts
        trades_data = [dict(t) for t in trades]
        
        # Convert datetime objects to strings
        for trade in trades_data:
            trade['created_at'] = trade['created_at'].isoformat()
            # Convert Decimal to float
            for key in ['amount', 'price', 'total']:
                if trade[key]:
                    trade[key] = float(trade[key])
        
        # Generate output filename if not specified
        if not output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f"trades_{timestamp}.{format}"
        
        # Export based on format
        if format == "csv":
            df = pd.DataFrame(trades_data)
            df.to_csv(output, index=False)
            print(f"✅ Exported {len(trades)} trades to {output}")
            
        elif format == "json":
            with open(output, 'w') as f:
                json.dump(trades_data, f, indent=2)
            print(f"✅ Exported {len(trades)} trades to {output}")
            
        elif format == "excel":
            df = pd.DataFrame(trades_data)
            output = output.replace('.xls', '.xlsx')
            df.to_excel(output, index=False)
            print(f"✅ Exported {len(trades)} trades to {output}")
        
        # Print summary
        if trades_data:
            print(f"\n📊 Export Summary:")
            print(f"   Total Trades: {len(trades)}")
            print(f"   Date Range: {trades_data[-1]['created_at'][:10]} to {trades_data[0]['created_at'][:10]}")
            
            # Calculate totals
            buy_volume = sum(t['total'] for t in trades_data if t['type'] == 'buy' and t['total'])
            sell_volume = sum(t['total'] for t in trades_data if t['type'] == 'sell' and t['total'])
            
            print(f"   Buy Volume: ${buy_volume:,.2f}")
            print(f"   Sell Volume: ${sell_volume:,.2f}")
        else:
            print("No trades to export")
        
        return output
        
    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export trades')
    parser.add_argument('--format', choices=['csv', 'json', 'excel'], default='csv', help='Export format')
    parser.add_argument('--output', help='Output filename')
    args = parser.parse_args()
    
    asyncio.run(export_trades(args.format, args.output))