"""Analyze strategy performance"""
import asyncio
import asyncpg
import argparse
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Dict, List

async def analyze_strategy(strategy: str = "all"):
    """Analyze strategy performance"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        print(f"\n📊 STRATEGY ANALYSIS - {strategy.upper()}")
        print("="*60)
        
        # Get positions by strategy
        if strategy == "all":
            positions = await conn.fetch("""
                SELECT * FROM positions
                WHERE status = 'closed'
                ORDER BY exit_time DESC
            """)
        else:
            positions = await conn.fetch("""
                SELECT * FROM positions
                WHERE status = 'closed'
                AND metadata->>'strategy' = $1
                ORDER BY exit_time DESC
            """, strategy)
        
        if not positions:
            print(f"No closed positions found for strategy: {strategy}")
            return
        
        # Calculate metrics
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['pnl'] and p['pnl'] > 0)
        losing_trades = sum(1 for p in positions if p['pnl'] and p['pnl'] < 0)
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(p['pnl'] or 0 for p in positions)
        avg_win = np.mean([p['pnl'] for p in positions if p['pnl'] and p['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p['pnl'] for p in positions if p['pnl'] and p['pnl'] < 0]) if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(p['pnl'] for p in positions if p['pnl'] and p['pnl'] > 0)
        gross_loss = abs(sum(p['pnl'] for p in positions if p['pnl'] and p['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        returns = [p['pnl_percentage'] for p in positions if p['pnl_percentage']]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum([p['pnl'] or 0 for p in positions])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Average holding time
        holding_times = []
        for p in positions:
            if p['entry_time'] and p['exit_time']:
                duration = (p['exit_time'] - p['entry_time']).total_seconds() / 3600
                holding_times.append(duration)
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        # Print results
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        print(f"   Average Win: ${avg_win:,.2f}")
        print(f"   Average Loss: ${avg_loss:,.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: ${max_drawdown:,.2f}")
        print(f"   Avg Holding Time: {avg_holding_time:.1f} hours")
        
        # Monthly breakdown
        monthly_pnl = {}
        for p in positions:
            if p['exit_time']:
                month_key = p['exit_time'].strftime('%Y-%m')
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + (p['pnl'] or 0)
        
        if monthly_pnl:
            print(f"\n📅 Monthly Performance:")
            for month in sorted(monthly_pnl.keys(), reverse=True)[:6]:
                print(f"   {month}: ${monthly_pnl[month]:,.2f}")
        
        # Best and worst trades
        sorted_positions = sorted(positions, key=lambda x: x['pnl'] or 0, reverse=True)
        
        print(f"\n🏆 Best Trades:")
        for i, p in enumerate(sorted_positions[:3], 1):
            print(f"   {i}. {p['token'][:10]}... +${p['pnl']:,.2f} ({p['pnl_percentage']:.1f}%)")
        
        print(f"\n💔 Worst Trades:")
        for i, p in enumerate(reversed(sorted_positions[-3:]), 1):
            print(f"   {i}. {p['token'][:10]}... ${p['pnl']:,.2f} ({p['pnl_percentage']:.1f}%)")
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": float(total_pnl),
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor
        }
        
    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze strategy performance')
    parser.add_argument('--strategy', default='all', help='Strategy to analyze')
    args = parser.parse_args()
    
    asyncio.run(analyze_strategy(args.strategy))