"""Detailed strategy analysis with monthly breakdown"""
import asyncio
import asyncpg
import argparse
import os
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

async def strategy_analysis(month: bool = False):
    """Analyze strategy performance with optional monthly breakdown"""
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        print("\nSTRATEGY ANALYSIS")
        print("="*60)
        
        # Get all strategies
        strategies = await conn.fetch("""
            SELECT DISTINCT metadata->>'strategy' as strategy
            FROM positions
            WHERE metadata->>'strategy' IS NOT NULL
        """)
        
        strategy_names = [s['strategy'] for s in strategies if s['strategy']]
        
        if not strategy_names:
            print("No strategy data found")
            return
        
        for strategy in strategy_names:
            print(f"\n{'='*60}")
            print(f"STRATEGY: {strategy.upper()}")
            print(f"{'='*60}")
            
            if month:
                # Monthly breakdown
                monthly_data = await conn.fetch("""
                    SELECT 
                        DATE_TRUNC('month', exit_time) as month,
                        COUNT(*) as trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        STDDEV(pnl_percentage) as volatility
                    FROM positions
                    WHERE status = 'closed'
                    AND metadata->>'strategy' = $1
                    AND exit_time IS NOT NULL
                    GROUP BY DATE_TRUNC('month', exit_time)
                    ORDER BY month DESC
                """, strategy)
                
                print("\nMonthly Performance:")
                print(f"{'Month':<12} {'Trades':>8} {'Win%':>8} {'P&L':>12} {'Avg':>10} {'Vol':>8}")
                print("-" * 60)
                
                for month_data in monthly_data:
                    month_str = month_data['month'].strftime('%Y-%m')
                    trades = month_data['trades']
                    wins = month_data['wins'] or 0
                    win_rate = (wins / trades * 100) if trades > 0 else 0
                    pnl = month_data['total_pnl'] or 0
                    avg_pnl = month_data['avg_pnl'] or 0
                    vol = month_data['volatility'] or 0
                    
                    print(f"{month_str:<12} {trades:>8} {win_rate:>7.1f}% ${pnl:>10.2f} ${avg_pnl:>8.2f} {vol:>7.1f}%")
            
            # Overall statistics
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss,
                    AVG(pnl_percentage) as avg_return,
                    STDDEV(pnl_percentage) as volatility,
                    AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/3600) as avg_duration_hours
                FROM positions
                WHERE status = 'closed'
                AND metadata->>'strategy' = $1
            """, strategy)
            
            total = stats['total_trades']
            wins = stats['winning_trades'] or 0
            win_rate = (wins / total * 100) if total > 0 else 0
            
            print(f"\nOverall Statistics:")
            print(f"  Total Trades: {total}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total P&L: ${stats['total_pnl'] or 0:,.2f}")
            print(f"  Avg P&L: ${stats['avg_pnl'] or 0:,.2f}")
            print(f"  Best Trade: ${stats['max_win'] or 0:,.2f}")
            print(f"  Worst Trade: ${stats['max_loss'] or 0:,.2f}")
            print(f"  Avg Return: {stats['avg_return'] or 0:.2f}%")
            print(f"  Volatility: {stats['volatility'] or 0:.2f}%")
            print(f"  Avg Duration: {stats['avg_duration_hours'] or 0:.1f} hours")
            
            # Calculate Sharpe ratio
            returns = await conn.fetch("""
                SELECT pnl_percentage
                FROM positions
                WHERE status = 'closed'
                AND metadata->>'strategy' = $1
                AND pnl_percentage IS NOT NULL
            """, strategy)
            
            if len(returns) > 1:
                returns_array = np.array([r['pnl_percentage'] for r in returns])
                sharpe = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252)
                print(f"  Sharpe Ratio: {sharpe:.2f}")
            
            # Risk metrics
            print(f"\nRisk Metrics:")
            
            # Calculate max drawdown
            positions = await conn.fetch("""
                SELECT pnl, exit_time
                FROM positions
                WHERE status = 'closed'
                AND metadata->>'strategy' = $1
                ORDER BY exit_time
            """, strategy)
            
            if positions:
                cumulative_pnl = np.cumsum([p['pnl'] or 0 for p in positions])
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = running_max - cumulative_pnl
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                
                print(f"  Max Drawdown: ${max_drawdown:,.2f}")
                
                # Profit factor
                gross_profit = sum(p['pnl'] for p in positions if p['pnl'] and p['pnl'] > 0)
                gross_loss = abs(sum(p['pnl'] for p in positions if p['pnl'] and p['pnl'] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                print(f"  Profit Factor: {profit_factor:.2f}")
            
            # Entry/Exit analysis
            print(f"\nTiming Analysis:")
            
            day_performance = await conn.fetch("""
                SELECT 
                    EXTRACT(DOW FROM entry_time) as day_of_week,
                    COUNT(*) as trades,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as win_rate
                FROM positions
                WHERE status = 'closed'
                AND metadata->>'strategy' = $1
                GROUP BY EXTRACT(DOW FROM entry_time)
                ORDER BY day_of_week
            """, strategy)
            
            days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            if day_performance:
                for perf in day_performance:
                    dow = int(perf['day_of_week'])
                    print(f"  {days[dow]}: {perf['trades']} trades, ${perf['avg_pnl'] or 0:.2f} avg, {perf['win_rate'] or 0:.1f}% win rate")
        
        print("\n" + "="*60)
        
    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strategy analysis')
    parser.add_argument('--month', action='store_true', help='Show monthly breakdown')
    args = parser.parse_args()
    
    asyncio.run(strategy_analysis(args.month))