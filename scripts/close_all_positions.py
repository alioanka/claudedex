"""Close all open positions"""
import asyncio
import asyncpg
import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from web3 import Web3
from datetime import datetime

async def close_all_positions(confirm: bool = False):
    """Close all open positions"""
    
    if not confirm:
        print("\nWARNING: This will close ALL open positions!")
        print("Use --confirm flag to proceed")
        return False
    
    print("CLOSING ALL POSITIONS")
    print("="*60)
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Get all open positions
        positions = await conn.fetch("""
            SELECT 
                id,
                token_address,
                token_symbol,
                position_type,
                entry_price,
                entry_amount,
                created_at
            FROM positions
            WHERE status = 'open'
            ORDER BY created_at
        """)
        
        if not positions:
            print("\nNo open positions to close")
            return True
        
        print(f"\nFound {len(positions)} open positions\n")
        
        # Setup Web3
        w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))
        if not w3.is_connected():
            print("Error: Cannot connect to Web3")
            return False
        
        closed_count = 0
        failed_count = 0
        
        for position in positions:
            print(f"Closing position {position['id']}...")
            print(f"  Token: {position['token_symbol']} ({position['token_address'][:10]}...)")
            print(f"  Type: {position['position_type']}")
            print(f"  Amount: {position['entry_amount']}")
            
            try:
                # Get current price (simplified - in production would use actual DEX price)
                current_price = position['entry_price']  # Placeholder
                
                # Calculate P&L
                if position['position_type'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['entry_amount']
                else:
                    pnl = (position['entry_price'] - current_price) * position['entry_amount']
                
                pnl_percentage = (pnl / (position['entry_price'] * position['entry_amount'])) * 100
                
                # Update position in database
                await conn.execute("""
                    UPDATE positions
                    SET 
                        status = 'closed',
                        exit_price = $1,
                        exit_amount = $2,
                        exit_time = NOW(),
                        pnl = $3,
                        pnl_percentage = $4,
                        metadata = metadata || '{"closed_reason": "manual_close_all"}'::jsonb
                    WHERE id = $5
                """, current_price, position['entry_amount'], pnl, pnl_percentage, position['id'])
                
                # Log the action
                await conn.execute("""
                    INSERT INTO audit_logs 
                    (event_type, severity, status, source, action, details)
                    VALUES ('position_closed', 'high', 'success', 'manual', 'close_all_positions',
                            $1)
                """, f'{{"position_id": {position["id"]}, "pnl": {pnl}}}')
                
                print(f"  Closed: P&L = ${pnl:.2f} ({pnl_percentage:.1f}%)")
                closed_count += 1
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                failed_count += 1
                continue
        
        # Summary
        print("\n" + "="*60)
        print(f"Successfully closed: {closed_count}")
        print(f"Failed: {failed_count}")
        
        # Calculate total P&L
        total_pnl = await conn.fetchval("""
            SELECT SUM(pnl)
            FROM positions
            WHERE status = 'closed'
            AND exit_time >= NOW() - INTERVAL '1 hour'
        """)
        
        if total_pnl:
            emoji = "+" if total_pnl >= 0 else "-"
            print(f"\nTotal P&L from closed positions: {emoji}${abs(total_pnl):.2f}")
        
        return closed_count > 0
        
    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Close all open positions')
    parser.add_argument('--confirm', action='store_true', help='Confirm closing all positions')
    args = parser.parse_args()
    
    result = asyncio.run(close_all_positions(args.confirm))
    sys.exit(0 if result else 1)