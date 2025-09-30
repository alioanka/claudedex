"""Emergency stop script - immediately halt all trading"""
import asyncio
import asyncpg
import redis
import os
from datetime import datetime

async def emergency_stop():
    """Emergency stop all trading operations"""
    
    print("EMERGENCY STOP INITIATED")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Set emergency flag in Redis
    print("\n1. Setting emergency flag...")
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )
        r.set("emergency_stop", "true")
        r.set("emergency_stop_time", datetime.now().isoformat())
        print("   Emergency flag set in Redis")
    except Exception as e:
        print(f"   Warning: Could not set Redis flag: {e}")
    
    # Update database
    print("\n2. Updating database...")
    try:
        conn = await asyncpg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "tradingbot"),
            user=os.getenv("DB_USER", "trading"),
            password=os.getenv("DB_PASSWORD", "trading123")
        )
        
        # Log emergency stop
        await conn.execute("""
            INSERT INTO audit_logs (event_type, severity, status, source, action, details)
            VALUES ('emergency_stop', 'critical', 'success', 'manual', 'emergency_stop', 
                    '{"reason": "Manual emergency stop triggered"}')
        """)
        
        # Get current open positions
        positions = await conn.fetch("""
            SELECT id, token_address, position_type, entry_amount
            FROM positions
            WHERE status = 'open'
        """)
        
        print(f"   Found {len(positions)} open positions")
        
        # Get pending orders
        pending_orders = await conn.fetchval("""
            SELECT COUNT(*)
            FROM trades
            WHERE status IN ('pending', 'processing')
        """)
        
        print(f"   Found {pending_orders or 0} pending orders")
        
        await conn.close()
        
    except Exception as e:
        print(f"   Warning: Database update failed: {e}")
    
    # Create emergency stop file
    print("\n3. Creating emergency stop file...")
    stop_file = "EMERGENCY_STOP"
    with open(stop_file, 'w') as f:
        f.write(f"Emergency stop activated at {datetime.now().isoformat()}\n")
        f.write(f"Open positions: {len(positions) if 'positions' in locals() else 'unknown'}\n")
        f.write(f"Pending orders: {pending_orders if 'pending_orders' in locals() else 'unknown'}\n")
    print(f"   Created {stop_file}")
    
    print("\n" + "="*60)
    print("EMERGENCY STOP COMPLETE")
    print("\nAll trading operations halted.")
    print("To resume:")
    print("  1. Remove EMERGENCY_STOP file")
    print("  2. Run: redis-cli DEL emergency_stop")
    print("  3. Restart bot")
    print("\nTo close positions:")
    print("  Run: python scripts/close_all_positions.py --confirm")

if __name__ == "__main__":
    asyncio.run(emergency_stop())