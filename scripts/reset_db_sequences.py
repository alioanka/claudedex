"""Reset database sequences after manual data manipulation"""
import asyncio
import asyncpg
import os

async def reset_sequences():
    """Reset all database sequences"""
    
    print("🔄 RESETTING DATABASE SEQUENCES")
    print("="*60)
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Get all sequences
        sequences = await conn.fetch("""
            SELECT 
                schemaname,
                sequencename,
                last_value
            FROM pg_sequences
            WHERE schemaname = 'public'
        """)
        
        print(f"\nFound {len(sequences)} sequences\n")
        
        for seq in sequences:
            seq_name = seq['sequencename']
            table_name = seq_name.replace('_id_seq', '')
            
            try:
                # Get max ID from table
                max_id = await conn.fetchval(f"""
                    SELECT COALESCE(MAX(id), 0) FROM {table_name}
                """)
                
                # Reset sequence
                await conn.execute(f"""
                    SELECT setval('{seq_name}', $1, true)
                """, max_id)
                
                print(f"✅ {seq_name}: Reset to {max_id}")
                
            except Exception as e:
                print(f"⚠️  {seq_name}: Skipped ({str(e)[:50]})")
        
        print("\n✅ Sequence reset complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_sequences())