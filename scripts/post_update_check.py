"""Post-update system check"""
import asyncio
import asyncpg
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from web3 import Web3
import redis
import os

async def post_update_check():
    """Verify system after update"""
    
    print("POST-UPDATE CHECK")
    print("="*60)
    
    checks_passed = 0
    checks_failed = 0
    
    # Check database
    print("\n1. Checking database connection...")
    try:
        conn = await asyncpg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "tradingbot"),
            user=os.getenv("DB_USER", "trading"),
            password=os.getenv("DB_PASSWORD", "trading123")
        )
        
        # Check tables exist
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        
        required_tables = ['trades', 'positions', 'market_data', 'audit_logs']
        existing_tables = {t['tablename'] for t in tables}
        
        missing = set(required_tables) - existing_tables
        if missing:
            print(f"   ✗ Missing tables: {missing}")
            checks_failed += 1
        else:
            print("   ✓ Database OK")
            checks_passed += 1
        
        await conn.close()
        
    except Exception as e:
        print(f"   ✗ Database check failed: {str(e)}")
        checks_failed += 1
    
    # Check Redis
    print("\n2. Checking Redis connection...")
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )
        r.ping()
        print("   ✓ Redis OK")
        checks_passed += 1
    except Exception as e:
        print(f"   ✗ Redis check failed: {str(e)}")
        checks_failed += 1
    
    # Check Web3 connection
    print("\n3. Checking Web3 connection...")
    try:
        w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))
        if w3.is_connected():
            block = w3.eth.block_number
            print(f"   ✓ Web3 OK (block: {block})")
            checks_passed += 1
        else:
            print("   ✗ Web3 not connected")
            checks_failed += 1
    except Exception as e:
        print(f"   ✗ Web3 check failed: {str(e)}")
        checks_failed += 1
    
    # Check ML models
    print("\n4. Checking ML models...")
    model_dir = Path("ml/models")
    required_models = [
        "pump_predictor_latest.pkl",
        "rug_classifier_latest.pkl",
        "volume_validator_latest.pkl"
    ]
    
    missing_models = []
    for model in required_models:
        if not (model_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"   ⚠ Missing models: {missing_models}")
        print("   Run: python scripts/retrain_models.py")
        checks_failed += 1
    else:
        print("   ✓ ML models OK")
        checks_passed += 1
    
    # Check configuration files
    print("\n5. Checking configuration...")
    config_files = [
        "config/trading.yaml",
        "config/security.yaml",
        "config/api.yaml"
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"   ✗ Missing configs: {missing_configs}")
        checks_failed += 1
    else:
        print("   ✓ Configuration OK")
        checks_passed += 1
    
    # Check environment variables
    print("\n6. Checking environment variables...")
    required_vars = [
        'DB_PASSWORD', 'REDIS_HOST', 'ETH_RPC_URL',
        'WALLET_PRIVATE_KEY', 'DEXSCREENER_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"   ✗ Missing vars: {missing_vars}")
        checks_failed += 1
    else:
        print("   ✓ Environment OK")
        checks_passed += 1
    
    # Check file permissions
    print("\n7. Checking file permissions...")
    sensitive_files = ['.env', 'config/security.yaml']
    permission_issues = []
    
    for file in sensitive_files:
        if os.path.exists(file):
            stat = os.stat(file)
            mode = oct(stat.st_mode)[-3:]
            if mode not in ['600', '400']:
                permission_issues.append(f"{file} ({mode})")
    
    if permission_issues:
        print(f"   ⚠ Insecure permissions: {permission_issues}")
        print("   Run: chmod 600 .env config/security.yaml")
    else:
        print("   ✓ Permissions OK")
    checks_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {checks_passed} passed, {checks_failed} failed")
    
    if checks_failed == 0:
        print("\n✓ System is ready!")
        return True
    else:
        print("\n✗ Please fix the issues above")
        return False

if __name__ == "__main__":
    result = asyncio.run(post_update_check())
    sys.exit(0 if result else 1)