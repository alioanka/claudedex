# scripts/health_check.py
"""
Health check script for monitoring
"""
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
from datetime import datetime
import sys

async def check_health():
    """Check health of all components"""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {}
    }
    
    # Check API
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health") as response:
                if response.status == 200:
                    health_status["checks"]["api"] = {"status": "healthy"}
                else:
                    health_status["checks"]["api"] = {"status": "unhealthy", "code": response.status}
                    health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["api"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Database
    try:
        conn = await asyncpg.connect(
            "postgresql://trading:trading123@localhost:5432/tradingbot"
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        health_status["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        r = redis.from_url("redis://localhost:6379/0")
        await r.ping()
        await r.close()
        health_status["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    return health_status

async def main():
    """Main health check"""
    health = await check_health()
    
    print(f"Health Status: {health['status']}")
    for component, status in health["checks"].items():
        icon = "✅" if status["status"] == "healthy" else "❌"
        print(f"{icon} {component}: {status['status']}")
    
    if health["status"] != "healthy":
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())