# ============================================
# scripts/test_alerts.py
# ============================================
"""Test alert system functionality"""
import asyncio
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.alerts import AlertsSystem, AlertType, AlertPriority

async def test_alerts():
    """Test all alert channels"""
    
    config = {
        "telegram": {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID")
        },
        "discord": {
            "webhook_url": os.getenv("DISCORD_WEBHOOK_URL")
        }
    }
    
    alerts = AlertsSystem(config)
    
    print("🔔 Testing Alert System...")
    
    # Test Telegram
    if config["telegram"]["bot_token"]:
        print("📱 Testing Telegram...")
        success = await alerts.send_telegram("🧪 Test alert from ClaudeDex Bot")
        print(f"  {'✅' if success else '❌'} Telegram")
    
    # Test Discord
    if config["discord"]["webhook_url"]:
        print("💬 Testing Discord...")
        success = await alerts.send_discord("🧪 Test alert from ClaudeDex Bot")
        print(f"  {'✅' if success else '❌'} Discord")
    
    # Test critical alert
    await alerts.send_alert(
        alert_type="system_error",
        message="This is a test critical alert",
        data={"test": True, "severity": "critical"}
    )
    
    print("\n✅ Alert testing complete!")

if __name__ == "__main__":
    asyncio.run(test_alerts())