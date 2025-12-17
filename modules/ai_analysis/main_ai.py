#!/usr/bin/env python3
"""
AI Analysis Module - Entry Point
"""
import sys
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load env
load_dotenv()

# Setup Logging
log_dir = Path("logs/ai_analysis")
log_dir.mkdir(parents=True, exist_ok=True)

# Formatters
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger
logger = logging.getLogger("AIModule")
logger.setLevel(logging.INFO)

# 1. Main Log
main_handler = RotatingFileHandler(log_dir / 'ai.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
main_handler.setLevel(logging.INFO)
logger.addHandler(main_handler)

# 2. Error Log
error_handler = RotatingFileHandler(log_dir / 'ai_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)

# Console
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

from modules.ai_analysis.core.sentiment_engine import SentimentEngine
from config.config_manager import ConfigManager

async def main():
    logger.info("🧠 AI Analysis Module Starting...")

    # Init DB
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("No DATABASE_URL found")
        return

    import asyncpg
    db_pool = await asyncpg.create_pool(db_url)

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Engine
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY')
    }
    engine = SentimentEngine(config, db_pool)
    await engine.initialize()

    try:
        await engine.run()
    except KeyboardInterrupt:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
