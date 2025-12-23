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

from modules.ai_analysis.core.sentiment_engine import SentimentEngine, AITradeExecutor
from config.config_manager import ConfigManager

# Also configure logging for the engine classes
for engine_name in ["SentimentEngine", "AITradeExecutor"]:
    engine_logger = logging.getLogger(engine_name)
    engine_logger.setLevel(logging.INFO)
    engine_logger.addHandler(main_handler)
    engine_logger.addHandler(error_handler)
    engine_logger.addHandler(console)

# Also add OpenAI API log
openai_log_handler = RotatingFileHandler(log_dir / 'openai_api.log', maxBytes=10*1024*1024, backupCount=5)
openai_log_handler.setFormatter(log_formatter)
openai_logger = logging.getLogger("OpenAI_API")
openai_logger.setLevel(logging.INFO)
openai_logger.addHandler(openai_log_handler)
openai_logger.addHandler(console)

# Also add Claude API log
claude_log_handler = RotatingFileHandler(log_dir / 'claude_api.log', maxBytes=10*1024*1024, backupCount=5)
claude_log_handler.setFormatter(log_formatter)
claude_logger = logging.getLogger("Claude_API")
claude_logger.setLevel(logging.INFO)
claude_logger.addHandler(claude_log_handler)
claude_logger.addHandler(console)

async def main():
    logger.info("🧠 AI Analysis Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

    # Check API Keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if openai_key:
        logger.info(f"   OpenAI API Key: {openai_key[:20]}...")
    else:
        logger.warning("⚠️ No OPENAI_API_KEY found")

    if anthropic_key:
        logger.info(f"   Anthropic API Key: {anthropic_key[:20]}...")
    else:
        logger.warning("⚠️ No ANTHROPIC_API_KEY found")

    if not openai_key and not anthropic_key:
        logger.error("❌ No AI API keys found - AI analysis will be disabled")

    # Init DB - Use Docker secrets or environment
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')

    if not db_url:
        logger.error("No database credentials found (Docker secrets or DATABASE_URL)")
        return

    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    # Init Config
    config_manager = ConfigManager()
    await config_manager.initialize()

    # Init Engine
    config = {
        'openai_api_key': openai_key,
        'anthropic_api_key': anthropic_key
    }
    engine = SentimentEngine(config, db_pool)

    try:
        await engine.initialize()
        logger.info("✅ Sentiment Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Engine initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    try:
        await engine.run()
    except KeyboardInterrupt:
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Engine error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
