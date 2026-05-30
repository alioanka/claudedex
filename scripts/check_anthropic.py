#!/usr/bin/env python3
"""
Anthropic key + model diagnostic (Wave-13).

Pulls the DECRYPTED Anthropic key exactly the way the bot does — via the
secrets manager (Docker secret -> encrypted secure_credentials DB row -> env) —
then (1) lists the models your key actually supports and (2) does a 1-token
ping against the model currently configured in ai_config.claude_model.

Run INSIDE the bot container so it has DB creds + the encryption key:
    docker exec trading-bot python scripts/check_anthropic.py

Read-only. Never prints the full key (masked to first/last 6 chars).
"""
import asyncio
import os
import sys

import aiohttp


def _mask(k: str) -> str:
    if not k:
        return "<none>"
    return f"{k[:10]}...{k[-6:]} (len={len(k)})" if len(k) > 20 else "<short>"


async def main() -> int:
    # 1) Bring up DB pool so the secrets manager can decrypt DB-stored keys
    #    (mirrors modules/ai_analysis/main_ai.py bootstrap order).
    db_pool = None
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url)
    except Exception as e:
        print(f"⚠️  Could not open DB pool ({e}); secrets manager will use env-only.")

    from security.secrets_manager import secrets as _secrets
    _secrets.initialize(db_pool)

    key = await _secrets.get_async("ANTHROPIC_API_KEY", log_access=False)
    if not key:
        key = os.getenv("ANTHROPIC_API_KEY")
    print(f"🔑 Anthropic key (decrypted, source=secrets_manager): {_mask(key)}")
    if not key:
        print("❌ No key resolved. Set it via /settings/credentials or ANTHROPIC_API_KEY.")
        return 1

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # 2) List models the key actually supports
    print("\n📋 Models available to this key (GET /v1/models):")
    available = []
    async with aiohttp.ClientSession() as s:
        async with s.get("https://api.anthropic.com/v1/models", headers=headers) as r:
            body = await r.json()
            if r.status != 200:
                print(f"   HTTP {r.status}: {body}")
            else:
                for m in body.get("data", []):
                    mid = m.get("id")
                    available.append(mid)
                    print(f"   • {mid}")
            if not available and r.status == 200:
                print("   (none returned)")

    # 3) Ping the currently-configured model
    configured = None
    if db_pool:
        try:
            async with db_pool.acquire() as c:
                configured = await c.fetchval(
                    "SELECT value FROM config_settings "
                    "WHERE config_type='ai_config' AND key='claude_model'"
                )
        except Exception as e:
            print(f"\n⚠️  Could not read configured model: {e}")
    print(f"\n🎯 Configured ai_config.claude_model = {configured!r}")

    test_model = configured or (available[0] if available else "claude-3-5-sonnet-20241022")
    print(f"🧪 Test ping against: {test_model}")
    payload = {
        "model": test_model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    async with aiohttp.ClientSession() as s:
        async with s.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as r:
            body = await r.json()
            if r.status == 200:
                print(f"   ✅ {test_model} WORKS (HTTP 200).")
            else:
                print(f"   ❌ HTTP {r.status}: {body.get('error', body)}")
                if available:
                    print(f"\n👉 Set a working model in DB, e.g.:")
                    print(f"   UPDATE config_settings SET value='{available[0]}' "
                          f"WHERE config_type='ai_config' AND key='claude_model';")

    if db_pool:
        await db_pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
