"""
AdvisorConfigManager — loads all advisor_config rows from config_settings.

Pattern mirrors FuturesConfigManager (config/futures_config_manager.py):
  - SELECT key, value, value_type FROM config_settings
    WHERE config_type = 'advisor_config'
  - Casts value by value_type (boolean, integer, float, string, json).
  - Returns dict for use by AdviceEngine, analyzers, risk engine, etc.
  - No live-reload (module restarts pick up new config); future wave may
    add hot-reload if operator demand warrants.

Key inventory (seeded by migration 058):
  Core
    advisor_anthropic_model        : Anthropic model ID for LLM rationale
    advisor_anthropic_api_key      : (resolved via secrets_manager, NOT plain text)
    enabled_markets                : comma-sep Market values
    enabled_horizons               : comma-sep Horizon values
    run_interval_minutes           : how often to run advice cycle

  Risk / Sim
    min_confidence                 : float, advice gate
    max_sim_positions              : int, open sim cap
    blocked_symbols                : comma-sep blocklist
    sim_default_enabled            : bool, auto-open sim positions
    sim_default_amount_usd         : float, notional per sim

  Watchlists (per-market, comma-separated symbols)
    watchlist_crypto
    watchlist_us_equities
    watchlist_bist
    watchlist_fx
    watchlist_midas_funds

  Data sources
    advisor_crypto_exchange        : ccxt exchange id (default "binance")
    advisor_bist_data_source       : "yfinance" | "matriks" | ""
    advisor_fx_data_source         : "yfinance" | "alphavantage" | "stooq" | ""
    advisor_midas_data_source      : "tefas_scrape" | "manual" | ""

  Telegram (separate bot)
    advisor_telegram_enabled       : bool
    advisor_telegram_bot_token     : (resolved via secrets_manager)
    advisor_telegram_chat_id       : target chat ID

  Kronos
    advisor_kronos_enabled         : bool (default false)
    advisor_kronos_variant         : "Kronos-mini" | "Kronos-small" | "Kronos-base"
    advisor_kronos_device          : "cpu" | "cuda"

  ML
    advisor_ml_enabled             : bool (default false)
    advisor_ml_daily_learning      : bool (default false)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("advisor.config")

# config_type value in the config_settings table
ADVISOR_CONFIG_TYPE = "advisor_config"

# Fallback defaults (used when key is missing from DB)
DEFAULTS: Dict[str, Any] = {
    "advisor_anthropic_model": "claude-opus-4-5",
    "enabled_markets": "crypto,us_equities",
    "enabled_horizons": "short,mid,long",
    "run_interval_minutes": "60",
    "min_confidence": "0.35",
    "max_sim_positions": "20",
    "blocked_symbols": "",
    "sim_default_enabled": "false",
    "sim_default_amount_usd": "1000.0",
    "watchlist_crypto": "BTC/USDT,ETH/USDT",
    "watchlist_us_equities": "AAPL,MSFT,NVDA",
    "watchlist_bist": "",
    "watchlist_fx": "",
    "watchlist_midas_funds": "",
    "advisor_crypto_exchange": "binance",
    "advisor_bist_data_source": "",
    "advisor_fx_data_source": "yfinance",
    "advisor_midas_data_source": "",
    # Fonoloji optional data source (migrations 078/080) — OFF unless a key is
    # set. VERIFIED contract (migration 080): base /v1, X-API-Key (bare key).
    "advisor_fonoloji_api_key": "",
    "advisor_fonoloji_base_url": "https://fonoloji.com/v1",
    "advisor_fonoloji_auth_header": "X-API-Key",
    "advisor_fonoloji_cache_ttl_s": "21600",
    # Fonoloji AI-advisor layer (migration 081) — all gated on the Fonoloji
    # key. No key => behaviour UNCHANGED. ai_summary REPLACES the paid
    # Anthropic rationale for Turkish funds (a saving, not a swap). analyst
    # recommendations add a broker-consensus VOTE to the BIST directional
    # signal. market_digest is light/optional (default off).
    "advisor_fonoloji_ai_summary_enabled": "true",
    "advisor_bist_use_analyst_recommendations": "true",
    "advisor_fonoloji_market_digest_enabled": "false",
    # GEMS BIST screener batch knobs (migration 080).
    "advisor_discovery_bist_screener_enabled": "true",
    "advisor_discovery_bist_pe_max": "25",
    "advisor_discovery_bist_roe_min": "15",
    "advisor_discovery_bist_pb_max": "",
    "advisor_discovery_bist_div_min": "",
    "advisor_discovery_bist_sort_by": "roe",
    "advisor_discovery_bist_sort_order": "desc",
    "advisor_discovery_bist_screener_limit": "30",
    # BIST / FX universe scan (migrations 078 + 083). BIST default 'auto' =
    # Fonoloji live /stocks/list when a key is present (auto-prefer), else
    # watchlist-only — matches the migration-083 DB seed. An explicit
    # 'watchlist' row is an operator opt-out and is honored.
    "advisor_bist_universe": "auto",
    "advisor_bist_universe_custom": "",
    "advisor_fx_universe": "watchlist",
    "advisor_fx_universe_custom": "",
    "advisor_universe_max": "60",
    # Discovery robustness (migration 078).
    "advisor_discovery_us_fallback_universe": "",
    "advisor_discovery_bist_use_universe": "true",
    # KAP-driven sim moderate polarity (migration 078).
    "advisor_kap_sim_include_moderate": "false",
    "advisor_telegram_enabled": "true",
    "advisor_telegram_bot_token": "",
    "advisor_telegram_chat_id": "",
    "advisor_kronos_enabled": "false",
    "advisor_kronos_variant": "Kronos-mini",
    "advisor_kronos_device": "cpu",
    "advisor_ml_enabled": "false",
    "advisor_ml_daily_learning": "false",
}


class AdvisorConfigManager:
    """Loads advisor_config rows from config_settings table."""

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self._config: Dict[str, str] = dict(DEFAULTS)

    async def load(self) -> Dict[str, Any]:
        """
        Load config from DB. Falls back to DEFAULTS for any missing key.
        Always returns a usable dict even if DB is unreachable.
        """
        if self.db_pool is None:
            logger.warning("[config] No DB pool — using default advisor config.")
            return dict(self._config)

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value, value_type "
                    "FROM config_settings WHERE config_type=$1",
                    ADVISOR_CONFIG_TYPE,
                )
            db_values = {r["key"]: (r["value"], r.get("value_type", "string"))
                         for r in rows}

            # Merge: DB wins over defaults
            merged = dict(DEFAULTS)
            for key, (raw_value, vtype) in db_values.items():
                merged[key] = _cast(key, raw_value, vtype)

            self._config = merged
            logger.info(
                f"[config] Loaded {len(db_values)} advisor_config keys from DB."
            )
        except Exception as exc:
            logger.error(
                f"[config] Failed to load advisor_config from DB: {exc}. "
                "Using defaults."
            )

        return dict(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default if default is not None else DEFAULTS.get(key))


def _cast(key: str, value: str, vtype: str) -> Any:
    """Cast a raw string config value to the appropriate Python type."""
    try:
        if vtype == "boolean":
            return value.lower() in ("true", "1", "yes")
        if vtype == "integer":
            return int(value)
        if vtype == "float":
            return float(value)
        if vtype == "json":
            import json
            return json.loads(value)
        return value  # string
    except Exception:
        return value
