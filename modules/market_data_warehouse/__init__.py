"""MARKET_DATA_WAREHOUSE module — unified historical market-data store.

PURE DATA. Periodically captures normalized OHLCV candles and scalar series
(funding rates, etc.) from free public sources into compact, deduplicated
Postgres tables so other modules (backtest_replay, regime_allocator,
param_tuner, options_vol, advisor/ML retraining) read one consistent history
instead of each re-fetching. Never trades, never signs, needs no API keys.

Read accessor for consumers: modules.market_data_warehouse.reader
"""
