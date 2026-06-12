"""execution_quality — read-only Transaction Cost Analysis (TCA) module.

Measures quoted-vs-realized execution cost (slippage, fees, gas, extras,
suspected MEV) across every trading module's closed-trade tables and persists
per-trade rows (tca_trade_costs) + per-module scorecards (tca_scorecards).
PURELY OBSERVES: never trades, never writes a flag, zero market risk.
"""
