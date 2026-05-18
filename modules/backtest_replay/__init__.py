"""Backtest replay — counterfactual simulator over the trade history.

NOT a subprocess. Invoked synchronously via POST /api/backtest/replay
which calls into core.replay_engine.run_replay(). Pure compute over
already-recorded DB rows; no live execution path.
"""
