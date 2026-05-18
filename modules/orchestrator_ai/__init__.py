"""Orchestrator AI module — advisory layer that scores each trading
module's recent performance and recommends enable/disable/dry/live
toggles to the operator.

Entry point: main_orchestrator_ai.py (added when wired into main.py).
Engine    : core/orchestrator_engine.py
Scoring   : core/performance_scorer.py
"""
