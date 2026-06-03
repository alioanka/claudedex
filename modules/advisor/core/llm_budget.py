"""
modules/advisor/core/llm_budget.py — re-export of the bot-wide LLM budget.

The hard daily paid-LLM cap is now a SINGLE global counter shared across all
modules (Advisor + AI Analysis), implemented in `core/llm_budget.py`. This shim
preserves the advisor's existing import path
(`from modules.advisor.core.llm_budget import try_consume`) while pointing at the
one shared budget file, so KAP classification and advice rationale draw from the
same daily allowance as the AI module.

See `core/llm_budget.py` for the implementation and config/env knobs.
"""
from __future__ import annotations

from core.llm_budget import snapshot, try_consume  # noqa: F401

__all__ = ["try_consume", "snapshot"]
