"""PARAM_TUNER — bandit-based, SHADOW-ONLY self-tuning of module knobs.

For an operator-registered whitelist of non-risk DB tunables (each with
hard [min, max] bounds), a deterministic UCB1 bandit accumulates rolling
performance evidence per discretized value and PROPOSES better settings
to the param_proposals table for operator approval.

PROPOSAL-ONLY BY DEFAULT. The auto_apply_enabled flag (default false)
may apply ONLY kind='exploit' proposals, ONLY within the operator's
bounds, ONLY to DB config_settings rows — logged to config_history and
reversible. It NEVER trades, NEVER touches code, env flags, risk knobs,
or logs/.killswitch.
"""
