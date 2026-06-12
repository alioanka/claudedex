"""META_CONTROLLER — self-deciding / self-improving advisory layer.

Reads every trading module's rolling performance (DRY_RUN and LIVE
separated), produces transparent ACTIVATE / KEEP / PAUSE decisions with
a per-module health score, and runs a daily self-improvement loop that
scores its own calibration and proposes bounded tunable nudges.

ADVISORY BY DEFAULT. Autopilot (flag-gated, default OFF) may only
write/clear logs/.pause_<module> flags and apply bounded config nudges.
It NEVER places trades, NEVER touches logs/.killswitch, and NEVER
bypasses a risk gate.
"""
