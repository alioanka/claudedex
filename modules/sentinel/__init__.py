"""SENTINEL module — cross-module anomaly detection + auto-freeze advisor.

Watches distributions the per-trade gates cannot see (stablecoin depeg,
cross-source price divergence, silent module death, abnormal loss velocity,
correlated drawdown) on a minutes-scale tick. ADVISORY BY DEFAULT: records
graded anomaly events to sentinel_anomalies and logs them. Autopilot
(sentinel_autopilot_enabled, default false) may ONLY write/clear
logs/.pause_<module> freeze flags — never the killswitch, never a trade.
"""
