"""TREASURY module — Phase 1 OBSERVE-ONLY wallet/gas/inventory observer.

Reads public wallet addresses, polls native + key token balances via the
shared RPC pool, reconciles against the trade ledgers, writes
treasury_snapshots, and logs alerts. NEVER signs, NEVER transfers, NEVER
touches the killswitch. Default DISABLED (TREASURY_MODULE_ENABLED=true opt-in).
"""

__version__ = "1.0.0"
