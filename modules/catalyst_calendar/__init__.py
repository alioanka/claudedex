"""CATALYST_CALENDAR module — advisory forward-catalyst feed.

Aggregates known forward catalysts (token unlock cliffs, exchange listing
announcements, scheduled macro events) from free sources into the `catalysts`
table. PURE ADVISORY: never trades, never writes flags, no keys required.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
