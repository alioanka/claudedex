"""
Route-authorization self-test (Wave-F6 RBAC).

Walks every registered route on an aiohttp app and reports mutating routes
(POST/PUT/DELETE/PATCH) whose handler does NOT carry an explicit RBAC gate
(the ``__rbac_gate__`` marker set by auth.middleware.require_admin /
require_operator / require_role).

Routes without an explicit gate are still protected by two layers —
session auth (auth_middleware_factory) and the VIEWER write-floor
(default-deny on all mutating methods) — but they lack the finer
admin-vs-operator distinction, so we surface them loudly at startup
instead of letting the gap go unnoticed for another wave.

Usage (dashboard startup):
    from auth.route_authz import log_route_authz_report
    log_route_authz_report(self.app, logger)
"""
import logging
from typing import List, Tuple

# Mutating endpoints that are deliberately reachable by every authenticated
# role (self-service auth) or unauthenticated (login). Not flagged.
_EXPECTED_UNGATED = frozenset((
    '/api/auth/login',
    '/api/auth/logout',
    '/api/auth/change-password',
))

_MUTATING_METHODS = frozenset(('POST', 'PUT', 'DELETE', 'PATCH'))


def audit_mutating_routes(app) -> List[Tuple[str, str]]:
    """Return [(method, path)] for mutating routes with no __rbac_gate__."""
    ungated = []
    for route in app.router.routes():
        method = (route.method or '').upper()
        if method not in _MUTATING_METHODS:
            continue
        resource = route.resource
        path = getattr(resource, 'canonical', None) or str(resource)
        if path in _EXPECTED_UNGATED or path.startswith('/socket.io'):
            continue
        if getattr(route.handler, '__rbac_gate__', None) is None:
            ungated.append((method, path))
    return sorted(set(ungated))


def log_route_authz_report(app, logger: logging.Logger = None) -> List[Tuple[str, str]]:
    """Log the self-test result; returns the ungated list for callers/tests.

    Fail-soft by design: the dashboard must come up even if a route table
    quirk breaks introspection, so this never raises.
    """
    log = logger or logging.getLogger(__name__)
    try:
        ungated = audit_mutating_routes(app)
    except Exception as e:  # pragma: no cover - introspection must not kill startup
        log.error(f"Route-authz self-test failed to run: {e}")
        return []
    if not ungated:
        log.info("Route-authz self-test: all mutating routes carry an explicit RBAC gate")
        return []
    log.warning(
        "Route-authz self-test: %d mutating route(s) rely ONLY on the "
        "session-auth + viewer write-floor layers (no explicit "
        "admin/operator gate):", len(ungated),
    )
    for method, path in ungated:
        log.warning("  UNGATED %s %s", method, path)
    return ungated
