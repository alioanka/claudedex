"""
CSRF Protection — double-submit-cookie pattern (MB-27).

Auth in the dashboard is cookie-based, which makes every POST/PUT/DELETE
vulnerable to CSRF unless requests prove they have read the cookie. We use
the stateless double-submit pattern: set a random token in a non-HTTPOnly
cookie, require the same token in the X-CSRF-Token header on state-changing
requests, and 403 on any mismatch.
"""
import os
import secrets
from typing import Callable

from aiohttp import web

CSRF_COOKIE = 'csrf_token'
CSRF_HEADER = 'X-CSRF-Token'

# Methods that mutate state — POST/PUT/DELETE/PATCH always need a token.
_PROTECTED_METHODS = frozenset(('POST', 'PUT', 'DELETE', 'PATCH'))

# Endpoints exempt from CSRF. /api/auth/login is the entrypoint (no cookie
# yet); /api/auth/logout is safe to allow either way.
_EXEMPT_PATHS = frozenset((
    '/api/auth/login',
    '/api/auth/logout',
))

# Path prefixes exempt from CSRF. Socket.IO long-poll uploads POST to
# /socket.io/?... and don't carry a CSRF token (the protocol manages its
# own session via sid). Auth middleware still gates these via session_id
# cookie, so dropping CSRF here doesn't lower the security floor.
_EXEMPT_PREFIXES = (
    '/socket.io/',
)


def _new_token() -> str:
    return secrets.token_urlsafe(32)


async def csrf_middleware_factory(app: web.Application, handler: Callable) -> Callable:
    async def middleware(request: web.Request):
        method = request.method.upper()
        path = request.path

        # State-changing requests must present a matching token unless exempt.
        is_exempt = (
            path in _EXEMPT_PATHS
            or any(path.startswith(p) for p in _EXEMPT_PREFIXES)
        )
        if method in _PROTECTED_METHODS and not is_exempt:
            cookie_token = request.cookies.get(CSRF_COOKIE)
            header_token = request.headers.get(CSRF_HEADER)
            if not cookie_token or not header_token or not secrets.compare_digest(
                cookie_token, header_token
            ):
                return web.json_response(
                    {'error': 'CSRF token missing or invalid'},
                    status=403,
                )

        response = await handler(request)

        # Make sure the client always has a fresh csrf_token cookie. Reuse the
        # existing token value when present (one stable token per session) but
        # REFRESH its expiry on every response so a long-lived tab never starts
        # 403'ing mid-session (sliding 7-day window).
        existing = request.cookies.get(CSRF_COOKIE)
        if hasattr(response, 'set_cookie'):
            token = existing or _new_token()
            # Secure-cookie decision is SCHEME-AWARE. Defaulting Secure=True on a
            # plain-HTTP deployment (the normal :8080 setup) made browsers
            # silently drop the cookie -> JS had no token -> every POST 403'd.
            # 'auto' (default): Secure only when the request is actually HTTPS
            # (direct or via X-Forwarded-Proto behind a TLS-terminating proxy).
            # Explicit DASHBOARD_HTTPS=true/false still forces the choice.
            env = os.getenv('DASHBOARD_HTTPS', 'auto').lower()
            if env in ('false', '0', 'no'):
                secure = False
            elif env in ('true', '1', 'yes'):
                secure = True
            else:  # 'auto'
                fwd = request.headers.get('X-Forwarded-Proto', '').lower()
                secure = request.scheme == 'https' or fwd == 'https'
            response.set_cookie(
                CSRF_COOKIE,
                token,
                httponly=False,  # JS must read this to echo it in the header
                secure=secure,
                samesite='Lax',
                max_age=7 * 24 * 3600,
                path='/',
            )
        return response

    return middleware
