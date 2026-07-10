"""
Authentication Middleware
Provides route protection decorators and middleware functions
"""
from functools import wraps
from aiohttp import web
from typing import Callable, Optional
import logging

from .models import UserRole
from .auth_service import AuthService

logger = logging.getLogger(__name__)


def require_auth(handler: Callable) -> Callable:
    """
    Decorator to require authentication for a route
    Usage: @require_auth
    """
    @wraps(handler)
    async def middleware(request: web.Request):
        # Check if auth_service is available
        if 'auth_service' not in request.app:
            logger.error("AuthService not initialized in app")
            return web.json_response({'error': 'Authentication system not available'}, status=500)

        auth_service: AuthService = request.app['auth_service']

        # Get session_id from cookie
        session_id = request.cookies.get('session_id')

        if not session_id:
            # Return login redirect for HTML pages, 401 for API
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Authentication required'}, status=401)
            else:
                return web.HTTPFound('/login')

        # Validate session
        user = await auth_service.validate_session(session_id)

        if not user:
            # Session invalid or expired
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Session expired or invalid'}, status=401)
            else:
                response = web.HTTPFound('/login')
                response.del_cookie('session_id')
                return response

        # Attach user to request for access in handlers
        request['user'] = user

        # Call the actual handler
        return await handler(request)

    return middleware


def require_admin(handler: Callable) -> Callable:
    """
    Decorator to require admin role
    Must be used together with @require_auth
    Usage:
        @require_auth
        @require_admin
    """
    @wraps(handler)
    async def middleware(request: web.Request):
        user = request.get('user')

        if not user:
            logger.warning(
                "RBAC deny (no session) gate=admin route=%s %s",
                request.method, request.path,
            )
            return web.json_response({'error': 'Authentication required'}, status=401)

        if user.role != UserRole.ADMIN:
            logger.warning(
                "RBAC deny actor=%s role=%s gate=admin route=%s %s",
                user.username, user.role.value, request.method, request.path,
            )
            return web.json_response({'error': 'Admin access required'}, status=403)

        logger.info(
            "RBAC allow actor=%s role=%s gate=admin route=%s %s",
            user.username, user.role.value, request.method, request.path,
        )
        return await handler(request)

    # Marker consumed by auth.route_authz.audit_mutating_routes — lets the
    # startup self-test verify every mutating route carries an RBAC gate.
    # functools.wraps propagates __dict__, so require_auth(require_admin(h))
    # keeps the marker on the outermost wrapper.
    middleware.__rbac_gate__ = 'admin'
    return middleware


def require_operator(handler: Callable) -> Callable:
    """
    Wave-F6 RBAC: decorator to require OPERATOR or ADMIN role.

    The gate for state-changing trading/settings routes: VIEWER — and any
    unknown/future role, default-deny — gets 403. Use together with
    @require_auth, same registration style as require_admin:
        require_auth(require_operator(handler))
    Every decision emits an audit log line (actor, role, route, outcome).
    """
    @wraps(handler)
    async def middleware(request: web.Request):
        user = request.get('user')

        if not user:
            logger.warning(
                "RBAC deny (no session) gate=operator route=%s %s",
                request.method, request.path,
            )
            return web.json_response({'error': 'Authentication required'}, status=401)

        if user.role not in (UserRole.ADMIN, UserRole.OPERATOR):
            logger.warning(
                "RBAC deny actor=%s role=%s gate=operator route=%s %s",
                user.username, user.role.value, request.method, request.path,
            )
            return web.json_response(
                {'error': 'Operator or admin access required'}, status=403
            )

        logger.info(
            "RBAC allow actor=%s role=%s gate=operator route=%s %s",
            user.username, user.role.value, request.method, request.path,
        )
        return await handler(request)

    middleware.__rbac_gate__ = 'operator'
    return middleware


def require_role(required_role: UserRole) -> Callable:
    """
    Decorator to require specific role
    Usage:
        @require_auth
        @require_role(UserRole.OPERATOR)
    """
    def decorator(handler: Callable) -> Callable:
        @wraps(handler)
        async def middleware(request: web.Request):
            user = request.get('user')

            if not user:
                return web.json_response({'error': 'Authentication required'}, status=401)

            # Admin has access to everything
            if user.role == UserRole.ADMIN:
                return await handler(request)

            if user.role != required_role:
                logger.warning(
                    "RBAC deny actor=%s role=%s gate=%s route=%s %s",
                    user.username, user.role.value, required_role.value,
                    request.method, request.path,
                )
                return web.json_response(
                    {'error': f'Role {required_role.value} required'},
                    status=403
                )

            return await handler(request)

        middleware.__rbac_gate__ = f'role:{required_role.value}'
        return middleware

    return decorator


# Mutating endpoints every authenticated role may call — self-service auth
# only. Everything else that mutates state is denied to VIEWER by the
# default-deny floor in auth_middleware_factory below.
MUTATION_ALLOWED_ANY_ROLE = frozenset((
    '/api/auth/change-password',
    '/api/auth/logout',
))

_MUTATING_METHODS = frozenset(('POST', 'PUT', 'DELETE', 'PATCH'))


async def auth_middleware_factory(app: web.Application, handler: Callable) -> Callable:
    """
    Global middleware to enforce authentication on all routes
    Actively blocks unauthenticated requests except for public routes
    """
    async def middleware(request: web.Request):
        # Public routes that don't require authentication. /health is
        # public so Docker healthchecks and scripts/health_check.py can
        # probe without carrying an auth cookie.
        public_routes = [
            '/login',
            '/api/auth/login',
            '/api/auth/logout',
            '/health',
            '/__routes__',
        ]

        # Public path prefixes. /socket.io/ is gated by the Socket.IO
        # library's own session/origin checks; piggybacking aiohttp auth
        # on top breaks the protocol (XHR doesn't follow CORS 302 redirects).
        public_prefixes = (
            '/static/',
            '/socket.io/',
        )

        # Check if this is a public route
        is_public = (
            request.path in public_routes or
            any(request.path.startswith(p) for p in public_prefixes)
        )

        if is_public:
            return await handler(request)

        # For all other routes, authentication is required
        if 'auth_service' not in app:
            # Auth service not initialized - this is a critical error
            logger.error(f"🚨 SECURITY: Auth service not initialized, blocking access to {request.path}")
            if request.path.startswith('/api/'):
                return web.json_response({
                    'error': 'Authentication system not initialized',
                    'message': 'Server is starting up. Please wait and try again.'
                }, status=503)
            else:
                return web.Response(
                    text='<h1>Server Starting Up</h1><p>Authentication system is initializing. Please wait a moment and refresh.</p>',
                    content_type='text/html',
                    status=503
                )

        auth_service: AuthService = app['auth_service']
        session_id = request.cookies.get('session_id')

        if not session_id:
            # No session cookie - redirect to login
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Authentication required'}, status=401)
            else:
                return web.HTTPFound('/login')

        # Validate session
        user = await auth_service.validate_session(session_id)

        if not user:
            # Invalid or expired session - redirect to login
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Session expired or invalid'}, status=401)
            else:
                response = web.HTTPFound('/login')
                response.del_cookie('session_id')
                return response

        # Session valid - attach user to request
        request['user'] = user

        # Wave-F6 RBAC floor (default-deny): a VIEWER session may never
        # issue state-changing requests, regardless of whether the route
        # carries its own require_admin/require_operator wrapper. This
        # centrally covers mutating routes registered by modules the
        # per-route sweep didn't reach (module_routes, test_runner, ...).
        # Self-service auth endpoints stay open to every role.
        if (
            request.method.upper() in _MUTATING_METHODS
            and request.path not in MUTATION_ALLOWED_ANY_ROLE
            and user.role == UserRole.VIEWER
        ):
            logger.warning(
                "RBAC deny actor=%s role=viewer gate=write-floor route=%s %s",
                user.username, request.method, request.path,
            )
            return web.json_response(
                {'error': 'Read-only role: write access denied'}, status=403
            )

        return await handler(request)

    return middleware


def get_client_ip(request: web.Request) -> str:
    """Get client IP address from request"""
    # Check for X-Forwarded-For header (if behind proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()

    # Check for X-Real-IP header
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Fall back to remote address
    peername = request.transport.get_extra_info('peername')
    if peername:
        return peername[0]

    return 'unknown'


def get_user_agent(request: web.Request) -> str:
    """Get user agent from request"""
    return request.headers.get('User-Agent', 'unknown')
