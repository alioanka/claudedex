"""
Authentication Routes for Dashboard
Handles login, logout, user management, and session validation
"""
from aiohttp import web
import logging
from auth.middleware import get_client_ip, get_user_agent, require_auth, require_admin
from auth.models import UserRole

logger = logging.getLogger(__name__)


class AuthRoutes:
    """Authentication routes"""

    def __init__(self, app: web.Application, auth_service):
        self.app = app
        self.auth_service = auth_service
        self._setup_routes()

    def _setup_routes(self):
        """Setup authentication routes"""
        # Public routes
        self.app.router.add_get('/login', self.login_page)
        self.app.router.add_post('/api/auth/login', self.api_login)
        self.app.router.add_post('/api/auth/logout', self.api_logout)

        # Protected routes
        self.app.router.add_get('/api/auth/session', require_auth(self.api_get_session))
        self.app.router.add_get('/api/auth/users', require_auth(require_admin(self.api_get_users)))
        self.app.router.add_post('/api/auth/users', require_auth(require_admin(self.api_create_user)))
        self.app.router.add_put('/api/auth/users/{user_id}', require_auth(require_admin(self.api_update_user)))
        self.app.router.add_delete('/api/auth/users/{user_id}', require_auth(require_admin(self.api_delete_user)))
        self.app.router.add_post('/api/auth/change-password', require_auth(self.api_change_password))

        # User management page
        self.app.router.add_get('/users', require_auth(require_admin(self.users_page)))

    # ==================== Pages ====================

    async def login_page(self, request):
        """Serve login page"""
        # If already logged in, redirect to dashboard
        if 'user' in request:
            return web.HTTPFound('/')

        with open('dashboard/templates/login.html', 'r') as f:
            content = f.read()

        return web.Response(text=content, content_type='text/html')

    async def users_page(self, request):
        """Serve user management page"""
        with open('dashboard/templates/users.html', 'r') as f:
            content = f.read()

        return web.Response(text=content, content_type='text/html')

    # ==================== API Endpoints ====================

    async def api_login(self, request):
        """Login endpoint"""
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            totp_code = data.get('totp_code')

            if not username or not password:
                return web.json_response({
                    'success': False,
                    'error': 'Username and password required'
                }, status=400)

            # Get client info
            ip_address = get_client_ip(request)
            user_agent = get_user_agent(request)

            # Authenticate
            success, user, error = await self.auth_service.authenticate(
                username, password, totp_code, ip_address, user_agent
            )

            if not success:
                return web.json_response({
                    'success': False,
                    'error': error
                }, status=401)

            # Create session
            session_id = await self.auth_service.create_session(user.id, ip_address, user_agent)

            if not session_id:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to create session'
                }, status=500)

            # Set secure cookie
            response = web.json_response({
                'success': True,
                'user': user.to_dict()
            })

            response.set_cookie(
                'session_id',
                session_id,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite='Lax',
                max_age=3600  # 1 hour
            )

            logger.info(f"User {username} logged in successfully")
            return response

        except Exception as e:
            logger.error(f"Login error: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': 'Internal server error'
            }, status=500)

    async def api_logout(self, request):
        """Logout endpoint"""
        try:
            session_id = request.cookies.get('session_id')

            if session_id:
                await self.auth_service.invalidate_session(session_id)

            response = web.json_response({'success': True})
            response.del_cookie('session_id')

            # Try to get username if available
            username = request.get('user', {}).username if 'user' in request else 'unknown'
            logger.info(f"User {username} logged out")
            return response

        except Exception as e:
            logger.error(f"Logout error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_get_session(self, request):
        """Get current session info"""
        user = request['user']
        return web.json_response({
            'success': True,
            'user': user.to_dict()
        })

    async def api_get_users(self, request):
        """Get all users (admin only)"""
        try:
            users = await self.auth_service.get_all_users()
            return web.json_response({
                'success': True,
                'users': [u.to_dict() for u in users]
            })
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_create_user(self, request):
        """Create new user (admin only)"""
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            role = data.get('role', 'viewer')
            email = data.get('email')
            require_2fa = data.get('require_2fa', False)

            if not username or not password:
                return web.json_response({
                    'success': False,
                    'error': 'Username and password required'
                }, status=400)

            # Validate role
            try:
                user_role = UserRole(role)
            except ValueError:
                return web.json_response({
                    'success': False,
                    'error': f'Invalid role: {role}'
                }, status=400)

            # Create user
            user = await self.auth_service.create_user(
                username, password, user_role, email, require_2fa,
                created_by=request['user'].id
            )

            if not user:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to create user'
                }, status=500)

            logger.info(f"Admin {request['user'].username} created user {username}")
            return web.json_response({
                'success': True,
                'user': user.to_dict()
            })

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_update_user(self, request):
        """Update user (admin only)"""
        try:
            user_id = int(request.match_info['user_id'])
            data = await request.json()

            # Update role if provided
            if 'role' in data:
                try:
                    new_role = UserRole(data['role'])
                    await self.auth_service.update_user_role(user_id, new_role)
                except ValueError:
                    return web.json_response({
                        'success': False,
                        'error': f'Invalid role: {data["role"]}'
                    }, status=400)

            # Update password if provided
            if 'password' in data:
                await self.auth_service.change_password(user_id, data['password'])

            logger.info(f"Admin {request['user'].username} updated user {user_id}")
            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_delete_user(self, request):
        """Delete/deactivate user (admin only)"""
        try:
            user_id = int(request.match_info['user_id'])

            # Don't allow deleting yourself
            if user_id == request['user'].id:
                return web.json_response({
                    'success': False,
                    'error': 'Cannot delete your own account'
                }, status=400)

            await self.auth_service.deactivate_user(user_id)

            logger.info(f"Admin {request['user'].username} deactivated user {user_id}")
            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def api_change_password(self, request):
        """Change own password"""
        try:
            data = await request.json()
            old_password = data.get('old_password')
            new_password = data.get('new_password')

            if not old_password or not new_password:
                return web.json_response({
                    'success': False,
                    'error': 'Old and new password required'
                }, status=400)

            # Verify old password
            user = request['user']
            ip_address = get_client_ip(request)
            user_agent = get_user_agent(request)

            success, _, error = await self.auth_service.authenticate(
                user.username, old_password, None, ip_address, user_agent
            )

            if not success:
                return web.json_response({
                    'success': False,
                    'error': 'Invalid current password'
                }, status=401)

            # Change password
            await self.auth_service.change_password(user.id, new_password)

            # Invalidate all other sessions
            await self.auth_service.invalidate_all_user_sessions(user.id)

            logger.info(f"User {user.username} changed password")
            return web.json_response({'success': True})

        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
