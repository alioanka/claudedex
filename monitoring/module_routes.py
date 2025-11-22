"""
Module Management Routes for Dashboard

API endpoints for managing trading modules through the web interface
"""

import logging
from aiohttp import web
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModuleRoutes:
    """
    Module management API routes

    Provides endpoints for:
    - Viewing module status
    - Enabling/disabling modules
    - Pausing/resuming modules
    - Configuring modules
    - Viewing module metrics
    """

    def __init__(self, module_manager, jinja_env=None):
        """
        Initialize module routes

        Args:
            module_manager: ModuleManager instance
            jinja_env: Jinja2 environment for templates
        """
        self.module_manager = module_manager
        self.jinja_env = jinja_env
        self.logger = logging.getLogger("ModuleRoutes")

    def setup_routes(self, app: web.Application):
        """
        Setup module routes on the application

        Args:
            app: aiohttp web application
        """
        # HTML pages
        app.router.add_get('/modules', self.modules_page)
        app.router.add_get('/modules/{module_name}', self.module_detail_page)

        # API endpoints
        app.router.add_get('/api/modules', self.get_modules_status)
        app.router.add_get('/api/modules/{module_name}', self.get_module_status)
        app.router.add_post('/api/modules/{module_name}/enable', self.enable_module)
        app.router.add_post('/api/modules/{module_name}/disable', self.disable_module)
        app.router.add_post('/api/modules/{module_name}/pause', self.pause_module)
        app.router.add_post('/api/modules/{module_name}/resume', self.resume_module)
        app.router.add_get('/api/modules/{module_name}/metrics', self.get_module_metrics)
        app.router.add_get('/api/modules/{module_name}/positions', self.get_module_positions)
        app.router.add_post('/api/modules/reallocate', self.reallocate_capital)

        self.logger.info("Module routes registered")

    async def modules_page(self, request: web.Request) -> web.Response:
        """
        Render modules management page

        Args:
            request: HTTP request

        Returns:
            web.Response: HTML response
        """
        try:
            if not self.module_manager:
                return web.Response(
                    text="Module manager not available",
                    status=500
                )

            # Get module status
            modules = []
            for module in self.module_manager.get_all_modules():
                status = module.get_status()
                modules.append(status)

            # Get aggregated metrics
            metrics = await self.module_manager.get_aggregated_metrics()

            # Render template
            if self.jinja_env:
                template = self.jinja_env.get_template('modules.html')
                html = template.render(
                    modules=modules,
                    metrics=metrics
                )
                return web.Response(text=html, content_type='text/html')
            else:
                return web.json_response({
                    'modules': modules,
                    'metrics': metrics
                })

        except Exception as e:
            self.logger.error(f"Error rendering modules page: {e}", exc_info=True)
            return web.Response(text=f"Error: {str(e)}", status=500)

    async def module_detail_page(self, request: web.Request) -> web.Response:
        """
        Render module detail page

        Args:
            request: HTTP request

        Returns:
            web.Response: HTML response
        """
        try:
            module_name = request.match_info['module_name']

            module = self.module_manager.get_module(module_name)
            if not module:
                return web.Response(text="Module not found", status=404)

            status = module.get_status()
            metrics = await module.get_metrics()
            positions = await module.get_positions()

            # For now, return JSON (TODO: create detail template)
            return web.json_response({
                'module': status,
                'metrics': metrics.to_dict(),
                'positions': positions
            })

        except Exception as e:
            self.logger.error(f"Error rendering module detail: {e}", exc_info=True)
            return web.Response(text=f"Error: {str(e)}", status=500)

    async def get_modules_status(self, request: web.Request) -> web.Response:
        """
        Get status of all modules

        Returns:
            web.Response: JSON response with modules status
        """
        try:
            if not self.module_manager:
                return web.json_response({
                    'success': False,
                    'error': 'Module manager not available'
                }, status=500)

            status = self.module_manager.get_status_summary()
            return web.json_response({
                'success': True,
                'data': status
            })

        except Exception as e:
            self.logger.error(f"Error getting modules status: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_module_status(self, request: web.Request) -> web.Response:
        """
        Get status of a specific module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response with module status
        """
        try:
            module_name = request.match_info['module_name']

            module = self.module_manager.get_module(module_name)
            if not module:
                return web.json_response({
                    'success': False,
                    'error': f'Module {module_name} not found'
                }, status=404)

            status = module.get_status()
            return web.json_response({
                'success': True,
                'data': status
            })

        except Exception as e:
            self.logger.error(f"Error getting module status: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def enable_module(self, request: web.Request) -> web.Response:
        """
        Enable a module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response
        """
        try:
            module_name = request.match_info['module_name']

            success = await self.module_manager.enable_module(module_name)

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} enabled'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to enable module {module_name}'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error enabling module: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def disable_module(self, request: web.Request) -> web.Response:
        """
        Disable a module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response
        """
        try:
            module_name = request.match_info['module_name']

            success = await self.module_manager.disable_module(module_name)

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} disabled'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to disable module {module_name}'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error disabling module: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def pause_module(self, request: web.Request) -> web.Response:
        """
        Pause a module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response
        """
        try:
            module_name = request.match_info['module_name']

            success = await self.module_manager.pause_module(module_name)

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} paused'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to pause module {module_name}'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error pausing module: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def resume_module(self, request: web.Request) -> web.Response:
        """
        Resume a module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response
        """
        try:
            module_name = request.match_info['module_name']

            success = await self.module_manager.resume_module(module_name)

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} resumed'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to resume module {module_name}'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error resuming module: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_module_metrics(self, request: web.Request) -> web.Response:
        """
        Get metrics for a specific module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response with metrics
        """
        try:
            module_name = request.match_info['module_name']

            module = self.module_manager.get_module(module_name)
            if not module:
                return web.json_response({
                    'success': False,
                    'error': f'Module {module_name} not found'
                }, status=404)

            metrics = await module.get_metrics()
            return web.json_response({
                'success': True,
                'data': metrics.to_dict()
            })

        except Exception as e:
            self.logger.error(f"Error getting module metrics: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_module_positions(self, request: web.Request) -> web.Response:
        """
        Get positions for a specific module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response with positions
        """
        try:
            module_name = request.match_info['module_name']

            module = self.module_manager.get_module(module_name)
            if not module:
                return web.json_response({
                    'success': False,
                    'error': f'Module {module_name} not found'
                }, status=404)

            positions = await module.get_positions()
            return web.json_response({
                'success': True,
                'data': positions
            })

        except Exception as e:
            self.logger.error(f"Error getting module positions: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def reallocate_capital(self, request: web.Request) -> web.Response:
        """
        Reallocate capital across modules

        Args:
            request: HTTP request with allocations in JSON body

        Returns:
            web.Response: JSON response
        """
        try:
            data = await request.json()
            allocations = data.get('allocations', {})

            if not allocations:
                return web.json_response({
                    'success': False,
                    'error': 'No allocations provided'
                }, status=400)

            success = await self.module_manager.reallocate_capital(allocations)

            if success:
                return web.json_response({
                    'success': True,
                    'message': 'Capital reallocated successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to reallocate capital'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error reallocating capital: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
