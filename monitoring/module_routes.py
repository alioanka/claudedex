"""
Module Management Routes for Dashboard

API endpoints for managing trading modules through the web interface
"""

import logging
from aiohttp import web
from typing import Dict, Optional

from auth.middleware import require_auth, require_admin

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
        app.router.add_get('/modules/{module_name}/details', self.module_detail_page)
        app.router.add_get('/modules/{module_name}/configure', self.module_configure_page)

        # DEX Module Pages
        app.router.add_get('/dex/dashboard', self.dex_dashboard_page)
        app.router.add_get('/dex/settings', self.dex_settings_page)

        # Futures Module Pages
        app.router.add_get('/futures/dashboard', self.futures_dashboard_page)
        app.router.add_get('/futures/positions', self.futures_positions_page)
        app.router.add_get('/futures/trades', self.futures_trades_page)
        app.router.add_get('/futures/performance', self.futures_performance_page)
        app.router.add_get('/futures/settings', self.futures_settings_page)

        # Solana Module Pages
        app.router.add_get('/solana/dashboard', self.solana_dashboard_page)
        app.router.add_get('/solana/positions', self.solana_positions_page)
        app.router.add_get('/solana/trades', self.solana_trades_page)
        app.router.add_get('/solana/performance', self.solana_performance_page)
        app.router.add_get('/solana/settings', self.solana_settings_page)

        # Module Control Page
        app.router.add_get('/module-control', self.module_control_page)

        # Bot Control API endpoints (MB-28: admin-gate all state-changing bot ops)
        app.router.add_post('/api/bot/start', require_auth(require_admin(self.bot_start)))
        app.router.add_post('/api/bot/stop', require_auth(require_admin(self.bot_stop)))
        app.router.add_post('/api/bot/restart', require_auth(require_admin(self.bot_restart)))
        app.router.add_post('/api/bot/emergency-exit', require_auth(require_admin(self.bot_emergency_exit)))
        # MB-31: backwards-compat alias - underscore form used by older JS/templates
        app.router.add_post('/api/bot/emergency_exit', require_auth(require_admin(self.bot_emergency_exit)))

        # API endpoints
        app.router.add_get('/api/modules', self.get_modules_status)
        app.router.add_get('/api/modules/{module_name}', self.get_module_status)
        app.router.add_post('/api/modules/{module_name}/start', self.start_module)
        app.router.add_post('/api/modules/{module_name}/enable', self.enable_module)
        app.router.add_post('/api/modules/{module_name}/disable', self.disable_module)
        # MB-30: admin-gated; handlers also write logs/.pause_<module> flag
        # so subprocess loops actually see the pause (in-process flip alone
        # never reached spawned trading-module subprocesses).
        app.router.add_post('/api/modules/{module_name}/pause',
                            require_auth(require_admin(self.pause_module)))
        app.router.add_post('/api/modules/{module_name}/resume',
                            require_auth(require_admin(self.resume_module)))
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
        """Retired page — 302 to the generic /module/<key> panel
        (Wave-F5 IA cleanup). The old module_details.html render path
        was unreachable from the nav and depended on error.html, which
        never existed (guaranteed 500 on any failure)."""
        module_name = request.match_info['module_name']
        raise web.HTTPFound(f'/module/{module_name}')

    async def module_configure_page(self, request: web.Request) -> web.Response:
        """Retired page — 302 to the generic settings engine
        (Wave-F5 IA cleanup). module_configure.html never existed, so
        this route rendered an inline placeholder at best; per-module
        config lives at /config/<config_type>."""
        module_name = request.match_info['module_name']
        raise web.HTTPFound(f'/config/{module_name}_config')

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

    async def start_module(self, request: web.Request) -> web.Response:
        """
        Start a module

        Args:
            request: HTTP request with module_name in path

        Returns:
            web.Response: JSON response
        """
        try:
            module_name = request.match_info['module_name']

            success = await self.module_manager.start_module(module_name)

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} started'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': f'Failed to start module {module_name}'
                }, status=400)

        except Exception as e:
            self.logger.error(f"Error starting module: {e}", exc_info=True)
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

            # MB-30: flip cross-process flag FIRST so subprocesses halt new live
            # writes via should_skip_live() even if the in-process call fails.
            from core.dry_run import set_module_pause
            flag_ok = set_module_pause(module_name, True)

            success = await self.module_manager.pause_module(module_name)

            if success or flag_ok:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} paused',
                    'cross_process': flag_ok,
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

            # MB-30: delete the cross-process pause flag so subprocesses
            # resume honoring their own dry_run / kill-switch state.
            from core.dry_run import set_module_pause
            flag_ok = set_module_pause(module_name, False)

            success = await self.module_manager.resume_module(module_name)

            if success or flag_ok:
                return web.json_response({
                    'success': True,
                    'message': f'Module {module_name} resumed',
                    'cross_process': flag_ok,
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

    # ========== Module-Specific Dashboard Pages ==========

    async def dex_dashboard_page(self, request: web.Request) -> web.Response:
        """Render DEX module dashboard"""
        try:
            template = self.jinja_env.get_template('dashboard.html')
            html = template.render(page='dex_dashboard')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering DEX dashboard: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def futures_dashboard_page(self, request: web.Request) -> web.Response:
        """Render Futures module dashboard"""
        try:
            template = self.jinja_env.get_template('dashboard_futures.html')
            html = template.render(page='futures_dashboard')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Futures dashboard: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def futures_positions_page(self, request: web.Request) -> web.Response:
        """Render Futures positions page"""
        try:
            template = self.jinja_env.get_template('positions_futures.html')
            html = template.render(page='futures_positions')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Futures positions: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def futures_trades_page(self, request: web.Request) -> web.Response:
        """Render Futures trades page"""
        try:
            template = self.jinja_env.get_template('trades_futures.html')
            html = template.render(page='futures_trades')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Futures trades: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def futures_performance_page(self, request: web.Request) -> web.Response:
        """Render Futures performance page"""
        try:
            template = self.jinja_env.get_template('performance_futures.html')
            html = template.render(page='futures_performance')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Futures performance: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def solana_dashboard_page(self, request: web.Request) -> web.Response:
        """Render Solana module dashboard"""
        try:
            template = self.jinja_env.get_template('dashboard_solana.html')
            html = template.render(page='solana_dashboard')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Solana dashboard: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def solana_positions_page(self, request: web.Request) -> web.Response:
        """Render Solana positions page"""
        try:
            template = self.jinja_env.get_template('positions_solana.html')
            html = template.render(page='solana_positions')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Solana positions: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def solana_trades_page(self, request: web.Request) -> web.Response:
        """Render Solana trades page"""
        try:
            template = self.jinja_env.get_template('trades_solana.html')
            html = template.render(page='solana_trades')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Solana trades: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def solana_performance_page(self, request: web.Request) -> web.Response:
        """Render Solana performance page"""
        try:
            template = self.jinja_env.get_template('performance_solana.html')
            html = template.render(page='solana_performance')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Solana performance: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def dex_settings_page(self, request: web.Request) -> web.Response:
        """Render DEX settings page"""
        try:
            template = self.jinja_env.get_template('settings_dex.html')
            html = template.render(page='dex_settings')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering DEX settings: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def futures_settings_page(self, request: web.Request) -> web.Response:
        """Render Futures settings page"""
        try:
            template = self.jinja_env.get_template('settings_futures.html')
            html = template.render(page='futures_settings')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Futures settings: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def solana_settings_page(self, request: web.Request) -> web.Response:
        """Render Solana settings page"""
        try:
            template = self.jinja_env.get_template('settings_solana.html')
            html = template.render(page='solana_settings')
            return web.Response(text=html, content_type='text/html')
        except Exception as e:
            self.logger.error(f"Error rendering Solana settings: {e}", exc_info=True)
            return web.Response(text=f"Error: {e}", status=500)

    async def module_control_page(self, request: web.Request) -> web.Response:
        """Retired page — /modules is the single module-control hub
        (Wave-F5 IA consolidation). Old bookmarks 302 there."""
        raise web.HTTPFound('/modules')

    async def bot_start(self, request: web.Request) -> web.Response:
        """Start all enabled modules"""
        try:
            self.logger.info("Bot start requested")
            started_modules = []
            
            for module in self.module_manager.get_all_modules():
                if module.config.enabled and module.status != 'RUNNING':
                    await module.start()
                    started_modules.append(module.config.name)
            
            return web.json_response({
                'success': True,
                'message': f'Started {len(started_modules)} modules',
                'modules': started_modules
            })
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def bot_stop(self, request: web.Request) -> web.Response:
        """Stop all modules"""
        try:
            self.logger.info("Bot stop requested")
            stopped_modules = []
            
            for module in self.module_manager.get_all_modules():
                if module.status == 'RUNNING':
                    await module.stop()
                    stopped_modules.append(module.config.name)
            
            return web.json_response({
                'success': True,
                'message': f'Stopped {len(stopped_modules)} modules',
                'modules': stopped_modules
            })
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def bot_restart(self, request: web.Request) -> web.Response:
        """Restart all modules"""
        try:
            self.logger.info("Bot restart requested")
            
            # Stop all modules
            for module in self.module_manager.get_all_modules():
                if module.status == 'RUNNING':
                    await module.stop()
            
            # Wait a moment
            import asyncio
            await asyncio.sleep(2)
            
            # Start enabled modules
            started_modules = []
            for module in self.module_manager.get_all_modules():
                if module.config.enabled:
                    await module.start()
                    started_modules.append(module.config.name)
            
            return web.json_response({
                'success': True,
                'message': f'Restarted {len(started_modules)} modules',
                'modules': started_modules
            })
        except Exception as e:
            self.logger.error(f"Error restarting bot: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def bot_emergency_exit(self, request: web.Request) -> web.Response:
        """Emergency: Close all positions and stop all modules"""
        try:
            self.logger.warning("EMERGENCY EXIT requested!")

            # MB-31: flip kill switch FIRST so even if downstream steps fail,
            # new live-write attempts will be blocked process-wide.
            try:
                from core.dry_run import set_global_kill_switch
                set_global_kill_switch(True)
                self.logger.warning("EMERGENCY EXIT: global kill switch SET")
            except Exception as e:
                self.logger.error(f"Failed to set global kill switch: {e}")

            # MB-31: write flag file so other processes polling logs/.killswitch
            # also halt (cross-process effect).
            try:
                from pathlib import Path
                import json, os
                from datetime import datetime, timezone
                flag = Path("logs/.killswitch")
                flag.parent.mkdir(parents=True, exist_ok=True)
                flag.write_text(json.dumps({
                    "reason": "/api/bot/emergency-exit HTTP",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                }))
            except Exception as e:
                self.logger.error(f"Failed to write killswitch flag file: {e}")

            closed_positions = []
            
            # Close all positions in all modules
            for module in self.module_manager.get_all_modules():
                try:
                    positions = await module.get_positions()
                    for position in positions:
                        await module.close_position(position)
                        closed_positions.append({
                            'module': module.config.name,
                            'position': position
                        })
                except Exception as e:
                    self.logger.error(f"Error closing positions in {module.config.name}: {e}")
            
            # Stop all modules
            for module in self.module_manager.get_all_modules():
                await module.stop()
            
            return web.json_response({
                'success': True,
                'message': f'Emergency exit completed. Closed {len(closed_positions)} positions.',
                'closed_positions': len(closed_positions)
            })
        except Exception as e:
            self.logger.error(f"Error in emergency exit: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
