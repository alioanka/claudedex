/**
 * Module Management JavaScript
 * Handles loading and controlling trading modules from the dashboard
 */

// Load modules when the Modules tab is clicked
async function loadModules() {
    const container = document.getElementById('modules-container');
    if (!container) return;

    try {
        const response = await fetch('/api/modules');
        const data = await response.json();

        if (!data.success) {
            container.innerHTML = `<div class="alert alert-danger">Error loading modules: ${data.error}</div>`;
            return;
        }

        const status = data.data;
        renderModules(status, container);

    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">Failed to load modules: ${error.message}</div>`;
    }
}

function renderModules(status, container) {
    const modules = status.modules || {};

    let html = '<div class="modules-grid">';

    // Add each module
    for (const [moduleName, module] of Object.entries(modules)) {
        const statusClass = module.status === 'running' ? 'success' : 'secondary';
        const statusIcon = module.status === 'running' ? '▶️' : '⏸️';

        html += `
            <div class="module-card" data-module="${moduleName}">
                <div class="module-header">
                    <h4>${module.name}</h4>
                    <span class="badge badge-${statusClass}">${statusIcon} ${module.status}</span>
                </div>
                <div class="module-body">
                    <p><strong>Type:</strong> ${module.module_type}</p>
                    <p><strong>Capital:</strong> $${module.metrics.capital_allocated.toFixed(2)}</p>
                    <p><strong>Positions:</strong> ${module.metrics.active_positions}</p>
                    <p><strong>PnL:</strong> <span class="${module.metrics.total_pnl >= 0 ? 'text-success' : 'text-danger'}">$${module.metrics.total_pnl.toFixed(2)}</span></p>
                </div>
                <div class="module-actions">
                    ${module.enabled ? `
                        <button class="btn btn-sm btn-warning" onclick="pauseModule('${moduleName}')">
                            <i class="fas fa-pause"></i> Pause
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="disableModule('${moduleName}')">
                            <i class="fas fa-stop"></i> Disable
                        </button>
                    ` : `
                        <button class="btn btn-sm btn-success" onclick="enableModule('${moduleName}')">
                            <i class="fas fa-play"></i> Enable
                        </button>
                    `}
                    <a href="/modules/${moduleName}" class="btn btn-sm btn-info">
                        <i class="fas fa-chart-line"></i> Details
                    </a>
                </div>
            </div>
        `;
    }

    html += '</div>';

    // Add link to full modules page
    html += `
        <div style="margin-top: 20px; text-align: center;">
            <a href="/modules" class="btn btn-primary">
                <i class="fas fa-cubes"></i> Open Full Module Management
            </a>
        </div>
    `;

    container.innerHTML = html;
}

async function enableModule(moduleName) {
    if (!confirm(`Enable module: ${moduleName}?`)) return;

    try {
        const response = await fetch(`/api/modules/${moduleName}/enable`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Success', `Module ${moduleName} enabled`, 'success');
            loadModules(); // Reload
        } else {
            showNotification('Error', data.error, 'danger');
        }
    } catch (error) {
        showNotification('Error', error.message, 'danger');
    }
}

async function disableModule(moduleName) {
    if (!confirm(`Disable module: ${moduleName}? All positions will be closed.`)) return;

    try {
        const response = await fetch(`/api/modules/${moduleName}/disable`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Success', `Module ${moduleName} disabled`, 'success');
            loadModules(); // Reload
        } else {
            showNotification('Error', data.error, 'danger');
        }
    } catch (error) {
        showNotification('Error', error.message, 'danger');
    }
}

async function pauseModule(moduleName) {
    try {
        const response = await fetch(`/api/modules/${moduleName}/pause`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Success', `Module ${moduleName} paused`, 'info');
            loadModules(); // Reload
        } else {
            showNotification('Error', data.error, 'danger');
        }
    } catch (error) {
        showNotification('Error', error.message, 'danger');
    }
}

function showNotification(title, message, type) {
    // Use existing notification system or simple alert
    alert(`${title}: ${message}`);
}

// Add event listener for tab clicks
document.addEventListener('DOMContentLoaded', function() {
    // Listen for modules tab click
    const modulesTab = document.querySelector('[data-category="modules"]');
    if (modulesTab) {
        modulesTab.addEventListener('click', loadModules);
    }
});
