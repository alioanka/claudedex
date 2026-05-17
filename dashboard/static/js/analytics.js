/**
 * Analytics Dashboard JavaScript
 *
 * Provides real-time analytics, charts, and performance tracking
 */

let currentModule = 'dex_trading';
let currentTimeframe = '24h';
let equityChart = null;
let dailyPnlChart = null;
let refreshInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startAutoRefresh();
});

async function initializeDashboard() {
    try {
        await loadPortfolioSummary();
        await loadModuleTabs();
        await loadModuleAnalytics(currentModule, currentTimeframe);
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showNotification('Error loading analytics', 'error');
    }
}

function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadModuleAnalytics(currentModule, currentTimeframe);
        });
    }

    // Timeframe selector
    const timeframeSelector = document.getElementById('timeframe-selector');
    if (timeframeSelector) {
        timeframeSelector.addEventListener('change', (e) => {
            currentTimeframe = e.target.value;
            loadModuleAnalytics(currentModule, currentTimeframe);
        });
    }
}

function startAutoRefresh() {
    // Refresh every 30 seconds
    refreshInterval = setInterval(() => {
        loadPortfolioSummary();
        loadModuleAnalytics(currentModule, currentTimeframe);
    }, 30000);
}

async function loadPortfolioSummary() {
    try {
        const response = await fetch('/api/analytics/portfolio');
        const result = await response.json();

        if (result.success) {
            const data = result.data;

            // Update summary stats
            updateElement('total-pnl', formatCurrency(data.total_pnl), data.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative');
            updateElement('total-trades', data.total_trades);
            updateElement('active-modules', data.active_modules);
            updateElement('best-performer', data.best_performer || '-');
        }
    } catch (error) {
        console.error('Error loading portfolio summary:', error);
    }
}

async function loadModuleTabs() {
    // FAILURE B fix: replace the 3 hardcoded tabs (DEX/Futures/Solana) with
    // the full set returned by /api/modules so SNIPER/ARBITRAGE/COPY/AI are
    // reachable. Stamp every dynamic tab with both class + data-module so
    // switchModule() can find it without textContent-prefix matching.
    try {
        const response = await fetch('/api/modules');
        const result = await response.json();

        if (result.success && result.data) {
            const tabsContainer = document.getElementById('module-tabs');
            tabsContainer.innerHTML = '';

            // Handle both array format and object format (modules as object)
            let modules = [];
            if (Array.isArray(result.data)) {
                modules = result.data;
            } else if (result.data.modules && typeof result.data.modules === 'object') {
                // Convert modules object to array
                modules = Object.entries(result.data.modules).map(([name, data]) => ({
                    name: name,
                    display_name: data.display_name || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    ...data
                }));
            }

            modules.forEach(module => {
                const tab = document.createElement('button');
                // Use BOTH legacy `tab-button` and `tab-btn` so existing CSS
                // and the inline-template event handler in analytics.html
                // both find it.
                tab.className = `tab-button tab-btn ${module.name === currentModule ? 'active' : ''}`;
                tab.setAttribute('data-module', module.name);
                const statusBadge = module.status === 'DISABLED'
                    ? ' <span style="opacity:0.5;font-size:0.7em;">(disabled)</span>'
                    : (module.status === 'ENABLED + RUNNING'
                        ? ' <span style="opacity:0.9;font-size:0.7em;color:#10b981;">●</span>'
                        : '');
                tab.innerHTML = (module.display_name || module.name) + statusBadge;
                tab.onclick = () => switchModule(module.name);
                tabsContainer.appendChild(tab);
            });
        }
    } catch (error) {
        console.error('Error loading module tabs:', error);
    }
}

async function switchModule(moduleName) {
    currentModule = moduleName;

    // FAILURE B fix: match strictly on data-module to avoid the prior
    // textContent prefix trick that wrongly marked multiple tabs active
    // (e.g. "copy_trading" matched both "Copy Trading" and "Copy Sniper").
    document.querySelectorAll('.tab-btn, .tab-button').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-module') === moduleName) {
            btn.classList.add('active');
        }
    });

    // Load module analytics
    await loadModuleAnalytics(moduleName, currentTimeframe);
}

async function loadModuleAnalytics(moduleName, timeframe) {
    // FAILURE B fix: surface 503/error responses to the operator instead
    // of silently rendering zeros. Previously every endpoint returning
    // 503 (because analytics_engine=None in the standalone dashboard
    // subprocess) was treated as "no data" — operators saw $0.00 + empty
    // charts with no hint of WHY. Now we show a notification when ANY of
    // the five endpoints fail.
    const endpoints = [
        ['performance', `/api/analytics/performance/${moduleName}?timeframe=${timeframe}`],
        ['risk',        `/api/analytics/risk/${moduleName}`],
        ['equity',      `/api/analytics/equity/${moduleName}?timeframe=${timeframe}`],
        ['daily-pnl',   `/api/analytics/daily-pnl/${moduleName}?timeframe=${timeframe}`],
        ['trades',      `/api/analytics/trades/${moduleName}?limit=10`],
    ];
    try {
        const responses = await Promise.all(endpoints.map(([_, url]) => fetch(url)));
        const results = await Promise.all(responses.map(async (r, i) => {
            const ok = r.ok;
            let payload = null;
            try { payload = await r.json(); } catch (_) {}
            return { kind: endpoints[i][0], status: r.status, ok, payload };
        }));
        const failed = results.filter(r => !r.ok || (r.payload && r.payload.success === false));
        if (failed.length === results.length) {
            const reason = failed[0].payload?.error || `HTTP ${failed[0].status}`;
            showNotification(`Analytics unavailable for ${moduleName}: ${reason}`, 'error');
        } else if (failed.length > 0) {
            const kinds = failed.map(f => f.kind).join(', ');
            showNotification(`Some analytics failed (${kinds}) for ${moduleName}`, 'warning');
        }
        const [perfResult, riskResult, equityResult, pnlResult, tradesResult] =
            results.map(r => r.payload || {});

        if (perfResult && perfResult.success) updatePerformanceMetrics(perfResult.data);
        if (riskResult && riskResult.success) updateRiskMetrics(riskResult.data);
        if (equityResult && equityResult.success) updateEquityChart(equityResult.data);
        if (pnlResult && pnlResult.success) updateDailyPnlChart(pnlResult.data);
        if (tradesResult && tradesResult.success) updateTradesTable(tradesResult.data.trades);

    } catch (error) {
        console.error('Error loading module analytics:', error);
        showNotification('Error loading analytics', 'error');
    }
}

function updatePerformanceMetrics(data) {
    updateElement('win-rate', `${data.win_rate}%`);
    updateElement('profit-factor', data.profit_factor.toFixed(2));
    updateElement('sharpe-ratio', data.sharpe_ratio.toFixed(2));
    updateElement('max-drawdown', `${data.max_drawdown}%`);
    updateElement('avg-win', formatCurrency(data.avg_win));
    updateElement('avg-loss', formatCurrency(data.avg_loss));
}

function updateRiskMetrics(data) {
    updateElement('total-exposure', formatCurrency(data.total_exposure));
    updateElement('net-exposure', formatCurrency(data.net_exposure), data.net_exposure >= 0 ? 'pnl-positive' : 'pnl-negative');
    updateElement('var-95', formatCurrency(data.var_95));
    // var_99 and cvar_95 were computed and discarded — see DASH-Q-12.
    updateElement('var-99', formatCurrency(data.var_99));
    updateElement('cvar-95', formatCurrency(data.cvar_95));
    updateElement('annual-vol', `${data.annual_volatility}%`);
    updateElement('avg-leverage', `${data.avg_leverage.toFixed(1)}x`);
    updateElement('largest-position', `${data.largest_position_pct}%`);
}

function updateEquityChart(data) {
    const ctx = document.getElementById('equity-chart');
    if (!ctx) return;

    const equityCurve = data.equity_curve || [];
    const labels = equityCurve.map((_, i) => `T${i+1}`);

    if (equityChart) {
        equityChart.destroy();
    }

    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity',
                data: equityCurve,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Equity: ${formatCurrency(context.parsed.y)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function updateDailyPnlChart(data) {
    const ctx = document.getElementById('daily-pnl-chart');
    if (!ctx) return;

    const dates = data.dates || [];
    const pnl = data.pnl || [];

    // Color bars based on positive/negative
    const colors = pnl.map(value => value >= 0 ? '#10b981' : '#ef4444');

    if (dailyPnlChart) {
        dailyPnlChart.destroy();
    }

    dailyPnlChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Daily P&L',
                data: pnl,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            return `P&L: ${formatCurrency(value)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('trades-table-body');
    if (!tbody) return;

    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="no-data">No trades found</td></tr>';
        return;
    }

    tbody.innerHTML = '';

    trades.forEach(trade => {
        const row = document.createElement('tr');

        const pnl = trade.pnl || 0;
        const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';

        // Calculate duration
        const duration = trade.duration_seconds ? formatDuration(trade.duration_seconds) : '-';

        row.innerHTML = `
            <td>${formatDateTime(trade.timestamp)}</td>
            <td>${formatTokenAddress(trade.token || '-')}</td>
            <td><span class="badge">${trade.side || 'BUY'}</span></td>
            <td>${formatCurrency(trade.entry_price || 0)}</td>
            <td>${formatCurrency(trade.exit_price || 0)}</td>
            <td>${formatCurrency(trade.size || 0)}</td>
            <td class="${pnlClass}">${formatCurrency(pnl)}</td>
            <td>${duration}</td>
        `;

        tbody.appendChild(row);
    });
}

// Utility functions
function updateElement(id, value, className = null) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        if (className) {
            // Remove existing pnl classes
            element.classList.remove('pnl-positive', 'pnl-negative');
            element.classList.add(className);
        }
    }
}

function formatCurrency(value) {
    if (value === null || value === undefined) return '$0.00';
    const num = parseFloat(value);
    if (isNaN(num)) return '$0.00';

    const sign = num >= 0 ? '' : '';
    return `${sign}$${Math.abs(num).toFixed(2)}`;
}

function formatDateTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatTokenAddress(address) {
    if (!address || address === '-') return '-';
    if (address.length <= 12) return address;
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
}

function formatDuration(seconds) {
    if (!seconds) return '-';

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m`;
    } else {
        return `${seconds}s`;
    }
}

function showNotification(message, type = 'info') {
    // Simple notification (could be enhanced with a toast library)
    console.log(`[${type.toUpperCase()}] ${message}`);

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    if (equityChart) {
        equityChart.destroy();
    }
    if (dailyPnlChart) {
        dailyPnlChart.destroy();
    }
});
