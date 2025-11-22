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
    try {
        const response = await fetch('/api/modules');
        const result = await response.json();

        if (result.success && result.data) {
            const tabsContainer = document.getElementById('module-tabs');
            tabsContainer.innerHTML = '';

            result.data.forEach(module => {
                const tab = document.createElement('button');
                tab.className = `tab-btn ${module.name === currentModule ? 'active' : ''}`;
                tab.textContent = module.display_name || module.name;
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

    // Update active tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Load module analytics
    await loadModuleAnalytics(moduleName, currentTimeframe);
}

async function loadModuleAnalytics(moduleName, timeframe) {
    try {
        // Load performance, risk, equity, daily PnL, and trades in parallel
        const [perfResponse, riskResponse, equityResponse, pnlResponse, tradesResponse] = await Promise.all([
            fetch(`/api/analytics/performance/${moduleName}?timeframe=${timeframe}`),
            fetch(`/api/analytics/risk/${moduleName}`),
            fetch(`/api/analytics/equity/${moduleName}?timeframe=${timeframe}`),
            fetch(`/api/analytics/daily-pnl/${moduleName}?timeframe=${timeframe}`),
            fetch(`/api/analytics/trades/${moduleName}?limit=10`)
        ]);

        const perfResult = await perfResponse.json();
        const riskResult = await riskResponse.json();
        const equityResult = await equityResponse.json();
        const pnlResult = await pnlResponse.json();
        const tradesResult = await tradesResponse.json();

        if (perfResult.success) {
            updatePerformanceMetrics(perfResult.data);
        }

        if (riskResult.success) {
            updateRiskMetrics(riskResult.data);
        }

        if (equityResult.success) {
            updateEquityChart(equityResult.data);
        }

        if (pnlResult.success) {
            updateDailyPnlChart(pnlResult.data);
        }

        if (tradesResult.success) {
            updateTradesTable(tradesResult.data.trades);
        }

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
