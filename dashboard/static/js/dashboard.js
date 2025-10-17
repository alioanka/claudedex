// Dashboard-specific functionality

// Dashboard state management
const dashboardState = {
    refreshInterval: null,
    charts: {},
    lastUpdate: null
};

// Initialize dashboard-specific features
function initDashboard() {
    loadHistoricalStats(); // Load historical data ONCE
    setupDashboardRefresh();
    setupQuickActions();
    setupActivityFeed();
    initializeDashboardCharts();
}

// Setup auto-refresh
function setupDashboardRefresh() {
    // Refresh every 2 seconds
    dashboardState.refreshInterval = setInterval(() => {
        refreshDashboardData();
    }, 2000);
}

// Refresh dashboard data
async function refreshDashboardData() {
    try {
        const response = await apiGet('/api/dashboard/summary');
        if (response.success) {
            updateDashboardMetrics(response.data);
            dashboardState.lastUpdate = new Date();
        }
    } catch (error) {
        console.error('Failed to refresh dashboard:', error);
    }
}

// Update dashboard metrics
function updateDashboardMetrics(data) {
    // ✅ ONLY update LIVE metrics, NOT historical stats
    
    // Update portfolio value (live data)
    const portfolioValueStat = document.getElementById('portfolioValueStat');
    if (portfolioValueStat) {
        portfolioValueStat.textContent = formatCurrency(data.portfolio_value);
    }
    
    // Update open positions (live data)
    const openPositionsStat = document.getElementById('openPositionsStat');
    if (openPositionsStat) {
        openPositionsStat.textContent = data.open_positions;
    }
    
    // ❌ REMOVE these lines - they overwrite historical data:
    // const totalPnlStat = document.getElementById('totalPnlStat');
    // const winRateStat = document.getElementById('winRateStat');
    
    // NOTE: totalPnlStat and winRateStat are now ONLY updated by loadHistoricalStats()
}

// ✅ NEW: Add function to load historical stats ONCE on page load
async function loadHistoricalStats() {
    try {
        const response = await apiGet('/api/performance/metrics');
        
        if (response.success && response.data && response.data.historical) {
            const hist = response.data.historical;
            
            // Update historical stats (these should NOT be overwritten by WebSocket)
            const totalPnlStat = document.getElementById('totalPnlStat');
            if (totalPnlStat) {
                totalPnlStat.textContent = formatCurrency(hist.total_pnl || 0);
                const card = totalPnlStat.closest('.stat-card');
                if (card) {
                    card.classList.remove('success', 'danger');
                    card.classList.add(hist.total_pnl >= 0 ? 'success' : 'danger');
                }
            }
            
            const winRateStat = document.getElementById('winRateStat');
            if (winRateStat) {
                winRateStat.textContent = `${(hist.win_rate || 0).toFixed(1)}%`;
            }
            
            // Update Trading Stats section
            const totalTrades = document.getElementById('totalTrades');
            if (totalTrades) totalTrades.textContent = hist.total_trades || 0;
            
            const winRate = document.getElementById('winRate');
            if (winRate) winRate.textContent = `${(hist.win_rate || 0).toFixed(2)}%`;
            
            const avgWin = document.getElementById('avgWin');
            if (avgWin) avgWin.textContent = formatCurrency(hist.avg_win || 0);
            
            const avgLoss = document.getElementById('avgLoss');
            if (avgLoss) avgLoss.textContent = formatCurrency(hist.avg_loss || 0);
        }
    } catch (error) {
        console.error('Error loading historical stats:', error);
    }
}

// Setup quick actions
function setupQuickActions() {
    const quickActions = document.querySelectorAll('.quick-action');
    quickActions.forEach(action => {
        action.addEventListener('click', function() {
            const actionType = this.dataset.action;
            handleQuickAction(actionType);
        });
    });
}

// Handle quick actions
async function handleQuickAction(actionType) {
    switch(actionType) {
        case 'new_trade':
            showNewTradeModal();
            break;
        case 'close_all':
            await handleEmergencyExit();
            break;
        case 'generate_report':
            window.location.href = '/reports';
            break;
        case 'view_performance':
            window.location.href = '/performance';
            break;
        default:
            console.log('Unknown action:', actionType);
    }
}

// Show new trade modal
function showNewTradeModal() {
    // Implementation for manual trade modal
    showToast('info', 'Manual trading interface coming soon');
}

// Setup activity feed
function setupActivityFeed() {
    loadActivityFeed();
    
    // Refresh activity feed every 10 seconds
    setInterval(loadActivityFeed, 10000);
}

// Load activity feed
async function loadActivityFeed() {
    try {
        const response = await apiGet('/api/alerts/recent?limit=10');
        if (response.success) {
            displayActivityFeed(response.data);
        }
    } catch (error) {
        console.error('Failed to load activity feed:', error);
    }
}

// Display activity feed
function displayActivityFeed(alerts) {
    const feedContainer = document.getElementById('activityFeed');
    if (!feedContainer) return;
    
    feedContainer.innerHTML = alerts.map(alert => {
        const iconClass = getAlertIconClass(alert.type);
        const iconType = getAlertIconType(alert.type);
        
        return `
            <div class="activity-item">
                <div class="activity-icon ${iconType}">
                    <i class="${iconClass}"></i>
                </div>
                <div class="activity-content">
                    <div class="activity-title">${alert.title}</div>
                    <div class="activity-description">${alert.message}</div>
                    <div class="activity-time">${timeAgo(alert.timestamp)}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Get alert icon class
function getAlertIconClass(type) {
    const iconMap = {
        'position_opened': 'fas fa-arrow-up',
        'position_closed': 'fas fa-arrow-down',
        'stop_loss_hit': 'fas fa-shield-alt',
        'take_profit_hit': 'fas fa-bullseye',
        'high_confidence_signal': 'fas fa-chart-line',
        'whale_movement': 'fas fa-fish',
        'volume_surge': 'fas fa-fire',
        'high_risk_warning': 'fas fa-exclamation-triangle'
    };
    return iconMap[type] || 'fas fa-bell';
}

// Get alert icon type
function getAlertIconType(type) {
    if (type.includes('opened') || type.includes('profit')) return 'buy';
    if (type.includes('closed') || type.includes('loss')) return 'sell';
    return 'alert';
}

// Initialize dashboard charts
function initializeDashboardCharts() {
    createPortfolioMiniChart();
    createPnlMiniChart();
}

// Create mini portfolio chart
function createPortfolioMiniChart() {
    const canvas = document.getElementById('portfolioMiniChart');
    if (!canvas) return;
    
    // Sample data - replace with actual data
    const data = Array.from({length: 24}, (_, i) => ({
        x: i,
        y: 10000 + Math.random() * 1000
    }));
    
    createChart('portfolioMiniChart', {
        type: 'line',
        data: {
            labels: data.map(d => d.x),
            datasets: [{
                data: data.map(d => d.y),
                borderColor: '#3b82f6',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

// Create mini P&L chart
function createPnlMiniChart() {
    const canvas = document.getElementById('pnlMiniChart');
    if (!canvas) return;
    
    // Sample data - replace with actual data
    const data = Array.from({length: 24}, (_, i) => ({
        x: i,
        y: Math.random() * 200 - 100
    }));
    
    createChart('pnlMiniChart', {
        type: 'bar',
        data: {
            labels: data.map(d => d.x),
            datasets: [{
                data: data.map(d => d.y),
                backgroundColor: data.map(d => 
                    d.y >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                ),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

// Price ticker functionality
function updatePriceTicker() {
    // Implementation for live price updates
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboardState.refreshInterval) {
        clearInterval(dashboardState.refreshInterval);
    }
});

// Export functions
window.initDashboard = initDashboard;
window.refreshDashboardData = refreshDashboardData;