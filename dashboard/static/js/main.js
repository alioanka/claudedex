// DexScreener Trading Bot - Main JavaScript - FIXED VERSION
//dashboard/static/js/main.js
// Global state
const state = {
    ws: null,
    socket: null,
    botStatus: 'offline',
    dashboardData: {},
    positions: [],
    orders: [],
    alerts: [],
    charts: {},
    historicalDataLoaded: false  // ✅ Track if historical data is loaded
};

// Initialize dashboard
function initDashboard() {
    // Setup menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            // ✅ Store state in localStorage
            localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
        });
        
        // ✅ Restore sidebar state
        const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
        if (isCollapsed) {
            sidebar.classList.add('collapsed');
        }
    }
    
    // Setup theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Setup notification bell
    const notificationBell = document.getElementById('notificationBell');
    const notificationPanel = document.getElementById('notificationPanel');
    const closeNotifications = document.getElementById('closeNotifications');
    
    if (notificationBell && notificationPanel) {
        notificationBell.addEventListener('click', () => {
            notificationPanel.classList.toggle('active');
        });
    }
    
    if (closeNotifications && notificationPanel) {
        closeNotifications.addEventListener('click', () => {
            notificationPanel.classList.remove('active');
        });
    }
    
    // Setup bot controls
    setupBotControls();
    
    // Start periodic updates
    startPeriodicUpdates();
}

// Theme management
function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    
    if (body.classList.contains('theme-dark')) {
        body.classList.remove('theme-dark');
        body.classList.add('theme-light');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.remove('theme-light');
        body.classList.add('theme-dark');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        localStorage.setItem('theme', 'dark');
    }
}

// Load saved theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    
    if (savedTheme === 'light') {
        body.classList.add('theme-light');
        if (themeToggle) themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        body.classList.add('theme-dark');
        if (themeToggle) themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    }
}

// Bot controls
function setupBotControls() {
    const startBot = document.getElementById('startBot');
    const stopBot = document.getElementById('stopBot');
    const restartBot = document.getElementById('restartBot');
    const emergencyExit = document.getElementById('emergencyExit');
    
    if (startBot) {
        startBot.addEventListener('click', () => handleBotControl('start'));
    }
    
    if (stopBot) {
        stopBot.addEventListener('click', () => handleBotControl('stop'));
    }
    
    if (restartBot) {
        restartBot.addEventListener('click', () => handleBotControl('restart'));
    }
    
    if (emergencyExit) {
        emergencyExit.addEventListener('click', () => handleEmergencyExit());
    }
}

async function handleBotControl(action) {
    const actionMap = {
        'start': { url: '/api/bot/start', message: 'Starting bot...' },
        'stop': { url: '/api/bot/stop', message: 'Stopping bot...' },
        'restart': { url: '/api/bot/restart', message: 'Restarting bot...' }
    };
    
    const config = actionMap[action];
    if (!config) return;
    
    try {
        showToast('info', config.message);
        
        const response = await fetch(config.url, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('success', data.message || `Bot ${action} successful`);
            updateBotStatus();
        } else {
            showToast('error', data.error || `Failed to ${action} bot`);
        }
    } catch (error) {
        console.error(`Error ${action} bot:`, error);
        showToast('error', `Failed to ${action} bot`);
    }
}

async function handleEmergencyExit() {
    const confirmed = await showConfirmation(
        'Emergency Exit',
        'This will close ALL open positions immediately. Are you sure?'
    );
    
    if (!confirmed) return;
    
    try {
        showToast('warning', 'Executing emergency exit...');
        
        const response = await fetch('/api/bot/emergency_exit', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            showToast('success', data.message);
            showAlert('success', 'Emergency Exit Complete', 
                `Closed: ${data.closed.length}, Failed: ${data.failed.length}`);
        } else {
            showToast('error', data.error || 'Emergency exit failed');
        }
    } catch (error) {
        console.error('Error in emergency exit:', error);
        showToast('error', 'Emergency exit failed');
    }
}

// ✅ FIX: Load dashboard data - DON'T update portfolio/P&L from this
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard/summary');
        const data = await response.json();
        
        if (data.success) {
            state.dashboardData = data.data;
            // ✅ ONLY update notification badge and open positions count
            updateDashboardUIMinimal(data.data);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// ✅ NEW: Minimal update - ONLY notification and positions count
function updateDashboardUIMinimal(data) {
    // Update notification badge
    const notificationBadge = document.getElementById('notificationBadge');
    if (notificationBadge) {
        const count = data.active_alerts || 0;
        notificationBadge.textContent = count;
        notificationBadge.style.display = count > 0 ? 'block' : 'none';
    }
    
    // Update open positions count (only in dashboard page)
    const openPositionsStat = document.getElementById('openPositionsStat');
    if (openPositionsStat && data.open_positions !== undefined) {
        openPositionsStat.textContent = data.open_positions;
    }
}

async function loadTopBarData() {
    try {
        const response = await apiGet('/api/performance/metrics');
        
        if (response.success && response.data && response.data.historical) {
            const hist = response.data.historical;
            
            // Get realized P&L from closed trades
            const startingBalance = 400;
            const realizedPnl = hist.total_pnl || 0;
            
            // Get unrealized P&L from open positions
            let unrealizedPnl = 0;
            try {
                const posResponse = await apiGet('/api/positions/open');
                if (posResponse.success && posResponse.data) {
                    unrealizedPnl = posResponse.data.reduce((sum, pos) => 
                        sum + parseFloat(pos.unrealized_pnl || 0), 0
                    );
                }
            } catch (e) {
                console.error('Error getting open positions P&L:', e);
            }
            
            // Calculate total portfolio value
            const portfolioValue = startingBalance + realizedPnl + unrealizedPnl;
            
            // Update top bar portfolio value
            const portfolioValueEl = document.querySelector('#portfolioValue .value');
            if (portfolioValueEl) {
                portfolioValueEl.textContent = formatCurrency(portfolioValue);
            }
            
            // Update dashboard stat card if exists
            const portfolioValueStat = document.getElementById('portfolioValueStat');
            if (portfolioValueStat) {
                portfolioValueStat.textContent = formatCurrency(portfolioValue);
            }
            
            // Calculate 24h P&L
            const recentResponse = await apiGet('/api/trades/recent?limit=1000');
            if (recentResponse.success && recentResponse.data) {
                const oneDayAgo = new Date();
                oneDayAgo.setDate(oneDayAgo.getDate() - 1);
                
                const dailyTrades = recentResponse.data.filter(t => {
                    if (!t.exit_timestamp || t.status !== 'closed') return false;
                    return new Date(t.exit_timestamp) > oneDayAgo;
                });
                
                const dailyPnl = dailyTrades.reduce((sum, t) => 
                    sum + parseFloat(t.profit_loss || 0), 0
                );
                
                // Update 24h P&L in top bar
                const pnlValueEl = document.querySelector('#pnlIndicator .value');
                if (pnlValueEl) {
                    pnlValueEl.textContent = formatCurrency(dailyPnl);
                    
                    const pnlIndicator = document.getElementById('pnlIndicator');
                    if (pnlIndicator) {
                        pnlIndicator.classList.remove('positive', 'negative');
                        pnlIndicator.classList.add(dailyPnl >= 0 ? 'positive' : 'negative');
                    }
                }
            }
            
            state.historicalDataLoaded = true;
        }
    } catch (error) {
        console.error('Error loading top bar data:', error);
    }
}
// Update dashboard UI (legacy - kept for compatibility)
function updateDashboardUI(data) {
    updateDashboardUIMinimal(data);
}

// Update bot status
async function updateBotStatus() {
    try {
        const response = await fetch('/api/bot/status');
        const data = await response.json();
        
        if (data.success && data.data) {
            const status = data.data;
            // ✅ FIX: Check if bot is running (handle both boolean and string)
            const isRunning = status.running === true || status.running === 'running' || status.running === 'active';
            state.botStatus = isRunning ? 'online' : 'offline';
            
            const statusIndicator = document.getElementById('botStatus');
            if (statusIndicator) {
                statusIndicator.classList.remove('online', 'offline');
                statusIndicator.classList.add(state.botStatus);
                
                const statusText = statusIndicator.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = isRunning ? 'Bot Running' : 'Bot Stopped';
                }
            }
        } else {
            // If API call fails but gives a response, assume offline
            state.botStatus = 'offline';
            const statusIndicator = document.getElementById('botStatus');
            if (statusIndicator) {
                statusIndicator.classList.remove('online');
                statusIndicator.classList.add('offline');
                const statusText = statusIndicator.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = 'Connecting...';
                }
            }
        }
    } catch (error) {
        console.error('Error updating bot status:', error);
        // If fetch fails, bot is unreachable
        state.botStatus = 'offline';
        const statusIndicator = document.getElementById('botStatus');
        if (statusIndicator) {
            statusIndicator.classList.remove('online');
            statusIndicator.classList.add('offline');
            const statusText = statusIndicator.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Error';
            }
        }
    }
}

// ✅ FIX: Periodic updates - Changed intervals
function startPeriodicUpdates() {
    // ✅ Load top bar data ONCE on startup
    loadTopBarData();
    
    // ✅ Update ONLY notification badge and positions count every 10 seconds (was 2)
    setInterval(loadDashboardData, 10000);
    
    // Update bot status every 5 seconds
    setInterval(updateBotStatus, 5000);
    
    // ✅ Refresh top bar portfolio/P&L every 5 minutes (not every 2 seconds!)
    setInterval(loadTopBarData, 300000);  // 5 minutes
}

// Toast notifications
function showToast(type, message, duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const iconMap = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${iconMap[type]}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, duration);
}

// Confirmation dialog
function showConfirmation(title, message) {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        const overlay = document.getElementById('modalOverlay');
        const titleEl = document.getElementById('confirmTitle');
        const messageEl = document.getElementById('confirmMessage');
        const okBtn = document.getElementById('confirmOk');
        const cancelBtn = document.getElementById('confirmCancel');
        
        if (!modal || !overlay) {
            resolve(false);
            return;
        }
        
        titleEl.textContent = title;
        messageEl.textContent = message;
        
        modal.classList.add('active');
        overlay.classList.add('active');
        
        const cleanup = () => {
            modal.classList.remove('active');
            overlay.classList.remove('active');
            okBtn.removeEventListener('click', handleOk);
            cancelBtn.removeEventListener('click', handleCancel);
            overlay.removeEventListener('click', handleCancel);
        };
        
        const handleOk = () => {
            cleanup();
            resolve(true);
        };
        
        const handleCancel = () => {
            cleanup();
            resolve(false);
        };
        
        okBtn.addEventListener('click', handleOk);
        cancelBtn.addEventListener('click', handleCancel);
        overlay.addEventListener('click', handleCancel);
    });
}

// Alert modal
function showAlert(type, title, message) {
    showToast(type, `${title}: ${message}`, 5000);
}

// Format helpers
function formatCurrency(value, decimals = 2) {
    // ✅ FIX: Handle invalid values
    if (value === null || value === undefined || isNaN(value)) {
        return '$0.00';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0.00%';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value / 100);  // ✅ Divide by 100 for percentages
}

function formatNumber(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0.00';
    }
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatDate(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Invalid Date';
    
    return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'Invalid Time';
    
    return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }).format(date);
}

function timeAgo(timestamp) {
    if (!timestamp) return 'unknown';
    
    const date = new Date(timestamp);
    if (isNaN(date.getTime())) return 'unknown';
    
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 0) return 'just now';  // Future date
    
    const intervals = {
        year: 31536000,
        month: 2592000,
        week: 604800,
        day: 86400,
        hour: 3600,
        minute: 60,
        second: 1
    };
    
    for (const [name, secondsInInterval] of Object.entries(intervals)) {
        const interval = Math.floor(seconds / secondsInInterval);
        if (interval >= 1) {
            return `${interval} ${name}${interval !== 1 ? 's' : ''} ago`;
        }
    }
    
    return 'just now';
}

// API helpers
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

async function apiGet(url) {
    return apiRequest(url, { method: 'GET' });
}

async function apiPost(url, body) {
    return apiRequest(url, {
        method: 'POST',
        body: JSON.stringify(body)
    });
}

// Chart helpers
function createChart(canvasId, config) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart if present
    if (state.charts[canvasId]) {
        state.charts[canvasId].destroy();
    }
    
    state.charts[canvasId] = new Chart(ctx, config);
    return state.charts[canvasId];
}

function updateChart(canvasId, newData) {
    const chart = state.charts[canvasId];
    if (!chart) return;
    
    chart.data = newData;
    chart.update();
}

// Export functions
async function exportData(format, endpoint) {
    try {
        showToast('info', `Generating ${format.toUpperCase()} export...`);
        
        const response = await fetch(`${endpoint}?format=${format}`);
        const blob = await response.blob();
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${new Date().toISOString().split('T')[0]}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        showToast('success', 'Export completed');
    } catch (error) {
        console.error('Export error:', error);
        showToast('error', 'Export failed');
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    loadTheme();
    initDashboard();
    initWebSocket(); // Initialize WebSocket connection

    // The following functions are called within initDashboard, so they are redundant here.
    // updateBotStatus();
    // loadTopBarData();
    // loadDashboardData();
});

// Export global functions
window.showToast = showToast;
window.showConfirmation = showConfirmation;
window.showAlert = showAlert;
window.formatCurrency = formatCurrency;
window.formatPercent = formatPercent;
window.formatNumber = formatNumber;
window.formatDate = formatDate;
window.formatTime = formatTime;
window.timeAgo = timeAgo;
window.apiGet = apiGet;
window.apiPost = apiPost;
window.createChart = createChart;
window.updateChart = updateChart;
window.exportData = exportData;
window.loadTopBarData = loadTopBarData;  // ✅ Export for manual refresh