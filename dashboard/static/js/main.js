// DexScreener Trading Bot - Main JavaScript

// Global state
const state = {
    ws: null,
    socket: null,
    botStatus: 'offline',
    dashboardData: {},
    positions: [],
    orders: [],
    alerts: [],
    charts: {}
};

// Initialize dashboard
function initDashboard() {
    // Setup menu toggle
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });
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

// Load dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard/summary');
        const data = await response.json();
        
        if (data.success) {
            state.dashboardData = data.data;
            updateDashboardUI(data.data);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Update dashboard UI
function updateDashboardUI(data) {
    // Update portfolio value
    const portfolioValue = document.getElementById('portfolioValue');
    if (portfolioValue) {
        const valueElement = portfolioValue.querySelector('.value');
        if (valueElement) {
            valueElement.textContent = formatCurrency(data.portfolio_value);
        }
    }
    
    // Update P&L
    const pnlIndicator = document.getElementById('pnlIndicator');
    if (pnlIndicator) {
        const valueElement = pnlIndicator.querySelector('.value');
        if (valueElement) {
            valueElement.textContent = formatCurrency(data.daily_pnl);
            pnlIndicator.classList.remove('positive', 'negative');
            pnlIndicator.classList.add(data.daily_pnl >= 0 ? 'positive' : 'negative');
        }
    }
    
    // Update notification badge
    const notificationBadge = document.getElementById('notificationBadge');
    if (notificationBadge) {
        notificationBadge.textContent = data.active_alerts || 0;
        notificationBadge.style.display = data.active_alerts > 0 ? 'block' : 'none';
    }
}

// Update bot status
async function updateBotStatus() {
    try {
        const response = await fetch('/api/bot/status');
        const data = await response.json();
        
        if (data.success) {
            const status = data.data;
            state.botStatus = status.running ? 'online' : 'offline';
            
            const statusIndicator = document.getElementById('botStatus');
            if (statusIndicator) {
                statusIndicator.classList.remove('online', 'offline');
                statusIndicator.classList.add(state.botStatus);
                
                const statusText = statusIndicator.querySelector('.status-text');
                if (statusText) {
                    statusText.textContent = status.running ? 'Bot Running' : 'Bot Stopped';
                }
            }
        }
    } catch (error) {
        console.error('Error updating bot status:', error);
    }
}

// Periodic updates
function startPeriodicUpdates() {
    // Update dashboard every 2 seconds
    setInterval(loadDashboardData, 2000);
    
    // Update bot status every 5 seconds
    setInterval(updateBotStatus, 5000);
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
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatPercent(value, decimals = 2) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatNumber(value, decimals = 2) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }).format(date);
}

function timeAgo(timestamp) {
    const seconds = Math.floor((new Date() - new Date(timestamp)) / 1000);
    
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
    loadDashboardData();
    updateBotStatus();
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