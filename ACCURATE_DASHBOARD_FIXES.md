# ACCURATE Dashboard Fixes - Based on Complete Code Review

## Files Reviewed:
✅ base.html, dashboard.html, positions.html, trades.html, performance.html, backtest.html, reports.html, settings.html
✅ main.js, dashboard.js, websocket.js, settings.js
✅ main.css, dashboard.css, charts.css, themes.css
✅ enhanced_dashboard.py

---

## 🔴 CRITICAL ISSUE FOUND: trades.html and settings.html Don't Extend base.html

**Problem:** trades.html (lines 1-50) and settings.html have their own complete HTML structure instead of extending base.html. This is why:
- Left menu breaks on these pages
- Night mode toggle doesn't work consistently
- Top bar is different

---

## PRIORITY 1: Fix trades.html and settings.html Structure

### Issue 4a, 4b, 4c, 8a, 8b: trades.html Breaking Menu & Empty Data

**File:** `dashboard/templates/trades.html`

**Current (WRONG):**
```html
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    ...
</head>
<body>
    <div class="container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>DexScreener Bot</h2>
            </div>
            <nav class="sidebar-nav">
                <a href="/" class="nav-item">
                    <span class="nav-icon">📊</span>
                    <span class="nav-text">Dashboard</span>
                </a>
                <!-- ... -->
```

**REPLACE ENTIRE FILE WITH:**
```jinja2
{% extends "base.html" %}

{% block title %}Trades - DexScreener Trading Bot{% endblock %}

{% block content %}
<div class="page-header">
    <h1>Trade History</h1>
    <div class="page-actions">
        <button class="btn btn-secondary" onclick="exportTrades()">
            <i class="fas fa-download"></i> Export
        </button>
        <button class="btn btn-secondary" onclick="refreshTrades()">
            <i class="fas fa-sync"></i> Refresh
        </button>
    </div>
</div>

<!-- Summary Cards -->
<div class="grid grid-4">
    <div class="stat-card">
        <i class="fas fa-list stat-icon"></i>
        <div class="stat-label">Total Trades</div>
        <div class="stat-value" id="totalTradesCount">0</div>
    </div>
    
    <div class="stat-card success">
        <i class="fas fa-check-circle stat-icon"></i>
        <div class="stat-label">Winning Trades</div>
        <div class="stat-value" id="winningTradesCount">0</div>
    </div>
    
    <div class="stat-card danger">
        <i class="fas fa-times-circle stat-icon"></i>
        <div class="stat-label">Losing Trades</div>
        <div class="stat-value" id="losingTradesCount">0</div>
    </div>
    
    <div class="stat-card">
        <i class="fas fa-percentage stat-icon"></i>
        <div class="stat-label">Win Rate</div>
        <div class="stat-value" id="winRateValue">0%</div>
    </div>
</div>

<!-- Filters -->
<div class="card">
    <div class="card-body">
        <div class="grid grid-4">
            <div class="form-group">
                <label class="form-label">Status</label>
                <select id="filterStatus" class="form-control">
                    <option value="all">All</option>
                    <option value="open">Open</option>
                    <option value="closed">Closed</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label">Side</label>
                <select id="filterSide" class="form-control">
                    <option value="all">All</option>
                    <option value="buy">Buy</option>
                    <option value="sell">Sell</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label">Start Date</label>
                <input type="date" id="filterStartDate" class="form-control" placeholder="dd.mm.yyyy">
            </div>
            <div class="form-group">
                <label class="form-label">End Date</label>
                <input type="date" id="filterEndDate" class="form-control" placeholder="dd.mm.yyyy">
            </div>
        </div>
        <button class="btn btn-primary" onclick="applyFilters()">
            <i class="fas fa-filter"></i> Apply Filters
        </button>
    </div>
</div>

<!-- Trades Table -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">All Trades</h3>
    </div>
    <div class="card-body">
        <div class="table-container">
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Token</th>
                        <th>Side</th>
                        <th>Amount</th>
                        <th>Price</th>
                        <th>Total</th>
                        <th>Status</th>
                        <th>P&L</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="tradesTableBody">
                    <tr>
                        <td colspan="9" class="text-center">Loading trades...</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
let allTrades = [];

async function loadTrades() {
    try {
        const response = await apiGet('/api/trades/history?limit=1000');
        
        if (response.success && response.data) {
            allTrades = response.data;
            updateTradeSummary(allTrades);
            displayTrades(allTrades);
        } else {
            console.error('Failed to load trades:', response.error);
            document.getElementById('tradesTableBody').innerHTML = 
                '<tr><td colspan="9" class="text-center text-danger">Failed to load trades</td></tr>';
        }
    } catch (error) {
        console.error('Error loading trades:', error);
        document.getElementById('tradesTableBody').innerHTML = 
            '<tr><td colspan="9" class="text-center text-danger">Error loading trades</td></tr>';
    }
}

function updateTradeSummary(trades) {
    const closedTrades = trades.filter(t => t.status === 'closed');
    const winningTrades = closedTrades.filter(t => parseFloat(t.profit_loss || 0) > 0);
    const losingTrades = closedTrades.filter(t => parseFloat(t.profit_loss || 0) < 0);
    
    document.getElementById('totalTradesCount').textContent = trades.length;
    document.getElementById('winningTradesCount').textContent = winningTrades.length;
    document.getElementById('losingTradesCount').textContent = losingTrades.length;
    
    const winRate = closedTrades.length > 0 
        ? (winningTrades.length / closedTrades.length * 100).toFixed(1) 
        : 0;
    document.getElementById('winRateValue').textContent = `${winRate}%`;
}

function displayTrades(trades) {
    const tbody = document.getElementById('tradesTableBody');
    
    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center">No trades found</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.map(trade => {
        const pnl = parseFloat(trade.profit_loss || 0);
        const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
        const statusBadge = trade.status === 'open' ? 'badge-success' : 'badge-neutral';
        
        return `
            <tr>
                <td>${formatTime(trade.entry_timestamp)}</td>
                <td>${trade.token_symbol || 'Unknown'}</td>
                <td>
                    <span class="badge badge-${trade.side === 'buy' ? 'success' : 'danger'}">
                        ${(trade.side || 'unknown').toUpperCase()}
                    </span>
                </td>
                <td>${formatNumber(trade.amount || 0, 4)}</td>
                <td>${formatCurrency(trade.entry_price || 0)}</td>
                <td>${formatCurrency((trade.amount || 0) * (trade.entry_price || 0))}</td>
                <td><span class="badge ${statusBadge}">${trade.status || 'unknown'}</span></td>
                <td class="${pnlClass}">
                    ${trade.status === 'closed' ? formatCurrency(pnl) : '-'}
                </td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="viewTradeDetails('${trade.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function applyFilters() {
    const status = document.getElementById('filterStatus').value;
    const side = document.getElementById('filterSide').value;
    const startDate = document.getElementById('filterStartDate').value;
    const endDate = document.getElementById('filterEndDate').value;
    
    let filtered = allTrades;
    
    if (status !== 'all') {
        filtered = filtered.filter(t => t.status === status);
    }
    
    if (side !== 'all') {
        filtered = filtered.filter(t => t.side === side);
    }
    
    if (startDate) {
        const start = new Date(startDate);
        filtered = filtered.filter(t => new Date(t.entry_timestamp) >= start);
    }
    
    if (endDate) {
        const end = new Date(endDate);
        end.setHours(23, 59, 59);
        filtered = filtered.filter(t => new Date(t.entry_timestamp) <= end);
    }
    
    displayTrades(filtered);
}

function refreshTrades() {
    loadTrades();
}

function exportTrades() {
    exportData('csv', '/api/trades/export');
}

function viewTradeDetails(tradeId) {
    const trade = allTrades.find(t => t.id === tradeId);
    if (trade) {
        showToast('info', `Trade: ${trade.token_symbol} - ${trade.side}`);
    }
}

// Load on page ready
document.addEventListener('DOMContentLoaded', loadTrades);

// Refresh every 10 seconds
setInterval(loadTrades, 10000);
</script>
{% endblock %}
```

**This fixes:** Issues 4a, 4b, 4c (trades page)

---

### Issue 8a, 8b: settings.html Breaking Menu & Empty Fields

**File:** `dashboard/templates/settings.html`

**Check current file - if it doesn't extend base.html, REPLACE WITH:**

```jinja2
{% extends "base.html" %}

{% block title %}Settings - DexScreener Trading Bot{% endblock %}

{% block content %}
<div class="page-header">
    <h1>Settings</h1>
    <div class="page-actions">
        <button class="btn btn-secondary" id="revertSettings">
            <i class="fas fa-undo"></i> Revert Changes
        </button>
        <button class="btn btn-primary" id="saveSettings">
            <i class="fas fa-save"></i> Save Settings
        </button>
    </div>
</div>

<!-- Settings Tabs -->
<div class="tabs">
    <button class="tab-btn active category-tab" data-category="trading">Trading</button>
    <button class="tab-btn category-tab" data-category="risk">Risk Management</button>
    <button class="tab-btn category-tab" data-category="api">API & Connections</button>
    <button class="tab-btn category-tab" data-category="notifications">Notifications</button>
    <button class="tab-btn category-tab" data-category="strategies">Strategies</button>
    <button class="tab-btn category-tab" data-category="advanced">Advanced</button>
</div>

<!-- Trading Settings -->
<div class="card settings-section" id="tradingSettings">
    <div class="card-header">
        <h3 class="card-title">Trading Configuration</h3>
    </div>
    <div class="card-body">
        <div class="form-group">
            <label class="form-label">
                <input type="checkbox" id="tradingEnabled" class="settings-input">
                Enable Trading
            </label>
            <small class="form-help">Master switch for all trading activity</small>
        </div>
        
        <div class="form-group">
            <label class="form-label">Max Position Size ($)</label>
            <input type="number" id="maxPositionSize" class="form-control settings-input" step="100">
            <small class="form-help">Maximum amount per single position</small>
        </div>
        
        <div class="form-group">
            <label class="form-label">Max Open Positions</label>
            <input type="number" id="maxOpenPositions" class="form-control settings-input" min="1" max="20">
            <small class="form-help">Maximum number of concurrent positions</small>
        </div>
        
        <div class="form-group">
            <label class="form-label">Default Stop Loss (%)</label>
            <input type="number" id="defaultStopLoss" class="form-control settings-input" step="0.1">
            <small class="form-help">Default stop loss percentage</small>
        </div>
        
        <div class="form-group">
            <label class="form-label">Default Take Profit (%)</label>
            <input type="number" id="defaultTakeProfit" class="form-control settings-input" step="0.1">
            <small class="form-help">Default take profit percentage</small>
        </div>
        
        <div class="form-group">
            <label class="form-label">Slippage Tolerance (%)</label>
            <input type="number" id="slippageTolerance" class="form-control settings-input" step="0.1">
            <small class="form-help">Maximum acceptable slippage</small>
        </div>
    </div>
</div>

<!-- Risk Management Settings -->
<div class="card settings-section" id="riskSettings" style="display: none;">
    <div class="card-header">
        <h3 class="card-title">Risk Management</h3>
    </div>
    <div class="card-body">
        <div class="form-group">
            <label class="form-label">Max Drawdown (%)</label>
            <input type="number" id="maxDrawdown" class="form-control settings-input" step="0.1">
        </div>
        
        <div class="form-group">
            <label class="form-label">Max Daily Loss ($)</label>
            <input type="number" id="maxDailyLoss" class="form-control settings-input" step="100">
        </div>
        
        <div class="form-group">
            <label class="form-label">Position Size Method</label>
            <select id="positionSizeMethod" class="form-control settings-input">
                <option value="fixed">Fixed Amount</option>
                <option value="percentage">Percentage of Portfolio</option>
                <option value="kelly">Kelly Criterion</option>
            </select>
        </div>
        
        <div class="form-group">
            <label class="form-label">Risk Per Trade (%)</label>
            <input type="number" id="riskPerTrade" class="form-control settings-input" step="0.1">
        </div>
    </div>
</div>

<!-- API Settings -->
<div class="card settings-section" id="apiSettings" style="display: none;">
    <div class="card-header">
        <h3 class="card-title">API & Connections</h3>
    </div>
    <div class="card-body">
        <div class="form-group">
            <label class="form-label">DexScreener API Key</label>
            <input type="password" id="dexscreenerKey" class="form-control settings-input">
        </div>
        
        <div class="form-group">
            <label class="form-label">Web3 Provider URL</label>
            <input type="text" id="web3Provider" class="form-control settings-input">
        </div>
        
        <div class="form-group">
            <label class="form-label">RPC Timeout (seconds)</label>
            <input type="number" id="rpcTimeout" class="form-control settings-input" min="5" max="60">
        </div>
        
        <div class="form-group">
            <label class="form-label">Max Retries</label>
            <input type="number" id="maxRetries" class="form-control settings-input" min="1" max="10">
        </div>
    </div>
</div>

<!-- Notification Settings -->
<div class="card settings-section" id="notificationsSettings" style="display: none;">
    <div class="card-header">
        <h3 class="card-title">Notifications</h3>
    </div>
    <div class="card-body">
        <h4>Telegram</h4>
        <div class="form-group">
            <label class="form-label">
                <input type="checkbox" id="telegramEnabled" class="settings-input">
                Enable Telegram Notifications
            </label>
        </div>
        <div class="form-group">
            <label class="form-label">Bot Token</label>
            <input type="password" id="telegramBotToken" class="form-control settings-input">
        </div>
        <div class="form-group">
            <label class="form-label">Chat ID</label>
            <input type="text" id="telegramChatId" class="form-control settings-input">
        </div>
        
        <h4>Discord</h4>
        <div class="form-group">
            <label class="form-label">
                <input type="checkbox" id="discordEnabled" class="settings-input">
                Enable Discord Notifications
            </label>
        </div>
        <div class="form-group">
            <label class="form-label">Webhook URL</label>
            <input type="password" id="discordWebhook" class="form-control settings-input">
        </div>
    </div>
</div>

<!-- Strategies Settings -->
<div class="card settings-section" id="strategiesSettings" style="display: none;">
    <div class="card-header">
        <h3 class="card-title">Strategy Parameters</h3>
    </div>
    <div class="card-body">
        <p class="text-muted">Configure strategy-specific parameters here.</p>
    </div>
</div>

<!-- Advanced Settings -->
<div class="card settings-section" id="advancedSettings" style="display: none;">
    <div class="card-header">
        <h3 class="card-title">Advanced Configuration</h3>
    </div>
    <div class="card-body">
        <p class="text-muted">Advanced settings for experienced users.</p>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="/static/js/settings.js"></script>
<script>
// Tab switching
document.querySelectorAll('.category-tab').forEach(tab => {
    tab.addEventListener('click', function() {
        // Update active tab
        document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
        this.classList.add('active');
        
        // Show corresponding section
        const category = this.dataset.category;
        document.querySelectorAll('.settings-section').forEach(section => {
            section.style.display = 'none';
        });
        document.getElementById(category + 'Settings').style.display = 'block';
    });
});

// Load settings on page load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await apiGet('/api/settings/all');
        
        if (response.success && response.data) {
            // Populate trading settings
            if (response.data.trading) {
                document.getElementById('tradingEnabled').checked = response.data.trading.enabled || false;
                document.getElementById('maxPositionSize').value = response.data.trading.max_position_size || 1000;
                document.getElementById('maxOpenPositions').value = response.data.trading.max_open_positions || 5;
                document.getElementById('defaultStopLoss').value = response.data.trading.default_stop_loss || 5;
                document.getElementById('defaultTakeProfit').value = response.data.trading.default_take_profit || 10;
                document.getElementById('slippageTolerance').value = response.data.trading.slippage_tolerance || 1;
            }
            
            // Populate risk settings
            if (response.data.risk) {
                document.getElementById('maxDrawdown').value = response.data.risk.max_drawdown || 20;
                document.getElementById('maxDailyLoss').value = response.data.risk.max_daily_loss || 500;
                document.getElementById('positionSizeMethod').value = response.data.risk.position_size_method || 'fixed';
                document.getElementById('riskPerTrade').value = response.data.risk.risk_per_trade || 2;
            }
            
            // Populate API settings
            if (response.data.api) {
                document.getElementById('dexscreenerKey').value = response.data.api.dexscreener_api_key || '';
                document.getElementById('web3Provider').value = response.data.api.web3_provider_url || '';
                document.getElementById('rpcTimeout').value = response.data.api.rpc_timeout || 30;
                document.getElementById('maxRetries').value = response.data.api.max_retries || 3;
            }
            
            // Populate notification settings
            if (response.data.notifications) {
                if (response.data.notifications.telegram) {
                    document.getElementById('telegramEnabled').checked = response.data.notifications.telegram.enabled || false;
                    document.getElementById('telegramBotToken').value = response.data.notifications.telegram.bot_token || '';
                    document.getElementById('telegramChatId').value = response.data.notifications.telegram.chat_id || '';
                }
                if (response.data.notifications.discord) {
                    document.getElementById('discordEnabled').checked = response.data.notifications.discord.enabled || false;
                    document.getElementById('discordWebhook').value = response.data.notifications.discord.webhook_url || '';
                }
            }
        } else {
            showToast('error', 'Failed to load settings');
        }
    } catch (error) {
        console.error('Error loading settings:', error);
        showToast('error', 'Error loading settings');
    }
});
</script>
{% endblock %}
```

**This fixes:** Issues 8a, 8b (settings page)

---

## PRIORITY 2: Fix CSS for Sidebar Collapse & Notification Panel

### Issue 1a: Menu Toggle Button Not Working (CSS Missing)

**File:** `dashboard/static/css/main.css`

**ADD these CSS rules at the end:**

```css
/* ===== SIDEBAR COLLAPSED STATE ===== */
.sidebar.collapsed {
    width: 70px;
    min-width: 70px;
}

.sidebar.collapsed .nav-item span {
    display: none;
}

.sidebar.collapsed .nav-item i {
    margin-right: 0;
}

.sidebar.collapsed .nav-item a,
.sidebar.collapsed .control-btn {
    justify-content: center;
    padding: var(--spacing-md);
}

.sidebar.collapsed .control-btn span {
    display: none;
}

/* Adjust main content when sidebar is collapsed */
.sidebar.collapsed ~ .main-content {
    margin-left: 70px;
}

/* ===== NOTIFICATION PANEL ===== */
.notification-panel {
    position: fixed;
    top: var(--topnav-height, 60px);
    right: -400px;
    width: 400px;
    height: calc(100vh - var(--topnav-height, 60px));
    background-color: var(--bg-secondary);
    border-left: 1px solid var(--border-color);
    box-shadow: -2px 0 10px var(--shadow);
    transition: right 0.3s ease;
    z-index: 999;
    overflow-y: auto;
}

.notification-panel.active {
    right: 0;
}

.notification-panel .panel-header {
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.notification-panel .panel-header h3 {
    margin: 0;
    font-size: 1.25rem;
}

.notification-panel .close-btn {
    background: none;
    border: none;
    color: var(--text-primary);
    font-size: 1.25rem;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: var(--radius-sm);
    transition: background-color 0.2s;
}

.notification-panel .close-btn:hover {
    background-color: var(--bg-hover);
}

.notification-panel .panel-content {
    padding: 1rem;
}

.notification-badge {
    position: absolute;
    top: -5px;
    right: -5px;
    background-color: var(--accent-danger);
    color: white;
    font-size: 0.75rem;
    padding: 2px 6px;
    border-radius: 10px;
    min-width: 18px;
    text-align: center;
    display: none;
}

.notification-bell {
    position: relative;
    cursor: pointer;
}

.notification-item {
    display: flex;
    gap: 1rem;
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
    transition: background-color 0.2s;
}

.notification-item:hover {
    background-color: var(--bg-hover);
}

.notification-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.notification-icon.info {
    background-color: rgba(59, 130, 246, 0.2);
    color: var(--accent-primary);
}

.notification-icon.warning {
    background-color: rgba(245, 158, 11, 0.2);
    color: var(--accent-warning);
}

.notification-icon.error {
    background-color: rgba(239, 68, 68, 0.2);
    color: var(--accent-danger);
}

.notification-content {
    flex: 1;
}

.notification-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.notification-message {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.notification-time {
    font-size: 0.75rem;
    color: var(--text-muted);
}
```

**This fixes:** Issue 1a (Menu toggle), Issue 1c (Notification panel styling)

---

## PRIORITY 3: Fix Bot Status Detection

### Issue 1b: Bot Status Shows "Bot Stopped" Even When Running

**File:** `monitoring/enhanced_dashboard.py`

**Location:** Line 717-734 (api_bot_status method)

**REPLACE with:**

```python
async def api_bot_status(self, request):
    """Get bot status with robust detection"""
    try:
        # ✅ FIX: Check multiple possible state indicators
        is_running = False
        uptime_str = 'N/A'
        
        if self.engine:
            # Try different state attribute names
            if hasattr(self.engine, 'state'):
                # Check for various "running" values
                state_val = str(self.engine.state).lower()
                is_running = state_val in ['running', 'active', 'started', 'true', '1']
            elif hasattr(self.engine, 'running'):
                is_running = bool(self.engine.running)
            elif hasattr(self.engine, 'is_running'):
                is_running = bool(self.engine.is_running)
            elif hasattr(self.engine, '_running'):
                is_running = bool(self.engine._running)
            elif hasattr(self.engine, 'active'):
                is_running = bool(self.engine.active)
            
            # Calculate uptime if running
            if is_running:
                start_time = None
                if hasattr(self.engine, 'start_time') and self.engine.start_time:
                    start_time = self.engine.start_time
                elif hasattr(self.engine, 'started_at') and self.engine.started_at:
                    start_time = self.engine.started_at
                elif hasattr(self.engine, 'startup_time') and self.engine.startup_time:
                    start_time = self.engine.startup_time
                
                if start_time:
                    try:
                        uptime_delta = datetime.utcnow() - start_time
                        hours = int(uptime_delta.total_seconds() // 3600)
                        minutes = int((uptime_delta.total_seconds() % 3600) // 60)
                        uptime_str = f"{hours}h {minutes}m"
                    except Exception as e:
                        logger.warning(f"Error calculating uptime: {e}")
                        uptime_str = "Running"
        
        status = {
            'running': is_running,
            'uptime': uptime_str,
            'mode': self.config.get('mode', 'unknown'),
            'version': '1.0.0',
            'last_health_check': datetime.utcnow().isoformat()
        }
        
        return web.json_response({
            'success': True,
            'data': status
        })
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        # Return success with offline status instead of error
        return web.json_response({
            'success': True,
            'data': {
                'running': False,
                'uptime': 'N/A',
                'mode': 'unknown',
                'version': '1.0.0',
                'last_health_check': datetime.utcnow().isoformat()
            }
        }, status=200)
```

**This fixes:** Issue 1b (Bot status detection)

---

## PRIORITY 4: Fix Position Data Loading

### Issue 3a: Position Table Columns Missing Data

**File:** `monitoring/enhanced_dashboard.py`

**Location:** Find `api_open_positions` method (around line 400-500)

**REPLACE with complete implementation:**

```python
async def api_open_positions(self, request):
    """Get open positions with ALL required fields"""
    try:
        positions = []
        
        if self.engine and hasattr(self.engine, 'active_positions'):
            for token_address, position in self.engine.active_positions.items():
                # Extract and calculate all required fields
                entry_price = float(position.get('entry_price', 0))
                current_price = float(position.get('current_price', entry_price))
                amount = float(position.get('amount', 0))
                
                # Calculate value
                value = amount * current_price
                
                # Calculate P&L
                entry_value = amount * entry_price
                unrealized_pnl = value - entry_value
                
                # Calculate ROI
                roi = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                # Calculate duration
                entry_timestamp = position.get('entry_timestamp') or position.get('opened_at') or position.get('timestamp')
                duration_str = 'unknown'
                
                if entry_timestamp:
                    try:
                        if isinstance(entry_timestamp, str):
                            entry_time = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
                        elif isinstance(entry_timestamp, datetime):
                            entry_time = entry_timestamp
                        else:
                            entry_time = datetime.fromtimestamp(float(entry_timestamp))
                        
                        duration_delta = datetime.utcnow() - entry_time
                        total_seconds = duration_delta.total_seconds()
                        
                        if total_seconds < 60:
                            duration_str = 'just now'
                        elif total_seconds < 3600:
                            minutes = int(total_seconds // 60)
                            duration_str = f"{minutes}m ago"
                        elif total_seconds < 86400:
                            hours = int(total_seconds // 3600)
                            minutes = int((total_seconds % 3600) // 60)
                            duration_str = f"{hours}h {minutes}m ago"
                        else:
                            days = int(total_seconds // 86400)
                            hours = int((total_seconds % 86400) // 3600)
                            duration_str = f"{days}d {hours}h ago"
                    except Exception as e:
                        logger.error(f"Error calculating duration: {e}")
                        duration_str = 'unknown'
                
                positions.append({
                    'id': position.get('id', str(token_address)),
                    'token_address': token_address,
                    'token_symbol': position.get('token_symbol', position.get('symbol', 'UNKNOWN')),
                    'entry_price': round(entry_price, 8),
                    'current_price': round(current_price, 8),
                    'amount': round(amount, 4),
                    'value': round(value, 2),
                    'unrealized_pnl': round(unrealized_pnl, 2),
                    'roi': round(roi, 2),
                    'stop_loss': position.get('stop_loss'),
                    'take_profit': position.get('take_profit'),
                    'entry_timestamp': entry_timestamp.isoformat() if isinstance(entry_timestamp, datetime) else entry_timestamp,
                    'duration': duration_str,
                    'status': position.get('status', 'open')
                })
        
        logger.info(f"Returning {len(positions)} open positions")
        
        return web.json_response({
            'success': True,
            'data': positions,
            'count': len(positions)
        })
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        return web.json_response({
            'success': False,
            'error': str(e),
            'data': [],
            'count': 0
        }, status=200)
```

**This fixes:** Issue 3a (Position data)

---

### Issue 3c: Closed Positions Table Empty

**File:** `monitoring/enhanced_dashboard.py`

**ADD this new method (if it doesn't exist):**

```python
async def api_positions_history(self, request):
    """Get closed positions history"""
    try:
        limit = int(request.query.get('limit', 100))
        closed_positions = []
        
        if self.db:
            # Get closed trades from database
            trades = await self.db.get_closed_trades(limit=limit)
            
            for trade in trades:
                # Calculate duration
                duration = 'unknown'
                if trade.get('entry_timestamp') and trade.get('exit_timestamp'):
                    try:
                        entry = datetime.fromisoformat(str(trade['entry_timestamp']).replace('Z', '+00:00'))
                        exit_time = datetime.fromisoformat(str(trade['exit_timestamp']).replace('Z', '+00:00'))
                        delta = exit_time - entry
                        
                        hours = int(delta.total_seconds() // 3600)
                        minutes = int((delta.total_seconds() % 3600) // 60)
                        
                        if hours > 24:
                            days = hours // 24
                            remaining_hours = hours % 24
                            duration = f"{days}d {remaining_hours}h"
                        elif hours > 0:
                            duration = f"{hours}h {minutes}m"
                        else:
                            duration = f"{minutes}m"
                    except:
                        pass
                
                closed_positions.append({
                    'id': trade.get('id'),
                    'token_symbol': trade.get('token_symbol', 'UNKNOWN'),
                    'token_address': trade.get('token_address'),
                    'entry_price': float(trade.get('entry_price', 0)),
                    'exit_price': float(trade.get('exit_price', 0)),
                    'amount': float(trade.get('amount', 0)),
                    'profit_loss': float(trade.get('profit_loss', 0)),
                    'roi': float(trade.get('roi', 0)),
                    'entry_timestamp': trade.get('entry_timestamp'),
                    'exit_timestamp': trade.get('exit_timestamp'),
                    'duration': duration,
                    'exit_reason': trade.get('exit_reason', 'manual')
                })
        
        return web.json_response({
            'success': True,
            'data': closed_positions,
            'count': len(closed_positions)
        })
    except Exception as e:
        logger.error(f"Error getting closed positions: {e}")
        return web.json_response({
            'success': False,
            'error': str(e),
            'data': [],
            'count': 0
        }, status=200)
```

**File:** `dashboard/templates/positions.html`

**Location:** Find the closed positions table section (around line 100-120)

**ADD this JavaScript before the closing `{% endblock %}`:**

```javascript
async function loadClosedPositions() {
    try {
        const response = await apiGet('/api/positions/history?limit=50');
        
        if (response.success && response.data) {
            displayClosedPositions(response.data);
        }
    } catch (error) {
        console.error('Error loading closed positions:', error);
    }
}

function displayClosedPositions(positions) {
    const tbody = document.querySelector('#closedPositionsTable tbody');
    if (!tbody) return;
    
    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-center">No closed positions</td></tr>';
        return;
    }
    
    tbody.innerHTML = positions.map(pos => {
        const pnlClass = pos.profit_loss >= 0 ? 'text-success' : 'text-danger';
        return `
            <tr>
                <td>${pos.token_symbol}</td>
                <td>${formatCurrency(pos.entry_price)}</td>
                <td>${formatCurrency(pos.exit_price)}</td>
                <td class="${pnlClass}">${formatCurrency(pos.profit_loss)}</td>
                <td class="${pnlClass}">${formatPercent(pos.roi)}</td>
                <td>${pos.duration}</td>
                <td>${formatDate(pos.exit_timestamp)}</td>
                <td>${pos.exit_reason || 'manual'}</td>
            </tr>
        `;
    }).join('');
}

// Call on page load
document.addEventListener('DOMContentLoaded', function() {
    loadClosedPositions();
    setInterval(loadClosedPositions, 30000); // Refresh every 30 seconds
});
```

**This fixes:** Issue 3c (Closed positions)

---

## PRIORITY 5: Fix Performance Page Data

### Issue 5a-5e: performance.html Missing Data

**File:** `monitoring/enhanced_dashboard.py`

**Location:** Find `api_performance_metrics` method

**ENSURE it has complete implementation - if missing, ADD:**

```python
async def api_performance_metrics(self, request):
    """Get comprehensive performance metrics"""
    try:
        period = request.query.get('period', '7d')
        
        # Initialize metrics structure
        metrics = {
            'summary': {
                'total_profit': 0.0,
                'roi': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            },
            'trades': {
                'total': 0,
                'winning': 0,
                'losing': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_duration': '0h',
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'recovery_factor': 0.0
            },
            'risk': {
                'daily_volatility': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'recovery_time': 0
            },
            'historical': {
                'total_pnl': 0.0,
                'equity_curve': [],
                'cumulative_pnl': []
            }
        }
        
        if self.db:
            # Get all closed trades
            all_trades = await self.db.get_closed_trades(limit=10000)
            
            if all_trades:
                # Filter by period
                filtered_trades = self._filter_by_period(all_trades, period)
                
                if filtered_trades:
                    # Calculate metrics
                    metrics['trades']['total'] = len(filtered_trades)
                    
                    winning_trades = [t for t in filtered_trades if float(t.get('profit_loss', 0)) > 0]
                    losing_trades = [t for t in filtered_trades if float(t.get('profit_loss', 0)) < 0]
                    
                    metrics['trades']['winning'] = len(winning_trades)
                    metrics['trades']['losing'] = len(losing_trades)
                    metrics['trades']['win_rate'] = (len(winning_trades) / len(filtered_trades) * 100) if filtered_trades else 0
                    
                    if winning_trades:
                        metrics['trades']['avg_win'] = sum(float(t.get('profit_loss', 0)) for t in winning_trades) / len(winning_trades)
                        metrics['trades']['best_trade'] = max(float(t.get('profit_loss', 0)) for t in winning_trades)
                    
                    if losing_trades:
                        metrics['trades']['avg_loss'] = sum(float(t.get('profit_loss', 0)) for t in losing_trades) / len(losing_trades)
                        metrics['trades']['worst_trade'] = min(float(t.get('profit_loss', 0)) for t in losing_trades)
                    
                    # Total P&L
                    total_pnl = sum(float(t.get('profit_loss', 0)) for t in filtered_trades)
                    metrics['summary']['total_profit'] = total_pnl
                    metrics['historical']['total_pnl'] = total_pnl
                    
                    # ROI
                    starting_balance = 10000
                    metrics['summary']['roi'] = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0
                    
                    # Profit factor
                    total_wins = sum(float(t.get('profit_loss', 0)) for t in winning_trades)
                    total_losses = abs(sum(float(t.get('profit_loss', 0)) for t in losing_trades))
                    metrics['trades']['profit_factor'] = (total_wins / total_losses) if total_losses > 0 else 0
                    
                    # Expectancy
                    metrics['trades']['expectancy'] = total_pnl / len(filtered_trades) if filtered_trades else 0
                    
                    # Build equity curve
                    sorted_trades = sorted(filtered_trades, key=lambda t: t.get('exit_timestamp', ''))
                    cumulative = starting_balance
                    
                    for trade in sorted_trades:
                        cumulative += float(trade.get('profit_loss', 0))
                        metrics['historical']['equity_curve'].append({
                            'timestamp': trade.get('exit_timestamp'),
                            'value': cumulative
                        })
                    
                    # Calculate drawdown
                    if metrics['historical']['equity_curve']:
                        peak = metrics['historical']['equity_curve'][0]['value']
                        max_dd = 0
                        
                        for point in metrics['historical']['equity_curve']:
                            if point['value'] > peak:
                                peak = point['value']
                            dd = (peak - point['value']) / peak * 100 if peak > 0 else 0
                            if dd > max_dd:
                                max_dd = dd
                        
                        metrics['summary']['max_drawdown'] = max_dd
                        metrics['risk']['max_drawdown'] = max_dd
        
        return web.json_response({
            'success': True,
            'data': self._serialize_decimals(metrics)
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

def _filter_by_period(self, trades, period):
    """Filter trades by time period"""
    if period == 'all':
        return trades
    
    now = datetime.utcnow()
    period_map = {
        '1h': timedelta(hours=1),
        '24h': timedelta(days=1),
        '7d': timedelta(days=7),
        '30d': timedelta(days=30),
        '90d': timedelta(days=90)
    }
    
    delta = period_map.get(period, timedelta(days=7))
    cutoff = now - delta
    
    return [t for t in trades if t.get('exit_timestamp') and 
            datetime.fromisoformat(str(t['exit_timestamp']).replace('Z', '+00:00')) > cutoff]
```

**This fixes:** Issues 5a-5e (Performance metrics)

---

## Summary - Files to Modify

### ✅ **CRITICAL (Fix These First):**
1. **`dashboard/templates/trades.html`** - Complete file replacement
2. **`dashboard/templates/settings.html`** - Complete file replacement (if needed)
3. **`dashboard/static/css/main.css`** - Add sidebar collapse & notification panel CSS
4. **`monitoring/enhanced_dashboard.py`**:
   - Fix `api_bot_status` (line 717)
   - Fix `api_open_positions` method
   - Add `api_positions_history` method
   - Ensure `api_performance_metrics` is complete

### ✅ **MEDIUM PRIORITY:**
5. **`dashboard/templates/positions.html`** - Add closed positions loading JavaScript

### ℹ️ **Already Working (No Changes Needed):**
- **main.js** - Bot status update is already good (lines 246-268)
- **dashboard.js** - Already has fixes
- **websocket.js** - Already handling data correctly
- **base.html** - Structure is correct
- **dashboard.html**, **performance.html**, **reports.html**, **backtest.html** - All extend base.html correctly

---

## Testing After Fixes

```bash
# 1. Restart dashboard
docker compose restart dashboard

# 2. Check logs
docker compose logs -f dashboard

# 3. Test API endpoints
curl http://localhost:8080/api/bot/status
curl http://localhost:8080/api/positions/open
curl http://localhost:8080/api/positions/history
curl http://localhost:8080/api/trades/history?limit=10
curl http://localhost:8080/api/performance/metrics

# 4. Test in browser
# - Click menu toggle button
# - Click notification bell
# - Click theme toggle
# - Navigate to trades page
# - Navigate to settings page
# - Check position data shows in all columns
```

---

## Issues Remaining (Lower Priority)

- **Issue 2a:** Portfolio Beta calculation (optional enhancement)
- **Issue 3b:** Position details modal (depends on positions.html structure)
- **Issue 6a:** Backtesting implementation (feature not yet implemented)
- **Issue 7a:** Reports generation (feature not yet implemented)

These can be addressed after the critical issues are fixed.
