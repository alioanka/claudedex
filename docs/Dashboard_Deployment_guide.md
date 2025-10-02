# Enhanced Dashboard Deployment Guide

## Overview

This guide covers the complete setup and deployment of the enhanced dashboard for the DexScreener Trading Bot.

## Directory Structure

```
project_root/
├── dashboard/
│   ├── templates/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── trades.html
│   │   ├── positions.html
│   │   ├── performance.html
│   │   ├── settings.html
│   │   ├── reports.html
│   │   └── backtest.html
│   ├── static/
│   │   ├── css/
│   │   │   ├── main.css
│   │   │   ├── dashboard.css
│   │   │   └── charts.css
│   │   ├── js/
│   │   │   ├── main.js
│   │   │   ├── websocket.js
│   │   │   ├── dashboard.js
│   │   │   ├── charts.js
│   │   │   └── settings.js
│   │   └── img/
│   └── Dockerfile
├── monitoring/
│   ├── dashboard.py (your existing basic dashboard)
│   └── enhanced_dashboard.py (new enhanced dashboard)
├── docker-compose.yml
└── requirements.txt
```

## Installation Steps

### 1. Install Python Dependencies

Add these to your `requirements.txt`:

```txt
aiohttp==3.9.1
aiohttp-cors==0.7.0
aiohttp-sse==2.2.0
python-socketio==5.10.0
jinja2==3.1.2
pandas==2.1.4
openpyxl==3.1.2
reportlab==4.0.7
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Create Dashboard Directory Structure

```bash
mkdir -p dashboard/templates
mkdir -p dashboard/static/{css,js,img}
```

### 3. Copy Dashboard Files

Place the files in the appropriate directories:

- `enhanced_dashboard.py` → `monitoring/enhanced_dashboard.py`
- `base.html` → `dashboard/templates/base.html`
- `dashboard.html` → `dashboard/templates/dashboard.html`
- `main.css` → `dashboard/static/css/main.css`
- `main.js` → `dashboard/static/js/main.js`
- `websocket.js` → `dashboard/static/js/websocket.js`

### 4. Create Remaining Templates

Create these additional templates based on the same pattern:

#### `dashboard/templates/trades.html`

```html
{% extends "base.html" %}

{% block title %}Recent Trades - DexScreener Trading Bot{% endblock %}

{% block content %}
<div class="page-header">
    <h1>Recent Trades</h1>
    <div class="page-actions">
        <button class="btn btn-secondary" onclick="refreshTrades()">
            <i class="fas fa-sync"></i> Refresh
        </button>
        <button class="btn btn-primary" onclick="exportData('csv', '/api/reports/export/csv')">
            <i class="fas fa-download"></i> Export CSV
        </button>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <h3 class="card-title">
            <i class="fas fa-exchange-alt"></i>
            Trade History
        </h3>
        <div class="card-actions">
            <input type="date" id="startDate" class="form-control" style="width: auto;">
            <input type="date" id="endDate" class="form-control" style="width: auto;">
            <select id="statusFilter" class="form-control" style="width: auto;">
                <option value="">All Status</option>
                <option value="open">Open</option>
                <option value="closed">Closed</option>
                <option value="cancelled">Cancelled</option>
            </select>
            <button class="btn btn-primary" onclick="filterTrades()">
                <i class="fas fa-filter"></i> Filter
            </button>
        </div>
    </div>
    <div class="card-body">
        <div class="table-container">
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Token</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Amount</th>
                        <th>P&L</th>
                        <th>ROI</th>
                        <th>Strategy</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="tradesTableBody">
                    <tr>
                        <td colspan="10" class="text-center">
                            <div class="spinner"></div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
async function loadTrades() {
    const tbody = document.getElementById('tradesTableBody');
    tbody.innerHTML = '<tr><td colspan="10" class="text-center"><div class="spinner"></div></td></tr>';
    
    try {
        const response = await apiGet('/api/trades/recent?limit=100');
        if (response.success) {
            displayTrades(response.data);
        }
    } catch (error) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-danger">Failed to load trades</td></tr>';
    }
}

function displayTrades(trades) {
    const tbody = document.getElementById('tradesTableBody');
    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No trades found</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.map(trade => {
        const pnl = parseFloat(trade.pnl || 0);
        const roi = parseFloat(trade.roi || 0);
        const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
        const sideClass = trade.side === 'buy' ? 'badge-success' : 'badge-danger';
        
        return `
            <tr>
                <td>${formatDate(trade.timestamp)}</td>
                <td>${trade.token_symbol}</td>
                <td><span class="badge ${sideClass}">${trade.side.toUpperCase()}</span></td>
                <td>${formatCurrency(trade.entry_price)}</td>
                <td>${trade.exit_price ? formatCurrency(trade.exit_price) : '-'}</td>
                <td>${formatNumber(trade.amount, 4)}</td>
                <td class="${pnlClass}">${formatCurrency(pnl)}</td>
                <td class="${pnlClass}">${formatPercent(roi)}</td>
                <td>${trade.strategy || 'Manual'}</td>
                <td><span class="badge badge-${trade.status === 'closed' ? 'success' : 'warning'}">${trade.status}</span></td>
            </tr>
        `;
    }).join('');
}

async function filterTrades() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const status = document.getElementById('statusFilter').value;
    
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    if (status) params.append('status', status);
    
    const tbody = document.getElementById('tradesTableBody');
    tbody.innerHTML = '<tr><td colspan="10" class="text-center"><div class="spinner"></div></td></tr>';
    
    try {
        const response = await apiGet(`/api/trades/history?${params.toString()}`);
        if (response.success) {
            displayTrades(response.data);
        }
    } catch (error) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-danger">Failed to load trades</td></tr>';
    }
}

function refreshTrades() {
    loadTrades();
}

document.addEventListener('DOMContentLoaded', loadTrades);
</script>
{% endblock %}
```

#### `dashboard/templates/settings.html`

```html
{% extends "base.html" %}

{% block title %}Settings - DexScreener Trading Bot{% endblock %}

{% block content %}
<div class="page-header">
    <h1>Bot Settings</h1>
    <div class="page-actions">
        <button class="btn btn-secondary" id="revertBtn" onclick="revertSettings()">
            <i class="fas fa-undo"></i> Revert Changes
        </button>
        <button class="btn btn-primary" id="saveBtn" onclick="saveSettings()">
            <i class="fas fa-save"></i> Save Settings
        </button>
    </div>
</div>

<!-- Settings Categories -->
<div class="settings-tabs">
    <button class="tab-btn active" data-tab="trading">Trading</button>
    <button class="tab-btn" data-tab="risk">Risk Management</button>
    <button class="tab-btn" data-tab="api">API Configuration</button>
    <button class="tab-btn" data-tab="monitoring">Monitoring</button>
    <button class="tab-btn" data-tab="ml">ML Models</button>
</div>

<!-- Trading Settings -->
<div class="tab-content active" id="trading-tab">
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Trading Configuration</h3>
        </div>
        <div class="card-body">
            <div class="form-group">
                <label class="form-label">Max Position Size (%)</label>
                <input type="number" class="form-control" id="maxPositionSize" step="0.1" min="0" max="100">
                <div class="form-help">Maximum percentage of portfolio per position</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Max Open Positions</label>
                <input type="number" class="form-control" id="maxOpenPositions" min="1" max="50">
                <div class="form-help">Maximum number of concurrent positions</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Min Trade Size (USD)</label>
                <input type="number" class="form-control" id="minTradeSize" min="10">
                <div class="form-help">Minimum trade size in USD</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Default Slippage Tolerance (%)</label>
                <input type="number" class="form-control" id="slippageTolerance" step="0.1" min="0">
                <div class="form-help">Maximum acceptable slippage</div>
            </div>
        </div>
    </div>
</div>

<!-- Risk Management Settings -->
<div class="tab-content" id="risk-tab">
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Risk Management</h3>
        </div>
        <div class="card-body">
            <div class="form-group">
                <label class="form-label">Max Daily Loss (%)</label>
                <input type="number" class="form-control" id="maxDailyLoss" step="0.1" min="0" max="100">
                <div class="form-help">Stop trading if daily loss exceeds this percentage</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Max Drawdown (%)</label>
                <input type="number" class="form-control" id="maxDrawdown" step="0.1" min="0" max="100">
                <div class="form-help">Maximum acceptable portfolio drawdown</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Stop Loss Default (%)</label>
                <input type="number" class="form-control" id="stopLossDefault" step="0.1" min="0">
                <div class="form-help">Default stop loss percentage</div>
            </div>
            
            <div class="form-group">
                <label class="form-label">Take Profit Default (%)</label>
                <input type="number" class="form-control" id="takeProfitDefault" step="0.1" min="0">
                <div class="form-help">Default take profit percentage</div>
            </div>
        </div>
    </div>
</div>

<script>
let currentSettings = {};
let originalSettings = {};

async function loadSettings() {
    try {
        const response = await apiGet('/api/settings/all');
        if (response.success) {
            currentSettings = response.data;
            originalSettings = JSON.parse(JSON.stringify(response.data));
            populateSettings(currentSettings);
        }
    } catch (error) {
        showToast('error', 'Failed to load settings');
    }
}

function populateSettings(settings) {
    // Trading settings
    if (settings.trading) {
        document.getElementById('maxPositionSize').value = settings.trading.max_position_size || 0;
        document.getElementById('maxOpenPositions').value = settings.trading.max_open_positions || 0;
        document.getElementById('minTradeSize').value = settings.trading.min_trade_size || 0;
        document.getElementById('slippageTolerance').value = settings.trading.slippage_tolerance || 0;
    }
    
    // Risk settings
    if (settings.risk) {
        document.getElementById('maxDailyLoss').value = settings.risk.max_daily_loss_percent || 0;
        document.getElementById('maxDrawdown').value = settings.risk.max_drawdown_percent || 0;
        document.getElementById('stopLossDefault').value = settings.risk.stop_loss_percent || 0;
        document.getElementById('takeProfitDefault').value = settings.risk.take_profit_percent || 0;
    }
}

async function saveSettings() {
    const confirmed = await showConfirmation(
        'Save Settings',
        'Changes will be applied immediately. Continue?'
    );
    
    if (!confirmed) return;
    
    try {
        // Collect updated settings
        const updates = {
            trading: {
                max_position_size: parseFloat(document.getElementById('maxPositionSize').value),
                max_open_positions: parseInt(document.getElementById('maxOpenPositions').value),
                min_trade_size: parseFloat(document.getElementById('minTradeSize').value),
                slippage_tolerance: parseFloat(document.getElementById('slippageTolerance').value)
            },
            risk: {
                max_daily_loss_percent: parseFloat(document.getElementById('maxDailyLoss').value),
                max_drawdown_percent: parseFloat(document.getElementById('maxDrawdown').value),
                stop_loss_percent: parseFloat(document.getElementById('stopLossDefault').value),
                take_profit_percent: parseFloat(document.getElementById('takeProfitDefault').value)
            }
        };
        
        // Save each config type
        for (const [configType, configData] of Object.entries(updates)) {
            const response = await apiPost('/api/settings/update', {
                config_type: configType,
                updates: configData
            });
            
            if (!response.success) {
                throw new Error(`Failed to update ${configType} settings`);
            }
        }
        
        showToast('success', 'Settings saved successfully');
        originalSettings = JSON.parse(JSON.stringify(updates));
        
    } catch (error) {
        showToast('error', 'Failed to save settings: ' + error.message);
    }
}

async function revertSettings() {
    const confirmed = await showConfirmation(
        'Revert Settings',
        'Restore previous settings version?'
    );
    
    if (!confirmed) return;
    
    try {
        const response = await apiPost('/api/settings/revert', {
            config_type: 'all',
            version: -1
        });
        
        if (response.success) {
            showToast('success', 'Settings reverted');
            await loadSettings();
        }
    } catch (error) {
        showToast('error', 'Failed to revert settings');
    }
}

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const tabName = this.getAttribute('data-tab');
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        
        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName + '-tab').classList.add('active');
    });
});

document.addEventListener('DOMContentLoaded', loadSettings);
</script>

<style>
.settings-tabs {
    display: flex;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-lg);
    border-bottom: 2px solid var(--border-color);
}

.tab-btn {
    padding: var(--spacing-md) var(--spacing-lg);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-secondary);
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition-fast);
    margin-bottom: -2px;
}

.tab-btn:hover {
    color: var(--text-primary);
}

.tab-btn.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}
</style>
{% endblock %}
```

### 5. Update main.py Integration

Add to your `main.py`:

```python
from monitoring.enhanced_dashboard import DashboardEndpoints

# In your TradingBotApplication.__init__:
self.dashboard = DashboardEndpoints(
    host="0.0.0.0",
    port=8080,
    config=self.config,
    trading_engine=self.engine,
    portfolio_manager=self.portfolio_manager,
    order_manager=self.order_manager,
    risk_manager=self.risk_manager,
    alerts_system=self.alerts_system,
    config_manager=self.config_manager,
    db_manager=self.db_manager
)

# In your run() method:
asyncio.create_task(self.dashboard.start())
```

### 6. Update Docker Configuration

Update `docker-compose.yml` to expose dashboard port:

```yaml
services:
  trading-bot:
    ports:
      - "8080:8080"  # Dashboard
```

### 7. Run the Dashboard

```bash
# Development
python main.py

# Production with Docker
docker-compose up -d
```

Access dashboard at: `http://localhost:8080`

## Features

### 1. Real-time Updates
- WebSocket connections for live data
- Auto-refresh every 1-2 seconds
- Real-time portfolio value tracking

### 2. Bot Controls
- Start/Stop/Restart bot
- Emergency exit (close all positions)
- View bot status and uptime

### 3. Settings Management
- Edit all configuration dynamically
- Changes apply immediately (no restart needed)
- Revert to previous versions
- View change history

### 4. Reports & Export
- Generate custom reports
- Export to CSV, Excel, PDF, JSON
- Filter by date range, strategy, tokens
- Schedule automated reports

### 5. Backtesting
- Run backtests from UI
- Configure strategy parameters
- View detailed results
- Compare multiple strategies

### 6. Manual Trading
- Execute trades manually
- Close/modify positions
- Cancel pending orders
- Set custom stop-loss/take-profit

## Security Notes

1. Dashboard has no authentication by default
2. Add nginx reverse proxy with SSL for production
3. Consider adding API key authentication
4. Use firewall rules to restrict access

## Troubleshooting

### Dashboard won't start
- Check port 8080 is not in use
- Verify all dependencies installed
- Check logs for errors

### WebSocket not connecting
- Ensure Socket.IO client version matches server
- Check CORS settings
- Verify firewall rules

### Charts not displaying
- Check browser console for errors
- Verify Chart.js is loading
- Ensure data endpoints returning valid JSON

## Support

For issues or questions, check the logs:
```bash
tail -f logs/trading_bot.log
```

## Next Steps

1. Customize colors/theme in `main.css`
2. Add custom metrics to dashboard
3. Create additional report templates
4. Add more chart types
5. Implement notification preferences