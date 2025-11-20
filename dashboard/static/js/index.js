// dashboard/static/js/index.js

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load initial data
    loadDashboardData();
    loadInsights();

    // Setup auto-refresh
    setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
});

async function loadInsights() {
    try {
        const response = await apiGet('/api/insights');
        if (response.success) {
            const insightsList = document.getElementById('insightsList');
            if (insightsList) {
                insightsList.innerHTML = response.data.map(insight => `<li>${insight}</li>`).join('');
            }
        }
    } catch (error) {
        console.error('Failed to load insights:', error);
    }
}

async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard/summary');
        const data = await response.json();

        if (data.success) {
            updateDashboard(data.data);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

function updateDashboard(data) {
    // Update metrics
    updateElement('portfolioValue', formatCurrency(data.portfolio_value));
    updateElement('dailyPnl', formatCurrency(data.daily_pnl));
    updateElement('totalPnl', formatCurrency(data.total_pnl));
    updateElement('winRate', `${data.win_rate}%`);
    updateElement('activePositions', data.open_positions);

    // Update changes
    updateChangeElement('portfolioChange', data.portfolio_change);
    updateChangeElement('dailyPnlChange', data.daily_pnl_change);
    updateChangeElement('totalPnlChange', data.total_pnl_change);
    updateElement('winRateChange', `${data.wins} wins / ${data.losses} losses`);

    // Update charts
    if (data.portfolio_history) {
        chartManager.createPortfolioChart('portfolioChart', {
            labels: data.portfolio_history.labels,
            values: data.portfolio_history.values
        });
    }

    if (data.win_loss_data) {
        chartManager.createWinRateChart('winRateChart',
            data.win_loss_data.wins,
            data.win_loss_data.losses
        );
    }

    // Update activity feed
    if (data.recent_activity) {
        updateActivityFeed(data.recent_activity);
    }

    // Update open positions
    if (data.open_positions_list) {
        updateOpenPositions(data.open_positions_list);
    }

    // Update bot status
    updateBotStatus(data.bot_status);
}

function updateActivityFeed(activities) {
    const feed = document.getElementById('activityFeed');
    if (!activities || activities.length === 0) return;

    feed.innerHTML = activities.map(activity => `
        <div class="activity-item">
            <div class="activity-icon ${activity.type}">
                ${activity.type === 'buy' ? '📈' : activity.type === 'sell' ? '📉' : '⚠️'}
            </div>
            <div class="activity-content">
                <div class="activity-title">${activity.title}</div>
                <div class="activity-description">${activity.description}</div>
                <div class="activity-time">${formatTimestamp(activity.timestamp)}</div>
            </div>
        </div>
    `).join('');
}

function updateOpenPositions(positions) {
    const container = document.getElementById('openPositions');
    if (!positions || positions.length === 0) return;

    container.innerHTML = positions.map(pos => `
        <div class="position-card">
            <div class="position-header">
                <span class="position-token">${pos.token_symbol}</span>
                <span class="position-type ${pos.position_type}">${pos.position_type}</span>
            </div>
            <div class="position-body">
                <div class="position-metric">
                    <div class="position-metric-label">Entry Price</div>
                    <div class="position-metric-value">${formatCurrency(pos.entry_price)}</div>
                </div>
                <div class="position-metric">
                    <div class="position-metric-label">Current Price</div>
                    <div class="position-metric-value">${formatCurrency(pos.current_price)}</div>
                </div>
                <div class="position-metric">
                    <div class="position-metric-label">P&L</div>
                    <div class="position-metric-value ${pos.pnl >= 0 ? 'positive' : 'negative'}">
                        ${formatCurrency(pos.pnl)}
                    </div>
                </div>
                <div class="position-metric">
                    <div class="position-metric-label">ROI</div>
                    <div class="position-metric-value ${pos.roi >= 0 ? 'positive' : 'negative'}">
                        ${pos.roi.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

function updateBotStatus(status) {
    const statusEl = document.getElementById('botStatus');
    const indicator = statusEl.querySelector('.status-indicator');
    const text = statusEl.querySelector('.status-text');

    indicator.className = 'status-indicator';
    if (status === 'running') {
        indicator.classList.add('status-online');
        text.textContent = 'Running';
    } else if (status === 'stopped') {
        indicator.classList.add('status-offline');
        text.textContent = 'Stopped';
    } else {
        indicator.classList.add('status-warning');
        text.textContent = 'Unknown';
    }
}

// Bot control functions
async function startBot() {
    if (!confirm('Start the trading bot?')) return;
    await botControl('start');
}

async function stopBot() {
    if (!confirm('Stop the trading bot?')) return;
    await botControl('stop');
}

async function restartBot() {
    if (!confirm('Restart the trading bot?')) return;
    await botControl('restart');
}

async function emergencyExit() {
    if (!confirm('EMERGENCY: Close all positions and stop trading?')) return;
    await botControl('emergency_exit');
}

async function botControl(action) {
    showLoading(`${action}...`);
    try {
        const response = await fetch(`/api/bot/${action}`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showSuccess(data.message || `Bot ${action} successful`);
            setTimeout(loadDashboardData, 2000);
        } else {
            showError(data.error || `Failed to ${action} bot`);
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}
