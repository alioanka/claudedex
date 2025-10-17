// WebSocket connection handler for real-time updates

let socket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000;

function initWebSocket() {
    // Initialize Socket.IO connection
    socket = io({
        reconnection: true,
        reconnectionDelay: RECONNECT_DELAY,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS
    });
    
    // Connection events
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('reconnect', handleReconnect);
    socket.on('reconnect_error', handleReconnectError);
    
    // Data events
    socket.on('initial_data', handleInitialData);
    socket.on('dashboard_update', handleDashboardUpdate);
    socket.on('positions_update', handlePositionsUpdate);
    socket.on('orders_update', handleOrdersUpdate);
    socket.on('alerts_update', handleAlertsUpdate);
    socket.on('chart_update', handleChartUpdate);
    socket.on('action_result', handleActionResult);
}

function handleConnect() {
    console.log('WebSocket connected');
    reconnectAttempts = 0;
    showToast('success', 'Connected to trading bot');
    
    // Update connection status
    updateConnectionStatus(true);
}

function handleDisconnect(reason) {
    console.log('WebSocket disconnected:', reason);
    showToast('warning', 'Disconnected from trading bot');
    
    // Update connection status
    updateConnectionStatus(false);
}

function handleReconnect(attemptNumber) {
    console.log('WebSocket reconnected after', attemptNumber, 'attempts');
    showToast('success', 'Reconnected to trading bot');
    reconnectAttempts = 0;
}

function handleReconnectError(error) {
    reconnectAttempts++;
    console.error('Reconnection error:', error);
    
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        showToast('error', 'Failed to reconnect. Please refresh the page.');
    }
}

function handleInitialData(data) {
    console.log('Received initial data:', data);
    
    // Update dashboard with initial data
    if (data.portfolio) {
        updateDashboardUI(data.portfolio);
    }
    
    // ✅ Only update if we have actual data
    if (data.positions && data.positions.length > 0) {
        state.positions = data.positions;
        updatePositionsTable(data.positions);
    } else {
        // Load from API instead
        if (typeof loadPositions === 'function') {
            loadPositions();
        }
    }
    
    if (data.orders && data.orders.length > 0) {
        state.orders = data.orders;
        updateOrdersTable(data.orders);
    } else {
        // Load from API instead
        if (typeof loadOrders === 'function') {
            loadOrders();
        }
    }
}

function handleDashboardUpdate(data) {
    // ✅ ONLY update LIVE metrics from WebSocket
    if (data.portfolio_value !== undefined) {
        const portfolioValueStat = document.getElementById('portfolioValueStat');
        if (portfolioValueStat) {
            portfolioValueStat.textContent = formatCurrency(data.portfolio_value);
        }
    }
    
    if (data.open_positions !== undefined) {
        const openPositionsStat = document.getElementById('openPositionsStat');
        if (openPositionsStat) {
            openPositionsStat.textContent = data.open_positions;
        }
    }
    
    if (data.daily_pnl !== undefined) {
        const pnlIndicator = document.getElementById('pnlIndicator');
        if (pnlIndicator) {
            const valueElement = pnlIndicator.querySelector('.value');
            if (valueElement) {
                valueElement.textContent = formatCurrency(data.daily_pnl);
                pnlIndicator.classList.remove('positive', 'negative');
                pnlIndicator.classList.add(data.daily_pnl >= 0 ? 'positive' : 'negative');
            }
        }
    }
    
    // ❌ DO NOT update totalPnlStat or winRateStat from WebSocket
    // Those are HISTORICAL and should only load once from /api/performance/metrics
}

function handlePositionsUpdate(data) {
    if (data.data) {
        state.positions = data.data;
        updatePositionsTable(data.data);
    }
}

function handleOrdersUpdate(data) {
    if (data.data) {
        state.orders = data.data;
        updateOrdersTable(data.data);
    }
}

function handleAlertsUpdate(data) {
    if (data.data) {
        state.alerts = data.data;
        showNewAlerts(data.data);
    }
}

function handleChartUpdate(data) {
    if (data.chart_id && data.data) {
        updateChart(data.chart_id, data.data);
    }
}

function handleActionResult(data) {
    if (data.success) {
        showToast('success', data.message || 'Action completed successfully');
    } else {
        showToast('error', data.message || 'Action failed');
    }
}

function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('botStatus');
    if (statusIndicator) {
        if (connected) {
            statusIndicator.classList.remove('offline');
            statusIndicator.classList.add('online');
        } else {
            statusIndicator.classList.remove('online');
            statusIndicator.classList.add('offline');
            
            const statusText = statusIndicator.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Connecting...';
            }
        }
    }
}

function updatePositionsTable(positions) {
    const table = document.getElementById('positionsTable');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    positions.forEach(position => {
        const row = createPositionRow(position);
        tbody.appendChild(row);
    });
}

function createPositionRow(position) {
    const row = document.createElement('tr');
    
    const pnl = parseFloat(position.unrealized_pnl || 0);
    const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';
    const pnlIcon = pnl >= 0 ? '▲' : '▼';
    
    row.innerHTML = `
        <td>${position.token_symbol || 'Unknown'}</td>
        <td>${formatCurrency(position.entry_price)}</td>
        <td>${formatCurrency(position.current_price || position.entry_price)}</td>
        <td>${formatNumber(position.amount, 4)}</td>
        <td class="${pnlClass}">${pnlIcon} ${formatCurrency(pnl)}</td>
        <td>
            <span class="badge badge-${position.status === 'open' ? 'success' : 'neutral'}">
                ${position.status || 'Unknown'}
            </span>
        </td>
        <td>
            <button class="btn btn-sm btn-secondary" onclick="viewPosition('${position.id}')">
                <i class="fas fa-eye"></i>
            </button>
            <button class="btn btn-sm btn-danger" onclick="closePosition('${position.id}')">
                <i class="fas fa-times"></i>
            </button>
        </td>
    `;
    
    return row;
}

function updateOrdersTable(orders) {
    const table = document.getElementById('ordersTable');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    orders.forEach(order => {
        const row = createOrderRow(order);
        tbody.appendChild(row);
    });
}

function createOrderRow(order) {
    const row = document.createElement('tr');
    
    const statusClass = {
        'pending': 'badge-warning',
        'filled': 'badge-success',
        'cancelled': 'badge-danger',
        'failed': 'badge-danger'
    }[order.status] || 'badge-neutral';
    
    row.innerHTML = `
        <td>${order.token_symbol || 'Unknown'}</td>
        <td>
            <span class="badge badge-${order.side === 'buy' ? 'success' : 'danger'}">
                ${order.side ? order.side.toUpperCase() : 'Unknown'}
            </span>
        </td>
        <td>${order.type || 'Market'}</td>
        <td>${formatCurrency(order.price || 0)}</td>
        <td>${formatNumber(order.amount, 4)}</td>
        <td><span class="badge ${statusClass}">${order.status || 'Unknown'}</span></td>
        <td>${timeAgo(order.timestamp)}</td>
        <td>
            ${order.status === 'pending' ? `
                <button class="btn btn-sm btn-danger" onclick="cancelOrder('${order.id}')">
                    <i class="fas fa-times"></i> Cancel
                </button>
            ` : ''}
        </td>
    `;
    
    return row;
}

function showNewAlerts(alerts) {
    const notificationContent = document.getElementById('notificationContent');
    if (!notificationContent) return;
    
    // Clear existing notifications
    notificationContent.innerHTML = '';
    
    // Add new notifications
    alerts.forEach(alert => {
        const notification = createNotification(alert);
        notificationContent.appendChild(notification);
    });
    
    // Update badge count
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        const unreadCount = alerts.filter(a => !a.read).length;
        badge.textContent = unreadCount;
        badge.style.display = unreadCount > 0 ? 'block' : 'none';
    }
    
    // Show toast for new critical alerts
    alerts.forEach(alert => {
        if (alert.priority === 'critical' && !alert.shown) {
            showToast('error', alert.message);
            alert.shown = true;
        }
    });
}

function createNotification(alert) {
    const div = document.createElement('div');
    div.className = 'notification-item';
    
    const priorityClass = {
        'low': 'info',
        'medium': 'warning',
        'high': 'warning',
        'critical': 'error'
    }[alert.priority] || 'info';
    
    const iconMap = {
        'low': 'fa-info-circle',
        'medium': 'fa-exclamation-circle',
        'high': 'fa-exclamation-triangle',
        'critical': 'fa-exclamation-triangle'
    };
    
    div.innerHTML = `
        <div class="notification-icon ${priorityClass}">
            <i class="fas ${iconMap[alert.priority]}"></i>
        </div>
        <div class="notification-content">
            <div class="notification-title">${alert.title || alert.type}</div>
            <div class="notification-message">${alert.message}</div>
            <div class="notification-time">${timeAgo(alert.timestamp)}</div>
        </div>
    `;
    
    return div;
}

// Position actions
async function viewPosition(positionId) {
    // Implement position details view
    console.log('View position:', positionId);
}

async function closePosition(positionId) {
    const confirmed = await showConfirmation(
        'Close Position',
        'Are you sure you want to close this position?'
    );
    
    if (!confirmed) return;
    
    try {
        const result = await apiPost('/api/position/close', { position_id: positionId });
        
        if (result.success) {
            showToast('success', 'Position closed successfully');
        } else {
            showToast('error', result.error || 'Failed to close position');
        }
    } catch (error) {
        console.error('Error closing position:', error);
        showToast('error', 'Failed to close position');
    }
}

// Order actions
async function cancelOrder(orderId) {
    try {
        const result = await apiPost('/api/order/cancel', { order_id: orderId });
        
        if (result.success) {
            showToast('success', 'Order cancelled successfully');
        } else {
            showToast('error', result.error || 'Failed to cancel order');
        }
    } catch (error) {
        console.error('Error cancelling order:', error);
        showToast('error', 'Failed to cancel order');
    }
}

// Export functions
window.initWebSocket = initWebSocket;
window.socket = socket;
window.viewPosition = viewPosition;
window.closePosition = closePosition;
window.cancelOrder = cancelOrder;