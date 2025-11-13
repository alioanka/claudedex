// dashboard/static/js/performance.js

async function loadPerformanceData() {
    const timeframe = document.getElementById('performanceTimeframe').value;

    try {
        // ✅ Try without period parameter first
        let response = await apiGet('/api/performance/metrics');

        if (response.success && response.data) {
            updatePerformanceMetrics(response.data);
            await loadPerformanceCharts(timeframe);
        } else {
            // Fallback with period parameter
            response = await apiGet(`/api/performance/metrics?period=${timeframe}`);
            if (response.success) {
                updatePerformanceMetrics(response.data);
                await loadPerformanceCharts(timeframe);
            }
        }
    } catch (error) {
        console.error('Failed to load performance data:', error);
        showToast('error', 'Failed to load performance data');
    }
}

function updatePerformanceMetrics(data) {
    // ✅ Handle both direct data and nested historical data
    const metrics = data.historical || data;

    // Key metrics
    document.getElementById('totalProfit').textContent = formatCurrency(metrics.total_pnl || 0);
    document.getElementById('totalRoi').textContent = formatPercent(metrics.roi || 0);
    document.getElementById('sharpeRatio').textContent = formatNumber(metrics.sharpe_ratio || 0, 2);
    document.getElementById('maxDrawdown').textContent = formatPercent(metrics.max_drawdown || 0);

    // Trade statistics
    document.getElementById('totalTrades').textContent = metrics.total_trades || 0;
    document.getElementById('winningTrades').textContent = metrics.winning_trades || 0;
    document.getElementById('losingTrades').textContent = metrics.losing_trades || 0;
    document.getElementById('winRate').textContent = formatPercent(metrics.win_rate || 0);
    document.getElementById('avgWin').textContent = formatCurrency(metrics.avg_win || 0);
    document.getElementById('avgLoss').textContent = formatCurrency(metrics.avg_loss || 0);
    document.getElementById('bestTrade').textContent = formatCurrency(metrics.best_trade || 0);
    document.getElementById('worstTrade').textContent = formatCurrency(metrics.worst_trade || 0);
    document.getElementById('profitFactor').textContent = formatNumber(metrics.profit_factor || 0, 2);
    document.getElementById('expectancy').textContent = formatCurrency(metrics.expectancy || 0);
    document.getElementById('recoveryFactor').textContent = formatNumber(metrics.recovery_factor || 0, 2);

    // Risk metrics
    document.getElementById('sharpeRatioDetail').textContent = formatNumber(metrics.sharpe_ratio || 0, 2);
    document.getElementById('sortinoRatio').textContent = formatNumber(metrics.sortino_ratio || 0, 2);
    document.getElementById('maxDrawdownDetail').textContent = formatPercent(metrics.max_drawdown || 0);
}

async function loadPerformanceCharts(timeframe) {
    try {
        const response = await apiGet(`/api/performance/charts?timeframe=${timeframe}`);
        if (response.success) {
            if (response.data.equity_curve) {
                createEquityCurveChart(response.data.equity_curve);
            }
            if (response.data.cumulative_pnl) {
                createCumulativePnlChart(response.data.cumulative_pnl);
            }
            if (response.data.strategy_performance) {
                createStrategyChart(response.data.strategy_performance);
            }
            if (response.data.win_loss) {
                createWinLossChart(response.data.win_loss);
            }
            if (response.data.monthly) {
                createMonthlyChart(response.data.monthly);
            }
        }
    } catch (error) {
        console.error('Failed to load charts:', error);
    }
}

function createEquityCurveChart(data) {
    createChart('equityCurveChart', {
        type: 'line',
        data: {
            labels: data.map(d => formatDate(d.timestamp)),
            datasets: [{
                label: 'Portfolio Value',
                data: data.map(d => d.value),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    ticks: { callback: value => formatCurrency(value) }
                }
            }
        }
    });
}

function createCumulativePnlChart(data) {
    createChart('cumulativePnlChart', {
        type: 'line',
        data: {
            labels: data.map(d => formatDate(d.timestamp)),
            datasets: [{
                label: 'Cumulative P&L',
                data: data.map(d => d.cumulative_pnl),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    ticks: { callback: value => formatCurrency(value) }
                }
            }
        }
    });
}

function createStrategyChart(data) {
    createChart('strategyChart', {
        type: 'bar',
        data: {
            labels: data.map(d => d.strategy),
            datasets: [{
                label: 'P&L',
                data: data.map(d => d.pnl),
                backgroundColor: data.map(d => d.pnl >= 0 ?
                    'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)')
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

function createWinLossChart(data) {
    createChart('winLossChart', {
        type: 'doughnut',
        data: {
            labels: ['Wins', 'Losses'],
            datasets: [{
                data: [data.wins, data.losses],
                backgroundColor: ['rgba(16, 185, 129, 0.8)', 'rgba(239, 68, 68, 0.8)']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function createMonthlyChart(data) {
    createChart('monthlyChart', {
        type: 'bar',
        data: {
            labels: data.map(d => d.month),
            datasets: [{
                label: 'Monthly P&L',
                data: data.map(d => d.pnl),
                backgroundColor: data.map(d => d.pnl >= 0 ?
                    'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)')
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

function refreshPerformance() {
    loadPerformanceData();
}

document.addEventListener('DOMContentLoaded', loadPerformanceData);
document.getElementById('performanceTimeframe').addEventListener('change', loadPerformanceData);
