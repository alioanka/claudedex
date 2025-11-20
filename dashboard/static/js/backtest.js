// dashboard/static/js/backtest.js

let currentBacktestId = null;

async function runBacktest() {
    const strategy = document.getElementById('strategySelect').value;
    const startDate = document.getElementById('backtestStartDate').value;
    const endDate = document.getElementById('backtestEndDate').value;
    const initialBalance = parseFloat(document.getElementById('initialBalance').value);
    const paramsText = document.getElementById('strategyParams').value;

    if (!startDate || !endDate) {
        showToast('warning', 'Please select date range');
        return;
    }

    let parameters = {};
    try {
        parameters = JSON.parse(paramsText);
    } catch (e) {
        showToast('error', 'Invalid JSON in parameters');
        return;
    }

    // Show progress
    document.getElementById('backtestProgress').style.display = 'block';
    document.getElementById('backtestResults').style.display = 'none';

    try {
        const response = await apiPost('/api/backtest/run', {
            strategy: strategy,
            start_date: startDate,
            end_date: endDate,
            initial_balance: initialBalance,
            parameters: parameters
        });

        if (response.success) {
            currentBacktestId = response.test_id;
            showToast('success', 'Backtest started');

            // Poll for results
            pollBacktestResults(response.test_id);
        } else {
            showToast('error', 'Failed to start backtest');
            document.getElementById('backtestProgress').style.display = 'none';
        }
    } catch (error) {
        showToast('error', 'Backtest failed: ' + error.message);
        document.getElementById('backtestProgress').style.display = 'none';
    }
}

async function pollBacktestResults(testId) {
    const maxAttempts = 60; // 5 minutes max
    let attempts = 0;

    const poll = setInterval(async () => {
        attempts++;

        try {
            const response = await apiGet(`/api/backtest/results/${testId}`);

            if (response.success && response.data.status === 'completed') {
                clearInterval(poll);
                displayBacktestResults(response.data);
                document.getElementById('backtestProgress').style.display = 'none';
                document.getElementById('backtestResults').style.display = 'block';
            } else if (response.data.status === 'failed') {
                clearInterval(poll);
                showToast('error', 'Backtest failed');
                document.getElementById('backtestProgress').style.display = 'none';
            } else {
                // Update progress
                document.getElementById('progressText').textContent =
                    response.data.progress || 'Processing...';
            }
        } catch (error) {
            console.error('Error polling backtest:', error);
        }

        if (attempts >= maxAttempts) {
            clearInterval(poll);
            showToast('error', 'Backtest timeout');
            document.getElementById('backtestProgress').style.display = 'none';
        }
    }, 5000); // Poll every 5 seconds
}

function displayBacktestResults(results) {
    // Summary stats
    document.getElementById('finalBalance').textContent = formatNumber(results.final_balance, 4) + ' ETH';
    document.getElementById('totalReturn').textContent = formatPercent(results.total_return);
    document.getElementById('backtestTotalTrades').textContent = results.total_trades;
    document.getElementById('backtestWinRate').textContent = formatPercent(results.win_rate);

    // Performance metrics
    document.getElementById('backtestPnl').textContent = formatNumber(results.total_pnl, 4) + ' ETH';
    document.getElementById('backtestSharpe').textContent = formatNumber(results.sharpe_ratio, 2);
    document.getElementById('backtestSortino').textContent = formatNumber(results.sortino_ratio, 2);
    document.getElementById('backtestMaxDD').textContent = formatPercent(results.max_drawdown);
    document.getElementById('backtestProfitFactor').textContent = formatNumber(results.profit_factor, 2);

    // Trade stats
    document.getElementById('backtestWins').textContent = results.winning_trades;
    document.getElementById('backtestLosses').textContent = results.losing_trades;
    document.getElementById('backtestAvgWin').textContent = formatNumber(results.avg_win, 4) + ' ETH';
    document.getElementById('backtestAvgLoss').textContent = formatNumber(results.avg_loss, 4) + ' ETH';
    document.getElementById('backtestLargestWin').textContent = formatNumber(results.largest_win, 4) + ' ETH';
    document.getElementById('backtestLargestLoss').textContent = formatNumber(results.largest_loss, 4) + ' ETH';

    // Equity curve chart
    if (results.equity_curve) {
        createChart('backtestEquityChart', {
            type: 'line',
            data: {
                labels: results.equity_curve.map(d => formatDate(d.timestamp)),
                datasets: [{
                    label: 'Portfolio Value',
                    data: results.equity_curve.map(d => d.value),
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
                        ticks: { callback: value => formatNumber(value, 4) + ' ETH' }
                    }
                }
            }
        });
    }

    // Trade log
    if (results.trades) {
        const tbody = document.getElementById('backtestTradeLog');
        tbody.innerHTML = results.trades.map(trade => {
            const pnl = trade.pnl || 0;
            const pnlClass = pnl >= 0 ? 'text-success' : 'text-danger';

            return `
                <tr>
                    <td>${formatDate(trade.entry_time)}</td>
                    <td><span class="badge badge-${trade.side === 'buy' ? 'success' : 'danger'}">${trade.side}</span></td>
                    <td>${formatNumber(trade.entry_price, 6)}</td>
                    <td>${formatNumber(trade.exit_price, 6)}</td>
                    <td class="${pnlClass}">${formatNumber(pnl, 6)}</td>
                    <td class="${pnlClass}">${formatPercent(trade.return)}</td>
                    <td>${trade.duration}</td>
                </tr>
            `;
        }).join('');
    }
}

async function exportBacktestResults() {
    if (!currentBacktestId) {
        showToast('warning', 'No backtest results to export');
        return;
    }

    await exportData('csv', `/api/backtest/results/${currentBacktestId}/export`);
}

// Set default dates
document.addEventListener('DOMContentLoaded', function() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 3); // 3 months ago

    document.getElementById('backtestStartDate').value = startDate.toISOString().split('T')[0];
    document.getElementById('backtestEndDate').value = endDate.toISOString().split('T')[0];
});
