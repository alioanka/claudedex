// dashboard/static/js/analysis.js

document.addEventListener('DOMContentLoaded', function() {
    async function fetchAnalysisData() {
        try {
            const response = await apiGet('/api/analysis');
            if (response.success) {
                renderCharts(response.data);
            }
        } catch (error) {
            console.error('Failed to fetch analysis data:', error);
            showToast('error', 'Failed to load analysis data');
        }
    }

    function renderCharts(data) {
        if (data.strategy_performance) {
            createStrategyPerformanceChart(data.strategy_performance);
        }
        if (data.hourly_profitability) {
            createHourlyProfitabilityChart(data.hourly_profitability);
        }
    }

    // Shared empty-state renderer for canvases in this file. Audit
    // agent 1 #14 — previously both charts rendered blank axes when
    // the underlying datasets were empty.
    function _renderEmptyOverlay(canvasId, message) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(148, 163, 184, 0.9)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '500 13px system-ui, -apple-system, sans-serif';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
        ctx.restore();
    }

    function createStrategyPerformanceChart(data) {
        if (!Array.isArray(data) || data.length === 0) {
            _renderEmptyOverlay('strategyPerformanceChart', 'No strategy performance data yet');
            return;
        }
        // Format strategy names to be user-friendly
        const formatStrategyName = (name) => {
            if (!name || name === 'unknown' || name === 'Unknown') {
                return 'Unspecified';
            }
            return name
                .replace(/_/g, ' ')
                .replace(/\b\w/g, c => c.toUpperCase());
        };

        const chartData = {
            labels: data.map(d => formatStrategyName(d.strategy)),
            datasets: [{
                label: 'Total P&L',
                data: data.map(d => d.total_pnl),
                backgroundColor: data.map(d => d.total_pnl >= 0 ?
                    'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)')
            }]
        };

        createChart('strategyPerformanceChart', {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
    }

    function createHourlyProfitabilityChart(data) {
        if (!Array.isArray(data) || data.length === 0) {
            _renderEmptyOverlay('hourlyProfitabilityChart', 'No hourly P&L data yet');
            return;
        }
        const chartData = {
            labels: data.map(d => `${String(d.hour).padStart(2, '0')}:00`),
            datasets: [{
                label: 'Average P&L',
                data: data.map(d => d.avg_pnl),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
            }]
        };

        createChart('hourlyProfitabilityChart', {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
            }
        });
    }

    fetchAnalysisData();
});
