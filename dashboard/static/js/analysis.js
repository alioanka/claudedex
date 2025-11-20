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

    function createStrategyPerformanceChart(data) {
        const chartData = {
            labels: data.map(d => d.strategy),
            datasets: [{
                label: 'Total P&L',
                data: data.map(d => d.total_pnl),
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
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
