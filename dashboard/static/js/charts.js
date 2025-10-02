// Chart Management and Utilities

class ChartManager {
    constructor() {
        this.charts = {};
        this.defaultOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: this.getThemeColor('text-primary'),
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: this.getThemeColor('bg-secondary'),
                    titleColor: this.getThemeColor('text-primary'),
                    bodyColor: this.getThemeColor('text-secondary'),
                    borderColor: this.getThemeColor('border-color'),
                    borderWidth: 1
                }
            }
        };
    }

    getThemeColor(varName) {
        return getComputedStyle(document.documentElement)
            .getPropertyValue(`--${varName}`).trim();
    }

    createPortfolioChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Portfolio Value',
                    data: data.values,
                    borderColor: this.getThemeColor('accent-primary'),
                    backgroundColor: `${this.getThemeColor('accent-primary')}20`,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            color: this.getThemeColor('text-secondary'),
                            callback: (value) => formatCurrency(value)
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    },
                    x: {
                        ticks: {
                            color: this.getThemeColor('text-secondary')
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createPnLChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Daily P&L',
                    data: data.values,
                    backgroundColor: data.values.map(v => 
                        v >= 0 ? this.getThemeColor('accent-success') : this.getThemeColor('accent-danger')
                    ),
                    borderWidth: 0
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: this.getThemeColor('text-secondary'),
                            callback: (value) => formatCurrency(value)
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    },
                    x: {
                        ticks: {
                            color: this.getThemeColor('text-secondary')
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createWinRateChart(canvasId, wins, losses) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Wins', 'Losses'],
                datasets: [{
                    data: [wins, losses],
                    backgroundColor: [
                        this.getThemeColor('accent-success'),
                        this.getThemeColor('accent-danger')
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                ...this.defaultOptions,
                cutout: '70%',
                plugins: {
                    ...this.defaultOptions.plugins,
                    legend: {
                        display: true,
                        position: 'bottom'
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createVolumeChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Trading Volume',
                    data: data.values,
                    backgroundColor: `${this.getThemeColor('accent-info')}80`,
                    borderColor: this.getThemeColor('accent-info'),
                    borderWidth: 1
                }]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: this.getThemeColor('text-secondary'),
                            callback: (value) => formatCurrency(value)
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    },
                    x: {
                        ticks: {
                            color: this.getThemeColor('text-secondary')
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createPositionsChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: [
                        this.getThemeColor('accent-primary'),
                        this.getThemeColor('accent-success'),
                        this.getThemeColor('accent-warning'),
                        this.getThemeColor('accent-info'),
                        this.getThemeColor('accent-secondary')
                    ],
                    borderWidth: 2,
                    borderColor: this.getThemeColor('bg-primary')
                }]
            },
            options: {
                ...this.defaultOptions,
                plugins: {
                    ...this.defaultOptions.plugins,
                    legend: {
                        display: true,
                        position: 'right'
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    createRiskGauge(canvasId, value, max = 100) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const percentage = (value / max) * 100;
        let color = this.getThemeColor('accent-success');
        
        if (percentage > 70) {
            color = this.getThemeColor('accent-danger');
        } else if (percentage > 40) {
            color = this.getThemeColor('accent-warning');
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [value, max - value],
                    backgroundColor: [color, `${color}20`],
                    borderWidth: 0
                }]
            },
            options: {
                ...this.defaultOptions,
                cutout: '75%',
                circumference: 180,
                rotation: -90,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });

        return this.charts[canvasId];
    }

    createPerformanceChart(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Cumulative P&L',
                        data: data.cumulative_pnl,
                        borderColor: this.getThemeColor('accent-primary'),
                        backgroundColor: `${this.getThemeColor('accent-primary')}20`,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Win Rate',
                        data: data.win_rate,
                        borderColor: this.getThemeColor('accent-success'),
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...this.defaultOptions,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: {
                            color: this.getThemeColor('text-secondary'),
                            callback: (value) => formatCurrency(value)
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: {
                            color: this.getThemeColor('text-secondary'),
                            callback: (value) => `${value}%`
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    },
                    x: {
                        ticks: {
                            color: this.getThemeColor('text-secondary')
                        },
                        grid: {
                            color: this.getThemeColor('chart-grid')
                        }
                    }
                }
            }
        });

        return this.charts[canvasId];
    }

    updateChart(canvasId, newData) {
        const chart = this.charts[canvasId];
        if (!chart) return;

        chart.data.labels = newData.labels;
        chart.data.datasets.forEach((dataset, i) => {
            if (newData.datasets && newData.datasets[i]) {
                dataset.data = newData.datasets[i].data;
            }
        });
        chart.update();
    }

    updateTheme() {
        Object.keys(this.charts).forEach(canvasId => {
            const chart = this.charts[canvasId];
            if (!chart) return;

            // Update colors based on new theme
            if (chart.options.scales) {
                Object.keys(chart.options.scales).forEach(scaleKey => {
                    const scale = chart.options.scales[scaleKey];
                    if (scale.ticks) {
                        scale.ticks.color = this.getThemeColor('text-secondary');
                    }
                    if (scale.grid) {
                        scale.grid.color = this.getThemeColor('chart-grid');
                    }
                });
            }

            if (chart.options.plugins.legend) {
                chart.options.plugins.legend.labels.color = this.getThemeColor('text-primary');
            }

            chart.update();
        });
    }

    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }

    destroyAll() {
        Object.keys(this.charts).forEach(canvasId => {
            this.destroyChart(canvasId);
        });
    }
}

// Initialize chart manager
const chartManager = new ChartManager();

// Export for use in other scripts
window.chartManager = chartManager;