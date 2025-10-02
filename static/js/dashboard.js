// Main Dashboard JavaScript

class Dashboard {
    constructor() {
        this.refreshInterval = null;
        this.autoRefresh = false;
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.setupCharts();
    }

    setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn')?.addEventListener('click', () => {
            this.loadDashboardData();
        });

        // Time range selection
        document.querySelectorAll('input[name="timeRange"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.loadDashboardData(parseInt(e.target.value));
            });
        });

        // Auto-refresh toggle
        const autoRefreshBtn = document.getElementById('autoRefreshBtn');
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => {
                this.toggleAutoRefresh();
            });
        }
    }

    async loadInitialData() {
        await this.loadDashboardData();
        this.updateLastRefreshTime();
    }

    async loadDashboardData(days = 7) {
        try {
            const response = await fetch(`/api/kpis?days=${days}`);
            const data = await response.json();
            
            this.updateKPICards(data.dashboard);
            this.updatePerformanceMetrics(data.dashboard);
            this.updateExceptions(data.dashboard);
            this.updateRecentActivity(days);
            this.updateCharts(data.dashboard);
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    updateKPICards(dashboard) {
        // Update main KPI cards
        this.updateElement('overallAccuracy', `${(dashboard.overall_accuracy * 100).toFixed(1)}%`);
        this.updateElement('stpRate', `${(dashboard.stp_rate * 100).toFixed(1)}%`);
        this.updateElement('throughput', `${dashboard.throughput_docs_per_hour.toFixed(0)}`);
        this.updateElement('costPerDoc', `$${dashboard.cost_per_document.toFixed(4)}`);
    }

    updatePerformanceMetrics(dashboard) {
        // Update performance metrics
        this.updateElement('latencyP50', `${dashboard.latency_p50.toFixed(2)}s`);
        this.updateElement('latencyP95', `${dashboard.latency_p95.toFixed(2)}s`);
        this.updateElement('slaAdherence', `${(dashboard.sla_adherence_rate * 100).toFixed(1)}%`);
        this.updateElement('timeToReview', `${dashboard.time_to_first_review_avg.toFixed(1)} min`);
        this.updateElement('timeToResolution', `${dashboard.time_to_resolution_avg.toFixed(1)} min`);
        this.updateElement('exceptionRate', `${(dashboard.exception_rate * 100).toFixed(1)}%`);
    }

    updateExceptions(dashboard) {
        const container = document.getElementById('topExceptions');
        if (!container) return;

        if (!dashboard.top_exceptions || dashboard.top_exceptions.length === 0) {
            container.innerHTML = '<div class="text-center text-success">No exceptions to report ✅</div>';
            return;
        }

        const html = dashboard.top_exceptions.slice(0, 5).map(exc => `
            <div class="exception-item">
                <div class="d-flex justify-content-between">
                    <strong>${exc.description || exc.type}</strong>
                    <span class="badge bg-warning">${exc.count}</span>
                </div>
                <small class="text-muted">${exc.percentage.toFixed(1)}% of total documents</small>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    async updateRecentActivity(days = 7) {
        try {
            const response = await fetch(`/api/metrics/documents?days=${days}&limit=10`);
            const data = await response.json();
            
            const tbody = document.querySelector('#recentActivityTable tbody');
            if (!tbody) return;

            if (!data.metrics || data.metrics.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No recent activity</td></tr>';
                return;
            }

            const html = data.metrics.map(metric => {
                const status = this.getStatusBadge(metric.processing_status);
                const cost = metric.cost_breakdown ? 
                    Object.values(JSON.parse(metric.cost_breakdown)).reduce((a, b) => a + b, 0) : 0;
                
                return `
                    <tr>
                        <td><code>${metric.document_id}</code></td>
                        <td><span class="badge bg-secondary">${metric.document_type}</span></td>
                        <td>${status}</td>
                        <td>${metric.total_processing_time.toFixed(2)}s</td>
                        <td>$${cost.toFixed(4)}</td>
                        <td>${new Date(metric.start_time).toLocaleString()}</td>
                    </tr>
                `;
            }).join('');

            tbody.innerHTML = html;
            
        } catch (error) {
            console.error('Error loading recent activity:', error);
        }
    }

    setupCharts() {
        this.setupTrendsChart();
        this.setupDocumentTypesChart();
    }

    setupTrendsChart() {
        const ctx = document.getElementById('trendsChart');
        if (!ctx) return;

        this.charts.trends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Accuracy',
                    data: [],
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    tension: 0.4
                }, {
                    label: 'STP Rate',
                    data: [],
                    borderColor: '#0dcaf0',
                    backgroundColor: 'rgba(13, 202, 240, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    setupDocumentTypesChart() {
        const ctx = document.getElementById('documentTypesChart');
        if (!ctx) return;

        this.charts.documentTypes = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#0d6efd',
                        '#198754',
                        '#ffc107',
                        '#dc3545',
                        '#6f42c1',
                        '#fd7e14'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    updateCharts(dashboard) {
        // Update trends chart with sample data
        if (this.charts.trends) {
            const labels = this.generateDateLabels(7);
            const accuracyData = this.generateTrendData(dashboard.overall_accuracy, 7);
            const stpData = this.generateTrendData(dashboard.stp_rate, 7);

            this.charts.trends.data.labels = labels;
            this.charts.trends.data.datasets[0].data = accuracyData;
            this.charts.trends.data.datasets[1].data = stpData;
            this.charts.trends.update();
        }

        // Update document types chart
        if (this.charts.documentTypes && dashboard.document_type_metrics) {
            const types = Object.keys(dashboard.document_type_metrics);
            const counts = types.map(type => dashboard.document_type_metrics[type].count || 0);

            this.charts.documentTypes.data.labels = types.map(type => type.replace('_', ' ').toUpperCase());
            this.charts.documentTypes.data.datasets[0].data = counts;
            this.charts.documentTypes.update();
        }
    }

    generateDateLabels(days) {
        const labels = [];
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString());
        }
        return labels;
    }

    generateTrendData(baseValue, days) {
        const data = [];
        for (let i = 0; i < days; i++) {
            // Generate realistic trend data around the base value
            const variation = (Math.random() - 0.5) * 0.1; // ±5% variation
            data.push(Math.max(0, Math.min(1, baseValue + variation)));
        }
        return data;
    }

    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        const btn = document.getElementById('autoRefreshBtn');
        
        if (this.autoRefresh) {
            btn.classList.remove('btn-outline-light');
            btn.classList.add('btn-success');
            btn.innerHTML = '<i class="fas fa-sync-alt fa-spin me-1"></i>Auto-Refresh';
            this.refreshInterval = setInterval(() => {
                this.loadDashboardData();
            }, 30000); // Refresh every 30 seconds
        } else {
            btn.classList.remove('btn-success');
            btn.classList.add('btn-outline-light');
            btn.innerHTML = '<i class="fas fa-sync me-1"></i>Auto-Refresh';
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            element.classList.add('fade-in');
        }
    }

    updateLastRefreshTime() {
        const element = document.getElementById('lastUpdated');
        if (element) {
            element.textContent = new Date().toLocaleString();
        }
    }

    getStatusBadge(status) {
        const statusMap = {
            'success': '<span class="badge bg-success">Success</span>',
            'hitl_required': '<span class="badge bg-warning">HITL Required</span>',
            'failed': '<span class="badge bg-danger">Failed</span>',
            'reprocessing': '<span class="badge bg-info">Reprocessing</span>'
        };
        return statusMap[status] || '<span class="badge bg-secondary">Unknown</span>';
    }

    showError(message) {
        // Show error toast or notification
        console.error(message);
        
        // Update system status
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>Error';
            statusElement.className = 'badge bg-danger me-2';
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            const statusElement = document.getElementById('systemStatus');
            if (statusElement) {
                if (health.status === 'healthy') {
                    statusElement.innerHTML = '<i class="fas fa-check-circle me-1"></i>Online';
                    statusElement.className = 'badge bg-success me-2';
                } else {
                    statusElement.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>Issues';
                    statusElement.className = 'badge bg-warning me-2';
                }
            }
            
        } catch (error) {
            this.showError('Health check failed');
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
    
    // Check health every minute
    setInterval(() => {
        window.dashboard.checkHealth();
    }, 60000);
    
    // Initial health check
    window.dashboard.checkHealth();
});