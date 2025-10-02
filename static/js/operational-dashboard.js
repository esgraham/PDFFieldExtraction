// Operational Dashboard JavaScript

class OperationalDashboard {
    constructor() {
        this.autoRefresh = false;
        this.refreshInterval = null;
        this.charts = {};
        this.alerts = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadOperationalData();
        this.setupOperationalCharts();
        this.startRealTimeUpdates();
    }

    setupEventListeners() {
        // Alerts button
        document.getElementById('alertsBtn')?.addEventListener('click', () => {
            this.showAlertsModal();
        });

        // Auto-refresh toggle
        document.getElementById('autoRefreshBtn')?.addEventListener('click', () => {
            this.toggleAutoRefresh();
        });

        // Table sorting and filtering could be added here
    }

    async loadOperationalData() {
        try {
            // Load main KPI data
            const kpiResponse = await fetch('/api/kpis?days=1&template=operational');
            const kpiData = await kpiResponse.json();
            
            // Load real-time metrics
            const realtimeResponse = await fetch('/api/metrics/realtime');
            const realtimeData = await realtimeResponse.json();
            
            this.updateRealtimeMetrics(realtimeData);
            this.updatePerformanceTable(kpiData.dashboard);
            this.updateExceptionAnalysis(kpiData.dashboard);
            this.updateQueueStatus(kpiData.dashboard);
            this.updateCostBreakdown(kpiData.dashboard);
            this.updateOperationalEfficiency(kpiData.dashboard);
            this.updateHITLOperations(kpiData.dashboard);
            this.updateOperationalCharts(kpiData.dashboard);
            this.checkForAlerts(kpiData.dashboard);
            
        } catch (error) {
            console.error('Error loading operational data:', error);
            this.showError('Failed to load operational dashboard data');
        }
    }

    updateRealtimeMetrics(data) {
        this.updateElement('currentThroughput', data.current_throughput?.toFixed(0) || '--');
        this.updateElement('queueDepth', '24'); // Mock data
        this.updateElement('activeWorkers', '8'); // Mock data
        this.updateElement('errorRate', ((1 - data.current_accuracy) * 100).toFixed(1) || '--');
    }

    updatePerformanceTable(dashboard) {
        // Update performance metrics table
        this.updateElement('perfAccuracy', `${(dashboard.overall_accuracy * 100).toFixed(1)}%`);
        this.updateElement('perfSTPRate', `${(dashboard.stp_rate * 100).toFixed(1)}%`);
        this.updateElement('perfThroughput', `${dashboard.throughput_docs_per_hour.toFixed(0)} docs/hr`);
        this.updateElement('perfLatency', `${dashboard.latency_p95.toFixed(2)}s`);

        // Update status badges
        this.updateStatusBadge('perfAccuracyStatus', dashboard.overall_accuracy, 0.9, 0.8);
        this.updateStatusBadge('perfSTPStatus', dashboard.stp_rate, 0.8, 0.7);
        this.updateStatusBadge('perfThroughputStatus', dashboard.throughput_docs_per_hour, 100, 50);
        this.updateStatusBadge('perfLatencyStatus', dashboard.latency_p95, 10, 30, true); // Inverse logic for latency

        // Update trend indicators (mock data)
        this.updateTrendIndicator('perfAccuracyTrend', 0.02);
        this.updateTrendIndicator('perfSTPTrend', -0.01);
        this.updateTrendIndicator('perfThroughputTrend', 0.05);
        this.updateTrendIndicator('perfLatencyTrend', -0.1);
    }

    updateExceptionAnalysis(dashboard) {
        const container = document.getElementById('exceptionsList');
        if (!container) return;

        if (!dashboard.top_exceptions || dashboard.top_exceptions.length === 0) {
            container.innerHTML = '<div class="text-center text-success">No exceptions in this period ✅</div>';
            return;
        }

        const html = dashboard.top_exceptions.slice(0, 5).map(exc => `
            <div class="exception-item d-flex justify-content-between align-items-center mb-2">
                <div>
                    <strong>${exc.description || exc.type}</strong>
                    <br><small class="text-muted">${exc.percentage.toFixed(1)}% of documents</small>
                </div>
                <span class="badge bg-warning">${exc.count}</span>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    updateQueueStatus(dashboard) {
        // Mock queue data - in real implementation, this would come from the actual queuing system
        this.updateElement('processingQueue', '45');
        this.updateElement('hitlQueue', '12');
        this.updateElement('poisonQueue', dashboard.poison_queue_count?.toString() || '2');
        this.updateElement('retryQueue', '3');
    }

    updateCostBreakdown(dashboard) {
        const tbody = document.getElementById('costBreakdownTable');
        if (!tbody) return;

        const costData = [
            {
                component: 'Compute',
                perDoc: dashboard.compute_cost_per_doc,
                percentage: (dashboard.compute_cost_per_doc / dashboard.cost_per_document * 100),
                monthly: dashboard.compute_cost_per_doc * 30000 // Assume 30k docs/month
            },
            {
                component: 'Storage',
                perDoc: dashboard.storage_cost_per_doc,
                percentage: (dashboard.storage_cost_per_doc / dashboard.cost_per_document * 100),
                monthly: dashboard.storage_cost_per_doc * 30000
            },
            {
                component: 'Review',
                perDoc: dashboard.review_cost_per_doc,
                percentage: (dashboard.review_cost_per_doc / dashboard.cost_per_document * 100),
                monthly: dashboard.review_cost_per_doc * 30000
            }
        ];

        const html = costData.map(item => `
            <tr>
                <td><strong>${item.component}</strong></td>
                <td>$${item.perDoc.toFixed(4)}</td>
                <td>${item.percentage.toFixed(1)}%</td>
                <td>$${item.monthly.toFixed(2)}</td>
            </tr>
        `).join('');

        tbody.innerHTML = html;
    }

    updateOperationalEfficiency(dashboard) {
        // Update progress bars
        const reprocessRate = dashboard.reprocess_rate * 100;
        const utilizationRate = 85; // Mock data
        const slaRate = dashboard.sla_adherence_rate * 100;

        this.updateProgressBar('reprocessProgress', reprocessRate, 'warning');
        this.updateProgressBar('utilizationProgress', utilizationRate, 'info');
        this.updateProgressBar('slaProgress', slaRate, 'success');

        this.updateElement('reprocessRate', `${reprocessRate.toFixed(1)}%`);
        this.updateElement('utilizationRate', `${utilizationRate.toFixed(1)}%`);
        this.updateElement('slaRate', `${slaRate.toFixed(1)}%`);
    }

    updateHITLOperations(dashboard) {
        // Update HITL metrics table
        this.updateElement('hitlFirstReview', `${dashboard.time_to_first_review_avg.toFixed(1)} min`);
        this.updateElement('hitlResolution', `${dashboard.time_to_resolution_avg.toFixed(1)} min`);
        this.updateElement('hitlRequired', `${((1 - dashboard.stp_rate) * 100).toFixed(1)}%`);

        // Mock today and 7-day averages
        this.updateElement('hitlFirstReviewToday', `${(dashboard.time_to_first_review_avg * 0.9).toFixed(1)} min`);
        this.updateElement('hitlResolutionToday', `${(dashboard.time_to_resolution_avg * 1.1).toFixed(1)} min`);
        this.updateElement('hitlRequiredToday', `${((1 - dashboard.stp_rate) * 100 * 0.95).toFixed(1)}%`);

        this.updateElement('hitlFirstReview7d', `${(dashboard.time_to_first_review_avg * 1.05).toFixed(1)} min`);
        this.updateElement('hitlResolution7d', `${(dashboard.time_to_resolution_avg * 0.95).toFixed(1)} min`);
        this.updateElement('hitlRequired7d', `${((1 - dashboard.stp_rate) * 100 * 1.02).toFixed(1)}%`);

        // Update review queue status
        this.updateElement('pendingReview', '18');
        this.updateElement('inProgressReview', '7');
        this.updateElement('completedToday', '34');
    }

    setupOperationalCharts() {
        this.setupExceptionTypesChart();
        this.setupQueueTrendChart();
        this.setupCostDistributionChart();
    }

    setupExceptionTypesChart() {
        const ctx = document.getElementById('exceptionTypesChart');
        if (!ctx) return;

        this.charts.exceptionTypes = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#dc3545',
                        '#ffc107',
                        '#fd7e14',
                        '#6f42c1',
                        '#20c997'
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

    setupQueueTrendChart() {
        const ctx = document.getElementById('queueTrendChart');
        if (!ctx) return;

        this.charts.queueTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Queue Depth',
                    data: [],
                    borderColor: '#0dcaf0',
                    backgroundColor: 'rgba(13, 202, 240, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    setupCostDistributionChart() {
        const ctx = document.getElementById('costDistributionChart');
        if (!ctx) return;

        this.charts.costDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Compute', 'Storage', 'Review'],
                datasets: [{
                    data: [],
                    backgroundColor: ['#0d6efd', '#198754', '#ffc107']
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

    updateOperationalCharts(dashboard) {
        // Update exception types chart
        if (this.charts.exceptionTypes && dashboard.top_exceptions) {
            const labels = dashboard.top_exceptions.slice(0, 5).map(exc => exc.description || exc.type);
            const data = dashboard.top_exceptions.slice(0, 5).map(exc => exc.count);
            
            this.charts.exceptionTypes.data.labels = labels;
            this.charts.exceptionTypes.data.datasets[0].data = data;
            this.charts.exceptionTypes.update();
        }

        // Update queue trend (mock data)
        if (this.charts.queueTrend) {
            const labels = this.generateTimeLabels(12); // Last 12 hours
            const queueData = this.generateQueueTrendData(12);
            
            this.charts.queueTrend.data.labels = labels;
            this.charts.queueTrend.data.datasets[0].data = queueData;
            this.charts.queueTrend.update();
        }

        // Update cost distribution
        if (this.charts.costDistribution) {
            const costData = [
                dashboard.compute_cost_per_doc,
                dashboard.storage_cost_per_doc,
                dashboard.review_cost_per_doc
            ];
            this.charts.costDistribution.data.datasets[0].data = costData;
            this.charts.costDistribution.update();
        }
    }

    checkForAlerts(dashboard) {
        this.alerts = [];

        // Check for critical thresholds
        if (dashboard.overall_accuracy < 0.85) {
            this.alerts.push({
                type: 'critical',
                message: `Accuracy dropped to ${(dashboard.overall_accuracy * 100).toFixed(1)}%`,
                timestamp: new Date()
            });
        }

        if (dashboard.stp_rate < 0.7) {
            this.alerts.push({
                type: 'warning',
                message: `STP rate is low: ${(dashboard.stp_rate * 100).toFixed(1)}%`,
                timestamp: new Date()
            });
        }

        if (dashboard.poison_queue_count > 5) {
            this.alerts.push({
                type: 'warning',
                message: `Poison queue has ${dashboard.poison_queue_count} documents`,
                timestamp: new Date()
            });
        }

        // Update alert badge
        const alertBadge = document.getElementById('alertCount');
        if (alertBadge) {
            alertBadge.textContent = this.alerts.length;
            alertBadge.style.display = this.alerts.length > 0 ? 'inline' : 'none';
        }
    }

    showAlertsModal() {
        const modal = new bootstrap.Modal(document.getElementById('alertsModal'));
        const alertsList = document.getElementById('alertsList');
        
        if (this.alerts.length === 0) {
            alertsList.innerHTML = '<div class="text-center text-success"><i class="fas fa-check-circle me-2"></i>No active alerts</div>';
        } else {
            const html = this.alerts.map(alert => {
                const iconClass = alert.type === 'critical' ? 'exclamation-triangle text-danger' : 'exclamation-circle text-warning';
                const bgClass = alert.type === 'critical' ? 'alert-danger' : 'alert-warning';
                
                return `
                    <div class="alert ${bgClass} alert-dismissible">
                        <i class="fas fa-${iconClass} me-2"></i>
                        <strong>${alert.message}</strong>
                        <br><small class="text-muted">${alert.timestamp.toLocaleString()}</small>
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                `;
            }).join('');
            
            alertsList.innerHTML = html;
        }
        
        modal.show();
    }

    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        const btn = document.getElementById('autoRefreshBtn');
        
        if (this.autoRefresh) {
            btn.classList.remove('btn-outline-light');
            btn.classList.add('btn-success');
            btn.innerHTML = '<i class="fas fa-sync-alt fa-spin me-1"></i>Auto-Refresh ON';
        } else {
            btn.classList.remove('btn-success');
            btn.classList.add('btn-outline-light');
            btn.innerHTML = '<i class="fas fa-sync me-1"></i>Auto-Refresh';
        }
    }

    startRealTimeUpdates() {
        // Update every 30 seconds
        setInterval(() => {
            if (this.autoRefresh) {
                this.loadOperationalData();
            }
        }, 30000);
    }

    // Utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateStatusBadge(id, value, goodThreshold, warningThreshold, inverse = false) {
        const element = document.getElementById(id);
        if (!element) return;

        let className, text;
        
        if (inverse) {
            // For metrics where lower is better (like latency)
            if (value <= goodThreshold) {
                className = 'bg-success';
                text = 'Good';
            } else if (value <= warningThreshold) {
                className = 'bg-warning';
                text = 'Monitor';
            } else {
                className = 'bg-danger';
                text = 'Critical';
            }
        } else {
            // For metrics where higher is better
            if (value >= goodThreshold) {
                className = 'bg-success';
                text = 'Good';
            } else if (value >= warningThreshold) {
                className = 'bg-warning';
                text = 'Monitor';
            } else {
                className = 'bg-danger';
                text = 'Critical';
            }
        }

        element.className = `badge ${className}`;
        element.textContent = text;
    }

    updateTrendIndicator(id, change) {
        const element = document.getElementById(id);
        if (!element) return;

        const icon = change > 0 ? 'fa-arrow-up' : change < 0 ? 'fa-arrow-down' : 'fa-minus';
        const color = change > 0 ? 'text-success' : change < 0 ? 'text-danger' : 'text-muted';
        const sign = change > 0 ? '+' : '';
        
        element.innerHTML = `<i class="fas ${icon} ${color} me-1"></i>${sign}${(change * 100).toFixed(1)}%`;
    }

    updateProgressBar(id, value, colorClass) {
        const element = document.getElementById(id);
        if (!element) return;

        element.style.width = `${Math.min(100, Math.max(0, value))}%`;
        element.className = `progress-bar bg-${colorClass}`;
    }

    generateTimeLabels(hours) {
        const labels = [];
        for (let i = hours - 1; i >= 0; i--) {
            const date = new Date();
            date.setHours(date.getHours() - i);
            labels.push(date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
        }
        return labels;
    }

    generateQueueTrendData(hours) {
        const data = [];
        for (let i = 0; i < hours; i++) {
            // Generate realistic queue depth data
            const baseValue = 25;
            const variation = Math.random() * 20 - 10; // ±10
            data.push(Math.max(0, baseValue + variation));
        }
        return data;
    }

    showError(message) {
        console.error(message);
    }
}

// Initialize operational dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.operationalDashboard = new OperationalDashboard();
});