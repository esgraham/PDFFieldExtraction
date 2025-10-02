// Executive Dashboard JavaScript

class ExecutiveDashboard {
    constructor() {
        this.currentAudience = 'internal';
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadExecutiveData();
        this.setupExecutiveCharts();
    }

    setupEventListeners() {
        // Generate summary button
        document.getElementById('generateSummary')?.addEventListener('click', () => {
            this.generateSummaryReport();
        });

        // Audience selection
        document.getElementById('audienceSelect')?.addEventListener('change', (e) => {
            this.currentAudience = e.target.value;
        });

        // Summary modal buttons
        document.getElementById('markdownBtn')?.addEventListener('click', () => {
            this.loadMarkdownSummary();
        });

        document.getElementById('llmBtn')?.addEventListener('click', () => {
            this.loadLLMSummary();
        });

        document.getElementById('copyBtn')?.addEventListener('click', () => {
            this.copySummaryToClipboard();
        });
    }

    async loadExecutiveData() {
        try {
            const response = await fetch('/api/kpis?days=7&template=executive');
            const data = await response.json();
            
            this.updateExecutiveKPIs(data.dashboard);
            this.updateExecutiveSummary(data.dashboard);
            this.updateQualityMetrics(data.dashboard);
            this.updatePerformanceMetrics(data.dashboard);
            this.updateCostMetrics(data.dashboard);
            this.updateTopIssues(data.dashboard);
            this.updateDocumentTypeTable(data.dashboard);
            this.updateExecutiveCharts(data.dashboard);
            
        } catch (error) {
            console.error('Error loading executive data:', error);
            this.showError('Failed to load executive dashboard data');
        }
    }

    updateExecutiveKPIs(dashboard) {
        // Update header KPIs
        this.updateElement('execAccuracy', `${(dashboard.overall_accuracy * 100).toFixed(1)}%`);
        this.updateElement('execSTP', `${(dashboard.stp_rate * 100).toFixed(1)}%`);
        this.updateElement('execThroughput', `${dashboard.throughput_docs_per_hour.toFixed(0)}`);
        this.updateElement('execCost', `$${dashboard.cost_per_document.toFixed(3)}`);
    }

    updateExecutiveSummary(dashboard) {
        const summaryElement = document.getElementById('executiveSummaryText');
        if (!summaryElement) return;

        const summary = `
            <div class="executive-highlights">
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-primary">System Performance</h6>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check-circle text-success me-2"></i>Processing ${dashboard.throughput_docs_per_hour.toFixed(0)} documents per hour</li>
                            <li><i class="fas fa-bullseye text-info me-2"></i>Achieving ${(dashboard.overall_accuracy * 100).toFixed(1)}% accuracy rate</li>
                            <li><i class="fas fa-forward text-warning me-2"></i>${(dashboard.stp_rate * 100).toFixed(1)}% straight-through processing</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-primary">Business Impact</h6>
                        <ul class="list-unstyled">
                            <li><i class="fas fa-dollar-sign text-success me-2"></i>Cost per document: $${dashboard.cost_per_document.toFixed(4)}</li>
                            <li><i class="fas fa-clock text-info me-2"></i>Average processing: ${dashboard.latency_p50.toFixed(1)}s</li>
                            <li><i class="fas fa-chart-line text-warning me-2"></i>SLA compliance: ${(dashboard.sla_adherence_rate * 100).toFixed(1)}%</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;

        summaryElement.innerHTML = summary;
    }

    updateQualityMetrics(dashboard) {
        this.updateElement('qualityAccuracy', `${(dashboard.overall_accuracy * 100).toFixed(1)}%`);
        this.updateElement('qualitySTP', `${(dashboard.stp_rate * 100).toFixed(1)}%`);
        this.updateElement('qualityExceptions', `${(dashboard.exception_rate * 100).toFixed(1)}%`);
    }

    updatePerformanceMetrics(dashboard) {
        this.updateElement('perfLatency', `${dashboard.latency_p50.toFixed(2)}s`);
        this.updateElement('perfP95', `${dashboard.latency_p95.toFixed(2)}s`);
        this.updateElement('perfSLA', `${(dashboard.sla_adherence_rate * 100).toFixed(1)}%`);
    }

    updateCostMetrics(dashboard) {
        this.updateElement('costPerDoc', `$${dashboard.cost_per_document.toFixed(4)}`);
        this.updateElement('costReprocess', `${(dashboard.reprocess_rate * 100).toFixed(1)}%`);
        this.updateElement('costEfficiency', `${((1 - dashboard.exception_rate) * 100).toFixed(1)}%`);
    }

    updateTopIssues(dashboard) {
        const container = document.getElementById('topIssues');
        if (!container) return;

        if (!dashboard.top_exceptions || dashboard.top_exceptions.length === 0) {
            container.innerHTML = '<div class="text-center text-success"><i class="fas fa-check-circle me-2"></i>No critical issues to report</div>';
            return;
        }

        const html = dashboard.top_exceptions.slice(0, 3).map((exc, index) => {
            const priority = index === 0 ? 'high' : index === 1 ? 'medium' : 'low';
            const icon = index === 0 ? 'exclamation-triangle' : index === 1 ? 'exclamation-circle' : 'info-circle';
            const color = index === 0 ? 'danger' : index === 1 ? 'warning' : 'info';
            
            return `
                <div class="issue-item alert alert-${color} alert-dismissible">
                    <div class="d-flex align-items-start">
                        <i class="fas fa-${icon} me-3 mt-1"></i>
                        <div class="flex-grow-1">
                            <strong>${exc.description || exc.type}</strong>
                            <p class="mb-1">${exc.count} occurrences (${exc.percentage.toFixed(1)}% of documents)</p>
                            <small class="text-muted">Priority: ${priority.toUpperCase()}</small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    updateDocumentTypeTable(dashboard) {
        const tbody = document.querySelector('#documentTypeTable tbody');
        if (!tbody || !dashboard.document_type_metrics) return;

        const html = Object.entries(dashboard.document_type_metrics).map(([type, metrics]) => {
            const statusClass = metrics.success_rate > 0.9 ? 'success' : metrics.success_rate > 0.8 ? 'warning' : 'danger';
            const statusText = metrics.success_rate > 0.9 ? 'Excellent' : metrics.success_rate > 0.8 ? 'Good' : 'Needs Attention';
            
            return `
                <tr>
                    <td><span class="badge bg-secondary">${type.replace('_', ' ').toUpperCase()}</span></td>
                    <td>${metrics.count}</td>
                    <td>${(metrics.success_rate * 100).toFixed(1)}%</td>
                    <td>${(metrics.stp_rate * 100).toFixed(1)}%</td>
                    <td>${metrics.avg_processing_time.toFixed(2)}s</td>
                    <td><span class="badge bg-${statusClass}">${statusText}</span></td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = html;
    }

    setupExecutiveCharts() {
        this.setupQualityTrendChart();
        this.setupPerformanceTrendChart();
        this.setupCostBreakdownChart();
    }

    setupQualityTrendChart() {
        const ctx = document.getElementById('qualityTrendChart');
        if (!ctx) return;

        this.charts.qualityTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Accuracy',
                    data: [],
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    tension: 0.4,
                    fill: true
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
                        display: false
                    }
                }
            }
        });
    }

    setupPerformanceTrendChart() {
        const ctx = document.getElementById('performanceTrendChart');
        if (!ctx) return;

        this.charts.performanceTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (s)',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
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

    setupCostBreakdownChart() {
        const ctx = document.getElementById('costBreakdownChart');
        if (!ctx) return;

        this.charts.costBreakdown = new Chart(ctx, {
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
                        display: false
                    }
                }
            }
        });
    }

    updateExecutiveCharts(dashboard) {
        const labels = this.generateDateLabels(7);

        // Update quality trend
        if (this.charts.qualityTrend) {
            const qualityData = this.generateTrendData(dashboard.overall_accuracy, 7);
            this.charts.qualityTrend.data.labels = labels;
            this.charts.qualityTrend.data.datasets[0].data = qualityData;
            this.charts.qualityTrend.update();
        }

        // Update performance trend
        if (this.charts.performanceTrend) {
            const latencyData = this.generateTrendData(dashboard.latency_p50, 7, 0.5);
            this.charts.performanceTrend.data.labels = labels;
            this.charts.performanceTrend.data.datasets[0].data = latencyData;
            this.charts.performanceTrend.update();
        }

        // Update cost breakdown
        if (this.charts.costBreakdown) {
            const costData = [
                dashboard.compute_cost_per_doc,
                dashboard.storage_cost_per_doc,
                dashboard.review_cost_per_doc
            ];
            this.charts.costBreakdown.data.datasets[0].data = costData;
            this.charts.costBreakdown.update();
        }
    }

    async generateSummaryReport() {
        const modal = new bootstrap.Modal(document.getElementById('summaryModal'));
        modal.show();
        
        // Load markdown summary by default
        await this.loadMarkdownSummary();
    }

    async loadMarkdownSummary() {
        const contentDiv = document.getElementById('summaryContent');
        if (!contentDiv) return;

        contentDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div></div>';

        try {
            const response = await fetch(`/api/summary/executive?days=7&audience=${this.currentAudience}`);
            const data = await response.json();
            
            contentDiv.innerHTML = `<pre class="summary-content">${data.summary}</pre>`;
            
        } catch (error) {
            console.error('Error loading markdown summary:', error);
            contentDiv.innerHTML = '<div class="alert alert-danger">Failed to generate summary</div>';
        }
    }

    async loadLLMSummary() {
        const contentDiv = document.getElementById('summaryContent');
        if (!contentDiv) return;

        contentDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-success" role="status"></div></div>';

        try {
            const response = await fetch(`/api/llm-summary?days=7&template=executive&audience=${this.currentAudience}`);
            const data = await response.json();
            
            contentDiv.innerHTML = `<div class="summary-content">${data.summary.replace(/\n/g, '<br>')}</div>`;
            
        } catch (error) {
            console.error('Error loading LLM summary:', error);
            contentDiv.innerHTML = '<div class="alert alert-warning">LLM enhancement not available</div>';
        }
    }

    copySummaryToClipboard() {
        const contentDiv = document.getElementById('summaryContent');
        if (!contentDiv) return;

        const text = contentDiv.textContent || contentDiv.innerText;
        navigator.clipboard.writeText(text).then(() => {
            // Show success feedback
            const copyBtn = document.getElementById('copyBtn');
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
            copyBtn.classList.add('btn-success');
            
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
                copyBtn.classList.remove('btn-success');
            }, 2000);
        });
    }

    generateDateLabels(days) {
        const labels = [];
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        }
        return labels;
    }

    generateTrendData(baseValue, days, scale = 0.1) {
        const data = [];
        for (let i = 0; i < days; i++) {
            const variation = (Math.random() - 0.5) * scale;
            data.push(Math.max(0, baseValue + variation));
        }
        return data;
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            element.classList.add('fade-in');
        }
    }

    showError(message) {
        console.error(message);
        // Could add toast notification here
    }
}

// Initialize executive dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.executiveDashboard = new ExecutiveDashboard();
});