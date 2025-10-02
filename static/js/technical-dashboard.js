// Technical Dashboard JavaScript

class TechnicalDashboard {
    constructor() {
        this.autoRefresh = false;
        this.refreshInterval = null;
        this.charts = {};
        this.systemStats = {};
        this.logs = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTechnicalData();
        this.setupTechnicalCharts();
        this.startRealTimeMonitoring();
    }

    setupEventListeners() {
        // Log level filter
        document.getElementById('logLevelFilter')?.addEventListener('change', (e) => {
            this.filterLogs(e.target.value);
        });

        // Auto-refresh toggle
        document.getElementById('autoRefreshBtn')?.addEventListener('click', () => {
            this.toggleAutoRefresh();
        });

        // System health refresh
        document.getElementById('refreshSystemBtn')?.addEventListener('click', () => {
            this.loadSystemHealth();
        });

        // Log refresh
        document.getElementById('refreshLogsBtn')?.addEventListener('click', () => {
            this.loadLogs();
        });

        // Database operations
        document.getElementById('dbHealthBtn')?.addEventListener('click', () => {
            this.showDatabaseHealth();
        });

        // Cache operations
        document.getElementById('cacheStatsBtn')?.addEventListener('click', () => {
            this.showCacheStats();
        });
    }

    async loadTechnicalData() {
        try {
            // Load system health
            await this.loadSystemHealth();
            
            // Load performance metrics
            await this.loadPerformanceMetrics();
            
            // Load logs
            await this.loadLogs();
            
            // Load API metrics
            await this.loadAPIMetrics();
            
            // Update technical charts
            this.updateTechnicalCharts();
            
        } catch (error) {
            console.error('Error loading technical data:', error);
            this.showError('Failed to load technical dashboard data');
        }
    }

    async loadSystemHealth() {
        try {
            // Mock system health data - in real implementation, this would come from system monitoring
            const systemHealth = {
                cpu_usage: Math.random() * 30 + 20, // 20-50%
                memory_usage: Math.random() * 20 + 40, // 40-60%
                disk_usage: Math.random() * 10 + 15, // 15-25%
                network_io: Math.random() * 100 + 50, // MB/s
                active_connections: Math.floor(Math.random() * 50 + 100),
                uptime: '7d 14h 23m',
                last_restart: '2024-01-15 09:30:00',
                status: 'healthy'
            };

            this.updateSystemHealth(systemHealth);
        } catch (error) {
            console.error('Error loading system health:', error);
        }
    }

    async loadPerformanceMetrics() {
        try {
            const response = await fetch('/api/kpis?days=1&template=technical');
            const data = await response.json();
            
            this.updatePerformanceMetrics(data.dashboard);
        } catch (error) {
            console.error('Error loading performance metrics:', error);
        }
    }

    async loadLogs() {
        try {
            // Mock log data - in real implementation, this would come from log aggregation system
            const logs = this.generateMockLogs(50);
            this.logs = logs;
            this.displayLogs(logs);
        } catch (error) {
            console.error('Error loading logs:', error);
        }
    }

    async loadAPIMetrics() {
        try {
            // Mock API metrics
            const apiMetrics = {
                endpoints: [
                    { path: '/api/upload', requests: 1247, avg_response_time: 156, error_rate: 0.02 },
                    { path: '/api/process', requests: 856, avg_response_time: 2340, error_rate: 0.01 },
                    { path: '/api/extract', requests: 632, avg_response_time: 890, error_rate: 0.03 },
                    { path: '/api/validate', requests: 445, avg_response_time: 234, error_rate: 0.01 },
                    { path: '/api/export', requests: 223, avg_response_time: 567, error_rate: 0.005 }
                ],
                total_requests: 3403,
                avg_response_time: 834,
                error_rate: 0.018
            };

            this.updateAPIMetrics(apiMetrics);
        } catch (error) {
            console.error('Error loading API metrics:', error);
        }
    }

    updateSystemHealth(health) {
        // Update system resource usage
        this.updateProgressBar('cpuUsage', health.cpu_usage, this.getUsageColor(health.cpu_usage, 70, 90));
        this.updateProgressBar('memoryUsage', health.memory_usage, this.getUsageColor(health.memory_usage, 80, 90));
        this.updateProgressBar('diskUsage', health.disk_usage, this.getUsageColor(health.disk_usage, 85, 95));

        this.updateElement('cpuPercent', `${health.cpu_usage.toFixed(1)}%`);
        this.updateElement('memoryPercent', `${health.memory_usage.toFixed(1)}%`);
        this.updateElement('diskPercent', `${health.disk_usage.toFixed(1)}%`);

        // Update network and connections
        this.updateElement('networkIO', `${health.network_io.toFixed(1)} MB/s`);
        this.updateElement('activeConnections', health.active_connections.toString());

        // Update system info
        this.updateElement('systemUptime', health.uptime);
        this.updateElement('lastRestart', health.last_restart);

        // Update system status badge
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            const statusClass = health.status === 'healthy' ? 'bg-success' : 
                               health.status === 'warning' ? 'bg-warning' : 'bg-danger';
            statusElement.className = `badge ${statusClass}`;
            statusElement.textContent = health.status.toUpperCase();
        }
    }

    updatePerformanceMetrics(dashboard) {
        // Document processing performance
        this.updateElement('processingThroughput', `${dashboard.throughput_docs_per_hour.toFixed(0)} docs/hr`);
        this.updateElement('averageLatency', `${dashboard.latency_p95.toFixed(2)}s`);
        this.updateElement('queueLatency', `${(dashboard.time_to_first_review_avg * 60).toFixed(0)}s`);
        this.updateElement('extractionRate', `${(dashboard.overall_accuracy * 100).toFixed(1)}%`);

        // Error rates
        const errorRate = (1 - dashboard.overall_accuracy) * 100;
        this.updateElement('errorRate', `${errorRate.toFixed(2)}%`);
        this.updateElement('retryRate', `${(dashboard.reprocess_rate * 100).toFixed(1)}%`);
        this.updateElement('timeoutRate', '0.05%'); // Mock data
        this.updateElement('crashRate', '0.01%'); // Mock data

        // Resource utilization
        this.updateElement('azureApiCalls', Math.floor(dashboard.throughput_docs_per_hour * 24).toString());
        this.updateElement('storageUsed', '2.4 TB'); // Mock data
        this.updateElement('computeHours', '156.7 hrs'); // Mock data
        this.updateElement('bandwidthUsed', '445 GB'); // Mock data
    }

    updateAPIMetrics(metrics) {
        const tbody = document.getElementById('apiMetricsTable');
        if (!tbody) return;

        const html = metrics.endpoints.map(endpoint => {
            const errorRatePercent = (endpoint.error_rate * 100).toFixed(2);
            const statusClass = endpoint.error_rate > 0.05 ? 'text-danger' : 
                               endpoint.error_rate > 0.02 ? 'text-warning' : 'text-success';

            return `
                <tr>
                    <td><code>${endpoint.path}</code></td>
                    <td>${endpoint.requests.toLocaleString()}</td>
                    <td>${endpoint.avg_response_time}ms</td>
                    <td><span class="${statusClass}">${errorRatePercent}%</span></td>
                    <td>
                        <div class="progress" style="height: 6px;">
                            <div class="progress-bar bg-primary" style="width: ${(endpoint.requests / Math.max(...metrics.endpoints.map(e => e.requests))) * 100}%"></div>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');

        tbody.innerHTML = html;

        // Update summary metrics
        this.updateElement('totalRequests', metrics.total_requests.toLocaleString());
        this.updateElement('avgResponseTime', `${metrics.avg_response_time}ms`);
        this.updateElement('overallErrorRate', `${(metrics.error_rate * 100).toFixed(2)}%`);
    }

    generateMockLogs(count) {
        const levels = ['INFO', 'WARN', 'ERROR', 'DEBUG'];
        const components = ['DocumentProcessor', 'FieldExtractor', 'APIHandler', 'DatabaseManager', 'QueueManager'];
        const messages = [
            'Document processing completed successfully',
            'Field extraction accuracy below threshold',
            'Azure API rate limit encountered',
            'Database connection timeout',
            'Queue depth exceeding normal levels',
            'Cache miss for document template',
            'Validation rules updated',
            'Background job completed',
            'Configuration reload triggered',
            'Health check passed'
        ];

        const logs = [];
        for (let i = 0; i < count; i++) {
            const timestamp = new Date(Date.now() - Math.random() * 86400000); // Last 24 hours
            const level = levels[Math.floor(Math.random() * levels.length)];
            const component = components[Math.floor(Math.random() * components.length)];
            const message = messages[Math.floor(Math.random() * messages.length)];

            logs.push({
                timestamp: timestamp.toISOString(),
                level,
                component,
                message,
                details: level === 'ERROR' ? 'Stack trace would appear here...' : null
            });
        }

        return logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    }

    displayLogs(logs) {
        const container = document.getElementById('logsContainer');
        if (!container) return;

        const html = logs.map(log => {
            const levelClass = {
                'ERROR': 'text-danger',
                'WARN': 'text-warning',
                'INFO': 'text-info',
                'DEBUG': 'text-muted'
            }[log.level] || 'text-muted';

            const timestamp = new Date(log.timestamp).toLocaleString();

            return `
                <div class="log-entry mb-2 p-2 border-start border-3 ${levelClass.replace('text-', 'border-')}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <small class="text-muted">${timestamp}</small>
                            <span class="badge ${levelClass.replace('text-', 'bg-')} ms-2">${log.level}</span>
                            <strong class="ms-2">[${log.component}]</strong>
                            <div class="mt-1">${log.message}</div>
                            ${log.details ? `<details class="mt-2"><summary class="text-muted">Details</summary><pre class="text-muted small mt-1">${log.details}</pre></details>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    filterLogs(level) {
        const filteredLogs = level === 'ALL' ? this.logs : this.logs.filter(log => log.level === level);
        this.displayLogs(filteredLogs);
    }

    setupTechnicalCharts() {
        this.setupResponseTimeChart();
        this.setupResourceUsageChart();
        this.setupErrorRateChart();
        this.setupThroughputChart();
    }

    setupResponseTimeChart() {
        const ctx = document.getElementById('responseTimeChart');
        if (!ctx) return;

        this.charts.responseTime = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Avg Response Time (ms)',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
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
                        title: {
                            display: true,
                            text: 'Response Time (ms)'
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

    setupResourceUsageChart() {
        const ctx = document.getElementById('resourceUsageChart');
        if (!ctx) return;

        this.charts.resourceUsage = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Disk %',
                        data: [],
                        borderColor: '#20c997',
                        backgroundColor: 'rgba(32, 201, 151, 0.1)',
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Usage %'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    setupErrorRateChart() {
        const ctx = document.getElementById('errorRateChart');
        if (!ctx) return;

        this.charts.errorRate = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Error Rate %',
                    data: [],
                    backgroundColor: 'rgba(220, 53, 69, 0.8)',
                    borderColor: '#dc3545',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Error Rate %'
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

    setupThroughputChart() {
        const ctx = document.getElementById('throughputChart');
        if (!ctx) return;

        this.charts.throughput = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Documents/Hour',
                    data: [],
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 2,
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
                        title: {
                            display: true,
                            text: 'Documents per Hour'
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

    updateTechnicalCharts() {
        // Generate mock time series data for the last 24 hours
        const labels = this.generateHourlyLabels(24);
        
        // Update response time chart
        if (this.charts.responseTime) {
            const responseTimeData = this.generateResponseTimeData(24);
            this.charts.responseTime.data.labels = labels;
            this.charts.responseTime.data.datasets[0].data = responseTimeData;
            this.charts.responseTime.update();
        }

        // Update resource usage chart
        if (this.charts.resourceUsage) {
            const cpuData = this.generateResourceData(24, 20, 60);
            const memoryData = this.generateResourceData(24, 40, 70);
            const diskData = this.generateResourceData(24, 15, 30);
            
            this.charts.resourceUsage.data.labels = labels;
            this.charts.resourceUsage.data.datasets[0].data = cpuData;
            this.charts.resourceUsage.data.datasets[1].data = memoryData;
            this.charts.resourceUsage.data.datasets[2].data = diskData;
            this.charts.resourceUsage.update();
        }

        // Update error rate chart
        if (this.charts.errorRate) {
            const errorData = this.generateErrorRateData(24);
            this.charts.errorRate.data.labels = labels;
            this.charts.errorRate.data.datasets[0].data = errorData;
            this.charts.errorRate.update();
        }

        // Update throughput chart
        if (this.charts.throughput) {
            const throughputData = this.generateThroughputData(24);
            this.charts.throughput.data.labels = labels;
            this.charts.throughput.data.datasets[0].data = throughputData;
            this.charts.throughput.update();
        }
    }

    showDatabaseHealth() {
        // Mock database health data
        const dbHealth = {
            status: 'Connected',
            response_time: '12ms',
            active_connections: 15,
            max_connections: 100,
            query_performance: 'Good',
            disk_usage: '68%',
            backup_status: 'Last backup: 2 hours ago'
        };

        const modalBody = document.getElementById('dbHealthModalBody');
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Connection Status</h6>
                        <p><span class="badge bg-success">${dbHealth.status}</span></p>
                        
                        <h6>Response Time</h6>
                        <p>${dbHealth.response_time}</p>
                        
                        <h6>Active Connections</h6>
                        <p>${dbHealth.active_connections} / ${dbHealth.max_connections}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Query Performance</h6>
                        <p><span class="badge bg-success">${dbHealth.query_performance}</span></p>
                        
                        <h6>Disk Usage</h6>
                        <p>${dbHealth.disk_usage}</p>
                        
                        <h6>Backup</h6>
                        <p class="text-success">${dbHealth.backup_status}</p>
                    </div>
                </div>
            `;
        }

        const modal = new bootstrap.Modal(document.getElementById('dbHealthModal'));
        modal.show();
    }

    showCacheStats() {
        // Mock cache statistics
        const cacheStats = {
            hit_rate: '94.2%',
            miss_rate: '5.8%',
            total_requests: '12,547',
            cache_size: '2.1 GB',
            evictions: '45',
            memory_usage: '78%'
        };

        const modalBody = document.getElementById('cacheStatsModalBody');
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Hit Rate</h6>
                        <p class="text-success">${cacheStats.hit_rate}</p>
                        
                        <h6>Miss Rate</h6>
                        <p class="text-warning">${cacheStats.miss_rate}</p>
                        
                        <h6>Total Requests</h6>
                        <p>${cacheStats.total_requests}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Cache Size</h6>
                        <p>${cacheStats.cache_size}</p>
                        
                        <h6>Evictions (24h)</h6>
                        <p>${cacheStats.evictions}</p>
                        
                        <h6>Memory Usage</h6>
                        <p>${cacheStats.memory_usage}</p>
                    </div>
                </div>
            `;
        }

        const modal = new bootstrap.Modal(document.getElementById('cacheStatsModal'));
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

    startRealTimeMonitoring() {
        // Update every 15 seconds for technical dashboard
        setInterval(() => {
            if (this.autoRefresh) {
                this.loadTechnicalData();
            }
        }, 15000);
    }

    // Utility methods
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateProgressBar(id, value, colorClass) {
        const element = document.getElementById(id);
        if (!element) return;

        element.style.width = `${Math.min(100, Math.max(0, value))}%`;
        element.className = `progress-bar ${colorClass}`;
    }

    getUsageColor(value, warningThreshold, criticalThreshold) {
        if (value >= criticalThreshold) return 'bg-danger';
        if (value >= warningThreshold) return 'bg-warning';
        return 'bg-success';
    }

    generateHourlyLabels(hours) {
        const labels = [];
        for (let i = hours - 1; i >= 0; i--) {
            const date = new Date();
            date.setHours(date.getHours() - i);
            labels.push(date.toLocaleTimeString('en-US', { hour: '2-digit' }));
        }
        return labels;
    }

    generateResponseTimeData(hours) {
        const data = [];
        for (let i = 0; i < hours; i++) {
            const baseValue = 800;
            const variation = Math.random() * 400 - 200; // ±200ms
            data.push(Math.max(100, baseValue + variation));
        }
        return data;
    }

    generateResourceData(hours, baseValue, maxValue) {
        const data = [];
        for (let i = 0; i < hours; i++) {
            const variation = Math.random() * 20 - 10; // ±10%
            data.push(Math.min(maxValue, Math.max(0, baseValue + variation)));
        }
        return data;
    }

    generateErrorRateData(hours) {
        const data = [];
        for (let i = 0; i < hours; i++) {
            const baseValue = 2;
            const variation = Math.random() * 3 - 1; // ±1%
            data.push(Math.max(0, baseValue + variation));
        }
        return data;
    }

    generateThroughputData(hours) {
        const data = [];
        for (let i = 0; i < hours; i++) {
            const baseValue = 85;
            const variation = Math.random() * 30 - 15; // ±15 docs/hr
            data.push(Math.max(0, baseValue + variation));
        }
        return data;
    }

    showError(message) {
        console.error(message);
    }
}

// Initialize technical dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.technicalDashboard = new TechnicalDashboard();
});