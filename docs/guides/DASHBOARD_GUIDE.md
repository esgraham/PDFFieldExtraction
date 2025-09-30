# PDF Field Extraction Dashboard

## Overview

Comprehensive analytics and KPI dashboard for PDF field extraction pipeline with deterministic JSON to Markdown summaries, optional LLM enhancement, and PII masking capabilities.

## Features

### 🎯 **Deterministic Summary Generation**
- **Structured JSON → Markdown Templates**: Clean, consistent summaries from validated JSON data
- **Multiple Template Types**: Executive, Operational, and Technical views
- **Template Customization**: Structured templates with consistent formatting
- **No Hallucination**: Deterministic generation from actual metrics data

### 🤖 **Optional LLM Enhancement**
- **LLM-Assisted Prose**: Uses structured JSON as context for enhanced summaries
- **Constrained Output**: Structured prompts prevent hallucination
- **Audience-Aware**: Adjusts technical depth based on audience
- **Factual Grounding**: All statements backed by actual metrics

### 🔒 **PII Masking with Presidio**
- **Audience-Based Masking**: Different levels for internal/restricted/public
- **Smart Detection**: Identifies names, emails, SSNs, credit cards, phone numbers
- **Configurable Rules**: Customizable masking patterns per audience type
- **Summary Protection**: Automatically masks PII in generated summaries

### 📊 **Comprehensive KPIs**

#### Quality Metrics
- **Field-level Performance**: Precision/Recall/F1 on labeled holdout sets
- **Document Type Breakdown**: Per-type accuracy and STP rates
- **Overall Accuracy**: System-wide success rates
- **STP Rate**: Percentage of documents with no human intervention
- **Exception Analysis**: Top failure reasons and trends

#### Speed & Reliability
- **Latency Distribution**: P50/P95 processing times
- **Throughput Metrics**: Documents per hour with trending
- **Time to Review**: Average time for HITL first review
- **Time to Resolution**: End-to-end resolution times
- **SLA Adherence**: Compliance with processing time SLAs

#### Cost & Operations
- **Cost Breakdown**: Compute, storage, and review costs per document
- **Operational Efficiency**: Reprocess rates and poison queue depth
- **Resource Utilization**: Cost optimization opportunities
- **Review Minutes**: Human intervention time tracking

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Analytics       │    │ Dashboard        │    │ Summary         │
│ Engine          │────│ Database         │────│ Generator       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ KPI             │    │ Web              │    │ PII             │
│ Calculation     │    │ Dashboard        │    │ Masking         │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Metrics Collection**: Document and field-level metrics from processing pipeline
2. **Analytics Engine**: Calculates comprehensive KPIs and trends
3. **Template Selection**: Choose appropriate summary template (executive/operational/technical)
4. **JSON Generation**: Create structured JSON with all metrics and analysis
5. **Markdown Conversion**: Deterministic conversion using templates
6. **PII Masking**: Apply audience-appropriate masking rules
7. **Optional LLM Enhancement**: Generate prose summary with LLM assistance
8. **Dashboard Display**: Present in web interface with real-time updates

## Quick Start

### 1. Install Dependencies

```bash
# Core dashboard dependencies
pip install -r requirements_dashboard.txt

# Optional: For LLM enhancement
export OPENAI_API_KEY="your-api-key"

# Optional: For advanced PII detection
pip install presidio-analyzer presidio-anonymizer
```

### 2. Run Dashboard Demo

```bash
# Demonstrate all features with sample data
python run_dashboard_demo.py
```

This generates:
- `dashboard_kpis.json` - Complete KPI data
- `summary_executive.md` - Executive summary
- `summary_operational.md` - Operational report  
- `summary_technical.md` - Technical analysis

### 3. Start Web Dashboard

```bash
# Start the web application
python src/dashboard_app.py

# Access dashboard at http://localhost:8000
```

### 4. Generate Custom Summaries

```python
from src.dashboard_analytics import AnalyticsEngine, JSONToMarkdownConverter

# Initialize components
analytics = AnalyticsEngine()
converter = JSONToMarkdownConverter()

# Load your metrics data
analytics.add_document_metric(your_document_metric)
analytics.add_field_metric(your_field_metric)

# Calculate KPIs
dashboard = analytics.calculate_kpis(start_date, end_date)

# Generate deterministic summary
summary = converter.convert_dashboard_to_markdown(dashboard, "executive")
print(summary)
```

## API Endpoints

### Dashboard Data
- `GET /api/kpis?days=7&template=executive` - Get KPI dashboard data
- `GET /api/metrics/realtime` - Real-time system metrics
- `GET /api/metrics/documents?days=7` - Document processing metrics

### Summary Generation
- `GET /api/summary/{template_type}?days=7&audience=internal` - Structured markdown summary
- `GET /api/llm-summary?days=7&template=executive` - LLM-enhanced summary

### Web Interface
- `GET /` - Main dashboard
- `GET /dashboard/executive` - Executive view
- `GET /dashboard/operational` - Operational view
- `GET /dashboard/technical` - Technical view

## Template Types

### Executive Template
- **Focus**: High-level business metrics and outcomes
- **Audience**: C-level executives, business stakeholders
- **Content**: Success rates, cost efficiency, key issues
- **Format**: Clean summaries with status indicators

### Operational Template  
- **Focus**: Day-to-day operations and performance
- **Audience**: Operations managers, team leads
- **Content**: Throughput, queue status, exception analysis
- **Format**: Detailed tables with operational metrics

### Technical Template
- **Focus**: System performance and technical details
- **Audience**: Engineers, technical teams
- **Content**: Latency distributions, field-level metrics, system health
- **Format**: Technical metrics with JSON data and performance analysis

## PII Masking Rules

### Internal Audience
- **Masking**: Minimal (SSN, credit cards only)
- **Purpose**: Full visibility for internal operations
- **Example**: Names and emails remain visible

### Restricted Audience
- **Masking**: Moderate (names, contact info, sensitive data)
- **Purpose**: Sharing with trusted partners/vendors
- **Example**: "John Doe" → "[REDACTED]"

### Public Audience
- **Masking**: Complete (all PII detected and masked)
- **Purpose**: External reporting and compliance
- **Example**: Full anonymization for public disclosure

## Configuration

### Environment Variables

```bash
# Optional: OpenAI API for LLM enhancement
OPENAI_API_KEY=your-openai-api-key

# Dashboard settings
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
DASHBOARD_DEBUG=false

# Database settings
DASHBOARD_DB_PATH=dashboard.db
```

### Custom Templates

Create custom summary templates by extending `JSONToMarkdownConverter`:

```python
class CustomConverter(JSONToMarkdownConverter):
    @staticmethod
    def _custom_template(dashboard: KPIDashboard) -> str:
        return f"""
        # Custom Report
        **Period:** {dashboard.period_start} to {dashboard.period_end}
        
        ## Key Findings
        - Accuracy: {dashboard.overall_accuracy:.1%}
        - Cost: ${dashboard.cost_per_document:.4f}/doc
        
        ## Custom Analysis
        {your_custom_analysis_logic()}
        """
```

## Integration with Field Extraction Pipeline

### Metrics Collection

```python
from src.dashboard_analytics import AnalyticsEngine, DocumentMetrics, FieldMetrics

# In your field extraction pipeline
analytics = AnalyticsEngine()

# Record document processing
doc_metric = DocumentMetrics(
    document_id="DOC-001",
    document_type=DocumentType.INVOICE,
    processing_status=ProcessingStatus.SUCCESS,
    start_time=start_time,
    end_time=end_time,
    total_processing_time=processing_time,
    field_count=extracted_fields_count,
    confidence_scores=field_confidences,
    cost_breakdown=cost_data
)

analytics.add_document_metric(doc_metric)

# Record field-level performance
field_metric = FieldMetrics(
    field_name="invoice_amount",
    document_type="invoice",
    precision=0.95,
    recall=0.92,
    f1_score=0.935,
    confidence_avg=0.87
)

analytics.add_field_metric(field_metric)
```

### Real-time Updates

The dashboard automatically updates metrics in real-time as documents are processed. Integration points:

1. **Document Processing**: Record metrics after each document
2. **Field Extraction**: Track field-level performance
3. **Validation Results**: Capture validation outcomes
4. **HITL Operations**: Track human review metrics
5. **Cost Tracking**: Monitor processing costs

## Sample Output

### Executive Summary
```markdown
# Executive Dashboard Summary

**Reporting Period:** 2024-09-23 to 2024-09-30
**Report Generated:** 2024-09-30 14:32:15

## Key Performance Indicators

### 📊 Quality Metrics
- **Overall Accuracy:** 94.2%
- **Straight-Through Processing Rate:** 87.3%
- **Exception Rate:** 5.8%

### ⚡ Performance Metrics  
- **Median Processing Time:** 2.8s
- **95th Percentile Latency:** 8.4s
- **Throughput:** 342 docs/hour
- **SLA Adherence:** 96.1%

### 💰 Cost Metrics
- **Cost per Document:** $0.0234
- **Reprocess Rate:** 3.2%
- **Operational Efficiency:** 94.2%

## Summary
The document processing system processed documents with **94.2% accuracy** 
and **87.3% straight-through processing rate**. Average cost per document 
is **$0.0234** with **342 documents/hour** throughput.
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install fastapi uvicorn jinja2
   ```

2. **PII Masking Not Working**
   ```bash
   pip install presidio-analyzer presidio-anonymizer
   ```

3. **LLM Enhancement Unavailable**
   ```bash
   export OPENAI_API_KEY=your-api-key
   pip install openai
   ```

4. **Dashboard Not Loading**
   - Check port 8000 is available
   - Verify templates directory exists
   - Check database permissions

### Performance Optimization

- **Database Indexing**: Add indexes on timestamp columns for large datasets
- **Metrics Aggregation**: Pre-calculate metrics for faster dashboard loading
- **Caching**: Use Redis for frequently accessed KPIs
- **Batch Processing**: Process metrics in batches for high-volume scenarios

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_dashboard.txt .
RUN pip install -r requirements_dashboard.txt

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8000
CMD ["python", "src/dashboard_app.py"]
```

### Security Considerations

- **Authentication**: Add OAuth2/JWT authentication for production
- **HTTPS**: Use TLS certificates for encrypted connections
- **Rate Limiting**: Implement API rate limiting
- **Input Validation**: Validate all input parameters
- **PII Compliance**: Ensure PII masking meets regulatory requirements

### Monitoring

- **Health Checks**: Use `/api/health` endpoint for monitoring
- **Metrics Export**: Export metrics to Prometheus/Grafana
- **Alerting**: Set up alerts for critical KPI thresholds
- **Logging**: Structured logging for audit trails

## License

This dashboard system is part of the PDF Field Extraction project and follows the same licensing terms.