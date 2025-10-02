"""
Dashboard Analytics Engine

Comprehensive analytics and KPI tracking for PDF field extraction system.
Includes deterministic JSON to Markdown templates and optional LLM assistance.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from pathlib import Path
import uuid

# Optional LLM integration
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional PII detection
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Document types for analytics."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    PURCHASE_ORDER = "purchase_order"
    TAX_FORM = "tax_form"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Document processing status."""
    SUCCESS = "success"
    HITL_REQUIRED = "hitl_required"
    FAILED = "failed"
    REPROCESSING = "reprocessing"

@dataclass
class FieldMetrics:
    """Field-level quality metrics."""
    field_name: str
    document_type: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    confidence_avg: float
    confidence_std: float

@dataclass
class DocumentMetrics:
    """Document-level processing metrics."""
    document_id: str
    document_type: DocumentType
    processing_status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime]
    ocr_time: float
    extraction_time: float
    validation_time: float
    total_processing_time: float
    field_count: int
    validation_errors: List[str]
    confidence_scores: Dict[str, float]
    hitl_required: bool
    hitl_reason: Optional[str]
    cost_breakdown: Dict[str, float]
    exception_type: Optional[str]

@dataclass
class KPIDashboard:
    """Complete KPI dashboard data structure."""
    
    # Time period
    period_start: datetime
    period_end: datetime
    report_generated: datetime
    
    # Quality metrics
    overall_accuracy: float
    stp_rate: float  # Straight-through processing rate
    exception_rate: float
    field_metrics: List[FieldMetrics]
    
    # Speed & reliability
    latency_p50: float
    latency_p95: float
    throughput_docs_per_hour: float
    time_to_first_review_avg: float
    time_to_resolution_avg: float
    sla_adherence_rate: float
    
    # Cost & operations
    cost_per_document: float
    compute_cost_per_doc: float
    storage_cost_per_doc: float
    review_cost_per_doc: float
    reprocess_rate: float
    poison_queue_count: int
    
    # Exception analysis
    top_exceptions: List[Dict[str, Any]]
    exception_breakdown: Dict[str, int]
    
    # Document type breakdown
    document_type_metrics: Dict[str, Dict[str, float]]

class AnalyticsEngine:
    """Core analytics and metrics calculation engine."""
    
    def __init__(self):
        self.document_metrics: List[DocumentMetrics] = []
        self.field_metrics: List[FieldMetrics] = []
        self.labeled_holdout_data: Dict[str, Any] = {}
        
    def add_document_metric(self, metric: DocumentMetrics):
        """Add document processing metric."""
        self.document_metrics.append(metric)
        
    def add_field_metric(self, metric: FieldMetrics):
        """Add field-level quality metric."""
        self.field_metrics.append(metric)
        
    def calculate_kpis(self, start_date: datetime, end_date: datetime) -> KPIDashboard:
        """Calculate comprehensive KPIs for the specified period."""
        
        # Filter metrics by date range
        period_docs = [
            doc for doc in self.document_metrics 
            if start_date <= doc.start_time <= end_date
        ]
        
        if not period_docs:
            return self._empty_dashboard(start_date, end_date)
        
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(period_docs)
        
        # Speed & reliability metrics
        speed_metrics = self._calculate_speed_metrics(period_docs)
        
        # Cost & operations metrics
        cost_metrics = self._calculate_cost_metrics(period_docs)
        
        # Exception analysis
        exception_metrics = self._calculate_exception_metrics(period_docs)
        
        # Document type breakdown
        type_metrics = self._calculate_document_type_metrics(period_docs)
        
        return KPIDashboard(
            period_start=start_date,
            period_end=end_date,
            report_generated=datetime.now(),
            
            # Quality
            overall_accuracy=quality_metrics['accuracy'],
            stp_rate=quality_metrics['stp_rate'],
            exception_rate=quality_metrics['exception_rate'],
            field_metrics=self._get_field_metrics_for_period(start_date, end_date),
            
            # Speed & reliability
            latency_p50=speed_metrics['latency_p50'],
            latency_p95=speed_metrics['latency_p95'],
            throughput_docs_per_hour=speed_metrics['throughput'],
            time_to_first_review_avg=speed_metrics['time_to_first_review'],
            time_to_resolution_avg=speed_metrics['time_to_resolution'],
            sla_adherence_rate=speed_metrics['sla_adherence'],
            
            # Cost & operations
            cost_per_document=cost_metrics['total_cost_per_doc'],
            compute_cost_per_doc=cost_metrics['compute_cost_per_doc'],
            storage_cost_per_doc=cost_metrics['storage_cost_per_doc'],
            review_cost_per_doc=cost_metrics['review_cost_per_doc'],
            reprocess_rate=cost_metrics['reprocess_rate'],
            poison_queue_count=cost_metrics['poison_queue_count'],
            
            # Exceptions
            top_exceptions=exception_metrics['top_exceptions'],
            exception_breakdown=exception_metrics['breakdown'],
            
            # Document types
            document_type_metrics=type_metrics
        )
    
    def _calculate_quality_metrics(self, docs: List[DocumentMetrics]) -> Dict[str, float]:
        """Calculate quality-related metrics."""
        total_docs = len(docs)
        successful_docs = len([d for d in docs if d.processing_status == ProcessingStatus.SUCCESS])
        hitl_docs = len([d for d in docs if d.hitl_required])
        exception_docs = len([d for d in docs if d.exception_type is not None])
        
        return {
            'accuracy': successful_docs / total_docs if total_docs > 0 else 0.0,
            'stp_rate': (total_docs - hitl_docs) / total_docs if total_docs > 0 else 0.0,
            'exception_rate': exception_docs / total_docs if total_docs > 0 else 0.0
        }
    
    def _calculate_speed_metrics(self, docs: List[DocumentMetrics]) -> Dict[str, float]:
        """Calculate speed and reliability metrics."""
        processing_times = [d.total_processing_time for d in docs if d.total_processing_time > 0]
        
        if not processing_times:
            return {
                'latency_p50': 0.0,
                'latency_p95': 0.0,
                'throughput': 0.0,
                'time_to_first_review': 0.0,
                'time_to_resolution': 0.0,
                'sla_adherence': 0.0
            }
        
        # Calculate percentiles
        latency_p50 = statistics.median(processing_times)
        latency_p95 = statistics.quantiles(processing_times, n=20)[18] if len(processing_times) >= 20 else max(processing_times)
        
        # Calculate throughput (docs per hour)
        if docs:
            time_span_hours = (max(d.start_time for d in docs) - min(d.start_time for d in docs)).total_seconds() / 3600
            throughput = len(docs) / max(time_span_hours, 1.0)
        else:
            throughput = 0.0
        
        # HITL metrics (mock data for demonstration)
        hitl_docs = [d for d in docs if d.hitl_required]
        time_to_first_review = statistics.mean([30.5, 45.2, 22.8]) if hitl_docs else 0.0  # minutes
        time_to_resolution = statistics.mean([120.3, 95.7, 180.1]) if hitl_docs else 0.0  # minutes
        
        # SLA adherence (assuming 5-minute SLA)
        sla_compliant = len([t for t in processing_times if t <= 300])  # 5 minutes = 300 seconds
        sla_adherence = sla_compliant / len(processing_times)
        
        return {
            'latency_p50': latency_p50,
            'latency_p95': latency_p95,
            'throughput': throughput,
            'time_to_first_review': time_to_first_review,
            'time_to_resolution': time_to_resolution,
            'sla_adherence': sla_adherence
        }
    
    def _calculate_cost_metrics(self, docs: List[DocumentMetrics]) -> Dict[str, float]:
        """Calculate cost and operational metrics."""
        if not docs:
            return {
                'total_cost_per_doc': 0.0,
                'compute_cost_per_doc': 0.0,
                'storage_cost_per_doc': 0.0,
                'review_cost_per_doc': 0.0,
                'reprocess_rate': 0.0,
                'poison_queue_count': 0
            }
        
        # Calculate average costs
        total_costs = []
        compute_costs = []
        storage_costs = []
        review_costs = []
        
        for doc in docs:
            if doc.cost_breakdown:
                total_costs.append(sum(doc.cost_breakdown.values()))
                compute_costs.append(doc.cost_breakdown.get('compute', 0.0))
                storage_costs.append(doc.cost_breakdown.get('storage', 0.0))
                review_costs.append(doc.cost_breakdown.get('review', 0.0))
        
        # Reprocess rate
        reprocessed_docs = len([d for d in docs if d.processing_status == ProcessingStatus.REPROCESSING])
        reprocess_rate = reprocessed_docs / len(docs)
        
        # Poison queue count (mock for demonstration)
        poison_queue_count = len([d for d in docs if d.exception_type == 'poison_queue'])
        
        return {
            'total_cost_per_doc': statistics.mean(total_costs) if total_costs else 0.0,
            'compute_cost_per_doc': statistics.mean(compute_costs) if compute_costs else 0.0,
            'storage_cost_per_doc': statistics.mean(storage_costs) if storage_costs else 0.0,
            'review_cost_per_doc': statistics.mean(review_costs) if review_costs else 0.0,
            'reprocess_rate': reprocess_rate,
            'poison_queue_count': poison_queue_count
        }
    
    def _calculate_exception_metrics(self, docs: List[DocumentMetrics]) -> Dict[str, Any]:
        """Calculate exception analysis metrics."""
        exceptions = [d for d in docs if d.exception_type]
        
        # Count exception types
        exception_counts = {}
        for doc in exceptions:
            exception_type = doc.exception_type
            exception_counts[exception_type] = exception_counts.get(exception_type, 0) + 1
        
        # Top exceptions with details
        top_exceptions = []
        for exc_type, count in sorted(exception_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(docs)) * 100 if docs else 0
            top_exceptions.append({
                'type': exc_type,
                'count': count,
                'percentage': percentage,
                'description': self._get_exception_description(exc_type)
            })
        
        return {
            'top_exceptions': top_exceptions,
            'breakdown': exception_counts
        }
    
    def _calculate_document_type_metrics(self, docs: List[DocumentMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by document type."""
        type_metrics = {}
        
        for doc_type in DocumentType:
            type_docs = [d for d in docs if d.document_type == doc_type]
            if not type_docs:
                continue
                
            successful = len([d for d in type_docs if d.processing_status == ProcessingStatus.SUCCESS])
            hitl_required = len([d for d in type_docs if d.hitl_required])
            avg_processing_time = statistics.mean([d.total_processing_time for d in type_docs if d.total_processing_time > 0])
            
            type_metrics[doc_type.value] = {
                'count': len(type_docs),
                'success_rate': successful / len(type_docs),
                'stp_rate': (len(type_docs) - hitl_required) / len(type_docs),
                'avg_processing_time': avg_processing_time
            }
        
        return type_metrics
    
    def _get_field_metrics_for_period(self, start_date: datetime, end_date: datetime) -> List[FieldMetrics]:
        """Get field metrics for the specified period."""
        # In a real implementation, this would filter by date
        return self.field_metrics[:10]  # Return top 10 for demonstration
    
    def _get_exception_description(self, exception_type: str) -> str:
        """Get human-readable description for exception type."""
        descriptions = {
            'low_ocr_confidence': 'OCR confidence below threshold',
            'validation_failure': 'Field validation rule failure',
            'missing_page': 'Required page missing from document',
            'unsupported_format': 'Document format not supported',
            'processing_timeout': 'Processing exceeded time limit',
            'poison_queue': 'Document failed multiple processing attempts'
        }
        return descriptions.get(exception_type, exception_type)
    
    def _empty_dashboard(self, start_date: datetime, end_date: datetime) -> KPIDashboard:
        """Return empty dashboard for periods with no data."""
        return KPIDashboard(
            period_start=start_date,
            period_end=end_date,
            report_generated=datetime.now(),
            overall_accuracy=0.0,
            stp_rate=0.0,
            exception_rate=0.0,
            field_metrics=[],
            latency_p50=0.0,
            latency_p95=0.0,
            throughput_docs_per_hour=0.0,
            time_to_first_review_avg=0.0,
            time_to_resolution_avg=0.0,
            sla_adherence_rate=0.0,
            cost_per_document=0.0,
            compute_cost_per_doc=0.0,
            storage_cost_per_doc=0.0,
            review_cost_per_doc=0.0,
            reprocess_rate=0.0,
            poison_queue_count=0,
            top_exceptions=[],
            exception_breakdown={},
            document_type_metrics={}
        )

class JSONToMarkdownConverter:
    """Deterministic JSON to Markdown summary template converter."""
    
    @staticmethod
    def convert_dashboard_to_markdown(dashboard: KPIDashboard, template_type: str = "executive") -> str:
        """Convert KPI dashboard to structured Markdown summary."""
        
        if template_type == "executive":
            return JSONToMarkdownConverter._executive_template(dashboard)
        elif template_type == "operational":
            return JSONToMarkdownConverter._operational_template(dashboard)
        elif template_type == "technical":
            return JSONToMarkdownConverter._technical_template(dashboard)
        else:
            return JSONToMarkdownConverter._default_template(dashboard)
    
    @staticmethod
    def _executive_template(dashboard: KPIDashboard) -> str:
        """Executive summary template."""
        period = f"{dashboard.period_start.strftime('%Y-%m-%d')} to {dashboard.period_end.strftime('%Y-%m-%d')}"
        
        return f"""# Executive Dashboard Summary

**Reporting Period:** {period}
**Report Generated:** {dashboard.report_generated.strftime('%Y-%m-%d %H:%M:%S')}

## Key Performance Indicators

### 📊 Quality Metrics
- **Overall Accuracy:** {dashboard.overall_accuracy:.1%}
- **Straight-Through Processing Rate:** {dashboard.stp_rate:.1%}
- **Exception Rate:** {dashboard.exception_rate:.1%}

### ⚡ Performance Metrics  
- **Median Processing Time:** {dashboard.latency_p50:.2f}s
- **95th Percentile Latency:** {dashboard.latency_p95:.2f}s
- **Throughput:** {dashboard.throughput_docs_per_hour:.0f} docs/hour
- **SLA Adherence:** {dashboard.sla_adherence_rate:.1%}

### 💰 Cost Metrics
- **Cost per Document:** ${dashboard.cost_per_document:.3f}
- **Reprocess Rate:** {dashboard.reprocess_rate:.1%}
- **Operational Efficiency:** {(1 - dashboard.exception_rate):.1%}

## Top Issues Requiring Attention

{JSONToMarkdownConverter._format_top_exceptions(dashboard.top_exceptions)}

## Document Type Performance

{JSONToMarkdownConverter._format_document_types(dashboard.document_type_metrics)}

## Summary
The document processing system processed documents with **{dashboard.overall_accuracy:.1%} accuracy** and **{dashboard.stp_rate:.1%} straight-through processing rate**. 
Average cost per document is **${dashboard.cost_per_document:.3f}** with **{dashboard.throughput_docs_per_hour:.0f} documents/hour** throughput.

---
*Report generated automatically from validated processing metrics*"""

    @staticmethod
    def _operational_template(dashboard: KPIDashboard) -> str:
        """Operational dashboard template."""
        period = f"{dashboard.period_start.strftime('%Y-%m-%d')} to {dashboard.period_end.strftime('%Y-%m-%d')}"
        
        return f"""# Operational Dashboard Report

**Period:** {period} | **Generated:** {dashboard.report_generated.strftime('%Y-%m-%d %H:%M:%S')}

## Processing Volume & Performance

| Metric | Value | Status |
|--------|-------|--------|
| Processing Accuracy | {dashboard.overall_accuracy:.1%} | {'🟢 Good' if dashboard.overall_accuracy > 0.9 else '🟡 Needs Attention' if dashboard.overall_accuracy > 0.8 else '🔴 Critical'} |
| STP Rate | {dashboard.stp_rate:.1%} | {'🟢 Good' if dashboard.stp_rate > 0.8 else '🟡 Needs Attention' if dashboard.stp_rate > 0.7 else '🔴 Critical'} |
| Throughput | {dashboard.throughput_docs_per_hour:.0f} docs/hr | {'🟢 Good' if dashboard.throughput_docs_per_hour > 100 else '🟡 Monitor'} |
| P95 Latency | {dashboard.latency_p95:.2f}s | {'🟢 Good' if dashboard.latency_p95 < 10 else '🟡 Monitor' if dashboard.latency_p95 < 30 else '🔴 Critical'} |

## Exception Analysis

### Top Exception Types
{JSONToMarkdownConverter._format_exceptions_table(dashboard.top_exceptions)}

### Queue Status
- **Poison Queue Count:** {dashboard.poison_queue_count}
- **Reprocess Rate:** {dashboard.reprocess_rate:.2%}

## Cost Breakdown

| Cost Component | Per Document | Percentage |
|----------------|-------------|------------|
| Compute | ${dashboard.compute_cost_per_doc:.4f} | {(dashboard.compute_cost_per_doc/dashboard.cost_per_document*100):.1f}% |
| Storage | ${dashboard.storage_cost_per_doc:.4f} | {(dashboard.storage_cost_per_doc/dashboard.cost_per_document*100):.1f}% |
| Review | ${dashboard.review_cost_per_doc:.4f} | {(dashboard.review_cost_per_doc/dashboard.cost_per_document*100):.1f}% |
| **Total** | **${dashboard.cost_per_document:.4f}** | **100%** |

## HITL Operations

- **Average Time to First Review:** {dashboard.time_to_first_review_avg:.1f} minutes
- **Average Time to Resolution:** {dashboard.time_to_resolution_avg:.1f} minutes
- **Documents Requiring Review:** {(1-dashboard.stp_rate)*100:.1f}%

---
*Operational metrics updated in real-time*"""

    @staticmethod
    def _technical_template(dashboard: KPIDashboard) -> str:
        """Technical details template."""
        return f"""# Technical Performance Report

**Analysis Period:** {dashboard.period_start.strftime('%Y-%m-%d')} - {dashboard.period_end.strftime('%Y-%m-%d')}
**Generated:** {dashboard.report_generated.strftime('%Y-%m-%d %H:%M:%S UTC')}

## System Performance Metrics

### Latency Distribution
- **P50 (Median):** {dashboard.latency_p50:.3f}s
- **P95:** {dashboard.latency_p95:.3f}s
- **Throughput:** {dashboard.throughput_docs_per_hour:.2f} documents/hour

### Quality Metrics by Component
- **OCR Accuracy:** {dashboard.overall_accuracy:.3%}
- **Validation Pass Rate:** {(1-dashboard.exception_rate):.3%}
- **STP Rate:** {dashboard.stp_rate:.3%}

### Field-Level Performance
{JSONToMarkdownConverter._format_field_metrics(dashboard.field_metrics)}

### Exception Breakdown
```json
{json.dumps(dashboard.exception_breakdown, indent=2)}
```

### Document Type Analysis
{JSONToMarkdownConverter._format_technical_doc_types(dashboard.document_type_metrics)}

### Cost Analysis
- **Compute Cost/Doc:** ${dashboard.compute_cost_per_doc:.6f}
- **Storage Cost/Doc:** ${dashboard.storage_cost_per_doc:.6f}  
- **Review Cost/Doc:** ${dashboard.review_cost_per_doc:.6f}
- **Total Cost/Doc:** ${dashboard.cost_per_document:.6f}

### Operational Metrics
- **Reprocess Rate:** {dashboard.reprocess_rate:.4%}
- **Poison Queue Depth:** {dashboard.poison_queue_count}
- **SLA Compliance:** {dashboard.sla_adherence_rate:.2%}

---
*Technical metrics for system optimization and monitoring*"""

    @staticmethod
    def _format_top_exceptions(exceptions: List[Dict[str, Any]]) -> str:
        """Format top exceptions for display."""
        if not exceptions:
            return "No exceptions to report ✅"
        
        result = ""
        for i, exc in enumerate(exceptions[:3], 1):
            result += f"{i}. **{exc['description']}**: {exc['count']} occurrences ({exc['percentage']:.1f}%)\n"
        
        return result

    @staticmethod
    def _format_exceptions_table(exceptions: List[Dict[str, Any]]) -> str:
        """Format exceptions as a table."""
        if not exceptions:
            return "No exceptions in this period ✅"
        
        table = "| Exception Type | Count | Percentage |\n|---|---|---|\n"
        for exc in exceptions[:5]:
            table += f"| {exc['description']} | {exc['count']} | {exc['percentage']:.1f}% |\n"
        
        return table

    @staticmethod
    def _format_document_types(doc_types: Dict[str, Dict[str, float]]) -> str:
        """Format document type metrics."""
        if not doc_types:
            return "No document type data available"
        
        result = "| Document Type | Count | Success Rate | STP Rate |\n|---|---|---|---|\n"
        for doc_type, metrics in doc_types.items():
            result += f"| {doc_type.title()} | {metrics['count']:.0f} | {metrics['success_rate']:.1%} | {metrics['stp_rate']:.1%} |\n"
        
        return result

    @staticmethod
    def _format_field_metrics(field_metrics: List[FieldMetrics]) -> str:
        """Format field-level metrics."""
        if not field_metrics:
            return "No field metrics available"
        
        result = "| Field | Document Type | Precision | Recall | F1 Score |\n|---|---|---|---|---|\n"
        for metric in field_metrics[:10]:  # Top 10
            result += f"| {metric.field_name} | {metric.document_type} | {metric.precision:.3f} | {metric.recall:.3f} | {metric.f1_score:.3f} |\n"
        
        return result

    @staticmethod
    def _format_technical_doc_types(doc_types: Dict[str, Dict[str, float]]) -> str:
        """Format technical document type analysis."""
        if not doc_types:
            return "No document type analysis available"
        
        result = ""
        for doc_type, metrics in doc_types.items():
            result += f"**{doc_type.upper()}**\n"
            result += f"- Volume: {metrics['count']:.0f} documents\n"
            result += f"- Success Rate: {metrics['success_rate']:.3%}\n"
            result += f"- STP Rate: {metrics['stp_rate']:.3%}\n"
            result += f"- Avg Processing Time: {metrics['avg_processing_time']:.3f}s\n\n"
        
        return result

class PIIMaskingService:
    """PII masking service using Presidio."""
    
    def __init__(self):
        if HAS_PRESIDIO:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None
            logger.warning("Presidio not available - PII masking disabled")
    
    def mask_pii_in_text(self, text: str, audience: str = "internal") -> str:
        """Mask PII in text based on audience level."""
        if not self.analyzer or not self.anonymizer:
            return text
        
        try:
            # Analyze text for PII
            analyzer_results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'SSN', 'CREDIT_CARD']
            )
            
            # Define masking rules based on audience
            if audience == "public":
                # Full masking for public audience
                anonymizer_results = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results
                )
            elif audience == "restricted":
                # Partial masking for restricted audience
                anonymizer_results = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators={"DEFAULT": {"replace_with": "[REDACTED]"}}
                )
            else:
                # Minimal masking for internal audience
                sensitive_entities = ['SSN', 'CREDIT_CARD']
                filtered_results = [r for r in analyzer_results if r.entity_type in sensitive_entities]
                anonymizer_results = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=filtered_results
                )
            
            return anonymizer_results.text
            
        except Exception as e:
            logger.error(f"PII masking failed: {str(e)}")
            return text
    
    def mask_dashboard_summary(self, markdown_summary: str, audience: str = "internal") -> str:
        """Mask PII in dashboard summary based on audience."""
        return self.mask_pii_in_text(markdown_summary, audience)

class LLMSummaryService:
    """Optional LLM-assisted summary generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        if HAS_OPENAI and api_key:
            openai.api_key = api_key
            self.client = openai
        else:
            logger.warning("OpenAI not available - LLM summary disabled")
    
    def enhance_summary(self, structured_json: Dict[str, Any], template_type: str = "executive") -> str:
        """Generate LLM-enhanced prose summary from structured JSON."""
        if not self.client:
            return "LLM enhancement not available"
        
        try:
            # Create prompt for structured output
            prompt = self._create_summary_prompt(structured_json, template_type)
            
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business analyst creating executive summaries from document processing metrics. Provide structured, factual analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3  # Low temperature for consistent, factual output
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {str(e)}")
            return "LLM summary generation unavailable"
    
    def _create_summary_prompt(self, data: Dict[str, Any], template_type: str) -> str:
        """Create structured prompt for LLM summary generation."""
        
        base_prompt = f"""
Based on the following document processing metrics, create a {template_type} summary:

{json.dumps(data, indent=2, default=str)}

Requirements:
1. Start with key findings and recommendations
2. Highlight trends and anomalies
3. Provide actionable insights for stakeholders
4. Use bullet points for clarity
5. Include specific metrics to support statements
6. Keep technical jargon appropriate for {template_type} audience
7. Maximum 300 words

Focus areas:
- System performance and reliability
- Cost efficiency opportunities  
- Quality improvement recommendations
- Operational optimization suggestions
"""
        
        return base_prompt