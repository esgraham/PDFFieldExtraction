"""
Dashboard System Demonstration

Complete demonstration of the KPI dashboard, analytics engine, and summary generation.
Shows deterministic JSON to Markdown conversion and optional LLM enhancement.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dashboard.dashboard_analytics import (
    AnalyticsEngine, KPIDashboard, JSONToMarkdownConverter,
    PIIMaskingService, LLMSummaryService, DocumentMetrics, FieldMetrics,
    generate_sample_metrics, DocumentType, ProcessingStatus
)

def demonstrate_analytics_engine():
    """Demonstrate the analytics engine with sample data."""
    print("🔧 Analytics Engine Demonstration")
    print("=" * 50)
    
    # Initialize analytics engine
    analytics = AnalyticsEngine()
    
    # Generate and load sample data
    doc_metrics, field_metrics = generate_sample_metrics()
    
    print(f"Generated {len(doc_metrics)} document metrics")
    print(f"Generated {len(field_metrics)} field metrics")
    
    for metric in doc_metrics:
        analytics.add_document_metric(metric)
    
    for metric in field_metrics:
        analytics.add_field_metric(metric)
    
    # Calculate KPIs for last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    dashboard = analytics.calculate_kpis(start_date, end_date)
    
    print(f"\n📊 KPI Dashboard for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Overall Accuracy: {dashboard.overall_accuracy:.1%}")
    print(f"STP Rate: {dashboard.stp_rate:.1%}")
    print(f"Exception Rate: {dashboard.exception_rate:.1%}")
    print(f"Throughput: {dashboard.throughput_docs_per_hour:.1f} docs/hour")
    print(f"P95 Latency: {dashboard.latency_p95:.2f}s")
    print(f"Cost per Document: ${dashboard.cost_per_document:.4f}")
    print(f"Reprocess Rate: {dashboard.reprocess_rate:.2%}")
    
    if dashboard.top_exceptions:
        print(f"\nTop Exceptions:")
        for exc in dashboard.top_exceptions[:3]:
            print(f"  • {exc['description']}: {exc['count']} ({exc['percentage']:.1f}%)")
    
    print(f"\nDocument Types:")
    for doc_type, metrics in dashboard.document_type_metrics.items():
        print(f"  • {doc_type.title()}: {metrics['count']:.0f} docs, {metrics['success_rate']:.1%} success")
    
    return dashboard

def demonstrate_markdown_conversion(dashboard: KPIDashboard):
    """Demonstrate JSON to Markdown conversion."""
    print("\n📝 Markdown Conversion Demonstration")
    print("=" * 50)
    
    converter = JSONToMarkdownConverter()
    
    # Generate different template types
    templates = ["executive", "operational", "technical"]
    
    for template in templates:
        print(f"\n🎯 {template.upper()} TEMPLATE:")
        print("-" * 30)
        
        markdown = converter.convert_dashboard_to_markdown(dashboard, template)
        
        # Show first few lines
        lines = markdown.split('\n')
        for line in lines[:15]:
            print(line)
        
        if len(lines) > 15:
            print(f"... ({len(lines) - 15} more lines)")
        
        # Save to file
        output_file = f"dashboard_summary_{template}.md"
        Path(output_file).write_text(markdown)
        print(f"💾 Saved to {output_file}")

def demonstrate_pii_masking():
    """Demonstrate PII masking capabilities."""
    print("\n🔒 PII Masking Demonstration")
    print("=" * 50)
    
    pii_service = PIIMaskingService()
    
    # Sample text with PII
    sample_text = """
    Executive Summary for Q4 2024
    
    Our document processing system handled 50,000 documents including:
    - Invoice from Acme Corp (contact: john.doe@acme.com, phone: 555-123-4567)
    - Customer data for Jane Smith (SSN: 123-45-6789)
    - Payment info with credit card 4532-1234-5678-9012
    
    Overall performance metrics show 94.2% accuracy with $0.0234 cost per document.
    """
    
    audiences = ["internal", "restricted", "public"]
    
    for audience in audiences:
        print(f"\n👥 {audience.upper()} AUDIENCE:")
        print("-" * 20)
        
        masked_text = pii_service.mask_pii_in_text(sample_text, audience)
        
        # Show differences
        if audience == "internal":
            print("✅ Original text (minimal masking)")
        else:
            print("🔒 PII masked for external sharing")
        
        lines = masked_text.strip().split('\n')
        for line in lines[:8]:
            print(line)
        
        if len(lines) > 8:
            print("...")

def demonstrate_llm_enhancement(dashboard: KPIDashboard):
    """Demonstrate LLM-enhanced summaries."""
    print("\n🤖 LLM Enhancement Demonstration")
    print("=" * 50)
    
    llm_service = LLMSummaryService()
    
    # Convert dashboard to structured JSON
    dashboard_dict = {
        "period": f"{dashboard.period_start.strftime('%Y-%m-%d')} to {dashboard.period_end.strftime('%Y-%m-%d')}",
        "quality_metrics": {
            "accuracy": dashboard.overall_accuracy,
            "stp_rate": dashboard.stp_rate,
            "exception_rate": dashboard.exception_rate
        },
        "performance_metrics": {
            "latency_p50": dashboard.latency_p50,
            "latency_p95": dashboard.latency_p95,
            "throughput": dashboard.throughput_docs_per_hour,
            "sla_adherence": dashboard.sla_adherence_rate
        },
        "cost_metrics": {
            "cost_per_document": dashboard.cost_per_document,
            "reprocess_rate": dashboard.reprocess_rate
        },
        "top_exceptions": dashboard.top_exceptions[:3]
    }
    
    print("📊 Structured JSON input:")
    print(json.dumps(dashboard_dict, indent=2, default=str)[:500] + "...")
    
    # Generate LLM summary
    enhanced_summary = llm_service.enhance_summary(dashboard_dict, "executive")
    
    print(f"\n🎯 LLM-Enhanced Executive Summary:")
    print("-" * 40)
    print(enhanced_summary)

def demonstrate_field_metrics(dashboard: KPIDashboard):
    """Demonstrate field-level quality metrics."""
    print("\n📈 Field-Level Quality Metrics")
    print("=" * 50)
    
    if dashboard.field_metrics:
        print("Top performing fields:")
        
        # Sort by F1 score
        sorted_fields = sorted(dashboard.field_metrics, key=lambda x: x.f1_score, reverse=True)
        
        print(f"{'Field':<20} {'Doc Type':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 65)
        
        for field in sorted_fields[:10]:
            print(f"{field.field_name:<20} {field.document_type:<15} {field.precision:<10.3f} {field.recall:<10.3f} {field.f1_score:<10.3f}")
    
    print(f"\n📋 Document Type Performance:")
    print(f"{'Type':<15} {'Count':<8} {'Success':<10} {'STP Rate':<10} {'Avg Time':<10}")
    print("-" * 63)
    
    for doc_type, metrics in dashboard.document_type_metrics.items():
        print(f"{doc_type:<15} {metrics['count']:<8.0f} {metrics['success_rate']:<10.1%} {metrics['stp_rate']:<10.1%} {metrics['avg_processing_time']:<10.2f}s")

def demonstrate_cost_analysis(dashboard: KPIDashboard):
    """Demonstrate cost analysis and operational metrics."""
    print("\n💰 Cost Analysis & Operations")
    print("=" * 50)
    
    print(f"Cost Breakdown per Document:")
    print(f"  • Compute: ${dashboard.compute_cost_per_doc:.4f}")
    print(f"  • Storage: ${dashboard.storage_cost_per_doc:.4f}")
    print(f"  • Review:  ${dashboard.review_cost_per_doc:.4f}")
    print(f"  • Total:   ${dashboard.cost_per_document:.4f}")
    
    # Calculate percentages
    total = dashboard.cost_per_document
    if total > 0:
        compute_pct = (dashboard.compute_cost_per_doc / total) * 100
        storage_pct = (dashboard.storage_cost_per_doc / total) * 100
        review_pct = (dashboard.review_cost_per_doc / total) * 100
        
        print(f"\nCost Distribution:")
        print(f"  • Compute: {compute_pct:.1f}%")
        print(f"  • Storage: {storage_pct:.1f}%")
        print(f"  • Review:  {review_pct:.1f}%")
    
    print(f"\nOperational Metrics:")
    print(f"  • Reprocess Rate: {dashboard.reprocess_rate:.2%}")
    print(f"  • Poison Queue Count: {dashboard.poison_queue_count}")
    print(f"  • SLA Adherence: {dashboard.sla_adherence_rate:.1%}")
    print(f"  • Time to First Review: {dashboard.time_to_first_review_avg:.1f} min")
    print(f"  • Time to Resolution: {dashboard.time_to_resolution_avg:.1f} min")

def save_demo_outputs(dashboard: KPIDashboard):
    """Save demonstration outputs to files."""
    print("\n💾 Saving Demo Outputs")
    print("=" * 50)
    
    # Save dashboard data as JSON
    dashboard_json = {
        "dashboard": dashboard,
        "generated_at": datetime.now().isoformat(),
        "demo_version": "1.0"
    }
    
    output_file = "dashboard_kpis.json"
    Path(output_file).write_text(json.dumps(dashboard_json, indent=2, default=str))
    print(f"📊 KPI dashboard saved to {output_file}")
    
    # Save markdown summaries
    converter = JSONToMarkdownConverter()
    
    templates = ["executive", "operational", "technical"]
    for template in templates:
        markdown = converter.convert_dashboard_to_markdown(dashboard, template)
        output_file = f"summary_{template}.md"
        Path(output_file).write_text(markdown)
        print(f"📝 {template.title()} summary saved to {output_file}")
    
    print(f"\n✅ All demo outputs saved successfully!")

def main():
    """Run complete dashboard demonstration."""
    print("🚀 PDF Field Extraction Dashboard Demo")
    print("=" * 60)
    print("Demonstrating comprehensive analytics, KPIs, and summaries")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Analytics Engine
        dashboard = demonstrate_analytics_engine()
        
        # Step 2: Markdown Conversion
        demonstrate_markdown_conversion(dashboard)
        
        # Step 3: PII Masking
        demonstrate_pii_masking()
        
        # Step 4: LLM Enhancement
        demonstrate_llm_enhancement(dashboard)
        
        # Step 5: Field Metrics
        demonstrate_field_metrics(dashboard)
        
        # Step 6: Cost Analysis
        demonstrate_cost_analysis(dashboard)
        
        # Step 7: Save Outputs
        save_demo_outputs(dashboard)
        
        print("\n🎉 Dashboard Demo Completed Successfully!")
        print("=" * 60)
        print("Features demonstrated:")
        print("✅ Analytics engine with comprehensive KPIs")
        print("✅ Deterministic JSON to Markdown templates")
        print("✅ PII masking with audience-based rules")
        print("✅ LLM-enhanced summary generation")
        print("✅ Field-level quality metrics")
        print("✅ Cost analysis and operational metrics")
        print("✅ Document type performance breakdown")
        print("✅ Exception analysis and trending")
        
        print(f"\n📁 Output files generated:")
        print("  • dashboard_kpis.json - Complete KPI data")
        print("  • summary_executive.md - Executive summary")
        print("  • summary_operational.md - Operational report")
        print("  • summary_technical.md - Technical analysis")
        
        print(f"\n🌐 Web Dashboard:")
        print("  Run 'python src/dashboard_app.py' to start the web interface")
        print("  Access at: http://localhost:8000")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")

if __name__ == "__main__":
    main()