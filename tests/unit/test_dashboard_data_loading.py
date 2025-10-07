#!/usr/bin/env python3
"""
Test Dashboard Data Loading

Test script to verify that dashboard can properly load real data from the database
and populate the analytics engine without FastAPI dependencies.
"""

import sys
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'dashboard'))

from dashboard_analytics import (
    AnalyticsEngine, DocumentMetrics, FieldMetrics, 
    DocumentType, ProcessingStatus
)

def adapt_datetime_iso(val):
    return val.isoformat()

def convert_timestamp(val):
    return datetime.fromisoformat(val.decode())

# Register SQLite adapters
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)

def test_dashboard_data_loading():
    """Test loading real data from database into analytics engine."""
    
    print("🧪 Testing Dashboard Data Loading from Database")
    print("=" * 60)
    
    # Create analytics engine
    analytics_engine = AnalyticsEngine()
    print(f"📊 Created empty analytics engine with {len(analytics_engine.document_metrics)} metrics")
    
    # Database path
    db_path = Path("data/databases/dashboard.db")
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Load data from database
    try:
        with sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all document metrics
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            cursor = conn.execute("""
                SELECT * FROM document_metrics 
                WHERE start_time BETWEEN ? AND ?
                ORDER BY start_time DESC
            """, (start_date, end_date))
            
            metrics_data = [dict(row) for row in cursor.fetchall()]
            print(f"📄 Found {len(metrics_data)} document metrics in database")
            
            # Convert to DocumentMetrics objects and add to analytics engine
            loaded_count = 0
            for metric_dict in metrics_data:
                try:
                    # Convert database row to DocumentMetrics object
                    doc_metric = DocumentMetrics(
                        document_id=metric_dict['document_id'],
                        document_type=DocumentType(metric_dict['document_type']),
                        processing_status=ProcessingStatus(metric_dict['processing_status']),
                        start_time=metric_dict['start_time'],
                        end_time=metric_dict['end_time'],
                        total_processing_time=metric_dict['total_processing_time'],
                        field_count=metric_dict['field_count'],
                        hitl_required=bool(metric_dict['hitl_required']),
                        hitl_reason=metric_dict['hitl_reason'],
                        exception_type=metric_dict['exception_type'],
                        confidence_scores=json.loads(metric_dict['confidence_scores']) if metric_dict['confidence_scores'] else {},
                        cost_breakdown=json.loads(metric_dict['cost_breakdown']) if metric_dict['cost_breakdown'] else {},
                        # Set default values for fields not stored in database
                        ocr_time=metric_dict['total_processing_time'] * 0.4,
                        extraction_time=metric_dict['total_processing_time'] * 0.3,
                        validation_time=metric_dict['total_processing_time'] * 0.2,
                        validation_errors=[]
                    )
                    
                    analytics_engine.add_document_metric(doc_metric)
                    loaded_count += 1
                    
                    print(f"✅ Loaded: {doc_metric.document_id} | {doc_metric.document_type.value} | {doc_metric.processing_status.value}")
                    
                except Exception as e:
                    print(f"❌ Error loading metric {metric_dict['document_id']}: {e}")
            
            print(f"📊 Successfully loaded {loaded_count}/{len(metrics_data)} metrics into analytics engine")
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Test KPI calculation
    print(f"\n🧮 Testing KPI Calculation...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        dashboard = analytics_engine.calculate_kpis(start_date, end_date)
        
        print(f"✅ KPI calculation successful!")
        print(f"   📊 Period: {dashboard.period_start.strftime('%Y-%m-%d')} to {dashboard.period_end.strftime('%Y-%m-%d')}")
        print(f"   📈 Overall Accuracy: {dashboard.overall_accuracy:.2%}")
        print(f"   🚀 STP Rate: {dashboard.stp_rate:.2%}")
        print(f"   ⚠️  Exception Rate: {dashboard.exception_rate:.2%}")
        print(f"   ⏱️  Latency P50: {dashboard.latency_p50:.2f}s")
        print(f"   💰 Cost per Document: ${dashboard.cost_per_document:.4f}")
        print(f"   📄 Throughput: {dashboard.throughput_docs_per_hour:.1f} docs/hour")
        
        return True
        
    except Exception as e:
        print(f"❌ KPI calculation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_current_database_status():
    """Show current database status."""
    
    print("\n📊 Current Database Status")
    print("=" * 60)
    
    db_path = Path("data/databases/dashboard.db")
    if not db_path.exists():
        print("❌ Dashboard database does not exist")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Document metrics
            cursor = conn.execute("SELECT COUNT(*) FROM document_metrics")
            doc_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT document_type) FROM document_metrics")
            type_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT processing_status) FROM document_metrics")
            status_count = cursor.fetchone()[0]
            
            print(f"📄 Document Metrics: {doc_count} records")
            print(f"📋 Document Types: {type_count} unique types")
            print(f"⚡ Processing Statuses: {status_count} unique statuses")
            
            # Field metrics
            cursor = conn.execute("SELECT COUNT(*) FROM field_metrics")
            field_count = cursor.fetchone()[0]
            print(f"🔍 Field Metrics: {field_count} records")
            
            # Recent documents
            cursor = conn.execute("""
                SELECT document_id, document_type, processing_status, total_processing_time 
                FROM document_metrics 
                ORDER BY created_at DESC 
                LIMIT 3
            """)
            
            print(f"\n📋 Recent Documents:")
            for row in cursor.fetchall():
                print(f"   {row[0]} | {row[1]} | {row[2]} | {row[3]}s")
            
    except Exception as e:
        print(f"❌ Database error: {e}")

def main():
    """Main test function."""
    
    print("🚀 Dashboard Real Data Integration Test")
    print("=" * 70)
    
    # Show current database status
    show_current_database_status()
    
    # Test data loading
    success = test_dashboard_data_loading()
    
    print("\n" + "=" * 70)
    print("🏆 TEST RESULTS")
    print("=" * 70)
    
    if success:
        print("✅ SUCCESS: Dashboard can load real data from database!")
        print("✅ SUCCESS: Analytics engine can calculate KPIs from real data!")
        print("\n🎯 What this means:")
        print("   • Real document processing data is saved to dashboard.db")
        print("   • Dashboard analytics engine can load and process this data")
        print("   • KPI calculations work with real processing metrics")
        print("   • Dashboard will display actual processing results")
        
        print("\n🔧 Next Steps:")
        print("   • Start the dashboard server to see real data in the web UI")
        print("   • Process more PDFs to see additional metrics")
        print("   • Data automatically updates every 5 minutes via background task")
        
        return True
    else:
        print("❌ FAILED: Dashboard data loading has issues")
        print("   Check the error messages above for troubleshooting")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)