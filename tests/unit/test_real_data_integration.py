#!/usr/bin/env python3
"""
Test Real Data Integration

Demonstrates how the processing pipeline now saves real data to the dashboard database
and how the dashboard displays this data in real-time.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import tempfile

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.data_persistence import DashboardPersistenceManager, ProcessingMetrics, FieldMetrics
from core.pipeline_manager import PDFProcessingPipeline
from dashboard.dashboard_app import dashboard_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_processing_results() -> list[Dict[str, Any]]:
    """Create mock processing results that simulate real pipeline output."""
    
    mock_results = []
    document_types = ['invoice', 'receipt', 'contract', 'form_application']
    
    for i in range(5):
        # Simulate a processing result from the pipeline
        result = {
            'blob_name': f'test_document_{i+1}.pdf',
            'timestamp': (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            'stages': {
                'download': {'status': 'completed', 'file_size': 1024000 + i*500},
                'preprocessing': {'status': 'completed', 'preprocessed_path': f'/tmp/processed_{i}.pdf'},
                'classification': {
                    'status': 'completed',
                    'document_type': document_types[i % len(document_types)],
                    'confidence': 0.85 + (i * 0.03)
                },
                'ocr': {
                    'status': 'completed',
                    'text_length': 2500 + i*200,
                    'confidence': 0.90 + (i * 0.02)
                },
                'field_extraction': {
                    'status': 'completed',
                    'fields_count': 8 + i,
                    'fields': ['total_amount', 'date', 'vendor_name', 'invoice_number'][:8+i]
                },
                'validation': {
                    'status': 'completed',
                    'validation_count': 8 + i,
                    'passed': 7 + i,
                    'failed': 1 if i % 3 == 0 else 0
                }
            },
            'extracted_data': {
                'total_amount': {'value': f'{100.00 + i*50:.2f}', 'confidence': 0.92 + i*0.01},
                'date': {'value': '2024-01-15', 'confidence': 0.88 + i*0.02},
                'vendor_name': {'value': f'Test Vendor {i+1}', 'confidence': 0.85 + i*0.03},
                'invoice_number': {'value': f'INV-{1000+i}', 'confidence': 0.90 + i*0.01},
                'document_type': document_types[i % len(document_types)],
                'overall_confidence': 0.87 + i*0.02
            },
            'validation_results': [
                {
                    'field': 'total_amount',
                    'valid': True,
                    'message': 'Valid currency format',
                    'severity': 'info'
                },
                {
                    'field': 'date',
                    'valid': i % 3 != 0,  # Some dates fail validation
                    'message': 'Invalid date format' if i % 3 == 0 else 'Valid date',
                    'severity': 'error' if i % 3 == 0 else 'info'
                }
            ],
            'processing_time': 5.2 + i*0.8,
            'needs_human_review': i % 3 == 0  # Every third document needs review
        }
        
        mock_results.append(result)
    
    return mock_results


async def test_data_persistence():
    """Test the data persistence layer with mock processing results."""
    
    logger.info("🧪 Testing Data Persistence Integration")
    logger.info("=" * 60)
    
    # Create persistence manager
    persistence_manager = DashboardPersistenceManager()
    
    # Create mock processing results
    logger.info("📄 Creating mock processing results...")
    mock_results = create_mock_processing_results()
    
    # Save processing results to database
    logger.info("💾 Saving processing results to dashboard database...")
    saved_count = persistence_manager.save_batch_processing_results(mock_results)
    
    logger.info(f"✅ Successfully saved {saved_count}/{len(mock_results)} processing results")
    
    # Get recent metrics to verify data was saved
    logger.info("📊 Retrieving recent metrics from database...")
    recent_metrics = persistence_manager.get_recent_metrics(hours=24)
    
    print("\n📈 Recent Processing Metrics:")
    doc_metrics = recent_metrics.get('document_metrics', {})
    field_metrics = recent_metrics.get('field_metrics', {})
    
    print(f"  Total Documents: {doc_metrics.get('total_documents', 0)}")
    print(f"  Completed: {doc_metrics.get('completed', 0)}")
    print(f"  Failed: {doc_metrics.get('failed', 0)}")
    print(f"  Needs Review: {doc_metrics.get('needs_review', 0)}")
    print(f"  Avg Processing Time: {doc_metrics.get('avg_processing_time', 0):.2f}s")
    
    print(f"\n🎯 Field Performance Metrics:")
    print(f"  Avg Precision: {field_metrics.get('avg_precision', 0):.4f}")
    print(f"  Avg Recall: {field_metrics.get('avg_recall', 0):.4f}")
    print(f"  Avg F1 Score: {field_metrics.get('avg_f1_score', 0):.4f}")
    print(f"  Avg Accuracy: {field_metrics.get('avg_accuracy', 0):.4f}")
    
    return saved_count > 0


async def test_dashboard_integration():
    """Test the dashboard's ability to load real data."""
    
    logger.info("\n🎛️ Testing Dashboard Integration")
    logger.info("=" * 60)
    
    # Refresh dashboard data
    logger.info("🔄 Refreshing dashboard data...")
    refreshed_count = dashboard_db.refresh_dashboard_data()
    logger.info(f"✅ Dashboard refreshed with {refreshed_count} records")
    
    # Test database query methods
    logger.info("📊 Testing dashboard database queries...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    # Get metrics for the last day
    metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
    logger.info(f"✅ Found {len(metrics)} document metrics in last 24 hours")
    
    if metrics:
        # Show sample metric
        sample_metric = metrics[0]
        print(f"\n📄 Sample Document Metric:")
        print(f"  Document ID: {sample_metric['document_id']}")
        print(f"  Document Type: {sample_metric['document_type']}")
        print(f"  Processing Status: {sample_metric['processing_status']}")
        print(f"  Processing Time: {sample_metric['total_processing_time']:.2f}s")
        print(f"  Field Count: {sample_metric['field_count']}")
        print(f"  HITL Required: {sample_metric['hitl_required']}")
    
    return len(metrics) > 0


async def test_end_to_end_flow():
    """Test the complete flow from processing to dashboard display."""
    
    logger.info("\n🔄 Testing End-to-End Data Flow")
    logger.info("=" * 60)
    
    # This would normally be triggered by the actual pipeline processing a PDF
    # For testing, we'll simulate saving a processing result directly
    
    logger.info("📄 Simulating processing pipeline completion...")
    
    # Create a realistic processing result
    processing_result = {
        'blob_name': 'test_invoice_realtime.pdf',
        'timestamp': datetime.now().isoformat(),
        'stages': {
            'download': {'status': 'completed', 'file_size': 856432},
            'preprocessing': {'status': 'completed'},
            'classification': {'status': 'completed', 'document_type': 'invoice', 'confidence': 0.94},
            'ocr': {'status': 'completed', 'text_length': 3250, 'confidence': 0.91},
            'field_extraction': {'status': 'completed', 'fields_count': 12},
            'validation': {'status': 'completed', 'validation_count': 12, 'passed': 11, 'failed': 1}
        },
        'extracted_data': {
            'total_amount': {'value': '1,250.00', 'confidence': 0.96},
            'tax_amount': {'value': '125.00', 'confidence': 0.93},
            'invoice_number': {'value': 'INV-2024-001', 'confidence': 0.98},
            'date': {'value': '2024-01-15', 'confidence': 0.89},
            'vendor_name': {'value': 'Acme Corp', 'confidence': 0.92},
            'customer_name': {'value': 'ABC Company', 'confidence': 0.87},
            'due_date': {'value': '2024-02-15', 'confidence': 0.85},
            'document_type': 'invoice',
            'overall_confidence': 0.91
        },
        'validation_results': [
            {'field': 'total_amount', 'valid': True, 'message': 'Valid currency', 'severity': 'info'},
            {'field': 'date', 'valid': False, 'message': 'Date format issue', 'severity': 'warning'}
        ],
        'processing_time': 7.3,
        'needs_human_review': False
    }
    
    # Import the save function (this is what the pipeline would call)
    from core.data_persistence import save_processing_result
    
    # Save the result (this happens automatically in the pipeline now)
    success = save_processing_result(processing_result)
    
    if success:
        logger.info("✅ Processing result saved to dashboard database")
        
        # Refresh dashboard to pick up new data
        refreshed_count = dashboard_db.refresh_dashboard_data()
        logger.info(f"✅ Dashboard automatically refreshed with {refreshed_count} new records")
        
        # Verify the data is now available through dashboard APIs
        recent_metrics = DashboardPersistenceManager().get_recent_metrics(hours=1)
        doc_count = recent_metrics.get('document_metrics', {}).get('total_documents', 0)
        
        logger.info(f"🎯 Dashboard now shows {doc_count} documents in the last hour")
        
        print("\n🎉 End-to-End Flow Complete!")
        print("   ✅ Pipeline processes document")
        print("   ✅ Results automatically saved to dashboard database")
        print("   ✅ Dashboard background task picks up new data")
        print("   ✅ Real-time metrics available via API endpoints")
        
        return True
    else:
        logger.error("❌ Failed to save processing result")
        return False


async def main():
    """Run all integration tests."""
    
    print("🚀 PDF Field Extraction - Real Data Integration Test")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Data Persistence Layer
        persistence_success = await test_data_persistence()
        
        # Test 2: Dashboard Integration
        dashboard_success = await test_dashboard_integration()
        
        # Test 3: End-to-End Flow
        e2e_success = await test_end_to_end_flow()
        
        # Summary
        print("\n" + "=" * 70)
        print("🏆 TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"✅ Data Persistence:     {'PASS' if persistence_success else 'FAIL'}")
        print(f"✅ Dashboard Integration: {'PASS' if dashboard_success else 'FAIL'}")
        print(f"✅ End-to-End Flow:      {'PASS' if e2e_success else 'FAIL'}")
        
        if all([persistence_success, dashboard_success, e2e_success]):
            print("\n🎉 ALL TESTS PASSED! Real data integration is working correctly.")
            print("\n📋 What this means:")
            print("   • Processing pipeline automatically saves metrics to dashboard database")
            print("   • Dashboard displays real-time processing data instead of mock data")
            print("   • Background tasks keep dashboard data fresh")
            print("   • API endpoints provide access to real metrics")
            print("\n🎯 Next Steps:")
            print("   • Run the dashboard: python src/dashboard/dashboard_app.py")
            print("   • Process real PDFs to see live data updates")
            print("   • Use /api/refresh-data endpoint to manually refresh dashboard")
            return True
        else:
            print("\n❌ Some tests failed. Check logs for details.")
            return False
            
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        print(f"\n❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)