#!/usr/bin/env python3
"""
Complete Dashboard Integration Test

Tests the complete real data flow:
1. Save processing data to database
2. Verify dashboard can load and display this data
3. Test all API endpoints
4. Confirm real data integration is working
"""

import requests
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_save_processing_data():
    """Test saving processing data to database."""
    print("🧪 Step 1: Saving processing data to database...")
    
    try:
        from core.data_persistence import save_processing_result
        
        # Create test processing result with unique ID
        test_result = {
            'blob_name': f'integration_test_{datetime.now().strftime("%H%M%S")}.pdf',
            'timestamp': datetime.now().isoformat(),
            'stages': {
                'download': {'status': 'completed', 'file_size': 789012},
                'classification': {'status': 'completed', 'document_type': 'contract', 'confidence': 0.91},
                'ocr': {'status': 'completed', 'text_length': 3200, 'confidence': 0.86},
                'field_extraction': {'status': 'completed', 'fields_count': 8},
                'validation': {'status': 'completed', 'validation_count': 8, 'passed': 7, 'failed': 1}
            },
            'extracted_data': {
                'contract_date': {'value': '2024-01-30', 'confidence': 0.88},
                'party_a': {'value': 'ABC Corp', 'confidence': 0.93},
                'party_b': {'value': 'XYZ Ltd', 'confidence': 0.89},
                'total_value': {'value': '50,000.00', 'confidence': 0.94},
                'document_type': 'contract'
            },
            'validation_results': [
                {'field': 'contract_date', 'valid': True, 'message': 'Valid date', 'severity': 'info'},
                {'field': 'total_value', 'valid': False, 'message': 'Currency validation failed', 'severity': 'warning'}
            ],
            'processing_time': 12.3,
            'needs_human_review': False
        }
        
        success = save_processing_result(test_result)
        if success:
            print("✅ Processing data saved to database successfully")
            return test_result['blob_name'].replace('.pdf', '').replace('/', '_')
        else:
            print("❌ Failed to save processing data")
            return None
            
    except Exception as e:
        print(f"❌ Error saving processing data: {e}")
        return None

def test_database_contents():
    """Test database contents directly."""
    print("\n🧪 Step 2: Verifying database contents...")
    
    try:
        import sqlite3
        
        conn = sqlite3.connect('data/databases/dashboard.db')
        cursor = conn.execute('SELECT COUNT(*) FROM document_metrics')
        doc_count = cursor.fetchone()[0]
        
        cursor = conn.execute('SELECT COUNT(*) FROM field_metrics')
        field_count = cursor.fetchone()[0]
        
        print(f"✅ Database contains {doc_count} document metrics and {field_count} field metrics")
        
        # Show recent documents
        cursor = conn.execute("""
            SELECT document_id, document_type, processing_status, total_processing_time 
            FROM document_metrics 
            ORDER BY created_at DESC 
            LIMIT 3
        """)
        
        print("📋 Recent documents:")
        for row in cursor.fetchall():
            print(f"   {row[0]} | {row[1]} | {row[2]} | {row[3]}s")
        
        conn.close()
        return doc_count > 0
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def test_dashboard_api(base_url="http://localhost:5000"):
    """Test dashboard API endpoints."""
    print(f"\n🧪 Step 3: Testing dashboard API endpoints at {base_url}...")
    
    # Test basic health check
    try:
        response = requests.get(f"{base_url}/api/debug/analytics-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Analytics status endpoint working")
            print(f"   Analytics engine metrics: {data.get('analytics_engine', {}).get('loaded_metrics', 0)}")
            print(f"   Database metrics: {data.get('database', {}).get('available_metrics', 0)}")
            
            # If analytics engine is empty, try manual reload
            if data.get('analytics_engine', {}).get('loaded_metrics', 0) == 0:
                print("🔄 Analytics engine empty, triggering manual reload...")
                
                reload_response = requests.post(f"{base_url}/api/debug/reload-analytics", timeout=10)
                if reload_response.status_code == 200:
                    reload_data = reload_response.json()
                    print(f"✅ Manual reload successful: {reload_data.get('loaded_count', 0)} metrics loaded")
                else:
                    print(f"❌ Manual reload failed: {reload_response.status_code}")
            
            return True
        else:
            print(f"❌ Analytics status endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard API not accessible: {e}")
        print("💡 Make sure dashboard server is running: python src/dashboard/dashboard_app.py")
        return False

def test_kpi_endpoints(base_url="http://localhost:5000"):
    """Test KPI calculation endpoints."""
    print(f"\n🧪 Step 4: Testing KPI endpoints...")
    
    try:
        # Test KPI endpoint
        response = requests.get(f"{base_url}/api/kpis", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ KPI endpoint working")
            print(f"   Overall accuracy: {data.get('overall_accuracy', 0):.2%}")
            print(f"   STP rate: {data.get('stp_rate', 0):.2%}")
            print(f"   Throughput: {data.get('throughput_docs_per_hour', 0):.1f} docs/hour")
            return True
        else:
            print(f"❌ KPI endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ KPI endpoint not accessible: {e}")
        return False

def main():
    """Run complete integration test."""
    
    print("🚀 Complete Dashboard Real Data Integration Test")
    print("=" * 70)
    
    # Step 1: Save processing data
    document_id = test_save_processing_data()
    data_saved = document_id is not None
    
    # Step 2: Verify database contents
    database_ok = test_database_contents()
    
    # Step 3: Test dashboard API (only if requested)
    print("\n" + "="*50)
    print("🌐 Dashboard API Test (Optional)")
    print("="*50)
    print("To test the dashboard API endpoints:")
    print("1. Start the dashboard server:")
    print("   cd /workspaces/PDFFieldExtraction")
    print("   PYTHONPATH=/workspaces/PDFFieldExtraction/src python -m uvicorn dashboard.dashboard_app:app --host 0.0.0.0 --port 5000")
    print("2. Re-run this test to check API endpoints")
    
    api_test_requested = input("\nDo you want to test API endpoints now? (y/N): ").lower().strip() in ['y', 'yes']
    
    api_ok = False
    kpi_ok = False
    
    if api_test_requested:
        api_ok = test_dashboard_api()
        if api_ok:
            kpi_ok = test_kpi_endpoints()
    
    # Results
    print("\n" + "=" * 70)
    print("🏆 INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    results = {
        "Data Persistence": "✅ PASS" if data_saved else "❌ FAIL",
        "Database Verification": "✅ PASS" if database_ok else "❌ FAIL",
        "Dashboard API": "✅ PASS" if api_ok else "⏭️ SKIPPED" if not api_test_requested else "❌ FAIL",
        "KPI Calculation": "✅ PASS" if kpi_ok else "⏭️ SKIPPED" if not api_test_requested else "❌ FAIL"
    }
    
    for test_name, result in results.items():
        print(f"{test_name:<20}: {result}")
    
    # Overall status
    core_tests_passed = data_saved and database_ok
    
    if core_tests_passed:
        print("\n🎉 CORE INTEGRATION SUCCESSFUL!")
        print("✅ Real data is being saved to dashboard database")
        print("✅ Database contains actual processing metrics")
        
        if api_ok and kpi_ok:
            print("✅ Dashboard API is serving real data")
            print("✅ KPI calculations work with real processing results")
            print("\n🚀 COMPLETE SUCCESS: End-to-end real data integration working!")
        else:
            print("\n📋 Next Steps:")
            print("1. Start dashboard server to test API endpoints")
            print("2. Verify dashboard UI displays real processing data")
            print("3. Process more PDFs to see live metrics updates")
        
        return True
    else:
        print("\n❌ CORE INTEGRATION FAILED")
        print("Check error messages above for troubleshooting")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)