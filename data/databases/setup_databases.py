#!/usr/bin/env python3
"""
Database Setup Script

Creates and initializes the dashboard databases:
1. dashboard_demo.db - Contains sample data for demonstration
2. dashboard_production.db - Empty database ready for production use
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import List, Dict, Any

# Fix for SQLite datetime adapter deprecation in Python 3.12+
def adapt_datetime_iso(val):
    """Adapt datetime to ISO string for SQLite storage."""
    return val.isoformat()

def convert_datetime(val):
    """Convert ISO string back to datetime from SQLite."""
    return datetime.fromisoformat(val.decode())

def convert_timestamp(val):
    """Convert timestamp string back to datetime from SQLite."""
    return datetime.fromisoformat(val.decode())

# Register the adapters and converters
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

def create_database_schema(db_path: str):
    """Create the database schema with all required tables."""
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        # Document metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE,
                document_type TEXT,
                processing_status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_processing_time REAL,
                field_count INTEGER,
                hitl_required BOOLEAN,
                hitl_reason TEXT,
                exception_type TEXT,
                cost_breakdown TEXT,
                confidence_scores TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Field metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS field_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field_name TEXT,
                document_type TEXT,
                precision REAL,
                recall REAL,
                f1_score REAL,
                accuracy REAL,
                total_predictions INTEGER,
                confidence_avg REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Dashboard snapshots table (for caching)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                snapshot_data TEXT,
                template_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print(f"✅ Database schema created: {db_path}")

def generate_demo_document_metrics(num_records: int = 100) -> List[Dict[str, Any]]:
    """Generate sample document metrics for demo purposes."""
    document_types = ['invoice', 'receipt', 'contract', 'id_document', 'general_document']
    processing_statuses = ['completed', 'failed', 'needs_review']
    exception_types = ['low_confidence', 'validation_error', 'ocr_failure', None]
    
    metrics = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        doc_type = random.choice(document_types)
        status = random.choice(processing_statuses)
        
        # Generate realistic processing times
        if doc_type == 'contract':
            processing_time = random.uniform(15.0, 45.0)  # Contracts take longer
        elif doc_type == 'invoice':
            processing_time = random.uniform(5.0, 15.0)
        else:
            processing_time = random.uniform(2.0, 10.0)
        
        start_time = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        end_time = start_time + timedelta(seconds=processing_time)
        
        # Generate cost breakdown
        cost_breakdown = {
            'compute_cost': round(random.uniform(0.001, 0.01), 4),
            'storage_cost': round(random.uniform(0.0001, 0.001), 4),
            'review_cost': round(random.uniform(0.0, 0.05), 4) if status == 'needs_review' else 0.0
        }
        
        # Generate confidence scores
        confidence_scores = {
            f'field_{j}': round(random.uniform(0.7, 0.99), 3)
            for j in range(random.randint(3, 12))
        }
        
        metrics.append({
            'document_id': f'DOC-{i+1:06d}',
            'document_type': doc_type,
            'processing_status': status,
            'start_time': start_time,
            'end_time': end_time,
            'total_processing_time': processing_time,
            'field_count': len(confidence_scores),
            'hitl_required': status == 'needs_review',
            'hitl_reason': 'Low confidence scores' if status == 'needs_review' else None,
            'exception_type': random.choice(exception_types) if status == 'failed' else None,
            'cost_breakdown': json.dumps(cost_breakdown),
            'confidence_scores': json.dumps(confidence_scores)
        })
    
    return metrics

def generate_demo_field_metrics(num_records: int = 50) -> List[Dict[str, Any]]:
    """Generate sample field metrics for demo purposes."""
    document_types = ['invoice', 'receipt', 'contract', 'id_document', 'general_document']
    field_names = [
        'total_amount', 'tax_amount', 'invoice_number', 'date', 'vendor_name',
        'customer_name', 'line_items', 'payment_terms', 'due_date', 'address',
        'phone_number', 'email', 'product_description', 'quantity', 'unit_price'
    ]
    
    metrics = []
    
    for doc_type in document_types:
        relevant_fields = random.sample(field_names, random.randint(8, 12))
        
        for field_name in relevant_fields:
            # Generate realistic performance metrics
            base_accuracy = random.uniform(0.85, 0.98)
            precision = base_accuracy + random.uniform(-0.05, 0.02)
            recall = base_accuracy + random.uniform(-0.03, 0.04)
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            metrics.append({
                'field_name': field_name,
                'document_type': doc_type,
                'precision': round(max(0.0, min(1.0, precision)), 4),
                'recall': round(max(0.0, min(1.0, recall)), 4),
                'f1_score': round(max(0.0, min(1.0, f1_score)), 4),
                'accuracy': round(base_accuracy, 4),
                'total_predictions': random.randint(50, 500),
                'confidence_avg': round(random.uniform(0.80, 0.95), 4)
            })
    
    return metrics

def populate_demo_database(db_path: str):
    """Populate the demo database with sample data."""
    print(f"🔄 Populating demo database: {db_path}")
    
    # Generate demo data
    doc_metrics = generate_demo_document_metrics(150)  # More data for better charts
    field_metrics = generate_demo_field_metrics(75)
    
    with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        # Insert document metrics
        for metric in doc_metrics:
            conn.execute("""
                INSERT INTO document_metrics 
                (document_id, document_type, processing_status, start_time, end_time,
                 total_processing_time, field_count, hitl_required, hitl_reason,
                 exception_type, cost_breakdown, confidence_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric['document_id'],
                metric['document_type'],
                metric['processing_status'],
                metric['start_time'],
                metric['end_time'],
                metric['total_processing_time'],
                metric['field_count'],
                metric['hitl_required'],
                metric['hitl_reason'],
                metric['exception_type'],
                metric['cost_breakdown'],
                metric['confidence_scores']
            ))
        
        # Insert field metrics
        for metric in field_metrics:
            conn.execute("""
                INSERT INTO field_metrics 
                (field_name, document_type, precision, recall, f1_score, 
                 accuracy, total_predictions, confidence_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric['field_name'],
                metric['document_type'],
                metric['precision'],
                metric['recall'],
                metric['f1_score'],
                metric['accuracy'],
                metric['total_predictions'],
                metric['confidence_avg']
            ))
        
        conn.commit()
    
    print(f"✅ Demo database populated with {len(doc_metrics)} document records and {len(field_metrics)} field records")

def main():
    """Main function to set up both databases."""
    databases_dir = Path(__file__).parent
    
    demo_db_path = databases_dir / "dashboard_demo.db"
    production_db_path = databases_dir / "dashboard.db"
    
    print("🚀 Setting up dashboard databases...")
    print("=" * 50)
    
    # Create demo database with sample data
    print("📊 Creating demo database...")
    if demo_db_path.exists():
        demo_db_path.unlink()  # Remove existing demo db
    
    create_database_schema(str(demo_db_path))
    populate_demo_database(str(demo_db_path))
    
    # Create production database (empty)
    print("\n🏭 Creating production database...")
    if production_db_path.exists():
        print(f"⚠️  Production database already exists: {production_db_path}")
        print("   Skipping creation to preserve existing data.")
    else:
        create_database_schema(str(production_db_path))
        print("✅ Empty production database created")
    
    print("\n🎯 Database setup complete!")
    print(f"📊 Demo database: {demo_db_path}")
    print(f"🏭 Production database: {production_db_path}")
    print("\nUsage:")
    print("- Demo database: Contains sample data for testing and demonstrations")
    print("- Production database: Empty database ready for real application data")

if __name__ == "__main__":
    main()