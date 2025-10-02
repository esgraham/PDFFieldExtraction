# Dashboard Real Data Configuration

The PDF Field Extraction Dashboard has been configured to use **real data** instead of sample data. This provides accurate metrics from actual document processing.

## 🔄 How It Works

### Data Sources
1. **Production Database**: `data/databases/dashboard_production.db`
2. **HITL System**: Imports completed tasks from `data/databases/enhanced_hitl.db`
3. **Processing Pipeline**: Direct integration with document processing components

### Startup Behavior
When the dashboard starts, it:
1. ✅ Attempts to import data from HITL database
2. ✅ Loads existing metrics from production database
3. ✅ Shows real processing statistics
4. ⚠️  Displays empty metrics if no real data exists

## 📊 Adding Real Data

### Method 1: Automatic Processing Integration
When documents are processed through the pipeline, metrics are automatically stored:
```python
from src.dashboard.dashboard_app import dashboard_db
from src.dashboard.dashboard_analytics import DocumentMetrics, DocumentType, ProcessingStatus

# Example: Store real processing result
metric = DocumentMetrics(
    document_id="DOC-001",
    document_type=DocumentType.INVOICE,
    processing_status=ProcessingStatus.SUCCESS,
    start_time=start_time,
    end_time=end_time,
    total_processing_time=processing_time,
    field_count=8,
    hitl_required=False,
    confidence_scores={"overall": 0.92}
)

dashboard_db.store_document_metric(metric)
```

### Method 2: HITL Data Import
Import completed HITL tasks:
```bash
# Via API endpoint
curl -X POST http://localhost:8000/api/import-hitl-data

# Or programmatically
from src.dashboard.dashboard_app import dashboard_db
imported_count = dashboard_db.import_hitl_data()
```

### Method 3: Manual Data Addition
Add sample real data for testing:
```bash
cd /workspaces/PDFFieldExtraction
python src/dashboard/add_processing_data.py --add-samples
```

## 🚀 Running the Dashboard

### Production Mode (Real Data)
```bash
cd /workspaces/PDFFieldExtraction/src/dashboard
python dashboard_app.py
```

### Demo Mode (Mixed Data)
```bash
cd /workspaces/PDFFieldExtraction/src/dashboard
python run_dashboard_demo.py
```

## 📈 Dashboard Features

### Real-Time Metrics
- **Document Processing Statistics**: Success rates, throughput, latency
- **Quality Metrics**: Accuracy, STP rates, confidence scores
- **Cost Analysis**: Processing costs, HITL review costs
- **Exception Tracking**: Error types, failure patterns

### Data Integration
- **HITL Integration**: Completed review tasks become metrics
- **Pipeline Integration**: Direct storage from processing components
- **Historical Analysis**: Long-term trend analysis
- **Real-Time Updates**: Live metrics as documents are processed

### API Endpoints
- `GET /api/kpis` - Real KPI dashboard data
- `GET /api/metrics/documents` - Document processing metrics
- `GET /api/metrics/realtime` - Real-time system metrics
- `POST /api/import-hitl-data` - Import HITL data
- `GET /api/health` - System health check

## 🔧 Configuration

### Database Paths
- **Production**: `data/databases/dashboard_production.db`
- **Demo**: `data/databases/dashboard_demo.db`
- **HITL Source**: `data/databases/enhanced_hitl.db`

### Environment Variables
```bash
# Optional: Override database path
DASHBOARD_DB_PATH=/path/to/custom/dashboard.db

# Optional: API keys for enhanced features
OPENAI_API_KEY=your_key_here
```

## 📋 Data Schema

### Document Metrics Table
```sql
CREATE TABLE document_metrics (
    id INTEGER PRIMARY KEY,
    document_id TEXT UNIQUE,
    document_type TEXT,           -- invoice, receipt, contract, etc.
    processing_status TEXT,       -- success, hitl_required, failed
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_processing_time REAL,
    field_count INTEGER,
    hitl_required BOOLEAN,
    hitl_reason TEXT,
    exception_type TEXT,
    cost_breakdown TEXT,          -- JSON string
    confidence_scores TEXT,       -- JSON string
    created_at TIMESTAMP
);
```

## 🎯 Migration from Sample Data

If you were previously using sample data:

1. **Stop the dashboard** if running
2. **Delete sample database** (optional): `rm data/databases/dashboard_demo.db`
3. **Add real data** using methods above
4. **Restart dashboard** - it will use production database

## 📊 Monitoring Real Data

### Check Database Status
```python
import sqlite3
with sqlite3.connect('data/databases/dashboard_production.db') as conn:
    count = conn.execute('SELECT COUNT(*) FROM document_metrics').fetchone()[0]
    print(f"Total document metrics: {count}")
```

### View Recent Metrics
```python
from src.dashboard.dashboard_app import dashboard_db
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=7)
metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
print(f"Metrics from last 7 days: {len(metrics)}")
```

## 🚨 Troubleshooting

### Empty Dashboard
If dashboard shows no data:
1. Check if production database exists and has data
2. Run HITL data import: `POST /api/import-hitl-data`
3. Add sample real data: `python src/dashboard/add_processing_data.py --add-samples`

### HITL Import Issues
- Ensure `enhanced_hitl.db` exists in `data/databases/`
- Check HITL tasks have status 'completed' or 'approved'
- Verify HITL database schema matches expected format

### Performance Issues
- Large datasets: Consider data retention policies
- Slow queries: Add database indexes if needed
- Memory usage: Monitor analytics engine memory footprint

---

🎉 **Your dashboard is now configured for real data!** It will show actual processing metrics instead of sample data, providing accurate insights into your PDF field extraction system's performance.