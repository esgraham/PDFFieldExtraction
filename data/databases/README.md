# Dashboard Databases

This folder contains the SQLite databases used by the dashboard system.

## Database Files

### `dashboard_demo.db`
- **Purpose**: Contains sample data for demonstration and testing
- **Usage**: Used by `run_dashboard_demo.py` 
- **Data**: Pre-populated with 150+ document processing records and field metrics
- **Regeneration**: Can be recreated by running `setup_databases.py`

### `dashboard_production.db`
- **Purpose**: Production database for real application data
- **Usage**: Used by `dashboard_app.py` by default
- **Data**: Empty initially, populated by actual system usage
- **Backup**: Should be backed up regularly in production environments

## Database Schema

Both databases share the same schema:

### `document_metrics` Table
Stores processing metrics for individual documents:
- `document_id` - Unique document identifier
- `document_type` - Type of document (invoice, receipt, etc.)
- `processing_status` - Status (completed, failed, needs_review)
- `start_time`, `end_time` - Processing timestamps
- `total_processing_time` - Duration in seconds
- `field_count` - Number of extracted fields
- `hitl_required` - Whether human review is needed
- `exception_type` - Type of processing exception (if any)
- `cost_breakdown` - JSON with cost details
- `confidence_scores` - JSON with field confidence scores

### `field_metrics` Table
Stores field-level extraction quality metrics:
- `field_name` - Name of the extracted field
- `document_type` - Document type this metric applies to
- `precision`, `recall`, `f1_score`, `accuracy` - Quality metrics
- `total_predictions` - Number of predictions made
- `confidence_avg` - Average confidence score

### `dashboard_snapshots` Table
Caches dashboard data for performance:
- `template_type` - Dashboard view type (executive, operational, technical)
- `dashboard_data` - JSON snapshot of dashboard data
- `created_at` - Timestamp of snapshot

## Setup and Maintenance

### Initial Setup
```bash
# Create both databases with schema
python setup_databases.py
```

### Reset Demo Database
```bash
# Remove and recreate demo database with fresh sample data
rm dashboard_demo.db
python setup_databases.py
```

### Backup Production Database
```bash
# Create backup
cp dashboard_production.db dashboard_production_backup_$(date +%Y%m%d).db
```

## SQLite Datetime Handling

The databases use proper datetime handling to avoid Python 3.12+ deprecation warnings:
- Datetime objects are stored as ISO strings
- Automatic conversion on read/write using adapters and converters
- All connections use `PARSE_DECLTYPES` flag for proper type handling

## Usage in Code

### Using Demo Database
```python
# Set environment variable to use demo database
os.environ['DASHBOARD_DB_PATH'] = '/path/to/dashboard_demo.db'
db = DashboardDatabase()
```

### Using Production Database
```python
# Uses production database by default
db = DashboardDatabase()

# Or specify explicitly
db = DashboardDatabase('/path/to/dashboard_production.db')
```

### Custom Database Location
```python
# Use custom database path
db = DashboardDatabase('/custom/path/to/dashboard.db')
```