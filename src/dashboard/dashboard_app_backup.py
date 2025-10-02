"""
Dashboard Web Application

FastAPI-based web application for displaying KPIs, analytics, and summaries.
Includes real-time monitoring and interactive visua    def save_dashboard_snapshot(self, dashboard: KPIDashboard, template_type: str):
        \"\"\"Save dashboard snapshot for caching.\"\"\"
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute(\"\"\"
                INSERT INTO dashboard_snapshots 
                (period_start, period_end, snapshot_data, template_type)
                VALUES (?, ?, ?, ?)
            \"\"\", (
                dashboard.period_start,
                dashboard.period_end,
                json.dumps(dashboard, default=str),
                template_type
            ))
    
    def import_hitl_data(self, hitl_db_path: str = None):
        \"\"\"Import data from HITL database for dashboard metrics.\"\"\"
        if hitl_db_path is None:
            project_root = Path(__file__).parent.parent.parent
            hitl_db_path = str(project_root / \"data\" / \"databases\" / \"enhanced_hitl.db\")
        
        if not Path(hitl_db_path).exists():
            logger.warning(f\"HITL database not found at {hitl_db_path}\")
            return 0
        
        imported_count = 0
        try:
            with sqlite3.connect(hitl_db_path, detect_types=sqlite3.PARSE_DECLTYPES) as hitl_conn:
                hitl_conn.row_factory = sqlite3.Row
                
                # Get completed HITL tasks that can be converted to document metrics
                cursor = hitl_conn.execute(\"\"\"
                    SELECT * FROM review_tasks 
                    WHERE status IN ('completed', 'approved') 
                    ORDER BY created_at DESC
                \"\"\")
                
                hitl_tasks = [dict(row) for row in cursor.fetchall()]
                
                with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    for task in hitl_tasks:
                        # Convert HITL task to document metric
                        processing_time = 5.0  # Default processing time for HITL tasks
                        
                        # Check if already imported
                        existing = conn.execute(
                            \"SELECT COUNT(*) FROM document_metrics WHERE document_id = ?\",
                            (task['document_id'],)
                        ).fetchone()[0]
                        
                        if existing > 0:
                            continue
                        
                        conn.execute(\"\"\"
                            INSERT INTO document_metrics 
                            (document_id, document_type, processing_status, start_time, end_time,
                             total_processing_time, field_count, hitl_required, hitl_reason,
                             exception_type, cost_breakdown, confidence_scores)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        \"\"\", (
                            task['document_id'],
                            task['document_type'] or 'unknown',
                            'success' if task['status'] == 'completed' else 'hitl_required',
                            task['created_at'],
                            task['completed_at'] or task['created_at'],
                            processing_time,
                            len(json.loads(task['extracted_fields'] or '{}').keys()) if task['extracted_fields'] else 0,
                            True,  # All HITL tasks required human review
                            task['validation_errors'] or 'Required human review',
                            None,
                            json.dumps({'hitl_review_cost': 2.50}),  # Estimated HITL review cost
                            json.dumps({'overall': 0.75})  # Conservative confidence for HITL tasks
                        ))
                        imported_count += 1
                
                logger.info(f\"Imported {imported_count} HITL tasks as document metrics\")
                return imported_count
                
        except Exception as e:
            logger.error(f\"Error importing HITL data: {e}\")
            return 0"

from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging
from contextlib import asynccontextmanager

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

from dashboard_analytics import (
    AnalyticsEngine, KPIDashboard, JSONToMarkdownConverter, 
    PIIMaskingService, LLMSummaryService, DocumentMetrics, 
    FieldMetrics, DocumentType, ProcessingStatus
)

logger = logging.getLogger(__name__)

class DashboardDatabase:
    """Database for storing dashboard data and metrics."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to production database in data/databases folder
            project_root = Path(__file__).parent.parent.parent
            self.db_path = str(project_root / "data" / "databases" / "dashboard_production.db")
        else:
            self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
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
    
    def store_document_metric(self, metric: DocumentMetrics):
        """Store document processing metric."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO document_metrics 
                (document_id, document_type, processing_status, start_time, end_time,
                 total_processing_time, field_count, hitl_required, hitl_reason,
                 exception_type, cost_breakdown, confidence_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.document_id,
                metric.document_type.value,
                metric.processing_status.value,
                metric.start_time,
                metric.end_time,
                metric.total_processing_time,
                metric.field_count,
                metric.hitl_required,
                metric.hitl_reason,
                metric.exception_type,
                json.dumps(metric.cost_breakdown) if metric.cost_breakdown else None,
                json.dumps(metric.confidence_scores) if metric.confidence_scores else None
            ))
    
    def store_field_metric(self, metric: FieldMetrics):
        """Store field-level quality metric."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                INSERT INTO field_metrics 
                (field_name, document_type, precision, recall, f1_score, accuracy,
                 total_predictions, confidence_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.field_name,
                metric.document_type,
                metric.precision,
                metric.recall,
                metric.f1_score,
                metric.accuracy,
                metric.total_predictions,
                metric.confidence_avg
            ))
    
    def get_metrics_for_period(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get document metrics for specified period."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_metrics 
                WHERE start_time BETWEEN ? AND ?
                ORDER BY start_time DESC
            """, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
    
    def save_dashboard_snapshot(self, dashboard: KPIDashboard, template_type: str):
        """Save dashboard snapshot for caching."""
        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute("""
                INSERT INTO dashboard_snapshots 
                (period_start, period_end, snapshot_data, template_type)
                VALUES (?, ?, ?, ?)
            """, (
                dashboard.period_start,
                dashboard.period_end,
                json.dumps(dashboard, default=str),
                template_type
            ))

# Global instances
analytics_engine = AnalyticsEngine()
markdown_converter = JSONToMarkdownConverter()
pii_service = PIIMaskingService()
llm_service = LLMSummaryService()  # Initialize with API key if available
dashboard_db = DashboardDatabase()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup: Load real data from database
    logger.info("Starting dashboard application...")
    
    # Load existing real data from database
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Load last year of data
    
    try:
        existing_metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
        logger.info(f"Found {len(existing_metrics)} existing document metrics in database")
        
        # Load existing metrics into analytics engine
        for metric_dict in existing_metrics:
            # Convert database row back to DocumentMetrics object
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
        
        if len(existing_metrics) == 0:
            logger.warning("⚠️  No real data found in production database")
            logger.info("📊 Dashboard will show empty metrics until real documents are processed")
            logger.info("💡 To add sample data for testing, run: python data/databases/setup_databases.py")
        else:
            logger.info(f"✅ Loaded {len(existing_metrics)} real document processing metrics")
            
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")
        logger.info("📊 Dashboard will start with empty metrics")
    
    yield
    
    # Shutdown
    logger.info("Shutting down dashboard application...")

# Create FastAPI app
app = FastAPI(
    title="PDF Field Extraction Dashboard",
    description="Comprehensive analytics and KPI dashboard for PDF processing pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Templates and static files
templates = Jinja2Templates(directory="templates")

# Create templates directory if it doesn't exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Handle case where static directory doesn't exist

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard home page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "PDF Processing Dashboard",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.get("/api/kpis")
async def get_kpis(
    days: int = Query(7, description="Number of days to analyze"),
    template: str = Query("executive", description="Template type: executive, operational, technical")
):
    """Get KPI dashboard data."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Calculate KPIs
        dashboard = analytics_engine.calculate_kpis(start_date, end_date)
        
        # Save snapshot
        dashboard_db.save_dashboard_snapshot(dashboard, template)
        
        return {
            "dashboard": dashboard,
            "period_days": days,
            "template_type": template,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating KPIs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating KPIs: {str(e)}")

@app.get("/api/summary/{template_type}")
async def get_markdown_summary(
    template_type: str,
    days: int = Query(7, description="Number of days to analyze"),
    audience: str = Query("internal", description="Audience for PII masking: internal, restricted, public")
):
    """Get structured markdown summary."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Calculate KPIs
        dashboard = analytics_engine.calculate_kpis(start_date, end_date)
        
        # Convert to markdown
        markdown_summary = markdown_converter.convert_dashboard_to_markdown(dashboard, template_type)
        
        # Apply PII masking based on audience
        masked_summary = pii_service.mask_dashboard_summary(markdown_summary, audience)
        
        return {
            "summary": masked_summary,
            "template_type": template_type,
            "audience": audience,
            "period_days": days,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/api/llm-summary")
async def get_llm_summary(
    days: int = Query(7, description="Number of days to analyze"),
    template: str = Query("executive", description="Template type"),
    audience: str = Query("internal", description="Audience for PII masking")
):
    """Get LLM-enhanced summary."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Calculate KPIs
        dashboard = analytics_engine.calculate_kpis(start_date, end_date)
        
        # Convert to structured JSON for LLM
        dashboard_dict = {
            "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
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
        
        # Generate LLM summary
        llm_summary = llm_service.enhance_summary(dashboard_dict, template)
        
        # Apply PII masking
        masked_summary = pii_service.mask_pii_in_text(llm_summary, audience)
        
        return {
            "summary": masked_summary,
            "template_type": template,
            "audience": audience,
            "period_days": days,
            "llm_enhanced": True,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating LLM summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating LLM summary: {str(e)}")

@app.get("/api/metrics/documents")
async def get_document_metrics(
    days: int = Query(7, description="Number of days"),
    limit: int = Query(100, description="Maximum number of records")
):
    """Get recent document processing metrics."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
        
        return {
            "metrics": metrics[:limit],
            "total_count": len(metrics),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching document metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching document metrics: {str(e)}")

@app.get("/api/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time system metrics."""
    try:
        # Get last 24 hours of data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)
        
        dashboard = analytics_engine.calculate_kpis(start_date, end_date)
        
        # Return key real-time metrics
        return {
            "current_throughput": dashboard.throughput_docs_per_hour,
            "current_stp_rate": dashboard.stp_rate,
            "current_accuracy": dashboard.overall_accuracy,
            "poison_queue_count": dashboard.poison_queue_count,
            "recent_exceptions": dashboard.top_exceptions[:3],
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching real-time metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching real-time metrics: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "analytics_engine": "online",
            "database": "online",
            "pii_service": "online" if pii_service.analyzer else "limited",
            "llm_service": "online" if llm_service.client else "offline"
        }
    }

@app.get("/dashboard/executive", response_class=HTMLResponse)
async def executive_dashboard(request: Request):
    """Executive dashboard view."""
    return templates.TemplateResponse("executive_dashboard.html", {
        "request": request,
        "title": "Executive Dashboard",
        "dashboard_type": "executive"
    })

@app.get("/dashboard/operational", response_class=HTMLResponse)
async def operational_dashboard(request: Request):
    """Operational dashboard view."""
    return templates.TemplateResponse("operational_dashboard.html", {
        "request": request,
        "title": "Operational Dashboard",
        "dashboard_type": "operational"
    })

@app.get("/dashboard/technical", response_class=HTMLResponse)
async def technical_dashboard(request: Request):
    """Technical dashboard view."""
    return templates.TemplateResponse("technical_dashboard.html", {
        "request": request,
        "title": "Technical Dashboard",
        "dashboard_type": "technical"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")