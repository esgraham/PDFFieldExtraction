"""
Dashboard Web Application

FastAPI-based web application for displaying KPIs, analytics, and summaries.
Includes real-time monitoring and interactive visualizations.
"""

from fastapi import FastAPI, Request, HTTPException, Query, Depends, BackgroundTasks
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
import asyncio

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
            self.db_path = str(project_root / "data" / "databases" / "dashboard.db")
        else:
            self.db_path = db_path
    
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
    
    def import_hitl_data(self, hitl_db_path: str = None):
        """Import data from HITL database for dashboard metrics."""
        if hitl_db_path is None:
            project_root = Path(__file__).parent.parent.parent
            hitl_db_path = str(project_root / "data" / "databases" / "enhanced_hitl.db")
        
        if not Path(hitl_db_path).exists():
            logger.warning(f"HITL database not found at {hitl_db_path}")
            return 0
        
        imported_count = 0
        try:
            with sqlite3.connect(hitl_db_path, detect_types=sqlite3.PARSE_DECLTYPES) as hitl_conn:
                hitl_conn.row_factory = sqlite3.Row
                
                # Get completed HITL tasks that can be converted to document metrics
                cursor = hitl_conn.execute("""
                    SELECT * FROM review_tasks 
                    WHERE status IN ('completed', 'approved') 
                    ORDER BY created_at DESC
                """)
                
                hitl_tasks = [dict(row) for row in cursor.fetchall()]
                
                with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    for task in hitl_tasks:
                        # Convert HITL task to document metric
                        processing_time = 5.0  # Default processing time for HITL tasks
                        
                        # Check if already imported
                        existing = conn.execute(
                            "SELECT COUNT(*) FROM document_metrics WHERE document_id = ?",
                            (task['document_id'],)
                        ).fetchone()[0]
                        
                        if existing > 0:
                            continue
                        
                        conn.execute("""
                            INSERT INTO document_metrics 
                            (document_id, document_type, processing_status, start_time, end_time,
                             total_processing_time, field_count, hitl_required, hitl_reason,
                             exception_type, cost_breakdown, confidence_scores)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
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
                
                logger.info(f"Imported {imported_count} HITL tasks as document metrics")
                return imported_count
                
        except Exception as e:
            logger.error(f"Error importing HITL data: {e}")
            return 0
    
    def refresh_dashboard_data(self):
        """Refresh dashboard data from all available sources."""
        refreshed_count = 0
        
        try:
            # Import new HITL data
            hitl_count = self.import_hitl_data()
            refreshed_count += hitl_count
            
            # Check for new processing results from pipeline
            # (Pipeline automatically saves to our database via data_persistence.py)
            
            logger.info(f"Dashboard data refresh completed. {refreshed_count} new records imported.")
            return refreshed_count
            
        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {e}")
            return 0
    
    def manually_reload_analytics_engine(self):
        """Manually reload analytics engine with data from database."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            existing_metrics = self.get_metrics_for_period(start_date, end_date)
            logger.info(f"Manual reload: Found {len(existing_metrics)} metrics in database")
            
            # Clear and reload analytics engine (reference to global analytics_engine)
            from dashboard.dashboard_app import analytics_engine
            analytics_engine.document_metrics.clear()
            
            loaded_count = 0
            for metric_dict in existing_metrics:
                try:
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
                        ocr_time=metric_dict['total_processing_time'] * 0.4,
                        extraction_time=metric_dict['total_processing_time'] * 0.3,
                        validation_time=metric_dict['total_processing_time'] * 0.2,
                        validation_errors=[]
                    )
                    analytics_engine.add_document_metric(doc_metric)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load metric {metric_dict['document_id']}: {e}")
            
            logger.info(f"Manual reload complete: {loaded_count}/{len(existing_metrics)} metrics loaded into analytics engine")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error in manual analytics engine reload: {e}")
            return 0

# Global instances
analytics_engine = AnalyticsEngine()
markdown_converter = JSONToMarkdownConverter()
pii_service = PIIMaskingService()
llm_service = LLMSummaryService()  # Initialize with API key if available
dashboard_db = DashboardDatabase()

# Background task for data refresh
async def background_data_refresh():
    """Background task to refresh dashboard data every 5 minutes."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            logger.debug("Running background data refresh...")
            refreshed_count = dashboard_db.refresh_dashboard_data()
            if refreshed_count > 0:
                logger.info(f"Background refresh: {refreshed_count} new records")
        except Exception as e:
            logger.error(f"Background data refresh error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup: Load real data from database
    logger.info("Starting dashboard application...")
    
    # Try to import data from HITL system first
    hitl_imported = dashboard_db.import_hitl_data()
    if hitl_imported > 0:
        logger.info(f"✅ Imported {hitl_imported} HITL tasks as document metrics")
    
    # Start background data refresh task
    refresh_task = asyncio.create_task(background_data_refresh())
    logger.info("✅ Started background data refresh task")
    
    # Load existing real data from database
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
        
        # Log analytics engine status
        logger.info(f"📊 Analytics engine now has {len(analytics_engine.document_metrics)} metrics loaded")
        
        if len(existing_metrics) == 0:
            logger.warning("⚠️  No real data found in production database")
            logger.info("📊 Dashboard will show empty metrics until real documents are processed")
            logger.info("💡 To add sample data for testing, run: python data/databases/setup_databases.py")
        else:
            logger.info(f"✅ Loaded {len(existing_metrics)} real document processing metrics into analytics engine")
            
    except Exception as e:
        logger.error(f"Error loading existing data: {e}")
        logger.info("📊 Dashboard will start with empty metrics")
    
    yield
    
    # Shutdown
    logger.info("Shutting down dashboard application...")
    refresh_task.cancel()
    logger.info("✅ Cancelled background data refresh task")

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

@app.post("/api/import-hitl-data")
async def import_hitl_data():
    """Manually trigger import of HITL data into dashboard metrics."""
    try:
        imported_count = dashboard_db.import_hitl_data()
        
        # Reload analytics engine with new data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        existing_metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
        
        # Clear and reload analytics engine
        analytics_engine.document_metrics.clear()
        for metric_dict in existing_metrics:
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
                ocr_time=metric_dict['total_processing_time'] * 0.4,
                extraction_time=metric_dict['total_processing_time'] * 0.3,
                validation_time=metric_dict['total_processing_time'] * 0.2,
                validation_errors=[]
            )
            analytics_engine.add_document_metric(doc_metric)
        
        return {
            "success": True,
            "imported_count": imported_count,
            "total_metrics": len(existing_metrics),
            "message": f"Successfully imported {imported_count} HITL tasks as document metrics"
        }
    
    except Exception as e:
        logger.error(f"Error importing HITL data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error importing HITL data: {str(e)}")

@app.post("/api/refresh-data")
async def refresh_dashboard_data_endpoint():
    """Manually refresh dashboard data from all sources."""
    try:
        refreshed_count = dashboard_db.refresh_dashboard_data()
        
        # Reload analytics engine with updated data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        existing_metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
        
        # Clear and reload analytics engine
        analytics_engine.document_metrics.clear()
        for metric_dict in existing_metrics:
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
                ocr_time=metric_dict['total_processing_time'] * 0.4,
                extraction_time=metric_dict['total_processing_time'] * 0.3,
                validation_time=metric_dict['total_processing_time'] * 0.2,
                validation_errors=[]
            )
            analytics_engine.add_document_metric(doc_metric)
        
        return {
            "success": True, 
            "refreshed_count": refreshed_count,
            "total_metrics": len(existing_metrics),
            "message": f"Successfully refreshed {refreshed_count} new records from all data sources"
        }
    except Exception as e:
        logger.error(f"Error refreshing dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")

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

@app.get("/api/debug/analytics-status")
async def get_analytics_status():
    """Debug endpoint to check analytics engine and database status."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        database_metrics = dashboard_db.get_metrics_for_period(start_date, end_date)
        
        return {
            "success": True,
            "analytics_engine": {
                "loaded_metrics": len(analytics_engine.document_metrics),
                "loaded_field_metrics": len(analytics_engine.field_metrics)
            },
            "database": {
                "available_metrics": len(database_metrics),
                "database_path": str(dashboard_db.db_path),
                "database_exists": dashboard_db.db_path.exists()
            },
            "sample_documents": [
                {
                    "id": doc.document_id,
                    "type": doc.document_type.value,
                    "status": doc.processing_status.value,
                    "processing_time": doc.total_processing_time
                }
                for doc in analytics_engine.document_metrics[:3]
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/debug/reload-analytics")
async def reload_analytics_engine():
    """Manually reload analytics engine with database data."""
    try:
        loaded_count = dashboard_db.manually_reload_analytics_engine()
        
        return {
            "success": True,
            "loaded_count": loaded_count,
            "analytics_engine_metrics": len(analytics_engine.document_metrics),
            "message": f"Successfully reloaded {loaded_count} metrics into analytics engine"
        }
    except Exception as e:
        logger.error(f"Manual analytics reload failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")