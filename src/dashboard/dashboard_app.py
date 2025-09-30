"""
Dashboard Web Application

FastAPI-based web application for displaying KPIs, analytics, and summaries.
Includes real-time monitoring and interactive visualizations.
"""

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

from dashboard_analytics import (
    AnalyticsEngine, KPIDashboard, JSONToMarkdownConverter, 
    PIIMaskingService, LLMSummaryService, DocumentMetrics, 
    FieldMetrics, generate_sample_metrics
)

logger = logging.getLogger(__name__)

class DashboardDatabase:
    """Database for storing dashboard data and metrics."""
    
    def __init__(self, db_path: str = "dashboard.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_metrics 
                WHERE start_time BETWEEN ? AND ?
                ORDER BY start_time DESC
            """, (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
    
    def save_dashboard_snapshot(self, dashboard: KPIDashboard, template_type: str):
        """Save dashboard snapshot for caching."""
        with sqlite3.connect(self.db_path) as conn:
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
    # Startup: Load sample data
    logger.info("Starting dashboard application...")
    
    # Generate and load sample data
    doc_metrics, field_metrics = generate_sample_metrics()
    
    for metric in doc_metrics:
        analytics_engine.add_document_metric(metric)
        dashboard_db.store_document_metric(metric)
    
    for metric in field_metrics:
        analytics_engine.add_field_metric(metric)
        dashboard_db.store_field_metric(metric)
    
    logger.info(f"Loaded {len(doc_metrics)} document metrics and {len(field_metrics)} field metrics")
    
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