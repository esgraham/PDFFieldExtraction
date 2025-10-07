"""
Data Persistence Layer

Saves processing pipeline results to the dashboard database for real-time monitoring
and analytics. Integrates with the pipeline_manager to persist document processing
metrics and field extraction performance data.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Data class for document processing metrics."""
    document_id: str
    document_type: str
    processing_status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_processing_time: Optional[float] = None
    field_count: Optional[int] = None
    hitl_required: bool = False
    hitl_reason: Optional[str] = None
    exception_type: Optional[str] = None
    cost_breakdown: Optional[Dict[str, float]] = None
    confidence_scores: Optional[Dict[str, float]] = None


@dataclass
class FieldMetrics:
    """Data class for field extraction performance metrics."""
    field_name: str
    document_type: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_predictions: int
    confidence_avg: float


class DashboardPersistenceManager:
    """Manages persistence of processing data to dashboard database."""
    
    def __init__(self, dashboard_db_path: Optional[str] = None):
        """
        Initialize persistence manager.
        
        Args:
            dashboard_db_path: Path to dashboard database. If None, uses default location.
        """
        if dashboard_db_path is None:
            # Default to production dashboard database
            current_dir = Path(__file__).parent.parent.parent
            dashboard_db_path = current_dir / "data" / "databases" / "dashboard.db"
        
        self.db_path = Path(dashboard_db_path)
        self._lock = threading.Lock()
        
        # Ensure database exists and has proper schema
        self._ensure_database_schema()
        
        logger.info(f"Dashboard persistence initialized: {self.db_path}")
    
    def _ensure_database_schema(self):
        """Ensure database exists with proper schema."""
        if not self.db_path.exists():
            logger.info("Dashboard database not found, creating...")
            self._create_schema()
        else:
            logger.info("Using existing dashboard database")
    
    def _create_schema(self):
        """Create database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fix for SQLite datetime adapter deprecation in Python 3.12+
        def adapt_datetime_iso(val):
            return val.isoformat()
        
        def convert_timestamp(val):
            return datetime.fromisoformat(val.decode())
        
        sqlite3.register_adapter(datetime, adapt_datetime_iso)
        sqlite3.register_converter("timestamp", convert_timestamp)
        
        with sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES) as conn:
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
            
            # Dashboard snapshots table
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
        
        logger.info("Dashboard database schema created")
    
    def save_processing_metrics(self, metrics: ProcessingMetrics) -> bool:
        """
        Save document processing metrics to database.
        
        Args:
            metrics: ProcessingMetrics object with processing data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    # Serialize complex data as JSON
                    cost_breakdown_json = json.dumps(metrics.cost_breakdown) if metrics.cost_breakdown else None
                    confidence_scores_json = json.dumps(metrics.confidence_scores) if metrics.confidence_scores else None
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO document_metrics 
                        (document_id, document_type, processing_status, start_time, end_time,
                         total_processing_time, field_count, hitl_required, hitl_reason,
                         exception_type, cost_breakdown, confidence_scores)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.document_id,
                        metrics.document_type,
                        metrics.processing_status,
                        metrics.start_time,
                        metrics.end_time,
                        metrics.total_processing_time,
                        metrics.field_count,
                        metrics.hitl_required,
                        metrics.hitl_reason,
                        metrics.exception_type,
                        cost_breakdown_json,
                        confidence_scores_json
                    ))
                    
                    conn.commit()
            
            logger.debug(f"Saved processing metrics for document: {metrics.document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processing metrics: {e}")
            return False
    
    def save_field_metrics(self, metrics: FieldMetrics) -> bool:
        """
        Save field extraction performance metrics to database.
        
        Args:
            metrics: FieldMetrics object with field performance data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                    conn.execute("""
                        INSERT INTO field_metrics 
                        (field_name, document_type, precision, recall, f1_score, 
                         accuracy, total_predictions, confidence_avg)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics.field_name,
                        metrics.document_type,
                        metrics.precision,
                        metrics.recall,
                        metrics.f1_score,
                        metrics.accuracy,
                        metrics.total_predictions,
                        metrics.confidence_avg
                    ))
                    
                    conn.commit()
            
            logger.debug(f"Saved field metrics for: {metrics.field_name} ({metrics.document_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save field metrics: {e}")
            return False
    
    def save_batch_processing_results(self, processing_results: List[Dict[str, Any]]) -> int:
        """
        Save a batch of processing results from pipeline.
        
        Args:
            processing_results: List of processing result dictionaries from pipeline_manager
            
        Returns:
            Number of successfully saved records
        """
        saved_count = 0
        
        for result in processing_results:
            try:
                # Extract metrics from processing result
                metrics = self._extract_processing_metrics(result)
                if self.save_processing_metrics(metrics):
                    saved_count += 1
                
                # Extract and save field metrics if available
                field_metrics_list = self._extract_field_performance_metrics(result)
                for field_metric in field_metrics_list:
                    self.save_field_metrics(field_metric)
                    
            except Exception as e:
                logger.error(f"Failed to process result for {result.get('blob_name', 'unknown')}: {e}")
        
        logger.info(f"Saved {saved_count}/{len(processing_results)} processing results to dashboard database")
        return saved_count
    
    def _extract_processing_metrics(self, processing_result: Dict[str, Any]) -> ProcessingMetrics:
        """Extract ProcessingMetrics from pipeline processing result."""
        
        # Generate document ID from blob name
        blob_name = processing_result.get('blob_name', 'unknown')
        document_id = blob_name.replace('.pdf', '').replace('/', '_')
        
        # Extract timestamps
        timestamp_str = processing_result.get('timestamp', datetime.now().isoformat())
        start_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
        
        processing_time = processing_result.get('processing_time', 0.0)
        end_time = start_time + timedelta(seconds=processing_time)
        
        # Determine processing status
        stages = processing_result.get('stages', {})
        failed_stages = [stage for stage, info in stages.items() if info.get('status') == 'failed']
        
        if failed_stages:
            processing_status = 'failed'
            exception_type = failed_stages[0]  # First failed stage
        elif processing_result.get('needs_human_review', False):
            processing_status = 'needs_review'
            exception_type = None
        else:
            processing_status = 'success'
            exception_type = None
        
        # Extract document type and field count
        classification_stage = stages.get('classification', {})
        document_type = classification_stage.get('document_type', 'unknown')
        
        field_extraction_stage = stages.get('field_extraction', {})
        field_count = field_extraction_stage.get('fields_count', 0)
        
        # Extract confidence scores from extracted data
        extracted_data = processing_result.get('extracted_data', {})
        confidence_scores = {}
        for field_name, field_data in extracted_data.items():
            if isinstance(field_data, dict) and 'confidence' in field_data:
                confidence_scores[field_name] = field_data['confidence']
        
        # Estimate costs (simplified)
        cost_breakdown = {
            'compute_cost': max(0.001, processing_time * 0.0001),  # Simple time-based cost
            'storage_cost': 0.0001,  # Fixed small storage cost
            'review_cost': 0.05 if processing_result.get('needs_human_review', False) else 0.0
        }
        
        # Determine HITL requirements
        hitl_required = processing_result.get('needs_human_review', False)
        hitl_reason = None
        if hitl_required:
            validation_results = processing_result.get('validation_results', [])
            failed_validations = [r for r in validation_results if not r.get('valid', True)]
            if failed_validations:
                hitl_reason = f"Validation failures: {len(failed_validations)} fields"
            else:
                hitl_reason = "Low confidence scores"
        
        return ProcessingMetrics(
            document_id=document_id,
            document_type=document_type,
            processing_status=processing_status,
            start_time=start_time,
            end_time=end_time,
            total_processing_time=processing_time,
            field_count=field_count,
            hitl_required=hitl_required,
            hitl_reason=hitl_reason,
            exception_type=exception_type,
            cost_breakdown=cost_breakdown,
            confidence_scores=confidence_scores
        )
    
    def _extract_field_performance_metrics(self, processing_result: Dict[str, Any]) -> List[FieldMetrics]:
        """Extract field performance metrics from processing result."""
        field_metrics = []
        
        try:
            # Get document type
            stages = processing_result.get('stages', {})
            classification_stage = stages.get('classification', {})
            document_type = classification_stage.get('document_type', 'unknown')
            
            # Extract field data
            extracted_data = processing_result.get('extracted_data', {})
            validation_results = processing_result.get('validation_results', [])
            
            # Create validation lookup
            validation_lookup = {
                r.get('field'): r for r in validation_results
            }
            
            # Calculate metrics for each field
            for field_name, field_data in extracted_data.items():
                if isinstance(field_data, dict) and 'confidence' in field_data:
                    confidence = field_data.get('confidence', 0.0)
                    validation_result = validation_lookup.get(field_name, {})
                    is_valid = validation_result.get('valid', True)
                    
                    # Simple metrics calculation (in a real system, this would be based on ground truth)
                    # For now, we'll use confidence as a proxy for performance
                    precision = min(0.95, confidence + 0.05) if is_valid else max(0.60, confidence - 0.20)
                    recall = min(0.95, confidence + 0.03) if is_valid else max(0.65, confidence - 0.15)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    accuracy = min(0.98, confidence + 0.08) if is_valid else max(0.70, confidence - 0.10)
                    
                    field_metrics.append(FieldMetrics(
                        field_name=field_name,
                        document_type=document_type,
                        precision=round(precision, 4),
                        recall=round(recall, 4),
                        f1_score=round(f1_score, 4),
                        accuracy=round(accuracy, 4),
                        total_predictions=1,  # This document contributed 1 prediction
                        confidence_avg=round(confidence, 4)
                    ))
            
        except Exception as e:
            logger.warning(f"Failed to extract field performance metrics: {e}")
        
        return field_metrics
    
    def get_recent_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get recent processing metrics for dashboard display.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with recent metrics summary
        """
        try:
            with sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES) as conn:
                # Get recent document metrics
                doc_metrics = conn.execute("""
                    SELECT COUNT(*) as total_docs,
                           AVG(total_processing_time) as avg_processing_time,
                           COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed,
                           COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed,
                           COUNT(CASE WHEN hitl_required = 1 THEN 1 END) as needs_review
                    FROM document_metrics 
                    WHERE created_at >= datetime('now', '-{} hours')
                """.format(hours)).fetchone()
                
                # Get field performance averages
                field_metrics = conn.execute("""
                    SELECT AVG(precision) as avg_precision,
                           AVG(recall) as avg_recall,
                           AVG(f1_score) as avg_f1,
                           AVG(accuracy) as avg_accuracy
                    FROM field_metrics 
                    WHERE created_at >= datetime('now', '-{} hours')
                """.format(hours)).fetchone()
                
                return {
                    'document_metrics': {
                        'total_documents': doc_metrics[0] or 0,
                        'avg_processing_time': round(doc_metrics[1] or 0, 2),
                        'completed': doc_metrics[2] or 0,
                        'failed': doc_metrics[3] or 0,
                        'needs_review': doc_metrics[4] or 0
                    },
                    'field_metrics': {
                        'avg_precision': round(field_metrics[0] or 0, 4),
                        'avg_recall': round(field_metrics[1] or 0, 4),
                        'avg_f1_score': round(field_metrics[2] or 0, 4),
                        'avg_accuracy': round(field_metrics[3] or 0, 4)
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return {
                'document_metrics': {},
                'field_metrics': {}
            }


# Global instance for easy access
_persistence_manager = None


def get_persistence_manager(dashboard_db_path: Optional[str] = None) -> DashboardPersistenceManager:
    """Get global persistence manager instance."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = DashboardPersistenceManager(dashboard_db_path)
    return _persistence_manager


def save_processing_result(processing_result: Dict[str, Any]) -> bool:
    """
    Convenience function to save processing result to dashboard database.
    
    Args:
        processing_result: Processing result dictionary from pipeline_manager
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        persistence_manager = get_persistence_manager()
        return persistence_manager.save_batch_processing_results([processing_result]) > 0
    except Exception as e:
        logger.error(f"Failed to save processing result: {e}")
        return False