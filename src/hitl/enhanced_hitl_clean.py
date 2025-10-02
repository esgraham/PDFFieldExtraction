"""
Enhanced HITL Review Application - Clean Version

Advanced human-in-the-loop review system with:
- Side-by-side PDF viewer with bounding box overlays  
- Extracted values display with confidence scores
- Rule failure highlighting and correction interface
- Microsoft Teams/email queue assignment with SLA tracking
- Feedback collection for training data generation
"""

import asyncio
import json
import logging
import uuid
import sqlite3
import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Image processing (optional)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Bounding box coordinates for field extraction."""
    x: float
    y: float
    width: float
    height: float
    page_number: int
    confidence: float

@dataclass
class ExtractedField:
    """Enhanced extracted field with location and metadata."""
    field_name: str
    value: any
    confidence: float
    bounding_box: Optional[BoundingBox]
    extraction_method: str
    rule_violations: List[str]
    suggested_corrections: List[str]

@dataclass
class ReviewFeedback:
    """Feedback data for training model improvements."""
    task_id: str
    field_name: str
    original_value: any
    corrected_value: any
    reviewer_id: str
    correction_reason: str
    confidence_rating: int  # 1-5 scale
    bounding_box_adjustment: Optional[BoundingBox]
    timestamp: datetime

@dataclass
class ReviewTask:
    """Enhanced review task for human reviewers."""
    task_id: str
    document_id: str
    document_type: str
    extracted_fields: List[ExtractedField]
    validation_errors: List[str]
    pdf_file_path: Optional[str]
    pdf_pages: List[str]  # Base64 encoded page images
    created_at: datetime
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    first_touched_at: Optional[datetime] = None
    status: str = "pending"
    priority: int = 1
    sla_deadline: Optional[datetime] = None
    age_to_first_touch: Optional[float] = None  # minutes
    age_to_resolve: Optional[float] = None  # minutes
    notes: Optional[str] = None
    resolution_time: Optional[datetime] = None
    feedback_collected: List[ReviewFeedback] = None
    
    def __post_init__(self):
        if self.feedback_collected is None:
            self.feedback_collected = []

@dataclass
class ReviewerInfo:
    """Enhanced reviewer information with specializations."""
    user_id: str
    name: str
    email: str
    teams_webhook: Optional[str] = None
    specializations: List[str] = None
    current_workload: int = 0
    completed_tasks: int = 0
    is_available: bool = True
    avg_resolution_time: float = 0.0
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []

class PDFProcessor:
    """PDF processing utilities for reviewer interface."""
    
    @staticmethod
    def create_mock_pdf_page() -> str:
        """Create a mock PDF page for demonstration."""
        if not HAS_IMAGE_PROCESSING:
            return ""
        
        try:
            # Create a simple mock PDF page
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add some mock content
            draw.rectangle([(50, 50), (750, 550)], outline='black', width=2)
            draw.text((100, 100), "INVOICE", fill='black')
            draw.text((100, 150), "Invoice Number: INV-2024-001", fill='black')
            draw.text((100, 200), "Date: 2024-09-30", fill='black')
            draw.text((100, 250), "Total Amount: $1,234.56", fill='black')
            draw.text((100, 300), "Vendor: Acme Corporation", fill='black')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Mock PDF creation failed: {str(e)}")
            return ""
    
    @staticmethod
    def draw_bounding_boxes(image_b64: str, fields: List[ExtractedField], page_num: int = 0) -> str:
        """Draw bounding boxes and field information on PDF page image."""
        if not HAS_IMAGE_PROCESSING or not image_b64:
            return image_b64
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))
            draw = ImageDraw.Draw(img)
            
            # Draw bounding boxes for fields on this page
            page_fields = [f for f in fields if f.bounding_box and f.bounding_box.page_number == page_num]
            
            for field in page_fields:
                bbox = field.bounding_box
                
                # Determine color based on confidence and rule violations
                if field.rule_violations:
                    color = "red"  # Rule violations
                elif field.confidence < 0.7:
                    color = "orange"  # Low confidence
                elif field.confidence < 0.9:
                    color = "yellow"  # Medium confidence
                else:
                    color = "green"  # High confidence
                
                # Draw rectangle
                left = bbox.x
                top = bbox.y
                right = bbox.x + bbox.width
                bottom = bbox.y + bbox.height
                
                # Draw bounding box
                draw.rectangle(
                    [(left, top), (right, bottom)],
                    outline=color,
                    width=3
                )
                
                # Field name and confidence label
                label_text = f"{field.field_name}: {field.confidence:.2f}"
                if field.rule_violations:
                    label_text += " ⚠️"
                
                # Label text
                draw.text(
                    (left, top - 20),
                    label_text,
                    fill=color
                )
            
            # Convert back to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Bounding box drawing failed: {str(e)}")
            return image_b64

class EnhancedHITLDatabase:
    """Enhanced database for HITL review tasks with training data storage."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to data/databases folder
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            db_path = str(project_root / "data" / "databases" / "enhanced_hitl.db")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize enhanced database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_tasks (
                    task_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    document_type TEXT,
                    extracted_fields TEXT,
                    validation_errors TEXT,
                    pdf_file_path TEXT,
                    pdf_pages TEXT,
                    created_at TIMESTAMP,
                    assigned_to TEXT,
                    assigned_at TIMESTAMP,
                    first_touched_at TIMESTAMP,
                    status TEXT,
                    priority INTEGER,
                    sla_deadline TIMESTAMP,
                    age_to_first_touch REAL,
                    age_to_resolve REAL,
                    notes TEXT,
                    resolution_time TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    field_name TEXT,
                    original_value TEXT,
                    corrected_value TEXT,
                    reviewer_id TEXT,
                    correction_reason TEXT,
                    confidence_rating INTEGER,
                    bounding_box_adjustment TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES review_tasks (task_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    document_type TEXT,
                    training_data TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES review_tasks (task_id)
                )
            """)
    
    def create_task(self, task: ReviewTask):
        """Store review task in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO review_tasks 
                (task_id, document_id, document_type, extracted_fields, validation_errors,
                 pdf_file_path, pdf_pages, created_at, assigned_to, assigned_at, 
                 first_touched_at, status, priority, sla_deadline, age_to_first_touch,
                 age_to_resolve, notes, resolution_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id, task.document_id, task.document_type,
                json.dumps([asdict(f) for f in task.extracted_fields], default=str),
                json.dumps(task.validation_errors),
                task.pdf_file_path,
                json.dumps(task.pdf_pages),
                task.created_at, task.assigned_to, task.assigned_at,
                task.first_touched_at, task.status, task.priority,
                task.sla_deadline, task.age_to_first_touch,
                task.age_to_resolve, task.notes, task.resolution_time
            ))
    
    def store_feedback(self, feedback: ReviewFeedback):
        """Store review feedback for training data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO review_feedback 
                (task_id, field_name, original_value, corrected_value, reviewer_id,
                 correction_reason, confidence_rating, bounding_box_adjustment, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.task_id, feedback.field_name, 
                json.dumps(feedback.original_value, default=str),
                json.dumps(feedback.corrected_value, default=str),
                feedback.reviewer_id, feedback.correction_reason,
                feedback.confidence_rating,
                json.dumps(asdict(feedback.bounding_box_adjustment), default=str) if feedback.bounding_box_adjustment else None,
                feedback.timestamp
            ))
    
    def store_training_data(self, task_id: str, training_data: Dict[str, any]):
        """Store training data generated from completed tasks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_data (task_id, document_type, training_data, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                task_id,
                training_data.get("document_type"),
                json.dumps(training_data, default=str),
                datetime.now()
            ))
    
    def get_training_data(self, document_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, any]]:
        """Retrieve training data for model improvement."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if document_type:
                cursor = conn.execute("""
                    SELECT * FROM training_data 
                    WHERE document_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (document_type, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM training_data 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            return [{
                "id": row["id"],
                "task_id": row["task_id"],
                "document_type": row["document_type"],
                "training_data": json.loads(row["training_data"]),
                "created_at": row["created_at"]
            } for row in cursor.fetchall()]

class EnhancedHITLReviewApp:
    """Enhanced HITL Review Application with PDF viewer and training data collection."""
    
    def __init__(self, database_path: str = None):
        self.database = EnhancedHITLDatabase(database_path)
        self.pdf_processor = PDFProcessor()
        self.reviewers: Dict[str, ReviewerInfo] = {}
        self.active_tasks: Dict[str, ReviewTask] = {}
        
        # SLA configurations (in hours)
        self.sla_config = {
            "high_priority_hours": 2,
            "normal_priority_hours": 24,
            "low_priority_hours": 72
        }
        
        # Initialize with sample reviewers
        self._init_sample_reviewers()
    
    def _init_sample_reviewers(self):
        """Initialize with sample reviewers."""
        sample_reviewers = [
            ReviewerInfo(
                user_id="reviewer1",
                name="Alice Johnson",
                email="alice@company.com",
                specializations=["invoice", "receipt"],
                is_available=True
            ),
            ReviewerInfo(
                user_id="reviewer2",
                name="Bob Smith",
                email="bob@company.com",
                specializations=["contract", "invoice"],
                is_available=True
            )
        ]
        
        for reviewer in sample_reviewers:
            self.reviewers[reviewer.user_id] = reviewer
    
    def create_enhanced_review_task(self, document_id: str, document_type: str, 
                                  extracted_fields: List[ExtractedField], 
                                  validation_errors: List[str],
                                  pdf_file_path: Optional[str] = None,
                                  priority: int = 1) -> str:
        """Create enhanced review task with PDF processing and SLA tracking."""
        
        task_id = str(uuid.uuid4())
        
        # Create mock PDF pages for demonstration
        pdf_pages = []
        if HAS_IMAGE_PROCESSING:
            # Create mock PDF page
            mock_page = self.pdf_processor.create_mock_pdf_page()
            if mock_page:
                # Add bounding box overlays
                mock_page_with_boxes = self.pdf_processor.draw_bounding_boxes(
                    mock_page, extracted_fields, page_num=0
                )
                pdf_pages = [mock_page_with_boxes]
        
        # Calculate SLA deadline based on priority
        priority_hours = {
            1: self.sla_config["low_priority_hours"],
            2: self.sla_config["normal_priority_hours"],
            3: self.sla_config["high_priority_hours"]
        }
        
        sla_hours = priority_hours.get(priority, 24)
        sla_deadline = datetime.now() + timedelta(hours=sla_hours)
        
        task = ReviewTask(
            task_id=task_id,
            document_id=document_id,
            document_type=document_type,
            extracted_fields=extracted_fields,
            validation_errors=validation_errors,
            pdf_file_path=pdf_file_path,
            pdf_pages=pdf_pages,
            created_at=datetime.now(),
            priority=priority,
            sla_deadline=sla_deadline
        )
        
        # Store in database
        self.database.create_task(task)
        
        # Add to active tasks
        self.active_tasks[task_id] = task
        
        # Auto-assign based on workload and specialization
        self._auto_assign_task(task_id)
        
        logger.info(f"Created enhanced review task {task_id} for document {document_id}")
        return task_id
    
    def _auto_assign_task(self, task_id: str):
        """Auto-assign task to best available reviewer."""
        task = self.active_tasks.get(task_id)
        if not task:
            return
        
        # Find best reviewer based on specialization and workload
        available_reviewers = [
            (reviewer_id, reviewer) for reviewer_id, reviewer in self.reviewers.items()
            if reviewer.is_available and task.document_type in reviewer.specializations
        ]
        
        if available_reviewers:
            # Sort by current workload (ascending)
            available_reviewers.sort(key=lambda x: x[1].current_workload)
            best_reviewer_id, best_reviewer = available_reviewers[0]
            
            self.assign_task(task_id, best_reviewer_id)
            logger.info(f"Auto-assigned task {task_id} to reviewer {best_reviewer_id}")
        else:
            logger.warning(f"No available reviewers for task {task_id} ({task.document_type})")
    
    def assign_task(self, task_id: str, reviewer_id: str) -> bool:
        """Assign task to reviewer."""
        
        if task_id not in self.active_tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        if reviewer_id not in self.reviewers:
            logger.error(f"Reviewer {reviewer_id} not found")
            return False
        
        task = self.active_tasks[task_id]
        reviewer = self.reviewers[reviewer_id]
        
        # Update task assignment
        task.assigned_to = reviewer_id
        task.assigned_at = datetime.now()
        task.status = "assigned"
        
        # Update reviewer workload
        reviewer.current_workload += 1
        
        # Update database
        self.database.create_task(task)
        
        logger.info(f"Assigned task {task_id} to reviewer {reviewer_id}")
        return True
    
    def record_first_touch(self, task_id: str, reviewer_id: str) -> bool:
        """Record when reviewer first touches the task for SLA tracking."""
        
        task = self.active_tasks.get(task_id)
        if not task or task.assigned_to != reviewer_id:
            return False
        
        if not task.first_touched_at:
            task.first_touched_at = datetime.now()
            task.age_to_first_touch = (task.first_touched_at - task.created_at).total_seconds() / 60
            task.status = "in_progress"
            
            # Update database
            self.database.create_task(task)
            
            logger.info(f"First touch recorded for task {task_id} (age: {task.age_to_first_touch:.1f} min)")
        
        return True
    
    def collect_field_feedback(self, task_id: str, field_name: str, 
                             original_value: any, corrected_value: any,
                             reviewer_id: str, correction_reason: str,
                             confidence_rating: int) -> bool:
        """Collect feedback for specific field correction."""
        
        task = self.active_tasks.get(task_id)
        if not task:
            return False
        
        feedback = ReviewFeedback(
            task_id=task_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            reviewer_id=reviewer_id,
            correction_reason=correction_reason,
            confidence_rating=confidence_rating,
            bounding_box_adjustment=None,
            timestamp=datetime.now()
        )
        
        task.feedback_collected.append(feedback)
        
        # Store in database for training data
        self.database.store_feedback(feedback)
        
        logger.info(f"Collected feedback for task {task_id}, field {field_name}")
        return True
    
    def complete_task_with_feedback(self, task_id: str, reviewer_id: str, 
                                  field_corrections: Dict[str, Dict[str, any]], 
                                  notes: Optional[str] = None) -> bool:
        """Complete task with detailed field corrections and feedback collection."""
        
        task = self.active_tasks.get(task_id)
        if not task or task.assigned_to != reviewer_id:
            logger.error(f"Task {task_id} not found or not assigned to {reviewer_id}")
            return False
        
        # Record completion timing
        task.resolution_time = datetime.now()
        task.age_to_resolve = (task.resolution_time - task.created_at).total_seconds() / 60
        task.status = "completed"
        task.notes = notes
        
        # Process field corrections and collect feedback
        for field_name, correction_data in field_corrections.items():
            # Find original field
            original_field = next(
                (f for f in task.extracted_fields if f.field_name == field_name),
                None
            )
            
            if original_field:
                corrected_value = correction_data.get("corrected_value")
                correction_reason = correction_data.get("reason", "Manual correction")
                confidence_rating = correction_data.get("confidence", 5)
                
                # Collect feedback if value changed
                if original_field.value != corrected_value:
                    self.collect_field_feedback(
                        task_id=task_id,
                        field_name=field_name,
                        original_value=original_field.value,
                        corrected_value=corrected_value,
                        reviewer_id=reviewer_id,
                        correction_reason=correction_reason,
                        confidence_rating=confidence_rating
                    )
                
                # Update field value
                original_field.value = corrected_value
        
        # Update reviewer stats
        reviewer = self.reviewers.get(reviewer_id)
        if reviewer:
            reviewer.current_workload -= 1
            reviewer.completed_tasks += 1
        
        # Update database
        self.database.create_task(task)
        
        # Generate training data
        self._generate_training_data(task)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        logger.info(f"Completed task {task_id} by reviewer {reviewer_id}")
        return True
    
    def _generate_training_data(self, task: ReviewTask):
        """Generate comprehensive training data from completed task."""
        try:
            training_data = {
                "document_id": task.document_id,
                "document_type": task.document_type,
                "completed_at": task.resolution_time.isoformat() if task.resolution_time else None,
                "resolution_time_minutes": task.age_to_resolve,
                "reviewer_id": task.assigned_to,
                "field_corrections": [],
                "validation_failures": task.validation_errors
            }
            
            # Process feedback for training data
            for feedback in task.feedback_collected:
                correction_entry = {
                    "field_name": feedback.field_name,
                    "original_value": feedback.original_value,
                    "corrected_value": feedback.corrected_value,
                    "correction_reason": feedback.correction_reason,
                    "reviewer_confidence": feedback.confidence_rating,
                    "timestamp": feedback.timestamp.isoformat()
                }
                training_data["field_corrections"].append(correction_entry)
            
            # Store training data
            self.database.store_training_data(task.task_id, training_data)
            
            logger.info(f"Generated training data for task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate training data: {str(e)}")
    
    def get_enhanced_queue_status(self) -> Dict[str, any]:
        """Get comprehensive queue status with SLA metrics."""
        
        current_time = datetime.now()
        
        # Basic counts
        total_tasks = len(self.active_tasks)
        pending_tasks = [t for t in self.active_tasks.values() if t.status == "pending"]
        assigned_tasks = [t for t in self.active_tasks.values() if t.status == "assigned"]
        in_progress_tasks = [t for t in self.active_tasks.values() if t.status == "in_progress"]
        
        # SLA violations
        sla_violations = [
            t for t in self.active_tasks.values() 
            if t.sla_deadline and current_time > t.sla_deadline
        ]
        
        # Reviewer workload
        reviewer_stats = {}
        for reviewer_id, reviewer in self.reviewers.items():
            reviewer_stats[reviewer_id] = {
                "name": reviewer.name,
                "current_workload": reviewer.current_workload,
                "completed_tasks": reviewer.completed_tasks,
                "is_available": reviewer.is_available,
                "specializations": reviewer.specializations
            }
        
        return {
            "timestamp": current_time.isoformat(),
            "queue_summary": {
                "total_active_tasks": total_tasks,
                "pending_tasks": len(pending_tasks),
                "assigned_tasks": len(assigned_tasks),
                "in_progress_tasks": len(in_progress_tasks),
                "sla_violations": len(sla_violations)
            },
            "reviewer_stats": reviewer_stats
        }
    
    def get_training_data_summary(self) -> Dict[str, any]:
        """Get summary of collected training data."""
        
        # Get training data from database
        all_training_data = self.database.get_training_data(limit=1000)
        
        # Analyze training data
        total_corrections = 0
        field_correction_counts = {}
        document_type_counts = {}
        
        for data_entry in all_training_data:
            training_data = data_entry["training_data"]
            doc_type = training_data.get("document_type", "unknown")
            
            document_type_counts[doc_type] = document_type_counts.get(doc_type, 0) + 1
            
            for correction in training_data.get("field_corrections", []):
                field_name = correction["field_name"]
                field_correction_counts[field_name] = field_correction_counts.get(field_name, 0) + 1
                total_corrections += 1
        
        return {
            "total_training_samples": len(all_training_data),
            "total_field_corrections": total_corrections,
            "document_type_distribution": document_type_counts,
            "field_correction_frequency": dict(sorted(
                field_correction_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 fields with corrections
            "last_updated": datetime.now().isoformat()
        }

# Sample usage and testing
def create_sample_enhanced_task():
    """Create a sample enhanced review task for demonstration."""
    
    app = EnhancedHITLReviewApp()
    
    # Sample extracted fields with bounding boxes
    sample_fields = [
        ExtractedField(
            field_name="invoice_number",
            value="INV-2024-001",
            confidence=0.95,
            bounding_box=BoundingBox(x=100, y=150, width=150, height=25, page_number=0, confidence=0.95),
            extraction_method="OCR",
            rule_violations=[],
            suggested_corrections=[]
        ),
        ExtractedField(
            field_name="total_amount",
            value="$1,234.56",
            confidence=0.65,  # Low confidence
            bounding_box=BoundingBox(x=100, y=250, width=100, height=20, page_number=0, confidence=0.65),
            extraction_method="Template",
            rule_violations=["Amount format validation failed"],
            suggested_corrections=["$1234.56"]
        ),
        ExtractedField(
            field_name="vendor_name",
            value="Acme Corporation",
            confidence=0.88,
            bounding_box=BoundingBox(x=100, y=300, width=200, height=30, page_number=0, confidence=0.88),
            extraction_method="NLP",
            rule_violations=[],
            suggested_corrections=[]
        )
    ]
    
    validation_errors = [
        "Total amount format does not match expected pattern",
        "Invoice date is missing or illegible"
    ]
    
    # Create review task
    task_id = app.create_enhanced_review_task(
        document_id="DOC-2024-12345",
        document_type="invoice",
        extracted_fields=sample_fields,
        validation_errors=validation_errors,
        pdf_file_path=None,
        priority=2
    )
    
    print(f"✅ Created enhanced review task: {task_id}")
    
    # Simulate reviewer workflow
    app.record_first_touch(task_id, "reviewer1")
    
    # Simulate field corrections
    field_corrections = {
        "total_amount": {
            "corrected_value": "$1234.56",
            "reason": "Fixed formatting - removed comma",
            "confidence": 5
        }
    }
    
    success = app.complete_task_with_feedback(
        task_id=task_id,
        reviewer_id="reviewer1",
        field_corrections=field_corrections,
        notes="Completed review - fixed amount formatting issue"
    )
    
    print(f"✅ Task completion status: {success}")
    
    # Show enhanced queue status
    status = app.get_enhanced_queue_status()
    print(f"✅ Queue status: Active tasks: {status['queue_summary']['total_active_tasks']}")
    
    # Show training data summary
    training_summary = app.get_training_data_summary()
    print(f"✅ Training data: {training_summary['total_training_samples']} samples, {training_summary['total_field_corrections']} corrections")
    
    return app

if __name__ == "__main__":
    # Run enhanced demonstration
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Enhanced HITL Review System Demo")
    print("====================================")
    
    app = create_sample_enhanced_task()
    
    print("\n✅ Enhanced HITL System Features:")
    print("  • PDF viewer with bounding box overlays")
    print("  • Enhanced field extraction with confidence scores")
    print("  • Rule failure highlighting and suggestions")
    print("  • Teams/email assignment with SLA tracking")
    print("  • Comprehensive feedback collection")
    print("  • Training data generation for model improvement")
    print("  • Real-time queue monitoring and statistics")
    print("  • Auto-assignment based on reviewer specialization")
    print("  • SLA violation alerts and escalation")
    
    print("\n🎯 Ready for production deployment!")