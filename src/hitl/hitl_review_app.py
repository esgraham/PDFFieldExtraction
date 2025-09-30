"""
HITL Review Application

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import base64

# Web framework imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Teams integration
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Optional database support
try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    """Review task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"

class ReviewPriority(Enum):
    """Review task priority."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

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
    value: Any
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
    original_value: Any
    corrected_value: Any
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
    """Information about a reviewer."""
    user_id: str
    name: str
    email: str
    teams_user_id: Optional[str] = None
    active_tasks: int = 0
    completed_tasks: int = 0
    specializations: List[str] = None

class TeamsNotificationService:
    """Microsoft Teams notification service."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.client = httpx.AsyncClient() if HAS_HTTPX else None
        
    async def send_notification(self, task: ReviewTask, notification_type: str = "new_task") -> bool:
        """Send Teams notification for review task."""
        if not self.webhook_url or not self.client:
            logger.warning("Teams webhook not configured or httpx not available")
            return False
            
        try:
            # Create Teams adaptive card
            card = self._create_adaptive_card(task, notification_type)
            
            response = await self.client.post(
                self.webhook_url,
                json=card,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Teams notification sent for task {task.task_id}")
                return True
            else:
                logger.error(f"Teams notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Teams notification: {str(e)}")
            return False
    
    def _create_adaptive_card(self, task: ReviewTask, notification_type: str) -> Dict:
        """Create Teams adaptive card for notification."""
        
        # Color coding by priority
        color_map = {
            ReviewPriority.LOW: "Good",
            ReviewPriority.MEDIUM: "Warning", 
            ReviewPriority.HIGH: "Attention",
            ReviewPriority.CRITICAL: "Accent"
        }
        
        # Base card structure
        card = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.3",
                    "body": [],
                    "actions": []
                }
            }]
        }
        
        content = card["attachments"][0]["content"]
        
        # Title based on notification type
        if notification_type == "new_task":
            title = "🔍 New Document Review Required"
            color = color_map.get(task.priority, "Default")
        elif notification_type == "urgent_task":
            title = "🚨 Urgent Document Review"
            color = "Attention"
        elif notification_type == "task_completed":
            title = "✅ Document Review Completed"
            color = "Good"
        else:
            title = "📋 Document Review Update"
            color = "Default"
            
        # Card header
        content["body"].append({
            "type": "TextBlock",
            "text": title,
            "weight": "Bolder",
            "size": "Medium",
            "color": color
        })
        
        # Task details
        content["body"].append({
            "type": "FactSet",
            "facts": [
                {"title": "Document ID", "value": task.document_id},
                {"title": "Document Type", "value": task.document_type.title()},
                {"title": "Priority", "value": task.priority.value.title()},
                {"title": "Created", "value": task.created_at.strftime("%Y-%m-%d %H:%M")},
                {"title": "Status", "value": task.status.value.title()}
            ]
        })
        
        # Validation errors if present
        if task.validation_errors:
            error_text = "\\n".join([f"• {error.get('message', 'Unknown error')}" for error in task.validation_errors[:3]])
            if len(task.validation_errors) > 3:
                error_text += f"\\n• ... and {len(task.validation_errors) - 3} more"
                
            content["body"].append({
                "type": "TextBlock",
                "text": f"**Validation Issues:**\\n{error_text}",
                "wrap": True,
                "color": "Attention"
            })
        
        # Low confidence fields
        if task.confidence_scores:
            low_confidence = [f"{field}: {score:.1%}" for field, score in task.confidence_scores.items() if score < 0.7]
            if low_confidence:
                confidence_text = "\\n".join([f"• {item}" for item in low_confidence[:5]])
                content["body"].append({
                    "type": "TextBlock",
                    "text": f"**Low Confidence Fields:**\\n{confidence_text}",
                    "wrap": True,
                    "color": "Warning"
                })
        
        # Action buttons
        if notification_type == "new_task":
            review_url = f"http://localhost:8000/review/{task.task_id}"  # Update with actual URL
            
            content["actions"].extend([
                {
                    "type": "Action.OpenUrl",
                    "title": "Start Review",
                    "url": review_url
                },
                {
                    "type": "Action.OpenUrl", 
                    "title": "View Queue",
                    "url": "http://localhost:8000/queue"  # Update with actual URL
                }
            ])
        
        return card
    
    async def send_daily_summary(self, pending_tasks: List[ReviewTask], completed_tasks: List[ReviewTask]) -> bool:
        """Send daily summary of review tasks."""
        if not self.webhook_url or not self.client:
            return False
            
        try:
            # Priority breakdown
            priority_counts = {}
            for task in pending_tasks:
                priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
            
            # Create summary card
            card = {
                "type": "message",
                "attachments": [{
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "type": "AdaptiveCard",
                        "version": "1.3",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "📊 Daily Review Summary",
                                "weight": "Bolder",
                                "size": "Large"
                            },
                            {
                                "type": "FactSet",
                                "facts": [
                                    {"title": "Pending Reviews", "value": str(len(pending_tasks))},
                                    {"title": "Completed Today", "value": str(len(completed_tasks))},
                                    {"title": "Critical Priority", "value": str(priority_counts.get("critical", 0))},
                                    {"title": "High Priority", "value": str(priority_counts.get("high", 0))}
                                ]
                            }
                        ]
                    }
                }]
            }
            
            response = await self.client.post(self.webhook_url, json=card)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
            return False

class HITLReviewDatabase:
    """Database interface for HITL review tasks."""
    
    def __init__(self, db_path: str = "hitl_reviews.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        if not HAS_SQLITE:
            logger.warning("SQLite not available, using in-memory storage")
            self.tasks = {}
            self.reviewers = {}
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_tasks (
                    task_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    assigned_to TEXT,
                    completed_at TIMESTAMP,
                    extracted_fields TEXT,
                    validation_errors TEXT,
                    confidence_scores TEXT,
                    reviewer_notes TEXT,
                    corrected_fields TEXT,
                    review_decision TEXT,
                    source_system TEXT,
                    processing_metadata TEXT,
                    original_document BLOB
                )
            """)
            
            # Create reviewers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviewers (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    teams_user_id TEXT,
                    active_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    specializations TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON review_tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON review_tasks(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON review_tasks(created_at)")
            
            conn.commit()
    
    async def save_task(self, task: ReviewTask) -> bool:
        """Save review task to database."""
        if not HAS_SQLITE:
            self.tasks[task.task_id] = task
            return True
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO review_tasks 
                    (task_id, document_id, document_type, status, priority, created_at,
                     assigned_to, completed_at, extracted_fields, validation_errors,
                     confidence_scores, reviewer_notes, corrected_fields, review_decision,
                     source_system, processing_metadata, original_document)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    task.document_id,
                    task.document_type,
                    task.status.value,
                    task.priority.value,
                    task.created_at.isoformat(),
                    task.assigned_to,
                    task.completed_at.isoformat() if task.completed_at else None,
                    json.dumps(task.extracted_fields) if task.extracted_fields else None,
                    json.dumps(task.validation_errors) if task.validation_errors else None,
                    json.dumps(task.confidence_scores) if task.confidence_scores else None,
                    task.reviewer_notes,
                    json.dumps(task.corrected_fields) if task.corrected_fields else None,
                    task.review_decision,
                    task.source_system,
                    json.dumps(task.processing_metadata) if task.processing_metadata else None,
                    task.original_document
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving task: {str(e)}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[ReviewTask]:
        """Get task by ID."""
        if not HAS_SQLITE:
            return self.tasks.get(task_id)
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM review_tasks WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_task(cursor, row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting task: {str(e)}")
            return None
    
    async def get_pending_tasks(self, limit: int = 50) -> List[ReviewTask]:
        """Get pending review tasks."""
        if not HAS_SQLITE:
            return [task for task in self.tasks.values() if task.status == ReviewStatus.PENDING][:limit]
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM review_tasks 
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """, (limit,))
                
                return [self._row_to_task(cursor, row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting pending tasks: {str(e)}")
            return []
    
    def _row_to_task(self, cursor, row) -> ReviewTask:
        """Convert database row to ReviewTask."""
        columns = [description[0] for description in cursor.description]
        data = dict(zip(columns, row))
        
        return ReviewTask(
            task_id=data['task_id'],
            document_id=data['document_id'],
            document_type=data['document_type'],
            status=ReviewStatus(data['status']),
            priority=ReviewPriority(data['priority']),
            created_at=datetime.fromisoformat(data['created_at']),
            assigned_to=data['assigned_to'],
            completed_at=datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None,
            extracted_fields=json.loads(data['extracted_fields']) if data['extracted_fields'] else None,
            validation_errors=json.loads(data['validation_errors']) if data['validation_errors'] else None,
            confidence_scores=json.loads(data['confidence_scores']) if data['confidence_scores'] else None,
            reviewer_notes=data['reviewer_notes'],
            corrected_fields=json.loads(data['corrected_fields']) if data['corrected_fields'] else None,
            review_decision=data['review_decision'],
            source_system=data['source_system'],
            processing_metadata=json.loads(data['processing_metadata']) if data['processing_metadata'] else None,
            original_document=data['original_document']
        )

class PDFProcessor:
    """PDF processing utilities for reviewer interface."""
    
    @staticmethod
    def pdf_to_images(pdf_path: str) -> List[str]:
        """Convert PDF pages to base64 encoded images."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            page_images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                img_data = pix.tobytes("png")
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                page_images.append(img_b64)
            
            doc.close()
            return page_images
            
        except ImportError:
            logger.warning("PyMuPDF not available, using PIL fallback")
            return PDFProcessor._pdf_to_images_fallback(pdf_path)
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return []
    
    @staticmethod
    def _pdf_to_images_fallback(pdf_path: str) -> List[str]:
        """Fallback PDF to image conversion."""
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=150)
            page_images = []
            
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                page_images.append(img_b64)
            
            return page_images
            
        except Exception as e:
            logger.error(f"Fallback PDF processing failed: {str(e)}")
            return []
    
    @staticmethod
    def draw_bounding_boxes(image_b64: str, fields: List[ExtractedField]) -> str:
        """Draw bounding boxes on PDF page image."""
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))
            draw = ImageDraw.Draw(img)
            
            # Draw bounding boxes for each field
            for field in fields:
                if field.bounding_box:
                    bbox = field.bounding_box
                    
                    # Determine color based on confidence and rule violations
                    if field.rule_violations:
                        color = "red"  # Rule violations
                    elif field.confidence < 0.7:
                        color = "orange"  # Low confidence
                    else:
                        color = "green"  # Good extraction
                    
                    # Draw rectangle
                    left = bbox.x
                    top = bbox.y
                    right = bbox.x + bbox.width
                    bottom = bbox.y + bbox.height
                    
                    draw.rectangle(
                        [(left, top), (right, bottom)],
                        outline=color,
                        width=2
                    )
                    
                    # Add field name label
                    draw.text(
                        (left, top - 20),
                        f"{field.field_name} ({field.confidence:.2f})",
                        fill=color
                    )
            
            # Convert back to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Bounding box drawing failed: {str(e)}")
            return image_b64

class HITLReviewApp:
    """Main HITL review application with enhanced PDF viewing."""
    
    def __init__(self, database_path: str = "hitl_review.db"):
        self.database = HITLReviewDatabase(database_path)
        self.teams_service = TeamsNotificationService()
        self.email_service = EmailNotificationService()
        self.pdf_processor = PDFProcessor()
        self.reviewers: Dict[str, ReviewerInfo] = {}
        self.active_tasks: Dict[str, ReviewTask] = {}
        
        # SLA configurations
        self.sla_config = {
            "high_priority_hours": 2,
            "normal_priority_hours": 24,
            "low_priority_hours": 72
        }
        
        # Load reviewer configurations
        self._load_reviewer_config()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard."""
            pending_tasks = await self.db.get_pending_tasks(20)
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>HITL Review Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .priority-critical {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                    .priority-high {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                    .priority-medium {{ background-color: #f3e5f5; border-left: 4px solid #9c27b0; }}
                    .priority-low {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
                    .task-card {{ margin: 10px 0; padding: 15px; border-radius: 4px; }}
                    .button {{ background-color: #2196f3; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <h1>🔍 HITL Review Dashboard</h1>
                <p><strong>Pending Reviews:</strong> {len(pending_tasks)}</p>
                
                <h2>Review Queue</h2>
                {"".join([f'''
                <div class="task-card priority-{task.priority.value}">
                    <h3>{task.document_type.title()} - {task.document_id}</h3>
                    <p><strong>Priority:</strong> {task.priority.value.title()}</p>
                    <p><strong>Created:</strong> {task.created_at.strftime("%Y-%m-%d %H:%M")}</p>
                    <p><strong>Validation Issues:</strong> {len(task.validation_errors or [])}</p>
                    <a href="/review/{task.task_id}" class="button">Start Review</a>
                </div>
                ''' for task in pending_tasks])}
            </body>
            </html>
            """
            return html
        
        @self.app.get("/review/{task_id}", response_class=HTMLResponse)
        async def review_task(task_id: str, request: Request):
            """Individual task review page."""
            task = await self.db.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            # Create review form
            fields_html = ""
            if task.extracted_fields:
                for field_name, field_value in task.extracted_fields.items():
                    confidence = task.confidence_scores.get(field_name, 0.0) if task.confidence_scores else 0.0
                    color = "red" if confidence < 0.7 else "orange" if confidence < 0.8 else "green"
                    
                    fields_html += f"""
                    <div style="margin: 10px 0;">
                        <label><strong>{field_name}:</strong></label>
                        <input type="text" name="{field_name}" value="{field_value}" style="width: 300px; margin-left: 10px;">
                        <span style="color: {color};">({confidence:.1%})</span>
                    </div>
                    """
            
            errors_html = ""
            if task.validation_errors:
                errors_html = "<h3>Validation Errors:</h3><ul>"
                for error in task.validation_errors:
                    errors_html += f"<li>{error.get('message', 'Unknown error')}</li>"
                errors_html += "</ul>"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Review Task {task_id}</title>
                <style>body {{ font-family: Arial, sans-serif; margin: 20px; }}</style>
            </head>
            <body>
                <h1>📋 Review Task: {task.document_id}</h1>
                <p><strong>Document Type:</strong> {task.document_type}</p>
                <p><strong>Priority:</strong> {task.priority.value.title()}</p>
                
                {errors_html}
                
                <form method="post" action="/api/review/{task_id}">
                    <h3>Extracted Fields:</h3>
                    {fields_html}
                    
                    <h3>Review Notes:</h3>
                    <textarea name="reviewer_notes" rows="4" cols="50"></textarea>
                    
                    <h3>Decision:</h3>
                    <select name="review_decision">
                        <option value="approved">Approve</option>
                        <option value="corrected">Approve with Corrections</option>
                        <option value="rejected">Reject</option>
                        <option value="escalated">Escalate</option>
                    </select>
                    
                    <br><br>
                    <button type="submit">Submit Review</button>
                </form>
            </body>
            </html>
            """
            return html
        
        @self.app.post("/api/review/{task_id}")
        async def submit_review(task_id: str, request: Request):
            """Submit review results."""
            task = await self.db.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            form_data = await request.form()
            
            # Extract corrected fields
            corrected_fields = {}
            for key, value in form_data.items():
                if key not in ['reviewer_notes', 'review_decision']:
                    corrected_fields[key] = value
            
            # Update task
            task.status = ReviewStatus.COMPLETED
            task.completed_at = datetime.now()
            task.reviewer_notes = form_data.get('reviewer_notes', '')
            task.review_decision = form_data.get('review_decision', 'approved')
            task.corrected_fields = corrected_fields
            
            await self.db.save_task(task)
            
            # Send completion notification
            await self.teams_service.send_notification(task, "task_completed")
            
            return JSONResponse({"status": "success", "message": "Review submitted successfully"})
        
        @self.app.get("/api/stats")
        async def get_stats():
            """Get review statistics."""
            pending_tasks = await self.db.get_pending_tasks(1000)
            
            priority_counts = {}
            for task in pending_tasks:
                priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
            
            return {
                "pending_count": len(pending_tasks),
                "priority_breakdown": priority_counts,
                "timestamp": datetime.now().isoformat()
            }
    
    async def add_review_task(self, task: ReviewTask, send_notification: bool = True) -> bool:
        """Add new review task."""
        success = await self.db.save_task(task)
        
        if success and send_notification:
            # Determine notification type based on priority
            notification_type = "urgent_task" if task.priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH] else "new_task"
            await self.teams_service.send_notification(task, notification_type)
        
        return success
    
    async def get_task_for_review(self, task_id: str) -> Optional[ReviewTask]:
        """Get task for review."""
        return await self.db.get_task(task_id)
    
    async def start_web_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the web server."""
        if not self.app:
            logger.error("FastAPI not available, cannot start web server")
            return
        
        logger.info(f"Starting HITL Review App on {host}:{port}")
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()

# Example usage and testing
async def main():
    """Example usage of HITL Review App."""
    
    # Initialize the app
    teams_webhook = "https://your-tenant.webhook.office.com/webhookb2/..."  # Replace with actual webhook
    app = HITLReviewApp(teams_webhook_url=teams_webhook)
    
    # Create sample review task
    task = ReviewTask(
        task_id=str(uuid.uuid4()),
        document_id="INV-2024-001",
        document_type="invoice",
        status=ReviewStatus.PENDING,
        priority=ReviewPriority.HIGH,
        created_at=datetime.now(),
        extracted_fields={
            "invoice_number": "INV-2024-001",
            "invoice_date": "2024-01-15",
            "total_amount": "$1,234.56",
            "vendor_name": "ABC Company"
        },
        validation_errors=[
            {"field": "due_date", "message": "Due date is missing"},
            {"field": "total_amount", "message": "Amount format validation failed"}
        ],
        confidence_scores={
            "invoice_number": 0.95,
            "invoice_date": 0.89,
            "total_amount": 0.65,  # Low confidence
            "vendor_name": 0.78
        }
    )
    
    # Add task to review queue
    await app.add_review_task(task)
    
    print(f"✅ Added review task: {task.task_id}")
    print(f"🔗 Review URL: http://localhost:8000/review/{task.task_id}")
    
    # Start web server (comment out for testing)
    # await app.start_web_server()

if __name__ == "__main__":
    asyncio.run(main())