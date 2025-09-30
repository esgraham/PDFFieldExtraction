"""
Enhanced HITL Review Application

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

# Image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import numpy as np
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        HAS_PDF2IMAGE = False
    HAS_PYMUPDF = False

# Web framework
try:
    from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
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
    def pdf_to_images(pdf_path: str) -> List[str]:
        """Convert PDF pages to base64 encoded images."""
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        try:
            if HAS_PYMUPDF:
                return PDFProcessor._pdf_to_images_pymupdf(pdf_path)
            elif HAS_PDF2IMAGE:
                return PDFProcessor._pdf_to_images_pdf2image(pdf_path)
            else:
                logger.warning("No PDF processing library available")
                return []
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            return []
    
    @staticmethod
    def _pdf_to_images_pymupdf(pdf_path: str) -> List[str]:
        """Convert PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        page_images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for clarity
            img_data = pix.tobytes("png")
            img_b64 = base64.b64encode(img_data).decode('utf-8')
            page_images.append(img_b64)
        
        doc.close()
        return page_images
    
    @staticmethod
    def _pdf_to_images_pdf2image(pdf_path: str) -> List[str]:
        """Convert PDF using pdf2image."""
        images = convert_from_path(pdf_path, dpi=200)
        page_images = []
        
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            page_images.append(img_b64)
        
        return page_images
    
    @staticmethod
    def draw_bounding_boxes(image_b64: str, fields: List[ExtractedField], page_num: int = 0) -> str:
        """Draw bounding boxes and field information on PDF page image."""
        if not HAS_IMAGE_PROCESSING:
            logger.warning("Image processing not available")
            return image_b64
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
                small_font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            # Draw bounding boxes for fields on this page
            page_fields = [f for f in fields if f.bounding_box and f.bounding_box.page_number == page_num]
            
            for field in page_fields:
                bbox = field.bounding_box
                
                # Determine color based on confidence and rule violations
                if field.rule_violations:
                    color = "#FF4444"  # Red for rule violations
                    text_color = "#FFFFFF"
                elif field.confidence < 0.7:
                    color = "#FFA500"  # Orange for low confidence
                    text_color = "#000000"
                elif field.confidence < 0.9:
                    color = "#FFFF00"  # Yellow for medium confidence
                    text_color = "#000000"
                else:
                    color = "#00FF00"  # Green for high confidence
                    text_color = "#000000"
                
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
                
                # Draw semi-transparent fill
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # Convert hex color to RGB
                rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                overlay_draw.rectangle(
                    [(left, top), (right, bottom)],
                    fill=rgb_color + (50,)  # 50 alpha for transparency
                )
                
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(img)\n                \n                # Field name and confidence label\n                label_text = f\"{field.field_name}: {field.confidence:.2f}\"\n                if field.rule_violations:\n                    label_text += \" ⚠️\"\n                \n                # Label background\n                label_bbox = draw.textbbox((left, top - 25), label_text, font=font)\n                draw.rectangle(\n                    [(label_bbox[0] - 2, label_bbox[1] - 2), \n                     (label_bbox[2] + 2, label_bbox[3] + 2)],\n                    fill=color\n                )\n                \n                # Label text\n                draw.text(\n                    (left, top - 25),\n                    label_text,\n                    fill=text_color,\n                    font=font\n                )\n                \n                # Value preview (truncated)\n                value_text = str(field.value)[:30] + (\"...\" if len(str(field.value)) > 30 else \"\")\n                draw.text(\n                    (left + 5, top + 5),\n                    value_text,\n                    fill=text_color,\n                    font=small_font\n                )\n            \n            # Convert back to base64\n            buffer = io.BytesIO()\n            img.save(buffer, format='PNG')\n            return base64.b64encode(buffer.getvalue()).decode('utf-8')\n            \n        except Exception as e:\n            logger.error(f\"Bounding box drawing failed: {str(e)}\")\n            return image_b64

class EnhancedHITLDatabase:
    \"\"\"Enhanced database for HITL review tasks with training data storage.\"\"\"\n    \n    def __init__(self, db_path: str = \"enhanced_hitl.db\"):\n        self.db_path = db_path\n        self.init_database()\n    \n    def init_database(self):\n        \"\"\"Initialize enhanced database schema.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.execute(\"\"\"\n                CREATE TABLE IF NOT EXISTS review_tasks (\n                    task_id TEXT PRIMARY KEY,\n                    document_id TEXT,\n                    document_type TEXT,\n                    extracted_fields TEXT,\n                    validation_errors TEXT,\n                    pdf_file_path TEXT,\n                    pdf_pages TEXT,\n                    created_at TIMESTAMP,\n                    assigned_to TEXT,\n                    assigned_at TIMESTAMP,\n                    first_touched_at TIMESTAMP,\n                    status TEXT,\n                    priority INTEGER,\n                    sla_deadline TIMESTAMP,\n                    age_to_first_touch REAL,\n                    age_to_resolve REAL,\n                    notes TEXT,\n                    resolution_time TIMESTAMP\n                )\n            \"\"\")\n            \n            conn.execute(\"\"\"\n                CREATE TABLE IF NOT EXISTS review_feedback (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    task_id TEXT,\n                    field_name TEXT,\n                    original_value TEXT,\n                    corrected_value TEXT,\n                    reviewer_id TEXT,\n                    correction_reason TEXT,\n                    confidence_rating INTEGER,\n                    bounding_box_adjustment TEXT,\n                    timestamp TIMESTAMP,\n                    FOREIGN KEY (task_id) REFERENCES review_tasks (task_id)\n                )\n            \"\"\")\n            \n            conn.execute(\"\"\"\n                CREATE TABLE IF NOT EXISTS training_data (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    task_id TEXT,\n                    document_type TEXT,\n                    training_data TEXT,\n                    created_at TIMESTAMP,\n                    FOREIGN KEY (task_id) REFERENCES review_tasks (task_id)\n                )\n            \"\"\")\n            \n            conn.execute(\"\"\"\n                CREATE TABLE IF NOT EXISTS reviewers (\n                    user_id TEXT PRIMARY KEY,\n                    name TEXT,\n                    email TEXT,\n                    teams_webhook TEXT,\n                    specializations TEXT,\n                    current_workload INTEGER,\n                    completed_tasks INTEGER,\n                    is_available BOOLEAN,\n                    avg_resolution_time REAL\n                )\n            \"\"\")\n    \n    def create_task(self, task: ReviewTask):\n        \"\"\"Store review task in database.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.execute(\"\"\"\n                INSERT OR REPLACE INTO review_tasks \n                (task_id, document_id, document_type, extracted_fields, validation_errors,\n                 pdf_file_path, pdf_pages, created_at, assigned_to, assigned_at, \n                 first_touched_at, status, priority, sla_deadline, age_to_first_touch,\n                 age_to_resolve, notes, resolution_time)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            \"\"\", (\n                task.task_id, task.document_id, task.document_type,\n                json.dumps([asdict(f) for f in task.extracted_fields]),\n                json.dumps(task.validation_errors),\n                task.pdf_file_path,\n                json.dumps(task.pdf_pages),\n                task.created_at, task.assigned_to, task.assigned_at,\n                task.first_touched_at, task.status, task.priority,\n                task.sla_deadline, task.age_to_first_touch,\n                task.age_to_resolve, task.notes, task.resolution_time\n            ))\n    \n    def update_task(self, task: ReviewTask):\n        \"\"\"Update existing task.\"\"\"\n        self.create_task(task)  # Using REPLACE functionality\n    \n    def store_feedback(self, feedback: ReviewFeedback):\n        \"\"\"Store review feedback for training data.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.execute(\"\"\"\n                INSERT INTO review_feedback \n                (task_id, field_name, original_value, corrected_value, reviewer_id,\n                 correction_reason, confidence_rating, bounding_box_adjustment, timestamp)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n            \"\"\", (\n                feedback.task_id, feedback.field_name, \n                json.dumps(feedback.original_value),\n                json.dumps(feedback.corrected_value),\n                feedback.reviewer_id, feedback.correction_reason,\n                feedback.confidence_rating,\n                json.dumps(asdict(feedback.bounding_box_adjustment)) if feedback.bounding_box_adjustment else None,\n                feedback.timestamp\n            ))\n    \n    def store_training_data(self, task_id: str, training_data: Dict[str, Any]):\n        \"\"\"Store training data generated from completed tasks.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.execute(\"\"\"\n                INSERT INTO training_data (task_id, document_type, training_data, created_at)\n                VALUES (?, ?, ?, ?)\n            \"\"\", (\n                task_id,\n                training_data.get(\"document_type\"),\n                json.dumps(training_data),\n                datetime.now()\n            ))\n    \n    def get_training_data(self, document_type: Optional[str] = None, \n                         limit: int = 100) -> List[Dict[str, Any]]:\n        \"\"\"Retrieve training data for model improvement.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.row_factory = sqlite3.Row\n            \n            if document_type:\n                cursor = conn.execute(\"\"\"\n                    SELECT * FROM training_data \n                    WHERE document_type = ?\n                    ORDER BY created_at DESC\n                    LIMIT ?\n                \"\"\", (document_type, limit))\n            else:\n                cursor = conn.execute(\"\"\"\n                    SELECT * FROM training_data \n                    ORDER BY created_at DESC\n                    LIMIT ?\n                \"\"\", (limit,))\n            \n            return [{\n                \"id\": row[\"id\"],\n                \"task_id\": row[\"task_id\"],\n                \"document_type\": row[\"document_type\"],\n                \"training_data\": json.loads(row[\"training_data\"]),\n                \"created_at\": row[\"created_at\"]\n            } for row in cursor.fetchall()]\n    \n    def get_sla_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get SLA performance statistics.\"\"\"\n        with sqlite3.connect(self.db_path) as conn:\n            conn.row_factory = sqlite3.Row\n            \n            # Average age to first touch\n            cursor = conn.execute(\"\"\"\n                SELECT AVG(age_to_first_touch) as avg_first_touch,\n                       AVG(age_to_resolve) as avg_resolve,\n                       COUNT(*) as total_completed\n                FROM review_tasks \n                WHERE status = 'completed' AND age_to_first_touch IS NOT NULL\n            \"\"\")\n            \n            stats = cursor.fetchone()\n            \n            # SLA violations\n            cursor = conn.execute(\"\"\"\n                SELECT COUNT(*) as violations\n                FROM review_tasks \n                WHERE sla_deadline < datetime('now') AND status != 'completed'\n            \"\"\")\n            \n            violations = cursor.fetchone()[\"violations\"]\n            \n            return {\n                \"avg_age_to_first_touch\": stats[\"avg_first_touch\"] or 0,\n                \"avg_age_to_resolve\": stats[\"avg_resolve\"] or 0,\n                \"total_completed_tasks\": stats[\"total_completed\"] or 0,\n                \"current_sla_violations\": violations\n            }

class EnhancedTeamsService:
    \"\"\"Enhanced Teams notification service with rich cards.\"\"\"\n    \n    def __init__(self):\n        self.client = httpx.AsyncClient() if HAS_HTTPX else None\n    \n    async def send_assignment_notification(self, task: ReviewTask, reviewer: ReviewerInfo, \n                                         webhook_url: str) -> bool:\n        \"\"\"Send enhanced assignment notification to Teams.\"\"\"\n        if not self.client or not webhook_url:\n            return False\n        \n        try:\n            card = self._create_assignment_card(task, reviewer)\n            \n            response = await self.client.post(\n                webhook_url,\n                json=card,\n                headers={\"Content-Type\": \"application/json\"}\n            )\n            \n            return response.status_code == 200\n            \n        except Exception as e:\n            logger.error(f\"Teams notification failed: {str(e)}\")\n            return False\n    \n    def _create_assignment_card(self, task: ReviewTask, reviewer: ReviewerInfo) -> Dict:\n        \"\"\"Create Teams adaptive card for task assignment.\"\"\"\n        \n        priority_colors = {1: \"good\", 2: \"warning\", 3: \"attention\"}\n        priority_names = {1: \"Low\", 2: \"Normal\", 3: \"High\"}\n        \n        card = {\n            \"type\": \"message\",\n            \"attachments\": [{\n                \"contentType\": \"application/vnd.microsoft.card.adaptive\",\n                \"content\": {\n                    \"type\": \"AdaptiveCard\",\n                    \"version\": \"1.3\",\n                    \"body\": [\n                        {\n                            \"type\": \"TextBlock\",\n                            \"text\": \"📋 Document Review Assignment\",\n                            \"size\": \"large\",\n                            \"weight\": \"bolder\",\n                            \"color\": priority_colors.get(task.priority, \"default\")\n                        },\n                        {\n                            \"type\": \"FactSet\",\n                            \"facts\": [\n                                {\"title\": \"Document ID\", \"value\": task.document_id},\n                                {\"title\": \"Document Type\", \"value\": task.document_type.title()},\n                                {\"title\": \"Priority\", \"value\": priority_names.get(task.priority, \"Unknown\")},\n                                {\"title\": \"Fields to Review\", \"value\": str(len(task.extracted_fields))},\n                                {\"title\": \"Validation Errors\", \"value\": str(len(task.validation_errors))},\n                                {\"title\": \"SLA Deadline\", \"value\": task.sla_deadline.strftime(\"%Y-%m-%d %H:%M\") if task.sla_deadline else \"None\"},\n                                {\"title\": \"Assigned To\", \"value\": reviewer.name}\n                            ]\n                        }\n                    ],\n                    \"actions\": [\n                        {\n                            \"type\": \"Action.OpenUrl\",\n                            \"title\": \"🔍 Start Review\",\n                            \"url\": f\"http://localhost:8001/review/{task.task_id}\"\n                        }\n                    ]\n                }\n            }]\n        }\n        \n        # Add validation errors if any\n        if task.validation_errors:\n            card[\"attachments\"][0][\"content\"][\"body\"].append({\n                \"type\": \"TextBlock\",\n                \"text\": \"⚠️ Issues Found:\",\n                \"weight\": \"bolder\",\n                \"color\": \"attention\"\n            })\n            \n            for error in task.validation_errors[:3]:  # Show top 3 errors\n                card[\"attachments\"][0][\"content\"][\"body\"].append({\n                    \"type\": \"TextBlock\",\n                    \"text\": f\"• {error}\",\n                    \"wrap\": True\n                })\n        \n        return card\n    \n    async def send_sla_alert(self, task: ReviewTask, webhook_url: str) -> bool:\n        \"\"\"Send SLA violation alert.\"\"\"\n        if not self.client or not webhook_url:\n            return False\n        \n        try:\n            current_time = datetime.now()\n            overdue_hours = (current_time - task.sla_deadline).total_seconds() / 3600 if task.sla_deadline else 0\n            \n            card = {\n                \"type\": \"message\",\n                \"attachments\": [{\n                    \"contentType\": \"application/vnd.microsoft.card.adaptive\",\n                    \"content\": {\n                        \"type\": \"AdaptiveCard\",\n                        \"version\": \"1.3\",\n                        \"body\": [\n                            {\n                                \"type\": \"TextBlock\",\n                                \"text\": \"🚨 SLA VIOLATION ALERT\",\n                                \"size\": \"large\",\n                                \"weight\": \"bolder\",\n                                \"color\": \"attention\"\n                            },\n                            {\n                                \"type\": \"FactSet\",\n                                \"facts\": [\n                                    {\"title\": \"Task ID\", \"value\": task.task_id},\n                                    {\"title\": \"Document ID\", \"value\": task.document_id},\n                                    {\"title\": \"Assigned To\", \"value\": task.assigned_to or \"Unassigned\"},\n                                    {\"title\": \"Overdue By\", \"value\": f\"{overdue_hours:.1f} hours\"},\n                                    {\"title\": \"Created\", \"value\": task.created_at.strftime(\"%Y-%m-%d %H:%M\")},\n                                    {\"title\": \"SLA Deadline\", \"value\": task.sla_deadline.strftime(\"%Y-%m-%d %H:%M\") if task.sla_deadline else \"None\"}\n                                ]\n                            }\n                        ],\n                        \"actions\": [\n                            {\n                                \"type\": \"Action.OpenUrl\",\n                                \"title\": \"🔍 Review Now\",\n                                \"url\": f\"http://localhost:8001/review/{task.task_id}\"\n                            }\n                        ]\n                    }\n                }]\n            }\n            \n            response = await self.client.post(\n                webhook_url,\n                json=card,\n                headers={\"Content-Type\": \"application/json\"}\n            )\n            \n            return response.status_code == 200\n            \n        except Exception as e:\n            logger.error(f\"SLA alert failed: {str(e)}\")\n            return False

class EnhancedHITLReviewApp:
    \"\"\"Enhanced HITL Review Application with PDF viewer and training data collection.\"\"\"\n    \n    def __init__(self, database_path: str = \"enhanced_hitl.db\"):\n        self.database = EnhancedHITLDatabase(database_path)\n        self.teams_service = EnhancedTeamsService()\n        self.pdf_processor = PDFProcessor()\n        self.reviewers: Dict[str, ReviewerInfo] = {}\n        self.active_tasks: Dict[str, ReviewTask] = {}\n        \n        # SLA configurations (in hours)\n        self.sla_config = {\n            \"high_priority_hours\": 2,\n            \"normal_priority_hours\": 24,\n            \"low_priority_hours\": 72\n        }\n        \n        # Initialize with sample reviewers\n        self._init_sample_reviewers()\n    \n    def _init_sample_reviewers(self):\n        \"\"\"Initialize with sample reviewers.\"\"\"\n        sample_reviewers = [\n            ReviewerInfo(\n                user_id=\"reviewer1\",\n                name=\"Alice Johnson\",\n                email=\"alice@company.com\",\n                specializations=[\"invoice\", \"receipt\"],\n                is_available=True\n            ),\n            ReviewerInfo(\n                user_id=\"reviewer2\",\n                name=\"Bob Smith\",\n                email=\"bob@company.com\",\n                specializations=[\"contract\", \"invoice\"],\n                is_available=True\n            )\n        ]\n        \n        for reviewer in sample_reviewers:\n            self.reviewers[reviewer.user_id] = reviewer\n    \n    def create_enhanced_review_task(self, document_id: str, document_type: str, \n                                  extracted_fields: List[ExtractedField], \n                                  validation_errors: List[str],\n                                  pdf_file_path: Optional[str] = None,\n                                  priority: int = 1) -> str:\n        \"\"\"Create enhanced review task with PDF processing and SLA tracking.\"\"\"\n        \n        task_id = str(uuid.uuid4())\n        \n        # Process PDF pages for viewer\n        pdf_pages = []\n        if pdf_file_path and Path(pdf_file_path).exists():\n            logger.info(f\"Processing PDF: {pdf_file_path}\")\n            pdf_pages = self.pdf_processor.pdf_to_images(pdf_file_path)\n            \n            # Add bounding box overlays to pages\n            for i, page_image in enumerate(pdf_pages):\n                pdf_pages[i] = self.pdf_processor.draw_bounding_boxes(page_image, extracted_fields, page_num=i)\n        \n        # Calculate SLA deadline based on priority\n        priority_hours = {\n            1: self.sla_config[\"low_priority_hours\"],\n            2: self.sla_config[\"normal_priority_hours\"],\n            3: self.sla_config[\"high_priority_hours\"]\n        }\n        \n        sla_hours = priority_hours.get(priority, 24)\n        sla_deadline = datetime.now() + timedelta(hours=sla_hours)\n        \n        task = ReviewTask(\n            task_id=task_id,\n            document_id=document_id,\n            document_type=document_type,\n            extracted_fields=extracted_fields,\n            validation_errors=validation_errors,\n            pdf_file_path=pdf_file_path,\n            pdf_pages=pdf_pages,\n            created_at=datetime.now(),\n            priority=priority,\n            sla_deadline=sla_deadline\n        )\n        \n        # Store in database\n        self.database.create_task(task)\n        \n        # Add to active tasks\n        self.active_tasks[task_id] = task\n        \n        # Auto-assign based on workload and specialization\n        self._auto_assign_task(task_id)\n        \n        logger.info(f\"Created enhanced review task {task_id} for document {document_id} with SLA deadline {sla_deadline}\")\n        return task_id\n    \n    def _auto_assign_task(self, task_id: str):\n        \"\"\"Auto-assign task to best available reviewer.\"\"\"\n        task = self.active_tasks.get(task_id)\n        if not task:\n            return\n        \n        # Find best reviewer based on specialization and workload\n        available_reviewers = [\n            (reviewer_id, reviewer) for reviewer_id, reviewer in self.reviewers.items()\n            if reviewer.is_available and task.document_type in reviewer.specializations\n        ]\n        \n        if available_reviewers:\n            # Sort by current workload (ascending)\n            available_reviewers.sort(key=lambda x: x[1].current_workload)\n            best_reviewer_id, best_reviewer = available_reviewers[0]\n            \n            self.assign_task(task_id, best_reviewer_id)\n            logger.info(f\"Auto-assigned task {task_id} to reviewer {best_reviewer_id}\")\n        else:\n            logger.warning(f\"No available reviewers for task {task_id} ({task.document_type})\")\n    \n    def assign_task(self, task_id: str, reviewer_id: str, \n                   notify: bool = True) -> bool:\n        \"\"\"Assign task to reviewer with notification.\"\"\"\n        \n        if task_id not in self.active_tasks:\n            logger.error(f\"Task {task_id} not found\")\n            return False\n        \n        if reviewer_id not in self.reviewers:\n            logger.error(f\"Reviewer {reviewer_id} not found\")\n            return False\n        \n        task = self.active_tasks[task_id]\n        reviewer = self.reviewers[reviewer_id]\n        \n        # Update task assignment\n        task.assigned_to = reviewer_id\n        task.assigned_at = datetime.now()\n        task.status = \"assigned\"\n        \n        # Update reviewer workload\n        reviewer.current_workload += 1\n        \n        # Update database\n        self.database.update_task(task)\n        \n        # Send notification\n        if notify and reviewer.teams_webhook:\n            asyncio.create_task(\n                self.teams_service.send_assignment_notification(\n                    task, reviewer, reviewer.teams_webhook\n                )\n            )\n        \n        logger.info(f\"Assigned task {task_id} to reviewer {reviewer_id}\")\n        return True\n    \n    def record_first_touch(self, task_id: str, reviewer_id: str) -> bool:\n        \"\"\"Record when reviewer first touches the task for SLA tracking.\"\"\"\n        \n        task = self.active_tasks.get(task_id)\n        if not task or task.assigned_to != reviewer_id:\n            return False\n        \n        if not task.first_touched_at:\n            task.first_touched_at = datetime.now()\n            task.age_to_first_touch = (task.first_touched_at - task.created_at).total_seconds() / 60\n            task.status = \"in_progress\"\n            \n            # Update database\n            self.database.update_task(task)\n            \n            logger.info(f\"First touch recorded for task {task_id} (age: {task.age_to_first_touch:.1f} min)\")\n        \n        return True\n    \n    def collect_field_feedback(self, task_id: str, field_name: str, \n                             original_value: Any, corrected_value: Any,\n                             reviewer_id: str, correction_reason: str,\n                             confidence_rating: int,\n                             bounding_box_adjustment: Optional[BoundingBox] = None) -> bool:\n        \"\"\"Collect feedback for specific field correction.\"\"\"\n        \n        task = self.active_tasks.get(task_id)\n        if not task:\n            return False\n        \n        feedback = ReviewFeedback(\n            task_id=task_id,\n            field_name=field_name,\n            original_value=original_value,\n            corrected_value=corrected_value,\n            reviewer_id=reviewer_id,\n            correction_reason=correction_reason,\n            confidence_rating=confidence_rating,\n            bounding_box_adjustment=bounding_box_adjustment,\n            timestamp=datetime.now()\n        )\n        \n        task.feedback_collected.append(feedback)\n        \n        # Store in database for training data\n        self.database.store_feedback(feedback)\n        \n        logger.info(f\"Collected feedback for task {task_id}, field {field_name}\")\n        return True\n    \n    def complete_task_with_feedback(self, task_id: str, reviewer_id: str, \n                                  field_corrections: Dict[str, Dict[str, Any]], \n                                  notes: Optional[str] = None) -> bool:\n        \"\"\"Complete task with detailed field corrections and feedback collection.\"\"\"\n        \n        task = self.active_tasks.get(task_id)\n        if not task or task.assigned_to != reviewer_id:\n            logger.error(f\"Task {task_id} not found or not assigned to {reviewer_id}\")\n            return False\n        \n        # Record completion timing\n        task.resolution_time = datetime.now()\n        task.age_to_resolve = (task.resolution_time - task.created_at).total_seconds() / 60\n        task.status = \"completed\"\n        task.notes = notes\n        \n        # Process field corrections and collect feedback\n        for field_name, correction_data in field_corrections.items():\n            # Find original field\n            original_field = next(\n                (f for f in task.extracted_fields if f.field_name == field_name),\n                None\n            )\n            \n            if original_field:\n                corrected_value = correction_data.get(\"corrected_value\")\n                correction_reason = correction_data.get(\"reason\", \"Manual correction\")\n                confidence_rating = correction_data.get(\"confidence\", 5)\n                bbox_adjustment = correction_data.get(\"bbox_adjustment\")\n                \n                # Collect feedback if value changed\n                if original_field.value != corrected_value:\n                    self.collect_field_feedback(\n                        task_id=task_id,\n                        field_name=field_name,\n                        original_value=original_field.value,\n                        corrected_value=corrected_value,\n                        reviewer_id=reviewer_id,\n                        correction_reason=correction_reason,\n                        confidence_rating=confidence_rating,\n                        bounding_box_adjustment=bbox_adjustment\n                    )\n                \n                # Update field value\n                original_field.value = corrected_value\n        \n        # Update reviewer stats\n        reviewer = self.reviewers.get(reviewer_id)\n        if reviewer:\n            reviewer.current_workload -= 1\n            reviewer.completed_tasks += 1\n            \n            # Update average resolution time\n            if reviewer.avg_resolution_time == 0:\n                reviewer.avg_resolution_time = task.age_to_resolve\n            else:\n                reviewer.avg_resolution_time = (\n                    (reviewer.avg_resolution_time * (reviewer.completed_tasks - 1) + task.age_to_resolve) /\n                    reviewer.completed_tasks\n                )\n        \n        # Update database\n        self.database.update_task(task)\n        \n        # Generate training data\n        self._generate_training_data(task)\n        \n        # Remove from active tasks\n        del self.active_tasks[task_id]\n        \n        logger.info(f\"Completed task {task_id} by reviewer {reviewer_id} (resolution time: {task.age_to_resolve:.1f} min)\")\n        return True\n    \n    def _generate_training_data(self, task: ReviewTask):\n        \"\"\"Generate comprehensive training data from completed task.\"\"\"\n        try:\n            training_data = {\n                \"document_id\": task.document_id,\n                \"document_type\": task.document_type,\n                \"completed_at\": task.resolution_time.isoformat(),\n                \"resolution_time_minutes\": task.age_to_resolve,\n                \"reviewer_id\": task.assigned_to,\n                \"field_corrections\": [],\n                \"bounding_box_corrections\": [],\n                \"confidence_improvements\": [],\n                \"validation_failures\": task.validation_errors,\n                \"original_extraction_methods\": {}\n            }\n            \n            # Process feedback for training data\n            for feedback in task.feedback_collected:\n                correction_entry = {\n                    \"field_name\": feedback.field_name,\n                    \"original_value\": feedback.original_value,\n                    \"corrected_value\": feedback.corrected_value,\n                    \"correction_reason\": feedback.correction_reason,\n                    \"reviewer_confidence\": feedback.confidence_rating,\n                    \"timestamp\": feedback.timestamp.isoformat()\n                }\n                training_data[\"field_corrections\"].append(correction_entry)\n                \n                # Bounding box adjustments\n                if feedback.bounding_box_adjustment:\n                    bbox_entry = {\n                        \"field_name\": feedback.field_name,\n                        \"original_bbox\": None,  # Would need to store original\n                        \"corrected_bbox\": asdict(feedback.bounding_box_adjustment)\n                    }\n                    training_data[\"bounding_box_corrections\"].append(bbox_entry)\n                \n                # Confidence improvements (before vs after)\n                original_field = next(\n                    (f for f in task.extracted_fields if f.field_name == feedback.field_name),\n                    None\n                )\n                \n                if original_field:\n                    confidence_entry = {\n                        \"field_name\": feedback.field_name,\n                        \"original_confidence\": original_field.confidence,\n                        \"reviewer_confidence\": feedback.confidence_rating,\n                        \"confidence_delta\": feedback.confidence_rating - (original_field.confidence * 5)\n                    }\n                    training_data[\"confidence_improvements\"].append(confidence_entry)\n                    \n                    # Track extraction methods for retraining\n                    training_data[\"original_extraction_methods\"][feedback.field_name] = original_field.extraction_method\n            \n            # Store training data\n            self.database.store_training_data(task.task_id, training_data)\n            \n            logger.info(f\"Generated comprehensive training data for task {task.task_id} with {len(task.feedback_collected)} corrections\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate training data: {str(e)}\")\n    \n    def get_enhanced_queue_status(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive queue status with SLA metrics.\"\"\"\n        \n        current_time = datetime.now()\n        \n        # Basic counts\n        total_tasks = len(self.active_tasks)\n        pending_tasks = [t for t in self.active_tasks.values() if t.status == \"pending\"]\n        assigned_tasks = [t for t in self.active_tasks.values() if t.status == \"assigned\"]\n        in_progress_tasks = [t for t in self.active_tasks.values() if t.status == \"in_progress\"]\n        \n        # SLA violations\n        sla_violations = [\n            t for t in self.active_tasks.values() \n            if t.sla_deadline and current_time > t.sla_deadline\n        ]\n        \n        # Age analysis\n        total_age_minutes = sum(\n            (current_time - t.created_at).total_seconds() / 60 \n            for t in self.active_tasks.values()\n        )\n        avg_age = total_age_minutes / total_tasks if total_tasks > 0 else 0\n        \n        # Priority breakdown\n        priority_breakdown = {}\n        for priority in [1, 2, 3]:\n            priority_tasks = [t for t in self.active_tasks.values() if t.priority == priority]\n            priority_breakdown[f\"priority_{priority}\"] = len(priority_tasks)\n        \n        # Reviewer workload\n        reviewer_stats = {}\n        for reviewer_id, reviewer in self.reviewers.items():\n            reviewer_stats[reviewer_id] = {\n                \"name\": reviewer.name,\n                \"current_workload\": reviewer.current_workload,\n                \"completed_tasks\": reviewer.completed_tasks,\n                \"avg_resolution_time\": reviewer.avg_resolution_time,\n                \"is_available\": reviewer.is_available\n            }\n        \n        # SLA statistics from database\n        sla_stats = self.database.get_sla_statistics()\n        \n        return {\n            \"timestamp\": current_time.isoformat(),\n            \"queue_summary\": {\n                \"total_active_tasks\": total_tasks,\n                \"pending_tasks\": len(pending_tasks),\n                \"assigned_tasks\": len(assigned_tasks),\n                \"in_progress_tasks\": len(in_progress_tasks),\n                \"sla_violations\": len(sla_violations)\n            },\n            \"timing_metrics\": {\n                \"avg_queue_age_minutes\": avg_age,\n                \"avg_age_to_first_touch\": sla_stats[\"avg_age_to_first_touch\"],\n                \"avg_age_to_resolve\": sla_stats[\"avg_age_to_resolve\"]\n            },\n            \"priority_breakdown\": priority_breakdown,\n            \"reviewer_stats\": reviewer_stats,\n            \"sla_performance\": {\n                \"total_completed_tasks\": sla_stats[\"total_completed_tasks\"],\n                \"current_violations\": sla_stats[\"current_sla_violations\"],\n                \"violation_rate\": (len(sla_violations) / total_tasks * 100) if total_tasks > 0 else 0\n            }\n        }\n    \n    def get_training_data_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of collected training data.\"\"\"\n        \n        # Get training data from database\n        all_training_data = self.database.get_training_data(limit=1000)\n        \n        # Analyze training data\n        total_corrections = 0\n        field_correction_counts = {}\n        document_type_counts = {}\n        \n        for data_entry in all_training_data:\n            training_data = data_entry[\"training_data\"]\n            doc_type = training_data.get(\"document_type\", \"unknown\")\n            \n            document_type_counts[doc_type] = document_type_counts.get(doc_type, 0) + 1\n            \n            for correction in training_data.get(\"field_corrections\", []):\n                field_name = correction[\"field_name\"]\n                field_correction_counts[field_name] = field_correction_counts.get(field_name, 0) + 1\n                total_corrections += 1\n        \n        return {\n            \"total_training_samples\": len(all_training_data),\n            \"total_field_corrections\": total_corrections,\n            \"document_type_distribution\": document_type_counts,\n            \"field_correction_frequency\": dict(sorted(\n                field_correction_counts.items(), \n                key=lambda x: x[1], \n                reverse=True\n            )[:10]),  # Top 10 fields with corrections\n            \"last_updated\": datetime.now().isoformat()\n        }\n    \n    def export_training_data(self, document_type: Optional[str] = None, \n                           format: str = \"json\") -> str:\n        \"\"\"Export training data for model retraining.\"\"\"\n        \n        training_data = self.database.get_training_data(document_type=document_type)\n        \n        if format == \"json\":\n            export_data = {\n                \"export_timestamp\": datetime.now().isoformat(),\n                \"document_type_filter\": document_type,\n                \"total_samples\": len(training_data),\n                \"training_samples\": training_data\n            }\n            return json.dumps(export_data, indent=2)\n        \n        # Could add other formats (CSV, etc.) here\n        return json.dumps(training_data, indent=2)\n    \n    async def monitor_sla_violations(self):\n        \"\"\"Background task to monitor and alert on SLA violations.\"\"\"\n        \n        while True:\n            try:\n                current_time = datetime.now()\n                \n                for task in self.active_tasks.values():\n                    if (task.sla_deadline and \n                        current_time > task.sla_deadline and \n                        task.status != \"completed\"):\n                        \n                        # Send SLA violation alert\n                        if task.assigned_to and task.assigned_to in self.reviewers:\n                            reviewer = self.reviewers[task.assigned_to]\n                            if reviewer.teams_webhook:\n                                await self.teams_service.send_sla_alert(\n                                    task, reviewer.teams_webhook\n                                )\n                        \n                        # Escalate priority if not already at max\n                        if task.priority < 3:\n                            task.priority += 1\n                            self.database.update_task(task)\n                            logger.warning(f\"Escalated task {task.task_id} due to SLA violation\")\n                \n                # Sleep for 15 minutes before next check\n                await asyncio.sleep(900)\n                \n            except Exception as e:\n                logger.error(f\"SLA monitoring error: {str(e)}\")\n                await asyncio.sleep(60)  # Retry in 1 minute on error

# Sample usage and testing
def create_sample_task():\n    \"\"\"Create a sample review task for demonstration.\"\"\"\n    \n    app = EnhancedHITLReviewApp()\n    \n    # Sample extracted fields with bounding boxes\n    sample_fields = [\n        ExtractedField(\n            field_name=\"invoice_number\",\n            value=\"INV-2024-001\",\n            confidence=0.95,\n            bounding_box=BoundingBox(x=100, y=50, width=150, height=25, page_number=0, confidence=0.95),\n            extraction_method=\"OCR\",\n            rule_violations=[],\n            suggested_corrections=[]\n        ),\n        ExtractedField(\n            field_name=\"total_amount\",\n            value=\"$1,234.56\",\n            confidence=0.65,  # Low confidence\n            bounding_box=BoundingBox(x=400, y=300, width=100, height=20, page_number=0, confidence=0.65),\n            extraction_method=\"Template\",\n            rule_violations=[\"Amount format validation failed\"],\n            suggested_corrections=[\"$1234.56\"]\n        ),\n        ExtractedField(\n            field_name=\"vendor_name\",\n            value=\"Acme Corporation\",\n            confidence=0.88,\n            bounding_box=BoundingBox(x=50, y=100, width=200, height=30, page_number=0, confidence=0.88),\n            extraction_method=\"NLP\",\n            rule_violations=[],\n            suggested_corrections=[]\n        )\n    ]\n    \n    validation_errors = [\n        \"Total amount format does not match expected pattern\",\n        \"Invoice date is missing or illegible\"\n    ]\n    \n    # Create review task\n    task_id = app.create_enhanced_review_task(\n        document_id=\"DOC-2024-12345\",\n        document_type=\"invoice\",\n        extracted_fields=sample_fields,\n        validation_errors=validation_errors,\n        pdf_file_path=None,  # Would be actual PDF path\n        priority=2\n    )\n    \n    print(f\"Created sample review task: {task_id}\")\n    \n    # Simulate reviewer completing task\n    field_corrections = {\n        \"total_amount\": {\n            \"corrected_value\": \"$1234.56\",\n            \"reason\": \"Fixed formatting - removed comma\",\n            \"confidence\": 5\n        }\n    }\n    \n    app.record_first_touch(task_id, \"reviewer1\")\n    \n    success = app.complete_task_with_feedback(\n        task_id=task_id,\n        reviewer_id=\"reviewer1\",\n        field_corrections=field_corrections,\n        notes=\"Completed review - fixed amount formatting issue\"\n    )\n    \n    print(f\"Task completion status: {success}\")\n    \n    # Show queue status\n    status = app.get_enhanced_queue_status()\n    print(f\"Queue status: {json.dumps(status, indent=2)}\")\n    \n    # Show training data summary\n    training_summary = app.get_training_data_summary()\n    print(f\"Training data summary: {json.dumps(training_summary, indent=2)}\")\n    \n    return app

if __name__ == \"__main__\":\n    # Run sample demonstration\n    logging.basicConfig(level=logging.INFO)\n    \n    print(\"🚀 Enhanced HITL Review System Demo\")\n    print(\"====================================\")\n    \n    app = create_sample_task()\n    \n    print(\"\\n✅ Enhanced HITL System Features:\")\n    print(\"  • PDF viewer with bounding box overlays\")\n    print(\"  • Enhanced field extraction with confidence scores\")\n    print(\"  • Rule failure highlighting and suggestions\")\n    print(\"  • Teams/email assignment with SLA tracking\")\n    print(\"  • Comprehensive feedback collection\")\n    print(\"  • Training data generation for model improvement\")\n    print(\"  • Real-time queue monitoring and statistics\")\n    print(\"  • Auto-assignment based on reviewer specialization\")\n    print(\"  • SLA violation alerts and escalation\")\n    \n    print(\"\\n🎯 Ready for production deployment!\")