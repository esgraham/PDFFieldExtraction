"""
Enhanced HITL Web Interface

Web application providing:
- Side-by-side PDF viewer with bounding box overlays
- Field extraction review and correction interface
- SLA tracking and assignment management
- Training data collection dashboard
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hitl.enhanced_hitl_clean import EnhancedHITLReviewApp, ExtractedField, BoundingBox, ReviewFeedback

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enhanced HITL Review System", version="1.0.0")

# Initialize templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Initialize HITL app
hitl_app = EnhancedHITLReviewApp()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view."""
    
    # Get queue status
    queue_status = hitl_app.get_enhanced_queue_status()
    
    # Get training data summary
    training_summary = hitl_app.get_training_data_summary()
    
    return templates.TemplateResponse("enhanced_hitl_dashboard.html", {
        "request": request,
        "queue_status": queue_status,
        "training_summary": training_summary,
        "active_tasks": list(hitl_app.active_tasks.values())
    })

@app.get("/api/queue-status", response_class=JSONResponse)
async def get_queue_status():
    """Get current queue status API."""
    return hitl_app.get_enhanced_queue_status()

@app.get("/api/training-summary", response_class=JSONResponse)
async def get_training_summary():
    """Get training data summary API."""
    return hitl_app.get_training_data_summary()

@app.get("/review/{task_id}", response_class=HTMLResponse)
async def review_task(request: Request, task_id: str):
    """Individual task review interface."""
    
    task = hitl_app.active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return templates.TemplateResponse("task_review.html", {
        "request": request,
        "task": task,
        "reviewers": hitl_app.reviewers
    })

@app.post("/api/tasks/{task_id}/first-touch")
async def record_first_touch(task_id: str, reviewer_id: str = Form(...)):
    """Record first touch for SLA tracking."""
    
    success = hitl_app.record_first_touch(task_id, reviewer_id)
    
    if success:
        return {"success": True, "message": "First touch recorded"}
    else:
        raise HTTPException(status_code=400, detail="Failed to record first touch")

@app.post("/api/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    reviewer_id: str = Form(...),
    field_corrections: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Complete task with corrections."""
    
    try:
        corrections_data = json.loads(field_corrections)
        
        success = hitl_app.complete_task_with_feedback(
            task_id=task_id,
            reviewer_id=reviewer_id,
            field_corrections=corrections_data,
            notes=notes
        )
        
        if success:
            return {"success": True, "message": "Task completed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to complete task")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid field corrections format")

@app.post("/api/tasks/{task_id}/assign")
async def assign_task(task_id: str, reviewer_id: str = Form(...)):
    """Assign task to reviewer."""
    
    success = hitl_app.assign_task(task_id, reviewer_id)
    
    if success:
        return {"success": True, "message": f"Task assigned to {reviewer_id}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to assign task")

@app.get("/api/tasks/{task_id}/pdf-page/{page_number}")
async def get_pdf_page(task_id: str, page_number: int):
    """Get PDF page image with bounding box overlays."""
    
    task = hitl_app.active_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if page_number >= len(task.pdf_pages):
        raise HTTPException(status_code=404, detail="Page not found")
    
    return {
        "page_image": task.pdf_pages[page_number],  # Base64 encoded
        "page_number": page_number,
        "total_pages": len(task.pdf_pages)
    }

@app.post("/api/create-sample-task")
async def create_sample_task():
    """Create a sample task for testing."""
    
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
            confidence=0.65,
            bounding_box=BoundingBox(x=100, y=250, width=100, height=20, page_number=0, confidence=0.65),
            extraction_method="Template",
            rule_violations=["Amount format validation failed"],
            suggested_corrections=["$1234.56"]
        )
    ]
    
    validation_errors = ["Total amount format validation failed"]
    
    task_id = hitl_app.create_enhanced_review_task(
        document_id=f"DOC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        document_type="invoice",
        extracted_fields=sample_fields,
        validation_errors=validation_errors,
        priority=2
    )
    
    return {"task_id": task_id, "message": "Sample task created"}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced HITL Web Interface")
    print("======================================")
    print("Features:")
    print("  • Interactive PDF viewer with bounding boxes")
    print("  • Field correction interface") 
    print("  • SLA tracking and assignment")
    print("  • Training data collection")
    print("  • Real-time queue monitoring")
    print("")
    print("🌐 Access at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)