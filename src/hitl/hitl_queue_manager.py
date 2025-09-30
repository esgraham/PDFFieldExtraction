"""
Human-in-the-Loop (HITL) Queue Management System

This module implements a queue-based system for routing documents that require
human review, with poison queue pattern for robust error handling and retries.

Features:
- Queue management for HITL processing
- Poison queue pattern for failed documents
- Retry logic with exponential backoff
- Priority-based routing
- Integration with Azure Service Bus (optional)
- Local queue fallback for development
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
import pickle
import threading
from queue import Queue, PriorityQueue, Empty
import hashlib

# Optional Azure Service Bus integration
try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
    from azure.servicebus.exceptions import ServiceBusError
    AZURE_SERVICEBUS_AVAILABLE = True
except ImportError:
    AZURE_SERVICEBUS_AVAILABLE = False

logger = logging.getLogger(__name__)

class HITLPriority(Enum):
    """Priority levels for HITL processing."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0

class HITLStatus(Enum):
    """Status of HITL processing."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    POISONED = "poisoned"
    CANCELLED = "cancelled"

class HITLReason(Enum):
    """Reasons for HITL routing."""
    LOW_CONFIDENCE = "low_confidence"
    VALIDATION_ERROR = "validation_error"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    EXTRACTION_FAILURE = "extraction_failure"
    MANUAL_REVIEW_REQUESTED = "manual_review"

@dataclass
class HITLTask:
    """Individual HITL task."""
    task_id: str
    document_id: str
    document_path: str
    template_type: str
    priority: HITLPriority
    reason: HITLReason
    created_at: datetime
    
    # Processing details
    extraction_result: Optional[Dict] = None
    validation_errors: List[Dict] = None
    confidence_score: float = 0.0
    
    # Queue management
    attempts: int = 0
    max_attempts: int = 3
    next_retry: Optional[datetime] = None
    last_error: Optional[str] = None
    
    # Human review
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    corrected_fields: Optional[Dict] = None
    
    # Status tracking
    status: HITLStatus = HITLStatus.PENDING
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
    
    def __lt__(self, other):
        """Priority queue comparison.""" 
        return self.priority.value < other.priority.value

@dataclass
class HITLQueueConfig:
    """Configuration for HITL queue system."""
    # Queue settings
    max_queue_size: int = 1000
    poison_queue_size: int = 100
    processing_timeout: int = 3600  # 1 hour
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_delay_base: int = 60  # Base delay in seconds
    retry_exponential_base: float = 2.0
    
    # Azure Service Bus settings (optional)
    connection_string: Optional[str] = None
    queue_name: str = "hitl-processing"
    poison_queue_name: str = "hitl-poison"
    
    # Local storage settings
    local_queue_path: str = "./data/hitl_queue"
    enable_persistence: bool = True
    
    # Processing settings
    batch_size: int = 10
    polling_interval: int = 5
    enable_auto_retry: bool = True

class HITLQueueManager:
    """
    Main HITL queue management system with poison queue pattern.
    """
    
    def __init__(self, config: HITLQueueConfig):
        """
        Initialize HITL queue manager.
        
        Args:
            config: Queue configuration
        """
        self.config = config
        
        # Initialize queues
        self.main_queue: PriorityQueue = PriorityQueue(maxsize=config.max_queue_size)
        self.poison_queue: Queue = Queue(maxsize=config.poison_queue_size)
        self.in_progress: Dict[str, HITLTask] = {}
        
        # Azure Service Bus client (if available)
        self.service_bus_client = None
        if config.connection_string and AZURE_SERVICEBUS_AVAILABLE:
            try:
                self.service_bus_client = ServiceBusClient.from_connection_string(
                    config.connection_string
                )
                logger.info("Azure Service Bus client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Service Bus: {e}")
        
        # Local persistence
        self.queue_path = Path(config.local_queue_path)
        if config.enable_persistence:
            self.queue_path.mkdir(parents=True, exist_ok=True)
            self._load_persisted_tasks()
        
        # Processing state
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            "tasks_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_poisoned": 0,
            "total_processing_time": 0.0,
            "average_retry_count": 0.0
        }
        
        # Task processors (can be overridden)
        self.task_processors: Dict[str, Callable] = {}
        
        logger.info("HITL queue manager initialized")
    
    def enqueue_task(
        self,
        document_id: str,
        document_path: str,
        template_type: str,
        reason: HITLReason,
        extraction_result: Optional[Dict] = None,
        validation_errors: Optional[List[Dict]] = None,
        confidence_score: float = 0.0,
        priority: HITLPriority = HITLPriority.NORMAL
    ) -> str:
        """
        Enqueue a document for HITL processing.
        
        Args:
            document_id: Unique document identifier
            document_path: Path to the document file
            template_type: Document template type
            reason: Reason for HITL routing
            extraction_result: Original extraction result
            validation_errors: List of validation errors
            confidence_score: Overall confidence score
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = HITLTask(
            task_id=task_id,
            document_id=document_id,
            document_path=document_path,
            template_type=template_type,
            priority=priority,
            reason=reason,
            created_at=datetime.now(),
            extraction_result=extraction_result,
            validation_errors=validation_errors or [],
            confidence_score=confidence_score
        )
        
        try:
            # Try Azure Service Bus first
            if self.service_bus_client:
                self._enqueue_to_service_bus(task)
            else:
                # Fallback to local queue
                self.main_queue.put(task, timeout=5)
            
            # Persist task if enabled
            if self.config.enable_persistence:
                self._persist_task(task)
            
            self.stats["tasks_received"] += 1
            
            logger.info(f"Enqueued HITL task {task_id} for document {document_id} (reason: {reason.value})")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue HITL task: {e}")
            raise
    
    def _enqueue_to_service_bus(self, task: HITLTask):
        """Enqueue task to Azure Service Bus."""
        if not self.service_bus_client:
            raise RuntimeError("Service Bus client not available")
        
        try:
            sender = self.service_bus_client.get_queue_sender(self.config.queue_name)
            
            # Serialize task
            message_body = json.dumps(asdict(task), default=str)
            message = ServiceBusMessage(
                body=message_body,
                message_id=task.task_id,
                user_properties={
                    "priority": task.priority.value,
                    "reason": task.reason.value,
                    "document_id": task.document_id
                }
            )
            
            sender.send_messages(message)
            sender.close()
            
        except ServiceBusError as e:
            logger.error(f"Service Bus error: {e}")
            # Fallback to local queue
            self.main_queue.put(task, timeout=5)
    
    def start_processing(self):
        """Start the HITL queue processing."""
        if self.is_running:
            logger.warning("HITL processing already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info("HITL queue processing started")
    
    def stop_processing(self):
        """Stop the HITL queue processing."""
        self.is_running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        
        # Persist any remaining tasks
        if self.config.enable_persistence:
            self._persist_all_tasks()
        
        logger.info("HITL queue processing stopped")
    
    def _process_queue(self):
        """Main queue processing loop."""
        logger.info("Starting HITL queue processing loop")
        
        while self.is_running:
            try:
                # Process local queue
                self._process_local_queue()
                
                # Process Service Bus queue if available
                if self.service_bus_client:
                    self._process_service_bus_queue()
                
                # Handle retries
                if self.config.enable_auto_retry:
                    self._process_retries()
                
                # Clean up timed-out tasks
                self._cleanup_timed_out_tasks()
                
                # Sleep between polling intervals
                time.sleep(self.config.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in queue processing loop: {e}")
                time.sleep(self.config.polling_interval)
    
    def _process_local_queue(self):
        """Process tasks from local priority queue."""
        processed_count = 0
        
        while processed_count < self.config.batch_size and not self.main_queue.empty():
            try:
                task = self.main_queue.get(timeout=1)
                self._process_task(task)
                processed_count += 1
                
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error processing local queue task: {e}")
    
    def _process_service_bus_queue(self):
        """Process tasks from Azure Service Bus queue."""
        if not self.service_bus_client:
            return
        
        try:
            receiver = self.service_bus_client.get_queue_receiver(self.config.queue_name)
            
            messages = receiver.receive_messages(
                max_message_count=self.config.batch_size,
                max_wait_time=5
            )
            
            for message in messages:
                try:
                    # Deserialize task
                    task_data = json.loads(str(message))
                    task = self._deserialize_task(task_data)
                    
                    # Process task
                    self._process_task(task)
                    
                    # Complete message
                    receiver.complete_message(message)
                    
                except Exception as e:
                    logger.error(f"Error processing Service Bus message: {e}")
                    # Abandon message for retry
                    receiver.abandon_message(message)
            
            receiver.close()
            
        except ServiceBusError as e:
            logger.error(f"Service Bus processing error: {e}")
    
    def _process_task(self, task: HITLTask):
        """Process a single HITL task."""
        start_time = time.time()
        
        try:
            # Mark task as in progress
            task.status = HITLStatus.IN_PROGRESS
            task.assigned_at = datetime.now()
            self.in_progress[task.task_id] = task
            
            logger.info(f"Processing HITL task {task.task_id} (attempt {task.attempts + 1})")
            
            # Find appropriate processor
            processor = self.task_processors.get(task.template_type)
            if not processor:
                processor = self._default_task_processor
            
            # Process the task
            result = processor(task)
            
            if result:
                # Task completed successfully
                task.status = HITLStatus.COMPLETED
                task.reviewed_at = datetime.now()
                self.stats["tasks_completed"] += 1
                
                # Remove from in-progress
                self.in_progress.pop(task.task_id, None)
                
                logger.info(f"HITL task {task.task_id} completed successfully")
                
            else:
                # Task failed, handle retry
                self._handle_task_failure(task, "Processing returned False")
            
        except Exception as e:
            # Task failed with exception
            self._handle_task_failure(task, str(e))
        
        finally:
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
    
    def _handle_task_failure(self, task: HITLTask, error_message: str):
        """Handle task processing failure."""
        task.attempts += 1
        task.last_error = error_message
        
        logger.warning(f"HITL task {task.task_id} failed (attempt {task.attempts}): {error_message}")
        
        if task.attempts >= task.max_attempts:
            # Move to poison queue
            task.status = HITLStatus.POISONED
            self.poison_queue.put(task)
            self.stats["tasks_poisoned"] += 1
            
            # Remove from in-progress
            self.in_progress.pop(task.task_id, None)
            
            logger.error(f"HITL task {task.task_id} moved to poison queue after {task.attempts} attempts")
            
            # Send poison queue alert
            self._send_poison_queue_alert(task)
            
        else:
            # Schedule for retry
            retry_delay = self._calculate_retry_delay(task.attempts)
            task.next_retry = datetime.now() + timedelta(seconds=retry_delay)
            task.status = HITLStatus.FAILED
            
            # Remove from in-progress (will be retried later)
            self.in_progress.pop(task.task_id, None)
            
            logger.info(f"HITL task {task.task_id} scheduled for retry in {retry_delay} seconds")
        
        self.stats["tasks_failed"] += 1
    
    def _calculate_retry_delay(self, attempt: int) -> int:
        """Calculate retry delay with exponential backoff."""
        base_delay = self.config.retry_delay_base
        exponential_factor = self.config.retry_exponential_base ** (attempt - 1)
        return int(base_delay * exponential_factor)
    
    def _process_retries(self):
        """Process tasks that are ready for retry."""
        current_time = datetime.now()
        
        # Find tasks ready for retry
        retry_tasks = []
        for task_file in self.queue_path.glob("*.task"):
            try:
                task = self._load_task_from_file(task_file)
                if (task.status == HITLStatus.FAILED and 
                    task.next_retry and 
                    task.next_retry <= current_time):
                    retry_tasks.append(task)
            except Exception as e:
                logger.error(f"Error loading task for retry: {e}")
        
        # Re-enqueue retry tasks
        for task in retry_tasks:
            try:
                task.status = HITLStatus.PENDING
                task.next_retry = None
                
                if self.service_bus_client:
                    self._enqueue_to_service_bus(task)
                else:
                    self.main_queue.put(task, timeout=1)
                
                logger.info(f"Re-queued task {task.task_id} for retry")
                
            except Exception as e:
                logger.error(f"Failed to re-queue task {task.task_id}: {e}")
    
    def _cleanup_timed_out_tasks(self):
        """Clean up tasks that have timed out."""
        current_time = datetime.now()
        timeout_threshold = current_time - timedelta(seconds=self.config.processing_timeout)
        
        timed_out_tasks = []
        for task_id, task in self.in_progress.items():
            if task.assigned_at and task.assigned_at < timeout_threshold:
                timed_out_tasks.append(task_id)
        
        for task_id in timed_out_tasks:
            task = self.in_progress.pop(task_id)
            self._handle_task_failure(task, "Processing timeout")
            logger.warning(f"Task {task_id} timed out and moved to retry queue")
    
    def _default_task_processor(self, task: HITLTask) -> bool:
        """
        Default task processor - creates a human review request.
        
        This is a placeholder that would typically integrate with a
        human review interface or external system.
        """
        logger.info(f"Default processor handling task {task.task_id}")
        
        # Create human review request file
        review_file = self.queue_path / f"review_{task.task_id}.json"
        
        review_data = {
            "task_id": task.task_id,
            "document_id": task.document_id,
            "document_path": task.document_path,
            "template_type": task.template_type,
            "reason": task.reason.value,
            "confidence_score": task.confidence_score,
            "extraction_result": task.extraction_result,
            "validation_errors": task.validation_errors,
            "created_at": task.created_at.isoformat(),
            "instructions": self._generate_review_instructions(task)
        }
        
        with open(review_file, 'w') as f:
            json.dump(review_data, f, indent=2)
        
        logger.info(f"Created human review request: {review_file}")
        
        # In a real implementation, this would:
        # 1. Send notification to human reviewers
        # 2. Create UI task in review system
        # 3. Wait for human review completion
        # 4. Return True when reviewed, False if failed
        
        # For demo purposes, return True (task "completed")
        return True
    
    def _generate_review_instructions(self, task: HITLTask) -> List[str]:
        """Generate human review instructions based on task details."""
        instructions = []
        
        if task.reason == HITLReason.LOW_CONFIDENCE:
            instructions.append(f"Document has low confidence score ({task.confidence_score:.2f})")
            instructions.append("Please verify all extracted field values")
        
        elif task.reason == HITLReason.VALIDATION_ERROR:
            instructions.append("Document failed validation rules:")
            for error in task.validation_errors:
                instructions.append(f"  - {error.get('message', 'Unknown error')}")
        
        elif task.reason == HITLReason.MISSING_REQUIRED_FIELD:
            instructions.append("Required fields are missing or have low confidence")
            instructions.append("Please extract missing field values manually")
        
        elif task.reason == HITLReason.BUSINESS_RULE_VIOLATION:
            instructions.append("Document violates business rules:")
            for error in task.validation_errors:
                if not error.get('is_valid', True):
                    instructions.append(f"  - {error.get('message', 'Unknown rule violation')}")
        
        instructions.append("\nActions required:")
        instructions.append("1. Review the document image")
        instructions.append("2. Verify or correct extracted field values")
        instructions.append("3. Ensure all required fields are present")
        instructions.append("4. Validate business rule compliance")
        instructions.append("5. Mark task as completed when done")
        
        return instructions
    
    def _send_poison_queue_alert(self, task: HITLTask):
        """Send alert for poison queue items."""
        alert_data = {
            "alert_type": "poison_queue",
            "task_id": task.task_id,
            "document_id": task.document_id,
            "template_type": task.template_type,
            "reason": task.reason.value,
            "attempts": task.attempts,
            "last_error": task.last_error,
            "created_at": task.created_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save alert to file (in production, this would send to monitoring system)
        alert_file = self.queue_path / f"poison_alert_{task.task_id}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.critical(f"POISON QUEUE ALERT: Task {task.task_id} failed after {task.attempts} attempts")
    
    def _persist_task(self, task: HITLTask):
        """Persist task to disk."""
        if not self.config.enable_persistence:
            return
        
        task_file = self.queue_path / f"{task.task_id}.task"
        
        with open(task_file, 'wb') as f:
            pickle.dump(task, f)
    
    def _persist_all_tasks(self):
        """Persist all in-progress tasks."""
        for task in self.in_progress.values():
            self._persist_task(task)
    
    def _load_persisted_tasks(self):
        """Load persisted tasks from disk."""
        if not self.queue_path.exists():
            return
        
        for task_file in self.queue_path.glob("*.task"):
            try:
                task = self._load_task_from_file(task_file)
                
                # Re-queue pending tasks
                if task.status == HITLStatus.PENDING:
                    self.main_queue.put(task)
                # Keep in-progress tasks in the in-progress dict
                elif task.status == HITLStatus.IN_PROGRESS:
                    self.in_progress[task.task_id] = task
                
                logger.info(f"Loaded persisted task {task.task_id}")
                
            except Exception as e:
                logger.error(f"Failed to load persisted task {task_file}: {e}")
    
    def _load_task_from_file(self, task_file: Path) -> HITLTask:
        """Load task from file."""
        with open(task_file, 'rb') as f:
            return pickle.load(f)
    
    def _deserialize_task(self, task_data: Dict) -> HITLTask:
        """Deserialize task from JSON data."""
        # Convert string enums back to enum objects
        task_data['priority'] = HITLPriority(task_data['priority'])
        task_data['reason'] = HITLReason(task_data['reason'])
        task_data['status'] = HITLStatus(task_data['status'])
        
        # Convert string dates back to datetime objects
        task_data['created_at'] = datetime.fromisoformat(task_data['created_at'])
        if task_data.get('assigned_at'):
            task_data['assigned_at'] = datetime.fromisoformat(task_data['assigned_at'])
        if task_data.get('reviewed_at'):
            task_data['reviewed_at'] = datetime.fromisoformat(task_data['reviewed_at'])
        if task_data.get('next_retry'):
            task_data['next_retry'] = datetime.fromisoformat(task_data['next_retry'])
        
        return HITLTask(**task_data)
    
    def register_task_processor(self, template_type: str, processor: Callable[[HITLTask], bool]):
        """Register a custom task processor for a specific template type."""
        self.task_processors[template_type] = processor
        logger.info(f"Registered custom processor for template type: {template_type}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        return {
            "queue_size": self.main_queue.qsize(),
            "poison_queue_size": self.poison_queue.qsize(),
            "in_progress_count": len(self.in_progress),
            "is_running": self.is_running,
            "statistics": self.stats,
            "config": {
                "max_queue_size": self.config.max_queue_size,
                "max_retry_attempts": self.config.max_retry_attempts,
                "processing_timeout": self.config.processing_timeout
            }
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check in-progress tasks
        if task_id in self.in_progress:
            task = self.in_progress[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "attempts": task.attempts,
                "assigned_at": task.assigned_at.isoformat() if task.assigned_at else None,
                "last_error": task.last_error
            }
        
        # Check persisted tasks
        if self.config.enable_persistence:
            task_file = self.queue_path / f"{task_id}.task"
            if task_file.exists():
                try:
                    task = self._load_task_from_file(task_file)
                    return {
                        "task_id": task_id,
                        "status": task.status.value,
                        "attempts": task.attempts,
                        "created_at": task.created_at.isoformat(),
                        "last_error": task.last_error
                    }
                except Exception as e:
                    logger.error(f"Error loading task status: {e}")
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        # Remove from in-progress
        if task_id in self.in_progress:
            task = self.in_progress.pop(task_id)
            task.status = HITLStatus.CANCELLED
            return True
        
        # Remove persisted task file
        if self.config.enable_persistence:
            task_file = self.queue_path / f"{task_id}.task"
            if task_file.exists():
                try:
                    task = self._load_task_from_file(task_file)
                    task.status = HITLStatus.CANCELLED
                    self._persist_task(task)
                    return True
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")
        
        return False