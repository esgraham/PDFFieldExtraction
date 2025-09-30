"""
Integrated Field Extraction System

This module combines field extraction, validation, and HITL routing into
a complete processing pipeline for document analysis.

Features:
- Template-based field extraction
- Business rules validation
- Confidence scoring and routing
- HITL queue integration
- OCR integration with Azure Document Intelligence
- Comprehensive error handling and retry logic
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from .field_extraction import (
    FieldExtractor, 
    DocumentTemplate, 
    ExtractionResult,
    HITLQueueConfig
)
from .hitl_queue_manager import (
    HITLQueueManager,
    HITLPriority,
    HITLReason,
    HITLTask
)
from .azure_document_intelligence import DocumentAnalysisResult
from .ocr_integration import EnhancedPDFListener, CompleteDocumentAnalysis

logger = logging.getLogger(__name__)

@dataclass
class FieldExtractionConfig:
    """Configuration for integrated field extraction system."""
    # Field extraction settings
    schema_directory: str = "./config/schemas"
    confidence_threshold: float = 0.7
    hitl_threshold: float = 0.6
    enable_business_rules: bool = True
    
    # HITL queue settings
    hitl_queue_config: HITLQueueConfig = None
    enable_hitl_processing: bool = True
    
    # Processing settings
    output_directory: str = "./processed_documents"
    enable_field_correction: bool = True
    auto_approve_high_confidence: bool = True
    high_confidence_threshold: float = 0.95
    
    def __post_init__(self):
        if self.hitl_queue_config is None:
            self.hitl_queue_config = HITLQueueConfig()

@dataclass
class ProcessingResult:
    """Complete processing result with field extraction and routing decision."""
    document_id: str
    extraction_result: ExtractionResult
    requires_hitl: bool
    hitl_task_id: Optional[str] = None
    routing_reason: Optional[HITLReason] = None
    processing_time: float = 0.0
    auto_approved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_id": self.document_id,
            "requires_hitl": self.requires_hitl,
            "hitl_task_id": self.hitl_task_id,
            "routing_reason": self.routing_reason.value if self.routing_reason else None,
            "processing_time": self.processing_time,
            "auto_approved": self.auto_approved,
            "overall_confidence": self.extraction_result.overall_confidence,
            "fields_extracted": len(self.extraction_result.extracted_fields),
            "validation_errors": len([
                r for r in self.extraction_result.validation_results 
                if not r.is_valid
            ]),
            "timestamp": self.extraction_result.timestamp.isoformat()
        }

class IntegratedFieldExtractor:
    """
    Integrated field extraction system with complete processing pipeline.
    
    Combines OCR, field extraction, validation, and HITL routing into
    a single processing system with comprehensive error handling.
    """
    
    def __init__(
        self,
        config: FieldExtractionConfig,
        ocr_listener: Optional[EnhancedPDFListener] = None
    ):
        """
        Initialize integrated field extraction system.
        
        Args:
            config: Field extraction configuration
            ocr_listener: Optional OCR listener for full pipeline
        """
        self.config = config
        self.ocr_listener = ocr_listener
        
        # Initialize field extractor
        self.field_extractor = FieldExtractor(
            schema_directory=config.schema_directory,
            enable_business_rules=config.enable_business_rules,
            confidence_threshold=config.confidence_threshold,
            hitl_threshold=config.hitl_threshold
        )
        
        # Initialize HITL queue manager
        self.hitl_manager = None
        if config.enable_hitl_processing:
            self.hitl_manager = HITLQueueManager(config.hitl_queue_config)
            self._register_custom_processors()
        
        # Output directory
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "auto_approved": 0,
            "sent_to_hitl": 0,
            "processing_errors": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0
        }
        
        logger.info("Integrated field extraction system initialized")
    
    async def process_document_complete(
        self,
        document_path: str,
        document_id: Optional[str] = None,
        template_type: Optional[DocumentTemplate] = None
    ) -> ProcessingResult:
        """
        Process document with complete field extraction pipeline.
        
        Args:
            document_path: Path to the document
            document_id: Optional document identifier
            template_type: Optional document template type
            
        Returns:
            Complete processing result
        """
        start_time = time.time()
        
        if document_id is None:
            document_id = Path(document_path).stem
        
        logger.info(f"Starting complete processing for document: {document_id}")
        
        try:
            # Step 1: OCR Processing (if OCR listener is available)
            ocr_result = None
            if self.ocr_listener:
                complete_analysis = self.ocr_listener.process_pdf_complete(
                    document_id, document_path
                )
                ocr_result = complete_analysis.ocr_result
                
                # Use classification result for template selection
                if template_type is None:
                    template_type = self._map_classification_to_template(
                        complete_analysis.document_class
                    )
            
            # Step 2: Field Extraction
            if template_type is None:
                template_type = DocumentTemplate.CUSTOM  # Default fallback
            
            extraction_result = self.field_extractor.extract_fields(
                ocr_result or self._create_mock_ocr_result(document_path),
                template_type,
                document_id
            )
            
            # Step 3: Routing Decision
            routing_decision = self._make_routing_decision(extraction_result)
            
            # Step 4: Process based on routing decision
            processing_result = await self._process_routing_decision(
                document_id,
                document_path,
                extraction_result,
                routing_decision,
                template_type
            )
            
            # Step 5: Save results
            await self._save_processing_results(processing_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            processing_result.processing_time = processing_time
            self._update_stats(processing_result)
            
            logger.info(
                f"Document processing completed: {document_id} "
                f"(confidence: {extraction_result.overall_confidence:.2f}, "
                f"HITL: {'yes' if processing_result.requires_hitl else 'no'})"
            )
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            self.stats["processing_errors"] += 1
            raise
    
    def _map_classification_to_template(self, document_class: str) -> DocumentTemplate:
        """Map document classification result to template type."""
        mapping = {
            "invoice": DocumentTemplate.INVOICE,
            "receipt": DocumentTemplate.RECEIPT,
            "purchase_order": DocumentTemplate.PURCHASE_ORDER,
            "tax_form": DocumentTemplate.TAX_FORM,
            "contract": DocumentTemplate.CONTRACT,
            "form": DocumentTemplate.FORM_APPLICATION,
            "bank_statement": DocumentTemplate.BANK_STATEMENT,
            "insurance_claim": DocumentTemplate.INSURANCE_CLAIM
        }
        
        return mapping.get(document_class.lower(), DocumentTemplate.CUSTOM)
    
    def _create_mock_ocr_result(self, document_path: str) -> Any:
        """Create mock OCR result for testing without full OCR pipeline."""
        from .azure_document_intelligence import DocumentAnalysisResult
        
        # In a real implementation without OCR, you might:
        # 1. Use basic text extraction
        # 2. Load cached OCR results
        # 3. Use alternative OCR engines
        
        return DocumentAnalysisResult(
            text_blocks=[],
            tables=[],
            fields=[],
            full_text=f"Mock text content for {document_path}",
            pages=1,
            language="en",
            processing_time=0.1,
            model_used="mock",
            confidence_scores={"text": 0.8}
        )
    
    def _make_routing_decision(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """Make routing decision based on extraction results."""
        decision = {
            "requires_hitl": extraction_result.requires_hitl,
            "routing_reason": None,
            "priority": HITLPriority.NORMAL,
            "auto_approve": False
        }
        
        # Determine routing reason
        if extraction_result.requires_hitl:
            # Check for specific reasons
            if extraction_result.overall_confidence < self.config.hitl_threshold:
                decision["routing_reason"] = HITLReason.LOW_CONFIDENCE
                decision["priority"] = HITLPriority.HIGH
            
            # Check for validation errors
            error_count = sum(
                1 for r in extraction_result.validation_results 
                if not r.is_valid and r.severity.value == "error"
            )
            if error_count > 0:
                decision["routing_reason"] = HITLReason.VALIDATION_ERROR
                decision["priority"] = HITLPriority.HIGH
            
            # Check for missing required fields
            required_fields_missing = any(
                field.field_name in ["invoice_number", "total_amount", "vendor_name"]
                and field.confidence < 0.8
                for field in extraction_result.extracted_fields
            )
            if required_fields_missing:
                decision["routing_reason"] = HITLReason.MISSING_REQUIRED_FIELD
                decision["priority"] = HITLPriority.URGENT
        
        else:
            # Check for auto-approval
            if (self.config.auto_approve_high_confidence and 
                extraction_result.overall_confidence >= self.config.high_confidence_threshold):
                decision["auto_approve"] = True
        
        return decision
    
    async def _process_routing_decision(
        self,
        document_id: str,
        document_path: str,
        extraction_result: ExtractionResult,
        routing_decision: Dict[str, Any],
        template_type: DocumentTemplate
    ) -> ProcessingResult:
        """Process the routing decision and create appropriate result."""
        
        processing_result = ProcessingResult(
            document_id=document_id,
            extraction_result=extraction_result,
            requires_hitl=routing_decision["requires_hitl"],
            routing_reason=routing_decision.get("routing_reason"),
            auto_approved=routing_decision.get("auto_approve", False)
        )
        
        if routing_decision["requires_hitl"] and self.hitl_manager:
            # Enqueue for HITL processing
            hitl_task_id = self.hitl_manager.enqueue_task(
                document_id=document_id,
                document_path=document_path,
                template_type=template_type.value,
                reason=routing_decision["routing_reason"],
                extraction_result=self._serialize_extraction_result(extraction_result),
                validation_errors=[
                    {
                        "field_name": r.field_name,
                        "is_valid": r.is_valid,
                        "severity": r.severity.value,
                        "message": r.message,
                        "rule_name": r.rule_name
                    }
                    for r in extraction_result.validation_results
                ],
                confidence_score=extraction_result.overall_confidence,
                priority=routing_decision["priority"]
            )
            
            processing_result.hitl_task_id = hitl_task_id
            
            logger.info(f"Document {document_id} routed to HITL queue (task: {hitl_task_id})")
        
        elif processing_result.auto_approved:
            logger.info(f"Document {document_id} auto-approved (confidence: {extraction_result.overall_confidence:.2f})")
        
        return processing_result
    
    def _serialize_extraction_result(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """Serialize extraction result for HITL queue."""
        return {
            "document_id": extraction_result.document_id,
            "template_type": extraction_result.template_type.value,
            "overall_confidence": extraction_result.overall_confidence,
            "extracted_fields": [
                {
                    "field_name": field.field_name,
                    "value": str(field.value),
                    "confidence": field.confidence,
                    "source_text": field.source_text,
                    "extraction_method": field.extraction_method,
                    "normalized_value": str(field.normalized_value) if field.normalized_value else None
                }
                for field in extraction_result.extracted_fields
            ],
            "extracted_tables": extraction_result.extracted_tables,
            "processing_time": extraction_result.processing_time,
            "timestamp": extraction_result.timestamp.isoformat()
        }
    
    async def _save_processing_results(self, processing_result: ProcessingResult):
        """Save processing results to output directory."""
        
        # Create output file
        output_file = self.output_path / f"{processing_result.document_id}_field_extraction.json"
        
        # Prepare output data
        output_data = {
            "processing_summary": processing_result.to_dict(),
            "extraction_details": {
                "document_id": processing_result.extraction_result.document_id,
                "template_type": processing_result.extraction_result.template_type.value,
                "overall_confidence": processing_result.extraction_result.overall_confidence,
                "processing_time": processing_result.extraction_result.processing_time,
                "timestamp": processing_result.extraction_result.timestamp.isoformat(),
                
                "extracted_fields": [
                    {
                        "field_name": field.field_name,
                        "value": field.value,
                        "normalized_value": field.normalized_value,
                        "confidence": field.confidence,
                        "source_text": field.source_text,
                        "extraction_method": field.extraction_method,
                        "bounding_box": field.bounding_box
                    }
                    for field in processing_result.extraction_result.extracted_fields
                ],
                
                "extracted_tables": processing_result.extraction_result.extracted_tables,
                
                "validation_results": [
                    {
                        "field_name": result.field_name,
                        "is_valid": result.is_valid,
                        "severity": result.severity.value,
                        "message": result.message,
                        "rule_name": result.rule_name,
                        "suggested_value": result.suggested_value
                    }
                    for result in processing_result.extraction_result.validation_results
                ]
            }
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Processing results saved to: {output_file}")
    
    def _register_custom_processors(self):
        """Register custom task processors for different document types."""
        if not self.hitl_manager:
            return
        
        # Register processors for different template types
        self.hitl_manager.register_task_processor(
            "invoice", self._process_invoice_hitl_task
        )
        self.hitl_manager.register_task_processor(
            "receipt", self._process_receipt_hitl_task
        )
        # Add more processors as needed
    
    def _process_invoice_hitl_task(self, task: HITLTask) -> bool:
        """Custom processor for invoice HITL tasks."""
        logger.info(f"Processing invoice HITL task: {task.task_id}")
        
        # Create specialized review request for invoices
        review_data = {
            "task_type": "invoice_review",
            "document_id": task.document_id,
            "priority_fields": [
                "invoice_number", "vendor_name", "total_amount", "invoice_date"
            ],
            "validation_focus": [
                "Verify total calculation accuracy",
                "Confirm vendor name is complete",
                "Validate invoice number format",
                "Check date format and reasonableness"
            ],
            "business_rules": [
                "Total should equal subtotal + tax",
                "Due date should be after invoice date",
                "Line items should sum to subtotal"
            ]
        }
        
        # Save specialized review request
        review_file = Path(self.config.hitl_queue_config.local_queue_path) / f"invoice_review_{task.task_id}.json"
        with open(review_file, 'w') as f:
            json.dump(review_data, f, indent=2)
        
        logger.info(f"Created specialized invoice review: {review_file}")
        return True
    
    def _process_receipt_hitl_task(self, task: HITLTask) -> bool:
        """Custom processor for receipt HITL tasks."""
        logger.info(f"Processing receipt HITL task: {task.task_id}")
        
        # Create specialized review request for receipts
        review_data = {
            "task_type": "receipt_review",
            "document_id": task.document_id,
            "priority_fields": [
                "merchant_name", "total_amount", "transaction_date"
            ],
            "validation_focus": [
                "Verify merchant name is legible",
                "Confirm total amount accuracy",
                "Check transaction date",
                "Validate item prices if visible"
            ],
            "business_rules": [
                "Items should sum to total amount",
                "Transaction date should be reasonable"
            ]
        }
        
        # Save specialized review request
        review_file = Path(self.config.hitl_queue_config.local_queue_path) / f"receipt_review_{task.task_id}.json"
        with open(review_file, 'w') as f:
            json.dump(review_data, f, indent=2)
        
        logger.info(f"Created specialized receipt review: {review_file}")
        return True
    
    def _update_stats(self, processing_result: ProcessingResult):
        """Update processing statistics."""
        self.stats["documents_processed"] += 1
        self.stats["total_processing_time"] += processing_result.processing_time
        
        if processing_result.auto_approved:
            self.stats["auto_approved"] += 1
        
        if processing_result.requires_hitl:
            self.stats["sent_to_hitl"] += 1
        
        # Update average confidence
        current_confidence = processing_result.extraction_result.overall_confidence
        total_docs = self.stats["documents_processed"]
        current_avg = self.stats["average_confidence"]
        
        self.stats["average_confidence"] = (
            (current_avg * (total_docs - 1) + current_confidence) / total_docs
        )
    
    def start_hitl_processing(self):
        """Start HITL queue processing."""
        if self.hitl_manager:
            self.hitl_manager.start_processing()
            logger.info("HITL processing started")
    
    def stop_hitl_processing(self):
        """Stop HITL queue processing."""
        if self.hitl_manager:
            self.hitl_manager.stop_processing()
            logger.info("HITL processing stopped")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.stats.copy()
        
        # Add field extractor stats
        field_stats = self.field_extractor.get_statistics()
        stats["field_extraction"] = field_stats
        
        # Add HITL queue stats
        if self.hitl_manager:
            hitl_stats = self.hitl_manager.get_queue_status()
            stats["hitl_queue"] = hitl_stats
        
        # Calculate derived metrics
        if stats["documents_processed"] > 0:
            stats["auto_approval_rate"] = stats["auto_approved"] / stats["documents_processed"]
            stats["hitl_rate"] = stats["sent_to_hitl"] / stats["documents_processed"]
            stats["average_processing_time"] = stats["total_processing_time"] / stats["documents_processed"]
        
        return stats
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific HITL task."""
        if self.hitl_manager:
            return self.hitl_manager.get_task_status(task_id)
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending HITL task."""
        if self.hitl_manager:
            return self.hitl_manager.cancel_task(task_id)
        return False