"""
OCR Integration Module

Integrates Azure Document Intelligence OCR with the existing PDF processing pipeline,
including preprocessing, classification, and Azure Storage monitoring.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json

from .azure_document_intelligence import (
    AzureDocumentIntelligenceOCR,
    DocumentIntelligenceIntegration,
    DocumentType,
    DocumentAnalysisResult,
    ExtractedText,
    ExtractedTable,
    ExtractedField
)
from .pdf_preprocessing import PDFPreprocessor
from .classification_integration import ClassificationIntegratedListener

logger = logging.getLogger(__name__)

@dataclass
class CompleteDocumentAnalysis:
    """Complete document analysis including OCR, classification, and metadata."""
    # OCR Results
    ocr_result: DocumentAnalysisResult
    
    # Classification Results
    document_class: str
    classification_confidence: float
    classification_features: Dict[str, Any]
    
    # Processing Metadata
    processing_pipeline: List[str]
    total_processing_time: float
    preprocessing_applied: bool
    custom_fields: Dict[str, Any]
    
    # Original Document Info
    original_file_path: str
    file_size_bytes: int
    pages_processed: int

class EnhancedPDFListener(ClassificationIntegratedListener):
    """
    Enhanced PDF listener with full OCR, preprocessing, and classification pipeline.
    
    Extends the classification listener to include Azure Document Intelligence OCR
    with support for handwritten text, table extraction, and custom field recognition.
    """
    
    def __init__(
        self,
        connection_string: str,
        container_name: str,
        azure_doc_intel_endpoint: str,
        azure_doc_intel_key: str,
        output_directory: str = "./processed_documents",
        enable_preprocessing: bool = True,
        enable_classification: bool = True,
        enable_ocr: bool = True,
        default_document_type: DocumentType = DocumentType.LAYOUT,
        custom_model_mappings: Optional[Dict[str, str]] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize enhanced PDF listener with full processing pipeline.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Container to monitor
            azure_doc_intel_endpoint: Azure Document Intelligence endpoint
            azure_doc_intel_key: Azure Document Intelligence API key
            output_directory: Directory for processed outputs
            enable_preprocessing: Enable image preprocessing
            enable_classification: Enable document classification
            enable_ocr: Enable OCR processing
            default_document_type: Default Document Intelligence model
            custom_model_mappings: Map document classes to custom OCR models
            confidence_thresholds: Confidence thresholds for different operations
        """
        # Initialize parent class
        super().__init__(
            connection_string=connection_string,
            container_name=container_name,
            output_directory=output_directory,
            enable_preprocessing=enable_preprocessing,
            enable_classification=enable_classification
        )
        
        # OCR Configuration
        self.enable_ocr = enable_ocr
        self.azure_doc_intel_endpoint = azure_doc_intel_endpoint
        self.azure_doc_intel_key = azure_doc_intel_key
        self.default_document_type = default_document_type
        
        # Custom model mappings (document class -> custom model ID)
        self.custom_model_mappings = custom_model_mappings or {}
        
        # Confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'ocr': 0.7,
            'classification': 0.6,
            'field_extraction': 0.8
        }
        
        # Initialize OCR engine if enabled
        self.document_intelligence = None
        if self.enable_ocr:
            try:
                self.document_intelligence = DocumentIntelligenceIntegration(
                    azure_endpoint=azure_doc_intel_endpoint,
                    azure_api_key=azure_doc_intel_key,
                    preprocessor=self.preprocessor,
                    default_model=default_document_type,
                    enable_preprocessing=enable_preprocessing
                )
                logger.info("Azure Document Intelligence OCR initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Document Intelligence OCR: {e}")
                self.enable_ocr = False
        
        # Processing statistics
        self.ocr_stats = {
            "documents_processed": 0,
            "handwritten_text_detected": 0,
            "tables_extracted": 0,
            "custom_fields_extracted": 0,
            "total_ocr_time": 0.0,
            "average_confidence": 0.0
        }
        
        logger.info("Enhanced PDF listener initialized with full processing pipeline")
    
    def process_pdf_complete(self, blob_name: str, local_file_path: str) -> CompleteDocumentAnalysis:
        """
        Complete PDF processing pipeline with OCR, classification, and analysis.
        
        Args:
            blob_name: Name of the blob in Azure Storage
            local_file_path: Local path to the downloaded PDF
            
        Returns:
            Complete document analysis result
        """
        start_time = time.time()
        processing_pipeline = []
        
        logger.info(f"Starting complete processing of: {blob_name}")
        
        # Step 1: Preprocessing (if enabled)
        preprocessed_images = None
        if self.enable_preprocessing and self.preprocessor:
            try:
                preprocessed_images = self.preprocessor.pdf_to_images(local_file_path)
                processing_pipeline.append("preprocessing")
                logger.info(f"Preprocessing completed: {len(preprocessed_images)} pages")
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                preprocessed_images = None
        
        # Step 2: Classification (if enabled)
        document_class = "unknown"
        classification_confidence = 0.0
        classification_features = {}
        
        if self.enable_classification and self.classifier:
            try:
                # Use preprocessed images for classification
                if preprocessed_images:
                    import asyncio
                    # Read PDF bytes for Azure analysis
                    with open(local_file_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    # Use Azure-enabled classification
                    class_result = asyncio.run(self.classifier.classify_document(pdf_bytes, preprocessed_images[0]))
                    document_class = class_result.document_type
                    classification_confidence = class_result.confidence
                    classification_features = class_result.features
                else:
                    # Fallback to basic classification without image
                    logger.warning("No preprocessed images available for classification")
                    document_class = "unknown"
                    classification_confidence = 0.0
                    classification_features = {}
                processing_pipeline.append("classification")
                
                logger.info(f"Classification: {document_class} (confidence: {classification_confidence:.2f})")
            except Exception as e:
                logger.error(f"Classification failed: {e}")
        
        # Step 3: OCR Processing (if enabled)
        ocr_result = None
        if self.enable_ocr and self.document_intelligence:
            try:
                # Determine document type for OCR
                document_type = self._determine_ocr_model(document_class)
                
                # Use preprocessed data or original file
                source_data = preprocessed_images[0] if preprocessed_images else local_file_path
                
                # Perform OCR analysis
                ocr_result, _ = self.document_intelligence.process_document_with_ocr(
                    document_data=source_data,
                    document_type=document_type,
                    apply_preprocessing=False  # Already preprocessed
                )
                
                processing_pipeline.append("ocr")
                self._update_ocr_stats(ocr_result)
                
                logger.info(f"OCR completed: {len(ocr_result.text_blocks)} text blocks, "
                          f"{len(ocr_result.tables)} tables, {len(ocr_result.fields)} fields")
            
            except Exception as e:
                logger.error(f"OCR processing failed: {e}")
                # Create empty OCR result
                ocr_result = self._create_empty_ocr_result()
        else:
            # Create empty OCR result if OCR is disabled
            ocr_result = self._create_empty_ocr_result()
        
        # Step 4: Post-processing and validation
        custom_fields = self._extract_custom_fields(ocr_result, document_class)
        
        # Step 5: Save results
        output_data = self._prepare_output_data(
            blob_name, ocr_result, document_class, classification_confidence,
            classification_features, custom_fields
        )
        
        # Save to JSON
        output_file = self._save_analysis_results(blob_name, output_data)
        processing_pipeline.append("output_generation")
        
        # Create complete analysis result
        file_stats = Path(local_file_path).stat()
        total_time = time.time() - start_time
        
        complete_analysis = CompleteDocumentAnalysis(
            ocr_result=ocr_result,
            document_class=document_class,
            classification_confidence=classification_confidence,
            classification_features=classification_features,
            processing_pipeline=processing_pipeline,
            total_processing_time=total_time,
            preprocessing_applied=preprocessed_images is not None,
            custom_fields=custom_fields,
            original_file_path=local_file_path,
            file_size_bytes=file_stats.st_size,
            pages_processed=ocr_result.pages
        )
        
        logger.info(f"Complete processing finished in {total_time:.2f}s: {blob_name}")
        return complete_analysis
    
    def _determine_ocr_model(self, document_class: str) -> DocumentType:
        """
        Determine the appropriate OCR model based on document classification.
        
        Args:
            document_class: Classified document type
            
        Returns:
            Appropriate DocumentType for OCR
        """
        # Check for custom model mappings first
        if document_class in self.custom_model_mappings:
            return DocumentType.CUSTOM
        
        # Map common document classes to prebuilt models
        class_to_model = {
            "invoice": DocumentType.INVOICE,
            "receipt": DocumentType.RECEIPT,
            "business_card": DocumentType.BUSINESS_CARD,
            "id_document": DocumentType.ID_DOCUMENT,
            "tax_form": DocumentType.TAX_US_W2,
            "form": DocumentType.LAYOUT,
            "report": DocumentType.LAYOUT,
            "letter": DocumentType.GENERAL
        }
        
        return class_to_model.get(document_class.lower(), self.default_document_type)
    
    def _extract_custom_fields(
        self,
        ocr_result: DocumentAnalysisResult,
        document_class: str
    ) -> Dict[str, Any]:
        """
        Extract custom fields based on document class and OCR results.
        
        Args:
            ocr_result: OCR analysis result
            document_class: Document classification
            
        Returns:
            Dictionary of custom extracted fields
        """
        custom_fields = {}
        
        # Add confidence scores
        custom_fields["confidence_scores"] = ocr_result.confidence_scores
        
        # Add document metadata
        custom_fields["document_metadata"] = {
            "pages": ocr_result.pages,
            "language": ocr_result.language,
            "model_used": ocr_result.model_used,
            "processing_time": ocr_result.processing_time
        }
        
        # Extract high-confidence fields
        high_confidence_fields = []
        threshold = self.confidence_thresholds.get('field_extraction', 0.8)
        
        for field in ocr_result.fields:
            if field.confidence >= threshold:
                high_confidence_fields.append({
                    "name": field.field_name,
                    "value": field.value,
                    "confidence": field.confidence,
                    "type": field.field_type
                })
        
        custom_fields["high_confidence_fields"] = high_confidence_fields
        
        # Extract table summaries
        if ocr_result.tables:
            table_summaries = []
            for i, table in enumerate(ocr_result.tables):
                summary = {
                    "table_index": i,
                    "rows": len(table.rows),
                    "columns": len(table.rows[0]) if table.rows else 0,
                    "has_headers": table.headers is not None,
                    "confidence": table.confidence
                }
                table_summaries.append(summary)
            
            custom_fields["table_summaries"] = table_summaries
        
        # Detect handwritten content
        handwritten_blocks = [
            block for block in ocr_result.text_blocks 
            if block.is_handwritten
        ]
        
        if handwritten_blocks:
            custom_fields["handwritten_content"] = {
                "blocks_detected": len(handwritten_blocks),
                "average_confidence": sum(b.confidence for b in handwritten_blocks) / len(handwritten_blocks),
                "sample_text": handwritten_blocks[0].content[:100] if handwritten_blocks else ""
            }
        
        return custom_fields
    
    def _prepare_output_data(
        self,
        blob_name: str,
        ocr_result: DocumentAnalysisResult,
        document_class: str,
        classification_confidence: float,
        classification_features: Dict[str, Any],
        custom_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare comprehensive output data structure."""
        
        return {
            "document_info": {
                "name": blob_name,
                "class": document_class,
                "classification_confidence": classification_confidence,
                "processing_timestamp": time.time(),
                "pages": ocr_result.pages,
                "language": ocr_result.language
            },
            
            "ocr_results": {
                "full_text": ocr_result.full_text,
                "text_blocks": [
                    {
                        "content": block.content,
                        "bounding_box": {
                            "x": block.bounding_box.x,
                            "y": block.bounding_box.y,
                            "width": block.bounding_box.width,
                            "height": block.bounding_box.height
                        },
                        "confidence": block.confidence,
                        "is_handwritten": block.is_handwritten,
                        "language": block.language
                    }
                    for block in ocr_result.text_blocks
                ],
                
                "tables": [
                    {
                        "table_index": i,
                        "rows": table.rows,
                        "headers": table.headers,
                        "confidence": table.confidence,
                        "bounding_box": {
                            "x": table.bounding_box.x,
                            "y": table.bounding_box.y,
                            "width": table.bounding_box.width,
                            "height": table.bounding_box.height
                        }
                    }
                    for i, table in enumerate(ocr_result.tables)
                ],
                
                "structured_fields": [
                    {
                        "field_name": field.field_name,
                        "value": field.value,
                        "type": field.field_type,
                        "confidence": field.confidence,
                        "bounding_box": {
                            "x": field.bounding_box.x,
                            "y": field.bounding_box.y,
                            "width": field.bounding_box.width,
                            "height": field.bounding_box.height
                        } if field.bounding_box else None
                    }
                    for field in ocr_result.fields
                ]
            },
            
            "classification_results": {
                "predicted_class": document_class,  # Legacy format
                "document_type": document_class,     # New format
                "confidence": classification_confidence,
                "features": classification_features
            },
            
            "custom_analysis": custom_fields,
            
            "processing_metadata": {
                "model_used": ocr_result.model_used,
                "processing_time": ocr_result.processing_time,
                "confidence_scores": ocr_result.confidence_scores
            }
        }
    
    def _save_analysis_results(self, blob_name: str, output_data: Dict[str, Any]) -> str:
        """Save complete analysis results to JSON file."""
        
        # Create output filename
        base_name = Path(blob_name).stem
        output_filename = f"{base_name}_complete_analysis.json"
        output_path = Path(self.output_directory) / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete analysis saved to: {output_path}")
        return str(output_path)
    
    def _create_empty_ocr_result(self) -> DocumentAnalysisResult:
        """Create empty OCR result for error cases."""
        from .azure_document_intelligence import DocumentAnalysisResult
        
        return DocumentAnalysisResult(
            text_blocks=[],
            tables=[],
            fields=[],
            full_text="",
            pages=0,
            language=None,
            processing_time=0.0,
            model_used="none",
            confidence_scores={}
        )
    
    def _update_ocr_stats(self, ocr_result: DocumentAnalysisResult):
        """Update OCR processing statistics."""
        self.ocr_stats["documents_processed"] += 1
        self.ocr_stats["total_ocr_time"] += ocr_result.processing_time
        
        # Count handwritten text blocks
        handwritten_count = sum(1 for block in ocr_result.text_blocks if block.is_handwritten)
        self.ocr_stats["handwritten_text_detected"] += handwritten_count
        
        # Count tables and fields
        self.ocr_stats["tables_extracted"] += len(ocr_result.tables)
        self.ocr_stats["custom_fields_extracted"] += len(ocr_result.fields)
        
        # Update average confidence
        if ocr_result.confidence_scores.get('text'):
            total_docs = self.ocr_stats["documents_processed"]
            current_avg = self.ocr_stats["average_confidence"]
            new_confidence = ocr_result.confidence_scores['text']
            self.ocr_stats["average_confidence"] = (
                (current_avg * (total_docs - 1) + new_confidence) / total_docs
            )
    
    def get_complete_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            "preprocessing": self.get_preprocessing_stats() if self.preprocessor else {},
            "classification": self.get_classification_stats() if self.classifier else {},
            "ocr": self.ocr_stats,
            "overall": {
                "total_documents": self.processing_stats["files_processed"],
                "total_processing_time": sum([
                    self.processing_stats.get("total_processing_time", 0),
                    self.ocr_stats.get("total_ocr_time", 0)
                ])
            }
        }
        
        return stats
    
    async def process_blob_async(self, blob_name: str):
        """
        Asynchronous blob processing with complete pipeline.
        
        Overrides parent method to include OCR processing.
        """
        logger.info(f"Processing blob with complete pipeline: {blob_name}")
        
        try:
            # Download file
            local_file_path = await self._download_blob_async(blob_name)
            
            # Run complete processing pipeline
            complete_analysis = self.process_pdf_complete(blob_name, local_file_path)
            
            # Log results
            logger.info(
                f"Complete processing results for {blob_name}:\n"
                f"  - Document Class: {complete_analysis.document_class}\n"
                f"  - Classification Confidence: {complete_analysis.classification_confidence:.2f}\n"
                f"  - Text Blocks: {len(complete_analysis.ocr_result.text_blocks)}\n"
                f"  - Tables: {len(complete_analysis.ocr_result.tables)}\n"
                f"  - Structured Fields: {len(complete_analysis.ocr_result.fields)}\n"
                f"  - Processing Time: {complete_analysis.total_processing_time:.2f}s\n"
                f"  - Pipeline: {' → '.join(complete_analysis.processing_pipeline)}"
            )
            
            # Update statistics
            self.processing_stats["files_processed"] += 1
            self.processing_stats["total_processing_time"] += complete_analysis.total_processing_time
            
            # Clean up local file
            Path(local_file_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Complete processing failed for {blob_name}: {e}")
            self.processing_stats["failed_files"] += 1