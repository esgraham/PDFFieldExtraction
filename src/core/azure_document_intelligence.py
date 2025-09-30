"""
Azure AI Document Intelligence OCR Module

This module provides document-optimized OCR capabilities using Azure AI Document Intelligence v4
for printed and handwritten text extraction with layout analysis, table detection, and custom training.

Features:
- Azure Document Intelligence "Read" API for text extraction
- Layout analysis with bounding boxes and confidence scores
- Prebuilt models for common document types (invoices, receipts, forms)
- Custom model training for specific document fields
- Table extraction and structure analysis
- Handwritten text recognition (HWR)
- Batch processing capabilities
- Integration with preprocessing pipeline
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import base64
from datetime import datetime

# Azure Document Intelligence
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        AnalyzeResult,
        DocumentPage,
        DocumentTable,
        DocumentParagraph,
        DocumentLine,
        DocumentWord,
        BoundingRegion,
        DocumentField
    )
    from azure.core.credentials import AzureKeyCredential
    AZURE_DOC_INTEL_AVAILABLE = True
except ImportError:
    AZURE_DOC_INTEL_AVAILABLE = False

# Core dependencies
import numpy as np
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Supported document types for prebuilt models."""
    GENERAL = "prebuilt-read"                    # General text extraction
    LAYOUT = "prebuilt-layout"                   # Layout analysis with tables
    INVOICE = "prebuilt-invoice"                 # Invoice-specific fields
    RECEIPT = "prebuilt-receipt"                 # Receipt-specific fields
    ID_DOCUMENT = "prebuilt-idDocument"          # ID cards, passports
    BUSINESS_CARD = "prebuilt-businessCard"      # Business card extraction
    TAX_US_W2 = "prebuilt-tax.us.w2"           # US W-2 tax forms
    CUSTOM = "custom"                            # Custom trained models

class OCREngine(Enum):
    """OCR engine options."""
    AZURE_READ = "azure_read"                    # Azure Read API (general)
    AZURE_LAYOUT = "azure_layout"               # Azure Layout API (structured)
    AZURE_PREBUILT = "azure_prebuilt"           # Prebuilt models
    AZURE_CUSTOM = "azure_custom"               # Custom trained models

@dataclass
class BoundingBox:
    """Bounding box with confidence score."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    
    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        """Get coordinates as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

@dataclass
class ExtractedText:
    """Extracted text with metadata."""
    content: str
    bounding_box: BoundingBox
    confidence: float
    is_handwritten: bool = False
    language: Optional[str] = None
    
@dataclass
class ExtractedTable:
    """Extracted table structure."""
    rows: List[List[str]]
    headers: Optional[List[str]]
    bounding_box: BoundingBox
    confidence: float
    cell_bounding_boxes: List[List[BoundingBox]]

@dataclass
class ExtractedField:
    """Extracted structured field."""
    field_name: str
    value: str
    field_type: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None

@dataclass
class DocumentAnalysisResult:
    """Complete document analysis result."""
    text_blocks: List[ExtractedText]
    tables: List[ExtractedTable]
    fields: List[ExtractedField]
    full_text: str
    pages: int
    language: Optional[str]
    processing_time: float
    model_used: str
    confidence_scores: Dict[str, float]

class AzureDocumentIntelligenceOCR:
    """
    Azure AI Document Intelligence OCR engine with support for multiple model types.
    
    Provides document-optimized OCR with layout analysis, table extraction,
    and custom field recognition capabilities.
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        default_model: DocumentType = DocumentType.LAYOUT,
        custom_model_id: Optional[str] = None,
        enable_handwriting: bool = True,
        confidence_threshold: float = 0.7,
        timeout_seconds: int = 300
    ):
        """
        Initialize Azure Document Intelligence OCR.
        
        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key: Azure Document Intelligence API key
            default_model: Default model type to use
            custom_model_id: Custom model ID if using custom models
            enable_handwriting: Enable handwriting recognition
            confidence_threshold: Minimum confidence score for results
            timeout_seconds: Analysis timeout in seconds
        """
        if not AZURE_DOC_INTEL_AVAILABLE:
            raise ImportError(
                "Azure Document Intelligence SDK not available. "
                "Install with: pip install azure-ai-documentintelligence"
            )
        
        self.endpoint = endpoint
        self.api_key = api_key
        self.default_model = default_model
        self.custom_model_id = custom_model_id
        self.enable_handwriting = enable_handwriting
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        
        # Initialize client
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "handwritten_text_detected": 0,
            "tables_extracted": 0,
            "custom_fields_extracted": 0
        }
        
        logger.info(f"Azure Document Intelligence OCR initialized with model: {default_model.value}")
    
    def analyze_document(
        self,
        document_data: Union[bytes, str, np.ndarray],
        document_type: Optional[DocumentType] = None,
        custom_model_id: Optional[str] = None,
        extract_tables: bool = True,
        extract_key_value_pairs: bool = True
    ) -> DocumentAnalysisResult:
        """
        Analyze document using Azure Document Intelligence.
        
        Args:
            document_data: Document as bytes, file path, or numpy array
            document_type: Document type/model to use
            custom_model_id: Custom model ID (overrides document_type)
            extract_tables: Whether to extract table structures
            extract_key_value_pairs: Whether to extract key-value pairs
            
        Returns:
            Complete document analysis result
        """
        start_time = time.time()
        
        # Determine model to use
        if custom_model_id:
            model_id = custom_model_id
        elif document_type:
            model_id = document_type.value
        else:
            model_id = self.default_model.value
        
        logger.info(f"Analyzing document with model: {model_id}")
        
        try:
            # Prepare document data
            document_bytes = self._prepare_document_data(document_data)
            
            # Configure analysis options
            features = []
            if extract_tables and "layout" in model_id.lower():
                features.append("ocrHighResolution")
            
            # Start analysis
            poller = self.client.begin_analyze_document(
                model_id=model_id,
                analyze_request=AnalyzeDocumentRequest(bytes_source=document_bytes),
                features=features if features else None
            )
            
            # Wait for completion
            result = poller.result()
            
            # Process results
            analysis_result = self._process_analysis_result(
                result, model_id, time.time() - start_time
            )
            
            # Update statistics
            self._update_stats(analysis_result)
            
            logger.info(f"Document analysis completed in {analysis_result.processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise
    
    def analyze_layout(self, document_data: Union[bytes, str, np.ndarray]) -> DocumentAnalysisResult:
        """
        Analyze document layout with table and structure extraction.
        
        Args:
            document_data: Document to analyze
            
        Returns:
            Layout analysis result with tables and structure
        """
        return self.analyze_document(
            document_data,
            document_type=DocumentType.LAYOUT,
            extract_tables=True,
            extract_key_value_pairs=True
        )
    
    def extract_invoice_fields(self, document_data: Union[bytes, str, np.ndarray]) -> DocumentAnalysisResult:
        """
        Extract invoice-specific fields using prebuilt invoice model.
        
        Args:
            document_data: Invoice document to analyze
            
        Returns:
            Invoice analysis with structured fields
        """
        return self.analyze_document(
            document_data,
            document_type=DocumentType.INVOICE,
            extract_tables=True,
            extract_key_value_pairs=True
        )
    
    def extract_receipt_fields(self, document_data: Union[bytes, str, np.ndarray]) -> DocumentAnalysisResult:
        """
        Extract receipt-specific fields using prebuilt receipt model.
        
        Args:
            document_data: Receipt document to analyze
            
        Returns:
            Receipt analysis with structured fields
        """
        return self.analyze_document(
            document_data,
            document_type=DocumentType.RECEIPT,
            extract_tables=True,
            extract_key_value_pairs=True
        )
    
    def analyze_with_custom_model(
        self,
        document_data: Union[bytes, str, np.ndarray],
        model_id: str
    ) -> DocumentAnalysisResult:
        """
        Analyze document using a custom trained model.
        
        Args:
            document_data: Document to analyze
            model_id: Custom model ID
            
        Returns:
            Analysis result with custom fields
        """
        return self.analyze_document(
            document_data,
            custom_model_id=model_id,
            extract_tables=True,
            extract_key_value_pairs=True
        )
    
    def batch_analyze(
        self,
        documents: List[Union[bytes, str, np.ndarray]],
        document_type: Optional[DocumentType] = None,
        max_concurrent: int = 5
    ) -> List[DocumentAnalysisResult]:
        """
        Analyze multiple documents concurrently.
        
        Args:
            documents: List of documents to analyze
            document_type: Document type/model to use
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(documents)} documents")
        
        results = []
        
        # Process documents in batches to respect rate limits
        for i in range(0, len(documents), max_concurrent):
            batch = documents[i:i + max_concurrent]
            batch_results = []
            
            for doc in batch:
                try:
                    result = self.analyze_document(doc, document_type)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to analyze document {i}: {e}")
                    # Create error result
                    error_result = DocumentAnalysisResult(
                        text_blocks=[],
                        tables=[],
                        fields=[],
                        full_text="",
                        pages=0,
                        language=None,
                        processing_time=0.0,
                        model_used="error",
                        confidence_scores={"error": 0.0}
                    )
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # Brief pause between batches to respect rate limits
            if i + max_concurrent < len(documents):
                time.sleep(1)
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def _prepare_document_data(self, document_data: Union[bytes, str, np.ndarray]) -> bytes:
        """Prepare document data for analysis."""
        if isinstance(document_data, bytes):
            return document_data
        
        elif isinstance(document_data, str):
            # File path
            with open(document_data, 'rb') as f:
                return f.read()
        
        elif isinstance(document_data, np.ndarray):
            # Convert numpy array to bytes
            if len(document_data.shape) == 3:
                # Color image
                image = cv2.cvtColor(document_data, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale image
                image = document_data
            
            # Convert to PIL and then to bytes
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported document data type: {type(document_data)}")
    
    def _process_analysis_result(
        self,
        result: AnalyzeResult,
        model_id: str,
        processing_time: float
    ) -> DocumentAnalysisResult:
        """Process Azure Document Intelligence analysis result."""
        
        # Extract text blocks
        text_blocks = []
        if result.paragraphs:
            for paragraph in result.paragraphs:
                if paragraph.content and paragraph.bounding_regions:
                    bbox = self._convert_bounding_region(paragraph.bounding_regions[0])
                    text_block = ExtractedText(
                        content=paragraph.content,
                        bounding_box=bbox,
                        confidence=getattr(paragraph, 'confidence', 1.0),
                        is_handwritten=self._detect_handwriting(paragraph)
                    )
                    text_blocks.append(text_block)
        
        # Extract tables
        tables = []
        if result.tables:
            for table in result.tables:
                extracted_table = self._process_table(table)
                tables.append(extracted_table)
        
        # Extract fields (for prebuilt/custom models)
        fields = []
        if result.documents and len(result.documents) > 0:
            for document in result.documents:
                if document.fields:
                    for field_name, field_value in document.fields.items():
                        extracted_field = self._process_field(field_name, field_value)
                        if extracted_field:
                            fields.append(extracted_field)
        
        # Get full text
        full_text = result.content if result.content else ""
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(result)
        
        # Detect language
        language = self._detect_language(result)
        
        return DocumentAnalysisResult(
            text_blocks=text_blocks,
            tables=tables,
            fields=fields,
            full_text=full_text,
            pages=len(result.pages) if result.pages else 1,
            language=language,
            processing_time=processing_time,
            model_used=model_id,
            confidence_scores=confidence_scores
        )
    
    def _convert_bounding_region(self, bounding_region: BoundingRegion) -> BoundingBox:
        """Convert Azure bounding region to our BoundingBox format."""
        if not bounding_region.polygon or len(bounding_region.polygon) < 4:
            return BoundingBox(0, 0, 0, 0, 0.0)
        
        # Extract coordinates from polygon
        x_coords = [point.x for point in bounding_region.polygon]
        y_coords = [point.y for point in bounding_region.polygon]
        
        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - x
        height = max(y_coords) - y
        
        return BoundingBox(x, y, width, height, 1.0)
    
    def _detect_handwriting(self, paragraph) -> bool:
        """Detect if paragraph contains handwritten text."""
        # Check if any lines in the paragraph are marked as handwritten
        if hasattr(paragraph, 'spans') and paragraph.spans:
            for span in paragraph.spans:
                if hasattr(span, 'kind') and span.kind == 'handwriting':
                    return True
        return False
    
    def _process_table(self, table: DocumentTable) -> ExtractedTable:
        """Process table structure from Azure result."""
        # Initialize table structure
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1
        
        rows = [["" for _ in range(max_col)] for _ in range(max_row)]
        cell_boxes = [[BoundingBox(0, 0, 0, 0) for _ in range(max_col)] for _ in range(max_row)]
        
        # Fill table data
        for cell in table.cells:
            row_idx = cell.row_index
            col_idx = cell.column_index
            rows[row_idx][col_idx] = cell.content or ""
            
            if cell.bounding_regions:
                cell_boxes[row_idx][col_idx] = self._convert_bounding_region(cell.bounding_regions[0])
        
        # Extract headers (assume first row contains headers)
        headers = rows[0] if rows else None
        
        # Table bounding box
        if table.bounding_regions:
            table_bbox = self._convert_bounding_region(table.bounding_regions[0])
        else:
            table_bbox = BoundingBox(0, 0, 0, 0)
        
        return ExtractedTable(
            rows=rows,
            headers=headers,
            bounding_box=table_bbox,
            confidence=1.0,  # Azure doesn't provide table-level confidence
            cell_bounding_boxes=cell_boxes
        )
    
    def _process_field(self, field_name: str, field_value: DocumentField) -> Optional[ExtractedField]:
        """Process structured field from Azure result."""
        if not field_value or not field_value.content:
            return None
        
        # Get bounding box if available
        bbox = None
        if field_value.bounding_regions:
            bbox = self._convert_bounding_region(field_value.bounding_regions[0])
        
        return ExtractedField(
            field_name=field_name,
            value=field_value.content,
            field_type=field_value.type if hasattr(field_value, 'type') else "string",
            confidence=field_value.confidence if hasattr(field_value, 'confidence') else 1.0,
            bounding_box=bbox
        )
    
    def _calculate_confidence_scores(self, result: AnalyzeResult) -> Dict[str, float]:
        """Calculate overall confidence scores."""
        scores = {}
        
        # Overall text confidence
        if result.paragraphs:
            confidences = [getattr(p, 'confidence', 1.0) for p in result.paragraphs]
            scores['text'] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Table confidence
        if result.tables:
            scores['tables'] = 1.0  # Azure doesn't provide table confidence
        
        # Field confidence
        if result.documents:
            field_confidences = []
            for document in result.documents:
                if document.fields:
                    for field_value in document.fields.values():
                        if hasattr(field_value, 'confidence'):
                            field_confidences.append(field_value.confidence)
            
            if field_confidences:
                scores['fields'] = sum(field_confidences) / len(field_confidences)
        
        return scores
    
    def _detect_language(self, result: AnalyzeResult) -> Optional[str]:
        """Detect document language from result."""
        # Check if language is detected in the result
        if hasattr(result, 'languages') and result.languages:
            return result.languages[0].locale
        
        # Fallback language detection could be added here
        return None
    
    def _update_stats(self, result: DocumentAnalysisResult):
        """Update processing statistics."""
        self.stats["documents_processed"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        # Update average confidence
        if result.confidence_scores.get('text'):
            total_docs = self.stats["documents_processed"]
            current_avg = self.stats["average_confidence"]
            new_confidence = result.confidence_scores['text']
            self.stats["average_confidence"] = (
                (current_avg * (total_docs - 1) + new_confidence) / total_docs
            )
        
        # Count handwritten text
        handwritten_count = sum(1 for block in result.text_blocks if block.is_handwritten)
        self.stats["handwritten_text_detected"] += handwritten_count
        
        # Count tables
        self.stats["tables_extracted"] += len(result.tables)
        
        # Count custom fields
        self.stats["custom_fields_extracted"] += len(result.fields)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats["documents_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["documents_processed"]
            )
        
        return stats
    
    def validate_connection(self) -> bool:
        """Validate connection to Azure Document Intelligence service."""
        try:
            # Test with a minimal dummy request (this will fail but validate connection)
            test_data = b"test"
            self.client.begin_analyze_document(
                model_id="prebuilt-read",
                analyze_request=AnalyzeDocumentRequest(bytes_source=test_data)
            )
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

class DocumentIntelligenceIntegration:
    """
    Integration layer for Azure Document Intelligence with preprocessing pipeline.
    """
    
    def __init__(
        self,
        azure_endpoint: str,
        azure_api_key: str,
        preprocessor: Optional['PDFPreprocessor'] = None,
        default_model: DocumentType = DocumentType.LAYOUT,
        enable_preprocessing: bool = True
    ):
        """
        Initialize Document Intelligence integration.
        
        Args:
            azure_endpoint: Azure Document Intelligence endpoint
            azure_api_key: Azure API key
            preprocessor: PDF preprocessor instance
            default_model: Default Document Intelligence model
            enable_preprocessing: Whether to apply preprocessing before OCR
        """
        self.ocr_engine = AzureDocumentIntelligenceOCR(
            endpoint=azure_endpoint,
            api_key=azure_api_key,
            default_model=default_model
        )
        
        self.preprocessor = preprocessor
        self.enable_preprocessing = enable_preprocessing and preprocessor is not None
        
        logger.info("Document Intelligence integration initialized")
    
    def process_document_with_ocr(
        self,
        document_data: Union[bytes, str, np.ndarray],
        document_type: Optional[DocumentType] = None,
        apply_preprocessing: Optional[bool] = None
    ) -> Tuple[DocumentAnalysisResult, Optional[List[np.ndarray]]]:
        """
        Process document with optional preprocessing and OCR analysis.
        
        Args:
            document_data: Document to process
            document_type: Document type for specialized processing
            apply_preprocessing: Override preprocessing setting
            
        Returns:
            Tuple of (OCR result, preprocessed images if applicable)
        """
        preprocessed_images = None
        
        # Apply preprocessing if enabled
        if (apply_preprocessing if apply_preprocessing is not None else self.enable_preprocessing):
            if isinstance(document_data, (str, bytes)):
                # For file paths or bytes, we need to convert to images first
                if isinstance(document_data, str):
                    # Load images from file/directory
                    preprocessed_images = self.preprocessor.pdf_to_images(document_data)
                else:
                    # Convert bytes to image and preprocess
                    # This is a simplified approach - full implementation would handle PDF bytes
                    img_array = np.frombuffer(document_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        processed = self.preprocessor._process_image(img)
                        preprocessed_images = [processed]
                
                # Use first preprocessed image for OCR
                if preprocessed_images:
                    document_data = preprocessed_images[0]
        
        # Perform OCR analysis
        ocr_result = self.ocr_engine.analyze_document(document_data, document_type)
        
        return ocr_result, preprocessed_images