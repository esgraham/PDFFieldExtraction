"""
PDF Processing Pipeline Manager

Orchestrates the complete PDF processing workflow with proper error handling
and stage tracking.
"""

import asyncio
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Core processing modules
from .pdf_preprocessor import PDFPreprocessor
from .document_classifier import DocumentClassifier
from .azure_document_intelligence import AzureDocumentIntelligenceOCR
from .field_extraction import DocumentTemplate, FieldExtractor
from .validation_engine import ComprehensiveValidator

logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Main PDF processing pipeline with all processing stages."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processing pipeline with configuration."""
        self.config = config
        self.results_history = []
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components."""
        try:
            # PDF Preprocessor
            self.preprocessor = PDFPreprocessor()
            
            # Initialize Document Classifier and OCR Engine
            azure_config = self.config.get('azure_document_intelligence', {})
            
            if azure_config.get('enabled', False):
                # Document Classifier with Azure support
                self.classifier = DocumentClassifier(
                    azure_endpoint=azure_config['endpoint'],
                    azure_api_key=azure_config['api_key'],
                    use_azure_prebuilt=True,
                    azure_confidence_threshold=azure_config.get('confidence_threshold', 0.7)
                )
                
                # Optionally discover and configure models dynamically
                if azure_config.get('auto_discover_models', False):
                    logger.info("🔍 Auto-discovering Azure models...")
                    try:
                        model_count = self.classifier.configure_models_from_discovery(
                            include_prebuilt=azure_config.get('include_prebuilt', True),
                            include_custom=azure_config.get('include_custom', False)
                        )
                        logger.info(f"✅ Auto-configured {model_count} Azure models")
                        
                        # Log discovered models
                        if logger.isEnabledFor(logging.INFO):
                            doc_types = self.classifier.get_registered_document_types()
                            logger.info(f"📋 Available document types: {', '.join(doc_types[:10])}{'...' if len(doc_types) > 10 else ''}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Auto-discovery failed: {e}")
                        logger.info("Falling back to default model configuration")
                
                # OCR Engine
                self.ocr_engine = AzureDocumentIntelligenceOCR(
                    endpoint=azure_config['endpoint'],
                    api_key=azure_config['api_key']
                )
            else:
                self.classifier = DocumentClassifier()
                self.ocr_engine = None
                logger.warning("Azure Document Intelligence not configured")
            
            # Field Extractor
            self.field_extractor = FieldExtractor()
            
            # Validator
            self.validator = ComprehensiveValidator()
            
            logger.info("✅ All processing components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def process_pdf(self, blob_name: str, blob_client) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            blob_name: Name of the blob/file
            blob_client: Azure blob client for file access
            
        Returns:
            Complete processing result with all stages
        """
        processing_start = time.time()
        
        result = {
            'blob_name': blob_name,
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'extracted_data': {},
            'validation_results': [],
            'summary': {},
            'processing_time': 0,
            'needs_human_review': False
        }
        
        temp_pdf_path = None
        
        try:
            # Stage 1: Download PDF
            logger.info(f"📥 Stage 1: Downloading {blob_name}")
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_pdf_path = temp_file.name
                blob_data = blob_client.download_blob().readall()
                temp_file.write(blob_data)
            
            result['stages']['download'] = {
                'status': 'completed',
                'file_size': len(blob_data),
                'temp_path': temp_pdf_path
            }
            
            # Stage 2: Preprocessing
            logger.info(f"🔧 Stage 2: Preprocessing PDF")
            try:
                preprocessed_path = await self._preprocess_pdf(temp_pdf_path)
                result['stages']['preprocessing'] = {
                    'status': 'completed',
                    'preprocessed_path': preprocessed_path
                }
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}")
                preprocessed_path = temp_pdf_path
                result['stages']['preprocessing'] = {
                    'status': 'failed',
                    'error': str(e),
                    'fallback_path': temp_pdf_path
                }
            
            # Stage 3: Document Classification
            logger.info(f"🏷️ Stage 3: Classifying document type")
            try:
                doc_type, confidence = await self._classify_document(preprocessed_path)
                result['stages']['classification'] = {
                    'status': 'completed',
                    'document_type': doc_type,
                    'confidence': confidence
                }
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                doc_type, confidence = 'unknown', 0.5
                result['stages']['classification'] = {
                    'status': 'failed',
                    'error': str(e),
                    'fallback_type': doc_type
                }
            
            # Stage 4: OCR and Text Extraction
            logger.info(f"📄 Stage 4: Performing OCR and text extraction")
            try:
                ocr_results = await self._perform_ocr(preprocessed_path)
                result['stages']['ocr'] = {
                    'status': 'completed',
                    'text_length': len(getattr(ocr_results, 'full_text', '')),
                    'confidence': getattr(ocr_results, 'confidence_scores', {}).get('text', 0.0)
                }
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                # Create mock OCR result for fallback
                class MockOCRResult:
                    def __init__(self):
                        self.full_text = f'OCR failed: {str(e)}'
                        self.fields = []
                        self.text_blocks = []
                        self.tables = []
                        self.confidence_scores = {'text': 0.0}
                
                ocr_results = MockOCRResult()
                result['stages']['ocr'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 5: Field Extraction
            logger.info(f"🔍 Stage 5: Extracting structured fields")
            try:
                extracted_fields = await self._extract_fields(ocr_results, doc_type)
                result['stages']['field_extraction'] = {
                    'status': 'completed',
                    'fields_count': len(extracted_fields),
                    'fields': list(extracted_fields.keys())
                }
                result['extracted_data'] = extracted_fields
            except Exception as e:
                logger.error(f"Field extraction failed: {e}")
                extracted_fields = {}
                result['stages']['field_extraction'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 6: Validation and Business Rules
            logger.info(f"✅ Stage 6: Validating data and applying business rules")
            try:
                validation_results = await self._validate_data(extracted_fields, doc_type)
                result['stages']['validation'] = {
                    'status': 'completed',
                    'validation_count': len(validation_results),
                    'passed': len([r for r in validation_results if r.is_valid]),
                    'failed': len([r for r in validation_results if not r.is_valid])
                }
                result['validation_results'] = [
                    {
                        'field': r.field_name,
                        'valid': r.is_valid,
                        'message': r.message,
                        'severity': r.severity.value
                    } for r in validation_results
                ]
            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                validation_results = []
                result['stages']['validation'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 7: Summary and Quality Assessment
            logger.info(f"📊 Stage 7: Generating summary and quality assessment")
            summary = self._generate_summary(result, processing_start)
            result['summary'] = summary
            result['processing_time'] = time.time() - processing_start
            result['needs_human_review'] = self._needs_human_review(result)
            
            # Store in history
            self.results_history.append(result)
            
            logger.info(f"✅ Processing completed for {blob_name} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed for {blob_name}: {e}")
            result['stages']['pipeline'] = {
                'status': 'failed',
                'error': str(e)
            }
            result['processing_time'] = time.time() - processing_start
            return result
            
        finally:
            # Cleanup temporary files
            if temp_pdf_path and Path(temp_pdf_path).exists():
                Path(temp_pdf_path).unlink()
    
    async def _preprocess_pdf(self, pdf_path: str) -> str:
        """Preprocess PDF with deskewing and enhancement."""
        if self.preprocessor:
            # Use the PDFPreprocessor to enhance the PDF
            loop = asyncio.get_event_loop()
            enhanced_path = await loop.run_in_executor(
                None, self.preprocessor.process_pdf, pdf_path
            )
            return enhanced_path
        else:
            # Fallback - return original path
            return pdf_path
    
    async def _classify_document(self, pdf_path: str) -> tuple:
        """
        Classify document type using prioritized approach:
        1. Azure AI Document Intelligence prebuilt models
        2. Custom models and traditional ML classification
        """
        logger.info("🔍 Starting document classification with prioritized approach")
        
        # Read PDF bytes for Azure analysis
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")
            return 'unknown', 0.0
        
        # Step 1: Try Azure AI Document Intelligence prebuilt models first
        if self.classifier and hasattr(self.classifier, 'azure_client') and self.classifier.azure_client:
            logger.info("🎯 Phase 1: Trying Azure AI Document Intelligence prebuilt models")
            try:
                # Use the new Azure-enabled classification method
                azure_result = await self.classifier.classify_with_azure_prebuilt(pdf_bytes)
                
                if azure_result and azure_result.confidence >= self.classifier.azure_confidence_threshold:
                    logger.info(f"✅ Azure prebuilt model success: {azure_result.document_type} "
                              f"(confidence: {azure_result.confidence:.3f}, model: {azure_result.model_used})")
                    
                    # Log detected fields for debugging
                    if azure_result.fields_detected:
                        field_count = len(azure_result.fields_detected)
                        logger.info(f"📋 Azure detected {field_count} fields")
                        
                        # Log top fields with high confidence
                        high_confidence_fields = [
                            field_name for field_name, field_info in azure_result.fields_detected.items()
                            if isinstance(field_info, dict) and field_info.get('confidence', 0) > 0.8
                        ]
                        if high_confidence_fields:
                            logger.info(f"🎯 High-confidence fields: {high_confidence_fields[:5]}")
                    
                    return azure_result.document_type, azure_result.confidence
                else:
                    if azure_result:
                        logger.info(f"🔄 Azure models tried but confidence too low: {azure_result.confidence:.3f} "
                                  f"< {self.classifier.azure_confidence_threshold}")
                    else:
                        logger.info("🔄 Azure prebuilt models did not provide confident classification")
                        
            except Exception as e:
                logger.warning(f"⚠️ Azure prebuilt model analysis failed: {e}")
        else:
            logger.info("⏭️ Azure AI Document Intelligence not available, skipping prebuilt models")
        
        # Step 2: Fall back to custom models and traditional ML classification
        logger.info("🎯 Phase 2: Falling back to custom models and traditional ML classification")
        
        if self.classifier:
            try:
                # Convert PDF first page to image for traditional classification
                def convert_first_page():
                    try:
                        from pdf2image import convert_from_path
                        pages = convert_from_path(pdf_path, first_page=1, last_page=1)
                        if pages:
                            import numpy as np
                            # Convert PIL image to numpy array for classifier
                            return np.array(pages[0])
                        return None
                    except Exception as e:
                        logger.warning(f"PDF to image conversion failed: {e}")
                        return None
                
                loop = asyncio.get_event_loop()
                first_page_image = await loop.run_in_executor(None, convert_first_page)
                
                if first_page_image is not None:
                    # Try traditional ML classification
                    if self.classifier.is_trained:
                        logger.info("🔧 Using trained traditional ML classifier")
                        result = await loop.run_in_executor(
                            None, self.classifier.classify, first_page_image
                        )
                        
                        logger.info(f"✅ Traditional ML classification: {result.document_type} "
                                  f"(confidence: {result.confidence:.3f})")
                        
                        return result.document_type, result.confidence
                    else:
                        logger.info("⚠️ Traditional ML classifier not trained, using rule-based fallback")
                        
                        # Rule-based fallback classification
                        fallback_type, fallback_confidence = await self._fallback_classification(pdf_bytes, first_page_image)
                        return fallback_type, fallback_confidence
                else:
                    logger.warning("❌ Could not convert PDF to image for traditional classification")
                    return 'unknown', 0.3
                    
            except Exception as e:
                logger.warning(f"⚠️ Traditional classification failed: {e}")
                return 'unknown', 0.2
        else:
            logger.warning("❌ No classifier available")
            return 'unknown', 0.1
    
    async def _fallback_classification(self, pdf_bytes: bytes, image_array) -> tuple:
        """
        Rule-based fallback classification when ML models are not available.
        Uses simple heuristics based on document content and structure.
        """
        logger.info("🔍 Using rule-based fallback classification")
        
        try:
            # Basic text extraction for heuristic analysis
            if self.ocr_engine:
                loop = asyncio.get_event_loop()
                ocr_result = await loop.run_in_executor(
                    None, self.ocr_engine.analyze_document, pdf_bytes
                )
                text_content = getattr(ocr_result, 'full_text', '').lower()
            else:
                text_content = ""
            
            # Simple keyword-based classification
            classification_keywords = {
                'invoice': ['invoice', 'bill to', 'invoice number', 'amount due', 'payment terms'],
                'receipt': ['receipt', 'purchased', 'total', 'thank you', 'transaction'],
                'contract': ['agreement', 'contract', 'party', 'terms and conditions', 'signature'],
                'form': ['application', 'form', 'please fill', 'submit', 'information'],
                'statement': ['statement', 'balance', 'account', 'period ending', 'summary'],
                'report': ['report', 'analysis', 'findings', 'conclusion', 'executive summary']
            }
            
            # Count keyword matches
            doc_scores = {}
            for doc_type, keywords in classification_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_content)
                if score > 0:
                    doc_scores[doc_type] = score / len(keywords)  # Normalize by keyword count
            
            if doc_scores:
                # Return the document type with highest score
                best_type = max(doc_scores, key=doc_scores.get)
                confidence = min(doc_scores[best_type] * 0.6, 0.8)  # Cap confidence for fallback
                
                logger.info(f"📝 Fallback classification: {best_type} (confidence: {confidence:.3f})")
                return best_type, confidence
            else:
                logger.info("🤷 No classification keywords found, defaulting to 'unknown'")
                return 'unknown', 0.4
                
        except Exception as e:
            logger.warning(f"⚠️ Fallback classification failed: {e}")
            return 'unknown', 0.2
    
    async def _perform_ocr(self, pdf_path: str) -> Any:
        """Perform OCR and handwriting recognition."""
        if self.ocr_engine:
            # Use Azure Document Intelligence - return DocumentAnalysisResult object
            try:
                # Use analyze_document directly instead of extract_text_async to get full object
                loop = asyncio.get_event_loop()
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                results = await loop.run_in_executor(
                    None, self.ocr_engine.analyze_document, pdf_bytes
                )
                return results
            except Exception as e:
                logger.warning(f"Azure OCR failed: {e}, using fallback")
                # Create a mock DocumentAnalysisResult-like object for fallback
                class MockOCRResult:
                    def __init__(self, error_msg):
                        self.full_text = f'OCR failed: {error_msg}'
                        self.fields = []
                        self.text_blocks = []
                        self.tables = []
                        self.confidence_scores = {'text': 0.0}
                return MockOCRResult(str(e))
        else:
            # Fallback OCR implementation - create mock object
            class MockOCRResult:
                def __init__(self):
                    self.full_text = 'OCR not configured'
                    self.fields = []
                    self.text_blocks = []
                    self.tables = []
                    self.confidence_scores = {'text': 0.0}
            return MockOCRResult()
    
    async def _extract_fields(self, ocr_results: Any, doc_type: str) -> Dict[str, Any]:
        """Extract structured fields from OCR results."""
        try:
            # Convert string doc_type to DocumentTemplate enum
            template_mapping = {
                'invoice': DocumentTemplate.INVOICE,
                'receipt': DocumentTemplate.RECEIPT,
                'purchase_order': DocumentTemplate.PURCHASE_ORDER,
                'tax_form': DocumentTemplate.TAX_FORM,
                'contract': DocumentTemplate.CONTRACT,
                'form_application': DocumentTemplate.FORM_APPLICATION,
                'bank_statement': DocumentTemplate.BANK_STATEMENT,
                'insurance_claim': DocumentTemplate.INSURANCE_CLAIM,
            }
            
            doc_template = template_mapping.get(doc_type.lower(), DocumentTemplate.CUSTOM)
            
            # Use the FieldExtractor to extract structured data
            loop = asyncio.get_event_loop()
            extraction_result = await loop.run_in_executor(
                None, self.field_extractor.extract_fields, ocr_results, doc_template
            )
            
            # Convert ExtractionResult to dictionary format
            if hasattr(extraction_result, 'extracted_fields'):
                # Convert field objects to dictionary format
                fields_dict = {}
                for field in extraction_result.extracted_fields:
                    fields_dict[field.field_name] = {
                        'value': field.normalized_value or field.value,
                        'confidence': field.confidence,
                        'source': getattr(field, 'source', 'extracted')
                    }
                
                fields_dict['document_type'] = doc_type
                fields_dict['template_type'] = extraction_result.template_type.value
                fields_dict['overall_confidence'] = extraction_result.overall_confidence
                return fields_dict
            else:
                # Fallback if result format is unexpected
                return {
                    'document_type': doc_type,
                    'extraction_result': str(extraction_result),
                    'confidence_scores': {}
                }
            
        except Exception as e:
            logger.warning(f"Field extraction failed: {e}")
            # Return minimal fallback data
            return {
                'document_type': doc_type,
                'extraction_error': str(e),
                'confidence_scores': {}
            }
    
    async def _validate_data(self, extracted_fields: Dict, doc_type: str) -> List[Any]:
        """Validate extracted data against business rules."""
        try:
            # Use the ComprehensiveValidator
            loop = asyncio.get_event_loop()
            validation_results = await loop.run_in_executor(
                None, self.validator.validate_document, extracted_fields, doc_type
            )
            
            return validation_results
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return []
    
    def _generate_summary(self, result: Dict, processing_start: float) -> Dict[str, Any]:
        """Generate processing summary and metrics."""
        stages_completed = len([s for s in result['stages'].values() if s.get('status') == 'completed'])
        stages_failed = len([s for s in result['stages'].values() if s.get('status') == 'failed'])
        
        return {
            'total_stages': len(result['stages']),
            'completed_stages': stages_completed,
            'failed_stages': stages_failed,
            'success_rate': stages_completed / len(result['stages']) if result['stages'] else 0,
            'data_quality_score': self._calculate_quality_score(result),
            'processing_time': time.time() - processing_start,
        }
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate overall data quality score."""
        # Simple quality scoring based on validation results and confidence
        return 0.75  # Placeholder implementation
    
    def _needs_human_review(self, result: Dict, data_quality_score: float = None) -> bool:
        """Determine if document needs human review."""
        # Check for critical failures
        if any(stage.get('status') == 'failed' for stage in result['stages'].values()):
            return True
        
        # Check validation failures
        validation_failures = [r for r in result.get('validation_results', []) if not r.get('valid')]
        if len(validation_failures) > 2:
            return True
        
        return False