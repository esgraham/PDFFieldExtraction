"""
Integration module for Azure PDF Listener with Preprocessing.

This module integrates the PDF preprocessing capabilities with the Azure Storage
PDF listener, providing automatic preprocessing of PDFs as they arrive.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from azure_pdf_listener import AzurePDFListener
from pdf_preprocessor import PDFPreprocessor, preprocess_for_ocr
from azure.storage.blob import BlobClient

logger = logging.getLogger(__name__)


class PreprocessingPDFListener(AzurePDFListener):
    """
    Extended Azure PDF Listener with automatic preprocessing capabilities.
    
    Automatically applies preprocessing (deskewing, denoising, enhancement)
    to PDFs as they arrive in Azure Storage.
    """
    
    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
        connection_string: Optional[str] = None,
        use_managed_identity: bool = False,
        polling_interval: int = 30,
        log_level: str = "INFO",
        # Preprocessing parameters
        enable_preprocessing: bool = True,
        preprocessing_dpi: int = 300,
        enable_deskew: bool = True,
        enable_denoise: bool = True,
        enable_enhancement: bool = True,
        save_preprocessed: bool = False,
        preprocessed_container: Optional[str] = None,
        local_processing_dir: str = "processing"
    ):
        """
        Initialize the preprocessing PDF listener.
        
        Args:
            storage_account_name: Azure Storage account name
            container_name: Container to monitor for PDFs
            connection_string: Azure Storage connection string
            use_managed_identity: Use managed identity for authentication
            polling_interval: Polling interval in seconds
            log_level: Logging level
            enable_preprocessing: Enable automatic preprocessing
            preprocessing_dpi: DPI for PDF to image conversion
            enable_deskew: Enable deskewing
            enable_denoise: Enable denoising
            enable_enhancement: Enable image enhancement
            save_preprocessed: Save preprocessed images back to Azure
            preprocessed_container: Container for preprocessed images
            local_processing_dir: Local directory for processing
        """
        super().__init__(
            storage_account_name=storage_account_name,
            container_name=container_name,
            connection_string=connection_string,
            use_managed_identity=use_managed_identity,
            polling_interval=polling_interval,
            log_level=log_level
        )
        
        # Preprocessing configuration
        self.enable_preprocessing = enable_preprocessing
        self.save_preprocessed = save_preprocessed
        self.preprocessed_container = preprocessed_container
        self.local_processing_dir = Path(local_processing_dir)
        
        # Create local processing directory
        self.local_processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize preprocessor
        if self.enable_preprocessing:
            self.preprocessor = PDFPreprocessor(
                dpi=preprocessing_dpi,
                enable_deskew=enable_deskew,
                enable_denoise=enable_denoise,
                enable_enhancement=enable_enhancement,
                debug_mode=True  # Enable debug for troubleshooting
            )
        else:
            self.preprocessor = None
        
        # Processing statistics
        self.processing_stats = {
            'pdfs_processed': 0,
            'preprocessing_failures': 0,
            'total_pages_processed': 0,
            'average_processing_time': 0.0
        }
    
    def set_preprocessing_callback(
        self, 
        callback: Callable[[str, List[Any], Dict[str, Any]], None]
    ):
        """
        Set callback for preprocessing completion.
        
        Args:
            callback: Function called with (blob_name, processed_images, stats)
        """
        self.preprocessing_callback = callback
    
    def process_pdf_with_preprocessing(
        self, 
        blob_name: str, 
        blob_client: BlobClient
    ) -> Optional[List[Any]]:
        """
        Process a PDF with preprocessing pipeline.
        
        Args:
            blob_name: Name of the PDF blob
            blob_client: Azure blob client
            
        Returns:
            List of preprocessed images or None if processing failed
        """
        if not self.enable_preprocessing:
            logger.info(f"Preprocessing disabled for {blob_name}")
            return None
        
        start_time = datetime.now()
        
        try:
            # Download PDF to local processing directory
            local_pdf_path = self.local_processing_dir / blob_name
            
            logger.info(f"Downloading {blob_name} for preprocessing...")
            with open(local_pdf_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            # Create output directory for this PDF
            output_dir = self.local_processing_dir / f"{Path(blob_name).stem}_processed"
            output_dir.mkdir(exist_ok=True)
            
            # Apply preprocessing
            logger.info(f"Applying preprocessing to {blob_name}...")
            processed_images = self.preprocessor.process_pdf(
                local_pdf_path, 
                output_dir
            )
            
            # Get processing statistics
            processing_stats = self.preprocessor.get_processing_stats()
            
            # Update global statistics
            self.processing_stats['pdfs_processed'] += 1
            self.processing_stats['total_pages_processed'] += len(processed_images)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            current_avg = self.processing_stats['average_processing_time']
            count = self.processing_stats['pdfs_processed']
            self.processing_stats['average_processing_time'] = (
                (current_avg * (count - 1) + processing_time) / count
            )
            
            logger.info(
                f"Preprocessing completed for {blob_name}: "
                f"{len(processed_images)} pages in {processing_time:.2f}s"
            )
            
            # Save preprocessed images back to Azure if configured
            if self.save_preprocessed and self.preprocessed_container:
                self._upload_preprocessed_images(
                    blob_name, processed_images, output_dir
                )
            
            # Call preprocessing callback if registered
            if hasattr(self, 'preprocessing_callback'):
                self.preprocessing_callback(blob_name, processed_images, processing_stats)
            
            # Cleanup local files
            self._cleanup_local_files(local_pdf_path, output_dir)
            
            return processed_images
            
        except Exception as e:
            self.processing_stats['preprocessing_failures'] += 1
            logger.error(f"Preprocessing failed for {blob_name}: {e}")
            
            # Cleanup on failure
            if 'local_pdf_path' in locals():
                self._cleanup_local_files(local_pdf_path, output_dir if 'output_dir' in locals() else None)
            
            return None
    
    def _upload_preprocessed_images(
        self, 
        original_blob_name: str, 
        processed_images: List[Any], 
        output_dir: Path
    ):
        """Upload preprocessed images back to Azure Storage."""
        try:
            # Get container client for preprocessed images
            preprocessed_container_client = self.blob_service_client.get_container_client(
                self.preprocessed_container
            )
            
            # Create container if it doesn't exist
            try:
                preprocessed_container_client.create_container()
            except Exception:
                pass  # Container probably already exists
            
            # Upload each processed image
            base_name = Path(original_blob_name).stem
            
            for i, image in enumerate(processed_images):
                # Save image locally first
                image_filename = f"{base_name}_page_{i:03d}_processed.png"
                image_path = output_dir / image_filename
                
                # Convert numpy array to image and save
                import cv2
                cv2.imwrite(str(image_path), image)
                
                # Upload to Azure
                blob_name = f"{base_name}/{image_filename}"
                
                with open(image_path, "rb") as data:
                    preprocessed_container_client.upload_blob(
                        name=blob_name,
                        data=data,
                        overwrite=True
                    )
                
                logger.debug(f"Uploaded preprocessed image: {blob_name}")
            
            logger.info(f"Uploaded {len(processed_images)} preprocessed images for {original_blob_name}")
            
        except Exception as e:
            logger.error(f"Failed to upload preprocessed images for {original_blob_name}: {e}")
    
    def _cleanup_local_files(self, pdf_path: Path, output_dir: Optional[Path] = None):
        """Clean up local processing files."""
        try:
            # Remove downloaded PDF
            if pdf_path.exists():
                pdf_path.unlink()
            
            # Remove processing directory
            if output_dir and output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
                
        except Exception as e:
            logger.warning(f"Failed to cleanup local files: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add Azure listener statistics
        azure_stats = super().get_processing_stats() if hasattr(super(), 'get_processing_stats') else {}
        
        # Add preprocessor statistics if available
        if self.preprocessor:
            preprocessor_stats = self.preprocessor.get_processing_stats()
            stats.update(preprocessor_stats)
        
        return {**stats, **azure_stats}


class OCRIntegratedListener(PreprocessingPDFListener):
    """
    PDF Listener with integrated OCR capabilities.
    
    Extends preprocessing listener with OCR extraction using popular OCR engines.
    """
    
    def __init__(
        self,
        *args,
        ocr_engine: str = "tesseract",
        ocr_languages: List[str] = ["eng"],
        extract_tables: bool = False,
        **kwargs
    ):
        """
        Initialize OCR integrated listener.
        
        Args:
            ocr_engine: OCR engine to use ('tesseract', 'easyocr', 'paddleocr')
            ocr_languages: List of languages for OCR
            extract_tables: Enable table extraction
            *args, **kwargs: Arguments for parent class
        """
        super().__init__(*args, **kwargs)
        
        self.ocr_engine = ocr_engine
        self.ocr_languages = ocr_languages
        self.extract_tables = extract_tables
        
        # Initialize OCR engine
        self._initialize_ocr_engine()
    
    def _initialize_ocr_engine(self):
        """Initialize the selected OCR engine."""
        try:
            if self.ocr_engine.lower() == "tesseract":
                import pytesseract
                self.ocr_processor = pytesseract
                
            elif self.ocr_engine.lower() == "easyocr":
                import easyocr
                self.ocr_processor = easyocr.Reader(self.ocr_languages)
                
            elif self.ocr_engine.lower() == "paddleocr":
                from paddleocr import PaddleOCR
                self.ocr_processor = PaddleOCR(use_angle_cls=True, lang='en')
                
            else:
                raise ValueError(f"Unsupported OCR engine: {self.ocr_engine}")
                
            logger.info(f"Initialized {self.ocr_engine} OCR engine")
            
        except ImportError as e:
            logger.error(f"Failed to import {self.ocr_engine}: {e}")
            self.ocr_processor = None
    
    def extract_text_from_images(self, images: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract text from preprocessed images using OCR.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        if not self.ocr_processor:
            logger.error("OCR engine not initialized")
            return []
        
        results = []
        
        for i, image in enumerate(images):
            try:
                result = {
                    'page_number': i + 1,
                    'text': '',
                    'confidence': 0.0,
                    'word_count': 0,
                    'bounding_boxes': []
                }
                
                if self.ocr_engine.lower() == "tesseract":
                    # Tesseract OCR
                    text = self.ocr_processor.image_to_string(
                        image, 
                        lang='+'.join(self.ocr_languages)
                    )
                    
                    # Get detailed data with bounding boxes
                    data = self.ocr_processor.image_to_data(
                        image, 
                        output_type=self.ocr_processor.Output.DICT
                    )
                    
                    result['text'] = text.strip()
                    result['word_count'] = len(text.split())
                    
                    # Extract confidence and bounding boxes
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        result['confidence'] = sum(confidences) / len(confidences)
                    
                    # Extract bounding boxes for words
                    for j in range(len(data['text'])):
                        if int(data['conf'][j]) > 30:  # Confidence threshold
                            result['bounding_boxes'].append({
                                'text': data['text'][j],
                                'confidence': int(data['conf'][j]),
                                'bbox': (
                                    data['left'][j],
                                    data['top'][j],
                                    data['width'][j],
                                    data['height'][j]
                                )
                            })
                
                elif self.ocr_engine.lower() == "easyocr":
                    # EasyOCR
                    results_easyocr = self.ocr_processor.readtext(image)
                    
                    text_parts = []
                    confidences = []
                    bboxes = []
                    
                    for (bbox, text, conf) in results_easyocr:
                        text_parts.append(text)
                        confidences.append(conf)
                        bboxes.append({
                            'text': text,
                            'confidence': conf * 100,  # Convert to percentage
                            'bbox': bbox
                        })
                    
                    result['text'] = ' '.join(text_parts)
                    result['confidence'] = sum(confidences) / len(confidences) * 100 if confidences else 0
                    result['word_count'] = len(result['text'].split())
                    result['bounding_boxes'] = bboxes
                
                results.append(result)
                logger.debug(f"Extracted {result['word_count']} words from page {i + 1}")
                
            except Exception as e:
                logger.error(f"OCR failed for page {i + 1}: {e}")
                results.append({
                    'page_number': i + 1,
                    'text': '',
                    'confidence': 0.0,
                    'word_count': 0,
                    'bounding_boxes': [],
                    'error': str(e)
                })
        
        return results


def create_preprocessing_listener(
    storage_account_name: str,
    container_name: str,
    connection_string: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> PreprocessingPDFListener:
    """
    Factory function to create a preprocessing PDF listener with sensible defaults.
    
    Args:
        storage_account_name: Azure Storage account name
        container_name: Container to monitor
        connection_string: Azure Storage connection string
        preprocessing_config: Optional preprocessing configuration
        
    Returns:
        Configured PreprocessingPDFListener instance
    """
    # Default preprocessing configuration
    default_config = {
        'enable_preprocessing': True,
        'preprocessing_dpi': 300,
        'enable_deskew': True,
        'enable_denoise': True,
        'enable_enhancement': True,
        'save_preprocessed': False,
        'local_processing_dir': 'processing'
    }
    
    # Merge with user configuration
    if preprocessing_config:
        default_config.update(preprocessing_config)
    
    return PreprocessingPDFListener(
        storage_account_name=storage_account_name,
        container_name=container_name,
        connection_string=connection_string,
        **default_config
    )


def create_ocr_listener(
    storage_account_name: str,
    container_name: str,
    connection_string: Optional[str] = None,
    ocr_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> OCRIntegratedListener:
    """
    Factory function to create an OCR integrated listener.
    
    Args:
        storage_account_name: Azure Storage account name
        container_name: Container to monitor
        connection_string: Azure Storage connection string
        ocr_config: OCR configuration
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Configured OCRIntegratedListener instance
    """
    # Default configurations
    default_ocr_config = {
        'ocr_engine': 'tesseract',
        'ocr_languages': ['eng'],
        'extract_tables': False
    }
    
    default_preprocessing_config = {
        'enable_preprocessing': True,
        'preprocessing_dpi': 300,
        'enable_deskew': True,
        'enable_denoise': True,
        'enable_enhancement': True
    }
    
    # Merge configurations
    if ocr_config:
        default_ocr_config.update(ocr_config)
    if preprocessing_config:
        default_preprocessing_config.update(preprocessing_config)
    
    return OCRIntegratedListener(
        storage_account_name=storage_account_name,
        container_name=container_name,
        connection_string=connection_string,
        **default_ocr_config,
        **default_preprocessing_config
    )