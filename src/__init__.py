"""
Azure Storage PDF File Listener with Preprocessing

A comprehensive Python package for monitoring Azure Storage containers for new PDF files
with advanced preprocessing and OCR capabilities.
"""

from .azure_pdf_listener import AzurePDFListener

# Initialize __all__ list
__all__ = ['AzurePDFListener']

# Optional preprocessing imports with error handling
try:
    from .pdf_preprocessor import PDFPreprocessor
    from .pdf_integration import PreprocessingPDFListener, OCRIntegratedListener
    PREPROCESSING_AVAILABLE = True
    __all__.extend(['PDFPreprocessor', 'PreprocessingPDFListener', 'OCRIntegratedListener'])
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    print(f"Warning: Preprocessing features not available: {e}")

# Optional classification imports with error handling
try:
    from .document_classifier import DocumentClassifier, DocumentClass, ClassificationResult
    from .classification_integration import ClassificationIntegratedListener, create_classification_pipeline
    CLASSIFICATION_AVAILABLE = True
    __all__.extend(['DocumentClassifier', 'DocumentClass', 'ClassificationResult', 
                   'ClassificationIntegratedListener', 'create_classification_pipeline'])
except ImportError as e:
    CLASSIFICATION_AVAILABLE = False
    print(f"Warning: Classification features not available: {e}")

__version__ = "1.1.0"
__author__ = "Azure PDF Listener Team"
__email__ = "support@example.com"

# Core exports
__all__ = ["AzurePDFListener"]

# Add preprocessing exports if available
if PREPROCESSING_AVAILABLE:
    __all__.extend([
        "PDFPreprocessor",
        "PreprocessingPDFListener", 
        "OCRIntegratedListener",
        "preprocess_for_ocr",
        "batch_preprocess",
        "create_preprocessing_listener",
        "create_ocr_listener"
    ])


def check_preprocessing_dependencies():
    """Check if preprocessing dependencies are installed."""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import skimage
    except ImportError:
        missing.append("scikit-image")
    
    try:
        import PIL
    except ImportError:
        missing.append("Pillow")
    
    try:
        import fitz
    except ImportError:
        missing.append("PyMuPDF")
    
    return missing


def check_ocr_dependencies():
    """Check if OCR dependencies are installed."""
    engines = {}
    
    try:
        import pytesseract
        engines['tesseract'] = True
    except ImportError:
        engines['tesseract'] = False
    
    try:
        import easyocr
        engines['easyocr'] = True
    except ImportError:
        engines['easyocr'] = False
    
    try:
        import paddleocr
        engines['paddleocr'] = True
    except ImportError:
        engines['paddleocr'] = False
    
    return engines


def get_installation_info():
    """Get information about available features and installation requirements."""
    info = {
        "core_features": {
            "azure_storage_monitoring": True,
            "pdf_processing": True,
            "event_driven_monitoring": True
        },
        "optional_features": {
            "preprocessing": PREPROCESSING_AVAILABLE,
            "classification": CLASSIFICATION_AVAILABLE,
            "ocr_integration": check_ocr_dependencies()
        }
    }