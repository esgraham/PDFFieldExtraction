"""
Core processing modules for PDF field extraction.

This package contains the core functionality for:
- Azure PDF monitoring and file handling
- PDF preprocessing (deskew, denoise)
- Document classification
- OCR integration with Azure Document Intelligence
- Field extraction and validation
- Integration components
"""

# Lazy imports to avoid slow startup with heavy dependencies like PyTorch
# Import modules directly in your code instead of using this __init__.py

__all__ = [
    'AzurePDFListener',
    'PDFPreprocessor', 
    'DocumentClassifier',
    'AzureDocumentIntelligence',
    'FieldExtractor',
    'ComprehensiveValidator'
]