"""
PDF Field Extraction Package

A comprehensive Python package for processing PDF files with field extraction capabilities.
"""

__version__ = "1.0.0"
__author__ = "PDF Field Extraction Team"

# Available core modules
try:
    from .core.field_extraction import FieldExtractor
    __all__ = ['FieldExtractor']
except ImportError:
    __all__ = []