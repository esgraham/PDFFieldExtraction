"""
Document Classification Module

This module provides document classification capabilities using layout analysis,
text embeddings, and visual features. Starts with traditional ML approaches
and can be upgraded to transformer-based models.

Features:
- Layout analysis (page size, text distribution, logo detection)
- Text embeddings for content classification
- Visual feature extraction
- Multi-modal classification pipeline
- Extensible architecture for transformer integration
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from collections import Counter

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def _get_sentence_transformer():
    """Lazy import of sentence transformers to avoid slow startup."""
    global _sentence_transformer_module, TRANSFORMERS_AVAILABLE
    if _sentence_transformer_module is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_module = SentenceTransformer
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
            _sentence_transformer_module = False
    return _sentence_transformer_module if _sentence_transformer_module is not False else None

# Optional OCR support
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Azure Document Intelligence support
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient, DocumentIntelligenceAdministrationClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
    from azure.core.credentials import AzureKeyCredential
    AZURE_DOC_INTEL_AVAILABLE = True
except ImportError:
    AZURE_DOC_INTEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Document types are now dynamically registered per classifier instance
# No hardcoded enum - full flexibility for any document type

@dataclass
class LayoutFeatures:
    """Document layout analysis features."""
    page_width: float
    page_height: float
    aspect_ratio: float
    text_density: float
    white_space_ratio: float
    text_block_count: int
    average_text_block_size: float
    has_logo: bool
    logo_position: Optional[Tuple[float, float]]
    text_alignment: str  # 'left', 'center', 'right', 'justified'
    font_size_distribution: Dict[str, float]
    margin_sizes: Dict[str, float]  # top, bottom, left, right

@dataclass
class TextFeatures:
    """Document text content features."""
    word_count: int
    unique_word_count: int
    avg_word_length: float
    sentence_count: int
    avg_sentence_length: float
    language: str
    has_numbers: bool
    has_dates: bool
    has_currency: bool
    has_email: bool
    has_phone: bool
    text_embedding: Optional[np.ndarray]
    key_phrases: List[str]

@dataclass
class VisualFeatures:
    """Document visual appearance features."""
    color_histogram: np.ndarray
    edge_density: float
    texture_features: np.ndarray
    line_density: float
    has_tables: bool
    has_images: bool
    has_stamps: bool
    dominant_colors: List[Tuple[int, int, int]]

@dataclass
class ClassificationResult:
    """Result of document classification."""
    document_type: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, Any]
    processing_time: float
    model_used: str
    azure_analysis: Optional[Dict[str, Any]] = None  # Azure prebuilt model results

@dataclass
class AzureClassificationResult:
    """Result from Azure Document Intelligence prebuilt model classification."""
    document_type: str
    confidence: float
    model_used: str
    fields_detected: Dict[str, Any]
    raw_result: AnalyzeResult

class DocumentClassifier:
    """
    Multi-modal document classifier using layout, text, and visual features.
    
    Supports traditional ML models with option to upgrade to transformers.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        use_transformers: bool = False,
        transformer_model: str = "all-MiniLM-L6-v2",
        enable_ocr: bool = True,
        cache_features: bool = True,
        debug_mode: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        use_azure_prebuilt: bool = True,
        azure_confidence_threshold: float = 0.7
    ):
        """
        Initialize the document classifier.
        
        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boost', 'svm', 'logistic')
            use_transformers: Whether to use transformer-based text embeddings
            transformer_model: Name of the sentence transformer model
            enable_ocr: Whether to extract text using OCR
            cache_features: Whether to cache extracted features
            debug_mode: Enable detailed logging and feature saving
            azure_endpoint: Azure Document Intelligence endpoint
            azure_api_key: Azure Document Intelligence API key
            use_azure_prebuilt: Whether to use Azure prebuilt models for classification
            azure_confidence_threshold: Minimum confidence for Azure classification
        """
        self.model_type = model_type
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.cache_features = cache_features
        self.debug_mode = debug_mode
        self.use_azure_prebuilt = use_azure_prebuilt and AZURE_DOC_INTEL_AVAILABLE
        self.azure_confidence_threshold = azure_confidence_threshold
        
        # Initialize models
        self.classifier = self._create_classifier()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize transformer model if requested (lazy loading)
        self.sentence_transformer = None
        self.transformer_model = transformer_model
        if self.use_transformers:
            # Check if transformers are available but don't load yet
            SentenceTransformer = _get_sentence_transformer()
            if SentenceTransformer is None:
                logger.warning("Sentence transformers not available, disabling transformer features")
                self.use_transformers = False
            else:
                logger.info(f"Transformer support enabled, will load {transformer_model} when needed")
        
        # Specific Azure prebuilt models as requested
        self.azure_prebuilt_models = [
            ("prebuilt-invoice", "invoice"),
            ("prebuilt-receipt", "receipt"),
            ("prebuilt-contract", "contract"), 
            ("prebuilt-idDocument", "id_document"),
            ("prebuilt-document", "general_document")
        ]
        
        # Document type registry with specific Azure types
        self.document_types = set(["unknown", "invoice", "receipt", "contract", "id_document", "general_document"])
        
        # Field mappings loaded from schema files
        self.azure_field_mappings = self._load_field_mappings_from_schemas()
        self.schemas_dir = Path(__file__).parent.parent.parent / "config" / "schemas"
        
        # Initialize Azure Document Intelligence clients
        self.azure_client = None
        self.azure_admin_client = None
        if self.use_azure_prebuilt and azure_endpoint and azure_api_key:
            try:
                credential = AzureKeyCredential(azure_api_key)
                self.azure_client = DocumentIntelligenceClient(
                    endpoint=azure_endpoint,
                    credential=credential
                )
                self.azure_admin_client = DocumentIntelligenceAdministrationClient(
                    endpoint=azure_endpoint,
                    credential=credential
                )
                logger.info("Azure Document Intelligence clients initialized")
                
                # Validate and filter available models
                self._validate_azure_models()
                logger.info(f"Using available Azure models: {[model_id for model_id, _ in self.azure_prebuilt_models]}")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize Azure clients: {e}")
                self.use_azure_prebuilt = False
        elif self.use_azure_prebuilt:
            logger.warning("Azure endpoint or API key not provided, disabling Azure prebuilt models")
            self.use_azure_prebuilt = False
        
        # Feature cache
        self.feature_cache = {} if cache_features else None
        
        # Model state
        self.is_trained = False
        self.feature_names = []
        
        logger.info(f"DocumentClassifier initialized: {model_type}, transformers={self.use_transformers}")
    
    def _load_field_mappings_from_schemas(self) -> Dict[str, List[str]]:
        """
        Load field mappings from schema JSON files.
        
        Returns:
            Dictionary mapping document types to field lists
        """
        schemas_dir = Path(__file__).parent.parent.parent / "config" / "schemas"
        field_mappings = {}
        
        # Expected schema files for our document types
        document_types = ["invoice", "receipt", "contract", "id_document", "general_document"]
        
        for doc_type in document_types:
            schema_file = schemas_dir / f"{doc_type}.json"
            
            if schema_file.exists():
                try:
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                    
                    # Extract field names from schema
                    field_names = []
                    if 'fields' in schema:
                        for field_def in schema['fields']:
                            if 'name' in field_def:
                                field_names.append(field_def['name'])
                    
                    field_mappings[doc_type] = field_names
                    logger.info(f"Loaded {len(field_names)} fields from {doc_type} schema")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to load schema for {doc_type}: {e}")
                    # Fallback to basic field mapping
                    field_mappings[doc_type] = self._get_fallback_fields(doc_type)
            else:
                logger.warning(f"Schema file not found: {schema_file}")
                # Fallback to basic field mapping
                field_mappings[doc_type] = self._get_fallback_fields(doc_type)
        
        return field_mappings
    
    def _get_fallback_fields(self, doc_type: str) -> List[str]:
        """
        Get fallback field mappings if schema files are not available.
        
        Args:
            doc_type: Document type
            
        Returns:
            List of basic field names for the document type
        """
        fallback_mappings = {
            "invoice": ["VendorName", "InvoiceTotal", "InvoiceDate", "InvoiceId"],
            "receipt": ["MerchantName", "Total", "TransactionDate"],
            "contract": ["Parties", "ContractDate", "Terms"],
            "id_document": ["FirstName", "LastName", "DocumentNumber", "DateOfBirth"],
            "general_document": ["Content", "Pages"]
        }
        return fallback_mappings.get(doc_type, ["content"])
    
    def reload_schemas(self):
        """
        Reload field mappings from schema files.
        Useful for updating schemas without restarting the application.
        """
        self.azure_field_mappings = self._load_field_mappings_from_schemas()
        logger.info("Schema field mappings reloaded")
    
    def _validate_azure_models(self):
        """
        Validate which Azure prebuilt models are actually available and filter out unavailable ones.
        """
        if not self.azure_admin_client:
            logger.warning("Cannot validate Azure models - admin client not available")
            return
        
        try:
            # Get list of available models
            available_models = set()
            for model in self.azure_admin_client.list_models():
                available_models.add(model.model_id)
            
            # Filter out unavailable models
            validated_models = []
            for model_id, doc_type in self.azure_prebuilt_models:
                if model_id in available_models:
                    validated_models.append((model_id, doc_type))
                    logger.debug(f"✅ Model available: {model_id}")
                else:
                    logger.warning(f"⚠️ Model not available in subscription: {model_id}")
            
            # Update the models list with only available models
            self.azure_prebuilt_models = validated_models
            
            if not validated_models:
                logger.error("❌ No Azure prebuilt models are available in this subscription")
                self.use_azure_prebuilt = False
            else:
                logger.info(f"✅ Validated {len(validated_models)} available Azure models")
                
        except Exception as e:
            logger.warning(f"Failed to validate Azure models: {e}")
            logger.info("Will attempt to use models as configured (may fail if unavailable)")
    
    def get_schema_for_document_type(self, doc_type: str) -> Optional[Dict[str, Any]]:
        """
        Load and return the complete schema for a document type.
        
        Args:
            doc_type: Document type
            
        Returns:
            Complete schema dictionary or None if not found
        """
        schema_file = self.schemas_dir / f"{doc_type}.json"
        
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load schema for {doc_type}: {e}")
        
        return None
    
    def get_field_synonyms(self, doc_type: str, field_name: str) -> List[str]:
        """
        Get synonyms for a field from the schema.
        
        Args:
            doc_type: Document type
            field_name: Field name
            
        Returns:
            List of synonyms for the field
        """
        schema = self.get_schema_for_document_type(doc_type)
        if not schema or 'fields' not in schema:
            return []
        
        for field_def in schema['fields']:
            if field_def.get('name') == field_name:
                return field_def.get('synonyms', [])
        
        return []
    
    def get_field_validation_rules(self, doc_type: str, field_name: str) -> List[str]:
        """
        Get validation rules for a field from the schema.
        
        Args:
            doc_type: Document type
            field_name: Field name
            
        Returns:
            List of validation rules for the field
        """
        schema = self.get_schema_for_document_type(doc_type)
        if not schema or 'fields' not in schema:
            return []
        
        for field_def in schema['fields']:
            if field_def.get('name') == field_name:
                return field_def.get('validation_rules', [])
        
        return []
    
    def _create_classifier(self):
        """Create the appropriate ML classifier."""
        classifiers = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            "svm": SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            "logistic": LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        return classifiers.get(self.model_type, classifiers["random_forest"])
    
    def extract_layout_features(self, image: np.ndarray) -> LayoutFeatures:
        """
        Extract layout-based features from document image.
        
        Args:
            image: Document image as numpy array
            
        Returns:
            Layout features dataclass
        """
        height, width = image.shape[:2]
        
        # Basic page dimensions
        aspect_ratio = width / height
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Text region detection using MSER or contours
        text_regions = self._detect_text_regions(gray)
        text_area = sum(cv2.contourArea(region) for region in text_regions)
        total_area = width * height
        text_density = text_area / total_area
        white_space_ratio = 1.0 - text_density
        
        # Logo detection (simple approach using template matching or feature detection)
        has_logo, logo_pos = self._detect_logo(gray)
        
        # Text alignment analysis
        text_alignment = self._analyze_text_alignment(text_regions, width)
        
        # Font size estimation
        font_sizes = self._estimate_font_sizes(text_regions)
        
        # Margin analysis
        margins = self._analyze_margins(text_regions, width, height)
        
        return LayoutFeatures(
            page_width=float(width),
            page_height=float(height),
            aspect_ratio=aspect_ratio,
            text_density=text_density,
            white_space_ratio=white_space_ratio,
            text_block_count=len(text_regions),
            average_text_block_size=text_area / max(len(text_regions), 1),
            has_logo=has_logo,
            logo_position=logo_pos,
            text_alignment=text_alignment,
            font_size_distribution=font_sizes,
            margin_sizes=margins
        )
    
    def extract_text_features(self, image: np.ndarray, text: Optional[str] = None) -> TextFeatures:
        """
        Extract text-based features from document.
        
        Args:
            image: Document image
            text: Pre-extracted text (if None, will use OCR)
            
        Returns:
            Text features dataclass
        """
        # Extract text if not provided
        if text is None and self.enable_ocr:
            text = pytesseract.image_to_string(image)
        elif text is None:
            text = ""
        
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        unique_words = set(words)
        unique_word_count = len(unique_words)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Content pattern detection
        has_numbers = bool(re.search(r'\d', text))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text))
        has_currency = bool(re.search(r'[$€£¥]|\b\d+\.\d{2}\b', text))
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        
        # Language detection (simple heuristic)
        language = self._detect_language(text)
        
        # Text embedding (lazy load transformer when needed)
        text_embedding = None
        if self.use_transformers and text.strip():
            try:
                # Lazy load the transformer model when actually needed
                if self.sentence_transformer is None:
                    SentenceTransformer = _get_sentence_transformer()
                    if SentenceTransformer:
                        self.sentence_transformer = SentenceTransformer(self.transformer_model)
                        logger.info(f"Loaded transformer model: {self.transformer_model}")
                    else:
                        logger.warning("Failed to load sentence transformers")
                        self.use_transformers = False
                
                if self.sentence_transformer:
                    text_embedding = self.sentence_transformer.encode(text)
            except Exception as e:
                logger.warning(f"Failed to generate text embedding: {e}")
        
        # Key phrase extraction
        key_phrases = self._extract_key_phrases(text)
        
        return TextFeatures(
            word_count=word_count,
            unique_word_count=unique_word_count,
            avg_word_length=avg_word_length,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            language=language,
            has_numbers=has_numbers,
            has_dates=has_dates,
            has_currency=has_currency,
            has_email=has_email,
            has_phone=has_phone,
            text_embedding=text_embedding,
            key_phrases=key_phrases
        )
    
    def extract_visual_features(self, image: np.ndarray) -> VisualFeatures:
        """
        Extract visual appearance features from document image.
        
        Args:
            image: Document image as numpy array
            
        Returns:
            Visual features dataclass
        """
        # Convert to different color spaces for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image.copy()
            hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        # Color histogram (handle grayscale vs color images)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Color image
            color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        else:
            # Grayscale image
            color_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        color_hist = color_hist.flatten()
        color_hist = color_hist / color_hist.sum()  # Normalize
        
        # Edge detection and density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis using Local Binary Patterns (simplified)
        texture_features = self._extract_texture_features(gray)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        line_density = len(lines) / edges.size if lines is not None else 0
        
        # Table detection (simplified)
        has_tables = self._detect_tables(gray)
        
        # Image detection
        has_images = self._detect_embedded_images(gray)
        
        # Stamp detection
        has_stamps = self._detect_stamps(gray)
        
        # Dominant colors
        dominant_colors = self._extract_dominant_colors(image if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        
        return VisualFeatures(
            color_histogram=color_hist,
            edge_density=edge_density,
            texture_features=texture_features,
            line_density=line_density,
            has_tables=has_tables,
            has_images=has_images,
            has_stamps=has_stamps,
            dominant_colors=dominant_colors
        )
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions in the image."""
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)
        
        # Filter regions by size and aspect ratio
        text_regions = []
        for region in regions:
            if len(region) > 10:  # Minimum region size
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                text_regions.append(hull)
        
        return text_regions
    
    def _detect_logo(self, gray_image: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Simple logo detection using corner detection."""
        # Look for logo-like features in typical logo positions (top corners, center top)
        height, width = gray_image.shape
        
        # Define regions where logos typically appear
        logo_regions = [
            gray_image[0:height//4, 0:width//4],  # Top-left
            gray_image[0:height//4, 3*width//4:width],  # Top-right
            gray_image[0:height//6, width//3:2*width//3]  # Top-center
        ]
        
        logo_positions = [(0.125, 0.125), (0.875, 0.125), (0.5, 0.083)]
        
        for i, region in enumerate(logo_regions):
            # Use corner detection to find logo-like features
            corners = cv2.goodFeaturesToTrack(region, maxCorners=100, qualityLevel=0.01, minDistance=10)
            if corners is not None and len(corners) > 20:  # Threshold for logo detection
                return True, logo_positions[i]
        
        return False, None
    
    def _analyze_text_alignment(self, text_regions: List[np.ndarray], page_width: int) -> str:
        """Analyze predominant text alignment."""
        if not text_regions:
            return "unknown"
        
        left_edges = []
        right_edges = []
        
        for region in text_regions:
            x_coords = region[:, 0, 0]
            left_edges.append(np.min(x_coords))
            right_edges.append(np.max(x_coords))
        
        # Analyze alignment patterns
        left_variance = np.var(left_edges)
        right_variance = np.var(right_edges)
        center_positions = [(l + r) / 2 for l, r in zip(left_edges, right_edges)]
        center_variance = np.var(center_positions)
        
        # Determine alignment type
        if left_variance < page_width * 0.01 and right_variance < page_width * 0.01:
            return "justified"
        elif left_variance < page_width * 0.01:
            return "left"
        elif right_variance < page_width * 0.01:
            return "right"
        elif center_variance < page_width * 0.01:
            return "center"
        else:
            return "mixed"
    
    def _estimate_font_sizes(self, text_regions: List[np.ndarray]) -> Dict[str, float]:
        """Estimate font size distribution."""
        heights = []
        for region in text_regions:
            y_coords = region[:, 0, 1]
            height = np.max(y_coords) - np.min(y_coords)
            heights.append(height)
        
        if not heights:
            return {"small": 0.0, "medium": 0.0, "large": 0.0}
        
        # Categorize font sizes
        heights = np.array(heights)
        small_threshold = np.percentile(heights, 33)
        large_threshold = np.percentile(heights, 67)
        
        small_count = np.sum(heights <= small_threshold)
        medium_count = np.sum((heights > small_threshold) & (heights <= large_threshold))
        large_count = np.sum(heights > large_threshold)
        
        total = len(heights)
        return {
            "small": small_count / total,
            "medium": medium_count / total,
            "large": large_count / total
        }
    
    def _analyze_margins(self, text_regions: List[np.ndarray], width: int, height: int) -> Dict[str, float]:
        """Analyze document margins."""
        if not text_regions:
            return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
        
        all_points = np.vstack(text_regions)
        x_coords = all_points[:, 0, 0]
        y_coords = all_points[:, 0, 1]
        
        left_margin = np.min(x_coords) / width
        right_margin = (width - np.max(x_coords)) / width
        top_margin = np.min(y_coords) / height
        bottom_margin = (height - np.max(y_coords)) / height
        
        return {
            "top": top_margin,
            "bottom": bottom_margin,
            "left": left_margin,
            "right": right_margin
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # This is a simplified approach - could be enhanced with proper language detection libraries
        if not text.strip():
            return "unknown"
        
        # Count character frequencies for basic language detection
        english_chars = sum(1 for c in text.lower() if c in 'abcdefghijklmnopqrstuvwxyz')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return "unknown"
        
        english_ratio = english_chars / total_chars
        return "english" if english_ratio > 0.8 else "other"
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Filter out common stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = [word for word, freq in word_freq.most_common(10) 
                    if word not in stop_words and len(word) > 3]
        
        return key_words[:5]  # Return top 5 key phrases
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract texture features using statistical measures."""
        # Calculate texture features using gray-level co-occurrence matrix (simplified)
        # This is a basic implementation - could be enhanced with more sophisticated texture analysis
        
        # Calculate basic statistical measures
        mean_val = np.mean(gray_image)
        std_val = np.std(gray_image)
        skewness = np.mean(((gray_image - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((gray_image - mean_val) / std_val) ** 4)
        
        # Edge statistics
        edges = cv2.Canny(gray_image, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        return np.array([mean_val, std_val, skewness, kurtosis, edge_ratio])
    
    def _detect_tables(self, gray_image: np.ndarray) -> bool:
        """Detect presence of tables in the document."""
        # Use line detection to identify table structures
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Check if there are intersecting horizontal and vertical lines (table indicator)
        table_mask = cv2.bitwise_and(horizontal_lines, vertical_lines)
        table_pixels = np.sum(table_mask > 0)
        
        return table_pixels > 100  # Threshold for table detection
    
    def _detect_embedded_images(self, gray_image: np.ndarray) -> bool:
        """Detect embedded images or graphics."""
        # Look for large connected components that might be images
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > gray_image.size * 0.05:  # Large area might be an image
                # Check if it's roughly rectangular (image-like)
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio < 3:  # Not too elongated
                    return True
        
        return False
    
    def _detect_stamps(self, gray_image: np.ndarray) -> bool:
        """Detect stamps or seals in the document."""
        # Look for circular or rectangular stamp-like features
        circles = cv2.HoughCircles(
            gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=100
        )
        
        if circles is not None and len(circles[0]) > 0:
            return True
        
        # Also check for rectangular stamps using template matching or feature detection
        # This is a simplified approach
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:  # Stamp-sized area
                # Check if roughly rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:  # Rectangular shape
                    return True
        
        return False
    
    def _extract_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering."""
        # Handle grayscale images by converting to pseudo-color
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Grayscale image - create pseudo RGB
            if len(image.shape) == 2:
                gray = image
            else:
                gray = image[:, :, 0]
            
            # For grayscale, return intensity levels as gray colors
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            # Find peaks in histogram
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist.flatten(), height=np.max(hist) * 0.1)
            
            # Take top k peaks as dominant "colors"
            peak_heights = hist.flatten()[peaks]
            top_peaks = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)[:k]
            
            return [(int(peak), int(peak), int(peak)) for peak, _ in top_peaks]
        
        else:
            # Color image
            pixels = image.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the colors and sort by frequency
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            color_counts = Counter(labels)
            
            # Sort colors by frequency
            sorted_colors = [colors[i] for i, _ in color_counts.most_common()]
            
            return [tuple(color) for color in sorted_colors]
    
    def extract_features(self, image: np.ndarray, text: Optional[str] = None) -> np.ndarray:
        """
        Extract all features from document image.
        
        Args:
            image: Document image as numpy array
            text: Optional pre-extracted text
            
        Returns:
            Feature vector as numpy array
        """
        # Check cache
        cache_key = hash(image.tobytes()) if self.cache_features else None
        if cache_key and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Extract different types of features
        layout_features = self.extract_layout_features(image)
        text_features = self.extract_text_features(image, text)
        visual_features = self.extract_visual_features(image)
        
        # Combine features into a single vector
        feature_vector = self._combine_features(layout_features, text_features, visual_features)
        
        # Cache the result
        if cache_key:
            self.feature_cache[cache_key] = feature_vector
        
        return feature_vector
    
    def _combine_features(self, layout: LayoutFeatures, text: TextFeatures, visual: VisualFeatures) -> np.ndarray:
        """Combine all feature types into a single vector."""
        features = []
        
        # Layout features
        features.extend([
            layout.page_width, layout.page_height, layout.aspect_ratio,
            layout.text_density, layout.white_space_ratio, layout.text_block_count,
            layout.average_text_block_size, float(layout.has_logo),
            layout.font_size_distribution.get('small', 0),
            layout.font_size_distribution.get('medium', 0),
            layout.font_size_distribution.get('large', 0),
            layout.margin_sizes.get('top', 0), layout.margin_sizes.get('bottom', 0),
            layout.margin_sizes.get('left', 0), layout.margin_sizes.get('right', 0)
        ])
        
        # Text features
        features.extend([
            text.word_count, text.unique_word_count, text.avg_word_length,
            text.sentence_count, text.avg_sentence_length,
            float(text.language == 'english'), float(text.has_numbers),
            float(text.has_dates), float(text.has_currency),
            float(text.has_email), float(text.has_phone)
        ])
        
        # Visual features
        features.extend([
            visual.edge_density, visual.line_density,
            float(visual.has_tables), float(visual.has_images), float(visual.has_stamps)
        ])
        features.extend(visual.texture_features)
        features.extend(visual.color_histogram[:50])  # Limit histogram size
        
        # Add text embedding if available
        if text.text_embedding is not None:
            features.extend(text.text_embedding)
        
        return np.array(features, dtype=np.float32)
    
    def train(self, images: List[np.ndarray], labels: List[str], texts: Optional[List[str]] = None):
        """
        Train the document classifier.
        
        Args:
            images: List of document images
            labels: List of corresponding document type strings
            texts: Optional list of extracted texts
        """
        if len(images) != len(labels):
            raise ValueError("Number of images and labels must match")
        
        if texts and len(texts) != len(images):
            raise ValueError("Number of texts must match number of images")
        
        logger.info(f"Training classifier on {len(images)} documents")
        
        # Extract features from all documents
        X = []
        y = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            try:
                text = texts[i] if texts else None
                features = self.extract_features(image, text)
                X.append(features)
                y.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract features from document {i}: {e}")
                continue
        
        if not X:
            raise ValueError("No valid features extracted from training data")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Register all document types from training labels
        for label in y:
            self.document_types.add(label)
        
        # Encode labels (now they are already strings)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y_encoded)
        
        # Store feature names for debugging
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.is_trained = True
        logger.info("Classifier training completed successfully")
    
    def classify(self, image: np.ndarray, text: Optional[str] = None) -> ClassificationResult:
        """
        Classify a document image.
        
        Args:
            image: Document image as numpy array
            text: Optional pre-extracted text
            
        Returns:
            Classification result
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before classification")
        
        import time
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(image, text)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Get predicted document type as string
        predicted_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Create probability dictionary
        class_probabilities = {}
        for i, prob in enumerate(probabilities):
            class_name = self.label_encoder.inverse_transform([i])[0]
            class_probabilities[class_name] = float(prob)
        
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            document_type=predicted_type,
            confidence=float(confidence),
            probabilities=class_probabilities,
            features={"feature_count": len(features)},
            processing_time=processing_time,
            model_used="Traditional ML"
        )
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "use_transformers": self.use_transformers,
            "feature_names": self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data["classifier"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.model_type = model_data["model_type"]
        self.use_transformers = model_data.get("use_transformers", False)
        self.feature_names = model_data.get("feature_names", [])
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def register_document_type(self, doc_type: str, expected_fields: Optional[List[str]] = None):
        """
        Register a new document type that can be returned by Azure classification.
        
        Args:
            doc_type: The document type name (e.g., "custom_contract")
            expected_fields: List of field names expected for this document type
        """
        self.document_types.add(doc_type)
        
        if expected_fields:
            self.azure_field_mappings[doc_type] = expected_fields
        
        logger.info(f"Registered document type: {doc_type}")
    
    def configure_document_fields(self, doc_type: str, expected_fields: List[str]):
        """
        Configure expected fields for a document type.
        
        Args:
            doc_type: The document type name
            expected_fields: List of field names expected for this document type
        """
        self.azure_field_mappings[doc_type] = expected_fields
        self.document_types.add(doc_type)
        logger.info(f"Configured fields for {doc_type}: {len(expected_fields)} fields")
    
    def get_registered_document_types(self) -> List[str]:
        """
        Get all registered document types.
        
        Returns:
            List of registered document type names
        """
        return sorted(list(self.document_types))
    
    def configure_azure_models(self, models: List[Tuple[str, str]]):
        """
        Configure which Azure models to use for classification.
        
        Args:
            models: List of (model_id, document_type) tuples
                   e.g., [("prebuilt-invoice", "invoice"), ("my-custom-model", "custom_type")]
        """
        self.azure_prebuilt_models = models
        
        # Register all document types from the models
        for model_id, doc_type in models:
            self.document_types.add(doc_type)
            
            # Set up minimal field mappings for confidence calculation
            if doc_type not in self.azure_field_mappings:
                self._setup_dynamic_fields_for_model(model_id, doc_type)
        
        logger.info(f"Configured {len(models)} Azure models")
    
    def _setup_dynamic_fields_for_model(self, model_id: str, doc_type: str):
        """
        Set up dynamic field mappings for a model based on patterns and heuristics.
        
        Args:
            model_id: Azure model ID
            doc_type: Document type
        """
        # Try to infer reasonable field expectations based on document type patterns
        inferred_fields = []
        
        # Use document type name to infer likely fields
        doc_type_lower = doc_type.lower()
        
        if 'invoice' in doc_type_lower:
            inferred_fields = ["vendor", "total", "date", "number", "items"]
        elif 'receipt' in doc_type_lower:
            inferred_fields = ["merchant", "total", "date", "items"]  
        elif 'business' in doc_type_lower or 'card' in doc_type_lower:
            inferred_fields = ["name", "company", "email", "phone"]
        elif 'id' in doc_type_lower or 'identity' in doc_type_lower:
            inferred_fields = ["name", "number", "date", "address"]
        elif 'health' in doc_type_lower or 'insurance' in doc_type_lower:
            inferred_fields = ["member", "insurer", "group", "policy"]
        elif 'tax' in doc_type_lower:
            inferred_fields = ["employee", "employer", "wages", "tax"]
        elif 'layout' in doc_type_lower:
            inferred_fields = ["pages", "tables", "paragraphs", "lines"]
        elif 'read' in doc_type_lower or 'text' in doc_type_lower:
            inferred_fields = ["content", "lines", "words"]
        else:
            # Generic fallback for unknown document types
            inferred_fields = ["content", "text", "data"]
        
        self.azure_field_mappings[doc_type] = inferred_fields
        logger.debug(f"Set up dynamic fields for {doc_type}: {inferred_fields}")
    
    def discover_azure_models(self) -> List[Dict[str, Any]]:
        """
        Discover all available Azure Document Intelligence models.
        
        Returns:
            List of model information dictionaries
        """
        if not self.azure_admin_client:
            logger.warning("Azure administration client not available")
            return []
        
        try:
            logger.info("🔍 Discovering available Azure Document Intelligence models...")
            
            # List all models
            models = list(self.azure_admin_client.list_models())
            
            model_info = []
            for model in models:
                info = {
                    'model_id': model.model_id,
                    'description': getattr(model, 'description', ''),
                    'created_date': getattr(model, 'created_date_time', None),
                    'status': getattr(model, 'status', 'unknown'),
                    'api_version': getattr(model, 'api_version', ''),
                    'tags': getattr(model, 'tags', {})
                }
                model_info.append(info)
            
            logger.info(f"📋 Discovered {len(model_info)} Azure models")
            return model_info
            
        except Exception as e:
            logger.error(f"❌ Failed to discover Azure models: {e}")
            return []
    
    def get_prebuilt_models(self) -> List[Dict[str, Any]]:
        """
        Get only prebuilt models from Azure.
        
        Returns:
            List of prebuilt model information
        """
        all_models = self.discover_azure_models()
        
        # Filter for prebuilt models (they typically start with 'prebuilt-')
        prebuilt_models = [
            model for model in all_models 
            if model['model_id'].startswith('prebuilt-')
        ]
        
        logger.info(f"📋 Found {len(prebuilt_models)} prebuilt models")
        return prebuilt_models
    
    def get_custom_models(self) -> List[Dict[str, Any]]:
        """
        Get only custom models from Azure.
        
        Returns:
            List of custom model information
        """
        all_models = self.discover_azure_models()
        
        # Filter for custom models (they typically don't start with 'prebuilt-')
        custom_models = [
            model for model in all_models 
            if not model['model_id'].startswith('prebuilt-')
        ]
        
        logger.info(f"📋 Found {len(custom_models)} custom models")
        return custom_models
    
    def configure_models_from_discovery(self, 
                                       include_prebuilt: bool = True, 
                                       include_custom: bool = True,
                                       model_filter: Optional[callable] = None) -> int:
        """
        Configure Azure models based on discovery results.
        
        Args:
            include_prebuilt: Whether to include prebuilt models
            include_custom: Whether to include custom models
            model_filter: Optional function to filter models (model_info -> bool)
            
        Returns:
            Number of models configured
        """
        logger.info("🔧 Configuring models from Azure discovery...")
        
        discovered_models = []
        
        if include_prebuilt:
            prebuilt = self.get_prebuilt_models()
            discovered_models.extend(prebuilt)
        
        if include_custom:
            custom = self.get_custom_models()
            discovered_models.extend(custom)
        
        # Apply filter if provided
        if model_filter:
            discovered_models = [m for m in discovered_models if model_filter(m)]
        
        # Convert to configuration format
        model_configs = []
        for model_info in discovered_models:
            model_id = model_info['model_id']
            
            # Generate document type from model ID
            if model_id.startswith('prebuilt-'):
                # Convert prebuilt model names to document types
                doc_type = self._prebuilt_model_to_doc_type(model_id)
            else:
                # Use custom model ID as document type (can be overridden)
                doc_type = model_id.replace('-', '_').lower()
            
            model_configs.append((model_id, doc_type))
        
        # Configure the models
        self.configure_azure_models(model_configs)
        
        logger.info(f"✅ Configured {len(model_configs)} models from discovery")
        return len(model_configs)
    
    def _prebuilt_model_to_doc_type(self, model_id: str) -> str:
        """
        Convert prebuilt model ID to document type.
        
        Args:
            model_id: Azure prebuilt model ID
            
        Returns:
            Corresponding document type string
        """
        # Dynamically convert model ID to document type
        # Remove 'prebuilt-' prefix and normalize the name
        if model_id.startswith('prebuilt-'):
            doc_type = model_id[9:]  # Remove 'prebuilt-' prefix
        else:
            doc_type = model_id
        
        # Normalize the document type name
        doc_type = doc_type.replace('.', '_').replace('-', '_').lower()
        
        return doc_type
    
    def print_available_models(self):
        """
        Print information about all available Azure models.
        """
        logger.info("📋 Available Azure Document Intelligence Models:")
        logger.info("=" * 60)
        
        prebuilt_models = self.get_prebuilt_models()
        custom_models = self.get_custom_models()
        
        if prebuilt_models:
            logger.info("\n🏭 PREBUILT MODELS:")
            for model in prebuilt_models:
                logger.info(f"  • {model['model_id']}")
                if model['description']:
                    logger.info(f"    Description: {model['description']}")
                logger.info(f"    Status: {model['status']}")
                if model['api_version']:
                    logger.info(f"    API Version: {model['api_version']}")
                logger.info()
        
        if custom_models:
            logger.info("\n🔧 CUSTOM MODELS:")
            for model in custom_models:
                logger.info(f"  • {model['model_id']}")
                if model['description']:
                    logger.info(f"    Description: {model['description']}")
                logger.info(f"    Status: {model['status']}")
                if model['created_date']:
                    logger.info(f"    Created: {model['created_date']}")
                if model['tags']:
                    logger.info(f"    Tags: {model['tags']}")
                logger.info()
        
        total_models = len(prebuilt_models) + len(custom_models)
        logger.info(f"\n📊 Total: {total_models} models ({len(prebuilt_models)} prebuilt, {len(custom_models)} custom)")
        logger.info("=" * 60)
    
    async def classify_with_azure_prebuilt(self, document_bytes: bytes) -> Optional[AzureClassificationResult]:
        """
        Classify document using Azure Document Intelligence prebuilt models.
        
        Args:
            document_bytes: Document as bytes
            
        Returns:
            Azure classification result or None if not confident enough
        """
        if not self.azure_client:
            return None
        
        # Use configured models (dynamically discovered or manually set)
        prebuilt_models = self.azure_prebuilt_models
        
        if not prebuilt_models:
            logger.warning("No Azure models configured. Use configure_models_from_discovery() or configure_azure_models()")
            return None
        
        best_result = None
        best_confidence = 0.0
        
        for model_id, doc_type in prebuilt_models:
            try:
                logger.debug(f"Testing Azure prebuilt model: {model_id}")
                
                # Analyze document with prebuilt model
                analyze_request = AnalyzeDocumentRequest(bytes_source=document_bytes)
                poller = self.azure_client.begin_analyze_document(
                    model_id=model_id,
                    body=analyze_request
                )
                result = poller.result()
                
                # Calculate confidence based on field detection and extraction quality
                confidence = self._calculate_prebuilt_confidence(result, doc_type)
                
                if confidence > best_confidence and confidence >= self.azure_confidence_threshold:
                    best_confidence = confidence
                    best_result = AzureClassificationResult(
                        document_type=doc_type,
                        confidence=confidence,
                        model_used=model_id,
                        fields_detected=self._extract_detected_fields(result),
                        raw_result=result
                    )
                
                logger.debug(f"Model {model_id} confidence: {confidence:.3f}")
                
            except Exception as e:
                if "ModelNotFound" in str(e) or "NotFound" in str(e):
                    logger.warning(f"⚠️ Azure model not available: {model_id} - {e}")
                    # Remove this model from future attempts
                    self.azure_prebuilt_models = [(mid, dt) for mid, dt in self.azure_prebuilt_models if mid != model_id]
                else:
                    logger.debug(f"Failed to analyze with {model_id}: {e}")
                continue
        
        return best_result
    
    def _calculate_prebuilt_confidence(self, result: AnalyzeResult, doc_type: str) -> float:
        """
        Calculate confidence score for prebuilt model result.
        
        Args:
            result: Azure analysis result
            doc_type: Document type being tested
            
        Returns:
            Confidence score between 0 and 1
        """
        if not result.documents or len(result.documents) == 0:
            return 0.0
        
        document = result.documents[0]
        if not document.fields:
            return 0.0
        
        #TODO:  Update this code with experiment to use AI to match the fields
        # Calculate confidence based on number and quality of detected fields
        field_confidences = []
        expected_fields = self._get_expected_fields_for_doc_type(doc_type)
        
        detected_expected_fields = 0
        for field_name, field_value in document.fields.items():
            if field_name in expected_fields:
                detected_expected_fields += 1
                if hasattr(field_value, 'confidence') and field_value.confidence:
                    field_confidences.append(field_value.confidence)
        
        if not field_confidences:
            return 0.0
        
        # Combine field detection rate and average field confidence
        detection_rate = detected_expected_fields / len(expected_fields) if expected_fields else 0
        avg_field_confidence = sum(field_confidences) / len(field_confidences)
        
        # Weighted combination
        overall_confidence = (detection_rate * 0.4) + (avg_field_confidence * 0.6)
        
        return min(overall_confidence, 1.0)
    
    def _get_expected_fields_for_doc_type(self, doc_type: str) -> List[str]:
        """Get expected fields for each document type from dynamic mapping."""
        return self.azure_field_mappings.get(doc_type, [])
    
    def _extract_detected_fields(self, result: AnalyzeResult) -> Dict[str, Any]:
        """Extract and format detected fields from Azure result."""
        detected_fields = {}
        
        if not result.documents or len(result.documents) == 0:
            return detected_fields
        
        document = result.documents[0]
        if not document.fields:
            return detected_fields
        
        for field_name, field_value in document.fields.items():
            field_info = {
                "content": getattr(field_value, 'content', str(field_value)) if field_value else None,
                "confidence": getattr(field_value, 'confidence', 0.0) if field_value else 0.0
            }
            detected_fields[field_name] = field_info
        
        return detected_fields
    
    async def classify_document(self, document_bytes: bytes, image: Optional[np.ndarray] = None, text: Optional[str] = None) -> ClassificationResult:
        """
        Classify document using Azure prebuilt models first, then fallback to traditional ML.
        
        Args:
            document_bytes: Document as bytes for Azure analysis
            image: Document image as numpy array for traditional ML
            text: Optional pre-extracted text
            
        Returns:
            Classification result
        """
        import time
        start_time = time.time()
        
        azure_result = None
        
        # Try Azure prebuilt models first
        if self.use_azure_prebuilt and self.azure_client:
            try:
                azure_result = await self.classify_with_azure_prebuilt(document_bytes)
                if azure_result and azure_result.confidence >= self.azure_confidence_threshold:
                    logger.info(f"Document classified by Azure as {azure_result.document_type} with confidence {azure_result.confidence:.3f}")
                    
                    processing_time = time.time() - start_time
                    return ClassificationResult(
                        document_type=azure_result.document_type,
                        confidence=azure_result.confidence,
                        probabilities={azure_result.document_type: azure_result.confidence},
                        features={"azure_fields": azure_result.fields_detected},
                        processing_time=processing_time,
                        model_used=f"Azure {azure_result.model_used}",
                        azure_analysis=azure_result.fields_detected
                    )
                else:
                    logger.debug("Azure prebuilt models did not provide confident classification")
            except Exception as e:
                logger.warning(f"Azure prebuilt classification failed: {e}")
        
        # Fallback to traditional ML classification
        if image is not None:
            logger.info("Falling back to traditional ML classification")
            traditional_result = self.classify(image, text)
            
            # Add Azure analysis to traditional result if available
            if azure_result:
                traditional_result.azure_analysis = azure_result.fields_detected
            
            return traditional_result
        else:
            # No image provided and Azure didn't work, return unknown
            processing_time = time.time() - start_time
            return ClassificationResult(
                document_type="unknown",
                confidence=0.0,
                probabilities={"unknown": 1.0},
                features={},
                processing_time=processing_time,
                model_used="none",
                azure_analysis=azure_result.fields_detected if azure_result else None
            )

# Import regex for text analysis
import re