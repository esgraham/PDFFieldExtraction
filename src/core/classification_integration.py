"""
Document Classification Integration

This module integrates document classification with the existing PDF processing
and Azure Storage monitoring system.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Union
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from azure_pdf_listener import AzurePDFListener
from pdf_preprocessor import PDFPreprocessor
from document_classifier import DocumentClassifier, DocumentClass, ClassificationResult
from pdf_integration import PreprocessingPDFListener

logger = logging.getLogger(__name__)

class ClassificationIntegratedListener(PreprocessingPDFListener):
    """
    PDF Listener with integrated preprocessing and document classification.
    
    Extends the preprocessing listener to add automatic document classification
    capabilities for incoming PDF files.
    """
    
    def __init__(
        self,
        connection_string: str,
        container_name: str,
        classifier_config: Optional[Dict[str, Any]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        auto_train: bool = False,
        classification_callback: Optional[Callable[[str, ClassificationResult], None]] = None
    ):
        """
        Initialize the classification-integrated listener.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to monitor
            classifier_config: Configuration for document classifier
            preprocessing_config: Configuration for PDF preprocessing
            model_path: Path to pre-trained classification model
            auto_train: Whether to automatically train classifier from incoming documents
            classification_callback: Callback function for classification results
        """
        super().__init__(connection_string, container_name, preprocessing_config)
        
        # Initialize document classifier
        self.classifier_config = classifier_config or self._get_default_classifier_config()
        self.classifier = DocumentClassifier(**self.classifier_config)
        
        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self.classifier.load_model(model_path)
            logger.info(f"Loaded pre-trained classification model from {model_path}")
        
        # Auto-training setup
        self.auto_train = auto_train
        self.training_data = {"images": [], "labels": [], "texts": []}
        self.min_training_samples = 10  # Minimum samples per class for training
        
        # Classification callback
        self.classification_callback = classification_callback
        
        # Statistics
        self.classification_stats = {
            "total_classified": 0,
            "class_counts": {cls.value: 0 for cls in DocumentClass},
            "avg_confidence": 0.0,
            "processing_times": []
        }
        
        logger.info("ClassificationIntegratedListener initialized")
    
    def _get_default_classifier_config(self) -> Dict[str, Any]:
        """Get default configuration for document classifier."""
        return {
            "model_type": "random_forest",
            "use_transformers": True,
            "transformer_model": "all-MiniLM-L6-v2",
            "enable_ocr": True,
            "cache_features": True,
            "debug_mode": False
        }
    
    def set_classification_callback(self, callback: Callable[[str, ClassificationResult], None]):
        """Set the callback function for classification results."""
        self.classification_callback = callback
        logger.info("Classification callback set")
    
    def process_pdf_with_classification(
        self, 
        blob_name: str, 
        blob_data: bytes,
        expected_class: Optional[DocumentClass] = None
    ) -> Dict[str, Any]:
        """
        Process PDF with preprocessing and classification.
        
        Args:
            blob_name: Name of the PDF blob
            blob_data: PDF file content as bytes
            expected_class: Expected document class (for training)
            
        Returns:
            Dictionary containing processing and classification results
        """
        logger.info(f"Processing PDF with classification: {blob_name}")
        
        try:
            # First, perform preprocessing
            preprocessing_result = self.process_pdf_with_preprocessing(blob_name, blob_data)
            
            if not preprocessing_result.get("success", False):
                return {
                    "success": False,
                    "error": "Preprocessing failed",
                    "preprocessing_result": preprocessing_result
                }
            
            processed_images = preprocessing_result.get("processed_images", [])
            extracted_texts = preprocessing_result.get("extracted_texts", [])
            
            if not processed_images:
                return {
                    "success": False,
                    "error": "No images extracted from PDF",
                    "preprocessing_result": preprocessing_result
                }
            
            # Classify each page (for multi-page PDFs, we'll classify the first page primarily)
            classification_results = []
            
            for i, (image, text) in enumerate(zip(processed_images, extracted_texts)):
                try:
                    if self.classifier.is_trained:
                        result = self.classifier.classify(image, text)
                        classification_results.append({
                            "page": i + 1,
                            "result": result
                        })
                        
                        # Update statistics
                        self._update_classification_stats(result)
                        
                        logger.info(f"Page {i+1} classified as {result.predicted_class.value} "
                                  f"with confidence {result.confidence:.2f}")
                    
                    # Store data for auto-training if enabled
                    if self.auto_train and expected_class:
                        self.training_data["images"].append(image)
                        self.training_data["labels"].append(expected_class)
                        self.training_data["texts"].append(text)
                        
                        # Check if we have enough data to retrain
                        self._check_and_retrain()
                
                except Exception as e:
                    logger.error(f"Classification failed for page {i+1}: {e}")
                    classification_results.append({
                        "page": i + 1,
                        "error": str(e)
                    })
            
            # Determine overall document class (majority vote for multi-page docs)
            overall_class = self._determine_overall_class(classification_results)
            
            result = {
                "success": True,
                "blob_name": blob_name,
                "timestamp": datetime.now().isoformat(),
                "preprocessing_result": preprocessing_result,
                "classification_results": classification_results,
                "overall_class": overall_class.value if overall_class else None,
                "page_count": len(processed_images)
            }
            
            # Call classification callback if set
            if self.classification_callback and overall_class:
                try:
                    # Create a summary classification result for the callback
                    summary_result = ClassificationResult(
                        predicted_class=overall_class,
                        confidence=self._calculate_overall_confidence(classification_results),
                        probabilities=self._aggregate_probabilities(classification_results),
                        features_used={"pages_processed": len(classification_results)},
                        processing_time=sum(r.get("result", {}).processing_time or 0 
                                          for r in classification_results if "result" in r)
                    )
                    self.classification_callback(blob_name, summary_result)
                except Exception as e:
                    logger.error(f"Classification callback failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF processing with classification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "blob_name": blob_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_classification_stats(self, result: ClassificationResult):
        """Update classification statistics."""
        self.classification_stats["total_classified"] += 1
        self.classification_stats["class_counts"][result.predicted_class.value] += 1
        
        # Update average confidence
        total = self.classification_stats["total_classified"]
        current_avg = self.classification_stats["avg_confidence"]
        self.classification_stats["avg_confidence"] = (
            (current_avg * (total - 1) + result.confidence) / total
        )
        
        self.classification_stats["processing_times"].append(result.processing_time)
    
    def _determine_overall_class(self, classification_results: List[Dict]) -> Optional[DocumentClass]:
        """Determine overall document class from page-level results."""
        if not classification_results:
            return None
        
        # Extract valid classification results
        valid_results = [r["result"] for r in classification_results if "result" in r]
        
        if not valid_results:
            return None
        
        # For single page, return the result
        if len(valid_results) == 1:
            return valid_results[0].predicted_class
        
        # For multiple pages, use majority vote weighted by confidence
        class_votes = {}
        for result in valid_results:
            cls = result.predicted_class
            weight = result.confidence
            class_votes[cls] = class_votes.get(cls, 0) + weight
        
        # Return class with highest weighted vote
        return max(class_votes.items(), key=lambda x: x[1])[0]
    
    def _calculate_overall_confidence(self, classification_results: List[Dict]) -> float:
        """Calculate overall confidence from page-level results."""
        valid_results = [r["result"] for r in classification_results if "result" in r]
        
        if not valid_results:
            return 0.0
        
        return sum(r.confidence for r in valid_results) / len(valid_results)
    
    def _aggregate_probabilities(self, classification_results: List[Dict]) -> Dict[DocumentClass, float]:
        """Aggregate probabilities from page-level results."""
        valid_results = [r["result"] for r in classification_results if "result" in r]
        
        if not valid_results:
            return {}
        
        # Average probabilities across pages
        aggregated_probs = {}
        for cls in DocumentClass:
            probs = [r.probabilities.get(cls, 0.0) for r in valid_results]
            aggregated_probs[cls] = sum(probs) / len(probs)
        
        return aggregated_probs
    
    def _check_and_retrain(self):
        """Check if we have enough data to retrain the classifier."""
        if not self.auto_train:
            return
        
        # Count samples per class
        class_counts = {}
        for label in self.training_data["labels"]:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Check if we have minimum samples for each class
        min_samples_met = all(count >= self.min_training_samples 
                            for count in class_counts.values())
        
        if min_samples_met and len(class_counts) >= 2:  # At least 2 classes
            logger.info("Sufficient training data available, retraining classifier...")
            try:
                self.classifier.train(
                    self.training_data["images"],
                    self.training_data["labels"],
                    self.training_data["texts"]
                )
                logger.info("Classifier retrained successfully")
                
                # Clear training data to avoid memory issues
                self.training_data = {"images": [], "labels": [], "texts": []}
                
            except Exception as e:
                logger.error(f"Auto-retraining failed: {e}")
    
    def train_classifier(
        self, 
        training_data: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the document classifier with provided data.
        
        Args:
            training_data: List of training samples with 'image', 'label', and optional 'text'
            save_path: Optional path to save the trained model
            
        Returns:
            Training results and statistics
        """
        logger.info(f"Training classifier with {len(training_data)} samples")
        
        images = []
        labels = []
        texts = []
        
        for sample in training_data:
            images.append(sample["image"])
            labels.append(DocumentClass(sample["label"]))
            texts.append(sample.get("text", ""))
        
        try:
            # Train the classifier
            self.classifier.train(images, labels, texts)
            
            # Save model if path provided
            if save_path:
                self.classifier.save_model(save_path)
                logger.info(f"Trained model saved to {save_path}")
            
            # Calculate training statistics
            label_counts = {}
            for label in labels:
                label_counts[label.value] = label_counts.get(label.value, 0) + 1
            
            return {
                "success": True,
                "samples_trained": len(training_data),
                "class_distribution": label_counts,
                "model_type": self.classifier.model_type,
                "features_extracted": len(self.classifier.feature_names) if self.classifier.feature_names else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Classifier training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        stats = self.classification_stats.copy()
        
        if stats["processing_times"]:
            stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
            stats["max_processing_time"] = max(stats["processing_times"])
            stats["min_processing_time"] = min(stats["processing_times"])
        
        stats["classifier_trained"] = self.classifier.is_trained
        stats["auto_training_enabled"] = self.auto_train
        
        if self.auto_train:
            stats["training_data_collected"] = {
                "images": len(self.training_data["images"]),
                "labels": len(self.training_data["labels"]),
                "texts": len(self.training_data["texts"])
            }
        
        return stats
    
    def classify_existing_document(
        self, 
        image_path: str, 
        text: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify an existing document image.
        
        Args:
            image_path: Path to document image
            text: Optional pre-extracted text
            
        Returns:
            Classification result
        """
        if not self.classifier.is_trained:
            raise ValueError("Classifier must be trained before classification")
        
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.classifier.classify(image, text)

def create_classification_pipeline(
    connection_string: str,
    container_name: str,
    model_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> ClassificationIntegratedListener:
    """
    Create a complete document classification pipeline.
    
    Args:
        connection_string: Azure Storage connection string
        container_name: Container to monitor
        model_config: Classification model configuration
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Configured classification pipeline
    """
    # Default configurations
    default_model_config = {
        "model_type": "random_forest",
        "use_transformers": True,
        "enable_ocr": True,
        "cache_features": True
    }
    
    default_preprocessing_config = {
        "dpi": 300,
        "enable_deskew": True,
        "enable_denoise": True,
        "enable_enhancement": True
    }
    
    # Merge with provided configs
    final_model_config = {**default_model_config, **(model_config or {})}
    final_preprocessing_config = {**default_preprocessing_config, **(preprocessing_config or {})}
    
    return ClassificationIntegratedListener(
        connection_string=connection_string,
        container_name=container_name,
        classifier_config=final_model_config,
        preprocessing_config=final_preprocessing_config
    )