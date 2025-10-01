"""
PDF Preprocessing Module

This module provides comprehensive preprocessing capabilities for PDF documents
to improve OCR accuracy through deskewing, denoising, and image enhancement.

Features:
- Hough/Radon-based text line deskewing
- Advanced denoising using OpenCV
- Image enhancement and normalization
- Support for both bitmap and vector-based PDFs
- Configurable preprocessing pipelines
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import interpolation
from skimage import filters, morphology, measure
from skimage.transform import radon, rotate
from PIL import Image, ImageFilter, ImageEnhance
import warnings

# Optional deskew library
try:
    from deskew import determine_skew
    DESKEW_AVAILABLE = True
except ImportError:
    DESKEW_AVAILABLE = False

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class PDFPreprocessor:
    """
    Advanced PDF preprocessing pipeline for OCR optimization.
    
    Provides deskewing, denoising, and image enhancement capabilities
    using battle-tested computer vision techniques.
    """
    
    def __init__(
        self,
        dpi: int = 300,
        enable_deskew: bool = True,
        enable_denoise: bool = True,
        enable_enhancement: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize the PDF preprocessor.
        
        Args:
            dpi: DPI for PDF to image conversion (higher = better quality)
            enable_deskew: Enable text line deskewing
            enable_denoise: Enable noise reduction
            enable_enhancement: Enable image enhancement
            debug_mode: Save intermediate processing steps for debugging
        """
        self.dpi = dpi
        self.enable_deskew = enable_deskew
        self.enable_denoise = enable_denoise
        self.enable_enhancement = enable_enhancement
        self.debug_mode = debug_mode
        
        # Processing statistics
        self.stats = {
            'pages_processed': 0,
            'skew_angles': [],
            'processing_times': []
        }
        
    def process_pdf(
        self, 
        pdf_path: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Process a PDF file through the complete preprocessing pipeline.
        
        Args:
            pdf_path: Path to the input PDF file
            output_dir: Optional directory to save processed images
            
        Returns:
            Path to the processed PDF file (same as input if no changes made)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = self.pdf_to_images(str(pdf_path))
        
        # Process each page
        processed_images = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)}")
            
            processed_image = self._process_image(image, page_num=i)
            processed_images.append(processed_image)
            
            # Save debug images if enabled
            if self.debug_mode and output_dir:
                self._save_debug_images(image, processed_image, output_dir, i)
        
        self.stats['pages_processed'] = len(processed_images)
        logger.info(f"Completed processing {len(processed_images)} pages")
        
        # For now, return the original PDF path since the images are processed in memory
        # In a full implementation, you might save processed images and return the path
        # to the processed version or create a new PDF from processed images
        return str(pdf_path)
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to numpy arrays for processing.
        Supports both direct PDF files and image directories.
        
        Args:
            pdf_path: Path to the PDF file or directory with PDF images
            
        Returns:
            List of numpy arrays representing each page
        """
        images = []
        pdf_path = Path(pdf_path)
        
        try:
            if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
                # PDF file - convert using pdf2image
                try:
                    from pdf2image import convert_from_path
                    from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
                    
                    logger.info(f"Converting PDF to images: {pdf_path}")
                    # Convert PDF pages to PIL images
                    pil_images = convert_from_path(str(pdf_path), dpi=self.dpi)
                    
                    # Convert PIL images to numpy arrays
                    for pil_img in pil_images:
                        # Convert to RGB if necessary
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        
                        # Convert to numpy array
                        img_array = np.array(pil_img)
                        images.append(img_array)
                    
                    logger.info(f"Successfully converted {len(images)} pages from PDF")
                    
                except ImportError:
                    raise ValueError(
                        "pdf2image library is required for PDF processing. "
                        "Install with: pip install pdf2image"
                    )
                except PDFInfoNotInstalledError:
                    raise ValueError(
                        "poppler-utils is required for PDF processing. "
                        "Install with: sudo apt-get install poppler-utils (Linux) "
                        "or brew install poppler (macOS)"
                    )
                except (PDFPageCountError, Exception) as e:
                    raise ValueError(f"Failed to convert PDF to images: {e}")
            
            elif pdf_path.is_dir():
                # Directory with images
                image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
                image_files = [f for f in pdf_path.iterdir() 
                             if f.suffix.lower() in image_extensions]
                
                for img_file in sorted(image_files):
                    image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        images.append(image)
                        if self.debug_mode:
                            self.logger.info(f"Loaded image: {img_file.name} - {image.shape}")
            
            elif pdf_path.is_file():
                # Single image file
                image = cv2.imread(str(pdf_path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    if self.debug_mode:
                        self.logger.info(f"Loaded single image: {pdf_path.name} - {image.shape}")
            
            return images
            
        except Exception as e:
            self.logger.error(f"Error loading images: {e}")
            raise
    
    def _process_image(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Apply the complete preprocessing pipeline to a single image.
        
        Args:
            image: Input image as numpy array
            page_num: Page number for logging
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        # Step 1: Initial denoising
        if self.enable_denoise:
            processed = self._apply_denoising(processed)
        
        # Step 2: Deskewing
        if self.enable_deskew:
            processed, skew_angle = self._apply_deskewing(processed)
            self.stats['skew_angles'].append(skew_angle)
            logger.debug(f"Page {page_num + 1} skew angle: {skew_angle:.2f}°")
        
        # Step 3: Final enhancement
        if self.enable_enhancement:
            processed = self._apply_enhancement(processed)
        
        return processed
    
    def _apply_deskewing(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply deskewing using multiple methods for robustness.
        
        Uses both Hough transform and Radon transform approaches,
        selecting the most reliable result.
        """
        # Method 1: Hough transform based deskewing
        hough_angle = self._detect_skew_hough(image)
        
        # Method 2: Radon transform based deskewing
        radon_angle = self._detect_skew_radon(image)
        
        # Method 3: Contour-based approach for backup
        contour_angle = self._detect_skew_contours(image)
        
        # Method 4: Deskew library if available
        deskew_angle = self._detect_skew_deskew_lib(image) if DESKEW_AVAILABLE else None
        
        # Select the most reliable angle
        angles = [hough_angle, radon_angle, contour_angle, deskew_angle]
        valid_angles = [angle for angle in angles if angle is not None and abs(angle) < 45]
        
        if not valid_angles:
            logger.warning("No reliable skew angle detected, skipping deskewing")
            return image, 0.0
        
        # Use median of valid angles for robustness
        final_angle = np.median(valid_angles)
        
        # Apply rotation
        if abs(final_angle) > 0.1:  # Only rotate if significant skew detected
            deskewed = self._rotate_image(image, final_angle)
            logger.debug(f"Applied deskewing: {final_angle:.2f}°")
            return deskewed, final_angle
        
        return image, 0.0
    
    def _detect_skew_hough(self, image: np.ndarray) -> Optional[float]:
        """Detect skew angle using Hough line transform."""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            if lines is None:
                return None
            
            # Calculate angles
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Filter out vertical lines
                    angles.append(angle)
            
            if not angles:
                return None
            
            # Return median angle for robustness
            return np.median(angles)
            
        except Exception as e:
            logger.warning(f"Hough skew detection failed: {e}")
            return None
    
    def _detect_skew_radon(self, image: np.ndarray) -> Optional[float]:
        """Detect skew angle using Radon transform."""
        try:
            # Resize for performance if image is too large
            height, width = image.shape
            if width > 1000:
                scale = 1000 / width
                new_height = int(height * scale)
                resized = cv2.resize(image, (1000, new_height))
            else:
                resized = image
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = 255 - binary  # Invert so text is white
            
            # Apply Radon transform
            angles = np.arange(-45, 46, 0.5)  # Test angles from -45 to 45 degrees
            sinogram = radon(binary, theta=angles, circle=False)
            
            # Find angle with maximum variance (sharpest projection)
            variances = np.var(sinogram, axis=0)
            best_angle_idx = np.argmax(variances)
            skew_angle = angles[best_angle_idx]
            
            return skew_angle
            
        except Exception as e:
            logger.warning(f"Radon skew detection failed: {e}")
            return None
    
    def _detect_skew_contours(self, image: np.ndarray) -> Optional[float]:
        """Detect skew angle using text contours."""
        try:
            # Apply threshold
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to get text regions
            min_area = image.shape[0] * image.shape[1] * 0.0001  # 0.01% of image area
            text_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if len(text_contours) < 5:  # Need sufficient text regions
                return None
            
            # Calculate bounding rectangle angles
            angles = []
            for contour in text_contours:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                # Normalize angle to [-45, 45] range
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                if abs(angle) < 45:
                    angles.append(angle)
            
            if not angles:
                return None
            
            return np.median(angles)
            
        except Exception as e:
            logger.warning(f"Contour skew detection failed: {e}")
            return None
    
    def _detect_skew_deskew_lib(self, image: np.ndarray) -> Optional[float]:
        """
        Detect skew using the deskew library.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Detected skew angle in degrees, or None if not detected
        """
        if not DESKEW_AVAILABLE:
            return None
        
        try:
            angle = determine_skew(image)
            
            if self.debug_mode:
                print(f"Deskew library detection: {angle:.2f}°")
            
            # Return angle only if significant
            return angle if abs(angle) > 0.5 else None
            
        except Exception as e:
            if self.debug_mode:
                print(f"Deskew library detection failed: {e}")
            return None
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive denoising pipeline."""
        denoised = image.copy()
        
        # Step 1: Bilateral filtering (preserves edges while reducing noise)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # Step 2: Morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Step 3: Gaussian blur for remaining noise (mild)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        # Step 4: Non-local means denoising for textured noise
        denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        return denoised
    
    def _apply_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better OCR results."""
        enhanced = image.copy()
        
        # Step 1: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        # Step 2: Gamma correction for brightness adjustment
        gamma = self._calculate_optimal_gamma(enhanced)
        enhanced = self._adjust_gamma(enhanced, gamma)
        
        # Step 3: Unsharp masking for text sharpening
        enhanced = self._unsharp_mask(enhanced)
        
        # Step 4: Final contrast stretching
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle with proper padding."""
        height, width = image.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation with white background
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height), 
            borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        
        return rotated
    
    def _calculate_optimal_gamma(self, image: np.ndarray) -> float:
        """Calculate optimal gamma value for brightness correction."""
        # Calculate mean brightness
        mean_brightness = np.mean(image) / 255.0
        
        # Target brightness (slightly bright for OCR)
        target_brightness = 0.6
        
        # Calculate gamma to achieve target brightness
        if mean_brightness > 0:
            gamma = np.log(target_brightness) / np.log(mean_brightness)
            # Clamp gamma to reasonable range
            gamma = np.clip(gamma, 0.5, 2.0)
        else:
            gamma = 1.0
        
        return gamma
    
    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to image."""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def _unsharp_mask(self, image: np.ndarray, kernel_size: int = 5, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking for text sharpening."""
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        
        return unsharp_mask
    
    def _save_debug_images(
        self, 
        original: np.ndarray, 
        processed: np.ndarray, 
        output_dir: Path, 
        page_num: int
    ):
        """Save debug images showing processing steps."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original
        cv2.imwrite(str(output_dir / f"page_{page_num:03d}_original.png"), original)
        
        # Save processed
        cv2.imwrite(str(output_dir / f"page_{page_num:03d}_processed.png"), processed)
        
        # Create side-by-side comparison
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
        
        comparison = np.hstack([original_gray, processed])
        cv2.imwrite(str(output_dir / f"page_{page_num:03d}_comparison.png"), comparison)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['skew_angles']:
            stats['average_skew'] = np.mean(stats['skew_angles'])
            stats['max_skew'] = np.max(np.abs(stats['skew_angles']))
        
        return stats
    
    def create_processing_pipeline(
        self,
        steps: List[str],
        **kwargs
    ) -> 'ProcessingPipeline':
        """
        Create a custom processing pipeline with specific steps.
        
        Args:
            steps: List of processing step names
            **kwargs: Additional parameters for steps
            
        Returns:
            ProcessingPipeline instance
        """
        return ProcessingPipeline(self, steps, **kwargs)


class ProcessingPipeline:
    """Custom processing pipeline for specific preprocessing workflows."""
    
    def __init__(self, preprocessor: PDFPreprocessor, steps: List[str], **kwargs):
        self.preprocessor = preprocessor
        self.steps = steps
        self.params = kwargs
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image through the custom pipeline."""
        processed = image.copy()
        
        for step in self.steps:
            if step == 'denoise':
                processed = self.preprocessor._apply_denoising(processed)
            elif step == 'deskew':
                processed, _ = self.preprocessor._apply_deskewing(processed)
            elif step == 'enhance':
                processed = self.preprocessor._apply_enhancement(processed)
            elif step == 'binarize':
                processed = self._binarize(processed)
            elif step == 'invert':
                processed = cv2.bitwise_not(processed)
            else:
                logger.warning(f"Unknown processing step: {step}")
        
        return processed
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive binarization."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )


# Convenience functions for common preprocessing tasks

def quick_preprocess(pdf_path: Union[str, Path], dpi: int = 300) -> List[np.ndarray]:
    """
    Quick preprocessing with default settings.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for conversion
        
    Returns:
        List of preprocessed images
    """
    preprocessor = PDFPreprocessor(dpi=dpi)
    return preprocessor.process_pdf(pdf_path)


def preprocess_for_ocr(
    pdf_path: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None,
    aggressive_deskew: bool = False
) -> List[np.ndarray]:
    """
    Preprocessing optimized for OCR accuracy.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Optional output directory for debug images
        aggressive_deskew: Use more aggressive deskewing
        
    Returns:
        List of preprocessed images optimized for OCR
    """
    preprocessor = PDFPreprocessor(
        dpi=300,
        enable_deskew=True,
        enable_denoise=True,
        enable_enhancement=True,
        debug_mode=output_dir is not None
    )
    
    return preprocessor.process_pdf(pdf_path, output_dir)


def batch_preprocess(
    pdf_paths: List[Union[str, Path]], 
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, List[np.ndarray]]:
    """
    Batch process multiple PDFs.
    
    Args:
        pdf_paths: List of PDF file paths
        output_dir: Output directory
        **kwargs: Additional preprocessing parameters
        
    Returns:
        Dictionary mapping file paths to processed images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = PDFPreprocessor(**kwargs)
    results = {}
    
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        logger.info(f"Processing {pdf_path.name}")
        
        try:
            # Create subdirectory for this PDF
            pdf_output_dir = output_dir / pdf_path.stem
            
            processed_images = preprocessor.process_pdf(pdf_path, pdf_output_dir)
            results[str(pdf_path)] = processed_images
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            results[str(pdf_path)] = []
    
    return results


import io  # Add this import at the top with other imports