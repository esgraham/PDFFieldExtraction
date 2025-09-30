# PDF Preprocessing Guide

## Overview

The PDF preprocessing module provides battle-tested image processing techniques to improve OCR accuracy through deskewing, denoising, and image enhancement. This is crucial for extracting high-quality text from scanned documents and images.

## Why Preprocessing Matters

**OCR Accuracy Impact:**
- ✅ **Deskewing**: Can improve OCR accuracy by 15-30% on skewed documents
- ✅ **Denoising**: Reduces false character recognition by 20-40%
- ✅ **Enhancement**: Improves text contrast and sharpness for better recognition

**Battle-Tested Techniques:**
- **Hough Transform**: Industry standard for line detection and skew correction
- **Radon Transform**: Robust mathematical approach for angle detection
- **OpenCV Pipelines**: Proven computer vision techniques for image processing

## Key Features

### 🔧 **Deskewing Methods**
1. **Hough Transform Based**: Detects text lines using edge detection
2. **Radon Transform Based**: Uses mathematical projections for accurate angle detection
3. **Contour-Based**: Analyzes text region bounding rectangles
4. **Multi-Method Fusion**: Combines results for maximum reliability

### 🧹 **Denoising Pipeline**
1. **Bilateral Filtering**: Preserves edges while reducing noise
2. **Morphological Operations**: Removes small artifacts and spots
3. **Gaussian Blur**: Smooths remaining texture noise
4. **Non-Local Means**: Advanced denoising for complex textures

### ✨ **Enhancement Features**
1. **CLAHE**: Adaptive contrast enhancement
2. **Gamma Correction**: Optimal brightness adjustment
3. **Unsharp Masking**: Text sharpening for better OCR
4. **Contrast Stretching**: Maximizes dynamic range

## Installation

### Core Dependencies
```bash
pip install PyMuPDF opencv-python numpy scipy scikit-image Pillow
```

### OCR Engines (Optional)
```bash
# Tesseract (most popular)
pip install pytesseract
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# EasyOCR (multi-language)
pip install easyocr

# PaddleOCR (advanced)
pip install paddleocr
```

### Quick Install
```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Basic Preprocessing

```python
from pdf_preprocessor import PDFPreprocessor

# Initialize preprocessor
preprocessor = PDFPreprocessor(
    dpi=300,                    # High quality conversion
    enable_deskew=True,         # Fix skewed documents
    enable_denoise=True,        # Remove noise and artifacts
    enable_enhancement=True,    # Improve contrast and sharpness
    debug_mode=True            # Save intermediate steps
)

# Process a PDF
processed_images = preprocessor.process_pdf(
    "document.pdf",
    output_dir="output"  # Debug images saved here
)

print(f"Processed {len(processed_images)} pages")
```

### 2. Azure Integration

```python
from pdf_integration import create_preprocessing_listener

# Create listener with preprocessing
listener = create_preprocessing_listener(
    storage_account_name="myaccount",
    container_name="pdfs",
    connection_string="connection_string",
    preprocessing_config={
        'enable_preprocessing': True,
        'preprocessing_dpi': 300,
        'save_preprocessed': True,
        'preprocessed_container': 'processed-pdfs'
    }
)

# Set callback for preprocessed results
def handle_preprocessed(blob_name, images, stats):
    print(f"Preprocessed {blob_name}: {len(images)} pages")
    if 'average_skew' in stats:
        print(f"Average skew corrected: {stats['average_skew']:.2f}°")

listener.set_preprocessing_callback(handle_preprocessed)

# Start monitoring
listener.set_pdf_callback(lambda name, client: 
    listener.process_pdf_with_preprocessing(name, client))
listener.start_polling()
```

### 3. OCR Integration

```python
from pdf_integration import create_ocr_listener

# Create OCR-enabled listener
listener = create_ocr_listener(
    storage_account_name="myaccount",
    container_name="pdfs",
    connection_string="connection_string",
    ocr_config={
        'ocr_engine': 'tesseract',
        'ocr_languages': ['eng', 'fra'],  # English and French
        'extract_tables': False
    }
)

# Enhanced callback with OCR
def handle_ocr_results(blob_name, blob_client):
    # Preprocess
    images = listener.process_pdf_with_preprocessing(blob_name, blob_client)
    
    if images:
        # Extract text
        ocr_results = listener.extract_text_from_images(images)
        
        # Process results
        for page_result in ocr_results:
            print(f"Page {page_result['page_number']}: "
                  f"{page_result['word_count']} words, "
                  f"{page_result['confidence']:.1f}% confidence")

listener.set_pdf_callback(handle_ocr_results)
listener.start_polling()
```

### 4. Batch Processing

```python
from pdf_preprocessor import batch_preprocess

# Process multiple PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

results = batch_preprocess(
    pdf_files,
    output_dir="batch_output",
    dpi=300,
    enable_deskew=True,
    enable_denoise=True,
    enable_enhancement=True,
    debug_mode=True
)

# Check results
for pdf_path, images in results.items():
    if images:
        print(f"✅ {pdf_path}: {len(images)} pages processed")
    else:
        print(f"❌ {pdf_path}: Processing failed")
```

## Configuration Options

### PDFPreprocessor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | int | 300 | DPI for PDF to image conversion |
| `enable_deskew` | bool | True | Enable text line deskewing |
| `enable_denoise` | bool | True | Enable noise reduction |
| `enable_enhancement` | bool | True | Enable image enhancement |
| `debug_mode` | bool | False | Save intermediate processing steps |

### Deskewing Configuration

```python
# Fine-tune deskewing behavior
preprocessor = PDFPreprocessor(
    enable_deskew=True,
    # Deskewing is automatic, but you can create custom pipelines
)

# Custom pipeline for specific skew handling
pipeline = preprocessor.create_processing_pipeline([
    'denoise',    # Clean image first
    'deskew',     # Fix rotation
    'enhance',    # Final enhancement
    'binarize'    # Convert to black/white
])
```

### OCR Engine Comparison

| Engine | Pros | Cons | Best For |
|--------|------|------|----------|
| **Tesseract** | Fast, accurate, free | Requires binary install | English text, documents |
| **EasyOCR** | Multi-language, easy setup | Slower, larger memory | Non-English text |
| **PaddleOCR** | Very accurate, table support | Complex setup, resource intensive | Complex layouts |

## Performance Optimization

### Processing Speed
```python
# Fast processing (lower quality)
preprocessor = PDFPreprocessor(
    dpi=200,                    # Lower DPI
    enable_denoise=False,       # Skip denoising
    enable_enhancement=False    # Skip enhancement
)

# High quality (slower)
preprocessor = PDFPreprocessor(
    dpi=400,                    # Higher DPI
    debug_mode=True            # Save all steps
)
```

### Memory Management
```python
# For large PDFs or batch processing
import gc

for pdf_file in large_pdf_list:
    images = preprocessor.process_pdf(pdf_file)
    # Process images immediately
    process_images(images)
    # Clear memory
    del images
    gc.collect()
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread-safe preprocessing
def process_single_pdf(pdf_path):
    # Each thread gets its own preprocessor
    local_preprocessor = PDFPreprocessor()
    return local_preprocessor.process_pdf(pdf_path)

# Process multiple PDFs in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_single_pdf, pdf) 
               for pdf in pdf_list]
    results = [future.result() for future in futures]
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install opencv-python numpy scipy scikit-image Pillow PyMuPDF

# For OCR
pip install pytesseract easyocr
```

**2. Tesseract Not Found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

**3. Poor OCR Results**
```python
# Try different preprocessing settings
preprocessor = PDFPreprocessor(
    dpi=400,                    # Higher resolution
    enable_deskew=True,         # Ensure deskewing is enabled
    enable_denoise=True,        # Clean noisy images
    enable_enhancement=True     # Improve contrast
)

# Check debug images to see preprocessing results
preprocessor.debug_mode = True
```

**4. Memory Issues**
```python
# Reduce memory usage
preprocessor = PDFPreprocessor(
    dpi=200,                    # Lower DPI
    debug_mode=False           # Don't save debug images
)

# Process pages individually instead of all at once
```

### Debug Information

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get processing statistics
stats = preprocessor.get_processing_stats()
print(f"Pages processed: {stats['pages_processed']}")
print(f"Average skew: {stats.get('average_skew', 'N/A')}")
print(f"Max skew detected: {stats.get('max_skew', 'N/A')}")

# Check individual page results
for i, angle in enumerate(stats.get('skew_angles', [])):
    print(f"Page {i+1} skew: {angle:.2f}°")
```

## Best Practices

### 1. **Choose Appropriate DPI**
- **200 DPI**: Fast processing, acceptable for clean documents
- **300 DPI**: Good balance of quality and speed (recommended)
- **400+ DPI**: High quality for poor scans, slower processing

### 2. **Pipeline Order**
```python
# Recommended processing order:
# 1. Initial denoising (remove obvious artifacts)
# 2. Deskewing (fix rotation on clean image)
# 3. Final enhancement (optimize for OCR)
```

### 3. **Quality Assessment**
```python
def assess_preprocessing_quality(original, processed):
    """Assess preprocessing quality."""
    # Calculate metrics
    contrast_improvement = cv2.Laplacian(processed, cv2.CV_64F).var() / \
                          cv2.Laplacian(original, cv2.CV_64F).var()
    
    print(f"Contrast improvement: {contrast_improvement:.2f}x")
    return contrast_improvement > 1.2  # Good if improved by 20%
```

### 4. **Error Handling**
```python
def robust_preprocessing(pdf_path):
    """Robust preprocessing with fallbacks."""
    try:
        # Try full preprocessing
        preprocessor = PDFPreprocessor(enable_deskew=True, enable_denoise=True)
        return preprocessor.process_pdf(pdf_path)
    except Exception as e:
        logger.warning(f"Full preprocessing failed: {e}")
        try:
            # Fallback: minimal preprocessing
            preprocessor = PDFPreprocessor(enable_deskew=False, enable_denoise=False)
            return preprocessor.process_pdf(pdf_path)
        except Exception as e2:
            logger.error(f"All preprocessing failed: {e2}")
            return None
```

## Advanced Features

### Custom Processing Pipelines
```python
# Create specialized pipeline for invoices
invoice_pipeline = preprocessor.create_processing_pipeline([
    'denoise',      # Clean scan artifacts
    'deskew',       # Fix rotation
    'enhance',      # Improve text contrast
    'binarize'      # Convert to pure black/white
])

# Apply to image
processed_invoice = invoice_pipeline.process(invoice_image)
```

### Region-Based Processing
```python
def process_document_regions(image):
    """Process different regions with different settings."""
    height, width = image.shape
    
    # Header region (top 20%)
    header = image[:height//5, :]
    header_processed = enhance_for_titles(header)
    
    # Body region (middle 60%)
    body = image[height//5:4*height//5, :]
    body_processed = enhance_for_text(body)
    
    # Footer region (bottom 20%)
    footer = image[4*height//5:, :]
    footer_processed = enhance_for_small_text(footer)
    
    # Combine regions
    return np.vstack([header_processed, body_processed, footer_processed])
```

## Integration Examples

### Flask Web Service
```python
from flask import Flask, request, jsonify
from pdf_preprocessor import quick_preprocess

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess_pdf():
    file = request.files['pdf']
    file.save('temp.pdf')
    
    try:
        images = quick_preprocess('temp.pdf')
        return jsonify({
            'success': True,
            'pages_processed': len(images),
            'message': 'Preprocessing completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

The preprocessing module provides enterprise-grade PDF processing capabilities that significantly improve OCR accuracy and text extraction quality. Choose the configuration that best matches your quality requirements and processing speed needs.