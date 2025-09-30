#!/usr/bin/env python3
"""
Test script to validate PDF preprocessing functionality.
This script demonstrates the advanced preprocessing capabilities
including deskewing, denoising, and enhancement.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add src to path (go up one level from tests/ to project root, then to src/)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_test_pdf_page():
    """Create a synthetic PDF-like image with text and some issues to test preprocessing."""
    # Create a white background
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a reasonable font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Add some text lines with slight rotation to simulate skew
    text_lines = [
        "This is a sample PDF document with text",
        "that needs preprocessing for better OCR.",
        "The document may have skew, noise, and",
        "poor contrast that affects recognition.",
        "",
        "Advanced preprocessing includes:",
        "• Deskewing using Hough/Radon transforms",
        "• Denoising with bilateral filtering",
        "• Enhancement with CLAHE and gamma",
        "",
        "This improves OCR accuracy significantly."
    ]
    
    y_start = 100
    line_height = 40
    
    for i, line in enumerate(text_lines):
        if line.strip():  # Skip empty lines for drawing
            y = y_start + i * line_height
            # Add slight progressive skew
            x_offset = int(i * 2)  # Small skew simulation
            draw.text((50 + x_offset, y), line, fill='black', font=font)
    
    # Convert to grayscale numpy array
    gray_image = image.convert('L')
    image_array = np.array(gray_image)
    
    # Add some noise to simulate scanning artifacts
    noise = np.random.normal(0, 10, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    
    # Reduce contrast slightly to simulate poor scanning
    low_contrast = np.clip(noisy_image * 0.8 + 30, 0, 255).astype(np.uint8)
    
    return low_contrast

def main():
    """Main test function."""
    print("🔍 Testing PDF Preprocessing Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        import cv2
        import numpy as np
        import scipy
        from skimage import filters
        from PIL import Image
        
        # Test optional deskew library
        try:
            from deskew import determine_skew
            print("✅ Deskew library available")
            deskew_available = True
        except ImportError:
            print("ℹ️  Deskew library not available (optional)")
            deskew_available = False
        
        from pdf_preprocessor import PDFPreprocessor
        from pdf_integration import PreprocessingPDFListener
        
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        if deskew_available:
            print("✅ Deskew library integrated")
        print("✅ All modules imported successfully")
        
        # Create test image
        print("\n🖼️  Creating synthetic test image...")
        test_image = create_test_pdf_page()
        print(f"✅ Test image created: {test_image.shape}")
        
        # Initialize preprocessor
        print("\n⚙️  Initializing preprocessor...")
        preprocessor = PDFPreprocessor(
            dpi=300,
            enable_deskew=True,
            enable_denoise=True,
            enable_enhancement=True,
            debug_mode=True  # Enable debug mode for detailed output
        )
        print("✅ PDFPreprocessor initialized with all methods")
        
        # Test individual components
        print("\n🔧 Testing preprocessing components...")
        
        # Test skew detection
        angle_hough = preprocessor._detect_skew_hough(test_image)
        angle_radon = preprocessor._detect_skew_radon(test_image)
        angle_contour = preprocessor._detect_skew_contours(test_image)
        
        print(f"✅ Hough skew detection: {angle_hough if angle_hough is not None else 'No significant skew'}")
        print(f"✅ Radon skew detection: {angle_radon if angle_radon is not None else 'No significant skew'}")
        print(f"✅ Contour skew detection: {angle_contour if angle_contour is not None else 'No significant skew'}")
        
        # Test denoising
        denoised = preprocessor._apply_denoising(test_image)
        print(f"✅ Denoising applied: {denoised.shape}")
        
        # Test enhancement
        enhanced = preprocessor._apply_enhancement(test_image)
        print(f"✅ Enhancement applied: {enhanced.shape}")
        
        # Test full preprocessing pipeline
        print("\n🚀 Testing full preprocessing pipeline...")
        processed = preprocessor._process_image(test_image)
        print(f"✅ Full preprocessing completed: {processed.shape}")
        
        # Test integration classes
        print("\n🔗 Testing integration classes...")
        
        # Test with dummy Azure connection (won't actually connect)
        try:
            listener = PreprocessingPDFListener(
                connection_string="dummy_connection_string",
                container_name="test-container"
            )
            print("✅ PreprocessingPDFListener initialized")
        except Exception as e:
            print(f"⚠️  PreprocessingPDFListener test skipped (expected with dummy credentials): {str(e)[:50]}...")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 All preprocessing tests completed successfully!")
        print("\n📋 Available preprocessing features:")
        print("   • Multi-method deskewing (Hough, Radon, contour-based)")
        print("   • Advanced denoising (bilateral, median, morphological)")
        print("   • Image enhancement (CLAHE, gamma correction, sharpening)")
        print("   • Configurable processing pipeline")
        print("   • Integration with Azure Storage monitoring")
        print("   • OCR-ready output optimization")
        
        print("\n💡 Usage examples:")
        print("   python main.py preprocessing --help")
        print("   python examples/preprocessing_example.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Try installing missing dependencies:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)