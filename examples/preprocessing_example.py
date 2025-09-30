"""
Advanced example showing PDF preprocessing with OCR integration.

This example demonstrates how to use the preprocessing capabilities
with the Azure PDF listener for document processing workflows.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_preprocessing_callback(blob_name, processed_images, stats):
    """
    Example callback for handling preprocessed PDF results.
    
    Args:
        blob_name: Name of the processed PDF
        processed_images: List of preprocessed images
        stats: Processing statistics
    """
    print(f"\n🔍 Preprocessing completed for: {blob_name}")
    print(f"📄 Pages processed: {len(processed_images)}")
    
    if 'average_skew' in stats:
        print(f"📐 Average skew angle: {stats['average_skew']:.2f}°")
    
    if 'max_skew' in stats:
        print(f"📐 Maximum skew detected: {stats['max_skew']:.2f}°")
    
    # Here you could add additional processing:
    # - Save images to database
    # - Trigger OCR processing
    # - Send notifications
    # - Update processing status
    
    print(f"✅ Ready for OCR processing")


def example_ocr_callback(blob_name, ocr_results, processing_stats):
    """
    Example callback for handling OCR results.
    
    Args:
        blob_name: Name of the processed PDF
        ocr_results: List of OCR results per page
        processing_stats: Processing statistics
    """
    print(f"\n📖 OCR completed for: {blob_name}")
    
    total_words = sum(result.get('word_count', 0) for result in ocr_results)
    avg_confidence = sum(result.get('confidence', 0) for result in ocr_results) / len(ocr_results) if ocr_results else 0
    
    print(f"📊 Total words extracted: {total_words}")
    print(f"📊 Average confidence: {avg_confidence:.1f}%")
    
    # Display sample text from each page
    for i, result in enumerate(ocr_results):
        text_preview = result.get('text', '')[:100]
        if text_preview:
            print(f"   Page {i+1}: {text_preview}...")
    
    # Here you could:
    # - Store extracted text in database
    # - Perform text analysis/NLP
    # - Extract specific fields or patterns
    # - Generate search indexes
    
    print("✅ Text extraction completed")


def example_1_basic_preprocessing():
    """Example 1: Basic preprocessing without OCR."""
    print("=== Example 1: Basic Preprocessing ===")
    
    try:
        from pdf_integration import create_preprocessing_listener
        
        # Configuration
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if not storage_account or not connection_string:
            print("❌ Please configure Azure Storage credentials")
            return
        
        # Create preprocessing listener
        listener = create_preprocessing_listener(
            storage_account_name=storage_account,
            container_name=container_name,
            connection_string=connection_string,
            preprocessing_config={
                'enable_preprocessing': True,
                'preprocessing_dpi': 300,
                'enable_deskew': True,
                'enable_denoise': True,
                'enable_enhancement': True,
                'save_preprocessed': True,
                'preprocessed_container': 'preprocessed-pdfs'
            }
        )
        
        # Set preprocessing callback
        listener.set_preprocessing_callback(example_preprocessing_callback)
        
        # Override the PDF callback to include preprocessing
        def enhanced_pdf_callback(blob_name, blob_client):
            print(f"🔄 Processing new PDF: {blob_name}")
            
            # Apply preprocessing
            processed_images = listener.process_pdf_with_preprocessing(blob_name, blob_client)
            
            if processed_images:
                print(f"✅ Preprocessing successful: {len(processed_images)} pages")
            else:
                print(f"❌ Preprocessing failed for {blob_name}")
        
        listener.set_pdf_callback(enhanced_pdf_callback)
        
        print("🚀 Starting preprocessing listener...")
        print("Upload a PDF to see preprocessing in action!")
        print("Press Ctrl+C to stop")
        
        try:
            listener.start_polling()
        except KeyboardInterrupt:
            print("\n🛑 Preprocessing listener stopped")
            
            # Show statistics
            stats = listener.get_processing_statistics()
            print(f"\n📊 Processing Statistics:")
            print(f"   PDFs processed: {stats.get('pdfs_processed', 0)}")
            print(f"   Total pages: {stats.get('total_pages_processed', 0)}")
            print(f"   Failures: {stats.get('preprocessing_failures', 0)}")
            print(f"   Avg processing time: {stats.get('average_processing_time', 0):.2f}s")
    
    except ImportError as e:
        print(f"❌ Missing dependencies for preprocessing: {e}")
        print("Run: pip install opencv-python numpy scipy scikit-image Pillow PyMuPDF")


def example_2_ocr_integration():
    """Example 2: Preprocessing with OCR integration."""
    print("\n=== Example 2: OCR Integration ===")
    
    try:
        from pdf_integration import create_ocr_listener
        
        # Configuration
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if not storage_account or not connection_string:
            print("❌ Please configure Azure Storage credentials")
            return
        
        # Create OCR integrated listener
        listener = create_ocr_listener(
            storage_account_name=storage_account,
            container_name=container_name,
            connection_string=connection_string,
            ocr_config={
                'ocr_engine': 'tesseract',  # or 'easyocr', 'paddleocr'
                'ocr_languages': ['eng'],
                'extract_tables': False
            },
            preprocessing_config={
                'enable_preprocessing': True,
                'preprocessing_dpi': 300,
                'enable_deskew': True,
                'enable_denoise': True,
                'enable_enhancement': True
            }
        )
        
        # Enhanced callback with OCR
        def ocr_pdf_callback(blob_name, blob_client):
            print(f"🔄 Processing PDF with OCR: {blob_name}")
            
            # Apply preprocessing
            processed_images = listener.process_pdf_with_preprocessing(blob_name, blob_client)
            
            if processed_images:
                # Extract text using OCR
                print(f"📖 Extracting text from {len(processed_images)} pages...")
                ocr_results = listener.extract_text_from_images(processed_images)
                
                # Call OCR callback
                example_ocr_callback(blob_name, ocr_results, {})
                
            else:
                print(f"❌ Preprocessing failed for {blob_name}")
        
        listener.set_pdf_callback(ocr_pdf_callback)
        
        print("🚀 Starting OCR-integrated listener...")
        print("Upload a PDF to see preprocessing + OCR in action!")
        print("Press Ctrl+C to stop")
        
        try:
            listener.start_polling()
        except KeyboardInterrupt:
            print("\n🛑 OCR listener stopped")
    
    except ImportError as e:
        print(f"❌ Missing dependencies for OCR: {e}")
        print("Install OCR engine: pip install pytesseract (requires Tesseract binary)")
        print("Alternative: pip install easyocr")


def example_3_standalone_preprocessing():
    """Example 3: Standalone preprocessing of local PDF."""
    print("\n=== Example 3: Standalone Preprocessing ===")
    
    try:
        from pdf_preprocessor import PDFPreprocessor, preprocess_for_ocr
        
        # Check if we have a sample PDF
        sample_pdf = Path("sample.pdf")
        if not sample_pdf.exists():
            print("❌ No sample.pdf found in current directory")
            print("   Place a PDF file named 'sample.pdf' to test standalone preprocessing")
            return
        
        print(f"📄 Processing sample PDF: {sample_pdf}")
        
        # Method 1: Quick preprocessing
        print("\n🔧 Method 1: Quick preprocessing")
        processed_images = preprocess_for_ocr(
            sample_pdf, 
            output_dir="preprocessing_output",
            aggressive_deskew=True
        )
        
        print(f"✅ Processed {len(processed_images)} pages")
        print("🔍 Check 'preprocessing_output' directory for debug images")
        
        # Method 2: Custom preprocessing pipeline
        print("\n🔧 Method 2: Custom preprocessing")
        
        preprocessor = PDFPreprocessor(
            dpi=300,
            enable_deskew=True,
            enable_denoise=True,
            enable_enhancement=True,
            debug_mode=True
        )
        
        custom_processed = preprocessor.process_pdf(
            sample_pdf,
            output_dir="custom_preprocessing"
        )
        
        # Get statistics
        stats = preprocessor.get_processing_stats()
        
        print(f"✅ Custom processing completed:")
        print(f"   Pages: {stats.get('pages_processed', 0)}")
        if 'average_skew' in stats:
            print(f"   Average skew: {stats['average_skew']:.2f}°")
        if 'max_skew' in stats:
            print(f"   Max skew: {stats['max_skew']:.2f}°")
        
        print("🔍 Check 'custom_preprocessing' directory for results")
    
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install: pip install PyMuPDF opencv-python numpy scipy scikit-image Pillow")


def example_4_batch_processing():
    """Example 4: Batch processing of multiple PDFs."""
    print("\n=== Example 4: Batch Processing ===")
    
    try:
        from pdf_preprocessor import batch_preprocess
        
        # Look for PDF files in current directory
        pdf_files = list(Path(".").glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in current directory")
            print("   Add some PDF files to test batch processing")
            return
        
        print(f"📄 Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
        
        # Batch process all PDFs
        results = batch_preprocess(
            pdf_files,
            output_dir="batch_output",
            dpi=300,
            enable_deskew=True,
            enable_denoise=True,
            enable_enhancement=True,
            debug_mode=True
        )
        
        # Report results
        successful = sum(1 for images in results.values() if images)
        failed = len(results) - successful
        total_pages = sum(len(images) for images in results.values())
        
        print(f"✅ Batch processing completed:")
        print(f"   Successful: {successful} PDFs")
        print(f"   Failed: {failed} PDFs")
        print(f"   Total pages: {total_pages}")
        print("🔍 Check 'batch_output' directory for results")
    
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")


def main():
    """Main function to run preprocessing examples."""
    
    print("🚀 Azure PDF Listener with Preprocessing Examples")
    print("=" * 60)
    
    # Check basic requirements
    required_vars = ["AZURE_STORAGE_ACCOUNT_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Some examples will be skipped")
    
    print("\nAvailable examples:")
    print("1. Basic preprocessing (Azure integration)")
    print("2. OCR integration (Azure + OCR)")
    print("3. Standalone preprocessing (local files)")
    print("4. Batch processing (multiple local files)")
    
    choice = input("\nSelect example (1-4) or 'all': ").strip().lower()
    
    if choice in ['1', 'all']:
        example_1_basic_preprocessing()
    
    if choice in ['2', 'all']:
        example_2_ocr_integration()
    
    if choice in ['3', 'all']:
        example_3_standalone_preprocessing()
    
    if choice in ['4', 'all']:
        example_4_batch_processing()
    
    if choice not in ['1', '2', '3', '4', 'all']:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()