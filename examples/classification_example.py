#!/usr/bin/env python3
"""
Document Classification Example

This script demonstrates how to use the document classification system
with layout analysis, text embeddings, and visual features.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_documents():
    """Create sample documents for testing classification."""
    documents = []
    
    # Document Type A: Invoice-like document
    doc_a = create_invoice_document()
    documents.append({
        "image": doc_a,
        "label": "A",
        "text": "INVOICE #12345\nDate: 2024-01-15\nAmount: $1,250.00\nCustomer: ABC Corp\nDescription: Professional services",
        "description": "Invoice document with structured layout"
    })
    
    # Document Type B: Report-like document
    doc_b = create_report_document()
    documents.append({
        "image": doc_b,
        "label": "B", 
        "text": "QUARTERLY REPORT\nQ3 2024 Performance Analysis\nRevenue increased by 15% compared to previous quarter. Key metrics show positive trends in customer acquisition and retention.",
        "description": "Report document with text-heavy layout"
    })
    
    # Document Type C: Form-like document
    doc_c = create_form_document()
    documents.append({
        "image": doc_c,
        "label": "C",
        "text": "APPLICATION FORM\nName: ________________\nAddress: ________________\nPhone: ________________\nEmail: ________________",
        "description": "Form document with input fields"
    })
    
    return documents

def create_invoice_document():
    """Create a synthetic invoice-like document."""
    # Create white background
    width, height = 600, 800
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # Header
    draw.text((50, 30), "INVOICE", fill='black', font=font_large)
    draw.text((400, 30), "#12345", fill='black', font=font_medium)
    
    # Company logo placeholder (rectangle)
    draw.rectangle([50, 80, 150, 120], outline='black', width=2)
    draw.text((60, 90), "LOGO", fill='black', font=font_small)
    
    # Invoice details
    draw.text((50, 150), "Date: January 15, 2024", fill='black', font=font_medium)
    draw.text((50, 180), "Invoice #: 12345", fill='black', font=font_medium)
    draw.text((50, 210), "Due Date: February 15, 2024", fill='black', font=font_medium)
    
    # Customer info
    draw.text((300, 150), "Bill To:", fill='black', font=font_medium)
    draw.text((300, 180), "ABC Corporation", fill='black', font=font_medium)
    draw.text((300, 210), "123 Business St", fill='black', font=font_medium)
    draw.text((300, 240), "City, ST 12345", fill='black', font=font_medium)
    
    # Table header
    y_start = 300
    draw.line([(50, y_start), (550, y_start)], fill='black', width=2)
    draw.text((60, y_start + 10), "Description", fill='black', font=font_medium)
    draw.text((300, y_start + 10), "Qty", fill='black', font=font_medium)
    draw.text((400, y_start + 10), "Rate", fill='black', font=font_medium)
    draw.text((480, y_start + 10), "Amount", fill='black', font=font_medium)
    draw.line([(50, y_start + 35), (550, y_start + 35)], fill='black', width=1)
    
    # Table rows
    items = [
        ("Professional Services", "10", "$125.00", "$1,250.00"),
        ("Consultation", "5", "$100.00", "$500.00")
    ]
    
    for i, (desc, qty, rate, amount) in enumerate(items):
        y = y_start + 50 + i * 30
        draw.text((60, y), desc, fill='black', font=font_small)
        draw.text((300, y), qty, fill='black', font=font_small)
        draw.text((400, y), rate, fill='black', font=font_small)
        draw.text((480, y), amount, fill='black', font=font_small)
    
    # Total
    draw.line([(400, y_start + 130), (550, y_start + 130)], fill='black', width=2)
    draw.text((400, y_start + 140), "TOTAL:", fill='black', font=font_medium)
    draw.text((480, y_start + 140), "$1,750.00", fill='black', font=font_medium)
    
    return np.array(image.convert('L'))

def create_report_document():
    """Create a synthetic report-like document."""
    width, height = 600, 800
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # Header
    draw.text((50, 30), "QUARTERLY PERFORMANCE REPORT", fill='black', font=font_large)
    draw.text((50, 60), "Q3 2024", fill='black', font=font_medium)
    
    # Content sections
    sections = [
        ("Executive Summary", 120),
        ("Financial Performance", 250),
        ("Market Analysis", 380),
        ("Recommendations", 510)
    ]
    
    for title, y_pos in sections:
        draw.text((50, y_pos), title, fill='black', font=font_medium)
        
        # Add some paragraph text
        paragraph_lines = [
            "This section provides detailed analysis of key",
            "performance indicators and market trends that",
            "have shaped our business during this quarter.",
            "",
            "Key findings include significant growth in",
            "customer acquisition and improved operational",
            "efficiency across all business units."
        ]
        
        for i, line in enumerate(paragraph_lines):
            draw.text((50, y_pos + 30 + i * 15), line, fill='black', font=font_small)
    
    # Add a simple chart placeholder
    chart_y = 650
    draw.rectangle([100, chart_y, 500, chart_y + 100], outline='black', width=2)
    draw.text((250, chart_y + 40), "Performance Chart", fill='black', font=font_medium)
    
    # Add some chart elements
    for i in range(5):
        x = 120 + i * 75
        height_bar = 20 + i * 10
        draw.rectangle([x, chart_y + 80 - height_bar, x + 20, chart_y + 80], fill='gray')
    
    return np.array(image.convert('L'))

def create_form_document():
    """Create a synthetic form-like document."""
    width, height = 600, 800
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # Header
    draw.text((200, 30), "APPLICATION FORM", fill='black', font=font_large)
    
    # Form fields
    fields = [
        ("Full Name:", 120),
        ("Date of Birth:", 170),
        ("Address:", 220),
        ("City, State, ZIP:", 270),
        ("Phone Number:", 320),
        ("Email Address:", 370),
        ("Occupation:", 420),
        ("Emergency Contact:", 470)
    ]
    
    for label, y_pos in fields:
        # Field label
        draw.text((50, y_pos), label, fill='black', font=font_medium)
        
        # Input line
        draw.line([(200, y_pos + 20), (550, y_pos + 20)], fill='black', width=1)
    
    # Checkboxes section
    checkbox_y = 550
    draw.text((50, checkbox_y), "Please check all that apply:", fill='black', font=font_medium)
    
    checkbox_options = [
        "□ I agree to the terms and conditions",
        "□ I would like to receive email updates",
        "□ I authorize background check"
    ]
    
    for i, option in enumerate(checkbox_options):
        draw.text((70, checkbox_y + 30 + i * 25), option, fill='black', font=font_small)
    
    # Signature section
    sig_y = 680
    draw.text((50, sig_y), "Signature:", fill='black', font=font_medium)
    draw.line([(150, sig_y + 20), (350, sig_y + 20)], fill='black', width=1)
    
    draw.text((400, sig_y), "Date:", fill='black', font=font_medium)
    draw.line([(450, sig_y + 20), (550, sig_y + 20)], fill='black', width=1)
    
    return np.array(image.convert('L'))

def demonstrate_classification():
    """Demonstrate document classification functionality."""
    print("🔍 Document Classification System Demo")
    print("=" * 50)
    
    try:
        from document_classifier import DocumentClassifier, DocumentClass
        
        # Create sample documents
        print("📄 Creating sample documents...")
        documents = create_sample_documents()
        print(f"✅ Created {len(documents)} sample documents")
        
        # Initialize classifier
        print("\n🤖 Initializing document classifier...")
        classifier = DocumentClassifier(
            model_type="random_forest",
            use_transformers=False,  # Start without transformers for simplicity
            enable_ocr=False,  # We'll provide text directly
            debug_mode=True
        )
        print("✅ Classifier initialized")
        
        # Prepare training data
        print("\n📚 Preparing training data...")
        training_images = []
        training_labels = []
        training_texts = []
        
        # Create multiple samples of each type for training
        for _ in range(3):  # Create 3 samples of each type
            for doc in documents:
                training_images.append(doc["image"])
                training_labels.append(DocumentClass(doc["label"]))
                training_texts.append(doc["text"])
        
        print(f"✅ Prepared {len(training_images)} training samples")
        
        # Train classifier
        print("\n🎯 Training classifier...")
        classifier.train(training_images, training_labels, training_texts)
        print("✅ Training completed")
        
        # Test classification on new samples
        print("\n🧪 Testing classification...")
        test_documents = create_sample_documents()
        
        for i, doc in enumerate(test_documents):
            print(f"\n📋 Testing Document {i+1} ({doc['description']}):")
            result = classifier.classify(doc["image"], doc["text"])
            
            print(f"   Predicted Class: {result.predicted_class.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Processing Time: {result.processing_time:.3f}s")
            print(f"   Expected: {doc['label']}")
            print(f"   ✅ Correct!" if result.predicted_class.value == doc['label'] else "❌ Incorrect")
            
            # Show all probabilities
            print("   Class Probabilities:")
            for cls, prob in result.probabilities.items():
                print(f"     {cls.value}: {prob:.3f}")
        
        # Feature analysis
        print("\n🔍 Feature Analysis:")
        sample_doc = test_documents[0]
        layout_features = classifier.extract_layout_features(sample_doc["image"])
        text_features = classifier.extract_text_features(sample_doc["image"], sample_doc["text"])
        visual_features = classifier.extract_visual_features(sample_doc["image"])
        
        print(f"   Layout Features:")
        print(f"     Page Size: {layout_features.page_width}x{layout_features.page_height}")
        print(f"     Aspect Ratio: {layout_features.aspect_ratio:.2f}")
        print(f"     Text Density: {layout_features.text_density:.3f}")
        print(f"     Has Logo: {layout_features.has_logo}")
        print(f"     Text Blocks: {layout_features.text_block_count}")
        
        print(f"   Text Features:")
        print(f"     Word Count: {text_features.word_count}")
        print(f"     Unique Words: {text_features.unique_word_count}")
        print(f"     Has Numbers: {text_features.has_numbers}")
        print(f"     Has Currency: {text_features.has_currency}")
        print(f"     Has Dates: {text_features.has_dates}")
        
        print(f"   Visual Features:")
        print(f"     Edge Density: {visual_features.edge_density:.3f}")
        print(f"     Line Density: {visual_features.line_density:.6f}")
        print(f"     Has Tables: {visual_features.has_tables}")
        print(f"     Dominant Colors: {len(visual_features.dominant_colors)}")
        
        print("\n🎉 Classification demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   • Add more document types and training data")
        print("   • Enable transformer-based text embeddings")
        print("   • Integrate with Azure Storage monitoring")
        print("   • Fine-tune classification thresholds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install missing dependencies:")
        print("   pip install scikit-learn sentence-transformers")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_integration():
    """Demonstrate integration with existing PDF processing system."""
    print("\n🔗 Integration with PDF Processing Demo")
    print("=" * 50)
    
    try:
        from classification_integration import create_classification_pipeline
        
        print("🏗️  Creating classification pipeline...")
        
        # This would normally use real Azure credentials
        pipeline = create_classification_pipeline(
            connection_string="test_connection_string",
            container_name="test-container",
            model_config={
                "model_type": "random_forest",
                "use_transformers": False,
                "enable_ocr": True
            },
            preprocessing_config={
                "dpi": 300,
                "enable_deskew": True,
                "enable_denoise": True,
                "enable_enhancement": True
            }
        )
        
        print("✅ Classification pipeline created")
        print("   • Includes PDF preprocessing")
        print("   • Includes document classification")
        print("   • Includes Azure Storage monitoring")
        print("   • Ready for production use")
        
        # Show configuration
        print(f"\n⚙️  Pipeline Configuration:")
        print(f"   Model Type: {pipeline.classifier.model_type}")
        print(f"   Transformers: {pipeline.classifier.use_transformers}")
        print(f"   OCR Enabled: {pipeline.classifier.enable_ocr}")
        print(f"   Auto Training: {pipeline.auto_train}")
        
        print("\n📋 Usage Example:")
        print("""
        # Start monitoring Azure Storage
        pipeline.start_polling()
        
        # Or process specific document
        result = pipeline.process_pdf_with_classification(
            blob_name="document.pdf",
            blob_data=pdf_bytes,
            expected_class=DocumentClass.CLASS_A  # For training
        )
        
        # Train classifier with collected data
        training_result = pipeline.train_classifier(training_data)
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration demo error: {e}")
        return False

if __name__ == "__main__":
    success = demonstrate_classification()
    
    if success:
        demonstrate_integration()
    
    print(f"\n{'✅ Demo completed successfully!' if success else '❌ Demo failed'}")