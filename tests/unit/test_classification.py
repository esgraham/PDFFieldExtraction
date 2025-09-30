#!/usr/bin/env python3
"""
Test Document Classification System

This script tests the document classification functionality
without requiring all dependencies to be installed.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_classification_imports():
    """Test that classification modules can be imported."""
    print("🧪 Testing Classification Module Imports")
    print("=" * 45)
    
    try:
        # Test core imports
        from document_classifier import DocumentClassifier, DocumentClass, ClassificationResult
        print("✅ Core classification classes imported")
        
        from classification_integration import ClassificationIntegratedListener, create_classification_pipeline
        print("✅ Integration classes imported")
        
        # Test enums
        print(f"   Document classes: {[cls.value for cls in DocumentClass]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install missing dependencies:")
        print("   pip install scikit-learn sentence-transformers")
        return False

def test_basic_functionality():
    """Test basic classification functionality."""
    print("\n🔧 Testing Basic Classification Functionality")
    print("=" * 50)
    
    try:
        from document_classifier import DocumentClassifier, DocumentClass
        
        # Test classifier initialization
        classifier = DocumentClassifier(
            model_type="random_forest",
            use_transformers=False,  # Disable to avoid transformer dependency
            enable_ocr=False,        # Disable to avoid OCR dependency
            debug_mode=True
        )
        print("✅ Classifier initialized successfully")
        
        # Test feature extraction on a simple image
        test_image = np.zeros((400, 600), dtype=np.uint8)
        # Add some simple patterns
        test_image[50:100, 50:550] = 255  # Header bar
        test_image[150:170, 50:300] = 200  # Text line 1
        test_image[180:200, 50:250] = 200  # Text line 2
        test_image[210:230, 50:350] = 200  # Text line 3
        
        print("✅ Test image created")
        
        # Test layout feature extraction
        layout_features = classifier.extract_layout_features(test_image)
        print(f"✅ Layout features extracted:")
        print(f"   Page size: {layout_features.page_width}x{layout_features.page_height}")
        print(f"   Aspect ratio: {layout_features.aspect_ratio:.2f}")
        print(f"   Text density: {layout_features.text_density:.3f}")
        print(f"   Text blocks: {layout_features.text_block_count}")
        
        # Test text feature extraction
        test_text = "Invoice #12345 Date: 2024-01-15 Amount: $1,250.00"
        text_features = classifier.extract_text_features(test_image, test_text)
        print(f"✅ Text features extracted:")
        print(f"   Word count: {text_features.word_count}")
        print(f"   Has currency: {text_features.has_currency}")
        print(f"   Has dates: {text_features.has_dates}")
        print(f"   Has numbers: {text_features.has_numbers}")
        
        # Test visual feature extraction
        visual_features = classifier.extract_visual_features(test_image)
        print(f"✅ Visual features extracted:")
        print(f"   Edge density: {visual_features.edge_density:.3f}")
        print(f"   Has tables: {visual_features.has_tables}")
        print(f"   Dominant colors: {len(visual_features.dominant_colors)}")
        
        # Test combined feature extraction
        features = classifier.extract_features(test_image, test_text)
        print(f"✅ Combined features: {len(features)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_setup():
    """Test integration with existing system."""
    print("\n🔗 Testing Integration Setup")
    print("=" * 35)
    
    try:
        from classification_integration import create_classification_pipeline
        
        # Test pipeline creation (won't actually connect due to test credentials)
        pipeline = create_classification_pipeline(
            connection_string="test_connection_string",
            container_name="test-container",
            model_config={
                "model_type": "random_forest",
                "use_transformers": False,
                "enable_ocr": False
            }
        )
        
        print("✅ Classification pipeline created successfully")
        print(f"   Model type: {pipeline.classifier.model_type}")
        print(f"   Transformers enabled: {pipeline.classifier.use_transformers}")
        print(f"   OCR enabled: {pipeline.classifier.enable_ocr}")
        print(f"   Debug mode: {pipeline.classifier.debug_mode}")
        
        # Test statistics
        stats = pipeline.get_classification_stats()
        print("✅ Statistics accessible:")
        print(f"   Total classified: {stats['total_classified']}")
        print(f"   Classifier trained: {stats['classifier_trained']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def test_dependencies():
    """Test which optional dependencies are available."""
    print("\n📦 Testing Optional Dependencies")
    print("=" * 35)
    
    dependencies = {
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers", 
        "pytesseract": "pytesseract",
        "PIL": "PIL",
        "cv2": "cv2",
        "numpy": "numpy",
        "pandas": "pandas"
    }
    
    available = []
    missing = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            available.append(name)
            print(f"✅ {name}: Available")
        except ImportError:
            missing.append(name)
            print(f"❌ {name}: Missing")
    
    print(f"\n📊 Dependency Summary:")
    print(f"   Available: {len(available)}/{len(dependencies)}")
    print(f"   Missing: {missing if missing else 'None'}")
    
    if "scikit-learn" in available:
        print("\n🎯 Core ML functionality: Ready")
    else:
        print("\n⚠️  Core ML functionality: Requires scikit-learn")
    
    return len(available) >= len(dependencies) * 0.6  # 60% availability threshold

def main():
    """Run all classification tests."""
    print("🤖 Document Classification System Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_classification_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Integration Setup", test_integration_setup),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Classification system is ready.")
    elif passed >= len(results) * 0.75:
        print("⚠️  Most tests passed. System functional with minor issues.")
    else:
        print("❌ Several tests failed. Check dependencies and setup.")
    
    print("\n💡 Next Steps:")
    if "scikit-learn" not in [dep for dep, _ in results if dep == "Dependencies"]:
        print("   • Install scikit-learn: pip install scikit-learn")
    print("   • Run full demo: python examples/classification_example.py")
    print("   • See integration: python main.py classification")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)