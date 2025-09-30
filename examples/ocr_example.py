"""
Azure Document Intelligence OCR Integration Example

This example demonstrates the complete document processing pipeline with:
- Azure Document Intelligence v4 OCR for printed and handwritten text
- Layout analysis with tables and structured fields
- Integration with preprocessing and classification
- Custom model support for specialized documents
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_sample_documents():
    """Create sample documents for testing OCR capabilities."""
    
    samples_dir = Path("./examples/sample_documents")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print("📄 Creating sample documents for OCR testing...")
    
    # Sample document configurations
    sample_configs = [
        {
            "name": "invoice_sample.json",
            "type": "invoice",
            "content": {
                "company": "ABC Corp",
                "invoice_number": "INV-2024-001",
                "date": "2024-03-15",
                "amount": "$1,250.00",
                "items": [
                    {"description": "Software License", "qty": 1, "price": 1000.00},
                    {"description": "Support Services", "qty": 1, "price": 250.00}
                ],
                "has_handwritten_signature": True,
                "contains_tables": True
            }
        },
        {
            "name": "receipt_sample.json",
            "type": "receipt",
            "content": {
                "merchant": "Tech Store",
                "date": "2024-03-14",
                "total": "$89.99",
                "items": ["Wireless Mouse", "USB Cable"],
                "payment_method": "Credit Card",
                "has_handwritten_notes": False,
                "contains_tables": False
            }
        },
        {
            "name": "form_sample.json",
            "type": "form",
            "content": {
                "form_type": "Application Form",
                "fields": [
                    {"name": "full_name", "value": "John Smith", "handwritten": True},
                    {"name": "email", "value": "john.smith@email.com", "handwritten": False},
                    {"name": "phone", "value": "555-0123", "handwritten": True},
                    {"name": "address", "value": "123 Main St", "handwritten": True}
                ],
                "has_checkboxes": True,
                "has_signature": True
            }
        }
    ]
    
    # Create sample document metadata
    for config in sample_configs:
        file_path = samples_dir / config["name"]
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✅ Created: {file_path}")
    
    return samples_dir

def demonstrate_azure_document_intelligence():
    """Demonstrate Azure Document Intelligence OCR capabilities."""
    
    print("\n🤖 Azure Document Intelligence OCR Demo")
    print("=" * 50)
    
    # Note: This is a demonstration with mock Azure credentials
    # Replace with actual Azure Document Intelligence credentials
    mock_endpoint = "https://your-doc-intel-endpoint.cognitiveservices.azure.com/"
    mock_api_key = "your-api-key-here"
    
    try:
        from src.azure_document_intelligence import (
            AzureDocumentIntelligenceOCR,
            DocumentType,
            DocumentAnalysisResult
        )
        
        print("📚 Available Document Intelligence Models:")
        for doc_type in DocumentType:
            print(f"  - {doc_type.name}: {doc_type.value}")
        
        # Mock OCR engine configuration
        print(f"\n🔧 OCR Engine Configuration:")
        print(f"  - Endpoint: {mock_endpoint}")
        print(f"  - Default Model: {DocumentType.LAYOUT.value}")
        print(f"  - Handwriting Support: Enabled")
        print(f"  - Table Extraction: Enabled")
        print(f"  - Custom Models: Supported")
        
        # Demonstrate different model capabilities
        model_capabilities = {
            DocumentType.GENERAL: {
                "description": "General text extraction from any document",
                "features": ["Text extraction", "Language detection", "Handwriting recognition"],
                "use_cases": ["Letters", "Reports", "General documents"]
            },
            DocumentType.LAYOUT: {
                "description": "Layout analysis with tables and structure",
                "features": ["Text extraction", "Table detection", "Layout analysis", "Paragraph structure"],
                "use_cases": ["Forms", "Reports", "Structured documents"]
            },
            DocumentType.INVOICE: {
                "description": "Invoice-specific field extraction",
                "features": ["Vendor details", "Line items", "Totals", "Dates", "Invoice numbers"],
                "use_cases": ["Invoices", "Bills", "Purchase orders"]
            },
            DocumentType.RECEIPT: {
                "description": "Receipt-specific field extraction",
                "features": ["Merchant info", "Items", "Total amount", "Date/time", "Payment method"],
                "use_cases": ["Receipts", "Sales slips", "Transaction records"]
            },
            DocumentType.ID_DOCUMENT: {
                "description": "Identity document processing",
                "features": ["Personal details", "Document numbers", "Expiry dates", "Photos"],
                "use_cases": ["Driver licenses", "Passports", "ID cards"]
            }
        }
        
        print(f"\n📋 Model Capabilities:")
        for model_type, capabilities in model_capabilities.items():
            print(f"\n  🔹 {model_type.name}:")
            print(f"     Description: {capabilities['description']}")
            print(f"     Features: {', '.join(capabilities['features'])}")
            print(f"     Use Cases: {', '.join(capabilities['use_cases'])}")
        
    except ImportError as e:
        print(f"⚠️  Azure Document Intelligence SDK not available: {e}")
        print("   Install with: pip install azure-ai-documentintelligence")

def demonstrate_ocr_integration():
    """Demonstrate OCR integration with preprocessing and classification."""
    
    print("\n🔄 OCR Integration Pipeline Demo")
    print("=" * 50)
    
    # Mock configuration for demonstration
    pipeline_config = {
        "azure_storage": {
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=...",
            "container_name": "pdf-documents"
        },
        "azure_document_intelligence": {
            "endpoint": "https://your-doc-intel-endpoint.cognitiveservices.azure.com/",
            "api_key": "your-api-key-here"
        },
        "processing_options": {
            "enable_preprocessing": True,
            "enable_classification": True,
            "enable_ocr": True,
            "default_document_type": "prebuilt-layout"
        },
        "confidence_thresholds": {
            "ocr": 0.7,
            "classification": 0.6,
            "field_extraction": 0.8
        }
    }
    
    print("🔧 Pipeline Configuration:")
    for section, config in pipeline_config.items():
        print(f"\n  📂 {section.replace('_', ' ').title()}:")
        for key, value in config.items():
            if "key" in key.lower() or "connection" in key.lower():
                print(f"    {key}: {'*' * 20} (hidden)")
            else:
                print(f"    {key}: {value}")
    
    # Demonstrate processing pipeline
    print(f"\n🔄 Processing Pipeline Steps:")
    pipeline_steps = [
        {
            "step": "1. Document Detection",
            "description": "Monitor Azure Storage for new PDF uploads",
            "inputs": ["PDF files"],
            "outputs": ["Download notifications"]
        },
        {
            "step": "2. Preprocessing",
            "description": "Deskew, denoise, and optimize images for OCR",
            "inputs": ["Raw PDF pages"],
            "outputs": ["Preprocessed images"]
        },
        {
            "step": "3. Document Classification",
            "description": "Classify document type using layout/text features",
            "inputs": ["Preprocessed images"],
            "outputs": ["Document class", "Confidence score"]
        },
        {
            "step": "4. OCR Model Selection",
            "description": "Select appropriate Azure Document Intelligence model",
            "inputs": ["Document class"],
            "outputs": ["Selected model ID"]
        },
        {
            "step": "5. OCR Processing",
            "description": "Extract text, tables, and structured fields",
            "inputs": ["Optimized images", "Model selection"],
            "outputs": ["Text blocks", "Tables", "Structured fields"]
        },
        {
            "step": "6. Post-processing",
            "description": "Validate results and extract custom fields",
            "inputs": ["OCR results", "Document class"],
            "outputs": ["Validated data", "Custom fields"]
        },
        {
            "step": "7. Output Generation",
            "description": "Generate comprehensive analysis report",
            "inputs": ["All processing results"],
            "outputs": ["JSON report", "Structured data"]
        }
    ]
    
    for step_info in pipeline_steps:
        print(f"\n  {step_info['step']}: {step_info['description']}")
        print(f"    📥 Inputs: {', '.join(step_info['inputs'])}")
        print(f"    📤 Outputs: {', '.join(step_info['outputs'])}")

def demonstrate_handwriting_recognition():
    """Demonstrate handwriting recognition capabilities."""
    
    print("\n✍️  Handwriting Recognition Demo")
    print("=" * 50)
    
    handwriting_features = {
        "Text Types Supported": [
            "Printed text (high accuracy)",
            "Cursive handwriting",
            "Block letters",
            "Mixed print/handwriting",
            "Signatures",
            "Form field entries"
        ],
        "Languages Supported": [
            "English", "Spanish", "French", "German", "Italian",
            "Portuguese", "Dutch", "Swedish", "Norwegian"
        ],
        "Special Capabilities": [
            "Confidence scoring per text block",
            "Bounding box coordinates",
            "Text orientation detection",
            "Line and word segmentation",
            "Mixed content handling"
        ],
        "Quality Requirements": [
            "Minimum 300 DPI resolution",
            "Clear contrast between text and background",
            "Minimal skew (< 30 degrees)",
            "Legible handwriting style",
            "Adequate lighting in source image"
        ]
    }
    
    for category, items in handwriting_features.items():
        print(f"\n📝 {category}:")
        for item in items:
            print(f"  ✓ {item}")
    
    # Example handwriting processing workflow
    print(f"\n🔄 Handwriting Processing Workflow:")
    workflow_steps = [
        "Image preprocessing (deskew, denoise)",
        "Text line detection and segmentation",
        "Character-level analysis",
        "Context-aware recognition",
        "Confidence scoring",
        "Output formatting with coordinates"
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"  {i}. {step}")

def demonstrate_table_extraction():
    """Demonstrate table extraction capabilities."""
    
    print("\n📊 Table Extraction Demo")
    print("=" * 50)
    
    table_features = {
        "Detection Capabilities": [
            "Bordered tables",
            "Borderless tables",
            "Complex nested structures",
            "Multi-page spanning tables",
            "Tables with merged cells"
        ],
        "Extraction Features": [
            "Cell content extraction",
            "Header row identification", 
            "Column/row structure preservation",
            "Cell coordinates and boundaries",
            "Confidence scoring per cell"
        ],
        "Supported Formats": [
            "PDF documents",
            "Scanned images (JPEG, PNG, TIFF)",
            "Multi-page documents",
            "Mixed content (text + tables)",
            "Various table layouts"
        ],
        "Output Formats": [
            "Structured JSON with coordinates",
            "CSV-compatible row/column data",
            "HTML table representation",
            "Custom field mappings",
            "Bounding box annotations"
        ]
    }
    
    for category, items in table_features.items():
        print(f"\n📋 {category}:")
        for item in items:
            print(f"  ✓ {item}")
    
    # Example table structure
    print(f"\n📄 Example Extracted Table Structure:")
    example_table = {
        "table_index": 0,
        "rows": 4,
        "columns": 3,
        "has_headers": True,
        "confidence": 0.95,
        "headers": ["Product", "Quantity", "Price"],
        "data": [
            ["Software License", "1", "$1,000.00"],
            ["Support Services", "1", "$250.00"],
            ["Training Materials", "2", "$150.00"]
        ],
        "bounding_box": {
            "x": 100, "y": 200, "width": 400, "height": 120
        }
    }
    
    print(json.dumps(example_table, indent=2))

def demonstrate_custom_models():
    """Demonstrate custom model training and usage."""
    
    print("\n🎯 Custom Models Demo")
    print("=" * 50)
    
    custom_model_info = {
        "Training Process": [
            "Collect 5+ sample documents of same type",
            "Label key fields and structures",
            "Upload training data to Azure",
            "Train custom model (5-10 minutes)",
            "Test and validate model accuracy",
            "Deploy for production use"
        ],
        "Use Cases": [
            "Company-specific forms",
            "Industry-specific documents",
            "Unique layout structures",
            "Specialized field types",
            "Legacy document formats"
        ],
        "Field Types Supported": [
            "Text fields",
            "Number fields", 
            "Date fields",
            "Currency amounts",
            "Checkboxes/selections",
            "Signature regions"
        ],
        "Integration Benefits": [
            "Higher accuracy for specific formats",
            "Reduced post-processing",
            "Consistent field extraction",
            "Automated workflow integration",
            "Custom validation rules"
        ]
    }
    
    for category, items in custom_model_info.items():
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"  ✓ {item}")
    
    # Example custom model configuration
    print(f"\n📋 Example Custom Model Configuration:")
    custom_config = {
        "model_id": "custom-invoice-model-v1",
        "document_type": "company_invoice",
        "trained_fields": [
            {
                "field_name": "vendor_name",
                "field_type": "text",
                "required": True,
                "validation": "non_empty"
            },
            {
                "field_name": "invoice_number", 
                "field_type": "text",
                "required": True,
                "validation": "alphanumeric"
            },
            {
                "field_name": "total_amount",
                "field_type": "currency",
                "required": True,
                "validation": "positive_number"
            },
            {
                "field_name": "due_date",
                "field_type": "date",
                "required": False,
                "validation": "future_date"
            }
        ],
        "accuracy_metrics": {
            "overall_accuracy": 0.94,
            "field_level_accuracy": {
                "vendor_name": 0.97,
                "invoice_number": 0.96,
                "total_amount": 0.92,
                "due_date": 0.89
            }
        }
    }
    
    print(json.dumps(custom_config, indent=2))

async def demonstrate_async_processing():
    """Demonstrate asynchronous document processing."""
    
    print("\n⚡ Asynchronous Processing Demo")
    print("=" * 50)
    
    # Simulate processing multiple documents
    documents = [
        {"name": "invoice_001.pdf", "type": "invoice", "pages": 2},
        {"name": "receipt_042.pdf", "type": "receipt", "pages": 1},
        {"name": "form_application.pdf", "type": "form", "pages": 3},
        {"name": "report_quarterly.pdf", "type": "report", "pages": 15}
    ]
    
    print(f"📁 Processing {len(documents)} documents concurrently...")
    
    async def process_document_mock(doc):
        """Mock document processing function."""
        processing_time = doc["pages"] * 0.5  # Simulate processing time
        await asyncio.sleep(processing_time)
        
        return {
            "document": doc["name"],
            "status": "completed",
            "processing_time": processing_time,
            "text_blocks": doc["pages"] * 5,  # Mock text blocks
            "tables": 1 if doc["type"] in ["invoice", "form"] else 0,
            "fields": doc["pages"] * 2,  # Mock structured fields
            "confidence": 0.85 + (0.1 * (doc["pages"] % 3))  # Mock confidence
        }
    
    # Process documents concurrently
    start_time = time.time()
    results = await asyncio.gather(*[process_document_mock(doc) for doc in documents])
    total_time = time.time() - start_time
    
    print(f"\n📊 Processing Results (completed in {total_time:.2f}s):")
    for result in results:
        print(f"\n  📄 {result['document']}:")
        print(f"    Status: {result['status']}")
        print(f"    Processing Time: {result['processing_time']:.1f}s")
        print(f"    Text Blocks: {result['text_blocks']}")
        print(f"    Tables: {result['tables']}")
        print(f"    Structured Fields: {result['fields']}")
        print(f"    Confidence: {result['confidence']:.2f}")
    
    # Calculate statistics
    total_docs = len(results)
    avg_processing_time = sum(r['processing_time'] for r in results) / total_docs
    total_text_blocks = sum(r['text_blocks'] for r in results)
    total_tables = sum(r['tables'] for r in results)
    avg_confidence = sum(r['confidence'] for r in results) / total_docs
    
    print(f"\n📈 Batch Processing Statistics:")
    print(f"  Total Documents: {total_docs}")
    print(f"  Total Processing Time: {total_time:.2f}s")
    print(f"  Average Processing Time: {avg_processing_time:.2f}s per document")
    print(f"  Total Text Blocks Extracted: {total_text_blocks}")
    print(f"  Total Tables Extracted: {total_tables}")
    print(f"  Average Confidence: {avg_confidence:.2f}")
    print(f"  Throughput: {total_docs / total_time:.1f} documents/second")

def main():
    """Run the complete OCR integration demonstration."""
    
    print("🚀 Azure Document Intelligence OCR Integration Demo")
    print("=" * 60)
    
    try:
        # Create sample documents
        samples_dir = create_sample_documents()
        
        # Demonstrate different components
        demonstrate_azure_document_intelligence()
        demonstrate_ocr_integration()
        demonstrate_handwriting_recognition()
        demonstrate_table_extraction()
        demonstrate_custom_models()
        
        # Run async demo
        asyncio.run(demonstrate_async_processing())
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Sample documents created in: {samples_dir}")
        print(f"\n📚 Next Steps:")
        print(f"  1. Set up Azure Document Intelligence service")
        print(f"  2. Configure API credentials in environment variables")
        print(f"  3. Install required dependencies: pip install azure-ai-documentintelligence")
        print(f"  4. Test with real documents")
        print(f"  5. Train custom models for specific document types")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()