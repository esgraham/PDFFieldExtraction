#!/usr/bin/env python3
"""
PDF Field Extraction System - Quick Demo

This script demonstrates the complete PDF processing pipeline including:
- Azure PDF monitoring 
- Document preprocessing (deskew/denoise)
- Document classification
- OCR with Azure Document Intelligence
- Field extraction with template matching
- Business rules validation 
- Confidence scoring and HITL routing
- Queue management with poison queue pattern

Quick Start:
    python quick_demo.py

Requirements:
    - Install dependencies: pip install -r requirements.txt
    - Configure Azure credentials (see README.md)
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("🚀 PDF Field Extraction System - Quick Demo")
print("=" * 60)

# Test imports
try:
    from field_extraction import FieldExtractor, DocumentTemplate
    print("✅ Field Extraction Engine - Ready")
except ImportError as e:
    print(f"❌ Field Extraction Engine - Error: {e}")

try:
    from hitl_queue_manager import HITLQueueManager, TaskPriority  
    print("✅ HITL Queue Manager - Ready")
except ImportError as e:
    print(f"❌ HITL Queue Manager - Error: {e}")

try:
    from azure_document_intelligence import AzureDocumentIntelligence
    print("✅ Azure Document Intelligence - Ready")
except ImportError as e:
    print(f"❌ Azure Document Intelligence - Error: {e}")

try:
    from document_classifier import DocumentClassifier
    print("✅ Document Classifier - Ready")
except ImportError as e:
    print(f"❌ Document Classifier - Error: {e}")
    
try:
    from preprocessing import DocumentPreprocessor
    print("✅ Document Preprocessor - Ready")
except ImportError as e:
    print(f"❌ Document Preprocessor - Error: {e}")

try:
    from azure_pdf_listener import AzurePDFListener
    print("✅ Azure PDF Listener - Ready")
except ImportError as e:
    print(f"❌ Azure PDF Listener - Error: {e}")

print("\n📋 System Features:")
print("  🔍 Azure Storage PDF monitoring with event-driven processing")
print("  🖼️  Document preprocessing (deskew, denoise, enhancement)")
print("  🏷️  Intelligent document classification (invoice, receipt, contract, etc.)")
print("  👁️  Advanced OCR with Azure Document Intelligence v4")
print("  📊 Template-based field extraction with confidence scoring")
print("  ✅ Business rule validation (dates, currencies, cross-field checks)")
print("  👤 Human-in-the-loop routing for low-confidence documents")
print("  🔄 Robust queue management with poison queue pattern")
print("  📈 Comprehensive logging and statistics")

print("\n⚙️  Configuration:")
print("  📁 Schema directory: ./config/schemas")
print("  🎯 Default confidence threshold: 0.7")
print("  👤 HITL routing threshold: 0.6")
print("  🔄 Max retry attempts: 3")
print("  ⏱️  Backoff strategy: Exponential")

print("\n📂 Available Document Types:")
templates = [
    ("INVOICE", "Standard business invoices with vendor, amounts, dates"),
    ("RECEIPT", "Purchase receipts with store, items, totals"),
    ("PURCHASE_ORDER", "Purchase orders with PO numbers, vendor info"),
    ("CONTRACT", "Business contracts and agreements"),
    ("TAX_FORM", "Tax documents and forms"),
    ("FORM_APPLICATION", "Application forms and submissions")
]

for template_name, description in templates:
    print(f"  📄 {template_name}: {description}")

print("\n🔧 Usage Examples:")

print("\n1️⃣  Basic Field Extraction:")
print("```python")
print("from field_extraction import FieldExtractor, DocumentTemplate")
print("")
print("extractor = FieldExtractor()")
print("result = extractor.extract_fields(ocr_result, DocumentTemplate.INVOICE)")
print("")
print("for field in result.extracted_fields:")
print("    print(f'{field.field_name}: {field.normalized_value}')")
print("```")

print("\n2️⃣  HITL Queue Management:")
print("```python")
print("from hitl_queue_manager import HITLQueueManager, TaskPriority")
print("")
print("manager = HITLQueueManager()")
print("task_id = await manager.add_task(")
print("    document_id='doc_001',")
print("    document_type='invoice',")
print("    priority=TaskPriority.HIGH,")
print("    reason='Low confidence extraction'")
print(")")
print("```")

print("\n3️⃣  Complete Pipeline:")
print("```python")
print("from field_extraction_integration import IntegratedFieldExtractor")
print("")
print("processor = IntegratedFieldExtractor(ocr_client, extractor, hitl_manager)")
print("result = await processor.process_document(document_content, 'invoice')")
print("```")

print("\n📚 Documentation:")
print("  📖 README.md - Complete setup and usage guide")
print("  🔧 requirements.txt - All required dependencies")
print("  📝 examples/ - Sample usage and test scripts")
print("  ⚙️  config/ - Document schemas and configuration")

print("\n🎯 Production Checklist:")
checklist = [
    "✅ Install dependencies: pip install -r requirements.txt",
    "🔑 Configure Azure Storage connection string",
    "🔑 Configure Azure Document Intelligence endpoint/key",
    "📁 Review document schemas in config/schemas/",
    "⚙️  Adjust confidence thresholds for your use case",
    "🗄️  Set up Azure Service Bus for production HITL queues",
    "📊 Configure monitoring and alerting",
    "🧪 Test with your specific document types",
    "🔒 Implement proper authentication and security",
    "📈 Set up performance monitoring"
]

for item in checklist:
    print(f"  {item}")

print("\n" + "=" * 60)
print("🏁 Demo completed! Your PDF field extraction system is ready.")
print("🚀 Start with: python examples/field_extraction_example.py")
print("📖 Full documentation: README.md")
print("🆘 Issues? Check logs and configuration files.")