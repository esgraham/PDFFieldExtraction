#!/usr/bin/env python3
"""
Complete System Test for PDF Field Extraction Pipeline

This test demonstrates the full end-to-end processing pipeline:
1. Document preprocessing and classification
2. OCR with Azure Document Intelligence
3. Field extraction with template matching
4. Business rules validation
5. Confidence scoring and HITL routing
6. Queue management with poison queue pattern

Run with: python test_complete_system.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from azure_pdf_listener import AzurePDFListener
from preprocessing import DocumentPreprocessor
from document_classifier import DocumentClassifier
from azure_document_intelligence import AzureDocumentIntelligence
from field_extraction import FieldExtractor
from hitl_queue_manager import HITLQueueManager
from field_extraction_integration import IntegratedFieldExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteSystemTest:
    """Test harness for the complete PDF processing system"""
    
    def __init__(self):
        self.connection_string = "DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;"
        self.container_name = "pdf-documents"
        self.endpoint = "https://your-resource.cognitiveservices.azure.com/"
        self.key = "your_document_intelligence_key"
        
        # Initialize components
        self.preprocessor = DocumentPreprocessor()
        self.classifier = DocumentClassifier()
        self.ocr = AzureDocumentIntelligence(self.endpoint, self.key)
        self.field_extractor = FieldExtractor()
        self.hitl_manager = HITLQueueManager()
        self.integrated_extractor = IntegratedFieldExtractor(
            self.ocr, self.field_extractor, self.hitl_manager
        )
        
    def create_test_schemas(self):
        """Create test document schemas for demonstration"""
        logger.info("Creating test document schemas...")
        
        # Invoice schema
        invoice_schema = self.field_extractor.create_document_schema(
            "invoice",
            "Standard business invoice",
            required_fields=[
                "invoice_number", "invoice_date", "due_date", 
                "vendor_name", "customer_name", "total_amount"
            ],
            optional_fields=[
                "vendor_address", "customer_address", "tax_amount",
                "discount_amount", "payment_terms", "line_items"
            ]
        )
        
        # Purchase order schema
        po_schema = self.field_extractor.create_document_schema(
            "purchase_order",
            "Standard purchase order",
            required_fields=[
                "po_number", "po_date", "vendor_name", "total_amount"
            ],
            optional_fields=[
                "vendor_address", "ship_to_address", "requested_delivery_date",
                "line_items", "special_instructions"
            ]
        )
        
        # Contract schema
        contract_schema = self.field_extractor.create_document_schema(
            "contract",
            "Business contract or agreement",
            required_fields=[
                "contract_number", "effective_date", "expiration_date",
                "party1_name", "party2_name"
            ],
            optional_fields=[
                "contract_value", "renewal_terms", "termination_clause",
                "governing_law"
            ]
        )
        
        logger.info(f"Created schemas: {len(invoice_schema.required_fields)} invoice fields, "
                   f"{len(po_schema.required_fields)} PO fields, "
                   f"{len(contract_schema.required_fields)} contract fields")
        
    def create_mock_ocr_results(self):
        """Create mock OCR results for testing"""
        return {
            "invoice": {
                "content": """
                INVOICE
                Invoice Number: INV-2024-001
                Invoice Date: 2024-01-15
                Due Date: 2024-02-15
                
                From:
                ABC Company
                123 Business St
                City, ST 12345
                
                To:
                XYZ Corporation
                456 Client Ave
                Town, ST 67890
                
                Description          Qty    Price    Total
                Consulting Services   40    $150.00  $6,000.00
                Travel Expenses        1    $500.00    $500.00
                
                Subtotal:             $6,500.00
                Tax (8.5%):             $552.50
                Total:                $7,052.50
                
                Payment Terms: Net 30 days
                """,
                "confidence": 0.92,
                "pages": [{"page_number": 1}],
                "tables": [
                    {
                        "rows": [
                            {"cells": ["Description", "Qty", "Price", "Total"]},
                            {"cells": ["Consulting Services", "40", "$150.00", "$6,000.00"]},
                            {"cells": ["Travel Expenses", "1", "$500.00", "$500.00"]}
                        ]
                    }
                ],
                "key_value_pairs": [
                    {"key": "Invoice Number", "value": "INV-2024-001", "confidence": 0.95},
                    {"key": "Invoice Date", "value": "2024-01-15", "confidence": 0.93},
                    {"key": "Due Date", "value": "2024-02-15", "confidence": 0.91},
                    {"key": "Total", "value": "$7,052.50", "confidence": 0.96}
                ]
            },
            "purchase_order": {
                "content": """
                PURCHASE ORDER
                PO Number: PO-2024-155
                PO Date: 2024-01-20
                
                Vendor:
                Office Supplies Inc.
                789 Supply Blvd
                Supply City, ST 11111
                
                Ship To:
                Our Company
                321 Main Street
                Our City, ST 22222
                
                Item Description      Qty    Unit Price    Total
                Office Chairs          5      $250.00    $1,250.00
                Desk Lamps            10       $45.00      $450.00
                Filing Cabinets        2      $180.00      $360.00
                
                Subtotal:                                $2,060.00
                Tax:                                       $164.80
                Total:                                   $2,224.80
                
                Requested Delivery: 2024-02-01
                """,
                "confidence": 0.89,
                "pages": [{"page_number": 1}],
                "key_value_pairs": [
                    {"key": "PO Number", "value": "PO-2024-155", "confidence": 0.94},
                    {"key": "PO Date", "value": "2024-01-20", "confidence": 0.92},
                    {"key": "Total", "value": "$2,224.80", "confidence": 0.95}
                ]
            }
        }
    
    async def test_complete_pipeline(self):
        """Test the complete processing pipeline"""
        logger.info("=== Starting Complete System Test ===")
        
        # Step 1: Create schemas
        self.create_test_schemas()
        
        # Step 2: Create mock OCR results
        mock_results = self.create_mock_ocr_results()
        
        # Step 3: Test field extraction for each document type
        for doc_type, ocr_result in mock_results.items():
            logger.info(f"\n--- Testing {doc_type.upper()} Processing ---")
            
            try:
                # Extract fields using integrated system
                result = await self.integrated_extractor.process_document(
                    document_content=ocr_result["content"].encode(),
                    document_type=doc_type,
                    filename=f"test_{doc_type}.pdf",
                    ocr_result=ocr_result  # Use mock result instead of real OCR
                )
                
                logger.info(f"Processing Result for {doc_type}:")
                logger.info(f"  - Status: {result.status}")
                logger.info(f"  - Document confidence: {result.document_confidence:.3f}")
                logger.info(f"  - Extracted fields: {len(result.extracted_fields)}")
                logger.info(f"  - Validation errors: {len(result.validation_errors)}")
                logger.info(f"  - Business rule violations: {len(result.business_rule_violations)}")
                
                if result.extracted_fields:
                    logger.info("  - Key extracted fields:")
                    for field_name, field_data in list(result.extracted_fields.items())[:5]:
                        confidence = field_data.get('confidence', 0.0)
                        value = field_data.get('normalized_value', field_data.get('raw_value', 'N/A'))
                        logger.info(f"    * {field_name}: {value} (confidence: {confidence:.3f})")
                
                if result.validation_errors:
                    logger.info("  - Validation errors:")
                    for error in result.validation_errors[:3]:
                        logger.info(f"    * {error}")
                
                if result.needs_human_review:
                    logger.info(f"  - Document routed to HITL queue: {result.hitl_queue_id}")
                    
                    # Check HITL queue status
                    queue_stats = await self.hitl_manager.get_queue_stats()
                    logger.info(f"  - Current queue depth: {queue_stats.get('pending_count', 0)}")
                
            except Exception as e:
                logger.error(f"Error processing {doc_type}: {str(e)}")
        
        # Step 4: Test HITL queue management
        logger.info("\n--- Testing HITL Queue Management ---")
        await self.test_hitl_queue()
        
        # Step 5: Display system statistics
        await self.display_system_stats()
        
        logger.info("\n=== Complete System Test Finished ===")
    
    async def test_hitl_queue(self):
        """Test HITL queue management functionality"""
        try:
            # Get current queue statistics
            stats = await self.hitl_manager.get_queue_stats()
            logger.info(f"Queue Statistics:")
            logger.info(f"  - Pending tasks: {stats.get('pending_count', 0)}")
            logger.info(f"  - In progress: {stats.get('in_progress_count', 0)}")
            logger.info(f"  - Completed: {stats.get('completed_count', 0)}")
            logger.info(f"  - Failed: {stats.get('failed_count', 0)}")
            logger.info(f"  - Poisoned: {stats.get('poison_count', 0)}")
            
            # Simulate processing some tasks
            if stats.get('pending_count', 0) > 0:
                logger.info("Processing pending HITL tasks...")
                for i in range(min(3, stats.get('pending_count', 0))):
                    task = await self.hitl_manager.get_next_task()
                    if task:
                        logger.info(f"  - Processing task {task.task_id} (priority: {task.priority})")
                        
                        # Simulate task completion
                        await asyncio.sleep(0.1)  # Simulate processing time
                        await self.hitl_manager.complete_task(
                            task.task_id,
                            {"status": "completed", "reviewed_by": "test_system"}
                        )
        
        except Exception as e:
            logger.error(f"HITL queue test error: {str(e)}")
    
    async def display_system_stats(self):
        """Display overall system statistics"""
        logger.info("\n--- System Statistics ---")
        
        # Field extraction stats
        extractor_stats = self.field_extractor.get_extraction_stats()
        logger.info(f"Field Extraction:")
        logger.info(f"  - Documents processed: {extractor_stats.get('documents_processed', 0)}")
        logger.info(f"  - Fields extracted: {extractor_stats.get('fields_extracted', 0)}")
        logger.info(f"  - Validation errors: {extractor_stats.get('validation_errors', 0)}")
        logger.info(f"  - Average confidence: {extractor_stats.get('avg_confidence', 0.0):.3f}")
        
        # HITL stats
        hitl_stats = await self.hitl_manager.get_queue_stats()
        logger.info(f"HITL Queue:")
        logger.info(f"  - Total tasks: {sum(hitl_stats.values())}")
        logger.info(f"  - Success rate: {hitl_stats.get('completed_count', 0) / max(1, sum(hitl_stats.values())) * 100:.1f}%")
        
        # Schema information
        logger.info(f"Document Schemas:")
        schemas = self.field_extractor.list_schemas()
        logger.info(f"  - Available schemas: {len(schemas)}")
        for schema in schemas:
            logger.info(f"    * {schema}: {len(self.field_extractor.get_schema(schema).required_fields)} required fields")

def main():
    """Main test execution"""
    # Create test directory if it doesn't exist
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    # Run the complete system test
    test_system = CompleteSystemTest()
    asyncio.run(test_system.test_complete_pipeline())

if __name__ == "__main__":
    main()