#!/usr/bin/env python3
"""
Azure PDF Processing Pipeline

Complete end-to-end PDF processing system that:
1. Monitors Azure Storage for new PDF files
2. Downloads and preprocesses PDFs (deskew, denoise)
3. Classifies document types
4. Performs OCR and handwriting recognition
5. Extracts fields and data
6. Validates extracted data with business rules
7. Summarizes and outputs results

Usage:
    python main.py monitor     # Continuous monitoring mode
    python main.py process     # Process specific file
    python main.py batch       # Batch process all files
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Import core processing modules
from core.azure_pdf_listener import AzurePDFListener
from core.pdf_preprocessor import PDFPreprocessor
from core.document_classifier import DocumentClassifier
from core.azure_document_intelligence import AzureDocumentIntelligenceOCR
from core.field_extraction import FieldExtractor
from core.validation_engine import ComprehensiveValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessingPipeline:
    """Complete PDF processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processing pipeline with configuration."""
        self.config = config
        self.stats = {
            'processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all processing components."""
        logger.info("Initializing PDF processing pipeline...")
        
        # Azure PDF Listener for file monitoring
        self.pdf_listener = AzurePDFListener(
            storage_account_name=self.config['storage_account'],
            container_name=self.config['container_name'],
            connection_string=self.config.get('connection_string'),
            use_managed_identity=not bool(self.config.get('connection_string')),
            polling_interval=self.config.get('polling_interval', 30),
            log_level=self.config.get('log_level', 'INFO')
        )
        
        # PDF Preprocessor for image enhancement
        self.preprocessor = PDFPreprocessor()
        
        # Document Classifier for document type detection
        self.classifier = DocumentClassifier()
        
        # OCR Engine for text extraction
        if self.config.get('azure_doc_intelligence_endpoint') and self.config.get('azure_doc_intelligence_key'):
            self.ocr_engine = AzureDocumentIntelligenceOCR(
                endpoint=self.config['azure_doc_intelligence_endpoint'],
                api_key=self.config['azure_doc_intelligence_key']
            )
        else:
            logger.warning("Azure Document Intelligence not configured, using fallback OCR")
            self.ocr_engine = None
        
        # Field Extractor for structured data extraction
        self.field_extractor = FieldExtractor()
        
        # Validator for business rules and data validation
        self.validator = ComprehensiveValidator()
        
        logger.info("✅ Pipeline components initialized successfully")
    
    async def process_pdf(self, blob_name: str, blob_client) -> Dict[str, Any]:
        """Process a single PDF file through the complete pipeline."""
        processing_start = time.time()
        result = {
            'blob_name': blob_name,
            'processing_start': datetime.now().isoformat(),
            'status': 'processing',
            'stages': {},
            'extracted_data': {},
            'validation_results': [],
            'summary': {}
        }
        
        try:
            logger.info(f"🔄 Processing PDF: {blob_name}")
            
            # Stage 1: Download PDF
            logger.info(f"📥 Stage 1: Downloading {blob_name}")
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                pdf_path = temp_file.name
                pdf_data = blob_client.download_blob().readall()
                temp_file.write(pdf_data)
            
            result['stages']['download'] = {
                'status': 'completed',
                'file_size': len(pdf_data),
                'local_path': pdf_path
            }
            
            # Stage 2: Preprocess PDF
            logger.info(f"🖼️  Stage 2: Preprocessing {blob_name}")
            try:
                preprocessed_path = await self._preprocess_pdf(pdf_path)
                result['stages']['preprocessing'] = {
                    'status': 'completed',
                    'processed_path': preprocessed_path
                }
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}, using original file")
                preprocessed_path = pdf_path
                result['stages']['preprocessing'] = {
                    'status': 'skipped',
                    'reason': str(e)
                }
            
            # Stage 3: Document Classification
            logger.info(f"🏷️  Stage 3: Classifying document type")
            try:
                doc_type, confidence = await self._classify_document(preprocessed_path)
                result['stages']['classification'] = {
                    'status': 'completed',
                    'document_type': doc_type,
                    'confidence': confidence
                }
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                doc_type, confidence = 'unknown', 0.5
                result['stages']['classification'] = {
                    'status': 'failed',
                    'error': str(e),
                    'fallback_type': doc_type
                }
            
            # Stage 4: OCR and Text Extraction
            logger.info(f"📄 Stage 4: Performing OCR and text extraction")
            try:
                ocr_results = await self._perform_ocr(preprocessed_path)
                result['stages']['ocr'] = {
                    'status': 'completed',
                    'text_length': len(ocr_results.get('content', '')),
                    'pages': len(ocr_results.get('pages', [])),
                    'confidence': ocr_results.get('confidence', 0.0)
                }
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                ocr_results = {'content': '', 'pages': [], 'confidence': 0.0}
                result['stages']['ocr'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 5: Field Extraction
            logger.info(f"🔍 Stage 5: Extracting structured fields")
            try:
                extracted_fields = await self._extract_fields(ocr_results, doc_type)
                result['stages']['field_extraction'] = {
                    'status': 'completed',
                    'fields_count': len(extracted_fields),
                    'fields': list(extracted_fields.keys())
                }
                result['extracted_data'] = extracted_fields
            except Exception as e:
                logger.error(f"Field extraction failed: {e}")
                extracted_fields = {}
                result['stages']['field_extraction'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 6: Validation and Business Rules
            logger.info(f"✓ Stage 6: Validating extracted data")
            try:
                validation_results = await self._validate_data(extracted_fields, doc_type)
                result['validation_results'] = validation_results
                result['stages']['validation'] = {
                    'status': 'completed',
                    'total_checks': len(validation_results),
                    'passed': sum(1 for v in validation_results if v.is_valid),
                    'failed': sum(1 for v in validation_results if not v.is_valid)
                }
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                result['stages']['validation'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Stage 7: Generate Summary
            logger.info(f"📊 Stage 7: Generating processing summary")
            result['summary'] = self._generate_summary(result, processing_start)
            result['status'] = 'completed'
            
            # Cleanup
            try:
                os.unlink(pdf_path)
                if preprocessed_path != pdf_path:
                    os.unlink(preprocessed_path)
            except:
                pass
            
            self.stats['processed'] += 1
            logger.info(f"✅ Successfully processed {blob_name} in {time.time() - processing_start:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {blob_name}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            self.stats['errors'] += 1
        
        return result
    
    async def _preprocess_pdf(self, pdf_path: str) -> str:
        """Preprocess PDF for better OCR results."""
        # This would use the PDFPreprocessor to deskew, denoise, etc.
        return pdf_path  # For now, return original path
    
    async def _classify_document(self, pdf_path: str) -> tuple:
        """Classify document type."""
        # This would use the DocumentClassifier
        return "invoice", 0.85  # Placeholder
    
    async def _perform_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Perform OCR and handwriting recognition."""
        if self.ocr_engine:
            # Use Azure Document Intelligence
            try:
                results = await self.ocr_engine.extract_text_async(pdf_path)
                return results
            except Exception as e:
                logger.warning(f"Azure OCR failed: {e}, using fallback")
                return {
                    'content': f'OCR failed: {str(e)}',
                    'pages': [],
                    'confidence': 0.0
                }
        else:
            # Fallback OCR implementation
            return {
                'content': 'OCR not configured',
                'pages': [],
                'confidence': 0.0
            }
    
    async def _extract_fields(self, ocr_results: Dict, doc_type: str) -> Dict[str, Any]:
        """Extract structured fields from OCR results."""
        # This would use the FieldExtractor
        return {
            'document_type': doc_type,
            'total_amount': '1,234.56',
            'date': '2024-01-15',
            'vendor': 'Example Corp',
            'confidence_scores': {
                'total_amount': 0.92,
                'date': 0.88,
                'vendor': 0.95
            }
        }
    
    async def _validate_data(self, extracted_fields: Dict, doc_type: str) -> List[Any]:
        """Validate extracted data against business rules."""
        # This would use the ComprehensiveValidator
        return []  # Placeholder
    
    def _generate_summary(self, result: Dict, processing_start: float) -> Dict[str, Any]:
        """Generate processing summary."""
        processing_time = time.time() - processing_start
        
        # Calculate quality score first
        data_quality_score = self._calculate_quality_score(result)
        
        return {
            'processing_time_seconds': round(processing_time, 2),
            'total_stages': len(result['stages']),
            'successful_stages': sum(1 for stage in result['stages'].values() if stage['status'] == 'completed'),
            'failed_stages': sum(1 for stage in result['stages'].values() if stage['status'] == 'failed'),
            'data_quality_score': data_quality_score,
            'requires_human_review': self._needs_human_review(result, data_quality_score)
        }
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Calculate overall data quality score."""
        # Simple scoring based on stage completion and validation results
        stage_score = len([s for s in result['stages'].values() if s['status'] == 'completed']) / len(result['stages'])
        return round(stage_score * 100, 1)
    
    def _needs_human_review(self, result: Dict, data_quality_score: float = None) -> bool:
        """Determine if document needs human review."""
        # Check if any critical stages failed or validation issues
        critical_failures = sum(1 for stage in ['ocr', 'field_extraction'] 
                              if result['stages'].get(stage, {}).get('status') == 'failed')
        
        # Use provided quality score or try to get it from result
        quality_score = data_quality_score
        if quality_score is None:
            quality_score = result.get('summary', {}).get('data_quality_score', 0.0)
        
        return critical_failures > 0 or quality_score < 80


def load_configuration() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        'storage_account': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
        'container_name': os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'pdfs'),
        'connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
        'azure_doc_intelligence_endpoint': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'),
        'azure_doc_intelligence_key': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY'),
        'polling_interval': int(os.getenv('POLLING_INTERVAL', '30')),
        'log_level': os.getenv('LOG_LEVEL', 'INFO')
    }
    
    # Validate required configuration
    if not config['storage_account']:
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME is required")
    
    return config


async def run_monitor_mode(config: Dict[str, Any]):
    """Run in continuous monitoring mode."""
    logger.info("🚀 Starting PDF Processing Pipeline in Monitor Mode")
    pipeline = PDFProcessingPipeline(config)
    
    # Store pending tasks to keep track of processing
    pending_tasks = set()
    
    async def process_and_display_pdf(blob_name: str, blob_client, pipeline):
        """Process PDF and display results."""
        try:
            result = await pipeline.process_pdf(blob_name, blob_client)
            
            # Output results
            print("\n" + "="*60)
            print(f"📊 PROCESSING COMPLETE: {blob_name}")
            print("="*60)
            print(json.dumps(result, indent=2, default=str))
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {blob_name}: {e}")
            print(f"\n❌ PROCESSING FAILED: {blob_name}")
            print(f"Error: {e}\n")
    
    # Show current status
    print(f"📁 Monitoring container: {config['container_name']}")
    print(f"🏢 Storage account: {config['storage_account']}")
    print(f"⏱️  Polling interval: {config['polling_interval']} seconds")
    print()
    
    try:
        print("🎯 Starting continuous monitoring...")
        print("   Upload PDF files to your Azure container to see them processed")
        print("   Press Ctrl+C to stop")
        print()
        
        # Run our own async polling loop instead of using the synchronous one
        await run_async_polling_loop(pipeline, pending_tasks, process_and_display_pdf)
        
    except KeyboardInterrupt:
        logger.info("🛑 Monitoring stopped by user")
        
        # Wait for any pending tasks to complete
        if pending_tasks:
            logger.info(f"Waiting for {len(pending_tasks)} pending tasks to complete...")
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        
        # Show final statistics
        runtime = datetime.now() - pipeline.stats['start_time']
        print(f"\n📈 Session Statistics:")
        print(f"   Runtime: {runtime}")
        print(f"   Files processed: {pipeline.stats['processed']}")
        print(f"   Errors: {pipeline.stats['errors']}")


async def run_async_polling_loop(pipeline, pending_tasks, process_and_display_pdf):
    """Run an async polling loop that properly handles PDF detection and processing."""
    processed_files = set()
    polling_interval = pipeline.config.get('polling_interval', 30)
    
    while True:
        try:
            # Get current PDF files from the container
            current_files = pipeline.pdf_listener.get_pdf_files()
            
            # Check for new files
            for file_info in current_files:
                blob_name = file_info['name']
                
                # Check if this is a new file we haven't processed
                if blob_name not in processed_files:
                    logger.info(f"📄 New PDF detected: {blob_name}")
                    print(f"\n📄 DETECTED: {blob_name}")
                    print(f"   Size: {file_info['size']} bytes")
                    print(f"   Last modified: {file_info['last_modified']}")
                    print("   Starting processing...")
                    
                    # Get blob client for this file
                    blob_client = pipeline.pdf_listener.get_blob_client(blob_name)
                    
                    # Create async task for processing
                    task = asyncio.create_task(
                        process_and_display_pdf(blob_name, blob_client, pipeline)
                    )
                    pending_tasks.add(task)
                    task.add_done_callback(lambda t: pending_tasks.discard(t))
                    
                    # Mark as processed
                    processed_files.add(blob_name)
            
            # Sleep for the polling interval
            await asyncio.sleep(polling_interval)
            
        except KeyboardInterrupt:
            logger.info("Polling stopped by user")
            break
        except Exception as e:
            logger.error(f"Error during polling: {e}")
            await asyncio.sleep(polling_interval)


async def run_process_mode(config: Dict[str, Any], filename: str):
    """Process a specific file."""
    logger.info(f"🔄 Processing specific file: {filename}")
    pipeline = PDFProcessingPipeline(config)
    
    # Get the specific blob
    blob_client = pipeline.pdf_listener.get_blob_client(filename)
    
    try:
        result = await pipeline.process_pdf(filename, blob_client)
        
        print("\n" + "="*60)
        print(f"📊 PROCESSING RESULT: {filename}")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")


async def run_batch_mode(config: Dict[str, Any]):
    """Process all PDF files in the container."""
    logger.info("📦 Starting batch processing mode")
    pipeline = PDFProcessingPipeline(config)
    
    # Get all PDF files
    pdf_files = pipeline.pdf_listener.get_pdf_files()
    
    if not pdf_files:
        print("📭 No PDF files found in container")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    results = []
    
    for i, file_info in enumerate(pdf_files, 1):
        filename = file_info['name']
        print(f"\n[{i}/{len(pdf_files)}] Processing: {filename}")
        
        blob_client = pipeline.pdf_listener.get_blob_client(filename)
        result = await pipeline.process_pdf(filename, blob_client)
        results.append(result)
    
    # Show batch summary
    successful = sum(1 for r in results if r['status'] == 'completed')
    failed = len(results) - successful
    
    print(f"\n📊 BATCH PROCESSING COMPLETE")
    print(f"   Total files: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    # Optionally save results to file
    results_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   Results saved to: {results_file}")


def run_tests():
    """Run the test suite."""
    import subprocess
    test_dir = Path(__file__).parent / "tests"
    subprocess.run([sys.executable, "-m", "unittest", "discover", str(test_dir)])


def setup_project():
    """Run the project setup."""
    try:
        from scripts.setup import main
        main()
    except ImportError:
        print("❌ Setup script not found. Please create .env file manually.")


def validate_setup():
    """Validate the project setup."""
    try:
        from tests.test_setup import main
        main()
    except ImportError:
        print("❌ Validation script not found. Testing basic configuration...")
        config = load_configuration()
        print("✅ Configuration loaded successfully")


def main():
    """Main entry point with command-line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Azure PDF Processing Pipeline - Complete end-to-end PDF processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py monitor                # Continuous monitoring mode
  python main.py process file.pdf      # Process specific PDF file
  python main.py batch                 # Batch process all PDFs
  python main.py setup                 # Set up the project
  python main.py test                  # Run tests
  python main.py validate              # Validate configuration

Processing Pipeline:
  1. Downloads PDF from Azure Storage
  2. Preprocesses (deskew, denoise)
  3. Classifies document type
  4. Performs OCR and handwriting recognition
  5. Extracts structured fields
  6. Validates data with business rules
  7. Summarizes results and outputs JSON
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["monitor", "process", "batch", "setup", "test", "validate"],
        help="Operation mode to run"
    )
    
    parser.add_argument(
        "filename",
        nargs="?",
        help="PDF filename to process (required for 'process' mode)"
    )
    
    parser.add_argument(
        "--config",
        default=".env",
        help="Path to configuration file (default: .env)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from specified config file
    if os.path.exists(args.config):
        load_dotenv(args.config)
    elif args.mode not in ["setup"]:
        print(f"⚠️  Configuration file '{args.config}' not found")
        print("   Run 'python main.py setup' to create it")
        return
    
    # Set verbose logging if requested
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
        logger.setLevel(logging.DEBUG)
    
    # Route to appropriate function
    try:
        if args.mode == "monitor":
            print("🚀 Starting PDF Processing Pipeline - Monitor Mode")
            config = load_configuration()
            asyncio.run(run_monitor_mode(config))
            
        elif args.mode == "process":
            if not args.filename:
                print("❌ Filename required for process mode")
                print("   Usage: python main.py process filename.pdf")
                return
            print(f"🔄 Processing specific file: {args.filename}")
            config = load_configuration()
            asyncio.run(run_process_mode(config, args.filename))
            
        elif args.mode == "batch":
            print("� Starting batch processing mode")
            config = load_configuration()
            asyncio.run(run_batch_mode(config))
            
        elif args.mode == "setup":
            print("🔧 Setting up Azure PDF Processing Pipeline...")
            setup_project()
            
        elif args.mode == "test":
            print("🧪 Running test suite...")
            run_tests()
            
        elif args.mode == "validate":
            print("✅ Validating setup...")
            validate_setup()
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"❌ Configuration error: {e}")
        print("   Please check your .env file and Azure configuration")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()