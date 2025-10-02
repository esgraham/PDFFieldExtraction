"""
PDF Monitor Service

Handles monitoring Azure Storage for new PDF files and orchestrates
the processing pipeline.
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Set
from datetime import datetime

from .azure_pdf_listener import AzurePDFListener
from .pipeline_manager import PDFProcessingPipeline

logger = logging.getLogger(__name__)


class PDFMonitorService:
    """Service for monitoring and processing PDF files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the monitor service."""
        self.config = config
        self.pipeline = PDFProcessingPipeline(config)
        self.listener = None
        self.processed_files: Set[str] = set()
        
        # Initialize Azure PDF Listener
        self._initialize_listener()
    
    def _initialize_listener(self):
        """Initialize the Azure PDF listener."""
        try:
            self.listener = AzurePDFListener(
                storage_account_name=self.config['azure']['account_name'],
                connection_string=self.config['azure']['connection_string'],
                container_name=self.config['azure']['container_name'],
                use_managed_identity=self.config['azure']['enable_managed_identity'],
                polling_interval=self.config['monitoring']['polling_interval'],
                log_level=self.config['logging']['level']
            )
            logger.info("✅ Azure PDF Listener initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure PDF Listener: {e}")
            raise
    
    async def run_monitor_mode(self):
        """Run continuous monitoring mode with async polling."""
        logger.info("🔄 Starting PDF monitoring service...")
        
        if not self.listener:
            raise RuntimeError("PDF Listener not initialized")
        
        # Processing queue and task management
        pending_tasks = []
        polling_interval = self.config.get('polling_interval', 30)
        
        try:
            while True:
                await self._run_polling_cycle(pending_tasks)
                
                # Wait before next poll
                try:
                    await asyncio.sleep(polling_interval)
                except asyncio.CancelledError:
                    logger.info("🛑 Monitor mode cancelled by user")
                    break
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitor mode interrupted by user")
        except Exception as e:
            logger.error(f"❌ Monitor mode failed: {e}")
            raise
        finally:
            # Wait for any pending tasks to complete
            if pending_tasks:
                logger.info(f"⏳ Waiting for {len(pending_tasks)} pending tasks to complete...")
                await asyncio.gather(*pending_tasks, return_exceptions=True)
    
    async def _run_polling_cycle(self, pending_tasks: List):
        """Run a single polling cycle."""
        try:
            # Clean up completed tasks
            pending_tasks[:] = [task for task in pending_tasks if not task.done()]
            
            # Get new PDF files
            pdf_files = await asyncio.get_event_loop().run_in_executor(
                None, self.listener.get_pdf_files
            )
            
            # Process new files
            new_files = [f for f in pdf_files if f['name'] not in self.processed_files]
            
            if new_files:
                logger.info(f"📂 Found {len(new_files)} new PDF files to process")
                
                for file_info in new_files:
                    blob_name = file_info['name']
                    blob_client = self.listener.get_blob_client(blob_name)
                    
                    # Create processing task
                    task = asyncio.create_task(
                        self._process_and_display_pdf(blob_name, blob_client)
                    )
                    pending_tasks.append(task)
                    
                    # Mark as being processed
                    self.processed_files.add(blob_name)
            else:
                logger.info("📂 No new PDF files found")
                
        except Exception as e:
            logger.error(f"❌ Polling cycle failed: {e}")
    
    async def _process_and_display_pdf(self, blob_name: str, blob_client):
        """Process a PDF and display results."""
        try:
            logger.info(f"🔄 Processing: {blob_name}")
            
            # Process the PDF through the pipeline
            result = await self.pipeline.process_pdf(blob_name, blob_client)
            
            # Display results
            self._display_processing_result(result)
            
        except Exception as e:
            logger.error(f"❌ Failed to process {blob_name}: {e}")
    
    def _display_processing_result(self, result: Dict[str, Any]):
        """Display processing results in a formatted way."""
        blob_name = result['blob_name']
        summary = result['summary']
        
        logger.info(f"📋 Processing Summary for {blob_name}:")
        logger.info(f"   ⏱️  Processing Time: {summary.get('processing_time', 0):.2f}s")
        logger.info(f"   ✅ Completed Stages: {summary.get('completed_stages', 0)}/{summary.get('total_stages', 0)}")
        logger.info(f"   📊 Success Rate: {summary.get('success_rate', 0):.1%}")
        logger.info(f"   🎯 Quality Score: {summary.get('data_quality_score', 0):.2f}")
        logger.info(f"   👤 Needs Review: {'Yes' if result.get('needs_human_review') else 'No'}")
        
        # Show extracted data summary
        extracted_data = result.get('extracted_data', {})
        if extracted_data:
            logger.info(f"   📄 Extracted Fields: {len(extracted_data)} fields")
            doc_type = extracted_data.get('document_type', 'unknown')
            logger.info(f"   🏷️  Document Type: {doc_type}")
        
        # Show validation summary
        validation_results = result.get('validation_results', [])
        if validation_results:
            passed = len([r for r in validation_results if r.get('valid')])
            total = len(validation_results)
            logger.info(f"   ✔️  Validation: {passed}/{total} passed")
        
        logger.info(f"" + "="*50)
    
    async def process_single_file(self, filename: str) -> Dict[str, Any]:
        """Process a single specific file."""
        try:
            blob_client = self.listener.get_blob_client(filename)
            result = await self.pipeline.process_pdf(filename, blob_client)
            return result
        except Exception as e:
            logger.error(f"❌ Failed to process single file {filename}: {e}")
            raise
    
    async def process_batch(self) -> List[Dict[str, Any]]:
        """Process all PDF files in batch mode."""
        logger.info("🔄 Starting batch processing...")
        
        try:
            # Get all PDF files
            pdf_files = await asyncio.get_event_loop().run_in_executor(
                None, self.listener.get_pdf_files
            )
            
            if not pdf_files:
                logger.info("📂 No PDF files found for batch processing")
                return []
            
            logger.info(f"📂 Found {len(pdf_files)} PDF files for batch processing")
            
            # Process all files concurrently (with some limit)
            semaphore = asyncio.Semaphore(self.config.get('max_concurrent_processing', 3))
            
            async def process_with_semaphore(file_info):
                async with semaphore:
                    blob_name = file_info['name']
                    blob_client = self.listener.get_blob_client(blob_name)
                    return await self.pipeline.process_pdf(blob_name, blob_client)
            
            # Process all files
            tasks = [process_with_semaphore(file_info) for file_info in pdf_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return successful results
            successful_results = [r for r in results if isinstance(r, dict)]
            failed_count = len(results) - len(successful_results)
            
            logger.info(f"✅ Batch processing completed: {len(successful_results)} successful, {failed_count} failed")
            return successful_results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            raise