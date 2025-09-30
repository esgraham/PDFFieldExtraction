"""
Advanced example showing PDF processing with download and metadata extraction.

This example demonstrates more advanced features like downloading PDFs,
extracting metadata, and processing files.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from azure_pdf_listener import AzurePDFListener

# Load environment variables
load_dotenv()


class PDFProcessor:
    """Advanced PDF processor with download and metadata extraction."""
    
    def __init__(self, download_dir="downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.processed_count = 0
    
    def process_pdf(self, blob_name, blob_client):
        """Advanced PDF processing with download and metadata extraction."""
        self.processed_count += 1
        
        print(f"🔍 Processing PDF #{self.processed_count}: {blob_name}")
        print("=" * 50)
        
        try:
            # Get blob properties and metadata
            properties = blob_client.get_blob_properties()
            
            # Display file information
            print(f"📄 File Information:")
            print(f"   Name: {blob_name}")
            print(f"   Size: {properties.size:,} bytes ({self._format_size(properties.size)})")
            print(f"   Content Type: {properties.content_settings.content_type}")
            print(f"   Created: {properties.creation_time}")
            print(f"   Modified: {properties.last_modified}")
            print(f"   ETag: {properties.etag}")
            
            # Display custom metadata if any
            if properties.metadata:
                print(f"   Metadata: {properties.metadata}")
            
            # Download the file
            local_path = self.download_dir / blob_name
            
            print(f"\n💾 Downloading to: {local_path}")
            
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            print(f"✅ Download completed")
            
            # Perform additional processing (placeholder for real processing)
            self._perform_additional_processing(local_path, properties)
            
            print(f"✅ Processing completed for {blob_name}")
            
        except Exception as e:
            print(f"❌ Error processing {blob_name}: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    def _format_size(self, size_bytes):
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
    
    def _perform_additional_processing(self, file_path, properties):
        """Placeholder for additional PDF processing."""
        print(f"🔧 Additional Processing:")
        
        # File validation
        if properties.size > 10 * 1024 * 1024:  # 10MB
            print(f"   ⚠️  Large file detected ({self._format_size(properties.size)})")
        
        # Check if file is very recent
        if properties.creation_time:
            time_diff = datetime.now(properties.creation_time.tzinfo) - properties.creation_time
            if time_diff.total_seconds() < 300:  # 5 minutes
                print(f"   🕐 Recently created file (within 5 minutes)")
        
        # Here you could add:
        # - PDF text extraction using PyPDF2, pdfplumber, or pymupdf
        # - Form field extraction
        # - OCR processing for scanned documents
        # - Database logging
        # - Email notifications
        # - File classification
        # - Security scanning
        
        print(f"   📊 Processing logic would go here...")


def main():
    """Main function to demonstrate advanced PDF processing."""
    
    # Check if required environment variables are set
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not storage_account or not connection_string:
        print("❌ Please set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_CONNECTION_STRING")
        print("   Copy config/.env.example to .env and fill in your details")
        return
    
    print("🚀 Advanced PDF Processor")
    print("=" * 40)
    print(f"Storage Account: {storage_account}")
    print(f"Container: {container_name}")
    print(f"Download Directory: downloads/")
    print()
    
    # Create PDF processor
    processor = PDFProcessor(download_dir="downloads")
    
    # Create the PDF listener
    listener = AzurePDFListener(
        storage_account_name=storage_account,
        container_name=container_name,
        connection_string=connection_string,
        polling_interval=20,  # Check every 20 seconds
        log_level="INFO"
    )
    
    # Set the advanced callback function
    listener.set_pdf_callback(processor.process_pdf)
    
    # Show existing files
    existing_files = listener.get_pdf_files()
    print(f"📋 Found {len(existing_files)} existing PDF file(s) in container")
    
    if existing_files:
        choice = input("Process existing files? (y/n): ").lower().strip()
        if choice.startswith('y'):
            print("\n🔄 Processing existing files...")
            for file_info in existing_files:
                blob_client = listener.get_blob_client(file_info['name'])
                processor.process_pdf(file_info['name'], blob_client)
    
    print("🔄 Starting continuous monitoring for new files...")
    print("Upload a PDF file to your container to see advanced processing!")
    print("Press Ctrl+C to stop")
    
    try:
        listener.start_polling()
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped. Processed {processor.processed_count} files total.")


if __name__ == "__main__":
    main()