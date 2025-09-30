"""
Example usage of the Azure PDF Listener class.

This example shows different ways to use the AzurePDFListener class
to monitor Azure Storage containers for new PDF files.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from azure_pdf_listener import AzurePDFListener
from azure.storage.blob import BlobClient

# Load environment variables from .env file
load_dotenv()


def advanced_pdf_callback(blob_name: str, blob_client: BlobClient):
    """
    Advanced callback function for handling new PDF files.
    
    This example shows how you might process PDF files when they arrive.
    """
    print(f"🔍 New PDF detected: {blob_name}")
    
    try:
        # Get blob properties
        properties = blob_client.get_blob_properties()
        
        print(f"📄 File size: {properties.size:,} bytes")
        print(f"📅 Last modified: {properties.last_modified}")
        print(f"🏷️  Content type: {properties.content_settings.content_type}")
        
        # Download the file to local directory
        local_dir = "downloaded_pdfs"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, blob_name)
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        print(f"💾 Downloaded to: {local_path}")
        
        # Here you could add your PDF processing logic:
        # - Extract text using PyPDF2, pdfplumber, or pymupdf
        # - Extract form fields
        # - Perform OCR if needed
        # - Save metadata to database
        # - Trigger other workflows
        
        # Example: Basic file validation
        if properties.size > 50 * 1024 * 1024:  # 50MB
            print("⚠️  Large file detected - consider async processing")
        
        print(f"✅ Successfully processed: {blob_name}\n")
        
    except Exception as e:
        print(f"❌ Error processing {blob_name}: {e}")


def simple_callback(blob_name: str, blob_client: BlobClient):
    """Simple callback that just logs new PDF files."""
    print(f"New PDF file: {blob_name}")


def example_1_basic_polling():
    """Example 1: Basic polling with simple callback."""
    print("=== Example 1: Basic Polling ===")
    
    listener = AzurePDFListener(
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs"),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        polling_interval=10,  # Check every 10 seconds for demo
        log_level="INFO"
    )
    
    listener.set_pdf_callback(simple_callback)
    
    # Check once for existing files
    print("Checking for existing PDF files...")
    listener.start_polling(run_once=True)


def example_2_continuous_monitoring():
    """Example 2: Continuous monitoring with advanced callback."""
    print("\n=== Example 2: Continuous Monitoring ===")
    
    listener = AzurePDFListener(
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs"),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        polling_interval=30,
        log_level="INFO"
    )
    
    listener.set_pdf_callback(advanced_pdf_callback)
    
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        listener.start_polling()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")


def example_3_managed_identity():
    """Example 3: Using Azure Managed Identity (for production)."""
    print("\n=== Example 3: Managed Identity ===")
    
    if os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true":
        listener = AzurePDFListener(
            storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
            container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs"),
            use_managed_identity=True,
            polling_interval=30,
            log_level="INFO"
        )
        
        listener.set_pdf_callback(simple_callback)
        listener.start_polling(run_once=True)
    else:
        print("Managed identity not enabled in configuration")


def example_4_list_existing_files():
    """Example 4: List all existing PDF files in the container."""
    print("\n=== Example 4: List Existing Files ===")
    
    listener = AzurePDFListener(
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        container_name=os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs"),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    
    try:
        pdf_files = listener.get_pdf_files()
        
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF file(s):")
            for file_info in pdf_files:
                print(f"  📄 {file_info['name']}")
                print(f"     Size: {file_info['size']:,} bytes")
                print(f"     Modified: {file_info['last_modified']}")
                print()
        else:
            print("No PDF files found in the container")
            
    except Exception as e:
        print(f"Error listing files: {e}")


def main():
    """Main function to run examples."""
    
    # Check if required environment variables are set
    required_vars = ["AZURE_STORAGE_ACCOUNT_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please copy .env.example to .env and fill in your Azure Storage details")
        return
    
    if not os.getenv("AZURE_STORAGE_CONNECTION_STRING") and os.getenv("USE_MANAGED_IDENTITY", "false").lower() != "true":
        print("❌ Either AZURE_STORAGE_CONNECTION_STRING must be set or USE_MANAGED_IDENTITY must be true")
        return
    
    print("🚀 Azure PDF Listener Examples")
    print("=" * 50)
    
    # Run examples
    example_4_list_existing_files()
    example_1_basic_polling()
    
    # Ask user if they want to start continuous monitoring
    response = input("\nDo you want to start continuous monitoring? (y/n): ")
    if response.lower().startswith('y'):
        example_2_continuous_monitoring()


if __name__ == "__main__":
    main()