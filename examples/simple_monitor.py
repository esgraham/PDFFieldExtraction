"""
Simple example showing basic PDF monitoring functionality.

This example demonstrates the simplest way to monitor an Azure Storage container
for new PDF files.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from azure_pdf_listener import AzurePDFListener

# Load environment variables
load_dotenv()


def simple_pdf_handler(blob_name, blob_client):
    """Simple handler that just prints information about new PDF files."""
    print(f"📄 New PDF detected: {blob_name}")
    
    # Get basic file information
    properties = blob_client.get_blob_properties()
    print(f"   Size: {properties.size:,} bytes")
    print(f"   Last modified: {properties.last_modified}")
    print()


def main():
    """Main function to demonstrate basic PDF monitoring."""
    
    # Check if required environment variables are set
    storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not storage_account or not connection_string:
        print("❌ Please set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_CONNECTION_STRING")
        print("   Copy config/.env.example to .env and fill in your details")
        return
    
    print("🚀 Simple PDF Monitor")
    print("=" * 30)
    print(f"Monitoring container: {container_name}")
    print(f"Storage account: {storage_account}")
    print()
    
    # Create the PDF listener
    listener = AzurePDFListener(
        storage_account_name=storage_account,
        container_name=container_name,
        connection_string=connection_string,
        polling_interval=15,  # Check every 15 seconds
        log_level="INFO"
    )
    
    # Set the callback function
    listener.set_pdf_callback(simple_pdf_handler)
    
    # List existing PDF files first
    print("Checking for existing PDF files...")
    existing_files = listener.get_pdf_files()
    
    if existing_files:
        print(f"Found {len(existing_files)} existing PDF file(s):")
        for file_info in existing_files:
            print(f"  📄 {file_info['name']} ({file_info['size']:,} bytes)")
    else:
        print("No existing PDF files found")
    
    print()
    print("Starting continuous monitoring...")
    print("Upload a PDF file to your container to see it detected!")
    print("Press Ctrl+C to stop")
    
    try:
        listener.start_polling()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")


if __name__ == "__main__":
    main()