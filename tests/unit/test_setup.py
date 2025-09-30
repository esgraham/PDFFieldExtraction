"""
Test script for Azure PDF Listener

This script helps verify that your Azure Storage configuration is working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from azure.storage.blob import BlobServiceClient
        print("✅ azure-storage-blob imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import azure-storage-blob: {e}")
        return False
    
    try:
        from azure.identity import DefaultAzureCredential
        print("✅ azure-identity imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import azure-identity: {e}")
        return False
    
    try:
        from azure_pdf_listener import AzurePDFListener
        print("✅ azure_pdf_listener imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import azure_pdf_listener: {e}")
        return False
    
    return True

def test_configuration():
    """Test if environment variables are configured."""
    print("\n🔧 Testing configuration...")
    
    required_vars = ["AZURE_STORAGE_ACCOUNT_NAME"]
    optional_vars = ["AZURE_STORAGE_CONTAINER_NAME", "AZURE_STORAGE_CONNECTION_STRING", "USE_MANAGED_IDENTITY"]
    
    config_ok = True
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set (required)")
            config_ok = False
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            if "CONNECTION_STRING" in var:
                # Don't show the full connection string for security
                print(f"✅ {var}: ***configured***")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set (optional)")
    
    # Check authentication method
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    use_managed_identity = os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
    
    if not connection_string and not use_managed_identity:
        print("❌ Either AZURE_STORAGE_CONNECTION_STRING must be set or USE_MANAGED_IDENTITY must be true")
        config_ok = False
    
    return config_ok

def test_connection():
    """Test connection to Azure Storage."""
    print("\n🔗 Testing Azure Storage connection...")
    
    try:
        from azure_pdf_listener import AzurePDFListener
        
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        use_managed_identity = os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
        
        if not storage_account:
            print("❌ AZURE_STORAGE_ACCOUNT_NAME not set")
            return False
        
        listener = AzurePDFListener(
            storage_account_name=storage_account,
            container_name=container_name,
            connection_string=connection_string,
            use_managed_identity=use_managed_identity,
            log_level="ERROR"  # Suppress info logs for test
        )
        
        # Try to list files (this will test the connection)
        files = listener.get_pdf_files()
        print(f"✅ Successfully connected to Azure Storage")
        print(f"✅ Container '{container_name}' is accessible")
        print(f"📄 Found {len(files)} PDF file(s) in the container")
        
        if files:
            print("   Sample files:")
            for i, file_info in enumerate(files[:3]):  # Show first 3 files
                print(f"   - {file_info['name']} ({file_info['size']:,} bytes)")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Azure PDF Listener Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Run: pip install -r requirements.txt")
        return 1
    
    # Test configuration
    if not test_configuration():
        print("\n❌ Configuration test failed. Check your .env file")
        return 1
    
    # Test connection
    if not test_connection():
        print("\n❌ Connection test failed. Check your Azure Storage settings")
        return 1
    
    print("\n🎉 All tests passed! Your Azure PDF Listener is ready to use.")
    print("\nNext steps:")
    print("1. Run 'python example_usage.py' to see it in action")
    print("2. Upload a PDF file to your container to test monitoring")
    print("3. Integrate the listener into your application")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())