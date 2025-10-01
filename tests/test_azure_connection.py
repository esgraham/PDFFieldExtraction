#!/usr/bin/env python3
"""
Azure Connection Test Script

This script helps validate your Azure Storage setup and authentication.
Run this script to test your connection before running the main application.
"""

import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Import Azure libraries directly first
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import AzureError
    print("✅ Azure libraries imported successfully")
    
    # Then try to import our module (temporarily skip problematic imports)
    try:
        from core.azure_pdf_listener import AzurePDFListener
        print("✅ PDF Listener imported successfully")
    except Exception as e:
        print(f"⚠️ PDF Listener import failed: {e}")
        print("   We'll create a minimal test instead")
        AzurePDFListener = None
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    if "libGL" in str(e):
        print("\n🔧 OpenGL library issue detected.")
        print("   Try: sudo apt-get install -y libgl1-mesa-glx")
        print("   This is common in headless environments.")
    else:
        print("Make sure you've installed requirements: pip install -r requirements.txt")
    sys.exit(1)


def test_azure_connection():
    """Test Azure Storage connection with current configuration."""
    
    print("🔍 Testing Azure Storage Connection...")
    print("=" * 50)
    
    # Get configuration from environment
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    print(f"Storage Account: {storage_account_name}")
    print(f"Container Name: {container_name}")
    print(f"Connection String: {'✅ Provided' if connection_string else '❌ Not provided'}")
    
    if not storage_account_name:
        print("\n❌ AZURE_STORAGE_ACCOUNT_NAME not set in environment")
        print("Set it in your .env file or environment variables")
        return False
    
    print("\n📋 Testing Authentication Methods...")
    print("-" * 30)
    
    # Use our custom listener if available, otherwise direct Azure SDK
    if AzurePDFListener:
        return test_with_custom_listener(storage_account_name, container_name, connection_string)
    else:
        return test_with_direct_azure_sdk(storage_account_name, container_name, connection_string)

def test_with_direct_azure_sdk(storage_account_name, container_name, connection_string):
    """Test Azure connection using direct Azure SDK."""
    print("🔧 Using direct Azure SDK (bypassing custom listener)")
    
    # Test connection string first
    if connection_string:
        print("1️⃣  Testing Connection String authentication...")
        try:
            from azure.storage.blob import BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            
            # List blobs to test connection
            blobs = list(container_client.list_blobs())
            pdf_files = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
            
            print(f"✅ Connection String works! Found {len(pdf_files)} PDF files")
            return True
            
        except Exception as e:
            if "Authorization with Shared Key is disabled" in str(e):
                print("⚠️  Shared Key authorization is disabled")
                print("   Trying Azure Identity...")
            else:
                print(f"❌ Connection String failed: {e}")
                return False
    
    # Test Azure Identity
    print("2️⃣  Testing Azure Identity authentication...")
    try:
        from azure.storage.blob import BlobServiceClient
        credential = DefaultAzureCredential()
        account_url = f"https://{storage_account_name}.blob.core.windows.net"
        
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        container_client = blob_service_client.get_container_client(container_name)
        
        # List blobs to test connection
        blobs = list(container_client.list_blobs())
        pdf_files = [blob for blob in blobs if blob.name.lower().endswith('.pdf')]
        
        print(f"✅ Azure Identity works! Found {len(pdf_files)} PDF files")
        return True
        
    except Exception as e:
        print(f"❌ Azure Identity failed: {e}")
        
        if "No credential in DefaultAzureCredential chain succeeded" in str(e):
            print("\n💡 Azure Identity Setup Required:")
            print("   Run: az login")
            print("   Or install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        elif "AuthorizationPermissionMismatch" in str(e) or "not authorized to perform this operation" in str(e):
            print("\n💡 Permission Issue Detected:")
            print("   Your user account needs storage permissions.")
            print("   Solutions:")
            print("   1. Go to Azure Portal → Storage Account → Access Control (IAM)")
            print("   2. Add role: 'Storage Blob Data Contributor' to your user")
            print("   3. Or use connection string authentication instead")
            print("   4. Check you're in the correct Azure subscription: az account show")
        
        return False

def test_with_custom_listener(storage_account_name, container_name, connection_string):
    """Test Azure connection using our custom PDF listener."""
    # Test connection string first
    if connection_string:
        print("1️⃣  Testing Connection String authentication...")
        try:
            listener = AzurePDFListener(
                storage_account_name=storage_account_name,
                container_name=container_name,
                connection_string=connection_string,
                use_managed_identity=False
            )
            pdf_files = listener.get_pdf_files()
            print(f"✅ Connection String works! Found {len(pdf_files)} PDF files")
            return True
            
        except Exception as e:
            if "Authorization with Shared Key is disabled" in str(e):
                print("⚠️  Shared Key authorization is disabled")
                print("   Trying Azure Identity...")
            else:
                print(f"❌ Connection String failed: {e}")
                return False
    
    # Test Azure Identity
    print("2️⃣  Testing Azure Identity authentication...")
    try:
        # Test if Azure CLI is logged in
        credential = DefaultAzureCredential()
        print("   Checking Azure CLI login...")
        
        listener = AzurePDFListener(
            storage_account_name=storage_account_name,
            container_name=container_name,
            use_managed_identity=True
        )
        pdf_files = listener.get_pdf_files()
        print(f"✅ Azure Identity works! Found {len(pdf_files)} PDF files")
        return True
        
    except Exception as e:
        print(f"❌ Azure Identity failed: {e}")
        
        if "No credential in DefaultAzureCredential chain succeeded" in str(e):
            print("\n💡 Azure Identity Setup Required:")
            print("   Run: az login")
            print("   Or install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
        elif "AuthorizationPermissionMismatch" in str(e) or "not authorized to perform this operation" in str(e):
            print("\n💡 Permission Issue Detected:")
            print("   Your user account needs storage permissions.")
            print("   Solutions:")
            print("   1. Go to Azure Portal → Storage Account → Access Control (IAM)")
            print("   2. Add role: 'Storage Blob Data Contributor' to your user")
            print("   3. Or use connection string authentication instead")
            print("   4. Check you're in the correct Azure subscription: az account show")
        
        return False


def test_container_access():
    """Test if we can access and list the container."""
    
    print("\n🗂️  Testing Container Access...")
    print("-" * 30)
    
    try:
        storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        # Use custom listener if available, otherwise direct SDK
        if AzurePDFListener:
            # Try to create listener (it will auto-detect auth method)
            listener = AzurePDFListener(
                storage_account_name=storage_account_name,
                container_name=container_name,
                connection_string=connection_string
            )
            pdf_files = listener.get_pdf_files()
        else:
            # Use direct Azure SDK
            pdf_files = []
            if connection_string:
                try:
                    from azure.storage.blob import BlobServiceClient
                    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                    container_client = blob_service_client.get_container_client(container_name)
                    blobs = list(container_client.list_blobs())
                    pdf_files = [{'name': blob.name, 'size': blob.size} for blob in blobs if blob.name.lower().endswith('.pdf')]
                except:
                    # Try Azure Identity
                    credential = DefaultAzureCredential()
                    account_url = f"https://{storage_account_name}.blob.core.windows.net"
                    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
                    container_client = blob_service_client.get_container_client(container_name)
                    blobs = list(container_client.list_blobs())
                    pdf_files = [{'name': blob.name, 'size': blob.size} for blob in blobs if blob.name.lower().endswith('.pdf')]
        
        print(f"✅ Container '{container_name}' accessible")
        print(f"📄 Found {len(pdf_files)} PDF files:")
        
        for i, file_info in enumerate(pdf_files[:5], 1):  # Show first 5
            if isinstance(file_info, dict):
                print(f"   {i}. {file_info['name']} ({file_info['size']} bytes)")
            else:
                print(f"   {i}. {file_info.name} ({file_info.size} bytes)")
        
        if len(pdf_files) > 5:
            print(f"   ... and {len(pdf_files) - 5} more files")
        
        if len(pdf_files) == 0:
            print("   💡 Upload a PDF file to test the monitoring system")
        
        return True
        
    except Exception as e:
        print(f"❌ Container access failed: {e}")
        
        if "ContainerNotFound" in str(e):
            print(f"\n💡 Container '{container_name}' doesn't exist")
            print("   Create it in Azure Portal or update AZURE_STORAGE_CONTAINER_NAME")
        
        return False


def main():
    """Main test function."""
    
    print("🚀 Azure PDF Extraction System - Connection Test")
    print("=" * 55)
    print()
    
    # Check .env file in parent directory
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print("✅ Found .env file")
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env")
        except ImportError:
            print("⚠️  python-dotenv not installed, loading manually...")
            # Simple .env parser
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
    else:
        print("⚠️  No .env file found, using system environment variables")
    
    print()
    
    # Run tests
    connection_ok = test_azure_connection()
    
    if connection_ok:
        container_ok = test_container_access()
        
        if container_ok:
            print("\n🎉 All tests passed! Your Azure setup is ready.")
            print("\n🚀 Next steps:")
            print("   1. Run: python main_enhanced_hitl.py")
            print("   2. Open: http://localhost:8000")
            print("   3. Upload a PDF to your Azure container to see it in action!")
            return True
    
    print("\n❌ Some tests failed. Please fix the issues above.")
    print("\n🔧 Quick fixes:")
    print("   • Check your .env file has correct values")
    print("   • Run 'az login' if using Azure Identity")
    print("   • Verify container exists in Azure Portal")
    
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)