# 🔧 Troubleshooting Guide

This guide helps you resolve common issues when setting up and running the Enhanced PDF Field Extraction System.

## Common Setup Issues

### "No module named 'azure'" Error
```bash
pip install -r requirements.txt
```

### "Storage account not found" Error
- Verify storage account name is correct in `.env`
- Check that storage account exists in Azure Portal
- Ensure connection string is complete and valid

### "Container not found" Error  
- Create container named `pdfs` in your storage account
- Or update `AZURE_STORAGE_CONTAINER_NAME` in `.env`

### "Authorization with Shared Key is disabled" Error
This happens when your Azure Storage account has disabled shared key authentication. **Two solutions:**

**Option 1: Enable Shared Key (Quickest)**
1. Go to Azure Portal → Your Storage Account → Configuration
2. Find "Allow shared key access" → Change to "Enabled"
3. Click "Save"

**Option 2: Use Azure Identity (More secure)**
```bash
# 1. Install Azure CLI first
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
# Or on macOS: brew install azure-cli
# Or on Windows: winget install Microsoft.AzureCLI

# 2. Install Azure Identity Python package
pip install azure-identity

# 3. Update your .env file - remove connection string, add these:
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_CONTAINER_NAME=pdfs
# Remove: AZURE_STORAGE_CONNECTION_STRING

# 4. Login to Azure CLI
az login
```

Then the system will automatically use your Azure credentials instead of connection strings.

### "AuthorizationPermissionMismatch" Error
This happens when your user account lacks permissions to access the storage account. **Solutions:**

**Option 1: Add Storage Permissions (Recommended)**
1. Go to Azure Portal → Your Storage Account → Access Control (IAM)
2. Click "Add" → "Add role assignment"
3. Select one of these roles:
   - **Storage Blob Data Contributor** (read/write access)
   - **Storage Blob Data Reader** (read-only access)
4. Search for your email/username and assign the role
5. Wait 5-10 minutes for permissions to propagate

**Option 2: Use Connection String Instead**
```bash
# In your .env file, add the connection string:
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

**Option 3: Check Your Subscription**
```bash
# Verify you're in the correct subscription
az account show
az account list --output table

# Switch if needed
az account set --subscription "your-subscription-name"
```

### "Key based authentication is not permitted" Error
This occurs when corporate policies prevent shared key access at the tenant level.

**Solution: Use Azure Identity with RBAC**
1. Ensure you have "Storage Blob Data Contributor" role on the storage account
2. Update your `.env` file to use Azure Identity:
   ```env
   USE_MANAGED_IDENTITY=true
   # Remove or comment out AZURE_STORAGE_CONNECTION_STRING
   ```
3. Login with Azure CLI: `az login`
4. Test connection: `python tests/test_azure_connection.py`

### "Firewalls and virtual networks settings may be blocking access" Error
This happens when your storage account has network restrictions enabled and your IP address isn't allowed.

**Solution Option 1: Add Your IP Address (Recommended)**
1. Go to Azure Portal → Your Storage Account → **Networking**
2. Under "Firewalls and virtual networks"
3. Click **"Add your client IP address"** (it will show your current IP)
4. Click **"Save"**
5. Wait 2-3 minutes for changes to propagate

**Solution Option 2: Allow All Networks (Less Secure)**
1. Go to Azure Portal → Your Storage Account → **Networking**
2. Select **"Enabled from all networks"**
3. Click **"Save"**

**Solution Option 3: Via Azure CLI**
```bash
# Add your current IP (replace with your IP address)
az storage account network-rule add \
    --account-name your_storage_account_name \
    --resource-group your_resource_group \
    --ip-address 174.182.122.183

# Or allow all networks
az storage account update \
    --name your_storage_account_name \
    --resource-group your_resource_group \
    --default-action Allow
```

### Can't access web interface
```bash
# Check if port 8000 is available
python main_enhanced_hitl.py
# Try different port if needed
```

### Import errors with OpenCV/libGL
This is common in headless environments like dev containers.

**Solution: Install system dependencies**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Python virtual environment issues
If you're getting module not found errors:

```bash
# Make sure you're using the configured virtual environment
.venv/bin/python tests/test_azure_connection.py

# Or activate the virtual environment
source .venv/bin/activate
python tests/test_azure_connection.py
```

## Connection Testing

### Quick Connection Test
Run the automated connection test:
```bash
.venv/bin/python tests/test_azure_connection.py
```

This will:
- ✅ Test Azure Storage connectivity
- ✅ Detect authentication method compatibility  
- ✅ List containers and PDF files
- ✅ Provide specific error guidance

### Manual Azure CLI Testing
Test your Azure connection directly:
```bash
# List storage accounts
az storage account list --output table

# List containers (replace with your account name)
az storage container list --account-name your_storage_account --output table

# Test with connection string
az storage container list --connection-string "your_connection_string" --output table
```

## Getting Help

1. **Check logs**: Most issues show helpful error messages
2. **Run connection test**: `python tests/test_azure_connection.py`
3. **Test connection**: Upload a PDF to your Azure container
4. **View documentation**: [docs/guides/](guides/) for detailed guides

## Advanced Troubleshooting

### Azure Identity Chain Issues
If Azure Identity fails to authenticate:

```bash
# Check current Azure context
az account show

# List available subscriptions  
az account list --output table

# Login with specific tenant
az login --tenant your-tenant-id

# Clear Azure CLI cache if needed
az account clear
az login
```

### Corporate Environment Issues
In Microsoft or enterprise environments:

1. **Shared Key Access** may be disabled by policy
2. **Connection strings** may be blocked
3. **Azure Identity** requires explicit RBAC permissions
4. **Multiple tenants** can cause authentication confusion

**Solution**: Use Azure Identity with proper RBAC roles assigned.

### Debug Mode
For detailed error information, set debug logging:

In your `.env` file:
```env
LOG_LEVEL=DEBUG
```

Or run with debug output:
```bash
export AZURE_LOG_LEVEL=DEBUG
python tests/test_azure_connection.py
```

## Need Azure Help?

- **Free Azure Account**: https://azure.microsoft.com/free/
- **Azure Storage Tutorial**: https://docs.microsoft.com/en-us/azure/storage/
- **Document Intelligence**: https://docs.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/
- **Azure Identity**: https://docs.microsoft.com/en-us/azure/developer/python/azure-sdk-authenticate
- **Azure RBAC**: https://docs.microsoft.com/en-us/azure/role-based-access-control/