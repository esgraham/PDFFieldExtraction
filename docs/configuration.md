# Configuration Guide

## Environment Variables

The Azure PDF Listener can be configured using environment variables or direct parameters.

### Required Variables

#### AZURE_STORAGE_ACCOUNT_NAME
Your Azure Storage account name.

**Example:** `mystorageaccount`

#### Authentication (Choose One)

**Option 1: Connection String (Recommended for Development)**
```bash
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=mystorageaccount;AccountKey=mykey;EndpointSuffix=core.windows.net
```

**Option 2: Managed Identity (Recommended for Production)**
```bash
USE_MANAGED_IDENTITY=true
```

### Optional Variables

#### AZURE_STORAGE_CONTAINER_NAME
Name of the container to monitor (default: "pdfs")

#### POLLING_INTERVAL
How often to check for new files in seconds (default: 30)

#### LOG_LEVEL
Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: "INFO")

## Configuration Files

### .env File Setup

1. Copy the example configuration:
   ```bash
   cp config/.env.example .env
   ```

2. Edit `.env` with your values:
   ```bash
   AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
   AZURE_STORAGE_CONTAINER_NAME=pdfs
   AZURE_STORAGE_CONNECTION_STRING=your_connection_string
   POLLING_INTERVAL=30
   LOG_LEVEL=INFO
   ```

## Azure Storage Setup

### Creating a Storage Account

1. **Azure Portal Method:**
   - Go to Azure Portal
   - Create new Storage Account
   - Choose appropriate settings for your use case

2. **Azure CLI Method:**
   ```bash
   az storage account create \
     --name mystorageaccount \
     --resource-group myresourcegroup \
     --location eastus \
     --sku Standard_LRS
   ```

### Creating a Container

1. **Azure Portal Method:**
   - Navigate to your Storage Account
   - Go to "Containers" under "Data storage"
   - Click "+ Container"
   - Set name (e.g., "pdfs") and access level

2. **Azure CLI Method:**
   ```bash
   az storage container create \
     --name pdfs \
     --account-name mystorageaccount
   ```

### Getting Connection String

1. **Azure Portal Method:**
   - Go to your Storage Account
   - Navigate to "Access keys" under "Security + networking"
   - Copy the connection string from Key1 or Key2

2. **Azure CLI Method:**
   ```bash
   az storage account show-connection-string \
     --name mystorageaccount \
     --resource-group myresourcegroup
   ```

## Authentication Methods

### Connection String (Development)

**Pros:**
- Simple to set up
- Good for development and testing
- Works from any environment

**Cons:**
- Requires storing secrets
- Less secure for production
- Manual key rotation

**Setup:**
```python
listener = AzurePDFListener(
    storage_account_name="mystorageaccount",
    container_name="pdfs",
    connection_string="your_connection_string"
)
```

### Managed Identity (Production)

**Pros:**
- No secrets to manage
- Automatic credential rotation
- Enhanced security
- Azure-native authentication

**Cons:**
- Only works within Azure
- Requires proper role assignments

**Setup:**
```python
listener = AzurePDFListener(
    storage_account_name="mystorageaccount",
    container_name="pdfs",
    use_managed_identity=True
)
```

**Required Permissions:**
- Assign "Storage Blob Data Reader" role to the managed identity
- Scope can be at storage account or container level

### Role Assignment Commands

```bash
# Get the principal ID of your managed identity
PRINCIPAL_ID=$(az identity show --name myidentity --resource-group mygroup --query principalId -o tsv)

# Assign Storage Blob Data Reader role
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Reader" \
  --scope "/subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/STORAGE_ACCOUNT"
```

## Performance Tuning

### Polling Interval

- **High frequency (5-15 seconds):** Good for real-time processing, higher API costs
- **Medium frequency (30-60 seconds):** Balanced approach
- **Low frequency (300+ seconds):** Cost-effective for batch processing

### Container Organization

- Use separate containers for different PDF types
- Consider using blob prefixes for organization
- Monitor container size and file count

### Scaling Considerations

- **Single instance:** Up to ~1000 files per container
- **Multiple instances:** Use Event Grid for better scalability
- **High volume:** Consider Azure Functions with Event Grid triggers

## Monitoring and Logging

### Enable Debug Logging

```python
listener = AzurePDFListener(
    # ... other parameters
    log_level="DEBUG"
)
```

### Log Output Includes

- Connection establishment
- File discovery events
- Processing results
- Error details and stack traces
- Performance metrics

### Azure Monitor Integration

For production environments, consider integrating with Azure Monitor:

```python
import logging
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter

# Configure Azure Monitor logging
exporter = AzureMonitorLogExporter(
    connection_string="InstrumentationKey=your-key"
)
```

## Security Best Practices

1. **Use Managed Identity** in production
2. **Limit permissions** to minimum required (Storage Blob Data Reader)
3. **Store connection strings** in Azure Key Vault
4. **Enable audit logging** for compliance
5. **Use private endpoints** for enhanced network security
6. **Regularly rotate** access keys if using connection strings
7. **Monitor access patterns** for anomalies

## Troubleshooting

### Common Configuration Issues

**"Authentication failed"**
- Verify connection string format
- Check if managed identity has proper permissions
- Ensure storage account name is correct

**"Container not found"**
- Verify container name spelling
- Check if container exists in the specified storage account
- Ensure proper permissions to access the container

**"No files detected"**
- Verify files have `.pdf` extension
- Check if files are in the correct container
- Enable DEBUG logging for detailed information

### Validation Script

Use the test script to validate your configuration:

```bash
python tests/test_setup.py
```