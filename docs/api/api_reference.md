# API Reference

## AzurePDFListener Class

The main class for monitoring Azure Storage containers for PDF files.

### Constructor

```python
AzurePDFListener(
    storage_account_name: str,
    container_name: str,
    connection_string: Optional[str] = None,
    use_managed_identity: bool = False,
    polling_interval: int = 30,
    log_level: str = "INFO"
)
```

#### Parameters

- **storage_account_name** (str): Name of the Azure Storage account
- **container_name** (str): Name of the container to monitor
- **connection_string** (Optional[str]): Azure Storage connection string
- **use_managed_identity** (bool): Whether to use Azure managed identity for authentication
- **polling_interval** (int): Interval in seconds for polling-based monitoring (default: 30)
- **log_level** (str): Logging level - DEBUG, INFO, WARNING, ERROR (default: "INFO")

### Methods

#### set_pdf_callback(callback)

Set the callback function to handle new PDF files.

**Parameters:**
- `callback` (Callable[[str, BlobClient], None]): Function that takes blob_name and blob_client

**Example:**
```python
def handle_pdf(blob_name, blob_client):
    print(f"New PDF: {blob_name}")

listener.set_pdf_callback(handle_pdf)
```

#### start_polling(run_once=False)

Start monitoring the container for new PDF files.

**Parameters:**
- `run_once` (bool): If True, check once and return. If False, poll continuously (default: False)

**Example:**
```python
listener.start_polling()  # Continuous monitoring
listener.start_polling(run_once=True)  # Single check
```

#### get_pdf_files()

Get all PDF files currently in the container.

**Returns:**
- List[Dict[str, Any]]: List of dictionaries containing blob information

**Example:**
```python
files = listener.get_pdf_files()
for file_info in files:
    print(f"File: {file_info['name']}, Size: {file_info['size']}")
```

#### download_pdf(blob_name, local_path=None)

Download a PDF file from the container.

**Parameters:**
- `blob_name` (str): Name of the blob to download
- `local_path` (Optional[str]): Local path to save the file

**Returns:**
- str: Path to the downloaded file

**Example:**
```python
local_file = listener.download_pdf("document.pdf", "downloads/document.pdf")
```

#### get_blob_client(blob_name)

Get a blob client for a specific blob.

**Parameters:**
- `blob_name` (str): Name of the blob

**Returns:**
- BlobClient: Azure BlobClient instance

#### get_blob_properties(blob_name)

Get properties of a specific blob.

**Parameters:**
- `blob_name` (str): Name of the blob

**Returns:**
- Dict[str, Any]: Dictionary containing blob properties

#### is_pdf_file(blob_name)

Check if a blob is a PDF file based on its name.

**Parameters:**
- `blob_name` (str): Name of the blob

**Returns:**
- bool: True if the blob appears to be a PDF file

#### process_event_grid_event(event_data)

Process an Azure Event Grid event for blob creation.

**Parameters:**
- `event_data` (Dict[str, Any]): Event Grid event data

**Example:**
```python
event = {
    'eventType': 'Microsoft.Storage.BlobCreated',
    'data': {'url': 'https://account.blob.core.windows.net/container/file.pdf'}
}
listener.process_event_grid_event(event)
```

## Error Handling

The class raises the following exceptions:

- **ValueError**: When invalid configuration is provided
- **AzureError**: For Azure Storage-specific errors
- **ResourceNotFoundError**: When containers or blobs are not found

## Logging

The class uses Python's standard logging module. Log messages include:

- Connection status
- New file detections
- Processing results
- Error messages with stack traces

Configure logging level via the `log_level` parameter or environment variables.