"""
Azure Storage PDF File Listener

This module provides a class to monitor an Azure Storage container for new PDF files.
It supports both polling-based and event-driven approaches for detecting new files.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError, ResourceNotFoundError
# Event Grid consumer is imported on-demand when needed
# from azure.eventgrid import EventGridEvent
from azure.identity import DefaultAzureCredential


class AzurePDFListener:
    """
    A class to monitor Azure Storage containers for new PDF files.
    
    Supports both polling-based monitoring and event-driven monitoring
    using Azure Event Grid.
    """
    
    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
        connection_string: Optional[str] = None,
        use_managed_identity: bool = False,
        polling_interval: int = 30,
        log_level: str = "INFO"
    ):
        """
        Initialize the Azure PDF Listener.
        
        Args:
            storage_account_name: Name of the Azure Storage account
            container_name: Name of the container to monitor
            connection_string: Azure Storage connection string (optional if using managed identity)
            use_managed_identity: Whether to use Azure managed identity for authentication
            polling_interval: Interval in seconds for polling-based monitoring
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.storage_account_name = storage_account_name
        self.container_name = container_name
        self.connection_string = connection_string
        self.use_managed_identity = use_managed_identity
        self.polling_interval = polling_interval
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure clients
        self._initialize_clients()
        
        # Track processed files to avoid duplicates
        self.processed_files: Dict[str, datetime] = {}
        
        # Callback function for handling new PDF files
        self.pdf_callback: Optional[Callable[[str, BlobClient], None]] = None
        
    def _initialize_clients(self):
        """Initialize Azure Storage clients."""
        try:
            # Try connection string first, then fall back to Azure Identity
            if self.connection_string and not self.use_managed_identity:
                try:
                    self.blob_service_client = BlobServiceClient.from_connection_string(
                        self.connection_string
                    )
                    self.logger.info("Using connection string authentication")
                except Exception as e:
                    if "Authorization with Shared Key is disabled" in str(e):
                        self.logger.warning("Shared Key disabled, falling back to Azure Identity")
                        self.use_managed_identity = True
                    else:
                        raise
            
            if self.use_managed_identity or not self.connection_string:
                credential = DefaultAzureCredential()
                account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=credential
                )
                self.logger.info("Using Azure Identity authentication")
            
            if not hasattr(self, 'blob_service_client'):
                raise ValueError(
                    "Either connection_string or use_managed_identity=True must be provided"
                )
            
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            # Test connection
            self.container_client.get_container_properties()
            self.logger.info(f"Successfully connected to container '{self.container_name}'")
            
        except AzureError as e:
            self.logger.error(f"Failed to initialize Azure clients: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during client initialization: {e}")
            raise
    
    def set_pdf_callback(self, callback: Callable[[str, BlobClient], None]):
        """
        Set the callback function to handle new PDF files.
        
        Args:
            callback: Function that takes (blob_name, blob_client) as parameters
        """
        self.pdf_callback = callback
        self.logger.info("PDF callback function registered")
    
    def is_pdf_file(self, blob_name: str) -> bool:
        """
        Check if a blob is a PDF file based on its name.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            True if the blob appears to be a PDF file
        """
        return blob_name.lower().endswith('.pdf')
    
    def get_pdf_files(self) -> List[Dict[str, Any]]:
        """
        Get all PDF files currently in the container.
        
        Returns:
            List of dictionaries containing blob information
        """
        pdf_files = []
        
        try:
            blobs = self.container_client.list_blobs()
            
            for blob in blobs:
                if self.is_pdf_file(blob.name):
                    pdf_files.append({
                        'name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'creation_time': blob.creation_time,
                        'etag': blob.etag
                    })
                    
        except AzureError as e:
            self.logger.error(f"Error listing blobs: {e}")
            raise
        
        return pdf_files
    
    def download_pdf(self, blob_name: str, local_path: Optional[str] = None) -> str:
        """
        Download a PDF file from the container.
        
        Args:
            blob_name: Name of the blob to download
            local_path: Local path to save the file (optional)
            
        Returns:
            Path to the downloaded file
        """
        if not self.is_pdf_file(blob_name):
            raise ValueError(f"File '{blob_name}' is not a PDF file")
        
        if local_path is None:
            local_path = os.path.join(os.getcwd(), "downloads", blob_name)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            self.logger.info(f"Downloaded '{blob_name}' to '{local_path}'")
            return local_path
            
        except AzureError as e:
            self.logger.error(f"Error downloading blob '{blob_name}': {e}")
            raise
    
    def _handle_new_pdf(self, blob_name: str):
        """
        Handle a newly detected PDF file.
        
        Args:
            blob_name: Name of the new PDF blob
        """
        self.logger.info(f"New PDF file detected: {blob_name}")
        
        if self.pdf_callback:
            try:
                blob_client = self.container_client.get_blob_client(blob_name)
                self.pdf_callback(blob_name, blob_client)
            except Exception as e:
                self.logger.error(f"Error in PDF callback for '{blob_name}': {e}")
        else:
            self.logger.warning("No PDF callback registered")
        
        # Mark as processed
        self.processed_files[blob_name] = datetime.now(timezone.utc)
    
    def start_polling(self, run_once: bool = False):
        """
        Start polling the container for new PDF files.
        
        Args:
            run_once: If True, check once and return. If False, poll continuously.
        """
        self.logger.info(f"Starting polling monitor for container '{self.container_name}'")
        
        while True:
            try:
                current_files = self.get_pdf_files()
                
                for file_info in current_files:
                    blob_name = file_info['name']
                    
                    # Check if this is a new file
                    if blob_name not in self.processed_files:
                        self._handle_new_pdf(blob_name)
                
                if run_once:
                    break
                
                self.logger.debug(f"Sleeping for {self.polling_interval} seconds")
                time.sleep(self.polling_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Polling stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error during polling: {e}")
                time.sleep(self.polling_interval)
    
    def process_event_grid_event(self, event_data: Dict[str, Any]):
        """
        Process an Azure Event Grid event for blob creation.
        
        Args:
            event_data: Event Grid event data
        """
        try:
            if event_data.get('eventType') == 'Microsoft.Storage.BlobCreated':
                blob_url = event_data.get('data', {}).get('url', '')
                blob_name = blob_url.split('/')[-1] if blob_url else ''
                
                if blob_name and self.is_pdf_file(blob_name):
                    self._handle_new_pdf(blob_name)
                else:
                    self.logger.debug(f"Ignoring non-PDF file: {blob_name}")
            else:
                self.logger.debug(f"Ignoring event type: {event_data.get('eventType')}")
                
        except Exception as e:
            self.logger.error(f"Error processing Event Grid event: {e}")
    
    def get_blob_client(self, blob_name: str) -> BlobClient:
        """
        Get a blob client for a specific blob.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            BlobClient instance
        """
        return self.container_client.get_blob_client(blob_name)
    
    def get_blob_properties(self, blob_name: str) -> Dict[str, Any]:
        """
        Get properties of a specific blob.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            Dictionary containing blob properties
        """
        try:
            blob_client = self.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
            
            return {
                'name': blob_name,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'creation_time': properties.creation_time,
                'etag': properties.etag,
                'content_type': properties.content_settings.content_type,
                'metadata': properties.metadata
            }
            
        except AzureError as e:
            self.logger.error(f"Error getting properties for blob '{blob_name}': {e}")
            raise


# Example usage and callback function
def example_pdf_callback(blob_name: str, blob_client: BlobClient):
    """
    Example callback function for handling new PDF files.
    
    Args:
        blob_name: Name of the new PDF blob
        blob_client: BlobClient instance for the new PDF
    """
    print(f"Processing new PDF: {blob_name}")
    
    # Get blob properties
    properties = blob_client.get_blob_properties()
    print(f"File size: {properties.size} bytes")
    print(f"Last modified: {properties.last_modified}")
    
    # You can add your PDF processing logic here
    # For example:
    # - Download the file
    # - Extract text or fields
    # - Store metadata in a database
    # - Trigger other workflows


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Configuration - you should set these via environment variables
    STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "pdfs")
    CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not STORAGE_ACCOUNT_NAME:
        print("Please set AZURE_STORAGE_ACCOUNT_NAME environment variable")
        sys.exit(1)
    
    if not CONNECTION_STRING:
        print("Please set AZURE_STORAGE_CONNECTION_STRING environment variable")
        sys.exit(1)
    
    # Create the listener
    listener = AzurePDFListener(
        storage_account_name=STORAGE_ACCOUNT_NAME,
        container_name=CONTAINER_NAME,
        connection_string=CONNECTION_STRING,
        polling_interval=30,  # Check every 30 seconds
        log_level="INFO"
    )
    
    # Set the callback function
    listener.set_pdf_callback(example_pdf_callback)
    
    # Start monitoring
    print(f"Starting to monitor container '{CONTAINER_NAME}' for new PDF files...")
    print("Press Ctrl+C to stop")
    
    try:
        listener.start_polling()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")