"""
Unit tests for Azure PDF Listener functionality.

These tests verify the core functionality of the AzurePDFListener class.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from azure_pdf_listener import AzurePDFListener


class TestAzurePDFListener(unittest.TestCase):
    """Test cases for AzurePDFListener class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.storage_account = "teststorageaccount"
        self.container_name = "testcontainer"
        self.connection_string = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=testkey;EndpointSuffix=core.windows.net"
    
    def test_is_pdf_file(self):
        """Test PDF file detection."""
        with patch('azure_pdf_listener.BlobServiceClient'):
            listener = AzurePDFListener(
                storage_account_name=self.storage_account,
                container_name=self.container_name,
                connection_string=self.connection_string
            )
            
            # Test positive cases
            self.assertTrue(listener.is_pdf_file("document.pdf"))
            self.assertTrue(listener.is_pdf_file("DOCUMENT.PDF"))
            self.assertTrue(listener.is_pdf_file("path/to/document.pdf"))
            
            # Test negative cases
            self.assertFalse(listener.is_pdf_file("document.txt"))
            self.assertFalse(listener.is_pdf_file("document.docx"))
            self.assertFalse(listener.is_pdf_file("document"))
    
    @patch('azure_pdf_listener.BlobServiceClient')
    def test_initialization_with_connection_string(self, mock_blob_service):
        """Test initialization with connection string."""
        mock_container_client = Mock()
        mock_blob_service.from_connection_string.return_value.get_container_client.return_value = mock_container_client
        
        listener = AzurePDFListener(
            storage_account_name=self.storage_account,
            container_name=self.container_name,
            connection_string=self.connection_string
        )
        
        self.assertEqual(listener.storage_account_name, self.storage_account)
        self.assertEqual(listener.container_name, self.container_name)
        self.assertEqual(listener.connection_string, self.connection_string)
        self.assertFalse(listener.use_managed_identity)
    
    def test_callback_registration(self):
        """Test callback function registration."""
        with patch('azure_pdf_listener.BlobServiceClient'):
            listener = AzurePDFListener(
                storage_account_name=self.storage_account,
                container_name=self.container_name,
                connection_string=self.connection_string
            )
            
            def test_callback(blob_name, blob_client):
                pass
            
            listener.set_pdf_callback(test_callback)
            self.assertEqual(listener.pdf_callback, test_callback)
    
    def test_initialization_without_auth(self):
        """Test that initialization fails without authentication."""
        with self.assertRaises(ValueError):
            AzurePDFListener(
                storage_account_name=self.storage_account,
                container_name=self.container_name
            )


class TestEventGridProcessing(unittest.TestCase):
    """Test cases for Event Grid event processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('azure_pdf_listener.BlobServiceClient'):
            self.listener = AzurePDFListener(
                storage_account_name="test",
                container_name="test",
                connection_string="test_connection_string"
            )
    
    def test_process_blob_created_event(self):
        """Test processing of blob created events."""
        # Mock callback function
        self.listener.pdf_callback = Mock()
        
        # Mock container client
        with patch.object(self.listener, 'container_client') as mock_container:
            mock_blob_client = Mock()
            mock_container.get_blob_client.return_value = mock_blob_client
            
            # Test PDF file event
            event_data = {
                'eventType': 'Microsoft.Storage.BlobCreated',
                'data': {
                    'url': 'https://test.blob.core.windows.net/container/test.pdf'
                }
            }
            
            self.listener.process_event_grid_event(event_data)
            
            # Verify callback was called
            self.listener.pdf_callback.assert_called_once_with('test.pdf', mock_blob_client)
    
    def test_ignore_non_pdf_events(self):
        """Test that non-PDF files are ignored."""
        self.listener.pdf_callback = Mock()
        
        event_data = {
            'eventType': 'Microsoft.Storage.BlobCreated',
            'data': {
                'url': 'https://test.blob.core.windows.net/container/test.txt'
            }
        }
        
        self.listener.process_event_grid_event(event_data)
        
        # Verify callback was not called
        self.listener.pdf_callback.assert_not_called()


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)