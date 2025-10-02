"""
Configuration Manager

Handles loading and validating configuration from environment variables
and configuration files.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages application configuration loading and validation."""

    @staticmethod
    def load_configuration() -> Dict[str, Any]:
        """Load configuration from environment variables and defaults."""
        use_managed_identity = os.getenv("USE_MANAGED_IDENTITY", 'false').lower() == 'true'
        config = {
            'azure': {
                'enable_managed_identity': use_managed_identity,
                'account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME', '') if use_managed_identity else None,
                'connection_string': os.getenv('AZURE_STORAGE_CONNECTION_STRING') if not use_managed_identity else None,
                'container_name': os.getenv('AZURE_CONTAINER_NAME', 'pdfs'),
            },
            'azure_document_intelligence': {
                'enabled': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENABLED', 'false').lower() == 'true',
                'endpoint': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT'),
                'api_key': os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY'),
            },
            'processing': {
                'max_concurrent_processing': int(os.getenv('MAX_CONCURRENT_PROCESSING', '3')),
                'enable_preprocessing': os.getenv('ENABLE_PREPROCESSING', 'true').lower() == 'true',
                'enable_classification': os.getenv('ENABLE_CLASSIFICATION', 'true').lower() == 'true',
                'enable_field_extraction': os.getenv('ENABLE_FIELD_EXTRACTION', 'true').lower() == 'true',
                'enable_validation': os.getenv('ENABLE_VALIDATION', 'true').lower() == 'true',
            },
            'monitoring': {
                'polling_interval': int(os.getenv('POLLING_INTERVAL', '30')),
                'watch_interval': int(os.getenv('WATCH_INTERVAL', '10')),
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO').upper(),
                'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            },
            'output': {
                'save_results': os.getenv('SAVE_RESULTS', 'true').lower() == 'true',
                'results_directory': os.getenv('RESULTS_DIR', 'results'),
                # json, csv, xlsx
                'export_format': os.getenv('EXPORT_FORMAT', 'json'),
            }
        }

        # Validate critical configuration
        ConfigurationManager._validate_configuration(config)

        logger.info("✅ Configuration loaded successfully")
        return config

    @staticmethod
    def _validate_configuration(config: Dict[str, Any]):
        """Validate that required configuration is present."""
        errors = []

        # Check Azure Storage configuration
        if config['azure']['enable_managed_identity']:
            if not config['azure']['account_name']:
             errors.append("AZURE_STORAGE_ACCOUNT_NAME is required when USE_MANAGED_IDENTITY is true")
        else:
            if not config['azure']['connection_string']:
             errors.append("AZURE_STORAGE_CONNECTION_STRING is required when USE_MANAGED_IDENTITY is false")

        # Check Azure Document Intelligence if enabled
        if config['azure_document_intelligence']['enabled']:
            if not config['azure_document_intelligence']['endpoint']:
                errors.append(
                    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is required when Azure Document Intelligence is enabled")
            if not config['azure_document_intelligence']['api_key']:
                errors.append(
                    "AZURE_DOCUMENT_INTELLIGENCE_KEY is required when Azure Document Intelligence is enabled")

        # Check output directory
        results_dir = Path(config['output']['results_directory'])
        if config['output']['save_results']:
            try:
                results_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(
                    f"Cannot create results directory {results_dir}: {e}")

        if errors:
            error_msg = "Configuration validation failed:\n" + \
                "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def setup_logging(config: Dict[str, Any]):
        """Setup logging configuration."""
        log_level = getattr(logging, config['logging']['level'], logging.INFO)
        log_format = config['logging']['format']

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_processing.log')
            ]
        )

        # Set specific logger levels
        logging.getLogger('azure').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        logger.info(
            f"Logging configured at {config['logging']['level']} level")

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            'python_version': os.sys.version,
            'platform': os.name,
            'working_directory': os.getcwd(),
            'environment_variables': {
                key: '***' if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower()
                else value
                for key, value in os.environ.items()
                if key.startswith(('AZURE_', 'LOG_', 'ENABLE_', 'MAX_', 'POLLING_'))
            }
        }

    @staticmethod
    def create_sample_env_file(filepath: str = '.env.sample'):
        """Create a sample environment file with all configuration options."""
        sample_content = '''# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
AZURE_STORAGE_ACCOUNT_NAME=your_account
AZURE_CONTAINER_NAME=pdfs
USE_MANAGED_IDENTITY=true

# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_ENABLED=true
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key

# Processing Configuration
MAX_CONCURRENT_PROCESSING=3
ENABLE_PREPROCESSING=true
ENABLE_CLASSIFICATION=true
ENABLE_FIELD_EXTRACTION=true
ENABLE_VALIDATION=true

# Monitoring Configuration
POLLING_INTERVAL=30
WATCH_INTERVAL=10

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Output Configuration
SAVE_RESULTS=true
RESULTS_DIR=results
EXPORT_FORMAT=json
'''

        with open(filepath, 'w') as f:
            f.write(sample_content)

        logger.info(f"Sample environment file created: {filepath}")
