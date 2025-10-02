#!/usr/bin/env python3
"""
Azure PDF Processing Pipeline - Main Entry Point (Safe Version)

A streamlined main entry point with careful module loading to avoid
import issues with heavy dependencies.

Usage:
    python main.py monitor     # Continuous monitoring mode
    python main.py process     # Process specific file  
    python main.py batch       # Batch process all files
    python main.py config      # Create sample config
    python main.py info        # Show environment information
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.core.config_manager import ConfigurationManager

# Load environment variables early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, loading environment variables from system")

logger = logging.getLogger(__name__)


def setup_basic_logging():
    """Setup basic logging before configuration is loaded."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def show_environment_info(config):
    """Show environment information for debugging."""
    env_info = ConfigurationManager.get_environment_info()
    
    print("🔧 Environment Information:")
    print(f"   Python: {env_info['python_version']}")
    print(f"   Platform: {env_info['platform']}")
    print(f"   Working Directory: {env_info['working_directory']}")
    
    print("\n⚙️  Configuration:")
    print(f"   Azure Container: {config['azure']['container_name']}")
    print(f"   Azure Doc Intel: {'Enabled' if config['azure_document_intelligence']['enabled'] else 'Disabled'}")
    print(f"   Polling Interval: {config['monitoring']['polling_interval']}s")
    print(f"   Max Concurrent: {config['processing']['max_concurrent_processing']}")
    print(f"   Save Results: {'Yes' if config['output']['save_results'] else 'No'}")
    print(f"   Export Format: {config['output']['export_format']}")


async def run_monitor_mode(config):
    """Run continuous monitoring mode."""
    logger.info("🚀 Starting PDF Processing Pipeline - Monitor Mode")
    
    try:
        # Import our modular components
        logger.info("Loading processing components...")
        
        from src.core.monitor_service import PDFMonitorService
        from src.core.result_handler import ResultHandler
        
        # Initialize services
        monitor_service = PDFMonitorService(config)
        result_handler = ResultHandler(config)
        
        # Run monitoring
        await monitor_service.run_monitor_mode()
        
    except Exception as e:
        logger.error(f"❌ Monitor mode failed: {e}")
        logger.info("💡 Make sure all required packages are installed and environment variables are set")
        raise


async def run_process_mode(config, filename):
    """Process a specific file."""
    logger.info(f"🚀 Starting PDF Processing Pipeline - Process Mode: {filename}")
    
    try:
        from src.core.monitor_service import PDFMonitorService
        from src.core.result_handler import ResultHandler
        
        # Initialize services
        monitor_service = PDFMonitorService(config)
        result_handler = ResultHandler(config)
        
        # Process the specific file
        result = await monitor_service.process_single_file(filename)
        
        # Save result
        if result:
            saved_path = await result_handler.save_result(result)
            if saved_path:
                logger.info(f"💾 Result saved to: {saved_path}")
            
            # Display summary
            logger.info("📋 Processing Summary:")
            summary = result.get('summary', {})
            logger.info(f"   ⏱️  Processing Time: {summary.get('processing_time', 0):.2f}s")
            logger.info(f"   ✅ Success Rate: {summary.get('success_rate', 0):.1%}")
            logger.info(f"   📊 Quality Score: {summary.get('data_quality_score', 0):.2f}")
            logger.info(f"   👤 Needs Review: {'Yes' if result.get('needs_human_review') else 'No'}")
        
    except Exception as e:
        logger.error(f"❌ Process mode failed: {e}")
        raise


async def run_batch_mode(config):
    """Process all files in batch mode."""
    logger.info("🚀 Starting PDF Processing Pipeline - Batch Mode")
    
    try:
        from src.core.monitor_service import PDFMonitorService
        from src.core.result_handler import ResultHandler
        
        # Initialize services
        monitor_service = PDFMonitorService(config)
        result_handler = ResultHandler(config)
        
        # Process all files
        results = await monitor_service.process_batch()
        
        if results:
            # Save batch results
            saved_path = await result_handler.save_batch_results(results)
            if saved_path:
                logger.info(f"💾 Batch results saved to: {saved_path}")
            
            # Display batch summary
            successful = len([r for r in results if not r.get('error')])
            failed = len(results) - successful
            
            logger.info("📋 Batch Processing Summary:")
            logger.info(f"   📄 Total Files: {len(results)}")
            logger.info(f"   ✅ Successful: {successful}")
            logger.info(f"   ❌ Failed: {failed}")
            logger.info(f"   📊 Success Rate: {successful/len(results):.1%}")
            
            if failed > 0:
                logger.info("❌ Failed files:")
                for result in results:
                    if result.get('error'):
                        logger.info(f"   - {result.get('blob_name', 'unknown')}: {result.get('error')}")
        else:
            logger.info("📂 No files found to process")
        
    except Exception as e:
        logger.error(f"❌ Batch mode failed: {e}")
        raise


def main():
    """Main entry point."""
    setup_basic_logging()
    
    parser = argparse.ArgumentParser(
        description="Azure PDF Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py monitor                    # Start monitoring mode
  python main.py monitor --watch-interval 5 # Monitor with 5s interval
  python main.py process invoice.pdf        # Process specific file
  python main.py batch                      # Process all files
  python main.py config                     # Create sample config
  python main.py info                       # Show environment info
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['monitor', 'process', 'batch', 'config', 'info'],
        help='Processing mode'
    )
    
    parser.add_argument(
        'filename',
        nargs='?',
        help='PDF filename to process (required for process mode)'
    )
    
    parser.add_argument(
        '--watch-interval',
        type=int,
        help='Polling interval in seconds for monitor mode'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level'
    )
    
    args = parser.parse_args()
    
    # Handle special modes first
    if args.mode == 'config':
        ConfigurationManager.create_sample_env_file()
        return
    
    try:
        # Load configuration
        config = ConfigurationManager.load_configuration()
        
        # Override config with command line arguments
        if args.watch_interval:
            config['monitoring']['polling_interval'] = args.watch_interval
        
        if args.log_level:
            config['logging']['level'] = args.log_level
        
        # Setup logging with ConfigurationManager
        ConfigurationManager.setup_logging(config)
        
        # Handle info mode
        if args.mode == 'info':
            show_environment_info(config)
            return
        
        # Validate arguments
        if args.mode == 'process' and not args.filename:
            logger.error("❌ Filename is required for process mode")
            parser.print_help()
            sys.exit(1)
        
        # Validate required configuration
        if not config['azure']['connection_string'] and not config['azure']['enable_managed_identity']:
            logger.error("❌ AZURE_STORAGE_CONNECTION_STRING or USE_MANAGED_IDENTITY is required")
            logger.info("💡 Run 'python main.py config' to create a sample configuration file")
            sys.exit(1)
        
        # Run the appropriate mode
        if args.mode == 'monitor':
            asyncio.run(run_monitor_mode(config))
        elif args.mode == 'process':
            asyncio.run(run_process_mode(config, args.filename))
        elif args.mode == 'batch':
            asyncio.run(run_batch_mode(config))
            
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()