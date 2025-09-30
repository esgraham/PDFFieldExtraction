#!/usr/bin/env python3
"""
Main entry point for Azure PDF Listener project.

This script provides a command-line interface for running the Azure PDF Listener
with different modes and configurations.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv


def run_simple_monitor():
    """Run the simple PDF monitor."""
    from examples.simple_monitor import main
    main()


def run_advanced_processing():
    """Run the advanced PDF processor."""
    from examples.advanced_processing import main
    main()


def run_preprocessing_example():
    """Run the preprocessing example."""
    from examples.preprocessing_example import main
    main()


def run_classification_example():
    """Run the document classification example."""
    from examples.classification_example import demonstrate_classification, demonstrate_integration
    success = demonstrate_classification()
    if success:
        demonstrate_integration()


def run_ocr_example():
    """Run the OCR integration example."""
    from examples.ocr_example import main
    main()


def run_tests():
    """Run the test suite."""
    import subprocess
    test_dir = Path(__file__).parent / "tests"
    subprocess.run([sys.executable, "-m", "unittest", "discover", str(test_dir)])


def setup_project():
    """Run the project setup."""
    from scripts.setup import main
    main()


def validate_setup():
    """Validate the project setup."""
    from tests.test_setup import main
    main()


def main():
    """Main entry point with command-line argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Azure PDF Listener - Monitor Azure Storage for PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py simple                 # Run simple monitoring
  python main.py advanced               # Run advanced processing
  python main.py preprocessing          # Run preprocessing examples
  python main.py classification         # Run document classification demo
  python main.py ocr                    # Run OCR integration demo
  python main.py setup                  # Set up the project
  python main.py test                   # Run tests
  python main.py validate               # Validate configuration
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["simple", "advanced", "preprocessing", "classification", "ocr", "setup", "test", "validate"],
        help="Operation mode to run"
    )
    
    parser.add_argument(
        "--config",
        default=".env",
        help="Path to configuration file (default: .env)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from specified config file
    if os.path.exists(args.config):
        load_dotenv(args.config)
    elif args.mode not in ["setup"]:
        print(f"⚠️  Configuration file '{args.config}' not found")
        print("   Run 'python main.py setup' to create it")
    
    # Set verbose logging if requested
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Route to appropriate function
    if args.mode == "simple":
        print("🚀 Starting Simple PDF Monitor...")
        run_simple_monitor()
    elif args.mode == "advanced":
        print("🚀 Starting Advanced PDF Processor...")
        run_advanced_processing()
    elif args.mode == "preprocessing":
        print("🚀 Starting Preprocessing Examples...")
        run_preprocessing_example()
    elif args.mode == "classification":
        print("🤖 Starting Document Classification Demo...")
        run_classification_example()
    elif args.mode == "ocr":
        print("📄 Starting OCR Integration Demo...")
        run_ocr_example()
    elif args.mode == "setup":
        print("🔧 Setting up Azure PDF Listener...")
        setup_project()
    elif args.mode == "test":
        print("🧪 Running test suite...")
        run_tests()
    elif args.mode == "validate":
        print("✅ Validating setup...")
        validate_setup()


if __name__ == "__main__":
    main()