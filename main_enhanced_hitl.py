#!/usr/bin/env python3
"""
Enhanced HITL System - Main Application

This is the main entry point for the Enhanced Human-in-the-Loop document processing system.
Run this to start the complete web interface with PDF viewer, field correction, and training data collection.

Usage:
    python main_enhanced_hitl.py

Features:
- Interactive PDF viewer with bounding box overlays
- Field correction interface with confidence scores  
- SLA tracking and automatic reviewer assignment
- Training data collection from reviewer feedback
- Real-time dashboard with queue monitoring
"""

import sys
from pathlib import Path
import logging
import uvicorn

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    
    print("🚀 Enhanced HITL Review System")
    print("=" * 50)
    print("Starting enhanced human-in-the-loop document processing system...")
    print("")
    print("Features:")
    print("  • Interactive PDF viewer with bounding box overlays")
    print("  • Field correction interface with confidence scores")
    print("  • SLA tracking and automatic reviewer assignment")
    print("  • Training data collection from reviewer feedback")
    print("  • Real-time dashboard with queue monitoring")
    print("")
    
    try:
        # Import and start the web application
        web_app_path = PROJECT_ROOT / "web" / "enhanced_hitl_web.py"
        
        if not web_app_path.exists():
            raise FileNotFoundError(f"Web application not found at {web_app_path}")
        
        print(f"🌐 Starting web server...")
        print(f"📁 Project root: {PROJECT_ROOT}")
        print(f"🔗 Access URL: http://localhost:8000")
        print("")
        print("Available endpoints:")
        print("  • Dashboard: http://localhost:8000")
        print("  • API Status: http://localhost:8000/api/queue-status")
        print("  • Create Sample: http://localhost:8000/api/create-sample-task")
        print("")
        
        # Change to web directory and run the application
        import os
        os.chdir(str(PROJECT_ROOT / "web"))
        
        # Import the FastAPI app
        sys.path.insert(0, str(PROJECT_ROOT / "web"))
        from enhanced_hitl_web import app
        
        # Run with uvicorn
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            reload=False
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Failed to import required modules.")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements/requirements_complete.txt")
        print("  # Or minimal: pip install -r requirements/requirements.txt")
        sys.exit(1)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print("❌ Required files not found.")
        print("Please ensure the project structure is complete.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()