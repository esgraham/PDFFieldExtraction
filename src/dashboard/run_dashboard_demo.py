#!/usr/bin/env python3
"""
Dashboard Demo Script

Runs the PDF Field Extraction Dashboard with sample data.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the dashboard demo."""
    
    logger.info("🚀 Starting PDF Field Extraction Dashboard Demo")
    logger.info("=" * 60)
    
    # Set up demo database path
    demo_db_path = project_root / "data" / "databases" / "dashboard_demo.db"
    
    # Check if demo database exists, if not create it
    if not demo_db_path.exists():
        logger.info("📊 Demo database not found, creating with sample data...")
        try:
            setup_script = project_root / "data" / "databases" / "setup_databases.py"
            if setup_script.exists():
                import subprocess
                subprocess.run([sys.executable, str(setup_script)], check=True)
            else:
                logger.error("❌ Database setup script not found!")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to create demo database: {e}")
            sys.exit(1)
    
    try:
        # Import dashboard app
        from src.dashboard.dashboard_app import app
        import uvicorn
        
        logger.info("📊 Dashboard components loaded successfully")
        logger.info("🌐 Starting web server...")
        logger.info("📍 Dashboard will be available at: http://localhost:8000")
        logger.info("")
        logger.info("Available endpoints:")
        logger.info("  • Main Dashboard: http://localhost:8000/")
        logger.info("  • Executive View: http://localhost:8000/dashboard/executive")
        logger.info("  • Operational View: http://localhost:8000/dashboard/operational")
        logger.info("  • Technical View: http://localhost:8000/dashboard/technical")
        logger.info("  • API Health: http://localhost:8000/api/health")
        logger.info("  • KPI Data: http://localhost:8000/api/kpis")
        logger.info("")
        logger.info("✨ Features included:")
        logger.info("  ✅ Real-time KPI monitoring")
        logger.info("  ✅ Interactive charts and visualizations")
        logger.info("  ✅ Executive summary generation")
        logger.info("  ✅ Operational metrics tracking")
        logger.info("  ✅ Technical performance analysis")
        logger.info("  ✅ Real data from production processing")
        logger.info("  ✅ HITL data import capability")
        logger.info("  ⚠️  LLM enhancement (requires OPENAI_API_KEY)")
        logger.info("  ⚠️  PII masking (requires presidio)")
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            reload=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.error("Please install dashboard requirements:")
        logger.error("pip install -r requirements/requirements_dashboard.txt")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Dashboard stopped by user")
        
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()