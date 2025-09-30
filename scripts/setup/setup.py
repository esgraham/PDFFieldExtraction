#!/usr/bin/env python3
"""
Setup script for Azure PDF Listener

This script helps set up the Azure PDF Listener environment.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    if not os.path.exists(".env"):
        if os.path.exists("config/.env.example"):
            print("📝 Creating .env file from template...")
            
            # Copy example file
            with open("config/.env.example", "r") as src:
                content = src.read()
            
            with open(".env", "w") as dst:
                dst.write(content)
            
            print("✅ Created .env file")
            print("⚠️  Please edit .env file with your Azure Storage credentials")
            return True
        else:
            print("❌ config/.env.example file not found")
            return False
    else:
        print("ℹ️  .env file already exists")
        return True

def create_downloads_directory():
    """Create downloads directory for PDF files."""
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
        print("✅ Created downloads directory")
    else:
        print("ℹ️  Downloads directory already exists")

def main():
    """Run setup steps."""
    print("🚀 Azure PDF Listener Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Create .env file
    if not create_env_file():
        return 1
    
    # Create downloads directory
    create_downloads_directory()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your Azure Storage credentials")
    print("2. Run 'python test_setup.py' to verify configuration")
    print("3. Run 'python example_usage.py' to start monitoring")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())