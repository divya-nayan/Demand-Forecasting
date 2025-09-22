#!/usr/bin/env python3
"""
Setup script for the Demand Forecasting Pipeline
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install project dependencies"""
    print("Installing dependencies...")
    try:
        # Try uv first (faster)
        result = subprocess.run(['uv', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Dependencies installed using uv")
        else:
            # Fallback to pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
            print("✓ Dependencies installed using pip")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)
    except FileNotFoundError:
        # uv not found, use pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
            print("✓ Dependencies installed using pip")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)


def create_directories():
    """Create necessary directories"""
    directories = [
        'demand_forecasting_outputs',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✓ Project directories created")


def verify_data_files():
    """Verify that required data files exist"""
    project_root = Path(__file__).parent.parent
    data_files = [
        'data/raw/static/training_data.csv',
        'data/raw/static/demand_data.csv'
    ]
    
    missing_files = []
    for file_path in data_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠ Warning: Missing data files: {missing_files}")
        print("Please ensure these files are available before running the pipeline")
    else:
        print("✓ Required data files found")


def main():
    """Main setup function"""
    print("Setting up Demand Forecasting Pipeline...")
    print("=" * 50)
    
    check_python_version()
    install_dependencies()
    create_directories()
    verify_data_files()
    
    print("=" * 50)
    print("✓ Setup complete!")
    print("\nTo run the pipeline:")
    print("  python main.py")
    print("\nTo upload results to database:")
    print("  python scripts/upload_to_database.py")
    print("\nTo explore the data:")
    print("  jupyter notebook notebooks/")


if __name__ == "__main__":
    main()