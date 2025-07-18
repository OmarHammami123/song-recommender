#!/usr/bin/env python3
"""
Song Recommender - Automated Setup Script

This script automates the data acquisition and setup process for the Song Recommender system.
Run this script after cloning the repository to set up everything automatically.

Usage:
    python setup.py [--dataset spotify|lastfm|million-song] [--skip-download]
    
Examples:
    python setup.py                          # Interactive setup
    python setup.py --dataset spotify        # Auto-download Spotify dataset
    python setup.py --skip-download          # Skip data download
"""

import os
import sys
import subprocess
import argparse
import json
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

# Configuration
DATASETS = {
    'spotify': {
        'name': 'Spotify Dataset 1921-2020 (160k+ tracks)',
        'kaggle_path': 'yamaerenay/spotify-dataset-19212020-160k-tracks',
        'file_mapping': {'data.csv': 'dataset.csv'},
        'description': 'Comprehensive Spotify dataset with audio features',
        'size': '~25MB',
        'recommended': True
    },
    'lastfm': {
        'name': 'Last.fm Music Dataset',
        'kaggle_path': 'pcbreviglieri/lastfm-music-artist-song-dataset',
        'file_mapping': {'lastfm_dataset.csv': 'dataset.csv'},
        'description': 'Last.fm dataset with user listening history',
        'size': '~15MB',
        'recommended': False
    },
    'million-song': {
        'name': 'Million Song Dataset (Subset)',
        'kaggle_path': 'notshrirang/million-song-dataset',
        'file_mapping': {'song_data.csv': 'dataset.csv'},
        'description': 'Subset of the Million Song Dataset',
        'size': '~10MB',
        'recommended': False
    }
}

class Colors:
    """Terminal colors for better output formatting"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message: str, color: str = Colors.GREEN) -> None:
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.END}")

def print_header(message: str) -> None:
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def check_command(command: str) -> bool:
    """Check if a command is available in PATH"""
    try:
        subprocess.run([command, '--version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL,
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_package(package: str) -> bool:
    """Install a Python package"""
    try:
        print_colored(f"Installing {package}...", Colors.YELLOW)
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                      check=True, capture_output=True)
        print_colored(f"✅ {package} installed successfully", Colors.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install {package}: {e}", Colors.RED)
        return False

def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured"""
    try:
        import kaggle
        kaggle.api.authenticate()
        print_colored("✅ Kaggle API configured correctly", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ Kaggle API not configured: {e}", Colors.RED)
        return False

def setup_environment() -> None:
    """Set up the environment file"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print_colored("✅ .env file already exists", Colors.GREEN)
        return
    
    if env_example.exists():
        print_colored("Creating .env file from template...", Colors.YELLOW)
        env_file.write_text(env_example.read_text())
        print_colored("✅ .env file created", Colors.GREEN)
        print_colored("⚠️  Please edit .env with your Kaggle credentials", Colors.YELLOW)
    else:
        print_colored("❌ .env.example not found", Colors.RED)

def create_directories() -> None:
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'logs',
        'models/__pycache__'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_colored("✅ Directory structure created", Colors.GREEN)

def download_dataset(dataset_key: str) -> bool:
    """Download dataset from Kaggle"""
    if dataset_key not in DATASETS:
        print_colored(f"❌ Unknown dataset: {dataset_key}", Colors.RED)
        return False
    
    dataset_info = DATASETS[dataset_key]
    
    print_colored(f"Downloading {dataset_info['name']}...", Colors.YELLOW)
    
    try:
        # Import kaggle after ensuring it's installed
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_info['kaggle_path'],
            path='data/raw',
            unzip=True
        )
        
        # Handle file mapping
        raw_path = Path('data/raw')
        for old_name, new_name in dataset_info['file_mapping'].items():
            old_file = raw_path / old_name
            new_file = raw_path / new_name
            
            if old_file.exists():
                if new_file.exists():
                    new_file.unlink()  # Remove existing file
                old_file.rename(new_file)
                print_colored(f"✅ Renamed {old_name} to {new_name}", Colors.GREEN)
            else:
                print_colored(f"⚠️  Expected file {old_name} not found", Colors.YELLOW)
        
        print_colored(f"✅ Dataset downloaded successfully", Colors.GREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Failed to download dataset: {e}", Colors.RED)
        return False

def validate_dataset() -> bool:
    """Validate the downloaded dataset"""
    dataset_path = Path('data/raw/dataset.csv')
    
    if not dataset_path.exists():
        print_colored("❌ Dataset file not found", Colors.RED)
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_columns = [
            'track_name', 'artists', 'danceability', 'energy', 
            'acousticness', 'instrumentalness', 'liveness', 
            'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print_colored(f"❌ Missing required columns: {missing_columns}", Colors.RED)
            return False
        
        print_colored(f"✅ Dataset validation passed", Colors.GREEN)
        print_colored(f"   - Rows: {len(df):,}", Colors.BLUE)
        print_colored(f"   - Columns: {len(df.columns)}", Colors.BLUE)
        print_colored(f"   - Size: {dataset_path.stat().st_size / 1024 / 1024:.1f} MB", Colors.BLUE)
        
        return True
        
    except Exception as e:
        print_colored(f"❌ Dataset validation failed: {e}", Colors.RED)
        return False

def show_dataset_options() -> str:
    """Show available datasets and get user selection"""
    print_header("Available Datasets")
    
    for i, (key, info) in enumerate(DATASETS.items(), 1):
        status = "⭐ RECOMMENDED" if info['recommended'] else ""
        print(f"{i}. {Colors.BOLD}{info['name']}{Colors.END} {status}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   Kaggle: {info['kaggle_path']}")
        print()
    
    while True:
        try:
            choice = input("Select a dataset (1-3) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(DATASETS):
                return list(DATASETS.keys())[choice_num - 1]
            else:
                print_colored("Invalid choice. Please try again.", Colors.RED)
        
        except ValueError:
            print_colored("Invalid input. Please enter a number.", Colors.RED)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Song Recommender Setup Script')
    parser.add_argument('--dataset', choices=list(DATASETS.keys()), 
                       help='Dataset to download automatically')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download')
    args = parser.parse_args()
    
    print_header("Song Recommender Setup")
    print_colored("🎵 Setting up your Song Recommender system...", Colors.BLUE)
    
    # 1. Check prerequisites
    print_header("Checking Prerequisites")
    
    if not check_command('python'):
        print_colored("❌ Python not found in PATH", Colors.RED)
        sys.exit(1)
    
    print_colored("✅ Python found", Colors.GREEN)
    
    # 2. Install required packages
    print_header("Installing Dependencies")
    
    required_packages = ['pandas', 'numpy', 'streamlit', 'scikit-learn', 'matplotlib', 'seaborn']
    
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"✅ {package} already installed", Colors.GREEN)
        except ImportError:
            if not install_package(package):
                print_colored(f"❌ Failed to install {package}. Please install manually.", Colors.RED)
                sys.exit(1)
    
    # Install kaggle if not skipping download
    if not args.skip_download:
        try:
            import kaggle
            print_colored("✅ Kaggle API already available", Colors.GREEN)
        except ImportError:
            if not install_package('kaggle'):
                print_colored("❌ Failed to install Kaggle API", Colors.RED)
                sys.exit(1)
    
    # 3. Set up environment
    print_header("Setting Up Environment")
    setup_environment()
    create_directories()
    
    # 4. Download dataset
    if not args.skip_download:
        print_header("Dataset Setup")
        
        if not check_kaggle_setup():
            print_colored("Please configure Kaggle API first:", Colors.YELLOW)
            print_colored("1. Go to: https://www.kaggle.com/settings/account", Colors.BLUE)
            print_colored("2. Create API token and download kaggle.json", Colors.BLUE)
            print_colored("3. Place in ~/.kaggle/ directory", Colors.BLUE)
            print_colored("4. Edit .env file with your credentials", Colors.BLUE)
            print_colored("5. Run this script again", Colors.BLUE)
            sys.exit(1)
        
        # Select dataset
        if args.dataset:
            dataset_key = args.dataset
        else:
            dataset_key = show_dataset_options()
        
        # Download and validate
        if download_dataset(dataset_key):
            if validate_dataset():
                print_colored("✅ Dataset ready for processing", Colors.GREEN)
            else:
                print_colored("⚠️  Dataset downloaded but validation failed", Colors.YELLOW)
        else:
            print_colored("❌ Dataset download failed", Colors.RED)
            sys.exit(1)
    
    # 5. Final instructions
    print_header("Setup Complete!")
    print_colored("🎉 Your Song Recommender system is ready!", Colors.GREEN)
    print()
    print_colored("Next steps:", Colors.BLUE)
    print_colored("1. Run: jupyter notebook data/explore.ipynb", Colors.YELLOW)
    print_colored("2. Execute all cells to process the data", Colors.YELLOW)
    print_colored("3. Run: streamlit run app.py", Colors.YELLOW)
    print_colored("4. Open your browser and start exploring!", Colors.YELLOW)
    print()
    print_colored("📚 For more information, see the README.md file", Colors.BLUE)

if __name__ == '__main__':
    main()
