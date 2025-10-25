"""
Environment setup helper for memory-efficient AI training.
This script checks system requirements and helps set up the correct environment.
"""
import sys
import platform
import os
from pathlib import Path

def check_python_version():
    """Check if we have a compatible Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor == 10:
        print("✓ Python 3.10 detected")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} detected. Please install Python 3.10")
        print("Download from: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe")
        return False

def check_ram():
    """Check available system RAM."""
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
        print(f"✓ System RAM: {total_ram:.1f} GB")
        if total_ram < 7.5:
            print("! Warning: Less than 8GB RAM available. Training may be slow.")
        return True
    except ImportError:
        print("! Could not check RAM. Installing psutil...")
        try:
            import pip
            pip.main(['install', 'psutil'])
            return check_ram()
        except:
            print("✗ Failed to install psutil")
            return False

def setup_environment():
    """Set up the Python environment with correct packages."""
    try:
        # Ensure pip is available
        import ensurepip
        ensurepip.bootstrap()
        
        # Install required packages
        import pip
        print("\nInstalling required packages...")
        
        # Install CPU version of PyTorch first
        print("\nInstalling PyTorch (CPU version)...")
        result = pip.main(['install', 'torch==2.1.0', '--index-url', 'https://download.pytorch.org/whl/cpu'])
        if result != 0:
            print("✗ Failed to install PyTorch")
            return False
            
        # Install other requirements
        requirements = [
            'transformers>=4.35.2',
            'sentencepiece',
            'tokenizers',
            'safetensors',
            'accelerate>=0.20.3',
            'datasets>=2.10.0',
            'einops',
            'scipy',
            'bitsandbytes>=0.41.0'
        ]
        
        for req in requirements:
            print(f"\nInstalling {req}...")
            result = pip.main(['install', req])
            if result != 0:
                print(f"✗ Failed to install {req}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error during setup: {str(e)}")
        return False

def main():
    """Run all checks and setup."""
    print("Checking system requirements...")
    print("-" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
        
    # Check RAM
    if not check_ram():
        return False
    
    # Setup environment
    print("\nSetting up Python environment...")
    print("-" * 50)
    if not setup_environment():
        return False
    
    print("\n✓ Setup completed successfully!")
    return True

if __name__ == "__main__":
    main()