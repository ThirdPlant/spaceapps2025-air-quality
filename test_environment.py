#!/usr/bin/env python3
"""
Test script to diagnose environment issues
"""

import os
import sys

def test_environment():
    print("🔍 Environment Diagnostic Test")
    print("=" * 40)
    
    # Test 1: Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Test 2: Current working directory
    print(f"\nCurrent working directory: {os.getcwd()}")
    
    # Test 3: List files in current directory
    print(f"\nFiles in current directory:")
    try:
        files = os.listdir('.')
        for file in files:
            print(f"  - {file}")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    # Test 4: Check if we can create files
    print(f"\nTesting file creation...")
    try:
        test_file = "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("Test")
        print(f"✅ Successfully created {test_file}")
        
        # Clean up
        os.remove(test_file)
        print(f"✅ Successfully removed {test_file}")
    except Exception as e:
        print(f"❌ Error with file operations: {e}")
    
    # Test 5: Check required packages
    print(f"\nTesting package imports...")
    packages = [
        'numpy', 'xarray', 'matplotlib', 'cartopy', 'harmony'
    ]
    
    for package in packages:
        try:
            if package == 'harmony':
                from harmony import Client
            else:
                __import__(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
    
    print(f"\nDiagnostic complete!")

if __name__ == "__main__":
    test_environment()
