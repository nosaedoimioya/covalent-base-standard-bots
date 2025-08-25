#!/usr/bin/env python3
"""
Setup and test script for BaseShaper C++ implementation with Python bindings.
This script helps build the C++ module and run tests.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    # Check if cmake is available
    try:
        subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        print("✓ CMake found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CMake not found. Please install CMake.")
        return False
    
    # Check if make is available
    try:
        subprocess.run(["make", "--version"], check=True, capture_output=True)
        print("✓ Make found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Make not found. Please install Make.")
        return False
    
    # Check if pybind11 is available
    try:
        import pybind11
        print("✓ pybind11 Python package found")
    except ImportError:
        print("✗ pybind11 Python package not found. Please install: pip install pybind11")
        return False
    
    return True

def build_cpp_module():
    """Build the C++ module with Python bindings"""
    print("\nBuilding C++ module...")
    
    # Get the current directory
    current_dir = Path(__file__).parent
    cpp_dir = current_dir / ".." / "cpp" / "control"
    build_dir = cpp_dir / "build"
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    try:
        # Configure with CMake
        print("Configuring with CMake...")
        subprocess.run([
            "cmake", "..",
            "-DCMAKE_BUILD_TYPE=Release"
        ], cwd=build_dir, check=True)
        
        # Build
        print("Building...")
        subprocess.run(["make"], cwd=build_dir, check=True)
        
        print("✓ C++ module built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        return False

def run_python_tests():
    """Run the Python test script"""
    print("\nRunning Python tests...")
    
    test_script = Path(__file__).parent / "test_base_shaper_python.py"
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✓ Python tests passed!")
            return True
        else:
            print("✗ Python tests failed!")
            return False
            
    except Exception as e:
        print(f"✗ Failed to run Python tests: {e}")
        return False

def main():
    """Main setup and test function"""
    print("BaseShaper C++ Setup and Test Script")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing dependencies.")
        return 1
    
    # Build C++ module
    if not build_cpp_module():
        print("\n❌ Build failed.")
        return 1
    
    # Run Python tests
    if not run_python_tests():
        print("\n❌ Python tests failed.")
        return 1
    
    print("\n🎉 Setup and tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
