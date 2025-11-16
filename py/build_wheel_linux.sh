#!/bin/bash

# TrimCI Wheel Package Build Script for Linux
# This script automates the process of building the TrimCI wheel package on Linux systems

set -e  # Exit on any error

echo "Starting TrimCI wheel package build for Linux..."

# Step 1: Check system requirements and Python version
echo "Step 1: Checking system requirements..."

# Check if Python is available
if ! command -v Python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

PYTHON_CMD="Python"
PYTHON_VERSION=$(Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python command: $PYTHON_CMD (version: $PYTHON_VERSION)"

# Check for eigen3
echo "Checking for eigen3..."
if pkg-config --exists eigen3; then
    echo "eigen3 found via pkg-config"
elif [ -d "/usr/include/eigen3" ] || [ -d "/usr/local/include/eigen3" ]; then
    echo "eigen3 found in system include directories"
elif [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/include/eigen3" ]; then
    echo "eigen3 found in conda environment: $CONDA_PREFIX/include/eigen3"
else
    echo "Warning: eigen3 not found. Please install eigen3 using one of:"
    echo "  - conda install -c conda-forge eigen"
    echo "  - System package manager (apt-get install libeigen3-dev)"
    echo "Continuing build - CMake will attempt to find eigen3..."
fi

if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler is not installed"
    echo "Please install: sudo apt-get install build-essential (Ubuntu/Debian) or sudo yum groupinstall 'Development Tools' (CentOS/RHEL)"
    exit 1
fi

echo "System requirements check passed."

# Step 2: Check system dependencies
echo "Step 2: Checking system dependencies..."
echo "Note: This script assumes system dependencies are already installed."
echo "Required dependencies:"
echo "- OpenMP development libraries (libomp-dev or libgomp-devel)"
echo "- Eigen3 development libraries (libeigen3-dev or eigen3-devel)"
echo "- Python3 development headers (python3-dev or python3-devel)"
echo "- C++ compiler (g++)"
echo "If dependencies are missing, install them using your system package manager."

# Step 4: Clean previous build files
echo "Step 4: Cleaning previous build files..."
rm -rf build/
rm -rf dist/
rm -rf trimci.egg-info/
rm -f trimci_core.cpython-*.so
echo "Previous build files cleaned."

# Step 5: Install Python build dependencies
echo "Step 5: Installing Python build dependencies..."
# Handle externally-managed-environment in Python 3.13+
if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "Python 3.13+ detected, using --break-system-packages --user flags"
    $PYTHON_CMD -m pip install --break-system-packages --user "scikit-build-core>=0.9" "pybind11>=2.11" "cmake>=3.18" ninja wheel
else
    $PYTHON_CMD -m pip install --user "scikit-build-core>=0.9" "pybind11>=2.11" "cmake>=3.18" ninja wheel
fi

# Step 6: Build the wheel package
echo "Step 6: Building wheel package..."
$PYTHON_CMD -m pip wheel .

# Step 7: Verify the build
echo "Step 7: Verifying build results..."
if [ -d "dist" ] && [ "$(ls -A dist/*.whl 2>/dev/null)" ]; then
    echo "Build successful! Generated files:"
    ls -lh dist/
    
    # Optional: Check wheel file contents
    WHEEL_FILE=$(ls dist/*.whl | head -n 1)
    echo "\nChecking wheel file contents:"
    $PYTHON_CMD -m zipfile -l "$WHEEL_FILE"
else
    echo "Build failed! No wheel file found."
    exit 1
fi

# Step 8: Clean up temporary files
echo "Step 8: Cleaning up temporary files..."
echo "Note: build artifacts preserved under dist/."

echo "TrimCI wheel package build for Linux completed successfully!"
echo "\nNext steps:"
echo "1. Test the wheel: $PYTHON_CMD -m pip install --user dist/*.whl"
echo "2. Verify installation: $PYTHON_CMD -c 'import trimci; print(trimci.__version__)'"
echo "3. Run tests to ensure everything works correctly"