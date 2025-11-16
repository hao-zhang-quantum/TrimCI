#!/bin/bash

# TrimCI ARM64 Installation Script
# This script installs TrimCI wheel package and its dependencies on ARM64 systems

set -e

echo "======================================"
echo "TrimCI ARM64 Installation Script"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check system dependencies
echo "Step 1: Checking system dependencies..."

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✓ Found python3 version: $PYTHON_VERSION"
else
    echo "✗ Error: python3 not found"
    echo "Please install Python 3.8+ first:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

# Check pip
if command_exists pip; then
    echo "✓ Found pip"
    PIP_CMD="pip"
elif command_exists pip3; then
    echo "✓ Found pip3"
    PIP_CMD="pip3"
else
    echo "✗ Error: pip not found"
    echo "Please install pip first:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3-pip"
    exit 1
fi

# Step 2: Check system dependencies...

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Warning: Cannot detect Linux distribution"
    OS="unknown"
fi

echo "Detected OS: $OS"
echo "Note: This script assumes system dependencies are already installed."
echo "Required system dependencies:"

# Show installation commands based on distribution
case $OS in
    ubuntu|debian)
        echo "For Ubuntu/Debian, install dependencies with:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install -y libeigen3-dev g++ cmake"
        ;;
    centos|rhel|fedora)
        if command_exists dnf; then
            echo "For Fedora/RHEL 8+, install dependencies with:"
            echo "  sudo dnf install -y eigen3-devel gcc-c++ cmake"
        elif command_exists yum; then
            echo "For CentOS/RHEL 7, install dependencies with:"
            echo "  sudo yum install -y eigen3-devel gcc-c++ cmake"
        fi
        ;;
    *)
        echo "For other distributions, install:"
        echo "  - Eigen3 development libraries"
        echo "  - C++ compiler (g++)"
        echo "  - CMake build system"
        echo "Or use conda: conda install -c conda-forge eigen"
        ;;
esac

echo "If using conda environment, ensure eigen is installed:"
echo "  conda install -c conda-forge eigen"

# Step 3: Install TrimCI wheel package
echo "Step 3: Installing TrimCI ARM64 wheel package..."

# Clean up any existing trimci installations first
echo "Cleaning up existing TrimCI installations..."
$PIP_CMD uninstall trimci -y 2>/dev/null || true

# Remove any leftover .so files
find /opt/homebrew/lib/python*/site-packages -name "trimci_core*.so" -delete 2>/dev/null || true
find /Users/*/Library/Python/*/lib/python/site-packages -name "trimci_core*.so" -delete 2>/dev/null || true

# Find wheel file (look for macOS ARM64 wheel first, then any ARM64 wheel)
WHEEL_FILE=$(find . -name "trimci-*-*-*-macosx_*_arm64.whl" | head -1)

if [ -z "$WHEEL_FILE" ]; then
    # Fallback to Linux ARM64 wheel
    WHEEL_FILE=$(find . -name "trimci-*-*-*-linux_aarch64.whl" | head -1)
fi

if [ -z "$WHEEL_FILE" ]; then
    # Fallback to any wheel file
    WHEEL_FILE=$(find . -name "trimci-*.whl" | head -1)
fi

if [ -z "$WHEEL_FILE" ]; then
    echo "✗ Error: No TrimCI wheel file found in current directory"
    echo "Please make sure you have built the ARM64 wheel package first:"
    echo "  ./build_wheel_arm64.sh"
    exit 1
fi

echo "Found wheel file: $WHEEL_FILE"

# Install the wheel
echo "Installing TrimCI..."
# Use the same Python executable that pip uses
PYTHON_CMD=$(which python3 2>/dev/null || which python)
# Handle externally-managed-environment in Python 3.13+
if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "Python 3.13+ detected, using --break-system-packages flags"
    $PIP_CMD install "$WHEEL_FILE" --no-deps --break-system-packages --force-reinstall
else
    $PIP_CMD install "$WHEEL_FILE" --no-deps --force-reinstall
fi

# Step 4: Verify installation
echo "Step 4: Verifying installation..."

# Use the same Python executable for verification
PYTHON_CMD=$(which python3 2>/dev/null || which python)

if $PYTHON_CMD -c "import trimci; print('TrimCI version:', trimci.__version__)" 2>/dev/null; then
    echo "✓ TrimCI ARM64 installation successful!"
else
    echo "✗ TrimCI installation failed"
    echo "Detailed error information:"
    $PYTHON_CMD -c "import trimci; print('TrimCI version:', trimci.__version__)" 2>&1 || true
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo "ARM64 installation completed successfully!"
echo "======================================"