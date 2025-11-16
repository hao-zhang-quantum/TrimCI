#!/usr/bin/env bash
set -euo pipefail

echo "[TrimCI] Building wheel for macOS ARM64 (Apple Silicon)"

# Step 1: Set Homebrew LLVM toolchain (clang/clang++)
HOMEBREW_LLVM=$(brew --prefix llvm 2>/dev/null || echo "/opt/homebrew/opt/llvm")
export CC="$HOMEBREW_LLVM/bin/clang"
export CXX="$HOMEBREW_LLVM/bin/clang++"

# Step 2: Ensure build frontend is available
if Python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
  Python -m pip install --break-system-packages --user build >/dev/null
else
  Python -m pip install --user build >/dev/null
fi

# Step 3: Clean previous build artifacts
rm -rf ../dist

# Step 4: Build wheel from repository root
pushd .. >/dev/null
Python -m build -w
popd >/dev/null

# Step 5: Verify build results
if ls ../dist/trimci-*.whl >/dev/null 2>&1; then
  echo "[TrimCI] Wheel built:"
  ls -lh ../dist/trimci-*.whl
else
  echo "Build failed! No wheel file found in ../dist."
  exit 1
fi