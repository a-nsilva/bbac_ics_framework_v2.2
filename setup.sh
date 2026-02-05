#!/bin/bash
# BBAC Framework - Setup (assumes Python 3.10 from devcontainer)
set -e

echo "=========================================="
echo "BBAC Framework - Complete Setup"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[$1]${NC} $2"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

PROJECT_DIR=$(pwd)
WORKSPACE_DIR="$HOME/bbac_ics_ws"

# ============================================================================
# Validar Python 3.10
# ============================================================================

print_step "0" "Validating Python 3.10"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    print_error "Python 3.10 required! (found $PYTHON_VERSION)"
    exit 1
fi

print_success "Python 3.10 OK"

# ============================================================================
# Install Python Dependencies
# ============================================================================

print_step "1" "Installing dependencies"

python3 -m pip install --upgrade pip setuptools wheel -q
python3 -m pip install -r requirements.txt -q

print_success "Dependencies installed"

# ============================================================================
# Setup ROS2
# ============================================================================

print_step "2" "Setting up ROS2"

source /opt/ros/humble/setup.bash

if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

print_success "ROS2 configured"

# ============================================================================
# Create Workspace
# ============================================================================

print_step "3" "Creating workspace"

rm -rf "$WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR/src"

cp -r "$PROJECT_DIR" "$WORKSPACE_DIR/src/bbac_framework"

# Clean
rm -rf "$WORKSPACE_DIR/src/bbac_framework/.git"
find "$WORKSPACE_DIR/src/bbac_framework" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$WORKSPACE_DIR/src/bbac_framework" -type f -name "*.pyc" -delete 2>/dev/null || true

print_success "Project copied"

# ============================================================================
# Build
# ============================================================================

print_step "4" "Building"

cd "$WORKSPACE_DIR"
source /opt/ros/humble/setup.bash

colcon build --packages-select bbac_framework --symlink-install
source "$WORKSPACE_DIR/install/setup.bash"

# Add to bashrc
if ! grep -q "source $WORKSPACE_DIR/install/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# BBAC Framework" >> ~/.bashrc
    echo "source $WORKSPACE_DIR/install/setup.bash" >> ~/.bashrc
fi

print_success "Build complete"

# ============================================================================
# Verification
# ============================================================================

print_step "5" "Verification"

if ros2 pkg list | grep -q "bbac_framework"; then
    print_success "Package registered"
else
    print_error "Package NOT found!"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next: source ~/.bashrc (or open new terminal)"
echo ""
