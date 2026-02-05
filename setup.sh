#!/bin/bash
# BBAC Framework - Complete Setup (Simplified)
set -e

echo "=========================================="
echo "BBAC Framework - Complete Setup"
echo "=========================================="
echo ""

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[$1]${NC} $2"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

PROJECT_DIR=$(pwd)
WORKSPACE_DIR="$HOME/ros2_ws"

# ============================================================================
# FASE 0: Validação Python 3.10
# ============================================================================

print_step "0" "Validating Python version"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python detected: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" != "3" ] || [ "$PYTHON_MINOR" != "10" ]; then
    print_error "Python 3.10 required, found $PYTHON_VERSION"
    echo ""
    echo "Install Python 3.10:"
    echo "  sudo apt update"
    echo "  sudo apt install -y python3.10 python3.10-dev"
    echo "  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1"
    echo "  sudo update-alternatives --config python3"
    echo ""
    exit 1
fi

print_success "Python 3.10 OK"

# ============================================================================
# FASE 1: Setup Básico Python + ROS2
# ============================================================================

print_step "1" "Basic Setup"

# Install pip if needed
if ! command -v pip3 &> /dev/null; then
    print_step "1.0" "Installing pip..."
    sudo apt-get update -qq
    sudo apt-get install -y python3-pip > /dev/null 2>&1
    print_success "pip3 installed"
fi

# Source ROS2
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    print_error "ROS2 Humble not found!"
    echo "Install ROS2 Humble first: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

source /opt/ros/humble/setup.bash

if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    print_success "Added ROS2 to ~/.bashrc"
fi

# Install Python dependencies
print_step "1.1" "Installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel -q
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt -q
    print_success "Python packages installed"
else
    print_warning "requirements.txt not found"
fi

# ============================================================================
# FASE 2: Setup ROS2 Workspace
# ============================================================================

print_step "2" "Setting up ROS2 Workspace"

# Clean and create workspace
print_step "2.1" "Creating workspace at $WORKSPACE_DIR"
if [ -d "$WORKSPACE_DIR" ]; then
    print_warning "Workspace exists, cleaning..."
    rm -rf "$WORKSPACE_DIR"
fi
mkdir -p "$WORKSPACE_DIR/src"
print_success "Workspace created"

# Copy project
print_step "2.2" "Copying project to workspace..."
cp -r "$PROJECT_DIR" "$WORKSPACE_DIR/src/bbac_framework"

# Clean unnecessary files
rm -rf "$WORKSPACE_DIR/src/bbac_framework/.git"
find "$WORKSPACE_DIR/src/bbac_framework" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$WORKSPACE_DIR/src/bbac_framework" -type f -name "*.pyc" -delete 2>/dev/null || true
print_success "Project copied"

# Create msg and srv directories if not exist
print_step "2.3" "Creating message/service directories..."
mkdir -p "$WORKSPACE_DIR/src/bbac_framework/msg"
mkdir -p "$WORKSPACE_DIR/src/bbac_framework/srv"
print_success "Directories created"

# Create message files (only if they don't exist)
print_step "2.4" "Creating ROS2 messages..."

if [ ! -f "$WORKSPACE_DIR/src/bbac_framework/msg/AccessRequest.msg" ]; then
cat > "$WORKSPACE_DIR/src/bbac_framework/msg/AccessRequest.msg" << 'EOF'
std_msgs/Header header
string request_id
string agent_id
string agent_type
string agent_role
string action
string resource
string resource_type
string location
bool human_present
bool emergency
string session_id
string previous_action
string auth_status
uint8 attempt_count
string policy_id
string zone
float64 priority
string[] context_data
EOF
fi

if [ ! -f "$WORKSPACE_DIR/src/bbac_framework/msg/AccessDecision.msg" ]; then
cat > "$WORKSPACE_DIR/src/bbac_framework/msg/AccessDecision.msg" << 'EOF'
std_msgs/Header header
string request_id
string decision
float64 confidence
float64 latency_ms
string reason
LayerDecisionDetail[] layer_decisions
bool logged
string log_id
EOF
fi

if [ ! -f "$WORKSPACE_DIR/src/bbac_framework/msg/LayerDecisionDetail.msg" ]; then
cat > "$WORKSPACE_DIR/src/bbac_framework/msg/LayerDecisionDetail.msg" << 'EOF'
string layer_name
string decision
float64 confidence
float64 latency_ms
string[] explanation_keys
string[] explanation_values
EOF
fi

if [ ! -f "$WORKSPACE_DIR/src/bbac_framework/msg/EmergencyAlert.msg" ]; then
cat > "$WORKSPACE_DIR/src/bbac_framework/msg/EmergencyAlert.msg" << 'EOF'
std_msgs/Header header
string emergency_type
string severity
uint8 priority
bool active
string status
string zone
string[] affected_resources
string[] required_actions
string[] notifications
string description
string[] context_data
EOF
fi

print_success "Messages created (4 files)"

# Create service file
print_step "2.5" "Creating service..."

if [ ! -f "$WORKSPACE_DIR/src/bbac_framework/srv/ReconfigureLayers.srv" ]; then
cat > "$WORKSPACE_DIR/src/bbac_framework/srv/ReconfigureLayers.srv" << 'EOF'
# Request
bool layer2_enabled
bool layer3_enabled
---
# Response
bool success
string message
string active_layers
EOF
print_success "Service created"
else
print_success "Service already exists"
fi

# ============================================================================
# FASE 3: Build Workspace
# ============================================================================

print_step "3" "Building Workspace"

cd "$WORKSPACE_DIR"
source /opt/ros/humble/setup.bash

print_step "3.1" "Building bbac_framework..."
colcon build --packages-select bbac_framework --symlink-install
print_success "bbac_framework built"

# Source workspace
source "$WORKSPACE_DIR/install/setup.bash"

# Add to bashrc
if ! grep -q "source $WORKSPACE_DIR/install/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# BBAC Framework" >> ~/.bashrc
    echo "source $WORKSPACE_DIR/install/setup.bash" >> ~/.bashrc
    print_success "Added workspace to ~/.bashrc"
fi

# ============================================================================
# FASE 4: Verification
# ============================================================================

print_step "4" "Verification"

# Check package
if ros2 pkg list 2>/dev/null | grep -q "bbac_framework"; then
    print_success "bbac_framework registered"
else
    print_warning "bbac_framework NOT found"
fi

# Check interfaces
MSG_COUNT=$(ros2 interface list 2>/dev/null | grep "bbac_framework" | wc -l)
if [ "$MSG_COUNT" -ge 5 ]; then
    print_success "Interfaces registered ($MSG_COUNT total)"
else
    print_warning "Some interfaces missing (found $MSG_COUNT, expected 5)"
fi

# ============================================================================
# FINALIZAÇÃO
# ============================================================================

echo ""
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "System Ready:"
echo "  ✓ ROS2 workspace: $WORKSPACE_DIR"
echo "  ✓ Python 3.10 validated"
echo "  ✓ Package built: bbac_framework"
echo "  ✓ Messages: AccessRequest, AccessDecision, LayerDecisionDetail, EmergencyAlert"
echo "  ✓ Service: ReconfigureLayers"
echo ""
echo "Next Steps:"
echo ""
echo "  1. Open NEW terminal (or: source ~/.bashrc)"
echo ""
echo "  2. Train models (REQUIRED):"
echo "     cd $WORKSPACE_DIR/src/bbac_framework"
echo "     python3 scripts/train_models.py --data-dir data/100k"
echo ""
echo "  3. Run experiments:"
echo ""
echo "     Terminal 1 - BBAC Node:"
echo "       ros2 run bbac_framework bbac_node.py"
echo ""
echo "     Terminal 2 - Experiments:"
echo "       cd $WORKSPACE_DIR/src/bbac_framework"
echo "       python3 experiments/ablation_study.py"
echo "       python3 experiments/baseline_comparison.py"
echo "       python3 experiments/adaptive_eval.py"
echo ""
echo "=========================================="
echo ""
