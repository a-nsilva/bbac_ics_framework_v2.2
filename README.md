# BBAC ICS Framework

**Behavioral-Based Access Control Framework for Industrial Control Systems**

A hybrid access control system combining rule-based policies, behavioral analysis, and machine learning for adaptive security in ICS environments.

## Features

- **Tri-layer Hybrid Architecture**
  - Layer 1: Rule-based Access Control (RuBAC)
  - Layer 2: Behavioral Analysis (Markov chains)
  - Layer 3: ML Anomaly Detection (Isolation Forest)

- **Adaptive Learning**
  - Sliding window baseline (70% recent + 30% historical)
  - Continuous profile updates with trust filter
  - Drift detection and adaptation

- **ROS2 Integration**
  - Real-time decision making (sub-100ms target)
  - Multi-agent support (robots + humans)
  - Emergency alert system

## Installation

### Prerequisites

**System Requirements:**
- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill
- **Python 3.10** (strict requirement for ROS2 Humble compatibility)

**Verify Python version:**
```bash
python3 --version
# Should output: Python 3.10.x
```

### Setup

#### 1. Install ROS2 Humble

Follow official guide: https://docs.ros.org/en/humble/Installation.html
```bash
# Quick install (Ubuntu 22.04)
sudo apt update
sudo apt install ros-humble-desktop
```

#### 2. Install System Dependencies
```bash
# Install Python system packages
sudo apt install -y \
  python3-pip \
  python3-numpy \
  python3-scipy \
  python3-pandas \
  python3-yaml \
  python3-sklearn \
  python3-matplotlib \
  python3-seaborn \
  python3-pytest
```

#### 3. Clone and Setup Framework
```bash
# Create workspace
mkdir -p ~/bbac_ws/src
cd ~/bbac_ws/src

# Clone repository
git clone https://github.com/yourusername/bbac-framework.git
cd bbac-framework

# Install Python dependencies (versions matched to ROS2 Humble)
pip3 install -r requirements.txt

# Important: This installs plotly and ensures correct numpy/scipy versions
```

#### 4. Build ROS2 Package
```bash
# Go to workspace root
cd ~/bbac_ws

# Build package
colcon build --packages-select bbac_framework

# Source workspace
source install/setup.bash

# Add to bashrc for convenience
echo "source ~/bbac_ws/install/setup.bash" >> ~/.bashrc
```

#### 5. Verify Installation
```bash
# Check if package is available
ros2 pkg list | grep bbac

# Should output: bbac_framework

# Verify Python imports
python3 -c "import numpy, scipy, pandas, sklearn, matplotlib, seaborn, plotly; print('All dependencies OK')"
```

## Usage

### Run BBAC Node
```bash
# Terminal 1: Launch BBAC node
ros2 run bbac_framework bbac_node.py

# Or with launch file (configurable parameters)
ros2 launch bbac_framework bbac.launch.py \
  enable_behavioral:=true \
  enable_ml:=true \
  enable_policy:=true
```

### Run Experiments
```bash
# Terminal 1: BBAC node (must be running)
ros2 run bbac_framework bbac_node.py

# Terminal 2: Run experiments sequentially
python3 experiments/ablation_study.py
python3 experiments/baseline_comparison.py
python3 experiments/adaptive_eval.py
```

## Configuration

Edit YAML files in `config/`:

- `baseline.yaml` - Baseline window settings (sliding window: 70% recent + 30% historical)
- `fusion.yaml` - Layer fusion weights (default: rule=0.4, behavioral=0.3, ml=0.3)
- `thresholds.yaml` - Decision thresholds (T1=0.7 grant, T2=0.5 MFA, T3=0.3 review)
- `ros_params.yaml` - System parameters (target latency: 100ms)

## Architecture
```
bbac_framework/
├── config/          # Configuration files (YAML)
├── data/            # Dataset (100k samples)
│   └── 100k/
│       ├── bbac_trainer.csv
│       ├── bbac_validation.csv
│       ├── bbac_test.csv
│       ├── agents.json
│       └── anomaly_metadata.json
├── experiments/     # Evaluation scripts
│   ├── ablation_study.py
│   ├── baseline_comparison.py
│   └── adaptive_eval.py
├── launch/          # ROS2 launch files
├── msg/             # ROS2 custom messages
├── src/
│   ├── core/        # 5 framework layers
│   │   ├── ingestion.py     # Layer 1: Auth + preprocessing
│   │   ├── modeling.py      # Layer 2: Baseline + profiles
│   │   ├── analysis.py      # Layer 3: Statistical + Sequence + Policy
│   │   ├── fusion.py        # Layer 4a: Score fusion
│   │   ├── decision.py      # Layer 4b: Risk classification + RBAC
│   │   └── learning.py      # Layer 5: Continuous learning
│   ├── models/      # ML models (LSTM - future)
│   ├── ros/         # ROS2 integration
│   │   └── bbac_node.py
│   └── util/        # Utilities
│       ├── config_loader.py
│       ├── data_structures.py
│       ├── evaluator.py
│       ├── logger.py
│       └── visualizer.py
└── tests/           # Unit tests
```

## Experiments

### Ablation Study
Tests individual layers vs hybrid approach:
- Rule-only (RuBAC)
- Statistical-only
- Sequence-only (Markov)
- Statistical + Sequence
- **Full hybrid** (all layers)

**Results:** `results/ablation_study/`

### Baseline Comparison
Compares BBAC against traditional methods:
- RBAC (Role-Based)
- ABAC (Attribute-Based)
- Rule-based only
- Behavioral-only
- **BBAC** (hybrid)

**Results:** `results/baseline_comparison/`

### Adaptive Evaluation
Evaluates 6 key ADAPTIVE/DYNAMIC metrics:

**ADAPTIVE:**
1. Baseline convergence rate
2. Drift detection accuracy
3. Sliding window effectiveness (70/30 validation)

**DYNAMIC:**
4. Rule update latency (<1s target)
5. Conflict resolution rate (100% target)

**INTERACTION:**
6. Concurrent drift + rule change handling

**Results:** `results/adaptive_evaluation/`

## Results

All results saved to `results/` with publication-quality figures:
```
results/
├── ablation_study/
│   ├── figures/
│   │   ├── roc_ablation.png
│   │   ├── metrics_ablation.png
│   │   └── latency_ablation.png
│   ├── ablation_summary.txt
│   └── *_results.json
├── baseline_comparison/
│   ├── figures/
│   │   ├── roc_baseline_comparison.png
│   │   ├── pr_baseline_comparison.png
│   │   └── metrics_baseline_comparison.png
│   ├── comparison_summary.txt
│   └── statistical_comparisons.json
└── adaptive_evaluation/
    ├── figures/
    │   ├── convergence_*.png
    │   └── sliding_window_comparison.png
    ├── adaptive_summary.txt
    └── adaptive_dynamic_results.json
```

## Python Version Compatibility

**Critical:** This framework requires **Python 3.10 exactly** due to ROS2 Humble constraints.

Package versions are carefully selected to avoid conflicts:
- `numpy <1.25.0` - Compatible with scipy/sklearn/tensorflow
- `scipy <1.12.0` - Avoids apt package conflicts
- `scikit-learn ~=1.3.0` - Allows patches (1.3.x) but not 1.4.x

**Do not upgrade** to Python 3.11+ or newer package versions without testing ROS2 compatibility.

## Troubleshooting

### Import Errors
```bash
# If you get numpy/scipy import errors:
pip3 uninstall numpy scipy scikit-learn
pip3 install -r requirements.txt
```

### ROS2 Node Not Found
```bash
# Re-source workspace
source ~/bbac_ws/install/setup.bash

# Rebuild if needed
cd ~/bbac_ws
colcon build --packages-select bbac_framework
```

### Permission Denied on Scripts
```bash
# Make scripts executable
chmod +x experiments/*.py
chmod +x src/ros/bbac_node.py
```

## Citation

If you use this framework in your research, please cite:
```bibtex
@article{yourname2026bbac,
  title={BBAC: Behavioral-Based Access Control for Industrial Control Systems},
  author={Your Name and Coauthor Name},
  journal={Journal Name},
  year={2026},
  note={Target IF>7}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

- Author: Your Name (your.email@example.com)
- GitHub: https://github.com/yourusername/bbac-framework
- Issues: https://github.com/yourusername/bbac-framework/issues

## Acknowledgments

- ROS2 Humble community
- scikit-learn, numpy, scipy contributors
- Research institution name