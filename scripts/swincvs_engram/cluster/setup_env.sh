#!/bin/bash
# =============================================================================
# E1 Environment Setup for UCL Cluster
# =============================================================================
# Run ONCE before submitting any jobs:
#   bash scripts/swincvs_engram/cluster/setup_env.sh
#
# This script:
# 1. Creates a virtual environment
# 2. Installs PyTorch with CUDA support
# 3. Installs E1 dependencies (including timm)
# 4. Installs SAM2 as editable package
# 5. Downloads SwinV2 pretrained weights
# =============================================================================

set -e  # Exit on error

PROJECT_DIR="/SAN/medic/surgical_vision_sam2"
SAM2_DIR="$PROJECT_DIR/sam2_finetune"
SWINCVS_DIR="$PROJECT_DIR/SwinCVS"
VENV_NAME="swincvs_engram_env"

echo "========================================"
echo "E1 Experiment - Environment Setup"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Load Python
echo "Loading Python 3.9.5..."
source /share/apps/source_files/python/python-3.9.5.source

cd $PROJECT_DIR

# Create virtual environment if not exists
if [ ! -d "$VENV_NAME" ]; then
    echo ""
    echo "Creating virtual environment: $VENV_NAME..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment already exists: $VENV_NAME"
fi

# Activate environment
source $VENV_NAME/bin/activate
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install E1 requirements
echo ""
echo "Installing E1 dependencies..."
if [ -f "$SAM2_DIR/scripts/swincvs_engram/requirements_e1.txt" ]; then
    pip install -r $SAM2_DIR/scripts/swincvs_engram/requirements_e1.txt
else
    echo "WARNING: requirements_e1.txt not found, installing core packages..."
    pip install timm pyyaml opencv-python-headless pillow pandas numpy matplotlib tqdm
fi

# Install SAM2 as editable package (fixes import shadowing issue)
echo ""
echo "Installing SAM2 as editable package..."
if [ -d "$SAM2_DIR/sam2" ]; then
    pip install -e $SAM2_DIR/sam2/
else
    echo "ERROR: SAM2 directory not found at $SAM2_DIR/sam2"
    exit 1
fi

# Download SwinV2 pretrained weights
echo ""
echo "Setting up SwinV2 pretrained weights..."
WEIGHTS_DIR="$SWINCVS_DIR/weights"
WEIGHTS_FILE="swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth"
WEIGHTS_URL="https://github.com/SwinTransformer/storage/releases/download/v2.0.0/$WEIGHTS_FILE"

mkdir -p $WEIGHTS_DIR

if [ ! -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    echo "Downloading SwinV2 pretrained weights..."
    echo "URL: $WEIGHTS_URL"
    cd $WEIGHTS_DIR
    wget -q --show-progress $WEIGHTS_URL
    if [ $? -eq 0 ]; then
        echo "Download complete: $WEIGHTS_DIR/$WEIGHTS_FILE"
        ls -lh $WEIGHTS_FILE
    else
        echo "ERROR: Failed to download weights"
        exit 1
    fi
    cd $PROJECT_DIR
else
    echo "SwinV2 weights already exist: $WEIGHTS_DIR/$WEIGHTS_FILE"
    ls -lh $WEIGHTS_DIR/$WEIGHTS_FILE
fi

# Create necessary directories
echo ""
echo "Creating output directories..."
mkdir -p $SAM2_DIR/logs
mkdir -p $SAM2_DIR/weights
mkdir -p $SAM2_DIR/results
mkdir -p $SWINCVS_DIR/weights
mkdir -p $SWINCVS_DIR/results

# Verify coarse model checkpoint exists
echo ""
echo "Checking coarse segmentation model..."
COARSE_CHECKPOINT="$SAM2_DIR/checkpoints/coarse_segmentation_v2/run_20260120_234049/best_model.pt"
if [ -f "$COARSE_CHECKPOINT" ]; then
    echo "Found: $COARSE_CHECKPOINT"
    ls -lh $COARSE_CHECKPOINT
else
    echo "WARNING: Coarse model checkpoint not found!"
    echo "Expected: $COARSE_CHECKPOINT"
    echo "You may need to copy it from your local machine."
fi

# Verify Endoscapes dataset
echo ""
echo "Checking Endoscapes dataset..."
ENDOSCAPES_DIR="$PROJECT_DIR/data/endoscapes"
if [ -d "$ENDOSCAPES_DIR" ]; then
    echo "Found: $ENDOSCAPES_DIR"
    ls $ENDOSCAPES_DIR | head -5
else
    echo "WARNING: Endoscapes dataset not found!"
    echo "Expected: $ENDOSCAPES_DIR"
fi

# Print summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Virtual environment: $PROJECT_DIR/$VENV_NAME"
echo ""
echo "To activate manually:"
echo "  source /share/apps/source_files/python/python-3.9.5.source"
echo "  source $PROJECT_DIR/$VENV_NAME/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Run test:  qsub $SAM2_DIR/scripts/swincvs_engram/cluster/run_e1_test.qsub.sh"
echo "  2. Check log: cat $SAM2_DIR/logs/e1_test_*.log"
echo "  3. Train:     qsub $SAM2_DIR/scripts/swincvs_engram/cluster/run_e1_train.qsub.sh"
echo "========================================"
