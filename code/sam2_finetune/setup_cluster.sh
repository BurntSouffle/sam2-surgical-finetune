#!/bin/bash
#############################################################################
# Setup script for SAM2.1 Fine-tuning on UCL CS Cluster
# Run this script once to set up the environment
#############################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - UPDATE THESE
# ============================================================================

# Your project store path (update this!)
PROJECT_STORE="${PROJECT_STORE:-/project/YOUR_USERNAME}"

# Dataset location (update if different)
DATASET_ROOT="${DATASET_ROOT:-/data/endoscapes}"

echo "=============================================="
echo "SAM2 Fine-tuning Setup Script"
echo "=============================================="
echo ""
echo "Project store: ${PROJECT_STORE}"
echo "Dataset root: ${DATASET_ROOT}"
echo ""

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

echo "Creating directory structure..."

mkdir -p ${PROJECT_STORE}/code/sam2_finetune
mkdir -p ${PROJECT_STORE}/checkpoints/pretrained
mkdir -p ${PROJECT_STORE}/checkpoints/finetuned
mkdir -p ${PROJECT_STORE}/outputs/generated_masks
mkdir -p ${PROJECT_STORE}/logs

echo "  Done."

# ============================================================================
# SETUP PYTHON ENVIRONMENT
# ============================================================================

echo ""
echo "Setting up Python environment..."

# Source Python
source /share/apps/source_files/python/python-3.9.5.source

# Create virtual environment if it doesn't exist
if [ ! -d "${PROJECT_STORE}/venv" ]; then
    echo "  Creating virtual environment..."
    python -m venv ${PROJECT_STORE}/venv
fi

# Activate virtual environment
source ${PROJECT_STORE}/venv/bin/activate

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

echo ""
echo "Installing PyTorch..."

# Install PyTorch with CUDA support
# Check cluster documentation for recommended CUDA version
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing other dependencies..."

# Install from requirements.txt if it exists in current directory
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install core dependencies
    pip install opencv-python pillow numpy scipy pycocotools matplotlib seaborn tqdm pyyaml hydra-core iopath
fi

# ============================================================================
# CLONE SAM2 REPOSITORY
# ============================================================================

echo ""
echo "Setting up SAM2 repository..."

SAM2_REPO="${PROJECT_STORE}/segment-anything-2"

if [ ! -d "${SAM2_REPO}" ]; then
    echo "  Cloning SAM2 repository..."
    cd ${PROJECT_STORE}
    git clone https://github.com/facebookresearch/segment-anything-2.git

    echo "  Installing SAM2..."
    cd ${SAM2_REPO}
    pip install -e .
else
    echo "  SAM2 repository already exists at ${SAM2_REPO}"
fi

# ============================================================================
# DOWNLOAD PRETRAINED CHECKPOINT
# ============================================================================

echo ""
echo "Downloading pretrained SAM2.1 Large checkpoint..."

CHECKPOINT_FILE="${PROJECT_STORE}/checkpoints/pretrained/sam2.1_hiera_large.pt"

if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "  Downloading..."
    wget -q --show-progress -O ${CHECKPOINT_FILE} \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    echo "  Done."
else
    echo "  Checkpoint already exists at ${CHECKPOINT_FILE}"
fi

# ============================================================================
# VERIFY DATASET
# ============================================================================

echo ""
echo "Verifying dataset..."

if [ -d "${DATASET_ROOT}" ]; then
    echo "  Dataset found at ${DATASET_ROOT}"
    echo "  Contents:"
    ls -la ${DATASET_ROOT} | head -10
else
    echo "  WARNING: Dataset not found at ${DATASET_ROOT}"
    echo "  Please update DATASET_ROOT variable or create a symlink"
fi

# ============================================================================
# UPDATE CONFIGURATION
# ============================================================================

echo ""
echo "Updating configuration files..."

# Update config.py paths
CONFIG_FILE="${PROJECT_STORE}/code/sam2_finetune/config.py"
if [ -f "${CONFIG_FILE}" ]; then
    # Use sed to update paths (careful with escaping)
    sed -i "s|/project/YOUR_USERNAME|${PROJECT_STORE}|g" ${CONFIG_FILE}
    sed -i "s|/data/endoscapes|${DATASET_ROOT}|g" ${CONFIG_FILE}
    echo "  Updated config.py"
fi

# Update train_sam2.sh paths
TRAIN_SCRIPT="${PROJECT_STORE}/code/sam2_finetune/train_sam2.sh"
if [ -f "${TRAIN_SCRIPT}" ]; then
    sed -i "s|/project/YOUR_USERNAME|${PROJECT_STORE}|g" ${TRAIN_SCRIPT}
    sed -i "s|/data/endoscapes|${DATASET_ROOT}|g" ${TRAIN_SCRIPT}
    chmod +x ${TRAIN_SCRIPT}
    echo "  Updated train_sam2.sh"
fi

# ============================================================================
# VERIFICATION
# ============================================================================

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Verification:"
echo ""

# Test PyTorch
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

# Test imports
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"

# Test SAM2
cd ${PROJECT_STORE}/code/sam2_finetune
export PYTHONPATH="${SAM2_REPO}:${PYTHONPATH}"
python -c "from sam2.build_sam import build_sam2; print('  SAM2: OK')" 2>/dev/null || echo "  SAM2: Install may need verification"

echo ""
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo ""
echo "1. Verify paths in config.py:"
echo "   vim ${CONFIG_FILE}"
echo ""
echo "2. Submit training job:"
echo "   cd ${PROJECT_STORE}/code/sam2_finetune"
echo "   qsub train_sam2.sh"
echo ""
echo "3. Monitor training:"
echo "   qstat"
echo "   tail -f ${PROJECT_STORE}/logs/sam2_finetune.o*"
echo ""
