#!/bin/bash
#############################################################################
# SGE Job Script for SAM2.1 Fine-tuning on Endoscapes2023
# UCL CS Cluster (vic.cs.ucl.ac.uk)
#############################################################################

#$ -S /bin/bash
#$ -N sam2_finetune
#$ -l tmem=60G
#$ -l h_rt=72:00:00
#$ -l gpu=true
#$ -pe gpu 4
#$ -R y
#$ -j y
#$ -o /project/YOUR_USERNAME/logs/
#$ -cwd

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
PROJECT_STORE="/project/YOUR_USERNAME"
DATASET_ROOT="/data/endoscapes"
CODE_DIR="${PROJECT_STORE}/code/sam2_finetune"
SAM2_REPO="${PROJECT_STORE}/segment-anything-2"

# Training parameters (can override defaults from config.py)
NUM_EPOCHS=30
BATCH_SIZE=2

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=============================================="
echo "SAM2.1 Fine-tuning Job Started"
echo "=============================================="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${JOB_ID}"
echo "Working directory: $(pwd)"
echo ""

# Source Python environment
echo "Setting up Python environment..."
source /share/apps/source_files/python/python-3.9.5.source

# Activate virtual environment (if using one)
if [ -d "${PROJECT_STORE}/venv" ]; then
    echo "Activating virtual environment..."
    source ${PROJECT_STORE}/venv/bin/activate
fi

# Check Python version
echo "Python version: $(python --version)"
echo "Python location: $(which python)"

# ============================================================================
# GPU SETUP
# ============================================================================

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Set CUDA device order to match nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUs available: ${NUM_GPUS}"

# Set visible devices (use all available)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

# For multi-GPU training with torchrun
export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 29500-29999 -n 1)  # Random port to avoid conflicts

echo ""
echo "Distributed training configuration:"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  World size: ${NUM_GPUS}"

# ============================================================================
# CHECK PREREQUISITES
# ============================================================================

echo ""
echo "Checking prerequisites..."

# Check SAM2 repo
if [ ! -d "${SAM2_REPO}" ]; then
    echo "ERROR: SAM2 repository not found at ${SAM2_REPO}"
    echo "Please clone it first: git clone https://github.com/facebookresearch/segment-anything-2 ${SAM2_REPO}"
    exit 1
fi

# Check code directory
if [ ! -f "${CODE_DIR}/train.py" ]; then
    echo "ERROR: Training script not found at ${CODE_DIR}/train.py"
    exit 1
fi

# Check dataset
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_ROOT}"
    exit 1
fi

echo "All prerequisites satisfied."

# ============================================================================
# DOWNLOAD PRETRAINED CHECKPOINT (if needed)
# ============================================================================

CHECKPOINT_DIR="${PROJECT_STORE}/checkpoints/pretrained"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/sam2.1_hiera_large.pt"

if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo ""
    echo "Downloading SAM2.1 Large checkpoint..."
    mkdir -p ${CHECKPOINT_DIR}
    wget -q --show-progress -O ${CHECKPOINT_FILE} \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    echo "Download complete."
else
    echo "Pretrained checkpoint found at ${CHECKPOINT_FILE}"
fi

# ============================================================================
# RUN TRAINING
# ============================================================================

echo ""
echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo ""

cd ${CODE_DIR}

# Add SAM2 to Python path
export PYTHONPATH="${SAM2_REPO}:${PYTHONPATH}"

# Run with torchrun for distributed training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --project-store ${PROJECT_STORE} \
    --dataset-root ${DATASET_ROOT} \
    --epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE}

TRAIN_EXIT_CODE=$?

# ============================================================================
# POST-TRAINING
# ============================================================================

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "End time: $(date)"

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Checkpoints saved to: ${PROJECT_STORE}/checkpoints/finetuned/"
    echo "Logs saved to: ${PROJECT_STORE}/logs/"

    # List saved checkpoints
    echo ""
    echo "Saved checkpoints:"
    ls -la ${PROJECT_STORE}/checkpoints/finetuned/
else
    echo ""
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}"
fi

exit ${TRAIN_EXIT_CODE}
