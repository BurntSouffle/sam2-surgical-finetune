#!/bin/bash
# ========================================
# SAM2 Fine-tuning Ablation Study
# Run all experiment configurations
# ========================================

set -e  # Exit on error

echo "========================================"
echo "SAM2 Fine-tuning Ablation Study"
echo "========================================"
echo ""
echo "This script will run 4 experiments:"
echo "  1. Baseline: Simple decoder + CE loss"
echo "  2. Focal:    Simple decoder + Focal loss"
echo "  3. UNet:     UNet decoder + CE loss"
echo "  4. UNet+Focal: UNet decoder + Focal loss"
echo ""
echo "Each experiment will train for 50 epochs."
echo "Estimated total time: 4-8 hours (depending on GPU)"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate conda environment (adjust path as needed)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate sam2_finetune
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate sam2_finetune conda environment"
        echo "Please create it first or activate manually"
        exit 1
    fi
else
    echo "WARNING: conda not found, assuming environment is already activated"
fi

echo ""
echo "========================================"
echo "[1/4] Baseline: Simple decoder + CE loss"
echo "========================================"
echo ""
python "${SCRIPT_DIR}/train_experiment.py" \
    --name baseline \
    --decoder simple \
    --loss ce \
    --epochs 50 \
    --batch_size 16

echo ""
echo "========================================"
echo "[2/4] Focal: Simple decoder + Focal loss"
echo "========================================"
echo ""
python "${SCRIPT_DIR}/train_experiment.py" \
    --name focal \
    --decoder simple \
    --loss focal \
    --gamma 2.0 \
    --epochs 50 \
    --batch_size 16

echo ""
echo "========================================"
echo "[3/4] UNet: UNet decoder + CE loss"
echo "========================================"
echo ""
python "${SCRIPT_DIR}/train_experiment.py" \
    --name unet_ce \
    --decoder unet \
    --loss ce \
    --epochs 50 \
    --batch_size 16

echo ""
echo "========================================"
echo "[4/4] UNet+Focal: UNet decoder + Focal loss"
echo "========================================"
echo ""
python "${SCRIPT_DIR}/train_experiment.py" \
    --name unet_focal \
    --decoder unet \
    --loss focal \
    --gamma 2.0 \
    --epochs 50 \
    --batch_size 16

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in: checkpoints/experiments/"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir checkpoints/experiments"
echo ""
