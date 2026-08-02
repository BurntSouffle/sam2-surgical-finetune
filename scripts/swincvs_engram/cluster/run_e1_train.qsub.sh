#!/bin/bash
# =============================================================================
# E1 Training Job - Full Training Run
# =============================================================================
# Submit with: qsub scripts/swincvs_engram/cluster/run_e1_train.qsub.sh
# Monitor:     qstat; tail -f logs/e1_train_*.log
# =============================================================================

#$ -S /bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -l gpu=true
#$ -N E1_train
#$ -wd /SAN/medic/surgical_vision_sam2/sam2_finetune
#$ -j y
#$ -o logs/e1_train_$JOB_ID.log

# Exit on error
set -e

# Environment setup
source /share/apps/source_files/python/python-3.9.5.source
source /SAN/medic/surgical_vision_sam2/swincvs_engram_env/bin/activate

echo "========================================"
echo "E1 TRAINING JOB"
echo "========================================"
echo "Start time: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo "========================================"
echo ""

# Record start time for duration calculation
START_TIME=$(date +%s)

# Run training
echo "Starting E1 training..."
echo "Config: scripts/swincvs_engram/cluster/SwinCVS_engram_config_cluster.yaml"
echo ""

python scripts/swincvs_engram/SwinCVS_engram.py \
    --config_path scripts/swincvs_engram/cluster/SwinCVS_engram_config_cluster.yaml

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "E1 Training Complete!"
echo "========================================"
echo "End time: $(date)"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to:"
echo "  - Weights: weights/E1_SwinCVS_Engram_v1_*.pt"
echo "  - Results: SwinCVS/results/E1_SwinCVS_Engram_v1_results.json"
echo "========================================"
