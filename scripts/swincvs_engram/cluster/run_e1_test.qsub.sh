#!/bin/bash
# =============================================================================
# E1 Test Job - Verify Setup Before Full Training
# =============================================================================
# Submit with: qsub scripts/swincvs_engram/cluster/run_e1_test.qsub.sh
# Monitor:     qstat
# Check log:   cat logs/e1_test_*.log
# =============================================================================

#$ -S /bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=00:30:00
#$ -l gpu=true
#$ -N E1_test
#$ -wd /SAN/medic/surgical_vision_sam2/sam2_finetune
#$ -j y
#$ -o logs/e1_test_$JOB_ID.log

# Exit on error
set -e

# Environment setup
source /share/apps/source_files/python/python-3.9.5.source
source /SAN/medic/surgical_vision_sam2/swincvs_engram_env/bin/activate

echo "========================================"
echo "E1 TEST JOB"
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
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"
echo ""

# Run test script
echo "Running E1 setup verification tests..."
echo ""

python scripts/swincvs_engram/test_e1_setup.py \
    --config_path scripts/swincvs_engram/cluster/SwinCVS_engram_config_cluster.yaml

echo ""
echo "========================================"
echo "Test finished: $(date)"
echo "========================================"
