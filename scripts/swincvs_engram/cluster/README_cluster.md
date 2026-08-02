# E1 Cluster Execution Guide

## Overview

This directory contains scripts for running the E1 (Visual Engrams) experiment on the UCL CS cluster.

**Experiment**: SwinCVS with 5-channel input (RGB + GB mask + Tool mask)
**Hypothesis**: Explicit segmentation masks improve C1 assessment

## Prerequisites

Before running E1, ensure:
1. The coarse segmentation model checkpoint exists on the cluster
2. The Endoscapes dataset is available at `/SAN/medic/surgical_vision_sam2/data/endoscapes/`

### Copy Coarse Model to Cluster (if needed)

```bash
# From your local machine
scp -r sam2_finetune/checkpoints/coarse_segmentation_v2 \
    username@comic:/SAN/medic/surgical_vision_sam2/sam2_finetune/checkpoints/
```

## First-Time Setup (Run Once)

```bash
# SSH to cluster
ssh comic

# Navigate to project
cd /SAN/medic/surgical_vision_sam2/sam2_finetune

# Run setup script
bash scripts/swincvs_engram/cluster/setup_env.sh
```

This will:
- Create virtual environment `swincvs_engram_env`
- Install PyTorch with CUDA 11.8
- Install E1 dependencies (including `timm`)
- Install SAM2 as editable package
- Download SwinV2 pretrained weights (~400MB)

## Running E1

### Step 1: Verify Setup (Recommended)

Run the test job first to ensure everything is configured correctly:

```bash
# Submit test job (30 min, 16GB)
qsub scripts/swincvs_engram/cluster/run_e1_test.qsub.sh

# Monitor job status
qstat

# Check results (wait for job to complete)
cat logs/e1_test_*.log
```

All 6 tests should pass:
- Config Loading
- Coarse Model Loading
- Dataset Creation
- Model Building
- Forward Pass
- Single Training Step

### Step 2: Full Training

Once tests pass, submit the training job:

```bash
# Submit training job (12 hours, 32GB)
qsub scripts/swincvs_engram/cluster/run_e1_train.qsub.sh

# Monitor job
qstat

# Watch live output
tail -f logs/e1_train_*.log
```

## Monitor Progress

```bash
# Check job status
qstat

# View recent output
tail -50 logs/e1_train_*.log

# Search for specific metrics
grep "mAP" logs/e1_train_*.log
grep "Epoch" logs/e1_train_*.log
```

## Output Files

After training completes:

```
sam2_finetune/
├── weights/
│   ├── E1_SwinCVS_Engram_v1_bestMAP.pt      # Best mAP checkpoint
│   └── E1_SwinCVS_Engram_v1_lastEpoch.pt    # Final epoch checkpoint
└── SwinCVS/results/
    └── E1_SwinCVS_Engram_v1_results.json    # Training metrics
```

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/e1_train_*.log

# Common fixes:
# - Environment not activated: Re-run setup_env.sh
# - Missing checkpoint: Copy coarse model from local
# - CUDA error: Check GPU availability with nvidia-smi
```

### Out of Memory
Reduce batch size in config:
```yaml
TRAIN:
  BATCH_SIZE: 2  # Reduce from 4
```

### SAM2 Import Error
Re-install SAM2:
```bash
source /SAN/medic/surgical_vision_sam2/swincvs_engram_env/bin/activate
pip install -e /SAN/medic/surgical_vision_sam2/sam2_finetune/sam2/
```

### Missing timm
```bash
source /SAN/medic/surgical_vision_sam2/swincvs_engram_env/bin/activate
pip install timm
```

## File Reference

| File | Description |
|------|-------------|
| `setup_env.sh` | One-time environment setup |
| `SwinCVS_engram_config_cluster.yaml` | Cluster-specific config |
| `run_e1_test.qsub.sh` | Test job (30 min) |
| `run_e1_train.qsub.sh` | Training job (12 hours) |

## Contact

For issues with cluster resources, contact UCL CS support.
For experiment-specific issues, check the main README at `scripts/swincvs_engram/README.md`.
