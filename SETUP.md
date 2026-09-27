# Detailed Setup Instructions

This guide provides step-by-step instructions for setting up the SAM2 fine-tuning environment.

## Prerequisites

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended for SAM2-Large)
- **CUDA**: 11.8 or 12.1
- **Conda**: Anaconda or Miniconda

## Step 1: Clone the Repository

```bash
git clone https://github.com/BurntSouffle/sam2-surgical-finetune.git
cd sam2-surgical-finetune
```

## Step 2: Create Conda Environment

```bash
# Create environment
conda create -n sam2_finetune python=3.10 -y
conda activate sam2_finetune

# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Step 3: Install SAM2

```bash
# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2

# Install in editable mode
pip install -e .

# Return to project root
cd ..
```

## Step 4: Download SAM2 Checkpoint

```bash
# Download SAM2-Large checkpoint (recommended)
python scripts/download_checkpoint.py --model large

# Or download all checkpoints
python scripts/download_checkpoint.py --model all
```

The checkpoint will be saved to `checkpoints/sam2.1_hiera_large.pt`.

## Step 5: Download Endoscapes Dataset

1. Visit the [Endoscapes GitHub repository](https://github.com/CAMMA-public/Endoscapes)
2. Follow their instructions to download the dataset
3. Organize the data as follows:

```
endoscapes/
├── train/
│   ├── frame_XXXXX_endo.jpg
│   └── annotation_coco.json
├── val/
│   ├── frame_XXXXX_endo.jpg
│   └── annotation_coco.json
├── test/
│   ├── frame_XXXXX_endo.jpg
│   └── annotation_coco.json
└── semseg/
    └── frame_XXXXX_endo_seg.png
```

## Step 6: Update Configuration Paths

Edit `scripts/box_prompted/train_box.py` and update the paths:

```python
CONFIG = {
    'checkpoint': Path(r'path/to/checkpoints/sam2.1_hiera_large.pt'),
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    # ... other settings
}
```

Similarly, update paths in:
- `scripts/box_prompted/dataset_box.py`
- `scripts/box_prompted/evaluate_box.py`
- `scripts/box_prompted/generate_masks.py`

## Step 7: Verify Installation

```bash
# Test environment
python scripts/step1_test_env.py

# Test SAM2 installation
python scripts/step2_test_sam2.py

# Test dataset loading
python scripts/box_prompted/dataset_box.py
```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce batch size in `train_box.py`
2. Use a smaller model (tiny or base)
3. Enable gradient checkpointing

### SAM2 Import Errors

Ensure SAM2 is installed in editable mode:
```bash
cd sam2
pip install -e .
```

### Windows Path Issues

On Windows, use raw strings for paths:
```python
path = Path(r"C:\Users\...\endoscapes")
```

### pycocotools Installation (Windows)

If pycocotools fails to install on Windows:
```bash
pip install pycocotools-windows
```

Or build from source:
```bash
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

## Hardware Requirements

| Model | VRAM Required | Training Time (50 epochs) |
|-------|---------------|---------------------------|
| SAM2-Tiny | 4 GB | ~2 hours |
| SAM2-Base | 6 GB | ~4 hours |
| SAM2-Large | 10 GB | ~8 hours |

*Times based on NVIDIA RTX 3080 with batch size 2*

## Directory Structure After Setup

```
sam2_finetune/
├── checkpoints/
│   └── sam2.1_hiera_large.pt
├── sam2/
│   ├── sam2/
│   ├── configs/
│   └── ...
├── scripts/
│   └── box_prompted/
├── outputs/
└── ...

endoscapes/
├── train/
├── val/
├── test/
└── semseg/
```

## Next Steps

After setup is complete:

1. **Train the model**:
   ```bash
   python scripts/box_prompted/train_box.py --epochs 50
   ```

2. **Evaluate results**:
   ```bash
   python scripts/box_prompted/evaluate_box.py --checkpoint checkpoints/box_prompted/run_XXXXX/best_model.pt
   ```

3. **Generate synthetic masks**:
   ```bash
   python scripts/box_prompted/generate_masks.py --split train
   ```

## Support

If you encounter issues:
1. Check the [Issues](https://github.com/BurntSouffle/sam2-surgical-finetune/issues) page
2. Ensure all paths are correctly configured
3. Verify CUDA and PyTorch compatibility
