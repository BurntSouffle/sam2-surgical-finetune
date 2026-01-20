# SAM2 Fine-tuning for Surgical Scene Segmentation

## Overview

Fine-tuning SAM2 (Segment Anything Model 2) with bounding box prompts for high-quality mask generation in laparoscopic cholecystectomy videos. This pipeline generates synthetic segmentation masks from bounding box annotations, enabling large-scale data expansion for surgical AI research.

## Key Results

| Model | Average IoU | Improvement |
|-------|-------------|-------------|
| Zero-shot SAM2-Large | 74.35% | baseline |
| **Fine-tuned SAM2-Large** | **78.12%** | **+3.77%** |

### Per-Class Performance

| Anatomy Class | Zero-Shot | Fine-Tuned | Δ |
|---------------|-----------|------------|---|
| Tool | 79.63% | 89.39% | +9.76% |
| Gallbladder | 80.56% | 85.38% | +4.82% |
| Cystic Artery | 66.81% | 71.08% | +4.27% |
| Cystic Duct | 69.81% | 72.91% | +3.10% |
| Cystic Plate | 71.89% | 73.53% | +1.64% |
| Calot Triangle | 77.41% | 76.41% | -1.00% |

## Dataset

This project uses the [Endoscapes Dataset](https://github.com/CAMMA-public/Endoscapes) for laparoscopic cholecystectomy video analysis.

- **Training**: 1,212 images (5,566 object instances)
- **Validation**: 409 images (1,733 instances)
- **Test**: 312 images (1,485 instances)
- **Classes**: 6 anatomical structures (cystic plate, calot triangle, cystic artery, cystic duct, gallbladder, tool)

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/sam2-surgical-finetune.git
cd sam2-surgical-finetune

# Create conda environment
conda create -n sam2_finetune python=3.10 -y
conda activate sam2_finetune

# Install dependencies
pip install -r requirements.txt

# Install SAM2
cd sam2
pip install -e .
cd ..

# Download SAM2 checkpoint
python scripts/download_checkpoint.py --model large
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Project Structure

```
sam2_finetune/
├── scripts/
│   ├── box_prompted/           # Main pipeline
│   │   ├── dataset_box.py      # Box-prompted dataset
│   │   ├── train_box.py        # Training script
│   │   ├── evaluate_box.py     # Evaluation script
│   │   └── generate_masks.py   # Synthetic mask generation
│   ├── download_checkpoint.py  # Checkpoint downloader
│   ├── step1_test_env.py       # Environment verification
│   ├── step2_test_sam2.py      # SAM2 installation test
│   └── ...
├── checkpoints/                # Model checkpoints
├── outputs/                    # Results and visualizations
├── sam2/                       # SAM2 repository (submodule)
├── requirements.txt
├── SETUP.md
├── LICENSE
└── README.md
```

## Usage

### 1. Training

```bash
# Fine-tune SAM2-Large with box prompts
python scripts/box_prompted/train_box.py --epochs 50 --batch_size 2

# Monitor training
tensorboard --logdir outputs/box_prompted/tensorboard
```

### 2. Evaluation

```bash
# Evaluate fine-tuned model
python scripts/box_prompted/evaluate_box.py \
    --checkpoint checkpoints/box_prompted/run_XXXXX/best_model.pt \
    --save_visualizations
```

### 3. Generate Synthetic Masks

```bash
# Generate masks for all splits
python scripts/box_prompted/generate_masks.py --split train
python scripts/box_prompted/generate_masks.py --split val
python scripts/box_prompted/generate_masks.py --split test
```

## Method

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SAM2-Large Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Image ──► [Image Encoder] ──┐                              │
│            (FROZEN 213M)     │                              │
│                              ├──► [Mask Decoder] ──► Mask   │
│  BBox  ──► [Prompt Encoder]──┘    (TRAINED 12M)             │
│            (FROZEN)                                          │
└─────────────────────────────────────────────────────────────┘
```

### Training Details

- **Model**: SAM2.1-Large (224M parameters)
- **Trainable**: Mask decoder only (12M parameters, 5%)
- **Loss**: Focal Loss + Dice Loss + IoU Prediction Loss
- **Optimizer**: AdamW (lr=1e-5, weight_decay=1e-4)
- **Batch size**: 2
- **Epochs**: 50 (best model at epoch 11)
- **Hardware**: NVIDIA RTX 3080 (12GB VRAM)

## Results

### Synthetic Mask Generation

| Split | Images | Masks Generated |
|-------|--------|-----------------|
| Train | 1,212 | 5,566 |
| Val | 409 | 1,733 |
| Test | 312 | 1,485 |
| **Total** | **1,933** | **8,784** |

### Data Expansion

- Original pixel-annotated: 493 images
- Synthetic masks generated: 1,933 images
- **4.6x increase in labeled data**

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sam2-surgical-finetune,
  author = {Abu Sufian Basith},
  title = {SAM2 Fine-tuning for Surgical Scene Segmentation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/BurntSouffle/sam2-surgical-finetune}
}
```

## Acknowledgments

- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2) by Meta AI
- [Endoscapes Dataset](https://github.com/CAMMA-public/Endoscapes) by CAMMA

## License

MIT License - see [LICENSE](LICENSE) for details.
