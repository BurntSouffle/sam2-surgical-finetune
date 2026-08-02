"""
Visualize engram masks for E1 experiment.
Creates a grid showing original RGB + GB mask overlay + Tool mask overlay.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
SWINCVS_ROOT = PROJECT_ROOT.parent / 'SwinCVS'

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import cv2
from PIL import Image

from scripts.f_environment import get_config
from f_dataset_engram import get_engram_datasets


def create_mask_visualization(output_path: Path, num_samples: int = 8):
    """Create visualization grid of RGB images with mask overlays."""

    print("="*60)
    print(" ENGRAM MASK VISUALIZATION")
    print("="*60)

    # Load config and datasets
    config_path = SCRIPT_DIR / 'config' / 'SwinCVS_engram_config.yaml'
    config, _ = get_config(str(config_path))

    print("\nLoading datasets...")
    train_dataset, val_dataset, test_dataset = get_engram_datasets(config)

    # Use training dataset for visualization
    dataset = train_dataset
    print(f"Dataset size: {len(dataset)}")

    # Select random samples
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Create figure
    fig = plt.figure(figsize=(20, num_samples * 3))
    gs = GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.1)

    print(f"\nGenerating visualizations for {len(indices)} samples...")

    for i, idx in enumerate(indices):
        print(f"  Processing sample {i+1}/{len(indices)} (idx={idx})...")

        # Get sample (shape: [T, C, H, W] for LSTM dataset)
        sample, label = dataset[idx]

        # Take middle frame from sequence
        if sample.dim() == 4:  # [T, C, H, W]
            frame = sample[2]  # Middle frame
        else:  # [C, H, W]
            frame = sample

        # Extract channels
        # Channels are normalized - need to denormalize
        rgb = frame[:3].numpy()  # [3, H, W]
        gb_mask = frame[3].numpy()  # [H, W]
        tool_mask = frame[4].numpy()  # [H, W]

        # Denormalize RGB (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        rgb = rgb * std + mean
        rgb = np.clip(rgb, 0, 1)
        rgb = rgb.transpose(1, 2, 0)  # [H, W, 3]

        # Denormalize masks (mask_mean=0.5, mask_std=0.5)
        gb_mask = gb_mask * 0.5 + 0.5  # Back to [0, 1]
        tool_mask = tool_mask * 0.5 + 0.5

        # Threshold masks for overlay
        gb_binary = (gb_mask > 0.5).astype(np.float32)
        tool_binary = (tool_mask > 0.5).astype(np.float32)

        # Create overlays
        gb_overlay = rgb.copy()
        gb_overlay[..., 0] = np.clip(gb_overlay[..., 0] + gb_binary * 0.5, 0, 1)  # Magenta = R + B
        gb_overlay[..., 2] = np.clip(gb_overlay[..., 2] + gb_binary * 0.5, 0, 1)

        tool_overlay = rgb.copy()
        tool_overlay[..., 1] = np.clip(tool_overlay[..., 1] + tool_binary * 0.5, 0, 1)  # Cyan = G + B
        tool_overlay[..., 2] = np.clip(tool_overlay[..., 2] + tool_binary * 0.5, 0, 1)

        # Combined overlay
        combined = rgb.copy()
        combined[..., 0] = np.clip(combined[..., 0] + gb_binary * 0.4, 0, 1)
        combined[..., 2] = np.clip(combined[..., 2] + gb_binary * 0.4, 0, 1)
        combined[..., 1] = np.clip(combined[..., 1] + tool_binary * 0.4, 0, 1)
        combined[..., 2] = np.clip(combined[..., 2] + tool_binary * 0.4, 0, 1)

        # Plot
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(rgb)
        ax1.set_title(f'RGB (idx={idx})', fontsize=10)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(gb_overlay)
        ax2.set_title(f'GB Mask (magenta)', fontsize=10)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(tool_overlay)
        ax3.set_title(f'Tool Mask (cyan)', fontsize=10)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(combined)
        label_str = f"C1={int(label[0])},C2={int(label[1])},C3={int(label[2])}"
        ax4.set_title(f'Combined [{label_str}]', fontsize=10)
        ax4.axis('off')

    # Add column headers
    fig.text(0.14, 0.98, 'Original RGB', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.38, 0.98, 'GB Mask Overlay', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.62, 0.98, 'Tool Mask Overlay', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.86, 0.98, 'Combined + Labels', ha='center', fontsize=14, fontweight='bold')

    plt.suptitle('E1 Engram Mask Visualization', fontsize=16, fontweight='bold', y=1.0)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Print mask statistics
    print("\n" + "="*60)
    print("MASK STATISTICS (last sample)")
    print("="*60)
    print(f"GB mask:   min={gb_mask.min():.3f}, max={gb_mask.max():.3f}, mean={gb_mask.mean():.3f}")
    print(f"Tool mask: min={tool_mask.min():.3f}, max={tool_mask.max():.3f}, mean={tool_mask.mean():.3f}")
    print(f"GB coverage:   {gb_binary.mean()*100:.1f}%")
    print(f"Tool coverage: {tool_binary.mean()*100:.1f}%")


if __name__ == '__main__':
    output_path = PROJECT_ROOT / 'diagnostics' / 'e1_masks_visualization.png'
    create_mask_visualization(output_path, num_samples=8)
