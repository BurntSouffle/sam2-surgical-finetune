"""
SAM2 Box-Prompted Fine-tuning Evaluation Script

Evaluates fine-tuned SAM2 model on test set and compares against zero-shot baselines.
Generates comparison tables and visualizations for dissertation.

Usage:
    python scripts/box_prompted/evaluate_box.py --checkpoint path/to/best_model.pt
    python scripts/box_prompted/evaluate_box.py --checkpoint path/to/best_model.pt --save_visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from dataset_box import EndoscapesBoxDataset, get_split_paths, CLASS_NAMES, CLASS_COLORS

# SAM2 imports
from sam2.build_sam import build_sam2

# Import the trainer wrapper
from train_box import SAM2BoxTrainer, CONFIG, collate_fn


# =============================================================================
# ZERO-SHOT BASELINES (from previous SAM2-Large experiments)
# =============================================================================
ZERO_SHOT_BASELINES = {
    'gallbladder': 80.56,
    'tool': 79.63,
    'cystic_duct': 69.81,
    'cystic_artery': 66.81,
    'calot_triangle': 77.41,
    'cystic_plate': 71.89,
}

# Class ID to name mapping (for consistent ordering)
CLASS_ORDER = [
    (5, 'gallbladder'),
    (6, 'tool'),
    (4, 'cystic_duct'),
    (3, 'cystic_artery'),
    (2, 'calot_triangle'),
    (1, 'cystic_plate'),
]


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth binary masks."""
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between predicted and ground truth masks."""
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)

    intersection = (pred_binary * gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2 * intersection) / total


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_finetuned_model(checkpoint_path: str, device: torch.device) -> SAM2BoxTrainer:
    """Load fine-tuned SAM2 model from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load base SAM2 model
    sam2_model = build_sam2(
        config_file=CONFIG['model_cfg'],
        ckpt_path=str(CONFIG['checkpoint']),
        device='cpu',
        mode='eval',
    )

    # Create trainer wrapper
    model = SAM2BoxTrainer(
        sam2_model=sam2_model,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        image_size=CONFIG['image_size'],
    )

    # Load fine-tuned mask decoder weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.model.sam_mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: mIoU = {checkpoint['metrics'].get('mean_class_iou', 0)*100:.1f}%")

    return model


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: SAM2BoxTrainer,
    dataloader: DataLoader,
    device: torch.device,
    save_visualizations: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model on dataset.

    Returns:
        Dict with per-class and overall metrics
    """
    model.eval()

    # Track per-class metrics
    class_ious = {class_id: [] for class_id, _ in CLASS_ORDER}
    class_dices = {class_id: [] for class_id, _ in CLASS_ORDER}

    # For visualizations
    vis_samples = {class_id: [] for class_id, _ in CLASS_ORDER}

    pbar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes = batch['boxes'].to(device)
        gt_masks = batch['masks'].to(device)
        class_ids = batch['class_ids']

        # Forward pass
        outputs = model(images, boxes)
        pred_masks = outputs['masks']

        # Convert to numpy for metrics
        pred_masks_np = torch.sigmoid(pred_masks).cpu().numpy()
        gt_masks_np = gt_masks.cpu().numpy()

        # Compute metrics for each sample in batch
        for i, class_id in enumerate(class_ids):
            pred = pred_masks_np[i, 0]  # (H, W)
            gt = gt_masks_np[i, 0]  # (H, W)

            iou = compute_iou(pred, gt)
            dice = compute_dice(pred, gt)

            class_ious[class_id].append(iou)
            class_dices[class_id].append(dice)

            # Store samples for visualization (up to 5 per class)
            if save_visualizations and len(vis_samples[class_id]) < 5:
                # Denormalize image
                img = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

                vis_samples[class_id].append({
                    'image': img,
                    'bbox': boxes[i].cpu().numpy(),
                    'gt_mask': gt,
                    'pred_mask': pred,
                    'iou': iou,
                })

    # Compute per-class averages
    results = {
        'per_class': {},
        'overall': {},
    }

    all_ious = []
    for class_id, class_name in CLASS_ORDER:
        ious = class_ious[class_id]
        dices = class_dices[class_id]

        if ious:
            mean_iou = np.mean(ious) * 100
            mean_dice = np.mean(dices) * 100
            std_iou = np.std(ious) * 100

            results['per_class'][class_name] = {
                'iou': mean_iou,
                'dice': mean_dice,
                'std': std_iou,
                'count': len(ious),
            }
            all_ious.append(mean_iou)
        else:
            results['per_class'][class_name] = {
                'iou': 0.0,
                'dice': 0.0,
                'std': 0.0,
                'count': 0,
            }

    # Overall average
    results['overall']['mean_iou'] = np.mean(all_ious) if all_ious else 0.0

    return results, vis_samples


# =============================================================================
# OUTPUT GENERATION
# =============================================================================
def generate_comparison_table(results: Dict, output_path: Path):
    """Generate comparison table in text format."""

    lines = []
    lines.append("")
    lines.append("=" * 76)
    lines.append(" SAM2-Large Box-Prompted Segmentation: Zero-Shot vs Fine-Tuned")
    lines.append("=" * 76)
    lines.append("")

    # Table header
    lines.append("+" + "-" * 18 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")
    lines.append("| {:<16} | {:<12} | {:<12} | {:<12} | {:<8} |".format(
        "Anatomy Class", "Zero-Shot", "Fine-Tuned", "Improvement", "Samples"
    ))
    lines.append("+" + "-" * 18 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")

    # Data rows
    total_improvement = 0
    for class_id, class_name in CLASS_ORDER:
        zero_shot = ZERO_SHOT_BASELINES.get(class_name, 0)
        fine_tuned = results['per_class'][class_name]['iou']
        improvement = fine_tuned - zero_shot
        count = results['per_class'][class_name]['count']
        total_improvement += improvement

        # Format class name nicely
        display_name = class_name.replace('_', ' ').title()

        imp_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"

        lines.append("| {:<16} | {:>10.2f}% | {:>10.2f}% | {:>12} | {:>8} |".format(
            display_name, zero_shot, fine_tuned, imp_str, count
        ))

    # Separator
    lines.append("+" + "-" * 18 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")

    # Average row
    zero_shot_avg = np.mean(list(ZERO_SHOT_BASELINES.values()))
    fine_tuned_avg = results['overall']['mean_iou']
    avg_improvement = fine_tuned_avg - zero_shot_avg

    imp_str = f"+{avg_improvement:.2f}%" if avg_improvement >= 0 else f"{avg_improvement:.2f}%"

    lines.append("| {:<16} | {:>10.2f}% | {:>10.2f}% | {:>12} | {:>8} |".format(
        "AVERAGE", zero_shot_avg, fine_tuned_avg, imp_str, "-"
    ))
    lines.append("+" + "-" * 18 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 10 + "+")

    lines.append("")

    # Save
    table_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(table_text)

    # Also print
    print(table_text)

    return table_text


def generate_bar_chart(results: Dict, output_path: Path):
    """Generate bar chart comparing zero-shot vs fine-tuned."""

    fig, ax = plt.subplots(figsize=(12, 6))

    classes = []
    zero_shot_values = []
    fine_tuned_values = []

    for class_id, class_name in CLASS_ORDER:
        display_name = class_name.replace('_', ' ').title()
        classes.append(display_name)
        zero_shot_values.append(ZERO_SHOT_BASELINES.get(class_name, 0))
        fine_tuned_values.append(results['per_class'][class_name]['iou'])

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, zero_shot_values, width, label='Zero-Shot SAM2-Large', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, fine_tuned_values, width, label='Fine-Tuned SAM2-Large', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('IoU (%)', fontsize=12)
    ax.set_title('SAM2-Large Box-Prompted Segmentation:\nZero-Shot vs Fine-Tuned on Endoscapes', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Add average lines
    zero_shot_avg = np.mean(zero_shot_values)
    fine_tuned_avg = np.mean(fine_tuned_values)

    ax.axhline(y=zero_shot_avg, color='#3498db', linestyle='--', alpha=0.5, label=f'Zero-Shot Avg: {zero_shot_avg:.1f}%')
    ax.axhline(y=fine_tuned_avg, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Fine-Tuned Avg: {fine_tuned_avg:.1f}%')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved bar chart: {output_path}")


def save_visualizations(vis_samples: Dict, output_dir: Path):
    """Save visualization images for each class."""

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in CLASS_ORDER:
        samples = vis_samples[class_id]
        if not samples:
            continue

        class_dir = vis_dir / class_name
        class_dir.mkdir(exist_ok=True)

        for i, sample in enumerate(samples):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            image = sample['image']
            bbox = sample['bbox']
            gt_mask = sample['gt_mask']
            pred_mask = sample['pred_mask']
            iou = sample['iou']

            # Column 1: Image with bbox
            axes[0].imshow(image)
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].set_title(f'Input + BBox\n{class_name.replace("_", " ").title()}')
            axes[0].axis('off')

            # Column 2: Ground truth mask
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # Column 3: Predicted mask
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title(f'Prediction\nIoU: {iou*100:.1f}%')
            axes[2].axis('off')

            # Column 4: Overlay
            overlay = image.copy()
            pred_binary = pred_mask > 0.5
            color = np.array(CLASS_COLORS.get(class_id, (0, 255, 0))) / 255.0
            overlay[pred_binary] = overlay[pred_binary] * 0.5 + color * 0.5
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(class_dir / f'sample_{i+1}_iou{iou*100:.0f}.png', dpi=100, bbox_inches='tight')
            plt.close()

    print(f"Saved visualizations: {vis_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate SAM2 Box-Prompted Fine-tuning')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned checkpoint (best_model.pt)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as checkpoint)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualization images')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load model
    model = load_finetuned_model(str(checkpoint_path), device)

    # Load test dataset
    print("\nLoading test dataset...")
    test_images_dir, masks_dir = get_split_paths('test')
    test_dataset = EndoscapesBoxDataset(
        images_dir=str(test_images_dir),
        masks_dir=str(masks_dir),
        split='test',
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")

    # Evaluate
    print("\n" + "=" * 50)
    print(" Running Evaluation")
    print("=" * 50)

    results, vis_samples = evaluate_model(
        model, test_loader, device,
        save_visualizations=args.save_visualizations,
        output_dir=output_dir,
    )

    # Generate outputs
    print("\n" + "=" * 50)
    print(" Generating Outputs")
    print("=" * 50)

    # 1. Comparison table
    generate_comparison_table(results, output_dir / 'comparison_table.txt')

    # 2. Bar chart
    generate_bar_chart(results, output_dir / 'comparison_chart.png')

    # 3. Save metrics JSON (convert numpy types to Python types for JSON serialization)
    def to_python_type(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(v) for v in obj]
        return obj

    metrics_output = {
        'checkpoint': str(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'zero_shot_baselines': ZERO_SHOT_BASELINES,
        'fine_tuned_results': to_python_type(results),
        'improvement': {
            class_name: float(results['per_class'][class_name]['iou'] - ZERO_SHOT_BASELINES.get(class_name, 0))
            for _, class_name in CLASS_ORDER
        },
        'overall_improvement': float(results['overall']['mean_iou'] - np.mean(list(ZERO_SHOT_BASELINES.values()))),
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=2)

    print(f"Saved metrics: {output_dir / 'metrics.json'}")

    # 4. Visualizations (if requested)
    if args.save_visualizations:
        save_visualizations(vis_samples, output_dir)

    # Final summary
    print("\n" + "=" * 50)
    print(" EVALUATION COMPLETE")
    print("=" * 50)

    zero_shot_avg = np.mean(list(ZERO_SHOT_BASELINES.values()))
    fine_tuned_avg = results['overall']['mean_iou']
    improvement = fine_tuned_avg - zero_shot_avg

    print(f"""
Summary:
  Zero-Shot Average:  {zero_shot_avg:.2f}%
  Fine-Tuned Average: {fine_tuned_avg:.2f}%
  Improvement:        {'+' if improvement >= 0 else ''}{improvement:.2f}%

Output files:
  - {output_dir / 'comparison_table.txt'}
  - {output_dir / 'comparison_chart.png'}
  - {output_dir / 'metrics.json'}
""")

    if args.save_visualizations:
        print(f"  - {output_dir / 'visualizations/'}")


if __name__ == '__main__':
    main()
