"""
Evaluation Script for Coarse (4-class) Semantic Segmentation

Evaluates trained coarse segmentation model and compares to 7-class model.

Usage:
    python scripts/coarse_segmentation/evaluate_coarse.py --checkpoint path/to/best_model.pt
    python scripts/coarse_segmentation/evaluate_coarse.py --checkpoint path/to/best_model.pt --compare_7class path/to/7class_model.pt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add scripts directories to path
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_coarse import (
    CoarseEndoscapesDataset, get_split_paths,
    COARSE_CLASS_NAMES, COARSE_NUM_CLASSES, COARSE_CLASS_COLORS,
    mask_to_color_coarse, LABEL_REMAP
)
from step4_dataset import (
    EndoscapesDataset, CLASS_NAMES as ORIG_CLASS_NAMES,
    NUM_CLASSES as ORIG_NUM_CLASSES, mask_to_color, IMAGENET_MEAN, IMAGENET_STD
)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "coarse_segmentation"


# =============================================================================
# METRICS
# =============================================================================
class EvaluationMetrics:
    """Compute comprehensive segmentation metrics."""

    def __init__(self, num_classes: int, class_names: List[str], ignore_index: int = -1):
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update metrics with a batch of predictions and targets."""
        pred = pred.flatten()
        target = target.flatten()

        # Filter out ignored pixels
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]

        # Update pixel accuracy
        self.correct_pixels += (pred == target).sum()
        self.total_pixels += len(target)

        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, any]:
        """Compute all metrics from accumulated data."""
        metrics = {}

        # Pixel accuracy
        metrics['pixel_accuracy'] = self.correct_pixels / max(self.total_pixels, 1)

        # Per-class IoU
        class_iou = {}
        for c in range(self.num_classes):
            intersection = self.confusion_matrix[c, c]
            union = (self.confusion_matrix[c, :].sum() +
                    self.confusion_matrix[:, c].sum() - intersection)
            if union > 0:
                class_iou[c] = intersection / union
            else:
                class_iou[c] = float('nan')

        metrics['class_iou'] = class_iou

        # Mean IoU (excluding NaN)
        valid_ious = [v for v in class_iou.values() if not np.isnan(v)]
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0

        # Per-class precision and recall
        class_precision = {}
        class_recall = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            class_precision[c] = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            class_recall[c] = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        metrics['class_precision'] = class_precision
        metrics['class_recall'] = class_recall

        # Dice coefficient per class
        class_dice = {}
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            class_dice[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')

        metrics['class_dice'] = class_dice
        valid_dice = [v for v in class_dice.values() if not np.isnan(v)]
        metrics['mean_dice'] = np.mean(valid_dice) if valid_dice else 0.0

        metrics['confusion_matrix'] = self.confusion_matrix

        return metrics


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_coarse_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Load trained coarse segmentation model."""
    from model_variants import create_model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model config
    config = checkpoint.get('config', {})
    decoder = config.get('decoder', 'unet')
    num_classes = checkpoint.get('num_classes', COARSE_NUM_CLASSES)

    print(f"Loading coarse model:")
    print(f"  Decoder: {decoder}")
    print(f"  Classes: {num_classes}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    model = create_model(
        variant=decoder,
        num_classes=num_classes,
        target_size=(512, 512),
        freeze_encoder=True,
        device='cpu',
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def load_7class_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Load trained 7-class segmentation model for comparison."""
    from model_variants import create_model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = checkpoint.get('config', {})
    decoder = config.get('decoder', 'unet')

    print(f"Loading 7-class model for comparison:")
    print(f"  Decoder: {decoder}")
    print(f"  Path: {checkpoint_path}")

    model = create_model(
        variant=decoder,
        num_classes=ORIG_NUM_CLASSES,
        target_size=(512, 512),
        freeze_encoder=True,
        device='cpu',
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    class_names: List[str],
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, any]:
    """Evaluate model on a dataset."""
    model.eval()
    metrics = EvaluationMetrics(num_classes, class_names)

    for batch in tqdm(loader, desc=desc):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()

        output = model(images)
        logits = output['logits']
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(pred.shape[0]):
            metrics.update(pred[i], masks[i])

    return metrics.compute()


def remap_7class_to_4class(pred_7class: np.ndarray) -> np.ndarray:
    """Remap 7-class predictions to 4-class for fair comparison."""
    remapped = np.full_like(pred_7class, fill_value=-1)

    for orig_label, coarse_label in LABEL_REMAP.items():
        if orig_label != 255:  # Skip ignore
            remapped[pred_7class == orig_label] = coarse_label

    return remapped


@torch.no_grad()
def evaluate_7class_on_4class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, any]:
    """
    Evaluate 7-class model remapped to 4 classes.

    This allows fair comparison with the coarse model.
    """
    model.eval()
    metrics = EvaluationMetrics(COARSE_NUM_CLASSES, COARSE_CLASS_NAMES)

    for batch in tqdm(loader, desc="Evaluating 7-class (remapped)"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()  # Already 4-class from coarse loader

        output = model(images)
        logits = output['logits']
        pred_7class = torch.argmax(logits, dim=1).cpu().numpy()

        # Remap predictions to 4 classes
        pred_4class = remap_7class_to_4class(pred_7class)

        for i in range(pred_4class.shape[0]):
            metrics.update(pred_4class[i], masks[i])

    return metrics.compute()


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_comparison_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    coarse_preds: torch.Tensor,
    seven_preds: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    save_path: Optional[Path] = None,
):
    """Create side-by-side comparison visualization."""
    # Denormalize images
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    num_samples = min(num_samples, images.shape[0])

    if seven_preds is not None:
        ncols = 5  # Image, GT, Coarse pred, 7-class remapped, Overlay
    else:
        ncols = 4  # Image, GT, Coarse pred, Overlay

    fig, axes = plt.subplots(num_samples, ncols, figsize=(4 * ncols, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt = masks[i].cpu().numpy()
        coarse_pred = coarse_preds[i].cpu().numpy()

        gt_color = mask_to_color_coarse(gt)
        coarse_pred_color = mask_to_color_coarse(coarse_pred)

        # Overlay
        overlay = img.copy()
        non_bg = coarse_pred > 0
        overlay[non_bg] = cv2.addWeighted(
            img[non_bg], 0.5,
            coarse_pred_color[non_bg], 0.5,
            0
        )

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(coarse_pred_color)
        axes[i, 2].set_title("Coarse (4-class)")
        axes[i, 2].axis("off")

        if seven_preds is not None:
            seven_pred = seven_preds[i].cpu().numpy()
            seven_remapped = remap_7class_to_4class(seven_pred)
            seven_color = mask_to_color_coarse(seven_remapped)

            axes[i, 3].imshow(seven_color)
            axes[i, 3].set_title("7-class (remapped)")
            axes[i, 3].axis("off")

            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title("Overlay")
            axes[i, 4].axis("off")
        else:
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Overlay")
            axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize by row (recall)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")

    plt.close()


def print_metrics_table(metrics: Dict, title: str, class_names: List[str]):
    """Print formatted metrics table."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%")
    print(f"  Mean IoU:       {metrics['mean_iou']*100:.2f}%")
    print(f"  Mean Dice:      {metrics['mean_dice']*100:.2f}%")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'IoU':>10} {'Dice':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 55)

    for c, name in enumerate(class_names):
        iou = metrics['class_iou'].get(c, float('nan'))
        dice = metrics['class_dice'].get(c, float('nan'))
        prec = metrics['class_precision'].get(c, float('nan'))
        rec = metrics['class_recall'].get(c, float('nan'))

        iou_str = f"{iou*100:.1f}%" if not np.isnan(iou) else "N/A"
        dice_str = f"{dice*100:.1f}%" if not np.isnan(dice) else "N/A"
        prec_str = f"{prec*100:.1f}%" if not np.isnan(prec) else "N/A"
        rec_str = f"{rec*100:.1f}%" if not np.isnan(rec) else "N/A"

        print(f"{name:<15} {iou_str:>10} {dice_str:>10} {prec_str:>10} {rec_str:>10}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Coarse Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to coarse model checkpoint')
    parser.add_argument('--compare_7class', type=str, default=None,
                        help='Path to 7-class model for comparison')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_vis', type=int, default=8,
                        help='Number of samples to visualize')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading Data")
    print("-" * 50)

    images_dir, masks_dir = get_split_paths(args.split)
    dataset = CoarseEndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Evaluation samples: {len(dataset)}")

    # =========================================================================
    # LOAD COARSE MODEL
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading Coarse Model")
    print("-" * 50)

    coarse_model = load_coarse_model(args.checkpoint, device)

    # =========================================================================
    # EVALUATE COARSE MODEL
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Evaluating Coarse Model")
    print("-" * 50)

    coarse_metrics = evaluate_model(
        coarse_model, loader, COARSE_NUM_CLASSES, COARSE_CLASS_NAMES,
        device, desc="Evaluating coarse model"
    )

    print_metrics_table(coarse_metrics, "COARSE MODEL (4 classes)", COARSE_CLASS_NAMES)

    # Save confusion matrix
    plot_confusion_matrix(
        coarse_metrics['confusion_matrix'],
        COARSE_CLASS_NAMES,
        save_path=OUTPUT_DIR / f"confusion_matrix_coarse_{args.split}.png",
        title="Coarse Model Confusion Matrix"
    )

    # =========================================================================
    # COMPARE WITH 7-CLASS MODEL (if provided)
    # =========================================================================
    seven_model = None
    seven_metrics = None

    if args.compare_7class:
        print("\n" + "-" * 50)
        print(" Loading 7-Class Model for Comparison")
        print("-" * 50)

        seven_model = load_7class_model(args.compare_7class, device)

        # Evaluate 7-class model on 4-class task (remapped)
        seven_metrics = evaluate_7class_on_4class(seven_model, loader, device)

        print_metrics_table(seven_metrics, "7-CLASS MODEL (remapped to 4)", COARSE_CLASS_NAMES)

        # Comparison
        print("\n" + "=" * 70)
        print(" COMPARISON: Coarse (4-class) vs Fine (7-class remapped)")
        print("=" * 70)
        print(f"\n{'Metric':<20} {'Coarse':>15} {'7-class':>15} {'Diff':>15}")
        print("-" * 65)

        for metric_name in ['pixel_accuracy', 'mean_iou', 'mean_dice']:
            coarse_val = coarse_metrics[metric_name]
            seven_val = seven_metrics[metric_name]
            diff = coarse_val - seven_val

            print(f"{metric_name:<20} {coarse_val*100:>14.2f}% {seven_val*100:>14.2f}% {diff*100:>+14.2f}%")

        print(f"\nPer-class IoU comparison:")
        print(f"{'Class':<15} {'Coarse':>12} {'7-class':>12} {'Diff':>12}")
        print("-" * 51)

        for c, name in enumerate(COARSE_CLASS_NAMES):
            coarse_iou = coarse_metrics['class_iou'].get(c, float('nan'))
            seven_iou = seven_metrics['class_iou'].get(c, float('nan'))

            if not np.isnan(coarse_iou) and not np.isnan(seven_iou):
                diff = coarse_iou - seven_iou
                print(f"{name:<15} {coarse_iou*100:>11.1f}% {seven_iou*100:>11.1f}% {diff*100:>+11.1f}%")
            else:
                print(f"{name:<15} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Creating Visualizations")
    print("-" * 50)

    # Collect samples for visualization
    vis_loader = DataLoader(dataset, batch_size=args.num_vis, shuffle=False, num_workers=0)
    vis_batch = next(iter(vis_loader))

    images = vis_batch['image'].to(device)
    masks = vis_batch['mask']

    with torch.no_grad():
        coarse_output = coarse_model(images)
        coarse_preds = torch.argmax(coarse_output['logits'], dim=1).cpu()

        seven_preds = None
        if seven_model is not None:
            seven_output = seven_model(images)
            seven_preds = torch.argmax(seven_output['logits'], dim=1).cpu()

    create_comparison_visualization(
        images.cpu(), masks, coarse_preds, seven_preds,
        num_samples=min(args.num_vis, 6),
        save_path=OUTPUT_DIR / f"predictions_coarse_{args.split}.png"
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'coarse_metrics': {
            'pixel_accuracy': float(coarse_metrics['pixel_accuracy']),
            'mean_iou': float(coarse_metrics['mean_iou']),
            'mean_dice': float(coarse_metrics['mean_dice']),
            'class_iou': {COARSE_CLASS_NAMES[k]: float(v) if not np.isnan(v) else None
                         for k, v in coarse_metrics['class_iou'].items()},
        },
        'checkpoint': args.checkpoint,
        'split': args.split,
    }

    if seven_metrics is not None:
        results['seven_class_metrics'] = {
            'pixel_accuracy': float(seven_metrics['pixel_accuracy']),
            'mean_iou': float(seven_metrics['mean_iou']),
            'mean_dice': float(seven_metrics['mean_dice']),
            'class_iou': {COARSE_CLASS_NAMES[k]: float(v) if not np.isnan(v) else None
                         for k, v in seven_metrics['class_iou'].items()},
        }
        results['compare_checkpoint'] = args.compare_7class

    import json
    results_path = OUTPUT_DIR / f"evaluation_results_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE")
    print("=" * 70)
    print(f"""
Results Summary:
  Coarse Model (4 classes):
    - Pixel Accuracy: {coarse_metrics['pixel_accuracy']*100:.2f}%
    - Mean IoU:       {coarse_metrics['mean_iou']*100:.2f}%
    - Mean Dice:      {coarse_metrics['mean_dice']*100:.2f}%

  Output directory: {OUTPUT_DIR}
""")

    if seven_metrics is not None:
        print(f"""
  7-Class Model (remapped):
    - Pixel Accuracy: {seven_metrics['pixel_accuracy']*100:.2f}%
    - Mean IoU:       {seven_metrics['mean_iou']*100:.2f}%
    - Mean Dice:      {seven_metrics['mean_dice']*100:.2f}%
""")


if __name__ == '__main__':
    main()
