"""
Step 8: Comprehensive Test Evaluation Script
Evaluates trained models with detailed metrics and visualizations.

Usage:
    python scripts/step8_evaluate.py --checkpoint path/to/best_model.pt
    python scripts/step8_evaluate.py --checkpoint path/to/best_model.pt --split test --save_predictions
    python scripts/step8_evaluate.py --compare checkpoint1.pt checkpoint2.pt checkpoint3.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from step4_dataset import (
    EndoscapesDataset, get_split_paths,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES,
    mask_to_color, create_overlay, IMAGENET_MEAN, IMAGENET_STD
)
from step5_model_setup import (
    SAM2SemanticSegmentation, CHECKPOINT_PATH, CONFIG_FILE
)

# Output directory
OUTPUT_DIR = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs")


# =============================================================================
# METRICS COMPUTATION
# =============================================================================
class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    Accumulates predictions over the dataset and computes final metrics.
    """

    def __init__(self, num_classes: int = 7, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        # Confusion matrix: [pred, target]
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.valid_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with batch predictions.

        Args:
            pred: Predicted class indices (B, H, W) or (H, W)
            target: Ground truth class indices, ignore_index for ignore
        """
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # Filter valid pixels
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]

        self.total_pixels += len(valid_mask)
        self.valid_pixels += valid_mask.sum()

        # Update confusion matrix
        for p, t in zip(pred, target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[p, t] += 1

    def compute_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Compute per-class IoU, Dice, Precision, Recall."""
        metrics = {}

        for c in range(self.num_classes):
            # True positives: diagonal
            tp = self.confusion_matrix[c, c]

            # False positives: predicted as c but not c
            fp = self.confusion_matrix[c, :].sum() - tp

            # False negatives: actually c but predicted as something else
            fn = self.confusion_matrix[:, c].sum() - tp

            # True negatives (for completeness)
            tn = self.confusion_matrix.sum() - tp - fp - fn

            # Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            # Pixel count (ground truth)
            pixel_count = self.confusion_matrix[:, c].sum()
            pixel_pct = pixel_count / self.valid_pixels if self.valid_pixels > 0 else 0.0

            metrics[c] = {
                'iou': iou,
                'dice': dice,
                'precision': precision,
                'recall': recall,
                'pixel_count': int(pixel_count),
                'pixel_pct': pixel_pct,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
            }

        return metrics

    def compute_aggregate_metrics(self, per_class: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Compute aggregate metrics."""
        # Collect valid class metrics (classes that exist in the data)
        ious = []
        dices = []
        precisions = []
        recalls = []
        weights = []

        for c in range(self.num_classes):
            if per_class[c]['pixel_count'] > 0:
                ious.append(per_class[c]['iou'])
                dices.append(per_class[c]['dice'])
                precisions.append(per_class[c]['precision'])
                recalls.append(per_class[c]['recall'])
                weights.append(per_class[c]['pixel_pct'])

        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        # Mean metrics (macro average)
        mean_iou = np.mean(ious) if ious else 0.0
        mean_dice = np.mean(dices) if dices else 0.0
        mean_precision = np.mean(precisions) if precisions else 0.0
        mean_recall = np.mean(recalls) if recalls else 0.0

        # Weighted IoU
        weighted_iou = sum(w * iou for w, iou in zip(weights, ious)) if ious else 0.0

        # Overall pixel accuracy
        correct = np.diag(self.confusion_matrix).sum()
        pixel_accuracy = correct / self.valid_pixels if self.valid_pixels > 0 else 0.0

        # Mean pixel accuracy (per-class accuracy averaged)
        class_accuracies = []
        for c in range(self.num_classes):
            total_c = self.confusion_matrix[:, c].sum()
            if total_c > 0:
                class_accuracies.append(self.confusion_matrix[c, c] / total_c)
        mean_pixel_accuracy = np.mean(class_accuracies) if class_accuracies else 0.0

        return {
            'mean_iou': mean_iou,
            'weighted_iou': weighted_iou,
            'mean_dice': mean_dice,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'pixel_accuracy': pixel_accuracy,
            'mean_pixel_accuracy': mean_pixel_accuracy,
            'valid_pixels': int(self.valid_pixels),
            'total_pixels': int(self.total_pixels),
        }

    def get_confusion_matrix(self, normalize: str = 'recall') -> np.ndarray:
        """
        Get confusion matrix, optionally normalized.

        Args:
            normalize: 'recall' (by row), 'precision' (by col), or None
        """
        cm = self.confusion_matrix.copy().astype(float)

        if normalize == 'recall':
            # Normalize by row (recall view)
            row_sums = cm.sum(axis=0, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums
        elif normalize == 'precision':
            # Normalize by column (precision view)
            col_sums = cm.sum(axis=1, keepdims=True)
            col_sums[col_sums == 0] = 1
            cm = cm / col_sums

        return cm


def bootstrap_confidence_interval(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_samples: int = 100,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mIoU.

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(predictions)
    mious = []

    for _ in range(num_samples):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)

        # Compute mIoU for this sample
        metrics = SegmentationMetrics()
        for idx in indices:
            metrics.update(predictions[idx], targets[idx])

        per_class = metrics.compute_per_class_metrics()
        aggregate = metrics.compute_aggregate_metrics(per_class)
        mious.append(aggregate['mean_iou'])

    mious = np.array(mious)
    mean = np.mean(mious)

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(mious, 100 * alpha / 2)
    upper = np.percentile(mious, 100 * (1 - alpha / 2))

    return mean, lower, upper


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix (Recall Normalized)"
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use shorter class name abbreviations for display
    abbrevs = ['bg', 'cp', 'ct', 'ca', 'cd', 'gb', 'tool']

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=abbrevs,
        yticklabels=abbrevs,
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add full class names as legend
    legend_text = '\n'.join([f'{abbrev}: {name}' for abbrev, name in zip(abbrevs, class_names)])
    ax.text(1.02, 0.5, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_class_iou_bar_chart(
    per_class_metrics: Dict[int, Dict[str, float]],
    class_names: List[str],
    save_path: Path,
):
    """Plot bar chart of per-class IoU."""
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(range(len(class_names)))
    ious = [per_class_metrics[c]['iou'] * 100 for c in classes]
    colors = [np.array(CLASS_COLORS[c]) / 255 for c in classes]

    bars = ax.bar(classes, ious, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xticks(classes)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU (%)', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('Per-Class IoU', fontsize=14)
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        ax.annotate(f'{iou:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Add mean IoU line
    mean_iou = np.mean(ious)
    ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean IoU: {mean_iou:.1f}%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_prediction_visualization(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Path,
):
    """Save visualization with image, GT, prediction, and overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Ground truth
    gt_color = mask_to_color(gt_mask)
    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Prediction
    pred_color = mask_to_color(pred_mask)
    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Overlay
    overlay = create_overlay(image, pred_mask, alpha=0.5)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================
def format_metrics_table(
    per_class: Dict[int, Dict[str, float]],
    aggregate: Dict[str, float],
    class_names: List[str],
) -> str:
    """Format metrics as a nice ASCII table."""
    lines = []

    # Header
    lines.append("Per-Class Results:")
    lines.append("+" + "-"*17 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*10 + "+")
    lines.append("| {:15s} | {:>5s} | {:>5s} | {:>5s} | {:>5s} | {:>8s} |".format(
        "Class", "IoU", "Dice", "Prec", "Recall", "Pixels"
    ))
    lines.append("+" + "-"*17 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*10 + "+")

    # Per-class rows
    for c in range(len(class_names)):
        m = per_class[c]
        lines.append("| {:15s} | {:5.1f}% | {:5.1f}% | {:5.1f}% | {:5.1f}% | {:7.1f}% |".format(
            class_names[c],
            m['iou'] * 100,
            m['dice'] * 100,
            m['precision'] * 100,
            m['recall'] * 100,
            m['pixel_pct'] * 100,
        ))

    lines.append("+" + "-"*17 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+" + "-"*10 + "+")

    # Aggregate metrics
    lines.append("")
    lines.append("Aggregate Metrics:")
    lines.append("")
    lines.append(f"  Mean IoU (mIoU):     {aggregate['mean_iou']*100:5.1f}%")
    lines.append(f"  Weighted IoU:        {aggregate['weighted_iou']*100:5.1f}%")
    lines.append(f"  Mean Dice:           {aggregate['mean_dice']*100:5.1f}%")
    lines.append(f"  Pixel Accuracy:      {aggregate['pixel_accuracy']*100:5.1f}%")
    lines.append(f"  Mean Pixel Accuracy: {aggregate['mean_pixel_accuracy']*100:5.1f}%")

    return "\n".join(lines)


def format_comparison_table(
    results: Dict[str, Dict],
    class_names: List[str],
) -> str:
    """Format comparison table for multiple checkpoints."""
    lines = []

    checkpoints = list(results.keys())
    n_ckpt = len(checkpoints)

    # Shorten checkpoint names
    short_names = [Path(c).stem[:20] for c in checkpoints]

    # Header
    header_width = 15 + 10 * n_ckpt
    lines.append("=" * header_width)
    lines.append("CHECKPOINT COMPARISON")
    lines.append("=" * header_width)

    # Aggregate metrics comparison
    lines.append("")
    lines.append("Aggregate Metrics:")
    lines.append("-" * header_width)

    # Header row
    header = f"{'Metric':<20}" + "".join([f"{name:>12}" for name in short_names])
    lines.append(header)
    lines.append("-" * header_width)

    metrics_to_compare = ['mean_iou', 'weighted_iou', 'mean_dice', 'pixel_accuracy']
    metric_names = ['Mean IoU', 'Weighted IoU', 'Mean Dice', 'Pixel Accuracy']

    for metric, name in zip(metrics_to_compare, metric_names):
        row = f"{name:<20}"
        for ckpt in checkpoints:
            val = results[ckpt]['aggregate'][metric] * 100
            row += f"{val:>11.1f}%"
        lines.append(row)

    # Per-class IoU comparison
    lines.append("")
    lines.append("Per-Class IoU:")
    lines.append("-" * header_width)

    header = f"{'Class':<20}" + "".join([f"{name:>12}" for name in short_names])
    lines.append(header)
    lines.append("-" * header_width)

    for c, class_name in enumerate(class_names):
        row = f"{class_name:<20}"
        for ckpt in checkpoints:
            val = results[ckpt]['per_class'][c]['iou'] * 100
            row += f"{val:>11.1f}%"
        lines.append(row)

    lines.append("=" * header_width)

    return "\n".join(lines)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # weights_only=False needed for loading our training checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config
    config = checkpoint.get('config', {})
    target_size = config.get('target_size', (512, 512))
    freeze_encoder = config.get('freeze_encoder', True)
    use_multiscale_head = config.get('use_multiscale_head', True)
    hidden_dim = config.get('hidden_dim', 128)

    # Load SAM2
    from sam2.build_sam import build_sam2

    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device='cpu',
        mode='eval',
    )

    # Create model
    model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=target_size,
        freeze_encoder=freeze_encoder,
        use_multiscale_head=use_multiscale_head,
        hidden_dim=hidden_dim,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")

    return model, config


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: EndoscapesDataset,
    device: torch.device,
    save_predictions: bool = False,
    save_visualizations: bool = False,
    output_dir: Optional[Path] = None,
    compute_ci: bool = True,
) -> Dict:
    """
    Evaluate model on dataset.

    Returns:
        Dict with per_class metrics, aggregate metrics, and confusion matrix
    """
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for saving
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    metrics_tracker = SegmentationMetrics()

    # For bootstrap CI
    all_preds = []
    all_targets = []

    # Create output directories if needed
    if save_predictions and output_dir:
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

    if save_visualizations and output_dir:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating on {len(dataset)} samples...")

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        filename = batch['filename'][0]

        # Forward pass
        output = model(images)
        logits = output['logits']
        pred = torch.argmax(logits, dim=1)

        # Update metrics
        metrics_tracker.update(pred, masks)

        # Store for bootstrap
        if compute_ci:
            all_preds.append(pred.cpu())
            all_targets.append(masks.cpu())

        # Save predictions
        if save_predictions and output_dir:
            pred_mask = pred[0].cpu().numpy().astype(np.uint8)
            pred_path = pred_dir / filename.replace('.jpg', '.png')
            cv2.imwrite(str(pred_path), pred_mask)

        # Save visualizations
        if save_visualizations and output_dir:
            # Get raw image
            raw_image, raw_mask, _ = dataset.get_raw_sample(i)
            pred_mask = pred[0].cpu().numpy()

            vis_path = vis_dir / filename.replace('.jpg', '_vis.png')
            save_prediction_visualization(raw_image, raw_mask, pred_mask, vis_path)

    # Compute final metrics
    per_class = metrics_tracker.compute_per_class_metrics()
    aggregate = metrics_tracker.compute_aggregate_metrics(per_class)
    confusion_matrix = metrics_tracker.get_confusion_matrix(normalize='recall')

    # Bootstrap confidence interval
    ci = None
    if compute_ci and len(all_preds) > 10:
        print("\nComputing bootstrap confidence interval...")
        mean, lower, upper = bootstrap_confidence_interval(all_preds, all_targets, num_samples=100)
        ci = {
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'margin': (upper - lower) / 2,
        }
        print(f"mIoU = {mean*100:.1f}% +/- {ci['margin']*100:.1f}%")

    return {
        'per_class': per_class,
        'aggregate': aggregate,
        'confusion_matrix': confusion_matrix.tolist(),
        'confidence_interval': ci,
    }


def run_evaluation(
    checkpoint_path: Path,
    split: str = 'test',
    save_predictions: bool = False,
    save_visualizations: bool = False,
):
    """Run full evaluation pipeline."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Load dataset
    target_size = config.get('target_size', (512, 512))
    images_dir, masks_dir = get_split_paths(split)

    dataset = EndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=target_size,
        normalize=True,
    )

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = checkpoint_path.stem
    output_dir = OUTPUT_DIR / f'eval_{checkpoint_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Evaluate
    results = evaluate_model(
        model, dataset, device,
        save_predictions=save_predictions,
        save_visualizations=save_visualizations,
        output_dir=output_dir,
    )

    # Print results
    print("\n" + "=" * 70)
    print(f" {split.upper()} SET EVALUATION - {checkpoint_name}")
    print("=" * 70)

    table = format_metrics_table(results['per_class'], results['aggregate'], CLASS_NAMES)
    print(table)

    if results['confidence_interval']:
        ci = results['confidence_interval']
        print(f"\n  95% CI for mIoU: {ci['mean']*100:.1f}% +/- {ci['margin']*100:.1f}%")
        print(f"                   [{ci['lower']*100:.1f}%, {ci['upper']*100:.1f}%]")

    print("\n" + "=" * 70)

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save table
    table_path = output_dir / 'metrics_table.txt'
    with open(table_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f" {split.upper()} SET EVALUATION - {checkpoint_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(table)
        if results['confidence_interval']:
            ci = results['confidence_interval']
            f.write(f"\n\n  95% CI for mIoU: {ci['mean']*100:.1f}% +/- {ci['margin']*100:.1f}%")
        f.write("\n\n" + "=" * 70 + "\n")
    print(f"Table saved to: {table_path}")

    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        CLASS_NAMES,
        cm_path,
    )
    print(f"Confusion matrix saved to: {cm_path}")

    # Plot IoU bar chart
    bar_path = output_dir / 'class_iou_bar_chart.png'
    plot_class_iou_bar_chart(results['per_class'], CLASS_NAMES, bar_path)
    print(f"IoU bar chart saved to: {bar_path}")

    return results


def run_comparison(checkpoint_paths: List[Path], split: str = 'test'):
    """Compare multiple checkpoints."""
    print("\n" + "=" * 70)
    print(" CHECKPOINT COMPARISON")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    for ckpt_path in checkpoint_paths:
        print(f"\n--- Evaluating: {ckpt_path.name} ---")

        # Load model
        model, config = load_model(ckpt_path, device)

        # Load dataset
        target_size = config.get('target_size', (512, 512))
        images_dir, masks_dir = get_split_paths(split)

        dataset = EndoscapesDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            target_size=target_size,
            normalize=True,
        )

        # Evaluate (no saving, no CI for speed)
        result = evaluate_model(
            model, dataset, device,
            save_predictions=False,
            save_visualizations=False,
            compute_ci=False,
        )

        results[str(ckpt_path)] = result

        # Clear model from GPU
        del model
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + format_comparison_table(results, CLASS_NAMES))

    # Save comparison CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_DIR / f'comparison_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'comparison.csv'
    with open(csv_path, 'w') as f:
        # Header
        checkpoints = list(results.keys())
        f.write("metric," + ",".join([Path(c).stem for c in checkpoints]) + "\n")

        # Aggregate metrics
        for metric in ['mean_iou', 'weighted_iou', 'mean_dice', 'pixel_accuracy']:
            f.write(metric)
            for ckpt in checkpoints:
                f.write(f",{results[ckpt]['aggregate'][metric]:.4f}")
            f.write("\n")

        # Per-class IoU
        for c, class_name in enumerate(CLASS_NAMES):
            f.write(f"iou_{class_name}")
            for ckpt in checkpoints:
                f.write(f",{results[ckpt]['per_class'][c]['iou']:.4f}")
            f.write("\n")

    print(f"\nComparison saved to: {csv_path}")

    # Save comparison table
    table_path = output_dir / 'comparison_table.txt'
    with open(table_path, 'w') as f:
        f.write(format_comparison_table(results, CLASS_NAMES))
    print(f"Table saved to: {table_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation models')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--compare', nargs='+', type=str, default=None,
                        help='Compare multiple checkpoints')

    args = parser.parse_args()

    if args.compare:
        # Comparison mode
        checkpoint_paths = [Path(p) for p in args.compare]
        for p in checkpoint_paths:
            if not p.exists():
                print(f"Error: Checkpoint not found: {p}")
                sys.exit(1)
        run_comparison(checkpoint_paths, args.split)

    elif args.checkpoint:
        # Single evaluation mode
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        run_evaluation(
            checkpoint_path,
            split=args.split,
            save_predictions=args.save_predictions,
            save_visualizations=args.save_visualizations,
        )

    else:
        print("Error: Must specify --checkpoint or --compare")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
