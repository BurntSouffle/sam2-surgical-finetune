"""
Evaluation Script for Coarse V2 (3-class) with Anatomy Search Region Analysis

Evaluates:
1. Standard 3-class metrics (background, gallbladder, tool)
2. Anatomy search region quality:
   - How well does model uncertainty capture anatomy pixels?
   - Precision: what % of search region is actual anatomy?
   - Recall: what % of GT anatomy falls within search region?

Usage:
    python scripts/coarse_segmentation_v2/evaluate_coarse_v2.py --checkpoint path/to/best_model.pt
    python scripts/coarse_segmentation_v2/evaluate_coarse_v2.py --checkpoint path/to/best_model.pt --threshold 0.7
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add scripts directories to path
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_coarse_v2 import (
    CoarseV2EndoscapesDataset, get_split_paths,
    COARSE_V2_CLASS_NAMES, COARSE_V2_NUM_CLASSES, COARSE_V2_CLASS_COLORS,
    mask_to_color_v2, mask_to_color_v2_with_anatomy, ANATOMY_COLOR
)
from step4_dataset import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "coarse_segmentation_v2"


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# =============================================================================
# METRICS FOR 3 CLASSES
# =============================================================================
class EvaluationMetricsV2:
    """Compute comprehensive segmentation metrics for 3 classes."""

    def __init__(self, num_classes: int = COARSE_V2_NUM_CLASSES, ignore_index: int = -1):
        self.num_classes = num_classes
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
        valid = target >= 0  # -1 is ignored
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

        # Per-class Dice
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
# ANATOMY SEARCH REGION ANALYSIS
# =============================================================================
class AnatomySearchRegionMetrics:
    """
    Evaluate how well model uncertainty captures anatomy regions.

    The idea: pixels where the model is uncertain (no class has high confidence)
    should correspond to anatomy (which was ignored during training).
    """

    def __init__(self, threshold: float = 0.8):
        """
        Args:
            threshold: Confidence threshold. Pixels where max(softmax) < threshold
                       are considered "uncertain" and part of the search region.
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.total_anatomy_pixels = 0
        self.anatomy_in_search = 0  # anatomy pixels captured by search region
        self.total_search_pixels = 0
        self.search_is_anatomy = 0  # search region pixels that are actually anatomy
        self.samples = []

    def update(
        self,
        probs: np.ndarray,       # (H, W, 3) softmax probabilities
        anatomy_mask: np.ndarray,  # (H, W) binary: 1=anatomy
        pred: np.ndarray = None,   # (H, W) argmax predictions (optional)
    ):
        """
        Update metrics for one sample.

        Args:
            probs: Softmax probabilities for 3 classes
            anatomy_mask: Binary mask where 1 = anatomy pixel
            pred: Argmax predictions (optional, for visualization)
        """
        # Compute search region: pixels where no class has high confidence
        max_prob = probs.max(axis=-1)  # (H, W)
        search_region = max_prob < self.threshold  # Uncertain pixels

        # Alternative: use entropy
        # entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        # search_region = entropy > entropy_threshold

        # Count pixels
        anatomy_pixels = anatomy_mask.sum()
        search_pixels = search_region.sum()

        if anatomy_pixels > 0:
            # Recall: how many anatomy pixels are in search region?
            anatomy_in_search = (search_region & (anatomy_mask == 1)).sum()
            recall = anatomy_in_search / anatomy_pixels
        else:
            anatomy_in_search = 0
            recall = float('nan')

        if search_pixels > 0:
            # Precision: how many search region pixels are anatomy?
            search_is_anatomy = (search_region & (anatomy_mask == 1)).sum()
            precision = search_is_anatomy / search_pixels
        else:
            search_is_anatomy = 0
            precision = float('nan')

        # Accumulate
        self.total_anatomy_pixels += anatomy_pixels
        self.anatomy_in_search += anatomy_in_search
        self.total_search_pixels += search_pixels
        self.search_is_anatomy += search_is_anatomy

        # Store per-sample metrics
        self.samples.append({
            'anatomy_pixels': anatomy_pixels,
            'search_pixels': search_pixels,
            'recall': recall,
            'precision': precision,
        })

    def compute(self) -> Dict[str, float]:
        """Compute aggregate search region metrics."""
        metrics = {}

        # Overall recall: % of all anatomy captured
        if self.total_anatomy_pixels > 0:
            metrics['recall'] = self.anatomy_in_search / self.total_anatomy_pixels
        else:
            metrics['recall'] = float('nan')

        # Overall precision: % of search region that is anatomy
        if self.total_search_pixels > 0:
            metrics['precision'] = self.search_is_anatomy / self.total_search_pixels
        else:
            metrics['precision'] = float('nan')

        # F1 score
        if metrics['recall'] > 0 and metrics['precision'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        # Per-sample statistics
        recalls = [s['recall'] for s in self.samples if not np.isnan(s['recall'])]
        precisions = [s['precision'] for s in self.samples if not np.isnan(s['precision'])]

        metrics['mean_recall'] = np.mean(recalls) if recalls else 0.0
        metrics['mean_precision'] = np.mean(precisions) if precisions else 0.0
        metrics['std_recall'] = np.std(recalls) if recalls else 0.0
        metrics['std_precision'] = np.std(precisions) if precisions else 0.0

        metrics['total_anatomy_pixels'] = self.total_anatomy_pixels
        metrics['total_search_pixels'] = self.total_search_pixels
        metrics['threshold'] = self.threshold

        return metrics


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_v2_model(checkpoint_path: str, device: str = 'cpu') -> nn.Module:
    """Load trained V2 (3-class) segmentation model."""
    from model_variants import create_model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model config
    config = checkpoint.get('config', {})
    decoder = config.get('decoder', 'unet')
    num_classes = checkpoint.get('num_classes', COARSE_V2_NUM_CLASSES)

    print(f"Loading V2 model:")
    print(f"  Decoder: {decoder}")
    print(f"  Classes: {num_classes}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Version: {checkpoint.get('version', 'unknown')}")

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


# =============================================================================
# EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate_v2_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.8,
) -> Tuple[Dict, Dict]:
    """
    Evaluate V2 model on 3-class task and compute search region metrics.

    Returns:
        Tuple of (segmentation_metrics, search_region_metrics)
    """
    model.eval()
    seg_metrics = EvaluationMetricsV2()
    search_metrics = AnatomySearchRegionMetrics(threshold=threshold)

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        anatomy_masks = batch['anatomy_mask'].numpy()

        output = model(images)
        logits = output['logits']

        # Softmax probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, 3, H, W)
        pred = np.argmax(probs, axis=1)  # (B, H, W)

        # Update metrics for each sample
        for i in range(pred.shape[0]):
            # Segmentation metrics
            seg_metrics.update(pred[i], masks[i])

            # Search region metrics
            probs_i = probs[i].transpose(1, 2, 0)  # (H, W, 3)
            search_metrics.update(probs_i, anatomy_masks[i], pred[i])

    return seg_metrics.compute(), search_metrics.compute()


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_search_region_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    anatomy_masks: torch.Tensor,
    probs: torch.Tensor,
    threshold: float = 0.8,
    num_samples: int = 4,
    save_path: Optional[Path] = None,
):
    """
    Create visualization showing search region and anatomy capture.

    Shows: Image | GT (3-class + anatomy) | Prediction | Search Region | Overlay
    """
    # Denormalize images
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    num_samples = min(num_samples, images.shape[0])

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mask = masks[i].cpu().numpy()
        anatomy_mask = anatomy_masks[i].cpu().numpy()
        prob = probs[i].cpu().numpy()  # (3, H, W)

        pred = np.argmax(prob, axis=0)
        max_prob = prob.max(axis=0)
        search_region = max_prob < threshold

        # GT with anatomy highlighted
        gt_color = mask_to_color_v2_with_anatomy(mask, anatomy_mask)

        # Prediction
        pred_color = mask_to_color_v2(pred)

        # Search region visualization
        search_vis = np.zeros((*search_region.shape, 3), dtype=np.uint8)
        search_vis[search_region] = [255, 255, 0]  # Yellow for search region
        search_vis[anatomy_mask == 1] = [255, 0, 0]  # Red for actual anatomy
        # Overlap (search region AND anatomy) = green
        overlap = search_region & (anatomy_mask == 1)
        search_vis[overlap] = [0, 255, 0]

        # Overlay: search region on image
        overlay = img.copy()
        overlay[search_region] = (0.5 * overlay[search_region] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)

        # Calculate metrics for this sample
        if anatomy_mask.sum() > 0:
            recall = overlap.sum() / anatomy_mask.sum() * 100
        else:
            recall = 0
        if search_region.sum() > 0:
            precision = overlap.sum() / search_region.sum() * 100
        else:
            precision = 0

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title("GT (anatomy=orange)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title("Prediction (3-class)")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(search_vis)
        axes[i, 3].set_title(f"Search (R={recall:.0f}% P={precision:.0f}%)")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title("Search Region Overlay")
        axes[i, 4].axis("off")

    # Legend
    fig.text(0.5, 0.02,
             "Search Region: Yellow=uncertain, Red=anatomy GT, Green=captured anatomy (overlap)",
             ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    plt.close()


def print_metrics_table_v2(metrics: Dict, title: str):
    """Print formatted metrics table for 3 classes."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%")
    print(f"  Mean IoU:       {metrics['mean_iou']*100:.2f}%")
    print(f"  Mean Dice:      {metrics['mean_dice']*100:.2f}%")

    print(f"\nPer-Class IoU:")
    print(f"{'Class':<15} {'IoU':>10} {'Dice':>10}")
    print("-" * 35)

    for c, name in enumerate(COARSE_V2_CLASS_NAMES):
        iou = metrics['class_iou'].get(c, float('nan'))
        dice = metrics['class_dice'].get(c, float('nan'))

        iou_str = f"{iou*100:.1f}%" if not np.isnan(iou) else "N/A"
        dice_str = f"{dice*100:.1f}%" if not np.isnan(dice) else "N/A"

        print(f"{name:<15} {iou_str:>10} {dice_str:>10}")


def print_search_region_metrics(metrics: Dict):
    """Print search region analysis results."""
    print("\n" + "=" * 70)
    print(" ANATOMY SEARCH REGION ANALYSIS")
    print("=" * 70)
    print(f"\nConfidence threshold: {metrics['threshold']}")
    print(f"Search region = pixels where max(softmax) < {metrics['threshold']}")

    print(f"\nSearch Region Quality:")
    print(f"  Recall:    {metrics['recall']*100:.1f}%  (% of anatomy captured by search region)")
    print(f"  Precision: {metrics['precision']*100:.1f}%  (% of search region that is anatomy)")
    print(f"  F1 Score:  {metrics['f1']*100:.1f}%")

    print(f"\nPer-Sample Statistics:")
    print(f"  Mean Recall:    {metrics['mean_recall']*100:.1f}% (+/- {metrics['std_recall']*100:.1f}%)")
    print(f"  Mean Precision: {metrics['mean_precision']*100:.1f}% (+/- {metrics['std_precision']*100:.1f}%)")

    print(f"\nPixel Counts:")
    print(f"  Total anatomy pixels: {metrics['total_anatomy_pixels']:,}")
    print(f"  Total search pixels:  {metrics['total_search_pixels']:,}")


# =============================================================================
# THRESHOLD SWEEP
# =============================================================================
def sweep_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
) -> Dict[float, Dict]:
    """Sweep different confidence thresholds to find optimal search region."""
    results = {}

    print("\n" + "=" * 70)
    print(" THRESHOLD SWEEP FOR SEARCH REGION")
    print("=" * 70)

    for threshold in thresholds:
        print(f"\nEvaluating threshold={threshold}...")
        _, search_metrics = evaluate_v2_model(model, loader, device, threshold)
        results[threshold] = search_metrics

        print(f"  Recall: {search_metrics['recall']*100:.1f}%, "
              f"Precision: {search_metrics['precision']*100:.1f}%, "
              f"F1: {search_metrics['f1']*100:.1f}%")

    # Summary table
    print("\n" + "-" * 60)
    print(f"{'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.8

    for threshold, metrics in results.items():
        recall = metrics['recall'] * 100 if not np.isnan(metrics['recall']) else 0
        precision = metrics['precision'] * 100 if not np.isnan(metrics['precision']) else 0
        f1 = metrics['f1'] * 100

        print(f"{threshold:>10.2f} {recall:>9.1f}% {precision:>9.1f}% {f1:>9.1f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-" * 60)
    print(f"Best threshold: {best_threshold} (F1={best_f1:.1f}%)")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Coarse V2 Model with Search Region Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to V2 model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Confidence threshold for search region')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_vis', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--sweep_thresholds', action='store_true',
                        help='Sweep multiple thresholds to find optimal')

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
    dataset = CoarseV2EndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        normalize=True,
        return_anatomy_mask=True,  # Need anatomy mask for evaluation
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
    # LOAD MODEL
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading V2 Model")
    print("-" * 50)

    model = load_v2_model(args.checkpoint, device)

    # =========================================================================
    # THRESHOLD SWEEP (optional)
    # =========================================================================
    if args.sweep_thresholds:
        sweep_results = sweep_thresholds(model, loader, device)

        # Save sweep results
        sweep_path = OUTPUT_DIR / f"threshold_sweep_{args.split}.json"
        sweep_save = {str(k): convert_to_json_serializable(v) for k, v in sweep_results.items()}
        with open(sweep_path, 'w') as f:
            json.dump(sweep_save, f, indent=2)
        print(f"\nSweep results saved to: {sweep_path}")

    # =========================================================================
    # EVALUATE WITH SELECTED THRESHOLD
    # =========================================================================
    print("\n" + "-" * 50)
    print(f" Evaluating (threshold={args.threshold})")
    print("-" * 50)

    seg_metrics, search_metrics = evaluate_v2_model(model, loader, device, args.threshold)

    print_metrics_table_v2(seg_metrics, "3-CLASS SEGMENTATION METRICS")
    print_search_region_metrics(search_metrics)

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
    anatomy_masks = vis_batch['anatomy_mask']

    with torch.no_grad():
        output = model(images)
        logits = output['logits']
        probs = F.softmax(logits, dim=1).cpu()

    create_search_region_visualization(
        images.cpu(), masks, anatomy_masks, probs,
        threshold=args.threshold,
        num_samples=min(args.num_vis, 6),
        save_path=OUTPUT_DIR / f"search_region_vis_{args.split}.png"
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'segmentation_metrics': {
            'pixel_accuracy': float(seg_metrics['pixel_accuracy']),
            'mean_iou': float(seg_metrics['mean_iou']),
            'mean_dice': float(seg_metrics['mean_dice']),
            'class_iou': {COARSE_V2_CLASS_NAMES[k]: float(v) if not np.isnan(v) else None
                         for k, v in seg_metrics['class_iou'].items()},
        },
        'search_region_metrics': {
            'threshold': args.threshold,
            'recall': float(search_metrics['recall']) if not np.isnan(search_metrics['recall']) else None,
            'precision': float(search_metrics['precision']) if not np.isnan(search_metrics['precision']) else None,
            'f1': float(search_metrics['f1']),
        },
        'checkpoint': args.checkpoint,
        'split': args.split,
    }

    results_path = OUTPUT_DIR / f"evaluation_results_v2_{args.split}.json"
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
V2 Model (3 classes - anatomy ignored):
  Pixel Accuracy: {seg_metrics['pixel_accuracy']*100:.2f}%
  Mean IoU:       {seg_metrics['mean_iou']*100:.2f}%

Per-Class IoU:
  background:  {seg_metrics['class_iou'][0]*100:.1f}%
  gallbladder: {seg_metrics['class_iou'][1]*100:.1f}%
  tool:        {seg_metrics['class_iou'][2]*100:.1f}%

Anatomy Search Region (threshold={args.threshold}):
  Recall:    {search_metrics['recall']*100:.1f}%  <- % of anatomy captured
  Precision: {search_metrics['precision']*100:.1f}%  <- % of search that is anatomy
  F1:        {search_metrics['f1']*100:.1f}%

Output: {OUTPUT_DIR}
""")


if __name__ == '__main__':
    main()
