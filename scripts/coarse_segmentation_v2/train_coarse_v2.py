"""
Training Script for Coarse V2 (3-class) Semantic Segmentation

Only trains on 3 classes: background, gallbladder, tool
Anatomy pixels are IGNORED in the loss function.

The model learns clean boundaries for the 3 main classes.
At inference, uncertain regions indicate where anatomy might be.

Usage:
    python scripts/coarse_segmentation_v2/train_coarse_v2.py
    python scripts/coarse_segmentation_v2/train_coarse_v2.py --epochs 100 --batch_size 8
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    mask_to_color_v2, COARSE_V2_IGNORE_INDEX, ANATOMY_COLOR
)
from step4_dataset import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.parent.parent

DEFAULT_CONFIG = {
    # Data
    'target_size': (512, 512),
    'batch_size': 16,
    'num_workers': 0,  # Windows compatibility

    # Training
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,

    # Model
    'decoder': 'unet',
    'freeze_encoder': True,

    # Loss - CE with ignore_index for anatomy
    'loss_type': 'ce',
    'use_class_weights': True,
    'ignore_index': -1,  # Anatomy pixels ignored

    # Augmentation
    'use_augmentation': True,
    'flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter': 0.2,

    # Checkpointing
    'save_every': 10,
    'save_best': True,

    # Logging
    'log_images_every': 5,
    'num_log_images': 4,

    # Paths
    'output_dir': BASE_DIR / 'checkpoints' / 'coarse_segmentation_v2',
}


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
class SegmentationAugmentation:
    """Data augmentation for semantic segmentation."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_degrees: float = 15,
        color_jitter: float = 0.2,
    ):
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 0.5:
                image, mask = self._rotate(image, mask, angle)

        # Color jitter (only on image)
        if self.color_jitter > 0:
            image = self._color_jitter(image)

        return image, mask

    def _rotate(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        angle: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy().astype(np.int32)

        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_rot = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        mask_rot = cv2.warpAffine(mask_np, M, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=-1)

        image = torch.from_numpy(img_rot).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask_rot).long()

        return image, mask

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = image * std + mean

        brightness_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        image = image * brightness_factor

        contrast_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        gray = image.mean(dim=0, keepdim=True)
        image = contrast_factor * image + (1 - contrast_factor) * gray

        image = torch.clamp(image, 0, 1)
        image = (image - mean) / std

        return image


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that applies augmentation."""

    def __init__(self, base_dataset: CoarseV2EndoscapesDataset, augmentation=None):
        self.base_dataset = base_dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        if self.augmentation is not None:
            image, mask = self.augmentation(sample['image'], sample['mask'])
            sample['image'] = image
            sample['mask'] = mask

        # Safety: ensure mask values are valid for 3 classes
        mask = sample['mask']
        invalid_mask = (mask < -1) | (mask > 2)
        if invalid_mask.any():
            mask[invalid_mask] = -1
            sample['mask'] = mask

        return sample


# =============================================================================
# CLASS WEIGHTS
# =============================================================================
def compute_class_weights_v2(dataset: CoarseV2EndoscapesDataset) -> torch.Tensor:
    """Compute class weights based on inverse frequency (3 classes only)."""
    print("\nComputing class weights from training data...")
    print("Note: Anatomy pixels are excluded from weight computation")

    class_pixels = torch.zeros(COARSE_V2_NUM_CLASSES)
    total_valid_pixels = 0

    for i in tqdm(range(len(dataset)), desc="Scanning masks"):
        sample = dataset[i]
        mask = sample['mask']

        for c in range(COARSE_V2_NUM_CLASSES):
            class_pixels[c] += (mask == c).sum()

        # Only count valid (non-ignored) pixels
        total_valid_pixels += (mask >= 0).sum()

    weights = total_valid_pixels / (COARSE_V2_NUM_CLASSES * class_pixels + 1e-6)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=10.0)

    print("\nClass weights (3 classes):")
    for c in range(COARSE_V2_NUM_CLASSES):
        pct = class_pixels[c] / total_valid_pixels * 100
        print(f"  {c}: {COARSE_V2_CLASS_NAMES[c]:15s} - {pct:6.2f}% valid pixels, weight={weights[c]:.3f}")

    return weights


# =============================================================================
# METRICS
# =============================================================================
class MetricTrackerV2:
    """Track and aggregate metrics over batches (3 classes)."""

    def __init__(self, num_classes: int = COARSE_V2_NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_pixels = 0
        self.class_intersection = torch.zeros(self.num_classes)
        self.class_union = torch.zeros(self.num_classes)
        self.num_batches = 0

    def update(self, loss: float, pred: torch.Tensor, target: torch.Tensor):
        self.total_loss += loss
        self.num_batches += 1

        pred = pred.view(-1)
        target = target.view(-1)

        # Only evaluate on valid (non-ignored) pixels
        valid = target >= 0  # -1 is ignored

        if valid.sum() == 0:
            return

        pred_valid = pred[valid]
        target_valid = target[valid]

        self.total_correct += (pred_valid == target_valid).sum().item()
        self.total_pixels += valid.sum().item()

        for c in range(self.num_classes):
            pred_c = pred_valid == c
            target_c = target_valid == c

            self.class_intersection[c] += (pred_c & target_c).sum().item()
            self.class_union[c] += (pred_c | target_c).sum().item()

    def compute(self) -> Dict[str, float]:
        metrics = {}
        metrics['loss'] = self.total_loss / max(self.num_batches, 1)
        metrics['pixel_accuracy'] = self.total_correct / max(self.total_pixels, 1)

        class_iou = {}
        for c in range(self.num_classes):
            if self.class_union[c] > 0:
                class_iou[c] = self.class_intersection[c] / self.class_union[c]
            else:
                class_iou[c] = float('nan')

        metrics['class_iou'] = class_iou

        valid_ious = [v for v in class_iou.values() if not np.isnan(v)]
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0

        return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization_grid_v2(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    num_images: int = 4,
) -> np.ndarray:
    """Create visualization grid for TensorBoard (3 classes)."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    num_images = min(num_images, images.shape[0])
    rows = []

    for i in range(num_images):
        img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt = masks[i].cpu().numpy()
        pred = preds[i].cpu().numpy()

        gt_color = mask_to_color_v2(gt)
        pred_color = mask_to_color_v2(pred)

        # Overlay
        overlay = img.copy()
        non_bg = pred > 0
        overlay[non_bg] = cv2.addWeighted(
            img[non_bg], 0.5,
            pred_color[non_bg], 0.5,
            0
        )

        row = np.concatenate([img, gt_color, pred_color, overlay], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    return grid


def format_class_iou_v2(class_iou: Dict[int, float]) -> str:
    """Format class IoU for display (3 classes)."""
    parts = []
    abbrevs = ['bg', 'gb', 'tool']
    for c, abbrev in enumerate(abbrevs):
        iou = class_iou.get(c, float('nan'))
        if np.isnan(iou):
            parts.append(f"{abbrev}=N/A")
        else:
            parts.append(f"{abbrev}={iou*100:.1f}")
    return ' '.join(parts)


# =============================================================================
# MODEL CREATION
# =============================================================================
def create_v2_model(
    decoder: str = 'unet',
    device: str = 'cpu',
    freeze_encoder: bool = True,
) -> nn.Module:
    """
    Create SAM2 segmentation model for 3-class segmentation.
    """
    from model_variants import create_model

    model = create_model(
        variant=decoder,
        num_classes=COARSE_V2_NUM_CLASSES,  # 3 classes
        target_size=DEFAULT_CONFIG['target_size'],
        freeze_encoder=freeze_encoder,
        device=device,
    )

    return model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    tracker = MetricTrackerV2()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        output = model(images)
        logits = output['logits']

        # Loss ignores pixels with mask=-1 (anatomy)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            tracker.update(loss.item(), pred, masks)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[Dict[str, float], Optional[Tuple]]:
    """Validate and return metrics + sample predictions."""
    model.eval()
    tracker = MetricTrackerV2()

    vis_images, vis_masks, vis_preds = None, None, None

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        output = model(images)
        logits = output['logits']

        loss = criterion(logits, masks)
        pred = torch.argmax(logits, dim=1)
        tracker.update(loss.item(), pred, masks)

        if i == 0:
            vis_images = images.cpu()
            vis_masks = masks.cpu()
            vis_preds = pred.cpu()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = tracker.compute()
    return metrics, (vis_images, vis_masks, vis_preds)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    config: Dict,
    path: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    config_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config_save,
        'num_classes': COARSE_V2_NUM_CLASSES,
        'class_names': COARSE_V2_CLASS_NAMES,
        'version': 'v2',  # Mark as V2 model
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


# =============================================================================
# MAIN TRAINING
# =============================================================================
def run_training(config: Dict):
    """Run V2 (3-class) training."""

    # =========================================================================
    # SETUP
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = config['output_dir'] / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = exp_dir / 'config.json'
    config_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)

    # TensorBoard
    writer = SummaryWriter(exp_dir / 'tensorboard')

    # =========================================================================
    # PRINT EXPERIMENT SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" COARSE SEGMENTATION V2 TRAINING (3 Classes - Anatomy IGNORED)")
    print("=" * 70)
    print(f"""
Classes:      {COARSE_V2_CLASS_NAMES}
IGNORED:      anatomy pixels (cystic_plate, calot_triangle, cystic_artery, cystic_duct)
Decoder:      {config['decoder']}
Loss:         CrossEntropyLoss with ignore_index=-1
Batch size:   {config['batch_size']}
Epochs:       {config['epochs']}
LR:           {config['lr']} -> 0 (cosine)
Augmentation: {config['use_augmentation']}
Output dir:   {exp_dir}

Key benefit: Model learns clean bg/gb/tool boundaries.
             Anatomy regions become "uncertain" at inference.
""")
    print("=" * 70)

    # Device info
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # =========================================================================
    # DATA
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading Data")
    print("-" * 50)

    # Training dataset
    train_images_dir, masks_dir = get_split_paths('train')
    train_dataset = CoarseV2EndoscapesDataset(
        images_dir=train_images_dir,
        masks_dir=masks_dir,
        target_size=config['target_size'],
        normalize=True,
    )

    # Apply augmentation
    if config['use_augmentation']:
        augmentation = SegmentationAugmentation(
            flip_prob=config['flip_prob'],
            rotation_degrees=config['rotation_degrees'],
            color_jitter=config['color_jitter'],
        )
        train_dataset_aug = AugmentedDataset(train_dataset, augmentation)
    else:
        train_dataset_aug = train_dataset

    # Validation dataset
    val_images_dir, masks_dir = get_split_paths('val')
    val_dataset = CoarseV2EndoscapesDataset(
        images_dir=val_images_dir,
        masks_dir=masks_dir,
        target_size=config['target_size'],
        normalize=True,
    )

    print(f"\nTraining samples: {len(train_dataset_aug)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Creating Model (3 output classes)")
    print("-" * 50)

    model = create_v2_model(
        decoder=config['decoder'],
        device=device,
        freeze_encoder=config['freeze_encoder'],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # =========================================================================
    # LOSS
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Setting up Loss Function")
    print("-" * 50)

    class_weights = None
    if config['use_class_weights']:
        class_weights = compute_class_weights_v2(train_dataset)
        class_weights = class_weights.to(device)

    # CrossEntropyLoss with ignore_index=-1 for anatomy
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=config['ignore_index']  # -1 = anatomy ignored
    )

    print(f"Loss function: CrossEntropyLoss")
    print(f"Ignore index: {config['ignore_index']} (anatomy pixels not in loss)")

    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01,
    )

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "=" * 70)
    print(" Starting Training")
    print("=" * 70)

    best_miou = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics, vis_data = validate(
            model, val_loader, criterion, device, epoch
        )

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # GPU memory tracking
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        epoch_time = time.time() - epoch_start

        # =====================================================================
        # LOGGING
        # =====================================================================

        # TensorBoard scalars
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/PixelAccuracy', train_metrics['pixel_accuracy'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/PixelAccuracy', val_metrics['pixel_accuracy'], epoch)
        writer.add_scalar('Val/MeanIoU', val_metrics['mean_iou'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        for c, iou in val_metrics['class_iou'].items():
            if not np.isnan(iou):
                writer.add_scalar(f'Val/IoU_{COARSE_V2_CLASS_NAMES[c]}', iou, epoch)

        # TensorBoard images
        if (epoch + 1) % config['log_images_every'] == 0 and vis_data is not None:
            vis_images, vis_masks, vis_preds = vis_data
            grid = create_visualization_grid_v2(
                vis_images, vis_masks, vis_preds,
                num_images=config['num_log_images']
            )
            writer.add_image('Validation/Predictions', grid, epoch, dataformats='HWC')

        # =====================================================================
        # CHECKPOINTING
        # =====================================================================

        is_best = val_metrics['mean_iou'] > best_miou
        if is_best:
            best_miou = val_metrics['mean_iou']
            best_epoch = epoch + 1

        if (epoch + 1) % config['save_every'] == 0 or is_best:
            checkpoint_path = exp_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, config, checkpoint_path, is_best
            )

        # =====================================================================
        # CONSOLE OUTPUT
        # =====================================================================

        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['pixel_accuracy']*100:.1f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['pixel_accuracy']*100:.1f}% | Val mIoU: {val_metrics['mean_iou']*100:.1f}%")
        print(f"Per-class IoU: {format_class_iou_v2(val_metrics['class_iou'])}")
        print(f"Time: {epoch_time:.0f}s | GPU Mem: {gpu_mem:.1f}GB | LR: {current_lr:.2e}")

        if is_best:
            print(f">>> New best model! (mIoU: {best_miou*100:.1f}%)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)

    summary = f"""
Coarse Segmentation V2 (3 classes - anatomy IGNORED):
  Classes:      {COARSE_V2_CLASS_NAMES}
  Best epoch:   {best_epoch}
  Best mIoU:    {best_miou*100:.2f}%
  Total time:   {total_time/60:.1f} minutes
  Checkpoints:  {exp_dir}
  Best model:   {exp_dir / 'best_model.pt'}
  TensorBoard:  {exp_dir / 'tensorboard'}

Next steps:
  - Run evaluate_coarse_v2.py to compute search region quality
  - Use inference_coarse_v2.py for downstream anatomy detection
"""
    print(summary)

    # Save final summary
    with open(exp_dir / 'summary.txt', 'w') as f:
        f.write(summary)

    writer.close()

    return {
        'exp_dir': str(exp_dir),
        'best_epoch': best_epoch,
        'best_miou': best_miou,
        'total_time': total_time,
    }


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Coarse Segmentation V2 Training (3 classes - anatomy ignored)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class weighting')

    return parser.parse_args()


def main():
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'use_augmentation': not args.no_augmentation,
        'use_class_weights': not args.no_class_weights,
    })

    result = run_training(config)

    print("\n" + "=" * 70)
    print(" EXPERIMENT FINISHED")
    print("=" * 70)
    print(f"Results saved to: {result['exp_dir']}")

    return result


if __name__ == '__main__':
    main()
