"""
Unified Training Script for SAM2 Semantic Segmentation Experiments

Supports all decoder and loss variants for ablation studies.

Usage:
    # Baseline: Simple decoder + CE loss
    python scripts/train_experiment.py --name baseline --decoder simple --loss ce

    # Focal loss experiment
    python scripts/train_experiment.py --name focal --decoder simple --loss focal --gamma 2.0

    # UNet decoder experiment
    python scripts/train_experiment.py --name unet_ce --decoder unet --loss ce

    # Full upgrade: UNet + Focal + Dice
    python scripts/train_experiment.py --name unet_focal_dice --decoder unet --loss focal+dice

    # Custom settings
    python scripts/train_experiment.py --name custom --decoder unet_large --loss focal \\
        --epochs 100 --batch_size 8 --lr 5e-5 --gamma 3.0
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

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from step4_dataset import (
    EndoscapesDataset, get_split_paths,
    CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, IGNORE_INDEX,
    mask_to_color, create_overlay, IMAGENET_MEAN, IMAGENET_STD
)
from model_variants import create_model, count_parameters, count_trainable_parameters, format_params
from losses import FocalLoss, DiceLoss, CombinedLoss


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG = {
    # Data
    'target_size': (512, 512),
    'batch_size': 4,
    'num_workers': 0,  # Windows compatibility

    # Training
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,

    # Model
    'decoder': 'simple',  # 'simple', 'unet', 'unet_large'
    'freeze_encoder': True,

    # Loss
    'loss_type': 'ce',  # 'ce', 'focal', 'dice', 'ce+dice', 'focal+dice'
    'gamma': 2.0,  # Focal loss gamma
    'dice_weight': 0.5,  # Weight for dice loss in combined

    # Class weighting
    'use_class_weights': True,
    'ignore_index': -1,

    # Augmentation
    'use_augmentation': True,
    'flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter': 0.2,

    # Checkpointing
    'save_every': 10,
    'save_best': True,

    # Logging
    'log_images_every': 10,
    'num_log_images': 4,

    # Paths
    'experiment_dir': BASE_DIR / 'checkpoints' / 'experiments',
}


# =============================================================================
# DATA AUGMENTATION (same as step7_train.py)
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

    def __init__(self, base_dataset: EndoscapesDataset, augmentation=None):
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

        # Safety: ensure mask values are valid
        mask = sample['mask']
        invalid_mask = (mask < -1) | (mask > 6)
        if invalid_mask.any():
            mask[invalid_mask] = -1
            sample['mask'] = mask

        return sample


# =============================================================================
# CLASS WEIGHTS
# =============================================================================
def compute_class_weights(dataset: EndoscapesDataset, num_classes: int = 7) -> torch.Tensor:
    """Compute class weights based on inverse frequency."""
    print("\nComputing class weights from training data...")

    class_pixels = torch.zeros(num_classes)
    total_pixels = 0

    for i in tqdm(range(len(dataset)), desc="Scanning masks"):
        sample = dataset[i]
        mask = sample['mask']

        for c in range(num_classes):
            class_pixels[c] += (mask == c).sum()

        total_pixels += (mask != -1).sum()

    weights = total_pixels / (num_classes * class_pixels + 1e-6)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=10.0)

    print("\nClass weights:")
    for c in range(num_classes):
        pct = class_pixels[c] / total_pixels * 100
        print(f"  {c}: {CLASS_NAMES[c]:15s} - {pct:6.2f}% pixels, weight={weights[c]:.3f}")

    return weights


# =============================================================================
# METRICS
# =============================================================================
class MetricTracker:
    """Track and aggregate metrics over batches."""

    def __init__(self, num_classes: int = 7):
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

        valid = target != -1

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
def create_visualization_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    num_images: int = 4,
) -> np.ndarray:
    """Create visualization grid for TensorBoard."""
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

        gt_color = mask_to_color(gt)
        pred_color = mask_to_color(pred)
        overlay = create_overlay(img, pred, alpha=0.5)

        row = np.concatenate([img, gt_color, pred_color, overlay], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    return grid


def format_class_iou(class_iou: Dict[int, float]) -> str:
    """Format class IoU for display."""
    parts = []
    abbrevs = ['bg', 'cp', 'ct', 'ca', 'cd', 'gb', 'tool']
    for c, abbrev in enumerate(abbrevs):
        iou = class_iou.get(c, float('nan'))
        if np.isnan(iou):
            parts.append(f"{abbrev}=N/A")
        else:
            parts.append(f"{abbrev}={iou*100:.1f}")
    return ' '.join(parts)


# =============================================================================
# LOSS CREATION
# =============================================================================
def create_criterion(
    loss_type: str,
    class_weights: Optional[torch.Tensor],
    gamma: float,
    dice_weight: float,
    ignore_index: int,
    device: torch.device,
) -> nn.Module:
    """
    Create loss function based on type.

    Args:
        loss_type: 'ce', 'focal', 'dice', 'ce+dice', 'focal+dice'
        class_weights: Tensor of class weights (optional)
        gamma: Focal loss gamma parameter
        dice_weight: Weight for dice loss in combined losses
        ignore_index: Index to ignore in loss computation
        device: Device to place weights on

    Returns:
        Loss module
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=gamma, ignore_index=ignore_index)

    elif loss_type == 'dice':
        return DiceLoss(ignore_index=ignore_index)

    elif loss_type in ['ce+dice', 'focal+dice']:
        return CombinedLoss(
            loss_type=loss_type,
            class_weights=class_weights,
            gamma=gamma,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
    tracker = MetricTracker()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()
        output = model(images)
        logits = output['logits']

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
    tracker = MetricTracker()

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
    # Convert Path objects to strings for JSON serialization
    config_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config_save,
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def run_experiment(config: Dict):
    """Run a training experiment with the given configuration."""

    # =========================================================================
    # SETUP
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config['name']}_{timestamp}"
    exp_dir = config['experiment_dir'] / exp_name
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
    decoder_params = {
        'simple': '~395K',
        'unet': '~2.25M',
        'unet_large': '~7.05M',
    }

    loss_desc = {
        'ce': 'CrossEntropyLoss',
        'focal': f"FocalLoss (gamma={config['gamma']})",
        'dice': 'DiceLoss',
        'ce+dice': f"CE + Dice (weight={config['dice_weight']})",
        'focal+dice': f"Focal (gamma={config['gamma']}) + Dice (weight={config['dice_weight']})",
    }

    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {exp_name}")
    print("=" * 60)
    print(f"""
Decoder:      {config['decoder']} ({decoder_params.get(config['decoder'], '???')} trainable params)
Loss:         {loss_desc.get(config['loss_type'], config['loss_type'])}
Batch size:   {config['batch_size']}
Epochs:       {config['epochs']}
LR:           {config['lr']} -> 0 (cosine)
Augmentation: {config['use_augmentation']}
Output dir:   {exp_dir}
""")
    print("=" * 60)

    # =========================================================================
    # DEVICE INFO
    # =========================================================================
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
    train_dataset = EndoscapesDataset(
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
    val_dataset = EndoscapesDataset(
        images_dir=val_images_dir,
        masks_dir=masks_dir,
        target_size=config['target_size'],
        normalize=True,
    )

    print(f"Training samples: {len(train_dataset_aug)}")
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
    print(" Creating Model")
    print("-" * 50)

    model = create_model(
        variant=config['decoder'],
        num_classes=NUM_CLASSES,
        target_size=config['target_size'],
        freeze_encoder=config['freeze_encoder'],
        device=device,
    )

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"Total parameters: {format_params(total_params)}")
    print(f"Trainable parameters: {format_params(trainable_params)}")

    # =========================================================================
    # LOSS
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Setting up Loss Function")
    print("-" * 50)

    class_weights = None
    if config['use_class_weights']:
        class_weights = compute_class_weights(train_dataset)

    criterion = create_criterion(
        loss_type=config['loss_type'],
        class_weights=class_weights,
        gamma=config['gamma'],
        dice_weight=config['dice_weight'],
        ignore_index=config['ignore_index'],
        device=device,
    )

    print(f"Loss function: {type(criterion).__name__}")

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
    print("\n" + "=" * 60)
    print(" Starting Training")
    print("=" * 60)

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
                writer.add_scalar(f'Val/IoU_{CLASS_NAMES[c]}', iou, epoch)

        # TensorBoard images
        if (epoch + 1) % config['log_images_every'] == 0 and vis_data is not None:
            vis_images, vis_masks, vis_preds = vis_data
            grid = create_visualization_grid(
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
        print(f"Per-class IoU: {format_class_iou(val_metrics['class_iou'])}")
        print(f"Time: {epoch_time:.0f}s | GPU Mem: {gpu_mem:.1f}GB | LR: {current_lr:.2e}")

        if is_best:
            print(f">>> New best model! (mIoU: {best_miou*100:.1f}%)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)

    summary = f"""
Experiment:     {exp_name}
Best epoch:     {best_epoch}
Best mIoU:      {best_miou*100:.2f}%
Total time:     {total_time/60:.1f} minutes
Checkpoints:    {exp_dir}
Best model:     {exp_dir / 'best_model.pt'}
TensorBoard:    {exp_dir / 'tensorboard'}
"""
    print(summary)

    # Save final summary
    with open(exp_dir / 'summary.txt', 'w') as f:
        f.write(summary)

    writer.close()

    return {
        'exp_name': exp_name,
        'best_epoch': best_epoch,
        'best_miou': best_miou,
        'total_time': total_time,
        'exp_dir': str(exp_dir),
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='SAM2 Semantic Segmentation Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment name
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name (e.g., baseline, focal, unet_ce)')

    # Model
    parser.add_argument('--decoder', type=str, default='simple',
                        choices=['simple', 'unet', 'unet_large'],
                        help='Decoder variant')

    # Loss
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'focal', 'dice', 'ce+dice', 'focal+dice'],
                        help='Loss function type')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--dice_weight', type=float, default=0.5,
                        help='Weight for dice loss in combined losses')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Data
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class weighting')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'name': args.name,
        'decoder': args.decoder,
        'loss_type': args.loss,
        'gamma': args.gamma,
        'dice_weight': args.dice_weight,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'use_augmentation': not args.no_augmentation,
        'use_class_weights': not args.no_class_weights,
        'num_workers': args.num_workers,
        'save_every': args.save_every,
    })

    # Run experiment
    result = run_experiment(config)

    print("\n" + "=" * 60)
    print(" EXPERIMENT FINISHED")
    print("=" * 60)
    print(f"Results saved to: {result['exp_dir']}")

    return result


if __name__ == '__main__':
    main()
