"""
Step 7: Full Training Script for SAM2 Semantic Segmentation
Includes proper evaluation, checkpointing, and TensorBoard logging.
"""

import os
import sys
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
from step5_model_setup import (
    SAM2SemanticSegmentation, CHECKPOINT_PATH, CONFIG_FILE
)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    'target_size': (512, 512),
    'batch_size': 4,  # Conservative for training with augmentation
    'num_workers': 0,  # Windows compatibility (set to 4 on Linux)

    # Training
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-4,

    # Model
    'freeze_encoder': True,
    'use_multiscale_head': True,
    'hidden_dim': 128,

    # Loss - handle class imbalance
    'use_class_weights': True,
    'ignore_index': -1,

    # Augmentation
    'use_augmentation': True,
    'flip_prob': 0.5,
    'rotation_degrees': 15,
    'color_jitter': 0.2,

    # Checkpointing
    'save_every': 5,
    'save_best': True,

    # Paths
    'checkpoint_dir': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\training_runs'),
    'log_dir': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs\tensorboard'),

    # Logging
    'log_images_every': 10,  # Log validation images every N epochs
    'num_log_images': 4,     # Number of images to log
}


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
class SegmentationAugmentation:
    """
    Data augmentation for semantic segmentation.
    Applies same geometric transforms to image and mask.
    """

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
        """
        Apply augmentations.

        Args:
            image: (C, H, W) tensor, normalized
            mask: (H, W) tensor, integer labels

        Returns:
            Augmented (image, mask) tuple
        """
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[2])  # Flip W dimension
            mask = torch.flip(mask, dims=[1])    # Flip W dimension

        # Random rotation
        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            if abs(angle) > 0.5:  # Only rotate if angle is significant
                image, mask = self._rotate(image, mask, angle)

        # Color jitter (only on image, not mask)
        if self.color_jitter > 0:
            image = self._color_jitter(image)

        return image, mask

    def _rotate(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        angle: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate image and mask by angle degrees."""
        # Convert to numpy for cv2
        img_np = image.permute(1, 2, 0).numpy()  # (H, W, C)
        mask_np = mask.numpy().astype(np.int32)  # (H, W)

        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate image (bilinear) - use reflect for borders
        img_rot = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

        # Rotate mask (nearest neighbor) - use constant value -1 for borders
        mask_rot = cv2.warpAffine(mask_np, M, (w, h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=-1)

        # Convert back to tensor
        image = torch.from_numpy(img_rot).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask_rot).long()

        return image, mask

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness and contrast adjustments."""
        # Denormalize
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = image * std + mean

        # Random brightness
        brightness_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        image = image * brightness_factor

        # Random contrast
        contrast_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        gray = image.mean(dim=0, keepdim=True)
        image = contrast_factor * image + (1 - contrast_factor) * gray

        # Clamp and renormalize
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

        # Safety: ensure mask values are valid (0-6 or -1 for ignore)
        # Map any invalid values (like 7, 255, etc.) to -1 (ignore)
        mask = sample['mask']
        invalid_mask = (mask < -1) | (mask > 6)
        if invalid_mask.any():
            mask[invalid_mask] = -1
            sample['mask'] = mask

        return sample


# =============================================================================
# CLASS WEIGHTS CALCULATION
# =============================================================================
def compute_class_weights(dataset: EndoscapesDataset, num_classes: int = 7) -> torch.Tensor:
    """
    Compute class weights based on inverse frequency.

    Formula: weight[c] = total_pixels / (num_classes * class_pixels[c])

    Args:
        dataset: Training dataset
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    print("\nComputing class weights from training data...")

    class_pixels = torch.zeros(num_classes)
    total_pixels = 0

    for i in tqdm(range(len(dataset)), desc="Scanning masks"):
        sample = dataset[i]
        mask = sample['mask']

        for c in range(num_classes):
            class_pixels[c] += (mask == c).sum()

        # Count valid pixels (not ignore)
        total_pixels += (mask != -1).sum()

    # Compute weights using inverse frequency
    # Add small epsilon to avoid division by zero
    weights = total_pixels / (num_classes * class_pixels + 1e-6)

    # Normalize weights to have mean = 1
    weights = weights / weights.mean()

    # Clip extreme weights
    weights = torch.clamp(weights, min=0.1, max=10.0)

    print("\nClass weights:")
    for c in range(num_classes):
        pct = class_pixels[c] / total_pixels * 100
        print(f"  {c}: {CLASS_NAMES[c]:15s} - {pct:6.2f}% pixels, weight={weights[c]:.3f}")

    return weights


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 7
) -> Dict[str, float]:
    """
    Compute segmentation metrics.

    Args:
        pred: Predicted class indices (B, H, W) or (H, W)
        target: Ground truth class indices, -1 for ignore
        num_classes: Number of classes

    Returns:
        Dict with pixel_accuracy, mean_iou, and per-class IoU
    """
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)

    # Valid mask (not ignore)
    valid = target != -1

    if valid.sum() == 0:
        return {'pixel_accuracy': 0.0, 'mean_iou': 0.0, 'class_iou': {}}

    pred = pred[valid]
    target = target[valid]

    # Pixel accuracy
    correct = (pred == target).sum().float()
    pixel_acc = correct / valid.sum().float()

    # Per-class IoU
    class_iou = {}
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            class_iou[c] = (intersection / union).item()
        else:
            class_iou[c] = float('nan')

    # Mean IoU (excluding NaN)
    valid_ious = [v for v in class_iou.values() if not np.isnan(v)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        'pixel_accuracy': pixel_acc.item(),
        'mean_iou': mean_iou,
        'class_iou': class_iou,
    }


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
        """Update metrics with batch results."""
        self.total_loss += loss
        self.num_batches += 1

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Valid mask
        valid = target != -1

        if valid.sum() == 0:
            return

        pred_valid = pred[valid]
        target_valid = target[valid]

        # Pixel accuracy
        self.total_correct += (pred_valid == target_valid).sum().item()
        self.total_pixels += valid.sum().item()

        # Per-class IoU
        for c in range(self.num_classes):
            pred_c = pred_valid == c
            target_c = target_valid == c

            self.class_intersection[c] += (pred_c & target_c).sum().item()
            self.class_union[c] += (pred_c | target_c).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}

        # Average loss
        metrics['loss'] = self.total_loss / max(self.num_batches, 1)

        # Pixel accuracy
        metrics['pixel_accuracy'] = self.total_correct / max(self.total_pixels, 1)

        # Per-class IoU
        class_iou = {}
        for c in range(self.num_classes):
            if self.class_union[c] > 0:
                class_iou[c] = self.class_intersection[c] / self.class_union[c]
            else:
                class_iou[c] = float('nan')

        metrics['class_iou'] = class_iou

        # Mean IoU
        valid_ious = [v for v in class_iou.values() if not np.isnan(v)]
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0

        return metrics


# =============================================================================
# VISUALIZATION FOR TENSORBOARD
# =============================================================================
def create_visualization_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    num_images: int = 4,
) -> np.ndarray:
    """
    Create visualization grid for TensorBoard.

    Returns:
        Grid image as numpy array (H, W, 3)
    """
    # Denormalize images
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

        # Concatenate horizontally: image | gt | pred | overlay
        row = np.concatenate([img, gt_color, pred_color, overlay], axis=1)
        rows.append(row)

    # Concatenate vertically
    grid = np.concatenate(rows, axis=0)

    return grid


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

        # Forward
        optimizer.zero_grad()
        output = model(images)
        logits = output['logits']

        # Loss
        loss = criterion(logits, masks)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            tracker.update(loss.item(), pred, masks)

        # Update progress bar
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
    """Validate and return metrics + sample predictions for visualization."""
    model.eval()
    tracker = MetricTracker()

    # Store samples for visualization
    vis_images, vis_masks, vis_preds = None, None, None

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward
        output = model(images)
        logits = output['logits']

        # Loss
        loss = criterion(logits, masks)

        # Predictions
        pred = torch.argmax(logits, dim=1)
        tracker.update(loss.item(), pred, masks)

        # Store first batch for visualization
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
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


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
# MAIN TRAINING FUNCTION
# =============================================================================
def train(config: Dict):
    """Main training function."""
    print("\n" + "="*70)
    print(" SAM2 SEMANTIC SEGMENTATION TRAINING")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = config['checkpoint_dir'] / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config['log_dir'] / f'run_{timestamp}'

    print(f"Run directory: {run_dir}")
    print(f"Log directory: {log_dir}")

    # =========================================================================
    # DATA
    # =========================================================================
    print("\n" + "-"*50)
    print(" Loading Data")
    print("-"*50)

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
        train_dataset = AugmentedDataset(train_dataset, augmentation)
        print("Data augmentation: ENABLED")

    # Validation dataset (no augmentation)
    val_images_dir, masks_dir = get_split_paths('val')
    val_dataset = EndoscapesDataset(
        images_dir=val_images_dir,
        masks_dir=masks_dir,
        target_size=config['target_size'],
        normalize=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
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
    # CLASS WEIGHTS
    # =========================================================================
    class_weights = None
    if config['use_class_weights']:
        # Use base dataset (without augmentation wrapper) for weight calculation
        base_train = train_dataset.base_dataset if hasattr(train_dataset, 'base_dataset') else train_dataset
        class_weights = compute_class_weights(base_train)
        class_weights = class_weights.to(device)

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n" + "-"*50)
    print(" Loading Model")
    print("-"*50)

    from sam2.build_sam import build_sam2

    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device='cpu',
        mode='eval',
    )

    model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=config['target_size'],
        freeze_encoder=config['freeze_encoder'],
        use_multiscale_head=config['use_multiscale_head'],
        hidden_dim=config['hidden_dim'],
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # LOSS, OPTIMIZER, SCHEDULER
    # =========================================================================
    print("\n" + "-"*50)
    print(" Training Setup")
    print("-"*50)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=config['ignore_index'],
    )

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

    print(f"Loss: CrossEntropyLoss (weighted={config['use_class_weights']})")
    print(f"Optimizer: AdamW (lr={config['lr']}, weight_decay={config['weight_decay']})")
    print(f"Scheduler: CosineAnnealingLR (T_max={config['epochs']})")

    # =========================================================================
    # TENSORBOARD
    # =========================================================================
    writer = SummaryWriter(log_dir)

    # Log config
    config_str = '\n'.join([f'{k}: {v}' for k, v in config.items()])
    writer.add_text('config', config_str)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*70)
    print(" Starting Training")
    print("="*70)

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

        # GPU memory
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        torch.cuda.reset_peak_memory_stats()

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

        # TensorBoard images (every N epochs)
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

        # Check for best model
        is_best = val_metrics['mean_iou'] > best_miou
        if is_best:
            best_miou = val_metrics['mean_iou']
            best_epoch = epoch + 1

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0 or is_best:
            checkpoint_path = run_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
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
            print(f">>> Saved best model (mIoU: {best_miou*100:.1f}%)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best mIoU: {best_miou*100:.2f}%")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"\nBest checkpoint: {run_dir / 'best_model.pt'}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nTo view TensorBoard: tensorboard --logdir {config['log_dir']}")

    writer.close()

    return run_dir / 'best_model.pt'


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train SAM2 Semantic Segmentation')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable augmentation')
    parser.add_argument('--no_class_weights', action='store_true', help='Disable class weights')

    args = parser.parse_args()

    # Update config with command line arguments
    config = CONFIG.copy()

    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.no_augmentation:
        config['use_augmentation'] = False
    if args.no_class_weights:
        config['use_class_weights'] = False

    # Run training
    train(config)


if __name__ == '__main__':
    main()
