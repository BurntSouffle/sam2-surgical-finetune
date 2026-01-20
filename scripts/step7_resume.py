"""
Step 7 (Resume): Resume Training from Checkpoint
Usage: python step7_resume.py --checkpoint path/to/checkpoint.pt [--epochs N]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from step4_dataset import (
    EndoscapesDataset, get_split_paths,
    CLASS_NAMES, NUM_CLASSES
)
from step5_model_setup import (
    SAM2SemanticSegmentation, CHECKPOINT_PATH, CONFIG_FILE
)
from step7_train import (
    SegmentationAugmentation, AugmentedDataset,
    compute_class_weights, MetricTracker,
    train_one_epoch, validate, save_checkpoint,
    format_class_iou, create_visualization_grid, CONFIG
)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load checkpoint and return model, optimizer state, and metadata."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    # weights_only=False needed for loading our training checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get('config', CONFIG)

    # Load SAM2 base model
    from sam2.build_sam import build_sam2

    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device='cpu',
        mode='eval',
    )

    # Create segmentation model
    model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=config['target_size'],
        freeze_encoder=config['freeze_encoder'],
        use_multiscale_head=config['use_multiscale_head'],
        hidden_dim=config['hidden_dim'],
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Get training state
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous val mIoU: {metrics.get('mean_iou', 0)*100:.2f}%")

    return model, checkpoint, config, start_epoch


def resume_training(
    checkpoint_path: Path,
    additional_epochs: Optional[int] = None,
):
    """Resume training from a checkpoint."""
    print("\n" + "="*70)
    print(" RESUME TRAINING")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    model, checkpoint, config, start_epoch = load_checkpoint(checkpoint_path, device)

    # Determine total epochs
    if additional_epochs is not None:
        total_epochs = start_epoch + additional_epochs
    else:
        total_epochs = config['epochs']

    if start_epoch >= total_epochs:
        print(f"\nAlready trained for {start_epoch} epochs (target: {total_epochs})")
        print("Use --epochs to specify additional epochs to train")
        return

    print(f"\nResuming from epoch {start_epoch} to {total_epochs}")

    # Setup directories (use same run directory as checkpoint)
    run_dir = checkpoint_path.parent
    log_dir = config['log_dir'] / run_dir.name

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
    if config.get('use_augmentation', True):
        augmentation = SegmentationAugmentation(
            flip_prob=config.get('flip_prob', 0.5),
            rotation_degrees=config.get('rotation_degrees', 15),
            color_jitter=config.get('color_jitter', 0.2),
        )
        train_dataset = AugmentedDataset(train_dataset, augmentation)

    # Validation dataset
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
    # LOSS, OPTIMIZER, SCHEDULER
    # =========================================================================

    # Class weights
    class_weights = None
    if config.get('use_class_weights', True):
        base_train = train_dataset.base_dataset if hasattr(train_dataset, 'base_dataset') else train_dataset
        class_weights = compute_class_weights(base_train)
        class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=config.get('ignore_index', -1),
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config['lr'] * 0.01,
    )

    # Load scheduler state if available
    if checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # =========================================================================
    # TENSORBOARD
    # =========================================================================
    writer = SummaryWriter(log_dir)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "="*70)
    print(" Continuing Training")
    print("="*70)

    best_miou = checkpoint.get('metrics', {}).get('mean_iou', 0.0)
    best_epoch = checkpoint['epoch'] + 1
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
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

        # TensorBoard images
        if (epoch + 1) % config.get('log_images_every', 10) == 0 and vis_data is not None:
            vis_images, vis_masks, vis_preds = vis_data
            grid = create_visualization_grid(
                vis_images, vis_masks, vis_preds,
                num_images=config.get('num_log_images', 4)
            )
            writer.add_image('Validation/Predictions', grid, epoch, dataformats='HWC')

        # =====================================================================
        # CHECKPOINTING
        # =====================================================================

        is_best = val_metrics['mean_iou'] > best_miou
        if is_best:
            best_miou = val_metrics['mean_iou']
            best_epoch = epoch + 1

        if (epoch + 1) % config.get('save_every', 5) == 0 or is_best:
            checkpoint_path_new = run_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, config, checkpoint_path_new, is_best
            )

        # =====================================================================
        # CONSOLE OUTPUT
        # =====================================================================

        print(f"\nEpoch {epoch+1}/{total_epochs}")
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
    print(f"Additional training time: {total_time/60:.1f} minutes")
    print(f"\nBest checkpoint: {run_dir / 'best_model.pt'}")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Resume SAM2 Training from Checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Additional epochs to train (added to current)')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    resume_training(checkpoint_path, args.epochs)


if __name__ == '__main__':
    main()
