"""
SAM2.1 Fine-tuning Script for Endoscapes2023 Dataset.

Supports:
- Multi-GPU distributed training with DDP
- Differential learning rates for encoder/decoder
- Mixed precision (bfloat16) training
- Gradient checkpointing for memory efficiency
- Combined Dice + Focal + IoU loss
- Checkpoint saving and resuming
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add SAM2 to path (assumes sam2 repo is cloned)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import Config, get_config
from dataset import EndoscapesDataset, collate_fn


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, H, W)
            target: Ground truth binary mask (N, H, W)
        """
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, H, W)
            target: Ground truth binary mask (N, H, W)
        """
        pred = pred.flatten(1)
        target = target.flatten(1)

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = torch.sigmoid(pred)

        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        focal_loss = focal_weight * bce
        return focal_loss.mean()


class IoULoss(nn.Module):
    """IoU loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (N, H, W)
            target: Ground truth binary mask (N, H, W)
        """
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Focal + IoU loss."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 20.0,
        iou_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: Predicted logits (N, H, W)
            target: Ground truth binary mask (N, H, W)

        Returns:
            total_loss: Combined loss value
            loss_dict: Individual loss components
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)

        total = (
            self.dice_weight * dice
            + self.focal_weight * focal
            + self.iou_weight * iou
        )

        loss_dict = {
            "dice": dice.item(),
            "focal": focal.item(),
            "iou": iou.item(),
            "total": total.item(),
        }

        return total, loss_dict


# ============================================================================
# Training Utilities
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_sam2_model(config: Config, device: torch.device):
    """Load SAM2 model."""
    checkpoint_path = config.paths.pretrained_dir / config.model.checkpoint_filename

    if not checkpoint_path.exists():
        logger.info(f"Downloading SAM2 checkpoint to {checkpoint_path}...")
        import urllib.request
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.model.checkpoint_url, checkpoint_path)
        logger.info("Download complete.")

    # Build SAM2 model
    model = build_sam2(
        config_file=config.model.model_config,
        ckpt_path=str(checkpoint_path),
        device=device,
    )

    return model


def setup_optimizer(model: nn.Module, config: Config) -> AdamW:
    """Setup optimizer with differential learning rates."""
    # Group parameters by component
    encoder_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "image_encoder" in name:
            encoder_params.append(param)
        elif "mask_decoder" in name or "sam_mask_decoder" in name:
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": encoder_params, "lr": config.train.lr_image_encoder},
        {"params": decoder_params, "lr": config.train.lr_mask_decoder},
        {"params": other_params, "lr": config.train.lr_mask_decoder},
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=config.train.weight_decay,
    )

    return optimizer


def setup_scheduler(optimizer: AdamW, config: Config, steps_per_epoch: int):
    """Setup learning rate scheduler with warmup."""
    warmup_steps = config.train.warmup_epochs * steps_per_epoch
    total_steps = config.train.num_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.train.min_lr,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU between prediction and target."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection

    if union == 0:
        return 1.0 if target.sum() == 0 else 0.0

    return (intersection / union).item()


# ============================================================================
# Training Loop
# ============================================================================

class SAM2Trainer:
    """SAM2 fine-tuning trainer."""

    def __init__(
        self,
        config: Config,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        self.is_main = rank == 0

        # Initialize model
        logger.info("Loading SAM2 model...")
        self.model = get_sam2_model(config, self.device)

        # Configure trainable parameters
        self._configure_trainable_params()

        # Enable gradient checkpointing
        if config.model.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Wrap with DDP if distributed
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        # Setup training components
        self.optimizer = setup_optimizer(
            self.model.module if world_size > 1 else self.model,
            config,
        )

        # Loss function
        self.criterion = CombinedLoss(
            dice_weight=config.train.dice_weight,
            focal_weight=config.train.focal_weight,
            iou_weight=config.train.iou_weight,
            focal_alpha=config.train.focal_alpha,
            focal_gamma=config.train.focal_gamma,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.train.use_amp else None
        self.amp_dtype = torch.bfloat16 if config.train.amp_dtype == "bfloat16" else torch.float16

        # Tracking
        self.best_miou = 0.0
        self.epoch = 0
        self.global_step = 0

        # Create SAM2 predictor for inference
        self.predictor = SAM2ImagePredictor(
            self.model.module if world_size > 1 else self.model
        )

    def _configure_trainable_params(self):
        """Configure which parameters to train."""
        model = self.model

        # Freeze/unfreeze image encoder
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = self.config.model.train_image_encoder

        # Freeze/unfreeze mask decoder
        for name, param in model.named_parameters():
            if "mask_decoder" in name or "sam_mask_decoder" in name:
                param.requires_grad = self.config.model.train_mask_decoder

        # Freeze/unfreeze prompt encoder
        for name, param in model.named_parameters():
            if "prompt_encoder" in name or "sam_prompt_encoder" in name:
                param.requires_grad = self.config.model.train_prompt_encoder

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.is_main:
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        model = self.model

        # Enable for image encoder if it has the method
        if hasattr(model, "image_encoder"):
            encoder = model.image_encoder
            if hasattr(encoder, "set_grad_checkpointing"):
                encoder.set_grad_checkpointing(True)
            elif hasattr(encoder, "gradient_checkpointing_enable"):
                encoder.gradient_checkpointing_enable()

        if self.is_main:
            logger.info("Gradient checkpointing enabled")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.epoch = epoch

        total_loss = 0.0
        loss_components = {"dice": 0.0, "focal": 0.0, "iou": 0.0}
        num_batches = 0
        total_iou = 0.0
        num_masks = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main,
        )

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            # Move data to device
            images = batch["images"].to(self.device)
            masks_list = batch["masks"]
            point_coords_list = batch["point_coords"]
            point_labels_list = batch["point_labels"]

            batch_loss = 0.0
            batch_iou = 0.0
            batch_masks = 0

            # Process each image in the batch
            for img_idx in range(len(images)):
                image = images[img_idx]
                gt_masks = masks_list[img_idx].to(self.device)
                point_coords = point_coords_list[img_idx].to(self.device)
                point_labels = point_labels_list[img_idx].to(self.device)

                # Process each mask/class
                for mask_idx in range(len(gt_masks)):
                    gt_mask = gt_masks[mask_idx].unsqueeze(0)  # (1, H, W)
                    coords = point_coords[mask_idx].unsqueeze(0)  # (1, P, 2)
                    labels = point_labels[mask_idx].unsqueeze(0)  # (1, P)

                    # Filter out padding points (label == -1)
                    valid_mask = labels[0] >= 0
                    if valid_mask.sum() == 0:
                        continue

                    coords = coords[:, valid_mask, :]
                    labels = labels[:, valid_mask]

                    # Forward pass
                    with autocast(
                        device_type="cuda",
                        dtype=self.amp_dtype,
                        enabled=self.config.train.use_amp,
                    ):
                        # Set image
                        self.predictor.set_image(
                            (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        )

                        # Predict with point prompts
                        pred_masks, scores, logits = self.predictor.predict(
                            point_coords=coords[0].cpu().numpy(),
                            point_labels=labels[0].cpu().numpy(),
                            multimask_output=False,
                        )

                        # Get logits and resize to match GT
                        pred_logits = torch.from_numpy(logits[0]).to(self.device)  # (H, W)

                        # Resize prediction to match GT mask size if needed
                        if pred_logits.shape != gt_mask.shape[1:]:
                            pred_logits = F.interpolate(
                                pred_logits.unsqueeze(0).unsqueeze(0),
                                size=gt_mask.shape[1:],
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).squeeze(0)

                        pred_logits = pred_logits.unsqueeze(0)  # (1, H, W)

                        # Compute loss
                        loss, loss_dict = self.criterion(pred_logits, gt_mask)

                    # Backward pass
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    batch_loss += loss.item()
                    batch_iou += compute_iou(pred_logits, gt_mask)
                    batch_masks += 1

                    # Accumulate loss components
                    for key in loss_components:
                        loss_components[key] += loss_dict[key]

            # Optimizer step
            if batch_masks > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                total_loss += batch_loss
                total_iou += batch_iou
                num_masks += batch_masks
                num_batches += 1

                # Update progress bar
                avg_loss = total_loss / num_masks if num_masks > 0 else 0
                avg_iou = total_iou / num_masks if num_masks > 0 else 0
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "iou": f"{avg_iou:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

        # Compute epoch averages
        metrics = {
            "train_loss": total_loss / num_masks if num_masks > 0 else 0,
            "train_iou": total_iou / num_masks if num_masks > 0 else 0,
            "train_dice_loss": loss_components["dice"] / num_masks if num_masks > 0 else 0,
            "train_focal_loss": loss_components["focal"] / num_masks if num_masks > 0 else 0,
            "train_iou_loss": loss_components["iou"] / num_masks if num_masks > 0 else 0,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_iou = 0.0
        class_ious = {cls: [] for cls in self.config.data.train_classes}
        num_masks = 0

        for batch in tqdm(val_loader, desc="Validating", disable=not self.is_main):
            if batch is None:
                continue

            images = batch["images"].to(self.device)
            masks_list = batch["masks"]
            point_coords_list = batch["point_coords"]
            point_labels_list = batch["point_labels"]
            class_ids_list = batch["class_ids"]

            for img_idx in range(len(images)):
                image = images[img_idx]
                gt_masks = masks_list[img_idx].to(self.device)
                point_coords = point_coords_list[img_idx].to(self.device)
                point_labels = point_labels_list[img_idx].to(self.device)
                class_ids = class_ids_list[img_idx]

                for mask_idx in range(len(gt_masks)):
                    gt_mask = gt_masks[mask_idx]
                    coords = point_coords[mask_idx]
                    labels = point_labels[mask_idx]
                    class_id = class_ids[mask_idx].item()

                    valid_mask = labels >= 0
                    if valid_mask.sum() == 0:
                        continue

                    coords = coords[valid_mask].unsqueeze(0)
                    labels = labels[valid_mask].unsqueeze(0)

                    # Set image
                    self.predictor.set_image(
                        (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    )

                    # Predict
                    pred_masks, scores, logits = self.predictor.predict(
                        point_coords=coords[0].cpu().numpy(),
                        point_labels=labels[0].cpu().numpy(),
                        multimask_output=False,
                    )

                    pred_logits = torch.from_numpy(logits[0]).to(self.device)

                    if pred_logits.shape != gt_mask.shape:
                        pred_logits = F.interpolate(
                            pred_logits.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape,
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()

                    iou = compute_iou(pred_logits.unsqueeze(0), gt_mask.unsqueeze(0))
                    total_iou += iou
                    class_ious[class_id].append(iou)
                    num_masks += 1

        # Compute metrics
        miou = total_iou / num_masks if num_masks > 0 else 0

        metrics = {
            "val_miou": miou,
            "val_num_masks": num_masks,
        }

        # Per-class IoU
        for cls_id, cls_name in enumerate(self.config.data.class_names):
            if cls_id in class_ious and len(class_ious[cls_id]) > 0:
                cls_iou = np.mean(class_ious[cls_id])
                metrics[f"val_iou_{cls_name}"] = cls_iou

        return metrics

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main:
            return

        model_state = (
            self.model.module.state_dict()
            if self.world_size > 1
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": self.best_miou,
            "config": {
                "model": self.config.model.__dict__,
                "train": self.config.train.__dict__,
                "data": self.config.data.__dict__,
            },
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        model = self.model.module if self.world_size > 1 else self.model
        model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_miou = checkpoint.get("best_miou", 0.0)

        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        # Setup scheduler (needs to know steps per epoch)
        steps_per_epoch = len(train_loader)
        self.scheduler = setup_scheduler(self.optimizer, self.config, steps_per_epoch)

        # Resume if specified
        if self.config.resume_from:
            self.load_checkpoint(Path(self.config.resume_from))

        start_epoch = self.epoch + 1

        # Training history
        history = {"train": [], "val": []}

        for epoch in range(start_epoch, self.config.train.num_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Log training metrics
            if self.is_main:
                logger.info(
                    f"Epoch {epoch} - "
                    f"Loss: {train_metrics['train_loss']:.4f}, "
                    f"IoU: {train_metrics['train_iou']:.4f}, "
                    f"LR: {train_metrics['learning_rate']:.2e}"
                )
                history["train"].append(train_metrics)

            # Validate
            if epoch % self.config.train.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)

                if self.is_main:
                    logger.info(
                        f"Validation - mIoU: {val_metrics['val_miou']:.4f}"
                    )

                    # Log per-class IoU
                    for cls_name in self.config.data.class_names[1:]:  # Skip background
                        key = f"val_iou_{cls_name}"
                        if key in val_metrics:
                            logger.info(f"  {cls_name}: {val_metrics[key]:.4f}")

                    history["val"].append(val_metrics)

                    # Check for best model
                    is_best = val_metrics["val_miou"] > self.best_miou
                    if is_best:
                        self.best_miou = val_metrics["val_miou"]
                        logger.info(f"New best mIoU: {self.best_miou:.4f}")

                    # Save checkpoint
                    if epoch % self.config.train.save_every_n_epochs == 0 or is_best:
                        ckpt_path = (
                            self.config.paths.finetuned_dir
                            / f"checkpoint_epoch_{epoch}.pt"
                        )
                        self.save_checkpoint(ckpt_path, is_best=is_best)

        # Save final model
        if self.is_main:
            final_path = self.config.paths.finetuned_dir / "final_model.pt"
            self.save_checkpoint(final_path)

            # Save training history
            history_path = self.config.paths.log_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved training history to {history_path}")

        return history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAM2.1 Fine-tuning on Endoscapes")
    parser.add_argument("--project-store", type=str, help="Project store path")
    parser.add_argument("--dataset-root", type=str, help="Dataset root path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Create config
    config_kwargs = {"seed": args.seed}
    if args.project_store:
        config_kwargs["project_store"] = args.project_store
    if args.dataset_root:
        config_kwargs["dataset_root"] = args.dataset_root
    if args.epochs:
        config_kwargs["num_epochs"] = args.epochs
    if args.batch_size:
        config_kwargs["batch_size"] = args.batch_size
    if args.resume:
        config_kwargs["resume_from"] = args.resume

    config = get_config(**config_kwargs)

    # Set seed
    set_seed(config.train.seed + rank)

    # Log configuration
    if rank == 0:
        logger.info("=" * 60)
        logger.info("SAM2.1 Fine-tuning on Endoscapes2023")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Project store: {config.paths.project_store}")
        logger.info(f"Dataset root: {config.paths.dataset_root}")
        logger.info(f"Epochs: {config.train.num_epochs}")
        logger.info(f"Batch size: {config.train.batch_size} x {world_size} GPUs")
        logger.info(f"LR (encoder): {config.train.lr_image_encoder}")
        logger.info(f"LR (decoder): {config.train.lr_mask_decoder}")
        logger.info("=" * 60)

    # Create datasets
    train_dataset = EndoscapesDataset(config, split="train", transform=True)
    val_dataset = EndoscapesDataset(config, split="val", transform=False)

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create trainer
    trainer = SAM2Trainer(config, rank, world_size, local_rank)

    # Train
    try:
        trainer.train(train_loader, val_loader)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
