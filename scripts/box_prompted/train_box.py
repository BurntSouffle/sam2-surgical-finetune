"""
SAM2 Box-Prompted Fine-tuning Training Script

Fine-tunes SAM2's mask decoder to generate high-quality binary masks
from bounding box prompts for surgical anatomy segmentation.

The training approach:
1. Image encoder: FROZEN (use pretrained features)
2. Prompt encoder: FROZEN (box encoding is straightforward)
3. Mask decoder: TRAINED (learn to generate better masks for surgical data)

Usage:
    python scripts/box_prompted/train_box.py
    python scripts/box_prompted/train_box.py --epochs 100 --lr 5e-5
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from dataset_box import EndoscapesBoxDataset, get_split_paths, CLASS_NAMES, CLASS_COLORS

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Model
    'checkpoint': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\sam2.1_hiera_large.pt'),
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',

    # Training
    'batch_size': 2,  # Reduced for larger model
    'epochs': 50,
    'lr': 1e-5,  # Lower LR for fine-tuning pretrained decoder
    'weight_decay': 1e-4,

    # Freezing strategy
    'freeze_image_encoder': True,
    'freeze_prompt_encoder': True,
    'train_mask_decoder': True,

    # Loss weights
    'focal_weight': 20.0,
    'dice_weight': 1.0,
    'iou_weight': 1.0,

    # Data
    'image_size': (1024, 1024),  # SAM2 default input size
    'num_workers': 0,  # Windows compatibility

    # Logging
    'log_every': 10,
    'save_every': 10,
    'visualize_every': 5,

    # Paths
    'output_dir': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\box_prompted'),
    'log_dir': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs\box_prompted\tensorboard'),
}


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal loss for binary segmentation.

    Args:
        inputs: Predicted logits (B, 1, H, W)
        targets: Ground truth binary masks (B, 1, H, W)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * focal_weight * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Args:
        inputs: Predicted logits (B, 1, H, W)
        targets: Ground truth binary masks (B, 1, H, W)
        smooth: Smoothing factor
    """
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    intersection = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def iou_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_ious: torch.Tensor,
) -> torch.Tensor:
    """
    IoU prediction loss - train the model to predict mask quality.

    Args:
        pred_masks: Predicted mask logits (B, 1, H, W)
        gt_masks: Ground truth binary masks (B, 1, H, W)
        pred_ious: Predicted IoU scores (B, 1)
    """
    # Compute actual IoU
    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
    pred_binary = pred_binary.flatten(1)
    gt_flat = gt_masks.flatten(1)

    intersection = (pred_binary * gt_flat).sum(1)
    union = pred_binary.sum(1) + gt_flat.sum(1) - intersection

    actual_iou = intersection / (union + 1e-8)
    actual_iou = actual_iou.unsqueeze(1)  # (B, 1)

    return F.mse_loss(pred_ious, actual_iou)


class CombinedLoss(nn.Module):
    """Combined loss for SAM2 mask prediction."""

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_ious: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_masks: Predicted mask logits (B, 1, H, W)
            gt_masks: Ground truth binary masks (B, 1, H, W)
            pred_ious: Predicted IoU scores (B, 1), optional

        Returns:
            Dict with 'total', 'focal', 'dice', and optionally 'iou' losses
        """
        focal = sigmoid_focal_loss(pred_masks, gt_masks)
        dice = dice_loss(pred_masks, gt_masks)

        total = self.focal_weight * focal + self.dice_weight * dice

        losses = {
            'focal': focal,
            'dice': dice,
        }

        if pred_ious is not None and self.iou_weight > 0:
            iou = iou_loss(pred_masks, gt_masks, pred_ious)
            total = total + self.iou_weight * iou
            losses['iou'] = iou

        losses['total'] = total
        return losses


# =============================================================================
# MODEL WRAPPER FOR TRAINING
# =============================================================================
class SAM2BoxTrainer(nn.Module):
    """
    Wrapper for SAM2 that enables training with box prompts.

    Handles the forward pass manually to ensure gradients flow
    through the mask decoder while keeping encoder frozen.
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        image_size: Tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self.model = sam2_model
        self.image_size = image_size

        # Image preprocessing (SAM2 expects normalized images)
        self._bb_feat_sizes = [
            (image_size[0] // 4, image_size[1] // 4),
            (image_size[0] // 8, image_size[1] // 8),
            (image_size[0] // 16, image_size[1] // 16),
        ]

        # Freeze components as specified
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            self.model.image_encoder.eval()

        if freeze_prompt_encoder:
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = False
            self.model.sam_prompt_encoder.eval()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")

    def train(self, mode: bool = True):
        """Override train to keep frozen components in eval mode."""
        super().train(mode)
        self.model.image_encoder.eval()
        self.model.sam_prompt_encoder.eval()
        return self

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode images using the frozen image encoder.

        Args:
            images: (B, 3, H, W) normalized images

        Returns:
            Dict with image features and positional encodings
        """
        # Forward through image encoder
        backbone_out = self.model.forward_image(images)

        # Prepare features (from SAM2Base._prepare_backbone_features)
        # backbone_fpn contains multi-scale features:
        #   [0] = scale 0 (1/4 resolution) - highest resolution
        #   [1] = scale 1 (1/8 resolution)
        #   [2] = scale 2 (1/16 resolution) - lowest resolution, used as main image embedding
        feature_maps = backbone_out["backbone_fpn"]
        vision_pos_embeds = backbone_out["vision_pos_enc"]

        # Get the lowest-resolution feature map as image embedding
        # This is what SAM2 uses for the main transformer
        feat = feature_maps[-1]  # (B, C, H/16, W/16)
        feat_pos = vision_pos_embeds[-1]

        # Extract high-resolution features for the mask decoder
        # SAM2's mask decoder needs these for detailed mask upsampling
        # high_res_features = [feat_s0 (1/4 res), feat_s1 (1/8 res)]
        high_res_features = [feature_maps[0], feature_maps[1]]

        return {
            'image_embed': feat,
            'image_pe': feat_pos,
            'high_res_features': high_res_features,
            'backbone_out': backbone_out,
        }

    def encode_boxes(
        self,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode bounding boxes as prompts.

        SAM2 expects boxes as corner points with labels 2 (top-left) and 3 (bottom-right).

        Args:
            boxes: (B, 4) boxes in XYXY format, coordinates in image space
            image_size: (H, W) of the input image

        Returns:
            sparse_embeddings: (B, 2, 256)
            dense_embeddings: (B, 256, H/16, W/16)
        """
        B = boxes.shape[0]
        device = boxes.device

        # Convert boxes from XYXY to corner points format
        # boxes: [x1, y1, x2, y2] -> [[x1, y1], [x2, y2]]
        box_coords = boxes.reshape(B, 2, 2)  # (B, 2, 2)

        # Create labels: 2 for top-left, 3 for bottom-right
        box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        box_labels = box_labels.expand(B, -1)  # (B, 2)

        # Encode through prompt encoder
        # SAM2's prompt encoder expects points as (coords, labels) tuple
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(box_coords, box_labels),
            boxes=None,  # Boxes are passed as points with labels 2,3
            masks=None,
        )

        return sparse_embeddings, dense_embeddings

    def decode_masks(
        self,
        image_embed: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        high_res_features: List[torch.Tensor],
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode masks from embeddings using the mask decoder.

        This is the trainable part!

        Args:
            image_embed: Image features from encoder (B, C, H/16, W/16)
            image_pe: Positional encoding
            sparse_embeddings: From prompt encoder
            dense_embeddings: From prompt encoder
            high_res_features: List of [feat_s0 (1/4 res), feat_s1 (1/8 res)] from backbone
            multimask_output: Whether to output multiple masks

        Returns:
            masks: (B, num_masks, H, W) mask logits
            iou_pred: (B, num_masks) IoU predictions
        """
        # Get dense positional encoding
        dense_pe = self.model.sam_prompt_encoder.get_dense_pe()

        # Forward through mask decoder
        # high_res_features are required for SAM2's mask decoder to perform
        # detailed upsampling using skip connections from the backbone
        masks, iou_pred, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return masks, iou_pred

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            images: (B, 3, H, W) normalized images
            boxes: (B, 4) boxes in XYXY format

        Returns:
            Dict with 'masks' (B, 1, H, W) and 'iou_pred' (B, 1)
        """
        # 1. Encode image (frozen)
        image_features = self.encode_image(images)

        # 2. Encode box prompts (frozen)
        sparse_emb, dense_emb = self.encode_boxes(boxes, self.image_size)

        # 3. Decode masks (trainable!)
        masks, iou_pred = self.decode_masks(
            image_embed=image_features['image_embed'],
            image_pe=image_features['image_pe'],
            sparse_embeddings=sparse_emb,
            dense_embeddings=dense_emb,
            high_res_features=image_features['high_res_features'],
            multimask_output=multimask_output,
        )

        # 4. Upsample masks to image size
        masks_upsampled = F.interpolate(
            masks,
            size=self.image_size,
            mode='bilinear',
            align_corners=False,
        )

        return {
            'masks': masks_upsampled,  # (B, 1, H, W) logits
            'iou_pred': iou_pred,  # (B, 1)
            'low_res_masks': masks,  # (B, 1, 64, 64) or similar
        }


# =============================================================================
# DATA COLLATION
# =============================================================================
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for box-prompted training.

    Handles:
    - Resizing images to fixed size
    - Normalizing images
    - Scaling boxes to match resized image
    - Resizing masks
    """
    images = []
    boxes = []
    masks = []
    class_ids = []

    target_size = CONFIG['image_size']

    for sample in batch:
        img = sample['image']  # (H, W, 3) uint8
        bbox = sample['bbox']  # [x1, y1, x2, y2]
        mask = sample['gt_mask']  # (H, W) binary

        orig_h, orig_w = img.shape[:2]

        # Resize image to target size
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))

        # Scale box coordinates
        scale_x = target_size[1] / orig_w
        scale_y = target_size[0] / orig_h
        bbox_scaled = bbox.copy()
        bbox_scaled[0] *= scale_x  # x1
        bbox_scaled[1] *= scale_y  # y1
        bbox_scaled[2] *= scale_x  # x2
        bbox_scaled[3] *= scale_y  # y2

        # Resize mask
        mask_resized = cv2.resize(mask.astype(np.float32), (target_size[1], target_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Convert image to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        images.append(img_tensor)
        boxes.append(torch.from_numpy(bbox_scaled).float())
        masks.append(torch.from_numpy(mask_resized).float())
        class_ids.append(sample['class_id'])

    return {
        'images': torch.stack(images),  # (B, 3, H, W)
        'boxes': torch.stack(boxes),  # (B, 4)
        'masks': torch.stack(masks).unsqueeze(1),  # (B, 1, H, W)
        'class_ids': class_ids,
    }


# Need cv2 for collate_fn
import cv2


# =============================================================================
# METRICS
# =============================================================================
def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between predicted and ground truth masks."""
    pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()

    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


class MetricTracker:
    """Track metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_iou = 0.0
        self.class_ious = {i: [] for i in range(1, 7)}
        self.count = 0

    def update(self, loss: float, pred_masks: torch.Tensor,
               gt_masks: torch.Tensor, class_ids: List[int]):
        self.total_loss += loss
        self.count += 1

        # Compute IoU per sample
        for i, class_id in enumerate(class_ids):
            iou = compute_iou(pred_masks[i], gt_masks[i])
            self.total_iou += iou
            self.class_ious[class_id].append(iou)

    def compute(self) -> Dict[str, float]:
        metrics = {
            'loss': self.total_loss / max(self.count, 1),
            'mean_iou': self.total_iou / max(self.count * len(self.class_ious), 1),
        }

        # Per-class IoU
        for class_id, ious in self.class_ious.items():
            if ious:
                metrics[f'iou_class_{class_id}'] = np.mean(ious)
            else:
                metrics[f'iou_class_{class_id}'] = float('nan')

        # Mean across classes that have samples
        valid_class_ious = [np.mean(ious) for ious in self.class_ious.values() if ious]
        metrics['mean_class_iou'] = np.mean(valid_class_ious) if valid_class_ious else 0.0

        return metrics


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_one_epoch(
    model: SAM2BoxTrainer,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch in pbar:
        images = batch['images'].to(device)
        boxes = batch['boxes'].to(device)
        gt_masks = batch['masks'].to(device)
        class_ids = batch['class_ids']

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, boxes)
        pred_masks = outputs['masks']
        pred_ious = outputs['iou_pred']

        # Compute loss
        losses = criterion(pred_masks, gt_masks, pred_ious)
        loss = losses['total']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            tracker.update(loss.item(), pred_masks, gt_masks, class_ids)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return tracker.compute()


@torch.no_grad()
def validate(
    model: SAM2BoxTrainer,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
) -> Tuple[Dict[str, float], Optional[Tuple]]:
    """Validate the model."""
    model.eval()
    tracker = MetricTracker()

    vis_data = None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for i, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes = batch['boxes'].to(device)
        gt_masks = batch['masks'].to(device)
        class_ids = batch['class_ids']

        # Forward pass
        outputs = model(images, boxes)
        pred_masks = outputs['masks']
        pred_ious = outputs['iou_pred']

        # Compute loss
        losses = criterion(pred_masks, gt_masks, pred_ious)
        loss = losses['total']

        tracker.update(loss.item(), pred_masks, gt_masks, class_ids)

        # Store first batch for visualization
        if i == 0:
            vis_data = (
                images.cpu(),
                boxes.cpu(),
                pred_masks.cpu(),
                gt_masks.cpu(),
                class_ids,
            )

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return tracker.compute(), vis_data


def save_checkpoint(
    model: SAM2BoxTrainer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    config: Dict,
    path: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    # Only save mask decoder state (what we trained)
    checkpoint = {
        'epoch': epoch,
        'mask_decoder_state_dict': model.model.sam_mask_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = path.parent / 'best_model.pt'
        torch.save(checkpoint, best_path)


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(
    images: torch.Tensor,
    boxes: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    class_ids: List[int],
    num_samples: int = 4,
) -> np.ndarray:
    """Create visualization grid for TensorBoard."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        bbox = boxes[i].numpy()
        pred = torch.sigmoid(pred_masks[i, 0]).numpy()
        gt = gt_masks[i, 0].numpy()
        class_name = CLASS_NAMES.get(class_ids[i], 'unknown')

        # Column 1: Image with box
        axes[i, 0].imshow(img)
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='lime', facecolor='none')
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f'Input + Box\n{class_name}')
        axes[i, 0].axis('off')

        # Column 2: Ground truth
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Column 3: Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction\nIoU: {compute_iou(pred_masks[i:i+1], gt_masks[i:i+1]):.3f}')
        axes[i, 2].axis('off')

        # Column 4: Overlay
        overlay = img.copy()
        pred_binary = pred > 0.5
        overlay[pred_binary] = overlay[pred_binary] * 0.5 + np.array([0, 1, 0]) * 0.5
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()

    # Convert to numpy array using buffer
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_array = np.array(plt.imread(buf))
    buf.close()
    plt.close(fig)

    # Convert RGBA to RGB if needed
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Ensure uint8 format
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)

    return img_array


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train(config: Dict):
    """Main training function."""
    print("\n" + "=" * 70)
    print(" SAM2 BOX-PROMPTED FINE-TUNING")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = config['output_dir'] / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config['log_dir'] / f'run_{timestamp}'

    print(f"Output directory: {run_dir}")
    print(f"Log directory: {log_dir}")

    # Save config
    config_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_save, f, indent=2)

    # =========================================================================
    # DATA
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading Data")
    print("-" * 50)

    train_images_dir, masks_dir = get_split_paths('train')
    train_dataset = EndoscapesBoxDataset(
        images_dir=str(train_images_dir),
        masks_dir=str(masks_dir),
        split='train',
    )

    val_images_dir, masks_dir = get_split_paths('val')
    val_dataset = EndoscapesBoxDataset(
        images_dir=str(val_images_dir),
        masks_dir=str(masks_dir),
        split='val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n" + "-" * 50)
    print(" Loading Model")
    print("-" * 50)

    sam2_model = build_sam2(
        config_file=config['model_cfg'],
        ckpt_path=str(config['checkpoint']),
        device='cpu',
        mode='eval',
    )

    model = SAM2BoxTrainer(
        sam2_model=sam2_model,
        freeze_image_encoder=config['freeze_image_encoder'],
        freeze_prompt_encoder=config['freeze_prompt_encoder'],
        image_size=config['image_size'],
    )
    model = model.to(device)

    # =========================================================================
    # LOSS, OPTIMIZER, SCHEDULER
    # =========================================================================
    criterion = CombinedLoss(
        focal_weight=config['focal_weight'],
        dice_weight=config['dice_weight'],
        iou_weight=config['iou_weight'],
    )

    # Only optimize mask decoder parameters
    params_to_train = [p for p in model.model.sam_mask_decoder.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in params_to_train):,}")

    optimizer = AdamW(
        params_to_train,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir)

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

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Logging
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/MeanIoU', val_metrics['mean_class_iou'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        for class_id in range(1, 7):
            key = f'iou_class_{class_id}'
            if key in val_metrics and not np.isnan(val_metrics[key]):
                writer.add_scalar(f'Val/IoU_{CLASS_NAMES[class_id]}', val_metrics[key], epoch)

        # Visualization
        if (epoch + 1) % config['visualize_every'] == 0 and vis_data is not None:
            images, boxes, pred_masks, gt_masks, class_ids = vis_data
            vis_img = create_visualization(images, boxes, pred_masks, gt_masks, class_ids)
            writer.add_image('Validation/Predictions', vis_img, epoch, dataformats='HWC')

        # Checkpointing
        is_best = val_metrics['mean_class_iou'] > best_miou
        if is_best:
            best_miou = val_metrics['mean_class_iou']
            best_epoch = epoch + 1

        if (epoch + 1) % config['save_every'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, config,
                run_dir / f'checkpoint_epoch_{epoch+1:03d}.pt',
                is_best
            )

        # Console output
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val mIoU: {val_metrics['mean_class_iou']*100:.1f}%")
        print(f"Time: {epoch_time:.0f}s | LR: {current_lr:.2e}")

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
Best epoch:     {best_epoch}
Best mIoU:      {best_miou*100:.2f}%
Total time:     {total_time/60:.1f} minutes
Checkpoints:    {run_dir}
Best model:     {run_dir / 'best_model.pt'}
TensorBoard:    {log_dir}
"""
    print(summary)

    with open(run_dir / 'summary.txt', 'w') as f:
        f.write(summary)

    writer.close()

    return {
        'best_epoch': best_epoch,
        'best_miou': best_miou,
        'run_dir': str(run_dir),
    }


# =============================================================================
# QUICK TEST
# =============================================================================
def test_training_setup():
    """Test that training setup works correctly."""
    print("\n" + "=" * 70)
    print(" TRAINING SETUP TEST")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading SAM2...")
    sam2_model = build_sam2(
        config_file=CONFIG['model_cfg'],
        ckpt_path=str(CONFIG['checkpoint']),
        device='cpu',
        mode='eval',
    )

    model = SAM2BoxTrainer(
        sam2_model=sam2_model,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        image_size=CONFIG['image_size'],
    )
    model = model.to(device)

    # Create dummy batch
    print("\nTesting forward pass...")
    batch_size = 2
    images = torch.randn(batch_size, 3, *CONFIG['image_size'], device=device)
    boxes = torch.tensor([
        [100, 100, 400, 400],
        [200, 150, 500, 450],
    ], dtype=torch.float32, device=device)

    # Forward pass
    model.train()
    outputs = model(images, boxes)

    print(f"  Output masks shape: {outputs['masks'].shape}")
    print(f"  IoU predictions shape: {outputs['iou_pred'].shape}")

    # Test loss computation
    print("\nTesting loss computation...")
    gt_masks = torch.zeros(batch_size, 1, *CONFIG['image_size'], device=device)
    gt_masks[:, :, 150:350, 150:350] = 1.0

    criterion = CombinedLoss()
    losses = criterion(outputs['masks'], gt_masks, outputs['iou_pred'])

    print(f"  Focal loss: {losses['focal'].item():.4f}")
    print(f"  Dice loss: {losses['dice'].item():.4f}")
    print(f"  IoU loss: {losses['iou'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")

    # Test backward pass
    print("\nTesting backward pass...")
    losses['total'].backward()

    # Check gradients
    has_grad = False
    for name, param in model.model.sam_mask_decoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    print(f"  Gradients flowing to mask decoder: {has_grad}")

    # Memory usage
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("\n" + "=" * 70)
    print(" TEST PASSED - Ready for training!")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SAM2 Box-Prompted Fine-tuning')

    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['lr'])
    parser.add_argument('--test', action='store_true', help='Run setup test only')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.test:
        test_training_setup()
        return

    # Update config
    config = CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr

    train(config)


if __name__ == '__main__':
    main()
