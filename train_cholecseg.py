"""
SAM2 Fine-tuning on CholecSeg8k for Gallbladder + Liver Segmentation
=====================================================================
Freezes the Hiera-L image encoder, inserts LoRA adapters into attention layers,
trains the mask decoder with point prompts from GT mask centroids.

After training: generates gallbladder + liver masks for all Endoscapes keyframes.
"""

import sys
import os
import time
import random
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from PIL import Image
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Add SAM2 to path
SAM2_DIR = Path("C:/Users/sufia/Documents/Uni/Masters/DISSERTATION/sam2_finetune/sam2")
sys.path.insert(0, str(SAM2_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/sufia/Documents/Uni/Masters/DISSERTATION")
CHOLECSEG_DIR = BASE_DIR / "CholecSeg8k"
ENDOSCAPES_DIR = BASE_DIR / "endoscapes"
SAM2_CKPT = BASE_DIR / "sam2_finetune" / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = BASE_DIR / "sam2_finetune" / "cholecseg_finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CholecSeg8k class mapping ───────────────────────────────────────────────
# Gallbladder: class 21 (wall) + class 22 (body) -> merged
# Liver: class 11
GALLBLADDER_CLASSES = [21, 22]
LIVER_CLASS = 11
TARGET_CLASSES = {"gallbladder": GALLBLADDER_CLASSES, "liver": [LIVER_CLASS]}

# ── Train/Val/Test split (by video) ─────────────────────────────────────────
TRAIN_VIDEOS = [1, 9, 12, 17, 18, 20, 24, 25, 26, 27, 28, 35, 43]
VAL_VIDEOS = [37, 48]
TEST_VIDEOS = [52, 55]

# ── Training config ──────────────────────────────────────────────────────────
BATCH_SIZE = 2
NUM_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 2
LR_DECODER = 1e-4
LR_LORA = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05
LORA_RANK = 8
LORA_ALPHA = 16
MIN_MASK_AREA = 100
SAM2_SIZE = 1024  # SAM2 native resolution


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear. Freezes original weights, adds low-rank AB."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return out + lora_out


def insert_lora_adapters(model, rank=LORA_RANK, alpha=LORA_ALPHA):
    """Insert LoRA into qkv of Hiera attention layers."""
    lora_count = 0
    # Determine device from existing model params
    device = next(model.parameters()).device
    for block in model.image_encoder.trunk.blocks:
        if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            old = block.attn.qkv
            lora_layer = LoRALinear(old, rank=rank, alpha=alpha).to(device)
            block.attn.qkv = lora_layer
            lora_count += 1
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if 'lora_' in n and p.requires_grad
    )
    print(f"  Inserted LoRA into {lora_count} attention layers")
    print(f"  LoRA parameters: {lora_params:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).flatten(1)
        target = target.flatten(1)
        inter = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=1.0, focal_w=20.0):
        super().__init__()
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, pred, target):
        d = self.dice(pred, target)
        f = self.focal(pred, target)
        return self.dice_w * d + self.focal_w * f, {"dice": d.item(), "focal": f.item()}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CholecSeg8kDataset(Dataset):
    """CholecSeg8k dataset for SAM2 fine-tuning on gallbladder + liver."""

    def __init__(self, video_ids, augment=False):
        self.augment = augment
        self.samples = []

        for vid in video_ids:
            vid_dir = CHOLECSEG_DIR / f"video{vid:02d}"
            if not vid_dir.exists():
                continue
            for clip_dir in sorted(vid_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                for mask_path in sorted(clip_dir.glob("*_endo_mask.png")):
                    img_name = mask_path.name.replace("_endo_mask.png", "_endo.png")
                    img_path = clip_dir / img_name
                    if img_path.exists():
                        self.samples.append({
                            "image_path": str(img_path),
                            "mask_path": str(mask_path),
                        })

    def __len__(self):
        return len(self.samples)

    def _get_centroid_point(self, binary_mask):
        """Get centroid of mask. If centroid is outside mask, find nearest inside point."""
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0:
            return None

        cx, cy = int(xs.mean()), int(ys.mean())

        # If centroid is inside mask, use it
        if binary_mask[cy, cx] > 0:
            return np.array([cx, cy])

        # Otherwise find nearest point inside mask
        dists = (xs - cx) ** 2 + (ys - cy) ** 2
        nearest_idx = np.argmin(dists)
        return np.array([xs[nearest_idx], ys[nearest_idx]])

    def _sample_extra_points(self, binary_mask, n=2):
        """Sample N random foreground points for prompt augmentation."""
        ys, xs = np.where(binary_mask > 0)
        if len(xs) < n:
            return np.column_stack([xs, ys]) if len(xs) > 0 else np.zeros((0, 2))
        indices = np.random.choice(len(xs), n, replace=False)
        return np.column_stack([xs[indices], ys[indices]])

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and mask
        img = cv2.imread(sample["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_raw = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        orig_h, orig_w = img.shape[:2]

        # Extract binary masks for each target class
        class_masks = {}
        class_points = {}

        for cls_name, cls_ids in TARGET_CLASSES.items():
            binary_mask = np.zeros_like(mask_raw, dtype=np.uint8)
            for cid in cls_ids:
                binary_mask[mask_raw == cid] = 1

            if binary_mask.sum() < MIN_MASK_AREA:
                continue

            # Get centroid point
            centroid = self._get_centroid_point(binary_mask)
            if centroid is None:
                continue

            # During training, add extra random points
            if self.augment:
                extras = self._sample_extra_points(binary_mask, n=random.randint(0, 2))
                all_points = np.vstack([centroid.reshape(1, 2), extras]) if len(extras) > 0 else centroid.reshape(1, 2)
            else:
                all_points = centroid.reshape(1, 2)

            class_masks[cls_name] = binary_mask
            class_points[cls_name] = all_points

        if len(class_masks) == 0:
            # No valid masks — return dummy
            return self._dummy(img, orig_h, orig_w)

        # Simple augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            for cls_name in class_masks:
                class_masks[cls_name] = class_masks[cls_name][:, ::-1].copy()
                pts = class_points[cls_name].copy()
                pts[:, 0] = orig_w - 1 - pts[:, 0]
                class_points[cls_name] = pts

        # Resize to SAM2 input size
        img_resized = cv2.resize(img, (SAM2_SIZE, SAM2_SIZE), interpolation=cv2.INTER_LINEAR)
        scale_x = SAM2_SIZE / orig_w
        scale_y = SAM2_SIZE / orig_h

        masks_out = []
        points_out = []
        labels_out = []
        cls_names_out = []

        for cls_name in class_masks:
            m = cv2.resize(class_masks[cls_name], (SAM2_SIZE, SAM2_SIZE),
                           interpolation=cv2.INTER_NEAREST)
            masks_out.append(m)

            pts = class_points[cls_name].astype(np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            points_out.append(pts)
            labels_out.append(np.ones(len(pts), dtype=np.int64))
            cls_names_out.append(cls_name)

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        masks_tensor = torch.stack([torch.from_numpy(m).float() for m in masks_out])

        return {
            "image": img_tensor,
            "masks": masks_tensor,
            "points": points_out,
            "point_labels": labels_out,
            "class_names": cls_names_out,
            "num_masks": len(masks_out),
            "image_path": sample["image_path"],
        }

    def _dummy(self, img, h, w):
        img_resized = cv2.resize(img, (SAM2_SIZE, SAM2_SIZE))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        return {
            "image": img_tensor,
            "masks": torch.zeros(1, SAM2_SIZE, SAM2_SIZE),
            "points": [np.array([[SAM2_SIZE // 2, SAM2_SIZE // 2]], dtype=np.float32)],
            "point_labels": [np.array([1], dtype=np.int64)],
            "class_names": ["dummy"],
            "num_masks": 0,
            "image_path": "",
        }


def collate_fn(batch):
    valid = [b for b in batch if b["num_masks"] > 0]
    if not valid:
        return None
    return {
        "images": torch.stack([b["image"] for b in valid]),
        "masks": [b["masks"] for b in valid],
        "points": [b["points"] for b in valid],
        "point_labels": [b["point_labels"] for b in valid],
        "class_names": [b["class_names"] for b in valid],
        "num_masks": [b["num_masks"] for b in valid],
        "image_paths": [b["image_path"] for b in valid],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(pred_logits, gt_mask, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - inter
    return (inter / union).item() if union > 0 else (1.0 if gt_mask.sum() == 0 else 0.0)


def forward_one_mask(model, image_tensor, point_coords, point_labels):
    """Forward pass through SAM2 model for one image+prompt, with gradients.

    Args:
        model: SAM2Base model
        image_tensor: (3, H, W) float tensor, already normalized to [0,1]
        point_coords: (P, 2) numpy array of (x,y) coords in input image space
        point_labels: (P,) numpy array of labels

    Returns:
        low_res_masks: (1, 1, H/4, W/4) logits tensor (with grad)
    """
    # 1) Image encoding (frozen encoder, but LoRA needs grad)
    img_batch = image_tensor.unsqueeze(0)  # (1, 3, H, W)
    backbone_out = model.forward_image(img_batch)

    # 2) Prepare backbone features
    # Get the final feature map and high-res features
    fpn_feats = backbone_out["backbone_fpn"]
    vision_pos = backbone_out["vision_pos_enc"]

    # backbone_features is the last FPN level
    backbone_features = fpn_feats[-1]  # (1, C, H', W')

    # high_res_features for SAM decoder
    high_res_features = [fpn_feats[0], fpn_feats[1]]

    # 3) Prepare point prompts as tensors
    pt_coords = torch.tensor(point_coords, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)  # (1, P, 2)
    pt_labels = torch.tensor(point_labels, dtype=torch.int32,
                             device=DEVICE).unsqueeze(0)  # (1, P)
    point_inputs = {"point_coords": pt_coords, "point_labels": pt_labels}

    # 4) Forward through SAM heads (prompt encoder + mask decoder)
    (low_res_multimasks, high_res_multimasks, ious,
     low_res_masks, high_res_masks, obj_ptr, obj_scores) = model._forward_sam_heads(
        backbone_features=backbone_features,
        point_inputs=point_inputs,
        high_res_features=high_res_features,
        multimask_output=False,
    )

    return high_res_masks  # (1, 1, H, W) at full resolution


def train_epoch(model, predictor, train_loader, criterion, optimizer, scheduler,
                scaler, epoch):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_masks = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if batch is None:
            continue

        images = batch["images"].to(DEVICE)
        optimizer.zero_grad()

        batch_loss = 0.0
        for img_idx in range(len(images)):
            image = images[img_idx]
            gt_masks = batch["masks"][img_idx].to(DEVICE)
            points_list = batch["points"][img_idx]
            labels_list = batch["point_labels"][img_idx]

            for mask_idx in range(len(gt_masks)):
                gt_mask = gt_masks[mask_idx].unsqueeze(0)  # (1, H, W)
                pts = points_list[mask_idx]
                lbls = labels_list[mask_idx]

                with autocast("cuda", dtype=torch.float16):
                    pred_masks = forward_one_mask(model, image, pts, lbls)
                    # pred_masks: (1, 1, H, W) logits
                    pred_logits = pred_masks.squeeze(1)  # (1, H, W)

                    # Resize to match GT if needed
                    if pred_logits.shape[1:] != gt_mask.shape[1:]:
                        pred_logits = F.interpolate(
                            pred_logits.unsqueeze(0),
                            size=gt_mask.shape[1:], mode="bilinear", align_corners=False
                        ).squeeze(0)

                    loss, _ = criterion(pred_logits, gt_mask)

                scaler.scale(loss).backward()
                batch_loss += loss.item()
                total_iou += compute_iou(pred_logits.detach(), gt_mask)
                n_masks += 1

        if n_masks > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += batch_loss
        if n_masks > 0:
            pbar.set_postfix(loss=f"{total_loss/n_masks:.4f}",
                             iou=f"{total_iou/n_masks:.4f}")

    return {
        "loss": total_loss / max(n_masks, 1),
        "iou": total_iou / max(n_masks, 1),
    }


@torch.no_grad()
def validate(model, predictor, val_loader, criterion):
    model.eval()
    class_ious = {"gallbladder": [], "liver": []}
    total_loss = 0.0
    n_masks = 0

    for batch in tqdm(val_loader, desc="Validating"):
        if batch is None:
            continue

        images = batch["images"].to(DEVICE)
        for img_idx in range(len(images)):
            image = images[img_idx]
            gt_masks = batch["masks"][img_idx].to(DEVICE)
            points_list = batch["points"][img_idx]
            labels_list = batch["point_labels"][img_idx]
            cls_names = batch["class_names"][img_idx]

            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            predictor.set_image(img_np)

            for mask_idx in range(len(gt_masks)):
                gt_mask = gt_masks[mask_idx].unsqueeze(0)
                pts = points_list[mask_idx]
                lbls = labels_list[mask_idx]
                cls_name = cls_names[mask_idx]

                _, _, logits = predictor.predict(
                    point_coords=pts, point_labels=lbls, multimask_output=False)

                pred_logits = torch.from_numpy(logits[0]).to(DEVICE)
                if pred_logits.shape != gt_mask.shape[1:]:
                    pred_logits = F.interpolate(
                        pred_logits.unsqueeze(0).unsqueeze(0),
                        size=gt_mask.shape[1:], mode="bilinear", align_corners=False
                    ).squeeze(0).squeeze(0)
                pred_logits = pred_logits.unsqueeze(0)

                loss, _ = criterion(pred_logits, gt_mask)
                total_loss += loss.item()

                iou = compute_iou(pred_logits, gt_mask)
                if cls_name in class_ious:
                    class_ious[cls_name].append(iou)
                n_masks += 1

    gb_iou = np.mean(class_ious["gallbladder"]) if class_ious["gallbladder"] else 0
    liver_iou = np.mean(class_ious["liver"]) if class_ious["liver"] else 0
    mean_iou = np.mean([gb_iou, liver_iou]) if (gb_iou > 0 or liver_iou > 0) else 0

    return {
        "loss": total_loss / max(n_masks, 1),
        "gb_iou": gb_iou,
        "liver_iou": liver_iou,
        "mean_iou": mean_iou,
        "n_gb": len(class_ious["gallbladder"]),
        "n_liver": len(class_ious["liver"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ON ENDOSCAPES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_endoscapes_masks(model, predictor):
    """Generate gallbladder + liver masks for all Endoscapes keyframes using grid prompts."""
    print("\n" + "=" * 70)
    print("GENERATING ENDOSCAPES MASKS")
    print("=" * 70)

    model.eval()
    mask_dir = OUTPUT_DIR / "endoscapes_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(ENDOSCAPES_DIR / "all_metadata.csv")
    keyframes = metadata[metadata["is_ds_keyframe"] == True]
    print(f"  Total keyframes: {len(keyframes)}")

    # Grid prompts: 5x3 grid covering the frame
    grid_x = np.linspace(0.1, 0.9, 5) * SAM2_SIZE
    grid_y = np.linspace(0.1, 0.9, 3) * SAM2_SIZE
    grid_points = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float32)

    stats = {"gallbladder": 0, "liver": 0, "total": 0}

    for _, row in tqdm(keyframes.iterrows(), total=len(keyframes), desc="Generating masks"):
        vid = int(row["vid"])
        frame = int(row["frame"])
        fname = f"{vid}_{frame}"

        # Determine split
        if vid <= 120:
            split = "train"
        elif vid <= 161:
            split = "val"
        else:
            split = "test"

        img_path = ENDOSCAPES_DIR / split / f"{fname}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Resize to SAM2 size for prompt coords
        img_resized = cv2.resize(img, (SAM2_SIZE, SAM2_SIZE))
        predictor.set_image(img_resized)

        stats["total"] += 1

        for cls_name in ["gallbladder", "liver"]:
            # Use grid prompts — SAM2 picks the best mask
            # Try centre prompt first (most likely to hit the target)
            if cls_name == "gallbladder":
                # GB typically centre-right of frame
                prompt_pts = np.array([[SAM2_SIZE * 0.55, SAM2_SIZE * 0.45]], dtype=np.float32)
            else:
                # Liver typically upper-centre
                prompt_pts = np.array([[SAM2_SIZE * 0.5, SAM2_SIZE * 0.3]], dtype=np.float32)

            prompt_labels = np.array([1], dtype=np.int64)

            masks, scores, _ = predictor.predict(
                point_coords=prompt_pts,
                point_labels=prompt_labels,
                multimask_output=True,
            )

            # Take the mask with highest score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

            # Resize back to original resolution
            best_mask = cv2.resize(best_mask.astype(np.uint8),
                                   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            if best_mask.sum() > MIN_MASK_AREA:
                save_path = mask_dir / f"{fname}_{cls_name}.png"
                cv2.imwrite(str(save_path), best_mask * 255)
                stats[cls_name] += 1

    print(f"\n  Generated masks:")
    print(f"    Gallbladder: {stats['gallbladder']}/{stats['total']} frames ({stats['gallbladder']/max(stats['total'],1)*100:.1f}%)")
    print(f"    Liver:       {stats['liver']}/{stats['total']} frames ({stats['liver']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Saved to: {mask_dir}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION ON ENDOSCAPES GT
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_on_endoscapes_gt(model, predictor):
    """Validate generated masks against Endoscapes GT segmentation."""
    print("\n" + "=" * 70)
    print("VALIDATING ON ENDOSCAPES GT MASKS")
    print("=" * 70)

    model.eval()
    mask_dir = OUTPUT_DIR / "endoscapes_masks"

    # Find Endoscapes frames with GT segmentation
    gt_mask_dir = ENDOSCAPES_DIR / "train" / "semantic"
    if not gt_mask_dir.exists():
        # Try alternative paths
        for candidate in [
            ENDOSCAPES_DIR / "synthetic_masks",
            ENDOSCAPES_DIR / "semseg",
        ]:
            if candidate.exists():
                gt_mask_dir = candidate
                break

    if not gt_mask_dir.exists():
        print("  No GT segmentation masks found. Skipping validation.")
        return {}

    # Endoscapes class mapping: gallbladder = 5, cystic_plate = 1
    ENDO_GB_CLASS = 5
    ENDO_LIVER_PROXY = 1  # cystic_plate is closest proxy for liver

    gt_files = list(gt_mask_dir.glob("*.png"))
    print(f"  Found {len(gt_files)} GT mask files in {gt_mask_dir}")

    ious = {"gallbladder": [], "liver": []}

    for gt_path in tqdm(gt_files, desc="Validating on GT"):
        fname = gt_path.stem  # e.g., "1_29850"
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # Check gallbladder
        gt_gb = (gt_mask == ENDO_GB_CLASS).astype(np.uint8)
        pred_gb_path = mask_dir / f"{fname}_gallbladder.png"
        if pred_gb_path.exists() and gt_gb.sum() > 0:
            pred_gb = cv2.imread(str(pred_gb_path), cv2.IMREAD_GRAYSCALE)
            pred_gb = (pred_gb > 127).astype(np.uint8)
            if pred_gb.shape != gt_gb.shape:
                pred_gb = cv2.resize(pred_gb, (gt_gb.shape[1], gt_gb.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            inter = (pred_gb & gt_gb).sum()
            union = (pred_gb | gt_gb).sum()
            iou = inter / union if union > 0 else 0
            ious["gallbladder"].append(iou)

        # Check liver (using cystic_plate as proxy)
        gt_liver = (gt_mask == ENDO_LIVER_PROXY).astype(np.uint8)
        pred_liver_path = mask_dir / f"{fname}_liver.png"
        if pred_liver_path.exists() and gt_liver.sum() > 0:
            pred_liver = cv2.imread(str(pred_liver_path), cv2.IMREAD_GRAYSCALE)
            pred_liver = (pred_liver > 127).astype(np.uint8)
            if pred_liver.shape != gt_liver.shape:
                pred_liver = cv2.resize(pred_liver, (gt_liver.shape[1], gt_liver.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            inter = (pred_liver & gt_liver).sum()
            union = (pred_liver | gt_liver).sum()
            iou = inter / union if union > 0 else 0
            ious["liver"].append(iou)

    print(f"\n  Gallbladder IoU: {np.mean(ious['gallbladder'])*100:.1f}% "
          f"(median {np.median(ious['gallbladder'])*100:.1f}%, n={len(ious['gallbladder'])})")
    print(f"  Liver IoU:       {np.mean(ious['liver'])*100:.1f}% "
          f"(median {np.median(ious['liver'])*100:.1f}%, n={len(ious['liver'])})")

    return ious


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("=" * 70)
    print("SAM2 Fine-tuning on CholecSeg8k (Gallbladder + Liver)")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = CholecSeg8kDataset(TRAIN_VIDEOS, augment=True)
    val_ds = CholecSeg8kDataset(VAL_VIDEOS, augment=False)
    test_ds = CholecSeg8kDataset(TEST_VIDEOS, augment=False)

    print(f"  Train: {len(train_ds)} frames from videos {TRAIN_VIDEOS}")
    print(f"  Val:   {len(val_ds)} frames from videos {VAL_VIDEOS}")
    print(f"  Test:  {len(test_ds)} frames from videos {TEST_VIDEOS}")

    # Class stats
    for name, ds, vids in [("Train", train_ds, TRAIN_VIDEOS),
                            ("Val", val_ds, VAL_VIDEOS),
                            ("Test", test_ds, TEST_VIDEOS)]:
        sample_count = min(len(ds), 200)
        gb_count = 0
        liver_count = 0
        for i in range(sample_count):
            s = ds[i]
            if "gallbladder" in s["class_names"]:
                gb_count += 1
            if "liver" in s["class_names"]:
                liver_count += 1
        print(f"  {name} (sampled {sample_count}): "
              f"GB in {gb_count/sample_count*100:.0f}%, "
              f"Liver in {liver_count/sample_count*100:.0f}%")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn,
                            pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nLoading SAM2 model...")
    model = build_sam2(
        config_file=SAM2_CONFIG,
        ckpt_path=str(SAM2_CKPT),
        device=DEVICE,
    )

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Insert LoRA adapters into encoder attention
    model = insert_lora_adapters(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    # Unfreeze mask decoder
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}")
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")

    # Create predictor
    predictor = SAM2ImagePredictor(model)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    lora_params = [p for n, p in model.named_parameters()
                   if 'lora_' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if 'mask_decoder' in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA},
        {"params": decoder_params, "lr": LR_DECODER},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_FRAC)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=total_steps - warmup_steps,
                                                         eta_min=1e-7)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                       [warmup, cosine],
                                                       milestones=[warmup_steps])

    criterion = CombinedLoss()
    scaler = GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("# TRAINING")
    print("#" * 70)

    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    ckpt_path = OUTPUT_DIR / "best_model.pt"
    hist_path = OUTPUT_DIR / "training_history.csv"

    # Resume check
    if ckpt_path.exists() and hist_path.exists():
        print("\n  [RESUME] Loading existing checkpoint...")
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        best_miou = state.get("best_miou", 0)
        best_epoch = state.get("epoch", 0)
        history = pd.read_csv(hist_path).to_dict("records")
        print(f"  Loaded: best mIoU {best_miou:.4f} at epoch {best_epoch}")
    else:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train_epoch(model, predictor, train_loader, criterion,
                                        optimizer, scheduler, scaler, epoch)
            val_metrics = validate(model, predictor, val_loader, criterion)

            print(f"  Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                  f"train_iou={train_metrics['iou']:.4f} | "
                  f"val_gb={val_metrics['gb_iou']:.4f} val_liver={val_metrics['liver_iou']:.4f} "
                  f"val_mean={val_metrics['mean_iou']:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_gb_iou": val_metrics["gb_iou"],
                "val_liver_iou": val_metrics["liver_iou"],
                "val_mean_iou": val_metrics["mean_iou"],
            })

            if val_metrics["mean_iou"] > best_miou:
                best_miou = val_metrics["mean_iou"]
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                }, ckpt_path)
                print(f"  >> New best: mIoU={best_miou:.4f} *")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"\n  Best val mIoU: {best_miou:.4f} at epoch {best_epoch}")

        # Load best model
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn,
                             pin_memory=True)
    test_metrics = validate(model, predictor, test_loader, criterion)
    print(f"  Test GB IoU:    {test_metrics['gb_iou']:.4f} (n={test_metrics['n_gb']})")
    print(f"  Test Liver IoU: {test_metrics['liver_iou']:.4f} (n={test_metrics['n_liver']})")
    print(f"  Test Mean IoU:  {test_metrics['mean_iou']:.4f}")

    # ── Generate Endoscapes masks ─────────────────────────────────────────
    gen_stats = generate_endoscapes_masks(model, predictor)

    # ── Validate on Endoscapes GT ─────────────────────────────────────────
    gt_ious = validate_on_endoscapes_gt(model, predictor)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best val mIoU: {best_miou:.4f} (epoch {best_epoch})")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
