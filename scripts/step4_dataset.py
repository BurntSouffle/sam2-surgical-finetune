"""
Step 4: PyTorch Dataset for SAM2 Fine-tuning on Endoscapes
Creates EndoscapesDataset class with proper image/mask handling.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Dataset paths
DATASET_ROOT = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes")
OUTPUT_DIR = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs")

# Class information
CLASS_NAMES = [
    "background",      # 0
    "cystic_plate",    # 1
    "calot_triangle",  # 2
    "cystic_artery",   # 3
    "cystic_duct",     # 4
    "gallbladder",     # 5
    "tool",            # 6
]
NUM_CLASSES = 7
IGNORE_INDEX = 255

# Color map for visualization (RGB)
CLASS_COLORS = [
    [0, 0, 0],         # 0: background - black
    [255, 0, 0],       # 1: cystic_plate - red
    [0, 255, 0],       # 2: calot_triangle - green
    [0, 0, 255],       # 3: cystic_artery - blue
    [255, 255, 0],     # 4: cystic_duct - yellow
    [255, 0, 255],     # 5: gallbladder - magenta
    [0, 255, 255],     # 6: tool - cyan
]

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_split_paths(split: str) -> Tuple[Path, Path]:
    """
    Get image and mask directory paths for a given split.

    The dataset uses text files as symlinks. The actual images are in:
    - train/ for training images
    - val/ for validation images
    - test/ for test images

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        Tuple of (images_dir, masks_dir)
    """
    masks_dir = DATASET_ROOT / "semseg"

    # Use actual image directories (not the symlink directories)
    if split == "train":
        images_dir = DATASET_ROOT / "train"
    elif split == "val":
        images_dir = DATASET_ROOT / "val"
    elif split == "test":
        images_dir = DATASET_ROOT / "test"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

    return images_dir, masks_dir


class EndoscapesDataset(Dataset):
    """
    PyTorch Dataset for Endoscapes semantic segmentation.

    Handles:
    - Automatic matching of images to masks by filename
    - Resizing to target size (512x512 by default)
    - Proper normalization for SAM2
    - Mask label preservation during resize
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        transform=None,
    ):
        """
        Initialize the dataset.

        Args:
            images_dir: Directory containing images (.jpg)
            masks_dir: Directory containing masks (.png)
            target_size: (H, W) to resize images and masks to
            normalize: Whether to apply ImageNet normalization
            transform: Optional additional transforms
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.transform = transform

        # Find matching pairs
        self.pairs = self._find_matching_pairs()

        print(f"[EndoscapesDataset] Initialized:")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Masks dir: {self.masks_dir}")
        print(f"  Target size: {self.target_size}")
        print(f"  Valid pairs found: {len(self.pairs)}")

    def _find_matching_pairs(self) -> List[Tuple[Path, Path]]:
        """Find image-mask pairs with matching filenames."""
        pairs = []

        # Get all mask stems (filename without extension)
        mask_stems = {p.stem: p for p in self.masks_dir.glob("*.png")}

        # Find images that have matching masks
        for img_path in self.images_dir.glob("*.jpg"):
            stem = img_path.stem
            if stem in mask_stems:
                pairs.append((img_path, mask_stems[stem]))

        # Sort for reproducibility
        pairs.sort(key=lambda x: x[0].name)

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess an image-mask pair.

        Returns:
            Dict with:
                - 'image': torch.Tensor of shape (3, H, W), normalized
                - 'mask': torch.Tensor of shape (H, W), integer labels
                - 'filename': str, the image filename
        """
        img_path, mask_path = self.pairs[idx]

        # Load image (BGR -> RGB) - use imdecode for Windows compatibility
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        # Resize image (bilinear interpolation)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_LINEAR)

        # Resize mask (nearest neighbor to preserve labels)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_NEAREST)

        # Convert image to float [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Mask: keep as (H, W) with integer labels
        # Map 255 (ignore) and any invalid values (>6) to -1 for loss computation
        mask = mask.astype(np.int64)
        mask[mask == 255] = -1
        mask[mask > 6] = -1  # Handle any unexpected class values
        mask = torch.from_numpy(mask).long()

        # Apply additional transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "filename": img_path.name,
        }

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get raw (non-transformed) sample for visualization.

        Returns:
            Tuple of (image_rgb, mask, filename)
        """
        img_path, mask_path = self.pairs[idx]

        # Use imdecode for Windows compatibility
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_NEAREST)

        return image, mask, img_path.name


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert grayscale mask to RGB color visualization."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(CLASS_COLORS):
        color_mask[mask == class_id] = color

    # Ignore regions (255 or -1) -> gray
    color_mask[(mask == 255) | (mask == -1)] = [128, 128, 128]

    return color_mask


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create image with mask overlay."""
    color_mask = mask_to_color(mask)

    # Only overlay non-background regions
    overlay = image.copy()
    non_bg = mask > 0
    overlay[non_bg] = cv2.addWeighted(
        image[non_bg], 1 - alpha,
        color_mask[non_bg], alpha,
        0
    )

    return overlay


def visualize_samples(dataset: EndoscapesDataset, num_samples: int = 3,
                      save_path: Optional[Path] = None):
    """
    Visualize dataset samples: image | overlay | mask.

    Args:
        dataset: EndoscapesDataset instance
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Sample indices (spread across dataset)
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for i, idx in enumerate(indices):
        image, mask, filename = dataset.get_raw_sample(idx)
        overlay = create_overlay(image, mask)
        color_mask = mask_to_color(mask)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Overlay")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(color_mask)
        axes[i, 2].set_title(f"Mask (unique: {np.unique(mask).tolist()})")
        axes[i, 2].axis("off")

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(color)/255,
                      label=f"{i}: {name}")
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved to: {save_path}")

    plt.close()


def test_dataset():
    """Test the EndoscapesDataset class."""
    print("\n" + "="*70)
    print(" STEP 4: Dataset Testing")
    print("="*70)

    # Test each split
    for split in ["train", "val", "test"]:
        print(f"\n{'='*50}")
        print(f" Testing {split.upper()} split")
        print("="*50)

        images_dir, masks_dir = get_split_paths(split)

        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            continue

        dataset = EndoscapesDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            target_size=(512, 512),
            normalize=True,
        )

        print(f"\nDataset size: {len(dataset)} samples")

        if len(dataset) == 0:
            print("WARNING: No valid pairs found!")
            continue

        # Test loading samples
        print("\n--- Sample Loading Test ---")
        num_test = min(5, len(dataset))

        for i in range(num_test):
            sample = dataset[i]
            img = sample["image"]
            mask = sample["mask"]
            fname = sample["filename"]

            unique_vals = torch.unique(mask).tolist()

            print(f"  [{i}] {fname}")
            print(f"      Image shape: {tuple(img.shape)}, dtype: {img.dtype}")
            print(f"      Image range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"      Mask shape: {tuple(mask.shape)}, dtype: {mask.dtype}")
            print(f"      Mask unique values: {unique_vals}")

            # Verify mask values
            valid_range = list(range(NUM_CLASSES)) + [-1]  # 0-6 and -1 (ignore)
            invalid = [v for v in unique_vals if v not in valid_range]
            if invalid:
                print(f"      WARNING: Invalid mask values: {invalid}")
            else:
                print(f"      Mask values: VALID")

    # Test DataLoader
    print(f"\n{'='*50}")
    print(" DataLoader Test")
    print("="*50)

    images_dir, masks_dir = get_split_paths("train")
    train_dataset = EndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
    )

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True,
        )

        print(f"\nDataLoader created:")
        print(f"  Batch size: 2")
        print(f"  Num batches: {len(train_loader)}")

        # Fetch one batch
        print("\n--- Fetching one batch ---")
        batch = next(iter(train_loader))

        print(f"  Batch images shape: {tuple(batch['image'].shape)}")
        print(f"  Batch masks shape: {tuple(batch['mask'].shape)}")
        print(f"  Filenames: {batch['filename']}")

        # Memory estimate
        img_mem = batch['image'].element_size() * batch['image'].nelement() / 1024**2
        mask_mem = batch['mask'].element_size() * batch['mask'].nelement() / 1024**2
        print(f"\n  Batch memory: {img_mem + mask_mem:.2f} MB (images: {img_mem:.2f} MB, masks: {mask_mem:.2f} MB)")

    # Visualization
    print(f"\n{'='*50}")
    print(" Visualization")
    print("="*50)

    if len(train_dataset) > 0:
        save_path = OUTPUT_DIR / "step4_dataloader_samples.png"
        visualize_samples(train_dataset, num_samples=3, save_path=save_path)

    # Summary
    print(f"\n{'='*50}")
    print(" SUMMARY")
    print("="*50)

    for split in ["train", "val", "test"]:
        images_dir, masks_dir = get_split_paths(split)
        if images_dir.exists():
            ds = EndoscapesDataset(images_dir, masks_dir, target_size=(512, 512))
            print(f"  {split:5s}: {len(ds):4d} samples")

    print("\nDataset class: EndoscapesDataset")
    print("  - Resizes to 512x512")
    print("  - Normalizes with ImageNet stats")
    print("  - Masks use nearest-neighbor interpolation")
    print("  - Ignore label (255) mapped to -1")
    print("\nStatus: READY FOR TRAINING")


if __name__ == "__main__":
    test_dataset()
