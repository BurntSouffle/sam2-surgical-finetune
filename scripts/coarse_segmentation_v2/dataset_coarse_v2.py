"""
Coarse Segmentation V2 Dataset - 3 Classes with Anatomy IGNORED

Maps the original 7-class Endoscapes labels to 3 classes:
- Anatomy pixels (cystic structures) are IGNORED in training
- Model only learns: background, gallbladder, tool
- Anatomy regions become "uncertain" at inference -> search region for SAM2

Label Remapping:
    Original         ->  New
    --------------------------------
    0 (background)   ->  0 (background)
    1 (cystic_plate) ->  -1 (IGNORE)
    2 (calot_tri)    ->  -1 (IGNORE)
    3 (cystic_art)   ->  -1 (IGNORE)
    4 (cystic_duct)  ->  -1 (IGNORE)
    5 (gallbladder)  ->  1 (gallbladder)
    6 (tool)         ->  2 (tool)
    255 (ignore)     ->  -1 (IGNORE)

Usage:
    python scripts/coarse_segmentation_v2/dataset_coarse_v2.py
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

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from step4_dataset import (
    DATASET_ROOT, IMAGENET_MEAN, IMAGENET_STD,
    get_split_paths as get_split_paths_original
)

# =============================================================================
# V2 CLASS CONFIGURATION - 3 CLASSES (anatomy ignored)
# =============================================================================

# 3 classes only - anatomy is ignored
COARSE_V2_CLASS_NAMES = [
    "background",   # 0
    "gallbladder",  # 1
    "tool",         # 2
]
COARSE_V2_NUM_CLASSES = 3
COARSE_V2_IGNORE_INDEX = -1

# Mapping from original 7 classes to 3 classes
# Anatomy classes (1-4) map to -1 (IGNORE)
LABEL_REMAP_V2 = {
    0: 0,    # background -> background
    1: -1,   # cystic_plate -> IGNORE
    2: -1,   # calot_triangle -> IGNORE
    3: -1,   # cystic_artery -> IGNORE
    4: -1,   # cystic_duct -> IGNORE
    5: 1,    # gallbladder -> gallbladder
    6: 2,    # tool -> tool
    255: -1, # ignore -> IGNORE
}

# Color map for 3-class visualization (RGB)
COARSE_V2_CLASS_COLORS = [
    [0, 0, 0],       # 0: background - black
    [255, 0, 255],   # 1: gallbladder - magenta
    [0, 255, 255],   # 2: tool - cyan
]

# Color for anatomy (shown in visualizations but ignored in training)
ANATOMY_COLOR = [255, 128, 0]  # Orange - for visualization only

# Original class colors for comparison
ORIGINAL_CLASS_COLORS = [
    [0, 0, 0],       # 0: background - black
    [255, 0, 0],     # 1: cystic_plate - red
    [0, 255, 0],     # 2: calot_triangle - green
    [0, 0, 255],     # 3: cystic_artery - blue
    [255, 255, 0],   # 4: cystic_duct - yellow
    [255, 0, 255],   # 5: gallbladder - magenta
    [0, 255, 255],   # 6: tool - cyan
]

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "coarse_segmentation_v2"


# =============================================================================
# V2 DATASET - 3 Classes
# =============================================================================

class CoarseV2EndoscapesDataset(Dataset):
    """
    PyTorch Dataset for 3-class semantic segmentation.

    Anatomy pixels are ignored in training (mapped to -1).
    This allows the model to focus on background, gallbladder, tool.
    At inference, uncertain regions indicate potential anatomy.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        transform=None,
        return_anatomy_mask: bool = False,  # Option to return original anatomy mask
    ):
        """
        Initialize the 3-class dataset.

        Args:
            images_dir: Directory containing images (.jpg)
            masks_dir: Directory containing masks (.png)
            target_size: (H, W) to resize images and masks to
            normalize: Whether to apply ImageNet normalization
            transform: Optional additional transforms
            return_anatomy_mask: If True, also return binary anatomy mask
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.transform = transform
        self.return_anatomy_mask = return_anatomy_mask

        # Find matching pairs
        self.pairs = self._find_matching_pairs()

        print(f"[CoarseV2EndoscapesDataset] Initialized:")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Masks dir: {self.masks_dir}")
        print(f"  Target size: {self.target_size}")
        print(f"  Valid pairs found: {len(self.pairs)}")
        print(f"  Classes: {COARSE_V2_CLASS_NAMES}")
        print(f"  Anatomy pixels: IGNORED (mapped to -1)")

    def _find_matching_pairs(self) -> List[Tuple[Path, Path]]:
        """Find image-mask pairs with matching filenames."""
        pairs = []

        mask_stems = {p.stem: p for p in self.masks_dir.glob("*.png")}

        for img_path in self.images_dir.glob("*.jpg"):
            stem = img_path.stem
            if stem in mask_stems:
                pairs.append((img_path, mask_stems[stem]))

        pairs.sort(key=lambda x: x[0].name)
        return pairs

    def _remap_labels(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remap original 7-class labels to 3-class labels.
        Also returns binary anatomy mask for evaluation.

        Args:
            mask: Original mask with values 0-6 and 255

        Returns:
            remapped: 3-class mask with values 0-2 and -1
            anatomy_mask: Binary mask where anatomy=1
        """
        remapped = np.full_like(mask, fill_value=-1, dtype=np.int64)
        anatomy_mask = np.zeros_like(mask, dtype=np.uint8)

        for orig_label, new_label in LABEL_REMAP_V2.items():
            remapped[mask == orig_label] = new_label

            # Track anatomy pixels (original classes 1-4)
            if orig_label in [1, 2, 3, 4]:
                anatomy_mask[mask == orig_label] = 1

        return remapped, anatomy_mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess an image-mask pair with 3-class labels.

        Returns:
            Dict with:
                - 'image': torch.Tensor of shape (3, H, W), normalized
                - 'mask': torch.Tensor of shape (H, W), 3-class labels (0-2, -1)
                - 'filename': str, the image filename
                - 'anatomy_mask': (optional) binary mask of anatomy pixels
        """
        img_path, mask_path = self.pairs[idx]

        # Load image (BGR -> RGB)
        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                         interpolation=cv2.INTER_NEAREST)

        # Remap labels to 3 classes (anatomy -> -1)
        mask, anatomy_mask = self._remap_labels(mask)

        # Convert image to float [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        # Apply additional transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        result = {
            "image": image,
            "mask": mask,
            "filename": img_path.name,
        }

        if self.return_anatomy_mask:
            result["anatomy_mask"] = torch.from_numpy(anatomy_mask).long()

        return result

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Get raw sample for visualization (original, 3-class, and anatomy masks).

        Returns:
            Tuple of (image_rgb, original_mask, v2_mask, anatomy_mask, filename)
        """
        img_path, mask_path = self.pairs[idx]

        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        original_mask = cv2.resize(original_mask, (self.target_size[1], self.target_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Create 3-class mask and anatomy mask
        v2_mask, anatomy_mask = self._remap_labels(original_mask)

        return image, original_mask, v2_mask, anatomy_mask, img_path.name


def get_split_paths(split: str) -> Tuple[Path, Path]:
    """Get image and mask directory paths for a given split."""
    return get_split_paths_original(split)


# =============================================================================
# VISUALIZATION
# =============================================================================

def mask_to_color_original(mask: np.ndarray) -> np.ndarray:
    """Convert original 7-class mask to RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(ORIGINAL_CLASS_COLORS):
        color_mask[mask == class_id] = color

    color_mask[(mask == 255) | (mask == -1)] = [128, 128, 128]
    return color_mask


def mask_to_color_v2(mask: np.ndarray, show_ignore_as_gray: bool = True) -> np.ndarray:
    """Convert 3-class mask to RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(COARSE_V2_CLASS_COLORS):
        color_mask[mask == class_id] = color

    if show_ignore_as_gray:
        color_mask[mask == -1] = [128, 128, 128]  # Gray for ignored

    return color_mask


def mask_to_color_v2_with_anatomy(mask: np.ndarray, anatomy_mask: np.ndarray) -> np.ndarray:
    """Convert 3-class mask to RGB, showing anatomy in orange."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(COARSE_V2_CLASS_COLORS):
        color_mask[mask == class_id] = color

    # Show anatomy (ignored) in orange
    color_mask[anatomy_mask == 1] = ANATOMY_COLOR

    # Other ignored pixels in gray
    other_ignored = (mask == -1) & (anatomy_mask == 0)
    color_mask[other_ignored] = [128, 128, 128]

    return color_mask


def visualize_v2_remapping(dataset: CoarseV2EndoscapesDataset, num_samples: int = 4,
                           save_path: Optional[Path] = None):
    """
    Visualize original vs 3-class masks with anatomy highlighted.

    Shows: Image | Original (7-class) | V2 (3-class) | Anatomy regions
    """
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for i, idx in enumerate(indices):
        image, orig_mask, v2_mask, anatomy_mask, filename = dataset.get_raw_sample(idx)

        orig_color = mask_to_color_original(orig_mask)
        v2_color = mask_to_color_v2_with_anatomy(v2_mask, anatomy_mask)

        # Anatomy highlight overlay
        anatomy_overlay = image.copy()
        anatomy_overlay[anatomy_mask == 1] = (
            0.5 * anatomy_overlay[anatomy_mask == 1] +
            0.5 * np.array(ANATOMY_COLOR)
        ).astype(np.uint8)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(orig_color)
        axes[i, 1].set_title(f"Original 7-class")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(v2_color)
        v2_unique = [v for v in np.unique(v2_mask) if v >= 0]
        axes[i, 2].set_title(f"V2 3-class (classes: {v2_unique})")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(anatomy_overlay)
        anatomy_pct = anatomy_mask.sum() / anatomy_mask.size * 100
        axes[i, 3].set_title(f"Anatomy IGNORED ({anatomy_pct:.1f}%)")
        axes[i, 3].axis("off")

    # Legend
    fig.text(0.5, 0.02,
             "V2: 0=background(black), 1=gallbladder(magenta), 2=tool(cyan), IGNORED=anatomy(orange)/other(gray)",
             ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved to: {save_path}")

    plt.close()


# =============================================================================
# CLASS DISTRIBUTION ANALYSIS
# =============================================================================

def compute_class_distribution_v2(dataset: CoarseV2EndoscapesDataset) -> Dict[str, any]:
    """
    Compute pixel distribution across 3 classes + ignored.

    Returns:
        Dict with class pixel counts, percentages, and weights
    """
    print("\n" + "=" * 60)
    print(" Computing V2 Class Distribution (3 classes)")
    print("=" * 60)

    class_pixels = np.zeros(COARSE_V2_NUM_CLASSES, dtype=np.int64)
    anatomy_pixels = 0
    other_ignore_pixels = 0
    total_pixels = 0

    from tqdm import tqdm

    # Temporarily enable anatomy mask return
    original_return_anatomy = dataset.return_anatomy_mask
    dataset.return_anatomy_mask = True

    for i in tqdm(range(len(dataset)), desc="Scanning masks"):
        sample = dataset[i]
        mask = sample['mask'].numpy()
        anatomy_mask = sample['anatomy_mask'].numpy()

        for c in range(COARSE_V2_NUM_CLASSES):
            class_pixels[c] += (mask == c).sum()

        anatomy_pixels += anatomy_mask.sum()
        other_ignore_pixels += ((mask == -1) & (anatomy_mask == 0)).sum()
        total_pixels += mask.size

    # Restore original setting
    dataset.return_anatomy_mask = original_return_anatomy

    # Compute percentages (of total pixels)
    total_valid = class_pixels.sum()
    percentages = class_pixels / total_pixels * 100
    anatomy_pct = anatomy_pixels / total_pixels * 100
    other_ignore_pct = other_ignore_pixels / total_pixels * 100

    # Compute class weights (only for 3 classes)
    weights = total_valid / (COARSE_V2_NUM_CLASSES * class_pixels + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, 10.0)

    # Print results
    print("\n" + "-" * 60)
    print(f" {'Category':<20} {'Pixels':>15} {'Percentage':>12} {'Weight':>10}")
    print("-" * 60)

    for c in range(COARSE_V2_NUM_CLASSES):
        print(f" {c}: {COARSE_V2_CLASS_NAMES[c]:<17} {class_pixels[c]:>15,} {percentages[c]:>11.2f}% {weights[c]:>10.3f}")

    print("-" * 60)
    print(f" {'IGNORED (anatomy)':<20} {anatomy_pixels:>15,} {anatomy_pct:>11.2f}%   {'N/A':>9}")
    print(f" {'IGNORED (other)':<20} {other_ignore_pixels:>15,} {other_ignore_pct:>11.2f}%   {'N/A':>9}")
    print("-" * 60)
    print(f" {'Total valid (3-class)':<20} {total_valid:>15,} {total_valid/total_pixels*100:>11.2f}%")
    print(f" {'Total ignored':<20} {anatomy_pixels + other_ignore_pixels:>15,} {(anatomy_pct + other_ignore_pct):>11.2f}%")
    print(f" {'Total pixels':<20} {total_pixels:>15,}")
    print("-" * 60)

    return {
        'class_pixels': class_pixels,
        'percentages': percentages,
        'weights': weights,
        'anatomy_pixels': anatomy_pixels,
        'anatomy_pct': anatomy_pct,
        'other_ignore_pixels': other_ignore_pixels,
        'total_valid': total_valid,
        'total_pixels': total_pixels,
    }


def verify_v2_remapping(dataset: CoarseV2EndoscapesDataset, num_samples: int = 5):
    """
    Verify that label remapping is correct for V2 (3 classes).
    """
    print("\n" + "=" * 60)
    print(" Verifying V2 Label Remapping (3 classes)")
    print("=" * 60)

    print("\nExpected mapping:")
    print("  0 (background)   -> 0 (background)")
    print("  1 (cystic_plate) -> -1 (IGNORED)")
    print("  2 (calot_tri)    -> -1 (IGNORED)")
    print("  3 (cystic_art)   -> -1 (IGNORED)")
    print("  4 (cystic_duct)  -> -1 (IGNORED)")
    print("  5 (gallbladder)  -> 1 (gallbladder)")
    print("  6 (tool)         -> 2 (tool)")
    print("  255 (ignore)     -> -1 (IGNORED)")

    print(f"\nChecking {num_samples} samples...")

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    all_correct = True
    for idx in indices:
        image, orig_mask, v2_mask, anatomy_mask, filename = dataset.get_raw_sample(idx)

        # Verify each original class maps correctly
        for orig_val in np.unique(orig_mask):
            expected = LABEL_REMAP_V2.get(orig_val, -1)

            orig_positions = (orig_mask == orig_val)
            v2_values_at_positions = v2_mask[orig_positions]

            if not np.all(v2_values_at_positions == expected):
                print(f"  ERROR in {filename}: orig={orig_val} should map to {expected}")
                all_correct = False

        # Count classes
        v2_unique = np.unique(v2_mask)
        anatomy_count = anatomy_mask.sum()

        print(f"  {filename}: v2_unique={v2_unique.tolist()}, anatomy_pixels={anatomy_count}")

    if all_correct:
        print("\n  All V2 remappings verified CORRECT!")
    else:
        print("\n  WARNING: Some remappings are incorrect!")

    return all_correct


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Test the V2 (3-class) dataset."""
    print("\n" + "=" * 70)
    print(" COARSE SEGMENTATION V2 DATASET TEST (3 Classes)")
    print("=" * 70)
    print("\nKey difference from V1:")
    print("  - V1: 4 classes (bg, gallbladder, tool, anatomy)")
    print("  - V2: 3 classes (bg, gallbladder, tool) - anatomy IGNORED")
    print("  - Anatomy pixels don't contribute to training loss")
    print("  - At inference, model uncertainty indicates anatomy regions")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training dataset
    images_dir, masks_dir = get_split_paths('train')

    print(f"\nLoading V2 dataset...")
    dataset = CoarseV2EndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        normalize=True,
        return_anatomy_mask=True,
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Verify remapping
    verify_v2_remapping(dataset, num_samples=5)

    # Compute class distribution
    dist = compute_class_distribution_v2(dataset)

    # Print expected vs actual
    print("\n" + "=" * 60)
    print(" Expected vs Actual Distribution")
    print("=" * 60)
    expected = {
        'background': 72,
        'gallbladder': 17,
        'tool': 7,
        'anatomy_ignored': 3,
    }
    print(f"\n {'Category':<20} {'Expected':>10} {'Actual':>10}")
    print("-" * 45)
    for c, name in enumerate(COARSE_V2_CLASS_NAMES):
        print(f" {name:<20} {expected[name]:>9}% {dist['percentages'][c]:>9.1f}%")
    print(f" {'anatomy (IGNORED)':<20} {expected['anatomy_ignored']:>9}% {dist['anatomy_pct']:>9.1f}%")

    # Save visualization
    print("\n" + "=" * 60)
    print(" Saving Visualizations")
    print("=" * 60)

    save_path = OUTPUT_DIR / "v2_remapping_samples.png"
    visualize_v2_remapping(dataset, num_samples=4, save_path=save_path)

    # Test DataLoader
    print("\n" + "=" * 60)
    print(" DataLoader Test")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"  Images: {tuple(batch['image'].shape)}")
    print(f"  Masks:  {tuple(batch['mask'].shape)}")
    print(f"  Anatomy masks: {tuple(batch['anatomy_mask'].shape)}")
    print(f"  Mask unique values: {torch.unique(batch['mask']).tolist()}")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
V2 Dataset Ready (3 classes):
  - {COARSE_V2_NUM_CLASSES} classes: {COARSE_V2_CLASS_NAMES}
  - Anatomy pixels: IGNORED ({dist['anatomy_pct']:.1f}% of pixels)
  - Training samples: {len(dataset)}
  - Class weights: {[f'{w:.3f}' for w in dist['weights']]}

Key benefit:
  - Model learns clear boundaries for bg/gb/tool
  - Anatomy regions become "uncertain" at inference
  - Use uncertainty to find anatomy search region for SAM2

Output saved to: {OUTPUT_DIR}

Usage:
    from dataset_coarse_v2 import CoarseV2EndoscapesDataset, get_split_paths
    from dataset_coarse_v2 import COARSE_V2_CLASS_NAMES, COARSE_V2_NUM_CLASSES

    images_dir, masks_dir = get_split_paths('train')
    dataset = CoarseV2EndoscapesDataset(images_dir, masks_dir)
""")


if __name__ == "__main__":
    main()
