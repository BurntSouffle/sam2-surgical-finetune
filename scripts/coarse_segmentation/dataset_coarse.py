"""
Coarse Segmentation Dataset for SAM2

Remaps the original 7-class Endoscapes labels to 4 coarse classes:
    Original → Coarse
    ─────────────────────────────
    0 (background)     → 0 (background)
    1 (cystic_plate)   → 3 (anatomy)
    2 (calot_triangle) → 3 (anatomy)
    3 (cystic_artery)  → 3 (anatomy)
    4 (cystic_duct)    → 3 (anatomy)
    5 (gallbladder)    → 1 (gallbladder)
    6 (tool)           → 2 (tool)
    255 (ignore)       → -1 (ignore)

Usage:
    python scripts/coarse_segmentation/dataset_coarse.py
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
# COARSE CLASS CONFIGURATION
# =============================================================================

# 4 coarse classes
COARSE_CLASS_NAMES = [
    "background",   # 0
    "gallbladder",  # 1
    "tool",         # 2
    "anatomy",      # 3 (cystic structures combined)
]
COARSE_NUM_CLASSES = 4
COARSE_IGNORE_INDEX = -1

# Mapping from original 7 classes to coarse 4 classes
# Original: 0=bg, 1=cystic_plate, 2=calot_triangle, 3=cystic_artery,
#           4=cystic_duct, 5=gallbladder, 6=tool, 255=ignore
LABEL_REMAP = {
    0: 0,    # background → background
    1: 3,    # cystic_plate → anatomy
    2: 3,    # calot_triangle → anatomy
    3: 3,    # cystic_artery → anatomy
    4: 3,    # cystic_duct → anatomy
    5: 1,    # gallbladder → gallbladder
    6: 2,    # tool → tool
    255: -1, # ignore → ignore
}

# Color map for coarse visualization (RGB)
COARSE_CLASS_COLORS = [
    [0, 0, 0],       # 0: background - black
    [255, 0, 255],   # 1: gallbladder - magenta
    [0, 255, 255],   # 2: tool - cyan
    [255, 128, 0],   # 3: anatomy - orange (combined cystic structures)
]

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

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "coarse_segmentation"


# =============================================================================
# COARSE DATASET
# =============================================================================

class CoarseEndoscapesDataset(Dataset):
    """
    PyTorch Dataset for coarse (4-class) semantic segmentation.

    Wraps the original Endoscapes data but remaps labels to 4 classes:
    - background (0)
    - gallbladder (1)
    - tool (2)
    - anatomy (3) - combines all cystic structures
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
        Initialize the coarse dataset.

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

        print(f"[CoarseEndoscapesDataset] Initialized:")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Masks dir: {self.masks_dir}")
        print(f"  Target size: {self.target_size}")
        print(f"  Valid pairs found: {len(self.pairs)}")
        print(f"  Classes: {COARSE_CLASS_NAMES}")

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

    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """
        Remap original 7-class labels to coarse 4-class labels.

        Args:
            mask: Original mask with values 0-6 and 255

        Returns:
            Remapped mask with values 0-3 and -1
        """
        remapped = np.full_like(mask, fill_value=-1, dtype=np.int64)

        for orig_label, coarse_label in LABEL_REMAP.items():
            remapped[mask == orig_label] = coarse_label

        return remapped

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess an image-mask pair with coarse labels.

        Returns:
            Dict with:
                - 'image': torch.Tensor of shape (3, H, W), normalized
                - 'mask': torch.Tensor of shape (H, W), coarse labels (0-3, -1)
                - 'filename': str, the image filename
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

        # Remap labels to coarse classes
        mask = self._remap_labels(mask)

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

        return {
            "image": image,
            "mask": mask,
            "filename": img_path.name,
        }

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Get raw sample for visualization (original and remapped masks).

        Returns:
            Tuple of (image_rgb, original_mask, coarse_mask, filename)
        """
        img_path, mask_path = self.pairs[idx]

        image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        original_mask = cv2.resize(original_mask, (self.target_size[1], self.target_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Create coarse mask
        coarse_mask = self._remap_labels(original_mask)

        return image, original_mask, coarse_mask, img_path.name


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


def mask_to_color_coarse(mask: np.ndarray) -> np.ndarray:
    """Convert coarse 4-class mask to RGB."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(COARSE_CLASS_COLORS):
        color_mask[mask == class_id] = color

    color_mask[mask == -1] = [128, 128, 128]
    return color_mask


def visualize_remapping(dataset: CoarseEndoscapesDataset, num_samples: int = 3,
                        save_path: Optional[Path] = None):
    """
    Visualize original vs coarse masks side by side.

    Shows: Image | Original Mask (7-class) | Coarse Mask (4-class)
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for i, idx in enumerate(indices):
        image, orig_mask, coarse_mask, filename = dataset.get_raw_sample(idx)

        orig_color = mask_to_color_original(orig_mask)
        coarse_color = mask_to_color_coarse(coarse_mask)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image: {filename}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(orig_color)
        orig_unique = [v for v in np.unique(orig_mask) if v not in [255, -1]]
        axes[i, 1].set_title(f"Original 7-class (classes: {orig_unique})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(coarse_color)
        coarse_unique = [v for v in np.unique(coarse_mask) if v != -1]
        axes[i, 2].set_title(f"Coarse 4-class (classes: {coarse_unique})")
        axes[i, 2].axis("off")

    # Legends
    fig.text(0.5, 0.02,
             "Original: 0=bg, 1=cystic_plate(red), 2=calot_tri(green), 3=cystic_art(blue), "
             "4=cystic_duct(yellow), 5=gallbladder(magenta), 6=tool(cyan)",
             ha='center', fontsize=9)
    fig.text(0.5, -0.01,
             "Coarse: 0=background(black), 1=gallbladder(magenta), 2=tool(cyan), 3=anatomy(orange)",
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

def compute_class_distribution(dataset: CoarseEndoscapesDataset) -> Dict[str, any]:
    """
    Compute pixel distribution across coarse classes.

    Returns:
        Dict with class pixel counts, percentages, and weights
    """
    print("\n" + "=" * 60)
    print(" Computing Coarse Class Distribution")
    print("=" * 60)

    class_pixels = np.zeros(COARSE_NUM_CLASSES, dtype=np.int64)
    total_valid_pixels = 0
    ignore_pixels = 0

    from tqdm import tqdm

    for i in tqdm(range(len(dataset)), desc="Scanning masks"):
        sample = dataset[i]
        mask = sample['mask'].numpy()

        for c in range(COARSE_NUM_CLASSES):
            class_pixels[c] += (mask == c).sum()

        total_valid_pixels += (mask >= 0).sum()
        ignore_pixels += (mask == -1).sum()

    # Compute percentages
    percentages = class_pixels / total_valid_pixels * 100

    # Compute class weights (inverse frequency, normalized)
    weights = total_valid_pixels / (COARSE_NUM_CLASSES * class_pixels + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, 10.0)

    # Print results
    print("\n" + "-" * 60)
    print(f" {'Class':<15} {'Pixels':>15} {'Percentage':>12} {'Weight':>10}")
    print("-" * 60)

    for c in range(COARSE_NUM_CLASSES):
        print(f" {c}: {COARSE_CLASS_NAMES[c]:<12} {class_pixels[c]:>15,} {percentages[c]:>11.2f}% {weights[c]:>10.3f}")

    print("-" * 60)
    print(f" {'Total valid':<15} {total_valid_pixels:>15,}")
    print(f" {'Ignore (-1)':<15} {ignore_pixels:>15,}")
    print("-" * 60)

    return {
        'class_pixels': class_pixels,
        'percentages': percentages,
        'weights': weights,
        'total_valid_pixels': total_valid_pixels,
        'ignore_pixels': ignore_pixels,
    }


def verify_remapping(dataset: CoarseEndoscapesDataset, num_samples: int = 5):
    """
    Verify that label remapping is correct by checking specific samples.
    """
    print("\n" + "=" * 60)
    print(" Verifying Label Remapping")
    print("=" * 60)

    print("\nExpected mapping:")
    print("  0 (background)     -> 0 (background)")
    print("  1 (cystic_plate)   -> 3 (anatomy)")
    print("  2 (calot_triangle) -> 3 (anatomy)")
    print("  3 (cystic_artery)  -> 3 (anatomy)")
    print("  4 (cystic_duct)    -> 3 (anatomy)")
    print("  5 (gallbladder)    -> 1 (gallbladder)")
    print("  6 (tool)           -> 2 (tool)")
    print("  255 (ignore)       -> -1 (ignore)")

    print(f"\nChecking {num_samples} samples...")

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    all_correct = True
    for idx in indices:
        image, orig_mask, coarse_mask, filename = dataset.get_raw_sample(idx)

        # Verify each original class maps correctly
        for orig_val in np.unique(orig_mask):
            if orig_val == 255:
                expected = -1
            else:
                expected = LABEL_REMAP.get(orig_val, -1)

            orig_positions = (orig_mask == orig_val)
            coarse_values_at_positions = coarse_mask[orig_positions]

            if not np.all(coarse_values_at_positions == expected):
                print(f"  ERROR in {filename}: orig={orig_val} should map to {expected}")
                all_correct = False

        print(f"  {filename}: orig_unique={np.unique(orig_mask).tolist()} -> coarse_unique={np.unique(coarse_mask).tolist()}")

    if all_correct:
        print("\n  All remappings verified CORRECT!")
    else:
        print("\n  WARNING: Some remappings are incorrect!")

    return all_correct


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Test the coarse dataset."""
    print("\n" + "=" * 70)
    print(" COARSE SEGMENTATION DATASET TEST")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training dataset
    images_dir, masks_dir = get_split_paths('train')

    print(f"\nLoading coarse dataset...")
    dataset = CoarseEndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        normalize=True,
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Verify remapping
    verify_remapping(dataset, num_samples=5)

    # Compute class distribution
    dist = compute_class_distribution(dataset)

    # Print expected vs actual
    print("\n" + "=" * 60)
    print(" Expected vs Actual Distribution")
    print("=" * 60)
    expected = {
        'background': 72,
        'gallbladder': 17,
        'tool': 7,
        'anatomy': 3,
    }
    print(f"\n {'Class':<15} {'Expected':>10} {'Actual':>10}")
    print("-" * 40)
    for c, name in enumerate(COARSE_CLASS_NAMES):
        print(f" {name:<15} {expected[name]:>9}% {dist['percentages'][c]:>9.1f}%")

    # Save visualization
    print("\n" + "=" * 60)
    print(" Saving Visualizations")
    print("=" * 60)

    save_path = OUTPUT_DIR / "coarse_remapping_samples.png"
    visualize_remapping(dataset, num_samples=4, save_path=save_path)

    # Test DataLoader
    print("\n" + "=" * 60)
    print(" DataLoader Test")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"\nBatch shapes:")
    print(f"  Images: {tuple(batch['image'].shape)}")
    print(f"  Masks:  {tuple(batch['mask'].shape)}")
    print(f"  Mask unique values: {torch.unique(batch['mask']).tolist()}")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
Coarse Dataset Ready:
  - {COARSE_NUM_CLASSES} classes: {COARSE_CLASS_NAMES}
  - Training samples: {len(dataset)}
  - Class weights: {[f'{w:.3f}' for w in dist['weights']]}

Output saved to: {OUTPUT_DIR}

Usage:
    from dataset_coarse import CoarseEndoscapesDataset, get_split_paths
    from dataset_coarse import COARSE_CLASS_NAMES, COARSE_NUM_CLASSES

    images_dir, masks_dir = get_split_paths('train')
    dataset = CoarseEndoscapesDataset(images_dir, masks_dir)
""")


if __name__ == "__main__":
    main()
