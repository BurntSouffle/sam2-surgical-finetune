"""
Box-Prompted Dataset for SAM2 Fine-tuning

This dataset returns (image, bbox, gt_binary_mask) triplets for training SAM2
with bounding box prompts. Each image can produce multiple samples - one per
object/class present in the image.

This is the CORRECT way to fine-tune SAM2:
- SAM2 takes an image + prompt (box, point, or mask)
- It outputs a binary mask for the prompted object
- We compare the predicted mask against the ground truth binary mask
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Class definitions (skip background class 0)
CLASS_NAMES = {
    0: 'background',
    1: 'cystic_plate',
    2: 'calot_triangle',
    3: 'cystic_artery',
    4: 'cystic_duct',
    5: 'gallbladder',
    6: 'tool',
}

# Classes to extract (skip background)
OBJECT_CLASSES = [1, 2, 3, 4, 5, 6]

# Colors for visualization (RGB)
CLASS_COLORS = {
    1: (255, 0, 0),      # cystic_plate - red
    2: (0, 255, 0),      # calot_triangle - green
    3: (0, 0, 255),      # cystic_artery - blue
    4: (255, 255, 0),    # cystic_duct - yellow
    5: (255, 0, 255),    # gallbladder - magenta
    6: (0, 255, 255),    # tool - cyan
}


def get_bbox_from_mask(binary_mask: np.ndarray, padding: int = 10) -> Optional[np.ndarray]:
    """
    Extract bounding box from binary mask with optional padding.

    Args:
        binary_mask: Binary mask array (H, W) with 0s and 1s
        padding: Pixels to add around the tight bounding box

    Returns:
        Bounding box as [x1, y1, x2, y2] (XYXY format) or None if mask is empty
    """
    # Find non-zero coordinates
    coords = np.where(binary_mask > 0)

    if len(coords[0]) == 0:
        return None  # Empty mask

    # Get bounding box (y coords in coords[0], x coords in coords[1])
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # Add padding
    h, w = binary_mask.shape
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w, x_max + padding)
    y2 = min(h, y_max + padding)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def resolve_symlink_path(file_path: Path) -> Path:
    """
    Resolve symlink-like text files in Endoscapes dataset.

    Some .jpg files are actually text files containing relative paths
    to the actual image files.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        # Check if it's a relative path reference
        if content.startswith('..'):
            # Resolve relative to the file's directory
            actual_path = (file_path.parent / content).resolve()
            if actual_path.exists():
                return actual_path
    except:
        pass

    return file_path


class EndoscapesBoxDataset:
    """
    Dataset for SAM2 box-prompted fine-tuning.

    Returns (image, bbox, gt_binary_mask) triplets. One image can produce
    multiple samples - one per object/class present in the image.

    Example:
        dataset = EndoscapesBoxDataset(images_dir, masks_dir, split='train')
        sample = dataset[0]
        # sample['image']: (H, W, 3) uint8 RGB numpy array
        # sample['bbox']: [x1, y1, x2, y2] float32 array
        # sample['gt_mask']: (H, W) binary numpy array (0 or 1)
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        split: str = 'train',
        min_mask_area: int = 100,
        bbox_padding: int = 10,
    ):
        """
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing semantic segmentation masks
            split: Data split ('train', 'val', 'test')
            min_mask_area: Minimum mask area in pixels to include (filter tiny objects)
            bbox_padding: Padding to add around bounding boxes
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.split = split
        self.min_mask_area = min_mask_area
        self.bbox_padding = bbox_padding

        # Build list of (image_path, mask_path, class_id) samples
        self.samples = self._build_sample_list()

        print(f"[EndoscapesBoxDataset] Initialized for {split}:")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Masks dir: {self.masks_dir}")
        print(f"  Total object samples: {len(self.samples)}")

        # Print class distribution
        class_counts = {}
        for _, _, class_id in self.samples:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print(f"  Class distribution:")
        for class_id in sorted(class_counts.keys()):
            print(f"    {CLASS_NAMES[class_id]:15s}: {class_counts[class_id]:4d} samples")

    def _build_sample_list(self) -> List[Tuple[Path, Path, int]]:
        """
        Build flat list of (image_path, mask_path, class_id) for all objects.

        Each image-mask pair can contribute multiple samples (one per class present).
        """
        samples = []

        # Find all mask files
        mask_files = sorted(self.masks_dir.glob("*.png"))

        for mask_path in mask_files:
            # Find corresponding image
            # Mask: "frame_123_endo_seg.png" -> Image: "frame_123_endo.jpg"
            mask_stem = mask_path.stem  # e.g., "frame_123_endo_seg"

            # Try different naming patterns
            possible_image_stems = [
                mask_stem.replace("_seg", ""),  # frame_123_endo
                mask_stem.replace("_endo_seg", "_endo"),  # frame_123_endo
                mask_stem,  # same name
            ]

            image_path = None
            for stem in possible_image_stems:
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = self.images_dir / f"{stem}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path:
                    break

            if image_path is None:
                continue

            # Resolve symlink-like files
            image_path = resolve_symlink_path(image_path)

            # Load mask to find which classes are present
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Check each object class
            for class_id in OBJECT_CLASSES:
                binary_mask = (mask == class_id).astype(np.uint8)
                mask_area = binary_mask.sum()

                # Skip if mask is too small
                if mask_area >= self.min_mask_area:
                    samples.append((image_path, mask_path, class_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dict with:
                - 'image': np.array (H, W, 3) uint8 RGB
                - 'bbox': np.array [x1, y1, x2, y2] float32
                - 'gt_mask': np.array (H, W) uint8 binary (0 or 1)
                - 'class_id': int
                - 'class_name': str
                - 'image_filename': str
                - 'original_size': (H, W) tuple
        """
        image_path, mask_path, class_id = self.samples[idx]

        # Load image (SAM2 expects RGB numpy array)
        image = cv2.imread(str(image_path))
        if image is None:
            # Try reading with numpy for Windows path compatibility
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")

        # Extract binary mask for this class
        gt_mask = (mask == class_id).astype(np.uint8)

        # Get bounding box
        bbox = get_bbox_from_mask(gt_mask, padding=self.bbox_padding)

        if bbox is None:
            raise RuntimeError(f"Empty mask for class {class_id} at {mask_path}")

        return {
            'image': image,
            'bbox': bbox,
            'gt_mask': gt_mask,
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'image_filename': image_path.name,
            'original_size': (image.shape[0], image.shape[1]),
        }


def get_split_paths(split: str = 'train') -> Tuple[Path, Path]:
    """Get image and mask directories for a split."""
    base_dir = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes")

    # Images are in train/, val/, test/ subdirectories
    images_dir = base_dir / split

    # All masks are in semseg/ directory
    masks_dir = base_dir / "semseg"

    return images_dir, masks_dir


def visualize_samples(dataset: EndoscapesBoxDataset, indices: List[int], save_path: str):
    """
    Visualize samples with bounding box and ground truth mask overlay.

    Args:
        dataset: Dataset instance
        indices: List of sample indices to visualize
        save_path: Path to save the visualization
    """
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]

        image = sample['image']
        bbox = sample['bbox']
        gt_mask = sample['gt_mask']
        class_name = sample['class_name']
        class_id = sample['class_id']

        # Column 1: Image with bounding box
        axes[i, 0].imshow(image)
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f'Image + BBox\n{class_name} (class {class_id})')
        axes[i, 0].axis('off')

        # Column 2: Ground truth mask
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title(f'GT Binary Mask\nArea: {gt_mask.sum()} px')
        axes[i, 1].axis('off')

        # Column 3: Image with mask overlay
        overlay = image.copy()
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        mask_rgb = np.zeros_like(image)
        mask_rgb[gt_mask > 0] = color
        overlay = cv2.addWeighted(overlay, 0.6, mask_rgb, 0.4, 0)

        # Draw bbox on overlay too
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay\nBBox: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]')
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {save_path}")


def test_dataset():
    """Test the dataset class."""
    print("=" * 70)
    print(" ENDOSCAPES BOX-PROMPTED DATASET TEST")
    print("=" * 70)

    # Load dataset for train split
    images_dir, masks_dir = get_split_paths('train')

    print(f"\nImages directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")

    dataset = EndoscapesBoxDataset(
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        split='train',
        min_mask_area=100,
        bbox_padding=10,
    )

    print(f"\n{'='*50}")
    print(f" DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total samples: {len(dataset)}")

    # Load and verify 5 random samples
    print(f"\n{'='*50}")
    print(f" SAMPLE VERIFICATION")
    print(f"{'='*50}")

    random.seed(42)
    test_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    for idx in test_indices:
        sample = dataset[idx]

        # Verify shapes
        image = sample['image']
        bbox = sample['bbox']
        gt_mask = sample['gt_mask']

        print(f"\nSample {idx}:")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Image range: [{image.min()}, {image.max()}]")
        print(f"  BBox: {bbox} (format: [x1, y1, x2, y2])")
        print(f"  GT Mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
        print(f"  GT Mask unique values: {np.unique(gt_mask)}")
        print(f"  Mask area: {gt_mask.sum()} pixels")
        print(f"  Class: {sample['class_name']} (id={sample['class_id']})")
        print(f"  Filename: {sample['image_filename']}")

        # Verify bbox is valid
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        assert 0 <= x1 < x2 <= w, f"Invalid bbox x coords: {x1}, {x2}, w={w}"
        assert 0 <= y1 < y2 <= h, f"Invalid bbox y coords: {y1}, {y2}, h={h}"

        # Verify mask is within bbox (with some tolerance)
        mask_coords = np.where(gt_mask > 0)
        if len(mask_coords[0]) > 0:
            mask_y_min, mask_y_max = mask_coords[0].min(), mask_coords[0].max()
            mask_x_min, mask_x_max = mask_coords[1].min(), mask_coords[1].max()
            assert mask_x_min >= x1 - 1 and mask_x_max <= x2 + 1, "Mask outside bbox (x)"
            assert mask_y_min >= y1 - 1 and mask_y_max <= y2 + 1, "Mask outside bbox (y)"

        print(f"  Verification: PASSED")

    # Visualize 3 samples
    print(f"\n{'='*50}")
    print(f" VISUALIZATION")
    print(f"{'='*50}")

    vis_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
    output_path = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs\box_prompted\dataset_samples.png")

    visualize_samples(dataset, vis_indices, str(output_path))

    # Also test val and test splits
    print(f"\n{'='*50}")
    print(f" OTHER SPLITS")
    print(f"{'='*50}")

    for split in ['val', 'test']:
        images_dir, masks_dir = get_split_paths(split)
        if images_dir.exists():
            ds = EndoscapesBoxDataset(
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                split=split,
                min_mask_area=100,
                bbox_padding=10,
            )
            print(f"\n{split}: {len(ds)} samples")

    print(f"\n{'='*50}")
    print(f" TEST COMPLETE")
    print(f"{'='*50}")
    print(f"""
Summary:
- Dataset creates one sample per object instance
- Each sample has: image (RGB), bbox [x1,y1,x2,y2], gt_mask (binary)
- Multiple objects per image = multiple samples
- Ready for SAM2 box-prompted training!
""")


if __name__ == "__main__":
    test_dataset()
