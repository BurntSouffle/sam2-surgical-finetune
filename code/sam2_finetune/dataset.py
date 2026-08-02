"""
Dataset loader for Endoscapes2023 semantic segmentation with SAM2 point prompts.

Handles:
- Loading images and semantic masks
- Converting semantic masks to per-class binary masks
- Generating random point prompts within mask regions
- Data augmentation
- Resizing to SAM2 input resolution
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

from config import Config, get_config


class EndoscapesDataset(Dataset):
    """
    Endoscapes2023 dataset for SAM2 fine-tuning.

    Each sample returns:
    - image: Resized to 1024x1024
    - masks: List of binary masks (one per class present)
    - point_coords: Point prompts for each mask
    - point_labels: Labels for points (1 = foreground)
    - class_ids: Which class each mask belongs to
    - original_size: Original image dimensions
    """

    def __init__(
        self,
        config: Config,
        split: str = "train",
        transform: bool = True,
    ):
        """
        Args:
            config: Configuration object
            split: One of "train", "val", "test"
            transform: Whether to apply data augmentation (only for train)
        """
        self.config = config
        self.split = split
        self.transform = transform and split == "train"

        # Set paths based on split
        if split == "train":
            self.image_dir = config.paths.train_images
            self.annotation_file = config.paths.train_annotations
        elif split == "val":
            self.image_dir = config.paths.val_images
            self.annotation_file = config.paths.val_annotations
        elif split == "test":
            self.image_dir = config.paths.test_images
            self.annotation_file = config.paths.test_annotations
        else:
            raise ValueError(f"Invalid split: {split}")

        self.mask_dir = config.paths.semantic_masks
        self.target_size = config.model.image_size

        # Load image list from COCO annotations
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load sample information from COCO annotations."""
        samples = []

        # Load COCO annotations
        with open(self.annotation_file, "r") as f:
            coco_data = json.load(f)

        # Build image id to filename mapping
        for img_info in coco_data["images"]:
            img_id = img_info["id"]
            img_filename = img_info["file_name"]

            # Check if image exists
            img_path = self.image_dir / img_filename
            if not img_path.exists():
                continue

            # Check if corresponding mask exists
            # Mask filename matches image filename but with .png extension
            mask_filename = Path(img_filename).stem + ".png"
            mask_path = self.mask_dir / mask_filename

            if not mask_path.exists():
                continue

            samples.append({
                "image_id": img_id,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "filename": img_filename,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """Load semantic mask as numpy array."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask

    def _extract_class_masks(
        self, semantic_mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract binary masks for each class present in semantic mask.

        Returns:
            masks: List of binary masks
            class_ids: List of class IDs
        """
        masks = []
        class_ids = []

        for class_id in self.config.data.train_classes:
            binary_mask = (semantic_mask == class_id).astype(np.uint8)

            # Skip if mask is too small
            if binary_mask.sum() < self.config.data.min_mask_area:
                continue

            masks.append(binary_mask)
            class_ids.append(class_id)

        return masks, class_ids

    def _sample_points_from_mask(
        self, mask: np.ndarray, num_points: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random foreground points from binary mask.

        Args:
            mask: Binary mask (H, W)
            num_points: Number of points to sample (random if None)

        Returns:
            point_coords: (N, 2) array of (x, y) coordinates
            point_labels: (N,) array of labels (all 1 for foreground)
        """
        if num_points is None:
            num_points = random.randint(
                self.config.data.min_points_per_mask,
                self.config.data.max_points_per_mask
            )

        # Get foreground pixel coordinates
        fg_coords = np.argwhere(mask > 0)  # (N, 2) as (y, x)

        if len(fg_coords) == 0:
            # Return center point if mask is empty
            h, w = mask.shape
            return np.array([[w // 2, h // 2]]), np.array([1])

        # Sample random points
        num_points = min(num_points, len(fg_coords))
        indices = np.random.choice(len(fg_coords), num_points, replace=False)
        sampled = fg_coords[indices]

        # Convert from (y, x) to (x, y) format
        point_coords = sampled[:, ::-1].copy()
        point_labels = np.ones(num_points, dtype=np.int64)

        return point_coords, point_labels

    def _apply_augmentation(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Apply data augmentation to image and masks."""
        # Convert to PIL for torchvision transforms
        image_pil = Image.fromarray(image)
        mask_pils = [Image.fromarray(m * 255) for m in masks]

        # Horizontal flip
        if random.random() < self.config.data.horizontal_flip_prob:
            image_pil = TF.hflip(image_pil)
            mask_pils = [TF.hflip(m) for m in mask_pils]

        # Color jitter (only on image)
        if random.random() < self.config.data.color_jitter_prob:
            image_pil = TF.adjust_brightness(image_pil, random.uniform(0.8, 1.2))
            image_pil = TF.adjust_contrast(image_pil, random.uniform(0.8, 1.2))
            image_pil = TF.adjust_saturation(image_pil, random.uniform(0.8, 1.2))

        # Affine transform
        if random.random() < self.config.data.affine_prob:
            angle = random.uniform(-15, 15)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            scale = random.uniform(0.9, 1.1)

            image_pil = TF.affine(
                image_pil, angle, translate, scale, shear=0,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            mask_pils = [
                TF.affine(m, angle, translate, scale, shear=0,
                         interpolation=TF.InterpolationMode.NEAREST)
                for m in mask_pils
            ]

        # Convert back to numpy
        image = np.array(image_pil)
        masks = [(np.array(m) > 127).astype(np.uint8) for m in mask_pils]

        return image, masks

    def _resize_sample(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        point_coords_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Resize image, masks, and scale point coordinates."""
        orig_h, orig_w = image.shape[:2]
        target_size = self.target_size

        # Resize image
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Resize masks
        resized_masks = []
        for mask in masks:
            resized_mask = cv2.resize(
                mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
            )
            resized_masks.append(resized_mask)

        # Scale point coordinates
        scale_x = target_size / orig_w
        scale_y = target_size / orig_h

        scaled_points_list = []
        for points in point_coords_list:
            scaled_points = points.copy().astype(np.float32)
            scaled_points[:, 0] *= scale_x
            scaled_points[:, 1] *= scale_y
            scaled_points_list.append(scaled_points)

        return image, resized_masks, scaled_points_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.

        Returns dict with:
            - image: (3, H, W) tensor, normalized
            - masks: (N, H, W) tensor of binary masks
            - point_coords: (N, P, 2) tensor of point coordinates
            - point_labels: (N, P) tensor of point labels
            - class_ids: (N,) tensor of class IDs
            - original_size: (H, W) original image size
            - image_path: path to original image
        """
        sample = self.samples[idx]

        # Load image and mask
        image = self._load_image(sample["image_path"])
        semantic_mask = self._load_mask(sample["mask_path"])

        original_size = image.shape[:2]

        # Extract per-class binary masks
        masks, class_ids = self._extract_class_masks(semantic_mask)

        # If no valid masks, return a dummy sample
        if len(masks) == 0:
            return self._get_dummy_sample(image, original_size, sample["image_path"])

        # Apply augmentation
        if self.transform:
            image, masks = self._apply_augmentation(image, masks)
            # Re-filter masks that became too small after augmentation
            valid_masks = []
            valid_class_ids = []
            for mask, cls_id in zip(masks, class_ids):
                if mask.sum() >= self.config.data.min_mask_area:
                    valid_masks.append(mask)
                    valid_class_ids.append(cls_id)
            masks = valid_masks
            class_ids = valid_class_ids

            if len(masks) == 0:
                return self._get_dummy_sample(image, original_size, sample["image_path"])

        # Sample points from each mask
        point_coords_list = []
        point_labels_list = []
        for mask in masks:
            coords, labels = self._sample_points_from_mask(mask)
            point_coords_list.append(coords)
            point_labels_list.append(labels)

        # Resize everything to target size
        image, masks, point_coords_list = self._resize_sample(
            image, masks, point_coords_list
        )

        # Convert to tensors
        # Image: (H, W, 3) -> (3, H, W), normalized to [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Masks: list of (H, W) -> (N, H, W)
        masks_tensor = torch.stack([torch.from_numpy(m).float() for m in masks])

        # Pad point coordinates to same length
        max_points = max(len(coords) for coords in point_coords_list)
        padded_coords = []
        padded_labels = []
        for coords, labels in zip(point_coords_list, point_labels_list):
            pad_len = max_points - len(coords)
            if pad_len > 0:
                coords = np.pad(coords, ((0, pad_len), (0, 0)), constant_values=0)
                labels = np.pad(labels, (0, pad_len), constant_values=-1)  # -1 = ignore
            padded_coords.append(coords)
            padded_labels.append(labels)

        point_coords_tensor = torch.from_numpy(np.stack(padded_coords)).float()
        point_labels_tensor = torch.from_numpy(np.stack(padded_labels)).long()
        class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)

        return {
            "image": image_tensor,
            "masks": masks_tensor,
            "point_coords": point_coords_tensor,
            "point_labels": point_labels_tensor,
            "class_ids": class_ids_tensor,
            "original_size": torch.tensor(original_size),
            "image_path": sample["image_path"],
            "num_masks": len(masks),
        }

    def _get_dummy_sample(
        self, image: np.ndarray, original_size: Tuple[int, int], image_path: str
    ) -> Dict[str, Any]:
        """Return a dummy sample when no valid masks exist."""
        # Resize image
        image = cv2.resize(
            image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR
        )
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Create empty tensors
        return {
            "image": image_tensor,
            "masks": torch.zeros(1, self.target_size, self.target_size),
            "point_coords": torch.zeros(1, 1, 2),
            "point_labels": torch.zeros(1, 1, dtype=torch.long) - 1,
            "class_ids": torch.tensor([0], dtype=torch.long),
            "original_size": torch.tensor(original_size),
            "image_path": image_path,
            "num_masks": 0,  # Flag to skip this sample in loss computation
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable number of masks per image.

    Returns batched tensors with appropriate padding.
    """
    # Filter out samples with no valid masks
    batch = [b for b in batch if b["num_masks"] > 0]

    if len(batch) == 0:
        return None

    # Stack images
    images = torch.stack([b["image"] for b in batch])

    # For masks and points, we need to handle variable sizes
    # Option 1: Return as lists (simpler, used here)
    # Option 2: Pad to max size (more efficient for some operations)

    return {
        "images": images,
        "masks": [b["masks"] for b in batch],
        "point_coords": [b["point_coords"] for b in batch],
        "point_labels": [b["point_labels"] for b in batch],
        "class_ids": [b["class_ids"] for b in batch],
        "original_sizes": torch.stack([b["original_size"] for b in batch]),
        "image_paths": [b["image_path"] for b in batch],
        "num_masks": [b["num_masks"] for b in batch],
    }


def get_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = EndoscapesDataset(config, split="train", transform=True)
    val_dataset = EndoscapesDataset(config, split="val", transform=False)

    # Test dataset is optional
    test_dataset = None
    test_loader = None
    try:
        test_dataset = EndoscapesDataset(config, split="test", transform=False)
    except Exception as e:
        print(f"Could not load test dataset: {e}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    config = get_config()

    print("Testing EndoscapesDataset...")

    try:
        dataset = EndoscapesDataset(config, split="train")
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Masks shape: {sample['masks'].shape}")
            print(f"Point coords shape: {sample['point_coords'].shape}")
            print(f"Point labels shape: {sample['point_labels'].shape}")
            print(f"Class IDs: {sample['class_ids']}")
            print(f"Original size: {sample['original_size']}")

            # Test dataloader
            train_loader, val_loader, _ = get_dataloaders(config)
            batch = next(iter(train_loader))
            if batch is not None:
                print(f"\nBatch images shape: {batch['images'].shape}")
                print(f"Number of samples in batch: {len(batch['masks'])}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update the paths in config.py")
