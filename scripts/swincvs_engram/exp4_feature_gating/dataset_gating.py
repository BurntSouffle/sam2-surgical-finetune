"""
Dataset for Feature Gating Experiment

Wraps the original SwinCVS dataset to add mask generation.
Returns both RGB frames AND segmentation masks as separate tensors.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
SWINCVS_ROOT = PROJECT_ROOT / 'SwinCVS'
SAM2_FINETUNE_ROOT = PROJECT_ROOT / 'sam2_finetune'

for path in [str(PROJECT_ROOT), str(SWINCVS_ROOT), str(SWINCVS_ROOT / 'scripts'),
             str(SAM2_FINETUNE_ROOT / 'scripts')]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import original SwinCVS data loading
from scripts.f_dataset import get_datasets as get_swincvs_datasets
from scripts.f_dataset import get_dataloaders as get_swincvs_dataloaders


class MaskGenerator:
    """
    Generate GB and Tool masks using the coarse segmentation model.
    Returns soft confidence values (probabilities), not binary masks.
    """

    def __init__(self, checkpoint_path, device='cuda', cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Load coarse segmentation model
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path):
        """Load the coarse segmentation V2 model."""
        if checkpoint_path is None:
            self.inference = None
            return

        try:
            from coarse_segmentation_v2.inference_coarse_v2 import CoarseV2Inference
            self.inference = CoarseV2Inference(
                checkpoint_path=checkpoint_path,
                device=self.device,
                threshold=0.5
            )
            print(f"Loaded coarse model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load coarse model: {e}")
            self.inference = None

    def generate_masks(self, image, image_id=None):
        """
        Generate GB and Tool masks for an image.

        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image
            image_id: Optional ID for caching

        Returns:
            gb_mask: (H, W) float array, gallbladder confidence 0-1
            tool_mask: (H, W) float array, tool confidence 0-1
        """
        # Convert PIL to numpy if needed
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))

        h_orig, w_orig = image.shape[:2]

        # Check cache first
        if self.cache_dir and image_id:
            cache_path = Path(self.cache_dir) / f"{image_id}_masks.npz"
            if cache_path.exists():
                cached = np.load(cache_path)
                return cached['gb'], cached['tool']

        # Generate masks
        if self.inference is None:
            return np.zeros((h_orig, w_orig), dtype=np.float32), np.zeros((h_orig, w_orig), dtype=np.float32)

        try:
            # Resize to 512x512 for coarse model
            image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

            result = self.inference.process_image(image_resized)
            probs = result['probs']  # (512, 512, 3)

            gb_mask = probs[:, :, 1].astype(np.float32)
            tool_mask = probs[:, :, 2].astype(np.float32)

            # Resize back to original size
            gb_mask = cv2.resize(gb_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            tool_mask = cv2.resize(tool_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            # Cache if enabled
            if self.cache_dir and image_id:
                np.savez_compressed(cache_path, gb=gb_mask, tool=tool_mask)

            return gb_mask, tool_mask

        except Exception as e:
            print(f"Warning: Mask generation failed for {image_id}: {e}")
            return np.zeros((h_orig, w_orig), dtype=np.float32), np.zeros((h_orig, w_orig), dtype=np.float32)


class SwinCVSDatasetWithMasks(Dataset):
    """
    Wraps original SwinCVS dataset to add mask generation.
    Returns frames, masks, and labels.
    """

    def __init__(self, original_dataset, mask_generator, target_size=(384, 384),
                 mask_mean=(0.5, 0.5), mask_std=(0.5, 0.5)):
        self.original_dataset = original_dataset
        self.mask_generator = mask_generator
        self.target_size = target_size
        self.mask_mean = torch.tensor(mask_mean).view(2, 1, 1)
        self.mask_std = torch.tensor(mask_std).view(2, 1, 1)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get original data (images already transformed)
        images, labels = self.original_dataset[idx]
        # images: (T, C, H, W) where T=5, C=3, H=W=384
        # labels: (3,) for C1, C2, C3

        T, C, H, W = images.shape

        # Generate masks for each frame
        # We need the original image paths to generate masks
        sequence_info = self.original_dataset.image_dataframe.iloc[idx]
        paths = [sequence_info[f'f{i}'] for i in range(5)]

        masks_list = []
        for i, path in enumerate(paths):
            # Load original image for mask generation
            try:
                pil_image = Image.open(path).convert('RGB')
                image_np = np.array(pil_image)

                # Get image ID for caching
                image_id = Path(path).stem

                # Generate masks
                gb_mask, tool_mask = self.mask_generator.generate_masks(image_np, image_id)

                # Apply same center crop as original
                h_orig, w_orig = image_np.shape[:2]
                crop_size = min(h_orig, w_orig)
                top = (h_orig - crop_size) // 2
                left = (w_orig - crop_size) // 2

                gb_cropped = gb_mask[top:top+crop_size, left:left+crop_size]
                tool_cropped = tool_mask[top:top+crop_size, left:left+crop_size]

                # Resize to target size
                gb_resized = cv2.resize(gb_cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
                tool_resized = cv2.resize(tool_cropped, self.target_size, interpolation=cv2.INTER_LINEAR)

                # Stack masks
                mask = np.stack([gb_resized, tool_resized], axis=0)  # (2, H, W)
                masks_list.append(torch.from_numpy(mask).float())

            except Exception as e:
                print(f"Warning: Failed to generate mask for {path}: {e}")
                masks_list.append(torch.zeros(2, H, W))

        # Stack all masks
        masks = torch.stack(masks_list, dim=0)  # (T, 2, H, W)

        # Normalize masks to [-1, 1]
        masks = (masks - self.mask_mean) / self.mask_std

        return {
            'frames': images,  # (T, 3, H, W)
            'masks': masks,    # (T, 2, H, W)
            'labels': labels.float()  # (3,)
        }


def get_gating_datasets(config, mask_generator):
    """
    Create datasets for feature gating experiment.
    Uses original SwinCVS data loading, wraps with mask generation.
    """
    # Get original SwinCVS datasets
    print("Loading original SwinCVS datasets...")
    train_dataset, val_dataset, test_dataset = get_swincvs_datasets(config)

    print(f"Original dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Wrap with mask generation
    train_with_masks = SwinCVSDatasetWithMasks(train_dataset, mask_generator)
    val_with_masks = SwinCVSDatasetWithMasks(val_dataset, mask_generator)
    test_with_masks = SwinCVSDatasetWithMasks(test_dataset, mask_generator)

    return train_with_masks, val_with_masks, test_with_masks


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=2, num_workers=0):
    """Create DataLoaders with custom collate for dict outputs."""

    def collate_fn(batch):
        frames = torch.stack([b['frames'] for b in batch])
        masks = torch.stack([b['masks'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])
        return {'frames': frames, 'masks': masks, 'labels': labels}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Testing dataset loading...")

    # Need a config object
    import yaml

    class Config:
        pass

    config = Config()
    config.DATASET_DIR = str(PROJECT_ROOT)
    config.SEED = 42
    config.MODEL = Config()
    config.MODEL.LSTM = True
    config.TRAIN = Config()
    config.TRAIN.LIMIT_DATA_FRACTION = 100  # Use 1% of data
    config.TRAIN.BATCH_SIZE = 2
    config.TRAIN.TRANSFORMS = Config()
    config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN = [0.485, 0.456, 0.406]
    config.TRAIN.TRANSFORMS.ENDOSCAPES_STD = [0.229, 0.224, 0.225]
    config.TRAIN.TRANSFORMS.CENTER_CROP = 480

    # Create mask generator
    checkpoint_path = SAM2_FINETUNE_ROOT / 'checkpoints' / 'coarse_segmentation_v2' / 'run_20260120_234049' / 'best_model.pt'

    if checkpoint_path.exists():
        mask_gen = MaskGenerator(str(checkpoint_path), device='cuda')
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        mask_gen = MaskGenerator(None, device='cuda')

    try:
        train_ds, val_ds, test_ds = get_gating_datasets(config, mask_gen)
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_ds)}")
        print(f"  Val: {len(val_ds)}")
        print(f"  Test: {len(test_ds)}")

        # Get one sample
        sample = train_ds[0]
        print(f"\nSample shapes:")
        print(f"  frames: {sample['frames'].shape}")
        print(f"  masks: {sample['masks'].shape}")
        print(f"  labels: {sample['labels'].shape}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
