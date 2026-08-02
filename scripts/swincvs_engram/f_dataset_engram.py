"""
E1 Experiment: Modified Dataset for Visual Engrams

This module extends the original SwinCVS dataset to generate 5-channel inputs:
- Channels 0-2: RGB image (ImageNet normalized)
- Channel 3: Gallbladder mask (from coarse segmentation model)
- Channel 4: Tool mask (from coarse segmentation model)

The coarse segmentation model is loaded once and used to generate masks on-the-fly
for each frame in the 5-frame sequences.

Usage:
    from f_dataset_engram import get_engram_datasets, get_dataloaders

    training_dataset, val_dataset, test_dataset = get_engram_datasets(config)
    train_loader, val_loader, test_loader = get_dataloaders(config, ...)
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent  # sam2_finetune
DISSERTATION_ROOT = PROJECT_ROOT.parent
SWINCVS_ROOT = DISSERTATION_ROOT / 'SwinCVS'

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))

# Fix SAM2 import path BEFORE importing coarse model (which uses SAM2)
try:
    from sam2_path_fix import fix_sam2_path, verify_sam2_import
    fix_sam2_path(verbose=False)
except ImportError:
    pass  # Path fix module not available, continue anyway

# Import original dataset utilities
from scripts.f_dataset import (
    check_dataset,
    get_three_dataframes,
    get_dataframe,
    get_frame_sequence_dataframe,
    update_dataframe,
    add_unlabelled_imgs,
    generate_path,
    get_class,
    get_endoscapes_mean_std,
)

# Import coarse model for engram generation
try:
    from coarse_segmentation_v2.inference_coarse_v2 import (
        CoarseV2Inference,
        load_v2_model,
        _preprocess_image,
    )
    COARSE_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Could not import coarse segmentation model: {e}")
    print("[WARN] Engram mode will use zero masks.")
    COARSE_MODEL_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default mask normalization (binary masks centered at 0.5)
DEFAULT_MASK_MEAN = [0.5, 0.5]  # [GB, Tool]
DEFAULT_MASK_STD = [0.5, 0.5]

# Coarse model settings
COARSE_INPUT_SIZE = 512
SWINCVS_INPUT_SIZE = 384


# =============================================================================
# ENGRAM MASK GENERATOR
# =============================================================================

class EngramMaskGenerator:
    """
    Generates GB and Tool masks from RGB images using the coarse segmentation model.

    This class wraps the coarse model and provides efficient mask generation
    with optional caching for repeated access to the same images.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        threshold: float = 0.5,
        cache_enabled: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the mask generator.

        Args:
            checkpoint_path: Path to coarse segmentation model checkpoint
            device: Device to run inference on (default: cuda if available)
            threshold: Confidence threshold for mask generation
            cache_enabled: Whether to cache generated masks to disk
            cache_dir: Directory for mask cache (if enabled)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.threshold = threshold
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Load coarse model
        if COARSE_MODEL_AVAILABLE:
            print(f"\n[EngramMaskGenerator] Loading coarse model...")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Device: {device}")

            self.model = load_v2_model(checkpoint_path, device)
            self.model.eval()
            self._model_loaded = True
            print(f"  Model loaded successfully!")
        else:
            print(f"\n[EngramMaskGenerator] Coarse model not available. Using zero masks.")
            self.model = None
            self._model_loaded = False

        # Setup cache directory if enabled
        if self.cache_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Cache enabled: {self.cache_dir}")

    def _get_cache_path(self, image_path: str) -> Path:
        """Get cache file path for an image."""
        if not self.cache_dir:
            return None
        # Use image filename as cache key
        img_name = Path(image_path).stem
        return self.cache_dir / f"{img_name}_masks.npz"

    def _load_from_cache(self, image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Try to load masks from cache."""
        cache_path = self._get_cache_path(image_path)
        if cache_path and cache_path.exists():
            try:
                data = np.load(cache_path)
                return data['gb_mask'], data['tool_mask']
            except Exception:
                return None
        return None

    def _save_to_cache(self, image_path: str, gb_mask: np.ndarray, tool_mask: np.ndarray):
        """Save masks to cache."""
        cache_path = self._get_cache_path(image_path)
        if cache_path:
            try:
                np.savez_compressed(cache_path, gb_mask=gb_mask, tool_mask=tool_mask)
            except Exception:
                pass

    @torch.no_grad()
    def generate_masks(
        self,
        image: Union[np.ndarray, str, Path],
        target_size: Tuple[int, int] = (SWINCVS_INPUT_SIZE, SWINCVS_INPUT_SIZE),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GB and Tool masks for an image.

        Args:
            image: RGB image as numpy array (H, W, 3) or path to image file
            target_size: Output mask size (H, W)

        Returns:
            gb_mask: Binary gallbladder mask (H, W), float32 in [0, 1]
            tool_mask: Binary tool mask (H, W), float32 in [0, 1]
        """
        # Handle image path input
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)

            # Try cache first
            if self.cache_enabled:
                cached = self._load_from_cache(image_path)
                if cached is not None:
                    gb_mask, tool_mask = cached
                    # Resize if needed
                    if gb_mask.shape != target_size:
                        gb_mask = cv2.resize(gb_mask, (target_size[1], target_size[0]),
                                            interpolation=cv2.INTER_NEAREST)
                        tool_mask = cv2.resize(tool_mask, (target_size[1], target_size[0]),
                                              interpolation=cv2.INTER_NEAREST)
                    return gb_mask.astype(np.float32), tool_mask.astype(np.float32)

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARN] Could not load image: {image_path}")
                return self._zero_masks(target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Return zero masks if model not available
        if not self._model_loaded:
            return self._zero_masks(target_size)

        try:
            # Resize to coarse model input size
            image_resized = cv2.resize(image, (COARSE_INPUT_SIZE, COARSE_INPUT_SIZE))

            # Preprocess and run inference
            image_tensor = _preprocess_image(image_resized, self.device)
            output = self.model(image_tensor)
            logits = output['logits']  # (1, 3, 512, 512)

            # Get predictions
            probs = F.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (512, 512)

            # Extract binary masks
            # Class 0 = background, 1 = gallbladder, 2 = tool
            gb_mask = (prediction == 1).astype(np.float32)
            tool_mask = (prediction == 2).astype(np.float32)

            # Optional: apply confidence threshold
            if self.threshold > 0.5:
                confidence = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
                low_conf = confidence < self.threshold
                gb_mask[low_conf & (prediction == 1)] = 0
                tool_mask[low_conf & (prediction == 2)] = 0

            # Resize to target size
            if gb_mask.shape != target_size:
                gb_mask = cv2.resize(gb_mask, (target_size[1], target_size[0]),
                                    interpolation=cv2.INTER_NEAREST)
                tool_mask = cv2.resize(tool_mask, (target_size[1], target_size[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # Cache if enabled
            if self.cache_enabled and image_path:
                self._save_to_cache(image_path, gb_mask, tool_mask)

            return gb_mask, tool_mask

        except Exception as e:
            print(f"[WARN] Mask generation failed: {e}")
            return self._zero_masks(target_size)

    def _zero_masks(self, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero masks as fallback."""
        zeros = np.zeros(target_size, dtype=np.float32)
        return zeros.copy(), zeros.copy()


# =============================================================================
# ENGRAM DATASET CLASSES
# =============================================================================

class EndoscapesEngram_Dataset(Dataset):
    """
    Dataset for SwinV2 backbone training with engram masks (non-LSTM version).
    Returns 5-channel inputs: [RGB + GB mask + Tool mask]
    """

    def __init__(
        self,
        image_dataframe: pd.DataFrame,
        transform_sequence,
        engram_config: Optional[Dict] = None,
        mask_generator: Optional[EngramMaskGenerator] = None,
    ):
        """
        Args:
            image_dataframe: DataFrame with 'path' and 'classification' columns
            transform_sequence: torchvision transforms for RGB
            engram_config: Configuration dict with mask settings
            mask_generator: Shared EngramMaskGenerator instance
        """
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        self.engram_config = engram_config or {}
        self.mask_generator = mask_generator

        # Mask normalization
        self.mask_mean = self.engram_config.get('MASK_MEAN', DEFAULT_MASK_MEAN)
        self.mask_std = self.engram_config.get('MASK_STD', DEFAULT_MASK_STD)
        self.engram_enabled = self.engram_config.get('ENABLED', False)

        # RGB normalization mode (same as LSTM version)
        self.normalize_01 = self.engram_config.get('NORMALIZE_01', False)

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        image_info = self.image_dataframe.iloc[idx]
        image_path = image_info['path']
        label = torch.tensor(image_info['classification'])

        # Load RGB image
        image = Image.open(image_path)

        # Apply transforms to RGB
        if self.transforms:
            image = self.transforms(image)

            # Optional [0,1] rescaling (for backward compatibility)
            # Default: OFF - use ImageNet normalization
            if self.normalize_01:
                image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image) + 1e-8)

        # Generate and concatenate masks if engram enabled
        if self.engram_enabled and self.mask_generator is not None:
            # Get target size from transformed image
            _, h, w = image.shape

            # Generate masks
            gb_mask, tool_mask = self.mask_generator.generate_masks(
                image_path, target_size=(h, w)
            )

            # Normalize masks
            gb_mask = (gb_mask - self.mask_mean[0]) / self.mask_std[0]
            tool_mask = (tool_mask - self.mask_mean[1]) / self.mask_std[1]

            # Convert to tensors and concatenate
            gb_tensor = torch.from_numpy(gb_mask).unsqueeze(0).float()
            tool_tensor = torch.from_numpy(tool_mask).unsqueeze(0).float()

            # Concatenate: [3 RGB, 1 GB, 1 Tool] = 5 channels
            image = torch.cat([image, gb_tensor, tool_tensor], dim=0)

        return image, label


class EndoscapesSwinCVS_Engram_Dataset(Dataset):
    """
    Dataset for SwinCVS (LSTM) training with engram masks.
    Returns 5-channel inputs for 5-frame sequences: [RGB + GB mask + Tool mask]

    This is the main dataset class for E1 experiment.
    """

    def __init__(
        self,
        image_dataframe: pd.DataFrame,
        transform_sequence,
        engram_config: Optional[Dict] = None,
        mask_generator: Optional[EngramMaskGenerator] = None,
    ):
        """
        Args:
            image_dataframe: DataFrame with f0-f4 paths and classification
            transform_sequence: torchvision transforms for RGB
            engram_config: Configuration dict with mask settings
            mask_generator: Shared EngramMaskGenerator instance (loaded once)
        """
        self.image_dataframe = image_dataframe
        self.transforms = transform_sequence
        self.engram_config = engram_config or {}
        self.mask_generator = mask_generator

        # Mask normalization
        self.mask_mean = self.engram_config.get('MASK_MEAN', DEFAULT_MASK_MEAN)
        self.mask_std = self.engram_config.get('MASK_STD', DEFAULT_MASK_STD)
        self.engram_enabled = self.engram_config.get('ENABLED', False)

        # RGB normalization mode:
        # - False (default): Use ImageNet normalization from transforms (CORRECT for pretrained weights)
        # - True: Rescale to [0,1] after transforms (matches original SwinCVS but suboptimal)
        self.normalize_01 = self.engram_config.get('NORMALIZE_01', False)

        # Target size for masks (after transforms)
        self.target_size = (SWINCVS_INPUT_SIZE, SWINCVS_INPUT_SIZE)

        if self.engram_enabled:
            print(f"\n[EndoscapesSwinCVS_Engram_Dataset]")
            print(f"  Engram enabled: {self.engram_enabled}")
            print(f"  Mask mean: {self.mask_mean}")
            print(f"  Mask std: {self.mask_std}")
            print(f"  Normalize [0,1]: {self.normalize_01} {'(matches original SwinCVS)' if self.normalize_01 else '(ImageNet normalized - RECOMMENDED)'}")
            print(f"  Output channels: 5 (RGB + GB + Tool)")

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        sequence_info = self.image_dataframe.iloc[idx]

        # Get paths for all 5 frames
        paths = [
            sequence_info['f0'],
            sequence_info['f1'],
            sequence_info['f2'],
            sequence_info['f3'],
            sequence_info['f4'],
        ]

        image_list = []

        # Use same random seed for all frames in sequence (consistent augmentation)
        seed = random.randint(0, 2**32)

        for path in paths:
            # Load RGB image
            image = Image.open(path)

            # Apply transforms with consistent randomness
            if self.transforms:
                torch.manual_seed(seed)
                random.seed(seed)
                image = self.transforms(image)

                # Optional [0,1] rescaling (for backward compatibility with original SwinCVS)
                # Default: OFF - use ImageNet normalization for proper pretrained weight usage
                if self.normalize_01:
                    image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image) + 1e-8)

            # Generate and concatenate masks if engram enabled
            if self.engram_enabled and self.mask_generator is not None:
                # Get target size from transformed image
                _, h, w = image.shape

                # Generate masks
                gb_mask, tool_mask = self.mask_generator.generate_masks(
                    path, target_size=(h, w)
                )

                # Normalize masks (same normalization as RGB uses mean/std)
                gb_mask = (gb_mask - self.mask_mean[0]) / self.mask_std[0]
                tool_mask = (tool_mask - self.mask_mean[1]) / self.mask_std[1]

                # Convert to tensors
                gb_tensor = torch.from_numpy(gb_mask).unsqueeze(0).float()
                tool_tensor = torch.from_numpy(tool_mask).unsqueeze(0).float()

                # Concatenate: [3 RGB, 1 GB, 1 Tool] = 5 channels
                image = torch.cat([image, gb_tensor, tool_tensor], dim=0)

            image_list.append(image)

        # Stack all frames: (5, C, H, W) where C=3 or 5
        images = torch.stack(image_list)
        label = torch.tensor(sequence_info['classification'])

        return images, label


# =============================================================================
# DATASET FACTORY FUNCTIONS
# =============================================================================

def get_engram_datasets(config):
    """
    Create dataset instances with engram mask generation.

    This is the main entry point for E1 experiment datasets.
    Replaces get_datasets() from original f_dataset.py.

    Args:
        config: Configuration object with ENGRAM settings

    Returns:
        training_dataset, val_dataset, test_dataset
    """
    print("\n" + "="*70)
    print("LOADING ENGRAM DATASETS")
    print("="*70)

    # Check/download dataset
    dataset_dir = check_dataset(config)
    print(f"\nDataset loaded from: {dataset_dir}")

    # Get dataframes
    train_df, val_df, test_df = get_three_dataframes(dataset_dir, lstm=config.MODEL.LSTM)

    # Get transforms
    transform_sequence = get_transform_sequence(config)

    # Check if engram is enabled
    engram_enabled = hasattr(config, 'ENGRAM') and getattr(config.ENGRAM, 'ENABLED', False)

    # Initialize mask generator if engram enabled
    mask_generator = None
    engram_config = {}

    if engram_enabled:
        print(f"\n[ENGRAM MODE ENABLED]")

        # Get engram configuration
        engram_config = {
            'ENABLED': True,
            'MASK_MEAN': getattr(config.ENGRAM, 'MASK_MEAN', DEFAULT_MASK_MEAN),
            'MASK_STD': getattr(config.ENGRAM, 'MASK_STD', DEFAULT_MASK_STD),
        }

        # Get checkpoint path
        checkpoint_path = getattr(config.ENGRAM, 'CHECKPOINT_PATH', None)
        if checkpoint_path:
            # Handle relative paths
            if not Path(checkpoint_path).is_absolute():
                # Try relative to sam2_finetune
                sam2_finetune_root = SCRIPTS_ROOT.parent
                checkpoint_path = sam2_finetune_root / checkpoint_path

            print(f"  Checkpoint: {checkpoint_path}")

            # Get cache settings
            cache_enabled = getattr(config.ENGRAM, 'CACHE_MASKS', False)
            cache_dir = getattr(config.ENGRAM, 'CACHE_DIR', None)
            if cache_dir is None and cache_enabled:
                cache_dir = dataset_dir / 'engram_cache'

            # Get threshold
            threshold = getattr(config.ENGRAM, 'MASK_THRESHOLD', 0.5)

            # Initialize mask generator (shared across all datasets)
            mask_generator = EngramMaskGenerator(
                checkpoint_path=str(checkpoint_path),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                threshold=threshold,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
            )
        else:
            print(f"  [WARN] No checkpoint path specified. Using zero masks.")
    else:
        print(f"\n[STANDARD MODE - 3 channel RGB]")

    # Apply data fraction limit
    fraction = config.TRAIN.LIMIT_DATA_FRACTION
    train_df = train_df[::fraction]
    val_df = val_df[::fraction]
    test_df = test_df[::fraction]

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Create datasets based on model type
    if config.MODEL.LSTM:
        # SwinCVS with LSTM (5-frame sequences)
        training_dataset = EndoscapesSwinCVS_Engram_Dataset(
            train_df, transform_sequence, engram_config, mask_generator
        )
        val_dataset = EndoscapesSwinCVS_Engram_Dataset(
            val_df, transform_sequence, engram_config, mask_generator
        )
        test_dataset = EndoscapesSwinCVS_Engram_Dataset(
            test_df, transform_sequence, engram_config, mask_generator
        )
    else:
        # SwinV2 backbone only (single frames)
        training_dataset = EndoscapesEngram_Dataset(
            train_df, transform_sequence, engram_config, mask_generator
        )
        val_dataset = EndoscapesEngram_Dataset(
            val_df, transform_sequence, engram_config, mask_generator
        )
        test_dataset = EndoscapesEngram_Dataset(
            test_df, transform_sequence, engram_config, mask_generator
        )

    print(f"\nOutput tensor shape per frame: ", end="")
    if engram_enabled:
        print(f"(5, 384, 384) - [RGB + GB + Tool]")
    else:
        print(f"(3, 384, 384) - [RGB only]")

    print("="*70 + "\n")

    return training_dataset, val_dataset, test_dataset


def get_dataloaders(config, training_dataset, val_dataset, test_dataset):
    """
    Create dataloaders from dataset instances.

    Same as original but with num_workers=0 for Windows compatibility
    when using CUDA in the mask generator.
    """
    print(f"Creating dataloaders with batch_size={config.TRAIN.BATCH_SIZE}")

    # Note: num_workers=0 required when using CUDA in dataset __getitem__
    # because CUDA tensors can't be shared across processes
    num_workers = 0

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_transform_sequence(config):
    """
    Get transform sequence for RGB images.

    Same as original - transforms are applied to RGB only.
    Masks are generated at the target size directly.
    """
    mean = config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN
    std = config.TRAIN.TRANSFORMS.ENDOSCAPES_STD

    # Note: Original code reverses for BGR->RGB but ENDOSCAPES stats are already RGB
    # Keeping same behavior as original
    mean = mean[::-1]
    std = std[::-1]

    transform_sequence = transforms.Compose([
        transforms.CenterCrop(config.TRAIN.TRANSFORMS.CENTER_CROP),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ])

    return transform_sequence


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def visualize_engram_sample(dataset, idx: int = 0, save_path: Optional[str] = None):
    """
    Visualize a sample from the engram dataset.

    Args:
        dataset: EndoscapesSwinCVS_Engram_Dataset instance
        idx: Sample index
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    images, label = dataset[idx]

    # images shape: (5, C, H, W) where C=3 or 5
    n_frames = images.shape[0]
    n_channels = images.shape[1]

    fig, axes = plt.subplots(3, n_frames, figsize=(15, 9))

    for i in range(n_frames):
        frame = images[i]

        # RGB (first 3 channels)
        rgb = frame[:3].permute(1, 2, 0).numpy()
        # Denormalize for visualization
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'Frame {i} - RGB')
        axes[0, i].axis('off')

        if n_channels >= 4:
            # GB mask (channel 3)
            gb = frame[3].numpy()
            gb = (gb - gb.min()) / (gb.max() - gb.min() + 1e-8)
            axes[1, i].imshow(gb, cmap='magenta')
            axes[1, i].set_title(f'Frame {i} - GB')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')

        if n_channels >= 5:
            # Tool mask (channel 4)
            tool = frame[4].numpy()
            tool = (tool - tool.min()) / (tool.max() - tool.min() + 1e-8)
            axes[2, i].imshow(tool, cmap='cyan')
            axes[2, i].set_title(f'Frame {i} - Tool')
            axes[2, i].axis('off')
        else:
            axes[2, i].axis('off')

    plt.suptitle(f'Sample {idx} - Label: {label.tolist()}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def test_engram_dataset():
    """Test the engram dataset with a mock config."""
    print("\n" + "#"*70)
    print("# TESTING ENGRAM DATASET")
    print("#"*70)

    # Create mock config
    class MockEngram:
        ENABLED = True
        CHECKPOINT_PATH = 'checkpoints/coarse_segmentation_v2/run_20260120_234049/best_model.pt'
        MASK_MEAN = [0.5, 0.5]
        MASK_STD = [0.5, 0.5]
        MASK_THRESHOLD = 0.5
        CACHE_MASKS = False
        CACHE_DIR = None

    class MockTransforms:
        ENDOSCAPES_MEAN = [0.485, 0.456, 0.406]
        ENDOSCAPES_STD = [0.229, 0.224, 0.225]
        CENTER_CROP = 480

    class MockTrain:
        TRANSFORMS = MockTransforms()
        BATCH_SIZE = 4
        LIMIT_DATA_FRACTION = 1

    class MockModel:
        LSTM = True

    class MockConfig:
        ENGRAM = MockEngram()
        TRAIN = MockTrain()
        MODEL = MockModel()
        DATASET_DIR = None

    print("\nMock config created.")
    print("Note: Full test requires dataset and checkpoint files.")
    print("Run from project root with data available.\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    test_engram_dataset()
