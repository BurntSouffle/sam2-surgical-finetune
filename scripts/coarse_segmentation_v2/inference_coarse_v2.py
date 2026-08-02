"""
Inference Utilities for Coarse V2 (3-class) Segmentation

Provides functions to:
1. Load trained V2 model
2. Get anatomy search region from model uncertainty
3. Generate point prompts for SAM2 from search region
4. Visualize search regions

This is the key module for downstream anatomy detection with SAM2.

Usage:
    from inference_coarse_v2 import (
        load_v2_model,
        get_anatomy_search_region,
        generate_search_points,
        CoarseV2Inference
    )

    # Load model
    model = load_v2_model('checkpoints/coarse_segmentation_v2/run_xxx/best_model.pt')

    # Get search region for an image
    search_mask = get_anatomy_search_region(model, image, threshold=0.8)

    # Or use the inference class
    inference = CoarseV2Inference('path/to/best_model.pt')
    result = inference.process_image(image)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Add scripts directories to path
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_coarse_v2 import (
    COARSE_V2_CLASS_NAMES, COARSE_V2_NUM_CLASSES,
    COARSE_V2_CLASS_COLORS, ANATOMY_COLOR
)
from step4_dataset import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_v2_model(
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> nn.Module:
    """
    Load trained V2 (3-class) segmentation model.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    from model_variants import create_model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = checkpoint.get('config', {})
    decoder = config.get('decoder', 'unet')
    num_classes = checkpoint.get('num_classes', COARSE_V2_NUM_CLASSES)

    model = create_model(
        variant=decoder,
        num_classes=num_classes,
        target_size=(512, 512),
        freeze_encoder=True,
        device='cpu',
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# SEARCH REGION COMPUTATION
# =============================================================================

@torch.no_grad()
def get_anatomy_search_region(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.8,
    return_probs: bool = False,
    device: str = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get binary mask of where to search for anatomy.

    The model was trained on 3 classes (bg, gallbladder, tool) with anatomy ignored.
    Pixels where the model is uncertain (no class has high confidence) are likely anatomy.

    Args:
        model: Trained V2 model
        image: Input image, either:
               - np.ndarray: (H, W, 3) RGB uint8 or (3, H, W) normalized float
               - torch.Tensor: (1, 3, H, W) or (3, H, W) normalized
        threshold: Confidence threshold (default 0.8)
                   Lower = smaller search region (more confident predictions)
                   Higher = larger search region (more uncertainty)
        return_probs: If True, also return softmax probabilities
        device: Device to use (default: model's device)

    Returns:
        search_mask: Binary np.ndarray (H, W) where 1 = search here for anatomy
        probs: (optional) np.ndarray (H, W, 3) softmax probabilities
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocess image
    image_tensor = _preprocess_image(image, device)

    # Forward pass
    output = model(image_tensor)
    logits = output['logits']  # (1, 3, H, W)

    # Softmax probabilities
    probs = F.softmax(logits, dim=1)  # (1, 3, H, W)

    # Max probability per pixel
    max_prob, _ = probs.max(dim=1)  # (1, H, W)

    # Search region: pixels where no class is confident
    search_mask = (max_prob < threshold).squeeze(0).cpu().numpy()  # (H, W)

    if return_probs:
        probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        return search_mask, probs_np

    return search_mask


@torch.no_grad()
def get_search_region_with_methods(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.8,
    entropy_threshold: float = 0.5,
    device: str = None,
) -> Dict[str, np.ndarray]:
    """
    Get search regions using multiple methods for comparison.

    Args:
        model: Trained V2 model
        image: Input image
        threshold: Confidence threshold for max-prob method
        entropy_threshold: Entropy threshold for entropy method
        device: Device to use

    Returns:
        Dict with different search region masks:
            - 'max_prob': max(softmax) < threshold
            - 'entropy': entropy > entropy_threshold
            - 'margin': (p1 - p2) < margin_threshold (small margin = uncertain)
            - 'combined': intersection of methods (high confidence uncertain)
    """
    if device is None:
        device = next(model.parameters()).device

    image_tensor = _preprocess_image(image, device)

    output = model(image_tensor)
    logits = output['logits']
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3, H, W)

    # Method 1: Max probability threshold
    max_prob = probs.max(axis=0)
    search_max_prob = max_prob < threshold

    # Method 2: Entropy-based
    # High entropy = uncertain
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
    max_entropy = np.log(COARSE_V2_NUM_CLASSES)  # Normalize to [0, 1]
    entropy_norm = entropy / max_entropy
    search_entropy = entropy_norm > entropy_threshold

    # Method 3: Margin-based
    # Small margin between top two predictions = uncertain
    sorted_probs = np.sort(probs, axis=0)[::-1]  # Descending
    margin = sorted_probs[0] - sorted_probs[1]  # p1 - p2
    search_margin = margin < 0.3  # Margin threshold

    # Combined: intersection (pixels uncertain by multiple methods)
    search_combined = search_max_prob & search_entropy

    return {
        'max_prob': search_max_prob,
        'entropy': search_entropy,
        'margin': search_margin,
        'combined': search_combined,
        'probs': probs.transpose(1, 2, 0),  # (H, W, 3)
        'entropy_map': entropy_norm,
        'margin_map': margin,
    }


def _preprocess_image(
    image: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Preprocess image to model input format."""

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # (H, W, 3) RGB uint8
            image = image.astype(np.float32) / 255.0
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        elif image.ndim == 3 and image.shape[0] == 3:
            # (3, H, W) already normalized
            image = torch.from_numpy(image).float()
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    return image.to(device)


# =============================================================================
# POINT GENERATION FOR SAM2
# =============================================================================

def generate_search_points(
    search_mask: np.ndarray,
    num_points: int = 10,
    method: str = 'grid',
    min_distance: int = 20,
) -> np.ndarray:
    """
    Generate point coordinates within the search region for SAM2 prompts.

    Args:
        search_mask: Binary mask (H, W) where 1 = search region
        num_points: Number of points to generate
        method: Point generation method:
                - 'grid': Uniform grid sampling
                - 'random': Random sampling
                - 'centroid': Cluster centroids
                - 'distance': Maximum distance from boundaries
        min_distance: Minimum distance between points (for grid/random)

    Returns:
        points: np.ndarray of shape (N, 2) with (x, y) coordinates
    """
    if search_mask.sum() == 0:
        return np.array([]).reshape(0, 2)

    # Get coordinates of search region pixels
    y_coords, x_coords = np.where(search_mask)

    if len(x_coords) == 0:
        return np.array([]).reshape(0, 2)

    if method == 'random':
        # Random sampling
        if len(x_coords) <= num_points:
            indices = np.arange(len(x_coords))
        else:
            indices = np.random.choice(len(x_coords), num_points, replace=False)
        points = np.stack([x_coords[indices], y_coords[indices]], axis=1)

    elif method == 'grid':
        # Grid sampling with spacing
        H, W = search_mask.shape
        grid_h = np.arange(min_distance // 2, H, min_distance)
        grid_w = np.arange(min_distance // 2, W, min_distance)

        points = []
        for y in grid_h:
            for x in grid_w:
                if 0 <= y < H and 0 <= x < W and search_mask[y, x]:
                    points.append([x, y])

        points = np.array(points) if points else np.array([]).reshape(0, 2)

        # Limit to num_points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]

    elif method == 'centroid':
        # Find connected components and use centroids
        from scipy import ndimage

        labeled, num_features = ndimage.label(search_mask)
        points = []

        for i in range(1, num_features + 1):
            component = labeled == i
            y_comp, x_comp = np.where(component)
            centroid_x = int(np.mean(x_comp))
            centroid_y = int(np.mean(y_comp))
            points.append([centroid_x, centroid_y])

        points = np.array(points) if points else np.array([]).reshape(0, 2)

        if len(points) > num_points:
            # Keep largest components
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            indices = np.argsort(sizes)[::-1][:num_points]
            points = points[indices]

    elif method == 'distance':
        # Points at maximum distance from boundaries (robust to noise)
        dist_transform = cv2.distanceTransform(
            search_mask.astype(np.uint8), cv2.DIST_L2, 5
        )

        # Find local maxima
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance, min_distance))
        local_max = cv2.dilate(dist_transform, kernel) == dist_transform
        local_max &= search_mask

        y_max, x_max = np.where(local_max)

        if len(x_max) == 0:
            # Fallback to random
            return generate_search_points(search_mask, num_points, 'random', min_distance)

        # Sort by distance (largest first)
        distances = dist_transform[y_max, x_max]
        sorted_indices = np.argsort(distances)[::-1]

        points = np.stack([x_max[sorted_indices], y_max[sorted_indices]], axis=1)
        points = points[:num_points]

    else:
        raise ValueError(f"Unknown method: {method}")

    return points.astype(np.int32)


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class CoarseV2Inference:
    """
    High-level inference class for V2 coarse segmentation.

    Provides easy-to-use interface for getting search regions and predictions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        threshold: float = 0.8,
    ):
        """
        Initialize inference module.

        Args:
            checkpoint_path: Path to trained V2 model checkpoint
            device: Device to use (default: cuda if available)
            threshold: Default confidence threshold for search region
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.threshold = threshold
        self.model = load_v2_model(checkpoint_path, device)

        print(f"CoarseV2Inference initialized:")
        print(f"  Device: {device}")
        print(f"  Threshold: {threshold}")
        print(f"  Classes: {COARSE_V2_CLASS_NAMES}")

    @torch.no_grad()
    def process_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        threshold: float = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process an image and return segmentation + search region.

        Args:
            image: Input image (H, W, 3) RGB uint8 or normalized tensor
            threshold: Confidence threshold (default: self.threshold)

        Returns:
            Dict with:
                - 'prediction': (H, W) argmax predictions (0=bg, 1=gb, 2=tool)
                - 'probs': (H, W, 3) softmax probabilities
                - 'search_region': (H, W) binary mask for anatomy search
                - 'confidence': (H, W) max probability map
        """
        if threshold is None:
            threshold = self.threshold

        image_tensor = _preprocess_image(image, self.device)

        output = self.model(image_tensor)
        logits = output['logits']
        probs = F.softmax(logits, dim=1).squeeze(0)  # (3, H, W)

        prediction = probs.argmax(dim=0).cpu().numpy()  # (H, W)
        confidence = probs.max(dim=0)[0].cpu().numpy()  # (H, W)
        probs_np = probs.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        search_region = confidence < threshold

        return {
            'prediction': prediction,
            'probs': probs_np,
            'search_region': search_region,
            'confidence': confidence,
        }

    def get_search_region(
        self,
        image: Union[np.ndarray, torch.Tensor],
        threshold: float = None,
    ) -> np.ndarray:
        """Get binary search region mask."""
        return get_anatomy_search_region(
            self.model, image, threshold or self.threshold, device=self.device
        )

    def get_search_points(
        self,
        image: Union[np.ndarray, torch.Tensor],
        num_points: int = 10,
        threshold: float = None,
        method: str = 'distance',
    ) -> np.ndarray:
        """
        Get point prompts within the search region.

        Args:
            image: Input image
            num_points: Number of points to generate
            threshold: Confidence threshold
            method: Point generation method

        Returns:
            points: (N, 2) array of (x, y) coordinates
        """
        search_region = self.get_search_region(image, threshold)
        return generate_search_points(search_region, num_points, method)

    def visualize(
        self,
        image: np.ndarray,
        threshold: float = None,
        show_points: bool = True,
        num_points: int = 5,
    ) -> np.ndarray:
        """
        Create visualization of prediction and search region.

        Args:
            image: Input image (H, W, 3) RGB uint8
            threshold: Confidence threshold
            show_points: Whether to show point prompts
            num_points: Number of points to show

        Returns:
            visualization: (H, W, 3) RGB visualization
        """
        result = self.process_image(image, threshold)

        # Create prediction visualization
        pred_color = np.zeros((*result['prediction'].shape, 3), dtype=np.uint8)
        for c, color in enumerate(COARSE_V2_CLASS_COLORS):
            pred_color[result['prediction'] == c] = color

        # Overlay search region in yellow
        vis = image.copy()
        search_mask = result['search_region']
        vis[search_mask] = (0.5 * vis[search_mask] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)

        # Add points
        if show_points:
            points = generate_search_points(search_mask, num_points, method='distance')
            for pt in points:
                cv2.circle(vis, tuple(pt), 5, (255, 0, 0), -1)  # Red points
                cv2.circle(vis, tuple(pt), 5, (255, 255, 255), 1)  # White border

        return vis


# =============================================================================
# MAIN - DEMO
# =============================================================================

def main():
    """Demo of inference utilities."""
    print("\n" + "=" * 70)
    print(" COARSE V2 INFERENCE UTILITIES")
    print("=" * 70)

    print("""
This module provides utilities for using the V2 (3-class) model:

1. load_v2_model(checkpoint_path)
   Load a trained V2 model for inference.

2. get_anatomy_search_region(model, image, threshold=0.8)
   Get binary mask of uncertain regions (likely anatomy).

3. generate_search_points(search_mask, num_points, method)
   Generate point coordinates for SAM2 prompts.

4. CoarseV2Inference(checkpoint_path)
   High-level class for easy inference.

Example usage:

    # Option 1: Low-level functions
    model = load_v2_model('checkpoints/coarse_segmentation_v2/.../best_model.pt')
    search_mask = get_anatomy_search_region(model, image, threshold=0.8)
    points = generate_search_points(search_mask, num_points=10, method='distance')

    # Option 2: High-level class
    inference = CoarseV2Inference('path/to/best_model.pt', threshold=0.8)
    result = inference.process_image(image)
    points = inference.get_search_points(image, num_points=10)

The search region indicates where the model is uncertain - these are
likely anatomy pixels that can be used as prompts for SAM2.
""")


if __name__ == '__main__':
    main()
