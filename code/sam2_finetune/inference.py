"""
Inference script for generating synthetic masks with fine-tuned SAM2.

This script generates semantic segmentation masks for unlabeled frames
using the fine-tuned SAM2 model as a "teacher" for pseudo-labeling.

Supports multiple prompting strategies:
1. Automatic mask generation (grid points)
2. Point prompts from existing masks (for label propagation)
3. Box prompts (if bounding boxes available)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add SAM2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from config import Config, get_config


class SAM2Inferencer:
    """SAM2 inference for mask generation."""

    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = build_sam2(
            config_file=config.model.model_config,
            ckpt_path=str(config.paths.pretrained_dir / config.model.checkpoint_filename),
            device=self.device,
        )

        # Load fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Create predictors
        self.predictor = SAM2ImagePredictor(self.model)
        self.auto_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # Class names
        self.class_names = config.data.class_names
        self.num_classes = config.data.num_classes

    def generate_masks_auto(
        self,
        image: np.ndarray,
    ) -> List[Dict]:
        """
        Generate masks automatically using grid point sampling.

        Returns list of mask dictionaries with:
        - segmentation: binary mask
        - area: mask area
        - bbox: bounding box [x, y, w, h]
        - predicted_iou: model's IoU prediction
        - stability_score: mask stability score
        """
        masks = self.auto_generator.generate(image)
        return masks

    def generate_masks_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks using point prompts.

        Args:
            image: RGB image (H, W, 3)
            point_coords: (N, 2) array of (x, y) coordinates
            point_labels: (N,) array of labels (1=foreground, 0=background)

        Returns:
            masks: (K, H, W) binary masks
            scores: (K,) IoU scores
            logits: (K, H, W) raw logits
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        return masks, scores, logits

    def generate_masks_with_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks using box prompt.

        Args:
            image: RGB image (H, W, 3)
            box: (4,) array [x1, y1, x2, y2]

        Returns:
            masks: (K, H, W) binary masks
            scores: (K,) IoU scores
            logits: (K, H, W) raw logits
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=True,
        )

        return masks, scores, logits

    @torch.no_grad()
    def generate_semantic_masks(
        self,
        image: np.ndarray,
        reference_mask: Optional[np.ndarray] = None,
        points_per_class: int = 5,
    ) -> np.ndarray:
        """
        Generate semantic segmentation mask for an image.

        If reference_mask is provided, use it to sample point prompts
        for each class (label propagation mode).

        If no reference_mask, use automatic mask generation and
        try to classify masks based on learned features.

        Args:
            image: RGB image (H, W, 3)
            reference_mask: Optional semantic mask to sample points from
            points_per_class: Number of points to sample per class

        Returns:
            semantic_mask: (H, W) semantic segmentation mask
        """
        h, w = image.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)

        if reference_mask is not None:
            # Label propagation mode: use reference mask to generate prompts
            self.predictor.set_image(image)

            for class_id in self.config.data.train_classes:
                class_mask = (reference_mask == class_id)

                if class_mask.sum() < self.config.data.min_mask_area:
                    continue

                # Sample points from the class region
                fg_coords = np.argwhere(class_mask)  # (N, 2) as (y, x)

                if len(fg_coords) == 0:
                    continue

                # Random sample
                num_points = min(points_per_class, len(fg_coords))
                indices = np.random.choice(len(fg_coords), num_points, replace=False)
                sampled = fg_coords[indices]

                # Convert to (x, y) format
                point_coords = sampled[:, ::-1].copy()
                point_labels = np.ones(num_points, dtype=np.int64)

                # Predict
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )

                # Take best mask
                pred_mask = masks[0]

                # Add to semantic mask (later classes overwrite earlier)
                semantic_mask[pred_mask > 0] = class_id

        else:
            # Automatic mode: generate all masks and assign based on area/position
            # This is a simplified heuristic - in practice you'd want
            # a classifier to assign classes to masks
            auto_masks = self.generate_masks_auto(image)

            # Sort by area (largest first)
            auto_masks.sort(key=lambda x: x["area"], reverse=True)

            # Assign masks to semantic mask
            # (This is a placeholder - real assignment would need a classifier)
            for i, mask_data in enumerate(auto_masks):
                mask = mask_data["segmentation"]
                # Simple heuristic: assign based on index (not reliable!)
                # In practice, you'd train a classifier or use other features
                class_id = (i % (self.num_classes - 1)) + 1  # Cycle through classes
                semantic_mask[mask > 0] = class_id

        return semantic_mask

    def process_video_frames(
        self,
        image_dir: Path,
        output_dir: Path,
        reference_masks_dir: Optional[Path] = None,
        file_pattern: str = "*.jpg",
        save_visualizations: bool = False,
    ) -> Dict[str, any]:
        """
        Process all frames in a directory.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save generated masks
            reference_masks_dir: Optional directory with reference masks for prompting
            file_pattern: Glob pattern for image files
            save_visualizations: Whether to save overlay visualizations

        Returns:
            Statistics about processing
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_visualizations:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

        # Get image files
        image_files = sorted(image_dir.glob(file_pattern))
        print(f"Found {len(image_files)} images to process")

        stats = {
            "total_images": len(image_files),
            "processed": 0,
            "with_reference": 0,
            "auto_generated": 0,
        }

        for img_path in tqdm(image_files, desc="Processing frames"):
            # Load image
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Check for reference mask
            reference_mask = None
            if reference_masks_dir:
                mask_path = reference_masks_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    stats["with_reference"] += 1
                else:
                    stats["auto_generated"] += 1
            else:
                stats["auto_generated"] += 1

            # Generate mask
            semantic_mask = self.generate_semantic_masks(
                image,
                reference_mask=reference_mask,
            )

            # Save mask
            mask_path = output_dir / (img_path.stem + ".png")
            cv2.imwrite(str(mask_path), semantic_mask)

            # Save visualization
            if save_visualizations:
                vis_path = vis_dir / (img_path.stem + "_vis.png")
                self._save_visualization(image, semantic_mask, vis_path)

            stats["processed"] += 1

        return stats

    def _save_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        save_path: Path,
    ):
        """Save overlay visualization."""
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Mask
        axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=self.num_classes - 1)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        # Overlay
        overlay = image.copy()
        colors = cm.tab10(np.linspace(0, 1, self.num_classes))[:, :3] * 255

        for class_id in range(1, self.num_classes):
            class_mask = mask == class_id
            if class_mask.any():
                overlay[class_mask] = (
                    overlay[class_mask] * 0.5 + colors[class_id] * 0.5
                ).astype(np.uint8)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAM2 Inference for Mask Generation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save generated masks"
    )
    parser.add_argument(
        "--reference-dir", type=str, default=None,
        help="Optional directory with reference masks for prompting"
    )
    parser.add_argument(
        "--project-store", type=str,
        help="Project store path"
    )
    parser.add_argument(
        "--dataset-root", type=str,
        help="Dataset root path"
    )
    parser.add_argument(
        "--pattern", type=str, default="*.jpg",
        help="Glob pattern for image files"
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    # Create config
    config_kwargs = {}
    if args.project_store:
        config_kwargs["project_store"] = args.project_store
    if args.dataset_root:
        config_kwargs["dataset_root"] = args.dataset_root

    config = get_config(**config_kwargs)

    # Create inferencer
    inferencer = SAM2Inferencer(config, args.checkpoint, args.device)

    # Process images
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    stats = inferencer.process_video_frames(
        image_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        reference_masks_dir=reference_dir,
        file_pattern=args.pattern,
        save_visualizations=args.save_vis,
    )

    # Print stats
    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print("=" * 50)
    print(f"Total images: {stats['total_images']}")
    print(f"Processed: {stats['processed']}")
    print(f"With reference masks: {stats['with_reference']}")
    print(f"Auto-generated: {stats['auto_generated']}")
    print(f"\nMasks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
