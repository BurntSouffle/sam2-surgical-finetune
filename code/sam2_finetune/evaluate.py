"""
Evaluation script for fine-tuned SAM2 model on Endoscapes2023.

Computes:
- mIoU (mean Intersection over Union)
- Per-class IoU breakdown
- Dice coefficient
- Precision and Recall
- Visualizations of predictions vs ground truth
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add SAM2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything-2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import Config, get_config
from dataset import EndoscapesDataset


class Evaluator:
    """Evaluator for SAM2 on Endoscapes."""

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

        # Create predictor
        self.predictor = SAM2ImagePredictor(self.model)

        # Class names for reporting
        self.class_names = config.data.class_names
        self.train_classes = config.data.train_classes

    def compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics between prediction and ground truth.

        Args:
            pred: Predicted logits or probabilities
            gt: Ground truth binary mask

        Returns:
            Dictionary of metrics
        """
        # Binarize prediction
        if pred.max() > 1 or pred.min() < 0:
            pred = 1 / (1 + np.exp(-pred))  # Sigmoid
        pred_binary = (pred > threshold).astype(np.float32)

        gt = gt.astype(np.float32)

        # Compute metrics
        intersection = (pred_binary * gt).sum()
        union = pred_binary.sum() + gt.sum() - intersection
        pred_sum = pred_binary.sum()
        gt_sum = gt.sum()

        # IoU
        iou = intersection / (union + 1e-8) if union > 0 else 1.0 if gt_sum == 0 else 0.0

        # Dice
        dice = (2 * intersection) / (pred_sum + gt_sum + 1e-8) if (pred_sum + gt_sum) > 0 else 1.0 if gt_sum == 0 else 0.0

        # Precision and Recall
        precision = intersection / (pred_sum + 1e-8) if pred_sum > 0 else 1.0 if gt_sum == 0 else 0.0
        recall = intersection / (gt_sum + 1e-8) if gt_sum > 0 else 1.0

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: EndoscapesDataset,
        num_samples: Optional[int] = None,
        save_visualizations: bool = False,
        vis_dir: Optional[Path] = None,
    ) -> Dict[str, any]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: EndoscapesDataset instance
            num_samples: Limit evaluation to this many samples (None for all)
            save_visualizations: Whether to save visualization images
            vis_dir: Directory to save visualizations

        Returns:
            Dictionary containing overall and per-class metrics
        """
        # Initialize metric accumulators
        all_metrics = {
            "iou": [],
            "dice": [],
            "precision": [],
            "recall": [],
        }
        class_metrics = {cls: {k: [] for k in all_metrics.keys()} for cls in self.train_classes}

        # Setup visualization directory
        if save_visualizations and vis_dir:
            vis_dir = Path(vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)

        num_samples = num_samples or len(dataset)
        num_samples = min(num_samples, len(dataset))

        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[idx]

            if sample["num_masks"] == 0:
                continue

            # Get data
            image = sample["image"]  # (3, H, W)
            gt_masks = sample["masks"]  # (N, H, W)
            point_coords = sample["point_coords"]  # (N, P, 2)
            point_labels = sample["point_labels"]  # (N, P)
            class_ids = sample["class_ids"]  # (N,)

            # Convert image for predictor
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            self.predictor.set_image(image_np)

            pred_masks_list = []

            for mask_idx in range(len(gt_masks)):
                gt_mask = gt_masks[mask_idx].numpy()
                coords = point_coords[mask_idx].numpy()
                labels = point_labels[mask_idx].numpy()
                class_id = class_ids[mask_idx].item()

                # Filter valid points
                valid = labels >= 0
                if valid.sum() == 0:
                    continue

                coords = coords[valid]
                labels = labels[valid]

                # Predict
                pred_masks, scores, logits = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )

                pred_logits = logits[0]  # (H, W)

                # Resize if needed
                if pred_logits.shape != gt_mask.shape:
                    pred_logits = cv2.resize(
                        pred_logits,
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # Compute metrics
                metrics = self.compute_metrics(pred_logits, gt_mask)

                # Accumulate overall metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)

                # Accumulate class-specific metrics
                if class_id in class_metrics:
                    for key, value in metrics.items():
                        class_metrics[class_id][key].append(value)

                pred_masks_list.append((pred_logits, gt_mask, class_id, coords))

            # Save visualization
            if save_visualizations and vis_dir and len(pred_masks_list) > 0:
                self._save_visualization(
                    image_np,
                    pred_masks_list,
                    vis_dir / f"sample_{idx:04d}.png",
                )

        # Compute aggregate metrics
        results = {
            "overall": {
                key: np.mean(values) if values else 0.0
                for key, values in all_metrics.items()
            },
            "per_class": {},
            "num_samples": num_samples,
            "num_masks_evaluated": len(all_metrics["iou"]),
        }

        # Per-class metrics
        for class_id in self.train_classes:
            class_name = self.class_names[class_id]
            if class_metrics[class_id]["iou"]:
                results["per_class"][class_name] = {
                    key: np.mean(values)
                    for key, values in class_metrics[class_id].items()
                }
                results["per_class"][class_name]["num_masks"] = len(class_metrics[class_id]["iou"])

        return results

    def _save_visualization(
        self,
        image: np.ndarray,
        pred_masks_list: List[Tuple],
        save_path: Path,
    ):
        """Save visualization of predictions vs ground truth."""
        num_masks = len(pred_masks_list)
        fig, axes = plt.subplots(num_masks, 3, figsize=(12, 4 * num_masks))

        if num_masks == 1:
            axes = axes.reshape(1, -1)

        for i, (pred_logits, gt_mask, class_id, points) in enumerate(pred_masks_list):
            class_name = self.class_names[class_id]
            pred_binary = (1 / (1 + np.exp(-pred_logits))) > 0.5

            # Original image with points
            axes[i, 0].imshow(image)
            axes[i, 0].scatter(points[:, 0], points[:, 1], c='red', s=50, marker='*')
            axes[i, 0].set_title(f"Image + Points ({class_name})")
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            # Prediction
            axes[i, 2].imshow(pred_binary, cmap='gray')
            metrics = self.compute_metrics(pred_logits, gt_mask)
            axes[i, 2].set_title(f"Prediction (IoU: {metrics['iou']:.3f})")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def print_results(self, results: Dict):
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nSamples evaluated: {results['num_samples']}")
        print(f"Total masks evaluated: {results['num_masks_evaluated']}")

        print("\n--- Overall Metrics ---")
        print(f"{'Metric':<15} {'Value':>10}")
        print("-" * 25)
        for metric, value in results["overall"].items():
            print(f"{metric:<15} {value:>10.4f}")

        print("\n--- Per-Class IoU ---")
        print(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'Count':>8}")
        print("-" * 60)

        for class_name, metrics in results["per_class"].items():
            print(
                f"{class_name:<20} "
                f"{metrics['iou']:>8.4f} "
                f"{metrics['dice']:>8.4f} "
                f"{metrics['precision']:>8.4f} "
                f"{metrics['recall']:>8.4f} "
                f"{metrics['num_masks']:>8d}"
            )

        # Highlight difficult classes
        print("\n--- Difficult Classes Analysis ---")
        difficult = ["cystic_artery", "cystic_duct"]
        for cls in difficult:
            if cls in results["per_class"]:
                m = results["per_class"][cls]
                print(f"{cls}: IoU={m['iou']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on Endoscapes")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate"
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
        "--num-samples", type=int, default=None,
        help="Limit evaluation to N samples"
    )
    parser.add_argument(
        "--save-vis", action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--vis-dir", type=str, default=None,
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON"
    )
    args = parser.parse_args()

    # Create config
    config_kwargs = {}
    if args.project_store:
        config_kwargs["project_store"] = args.project_store
    if args.dataset_root:
        config_kwargs["dataset_root"] = args.dataset_root

    config = get_config(**config_kwargs)

    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)

    # Create dataset
    dataset = EndoscapesDataset(config, split=args.split, transform=False)
    print(f"Evaluating on {args.split} split ({len(dataset)} samples)")

    # Setup visualization directory
    vis_dir = None
    if args.save_vis:
        vis_dir = Path(args.vis_dir) if args.vis_dir else config.paths.output_dir / "visualizations" / args.split
        print(f"Saving visualizations to {vis_dir}")

    # Evaluate
    results = evaluator.evaluate_dataset(
        dataset,
        num_samples=args.num_samples,
        save_visualizations=args.save_vis,
        vis_dir=vis_dir,
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.paths.output_dir / f"eval_results_{args.split}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
