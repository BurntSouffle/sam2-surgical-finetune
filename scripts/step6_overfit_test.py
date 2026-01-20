"""
Step 6: Overfitting Test - Verify Model Can Learn
Trains on a SINGLE sample to verify the training pipeline works.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from step4_dataset import EndoscapesDataset, get_split_paths, CLASS_NAMES, CLASS_COLORS, mask_to_color
from step5_model_setup import SAM2SemanticSegmentation, CHECKPOINT_PATH, CONFIG_FILE, NUM_CLASSES

# Output directory
OUTPUT_DIR = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs")


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print("="*70)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> Dict[str, float]:
    """
    Compute segmentation metrics.

    Args:
        pred: Predicted class indices (H, W)
        target: Ground truth class indices (H, W), -1 for ignore
        num_classes: Number of classes

    Returns:
        Dict with pixel_accuracy and per-class IoU
    """
    # Mask for valid pixels (not ignore)
    valid_mask = target != -1

    # Pixel accuracy
    correct = (pred == target) & valid_mask
    pixel_acc = correct.sum().float() / valid_mask.sum().float()

    # Per-class IoU
    ious = {}
    for c in range(num_classes):
        pred_c = (pred == c) & valid_mask
        target_c = (target == c) & valid_mask

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            ious[c] = (intersection / union).item()
        else:
            ious[c] = float('nan')  # Class not present

    # Mean IoU (excluding NaN)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        'pixel_accuracy': pixel_acc.item(),
        'mean_iou': mean_iou,
        'class_iou': ious,
    }


def visualize_results(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    losses: list,
    save_path: Path,
):
    """Create visualization of overfitting test results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Panel a) Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("(a) Original Image", fontsize=12)
    axes[0, 0].axis("off")

    # Panel b) Ground truth mask
    gt_color = mask_to_color(gt_mask)
    axes[0, 1].imshow(gt_color)
    axes[0, 1].set_title("(b) Ground Truth Mask", fontsize=12)
    axes[0, 1].axis("off")

    # Panel c) Predicted mask
    pred_color = mask_to_color(pred_mask)
    axes[1, 0].imshow(pred_color)
    axes[1, 0].set_title("(c) Predicted Mask (after overfitting)", fontsize=12)
    axes[1, 0].axis("off")

    # Panel d) Loss curve
    axes[1, 1].plot(losses, 'b-', linewidth=2)
    axes[1, 1].set_xlabel("Iteration", fontsize=11)
    axes[1, 1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1, 1].set_title("(d) Training Loss Curve", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)

    # Add horizontal line at random baseline
    random_loss = np.log(NUM_CLASSES)  # ln(7) ≈ 1.95
    axes[1, 1].axhline(y=random_loss, color='r', linestyle='--', label=f'Random baseline (ln(7)={random_loss:.2f})')
    axes[1, 1].legend()

    # Add legend for classes
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(color)/255, label=f"{i}: {name}")
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {save_path}")


def run_overfit_test():
    """Run the overfitting test on a single sample."""
    print_header("STEP 6: Overfitting Test")

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        return

    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # =========================================================================
    # 1. SETUP
    # =========================================================================
    print_header("1. Setup")

    # Load dataset
    print("\nLoading dataset...")
    images_dir, masks_dir = get_split_paths("train")
    dataset = EndoscapesDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=(512, 512),
        normalize=True,
    )

    # Get single sample
    sample_idx = 0
    sample = dataset[sample_idx]
    image = sample['image'].unsqueeze(0).to(device)  # (1, 3, 512, 512)
    mask = sample['mask'].unsqueeze(0).to(device)    # (1, 512, 512)
    filename = sample['filename']

    print(f"Sample: {filename}")
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Mask shape: {tuple(mask.shape)}")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")

    # Load model
    print("\nLoading model...")
    from sam2.build_sam import build_sam2

    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device="cpu",
        mode="eval"
    )

    model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=(512, 512),
        freeze_encoder=True,
        use_multiscale_head=True,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # 2. TRAINING CONFIG
    # =========================================================================
    print_header("2. Training Configuration")

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.01,
    )

    # Training settings
    num_iterations = 200
    print_every = 20

    print(f"Optimizer: AdamW (lr=1e-3, weight_decay=0.01)")
    print(f"Loss: CrossEntropyLoss (ignore_index=-1)")
    print(f"Iterations: {num_iterations}")

    # =========================================================================
    # 3. TRAINING LOOP
    # =========================================================================
    print_header("3. Training Loop (Overfitting on Single Sample)")

    torch.cuda.reset_peak_memory_stats()
    losses = []

    model.train()

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        output = model(image)
        logits = output['logits']  # (1, 7, 512, 512)

        # Compute loss
        loss = criterion(logits, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if (iteration + 1) % print_every == 0 or iteration == 0:
            print(f"  Iteration {iteration+1:3d}/{num_iterations}: Loss = {loss.item():.6f}")

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    # =========================================================================
    # 4. SUCCESS CRITERIA CHECK
    # =========================================================================
    print_header("4. Success Criteria Check")

    random_baseline = np.log(NUM_CLASSES)  # ln(7) ≈ 1.95
    starting_loss = losses[0]
    final_loss = losses[-1]

    print(f"Random baseline (ln(7)): {random_baseline:.4f}")
    print(f"Starting loss: {starting_loss:.4f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {(1 - final_loss/starting_loss)*100:.1f}%")

    # Check criteria
    if final_loss < 0.1:
        print("\n[PASS] Loss < 0.1: Model CAN learn!")
    elif final_loss < 0.5:
        print("\n[WARN] Loss < 0.5 but > 0.1: Model learning, may need more iterations")
    elif final_loss < starting_loss:
        print("\n[WARN] Loss decreased but plateaued high: Check architecture")
    else:
        print("\n[FAIL] Loss didn't decrease: SOMETHING IS WRONG")

    # =========================================================================
    # 5. INFERENCE AND METRICS
    # =========================================================================
    print_header("5. Inference and Metrics")

    model.eval()
    with torch.no_grad():
        output = model(image)
        logits = output['logits']  # (1, 7, 512, 512)
        pred = torch.argmax(logits, dim=1)  # (1, 512, 512)

    # Compute metrics
    metrics = compute_metrics(pred[0], mask[0], NUM_CLASSES)

    print(f"\nPixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%")
    print(f"Mean IoU: {metrics['mean_iou']*100:.2f}%")
    print(f"\nPer-class IoU:")
    for c, iou in metrics['class_iou'].items():
        if not np.isnan(iou):
            print(f"  {c}: {CLASS_NAMES[c]:15s} {iou*100:6.2f}%")
        else:
            print(f"  {c}: {CLASS_NAMES[c]:15s}    N/A (not in sample)")

    # =========================================================================
    # 6. VISUALIZATION
    # =========================================================================
    print_header("6. Visualization")

    # Get raw image for visualization
    raw_image, raw_mask, _ = dataset.get_raw_sample(sample_idx)
    pred_mask = pred[0].cpu().numpy()

    save_path = OUTPUT_DIR / "step6_overfit_result.png"
    visualize_results(raw_image, raw_mask, pred_mask, losses, save_path)

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print_header("FINAL REPORT")

    print(f"""
Sample: {filename}
Device: {torch.cuda.get_device_name(0)}

Training:
  Iterations: {num_iterations}
  Starting loss: {starting_loss:.4f}
  Final loss: {final_loss:.6f}
  Loss reduction: {(1 - final_loss/starting_loss)*100:.1f}%

Metrics:
  Pixel Accuracy: {metrics['pixel_accuracy']*100:.2f}%
  Mean IoU: {metrics['mean_iou']*100:.2f}%

Memory:
  Peak GPU memory: {peak_memory:.1f} MB

Verdict: {"PASS - Model can learn!" if final_loss < 0.1 else "NEEDS INVESTIGATION"}
""")

    # Final success check
    if final_loss < 0.1 and metrics['pixel_accuracy'] > 0.95:
        print("="*70)
        print(" SUCCESS: Model is ready for full training!")
        print("="*70)
    elif final_loss < 0.5:
        print("="*70)
        print(" PARTIAL SUCCESS: Model learning, consider tuning")
        print("="*70)
    else:
        print("="*70)
        print(" FAILURE: Debug training pipeline before proceeding")
        print("="*70)


if __name__ == "__main__":
    run_overfit_test()
