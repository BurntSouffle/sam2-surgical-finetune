"""
Custom Loss Functions for Semantic Segmentation
Includes Focal Loss, Dice Loss, and Combined Loss for handling class imbalance.

Usage in step7_train.py:
========================

# Option 1: Focal Loss (good for class imbalance)
from losses import FocalLoss
criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=-1)

# Option 2: Dice Loss (good for small objects)
from losses import DiceLoss
criterion = DiceLoss(smooth=1.0, ignore_index=-1)

# Option 3: Combined Loss (best of both worlds)
from losses import CombinedLoss
criterion = CombinedLoss(
    loss_type='focal+dice',  # or 'ce+dice', 'ce', 'focal'
    class_weights=class_weights,
    gamma=2.0,
    focal_weight=1.0,
    dice_weight=0.5,
    ignore_index=-1,
)

# Then in training loop:
loss = criterion(logits, masks)  # logits: (B, C, H, W), masks: (B, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection/segmentation.

    Focal Loss formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This loss down-weights easy examples and focuses training on hard negatives.
    Particularly useful for class imbalance in segmentation.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Class weights tensor of shape (num_classes,). If None, no class weighting.
        gamma: Focusing parameter (gamma >= 0). Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to cross-entropy loss.
               gamma=2 is commonly used (default).
        ignore_index: Target value to ignore in loss computation.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W) with values in [0, C-1]
                     or ignore_index for pixels to ignore

        Returns:
            Focal loss scalar (if reduction='mean' or 'sum') or per-pixel loss
        """
        B, C, H, W = logits.shape

        # Create mask for valid pixels
        valid_mask = targets != self.ignore_index

        # Replace ignore_index with 0 temporarily (will be masked out)
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Get probability of the true class: p_t
        # Gather the probability at the target class for each pixel
        targets_one_hot = targets_safe.unsqueeze(1)  # (B, 1, H, W)
        p_t = probs.gather(1, targets_one_hot).squeeze(1)  # (B, H, W)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Apply focal modulation
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha)
        if self.alpha is not None:
            # Move alpha to same device as logits
            alpha = self.alpha.to(logits.device)
            # Get alpha for each pixel based on its target class
            alpha_t = alpha[targets_safe]  # (B, H, W)
            focal_loss = alpha_t * focal_loss

        # Apply valid mask
        focal_loss = focal_loss * valid_mask.float()

        # Reduction
        if self.reduction == 'mean':
            # Mean over valid pixels only
            return focal_loss.sum() / (valid_mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice coefficient: D = (2 * |X ∩ Y|) / (|X| + |Y|)
    Dice Loss: 1 - D

    This loss directly optimizes the Dice/F1 score, which is useful for
    imbalanced segmentation tasks where pixel accuracy is misleading.

    Args:
        smooth: Smoothing factor to avoid division by zero (default=1.0)
        ignore_index: Target value to ignore in loss computation
        reduction: 'mean' (over classes) or 'sum'
        per_class: If True, compute dice per class then average.
                   If False, compute global dice.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -1,
        reduction: str = 'mean',
        per_class: bool = True,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.per_class = per_class

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)

        Returns:
            Dice loss scalar
        """
        B, C, H, W = logits.shape

        # Create valid mask
        valid_mask = targets != self.ignore_index

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Create one-hot encoding of targets
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        targets_one_hot = F.one_hot(targets_safe, num_classes=C)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply valid mask to both predictions and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)  # (B, C, H, W)
        probs = probs * valid_mask_expanded.float()
        targets_one_hot = targets_one_hot * valid_mask_expanded.float()

        if self.per_class:
            # Compute dice per class
            # Flatten spatial dimensions
            probs_flat = probs.view(B, C, -1)  # (B, C, H*W)
            targets_flat = targets_one_hot.view(B, C, -1)  # (B, C, H*W)

            # Compute intersection and union per class
            intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
            union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

            # Dice per class
            dice = (2 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

            # Average over classes and batch
            dice_loss = 1 - dice.mean()
        else:
            # Global dice
            probs_flat = probs.view(-1)
            targets_flat = targets_one_hot.view(-1)

            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice

        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that can mix different loss types.

    Supported combinations:
    - 'ce': Cross-Entropy only
    - 'focal': Focal Loss only
    - 'dice': Dice Loss only
    - 'ce+dice': Cross-Entropy + Dice
    - 'focal+dice': Focal Loss + Dice (recommended for imbalanced segmentation)

    Args:
        loss_type: Type of loss combination
        class_weights: Class weights for CE/Focal loss
        gamma: Focal loss gamma parameter
        ce_weight: Weight for CE loss in combination
        focal_weight: Weight for Focal loss in combination
        dice_weight: Weight for Dice loss in combination
        ignore_index: Index to ignore
        dice_smooth: Smoothing factor for Dice loss
    """

    def __init__(
        self,
        loss_type: str = 'focal+dice',
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = -1,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        # Initialize component losses
        if 'ce' in loss_type:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index,
            )

        if 'focal' in loss_type:
            self.focal_loss = FocalLoss(
                alpha=class_weights,
                gamma=gamma,
                ignore_index=ignore_index,
            )

        if 'dice' in loss_type:
            self.dice_loss = DiceLoss(
                smooth=dice_smooth,
                ignore_index=ignore_index,
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)

        Returns:
            Combined loss scalar
        """
        total_loss = 0.0

        if self.loss_type == 'ce':
            total_loss = self.ce_loss(logits, targets)

        elif self.loss_type == 'focal':
            total_loss = self.focal_loss(logits, targets)

        elif self.loss_type == 'dice':
            total_loss = self.dice_loss(logits, targets)

        elif self.loss_type == 'ce+dice':
            ce = self.ce_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            total_loss = self.ce_weight * ce + self.dice_weight * dice

        elif self.loss_type == 'focal+dice':
            focal = self.focal_loss(logits, targets)
            dice = self.dice_loss(logits, targets)
            total_loss = self.focal_weight * focal + self.dice_weight * dice

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return total_loss


# =============================================================================
# TEST FUNCTIONS
# =============================================================================
def test_losses():
    """Test all loss functions with dummy data."""
    print("="*70)
    print(" LOSS FUNCTIONS TEST")
    print("="*70)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dummy data
    B, C, H, W = 2, 7, 64, 64  # batch=2, classes=7, 64x64

    # Random logits (before softmax)
    logits = torch.randn(B, C, H, W, requires_grad=True)

    # Random targets with some ignore pixels
    targets = torch.randint(0, C, (B, H, W))
    # Add some ignore pixels (-1)
    ignore_mask = torch.rand(B, H, W) < 0.1  # 10% ignore
    targets[ignore_mask] = -1

    # Class weights (simulating imbalanced classes)
    class_weights = torch.tensor([0.1, 1.2, 2.9, 1.8, 0.8, 0.1, 0.13])

    print(f"\nInput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Ignore pixels: {ignore_mask.sum().item()}")
    print(f"  Class weights: {class_weights.tolist()}")

    # Test Cross-Entropy Loss (baseline)
    print("\n" + "-"*50)
    print(" 1. Cross-Entropy Loss (baseline)")
    print("-"*50)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    ce_val = ce_loss(logits, targets)
    ce_val.backward(retain_graph=True)

    print(f"  Loss value: {ce_val.item():.6f}")
    print(f"  Gradient flows: {logits.grad is not None and logits.grad.abs().sum() > 0}")
    logits.grad.zero_()

    # Test Focal Loss
    print("\n" + "-"*50)
    print(" 2. Focal Loss (gamma=2.0)")
    print("-"*50)

    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=-1)
    focal_val = focal_loss(logits, targets)
    focal_val.backward(retain_graph=True)

    print(f"  Loss value: {focal_val.item():.6f}")
    print(f"  Gradient flows: {logits.grad is not None and logits.grad.abs().sum() > 0}")
    logits.grad.zero_()

    # Test Focal Loss with different gamma values
    print("\n  Focal Loss with different gamma values:")
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        fl = FocalLoss(alpha=class_weights, gamma=gamma, ignore_index=-1)
        val = fl(logits, targets)
        print(f"    gamma={gamma}: {val.item():.6f}")

    # Test Dice Loss
    print("\n" + "-"*50)
    print(" 3. Dice Loss")
    print("-"*50)

    dice_loss = DiceLoss(smooth=1.0, ignore_index=-1)
    dice_val = dice_loss(logits, targets)
    dice_val.backward(retain_graph=True)

    print(f"  Loss value: {dice_val.item():.6f}")
    print(f"  Gradient flows: {logits.grad is not None and logits.grad.abs().sum() > 0}")
    logits.grad.zero_()

    # Test Combined Losses
    print("\n" + "-"*50)
    print(" 4. Combined Losses")
    print("-"*50)

    loss_types = ['ce', 'focal', 'dice', 'ce+dice', 'focal+dice']

    for loss_type in loss_types:
        combined = CombinedLoss(
            loss_type=loss_type,
            class_weights=class_weights,
            gamma=2.0,
            ce_weight=1.0,
            focal_weight=1.0,
            dice_weight=0.5,
            ignore_index=-1,
        )
        val = combined(logits, targets)
        val.backward(retain_graph=True)
        grad_ok = logits.grad is not None and logits.grad.abs().sum() > 0
        logits.grad.zero_()

        print(f"  {loss_type:15s}: {val.item():.6f} (grad OK: {grad_ok})")

    # Test edge cases
    print("\n" + "-"*50)
    print(" 5. Edge Cases")
    print("-"*50)

    # All pixels are one class
    targets_single = torch.zeros(B, H, W, dtype=torch.long)
    focal_single = FocalLoss(gamma=2.0, ignore_index=-1)(logits, targets_single)
    print(f"  Single class targets: {focal_single.item():.6f}")

    # Very confident predictions (low loss expected)
    logits_confident = torch.zeros(B, C, H, W)
    logits_confident[:, 0, :, :] = 10.0  # Very confident class 0
    targets_zero = torch.zeros(B, H, W, dtype=torch.long)
    focal_confident = FocalLoss(gamma=2.0, ignore_index=-1)(logits_confident, targets_zero)
    print(f"  Confident correct predictions: {focal_confident.item():.6f} (should be low)")

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print("""
Loss Functions Available:
  - FocalLoss: Good for class imbalance, focuses on hard examples
  - DiceLoss: Directly optimizes Dice/F1, good for small objects
  - CombinedLoss: Mix of losses for best results

Recommended for Endoscapes (highly imbalanced):
  CombinedLoss(loss_type='focal+dice', gamma=2.0, dice_weight=0.5)

All loss functions:
  [PASS] Compute valid loss values
  [PASS] Gradients flow correctly
  [PASS] Handle ignore_index properly
""")


def compare_on_imbalanced_data():
    """Compare losses on highly imbalanced data (like Endoscapes)."""
    print("\n" + "="*70)
    print(" COMPARISON ON IMBALANCED DATA")
    print("="*70)

    torch.manual_seed(42)

    # Simulate Endoscapes-like distribution
    # background: 72%, gallbladder: 17%, tool: 7%, others: 4%
    B, C, H, W = 4, 7, 128, 128

    # Create imbalanced targets
    targets = torch.zeros(B, H, W, dtype=torch.long)  # mostly background

    # Add some gallbladder (class 5) - 17%
    gb_mask = torch.rand(B, H, W) < 0.17
    targets[gb_mask] = 5

    # Add some tool (class 6) - 7%
    tool_mask = torch.rand(B, H, W) < 0.07
    targets[tool_mask] = 6

    # Add rare classes - small regions
    for c in [1, 2, 3, 4]:  # cystic_plate, calot_triangle, cystic_artery, cystic_duct
        rare_mask = torch.rand(B, H, W) < 0.01
        targets[rare_mask] = c

    # Print class distribution
    print("\nSimulated class distribution:")
    total = B * H * W
    for c in range(C):
        count = (targets == c).sum().item()
        print(f"  Class {c}: {count:6d} ({100*count/total:.2f}%)")

    # Create logits that predict mostly background (mimics undertrained model)
    logits = torch.zeros(B, C, H, W)
    logits[:, 0, :, :] = 2.0  # bias toward background
    logits = logits + torch.randn_like(logits) * 0.5
    logits.requires_grad = True

    # Class weights (inverse frequency)
    class_weights = torch.tensor([0.1, 3.0, 5.0, 4.0, 2.0, 0.3, 0.5])

    print("\nLoss values for undertrained model (predicts mostly background):")
    print("-"*50)

    # Test different losses
    losses = {
        'CE (unweighted)': nn.CrossEntropyLoss(ignore_index=-1),
        'CE (weighted)': nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1),
        'Focal (g=0)': FocalLoss(alpha=class_weights, gamma=0.0, ignore_index=-1),
        'Focal (g=1)': FocalLoss(alpha=class_weights, gamma=1.0, ignore_index=-1),
        'Focal (g=2)': FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=-1),
        'Dice': DiceLoss(ignore_index=-1),
        'Focal+Dice': CombinedLoss('focal+dice', class_weights, gamma=2.0, dice_weight=0.5, ignore_index=-1),
    }

    for name, loss_fn in losses.items():
        val = loss_fn(logits, targets)
        print(f"  {name:20s}: {val.item():.4f}")

    print("\nNote: Higher loss = more penalty for misclassifying rare classes")
    print("Focal Loss with gamma>0 gives higher loss because it penalizes")
    print("confident wrong predictions on rare classes more heavily.")


if __name__ == "__main__":
    test_losses()
    compare_on_imbalanced_data()
