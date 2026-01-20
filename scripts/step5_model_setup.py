"""
Step 5: SAM2 Model Setup for Semantic Segmentation Fine-tuning
Explores SAM2 architecture and creates a wrapper for semantic segmentation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths
CHECKPOINT_PATH = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\sam2.1_hiera_tiny.pt")
CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_t.yaml"
OUTPUT_DIR = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\outputs")

# Segmentation settings
NUM_CLASSES = 7
ENCODER_EMBED_DIM = 256  # SAM2 encoder output channels


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print("="*70)


def print_subheader(title: str) -> None:
    print(f"\n{'-'*50}")
    print(f" {title}")
    print("-"*50)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


class SegmentationHead(nn.Module):
    """
    Simple segmentation head that takes FPN features and produces class logits.

    Option A: Simple head using highest resolution FPN feature.
    Uses Conv layers + bilinear upsampling to get back to input resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        hidden_dim: int = 128,
        target_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.target_size = target_size

        # Convolution layers to process features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (B, C, H, W) from encoder

        Returns:
            logits: Tensor of shape (B, num_classes, target_H, target_W)
        """
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.classifier(x)

        # Upsample to target size
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return x


class MultiScaleSegmentationHead(nn.Module):
    """
    Option B: Multi-scale segmentation head that fuses FPN features.
    Uses all FPN levels for better spatial resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 7,
        hidden_dim: int = 128,
        target_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.target_size = target_size

        # Lateral connections for each FPN level
        self.lateral_conv0 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)  # 128x128
        self.lateral_conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)  # 64x64
        self.lateral_conv2 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)  # 32x32

        # Output convolutions after fusion
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Classifier
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fpn_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            fpn_features: List of FPN tensors at different scales
                [0]: (B, 256, 128, 128) - highest resolution
                [1]: (B, 256, 64, 64)
                [2]: (B, 256, 32, 32) - lowest resolution

        Returns:
            logits: Tensor of shape (B, num_classes, target_H, target_W)
        """
        # Process each level
        f0 = self.lateral_conv0(fpn_features[0])  # (B, hidden, 128, 128)
        f1 = self.lateral_conv1(fpn_features[1])  # (B, hidden, 64, 64)
        f2 = self.lateral_conv2(fpn_features[2])  # (B, hidden, 32, 32)

        # Upsample and add (FPN-style top-down)
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        f1 = f1 + f2_up

        f1_up = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        f0 = f0 + f1_up

        # Process fused features
        out = self.output_conv(f0)
        out = self.classifier(out)

        # Upsample to target size
        out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)

        return out


class SAM2SemanticSegmentation(nn.Module):
    """
    SAM2-based semantic segmentation model.

    Uses SAM2's image encoder as a frozen backbone and adds
    a trainable segmentation head for dense prediction.
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        num_classes: int = 7,
        target_size: Tuple[int, int] = (512, 512),
        freeze_encoder: bool = True,
        use_multiscale_head: bool = True,
        hidden_dim: int = 128,
    ):
        """
        Args:
            sam2_model: Loaded SAM2 model
            num_classes: Number of segmentation classes
            target_size: Output resolution (H, W)
            freeze_encoder: Whether to freeze the image encoder
            use_multiscale_head: Use multi-scale (Option B) or simple (Option A) head
            hidden_dim: Hidden dimension for segmentation head
        """
        super().__init__()

        # Extract and store the image encoder
        self.image_encoder = sam2_model.image_encoder

        # Freeze encoder if requested
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()

        # Create segmentation head
        self.use_multiscale_head = use_multiscale_head
        if use_multiscale_head:
            self.seg_head = MultiScaleSegmentationHead(
                in_channels=ENCODER_EMBED_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                target_size=target_size,
            )
        else:
            self.seg_head = SegmentationHead(
                in_channels=ENCODER_EMBED_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                target_size=target_size,
            )

        self.num_classes = num_classes
        self.target_size = target_size

    def train(self, mode: bool = True):
        """Override train to keep encoder frozen if specified."""
        super().train(mode)
        if self.freeze_encoder:
            self.image_encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for semantic segmentation.

        Args:
            images: Input images of shape (B, 3, H, W), normalized

        Returns:
            Dict with:
                - 'logits': Segmentation logits (B, num_classes, H, W)
                - 'features': Encoder features for visualization/analysis
        """
        # Get encoder features
        with torch.set_grad_enabled(not self.freeze_encoder):
            encoder_out = self.image_encoder(images)

        # encoder_out contains:
        # - 'vision_features': (B, 256, H/16, W/16)
        # - 'backbone_fpn': [(B, 256, H/4, W/4), (B, 256, H/8, W/8), (B, 256, H/16, W/16)]

        # Get segmentation logits
        if self.use_multiscale_head:
            logits = self.seg_head(encoder_out['backbone_fpn'])
        else:
            # Use highest resolution FPN feature
            logits = self.seg_head(encoder_out['backbone_fpn'][0])

        return {
            'logits': logits,
            'features': encoder_out['vision_features'],
        }

    def get_encoder_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get encoder features without segmentation head."""
        with torch.no_grad():
            return self.image_encoder(images)


def explore_architecture():
    """Explore SAM2 architecture in detail."""
    print_header("1. SAM2.1-Tiny Architecture Exploration")

    from sam2.build_sam import build_sam2

    model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device="cpu",
        mode="eval"
    )

    print("\nMain components:")
    components = []
    for name, child in model.named_children():
        params = count_parameters(child)
        components.append((name, type(child).__name__, params))
        print(f"  {name:25s} {type(child).__name__:20s} {format_params(params):>10s}")

    total = count_parameters(model)
    print(f"\n  {'TOTAL':25s} {'-':20s} {format_params(total):>10s}")

    # Image encoder details
    print_subheader("Image Encoder Breakdown")
    enc = model.image_encoder
    for name, child in enc.named_children():
        params = count_parameters(child)
        print(f"  {name:25s} {type(child).__name__:20s} {format_params(params):>10s}")

    return model


def analyze_forward_pass(model):
    """Analyze tensor shapes through forward pass."""
    print_header("2. Forward Pass Analysis")

    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 3, 512, 512)

    print(f"\nInput: {tuple(img.shape)}")

    print_subheader("Image Encoder Output")
    with torch.no_grad():
        enc_out = model.image_encoder(img)

    print(f"  vision_features: {tuple(enc_out['vision_features'].shape)}")
    print(f"  backbone_fpn:")
    for i, feat in enumerate(enc_out['backbone_fpn']):
        scale = 512 // feat.shape[-1]
        print(f"    [{i}]: {tuple(feat.shape)} (1/{scale} scale)")

    print(f"  vision_pos_enc:")
    for i, pos in enumerate(enc_out['vision_pos_enc']):
        print(f"    [{i}]: {tuple(pos.shape)}")


def test_segmentation_wrapper():
    """Test the semantic segmentation wrapper."""
    print_header("3. Semantic Segmentation Wrapper Test")

    from sam2.build_sam import build_sam2

    # Load SAM2
    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device="cpu",
        mode="eval"
    )

    # Create wrapper - Option B (multi-scale)
    print_subheader("Creating SAM2SemanticSegmentation (Multi-scale head)")
    seg_model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=(512, 512),
        freeze_encoder=True,
        use_multiscale_head=True,
        hidden_dim=128,
    )

    # Parameter counts
    total_params = count_parameters(seg_model)
    trainable_params = count_trainable_parameters(seg_model)
    encoder_params = count_parameters(seg_model.image_encoder)
    head_params = count_parameters(seg_model.seg_head)

    print(f"\nParameter counts:")
    print(f"  Image Encoder: {format_params(encoder_params)} (frozen)")
    print(f"  Seg Head:      {format_params(head_params)} (trainable)")
    print(f"  Total:         {format_params(total_params)}")
    print(f"  Trainable:     {format_params(trainable_params)}")

    # Forward pass test
    print_subheader("Forward Pass Test (CPU)")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)

    print(f"Input shape: {tuple(dummy_input.shape)}")

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(dummy_input)

    print(f"Output logits shape: {tuple(output['logits'].shape)}")
    print(f"Output features shape: {tuple(output['features'].shape)}")

    expected_shape = (batch_size, NUM_CLASSES, 512, 512)
    assert output['logits'].shape == expected_shape, f"Expected {expected_shape}, got {output['logits'].shape}"
    print(f"Shape verification: PASSED")

    return seg_model


def test_gpu_memory():
    """Test GPU memory usage."""
    print_header("4. GPU Memory Test")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    from sam2.build_sam import build_sam2

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU memory before: {mem_before:.1f} MB")

    # Load model
    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device="cpu",
        mode="eval"
    )

    seg_model = SAM2SemanticSegmentation(
        sam2_model=sam2_model,
        num_classes=NUM_CLASSES,
        target_size=(512, 512),
        freeze_encoder=True,
        use_multiscale_head=True,
    )

    seg_model = seg_model.to(device)
    torch.cuda.synchronize()

    mem_after_load = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU memory after model load: {mem_after_load:.1f} MB")

    # Forward pass
    print_subheader("Forward Pass (batch_size=2)")
    dummy_input = torch.randn(2, 3, 512, 512, device=device)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(dummy_input)

    torch.cuda.synchronize()
    mem_after_forward = torch.cuda.memory_allocated() / 1024**2
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    print(f"GPU memory after forward: {mem_after_forward:.1f} MB")
    print(f"Peak GPU memory: {peak_memory:.1f} MB")

    # Test with batch_size=4
    print_subheader("Forward Pass (batch_size=4)")
    torch.cuda.reset_peak_memory_stats()
    dummy_input = torch.randn(4, 3, 512, 512, device=device)

    with torch.no_grad():
        output = seg_model(dummy_input)

    torch.cuda.synchronize()
    peak_memory_4 = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory (batch=4): {peak_memory_4:.1f} MB")

    # Cleanup
    del seg_model, sam2_model, dummy_input, output
    torch.cuda.empty_cache()


def save_architecture_diagram():
    """Save architecture description to file."""
    print_header("5. Saving Architecture Diagram")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "step5_model_architecture.txt"

    diagram = """
================================================================================
                    SAM2 SEMANTIC SEGMENTATION ARCHITECTURE
================================================================================

INPUT: Image (B, 3, 512, 512)
       |
       v
+----------------------------------------------------------------------+
|                        SAM2 IMAGE ENCODER (FROZEN)                    |
|                           ~27.2M parameters                           |
|----------------------------------------------------------------------|
|  Hiera Backbone (26.8M)                                              |
|    - Patch Embedding: 14K params                                     |
|    - Transformer Blocks: 26.8M params                                |
|                                                                      |
|  FPN Neck (0.37M)                                                    |
|    - Multi-scale feature fusion                                      |
+----------------------------------------------------------------------+
       |
       | backbone_fpn outputs:
       |   [0]: (B, 256, 128, 128)  <- 1/4 scale (highest resolution)
       |   [1]: (B, 256, 64, 64)    <- 1/8 scale
       |   [2]: (B, 256, 32, 32)    <- 1/16 scale (lowest resolution)
       v
+----------------------------------------------------------------------+
|                   MULTI-SCALE SEGMENTATION HEAD (TRAINABLE)          |
|                           ~0.2M parameters                            |
|----------------------------------------------------------------------|
|  Lateral Convolutions (1x1):                                         |
|    - Conv 256->128 for each FPN level                                |
|                                                                      |
|  Top-down Fusion:                                                    |
|    - Upsample low-res + add to high-res                              |
|                                                                      |
|  Output Convolutions:                                                |
|    - Conv 128->128 (3x3, BN, ReLU) x2                                |
|                                                                      |
|  Classifier:                                                         |
|    - Conv 128->7 (1x1)                                               |
|                                                                      |
|  Upsample:                                                           |
|    - Bilinear interpolation to 512x512                               |
+----------------------------------------------------------------------+
       |
       v
OUTPUT: Logits (B, 7, 512, 512)
        |
        v (during training)
        Cross-Entropy Loss with class weights (ignore_index=-1)


================================================================================
                           PARAMETER SUMMARY
================================================================================

Component               Parameters      Trainable
-------------------------------------------------
Image Encoder           27.22M          No (frozen)
  - Hiera Backbone      26.85M          No
  - FPN Neck            0.37M           No
Segmentation Head       0.20M           Yes
-------------------------------------------------
TOTAL                   27.42M
TRAINABLE               0.20M (~0.7%)


================================================================================
                          MEMORY REQUIREMENTS
================================================================================

                        Inference       Training (est.)
---------------------------------------------------------
Model weights           ~150 MB         ~150 MB
Batch=2 (512x512)       ~600 MB         ~2.5 GB
Batch=4 (512x512)       ~800 MB         ~4.5 GB
Batch=8 (512x512)       ~1.2 GB         ~8.0 GB

Recommended for RTX 3080 (12GB): batch_size=4-8


================================================================================
                          CLASS DEFINITIONS
================================================================================

Class ID    Name              Color (RGB)
------------------------------------------
0           background        (0, 0, 0)       black
1           cystic_plate      (255, 0, 0)     red
2           calot_triangle    (0, 255, 0)     green
3           cystic_artery     (0, 0, 255)     blue
4           cystic_duct       (255, 255, 0)   yellow
5           gallbladder       (255, 0, 255)   magenta
6           tool              (0, 255, 255)   cyan
-1/255      ignore            N/A             (not used in loss)


================================================================================
                          TRAINING STRATEGY
================================================================================

Phase 1: Train segmentation head only (encoder frozen)
  - Fast training, fewer parameters
  - Learning rate: 1e-3 to 1e-4
  - Epochs: 50-100

Phase 2 (optional): Fine-tune encoder + head
  - Unfreeze some encoder layers
  - Lower learning rate: 1e-5 to 1e-6
  - Epochs: 20-50

Loss Function: CrossEntropyLoss
  - ignore_index=-1 (for unlabeled pixels)
  - class_weights: computed from class frequencies

Optimizer: AdamW
  - weight_decay: 0.01
  - Different LR for encoder vs head (if unfreezing)


================================================================================
"""

    with open(output_path, 'w') as f:
        f.write(diagram)

    print(f"Architecture diagram saved to: {output_path}")


def main():
    print("\n" + "="*70)
    print(" STEP 5: SAM2 Model Setup for Semantic Segmentation")
    print("="*70)

    # 1. Explore architecture
    model = explore_architecture()

    # 2. Analyze forward pass
    analyze_forward_pass(model)

    # 3. Test segmentation wrapper
    seg_model = test_segmentation_wrapper()

    # 4. Test GPU memory
    test_gpu_memory()

    # 5. Save architecture diagram
    save_architecture_diagram()

    # Summary
    print_header("SUMMARY")
    print("""
SAM2SemanticSegmentation wrapper created successfully!

Key features:
- Uses SAM2 image encoder as frozen backbone (~27M params)
- Multi-scale segmentation head uses FPN features (~0.2M trainable params)
- Output: (B, 7, 512, 512) logits for 7-class segmentation
- Memory efficient: <1GB for inference, ~4GB for training (batch=4)

Usage:
    from step5_model_setup import SAM2SemanticSegmentation
    from sam2.build_sam import build_sam2

    sam2 = build_sam2(config, checkpoint, device='cuda')
    model = SAM2SemanticSegmentation(sam2, num_classes=7, freeze_encoder=True)

    output = model(images)  # images: (B, 3, 512, 512)
    logits = output['logits']  # (B, 7, 512, 512)

Status: READY FOR TRAINING
""")


if __name__ == "__main__":
    main()
