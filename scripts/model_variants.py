"""
Model Variants for SAM2 Semantic Segmentation

Contains different decoder architectures:
1. Simple head (baseline) - from step5_model_setup.py (~395K trainable)
2. UNet decoder - Better spatial resolution with skip connections (~2.25M trainable)
3. UNet Large - Maximum decoder capacity (~7.05M trainable)

Usage:
    from model_variants import create_model

    # Simple head (default, ~395K trainable params)
    model = create_model(variant='simple', device='cuda')

    # UNet decoder (~2.25M trainable params, better for small structures)
    model = create_model(variant='unet', device='cuda')

    # UNet Large decoder (~7.05M trainable params, maximum capacity)
    model = create_model(variant='unet_large', device='cuda')
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from step5_model_setup import (
    CHECKPOINT_PATH, CONFIG_FILE, ENCODER_EMBED_DIM, NUM_CLASSES,
    count_parameters, count_trainable_parameters, format_params,
    MultiScaleSegmentationHead
)


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """
    Double convolution block used in UNet.

    Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block with skip connection.

    Upsample -> Concat with skip -> ConvBlock
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )

        # After concat: in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch (pad if necessary)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]

        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


# =============================================================================
# UNET DECODER
# =============================================================================

class UNetDecoder(nn.Module):
    """
    UNet-style decoder for SAM2 features.

    Takes multi-scale FPN features and produces segmentation mask
    using skip connections and progressive upsampling.

    Architecture:
    ```
    Encoder features:           Decoder:
    1/16 (32x32, 256ch) ------> Conv Block ----+
                                               | Upsample + Concat
    1/8  (64x64, 256ch) ------> Conv Block <---+
                                               |
                                               | Upsample + Concat
    1/4  (128x128, 256ch) ----> Conv Block <---+
                                               |
                                               | Upsample x4
                                               v
                        Final: (512x512) --> num_classes
    ```

    This preserves spatial details better than simple upsampling,
    particularly for small structures like cystic artery/duct.
    """

    def __init__(
        self,
        encoder_channels: int = 256,
        decoder_channels: List[int] = [256, 128, 64],
        num_classes: int = 7,
        target_size: Tuple[int, int] = (512, 512),
        bilinear: bool = True,
    ):
        """
        Args:
            encoder_channels: Number of channels in FPN features (256 for SAM2)
            decoder_channels: Channel progression in decoder [d0, d1, d2]
            num_classes: Number of output classes
            target_size: Final output resolution (H, W)
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
        """
        super().__init__()
        self.target_size = target_size

        # Process the deepest features (1/16 scale, 32x32)
        self.bottom_conv = ConvBlock(encoder_channels, decoder_channels[0])

        # Decoder path with skip connections
        # Up1: 32x32 -> 64x64 (concat with fpn[1])
        self.up1 = UpBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels,
            out_channels=decoder_channels[1],
            bilinear=bilinear,
        )

        # Up2: 64x64 -> 128x128 (concat with fpn[0])
        self.up2 = UpBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels,
            out_channels=decoder_channels[2],
            bilinear=bilinear,
        )

        # Final output convolution
        self.out_conv = nn.Sequential(
            ConvBlock(decoder_channels[2], decoder_channels[2]),
            nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
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
        Forward pass through UNet decoder.

        Args:
            fpn_features: List of FPN tensors from SAM2 encoder
                [0]: (B, 256, 128, 128) - 1/4 scale (highest resolution)
                [1]: (B, 256, 64, 64)   - 1/8 scale
                [2]: (B, 256, 32, 32)   - 1/16 scale (lowest resolution)

        Returns:
            logits: (B, num_classes, target_H, target_W)
        """
        # Unpack FPN features (indexed by resolution: 0=highest, 2=lowest)
        f_high = fpn_features[0]   # (B, 256, 128, 128) - skip connection
        f_mid = fpn_features[1]    # (B, 256, 64, 64)   - skip connection
        f_low = fpn_features[2]    # (B, 256, 32, 32)   - starting point

        # Bottom processing
        x = self.bottom_conv(f_low)  # (B, 256, 32, 32)

        # Decoder path with skip connections
        x = self.up1(x, f_mid)       # (B, 128, 64, 64)
        x = self.up2(x, f_high)      # (B, 64, 128, 128)

        # Output convolution
        x = self.out_conv(x)         # (B, num_classes, 128, 128)

        # Upsample to target size (128 -> 512 = 4x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return x


# =============================================================================
# ENHANCED UNET DECODER (with more capacity)
# =============================================================================

class UNetDecoderLarge(nn.Module):
    """
    Enhanced UNet decoder with more capacity.

    Same architecture as UNetDecoder but with larger channel dimensions
    for better feature learning. Use when simple UNet isn't sufficient.
    """

    def __init__(
        self,
        encoder_channels: int = 256,
        num_classes: int = 7,
        target_size: Tuple[int, int] = (512, 512),
        bilinear: bool = True,
    ):
        super().__init__()
        self.target_size = target_size

        # Larger decoder channels
        d0, d1, d2 = 512, 256, 128

        # Process deepest features
        self.bottom_conv = ConvBlock(encoder_channels, d0)

        # Decoder path
        self.up1 = UpBlock(d0, encoder_channels, d1, bilinear)
        self.up2 = UpBlock(d1, encoder_channels, d2, bilinear)

        # Output with extra refinement
        self.refine = nn.Sequential(
            ConvBlock(d2, d2),
            ConvBlock(d2, d2 // 2),
        )
        self.out_conv = nn.Conv2d(d2 // 2, num_classes, kernel_size=1)

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
        f_high = fpn_features[0]
        f_mid = fpn_features[1]
        f_low = fpn_features[2]

        x = self.bottom_conv(f_low)
        x = self.up1(x, f_mid)
        x = self.up2(x, f_high)

        x = self.refine(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return x


# =============================================================================
# FULL MODEL WITH UNET DECODER
# =============================================================================

class SAM2UNetSegmentation(nn.Module):
    """
    SAM2-based semantic segmentation with UNet decoder.

    Uses SAM2's image encoder as frozen backbone with a UNet-style
    decoder for better spatial resolution and detail preservation.

    Compared to simple head:
    - More trainable parameters (~2.25M vs ~395K)
    - Better preserves spatial details via skip connections
    - Improved performance on small structures
    - Slightly slower forward pass
    """

    def __init__(
        self,
        sam2_model: nn.Module,
        num_classes: int = 7,
        target_size: Tuple[int, int] = (512, 512),
        freeze_encoder: bool = True,
        decoder_channels: List[int] = [256, 128, 64],
        use_large_decoder: bool = False,
    ):
        """
        Args:
            sam2_model: Loaded SAM2 model
            num_classes: Number of segmentation classes
            target_size: Output resolution (H, W)
            freeze_encoder: Whether to freeze the image encoder
            decoder_channels: Channel progression for decoder
            use_large_decoder: Use larger decoder with more capacity
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

        # Create UNet decoder
        if use_large_decoder:
            self.decoder = UNetDecoderLarge(
                encoder_channels=ENCODER_EMBED_DIM,
                num_classes=num_classes,
                target_size=target_size,
            )
        else:
            self.decoder = UNetDecoder(
                encoder_channels=ENCODER_EMBED_DIM,
                decoder_channels=decoder_channels,
                num_classes=num_classes,
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
                - 'features': Encoder features for visualization
        """
        # Get encoder features
        with torch.set_grad_enabled(not self.freeze_encoder):
            encoder_out = self.image_encoder(images)

        # Get segmentation logits from UNet decoder
        logits = self.decoder(encoder_out['backbone_fpn'])

        return {
            'logits': logits,
            'features': encoder_out['vision_features'],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model(
    variant: str = 'simple',
    num_classes: int = NUM_CLASSES,
    target_size: Tuple[int, int] = (512, 512),
    freeze_encoder: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> nn.Module:
    """
    Factory function to create SAM2 segmentation models.

    Args:
        variant: Model variant to create
            - 'simple': Multi-scale head (~395K trainable params)
            - 'unet': UNet decoder (~1.5M trainable params)
            - 'unet_large': Large UNet decoder (~3M trainable params)
        num_classes: Number of segmentation classes
        target_size: Output resolution (H, W)
        freeze_encoder: Whether to freeze SAM2 encoder
        checkpoint_path: Path to trained checkpoint (optional)
        device: Device to load model on

    Returns:
        Configured model ready for training/inference

    Example:
        # For training
        model = create_model(variant='unet', device='cuda')

        # For inference with trained weights
        model = create_model(
            variant='unet',
            checkpoint_path='checkpoints/best_model.pt',
            device='cuda'
        )
    """
    from sam2.build_sam import build_sam2

    # Load SAM2 base model
    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device='cpu',  # Load to CPU first
        mode='eval',
    )

    # Create appropriate model variant
    if variant == 'simple':
        from step5_model_setup import SAM2SemanticSegmentation
        model = SAM2SemanticSegmentation(
            sam2_model=sam2_model,
            num_classes=num_classes,
            target_size=target_size,
            freeze_encoder=freeze_encoder,
            use_multiscale_head=True,
            hidden_dim=128,
        )
    elif variant == 'unet':
        model = SAM2UNetSegmentation(
            sam2_model=sam2_model,
            num_classes=num_classes,
            target_size=target_size,
            freeze_encoder=freeze_encoder,
            decoder_channels=[256, 128, 64],
            use_large_decoder=False,
        )
    elif variant == 'unet_large':
        model = SAM2UNetSegmentation(
            sam2_model=sam2_model,
            num_classes=num_classes,
            target_size=target_size,
            freeze_encoder=freeze_encoder,
            use_large_decoder=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from: simple, unet, unet_large")

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {checkpoint_path}")

    # Move to device
    model = model.to(device)

    return model


# =============================================================================
# TEST AND COMPARISON
# =============================================================================

def compare_architectures():
    """Compare different decoder architectures."""
    print("=" * 70)
    print(" MODEL VARIANTS COMPARISON")
    print("=" * 70)

    from sam2.build_sam import build_sam2

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load SAM2 base model once
    print("\nLoading SAM2 base model...")
    sam2_model = build_sam2(
        config_file=CONFIG_FILE,
        ckpt_path=str(CHECKPOINT_PATH),
        device='cpu',
        mode='eval',
    )

    # Store results
    results = {}

    # Test configurations
    variants = [
        ('simple', 'Multi-scale Head', lambda sam2: create_simple_model(sam2)),
        ('unet', 'UNet Decoder', lambda sam2: create_unet_model(sam2, large=False)),
        ('unet_large', 'UNet Large', lambda sam2: create_unet_model(sam2, large=True)),
    ]

    def create_simple_model(sam2):
        from step5_model_setup import SAM2SemanticSegmentation
        return SAM2SemanticSegmentation(
            sam2_model=sam2,
            num_classes=NUM_CLASSES,
            target_size=(512, 512),
            freeze_encoder=True,
            use_multiscale_head=True,
            hidden_dim=128,
        )

    def create_unet_model(sam2, large=False):
        return SAM2UNetSegmentation(
            sam2_model=sam2,
            num_classes=NUM_CLASSES,
            target_size=(512, 512),
            freeze_encoder=True,
            use_large_decoder=large,
        )

    # Test each variant
    for variant_id, name, create_fn in variants:
        print(f"\n{'-'*50}")
        print(f" Testing: {name}")
        print(f"{'-'*50}")

        # Reload SAM2 for clean model
        sam2_model = build_sam2(
            config_file=CONFIG_FILE,
            ckpt_path=str(CHECKPOINT_PATH),
            device='cpu',
            mode='eval',
        )

        model = create_fn(sam2_model)
        model = model.to(device)
        model.eval()

        # Parameter counts
        total_params = count_parameters(model)
        trainable_params = count_trainable_parameters(model)

        print(f"  Total params: {format_params(total_params)}")
        print(f"  Trainable params: {format_params(trainable_params)}")

        # Forward pass timing
        dummy_input = torch.randn(2, 3, 512, 512, device=device)

        # Warmup
        with torch.no_grad():
            _ = model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Timed forward pass (average of 10 runs)
        n_runs = 10
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                output = model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = (time.time() - start_time) / n_runs * 1000  # ms

        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            gpu_mem = 0

        # Verify output shape
        output_shape = tuple(output['logits'].shape)
        expected_shape = (2, NUM_CLASSES, 512, 512)
        shape_ok = output_shape == expected_shape

        print(f"  Forward time: {elapsed_time:.1f} ms")
        print(f"  GPU memory: {gpu_mem:.0f} MB")
        print(f"  Output shape: {output_shape} {'OK' if shape_ok else 'MISMATCH'}")

        results[variant_id] = {
            'name': name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'forward_time_ms': elapsed_time,
            'gpu_memory_mb': gpu_mem,
            'output_shape': output_shape,
        }

        # Cleanup
        del model, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 70)
    print(" COMPARISON TABLE")
    print("=" * 70)

    # Table header
    print("\n" + "+" + "-"*18 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
    print(f"| {'Metric':<16} | {'Simple Head':^13} | {'UNet':^13} | {'UNet Large':^13} |")
    print("+" + "-"*18 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")

    # Trainable params
    row = f"| {'Trainable params':<16} |"
    for v in ['simple', 'unet', 'unet_large']:
        row += f" {format_params(results[v]['trainable_params']):^13} |"
    print(row)

    # Forward time
    row = f"| {'Forward time':<16} |"
    for v in ['simple', 'unet', 'unet_large']:
        row += f" {results[v]['forward_time_ms']:.1f} ms{' ':^6} |"
    print(row)

    # GPU memory
    row = f"| {'GPU memory':<16} |"
    for v in ['simple', 'unet', 'unet_large']:
        row += f" {results[v]['gpu_memory_mb']:.0f} MB{' ':^6} |"
    print(row)

    print("+" + "-"*18 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
Model Variant Recommendations:

1. Simple Head (~395K params)
   - Fastest training and inference
   - Good baseline performance
   - Use when: Quick experiments, limited GPU memory

2. UNet Decoder (~2.25M params)
   - Better spatial detail preservation
   - Skip connections help small structure segmentation
   - Use when: Need better accuracy on small structures

3. UNet Large (~7.05M params)
   - Maximum decoder capacity
   - Best for complex segmentation tasks
   - Use when: Have GPU memory and need best accuracy

Usage:
    from model_variants import create_model

    # Simple head
    model = create_model(variant='simple', device='cuda')

    # UNet decoder (recommended for Endoscapes)
    model = create_model(variant='unet', device='cuda')

    # Large UNet decoder
    model = create_model(variant='unet_large', device='cuda')
""")

    return results


def test_decoder_outputs():
    """Test decoder output shapes with dummy FPN features."""
    print("\n" + "=" * 70)
    print(" DECODER OUTPUT SHAPE TEST")
    print("=" * 70)

    batch_size = 2

    # Create dummy FPN features matching SAM2 output
    fpn_features = [
        torch.randn(batch_size, 256, 128, 128),  # 1/4 scale
        torch.randn(batch_size, 256, 64, 64),    # 1/8 scale
        torch.randn(batch_size, 256, 32, 32),    # 1/16 scale
    ]

    print(f"\nInput FPN features:")
    for i, f in enumerate(fpn_features):
        print(f"  [{i}]: {tuple(f.shape)}")

    # Test UNet decoder
    print("\nTesting UNet Decoder...")
    decoder = UNetDecoder(
        encoder_channels=256,
        decoder_channels=[256, 128, 64],
        num_classes=NUM_CLASSES,
        target_size=(512, 512),
    )

    output = decoder(fpn_features)
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Expected: ({batch_size}, {NUM_CLASSES}, 512, 512)")
    assert output.shape == (batch_size, NUM_CLASSES, 512, 512), "Shape mismatch!"
    print("  PASSED")

    # Test UNet Large decoder
    print("\nTesting UNet Large Decoder...")
    decoder_large = UNetDecoderLarge(
        encoder_channels=256,
        num_classes=NUM_CLASSES,
        target_size=(512, 512),
    )

    output_large = decoder_large(fpn_features)
    print(f"  Output shape: {tuple(output_large.shape)}")
    assert output_large.shape == (batch_size, NUM_CLASSES, 512, 512), "Shape mismatch!"
    print("  PASSED")

    # Parameter comparison
    print(f"\n{'Decoder':<20} {'Parameters':<15}")
    print("-" * 35)
    print(f"{'UNet':<20} {format_params(count_parameters(decoder)):<15}")
    print(f"{'UNet Large':<20} {format_params(count_parameters(decoder_large)):<15}")

    print("\nAll decoder tests passed!")


if __name__ == "__main__":
    # Run tests
    test_decoder_outputs()
    compare_architectures()
