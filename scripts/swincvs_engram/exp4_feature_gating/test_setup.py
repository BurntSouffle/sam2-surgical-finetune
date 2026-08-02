"""
Quick test to verify E4 setup before full training.
"""

import os
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
SWINCVS_ROOT = PROJECT_ROOT / 'SwinCVS'
SAM2_FINETUNE_ROOT = PROJECT_ROOT / 'sam2_finetune'

for path in [str(PROJECT_ROOT), str(SWINCVS_ROOT), str(SWINCVS_ROOT / 'scripts'),
             str(SAM2_FINETUNE_ROOT / 'scripts')]:
    if path not in sys.path:
        sys.path.insert(0, path)


def test_mask_encoder():
    """Test MaskEncoder forward pass."""
    print("\n" + "="*60)
    print("TEST 1: MaskEncoder")
    print("="*60)

    try:
        from model_gating import MaskEncoder

        encoder = MaskEncoder(feature_dim=1024)
        encoder.eval()

        # Test input
        B, T = 2, 5
        dummy_masks = torch.randn(B * T, 2, 384, 384)

        start = time.time()
        with torch.no_grad():
            gate = encoder(dummy_masks)
        elapsed = time.time() - start

        print(f"  Input shape: {dummy_masks.shape}")
        print(f"  Output shape: {gate.shape}")
        print(f"  Expected: torch.Size([{B*T}, 1024])")
        print(f"  Time: {elapsed*1000:.1f}ms")

        # Count params
        total = sum(p.numel() for p in encoder.parameters())
        print(f"  Parameters: {total:,} ({total/1e6:.2f}M)")

        assert gate.shape == (B * T, 1024), f"Wrong shape: {gate.shape}"
        print("\n  [PASS] MaskEncoder works!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading pretrained checkpoint and wrapping with gating."""
    print("\n" + "="*60)
    print("TEST 2: Model Loading")
    print("="*60)

    checkpoint_path = SWINCVS_ROOT / 'weights' / 'SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt'

    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  [SKIP] Cannot test model loading")
        return None

    try:
        from model_gating import create_gated_model_from_checkpoint

        print(f"  Loading from: {checkpoint_path}")
        start = time.time()
        model = create_gated_model_from_checkpoint(
            str(checkpoint_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            freeze_backbone=True,
            freeze_lstm=False
        )
        elapsed = time.time() - start
        print(f"  Load time: {elapsed:.2f}s")

        # Print parameter counts
        counts = model.count_parameters()
        print(f"\n  Parameters:")
        print(f"    Total: {counts['total']:,}")
        print(f"    Trainable: {counts['trainable']:,}")
        print(f"    Frozen: {counts['frozen']:,}")
        print(f"    - Backbone: {counts['backbone']:,}")
        print(f"    - LSTM: {counts['lstm']:,}")
        print(f"    - FC: {counts['fc']:,}")
        print(f"    - MaskEncoder: {counts['mask_encoder']:,}")

        print("\n  [PASS] Model loaded successfully!")
        return model

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)

    if model is None:
        print("  [SKIP] No model to test")
        return None

    try:
        device = next(model.parameters()).device
        model.eval()

        # Dummy data
        B, T = 2, 5
        frames = torch.randn(B, T, 3, 384, 384, device=device)
        masks = torch.randn(B, T, 2, 384, 384, device=device)

        print(f"  Input frames: {frames.shape}")
        print(f"  Input masks: {masks.shape}")

        start = time.time()
        with torch.no_grad():
            output = model(frames, masks)
        elapsed = time.time() - start

        print(f"  Output: {output.shape}")
        print(f"  Expected: torch.Size([{B}, 3])")
        print(f"  Time: {elapsed*1000:.1f}ms")

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print("\n  [PASS] Forward pass works!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(model):
    """Test backward pass and gradient flow."""
    print("\n" + "="*60)
    print("TEST 4: Backward Pass")
    print("="*60)

    if model is None:
        print("  [SKIP] No model to test")
        return None

    try:
        device = next(model.parameters()).device
        model.train()

        # Dummy data
        B, T = 2, 5
        frames = torch.randn(B, T, 3, 384, 384, device=device)
        masks = torch.randn(B, T, 2, 384, 384, device=device)
        labels = torch.randint(0, 2, (B, 3), device=device).float()

        # Forward
        output = model(frames, masks)

        # Loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, labels)
        print(f"  Loss: {loss.item():.4f}")

        # Backward
        loss.backward()

        # Check gradients
        mask_enc_grads = []
        for name, param in model.mask_encoder.named_parameters():
            if param.grad is not None:
                mask_enc_grads.append(param.grad.norm().item())

        backbone_grads = []
        for name, param in model.swinv2_model.named_parameters():
            if param.grad is not None:
                backbone_grads.append(param.grad.norm().item())

        print(f"\n  MaskEncoder gradients: {len(mask_enc_grads)} params have grads")
        if mask_enc_grads:
            print(f"    Mean grad norm: {sum(mask_enc_grads)/len(mask_enc_grads):.6f}")

        print(f"  Backbone gradients: {len(backbone_grads)} params have grads")
        if backbone_grads:
            print(f"    Mean grad norm: {sum(backbone_grads)/len(backbone_grads):.6f}")
        else:
            print("    (Expected: 0 since backbone is frozen)")

        assert len(mask_enc_grads) > 0, "MaskEncoder has no gradients!"

        print("\n  [PASS] Backward pass works!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mask_generator():
    """Test MaskGenerator with coarse model."""
    print("\n" + "="*60)
    print("TEST 5: MaskGenerator")
    print("="*60)

    checkpoint_path = SAM2_FINETUNE_ROOT / 'checkpoints' / 'coarse_segmentation_v2' / 'run_20260120_234049' / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  [SKIP] Cannot test mask generator")
        return None

    try:
        from dataset_gating import MaskGenerator
        import numpy as np

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Loading coarse model on {device}...")

        start = time.time()
        mask_gen = MaskGenerator(str(checkpoint_path), device=device)
        elapsed = time.time() - start
        print(f"  Load time: {elapsed:.2f}s")

        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)

        start = time.time()
        gb_mask, tool_mask = mask_gen.generate_masks(dummy_image, "test_image")
        elapsed = time.time() - start

        print(f"\n  Input: {dummy_image.shape}")
        print(f"  GB mask: {gb_mask.shape}, range [{gb_mask.min():.3f}, {gb_mask.max():.3f}]")
        print(f"  Tool mask: {tool_mask.shape}, range [{tool_mask.min():.3f}, {tool_mask.max():.3f}]")
        print(f"  Time: {elapsed*1000:.1f}ms")

        print("\n  [PASS] MaskGenerator works!")
        return True

    except Exception as e:
        print(f"\n  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("E4 FEATURE GATING - SETUP VERIFICATION")
    print("="*60)

    results = {}

    # Test 1: MaskEncoder
    results['mask_encoder'] = test_mask_encoder()

    # Test 2: Model loading
    model = test_model_loading()
    results['model_loading'] = model is not None and model is not False

    # Test 3: Forward pass
    if results['model_loading']:
        results['forward_pass'] = test_forward_pass(model)
    else:
        results['forward_pass'] = None

    # Test 4: Backward pass
    if results['model_loading']:
        results['backward_pass'] = test_backward_pass(model)
    else:
        results['backward_pass'] = None

    # Test 5: MaskGenerator
    results['mask_generator'] = test_mask_generator()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is True:
            status = "[PASS]"
            passed += 1
        elif result is False:
            status = "[FAIL]"
            failed += 1
        else:
            status = "[SKIP]"
            skipped += 1
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\nAll tests passed! Ready to run training.")
        print("\nRun training with:")
        print(f"  py {SCRIPT_DIR / 'train_gating.py'}")
    else:
        print("\nSome tests failed. Fix issues before training.")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
