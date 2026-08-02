"""
E1 Experiment: Setup Verification Test Script

This script runs quick sanity checks to verify the E1 (Visual Engrams)
implementation is correctly configured and working.

Tests:
1. Config loading and validation
2. Coarse segmentation model loading
3. Dataset creation with engram masks
4. Model building with 5-channel input
5. Forward pass verification
6. Single training step (optional)

Usage:
    python scripts/swincvs_engram/test_e1_setup.py

    # With custom config (e.g., for cluster):
    python scripts/swincvs_engram/test_e1_setup.py \
        --config_path scripts/swincvs_engram/cluster/SwinCVS_engram_config_cluster.yaml

Should complete in under 2 minutes.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional
import traceback

# Setup paths FIRST
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent  # sam2_finetune
DISSERTATION_ROOT = PROJECT_ROOT.parent
SWINCVS_ROOT = DISSERTATION_ROOT / 'SwinCVS'

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))

# Fix SAM2 import path BEFORE any imports that use SAM2
try:
    from sam2_path_fix import fix_sam2_path, verify_sam2_import
    _removed = fix_sam2_path(verbose=False)
    if _removed:
        print(f"[sam2_path_fix] Removed {len(_removed)} problematic paths from sys.path")
except ImportError:
    print("[WARN] sam2_path_fix module not found, SAM2 imports may fail")


# =============================================================================
# GLOBAL CONFIG PATH (can be overridden via --config_path argument)
# =============================================================================
CONFIG_PATH = None  # Set in main() from command line args


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Store test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration = 0.0
        self.details = {}

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration:.2f}s)"


def run_test(name: str, test_func, *args, **kwargs) -> TestResult:
    """Run a test function and capture results."""
    result = TestResult(name)
    start_time = time.time()

    try:
        passed, message, details = test_func(*args, **kwargs)
        result.passed = passed
        result.message = message
        result.details = details or {}
    except Exception as e:
        result.passed = False
        result.message = f"Exception: {str(e)}"
        result.details = {'traceback': traceback.format_exc()}

    result.duration = time.time() - start_time
    return result


def print_result(result: TestResult):
    """Print test result with formatting."""
    status_color = "\033[92m" if result.passed else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    status = "PASS" if result.passed else "FAIL"

    print(f"\n{status_color}[{status}]{reset_color} {result.name}")
    print(f"       Duration: {result.duration:.2f}s")
    if result.message:
        print(f"       Message: {result.message}")
    if result.details and not result.passed:
        for key, value in result.details.items():
            if key != 'traceback':
                print(f"       {key}: {value}")


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_path() -> Path:
    """Get the config path (from global or default)."""
    global CONFIG_PATH
    if CONFIG_PATH is not None:
        return Path(CONFIG_PATH)
    return SCRIPT_DIR / 'config' / 'SwinCVS_engram_config.yaml'


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_config_loading() -> Tuple[bool, str, dict]:
    """Test 1: Config loading and validation."""
    from scripts.f_environment import get_config

    config_path = get_config_path()

    if not config_path.exists():
        return False, f"Config file not found: {config_path}", {}

    config, experiment_name = get_config(str(config_path))

    details = {}

    # Check ENGRAM section exists
    if not hasattr(config, 'ENGRAM'):
        return False, "Config missing ENGRAM section", details

    details['ENGRAM.ENABLED'] = getattr(config.ENGRAM, 'ENABLED', 'MISSING')
    details['ENGRAM.CHECKPOINT_PATH'] = getattr(config.ENGRAM, 'CHECKPOINT_PATH', 'MISSING')
    details['ENGRAM.MASK_MEAN'] = getattr(config.ENGRAM, 'MASK_MEAN', 'MISSING')
    details['ENGRAM.MASK_STD'] = getattr(config.ENGRAM, 'MASK_STD', 'MISSING')

    # Check IN_CHANS = 5
    in_chans = config.BACKBONE.SWINV2.IN_CHANS
    details['BACKBONE.SWINV2.IN_CHANS'] = in_chans

    if in_chans != 5:
        return False, f"IN_CHANS should be 5, got {in_chans}", details

    # Check ENGRAM.ENABLED = True
    if not config.ENGRAM.ENABLED:
        return False, "ENGRAM.ENABLED should be True", details

    details['experiment_name'] = experiment_name

    return True, "Config loaded successfully with correct engram settings", details


def test_coarse_model_loading() -> Tuple[bool, str, dict]:
    """Test 2: Coarse segmentation model loading."""
    import torch
    import numpy as np

    try:
        from coarse_segmentation_v2.inference_coarse_v2 import load_v2_model
    except ImportError as e:
        return False, f"Could not import coarse model: {e}", {}

    # Get checkpoint path from config
    from scripts.f_environment import get_config
    config_path = get_config_path()
    config, _ = get_config(str(config_path))

    checkpoint_path = config.ENGRAM.CHECKPOINT_PATH

    # Handle relative path
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    details = {'checkpoint_path': str(checkpoint_path)}

    if not Path(checkpoint_path).exists():
        return False, f"Checkpoint not found: {checkpoint_path}", details

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    details['device'] = device

    model = load_v2_model(str(checkpoint_path), device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    logits = output['logits']
    details['output_shape'] = list(logits.shape)

    # Verify output shape
    expected_shape = [1, 3, 512, 512]
    if list(logits.shape) != expected_shape:
        return False, f"Output shape {list(logits.shape)} != expected {expected_shape}", details

    # Get predictions and verify classes
    preds = logits.argmax(dim=1).squeeze().cpu().numpy()
    unique_classes = np.unique(preds)
    details['unique_classes'] = unique_classes.tolist()

    # Classes should be subset of {0, 1, 2}
    valid_classes = {0, 1, 2}
    if not set(unique_classes).issubset(valid_classes):
        return False, f"Invalid classes in output: {unique_classes}", details

    return True, "Coarse model loaded and inference successful", details


def test_dataset_creation() -> Tuple[bool, str, dict]:
    """Test 3: Dataset creation with engram masks."""
    import torch

    # Create a mock config for limited testing
    from scripts.f_environment import get_config
    config_path = get_config_path()
    config, _ = get_config(str(config_path))

    # Limit data for quick testing
    original_fraction = config.TRAIN.LIMIT_DATA_FRACTION
    config.TRAIN.LIMIT_DATA_FRACTION = 100  # Use only 1% of data

    details = {}

    try:
        from swincvs_engram.f_dataset_engram import get_engram_datasets

        training_dataset, val_dataset, test_dataset = get_engram_datasets(config)

        details['train_size'] = len(training_dataset)
        details['val_size'] = len(val_dataset)
        details['test_size'] = len(test_dataset)

        if len(training_dataset) == 0:
            return False, "Training dataset is empty", details

        # Get one sample
        sample, label = training_dataset[0]
        details['sample_shape'] = list(sample.shape)
        details['label_shape'] = list(label.shape)

        # Verify shape: (5, 5, 384, 384) for LSTM mode [T, C, H, W]
        expected_shape = [5, 5, 384, 384]
        if list(sample.shape) != expected_shape:
            return False, f"Sample shape {list(sample.shape)} != expected {expected_shape}", details

        # Check channel values
        rgb_channels = sample[:, :3, :, :]  # First 3 channels (RGB)
        mask_channels = sample[:, 3:, :, :]  # Last 2 channels (GB, Tool)

        details['rgb_min'] = float(rgb_channels.min())
        details['rgb_max'] = float(rgb_channels.max())
        details['mask_min'] = float(mask_channels.min())
        details['mask_max'] = float(mask_channels.max())

        # RGB should be normalized (roughly -2 to +2 for ImageNet)
        if rgb_channels.max() > 5 or rgb_channels.min() < -5:
            return False, f"RGB values seem unnormalized: [{rgb_channels.min():.2f}, {rgb_channels.max():.2f}]", details

        # Masks should be normalized around -1 to 1 (with mean=0.5, std=0.5)
        # Denormalized would be 0 or 1
        mask_denorm = mask_channels * 0.5 + 0.5
        details['mask_denorm_min'] = float(mask_denorm.min())
        details['mask_denorm_max'] = float(mask_denorm.max())

        return True, f"Dataset created with {len(training_dataset)} train samples", details

    finally:
        config.TRAIN.LIMIT_DATA_FRACTION = original_fraction


def test_model_building() -> Tuple[bool, str, dict]:
    """Test 4: Model building with 5-channel input."""
    import torch

    from scripts.f_environment import get_config
    config_path = get_config_path()
    config, _ = get_config(str(config_path))

    details = {}

    try:
        from swincvs_engram.f_build_engram import build_engram_model

        # Set inference mode to skip loading pretrained weights issues
        original_inference = config.MODEL.INFERENCE
        config.MODEL.INFERENCE = False

        model = build_engram_model(config)

        config.MODEL.INFERENCE = original_inference

        # Check patch_embed.proj.weight shape
        if hasattr(model, 'backbone'):
            patch_weight = model.backbone.patch_embed.proj.weight
        else:
            patch_weight = model.patch_embed.proj.weight

        details['patch_embed_shape'] = list(patch_weight.shape)

        # Expected: [128, 5, 4, 4]
        expected_shape = [128, 5, 4, 4]
        if list(patch_weight.shape) != expected_shape:
            return False, f"patch_embed shape {list(patch_weight.shape)} != expected {expected_shape}", details

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        details['total_params'] = f"{total_params:,}"
        details['trainable_params'] = f"{trainable_params:,}"

        # Test forward pass with dummy input
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        # Dummy input: [B, T, C, H, W] = [1, 5, 5, 384, 384]
        dummy_input = torch.randn(1, 5, 5, 384, 384).to(device)

        with torch.no_grad():
            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                out_swin, out_lstm = model(dummy_input)
                output = out_lstm
            else:
                output = model(dummy_input)

        details['output_shape'] = list(output.shape)

        # Expected output: [1, 3] (3 CVS criteria)
        expected_output_shape = [1, 3]
        if list(output.shape) != expected_output_shape:
            return False, f"Output shape {list(output.shape)} != expected {expected_output_shape}", details

        return True, f"Model built successfully with {trainable_params:,} trainable params", details

    except Exception as e:
        details['error'] = str(e)
        return False, f"Model building failed: {e}", details


def test_forward_pass() -> Tuple[bool, str, dict]:
    """Test 5: Forward pass verification."""
    import torch

    from scripts.f_environment import get_config
    config_path = get_config_path()
    config, _ = get_config(str(config_path))

    details = {}

    try:
        from swincvs_engram.f_build_engram import build_engram_model

        config.MODEL.INFERENCE = False
        model = build_engram_model(config)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        # Test with batch size > 1
        batch_size = 2
        dummy_input = torch.randn(batch_size, 5, 5, 384, 384).to(device)

        details['input_shape'] = list(dummy_input.shape)
        details['device'] = device

        start_time = time.time()

        with torch.no_grad():
            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                out_swin, out_lstm = model(dummy_input)
                output = out_lstm
                details['multiclassifier'] = True
                details['out_swin_shape'] = list(out_swin.shape)
            else:
                output = model(dummy_input)
                details['multiclassifier'] = False

        forward_time = time.time() - start_time
        details['forward_time_ms'] = round(forward_time * 1000, 2)
        details['output_shape'] = list(output.shape)

        # Check output
        expected_shape = [batch_size, 3]
        if list(output.shape) != expected_shape:
            return False, f"Output shape {list(output.shape)} != expected {expected_shape}", details

        # Check output values are valid (not NaN, not Inf)
        if torch.isnan(output).any():
            return False, "Output contains NaN values", details
        if torch.isinf(output).any():
            return False, "Output contains Inf values", details

        details['output_sample'] = output[0].cpu().tolist()

        return True, f"Forward pass successful in {forward_time*1000:.1f}ms", details

    except Exception as e:
        details['error'] = str(e)
        return False, f"Forward pass failed: {e}", details


def test_single_training_step() -> Tuple[bool, str, dict]:
    """Test 6: Single training step (optional)."""
    import torch
    import torch.nn as nn

    from scripts.f_environment import get_config
    config_path = get_config_path()
    config, _ = get_config(str(config_path))

    details = {}

    try:
        from swincvs_engram.f_build_engram import build_engram_model
        from swincvs_engram.f_dataset_engram import get_engram_datasets, get_dataloaders

        # Build model
        config.MODEL.INFERENCE = False
        model = build_engram_model(config)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()

        # Create small dataset
        config.TRAIN.LIMIT_DATA_FRACTION = 100  # Use 1% of data
        config.TRAIN.BATCH_SIZE = 2

        training_dataset, _, _ = get_engram_datasets(config)
        train_loader, _, _ = get_dataloaders(config, training_dataset, training_dataset, training_dataset)

        # Get one batch
        samples, targets = next(iter(train_loader))
        samples = samples.to(device)
        targets = targets.to(device)

        details['batch_samples_shape'] = list(samples.shape)
        details['batch_targets_shape'] = list(targets.shape)

        # Setup loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward pass
        optimizer.zero_grad()

        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            out_swin, out_lstm = model(samples)
            loss = criterion(out_lstm, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

        details['loss_value'] = float(loss.item())

        # Check loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            return False, f"Invalid loss value: {loss.item()}", details

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_gradients = False
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norms.append(param.grad.norm().item())

        if not has_gradients:
            return False, "No gradients computed", details

        details['num_params_with_grad'] = len(grad_norms)
        details['mean_grad_norm'] = round(sum(grad_norms) / len(grad_norms), 6)
        details['max_grad_norm'] = round(max(grad_norms), 6)

        # Optimizer step
        optimizer.step()

        return True, f"Training step successful, loss={loss.item():.4f}", details

    except Exception as e:
        details['error'] = str(e)
        details['traceback'] = traceback.format_exc()
        return False, f"Training step failed: {e}", details


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='E1 Experiment Setup Verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default config (local):
    python scripts/swincvs_engram/test_e1_setup.py

    # Use cluster config:
    python scripts/swincvs_engram/test_e1_setup.py \\
        --config_path scripts/swincvs_engram/cluster/SwinCVS_engram_config_cluster.yaml
        """
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Path to config file (default: config/SwinCVS_engram_config.yaml)'
    )
    return parser.parse_args()


def main():
    """Run all E1 setup verification tests."""
    global CONFIG_PATH

    # Parse command line arguments
    args = parse_args()
    if args.config_path:
        CONFIG_PATH = args.config_path

    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  E1 EXPERIMENT: SETUP VERIFICATION".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    print(f"\nScript directory: {SCRIPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SwinCVS root: {SWINCVS_ROOT}")
    if CONFIG_PATH:
        print(f"Config path: {CONFIG_PATH}")

    # Check CUDA availability
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = []
    total_start = time.time()

    # =========================================================================
    # TEST 1: Config Loading
    # =========================================================================
    print_section("TEST 1: Config Loading")
    result = run_test("Config Loading", test_config_loading)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # TEST 2: Coarse Model Loading
    # =========================================================================
    print_section("TEST 2: Coarse Model Loading")
    result = run_test("Coarse Model Loading", test_coarse_model_loading)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # TEST 3: Dataset Creation
    # =========================================================================
    print_section("TEST 3: Dataset Creation")
    result = run_test("Dataset Creation", test_dataset_creation)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # TEST 4: Model Building
    # =========================================================================
    print_section("TEST 4: Model Building")
    result = run_test("Model Building", test_model_building)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # TEST 5: Forward Pass
    # =========================================================================
    print_section("TEST 5: Forward Pass")
    result = run_test("Forward Pass", test_forward_pass)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # TEST 6: Single Training Step
    # =========================================================================
    print_section("TEST 6: Single Training Step")
    result = run_test("Single Training Step", test_single_training_step)
    print_result(result)
    if result.passed:
        for key, value in result.details.items():
            if key != 'traceback':
                print(f"       {key}: {value}")
    results.append(result)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for result in results:
        status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {result.name} ({result.duration:.2f}s)")

    print(f"\n  Total: {passed} passed, {failed} failed")
    print(f"  Total time: {total_time:.2f}s")

    # Final status
    if failed == 0:
        print("\n" + "\033[92m" + "="*70)
        print(" ALL TESTS PASSED - E1 SETUP IS READY!")
        print("="*70 + "\033[0m")
        print("\nYou can now run the full training with:")
        print("  python scripts/swincvs_engram/SwinCVS_engram.py")
        return 0
    else:
        print("\n" + "\033[91m" + "="*70)
        print(f" {failed} TEST(S) FAILED - PLEASE FIX BEFORE TRAINING")
        print("="*70 + "\033[0m")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
