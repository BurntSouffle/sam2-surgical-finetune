"""
Step 2: SAM2 Installation Test Script
Tests SAM2 import, checkpoint loading, and forward pass.
"""

import os


def print_separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def test_sam2_import() -> bool:
    print_separator("SAM2 Import Test")
    try:
        import sam2
        print(f"SAM2 package location: {sam2.__path__}")

        from sam2.build_sam import build_sam2
        print("build_sam2 imported: SUCCESS")

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("SAM2ImagePredictor imported: SUCCESS")

        return True
    except ImportError as e:
        print(f"Import FAILED: {e}")
        return False


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(num_params: int) -> str:
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    return str(num_params)


def get_gpu_memory_mb() -> float:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024**2)
    return 0.0


def test_checkpoint_loading():
    print_separator("Checkpoint Loading Test")

    import torch
    from sam2.build_sam import build_sam2

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    checkpoint_path = os.path.join(project_dir, "checkpoints", "sam2.1_hiera_tiny.pt")
    config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"

    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_path}")

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None

    checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
    print(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")

    # Clear GPU memory before loading
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nLoading model on CPU first...")
    model = build_sam2(
        config_file=config_file,
        ckpt_path=checkpoint_path,
        device="cpu",
        mode="eval"
    )
    print("Model loaded on CPU: SUCCESS")

    return model


def test_model_parameters(model):
    print_separator("Model Parameters")

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print(f"Total parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})")

    # Expected ~38.9M for tiny model
    expected_min = 35_000_000
    expected_max = 45_000_000

    if expected_min <= total_params <= expected_max:
        print(f"Parameter count: VALID (expected ~38.9M)")
    else:
        print(f"WARNING: Parameter count outside expected range (35M-45M)")


def test_gpu_loading(model):
    print_separator("GPU Loading Test")

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return None

    device = torch.device("cuda:0")
    print(f"Target device: {device}")

    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = get_gpu_memory_mb()
    print(f"GPU memory before loading: {mem_before:.2f} MB")

    # Move model to GPU
    model = model.to(device)
    torch.cuda.synchronize()

    mem_after = get_gpu_memory_mb()
    mem_reserved = get_gpu_memory_reserved_mb()

    print(f"GPU memory after loading: {mem_after:.2f} MB")
    print(f"GPU memory reserved: {mem_reserved:.2f} MB")
    print(f"Model memory footprint: {mem_after - mem_before:.2f} MB")
    print("GPU loading: SUCCESS")

    return model


def test_forward_pass(model):
    print_separator("Forward Pass Test")

    import torch
    import numpy as np
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = next(model.parameters()).device
    print(f"Model device: {device}")

    # Clear memory stats
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_mb()
    print(f"GPU memory before forward pass: {mem_before:.2f} MB")

    # Create predictor
    predictor = SAM2ImagePredictor(model)

    # Create dummy image (512x512 RGB)
    print("\nCreating dummy 512x512 RGB image...")
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Set image (this runs the image encoder)
    print("Running image encoder (set_image)...")
    with torch.no_grad():
        predictor.set_image(dummy_image)

    torch.cuda.synchronize()
    mem_after_encode = get_gpu_memory_mb()
    print(f"GPU memory after image encoding: {mem_after_encode:.2f} MB")

    # Run prediction with a point prompt
    print("\nRunning mask prediction with point prompt...")
    point_coords = np.array([[256, 256]])  # Center of image
    point_labels = np.array([1])  # Foreground point

    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

    torch.cuda.synchronize()
    mem_after_predict = get_gpu_memory_mb()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"\nPrediction results:")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Scores: {scores}")
    print(f"  Number of masks: {len(masks)}")

    print(f"\nGPU memory after prediction: {mem_after_predict:.2f} MB")
    print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

    print("\nForward pass: SUCCESS")

    return True


def main():
    print("\n" + "="*60)
    print(" SAM2.1 Installation Test")
    print("="*60)

    # Test 1: Import
    if not test_sam2_import():
        print("\nFATAL: SAM2 import failed")
        return

    # Test 2: Load checkpoint
    model = test_checkpoint_loading()
    if model is None:
        print("\nFATAL: Checkpoint loading failed")
        return

    # Test 3: Parameter count
    test_model_parameters(model)

    # Test 4: GPU loading
    model = test_gpu_loading(model)
    if model is None:
        print("\nFATAL: GPU loading failed")
        return

    # Test 5: Forward pass
    success = test_forward_pass(model)

    # Summary
    print_separator("Summary")

    import torch

    print("SAM2 import: SUCCESS")
    print("Checkpoint loaded: sam2.1_hiera_tiny.pt")
    print(f"Parameters: {format_params(count_parameters(model))}")
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")

    if success:
        print("\nStatus: SAM2.1-Tiny READY FOR FINE-TUNING")
    else:
        print("\nStatus: TESTS FAILED")

    print()


if __name__ == "__main__":
    main()
