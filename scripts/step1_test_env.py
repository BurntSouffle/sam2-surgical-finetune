"""
Step 1: Environment Test Script for SAM2.1 Fine-tuning
Tests Python, PyTorch, CUDA, and GPU configuration.
"""

import sys


def print_separator(title: str) -> None:
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)


def test_python_version() -> None:
    print_separator("Python Version")
    print(f"Python: {sys.version}")
    print(f"Version info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def test_pytorch() -> None:
    print_separator("PyTorch Configuration")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        else:
            print("WARNING: CUDA is not available!")

    except ImportError as e:
        print(f"ERROR: Failed to import torch: {e}")


def test_gpu() -> None:
    print_separator("GPU Information")
    try:
        import torch

        if not torch.cuda.is_available():
            print("ERROR: No CUDA-capable GPU detected")
            return

        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total VRAM: {vram_gb:.2f} GB")
            print(f"  Multi-processors: {props.multi_processor_count}")

    except Exception as e:
        print(f"ERROR: Failed to get GPU info: {e}")


def test_gpu_compute() -> None:
    print_separator("GPU Compute Test")
    try:
        import torch

        if not torch.cuda.is_available():
            print("SKIPPED: No GPU available")
            return

        device = torch.device("cuda:0")
        print(f"Using device: {device}")

        # Create tensors on GPU
        print("\nCreating tensors on GPU...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Matrix multiplication
        print("Performing matrix multiplication (1000x1000)...")
        c = torch.matmul(a, b)

        # Synchronize and verify
        torch.cuda.synchronize()
        print(f"Result shape: {c.shape}")
        print(f"Result sum: {c.sum().item():.4f}")

        # Memory usage
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        print(f"\nGPU Memory allocated: {allocated:.2f} MB")
        print(f"GPU Memory reserved: {reserved:.2f} MB")

        print("\nGPU compute test: PASSED")

    except Exception as e:
        print(f"GPU compute test: FAILED")
        print(f"Error: {e}")


def test_dependencies() -> None:
    print_separator("Additional Dependencies")

    packages = [
        ("cv2", "opencv-python"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard"),
    ]

    for import_name, package_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {package_name}: {version}")
        except ImportError:
            print(f"  {package_name}: NOT INSTALLED")


def main() -> None:
    print("\n" + "="*50)
    print(" SAM2.1 Fine-tuning Environment Test")
    print("="*50)

    test_python_version()
    test_pytorch()
    test_gpu()
    test_gpu_compute()
    test_dependencies()

    print_separator("Summary")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Status: READY FOR SAM2.1 FINE-TUNING")
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {vram:.2f} GB")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("Status: NOT READY - CUDA unavailable")
    except Exception as e:
        print(f"Status: NOT READY - {e}")

    print("\n")


if __name__ == "__main__":
    main()
