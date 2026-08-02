"""
Configuration for SAM2.1 Fine-tuning on Endoscapes2023 Dataset.

Replace placeholder paths before running:
- PROJECT_STORE_PATH: Your project storage directory (e.g., /SAN/medic/your_project)
- DATASET_PATH: Path to Endoscapes dataset
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class PathConfig:
    """Path configuration - UPDATE THESE PATHS."""

    # === REPLACE THESE PLACEHOLDERS ===
    project_store: str = "/project/YOUR_USERNAME"  # Your project store path
    dataset_root: str = "/data/endoscapes"          # Endoscapes dataset path

    # Derived paths (no need to modify)
    @property
    def code_dir(self) -> Path:
        return Path(self.project_store) / "code" / "sam2_finetune"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.project_store) / "checkpoints"

    @property
    def pretrained_dir(self) -> Path:
        return self.checkpoint_dir / "pretrained"

    @property
    def finetuned_dir(self) -> Path:
        return self.checkpoint_dir / "finetuned"

    @property
    def output_dir(self) -> Path:
        return Path(self.project_store) / "outputs"

    @property
    def log_dir(self) -> Path:
        return Path(self.project_store) / "logs"

    @property
    def generated_masks_dir(self) -> Path:
        return self.output_dir / "generated_masks"

    # Dataset paths
    @property
    def train_images(self) -> Path:
        return Path(self.dataset_root) / "train_seg"

    @property
    def val_images(self) -> Path:
        return Path(self.dataset_root) / "val_seg"

    @property
    def test_images(self) -> Path:
        return Path(self.dataset_root) / "test_seg"

    @property
    def semantic_masks(self) -> Path:
        return Path(self.dataset_root) / "semseg"

    @property
    def train_annotations(self) -> Path:
        return Path(self.dataset_root) / "train" / "annotation_coco.json"

    @property
    def val_annotations(self) -> Path:
        return Path(self.dataset_root) / "val" / "annotation_coco.json"

    @property
    def test_annotations(self) -> Path:
        return Path(self.dataset_root) / "test" / "annotation_coco.json"


@dataclass
class ModelConfig:
    """SAM2.1 Model configuration."""

    # Model variant
    model_name: str = "sam2.1_hiera_large"
    model_config: str = "sam2.1_hiera_l.yaml"

    # Pretrained checkpoint URL
    checkpoint_url: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    checkpoint_filename: str = "sam2.1_hiera_large.pt"

    # Model architecture params
    image_size: int = 1024  # SAM2 native resolution

    # Which components to train
    train_image_encoder: bool = True
    train_mask_decoder: bool = True
    train_prompt_encoder: bool = False  # Usually frozen

    # Memory optimization
    use_gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""

    # Endoscapes classes
    num_classes: int = 7
    class_names: List[str] = field(default_factory=lambda: [
        "background",      # 0
        "cystic_plate",    # 1
        "calot_triangle",  # 2 (hepatocystic triangle)
        "cystic_artery",   # 3
        "cystic_duct",     # 4
        "gallbladder",     # 5
        "tool",            # 6
    ])

    # Classes to train on (exclude background)
    train_classes: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])

    # Original image dimensions
    original_height: int = 480
    original_width: int = 854

    # Point prompt configuration
    min_points_per_mask: int = 1
    max_points_per_mask: int = 10

    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_prob: float = 0.3
    affine_prob: float = 0.3

    # Minimum mask area (pixels) to include in training
    min_mask_area: int = 100


@dataclass
class TrainConfig:
    """Training configuration."""

    # Training params
    num_epochs: int = 30
    batch_size: int = 2  # Per GPU
    num_workers: int = 4

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # Learning rates (differential)
    lr_image_encoder: float = 3e-6   # Lower for pretrained features
    lr_mask_decoder: float = 5e-5    # Higher for decoder
    lr_prompt_encoder: float = 1e-5  # If trained

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-7

    # Loss weights
    dice_weight: float = 1.0
    focal_weight: float = 20.0
    iou_weight: float = 1.0

    # Focal loss params
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 for A100s

    # Gradient accumulation (effective batch = batch_size * num_gpus * grad_accum)
    gradient_accumulation_steps: int = 1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 2
    save_best: bool = True
    best_metric: str = "val_miou"

    # Validation
    val_every_n_epochs: int = 1

    # Reproducibility
    seed: int = 42

    # Distributed training
    num_gpus: int = 4


@dataclass
class Config:
    """Main configuration class."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Experiment name
    experiment_name: str = "sam2_endoscapes_finetune"

    # Resume from checkpoint
    resume_from: Optional[str] = None

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [
            self.paths.checkpoint_dir,
            self.paths.pretrained_dir,
            self.paths.finetuned_dir,
            self.paths.output_dir,
            self.paths.log_dir,
            self.paths.generated_masks_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, **kwargs) -> "Config":
        """Create config with custom arguments."""
        config = cls()

        # Update paths
        if "project_store" in kwargs:
            config.paths.project_store = kwargs["project_store"]
        if "dataset_root" in kwargs:
            config.paths.dataset_root = kwargs["dataset_root"]

        # Update training params
        for key, value in kwargs.items():
            if hasattr(config.train, key):
                setattr(config.train, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)

        return config


def get_config(**kwargs) -> Config:
    """Get configuration with optional overrides."""
    return Config.from_args(**kwargs)


if __name__ == "__main__":
    # Print default configuration
    config = get_config()
    print("=== SAM2 Fine-tuning Configuration ===")
    print(f"\nPaths:")
    print(f"  Project store: {config.paths.project_store}")
    print(f"  Dataset root: {config.paths.dataset_root}")
    print(f"\nModel:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Image size: {config.model.image_size}")
    print(f"\nTraining:")
    print(f"  Epochs: {config.train.num_epochs}")
    print(f"  Batch size: {config.train.batch_size} per GPU")
    print(f"  LR (encoder): {config.train.lr_image_encoder}")
    print(f"  LR (decoder): {config.train.lr_mask_decoder}")
    print(f"\nData:")
    print(f"  Classes: {config.data.class_names}")
    print(f"  Points per mask: {config.data.min_points_per_mask}-{config.data.max_points_per_mask}")
