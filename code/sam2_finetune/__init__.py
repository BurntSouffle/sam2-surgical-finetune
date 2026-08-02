"""
SAM2.1 Fine-tuning Pipeline for Endoscapes2023 Dataset.

This package provides tools for:
- Fine-tuning SAM2.1 on surgical segmentation data
- Evaluating model performance with per-class metrics
- Generating pseudo-labels for unlabeled frames
"""

from .config import Config, get_config
from .dataset import EndoscapesDataset, get_dataloaders

__version__ = "1.0.0"
__all__ = ["Config", "get_config", "EndoscapesDataset", "get_dataloaders"]
