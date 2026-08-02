"""
Coarse Segmentation V2 Module - 3 Classes with Anatomy IGNORED

This version trains only on 3 classes:
    0: background
    1: gallbladder
    2: tool

Anatomy pixels (cystic structures) are IGNORED during training.
At inference, model uncertainty indicates where anatomy might be.

Key Modules:
    - dataset_coarse_v2: Dataset with 3-class mapping
    - train_coarse_v2: Training script
    - evaluate_coarse_v2: Evaluation with search region analysis
    - inference_coarse_v2: Utilities for downstream SAM2 integration
"""

from .dataset_coarse_v2 import (
    CoarseV2EndoscapesDataset,
    get_split_paths,
    COARSE_V2_CLASS_NAMES,
    COARSE_V2_NUM_CLASSES,
    COARSE_V2_CLASS_COLORS,
    LABEL_REMAP_V2,
    mask_to_color_v2,
    compute_class_distribution_v2,
)

from .inference_coarse_v2 import (
    load_v2_model,
    get_anatomy_search_region,
    generate_search_points,
    CoarseV2Inference,
)
