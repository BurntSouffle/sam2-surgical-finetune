"""
Coarse Segmentation Module

4-class semantic segmentation for endoscopy images:
    0: background
    1: gallbladder
    2: tool
    3: anatomy (combined cystic structures)
"""

from .dataset_coarse import (
    CoarseEndoscapesDataset,
    get_split_paths,
    COARSE_CLASS_NAMES,
    COARSE_NUM_CLASSES,
    COARSE_CLASS_COLORS,
    LABEL_REMAP,
    mask_to_color_coarse,
    compute_class_distribution,
)
