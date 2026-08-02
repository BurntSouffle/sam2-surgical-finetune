"""
SwinCVS Engram Module - Visual Engrams for CVS Classification

Experiment E1: Augmenting SwinCVS with explicit segmentation masks as additional
input channels to provide spatial priors for CVS criteria assessment.

Architecture:
    - Input: 5-channel tensor (RGB + GB mask + Tool mask)
    - Backbone: Modified SwinV2 to accept 5 channels
    - LSTM: Temporal modeling across 5-frame sequences
    - Output: 3 CVS criteria predictions (C1, C2, C3)

Key Modules:
    - dataset_engram: Dataset with mask channel concatenation
    - model_engram: Modified SwinCVS with 5-channel input
    - SwinCVS_engram: Main training script
    - inference_engram: Inference utilities

Dependencies:
    - Coarse segmentation model for GB/Tool mask generation
    - Original SwinCVS for baseline comparison
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Version info
__version__ = "0.1.0"
__experiment__ = "E1"
__description__ = "Visual Engrams for CVS Classification"
