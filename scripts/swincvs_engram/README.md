# E1 - Visual Engrams for CVS Classification

## Hypothesis

Providing explicit GB+Tool segmentation masks as additional input channels helps SwinCVS assess C1 criteria.

**Rationale**: Thread 0 findings revealed that C1 (hepatocystic triangle clearance) is an *assessment problem*, not a detection problem. The model can locate the relevant anatomy but struggles to assess whether clearance has been achieved. By providing explicit spatial priors through segmentation masks, we give the model direct access to GB and Tool boundaries, which should improve its ability to assess the spatial relationship between these structures.

## Architecture Change

```
Original SwinCVS:
    Input: 3-channel RGB (H, W, 3)

E1 SwinCVS-Engram:
    Input: 5-channel (H, W, 5)
        - Channels 0-2: RGB image
        - Channel 3: Gallbladder mask (binary, from coarse segmentation)
        - Channel 4: Tool mask (binary, from coarse segmentation)
```

### Model Modifications

1. **First Conv Layer**: Modify SwinV2 patch embedding to accept 5 input channels instead of 3
2. **Mask Generation**: Use trained coarse segmentation model (V2) to generate GB/Tool masks
3. **Mask Preprocessing**:
   - Resize masks to match input resolution (384x384)
   - Normalize to [0, 1] range
   - Concatenate with normalized RGB

## Directory Structure

```
swincvs_engram/
├── __init__.py              # Module init with path setup
├── README.md                # This file
├── config/
│   ├── __init__.py
│   └── SwinCVS_engram_config.yaml  # Training configuration
├── dataset_engram.py        # Dataset with mask concatenation
├── model_engram.py          # Modified SwinCVS architecture
├── SwinCVS_engram.py        # Main training script
└── inference_engram.py      # Inference utilities
```

## How to Run

```bash
# From project root
cd sam2_finetune/scripts/swincvs_engram

# Train the model
python SwinCVS_engram.py --config_path config/SwinCVS_engram_config.yaml

# Or with custom options
python SwinCVS_engram.py \
    --config_path config/SwinCVS_engram_config.yaml \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4
```

## Expected Improvement

- **Primary**: Better C1 mAP due to explicit spatial priors for hepatocystic triangle assessment
- **Secondary**: Potentially improved C2 (cystic structures) as tool position relative to anatomy is explicit
- **C3**: Minimal impact expected (already high performance in baseline)

## Setup

### 1. Install Dependencies

```bash
# From sam2_finetune/scripts/swincvs_engram directory
pip install -r requirements_e1.txt

# Or install key missing packages manually:
pip install timm  # Required for SwinV2 backbone
```

### 2. Install SAM2 Package (REQUIRED)

SAM2 must be installed as an editable package to avoid import shadowing:

```bash
# From sam2_finetune directory
pip install -e sam2/

# Or from the sam2 directory itself
cd sam2
pip install -e .
```

**Note**: Running Python from the DISSERTATION parent directory can cause the `sam2` folder to shadow the installed `sam2` package. The `sam2_path_fix.py` module handles this automatically, but proper installation is recommended.

### 3. Verify Setup

```bash
# Run the verification tests
python scripts/swincvs_engram/test_e1_setup.py
```

All 6 tests should pass:
- Test 1: Config Loading
- Test 2: Coarse Model Loading
- Test 3: Dataset Creation
- Test 4: Model Building
- Test 5: Forward Pass
- Test 6: Single Training Step

## Dependencies

### Pretrained Models Required
1. **Coarse Segmentation V2**: `checkpoints/coarse_segmentation_v2/run_20260120_234049/best_model.pt`
   - Provides GB mask (class 1) and Tool mask (class 2)
   - Input: 512x512 RGB, Output: 512x512 3-class segmentation

2. **SwinV2 Backbone**: Downloaded automatically via timm

### Python Dependencies
- PyTorch >= 2.0
- **timm** (for SwinV2) - MUST be installed!
- SAM2 (installed as editable package)
- Same as original SwinCVS

See `requirements_e1.txt` for complete list.

## Reference

This experiment is based on findings from Thread 0 analysis:

> "C1 is fundamentally an assessment problem - the model can detect the hepatocystic triangle region but struggles to determine whether adequate clearance has been achieved. This suggests that explicit spatial information about gallbladder and tool positions could help the model make better assessments."

Key insights:
- C1 requires understanding spatial relationships between GB, tools, and the hepatocystic triangle
- Providing explicit segmentation masks as "visual engrams" gives the model direct access to these spatial boundaries
- This is analogous to how surgeons mentally register the positions of key structures before assessing clearance

## Baseline Comparison

| Metric | SwinCVS Baseline | SwinCVS-Engram (E1) |
|--------|------------------|---------------------|
| C1 mAP | TBD              | Target: +5-10%      |
| C2 mAP | TBD              | Target: +2-5%       |
| C3 mAP | TBD              | ~same               |
| Overall mAP | TBD         | Target: +3-5%       |

## Implementation Notes

### Mask Generation Pipeline

```python
# Pseudo-code for inference pipeline
from coarse_segmentation_v2.inference_coarse_v2 import CoarseV2Inference

# Load coarse segmentation model
coarse_model = CoarseV2Inference(checkpoint_path)

# For each frame in sequence
for frame in frames:
    # Get segmentation
    result = coarse_model.process_image(frame)
    prediction = result['prediction']  # (512, 512)

    # Extract masks
    gb_mask = (prediction == 1).astype(np.float32)    # Gallbladder
    tool_mask = (prediction == 2).astype(np.float32)  # Tool

    # Resize to SwinCVS input size (384x384)
    gb_mask = cv2.resize(gb_mask, (384, 384))
    tool_mask = cv2.resize(tool_mask, (384, 384))

    # Concatenate with RGB
    rgb_normalized = normalize(frame)  # (384, 384, 3)
    input_5ch = np.concatenate([rgb_normalized, gb_mask[..., None], tool_mask[..., None]], axis=-1)
```

### First Layer Modification

```python
# Original SwinV2 patch embedding expects 3 channels
# We modify to accept 5 channels

# Option 1: Reinitialize first layer (loses pretrained weights for RGB)
model.patch_embed.proj = nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size)

# Option 2: Extend first layer (preserves RGB weights, initializes mask channels to zero)
with torch.no_grad():
    old_weight = model.patch_embed.proj.weight  # (embed_dim, 3, patch_h, patch_w)
    new_weight = torch.zeros(embed_dim, 5, patch_h, patch_w)
    new_weight[:, :3, :, :] = old_weight  # Copy RGB weights
    # Mask channels initialized to zero (model learns from scratch)
    model.patch_embed.proj = nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size)
    model.patch_embed.proj.weight = nn.Parameter(new_weight)
```

## Experiment Log

| Date | Status | Notes |
|------|--------|-------|
| 2026-01-23 | Directory created | Initial setup |
| TBD | Dataset ready | |
| TBD | Model modified | |
| TBD | Training started | |
| TBD | Evaluation complete | |
