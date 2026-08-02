# E1 Visual Engrams - Diagnostic Report
**Date:** 2026-01-24

## Summary

| Check | Status | Notes |
|-------|--------|-------|
| Pretrained weights loaded | **YES** | Correlation >0.998 confirms ImageNet weights were used |
| Engram masks correct | **YES** | Masks generated correctly, 16% GB / 16% Tool coverage |
| Training pipeline works | **YES** | Loss decreased from 17759 to 8010 |
| Data shapes correct | **YES** | [B, 5, 5, 384, 384] with 5 channels |
| No NaN/Inf | **YES** | Clean training |

## Critical Finding: Pretrained Weights WERE Loaded

**Contrary to initial belief, the ImageNet pretrained weights WERE correctly loaded.**

Evidence:
- Layer weight correlation between trained E1 checkpoint and pretrained weights:
  - `norm1.weight`: 0.999999
  - `attn.qkv.weight`: 0.999030
  - `patch_embed (RGB)`: 0.998459

**Conclusion:** The 0.45 mAP (validation) / 0.36 mAP (test) IS the actual E1 performance WITH pretrained weights.

## Bug Found: RGB Normalization

**Location:** `f_dataset_engram.py:419`

```python
# This line OVERWRITES ImageNet normalization!
image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image) + 1e-8)
```

**Impact:**
- RGB values are [0, 1] instead of ImageNet-normalized [-2.5, 2.5]
- SwinV2 expects ImageNet normalization
- Model may not fully utilize pretrained features

**Note:** E1 was trained WITH this bug, so current weights are adapted to it. Fixing this would require retraining.

## Training Summary

| Metric | Value |
|--------|-------|
| Total epochs | 16 |
| Best val mAP | 0.4525 (Epoch 10) |
| Final val mAP | 0.4076 |
| Test mAP | 0.3628 |
| Train loss (start → end) | 17759 → 8010 |
| Val loss (start → end) | 5801 → 6231 |

## Data Pipeline Verification

```
Batch shape: [1, 5, 5, 384, 384]  ✓
  - Batch size: 1
  - Sequence length: 5 frames
  - Channels: 5 (RGB + GB + Tool)
  - Resolution: 384x384

RGB range: [0.000, 1.000]  ⚠️ (Should be ImageNet normalized)
GB mask range: [-1.000, 1.000]  ✓
Tool mask range: [-1.000, 1.000]  ✓
Labels: [B, 3] binary  ✓
```

## Mask Quality

From visualization of 8 random training samples:
- GB masks correctly highlight gallbladder regions (magenta)
- Tool masks correctly highlight surgical instruments (cyan)
- Average coverage: ~16% for both mask types
- Masks are binary with clean boundaries

## Generated Files

1. `diagnostics/e1_masks_visualization.png` - 8 samples showing RGB + mask overlays
2. `diagnostics/e1_training_curves.png` - Loss and mAP progression
3. `diagnostics/E1_DIAGNOSTIC_REPORT.md` - This report

## Monday Checklist

### If retraining (recommended to fix normalization bug):

1. [ ] Fix `f_dataset_engram.py:419` - Remove the [0,1] re-normalization
2. [ ] Verify RGB values are ImageNet normalized after fix
3. [ ] Run baseline SwinCVS (3-channel) first → expect ~67% mAP
4. [ ] Run E1 (5-channel) with fixed normalization
5. [ ] Compare E1 vs baseline

### To fix normalization:

```python
# In f_dataset_engram.py, line 419
# REMOVE this line:
# image = (image - torch.min(image)) / (-torch.min(image) + torch.max(image) + 1e-8)

# The transforms.Compose already includes Normalize() which is sufficient
```

### Alternative: Use current model as-is

If 36% test mAP is acceptable for comparison purposes:
1. [ ] Use current E1 checkpoint
2. [ ] Run baseline with same [0,1] normalization for fair comparison
3. [ ] Report results as "E1 vs baseline (both with suboptimal normalization)"

## Interpretation

The E1 experiment achieved:
- **Validation mAP: 0.4525** (best at epoch 10)
- **Test mAP: 0.3628**

This ~9% generalization gap is notable. Possible causes:
1. The normalization bug may hurt generalization
2. The model may be overfitting to training mask patterns
3. Test distribution may differ from validation

## Recommendation

**Retrain E1 with the normalization fix before concluding whether visual engrams help or not.** The current results may be artificially limited by the normalization bug.
