"""
Full Visibility Analysis with CORRECT Class IDs
================================================

CORRECT Class IDs (from seg_label_map.txt):
- 0: background
- 1: cystic_plate
- 2: calot_triangle
- 3: cystic_artery  <- C1 relevant
- 4: cystic_duct    <- C1 relevant
- 5: gallbladder
- 6: tool

Author: Sufian
Date: January 2026
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# =============================================================================
# CONFIGURATION - CORRECT CLASS IDs
# =============================================================================

CONFIG = {
    "endoscapes_root": r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
    "seg50_masks_dir": "semseg",
    "cvs_labels_file": "all_metadata.csv",

    # CORRECT Class IDs
    "cystic_artery_id": 3,
    "cystic_duct_id": 4,
    "gallbladder_id": 5,
    "tool_id": 6,
}


def load_cvs_labels(csv_path):
    """Load CVS labels and binarize using majority vote (>=0.5 = positive)."""
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        frame_id = f"{int(row['vid'])}_{int(row['frame'])}"
        c1_raw = float(row['C1']) if pd.notna(row['C1']) else None
        c1_bin = 1 if c1_raw is not None and c1_raw >= 0.5 else 0
        labels[frame_id] = {
            'C1': c1_bin,
            'C1_raw': c1_raw,
        }
    return labels


def analyze_mask(mask_path, config):
    """Analyze a single mask for structure visibility."""
    mask = np.array(Image.open(mask_path))

    artery_id = config["cystic_artery_id"]  # Class 3
    duct_id = config["cystic_duct_id"]      # Class 4
    gb_id = config["gallbladder_id"]        # Class 5

    artery_pixels = int(np.sum(mask == artery_id))
    duct_pixels = int(np.sum(mask == duct_id))
    gb_pixels = int(np.sum(mask == gb_id))

    return {
        'artery_visible': artery_pixels > 0,
        'duct_visible': duct_pixels > 0,
        'gb_visible': gb_pixels > 0,
        'artery_pixels': artery_pixels,
        'duct_pixels': duct_pixels,
        'gb_pixels': gb_pixels,
    }


def main():
    print("=" * 70)
    print("FULL VISIBILITY ANALYSIS - CORRECT CLASS IDs")
    print("=" * 70)
    print("\nClass mapping (from seg_label_map.txt):")
    print("  Class 3: Cystic Artery")
    print("  Class 4: Cystic Duct")
    print("  Class 5: Gallbladder")
    print("  Class 6: Tool")

    # Load CVS labels
    cvs_path = os.path.join(CONFIG['endoscapes_root'], CONFIG['cvs_labels_file'])
    print(f"\nLoading CVS labels from: {cvs_path}")
    cvs_labels = load_cvs_labels(cvs_path)

    # Get all Seg50 masks
    masks_dir = os.path.join(CONFIG['endoscapes_root'], CONFIG['seg50_masks_dir'])
    mask_files = list(Path(masks_dir).glob("*.png"))
    print(f"Found {len(mask_files)} Seg50 masks")

    # Analyze all masks
    print("\nAnalyzing all masks...")

    # Storage for analysis
    all_frames = []

    for mask_file in mask_files:
        frame_id = mask_file.stem

        if frame_id not in cvs_labels:
            continue

        c1_label = cvs_labels[frame_id]['C1']
        c1_raw = cvs_labels[frame_id]['C1_raw']

        vis = analyze_mask(str(mask_file), CONFIG)

        all_frames.append({
            'frame_id': frame_id,
            'c1_label': c1_label,
            'c1_raw': c1_raw,
            **vis
        })

    df = pd.DataFrame(all_frames)
    print(f"Analyzed {len(df)} frames with CVS labels")

    # ==========================================================================
    # 1. C1-NEGATIVE BREAKDOWN
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. C1-NEGATIVE FRAMES BREAKDOWN")
    print("=" * 70)

    c1_neg = df[df['c1_label'] == 0]
    n_c1_neg = len(c1_neg)
    print(f"\nTotal C1-negative frames: {n_c1_neg}")

    # Categories
    both_visible = c1_neg[(c1_neg['artery_visible']) & (c1_neg['duct_visible'])]
    only_duct = c1_neg[(~c1_neg['artery_visible']) & (c1_neg['duct_visible'])]
    only_artery = c1_neg[(c1_neg['artery_visible']) & (~c1_neg['duct_visible'])]
    neither = c1_neg[(~c1_neg['artery_visible']) & (~c1_neg['duct_visible'])]

    n_both = len(both_visible)
    n_only_duct = len(only_duct)
    n_only_artery = len(only_artery)
    n_neither = len(neither)

    print(f"\n{'Category':<35} {'Count':>8} {'Percentage':>12}")
    print("-" * 55)
    print(f"{'Both artery AND duct visible':<35} {n_both:>8} {100*n_both/n_c1_neg:>11.1f}%")
    print(f"{'Only duct visible (no artery)':<35} {n_only_duct:>8} {100*n_only_duct/n_c1_neg:>11.1f}%")
    print(f"{'Only artery visible (no duct)':<35} {n_only_artery:>8} {100*n_only_artery/n_c1_neg:>11.1f}%")
    print(f"{'Neither visible':<35} {n_neither:>8} {100*n_neither/n_c1_neg:>11.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<35} {n_c1_neg:>8} {'100.0%':>12}")

    # Additional: Any C1 structure visible
    any_visible = c1_neg[(c1_neg['artery_visible']) | (c1_neg['duct_visible'])]
    print(f"\nAny C1 structure visible (artery OR duct): {len(any_visible)} ({100*len(any_visible)/n_c1_neg:.1f}%)")

    # ==========================================================================
    # 2. DETECTION-ONLY BASELINE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. DETECTION-ONLY BASELINE PERFORMANCE")
    print("=" * 70)
    print("\nRule: Predict C1=1 iff both artery AND duct have >0 pixels")

    # Create predictions
    df['pred_c1'] = ((df['artery_visible']) & (df['duct_visible'])).astype(int)

    y_true = df['c1_label'].values
    y_pred = df['pred_c1'].values

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'Precision':<20} {precision:>10.4f}")
    print(f"{'Recall':<20} {recall:>10.4f}")
    print(f"{'F1 Score':<20} {f1:>10.4f}")
    print(f"{'Accuracy':<20} {accuracy:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Neg    Pos")
    print(f"  Actual Neg    {cm[0,0]:>5}  {cm[0,1]:>5}")
    print(f"  Actual Pos    {cm[1,0]:>5}  {cm[1,1]:>5}")

    # Breakdown of errors
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Negatives (TN):  {tn:>5} - Correctly predicted C1=0")
    print(f"  False Positives (FP): {fp:>5} - Predicted C1=1, actually C1=0 (both visible but negative)")
    print(f"  False Negatives (FN): {fn:>5} - Predicted C1=0, actually C1=1 (C1+ but not both visible)")
    print(f"  True Positives (TP):  {tp:>5} - Correctly predicted C1=1")

    # ==========================================================================
    # 3. C1-POSITIVE FRAMES ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. C1-POSITIVE FRAMES VISIBILITY")
    print("=" * 70)

    c1_pos = df[df['c1_label'] == 1]
    n_c1_pos = len(c1_pos)
    print(f"\nTotal C1-positive frames: {n_c1_pos}")

    # Categories for C1-positive
    pos_both_visible = c1_pos[(c1_pos['artery_visible']) & (c1_pos['duct_visible'])]
    pos_only_duct = c1_pos[(~c1_pos['artery_visible']) & (c1_pos['duct_visible'])]
    pos_only_artery = c1_pos[(c1_pos['artery_visible']) & (~c1_pos['duct_visible'])]
    pos_neither = c1_pos[(~c1_pos['artery_visible']) & (~c1_pos['duct_visible'])]

    n_pos_both = len(pos_both_visible)
    n_pos_only_duct = len(pos_only_duct)
    n_pos_only_artery = len(pos_only_artery)
    n_pos_neither = len(pos_neither)

    print(f"\n{'Category':<35} {'Count':>8} {'Percentage':>12}")
    print("-" * 55)
    print(f"{'Both artery AND duct visible':<35} {n_pos_both:>8} {100*n_pos_both/n_c1_pos:>11.1f}%")
    print(f"{'Only duct visible (no artery)':<35} {n_pos_only_duct:>8} {100*n_pos_only_duct/n_c1_pos:>11.1f}%")
    print(f"{'Only artery visible (no duct)':<35} {n_pos_only_artery:>8} {100*n_pos_only_artery/n_c1_pos:>11.1f}%")
    print(f"{'Neither visible':<35} {n_pos_neither:>8} {100*n_pos_neither/n_c1_pos:>11.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<35} {n_c1_pos:>8} {'100.0%':>12}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print(f"\n1. C1-NEGATIVE frames where BOTH structures visible: {n_both}/{n_c1_neg} ({100*n_both/n_c1_neg:.1f}%)")
    print(f"   -> These are 'assessment failures': structures present but criterion not met")

    print(f"\n2. C1-POSITIVE frames where BOTH structures visible: {n_pos_both}/{n_c1_pos} ({100*n_pos_both/n_c1_pos:.1f}%)")
    if n_pos_both < n_c1_pos:
        missing = n_c1_pos - n_pos_both
        print(f"   -> WARNING: {missing} C1-positive frames DON'T have both structures visible!")
        print(f"   -> This suggests C1 can be positive even without both structures in view")
        print(f"   -> Possible reasons: temporal context, annotator judgment, partial visibility")

    print(f"\n3. Detection-only baseline F1: {f1:.4f}")
    print(f"   -> This is the maximum achievable by pure visibility detection")
    print(f"   -> Any model scoring higher must be doing some 'assessment'")

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
    VISIBILITY ANALYSIS (Seg50, n={len(df)})
    ========================================

    C1-Negative (n={n_c1_neg}):
      - Both visible:    {n_both:>4} ({100*n_both/n_c1_neg:>5.1f}%)  <- Assessment problem
      - Only duct:       {n_only_duct:>4} ({100*n_only_duct/n_c1_neg:>5.1f}%)
      - Only artery:     {n_only_artery:>4} ({100*n_only_artery/n_c1_neg:>5.1f}%)
      - Neither:         {n_neither:>4} ({100*n_neither/n_c1_neg:>5.1f}%)  <- Visibility problem

    C1-Positive (n={n_c1_pos}):
      - Both visible:    {n_pos_both:>4} ({100*n_pos_both/n_c1_pos:>5.1f}%)
      - Only duct:       {n_pos_only_duct:>4} ({100*n_pos_only_duct/n_c1_pos:>5.1f}%)
      - Only artery:     {n_pos_only_artery:>4} ({100*n_pos_only_artery/n_c1_pos:>5.1f}%)
      - Neither:         {n_pos_neither:>4} ({100*n_pos_neither/n_c1_pos:>5.1f}%)

    Detection Baseline (predict C1=1 iff both visible):
      - Precision: {precision:.4f}
      - Recall:    {recall:.4f}
      - F1 Score:  {f1:.4f}
      - Accuracy:  {accuracy:.4f}
    """)


if __name__ == "__main__":
    main()
