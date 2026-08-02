"""
Thread 0: Visibility Analysis for C1 Classification (CORRECTED)
================================================================

This script analyzes the visibility problem in C1 classification by stratifying
C1=0 frames into "structures visible" vs "structures not visible" categories.

IMPORTANT: Endoscapes CVS labels are FRACTIONAL (0.00, 0.33, 0.67, 1.00)
representing annotator agreement. This script properly handles binarisation.

Data Sources:
- Endoscapes-Seg50: 493 frames with GT pixel masks (semseg/)
- Synthetic masks: 1,933 frames from box-prompted SAM2 (synthetic_masks/)

Class IDs (0-indexed, from seg_label_map.txt):
- 0: background
- 1: cystic_plate
- 2: calot_triangle
- 3: cystic_artery    <- C1 relevant
- 4: cystic_duct      <- C1 relevant
- 5: gallbladder
- 6: tool
- 255: ignore/void

CVS Labels:
- all_metadata.csv with columns: vid, frame, C1, C2, C3
- Labels are FRACTIONAL: 0.00, 0.33, 0.67, 1.00 (annotator agreement)
- Binarisation rules supported:
  * majority_vote: C1 >= 0.5 is positive (RECOMMENDED, matches SwinCVS paper)
  * lenient: C1 > 0 is positive
  * strict: C1 == 1.0 is positive (excludes ambiguous)

Author: Sufian
Date: January 2026 (Corrected)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Base path to Endoscapes dataset
    "endoscapes_root": r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
    
    # Relative paths within endoscapes
    "seg50_masks_dir": "semseg",
    "synthetic_masks_dirs": [
        "synthetic_masks/train/semantic",
        "synthetic_masks/val/semantic", 
        "synthetic_masks/test/semantic"
    ],
    "cvs_labels_file": "all_metadata.csv",
    
    # Output directory
    "output_dir": "visibility_analysis_results",
    
    # Class IDs (0-indexed, from seg_label_map.txt)
    "cystic_artery_id": 3,   # Class 3: cystic_artery
    "cystic_duct_id": 4,     # Class 4: cystic_duct
    "gallbladder_id": 5,     # Class 5: gallbladder
    "tool_id": 6,            # Class 6: tool
    "ignore_id": 255,
    
    # Binarisation rule: 'majority_vote', 'lenient', or 'strict'
    "binarisation_rule": "majority_vote",
}


# =============================================================================
# LABEL BINARISATION
# =============================================================================

def binarise_label(value, rule="majority_vote"):
    """
    Binarise a fractional CVS label to 0 or 1.
    
    Endoscapes labels are fractional (0.00, 0.33, 0.67, 1.00) representing
    annotator agreement across 3 annotators.
    
    Args:
        value: float, the raw label value
        rule: str, one of:
            - 'majority_vote': >= 0.5 is positive (RECOMMENDED)
            - 'lenient': > 0 is positive (any annotator said yes)
            - 'strict': == 1.0 is positive (unanimous only)
    
    Returns:
        int: 0 or 1, or None if value should be excluded (strict rule)
    """
    if pd.isna(value):
        return None
    
    value = float(value)
    
    if rule == "majority_vote":
        # >= 0.5 is positive (0.67 and 1.00)
        # < 0.5 is negative (0.00 and 0.33)
        return 1 if value >= 0.5 else 0
    
    elif rule == "lenient":
        # Any positive is positive (0.33, 0.67, 1.00)
        # Only 0.00 is negative
        return 1 if value > 0 else 0
    
    elif rule == "strict":
        # Only unanimous (1.00) is positive
        # Only unanimous negative (0.00) is negative
        # Ambiguous (0.33, 0.67) are excluded
        if value == 1.0:
            return 1
        elif value == 0.0:
            return 0
        else:
            return None  # Exclude ambiguous
    
    else:
        raise ValueError(f"Unknown binarisation rule: {rule}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cvs_labels(csv_path, binarisation_rule="majority_vote"):
    """
    Load CVS labels from all_metadata.csv.
    
    IMPORTANT: Labels are stored as FLOATS and binarised explicitly.
    
    Returns dict: {"{vid}_{frame}": {"C1": 0/1/None, "C2": 0/1/None, "C3": 0/1/None, 
                                      "C1_raw": float, "C2_raw": float, "C3_raw": float}}
    """
    print(f"Loading CVS labels from {csv_path}...")
    print(f"Binarisation rule: {binarisation_rule}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['vid', 'frame', 'C1', 'C2', 'C3']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")
    
    # Show raw label distribution
    print(f"\nRaw C1 value distribution:")
    print(df['C1'].value_counts().sort_index())
    
    # Build lookup dict with FLOAT labels and binarised labels
    labels = {}
    binarised_counts = {"positive": 0, "negative": 0, "excluded": 0}
    
    for _, row in df.iterrows():
        frame_id = f"{int(row['vid'])}_{int(row['frame'])}"
        
        # Store raw float values
        c1_raw = float(row['C1']) if pd.notna(row['C1']) else None
        c2_raw = float(row['C2']) if pd.notna(row['C2']) else None
        c3_raw = float(row['C3']) if pd.notna(row['C3']) else None
        
        # Binarise
        c1_bin = binarise_label(c1_raw, binarisation_rule)
        c2_bin = binarise_label(c2_raw, binarisation_rule)
        c3_bin = binarise_label(c3_raw, binarisation_rule)
        
        labels[frame_id] = {
            "C1": c1_bin,
            "C2": c2_bin,
            "C3": c3_bin,
            "C1_raw": c1_raw,
            "C2_raw": c2_raw,
            "C3_raw": c3_raw,
        }
        
        # Count binarisation outcomes
        if c1_bin == 1:
            binarised_counts["positive"] += 1
        elif c1_bin == 0:
            binarised_counts["negative"] += 1
        else:
            binarised_counts["excluded"] += 1
    
    print(f"\nAfter binarisation ({binarisation_rule}):")
    print(f"  C1 positive: {binarised_counts['positive']}")
    print(f"  C1 negative: {binarised_counts['negative']}")
    print(f"  C1 excluded: {binarised_counts['excluded']}")
    print(f"  Total: {len(labels)}")
    
    return labels


def load_mask(mask_path):
    """Load a segmentation mask as numpy array."""
    mask = np.array(Image.open(mask_path))
    return mask


def get_frame_id_from_path(mask_path):
    """Extract frame ID from mask path: '10_16600.png' -> '10_16600'"""
    return Path(mask_path).stem


# =============================================================================
# VISIBILITY ANALYSIS
# =============================================================================

def check_c1_visibility(mask, config, pixel_threshold=1):
    """
    Check if C1 structures (cystic artery or cystic duct) are visible in mask.
    
    Args:
        mask: numpy array, segmentation mask
        config: dict, configuration
        pixel_threshold: int, minimum pixels to count as "visible"
    
    Returns dict with visibility info.
    """
    artery_id = config["cystic_artery_id"]
    duct_id = config["cystic_duct_id"]
    gb_id = config["gallbladder_id"]
    ignore_id = config["ignore_id"]
    
    # Count pixels (IMPORTANT: count on full mask, ignore_id won't match class IDs)
    artery_pixels = int(np.sum(mask == artery_id))
    duct_pixels = int(np.sum(mask == duct_id))
    gb_pixels = int(np.sum(mask == gb_id))
    
    # Total pixels excluding ignore for percentage calculation
    total_pixels = int(np.sum(mask != ignore_id))
    if total_pixels == 0:
        total_pixels = mask.size  # Fallback
    
    # Visibility based on threshold
    artery_visible = artery_pixels >= pixel_threshold
    duct_visible = duct_pixels >= pixel_threshold
    gb_visible = gb_pixels >= pixel_threshold
    c1_visible = artery_visible or duct_visible
    
    return {
        "artery_visible": artery_visible,
        "duct_visible": duct_visible,
        "gb_visible": gb_visible,
        "c1_visible": c1_visible,
        "artery_pixels": artery_pixels,
        "duct_pixels": duct_pixels,
        "gb_pixels": gb_pixels,
        "total_pixels": total_pixels,
        "artery_percentage": (artery_pixels / total_pixels) * 100,
        "duct_percentage": (duct_pixels / total_pixels) * 100,
        "gb_percentage": (gb_pixels / total_pixels) * 100,
        "c1_percentage": ((artery_pixels + duct_pixels) / total_pixels) * 100,
    }


def analyze_masks(masks_dir, cvs_labels, config, source_name="", pixel_threshold=1):
    """
    Analyze visibility distribution for all masks in a directory.
    """
    results = {
        "source": source_name,
        "binarisation_rule": config["binarisation_rule"],
        "pixel_threshold": pixel_threshold,
        "total_frames": 0,
        "frames_with_cvs_labels": 0,
        "frames_excluded": 0,  # Due to strict binarisation
        "c1_positive": [],           # C1=1
        "c1_negative_visible": [],   # C1=0, structures visible
        "c1_negative_not_visible": [],  # C1=0, structures not visible
        "c1_label_missing": [],      # No CVS label available
        "all_frames": [],
    }
    
    masks_path = Path(masks_dir)
    if not masks_path.exists():
        print(f"  WARNING: Directory not found: {masks_dir}")
        return results
    
    mask_files = list(masks_path.glob("*.png"))
    print(f"  Found {len(mask_files)} masks in {masks_dir}")
    
    for mask_file in mask_files:
        frame_id = get_frame_id_from_path(mask_file)
        results["total_frames"] += 1
        
        # Load and analyze mask
        mask = load_mask(mask_file)
        visibility = check_c1_visibility(mask, config, pixel_threshold)
        
        # Get CVS label (already binarised)
        c1_label = None
        c1_raw = None
        if frame_id in cvs_labels:
            c1_label = cvs_labels[frame_id].get("C1")
            c1_raw = cvs_labels[frame_id].get("C1_raw")
            
            if c1_label is not None:
                results["frames_with_cvs_labels"] += 1
            else:
                results["frames_excluded"] += 1  # Excluded by strict rule
        
        frame_data = {
            "frame_id": frame_id,
            "c1_label": c1_label,
            "c1_raw": c1_raw,
            **visibility
        }
        results["all_frames"].append(frame_data)
        
        # Stratify (only if label is not None)
        if c1_label is None:
            results["c1_label_missing"].append(frame_data)
        elif c1_label == 1:
            results["c1_positive"].append(frame_data)
        elif c1_label == 0:
            if visibility["c1_visible"]:
                results["c1_negative_visible"].append(frame_data)
            else:
                results["c1_negative_not_visible"].append(frame_data)
    
    return results


def analyze_synthetic_masks(synthetic_dirs, cvs_labels, config):
    """Analyze all synthetic mask directories and combine results."""
    combined = {
        "source": "Synthetic (Combined)",
        "binarisation_rule": config["binarisation_rule"],
        "pixel_threshold": 1,
        "total_frames": 0,
        "frames_with_cvs_labels": 0,
        "frames_excluded": 0,
        "c1_positive": [],
        "c1_negative_visible": [],
        "c1_negative_not_visible": [],
        "c1_label_missing": [],
        "all_frames": [],
    }
    
    seen_frame_ids = set()  # Track for duplicate detection
    
    for rel_dir in synthetic_dirs:
        full_path = os.path.join(config["endoscapes_root"], rel_dir)
        split_name = rel_dir.split("/")[1]  # train, val, or test
        
        results = analyze_masks(full_path, cvs_labels, config, f"Synthetic-{split_name}")
        
        # Check for duplicates
        for frame in results["all_frames"]:
            if frame["frame_id"] in seen_frame_ids:
                print(f"  WARNING: Duplicate frame_id detected: {frame['frame_id']}")
            seen_frame_ids.add(frame["frame_id"])
        
        # Combine
        combined["total_frames"] += results["total_frames"]
        combined["frames_with_cvs_labels"] += results["frames_with_cvs_labels"]
        combined["frames_excluded"] += results["frames_excluded"]
        combined["c1_positive"].extend(results["c1_positive"])
        combined["c1_negative_visible"].extend(results["c1_negative_visible"])
        combined["c1_negative_not_visible"].extend(results["c1_negative_not_visible"])
        combined["c1_label_missing"].extend(results["c1_label_missing"])
        combined["all_frames"].extend(results["all_frames"])
    
    print(f"\n  Synthetic masks: {len(seen_frame_ids)} unique frame IDs")
    
    return combined


# =============================================================================
# REPORTING
# =============================================================================

def print_analysis_summary(results):
    """Print comprehensive summary statistics."""
    source = results.get("source", "Analysis")
    rule = results.get("binarisation_rule", "unknown")
    
    print(f"\n{'='*70}")
    print(f"{source} Summary (binarisation: {rule})")
    print(f"{'='*70}")
    
    total = results["total_frames"]
    with_labels = results["frames_with_cvs_labels"]
    excluded = results["frames_excluded"]
    
    print(f"Total frames with masks: {total}")
    print(f"Frames with binarised CVS labels: {with_labels}")
    print(f"Frames excluded (ambiguous): {excluded}")
    
    if with_labels == 0:
        print("\n[WARNING] No CVS labels found! Check that frame IDs match.")
        if results["all_frames"]:
            sample_ids = [f["frame_id"] for f in results["all_frames"][:5]]
            print(f"   Sample mask frame IDs: {sample_ids}")
        return
    
    c1_pos = len(results["c1_positive"])
    c1_neg_vis = len(results["c1_negative_visible"])
    c1_neg_not_vis = len(results["c1_negative_not_visible"])
    c1_neg_total = c1_neg_vis + c1_neg_not_vis
    
    print(f"\n{'-'*70}")
    print("C1 LABEL DISTRIBUTION")
    print(f"{'-'*70}")
    print(f"  C1 = 1 (positive):  {c1_pos:>5} ({100*c1_pos/with_labels:>5.1f}%)")
    print(f"  C1 = 0 (negative):  {c1_neg_total:>5} ({100*c1_neg_total/with_labels:>5.1f}%)")
    
    if c1_neg_total > 0:
        print(f"\n{'-'*70}")
        print("[KEY INSIGHT] C1=0 STRATIFICATION")
        print(f"{'-'*70}")
        print(f"  C1=0, structures VISIBLE:      {c1_neg_vis:>5} ({100*c1_neg_vis/c1_neg_total:>5.1f}% of C1=0)")
        print(f"  C1=0, structures NOT VISIBLE:  {c1_neg_not_vis:>5} ({100*c1_neg_not_vis/c1_neg_total:>5.1f}% of C1=0)")
        
        # Further breakdown: both vs one
        both_visible = sum(1 for f in results["c1_negative_visible"] 
                          if f["artery_visible"] and f["duct_visible"])
        only_duct = sum(1 for f in results["c1_negative_visible"]
                       if f["duct_visible"] and not f["artery_visible"])
        only_artery = sum(1 for f in results["c1_negative_visible"]
                         if f["artery_visible"] and not f["duct_visible"])
        
        print(f"\n  Breakdown of 'visible' C1=0 frames:")
        print(f"    BOTH visible:        {both_visible:>5} ({100*both_visible/c1_neg_total:>5.1f}% of C1=0)")
        print(f"    ONLY duct visible:   {only_duct:>5} ({100*only_duct/c1_neg_total:>5.1f}% of C1=0)")
        print(f"    ONLY artery visible: {only_artery:>5} ({100*only_artery/c1_neg_total:>5.1f}% of C1=0)")
        
        print(f"\n{'-'*70}")
        print("INTERPRETATION")
        print(f"{'-'*70}")
        vis_ratio = c1_neg_vis / c1_neg_total
        if vis_ratio > 0.6:
            print("  [*] MAJORITY of C1=0 frames have VISIBLE structures")
            print("  --> This suggests the primary challenge is ASSESSMENT")
            print("  --> Recommended: Engrams, V-JEPA, ROI approaches")
        elif vis_ratio < 0.4:
            print("  [*] MAJORITY of C1=0 frames have NO VISIBLE structures")
            print("  --> This suggests the primary challenge is VISIBILITY RECOGNITION")
            print("  --> Recommended: Visibility Gate architecture")
        else:
            print("  [!] MIXED: Roughly equal visible vs not-visible")
            print("  --> Both challenges are significant")
    
    # Sanity check: C1=1 should have visible structures
    if c1_pos > 0:
        c1_pos_not_visible = sum(1 for f in results["c1_positive"] if not f["c1_visible"])
        if c1_pos_not_visible > 0:
            print(f"\n[ANOMALY] {c1_pos_not_visible} frames have C1=1 but NO visible structures!")
            print("   Possible causes: labeling from video context, mask quality issues")


def create_visualization(results, output_dir, filename_prefix=""):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    c1_neg_vis = len(results["c1_negative_visible"])
    c1_neg_not_vis = len(results["c1_negative_not_visible"])
    
    if c1_neg_vis + c1_neg_not_vis == 0:
        print("  No C1=0 frames to visualize")
        return
    
    # Calculate detailed breakdown
    both_visible = sum(1 for f in results["c1_negative_visible"] 
                      if f["artery_visible"] and f["duct_visible"])
    only_duct = sum(1 for f in results["c1_negative_visible"]
                   if f["duct_visible"] and not f["artery_visible"])
    only_artery = sum(1 for f in results["c1_negative_visible"]
                     if f["artery_visible"] and not f["duct_visible"])
    neither = c1_neg_not_vis
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Detailed pie chart
    ax1 = axes[0]
    labels = ['Both Visible\n(Assessment)', 'Only Duct\n(Partial)', 
              'Only Artery\n(Rare)', 'Neither\n(Visibility)']
    sizes = [both_visible, only_duct, only_artery, neither]
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    explode = (0.05, 0.02, 0.02, 0.05)
    
    # Filter out zero values for cleaner plot
    non_zero = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
    if non_zero:
        labels, sizes, colors, explode = zip(*non_zero)
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, 
                autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title(f'C1=0 Stratification: {results["source"]}\n(Binarisation: {results["binarisation_rule"]})', 
                  fontsize=12, fontweight='bold')
    
    # Plot 2: Histogram of pixel coverage
    ax2 = axes[1]
    if results["c1_negative_visible"]:
        coverages = [f["c1_percentage"] for f in results["c1_negative_visible"]]
        ax2.hist(coverages, bins=30, color='#28a745', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(coverages), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(coverages):.2f}%')
        ax2.set_xlabel('C1 Structure Coverage (% of image pixels)', fontsize=11)
        ax2.set_ylabel('Number of Frames', fontsize=11)
        ax2.set_title('When Visible: Pixel Coverage Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    
    filename = f'{filename_prefix}c1_visibility_analysis.png' if filename_prefix else 'c1_visibility_analysis.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved visualization: {save_path}")


def save_results_to_csv(results, output_dir, filename_prefix=""):
    """Save detailed results to CSV for further analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results["all_frames"])
    
    filename = f'{filename_prefix}visibility_details.csv' if filename_prefix else 'visibility_details.csv'
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"  Saved details: {save_path}")
    
    # Detailed breakdown
    both_visible = sum(1 for f in results["c1_negative_visible"] 
                      if f["artery_visible"] and f["duct_visible"])
    only_duct = sum(1 for f in results["c1_negative_visible"]
                   if f["duct_visible"] and not f["artery_visible"])
    only_artery = sum(1 for f in results["c1_negative_visible"]
                     if f["artery_visible"] and not f["duct_visible"])
    
    c1_neg_total = len(results["c1_negative_visible"]) + len(results["c1_negative_not_visible"])
    
    summary = {
        "source": results["source"],
        "binarisation_rule": results["binarisation_rule"],
        "total_frames": results["total_frames"],
        "frames_with_cvs_labels": results["frames_with_cvs_labels"],
        "frames_excluded": results["frames_excluded"],
        "c1_positive": len(results["c1_positive"]),
        "c1_negative_total": c1_neg_total,
        "c1_negative_both_visible": both_visible,
        "c1_negative_only_duct": only_duct,
        "c1_negative_only_artery": only_artery,
        "c1_negative_neither": len(results["c1_negative_not_visible"]),
        "both_visible_pct": both_visible / c1_neg_total * 100 if c1_neg_total > 0 else 0,
        "any_visible_pct": len(results["c1_negative_visible"]) / c1_neg_total * 100 if c1_neg_total > 0 else 0,
    }
    
    summary_filename = f'{filename_prefix}summary.csv' if filename_prefix else 'summary.csv'
    summary_path = os.path.join(output_dir, summary_filename)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Thread 0: C1 Visibility Analysis (Corrected)")
    parser.add_argument("--endoscapes_root", type=str, 
                        default=r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
                        help="Path to Endoscapes dataset root")
    parser.add_argument("--output_dir", type=str, default="visibility_analysis_results",
                        help="Output directory for results")
    parser.add_argument("--binarisation", type=str, default="majority_vote",
                        choices=["majority_vote", "lenient", "strict"],
                        help="Label binarisation rule (default: majority_vote)")
    parser.add_argument("--seg50_only", action="store_true",
                        help="Only analyze Seg50 (skip synthetic masks)")
    parser.add_argument("--synthetic_only", action="store_true",
                        help="Only analyze synthetic masks (skip Seg50)")
    parser.add_argument("--all_rules", action="store_true",
                        help="Run analysis with all three binarisation rules for comparison")
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["endoscapes_root"] = args.endoscapes_root
    config["output_dir"] = args.output_dir
    config["binarisation_rule"] = args.binarisation
    
    print("="*70)
    print("THREAD 0: C1 VISIBILITY ANALYSIS (CORRECTED)")
    print("="*70)
    print(f"\nEndoscapes root: {config['endoscapes_root']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Binarisation rule: {config['binarisation_rule']}")
    
    if args.all_rules:
        # Run with all three rules for comparison
        rules = ["majority_vote", "lenient", "strict"]
        for rule in rules:
            print(f"\n\n{'#'*70}")
            print(f"# BINARISATION RULE: {rule.upper()}")
            print(f"{'#'*70}")
            
            config["binarisation_rule"] = rule
            cvs_path = os.path.join(config["endoscapes_root"], config["cvs_labels_file"])
            cvs_labels = load_cvs_labels(cvs_path, rule)
            
            if not args.synthetic_only:
                seg50_path = os.path.join(config["endoscapes_root"], config["seg50_masks_dir"])
                seg50_results = analyze_masks(seg50_path, cvs_labels, config, f"Seg50 ({rule})")
                print_analysis_summary(seg50_results)
        return
    
    # Load CVS labels with specified binarisation
    cvs_path = os.path.join(config["endoscapes_root"], config["cvs_labels_file"])
    cvs_labels = load_cvs_labels(cvs_path, config["binarisation_rule"])
    
    # Analyze Seg50 (Ground Truth)
    if not args.synthetic_only:
        print(f"\n{'='*70}")
        print("ANALYZING SEG50 (Ground Truth Masks)")
        print(f"{'='*70}")
        
        seg50_path = os.path.join(config["endoscapes_root"], config["seg50_masks_dir"])
        seg50_results = analyze_masks(seg50_path, cvs_labels, config, "Seg50 (GT)")
        
        print_analysis_summary(seg50_results)
        create_visualization(seg50_results, config["output_dir"], "seg50_")
        save_results_to_csv(seg50_results, config["output_dir"], "seg50_")
    
    # Analyze Synthetic Masks
    if not args.seg50_only:
        print(f"\n{'='*70}")
        print("ANALYZING SYNTHETIC MASKS (Box-Prompted SAM2)")
        print(f"{'='*70}")
        
        synthetic_results = analyze_synthetic_masks(
            config["synthetic_masks_dirs"], cvs_labels, config
        )
        
        print_analysis_summary(synthetic_results)
        create_visualization(synthetic_results, config["output_dir"], "synthetic_")
        save_results_to_csv(synthetic_results, config["output_dir"], "synthetic_")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()
