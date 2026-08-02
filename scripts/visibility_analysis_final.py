"""
Thread 0: Visibility Analysis for C1 Classification
=====================================================

This script analyzes the visibility problem in C1 classification by stratifying
C1=0 frames into "structures visible" vs "structures not visible" categories.

Data Sources:
- Endoscapes-Seg50: 493 frames with GT pixel masks (semseg/)
- Synthetic masks: 1,933 frames from box-prompted SAM2 (synthetic_masks/)

Class IDs (0-indexed):
- 0: background
- 2: cystic_plate
- 3: calot_triangle
- 4: cystic_artery    <- C1 relevant
- 5: cystic_duct      <- C1 relevant
- 6: gallbladder
- 7: tool
- 255: ignore/void

CVS Labels:
- all_metadata.csv with columns: vid, frame, C1, C2, C3
- Join key: {vid}_{frame} matches mask filename (without .png)

Author: Sufian
Date: January 2026
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
    
    # Class IDs (0-indexed as per actual mask values)
    "cystic_artery_id": 4,
    "cystic_duct_id": 5,
    "ignore_id": 255,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cvs_labels(csv_path):
    """
    Load CVS labels from all_metadata.csv.
    
    Returns dict: {"{vid}_{frame}": {"C1": 0/1, "C2": 0/1, "C3": 0/1}}
    """
    print(f"Loading CVS labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['vid', 'frame', 'C1', 'C2', 'C3']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")
    
    # Build lookup dict
    labels = {}
    for _, row in df.iterrows():
        frame_id = f"{int(row['vid'])}_{int(row['frame'])}"
        labels[frame_id] = {
            "C1": int(row['C1']) if pd.notna(row['C1']) else None,
            "C2": int(row['C2']) if pd.notna(row['C2']) else None,
            "C3": int(row['C3']) if pd.notna(row['C3']) else None,
        }
    
    print(f"  Loaded {len(labels)} frame labels")
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

def check_c1_visibility(mask, config):
    """
    Check if C1 structures (cystic artery or cystic duct) are visible in mask.
    
    Returns dict with visibility info.
    """
    artery_id = config["cystic_artery_id"]
    duct_id = config["cystic_duct_id"]
    ignore_id = config["ignore_id"]
    
    # Count pixels (excluding ignore regions)
    valid_mask = mask[mask != ignore_id]
    total_pixels = valid_mask.size if valid_mask.size > 0 else mask.size
    
    artery_pixels = np.sum(mask == artery_id)
    duct_pixels = np.sum(mask == duct_id)
    
    artery_visible = artery_pixels > 0
    duct_visible = duct_pixels > 0
    c1_visible = artery_visible or duct_visible
    
    return {
        "artery_visible": artery_visible,
        "duct_visible": duct_visible,
        "c1_visible": c1_visible,
        "artery_pixels": int(artery_pixels),
        "duct_pixels": int(duct_pixels),
        "artery_percentage": (artery_pixels / total_pixels) * 100,
        "duct_percentage": (duct_pixels / total_pixels) * 100,
        "c1_percentage": ((artery_pixels + duct_pixels) / total_pixels) * 100,
    }


def analyze_masks(masks_dir, cvs_labels, config, source_name=""):
    """
    Analyze visibility distribution for all masks in a directory.
    """
    results = {
        "source": source_name,
        "total_frames": 0,
        "frames_with_cvs_labels": 0,
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
        visibility = check_c1_visibility(mask, config)
        
        # Get CVS label
        c1_label = None
        if frame_id in cvs_labels:
            c1_label = cvs_labels[frame_id].get("C1")
            if c1_label is not None:
                results["frames_with_cvs_labels"] += 1
        
        frame_data = {
            "frame_id": frame_id,
            "c1_label": c1_label,
            **visibility
        }
        results["all_frames"].append(frame_data)
        
        # Stratify
        if c1_label is None:
            results["c1_label_missing"].append(frame_data)
        elif c1_label == 1:
            results["c1_positive"].append(frame_data)
        else:  # c1_label == 0
            if visibility["c1_visible"]:
                results["c1_negative_visible"].append(frame_data)
            else:
                results["c1_negative_not_visible"].append(frame_data)
    
    return results


def analyze_synthetic_masks(synthetic_dirs, cvs_labels, config):
    """Analyze all synthetic mask directories and combine results."""
    combined = {
        "source": "Synthetic (Combined)",
        "total_frames": 0,
        "frames_with_cvs_labels": 0,
        "c1_positive": [],
        "c1_negative_visible": [],
        "c1_negative_not_visible": [],
        "c1_label_missing": [],
        "all_frames": [],
    }
    
    for rel_dir in synthetic_dirs:
        full_path = os.path.join(config["endoscapes_root"], rel_dir)
        split_name = rel_dir.split("/")[1]  # train, val, or test
        
        results = analyze_masks(full_path, cvs_labels, config, f"Synthetic-{split_name}")
        
        # Combine
        combined["total_frames"] += results["total_frames"]
        combined["frames_with_cvs_labels"] += results["frames_with_cvs_labels"]
        combined["c1_positive"].extend(results["c1_positive"])
        combined["c1_negative_visible"].extend(results["c1_negative_visible"])
        combined["c1_negative_not_visible"].extend(results["c1_negative_not_visible"])
        combined["c1_label_missing"].extend(results["c1_label_missing"])
        combined["all_frames"].extend(results["all_frames"])
    
    return combined


# =============================================================================
# REPORTING
# =============================================================================

def print_analysis_summary(results):
    """Print comprehensive summary statistics."""
    source = results.get("source", "Analysis")
    
    print(f"\n{'='*70}")
    print(f"{source} Summary")
    print(f"{'='*70}")
    
    total = results["total_frames"]
    with_labels = results["frames_with_cvs_labels"]
    
    print(f"Total frames with masks: {total}")
    print(f"Frames with CVS labels:  {with_labels}")
    print(f"Frames missing labels:   {len(results['c1_label_missing'])}")
    
    if with_labels == 0:
        print("\n[WARNING] No CVS labels found! Check that frame IDs match.")
        # Show some frame IDs for debugging
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
        
        print(f"\n{'-'*70}")
        print("INTERPRETATION")
        print(f"{'-'*70}")
        if c1_neg_total > 0:
            vis_ratio = c1_neg_vis / c1_neg_total
            if vis_ratio > 0.6:
                print("  [*] MAJORITY of C1=0 frames have VISIBLE structures")
                print("  --> Primary challenge: ASSESSMENT (visible but criteria not met)")
                print("  --> Recommended: Engrams, V-JEPA, ROI approaches")
            elif vis_ratio < 0.4:
                print("  [*] MAJORITY of C1=0 frames have NO VISIBLE structures")
                print("  --> Primary challenge: VISIBILITY RECOGNITION")
                print("  --> Recommended: Visibility Gate architecture (Thread 6)")
            else:
                print("  [!] MIXED: Roughly equal visible vs not-visible")
                print("  --> Both challenges are significant")
                print("  --> Recommended: Pursue both approaches")
    
    # Pixel coverage statistics
    if c1_neg_vis > 0:
        coverages = [f["c1_percentage"] for f in results["c1_negative_visible"]]
        print(f"\n{'-'*70}")
        print("PIXEL COVERAGE (when C1 structures visible)")
        print(f"{'-'*70}")
        print(f"  Mean coverage:   {np.mean(coverages):>6.2f}%")
        print(f"  Median coverage: {np.median(coverages):>6.2f}%")
        print(f"  Max coverage:    {np.max(coverages):>6.2f}%")
        print(f"  Min coverage:    {np.min(coverages):>6.2f}%")
    
    # Sanity check: C1=1 should have visible structures
    if c1_pos > 0:
        c1_pos_not_visible = sum(1 for f in results["c1_positive"] if not f["c1_visible"])
        if c1_pos_not_visible > 0:
            print(f"\n[ANOMALY] {c1_pos_not_visible} frames have C1=1 but NO visible structures!")
            print("   Possible causes: labeling errors, mask quality, or C1 assessed from video context")
            anomaly_frames = [f["frame_id"] for f in results["c1_positive"] if not f["c1_visible"]][:5]
            print(f"   Example frames: {anomaly_frames}")


def create_visualization(results, output_dir, filename_prefix=""):
    """Create visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    c1_neg_vis = len(results["c1_negative_visible"])
    c1_neg_not_vis = len(results["c1_negative_not_visible"])
    
    if c1_neg_vis + c1_neg_not_vis == 0:
        print("  No C1=0 frames to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Pie chart of C1=0 stratification
    ax1 = axes[0]
    labels = ['Visible\n(Assessment Challenge)', 'Not Visible\n(Visibility Challenge)']
    sizes = [c1_neg_vis, c1_neg_not_vis]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=90
    )
    ax1.set_title(f'C1=0 Stratification: {results["source"]}\n(The Key Insight)', 
                  fontsize=12, fontweight='bold')
    
    # Plot 2: Histogram of pixel coverage
    ax2 = axes[1]
    if results["c1_negative_visible"]:
        coverages = [f["c1_percentage"] for f in results["c1_negative_visible"]]
        ax2.hist(coverages, bins=30, color='#ff9999', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(coverages), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(coverages):.2f}%')
        ax2.axvline(np.median(coverages), color='darkred', linestyle=':', 
                    linewidth=2, label=f'Median: {np.median(coverages):.2f}%')
        ax2.set_xlabel('C1 Structure Coverage (% of image pixels)', fontsize=11)
        ax2.set_ylabel('Number of Frames', fontsize=11)
        ax2.set_title('When Visible: How Much of the Image?', fontsize=12, fontweight='bold')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No visible structures\nin C1=0 frames', 
                 ha='center', va='center', fontsize=14)
        ax2.set_title('Pixel Coverage', fontsize=12)
    
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
    
    # Summary CSV
    summary = {
        "source": results["source"],
        "total_frames": results["total_frames"],
        "frames_with_cvs_labels": results["frames_with_cvs_labels"],
        "c1_positive": len(results["c1_positive"]),
        "c1_negative_visible": len(results["c1_negative_visible"]),
        "c1_negative_not_visible": len(results["c1_negative_not_visible"]),
        "visibility_ratio": len(results["c1_negative_visible"]) / max(1, len(results["c1_negative_visible"]) + len(results["c1_negative_not_visible"])),
    }
    
    summary_filename = f'{filename_prefix}summary.csv' if filename_prefix else 'summary.csv'
    summary_path = os.path.join(output_dir, summary_filename)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Thread 0: C1 Visibility Analysis")
    parser.add_argument("--endoscapes_root", type=str, 
                        default=r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
                        help="Path to Endoscapes dataset root")
    parser.add_argument("--output_dir", type=str, default="visibility_analysis_results",
                        help="Output directory for results")
    parser.add_argument("--seg50_only", action="store_true",
                        help="Only analyze Seg50 (skip synthetic masks)")
    parser.add_argument("--synthetic_only", action="store_true",
                        help="Only analyze synthetic masks (skip Seg50)")
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["endoscapes_root"] = args.endoscapes_root
    config["output_dir"] = args.output_dir
    
    print("="*70)
    print("THREAD 0: C1 VISIBILITY ANALYSIS")
    print("="*70)
    print(f"\nEndoscapes root: {config['endoscapes_root']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Load CVS labels
    cvs_path = os.path.join(config["endoscapes_root"], config["cvs_labels_file"])
    cvs_labels = load_cvs_labels(cvs_path)
    
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
    
    # Final summary comparison
    if not args.seg50_only and not args.synthetic_only:
        print(f"\n{'='*70}")
        print("COMPARISON: Seg50 vs Synthetic")
        print(f"{'='*70}")
        
        seg50_vis_ratio = len(seg50_results["c1_negative_visible"]) / max(1, len(seg50_results["c1_negative_visible"]) + len(seg50_results["c1_negative_not_visible"]))
        synth_vis_ratio = len(synthetic_results["c1_negative_visible"]) / max(1, len(synthetic_results["c1_negative_visible"]) + len(synthetic_results["c1_negative_not_visible"]))
        
        print(f"  Seg50 visibility ratio:     {seg50_vis_ratio:.1%}")
        print(f"  Synthetic visibility ratio: {synth_vis_ratio:.1%}")
        
        if abs(seg50_vis_ratio - synth_vis_ratio) < 0.1:
            print("\n  [OK] Ratios are consistent - synthetic masks are reliable for this analysis")
        else:
            print("\n  [WARNING] Ratios differ - investigate mask quality differences")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()
