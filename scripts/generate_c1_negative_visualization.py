"""
Generate visualization of C1-Negative frames where BOTH cystic duct and cystic artery are visible.

This demonstrates the key finding from Thread 0 visibility analysis:
C1-negative frames often have BOTH structures visible in the segmentation mask,
proving that C1 is fundamentally an ASSESSMENT problem, not a detection problem.

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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "endoscapes_root": r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes",
    "seg50_masks_dir": "semseg",
    "cvs_labels_file": "all_metadata.csv",
    "output_path": r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\example_c1_negative_both_visible.png",

    # CORRECT Class IDs (from seg_label_map.txt)
    "cystic_artery_id": 3,   # Class 3
    "cystic_duct_id": 4,     # Class 4
    "gallbladder_id": 5,     # Class 5
    "tool_id": 6,            # Class 6

    # Visualization colors (RGB, 0-1)
    "artery_color": (1.0, 0.2, 0.2),    # Red for cystic artery
    "duct_color": (0.2, 0.4, 1.0),       # Blue for cystic duct
    "gb_color": (0.2, 0.8, 0.2),         # Green for gallbladder
    "tool_color": (0.8, 0.8, 0.2),       # Yellow for tool

    "overlay_alpha": 0.5,
    "num_examples": 6,
}


def find_image_path(frame_id, endoscapes_root):
    """Find the RGB image for a given frame_id (e.g., '100_27925')."""
    splits = ["train", "val", "test", "train_seg", "val_seg", "test_seg"]

    for split in splits:
        img_path = os.path.join(endoscapes_root, split, f"{frame_id}.jpg")
        if os.path.exists(img_path):
            return img_path

    # Also check 'all' directory if exists
    img_path = os.path.join(endoscapes_root, "all", f"{frame_id}.jpg")
    if os.path.exists(img_path):
        return img_path

    return None


def load_mask(mask_path):
    """Load segmentation mask as numpy array."""
    return np.array(Image.open(mask_path))


def load_image(image_path):
    """Load RGB image as numpy array."""
    return np.array(Image.open(image_path))


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


def check_visibility(mask, config):
    """Check visibility of C1 structures using CORRECT class IDs."""
    artery_id = config["cystic_artery_id"]  # Class 3
    duct_id = config["cystic_duct_id"]      # Class 4
    gb_id = config["gallbladder_id"]        # Class 5

    artery_pixels = int(np.sum(mask == artery_id))
    duct_pixels = int(np.sum(mask == duct_id))
    gb_pixels = int(np.sum(mask == gb_id))

    total_pixels = mask.size

    return {
        'artery_visible': artery_pixels > 0,
        'duct_visible': duct_pixels > 0,
        'gb_visible': gb_pixels > 0,
        'artery_pixels': artery_pixels,
        'duct_pixels': duct_pixels,
        'gb_pixels': gb_pixels,
        'artery_pct': (artery_pixels / total_pixels) * 100,
        'duct_pct': (duct_pixels / total_pixels) * 100,
        'gb_pct': (gb_pixels / total_pixels) * 100,
    }


def create_overlay(image, mask, config):
    """Create image with segmentation overlay for cystic structures.

    Uses CORRECT class IDs:
    - Class 3: Cystic Artery (RED)
    - Class 4: Cystic Duct (BLUE)
    - Class 5: Gallbladder (GREEN)
    - Class 6: Tool (YELLOW, subtle)
    """
    # Normalize image to 0-1 if needed
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    overlay = image.copy()

    artery_id = config["cystic_artery_id"]  # Class 3
    duct_id = config["cystic_duct_id"]      # Class 4
    gb_id = config["gallbladder_id"]        # Class 5
    tool_id = config["tool_id"]             # Class 6
    alpha = config["overlay_alpha"]

    # Apply gallbladder overlay (green, subtle)
    gb_mask = mask == gb_id
    if gb_mask.any():
        for c, color_val in enumerate(config["gb_color"]):
            overlay[:, :, c] = np.where(
                gb_mask,
                overlay[:, :, c] * (1 - alpha * 0.4) + color_val * alpha * 0.4,
                overlay[:, :, c]
            )

    # Apply tool overlay (yellow, very subtle - just for context)
    tool_mask = mask == tool_id
    if tool_mask.any():
        for c, color_val in enumerate(config["tool_color"]):
            overlay[:, :, c] = np.where(
                tool_mask,
                overlay[:, :, c] * (1 - alpha * 0.2) + color_val * alpha * 0.2,
                overlay[:, :, c]
            )

    # Apply cystic artery overlay (RED - prominent)
    artery_mask = mask == artery_id
    if artery_mask.any():
        for c, color_val in enumerate(config["artery_color"]):
            overlay[:, :, c] = np.where(
                artery_mask,
                overlay[:, :, c] * (1 - alpha) + color_val * alpha,
                overlay[:, :, c]
            )

    # Apply cystic duct overlay (BLUE - prominent)
    duct_mask = mask == duct_id
    if duct_mask.any():
        for c, color_val in enumerate(config["duct_color"]):
            overlay[:, :, c] = np.where(
                duct_mask,
                overlay[:, :, c] * (1 - alpha) + color_val * alpha,
                overlay[:, :, c]
            )

    return np.clip(overlay, 0, 1)


def main():
    print("=" * 70)
    print("Generating C1-Negative Visualization (CORRECTED CLASS IDs)")
    print("=" * 70)
    print("\nUsing CORRECT class IDs:")
    print(f"  Class {CONFIG['cystic_artery_id']}: Cystic Artery")
    print(f"  Class {CONFIG['cystic_duct_id']}: Cystic Duct")
    print(f"  Class {CONFIG['gallbladder_id']}: Gallbladder")
    print(f"  Class {CONFIG['tool_id']}: Tool")

    # Load CVS labels
    cvs_path = os.path.join(CONFIG['endoscapes_root'], CONFIG['cvs_labels_file'])
    print(f"\nLoading CVS labels from: {cvs_path}")
    cvs_labels = load_cvs_labels(cvs_path)
    print(f"Loaded labels for {len(cvs_labels)} frames")

    # Get all Seg50 masks
    masks_dir = os.path.join(CONFIG['endoscapes_root'], CONFIG['seg50_masks_dir'])
    mask_files = list(Path(masks_dir).glob("*.png"))
    print(f"Found {len(mask_files)} Seg50 masks")

    # Re-analyze with CORRECT class IDs
    print("\nRe-analyzing masks with correct class IDs...")
    c1_negative_both_visible = []
    c1_negative_total = 0

    for mask_file in mask_files:
        frame_id = mask_file.stem

        # Get CVS label
        if frame_id not in cvs_labels:
            continue

        c1_label = cvs_labels[frame_id]['C1']
        c1_raw = cvs_labels[frame_id]['C1_raw']

        # Only interested in C1=0 (negative)
        if c1_label != 0:
            continue

        c1_negative_total += 1

        # Load mask and check visibility with CORRECT class IDs
        mask = load_mask(str(mask_file))
        vis = check_visibility(mask, CONFIG)

        # Check if BOTH artery AND duct are visible
        if vis['artery_visible'] and vis['duct_visible']:
            img_path = find_image_path(frame_id, CONFIG['endoscapes_root'])
            if img_path:
                c1_negative_both_visible.append({
                    'frame_id': frame_id,
                    'image_path': img_path,
                    'mask_path': str(mask_file),
                    'c1_raw': c1_raw,
                    'artery_pct': vis['artery_pct'],
                    'duct_pct': vis['duct_pct'],
                    'combined_pct': vis['artery_pct'] + vis['duct_pct'],
                })

    print(f"\nC1-negative frames total: {c1_negative_total}")
    print(f"C1-negative with BOTH structures visible: {len(c1_negative_both_visible)}")
    if c1_negative_total > 0:
        pct = 100 * len(c1_negative_both_visible) / c1_negative_total
        print(f"Percentage: {pct:.1f}%")

    if len(c1_negative_both_visible) == 0:
        print("ERROR: No matching frames found!")
        return

    # Sort by combined visibility percentage
    c1_negative_both_visible.sort(key=lambda x: x['combined_pct'], reverse=True)

    # Select diverse examples
    n_examples = min(CONFIG['num_examples'], len(c1_negative_both_visible))
    selected_frames = []
    video_ids_used = set()

    for frame_data in c1_negative_both_visible:
        vid = frame_data['frame_id'].split('_')[0]

        if vid not in video_ids_used or len(selected_frames) < 2:
            selected_frames.append(frame_data)
            video_ids_used.add(vid)

            if len(selected_frames) >= n_examples:
                break

    print(f"\nSelected {len(selected_frames)} frames for visualization")

    # Create figure
    n_cols = 3
    n_rows = (len(selected_frames) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes for easy indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Add main title
    fig.suptitle(
        "C1-Negative Frames with Both Structures Visible\n"
        "(Cystic Duct + Cystic Artery present but C1 criterion NOT met)",
        fontsize=14, fontweight='bold', y=0.98
    )

    for idx, frame_data in enumerate(selected_frames):
        ax = axes_flat[idx]

        # Load image and mask
        image = load_image(frame_data['image_path'])
        mask = load_mask(frame_data['mask_path'])

        # Create overlay
        overlay = create_overlay(image, mask, CONFIG)

        # Display
        ax.imshow(overlay)
        ax.axis('off')

        # Add frame info
        c1_raw = frame_data['c1_raw']
        artery_pct = frame_data['artery_pct']
        duct_pct = frame_data['duct_pct']

        ax.set_title(
            f"Frame: {frame_data['frame_id']}\n"
            f"C1 = Negative (raw: {c1_raw:.2f})\n"
            f"Artery: {artery_pct:.1f}% | Duct: {duct_pct:.1f}%",
            fontsize=10, pad=5
        )

        # Add text box
        textstr = "Both structures\nVISIBLE"
        props = dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, fontweight='bold')

    # Hide unused axes
    for idx in range(len(selected_frames), len(axes_flat)):
        axes_flat[idx].axis('off')

    # Add legend with CORRECT class IDs
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Cystic Artery (Class 3)'),
        Patch(facecolor='blue', alpha=0.6, label='Cystic Duct (Class 4)'),
        Patch(facecolor='green', alpha=0.3, label='Gallbladder (Class 5)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02)
    )

    # Add explanatory text with computed percentage
    both_visible_pct = 100 * len(c1_negative_both_visible) / c1_negative_total if c1_negative_total > 0 else 0
    fig.text(
        0.5, 0.06,
        f"Key Finding: {both_visible_pct:.1f}% of C1-negative frames have BOTH structures visible.\n"
        "This proves C1 is an ASSESSMENT problem (evaluating clearance), not a DETECTION problem.",
        ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    # Save
    output_path = CONFIG['output_path']
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
