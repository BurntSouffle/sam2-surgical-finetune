"""
Single frame visualization: C1-Negative with both structures visible
Frame: 170_57050
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# Configuration
ENDOSCAPES_ROOT = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes"
FRAME_ID = "170_57050"
OUTPUT_PATH = r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\example_c1_negative_both_visible.png"

# CORRECT Class IDs
CYSTIC_ARTERY_ID = 3
CYSTIC_DUCT_ID = 4
GALLBLADDER_ID = 5
TOOL_ID = 6

# Colors
ARTERY_COLOR = (1.0, 0.2, 0.2)    # Red
DUCT_COLOR = (0.2, 0.4, 1.0)       # Blue
GB_COLOR = (0.2, 0.8, 0.2)         # Green
TOOL_COLOR = (0.9, 0.9, 0.2)       # Yellow


def find_image(frame_id):
    """Find image path."""
    for split in ["train", "val", "test", "train_seg", "val_seg", "test_seg"]:
        path = os.path.join(ENDOSCAPES_ROOT, split, f"{frame_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def create_overlay(image, mask, alpha=0.5):
    """Create overlay with correct class IDs."""
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0

    overlay = image.copy()

    # Gallbladder (green, subtle)
    gb_mask = mask == GALLBLADDER_ID
    if gb_mask.any():
        for c, val in enumerate(GB_COLOR):
            overlay[:, :, c] = np.where(gb_mask, overlay[:, :, c] * 0.7 + val * 0.3, overlay[:, :, c])

    # Tool (yellow, very subtle)
    tool_mask = mask == TOOL_ID
    if tool_mask.any():
        for c, val in enumerate(TOOL_COLOR):
            overlay[:, :, c] = np.where(tool_mask, overlay[:, :, c] * 0.85 + val * 0.15, overlay[:, :, c])

    # Cystic Artery (RED - prominent)
    artery_mask = mask == CYSTIC_ARTERY_ID
    if artery_mask.any():
        for c, val in enumerate(ARTERY_COLOR):
            overlay[:, :, c] = np.where(artery_mask, overlay[:, :, c] * (1-alpha) + val * alpha, overlay[:, :, c])

    # Cystic Duct (BLUE - prominent)
    duct_mask = mask == CYSTIC_DUCT_ID
    if duct_mask.any():
        for c, val in enumerate(DUCT_COLOR):
            overlay[:, :, c] = np.where(duct_mask, overlay[:, :, c] * (1-alpha) + val * alpha, overlay[:, :, c])

    return np.clip(overlay, 0, 1)


def main():
    print(f"Generating visualization for frame: {FRAME_ID}")

    # Load image and mask
    img_path = find_image(FRAME_ID)
    mask_path = os.path.join(ENDOSCAPES_ROOT, "semseg", f"{FRAME_ID}.png")

    if not img_path:
        print(f"ERROR: Image not found for {FRAME_ID}")
        return
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask not found at {mask_path}")
        return

    print(f"  Image: {img_path}")
    print(f"  Mask: {mask_path}")

    image = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))

    # Calculate pixel stats
    artery_pixels = np.sum(mask == CYSTIC_ARTERY_ID)
    duct_pixels = np.sum(mask == CYSTIC_DUCT_ID)
    gb_pixels = np.sum(mask == GALLBLADDER_ID)
    total = mask.size

    print(f"\n  Cystic Artery (class 3): {artery_pixels} pixels ({100*artery_pixels/total:.2f}%)")
    print(f"  Cystic Duct (class 4): {duct_pixels} pixels ({100*duct_pixels/total:.2f}%)")
    print(f"  Gallbladder (class 5): {gb_pixels} pixels ({100*gb_pixels/total:.2f}%)")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Frame", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Overlay
    overlay = create_overlay(image, mask)
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation Overlay", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Add info text box
    info_text = (
        f"Frame: {FRAME_ID}\n"
        f"C1 Label: NEGATIVE\n"
        f"Both structures VISIBLE\n\n"
        f"Artery: {100*artery_pixels/total:.1f}%\n"
        f"Duct: {100*duct_pixels/total:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props, family='monospace')

    # Main title
    fig.suptitle(
        "C1-Negative Frame with Both Structures Visible\n"
        "(Cystic Artery + Cystic Duct present, but C1 criterion NOT met)",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Cystic Artery (Class 3)'),
        Patch(facecolor='blue', alpha=0.6, label='Cystic Duct (Class 4)'),
        Patch(facecolor='green', alpha=0.4, label='Gallbladder (Class 5)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, 0.02))

    # Explanatory text
    fig.text(0.5, 0.08,
             "This frame demonstrates the ASSESSMENT problem: structures are clearly visible,\n"
             "but the hepatocystic triangle has not been adequately cleared.",
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
