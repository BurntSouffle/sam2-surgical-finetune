"""
Step 3: Explore Endoscapes Dataset Structure
Analyzes directory structure, images, masks, and annotations.
READ-ONLY - does not modify any files.
"""

import os
import json
import random
from pathlib import Path
from collections import Counter


# Dataset path
DATASET_PATH = Path(r"C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes")


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def print_subheader(title: str) -> None:
    print(f"\n{'-'*50}")
    print(f" {title}")
    print('-'*50)


def explore_directory_structure():
    """List all top-level folders and file counts."""
    print_header("1. DIRECTORY STRUCTURE")

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset path does not exist: {DATASET_PATH}")
        return False

    print(f"Dataset root: {DATASET_PATH}")
    print(f"\nTop-level contents:")

    # Get all items in root
    items = list(DATASET_PATH.iterdir())
    dirs = [d for d in items if d.is_dir()]
    files = [f for f in items if f.is_file()]

    print(f"  Directories: {len(dirs)}")
    print(f"  Files: {len(files)}")

    # List directories with file counts
    print(f"\nDirectory breakdown:")
    for d in sorted(dirs):
        file_count = sum(1 for _ in d.rglob('*') if _.is_file())
        subdir_count = sum(1 for _ in d.iterdir() if _.is_dir())
        print(f"  {d.name}/")
        print(f"    Files: {file_count}, Subdirs: {subdir_count}")

    # List root files
    if files:
        print(f"\nRoot files:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")

    # Check specific directories of interest
    print_subheader("Key Directories Check")
    key_dirs = ['train_seg', 'val_seg', 'test_seg', 'semseg', 'train', 'val', 'test']
    for dirname in key_dirs:
        dirpath = DATASET_PATH / dirname
        if dirpath.exists():
            file_count = sum(1 for _ in dirpath.rglob('*') if _.is_file())
            print(f"  {dirname}/: EXISTS ({file_count} files)")
        else:
            print(f"  {dirname}/: NOT FOUND")

    return True


def analyze_images():
    """Analyze images from train_seg or similar directory."""
    print_header("2. IMAGE ANALYSIS")

    # Try different possible image directories
    possible_dirs = ['train_seg', 'train', 'images/train', 'images']
    image_dir = None

    for dirname in possible_dirs:
        dirpath = DATASET_PATH / dirname
        if dirpath.exists():
            image_dir = dirpath
            break

    if image_dir is None:
        print("ERROR: Could not find image directory")
        print(f"Searched: {possible_dirs}")
        return []

    print(f"Image directory: {image_dir}")

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    for ext in image_extensions:
        images.extend(image_dir.rglob(f'*{ext}'))
        images.extend(image_dir.rglob(f'*{ext.upper()}'))

    images = list(set(images))  # Remove duplicates
    print(f"Total images found: {len(images)}")

    if not images:
        print("No images found!")
        return []

    # Analyze filename patterns
    print_subheader("Filename Pattern Analysis")
    sample_names = [img.name for img in random.sample(images, min(10, len(images)))]
    print("Sample filenames:")
    for name in sample_names[:5]:
        print(f"  {name}")

    # Try to detect pattern
    first_name = images[0].stem
    if '_' in first_name:
        parts = first_name.split('_')
        print(f"\nDetected pattern: {len(parts)} parts separated by '_'")
        print(f"Example breakdown: {parts}")

    # Load and analyze sample images
    print_subheader("Sample Image Analysis")
    try:
        import cv2
        sample_images = random.sample(images, min(3, len(images)))
        for img_path in sample_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                print(f"  {img_path.name}: shape={img.shape}, dtype={img.dtype}")
            else:
                print(f"  {img_path.name}: FAILED TO LOAD")
    except ImportError:
        print("OpenCV not available, trying PIL...")
        try:
            from PIL import Image
            import numpy as np
            sample_images = random.sample(images, min(3, len(images)))
            for img_path in sample_images:
                img = Image.open(img_path)
                arr = np.array(img)
                print(f"  {img_path.name}: shape={arr.shape}, mode={img.mode}")
        except ImportError:
            print("Neither OpenCV nor PIL available for image loading")

    return images


def analyze_masks():
    """Analyze masks from semseg or similar directory."""
    print_header("3. MASK ANALYSIS")

    # Try different possible mask directories
    possible_dirs = ['semseg', 'masks', 'annotations', 'labels', 'semantic_masks']
    mask_dir = None

    for dirname in possible_dirs:
        dirpath = DATASET_PATH / dirname
        if dirpath.exists():
            mask_dir = dirpath
            break

    if mask_dir is None:
        print("ERROR: Could not find mask directory")
        print(f"Searched: {possible_dirs}")
        return []

    print(f"Mask directory: {mask_dir}")

    # Check if there are subdirectories
    subdirs = [d for d in mask_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"Subdirectories found: {[d.name for d in subdirs]}")

    # Find all mask files
    mask_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    masks = []
    for ext in mask_extensions:
        masks.extend(mask_dir.rglob(f'*{ext}'))
        masks.extend(mask_dir.rglob(f'*{ext.upper()}'))

    masks = list(set(masks))
    print(f"Total masks found: {len(masks)}")

    if not masks:
        print("No masks found!")
        return []

    # Analyze mask values
    print_subheader("Sample Mask Analysis")
    try:
        import cv2
        import numpy as np

        sample_masks = random.sample(masks, min(3, len(masks)))
        all_unique_values = set()

        for mask_path in sample_masks:
            # Try grayscale first
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"  {mask_path.name}: FAILED TO LOAD")
                continue

            unique_vals = np.unique(mask)
            all_unique_values.update(unique_vals.tolist())

            is_rgb = len(mask.shape) == 3
            mode = "RGB" if is_rgb else "Grayscale"

            print(f"  {mask_path.name}:")
            print(f"    Shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"    Mode: {mode}")
            print(f"    Unique values: {sorted(unique_vals)[:20]}{'...' if len(unique_vals) > 20 else ''}")
            print(f"    Value range: [{mask.min()}, {mask.max()}]")

        print_subheader("Class ID Verification")
        print(f"All unique values found across samples: {sorted(all_unique_values)}")

        expected_classes = set(range(7))  # 0-6
        if all_unique_values.issubset(expected_classes):
            print("Class IDs MATCH expected range (0-6)")
        else:
            extra = all_unique_values - expected_classes
            print(f"WARNING: Found values outside expected range: {extra}")

    except ImportError:
        print("OpenCV not available for mask analysis")
        try:
            from PIL import Image
            import numpy as np

            sample_masks = random.sample(masks, min(3, len(masks)))
            for mask_path in sample_masks:
                mask = Image.open(mask_path)
                arr = np.array(mask)
                unique_vals = np.unique(arr)
                print(f"  {mask_path.name}:")
                print(f"    Shape: {arr.shape}, mode: {mask.mode}")
                print(f"    Unique values: {sorted(unique_vals)[:20]}")
        except ImportError:
            print("Neither OpenCV nor PIL available")

    return masks


def verify_matches(images, masks):
    """Check overlap between images and masks."""
    print_header("4. MATCH VERIFICATION")

    if not images or not masks:
        print("Cannot verify matches - missing images or masks")
        return

    # Get basenames without extensions
    image_stems = {img.stem: img for img in images}
    mask_stems = {mask.stem: mask for mask in masks}

    print(f"Unique image names: {len(image_stems)}")
    print(f"Unique mask names: {len(mask_stems)}")

    # Find matches
    matched_stems = set(image_stems.keys()) & set(mask_stems.keys())
    images_without_masks = set(image_stems.keys()) - set(mask_stems.keys())
    masks_without_images = set(mask_stems.keys()) - set(image_stems.keys())

    print_subheader("Match Statistics")
    print(f"MATCHED pairs (image + mask): {len(matched_stems)}")
    print(f"Images WITHOUT masks: {len(images_without_masks)}")
    print(f"Masks WITHOUT images: {len(masks_without_images)}")

    # Coverage percentage
    if image_stems:
        coverage = len(matched_stems) / len(image_stems) * 100
        print(f"\nImage coverage: {coverage:.1f}% of images have masks")

    # Show example matches
    if matched_stems:
        print_subheader("Example Matched Pairs")
        sample_matches = list(matched_stems)[:5]
        for stem in sample_matches:
            img_path = image_stems[stem]
            mask_path = mask_stems[stem]
            print(f"  {img_path.name}")
            print(f"    -> {mask_path.relative_to(DATASET_PATH)}")

    # Show some unmatched examples
    if images_without_masks:
        print_subheader("Example Images WITHOUT Masks (first 5)")
        for stem in list(images_without_masks)[:5]:
            print(f"  {image_stems[stem].name}")

    if masks_without_images:
        print_subheader("Example Masks WITHOUT Images (first 5)")
        for stem in list(masks_without_images)[:5]:
            print(f"  {mask_stems[stem].name}")


def analyze_annotations():
    """Analyze JSON annotation files."""
    print_header("5. ANNOTATION FILES")

    # Find all JSON files
    json_files = list(DATASET_PATH.rglob('*.json'))

    print(f"JSON files found: {len(json_files)}")
    for jf in json_files:
        rel_path = jf.relative_to(DATASET_PATH)
        size_kb = jf.stat().st_size / 1024
        print(f"  {rel_path} ({size_kb:.1f} KB)")

    # Look for COCO-style annotations
    coco_files = [jf for jf in json_files if 'coco' in jf.name.lower() or 'annotation' in jf.name.lower()]

    if not coco_files:
        # Try any JSON file
        coco_files = json_files

    for json_path in coco_files[:3]:  # Analyze up to 3 files
        print_subheader(f"Analyzing: {json_path.name}")

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            print(f"  Top-level keys: {list(data.keys())}")

            # COCO format analysis
            if 'images' in data:
                print(f"  Number of images: {len(data['images'])}")
                if data['images']:
                    sample_img = data['images'][0]
                    print(f"  Sample image entry keys: {list(sample_img.keys())}")

            if 'annotations' in data:
                print(f"  Number of annotations: {len(data['annotations'])}")
                if data['annotations']:
                    sample_ann = data['annotations'][0]
                    print(f"  Sample annotation keys: {list(sample_ann.keys())}")

            if 'categories' in data:
                print(f"  Number of categories: {len(data['categories'])}")
                print(f"  Categories:")
                for cat in data['categories']:
                    cat_id = cat.get('id', '?')
                    cat_name = cat.get('name', '?')
                    print(f"    {cat_id}: {cat_name}")

            # Check for other common keys
            for key in ['info', 'licenses']:
                if key in data:
                    print(f"  {key}: {data[key]}")

        except json.JSONDecodeError as e:
            print(f"  ERROR: Invalid JSON - {e}")
        except Exception as e:
            print(f"  ERROR: {e}")


def summarize_dataset():
    """Print final summary."""
    print_header("DATASET SUMMARY")

    # Quick stats
    train_seg = DATASET_PATH / 'train_seg'
    val_seg = DATASET_PATH / 'val_seg'
    test_seg = DATASET_PATH / 'test_seg'
    semseg = DATASET_PATH / 'semseg'

    stats = {}
    for name, path in [('train_seg', train_seg), ('val_seg', val_seg),
                        ('test_seg', test_seg), ('semseg', semseg)]:
        if path.exists():
            count = sum(1 for _ in path.rglob('*') if _.is_file())
            stats[name] = count
        else:
            stats[name] = 'N/A'

    print("Quick Stats:")
    for name, count in stats.items():
        print(f"  {name}: {count} files")


def main():
    print("\n" + "="*70)
    print(" ENDOSCAPES DATASET EXPLORATION")
    print(" READ-ONLY Analysis")
    print("="*70)

    # Set random seed for reproducibility
    random.seed(42)

    # 1. Directory structure
    if not explore_directory_structure():
        return

    # 2. Image analysis
    images = analyze_images()

    # 3. Mask analysis
    masks = analyze_masks()

    # 4. Match verification
    verify_matches(images, masks)

    # 5. Annotation files
    analyze_annotations()

    # Final summary
    summarize_dataset()

    print("\n" + "="*70)
    print(" EXPLORATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
