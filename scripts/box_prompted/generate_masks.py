"""
Generate Synthetic Masks for Endoscapes Dataset using Fine-tuned SAM2-Large

This script uses the fine-tuned SAM2 model to generate high-quality segmentation
masks from bounding box annotations in COCO format.

Usage:
    python scripts/box_prompted/generate_masks.py --split train
    python scripts/box_prompted/generate_masks.py --split all
    python scripts/box_prompted/generate_masks.py --split train --resume

Output Structure:
    endoscapes/synthetic_masks/
    ├── train/
    │   ├── binary/           # Individual binary masks per object
    │   └── semantic/         # Combined semantic masks
    ├── val/
    └── test/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

# SAM2 imports
from sam2.build_sam import build_sam2
from train_box import SAM2BoxTrainer, CONFIG as TRAIN_CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Model
    'checkpoint': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\box_prompted\run_20260119_211504\best_model.pt'),
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    'sam2_checkpoint': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\checkpoints\sam2.1_hiera_large.pt'),

    # Data paths
    'data_root': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'),
    'output_root': Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes\synthetic_masks'),

    # Processing
    'image_size': (1024, 1024),
    'batch_size': 1,  # Process one image at a time for simplicity
}

# Category mapping from COCO annotations
CATEGORIES = {
    1: 'cystic_plate',
    2: 'calot_triangle',
    3: 'cystic_artery',
    4: 'cystic_duct',
    5: 'gallbladder',
    6: 'tool',
}

CATEGORY_NAMES_TO_IDS = {v: k for k, v in CATEGORIES.items()}


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(device: torch.device) -> SAM2BoxTrainer:
    """Load fine-tuned SAM2 model."""
    print("\nLoading fine-tuned SAM2-Large model...")

    # Load base SAM2 model
    sam2_model = build_sam2(
        config_file=CONFIG['model_cfg'],
        ckpt_path=str(CONFIG['sam2_checkpoint']),
        device='cpu',
        mode='eval',
    )

    # Create trainer wrapper
    model = SAM2BoxTrainer(
        sam2_model=sam2_model,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        image_size=CONFIG['image_size'],
    )

    # Load fine-tuned mask decoder weights
    checkpoint = torch.load(str(CONFIG['checkpoint']), map_location='cpu', weights_only=False)
    model.model.sam_mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


# =============================================================================
# DATA LOADING
# =============================================================================
def load_coco_annotations(split: str) -> Tuple[Dict, Dict, List]:
    """
    Load COCO format annotations.

    Returns:
        images_dict: {image_id: image_info}
        annotations_by_image: {image_id: [annotations]}
        categories: list of category dicts
    """
    ann_path = CONFIG['data_root'] / split / 'annotation_coco.json'

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    print(f"Loading annotations from: {ann_path}")

    with open(ann_path) as f:
        data = json.load(f)

    # Create lookup dicts
    images_dict = {img['id']: img for img in data['images']}

    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    print(f"  Images: {len(images_dict)}")
    print(f"  Annotations: {len(data['annotations'])}")

    return images_dict, dict(annotations_by_image), data['categories']


def coco_bbox_to_xyxy(bbox: List[float]) -> np.ndarray:
    """
    Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2] format.
    """
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess image for SAM2 inference.

    Args:
        image: (H, W, 3) RGB numpy array

    Returns:
        Tensor (1, 3, 1024, 1024) normalized
    """
    target_size = CONFIG['image_size']

    # Resize
    img_resized = cv2.resize(image, (target_size[1], target_size[0]))

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)  # Add batch dimension


def scale_bbox(bbox: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
    """
    Scale bounding box from original image size to model input size (1024x1024).

    Args:
        bbox: [x1, y1, x2, y2] in original image coordinates
        orig_size: (height, width) of original image

    Returns:
        Scaled bbox for 1024x1024 input
    """
    orig_h, orig_w = orig_size
    target_h, target_w = CONFIG['image_size']

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled = bbox.copy()
    scaled[0] *= scale_x  # x1
    scaled[1] *= scale_y  # y1
    scaled[2] *= scale_x  # x2
    scaled[3] *= scale_y  # y2

    return scaled


# =============================================================================
# MASK GENERATION
# =============================================================================
@torch.no_grad()
def generate_mask(
    model: SAM2BoxTrainer,
    image_tensor: torch.Tensor,
    bbox: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Generate binary mask for a single bounding box.

    Args:
        model: Fine-tuned SAM2 model
        image_tensor: (1, 3, H, W) preprocessed image
        bbox: (1, 4) bounding box [x1, y1, x2, y2]
        device: torch device

    Returns:
        Binary mask (H, W) as numpy array with values 0 or 1
    """
    image_tensor = image_tensor.to(device)
    bbox = bbox.to(device)

    # Forward pass
    outputs = model(image_tensor, bbox)
    pred_mask = outputs['masks']  # (1, 1, H, W)

    # Apply sigmoid and threshold
    mask = torch.sigmoid(pred_mask[0, 0]).cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)

    return binary_mask


def resize_mask_to_original(mask: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask from model output size back to original image size.

    Args:
        mask: Binary mask at model resolution
        orig_size: (height, width) of original image

    Returns:
        Resized binary mask
    """
    return cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================
def process_split(
    model: SAM2BoxTrainer,
    split: str,
    device: torch.device,
    resume: bool = False,
) -> Dict:
    """
    Process all images in a split and generate masks.

    Args:
        model: Fine-tuned SAM2 model
        split: 'train', 'val', or 'test'
        device: torch device
        resume: Skip already processed images

    Returns:
        Statistics dict
    """
    print(f"\n{'='*60}")
    print(f" Processing {split.upper()} split")
    print(f"{'='*60}")

    # Load annotations
    images_dict, annotations_by_image, categories = load_coco_annotations(split)

    # Create output directories
    binary_dir = CONFIG['output_root'] / split / 'binary'
    semantic_dir = CONFIG['output_root'] / split / 'semantic'
    binary_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_images': len(images_dict),
        'processed_images': 0,
        'skipped_images': 0,
        'total_masks': 0,
        'masks_per_class': defaultdict(int),
        'failed_images': [],
        'inference_times': [],
    }

    # Process each image
    image_ids = list(images_dict.keys())

    pbar = tqdm(image_ids, desc=f"Generating masks [{split}]")

    for image_id in pbar:
        img_info = images_dict[image_id]
        file_name = img_info['file_name']
        img_stem = Path(file_name).stem

        # Check if already processed (for resume)
        semantic_path = semantic_dir / f"{img_stem}.png"
        if resume and semantic_path.exists():
            stats['skipped_images'] += 1
            continue

        # Load image
        img_path = CONFIG['data_root'] / split / file_name
        if not img_path.exists():
            stats['failed_images'].append(str(img_path))
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            # Try numpy decode for Windows paths
            image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            stats['failed_images'].append(str(img_path))
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = (image.shape[0], image.shape[1])

        # Preprocess image
        image_tensor = preprocess_image(image)

        # Get annotations for this image
        annotations = annotations_by_image.get(image_id, [])

        if not annotations:
            # No annotations - save empty semantic mask
            semantic_mask = np.zeros(orig_size, dtype=np.uint8)
            cv2.imwrite(str(semantic_path), semantic_mask)
            stats['processed_images'] += 1
            continue

        # Initialize semantic mask (background = 0)
        semantic_mask = np.zeros(orig_size, dtype=np.uint8)

        # Process each annotation
        start_time = time.time()

        for ann_idx, ann in enumerate(annotations):
            category_id = ann['category_id']
            category_name = CATEGORIES.get(category_id, f'class_{category_id}')

            # Convert bbox from COCO format
            bbox_orig = coco_bbox_to_xyxy(ann['bbox'])

            # Scale bbox to model input size
            bbox_scaled = scale_bbox(bbox_orig, orig_size)
            bbox_tensor = torch.from_numpy(bbox_scaled).float().unsqueeze(0)

            # Generate mask
            try:
                mask = generate_mask(model, image_tensor, bbox_tensor, device)

                # Resize to original size
                mask_orig = resize_mask_to_original(mask, orig_size)

                # Save binary mask
                binary_path = binary_dir / f"{img_stem}_{category_name}_{ann_idx}.png"
                cv2.imwrite(str(binary_path), mask_orig * 255)

                # Add to semantic mask (later classes overwrite earlier ones)
                semantic_mask[mask_orig > 0] = category_id

                stats['total_masks'] += 1
                stats['masks_per_class'][category_name] += 1

            except Exception as e:
                tqdm.write(f"Error processing {file_name} ann {ann_idx}: {e}")

        inference_time = time.time() - start_time
        stats['inference_times'].append(inference_time)

        # Save semantic mask
        cv2.imwrite(str(semantic_path), semantic_mask)

        stats['processed_images'] += 1

        # Update progress bar
        pbar.set_postfix({
            'masks': stats['total_masks'],
            'time': f"{np.mean(stats['inference_times'][-10:]):.2f}s"
        })

    return stats


def print_statistics(stats: Dict, split: str):
    """Print generation statistics."""
    print(f"\n{'='*60}")
    print(f" Statistics for {split.upper()}")
    print(f"{'='*60}")

    print(f"\nImages:")
    print(f"  Total:     {stats['total_images']}")
    print(f"  Processed: {stats['processed_images']}")
    print(f"  Skipped:   {stats['skipped_images']}")
    print(f"  Failed:    {len(stats['failed_images'])}")

    print(f"\nMasks generated: {stats['total_masks']}")
    print(f"  Per class:")
    for class_name, count in sorted(stats['masks_per_class'].items()):
        print(f"    {class_name:15s}: {count:5d}")

    if stats['inference_times']:
        avg_time = np.mean(stats['inference_times'])
        print(f"\nAverage inference time: {avg_time:.2f}s per image")

    if stats['failed_images']:
        print(f"\nFailed images ({len(stats['failed_images'])}):")
        for path in stats['failed_images'][:5]:
            print(f"  - {path}")
        if len(stats['failed_images']) > 5:
            print(f"  ... and {len(stats['failed_images']) - 5} more")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate synthetic masks using fine-tuned SAM2')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed images')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned checkpoint (optional override)')

    args = parser.parse_args()

    # Override checkpoint if provided
    if args.checkpoint:
        CONFIG['checkpoint'] = Path(args.checkpoint)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nConfiguration:")
    print(f"  Checkpoint: {CONFIG['checkpoint']}")
    print(f"  Data root:  {CONFIG['data_root']}")
    print(f"  Output:     {CONFIG['output_root']}")
    print(f"  Resume:     {args.resume}")

    # Load model
    model = load_model(device)

    # Determine splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    # Process each split
    all_stats = {}
    total_start = time.time()

    for split in splits:
        stats = process_split(model, split, device, args.resume)
        all_stats[split] = stats
        print_statistics(stats, split)

    total_time = time.time() - total_start

    # Final summary
    print(f"\n{'='*60}")
    print(f" GENERATION COMPLETE")
    print(f"{'='*60}")

    total_masks = sum(s['total_masks'] for s in all_stats.values())
    total_images = sum(s['processed_images'] for s in all_stats.values())

    print(f"""
Summary:
  Splits processed: {', '.join(splits)}
  Total images:     {total_images}
  Total masks:      {total_masks}
  Total time:       {total_time/60:.1f} minutes

Output directory: {CONFIG['output_root']}
""")

    # Save statistics to JSON
    stats_path = CONFIG['output_root'] / 'generation_stats.json'

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(stats_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(CONFIG['checkpoint']),
            'splits': convert_to_serializable(all_stats),
            'total_time_minutes': total_time / 60,
        }, f, indent=2)

    print(f"Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()
