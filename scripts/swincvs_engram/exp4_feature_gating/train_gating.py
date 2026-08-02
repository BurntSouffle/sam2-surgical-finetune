"""
Training Script for Feature Gating Experiment (E4)

Trains the MaskEncoder while keeping SwinV2 backbone frozen.
Optionally fine-tunes LSTM and classifier.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from tqdm import tqdm

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
SWINCVS_ROOT = PROJECT_ROOT / 'SwinCVS'
SAM2_FINETUNE_ROOT = PROJECT_ROOT / 'sam2_finetune'

for path in [str(PROJECT_ROOT), str(SWINCVS_ROOT), str(SWINCVS_ROOT / 'scripts'),
             str(SAM2_FINETUNE_ROOT / 'scripts')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from model_gating import SwinCVSModelWithGating, MaskEncoder, create_gated_model_from_checkpoint
from dataset_gating import MaskGenerator, get_gating_datasets, get_dataloaders


def load_config(config_path):
    """Load YAML config file."""
    import yaml

    class ConfigDict(dict):
        """Dict that allows attribute access."""
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(f"Config has no attribute '{key}'")

        def __setattr__(self, key, value):
            self[key] = value

    def dict_to_config(d):
        if isinstance(d, dict):
            return ConfigDict({k: dict_to_config(v) for k, v in d.items()})
        return d

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return dict_to_config(config)


def compute_metrics(outputs, labels):
    """Compute AP and balanced accuracy for each criterion."""
    outputs = torch.sigmoid(outputs).cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = {}
    aps = []

    for i, name in enumerate(['C1', 'C2', 'C3']):
        # Average Precision
        try:
            ap = average_precision_score(labels[:, i], outputs[:, i])
        except:
            ap = 0.0
        aps.append(ap)
        metrics[f'AP_{name}'] = ap

        # Balanced Accuracy
        preds = (outputs[:, i] > 0.5).astype(int)
        try:
            ba = balanced_accuracy_score(labels[:, i], preds)
        except:
            ba = 0.0
        metrics[f'BA_{name}'] = ba

    metrics['mAP'] = np.mean(aps)
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_outputs = []
    all_labels = []
    batch_times = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        start_time = time.time()

        frames = batch['frames'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(frames, masks)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        total_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['avg_batch_time'] = np.mean(batch_times)

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()

    total_loss = 0
    all_outputs = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Evaluating'):
        frames = batch['frames'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(frames, masks)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Feature Gating Model')
    parser.add_argument('--config', type=str,
                        default=str(SCRIPT_DIR / 'config_gating.yaml'),
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                        default=str(SWINCVS_ROOT / 'weights' / 'SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt'),
                        help='Path to pretrained SwinCVS checkpoint')
    parser.add_argument('--coarse_checkpoint', type=str,
                        default=str(SAM2_FINETUNE_ROOT / 'checkpoints' / 'coarse_segmentation_v2' / 'run_20260120_234049' / 'best_model.pt'),
                        help='Path to coarse segmentation checkpoint')
    parser.add_argument('--freeze_lstm', action='store_true',
                        help='Freeze LSTM and classifier (only train MaskEncoder)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str,
                        default=str(SWINCVS_ROOT / 'results'),
                        help='Output directory for results')
    parser.add_argument('--experiment_name', type=str,
                        default='E4_FeatureGating',
                        help='Experiment name for saving results')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config - need to match SwinCVS expected format
    print("Creating config for SwinCVS compatibility...")

    class Config:
        pass

    config = Config()
    config.DATASET_DIR = str(PROJECT_ROOT)
    config.SEED = 42

    config.MODEL = Config()
    config.MODEL.LSTM = True  # Required for LSTM dataset

    config.TRAIN = Config()
    config.TRAIN.LIMIT_DATA_FRACTION = 1  # Use all data
    config.TRAIN.BATCH_SIZE = args.batch_size
    config.TRAIN.CLASS_WEIGHTS = [3.19852941, 4.46153846, 2.79518072]

    config.TRAIN.TRANSFORMS = Config()
    config.TRAIN.TRANSFORMS.ENDOSCAPES_MEAN = [0.485, 0.456, 0.406]
    config.TRAIN.TRANSFORMS.ENDOSCAPES_STD = [0.229, 0.224, 0.225]
    config.TRAIN.TRANSFORMS.CENTER_CROP = 480

    print(f"Config: DATASET_DIR={config.DATASET_DIR}, LIMIT_DATA_FRACTION={config.TRAIN.LIMIT_DATA_FRACTION}")

    # Set seed
    seed = getattr(config, 'SEED', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mask generator
    print(f"\nLoading coarse segmentation model...")
    if os.path.exists(args.coarse_checkpoint):
        mask_generator = MaskGenerator(args.coarse_checkpoint, device=str(device))
    else:
        print(f"Warning: Coarse checkpoint not found at {args.coarse_checkpoint}")
        print("Using zero masks")
        mask_generator = MaskGenerator(None, device=str(device))

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = get_gating_datasets(config, mask_generator)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size, num_workers=0
    )

    # Create model
    print(f"\nLoading pretrained model from {args.checkpoint}")
    model = create_gated_model_from_checkpoint(
        args.checkpoint,
        device=str(device),
        freeze_backbone=True,
        freeze_lstm=args.freeze_lstm
    )

    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    print(f"  - Backbone: {param_counts['backbone']:,}")
    print(f"  - LSTM: {param_counts['lstm']:,}")
    print(f"  - FC: {param_counts['fc']:,}")
    print(f"  - MaskEncoder: {param_counts['mask_encoder']:,}")

    # Loss function with class weights
    class_weights = getattr(config.TRAIN, 'CLASS_WEIGHTS', [1.0, 1.0, 1.0])
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer - only trainable params
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {args.experiment_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Freeze LSTM: {args.freeze_lstm}")
    print(f"{'='*60}\n")

    best_val_map = 0
    best_epoch = 0
    results = {
        'experiment': args.experiment_name,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'freeze_lstm': args.freeze_lstm,
            'checkpoint': args.checkpoint,
            'coarse_checkpoint': args.coarse_checkpoint,
        },
        'epochs': {},
        'best_epoch': 0,
        'best_val_map': 0,
        'test_results': None
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, mAP: {train_metrics['mAP']*100:.2f}%")
        print(f"        AP: C1={train_metrics['AP_C1']*100:.1f}%, C2={train_metrics['AP_C2']*100:.1f}%, C3={train_metrics['AP_C3']*100:.1f}%")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['mAP']*100:.2f}%")
        print(f"        AP: C1={val_metrics['AP_C1']*100:.1f}%, C2={val_metrics['AP_C2']*100:.1f}%, C3={val_metrics['AP_C3']*100:.1f}%")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_metrics['mAP'] > best_val_map:
            best_val_map = val_metrics['mAP']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_map': best_val_map,
            }, output_dir / f'{args.experiment_name}_best.pt')
            print(f"  -> New best model saved (mAP: {best_val_map*100:.2f}%)")

        # Store epoch results
        results['epochs'][epoch + 1] = {
            'train': {k: float(v) for k, v in train_metrics.items()},
            'val': {k: float(v) for k, v in val_metrics.items()}
        }

    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/60:.1f} minutes")
    print(f"Best validation mAP: {best_val_map*100:.2f}% (epoch {best_epoch})")
    print(f"{'='*60}")

    # Load best model for testing
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(output_dir / f'{args.experiment_name}_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test  - Loss: {test_metrics['loss']:.4f}, mAP: {test_metrics['mAP']*100:.2f}%")
    print(f"        AP: C1={test_metrics['AP_C1']*100:.1f}%, C2={test_metrics['AP_C2']*100:.1f}%, C3={test_metrics['AP_C3']*100:.1f}%")
    print(f"        BA: C1={test_metrics['BA_C1']*100:.1f}%, C2={test_metrics['BA_C2']*100:.1f}%, C3={test_metrics['BA_C3']*100:.1f}%")

    # Save results
    results['best_epoch'] = best_epoch
    results['best_val_map'] = float(best_val_map)
    results['test_results'] = {k: float(v) for k, v in test_metrics.items()}
    results['training_time_minutes'] = training_time / 60

    results_path = output_dir / f'{args.experiment_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Test mAP: {test_metrics['mAP']*100:.2f}%")
    print(f"  C1 AP: {test_metrics['AP_C1']*100:.2f}%")
    print(f"  C2 AP: {test_metrics['AP_C2']*100:.2f}%")
    print(f"  C3 AP: {test_metrics['AP_C3']*100:.2f}%")
    print(f"\nBaseline comparison:")
    print(f"  RGB-only: 68.82% mAP")
    print(f"  This exp: {test_metrics['mAP']*100:.2f}% mAP")
    print(f"  Delta: {(test_metrics['mAP']*100 - 68.82):+.2f}%")


if __name__ == '__main__':
    main()
