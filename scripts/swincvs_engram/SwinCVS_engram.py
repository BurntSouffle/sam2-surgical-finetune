"""
E1 Experiment: SwinCVS with Visual Engrams

This is the main training script for the E1 (Visual Engrams) experiment.
It trains SwinCVS with 5-channel input: RGB + GB mask + Tool mask.

The hypothesis is that providing explicit segmentation masks as additional
input channels helps the model assess C1 (hepatocystic triangle clearance).

Usage:
    # From sam2_finetune directory:
    python scripts/swincvs_engram/SwinCVS_engram.py --config_path scripts/swincvs_engram/config/SwinCVS_engram_config.yaml

    # Or with default config:
    python scripts/swincvs_engram/SwinCVS_engram.py

    # Resume from checkpoint:
    python scripts/swincvs_engram/SwinCVS_engram.py --resume weights/E1_SwinCVS_Engram_v1_checkpoint.pt

    # Resume and train for more epochs:
    python scripts/swincvs_engram/SwinCVS_engram.py --resume weights/E1_SwinCVS_Engram_v1_checkpoint.pt --epochs 16

Based on original SwinCVS.py with modifications for engram input handling.
"""

print('='*70)
print(' E1 EXPERIMENT: SwinCVS with Visual Engrams')
print('='*70)
print('\nImporting libraries...')

# Standard library imports
import argparse
import time
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

# Third-party imports
import torch
import numpy as np
import torch.nn as nn

# Setup paths before local imports
SCRIPT_DIR = Path(__file__).parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent  # sam2_finetune
DISSERTATION_ROOT = PROJECT_ROOT.parent
SWINCVS_ROOT = DISSERTATION_ROOT / 'SwinCVS'

# Add to sys.path
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))

# Fix SAM2 import path BEFORE any imports that use SAM2
try:
    from sam2_path_fix import fix_sam2_path
    fix_sam2_path(verbose=False)
except ImportError:
    pass  # Path fix not available, continue anyway

# Local imports - ENGRAM versions
from swincvs_engram.f_dataset_engram import get_engram_datasets, get_dataloaders
from swincvs_engram.f_build_engram import build_engram_model, verify_engram_model

# Local imports - Original SwinCVS utilities (keep these)
from scripts.f_environment import get_config, set_deterministic_behaviour, verify_results_weights_folder
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount
from scripts.f_metrics import get_map, get_balanced_accuracies
from scripts.f_training import save_weights

warnings.filterwarnings("ignore")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_engram_config(config):
    """Print engram-specific configuration details."""
    print("\n" + "="*70)
    print(" ENGRAM CONFIGURATION")
    print("="*70)

    # Check if engram is enabled
    engram_enabled = hasattr(config, 'ENGRAM') and getattr(config.ENGRAM, 'ENABLED', False)

    print(f"\n  ENGRAM.ENABLED:     {engram_enabled}")

    if engram_enabled:
        print(f"  CHECKPOINT_PATH:    {getattr(config.ENGRAM, 'CHECKPOINT_PATH', 'Not specified')}")
        print(f"  GB_CHANNEL:         {getattr(config.ENGRAM, 'GB_CHANNEL', True)}")
        print(f"  TOOL_CHANNEL:       {getattr(config.ENGRAM, 'TOOL_CHANNEL', True)}")
        print(f"  MASK_THRESHOLD:     {getattr(config.ENGRAM, 'MASK_THRESHOLD', 0.5)}")
        print(f"  MASK_MEAN:          {getattr(config.ENGRAM, 'MASK_MEAN', [0.5, 0.5])}")
        print(f"  MASK_STD:           {getattr(config.ENGRAM, 'MASK_STD', [0.5, 0.5])}")
        print(f"  CACHE_MASKS:        {getattr(config.ENGRAM, 'CACHE_MASKS', False)}")

    print(f"\n  BACKBONE.IN_CHANS:  {config.BACKBONE.SWINV2.IN_CHANS}")

    expected_chans = 5 if engram_enabled else 3
    if config.BACKBONE.SWINV2.IN_CHANS != expected_chans:
        print(f"  [WARN] IN_CHANS mismatch! Expected {expected_chans} for ENGRAM.ENABLED={engram_enabled}")

    print("="*70 + "\n")


def print_model_summary(model, config):
    """Print model architecture summary."""
    print("\n" + "="*70)
    print(" MODEL SUMMARY")
    print("="*70)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")

    print(f"\n  LSTM:                 {config.MODEL.LSTM}")
    print(f"  E2E (unfrozen):       {config.MODEL.E2E}")
    print(f"  MULTICLASSIFIER:      {config.MODEL.MULTICLASSIFIER}")

    # Check input channels
    if hasattr(model, 'backbone'):
        in_chans = model.backbone.patch_embed.proj.in_channels
    else:
        in_chans = model.patch_embed.proj.in_channels
    print(f"  Input channels:       {in_chans}")

    print("="*70 + "\n")


def save_checkpoint(model, optimizer, scheduler, epoch, best_mAP, best_epoch,
                    experiment_name, checkpoint_dir, results_dict=None):
    """
    Save a full training checkpoint for resumption.

    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler (can be None)
        epoch: Current epoch number (0-indexed)
        best_mAP: Best mAP achieved so far
        best_epoch: Epoch that achieved best mAP
        experiment_name: Name of the experiment
        checkpoint_dir: Directory to save checkpoint
        results_dict: Optional results dictionary to include
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'experiment_name': experiment_name,
    }

    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Add results dict if provided
    if results_dict is not None:
        checkpoint['results_dict'] = results_dict

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'{experiment_name}_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"  >>> Checkpoint saved: {checkpoint_path.name}")

    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """
    Load a training checkpoint and restore state.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to restore state
        scheduler: Optional scheduler to restore state

    Returns:
        dict with: start_epoch, best_mAP, best_epoch, results_dict
    """
    print(f"\n[RESUMING FROM CHECKPOINT]")
    print(f"  Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model weights restored")

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"  Optimizer state restored")

    # Restore scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Scheduler state restored")

    # Get training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_mAP = checkpoint.get('best_mAP', 0)
    best_epoch = checkpoint.get('best_epoch', 0)
    results_dict = checkpoint.get('results_dict', None)

    print(f"\n  Resuming from epoch {start_epoch}")
    print(f"  Best mAP so far: {best_mAP:.4f} (epoch {best_epoch})")

    return {
        'start_epoch': start_epoch,
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'results_dict': results_dict,
    }


def get_engram_results_metadata(config):
    """Get engram-specific metadata for results saving."""
    metadata = {
        'experiment': 'E1_Visual_Engrams',
        'timestamp': datetime.now().isoformat(),
    }

    if hasattr(config, 'ENGRAM'):
        metadata['engram'] = {
            'enabled': getattr(config.ENGRAM, 'ENABLED', False),
            'checkpoint_path': str(getattr(config.ENGRAM, 'CHECKPOINT_PATH', '')),
            'gb_channel': getattr(config.ENGRAM, 'GB_CHANNEL', True),
            'tool_channel': getattr(config.ENGRAM, 'TOOL_CHANNEL', True),
            'mask_threshold': getattr(config.ENGRAM, 'MASK_THRESHOLD', 0.5),
            'mask_mean': getattr(config.ENGRAM, 'MASK_MEAN', [0.5, 0.5]),
            'mask_std': getattr(config.ENGRAM, 'MASK_STD', [0.5, 0.5]),
        }

    metadata['backbone'] = {
        'in_chans': config.BACKBONE.SWINV2.IN_CHANS,
        'embed_dim': config.BACKBONE.SWINV2.EMBED_DIM,
        'pretrained': config.BACKBONE.PRETRAINED,
    }

    metadata['training'] = {
        'epochs': config.TRAIN.EPOCHS,
        'batch_size': config.TRAIN.BATCH_SIZE,
        'lr': config.TRAIN.BASE_LR,
        'seed': config.SEED,
    }

    return metadata


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function for E1 experiment."""

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    pwd = Path.cwd()
    print(f"\nCurrent working directory: {pwd}")

    # Verify we're in the right place (should be in SwinCVS directory for weights)
    # Or set up paths appropriately
    swincvs_pwd = SWINCVS_ROOT
    print(f"SwinCVS directory: {swincvs_pwd}")

    # Verify necessary folder structure and download weights
    # This creates results/ and weights/ folders if needed
    verify_results_weights_folder(swincvs_pwd)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    parser = argparse.ArgumentParser(description="E1 Experiment: SwinCVS with Visual Engrams")
    parser.add_argument(
        '--config_path',
        type=str,
        default=str(SCRIPT_DIR / 'config' / 'SwinCVS_engram_config.yaml'),
        help='Path to config YAML file (default: engram config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs (useful when resuming to train longer)'
    )
    args = parser.parse_args()
    config_path = args.config_path
    resume_path = args.resume
    override_epochs = args.epochs

    print(f"\nLoading config from: {config_path}")
    config, experiment_name = get_config(config_path)

    # Set deterministic behavior
    seed = config.SEED
    set_deterministic_behaviour(seed)

    # Print engram configuration
    print_engram_config(config)

    # =========================================================================
    # DATASET AND DATALOADER (ENGRAM VERSION)
    # =========================================================================
    print("\n" + "#"*70)
    print("# LOADING DATASETS")
    print("#"*70)

    training_dataset, val_dataset, test_dataset = get_engram_datasets(config)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        config, training_dataset, val_dataset, test_dataset
    )

    print(f"\nDataloader sizes:")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches:   {len(val_dataloader)}")
    print(f"  Test batches:  {len(test_dataloader)}")

    # =========================================================================
    # MODEL INITIALIZATION (ENGRAM VERSION)
    # =========================================================================
    print("\n" + "#"*70)
    print("# BUILDING MODEL")
    print("#"*70)

    model = build_engram_model(config)
    print('\nFull model initialized successfully!')

    # Print model summary
    print_model_summary(model, config)

    # Load saved weights for inference mode
    if config.MODEL.INFERENCE:
        weights = swincvs_pwd / "weights" / config.MODEL.INFERENCE_WEIGHTS
        model.load_state_dict(torch.load(weights, weights_only=False))
        print(f"Trained weights loaded for INFERENCE: {config.MODEL.INFERENCE_WEIGHTS}")

    model.to('cuda')
    torch.cuda.empty_cache()

    # =========================================================================
    # OPTIMIZER AND LOSS
    # =========================================================================
    print("\n" + "#"*70)
    print("# SETTING UP TRAINING")
    print("#"*70)

    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount()
    class_weights = torch.tensor(config.TRAIN.CLASS_WEIGHTS).to('cuda')
    criterion = nn.BCEWithLogitsLoss(weight=class_weights).to('cuda')

    print(f"\n  Optimizer:      {config.TRAIN.OPTIMIZER.NAME}")
    print(f"  Base LR:        {config.TRAIN.BASE_LR}")
    print(f"  Class weights:  {config.TRAIN.CLASS_WEIGHTS}")
    print(f"  Gradient clip:  {config.TRAIN.CLIP_GRAD}")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "#"*70)
    print(f"# EXPERIMENT: {experiment_name}")
    print("#"*70)

    # Initialize results with metadata
    results_dict = get_engram_results_metadata(config)
    results_dict['epochs'] = {}

    # Training timing statistics
    batch_times = []
    epoch_times = []

    if not config.MODEL.INFERENCE:
        # Use override epochs if provided, otherwise use config
        num_epochs = override_epochs if override_epochs is not None else config.TRAIN.EPOCHS
        checkpoint_dir = swincvs_pwd / 'weights'
        best_mAP = 0
        best_epoch = 0
        start_epoch = 0

        if config.MODEL.MULTICLASSIFIER:
            multiclassifier_alpha = config.TRAIN.MULTICLASSIFIER_ALPHA
            multiclassifier_beta = 1 - multiclassifier_alpha

        # Handle resumption from checkpoint
        if resume_path is not None:
            resume_checkpoint = Path(resume_path)
            if not resume_checkpoint.is_absolute():
                resume_checkpoint = swincvs_pwd / resume_path

            if resume_checkpoint.exists():
                resume_state = load_checkpoint(
                    resume_checkpoint, model, optimizer, scheduler=None
                )
                start_epoch = resume_state['start_epoch']
                best_mAP = resume_state['best_mAP']
                best_epoch = resume_state['best_epoch']

                # Restore results dict if available
                if resume_state['results_dict'] is not None:
                    results_dict = resume_state['results_dict']

                # Adjust epochs if not overridden
                if override_epochs is None:
                    # Keep original total epochs from config
                    pass
                else:
                    num_epochs = override_epochs

                print(f"\n  Continuing training from epoch {start_epoch + 1} to {num_epochs}")
            else:
                print(f"\n[WARNING] Checkpoint not found: {resume_checkpoint}")
                print("  Starting training from scratch...")

        print(f"\n  Total epochs:   {num_epochs}")
        print(f"  Start epoch:    {start_epoch + 1}")
        print(f"  Batch size:     {config.TRAIN.BATCH_SIZE}")
        print(f"  Warmup epochs:  {config.TRAIN.WARMUP_EPOCHS}")

        print("\n" + "-"*70)
        print("Beginning training...")
        print("-"*70)

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            print(f"\n{'='*70}")
            print(f"Epoch: {epoch+1:02}/{num_epochs:02}")
            print(f"{'='*70}")

            # Update weight scaling parameters
            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                multiclassifier_alpha, multiclassifier_beta = update_params(
                    multiclassifier_alpha, multiclassifier_beta, epoch
                )

            train_loss = 0.0
            val_loss = 0.0
            epoch_batch_times = []

            # -----------------------------------------------------------------
            # TRAINING
            # -----------------------------------------------------------------
            model.train()
            optimizer.zero_grad()

            print("\n[TRAINING]")
            for idx, (samples, targets) in enumerate(train_dataloader):
                batch_start = time.time()
                print(f"  Batch: {idx+1:04}/{len(train_dataloader):04}", end="\r")

                # Get predictions
                samples, targets = samples.to('cuda'), targets.to('cuda')

                with torch.amp.autocast("cuda", enabled=True):
                    if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                        outputs_swin, outputs_lstm = model(samples)
                    else:
                        outputs_lstm = model(samples)

                # Get loss and backpropagation
                if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                    loss_train = (multiclassifier_alpha * criterion(outputs_swin, targets) +
                                  multiclassifier_beta * criterion(outputs_lstm, targets))
                else:
                    loss_train = criterion(outputs_lstm, targets)

                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(
                    loss_train, optimizer,
                    clip_grad=config.TRAIN.CLIP_GRAD,
                    parameters=model.parameters(),
                    create_graph=is_second_order,
                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
                )
                optimizer.zero_grad()
                train_loss += loss_train.item()
                torch.cuda.synchronize()

                # Track batch time
                batch_time = time.time() - batch_start
                epoch_batch_times.append(batch_time)

            # Calculate average batch time for this epoch
            avg_batch_time = np.mean(epoch_batch_times) * 1000  # ms
            batch_times.extend(epoch_batch_times)

            print(f"\n  Train loss:     {train_loss:.4f}")
            print(f"  Avg batch time: {avg_batch_time:.1f}ms")

            # -----------------------------------------------------------------
            # VALIDATION
            # -----------------------------------------------------------------
            print("\n[VALIDATION]")
            model.eval()
            val_probabilities = []
            val_predictions = []
            val_targets = []

            with torch.inference_mode():
                for idx, (samples, targets) in enumerate(val_dataloader):
                    print(f"  Batch: {idx+1:04}/{len(val_dataloader):04}", end="\r")

                    samples, targets = samples.to('cuda'), targets.to('cuda')

                    if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                        outputs_swin, outputs_lstm = model(samples)
                    else:
                        outputs_lstm = model(samples)

                    # Get outputs
                    val_probability = torch.sigmoid(outputs_lstm)
                    val_prediction = torch.round(val_probability)

                    # Save outputs
                    val_probabilities.append(val_probability.to('cpu'))
                    val_predictions.append(val_prediction.to('cpu'))
                    val_targets.append(targets.to('cpu'))

                    # Loss
                    loss_val = criterion(outputs_lstm, targets)
                    val_loss += loss_val.item()
                    torch.cuda.synchronize()

            # Get validation scores
            C1_bacc, C2_bacc, C3_bacc, total_bacc = get_balanced_accuracies(val_targets, val_predictions)
            C1_ap, C2_ap, C3_ap, mAP = get_map(val_targets, val_probabilities)

            print(f"\n  Val loss:       {val_loss:.4f}")
            print(f"\n  Balanced Accuracy:")
            print(f"    C1: {C1_bacc:.4f}  C2: {C2_bacc:.4f}  C3: {C3_bacc:.4f}  Avg: {total_bacc:.4f}")
            print(f"  Average Precision (AP):")
            print(f"    C1: {C1_ap:.4f}  C2: {C2_ap:.4f}  C3: {C3_ap:.4f}  mAP: {mAP:.4f}")

            # Save validation scores
            val_predictions_2save = torch.cat(val_predictions, dim=0).tolist()
            val_probabilities_2save = torch.cat(val_probabilities, dim=0).tolist()
            val_targets_2save = torch.cat(val_targets, dim=0).tolist()

            epoch_results = {
                'avg_bal_acc': round(total_bacc, 4),
                'C1_bacc': round(C1_bacc, 4),
                'C2_bacc': round(C2_bacc, 4),
                'C3_bacc': round(C3_bacc, 4),
                'avg_map': round(mAP, 4),
                'C1_map': round(C1_ap, 4),
                'C2_map': round(C2_ap, 4),
                'C3_map': round(C3_ap, 4),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'avg_batch_time_ms': round(avg_batch_time, 2),
                'preds': val_predictions_2save,
                'true': val_targets_2save,
                'preds_prob': val_probabilities_2save,
            }
            results_dict['epochs'][f"Epoch_{epoch+1:02}"] = epoch_results

            # Save results after each epoch
            results_path = swincvs_pwd / 'results' / f'{experiment_name}_results.json'
            with open(results_path, 'w') as file:
                json.dump(results_dict, file, indent=4)

            # Estimate remaining time
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = np.mean(epoch_times) * remaining_epochs

            print(f"\n  Epoch duration: {epoch_time:.1f}s")
            if remaining_epochs > 0:
                print(f"  Est. remaining: {estimated_remaining:.0f}s ({estimated_remaining/60:.1f}min)")

            # Save weights of the best epoch
            if mAP >= best_mAP:
                best_mAP = mAP
                best_epoch = epoch + 1
                print(f"\n  >>> New best mAP: {best_mAP:.4f} (Epoch {best_epoch})")
                print(f"  >>> Saving best weights...")
                save_weights(model, config, experiment_name)
            else:
                print(f"\n  Best so far: mAP={best_mAP:.4f} @ Epoch {best_epoch}")

            # Save checkpoint for resumption (every epoch)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,  # Add scheduler here if using one
                epoch=epoch,
                best_mAP=best_mAP,
                best_epoch=best_epoch,
                experiment_name=experiment_name,
                checkpoint_dir=checkpoint_dir,
                results_dict=results_dict
            )

        # Training summary
        print("\n" + "="*70)
        print(" TRAINING COMPLETE")
        print("="*70)
        print(f"\n  Best mAP:       {best_mAP:.4f}")
        print(f"  Best epoch:     {best_epoch}")
        print(f"  Total time:     {sum(epoch_times):.1f}s ({sum(epoch_times)/60:.1f}min)")
        print(f"  Avg epoch time: {np.mean(epoch_times):.1f}s")
        print(f"  Avg batch time: {np.mean(batch_times)*1000:.1f}ms")

        # Print checkpoint info for resumption
        checkpoint_file = checkpoint_dir / f'{experiment_name}_checkpoint.pt'
        print(f"\n  Checkpoint:     {checkpoint_file}")
        print(f"  Best weights:   {checkpoint_dir / f'{experiment_name}_bestMAP.pt'}")
        print(f"\n  To continue training, run:")
        print(f"    python scripts/swincvs_engram/SwinCVS_engram.py --resume {checkpoint_file.relative_to(swincvs_pwd)} --epochs <N>")

        # Store training summary
        results_dict['training_summary'] = {
            'best_mAP': round(best_mAP, 4),
            'best_epoch': best_epoch,
            'total_time_s': round(sum(epoch_times), 1),
            'avg_epoch_time_s': round(np.mean(epoch_times), 1),
            'avg_batch_time_ms': round(np.mean(batch_times) * 1000, 1),
        }

    # =========================================================================
    # TESTING
    # =========================================================================
    print("\n" + "#"*70)
    print("# TESTING")
    print("#"*70)

    # Load best weights if we trained
    if not config.MODEL.INFERENCE and 'best_epoch' in dir():
        print(f"\nLoading best weights from epoch {best_epoch}...")
        weights_path = swincvs_pwd / 'weights' / f'{experiment_name}_bestMAP.pt'
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, weights_only=False))
            print(f"  Loaded: {weights_path.name}")

    # Test time measurement variables
    test_times = []

    # Performance measurement variables
    test_probabilities = []
    test_predictions = []
    test_targets = []

    print(f"\n[TESTING on {len(test_dataloader)} samples]")
    model.eval()

    with torch.inference_mode():
        for idx, (samples, targets) in enumerate(test_dataloader):
            print(f"  Batch: {idx+1:04}/{len(test_dataloader):04}", end="\r")

            # Time start
            start_time = time.time()

            # Get predictions
            samples, targets = samples.to('cuda'), targets.to('cuda')

            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER and not config.MODEL.INFERENCE:
                outputs_swin, outputs_lstm = model(samples)
            else:
                outputs_lstm = model(samples)

            # Get outputs
            test_probability = torch.sigmoid(outputs_lstm)
            test_prediction = torch.round(test_probability)

            # Time end
            elapsed_time = time.time() - start_time
            test_times.append(elapsed_time)

            # Save results from a batch to a list
            test_probabilities.append(test_probability.to('cpu'))
            test_predictions.append(test_prediction.to('cpu'))
            test_targets.append(targets.to('cpu'))

            torch.cuda.synchronize()

    # Calculate metrics
    C1_bacc, C2_bacc, C3_bacc, total_bacc = get_balanced_accuracies(test_targets, test_predictions)
    C1_ap, C2_ap, C3_ap, mAP = get_map(test_targets, test_probabilities)

    # Print metrics
    print("\n\n" + "="*70)
    print(" TEST RESULTS")
    print("="*70)
    print(f"\n  Balanced Accuracy:")
    print(f"    C1: {C1_bacc:.4f}  C2: {C2_bacc:.4f}  C3: {C3_bacc:.4f}  Avg: {total_bacc:.4f}")
    print(f"  Average Precision (AP):")
    print(f"    C1: {C1_ap:.4f}  C2: {C2_ap:.4f}  C3: {C3_ap:.4f}  mAP: {mAP:.4f}")

    mean_inference = round(np.mean(test_times) * 1000, 1)
    std_inference = round(np.std(test_times) * 1000, 1)
    total_inference = round(np.sum(test_times), 1)
    print(f"\n  Inference time: mean={mean_inference}ms, std={std_inference}ms, total={total_inference}s")

    # Save test results
    test_predictions_2save = torch.cat(test_predictions, dim=0).tolist()
    test_probabilities_2save = torch.cat(test_probabilities, dim=0).tolist()
    test_targets_2save = torch.cat(test_targets, dim=0).tolist()

    test_results = {
        'avg_bal_acc': round(total_bacc, 4),
        'C1_bacc': round(C1_bacc, 4),
        'C2_bacc': round(C2_bacc, 4),
        'C3_bacc': round(C3_bacc, 4),
        'avg_map': round(mAP, 4),
        'C1_map': round(C1_ap, 4),
        'C2_map': round(C2_ap, 4),
        'C3_map': round(C3_ap, 4),
        'mean_inference_ms': mean_inference,
        'std_inference_ms': std_inference,
        'total_inference_s': total_inference,
        'preds': test_predictions_2save,
        'true': test_targets_2save,
        'preds_prob': test_probabilities_2save,
    }

    best_epoch_key = best_epoch if 'best_epoch' in dir() else 0
    results_dict['test_results'] = test_results
    results_dict['test_results']['best_epoch'] = best_epoch_key

    # Save final results
    results_path = swincvs_pwd / 'results' / f'{experiment_name}_results.json'
    with open(results_path, 'w') as file:
        json.dump(results_dict, file, indent=4)

    print(f"\n  Results saved to: {results_path}")
    print("="*70)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "#"*70)
    print("# E1 EXPERIMENT COMPLETE")
    print("#"*70)
    print(f"\n  Experiment:     {experiment_name}")
    print(f"  Test mAP:       {mAP:.4f}")
    print(f"    C1 AP:        {C1_ap:.4f}")
    print(f"    C2 AP:        {C2_ap:.4f}")
    print(f"    C3 AP:        {C3_ap:.4f}")

    engram_enabled = hasattr(config, 'ENGRAM') and getattr(config.ENGRAM, 'ENABLED', False)
    print(f"\n  Engram enabled: {engram_enabled}")
    print(f"  Input channels: {config.BACKBONE.SWINV2.IN_CHANS}")
    print("#"*70 + "\n")

    return results_dict


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
