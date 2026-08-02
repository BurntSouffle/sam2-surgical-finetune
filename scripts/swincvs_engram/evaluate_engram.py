"""
E1 Experiment Evaluation Script
Evaluates the trained SwinCVS-Engram model and generates visualizations.

Outputs:
- Confusion matrices for C1, C2, C3
- Precision-Recall curves
- ROC curves
- Per-class metrics table
- Example predictions with engram masks
"""

import sys
from pathlib import Path

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
SWINCVS_ROOT = PROJECT_ROOT.parent / 'SwinCVS'

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT))
sys.path.insert(0, str(SWINCVS_ROOT / 'scripts'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, auc, balanced_accuracy_score
)
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import engram modules
from swincvs_engram.f_build_engram import build_engram_model
from swincvs_engram.f_dataset_engram import get_engram_datasets, get_dataloaders
from scripts.f_environment import get_config


def evaluate_model(model, dataloader, device='cuda'):
    """Run evaluation and collect predictions."""
    model.eval()

    # Set inference mode to avoid multiclassifier tuple output
    if hasattr(model, 'inference'):
        model.inference = True

    all_probs = []
    all_preds = []
    all_targets = []

    print(f"\nEvaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # Handle multiclassifier output (tuple of swin_classification, lstm_classification)
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # Use LSTM output

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return all_probs, all_preds, all_targets


def compute_metrics(probs, preds, targets, class_names=['C1', 'C2', 'C3']):
    """Compute detailed metrics for each class."""
    metrics = {}

    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        y_prob = probs[:, i]

        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Average Precision (AP)
        ap = average_precision_score(y_true, y_prob)

        # Balanced accuracy
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        metrics[name] = {
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1': f1,
            'AP': ap,
            'Balanced_Accuracy': bal_acc,
            'ROC_AUC': roc_auc,
            'Support_Positive': int(tp + fn),
            'Support_Negative': int(tn + fp)
        }

    # Overall mAP
    metrics['mAP'] = np.mean([metrics[name]['AP'] for name in class_names])
    metrics['mean_Balanced_Accuracy'] = np.mean([metrics[name]['Balanced_Accuracy'] for name in class_names])

    return metrics


def plot_confusion_matrices(preds, targets, class_names=['C1', 'C2', 'C3'], save_path=None):
    """Plot confusion matrices for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    criterion_labels = {
        'C1': 'C1: Two Structures',
        'C2': 'C2: Hepatocystic Triangle',
        'C3': 'C3: Cystic Plate'
    }

    for i, (name, ax) in enumerate(zip(class_names, axes)):
        cm = confusion_matrix(targets[:, i], preds[:, i], labels=[0, 1])

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])

        # Add percentages
        for j in range(2):
            for k in range(2):
                ax.text(k + 0.5, j + 0.7, f'({cm_norm[j, k]:.1%})',
                       ha='center', va='center', fontsize=9, color='gray')

        ax.set_title(f'{criterion_labels[name]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices - SwinCVS Engram (E1)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_precision_recall_curves(probs, targets, class_names=['C1', 'C2', 'C3'], save_path=None):
    """Plot Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    criterion_labels = {
        'C1': 'C1: Two Structures',
        'C2': 'C2: Hepatocystic Triangle',
        'C3': 'C3: Cystic Plate'
    }

    for i, (name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(targets[:, i], probs[:, i])
        ap = average_precision_score(targets[:, i], probs[:, i])

        ax.plot(recall, precision, color=color, lw=2,
               label=f'{criterion_labels[name]} (AP={ap:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - SwinCVS Engram (E1)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add mAP annotation
    mAP = np.mean([average_precision_score(targets[:, i], probs[:, i]) for i in range(3)])
    ax.text(0.95, 0.95, f'mAP: {mAP:.3f}', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_roc_curves(probs, targets, class_names=['C1', 'C2', 'C3'], save_path=None):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    criterion_labels = {
        'C1': 'C1: Two Structures',
        'C2': 'C2: Hepatocystic Triangle',
        'C3': 'C3: Cystic Plate'
    }

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
               label=f'{criterion_labels[name]} (AUC={roc_auc:.3f})')

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - SwinCVS Engram (E1)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_metrics_summary(metrics, save_path=None):
    """Plot summary bar chart of metrics."""
    class_names = ['C1', 'C2', 'C3']
    metric_names = ['AP', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metric_names))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        values = [metrics[name][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics - SwinCVS Engram (E1)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)

    # Add mAP annotation
    ax.text(0.02, 0.98, f"mAP: {metrics['mAP']:.3f}", transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def print_metrics_table(metrics):
    """Print formatted metrics table."""
    class_names = ['C1', 'C2', 'C3']

    print("\n" + "="*80)
    print(" EVALUATION METRICS - SwinCVS Engram (E1)")
    print("="*80)

    # Header
    print(f"\n{'Metric':<20} {'C1':>12} {'C2':>12} {'C3':>12} {'Mean':>12}")
    print("-"*68)

    metric_display = [
        ('AP', 'Average Precision'),
        ('Balanced_Accuracy', 'Balanced Acc'),
        ('Precision', 'Precision'),
        ('Recall', 'Recall'),
        ('Specificity', 'Specificity'),
        ('F1', 'F1 Score'),
        ('ROC_AUC', 'ROC AUC'),
    ]

    for key, display_name in metric_display:
        values = [metrics[name][key] for name in class_names]
        mean_val = np.mean(values)
        print(f"{display_name:<20} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f} {mean_val:>12.4f}")

    print("-"*68)
    print(f"{'mAP':<20} {'':<12} {'':<12} {'':<12} {metrics['mAP']:>12.4f}")
    print("="*80)

    # Confusion matrix summary
    print("\nConfusion Matrix Summary:")
    print(f"{'Class':<8} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8} {'Support+':>10} {'Support-':>10}")
    print("-"*62)
    for name in class_names:
        m = metrics[name]
        print(f"{name:<8} {m['TP']:>8} {m['TN']:>8} {m['FP']:>8} {m['FN']:>8} {m['Support_Positive']:>10} {m['Support_Negative']:>10}")
    print("="*80)


def main():
    print("="*70)
    print(" E1 EXPERIMENT EVALUATION")
    print("="*70)

    # Configuration
    config_path = SCRIPT_DIR / 'config' / 'SwinCVS_engram_config.yaml'
    # Use the checkpoint file which contains the best model from epoch 10 (mAP 0.4525)
    weights_path = SWINCVS_ROOT / 'weights' / 'SwinCVS_E2E_MC_IMNP_sd5_checkpoint.pt'
    output_dir = PROJECT_ROOT / 'evaluation_results' / 'E1_engram'

    print(f"\nConfig: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print("\nLoading configuration...")
    config, experiment_name = get_config(str(config_path))

    # Load datasets
    print("\nLoading test dataset...")
    train_dataset, val_dataset, test_dataset = get_engram_datasets(config)

    _, _, test_loader = get_dataloaders(
        config, train_dataset, val_dataset, test_dataset
    )

    print(f"Test samples: {len(test_dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_engram_model(config)

    # Load weights from checkpoint
    print(f"\nLoading weights from: {weights_path.name}")
    checkpoint = torch.load(weights_path, map_location='cuda', weights_only=False)

    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
        print(f"  Best mAP (val): {checkpoint.get('best_mAP', 'N/A'):.4f}")
    else:
        # Direct state dict (legacy format)
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()

    # Evaluate
    probs, preds, targets = evaluate_model(model, test_loader)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(probs, preds, targets)

    # Print metrics table
    print_metrics_table(metrics)

    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Confusion matrices
    plot_confusion_matrices(
        preds, targets,
        save_path=output_dir / 'confusion_matrices.png'
    )

    # 2. Precision-Recall curves
    plot_precision_recall_curves(
        probs, targets,
        save_path=output_dir / 'precision_recall_curves.png'
    )

    # 3. ROC curves
    plot_roc_curves(
        probs, targets,
        save_path=output_dir / 'roc_curves.png'
    )

    # 4. Metrics summary bar chart
    plot_metrics_summary(
        metrics,
        save_path=output_dir / 'metrics_summary.png'
    )

    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  - metrics.json")
    print(f"  - confusion_matrices.png")
    print(f"  - precision_recall_curves.png")
    print(f"  - roc_curves.png")
    print(f"  - metrics_summary.png")

    return metrics


if __name__ == '__main__':
    metrics = main()
