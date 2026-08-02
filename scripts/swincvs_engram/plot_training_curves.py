"""
Plot training curves from E1 experiment results.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_path: Path) -> dict:
    """Load results JSON file."""
    with open(results_path, 'r') as f:
        # Read line by line for large JSON
        content = f.read()
        return json.loads(content)


def plot_training_curves(output_path: Path):
    """Plot training and validation curves from E1 results."""

    results_path = Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\SwinCVS\results\SwinCVS_E2E_MC_IMNP_sd5_results.json')

    print("="*60)
    print(" E1 TRAINING CURVES")
    print("="*60)

    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)

    # Extract epoch data
    epochs_data = results['epochs']
    epoch_nums = []
    train_losses = []
    val_losses = []
    val_maps = []
    c1_maps = []
    c2_maps = []
    c3_maps = []

    for epoch_key, data in sorted(epochs_data.items()):
        epoch_num = int(epoch_key.split('_')[1])
        epoch_nums.append(epoch_num)
        train_losses.append(data.get('train_loss', 0))
        val_losses.append(data.get('val_loss', 0))
        val_maps.append(data.get('avg_map', 0))
        c1_maps.append(data.get('C1_map', 0))
        c2_maps.append(data.get('C2_map', 0))
        c3_maps.append(data.get('C3_map', 0))

    print(f"Found {len(epoch_nums)} epochs")
    print(f"Epoch range: {min(epoch_nums)} to {max(epoch_nums)}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epoch_nums, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Check for NaN or explosion
    if any(np.isnan(train_losses)) or any(np.isinf(train_losses)):
        ax1.text(0.5, 0.5, 'WARNING: NaN/Inf detected!', transform=ax1.transAxes,
                fontsize=14, color='red', ha='center')

    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(epoch_nums, val_losses, 'r-', linewidth=2, marker='o', markersize=4, label='Val Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Validation mAP
    ax3 = axes[1, 0]
    ax3.plot(epoch_nums, val_maps, 'g-', linewidth=2, marker='o', markersize=4, label='Val mAP')
    ax3.axhline(y=max(val_maps), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(val_maps):.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('mAP', fontsize=12)
    ax3.set_title('Validation mAP', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 0.6])

    # Mark best epoch
    best_idx = np.argmax(val_maps)
    ax3.scatter([epoch_nums[best_idx]], [val_maps[best_idx]], color='red', s=100, zorder=5)
    ax3.annotate(f'Best: Epoch {epoch_nums[best_idx]}',
                xy=(epoch_nums[best_idx], val_maps[best_idx]),
                xytext=(epoch_nums[best_idx]+1, val_maps[best_idx]+0.02),
                fontsize=10)

    # Plot 4: Per-class mAP
    ax4 = axes[1, 1]
    ax4.plot(epoch_nums, c1_maps, 'g-', linewidth=2, marker='o', markersize=3, label='C1 (Two Structures)')
    ax4.plot(epoch_nums, c2_maps, 'b-', linewidth=2, marker='s', markersize=3, label='C2 (Hepatocystic)')
    ax4.plot(epoch_nums, c3_maps, 'r-', linewidth=2, marker='^', markersize=3, label='C3 (Cystic Plate)')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('AP', fontsize=12)
    ax4.set_title('Per-Class Average Precision', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 0.6])

    plt.suptitle('E1 Visual Engrams Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs:     {len(epoch_nums)}")
    print(f"Initial train loss: {train_losses[0]:.2f}")
    print(f"Final train loss:   {train_losses[-1]:.2f}")
    print(f"Initial val loss:   {val_losses[0]:.2f}")
    print(f"Final val loss:     {val_losses[-1]:.2f}")
    print(f"Best val mAP:       {max(val_maps):.4f} (Epoch {epoch_nums[np.argmax(val_maps)]})")
    print(f"Final val mAP:      {val_maps[-1]:.4f}")

    # Check for issues
    print("\n" + "="*60)
    print("DIAGNOSTICS")
    print("="*60)

    # Loss decreased?
    if train_losses[-1] < train_losses[0]:
        print("[OK] Training loss decreased (pipeline works)")
    else:
        print("[WARNING] Training loss did NOT decrease")

    # NaN/Inf?
    if any(np.isnan(train_losses)) or any(np.isinf(train_losses)):
        print("[ERROR] NaN/Inf in training loss!")
    else:
        print("[OK] No NaN/Inf in losses")

    # Early plateau?
    if len(val_maps) > 5:
        later_maps = val_maps[5:]
        early_maps = val_maps[:5]
        if np.std(later_maps) < 0.01 and np.mean(later_maps) < 0.4:
            print("[INFO] Loss may have plateaued early")
        else:
            print("[OK] Training shows continued improvement")


if __name__ == '__main__':
    output_path = Path(r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\sam2_finetune\diagnostics\e1_training_curves.png')
    plot_training_curves(output_path)
