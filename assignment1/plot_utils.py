"""
This module implements plotting utilities for analyzing MLP training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def plot_training_curves(logging_dict, val_accuracies, test_accuracy, save_dir='plots', framework='numpy'):
    """
    Creates and saves comprehensive training plots.

    Args:
        logging_dict: Dictionary containing training metrics:
            - 'batch_loss': list of losses per batch
            - 'batch_train_accuracy': list of training accuracies per batch
            - 'train_accuracy_per_epoch': list of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy (scalar)
        save_dir: Directory to save plots (default: 'plots')
        framework: Framework used ('numpy' or 'pytorch', default: 'numpy')
    """

    # create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # format framework name for display
    framework_display = framework.upper()

    # extract data
    batch_losses = logging_dict['batch_loss']
    batch_train_accuracies = logging_dict['batch_train_accuracy']
    train_accuracies_per_epoch = logging_dict['train_accuracy_per_epoch']

    # number of epochs
    num_epochs = len(train_accuracies_per_epoch)
    batches_per_epoch = len(batch_losses) // num_epochs if num_epochs > 0 else 0

    # create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'MLP Training Analysis ({framework_display})', fontsize=16, fontweight='bold')

    # --------------------- plot 1: Training Loss per Batch ---------------------
    ax1 = axes[0, 0]
    ax1.plot(batch_losses, alpha=0.6, linewidth=0.8, label='Batch Loss')

    # add epoch markers
    for epoch in range(1, num_epochs):
        ax1.axvline(x=epoch * batches_per_epoch, color='red', linestyle='--',
                   alpha=0.3, linewidth=1)

    # add smoothed curve (moving average)
    if len(batch_losses) > 50:
        window_size = min(50, len(batch_losses) // 10)
        smoothed_loss = np.convolve(batch_losses,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
        ax1.plot(range(window_size//2, len(smoothed_loss) + window_size//2),
                smoothed_loss, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')

    ax1.set_xlabel('Batch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss per Batch', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --------------------- plot 2: Training Accuracy per Batch ---------------------
    ax2 = axes[0, 1]
    ax2.plot(batch_train_accuracies, alpha=0.6, linewidth=0.8, label='Batch Accuracy')

    # add epoch markers
    for epoch in range(1, num_epochs):
        ax2.axvline(x=epoch * batches_per_epoch, color='red', linestyle='--',
                   alpha=0.3, linewidth=1)

    # add smoothed curve
    if len(batch_train_accuracies) > 50:
        window_size = min(50, len(batch_train_accuracies) // 10)
        smoothed_acc = np.convolve(batch_train_accuracies,
                                   np.ones(window_size)/window_size,
                                   mode='valid')
        ax2.plot(range(window_size//2, len(smoothed_acc) + window_size//2),
                smoothed_acc, 'g-', linewidth=2, label=f'Smoothed (window={window_size})')

    ax2.set_xlabel('Batch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy per Batch', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # --------------------- plot 3: Train vs Validation Accuracy per Epoch ---------------------
    ax3 = axes[1, 0]
    epochs = range(1, num_epochs + 1)

    ax3.plot(epochs, train_accuracies_per_epoch, 'o-', linewidth=2,
            markersize=8, label='Train Accuracy', color='blue')
    ax3.plot(epochs, val_accuracies, 's-', linewidth=2,
            markersize=8, label='Validation Accuracy', color='orange')

    # mark best validation accuracy
    best_val_idx = np.argmax(val_accuracies)
    best_val_acc = val_accuracies[best_val_idx]
    ax3.plot(best_val_idx + 1, best_val_acc, 'r*', markersize=20,
            label=f'Best Val Acc: {best_val_acc:.4f}')

    # add test accuracy as horizontal line
    ax3.axhline(y=test_accuracy, color='green', linestyle='--', linewidth=2,
               label=f'Test Accuracy: {test_accuracy:.4f}')

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Train vs Validation Accuracy per Epoch', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    ax3.set_xticks(epochs)

    # --------------------- plot 4: Loss Distribution per Epoch ---------------------
    ax4 = axes[1, 1]

    # create box plot of losses per epoch
    epoch_losses = []
    for epoch in range(num_epochs):
        start_idx = epoch * batches_per_epoch
        end_idx = (epoch + 1) * batches_per_epoch
        epoch_losses.append(batch_losses[start_idx:end_idx])

    bp = ax4.boxplot(epoch_losses, labels=epochs, patch_artist=True)

    # customize box plot colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Loss Distribution per Epoch', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # adjust layout
    plt.tight_layout()

    # save figure with framework name in filename
    save_path = os.path.join(save_dir, f'training_curves_{framework}_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    pdf_path = os.path.join(save_dir, f'training_curves_{framework}_{timestamp}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Training curves saved to: {pdf_path}")

    plt.close()

    return save_path


def plot_loss_curve_only(logging_dict, save_dir='plots', framework='numpy'):
    """
    Creates a focused plot of just the training loss curve.
    Useful for the assignment requirement to show training loss curve.

    Args:
        logging_dict: Dictionary containing 'batch_loss' key
        save_dir: Directory to save plots (default: 'plots')
        framework: Framework used ('numpy' or 'pytorch', default: 'numpy')
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    batch_losses = logging_dict['batch_loss']
    train_accuracies_per_epoch = logging_dict['train_accuracy_per_epoch']
    num_epochs = len(train_accuracies_per_epoch)
    batches_per_epoch = len(batch_losses) // num_epochs if num_epochs > 0 else 0
    
    # format framework name for display
    framework_display = framework.upper()

    # create figure
    plt.figure(figsize=(10, 6))

    # plot raw loss
    plt.plot(batch_losses, alpha=0.5, linewidth=0.8, color='gray', label='Batch Loss')

    # add smoothed curve
    if len(batch_losses) > 50:
        window_size = min(100, len(batch_losses) // 10)
        smoothed_loss = np.convolve(batch_losses,
                                    np.ones(window_size)/window_size,
                                    mode='valid')
        plt.plot(range(window_size//2, len(smoothed_loss) + window_size//2),
                smoothed_loss, 'b-', linewidth=2.5, label=f'Smoothed Loss')

    # add epoch markers
    for epoch in range(1, num_epochs):
        plt.axvline(x=epoch * batches_per_epoch, color='red', linestyle='--',
                   alpha=0.3, linewidth=1.5)

    plt.xlabel('Training Iteration (Batch)', fontsize=13)
    plt.ylabel('Cross-Entropy Loss', fontsize=13)
    plt.title(f'Training Loss Curve ({framework_display})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # save with framework name in filename
    save_path = os.path.join(save_dir, f'loss_curve_{framework}_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to: {save_path}")

    plt.close()

    return save_path


def print_summary_statistics(logging_dict, val_accuracies, test_accuracy):
    """
    Prints summary statistics of the training process.

    Args:
        logging_dict: Dictionary containing training metrics
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy (scalar)
    """

    batch_losses = logging_dict['batch_loss']
    batch_train_accuracies = logging_dict['batch_train_accuracy']
    train_accuracies_per_epoch = logging_dict['train_accuracy_per_epoch']

    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)

    print(f"\nLoss Statistics:")
    print(f"  Initial Loss:        {batch_losses[0]:.4f}")
    print(f"  Final Loss:          {batch_losses[-1]:.4f}")
    print(f"  Minimum Loss:        {min(batch_losses):.4f}")
    print(f"  Average Loss:        {np.mean(batch_losses):.4f}")
    print(f"  Loss Std Dev:        {np.std(batch_losses):.4f}")

    print(f"\nAccuracy Statistics:")
    print(f"  Initial Train Acc:   {batch_train_accuracies[0]:.4f}")
    print(f"  Final Train Acc:     {train_accuracies_per_epoch[-1]:.4f}")
    print(f"  Best Val Acc:        {max(val_accuracies):.4f} (Epoch {np.argmax(val_accuracies) + 1})")
    print(f"  Final Val Acc:       {val_accuracies[-1]:.4f}")
    print(f"  Test Acc:            {test_accuracy:.4f}")

    print(f"\nTraining Progress:")
    improvement = train_accuracies_per_epoch[-1] - batch_train_accuracies[0]
    print(f"  Accuracy Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    print(f"  Total Batches:        {len(batch_losses)}")
    print(f"  Total Epochs:         {len(train_accuracies_per_epoch)}")

    print("="*60 + "\n")


def save_metrics_to_file(logging_dict, val_accuracies, test_accuracy,
                         hyperparameters, save_dir='plots', framework='numpy'):
    """
    Saves all metrics and hyperparameters to a text file for the assignment report.

    Args:
        logging_dict: Dictionary containing training metrics
        val_accuracies: List of validation accuracies per epoch
        test_accuracy: Final test accuracy
        hyperparameters: Dictionary of hyperparameters used
        save_dir: Directory to save file
        framework: Framework used ('numpy' or 'pytorch', default: 'numpy')
    """

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'training_metrics_{framework}_{timestamp}.txt')
    
    framework_display = framework.upper()

    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"MLP TRAINING REPORT - CIFAR-10 ({framework_display})\n")
        f.write("="*70 + "\n\n")

        f.write("HYPERPARAMETERS:\n")
        f.write("-"*70 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key:20s}: {value}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("FINAL RESULTS:\n")
        f.write("="*70 + "\n")
        f.write(f"  Best Validation Accuracy: {max(val_accuracies):.4f}\n")
        f.write(f"  Test Accuracy:            {test_accuracy:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("ACCURACY PER EPOCH:\n")
        f.write("="*70 + "\n")
        f.write(f"{'Epoch':<10}{'Train Acc':<15}{'Val Acc':<15}\n")
        f.write("-"*70 + "\n")

        for i, (train_acc, val_acc) in enumerate(zip(logging_dict['train_accuracy_per_epoch'],
                                                      val_accuracies), 1):
            marker = " *" if val_acc == max(val_accuracies) else ""
            f.write(f"{i:<10}{train_acc:<15.4f}{val_acc:<15.4f}{marker}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Training metrics saved to: {save_path}")
    return save_path

