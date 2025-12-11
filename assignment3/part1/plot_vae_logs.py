import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir):
    """Load scalar data from tensorboard event files."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_time': [e.wall_time for e in events]
        }
    return scalars


def get_step_to_epoch_mapping(scalars):
    """Get step to epoch mapping from 'epoch' scalar if available."""
    epoch_keys = [k for k in scalars.keys() if k.lower() == 'epoch']
    if epoch_keys:
        epoch_data = scalars[epoch_keys[0]]
        # Create a dict mapping step -> epoch
        return dict(zip(epoch_data['steps'], epoch_data['values']))
    return None


def steps_to_epochs(steps, step_to_epoch_map, fallback_steps_per_epoch=422):
    """Convert steps to epochs using mapping or fallback calculation."""
    if step_to_epoch_map:
        return [step_to_epoch_map.get(s, s / fallback_steps_per_epoch) for s in steps]
    else:
        return [s / fallback_steps_per_epoch for s in steps]


def plot_bpd_metrics(base_path):
    """Plot train_bpd and val_bpd for each experiment."""
    lightning_logs_path = os.path.join(base_path, 'lightning_logs')
    
    # Find all version directories
    versions = [d for d in os.listdir(lightning_logs_path) 
                if d.startswith('version_') and os.path.isdir(os.path.join(lightning_logs_path, d))]
    
    for version in sorted(versions):
        version_path = os.path.join(lightning_logs_path, version)
        print(f"Processing {version}...")
        
        # Load tensorboard data
        try:
            scalars = load_tensorboard_logs(version_path)
        except Exception as e:
            print(f"  Error loading {version}: {e}")
            continue
        
        # Print available tags
        print(f"  Available tags: {list(scalars.keys())}")
        
        # Get step to epoch mapping from 'epoch' scalar if available
        step_to_epoch = get_step_to_epoch_mapping(scalars)
        if step_to_epoch:
            print(f"  Using 'epoch' scalar for step-to-epoch mapping")
        else:
            print(f"  No 'epoch' scalar found, using fallback (422 steps/epoch)")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Track best scores (lower BPD is better)
        best_train_bpd = None
        best_val_bpd = None
        
        # Plot train_bpd if available
        train_keys = [k for k in scalars.keys() if 'train' in k.lower() and 'bpd' in k.lower()]
        for key in train_keys:
            steps = scalars[key]['steps']
            values = scalars[key]['values']
            epochs = steps_to_epochs(steps, step_to_epoch)
            ax.plot(epochs, values, label=key, marker='o', markersize=2)
            if values:
                best_train_bpd = min(values)
        
        # Plot val_bpd if available
        val_keys = [k for k in scalars.keys() if 'val' in k.lower() and 'bpd' in k.lower()]
        for key in val_keys:
            steps = scalars[key]['steps']
            values = scalars[key]['values']
            epochs = steps_to_epochs(steps, step_to_epoch)
            ax.plot(epochs, values, label=key, marker='s', markersize=2)
            if values:
                best_val_bpd = min(values)
        
        # If no bpd keys found, try to plot any available metrics
        if not train_keys and not val_keys:
            print(f"  No bpd metrics found. Plotting all available scalars...")
            for key, data in scalars.items():
                epochs = steps_to_epochs(data['steps'], step_to_epoch)
                ax.plot(epochs, data['values'], label=key, marker='o', markersize=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BPD (bits per dimension)')
        
        # Build title with best scores
        title = 'VAE BPD per epoch'
        if best_train_bpd is not None or best_val_bpd is not None:
            title += '\n'
            if best_train_bpd is not None:
                title += f'Best Train BPD: {best_train_bpd:.4f}'
            if best_val_bpd is not None:
                if best_train_bpd is not None:
                    title += ' | '
                title += f'Best Val BPD: {best_val_bpd:.4f}'
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print best scores to console
        if best_train_bpd is not None:
            print(f"  Best Train BPD: {best_train_bpd:.4f}")
        if best_val_bpd is not None:
            print(f"  Best Val BPD: {best_val_bpd:.4f}")
        
        # Save figure
        output_path = os.path.join(version_path, f'{version}_bpd_plot.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output_path}")
        plt.close()
    
    print("\nDone!")

if __name__ == '__main__':
    # Update this path to your VAE_logs folder
    base_path = '/Users/maybenzion/MSCAI/DL1/uvadlc_practicals_2025/assignment3/VAE_logs'
    plot_bpd_metrics(base_path)