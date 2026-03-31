import numpy as np
import matplotlib.pyplot as plt



def smooth_curve(values: list[float], window: int = 3) -> np.ndarray:
    """
    Applies a simple moving average to smooth out noise in the curves.
    """
    if window <= 1:
        return np.array(values)
    
    # Pad the start to maintain the same length as the input
    padded_values = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded_values, np.ones(window)/window, mode='valid')

def plot_training_results(
    history: dict[str, list[dict[str, float]]], 
    save_path: str | None = None,
    window_size: int = 5
) -> None:
    """
    Plots Training vs. Validation metrics using only smoothed loss curves
    for maximum clarity in the final report.
    """
    epochs = np.arange(1, len(history["train"]) + 1)
    
    # Extract and Smooth Loss
    train_loss_raw = [step["loss"] for step in history["train"]]
    val_loss_raw = [step["loss"] for step in history["val"]]
    
    train_loss_smooth = smooth_curve(train_loss_raw, window=window_size)
    val_loss_smooth = smooth_curve(val_loss_raw, window=window_size)
    
    # Extract Metrics
    train_auroc = [step["auroc"] for step in history["train"]]
    val_auroc = [step["auroc"] for step in history["val"]]
    
    train_auprc = [step["auprc"] for step in history["train"]]
    val_auprc = [step["auprc"] for step in history["val"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Smoothed Loss Plot ---
    axes[0].plot(epochs, train_loss_smooth, color='royalblue', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, val_loss_smooth, color='darkorange', linewidth=2, label='Val Loss')
    
    axes[0].set_title(f'Binary Cross-Entropy Loss (SMA-{window_size})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- AuROC Plot ---
    axes[1].plot(epochs, train_auroc, color='forestgreen', linestyle='--', label='Train AuROC')
    axes[1].plot(epochs, val_auroc, color='forestgreen', linewidth=2, label='Val AuROC')
    axes[1].set_title('Area Under ROC Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AuROC')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # --- AuPRC Plot ---
    axes[2].plot(epochs, train_auprc, color='crimson', linestyle='--', label='Train AuPRC')
    axes[2].plot(epochs, val_auprc, color='crimson', linewidth=2, label='Val AuPRC')
    axes[2].set_title('Area Under PR Curve')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AuPRC')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cleaned plots saved to: {save_path}")
    
    plt.show()


def report_test_results(
    results: dict[str, float], 
    model_name: str = "Transformer (Task 2.3b)",
    save_path: str | None = None
) -> None:
    """
    Prints a formatted report of the test set performance and generates 
    a LaTeX table row. Optionally plots and saves the metrics.
    """
    print("\n" + "="*45)
    print(f" FINAL PERFORMANCE REPORT: {model_name}")
    print("="*45)
    
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 28)
    print(f"{'AuROC':<15} | {results['auroc']:.4f}")
    print(f"{'AuPRC':<15} | {results['auprc']:.4f}")
    print(f"{'Test Loss':<15} | {results['loss']:.4f}")
    print("-" * 45)

    # Table Generation
    latex_row = (
        f"{model_name} & {results['auroc']:.4f} & "
        f"{results['auprc']:.4f} & {results['loss']:.4f} \\\\"
    )
    
    print(latex_row)
    print("="*45 + "\n")

    # --- Metrics Plotting ---
    metrics = ["AuROC", "AuPRC", "Loss"]
    values = [results['auroc'], results['auprc'], results['loss']]
    colors = ['forestgreen', 'crimson', 'royalblue']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', 
                 ha='center', va='bottom', fontweight='bold')

    plt.title(f'Final Test Metrics: {model_name}', fontsize=14, pad=15)
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, max(values) * 1.15) # Leave space for labels
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Test metrics plot saved to: {save_path}")
    
    plt.show()
