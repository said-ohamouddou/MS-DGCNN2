"""Plot helpers for ablation figures."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional


plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def create_kscale_heatmap(
    results: Dict[Tuple[int, int], Dict[str, Tuple[float, float]]],
    k_local_values: List[int],
    k_branch_values: List[int],
    metric: str = 'accuracy',
    title: str = 'K-Scale Sensitivity Analysis',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdYlGn'
) -> plt.Figure:
    """
    Create a publication-quality heatmap for k-scale sensitivity analysis.
    
    Args:
        results: Dict mapping (k_local, k_branch) -> {metric: (mean, std)}
        k_local_values: List of k_local values tested
        k_branch_values: List of k_branch values tested
        metric: Metric to visualize
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        cmap: Colormap name
    
    Returns:
        matplotlib Figure object
    """
    # Create data matrix
    data = np.zeros((len(k_local_values), len(k_branch_values)))
    std_data = np.zeros_like(data)
    
    for i, k_local in enumerate(k_local_values):
        for j, k_branch in enumerate(k_branch_values):
            key = (k_local, k_branch)
            if key in results and metric in results[key]:
                data[i, j] = results[key][metric][0]
                std_data[i, j] = results[key][metric][1]
            else:
                data[i, j] = np.nan
                std_data[i, j] = np.nan
    
    # Find best value
    best_idx = np.unravel_index(np.nanargmax(data), data.shape)
    best_val = data[best_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', 
                       fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(len(k_branch_values)))
    ax.set_yticks(np.arange(len(k_local_values)))
    ax.set_xticklabels(k_branch_values, fontweight='bold')
    ax.set_yticklabels(k_local_values, fontweight='bold')
    
    # Labels
    ax.set_xlabel(r'$k_{branch}$', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$k_{local}$', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(k_local_values)):
        for j in range(len(k_branch_values)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                std = std_data[i, j]
                
                # Determine text color based on background
                bg_color = im.cmap(im.norm(val))
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                
                # Bold for best value
                weight = 'bold' if (i, j) == best_idx else 'normal'
                fontsize = 11 if (i, j) == best_idx else 10
                
                text = f'{val:.1f}\n±{std:.1f}'
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontsize=fontsize, fontweight=weight)
    
    # Highlight best cell
    rect = plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                         fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(rect)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved heatmap to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_bar_comparison(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    variants: List[str],
    metrics: List[str] = None,
    title: str = 'Model Comparison',
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a grouped bar chart comparing multiple variants.
    
    Args:
        results: Dict mapping variant_name -> {metric: (mean, std)}
        variants: List of variant names to include
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'f1_macro']
    
    metric_labels = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Acc',
        'f1_macro': 'F1 Macro',
        'kappa': 'Kappa',
        'mcc': 'MCC'
    }
    
    # Prepare data
    x = np.arange(len(metrics))
    width = 0.8 / len(variants)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
    
    for i, variant in enumerate(variants):
        if variant not in results:
            continue
        
        means = [results[variant].get(m, (0, 0))[0] for m in metrics]
        stds = [results[variant].get(m, (0, 0))[1] for m in metrics]
        
        offset = (i - len(variants) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=variant, 
                     color=colors[i], yerr=stds, capsize=3)
    
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics], 
                       fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved bar chart to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == '__main__':
    # Test heatmap
    k_local_values = [3, 5, 10, 15]
    k_branch_values = [10, 20, 30, 40]
    
    # Generate dummy results
    results = {}
    np.random.seed(42)
    for kl in k_local_values:
        for kb in k_branch_values:
            base = 80 + 5 * np.log(kl * kb / 30)
            results[(kl, kb)] = {
                'accuracy': (base + np.random.randn() * 2, np.random.rand() * 1.5)
            }
    
    create_kscale_heatmap(
        results, k_local_values, k_branch_values,
        metric='accuracy',
        title='K-Scale Sensitivity Analysis',
        show=True
    )
