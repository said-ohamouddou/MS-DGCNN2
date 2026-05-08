"""Experiment 3: Gaussian noise sweep across encoding variants."""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, TreeSpeciesDatasetCV, evaluate
)

import matplotlib.pyplot as plt
import pandas as pd
import logging
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    'raw_only': {
        'use_normalized_features': False,
        'local_use_normalized': False,
        'branch_feature_mode': 'full',
        'label': '(a) Raw-only'
    },
    'hybrid_everywhere': {
        'use_normalized_features': True,
        'local_use_normalized': True,
        'branch_feature_mode': 'full',
        'label': '(b) Hybrid-everywhere'
    },
    'default': {
        'use_normalized_features': True,
        'local_use_normalized': False,
        'branch_feature_mode': 'full',
        'label': '(c) Default asymmetric'
    },
    'reversed': {
        'use_normalized_features': False,
        'local_use_normalized': True,
        'branch_feature_mode': 'full',
        'label': '(d) Reversed asymmetric'
    }
}

NOISE_LEVELS = [0.5, 1, 2, 5, 10, 20, 50, 75, 100, 150]


def compute_snr_stats(data_path, fold, noise_sigma, k1=5, k2=20, 
                      num_points=1024, seed=42, k_folds=5):
    """Compute empirical SNR at k1 and k2 scales."""
    from scipy.spatial.distance import cdist
    
    dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val', num_points=num_points,
        augment=False, seed=seed, k_folds=k_folds,
        noise_sigma=noise_sigma
    )
    
    s1_distances = []
    s2_distances = []
    
    for idx in range(min(50, len(dataset))):
        points, _ = dataset[idx]
        points = points.numpy()
        
        dists = cdist(points, points)
        
        for i in range(len(points)):
            sorted_dists = np.sort(dists[i])[1:]
            
            if len(sorted_dists) >= k1:
                s1_distances.append(np.mean(sorted_dists[:k1]))
            
            if len(sorted_dists) >= k2:
                s2_distances.append(np.mean(sorted_dists[:k2]))
    
    s1_mean = np.mean(s1_distances)
    s2_mean = np.mean(s2_distances)
    
    sigma_normalized = noise_sigma / 1000.0 if noise_sigma > 0 else 1e-10
    
    snr1 = s1_mean / sigma_normalized if noise_sigma > 0 else float('inf')
    snr2 = s2_mean / sigma_normalized if noise_sigma > 0 else float('inf')
    
    return {
        's1_mean': s1_mean,
        's2_mean': s2_mean,
        'snr1': snr1,
        'snr2': snr2
    }


def load_previous_results(results_path):
    """Load previous results from JSON file."""
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to proper types
    results = {}
    for variant, sigmas in data.items():
        results[variant] = {}
        for sigma_str, folds in sigmas.items():
            sigma = float(sigma_str)
            # Handle integer sigma values
            if sigma == int(sigma):
                sigma = int(sigma)
            results[variant][sigma] = {}
            for fold_str, metrics in folds.items():
                fold = int(fold_str)
                # Convert lists back to numpy arrays where needed
                results[variant][sigma][fold] = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in metrics.items()
                }
    
    return results


def load_previous_snr_stats(snr_path):
    """Load previous SNR stats from JSON file."""
    if not snr_path.exists():
        return {}
    
    with open(snr_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to proper types
    stats = {}
    for sigma_str, values in data.items():
        sigma = float(sigma_str)
        if sigma == int(sigma):
            sigma = int(sigma)
        stats[sigma] = values
    
    return stats


def load_checkpoint_and_evaluate(ckpt_path, model, val_loader, device):
    """Load checkpoint and evaluate to get metrics."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # If metrics are stored in checkpoint, use them
    if 'metrics' in checkpoint and checkpoint['metrics']:
        return checkpoint['metrics']
    
    # Otherwise, re-evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, device)
    return metrics


def run_experiment(args):
    """Run Experiment 3: Synthetic Noise Sweep with resume capability."""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Use existing output directory if resuming, otherwise create new
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            raise ValueError(f"Resume directory not found: {args.resume_dir}")
        logger.info(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f'experiment3_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Load previous results if resuming
    previous_results = load_previous_results(output_dir / 'all_results.json')
    previous_snr = load_previous_snr_stats(output_dir / 'snr_stats.json')
    
    if previous_results:
        logger.info(f"Loaded previous results with {len(list(previous_results.values())[0])} noise levels")
    
    # Save updated config
    config = {
        'experiment': 'Experiment 3: Synthetic Noise Sweep (Extended)',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'noise_levels': NOISE_LEVELS,
        'seed': args.seed,
        'variants': list(VARIANTS.keys()),
        'resumed': args.resume_dir is not None
    }
    save_config(config, output_dir / 'config.json')
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    # Initialize results storage
    all_results = {v: {sigma: {} for sigma in NOISE_LEVELS} for v in VARIANTS.keys()}
    
    # Copy previous results
    if previous_results:
        for variant in VARIANTS.keys():
            if variant in previous_results:
                for sigma in previous_results[variant]:
                    if sigma in all_results[variant]:
                        all_results[variant][sigma] = previous_results[variant][sigma]
    
    # Compute SNR stats (use cached if available)
    snr_stats = {}
    logger.info("Computing/loading SNR statistics at each noise level...")
    for sigma in NOISE_LEVELS:
        if sigma in previous_snr:
            snr_stats[sigma] = previous_snr[sigma]
            logger.info(f"σ={sigma}mm: (cached) SNR₂={snr_stats[sigma]['snr2']:.2f}")
        else:
            stats = compute_snr_stats(
                args.data_path, fold=0, noise_sigma=sigma,
                k1=5, k2=20, num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
            )
            snr_stats[sigma] = stats
            logger.info(f"σ={sigma}mm: s̄₁={stats['s1_mean']:.4f}, s̄₂={stats['s2_mean']:.4f}, "
                       f"SNR₁={stats['snr1']:.2f}, SNR₂={stats['snr2']:.2f}")
    
    # Count runs needed
    runs_needed = 0
    runs_skipped = 0
    for variant_name in VARIANTS.keys():
        for noise_sigma in NOISE_LEVELS:
            for fold in range(args.k_folds):
                ckpt_path = checkpoints_dir / f'{variant_name}_sigma{noise_sigma}_fold{fold}.pth'
                if ckpt_path.exists():
                    runs_skipped += 1
                else:
                    runs_needed += 1
    
    logger.info(f"Runs to skip (checkpoint exists): {runs_skipped}")
    logger.info(f"Runs to train: {runs_needed}")
    
    run_count = 0
    
    for variant_name, variant_config in VARIANTS.items():
        for noise_sigma in NOISE_LEVELS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Variant: {variant_config['label']}, Noise: σ={noise_sigma}mm")
            logger.info(f"{'='*60}")
            
            for fold in range(args.k_folds):
                ckpt_path = checkpoints_dir / f'{variant_name}_sigma{noise_sigma}_fold{fold}.pth'
                
                # Check if checkpoint exists
                if ckpt_path.exists():
                    # Load existing checkpoint and get metrics
                    if noise_sigma in all_results[variant_name] and fold in all_results[variant_name][noise_sigma]:
                        logger.info(f"Fold {fold}: Using cached results (OA={all_results[variant_name][noise_sigma][fold]['oa']:.2f}%)")
                        continue
                    
                    logger.info(f"Fold {fold}: Loading checkpoint and evaluating...")
                    
                    # Create model and load checkpoint
                    model = create_model(
                        num_classes=num_classes,
                        k_scales=[5, 20, 30],
                        use_multiscale=True,
                        use_normalized_features=variant_config['use_normalized_features'],
                        local_use_normalized=variant_config['local_use_normalized'],
                        branch_feature_mode=variant_config['branch_feature_mode'],
                        fusion_type='concat_conv',
                        emb_dims=1024,
                        dropout=0.5
                    ).to(device)
                    
                    _, val_loader, _, _ = get_cv_dataloaders(
                        args.data_path, fold=fold, batch_size=args.batch_size,
                        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                        noise_sigma=noise_sigma
                    )
                    
                    metrics = load_checkpoint_and_evaluate(ckpt_path, model, val_loader, device)
                    all_results[variant_name][noise_sigma][fold] = metrics
                    logger.info(f"Fold {fold}: OA={metrics['oa']:.2f}%, mAcc={metrics['macc']:.2f}%")
                    continue
                
                # Train new model
                run_count += 1
                logger.info(f"\n--- Training Run {run_count}/{runs_needed}: Fold {fold + 1}/{args.k_folds} ---")
                
                # Use noise-independent seeding for model initialization (matches Exp1)
                set_seed(args.seed + fold)
                
                train_loader, val_loader, num_classes, _ = get_cv_dataloaders(
                    args.data_path, fold=fold, batch_size=args.batch_size,
                    num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                    noise_sigma=noise_sigma
                )
                
                model = create_model(
                    num_classes=num_classes,
                    k_scales=[5, 20, 30],
                    use_multiscale=True,
                    use_normalized_features=variant_config['use_normalized_features'],
                    local_use_normalized=variant_config['local_use_normalized'],
                    branch_feature_mode=variant_config['branch_feature_mode'],
                    fusion_type='concat_conv',
                    emb_dims=1024,
                    dropout=0.5
                ).to(device)
                
                best_metrics, model = train_model(
                    model, train_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    save_path=str(ckpt_path), patience=args.patience
                )
                
                all_results[variant_name][noise_sigma][fold] = best_metrics
                logger.info(f"OA: {best_metrics['oa']:.2f}%, mAcc: {best_metrics['macc']:.2f}%")
                
                # Save intermediate results
                save_results_json(all_results, output_dir / 'all_results.json')
    
    # Save final results
    save_results_json(all_results, output_dir / 'all_results.json')
    save_snr_stats(snr_stats, output_dir / 'snr_stats.json')
    
    # Generate visualizations
    generate_visualizations(all_results, snr_stats, class_names, output_dir)

    logger.info(f"\nExperiment 3 completed. Results saved to {output_dir}")


def save_results_json(results, path):
    """Save results to JSON."""
    serializable = {}
    for variant, sigmas in results.items():
        serializable[variant] = {}
        for sigma, folds in sigmas.items():
            serializable[variant][str(sigma)] = {}
            for fold, metrics in folds.items():
                serializable[variant][str(sigma)][str(fold)] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def save_snr_stats(stats, path):
    """Save SNR statistics to JSON."""
    serializable = {str(k): v for k, v in stats.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def generate_visualizations(results, snr_stats, class_names, output_dir):
    """Generate visualizations for Experiment 3."""
    
    from plot_style import set_publication_style, get_variant_colors, add_panel_label, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    variant_names = list(VARIANTS.keys())
    variant_labels = [VARIANTS[v]['label'] for v in variant_names]
    colors = get_variant_colors(len(variant_names))
    
    # Get noise levels that have results
    available_sigmas = []
    for sigma in NOISE_LEVELS:
        if sigma in results[variant_names[0]] and results[variant_names[0]][sigma]:
            available_sigmas.append(sigma)
    
    # 1. Accuracy vs SNR curve
    fig, ax = plt.subplots(figsize=(12, 7))
    
    snr2_values = []
    for sigma in available_sigmas:
        snr2 = snr_stats[sigma]['snr2']
        if snr2 == float('inf'):
            snr2 = 1000
        snr2_values.append(snr2)
    
    for variant, color in zip(variant_names, colors):
        oa_means = []
        oa_stds = []
        for sigma in available_sigmas:
            folds = results[variant][sigma]
            oas = [folds[f]['oa'] for f in folds]
            oa_means.append(np.mean(oas))
            oa_stds.append(np.std(oas))
        
        ax.plot(snr2_values, oa_means, 'o-', label=VARIANTS[variant]['label'],
                color=color, linewidth=2.5, markersize=10)
        ax.fill_between(snr2_values,
                        np.array(oa_means) - np.array(oa_stds),
                        np.array(oa_means) + np.array(oa_stds),
                        alpha=0.2, color=color)
    
    # Theoretical crossover line
    crossover_snr = 1.0 / 0.816  # ≈ 1.22
    ax.axvline(x=crossover_snr, color='#C73E1D', linestyle='--', linewidth=3.0,
               label=r'Theoretical crossover (SNR$_2$ $\approx$ 1.22)')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'SNR$_2$ (log scale)', fontsize=21, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=21, fontweight='bold')
    ax.set_title('Accuracy vs. Signal-to-Noise Ratio', fontsize=24, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.4, linewidth=1.0)
    ax.tick_params(axis='both', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'accuracy_vs_snr')
    plt.close()
    
    # 2. Zoomed inset (SNR2 ∈ [0.5, 5])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for variant, color in zip(variant_names, colors):
        oa_means = []
        for sigma in available_sigmas:
            folds = results[variant][sigma]
            oas = [folds[f]['oa'] for f in folds]
            oa_means.append(np.mean(oas))
        
        ax.plot(snr2_values, oa_means, 'o-', label=VARIANTS[variant]['label'],
                color=color, linewidth=2.5, markersize=10)
    
    ax.axvline(x=crossover_snr, color='#C73E1D', linestyle='--', linewidth=3.0,
               label=r'Theoretical crossover (SNR$_2$ $\approx$ 1.22)')
    
    ax.set_xscale('log')
    ax.set_xlim(0.3, 10)
    ax.set_xlabel(r'SNR$_2$ (log scale)', fontsize=21, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=21, fontweight='bold')
    ax.set_title(r'Zoomed View: SNR$_2$ $\in$ [0.5, 5]', fontsize=24, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=16, framealpha=0.9)
    ax.grid(True, alpha=0.4, linewidth=1.0)
    ax.tick_params(axis='both', labelsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'accuracy_vs_snr_zoomed')
    plt.close()
    
    # 3. Full LaTeX table with ALL noise levels and bold best values
    # Compute OA means for each variant at each noise level
    oa_data = {}
    for variant in variant_names:
        oa_data[variant] = {}
        for sigma in available_sigmas:
            if sigma in results[variant] and results[variant][sigma]:
                folds = results[variant][sigma]
                oas = [folds[f]['oa'] for f in folds]
                oa_data[variant][sigma] = {'mean': np.mean(oas), 'std': np.std(oas)}
    
    # Find best variant at each noise level
    best_at_sigma = {}
    for sigma in available_sigmas:
        best_oa = -1
        best_variant = None
        for variant in variant_names:
            if sigma in oa_data[variant]:
                if oa_data[variant][sigma]['mean'] > best_oa:
                    best_oa = oa_data[variant][sigma]['mean']
                    best_variant = variant
        best_at_sigma[sigma] = best_variant
    
    # Generate LaTeX table
    latex_lines = []
    # Header
    col_spec = "l" + "c" * len(available_sigmas)
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\toprule")
    header = "Variant & " + " & ".join([f"$\\sigma$={s}" for s in available_sigmas]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")
    
    # Data rows
    for variant in variant_names:
        row_cells = [VARIANTS[variant]['label']]
        for sigma in available_sigmas:
            if sigma in oa_data[variant]:
                mean = oa_data[variant][sigma]['mean']
                std = oa_data[variant][sigma]['std']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                # Bold if best at this noise level
                if best_at_sigma[sigma] == variant:
                    cell = f"\\textbf{{{cell}}}"
                row_cells.append(cell)
            else:
                row_cells.append("--")
        latex_lines.append(" & ".join(row_cells) + r" \\")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    latex_full = "\n".join(latex_lines)
    with open(tables_dir / 'full_results_table.tex', 'w') as f:
        f.write(latex_full)

    latex_wrapped_lines = []
    latex_wrapped_lines.append(r"\begin{table}[htbp]")
    latex_wrapped_lines.append(r"\centering")
    latex_wrapped_lines.append(r"\caption{Experiment 3: Overall Accuracy (\%) at All Noise Levels}")
    latex_wrapped_lines.append(r"\label{tab:exp3_full}")
    latex_wrapped_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_wrapped_lines.extend(latex_lines)
    latex_wrapped_lines.append(r"}")
    latex_wrapped_lines.append(r"\end{table}")

    with open(tables_dir / 'full_results_table_wrapped.tex', 'w') as f:
        f.write("\n".join(latex_wrapped_lines))
    
    # Also save CSV
    table_data = []
    for variant in variant_names:
        row = [VARIANTS[variant]['label']]
        for sigma in available_sigmas:
            if sigma in oa_data[variant]:
                row.append(f"{oa_data[variant][sigma]['mean']:.2f} ± {oa_data[variant][sigma]['std']:.2f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    columns = ['Variant'] + [f'σ={s}mm' for s in available_sigmas]
    df = pd.DataFrame(table_data, columns=columns)
    df.to_csv(tables_dir / 'full_results_table.csv', index=False)
    
    # 4. Per-scale SNR table
    snr_table = []
    for sigma in available_sigmas:
        stats = snr_stats[sigma]
        snr1_str = f"{stats['snr1']:.2f}" if stats['snr1'] != float('inf') else "∞"
        snr2_str = f"{stats['snr2']:.2f}" if stats['snr2'] != float('inf') else "∞"
        snr_table.append([
            sigma,
            f"{stats['s1_mean']:.4f}",
            f"{stats['s2_mean']:.4f}",
            snr1_str,
            snr2_str
        ])
    
    columns = ['σ (mm)', 's̄₁', 's̄₂', 'SNR₁', 'SNR₂']
    df_snr = pd.DataFrame(snr_table, columns=columns)
    df_snr.to_csv(tables_dir / 'snr_table.csv', index=False)
    
    logger.info(f"Visualizations saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Synthetic Noise Sweep (Resume/Extend)')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/extend')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
