"""Experiment 2: upper-canopy density dropout across retention rates."""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    ExperimentResults, plot_degradation_curves, save_results_table,
    generate_cv_splits, TreeSpeciesDatasetCV, evaluate
)

import matplotlib.pyplot as plt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    'raw_only': {
        'use_normalized_features': False,
        'local_use_normalized': False,
        'label': '(a) Raw-only'
    },
    'hybrid_everywhere': {
        'use_normalized_features': True,
        'local_use_normalized': True,
        'label': '(b) Hybrid-everywhere'
    },
    'default': {
        'use_normalized_features': True,
        'local_use_normalized': False,
        'label': '(c) Default asymmetric'
    },
    'reversed': {
        'use_normalized_features': False,
        'local_use_normalized': True,
        'label': '(d) Reversed asymmetric'
    }
}

RETENTION_RATES = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]


def compute_k2_neighbor_distance(data_path, fold, retention_rate, k2=20, 
                                  num_points=1024, seed=42, k_folds=5):
    """Compute mean k2-neighbor distance for upper canopy points in test set."""
    from torch.utils.data import DataLoader
    
    dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val', num_points=num_points,
        augment=False, seed=seed, k_folds=k_folds,
        density_retention=retention_rate, density_seed=seed
    )
    
    distances = []
    
    for idx in range(min(100, len(dataset))):  # Sample 100 point clouds
        points, _ = dataset[idx]
        points = points.numpy()
        
        # Compute median height
        median_z = np.median(points[:, 2])
        upper_mask = points[:, 2] > median_z
        upper_points = points[upper_mask]
        
        if len(upper_points) < k2:
            continue
        
        # Compute pairwise distances for upper canopy
        from scipy.spatial.distance import cdist
        dists = cdist(upper_points, upper_points)
        
        # Get k2-nearest neighbor distances (excluding self)
        for i in range(len(upper_points)):
            sorted_dists = np.sort(dists[i])[1:k2+1]  # Exclude self (0)
            distances.append(np.mean(sorted_dists))
    
    return np.mean(distances), np.std(distances)


def run_experiment(args):
    """Run Experiment 2: Controlled Density Dropout."""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Handle resume mode
    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            raise ValueError(f"Resume directory not found: {args.resume_dir}")
        logger.info(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f'experiment2_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    config = {
        'experiment': 'Experiment 2: Controlled Density Dropout',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'retention_rates': RETENTION_RATES,
        'seed': args.seed,
        'variants': list(VARIANTS.keys())
    }
    save_config(config, output_dir / 'config.json')
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    # Results storage: {variant: {retention: {fold: metrics}}}
    all_results = {v: {r: {} for r in RETENTION_RATES} for v in VARIANTS.keys()}
    
    # Compute k2-neighbor distances at each retention rate
    k2_distances = {}
    logger.info("Computing k2-neighbor distances at each retention rate...")
    for rate in RETENTION_RATES:
        mean_dist, std_dist = compute_k2_neighbor_distance(
            args.data_path, fold=0, retention_rate=rate,
            k2=20, num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
        )
        k2_distances[rate] = {'mean': mean_dist, 'std': std_dist}
        logger.info(f"Retention {rate*100:.0f}%: k2-distance = {mean_dist:.4f} ± {std_dist:.4f}")
    
    # Total runs: 4 variants × 6 rates × 5 folds = 120
    total_runs = len(VARIANTS) * len(RETENTION_RATES) * args.k_folds
    run_count = 0
    
    # Count checkpoints
    runs_to_skip = 0
    runs_to_train = 0
    for variant_name in VARIANTS.keys():
        for retention_rate in RETENTION_RATES:
            for fold in range(args.k_folds):
                ckpt_path = checkpoints_dir / f'{variant_name}_r{int(retention_rate*100)}_fold{fold}.pth'
                if ckpt_path.exists():
                    runs_to_skip += 1
                else:
                    runs_to_train += 1
    
    logger.info(f"Checkpoints found: {runs_to_skip}, to train: {runs_to_train}")
    
    for variant_name, variant_config in VARIANTS.items():
        for retention_rate in RETENTION_RATES:
            logger.info(f"\n{'='*60}")
            logger.info(f"Variant: {variant_config['label']}, Retention: {retention_rate*100:.0f}%")
            logger.info(f"{'='*60}")
            
            for fold in range(args.k_folds):
                run_count += 1
                ckpt_path = checkpoints_dir / f'{variant_name}_r{int(retention_rate*100)}_fold{fold}.pth'
                
                # Check if checkpoint exists
                if ckpt_path.exists():
                    logger.info(f"\n--- Run {run_count}/{total_runs}: Loading checkpoint ---")
                    
                    model = create_model(
                        num_classes=num_classes,
                        k_scales=[5, 20, 30],
                        use_multiscale=True,
                        use_normalized_features=variant_config['use_normalized_features'],
                        local_use_normalized=variant_config['local_use_normalized'],
                        fusion_type='concat_conv',
                        emb_dims=1024,
                        dropout=0.5
                    ).to(device)
                    
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'metrics' in checkpoint and checkpoint['metrics']:
                        best_metrics = checkpoint['metrics']
                    else:
                        # Re-evaluate
                        _, val_loader, _, _ = get_cv_dataloaders(
                            args.data_path, fold=fold, batch_size=args.batch_size,
                            num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                            density_retention=retention_rate, density_seed=args.seed + fold
                        )
                        import torch.nn as nn
                        criterion = nn.CrossEntropyLoss()
                        best_metrics = evaluate(model, val_loader, criterion, device)
                    
                    all_results[variant_name][retention_rate][fold] = best_metrics
                    logger.info(f"OA: {best_metrics['oa']:.2f}%, mAcc: {best_metrics['macc']:.2f}%")
                    continue
                
                # Train new model
                logger.info(f"\n--- Run {run_count}/{total_runs}: Training Fold {fold + 1}/{args.k_folds} ---")
                
                # Use retention-independent seeding for model initialization (matches Exp1)
                set_seed(args.seed + fold)
                
                # Retention-specific seed only for data dropout
                train_loader, val_loader, num_classes, _ = get_cv_dataloaders(
                    args.data_path, fold=fold, batch_size=args.batch_size,
                    num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                    density_retention=retention_rate, density_seed=args.seed + int(retention_rate * 100)
                )
                
                model = create_model(
                    num_classes=num_classes,
                    k_scales=[5, 20, 30],
                    use_multiscale=True,
                    use_normalized_features=variant_config['use_normalized_features'],
                    local_use_normalized=variant_config['local_use_normalized'],
                    fusion_type='concat_conv',
                    emb_dims=1024,
                    dropout=0.5
                ).to(device)
                
                best_metrics, model = train_model(
                    model, train_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    save_path=str(ckpt_path), patience=args.patience
                )
                
                all_results[variant_name][retention_rate][fold] = best_metrics
                logger.info(f"OA: {best_metrics['oa']:.2f}%, mAcc: {best_metrics['macc']:.2f}%")
    
    # Save results
    results_path = output_dir / 'all_results.json'
    save_results_json(all_results, results_path)
    
    # Generate visualizations
    generate_visualizations(all_results, k2_distances, class_names, output_dir)

    logger.info(f"\nExperiment 2 completed. Results saved to {output_dir}")


def save_results_json(results, path):
    """Save results to JSON with numpy array conversion."""
    import json
    
    serializable = {}
    for variant, rates in results.items():
        serializable[variant] = {}
        for rate, folds in rates.items():
            serializable[variant][str(rate)] = {}
            for fold, metrics in folds.items():
                serializable[variant][str(rate)][fold] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def generate_latex_tables(results, class_names, tables_dir):
    """Generate LaTeX tables for all metrics."""
    
    variant_names = list(VARIANTS.keys())
    variant_labels = [VARIANTS[v]['label'] for v in variant_names]
    
    # Table 1: OA and mAcc at all retention rates
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Experiment 2: Overall Accuracy and Mean Accuracy at Different Retention Rates}")
    lines.append(r"\label{tab:exp2_oa_macc}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l" + "cc" * len(RETENTION_RATES) + "}")
    lines.append(r"\toprule")
    
    # Header row
    header = "Variant"
    for rate in RETENTION_RATES:
        header += f" & \\multicolumn{{2}}{{c}}{{r={int(rate*100)}\\%}}"
    header += r" \\"
    lines.append(header)
    
    # Sub-header
    subheader = ""
    for _ in RETENTION_RATES:
        subheader += " & OA & mAcc"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")
    
    # Data rows
    for variant in variant_names:
        row = VARIANTS[variant]['label']
        for rate in RETENTION_RATES:
            folds = results[variant][rate]
            oas = [folds[f]['oa'] for f in folds]
            maccs = [folds[f]['macc'] for f in folds]
            oa_mean, oa_std = np.mean(oas), np.std(oas)
            macc_mean, macc_std = np.mean(maccs), np.std(maccs)
            row += f" & ${oa_mean:.1f} \\pm {oa_std:.1f}$ & ${macc_mean:.1f} \\pm {macc_std:.1f}$"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    with open(tables_dir / 'exp2_oa_macc.tex', 'w') as f:
        f.write("\n".join(lines))
    
    # Table 2: Per-class F1 at r=100% and r=5%
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Experiment 2: Per-Class F1 Scores at Full Density (r=100\%) and Extreme Sparsity (r=5\%)}")
    lines.append(r"\label{tab:exp2_f1}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l" + "cc" * len(variant_names) + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Species"
    for v in variant_names:
        header += f" & \\multicolumn{{2}}{{c}}{{{VARIANTS[v]['label']}}}"
    header += r" \\"
    lines.append(header)
    
    # Sub-header
    subheader = ""
    for _ in variant_names:
        subheader += " & 100\\% & 5\\%"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")
    
    # Data rows
    for i, class_name in enumerate(class_names):
        row = class_name
        for variant in variant_names:
            f1_100 = np.mean([results[variant][1.0][f]['f1'][i] for f in results[variant][1.0]])
            f1_5 = np.mean([results[variant][0.05][f]['f1'][i] for f in results[variant][0.05]])
            row += f" & {f1_100:.3f} & {f1_5:.3f}"
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    with open(tables_dir / 'exp2_f1.tex', 'w') as f:
        f.write("\n".join(lines))
    
    # Table 3: Delta OA (degradation from 100% to 5%)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Experiment 2: Accuracy Degradation ($\Delta$OA = OA(100\%) - OA(5\%))}")
    lines.append(r"\label{tab:exp2_delta}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $\Delta$OA (\%) & $\Delta$mAcc (\%) \\")
    lines.append(r"\midrule")
    
    for variant in variant_names:
        oa_100 = np.mean([results[variant][1.0][f]['oa'] for f in results[variant][1.0]])
        oa_5 = np.mean([results[variant][0.05][f]['oa'] for f in results[variant][0.05]])
        macc_100 = np.mean([results[variant][1.0][f]['macc'] for f in results[variant][1.0]])
        macc_5 = np.mean([results[variant][0.05][f]['macc'] for f in results[variant][0.05]])
        delta_oa = oa_100 - oa_5
        delta_macc = macc_100 - macc_5
        lines.append(f"{VARIANTS[variant]['label']} & {delta_oa:.2f} & {delta_macc:.2f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    with open(tables_dir / 'exp2_delta.tex', 'w') as f:
        f.write("\n".join(lines))
    
    logger.info(f"LaTeX tables saved to {tables_dir}")


def generate_visualizations(results, k2_distances, class_names, output_dir):
    """Generate visualizations for Experiment 2."""
    
    from plot_style import set_publication_style, get_variant_colors, add_panel_label, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    variant_names = list(VARIANTS.keys())
    variant_labels = [VARIANTS[v]['label'] for v in variant_names]
    colors = get_variant_colors(len(variant_names))
    
    # 1. Degradation curves
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    x_values = [r * 100 for r in RETENTION_RATES]
    
    for variant, color in zip(variant_names, colors):
        oa_means = []
        oa_stds = []
        for rate in RETENTION_RATES:
            folds = results[variant][rate]
            oas = [folds[f]['oa'] for f in folds]
            oa_means.append(np.mean(oas))
            oa_stds.append(np.std(oas))
        
        ax1.plot(x_values, oa_means, 'o-', label=VARIANTS[variant]['label'], 
                 color=color, linewidth=3.0, markersize=12)
        ax1.fill_between(x_values, 
                         np.array(oa_means) - np.array(oa_stds),
                         np.array(oa_means) + np.array(oa_stds),
                         alpha=0.2, color=color)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Retention Rate (%)', fontsize=21, fontweight='bold')
    ax1.set_ylabel('Overall Accuracy (%)', fontsize=21, fontweight='bold')
    ax1.set_title('Accuracy vs. Upper Canopy Retention Rate', fontsize=24, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=16, framealpha=0.9)
    ax1.grid(True, alpha=0.4, linewidth=1.0)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([f'{int(x)}%' for x in x_values], fontsize=18, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=18)
    ax1.spines['top'].set_visible(False)
    
    # Secondary y-axis for k2-distance
    ax2 = ax1.twinx()
    k2_means = [k2_distances[r]['mean'] for r in RETENTION_RATES]
    ax2.plot(x_values, k2_means, 's--', color='#666666', label=r'Mean k$_2$-distance', 
             alpha=0.7, linewidth=2.5, markersize=10)
    ax2.set_ylabel(r'Mean k$_2$-neighbor distance', color='#666666', fontsize=21, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#666666', labelsize=18)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'degradation_curves')
    plt.close()
    
    # 2. Per-class degradation table
    table_data = []
    for i, class_name in enumerate(class_names):
        row = [class_name]
        for variant in variant_names:
            # F1 at 100%
            f1_100 = np.mean([results[variant][1.0][f]['f1'][i] for f in results[variant][1.0]])
            # F1 at 5%
            f1_5 = np.mean([results[variant][0.05][f]['f1'][i] for f in results[variant][0.05]])
            row.extend([f'{f1_100:.3f}', f'{f1_5:.3f}'])
        table_data.append(row)
    
    columns = ['Species'] + [f'{VARIANTS[v]["label"]} r=100%' for v in variant_names] + \
              [f'{VARIANTS[v]["label"]} r=5%' for v in variant_names]
    # Reorder columns
    columns = ['Species']
    for v in variant_names:
        columns.extend([f'{VARIANTS[v]["label"]} 100%', f'{VARIANTS[v]["label"]} 5%'])
    
    # Rebuild table data with correct column order
    table_data = []
    for i, class_name in enumerate(class_names):
        row = [class_name]
        for variant in variant_names:
            f1_100 = np.mean([results[variant][1.0][f]['f1'][i] for f in results[variant][1.0]])
            f1_5 = np.mean([results[variant][0.05][f]['f1'][i] for f in results[variant][0.05]])
            row.extend([f'{f1_100:.3f}', f'{f1_5:.3f}'])
        table_data.append(row)
    
    df = pd.DataFrame(table_data, columns=columns)
    df.to_csv(tables_dir / 'per_class_degradation.csv', index=False)
    
    # Generate LaTeX tables
    generate_latex_tables(results, class_names, tables_dir)
    
    # 3. Delta-Accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_oas = []
    for variant in variant_names:
        oa_100 = np.mean([results[variant][1.0][f]['oa'] for f in results[variant][1.0]])
        oa_5 = np.mean([results[variant][0.05][f]['oa'] for f in results[variant][0.05]])
        delta_oas.append(oa_100 - oa_5)
    
    x = np.arange(len(variant_names))
    bars = ax.bar(x, delta_oas, color=colors, edgecolor='#333333', linewidth=2.0)
    
    ax.set_xlabel('Variant', fontsize=21, fontweight='bold')
    ax.set_ylabel(r'$\Delta$OA = OA(100%) - OA(5%)', fontsize=21, fontweight='bold')
    ax.set_title(r'Accuracy Degradation: 100% $\rightarrow$ 5% Retention', fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=15, ha='right', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, delta in zip(bars, delta_oas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{delta:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'delta_accuracy')
    plt.close()
    
    logger.info(f"Visualizations saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Controlled Density Dropout')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/regenerate')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
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
