"""Experiment 5: feature-space spectrum / effective rank (uses Exp2 checkpoints)."""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, set_seed, save_config, generate_cv_splits
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    'raw_only': {
        'use_normalized_features': False,
        'local_use_normalized': False,
        'label': 'Raw-only',
        'exp2_name': 'raw_only'
    },
    'default': {
        'use_normalized_features': True,
        'local_use_normalized': False,
        'label': 'Default',
        'exp2_name': 'default'
    },
    'hybrid_everywhere': {
        'use_normalized_features': True,
        'local_use_normalized': True,
        'label': 'Hybrid',
        'exp2_name': 'hybrid_everywhere'
    }
}


def compute_effective_rank(singular_values):
    """
    Compute effective rank using entropy-based formula.
    erank = exp(-sum(p * log(p))) where p = s / sum(s)
    """
    s = singular_values
    s = s[s > 1e-10]  # Filter near-zero values
    
    if len(s) == 0:
        return 0
    
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    erank = np.exp(entropy)
    
    return erank


def extract_features(model, dataloader, device):
    """
    Extract intermediate features from model.
    
    Returns:
        features: dict with 'local', 'branch', 'fused' feature matrices
    """
    model.eval()
    
    all_local = []
    all_branch = []
    all_fused = []
    
    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Extracting features"):
            points = points.to(device)
            
            # Forward with intermediates
            logits, intermediates = model.forward_with_intermediates(points)
            
            # Get features [B, C, N] -> flatten to [B*N, C]
            x_local = intermediates['x_local']  # [B, 64, N]
            x_fused = intermediates['x1']  # [B, 64 or 128, N]
            
            B, C_local, N = x_local.shape
            
            # Reshape to [B*N, C]
            x_local_flat = x_local.permute(0, 2, 1).reshape(-1, C_local)
            all_local.append(x_local_flat.cpu().numpy())
            
            C_fused = x_fused.shape[1]
            x_fused_flat = x_fused.permute(0, 2, 1).reshape(-1, C_fused)
            all_fused.append(x_fused_flat.cpu().numpy())
            
            # Branch features (only for multiscale)
            if intermediates['x_branch'] is not None:
                x_branch = intermediates['x_branch']
                C_branch = x_branch.shape[1]
                x_branch_flat = x_branch.permute(0, 2, 1).reshape(-1, C_branch)
                all_branch.append(x_branch_flat.cpu().numpy())
    
    features = {
        'local': np.vstack(all_local),
        'fused': np.vstack(all_fused)
    }
    
    if all_branch:
        features['branch'] = np.vstack(all_branch)
    
    return features


def analyze_isotropy(features):
    """
    Perform SVD analysis on feature matrices.
    
    Returns:
        results: dict with singular values and effective rank
    """
    results = {}
    
    for name, F in features.items():
        # Center the features
        F_centered = F - F.mean(axis=0)
        
        # SVD
        try:
            U, S, Vh = np.linalg.svd(F_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning(f"SVD failed for {name}, using truncated SVD")
            from scipy.sparse.linalg import svds
            k = min(F_centered.shape) - 1
            U, S, Vh = svds(F_centered, k=k)
            S = S[::-1]  # svds returns in ascending order
        
        # Normalize singular values
        s_norm = S / (S[0] + 1e-10)
        
        # Effective rank
        erank = compute_effective_rank(S)
        
        results[name] = {
            'singular_values': S,
            'singular_values_normalized': s_norm,
            'effective_rank': erank,
            'num_features': F.shape[1]
        }
    
    return results


def run_experiment(args):
    """Run Experiment 5 v2: Feature-Space Isotropy Analysis with Exp2 checkpoints."""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'experiment5_v2_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Retention rates to compare
    retention_rates = [1.0, 0.05]  # 100% and 5%
    
    config = {
        'experiment': 'Experiment 5 v2: Feature-Space Isotropy Analysis (Exp2 checkpoints)',
        'data_path': args.data_path,
        'checkpoint_dir': args.checkpoint_dir,
        'retention_rates': retention_rates,
        'k_folds': args.k_folds,
        'seed': args.seed,
        'variants': list(VARIANTS.keys())
    }
    save_config(config, output_dir / 'config.json')
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    # Results storage: {retention: {variant: {'folds': [...]}}}
    all_results = {}
    
    for retention in retention_rates:
        retention_pct = int(retention * 100)
        logger.info(f"\n{'#'*70}")
        logger.info(f"Processing retention rate: {retention_pct}%")
        logger.info(f"{'#'*70}")
        
        all_results[retention] = {v: {'folds': []} for v in VARIANTS.keys()}
        
        for variant_name, variant_config in VARIANTS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing: {variant_config['label']} (r={retention_pct}%)")
            logger.info(f"{'='*60}")
            
            for fold in range(args.k_folds):
                logger.info(f"\n--- Fold {fold + 1}/{args.k_folds} ---")
                
                # Create model
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
                
                # Load checkpoint from Exp2
                checkpoint_loaded = False
                if args.checkpoint_dir:
                    exp2_name = variant_config['exp2_name']
                    # Try different naming conventions
                    possible_paths = [
                        Path(args.checkpoint_dir) / f'{exp2_name}_r{retention_pct}_fold{fold}.pth',
                        Path(args.checkpoint_dir) / f'{exp2_name}_fold{fold}.pth',
                        Path(args.checkpoint_dir) / f'{exp2_name}_sigma0_fold{fold}.pth',
                    ]
                    
                    for ckpt_path in possible_paths:
                        if ckpt_path.exists():
                            checkpoint = torch.load(ckpt_path, map_location=device)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"Loaded checkpoint: {ckpt_path}")
                            checkpoint_loaded = True
                            break
                    
                    if not checkpoint_loaded:
                        logger.warning(f"No checkpoint found for {exp2_name} r{retention_pct} fold{fold}, using random weights")
                
                # Get validation loader
                _, val_loader, _, _ = get_cv_dataloaders(
                    args.data_path, fold=fold, batch_size=args.batch_size,
                    num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
                )
                
                # Extract features
                features = extract_features(model, val_loader, device)
                
                # Analyze isotropy
                isotropy_results = analyze_isotropy(features)
                
                all_results[retention][variant_name]['folds'].append(isotropy_results)
                
                for feat_name, res in isotropy_results.items():
                    logger.info(f"{feat_name}: effective_rank = {res['effective_rank']:.2f}")
    
    # Aggregate results
    aggregated = aggregate_results(all_results, retention_rates)
    
    # Generate visualizations (single figure per visualization type, both retentions)
    generate_visualizations(all_results, aggregated, retention_rates, output_dir)

    logger.info(f"\nExperiment 5 v2 completed. Results saved to {output_dir}")


def aggregate_results(all_results, retention_rates):
    """Aggregate results across folds for each retention rate."""
    aggregated = {}
    
    for retention in retention_rates:
        aggregated[retention] = {}
        
        for variant, data in all_results[retention].items():
            aggregated[retention][variant] = {}
            
            # Get feature names from first fold
            feature_names = list(data['folds'][0].keys())
            
            for feat_name in feature_names:
                eranks = [fold[feat_name]['effective_rank'] for fold in data['folds']]
                aggregated[retention][variant][feat_name] = {
                    'erank_mean': np.mean(eranks),
                    'erank_std': np.std(eranks),
                    'singular_values_normalized': data['folds'][0][feat_name]['singular_values_normalized']
                }
    
    return aggregated


def generate_visualizations(all_results, aggregated, retention_rates, output_dir):
    """Generate visualizations for Experiment 5 v2 with font=70."""
    
    from plot_style import set_publication_style, get_variant_colors, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    variant_names = list(VARIANTS.keys())
    variant_labels = [VARIANTS[v]['label'] for v in variant_names]
    colors = get_variant_colors(3)
    
    # Font=70 settings
    TITLE_FONT = 70
    LABEL_FONT = 70
    TICK_FONT = 50
    LEGEND_FONT = 50
    
    feat_labels = {'local': 'Local', 'branch': 'Intermediate', 'fused': 'Fused'}
    feat_styles = {'local': '-', 'branch': '--', 'fused': ':'}
    feat_colors = {'local': '#2E86AB', 'branch': '#F18F01', 'fused': '#2A9D8F'}
    
    # 1. Singular value spectrum - 2 rows (retentions) x 3 cols (variants)
    fig, axes = plt.subplots(2, 3, figsize=(60, 40))
    
    for row_idx, retention in enumerate(retention_rates):
        retention_pct = int(retention * 100)
        
        for col_idx, variant in enumerate(variant_names):
            ax = axes[row_idx, col_idx]
            
            for feat_name in ['local', 'branch', 'fused']:
                if feat_name in aggregated[retention][variant]:
                    s_norm = aggregated[retention][variant][feat_name]['singular_values_normalized']
                    ax.semilogy(range(len(s_norm)), s_norm, feat_styles[feat_name], 
                               label=feat_labels[feat_name], linewidth=4, color=feat_colors[feat_name])
            
            ax.set_xlabel('Rank index', fontsize=LABEL_FONT, fontweight='bold')
            ax.set_ylabel(r'$\sigma_i$ / $\sigma_1$', fontsize=LABEL_FONT, fontweight='bold')
            ax.set_title(f'{VARIANTS[variant]["label"]} (r={retention_pct}%)', fontsize=TITLE_FONT, fontweight='bold', pad=20)
            ax.legend(fontsize=LEGEND_FONT, framealpha=0.9)
            ax.grid(True, alpha=0.4, linewidth=2)
            ax.tick_params(axis='both', labelsize=TICK_FONT)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'singular_value_spectrum_combined')
    plt.close()
    
    # 2. Effective rank bar chart - side by side for both retentions
    fig, axes = plt.subplots(1, 2, figsize=(50, 20))
    
    feature_names = ['local', 'branch', 'fused']
    feature_display = ['Local', 'Intermediate', 'Fused']
    x = np.arange(len(feature_names))
    width = 0.25
    
    for ax_idx, retention in enumerate(retention_rates):
        ax = axes[ax_idx]
        retention_pct = int(retention * 100)
        
        for i, (variant, color) in enumerate(zip(variant_names, colors)):
            eranks = []
            stds = []
            for feat in feature_names:
                if feat in aggregated[retention][variant]:
                    eranks.append(aggregated[retention][variant][feat]['erank_mean'])
                    stds.append(aggregated[retention][variant][feat]['erank_std'])
                else:
                    eranks.append(0)
                    stds.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, eranks, width, label=VARIANTS[variant]['label'],
                         color=color, yerr=stds, capsize=10, edgecolor='#333333', linewidth=2,
                         error_kw={'linewidth': 3, 'capthick': 3})
        
        ax.set_xlabel('Feature Location', fontsize=LABEL_FONT, fontweight='bold')
        ax.set_ylabel('Effective Rank', fontsize=LABEL_FONT, fontweight='bold')
        ax.set_title(f'Retention = {retention_pct}%', fontsize=TITLE_FONT, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_display, fontsize=TICK_FONT, fontweight='bold')
        ax.tick_params(axis='y', labelsize=TICK_FONT)
        ax.legend(fontsize=LEGEND_FONT, framealpha=0.9)
        ax.grid(axis='y', alpha=0.4, linewidth=1.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'effective_rank_bar_combined')
    plt.close()
    
    # 3. Effective rank comparison across retentions (grouped by variant)
    fig, axes = plt.subplots(1, 3, figsize=(60, 20))
    
    retention_labels = [f'{int(r*100)}%' for r in retention_rates]
    x = np.arange(len(feature_names))
    width = 0.35
    
    for ax_idx, variant in enumerate(variant_names):
        ax = axes[ax_idx]
        
        for i, retention in enumerate(retention_rates):
            eranks = []
            stds = []
            for feat in feature_names:
                if feat in aggregated[retention][variant]:
                    eranks.append(aggregated[retention][variant][feat]['erank_mean'])
                    stds.append(aggregated[retention][variant][feat]['erank_std'])
                else:
                    eranks.append(0)
                    stds.append(0)
            
            offset = (i - 0.5) * width
            color = '#2A9D8F' if retention == 1.0 else '#E63946'
            bars = ax.bar(x + offset, eranks, width, label=f'r={int(retention*100)}%',
                         color=color, yerr=stds, capsize=10, edgecolor='#333333', linewidth=2,
                         error_kw={'linewidth': 3, 'capthick': 3})
        
        ax.set_xlabel('Feature Location', fontsize=LABEL_FONT, fontweight='bold')
        ax.set_ylabel('Effective Rank', fontsize=LABEL_FONT, fontweight='bold')
        ax.set_title(f'{VARIANTS[variant]["label"]}', fontsize=TITLE_FONT, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_display, fontsize=TICK_FONT, fontweight='bold')
        ax.tick_params(axis='y', labelsize=TICK_FONT)
        ax.legend(fontsize=LEGEND_FONT, framealpha=0.9)
        ax.grid(axis='y', alpha=0.4, linewidth=1.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'effective_rank_retention_comparison')
    plt.close()
    
    # 4. Summary table
    table_data = []
    for feat in feature_names:
        for retention in retention_rates:
            row = [feat.capitalize(), f'{int(retention*100)}%']
            for variant in variant_names:
                if feat in aggregated[retention][variant]:
                    mean = aggregated[retention][variant][feat]['erank_mean']
                    std = aggregated[retention][variant][feat]['erank_std']
                    row.append(f"{mean:.2f} ± {std:.2f}")
                else:
                    row.append("N/A")
            table_data.append(row)
    
    columns = ['Feature', 'Retention'] + variant_labels
    df = pd.DataFrame(table_data, columns=columns)
    df.to_csv(tables_dir / 'effective_rank_table.csv', index=False)
    
    # LaTeX
    latex = df.to_latex(index=False, caption='Effective Rank by Feature Location and Retention',
                        label='tab:exp5_v2_erank')
    with open(tables_dir / 'effective_rank_table.tex', 'w') as f:
        f.write(latex)
    
    logger.info(f"Visualizations saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 5 v2: Feature-Space Isotropy Analysis (Exp2)')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing trained checkpoints from Experiment 2')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
