"""Experiment 4: max-pooling neighbor provenance (raw vs normalized edges)."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model, get_hierarchical_graph_feature
from utils.experiment_utils import (
    get_cv_dataloaders, set_seed, save_config, generate_cv_splits,
    TreeSpeciesDatasetCV, evaluate
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_neighbor_distance_labels(points, k2=20):
    """
    For each point, label its k2 neighbors as near or far based on distance.
    
    Near = closer half of neighbors (indices 0 to k2/2-1 in sorted order)
    Far = farther half of neighbors (indices k2/2 to k2-1 in sorted order)
    
    Returns:
        neighbor_labels: [N, k2] boolean array (True = far, False = near)
        neighbor_distances: [N, k2] distance values
        neighbor_indices: [N, k2] indices of neighbors
    """
    dists = cdist(points, points)
    N = len(points)
    
    neighbor_indices = np.zeros((N, k2), dtype=int)
    neighbor_distances = np.zeros((N, k2))
    neighbor_labels = np.zeros((N, k2), dtype=bool)  # True = far
    
    half_k = k2 // 2
    
    for i in range(N):
        # Sort by distance, exclude self (index 0 after sort is self with dist=0)
        sorted_idx = np.argsort(dists[i])[1:k2+1]  # k2 nearest neighbors
        neighbor_indices[i] = sorted_idx
        neighbor_distances[i] = dists[i, sorted_idx]
        
        # Label: first half = near (False), second half = far (True)
        neighbor_labels[i, half_k:] = True
    
    return neighbor_labels, neighbor_distances, neighbor_indices


def analyze_maxpool_provenance(model, dataloader, device, k2=20):
    """
    Analyze which neighbors win max-pooling based on distance (near vs far).
    
    Returns:
        results: dict with win statistics and per-channel correlations
    """
    model.eval()
    
    all_near_wins = []
    all_far_wins = []
    
    # Per-channel statistics
    channel_near_wins = [[] for _ in range(64)]
    channel_far_wins = [[] for _ in range(64)]
    
    # Per-point statistics for visualization
    point_far_proportions = []
    
    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Analyzing max-pool"):
            points_np = points.numpy()
            points = points.to(device)
            
            batch_size = points.size(0)
            
            # Get hierarchical features
            if points.dim() == 3 and points.shape[-1] == 3:
                x = points.permute(0, 2, 1)
            else:
                x = points
            
            k_scales = [model.k_local, model.k_branch]
            hierarchical_features = get_hierarchical_graph_feature(
                x, k_scales,
                use_normalized=model.use_normalized_features,
                branch_feature_mode=model.branch_feature_mode,
                local_use_normalized=model.local_use_normalized
            )
            
            # Get branch features before and after max-pool
            x_branch_pre = model.conv1_branch(hierarchical_features[1])  # [B, 64, N, k2]
            max_values, max_indices = x_branch_pre.max(dim=-1)  # [B, 64, N]
            
            max_indices = max_indices.cpu().numpy()  # [B, 64, N]
            
            # Analyze each sample
            for b in range(batch_size):
                pts = points_np[b]
                neighbor_labels, neighbor_dists, neighbor_idx = get_neighbor_distance_labels(pts, k2=k2)
                
                # Per-point far win proportion
                point_far_props = np.zeros(pts.shape[0])
                
                # For each point and channel, check if winner is near or far
                for n in range(pts.shape[0]):
                    far_count = 0
                    for c in range(64):
                        winner_idx = max_indices[b, c, n]
                        if winner_idx < k2:
                            is_far = neighbor_labels[n, winner_idx]
                            if is_far:
                                all_far_wins.append(1)
                                all_near_wins.append(0)
                                channel_far_wins[c].append(1)
                                channel_near_wins[c].append(0)
                                far_count += 1
                            else:
                                all_far_wins.append(0)
                                all_near_wins.append(1)
                                channel_far_wins[c].append(0)
                                channel_near_wins[c].append(1)
                    
                    point_far_props[n] = far_count / 64.0
                
                point_far_proportions.append(point_far_props)
    
    # Compute statistics
    total_wins = len(all_far_wins)
    far_win_rate = np.sum(all_far_wins) / total_wins
    near_win_rate = np.sum(all_near_wins) / total_wins
    
    # Per-channel statistics
    channel_far_rates = [np.mean(channel_far_wins[c]) if channel_far_wins[c] else 0.5 
                         for c in range(64)]
    
    return {
        'far_win_rate': far_win_rate,
        'near_win_rate': near_win_rate,
        'total_wins': total_wins,
        'channel_far_rates': channel_far_rates,
        'point_far_proportions': point_far_proportions
    }


def run_experiment(args):
    """Run Experiment 4: Max-Pooling Activation Provenance (V2)."""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'experiment4_v2_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Determine checkpoint sources to test
    checkpoint_sources = []
    if args.checkpoint_dir:
        checkpoint_sources.append({
            'name': 'exp1',
            'dir': Path(args.checkpoint_dir),
            'label': 'Exp1 (Clean Training)'
        })
    if args.checkpoint_dir_exp2:
        checkpoint_sources.append({
            'name': 'exp2', 
            'dir': Path(args.checkpoint_dir_exp2),
            'label': 'Exp2 (Density Dropout Training)'
        })
    
    if not checkpoint_sources:
        raise ValueError("At least one checkpoint directory must be provided (--checkpoint_dir or --checkpoint_dir_exp2)")
    
    # Determine data conditions to test
    if args.retention is not None:
        retention_rates = args.retention if isinstance(args.retention, list) else [args.retention]
    else:
        retention_rates = [1.0]  # Always test clean data
        if args.include_density_drop:
            retention_rates.append(0.05)  # Add r=5% stressed data
    
    config = {
        'experiment': 'Experiment 4: Max-Pooling Activation Provenance (V2)',
        'data_path': args.data_path,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_dir_exp2': args.checkpoint_dir_exp2,
        'checkpoint_sources': [s['name'] for s in checkpoint_sources],
        'k_folds': args.k_folds,
        'seed': args.seed,
        'retention_rates': retention_rates,
        'analysis_type': 'distance-based (near/far)'
    }
    save_config(config, output_dir / 'config.json')
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    # Define the two probe models
    # Note: For Exp2 checkpoints, we use 'default' as the normalized probe since
    # 'normalized_only' was not trained in Exp2. The 'default' model uses normalized
    # features at the branch scale where max-pooling provenance is analyzed.
    probes = {
        'raw_input': {
            'use_normalized_features': False,
            'local_use_normalized': False,
            'branch_feature_mode': 'full',
            'label': 'Raw-input',
            'exp1_name': 'raw_only',
            'exp2_name': 'raw_only'
        },
        'normalized_input': {
            'use_normalized_features': True,
            'local_use_normalized': False,
            'branch_feature_mode': 'full',  # 'default' uses full mode with normalization
            'label': 'Normalized-input',
            'exp1_name': 'normalized_only',
            'exp2_name': 'default'  # Use 'default' for Exp2 checkpoints
        }
    }
    
    all_results = {}
    
    # Loop over checkpoint sources (exp1, exp2, or both)
    for ckpt_source in checkpoint_sources:
        source_name = ckpt_source['name']
        ckpt_dir = ckpt_source['dir']
        source_label = ckpt_source['label']
        
        logger.info(f"\n{'*'*80}")
        logger.info(f"CHECKPOINT SOURCE: {source_label}")
        logger.info(f"Directory: {ckpt_dir}")
        logger.info(f"{'*'*80}")
        
        all_results[source_name] = {}
        
        for retention in retention_rates:
            retention_key = f'r{int(retention*100)}'
            all_results[source_name][retention_key] = {}
            
            logger.info(f"\n{'#'*70}")
            logger.info(f"Testing with retention rate: {retention*100:.0f}%")
            logger.info(f"{'#'*70}")
            
            for probe_name, probe_config in probes.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Analyzing: {probe_config['label']}")
                logger.info(f"{'='*60}")
                
                fold_results = []
                
                for fold in range(args.k_folds):
                    logger.info(f"\n--- Fold {fold + 1}/{args.k_folds} ---")
                    
                    # Create model
                    model = create_model(
                        num_classes=num_classes,
                        k_scales=[5, 20, 30],
                        use_multiscale=True,
                        use_normalized_features=probe_config['use_normalized_features'],
                        local_use_normalized=probe_config['local_use_normalized'],
                        branch_feature_mode=probe_config['branch_feature_mode'],
                        fusion_type='concat_conv',
                        emb_dims=1024,
                        dropout=0.5
                    ).to(device)
                    
                    # Load checkpoint - try multiple naming conventions
                    # Use exp2_name for Exp2 checkpoints, exp1_name for Exp1/Exp3
                    ckpt_loaded = False
                    retention_pct = int(retention * 100)
                    exp1_variant = probe_config['exp1_name']
                    exp2_variant = probe_config.get('exp2_name', exp1_variant)
                    
                    possible_paths = [
                        # Experiment 2 naming: {variant}_r{retention}_fold{fold}.pth (use exp2_name)
                        ckpt_dir / f"{exp2_variant}_r{retention_pct}_fold{fold}.pth",
                        # Experiment 1 naming: {variant}_fold{fold}.pth (use exp1_name)
                        ckpt_dir / f"{exp1_variant}_fold{fold}.pth",
                        # Experiment 3 naming: {variant}_sigma0_fold{fold}.pth (use exp1_name)
                        ckpt_dir / f"{exp1_variant}_sigma0_fold{fold}.pth",
                    ]
                    
                    for ckpt_path in possible_paths:
                        if ckpt_path.exists():
                            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"Loaded checkpoint: {ckpt_path}")
                            ckpt_loaded = True
                            break
                    
                    if not ckpt_loaded:
                        logger.warning(f"No checkpoint found for {exp1_variant}/{exp2_variant} fold {fold}, using random weights")
                    
                    # Get validation loader with appropriate retention
                    _, val_loader, _, _ = get_cv_dataloaders(
                        args.data_path, fold=fold, batch_size=args.batch_size,
                        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                        density_retention=retention, density_seed=args.seed + fold
                    )
                    
                    # Analyze max-pooling provenance
                    fold_result = analyze_maxpool_provenance(model, val_loader, device, k2=20)
                    fold_results.append(fold_result)
                    
                    logger.info(f"Near win rate: {fold_result['near_win_rate']*100:.2f}%")
                    logger.info(f"Far win rate: {fold_result['far_win_rate']*100:.2f}%")
                
                # Aggregate across folds
                far_rates = [r['far_win_rate'] for r in fold_results]
                near_rates = [r['near_win_rate'] for r in fold_results]
                
                # Aggregate channel statistics
                all_channel_far_rates = np.array([r['channel_far_rates'] for r in fold_results])
                channel_far_mean = np.mean(all_channel_far_rates, axis=0)
                channel_far_std = np.std(all_channel_far_rates, axis=0)
                
                all_results[source_name][retention_key][probe_name] = {
                    'far_win_mean': np.mean(far_rates),
                    'far_win_std': np.std(far_rates),
                    'near_win_mean': np.mean(near_rates),
                    'near_win_std': np.std(near_rates),
                    'channel_far_mean': channel_far_mean.tolist(),
                    'channel_far_std': channel_far_std.tolist(),
                    'checkpoint_source': source_name,
                    'fold_results': [
                        {k: v if not isinstance(v, list) else 'omitted' 
                         for k, v in r.items() if k != 'point_far_proportions'}
                        for r in fold_results
                    ]
                }
    
    # Save results
    save_results_json(all_results, output_dir / 'all_results.json')
    
    # Generate visualizations
    generate_visualizations(all_results, probes, retention_rates, checkpoint_sources, output_dir)

    logger.info(f"\nExperiment 4 (V2) completed. Results saved to {output_dir}")


def save_results_json(results, path):
    """Save results to JSON."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def generate_visualizations(results, probes, retention_rates, checkpoint_sources, output_dir):
    """Generate visualizations for Experiment 4 V2."""
    
    from plot_style import set_publication_style, get_variant_colors, add_panel_label, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    probe_names = list(probes.keys())
    probe_labels = [probes[p]['label'] for p in probe_names]
    source_names = [s['name'] for s in checkpoint_sources]
    source_labels = {s['name']: s['label'] for s in checkpoint_sources}
    
    # Generate plots for each checkpoint source
    for source_name in source_names:
        source_results = results[source_name]
        source_label = source_labels[source_name]
        
        # 1. Stacked bar chart - Near vs Far wins (grid layout for publication)
        # Font=70 for all texts
        n_rates = len(retention_rates)
        if n_rates == 5:
            # Publication-ready 2x3 grid (5 panels + 1 empty)
            fig, axes = plt.subplots(2, 3, figsize=(42, 28))
            axes = axes.flatten()
            panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
            # Hide the 6th (empty) panel
            axes[5].set_visible(False)
        elif n_rates == 4:
            # Publication-ready 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(34, 28))
            axes = axes.flatten()
            panel_labels = ['(a)', '(b)', '(c)', '(d)']
        elif n_rates == 1:
            fig, axes = plt.subplots(1, 1, figsize=(20, 18))
            axes = [axes]
            panel_labels = ['']
        else:
            # Fallback to appropriate grid
            if n_rates <= 3:
                fig, axes = plt.subplots(1, n_rates, figsize=(14*n_rates, 14))
            else:
                cols = 3
                rows = (n_rates + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(14*cols, 14*rows))
                axes = axes.flatten()
            if n_rates == 1:
                axes = [axes]
            elif not hasattr(axes, '__len__'):
                axes = [axes]
            panel_labels = [f'({chr(97+i)})' for i in range(n_rates)]
        
        # Store handles for shared legend
        legend_handles = None
        legend_labels = None
        
        for idx, (ax, retention) in enumerate(zip(axes, retention_rates)):
            retention_key = f'r{int(retention*100)}'
            
            if retention_key not in source_results:
                continue
            
            x = np.arange(len(probe_names))
            width = 0.6
            
            near_means = [source_results[retention_key][p]['near_win_mean'] * 100 for p in probe_names if p in source_results[retention_key]]
            far_means = [source_results[retention_key][p]['far_win_mean'] * 100 for p in probe_names if p in source_results[retention_key]]
            near_stds = [source_results[retention_key][p]['near_win_std'] * 100 for p in probe_names if p in source_results[retention_key]]
            far_stds = [source_results[retention_key][p]['far_win_std'] * 100 for p in probe_names if p in source_results[retention_key]]
            
            if not near_means:
                continue
            
            # Font=70 for all texts
            bars1 = ax.bar(x, near_means, width, label='Near neighbors', color='#457B9D',
                           yerr=near_stds, capsize=10, edgecolor='#333333', linewidth=2,
                           error_kw={'linewidth': 3, 'capthick': 3})
            bars2 = ax.bar(x, far_means, width, bottom=near_means, label='Far neighbors',
                           color='#E63946', yerr=far_stds, capsize=10, edgecolor='#333333', linewidth=2,
                           error_kw={'linewidth': 3, 'capthick': 3})
            
            ax.axhline(y=50, color='#333333', linestyle='--', linewidth=3, label='Chance (50%)')
            
            ax.set_ylabel('Win Rate (%)', fontsize=70, fontweight='bold')
            ax.set_title(f'{panel_labels[idx]} r = {retention*100:.0f}%', fontsize=70, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(probe_labels, fontsize=50, fontweight='bold')
            ax.tick_params(axis='y', labelsize=50)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Store legend handles from first panel
            if idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            
            # Add value labels (font=50)
            fontsize_val = 50
            for i, (near, far) in enumerate(zip(near_means, far_means)):
                ax.text(i, near/2, f'{near:.1f}%', ha='center', va='center', fontsize=fontsize_val, color='white', fontweight='bold')
                ax.text(i, near + far/2, f'{far:.1f}%', ha='center', va='center', fontsize=fontsize_val, color='white', fontweight='bold')
        
        # Add shared legend as horizontal line at the very bottom
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3, 
                      fontsize=50, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig, figures_dir / f'stacked_bar_chart_{source_name}')
        plt.close()
        
        # 2. Channel-level scatter plot (font=70)
        fig, axes = plt.subplots(1, len(retention_rates), figsize=(20*len(retention_rates), 16))
        if len(retention_rates) == 1:
            axes = [axes]
        
        for idx, (ax, retention) in enumerate(zip(axes, retention_rates)):
            retention_key = f'r{int(retention*100)}'
            
            if retention_key not in source_results:
                continue
            
            # Use distinct colors for each probe
            colors = {'raw_input': '#E63946', 'normalized_input': '#457B9D'}
            markers = {'raw_input': 'o', 'normalized_input': 's'}
            
            for probe_name in probe_names:
                if probe_name not in source_results[retention_key]:
                    continue
                channel_rates = source_results[retention_key][probe_name]['channel_far_mean']
                ax.scatter(range(64), channel_rates, c=colors.get(probe_name, '#333333'), 
                          marker=markers.get(probe_name, 'o'),
                          label=probes[probe_name]['label'], alpha=0.8, s=200, edgecolors='#333333', linewidths=1)
            
            ax.axhline(y=0.5, color='#333333', linestyle='--', linewidth=3)
            ax.axhspan(0.45, 0.55, alpha=0.15, color='#666666', label='±5% from chance')
            
            # Font=70 for all texts
            ax.set_xlabel('Channel Index', fontsize=70, fontweight='bold')
            ax.set_ylabel('Far Neighbor Win Rate', fontsize=70, fontweight='bold')
            ax.set_title(f'{source_label}\nPer-Channel (r={retention*100:.0f}%)', fontsize=70, fontweight='bold', pad=20)
            ax.legend(loc='upper right', fontsize=50, framealpha=0.9)
            ax.set_xlim(-1, 64)
            # Expand y-axis to show raw_input points (~0.8)
            ax.set_ylim(0.0, 1.0)
            ax.tick_params(axis='both', labelsize=50)
            ax.grid(True, alpha=0.4, linewidth=1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        save_figure(fig, figures_dir / f'channel_scatter_{source_name}')
        plt.close()
    
    # 3. Comparison across checkpoint sources (if multiple)
    if len(source_names) > 1:
        for retention in retention_rates:
            retention_key = f'r{int(retention*100)}'
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(probe_names))
            width = 0.35
            bar_colors = ['#2A9D8F', '#E9C46A', '#E76F51', '#264653']
            
            for i, source_name in enumerate(source_names):
                if retention_key not in results[source_name]:
                    continue
                far_means = [results[source_name][retention_key][p]['far_win_mean'] * 100 
                            for p in probe_names if p in results[source_name][retention_key]]
                far_stds = [results[source_name][retention_key][p]['far_win_std'] * 100 
                           for p in probe_names if p in results[source_name][retention_key]]
                
                if not far_means:
                    continue
                
                offset = (i - (len(source_names)-1)/2) * width
                bars = ax.bar(x + offset, far_means, width, 
                             label=source_labels[source_name], color=bar_colors[i % len(bar_colors)],
                             yerr=far_stds, capsize=6, edgecolor='#333333', linewidth=1.5,
                             error_kw={'linewidth': 2, 'capthick': 2})
            
            ax.axhline(y=50, color='#333333', linestyle='--', linewidth=3, label='Chance (50%)')
            
            # Font=70 for all texts
            ax.set_xlabel('Model Type', fontsize=70, fontweight='bold')
            ax.set_ylabel('Far Neighbor Win Rate (%)', fontsize=70, fontweight='bold')
            ax.set_title(f'Exp1 vs Exp2 Comparison (r={retention*100:.0f}%)', fontsize=70, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(probe_labels, fontsize=50, fontweight='bold')
            ax.tick_params(axis='y', labelsize=50)
            ax.legend(fontsize=50, framealpha=0.9)
            ax.grid(axis='y', alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            save_figure(fig, figures_dir / f'source_comparison_r{int(retention*100)}')
            plt.close()
    
    # 4. Summary table (all sources combined)
    table_data = []
    for probe_name in probe_names:
        row = [probes[probe_name]['label']]
        for source_name in source_names:
            for retention in retention_rates:
                retention_key = f'r{int(retention*100)}'
                if source_name in results and retention_key in results[source_name] and probe_name in results[source_name][retention_key]:
                    r = results[source_name][retention_key][probe_name]
                    row.append(f"{r['near_win_mean']*100:.2f} ± {r['near_win_std']*100:.2f}")
                    row.append(f"{r['far_win_mean']*100:.2f} ± {r['far_win_std']*100:.2f}")
                else:
                    row.append("N/A")
                    row.append("N/A")
        table_data.append(row)
    
    columns = ['Model']
    for source_name in source_names:
        for retention in retention_rates:
            columns.extend([f'Near ({source_name} r={retention*100:.0f}%)', f'Far ({source_name} r={retention*100:.0f}%)'])
    
    df = pd.DataFrame(table_data, columns=columns)
    df.to_csv(tables_dir / 'provenance_summary.csv', index=False)
    
    latex = df.to_latex(index=False, caption='Max-Pooling Provenance: Near vs Far Neighbor Win Rates',
                        label='tab:exp4_provenance')
    with open(tables_dir / 'provenance_summary.tex', 'w') as f:
        f.write(latex)
    
    logger.info(f"Visualizations saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 4: Max-Pooling Provenance (V2)')
    parser.add_argument('--data_path', type=str, default='STPCTLC')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing Experiment 1 checkpoints (for r=100%)')
    parser.add_argument('--checkpoint_dir_exp2', type=str, required=True,
                        help='Directory containing Experiment 2 checkpoints (REQUIRED)')
    parser.add_argument('--include_density_drop', action='store_true',
                        help='Also analyze on density-dropped data (r=5%)')
    parser.add_argument('--retention', type=float, nargs='+', default=None,
                        help='Retention rate(s) to test (e.g., 0.75 0.5 0.25 0.05 for 75%%, 50%%, 25%%, 5%%). Overrides default behavior.')
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
