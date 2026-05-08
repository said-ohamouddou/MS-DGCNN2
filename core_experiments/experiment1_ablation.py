"""Experiment 1: per-scale edge encoding ablations."""

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
    ExperimentResults, plot_grouped_bar_chart, plot_heatmap,
    plot_confusion_matrix_grid, save_results_table, generate_cv_splits,
    evaluate
)
import torch.nn as nn
import json

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


def run_experiment(args):
    """Run Experiment 1: Per-Scale Encoding Ablation."""
    
    # Setup
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
        output_dir = Path(args.output_dir) / f'experiment1_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Generate CV splits (ensures same splits for all experiments)
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Save experiment config
    config = {
        'experiment': 'Experiment 1: Per-Scale Encoding Ablation',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'k_scales': [5, 20, 30],
        'fusion_type': 'concat_conv',
        'emb_dims': 1024,
        'dropout': 0.5,
        'seed': args.seed,
        'variants': list(VARIANTS.keys())
    }
    save_config(config, output_dir / 'config.json')
    
    # Results tracker
    results = ExperimentResults('experiment1', output_dir)
    
    # Get class names from first fold
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    logger.info(f"Dataset: {num_classes} classes - {class_names}")
    
    # Count how many checkpoints exist vs need training
    runs_to_skip = 0
    runs_to_train = 0
    for variant_name in VARIANTS.keys():
        for fold in range(args.k_folds):
            ckpt_path = checkpoints_dir / f'{variant_name}_fold{fold}.pth'
            if ckpt_path.exists():
                runs_to_skip += 1
            else:
                runs_to_train += 1
    
    logger.info(f"Checkpoints found: {runs_to_skip}, to train: {runs_to_train}")
    
    # Train all variants across all folds
    for variant_name, variant_config in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Variant: {variant_config['label']}")
        logger.info(f"{'='*60}")
        
        for fold in range(args.k_folds):
            # Checkpoint path
            ckpt_path = checkpoints_dir / f'{variant_name}_fold{fold}.pth'
            
            # Check if checkpoint exists
            if ckpt_path.exists():
                logger.info(f"\n--- Fold {fold + 1}/{args.k_folds}: Loading checkpoint ---")
                
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
                
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Get metrics from checkpoint or re-evaluate
                if 'metrics' in checkpoint and checkpoint['metrics']:
                    best_metrics = checkpoint['metrics']
                else:
                    # Re-evaluate
                    _, val_loader, _, _ = get_cv_dataloaders(
                        args.data_path, fold=fold, batch_size=args.batch_size,
                        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
                    )
                    criterion = nn.CrossEntropyLoss()
                    best_metrics = evaluate(model, val_loader, criterion, device)
                
                results.add_fold_result(variant_name, fold, best_metrics)
                logger.info(f"Fold {fold} - OA: {best_metrics['oa']:.2f}%, mAcc: {best_metrics['macc']:.2f}%")
                continue
            
            # Train new model
            logger.info(f"\n--- Fold {fold + 1}/{args.k_folds}: Training ---")
            
            # Set seed for reproducibility
            set_seed(args.seed + fold)
            
            # Get dataloaders for this fold
            train_loader, val_loader, num_classes, _ = get_cv_dataloaders(
                args.data_path, fold=fold, batch_size=args.batch_size,
                num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
            )
            
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
            
            # Train
            best_metrics, model = train_model(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                save_path=str(ckpt_path), patience=args.patience
            )
            
            # Store results
            results.add_fold_result(variant_name, fold, best_metrics)
            
            logger.info(f"Fold {fold} - OA: {best_metrics['oa']:.2f}%, mAcc: {best_metrics['macc']:.2f}%")
    
    # Save all results
    results.save()
    
    # Generate visualizations
    generate_visualizations(results, class_names, output_dir)

    logger.info(f"\nExperiment 1 completed. Results saved to {output_dir}")


def generate_visualizations(results, class_names, output_dir):
    """Generate all required visualizations for Experiment 1."""
    
    from plot_style import set_publication_style, get_variant_colors, add_panel_label, save_figure
    set_publication_style()
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    
    variant_names = list(VARIANTS.keys())
    variant_labels = [VARIANTS[v]['label'] for v in variant_names]
    colors = get_variant_colors(len(variant_names))
    
    # 1. Grouped bar chart - OA with error bars
    oa_means = []
    oa_stds = []
    for variant in variant_names:
        agg = results.get_aggregated_metrics(variant)
        oa_means.append(agg['oa_mean'])
        oa_stds.append(agg['oa_std'])
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variant_names))
    
    bars = ax.bar(x, oa_means, yerr=oa_stds, capsize=8, color=colors, 
                  edgecolor='#333333', linewidth=2.0, error_kw={'linewidth': 2.5, 'capthick': 2.5})
    ax.set_xlabel('Variant', fontsize=21, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=21, fontweight='bold')
    ax.set_title('Per-Scale Encoding Ablation', fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=15, ha='right', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, oa_means, oa_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.8,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, figures_dir / 'oa_bar_chart')
    plt.close()
    
    # 2. Per-class F1 heatmap
    f1_matrix = []
    for variant in variant_names:
        agg = results.get_aggregated_metrics(variant)
        f1_matrix.append(agg['f1_mean'])
    f1_matrix = np.array(f1_matrix).T  # [n_classes, n_variants]
    
    # Sort class names alphabetically
    sorted_indices = np.argsort(class_names)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    f1_matrix_sorted = f1_matrix[sorted_indices]
    
    # Custom heatmap with publication style
    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(f1_matrix_sorted, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=variant_labels, yticklabels=sorted_class_names,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'F1 Score'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_xlabel('Variant', fontsize=21, fontweight='bold')
    ax.set_ylabel('Species', fontsize=21, fontweight='bold')
    ax.set_title('Per-Class F1 Score by Variant', fontsize=24, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    plt.tight_layout()
    save_figure(fig, figures_dir / 'f1_heatmap')
    plt.close()
    
    # 3. Confusion matrices - separate figures for each variant
    for variant, label in zip(variant_names, variant_labels):
        fold_results = results.results[variant][0]
        cm = fold_results['confusion_matrix']
        
        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_title(f'Confusion Matrix: {label}', fontsize=24, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted', fontsize=21, fontweight='bold')
        ax.set_ylabel('True', fontsize=21, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontweight='bold')
        plt.tight_layout()
        save_figure(fig, figures_dir / f'confusion_matrix_{variant}')
        plt.close()
    
    # 4. Full results table
    table_data = []
    for variant in variant_names:
        agg = results.get_aggregated_metrics(variant)
        row = [
            f"{agg['oa_mean']:.2f} ± {agg['oa_std']:.2f}",
            f"{agg['macc_mean']:.2f} ± {agg['macc_std']:.2f}"
        ]
        for i in range(len(class_names)):
            row.append(f"{agg['f1_mean'][i]:.3f}")
        table_data.append(row)
    
    columns = ['OA ± std', 'mAcc ± std'] + [f'F1_{c}' for c in class_names]
    
    import pandas as pd
    df = pd.DataFrame(table_data, columns=columns, index=variant_labels)
    df.to_csv(tables_dir / 'full_results.csv')
    
    # LaTeX table
    latex = df.to_latex(float_format='%.2f', caption='Experiment 1: Per-Scale Encoding Ablation Results',
                        label='tab:exp1_results')
    with open(tables_dir / 'full_results.tex', 'w') as f:
        f.write(latex)
    
    logger.info(f"Visualizations saved to {figures_dir}")
    logger.info(f"Tables saved to {tables_dir}")


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Per-Scale Encoding Ablation')
    parser.add_argument('--data_path', type=str, 
                        default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='Path to existing experiment directory to resume/regenerate')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_path)
    
    run_experiment(args)


if __name__ == '__main__':
    main()
