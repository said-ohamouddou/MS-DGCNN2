"""MS-DGCNN++ component ablations (5-fold CV, density dropout at 100% / 5% retention)."""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, TreeSpeciesDatasetCV
)
from utils.latex_utils import save_latex_table

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]
RETENTION_RATES = [1.0, 0.05]

BASELINE_CONFIG = {
    'MS-DGCNN++ Default': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='concat_conv',
            emb_dims=1024,
            dropout=0.5
        ),
        'label': 'MS-DGCNN++ Default'
    }
}

SINGLE_SCALE_CONFIGS = {
    'DGCNN (k1=5, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[5, 5, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=5, k=30)'
    },
    'DGCNN (k1=20, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[20, 20, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=20, k=30)'
    },
    'DGCNN (k1=30, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[30, 30, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN (k1=30, k=30)'
    },
    'DGCNN+Norm (k1=5, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[5, 5, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=5, k=30)'
    },
    'DGCNN+Norm (k1=20, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[20, 20, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=20, k=30)'
    },
    'DGCNN+Norm (k1=30, k=30)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=[30, 30, 30],
            use_multiscale=False,
            use_normalized_features=False,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN+Norm (k1=30, k=30)'
    },
}

FUSION_CONFIGS = {
    'MS-DGCNN++ (concat_only)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='concat_only',
        ),
        'label': 'concat_only'
    },
    'MS-DGCNN++ (add)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='add',
        ),
        'label': 'add'
    },
    'MS-DGCNN++ (attention)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='attention',
        ),
        'label': 'attention'
    },
    'MS-DGCNN++ (gated)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='gated',
        ),
        'label': 'gated'
    },
    'MS-DGCNN++ (se_fusion)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='se_fusion',
        ),
        'label': 'se_fusion'
    },
}



def get_checkpoint_filename(model_name: str, fold: int) -> str:
    """Generate checkpoint filename."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('=', '')
    return f'{safe_name}_fold{fold}.pth'



@torch.no_grad()
def evaluate_model(model: nn.Module, val_loader, device: str) -> Dict:
    """Evaluate model on validation set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for points, labels in val_loader:
        points = points.to(device)
        labels = labels.to(device)
        
        logits = model(points)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    return {
        'oa': accuracy_score(all_labels, all_preds) * 100,
        'macc': balanced_accuracy_score(all_labels, all_preds) * 100,
        'f1': f1_score(all_labels, all_preds, average='macro') * 100,
    }



def generate_combined_latex_table_dual_retention(
    baseline_results: Dict, 
    single_scale_results: Dict, 
    fusion_results: Dict,
    retention_rates: List[float]
) -> str:
    """Generate a combined LaTeX table with both retention rates, bold best values, and epoch time."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{MS-DGCNN++ Ablation Study Results}")
    lines.append(r"\label{tab:ablation_results}")
    lines.append(r"\resizebox{\textwidth}{!}{")

    lines.append(r"\begin{tabular}{l|cc|cc|cc|c}")
    lines.append(r"\toprule")
    lines.append(r"Model & \multicolumn{2}{c|}{OA (\%)} & \multicolumn{2}{c|}{mAcc (\%)} & \multicolumn{2}{c|}{F1 (\%)} & Time \\")
    ret_labels = [f"r={int(r*100)}\\%" for r in retention_rates]
    lines.append(f"      & {ret_labels[0]} & {ret_labels[1]} & {ret_labels[0]} & {ret_labels[1]} & {ret_labels[0]} & {ret_labels[1]} & (s/epoch) \\\\")
    lines.append(r"\midrule")

    all_models = {}
    for model_name in BASELINE_CONFIG.keys():
        all_models[model_name] = baseline_results.get(model_name, {})
    for model_name in SINGLE_SCALE_CONFIGS.keys():
        all_models[model_name] = single_scale_results.get(model_name, {})
    for model_name in FUSION_CONFIGS.keys():
        all_models[model_name] = fusion_results.get(model_name, {})

    best_values = {}
    for retention in retention_rates:
        ret_key = f'r{int(retention*100)}'
        for metric in ['oa', 'macc', 'f1']:
            values = []
            for model_name, model_data in all_models.items():
                if ret_key in model_data and 'mean' in model_data[ret_key]:
                    val = model_data[ret_key]['mean'].get(metric, 0)
                    if val < 2:
                        val = val * 100
                    values.append(val)
            best_values[(ret_key, metric)] = max(values) if values else 0
    
    def format_cell(model_data, ret_key, metric, best_val):
        """Format a cell with optional bold for best value."""
        if ret_key not in model_data or 'mean' not in model_data[ret_key]:
            return "-"
        mean_val = model_data[ret_key]['mean'].get(metric, 0)
        std_val = model_data[ret_key]['std'].get(metric, 0)
        if mean_val < 2:
            mean_val = mean_val * 100
            std_val = std_val * 100
        cell = f"{mean_val:.1f}$\\pm${std_val:.1f}"
        if abs(mean_val - best_val) < 0.5 and mean_val > 0:
            cell = f"\\textbf{{{cell}}}"
        return cell
    
    def format_epoch_time(model_data):
        """Format epoch time from any available retention rate."""
        for ret_key in [f'r{int(r*100)}' for r in retention_rates]:
            if ret_key in model_data and 'mean' in model_data[ret_key]:
                epoch_time = model_data[ret_key]['mean'].get('epoch_time', 0)
                if epoch_time > 0:
                    return f"{epoch_time:.2f}"
        return "-"
    
    def add_model_row(model_name, model_data, label=None):
        """Add a row for a model."""
        display_name = label if label else model_name
        cells = [display_name]
        for metric in ['oa', 'macc', 'f1']:
            for retention in retention_rates:
                ret_key = f'r{int(retention*100)}'
                best_val = best_values.get((ret_key, metric), 0)
                cells.append(format_cell(model_data, ret_key, metric, best_val))
        cells.append(format_epoch_time(model_data))
        return " & ".join(cells) + " \\\\"

    lines.append(r"\multicolumn{8}{l}{\textbf{Baseline}} \\")
    for model_name in BASELINE_CONFIG.keys():
        if model_name in all_models:
            lines.append(add_model_row(model_name, all_models[model_name]))
    lines.append(r"\midrule")

    lines.append(r"\multicolumn{8}{l}{\textbf{Single-Scale DGCNN}} \\")
    for model_name in SINGLE_SCALE_CONFIGS.keys():
        if model_name in all_models:
            lines.append(add_model_row(model_name, all_models[model_name]))
    lines.append(r"\midrule")

    lines.append(r"\multicolumn{8}{l}{\textbf{Multi-Scale Fusion}} \\")
    for model_name in FUSION_CONFIGS.keys():
        if model_name in all_models:
            label = FUSION_CONFIGS[model_name]['label']
            lines.append(add_model_row(model_name, all_models[model_name], label))
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)



def plot_ablation_results(single_scale_results: Dict, fusion_results: Dict, output_dir: Path):
    """Create bar plots for ablation results."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 13,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    names = list(SINGLE_SCALE_CONFIGS.keys())
    accs = [single_scale_results[n]['mean']['oa'] if n in single_scale_results else 0 for n in names]
    stds = [single_scale_results[n]['std']['oa'] if n in single_scale_results else 0 for n in names]
    
    x = np.arange(len(names))
    bars = ax1.bar(x, accs, yerr=stds, capsize=5, color=['#E74C3C', '#3498DB', '#27AE60'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('DGCNN ', '') for n in names], rotation=0)
    ax1.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax1.set_title('Single-Scale DGCNN: k Value Effect', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    names = list(FUSION_CONFIGS.keys())
    accs = [fusion_results[n]['mean']['oa'] if n in fusion_results else 0 for n in names]
    stds = [fusion_results[n]['std']['oa'] if n in fusion_results else 0 for n in names]
    
    x = np.arange(len(names))
    colors = ['#1ABC9C', '#9B59B6', '#F39C12', '#E74C3C', '#3498DB', '#27AE60']
    bars = ax2.bar(x, accs, yerr=stds, capsize=5, color=colors[:len(names)])
    ax2.set_xticks(x)
    ax2.set_xticklabels([FUSION_CONFIGS[n]['label'] for n in names], rotation=45, ha='right')
    ax2.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax2.set_title('MS-DGCNN++: Fusion Method Comparison', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    out_path = output_dir / 'ablation_results.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved ablation plot to: {out_path}")



def get_checkpoint_filename_with_retention(model_name: str, fold: int, retention: float) -> str:
    """Generate checkpoint filename with retention rate."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('=', '')
    ret_pct = int(retention * 100)
    return f'{safe_name}_r{ret_pct}_fold{fold}.pth'


def train_or_load_model(
    model_config: Dict,
    model_name: str,
    fold: int,
    retention: float,
    checkpoints_dir: Path,
    import_checkpoints: Path,
    args,
    device,
    num_classes: int,
    eval_only: bool = False
) -> Dict:
    """Train or load a model for a specific fold and retention rate."""
    
    ret_pct = int(retention * 100)
    ckpt_filename = get_checkpoint_filename_with_retention(model_name, fold, retention)
    ckpt_path = checkpoints_dir / ckpt_filename

    if import_checkpoints:
        import_path = import_checkpoints / ckpt_filename

        if import_path.exists() and not ckpt_path.exists():
            import shutil
            shutil.copy(import_path, ckpt_path)
            logger.info(f"Imported checkpoint: {ckpt_filename}")
        elif retention == 1.0:
            old_ckpt_filename = get_checkpoint_filename(model_name, fold)
            import_path = import_checkpoints / old_ckpt_filename
            if import_path.exists() and not ckpt_path.exists():
                import shutil
                shutil.copy(import_path, ckpt_path)
                logger.info(f"Imported checkpoint: {old_ckpt_filename} -> {ckpt_filename}")

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        metrics = checkpoint.get('metrics', {})
        logger.info(f"    Fold {fold} r={ret_pct}%: OA={metrics.get('oa', 0):.1f}% (cached)")
        return metrics

    if eval_only:
        logger.warning(f"    Fold {fold} r={ret_pct}%: Checkpoint not found, skipping (eval_only mode)")
        return {}

    logger.info(f"    Fold {fold} r={ret_pct}%: Training...")

    set_seed(args.seed + fold)

    train_loader, val_loader, _, _ = get_cv_dataloaders(
        args.data_path, fold=fold, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
        density_retention=retention, density_seed=args.seed
    )
    
    try:
        model = model_config['factory'](num_classes)
        model = model.to(device)
        
        best_metrics, model = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            save_path=str(ckpt_path), patience=args.patience
        )
        
        logger.info(f"    Fold {fold} r={ret_pct}%: OA={best_metrics.get('oa', 0):.1f}%")
        return best_metrics
    except Exception as e:
        logger.error(f"    Fold {fold} r={ret_pct}%: Training failed: {e}")
        return {}


def aggregate_fold_results(folds_data: List[Dict], epochs: int = 150) -> Dict:
    """Compute mean and std from fold results, including epoch time."""
    if not folds_data:
        return {}

    epoch_times = []
    for f in folds_data:
        total_time = f.get('total_training_time', f.get('training_time', 0))
        if total_time > 0:
            epoch_times.append(total_time / epochs)
    
    return {
        'mean': {
            'oa': np.mean([f.get('oa', 0) for f in folds_data]),
            'macc': np.mean([f.get('macc', 0) for f in folds_data]),
            'f1': np.mean([f.get('f1', 0) for f in folds_data]),
            'epoch_time': np.mean(epoch_times) if epoch_times else 0,
        },
        'std': {
            'oa': np.std([f.get('oa', 0) for f in folds_data]),
            'macc': np.std([f.get('macc', 0) for f in folds_data]),
            'f1': np.std([f.get('f1', 0) for f in folds_data]),
            'epoch_time': np.std(epoch_times) if epoch_times else 0,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='MS-DGCNN2 Component Ablation Studies')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/ablations',
                        help='Output directory for results')
    parser.add_argument('--import_checkpoints', type=str, 
                        default=None,
                        help='Path to import existing checkpoints (all retention rates)')
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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluation only mode - load checkpoints and generate LaTeX, no training')
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(script_dir, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    if args.import_checkpoints and not os.path.isabs(args.import_checkpoints):
        args.import_checkpoints = os.path.join(script_dir, args.import_checkpoints)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    checkpoints_dir = output_dir / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    import_checkpoints = None
    if args.import_checkpoints:
        import_path = Path(args.import_checkpoints)
        if import_path.exists():
            import_checkpoints = import_path
            logger.info(f"Will import checkpoints from: {import_path}")
        else:
            logger.warning(f"Import path not found: {import_path}")

    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)

    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    all_configs = {**BASELINE_CONFIG, **SINGLE_SCALE_CONFIGS, **FUSION_CONFIGS}
    total_experiments = len(all_configs) * args.k_folds * len(RETENTION_RATES)
    
    print("=" * 70)
    print("MS-DGCNN2 COMPONENT ABLATION STUDIES")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Retention rates: {[f'{int(r*100)}%' for r in RETENTION_RATES]}")
    print(f"Baseline: {list(BASELINE_CONFIG.keys())}")
    print(f"Single-scale variants: {list(SINGLE_SCALE_CONFIGS.keys())}")
    print(f"Fusion variants: {list(FUSION_CONFIGS.keys())}")
    print(f"Folds: {args.k_folds}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 70)

    config = {
        'experiment': 'Component Ablation Studies with Density Dropout',
        'data_path': args.data_path,
        'k_folds': args.k_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_points': args.num_points,
        'k_scales': K_SCALES,
        'retention_rates': RETENTION_RATES,
        'seed': args.seed,
        'emb_dims': 1024,
        'dropout': 0.5,
        'timestamp': datetime.now().isoformat()
    }
    save_config(config, output_dir / 'config.json')

    baseline_results = {m: {} for m in BASELINE_CONFIG.keys()}
    single_scale_results = {m: {} for m in SINGLE_SCALE_CONFIGS.keys()}
    fusion_results = {m: {} for m in FUSION_CONFIGS.keys()}
    
    for retention in RETENTION_RATES:
        ret_pct = int(retention * 100)
        ret_key = f'r{ret_pct}'
        
        print(f"\n{'#'*70}")
        print(f"RETENTION RATE: {ret_pct}%")
        print(f"{'#'*70}")
        
        print(f"\n{'='*60}")
        print("BASELINE: MS-DGCNN++ Default")
        print(f"{'='*60}")
        
        for model_name, model_config in BASELINE_CONFIG.items():
            folds_data = []
            for fold in range(args.k_folds):
                metrics = train_or_load_model(
                    model_config, model_name, fold, retention,
                    checkpoints_dir, import_checkpoints,
                    args, device, num_classes, args.eval_only
                )
                if metrics:
                    folds_data.append(metrics)
            
            baseline_results[model_name][ret_key] = aggregate_fold_results(folds_data, args.epochs)
        
        print(f"\n{'='*60}")
        print("ABLATION 1: Single-Scale DGCNN (k value effect)")
        print(f"{'='*60}")
        
        for model_name, model_config in SINGLE_SCALE_CONFIGS.items():
            print(f"\n  {model_config['label']}")
            folds_data = []
            for fold in range(args.k_folds):
                metrics = train_or_load_model(
                    model_config, model_name, fold, retention,
                    checkpoints_dir, import_checkpoints,
                    args, device, num_classes, args.eval_only
                )
                if metrics:
                    folds_data.append(metrics)
            
            single_scale_results[model_name][ret_key] = aggregate_fold_results(folds_data, args.epochs)
        
        print(f"\n{'='*60}")
        print("ABLATION 2: Fusion Method Comparison")
        print(f"{'='*60}")
        
        for model_name, model_config in FUSION_CONFIGS.items():
            print(f"\n  {model_config['label']}")
            folds_data = []
            for fold in range(args.k_folds):
                metrics = train_or_load_model(
                    model_config, model_name, fold, retention,
                    checkpoints_dir, import_checkpoints,
                    args, device, num_classes, args.eval_only
                )
                if metrics:
                    folds_data.append(metrics)
            
            fusion_results[model_name][ret_key] = aggregate_fold_results(folds_data, args.epochs)

    results_save = {
        'k_folds': args.k_folds,
        'retention_rates': RETENTION_RATES,
        'baseline': baseline_results,
        'single_scale': single_scale_results,
        'fusion': fusion_results,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    latex_combined = generate_combined_latex_table_dual_retention(
        baseline_results, single_scale_results, fusion_results, RETENTION_RATES
    )
    with open(output_dir / 'ablation_results.tex', 'w') as f:
        f.write(latex_combined)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for retention in RETENTION_RATES:
        ret_pct = int(retention * 100)
        ret_key = f'r{ret_pct}'
        print(f"\n--- Retention = {ret_pct}% ---")
        
        print("\nBaseline:")
        for m in BASELINE_CONFIG.keys():
            if m in baseline_results and ret_key in baseline_results[m]:
                r = baseline_results[m][ret_key]
                if 'mean' in r:
                    print(f"  {m}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
        
        print("\nSingle-Scale DGCNN:")
        for m in SINGLE_SCALE_CONFIGS.keys():
            if m in single_scale_results and ret_key in single_scale_results[m]:
                r = single_scale_results[m][ret_key]
                if 'mean' in r:
                    print(f"  {m}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
        
        print("\nFusion Methods:")
        for m in FUSION_CONFIGS.keys():
            if m in fusion_results and ret_key in fusion_results[m]:
                r = fusion_results[m][ret_key]
                if 'mean' in r:
                    label = FUSION_CONFIGS[m]['label']
                    print(f"  {label}: OA={r['mean']['oa']:.1f}±{r['std']['oa']:.1f}")
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/ablation_results.tex")
    print("=" * 70)


if __name__ == '__main__':
    main()
