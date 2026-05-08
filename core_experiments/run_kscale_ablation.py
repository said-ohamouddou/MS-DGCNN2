"""MS-DGCNN++ k-scale sensitivity (5-fold CV)."""

import os
import sys
import json
import argparse
import time
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



BASELINE_K_SCALES = [5, 20, 30]
K_LOCAL_VALUES = [3, 5, 10, 15, 20, 25, 30]
K_INTERMEDIATE_VALUES = [5, 10, 15, 20, 25, 30, 35]
K_GLOBAL_VALUES = [10, 20, 25, 30, 35, 40, 45]



def create_default_asymmetric(num_classes: int, k_scales: List[int]) -> nn.Module:
    """Create MS-DGCNN++ (Default Asymmetric) model."""
    return create_model(
        num_classes=num_classes,
        k_scales=k_scales,
        use_multiscale=True,
        use_normalized_features=True,
        local_use_normalized=False,
        fusion_type='concat_conv',
    )



def get_checkpoint_filename(scale_name: str, k_value: int, fold: int, retention: float = 1.0) -> str:
    """Generate checkpoint filename."""
    ret_pct = int(retention * 100)
    return f'default_asymmetric_{scale_name}_{k_value}_r{ret_pct}_fold{fold}.pth'



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


def measure_avg_epoch_time(model: nn.Module, train_loader, device: str, num_batches: int = 10) -> float:
    """
    Measure average epoch time by timing forward+backward passes.
    
    Args:
        model: The model to measure
        train_loader: Training data loader
        device: Device to run on
        num_batches: Number of batches to average over
        
    Returns:
        Estimated epoch time in seconds
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Warm-up
    for i, (points, labels) in enumerate(train_loader):
        if i >= 2:
            break
        points = points.to(device)
        labels = labels.to(device)
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
    
    # Synchronize CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time
    batch_times = []
    for i, (points, labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        points = points.to(device)
        labels = labels.to(device)
        
        start = time.time()
        logits = model(points)
        loss = criterion(logits, labels)
        loss.backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        batch_times.append(time.time() - start)
    
    # Estimate epoch time
    avg_batch_time = np.mean(batch_times)
    total_batches = len(train_loader)
    estimated_epoch_time = avg_batch_time * total_batches
    
    # Clear gradients
    model.zero_grad()
    
    return estimated_epoch_time



def generate_combined_kscale_latex_table(all_results: Dict, scale_experiments: List[Tuple], time_results: Dict = None) -> str:
    """Generate a single combined LaTeX table for all k-scale ablation results with epoch time."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{K-Scale Sensitivity Analysis: MS-DGCNN++ (Default Asymmetric)}")
    lines.append(r"\label{tab:kscale_ablation}")
    
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Scale & k Value & OA (\%) & mAcc (\%) & F1 (\%) & Time (s/epoch) \\")
    lines.append(r"\midrule")
    
    # Find global best OA across all scales
    all_oas = []
    for scale_name, k_values, baseline in scale_experiments:
        if scale_name in all_results:
            for k in k_values:
                if k in all_results[scale_name] and 'mean' in all_results[scale_name][k]:
                    all_oas.append(all_results[scale_name][k]['mean']['oa'])
    best_oa = max(all_oas) if all_oas else 0
    
    for idx, (scale_name, k_values, baseline) in enumerate(scale_experiments):
        if scale_name not in all_results:
            continue
        
        results = all_results[scale_name]
        first_row = True
        
        for k in k_values:
            if k not in results or 'mean' not in results[k]:
                continue
            
            r = results[k]
            mean_oa = r['mean']['oa']
            std_oa = r['std']['oa']
            mean_macc = r['mean']['macc']
            std_macc = r['std']['macc']
            # F1 score: convert to percentage if stored as decimal
            mean_f1 = r['mean']['f1'] * 100 if r['mean']['f1'] < 1.5 else r['mean']['f1']
            std_f1 = r['std']['f1'] * 100 if r['std']['f1'] < 1.5 else r['std']['f1']
            
            # Get epoch time if available
            epoch_time_str = "-"
            if time_results and scale_name in time_results and k in time_results[scale_name]:
                epoch_time_str = f"{time_results[scale_name][k]:.2f}"
            elif 'epoch_time' in r['mean'] and r['mean']['epoch_time'] > 0:
                epoch_time_str = f"{r['mean']['epoch_time']:.2f}"
            
            # Bold best OA
            oa_cell = f"{mean_oa:.1f}$\\pm${std_oa:.1f}"
            if abs(mean_oa - best_oa) < 0.01:
                oa_cell = f"\\textbf{{{oa_cell}}}"
            
            # Mark baseline with asterisk
            k_display = f"{k}$^*$" if k == baseline else str(k)
            
            # First row gets the scale name
            scale_display = scale_name.replace('_', '\\_') if first_row else ""
            first_row = False
            
            row = f"{scale_display} & {k_display} & {oa_cell} & {mean_macc:.1f}$\\pm${std_macc:.1f} & {mean_f1:.1f}$\\pm${std_f1:.1f} & {epoch_time_str} \\\\"
            lines.append(row)
        
        # Add separator between scales (except after last)
        if idx < len(scale_experiments) - 1:
            lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{6}{l}{\footnotesize $^*$Baseline value}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)



def plot_kscale_sensitivity(all_results: Dict, output_dir: Path, time_results: Dict = None):
    """Create visualization showing impact of each scale on OA, F1, and execution time."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 22,
        'font.weight': 'bold',
        'axes.labelsize': 26,
        'axes.labelweight': 'bold',
        'axes.titlesize': 28,
        'axes.titleweight': 'bold',
        'legend.fontsize': 20,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
    })
    
    scale_configs = [
        ('k_local', K_LOCAL_VALUES, BASELINE_K_SCALES[0], '#2E86AB'),
        ('k_intermediate', K_INTERMEDIATE_VALUES, BASELINE_K_SCALES[1], '#A23B72'),
        ('k_global', K_GLOBAL_VALUES, BASELINE_K_SCALES[2], '#F18F01'),
    ]
    
    # Determine if we have time data
    has_time_data = time_results is not None and any(time_results.get(s[0]) for s in scale_configs)
    
    # Create figure with 3 rows: OA+F1, Time (if available)
    if has_time_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metric_axes = axes[0]
        time_axes = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metric_axes = axes
        time_axes = None
    
    for idx, (scale_name, k_values, baseline, color) in enumerate(scale_configs):
        ax = metric_axes[idx]
        
        if scale_name not in all_results:
            continue
        
        results = all_results[scale_name]
        
        # Collect OA data
        oa_vals = []
        oa_stds = []
        valid_k_oa = []
        
        # Collect F1 data
        f1_vals = []
        f1_stds = []
        valid_k_f1 = []
        
        for k in k_values:
            if k in results and 'mean' in results[k]:
                # OA
                oa_vals.append(results[k]['mean']['oa'])
                oa_stds.append(results[k]['std']['oa'])
                valid_k_oa.append(k)
                
                # F1 (convert to percentage if stored as decimal)
                f1_mean = results[k]['mean']['f1']
                f1_std = results[k]['std']['f1']
                if f1_mean < 2:  # Stored as decimal
                    f1_mean *= 100
                    f1_std *= 100
                f1_vals.append(f1_mean)
                f1_stds.append(f1_std)
                valid_k_f1.append(k)
        
        # Plot OA
        if oa_vals:
            ax.errorbar(valid_k_oa, oa_vals, yerr=oa_stds,
                       marker='o', color=color, linewidth=2.5, markersize=10,
                       capsize=4, capthick=2, label='OA')
        
        # Plot F1 (dashed line, lighter color)
        if f1_vals:
            f1_color = color + '80'  # Add transparency
            ax.errorbar(valid_k_f1, f1_vals, yerr=f1_stds,
                       marker='s', color=color, linewidth=2.5, markersize=8,
                       capsize=4, capthick=2, linestyle='--', alpha=0.7, label='F1')
        
        # Mark baseline
        if baseline in results and 'mean' in results[baseline]:
            baseline_oa = results[baseline]['mean']['oa']
            ax.axvline(x=baseline, color='gray', linestyle='--', alpha=0.7)
            ax.scatter([baseline], [baseline_oa], s=200, c='gold', 
                      edgecolors='black', linewidths=2, zorder=5, marker='*')
        
        ax.set_xlabel(scale_name, fontsize=26, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=26, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', which='major', labelsize=11)
        ax.tick_params(axis='y', which='major', labelsize=22)
        ax.set_xticks(k_values)
        ax.legend(loc='best', fontsize=20)
        
        # Plot execution time if available
        if has_time_data and time_axes is not None and scale_name in time_results:
            tax = time_axes[idx]
            time_data = time_results[scale_name]
            
            times = []
            valid_k_time = []
            
            for k in k_values:
                if k in time_data:
                    times.append(time_data[k])
                    valid_k_time.append(k)
            
            if times:
                tax.plot(valid_k_time, times, marker='s', color=color, 
                        linewidth=2.5, markersize=10)
                
                # Mark baseline
                if baseline in time_data:
                    baseline_time = time_data[baseline]
                    tax.axvline(x=baseline, color='gray', linestyle='--', alpha=0.7)
                    tax.scatter([baseline], [baseline_time], s=200, c='gold', 
                              edgecolors='black', linewidths=2, zorder=5, marker='*')
            
            tax.set_xlabel(scale_name, fontsize=26, fontweight='bold')
            tax.set_ylabel('Avg Epoch Time (s)', fontsize=26, fontweight='bold')
            tax.grid(True, alpha=0.3)
            tax.tick_params(axis='x', which='major', labelsize=11)
            tax.tick_params(axis='y', which='major', labelsize=22)
            tax.set_xticks(k_values)
    
    plt.tight_layout()
    
    out_path = output_dir / 'kscale_sensitivity.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(out_path).replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved k-scale plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='K-Scale Ablation Study')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/kscale_ablation',
                        help='Output directory for results')
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
                        help='Evaluation only mode (load existing checkpoints)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a previous run directory (e.g., 20260314_144548). Skips models with existing checkpoints.')
    parser.add_argument('--retention', type=float, default=1.0,
                        help='Canopy retention rate (1.0 for 100%%, 0.05 for 5%%)')
    
    args = parser.parse_args()
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(script_dir, args.data_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    checkpoints_dir = output_dir / 'checkpoints'
    
    # Directory to import existing checkpoints from (for resuming)
    import_checkpoints_dir = None
    
    if args.eval_only:
        # Find latest results directory
        results_dirs = sorted(Path(args.output_dir).glob('*'))
        if not results_dirs:
            raise ValueError(f"No results found in {args.output_dir}. Run training first.")
        output_dir = results_dirs[-1]
        checkpoints_dir = output_dir / 'checkpoints'
        if not checkpoints_dir.exists():
            raise ValueError(f"Checkpoints not found in {checkpoints_dir}")
        logger.info(f"Eval-only mode: Loading from {output_dir}")
    elif args.resume_from:
        # Resume from a previous run - import checkpoints
        import_dir = Path(args.output_dir) / args.resume_from / 'checkpoints'
        if not import_dir.exists():
            raise ValueError(f"Checkpoints not found in {import_dir}")
        import_checkpoints_dir = import_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        # Copy existing checkpoints to new directory
        import shutil
        for ckpt_file in import_checkpoints_dir.glob('*.pth'):
            dest = checkpoints_dir / ckpt_file.name
            if not dest.exists():
                shutil.copy(ckpt_file, dest)
                logger.info(f"Imported checkpoint: {ckpt_file.name}")
        logger.info(f"Resumed from {args.resume_from}, imported checkpoints to {checkpoints_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
    
    # Generate CV splits
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    # Define scale experiments
    scale_experiments = [
        ('k_local', K_LOCAL_VALUES, BASELINE_K_SCALES[0]),
        ('k_intermediate', K_INTERMEDIATE_VALUES, BASELINE_K_SCALES[1]),
        ('k_global', K_GLOBAL_VALUES, BASELINE_K_SCALES[2]),
    ]
    
    total_experiments = sum(len(vals) for _, vals, _ in scale_experiments) * args.k_folds
    
    ret_pct = int(args.retention * 100)
    print("=" * 70)
    print("K-SCALE ABLATION STUDY (MS-DGCNN++ Default Asymmetric)" + 
          (" (EVAL ONLY)" if args.eval_only else ""))
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Baseline k-scales: {BASELINE_K_SCALES}")
    print(f"Retention: {ret_pct}%")
    print(f"Folds: {args.k_folds}")
    print(f"Total experiments: {total_experiments}")
    print("=" * 70)
    
    # Save config
    if not args.eval_only:
        config = {
            'experiment': 'K-Scale Ablation',
            'model': 'MS-DGCNN++ (Default Asymmetric)',
            'data_path': args.data_path,
            'k_folds': args.k_folds,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'num_points': args.num_points,
            'baseline_k_scales': BASELINE_K_SCALES,
            'retention': args.retention,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        }
        save_config(config, output_dir / 'config.json')
    
    # Results: {scale_name: {k_value: {fold: metrics}}}
    all_results = {}
    
    exp_counter = 0
    
    for scale_name, k_values, baseline in scale_experiments:
        print(f"\n{'='*60}")
        print(f"Scale: {scale_name}")
        print(f"{'='*60}")
        
        scale_results = {k: {'folds': []} for k in k_values}
        
        for k_value in k_values:
            # Determine k_scales for this experiment
            if scale_name == 'k_local':
                k_scales = [k_value, BASELINE_K_SCALES[1], BASELINE_K_SCALES[2]]
            elif scale_name == 'k_intermediate':
                k_scales = [BASELINE_K_SCALES[0], k_value, BASELINE_K_SCALES[2]]
            else:  # k_global
                k_scales = [BASELINE_K_SCALES[0], BASELINE_K_SCALES[1], k_value]
            
            print(f"\n  {scale_name}={k_value} (k_scales={k_scales})")
            
            for fold in range(args.k_folds):
                exp_counter += 1
                ckpt_path = checkpoints_dir / get_checkpoint_filename(scale_name, k_value, fold, args.retention)
                
                # Get dataloaders with density retention
                train_loader, val_loader, num_classes, _ = get_cv_dataloaders(
                    args.data_path, fold=fold, batch_size=args.batch_size,
                    num_points=args.num_points, seed=args.seed, k_folds=args.k_folds,
                    density_retention=args.retention, density_seed=args.seed
                )
                
                if args.eval_only:
                    # Load existing checkpoint
                    if not ckpt_path.exists():
                        logger.warning(f"    Fold {fold}: Checkpoint not found: {ckpt_path}")
                        continue
                    
                    model = create_default_asymmetric(num_classes, k_scales)
                    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    
                    metrics = evaluate_model(model, val_loader, device)
                    scale_results[k_value]['folds'].append(metrics)
                    print(f"    Fold {fold}: OA={metrics['oa']:.1f}%")
                else:
                    # Check if checkpoint exists
                    if ckpt_path.exists():
                        logger.info(f"    Fold {fold}: Loading existing checkpoint")
                        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                        metrics = checkpoint.get('metrics', {})
                        scale_results[k_value]['folds'].append(metrics)
                        print(f"    Fold {fold}: OA={metrics.get('oa', 0):.1f}% (cached)")
                        continue
                    
                    # Train model
                    print(f"    Fold {fold}: Training [{exp_counter}/{total_experiments}]...")
                    
                    # Set seed for reproducibility (different per fold)
                    set_seed(args.seed + fold)
                    
                    try:
                        model = create_default_asymmetric(num_classes, k_scales)
                        model = model.to(device)
                        
                        best_metrics, model = train_model(
                            model, train_loader, val_loader, device,
                            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                            save_path=str(ckpt_path), patience=args.patience
                        )
                        
                        scale_results[k_value]['folds'].append(best_metrics)
                        print(f"    Fold {fold}: OA={best_metrics.get('oa', 0):.1f}%")
                    except Exception as e:
                        logger.error(f"    Fold {fold}: Training failed: {e}")
                        continue
        
        # Compute mean and std for this scale
        for k_value in k_values:
            folds_data = scale_results[k_value]['folds']
            if folds_data:
                scale_results[k_value] = {
                    'mean': {
                        'oa': np.mean([f.get('oa', 0) for f in folds_data]),
                        'macc': np.mean([f.get('macc', 0) for f in folds_data]),
                        'f1': np.mean([f.get('f1', 0) for f in folds_data]),
                    },
                    'std': {
                        'oa': np.std([f.get('oa', 0) for f in folds_data]),
                        'macc': np.std([f.get('macc', 0) for f in folds_data]),
                        'f1': np.std([f.get('f1', 0) for f in folds_data]),
                    }
                }
        
        all_results[scale_name] = scale_results
    
    # Save results
    results_save = {
        'baseline_k_scales': BASELINE_K_SCALES,
        'k_folds': args.k_folds,
        'results': {}
    }
    for scale_name, scale_results in all_results.items():
        results_save['results'][scale_name] = {
            str(k): v for k, v in scale_results.items() if isinstance(v, dict) and 'mean' in v
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    # Measure execution time for each k-scale configuration (without retraining)
    print("\n" + "=" * 70)
    print("MEASURING EXECUTION TIME")
    print("=" * 70)
    
    time_results = {}
    
    for scale_name, k_values, baseline in scale_experiments:
        print(f"\n  Measuring {scale_name}...")
        time_results[scale_name] = {}
        
        for k_value in k_values:
            # Determine k_scales for this experiment
            if scale_name == 'k_local':
                k_scales = [k_value, BASELINE_K_SCALES[1], BASELINE_K_SCALES[2]]
            elif scale_name == 'k_intermediate':
                k_scales = [BASELINE_K_SCALES[0], k_value, BASELINE_K_SCALES[2]]
            else:  # k_global
                k_scales = [BASELINE_K_SCALES[0], BASELINE_K_SCALES[1], k_value]
            
            try:
                model = create_default_asymmetric(num_classes, k_scales)
                model = model.to(device)
                
                # Measure average epoch time
                epoch_time = measure_avg_epoch_time(model, train_loader, device, num_batches=10)
                time_results[scale_name][k_value] = epoch_time
                print(f"    {scale_name}={k_value}: {epoch_time:.2f}s/epoch")
                
                # Clean up
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            except Exception as e:
                logger.warning(f"    Failed to measure time for {scale_name}={k_value}: {e}")
    
    # Save time results
    with open(output_dir / 'time_results.json', 'w') as f:
        json.dump(time_results, f, indent=2)
    
    # Generate combined LaTeX table with time results
    latex_table = generate_combined_kscale_latex_table(all_results, scale_experiments, time_results)
    with open(output_dir / 'kscale_ablation.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plot with time curves
    plot_kscale_sensitivity(all_results, output_dir, time_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for scale_name, k_values, baseline in scale_experiments:
        if scale_name not in all_results:
            continue
        
        print(f"\n{scale_name}:")
        header = "  k  |"
        for k in k_values:
            header += f" {k:5d} |"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        row = " OA  |"
        for k in k_values:
            if k in all_results[scale_name] and 'mean' in all_results[scale_name][k]:
                oa = all_results[scale_name][k]['mean']['oa']
                row += f" {oa:5.1f} |"
            else:
                row += "    -- |"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/kscale_ablation.tex")
    print(f"  - {output_dir}/kscale_sensitivity.png/pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
