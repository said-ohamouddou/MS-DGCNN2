"""Few-shot / data-fraction training from scratch (5-fold CV)."""

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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from models.msdgcnn import create_ms_dgcnn
from models.pointm2ae.pointm2ae import create_pointm2ae
from utils.experiment_utils import (
    get_cv_dataloaders, train_model, set_seed, save_config,
    generate_cv_splits, TreeSpeciesDatasetCV
)
from utils.latex_utils import save_latex_table

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]

DATA_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]

MODEL_CONFIGS = {
    'DGCNN': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=False,
            use_normalized_features=False,
            fusion_type='concat_conv',
        ),
        'label': 'DGCNN'
    },
    'MS-DGCNN': {
        'factory': lambda nc: create_ms_dgcnn(
            num_classes=nc,
            k_scales=K_SCALES,
        ),
        'label': 'MS-DGCNN'
    },
    'Point-M2AE': {
        'factory': lambda nc: create_pointm2ae(
            num_classes=nc,
            ckpt_path='pointm2ae/pretrained_m2ae.pth'
        ),
        'label': 'Point-M2AE'
    },
    'MS-DGCNN++ (Raw-only)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=False,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Raw-only)'
    },
    'MS-DGCNN++ (Hybrid-everywhere)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=True,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Hybrid-everywhere)'
    },
    'MS-DGCNN++ (Default Asymmetric)': {
        'factory': lambda nc: create_model(
            num_classes=nc,
            k_scales=K_SCALES,
            use_multiscale=True,
            use_normalized_features=True,
            local_use_normalized=False,
            fusion_type='concat_conv',
        ),
        'label': 'MS-DGCNN++ (Default Asymmetric)'
    },
}



def get_stratified_subset_indices(labels: np.ndarray, fraction: float, 
                                   seed: int = 0) -> List[int]:
    """Get stratified subset indices maintaining class balance."""
    if fraction >= 1.0:
        return list(range(len(labels)))
    
    indices = np.arange(len(labels))
    num_classes = len(np.unique(labels))
    target_size = int(len(labels) * fraction)
    
    # If target size is too small for stratified sampling, use per-class minimum
    if target_size < num_classes * 2:
        np.random.seed(seed)
        subset_indices = []
        
        for c in np.unique(labels):
            class_indices = indices[labels == c]
            n_samples = max(1, int(len(class_indices) * fraction))
            n_samples = min(n_samples, len(class_indices))
            selected = np.random.choice(class_indices, n_samples, replace=False)
            subset_indices.extend(selected.tolist())
        
        return subset_indices
    
    # Stratified split
    subset_indices, _ = train_test_split(
        indices, 
        train_size=fraction,
        stratify=labels,
        random_state=seed
    )
    
    return subset_indices.tolist()


class StratifiedSubsetDataset(Dataset):
    """Dataset wrapper for stratified subset."""
    
    def __init__(self, base_dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        
        # Copy attributes
        if hasattr(base_dataset, 'num_classes'):
            self.num_classes = base_dataset.num_classes
        if hasattr(base_dataset, 'class_names'):
            self.class_names = base_dataset.class_names
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def get_few_shot_dataloaders(data_path: str, fold: int, fraction: float,
                              batch_size: int = 16, num_points: int = 1024,
                              seed: int = 42, k_folds: int = 5):
    """Create train and val dataloaders with specified training fraction."""
    
    # Get full training dataset for this fold
    train_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='train',
        num_points=num_points, augment=True,
        seed=seed, k_folds=k_folds
    )
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val',
        num_points=num_points, augment=False,
        seed=seed, k_folds=k_folds
    )
    
    # Get labels for stratified sampling
    labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if isinstance(label, torch.Tensor):
            labels.append(label.item())
        else:
            labels.append(label)
    labels = np.array(labels)
    
    # Get stratified subset
    subset_indices = get_stratified_subset_indices(labels, fraction, seed + fold)
    train_subset = StratifiedSubsetDataset(train_dataset, subset_indices)
    actual_train_size = len(subset_indices)
    
    # Adjust batch size for small datasets
    effective_batch_size = min(batch_size, actual_train_size)
    drop_last = actual_train_size > batch_size
    
    train_loader = DataLoader(
        train_subset, batch_size=effective_batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    
    num_classes = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else len(np.unique(labels))
    
    return train_loader, val_loader, num_classes, actual_train_size



def get_checkpoint_filename(model_name: str, fold: int, fraction: float) -> str:
    """Generate checkpoint filename."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    frac_str = f"{int(fraction*100):03d}pct"
    return f'{safe_name}_fold{fold}_{frac_str}.pth'



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



def generate_fewshot_latex_table(results: Dict, fractions: List[float],
                                  model_names: List[str]) -> str:
    """Generate LaTeX table for few-shot results."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Few-Shot Learning: OA (\%) at Different Training Data Fractions}")
    lines.append(r"\label{tab:few_shot}")
    
    col_spec = "l" + "c" * len(fractions)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Model"
    for frac in fractions:
        header += f" & {frac:.0%}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Find best values at each fraction
    best_at_frac = {}
    for frac in fractions:
        values = []
        for model_name in model_names:
            if frac in results[model_name]:
                values.append(results[model_name][frac]['mean']['oa'])
        best_at_frac[frac] = max(values) if values else 0
    
    # Data rows
    for model_name in model_names:
        row = model_name.replace('_', r'\_').replace('++', r'\texttt{++}')
        
        for frac in fractions:
            if frac in results[model_name]:
                mean = results[model_name][frac]['mean']['oa']
                std = results[model_name][frac]['std']['oa']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                if abs(mean - best_at_frac[frac]) < 0.01:
                    row += f" & \\textbf{{{cell}}}"
                else:
                    row += f" & {cell}"
            else:
                row += " & --"
        
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)



def plot_few_shot_curves(results: Dict, fractions: List[float],
                         model_names: List[str], out_path: str):
    """Plot accuracy vs data fraction for all models."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'legend.fontsize': 10,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    
    colors = {
        'DGCNN': '#E74C3C',
        'MS-DGCNN': '#3498DB',
        'Point-M2AE': '#F39C12',
        'MS-DGCNN++ (Raw-only)': '#9B59B6',
        'MS-DGCNN++ (Hybrid-everywhere)': '#1ABC9C',
        'MS-DGCNN++ (Default Asymmetric)': '#27AE60',
    }
    markers = {
        'DGCNN': 'o',
        'MS-DGCNN': 's',
        'Point-M2AE': 'X',
        'MS-DGCNN++ (Raw-only)': 'D',
        'MS-DGCNN++ (Hybrid-everywhere)': '^',
        'MS-DGCNN++ (Default Asymmetric)': 'v',
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in model_names:
        accs = []
        stds = []
        valid_fracs = []
        
        for frac in fractions:
            if frac in results[model_name]:
                accs.append(results[model_name][frac]['mean']['oa'])
                stds.append(results[model_name][frac]['std']['oa'])
                valid_fracs.append(frac * 100)
        
        if accs:
            ax.errorbar(valid_fracs, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=8,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel('Training Data Fraction (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Few-Shot Learning: Data Efficiency', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='Few-Shot Learning Evaluation')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--output_dir', type=str, default='results/few_shot',
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
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
    
    # Generate CV splits
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Get class info
    train_loader, val_loader, num_classes, _ = get_few_shot_dataloaders(
        args.data_path, fold=0, fraction=1.0,
        batch_size=args.batch_size, num_points=args.num_points,
        seed=args.seed, k_folds=args.k_folds
    )
    
    print("=" * 70)
    print("FEW-SHOT LEARNING EVALUATION" + (" (EVAL ONLY)" if args.eval_only else ""))
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Output: {output_dir}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Data fractions: {DATA_FRACTIONS}")
    print(f"Folds: {args.k_folds}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)
    
    # Save config
    if not args.eval_only:
        config = {
            'experiment': 'Few-Shot Learning',
            'data_path': args.data_path,
            'k_folds': args.k_folds,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'num_points': args.num_points,
            'data_fractions': DATA_FRACTIONS,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat()
        }
        save_config(config, output_dir / 'config.json')
    
    # Results: {model: {fraction: {fold: metrics}}}
    all_results = {m: {f: {'folds': []} for f in DATA_FRACTIONS} for m in MODEL_CONFIGS.keys()}
    
    # Train/evaluate each model at each fraction
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_config['label']}")
        print(f"{'='*60}")
        
        for fraction in DATA_FRACTIONS:
            print(f"\n  Data fraction: {fraction:.0%}")
            
            for fold in range(args.k_folds):
                ckpt_path = checkpoints_dir / get_checkpoint_filename(model_name, fold, fraction)
                
                # Get dataloaders
                train_loader, val_loader, num_classes, train_size = get_few_shot_dataloaders(
                    args.data_path, fold=fold, fraction=fraction,
                    batch_size=args.batch_size, num_points=args.num_points,
                    seed=args.seed, k_folds=args.k_folds
                )
                
                # Check if checkpoint exists and is valid
                checkpoint_valid = False
                cached_metrics = None
                
                if ckpt_path.exists():
                    try:
                        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                        if 'model_state_dict' in checkpoint and 'metrics' in checkpoint:
                            cached_metrics = checkpoint.get('metrics', {})
                            if cached_metrics.get('oa', 0) > 0:  # Valid metrics
                                checkpoint_valid = True
                    except Exception as e:
                        logger.warning(f"    Fold {fold}: Corrupted checkpoint, will retrain: {e}")
                        checkpoint_valid = False
                
                if args.eval_only:
                    # Load existing checkpoint for evaluation
                    if not checkpoint_valid:
                        logger.warning(f"    Fold {fold}: Checkpoint not found or invalid: {ckpt_path}")
                        continue
                    
                    model = model_config['factory'](num_classes)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    
                    metrics = evaluate_model(model, val_loader, device)
                    all_results[model_name][fraction]['folds'].append(metrics)
                    print(f"    Fold {fold}: OA={metrics['oa']:.1f}%")
                else:
                    # Training mode - skip if valid checkpoint exists
                    if checkpoint_valid:
                        all_results[model_name][fraction]['folds'].append(cached_metrics)
                        print(f"    Fold {fold}: OA={cached_metrics.get('oa', 0):.1f}% (cached)")
                        continue
                    
                    # Train model
                    print(f"    Fold {fold}: Training ({train_size} samples)...")
                    
                    # Set seed for reproducibility (different per fold)
                    set_seed(args.seed + fold)
                    
                    try:
                        model = model_config['factory'](num_classes)
                        model = model.to(device)
                        
                        best_metrics, model = train_model(
                            model, train_loader, val_loader, device,
                            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                            save_path=str(ckpt_path), patience=args.patience
                        )
                        
                        all_results[model_name][fraction]['folds'].append(best_metrics)
                        print(f"    Fold {fold}: OA={best_metrics.get('oa', 0):.1f}%")
                    except Exception as e:
                        logger.error(f"    Fold {fold}: Training failed: {e}")
                        continue
    
    # Compute mean and std across folds
    final_results = {m: {} for m in MODEL_CONFIGS.keys()}
    
    for model_name in MODEL_CONFIGS.keys():
        for fraction in DATA_FRACTIONS:
            folds_data = all_results[model_name][fraction]['folds']
            if folds_data:
                final_results[model_name][fraction] = {
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
    
    # Save results
    results_save = {
        'data_fractions': DATA_FRACTIONS,
        'model_names': list(MODEL_CONFIGS.keys()),
        'k_folds': args.k_folds,
        'results': {}
    }
    for m in MODEL_CONFIGS.keys():
        results_save['results'][m] = {
            str(k): v for k, v in final_results[m].items()
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    # Generate LaTeX table
    latex_table = generate_fewshot_latex_table(
        final_results, DATA_FRACTIONS, list(MODEL_CONFIGS.keys())
    )
    with open(output_dir / 'few_shot_results.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plot
    plot_few_shot_curves(
        final_results, DATA_FRACTIONS, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'few_shot_curves.png')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    header = f"{'Model':<35}"
    for frac in DATA_FRACTIONS:
        header += f" | {frac:4.0%}"
    print(header)
    print("-" * len(header))
    
    for model_name in MODEL_CONFIGS.keys():
        row = f"{model_name:<35}"
        for frac in DATA_FRACTIONS:
            if frac in final_results[model_name]:
                oa = final_results[model_name][frac]['mean']['oa']
                row += f" | {oa:4.1f}"
            else:
                row += " |   --"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/few_shot_results.tex")
    print(f"  - {output_dir}/few_shot_curves.png/pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
