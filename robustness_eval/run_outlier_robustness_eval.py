"""Outlier robustness evaluation (validation-set outliers)."""

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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.msdgcnn2 import create_model
from models.msdgcnn import create_ms_dgcnn
from models.pointm2ae.pointm2ae import create_pointm2ae
from utils.experiment_utils import (
    get_cv_dataloaders, set_seed, generate_cv_splits,
    TreeSpeciesDatasetCV
)
from utils.latex_utils import save_latex_table

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]

OUTLIER_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

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



class OutlierDataset(Dataset):
    """
    Wraps TreeSpeciesDatasetCV and injects random outlier points.
    
    Outliers are generated within an expanded bounding box of the original
    point cloud, simulating multi-path reflections, sensor artifacts, etc.
    """
    
    def __init__(self, base_dataset, outlier_rate: float = 0.0, seed: int = None):
        """
        Args:
            base_dataset: Base dataset (TreeSpeciesDatasetCV)
            outlier_rate: Fraction of points to replace with outliers (0.0 to 0.5)
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.outlier_rate = outlier_rate
        self.seed = seed
        
        # Copy attributes from base dataset
        if hasattr(base_dataset, 'class_names'):
            self.class_names = base_dataset.class_names
        if hasattr(base_dataset, 'classes'):
            self.classes = base_dataset.classes
        if hasattr(base_dataset, 'num_classes'):
            self.num_classes = base_dataset.num_classes
    
    def __len__(self):
        return len(self.base_dataset)
    
    def inject_outliers(self, points, idx):
        """Inject random outlier points."""
        if self.outlier_rate <= 0:
            return points
        
        # Use deterministic seed for reproducibility
        if self.seed is not None:
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random
        
        # Handle tensor input
        if isinstance(points, torch.Tensor):
            points_np = points.numpy().copy()
            is_tensor = True
        else:
            points_np = points.copy()
            is_tensor = False
        
        # Handle [3, N] format
        if points_np.shape[0] == 3:
            points_np = points_np.T  # Convert to [N, 3]
            transposed = True
        else:
            transposed = False
        
        N = points_np.shape[0]
        num_outliers = int(N * self.outlier_rate)
        
        if num_outliers > 0:
            # Get bounding box of original points
            min_vals = points_np.min(axis=0)
            max_vals = points_np.max(axis=0)
            
            # Expand bounding box (1.5x) for outliers
            center = (min_vals + max_vals) / 2
            extent = (max_vals - min_vals) / 2 * 1.5
            
            # Generate random outliers within expanded bounding box
            outliers = rng.uniform(-1, 1, (num_outliers, 3))
            outliers = center + outliers * extent
            
            # Replace some original points with outliers (to keep N constant)
            replace_indices = rng.choice(N, num_outliers, replace=False)
            points_np[replace_indices] = outliers
        
        # Convert back to original format
        if transposed:
            points_np = points_np.T
        
        if is_tensor:
            points_np = torch.from_numpy(points_np).float()
        
        return points_np
    
    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        
        # Inject outliers
        points = self.inject_outliers(points, idx)
        
        return points, label


def get_outlier_val_loader(data_path: str, fold: int, outlier_rate: float,
                           batch_size: int = 16, num_points: int = 1024,
                           seed: int = 42, k_folds: int = 5):
    """Create validation dataloader with outlier injection."""
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val',
        num_points=num_points, augment=False,
        seed=seed, k_folds=k_folds
    )
    
    outlier_dataset = OutlierDataset(
        val_dataset, outlier_rate=outlier_rate, seed=seed
    )
    
    val_loader = DataLoader(
        outlier_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    return val_loader



def get_checkpoint_filename(model_name: str, fold: int) -> str:
    """Generate checkpoint filename matching train_clean_checkpoints.py."""
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    return f'{safe_name}_fold{fold}.pth'


def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: str):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



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



def generate_outlier_latex_table(results: Dict, outlier_rates: List[float],
                                  model_names: List[str]) -> str:
    """Generate LaTeX table for outlier robustness results."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Outlier Injection Robustness: OA (\%) at Different Outlier Rates}")
    lines.append(r"\label{tab:outlier_robustness}")
    
    col_spec = "l" + "c" * len(outlier_rates)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Model"
    for rate in outlier_rates:
        header += f" & {rate:.0%}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Find best values at each outlier rate
    best_at_rate = {}
    for rate in outlier_rates:
        values = []
        for model_name in model_names:
            if rate in results[model_name]:
                values.append(results[model_name][rate]['mean']['oa'])
        best_at_rate[rate] = max(values) if values else 0
    
    # Data rows
    for model_name in model_names:
        row = model_name.replace('_', r'\_').replace('++', r'\texttt{++}')
        
        for rate in outlier_rates:
            if rate in results[model_name]:
                mean = results[model_name][rate]['mean']['oa']
                std = results[model_name][rate]['std']['oa']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                if abs(mean - best_at_rate[rate]) < 0.01:
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



def plot_outlier_robustness(results: Dict, outlier_rates: List[float],
                            model_names: List[str], out_path: str):
    """Plot accuracy vs outlier rate for all models."""
    import matplotlib.pyplot as plt
    
    # Publication-ready settings
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
        valid_rates = []
        
        for rate in outlier_rates:
            if rate in results[model_name]:
                accs.append(results[model_name][rate]['mean']['oa'])
                stds.append(results[model_name][rate]['std']['oa'])
                valid_rates.append(rate * 100)  # Convert to percentage
        
        if accs:
            ax.errorbar(valid_rates, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=8,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel('Outlier Injection Rate (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Outlier Robustness (Train Clean, Test with Outliers)', 
                 fontsize=18, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 26)
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    # Save PNG and PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='Outlier Robustness Evaluation')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/clean_checkpoints/clean_cv',
                        help='Directory with pre-trained checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/outlier_robustness',
                        help='Output directory for results')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(script_dir, args.data_path)
    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.join(script_dir, args.checkpoint_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {args.checkpoint_dir}\n"
                        f"Run train_clean_checkpoints.py first.")
    
    # Generate CV splits
    generate_cv_splits(args.data_path, k_folds=args.k_folds, seed=args.seed)
    
    # Get class info
    train_loader, val_loader, num_classes, class_names = get_cv_dataloaders(
        args.data_path, fold=0, batch_size=args.batch_size,
        num_points=args.num_points, seed=args.seed, k_folds=args.k_folds
    )
    
    print("=" * 70)
    print("OUTLIER ROBUSTNESS EVALUATION")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Outlier rates: {OUTLIER_RATES}")
    print(f"Folds: {args.k_folds}")
    print("=" * 70)
    
    # Results: {model: {outlier_rate: {fold: metrics}}}
    all_results = {m: {r: {'folds': []} for r in OUTLIER_RATES} for m in MODEL_CONFIGS.keys()}
    
    # Evaluate each model
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_config['label']}")
        print(f"{'='*60}")
        
        for fold in range(args.k_folds):
            # Load checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, 
                                     get_checkpoint_filename(model_name, fold))
            
            if not os.path.exists(ckpt_path):
                logger.warning(f"  Fold {fold}: Checkpoint not found: {ckpt_path}")
                continue
            
            # Create and load model
            try:
                model = model_config['factory'](num_classes)
                model = load_model_checkpoint(model, ckpt_path, device)
                model = model.to(device)
            except Exception as e:
                logger.error(f"  Fold {fold}: Failed to load model: {e}")
                continue
            
            print(f"  Fold {fold}:", end=" ")
            
            # Evaluate at each outlier rate
            for outlier_rate in OUTLIER_RATES:
                val_loader = get_outlier_val_loader(
                    args.data_path, fold=fold, outlier_rate=outlier_rate,
                    batch_size=args.batch_size, num_points=args.num_points,
                    seed=args.seed, k_folds=args.k_folds
                )
                
                metrics = evaluate_model(model, val_loader, device)
                all_results[model_name][outlier_rate]['folds'].append(metrics)
                
                print(f"r={outlier_rate:.0%}:{metrics['oa']:.1f}", end=" ")
            
            print()
    
    # Compute mean and std across folds
    final_results = {m: {} for m in MODEL_CONFIGS.keys()}
    
    for model_name in MODEL_CONFIGS.keys():
        for outlier_rate in OUTLIER_RATES:
            folds_data = all_results[model_name][outlier_rate]['folds']
            if folds_data:
                final_results[model_name][outlier_rate] = {
                    'mean': {
                        'oa': np.mean([f['oa'] for f in folds_data]),
                        'macc': np.mean([f['macc'] for f in folds_data]),
                        'f1': np.mean([f['f1'] for f in folds_data]),
                    },
                    'std': {
                        'oa': np.std([f['oa'] for f in folds_data]),
                        'macc': np.std([f['macc'] for f in folds_data]),
                        'f1': np.std([f['f1'] for f in folds_data]),
                    }
                }
    
    # Save results
    results_save = {
        'outlier_rates': OUTLIER_RATES,
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
    latex_table = generate_outlier_latex_table(
        final_results, OUTLIER_RATES, list(MODEL_CONFIGS.keys())
    )
    with open(output_dir / 'outlier_results.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plot
    plot_outlier_robustness(
        final_results, OUTLIER_RATES, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'outlier_curves.png')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    header = f"{'Model':<35}"
    for rate in OUTLIER_RATES:
        header += f" | {rate:4.0%}"
    print(header)
    print("-" * len(header))
    
    for model_name in MODEL_CONFIGS.keys():
        row = f"{model_name:<35}"
        for rate in OUTLIER_RATES:
            if rate in final_results[model_name]:
                oa = final_results[model_name][rate]['mean']['oa']
                row += f" | {oa:4.1f}"
            else:
                row += " |   --"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/outlier_results.tex")
    print(f"  - {output_dir}/outlier_curves.png/pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
