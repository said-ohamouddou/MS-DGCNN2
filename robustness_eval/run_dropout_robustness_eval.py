"""Upper-canopy density dropout on the validation set (multiple retention rates)."""

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
    TreeSpeciesDatasetCV, evaluate
)
from utils.latex_utils import save_latex_table
from utils.plot_utils import create_bar_comparison

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]

RETENTION_RATES = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]

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



class DensityDropoutDataset(Dataset):
    """
    Wraps TreeSpeciesDatasetCV and applies upper canopy density-based dropout.
    
    This simulates realistic LiDAR occlusion where upper canopy points
    are more likely to be missing due to scan angle and occlusion.
    """
    
    def __init__(self, base_dataset, retention_rate: float = 1.0, 
                 num_points: int = 1024, seed: int = None):
        """
        Args:
            base_dataset: Base dataset (TreeSpeciesDatasetCV)
            retention_rate: Fraction of upper canopy points to keep (0.0 to 1.0)
            num_points: Number of points to output
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.retention_rate = retention_rate
        self.num_points = num_points
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
    
    def apply_density_dropout(self, points, idx):
        """Apply density dropout to upper canopy points."""
        if self.retention_rate >= 1.0:
            return points
        
        # Use deterministic seed for reproducibility
        if self.seed is not None:
            rng = np.random.RandomState(self.seed + idx)
        else:
            rng = np.random
        
        # Handle tensor input
        if isinstance(points, torch.Tensor):
            points_np = points.numpy()
            is_tensor = True
        else:
            points_np = points
            is_tensor = False
        
        # Handle [3, N] format
        if points_np.shape[0] == 3:
            points_np = points_np.T  # Convert to [N, 3]
            transposed = True
        else:
            transposed = False
        
        # Compute median height
        median_z = np.median(points_np[:, 2])
        
        # Split into upper and lower
        upper_mask = points_np[:, 2] > median_z
        lower_mask = ~upper_mask
        
        upper_points = points_np[upper_mask]
        lower_points = points_np[lower_mask]
        
        # Subsample upper canopy
        n_upper = len(upper_points)
        n_keep = max(1, int(n_upper * self.retention_rate))
        
        if n_keep < n_upper:
            keep_idx = rng.choice(n_upper, n_keep, replace=False)
            upper_points = upper_points[keep_idx]
        
        # Combine
        combined = np.vstack([lower_points, upper_points])
        
        # Resample to num_points if needed
        if len(combined) < self.num_points:
            # Duplicate points
            repeat_factor = (self.num_points // len(combined)) + 1
            combined = np.tile(combined, (repeat_factor, 1))[:self.num_points]
        elif len(combined) > self.num_points:
            idx = rng.choice(len(combined), self.num_points, replace=False)
            combined = combined[idx]
        
        # Convert back to original format
        if transposed:
            combined = combined.T
        
        if is_tensor:
            combined = torch.from_numpy(combined).float()
        
        return combined
    
    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        
        # Apply density dropout
        points = self.apply_density_dropout(points, idx)
        
        return points, label


def get_dropout_val_loader(data_path: str, fold: int, retention_rate: float,
                           batch_size: int = 16, num_points: int = 1024,
                           seed: int = 42, k_folds: int = 5):
    """Create validation dataloader with density dropout."""
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val',
        num_points=num_points, augment=False,
        seed=seed, k_folds=k_folds
    )
    
    dropout_dataset = DensityDropoutDataset(
        val_dataset, retention_rate=retention_rate,
        num_points=num_points, seed=seed
    )
    
    val_loader = DataLoader(
        dropout_dataset, batch_size=batch_size, shuffle=False,
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



def generate_dropout_latex_table(results: Dict, retention_rates: List[float],
                                  model_names: List[str]) -> str:
    """Generate LaTeX table for dropout robustness results."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Upper Canopy Density Dropout Robustness: OA (\%) at Different Retention Rates}")
    lines.append(r"\label{tab:dropout_robustness}")
    
    col_spec = "l" + "c" * len(retention_rates)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Model"
    for ret in retention_rates:
        header += f" & {ret:.0%}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Find best values at each retention rate
    best_at_rate = {}
    for ret in retention_rates:
        values = []
        for model_name in model_names:
            if ret in results[model_name]:
                values.append(results[model_name][ret]['mean']['oa'])
        best_at_rate[ret] = max(values) if values else 0
    
    # Data rows
    for model_name in model_names:
        row = model_name.replace('_', r'\_').replace('++', r'\texttt{++}')
        
        for ret in retention_rates:
            if ret in results[model_name]:
                mean = results[model_name][ret]['mean']['oa']
                std = results[model_name][ret]['std']['oa']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                if abs(mean - best_at_rate[ret]) < 0.01:
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



def plot_dropout_robustness(results: Dict, retention_rates: List[float],
                            model_names: List[str], out_path: str):
    """Plot accuracy vs retention rate for all models."""
    import matplotlib.pyplot as plt
    
    # Publication-ready settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'legend.fontsize': 11,
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
        
        for ret in retention_rates:
            if ret in results[model_name]:
                accs.append(results[model_name][ret]['mean']['oa'])
                stds.append(results[model_name][ret]['std']['oa'])
                valid_rates.append(ret * 100)  # Convert to percentage
        
        if accs:
            ax.errorbar(valid_rates, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=10,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel('Upper Canopy Retention Rate (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Density Dropout Robustness (Train Clean, Test Degraded)', 
                 fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.invert_xaxis()  # Higher retention on left
    
    plt.tight_layout()
    
    # Save PNG and PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='Density Dropout Robustness Evaluation')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/clean_checkpoints/clean_cv',
                        help='Directory with pre-trained checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/dropout_robustness',
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
    print("DENSITY DROPOUT ROBUSTNESS EVALUATION")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Retention rates: {RETENTION_RATES}")
    print(f"Folds: {args.k_folds}")
    print("=" * 70)
    
    # Results: {model: {retention: {fold: metrics}}}
    all_results = {m: {r: {'folds': []} for r in RETENTION_RATES} for m in MODEL_CONFIGS.keys()}
    
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
            
            # Evaluate at each retention rate
            for retention_rate in RETENTION_RATES:
                val_loader = get_dropout_val_loader(
                    args.data_path, fold=fold, retention_rate=retention_rate,
                    batch_size=args.batch_size, num_points=args.num_points,
                    seed=args.seed, k_folds=args.k_folds
                )
                
                metrics = evaluate_model(model, val_loader, device)
                all_results[model_name][retention_rate]['folds'].append(metrics)
                
                print(f"r={retention_rate:.0%}:{metrics['oa']:.1f}%", end=" ")
            
            print()
    
    # Compute mean and std across folds
    final_results = {m: {} for m in MODEL_CONFIGS.keys()}
    
    for model_name in MODEL_CONFIGS.keys():
        for retention_rate in RETENTION_RATES:
            folds_data = all_results[model_name][retention_rate]['folds']
            if folds_data:
                final_results[model_name][retention_rate] = {
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
        'retention_rates': RETENTION_RATES,
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
    latex_table = generate_dropout_latex_table(
        final_results, RETENTION_RATES, list(MODEL_CONFIGS.keys())
    )
    with open(output_dir / 'dropout_results.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plot
    plot_dropout_robustness(
        final_results, RETENTION_RATES, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'dropout_robustness.png')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    header = f"{'Model':<35}"
    for ret in RETENTION_RATES:
        header += f" | {ret:5.0%}"
    print(header)
    print("-" * len(header))
    
    for model_name in MODEL_CONFIGS.keys():
        row = f"{model_name:<35}"
        for ret in RETENTION_RATES:
            if ret in final_results[model_name]:
                oa = final_results[model_name][ret]['mean']['oa']
                row += f" | {oa:5.1f}"
            else:
                row += " |    --"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/dropout_results.tex")
    print(f"  - {output_dir}/dropout_robustness.png")
    print(f"  - {output_dir}/dropout_robustness.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
