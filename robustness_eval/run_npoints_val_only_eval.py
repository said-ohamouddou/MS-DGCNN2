"""Validation-set n-points sensitivity (FPS subsampling; train at 1024)."""

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

NPOINTS_LIST = [128, 256, 512, 768, 1024]

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



try:
    from pointnet2_ops import pointnet2_utils
    HAS_POINTNET2 = True
except ImportError:
    HAS_POINTNET2 = False
    logger.warning("pointnet2_ops not available, using CPU FPS fallback")


def fps_subsample_cpu(points: np.ndarray, num_points: int, seed: int = None) -> np.ndarray:
    """CPU implementation of Farthest Point Sampling."""
    if seed is not None:
        np.random.seed(seed)
    
    N = points.shape[0]
    if N <= num_points:
        # Pad by repeating points
        if N < num_points:
            repeat_times = (num_points // N) + 1
            points = np.tile(points, (repeat_times, 1))[:num_points]
        return points
    
    selected = np.zeros(num_points, dtype=np.int64)
    distances = np.full(N, np.inf)
    
    # Start with random point
    selected[0] = np.random.randint(N)
    
    for i in range(1, num_points):
        last_selected = selected[i - 1]
        dist_to_last = np.sum((points - points[last_selected]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)
    
    return points[selected]


def fps_subsample(points: torch.Tensor, num_points: int, seed: int = None) -> torch.Tensor:
    """Subsample points using Farthest Point Sampling."""
    if points.shape[0] <= num_points:
        # Pad by repeating points if needed
        if points.shape[0] < num_points:
            repeat_times = (num_points // points.shape[0]) + 1
            points = points.repeat(repeat_times, 1)[:num_points]
        return points
    
    if HAS_POINTNET2 and points.is_cuda:
        # Use CUDA FPS
        points_batch = points.unsqueeze(0).contiguous()  # (1, N, 3)
        idx = pointnet2_utils.furthest_point_sample(points_batch, num_points)  # (1, num_points)
        points_sampled = pointnet2_utils.gather_operation(
            points_batch.transpose(1, 2).contiguous(),  # (1, 3, N)
            idx
        ).transpose(1, 2).squeeze(0)  # (num_points, 3)
        return points_sampled
    else:
        # CPU fallback
        points_np = points.cpu().numpy()
        sampled = fps_subsample_cpu(points_np, num_points, seed)
        return torch.from_numpy(sampled).float()



class FPSDataset(Dataset):
    """Wraps TreeSpeciesDatasetCV and applies FPS to subsample points."""
    
    def __init__(self, base_dataset, num_points: int = 1024, seed: int = None):
        """
        Args:
            base_dataset: Base dataset (TreeSpeciesDatasetCV)
            num_points: Number of points to sample via FPS
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
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
    
    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        
        # Handle [3, N] format
        if points.shape[0] == 3:
            points = points.T  # Convert to [N, 3]
            transposed = True
        else:
            transposed = False
        
        # Apply FPS subsampling
        seed = self.seed + idx if self.seed is not None else None
        points = fps_subsample(points, self.num_points, seed)
        
        # Convert back to original format
        if transposed:
            points = points.T
        
        return points, label


def get_fps_val_loader(data_path: str, fold: int, num_points: int,
                       batch_size: int = 16, seed: int = 42, k_folds: int = 5):
    """Create validation dataloader with FPS subsampling."""
    
    # Load with more points than needed for FPS
    base_num_points = max(2048, num_points * 2)
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val',
        num_points=base_num_points, augment=False,
        seed=seed, k_folds=k_folds
    )
    
    fps_dataset = FPSDataset(val_dataset, num_points=num_points, seed=seed)
    
    val_loader = DataLoader(
        fps_dataset, batch_size=batch_size, shuffle=False,
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



def generate_npoints_latex_table(results: Dict, npoints_list: List[int],
                                  model_names: List[str]) -> str:
    """Generate LaTeX table for N-points sensitivity results."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{N-Points Sensitivity (Val Only): OA (\%) at Different Point Counts}")
    lines.append(r"\label{tab:npoints_val_only}")
    
    col_spec = "l" + "c" * len(npoints_list)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Model"
    for npts in npoints_list:
        header += f" & {npts}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Find best values at each npoints
    best_at_npts = {}
    for npts in npoints_list:
        values = []
        for model_name in model_names:
            if npts in results[model_name]:
                values.append(results[model_name][npts]['mean']['oa'])
        best_at_npts[npts] = max(values) if values else 0
    
    # Data rows
    for model_name in model_names:
        row = model_name.replace('_', r'\_').replace('++', r'\texttt{++}')
        
        for npts in npoints_list:
            if npts in results[model_name]:
                mean = results[model_name][npts]['mean']['oa']
                std = results[model_name][npts]['std']['oa']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                if abs(mean - best_at_npts[npts]) < 0.01:
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



def plot_npoints_sensitivity(results: Dict, npoints_list: List[int],
                             model_names: List[str], out_path: str):
    """Plot accuracy vs number of points for all models."""
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
        valid_npts = []
        
        for npts in npoints_list:
            if npts in results[model_name]:
                accs.append(results[model_name][npts]['mean']['oa'])
                stds.append(results[model_name][npts]['std']['oa'])
                valid_npts.append(npts)
        
        if accs:
            ax.errorbar(valid_npts, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=8,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel('Number of Points (Validation)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('N-Points Sensitivity (Train @ 1024, Val Varies)', 
                 fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(npoints_list)
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    # Save PNG and PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='N-Points Sensitivity (Val Only) with FPS')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/clean_checkpoints/clean_cv',
                        help='Directory with pre-trained checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/npoints_val_only',
                        help='Output directory for results')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
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
        num_points=1024, seed=args.seed, k_folds=args.k_folds
    )
    
    print("=" * 70)
    print("N-POINTS SENSITIVITY EVALUATION (VAL ONLY) WITH FPS")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Point counts: {NPOINTS_LIST}")
    print(f"Folds: {args.k_folds}")
    print(f"FPS available (CUDA): {HAS_POINTNET2}")
    print("=" * 70)
    
    # Results: {model: {npoints: {fold: metrics}}}
    all_results = {m: {n: {'folds': []} for n in NPOINTS_LIST} for m in MODEL_CONFIGS.keys()}
    
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
            
            # Evaluate at each npoints
            for npts in NPOINTS_LIST:
                val_loader = get_fps_val_loader(
                    args.data_path, fold=fold, num_points=npts,
                    batch_size=args.batch_size, seed=args.seed, k_folds=args.k_folds
                )
                
                metrics = evaluate_model(model, val_loader, device)
                all_results[model_name][npts]['folds'].append(metrics)
                
                print(f"n={npts}:{metrics['oa']:.1f}", end=" ")
            
            print()
    
    # Compute mean and std across folds
    final_results = {m: {} for m in MODEL_CONFIGS.keys()}
    
    for model_name in MODEL_CONFIGS.keys():
        for npts in NPOINTS_LIST:
            folds_data = all_results[model_name][npts]['folds']
            if folds_data:
                final_results[model_name][npts] = {
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
        'npoints_list': NPOINTS_LIST,
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
    latex_table = generate_npoints_latex_table(
        final_results, NPOINTS_LIST, list(MODEL_CONFIGS.keys())
    )
    with open(output_dir / 'npoints_valonly_results.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plot
    plot_npoints_sensitivity(
        final_results, NPOINTS_LIST, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'npoints_valonly_curves.png')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    header = f"{'Model':<35}"
    for npts in NPOINTS_LIST:
        header += f" | {npts:4d}"
    print(header)
    print("-" * len(header))
    
    for model_name in MODEL_CONFIGS.keys():
        row = f"{model_name:<35}"
        for npts in NPOINTS_LIST:
            if npts in final_results[model_name]:
                oa = final_results[model_name][npts]['mean']['oa']
                row += f" | {oa:4.1f}"
            else:
                row += " |   --"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/npoints_valonly_results.tex")
    print(f"  - {output_dir}/npoints_valonly_curves.png/pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
