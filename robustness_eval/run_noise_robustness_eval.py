"""Gaussian noise robustness on the validation set with SNR metrics."""

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
from scipy.spatial.distance import cdist
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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



K_SCALES = [5, 20, 30]

NOISE_LEVELS_MM = [0, 0.5, 1, 2, 5, 10, 20, 50, 75, 100, 150]

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



class NoisyDataset(Dataset):
    """
    Wraps TreeSpeciesDatasetCV and adds Gaussian noise to point clouds.
    """
    
    def __init__(self, base_dataset, noise_sigma_mm: float = 0.0, seed: int = None):
        """
        Args:
            base_dataset: Base dataset (TreeSpeciesDatasetCV)
            noise_sigma_mm: Noise standard deviation in mm
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.noise_sigma_mm = noise_sigma_mm
        # Convert mm to normalized units (assuming data is normalized to unit sphere)
        self.noise_sigma = noise_sigma_mm / 1000.0
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
        
        if self.noise_sigma > 0:
            # Use deterministic seed for reproducibility
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + idx)
                noise = torch.from_numpy(
                    rng.randn(*points.shape).astype(np.float32)
                ) * self.noise_sigma
            else:
                noise = torch.randn_like(points) * self.noise_sigma
            points = points + noise
        
        return points, label


def get_noisy_val_loader(data_path: str, fold: int, noise_sigma_mm: float,
                         batch_size: int = 16, num_points: int = 1024,
                         seed: int = 42, k_folds: int = 5):
    """Create validation dataloader with Gaussian noise."""
    
    val_dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val',
        num_points=num_points, augment=False,
        seed=seed, k_folds=k_folds
    )
    
    noisy_dataset = NoisyDataset(
        val_dataset, noise_sigma_mm=noise_sigma_mm, seed=seed
    )
    
    val_loader = DataLoader(
        noisy_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    return val_loader



def compute_snr_stats(data_path: str, fold: int, noise_sigma_mm: float,
                      k1: int = 5, k2: int = 20, num_points: int = 1024,
                      seed: int = 42, k_folds: int = 5, num_samples: int = 50) -> Dict:
    """
    Compute empirical SNR at k1 and k2 scales.
    
    SNR = mean_neighbor_distance / noise_sigma
    """
    dataset = TreeSpeciesDatasetCV(
        data_path, fold=fold, partition='val', num_points=num_points,
        augment=False, seed=seed, k_folds=k_folds
    )
    
    s1_distances = []
    s2_distances = []
    
    for idx in range(min(num_samples, len(dataset))):
        points, _ = dataset[idx]
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        
        # Handle [3, N] format
        if points.shape[0] == 3:
            points = points.T
        
        # Compute pairwise distances
        dists = cdist(points, points)
        
        for i in range(len(points)):
            sorted_dists = np.sort(dists[i])[1:]  # Exclude self
            
            # k1-neighbor mean distance
            if len(sorted_dists) >= k1:
                s1_distances.append(np.mean(sorted_dists[:k1]))
            
            # k2-neighbor mean distance
            if len(sorted_dists) >= k2:
                s2_distances.append(np.mean(sorted_dists[:k2]))
    
    s1_mean = np.mean(s1_distances)
    s2_mean = np.mean(s2_distances)
    
    # SNR = mean_distance / sigma (convert sigma from mm to normalized units)
    sigma_normalized = noise_sigma_mm / 1000.0 if noise_sigma_mm > 0 else 1e-10
    
    snr1 = s1_mean / sigma_normalized if noise_sigma_mm > 0 else float('inf')
    snr2 = s2_mean / sigma_normalized if noise_sigma_mm > 0 else float('inf')
    
    return {
        's1_mean': s1_mean,
        's2_mean': s2_mean,
        'snr1': snr1,
        'snr2': snr2,
        'noise_sigma_mm': noise_sigma_mm
    }



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



def generate_noise_latex_table(results: Dict, noise_levels: List[float],
                               model_names: List[str]) -> str:
    """Generate LaTeX table for noise robustness results."""
    lines = []
    
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Noise Robustness: OA (\%) at Different Noise Levels (mm)}")
    lines.append(r"\label{tab:noise_robustness}")
    
    # Select subset of noise levels for table
    display_levels = [0, 1, 5, 10, 20, 50, 100, 150]
    display_levels = [n for n in display_levels if n in noise_levels]
    
    col_spec = "l" + "c" * len(display_levels)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Header
    header = "Model"
    for noise in display_levels:
        header += f" & {noise}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    
    # Find best values at each noise level
    best_at_level = {}
    for noise in display_levels:
        values = []
        for model_name in model_names:
            if noise in results[model_name]:
                values.append(results[model_name][noise]['mean']['oa'])
        best_at_level[noise] = max(values) if values else 0
    
    # Data rows
    for model_name in model_names:
        row = model_name.replace('_', r'\_').replace('++', r'\texttt{++}')
        
        for noise in display_levels:
            if noise in results[model_name]:
                mean = results[model_name][noise]['mean']['oa']
                std = results[model_name][noise]['std']['oa']
                cell = f"{mean:.1f}$\\pm${std:.1f}"
                if abs(mean - best_at_level[noise]) < 0.01:
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



def plot_noise_vs_accuracy(results: Dict, noise_levels: List[float],
                           model_names: List[str], out_path: str):
    """Plot accuracy vs noise level for all models."""
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
        valid_levels = []
        
        for noise in noise_levels:
            if noise in results[model_name]:
                accs.append(results[model_name][noise]['mean']['oa'])
                stds.append(results[model_name][noise]['std']['oa'])
                valid_levels.append(noise)
        
        if accs:
            ax.errorbar(valid_levels, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=8,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel('Noise Level σ (mm)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Noise Robustness (Train Clean, Test Noisy)', 
                 fontsize=18, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('symlog', linthresh=1)  # Symmetric log scale for noise
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    # Save PNG and PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved plot to: {out_path}")


def plot_accuracy_vs_snr(results: Dict, snr_data: Dict, model_names: List[str], 
                         out_path: str, snr_type: str = 'snr2'):
    """Plot accuracy vs SNR (at k2 scale) for all models."""
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
        snrs = []
        
        for noise_mm, snr_info in snr_data.items():
            if noise_mm in results[model_name] and snr_info[snr_type] != float('inf'):
                accs.append(results[model_name][noise_mm]['mean']['oa'])
                stds.append(results[model_name][noise_mm]['std']['oa'])
                snrs.append(snr_info[snr_type])
        
        if accs and snrs:
            # Sort by SNR
            sorted_idx = np.argsort(snrs)
            snrs = [snrs[i] for i in sorted_idx]
            accs = [accs[i] for i in sorted_idx]
            stds = [stds[i] for i in sorted_idx]
            
            ax.errorbar(snrs, accs, yerr=stds,
                       marker=markers.get(model_name, 'o'),
                       color=colors.get(model_name, 'gray'),
                       label=model_name, linewidth=2.5, markersize=8,
                       capsize=3, capthick=1.5)
    
    ax.set_xlabel(f'SNR at k₂={K_SCALES[1]}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Accuracy vs Signal-to-Noise Ratio', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    plt.tight_layout()
    
    # Save PNG and PDF
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    pdf_path = out_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved SNR plot to: {out_path}")



def main():
    parser = argparse.ArgumentParser(description='Noise Robustness Evaluation with SNR')
    parser.add_argument('--data_path', type=str, default='data/STPCTLC',
                        help='Path to STPCTLC data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/clean_checkpoints/clean_cv',
                        help='Directory with pre-trained checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/noise_robustness',
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
    print("NOISE ROBUSTNESS EVALUATION WITH SNR")
    print("=" * 70)
    print(f"Data: {args.data_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Noise levels (mm): {NOISE_LEVELS_MM}")
    print(f"Folds: {args.k_folds}")
    print("=" * 70)
    
    # Compute SNR at each noise level (using fold 0)
    print("\nComputing SNR statistics...")
    snr_data = {}
    for noise_mm in NOISE_LEVELS_MM:
        if noise_mm > 0:
            snr_stats = compute_snr_stats(
                args.data_path, fold=0, noise_sigma_mm=noise_mm,
                k1=K_SCALES[0], k2=K_SCALES[1], num_points=args.num_points,
                seed=args.seed, k_folds=args.k_folds
            )
            snr_data[noise_mm] = snr_stats
            print(f"  σ={noise_mm}mm: SNR₁={snr_stats['snr1']:.2f}, SNR₂={snr_stats['snr2']:.2f}")
        else:
            snr_data[noise_mm] = {'snr1': float('inf'), 'snr2': float('inf')}
    
    # Results: {model: {noise_mm: {fold: metrics}}}
    all_results = {m: {n: {'folds': []} for n in NOISE_LEVELS_MM} for m in MODEL_CONFIGS.keys()}
    
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
            
            # Evaluate at each noise level
            for noise_mm in NOISE_LEVELS_MM:
                val_loader = get_noisy_val_loader(
                    args.data_path, fold=fold, noise_sigma_mm=noise_mm,
                    batch_size=args.batch_size, num_points=args.num_points,
                    seed=args.seed, k_folds=args.k_folds
                )
                
                metrics = evaluate_model(model, val_loader, device)
                all_results[model_name][noise_mm]['folds'].append(metrics)
                
                print(f"σ={noise_mm}:{metrics['oa']:.1f}", end=" ")
            
            print()
    
    # Compute mean and std across folds
    final_results = {m: {} for m in MODEL_CONFIGS.keys()}
    
    for model_name in MODEL_CONFIGS.keys():
        for noise_mm in NOISE_LEVELS_MM:
            folds_data = all_results[model_name][noise_mm]['folds']
            if folds_data:
                final_results[model_name][noise_mm] = {
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
        'noise_levels_mm': NOISE_LEVELS_MM,
        'model_names': list(MODEL_CONFIGS.keys()),
        'k_folds': args.k_folds,
        'snr_data': {str(k): v for k, v in snr_data.items()},
        'results': {}
    }
    for m in MODEL_CONFIGS.keys():
        results_save['results'][m] = {
            str(k): v for k, v in final_results[m].items()
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    # Generate LaTeX table
    latex_table = generate_noise_latex_table(
        final_results, NOISE_LEVELS_MM, list(MODEL_CONFIGS.keys())
    )
    with open(output_dir / 'noise_results.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate plots
    plot_noise_vs_accuracy(
        final_results, NOISE_LEVELS_MM, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'noise_vs_accuracy.png')
    )
    
    plot_accuracy_vs_snr(
        final_results, snr_data, list(MODEL_CONFIGS.keys()),
        str(output_dir / 'accuracy_vs_snr.png')
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    display_levels = [0, 5, 20, 50, 100, 150]
    header = f"{'Model':<35}"
    for noise in display_levels:
        header += f" | σ={noise:3d}"
    print(header)
    print("-" * len(header))
    
    for model_name in MODEL_CONFIGS.keys():
        row = f"{model_name:<35}"
        for noise in display_levels:
            if noise in final_results[model_name]:
                oa = final_results[model_name][noise]['mean']['oa']
                row += f" | {oa:5.1f}"
            else:
                row += " |    --"
        print(row)
    
    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/noise_results.tex")
    print(f"  - {output_dir}/noise_vs_accuracy.png/pdf")
    print(f"  - {output_dir}/accuracy_vs_snr.png/pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
